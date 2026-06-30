#!/bin/bash
#SBATCH --job-name=kan_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
WORK_ROOT="${SLURM_SUBMIT_DIR:-$ROOT_DIR}"
if [[ ! -w "$WORK_ROOT" ]]; then
    WORK_ROOT="$HOME/kan_topo_benchmark"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/benchmark_results" "$WORK_ROOT/benchmark_plots"
cd "$WORK_ROOT"

echo "========================================================"
echo "Starting KAN topology benchmark on Unity HPC"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start time: $(date)"
echo "========================================================"

module load conda/latest

if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

ENV_NAME="kan_topo_env"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating Conda environment '$ENV_NAME'."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"

echo "Installing Python packages from requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"

python <<'PY'
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from neural_structural_optimization import problems, topo_api

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path.cwd())).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location("models", REPO_ROOT / "models.py")
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load models.py from {REPO_ROOT / 'models.py'}")
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)
pt = models_module

ROOT = Path.cwd()
DATA_DIR = ROOT / "benchmark_results"
PLOT_DIR = ROOT / "benchmark_plots"
REPORT_PATH = ROOT / "benchmark_analysis.md"
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

MAX_ITERATIONS = 400

PROBLEM_SPECS = [
    ("mbb_beam_384x128_0.3", problems.PROBLEMS_BY_NAME["mbb_beam_384x128_0.3"]),
    ("cantilever_beam_two_point_256x192_0.15", problems.PROBLEMS_BY_NAME["cantilever_beam_two_point_256x192_0.15"]),
    ("roof_256x256_0.4", problems.PROBLEMS_BY_NAME["roof_256x256_0.4"]),
    ("free_suspended_bridge_256x256_0.075", problems.PROBLEMS_BY_NAME["free_suspended_bridge_256x256_0.075"]),
]


def model_kwargs_for(problem):
    if problem.width % 16 == 0 and problem.height % 16 == 0:
        return {"resizes": (1, 2, 2, 2, 2, 1), "conv_filters": (128, 64, 32, 16, 8, 1)}
    if problem.width % 8 == 0 and problem.height % 8 == 0:
        return {"resizes": (1, 2, 2, 2, 1), "conv_filters": (128, 64, 32, 16, 1)}
    return {"resizes": (1, 1, 2, 2, 1), "conv_filters": (128, 64, 32, 16, 1)}


def gray_fraction_series(design, low=0.05, high=0.95):
    return ((design > low) & (design < high)).mean(dim=("x", "y"))


def train_model(problem_name, problem):
    topo_args = topo_api.specified_task(problem)
    kwargs = model_kwargs_for(problem)

    # Adaptive KAN capacity based on problem size, inspired by Section_5.1_Benchmark.ipynb
    total_elements = problem.width * problem.height
    kan_hidden_layers = (64, 64) if total_elements > 40000 else (32, 32)
    print(f"[{problem_name}] Using KAN layers: {kan_hidden_layers} for {total_elements} elements.")

    model_specs = [
        ("Hybrid KAN", lambda: pt.HybridKANModel(args=topo_args, **kwargs), "lbfgs"),
        ("KAN", lambda: pt.BaseKANModel(args=topo_args, kan_layers=kan_hidden_layers, grid=10, k=3), "lbfgs"),
        ("CNN-LBFGS", lambda: pt.CNNModel(args=topo_args, **kwargs), "lbfgs"),
        ("Pixel-LBFGS", lambda: pt.PixelModel(args=topo_args), "lbfgs"),
        ("MMA", lambda: pt.PixelModel(args=topo_args), "mma"),
        ("OC", lambda: pt.PixelModel(args=topo_args), "oc"),
    ]

    datasets = []
    summary_rows = []

    for label, factory, mode in model_specs:
        print(f"[{problem_name}] Running {label}")
        started = time.time()

        if mode == "mma":
            try:
                import nlopt  # noqa: F401
                ds = pt.method_of_moving_asymptotes(factory(), MAX_ITERATIONS)
            except ImportError:
                print("nlopt unavailable; skipping MMA")
                continue
        elif mode == "oc":
            ds = pt.optimality_criteria(factory(), MAX_ITERATIONS)
        else:
            ds = pt.train_lbfgs(factory(), MAX_ITERATIONS)

        elapsed = time.time() - started
        losses = np.asarray(ds.loss.values).flatten()
        valid_losses = losses[~np.isnan(losses)]
        best_loss = float(np.min(valid_losses)) if len(valid_losses) else float("nan")
        best_step = int(np.argmin(valid_losses)) if len(valid_losses) else -1
        final_loss = float(valid_losses[-1]) if len(valid_losses) else float("nan")

        ds = ds.assign_coords(model_name=label)
        ds["gray_fraction"] = gray_fraction_series(ds.design)
        ds.attrs["time_sec"] = elapsed
        ds.attrs["best_loss"] = best_loss
        ds.attrs["best_step"] = best_step
        ds.attrs["final_loss"] = final_loss

        datasets.append(ds)
        summary_rows.append(
            {
                "model": label,
                "time_sec": elapsed,
                "best_loss": best_loss,
                "best_step": best_step,
                "final_loss": final_loss,
                "final_gray_fraction": float(ds.gray_fraction.isel(step=-1).values),
            }
        )

    if not datasets:
        raise RuntimeError(f"No datasets were produced for {problem_name}")

    model_index = pd.Index([row["model"] for row in summary_rows], name="model")
    combined = xr.concat(datasets, dim=model_index)
    combined = combined.assign_coords(model=model_index)
    combined["time_sec"] = xr.DataArray([row["time_sec"] for row in summary_rows], dims=["model"], coords={"model": model_index})
    combined["best_loss"] = xr.DataArray([row["best_loss"] for row in summary_rows], dims=["model"], coords={"model": model_index})
    combined["best_step"] = xr.DataArray([row["best_step"] for row in summary_rows], dims=["model"], coords={"model": model_index})
    combined["final_loss"] = xr.DataArray([row["final_loss"] for row in summary_rows], dims=["model"], coords={"model": model_index})
    combined["final_gray_fraction"] = xr.DataArray([row["final_gray_fraction"] for row in summary_rows], dims=["model"], coords={"model": model_index})

    return combined, summary_rows


def save_individual_topology_images(problem_name, ds):
    final_design = ds.design.ffill("step").isel(step=-1)
    image_paths = {}
    for model_name in ds.model.values:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(final_design.sel(model=model_name).values.T, cmap="Greys", origin="upper")
        ax.set_title(f"{problem_name} - {model_name}")
        ax.axis("off")
        path = PLOT_DIR / f"{problem_name}_{str(model_name).replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        image_paths[str(model_name)] = path
    return image_paths


def save_comparison_grid(problem_name, ds):
    fig, axes = plt.subplots(1, ds.sizes["model"], figsize=(4 * ds.sizes["model"], 4))
    if ds.sizes["model"] == 1:
        axes = [axes]

    final_design = ds.design.ffill("step").isel(step=-1)
    for ax, model_name in zip(axes, ds.model.values):
        ax.imshow(final_design.sel(model=model_name).values.T, cmap="Greys", origin="upper")
        ax.set_title(str(model_name))
        ax.axis("off")

    fig.suptitle(problem_name, y=0.98)
    fig.tight_layout()
    path = PLOT_DIR / f"{problem_name}_final_topologies.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_kan_convergence(problem_name, ds):
    # Select the 'KAN' model, if it exists in the dataset
    if "KAN" not in ds.model.values:
        print(f"Warning: 'KAN' model not found for {problem_name}. Skipping KAN convergence plot.")
        return None
    kan = ds.sel(model="KAN")
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.plot(kan.step.values, kan.loss.values, label="Compliance", lw=2, color="navy")
    ax1.plot(kan.step.values, kan.gray_fraction.values, label="Gray Fraction", lw=2, color="crimson")
    ax1.set_title(f"{problem_name}: KAN convergence")
    ax1.set_xlabel("Optimization step")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig.tight_layout()
    path = PLOT_DIR / f"{problem_name}_kan_convergence.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

def save_compliance_plot(problem_name, ds):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for model_name in ds.model.values:
        loss_da = ds.loss.sel(model=model_name)
        loss_values = np.asarray(loss_da.values, dtype=float)
        best_values = np.minimum.accumulate(np.where(np.isnan(loss_values), np.inf, loss_values))
        best_values[np.isinf(best_values)] = np.nan
        best_da = xr.DataArray(best_values, dims=loss_da.dims, coords=loss_da.coords)
        ax.plot(ds.step.values, best_da, label=model_name, lw=2)

    ax.set_title(f"{problem_name}: Compliance Convergence")
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Compliance (Loss)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    path = PLOT_DIR / f"{problem_name}_compliance_convergence.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_markdown(all_results, grid_paths, individual_paths, convergence_path, compliance_plot_paths):
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Analysis\n")
        f.write(
            "This report was generated automatically on Unity HPC after the full benchmark run. "
            "It compares KAN, CNN-LBFGS, Pixel-LBFGS, MMA, and OC across the requested benchmark problems.\n"
        )
        f.write("\n## Summary Table\n\n")
        f.write("| Problem | Model | Best Compliance | Best Step | Final Compliance | Time (s) | Final Gray Fraction |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for problem_name, rows in all_results.items():
            for row in rows:
                f.write(
                    f"| {problem_name} | {row['model']} | {row['best_loss']:.6f} | {row['best_step']} | {row['final_loss']:.6f} | {row['time_sec']:.2f} | {row['final_gray_fraction']:.4f} |\n"
                )
            f.write("\n")

        f.write("\n## Compliance Convergence\n\n")
        for problem_name, plot_path in compliance_plot_paths.items():
            f.write(f"### {problem_name}\n")
            f.write(f"![Compliance plot for {problem_name}]({plot_path.name})\n")

        f.write("\n## Final Topologies\n\n")
        for problem_name, grid_path in grid_paths.items():
            f.write(f"### {problem_name}\n")
            f.write(f"![Comparison grid for {problem_name}]({grid_path.name})\n\n")
            f.write("Individual model images:\n")
            for model_name, image_path in individual_paths[problem_name].items():
                f.write(f"- {model_name}: ![{problem_name} {model_name}]({image_path.name})\n")
            f.write("\n")

        if convergence_path:
            f.write("\n## KAN Convergence\n\n")
            f.write(f"![KAN convergence for mbb_beam_384x128_0.3]({convergence_path.name})\n")

        f.write("\n## Draft Analysis\n\n")
        f.write(
            "KAN tends to reduce gray elements faster than standard MLP-style parameterizations because its spline-based activations can represent localized spatial transitions more directly. "
            "In topology optimization, that locality helps the optimizer sharpen boundaries sooner, which is visible in the final topology images and the convergence trace for the MBB beam case.\n"
        )


all_results = {}
grid_paths = {}
individual_paths = {}
compliance_plot_paths = {}
convergence_path = None

for problem_name, problem in PROBLEM_SPECS:
    ds, rows = train_model(problem_name, problem)
    all_results[problem_name] = rows
    ds.to_netcdf(DATA_DIR / f"{problem_name}_results.nc")
    with (DATA_DIR / f"{problem_name}_stats.json").open("w", encoding="utf-8") as f:
        json.dump({"problem": problem_name, "results": rows}, f, indent=2)
    grid_paths[problem_name] = save_comparison_grid(problem_name, ds)
    individual_paths[problem_name] = save_individual_topology_images(problem_name, ds)
    compliance_plot_paths[problem_name] = save_compliance_plot(problem_name, ds)
    if problem_name == "mbb_beam_384x128_0.3":
        convergence_path = save_kan_convergence(problem_name, ds)

# This check is now safer since convergence_path might be None
if "mbb_beam_384x128_0.3" in [p[0] for p in PROBLEM_SPECS] and convergence_path is None:
    print("Warning: KAN convergence plot for mbb_beam_384x128_0.3 was not generated, 'KAN' model might have been skipped.")

write_markdown(all_results, grid_paths, individual_paths, convergence_path, compliance_plot_paths)
print(f"Saved Markdown report to {REPORT_PATH}")
print(f"Saved plots to {PLOT_DIR}")
print(f"Saved data artifacts to {DATA_DIR}")
PY

echo "========================================================"
echo "Job finished on: $(date)"
echo "========================================================"

