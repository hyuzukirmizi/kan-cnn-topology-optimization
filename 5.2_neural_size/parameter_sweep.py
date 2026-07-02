import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Setup Paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from neural_structural_optimization import problems, topo_api
    import models as pt
except ImportError as exc:
    print(f"Error importing modules: {exc}")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
DEFAULT_MAX_ITERATIONS = 400
PROBLEM_SPECS = {
    "mbb_beam_384x128_0.3": problems.PROBLEMS_BY_NAME["mbb_beam_384x128_0.3"],
    "cantilever_beam_two_point_256x192_0.15": problems.PROBLEMS_BY_NAME["cantilever_beam_two_point_256x192_0.15"],
    "roof_256x256_0.4": problems.PROBLEMS_BY_NAME["roof_256x256_0.4"],
    "free_suspended_bridge_256x256_0.075": problems.PROBLEMS_BY_NAME["free_suspended_bridge_256x256_0.075"],
}

PARAMS_TO_SWEEP = {
    "hidden_layers": [(64, 64), (32, 32), (16, 16), (8, 8)],
    "grid": [10, 8, 6, 4],
    "k": [4, 3, 2, 1],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep BaseKANModel parameters for the 5.1 validation problems")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_ITERATIONS, help="Optimization steps per run")
    parser.add_argument("--problems", nargs="+", default=list(PROBLEM_SPECS.keys()), help="Subset of problems to run")
    parser.add_argument("--sweeps", nargs="+", default=["hidden_layers", "grid", "k"], choices=["hidden_layers", "grid", "k"], help="Sweeps to execute")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "5.2_neural_size" / "results"), help="Directory for plots, CSV and JSON outputs")
    return parser.parse_args()


def get_baseline_params(problem_name: str) -> dict:
    total_elements = PROBLEM_SPECS[problem_name].width * PROBLEM_SPECS[problem_name].height
    if total_elements > 40000:
        return {"hidden_layers": (64, 64), "grid": 10, "k": 3}
    return {"hidden_layers": (32, 32), "grid": 10, "k": 3}


def run_single_test(problem_name: str, problem, model_params: dict, step_count: int, sweep_name: str, parameter_name: str, parameter_value: object) -> dict:
    print(f"Running {problem_name} [{sweep_name}={parameter_value}]")
    start_time = time.perf_counter()

    kan_model_params = model_params.copy()
    kan_model_params["kan_layers"] = kan_model_params.pop("hidden_layers")

    try:
        topo_args = topo_api.specified_task(problem)
        model = pt.BaseKANModel(args=topo_args, **kan_model_params)
        ds = pt.train_lbfgs(model, step_count)

        losses = np.asarray(ds.loss.values).flatten()
        valid_losses = losses[~np.isnan(losses)]
        best_compliance = float(np.min(valid_losses)) if len(valid_losses) > 0 else float("nan")
        final_design = ds.design.ffill("step").isel(step=-1)
        design_values = np.asarray(final_design.values)
        mask = np.broadcast_to(topo_args["mask"], design_values.shape) > 0
        if design_values.size == 0 or not np.any(mask):
            volume_fraction = float("nan")
            volume_violation = float("nan")
        else:
            active_values = design_values[mask]
            volume_fraction = float(np.mean(active_values) / np.mean(mask.astype(float)))
            volume_violation = float(volume_fraction - topo_args["volfrac"])
        elapsed_time = time.perf_counter() - start_time

        return {
            "problem": problem_name,
            "sweep": sweep_name,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "best_compliance": best_compliance,
            "volume_fraction": volume_fraction,
            "volume_violation": volume_violation,
            "time_sec": elapsed_time,
            "config": {
                "kan_layers": list(kan_model_params["kan_layers"]),
                "grid": int(kan_model_params["grid"]),
                "k": int(kan_model_params["k"]),
                "max_steps": int(step_count),
            },
            "final_design": final_design,
            "error": None,
        }
    except Exception as exc:
        print(f"ERROR for {problem_name} [{sweep_name}={parameter_value}]: {exc}")
        return {
            "problem": problem_name,
            "sweep": sweep_name,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "best_compliance": float("nan"),
            "time_sec": time.perf_counter() - start_time,
            "config": {
                "kan_layers": list(kan_model_params.get("kan_layers", [])),
                "grid": int(kan_model_params.get("grid", 0)),
                "k": int(kan_model_params.get("k", 0)),
                "max_steps": int(step_count),
            },
            "final_design": None,
            "error": str(exc),
        }


def save_topology_image(problem_name: str, sweep_name: str, parameter_value: object, design, out_dir: Path) -> None:
    if design is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(design.values.T, cmap="Greys", origin="upper")
    ax.set_title(f"{problem_name}\n{sweep_name}={parameter_value}")
    ax.axis("off")

    param_str = str(parameter_value).replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")
    filename = f"{problem_name}_{sweep_name}_{param_str}_topology.png"
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_parameter_impact(problem_name: str, results_df: pd.DataFrame, sweep_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    x_vals = results_df["parameter_value"].astype(str)
    y_vals = results_df["best_compliance"].astype(float)
    ax.plot(x_vals, y_vals, marker="o", linestyle="-", color="royalblue")

    ax.set_xlabel(sweep_name.replace("_", " ").title())
    ax.set_ylabel("Best compliance score")
    ax.set_title(f"{problem_name}: {sweep_name} vs compliance")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for i, val in enumerate(y_vals):
        if not np.isnan(val):
            ax.text(i, val, f" {val:.2f}", va="bottom", ha="left")

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    path = out_dir / f"{problem_name}_{sweep_name}_impact.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_json_results(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for problem_name, subset in df.groupby("problem", sort=True):
        problem_records = []
        for _, row in subset.iterrows():
            record = {
                "problem": problem_name,
                "sweep": row.get("sweep"),
                "parameter_name": row.get("parameter_name"),
                "parameter_value": row.get("parameter_value"),
                "best_compliance": None if pd.isna(row["best_compliance"]) else float(row["best_compliance"]),
                "volume_fraction": None if pd.isna(row.get("volume_fraction", np.nan)) else float(row["volume_fraction"]),
                "volume_violation": None if pd.isna(row.get("volume_violation", np.nan)) else float(row["volume_violation"]),
                "time_sec": float(row.get("time_sec", 0.0)),
                "config": row.get("config", {}),
                "error": row.get("error"),
            }
            problem_records.append(record)
            records.append(record)

        with open(out_dir / f"{problem_name}_parameter_sweep_results.json", "w", encoding="utf-8") as handle:
            json.dump({"problem": problem_name, "runs": problem_records}, handle, indent=2)

    with open(out_dir / "parameter_sweep_results.json", "w", encoding="utf-8") as handle:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": records}, handle, indent=2)


def run_sweeps(problem_name: str, problem, max_steps: int, sweeps: List[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    baseline_params = get_baseline_params(problem_name)

    for sweep_name in sweeps:
        if sweep_name == "hidden_layers":
            values = PARAMS_TO_SWEEP[sweep_name]
            for param_value in values:
                current_params = baseline_params.copy()
                current_params["hidden_layers"] = param_value
                result = run_single_test(problem_name, problem, current_params, max_steps, sweep_name, "hidden_layers", list(param_value))
                rows.append(result)
                save_topology_image(problem_name, sweep_name, param_value, result.get("final_design"), out_dir)
        elif sweep_name == "grid":
            values = PARAMS_TO_SWEEP[sweep_name]
            for param_value in values:
                current_params = baseline_params.copy()
                current_params["grid"] = param_value
                result = run_single_test(problem_name, problem, current_params, max_steps, sweep_name, "grid", param_value)
                rows.append(result)
                save_topology_image(problem_name, sweep_name, param_value, result.get("final_design"), out_dir)
        elif sweep_name == "k":
            values = PARAMS_TO_SWEEP[sweep_name]
            for param_value in values:
                current_params = baseline_params.copy()
                current_params["k"] = param_value
                result = run_single_test(problem_name, problem, current_params, max_steps, sweep_name, "k", param_value)
                rows.append(result)
                save_topology_image(problem_name, sweep_name, param_value, result.get("final_design"), out_dir)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for problem_name in args.problems:
        if problem_name not in PROBLEM_SPECS:
            raise KeyError(f"Unknown problem: {problem_name}")
        problem = PROBLEM_SPECS[problem_name]
        df = run_sweeps(problem_name, problem, args.max_steps, args.sweeps, out_dir)
        all_rows.append(df)
        print(f"Completed {problem_name}")

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    csv_path = out_dir / "base_kan_parameter_sweep_results.csv"
    combined.to_csv(csv_path, index=False)

    for sweep_name in args.sweeps:
        subset = combined[combined["sweep"] == sweep_name]
        if subset.empty:
            continue
        for problem_name in sorted(subset["problem"].unique()):
            problem_subset = subset[subset["problem"] == problem_name]
            plot_parameter_impact(problem_name, problem_subset, sweep_name, out_dir)

    save_json_results(combined, out_dir)

    report_path = out_dir / "parameter_sweep_report.md"
    report_content = "# Base KAN Parameter Sweep Analysis\n\n"
    for sweep_name in args.sweeps:
        report_content += f"## {sweep_name}\n\n"
        for problem_name in sorted(combined["problem"].dropna().unique()):
            problem_subset = combined[(combined["problem"] == problem_name) & (combined["sweep"] == sweep_name)]
            if problem_subset.empty:
                continue
            plot_name = f"{problem_name}_{sweep_name}_impact.png"
            report_content += f"### {problem_name}\n\n"
            report_content += f"![{problem_name} {sweep_name} plot]({plot_name})\n\n"
            report_content += problem_subset[["parameter_value", "best_compliance", "time_sec", "config"]].to_string(index=False)
            report_content += "\n\n"

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_content)

    print(f"Saved CSV results to {csv_path}")
    print(f"Saved JSON results to {out_dir / 'parameter_sweep_results.json'}")
    print(f"Saved plots and report to {out_dir}")


if __name__ == "__main__":
    main()
