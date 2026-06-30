#!/usr/bin/env python3
"""Sweep BaseKANModel hyperparameters over the four 5.1 validation problems.

This script isolates the coordinate-based BaseKANModel and evaluates how three
key hyperparameters affect optimization quality:

1. hidden layer widths (`kan_layers`)
2. spline grid size (`grid`)
3. spline order (`k`)

It mirrors the 5.1 setup by running the same four benchmark problems and
records the best compliance (loss) achieved for each parameter setting.

Outputs:
- CSV file with all runs
- one plot per sweep with one subplot per problem
- a concise summary printed to stdout
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_structural_optimization import problems, topo_api
import models as pt

warnings.filterwarnings("ignore", category=UserWarning)

PROBLEM_NAMES = [
    "mbb_beam_384x128_0.3",
    "cantilever_beam_two_point_256x192_0.15",
    "roof_256x256_0.4",
    "free_suspended_bridge_256x256_0.075",
]

# 5.1 defaults for BaseKANModel
DEFAULT_HIDDEN_LAYERS = [(64, 64), (32, 32), (16, 16), (8, 8)]
DEFAULT_GRIDS = [4, 6, 8, 10, 12]
DEFAULT_KS = [1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep BaseKANModel parameters for the 5.1 validation problems")
    parser.add_argument("--max-steps", type=int, default=50, help="Optimization steps per run (default: 50)")
    parser.add_argument("--problems", nargs="+", default=PROBLEM_NAMES, help="Subset of problem names to run")
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=["hidden_layers", "grid", "k"],
        choices=["hidden_layers", "grid", "k"],
        help="Which sweeps to run",
    )
    parser.add_argument("--hidden-layers", nargs="+", default=["64,64", "32,32", "16,16", "8,8"], help="Comma-separated hidden layer sizes")
    parser.add_argument("--grids", nargs="+", type=int, default=DEFAULT_GRIDS, help="Grid sizes to sweep")
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS, help="Spline orders to sweep")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to store CSV plots (default: 5.2_neural_size/results)")
    return parser.parse_args()


def parse_hidden_layers(specs: Iterable[str]) -> List[Tuple[int, ...]]:
    parsed: List[Tuple[int, ...]] = []
    for item in specs:
        if "," in item:
            parsed.append(tuple(int(x.strip()) for x in item.split(",") if x.strip()))
        else:
            parsed.append((int(item),))
    return parsed


def run_single(problem_name: str, problem, max_steps: int, kan_layers: Tuple[int, ...], grid: int, k: int) -> dict:
    topo_args = topo_api.specified_task(problem)
    model = pt.BaseKANModel(args=topo_args, kan_layers=kan_layers, grid=grid, k=k)
    ds = pt.train_lbfgs(model, max_steps)

    losses = np.asarray(ds.loss.values, dtype=float)
    valid_losses = losses[~np.isnan(losses)]
    best_loss = float(np.nanmin(valid_losses)) if valid_losses.size else float("nan")
    best_step = int(np.nanargmin(losses)) if losses.size else -1

    return {
        "problem": problem_name,
        "kan_layers": kan_layers,
        "grid": grid,
        "k": k,
        "best_loss": best_loss,
        "best_step": best_step,
    }


def run_sweeps(problem_name: str, problem, max_steps: int, hidden_layers: List[Tuple[int, ...]], grids: List[int], ks: List[int], sweeps: List[str]) -> pd.DataFrame:
    rows = []

    if "hidden_layers" in sweeps:
        for layers in hidden_layers:
            rows.append(run_single(problem_name, problem, max_steps, layers, grid=10, k=3))

    if "grid" in sweeps:
        for grid in grids:
            rows.append(run_single(problem_name, problem, max_steps, kan_layers=(32, 32), grid=grid, k=3))

    if "k" in sweeps:
        for k in ks:
            rows.append(run_single(problem_name, problem, max_steps, kan_layers=(32, 32), grid=10, k=k))

    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, out_dir: Path, sweeps: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for sweep_name in sweeps:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
        axes = axes.ravel()

        for ax, problem_name in zip(axes, sorted(df["problem"].unique())):
            subset = df[df["problem"] == problem_name]
            if sweep_name == "hidden_layers":
                x_vals = [str(tuple(l)) for l in subset["kan_layers"]]
                y_vals = subset["best_loss"].astype(float)
                ax.plot(range(len(x_vals)), y_vals, marker="o", linewidth=1.8)
                ax.set_xticks(range(len(x_vals)))
                ax.set_xticklabels(x_vals, rotation=30, ha="right")
                ax.set_title(problem_name)
                ax.set_ylabel("Best compliance")
                ax.set_xlabel("kan_layers")
            elif sweep_name == "grid":
                x_vals = subset["grid"].astype(int)
                y_vals = subset["best_loss"].astype(float)
                ax.plot(x_vals, y_vals, marker="o", linewidth=1.8)
                ax.set_title(problem_name)
                ax.set_ylabel("Best compliance")
                ax.set_xlabel("grid")
            else:
                x_vals = subset["k"].astype(int)
                y_vals = subset["best_loss"].astype(float)
                ax.plot(x_vals, y_vals, marker="o", linewidth=1.8)
                ax.set_title(problem_name)
                ax.set_ylabel("Best compliance")
                ax.set_xlabel("k")

        fig.suptitle(f"BaseKANModel {sweep_name} sweep")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_dir / f"base_kan_{sweep_name}_sweep.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "5.2_neural_size" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    problem_names = args.problems

    print("BaseKAN parameter sweep")
    print("=======================")
    print("Problems:", ", ".join(problem_names))
    print("Max steps:", args.max_steps)
    print("5.1 baseline settings:")
    print("  - hidden layers: (64,64) for larger problems, (32,32) for smaller ones")
    print("  - grid: 10")
    print("  - k: 3")
    print()

    all_rows = []
    for problem_name in problem_names:
        if problem_name not in problems.PROBLEMS_BY_NAME:
            raise KeyError(f"Unknown problem: {problem_name}")
        problem = problems.PROBLEMS_BY_NAME[problem_name]
        df = run_sweeps(problem_name, problem, args.max_steps, hidden_layers, args.grids, args.ks, args.sweeps)
        all_rows.append(df)
        print(f"Completed {problem_name}")

    combined = pd.concat(all_rows, ignore_index=True)
    csv_path = out_dir / "base_kan_parameter_sweep_results.csv"
    combined.to_csv(csv_path, index=False)
    make_plots(combined, out_dir, args.sweeps)

    print(f"Saved results to {csv_path}")
    print(f"Saved plots to {out_dir}")
    print()
    print(combined.head().to_string(index=False))


if __name__ == "__main__":
    main()
