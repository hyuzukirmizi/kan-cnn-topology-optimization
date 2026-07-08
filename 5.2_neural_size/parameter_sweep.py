"""Checkpointed 5.2 BaseKAN parameter sweep.

The sweep config lives in one place below. Add a new test type by extending
`TEST_TYPE_CONFIGS`, and add new problems by extending `PROBLEM_SPECS`.

Persistence is incremental: every finished parameter configuration writes its
own topology image, per-run JSON, progress record, and aggregated test-type
summary before the next run starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from neural_structural_optimization import problems, topo_api
    import models as pt
except ImportError as exc:
    print(f"Error importing modules: {exc}")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MAX_STEPS = 400

PROBLEM_SPECS = {
    "mbb_beam_384x128_0.3": problems.PROBLEMS_BY_NAME["mbb_beam_384x128_0.3"],
    "cantilever_beam_two_point_256x192_0.15": problems.PROBLEMS_BY_NAME["cantilever_beam_two_point_256x192_0.15"],
    "roof_256x256_0.4": problems.PROBLEMS_BY_NAME["roof_256x256_0.4"],
    "free_suspended_bridge_256x256_0.075": problems.PROBLEMS_BY_NAME["free_suspended_bridge_256x256_0.075"],
}

TEST_TYPE_CONFIGS = {
    "hidden_layers": {
        "parameter_name": "hidden_layers",
        "label": "Hidden layers",
        "values": [(64, 64), (32, 32), (16, 16), (8, 8)],
        "baseline": {"grid": 10, "k": 3},
    },
    "grid": {
        "parameter_name": "grid",
        "label": "Grid",
        "values": [10, 8, 6, 4],
        "baseline": {"hidden_layers": (32, 32), "k": 3},
    },
    "k": {
        "parameter_name": "k",
        "label": "Spline order",
        "values": [3, 2, 1],
        "baseline": {"hidden_layers": (32, 32), "grid": 10},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed BaseKAN parameter sweep")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Optimization steps per run")
    parser.add_argument("--problems", nargs="+", default=list(PROBLEM_SPECS), help="Problem names to run")
    parser.add_argument(
        "--test-types",
        nargs="+",
        default=list(TEST_TYPE_CONFIGS),
        choices=list(TEST_TYPE_CONFIGS),
        help="Test types to execute; extend TEST_TYPE_CONFIGS to add a new one",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "5.2_neural_size" / "results"),
        help="Directory for plots, JSON checkpoints, and summaries",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from completed run files")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Re-run all parameter configurations")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SWEEP_WORKERS", "2")), help="Parallel workers per test type")
    parser.add_argument("--no-global-summary", action="store_true", help="Skip writing the combined CSV/JSON during run execution")
    parser.add_argument("--aggregate-only", action="store_true", help="Only rebuild the combined CSV/JSON from existing run files")
    return parser.parse_args()


def _serialise_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _slugify_value(value: Any) -> str:
    serialised = _serialise_value(value)
    if isinstance(serialised, list):
        return "x".join(str(item) for item in serialised)
    return str(serialised).replace(".", "p")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(path)


def _append_progress_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def _get_baseline_params(problem_name: str) -> dict[str, Any]:
    total_elements = PROBLEM_SPECS[problem_name].width * PROBLEM_SPECS[problem_name].height
    if total_elements > 40000:
        return {"hidden_layers": (64, 64), "grid": 10, "k": 3}
    return {"hidden_layers": (32, 32), "grid": 10, "k": 3}


def _build_model_params(baseline_params: dict[str, Any], test_type: str, parameter_value: Any) -> dict[str, Any]:
    params = baseline_params.copy()
    params[TEST_TYPE_CONFIGS[test_type]["parameter_name"]] = parameter_value
    params["kan_layers"] = params.pop("hidden_layers")
    return params


def _run_single_test(problem_name: str, problem: Any, model_params: dict[str, Any], step_count: int, test_type: str, parameter_value: Any) -> tuple[dict[str, Any], np.ndarray | None]:
    start_time = time.perf_counter()
    parameter_name = TEST_TYPE_CONFIGS[test_type]["parameter_name"]
    serialised_value = _serialise_value(parameter_value)
    print(f"Running {problem_name} [{test_type}={serialised_value}]")

    try:
        topo_args = topo_api.specified_task(problem)
        model = pt.BaseKANModel(args=topo_args, **model_params)
        dataset = pt.train_lbfgs(model, step_count)

        losses = np.asarray(dataset.loss.values, dtype=float).reshape(-1)
        valid_losses = losses[~np.isnan(losses)]
        best_compliance = float(np.min(valid_losses)) if valid_losses.size else float("nan")
        final_compliance = float(valid_losses[-1]) if valid_losses.size else float("nan")
        best_step = int(np.nanargmin(losses)) if valid_losses.size else -1

        final_design = dataset.design.ffill("step").isel(step=-1)
        design_values = np.asarray(final_design.values, dtype=float)
        mask = np.broadcast_to(np.asarray(topo_args["mask"]), design_values.shape) > 0

        if design_values.size == 0 or not np.any(mask):
            volume_fraction = float("nan")
            volume_violation = float("nan")
            design_stats = {}
        else:
            active_values = design_values[mask]
            volume_fraction = float(np.mean(active_values))
            volume_violation = float(volume_fraction - topo_args["volfrac"])
            design_stats = {
                "shape": list(design_values.shape),
                "min": float(np.min(design_values)),
                "max": float(np.max(design_values)),
                "mean": float(np.mean(design_values)),
                "std": float(np.std(design_values)),
                "active_mean": float(np.mean(active_values)),
                "active_count": int(active_values.size),
            }

        elapsed_time = time.perf_counter() - start_time
        return (
            {
                "problem": problem_name,
                "test_type": test_type,
                "parameter_name": parameter_name,
                "parameter_value": serialised_value,
                "best_compliance": best_compliance,
                "final_compliance": final_compliance,
                "best_step": best_step,
                "volume_fraction": volume_fraction,
                "volume_violation": volume_violation,
                "time_sec": elapsed_time,
                "config": {
                    "kan_layers": list(model_params["kan_layers"]),
                    "grid": int(model_params["grid"]),
                    "k": int(model_params["k"]),
                    "max_steps": int(step_count),
                },
                "design_stats": design_stats,
                "error": None,
            },
            design_values,
        )
    except Exception as exc:
        print(f"ERROR for {problem_name} [{test_type}={serialised_value}]: {exc}")
        return (
            {
                "problem": problem_name,
                "test_type": test_type,
                "parameter_name": parameter_name,
                "parameter_value": serialised_value,
                "best_compliance": float("nan"),
                "final_compliance": float("nan"),
                "best_step": -1,
                "volume_fraction": float("nan"),
                "volume_violation": float("nan"),
                "time_sec": time.perf_counter() - start_time,
                "config": {
                    "kan_layers": list(model_params.get("kan_layers", [])),
                    "grid": int(model_params.get("grid", 0)),
                    "k": int(model_params.get("k", 0)),
                    "max_steps": int(step_count),
                },
                "design_stats": {},
                "error": str(exc),
            },
            None,
        )


def _run_single_test_by_name(problem_name: str, model_params: dict[str, Any], step_count: int, test_type: str, parameter_value: Any) -> tuple[dict[str, Any], np.ndarray | None]:
    return _run_single_test(problem_name, PROBLEM_SPECS[problem_name], model_params, step_count, test_type, parameter_value)


def _write_topology_artifacts(problem_name: str, test_type: str, parameter_value: Any, design_values: np.ndarray | None, artifact_dir: Path) -> dict[str, str] | None:
    if design_values is None:
        return None

    artifact_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify_value(parameter_value)

    array_path = artifact_dir / f"{problem_name}_{test_type}_{slug}_design.npy"
    np.save(array_path, design_values)

    image_path = artifact_dir / f"{problem_name}_{test_type}_{slug}_topology.png"
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(design_values.T, cmap="Greys", origin="lower")
    ax.set_title(f"{problem_name}\n{test_type}={_serialise_value(parameter_value)}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    return {"design_npy": str(array_path), "topology_png": str(image_path)}


def _load_completed_run_keys(run_dir: Path) -> set[str]:
    return {path.stem for path in run_dir.glob("*.json")}


def _save_test_type_summary(problem_name: str, test_type: str, records: list[dict[str, Any]], summary_dir: Path, parameter_order: list[Any]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    parameter_name = TEST_TYPE_CONFIGS[test_type]["parameter_name"]
    label = TEST_TYPE_CONFIGS[test_type]["label"]

    serialised_order = [_serialise_value(item) for item in parameter_order]
    ordered_records = sorted(records, key=lambda record: serialised_order.index(record["parameter_value"]))
    frame = pd.DataFrame(ordered_records)

    if not frame.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_labels = [str(_serialise_value(item)) for item in frame["parameter_value"].tolist()]
        y_vals = frame["best_compliance"].astype(float).to_numpy()
        ax.plot(range(len(x_labels)), y_vals, marker="o", linestyle="-", color="royalblue")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel(label)
        ax.set_ylabel("Compliance score")
        ax.set_title(f"{problem_name}: {parameter_name} vs compliance")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        fig.savefig(summary_dir / f"{problem_name}_{test_type}_compliance_vs_parameter.png", dpi=180)
        plt.close(fig)

    summary_payload = {
        "problem": problem_name,
        "test_type": test_type,
        "parameter_name": parameter_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_runs": ordered_records,
    }
    _atomic_write_json(summary_dir / f"{problem_name}_{test_type}_results.json", summary_payload)


def _write_progress_state(problem_name: str, test_type: str, records: list[dict[str, Any]], total_count: int, state_dir: Path) -> None:
    completed = len(records)
    state_payload = {
        "problem": problem_name,
        "test_type": test_type,
        "total_configurations": total_count,
        "completed_configurations": completed,
        "pending_configurations": max(total_count - completed, 0),
        "completed_parameters": [record["parameter_value"] for record in records],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_write_json(state_dir / f"{problem_name}_{test_type}_progress.json", state_payload)


def _persist_global_summary(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run_file in out_dir.glob("**/runs/*.json"):
        with open(run_file, "r", encoding="utf-8") as handle:
            rows.append(json.load(handle))

    rows.sort(key=lambda item: (item["problem"], item["test_type"], str(item["parameter_value"])))

    _atomic_write_json(
        out_dir / "parameter_sweep_results.json",
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": rows},
    )

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "base_kan_parameter_sweep_results.csv", index=False)


def _merge_existing_results(out_dir: Path) -> None:
    _persist_global_summary(out_dir)


def _run_test_type(problem_name: str, problem: Any, test_type: str, max_steps: int, out_dir: Path, resume: bool, workers: int) -> list[dict[str, Any]]:
    config = TEST_TYPE_CONFIGS[test_type]
    parameter_values = config["values"]
    parameter_name = config["parameter_name"]

    problem_root = out_dir / problem_name / test_type
    run_dir = problem_root / "runs"
    topology_dir = problem_root / "topologies"
    summary_dir = problem_root
    state_dir = problem_root / "state"
    progress_log = problem_root / "progress.log"

    completed_keys = _load_completed_run_keys(run_dir) if resume else set()
    baseline_params = _get_baseline_params(problem_name)
    records: list[dict[str, Any]] = []

    if resume:
        for run_file in sorted(run_dir.glob("*.json")):
            with open(run_file, "r", encoding="utf-8") as handle:
                records.append(json.load(handle))

    pending_values = []
    for parameter_value in parameter_values:
        slug = _slugify_value(parameter_value)
        if slug in completed_keys and resume:
            print(f"Skipping completed run {problem_name} [{test_type}={_serialise_value(parameter_value)}]")
            continue
        pending_values.append(parameter_value)

    if pending_values:
        worker_count = max(1, min(int(workers), len(pending_values)))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {}
            for parameter_value in pending_values:
                model_params = _build_model_params(baseline_params, test_type, parameter_value)
                future = executor.submit(_run_single_test_by_name, problem_name, model_params, max_steps, test_type, parameter_value)
                future_map[future] = parameter_value

            for future in as_completed(future_map):
                parameter_value = future_map[future]
                slug = _slugify_value(parameter_value)
                record, design_values = future.result()
                topology_artifacts = _write_topology_artifacts(problem_name, test_type, parameter_value, design_values, topology_dir)

                if topology_artifacts is not None:
                    record["artifacts"] = topology_artifacts
                record["checkpoint"] = {
                    "problem": problem_name,
                    "test_type": test_type,
                    "parameter_value": _serialise_value(parameter_value),
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                run_json = run_dir / f"{slug}.json"
                _atomic_write_json(run_json, record)
                records.append(record)

                _append_progress_line(progress_log, f"completed {parameter_name}={_serialise_value(parameter_value)}")
                _write_progress_state(problem_name, test_type, records, len(parameter_values), state_dir)
                _save_test_type_summary(problem_name, test_type, records, summary_dir, list(parameter_values))
                _persist_global_summary(out_dir)

    return records


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        _merge_existing_results(out_dir)
        print(f"Rebuilt combined summary in {out_dir}")
        return

    all_records: list[dict[str, Any]] = []

    for problem_name in args.problems:
        if problem_name not in PROBLEM_SPECS:
            raise KeyError(f"Unknown problem: {problem_name}")

        problem = PROBLEM_SPECS[problem_name]
        print(f"\n=== Problem: {problem_name} ===")
        for test_type in args.test_types:
            print(f"--- Test type: {test_type} ---")
            records = _run_test_type(problem_name, problem, test_type, args.max_steps, out_dir, args.resume, args.workers)
            all_records.extend(records)

    if not args.no_global_summary:
        _persist_global_summary(out_dir)
    if all_records:
        print(pd.DataFrame(all_records).head().to_string(index=False))
    print(f"Saved incremental results to {out_dir}")


if __name__ == "__main__":
    main()
