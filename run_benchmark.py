import sys
import time
import json
import argparse
import numpy as np
import xarray as xr
from pathlib import Path

from neural_structural_optimization import problems, topo_api
import models as pt

def get_model_kwargs(width, height):
    """Automatically determine safe CNN/KAN upscaling based on grid dimensions."""
    if width % 16 == 0 and height % 16 == 0:
        return {"resizes": (1, 2, 2, 2, 2, 1), "conv_filters": (128, 64, 32, 16, 8, 1)}
    elif width % 8 == 0 and height % 8 == 0:
        return {"resizes": (1, 2, 2, 2, 1), "conv_filters": (128, 64, 32, 16, 1)}
    else:
        return {"resizes": (1, 1, 2, 2, 1), "conv_filters": (128, 64, 32, 16, 1)}

def run_model(model_fn, name, max_steps):
    print(f"[{name}] Starting...")
    start_time = time.time()
    ds = pt.train_lbfgs(model_fn(), max_steps)
    duration = time.time() - start_time
    
    best_loss = float(np.nanmin(ds.loss.values))
    best_step = int(np.nanargmin(ds.loss.values))
    
    print(f"[{name}] Finished in {duration:.2f}s | Best Loss: {best_loss:.4f}")
    return ds, {"model": name, "time_sec": duration, "best_loss": best_loss, "best_step": best_step}

def run_adaptive_kan(model_fn, name, max_steps):
    print(f"[{name}] Starting Adaptive Refinement...")
    start_time = time.time()
    
    # Schedule: 1st half of steps at grid=10, 2nd half refines to grid=30
    schedule = [(max_steps // 2, 10), (max_steps - (max_steps // 2), 30)]
    ds = pt.train_lbfgs_adaptive_kan(model_fn(), schedule)
    duration = time.time() - start_time
    
    best_loss = float(np.nanmin(ds.loss.values))
    best_step = int(np.nanargmin(ds.loss.values))
    print(f"[{name}] Finished in {duration:.2f}s | Best Loss: {best_loss:.4f}")
    return ds, {"model": name, "time_sec": duration, "best_loss": best_loss, "best_step": best_step}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()
    
    # Strip any hidden Windows carriage returns (\r) or whitespace
    args.problem = args.problem.strip()

    prob = problems.PROBLEMS_BY_NAME[args.problem]
    topo_args = topo_api.specified_task(prob)
    model_kwargs = get_model_kwargs(prob.width, prob.height)

    results = []
    datasets = []
    labels = []

    # 1. Pixel L-BFGS
    ds, metrics = run_model(lambda: pt.PixelModel(args=topo_args), "pixel", args.max_steps)
    results.append(metrics); datasets.append(ds); labels.append("pixel")

    # 2. CNN L-BFGS
    ds, metrics = run_model(lambda: pt.CNNModel(args=topo_args, **model_kwargs), "cnn", args.max_steps)
    results.append(metrics); datasets.append(ds); labels.append("cnn")

    # 3. Hybrid KAN
    ds, metrics = run_model(lambda: pt.KANModel(args=topo_args, **model_kwargs), "hybrid_kan", args.max_steps)
    results.append(metrics); datasets.append(ds); labels.append("hybrid_kan")

    # 4. Baseline KAN (Adaptive Refinement starting from a coarse grid)
    ds, metrics = run_adaptive_kan(lambda: pt.CoordKANModel(args=topo_args, grid=10), "baseline_kan", args.max_steps)
    results.append(metrics); datasets.append(ds); labels.append("baseline_kan")

    # Save benchmark stats to JSON
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / f"{args.problem}_stats.json", "w") as f:
        json.dump({"problem": args.problem, "width": prob.width, "height": prob.height, "metrics": results}, f, indent=2)

    # Save physical designs to NetCDF
    dims = pd.Index(labels, name='model')
    xr.concat(datasets, dim=dims).to_netcdf(out_dir / f"{args.problem}_designs.nc")

    # Print summary table
    print("\n" + "="*50)
    print(f"BENCHMARK SUMMARY: {args.problem}")
    print("="*50)
    print(f"{'Model':<15} | {'Best Loss':<12} | {'Step':<8} | {'Time (s)':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['model']:<15} | {r['best_loss']:<12.4f} | {r['best_step']:<8} | {r['time_sec']:<10.2f}")
    print("="*50)

if __name__ == "__main__":
    import pandas as pd # Needed for Index
    main()