import sys
import time
import json
import argparse
import numpy as np
import xarray as xr
from pathlib import Path

from neural_structural_optimization import problems, topo_api
import models as pt

def get_resizes(width, height):
    """Automatically determine safe CNN/KAN upscaling based on grid dimensions."""
    if width % 16 == 0 and height % 16 == 0:
        return (1, 2, 2, 2, 2, 1)
    elif width % 8 == 0 and height % 8 == 0:
        return (1, 2, 2, 2, 1)
    else:
        return (1, 1, 2, 2, 1)

def run_model(model_fn, name, max_steps):
    print(f"[{name}] Starting...")
    start_time = time.time()
    ds = pt.train_lbfgs(model_fn(), max_steps)
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

    prob = problems.PROBLEMS_BY_NAME[args.problem]
    topo_args = topo_api.specified_task(prob)
    resizes = get_resizes(prob.width, prob.height)

    results = []
    datasets = []
    labels = []

    # 1. Pixel L-BFGS
    ds, metrics = run_model(lambda: pt.PixelModel(args=topo_args), "pixel", args.max_steps)
    results.append(metrics); datasets.append(ds); labels.append("pixel")

    # 2. CNN L-BFGS
    ds, metrics = run_model(lambda: pt.CNNModel(args=topo_args, resizes=resizes), "cnn", args.max_steps)
    results.append(metrics); datasets.append(ds); labels.append("cnn")

    # 3. Hybrid KAN
    ds, metrics = run_model(lambda: pt.KANModel(args=topo_args, resizes=resizes), "hybrid_kan", args.max_steps)
    results.append(metrics); datasets.append(ds); labels.append("hybrid_kan")

    # 4. Baseline KAN (with increased grid size for sharpness)
    ds, metrics = run_model(lambda: pt.CoordKANModel(args=topo_args, grid=100), "baseline_kan", args.max_steps)
    results.append(metrics); datasets.append(ds); labels.append("baseline_kan")

    # Save benchmark stats to JSON
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / f"{args.problem}_stats.json", "w") as f:
        json.dump({"problem": args.problem, "width": prob.width, "height": prob.height, "metrics": results}, f, indent=2)

    # Save physical designs to NetCDF
    dims = pd.Index(labels, name='model')
    xr.concat(datasets, dim=dims).to_netcdf(out_dir / f"{args.problem}_designs.nc")

if __name__ == "__main__":
    import pandas as pd # Needed for Index
    main()