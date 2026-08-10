#!/bin/bash
#SBATCH --job-name=kan_topo_evolution
#SBATCH --output=logs/topo_evolution_%j.out
#SBATCH --error=logs/topo_evolution_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # sbatch stages a copy of this script under /var/spool/slurm/..., so
    # ${BASH_SOURCE[0]} does NOT point at the repo when running under Slurm.
    # SLURM_SUBMIT_DIR is the directory sbatch was invoked from instead.
    ROOT_DIR="$SLURM_SUBMIT_DIR"
else
    # Fallback for local (non-Slurm) execution.
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
export REPO_ROOT
export ROOT_DIR
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
WORK_ROOT="$ROOT_DIR"
if [[ ! -w "$WORK_ROOT" ]]; then
    WORK_ROOT="$HOME/kan_topo_benchmark"
fi

# Unity HPC recommendation: keep conda's package cache and env directories
# off $HOME so they don't fill up the home quota.
export CONDA_PKGS_DIRS="${WORK_ROOT}/.conda/pkgs"
export CONDA_ENVS_PATH="${WORK_ROOT}/.conda/envs"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/topology_evolution_results" "$WORK_ROOT/topology_evolution_plots"
cd "$WORK_ROOT"

echo "========================================================"
echo "Starting KAN topology evolution figure generation on Unity HPC"
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
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr

from neural_structural_optimization import problems, topo_api

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path.cwd())).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pt = _load_module("models", REPO_ROOT / "models.py")

# ROOT_DIR is the directory this script lives in (5.1_validation_benchmarks),
# resolved explicitly rather than relying on cwd, since sbatch may stage this
# script elsewhere while WORK_ROOT (cwd) can differ from it.
_ROOT_DIR = Path(os.environ["ROOT_DIR"]).resolve()
tep = _load_module("topology_evolution_plotting", _ROOT_DIR / "topology_evolution_plotting.py")

ROOT = Path.cwd()
DATA_DIR = ROOT / "topology_evolution_results"
PLOT_DIR = ROOT / "topology_evolution_plots"
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

MAX_ITERATIONS = 400
KAN_LAYERS = (64, 64)   # matches the 5.1 benchmark: total elements > 40,000 for both problems
GRID = 10
SPLINE_ORDER = 3
MILESTONE_STEPS = (25, 100, 250, 400)
THUMB_WIDTH_IN = 0.28   # confirmed locally in test_topology_evolution_plot.ipynb

# (display label, PROBLEMS_BY_NAME key)
PROBLEM_SPECS = [
    ("Two-Point Cantilever", "cantilever_beam_two_point_256x192_0.15"),
    ("Free Suspended Bridge", "free_suspended_bridge_256x256_0.075"),
]


def train_kan(problem_name, problem):
    topo_args = topo_api.specified_task(problem)
    model = pt.BaseKANModel(args=topo_args, kan_layers=KAN_LAYERS, grid=GRID, k=SPLINE_ORDER)
    print(f"[{problem_name}] Training KAN for {MAX_ITERATIONS} iterations "
          f"(layers={KAN_LAYERS}, grid={GRID}, k={SPLINE_ORDER})")
    # progress_every is omitted for compatibility with older checked-out
    # copies of models.py that predate that keyword argument.
    ds = pt.train_lbfgs(model, MAX_ITERATIONS)
    ds["gray_fraction"] = tep.gray_fraction_series(ds.design)
    ds["relative_compliance"] = ds.loss / float(ds.loss.isel(step=0))
    ds.to_netcdf(DATA_DIR / f"{problem_name}_kan_history.nc")
    return ds


CAPTION = (
    r"Convergence histories for the Two-Point Cantilever and Free Suspended Bridge "
    r"problems. The plots display Compliance (solid lines) and Gray Element "
    r"Fraction (dashed lines) over 400 optimization iterations, with thumbnail overlays "
    r"showing the KAN topology evolution at intermediate steps."
)

results = []
for label, problem_key in PROBLEM_SPECS:
    problem = problems.PROBLEMS_BY_NAME[problem_key]
    ds = train_kan(problem_key, problem)
    results.append((label, ds))

FIGURE_PATH = PLOT_DIR / "topology_evolution_cantilever_bridge.png"
tep.plot_topology_evolution(
    results, FIGURE_PATH,
    max_iterations=MAX_ITERATIONS, milestone_steps=MILESTONE_STEPS,
    thumb_width_in=THUMB_WIDTH_IN,
)

CAPTION_PATH = PLOT_DIR / "topology_evolution_cantilever_bridge_caption.tex"
with CAPTION_PATH.open("w", encoding="utf-8") as f:
    f.write("\\begin{figure}[htbp]\n")
    f.write("    \\centering\n")
    f.write(f"    \\includegraphics[width=\\textwidth]{{{FIGURE_PATH.name}}}\n")
    f.write(f"    \\caption{{{CAPTION}}}\n")
    f.write("    \\label{fig:topology_evolution}\n")
    f.write("\\end{figure}\n")

print(f"Saved figure to {FIGURE_PATH}")
print(f"Saved LaTeX snippet to {CAPTION_PATH}")
print(f"Saved raw histories to {DATA_DIR}")
PY

echo "========================================================"
echo "Job finished on: $(date)"
echo "========================================================"
