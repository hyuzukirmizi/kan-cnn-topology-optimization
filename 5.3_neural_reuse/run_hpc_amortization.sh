#!/bin/bash
#SBATCH --job-name=kan_amortization
#SBATCH --output=logs/amortization_%A_%a.out
#SBATCH --error=logs/amortization_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
# One array task per category: 0 = mbb_beam, 1 = cantilever_beam_full.
# Adjust --array if AMORT_CATEGORIES below is changed.
#SBATCH --array=0-1
#SBATCH --time=08:00:00

# KAN reusability amortization experiment (Section 5.3) on Unity HPC.
#
# Usage (from this directory on Unity):
#     sbatch run_hpc_amortization.sh
#
# Override any knob via environment variables at submit time, e.g.:
#     AMORT_STEPS=500 AMORT_PRETRAIN_STEPS=200 sbatch run_hpc_amortization.sh
#
# Results (per category): topology PNGs, compliance-vs-time curve PNGs,
# results.json and summary.md under $AMORT_OUT_DIR.

set -euo pipefail

# --- Experiment configuration (edit here or override via environment) ---
export AMORT_CATEGORIES="${AMORT_CATEGORIES:-mbb_beam cantilever_beam_full}"
export AMORT_PRETRAIN_STEPS="${AMORT_PRETRAIN_STEPS:-100}"
export AMORT_STEPS="${AMORT_STEPS:-300}"
export AMORT_KAN_LAYERS="${AMORT_KAN_LAYERS:-16,16}"
export AMORT_GRID="${AMORT_GRID:-8}"
export AMORT_K="${AMORT_K:-3}"
export AMORT_SEED="${AMORT_SEED:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CATEGORIES_ARRAY=($AMORT_CATEGORIES)
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CURRENT_CATEGORY="${CATEGORIES_ARRAY[$ARRAY_TASK_ID]}"

# --- Environment setup (mirrors 5.2_neural_size/run_hpc_sweep.sh) ---
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
    REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
fi
export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Keep the HPC checkout in sync with the latest repository state before running.
if [ -d "$REPO_ROOT/.git" ]; then
    echo "Syncing repository checkout from git..."
    git -C "$REPO_ROOT" fetch --all --prune >/dev/null 2>&1 || true
    branch="$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || true)"
    if [ -n "$branch" ]; then
        git -C "$REPO_ROOT" pull --ff-only origin "$branch" >/dev/null 2>&1 || echo "git pull skipped or failed; continuing with the existing checkout"
    fi
fi

WORK_ROOT="${SLURM_SUBMIT_DIR:-$ROOT_DIR}"
if [[ ! -w "$WORK_ROOT" ]]; then
    WORK_ROOT="$HOME/kan_amortization"
fi

export AMORT_OUT_DIR="${AMORT_OUT_DIR:-$WORK_ROOT/amortization_results}"

# Unity HPC recommendation: conda cache/env dirs on the /work partition.
export CONDA_PKGS_DIRS="${WORK_ROOT}/.conda/pkgs"
export CONDA_ENVS_PATH="${WORK_ROOT}/.conda/envs"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

mkdir -p "$WORK_ROOT/logs"
cd "$WORK_ROOT"

echo "========================================================"
echo "KAN Reusability Amortization Experiment on Unity HPC"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task ID: ${ARRAY_TASK_ID}"
echo "Category: ${CURRENT_CATEGORY}"
echo "Pretrain steps: ${AMORT_PRETRAIN_STEPS}, target budget: ${AMORT_STEPS}"
echo "KAN: layers=${AMORT_KAN_LAYERS}, grid=${AMORT_GRID}, k=${AMORT_K}, seed=${AMORT_SEED}"
echo "Repository Root: ${REPO_ROOT}"
echo "Output directory: ${AMORT_OUT_DIR}"
echo "Start time: $(date)"
echo "========================================================"

# --- Conda environment ---
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
    echo "Creating Conda environment '$ENV_NAME' with Python 3.11."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"

echo "Installing Python packages from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

# --- Sparse-solver patch (same workaround as 5.2's sweep script) ---
cat > "$WORK_ROOT/sitecustomize.py" <<'PY'
import os
import sys
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

repo_root = os.environ.get('REPO_ROOT')
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    import neural_structural_optimization.autograd_lib as autograd_lib
except Exception:
    autograd_lib = None

if autograd_lib is not None:
    def _patched_get_solver(a_entries, a_indices, size, sym_pos):
        del sym_pos
        a = scipy.sparse.coo_matrix((a_entries, a_indices), shape=(size,)*2).tocsc()
        try:
            splu_solver = scipy.sparse.linalg.splu(a)
            if splu_solver is not None:
                return splu_solver.solve
        except Exception:
            pass

        def solver(rhs):
            rhs = np.asarray(rhs)
            return np.asarray(scipy.sparse.linalg.spsolve(a, rhs))

        return solver

    autograd_lib._get_solver = _patched_get_solver
PY

echo "Running a tiny sparse-solver smoke test..."
python - <<PY
import numpy as np
import scipy.sparse
from neural_structural_optimization import autograd_lib

print('solver_module', autograd_lib.__file__)
a = scipy.sparse.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
entries = a.data
indices = np.vstack([a.tocoo().row, a.tocoo().col])
solver = autograd_lib._get_solver(entries, indices, 2, True)
if solver is None:
    raise RuntimeError('solver fallback returned None')
x = solver(np.array([1.0, 2.0]))
print('solver_smoke_test_ok', x)
PY

# --- Run the amortization experiment for this task's category ---
echo "--------------------------------------------------------"
echo "Running amortization experiment for category: ${CURRENT_CATEGORY}"

python "${REPO_ROOT}/5.3_neural_reuse/amortization_experiment.py" \
    --categories "${CURRENT_CATEGORY}" \
    --pretrain_steps "${AMORT_PRETRAIN_STEPS}" \
    --steps "${AMORT_STEPS}" \
    --kan_layers "${AMORT_KAN_LAYERS}" \
    --grid "${AMORT_GRID}" \
    --k "${AMORT_K}" \
    --seed "${AMORT_SEED}" \
    --out "${AMORT_OUT_DIR}/${CURRENT_CATEGORY}"

echo "--------------------------------------------------------"
echo "Amortization run for ${CURRENT_CATEGORY} completed."
echo "Results in: ${AMORT_OUT_DIR}/${CURRENT_CATEGORY}"
echo "========================================================"
echo "Job finished on: $(date)"
echo "========================================================"
