#!/bin/bash
#SBATCH --job-name=kan_param_sweep
#SBATCH --output=logs/param_sweep_%A_%a.out
#SBATCH --error=logs/param_sweep_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
# The --array directive creates one job per hyperparameter (SWEEP_TEST_TYPES
# below). Default is "hidden_layers grid k" -> three jobs: array task 0 runs
# the hidden_layers sweep, task 1 runs the grid sweep, task 2 runs the k
# (spline order) sweep. Each job runs all SWEEP_PROBLEMS sequentially,
# single-threaded (no multiprocessing). If you change the number of entries
# in SWEEP_TEST_TYPES, update this range to match (0-(N-1)).
#SBATCH --array=0-2
#SBATCH --time=48:00:00
# For longer allocations, submit with a different --time value or export
# SWEEP_TIME_LIMIT before launch.

set -euo pipefail

# --- Single-location sweep configuration ---
# Edit these values to change the sweep without touching the Python runner.
export SWEEP_MAX_STEPS="${SWEEP_MAX_STEPS:-400}"
# The four 5.1 validation benchmark problems, run sequentially within each job.
export SWEEP_PROBLEMS="${SWEEP_PROBLEMS:-mbb_beam_384x128_0.3 cantilever_beam_two_point_256x192_0.15 roof_256x256_0.4 free_suspended_bridge_256x256_0.075}"
# One SLURM array task per hyperparameter: task 0 = hidden_layers, task 1 =
# grid, task 2 = k. Update #SBATCH --array above if you add/remove entries.
export SWEEP_TEST_TYPES="${SWEEP_TEST_TYPES:-hidden_layers grid k}"
# Runs are single-threaded (no multiprocessing) by design; kept at 1 worker.
export SWEEP_WORKERS="${SWEEP_WORKERS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

TEST_TYPE_ARRAY=($SWEEP_TEST_TYPES)
PROBLEMS_ARRAY=($SWEEP_PROBLEMS)
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CURRENT_TEST_TYPE="${TEST_TYPE_ARRAY[$ARRAY_TASK_ID]}"

# --- Environment Setup ---
# Get the root directory of the repository
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # When running via sbatch, SLURM_SUBMIT_DIR is the directory where the job was submitted.
    # The user has indicated they run sbatch from the script's directory inside the repo.
    # Therefore, the repo root is one level above SLURM_SUBMIT_DIR.
    ROOT_DIR="$SLURM_SUBMIT_DIR"
    REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
else
    # Fallback for local execution when not using Slurm.
    # This determines the script's own directory and goes up one level.
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

# Set a working directory; use SLURM's submit directory or a default
WORK_ROOT="${SLURM_SUBMIT_DIR:-$ROOT_DIR}"
if [[ ! -w "$WORK_ROOT" ]]; then
    # If the default work root isn't writable, fall back to a directory in $HOME
    WORK_ROOT="$HOME/kan_topo_sweep"
fi

export SWEEP_OUT_DIR="${SWEEP_OUT_DIR:-$WORK_ROOT/neural_size_results}"

# Unity HPC recommendation: Set conda cache and env directories to the /work partition
export CONDA_PKGS_DIRS="${WORK_ROOT}/.conda/pkgs"
export CONDA_ENVS_PATH="${WORK_ROOT}/.conda/envs"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

# Create directories for logs and results within the working directory
mkdir -p "$WORK_ROOT/logs"
cd "$WORK_ROOT"

echo "========================================================"
echo "Starting KAN Parameter Sweep on Unity HPC"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task ID: ${ARRAY_TASK_ID}"
echo "Test type: ${CURRENT_TEST_TYPE}"
echo "Problems: ${SWEEP_PROBLEMS}"
echo "Repository Root: ${REPO_ROOT}"
echo "Working Directory: $(pwd)"
echo "Start time: $(date)"
echo "========================================================"

# --- Conda Environment ---
# Load the conda module
module load conda/latest

# Set the path to conda if not already set
if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
else
    # Fallback to the default conda installation path
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

ENV_NAME="kan_topo_env"

# Create the conda environment if it doesn't exist
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating Conda environment '$ENV_NAME' with Python 3.11."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

# Activate the environment
conda activate "$ENV_NAME"

# --- Install Dependencies ---
echo "Installing Python packages from requirements.txt..."
python -m pip install --upgrade pip
# Ensure requirements are installed from the repository root
python -m pip install -r "${REPO_ROOT}/requirements.txt"
# --- Sanity Check the Solver ---
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

# --- Run the Sweep ---
echo "Executing ${CURRENT_TEST_TYPE} sweep across all problems"

# Single-threaded by design: one worker, no multiprocessing.
WORKERS_PER_RUN="1"

echo "Max steps: ${SWEEP_MAX_STEPS}"
echo "Test type for this job (array task ${ARRAY_TASK_ID}): ${CURRENT_TEST_TYPE}"
echo "Problems to run sequentially: ${SWEEP_PROBLEMS}"
echo "Workers per run: $WORKERS_PER_RUN"

for CURRENT_PROBLEM in "${PROBLEMS_ARRAY[@]}"; do
    echo "--------------------------------------------------------"
    echo "Starting ${CURRENT_TEST_TYPE} sweep for problem: ${CURRENT_PROBLEM}"

    # Create a specific output directory for each run
    SWEEP_OUT_DIR_RUN="${SWEEP_OUT_DIR}/${CURRENT_PROBLEM}/${CURRENT_TEST_TYPE}"
    mkdir -p "$SWEEP_OUT_DIR_RUN"

    python "${REPO_ROOT}/5.2_neural_size/parameter_sweep.py" \
        --max-steps "${SWEEP_MAX_STEPS}" \
        --problems "${CURRENT_PROBLEM}" \
        --test-types "${CURRENT_TEST_TYPE}" \
        --workers "${WORKERS_PER_RUN}" \
        --out-dir "${SWEEP_OUT_DIR_RUN}" \
        --resume
done

echo "--------------------------------------------------------"
echo "All ${CURRENT_TEST_TYPE} sweep runs across problems have completed."


echo "========================================================"
echo "Job finished on: $(date)"
echo "========================================================"
