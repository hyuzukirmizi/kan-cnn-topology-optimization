#!/bin/bash
#SBATCH --job-name=kan_param_sweep
#SBATCH --output=logs/param_sweep_%j.out
#SBATCH --error=logs/param_sweep_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

# --- Environment Setup ---
# Get the root directory of the repository
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Set a working directory; use SLURM's submit directory or a default
WORK_ROOT="${SLURM_SUBMIT_DIR:-$ROOT_DIR}"
if [[ ! -w "$WORK_ROOT" ]]; then
    # If the default work root isn't writable, fall back to a directory in $HOME
    WORK_ROOT="$HOME/kan_topo_sweep"
fi

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

# --- Run the Sweep ---
echo "Executing the parameter sweep script..."
# Run the Python script from the 5.2_neural_size directory
python "${REPO_ROOT}/5.2_neural_size/parameter_sweep.py"

echo "========================================================"
echo "Job finished on: $(date)"
echo "========================================================"
