#!/bin/bash
#SBATCH --job-name=nso_bench
#SBATCH --output=logs/job_%a.out
#SBATCH --error=logs/job_%a.err
#SBATCH --partition=cpu        # Unity's standard CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2      # 2 cores handles the sparse solver well
#SBATCH --mem=12G              # 12GB to comfortably fit the massive 512x128 matrices max grids
#SBATCH --time=04:00:00        # Max time of 4 hours per task
#SBATCH --array=1-116           # Evaluates line 1 through 93 of problems.txt in parallel

# Load Unity modules (update miniconda version if needed on Unity)
module load miniconda/22.11.1-1

# IMPORTANT: Replace 'my_env' with the name of the conda environment you use on Unity
conda activate my_env

# Create the output directories if they don't exist
mkdir -p logs
mkdir -p benchmark_results

# Extract the specific problem name for this array task ID from problems.txt
# %a in SLURM maps to SLURM_ARRAY_TASK_ID
PROBLEM_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" problems.txt)

echo "Starting evaluation on problem: $PROBLEM_NAME"
python run_benchmark.py --problem "$PROBLEM_NAME" --max_steps 200