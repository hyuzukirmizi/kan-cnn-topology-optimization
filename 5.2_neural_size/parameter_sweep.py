
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# --- Setup Paths ---
# Correctly resolve the repository root from the script's location
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().resolve()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Dynamically import the models and problems
try:
    from neural_structural_optimization import problems, topo_api
    import models as pt
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run from the `5.2_neural_size` directory or that the repository root is in the Python path.")
    sys.exit(1)


# --- Configuration ---
MAX_ITERATIONS = 400
RESULTS_DIR = REPO_ROOT / "5.2_neural_size" / "results"
PLOTS_DIR = RESULTS_DIR
REPORTS_DIR = RESULTS_DIR
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# --- Problem and Parameter Definitions ---
PROBLEM_SPECS = {
    "mbb_beam_384x128_0.3": problems.PROBLEMS_BY_NAME["mbb_beam_384x128_0.3"],
    "cantilever_beam_two_point_256x192_0.15": problems.PROBLEMS_BY_NAME["cantilever_beam_two_point_256x192_0.15"],
    "roof_256x256_0.4": problems.PROBLEMS_BY_NAME["roof_256x256_0.4"],
    "free_suspended_bridge_256x256_0.075": problems.PROBLEMS_BY_NAME["free_suspended_bridge_256x256_0.075"],
}

# Parameter sweeps
PARAMS_TO_SWEEP = {
    "hidden_layers": [(l, l) for l in [64, 32, 16, 8]],
    "grid": [10, 8, 6, 4],
    "k": [4, 3, 2, 1],
}

# Baseline parameters (from the 5.1 benchmark)
def get_baseline_params(problem_name):
    total_elements = PROBLEM_SPECS[problem_name].width * PROBLEM_SPECS[problem_name].height
    if total_elements > 40000:
        return {"hidden_layers": (64, 64), "grid": 10, "k": 3}
    else:
        return {"hidden_layers": (32, 32), "grid": 10, "k": 3}

# --- Core Functions ---
def run_single_test(problem_name, problem, model_params, step_count):
    """Runs a single topology optimization test with given parameters."""
    print(f"Running test for {problem_name} with params: {model_params}")
    topo_args = topo_api.specified_task(problem)
    
    # Use 'kan_layers' argument for BaseKANModel
    kan_model_params = model_params.copy()
    kan_model_params['kan_layers'] = kan_model_params.pop('hidden_layers')
    
    model_factory = lambda: pt.BaseKANModel(args=topo_args, **kan_model_params)
    
    start_time = time.time()
    try:
        ds = pt.train_lbfgs(model_factory(), step_count)
        elapsed_time = time.time() - start_time
        
        losses = np.asarray(ds.loss.values).flatten()
        valid_losses = losses[~np.isnan(losses)]
        best_compliance = float(np.min(valid_losses)) if len(valid_losses) > 0 else float('nan')
        
        final_design = ds.design.ffill("step").isel(step=-1)
        
        return {
            "best_compliance": best_compliance,
            "time_sec": elapsed_time,
            "final_design": final_design,
            "params": model_params,
        }
    except Exception as e:
        print(f"ERROR running test for {problem_name} with params {model_params}: {e}")
        return {
            "best_compliance": float('nan'),
            "time_sec": time.time() - start_time,
            "final_design": None,
            "params": model_params,
            "error": str(e),
        }

def save_topology_image(problem_name, param_name, param_value, design):
    """Saves an image of the final topology."""
    if design is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(design.values.T, cmap="Greys", origin="upper")
    ax.set_title(f"{problem_name}\n{param_name}={param_value}")
    ax.axis("off")
    
    # Sanitize param_value for filename
    param_str = str(param_value).replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")
    filename = f"{problem_name}_{param_name}_{param_str}_topology.png"
    path = PLOTS_DIR / filename
    
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_parameter_impact(problem_name, results_df, param_name):
    """Plots the impact of a single parameter on compliance using a line plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Ensure param values are strings for categorical plotting, which works well for line plots too
    results_df[param_name] = results_df[param_name].astype(str)
    
    # Create the line plot
    ax.plot(
        results_df[param_name], 
        results_df["best_compliance"], 
        marker='o', 
        linestyle='-', 
        color="royalblue"
    )
    
    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel("Best Compliance (Loss)")
    ax.set_title(f"Impact of {param_name.title()} on Compliance for {problem_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Annotate points with compliance values
    for i, val in enumerate(results_df["best_compliance"]):
        if not np.isnan(val):
            ax.text(i, val, f" {val:.2f}", va='bottom', ha='left')

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    
    filename = f"{problem_name}_{param_name}_impact.png"
    path = PLOTS_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path

# --- Main Execution Logic ---
def main():
    full_results = {}

    for problem_name, problem in PROBLEM_SPECS.items():
        print(f"--- Starting Parameter Sweep for: {problem_name} ---")
        problem_results = {"hidden_layers": [], "grid": [], "k": []}

        # 1. Sweep Hidden Layers
        param_name = "hidden_layers"
        baseline_params = get_baseline_params(problem_name)
        for param_value in PARAMS_TO_SWEEP[param_name]:
            current_params = baseline_params.copy()
            current_params[param_name] = param_value
            result = run_single_test(problem_name, problem, current_params, MAX_ITERATIONS)
            problem_results[param_name].append(result)
            save_topology_image(problem_name, param_name, param_value, result.get("final_design"))

        # 2. Sweep Grid Size
        param_name = "grid"
        baseline_params = get_baseline_params(problem_name)
        for param_value in PARAMS_TO_SWEEP[param_name]:
            current_params = baseline_params.copy()
            current_params[param_name] = param_value
            result = run_single_test(problem_name, problem, current_params, MAX_ITERATIONS)
            problem_results[param_name].append(result)
            save_topology_image(problem_name, param_name, param_value, result.get("final_design"))
            
        # 3. Sweep Spline Order (k)
        param_name = "k"
        baseline_params = get_baseline_params(problem_name)
        for param_value in PARAMS_TO_SWEEP[param_name]:
            current_params = baseline_params.copy()
            current_params[param_name] = param_value
            result = run_single_test(problem_name, problem, current_params, MAX_ITERATIONS)
            problem_results[param_name].append(result)
            save_topology_image(problem_name, param_name, param_value, result.get("final_design"))

        full_results[problem_name] = problem_results

    # --- Generate Plots and Reports ---
    print("\n--- Generating Plots and Reports ---")
    report_content = "# Base KAN Parameter Sweep Analysis\n\n"

    for problem_name, results_by_param in full_results.items():
        report_content += f"## {problem_name}\n\n"
        
        for param_name, results_list in results_by_param.items():
            if not results_list:
                continue
                
            df = pd.DataFrame(results_list)
            # Extract the parameter value from the 'params' dictionary
            df[param_name] = df['params'].apply(lambda p: p[param_name])
            df = df.sort_values(by=param_name, ascending=False)
            
            # Generate and save plot
            plot_path = plot_parameter_impact(problem_name, df, param_name)
            
            # Add to report
            report_content += f"### {param_name.title()} Sweep\n\n"
            report_content += f"![{param_name} impact plot]({plot_path.name})\n\n"
            report_content += df[["best_compliance", "time_sec", param_name]].to_markdown(index=False)
            report_content += "\n\n"

    report_path = REPORTS_DIR / "parameter_sweep_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
