import json
import copy

def write_notebook(path, cells):
    notebook = {
        'cells': [],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    for ctype, src in cells:
        notebook['cells'].append({
            'cell_type': ctype,
            'metadata': {},
            'source': [line + '\n' for line in src.strip().split('\n')]
        })
        if ctype == 'code':
            notebook['cells'][-1]['outputs'] = []
            notebook['cells'][-1]['execution_count'] = None

    with open(path, 'w') as f:
        json.dump(notebook, f, indent=2)


run_tests_cells = [
    ('markdown', '## Run Tests (MMA, OC, Pixel, CNN, and KAN)\nRun each optimization procedure separately to validate the model components locally.'),
    ('code', '''import os
import sys
from pathlib import Path

# Windows DLL fix for torch inside notebook kernels
if sys.platform == "win32":
    import importlib.util
    _spec = importlib.util.find_spec("torch")
    if _spec:
        _torch_lib = Path(_spec.origin).parent / "lib"
        if _torch_lib.is_dir():
            try:
                os.add_dll_directory(str(_torch_lib))
            except Exception:
                pass

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# Import local standalone modules
from neural_structural_optimization import problems, topo_api
import models as pt

print(f"PyTorch Version: {torch.__version__}")
'''),
    ('markdown', '### 1. Problem Setup'),
    ('code', '''# Basic MBB Beam testing case (60x20)
problem = problems.mbb_beam(height=20, width=60, density=0.5)
args = topo_api.specified_task(problem)

max_iterations = 20  # You can increase this for full benchmarking

print(f"Solving for MBB Beam of size: {problem.width} x {problem.height}")
print("Note: CNN and KAN network seed=0 init guarantees fully reproducible baselines.")
'''),
    ('markdown', '### 2. Method of Moving Asymptotes (MMA)'),
    ('code', '''# Runs MMA using nlopt (if installed)
try:
    import nlopt
    # PixelModel is the appropriate formulation for raw generic MMA grid scaling
    pixel_mma = pt.PixelModel(args=args)
    ds_mma = pt.method_of_moving_asymptotes(pixel_mma, max_iterations)
    
    plt.figure(figsize=(6, 4))
    plt.plot(ds_mma.loss.values)
    plt.title("MMA Compliance Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
except ImportError:
    print("nlopt is not installed. Skipping MMA.")
    ds_mma = None
'''),
    ('markdown', '### 3. Optimality Criteria (OC)'),
    ('code', '''# Runs Optimality Criteria method directly matrix-driven
pixel_oc = pt.PixelModel(args=args)
ds_oc = pt.optimality_criteria(pixel_oc, max_iterations)

plt.figure(figsize=(6, 4))
plt.plot(ds_oc.loss.values)
plt.title("OC Compliance Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
'''),
    ('markdown', '### 4. Pixel Model (L-BFGS)'),
    ('code', '''# Runs directly on Pixel values optimized through Autograd back to Scipy L-BFGS
pixel_lbfgs = pt.PixelModel(args=args)
ds_pixel = pt.train_lbfgs(pixel_lbfgs, max_iterations)

plt.figure(figsize=(6, 4))
plt.plot(ds_pixel.loss.values)
plt.title("Pixel (L-BFGS) Compliance Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
'''),
    ('markdown', '### 5. CNN Model (L-BFGS)'),
    ('code', '''# Reparameterized structural generation using an UpSampling CNN
cnn_kwargs = {
    "resizes": (1, 1, 2, 2, 1) # Matches 60x20 dimension scale
}
# Using default seed=0
cnn_model = pt.CNNModel(args=args, **cnn_kwargs)
ds_cnn = pt.train_lbfgs(cnn_model, max_iterations)

plt.figure(figsize=(6, 4))
plt.plot(ds_cnn.loss.values)
plt.title("CNN (L-BFGS) Compliance Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
'''),
    ('markdown', '### 6. Hybrid KAN Model (L-BFGS)'),
    ('code', '''# Reparameterized formulation substituting dense projection paths with B-Splines (KAN)
kan_kwargs = {
    "resizes": (1, 1, 2, 2, 1),
    "latent_size": 32,
    "hidden_size": 128,
    "num_kan_layers": 2,
    "grid": 5,
    "k": 3,
}
# Using default seed=0
kan_model = pt.KANModel(args=args, **kan_kwargs)
ds_kan = pt.train_lbfgs(kan_model, max_iterations)

plt.figure(figsize=(6, 4))
plt.plot(ds_kan.loss.values)
plt.title("KAN (L-BFGS) Compliance Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
'''),
    ('markdown', '### Summary Table Comparison'),
    ('code', '''# Format and display results tabularly
results_list = []
datasets = [
    ("MMA", ds_mma),
    ("OC", ds_oc),
    ("Pixel-LBFGS", ds_pixel),
    ("CNN-LBFGS", ds_cnn),
    ("KAN-LBFGS", ds_kan)
]

for name, ds in datasets:
    if ds is not None:
        best_step = int(np.nanargmin(ds.loss.values))
        results_list.append({
            "model": name,
            "best_compliance": float(ds.loss.values[best_step]),
            "converged_step": best_step
        })

df_summary = pd.DataFrame(results_list).sort_values("best_compliance").reset_index(drop=True)
display(df_summary)
''')
]

write_notebook(r'C:\Users\yzkrm\Desktop\MIE Research\kan-cnn-comparison\run_tests.ipynb', run_tests_cells)


# Now create optimization-examples.ipynb
example_cells = [
    ('markdown', '# Examples of neural reparameterization for structural optimization\nThis notebook demonstrates step-by-step visualizations and comparisons across models.'),
    ('code', '''import os
import sys
from pathlib import Path

# Windows DLL fix for torch inside notebook kernels
if sys.platform == "win32":
    import importlib.util
    _spec = importlib.util.find_spec("torch")
    if _spec:
        _torch_lib = Path(_spec.origin).parent / "lib"
        if _torch_lib.is_dir():
            try:
                os.add_dll_directory(str(_torch_lib))
            except Exception:
                pass

import torch
import numpy as np
import matplotlib.pyplot as plt

from neural_structural_optimization import problems, topo_api
import models as pt

def plot_design(ds, iteration, title):
    """Utility to plot the spatial design matrices properly."""
    idx = min(iteration, len(ds.design.values) - 1)
    # The grid is an xarray DataArray where we grab the specific step indices.
    image = ds.design.values[idx]
    
    plt.figure(figsize=(10, 4))
    plt.imshow(image, cmap='Greys', vmin=0, vmax=1)
    plt.colorbar(label='Material Density')
    plt.title(f"{title} (Step {idx+1} / Loss: {ds.loss.values[idx]:.3f})")
    plt.axis('off')
    plt.show()
'''),
    ('markdown', '## Initialization and Setup\nWe initialize an MBB Beam structure directly mimicking classic benchmarks.'),
    ('code', '''problem = problems.mbb_beam(height=20, width=60, density=0.5)
args = topo_api.specified_task(problem)
max_iterations = 30
print(f"Problem grid: {problem.width} x {problem.height}")
'''),
    ('markdown', '## 1. Run Optimality Criteria Engine (Baseline)\nWatch the structure iteratively form through basic OC solver step updates.'),
    ('code', '''pixel_oc = pt.PixelModel(args=args)
ds_oc = pt.optimality_criteria(pixel_oc, max_iterations=max_iterations)

# Plot early stage
plot_design(ds_oc, iteration=5, title="OC Model - Early Iteration")
# Plot late stage
plot_design(ds_oc, iteration=max_iterations, title="OC Model - Final Iteration")
'''),
    ('markdown', '## 2. Evaluate Pure Pixel Model (L-BFGS)\nA pure pixel-by-pixel grid evaluation driven by neural autograd.'),
    ('code', '''pixel_model = pt.PixelModel(args=args)
ds_pixel = pt.train_lbfgs(pixel_model, max_iterations=max_iterations)
plot_design(ds_pixel, iteration=max_iterations, title="Pixel Model L-BFGS - Final")
'''),
    ('markdown', '## 3. Train Structural Optimization via CNN Reparameterization\nNotice how the CNN enforces spatial smoothness because its convolutions tie local densities together systematically reducing checkerboarding.'),
    ('code', '''# Utilizing default reproducible deterministic config -> Seed = 0 internally inside Model.__init__
cnn_model = pt.CNNModel(args=args, resizes=(1, 1, 2, 2, 1))
ds_cnn = pt.train_lbfgs(cnn_model, max_iterations=max_iterations)

plot_design(ds_cnn, iteration=5, title="CNN Model - Early Iteration")
plot_design(ds_cnn, iteration=max_iterations, title="CNN Model - Final Output")
'''),
    ('markdown', '## 4. Train Topological Optimization via KAN\nTesting B-Splines for latent activation attention weighting locally substituting rigid dense paths.'),
    ('code', '''# Same initialization seeds ensuring zero external parameter mismatch
kan_model = pt.KANModel(args=args, resizes=(1, 1, 2, 2, 1), num_kan_layers=2)
ds_kan = pt.train_lbfgs(kan_model, max_iterations=max_iterations)

plot_design(ds_kan, iteration=max_iterations, title="Hybrid KAN Model - Final Output")
''')
]

write_notebook(r'C:\Users\yzkrm\Desktop\MIE Research\kan-cnn-comparison\optimization-examples.ipynb', example_cells)
