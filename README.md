# KAN vs CNN Topology Optimization Models Comparison

This repository provides a standalone comparison of the topological structural optimization using various models including **MMA** (Method of Moving Asymptotes), **OC** (Optimality Criteria), **Pixel-based** density modeling, **CNN** (Convolutional Neural Networks), and **KAN** (Kolmogorov-Arnold Networks). 

The underlying optimization problem is solved completely utilizing a PyTorch parameterization and optimization framework without relying on TensorFlow. 

## Structure
- `models.py`: All optimized network architectures (`CNNModel`, `KANModel` (Hybrid), `CoordKANModel` (Baseline KAN), `PixelModel`) natively written in PyTorch. Includes wrappers for PyTorch optimization (LBFGS, MMA, OC).
- `optimization-examples.ipynb`: A standalone notebook defining structural topology cases and benchmarks running MMA, OC, Pixel, CNN, Hybrid KAN, and Baseline KAN optimizations. Compiles training iteration stats, logs visual comparisons, and generates precise numerical compliance benchmark tables side-by-side.
- `neural_structural_optimization/`: The backend Finite Element Methods (FEM) optimization simulation package utilizing `autograd`.
- `kan/`: Specialized modular package running actual network implementations for KANs via b-spline parameters.

## Prerequisites and Installation

Because it relies on optimization using standard matrices and deep learning optimizers via PyTorch, setting up a proper environment is essential.

1. Ensure you have Python installed natively (preferably `Python 3.9+`).
2. Create and activate a Virtual Environment (Optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate.ps1
# On Mac/Linux
source venv/bin/activate
```
3. Install the dependencies specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch
- `autograd`, `numpy`, `scipy` (simulation constraints and FEA matrix resolution)
- `pandas`, `xarray`, `bottleneck`, `matplotlib`, `seaborn` (for storing evaluation datasets, rapid forward-fills, and plotting)
- `nlopt` (Requirement to run MMA)
- `jupyter`

## Running the benchmarks

Once the installation is complete, everything can be executed dynamically through the provided Jupyter Notebook.

Run Jupyter:
```bash
jupyter notebook
```
Then navigate and execute `optimization-examples.ipynb`. The notebook defines optimization scenarios and will iterate through MMA, OC, Pixel, CNN, Hybrid KAN, and pure Baseline KAN algorithms natively, providing plotting, graphs, and final compliance numerical tables against standard structural benchmarks.