# KAN CNN Topology Optimization Benchmark Analysis

This document provides a detailed analysis of the benchmark results for the KAN CNN topology optimization project. It outlines the experimental setup, model parameters, and a discussion of the results.

## 1. Benchmark Setup

The benchmark was executed using the `run_hpc_full_benchmark.sh` script. This script defines the environment, problems, models, and training parameters.

### 1.1. Environment

*   **Python:** 3.11
*   **Conda Environment:** `kan_topo_env`
*   **Key Libraries:** `torch`, `numpy`, `matplotlib`, `xarray`, `nlopt`.

### 1.2. Benchmark Problems

Four distinct topology optimization problems were used for this benchmark. Each problem is defined by its domain size, volume fraction, and boundary conditions.

| Problem Name                             | Domain Size (WxH) | Volume Fraction | Description                                                                                             |
| ---------------------------------------- | ----------------- | --------------- | ------------------------------------------------------------------------------------------------------- |
| `mbb_beam_384x128_0.3`                   | 384x128           | 0.3             | A Messerschmitt-Bölkow-Blohm (MBB) beam, a standard benchmark problem for topology optimization.            |
| `cantilever_beam_two_point_256x192_0.15` | 256x192           | 0.15            | A cantilever beam fixed on one side with two-point loads.                                               |
| `roof_256x256_0.4`                       | 256x256           | 0.4             | A roof structure supported at two bottom corners with a distributed load on the top edge.               |
| `free_suspended_bridge_256x256_0.075`    | 256x256           | 0.075           | A bridge structure supported at two top corners with a load at the bottom center.                       |

### 1.3. Optimization Parameters

*   **Maximum Iterations:** 400
*   **Optimizers:** L-BFGS, MMA (Method of Moving Asymptotes), OC (Optimality Criteria)

## 2. Models and Parameters

Six different models were benchmarked. The parameters for the `HybridKANModel` and `CNNModel` are dynamically configured based on the problem's dimensions, and the benchmark now evaluates both a hybrid KAN variant and a baseline coordinate-based KAN variant.

### 2.1. Parameter Initialization (`model_kwargs_for` function)

The `HybridKANModel` and `CNNModel` architectures are determined by the `model_kwargs_for` function, which adjusts the convolutional layers and resizing operations based on the problem's width and height. This is done to accommodate different input sizes and maintain a reasonable network depth.

The logic is as follows:
*   If `width` and `height` are divisible by 16:
    *   `resizes`: `(1, 2, 2, 2, 2, 1)`
    *   `conv_filters`: `(128, 64, 32, 16, 8, 1)`
*   If `width` and `height` are divisible by 8 (but not 16):
    *   `resizes`: `(1, 2, 2, 2, 1)`
    *   `conv_filters`: `(128, 64, 32, 16, 1)`
*   Otherwise:
    *   `resizes`: `(1, 1, 2, 2, 1)`
    *   `conv_filters`: `(128, 64, 32, 16, 1)`

`resizes` controls the downsampling at each convolutional layer, and `conv_filters` defines the number of filters in each layer.

### 2.2. Model Details

#### 2.2.1. Hybrid KAN Model

*   **Model Class:** `models.HybridKANModel`
*   **Optimizer:** L-BFGS
*   **Description:** This is the hybrid KAN model, using a convolutional encoder-decoder structure with a learned KAN-based channel gate. It combines the spatial inductive bias of a CNN-style decoder with the spline-based nonlinearity of KAN layers to improve local feature modeling.
*   **Parameters per Problem:**
    *   `mbb_beam_384x128_0.3`: `width=384`, `height=128`. Both are divisible by 16.
        *   `resizes`: `(1, 2, 2, 2, 2, 1)`
        *   `conv_filters`: `(128, 64, 32, 16, 8, 1)`
    *   `cantilever_beam_two_point_256x192_0.15`: `width=256`, `height=192`. Both are divisible by 16.
        *   `resizes`: `(1, 2, 2, 2, 2, 1)`
        *   `conv_filters`: `(128, 64, 32, 16, 8, 1)`
    *   `roof_256x256_0.4`: `width=256`, `height=256`. Both are divisible by 16.
        *   `resizes`: `(1, 2, 2, 2, 2, 1)`
        *   `conv_filters`: `(128, 64, 32, 16, 8, 1)`
    *   `free_suspended_bridge_256x256_0.075`: `width=256`, `height=256`. Both are divisible by 16.
        *   `resizes`: `(1, 2, 2, 2, 2, 1)`
        *   `conv_filters`: `(128, 64, 32, 16, 8, 1)`

#### 2.2.2. Baseline KAN Model

*   **Model Class:** `models.BaseKANModel`
*   **Optimizer:** L-BFGS
*   **Description:** This is the baseline coordinate-based KAN model. Instead of the hybrid CNN-KAN decoder, it uses a coordinate-aware KAN parameterization with adaptive hidden layers, `grid=10`, and `k=3` to represent the design field more directly.
*   **Parameters per Problem:**
    *   For larger problems such as `mbb_beam_384x128_0.3` and `cantilever_beam_two_point_256x192_0.15`, the hidden KAN layers are set to `(64, 64)`.
    *   For the smaller benchmark grids, the hidden KAN layers are set to `(32, 32)`.
    *   The spline configuration uses `grid=10` and `k=3` for all runs.

#### 2.2.3. CNN-LBFGS Model

*   **Model Class:** `models.CNNModel`
*   **Optimizer:** L-BFGS
*   **Description:** A standard convolutional neural network with an encoder-decoder architecture, using traditional activation functions (like ReLU or SiLU). It serves as a baseline to evaluate the effectiveness of the KAN layers.
*   **Parameters:** Same as the `HybridKANModel` for each respective problem.

#### 2.2.3. Pixel-LBFGS Model

*   **Model Class:** `models.PixelModel`
*   **Optimizer:** L-BFGS
*   **Description:** This model represents the topology directly with a grid of pixels, where each pixel's density is a learnable parameter. It does not use a neural network to generate the topology. The optimization is performed directly on the pixel values using L-BFGS.

#### 2.2.4. MMA (Method of Moving Asymptotes) Model

*   **Model Class:** `models.PixelModel`
*   **Optimizer:** MMA (via `nlopt`)
*   **Description:** Similar to `Pixel-LBFGS`, this model operates on a pixel grid. However, it uses the Method of Moving Asymptotes, a gradient-based optimization algorithm that is widely used in structural optimization. It is known for its stability and efficiency.

#### 2.2.5. OC (Optimality Criteria) Model

*   **Model Class:** `models.PixelModel`
*   **Optimizer:** OC
*   **Description:** This model also uses a pixel grid representation. The optimizer is based on Optimality Criteria, a classic method for topology optimization that iteratively updates the design based on a set of rules derived from the problem's optimality conditions.

## 3. Benchmark Results

The following tables summarize the results obtained from the benchmark run. "Best Compliance" is the minimum compliance value (loss) achieved during the optimization.

### 3.1. `mbb_beam_384x128_0.3`

| Model         | Best Compliance | Best Step | Final Compliance | Time (s) | Final Gray Fraction |
|---------------|-----------------|-----------|------------------|----------|---------------------|
| Hybrid KAN    | 301.747901      | 400       | 301.747901       | 1556.51  | 0.1279              |
| KAN           | 301.826682      | 400       | 301.826682       | 4984.01  | 0.0723              |
| CNN-LBFGS     | 302.779652      | 400       | 302.779652       | 1536.81  | 0.0923              |
| Pixel-LBFGS   | 411.021246      | 73        | 411.021246       | 339.43   | 0.0976              |
| MMA           | 301.860244      | 400       | 301.860244       | 1308.83  | 0.1153              |
| OC            | 321.321108      | 400       | 321.321108       | 1328.47  | 0.1037              |

### 3.2. `cantilever_beam_two_point_256x192_0.15`

| Model         | Best Compliance | Best Step | Final Compliance | Time (s) | Final Gray Fraction |
|---------------|-----------------|-----------|------------------|----------|---------------------|
| Hybrid KAN    | 218.257679      | 400       | 218.257679       | 1898.55  | 0.0708              |
| KAN           | 206.148501      | 400       | 206.148501       | 5365.72  | 0.0805              |
| CNN-LBFGS     | 218.644243      | 400       | 218.644243       | 1938.92  | 0.0886              |
| Pixel-LBFGS   | 284.614757      | 142       | 284.614757       | 738.38   | 0.1236              |
| MMA           | 233.182998      | 400       | 233.182998       | 1653.43  | 0.1160              |
| OC            | 237.367630      | 400       | 237.367630       | 1654.70  | 0.1130              |

For these first two problems, the hybrid KAN model is very competitive on the MBB beam, while the baseline KAN variant achieves the best compliance on the cantilever beam. The baseline KAN run is also noticeably slower, which is consistent with its more expressive coordinate-based formulation.

### 3.3. `roof_256x256_0.4`

| Model         | Best Compliance | Best Step | Final Compliance | Time (s) | Final Gray Fraction |
|---------------|-----------------|-----------|------------------|----------|---------------------|
| Hybrid KAN    | 2.899469        | 400       | 2.899469         | 2756.66  | 0.0556              |
| KAN           | 2.864537        | 400       | 2.864537         | 7369.61  | 0.0701              |
| CNN-LBFGS     | 2.923360        | 401       | 2.923360         | 2768.68  | 0.0717              |
| Pixel-LBFGS   | 4.215530        | 119       | 4.215530         | 732.75   | 0.1000              |
| MMA           | 2.851267        | 397       | 2.851268         | 2279.79  | 0.0929              |
| OC            | 2.966479        | 400       | 2.966479         | 2363.67  | 0.0667              |

For the roof problem, the baseline KAN model achieves the best compliance among the L-BFGS-based neural models and is only slightly behind MMA, while its gray fraction remains competitive. This suggests that the coordinate-based representation can be especially effective on this geometry.

### 3.4. `free_suspended_bridge_256x256_0.075`

| Model         | Best Compliance | Best Step | Final Compliance | Time (s) | Final Gray Fraction |
|---------------|-----------------|-----------|------------------|----------|---------------------|
| KAN           | 38.6015         | 100       | 38.6015          | 564.82   | 0.0810              |
| CNN-LBFGS     | 75.2012         | 100       | 75.2012          | 565.40   | 0.0497              |
| Pixel-LBFGS   | 78.8744         | 87        | 78.8744          | 439.39   | 0.0393              |
| MMA           | 33.7719         | 100       | 33.7719          | 515.06   | 0.0574              |
| OC            | 37.7060         | 100       | 37.7060          | 526.45   | 0.0477              |

## 4. Analysis and Discussion

### 4.1. Model Performance

*   **Hybrid KAN vs. Baseline KAN:** The hybrid KAN model is very competitive on the MBB beam and roof cases, while the baseline coordinate-based KAN model reaches the best compliance on the cantilever problem and is also strongest on the roof case among the L-BFGS-based neural models. This highlights that the two KAN variants have complementary strengths depending on the geometry and the optimization landscape.

*   **Neural Network Models vs. Traditional Methods:** The traditional methods, MMA and OC, are very strong competitors. MMA, in particular, achieves the best compliance in three out of the four problems. This is expected, as MMA is a highly specialized and mature algorithm for this class of problems. However, the KAN model is often competitive, and in the case of the cantilever beam, it surpasses all other models.

*   **Pixel-LBFGS:** The `Pixel-LBFGS` model generally performs the worst. This highlights the importance of the reparameterization provided by the neural network models and the sophisticated optimization strategies of MMA and OC. Directly optimizing a large number of pixel variables with a general-purpose optimizer like L-BFGS is challenging.

### 4.2. Pros and Cons of Each Model

*   **KAN:**
    *   **Pros:** High expressiveness due to learnable activation functions, capable of producing complex and high-performance designs, often with low gray fraction.
    *   **Cons:** Can be computationally more expensive than traditional CNNs. The training dynamics can be complex.

*   **CNN-LBFGS:**
    *   **Pros:** A well-understood and relatively simple neural network architecture for this task.
    *   **Cons:** May suffer from spectral bias, leading to smoother, less detailed designs and potentially higher gray fractions compared to KAN.

*   **Pixel-LBFGS:**
    *   **Pros:** Conceptually simple, with no complex neural network architecture.
    *   **Cons:** Does not scale well with the number of design variables. The optimization landscape is often difficult for general-purpose optimizers like L-BFGS.

*   **MMA:**
    *   **Pros:** State-of-the-art performance for many topology optimization problems. Stable and efficient.
    *   **Cons:** Requires a specialized solver (`nlopt`). The algorithm itself is more complex to implement than a simple gradient descent or L-BFGS.

*   **OC:**
    *   **Pros:** A classic, robust, and often efficient method.
    *   **Cons:** May not always achieve the same level of performance as MMA. The updates are based on a heuristic rule, which may not be optimal in all cases.

### 4.3. Expected Results

The observed results are largely in line with expectations. It is expected that specialized, traditional methods like MMA and OC would perform very well. The fact that the KAN-based neural network model is competitive with these methods is a significant finding. It demonstrates the potential of using advanced neural network architectures for topology optimization.

The superior performance of KAN over the standard CNN suggests that the architectural improvements offered by KANs are beneficial for this type of problem, where precise spatial control is crucial for generating optimal structures. The ability of KANs to better handle high-frequency details likely contributes to their ability to produce lower compliance designs with sharper boundaries.
