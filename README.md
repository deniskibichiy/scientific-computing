# Scientific Computing — Applied to ML, Data Science & MLOps

A collection of 18 standalone scientific computing exercises (`BIT 2118`) implemented in Python using **NumPy**, **SciPy**, and **Matplotlib**. Every exercise pairs core numerical methods with a real-world **Machine Learning**, **Data Science**, **Computer Vision**, or **MLOps** problem.

## Repository Structure

Each exercise is self-contained in its own directory:

```text
exercise_XX/
├── README.md      # Concept, real-world problem framing, and interpretation
├── solution.py    # Executable Python implementation
├── output.png     # Terminal execution screenshot
└── plot.png       # Automated Matplotlib visualization
```

## Exercise Index

| # | Directory | Numerical Topic | Applied ML / Data Science / MLOps Problem |
|---|---|---|---|
| 01 | [`exercise_01`](./exercise_01) | NumPy Arrays & Indexing | ML feature matrix preprocessing & weather telemetry |
| 02 | [`exercise_02`](./exercise_02) | Vectorization & Dot Products | High-dimensional NLP embedding similarity |
| 03 | [`exercise_03`](./exercise_03) | Floating-Point Errors | ML loss evaluation & numerical gradient stability |
| 04 | [`exercise_04`](./exercise_04) | Matrix Algebra | Computer Vision bounding box augmentation |
| 05 | [`exercise_05`](./exercise_05) | Direct Linear Systems ($Ax = b$) | MLOps inference cluster resource allocation |
| 06 | [`exercise_06`](./exercise_06) | LU Decomposition | Multi-Task Ridge Regression solver & scaling |
| 07 | [`exercise_07`](./exercise_07) | Iterative Solvers (Jacobi & Gauss-Seidel) | Graph-based semi-supervised label propagation |
| 08 | [`exercise_08`](./exercise_08) | Eigenvalues & Eigenvectors | Graph ML eigenvector centrality & influence ranking |
| 09 | [`exercise_09`](./exercise_09) | Root Finding (Bisection & Fixed-Point) | Fraud-detection ML classification threshold calibration |
| 10 | [`exercise_10`](./exercise_10) | Root Finding (Newton-Raphson & Secant) | AI infrastructure investment ROI / IRR analysis |
| 11 | [`exercise_11`](./exercise_11) | Nonlinear Systems of Equations | 2D ML anomaly & classifier decision boundary intersections |
| 12 | [`exercise_12`](./exercise_12) | Interpolation (Linear & Cubic Splines) | Time-series sensor upsampling for ML feature alignment |
| 13 | [`exercise_13`](./exercise_13) | Curve Fitting & Least Squares | ML inference API latency scaling & SLA forecasting |
| 14 | [`exercise_14`](./exercise_14) | Numerical Differentiation | Autonomous vehicle velocity feature extraction |
| 15 | [`exercise_15`](./exercise_15) | Numerical Integration (Trapezoid & Simpson) | Green AI GPU training energy & carbon footprint tracking |
| 16 | [`exercise_16`](./exercise_16) | Ordinary Differential Equations (Euler) | MLOps GPU thermal cooldown simulation |
| 17 | [`exercise_17`](./exercise_17) | Constrained Numerical Optimization (SLSQP) | MLOps GPU inference cluster cost optimization |
| 18 | [`exercise_18`](./exercise_18) | Random Numbers & Monte Carlo Methods | MLOps inference cluster SLA overload risk analysis |

## Setup & Usage

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd scientific-computing
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy scipy matplotlib
   ```

4. **Run any exercise**
   ```bash
   cd exercise_18
   python solution.py
   ```