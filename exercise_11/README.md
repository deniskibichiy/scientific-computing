# Exercise 11: Polynomial and Nonlinear Equation Solving

## 1. Explanation of Concept
A **polynomial equation** takes the form $a_n x^n + \dots + a_1 x + a_0 = 0$. More generally, a **system of nonlinear equations** involves multiple coupled equations where unknowns appear in nonlinear terms (such as powers, products, or transcendental functions). 

Unlike linear systems ($Ax = b$), nonlinear systems can have zero, one, or multiple distinct solutions, and rarely admit simple closed-form formulas. Numerical solvers (like Powell's hybrid method in `scipy.optimize.root`) start from an initial guess vector $z_0 = [x_0, y_0]$ and iteratively drive the **residual vector** $F(z) = 0$ toward zero. Because different starting guesses can converge to different roots, running the solver from multiple initial guesses is essential to map all solutions.

## 2. Application
Nonlinear solvers are used in robotics kinematics, chemical equilibrium modeling, geometric intersection, control systems, and finding operating or boundary points in nonlinear Machine Learning models.

**Example Application:** Finding the intersection points of a circle ($x^2 + y^2 = 4$) and a parabola ($y = x^2 - 1$) by reformulating them as a 2-variable residual system.
* **Inputs & Operations:** A vector-valued residual function `equations(z)` returning `[x**2 + y**2 - 4, y - (x**2 - 1)]`, and two initial guesses `[1.0, 1.0]` and `[-1.0, 1.0]`. Operations include multi-variable root finding via `scipy.optimize.root()`, convergence checking (`solution.success`), and residual verification.
* **Parameter Scaling:** If we shift the parabola's vertical offset parameter from `-1` to `+3` ($y = x^2 + 3$), its vertex moves to `(0, 3)`, completely outside the circle of radius `2`. Because no real intersection exists, `scipy.optimize.root()` will fail to converge (`Converged: False`) and return large non-zero residuals.

## 3. Real-World Problem Description
**Scenario:** In an ensemble **Machine Learning Anomaly Detection** system operating in a 2D PCA latent feature space $(z_1, z_2)$:
* **Model 1 (One-Class RBF / Hypersphere Detector):** Flags any point outside the circular decision boundary $z_1^2 + z_2^2 = 4$ as anomalous.
* **Model 2 (Quadratic Polynomial Classifier):** Separates two operational regimes along the parabolic boundary $z_2 = z_1^2 - 1$.

**Problem:** To audit where both models simultaneously switch their classifications (critical tie-breaker operating points), we need to compute the exact coordinates $(z_1, z_2)$ where the circular and parabolic decision boundaries intersect.

**Assumptions & Data:** Both boundaries are continuous and smooth in $\mathbb{R}^2$. Based on symmetry around the $z_2$-axis, we supply symmetric initial guesses `[1.0, 1.0]` and `[-1.0, 1.0]` to capture both intersections.

## 4. How the Concept Solves It
We rewrite both decision boundaries as zero-residual equations:
1. $f_1(z_1, z_2) = z_1^2 + z_2^2 - 4 = 0$
2. $f_2(z_1, z_2) = z_2 - (z_1^2 - 1) = 0$

Passing this system to `scipy.optimize.root()` from both initial guesses numerically locates each intersection to machine precision without needing manual algebraic substitution. Checking that the returned residuals are $\approx 0$ confirms valid intersections.

## 5. Implementation & Output
*The code for this implementation is in `solution.py`.*

**Terminal Output Screenshot:**
![Execution Output](./task11_output.png)

**Visual Decision Boundary Intersection Graph:**
![Nonlinear System Intersections](plot.png)

**Output & Graph Interpretation:**
From the two initial guesses, `scipy.optimize.root()` converges (`True`) to the two symmetric intersection points: `(1.5175, 1.3028)` and `(-1.5175, 1.3028)`, with residuals on the order of `1e-16`. In the plot, the blue circle represents the hypersphere anomaly boundary and the orange curve represents the quadratic classifier boundary. The red markers confirm the exact coordinates where both boundaries cross, showing how each initial guess (green `x`) converged to its nearest root.