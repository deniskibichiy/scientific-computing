# Exercise 16: Ordinary Differential Equations — Euler Method

## 1. Explanation of Concept
An **Ordinary Differential Equation (ODE)** describes how a quantity changes over time:
$$\frac{dy}{dt} = f(t, y)$$
When an analytical solution is unavailable or impractical, **Euler's Method** approximates the solution numerically by starting from a known initial value $y_0$ and stepping forward in small time increments $h$:
$$y_{n+1} = y_n + h \cdot f(t_n, y_n)$$
Geometrically, Euler's method follows the local tangent line at $(t_n, y_n)$ for a step of width $h$. It is a **first-order method** ($O(h)$ global error): halving the step size $h$ roughly halves the approximation error, at the cost of doubling the number of computation steps.

## 2. Application
ODE solvers power thermal modeling, population and epidemic dynamics, capacitor charging, neural ODEs in Deep Learning, and continuous-time physics simulations.

**Example Application:** Simulating Newton's Law of Cooling $\frac{dT}{dt} = -k(T - T_{\text{ambient}})$ over 30 minutes using Euler's method and validating it against the exact analytical solution $T(t) = T_{\text{ambient}} + (T_0 - T_{\text{ambient}})e^{-kt}$.
* **Inputs & Operations:** Initial temperature $T_0 = 90.0^\circ\text{C}$, ambient temperature $T_{\text{ambient}} = 25.0^\circ\text{C}$, cooling constant $k = 0.12\text{ min}^{-1}$, step size $h = 1.0\text{ min}$, and `end_time = 30.0 min`. Operations include sequential Euler updates, exact exponential evaluation, and absolute error tracking.
* **Parameter Scaling:** Decreasing the step size parameter from `h = 1.0` min to `h = 0.25` min reduces the global truncation error by roughly $4\times$, pulling the Euler curve tightly onto the exact solution. Conversely, increasing `h` too aggressively (e.g., $h > \frac{2}{k} \approx 16.67\text{ min}$) violates Euler's stability condition, causing the simulated temperature to oscillate wildly and diverge to infinity.

## 3. Real-World Problem Description
**Scenario:** In an **MLOps GPU Cluster**, a high-performance training GPU reaches a peak operating temperature of `90.0 °C` during a large language model training run. Once the job finishes, the datacenter's cooling system (ambient temperature `25.0 °C`) cools the GPU down according to Newton's Law of Cooling with a thermal dissipation constant $k = 0.12\text{ min}^{-1}$.

**Problem:** The cluster orchestrator needs to simulate the GPU's cooldown trajectory over the next 30 minutes to predict when it will safely drop below thermal thresholds for hardware diagnostics or next-job scheduling, while quantifying the numerical error introduced by discrete time-stepping.

**Assumptions & Data:** Ambient datacenter temperature remains constant at `25.0 °C`, and the cooling coefficient $k = 0.12\text{ min}^{-1}$ is uniform throughout the cooldown phase.

## 4. How the Concept Solves It
We express the thermal dynamics as the first-order ODE $\frac{dT}{dt} = -0.12(T - 25.0)$ with $T(0) = 90.0$. Applying Euler's update rule step-by-step from $t = 0$ to $t = 30$ minutes predicts the GPU temperature at each minute. Comparing $h = 1.0\text{ min}$ and $h = 0.25\text{ min}$ against the exact analytical curve directly demonstrates how step-size selection governs numerical accuracy.

## 5. Implementation & Output
*The code for this implementation is in `solution.py`.*

**Terminal Output Screenshot:**
![Execution Output](./task16_output.png)

**Visual Cooldown Curve & Step-Size Error Graph:**
![GPU Cooldown Euler ODE and Error](plot.png)

**Output & Graph Interpretation:**
After 30 minutes, the exact analytical GPU temperature is **`26.78 °C`**. Euler's method with $h = 1.0\text{ min}$ predicts **`26.40 °C`** (a final error of `0.3742 °C`), while refining the step size to $h = 0.25\text{ min}$ improves the prediction to **`26.69 °C`** (reducing final error to `0.0886 °C`).
* **Left Panel (Cooldown Trajectory):** Euler's method consistently underestimates the true temperature because the cooling curve is concave up—following the steeper initial tangent line overshoots the drop at every step.
* **Right Panel (Absolute Error Over Time):** Shows the error peaking around minute 8 (where curvature is strongest) before decaying as both curves approach the `25 °C` asymptote. Cutting $h$ by $4\times$ visibly drops the error curve by roughly $4\times$ across the entire timeline.