# Exercise 13: Curve Fitting and Approximation

## 1. Explanation of Concept
**Curve fitting** finds the optimal parameters of a chosen mathematical model so that it approximates noisy or empirical data as closely as possible. Unlike interpolation (which forces a curve to pass *exactly* through every data point), curve fitting captures the underlying trend while filtering out measurement noise.

In **Least Squares Approximation**, the solver chooses model parameters $\theta$ that minimize the Sum of Squared Residuals (SSR):
$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
where $y_i$ is the observed value and $\hat{y}_i = f(x_i; \theta)$ is the model's prediction. Model quality is commonly summarized using the **Root Mean Squared Error (RMSE)** and by inspecting whether the **residuals** ($r_i = y_i - \hat{y}_i$) are small and randomly scattered around zero.

## 2. Application
Curve fitting is foundational to regression modeling in Machine Learning, performance benchmarking, sensor calibration, growth-curve analysis, and demand forecasting.

**Example Application:** Fitting a quadratic model $y = ax^2 + bx + c$ to 7 empirical load-test measurements using `scipy.optimize.curve_fit` and forecasting latency at a higher load.
* **Inputs & Operations:** Two 1D NumPy arrays (`requests` and `latency_ms`) and a parameterized Python function `model(x, a, b, c)`. Operations include nonlinear least-squares optimization (`curve_fit`), prediction evaluation, RMSE calculation, and residual analysis.
* **Parameter Scaling:** If we downgrade the model complexity from a quadratic function ($ax^2 + bx + c$) to a simple linear model ($bx + c$), the RMSE jumps significantly and the residual plot forms a clear U-shaped pattern—proving the linear model underfits the nonlinear queuing delay. Conversely, extrapolating far beyond the observed range (e.g., predicting at `1000` requests instead of `400`) becomes unreliable because hardware saturation or memory limits will alter the true curve.

## 3. Real-World Problem Description
**Scenario:** An MLOps team is load-testing a deployed **Machine Learning Inference API** (serving a transformer model on a GPU). As concurrent API requests increase from `50` to `350`, GPU batch queuing causes response latency to grow nonlinearly.

**Problem:** The team needs a mathematical latency model to predict response time at `400` concurrent requests so they can configure Kubernetes autoscaling rules before latency breaches the `350 ms` Service Level Agreement (SLA).

**Assumptions & Data:** Within the `50–400` concurrent request regime, queuing delay follows a smooth quadratic trend without hitting hard out-of-memory (OOM) crashes. Observed telemetry:
* **Concurrent Requests:** `[50, 100, 150, 200, 250, 300, 350]`
* **Measured Latency (ms):** `[82, 95, 115, 145, 185, 238, 305]`

## 4. How the Concept Solves It
By passing the quadratic model `a*x**2 + b*x + c` and the load-test telemetry to `scipy.optimize.curve_fit()`, SciPy uses Levenberg-Marquardt least-squares optimization to find the exact coefficients `(a, b, c)` that minimize squared latency errors. Evaluating the fitted curve at `x = 400` yields a data-driven latency forecast, while inspecting the residuals confirms that the quadratic approximation accurately captures the system's scaling behavior.

## 5. Implementation & Output
*The code for this implementation is in `solution.py`.*

**Terminal Output Screenshot:**
![Execution Output](output.png)

**Visual Curve Fit & Residual Analysis Graph:**
![Latency Curve Fit and Residuals](plot.png)

**Output & Graph Interpretation:**
The least-squares fit yields $y = 0.002086x^2 - 0.0914x + 82.00$ with an **RMSE of `1.12 ms`** and **$R^2 = 0.9998$**, confirming an exceptionally tight fit to the observed telemetry. At `400` concurrent requests, the model forecasts a latency of **`379.14 ms`**, signaling that autoscaling must trigger *before* reaching 400 concurrent users to stay under a `350 ms` SLA.
* **Left Panel (Fitted Curve & Forecast):** Shows the observed load-test points (black dots), the smooth quadratic curve (blue), and the extrapolated forecast at 400 requests (red diamond).
* **Right Panel (Residual Plot):** Shows the prediction errors ($y_i - \hat{y}_i$) staying within $\pm 2\text{ ms}$ with no strong systematic drift, validating the quadratic model choice.