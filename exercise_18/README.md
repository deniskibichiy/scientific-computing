# Exercise 18: Random Numbers and Monte Carlo Methods

## 1. Explanation of Concept
**Monte Carlo methods** estimate probabilities, expectations, or complex integrals by running a large number of independent random simulations. By the **Law of Large Numbers**, if an event occurs with true probability $p$, the observed fraction of occurrences across $N$ independent trials converges to $p$ as $N \to \infty$:
$$\hat{p}_N = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{event}_i) \xrightarrow{N \to \infty} p$$
The standard error of a Monte Carlo estimate shrinks at a rate of $O(1/\sqrt{N})$, meaning quadrupling the number of simulations halves the estimation error. Using a fixed random seed (`np.random.default_rng(42)`) ensures full reproducibility.

## 2. Application
Monte Carlo simulation is used in cloud reliability engineering, financial risk modeling (Value at Risk), Bayesian inference, reinforcement learning rollouts, uncertainty quantification, and high-dimensional numerical integration.

**Example Application:** Simulating `100,000` random traffic scenarios from a normal distribution $\mathcal{N}(850, 120^2)$ to estimate the probability that demand exceeds a fixed capacity of `1000`.
* **Inputs & Operations:** Reproducible random number generator (`np.random.default_rng(42)`), `capacity = 1000.0`, `mean_demand = 850.0`, `std_demand = 120.0`, and `N = 100_000`. Operations include vectorized normal sampling (`rng.normal`), boolean thresholding (`demand > capacity`), cumulative convergence tracking (`np.cumsum`), and validation against the exact normal survival function (`scipy.stats.norm.sf`).
* **Parameter Scaling:** Increasing the simulation count $N$ from `1,000` to `100,000` dramatically dampens random jitter in the estimated probability, converging smoothly to the exact theoretical value (`10.565%`). Conversely, if traffic volatility (`std_demand`) increases from `120` to `180 req/s`, the distribution tails widen and the overload probability jumps from `~10.56%` to `~20.23%`, even though the average demand hasn't changed.

## 3. Real-World Problem Description
**Scenario:** An **MLOps team** operates an AI inference cluster provisioned for a maximum throughput of **`1,000 requests/sec`** (matching the SLA target from Exercise 17). In production, incoming user traffic is stochastic rather than constant—averaging **`850 req/s`** with a standard deviation of **`120 req/s`** due to unpredictable bursty usage.

**Problem:** Before committing to this cluster size, leadership needs to know the exact probability that a random traffic spike will overload the cluster and breach latency SLAs.

**Assumptions & Data:** Incoming request rates follow an independent normal distribution $\mathcal{N}(\mu = 850, \sigma = 120)$. Any scenario where `demand > 1000.0` counts as an overload event.

## 4. How the Concept Solves It
Instead of relying solely on average load (which looks safe at `850 < 1000`), we generate `100,000` synthetic demand scenarios in milliseconds using NumPy's vectorized `rng.normal()`. Counting the fraction of scenarios exceeding `1000 req/s` directly estimates the tail risk. Tracking the running average on a logarithmic x-axis confirms that `N = 100,000` is large enough for the estimate to stabilize.

## 5. Implementation & Output
*The code for this implementation is in `solution.py`.*

**Terminal Output Screenshot:**
![Execution Output](./task18_output.png)

**Visual Demand Distribution & Convergence Graph:**
![Monte Carlo Overload Risk and Convergence](plot.png)

**Output & Graph Interpretation:**
Across `100,000` simulated scenarios, the cluster overloads **`10,567` times**, yielding an estimated overload probability of **`10.5670%`**—within `0.002%` of the exact analytical probability (`10.5650%`).
* **Left Panel (Simulated Demand Distribution):** Shows the histogram of all `100,000` traffic scenarios. The red dashed line marks the `1,000 req/s` cluster ceiling, and the red shaded tail highlights the `10.57%` of scenarios that cause an SLA breach.
* **Right Panel (Monte Carlo Convergence):** Plots the running probability estimate against simulation count on a log scale. Early estimates (`N < 1,000`) fluctuate heavily, but as $N$ grows past `10,000`, the curve locks tightly onto the exact theoretical probability (red dashed line).