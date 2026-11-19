import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 1. Input Parameters: MLOps Inference Cluster Traffic Simulation
rng = np.random.default_rng(42)
capacity = 1000.0      # Max cluster throughput (req/s)
mean_demand = 850.0    # Average incoming traffic (req/s)
std_demand = 120.0     # Traffic volatility / std dev (req/s)
N = 100_000            # Number of Monte Carlo scenarios

# 2. Run Vectorized Monte Carlo Simulation
demand = rng.normal(mean_demand, std_demand, N)
overloaded = demand > capacity
overload_probability = overloaded.mean()

# Exact theoretical probability for validation: P(X > capacity)
exact_probability = norm.sf(capacity, loc=mean_demand, scale=std_demand)

print("--- MLOps Inference Cluster Overload Risk (Monte Carlo) ---")
print(f"Total simulations (N):          {N:,}")
print(f"Overload scenarios observed:    {overloaded.sum():,} out of {N:,}")
print(f"Estimated overload probability: {overload_probability:.4%}")
print(f"Exact theoretical probability:  {exact_probability:.4%}")
print(f"Absolute estimation error:      {abs(overload_probability - exact_probability):.6%}")

# 3. Track Running Probability Convergence on a Log-Spaced Grid
running_probability = np.cumsum(overloaded) / np.arange(1, N + 1)
indices = np.unique(np.logspace(1, np.log10(N), 200).astype(int)) - 1

# 4. Visualize Demand Distribution & Monte Carlo Convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Demand Histogram with Shaded Overload Tail
bins = np.linspace(mean_demand - 4 * std_demand, mean_demand + 4 * std_demand, 60)
counts, edges = np.histogram(demand, bins=bins, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])

safe_mask = centers <= capacity
ax1.bar(centers[safe_mask], counts[safe_mask], width=edges[1] - edges[0],
        color='#1f77b4', alpha=0.75, label='Safe Load ($\\leq 1000$ req/s)')
ax1.bar(centers[~safe_mask], counts[~safe_mask], width=edges[1] - edges[0],
        color='#d62728', alpha=0.85, label=f'Overload Tail ({overload_probability:.2%})')
ax1.axvline(capacity, color='black', linestyle='--', linewidth=1.8, label='Cluster Capacity (1000 req/s)')

ax1.set_xlabel("Incoming Traffic Demand (req/s)")
ax1.set_ylabel("Probability Density")
ax1.set_title("Simulated Traffic Demand Distribution")
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Right Panel: Running Probability Convergence (Semi-Log)
ax2.semilogx(indices + 1, running_probability[indices] * 100,
             color='#1f77b4', linewidth=2, label='Monte Carlo Estimate')
ax2.axhline(exact_probability * 100, color='red', linestyle='--', linewidth=1.5,
            label=f'Exact Theoretical ({exact_probability:.2%})')
ax2.set_xlabel("Number of Simulations ($N$, Log Scale)")
ax2.set_ylabel("Estimated Overload Probability (%)")
ax2.set_title("Monte Carlo Convergence Toward True Risk")
ax2.grid(True, which='both', linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()