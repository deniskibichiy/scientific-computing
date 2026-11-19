import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Problem Definition: MLOps GPU Inference Cluster Cost Optimization
# x = Type A GPU nodes ($12/hr, 100 req/s)
# y = Type B GPU nodes ($18/hr, 170 req/s)
# Constraint: 100*x + 170*y >= 1000 req/s

def cost(z):
    x, y = z
    return 12 * x + 18 * y


def capacity_constraint(z):
    x, y = z
    return 100 * x + 170 * y - 1000


constraints = {
    "type": "ineq",
    "fun": capacity_constraint
}
bounds = [(0, None), (0, None)]

# 2. Solve Continuous Relaxation with SLSQP
solution = minimize(
    cost,
    x0=[5, 3],
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

x_opt, y_opt = solution.x
print("--- MLOps GPU Cluster Allocation (Constrained Optimization) ---")
print(f"Optimization success:  {solution.success}")
print(f"Type A allocation (x): {x_opt:.3f} nodes")
print(f"Type B allocation (y): {y_opt:.3f} nodes")
print(f"Total throughput:      {100 * x_opt + 170 * y_opt:.2f} req/s")
print(f"Minimum hourly cost:   ${solution.fun:.2f}/hr")

# 3. Search Nearby Integer Combinations for Whole-Node Deployment
best_integer = None
feasible_ints = []

for a in range(0, 15):
    for b in range(0, 12):
        capacity = 100 * a + 170 * b
        if capacity >= 1000:
            total_cost = 12 * a + 18 * b
            feasible_ints.append((a, b))
            if best_integer is None or total_cost < best_integer[0]:
                best_integer = (total_cost, a, b, capacity)

int_cost, int_a, int_b, int_cap = best_integer
print("\n--- Best Whole-Node (Integer) Deployment ---")
print(f"Type A nodes:          {int_a}")
print(f"Type B nodes:          {int_b}")
print(f"Total throughput:      {int_cap} req/s")
print(f"Minimum integer cost:  ${int_cost:.2f}/hr")

# 4. Visualize Feasible Region, Cost Contours, and Optima
x_vals = np.linspace(0, 12, 300)
y_vals = np.linspace(0, 10, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z_cost = 12 * X + 18 * Y

# Constraint boundary: 100*x + 170*y = 1000  =>  y = (1000 - 100*x) / 170
y_boundary = (1000 - 100 * x_vals) / 170

plt.figure(figsize=(9, 6))

# Plot cost contours
contours = plt.contour(X, Y, Z_cost, levels=np.arange(60, 260, 20),
                       colors='gray', linestyles='--', linewidths=1.0, alpha=0.7)
plt.clabel(contours, inline=True, fontsize=8, fmt='$%d/hr')

# Shade feasible region (above the constraint line)
plt.fill_between(x_vals, np.maximum(y_boundary, 0), 10,
                 color='#2ca02c', alpha=0.18, label='Feasible Region ($100x + 170y \\geq 1000$)')
plt.plot(x_vals, y_boundary, color='#2ca02c', linewidth=2, label='Throughput SLA Boundary')

# Plot feasible integer grid points
fi = np.array(feasible_ints)
plt.scatter(fi[:, 0], fi[:, 1], color='darkgreen', s=18, alpha=0.45, label='Feasible Integer Nodes')

# Highlight Continuous and Integer Optima
plt.scatter([x_opt], [y_opt], color='red', marker='*', s=220, zorder=6,
            label=f'Continuous Optimum ({x_opt:.2f}, {y_opt:.2f}) — ${solution.fun:.2f}/hr')
plt.scatter([int_a], [int_b], color='blue', marker='D', s=90, zorder=7,
            label=f'Best Integer ({int_a}, {int_b}) — ${int_cost:.2f}/hr')

plt.xlim(-0.5, 12)
plt.ylim(-0.5, 10)
plt.xlabel("Type A GPU Nodes ($12/hr, 100 req/s)")
plt.ylabel("Type B GPU Nodes ($18/hr, 170 req/s)")
plt.title("GPU Cluster Cost Optimization: Feasible Region & Cost Contours")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()