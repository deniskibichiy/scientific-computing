import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

# 1. Input Data: Shared Regularized Gram Matrix A = (X^T X + lambda*I)
A = np.array([
    [ 4.0, -1.0,  0.0],
    [-1.0,  4.0, -1.0],
    [ 0.0, -1.0,  3.0]
])

# Three different target projection vectors (b_i = X^T y_i) for 3 ML tasks
target_vectors = [
    np.array([15.0, 10.0, 10.0]),  # Task 1: Peak Load
    np.array([16.0, 11.0,  9.0]),  # Task 2: Average Load
    np.array([14.0, 12.0, 11.0])   # Task 3: Grid Loss
]

print("--- Multi-Task Ridge Regression via LU Decomposition ---")

# 2. Factor A ONLY ONCE
lu, piv = lu_factor(A)

solutions = []
for i, b in enumerate(target_vectors, start=1):
    w = lu_solve((lu, piv), b)
    solutions.append(w)
    residual = b - A @ w
    print(f"Task {i} Optimal Weights: {np.round(w, 4)}")
    print(f"Task {i} Residual Norm:   {np.linalg.norm(residual):.2e}\n")

# Verify against direct solver on Task 1
direct_w1 = np.linalg.solve(A, target_vectors[0])
print("Direct np.linalg.solve (Task 1):", np.round(direct_w1, 4))

# 3. Benchmark: LU Reuse vs Repeated Direct Solve on a 250x250 Matrix
rng = np.random.default_rng(42)
N = 250
# Create a well-conditioned symmetric positive-definite matrix
M = rng.normal(size=(N, N))
A_large = M.T @ M + 10.0 * np.eye(N)

k_values = np.arange(1, 41, 4)
B_all = rng.normal(size=(40, N))

direct_times = []
lu_times = []

for k in k_values:
    # Repeated direct solve
    t0 = time.perf_counter()
    for j in range(k):
        np.linalg.solve(A_large, B_all[j])
    direct_times.append((time.perf_counter() - t0) * 1000)

    # Factor once, solve k times
    t0 = time.perf_counter()
    lu_large, piv_large = lu_factor(A_large)
    for j in range(k):
        lu_solve((lu_large, piv_large), B_all[j])
    lu_times.append((time.perf_counter() - t0) * 1000)

# 4. Visualize Solutions & Runtime Scaling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Subplot: Solved Weights per Task
x_idx = np.arange(3)
width = 0.25
for idx, sol in enumerate(solutions):
    ax1.bar(x_idx + idx * width, sol, width, label=f'Task {idx + 1}')

ax1.set_xticks(x_idx + width)
ax1.set_xticklabels(['Weight $w_1$', 'Weight $w_2$', 'Weight $w_3$'])
ax1.set_ylabel('Weight Value')
ax1.set_title('Multi-Task Regression Weights (Single LU Factor)')
ax1.grid(axis='y', linestyle=':', alpha=0.7)
ax1.legend()

# Right Subplot: Cumulative Runtime Scaling
ax2.plot(k_values, direct_times, 'r-o', linewidth=2, label='Repeated np.linalg.solve ($O(k n^3)$)')
ax2.plot(k_values, lu_times, 'b-s', linewidth=2, label='LU Factor Once + Solve ($O(n^3 + k n^2)$)')
ax2.set_xlabel('Number of Target Vectors ($k$)')
ax2.set_ylabel('Total Runtime (ms)')
ax2.set_title(f'Scaling Advantage of LU Reuse ({N}x{N} Matrix)')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()