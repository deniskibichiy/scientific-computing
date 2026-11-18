import numpy as np
import matplotlib.pyplot as plt

# 1. Input Data: 1D Graph Laplacian for 5 unlabeled accounts in a chain
# Each interior node satisfies: 2*x_i - x_{i-1} - x_{i+1} = 0
# Diagonally dominant, guaranteeing convergence for Jacobi and Gauss-Seidel
A = np.array([
    [ 2., -1.,  0.,  0.,  0.],
    [-1.,  2., -1.,  0.,  0.],
    [ 0., -1.,  2., -1.,  0.],
    [ 0.,  0., -1.,  2., -1.],
    [ 0.,  0.,  0., -1.,  2.]
])

# Known anchor labels: Left anchor = 100 (High Risk), Right anchor = 20 (Low Risk)
b = np.array([100., 0., 0., 0., 20.])


def jacobi(A, b, tol=1e-8, max_iter=500):
    D = np.diag(A)
    R = A - np.diagflat(D)
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x_new, errors
        x = x_new
    return x, errors


def gauss_seidel(A, b, tol=1e-8, max_iter=500):
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (
                b[i]
                - A[i, :i] @ x[:i]
                - A[i, i + 1:] @ old[i + 1:]
            ) / A[i, i]
        error = np.linalg.norm(x - old, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x, errors
    return x, errors


# 2. Solve using Iterative Methods & Compare with Direct Solver
x_j, err_j = jacobi(A, b)
x_gs, err_gs = gauss_seidel(A, b)
x_exact = np.linalg.solve(A, b)

print("--- Graph Label Propagation (Iterative Solvers) ---")
print(f"Jacobi ({len(err_j)} iters):       {np.round(x_j, 4)}")
print(f"Gauss-Seidel ({len(err_gs)} iters): {np.round(x_gs, 4)}")
print(f"Direct exact solution:    {np.round(x_exact, 4)}")
print(f"Speedup (fewer iters):    {len(err_j) / len(err_gs):.2f}x")

# 3. Visualize Convergence & Propagated Node Scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Semi-log Convergence Curve
ax1.semilogy(err_j, label=f"Jacobi ({len(err_j)} iters)", linewidth=2)
ax1.semilogy(err_gs, label=f"Gauss-Seidel ({len(err_gs)} iters)", linewidth=2)
ax1.axhline(1e-8, color='gray', linestyle='--', linewidth=1, label='Tolerance (1e-8)')
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Maximum Change (Infinity Norm)")
ax1.set_title("Convergence: Jacobi vs. Gauss-Seidel")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# Right Panel: Propagated Risk Scores Across the Graph Chain
nodes = np.arange(0, 7)
full_scores = np.concatenate(([100.0], x_gs, [20.0]))
ax2.plot(nodes, full_scores, 'b--o', linewidth=2, label='Propagated Risk Score')
ax2.scatter([0, 6], [100.0, 20.0], color='red', s=100, zorder=5, label='Labeled Anchor Nodes')
ax2.set_xticks(nodes)
ax2.set_xticklabels(['Anchor\n(100)', 'Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5', 'Anchor\n(20)'])
ax2.set_ylabel("Risk Score")
ax2.set_title("Steady-State Label Propagation Across Graph")
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()