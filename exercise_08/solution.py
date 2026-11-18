import numpy as np
import matplotlib.pyplot as plt

# 1. Input Data: Adjacency matrix for a 4-user social interaction network
# Node order: User A, User B, User C, User D
nodes = ["User A", "User B", "User C", "User D"]
A = np.array([
    [0., 1., 1., 0.],  # User A connects to B, C
    [1., 0., 1., 1.],  # User B connects to A, C, D
    [1., 1., 0., 1.],  # User C connects to A, B, D
    [0., 1., 1., 0.]   # User D connects to B, C
])

# 2. Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Extract the dominant eigenvalue and corresponding eigenvector
dominant_idx = np.argmax(np.real(eigenvalues))
dominant_value = np.real(eigenvalues[dominant_idx])
v = np.real(eigenvectors[:, dominant_idx])

# Normalize dominant eigenvector to get centrality scores summing to 1.0
dominant_vector = np.abs(v)
centrality = dominant_vector / dominant_vector.sum()

# Verify Av = lambda * v
verification_error = np.linalg.norm(A @ v - dominant_value * v)

print("--- Graph ML: Eigenvector Centrality Ranking ---")
print(f"Dominant Eigenvalue (lambda): {dominant_value:.4f}")
print("\nNormalized Influence Scores:")
for node, score in zip(nodes, centrality):
    print(f"  {node}: {score:.4f} ({score * 100:.2f}%)")
print(f"\nVerification Error ||Av - lambda*v||: {verification_error:.2e}")

# 3. Visualize the Network Topology & Centrality Scores (pure matplotlib)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Network Graph with node size proportional to centrality
coords = {
    0: (-1.0,  0.0),  # User A (left)
    1: ( 0.0,  0.8),  # User B (top hub)
    2: ( 0.0, -0.8),  # User C (bottom hub)
    3: ( 1.0,  0.0)   # User D (right)
}

# Draw edges
n = len(nodes)
for i in range(n):
    for j in range(i + 1, n):
        if A[i, j] == 1.0:
            x_vals = [coords[i][0], coords[j][0]]
            y_vals = [coords[i][1], coords[j][1]]
            ax1.plot(x_vals, y_vals, color='gray', linewidth=2, zorder=1)

# Draw nodes scaled by centrality
x_pos = [coords[i][0] for i in range(n)]
y_pos = [coords[i][1] for i in range(n)]
node_sizes = centrality * 6000
colors = ['#4c72b0', '#dd8452', '#dd8452', '#4c72b0']

ax1.scatter(x_pos, y_pos, s=node_sizes, c=colors, edgecolors='black', linewidth=1.5, zorder=2)
for i, name in enumerate(nodes):
    ax1.text(coords[i][0], coords[i][1], f"{name}\n({centrality[i]:.2f})",
             ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=3)

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.3, 1.3)
ax1.set_title("Interaction Network (Node Size = Influence)")
ax1.axis('off')

# Right Panel: Bar Chart of Eigenvector Centrality Scores
bars = ax2.bar(nodes, centrality, color=colors, edgecolor='black', alpha=0.9)
for bar, score in zip(bars, centrality):
    ax2.text(bar.get_x() + bar.get_width() / 2, score + 0.008,
             f"{score:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylim(0, 0.35)
ax2.set_ylabel("Normalized Eigenvector Centrality")
ax2.set_title(f"User Influence Ranking (Dominant $\\lambda$ = {dominant_value:.2f})")
ax2.grid(axis='y', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()