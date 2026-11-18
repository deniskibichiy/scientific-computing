import numpy as np
import matplotlib.pyplot as plt

# 1. Input Data: ML Inference Cluster Telemetry
# x = active instances of Model A (Vision Transformer: 4 GPU cores, 8 GB VRAM)
# y = active instances of Model B (LLM Microservice:   2 GPU cores, 12 GB VRAM)
# Total observed usage: 20 GPU cores, 72 GB VRAM

A = np.array([
    [4.0, 2.0],   # GPU core constraint: 4x + 2y = 20
    [8.0, 12.0]   # VRAM constraint:     8x + 12y = 72
])
b = np.array([20.0, 72.0])

# 2. Solve Ax = b
solution = np.linalg.solve(A, b)
residual = b - A @ solution

print("--- MLOps Cluster Resource Allocation ---")
print(f"Active Model A (Vision) instances: {solution[0]:.0f}")
print(f"Active Model B (LLM) instances:    {solution[1]:.0f}")
print("Verification (A @ solution):      ", A @ solution)
print("Residual vector (b - A @ x):      ", residual)
print(f"Residual norm:                     {np.linalg.norm(residual):.2e}")

# 3. Visualize the Linear System as Intersecting Lines
x_vals = np.linspace(0, 6, 100)

# Rearrange equations to solve for y:
# Eq 1: 4x + 2y = 20  ->  y = (20 - 4x) / 2
# Eq 2: 8x + 12y = 72 ->  y = (72 - 8x) / 12
y_gpu = (b[0] - A[0, 0] * x_vals) / A[0, 1]
y_vram = (b[1] - A[1, 0] * x_vals) / A[1, 1]

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_gpu, label='GPU Constraint: 4x + 2y = 20', color='#1f77b4', linewidth=2)
plt.plot(x_vals, y_vram, label='VRAM Constraint: 8x + 12y = 72', color='#ff7f0e', linewidth=2)

# Highlight the exact intersection point
plt.scatter(solution[0], solution[1], color='red', zorder=5, s=80,
            label=f'Solution: Model A={solution[0]:.0f}, Model B={solution[1]:.0f}')
plt.annotate(f'({solution[0]:.0f}, {solution[1]:.0f})',
             xy=(solution[0], solution[1]),
             xytext=(solution[0] + 0.3, solution[1] + 0.6),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=11, fontweight='bold', color='red')

plt.xlim(0, 6)
plt.ylim(0, 10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.title('Linear System Solution: ML Cluster Resource Allocation')
plt.xlabel('Model A Instances (Vision Transformer)')
plt.ylabel('Model B Instances (LLM Microservice)')
plt.legend(loc='upper right')

plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()