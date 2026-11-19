import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# 1. Define the Nonlinear System (ML Decision Boundaries in 2D Latent Space)
# Model 1 (Circular Anomaly Boundary):  z1^2 + z2^2 = 4
# Model 2 (Quadratic Classifier):       z2 = z1^2 - 1
def equations(z):
    z1, z2 = z
    return [
        z1**2 + z2**2 - 4,
        z2 - (z1**2 - 1)
    ]

# 2. Solve from Multiple Initial Guesses to Capture All Intersections
initial_guesses = ([1.0, 1.0], [-1.0, 1.0])
intersections = []

print("--- ML Decision Boundary Intersections (Nonlinear System) ---")
for guess in initial_guesses:
    solution = root(equations, guess)
    res = equations(solution.x)
    intersections.append(solution.x)
    print(f"Initial guess: {guess}")
    print(f"Intersection:  [{solution.x[0]:.6f}, {solution.x[1]:.6f}]")
    print(f"Residuals:     [{res[0]:.2e}, {res[1]:.2e}]")
    print(f"Converged:     {solution.success}\n")

# 3. Visualize the Nonlinear Boundaries and Intersections
theta = np.linspace(0, 2 * np.pi, 400)
circle_x = 2 * np.cos(theta)
circle_y = 2 * np.sin(theta)

parabola_x = np.linspace(-2.5, 2.5, 400)
parabola_y = parabola_x**2 - 1

plt.figure(figsize=(8, 7))
plt.plot(circle_x, circle_y, color='#1f77b4', linewidth=2,
         label='Model 1: Circle ($z_1^2 + z_2^2 = 4$)')
plt.plot(parabola_x, parabola_y, color='#ff7f0e', linewidth=2,
         label='Model 2: Parabola ($z_2 = z_1^2 - 1$)')

# Plot initial guesses
guesses_arr = np.array(initial_guesses)
plt.scatter(guesses_arr[:, 0], guesses_arr[:, 1], color='green', marker='x',
            s=80, linewidths=2, zorder=4, label='Initial Guesses')

# Plot converged intersection points
pts = np.array(intersections)
plt.scatter(pts[:, 0], pts[:, 1], color='red', s=90, zorder=5,
            label='Converged Intersections')

for pt in pts:
    offset_x = 0.2 if pt[0] > 0 else -1.35
    plt.annotate(f'({pt[0]:.3f}, {pt[1]:.3f})',
                 xy=(pt[0], pt[1]),
                 xytext=(pt[0] + offset_x, pt[1] + 0.35),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, fontweight='bold', color='red')

plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
plt.xlim(-3.0, 3.0)
plt.ylim(-2.5, 3.5)
plt.gca().set_aspect('equal', adjustable='box')  # Keep circle geometrically true
plt.grid(True, linestyle=':', alpha=0.7)
plt.title('Nonlinear System Solution: 2D ML Decision Boundary Intersections')
plt.xlabel('Latent Feature $z_1$')
plt.ylabel('Latent Feature $z_2$')
plt.legend(loc='lower right')

plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("Graph saved as 'plot.png'")
plt.show()