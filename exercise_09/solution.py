import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# 1. Input Model: ML Fraud-Detection Cost-Balance Function
# f(tau) = 0 represents the equilibrium decision threshold where
# false-positive friction balances exponentially decaying fraud risk.
def cost_balance(tau):
    return tau - 0.25 - 0.10 * np.exp(-tau / 0.20)

# 2. Method 1: Bisection Method (bracketing a sign change on [0.20, 0.40])
bisect_sol = root_scalar(
    cost_balance,
    bracket=[0.20, 0.40],
    method="bisect",
    xtol=1e-10
)

# 3. Method 2: Fixed-Point Iteration
# Rewrite f(tau) = 0 as tau = g(tau) = 0.25 + 0.10 * exp(-tau / 0.20)
def g(tau):
    return 0.25 + 0.10 * np.exp(-tau / 0.20)

tau = 0.30  # Initial guess
history = [tau]
step_errors = []

for _ in range(100):
    tau_new = g(tau)
    err = abs(tau_new - tau)
    history.append(tau_new)
    step_errors.append(err)
    if err < 1e-10:
        break
    tau = tau_new

print("--- ML Decision Threshold Calibration (Root Finding) ---")
print(f"Bisection equilibrium threshold:   {bisect_sol.root:.6f} ({bisect_sol.iterations} iterations)")
print(f"Fixed-point equilibrium threshold: {history[-1]:.6f} ({len(history) - 1} iterations)")
print(f"Function value at root f(tau*):    {cost_balance(bisect_sol.root):.2e}")

# 4. Visualize the Root & Fixed-Point Convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Cost-Balance Function and Zero-Crossing
taus = np.linspace(0.20, 0.40, 300)
ax1.plot(taus, cost_balance(taus), color='#1f77b4', linewidth=2, label='$f(\\tau) = \\tau - 0.25 - 0.10e^{-\\tau/0.20}$')
ax1.axhline(0, color='black', linewidth=0.9)
ax1.axvline(bisect_sol.root, color='red', linestyle='--', linewidth=1.5,
            label=f'Root $\\tau^* = {bisect_sol.root:.4f}$')
ax1.scatter([bisect_sol.root], [0], color='red', s=70, zorder=5)
ax1.set_xlabel("Decision Threshold ($\\tau$)")
ax1.set_ylabel("Net Cost Imbalance $f(\\tau)$")
ax1.set_title("Equilibrium ML Decision Threshold")
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Right Panel: Fixed-Point Iteration Convergence (Semi-Log)
iterations = np.arange(1, len(step_errors) + 1)
ax2.semilogy(iterations, step_errors, 'g-o', linewidth=2, label='$|\\tau_{n+1} - \\tau_n|$')
ax2.axhline(1e-10, color='gray', linestyle='--', linewidth=1, label='Tolerance (1e-10)')
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Step Change (Log Scale)")
ax2.set_title("Fixed-Point Iteration Convergence")
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()