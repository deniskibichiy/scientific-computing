import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# 1. Input Data: AI Infrastructure Deployment Cash Flows ($)
# t=0: -$10,000 upfront GPU server & ML pipeline setup
# t=1..4: Annual operational savings / revenue inflows
cash_flows = [-10000, 3000, 3500, 4000, 2500]


def npv(rate):
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


def d_npv(rate):
    return sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows) if t > 0)


# 2. Solve with SciPy's Newton-Raphson and Secant Methods
newton = root_scalar(
    npv,
    fprime=d_npv,
    x0=0.10,
    method="newton",
    xtol=1e-12
)

secant = root_scalar(
    npv,
    x0=0.05,
    x1=0.20,
    method="secant",
    xtol=1e-12
)

print("--- AI Infrastructure ROI: IRR via Newton-Raphson & Secant ---")
print(f"IRR using Newton-Raphson: {newton.root:.6%} ({newton.iterations} iterations)")
print(f"IRR using Secant method:  {secant.root:.6%} ({secant.iterations} iterations)")
print(f"NPV at Newton solution:   ${npv(newton.root):.6f}")

# 3. Track Iteration Histories for Convergence Visualization
# Manual Newton-Raphson trace
x_n = 0.10
newton_err = []
for _ in range(20):
    x_next = x_n - npv(x_n) / d_npv(x_n)
    err = abs(x_next - x_n)
    newton_err.append(err)
    x_n = x_next
    if err < 1e-12:
        break

# Manual Secant trace
x0, x1 = 0.05, 0.20
secant_err = []
for _ in range(20):
    f0, f1 = npv(x0), npv(x1)
    x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
    err = abs(x_next - x1)
    secant_err.append(err)
    x0, x1 = x1, x_next
    if err < 1e-12:
        break

# 4. Visualize NPV Curve & Convergence Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: NPV Curve and IRR Zero-Crossing
rates = np.linspace(0.0, 0.25, 300)
npv_vals = [npv(r) for r in rates]

ax1.plot(rates * 100, npv_vals, color='#1f77b4', linewidth=2, label='NPV(r)')
ax1.axhline(0, color='black', linewidth=0.9)
ax1.axvline(newton.root * 100, color='red', linestyle='--', linewidth=1.5,
            label=f'IRR = {newton.root:.2%}')
ax1.scatter([newton.root * 100], [0], color='red', s=70, zorder=5)
ax1.set_xlabel("Discount Rate (%)")
ax1.set_ylabel("Net Present Value ($)")
ax1.set_title("ML Infrastructure NPV vs. Discount Rate")
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Right Panel: Convergence Rate Comparison (Semi-Log)
ax2.semilogy(range(1, len(newton_err) + 1), newton_err, 'r-o', linewidth=2,
             label=f'Newton-Raphson ({len(newton_err)} iters)')
ax2.semilogy(range(1, len(secant_err) + 1), secant_err, 'b-s', linewidth=2,
             label=f'Secant Method ({len(secant_err)} iters)')
ax2.axhline(1e-12, color='gray', linestyle='--', linewidth=1, label='Tolerance (1e-12)')
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Step Change $|r_{n+1} - r_n|$ (Log Scale)")
ax2.set_title("Quadratic vs. Superlinear Convergence")
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()