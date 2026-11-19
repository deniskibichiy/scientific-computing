import numpy as np
import matplotlib.pyplot as plt

# 1. Input Parameters: GPU Thermal Cooldown (Newton's Law of Cooling)
# dT/dt = -k * (T - T_ambient)
T_ambient = 25.0   # Datacenter ambient temperature (°C)
T_initial = 90.0   # Peak GPU temperature at end of training (°C)
k = 0.12           # Cooling constant (1/min)
end_time = 30.0    # Simulation duration (minutes)


def cooling_rate(T):
    return -k * (T - T_ambient)


def exact_solution(t):
    return T_ambient + (T_initial - T_ambient) * np.exp(-k * t)


def euler_solve(h):
    time = np.arange(0, end_time + 1e-9, h)
    temp = np.zeros_like(time)
    temp[0] = T_initial
    for i in range(len(time) - 1):
        temp[i + 1] = temp[i] + h * cooling_rate(temp[i])
    return time, temp


# 2. Run Euler's Method for h = 1.0 min (baseline) and h = 0.25 min (refined)
t_coarse, T_coarse = euler_solve(h=1.0)
t_fine, T_fine = euler_solve(h=0.25)

T_exact_coarse = exact_solution(t_coarse)
T_exact_fine = exact_solution(t_fine)

err_coarse = np.abs(T_coarse - T_exact_coarse)
err_fine = np.abs(T_fine - T_exact_fine)

print("--- MLOps GPU Thermal Cooldown (Euler ODE Solver) ---")
print(f"Exact temperature after 30 min:         {T_exact_coarse[-1]:.2f} C")
print(f"Euler (h=1.00 min) after 30 min:        {T_coarse[-1]:.2f} C  | Final Error: {err_coarse[-1]:.4f} C")
print(f"Euler (h=0.25 min) after 30 min:        {T_fine[-1]:.2f} C  | Final Error: {err_fine[-1]:.4f} C")
print(f"Max absolute error (h=1.00 min):        {np.max(err_coarse):.4f} C")
print(f"Max absolute error (h=0.25 min):        {np.max(err_fine):.4f} C")

# 3. Visualize Cooldown Curves & Absolute Error Over Time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Euler vs Exact Solution
t_smooth = np.linspace(0, end_time, 300)
ax1.plot(t_smooth, exact_solution(t_smooth), 'k-', linewidth=2, label='Exact Analytical Solution')
ax1.plot(t_coarse, T_coarse, 'r--o', markersize=4, linewidth=1.5, label='Euler ($h = 1.0$ min)')
ax1.plot(t_fine, T_fine, 'b:', linewidth=2, label='Euler ($h = 0.25$ min)')
ax1.axhline(T_ambient, color='gray', linestyle='-.', linewidth=1, label='Ambient (25 °C)')

ax1.set_xlabel("Time (minutes)")
ax1.set_ylabel("GPU Temperature (°C)")
ax1.set_title("GPU Cooldown Simulation: Euler vs. Exact")
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Right Panel: Absolute Error Over Time
ax2.plot(t_coarse, err_coarse, 'r-o', markersize=4, linewidth=1.5, label='Error ($h = 1.0$ min)')
ax2.plot(t_fine, err_fine, 'b-', linewidth=2, label='Error ($h = 0.25$ min)')
ax2.set_xlabel("Time (minutes)")
ax2.set_ylabel("Absolute Error $|T_{Euler} - T_{exact}|$ (°C)")
ax2.set_title("Global Truncation Error vs. Step Size")
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()