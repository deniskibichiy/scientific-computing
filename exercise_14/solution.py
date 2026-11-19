import numpy as np
import matplotlib.pyplot as plt

# 1. Input Data: Discrete Autonomous Vehicle GPS Telemetry
time = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
position = np.array([0.0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0])

speed = np.zeros_like(position)

# Forward difference at the left boundary (t = 0)
speed[0] = (position[1] - position[0]) / (time[1] - time[0])

# Central differences for interior points (t = 1..5)
for i in range(1, len(time) - 1):
    speed[i] = (position[i + 1] - position[i - 1]) / (time[i + 1] - time[i - 1])

# Backward difference at the right boundary (t = 6)
speed[-1] = (position[-1] - position[-2]) / (time[-1] - time[-2])

# Verify against NumPy's built-in gradient function
np_speed = np.gradient(position, time)

print("--- Autonomous Vehicle Velocity Feature Extraction ---")
for t, x, v in zip(time, position, speed):
    print(f"t={t:.0f}s | position={x:5.1f} m | estimated speed={v:.2f} m/s")
print(f"\nMax diff vs np.gradient: {np.max(np.abs(speed - np_speed)):.2e} m/s")

# 2. Step-Size Error Analysis: Forward O(h) vs Central O(h^2) on f(x) = sin(x)
x0 = 1.0
true_deriv = np.cos(x0)
h_vals = np.logspace(-12, -1, 50)

fwd_err = np.abs((np.sin(x0 + h_vals) - np.sin(x0)) / h_vals - true_deriv)
cen_err = np.abs((np.sin(x0 + h_vals) - np.sin(x0 - h_vals)) / (2 * h_vals) - true_deriv)

# 3. Visualize Estimated Speed & Step-Size Error Trade-off
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Estimated Vehicle Speed
ax1.plot(time, speed, 'b-o', linewidth=2, markersize=7, label='Estimated Speed (Finite Diff)')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Estimated Speed (m/s)")
ax1.set_title("Instantaneous Speed from Discrete GPS Telemetry")
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Right Panel: Log-Log Truncation vs Round-Off Error
ax2.loglog(h_vals, fwd_err, 'r--o', markersize=4, linewidth=1.5, label='Forward Difference $O(h)$')
ax2.loglog(h_vals, cen_err, 'g-s', markersize=4, linewidth=1.5, label='Central Difference $O(h^2)$')
ax2.set_xlabel("Step Size ($h$)")
ax2.set_ylabel("Absolute Error $|f'_{approx} - f'_{true}|$")
ax2.set_title("Truncation vs. Round-Off Error (Log-Log)")
ax2.grid(True, which='both', linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()