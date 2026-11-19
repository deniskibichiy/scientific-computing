import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

# 1. Input Data: Hourly Weather Telemetry (08:00 to 14:00)
hours = np.array([8, 9, 10, 11, 12, 13, 14], dtype=float)
temperature = np.array([18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0])

# 2. Build Linear and Cubic Spline Interpolators
linear_interp = interp1d(hours, temperature, kind="linear")
cubic_interp = CubicSpline(hours, temperature)

# 3. Upsample to 10-Minute Intervals (1/6 hour) for ML Feature Alignment
ten_minute_times = np.arange(8, 14 + 1e-9, 1 / 6)
linear_estimates = linear_interp(ten_minute_times)
cubic_estimates = cubic_interp(ten_minute_times)

# Evaluate at a specific target timestamp (10:30 -> 10.5)
target_time = 10.5
lin_val = float(linear_interp(target_time))
cub_val = float(cubic_interp(target_time))

print("--- ML Time-Series Upsampling (Interpolation) ---")
print(f"Original hourly samples:     {len(hours)}")
print(f"Upsampled 10-min samples:    {len(ten_minute_times)}")
print(f"Linear estimate at 10:30:    {lin_val:.2f} C")
print(f"Cubic spline at 10:30:       {cub_val:.2f} C")
print(f"Difference at 10:30:         {cub_val - lin_val:+.4f} C")
print(f"Max absolute difference:     {np.max(np.abs(cubic_estimates - linear_estimates)):.4f} C")

# 4. Visualize Upsampled Curves & Difference Profile
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Linear vs. Cubic Spline Interpolation
ax1.scatter(hours, temperature, color='black', s=60, zorder=5, label='Hourly Sensor Readings')
ax1.plot(ten_minute_times, linear_estimates, '--', color='#1f77b4', linewidth=2, label='Linear Interpolation')
ax1.plot(ten_minute_times, cubic_estimates, '-', color='#ff7f0e', linewidth=2, label='Cubic Spline')
ax1.axvline(target_time, color='red', linestyle=':', linewidth=1.2, label='Target Time (10:30)')
ax1.scatter([target_time, target_time], [lin_val, cub_val], color='red', s=40, zorder=6)

ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Upsampling Hourly Telemetry to 10-Minute Intervals")
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Right Panel: Difference Between Cubic Spline and Linear Estimates
diff = cubic_estimates - linear_estimates
ax2.plot(ten_minute_times, diff, color='#2ca02c', linewidth=2, label='Cubic - Linear')
ax2.axhline(0, color='black', linewidth=0.9)
ax2.scatter(hours, np.zeros_like(hours), color='black', s=35, zorder=5, label='Known Hourly Knots (Diff = 0)')
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Temperature Difference (°C)")
ax2.set_title("Deviation: Cubic Spline vs. Linear")
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()