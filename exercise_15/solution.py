import numpy as np
from scipy.integrate import simpson, cumulative_trapezoid
import matplotlib.pyplot as plt

# 1. Input Data: 12-Hour GPU Training Cluster Power Telemetry (sampled every 2 hours)
time_hours = np.array([0, 2, 4, 6, 8, 10, 12], dtype=float)
power_kw = np.array([0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9])

# 2. Numerical Integration (Total Energy in kWh)
energy_trapezoid = np.trapezoid(power_kw, time_hours)
energy_simpson = simpson(power_kw, x=time_hours)
diff = abs(energy_simpson - energy_trapezoid)

# Estimate carbon footprint (assuming 0.40 kg CO2e per kWh)
carbon_intensity = 0.40
co2_kg = energy_simpson * carbon_intensity

# Cumulative energy profile over time
cum_energy = cumulative_trapezoid(power_kw, time_hours, initial=0)

print("--- Green AI: GPU Training Energy Integration ---")
print(f"Energy using Trapezoidal rule: {energy_trapezoid:.3f} kWh")
print(f"Energy using Simpson's rule:   {energy_simpson:.3f} kWh")
print(f"Difference between estimates:  {diff:.3f} kWh")
print(f"Estimated Carbon Footprint:    {co2_kg:.2f} kg CO2e (using Simpson's)")

# 3. Visualize Power Curve Area & Cumulative Energy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Power Draw with Shaded Trapezoidal Segments
ax1.plot(time_hours, power_kw, 'b-o', linewidth=2, markersize=7, label='GPU Power Draw (kW)')
ax1.fill_between(time_hours, power_kw, color='#1f77b4', alpha=0.2,
                 label=f'Integrated Area ({energy_trapezoid:.2f} kWh Trapz)')

# Draw vertical edges of each trapezoid
for t, p in zip(time_hours, power_kw):
    ax1.vlines(t, 0, p, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)

ax1.set_xlim(0, 12)
ax1.set_ylim(0, 2.2)
ax1.set_xlabel("Training Time (hours)")
ax1.set_ylabel("Power Draw (kW)")
ax1.set_title("GPU Cluster Power Telemetry (Area = Energy)")
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper right')

# Right Panel: Cumulative Energy Consumed Over Time
ax2.plot(time_hours, cum_energy, 'g-s', linewidth=2, markersize=6, label='Cumulative Energy (kWh)')
ax2.scatter([time_hours[-1]], [energy_simpson], color='red', marker='D', s=70, zorder=5,
            label=f"Simpson's Final Total ({energy_simpson:.2f} kWh)")
ax2.set_xlim(0, 12.5)
ax2.set_ylim(0, 18)
ax2.set_xlabel("Training Time (hours)")
ax2.set_ylabel("Cumulative Energy (kWh)")
ax2.set_title("Accumulated Energy Consumption Across Training Run")
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()