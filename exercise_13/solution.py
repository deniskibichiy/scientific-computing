import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 1. Input Data: ML Inference API Load-Test Telemetry
# Concurrent API requests vs. measured inference latency (ms)
requests = np.array([50, 100, 150, 200, 250, 300, 350], dtype=float)
latency_ms = np.array([82, 95, 115, 145, 185, 238, 305], dtype=float)

# 2. Define Quadratic Scaling Model for Queuing Delay
def model(x, a, b, c):
    return a * x**2 + b * x + c

# 3. Fit Model Parameters via Least Squares
parameters, covariance = curve_fit(model, requests, latency_ms)
a, b, c = parameters

# Evaluate predictions, residuals, RMSE, and R-squared
predicted = model(requests, a, b, c)
residuals = latency_ms - predicted
rmse = np.sqrt(np.mean(residuals**2))
ss_res = np.sum(residuals**2)
ss_tot = np.sum((latency_ms - np.mean(latency_ms))**2)
r_squared = 1 - (ss_res / ss_tot)

# Forecast latency at 400 concurrent requests
forecast_load = 400.0
forecast_latency = model(forecast_load, a, b, c)

print("--- ML Inference API Latency Scaling (Curve Fitting) ---")
print(f"Fitted model: y = {a:.6f}x^2 + ({b:.4f})x + {c:.2f}")
print(f"RMSE:         {rmse:.2f} ms")
print(f"R-squared:    {r_squared:.5f}")
print(f"Predicted latency at {forecast_load:.0f} requests: {forecast_latency:.2f} ms")

# 4. Visualize Fitted Curve & Residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left Panel: Observed Data, Fitted Curve, and 400-User Forecast
x_plot = np.linspace(requests.min(), forecast_load, 300)
ax1.scatter(requests, latency_ms, color='black', s=60, zorder=5, label='Measured Latency')
ax1.plot(x_plot, model(x_plot, a, b, c), color='#1f77b4', linewidth=2,
         label=f'Quadratic Fit (RMSE = {rmse:.2f} ms)')
ax1.scatter([forecast_load], [forecast_latency], color='red', marker='D', s=80, zorder=6,
            label=f'Forecast @ 400 reqs ({forecast_latency:.1f} ms)')
ax1.axhline(350, color='orange', linestyle='--', linewidth=1.2, label='SLA Limit (350 ms)')

ax1.set_xlabel("Concurrent API Requests")
ax1.set_ylabel("Inference Latency (ms)")
ax1.set_title("ML Inference Latency Curve Fitting")
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Right Panel: Residual Inspection Plot
ax2.axhline(0, color='black', linewidth=1.0, linestyle='--')
ax2.stem(requests, residuals, linefmt='#2ca02c', markerfmt='go', basefmt=' ')
ax2.set_xlabel("Concurrent API Requests")
ax2.set_ylabel("Residual: Observed - Predicted (ms)")
ax2.set_title("Residual Analysis (Model Fit Quality)")
ax2.set_ylim(-5, 5)
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'plot.png'")
plt.show()