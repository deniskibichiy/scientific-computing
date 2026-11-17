import time
import numpy as np

print("Generating high-dimensional NLP vectors...")
# Input Data: Simulated NLP document features and model weights
features_list = list(range(100000, 200000))
weights_list = list(range(200000, 300000))

# Convert to NumPy arrays for vectorized operations
np_features = np.array(features_list)
np_weights = np.array(weights_list)

print("-" * 50)
print("1. Traditional Loop Approach (The Problem)")
tic = time.process_time()
prediction_score_loop = 0.0

for i in range(len(features_list)):
    prediction_score_loop += features_list[i] * weights_list[i]
    
toc = time.process_time()
loop_time_ms = 1000 * (toc - tic)

print(f"Prediction Score = {prediction_score_loop}")
print(f"Loop computation time = {loop_time_ms:.2f} ms")


print("-" * 50)
print("2. Vectorized NumPy Approach (The Solution)")
n_tic = time.process_time()

prediction_score_numpy = np.dot(np_features, np_weights)

n_toc = time.process_time()
numpy_time_ms = 1000 * (n_toc - n_tic)

print(f"Prediction Score = {prediction_score_numpy}")
print(f"NumPy computation time = {numpy_time_ms:.2f} ms")


print("-" * 50)
speedup = loop_time_ms / numpy_time_ms if numpy_time_ms > 0 else 0
print(f"System Impact: NumPy vectorization is {speedup:.1f}x faster")