import numpy as np

print("--- 1. ML Model Evaluation: The Equality Trap ---")
# Simulating batch loss calculations
batch_1_loss = 0.1
batch_2_loss = 0.2
calculated_total_loss = batch_1_loss + batch_2_loss
target_loss = 0.3

print(f"Calculated Loss (0.1 + 0.2) = {calculated_total_loss}")
print(f"Target Loss = {target_loss}")
print(f"Naive Exact comparison (calculated == target): {calculated_total_loss == target_loss}")
print(f"Tolerance-based comparison (np.isclose): {np.isclose(calculated_total_loss, target_loss)}")


print("\n--- 2. Measuring Approximation Error ---")
# True theoretical gradient vs a truncated approximation
true_gradient = np.sqrt(2)
approx_gradient = 1.414

absolute_error = abs(true_gradient - approx_gradient)
relative_error = absolute_error / abs(true_gradient)

print(f"True Value: {true_gradient:.8f}, Approximated: {approx_gradient}")
print(f"Absolute error: {absolute_error:.8f}")
print(f"Relative error: {relative_error:.8%}")


print("\n--- 3. Numerical Gradients: Unstable vs Stable Formulation ---")
print("Testing limit as perturbation (x) approaches 0 for function: f(x) = (sqrt(1+x)-1)/x")

# Testing progressively smaller perturbations
perturbations = [1e-4, 1e-8, 1e-12, 1e-15]

for x in perturbations:
    # The direct formula suffers catastrophic cancellation because sqrt(1+x) approaches 1
    # 1 - 1 = 0, wiping out the precision bits.
    direct = (np.sqrt(1 + x) - 1) / x
    
    # The stable formula is algebraically identical (multiplied by conjugate)
    # but avoids subtracting nearly equal numbers.
    stable = 1 / (np.sqrt(1 + x) + 1)
    
    print(f"x={x:.0e}: direct={direct:.12f}, stable={stable:.12f}")