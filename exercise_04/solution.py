import numpy as np
import matplotlib.pyplot as plt

# 1. Input Data: 4 corner coordinates [x, y] of an object's bounding box
bbox_points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

# 2. Define Data Augmentation Parameters (45-degree rotation + scaling)
angle = np.deg2rad(45)
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])

scale_matrix = np.array([
    [2.0, 0.0],  # Scale x-axis by 2.0
    [0.0, 1.5]   # Scale y-axis by 1.5
])

# 3. Combine Transformations (Matrix Multiplication)
transformation = rotation_matrix @ scale_matrix
augmented_bbox = bbox_points @ transformation.T

# 4. Check Determinant (Area scaling factor & invertibility check)
det = np.linalg.det(transformation)

print("--- Computer Vision: Bounding Box Augmentation ---")
print("Original Bounding Box Corners:\n", bbox_points)
print("\nComposite Transformation Matrix:\n", np.round(transformation, 4))
print("\nAugmented Bounding Box Corners:\n", np.round(augmented_bbox, 4))
print(f"\nDeterminant (Area Scaling Factor): {det:.2f}")

# 5. Reverse the Transformation using the Matrix Inverse
if not np.isclose(det, 0.0):
    inv_transformation = np.linalg.inv(transformation)
    recovered_bbox = augmented_bbox @ inv_transformation.T
    print("\nRecovered Original Corners (via Matrix Inverse):\n", np.round(recovered_bbox, 4))
else:
    print("\nMatrix is singular (Determinant is 0); transformation cannot be inverted.")

# 6. Visualize Original vs. Augmented Bounding Box
# Append the first point to the end to close the 4-sided polygon when plotting
orig_closed = np.vstack([bbox_points, bbox_points[0]])
aug_closed = np.vstack([augmented_bbox, augmented_bbox[0]])

plt.figure(figsize=(7, 7))
plt.plot(orig_closed[:, 0], orig_closed[:, 1], 'b-o', linewidth=2, label='Original Bounding Box (Area = 1.0)')
plt.fill(orig_closed[:, 0], orig_closed[:, 1], color='blue', alpha=0.15)

plt.plot(aug_closed[:, 0], aug_closed[:, 1], 'r-o', linewidth=2, label=f'Augmented Bounding Box (Area = {det:.1f})')
plt.fill(aug_closed[:, 0], aug_closed[:, 1], color='red', alpha=0.15)

# Formatting for geometric accuracy
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis('equal')  # Prevents visual distortion of angles and lengths
plt.title('2D Bounding Box Data Augmentation (Scale + 45° Rotation)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend(loc='upper left')

# Save the figure directly for your GitHub README
plt.savefig('bbox_transformation.png', dpi=150, bbox_inches='tight')
print("\nGraph saved as 'bbox_transformation.png'")
plt.show()