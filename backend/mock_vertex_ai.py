import trimesh
import numpy as np

print("Generating mock sphere mesh...")
# Create a dummy mesh
mesh = trimesh.creation.icosphere(subdivisions=3, radius=10)

print(f"Mesh bounding box: {mesh.bounds}")

# We want a 60x60x60 voxel grid.
# The pitch is the size of one voxel in the mesh's coordinate system.
max_extent = np.max(mesh.extents)
pitch = max_extent / 60.0

print(f"Voxelizing with pitch: {pitch}")
voxel_grid = mesh.voxelized(pitch=pitch).fill()

# Get the dense boolean matrix
matrix = voxel_grid.matrix

print(f"Raw voxel matrix shape: {matrix.shape}")

# We need exactly 60x60x60. Let's pad or crop.
target_dim = 60
final_matrix = np.zeros((target_dim, target_dim, target_dim), dtype=np.uint8)

# Calculate dimensions to copy
x_len = min(matrix.shape[0], target_dim)
y_len = min(matrix.shape[1], target_dim)
z_len = min(matrix.shape[2], target_dim)

# Center it
x_start = (target_dim - x_len) // 2
y_start = (target_dim - y_len) // 2
z_start = (target_dim - z_len) // 2

final_matrix[x_start:x_start+x_len, y_start:y_start+y_len, z_start:z_start+z_len] = matrix[:x_len, :y_len, :z_len]

print(f"Final voxel matrix shape: {final_matrix.shape}")
print(f"Total beads (voxels): {np.sum(final_matrix)}")
