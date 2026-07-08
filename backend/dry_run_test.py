import os
import json
import numpy as np
import matplotlib.pyplot as plt

from voxelizer import VertexAI3DEstimator
from quantizer import VoxelQuantizer
from color_quantizer import ColorQuantizer
from hybrid_assembler import HybridAssembler

import cv2

image_path = r"C:\Users\User\Downloads\Screenshot 2026-07-05 200959.png"

print("Initializing AI Pipeline...")
depth_estimator = VertexAI3DEstimator()
voxel_quantizer = VoxelQuantizer(max_dim_xy=60, max_dim_z=4)
color_quantizer = ColorQuantizer()
print("Pipeline Ready.")

print(f"Loading image from: {image_path}")

print("1. Generative 3D Mesh via Vertex AI")
obj_path = "validation_test.obj"
depth_estimator.generate_3d_mesh(image_path, obj_path)

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("2. LAB Color Quantization")
quantized_color_img = color_quantizer.quantize(img_rgb)

print("3. Mesh Voxelization (Scaling to 60x60x60)")
q_img, _, voxel_matrix = voxel_quantizer.spatial_quantize(quantized_color_img, obj_path)

print("Saving 2D Quantized Image for review...")
# q_img is RGB because img_rgb was converted to RGB initially. Let's save it.
# We will use matplotlib to save it without axes to see the exact pixels.
plt.imsave("quantized_2d_elephant.png", q_img.astype(np.uint8))
print("Saved 2D quantized picture to quantized_2d_elephant.png")

print("4. Assembly-Optimized Hybrid Decomposition")
assembler = HybridAssembler(voxel_matrix, q_img)
instructions = assembler.decompose()

print(f"Voxel shape: {voxel_matrix.shape}")
print(f"Total layers generated: {len(instructions)}")

print("5. Rendering 3D visualization...")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

colors = np.empty(voxel_matrix.shape, dtype=object)
for x in range(voxel_matrix.shape[0]):
    for y in range(voxel_matrix.shape[1]):
        for z in range(voxel_matrix.shape[2]):
            if voxel_matrix[x, y, z] == 1:
                # Color is from q_img[y, x] mapping (height, width, 3)
                r, g, b = q_img[y, x] / 255.0
                colors[x, y, z] = (r, g, b, 0.9)

ax.voxels(voxel_matrix, facecolors=colors, edgecolor='k')
ax.set_title("Generated 3D Perler Bead Model")
ax.set_xlabel('X axis (Width)')
ax.set_ylabel('Y axis (Height/Image Z)')
ax.set_zlabel('Z axis (Depth/Image Y)')

plt.savefig("validation_3d_model.png", dpi=300)
print("Saved 3D validation picture to validation_3d_model.png")

# Save outputs
np.save("voxel_matrix.npy", voxel_matrix)
with open("instructions.json", "w") as f:
    json.dump(instructions, f, indent=2)

print("Results saved to voxel_matrix.npy and instructions.json")
