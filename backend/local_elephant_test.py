import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\User\Downloads\Screenshot 2026-07-05 200959.png"
max_dim = 20

print("1. Loading MiDaS for Depth Estimation...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print("2. Estimating Depth...")
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()
depth_min, depth_max = depth_map.min(), depth_map.max()
if depth_max - depth_min > 0:
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
else:
    depth_map = np.zeros_like(depth_map)

print("3. Spatial Quantization (Symmetrical Extrusion)...")
h, w = img_rgb.shape[:2]
scale = max_dim / max(h, w)
new_w = max(1, int(w * scale))
new_h = max(1, int(h * scale))

quantized_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
quantized_depth = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_AREA)

voxel_matrix = np.zeros((max_dim, max_dim, max_dim), dtype=np.uint8)
center_z = max_dim // 2

# Background Removal and Extrusion
for x in range(new_w):
    for y in range(new_h):
        color = quantized_img[y, x]
        # Ignore near-white pixels
        if np.all(color > 240):
            continue
        
        thickness = int(np.round(quantized_depth[y, x] * (max_dim // 2 - 1)))
        
        # Center the model within the 60x60 grid
        x_idx = x + (max_dim - new_w) // 2
        y_idx = y + (max_dim - new_h) // 2
        
        for z in range(center_z - thickness, center_z + thickness + 1):
            voxel_matrix[x_idx, y_idx, z] = 1

print("4. Rendering 3D Visualization...")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

colors = np.empty(voxel_matrix.shape, dtype=object)
for x in range(voxel_matrix.shape[0]):
    for y in range(voxel_matrix.shape[1]):
        for z in range(voxel_matrix.shape[2]):
            if voxel_matrix[x, y, z] == 1:
                orig_x = x - (max_dim - new_w) // 2
                orig_y = y - (max_dim - new_h) // 2
                
                if 0 <= orig_x < new_w and 0 <= orig_y < new_h:
                    r, g, b = quantized_img[orig_y, orig_x] / 255.0
                    colors[x, y, z] = (r, g, b, 0.9)
                else:
                    colors[x, y, z] = (0.5, 0.5, 0.5, 0.9)

ax.voxels(voxel_matrix, facecolors=colors, edgecolor='k')
ax.set_title(f"Symmetrical Extrusion Elephant ({max_dim}x{max_dim}x{max_dim})")
ax.set_xlabel('X axis (Width)')
ax.set_ylabel('Y axis (Height/Image Z)')
ax.set_zlabel('Z axis (Depth/Image Y)')

out_file = "validation_3d_elephant_20.png"
plt.savefig(out_file, dpi=300)
print(f"Saved to {out_file}")
