import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from voxelizer import HuggingFace3DEstimator
from color_quantizer import ColorQuantizer
from quantizer import VoxelQuantizer
from hybrid_assembler import HybridAssembler
import trimesh

REPORT_DIR = "report_images"
os.makedirs(REPORT_DIR, exist_ok=True)

IMAGE_PATH = r"C:\Users\User\.gemini\antigravity-ide\brain\28ae8589-cd53-4fac-8067-b73e9520a10a\cow_test_image_1783723645573.png"
OBJ_PATH = "cow_mesh.obj"

def run_diagnostic():
    print("--- PerlerBeadMe Diagnostic Pipeline ---")
    
    # 0. Load Original Image
    print("[Step 0] Loading Original Image...")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise Exception(f"Failed to load image: {IMAGE_PATH}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.title("Original Input Image")
    plt.axis("off")
    plt.savefig(os.path.join(REPORT_DIR, "0_original.png"), dpi=150)
    plt.close()
    
    # 1. HuggingFace 3D Estimator
    print("[Step 1] Running Generative 3D Mesh (HuggingFace)...")
    estimator = HuggingFace3DEstimator()
    # To save time if it already exists during testing
    if not os.path.exists(OBJ_PATH):
        estimator.generate_3d_mesh(IMAGE_PATH, OBJ_PATH)
    else:
        print("Using cached 3D Mesh...")
        
    # Plot the 3D Mesh
    mesh = trimesh.load(OBJ_PATH, force='mesh')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2], s=1, c='gray')
    ax.set_title("Raw 3D Mesh Output")
    plt.savefig(os.path.join(REPORT_DIR, "1_raw_mesh.png"), dpi=150)
    plt.close()

    # 2. Color Quantizer
    print("[Step 2] Running Color Quantization...")
    c_quantizer = ColorQuantizer()
    q_color_img = c_quantizer.quantize(img_rgb)
    
    plt.imshow(q_color_img)
    plt.title("Quantized Perler Colors")
    plt.axis("off")
    plt.savefig(os.path.join(REPORT_DIR, "2_quantized_color.png"), dpi=150)
    plt.close()
    
    # 3. Spatial Voxel Quantizer
    print("[Step 3] Running Spatial Voxelization...")
    v_quantizer = VoxelQuantizer(max_dim_xy=60, max_dim_z=4)
    # The quantizer expects quantized_color_img and obj_path
    # returns q_img, None, voxel_matrix
    final_q_img, _, voxel_matrix = v_quantizer.spatial_quantize(q_color_img, OBJ_PATH)
    
    # Plot 3D Voxels
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract colors for voxels
    colors = np.empty(voxel_matrix.shape, dtype=object)
    
    h, w = final_q_img.shape[:2]
    max_dim = voxel_matrix.shape[0]
    
    for x in range(max_dim):
        for y in range(max_dim):
            for z in range(voxel_matrix.shape[2]):
                if voxel_matrix[x, y, z] == 1:
                    # Map to final_q_img coords
                    orig_x = x - (max_dim - w) // 2
                    orig_y = y - (max_dim - h) // 2
                    if 0 <= orig_x < w and 0 <= orig_y < h:
                        colors[x, y, z] = tuple(final_q_img[orig_y, orig_x] / 255.0) + (1.0,)
                    else:
                        colors[x, y, z] = (0.5, 0.5, 0.5, 1.0)
                        
    ax.voxels(voxel_matrix, facecolors=colors, edgecolor='k')
    ax.set_title("Voxelized 3D Matrix (max_z=4)")
    plt.savefig(os.path.join(REPORT_DIR, "3_voxels.png"), dpi=300)
    plt.close()
    
    # 4. Hybrid Assembler (Decomposition)
    print("[Step 4] Running Assembly Decomposition...")
    assembler = HybridAssembler(voxel_matrix, final_q_img)
    instructions = assembler.decompose()
    
    # Plot each layer instruction
    fig, axes = plt.subplots(1, len(instructions), figsize=(15, 5))
    if len(instructions) == 1:
        axes = [axes]
    
    for idx, inst in enumerate(instructions):
        layer_arr = np.array(inst["layout"])
        layer_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8) + 255 # white bg
        for i in range(max_dim):
            for j in range(max_dim):
                if layer_arr[i, j] == 1:
                    orig_x = i - (max_dim - w) // 2
                    orig_y = j - (max_dim - h) // 2
                    if 0 <= orig_x < w and 0 <= orig_y < h:
                        layer_img[i, j] = final_q_img[orig_y, orig_x]
                    else:
                        layer_img[i, j] = [100, 100, 100]
        
        axes[idx].imshow(layer_img)
        axes[idx].set_title(f"Layer {idx+1}")
        axes[idx].axis("off")
        
    plt.suptitle("Generated Bead Assembly Instructions")
    plt.savefig(os.path.join(REPORT_DIR, "4_assembly_layers.png"), dpi=300)
    plt.close()
    
    print("Diagnostics Complete. Check 'report_images' folder.")

if __name__ == "__main__":
    run_diagnostic()
