import numpy as np
import cv2

class VoxelQuantizer:
    def __init__(self, max_dim=20):
        """
        Handles the spatial quantization of the 2D image and depth map
        into the strict 20x20x20 physical constraint for Perler beads.
        """
        self.max_dim = max_dim

    def spatial_quantize(self, img_rgb: np.ndarray, depth_map: np.ndarray):
        """
        Downsamples the RGB image and Depth Map to the 20x20 grid limit.
        Returns the quantized image, quantized depth map, and the 3D voxel array.
        """
        h, w = img_rgb.shape[:2]
        
        # Calculate aspect ratio to fit inside max_dim x max_dim
        scale = self.max_dim / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # 1. Spatial Downsampling (X, Y)
        # We use INTER_AREA for downsampling as it yields the best results for shrinking
        quantized_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        quantized_depth = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Z-Axis Quantization
        # The depth map is normalized 0.0 to 1.0. 
        # We multiply by max_dim - 1 (19) to get integer bins 0 to 19.
        z_bins = np.round(quantized_depth * (self.max_dim - 1)).astype(np.int32)

        # 3. Create 3D Voxel Matrix (X, Y, Z)
        voxel_matrix = np.zeros((new_w, new_h, self.max_dim), dtype=np.uint8)
        
        # Populate the voxel matrix
        # For a standard Perler model, if depth is Z, it means beads exist from 0 up to Z.
        for x in range(new_w):
            for y in range(new_h):
                z_height = z_bins[y, x]
                # Fill beads from the base (0) up to z_height
                for z in range(z_height + 1):
                    voxel_matrix[x, y, z] = 1

        return quantized_img, z_bins, voxel_matrix

if __name__ == "__main__":
    print("Testing Spatial Quantizer...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_depth = np.random.rand(100, 100)
    
    quantizer = VoxelQuantizer(max_dim=20)
    q_img, q_depth, voxel_matrix = quantizer.spatial_quantize(dummy_img, dummy_depth)
    
    print(f"Voxel Matrix Shape: {voxel_matrix.shape}")
    print("Successfully initialized Voxel Quantizer!")
