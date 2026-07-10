import numpy as np
import cv2
import trimesh

class VoxelQuantizer:
    def __init__(self, max_dim_xy=60):
        """
        Handles the true 3D spatial quantization of a .obj mesh.
        Limits dimensions to max_dim_xy (cubic) to maintain kid-friendly assembly.
        """
        self.max_dim_xy = max_dim_xy

    def _generate_slotted_planes(self, matrix):
        """
        Decomposes the solid 3D matrix into two intersecting perpendicular 2D planes.
        Calculates an exact 1-bead wide slot at their intersection so they can be friction-fitted together.
        """
        slotted = np.zeros_like(matrix)
        
        mid_x = matrix.shape[0] // 2
        mid_y = matrix.shape[1] // 2
        mid_z = matrix.shape[2] // 2
        
        # Extract Plane 1 (YZ plane at X=mid_x) and Plane 2 (XZ plane at Y=mid_y)
        slotted[mid_x, :, :] = matrix[mid_x, :, :]
        slotted[:, mid_y, :] = matrix[:, mid_y, :]
        
        # Create the 1-bead wide interlocking slots at the intersection line (X=mid_x, Y=mid_y)
        # For Plane 1, slot from bottom to middle
        slotted[mid_x, mid_y, :mid_z] = 0
        
        # For Plane 2, slot from middle to top
        slotted[mid_x, mid_y, mid_z:] = 0
        
        return slotted

    def spatial_quantize(self, img_rgb: np.ndarray, obj_path: str):
        """
        Voxelizes a true 3D .obj mesh to max_dim_xy cubic, then extracts interlocking slotted planes.
        """
        h, w = img_rgb.shape[:2]
        
        # 1. Spatial Downsampling of 2D Image (X, Y) for Color Mapping
        scale = self.max_dim_xy / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        quantized_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to exactly max_dim_xy x max_dim_xy
        padded_img = np.zeros((self.max_dim_xy, self.max_dim_xy, 3), dtype=np.uint8)
        y_start = (self.max_dim_xy - new_h) // 2
        x_start = (self.max_dim_xy - new_w) // 2
        padded_img[y_start:y_start+new_h, x_start:x_start+new_w] = quantized_img

        # 2. Voxelize the 3D Mesh
        mesh = trimesh.load(obj_path, force='mesh')
        max_extent = np.max(mesh.extents)
        if max_extent == 0:
             max_extent = 1.0 # fallback
        pitch = max_extent / self.max_dim_xy
        
        voxel_grid = mesh.voxelized(pitch=pitch).fill()
        matrix = voxel_grid.matrix
        
        # 3. Create Final Padded Cubic Matrix (max_dim_xy, max_dim_xy, max_dim_xy)
        voxel_matrix = np.zeros((self.max_dim_xy, self.max_dim_xy, self.max_dim_xy), dtype=np.uint8)
        x_len = min(matrix.shape[0], self.max_dim_xy)
        y_len = min(matrix.shape[1], self.max_dim_xy)
        z_len = min(matrix.shape[2], self.max_dim_xy)
        
        x_start = (self.max_dim_xy - x_len) // 2
        y_start = (self.max_dim_xy - y_len) // 2
        z_start = (self.max_dim_xy - z_len) // 2
        
        voxel_matrix[x_start:x_start+x_len, y_start:y_start+y_len, z_start:z_start+z_len] = matrix[:x_len, :y_len, :z_len]

        # 4. Generate Friction-Fit Slotted Planes
        slotted_matrix = self._generate_slotted_planes(voxel_matrix)

        return padded_img, None, slotted_matrix

if __name__ == "__main__":
    print("Testing Spatial Quantizer...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_depth = np.random.rand(100, 100)
    
    quantizer = VoxelQuantizer(max_dim_xy=60)
    # Note: Requires a valid .obj file for actual spatial_quantize testing
    print("Successfully initialized Slotted Voxel Quantizer!")
