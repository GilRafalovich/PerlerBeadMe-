import numpy as np
import cv2
import trimesh

class VoxelQuantizer:
    def __init__(self, max_dim_xy=60, max_dim_z=4):
        """
        Handles the true 3D spatial quantization of a .obj mesh.
        Limits X/Y for detail, but strictly limits Z to max_dim_z for kid-friendly assembly.
        """
        self.max_dim_xy = max_dim_xy
        self.max_dim_z = max_dim_z

    def _hollow_out(self, matrix):
        """
        Removes internal voxels (the middle wall) to save beads and simplify assembly.
        A voxel is removed if it is completely surrounded in 3D space.
        """
        hollowed = matrix.copy()
        for x in range(1, matrix.shape[0]-1):
            for y in range(1, matrix.shape[1]-1):
                for z in range(1, matrix.shape[2]-1):
                    if matrix[x, y, z] == 1:
                        if (matrix[x-1, y, z] == 1 and matrix[x+1, y, z] == 1 and
                            matrix[x, y-1, z] == 1 and matrix[x, y+1, z] == 1 and
                            matrix[x, y, z-1] == 1 and matrix[x, y, z+1] == 1):
                            hollowed[x, y, z] = 0
        return hollowed

    def spatial_quantize(self, img_rgb: np.ndarray, obj_path: str):
        """
        Voxelizes a true 3D .obj mesh to max_dim_xy x max_dim_xy x max_dim_z.
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
        
        # 3. Squash Z-axis into max_dim_z layers (e.g., 4 layers)
        chunk_size = matrix.shape[2] / self.max_dim_z
        z_squashed = np.zeros((matrix.shape[0], matrix.shape[1], self.max_dim_z), dtype=np.uint8)
        
        for z in range(self.max_dim_z):
            start = int(z * chunk_size)
            end = int((z + 1) * chunk_size) if z < self.max_dim_z - 1 else matrix.shape[2]
            if start < end:
                z_squashed[:, :, z] = np.max(matrix[:, :, start:end], axis=2)
                
        # 4. Create Final Padded Matrix (max_dim_xy, max_dim_xy, max_dim_z)
        voxel_matrix = np.zeros((self.max_dim_xy, self.max_dim_xy, self.max_dim_z), dtype=np.uint8)
        x_len = min(z_squashed.shape[0], self.max_dim_xy)
        y_len = min(z_squashed.shape[1], self.max_dim_xy)
        
        x_start = (self.max_dim_xy - x_len) // 2
        y_start = (self.max_dim_xy - y_len) // 2
        
        voxel_matrix[x_start:x_start+x_len, y_start:y_start+y_len, :] = z_squashed[:x_len, :y_len, :]

        # 5. Hollow out the middle to simplify assembly and reduce beads
        voxel_matrix = self._hollow_out(voxel_matrix)

        return padded_img, None, voxel_matrix

if __name__ == "__main__":
    print("Testing Spatial Quantizer...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_depth = np.random.rand(100, 100)
    
    quantizer = VoxelQuantizer(max_dim_xy=60, max_dim_z=4)
    # Note: Requires a valid .obj file for actual spatial_quantize testing
    print("Successfully initialized Voxel Quantizer!")
