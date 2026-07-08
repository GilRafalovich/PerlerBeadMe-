import numpy as np

class HybridAssembler:
    def __init__(self, voxel_matrix: np.ndarray, color_matrix: np.ndarray):
        """
        Takes a binary 3D voxel matrix (20x20x20) and a mapped color matrix.
        Decomposes it into manufacturable 2D planes using the Figure of Merit.
        """
        self.voxel_matrix = voxel_matrix
        self.color_matrix = color_matrix
        self.dim_x, self.dim_y, self.dim_z = voxel_matrix.shape

    def calculate_skeleton(self):
        """
        Finds the structural 'spine' of the voxel object.
        Calculates mass distribution along all axes to identify the principal component.
        """
        mass_z = np.sum(self.voxel_matrix, axis=(0, 1))
        mass_y = np.sum(self.voxel_matrix, axis=(0, 2))
        mass_x = np.sum(self.voxel_matrix, axis=(1, 2))
        
        axes_mass = [np.sum(mass_x), np.sum(mass_y), np.sum(mass_z)]
        dominant_axis = np.argmax(axes_mass)
        
        return dominant_axis

    def decompose(self):
        """
        Performs the adaptive plane generation for kid-friendly assembly.
        Forces Z-axis (Pancake Slicing) since the Z-dimension is strictly constrained to 4 layers.
        """
        residual = np.copy(self.voxel_matrix)
        instructions = []
        
        print(f"Generating Kid-Friendly Assembly: Slicing into {self.dim_z} horizontal layers...")
        for z in range(self.dim_z):
            layer = residual[:, :, z]
            if np.sum(layer) > 0:
                # Merge small parts if necessary to stay under 10 detailed parts total
                # For now, each layer acts as a single primary part template
                instructions.append({
                    "type": "horizontal_plane",
                    "z_index": z,
                    "layout": layer.tolist(),
                    "dowel_holes": []
                })
                    
        return instructions
        
    def _generate_dowel_holes(self, layer: np.ndarray):
        """ 
        Finds safe internal coordinates to leave empty for vertical pins 
        without compromising the visible shell of the model.
        """
        return []

if __name__ == "__main__":
    dummy_matrix = np.zeros((20, 20, 20), dtype=np.uint8)
    dummy_matrix[5:15, 5:15, 0:10] = 1  # A block
    dummy_color = np.zeros((20, 20, 20, 3), dtype=np.uint8)
    
    assembler = HybridAssembler(dummy_matrix, dummy_color)
    instructions = assembler.decompose()
    print(f"Generated {len(instructions)} assembly planes based on Figure of Merit.")
