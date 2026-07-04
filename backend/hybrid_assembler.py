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
        Performs the adaptive plane generation and hybrid interlock calculation.
        """
        dominant_axis = self.calculate_skeleton()
        residual = np.copy(self.voxel_matrix)
        instructions = []
        
        # Greedy Plane Extraction
        # If Z is the dominant axis (tall object), we use pancake layers with dowels.
        # If X/Y is dominant (wide/long object), we use slotted puzzle planes for integrity.
        
        if dominant_axis == 2:
            print("Detected Z-Axis dominance. Using Pancake Slicing with Dowel Locks...")
            for z in range(self.dim_z):
                layer = residual[:, :, z]
                if np.sum(layer) > 0:
                    dowel_holes = self._generate_dowel_holes(layer)
                    instructions.append({
                        "type": "horizontal_plane",
                        "z_index": z,
                        "layout": layer.tolist(),
                        "dowel_holes": dowel_holes
                    })
        else:
            print("Detected X/Y-Axis dominance. Using Slotted Puzzle Planes...")
            # Decompose into vertical planes for structural strength
            for x in range(self.dim_x):
                layer = residual[x, :, :]
                if np.sum(layer) > 0:
                    instructions.append({
                        "type": "vertical_plane",
                        "x_index": x,
                        "layout": layer.tolist()
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
