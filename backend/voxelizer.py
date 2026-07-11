import os
import shutil
from gradio_client import Client, handle_file

class HuggingFace3DEstimator:
    """
    Communicates with the official free HuggingFace TripoSR space
    to completely automate the 3D generation process.
    """
    def __init__(self):
        print("Connecting to HuggingFace TripoSR GPU Server...")
        self.client = Client("stabilityai/TripoSR")
        
    def generate_3d_mesh(self, image_path: str, output_obj_path: str):
        """
        Sends the 2D image to HuggingFace, runs background removal,
        generates the 3D mesh, and downloads the .obj file.
        """
        print("1. Removing Background (HuggingFace)...")
        # The /preprocess endpoint takes: input_image, remove_background, foreground_ratio
        processed_image_path = self.client.predict(
            handle_file(image_path),
            True,
            0.85,
            api_name="/preprocess"
        )
        
        print("2. Generating 3D Mesh (HuggingFace GPUs)...")
        # The /generate endpoint takes: processed_image, marching_cubes_resolution
        result = self.client.predict(
            handle_file(processed_image_path),
            256,
            api_name="/generate"
        )
        
        # Result is a tuple: (obj_path, glb_path)
        hf_obj_path = result[0]
        
        print("3. Downloading 3D Mesh...")
        shutil.copy2(hf_obj_path, output_obj_path)
        print(f"Successfully retrieved 3D mesh: {output_obj_path}")
        return output_obj_path
