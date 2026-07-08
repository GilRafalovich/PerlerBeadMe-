import os
import torch
import cv2
import numpy as np
import requests

class VertexAI3DEstimator:
    """
    Communicates with the Google Cloud Vertex AI Custom Endpoint
    running the generative 3D model (TripoSR).
    """
    def __init__(self, endpoint_url=None):
        self.endpoint_url = endpoint_url or os.environ.get("VERTEX_AI_ENDPOINT_URL", "")
        
    def generate_3d_mesh(self, image_path: str, output_obj_path: str):
        """
        Sends the 2D image to the Vertex AI Endpoint.
        Downloads the generated .obj file.
        """
        if not self.endpoint_url:
            print("VERTEX_AI_ENDPOINT_URL not set. Using local mock generation for testing...")
            # Local Mock: Generate a simple 3D mesh for testing without GCP deployment
            import trimesh
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=10)
            mesh.export(output_obj_path)
            return output_obj_path
            
        print(f"Sending image to Vertex AI Endpoint: {self.endpoint_url}")
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            response = requests.post(self.endpoint_url, files=files)
            
        if response.status_code == 200:
            with open(output_obj_path, "wb") as f:
                f.write(response.content)
            print(f"Successfully retrieved 3D mesh: {output_obj_path}")
            return output_obj_path
        else:
            raise Exception(f"Vertex AI Endpoint Failed: {response.status_code} - {response.text}")
