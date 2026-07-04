import torch
import cv2
import numpy as np
import urllib.request
import os

class DepthEstimator:
    """
    Implements AI-Driven Monocular Depth Estimation using MiDaS.
    This takes a standard 2D image and infers the Z-axis (depth).
    """
    def __init__(self):
        print("Loading AI Depth Estimation Model (MiDaS)...")
        # We use MiDaS_small for token-optimized, fast, lightweight CPU/GPU inference.
        self.model_type = "MiDaS_small"
        
        # Load the pre-trained model from PyTorch Hub
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        
        # Auto-detect if CUDA (GPU) is available, otherwise fallback to CPU
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load the corresponding image transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def generate_depth_map(self, image_path: str) -> np.ndarray:
        """
        Analyzes the 2D image and outputs a normalized depth map array.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # OpenCV reads in BGR format, so we convert to RGB for the AI model
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transforms
        input_batch = self.transform(img_rgb).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize the prediction to match the original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Move back to CPU and convert to numpy array
        depth_map = prediction.cpu().numpy()
        
        # Normalize the depth map to a 0.0 - 1.0 range so we can quantize it later
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        
        return normalized_depth, img_rgb

if __name__ == "__main__":
    print("Testing Depth Estimator Initialization...")
    try:
        estimator = DepthEstimator()
        print("Successfully initialized Monocular Depth Estimator!")
    except Exception as e:
        print(f"Failed to initialize: {e}")
