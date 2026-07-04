import numpy as np
import cv2

# Standard Perler Bead Palette (Subset for Proof of Concept)
PERLER_PALETTE = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Red": (219, 66, 62),
    "Blue": (40, 93, 163),
    "Green": (78, 153, 91),
    "Yellow": (235, 203, 67),
    "Orange": (224, 122, 53),
    "Purple": (102, 63, 133),
    "Brown": (97, 61, 48),
    "Grey": (145, 148, 156)
}

class ColorQuantizer:
    def __init__(self):
        # Convert dictionary to numpy array for fast distance calculation
        self.color_names = list(PERLER_PALETTE.keys())
        self.palette_rgb = np.array(list(PERLER_PALETTE.values()), dtype=np.uint8)
        
        # Convert palette to LAB color space for perceptual accuracy
        # cv2 requires shape (1, N, 3) for cvtColor
        palette_rgb_reshaped = self.palette_rgb.reshape(1, -1, 3)
        self.palette_lab = cv2.cvtColor(palette_rgb_reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    def quantize(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Maps every pixel in the RGB image to the closest Perler bead color
        using Euclidean distance in the LAB color space.
        """
        h, w = img_rgb.shape[:2]
        
        # Convert input image to LAB
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        
        # Flatten image
        pixels_lab = img_lab.reshape(-1, 3).astype(np.float32)
        palette_lab_f = self.palette_lab.astype(np.float32)
        
        # Calculate distances to all palette colors for all pixels using broadcasting
        diff = pixels_lab[:, np.newaxis, :] - palette_lab_f[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        
        # Find the index of the closest color
        closest_indices = np.argmin(distances, axis=1)
        
        # Map indices back to RGB palette
        quantized_pixels_rgb = self.palette_rgb[closest_indices]
        
        # Reshape back to image dimensions
        quantized_img = quantized_pixels_rgb.reshape(h, w, 3)
        
        return quantized_img

if __name__ == "__main__":
    print("Testing Color Quantizer...")
    dummy_img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    quantizer = ColorQuantizer()
    q_img = quantizer.quantize(dummy_img)
    print(f"Quantized Image Shape: {q_img.shape}")
    print("Successfully initialized Color Quantizer!")
