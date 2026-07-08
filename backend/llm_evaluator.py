import os
import json
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image

def render_voxel_matrix_to_image(voxel_matrix, output_path):
    """
    Takes a 3D boolean numpy array and plots it as a 3D scatter plot.
    Saves the visualization to output_path.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of all solid voxels
    z, x, y = np.nonzero(voxel_matrix)
    
    # Plot as a 3D scatter
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='s', s=40, alpha=0.8)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Generated Voxel Matrix')
    
    # Hide axes for cleaner image
    ax.set_axis_off()
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def evaluate_with_gemini(original_image_path, voxel_matrix):
    """
    Uses Gemini to evaluate the accuracy of the generated 3D shape against the original 2D image.
    Returns (score_1_to_10, critique).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment. Skipping LLM verification.")
        return None, None
        
    genai.configure(api_key=api_key)
    
    try:
        # 1. Render the voxel matrix
        temp_render_path = "temp_voxel_render.png"
        render_voxel_matrix_to_image(voxel_matrix, temp_render_path)
        
        # 2. Load images for Gemini
        img1 = Image.open(original_image_path)
        img2 = Image.open(temp_render_path)
        
        # 3. Create Model
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        prompt = """
        You are an expert 3D modeler and Quality Assurance agent.
        I am giving you two images:
        1. An original 2D photograph of an object.
        2. A rendered 3D scatter plot of a voxelized model generated from that object.
        
        Please compare the structural similarity of the generated 3D shape to the original object.
        Consider that the 3D model is intentionally quantized to a low-resolution grid (max 60x60x4).
        
        Respond with ONLY a valid JSON object in the exact following format:
        {
          "score": <integer from 1 to 10>,
          "critique": "<A single short sentence explaining the score>"
        }
        """
        
        response = model.generate_content([prompt, img1, img2])
        
        # Cleanup temp render
        if os.path.exists(temp_render_path):
            os.remove(temp_render_path)
            
        # Parse JSON from response
        text = response.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
            
        result = json.loads(text)
        return result.get("score"), result.get("critique")
        
    except Exception as e:
        print(f"Gemini LLM Verification Error: {e}")
        return None, None
