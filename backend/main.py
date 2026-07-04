from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2

# Import our modular pipeline
from voxelizer import DepthEstimator
from quantizer import VoxelQuantizer
from color_quantizer import ColorQuantizer
from hybrid_assembler import HybridAssembler

app = FastAPI(title="PerlerBeadMe API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing AI Pipeline...")
depth_estimator = DepthEstimator()
voxel_quantizer = VoxelQuantizer(max_dim=20)
color_quantizer = ColorQuantizer()
print("Pipeline Ready.")

@app.post("/api/process_image")
async def process_image(file: UploadFile = File(...)):
    """
    The main endpoint. Takes an uploaded image and runs it through the
    entire Voxelization, Quantization, and Structural Decomposition pipeline.
    """
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Phase 2 Pipeline Execution:
        
        # 1. Monocular Depth Estimation
        depth_map, img_rgb = depth_estimator.generate_depth_map(temp_path)
        
        # 2. LAB Color Quantization
        quantized_color_img = color_quantizer.quantize(img_rgb)
        
        # 3. Spatial Quantization (Shrinking to 20x20x20)
        q_img, q_depth, voxel_matrix = voxel_quantizer.spatial_quantize(quantized_color_img, depth_map)
        
        # 4. Assembly-Optimized Hybrid Decomposition
        assembler = HybridAssembler(voxel_matrix, q_img)
        instructions = assembler.decompose()
        
        return {
            "status": "success",
            "voxel_shape": voxel_matrix.shape,
            "total_layers": len(instructions),
            "instructions": instructions
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
