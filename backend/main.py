# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import json

# Import our modular pipeline
from voxelizer import VertexAI3DEstimator
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
depth_estimator = VertexAI3DEstimator()
voxel_quantizer = VoxelQuantizer(max_dim_xy=60, max_dim_z=4)
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
        
    obj_path = f"{temp_path}.obj"
        
    try:
        # Phase 2 Pipeline Execution:
        
        # 1. Generative 3D Mesh (Vertex AI Endpoint)
        depth_estimator.generate_3d_mesh(temp_path, obj_path)
        
        # We still need img_rgb for color quantization
        img = cv2.imread(temp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. LAB Color Quantization
        quantized_color_img = color_quantizer.quantize(img_rgb)
        
        # 3. High-Res 60x60x60 Mesh Voxelization
        q_img, _, voxel_matrix = voxel_quantizer.spatial_quantize(quantized_color_img, obj_path)
        
        # 4. Assembly-Optimized Hybrid Decomposition
        assembler = HybridAssembler(voxel_matrix, q_img)
        instructions = assembler.decompose()
        
        return {
            "status": "success",
            "voxel_shape": voxel_matrix.shape,
            "voxel_matrix": voxel_matrix.tolist(),
            "color_matrix": q_img.tolist(),
            "total_layers": len(instructions),
            "instructions": instructions
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/metrics")
async def get_metrics():
    """Returns the verification metrics JSON."""
    results_path = "verification_results.json"
    if not os.path.exists(results_path):
        return {"status": "error", "message": "No verification results found. Please run a batch evaluation first."}
    with open(results_path, "r") as f:
        return json.load(f)

@app.post("/api/run_verification")
async def trigger_verification(limit: int = 100):
    """Spawns a background thread to run the verification pipeline."""
    import threading
    from verification_pipeline import run_verification
    
    # Run in background to avoid blocking the API
    thread = threading.Thread(target=run_verification, args=(limit,))
    thread.daemon = True
    thread.start()
    
    return {"status": "success", "message": f"Verification for {limit} samples started in the background."}

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
