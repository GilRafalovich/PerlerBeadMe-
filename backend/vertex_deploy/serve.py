import os
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

# TripoSR imports
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

app = FastAPI(title="TripoSR Vertex AI Endpoint")

print("Loading TripoSR Model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)
print(f"TripoSR Model Loaded on {device}.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Takes a 2D image upload and returns a generated 3D .obj mesh file.
    """
    temp_img_path = f"/tmp/{file.filename}"
    with open(temp_img_path, "wb") as f:
        f.write(await file.read())
        
    image = Image.open(temp_img_path).convert("RGBA")
    
    # 1. Background removal and foreground resizing (Standard TripoSR preprocessing)
    image = remove_background(image, rembg_session=None)
    image = resize_foreground(image, 0.85)
    
    # 2. Generative 3D Mesh Extraction
    with torch.no_grad():
        scene_codes = model(image, device=device)
        mesh = model.extract_mesh(scene_codes, resolution=256)[0]
    
    # 3. Export as OBJ
    out_obj_path = f"/tmp/{file.filename}.obj"
    mesh.export(out_obj_path)
    
    return FileResponse(out_obj_path, media_type="application/octet-stream", filename="generated_model.obj")

@app.get("/health")
def health():
    """Vertex AI Health Check"""
    return {"status": "ok"}
