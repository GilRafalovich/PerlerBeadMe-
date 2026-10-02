# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import json
import uuid
from dataclasses import asdict

# Import our modular pipeline
from voxelizer import HuggingFace3DEstimator
from quantizer import VoxelQuantizer, KidModeParams
from color_quantizer import ColorQuantizer
from hybrid_assembler import HybridAssembler
from instruction_renderer import write_kid_artifacts
from kid_pipeline import run_kid_mode

app = FastAPI(title="PerlerBeadMe API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_ROOT = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_ROOT, exist_ok=True)

print("Initializing AI Pipeline...")
depth_estimator = HuggingFace3DEstimator()
voxel_quantizer = VoxelQuantizer(max_dim_xy=60, max_dim_z=4)
color_quantizer = ColorQuantizer()
print("Pipeline Ready.")

# Serve generated kid-mode artifacts
if os.path.isdir(ARTIFACTS_ROOT):
    app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_ROOT), name="artifacts")


@app.post("/api/process_image")
async def process_image(
    file: UploadFile = File(...),
    kid_mode: bool = Query(False, description="Enable kid-mode pancake build + instructions"),
    max_footprint: int = Query(18, ge=10, le=36),
    max_layers: int = Query(10, ge=4, le=16),
    max_colors: int = Query(5, ge=2, le=10),
):
    """
    Main endpoint. Uploaded image → mesh → voxels → assembly.
    When kid_mode=true, returns instruction artifact paths/payload
    (layer sheets, shopping list, PDF/MD/JSON), not only voxel preview.
    """
    job_id = uuid.uuid4().hex[:12]
    temp_path = f"temp_{job_id}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    obj_path = f"{temp_path}.obj"

    try:
        # 1. Generative 3D Mesh
        depth_estimator.generate_3d_mesh(temp_path, obj_path)

        img = cv2.imread(temp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if kid_mode:
            out_dir = os.path.join(ARTIFACTS_ROOT, f"kid_{job_id}")
            params = KidModeParams(
                max_footprint=max_footprint,
                max_layers=max_layers,
                hollow=False,
                max_colors=max_colors,
                auto_yaw=True,
                auto_up_axis=True,
                photo_matched_yaw=True,
                verify_gate=True,
            )
            summary = run_kid_mode(
                temp_path,
                obj_path,
                out_dir,
                params=params,
                title="Kid Perler Build",
            )
            # Copy mesh into artifacts for debugging
            if os.path.exists(obj_path):
                shutil.copy2(obj_path, os.path.join(out_dir, "02_mesh.obj"))

            rel = f"/artifacts/kid_{job_id}"
            artifacts = summary.get("artifacts", {})
            return {
                "status": "success",
                "mode": "kid_mode",
                "job_id": job_id,
                "voxel_shape": summary["voxel_shape"],
                "total_beads": summary["occupied_beads"],
                "total_layers": summary["num_layers"],
                "beads_per_layer": summary["beads_per_layer"],
                "shopping_list": summary["colors_used"],
                "difficulty": summary["difficulty"],
                "approx_size_mm": summary["approx_size_mm"],
                "silhouette": summary.get("silhouette"),
                "params": summary.get("params"),
                "artifacts_dir": rel,
                "artifacts": {k: f"{rel}/{v}" if isinstance(v, str) else [f"{rel}/{x}" for x in v]
                              for k, v in artifacts.items()},
                # Compact instructions payload (no huge color grids)
                "instructions": json.load(
                    open(os.path.join(out_dir, "instructions.json"))
                ),
            }

        # --- Legacy relief path ---
        quantized_color_img = color_quantizer.quantize(img_rgb)
        q_img, _, voxel_matrix = voxel_quantizer.spatial_quantize(
            quantized_color_img, obj_path, kid_mode=False
        )
        assembler = HybridAssembler(voxel_matrix, q_img)
        instructions = assembler.decompose()

        return {
            "status": "success",
            "mode": "default",
            "voxel_shape": list(voxel_matrix.shape),
            "voxel_matrix": voxel_matrix.tolist(),
            "color_matrix": q_img.tolist(),
            "total_layers": len(instructions),
            "instructions": instructions,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(obj_path):
            try:
                os.remove(obj_path)
            except OSError:
                pass


@app.get("/api/metrics")
async def get_metrics():
    """Returns the verification metrics JSON."""
    results_path = os.path.join("test_results", "verification_results.json")
    if not os.path.exists(results_path):
        results_path = "verification_results.json"
        if not os.path.exists(results_path):
            return {
                "status": "error",
                "message": "No verification results found. Please run a batch evaluation first.",
            }
    with open(results_path, "r") as f:
        return json.load(f)


@app.post("/api/run_verification")
async def trigger_verification(limit: int = 100):
    """Spawns a background thread to run the verification pipeline."""
    import threading
    from verification_pipeline import run_verification

    thread = threading.Thread(target=run_verification, args=(limit,))
    thread.daemon = True
    thread.start()

    return {
        "status": "success",
        "message": f"Verification for {limit} samples started in the background.",
    }


@app.post("/api/kid_mode_local")
async def kid_mode_local(
    image_path: str = Query(...),
    mesh_path: str = Query(""),
    max_footprint: int = Query(18),
    max_layers: int = Query(10),
    procedural_dog: bool = Query(False),
):
    """
    Dry-run / local invocation of kid_mode from existing image+mesh paths
    (no TripoSR). Useful for demos and verification.
    """
    from kid_pipeline import make_standing_dog_mesh

    job_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(ARTIFACTS_ROOT, f"kid_local_{job_id}")
    if procedural_dog:
        mesh_path = os.path.join(out_dir, "02_mesh_procedural_dog.obj")
        os.makedirs(out_dir, exist_ok=True)
        make_standing_dog_mesh(mesh_path)

    params = KidModeParams(
        max_footprint=max_footprint,
        max_layers=max_layers,
        hollow=False,
        auto_yaw=True,
        auto_up_axis=True,
        photo_matched_yaw=True,
        verify_gate=True,
    )
    summary = run_kid_mode(image_path, mesh_path, out_dir, params=params)
    rel = f"/artifacts/kid_local_{job_id}"
    return {
        "status": "success",
        "mode": "kid_mode",
        "artifacts_dir": rel,
        "summary": summary,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
