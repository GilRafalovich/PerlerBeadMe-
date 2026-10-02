# How to run PerlerBeadMe (kid pipeline & demos)

Assume venv at `/workspace/PerlerBeadMe/.venv` and cwd:

```bash
cd /workspace/PerlerBeadMe/backend
```

Python: `/workspace/PerlerBeadMe/.venv/bin/python`

---

## Prerequisites

```bash
cd /workspace/PerlerBeadMe
# if needed:
# python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
```

Optional TripoSR override:

```bash
export REMOTE_3D_ENDPOINT_URL="https://ahmedbelaid1-triposr.hf.space/generate"
```

---

## A. Kid pipeline from existing photo + mesh

### Oriented kit (auto standing-axis + photo yaw + verify)

```bash
cd /workspace/PerlerBeadMe/backend
../.venv/bin/python kid_pipeline.py \
  --image /workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg \
  --mesh /workspace/PerlerBeadMe/backend/demo_pikachu/02_mesh_triposr.obj \
  --out /workspace/PerlerBeadMe/backend/demo_pikachu_oriented \
  --footprint 18 --layers 16 --max-colors 6 \
  --title "Kid Perler Pikachu" \
  --verify-threshold 0.50
```

### Photo-align mesh, then voxelize (Y-up, face photo)

```bash
cd /workspace/PerlerBeadMe/backend
../.venv/bin/python - <<'PY'
from backward_verify import export_aligned_mesh_with_overlays
export_aligned_mesh_with_overlays(
    "/workspace/PerlerBeadMe/backend/demo_pikachu/02_mesh_triposr.obj",
    "/workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg",
    "/workspace/PerlerBeadMe/backend/demo_pikachu_photo_orient",
)
PY

../.venv/bin/python kid_pipeline.py \
  --image /workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg \
  --mesh /workspace/PerlerBeadMe/backend/demo_pikachu_photo_orient/02_mesh_aligned.obj \
  --out /workspace/PerlerBeadMe/backend/demo_pikachu_photo_orient \
  --footprint 18 --layers 22 --max-colors 6 \
  --up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw \
  --verify-threshold 0.55 \
  --title "Kid Perler Pikachu"
```

### Same align + image-projected colors (yellow from photo)

```bash
cd /workspace/PerlerBeadMe/backend
../.venv/bin/python kid_pipeline.py \
  --image /workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg \
  --mesh /workspace/PerlerBeadMe/backend/demo_pikachu_photo_orient/02_mesh_aligned.obj \
  --out /workspace/PerlerBeadMe/backend/demo_pikachu_imgproj \
  --footprint 18 --layers 22 --max-colors 6 \
  --color-mode image_project \
  --up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw \
  --verify-threshold 0.55 \
  --title "Kid Perler Pikachu (image_project)"
```

### Useful flags

| Flag | Effect |
|------|--------|
| `--color-mode structural\|image_project` | Coloring strategy |
| `--up-axis 0\|1\|2` | Force pancake up (disables auto) |
| `--no-auto-up-axis` | Keep forced/default up |
| `--yaw DEG` / `--no-auto-yaw` | Force / skip auto yaw |
| `--no-photo-yaw` | Silhouette-only yaw |
| `--no-verify-gate` | Skip post-voxelize gate |
| `--verify-threshold 0.50` | Composite error fail line |
| `--verify-strict` | Exit code 2 if gate fails |
| `--restyle` | Kid restyle before palette |
| `--triposr` | Regenerate mesh via HTTP |
| `--procedural-dog` | Built-in standing dog mesh |

---

## B. Dry-run smoke (`dry_run_kid.py`)

```bash
cd /workspace/PerlerBeadMe/backend

# Restyle + procedural dog mesh
../.venv/bin/python dry_run_kid.py \
  --restyle --procedural-dog \
  --image /workspace/PerlerBeadMe/backend/demo_dog/01_input.jpg \
  --out /workspace/PerlerBeadMe/backend/demo_dog_restyle

# Restyle + try TripoSR (fallback procedural)
../.venv/bin/python dry_run_kid.py \
  --restyle --triposr \
  --image /workspace/PerlerBeadMe/backend/demo_dog/01_input.jpg \
  --out /workspace/PerlerBeadMe/backend/demo_dog_restyle

# Existing mesh
../.venv/bin/python dry_run_kid.py \
  --image /workspace/PerlerBeadMe/backend/demo_dog/01_input.jpg \
  --mesh /workspace/PerlerBeadMe/backend/demo_dog/02_mesh.obj \
  --out /workspace/PerlerBeadMe/backend/demo_dog_kid_pipeline/from_old_mesh
```

---

## C. Regenerate mesh only (TripoSR)

```bash
cd /workspace/PerlerBeadMe/backend
../.venv/bin/python - <<'PY'
from voxelizer import HuggingFace3DEstimator
HuggingFace3DEstimator().generate_3d_mesh(
    "/workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg",
    "/workspace/PerlerBeadMe/backend/demo_pikachu/02_mesh_triposr.obj",
)
PY
```

Or via pipeline:

```bash
../.venv/bin/python kid_pipeline.py \
  --image /workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg \
  --triposr \
  --out /workspace/PerlerBeadMe/backend/demo_pikachu_rerun \
  --footprint 18 --layers 16 --max-colors 6
```

---

## D. Lego interlock (post-hoc on a kit)

See `/workspace/PerlerBeadMe/backend/demo_pikachu/lego_interlock/README.md` and  
`apply_pikachu_interlock.py` in that folder. Core API:

```python
from lego_interlock import apply_interlock, InterlockParams
# voxel (X,Z,H), colors (X,Z,H,3) from a finished kid run
result = apply_interlock(voxel, colors, InterlockParams())
```

---

## E. FastAPI + frontend

```bash
# Terminal 1 — API
cd /workspace/PerlerBeadMe/backend
../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — UI
cd /workspace/PerlerBeadMe/frontend
npm install   # first time
npm run dev
```

Kid mode via API (UI may not pass the query yet):

```bash
curl -X POST "http://localhost:8000/api/process_image?kid_mode=true&max_footprint=18&max_layers=12&max_colors=6" \
  -F "file=@/workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg"
```

Local paths without TripoSR:

```bash
curl -X POST "http://localhost:8000/api/kid_mode_local?image_path=/workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg&mesh_path=/workspace/PerlerBeadMe/backend/demo_pikachu_photo_orient/02_mesh_aligned.obj&max_footprint=18&max_layers=22"
```

---

## F. Where to look after a run

| Artifact | Path pattern |
|----------|----------------|
| Summary | `<out>/00_summary.json` |
| Layer sheets | `<out>/layer_XX_instruction.png` |
| PDF / MD / JSON | `<out>/instructions.{pdf,md,json}` |
| Verify gate | `<out>/verify/gate_report.json` |
| Overlays | `<out>/verify/overlay_front.png` |
| Mesh align | `<out>/02_mesh_align_report.json`, `02_mesh_aligned.obj` |

### Existing Pikachu demos

| Dir | Notes |
|-----|--------|
| `/workspace/PerlerBeadMe/backend/demo_pikachu/` | Baseline + TripoSR mesh + lego_interlock |
| `/workspace/PerlerBeadMe/backend/demo_pikachu_oriented/` | Orientation guards |
| `/workspace/PerlerBeadMe/backend/demo_pikachu_rerun/` | Oriented rerun |
| `/workspace/PerlerBeadMe/backend/demo_pikachu_photo_orient/` | Photo-aligned mesh + structural colors |
| `/workspace/PerlerBeadMe/backend/demo_pikachu_imgproj/` | Photo-aligned + `image_project` |
