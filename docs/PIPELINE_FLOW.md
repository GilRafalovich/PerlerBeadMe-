# PerlerBeadMe — End-to-End Pipeline Flow

This document describes the current **kid_mode** path from photo to printable instructions (and optional Lego interlock / React UI). Paths are absolute on the box clone; the Windows checkout is `C:\Projects\PerlerBeadMe`.

**Primary modules**

| Stage | Module |
|-------|--------|
| TripoSR HTTP mesh | `/workspace/PerlerBeadMe/backend/voxelizer.py` (`HuggingFace3DEstimator`) |
| Kid restyle | `/workspace/PerlerBeadMe/backend/kid_restyle.py` |
| Orchestration CLI | `/workspace/PerlerBeadMe/backend/kid_pipeline.py` |
| Voxel + orientation + colors | `/workspace/PerlerBeadMe/backend/quantizer.py` (`KidModeParams`, `VoxelQuantizer`) |
| Photo↔mesh align | `/workspace/PerlerBeadMe/backend/backward_verify.py` (`align_mesh_to_photo`) |
| Palette | `/workspace/PerlerBeadMe/backend/color_quantizer.py` |
| Layer decompose | `/workspace/PerlerBeadMe/backend/hybrid_assembler.py` |
| Instructions / PDF | `/workspace/PerlerBeadMe/backend/instruction_renderer.py` |
| Verify gate metrics | `/workspace/PerlerBeadMe/backend/backward_verify.py` |
| Lego peg interlock | `/workspace/PerlerBeadMe/backend/lego_interlock.py` |
| FastAPI | `/workspace/PerlerBeadMe/backend/main.py` |
| React UI | `/workspace/PerlerBeadMe/frontend/` |

---

## Flow diagram (kid_mode)

```
Photo (JPG/PNG)
    │
    ├─[optional] kid_restyle ──► 02_restyled.png (+ invent body if head-only)
    │
    ├─[optional] TripoSR HTTP ──► 02_mesh_triposr.obj
    │         (REMOTE_3D_ENDPOINT_URL or default HF Space)
    │         on failure → procedural standing-dog mesh (dog demos)
    │
    ├─[recommended] photo-align mesh
    │         align_mesh_to_photo / export_aligned_mesh_with_overlays
    │         ──► 02_mesh_aligned.obj + overlays + align report
    │
    ▼
ColorQuantizer on framed subject ──► 03_palette_quantized.png
    │
    ▼
VoxelQuantizer.spatial_quantize(kid_mode=True)
    │  • auto_up_axis: tallest mesh extent → pancake up
    │  • auto_yaw / photo_matched_yaw: minimize composite_error vs photo
    │  • solid pancakes, footprint/layers caps, base widen
    │  • color_mode: structural | image_project
    │
    ▼
HybridAssembler.decompose_kid() ──► per-layer bead maps + shopping list
    │
    ▼
Verify gate (default on)
    │  render front/side ortho beads vs photo
    │  composite_error = 0.40*(1−SSIM) + 0.25*LPIPS-lite + 0.35*(1−sil IoU)
    │  if fail → retry alternate up-axes (verify_retry_alt_axes)
    │  artifacts under <out>/verify/
    │
    ▼
write_kid_artifacts
    │  layer_XX_instruction.png, instructions.md/json/pdf
    │  color key, voxel preview, side silhouette, assembly overview
    │  00_summary.json
    │
    ├─[optional] lego_interlock.apply_interlock ──► stud/socket sheets + pegs
    │
    └─[optional] FastAPI / React
          POST /api/process_image?kid_mode=true
          frontend ThreeVisualizer + AssemblyGuide
```

---

## Stage details

### 1. Photo input

- Typical demo: `/workspace/PerlerBeadMe/backend/demo_pikachu/01_input.jpg`
- Pipeline copies `01_input.jpg` and a framed preview `01_framed.jpg` into the out dir.

### 2. Restyle (optional)

- Flag: `--restyle` on `kid_pipeline.py` / `dry_run_kid.py`
- Writes `02_restyled.png` + `02_restyle_meta.json`
- Can invent a standing body when the photo is head-only (`invent_body_if_head_only`)
- Restyled plate becomes the palette / framing guide; mesh may still come from TripoSR on that plate

### 3. TripoSR mesh (optional / API default)

- `HuggingFace3DEstimator.generate_3d_mesh(image, out.obj)`
- HTTP POST to `REMOTE_3D_ENDPOINT_URL` or  
  `https://ahmedbelaid1-triposr.hf.space/generate`
- Params: background removal, `foreground_ratio=0.85`, `mc_resolution=256`, `bake_texture_flag=false` (raw OBJ bytes)
- FastAPI `/api/process_image` always generates a mesh this way before kid or legacy path
- CLI: `--triposr`; on failure kid CLI can fall back to `--procedural-dog`

### 4. Photo-aligned mesh (recommended before voxelize)

TripoSR meshes are often **correct but rotated** (e.g. Y shortest). Separate alignment step:

1. Pick standing axis = **tallest extent** → remap to +Y
2. Search yaw / roll / pitch minimizing **front-only** composite vs photo (+ aspect penalty)
3. Front rasters use **uniform letterbox** (no X/Y stretch)
4. Export `02_mesh_aligned.obj` + photo↔mesh overlays

API: `backward_verify.export_aligned_mesh_with_overlays(mesh, photo, out_dir)`  
Evidence: `/workspace/PerlerBeadMe/backend/demo_pikachu_photo_orient/02_mesh_align_report.json`  
(yaw/roll/pitch ≈ 70°/20°/25°, mesh front sil IoU ≈ 0.53)

After alignment, run kid voxelize with `--up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw` so pancakes stay Y-up and face the photo.

### 5. Kid voxelize

`KidModeParams` defaults (see `quantizer.py`):

- `max_footprint=18`, `max_layers=10` (demos often raise layers to 16–22)
- `hollow=False`, base widen on bottom layers
- `auto_up_axis=True`, `auto_yaw=True`, `photo_matched_yaw=True`
- `verify_gate=True`, `verify_error_threshold=0.50` (some demos used 0.55)
- `color_mode`: `"structural"` (default) or `"image_project"`

Orientation protection inside voxelize (when mesh not pre-aligned):

1. **Standing-axis detection** — tallest extent as pancake up; warn if forced up is shortest
2. **Photo-matched yaw** — minimize `composite_error` vs ref RGB (else silhouette feature score)
3. **Verify gate** — post-voxelize score; retry alt axes; optional `--verify-strict` exit 2

### 6. Coloring

| Mode | Behavior |
|------|----------|
| `structural` | Height / structure bands → often brown/grey dominant (not photo yellow) |
| `image_project` | Project framed photo colors onto voxel layers → Pikachu mostly Yellow |

Palette capped by `max_colors` via Perler nearest colors (`ColorQuantizer`).

**Not implemented:** feature preservation (forced eyes / cheeks / ear tips / tail stripes). Coarse projection alone erases face details at ~19×14 front. See `docs/FEATURES.md`.

### 7. Assembly + instructions

- `HybridAssembler.decompose_kid()` → bottom→top layers, bead counts, shopping list, difficulty estimate
- `write_kid_artifacts()` → PNG sheets, MD, JSON, multi-page PDF, previews
- Summary: `00_summary.json` (shape, colors, silhouette, orientation, verify_gate)

### 8. Verify gate artifacts

Under `<out>/verify/`:

- `gate_report.json`
- `render_front.png`, `render_side.png`
- `overlay_front.png`, `overlay_side.png`

**Shape metrics only** — no color histogram in the score.

### 9. Optional Lego interlock

- Module: `lego_interlock.py` — stud-on-bottom / socket-on-top lattice + connector pegs
- Design notes: `/workspace/PerlerBeadMe/backend/docs/lego_interlock_design.md`
- Demo: `/workspace/PerlerBeadMe/backend/demo_pikachu/lego_interlock/`
- Not yet wired as a default flag inside `kid_pipeline.py` main; apply post-hoc on voxel/color volumes

### 10. React UI + API

- Backend: `uvicorn` on `main.py` — `POST /api/process_image?kid_mode=true`
- Serves artifacts under `/artifacts/kid_<job_id>/`
- Frontend (`App.tsx`): upload → `localhost:8000/api/process_image` (legacy path unless `kid_mode` query is passed); `ThreeVisualizer`, `AssemblyGuide`, metrics `Dashboard`

---

## Reference demo outputs (Pikachu)

| Out dir | What it shows |
|---------|----------------|
| `/workspace/PerlerBeadMe/backend/demo_pikachu/` | Baseline stubby kit (pre-orientation) |
| `/workspace/PerlerBeadMe/backend/demo_pikachu_oriented/` | Auto standing-axis + photo yaw + gate |
| `/workspace/PerlerBeadMe/backend/demo_pikachu_rerun/` | Oriented rerun after commit |
| `/workspace/PerlerBeadMe/backend/demo_pikachu_photo_orient/` | Mesh photo-align + structural colors |
| `/workspace/PerlerBeadMe/backend/demo_pikachu_imgproj/` | Same align + `image_project` colors |

Latest imgproj kit (example): grid **19×14×22**, **1596** beads, Yellow **1395**, verify err **~0.46**, front sil IoU **~0.70**.

---

## Related short docs in backend

- `/workspace/PerlerBeadMe/backend/docs/orientation_protection.md`
- `/workspace/PerlerBeadMe/backend/docs/lego_interlock_design.md`
