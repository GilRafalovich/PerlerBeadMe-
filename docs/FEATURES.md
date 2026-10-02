# PerlerBeadMe — Features

Status as of the Oct 2, 2026 session (box clone `/workspace/PerlerBeadMe`).  
**Shape quality metrics never use a color histogram.**

---

## Implemented

### Kid mode (`kid_mode`)

Kid-friendly **stacked pancake** builds instead of the legacy shallow relief.

- Solid layers (hollowing off) so silhouettes stay readable
- Small footprint (default max **18** beads on longest horizontal extent)
- Layer height capped (`max_layers`, demos often 10–22)
- Bottom layers morphologically widened for stable feet
- Caps distinct Perler colors (`max_colors`, typically 4–6)
- Difficulty + time estimate from bead/layer counts
- Assembly order **bottom → top**

**Code:** `KidModeParams` + `VoxelQuantizer` in  
`/workspace/PerlerBeadMe/backend/quantizer.py`  
**Entry:** `/workspace/PerlerBeadMe/backend/kid_pipeline.py`, FastAPI `kid_mode=true`

---

### Orientation guards

Protects against TripoSR meshes that are upright in the wrong axis (stubby kits).

1. **Standing-axis detection** (`auto_up_axis=True`)  
   Choose the **tallest** mesh extent as pancake up (PCA noted). Warn if forced up is the shortest axis.

2. **Photo-matched yaw** (`photo_matched_yaw=True`)  
   When a reference image is available, auto-yaw minimizes `backward_verify.composite_error`. Without a ref, falls back to silhouette feature score.

3. **Verify gate** (`verify_gate=True`)  
   After voxelize+color, score reconstructed ortho renders vs the photo. On failure, retry alternate up-axes with photo-matched yaw. Summary records `verify_gate.passed` / `fail_reason`. CLI `--verify-strict` exits 2 on fail.

**Also:** `align_mesh_to_photo` / `export_aligned_mesh_with_overlays` remaps the mesh to Y-up + photo facing *before* voxelize (yaw/roll/pitch search, aspect penalty, uniform letterbox).

**Docs:** `/workspace/PerlerBeadMe/backend/docs/orientation_protection.md`

---

### Image-projected colors (`color_mode=image_project`)

- **`structural`** (default): colors by mesh structure / height bands — often brown/grey even when the subject is yellow
- **`image_project`**: project framed photo RGB onto layers, then snap to Perler palette

Pikachu evidence (`demo_pikachu_imgproj`): Yellow **1395** / Brown **31** / Orange **124** / Red **7** on a 1596-bead kit (vs structural Yellow **375** / Brown **668**).

CLI: `--color-mode image_project`

---

### Backward verify metrics (no color histogram)

Module: `/workspace/PerlerBeadMe/backend/backward_verify.py`

| Metric | Meaning |
|--------|---------|
| **SSIM** | Structural similarity on aligned grayscale views (higher better) |
| **LPIPS-lite** | Multi-scale LAB + Sobel edge L2, torch-free (lower better) |
| **Silhouette IoU** | Binary mask IoU after letterbox align (higher better) |
| **composite_error** | Weighted sum used for yaw search + gate (lower better) |

```
composite_error = 0.40*(1 − SSIM) + 0.25*LPIPS-lite + 0.35*(1 − sil IoU)
```

Used for: photo-matched yaw, mesh align search, post-voxelize verify gate, diagnose overlays/collages.  
**Explicitly excluded:** color histogram / palette distribution in the quality score.

---

### Lego-style peg interlock

Mechanical registration between pancake layers (not molded brick studs).

- **Stud on bottom / socket on top** lattice (inset from rim)
- Clears socket cells; kids seat short **2-bead pegs**
- Instruction glyphs + seating diagram
- Feasible with ordinary fuse plates; dovetail / half-offset rejected for stacked flats

**Code:** `/workspace/PerlerBeadMe/backend/lego_interlock.py`  
**Design:** `/workspace/PerlerBeadMe/backend/docs/lego_interlock_design.md`  
**Demo:** `/workspace/PerlerBeadMe/backend/demo_pikachu/lego_interlock/`

---

### Instruction renderer

`/workspace/PerlerBeadMe/backend/instruction_renderer.py` → `write_kid_artifacts`

Outputs per kit:

- `layer_XX_instruction.png` — numbered grid sheets + legend
- `00_color_key.png` — shopping poster
- `07_voxel_preview_3d.png`, `07_side_silhouette.png`, `08_assembly_overview.png`
- `instructions.md`, `instructions.json`, `instructions.pdf`
- `00_summary.json`

---

### Kid restyle

`/workspace/PerlerBeadMe/backend/kid_restyle.py` — optional stylized plate; can invent standing body for head-only photos. Used heavily in dog demos; optional for Pikachu.

---

### TripoSR HTTP mesh

`/workspace/PerlerBeadMe/backend/voxelizer.py` — remote mesh generation (HF Space by default). Replaces local-only / brittle offline paths for demos and API.

---

### FastAPI + React UI

- **API:** `/workspace/PerlerBeadMe/backend/main.py`  
  - `POST /api/process_image?kid_mode=true`  
  - `POST /api/kid_mode_local` (paths, no TripoSR)  
  - metrics / verification triggers  
  - static `/artifacts/...`
- **UI:** `/workspace/PerlerBeadMe/frontend/` — upload, `ThreeVisualizer`, `AssemblyGuide`, metrics `Dashboard`

---

## Feature preservation (Epic 1 — uncommitted)

**Implemented on box (ask before commit).** Spec: `backend/docs/feature_preservation_epic.md`.

- **Algorithm:** `backend/feature_protect.py` + hooks in `quantizer.py` / `kid_pipeline.py` (`--feature-protect`, min cheek/eye/ear). Post-`image_project` injects Perler Red/Black for cheeks/eyes/ear tips; debug masks under `feature_masks/`.
- **Demo:** `backend/demo_pikachu_imgproj_feat/` — Red 7→15, Black 0→4; cheeks 4+4, eyes 1+1, ear tips 2+2; shape `composite_error` 0.466 (+0.009 vs baseline 0.457, soft +0.05 OK).
- **Verification gate:** `feature_gate` / `feature_visibility` in `00_summary.json` and `verify/gate_report.json` via `evaluate_feature_visibility` (score = mean painted/min over required features; **no color histogram**). Shape `composite_error` unchanged.
- **Notes on demo:** left cheek mirrored; eyes geometric seed — recorded in gate notes, not auto-pass.
- **Still open:** mouth/tail accent optional; Backend/Frontend API wrap parked until Gil OK.

**Ownership:** Architect designs → Algorithm protects → Verification gates. See `docs/AGENT_TEAM.md`.

---

## Quick feature matrix

| Feature | Status |
|---------|--------|
| Kid pancake voxelize | ✅ |
| Standing-axis + photo yaw + verify gate | ✅ |
| Photo-align mesh (yaw/roll/pitch) | ✅ |
| `image_project` colors | ✅ |
| Shape metrics SSIM / LPIPS / IoU | ✅ |
| Color histogram in score | ❌ by design |
| Instruction sheets + PDF | ✅ |
| Lego stud/socket interlock | ✅ (post-hoc / demo) |
| Feature preservation (face/eyes/ears/tail) | ✅ box demo + `feature_gate` (uncommitted) |
| Frontend `kid_mode` query by default | ⚠️ API supports; UI may omit query |
