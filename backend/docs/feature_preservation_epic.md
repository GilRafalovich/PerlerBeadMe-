# Epic: Feature preservation (image_project identity)

**Status:** Architect-approved for Algorithm implementation  
**Ask before commits.** Report paths/metrics via PerlerBeadMe coordinator.

## Problem

At kid footprint ~19×14, `color_mode=image_project` washes out identity marks (eyes, cheeks, ear tips, mouth, tail accents). Shape verify can still **pass** because `backward_verify` is shape/structure only (no color histogram) and does not score feature visibility.

### Baseline — `demo_pikachu_imgproj`

| Item | Value |
|--|--|
| Photo | `backend/demo_pikachu/01_input.jpg` (also copied as `demo_pikachu_imgproj/01_input.jpg`) |
| Out | `backend/demo_pikachu_imgproj/` |
| Voxel | **19×14×22**, 1596 beads |
| Colors | Yellow 1395, Orange 124, Grey 36, Brown 31, **Red 7**, Green 3 — **no Black** |
| Face layers 14–21 | 393 beads; **Red=1**, Brown=6, rest mostly Yellow |
| Verify gate | **passed** — `composite_error` **0.457** (threshold 0.55); front sil IoU 0.695 |
| Visual | `07_voxel_preview_3d.png` — yellow blob; cheeks/eyes/ear tips not readable |

Root cause in `quantizer.VoxelQuantizer._color_volume_image_project` (`backend/quantizer.py` ~668–722): horizontal photo bands resized with `INTER_NEAREST` onto each pancake. Sub-bead features disappear; `limit_colors` + majority fill favor Yellow; no post-pass forces identity beads.

## Goal

Detect/protect key photo regions → **forced Perler colors + minimum bead sizes**, then gate kits on **feature visibility** while keeping the existing shape composite (still **no color histogram**).

## Spec (Algorithm)

### A. Feature detect (photo space)

New module preferred: `backend/feature_protect.py` (or clear section in `quantizer.py`).

From framed/quantized RGB + subject mask (reuse `_subject_color_mask`):

| Feature | Detect heuristic (v1) | Forced color | Min beads / geometry |
|--|--|--|--|
| Cheeks | Warm red HSV blobs mid-face L/R | Perler **Red** | ≥ 2 beads each side (prefer 2×2 or 2 connected) |
| Eyes | Dark low-V blobs upper face | Perler **Black** (or darkest palette) | ≥ 1 bead each eye |
| Ear tips | Dark tips on top protrusions | **Black** | ≥ 1 bead per ear tip |
| Mouth / nose (optional v1) | Small dark mid-face | **Black** | ≥ 1 if confidently detected |
| Tail accent | High-contrast edge on rear/side bolt | keep contrast color (Brown/Black) | ≥ 2 beads if mask ≥ threshold |

Emit debug masks under out dir, e.g. `feature_masks/*.png`.

### B. Protect / inject (voxel space)

Hook **after** `_color_volume_image_project` (inside `build_color_volume` when `color_mode==image_project`):

1. Map each 2D feature mask → front-facing voxels using the **same** height→layer and subject-bbox mapping as image_project (document the mapping in code comments).
2. Overwrite those voxels with forced RGB; if target cell empty, optionally grow into nearest occupied neighbor on the same layer (do not invent floating beads outside silhouette unless ear tip already occupied).
3. Enforce min sizes by dilating in voxel XY on the feature’s layer band until min count or max dilate steps.
4. Ensure Black stays in palette when eyes/ear tips are present (`max_colors` must not drop Black — reserve slots for protected colors).

### C. Params

Add to `KidModeParams` (defaults on for character kits / demo):

- `feature_protect: bool = True`
- `feature_min_cheek: int = 2`
- `feature_min_eye: int = 1`
- `feature_min_ear_tip: int = 1`

CLI flags on `kid_pipeline.py` mirroring these.

### D. Acceptance (demo_pikachu_imgproj rerun)

Re-run same mesh/photo/footprint into a new out dir (e.g. `demo_pikachu_imgproj_feat`) **or** overwrite only after Gil approves:

```bash
cd /workspace/PerlerBeadMe/backend
../.venv/bin/python kid_pipeline.py \
  --image demo_pikachu/01_input.jpg \
  --mesh demo_pikachu_imgproj/02_mesh_aligned.obj \
  --out demo_pikachu_imgproj_feat \
  --footprint 18 --layers 22 --max-colors 6 \
  --color-mode image_project \
  --up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw
```

**Must:**

1. Face/readable: ≥2 Red cheek beads per side (or documented detect miss), ≥1 Black per eye, Black ear tips present.
2. Existing shape gate still runs; `composite_error` must not regress worse than **+0.05** vs baseline 0.457 (soft) unless feature gate would fail without protect.
3. No color-histogram metric introduced.

## Verification (PBM Verification — after Algorithm)

New gate **alongside** existing shape composite — **not** a histogram:

- `feature_visibility` score from protected masks vs final color volume (e.g. fraction of required min beads present with correct palette name within ΔE tolerance).
- Fail kit if required features missing when masks were confidently detected.
- Keep `backward_verify.composite_error` unchanged in definition (SSIM + LPIPS-lite + sil IoU).

Wire into `kid_pipeline` verify summary JSON: `feature_gate: {passed, details...}`.

## Backend / Frontend

**Not in this epic** unless Algorithm needs a flag exposed on the API. Default: algorithm-only; Backend wraps later if `KidModeParams` already flow through REST.

## Delegation

| Role | Task |
|--|--|
| **PBM Algorithm** | Implement A–D; local demo rerun; report bead counts + paths; **no commit** until asked |
| **PBM Verification** | After Algorithm signals ready: feature_visibility gate + regression on demo |
| **PBM Backend / Frontend** | Only if API/UI needed (Architect will re-open) |

## Feedback loop

Verification fail → Architect re-specs → Algorithm fix. Do not weaken shape composite to hide identity loss.
