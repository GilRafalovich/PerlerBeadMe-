# Orientation protection (kid_mode)

TripoSR meshes are often **correct but rotated** (default Y-up is the shortest axis → stubby kits). Kid pipeline now protects against that.

## Protections

1. **Standing-axis detection** (`auto_up_axis=True`)  
   Pick the **tallest** mesh extent as pancake up (PCA noted for agreement). Warn if chosen/forced up is the shortest axis.

2. **Photo-matched yaw** (`photo_matched_yaw=True`)  
   When a reference image is available, auto-yaw **minimizes** `backward_verify.composite_error` (SSIM + LPIPS-lite + silhouette IoU). Falls back to silhouette feature score without a ref.

3. **Verify gate** (`verify_gate=True`, default threshold `0.50`)  
   After voxelize+color, score vs the photo. If above threshold, retry alternate up-axes with photo-matched yaw. On failure, summary JSON includes `verify_gate.passed=false` and `fail_reason` (use `--verify-strict` to exit 2).

## Metrics (no color histogram)

```
composite_error = 0.40*(1−SSIM) + 0.25*LPIPS-lite + 0.35*(1−sil IoU)
```

Module: `backend/backward_verify.py`.

## CLI

```bash
cd backend
../.venv/bin/python kid_pipeline.py \
  --image demo_pikachu/01_input.jpg \
  --mesh demo_pikachu/02_mesh_triposr.obj \
  --out demo_pikachu_oriented \
  --footprint 18 --layers 16 --max-colors 6 \
  --verify-threshold 0.50
```

Flags: `--up-axis N`, `--no-auto-up-axis`, `--no-photo-yaw`, `--no-verify-gate`, `--verify-threshold`, `--verify-strict`, `--yaw`, `--no-auto-yaw`.

## Pikachu evidence

| | composite_error | sil IoU | up_axis | notes |
|--|--|--|--|--|
| Baseline (Y-up, yaw 120) | ~0.56 | ~0.29 | 1 | Y shortest → stubby |
| Oriented (auto + gate) | target ≤0.50 (combo ~0.43) | ~0.67 | tallest | see `demo_pikachu_oriented/` |

## Photo-matched mesh alignment

TripoSR meshes are remapped to **Y-up + photo facing** before voxelize when using
`backward_verify.align_mesh_to_photo` / `export_aligned_mesh_with_overlays`:

1. Standing axis = **tallest extent** → transform to +Y
2. Search yaw / roll / pitch minimizing **front-only** composite vs photo
   (SSIM + LPIPS-lite + sil IoU) **plus aspect-ratio penalty**
3. Mesh front rasters use **uniform letterbox scale** (no X/Y stretch) so lying-down
   blobs cannot fake a high IoU
4. Writes `02_mesh_aligned.obj` + photo↔mesh overlays

Kid yaw scoring / verify gate also use **front-only vs photo** (side is diagnostic only).

```bash
../.venv/bin/python - <<'PY'
from backward_verify import export_aligned_mesh_with_overlays
export_aligned_mesh_with_overlays(
    'demo_pikachu/02_mesh_triposr.obj',
    'demo_pikachu/01_input.jpg',
    'demo_pikachu_photo_orient',
)
PY
../.venv/bin/python kid_pipeline.py \
  --image demo_pikachu/01_input.jpg \
  --mesh demo_pikachu_photo_orient/02_mesh_aligned.obj \
  --out demo_pikachu_photo_orient \
  --footprint 18 --layers 22 --max-colors 6 \
  --up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw
```
