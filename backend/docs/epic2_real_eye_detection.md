# Epic 2 — Real eye detection

**Status:** Architect-approved for Algorithm  
**Ask before commits.** Closed loop: Algorithm → Verification hard gate → Architect reject/re-delegate until PASS → Gil commit OK.

## Problem (Epic 1 leftover)

On `demo_pikachu_imgproj_feat`, eyes did **not** come from the photo:

| Eye | detected | method | painted |
|--|--|--|--|
| left | true* | `geometric_eye` | 1 |
| right | **false** | `geometric_eye` | 1 |

Notes: `eyes_geometric_seed`, `eyes_left_geometric_eye`, `eyes_right_geometric_eye`.

Root cause in `feature_protect.detect_features` (~239–265): eye path is **dark blob** (`V <= 80`, `min_area=5`) in an upper-face band. Pikachu’s photo eyes are **thin closed smile lines**, not filled blobs — detection misses → geometric seed disks → voxel `geometric_eye_hits`. `evaluate_feature_visibility` currently accepts painted≥min even when `detected=false` / method is geometric.

Baseline shape to protect: **composite_error 0.466** (`demo_pikachu_imgproj_feat`).

## Goal

Both eyes **`detected:true` from the photo** (not geometric seed); Black ≥1 bead each; `feature_gate` still passes under the **new hard rule**; soft shape regress ≤ **+0.05** vs 0.466 (i.e. composite_error ≤ **0.516**).

## Spec (Algorithm)

### A. Photo eye detector (replace blob-only)

In `feature_protect.detect_features` (keep cheeks/ears unless a change is required for eyes):

1. Prefer **thin dark strokes / closed-eye arcs** in the upper face band:
   - Edge or morphological black-hat / Canny on V or grayscale within subject ∩ eye_band
   - Or adaptive threshold + small connected components with **elongation** (line-like), not only area≥5 blobs
2. Pick **left and right** components (mid_x split); require one hit each side.
3. Only if a side is still empty after photo methods → leave that mask empty (do **not** write `eyes_geometric_seed` disks for Epic 2 acceptance path). Geometric seed may remain as an explicit **debug-only** flag (`feature_eye_geometric_fallback=False` default **off**).
4. Debug: write `feature_masks/eyes_left.png`, `eyes_right.png`, and note method (`photo_edge` / `photo_blob` / miss).

### B. Inject / paint

- Map photo eye masks → voxels (existing photo_map path).
- Force Perler **Black**, `min_eye` ≥ 1 each.
- **Do not** call `geometric_eye_hits` when `feature_eye_geometric_fallback` is false (default).
- Report per-eye: `detected`, `method` ∈ {`photo_map`, `photo_edge`, …} — **never** `geometric_eye` on the acceptance demo.

### C. Gate hardening (Algorithm + Verification)

Update `evaluate_feature_visibility` so for `eyes_left` / `eyes_right`:

- **FAIL** if `detected` is false **or** `method` starts with `geometric` **or** painted < min.
- Cheeks/ears keep Epic 1 rules (painted≥min; geometric ear notes allowed unless we later tighten).
- Still **no color histogram**.

Kid summary + `verify/gate_report.json` must show the hardened `feature_gate`.

### D. Demo rerun

```bash
cd /workspace/PerlerBeadMe/backend
../.venv/bin/python kid_pipeline.py \
  --image demo_pikachu/01_input.jpg \
  --mesh demo_pikachu_imgproj/02_mesh_aligned.obj \
  --out demo_pikachu_imgproj_eyes \
  --footprint 18 --layers 22 --max-colors 6 \
  --color-mode image_project \
  --feature-protect \
  --up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw
```

(Use same mesh/photo/footprint as Epic 1 feat.)

## Acceptance (PASS)

All must hold on `backend/demo_pikachu_imgproj_eyes/`:

1. `feature_gate.details.eyes_left.detected == true` and method is **photo-*** (not geometric)
2. Same for `eyes_right`
3. Black painted ≥ 1 each eye; overall kit still has Black (ear tips OK)
4. `feature_gate.passed == true` under hardened rules
5. Shape `composite_error` ≤ **0.516** (0.466 + 0.05); prefer not worse than +0.05
6. Artifacts: `00_summary.json`, `verify/gate_report.json`, `07_voxel_preview_3d.png`, `feature_masks/`

## Fail criteria (Verification → escalate to Architect)

Hard FAIL (no silent pass) if any of:

- Either eye `detected=false`
- Either eye `method` is geometric / seed
- painted < min for either eye
- `feature_gate.passed=false`
- `composite_error` > 0.516
- Missing required artifact paths

On FAIL: post paths + JSON snippets + preview path to Dev; Architect re-specs / re-delegates Algorithm. Do not ask Gil for commit.

## Delegation

| Role | Task |
|--|--|
| **PBM Algorithm** | A–D; no commit; post metrics when demo lands |
| **PBM Verification** | Hard gate above; PASS/FAIL with specifics |
| **PBM Architect** | Reject/re-delegate until PASS; then ask Gil commit |
| **Backend / Frontend** | Parked until Gil OKs API after Epic 2 PASS |

## Feedback loop

Verification FAIL → Architect adjusts spec → Algorithm fix → Verification again. No weakening of photo-detect requirement to “get a green check.”
