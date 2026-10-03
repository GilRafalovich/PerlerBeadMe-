# Epic 3.1 — Kid-scale with shape quality (not ship Epic 3)

**Status:** Architect-approved for Algorithm (re-open)  
**Ask before commits.** Closed loop: Algorithm → Verification → Architect reject/re-delegate until PASS → Gil commit OK.

## Problem (Gil on Epic 3)

`demo_pikachu_imgproj_kid/` met bead/feature mins but **looks bad**:

| | Epic 2 eyes | Epic 3 kid |
|--|--|--|
| Beads | 1596 | **600** |
| Shape | 19×14×22 | **15×11×11** (cube stump) |
| `composite_error` | **0.466** | **0.528** |
| Front sil IoU | **0.695** | **0.574** |
| Eyes/cheeks | 2→6 OK | 6/6 OK |
| Colors | cleaner yellow | stray **green/brown** on body |

Eyes/cheeks readable is not enough — proportion + silhouette must hold at kid scale.

## Goal

Keep kid budget and readable features, **and** recover silhouette:

1. Beads ≤ **800** (preferred band **700–800**; stretch ≤600 only if shape still PASSes)
2. Eyes Black ≥ **3** each, cheeks Red ≥ **3** each; photo-detected eyes (not geometric)
3. Shape hard: **front** `silhouette_iou` ≥ **0.65** **OR** `composite_error` ≤ **0.48**
4. Taller proportion (layers/footprint) — avoid near-cube stump like 15×11×11
5. Strip stray green/brown on body (map to Yellow / majority body) except protected features + ear tips

## Spec (Algorithm)

### A. Proportion search

Same photo/mesh as Epic 2/3:

- `--image demo_pikachu/01_input.jpg`
- `--mesh demo_pikachu_imgproj/02_mesh_aligned.obj`

Sweep footprint/layers targeting **700–800** beads with **height ≥ ~1.2 × max(footprint axes)** in voxel shape (e.g. prefer `~14×12×16`–`~16×12×18` style — taller than wide/deep). Reject cube-like shapes where `layers ≈ min(fx,fz)`.

Do **not** ship another 11-layer cube even if beads ≤800.

### B. Features (keep Epic 3 mins)

- `feature_min_eye=3`, `feature_min_cheek=3`
- Photo eyes (`photo_edge` / dilate OK); geometric fallback OFF
- Readable Black eye clusters in preview

### C. Body color cleanup

After `image_project` + feature protect:

- On occupied voxels that are **not** protected feature hits (eyes/cheeks/ear tips/mouth): remapping **Green** and small **Brown** islands on the torso/head yellow body → Perler **Yellow** (or subject majority warm yellow)
- Keep Brown only where it supports ear tips / intentional dark accents if photo-mapped; do not leave speckled green on body
- Report before/after color counts in summary notes

### D. Soft vs hard shape

| Metric | Pass if |
|--|--|
| Front sil IoU | ≥ **0.65** |
| **or** composite_error | ≤ **0.48** |

Must satisfy **at least one**. Prefer both. Side view diagnostic only.

Out: `backend/demo_pikachu_imgproj_kid_v2/`

```bash
cd /workspace/PerlerBeadMe/backend
../.venv/bin/python kid_pipeline.py \
  --image demo_pikachu/01_input.jpg \
  --mesh demo_pikachu_imgproj/02_mesh_aligned.obj \
  --out demo_pikachu_imgproj_kid_v2 \
  --footprint <chosen> --layers <chosen> --max-colors 6 \
  --color-mode image_project --feature-protect \
  --feature-min-eye 3 --feature-min-cheek 3 \
  --up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw
```

## Acceptance (PASS)

1. `occupied_beads` ≤ 800 (note if in 700–800 band)
2. Eyes ≥3 Black each, cheeks ≥3 Red each; photo methods; `feature_gate.passed`
3. Front sil IoU ≥ 0.65 **OR** composite_error ≤ 0.48
4. Voxel shape clearly taller than Epic 3’s cube (document `fx,fz,h` and aspect)
5. Body green/brown speckles largely gone vs Epic 3 (Yellow dominates body)
6. Medium/Easy difficulty still appropriate for bead count
7. Artifacts under `demo_pikachu_imgproj_kid_v2/`

## Fail criteria (Verification → escalate)

Hard FAIL if:

- beads > 800
- eye/cheek mins fail or geometric eyes
- **neither** front sil IoU ≥ 0.65 **nor** composite ≤ 0.48
- shape still cube-stump (layers ≈ min lateral) with no justification
- obvious green/brown body noise remains on preview
- missing artifacts

## Delegation

| Role | Task |
|--|--|
| **PBM Algorithm** | A–D; no commit; post beads, aspect, front sil IoU, composite, colors, feature paints |
| **PBM Verification** | Hard gate; compare vs Epic 2 eyes + Epic 3 kid baselines |
| **PBM Architect** | Re-delegate until PASS; ask Gil before commit |
| **Backend / Frontend** | Parked |

Epic 3 (`demo_pikachu_imgproj_kid/`) is **not ship** — keep for regression reference only.
