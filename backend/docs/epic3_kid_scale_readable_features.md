# Epic 3 — Kid-scale + readable features

**Status:** Architect-approved for Algorithm  
**Ask before commits.** Closed loop: Algorithm → Verification hard gate → Architect reject/re-delegate until PASS → Gil commit OK.

## Problem (Gil feedback on Epic 2)

`demo_pikachu_imgproj_eyes/` is the verified Epic 2 product but fails kid UX:

| Issue | Epic 2 |
|--|--|
| Beads | **1596** — too many for kids |
| Difficulty | Hard (~90 min) |
| Eyes | Black painted **2** each — not clearly seen in 3D preview |
| Cheeks | Red painted **4** each — OK count, but overall kit too dense |

Footprint 18 × layers 22 → shape `19×14×22`. Feature mins were eye=1 / cheek=2.

## Goal

A **kid-scale** kit where identity marks are **obvious** in the 3D preview:

1. Total beads ≤ **800** (stretch goal ≤ **600**)
2. Difficulty **Easy or Medium**, ~**45–60 min**
3. Each eye Black ≥ **3** painted beads (clearly visible); each cheek Red ≥ **3**
4. `feature_gate` still passes with **photo** eyes (`detected:true`, method not geometric)
5. Shape may trade off — soft composite budget below (not a hard fail at Epic 2’s 0.466)

## Spec (Algorithm)

### A. Scale down voxel budget

Search / pick kid params (same mesh + photo as Epic 2):

- Photo: `demo_pikachu/01_input.jpg`
- Mesh: `demo_pikachu_imgproj/02_mesh_aligned.obj` (or eyes demo’s mesh if identical)
- Start candidates: `max_footprint` **12–14**, `max_layers` **10–14**, `max_colors` 6, `feature_protect` on
- Prior art: older `demo_pikachu` hit **682** beads at shape `13×19×8` (pre–feature-protect) — use as scale reference, not as quality target

Stop when occupied beads ≤ 800 (prefer ≤ 600) **and** feature mins are met. If features can’t fit at ≤800, escalate to Architect (do not silently raise bead cap).

### B. Raise feature mins + visibility

`KidModeParams` / CLI for this demo:

- `feature_min_eye=3`
- `feature_min_cheek=3`
- `feature_min_ear_tip` ≥ 1 (keep Epic 2 ear behavior unless it blows bead budget)

Inject path (`feature_protect`):

- Grow eye / cheek clusters in voxel space until mins met (dilate / compact disk), still mapped from **photo** detections when possible
- Eyes must remain `method` photo_* (`photo_edge` / `photo_map`); geometric fallback stays **default OFF**
- If photo eye mask is thin, expand the projected hit set to ≥3 beads **around** the photo-mapped centroid (still count as photo-derived, not geometric seed) — document as `photo_edge_dilate` or similar in the per-eye method/notes

### C. Difficulty

Use existing `estimate_difficulty` (or adjust thresholds if needed) so summary reports Easy or Medium and ~45–60 minutes when beads ≤ 800. If the estimator still says Hard at ≤800 beads, fix the estimator thresholds for kid mode (Algorithm) so UX matches Gil’s ask.

### D. Shape soft budget

Epic 2 baseline: `composite_error` **0.466**.

| Tier | composite_error | Gate |
|--|--|--|
| Target | ≤ **0.55** | soft — prefer |
| Soft allow | ≤ **0.62** | OK if beads + features PASS |
| Hard FAIL | > **0.62** | escalate — kit too abstract |

No color histogram. Front sil IoU is diagnostic only for this epic.

### E. Demo rerun

```bash
cd /workspace/PerlerBeadMe/backend
../.venv/bin/python kid_pipeline.py \
  --image demo_pikachu/01_input.jpg \
  --mesh demo_pikachu_imgproj/02_mesh_aligned.obj \
  --out demo_pikachu_imgproj_kid \
  --footprint <chosen> --layers <chosen> --max-colors 6 \
  --color-mode image_project \
  --feature-protect \
  --feature-min-eye 3 --feature-min-cheek 3 \
  --up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw
```

Out: `backend/demo_pikachu_imgproj_kid/`

## Acceptance (PASS)

All must hold:

1. `occupied_beads` ≤ **800** (report stretch if ≤600)
2. Difficulty Easy or Medium; estimated minutes in **45–60** (or clearly labeled Easy/Medium if estimator uses a nearby band)
3. `feature_gate.details.eyes_*.painted` ≥ **3**, `detected:true`, method **not** geometric
4. `feature_gate.details.cheeks_*.painted` ≥ **3**
5. `feature_gate.passed == true`
6. `composite_error` ≤ **0.62**
7. Preview `07_voxel_preview_3d.png` shows eyes as obvious Black clusters (Algorithm + Verification visual check)

## Fail criteria (Verification → escalate to Architect)

Hard FAIL (no silent pass) if any of:

- beads > 800
- either eye Black painted < 3
- either cheek Red painted < 3
- either eye not photo-detected / geometric method
- `feature_gate.passed=false`
- `composite_error` > 0.62
- missing artifacts / preview

On FAIL: post paths + metrics in Dev; Architect re-specs / re-delegates.

## Delegation

| Role | Task |
|--|--|
| **PBM Algorithm** | A–E; no commit; post beads, eye/cheek painted, composite_error, difficulty |
| **PBM Verification** | Hard gate above; visual eye readability on preview |
| **PBM Architect** | Reject/re-delegate until PASS; then ask Gil commit |
| **Backend / Frontend** | Parked |

## Feedback loop

Verification FAIL → Architect adjusts (scale vs feature mins vs soft shape) → Algorithm → Verification. Prefer cutting layers/footprint before weakening feature mins. Never drop eye min below 3 to pass.
