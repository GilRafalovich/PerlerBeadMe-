# Epic 4 — Character kit (template / restyle-first)

**Branch:** `epic4/character-kit` (same repo — no new project)  
**Status:** Architect-approved for Algorithm  
**Ask before commits.** Closed loop: Algorithm → Verification → Architect reject/re-delegate until PASS → Gil commit OK.

## Problem

Epics 1–3.1 optimized the **photo → mesh → image_project** path. Even at kid scale with feature protect, ship quality still fights photo noise, stubby meshes, and SSIM/composite chasing. Gil wants a **different product path**: build a recognizable **character kit** from a **template or generative restyle**, not more photo-mesh gates.

Photo SSIM / `composite_error` vs the raw photo is **not** the ship gate for this epic.

## Goal

A kid-buildable **Pikachu character kit** where:

1. Input photo (optional) drives **restyle / template** first — not direct mesh coloring as the identity source
2. Kit is **part-aware**: body, head, ears, eyes, cheeks, tail (and optional limbs) with forced roles
3. **Hard Pikachu palette** only: Yellow, Red, Black, Brown (ear tips / accents). No green/grey/noise colors on the body
4. Kid-scale: beads ≤ **800** (prefer 700–800), Easy/Medium ~45–60 min
5. Eyes ≥3 Black each, cheeks ≥3 Red each, Black ear tips — clearly readable in 3D preview
6. Verification ships on **recognizability + buildability**, not photo SSIM

## Spec (Algorithm)

### A. Character-kit pipeline (new path)

Prefer a dedicated entry (CLI flag or module), e.g. `--character-kit pikachu` on `kid_pipeline` or `character_kit.py`:

```
photo (optional) → restyle OR character template plate
                 → part labels / masks (body, head, ears, eyes, cheeks, tail)
                 → voxelize (mesh from restyle/TripoSR OR procedural character voxel)
                 → paint by part + hard palette
                 → feature_protect mins
                 → instructions
```

Allowed identity sources (pick one primary for v1, document which):

1. **Template-first:** procedural / hand-authored Pikachu part masks + silhouette (like standing-dog template in `kid_restyle.py`, but Pikachu parts + hard palette)
2. **Restyle-first:** generative or OpenCV restyle → flat character plate → part segmentation → kit
3. Hybrid: restyle head cues + template body/parts

Do **not** require beating Epic 2/3.1 photo `composite_error` to pass.

### B. Hard Pikachu palette

Only these Perler names on occupied voxels:

| Part | Color |
|--|--|
| Body / head / ears (base) | **Yellow** |
| Cheeks | **Red** |
| Eyes, nose/mouth accents, ear tips | **Black** |
| Optional ear tip / back stripe | **Brown** (≤ small %) |

Reject / remapping: Green, Grey, Orange noise → Yellow (or nearest allowed). `max_colors` ≤ 4 for this mode.

### C. Part-aware painting

Emit part masks (2D and/or voxel) under out dir `parts/`:

- `body`, `head`, `ear_left`, `ear_right`, `eye_left`, `eye_right`, `cheek_left`, `cheek_right`, `tail`

Paint voxels from part labels, not raw image_project bands. Feature mins: eye ≥3, cheek ≥3, ear tip ≥1 Black each.

### D. Scale / difficulty

Reuse Epic 3.1 kid band: footprint/layers targeting **700–800** beads, taller proportion (avoid cube stump). Medium ~45–60 min.

### E. Demo

```bash
cd /workspace/PerlerBeadMe/backend
# Exact CLI TBD by Algorithm; example:
../.venv/bin/python kid_pipeline.py \
  --image demo_pikachu/01_input.jpg \
  --character-kit pikachu \
  --out demo_pikachu_character_kit \
  --footprint <kid> --layers <kid> \
  --feature-protect --feature-min-eye 3 --feature-min-cheek 3
```

Out: `backend/demo_pikachu_character_kit/`

Work only on branch `epic4/character-kit`.

## Verification gates (ship) — NOT photo SSIM

| Gate | Pass |
|--|--|
| Beads | ≤ 800 |
| Palette | only Yellow/Red/Black/Brown (plus empty); Green=0 Grey=0 |
| Eyes | ≥3 Black each, readable on preview |
| Cheeks | ≥3 Red each |
| Ear tips | ≥1 Black each (or documented miss) |
| Parts | `parts/` present; eyes/cheeks/ears labeled |
| Buildability | Medium or Easy; layers coherent; no floating speckles |
| Recognizability | Verification visual: “reads as Pikachu” (ears + yellow body + red cheeks + black eyes) |

**Do not** hard-fail on photo `composite_error` / SSIM. May log them as diagnostics only.

## Fail → escalate to Architect

Hard FAIL if beads >800, illegal palette colors, eye/cheek mins fail, parts missing, or kit not recognizably Pikachu on preview. No silent pass.

## Delegation

| Role | Task |
|--|--|
| **PBM Algorithm** | A–E on `epic4/character-kit`; no commit until Gil OK |
| **PBM Verification** | Recognizability/buildability gates above |
| **PBM Architect** | Re-delegate until PASS; ask Gil commit |
| **Backend / Frontend** | Parked until API/UI assigned after Gil OK |

## Out of scope

- More photo-mesh silhouette tuning as the product path
- New GitHub project / repo
- API/UI wrap (later epic)
