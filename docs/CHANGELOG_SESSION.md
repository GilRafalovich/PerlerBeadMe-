# Session changelog — PerlerBeadMe (Oct 2, 2026)

Concrete changes from this development session on the box clone  
`/workspace/PerlerBeadMe` (remote `https://github.com/GilRafalovich/PerlerBeadMe-.git`).

**No commit/push implied by this file** — local commit `4c2c501` was created earlier; later photo-align / image_project work may still be uncommitted. Ask Gil before committing or pushing.

---

## Git

| Ref | Summary |
|-----|---------|
| `4c2c501` | *Add kid_mode orientation protection and verify gate* — standing-axis, photo-matched yaw, verify gate; also landed kid pipeline, restyle, lego interlock, instruction renderer, TripoSR HTTP voxelizer, FastAPI kid path, docs under `backend/docs/` |
| Working tree (post-commit) | Further edits: `backward_verify.py` (mesh photo-align), `quantizer.py` / `kid_pipeline.py` (`image_project`, align flags), `orientation_protection.md`, demo dirs `demo_pikachu_*` |

Branch was **ahead of `origin/main` by 1** while GitHub auth was pending for push.

---

## HTTP TripoSR (`voxelizer.py`)

- Replaced fragile local mesh path with **HTTP POST** to TripoSR
- Default endpoint: `https://ahmedbelaid1-triposr.hf.space/generate`
- Override: env `REMOTE_3D_ENDPOINT_URL`
- Returns raw OBJ (`bake_texture_flag=false`); used by FastAPI and `--triposr` CLI

---

## Kid mode pipeline

New / expanded modules:

- `kid_pipeline.py` — end-to-end CLI + `run_kid_mode()`
- `kid_restyle.py` — kid-friendly restyle / head-only body invent
- `kid_mode_prototype.py` — earlier prototype retained
- `hybrid_assembler.py` — `decompose_kid()`
- `instruction_renderer.py` — sheets, PDF, MD, JSON
- `dry_run_kid.py` — smoke entry with restyle / procedural / TripoSR flags
- `quantizer.py` — `KidModeParams`, pancake voxelize, orientation, coloring
- `main.py` — `kid_mode` query on `/api/process_image`, `/api/kid_mode_local`

---

## Orientation protection

Problem: TripoSR Pikachu extents ≈ `[0.99, 0.42, 0.73]` — **Y shortest**, default Y-up → stubby **8**-layer kit.

Added:

1. `auto_up_axis` — tallest extent as pancake up  
2. `photo_matched_yaw` — yaw from `composite_error` vs photo  
3. `verify_gate` — fail/retry with alt axes  

**Before → after** (`demo_pikachu` vs `demo_pikachu_oriented`):

| | composite_error | sil IoU | shape | beads |
|--|--|--|--|--|
| Before | ~0.60 | ~0.29 | 13×19×8 | 682 |
| After | ~0.47 | ~0.65 | 18×19×16 | 1893 |

Source: `demo_pikachu_oriented/before_after_metrics.json` / `backend/docs/pikachu_oriented_before_after_metrics.json`

---

## Photo-align mesh

- `align_mesh_to_photo`, `export_aligned_mesh_with_overlays` in `backward_verify.py`
- Remap standing axis → Y-up; search yaw/roll/pitch; aspect penalty; uniform letterbox
- Demo: `demo_pikachu_photo_orient/` — align report yaw/roll/pitch **70°/20°/25°**, mesh front IoU **~0.53**; kid kit **19×14×22** / 1596 beads; verify err **~0.46**, front IoU **~0.70**

---

## Image project colors

- `KidModeParams.color_mode = "structural" | "image_project"`
- CLI `--color-mode image_project`
- Demo `demo_pikachu_imgproj/`: Yellow **1395** (was Brown-heavy under structural); Red cheeks only **7** beads at this resolution — motivates feature preservation

---

## Lego peg interlock

- New `lego_interlock.py` + design doc
- Demo under `demo_pikachu/lego_interlock/` (stud/socket on baseline Pikachu layers)

---

## Verify / diagnose tooling

- Expanded `backward_verify.py` (renders, SSIM, LPIPS-lite, sil IoU, overlays, diagnose collage, mesh align)
- `demo_pikachu_verify_optimize.py` and verify reports under `demo_pikachu/verify/`

**Policy locked:** no color histogram in photo-quality score.

---

## Agent team (session)

Live Grok teammates + channel (not only `.agents/skills/` personas):

| Teammate | Role |
|----------|------|
| PBM Architect | Requirements, architecture, re-delegate on fail |
| PBM Algorithm | Voxel / orient / color / feature preservation |
| PBM Verification | Gates, regressions, feature presence checks |
| PBM Backend | FastAPI wrap |
| PBM Frontend | React Three.js UI |

Shared room: **PerlerBeadMe Dev**.  
Repo skill mirrors remain under `/workspace/PerlerBeadMe/.agents/skills/`.  
See `docs/AGENT_TEAM.md`.

---

## Documentation this pass

Created under `/workspace/PerlerBeadMe/docs/`:

- `PIPELINE_FLOW.md`
- `FEATURES.md`
- `CHANGELOG_SESSION.md` (this file)
- `HOW_TO_RUN.md`
- `AGENT_TEAM.md`

README updated with a docs pointer.
