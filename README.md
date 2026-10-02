# PerlerBeadMe

3D Agentic Perler Bead Modeler — turn a photo into kid-friendly, stackable Perler bead “pancake” layers (optional Lego-style peg interlock), with a FastAPI backend and React Three.js UI.

## Docs

Full documentation lives under [`docs/`](docs/):

| Doc | Contents |
|-----|----------|
| [docs/PIPELINE_FLOW.md](docs/PIPELINE_FLOW.md) | End-to-end flow: photo → restyle → TripoSR → photo-align → kid voxelize → colors → verify → instructions → interlock → UI |
| [docs/FEATURES.md](docs/FEATURES.md) | Kid mode, orientation, image_project, verify metrics, Lego pegs, instructions; planned feature preservation |
| [docs/CHANGELOG_SESSION.md](docs/CHANGELOG_SESSION.md) | Concrete session changes (HTTP TripoSR, kid_mode, orientation, photo-align, image_project, agent team) |
| [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | Commands for kid_pipeline, dry_run, Pikachu demos, API/UI |
| [docs/AGENT_TEAM.md](docs/AGENT_TEAM.md) | PBM Architect / Algorithm / Verification / Backend / Frontend + Dev channel |

Shorter backend notes: `backend/docs/orientation_protection.md`, `backend/docs/lego_interlock_design.md`.

## Quick start

```bash
cd backend
../.venv/bin/python kid_pipeline.py \
  --image demo_pikachu/01_input.jpg \
  --mesh demo_pikachu_photo_orient/02_mesh_aligned.obj \
  --out demo_pikachu_imgproj \
  --color-mode image_project \
  --footprint 18 --layers 22 --max-colors 6 \
  --up-axis 1 --no-auto-up-axis --yaw 0 --no-auto-yaw
```

See [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) for TripoSR, photo-align, API, and frontend.
