# PerlerBeadMe — Agent team & Dev channel

PerlerBeadMe is developed with a small multi-agent team that mirrors the repo’s  
`.agents/skills/` personas, plus live Grok teammates that hand work to each other.

**Project roots**

- Box clone: `/workspace/PerlerBeadMe`
- Windows checkout: `C:\Projects\PerlerBeadMe`
- GitHub: `https://github.com/GilRafalovich/PerlerBeadMe-.git`

**Rules of engagement**

- Coordinate in the **PerlerBeadMe Dev** channel
- Report concrete paths and metrics (SSIM / LPIPS-lite / sil IoU / composite_error — **no color histogram**)
- **Ask Gil before commits or pushes**
- Do not reinvent stale architecture docs; prefer code + `docs/` here

---

## Live teammates (Grok)

| Name | Focus |
|------|--------|
| **PBM Architect** | Lead architect: analyze requirements, design modules, delegate to Algorithm / Backend / Frontend / Verification. Owns feedback loops — **reject** builds that fail requirements and re-delegate. Mandate kid-friendly kits and **feature preservation** (face / eyes / ears / tail). |
| **PBM Algorithm** | Voxel math, photo-align, `image_project`, Lego interlock, **feature preservation** implementation. Primary files: `backend/quantizer.py`, `voxelizer.py`, `backward_verify.py`, `kid_pipeline.py`, `lego_interlock.py`. Hands verified modules to Backend. |
| **PBM Verification** | Run verify gates; compare bead renders to photo; fail on silhouette IoU / composite_error regressions; check **feature presence** (face/eyes visible). Escalate with logs + image paths to Architect / Algorithm. |
| **PBM Backend** | FastAPI (`main.py`), kid_mode APIs, artifact I/O, schemas for Frontend. |
| **PBM Frontend** | React Three.js UI: upload, layer instructions, 3D bead preview; consume Backend schemas. |

**Shared channel:** PerlerBeadMe Dev — Architect posts blueprints; Algorithm posts metrics/paths; Verification posts pass/fail; Backend/Frontend sync on API contracts.

---

## Feedback loop

```
Gil (product ask)
    │
    ▼
PBM Architect ──blueprint──► PBM Algorithm
    ▲                              │
    │                         implement + local smoke
    │                              │
    │                              ▼
    │                        PBM Verification
    │                         pass? ──yes──► PBM Backend ──► PBM Frontend
    │                           │
    │                          no
    │                           │
    └────── reject + re-delegate ◄┘
```

1. Architect breaks the ask into file-level tasks and acceptance metrics  
2. Algorithm implements and reports numbers (e.g. front sil IoU, composite_error, bead counts)  
3. Verification re-runs gates / demos; **fails the build** on regressions or missing features  
4. On fail, Architect re-analyzes and re-delegates (does not quietly ship)  
5. Backend wraps stable APIs; Frontend consumes schemas  
6. Commits/pushes only after Gil’s OK

---

## Current mandate: feature preservation

**Problem:** `image_project` makes Pikachu yellow but still has **no face** — eyes/cheeks vanish at coarse grid.

**Next design (Architect → Algorithm → Verification)**

- Detect or mark key regions on the photo (face ROI, eyes, cheeks, ear tips, tail)
- Protect those cells: minimum bead count, forced palette colors
- Optional: feature term in verify (not color histogram of whole image)
- Acceptance: kid kit still “reads as” the character in front render + human spot-check

---

## Repo skill personas (reference)

Under `/workspace/PerlerBeadMe/.agents/skills/`:

| Skill folder | Maps roughly to |
|--------------|-----------------|
| `architect_agent` | PBM Architect |
| `algorithm_agent` | PBM Algorithm |
| `verification_agent` / `qa_agent` | PBM Verification |
| `backend_agent` | PBM Backend |
| `frontend_agent` | PBM Frontend |
| `integrator_agent` | E2E / DevOps bridge |
| `researcher_agent` | External mechanics / refs |
| `optimization_agent` | Perf / quality tuning |
| `3d_algorithm_debugger` | Voxel debug specialist |

Live teammates supersede skill text when they conflict with current `docs/` and code (e.g. kid pancakes + Lego pegs are the active kid path; Plus-Plus friction fit remains a research/alternate track).

---

## Where agents should look

| Concern | Path |
|---------|------|
| Pipeline map | `docs/PIPELINE_FLOW.md` |
| Feature status | `docs/FEATURES.md` |
| Run commands | `docs/HOW_TO_RUN.md` |
| Session deltas | `docs/CHANGELOG_SESSION.md` |
| Orientation | `backend/docs/orientation_protection.md` |
| Lego pegs | `backend/docs/lego_interlock_design.md` |
| Pikachu demos | `backend/demo_pikachu*` |
