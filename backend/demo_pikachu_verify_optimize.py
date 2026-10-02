#!/usr/bin/env python3
"""
Pikachu demo: diagnose → backward verify → optimize knobs.

Uses existing mesh + images under demo_pikachu/. Writes under demo_pikachu/verify/.
No commit. Run with repo .venv from backend/:
  ../.venv/bin/python demo_pikachu_verify_optimize.py
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backward_verify import (
    VerifyReport,
    load_rgb,
    lpips_lite,
    make_diagnose_collage,
    red_pixel_fraction,
    render_mesh_ortho,
    render_voxel_ortho,
    save_overlay,
    score_views,
    silhouette_iou,
    ssim_score,
    subject_mask,
    verify_voxel_against_image,
)
from color_quantizer import ColorQuantizer
from quantizer import KidModeParams, VoxelQuantizer

DEMO = os.path.join(os.path.dirname(__file__), "demo_pikachu")
OUT = os.path.join(DEMO, "verify")
MESH = os.path.join(DEMO, "02_mesh_triposr.obj")
INPUT = os.path.join(DEMO, "01_input.jpg")
RESTYLE = os.path.join(DEMO, "02_restyled.png")
PALETTE = os.path.join(DEMO, "03_palette_quantized.png")
MESH_PREV = os.path.join(DEMO, "02_mesh_triposr_preview.png")
VOXEL_PREV = os.path.join(DEMO, "07_voxel_preview_3d.png")
ANGLE_FRONT = os.path.join(DEMO, "angles", "front.png")
BASE_VOXEL = os.path.join(DEMO, "04_voxel_matrix.npy")
BASE_COLOR = os.path.join(DEMO, "04_color_volume.npy")


def abs_p(p: str) -> str:
    return os.path.abspath(p)


def diagnose() -> dict:
    os.makedirs(OUT, exist_ok=True)
    inp = load_rgb(INPUT)
    rest = load_rgb(RESTYLE)
    pal = load_rgb(PALETTE)
    front = load_rgb(ANGLE_FRONT) if os.path.exists(ANGLE_FRONT) else load_rgb(VOXEL_PREV)

    mesh = trimesh.load(MESH, force="mesh")
    extents = [float(x) for x in mesh.extents]
    vox = np.load(BASE_VOXEL)
    col = np.load(BASE_COLOR)
    uniq = np.unique(col.reshape(-1, 3), axis=0)
    # occupied unique
    occ_cols = np.unique(col[vox > 0], axis=0)

    red_in = red_pixel_fraction(inp)
    red_re = red_pixel_fraction(rest)
    red_pa = red_pixel_fraction(pal)
    red_vo = red_pixel_fraction(front)

    # Mesh orientation evidence
    pitch = float(max(extents)) / 18.0
    raw = mesh.voxelized(pitch=pitch).fill().matrix.astype(np.uint8)
    shapes = {}
    for up in (0, 1, 2):
        if up == 1:
            p = np.transpose(raw, (0, 2, 1))
        elif up == 0:
            p = np.transpose(raw, (1, 2, 0))
        else:
            p = raw
        shapes[f"up_axis_{up}"] = list(p.shape)

    with open(os.path.join(DEMO, "00_summary.json")) as f:
        summary = json.load(f)
    yaw = summary.get("params", {}).get("yaw_deg") or summary.get("silhouette", {}).get("yaw_deg")

    findings = [
        f"A) RESTYLE LOSS: red-cheek frac {red_in:.3f}→{red_re:.3f}; cheeks→brown (opencv k-means).",
        f"B) PALETTE CRUSH: max_colors=4 + structural → {len(occ_cols)} bead colors, red%={red_pa:.3f}/{red_vo:.3f}.",
        f"C) TRIPOSR DRIFT: extents XYZ={[round(e,3) for e in extents]}; Y shortest → stubby kit.",
        f"D) COARSE GRID: voxel {list(vox.shape)} ({int(vox.sum())} beads) cannot resolve ears/cheeks/tail.",
        f"E) YAW={yaw}° from auto silhouette score — not photo-matched 3/4 pose.",
        f"F) up_axis=1 → h={shapes['up_axis_1'][2]} layers; alt up0 h={shapes['up_axis_0'][2]} (mesh flat on Y).",
        "G) structural color ignores cheek loci; White/Yellow/Black only (no Red in volume).",
    ]

    collage = make_diagnose_collage(
        {
            "input": INPUT,
            "restyle": RESTYLE,
            "mesh": MESH_PREV,
            "palette": PALETTE,
            "voxel_front": ANGLE_FRONT if os.path.exists(ANGLE_FRONT) else VOXEL_PREV,
            "voxel_iso": VOXEL_PREV,
        },
        os.path.join(OUT, "diagnose_collage.png"),
        findings,
    )

    # Also write annotated text findings
    findings_path = os.path.join(OUT, "diagnose_findings.md")
    with open(findings_path, "w") as f:
        f.write("# Pikachu voxel kit — root-cause diagnosis\n\n")
        f.write("## Pipeline stages vs photo\n\n")
        f.write("| Stage | Red cheek frac | Notes |\n|---|---|---|\n")
        f.write(f"| Input `{abs_p(INPUT)}` | {red_in:.4f} | Clear red pouches |\n")
        f.write(f"| Restyle `{abs_p(RESTYLE)}` | {red_re:.4f} | Cheeks browned / muted |\n")
        f.write(f"| Palette `{abs_p(PALETTE)}` | {red_pa:.4f} | Red eliminated |\n")
        f.write(f"| Voxel front | {red_vo:.4f} | Colors: {occ_cols.tolist()} |\n\n")
        f.write("## Findings\n\n")
        for line in findings:
            f.write(f"- {line}\n")
        f.write("\n## Mesh axis shapes (pitch≈max/18)\n\n")
        f.write(f"```\n{json.dumps(shapes, indent=2)}\n```\n")
        f.write(f"\nCollage: `{abs_p(collage)}`\n")

    return {
        "findings": findings,
        "collage": abs_p(collage),
        "findings_md": abs_p(findings_path),
        "red_fractions": {
            "input": red_in,
            "restyle": red_re,
            "palette": red_pa,
            "voxel_front": red_vo,
        },
        "mesh_extents": extents,
        "voxel_shape": list(vox.shape),
        "occupied_colors": occ_cols.tolist(),
        "up_axis_shapes": shapes,
        "baseline_yaw": yaw,
    }


def verify_baseline(diag: dict) -> dict:
    vox = np.load(BASE_VOXEL)
    col = np.load(BASE_COLOR)
    reports = {}
    for ref_name, ref_path in [("input", INPUT), ("restyle", RESTYLE)]:
        rep = verify_voxel_against_image(
            vox,
            col,
            ref_path,
            OUT,
            label=f"baseline_vs_{ref_name}",
            views=["front", "side"],
            silhouette_weight=0.30,
        )
        # Mesh ortho at baseline yaw for extra evidence
        mesh = trimesh.load(MESH, force="mesh")
        yaw = float(diag.get("baseline_yaw") or 120)
        for view in ("front", "side"):
            mimg = render_mesh_ortho(mesh, view=view, yaw_deg=yaw, out_size=256)
            mp = os.path.join(OUT, f"render_mesh_yaw{int(yaw)}_{view}.png")
            cv2.imwrite(mp, cv2.cvtColor(mimg, cv2.COLOR_RGB2BGR))
            rep.artifact_paths[f"mesh_{view}"] = abs_p(mp)
        jpath = os.path.join(OUT, f"error_report_baseline_vs_{ref_name}.json")
        with open(jpath, "w") as f:
            json.dump(rep.to_dict(), f, indent=2)
        rep.artifact_paths["error_report"] = abs_p(jpath)
        reports[ref_name] = rep.to_dict()
    return reports


def recolor_volume(voxel, img_rgb, params: KidModeParams) -> np.ndarray:
    vq = VoxelQuantizer(kid_params=params)
    cq = ColorQuantizer()
    framed = VoxelQuantizer.crop_subject_rgb(img_rgb)
    q = cq.quantize(framed)
    if params.max_colors:
        q = vq.limit_colors(q, params.max_colors)
    return vq.build_color_volume(voxel, q, params)


def run_candidate(mesh, img_rgb, params: KidModeParams):
    vq = VoxelQuantizer(kid_params=params)
    voxel = vq.voxelize_kid(mesh, params)
    colors = recolor_volume(voxel, img_rgb, params)
    yaw = getattr(vq, "_last_kid_yaw", params.yaw_deg or 0.0)
    return voxel, colors, float(yaw or 0.0)


def evaluate_candidate(voxel, colors, ref_rgb, silhouette_weight: float) -> dict:
    renders = {
        "front": render_voxel_ortho(voxel, colors, "front", 256),
        "side": render_voxel_ortho(voxel, colors, "side", 256),
    }
    rows, summary = score_views(
        ref_rgb, renders, size=256, silhouette_weight=silhouette_weight
    )
    summary["views"] = [asdict(r) for r in rows]
    summary["shape"] = list(voxel.shape)
    summary["beads"] = int(voxel.sum())
    # red retention in render
    summary["red_frac_front"] = red_pixel_fraction(renders["front"])
    return summary, renders


def optimize(diag: dict, baseline_reports: dict) -> dict:
    mesh = trimesh.load(MESH, force="mesh")
    # Prefer restyle as color source (pipeline did), score vs INPUT for likeness
    img_bgr = cv2.imread(RESTYLE)
    if img_bgr is None:
        img_rgb = load_rgb(RESTYLE)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ref = load_rgb(INPUT)

    baseline_params = KidModeParams(
        max_footprint=18,
        max_layers=10,
        min_layers=6,
        hollow=False,
        color_mode="structural",
        max_colors=4,
        up_axis=1,
        yaw_deg=120.0,
        auto_yaw=False,
    )

    # Search grid (small)
    yaw_opts = [0, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    footprint_opts = [18, 24, 28]
    layer_opts = [10, 14]
    max_color_opts = [4, 6, 8]
    color_modes = ["structural", "image_project"]
    sil_weights = [0.20, 0.30, 0.45]

    trials = []
    # 1) Baseline exact
    trials.append(("baseline", baseline_params, 0.30))

    # 2) Yaw sweep at baseline footprint
    for yaw in yaw_opts:
        if yaw == 120:
            continue
        p = deepcopy(baseline_params)
        p.yaw_deg = float(yaw)
        p.auto_yaw = False
        trials.append((f"yaw_{yaw}", p, 0.30))

    # 3) Footprint / layers / colors / color_mode
    for fp in footprint_opts:
        for layers in layer_opts:
            for mc in max_color_opts:
                for cm in color_modes:
                    if fp == 18 and layers == 10 and mc == 4 and cm == "structural":
                        continue  # baseline
                    p = deepcopy(baseline_params)
                    p.max_footprint = fp
                    p.max_layers = layers
                    p.max_colors = mc
                    p.color_mode = cm
                    p.yaw_deg = 120.0
                    p.auto_yaw = False
                    trials.append((f"fp{fp}_L{layers}_c{mc}_{cm}", p, 0.30))

    # 4) Best yaw so far will be refined; also try silhouette weight on scoring only
    # Limit combinatorial explosion: cap at ~40 trials by sampling if needed
    if len(trials) > 50:
        # keep baseline + all yaws + a subset of others
        head = trials[: 1 + len(yaw_opts)]
        rest = trials[1 + len(yaw_opts) :]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(rest), size=min(25, len(rest)), replace=False)
        trials = head + [rest[i] for i in idx]

    results = []
    best = None
    best_renders = None
    best_voxel = None
    best_colors = None

    print(f"Optimize: {len(trials)} trials...")
    for name, params, sw in trials:
        try:
            voxel, colors, yaw_used = run_candidate(mesh, img_rgb, params)
            summary, renders = evaluate_candidate(voxel, colors, ref, sw)
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            continue
        row = {
            "name": name,
            "params": asdict(params),
            "yaw_used": yaw_used,
            "silhouette_weight": sw,
            **{k: summary[k] for k in summary if k != "views"},
            "views": summary["views"],
        }
        results.append(row)
        print(
            f"  {name}: err={summary['composite_error']:.4f} "
            f"ssim={summary['mean_ssim']:.3f} iou={summary['mean_silhouette_iou']:.3f} "
            f"shape={summary['shape']} red={summary['red_frac_front']:.3f}"
        )
        if best is None or summary["composite_error"] < best["composite_error"]:
            best = row
            best_renders = renders
            best_voxel = voxel
            best_colors = colors

    # Iterative nudge around best yaw ±15° and footprint ±2
    if best is not None:
        base_p = KidModeParams(**{k: best["params"][k] for k in best["params"]})
        nudge_yaws = sorted(
            {
                float(base_p.yaw_deg or 0) + d
                for d in (-20, -10, -5, 5, 10, 20)
            }
        )
        for yaw in nudge_yaws:
            yaw = yaw % 360
            p = deepcopy(base_p)
            p.yaw_deg = float(yaw)
            p.auto_yaw = False
            name = f"nudge_yaw_{int(yaw)}"
            try:
                voxel, colors, yaw_used = run_candidate(mesh, img_rgb, p)
                summary, renders = evaluate_candidate(voxel, colors, ref, 0.30)
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                continue
            row = {
                "name": name,
                "params": asdict(p),
                "yaw_used": yaw_used,
                "silhouette_weight": 0.30,
                **{k: summary[k] for k in summary if k != "views"},
                "views": summary["views"],
            }
            results.append(row)
            print(
                f"  {name}: err={summary['composite_error']:.4f} "
                f"ssim={summary['mean_ssim']:.3f} iou={summary['mean_silhouette_iou']:.3f}"
            )
            if summary["composite_error"] < best["composite_error"]:
                best = row
                best_renders = renders
                best_voxel = voxel
                best_colors = colors

        # Score-weight sensitivity on best geometry (does not change voxels)
        for sw in sil_weights:
            summary, _ = evaluate_candidate(best_voxel, best_colors, ref, sw)
            results.append(
                {
                    "name": f"score_w_sil_{sw}_on_best",
                    "params": best["params"],
                    "yaw_used": best["yaw_used"],
                    "silhouette_weight": sw,
                    **{k: summary[k] for k in summary if k != "views"},
                    "views": summary["views"],
                    "note": "scoring-only; geometry unchanged",
                }
            )

    # Save best artifacts
    best_dir = os.path.join(OUT, "best")
    os.makedirs(best_dir, exist_ok=True)
    if best_voxel is not None:
        np.save(os.path.join(best_dir, "voxel_matrix.npy"), best_voxel)
        np.save(os.path.join(best_dir, "color_volume.npy"), best_colors)
        for view, img in best_renders.items():
            p = os.path.join(best_dir, f"render_{view}.png")
            cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            save_overlay(
                ref,
                img,
                os.path.join(best_dir, f"overlay_{view}.png"),
                title=f"best/{view} vs input",
            )
        # Also baseline overlays already exist; write comparison strip
        base_front = render_voxel_ortho(
            np.load(BASE_VOXEL), np.load(BASE_COLOR), "front", 256
        )
        strip = np.ones((256 + 40, 256 * 3 + 20, 3), dtype=np.uint8) * 245
        strip[40:, 0:256] = cv2.resize(
            load_rgb(INPUT), (256, 256), interpolation=cv2.INTER_AREA
        )
        strip[40:, 266 : 266 + 256] = base_front
        strip[40:, 532 : 532 + 256] = best_renders["front"]
        for i, lab in enumerate(["input", "baseline", "best"]):
            cv2.putText(
                strip,
                lab,
                (10 + i * 266, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(
            os.path.join(best_dir, "compare_input_baseline_best.png"),
            cv2.cvtColor(strip, cv2.COLOR_RGB2BGR),
        )

    # Baseline metrics for report
    base_err = baseline_reports["input"]["composite_error"]
    opt = {
        "baseline_composite_error": base_err,
        "best": best,
        "improvement": None
        if best is None
        else float(base_err - best["composite_error"]),
        "n_trials": len(results),
        "trials": sorted(results, key=lambda r: r["composite_error"]),
        "artifacts": {
            "best_dir": abs_p(best_dir),
            "compare": abs_p(os.path.join(best_dir, "compare_input_baseline_best.png")),
        },
    }
    with open(os.path.join(OUT, "optimize_report.json"), "w") as f:
        json.dump(opt, f, indent=2)
    return opt


def recommendations(diag: dict, opt: dict) -> str:
    lines = [
        "# Recommended algorithm changes (real pipeline)",
        "",
        "1. **Restyle must preserve brand colors**: replace OpenCV k-means cartoon with "
        "palette-aware restyle (force Perler Red/Yellow/Black/Brown seeds; protect cheek "
        "hue via HSV gate before posterize). Optionally skip cartoon when subject already "
        "has flat anime colors.",
        "2. **Photo-matched yaw, not silhouette-only**: auto_yaw should maximize "
        "SSIM/IoU of orthographic mesh/voxel render vs input (this verify loop), not only "
        "`_silhouette_feature_score`. Seed candidates with pose estimate from photo.",
        "3. **Mesh orientation repair**: TripoSR extents show Y shortest — detect "
        "standing axis (tallest PCA / foot-ground heuristic) and remap `up_axis` before "
        "kid voxelize; reject or re-run TripoSR when height/width < ~0.7 for bipeds.",
        "4. **Raise kid defaults for character kits**: footprint 24–32, layers 12–16, "
        "`max_colors` ≥ 6 (keep Red). Coarse 13×19×8 cannot hold ears + cheeks + tail.",
        "5. **Color mode**: prefer `image_project` (or back-project restyle UVs) over "
        "dog-centric `structural` regions for characters; inject cheek blobs from 2D "
        "red mask onto front voxels.",
        "6. **Closed-loop optimize in pipeline**: after voxelize, run `backward_verify` "
        "and accept/reject; optional short yaw/footprint search when composite_error high.",
        "7. **Texture bake**: TripoSR vertex colors still have ~1.8% red — sample mesh "
        "vertex colors into voxels instead of 2D band project when available.",
        "",
        "## This run",
        f"- Diagnose collage: `{diag['collage']}`",
        f"- Baseline vs input composite_error: {opt['baseline_composite_error']:.4f}",
    ]
    if opt.get("best"):
        b = opt["best"]
        lines += [
            f"- Best trial `{b['name']}` composite_error: {b['composite_error']:.4f} "
            f"(Δ {opt['improvement']:+.4f})",
            f"- Best params: footprint={b['params']['max_footprint']} layers={b['params']['max_layers']} "
            f"max_colors={b['params']['max_colors']} color_mode={b['params']['color_mode']} "
            f"yaw={b['params']['yaw_deg']}",
            f"- Best compare image: `{opt['artifacts']['compare']}`",
        ]
    path = os.path.join(OUT, "recommendations.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return abs_p(path)


def main():
    print("=== 1. DIAGNOSE ===")
    diag = diagnose()
    for line in diag["findings"]:
        print(line)
    print("collage:", diag["collage"])

    print("\n=== 2. BACKWARD VERIFY (baseline) ===")
    reports = verify_baseline(diag)
    for k, r in reports.items():
        print(
            f"vs {k}: composite_error={r['composite_error']:.4f} "
            f"ssim={r['mean_ssim']:.3f} iou={r['mean_silhouette_iou']:.3f} "
            f"lpips_lite={r['mean_lpips_lite']:.3f}"
        )

    print("\n=== 3. OPTIMIZE ===")
    opt = optimize(diag, reports)
    if opt.get("best"):
        print(
            f"BEST {opt['best']['name']} err={opt['best']['composite_error']:.4f} "
            f"improvement={opt['improvement']:+.4f}"
        )

    print("\n=== 4. RECOMMENDATIONS ===")
    rec = recommendations(diag, opt)
    print(rec)

    master = {
        "diagnose": diag,
        "baseline_verify": reports,
        "optimize": {
            "baseline_composite_error": opt["baseline_composite_error"],
            "best": opt["best"],
            "improvement": opt["improvement"],
            "n_trials": opt["n_trials"],
            "top5": opt["trials"][:5],
            "artifacts": opt["artifacts"],
        },
        "recommendations_md": rec,
        "verify_dir": abs_p(OUT),
    }
    master_path = os.path.join(OUT, "master_report.json")
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)
    print("MASTER", abs_p(master_path))


if __name__ == "__main__":
    main()
