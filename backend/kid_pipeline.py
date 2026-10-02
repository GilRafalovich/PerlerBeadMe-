"""
End-to-end kid_mode pipeline: image + mesh → voxel pancakes + instruction artifacts.

Usage:
  python kid_pipeline.py \\
      --image demo_dog/01_input.jpg \\
      --mesh demo_dog/02_mesh.obj \\
      --out demo_dog_kid_pipeline

Or with silhouette-friendly defaults for a standing subject:
  python kid_pipeline.py --image ... --mesh ... --out ... --footprint 18 --layers 10
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np

from color_quantizer import ColorQuantizer
from hybrid_assembler import HybridAssembler
from instruction_renderer import write_kid_artifacts
from quantizer import KidModeParams, VoxelQuantizer
from kid_restyle import restyle_for_kid


def run_kid_mode(
    image_path: str,
    mesh_path: str,
    out_dir: str,
    params: Optional[KidModeParams] = None,
    title: str = "Kid Perler Build",
    copy_input: bool = True,
    restyle: bool = False,
    invent_body: bool = True,
) -> dict:
    """
    Run kid_mode quantization + assembly + instruction rendering.
    When restyle=True, runs kid-friendly restyle first and uses the restyled
    image for palette / framing (mesh path is unchanged — caller may regenerate
    mesh from the restyled plate).
    Returns the summary dict (also written under out_dir).
    """
    params = params or KidModeParams()
    os.makedirs(out_dir, exist_ok=True)

    color_image_path = image_path
    restyle_meta = None
    if restyle:
        restyled_path = os.path.join(out_dir, "02_restyled.png")
        restyle_meta = restyle_for_kid(
            image_path,
            restyled_path,
            invent_body_if_head_only=invent_body,
        )
        color_image_path = restyled_path
        with open(os.path.join(out_dir, "02_restyle_meta.json"), "w") as f:
            json.dump(restyle_meta.to_dict(), f, indent=2)
        print(f"restyle -> {restyled_path} method={restyle_meta.method} "
              f"head_only={restyle_meta.head_only_detected} body={restyle_meta.body_method}")

    img_bgr = cv2.imread(color_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {color_image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    color_q = ColorQuantizer()
    # Frame subject before palette map for cleaner contrast
    framed = VoxelQuantizer.crop_subject_rgb(img_rgb)
    quantized = color_q.quantize(framed)
    if params.max_colors:
        vq_tmp = VoxelQuantizer(kid_params=params)
        quantized = vq_tmp.limit_colors(quantized, params.max_colors)

    cv2.imwrite(
        os.path.join(out_dir, "03_palette_quantized.png"),
        cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR),
    )
    if copy_input:
        # Always preserve original upload when restyling
        if restyle and os.path.exists(image_path):
            raw = cv2.imread(image_path)
            if raw is not None:
                cv2.imwrite(os.path.join(out_dir, "01_input.jpg"), raw)
            else:
                cv2.imwrite(os.path.join(out_dir, "01_input.jpg"), img_bgr)
        else:
            cv2.imwrite(os.path.join(out_dir, "01_input.jpg"), img_bgr)
        # Keep a framed preview too (from restyled/color image)
        cv2.imwrite(
            os.path.join(out_dir, "01_framed.jpg"),
            cv2.cvtColor(framed, cv2.COLOR_RGB2BGR),
        )

    vq = VoxelQuantizer(kid_params=params)
    # spatial_quantize kid path expects already-ish quantized colors; pass quantized
    padded, color_volume, voxel = vq.spatial_quantize(
        quantized, mesh_path, kid_mode=True, kid_params=params
    )
    # Prefer snapped color volume from assembler
    assembler = HybridAssembler(
        voxel,
        padded,
        color_volume=color_volume,
        kid_mode=True,
        bead_mm=params.bead_mm,
        color_quantizer=color_q,
    )
    result = assembler.decompose_kid()
    colors = result.pop("color_volume")

    used_params = getattr(vq, "_last_kid_params", params)
    summary = write_kid_artifacts(
        out_dir,
        voxel,
        colors,
        result,
        title=title,
        params_dict=asdict(used_params),
    )

    # Lightweight silhouette metric for reports
    side = (np.max(voxel, axis=1) > 0).astype(np.uint8).T
    peri = int(
        cv2.dilate(side, np.ones((3, 3), np.uint8)).sum()
        - cv2.erode(side, np.ones((3, 3), np.uint8)).sum()
    )
    summary["silhouette"] = {
        "side_occupied": int(side.sum()),
        "side_perimeter": peri,
        "yaw_deg": getattr(vq, "_last_kid_yaw", used_params.yaw_deg),
        "up_axis": int(getattr(used_params, "up_axis", 1)),
        "feature_score": float(VoxelQuantizer._silhouette_feature_score(voxel)),
    }
    orient = getattr(vq, "_last_orient_info", None)
    if orient:
        summary["orientation"] = orient
    gate = getattr(vq, "_last_verify_gate", None)
    if gate:
        # Persist verify renders/overlays under out_dir/verify/
        verify_dir = os.path.join(out_dir, "verify")
        os.makedirs(verify_dir, exist_ok=True)
        try:
            from backward_verify import save_overlay, load_rgb

            ref = load_rgb(color_image_path)
            for view, img in (gate.pop("_renders", None) or {}).items():
                pass  # renders not stored on gate
            # Re-render for artifacts
            from backward_verify import render_voxel_ortho

            for view in ("front", "side"):
                img = render_voxel_ortho(voxel, colors, view, 256)
                p = os.path.join(verify_dir, f"render_{view}.png")
                cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                ov = os.path.join(verify_dir, f"overlay_{view}.png")
                save_overlay(ref, img, ov, title=f"verify/{view}")
            gate_public = {k: v for k, v in gate.items() if k != "renders"}
            with open(os.path.join(verify_dir, "gate_report.json"), "w") as f:
                json.dump(gate_public, f, indent=2)
            summary["verify_gate"] = gate_public
            summary["verify_dir"] = verify_dir
        except Exception as e:
            summary["verify_gate"] = {k: v for k, v in gate.items() if k != "renders"}
            summary["verify_gate"]["artifact_error"] = str(e)

    if restyle_meta is not None:
        summary["restyle"] = restyle_meta.to_dict()
    with open(os.path.join(out_dir, "00_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== KID MODE PIPELINE DONE ===")
    print(f"shape={voxel.shape} beads={summary['occupied_beads']} layers={voxel.shape[2]}")
    print(f"colors={summary['colors_used']}")
    print(f"difficulty={summary['difficulty']}")
    print(f"silhouette={summary['silhouette']}")
    if summary.get("verify_gate"):
        vg = summary["verify_gate"]
        print(
            f"verify_gate passed={vg.get('passed')} err={vg.get('composite_error'):.4f} "
            f"thr={vg.get('threshold')} up={vg.get('up_axis')} yaw={vg.get('yaw_deg')}"
        )
        if not vg.get("passed"):
            print(f"VERIFY GATE FAIL: {vg.get('fail_reason')}")
    print(f"out={out_dir}")
    return summary


def make_standing_dog_mesh(obj_path: str) -> str:
    """
    Procedural standing-dog mesh with clear head/ears/legs/tail silhouette.
    Used when the photo mesh is a blob (e.g. swimming head-only) so kid kits
    can still show a recognizable outline. Colors come from the photo later.
    """
    import trimesh

    parts = []

    def box(center, extents, name=None):
        b = trimesh.creation.box(extents=extents)
        b.apply_translation(center)
        parts.append(b)

    # Body (loaf) — elongated along +X (nose direction), thicker Z for kid footprint
    box([0.0, 0.38, 0.0], [1.25, 0.5, 0.72])
    # Chest bump
    box([0.48, 0.34, 0.0], [0.38, 0.42, 0.62])
    # Neck
    box([0.72, 0.58, 0.0], [0.3, 0.3, 0.32])
    # Head
    box([1.0, 0.75, 0.0], [0.42, 0.38, 0.42])
    # Snout
    box([1.28, 0.65, 0.0], [0.32, 0.22, 0.26])
    # Ears (two upright) — exaggerated for kid recognition
    box([0.92, 1.08, 0.16], [0.14, 0.34, 0.12])
    box([0.92, 1.08, -0.16], [0.14, 0.34, 0.12])
    # Legs (4) — distinct ground contacts, slightly flared
    for x, z in [(-0.42, 0.22), (-0.42, -0.22), (0.38, 0.22), (0.38, -0.22)]:
        box([x, 0.12, z], [0.18, 0.32, 0.18])
    # Paws (widen feet for base stability + leg readability)
    for x, z in [(-0.42, 0.22), (-0.42, -0.22), (0.38, 0.22), (0.38, -0.22)]:
        box([x, 0.02, z], [0.22, 0.08, 0.22])
    # Tail
    box([-0.78, 0.55, 0.0], [0.38, 0.14, 0.12])
    box([-0.98, 0.68, 0.0], [0.16, 0.16, 0.1])

    dog = trimesh.util.concatenate(parts)
    # Center on origin, Y-up already
    dog.vertices -= dog.bounds.mean(axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(obj_path)) or ".", exist_ok=True)
    dog.export(obj_path)
    return obj_path


def main():
    ap = argparse.ArgumentParser(description="Run PerlerBeadMe kid_mode pipeline")
    ap.add_argument("--image", required=True, help="Input photo path")
    ap.add_argument("--mesh", default=None, help="Input .obj path (optional if --procedural-dog)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--footprint", type=int, default=18)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--max-colors", type=int, default=5)
    ap.add_argument("--yaw", type=float, default=None, help="Force yaw degrees (skip auto)")
    ap.add_argument("--no-auto-yaw", action="store_true")
    ap.add_argument("--up-axis", type=int, default=None, choices=[0, 1, 2],
                    help="Force pancake up axis (disables auto standing-axis)")
    ap.add_argument("--no-auto-up-axis", action="store_true",
                    help="Keep default/forced up_axis; do not detect tallest axis")
    ap.add_argument("--no-photo-yaw", action="store_true",
                    help="Use silhouette-only yaw (skip photo-matched composite)")
    ap.add_argument("--no-verify-gate", action="store_true")
    ap.add_argument("--verify-threshold", type=float, default=0.50,
                    help="Fail/retry when composite_error exceeds this (default 0.50)")
    ap.add_argument("--verify-strict", action="store_true",
                    help="Exit non-zero if verify gate fails after retries")
    ap.add_argument(
        "--procedural-dog",
        action="store_true",
        help="Use built-in standing-dog mesh (clear head/ears/legs) instead of --mesh",
    )
    ap.add_argument("--title", default="Kid Perler Dog")
    ap.add_argument(
        "--restyle",
        action="store_true",
        help="Run kid-friendly restyle before palette/voxel (photo → stylized reference)",
    )
    ap.add_argument(
        "--no-invent-body",
        action="store_true",
        help="With --restyle, do not composite head onto standing template",
    )
    ap.add_argument(
        "--triposr",
        action="store_true",
        help="Regenerate mesh via TripoSR HTTP from (restyled) image; falls back to procedural on failure",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mesh_path = args.mesh
    mesh_source = "provided"

    # Optional restyle first so TripoSR can consume the stylized plate
    guide_image = args.image
    if args.restyle:
        from kid_restyle import restyle_for_kid as _restyle
        restyled = os.path.join(args.out, "02_restyled.png")
        meta = _restyle(
            args.image,
            restyled,
            invent_body_if_head_only=not args.no_invent_body,
        )
        with open(os.path.join(args.out, "02_restyle_meta.json"), "w") as f:
            json.dump(meta.to_dict(), f, indent=2)
        guide_image = restyled
        print(f"pre-restyle -> {restyled} head_only={meta.head_only_detected}")

    if args.triposr:
        try:
            from voxelizer import HuggingFace3DEstimator
            mesh_path = os.path.join(args.out, "02_mesh_triposr.obj")
            HuggingFace3DEstimator().generate_3d_mesh(guide_image, mesh_path)
            mesh_source = "triposr"
            print(f"TripoSR mesh -> {mesh_path}")
        except Exception as e:
            print(f"TripoSR failed ({e}); falling back to procedural standing dog")
            mesh_path = None
            mesh_source = "triposr_failed"

    if args.procedural_dog or mesh_path is None:
        mesh_path = os.path.join(args.out, "02_mesh_procedural_dog.obj")
        make_standing_dog_mesh(mesh_path)
        mesh_source = "procedural_dog" if mesh_source != "triposr_failed" else "procedural_fallback"
        print(f"Wrote procedural standing dog mesh: {mesh_path}")
    elif not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)

    params = KidModeParams(
        max_footprint=args.footprint,
        max_layers=args.layers,
        hollow=False,
        thick_shell_wall=0,
        base_widen_layers=2,
        base_widen_iters=1,
        max_colors=args.max_colors,
        yaw_deg=args.yaw,
        auto_yaw=not args.no_auto_yaw and args.yaw is None,
        up_axis=args.up_axis if args.up_axis is not None else 1,
        auto_up_axis=not args.no_auto_up_axis and args.up_axis is None,
        photo_matched_yaw=not args.no_photo_yaw,
        verify_gate=not args.no_verify_gate,
        verify_error_threshold=args.verify_threshold,
        verify_retry_alt_axes=not args.no_verify_gate,
    )
    # Avoid double-restyle inside run_kid_mode if we already wrote 02_restyled
    do_restyle = args.restyle and not os.path.exists(os.path.join(args.out, "02_restyled.png"))
    summary = run_kid_mode(
        guide_image if args.restyle else args.image,
        mesh_path,
        args.out,
        params=params,
        title=args.title,
        restyle=do_restyle,
        invent_body=not args.no_invent_body,
    )
    # If we restyled externally, still copy original input + attach meta
    if args.restyle:
        raw = cv2.imread(args.image)
        if raw is not None:
            cv2.imwrite(os.path.join(args.out, "01_input.jpg"), raw)
        meta_path = os.path.join(args.out, "02_restyle_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                summary["restyle"] = json.load(f)
        summary["mesh_source"] = mesh_source
        with open(os.path.join(args.out, "00_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    else:
        summary["mesh_source"] = mesh_source
        with open(os.path.join(args.out, "00_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if args.verify_strict:
        gate = summary.get("verify_gate") or {}
        if gate.get("enabled") and not gate.get("passed"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
