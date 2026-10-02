"""
Trivial dry-run / smoke entry for kid_mode (no commit).

Examples:
  # Restyle swimming photo → standing template → procedural mesh → kid kit
  python dry_run_kid.py --restyle --procedural-dog \\
      --image demo_dog/01_input.jpg --out demo_dog_restyle

  # Silhouette-first without restyle
  python dry_run_kid.py --procedural-dog \\
      --image demo_dog_kid_pipeline/ref_candidates/pixabay_side.jpg \\
      --out demo_dog_kid_pipeline

  # Existing demo_dog mesh (often blob from swimming photo)
  python dry_run_kid.py --image demo_dog/01_input.jpg --mesh demo_dog/02_mesh.obj \\
      --out demo_dog_kid_pipeline/from_old_mesh

  # Restyle + try TripoSR (falls back to procedural on failure)
  python dry_run_kid.py --restyle --triposr \\
      --image demo_dog/01_input.jpg --out demo_dog_restyle
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

from kid_pipeline import make_standing_dog_mesh, run_kid_mode
from kid_restyle import restyle_for_kid
from quantizer import KidModeParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--out", default="demo_dog_kid_pipeline")
    ap.add_argument("--procedural-dog", action="store_true")
    ap.add_argument("--restyle", action="store_true")
    ap.add_argument("--no-invent-body", action="store_true")
    ap.add_argument("--triposr", action="store_true", help="Try TripoSR HTTP; fallback procedural")
    ap.add_argument("--footprint", type=int, default=18)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--max-colors", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    guide_image = args.image
    restyle_meta = None

    if args.restyle:
        restyled = os.path.join(args.out, "02_restyled.png")
        restyle_meta = restyle_for_kid(
            args.image,
            restyled,
            invent_body_if_head_only=not args.no_invent_body,
        )
        with open(os.path.join(args.out, "02_restyle_meta.json"), "w") as f:
            json.dump(restyle_meta.to_dict(), f, indent=2)
        raw = cv2.imread(args.image)
        if raw is not None:
            cv2.imwrite(os.path.join(args.out, "01_input.jpg"), raw)
        guide_image = restyled
        print(
            f"restyle -> {restyled} method={restyle_meta.method} "
            f"head_only={restyle_meta.head_only_detected} body={restyle_meta.body_method}"
        )

    mesh = args.mesh
    mesh_source = "provided"
    if args.triposr:
        try:
            from voxelizer import HuggingFace3DEstimator

            mesh = os.path.join(args.out, "02_mesh_triposr.obj")
            HuggingFace3DEstimator().generate_3d_mesh(guide_image, mesh)
            mesh_source = "triposr"
            print(f"triposr mesh -> {mesh}")
        except Exception as e:
            print(f"TripoSR failed ({e}); procedural fallback")
            mesh = None
            mesh_source = "procedural_fallback"

    if args.procedural_dog or not mesh:
        mesh = os.path.join(args.out, "02_mesh_procedural_dog.obj")
        make_standing_dog_mesh(mesh)
        if mesh_source == "provided":
            mesh_source = "procedural_dog"
        print(f"procedural mesh -> {mesh}")

    params = KidModeParams(
        max_footprint=args.footprint,
        max_layers=args.layers,
        hollow=False,
        thick_shell_wall=0,
        base_widen_layers=2,
        base_widen_iters=1,
        max_colors=args.max_colors,
        auto_yaw=True,
    )
    summary = run_kid_mode(
        guide_image,
        mesh,
        args.out,
        params=params,
        title="Kid Perler Dog (restyle)" if args.restyle else "Kid Perler Dog",
        restyle=False,  # already done above
    )
    if restyle_meta is not None:
        summary["restyle"] = restyle_meta.to_dict()
        # Ensure original input is preserved
        raw = cv2.imread(args.image)
        if raw is not None:
            cv2.imwrite(os.path.join(args.out, "01_input.jpg"), raw)
    summary["mesh_source"] = mesh_source
    with open(os.path.join(args.out, "00_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("SMOKE OK", summary["voxel_shape"], summary["occupied_beads"], mesh_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
