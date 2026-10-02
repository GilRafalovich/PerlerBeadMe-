"""Kid-mode instruction artifacts: layer PNGs, shopping list, PDF/MD/JSON."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont

from color_quantizer import PERLER_PALETTE


def _fonts():
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_lg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
        return font, font_sm, font_lg
    except Exception:
        d = ImageFont.load_default()
        return d, d, d


def render_instruction_sheet(
    layer_idx: int,
    total_layers: int,
    voxel_layer: np.ndarray,
    color_layer: np.ndarray,
    color_counts: dict,
    out_path: str,
    cell: int = 18,
):
    """Print-friendly numbered layer sheet with grid + legend."""
    fx, fz = voxel_layer.shape
    margin = 40
    legend_w = 240
    title_h = 90
    footer_h = 56
    grid_w = fz * cell
    grid_h = fx * cell
    W = max(720, margin * 2 + grid_w + legend_w)
    H = title_h + margin + grid_h + footer_h

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font, font_sm, font_lg = _fonts()

    step = layer_idx + 1
    line1 = f"STEP {step} of {total_layers}  —  Layer {step}"
    if layer_idx == 0:
        line1 += "  ★ BASE / FEET"
    line2 = "Build from the BOTTOM up  ·  stack this sheet, then the next"
    draw.text((margin, 14), line1, fill=(20, 20, 20), font=font_lg)
    draw.text((margin, 48), line2, fill=(60, 60, 60), font=font)

    ox, oy = margin, title_h
    for i in range(fx):
        for j in range(fz):
            x0 = ox + j * cell
            y0 = oy + i * cell
            if voxel_layer[i, j]:
                rgb = tuple(int(v) for v in color_layer[i, j])
                draw.rectangle(
                    [x0, y0, x0 + cell - 1, y0 + cell - 1], fill=rgb, outline=(60, 60, 60)
                )
            else:
                draw.rectangle(
                    [x0, y0, x0 + cell - 1, y0 + cell - 1],
                    fill=(245, 245, 245),
                    outline=(220, 220, 220),
                )

    lx = ox + grid_w + 24
    ly = oy
    draw.text((lx, ly), "Color key", fill=(0, 0, 0), font=font)
    ly += 28
    beads_here = int(voxel_layer.sum())
    draw.text((lx, ly), f"Beads this layer: {beads_here}", fill=(40, 40, 40), font=font_sm)
    ly += 24
    for name, count in sorted(color_counts.items(), key=lambda kv: -kv[1]):
        rgb = PERLER_PALETTE.get(name, (128, 128, 128))
        draw.rectangle([lx, ly, lx + 22, ly + 22], fill=rgb, outline=(0, 0, 0))
        draw.text((lx + 30, ly + 2), f"{name}: {count}", fill=(0, 0, 0), font=font_sm)
        ly += 28

    tip = "Iron this layer flat (adult help!), cool, then stack the next layer on top and fuse."
    if layer_idx == 0:
        tip = "This is the BASE — make it wide and solid so your figure stands up!"
    draw.text((margin, H - 36), tip, fill=(80, 80, 80), font=font_sm)
    img.save(out_path)
    return out_path


def render_color_key_poster(global_counts: dict, total: int, out_path: str):
    rows = max(1, len(global_counts))
    W, H = 640, 80 + rows * 48 + 40
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _, font, font_lg = _fonts()
    try:
        font_lg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        pass
    draw.text((24, 20), f"Shopping list — {total} beads total", fill=(0, 0, 0), font=font_lg)
    y = 70
    for name, count in sorted(global_counts.items(), key=lambda kv: -kv[1]):
        rgb = PERLER_PALETTE.get(name, (128, 128, 128))
        draw.rectangle([24, y, 24 + 36, y + 36], fill=rgb, outline=(0, 0, 0))
        draw.text((72, y + 6), f"{name}: {count} beads", fill=(0, 0, 0), font=font)
        y += 48
    img.save(out_path)


def render_voxel_preview(voxel: np.ndarray, colors: np.ndarray, out_path: str, title: str):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    filled = voxel.astype(bool)
    facecolors = np.zeros(voxel.shape + (4,), dtype=float)
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if filled[x, y, z]:
                    rgb = colors[x, y, z] / 255.0
                    facecolors[x, y, z] = (*rgb, 1.0)
    ax.voxels(filled, facecolors=facecolors, edgecolor="k", linewidth=0.15)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z (depth)")
    ax.set_zlabel("Layer (up)")
    max_range = max(voxel.shape)
    ax.set_box_aspect(
        (
            voxel.shape[0] / max_range,
            voxel.shape[1] / max_range,
            voxel.shape[2] / max_range * 1.2,
        )
    )
    # Prefer a 3/4 view that reads silhouette (head left/high, legs down)
    ax.view_init(elev=18, azim=-55)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def render_side_silhouette(voxel: np.ndarray, colors: np.ndarray, out_path: str, title: str):
    """Orthographic side silhouette (X vs height) — kid-readable outline."""
    fx, fz, h = voxel.shape
    # Project along Z (depth): max over depth for occupancy + majority color
    sil = np.zeros((h, fx, 3), dtype=np.uint8) + 255
    for x in range(fx):
        for zi in range(h):
            col = colors[x, :, zi]
            mask = voxel[x, :, zi] > 0
            if not np.any(mask):
                continue
            # majority color along depth
            pts = col[mask]
            # pick median RGB as robust color
            rgb = np.median(pts, axis=0).astype(np.uint8)
            # image row 0 = top layer
            sil[h - 1 - zi, x] = rgb
    # Upscale nearest for visibility
    scale = max(12, 240 // max(fx, h))
    up = cv2.resize(sil, (fx * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    # Draw outline
    gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY)
    occupied = (gray < 250).astype(np.uint8) * 255
    contours, _ = cv2.findContours(occupied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = up.copy()
    cv2.drawContours(outlined, contours, -1, (20, 20, 20), 2)
    # Title bar
    canvas = np.ones((outlined.shape[0] + 40, outlined.shape[1], 3), dtype=np.uint8) * 255
    canvas[40:] = outlined
    cv2.putText(
        canvas,
        title[:80],
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_path


def render_assembly_overview(
    voxel: np.ndarray, colors: np.ndarray, out_path: str
):
    n = voxel.shape[2]
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for z, ax in enumerate(axes):
        rgb = np.zeros((voxel.shape[0], voxel.shape[1], 3), dtype=np.uint8) + 255
        m = voxel[:, :, z] > 0
        rgb[m] = colors[:, :, z][m]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"L{z + 1}" + (" BASE" if z == 0 else ""), fontsize=9)
        ax.axis("off")
    plt.suptitle("Assembly order: bottom (L1) → top", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def write_instructions_md(
    path: str,
    summary: dict,
    per_layer: list,
    global_counts: dict,
    title: str = "Kid Perler Build",
):
    diff = summary["difficulty"]
    shape = summary["voxel_shape"]
    lines = [
        f"# {title}",
        "",
        f"**Difficulty:** {diff['difficulty']}  ",
        f"**Time:** {diff['estimated_time_label']}  ",
        f"**Total beads:** {summary['occupied_beads']}  ",
        f"**Layers:** {summary['num_layers']} (stack bottom → top)  ",
        f"**Grid:** {shape[0]}×{shape[1]} footprint, {summary['num_layers']} high  ",
        f"**Approx size:** ~{summary['approx_size_mm']['width_mm']:.0f}×"
        f"{summary['approx_size_mm']['depth_mm']:.0f}×"
        f"{summary['approx_size_mm']['height_mm']:.0f} mm",
        "",
        "## What you need",
        "",
        "- Perler/Hama/Artkal beads in the colors below",
        "- Pegboard large enough for the footprint",
        "- Ironing paper + iron (**adult does the ironing**)",
        "- Optional: tweezers",
        "",
        "## Shopping list (by color)",
        "",
    ]
    for name, count in sorted(global_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{name}**: {count}")
    lines += [
        "",
        "## Assembly order (bottom → top)",
        "",
        "Build **Layer 1 first** (the base/feet). Iron each layer flat with an adult,",
        "let it cool, then stack the next layer on top and fuse the stack.",
        "",
    ]
    for i, layer in enumerate(per_layer):
        star = " ★ BASE" if i == 0 else ""
        lines.append(f"### Step {i + 1}: Layer {i + 1}{star}")
        lines.append(f"- Beads: **{layer['beads']}**")
        for name, count in sorted(layer["by_color"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {name}: {count}")
        lines.append(f"- Pattern sheet: `layer_{i:02d}_instruction.png`")
        lines.append("")
    lines += [
        "## Safety",
        "",
        "- Adults only for ironing (hot!).",
        "- Small beads are a choking hazard for toddlers — keep away from under-3s.",
        "",
        "## Files",
        "",
        "- `instructions.pdf` — printable packet",
        "- `layer_XX_instruction.png` — one sheet per layer",
        "- `00_color_key.png` — shopping list poster",
        "- `07_voxel_preview_3d.png` — finished look",
        "- `07_side_silhouette.png` — side outline (head/ears/legs check)",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_instructions_pdf(
    pdf_path: str,
    sheet_paths: list[str],
    color_key_path: str,
    cover_lines: list[str],
    extra_images: Optional[list[str]] = None,
):
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.9, "Kid Perler Build Instructions", fontsize=20, weight="bold")
        y = 0.82
        for line in cover_lines[:18]:
            fig.text(0.1, y, line[:100], fontsize=10, family="monospace")
            y -= 0.035
        pdf.savefig(fig)
        plt.close(fig)

        if os.path.exists(color_key_path):
            img = plt.imread(color_key_path)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.08, 0.2, 0.84, 0.7])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.1, 0.92, "Shopping list / color key", fontsize=16, weight="bold")
            pdf.savefig(fig)
            plt.close(fig)

        for p in (extra_images or []) + sheet_paths:
            if not os.path.exists(p):
                continue
            img = plt.imread(p)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.05, 0.1, 0.9, 0.82])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


def write_kid_artifacts(
    out_dir: str,
    voxel: np.ndarray,
    colors: np.ndarray,
    assemble_result: dict[str, Any],
    title: str = "Kid Perler Build",
    params_dict: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Write full kid-mode artifact pack into out_dir.
    Returns summary dict with artifact relative paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    occupied = int(assemble_result["total_beads"])
    per_layer = assemble_result["layers"]
    global_counts = assemble_result["shopping_list"]
    diff = assemble_result["difficulty"]
    approx = assemble_result["approx_size_mm"]

    sheet_paths = []
    for z in range(voxel.shape[2]):
        sheet = os.path.join(out_dir, f"layer_{z:02d}_instruction.png")
        render_instruction_sheet(
            z,
            voxel.shape[2],
            voxel[:, :, z],
            colors[:, :, z],
            per_layer[z]["by_color"],
            sheet,
        )
        sheet_paths.append(sheet)

        layer_rgb = np.zeros((voxel.shape[0], voxel.shape[1], 3), dtype=np.uint8)
        layer_rgb[:] = 40
        mask = voxel[:, :, z] > 0
        layer_rgb[mask] = colors[:, :, z][mask]
        up = cv2.resize(
            layer_rgb,
            (voxel.shape[1] * 8, voxel.shape[0] * 8),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(
            os.path.join(out_dir, f"06_layer_{z:02d}.png"),
            cv2.cvtColor(up, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(out_dir, f"06_layer_{z:02d}_mask.png"),
            (mask.astype(np.uint8) * 255),
        )

    color_key_path = os.path.join(out_dir, "00_color_key.png")
    render_color_key_poster(global_counts, occupied, color_key_path)

    preview_path = os.path.join(out_dir, "07_voxel_preview_3d.png")
    render_voxel_preview(
        voxel,
        colors,
        preview_path,
        f"Kid mode — {voxel.shape[0]}x{voxel.shape[1]}x{voxel.shape[2]} "
        f"({occupied} beads)",
    )

    side_path = os.path.join(out_dir, "07_side_silhouette.png")
    render_side_silhouette(
        voxel,
        colors,
        side_path,
        "Side silhouette (look for head / ears / legs)",
    )

    overview_path = os.path.join(out_dir, "08_assembly_overview.png")
    render_assembly_overview(voxel, colors, overview_path)

    np.save(os.path.join(out_dir, "04_voxel_matrix.npy"), voxel)
    np.save(os.path.join(out_dir, "04_color_volume.npy"), colors)

    summary = {
        "mode": "kid_mode",
        "title": title,
        "params": params_dict or {},
        "voxel_shape": list(voxel.shape),
        "num_layers": int(voxel.shape[2]),
        "occupied_beads": occupied,
        "beads_per_layer": assemble_result["beads_per_layer"],
        "colors_used": global_counts,
        "num_colors": len(global_counts),
        "hollow": False,
        "approx_size_mm": approx,
        "difficulty": diff,
        "assembly_order": "bottom_to_top",
        "base_layer": 0,
        "artifacts": {
            "instructions_md": "instructions.md",
            "instructions_json": "instructions.json",
            "instructions_pdf": "instructions.pdf",
            "color_key": "00_color_key.png",
            "layer_sheets": [os.path.basename(p) for p in sheet_paths],
            "voxel_preview": "07_voxel_preview_3d.png",
            "side_silhouette": "07_side_silhouette.png",
            "assembly_overview": "08_assembly_overview.png",
        },
    }

    instructions_json = {
        "title": title,
        "assembly_order": "bottom_to_top",
        "total_beads": occupied,
        "layers": per_layer,
        "steps": [
            {k: v for k, v in s.items() if k not in ("color_layout_rgb", "color_layout_names")}
            for s in assemble_result.get("steps", [])
        ],
        "shopping_list": global_counts,
        "difficulty": diff,
        "tips": assemble_result.get("tips", []),
    }

    with open(os.path.join(out_dir, "instructions.json"), "w") as f:
        json.dump(instructions_json, f, indent=2)
    with open(os.path.join(out_dir, "00_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"kid": summary}, f, indent=2)

    write_instructions_md(
        os.path.join(out_dir, "instructions.md"),
        summary,
        per_layer,
        global_counts,
        title=title,
    )

    cover_lines = [
        f"Total beads: {occupied}",
        f"Layers: {voxel.shape[2]} (bottom → top)",
        f"Grid: {voxel.shape[0]} x {voxel.shape[1]} x {voxel.shape[2]}",
        f"Difficulty: {diff['difficulty']} ({diff['estimated_time_label']})",
        f"Size ~ {approx['width_mm']:.0f}x{approx['depth_mm']:.0f}x{approx['height_mm']:.0f} mm",
        "",
        "Shopping list:",
    ] + [f"  {k}: {v}" for k, v in sorted(global_counts.items(), key=lambda kv: -kv[1])]

    write_instructions_pdf(
        os.path.join(out_dir, "instructions.pdf"),
        sheet_paths,
        color_key_path,
        cover_lines,
        extra_images=[side_path, overview_path],
    )

    return summary
