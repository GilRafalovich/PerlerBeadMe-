#!/usr/bin/env python3
"""
Kid-mode Perler prototype: stacked pancake figures like Pokémon Perler sculptures.

Goals vs old demo_dog (60x60x4 hollow, 2481 beads):
- Smaller footprint (~28-36)
- Horizontal layers along UP (TripoSR Y), not thin Z-relief
- Solid layers (no aggressive hollowing) so silhouette stays readable
- Wide stable base (dilate bottom layer)
- Assembly instructions a parent/kid can follow
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont

from color_quantizer import PERLER_PALETTE, ColorQuantizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_dog")
OUT_DIR = os.path.join(os.path.dirname(__file__), "demo_dog_kid")

OLD_SUMMARY = {
    "mode": "old_default",
    "voxel_shape": [60, 60, 4],
    "occupied_beads": 2481,
    "beads_per_layer": [1019, 550, 533, 379],
    "hollow": True,
    "notes": "60x60x4 hollow relief; too many beads; hollow shells hard for kids",
}


@dataclass
class KidModeParams:
    # Target longest horizontal extent in beads (XZ plane after Y-up)
    max_footprint: int = 30
    # Cap height layers (merge if taller). None = keep native height.
    max_layers: Optional[int] = 12
    # Min layers after merge
    min_layers: int = 6
    hollow: bool = False
    # Mild core carve (0=solid). Keeps silhouette; base layers stay solid.
    thick_shell_wall: int = 0
    # Morphological widen of bottom N layers for stable feet
    base_widen_layers: int = 2
    base_widen_iters: int = 1
    # Drop tiny floating components / dust
    min_component_beads: int = 3
    # Color: project from quantized photo (front-ish) or dominant palette
    color_mode: str = "image_project"  # or "palette_majority"
    bead_mm: float = 5.0  # standard Perler ~5mm
    up_axis: int = 1  # TripoSR Y-up


# ---------------------------------------------------------------------------
# Voxelization helpers
# ---------------------------------------------------------------------------

def _reorient_y_up_matrix(matrix: np.ndarray, up_axis: int) -> np.ndarray:
    """Return matrix with shape (X, Z, Y_up) so[:,:,k] is a horizontal pancake."""
    # Move up_axis to last
    if up_axis == 2:
        return matrix  # already (x,y,z_up) — treat as (x,y,up)
    if up_axis == 1:
        # (x,y_up,z) -> (x,z,y_up)
        return np.transpose(matrix, (0, 2, 1))
    if up_axis == 0:
        # (x_up,y,z) -> (y,z,x_up)
        return np.transpose(matrix, (1, 2, 0))
    raise ValueError(up_axis)


def voxelize_kid(mesh: trimesh.Trimesh, params: KidModeParams) -> np.ndarray:
    extents = mesh.extents
    # Horizontal plane = the two axes that are NOT up
    horiz = [extents[i] for i in range(3) if i != params.up_axis]
    max_horiz = max(horiz) if horiz else float(np.max(extents))
    pitch = max_horiz / max(1, params.max_footprint)

    vg = mesh.voxelized(pitch=pitch).fill()
    raw = vg.matrix.astype(np.uint8)
    pancakes = _reorient_y_up_matrix(raw, params.up_axis)  # (fx, fz, h)

    # Crop to content bbox with 1-cell padding
    occupied = np.argwhere(pancakes > 0)
    if occupied.size == 0:
        return pancakes
    mins = occupied.min(axis=0)
    maxs = occupied.max(axis=0)
    pad = 1
    slices = []
    for d in range(3):
        lo = max(0, int(mins[d]) - pad)
        hi = min(pancakes.shape[d], int(maxs[d]) + pad + 1)
        slices.append(slice(lo, hi))
    pancakes = pancakes[slices[0], slices[1], slices[2]]

    # Optionally merge height into fewer structural plates
    h = pancakes.shape[2]
    target_h = h
    if params.max_layers is not None and h > params.max_layers:
        target_h = params.max_layers
    if target_h < params.min_layers and h >= params.min_layers:
        target_h = params.min_layers
    if target_h != h:
        pancakes = _squash_height(pancakes, target_h)

    if params.hollow:
        pancakes = _mild_hollow(pancakes)
    elif getattr(params, 'thick_shell_wall', 0) > 0:
        pancakes = _thick_shell(pancakes, wall=params.thick_shell_wall, solid_base_layers=max(1, params.base_widen_layers))

    pancakes = _widen_base(
        pancakes,
        n_layers=params.base_widen_layers,
        iterations=params.base_widen_iters,
    )
    pancakes = _drop_dust(pancakes, params.min_component_beads)
    return pancakes.astype(np.uint8)


def _squash_height(matrix: np.ndarray, target_h: int) -> np.ndarray:
    """OR-merge contiguous Z bands into target_h plates (preserves silhouette)."""
    fx, fz, h = matrix.shape
    out = np.zeros((fx, fz, target_h), dtype=np.uint8)
    for zi in range(target_h):
        start = int(zi * h / target_h)
        end = int((zi + 1) * h / target_h)
        if end <= start:
            end = start + 1
        end = min(end, h)
        out[:, :, zi] = np.max(matrix[:, :, start:end], axis=2)
    return out



def _thick_shell(matrix: np.ndarray, wall: int = 1, solid_base_layers: int = 2) -> np.ndarray:
    """Carve only deep interior cores per layer; keep outer silhouette and solid base."""
    out = matrix.copy()
    kernel = np.ones((3, 3), np.uint8)
    for z in range(matrix.shape[2]):
        if z < solid_base_layers:
            continue
        layer = matrix[:, :, z].astype(np.uint8)
        if int(layer.sum()) == 0:
            continue
        eroded = layer.copy()
        for _ in range(max(1, wall)):
            eroded = cv2.erode(eroded, kernel, iterations=1)
        if int(eroded.sum()) == 0:
            continue
        core = cv2.erode(eroded, kernel, iterations=1)
        if int(core.sum()) == 0:
            continue
        layer2 = layer.copy()
        layer2[core > 0] = 0
        out[:, :, z] = layer2
    return out

def _mild_hollow(matrix: np.ndarray) -> np.ndarray:
    """Only carve fully surrounded interior cells; keep all exterior silhouette."""
    # Same as current hollow — avoided in kid mode by default.
    hollowed = matrix.copy()
    for x in range(1, matrix.shape[0] - 1):
        for y in range(1, matrix.shape[1] - 1):
            for z in range(1, matrix.shape[2] - 1):
                if matrix[x, y, z] == 1:
                    if (
                        matrix[x - 1, y, z]
                        and matrix[x + 1, y, z]
                        and matrix[x, y - 1, z]
                        and matrix[x, y + 1, z]
                        and matrix[x, y, z - 1]
                        and matrix[x, y, z + 1]
                    ):
                        hollowed[x, y, z] = 0
    return hollowed


def _widen_base(matrix: np.ndarray, n_layers: int = 2, iterations: int = 1) -> np.ndarray:
    """Dilate bottom layers so feet form a wider stable platform."""
    out = matrix.copy()
    n = min(n_layers, out.shape[2])
    kernel = np.ones((3, 3), np.uint8)
    for z in range(n):
        layer = out[:, :, z].astype(np.uint8)
        if layer.sum() == 0:
            continue
        dilated = cv2.dilate(layer, kernel, iterations=iterations)
        out[:, :, z] = dilated
    return out


def _drop_dust(matrix: np.ndarray, min_beads: int) -> np.ndarray:
    """Remove tiny disconnected blobs per layer."""
    out = matrix.copy()
    for z in range(out.shape[2]):
        layer = out[:, :, z].astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(layer, connectivity=8)
        cleaned = np.zeros_like(layer)
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] >= min_beads:
                cleaned[labels == lab] = 1
        out[:, :, z] = cleaned
    return out


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

def build_color_volume(
    voxel: np.ndarray,
    img_rgb: np.ndarray,
    quantized: np.ndarray,
    params: KidModeParams,
) -> np.ndarray:
    """
    Assign Perler RGB to each occupied voxel.
    Image-project: map XZ of each layer from a downscaled quantized photo
    (good enough for prototype; bake better UV/vertex colors later).
    """
    fx, fz, h = voxel.shape
    colors = np.zeros((fx, fz, h, 3), dtype=np.uint8)

    # Resize quantized image to footprint
    q = cv2.resize(quantized, (fz, fx), interpolation=cv2.INTER_AREA)

    # Slight vertical color banding from image rows (head vs body)
    qh, qw = quantized.shape[:2]
    for z in range(h):
        # Map layer z to a horizontal band of the photo (bottom feet -> lower image)
        # Sitting dog photo: bottom of image ~= body/water, top ~= head — invert-ish
        row_frac = 1.0 - (z + 0.5) / max(1, h)
        y0 = int(row_frac * (qh - 1))
        band = quantized[max(0, y0 - qh // 12) : min(qh, y0 + qh // 12) + 1]
        if band.size == 0:
            band = quantized
        band_small = cv2.resize(band, (fz, fx), interpolation=cv2.INTER_AREA)
        mask = voxel[:, :, z] > 0
        colors[:, :, z][mask] = band_small[mask]
    return colors


def nearest_palette_name(rgb: tuple[int, int, int], quantizer: ColorQuantizer) -> str:
    rgb_arr = np.array([[rgb]], dtype=np.uint8)
    q = quantizer.quantize(rgb_arr)[0, 0]
    for name, val in PERLER_PALETTE.items():
        if tuple(val) == tuple(int(c) for c in q):
            return name
    # fallback nearest
    diffs = np.sum((quantizer.palette_rgb.astype(int) - np.array(q, dtype=int)) ** 2, axis=1)
    return quantizer.color_names[int(np.argmin(diffs))]


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------

def estimate_difficulty(total_beads: int, layers: int, colors_used: int) -> dict:
    """Kid-mode heuristics: ≤800 beads report Easy/Medium (~45–60 min).

    Epic 3: kits at kid scale must not show Hard solely because layer count
    exceeds the old ≤10 Medium cap — bead budget is the primary signal.
    """
    if total_beads <= 400 and layers <= 10 and colors_used <= 6:
        level, minutes = "Easy", 45
    elif total_beads <= 800 and layers <= 20 and colors_used <= 8:
        # Medium band: stretch ≤600 → ~45 min; up to 800 → ~60 min
        level, minutes = "Medium", 45 if total_beads <= 600 else 60
    elif total_beads <= 1200 and layers <= 20:
        level, minutes = "Hard (ask a grown-up)", 75
    else:
        level, minutes = "Hard (ask a grown-up)", 90
    return {
        "difficulty": level,
        "estimated_minutes": minutes,
        "estimated_time_label": f"about {minutes} minutes",
    }


def layer_bead_counts(colors: np.ndarray, voxel: np.ndarray, quantizer: ColorQuantizer):
    """Per-layer and global color tallies."""
    per_layer = []
    global_counts: Counter = Counter()
    fx, fz, h = voxel.shape
    for z in range(h):
        c = Counter()
        for x in range(fx):
            for y in range(fz):
                if voxel[x, y, z]:
                    rgb = tuple(int(v) for v in colors[x, y, z])
                    # snap to palette name
                    name = nearest_palette_name(rgb, quantizer)
                    c[name] += 1
                    global_counts[name] += 1
        per_layer.append({"layer_index": z, "beads": int(sum(c.values())), "by_color": dict(c)})
    return per_layer, dict(global_counts)


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
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = font_sm = font_lg = ImageFont.load_default()

    step = layer_idx + 1
    line1 = f"STEP {step} of {total_layers}  —  Layer {step}"
    if layer_idx == 0:
        line1 += "  ★ BASE / FEET"
    line2 = "Build from the BOTTOM up  ·  stack this sheet, then the next"
    draw.text((margin, 14), line1, fill=(20, 20, 20), font=font_lg)
    draw.text((margin, 48), line2, fill=(60, 60, 60), font=font)

    ox, oy = margin, title_h
    # Draw cells
    for i in range(fx):
        for j in range(fz):
            x0 = ox + j * cell
            y0 = oy + i * cell
            if voxel_layer[i, j]:
                rgb = tuple(int(v) for v in color_layer[i, j])
                draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=rgb, outline=(60, 60, 60))
            else:
                draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=(245, 245, 245), outline=(220, 220, 220))

    # Legend
    lx = ox + grid_w + 24
    ly = oy
    draw.text((lx, ly), "Color key", fill=(0, 0, 0), font=font)
    ly += 28
    beads_here = int(voxel_layer.sum())
    draw.text((lx, ly), f"Beads this layer: {beads_here}", fill=(40, 40, 40), font=font_sm)
    ly += 24
    for name, count in sorted(color_counts.items(), key=lambda kv: -kv[1]):
        rgb = PERLER_PALETTE[name]
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
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = font_lg = ImageFont.load_default()
    draw.text((24, 20), f"Shopping list — {total} beads total", fill=(0, 0, 0), font=font_lg)
    y = 70
    for name, count in sorted(global_counts.items(), key=lambda kv: -kv[1]):
        rgb = PERLER_PALETTE[name]
        draw.rectangle([24, y, 24 + 36, y + 36], fill=rgb, outline=(0, 0, 0))
        draw.text((72, y + 6), f"{name}: {count} beads", fill=(0, 0, 0), font=font)
        y += 48
    img.save(out_path)


def render_voxel_preview(voxel: np.ndarray, colors: np.ndarray, out_path: str, title: str):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    # matplotlib voxels expects (x,y,z); our matrix is (fx,fz,h)
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
    # Equal-ish aspect
    max_range = max(voxel.shape)
    ax.set_box_aspect((voxel.shape[0] / max_range, voxel.shape[1] / max_range, voxel.shape[2] / max_range * 1.2))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_instructions_md(path: str, summary: dict, per_layer: list, global_counts: dict):
    diff = summary["difficulty"]
    lines = [
        f"# Kid Perler Build — Dog",
        "",
        f"**Difficulty:** {diff['difficulty']}  ",
        f"**Time:** {diff['estimated_time_label']}  ",
        f"**Total beads:** {summary['occupied_beads']}  ",
        f"**Layers:** {summary['num_layers']} (stack bottom → top)  ",
        f"**Grid:** {summary['voxel_shape'][0]}×{summary['voxel_shape'][1]} footprint, {summary['num_layers']} high  ",
        f"**Approx size:** ~{summary['approx_size_mm']['width_mm']:.0f}×{summary['approx_size_mm']['depth_mm']:.0f}×{summary['approx_size_mm']['height_mm']:.0f} mm",
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
        "Like the Pokémon Perler figures: pancake layers with a wide base.",
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
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_instructions_pdf(pdf_path: str, sheet_paths: list[str], color_key_path: str, md_summary_lines: list[str]):
    with PdfPages(pdf_path) as pdf:
        # Cover
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.9, "Kid Perler Build Instructions", fontsize=20, weight="bold")
        y = 0.82
        for line in md_summary_lines[:18]:
            fig.text(0.1, y, line[:100], fontsize=10, family="monospace")
            y -= 0.035
        pdf.savefig(fig)
        plt.close(fig)

        # Color key
        if os.path.exists(color_key_path):
            img = plt.imread(color_key_path)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.08, 0.2, 0.84, 0.7])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.1, 0.92, "Shopping list / color key", fontsize=16, weight="bold")
            pdf.savefig(fig)
            plt.close(fig)

        for p in sheet_paths:
            img = plt.imread(p)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.05, 0.1, 0.9, 0.82])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(params: Optional[KidModeParams] = None):
    params = params or KidModeParams()
    os.makedirs(OUT_DIR, exist_ok=True)

    img_path = os.path.join(DEMO_DIR, "01_input.jpg")
    mesh_path = os.path.join(DEMO_DIR, "02_mesh.obj")
    if not os.path.exists(mesh_path):
        mesh_path = os.path.join(DEMO_DIR, "mesh.obj")

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    quantizer = ColorQuantizer()
    quantized = quantizer.quantize(img_rgb)
    cv2.imwrite(
        os.path.join(OUT_DIR, "03_palette_quantized.png"),
        cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR),
    )

    mesh = trimesh.load(mesh_path, force="mesh")
    voxel = voxelize_kid(mesh, params)
    colors = build_color_volume(voxel, img_rgb, quantized, params)

    # Snap colors onto palette for consistency in sheets
    flat = colors.reshape(-1, 3)
    snapped = quantizer.quantize(flat.reshape(-1, 1, 3).astype(np.uint8)).reshape(-1, 3)
    colors = snapped.reshape(colors.shape)

    np.save(os.path.join(OUT_DIR, "04_voxel_matrix.npy"), voxel)
    np.save(os.path.join(OUT_DIR, "04_color_volume.npy"), colors)

    occupied = int(voxel.sum())
    beads_per_layer = [int(voxel[:, :, z].sum()) for z in range(voxel.shape[2])]
    per_layer, global_counts = layer_bead_counts(colors, voxel, quantizer)
    diff = estimate_difficulty(occupied, voxel.shape[2], len(global_counts))

    approx = {
        "width_mm": voxel.shape[0] * params.bead_mm,
        "depth_mm": voxel.shape[1] * params.bead_mm,
        "height_mm": voxel.shape[2] * params.bead_mm,
    }

    # Layer instruction PNGs + masks + simple color layers
    sheet_paths = []
    for z in range(voxel.shape[2]):
        # instruction sheet (print-friendly)
        sheet = os.path.join(OUT_DIR, f"layer_{z:02d}_instruction.png")
        render_instruction_sheet(
            z,
            voxel.shape[2],
            voxel[:, :, z],
            colors[:, :, z],
            per_layer[z]["by_color"],
            sheet,
        )
        sheet_paths.append(sheet)

        # also save compact preview like old demo
        layer_rgb = np.zeros((voxel.shape[0], voxel.shape[1], 3), dtype=np.uint8)
        layer_rgb[:] = 40
        mask = voxel[:, :, z] > 0
        layer_rgb[mask] = colors[:, :, z][mask]
        # upscale for visibility
        up = cv2.resize(layer_rgb, (voxel.shape[1] * 8, voxel.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT_DIR, f"06_layer_{z:02d}.png"), cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            os.path.join(OUT_DIR, f"06_layer_{z:02d}_mask.png"),
            (mask.astype(np.uint8) * 255),
        )

    color_key_path = os.path.join(OUT_DIR, "00_color_key.png")
    render_color_key_poster(global_counts, occupied, color_key_path)

    preview_path = os.path.join(OUT_DIR, "07_voxel_preview_3d.png")
    render_voxel_preview(
        voxel,
        colors,
        preview_path,
        f"Kid mode — {voxel.shape[0]}x{voxel.shape[1]}x{voxel.shape[2]} solid ({occupied} beads)",
    )

    # Assembly overview strip
    n = voxel.shape[2]
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for z, ax in enumerate(axes):
        rgb = np.zeros((voxel.shape[0], voxel.shape[1], 3), dtype=np.uint8) + 255
        m = voxel[:, :, z] > 0
        rgb[m] = colors[:, :, z][m]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"L{z+1}" + (" BASE" if z == 0 else ""), fontsize=9)
        ax.axis("off")
    plt.suptitle("Assembly order: bottom (L1) → top", fontsize=12)
    plt.tight_layout()
    overview_path = os.path.join(OUT_DIR, "08_assembly_overview.png")
    plt.savefig(overview_path, dpi=140)
    plt.close()

    kid_summary = {
        "mode": "kid_mode",
        "params": asdict(params),
        "voxel_shape": list(voxel.shape),
        "num_layers": int(voxel.shape[2]),
        "occupied_beads": occupied,
        "beads_per_layer": beads_per_layer,
        "colors_used": global_counts,
        "num_colors": len(global_counts),
        "hollow": params.hollow,
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
            "assembly_overview": "08_assembly_overview.png",
        },
    }

    comparison = {
        "old": OLD_SUMMARY,
        "kid": {
            "voxel_shape": kid_summary["voxel_shape"],
            "occupied_beads": occupied,
            "beads_per_layer": beads_per_layer,
            "hollow": False,
            "num_layers": kid_summary["num_layers"],
            "num_colors": kid_summary["num_colors"],
            "difficulty": diff,
        },
        "delta": {
            "bead_reduction": OLD_SUMMARY["occupied_beads"] - occupied,
            "bead_ratio_kid_over_old": round(occupied / OLD_SUMMARY["occupied_beads"], 3),
            "why_kid_easier": [
                "Smaller footprint (pegboard-friendly)",
                "Solid layers keep silhouette (no hollow shells)",
                "Horizontal pancakes along up-axis (freestanding stack)",
                "Widened base layer for stability",
                "Fewer total beads + print-ready step sheets",
            ],
        },
        "recommended_code_changes": [
            {
                "where": "quantizer.py VoxelQuantizer",
                "change": "Add kid_mode flag / KidModeParams: max_footprint, max_layers, hollow=False, up_axis, base_widen_*",
                "type": "params + algorithm",
            },
            {
                "where": "quantizer.py spatial_quantize",
                "change": "Reorient mesh so TripoSR Y-up becomes layer axis; stop squashing mesh-Z into 4 relief slices for kid builds",
                "type": "algorithm",
            },
            {
                "where": "quantizer.py _hollow_out",
                "change": "Skip hollowing in kid_mode (destroys silhouette / creates fragile rings)",
                "type": "params",
            },
            {
                "where": "new: base widen + tight bbox crop",
                "change": "Dilate bottom 1–2 layers; crop empty margins — silhouette/base stability",
                "type": "algorithm",
            },
            {
                "where": "hybrid_assembler.py",
                "change": "Emit colored layouts, per-color counts, step order, difficulty/time — not just binary masks",
                "type": "algorithm",
            },
            {
                "where": "main.py /api/process_image",
                "change": "Accept kid_mode=true query/body; return instructions + shopping list + layer sheet URLs",
                "type": "params/API",
            },
            {
                "where": "new instruction renderer",
                "change": "Bake layer_XX_instruction.png + instructions.pdf/md generation into pipeline (this prototype)",
                "type": "new module",
            },
        ],
    }

    instructions_json = {
        "title": "Kid Perler Dog",
        "assembly_order": "bottom_to_top",
        "total_beads": occupied,
        "layers": per_layer,
        "shopping_list": global_counts,
        "difficulty": diff,
        "tips": [
            "Start with Layer 1 (base). Keep it wide so the figure stands.",
            "Adult irons each layer. Cool fully before stacking the next.",
            "Match bead colors to the color key on each sheet.",
        ],
    }

    with open(os.path.join(OUT_DIR, "instructions.json"), "w") as f:
        json.dump(instructions_json, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"kid": kid_summary, "comparison": comparison}, f, indent=2)
    with open(os.path.join(OUT_DIR, "00_summary.json"), "w") as f:
        json.dump(kid_summary, f, indent=2)

    write_instructions_md(
        os.path.join(OUT_DIR, "instructions.md"),
        kid_summary,
        per_layer,
        global_counts,
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
        os.path.join(OUT_DIR, "instructions.pdf"),
        sheet_paths,
        color_key_path,
        cover_lines,
    )

    # Also copy input ref for convenience
    cv2.imwrite(os.path.join(OUT_DIR, "01_input.jpg"), img_bgr)

    print("=== KID MODE DONE ===")
    print(f"shape={voxel.shape} beads={occupied} layers={voxel.shape[2]}")
    print(f"per_layer={beads_per_layer}")
    print(f"colors={global_counts}")
    print(f"difficulty={diff}")
    print(f"out={OUT_DIR}")
    return kid_summary, comparison


def tune_for_target(target_lo=400, target_hi=800):
    """Pick footprint/layers/shell to land near the kid bead budget."""
    mesh = trimesh.load(os.path.join(DEMO_DIR, "02_mesh.obj"), force="mesh")
    best = None
    # Prefer recognizable size with 6–10 layers; mild shell OK if solid overshoots.
    for fp, ml, wall in [
        (18, 8, 1), (16, 8, 0), (16, 8, 1), (18, 6, 0), (16, 6, 0),
        (20, 8, 1), (14, 8, 0), (18, 10, 1),
    ]:
        p = KidModeParams(
            max_footprint=fp,
            max_layers=ml,
            hollow=False,
            thick_shell_wall=wall,
            base_widen_layers=2,
            base_widen_iters=2,  # wider feet
        )
        v = voxelize_kid(mesh, p)
        n = int(v.sum())
        print(f"trial fp={fp} L={ml} wall={wall} -> {v.shape} beads={n} base={int(v[:,:,0].sum())}")
        score = abs(n - 650) + (0 if 400 <= n <= 800 else 500)
        # prefer more layers + mild/no shell for silhouette
        score -= ml * 5
        score += wall * 20
        if best is None or score < best[2]:
            best = (p, n, score)
        if target_lo <= n <= target_hi and wall <= 1 and ml >= 8:
            return p, n
    return (best[0], best[1]) if best else (KidModeParams(), -1)


if __name__ == "__main__":
    # Proven kid defaults for demo_dog: solid pancakes, widened base, ~750 beads
    params = KidModeParams(
        max_footprint=14,
        max_layers=8,
        hollow=False,
        thick_shell_wall=0,
        base_widen_layers=2,
        base_widen_iters=2,
    )
    print(f"Using kid defaults: {params}")
    run(params)
