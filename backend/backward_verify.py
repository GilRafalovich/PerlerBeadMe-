#!/usr/bin/env python3
"""
Backward verification: render 3D beads/mesh → compare to original/restyle → score.

Metrics (shape/structure only — color histogram deliberately excluded):
  - ssim: structural similarity on grayscale aligned views
  - lpips_lite: multi-scale LAB + edge feature L2 (lower better)
  - silhouette_iou: binary mask IoU after alignment
  - composite_error: weighted sum of (1-ssim), lpips_lite, (1-iou) for optimization (lower better)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:  # pragma: no cover
    ssim_fn = None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_voxel_ortho(
    voxel: np.ndarray,
    colors: np.ndarray,
    view: str = "front",
    out_size: int = 256,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Orthographic RGB render of pancake voxels (fx, fz, h).
    Views:
      front  — project along +Z (depth), image X←fx, Y↑h
      side   — project along +X, image X←fz, Y↑h
      top    — project along +h, image X←fx, Y←fz
      back / left / right — mirrored variants
    """
    fx, fz, h = voxel.shape
    if view in ("front", "back"):
        # Along depth Z: for each (x, zi) take nearest occupied from front or back
        w, ht = fx, h
        img = np.zeros((ht, w, 3), dtype=np.uint8)
        img[:] = bg
        z_order = range(fz) if view == "front" else range(fz - 1, -1, -1)
        for x in range(fx):
            for zi in range(h):
                for y in z_order:
                    if voxel[x, y, zi]:
                        img[ht - 1 - zi, x] = colors[x, y, zi]
                        break
    elif view in ("side", "left", "right"):
        # Along X
        w, ht = fz, h
        img = np.zeros((ht, w, 3), dtype=np.uint8)
        img[:] = bg
        x_order = range(fx) if view in ("side", "right") else range(fx - 1, -1, -1)
        for y in range(fz):
            for zi in range(h):
                for x in x_order:
                    if voxel[x, y, zi]:
                        img[ht - 1 - zi, y] = colors[x, y, zi]
                        break
    elif view == "top":
        w, ht = fx, fz
        img = np.zeros((ht, w, 3), dtype=np.uint8)
        img[:] = bg
        for x in range(fx):
            for y in range(fz):
                for zi in range(h - 1, -1, -1):
                    if voxel[x, y, zi]:
                        img[y, x] = colors[x, y, zi]
                        break
    else:
        raise ValueError(view)

    # Upscale nearest
    scale = max(1, out_size // max(img.shape[0], img.shape[1]))
    up = cv2.resize(
        img,
        (img.shape[1] * scale, img.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    # Pad to square out_size
    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    canvas[:] = bg
    y0 = (out_size - up.shape[0]) // 2
    x0 = (out_size - up.shape[1]) // 2
    y1, x1 = y0 + up.shape[0], x0 + up.shape[1]
    # clip if larger
    uy1 = min(up.shape[0], out_size - max(0, y0))
    ux1 = min(up.shape[1], out_size - max(0, x0))
    cy0, cx0 = max(0, y0), max(0, x0)
    canvas[cy0 : cy0 + uy1, cx0 : cx0 + ux1] = up[:uy1, :ux1]
    return canvas


def render_mesh_ortho(
    mesh,
    view: str = "front",
    out_size: int = 256,
    bg: Tuple[int, int, int] = (255, 255, 255),
    yaw_deg: float = 0.0,
    up_axis: int = 1,
) -> np.ndarray:
    """Simple soft raster of mesh vertices colored by vertex RGB (orthographic)."""
    import trimesh

    m = mesh.copy() if hasattr(mesh, "copy") else mesh
    if yaw_deg:
        rad = np.deg2rad(yaw_deg)
        axis = [0, 1, 0] if up_axis == 1 else ([0, 0, 1] if up_axis == 2 else [1, 0, 0])
        T = trimesh.transformations.rotation_matrix(rad, axis)
        m.apply_transform(T)

    v = np.asarray(m.vertices)
    try:
        vc = np.asarray(m.visual.vertex_colors)[:, :3]
    except Exception:
        vc = np.full((len(v), 3), 200, dtype=np.uint8)

    # Map view → which axes become image x/y
    # After yaw about up: front looks along +Z (or -Z), side along +X
    if up_axis == 1:
        # coords (X, Y_up, Z)
        if view in ("front", "back"):
            ix, iy = 0, 1  # X, Y
            depth = 2
            flip_d = view == "back"
        elif view in ("side", "left", "right"):
            ix, iy = 2, 1  # Z, Y
            depth = 0
            flip_d = view == "left"
        else:  # top
            ix, iy = 0, 2
            depth = 1
            flip_d = False
    else:
        # generic: treat last as up after reindex — keep simple for Y-up meshes
        ix, iy, depth = 0, 1, 2
        flip_d = False

    pts = v.copy()
    # Normalize to [0,1] in ix/iy
    mins = pts.min(0)
    maxs = pts.max(0)
    span = np.maximum(maxs - mins, 1e-6)
    # Depth sort
    order = np.argsort(pts[:, depth] if not flip_d else -pts[:, depth])
    img = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    img[:] = bg
    # Leave margin
    margin = 0.08
    xs = (pts[order, ix] - mins[ix]) / span[ix]
    ys = (pts[order, iy] - mins[iy]) / span[iy]
    # Fit both into square
    px = ((margin + xs * (1 - 2 * margin)) * (out_size - 1)).astype(int)
    # image y: top = high iy
    py = ((1.0 - (margin + ys * (1 - 2 * margin))) * (out_size - 1)).astype(int)
    cols = vc[order]
    r = max(1, out_size // 128)
    for i in range(len(order)):
        x, y = int(px[i]), int(py[i])
        cv2.circle(img, (x, y), r, cols[i].tolist(), -1)
    return img


# ---------------------------------------------------------------------------
# Alignment + masks
# ---------------------------------------------------------------------------

def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        im = np.array(Image.open(path).convert("RGB"))
        return im
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def subject_mask(rgb: np.ndarray, bg_thresh: int = 245) -> np.ndarray:
    """Binary subject mask (True = foreground)."""
    if rgb.shape[2] == 4:
        return rgb[:, :, 3] > 16
    near_white = (
        (rgb[:, :, 0] >= bg_thresh)
        & (rgb[:, :, 1] >= bg_thresh)
        & (rgb[:, :, 2] >= bg_thresh)
    )
    # also treat very light gray as bg
    return ~near_white


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, pad: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return rgb, mask
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def align_pair(
    ref_rgb: np.ndarray,
    cand_rgb: np.ndarray,
    size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop both to subject, resize to size×size (letterbox), return
    (ref_img, cand_img, ref_mask, cand_mask) all size×size.
    """
    rm = subject_mask(ref_rgb)
    cm = subject_mask(cand_rgb)
    r, rm = crop_to_mask(ref_rgb, rm)
    c, cm = crop_to_mask(cand_rgb, cm)

    def letterbox(img, mask, size):
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        im2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
        mcanvas = np.zeros((size, size), dtype=bool)
        y0 = (size - nh) // 2
        x0 = (size - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = im2
        mcanvas[y0 : y0 + nh, x0 : x0 + nw] = m2
        # punch bg white outside mask
        canvas[~mcanvas] = 255
        return canvas, mcanvas

    r2, rm2 = letterbox(r, rm, size)
    c2, cm2 = letterbox(c, cm, size)
    return r2, c2, rm2, cm2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def silhouette_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union else 0.0


def ssim_score(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    if ssim_fn is not None:
        # skimage ≥0.19: channel_axis; use data_range
        score = float(ssim_fn(ga, gb, data_range=255))
        return score
    # Fallback: correlation on masked region
    if mask is not None:
        ga = ga.astype(np.float32)[mask]
        gb = gb.astype(np.float32)[mask]
    else:
        ga = ga.astype(np.float32).ravel()
        gb = gb.astype(np.float32).ravel()
    ga = (ga - ga.mean()) / (ga.std() + 1e-6)
    gb = (gb - gb.mean()) / (gb.std() + 1e-6)
    return float(np.clip(np.mean(ga * gb), -1, 1))


def lpips_lite(a: np.ndarray, b: np.ndarray) -> float:
    """
    Torch-free LPIPS-lite: multi-scale L2 in LAB + Sobel edge maps.
    Lower is better (distance).
    """
    def feats(img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        out = []
        pyr = lab
        eg = np.stack([gx, gy], axis=-1)
        for _ in range(3):
            out.append(pyr.reshape(-1, 3).mean(0))
            out.append(pyr.reshape(-1, 3).std(0))
            out.append(eg.reshape(-1, 2).mean(0))
            out.append(eg.reshape(-1, 2).std(0))
            # spatial grid stats 4x4
            h, w = pyr.shape[:2]
            gh, gw = max(1, h // 4), max(1, w // 4)
            small = cv2.resize(pyr, (4 * gw, 4 * gh), interpolation=cv2.INTER_AREA)
            cells = small.reshape(4, gh, 4, gw, 3).mean(axis=(1, 3)).reshape(-1)
            out.append(cells)
            pyr = cv2.pyrDown(pyr)
            eg = cv2.pyrDown(eg)
        return np.concatenate([np.asarray(x).ravel() for x in out])

    fa, fb = feats(a), feats(b)
    n = min(len(fa), len(fb))
    d = np.linalg.norm(fa[:n] - fb[:n]) / (np.linalg.norm(fa[:n]) + 1e-6)
    return float(d)



@dataclass
class ViewMetrics:
    view: str
    ssim: float
    lpips_lite: float
    silhouette_iou: float
    composite_error: float


@dataclass
class VerifyReport:
    ref_path: str
    cand_label: str
    views: List[ViewMetrics] = field(default_factory=list)
    mean_ssim: float = 0.0
    mean_lpips_lite: float = 0.0
    mean_silhouette_iou: float = 0.0
    composite_error: float = 0.0
    notes: List[str] = field(default_factory=list)
    artifact_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def composite_error(
    ssim: float,
    lpips: float,
    iou: float,
    w_ssim: float = 0.40,
    w_lpips: float = 0.25,
    w_iou: float = 0.35,
) -> float:
    """Lower is better. Shape/structure only (no color histogram). Weights sum to 1."""
    return (
        w_ssim * (1.0 - ssim)
        + w_lpips * lpips
        + w_iou * (1.0 - iou)
    )


def score_views(
    ref_rgb: np.ndarray,
    renders: Dict[str, np.ndarray],
    size: int = 256,
    silhouette_weight: float = 0.35,
) -> Tuple[List[ViewMetrics], dict]:
    rows = []
    for view, cand in renders.items():
        r, c, rm, cm = align_pair(ref_rgb, cand, size=size)
        ss = ssim_score(r, c)
        lp = lpips_lite(r, c)
        iou = silhouette_iou(rm, cm)
        err = composite_error(ss, lp, iou, w_iou=silhouette_weight)
        rows.append(
            ViewMetrics(
                view=view,
                ssim=ss,
                lpips_lite=lp,
                silhouette_iou=iou,
                composite_error=err,
            )
        )
    summary = {
        "mean_ssim": float(np.mean([x.ssim for x in rows])) if rows else 0.0,
        "mean_lpips_lite": float(np.mean([x.lpips_lite for x in rows])) if rows else 0.0,
        "mean_silhouette_iou": float(np.mean([x.silhouette_iou for x in rows])) if rows else 0.0,
        "composite_error": float(np.mean([x.composite_error for x in rows])) if rows else 9.0,
    }
    return rows, summary


def save_overlay(
    ref_rgb: np.ndarray,
    cand_rgb: np.ndarray,
    out_path: str,
    title: str = "",
    size: int = 256,
) -> str:
    r, c, rm, cm = align_pair(ref_rgb, cand_rgb, size=size)
    # Difference heat (abs RGB)
    diff = np.abs(r.astype(np.float32) - c.astype(np.float32)).mean(axis=2)
    diff_n = (np.clip(diff / 80.0, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff_n, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    # silhouette XOR
    xor = np.logical_xor(rm, cm)
    sil = np.zeros_like(r)
    sil[rm & cm] = (80, 200, 80)
    sil[rm & ~cm] = (220, 60, 60)  # missing in cand
    sil[~rm & cm] = (60, 120, 220)  # extra in cand

    gap = 8
    W = size * 4 + gap * 3
    H = size + 36
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 245
    panels = [r, c, heat, sil]
    labels = ["ref", "cand", "|Δ| heat", "sil IoU (G∩ Rmiss Bextra)"]
    for i, (p, lab) in enumerate(zip(panels, labels)):
        x0 = i * (size + gap)
        canvas[36 : 36 + size, x0 : x0 + size] = p
        cv2.putText(
            canvas, lab, (x0 + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA
        )
    if title:
        cv2.putText(
            canvas, title[:90], (4, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_path


def verify_voxel_against_image(
    voxel: np.ndarray,
    colors: np.ndarray,
    ref_path: str,
    out_dir: str,
    label: str = "voxel",
    views: Optional[List[str]] = None,
    silhouette_weight: float = 0.35,
    size: int = 256,
) -> VerifyReport:
    views = views or ["front", "side"]
    os.makedirs(out_dir, exist_ok=True)
    ref = load_rgb(ref_path)
    renders = {}
    paths = {}
    for v in views:
        img = render_voxel_ortho(voxel, colors, view=v, out_size=size)
        p = os.path.join(out_dir, f"render_{label}_{v}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        renders[v] = img
        paths[f"render_{v}"] = os.path.abspath(p)
        ov = os.path.join(out_dir, f"overlay_{label}_{v}.png")
        save_overlay(ref, img, ov, title=f"{label}/{v} vs {os.path.basename(ref_path)}")
        paths[f"overlay_{v}"] = os.path.abspath(ov)

    rows, summary = score_views(ref, renders, size=size, silhouette_weight=silhouette_weight)
    report = VerifyReport(
        ref_path=os.path.abspath(ref_path),
        cand_label=label,
        views=rows,
        notes=[],
        artifact_paths=paths,
        **summary,
    )
    return report


# ---------------------------------------------------------------------------
# Diagnose collage
# ---------------------------------------------------------------------------

def make_diagnose_collage(
    paths: Dict[str, str],
    out_path: str,
    findings: List[str],
    cell: int = 280,
) -> str:
    """paths keys: input, restyle, mesh, palette, voxel_front, voxel_iso (optional)."""
    order = [
        ("input", "1. Input photo"),
        ("restyle", "2. Restyle"),
        ("mesh", "3. TripoSR mesh"),
        ("palette", "4. Palette quant"),
        ("voxel_front", "5. Voxel front"),
        ("voxel_iso", "6. Voxel 3D / angles"),
    ]
    imgs = []
    for key, title in order:
        p = paths.get(key)
        if p and os.path.exists(p):
            im = load_rgb(p)
        else:
            im = np.ones((cell, cell, 3), dtype=np.uint8) * 230
        # letterbox
        h, w = im.shape[:2]
        scale = cell / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        im2 = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.ones((cell, cell, 3), dtype=np.uint8) * 255
        y0, x0 = (cell - nh) // 2, (cell - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = im2
        imgs.append((canvas, title))

    cols = 3
    rows = 2
    header = 36
    footer_lines = max(6, len(findings))
    footer = 22 * footer_lines + 16
    W = cols * cell + (cols + 1) * 10
    H = rows * (cell + header) + (rows + 1) * 10 + footer
    big = np.ones((H, W, 3), dtype=np.uint8) * 245
    for i, (im, title) in enumerate(imgs):
        r, c = divmod(i, cols)
        x = 10 + c * (cell + 10)
        y = 10 + r * (cell + header + 10)
        cv2.putText(
            big, title, (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA
        )
        big[y + header : y + header + cell, x : x + cell] = im

    fy = rows * (cell + header + 10) + 8
    cv2.putText(
        big,
        "FINDINGS",
        (12, fy + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 40, 40),
        2,
        cv2.LINE_AA,
    )
    for i, line in enumerate(findings):
        cv2.putText(
            big,
            line[:110],
            (12, fy + 42 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    return os.path.abspath(out_path)


def red_pixel_fraction(rgb: np.ndarray) -> float:
    m = subject_mask(rgb)
    pix = rgb[m].astype(np.float32)
    if len(pix) == 0:
        return 0.0
    red = (pix[:, 0] > 150) & (pix[:, 1] < 120) & (pix[:, 2] < 120)
    return float(red.mean())
