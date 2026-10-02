"""
Kid-friendly pre-mesh restyle: photo → stylized reference for TripoSR / kid voxelize.

Pipeline intent
---------------
uploaded photo → restyle (this module) → mesh (TripoSR HTTP or procedural fallback)
→ kid voxelize → instructions

Approach (free-first)
---------------------
1. Local OpenCV/PIL cartoonize: bilateral smoothing + edge overlay + k-means flat colors.
2. Background remove via rembg (u2net, free/local ONNX) when available; else GrabCut-ish
   flood from borders.
3. Optional standing-character composite when the subject is head-only / truncated
   (e.g. swimming dog): paste the restyled head onto a simple full-body cartoon dog
   silhouette painted with the photo's palette. This is a *template*, not generative
   limb invention — TripoSR still needs a full silhouette to avoid blob meshes.

What this CANNOT do
-------------------
- Invent the true missing body/legs/tail of a head-only photo (OpenCV cannot hallucinate).
- Reliable free HF cartoon/img2img Spaces were broken/flaky at authoring time; optional
  Gradio/HF hooks are stubs that document required keys/endpoints.

Recommended next model if local restyle is weak
-----------------------------------------------
- FLUX.1-Kontext (img2img / instruction edit) or SDXL img2img with a kid-toy prompt
  ("cute flat-color toy dog, full body standing side view, simple silhouette, white bg")
  via a working HF Space / Replicate / local diffusers. Needs GPU or a paid/hosted key
  for reliable quality. Alternative: InstantID / IP-Adapter face-lock + ControlNet pose.

Usage
-----
  from kid_restyle import restyle_for_kid
  out = restyle_for_kid("photo.jpg", "out/02_restyled.png")

  python kid_restyle.py --image demo_dog/01_input.jpg --out demo_dog_restyle
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class RestyleResult:
    input_path: str
    restyled_path: str
    method: str
    head_only_detected: bool
    body_invented: bool
    body_method: str
    bg_removed: bool
    notes: List[str] = field(default_factory=list)
    palette_rgb: List[List[int]] = field(default_factory=list)
    limits: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Color / cartoon helpers
# ---------------------------------------------------------------------------

def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def kmeans_flat_colors(rgb: np.ndarray, k: int = 6, attempts: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize RGB image to k flat colors. Returns (quantized_rgb, centers_uint8Kx3)."""
    h, w = rgb.shape[:2]
    data = rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _compact, labels, centers = cv2.kmeans(
        data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
    out = centers_u8[labels.flatten()].reshape(h, w, 3)
    return out, centers_u8


def opencv_cartoonize(rgb: np.ndarray, color_k: int = 6) -> np.ndarray:
    """
    Classic OpenCV cartoon look: bilateral smooth + adaptive edges + flat colors.
    Good for kid/toy aesthetic without any API key.
    """
    # Downscale for speed, then upscale with nearest for flatter look
    h, w = rgb.shape[:2]
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    if scale < 1.0:
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = rgb.copy()

    # Strong bilateral smoothing (flatten fur texture toward toy/plastic look)
    smooth = small
    for _ in range(4):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=90, sigmaSpace=90)

    flat, _centers = kmeans_flat_colors(smooth, k=max(3, color_k))

    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )
    edges = cv2.erode(edges, np.ones((2, 2), np.uint8), iterations=1)

    cartoon = flat.copy()
    cartoon[edges == 0] = (cartoon[edges == 0] * 0.28).astype(np.uint8)

    if scale < 1.0:
        cartoon = cv2.resize(cartoon, (w, h), interpolation=cv2.INTER_NEAREST)
    return cartoon


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

def remove_background_rembg(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rgba_uint8, alpha_mask_0_255). Raises if rembg unavailable."""
    from rembg import remove

    pil = Image.fromarray(rgb)
    out = remove(pil)  # RGBA PIL
    rgba = np.array(out)
    if rgba.shape[2] == 3:
        alpha = np.full(rgba.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([rgba, alpha])
    return rgba, rgba[:, :, 3]


def remove_background_opencv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Border-flood + GrabCut fallback when rembg is missing.
    Tuned for subjects on relatively uniform water/sky backgrounds.
    """
    h, w = rgb.shape[:2]
    # Soft subject mask from variance/saturation (same idea as crop_subject)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    diff = cv2.absdiff(gray, blur)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    mask = ((diff > 6) | (sat > 28)).astype(np.uint8) * 255
    # Suppress reflection: lower half often water ripples — keep only largest blob
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n > 1:
        # skip background label 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        mask = np.where(labels == best, 255, 0).astype(np.uint8)

    # GrabCut refine
    gc = np.zeros((h, w), np.uint8)
    gc[mask == 0] = cv2.GC_BGD
    gc[mask == 255] = cv2.GC_PR_FGD
    # Seed sure FG from eroded mask
    sure = cv2.erode(mask, np.ones((21, 21), np.uint8), iterations=1)
    gc[sure == 255] = cv2.GC_FGD
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        alpha = mask

    rgba = np.dstack([rgb, alpha])
    return rgba, alpha


def remove_background(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        rgba, alpha = remove_background_rembg(rgb)
        return rgba, alpha, "rembg"
    except Exception as e:
        rgba, alpha = remove_background_opencv(rgb)
        return rgba, alpha, f"opencv_fallback({type(e).__name__})"




def _edge_band_densities(rgb: np.ndarray, n_bands: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    h = edges.shape[0]
    dens = []
    for i in range(n_bands):
        y0, y1 = int(i * h / n_bands), int((i + 1) * h / n_bands)
        dens.append(float(edges[y0:y1].mean()))
    return np.array(dens, dtype=np.float32)


def strip_reflection(alpha: np.ndarray, rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Drop water-reflection under a floating subject.

    rembg often keeps the reflection glued to the head. Detect swimming / water
    photos via low edge density in the bottom band, then cut near the waterline
    (dark horizontal band / max alpha-width row).
    """
    a = (alpha > 32).astype(np.uint8) * 255
    h, w = a.shape
    ys, xs = np.where(a > 0)
    if len(ys) < 50:
        return alpha

    if rgb is not None and rgb.shape[:2] == (h, w):
        dens = _edge_band_densities(rgb)
        bottom = float(dens[-1])
        mid = float(dens[len(dens) // 2])
        swimming = bottom < 6.0 and mid > 12.0 and (ys.max() > 0.85 * h)
        if swimming:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            lum = gray.mean(axis=1).astype(np.float32)
            row_e = cv2.Canny(gray, 80, 160).mean(axis=1).astype(np.float32)
            score = (180.0 - np.clip(lum, 0, 180)) * (1.0 + row_e / 50.0)
            y0s, y1s = int(0.35 * h), int(0.80 * h)
            waterline = y0s + int(np.argmax(score[y0s:y1s]))
            row_w = (a > 0).sum(axis=1).astype(np.float32)
            peak = int(np.argmax(row_w))
            # Cut slightly below the narrower of waterline / width-peak
            cut = int(min(waterline, peak + int(0.04 * h)))
            # Never cut above ~45% of subject top→peak (keep whole head)
            cut = max(cut, int(ys.min() + 0.45 * (peak - ys.min() + 1)))
            out = alpha.copy()
            out[cut:, :] = 0
            n, labels, stats, _ = cv2.connectedComponentsWithStats((out > 32).astype(np.uint8), 8)
            if n > 1:
                best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                out = np.where(labels == best, out, 0).astype(np.uint8)
            return out

    # Valley pinch between head and reflection
    row = (a > 0).sum(axis=1).astype(np.float32)
    if row.max() >= 10:
        k = max(5, h // 40)
        smooth = np.convolve(row, np.ones(k, np.float32) / k, mode="same")
        y0 = int(0.20 * h)
        y1 = int(0.75 * h)
        segment = smooth[y0:y1]
        if segment.size >= 5:
            valley = y0 + int(np.argmin(segment))
            upper_peak = float(smooth[:valley].max()) if valley > 0 else 0.0
            lower_peak = float(smooth[valley:].max()) if valley < h else 0.0
            valley_val = float(smooth[valley])
            pinch = (
                min(upper_peak, lower_peak) > 0
                and valley_val < 0.45 * min(upper_peak, lower_peak)
            )
            dual = upper_peak > 0.15 * w and lower_peak > 0.12 * w
            if pinch and dual:
                out = alpha.copy()
                out[valley:, :] = 0
                n, labels, stats, _ = cv2.connectedComponentsWithStats((out > 32).astype(np.uint8), 8)
                if n > 1:
                    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    out = np.where(labels == best, out, 0).astype(np.uint8)
                return out
    return alpha


def looks_like_head_crop(alpha: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """After reflection strip: head crops are roughly round / no leg columns."""
    a = (alpha > 32).astype(np.uint8)
    ys, xs = np.where(a > 0)
    if len(ys) < 50:
        return False, {"empty": 1.0}
    h, w = a.shape
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    sub_h = max(1, y1 - y0 + 1)
    sub_w = max(1, x1 - x0 + 1)
    aspect = sub_w / float(sub_h)
    cy = 0.5 * (y0 + y1) / h
    # Leg-like: multiple vertical protrusions in lower 35% of bbox
    lower = a[y0 + int(0.65 * sub_h) : y1 + 1, x0 : x1 + 1]
    col = lower.sum(axis=0) if lower.size else np.array([0])
    # count separated column groups
    active = (col > 0.05 * lower.shape[0]).astype(np.uint8) if lower.size and lower.shape[0] > 0 else np.array([0], np.uint8)
    transitions = int(np.sum(np.abs(np.diff(active.astype(np.int16))))) if active.size > 1 else 0
    n_legs_est = (transitions + 1) // 2 if active.any() else 0
    fill = float(a[y0 : y1 + 1, x0 : x1 + 1].mean())
    # Head: few/no legs, fairly square/round, reasonably filled
    head = (n_legs_est <= 1) and (0.7 < aspect < 1.7) and (fill > 0.25)
    metrics = {
        "centroid_y_frac": round(cy, 3),
        "bbox_aspect_w_over_h": round(aspect, 3),
        "n_legs_est": float(n_legs_est),
        "bbox_fill": round(fill, 3),
        "bbox": [x0, y0, x1, y1],
    }
    return bool(head), metrics


# ---------------------------------------------------------------------------
# Head-only detection + standing template composite
# ---------------------------------------------------------------------------

def detect_head_only(rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> Tuple[bool, Dict[str, float]]:
    """
    Heuristic: subject occupies mostly the upper portion and/or is wider than tall
    with empty lower region (swimming / cropped headshots).
    """
    h, w = rgb.shape[:2]
    if alpha is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        diff = cv2.absdiff(gray, blur)
        sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
        alpha = (((diff > 6) | (sat > 28)).astype(np.uint8) * 255)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    ys, xs = np.where(alpha > 32)
    if len(ys) < 50:
        return False, {"reason": 0.0}

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    sub_h = max(1, y1 - y0 + 1)
    sub_w = max(1, x1 - x0 + 1)
    cy = 0.5 * (y0 + y1) / h
    fill_lower = float((alpha[int(h * 0.55) :, :] > 32).mean())
    aspect = sub_w / float(sub_h)  # head crops often ~1.0–1.6
    top_heavy = cy < 0.55
    little_legs = fill_lower < 0.08
    head_only = bool(top_heavy and little_legs and aspect < 2.2)
    metrics = {
        "centroid_y_frac": round(cy, 3),
        "lower_fill": round(fill_lower, 3),
        "bbox_aspect_w_over_h": round(aspect, 3),
        "bbox": [x0, y0, x1, y1],
    }
    return head_only, metrics


def extract_subject_palette(rgb: np.ndarray, alpha: np.ndarray, k: int = 5) -> np.ndarray:
    """Dominant colors of the opaque subject (Kx3 uint8)."""
    sel = alpha > 64
    if sel.sum() < 100:
        flat, centers = kmeans_flat_colors(rgb, k=k)
        return centers
    pixels = rgb[sel].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _c, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return np.clip(centers, 0, 255).astype(np.uint8)


def _pick_fur_colors(palette: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick body / accent / light colors from palette (skip near-black/white when possible)."""
    colors = [c for c in palette]
    def lum(c):
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    mid = sorted(colors, key=lum)
    body = mid[len(mid) // 2]
    dark = mid[0]
    light = mid[-1]
    # Prefer a warm-ish mid for dog body if available
    warm = [c for c in mid if c[0] > c[2] and 40 < lum(c) < 200]
    if warm:
        body = warm[len(warm) // 2]
    return body.astype(np.uint8), dark.astype(np.uint8), light.astype(np.uint8)


def draw_standing_dog_silhouette(
    size: int = 768,
    body_rgb: np.ndarray = None,
    accent_rgb: np.ndarray = None,
    light_rgb: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a simple full-body standing dog cartoon (side-ish 3/4) on transparent canvas.
    Returns (rgba, alpha). Pure procedural — not the real dog's body.
    """
    body_rgb = body_rgb if body_rgb is not None else np.array([180, 120, 70], np.uint8)
    accent_rgb = accent_rgb if accent_rgb is not None else np.array([40, 30, 25], np.uint8)
    light_rgb = light_rgb if light_rgb is not None else np.array([230, 210, 180], np.uint8)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    # White-ish opaque bg for TripoSR friendliness (we keep alpha for compositing too)
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 250
    mask = np.zeros((size, size), dtype=np.uint8)

    def fill_ellipse(cx, cy, ax, ay, color, mval=255):
        cv2.ellipse(canvas, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, color.tolist(), -1)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, int(mval), -1)

    def fill_rect(x0, y0, x1, y1, color, mval=255):
        cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color.tolist(), -1)
        cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), int(mval), -1)

    s = size
    # Body loaf (no drawn head/ears — photo head is pasted later)
    fill_ellipse(0.48 * s, 0.58 * s, 0.28 * s, 0.16 * s, body_rgb)
    # Chest
    fill_ellipse(0.62 * s, 0.55 * s, 0.12 * s, 0.14 * s, body_rgb)
    # Neck stump (head attaches here)
    fill_ellipse(0.72 * s, 0.45 * s, 0.08 * s, 0.07 * s, body_rgb)
    # Legs
    for x in (0.34, 0.42, 0.55, 0.63):
        fill_rect(x * s - 0.025 * s, 0.62 * s, x * s + 0.025 * s, 0.82 * s, body_rgb)
        fill_ellipse(x * s, 0.83 * s, 0.035 * s, 0.025 * s, accent_rgb)
    # Tail
    fill_ellipse(0.22 * s, 0.52 * s, 0.08 * s, 0.035 * s, body_rgb)
    fill_ellipse(0.15 * s, 0.46 * s, 0.04 * s, 0.04 * s, accent_rgb)
    # Belly highlight
    fill_ellipse(0.50 * s, 0.63 * s, 0.16 * s, 0.06 * s, light_rgb)

    # Soft outline
    edges = cv2.Canny(mask, 50, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    canvas[edges > 0] = (30, 25, 20)

    rgba[:, :, :3] = canvas
    rgba[:, :, 3] = np.where(mask > 0, 255, 0).astype(np.uint8)
    # Force white background for mesh-friendly plate
    out_rgb = canvas.copy()
    out_rgb[mask == 0] = (255, 255, 255)
    return out_rgb, mask


def composite_head_on_standing_body(
    head_rgba: np.ndarray,
    head_alpha: np.ndarray,
    palette: np.ndarray,
    canvas_size: int = 768,
) -> Tuple[np.ndarray, str]:
    """
    Paste restyled head onto procedural standing dog body.
    Honest limit: body is a generic template tinted with photo colors — not the real dog.
    """
    body_c, accent_c, light_c = _pick_fur_colors(palette)
    body_rgb, body_mask = draw_standing_dog_silhouette(
        canvas_size, body_c, accent_c, light_c
    )

    # Crop head tightly
    ys, xs = np.where(head_alpha > 64)
    if len(ys) < 20:
        return body_rgb, "template_only_no_head_mask"

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # Prefer upper portion of mask (exclude water reflection if any)
    mid_y = (y0 + y1) // 2
    # Keep from top to ~70% of bbox height
    y1_clip = y0 + int(0.72 * (y1 - y0 + 1))
    head_crop = head_rgba[y0:y1_clip, x0:x1].copy()
    if head_crop.shape[2] == 4:
        head_rgb = head_crop[:, :, :3]
        head_a = head_crop[:, :, 3]
    else:
        head_rgb = head_crop
        head_a = head_alpha[y0:y1_clip, x0:x1]

    # Target head slot on body (upper-right / neck stump)
    s = canvas_size
    target_h = int(0.34 * s)
    scale = target_h / max(1, head_rgb.shape[0])
    tw = max(1, int(head_rgb.shape[1] * scale))
    th = max(1, int(head_rgb.shape[0] * scale))
    head_r = cv2.resize(head_rgb, (tw, th), interpolation=cv2.INTER_AREA)
    a_r = cv2.resize(head_a, (tw, th), interpolation=cv2.INTER_LINEAR)
    # Feather alpha edge so photo/cartoon head blends onto flat body
    a_r = cv2.GaussianBlur(a_r, (7, 7), 0)

    tx = int(0.78 * s - tw / 2)
    ty = int(0.26 * s - th / 2)
    tx = max(0, min(s - tw, tx))
    ty = max(0, min(s - th, ty))

    out = body_rgb.copy()
    roi = out[ty : ty + th, tx : tx + tw]
    alpha_f = (a_r.astype(np.float32) / 255.0)[..., None]
    blended = (head_r.astype(np.float32) * alpha_f + roi.astype(np.float32) * (1.0 - alpha_f)).astype(
        np.uint8
    )
    out[ty : ty + th, tx : tx + tw] = blended

    # Clean white bg
    # Keep body mask | pasted head
    full_mask = body_mask.copy()
    full_mask[ty : ty + th, tx : tx + tw] = np.maximum(
        full_mask[ty : ty + th, tx : tx + tw], a_r
    )
    out[full_mask == 0] = (255, 255, 255)
    return out, "procedural_standing_template+photo_head"


def rgba_on_white(rgba: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3] if rgba.shape[2] >= 3 else rgba
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    white = np.ones_like(rgb, dtype=np.float32) * 255.0
    return (rgb.astype(np.float32) * a + white * (1.0 - a)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Optional remote cartoon (documented stub)
# ---------------------------------------------------------------------------

def try_remote_cartoon(rgb: np.ndarray, timeout: int = 30) -> Optional[np.ndarray]:
    """
    Attempt free HTTP cartoon endpoints. Returns None if unavailable.

    Documented options (often need HF token or are broken):
    - HF Inference API cartoon models → needs HF_TOKEN
    - Gradio client to a live Space → needs gradio_client + running Space
    Current free cartoon Spaces checked were erroring; we skip silently.
    """
    endpoint = os.environ.get("KID_RESTYLE_HTTP_URL", "").strip()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not endpoint:
        return None
    try:
        import requests
        from io import BytesIO

        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        buf.seek(0)
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.post(
            endpoint,
            files={"image": ("input.png", buf.getvalue(), "image/png")},
            headers=headers,
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return _bgr_to_rgb(img)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main restyle API
# ---------------------------------------------------------------------------

def restyle_for_kid(
    image_path: str,
    out_path: str,
    color_k: int = 6,
    invent_body_if_head_only: bool = True,
    prefer_remote: bool = False,
) -> RestyleResult:
    """
    Produce a kid-friendly reference image suitable for TripoSR / procedural guidance.
    """
    notes: List[str] = []
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = _bgr_to_rgb(bgr)

    method = "opencv_cartoon"
    restyled = None
    if prefer_remote:
        remote = try_remote_cartoon(rgb)
        if remote is not None:
            restyled = remote
            method = "remote_http"
            notes.append(f"Used KID_RESTYLE_HTTP_URL remote restyle")
        else:
            notes.append(
                "Remote restyle skipped/unavailable "
                "(set KID_RESTYLE_HTTP_URL; HF cartoon Spaces were broken without keys)"
            )

    if restyled is None:
        restyled = opencv_cartoonize(rgb, color_k=color_k)
        notes.append("Local OpenCV cartoonize (bilateral + edges + k-means flat colors)")

    # Prefer cutout-first: remove bg on original photo, then cartoonize the subject.
    # This avoids water/ripples becoming flat-color blobs in the restyle.
    rgba0, alpha0, bg_method = remove_background(rgb)
    notes.append(f"Background remove: {bg_method}")
    alpha0 = strip_reflection(alpha0, rgb)
    notes.append("Applied reflection/waterline strip heuristic")

    cut = rgba_on_white(rgba0, alpha0)
    # Tight crop before cartoon
    ys, xs = np.where(alpha0 > 32)
    if len(xs) > 50:
        pad = 12
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(cut.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(cut.shape[1], int(xs.max()) + pad + 1)
        cut = cut[y0:y1, x0:x1]
        alpha_c = alpha0[y0:y1, x0:x1]
    else:
        alpha_c = alpha0

    if method == "opencv_cartoon":
        # Re-cartoonize the cleaned cutout for cleaner flat colors
        restyled = opencv_cartoonize(cut, color_k=color_k)
        # Rebuild alpha from non-white
        white = np.all(restyled > 245, axis=2)
        alpha = np.where(white, 0, 255).astype(np.uint8)
        # Prefer original cutout alpha resized
        alpha = cv2.resize(alpha_c, (restyled.shape[1], restyled.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgba = np.dstack([restyled, alpha])
    else:
        restyled = cut
        alpha = alpha_c
        rgba = np.dstack([restyled, alpha]) if restyled.shape[2] == 3 else restyled

    head_only, metrics = looks_like_head_crop(alpha)
    # If reflection strip substantially shortened the mask vs raw rembg → treat as head-only
    ys_raw = np.where(alpha0 > 32)[0]
    ys_now = np.where(alpha > 32)[0]
    if len(ys_raw) > 50 and len(ys_now) > 50:
        h_raw = int(ys_raw.max() - ys_raw.min() + 1)
        h_now = int(ys_now.max() - ys_now.min() + 1)
        if h_now < 0.72 * h_raw:
            head_only = True
            metrics["stripped_height_ratio"] = round(h_now / float(h_raw), 3)
            metrics["forced_head_only_after_strip"] = 1.0
    if not head_only:
        head_only2, metrics2 = detect_head_only(rgb, alpha)
        if head_only2:
            head_only, metrics = head_only2, metrics2
    notes.append(f"head_only_detect={head_only} metrics={metrics}")

    palette = extract_subject_palette(restyled, alpha, k=min(color_k, 5))
    body_invented = False
    body_method = "none"

    if head_only and invent_body_if_head_only:
        # Composite photo head onto procedural standing body
        plate, body_method = composite_head_on_standing_body(rgba, alpha, palette)
        final = plate
        body_invented = True
        notes.append(
            "Head-only subject: composited restyled head onto procedural standing-dog "
            "silhouette tinted with photo palette. Body/legs are TEMPLATE, not the real dog."
        )
        limits = (
            "Cannot invent the real missing body/limbs from a swimming/head-only photo. "
            "Standing body is a generic cartoon template for silhouette guidance only. "
            "For true full-body invention use generative img2img (FLUX.1-Kontext / SDXL) "
            "with a full-body toy prompt (usually needs GPU or API key)."
        )
    else:
        final = rgba_on_white(rgba, alpha)
        # Tighten crop
        ys, xs = np.where(alpha > 32)
        if len(xs) > 50:
            pad = 16
            y0 = max(0, ys.min() - pad)
            y1 = min(final.shape[0], ys.max() + pad + 1)
            x0 = max(0, xs.min() - pad)
            x1 = min(final.shape[1], xs.max() + pad + 1)
            final = final[y0:y1, x0:x1]
        # Pad to square white
        h, w = final.shape[:2]
        side = max(h, w)
        square = np.ones((side, side, 3), dtype=np.uint8) * 255
        oy, ox = (side - h) // 2, (side - w) // 2
        square[oy : oy + h, ox : ox + w] = final
        final = square
        limits = (
            "Local cartoon + bg-remove only. Does not invent missing geometry. "
            "If TripoSR still blobs, prefer a standing full-body photo or procedural mesh."
        )
        notes.append("Full-ish subject: restyled cutout on white square")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    Image.fromarray(final).save(out_path)

    return RestyleResult(
        input_path=image_path,
        restyled_path=out_path,
        method=method,
        head_only_detected=head_only,
        body_invented=body_invented,
        body_method=body_method,
        bg_removed=True,
        notes=notes,
        palette_rgb=palette.tolist(),
        limits=limits,
    )


def main():
    ap = argparse.ArgumentParser(description="Kid-friendly restyle (pre-mesh)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="Output dir or .png path")
    ap.add_argument("--color-k", type=int, default=6)
    ap.add_argument("--no-invent-body", action="store_true")
    ap.add_argument("--prefer-remote", action="store_true")
    args = ap.parse_args()

    out = args.out
    if out.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        out_path = out
        out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    else:
        out_dir = out
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "02_restyled.png")

    # Copy input preview
    os.makedirs(out_dir, exist_ok=True)
    src = cv2.imread(args.image)
    if src is not None:
        cv2.imwrite(os.path.join(out_dir, "01_input.jpg"), src)

    result = restyle_for_kid(
        args.image,
        out_path,
        color_k=args.color_k,
        invent_body_if_head_only=not args.no_invent_body,
        prefer_remote=args.prefer_remote,
    )
    meta_path = os.path.join(out_dir, "02_restyle_meta.json")
    with open(meta_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
