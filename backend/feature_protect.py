"""Feature preservation for image_project kid kits.

Detects identity marks (cheeks, eyes, ear tips) in the framed photo and
forces Perler Red/Black into the matching voxels after image_project mapping.

Voxel mapping (must stay in sync with VoxelQuantizer._color_volume_image_project):
  1. Subject mask → tight bbox crop of the framed RGB (same pad as image_project).
  2. Photo height → pancake layer:
       row_frac = 1.0 - (z + 0.5) / H
       y0 = int(row_frac * (qh - 1)); band = crop[y0±half]
  3. Band resized with INTER_NEAREST to (fz, fx) so band_small[vx, vz] lines up
     with voxel[vx, vz, z]. Feature masks use the identical band+resize path.
  4. Only occupied voxels are painted (no floating beads). Min sizes grow by
     dilating within the layer's occupied silhouette.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from color_quantizer import PERLER_PALETTE

RED = tuple(int(x) for x in PERLER_PALETTE["Red"])
BLACK = tuple(int(x) for x in PERLER_PALETTE["Black"])


@dataclass
class FeatureMasks:
    """Boolean masks in framed-photo space (same H×W as detection input)."""

    cheeks_left: np.ndarray
    cheeks_right: np.ndarray
    eyes_left: np.ndarray
    eyes_right: np.ndarray
    ear_tips_left: np.ndarray
    ear_tips_right: np.ndarray
    mouth: Optional[np.ndarray] = None
    subject: Optional[np.ndarray] = None
    # Crop used for image_project-equivalent mapping (y0, y1, x0, x1)
    crop_box: Optional[Tuple[int, int, int, int]] = None
    notes: List[str] = field(default_factory=list)
    # Per-feature detection method labels (e.g. photo_edge / photo_blob)
    detection_methods: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, np.ndarray]:
        d = {
            "cheeks_left": self.cheeks_left,
            "cheeks_right": self.cheeks_right,
            "eyes_left": self.eyes_left,
            "eyes_right": self.eyes_right,
            "ear_tips_left": self.ear_tips_left,
            "ear_tips_right": self.ear_tips_right,
        }
        if self.mouth is not None:
            d["mouth"] = self.mouth
        if self.subject is not None:
            d["subject"] = self.subject
        return d


def _empty_mask(shape) -> np.ndarray:
    return np.zeros(shape[:2], dtype=bool)


def _largest_components(mask: np.ndarray, min_area: int = 8, top_k: int = 8):
    u8 = mask.astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(u8, connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comps.append(
            {
                "label": i,
                "area": area,
                "cx": float(cents[i][0]),
                "cy": float(cents[i][1]),
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "mask": labels == i,
            }
        )
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps[:top_k]


def _tight_subject(
    framed: np.ndarray, fallback_mask: np.ndarray
) -> np.ndarray:
    """Prefer grabCut when flood-fill subject is too loose (busy warm backgrounds)."""
    h, w = framed.shape[:2]
    if float(fallback_mask.mean()) < 0.85 and float(fallback_mask.mean()) > 0.05:
        return fallback_mask.astype(bool)

    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    border = max(8, min(h, w) // 30)
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD
    cy, cx = h // 2, w // 2
    seed = max(20, min(h, w) // 12)
    mask[cy - seed : cy + seed, cx - seed : cx + seed] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(framed, mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        sub = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except Exception:
        return fallback_mask.astype(bool)

    sub_u8 = sub.astype(np.uint8) * 255
    sub_u8 = cv2.morphologyEx(
        sub_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sub_u8, 8)
    if num <= 1:
        return fallback_mask.astype(bool)
    best = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, num)]))
    sub = labels == best
    if float(sub.mean()) < 0.03:
        return fallback_mask.astype(bool)
    return sub


def _subject_bbox(subject: np.ndarray, pad_frac: float = 0.05):
    ys, xs = np.where(subject)
    if len(xs) < 20:
        h, w = subject.shape
        return 0, h, 0, w
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    fh, fw = y1 - y0, x1 - x0
    pad_y = max(1, int(fh * pad_frac))
    pad_x = max(1, int(fw * pad_frac))
    h, w = subject.shape
    return (
        max(0, y0 - pad_y),
        min(h, y1 + pad_y),
        max(0, x0 - pad_x),
        min(w, x1 + pad_x),
    )


def _disk(shape, cx: float, cy: float, radius: int) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def _pick_lr_blobs(
    comps: list, mid_x: float, min_area: int = 20
) -> Tuple[Optional[dict], Optional[dict]]:
    left = [c for c in comps if c["cx"] < mid_x and c["area"] >= min_area]
    right = [c for c in comps if c["cx"] >= mid_x and c["area"] >= min_area]
    return (left[0] if left else None, right[0] if right else None)



def _detect_eye_strokes(
    framed_rgb: np.ndarray,
    subject: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    fh: int,
    fw: int,
    mid_x: float,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Find thin dark closed-eye arcs in the upper-face band (photo methods).

    Uses horizontal black-hat, adaptive threshold, and Canny edges, scored for
    elongation. Left/right halves are searched separately so arcs do not merge.
    Returns (eyes_left, eyes_right, method_left, method_right) where method is
    ``photo_edge`` or ``miss``.
    """
    h, w = framed_rgb.shape[:2]
    gray = cv2.cvtColor(framed_rgb, cv2.COLOR_RGB2GRAY)
    V = cv2.cvtColor(framed_rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    sub_d = cv2.dilate(subject.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))

    eye_band = _empty_mask(framed_rgb.shape)
    eye_band[
        y0 + int(0.26 * fh) : y0 + int(0.38 * fh),
        x0 + int(0.22 * fw) : x0 + int(0.82 * fw),
    ] = True

    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    local_dark = (blur.astype(np.int16) - gray.astype(np.int16)) >= 10

    bh_acc = np.zeros((h, w), np.uint8)
    for kx in (9, 13, 17, 23):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
        bh_acc = np.maximum(
            bh_acc, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)
        )

    adapt = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4,
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 90)

    # Soft dark: quantized yellow kits crush absolute V; use local contrast OR bh.
    soft_dark = local_dark | (V <= 120)
    base = (
        (
            (bh_acc > 10)
            | ((adapt > 0) & soft_dark)
            | ((edges > 0) & soft_dark)
        )
        & eye_band
        & (sub_d > 0)
    )

    gap = max(3, int(0.03 * fw))
    out_masks = []
    out_methods = []
    for side, x_lo, x_hi, thr_bh in (
        ("left", x0 + int(0.28 * fw), int(mid_x) - gap, 8),
        ("right", int(mid_x) + gap, x0 + int(0.78 * fw), 6),
    ):
        roi = _empty_mask(framed_rgb.shape)
        x_lo_i = max(0, int(x_lo))
        x_hi_i = min(w, int(x_hi))
        if x_hi_i > x_lo_i:
            roi[:, x_lo_i:x_hi_i] = True
        extra = (
            (bh_acc > thr_bh)
            & roi
            & eye_band
            & (sub_d > 0)
            & local_dark
        )
        m = ((base & roi) | extra).astype(np.uint8) * 255
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE, np.ones((2, 4), np.uint8), iterations=2
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, 2), np.uint8))
        comps = _largest_components(m > 0, min_area=6, top_k=15)
        best = None
        best_s = -1e9
        band_top = y0 + 0.26 * fh
        band_h = max(1e-6, 0.12 * fh)
        for c in comps:
            wh = c["w"] / max(1, c["h"])
            elong = max(c["w"], c["h"]) / max(1, min(c["w"], c["h"]))
            if c["h"] > 22 or c["area"] > 280:
                continue
            if wh < 1.1 and elong < 1.5:
                continue
            area_term = 4.0 - abs(np.log(max(c["area"], 1) / 55.0))
            s = min(wh, 5) * 2.0 + min(elong, 5) * 0.5 + area_term * 3.5
            # Tiny strokes (<12px) rarely survive voxel resize — prefer larger.
            if c["area"] < 12:
                s -= 3.0
            elif c["area"] >= 25:
                s += 2.0
            rely = (c["cy"] - band_top) / band_h
            # Closed eyes sit mid-band; boost that zone.
            if 0.25 <= rely <= 0.75:
                s += 4.0
            elif 0.1 <= rely <= 0.9:
                s += 1.0
            else:
                s -= 2.0
            s += max(0.0, (150.0 - float(V[c["mask"]].mean())) / 25.0)
            if s > best_s:
                best_s = s
                best = c
        if best is not None:
            # Keep the stroke, and anchor a disk at its centroid so the thin
            # arc survives downsampling into the voxel grid.
            stroke = cv2.dilate(
                best["mask"].astype(np.uint8), np.ones((3, 3), np.uint8)
            ).astype(bool)
            r = max(6, fw // 22)
            disk = _disk(framed_rgb.shape, best["cx"], best["cy"], r)
            # Clip to subject only (not eye_band) so the disk stays large enough
            # to survive INTER_AREA / voxel resize.
            mask = (stroke | disk) & subject
            if not mask.any():
                mask = stroke & (sub_d > 0)
            out_masks.append(mask)
            out_methods.append("photo_edge")
        else:
            out_masks.append(_empty_mask(framed_rgb.shape))
            out_methods.append("miss")
    return out_masks[0], out_masks[1], out_methods[0], out_methods[1]


def detect_features(
    framed_rgb: np.ndarray,
    subject_mask: np.ndarray,
    *,
    eye_geometric_fallback: bool = False,
) -> FeatureMasks:
    """Detect cheeks / eyes / ear tips in framed photo space.

    Eyes prefer thin dark strokes (black-hat / edges). Geometric eye seed disks
    are off by default (``eye_geometric_fallback=False``).
    """
    h, w = framed_rgb.shape[:2]
    notes: List[str] = []
    subject = _tight_subject(framed_rgb, subject_mask)
    y0, y1, x0, x1 = _subject_bbox(subject, pad_frac=0.02)
    # Match image_project pad (~1/20 of bbox)
    fh, fw = max(1, y1 - y0), max(1, x1 - x0)
    pad_y = max(1, fh // 20)
    pad_x = max(1, fw // 20)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    fh, fw = max(1, y1 - y0), max(1, x1 - x0)
    mid_x = x0 + fw / 2.0

    hsv = cv2.cvtColor(framed_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    sub_d = cv2.dilate(subject.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))

    # --- Cheeks: warm red HSV blobs, mid-face L/R ---
    red1 = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 70, 70), (179, 255, 255))
    red = ((red1 > 0) | (red2 > 0)) & (sub_d > 0)
    red_u = cv2.morphologyEx(
        red.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    red_u = cv2.morphologyEx(
        red_u, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    cheek_band = _empty_mask(framed_rgb.shape)
    cheek_band[y0 + int(0.30 * fh) : y0 + int(0.58 * fh), x0:x1] = True
    cheek_cand = (red_u > 0) & cheek_band
    cheek_comps = _largest_components(cheek_cand, min_area=25, top_k=10)
    left_c, right_c = _pick_lr_blobs(cheek_comps, mid_x, min_area=25)

    cheeks_left = _empty_mask(framed_rgb.shape)
    cheeks_right = _empty_mask(framed_rgb.shape)
    if left_c is not None:
        cheeks_left = left_c["mask"]
    if right_c is not None:
        cheeks_right = right_c["mask"]

    # Mirror a confident cheek onto the missing side (sun-washed opposite cheek)
    if left_c is not None and right_c is None:
        notes.append("cheek_right_mirrored_from_left")
        for yy, xx in zip(*np.where(cheeks_left)):
            mx = int(round(2 * mid_x - xx))
            if 0 <= mx < w:
                cheeks_right[yy, mx] = True
        cheeks_right = cv2.dilate(
            cheeks_right.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
    elif right_c is not None and left_c is None:
        notes.append("cheek_left_mirrored_from_right")
        for yy, xx in zip(*np.where(cheeks_right)):
            mx = int(round(2 * mid_x - xx))
            if 0 <= mx < w:
                cheeks_left[yy, mx] = True
        cheeks_left = cv2.dilate(
            cheeks_left.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
    elif left_c is None and right_c is None:
        notes.append("cheeks_geometric_fallback")
        r = max(6, fw // 18)
        cy = y0 + int(0.45 * fh)
        cx_l = x0 + int(0.30 * fw)
        cx_r = x0 + int(0.70 * fw)
        cheeks_left = _disk(framed_rgb.shape, cx_l, cy, r) & subject
        cheeks_right = _disk(framed_rgb.shape, cx_r, cy, r) & subject

    # --- Eyes: thin dark strokes / closed-eye arcs (photo), then blob, then optional geometric ---
    detection_methods: Dict[str, str] = {}
    eyes_left, eyes_right, meth_l, meth_r = _detect_eye_strokes(
        framed_rgb, subject, y0, y1, x0, x1, fh, fw, mid_x
    )
    if meth_l == "photo_edge":
        detection_methods["eyes_left"] = "photo_edge"
    if meth_r == "photo_edge":
        detection_methods["eyes_right"] = "photo_edge"

    # Secondary photo method: dark low-V blobs (still photo_*, never geometric)
    if not eyes_left.any() or not eyes_right.any():
        eye_band = _empty_mask(framed_rgb.shape)
        eye_band[
            y0 + int(0.18 * fh) : y0 + int(0.40 * fh),
            x0 + int(0.12 * fw) : x0 + int(0.88 * fw),
        ] = True
        eye_dark = (V <= 80) & eye_band & (sub_d > 0)
        eye_u = cv2.morphologyEx(
            eye_dark.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )
        eye_comps = _largest_components(eye_u > 0, min_area=5, top_k=12)
        left_e, right_e = _pick_lr_blobs(eye_comps, mid_x, min_area=5)
        if not eyes_left.any() and left_e is not None:
            eyes_left = left_e["mask"]
            detection_methods["eyes_left"] = "photo_blob"
            notes.append("eyes_left_photo_blob")
        if not eyes_right.any() and right_e is not None:
            eyes_right = right_e["mask"]
            detection_methods["eyes_right"] = "photo_blob"
            notes.append("eyes_right_photo_blob")

    if not eyes_left.any():
        notes.append("eyes_left_miss")
        detection_methods.setdefault("eyes_left", "miss")
    if not eyes_right.any():
        notes.append("eyes_right_miss")
        detection_methods.setdefault("eyes_right", "miss")

    # Debug-only geometric seed disks (OFF by default for Epic 2 acceptance)
    if eye_geometric_fallback and (not eyes_left.any() or not eyes_right.any()):
        notes.append("eyes_geometric_seed")
        r = max(3, fw // 35)
        cy = y0 + int(0.28 * fh)
        if not eyes_left.any():
            eyes_left = _disk(framed_rgb.shape, x0 + int(0.32 * fw), cy, r) & subject
            detection_methods["eyes_left"] = "geometric_seed"
        if not eyes_right.any():
            eyes_right = _disk(framed_rgb.shape, x0 + int(0.68 * fw), cy, r) & subject
            detection_methods["eyes_right"] = "geometric_seed"

    # --- Ear tips: dark tips on top protrusions (silhouette peaks + dark) ---
    ear_tips_left = _empty_mask(framed_rgb.shape)
    ear_tips_right = _empty_mask(framed_rgb.shape)
    # Silhouette top peaks in left/right halves of subject
    left_peak = None
    right_peak = None
    for x in range(x0, int(mid_x)):
        col = np.where(subject[:, x])[0]
        if len(col) == 0:
            continue
        yt = int(col.min())
        if left_peak is None or yt < left_peak[1]:
            left_peak = (x, yt)
    for x in range(int(mid_x), x1):
        col = np.where(subject[:, x])[0]
        if len(col) == 0:
            continue
        yt = int(col.min())
        if right_peak is None or yt < right_peak[1]:
            right_peak = (x, yt)

    ear_r = max(5, fw // 22)
    dark_top = (V <= 70) & (sub_d > 0)
    dark_top[: max(0, y0)] = False
    dark_top[y0 + int(0.28 * fh) :] = False

    def _ear_mask(peak):
        if peak is None:
            return _empty_mask(framed_rgb.shape)
        px, py = peak
        m = _disk(framed_rgb.shape, px, py + ear_r // 3, ear_r)
        # Prefer dark pixels near peak when available
        local = dark_top & _disk(framed_rgb.shape, px, py + ear_r // 2, ear_r * 2)
        if int(local.sum()) >= 8:
            m = m | local
        return m & (sub_d > 0)

    if left_peak is not None:
        ear_tips_left = _ear_mask(left_peak)
    else:
        notes.append("ear_tip_left_miss")
    if right_peak is not None:
        ear_tips_right = _ear_mask(right_peak)
    else:
        notes.append("ear_tip_right_miss")

    # Optional mouth: small dark mid-face under eyes
    mouth = _empty_mask(framed_rgb.shape)
    mouth_band = _empty_mask(framed_rgb.shape)
    mouth_band[
        y0 + int(0.38 * fh) : y0 + int(0.52 * fh),
        x0 + int(0.35 * fw) : x0 + int(0.65 * fw),
    ] = True
    mouth_dark = (V <= 75) & mouth_band & subject
    mouth_comps = _largest_components(mouth_dark, min_area=8, top_k=3)
    if mouth_comps:
        mouth = mouth_comps[0]["mask"]
    else:
        mouth = None

    return FeatureMasks(
        cheeks_left=cheeks_left,
        cheeks_right=cheeks_right,
        eyes_left=eyes_left,
        eyes_right=eyes_right,
        ear_tips_left=ear_tips_left,
        ear_tips_right=ear_tips_right,
        mouth=mouth,
        subject=subject,
        crop_box=(y0, y1, x0, x1),
        notes=notes,
        detection_methods=detection_methods,
    )


def save_feature_masks(masks: FeatureMasks, framed_rgb: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    overlay = framed_rgb.copy()
    for name, m in masks.as_dict().items():
        if name == "subject":
            continue
        path = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(path, (m.astype(np.uint8) * 255))
        if "cheek" in name:
            overlay[m] = (255, 40, 40)
        elif "eye" in name or name == "mouth":
            overlay[m] = (0, 0, 0)
        elif "ear" in name:
            overlay[m] = (40, 40, 200)
    if masks.subject is not None:
        cv2.imwrite(
            os.path.join(out_dir, "subject.png"),
            masks.subject.astype(np.uint8) * 255,
        )
    cv2.imwrite(
        os.path.join(out_dir, "overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    with open(os.path.join(out_dir, "notes.txt"), "w") as f:
        f.write("\n".join(masks.notes) if masks.notes else "ok\n")


def _crop_mask(mask: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    return mask[y0:y1, x0:x1]


def _layers_for_mask(mask_cropped: np.ndarray, n_layers: int) -> List[int]:
    """Photo-Y → layer indices using the same row_frac as image_project."""
    qh = mask_cropped.shape[0]
    if qh <= 0 or not mask_cropped.any():
        return []
    ys = np.where(mask_cropped.any(axis=1))[0]
    if len(ys) == 0:
        return []
    layers = set()
    for y in (int(ys.min()), int(ys.mean()), int(ys.max())):
        # Invert image_project: y0 = row_frac*(qh-1), row_frac = 1-(z+0.5)/H
        row_frac = y / max(1, qh - 1)
        z = (1.0 - row_frac) * n_layers - 0.5
        layers.add(int(np.clip(round(z), 0, n_layers - 1)))
    # Also include neighbors so thin features don't miss their band
    expanded = set()
    for z in layers:
        for dz in (-1, 0, 1):
            zz = z + dz
            if 0 <= zz < n_layers:
                expanded.add(zz)
    return sorted(expanded)


def map_mask_to_layer_hits(
    mask_cropped: np.ndarray,
    voxel: np.ndarray,
    layers: Optional[List[int]] = None,
    snap_to_occupied: bool = True,
) -> Dict[int, np.ndarray]:
    """Resize photo-space mask bands onto each pancake (same as image_project).

    Returns dict layer_z → boolean (fx, fz) hit map on occupancy.
    When snap_to_occupied, empty projected cells move to the nearest occupied
    neighbor on that layer (epic: grow into nearest occupied; no floating beads).
    """
    fx, fz, h = voxel.shape
    qh = mask_cropped.shape[0]
    if qh <= 0:
        return {}
    if layers is None:
        layers = list(range(h))
    half = max(1, qh // 12)
    out: Dict[int, np.ndarray] = {}
    for z in layers:
        row_frac = 1.0 - (z + 0.5) / max(1, h)
        y0 = int(row_frac * (qh - 1))
        band = mask_cropped[
            max(0, y0 - half) : min(qh, y0 + half) + 1
        ].astype(np.uint8)
        if band.size == 0:
            continue
        # dsize=(width=fz, height=fx). Use AREA (not NEAREST) so thin photo
        # strokes/eyes still produce a positive cell after heavy downsample;
        # any coverage in a source block survives as proj>0.
        band_f = band.astype(np.float32)
        band_small = cv2.resize(band_f, (fz, fx), interpolation=cv2.INTER_AREA)
        proj = band_small > 0.0
        if not proj.any():
            continue
        occ = voxel[:, :, z] > 0
        hit = proj & occ
        if snap_to_occupied and not hit.any() and occ.any():
            # Snap each projected empty cell to nearest occupied on this layer
            hit = np.zeros_like(occ)
            oy, ox = np.where(occ)
            for py, px in zip(*np.where(proj)):
                d2 = (oy - py) ** 2 + (ox - px) ** 2
                j = int(np.argmin(d2))
                hit[oy[j], ox[j]] = True
        elif snap_to_occupied and proj.any() and occ.any():
            # Also snap projected-but-empty cells so thin features still land
            miss = proj & ~occ
            if miss.any():
                oy, ox = np.where(occ)
                for py, px in zip(*np.where(miss)):
                    d2 = (oy - py) ** 2 + (ox - px) ** 2
                    j = int(np.argmin(d2))
                    hit[oy[j], ox[j]] = True
        if hit.any():
            out[z] = hit
    return out


def geometric_ear_tip_hits(
    voxel: np.ndarray, side: str, min_n: int = 1
) -> Dict[int, np.ndarray]:
    """Black ear tips from top-layer silhouette peaks (L/R by voxel X)."""
    fx, fz, h = voxel.shape
    # Search top 30% of layers for occupied cells
    z0 = max(0, int(h * 0.70))
    hits: Dict[int, np.ndarray] = {}
    mid = fx / 2.0
    best = None  # (topness, z, x, d)
    for z in range(h - 1, z0 - 1, -1):
        occ = voxel[:, :, z] > 0
        xs, ds = np.where(occ)
        if len(xs) == 0:
            continue
        for x, d in zip(xs, ds):
            if side == "left" and x >= mid:
                continue
            if side == "right" and x < mid:
                continue
            # Prefer highest layer, then extreme X
            score = (z, abs(x - mid), -d)  # high z, far from mid
            if best is None or score > best[0]:
                best = (score, z, int(x), int(d))
    if best is None:
        # Fallback: ignore mid split, take global top extremes
        for z in range(h - 1, z0 - 1, -1):
            occ = voxel[:, :, z] > 0
            xs, ds = np.where(occ)
            if len(xs) == 0:
                continue
            if side == "left":
                j = int(np.argmin(xs))
            else:
                j = int(np.argmax(xs))
            best = ((z, 0, 0), z, int(xs[j]), int(ds[j]))
            break
    if best is None:
        return {}
    _, z, x, d = best
    hit = np.zeros((fx, fz), dtype=bool)
    hit[x, d] = True
    # Dilate within occupancy toward min_n
    layer_hits = {z: hit}
    return _dilate_hits(layer_hits, voxel, min_n, max_steps=3)


def geometric_eye_hits(
    voxel: np.ndarray, side: str, min_n: int = 1
) -> Dict[int, np.ndarray]:
    """Eyes on upper-face layers: L/R by voxel X, forward-ish depth."""
    fx, fz, h = voxel.shape
    # Face band ~ 55–85% height
    z_lo = max(0, int(h * 0.55))
    z_hi = min(h, int(h * 0.85))
    mid = fx / 2.0
    best = None
    for z in range(z_hi - 1, z_lo - 1, -1):
        occ = voxel[:, :, z] > 0
        xs, ds = np.where(occ)
        if len(xs) == 0:
            continue
        for x, d in zip(xs, ds):
            if side == "left" and x >= mid - 0.5:
                continue
            if side == "right" and x < mid:
                continue
            # Prefer forward (high depth index often = front depending on yaw);
            # with yaw=0 Pikachu, front tends toward higher z-depth in this mesh.
            score = (z, d if True else 0, abs(x - mid))
            cand = (score, z, int(x), int(d))
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        return {}
    _, z, x, d = best
    hit = np.zeros((fx, fz), dtype=bool)
    hit[x, d] = True
    return _dilate_hits({z: hit}, voxel, min_n, max_steps=2)


def _cap_hits_to_cluster(
    layer_hits: Dict[int, np.ndarray],
    target: int,
    max_beads: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Keep up to max_beads cells nearest the feature centroid (compact cheeks/eyes)."""
    if max_beads is None:
        max_beads = max(target * 2, target)
    coords = []
    for z, hit in layer_hits.items():
        xs, ds = np.where(hit)
        for x, d in zip(xs, ds):
            coords.append((int(z), int(x), int(d)))
    if len(coords) <= max_beads:
        return layer_hits
    arr = np.array(coords, dtype=np.float64)
    centroid = arr.mean(axis=0)
    d2 = ((arr - centroid) ** 2).sum(axis=1)
    keep_idx = set(np.argsort(d2)[:max_beads].tolist())
    out: Dict[int, np.ndarray] = {}
    # Infer shape from first hit
    sample = next(iter(layer_hits.values()))
    for i, (z, x, d) in enumerate(coords):
        if i not in keep_idx:
            continue
        if z not in out:
            out[z] = np.zeros_like(sample)
        out[z][x, d] = True
    return out


def _count_hits(layer_hits: Dict[int, np.ndarray]) -> int:
    return int(sum(int(h.sum()) for h in layer_hits.values()))


def _dilate_hits(
    layer_hits: Dict[int, np.ndarray],
    voxel: np.ndarray,
    min_count: int,
    max_steps: int = 4,
) -> Dict[int, np.ndarray]:
    """Grow feature within occupied layer silhouette until min_count."""
    if min_count <= 0 or _count_hits(layer_hits) >= min_count:
        return layer_hits
    kernel = np.ones((3, 3), np.uint8)
    hits = {z: h.copy() for z, h in layer_hits.items()}
    for _ in range(max_steps):
        if _count_hits(hits) >= min_count:
            break
        for z in list(hits.keys()):
            occ = voxel[:, :, z] > 0
            grown = cv2.dilate(hits[z].astype(np.uint8), kernel, iterations=1).astype(
                bool
            )
            hits[z] = grown & occ
    return hits


def _paint(colors: np.ndarray, layer_hits: Dict[int, np.ndarray], rgb: Tuple[int, int, int]):
    for z, hit in layer_hits.items():
        colors[:, :, z][hit] = np.array(rgb, dtype=np.uint8)


def image_project_crop_box(subject_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Tight subject bbox + pad — must match _color_volume_image_project."""
    ys, xs = np.where(subject_mask)
    h, w = subject_mask.shape[:2]
    if len(xs) < 20:
        return 0, h, 0, w
    y0b, y1b = int(ys.min()), int(ys.max()) + 1
    x0b, x1b = int(xs.min()), int(xs.max()) + 1
    pad_y = max(1, (y1b - y0b) // 20)
    pad_x = max(1, (x1b - x0b) // 20)
    y0b = max(0, y0b - pad_y)
    y1b = min(h, y1b + pad_y)
    x0b = max(0, x0b - pad_x)
    x1b = min(w, x1b + pad_x)
    return y0b, y1b, x0b, x1b


def apply_feature_protect(
    colors: np.ndarray,
    voxel: np.ndarray,
    framed_rgb: np.ndarray,
    subject_mask: np.ndarray,
    *,
    min_cheek: int = 2,
    min_eye: int = 1,
    min_ear_tip: int = 1,
    debug_dir: Optional[str] = None,
    eye_geometric_fallback: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Detect features and overwrite voxel colors. Returns (colors, report)."""
    masks = detect_features(
        framed_rgb,
        subject_mask,
        eye_geometric_fallback=eye_geometric_fallback,
    )
    if debug_dir:
        save_feature_masks(masks, framed_rgb, debug_dir)

    # Mapping crop must match image_project (flood-fill subject), not grabCut.
    box = image_project_crop_box(subject_mask)
    masks.crop_box = box
    out = colors.copy()
    report = {"notes": list(masks.notes), "features": {}}

    def _inject(
        name: str,
        mask: Optional[np.ndarray],
        rgb: Tuple[int, int, int],
        min_n: int,
        geometric: Optional[str] = None,
    ):
        detected = mask is not None and mask.any()
        hits: Dict[int, np.ndarray] = {}
        method = "none"
        if detected:
            cropped = _crop_mask(mask, box)
            layers = _layers_for_mask(cropped, voxel.shape[2])
            hits = map_mask_to_layer_hits(cropped, voxel, layers, snap_to_occupied=True)
            if _count_hits(hits) == 0:
                hits = map_mask_to_layer_hits(
                    cropped, voxel, list(range(voxel.shape[2])), snap_to_occupied=True
                )
            hits = _dilate_hits(hits, voxel, min_n)
            # Cheeks/eyes: keep a compact cluster (≥min, ≤~2×min) not the whole blob
            if name.startswith("cheek"):
                hits = _cap_hits_to_cluster(hits, min_n, max_beads=max(4, min_n * 2))
            elif name.startswith("eye"):
                hits = _cap_hits_to_cluster(hits, min_n, max_beads=max(2, min_n * 2))
            elif name.startswith("ear"):
                hits = _cap_hits_to_cluster(hits, min_n, max_beads=max(2, min_n * 2))
            det_m = (masks.detection_methods or {}).get(name, "")
            if isinstance(det_m, str) and det_m.startswith("photo"):
                method = det_m
            else:
                method = "photo_map"
        if _count_hits(hits) < min_n and geometric == "ear_left":
            hits = geometric_ear_tip_hits(voxel, "left", min_n)
            method = "geometric_ear"
            report["notes"].append(f"{name}_geometric_ear")
        elif _count_hits(hits) < min_n and geometric == "ear_right":
            hits = geometric_ear_tip_hits(voxel, "right", min_n)
            method = "geometric_ear"
            report["notes"].append(f"{name}_geometric_ear")
        elif _count_hits(hits) < min_n and geometric == "eye_left":
            hits = geometric_eye_hits(voxel, "left", min_n)
            method = "geometric_eye"
            report["notes"].append(f"{name}_geometric_eye")
        elif _count_hits(hits) < min_n and geometric == "eye_right":
            hits = geometric_eye_hits(voxel, "right", min_n)
            method = "geometric_eye"
            report["notes"].append(f"{name}_geometric_eye")
        painted = _count_hits(hits)
        if painted > 0:
            _paint(out, hits, rgb)
        report["features"][name] = {
            "detected": bool(detected),
            "painted": painted,
            "min": min_n,
            "layers": sorted(hits.keys()),
            "method": method,
            "ok": painted >= min_n,
        }

    _inject("cheeks_left", masks.cheeks_left, RED, min_cheek)
    _inject("cheeks_right", masks.cheeks_right, RED, min_cheek)
    eye_geo_l = "eye_left" if eye_geometric_fallback else None
    eye_geo_r = "eye_right" if eye_geometric_fallback else None
    _inject("eyes_left", masks.eyes_left, BLACK, min_eye, geometric=eye_geo_l)
    _inject("eyes_right", masks.eyes_right, BLACK, min_eye, geometric=eye_geo_r)
    _inject(
        "ear_tips_left", masks.ear_tips_left, BLACK, min_ear_tip, geometric="ear_left"
    )
    _inject(
        "ear_tips_right",
        masks.ear_tips_right,
        BLACK,
        min_ear_tip,
        geometric="ear_right",
    )
    if masks.mouth is not None and masks.mouth.any():
        _inject("mouth", masks.mouth, BLACK, 1)

    report["ok"] = all(
        report["features"].get(k, {}).get("ok", False)
        for k in (
            "cheeks_left",
            "cheeks_right",
            "eyes_left",
            "eyes_right",
            "ear_tips_left",
            "ear_tips_right",
        )
    )
    return out, report



REQUIRED_FEATURES = (
    "cheeks_left",
    "cheeks_right",
    "eyes_left",
    "eyes_right",
    "ear_tips_left",
    "ear_tips_right",
)


def evaluate_feature_visibility(
    feat_report: Optional[dict],
    *,
    enabled: bool = True,
) -> dict:
    """Verification feature_visibility gate (no color histogram).

    Score = mean(painted/min) over required features. Cheeks/ears: pass when
    painted≥min (geometric ear allowed). Eyes (Epic 2 hard rule): FAIL if
    detected is false, method starts with "geometric", or painted < min.
    Notes are recorded but never auto-pass a short or geometric eye.
    """
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "score": None,
            "details": {},
            "notes": [],
            "fail_reason": None,
        }
    if not feat_report:
        return {
            "enabled": True,
            "passed": False,
            "score": 0.0,
            "details": {},
            "notes": [],
            "fail_reason": "missing_feature_protect_report",
        }

    features = feat_report.get("features") or {}
    details = {}
    fracs = []
    fails = []
    for name in REQUIRED_FEATURES:
        feat = features.get(name) or {}
        painted = int(feat.get("painted", 0) or 0)
        min_n = int(feat.get("min", 1) or 1)
        detected = bool(feat.get("detected", False))
        method = str(feat.get("method") or "none")
        frac = 1.0 if min_n <= 0 else min(1.0, painted / float(min_n))
        is_eye = name.startswith("eyes_")
        geometric = method.lower().startswith("geometric")
        count_ok = painted >= min_n
        if is_eye:
            # Epic 2: photo detection required — geometric seed is a hard fail.
            ok = bool(detected) and (not geometric) and count_ok
            if not ok:
                parts = []
                if not detected:
                    parts.append("detected=false")
                if geometric:
                    parts.append(f"method={method}")
                if not count_ok:
                    parts.append(f"painted={painted} < min={min_n}")
                fails.append(f"{name}: " + ", ".join(parts))
        else:
            ok = count_ok
            if not ok:
                fails.append(
                    f"{name}: painted={painted} < min={min_n} "
                    f"(detected={detected}, method={method})"
                )
        details[name] = {
            "painted": painted,
            "min": min_n,
            "detected": detected,
            "method": method,
            "ok": ok,
            "visibility": frac,
        }
        fracs.append(frac)

    score = float(sum(fracs) / len(fracs)) if fracs else 0.0
    passed = len(fails) == 0
    return {
        "enabled": True,
        "passed": passed,
        "score": score,
        "details": details,
        "notes": list(feat_report.get("notes") or []),
        "fail_reason": None if passed else "; ".join(fails),
        "eye_rule": "photo_detected_no_geometric",
    }


def reserved_feature_colors() -> List[Tuple[int, int, int]]:
    """Palette RGBs that limit_colors must keep when feature_protect is on."""
    return [BLACK, RED]
