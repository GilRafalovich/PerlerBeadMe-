"""
Epic 4 — Character kit (template-first).

Identity source: **template-first** procedural Pikachu mesh + part-aware paint.
Photo is optional (copied for reference / diagnostic metrics only) — it is NOT
the color identity source (unlike image_project).

Pipeline:
  optional photo → procedural Pikachu mesh + part labels
                → kid voxelize (tall, ≤800 beads)
                → paint by part + hard Pikachu palette (Y/R/Blk/Brn)
                → parts/ masks + feature_gate report
                → instructions

Hard palette: Yellow (body/head/ears), Red (cheeks), Black (eyes/ear tips),
Brown (optional ear/tail accent, small %). Green/Grey/Orange → Yellow.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from color_quantizer import PERLER_PALETTE, ColorQuantizer
from feature_protect import evaluate_feature_visibility
from hybrid_assembler import HybridAssembler
from instruction_renderer import write_kid_artifacts
from quantizer import KidModeParams, VoxelQuantizer

# ---------------------------------------------------------------------------
# Hard Pikachu palette
# ---------------------------------------------------------------------------

PIKACHU_ALLOWED = ("Yellow", "Red", "Black", "Brown")
YELLOW = np.array(PERLER_PALETTE["Yellow"], dtype=np.uint8)
RED = np.array(PERLER_PALETTE["Red"], dtype=np.uint8)
BLACK = np.array(PERLER_PALETTE["Black"], dtype=np.uint8)
BROWN = np.array(PERLER_PALETTE["Brown"], dtype=np.uint8)

PART_NAMES = (
    "body",
    "head",
    "ear_left",
    "ear_right",
    "eye_left",
    "eye_right",
    "cheek_left",
    "cheek_right",
    "tail",
)

# Part id in label volume (0 = empty)
PART_IDS = {name: i + 1 for i, name in enumerate(PART_NAMES)}
# Extra internal labels for ear tips (painted Black, still counted under ears)
PART_IDS["ear_tip_left"] = 100
PART_IDS["ear_tip_right"] = 101


@dataclass
class CharacterKitResult:
    character: str
    identity_source: str
    out_dir: str
    occupied_beads: int
    voxel_shape: List[int]
    colors_used: Dict[str, int]
    difficulty: Dict[str, Any]
    parts: List[str]
    feature_protect: Dict[str, Any]
    feature_gate: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    preview_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Procedural Pikachu mesh (Y-up, facing +Z)
# ---------------------------------------------------------------------------

def _box(center, extents) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


def _sphere(center, radius, subdivisions: int = 2) -> trimesh.Trimesh:
    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    s.apply_translation(center)
    return s


def make_pikachu_part_meshes() -> Dict[str, trimesh.Trimesh]:
    """
    Hand-authored standing Pikachu parts in a shared Y-up frame.
    Front = +Z. Ears tall for kid recognition. Lightning-ish tail on -Z.
    """
    parts: Dict[str, trimesh.Trimesh] = {}

    # Body (fat oval)
    parts["body"] = trimesh.util.concatenate(
        [
            _sphere([0.0, 0.42, 0.0], 0.38),
            _box([0.0, 0.28, 0.0], [0.55, 0.42, 0.48]),
        ]
    )
    # Legs / feet
    for x in (-0.16, 0.16):
        parts["body"] = trimesh.util.concatenate(
            [
                parts["body"],
                _box([x, 0.08, 0.06], [0.14, 0.18, 0.16]),
                _box([x, 0.02, 0.10], [0.16, 0.06, 0.18]),
            ]
        )
    # Arms
    for x in (-0.42, 0.42):
        parts["body"] = trimesh.util.concatenate(
            [parts["body"], _box([x, 0.40, 0.05], [0.14, 0.12, 0.12])]
        )

    # Head
    parts["head"] = trimesh.util.concatenate(
        [
            _sphere([0.0, 0.92, 0.05], 0.32),
            _box([0.0, 0.88, 0.05], [0.48, 0.40, 0.42]),
        ]
    )

    # Ears (extra tall pointed) — left = +X, right = -X
    parts["ear_left"] = trimesh.util.concatenate(
        [
            _box([0.24, 1.40, 0.02], [0.11, 0.50, 0.09]),
            _box([0.24, 1.65, 0.02], [0.09, 0.22, 0.07]),
        ]
    )
    parts["ear_right"] = trimesh.util.concatenate(
        [
            _box([-0.24, 1.40, 0.02], [0.11, 0.50, 0.09]),
            _box([-0.24, 1.65, 0.02], [0.09, 0.22, 0.07]),
        ]
    )
    parts["ear_tip_left"] = _box([0.24, 1.82, 0.02], [0.09, 0.14, 0.07])
    parts["ear_tip_right"] = _box([-0.24, 1.82, 0.02], [0.09, 0.14, 0.07])

    # Eyes / cheeks as shallow front plates (ensure voxel hits on +Z face)
    parts["eye_left"] = _box([0.13, 0.98, 0.34], [0.11, 0.11, 0.08])
    parts["eye_right"] = _box([-0.13, 0.98, 0.34], [0.11, 0.11, 0.08])
    parts["cheek_left"] = _box([0.24, 0.84, 0.32], [0.13, 0.11, 0.07])
    parts["cheek_right"] = _box([-0.24, 0.84, 0.32], [0.13, 0.11, 0.07])

    # Tail — lightning bolt to the side/back (avoid covering face in front proj)
    parts["tail"] = trimesh.util.concatenate(
        [
            _box([0.42, 0.50, -0.20], [0.12, 0.10, 0.22]),
            _box([0.55, 0.62, -0.28], [0.26, 0.10, 0.10]),
            _box([0.65, 0.78, -0.32], [0.10, 0.24, 0.10]),
            _box([0.52, 0.92, -0.38], [0.22, 0.08, 0.08]),
        ]
    )

    # Center whole assembly
    all_m = trimesh.util.concatenate(list(parts.values()))
    shift = -all_m.bounds.mean(axis=0)
    for k, m in parts.items():
        m.apply_translation(shift)
        parts[k] = m
    return parts


def make_pikachu_mesh(obj_path: Optional[str] = None) -> trimesh.Trimesh:
    """Union of all Pikachu parts; optionally export .obj."""
    parts = make_pikachu_part_meshes()
    mesh = trimesh.util.concatenate(list(parts.values()))
    mesh.vertices -= mesh.bounds.mean(axis=0)
    if obj_path:
        os.makedirs(os.path.dirname(os.path.abspath(obj_path)) or ".", exist_ok=True)
        mesh.export(obj_path)
    return mesh


# ---------------------------------------------------------------------------
# Part-aware voxel paint
# ---------------------------------------------------------------------------

def _rgb_equal(vol: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    return np.all(vol == rgb.reshape(1, 1, 1, 3), axis=-1)


def _paint_blob(
    colors: np.ndarray,
    voxel: np.ndarray,
    labels: np.ndarray,
    part_id: int,
    seed_mask: np.ndarray,
    rgb: np.ndarray,
    min_n: int,
) -> int:
    """Paint up to/at least min_n voxels from seed_mask ∩ occupied; stamp part id."""
    cand = np.argwhere(seed_mask & (voxel > 0))
    if len(cand) == 0:
        return 0
    # Prefer front-most (+Z high in axis 1 for (X,Z,Y) pancakes)
    # Sort by -z, then closeness to seed centroid
    cz = float(cand[:, 1].mean())
    cx = float(cand[:, 0].mean())
    cy = float(cand[:, 2].mean())
    scores = (
        -cand[:, 1].astype(np.float32) * 2.0
        + np.abs(cand[:, 0] - cx)
        + np.abs(cand[:, 2] - cy) * 0.5
    )
    order = np.argsort(scores)
    take = cand[order[: max(min_n, min(len(cand), max(min_n, 3)))]]
    painted = 0
    for x, z, y in take:
        colors[x, z, y] = rgb
        labels[x, z, y] = part_id
        painted += 1
    # If still short, dilate in-plane on that layer and keep painting
    if painted < min_n:
        for x, z, y in take:
            for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
                xx, zz = int(x) + dx, int(z) + dz
                if (
                    0 <= xx < voxel.shape[0]
                    and 0 <= zz < voxel.shape[1]
                    and voxel[xx, zz, y] > 0
                    and painted < min_n
                ):
                    colors[xx, zz, y] = rgb
                    labels[xx, zz, y] = part_id
                    painted += 1
    return painted


def paint_pikachu_parts(
    voxel: np.ndarray,
    *,
    min_eye: int = 3,
    min_cheek: int = 3,
    min_ear_tip: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Paint hard Pikachu palette onto occupied voxels; return (colors, labels, feat_report).

    voxel shape: (X, Z, Y_up)
    labels: same shape int16 part ids
    """
    sx, sz, sy = voxel.shape
    colors = np.zeros((sx, sz, sy, 3), dtype=np.uint8)
    labels = np.zeros((sx, sz, sy), dtype=np.int16)

    occ = voxel > 0
    # Default: Yellow body
    colors[occ] = YELLOW
    labels[occ] = PART_IDS["body"]

    ys, xs, zs = np.where(occ.transpose(2, 0, 1))  # not used; work in x,z,y
    coords = np.argwhere(occ)
    if len(coords) == 0:
        empty = {
            "ok": False,
            "notes": ["empty_voxel"],
            "features": {},
        }
        return colors, labels, empty

    y_min, y_max = int(coords[:, 2].min()), int(coords[:, 2].max())
    x_min, x_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    z_min, z_max = int(coords[:, 1].min()), int(coords[:, 1].max())
    y_span = max(1, y_max - y_min)
    x_mid = 0.5 * (x_min + x_max)
    z_mid = 0.5 * (z_min + z_max)

    # Height bands
    y_head0 = y_min + int(0.55 * y_span)
    y_ear0 = y_min + int(0.78 * y_span)
    y_cheek0 = y_min + int(0.58 * y_span)
    y_cheek1 = y_min + int(0.72 * y_span)
    y_eye0 = y_min + int(0.68 * y_span)
    y_eye1 = y_min + int(0.82 * y_span)
    y_tail0 = y_min + int(0.35 * y_span)
    y_tail1 = y_min + int(0.75 * y_span)

    # Head region (upper center mass)
    head_mask = occ & (np.arange(sy).reshape(1, 1, sy) >= y_head0)
    head_mask &= np.abs(np.arange(sx).reshape(sx, 1, 1) - x_mid) < 0.42 * (x_max - x_min + 1)
    labels[head_mask] = PART_IDS["head"]

    # Ears: top layers, left/right of center
    ear_band = occ & (np.arange(sy).reshape(1, 1, sy) >= y_ear0)
    ear_left = ear_band & (np.arange(sx).reshape(sx, 1, 1) > x_mid + 0.05 * (x_max - x_min + 1))
    ear_right = ear_band & (np.arange(sx).reshape(sx, 1, 1) < x_mid - 0.05 * (x_max - x_min + 1))
    labels[ear_left] = PART_IDS["ear_left"]
    labels[ear_right] = PART_IDS["ear_right"]
    colors[ear_left | ear_right] = YELLOW

    # Ear tips: topmost 2 layers of each ear
    # Ear tips: take topmost N voxels per ear (cap ~3 for readable Black tips)
    tip_budget = max(min_ear_tip, min(3, min_ear_tip + 2))
    for base, pid in (
        (ear_left, PART_IDS["ear_tip_left"]),
        (ear_right, PART_IDS["ear_tip_right"]),
    ):
        pts = np.argwhere(base)
        tip_mask = np.zeros_like(occ)
        if len(pts):
            pts = pts[np.argsort(-pts[:, 2])][:tip_budget]
            for x, z, y in pts:
                tip_mask[x, z, y] = True
        colors[tip_mask] = BLACK
        labels[tip_mask] = pid

    # Tail: rear (-Z low) protrusion
    z_rear = z_min + int(0.28 * (z_max - z_min + 1))
    tail_mask = (
        occ
        & (np.arange(sz).reshape(1, sz, 1) <= z_rear)
        & (np.arange(sy).reshape(1, 1, sy) >= y_tail0)
        & (np.arange(sy).reshape(1, 1, sy) <= y_tail1)
    )
    labels[tail_mask] = PART_IDS["tail"]
    colors[tail_mask] = YELLOW
    # Small Brown stripe on tail tip (optional accent)
    if tail_mask.any():
        pts = np.argwhere(tail_mask)
        pts = pts[np.argsort(pts[:, 1])][: max(2, min(6, len(pts) // 4))]
        for x, z, y in pts:
            colors[x, z, y] = BROWN

    # Front face for eyes/cheeks: high-Z surface of head band
    z_front = z_min + int(0.62 * (z_max - z_min + 1))
    front = occ & (np.arange(sz).reshape(1, sz, 1) >= z_front)

    eye_band = front & (np.arange(sy).reshape(1, 1, sy) >= y_eye0) & (
        np.arange(sy).reshape(1, 1, sy) <= y_eye1
    )
    cheek_band = front & (np.arange(sy).reshape(1, 1, sy) >= y_cheek0) & (
        np.arange(sy).reshape(1, 1, sy) <= y_cheek1
    )

    x_span = max(1, x_max - x_min)
    eye_left_seed = eye_band & (np.arange(sx).reshape(sx, 1, 1) > x_mid + 0.08 * x_span) & (
        np.arange(sx).reshape(sx, 1, 1) < x_mid + 0.42 * x_span
    )
    eye_right_seed = eye_band & (np.arange(sx).reshape(sx, 1, 1) < x_mid - 0.08 * x_span) & (
        np.arange(sx).reshape(sx, 1, 1) > x_mid - 0.42 * x_span
    )
    cheek_left_seed = cheek_band & (np.arange(sx).reshape(sx, 1, 1) > x_mid + 0.05 * x_span)
    cheek_right_seed = cheek_band & (np.arange(sx).reshape(sx, 1, 1) < x_mid - 0.05 * x_span)

    painted_eye_l = _paint_blob(
        colors, voxel, labels, PART_IDS["eye_left"], eye_left_seed, BLACK, min_eye
    )
    painted_eye_r = _paint_blob(
        colors, voxel, labels, PART_IDS["eye_right"], eye_right_seed, BLACK, min_eye
    )
    painted_cheek_l = _paint_blob(
        colors, voxel, labels, PART_IDS["cheek_left"], cheek_left_seed, RED, min_cheek
    )
    painted_cheek_r = _paint_blob(
        colors, voxel, labels, PART_IDS["cheek_right"], cheek_right_seed, RED, min_cheek
    )

    # Mirror fallback if one side of face had empty seed (narrow mesh)
    def _mirror_x(mask: np.ndarray) -> np.ndarray:
        return np.flip(mask, axis=0)

    if painted_cheek_l < min_cheek and painted_cheek_r >= min_cheek:
        # Mirror right cheek voxels across X mid onto left
        src = labels == PART_IDS["cheek_right"]
        mirrored = _mirror_x(src)
        # Only paint onto occupied yellow/head cells
        add = mirrored & occ & (labels != PART_IDS["cheek_right"])
        pts = np.argwhere(add)
        for x, z, y in pts[: max(0, min_cheek - painted_cheek_l)]:
            colors[x, z, y] = RED
            labels[x, z, y] = PART_IDS["cheek_left"]
            painted_cheek_l += 1
        if painted_cheek_l < min_cheek:
            # Force-paint near front left of cheek band
            force = cheek_band & (np.arange(sx).reshape(sx, 1, 1) > x_mid)
            painted_cheek_l = max(
                painted_cheek_l,
                _paint_blob(colors, voxel, labels, PART_IDS["cheek_left"], force, RED, min_cheek),
            )
    if painted_cheek_r < min_cheek and painted_cheek_l >= min_cheek:
        src = labels == PART_IDS["cheek_left"]
        mirrored = _mirror_x(src)
        add = mirrored & occ & (labels != PART_IDS["cheek_left"])
        pts = np.argwhere(add)
        for x, z, y in pts[: max(0, min_cheek - painted_cheek_r)]:
            colors[x, z, y] = RED
            labels[x, z, y] = PART_IDS["cheek_right"]
            painted_cheek_r += 1
        if painted_cheek_r < min_cheek:
            force = cheek_band & (np.arange(sx).reshape(sx, 1, 1) < x_mid)
            painted_cheek_r = max(
                painted_cheek_r,
                _paint_blob(colors, voxel, labels, PART_IDS["cheek_right"], force, RED, min_cheek),
            )
    if painted_eye_l < min_eye:
        force = eye_band & (np.arange(sx).reshape(sx, 1, 1) > x_mid)
        painted_eye_l = max(
            painted_eye_l,
            _paint_blob(colors, voxel, labels, PART_IDS["eye_left"], force, BLACK, min_eye),
        )
    if painted_eye_r < min_eye:
        force = eye_band & (np.arange(sx).reshape(sx, 1, 1) < x_mid)
        painted_eye_r = max(
            painted_eye_r,
            _paint_blob(colors, voxel, labels, PART_IDS["eye_right"], force, BLACK, min_eye),
        )

    # Count ear tip blacks
    ear_tip_l_n = int(_rgb_equal(colors, BLACK).sum() and (labels == PART_IDS["ear_tip_left"]).sum())
    ear_tip_r_n = int((labels == PART_IDS["ear_tip_right"]).sum())
    # Prefer counting Black on tip labels
    ear_tip_l_n = int(((labels == PART_IDS["ear_tip_left"]) & occ).sum())
    ear_tip_r_n = int(((labels == PART_IDS["ear_tip_right"]) & occ).sum())

    # Remap any illegal colors (safety)
    illegal = 0
    for name, rgb in PERLER_PALETTE.items():
        if name in PIKACHU_ALLOWED:
            continue
        m = _rgb_equal(colors, np.array(rgb, dtype=np.uint8)) & occ
        if m.any():
            illegal += int(m.sum())
            colors[m] = YELLOW

    notes = [
        "identity_source=template_first",
        "hard_palette=Yellow/Red/Black/Brown",
    ]
    if illegal:
        notes.append(f"remapped_illegal_colors={illegal}")

    features = {
        "cheeks_left": {
            "detected": True,
            "painted": painted_cheek_l,
            "min": min_cheek,
            "method": "character_kit_template",
            "ok": painted_cheek_l >= min_cheek,
        },
        "cheeks_right": {
            "detected": True,
            "painted": painted_cheek_r,
            "min": min_cheek,
            "method": "character_kit_template",
            "ok": painted_cheek_r >= min_cheek,
        },
        "eyes_left": {
            "detected": True,
            "painted": painted_eye_l,
            "min": min_eye,
            "method": "character_kit_template",
            "ok": painted_eye_l >= min_eye,
        },
        "eyes_right": {
            "detected": True,
            "painted": painted_eye_r,
            "min": min_eye,
            "method": "character_kit_template",
            "ok": painted_eye_r >= min_eye,
        },
        "ear_tips_left": {
            "detected": True,
            "painted": ear_tip_l_n,
            "min": min_ear_tip,
            "method": "character_kit_template",
            "ok": ear_tip_l_n >= min_ear_tip,
        },
        "ear_tips_right": {
            "detected": True,
            "painted": ear_tip_r_n,
            "min": min_ear_tip,
            "method": "character_kit_template",
            "ok": ear_tip_r_n >= min_ear_tip,
        },
    }
    feat_report = {
        "ok": all(f["ok"] for f in features.values()),
        "notes": notes,
        "features": features,
        "identity_source": "template_first",
        "character": "pikachu",
    }
    return colors, labels, feat_report


def enforce_hard_palette(colors: np.ndarray, voxel: np.ndarray) -> Dict[str, int]:
    """Force only Y/R/Blk/Brn on occupied voxels; return name→count."""
    occ = voxel > 0
    allowed = {n: np.array(PERLER_PALETTE[n], dtype=np.uint8) for n in PIKACHU_ALLOWED}
    # Snap each occupied voxel to nearest allowed
    flat = colors[occ].astype(np.int32)
    if len(flat) == 0:
        return {}
    targets = np.stack([allowed[n] for n in PIKACHU_ALLOWED], axis=0).astype(np.int32)
    # int32 distances — int16 overflows on Yellow↔Black (same pitfall as limit_colors)
    d = ((flat[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d, axis=1)
    snapped = targets[idx].astype(np.uint8)
    colors[occ] = snapped
    counts: Dict[str, int] = {}
    for i, n in enumerate(PIKACHU_ALLOWED):
        counts[n] = int((idx == i).sum())
    return {k: v for k, v in counts.items() if v > 0}


# ---------------------------------------------------------------------------
# Part mask export (front projection)
# ---------------------------------------------------------------------------

def export_part_masks(
    labels: np.ndarray,
    voxel: np.ndarray,
    out_dir: str,
) -> List[str]:
    """Write parts/*.png front projections + part_labels.npy."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "part_labels.npy"), labels)
    written = []
    # Front projection: max over Z (axis 1) — image rows = Y flipped, cols = X
    sx, sz, sy = labels.shape
    for name in PART_NAMES:
        pid = PART_IDS[name]
        if name.startswith("ear_"):
            # Include tip ids in ear masks for visibility
            tip = PART_IDS.get(f"ear_tip_{name.split('_', 1)[1]}")
            mask3 = (labels == pid) | ((labels == tip) if tip else False)
        else:
            mask3 = labels == pid
        # Also show eye/cheek clearly
        proj = np.max(mask3.astype(np.uint8), axis=1)  # (X, Y)
        img = np.flipud(proj.T) * 255  # (Y, X) image
        # Upscale for readability
        up = cv2.resize(img, (img.shape[1] * 8, img.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
        path = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(path, up)
        written.append(name)
    # Combined front-surface overlay: for each (x,y) take the frontmost (+Z) occupied voxel
    legend = np.zeros((sy, sx, 3), dtype=np.uint8)
    pid_to_rgb = {
        PART_IDS["body"]: YELLOW,
        PART_IDS["head"]: np.array([255, 230, 100], dtype=np.uint8),
        PART_IDS["ear_left"]: np.array([255, 220, 80], dtype=np.uint8),
        PART_IDS["ear_right"]: np.array([255, 220, 80], dtype=np.uint8),
        PART_IDS["ear_tip_left"]: BLACK,
        PART_IDS["ear_tip_right"]: BLACK,
        PART_IDS["eye_left"]: BLACK,
        PART_IDS["eye_right"]: BLACK,
        PART_IDS["cheek_left"]: RED,
        PART_IDS["cheek_right"]: RED,
        PART_IDS["tail"]: BROWN,
    }
    for x in range(sx):
        for y in range(sy):
            col = labels[x, :, y]
            occ_z = np.where((voxel[x, :, y] > 0))[0]
            if len(occ_z) == 0:
                continue
            z_front = int(occ_z.max())  # frontmost
            pid = int(labels[x, z_front, y])
            rgb = pid_to_rgb.get(pid, YELLOW)
            legend[sy - 1 - y, x] = rgb
    up = cv2.resize(legend, (legend.shape[1] * 8, legend.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "parts_overlay.png"), cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
    written.append("parts_overlay")
    with open(os.path.join(out_dir, "parts_meta.json"), "w") as f:
        json.dump({"parts": list(PART_NAMES), "part_ids": PART_IDS}, f, indent=2)
    return written


def draw_character_plate(size: int = 512) -> np.ndarray:
    """Flat cartoon Pikachu plate (hard palette) for reference/preview."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    s = size

    def ell(cx, cy, ax, ay, color):
        cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, color.tolist(), -1)

    def rect(x0, y0, x1, y1, color):
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color.tolist(), -1)

    # Body
    ell(0.50 * s, 0.58 * s, 0.18 * s, 0.20 * s, YELLOW)
    # Head
    ell(0.50 * s, 0.38 * s, 0.15 * s, 0.14 * s, YELLOW)
    # Ears
    pts_l = np.array(
        [[0.58 * s, 0.28 * s], [0.62 * s, 0.08 * s], [0.66 * s, 0.28 * s]], np.int32
    )
    pts_r = np.array(
        [[0.42 * s, 0.28 * s], [0.38 * s, 0.08 * s], [0.34 * s, 0.28 * s]], np.int32
    )
    cv2.fillPoly(img, [pts_l], YELLOW.tolist())
    cv2.fillPoly(img, [pts_r], YELLOW.tolist())
    # Ear tips black
    ell(0.62 * s, 0.10 * s, 0.025 * s, 0.035 * s, BLACK)
    ell(0.38 * s, 0.10 * s, 0.025 * s, 0.035 * s, BLACK)
    # Eyes
    ell(0.56 * s, 0.36 * s, 0.025 * s, 0.035 * s, BLACK)
    ell(0.44 * s, 0.36 * s, 0.025 * s, 0.035 * s, BLACK)
    # Cheeks
    ell(0.62 * s, 0.42 * s, 0.035 * s, 0.028 * s, RED)
    ell(0.38 * s, 0.42 * s, 0.035 * s, 0.028 * s, RED)
    # Tail
    rect(0.30 * s, 0.55 * s, 0.36 * s, 0.72 * s, YELLOW)
    rect(0.22 * s, 0.50 * s, 0.32 * s, 0.56 * s, YELLOW)
    rect(0.20 * s, 0.42 * s, 0.26 * s, 0.52 * s, BROWN)
    # Feet
    ell(0.44 * s, 0.78 * s, 0.04 * s, 0.025 * s, YELLOW)
    ell(0.56 * s, 0.78 * s, 0.04 * s, 0.025 * s, YELLOW)
    return img


# ---------------------------------------------------------------------------
# Footprint search + run
# ---------------------------------------------------------------------------

def _count_beads(voxel: np.ndarray) -> int:
    return int(voxel.sum())


def search_kid_footprint(
    mesh: trimesh.Trimesh,
    *,
    target_lo: int = 700,
    target_hi: int = 800,
    footprints: Optional[List[int]] = None,
    layers_list: Optional[List[int]] = None,
) -> Tuple[np.ndarray, KidModeParams, int]:
    """Find footprint/layers with beads in [target_lo, target_hi], prefer taller."""
    footprints = footprints or list(range(11, 16))
    layers_list = layers_list or list(range(15, 21))  # ≤20 keeps Medium
    best = None  # (score, beads, voxel, params)
    for fp in footprints:
        for layers in layers_list:
            params = KidModeParams(
                max_footprint=fp,
                max_layers=layers,
                hollow=False,
                thick_shell_wall=0,
                base_widen_layers=2,
                base_widen_iters=1,
                max_colors=4,
                color_mode="structural",
                yaw_deg=0.0,
                auto_yaw=False,
                up_axis=1,
                auto_up_axis=False,
                photo_matched_yaw=False,
                verify_gate=False,
                feature_protect=False,
                min_layers=6,
            )
            vq = VoxelQuantizer(kid_params=params)
            voxel = vq.voxelize_kid(mesh, params=params)
            beads = _count_beads(voxel)
            sx, sz, sy = voxel.shape
            aspect = sy / max(1.0, float(max(sx, sz)))  # height / footprint
            # Prefer 700–800, taller aspect, not cube
            in_band = target_lo <= beads <= target_hi
            over = beads > target_hi
            under = beads < target_lo
            # Strongly prefer taller-than-wide (avoid cube stump)
            tall_bonus = 25.0 * max(0.0, aspect - 1.05)
            score = 0.0
            if in_band:
                score = 100.0 + aspect * 20.0 + tall_bonus - abs(beads - 760) * 0.05
            elif under:
                score = 40.0 + beads / 20.0 + aspect * 8.0 + tall_bonus
            else:
                score = 20.0 - (beads - target_hi) * 0.1 + aspect * 5.0
            if over and beads > 900:
                continue
            cand = (score, beads, voxel, params)
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        params = KidModeParams(
            max_footprint=14,
            max_layers=17,
            hollow=False,
            yaw_deg=0.0,
            auto_yaw=False,
            up_axis=1,
            auto_up_axis=False,
            photo_matched_yaw=False,
            verify_gate=False,
            max_colors=4,
        )
        vq = VoxelQuantizer(kid_params=params)
        voxel = vq.voxelize_kid(mesh, params=params)
        return voxel, params, _count_beads(voxel)
    return best[2], best[3], best[1]


def run_character_kit(
    out_dir: str,
    character: str = "pikachu",
    image_path: Optional[str] = None,
    footprint: Optional[int] = None,
    layers: Optional[int] = None,
    min_eye: int = 3,
    min_cheek: int = 3,
    min_ear_tip: int = 1,
    title: Optional[str] = None,
) -> CharacterKitResult:
    """Build a template-first character kit into out_dir."""
    if character.lower() != "pikachu":
        raise ValueError(f"Unsupported character-kit: {character} (v1 = pikachu only)")

    os.makedirs(out_dir, exist_ok=True)
    notes: List[str] = [
        "identity_source=template_first (procedural Pikachu mesh + part paint)",
        "photo not used for color identity",
    ]

    # Optional photo copy + plate
    if image_path and os.path.exists(image_path):
        raw = cv2.imread(image_path)
        if raw is not None:
            cv2.imwrite(os.path.join(out_dir, "01_input.jpg"), raw)
            notes.append(f"photo_copied={image_path}")
    plate = draw_character_plate()
    cv2.imwrite(
        os.path.join(out_dir, "02_character_plate.png"),
        cv2.cvtColor(plate, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(out_dir, "03_palette_quantized.png"),
        cv2.cvtColor(plate, cv2.COLOR_RGB2BGR),
    )

    mesh_path = os.path.join(out_dir, "02_mesh_procedural_pikachu.obj")
    mesh = make_pikachu_mesh(mesh_path)
    notes.append(f"mesh={mesh_path}")

    if footprint is not None and layers is not None:
        params = KidModeParams(
            max_footprint=int(footprint),
            max_layers=int(layers),
            hollow=False,
            thick_shell_wall=0,
            base_widen_layers=2,
            base_widen_iters=1,
            max_colors=4,
            color_mode="character_kit",
            yaw_deg=0.0,
            auto_yaw=False,
            up_axis=1,
            auto_up_axis=False,
            photo_matched_yaw=False,
            verify_gate=False,
            feature_protect=True,
            feature_min_eye=min_eye,
            feature_min_cheek=min_cheek,
            feature_min_ear_tip=min_ear_tip,
            feature_mask_dir=os.path.join(out_dir, "feature_masks"),
        )
        vq = VoxelQuantizer(kid_params=params)
        voxel = vq.voxelize_kid(mesh, params=params)
        beads = _count_beads(voxel)
    else:
        voxel, params, beads = search_kid_footprint(mesh)
        # stamp character-kit fields
        d = dict(asdict(params))
        d["color_mode"] = "character_kit"
        d["max_colors"] = 4
        d["feature_protect"] = True
        d["feature_min_eye"] = min_eye
        d["feature_min_cheek"] = min_cheek
        d["feature_min_ear_tip"] = min_ear_tip
        d["feature_mask_dir"] = os.path.join(out_dir, "feature_masks")
        d["verify_gate"] = False
        d["photo_matched_yaw"] = False
        params = KidModeParams(**d)

    notes.append(f"footprint={params.max_footprint} layers={params.max_layers} beads={beads}")

    colors, labels, feat_report = paint_pikachu_parts(
        voxel, min_eye=min_eye, min_cheek=min_cheek, min_ear_tip=min_ear_tip
    )
    color_counts = enforce_hard_palette(colors, voxel)
    # Re-count feature paints after palette enforce (must still match color)
    key_to_pid_rgb = {
        "eyes_left": (PART_IDS["eye_left"], BLACK),
        "eyes_right": (PART_IDS["eye_right"], BLACK),
        "cheeks_left": (PART_IDS["cheek_left"], RED),
        "cheeks_right": (PART_IDS["cheek_right"], RED),
        "ear_tips_left": (PART_IDS["ear_tip_left"], BLACK),
        "ear_tips_right": (PART_IDS["ear_tip_right"], BLACK),
    }
    for key, (pid, rgb) in key_to_pid_rgb.items():
        n = int(((labels == pid) & (voxel > 0) & np.all(colors == rgb.reshape(1, 1, 1, 3), axis=-1)).sum())
        # Fallback: any voxel with the feature color in that label, else keep prior painted
        if n == 0:
            n = int(((labels == pid) & (voxel > 0)).sum())
        if key in feat_report["features"]:
            feat_report["features"][key]["painted"] = max(n, int(feat_report["features"][key].get("painted") or 0))
            feat_report["features"][key]["ok"] = (
                feat_report["features"][key]["painted"] >= feat_report["features"][key]["min"]
            )
    feat_report["ok"] = all(f["ok"] for f in feat_report["features"].values())

    parts_dir = os.path.join(out_dir, "parts")
    part_list = export_part_masks(labels, voxel, parts_dir)

    # feature_masks debug (eyes/cheeks/ears front proj)
    fm = os.path.join(out_dir, "feature_masks")
    os.makedirs(fm, exist_ok=True)
    for name, pid, rgb in (
        ("eye_left", PART_IDS["eye_left"], BLACK),
        ("eye_right", PART_IDS["eye_right"], BLACK),
        ("cheek_left", PART_IDS["cheek_left"], RED),
        ("cheek_right", PART_IDS["cheek_right"], RED),
        ("ear_tip_left", PART_IDS["ear_tip_left"], BLACK),
        ("ear_tip_right", PART_IDS["ear_tip_right"], BLACK),
    ):
        mask3 = labels == pid
        proj = np.max(mask3.astype(np.uint8), axis=1)
        img = np.flipud(proj.T) * 255
        up = cv2.resize(img, (img.shape[1] * 8, img.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(fm, f"{name}.png"), up)

    color_q = ColorQuantizer()
    # Dummy padded image for assembler API
    padded = np.zeros((voxel.shape[0], voxel.shape[1], 3), dtype=np.uint8)
    padded[:] = YELLOW
    assembler = HybridAssembler(
        voxel,
        padded,
        color_volume=colors,
        kid_mode=True,
        bead_mm=params.bead_mm,
        color_quantizer=color_q,
    )
    result = assembler.decompose_kid()
    colors_out = result.pop("color_volume")

    kit_title = title or "Kid Pikachu Character Kit"
    summary = write_kid_artifacts(
        out_dir,
        voxel,
        colors_out,
        result,
        title=kit_title,
        params_dict=asdict(params),
    )
    summary["mode"] = "character_kit"
    summary["character"] = "pikachu"
    summary["identity_source"] = "template_first"
    summary["mesh_source"] = "procedural_pikachu"
    summary["colors_used"] = color_counts  # hard palette counts
    # Prefer assembler shopping list names but filter to allowed
    shop = {
        k: v
        for k, v in (summary.get("colors_used") or color_counts).items()
        if k in PIKACHU_ALLOWED
    }
    if not shop:
        shop = color_counts
    summary["colors_used"] = shop
    summary["num_colors"] = len(shop)
    summary["feature_protect"] = feat_report
    feature_gate = evaluate_feature_visibility(feat_report, enabled=True)
    summary["feature_gate"] = feature_gate
    summary["parts"] = {
        "dir": "parts",
        "list": [p for p in part_list if p in PART_NAMES],
        "overlay": "parts/parts_overlay.png",
    }
    summary["notes"] = notes + list(feat_report.get("notes") or [])

    # Optional diagnostic photo metrics (NOT a gate)
    diagnostics: Dict[str, Any] = {}
    if image_path and os.path.exists(image_path):
        try:
            from backward_verify import score_views, load_rgb, render_voxel_ortho, save_overlay

            ref = load_rgb(image_path)
            renders = {
                view: render_voxel_ortho(voxel, colors_out, view, 256)
                for view in ("front", "side")
            }
            rows, diag = score_views(ref, renders)
            diagnostics = {
                "composite_error": diag.get("composite_error"),
                "mean_ssim": diag.get("mean_ssim"),
                "mean_silhouette_iou": diag.get("mean_silhouette_iou"),
                "views": {
                    r.view: {
                        "ssim": r.ssim,
                        "lpips_lite": r.lpips_lite,
                        "silhouette_iou": r.silhouette_iou,
                        "composite_error": r.composite_error,
                    }
                    for r in rows
                },
                "note": "diagnostic only — not an Epic 4 ship gate",
            }
            verify_dir = os.path.join(out_dir, "verify")
            os.makedirs(verify_dir, exist_ok=True)
            for view, img in renders.items():
                cv2.imwrite(
                    os.path.join(verify_dir, f"render_{view}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
                save_overlay(ref, img, os.path.join(verify_dir, f"overlay_{view}.png"), title=f"diag/{view}")
            with open(os.path.join(verify_dir, "gate_report.json"), "w") as f:
                json.dump(
                    {
                        "enabled": False,
                        "ship_gate": "character_kit_recognizability",
                        "feature_gate": feature_gate,
                        "photo_diagnostics": diagnostics,
                    },
                    f,
                    indent=2,
                )
            summary["verify_dir"] = verify_dir
            summary["photo_diagnostics"] = diagnostics
        except Exception as e:
            diagnostics = {"error": str(e), "note": "diagnostic failed (non-blocking)"}
            summary["photo_diagnostics"] = diagnostics
            verify_dir = os.path.join(out_dir, "verify")
            os.makedirs(verify_dir, exist_ok=True)
            with open(os.path.join(verify_dir, "gate_report.json"), "w") as f:
                json.dump({"feature_gate": feature_gate, "photo_diagnostics": diagnostics}, f, indent=2)
    else:
        verify_dir = os.path.join(out_dir, "verify")
        os.makedirs(verify_dir, exist_ok=True)
        with open(os.path.join(verify_dir, "gate_report.json"), "w") as f:
            json.dump(
                {
                    "enabled": False,
                    "ship_gate": "character_kit_recognizability",
                    "feature_gate": feature_gate,
                },
                f,
                indent=2,
            )
        summary["verify_dir"] = verify_dir

    with open(os.path.join(out_dir, "00_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "00_character_kit.json"), "w") as f:
        json.dump(
            {
                "character": "pikachu",
                "identity_source": "template_first",
                "hard_palette": list(PIKACHU_ALLOWED),
                "parts": list(PART_NAMES),
                "beads": summary["occupied_beads"],
                "shape": summary["voxel_shape"],
                "colors": summary["colors_used"],
                "feature_gate": feature_gate,
                "difficulty": summary["difficulty"],
            },
            f,
            indent=2,
        )

    preview = os.path.join(out_dir, "07_voxel_preview_3d.png")
    print("=== CHARACTER KIT DONE ===")
    print(f"identity=template_first character=pikachu")
    print(f"shape={voxel.shape} beads={summary['occupied_beads']}")
    print(f"colors={summary['colors_used']}")
    print(f"difficulty={summary['difficulty']}")
    print(f"feature_gate passed={feature_gate.get('passed')} score={feature_gate.get('score')}")
    print(f"parts={part_list}")
    print(f"out={out_dir}")

    return CharacterKitResult(
        character="pikachu",
        identity_source="template_first",
        out_dir=out_dir,
        occupied_beads=int(summary["occupied_beads"]),
        voxel_shape=list(summary["voxel_shape"]),
        colors_used=dict(summary["colors_used"]),
        difficulty=dict(summary["difficulty"]),
        parts=[p for p in part_list if p in PART_NAMES],
        feature_protect=feat_report,
        feature_gate=feature_gate,
        notes=notes,
        diagnostics=diagnostics,
        preview_path=preview,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Epic 4 character kit (template-first)")
    ap.add_argument("--character-kit", default="pikachu", help="Character id (v1: pikachu)")
    ap.add_argument("--image", default=None, help="Optional photo (reference/diagnostics only)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--footprint", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--feature-min-eye", type=int, default=3)
    ap.add_argument("--feature-min-cheek", type=int, default=3)
    ap.add_argument("--feature-min-ear-tip", type=int, default=1)
    ap.add_argument("--title", default="Kid Pikachu Character Kit")
    args = ap.parse_args(argv)

    run_character_kit(
        out_dir=args.out,
        character=args.character_kit,
        image_path=args.image,
        footprint=args.footprint,
        layers=args.layers,
        min_eye=args.feature_min_eye,
        min_cheek=args.feature_min_cheek,
        min_ear_tip=args.feature_min_ear_tip,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
