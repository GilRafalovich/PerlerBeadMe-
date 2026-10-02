"""Spatial voxel quantization for Perler builds (standard relief + kid_mode pancakes)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import trimesh


@dataclass
class KidModeParams:
    """Kid-friendly stacked-pancake voxelization (Y-up layers, solid, small footprint)."""

    # Target longest horizontal extent in beads (XZ plane after Y-up).
    # Slightly larger than prototype 14 → more room for head/ears/legs silhouette.
    max_footprint: int = 18
    # Cap height layers (merge if taller). None = keep native height.
    max_layers: Optional[int] = 10
    # Min layers after merge
    min_layers: int = 6
    hollow: bool = False
    # Mild core carve (0=solid). Keeps silhouette; base layers stay solid.
    thick_shell_wall: int = 0
    # Morphological widen of bottom N layers for stable feet
    base_widen_layers: int = 2
    base_widen_iters: int = 1
    # Drop tiny floating components / dust
    min_component_beads: int = 2
    # Color: project from quantized photo
    color_mode: str = "structural"  # structural|image_project
    # Cap distinct Perler colors for kid contrast (None = no extra cap)
    max_colors: Optional[int] = 5
    bead_mm: float = 5.0  # standard Perler ~5mm
    # Standing / pancake up axis (0=X, 1=Y, 2=Z). Overridden when auto_up_axis.
    up_axis: int = 1
    # Detect tallest mesh axis (extents/PCA) so flat TripoSR meshes don't stay stubby.
    auto_up_axis: bool = True
    # Rotate mesh about up-axis to prefer side / 3/4 silhouette (degrees).
    # None = auto-pick yaw (photo-matched composite when ref image available).
    yaw_deg: Optional[float] = None
    auto_yaw: bool = True
    # When a reference RGB is available, minimize backward_verify composite_error
    # instead of silhouette-only feature score.
    photo_matched_yaw: bool = True
    # After voxelize: score vs ref; retry alt up_axis/yaw or fail clearly.
    verify_gate: bool = True
    # Tuned from Pikachu: baseline ~0.56, oriented combo ~0.43 → gate at 0.50.
    verify_error_threshold: float = 0.50
    verify_retry_alt_axes: bool = True
    # Crop empty margins tightly after voxelize (pad cells)
    crop_pad: int = 0


DEFAULT_KID_PARAMS = KidModeParams()


class VoxelQuantizer:
    def __init__(
        self,
        max_dim_xy: int = 60,
        max_dim_z: int = 4,
        kid_params: Optional[KidModeParams] = None,
    ):
        """
        Handles 3D spatial quantization of a .obj mesh.
        Default path: limits X/Y for detail, squashes Z to max_dim_z (relief).
        kid_mode path: Y-up pancakes, solid, small footprint (~14–18), base widen.
        """
        self.max_dim_xy = max_dim_xy
        self.max_dim_z = max_dim_z
        self.kid_params = kid_params or KidModeParams()

    def _hollow_out(self, matrix: np.ndarray) -> np.ndarray:
        """
        Removes internal voxels (the middle wall) to save beads and simplify assembly.
        A voxel is removed if it is completely surrounded in 3D space.
        Skipped in kid_mode (destroys silhouette / fragile rings).
        """
        hollowed = matrix.copy()
        for x in range(1, matrix.shape[0] - 1):
            for y in range(1, matrix.shape[1] - 1):
                for z in range(1, matrix.shape[2] - 1):
                    if matrix[x, y, z] == 1:
                        if (
                            matrix[x - 1, y, z] == 1
                            and matrix[x + 1, y, z] == 1
                            and matrix[x, y - 1, z] == 1
                            and matrix[x, y + 1, z] == 1
                            and matrix[x, y, z - 1] == 1
                            and matrix[x, y, z + 1] == 1
                        ):
                            hollowed[x, y, z] = 0
        return hollowed

    # ------------------------------------------------------------------
    # Kid-mode helpers (from kid_mode_prototype)
    # ------------------------------------------------------------------

    @staticmethod
    def _reorient_y_up_matrix(matrix: np.ndarray, up_axis: int) -> np.ndarray:
        """Return matrix with shape (X, Z, Y_up) so[:,:,k] is a horizontal pancake."""
        if up_axis == 2:
            return matrix
        if up_axis == 1:
            return np.transpose(matrix, (0, 2, 1))
        if up_axis == 0:
            return np.transpose(matrix, (1, 2, 0))
        raise ValueError(f"Unsupported up_axis: {up_axis}")

    @staticmethod
    def _crop_tight_bbox(matrix: np.ndarray, pad: int = 1) -> np.ndarray:
        occupied = np.argwhere(matrix > 0)
        if occupied.size == 0:
            return matrix
        mins = occupied.min(axis=0)
        maxs = occupied.max(axis=0)
        slices = []
        for d in range(3):
            lo = max(0, int(mins[d]) - pad)
            hi = min(matrix.shape[d], int(maxs[d]) + pad + 1)
            slices.append(slice(lo, hi))
        return matrix[slices[0], slices[1], slices[2]]

    @staticmethod
    def _squash_height(matrix: np.ndarray, target_h: int) -> np.ndarray:
        """OR-merge contiguous height bands into target_h plates."""
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

    @staticmethod
    def _widen_base(
        matrix: np.ndarray, n_layers: int = 2, iterations: int = 1
    ) -> np.ndarray:
        """Dilate bottom layers so feet form a wider stable platform."""
        out = matrix.copy()
        n = min(n_layers, out.shape[2])
        kernel = np.ones((3, 3), np.uint8)
        for z in range(n):
            layer = out[:, :, z].astype(np.uint8)
            if layer.sum() == 0:
                continue
            out[:, :, z] = cv2.dilate(layer, kernel, iterations=iterations)
        return out

    @staticmethod
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

    @staticmethod
    def _thick_shell(
        matrix: np.ndarray, wall: int = 1, solid_base_layers: int = 2
    ) -> np.ndarray:
        """Carve only deep interior cores per layer; keep silhouette and solid base."""
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

    @staticmethod
    def _rotate_yaw(mesh: trimesh.Trimesh, yaw_deg: float, up_axis: int = 1) -> trimesh.Trimesh:
        """Rotate mesh about up-axis (degrees)."""
        m = mesh.copy()
        rad = np.deg2rad(yaw_deg)
        if up_axis == 1:
            T = trimesh.transformations.rotation_matrix(rad, [0, 1, 0])
        elif up_axis == 2:
            T = trimesh.transformations.rotation_matrix(rad, [0, 0, 1])
        else:
            T = trimesh.transformations.rotation_matrix(rad, [1, 0, 0])
        m.apply_transform(T)
        return m

    @staticmethod
    def _silhouette_feature_score(pancakes: np.ndarray) -> float:
        """
        Score side-view silhouette for kid readability.
        Rewards: perimeter complexity, vertical span, multiple ground contacts (legs),
        top protrusions (ears/head), aspect not a solid blob.
        """
        if pancakes.size == 0 or int(pancakes.sum()) == 0:
            return -1e9
        # Side projection: occupancy along depth -> (height, width)
        side = (np.max(pancakes, axis=1) > 0).astype(np.uint8)  # (fx, h) -> transpose
        side = side.T  # (h, fx) with row0=bottom
        h, w = side.shape
        area = float(side.sum())
        if area < 1:
            return -1e9
        # Perimeter via morphological gradient
        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(side, kernel, iterations=1)
        ero = cv2.erode(side, kernel, iterations=1)
        peri = float((dil - ero).sum())
        # Ground contacts (legs): connected components on bottom 2 rows
        ground = side[: max(1, h // 5), :]
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(ground, connectivity=8)
        n_feet = max(0, n_labels - 1)
        # Top protrusions: components on top 25%
        top = side[int(h * 0.7) :, :]
        n_top, _, top_stats, _ = cv2.connectedComponentsWithStats(top, connectivity=8)
        n_ears = max(0, n_top - 1)
        # Prefer elongated side profile (not square blob)
        ys, xs = np.where(side > 0)
        aspect = (xs.max() - xs.min() + 1) / max(1, (ys.max() - ys.min() + 1))
        fill = area / max(1.0, (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        # Higher perimeter/area = more articulated; feet & ears bonus; penalize solid fill
        score = (peri / area) * 100.0 + n_feet * 25.0 + n_ears * 15.0
        score += min(aspect, 2.5) * 10.0
        score -= fill * 40.0
        return float(score)

    @staticmethod
    def detect_standing_axis(mesh: trimesh.Trimesh) -> Tuple[int, dict]:
        """
        Pick pancake up-axis from mesh extents (+ PCA agreement).

        TripoSR often emits a correct mesh lying flat (default Y shortest). Using
        the tallest axis as up avoids stubby kits. Warns when the chosen axis is
        the shortest (should not happen for auto) or when default Y is shortest.
        """
        extents = np.asarray(mesh.extents, dtype=np.float64)
        order = list(np.argsort(-extents))  # tallest first
        up = int(order[0])
        warnings: List[str] = []
        if int(np.argmin(extents)) == 1:
            warnings.append(
                f"default Y-up is shortest extent ({extents[1]:.3f}); "
                f"auto standing-axis → {up} (tallest={extents[up]:.3f})"
            )
        # PCA: largest variance axis should usually agree with tallest extent
        try:
            v = np.asarray(mesh.vertices, dtype=np.float64)
            v = v - v.mean(axis=0)
            cov = np.cov(v.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            pca_axis = int(np.argmax(np.abs(eigvecs[:, int(np.argmax(eigvals))])))
            if pca_axis != up:
                # Prefer extent tallest; note disagreement
                warnings.append(
                    f"PCA primary axis={pca_axis} disagrees with tallest extent={up}; "
                    "using tallest extent"
                )
        except Exception as e:
            pca_axis = None
            warnings.append(f"PCA skipped: {e}")

        if up == int(np.argmin(extents)):
            warnings.append(
                f"REJECT/WARN: chosen up_axis={up} is the shortest extent — flat mesh risk"
            )

        info = {
            "up_axis": up,
            "extents": [float(x) for x in extents],
            "sorted_axes_tallest_first": [int(i) for i in order],
            "pca_axis": pca_axis,
            "warnings": warnings,
            "is_shortest": up == int(np.argmin(extents)),
            "height_over_median_horiz": float(
                extents[up] / max(1e-9, float(np.median([extents[i] for i in range(3) if i != up])))
            ),
        }
        return up, info

    def _coarse_pancakes(
        self,
        mesh: trimesh.Trimesh,
        up_axis: int,
        yaw_deg: float,
        max_footprint: int,
        max_layers: Optional[int],
        coarse_fp: Optional[int] = None,
    ) -> np.ndarray:
        """Fast pancake occupancy for yaw / axis scoring."""
        m = self._rotate_yaw(mesh, yaw_deg, up_axis) if yaw_deg else mesh.copy()
        extents = m.extents
        horiz = [extents[i] for i in range(3) if i != up_axis]
        max_horiz = max(horiz) if horiz else float(np.max(extents))
        fp = coarse_fp or min(14, max(8, max_footprint))
        pitch = max_horiz / max(1, fp)
        vg = m.voxelized(pitch=pitch).fill()
        raw = vg.matrix.astype(np.uint8)
        pancakes = self._reorient_y_up_matrix(raw, up_axis)
        pancakes = self._crop_tight_bbox(pancakes, pad=0)
        if max_layers and pancakes.shape[2] > max_layers:
            pancakes = self._squash_height(pancakes, max_layers)
        return pancakes

    def _score_yaw_photo(
        self,
        pancakes: np.ndarray,
        ref_rgb: np.ndarray,
    ) -> float:
        """Lower is better: composite_error of front(+side) voxel render vs ref."""
        from backward_verify import (
            composite_error,
            render_voxel_ortho,
            score_views,
        )

        # Neutral gray colors — orientation is shape-only (no histogram)
        colors = np.zeros((*pancakes.shape, 3), dtype=np.uint8)
        colors[pancakes > 0] = (180, 180, 180)
        renders = {
            "front": render_voxel_ortho(pancakes, colors, "front", 128),
            "side": render_voxel_ortho(pancakes, colors, "side", 128),
        }
        _rows, summary = score_views(
            ref_rgb, renders, size=128, silhouette_weight=0.35
        )
        return float(summary["composite_error"])

    def _pick_best_yaw(
        self,
        mesh: trimesh.Trimesh,
        params: KidModeParams,
        candidates: Optional[list] = None,
        ref_rgb: Optional[np.ndarray] = None,
    ) -> float:
        """
        Try yaw angles.
        - With ref_rgb + photo_matched_yaw: minimize backward_verify composite_error.
        - Else: maximize side-silhouette feature score (legacy).
        """
        candidates = candidates or [
            0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 240, 270, 300, 330
        ]
        use_photo = (
            ref_rgb is not None
            and getattr(params, "photo_matched_yaw", True)
        )
        if use_photo:
            best_yaw, best_err = 0.0, 1e18
            for yaw in candidates:
                try:
                    pancakes = self._coarse_pancakes(
                        mesh,
                        params.up_axis,
                        float(yaw),
                        params.max_footprint,
                        params.max_layers,
                    )
                    if int(pancakes.sum()) == 0:
                        continue
                    err = self._score_yaw_photo(pancakes, ref_rgb)
                except Exception:
                    continue
                if err < best_err:
                    best_err, best_yaw = err, float(yaw)
            self._last_yaw_score = {"mode": "photo_matched", "composite_error": best_err}
            return best_yaw

        best_yaw, best_score = 0.0, -1e18
        for yaw in candidates:
            try:
                pancakes = self._coarse_pancakes(
                    mesh,
                    params.up_axis,
                    float(yaw),
                    params.max_footprint,
                    params.max_layers,
                )
                score = self._silhouette_feature_score(pancakes)
            except Exception:
                score = -1e18
            if score > best_score:
                best_score, best_yaw = score, float(yaw)
        self._last_yaw_score = {"mode": "silhouette_feature", "score": best_score}
        return best_yaw

    def voxelize_kid(
        self,
        mesh: trimesh.Trimesh,
        params: Optional[KidModeParams] = None,
        ref_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Voxelize mesh into solid Y-up pancake layers for kid builds."""
        params = params or self.kid_params
        work = mesh.copy()
        orient_info: Dict[str, Any] = {"warnings": []}

        # --- Standing-axis protection ---
        if getattr(params, "auto_up_axis", True):
            up, info = self.detect_standing_axis(work)
            orient_info["standing_axis"] = info
            chosen = dict(asdict(params))
            chosen["up_axis"] = up
            params = KidModeParams(**chosen)
            if info.get("is_shortest"):
                orient_info["warnings"].extend(info.get("warnings") or [])
        else:
            extents = np.asarray(work.extents, dtype=np.float64)
            if params.up_axis == int(np.argmin(extents)):
                msg = (
                    f"up_axis={params.up_axis} is shortest extent "
                    f"{extents[params.up_axis]:.3f} (flat TripoSR risk); "
                    "enable auto_up_axis or pick tallest axis"
                )
                orient_info["warnings"].append(msg)

        # Prefer side / 3/4 pose via yaw so head/ears/legs read in silhouette
        yaw = params.yaw_deg
        if yaw is None and params.auto_yaw:
            yaw = self._pick_best_yaw(work, params, ref_rgb=ref_rgb)
        # Record chosen yaw without mutating caller's dataclass unexpectedly
        chosen = dict(asdict(params))
        chosen["yaw_deg"] = float(yaw or 0.0)
        params = KidModeParams(**chosen)
        if params.yaw_deg:
            work = self._rotate_yaw(work, float(params.yaw_deg), params.up_axis)

        extents = work.extents
        horiz = [extents[i] for i in range(3) if i != params.up_axis]
        max_horiz = max(horiz) if horiz else float(np.max(extents))
        pitch = max_horiz / max(1, params.max_footprint)

        vg = work.voxelized(pitch=pitch).fill()
        raw = vg.matrix.astype(np.uint8)
        pancakes = self._reorient_y_up_matrix(raw, params.up_axis)

        pancakes = self._crop_tight_bbox(pancakes, pad=1)

        h = pancakes.shape[2]
        target_h = h
        if params.max_layers is not None and h > params.max_layers:
            target_h = params.max_layers
        if target_h < params.min_layers and h >= params.min_layers:
            target_h = params.min_layers
        if target_h != h:
            pancakes = self._squash_height(pancakes, target_h)

        # kid_mode: never use aggressive _hollow_out by default
        if params.hollow:
            pancakes = self._hollow_out(pancakes)
        elif params.thick_shell_wall > 0:
            pancakes = self._thick_shell(
                pancakes,
                wall=params.thick_shell_wall,
                solid_base_layers=max(1, params.base_widen_layers),
            )

        pancakes = self._widen_base(
            pancakes,
            n_layers=params.base_widen_layers,
            iterations=params.base_widen_iters,
        )
        # Re-crop after base dilate (may expand footprint)
        pancakes = self._crop_tight_bbox(pancakes, pad=params.crop_pad)
        pancakes = self._drop_dust(pancakes, params.min_component_beads)
        # Stash chosen yaw / orientation for callers
        self._last_kid_yaw = float(yaw or 0.0)
        self._last_kid_params = params
        orient_info["up_axis"] = int(params.up_axis)
        orient_info["yaw_deg"] = float(yaw or 0.0)
        orient_info["yaw_score"] = getattr(self, "_last_yaw_score", None)
        self._last_orient_info = orient_info
        return pancakes.astype(np.uint8)

    @staticmethod
    def crop_subject_rgb(img_rgb: np.ndarray, pad_frac: float = 0.08) -> np.ndarray:
        """Frame the subject by removing near-uniform background margins."""
        if img_rgb.size == 0:
            return img_rgb
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # Soft edge/sat mask: subject usually has more local variance
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        diff = cv2.absdiff(gray, blur)
        sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
        mask = ((diff > 6) | (sat > 28)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        ys, xs = np.where(mask > 0)
        if len(xs) < 50:
            return img_rgb
        h, w = gray.shape
        pad_y = int(pad_frac * h)
        pad_x = int(pad_frac * w)
        y0 = max(0, ys.min() - pad_y)
        y1 = min(h, ys.max() + pad_y + 1)
        x0 = max(0, xs.min() - pad_x)
        x1 = min(w, xs.max() + pad_x + 1)
        return img_rgb[y0:y1, x0:x1]

    def limit_colors(self, quantized_rgb: np.ndarray, max_colors: int) -> np.ndarray:
        """Keep the N most frequent palette colors; remap others to nearest kept."""
        if max_colors is None or max_colors <= 0:
            return quantized_rgb
        flat = quantized_rgb.reshape(-1, 3)
        uniq, counts = np.unique(flat, axis=0, return_counts=True)
        if len(uniq) <= max_colors:
            return quantized_rgb
        order = np.argsort(-counts)
        keep = uniq[order[:max_colors]].astype(np.int16)
        # Vectorized nearest-keep remap
        d = np.sum(
            (flat.astype(np.int16)[:, None, :] - keep[None, :, :]) ** 2, axis=2
        )
        out = keep[np.argmin(d, axis=1)]
        return out.reshape(quantized_rgb.shape).astype(np.uint8)

    @staticmethod
    def _dominant_subject_palette(quantized_rgb: np.ndarray, k: int = 4):
        """Pick top subject colors, preferring warm fur tones over green/blue backgrounds."""
        from color_quantizer import PERLER_PALETTE
        flat = quantized_rgb.reshape(-1, 3)
        uniq, counts = np.unique(flat, axis=0, return_counts=True)
        # Score: frequency, but penalize green/blue-heavy (common photo backgrounds)
        scored = []
        for rgb, c in zip(uniq, counts):
            r, g, b = [int(x) for x in rgb]
            bg_pen = 0.0
            if g > r + 25 and g > b + 15:  # leafy green
                bg_pen = 0.85
            elif b > r + 30 and b > g + 10:  # sky/water blue
                bg_pen = 0.7
            scored.append((c * (1.0 - bg_pen), tuple(int(x) for x in rgb)))
        scored.sort(reverse=True)
        keep = [rgb for _, rgb in scored[: max(1, k)]]
        # Ensure we have a dark accent if possible
        darks = [tuple(v) for v in PERLER_PALETTE.values() if sum(v) < 80]
        if darks and not any(sum(k_) < 80 for k_ in keep):
            keep.append(min(darks, key=lambda t: sum(t)))
        return keep[: max(k, 1)]

    def build_color_volume(
        self,
        voxel: np.ndarray,
        quantized_rgb: np.ndarray,
        params: Optional[KidModeParams] = None,
    ) -> np.ndarray:
        """
        Assign Perler RGB to each occupied voxel.
        - image_project: framed photo bands onto layers (default)
        - structural: map by height/region using subject palette (better for
          procedural meshes + busy photo backgrounds)
        """
        params = params or self.kid_params
        framed = self.crop_subject_rgb(quantized_rgb)
        if params.max_colors:
            framed = self.limit_colors(framed, params.max_colors)

        if params.color_mode == "structural":
            return self._color_volume_structural(voxel, framed, params)

        fx, fz, h = voxel.shape
        colors = np.zeros((fx, fz, h, 3), dtype=np.uint8)
        qh = framed.shape[0]
        for z in range(h):
            row_frac = 1.0 - (z + 0.5) / max(1, h)
            y0 = int(row_frac * (qh - 1))
            band = framed[max(0, y0 - qh // 12) : min(qh, y0 + qh // 12) + 1]
            if band.size == 0:
                band = framed
            band_small = cv2.resize(band, (fz, fx), interpolation=cv2.INTER_AREA)
            mask = voxel[:, :, z] > 0
            colors[:, :, z][mask] = band_small[mask]
        return colors

    def _color_volume_structural(
        self, voxel: np.ndarray, framed: np.ndarray, params: KidModeParams
    ) -> np.ndarray:
        """Region colors: feet/legs, body, belly stripe, head/ears, snout tip."""
        from color_quantizer import PERLER_PALETTE
        palette = self._dominant_subject_palette(framed, k=params.max_colors or 4)
        # Fallback named roles
        def pick(*names_or_rgb):
            for n in names_or_rgb:
                if isinstance(n, str) and n in PERLER_PALETTE:
                    return PERLER_PALETTE[n]
                if isinstance(n, tuple):
                    return n
            return palette[0]

        # Prefer warm tones from extracted palette for body
        body = palette[0]
        accent = palette[1] if len(palette) > 1 else pick("White")
        dark = next((c for c in palette if sum(c) < 100), pick("Black"))
        light = next((c for c in palette if sum(c) > 500), pick("White", accent))
        mid = next(
            (c for c in palette if c not in (body, dark, light)),
            pick("Orange", "Brown", body),
        )

        fx, fz, h = voxel.shape
        colors = np.zeros((fx, fz, h, 3), dtype=np.uint8)
        for x in range(fx):
            xf = x / max(1, fx - 1)  # 0=tail, 1=nose (after typical yaw)
            for y in range(fz):
                yf = y / max(1, fz - 1)
                for z in range(h):
                    if not voxel[x, y, z]:
                        continue
                    zf = z / max(1, h - 1)
                    # Legs / base
                    if zf < 0.22:
                        rgb = dark if (yf < 0.25 or yf > 0.75) else mid
                    # Belly stripe
                    elif zf < 0.45 and 0.3 < yf < 0.7:
                        rgb = light
                    # Ears (top + sides near head)
                    elif zf > 0.78 and xf > 0.55 and (yf < 0.35 or yf > 0.65):
                        rgb = mid
                    # Snout tip
                    elif xf > 0.88 and 0.35 < zf < 0.65:
                        rgb = dark if xf > 0.94 else light
                    # Head
                    elif xf > 0.65 and zf > 0.5:
                        rgb = mid
                    # Tail tip
                    elif xf < 0.12:
                        rgb = dark if zf > 0.55 else mid
                    else:
                        rgb = body
                    colors[x, y, z] = rgb
        return colors

    def spatial_quantize(
        self,
        img_rgb: np.ndarray,
        obj_path: str,
        kid_mode: bool = False,
        kid_params: Optional[KidModeParams] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Voxelizes a true 3D .obj mesh.

        Returns:
            padded_img (2D color grid for legacy path),
            color_volume (3D RGB for kid_mode, else None),
            voxel_matrix (binary occupancy)
        """
        if kid_mode:
            return self._spatial_quantize_kid(img_rgb, obj_path, kid_params)

        h, w = img_rgb.shape[:2]

        # 1. Spatial Downsampling of 2D Image (X, Y) for Color Mapping
        scale = self.max_dim_xy / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        quantized_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to exactly max_dim_xy x max_dim_xy
        padded_img = np.zeros((self.max_dim_xy, self.max_dim_xy, 3), dtype=np.uint8)
        y_start = (self.max_dim_xy - new_h) // 2
        x_start = (self.max_dim_xy - new_w) // 2
        padded_img[y_start : y_start + new_h, x_start : x_start + new_w] = quantized_img

        # 2. Voxelize the 3D Mesh
        mesh = trimesh.load(obj_path, force="mesh")
        max_extent = np.max(mesh.extents)
        if max_extent == 0:
            max_extent = 1.0
        pitch = max_extent / self.max_dim_xy

        voxel_grid = mesh.voxelized(pitch=pitch).fill()
        matrix = voxel_grid.matrix

        # 3. Squash Z-axis into max_dim_z layers
        chunk_size = matrix.shape[2] / self.max_dim_z
        z_squashed = np.zeros(
            (matrix.shape[0], matrix.shape[1], self.max_dim_z), dtype=np.uint8
        )

        for z in range(self.max_dim_z):
            start = int(z * chunk_size)
            end = (
                int((z + 1) * chunk_size)
                if z < self.max_dim_z - 1
                else matrix.shape[2]
            )
            if start < end:
                z_squashed[:, :, z] = np.max(matrix[:, :, start:end], axis=2)

        # 4. Create Final Padded Matrix
        voxel_matrix = np.zeros(
            (self.max_dim_xy, self.max_dim_xy, self.max_dim_z), dtype=np.uint8
        )
        x_len = min(z_squashed.shape[0], self.max_dim_xy)
        y_len = min(z_squashed.shape[1], self.max_dim_xy)

        x_start = (self.max_dim_xy - x_len) // 2
        y_start = (self.max_dim_xy - y_len) // 2

        voxel_matrix[x_start : x_start + x_len, y_start : y_start + y_len, :] = (
            z_squashed[:x_len, :y_len, :]
        )

        # 5. Hollow out (legacy relief path only)
        voxel_matrix = self._hollow_out(voxel_matrix)

        return padded_img, None, voxel_matrix

    def verify_kid_orientation(
        self,
        voxel: np.ndarray,
        colors: np.ndarray,
        ref_rgb: np.ndarray,
        silhouette_weight: float = 0.35,
        size: int = 256,
    ) -> dict:
        """Shape-only composite vs reference (SSIM / LPIPS-lite / sil IoU)."""
        from backward_verify import render_voxel_ortho, score_views

        renders = {
            "front": render_voxel_ortho(voxel, colors, "front", size),
            "side": render_voxel_ortho(voxel, colors, "side", size),
        }
        rows, summary = score_views(
            ref_rgb, renders, size=size, silhouette_weight=silhouette_weight
        )
        return {
            **summary,
            "views": [
                {
                    "view": r.view,
                    "ssim": r.ssim,
                    "lpips_lite": r.lpips_lite,
                    "silhouette_iou": r.silhouette_iou,
                    "composite_error": r.composite_error,
                }
                for r in rows
            ],
            "renders": renders,
        }

    def voxelize_kid_with_verify_gate(
        self,
        mesh: trimesh.Trimesh,
        img_rgb: np.ndarray,
        params: Optional[KidModeParams] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Voxelize + color, then verify against the photo.
        If composite_error > threshold, retry alternate up_axis (+ yaw) and keep best.
        Returns (voxel, color_volume, gate_report).
        """
        params = params or self.kid_params
        threshold = float(getattr(params, "verify_error_threshold", 0.50))
        do_gate = bool(getattr(params, "verify_gate", True))
        retry = bool(getattr(params, "verify_retry_alt_axes", True))

        # Primary attempt (auto standing-axis + photo-matched yaw inside voxelize_kid)
        voxel = self.voxelize_kid(mesh, params, ref_rgb=img_rgb)
        used = getattr(self, "_last_kid_params", params)
        colors = self.build_color_volume(voxel, img_rgb, used)
        metrics = self.verify_kid_orientation(voxel, colors, img_rgb)
        attempts = [{
            "up_axis": int(used.up_axis),
            "yaw_deg": float(getattr(self, "_last_kid_yaw", used.yaw_deg or 0.0)),
            "composite_error": metrics["composite_error"],
            "mean_ssim": metrics["mean_ssim"],
            "mean_lpips_lite": metrics["mean_lpips_lite"],
            "mean_silhouette_iou": metrics["mean_silhouette_iou"],
            "shape": list(voxel.shape),
            "label": "primary",
        }]
        best = {
            "voxel": voxel,
            "colors": colors,
            "params": used,
            "metrics": metrics,
            "orient": getattr(self, "_last_orient_info", {}),
        }

        if do_gate and retry and metrics["composite_error"] > threshold:
            extents = np.asarray(mesh.extents, dtype=np.float64)
            # Try other axes tallest-first (skip already used); one photo-matched yaw each
            alt_axes = [int(i) for i in np.argsort(-extents) if int(i) != int(used.up_axis)]
            for ax in alt_axes:
                trial = dict(asdict(params))
                trial["up_axis"] = ax
                trial["auto_up_axis"] = False
                trial["yaw_deg"] = None
                trial["auto_yaw"] = True
                trial["photo_matched_yaw"] = True
                p = KidModeParams(**trial)
                try:
                    v = self.voxelize_kid(mesh, p, ref_rgb=img_rgb)
                    c = self.build_color_volume(
                        v, img_rgb, getattr(self, "_last_kid_params", p)
                    )
                    m = self.verify_kid_orientation(v, c, img_rgb)
                except Exception as e:
                    attempts.append({
                        "up_axis": ax,
                        "yaw_deg": None,
                        "error": str(e),
                        "label": "retry_fail",
                    })
                    continue
                yaw_used = float(getattr(self, "_last_kid_yaw", 0.0))
                attempts.append({
                    "up_axis": ax,
                    "yaw_deg": yaw_used,
                    "composite_error": m["composite_error"],
                    "mean_ssim": m["mean_ssim"],
                    "mean_lpips_lite": m["mean_lpips_lite"],
                    "mean_silhouette_iou": m["mean_silhouette_iou"],
                    "shape": list(v.shape),
                    "label": "retry",
                })
                if m["composite_error"] < best["metrics"]["composite_error"]:
                    best = {
                        "voxel": v,
                        "colors": c,
                        "params": getattr(self, "_last_kid_params", p),
                        "metrics": m,
                        "orient": getattr(self, "_last_orient_info", {}),
                    }

        final_err = float(best["metrics"]["composite_error"])
        passed = final_err <= threshold if do_gate else True
        gate = {
            "enabled": do_gate,
            "threshold": threshold,
            "passed": passed,
            "composite_error": final_err,
            "mean_ssim": best["metrics"]["mean_ssim"],
            "mean_lpips_lite": best["metrics"]["mean_lpips_lite"],
            "mean_silhouette_iou": best["metrics"]["mean_silhouette_iou"],
            "views": best["metrics"].get("views"),
            "up_axis": int(best["params"].up_axis),
            "yaw_deg": float(best["params"].yaw_deg or 0.0),
            "shape": list(best["voxel"].shape),
            "attempts": attempts,
            "orient_info": best.get("orient") or {},
            "fail_reason": None
            if passed
            else (
                f"composite_error {final_err:.4f} > threshold {threshold:.2f} "
                f"after {len(attempts)} attempt(s); kit may not match photo"
            ),
        }
        # Stash for callers
        self._last_kid_params = best["params"]
        self._last_kid_yaw = float(best["params"].yaw_deg or 0.0)
        self._last_orient_info = best.get("orient") or {}
        self._last_verify_gate = gate
        # Drop renders from gate JSON-friendliness (kept on metrics only in memory)
        return best["voxel"], best["colors"], gate

    def _spatial_quantize_kid(
        self,
        img_rgb: np.ndarray,
        obj_path: str,
        kid_params: Optional[KidModeParams] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = kid_params or self.kid_params
        mesh = trimesh.load(obj_path, force="mesh")
        if getattr(params, "verify_gate", True):
            voxel, color_volume, gate = self.voxelize_kid_with_verify_gate(
                mesh, img_rgb, params
            )
            self._last_verify_gate = gate
        else:
            voxel = self.voxelize_kid(mesh, params, ref_rgb=img_rgb)
            color_volume = self.build_color_volume(voxel, img_rgb, getattr(self, "_last_kid_params", params))

        # 2D footprint preview (top-down composite) for API compatibility
        fx, fz, h = voxel.shape
        padded_img = np.zeros((fx, fz, 3), dtype=np.uint8)
        for z in range(h):
            mask = voxel[:, :, z] > 0
            padded_img[mask] = color_volume[:, :, z][mask]

        return padded_img, color_volume, voxel


if __name__ == "__main__":
    print("Testing Spatial Quantizer...")
    quantizer = VoxelQuantizer(max_dim_xy=60, max_dim_z=4)
    print("Successfully initialized Voxel Quantizer!")
    print(f"Default kid params: {asdict(DEFAULT_KID_PARAMS)}")
