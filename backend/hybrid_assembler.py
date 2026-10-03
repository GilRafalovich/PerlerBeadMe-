"""Hybrid assembly decomposition — binary planes (legacy) or colored kid steps."""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np

from color_quantizer import PERLER_PALETTE, ColorQuantizer


def estimate_difficulty(total_beads: int, layers: int, colors_used: int) -> dict:
    """Kid-mode heuristics: ≤800 beads report Easy/Medium (~45–60 min).

    Epic 3: kits at kid scale must not show Hard solely because layer count
    exceeds the old ≤10 Medium cap — bead budget is the primary signal.
    """
    if total_beads <= 400 and layers <= 10 and colors_used <= 6:
        level, minutes = "Easy", 45
    elif total_beads <= 800 and layers <= 20 and colors_used <= 8:
        # Medium band: bead budget primary (Epic 3.1 taller kits may use 17–18 layers)
        # stretch ≤600 → ~45 min; up to 800 → ~60 min
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


def nearest_palette_name(rgb: tuple, quantizer: ColorQuantizer) -> str:
    rgb_arr = np.array([[rgb]], dtype=np.uint8)
    q = quantizer.quantize(rgb_arr)[0, 0]
    for name, val in PERLER_PALETTE.items():
        if tuple(val) == tuple(int(c) for c in q):
            return name
    diffs = np.sum(
        (quantizer.palette_rgb.astype(int) - np.array(q, dtype=int)) ** 2, axis=1
    )
    return quantizer.color_names[int(np.argmin(diffs))]


class HybridAssembler:
    def __init__(
        self,
        voxel_matrix: np.ndarray,
        color_matrix: np.ndarray,
        color_volume: Optional[np.ndarray] = None,
        kid_mode: bool = False,
        bead_mm: float = 5.0,
        color_quantizer: Optional[ColorQuantizer] = None,
    ):
        """
        Takes a binary 3D voxel matrix and color data.
        Legacy: color_matrix is 2D (H,W,3) mapped onto XY.
        Kid mode: color_volume is (X,Z,H,3) per-voxel RGB; emits rich step sheets.
        """
        self.voxel_matrix = voxel_matrix
        self.color_matrix = color_matrix
        self.color_volume = color_volume
        self.kid_mode = kid_mode
        self.bead_mm = bead_mm
        self.quantizer = color_quantizer or ColorQuantizer()
        self.dim_x, self.dim_y, self.dim_z = voxel_matrix.shape

    def calculate_skeleton(self):
        """Finds the structural spine via mass distribution along axes."""
        mass_z = np.sum(self.voxel_matrix, axis=(0, 1))
        mass_y = np.sum(self.voxel_matrix, axis=(0, 2))
        mass_x = np.sum(self.voxel_matrix, axis=(1, 2))

        axes_mass = [np.sum(mass_x), np.sum(mass_y), np.sum(mass_z)]
        dominant_axis = np.argmax(axes_mass)
        return dominant_axis

    def _ensure_color_volume(self) -> np.ndarray:
        if self.color_volume is not None:
            return self.color_volume
        # Fallback: broadcast 2D color_matrix onto each layer
        fx, fy, h = self.voxel_matrix.shape
        vol = np.zeros((fx, fy, h, 3), dtype=np.uint8)
        cm = self.color_matrix
        if cm.ndim == 3 and cm.shape[0] == fx and cm.shape[1] == fy:
            for z in range(h):
                mask = self.voxel_matrix[:, :, z] > 0
                vol[:, :, z][mask] = cm[mask]
        return vol

    def _snap_colors(self, colors: np.ndarray) -> np.ndarray:
        flat = colors.reshape(-1, 3)
        snapped = self.quantizer.quantize(
            flat.reshape(-1, 1, 3).astype(np.uint8)
        ).reshape(-1, 3)
        return snapped.reshape(colors.shape)

    def layer_bead_counts(self, colors: np.ndarray):
        """Per-layer and global color tallies."""
        per_layer = []
        global_counts: Counter = Counter()
        fx, fz, h = self.voxel_matrix.shape
        for z in range(h):
            c: Counter = Counter()
            for x in range(fx):
                for y in range(fz):
                    if self.voxel_matrix[x, y, z]:
                        rgb = tuple(int(v) for v in colors[x, y, z])
                        name = nearest_palette_name(rgb, self.quantizer)
                        c[name] += 1
                        global_counts[name] += 1
            per_layer.append(
                {"layer_index": z, "beads": int(sum(c.values())), "by_color": dict(c)}
            )
        return per_layer, dict(global_counts)

    def decompose(self) -> list | dict[str, Any]:
        """
        Adaptive plane generation.
        Legacy: list of binary horizontal_plane dicts.
        Kid mode: dict with colored layouts, shopping list, difficulty, step order.
        """
        if self.kid_mode:
            return self.decompose_kid()

        residual = np.copy(self.voxel_matrix)
        instructions = []

        print(
            f"Generating Kid-Friendly Assembly: Slicing into {self.dim_z} horizontal layers..."
        )
        for z in range(self.dim_z):
            layer = residual[:, :, z]
            if np.sum(layer) > 0:
                instructions.append(
                    {
                        "type": "horizontal_plane",
                        "z_index": z,
                        "layout": layer.tolist(),
                        "dowel_holes": [],
                    }
                )

        return instructions

    def decompose_kid(self) -> dict[str, Any]:
        """Colored layouts, per-color counts, step order, difficulty/time for kid builds."""
        colors = self._snap_colors(self._ensure_color_volume())
        self.color_volume = colors

        occupied = int(self.voxel_matrix.sum())
        beads_per_layer = [
            int(self.voxel_matrix[:, :, z].sum()) for z in range(self.dim_z)
        ]
        per_layer, global_counts = self.layer_bead_counts(colors)
        diff = estimate_difficulty(occupied, self.dim_z, len(global_counts))

        steps = []
        for z in range(self.dim_z):
            mask = self.voxel_matrix[:, :, z] > 0
            if not np.any(mask):
                continue
            color_layout = colors[:, :, z]
            # Named color grid (None for empty cells) for renderers / clients
            name_grid = [[None] * self.dim_y for _ in range(self.dim_x)]
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if mask[x, y]:
                        rgb = tuple(int(v) for v in color_layout[x, y])
                        name_grid[x][y] = nearest_palette_name(rgb, self.quantizer)

            steps.append(
                {
                    "step": z + 1,
                    "layer_index": z,
                    "is_base": z == 0,
                    "type": "horizontal_plane",
                    "layout": self.voxel_matrix[:, :, z].astype(int).tolist(),
                    "color_layout_rgb": color_layout.tolist(),
                    "color_layout_names": name_grid,
                    "beads": per_layer[z]["beads"],
                    "by_color": per_layer[z]["by_color"],
                    "dowel_holes": [],
                    "tip": (
                        "This is the BASE — make it wide and solid so your figure stands up!"
                        if z == 0
                        else "Iron this layer flat (adult help!), cool, then stack the next layer on top and fuse."
                    ),
                }
            )

        approx = {
            "width_mm": self.dim_x * self.bead_mm,
            "depth_mm": self.dim_y * self.bead_mm,
            "height_mm": self.dim_z * self.bead_mm,
        }

        return {
            "mode": "kid_mode",
            "assembly_order": "bottom_to_top",
            "base_layer": 0,
            "voxel_shape": list(self.voxel_matrix.shape),
            "num_layers": int(self.dim_z),
            "total_beads": occupied,
            "beads_per_layer": beads_per_layer,
            "shopping_list": global_counts,
            "num_colors": len(global_counts),
            "difficulty": diff,
            "approx_size_mm": approx,
            "steps": steps,
            "layers": per_layer,
            "tips": [
                "Start with Layer 1 (base). Keep it wide so the figure stands.",
                "Adult irons each layer. Cool fully before stacking the next.",
                "Match bead colors to the color key on each sheet.",
            ],
            "color_volume": colors,
        }

    def _generate_dowel_holes(self, layer: np.ndarray):
        return []


if __name__ == "__main__":
    dummy_matrix = np.zeros((20, 20, 20), dtype=np.uint8)
    dummy_matrix[5:15, 5:15, 0:10] = 1
    dummy_color = np.zeros((20, 20, 20, 3), dtype=np.uint8)

    assembler = HybridAssembler(dummy_matrix, dummy_color)
    instructions = assembler.decompose()
    print(f"Generated {len(instructions)} assembly planes based on Figure of Merit.")
