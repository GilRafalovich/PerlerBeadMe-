"""
Lego-style stud/socket interlocking for kid-mode Perler pancake layers.

Modifies consecutive horizontal layers so the lower layer carries STUD marks
(connector peg seats) and the upper layer has SOCKET holes that those pegs
drop into — mechanical registration without molded brick studs.

See backend/docs/lego_interlock_design.md for options and kid rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# Role codes stored in role_volume (uint8)
EMPTY = 0
BEAD = 1
STUD = 2
SOCKET = 3

ROLE_NAMES = {EMPTY: "empty", BEAD: "bead", STUD: "stud", SOCKET: "socket"}


@dataclass
class InterlockParams:
    """Knobs for stud/socket placement."""

    lattice_stride: int = 3
    """Place candidates every N cells along X and Z."""

    edge_inset: int = 1
    """Require this many occupied neighbors toward the rim (erode overlap)."""

    min_per_interface: int = 2
    max_per_interface: int = 8
    """Clamp connector count by overlap size."""

    min_overlap_cells: int = 8
    """Skip interface if overlap after erosion is too small."""

    checkerboard: bool = True
    """Offset lattice by layer index for staggered studs."""

    peg_beads: int = 2
    """Beads per connector peg (shopping list)."""

    clear_socket: bool = True
    """Remove the bead from the upper layer at socket sites."""


@dataclass
class InterfacePlan:
    z_bottom: int
    z_top: int
    sites: list[tuple[int, int]]  # (x, z_depth) in matrix coords
    overlap_cells: int


@dataclass
class InterlockResult:
    voxel: np.ndarray
    """Modified occupancy (X, Z, H), sockets cleared if requested."""

    colors: Optional[np.ndarray]
    """Modified color volume, sockets zeroed."""

    roles: np.ndarray
    """uint8 role volume (X, Z, H) — EMPTY/BEAD/STUD/SOCKET."""

    interfaces: list[InterfacePlan] = field(default_factory=list)
    params: InterlockParams = field(default_factory=InterlockParams)
    original_voxel: Optional[np.ndarray] = None
    original_colors: Optional[np.ndarray] = None

    @property
    def peg_count(self) -> int:
        return sum(len(i.sites) for i in self.interfaces)

    def summary(self) -> dict[str, Any]:
        return {
            "approach": "stud_on_bottom_socket_on_top",
            "params": {
                "lattice_stride": self.params.lattice_stride,
                "edge_inset": self.params.edge_inset,
                "min_per_interface": self.params.min_per_interface,
                "max_per_interface": self.params.max_per_interface,
                "peg_beads": self.params.peg_beads,
            },
            "num_interfaces": len(self.interfaces),
            "peg_count": self.peg_count,
            "connector_beads_needed": self.peg_count * self.params.peg_beads,
            "interfaces": [
                {
                    "z_bottom": i.z_bottom,
                    "z_top": i.z_top,
                    "sites": [{"x": x, "z": z} for x, z in i.sites],
                    "n_sites": len(i.sites),
                    "overlap_cells": i.overlap_cells,
                }
                for i in self.interfaces
            ],
            "role_counts": {
                ROLE_NAMES[k]: int((self.roles == k).sum())
                for k in (EMPTY, BEAD, STUD, SOCKET)
            },
            "beads_before": int(self.original_voxel.sum()) if self.original_voxel is not None else None,
            "beads_after": int(self.voxel.sum()),
        }


def _erode_binary(mask: np.ndarray, inset: int) -> np.ndarray:
    """Morphological erosion with a (2*inset+1) square for edge safety."""
    if inset <= 0:
        return mask.astype(bool)
    m = mask.astype(bool)
    h, w = m.shape
    out = np.zeros_like(m)
    for i in range(inset, h - inset):
        for j in range(inset, w - inset):
            if not m[i, j]:
                continue
            if m[i - inset : i + inset + 1, j - inset : j + inset + 1].all():
                out[i, j] = True
    return out


def _lattice_candidates(
    mask: np.ndarray,
    stride: int,
    layer_parity: int,
    checkerboard: bool,
) -> list[tuple[int, int]]:
    fx, fz = mask.shape
    ox = (layer_parity * (stride // 2)) % stride if checkerboard else 0
    oz = (layer_parity * (stride // 2)) % stride if checkerboard else 0
    sites: list[tuple[int, int]] = []
    for x in range(ox, fx, stride):
        for z in range(oz, fz, stride):
            if mask[x, z]:
                # Prefer true checkerboard of lattice points
                if checkerboard and ((x // stride) + (z // stride) + layer_parity) % 2:
                    continue
                sites.append((x, z))
    return sites


def _score_site(mask: np.ndarray, x: int, z: int) -> float:
    """Prefer central, well-supported cells."""
    fx, fz = mask.shape
    # Distance from bbox center of mask
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0
    cy, cx = float(ys.mean()), float(xs.mean())
    dist = abs(x - cy) + abs(z - cx)
    # Neighbor support (8-connected)
    support = 0
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            nx, nz = x + dx, z + dz
            if 0 <= nx < fx and 0 <= nz < fz and mask[nx, nz]:
                support += 1
    return support * 10.0 - dist


def _pick_sites(
    candidates: list[tuple[int, int]],
    mask: np.ndarray,
    min_n: int,
    max_n: int,
) -> list[tuple[int, int]]:
    if not candidates:
        return []
    scored = sorted(
        candidates, key=lambda p: _score_site(mask, p[0], p[1]), reverse=True
    )
    # Greedy with minimum separation of 2 cells
    chosen: list[tuple[int, int]] = []
    for x, z in scored:
        if len(chosen) >= max_n:
            break
        if any(abs(x - cx) + abs(z - cz) < 2 for cx, cz in chosen):
            continue
        chosen.append((x, z))
    # If too few, relax separation
    if len(chosen) < min_n:
        chosen = []
        for x, z in scored:
            if len(chosen) >= max_n:
                break
            if any(abs(x - cx) + abs(z - cz) < 1 for cx, cz in chosen):
                continue
            chosen.append((x, z))
    # Cap by overlap scale: ~1 site per 20 overlap cells
    return chosen[:max_n]


def plan_interfaces(
    voxel: np.ndarray, params: InterlockParams
) -> list[InterfacePlan]:
    """Compute stud/socket sites between consecutive layers."""
    fx, fz, h = voxel.shape
    plans: list[InterfacePlan] = []
    for z in range(h - 1):
        bottom = voxel[:, :, z] > 0
        top = voxel[:, :, z + 1] > 0
        overlap = bottom & top
        safe = _erode_binary(overlap, params.edge_inset)
        overlap_n = int(overlap.sum())
        if int(safe.sum()) < params.min_overlap_cells and overlap_n < params.min_overlap_cells * 2:
            # Tiny interface — skip or place at most 1 if any safe cell
            if int(safe.sum()) == 0:
                continue
        # Scale max by overlap
        scaled_max = max(
            params.min_per_interface,
            min(params.max_per_interface, max(1, overlap_n // 18)),
        )
        cands = _lattice_candidates(
            safe if safe.any() else overlap,
            params.lattice_stride,
            layer_parity=z,
            checkerboard=params.checkerboard,
        )
        # Fallback: any overlap cell if lattice empty
        if not cands:
            ys, xs = np.where(safe if safe.any() else overlap)
            cands = list(zip(ys.tolist(), xs.tolist()))
        sites = _pick_sites(
            cands,
            safe if safe.any() else overlap,
            min_n=min(params.min_per_interface, scaled_max),
            max_n=scaled_max,
        )
        # If lattice was too sparse, fall back to best-scoring overlap cells
        if len(sites) < min(params.min_per_interface, scaled_max):
            ys, xs = np.where(safe if safe.any() else overlap)
            fallback = list(zip(ys.tolist(), xs.tolist()))
            sites = _pick_sites(
                fallback,
                safe if safe.any() else overlap,
                min_n=min(params.min_per_interface, scaled_max),
                max_n=scaled_max,
            )
        if not sites:
            continue
        plans.append(
            InterfacePlan(
                z_bottom=z,
                z_top=z + 1,
                sites=sites,
                overlap_cells=overlap_n,
            )
        )
    return plans


def apply_interlock(
    voxel: np.ndarray,
    colors: Optional[np.ndarray] = None,
    params: Optional[InterlockParams] = None,
) -> InterlockResult:
    """
    Take kid layer grids and emit interlocking-aware layers.

    - STUD cells on layer z keep their bead (peg sits on top after ironing).
    - SOCKET cells on layer z+1 are cleared (hole for the peg).
    """
    params = params or InterlockParams()
    voxel = np.asarray(voxel, dtype=np.uint8).copy()
    original = voxel.copy()
    original_colors = None if colors is None else np.asarray(colors).copy()
    if colors is not None:
        colors = np.asarray(colors).copy()
        if colors.shape[:3] != voxel.shape:
            raise ValueError(
                f"color volume shape {colors.shape} != voxel {voxel.shape}"
            )

    roles = np.zeros(voxel.shape, dtype=np.uint8)
    roles[voxel > 0] = BEAD

    interfaces = plan_interfaces(voxel, params)
    for iface in interfaces:
        for x, z in iface.sites:
            # Bottom: stud
            if voxel[x, z, iface.z_bottom] > 0:
                roles[x, z, iface.z_bottom] = STUD
            # Top: socket — clear bead
            if params.clear_socket:
                voxel[x, z, iface.z_top] = 0
                if colors is not None:
                    colors[x, z, iface.z_top] = 0
            roles[x, z, iface.z_top] = SOCKET

    return InterlockResult(
        voxel=voxel,
        colors=colors,
        roles=roles,
        interfaces=interfaces,
        params=params,
        original_voxel=original,
        original_colors=original_colors,
    )


def layer_role_grid(roles: np.ndarray, layer: int) -> np.ndarray:
    return roles[:, :, layer]


def role_maps_as_lists(result: InterlockResult) -> list[dict[str, Any]]:
    """JSON-friendly per-layer maps for instructions."""
    fx, fz, h = result.roles.shape
    out = []
    for zi in range(h):
        studs, sockets, beads = [], [], []
        for x in range(fx):
            for z in range(fz):
                r = int(result.roles[x, z, zi])
                cell = {"x": x, "z": z}
                if r == STUD:
                    studs.append(cell)
                elif r == SOCKET:
                    sockets.append(cell)
                elif r == BEAD:
                    beads.append(cell)
        out.append(
            {
                "layer_index": zi,
                "studs": studs,
                "sockets": sockets,
                "n_beads": int((result.voxel[:, :, zi] > 0).sum()),
                "n_studs": len(studs),
                "n_sockets": len(sockets),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Rendering helpers (PIL) — keep matplotlib optional for 3D seating diagram
# ---------------------------------------------------------------------------

def _fonts():
    from PIL import ImageFont

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
        font_lg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
        return font, font_sm, font_lg
    except Exception:
        d = ImageFont.load_default()
        return d, d, d


def render_layer_sheet(
    layer_idx: int,
    total_layers: int,
    voxel_layer: np.ndarray,
    color_layer: np.ndarray,
    role_layer: np.ndarray,
    out_path: str,
    cell: int = 20,
    title_suffix: str = "",
    show_roles: bool = True,
):
    """Instruction-style sheet; studs = raised circle, sockets = dashed hole."""
    from PIL import Image, ImageDraw

    fx, fz = voxel_layer.shape
    margin = 36
    legend_w = 260
    title_h = 88
    footer_h = 64
    grid_w = fz * cell
    grid_h = fx * cell
    W = max(760, margin * 2 + grid_w + legend_w)
    H = title_h + margin + grid_h + footer_h

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font, font_sm, font_lg = _fonts()

    step = layer_idx + 1
    line1 = f"STEP {step} of {total_layers}  —  Layer {step}{title_suffix}"
    if layer_idx == 0:
        line1 += "  ★ BASE"
    line2 = (
        "Studs ▲ = place peg after ironing   ·   Sockets ○ = leave empty (peg seats here)"
        if show_roles
        else "Flat stack (no interlocking marks)"
    )
    draw.text((margin, 12), line1, fill=(20, 20, 20), font=font_lg)
    draw.text((margin, 46), line2, fill=(60, 60, 60), font=font_sm)

    ox, oy = margin, title_h
    n_stud = n_sock = 0
    for i in range(fx):
        for j in range(fz):
            x0 = ox + j * cell
            y0 = oy + i * cell
            role = int(role_layer[i, j]) if show_roles else (
                BEAD if voxel_layer[i, j] else EMPTY
            )
            if role == SOCKET:
                n_sock += 1
                draw.rectangle(
                    [x0, y0, x0 + cell - 1, y0 + cell - 1],
                    fill=(255, 245, 230),
                    outline=(200, 120, 40),
                )
                # dashed hole
                pad = 4
                draw.ellipse(
                    [x0 + pad, y0 + pad, x0 + cell - pad, y0 + cell - pad],
                    outline=(200, 80, 20),
                    width=2,
                )
                draw.line(
                    [x0 + pad, y0 + cell // 2, x0 + cell - pad, y0 + cell // 2],
                    fill=(200, 80, 20),
                    width=1,
                )
            elif voxel_layer[i, j] or role == STUD:
                rgb = tuple(int(v) for v in color_layer[i, j])
                if rgb == (0, 0, 0) and role != STUD and not voxel_layer[i, j]:
                    rgb = (200, 200, 200)
                draw.rectangle(
                    [x0, y0, x0 + cell - 1, y0 + cell - 1],
                    fill=rgb if sum(rgb) > 0 else (235, 203, 67),
                    outline=(60, 60, 60),
                )
                if role == STUD:
                    n_stud += 1
                    # raised stud glyph
                    pad = 5
                    draw.ellipse(
                        [x0 + pad, y0 + pad, x0 + cell - pad, y0 + cell - pad],
                        fill=(40, 120, 220),
                        outline=(10, 40, 100),
                    )
                    draw.ellipse(
                        [
                            x0 + pad + 3,
                            y0 + pad + 3,
                            x0 + cell - pad - 3,
                            y0 + cell - pad - 3,
                        ],
                        fill=(120, 180, 255),
                        outline=(10, 40, 100),
                    )
            else:
                draw.rectangle(
                    [x0, y0, x0 + cell - 1, y0 + cell - 1],
                    fill=(245, 245, 245),
                    outline=(220, 220, 220),
                )

    lx = ox + grid_w + 20
    ly = oy
    draw.text((lx, ly), "Legend", fill=(0, 0, 0), font=font)
    ly += 26
    beads_here = int((voxel_layer > 0).sum())
    draw.text((lx, ly), f"Beads: {beads_here}", fill=(40, 40, 40), font=font_sm)
    ly += 22
    if show_roles:
        draw.ellipse([lx, ly, lx + 18, ly + 18], fill=(120, 180, 255), outline=(10, 40, 100))
        draw.text((lx + 26, ly + 1), f"Stud (peg here): {n_stud}", fill=(0, 0, 0), font=font_sm)
        ly += 24
        draw.ellipse([lx, ly, lx + 18, ly + 18], outline=(200, 80, 20), width=2)
        draw.text((lx + 26, ly + 1), f"Socket (hole): {n_sock}", fill=(0, 0, 0), font=font_sm)
        ly += 28

    tip = (
        "Iron flat (adult!). Cool. Press 2-bead pegs onto blue studs, then seat next layer’s holes."
        if show_roles
        else "Iron flat, cool, stack next layer and fuse/glue — no mechanical pegs."
    )
    draw.text((margin, H - 40), tip, fill=(80, 80, 80), font=font_sm)
    img.save(out_path)
    return out_path


def render_seating_diagram(
    result: InterlockResult,
    out_path: str,
    max_interfaces: int = 4,
):
    """
    Side-view diagram: how pegs on layer z seat into sockets on layer z+1.
    """
    from PIL import Image, ImageDraw

    interfaces = result.interfaces[:max_interfaces]
    if not interfaces:
        # empty placeholder
        img = Image.new("RGB", (640, 200), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font, _, font_lg = _fonts()
        draw.text((40, 80), "No interfaces (overlap too small)", fill=(80, 80, 80), font=font_lg)
        img.save(out_path)
        return out_path

    panel_w, panel_h = 280, 220
    cols = min(2, len(interfaces))
    rows = (len(interfaces) + cols - 1) // cols
    margin = 24
    header = 70
    W = margin * 2 + cols * panel_w + (cols - 1) * 16
    H = header + margin + rows * panel_h + (rows - 1) * 16 + 40

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font, font_sm, font_lg = _fonts()

    draw.text(
        (margin, 14),
        "How layers seat (Lego-style peg → socket)",
        fill=(20, 20, 20),
        font=font_lg,
    )
    draw.text(
        (margin, 42),
        "Blue stud on lower pancake · orange hole on upper · peg columns lock sideways slide",
        fill=(70, 70, 70),
        font=font_sm,
    )

    for idx, iface in enumerate(interfaces):
        r, c = divmod(idx, cols)
        px = margin + c * (panel_w + 16)
        py = header + margin + r * (panel_h + 16)
        draw.rectangle(
            [px, py, px + panel_w - 1, py + panel_h - 1],
            outline=(200, 200, 200),
            fill=(250, 250, 252),
        )
        draw.text(
            (px + 10, py + 8),
            f"Layer {iface.z_bottom + 1} → {iface.z_top + 1}  ({len(iface.sites)} pegs)",
            fill=(30, 30, 30),
            font=font,
        )

        # Draw two horizontal slabs with pegs
        slab_h = 36
        lower_y = py + panel_h - 55
        upper_y = lower_y - 70
        slab_x0 = px + 30
        slab_x1 = px + panel_w - 30
        # lower slab
        draw.rectangle(
            [slab_x0, lower_y, slab_x1, lower_y + slab_h],
            fill=(235, 203, 67),
            outline=(80, 80, 80),
        )
        draw.text((slab_x0, lower_y + slab_h + 4), f"L{iface.z_bottom + 1} (studs)", fill=(40, 40, 40), font=font_sm)
        # upper slab
        draw.rectangle(
            [slab_x0, upper_y, slab_x1, upper_y + slab_h],
            fill=(250, 250, 250),
            outline=(80, 80, 80),
        )
        draw.text((slab_x0, upper_y - 18), f"L{iface.z_top + 1} (sockets)", fill=(40, 40, 40), font=font_sm)

        n = max(1, len(iface.sites))
        for si in range(min(n, 5)):
            t = (si + 1) / (min(n, 5) + 1)
            cx = int(slab_x0 + t * (slab_x1 - slab_x0))
            # socket hole in upper
            draw.ellipse([cx - 8, upper_y + 8, cx + 8, upper_y + slab_h - 8], outline=(200, 80, 20), width=2)
            # peg column
            peg_top = upper_y + 10
            peg_bot = lower_y + 8
            draw.rectangle([cx - 5, peg_top, cx + 5, peg_bot], fill=(40, 120, 220), outline=(10, 40, 100))
            # stud disc on lower
            draw.ellipse([cx - 9, lower_y + 6, cx + 9, lower_y + 24], fill=(120, 180, 255), outline=(10, 40, 100))

        # arrow
        mid_x = (slab_x0 + slab_x1) // 2
        draw.polygon(
            [
                (mid_x, upper_y + slab_h + 6),
                (mid_x - 8, upper_y + slab_h + 20),
                (mid_x + 8, upper_y + slab_h + 20),
            ],
            fill=(100, 100, 100),
        )

    draw.text(
        (margin, H - 28),
        f"Total connector pegs for kit: {result.peg_count}  "
        f"(×{result.params.peg_beads} beads each ≈ {result.peg_count * result.params.peg_beads} beads)",
        fill=(50, 50, 50),
        font=font_sm,
    )
    img.save(out_path)
    return out_path


def render_before_after_overview(
    before_voxel: np.ndarray,
    after_voxel: np.ndarray,
    after_roles: np.ndarray,
    colors: np.ndarray,
    out_path: str,
):
    """Top-down collage of all layers before vs after (roles tinted)."""
    from PIL import Image, ImageDraw

    fx, fz, h = before_voxel.shape
    cell = 8
    gap = 10
    label_h = 22
    col_w = fz * cell + 8
    row_h = fx * cell + label_h + 4
    W = 40 + 2 * (col_w * h + gap * (h - 1)) + 80
    # Actually stack: before row, after row
    W = 24 + h * (col_w + gap)
    H = 60 + 2 * row_h + 40

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font, font_sm, font_lg = _fonts()
    draw.text((16, 12), "Pikachu layers — BEFORE (flat) vs AFTER (stud/socket)", fill=(20, 20, 20), font=font_lg)

    def paint_row(y0: int, voxel, roles, title: str, use_roles: bool):
        draw.text((16, y0), title, fill=(40, 40, 40), font=font)
        for zi in range(h):
            ox = 16 + zi * (col_w + gap)
            oy = y0 + 20
            draw.text((ox, oy), f"L{zi+1}", fill=(80, 80, 80), font=font_sm)
            gy = oy + label_h - 4
            for i in range(fx):
                for j in range(fz):
                    x0 = ox + j * cell
                    y0c = gy + i * cell
                    role = int(roles[i, j, zi]) if use_roles else (
                        BEAD if voxel[i, j, zi] else EMPTY
                    )
                    if role == SOCKET:
                        draw.rectangle([x0, y0c, x0 + cell - 1, y0c + cell - 1], fill=(255, 200, 150), outline=(180, 100, 40))
                    elif voxel[i, j, zi] or role == STUD:
                        rgb = tuple(int(v) for v in colors[i, j, zi])
                        if sum(rgb) == 0:
                            rgb = (235, 203, 67)
                        draw.rectangle([x0, y0c, x0 + cell - 1, y0c + cell - 1], fill=rgb, outline=(90, 90, 90))
                        if role == STUD:
                            draw.ellipse([x0 + 1, y0c + 1, x0 + cell - 2, y0c + cell - 2], fill=(80, 140, 230))
                    else:
                        draw.rectangle([x0, y0c, x0 + cell - 1, y0c + cell - 1], fill=(240, 240, 240), outline=(230, 230, 230))

    paint_row(50, before_voxel, after_roles, "BEFORE — solid pancakes (glue/fuse only)", False)
    paint_row(50 + row_h + 20, after_voxel, after_roles, "AFTER — blue studs / orange sockets", True)
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    # Quick self-test with a synthetic stack
    v = np.zeros((8, 8, 3), dtype=np.uint8)
    v[1:7, 1:7, :] = 1
    r = apply_interlock(v)
    print(r.summary())
