# Lego-style layer interlocking for Perler pancakes

**Audience:** kid builds (fuse-plate layers stacked bottom → top)  
**Demo:** `backend/demo_pikachu/`  
**Module:** `backend/lego_interlock.py`

## Problem

Kid-mode figures are horizontal “pancake” sheets fused flat, then stacked.
Today those sheets are **glued / re-fused** as flat stacks. Gil’s ask: layers
should **seat into each other like Lego** — mechanical registration, not just
surface glue.

## Geometry reality (feasibility limits)

A standard Perler / Hama / Artkal bead is a **flat disc with a center hole**
(~5 mm). Ironing on a fuse plate melts neighbors into a **2D sheet**. That
means:

| Idea | Can Perler do it? | Kid-friendly? |
|------|-------------------|---------------|
| Molded Lego studs on a fused sheet | **No** — ironing flattens | — |
| Vertical peg columns through holes | **Yes** — leave empty cells, insert short pegs | ✅ Best |
| Stud bead left proud on lower layer | Partial — 2nd bead fused on top after flat iron | ⚠️ fiddly |
| In-plane dovetail / keyed tabs | Yes as silhouette tabs, but **don’t lock stacked pancakes** without vertical engagement | ❌ wrong geometry |
| Half-bead XY offset nesting | Weak — fused flats don’t nest like brick bottoms | ❌ |
| Plus-Plus perpendicular slots | Different build system (edge slots, 90°) | Separate track |
| Separate connector strips / dowels | Yes — mini fused 2-bead posts | ✅ |

**What “Lego” can mean in Perler:** not molded studs, but **registration pegs
that seat into sockets** so layers can’t slide, plus optional light fuse/glue
only as insurance.

## Options considered

### A — Stud-on-bottom / socket-on-top (recommended)

1. Find overlap of consecutive occupied layers.
2. Place connector sites on an interior lattice (inset 1 bead from the rim).
3. **Bottom layer:** keep the bead; mark cell as **STUD** (after ironing, kid
   places a 1–2 bead tall connector peg upright on that cell).
4. **Top layer:** clear that cell → **SOCKET** hole; peg from below seats in.
5. Instructions show studs as raised circles and sockets as dashed holes.

*Pros:* looks like Lego to kids; works with ordinary fuse plates; each layer
still ironed flat.  
*Cons:* pegs are a second mini-build; proud studs need care when stacking.

### B — Through-holes + shared dowel pegs

Both layers leave the same cell empty; a separate 2-bead fused peg (or a
plastic mini-dowel) goes through both sheets like a rivet.

*Pros:* strongest shear lock; symmetric.  
*Cons:* holes weaken small footprints; more empty cells in the art.

### C — Silhouette keyed tabs / dovetails

Grow tabs on layer *N* and matching notches on *N+1* in the XY silhouette.

*Pros:* no extra parts.  
*Cons:* stacked flat sheets **do not mechanically engage** tabs; tabs only
help if planes meet edge-on (Plus-Plus). Rejected for pancake stacks.

### D — Half-bead offsets

Shift odd layers by ½ cell so beads sit in valleys.

*Pros:* Lego-brick intuition.  
*Cons:* pegboard grid is integer; fused discs don’t nest; ruins pattern
alignment. Rejected.

## Recommended approach for kids (A)

**Stud / socket lattice with short connector pegs.**

- Density: every ~3 cells on a checkerboard, clamped to 2–8 sites per
  interface (scaled to overlap area).
- Always inset ≥1 bead from the layer rim so sockets don’t open the outline.
- Prefer sites that stay inside both silhouettes after socket clearing.
- Connector inventory: one **2-bead yellow (or clear) peg** per site —
  kids fuse a tiny strip of 2 beads, or use loose beads on a spare pegboard
  pin as a temporary post.
- Assembly: iron layers flat → cool → press pegs onto stud marks → drop next
  layer so sockets seat → optional light fuse of the stack.

This stays buildable on a standard square pegboard, needs no special bricks,
and reads clearly on instruction sheets.

## Prototype I/O

**Input:** kid voxel grid `(X,Z,H)` + optional color volume `(X,Z,H,3)`.  
**Output:**

- Modified voxel / color volumes (sockets cleared).
- Per-layer role maps: `bead | stud | socket | empty`.
- Interface list: `(z_bottom, z_top, [(x,z), ...])`.
- Peg count + shopping add-on.
- Instruction PNGs with stud/socket glyphs + seating diagram.

See `demo_pikachu/lego_interlock/` for before/after Pikachu sheets.
