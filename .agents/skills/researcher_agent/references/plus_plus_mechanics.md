# Plus-Plus Interlocking Mechanics (Friction Fit)

## Core Concept
The "Plus-Plus" building toy utilizes an interlocking mechanism based entirely on a "Friction Fit". Unlike vertical stacking (e.g., LEGO), 3D volume is achieved by constructing flat 2D shapes (mosaics) and connecting them perpendicularly at 90-degree angles by sliding their edges into predefined slots.

## Application to PerlerBeadMe
Because Perler Beads melt into flat 2D planes, we can replicate this exact mechanical physics.
Instead of stacking 4 solid matrices on top of each other, the pipeline should compute:
1. **Base Plane**: A flat 2D layer representing the silhouette.
2. **Support Planes**: Additional 2D layers designed to intersect the Base Plane perpendicularly.
3. **Slotted Interlocking**: Each intersecting point must have a perfectly aligned "gap" or "slot" left completely empty of beads. 
    - **Slot Sizing**: The slot must be exactly 1-bead wide, and structurally deep enough to friction-hold the intersecting plane (usually 1 or 2 beads deep).

This completely eliminates the need for glue, generating a vastly superior kid-friendly assembly experience.
