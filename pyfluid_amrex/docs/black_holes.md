# Black-Hole Initial Data

PyFluid's black-hole boundary is the existing physical-ID/materialization
interface. A provider constructs ADM/Z4c data on the grid appropriate to the
ID method; the saved artifact can then be materialized onto a separate 3D
Cartesian evolution hierarchy. Evolution does not own a second initial-data
formula.

## Current Kerr--Schild foundation

`spacetime.initial_data = kerr_schild` selects exact ingoing Cartesian
Kerr--Schild exterior data. `spacetime.mass`, `spacetime.origin1/2/3`, and the
dimensionless Cartesian axial vector
`spacetime.dimensionless_spin1/2/3` define the hole. The spin magnitude must
be below one. The provider computes the full spatial metric, lapse, shift, and
extrinsic curvature and then uses the shared ADM-to-Z4c conversion.

For a fixed-background run, set both `spacetime.initial_data` and
`spacetime.fixed_metric` to `kerr_schild`. For evolved Z4c, leave
`spacetime.fixed_metric = minkowski`; the stored initial data, not that fixed
metric selector, seed evolution.

`spacetime.black_hole_interior` owns the horizon-interior contract:

- `exterior` requires every evolution cell to remain outside the outer
  horizon. It is exact Kerr--Schild data and is useful for fixed-background
  tests and exterior-domain development.
- `filled` requires `spacetime.black_hole_fill_radius` to lie strictly inside
  the outer horizon. It keeps exact Kerr--Schild fields outside the inner
  transition sphere, joins them C2-smoothly to regular Minkowski data without
  sampling at or inside the Kerr ring, and supports a complete Cartesian grid.
  The fill is deliberately constraint violating inside the horizon; it is not
  called excision. Consequently, unmasked whole-domain constraint norms are
  dominated by the arbitrary interior and are finite-state safety diagnostics,
  not an exterior-vacuum accuracy measurement. A production full-grid method
  still needs a constraint-improved interior and horizon/exterior diagnostics.
- `excision` is a reserved contract and currently fails configuration. It will
  require an active-cell mask, a pure-outflow inner evolution boundary, and
  the Cook--Pfeiffer XCTS apparent-horizon boundary before production use.

The existing `slow_start` lapse choice remains the puncture-oriented gauge:
it is 1+log plus a Gaussian-in-time relaxation of the lapse toward
`sqrt(chi)`. `gamma_driver` is the corresponding current shift choice. These
are independent of the interior policy.

The focused black-hole gate evolves the filled data at CFL 0.9 without C2P
fallback or atmosphere reset, reloads its physical-ID artifact on a changed
Cartesian grid, and exercises an exact exterior fixed-background control.
Matched full/equatorial Kerr domains at two resolutions and a matched
full/octant Schwarzschild domain agree in minimum lapse and constraint extrema
to roundoff after one step. The gate also proves that unsupported excision,
incompatible physical-ID spin, and incompatible reflection planes fail before
evolution instead of changing the requested problem.

## Symmetry and tilt contract

Kerr spin is an axial vector. A Cartesian reflection plane is valid only when
it passes through the black-hole origin and the spin is normal to that plane.
Thus a z-spin hole may use equatorial z reflection; a Schwarzschild hole may
use three reflection planes. Independent x/y reflections are not valid for a
nonzero z spin.

Disk orientation is separately represented by `initial_data.disk_axis1/2/3`
and persisted in physical-ID metadata. The present reduced r-theta torus owner
accepts only the aligned +z axis. A future tilted disk is a 3D ID owner using
the same materialization and metadata contract; it must not reinterpret the
stellar rotation axis or the black-hole spin axis.

The orientation contract therefore has three independent axial quantities:
black-hole spin, stellar rotation axis, and disk axis. Aligned axisymmetric ID
may reduce to `r theta`; a genuinely tilted composition is full 3D even though
the target evolution interface and stored ADM/fluid field schema are unchanged.
No current input is silently rotated into the reduced owner.

## Next solver layer

The primary constraint-solved black-hole method is Kerr--Schild-seeded,
conformally curved XCTS with Cook--Pfeiffer quasi-equilibrium apparent-horizon
boundary conditions. Exact Kerr--Schild remains the fixed-background and
vacuum control. Bowen--York puncture data are not the primary route. True
evolution excision and full-grid filled evolution share the exterior solution,
but have separate interior ownership and validation gates.

True excision is two contracts, not one flag: the ID solve owns an apparent-
horizon inner boundary, while evolution owns an active-domain mask, one-sided
stencils/prolongation near the mask, and a causally pure-outflow inner boundary.
The present configuration reserves this policy but rejects it until those
owners and their moving/static-mask validation exist.
