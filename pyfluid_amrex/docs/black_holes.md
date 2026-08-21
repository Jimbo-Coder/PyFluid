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
  still needs a constraint-improved interior or long-lived gauge treatment.
- `excision` is a reserved contract and currently fails configuration. It will
  require an active-cell mask, a pure-outflow inner evolution boundary, and
  the Cook--Pfeiffer XCTS apparent-horizon boundary before production use.

The filled Kerr control now uses ordinary advective `one_plus_log`,
`gamma_driver`, RK4, and the input-specific tested KO strength `0.32`.
Positive `spacetime.chi_floor` and `spacetime.lapse_floor` values are enforced
by the Z4c algebraic projection; both default to `1e-10`. The separate
`slow_start` lapse choice remains available: it adds a Gaussian-in-time
relaxation of the lapse toward `sqrt(chi)`. In the present coarse filled-Kerr
stress it failed later than the old low-dissipation setup but earlier than
ordinary 1+log, so it is an optional experiment rather than the production
recommendation. Gauge choices are independent of the interior policy.

The focused black-hole gate evolves the filled data at CFL 0.9 without C2P
fallback or atmosphere reset, includes a 16-step moving-puncture-gauge stress,
reloads its physical-ID artifact on a changed Cartesian grid, and exercises an
exact exterior fixed-background control.
Matched full/equatorial Kerr domains at two resolutions and a matched
full/octant Schwarzschild domain agree in minimum lapse and constraint extrema
to roundoff after one step. The gate also proves that unsupported excision,
incompatible physical-ID spin, and incompatible reflection planes fail before
evolution instead of changing the requested problem.

## Horizon diagnostics and finder boundary

`black_hole_horizon_radius` reports the analytic Kerr--Schild seed radius
`r_+`; it is not a numerically found horizon. The exterior Hamiltonian L2/Linf
and momentum Linf diagnostics accept a point only when the complete fourth-
order derivative stencil lies at `r_KS > r_+`. This prevents arbitrary filled
interior data from leaking into the vacuum accuracy norm. On the exact Kerr
control the N=32 to N=64 exterior Hamiltonian Linf/L2 and momentum Linf improve
by factors `4.91`, `3.33`, and `4.09` after one step, while matched symmetry
domains retain identical extrema.

`apparent_horizon_expansion()` is the shared pointwise geometric owner for

`Theta = D_i s^i + K_ij s^i s^j - K`.

It accepts a general Cartesian level-set gradient and Hessian and is checked
against both the analytic expansion `2/R` of a sphere in flat space and zero
expansion on the exact Kerr horizon. This establishes the equation and sign
convention, but does not yet constitute an apparent-horizon finder.

The first full finder should follow BHaHAHA's infrastructure-independent
hyperbolic relaxation design: interpolate Cartesian `gamma_ij` and `K_ij`
from the AMR hierarchy to a solver-private spherical volume, relax a radial
graph `r=h(theta,phi)` with a damped wave system, and use coarse-to-fine angular
continuation. It should reuse the existing hierarchy interpolation and surface
diagnostic owners and persist only the horizon shape and diagnostics. A direct
Newton backend in the style of AHFinderDirect remains a useful later polishing
or comparison method. Neither implementation should enter the evolution flux
template product.

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
owners and their moving/static-mask validation exist. A found horizon is a
prerequisite for dynamic masks and black-hole diagnostics, not by itself an
authorization to excise cells.
