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
relaxation of the lapse toward `sqrt(chi)`. Its calibrated defaults are
`spacetime.slow_start_strength = 0.6` and
`spacetime.slow_start_time = 20`. The former amplitude-one/time-five stress
overdrove the gauge transition and failed at step 54. With the calibrated
values the same coarse Kerr case reaches 64 CFL-0.9 steps with zero fluid
recovery, but its exterior Hamiltonian and momentum norms remain roughly three
to five times larger than the matched ordinary-1+log control. Slow start is
therefore functional and available for puncture experiments, not the selected
filled-Kerr gauge. Gauge choices are independent of the interior policy.

The focused black-hole gate evolves the filled data at CFL 0.9 without C2P
fallback or atmosphere reset, includes a 16-step moving-puncture-gauge stress,
reloads its physical-ID artifact on a changed Cartesian grid, and exercises an
exact exterior fixed-background control.
Matched full/equatorial Kerr domains at two resolutions and a matched
full/octant Schwarzschild domain agree in minimum lapse and constraint extrema
to roundoff after one step. The gate also proves that unsupported excision,
incompatible physical-ID spin, and incompatible reflection planes fail before
evolution instead of changing the requested problem.

The longer `a/M=0.5` AMR control reaches `t=10.690 M` in 64 CFL-0.9 steps with
zero recovery/reset. Its `spacetime` policy retains 98,304 cells, two unused
levels, and 15 no-op regrids; it follows strong-field `abs(chi-1)` support
rather than an outgoing gauge tail. Mean coordinate horizon radius grows from
`1.908 M` to `2.417 M`, so the usual coordinate shrinkage is absent. Proper
area grows from `47.029 M^2` to `49.184 M^2` (4.9 percent above exact at the
end), which is a real accuracy limitation hidden by a radius-only statement.
At the same routine near-hole spacing, `a/M=0.7` reaches `t=2.476 M` with final
area 2.6 percent above exact. A filled `a/M=0.8` hole is strongly sensitive to
the arbitrary C2 interior transition and degrades by `t=1.19 M`; treat this as
motivation for a constraint-improved interior or excision, not a physical spin
limit of Kerr or the finder.

The aligned fixed-background BH-disk control uses the generated q-power law
`j=A^2 Omega[(Omega_c/Omega)^q-1]` with catalog `q=1.05`,
`Omega_c=0.1355`, `A=6 M`, and requested `r_in=10 M`. The catalog exponent
corresponds asymptotically to the conventional disk shear exponent
`q_Omega=2/q=1.905`, slightly below constant-angular-momentum-like `q_Omega=2`.
The saved `64^3` artifact has resolved support `14.098--25.981 M`. Its
ratio-4/2 AMR evolution reaches `t=51.753 M` in 24 CFL-0.9 steps with zero
recovery/reset, one level of headroom, and 355,840 cells (47.4 percent of its
guard, versus 2.10M cells for uniform `dx=1 M`). The maximum density decreases
3.58 percent and the boundary-corrected mass residual is `4.28e-13`. Large
late-time Lorentz factors are confined to atmosphere in the filled hole and
remain the practical reason to implement true excision before calling this a
long-lived accretion evolution.

## Constraint-solved Schwarzschild shell

`pyfluid.ic = black_hole`, `spacetime.initial_data = xcts`, and
`initial_data.solve_only = 1` select the first native constraint-solved
black-hole owner. It is deliberately the smallest physically complete case: a
spinless origin-centered vacuum hole on a 2D spherical `r theta` shell whose
inner face is the isotropic Schwarzschild apparent horizon `r=M/2`. The scalar
XCTS unknowns use
the Cook--Pfeiffer apparent-horizon Robin condition for the conformal factor,
zero lapse at the inner face, zero shift, and asymptotically flat outer Robin
conditions. The atmosphere is only the stored fluid-floor convention and is
excluded from the vacuum elliptic sources.

Uniform and geometrically stretched finite shells share one separable
axisymmetric elliptic operator. The supplied `M=1` mapped control spans
`[M/2,32M]` with a last/first radial-cell-width ratio of 16. The `256x32`
analytic control has
Hamiltonian/lapse/momentum residuals
`2.11e-12/4.89e-12/0`, conformal-factor second-order convergence, zero shift
and extrinsic curvature, and a horizon lapse of exactly zero in the portable
nodal ADM artifact. Its losslessly compressed artifact is about 8.1 MiB. A
saved `128x16` shell can seed and re-solve a changed `256x32` mapped shell; the
result is checked directly against isotropic Schwarzschild and uses no fluid
recovery.

This is an exterior solve-only owner, not yet a complete-grid evolution model.
Spinning Cook--Pfeiffer data, black-hole/disk coupling, target-grid constraint
polishing, and the separate filled or true-excision evolution interiors remain
open. Likewise, `N x M x 1` will be a 3D `r theta phi` degeneracy gate against
this 2D solution once the native 3D curvilinear elliptic owner exists; it is
not represented as another owner in the method catalog.

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
expansion on the exact Kerr horizon.

`horizon.enabled = 1` activates the independent BHaHAHA-style finder. The AMR
adapter samples Cartesian `gamma_ij`, `gamma^ij`, their spatial derivatives,
and `K_ij` once onto a solver-private spherical volume. The IO rank then
relaxes the radial graph `r=h(theta,phi)` with the published damped-wave
system, SSPRK3, sixth-order pole-parity angular differences, stage-consistent
radial interpolation, scale-aware residuals, coarse-to-fine angular
continuation, and residual-tested over-relaxation. The resulting surface is
broadcast; it never enters the reconstruction/Riemann template product.

The principal runtime controls are `horizon.find_int` (default 8),
`radial_points` (48), `polar_points` (32, with twice as many azimuthal points),
`coarsest_polar_points` (8), the radial search bounds and center, `cfl` (0.98),
`eta_times_mass` (7), the L-infinity/L2 expansion tolerances, and the iteration
limit. Every successful search appends `apparent_horizon.tsv` and writes the
radial shape under `apparent_horizon/ah########.tsv`. A restart restores the
latest compatible shape at or before its checkpoint step. An exact-step shape
is immediately valid for masks; an older shape is only an initial guess and
forces a fresh search before output or AMR use. Current diagnostics include
expansion residuals, proper
area, irreducible mass, area centroid, and minimum/maximum/mean coordinate
radius.

`apparent_horizon_mask` and `excision_mask` are catalog-owned cell-centered
auxiliary GFs. The first marks the found interior; the second is a configurable
strictly nested fraction of it. `horizon.refine_band_cells > 0` tags a band
from the surface's signed radial distance for ordinary AMReX regridding. Both
masks are geometry
and tracking products only: they do not remove active cells, reflect fluid,
change fluxes, or claim to be an event horizon. The focused gate checks a
spinning Kerr surface against the exact radius/area, creates a dynamic fine
level from the horizon band, writes nested masks on both levels, and restores
the tracked shape across checkpoint restart.

The current owner finds one star-shaped horizon about a configured fixed
center in 3D Cartesian stored spacetime. Multiple horizons, moving-center
tracking, quasilocal spin, an AHFinderDirect-style Newton polishing backend,
and true evolution excision remain later capabilities.

## Event-horizon postprocessing boundary

An event horizon is global in time and is not a live replacement for the
apparent-horizon finder. The production direction is an offline owner that
integrates an outgoing null surface backward through a completed spacetime
history. A late, settled apparent horizon supplies a useful terminal seed, but
the result is an event horizon only after the backward history solve.

The evolution-side prerequisite is therefore a compact, opt-in spacetime
history, not another checkpoint flavor. Each record must carry an ordered time,
the Cartesian hierarchy/coordinate metadata, and the lapse, shift, and spatial
metric needed to interpolate the four-metric in space and time. The writer and
offline reader must share one declared field schema and reject gaps or layout
changes that the selected temporal interpolant cannot represent. Initial-data
artifacts and ordinary sparse checkpoints do not by themselves satisfy this
contract.

For a settled single hole, a star-shaped backward surface is a useful first
validation backend. A general BBH/multiple-horizon backend cannot be restricted
to one radial graph: it must represent disconnected components, merger
topology change, caustics/crossovers, and generator entry. A volume level set
is the clean general owner; extracted surface meshes and generator histories
are postprocessing products, never solver GFs or evolution masks. The first
tests should be analytic stationary Schwarzschild/Kerr histories followed by a
stored dynamical single-hole control. BBH support should wait for long,
well-resolved spacetime histories and multi-horizon apparent-horizon tracking.

No runtime `event_horizon` option is registered yet. Adding an empty selector
before the history writer, backward operator, and analytic gate exist would
create a misleading compatibility promise.

## Symmetry and tilt contract

Kerr spin is an axial vector. A Cartesian reflection plane is valid only when
it passes through the black-hole origin and the spin is normal to that plane.
Thus a z-spin hole may use equatorial z reflection; a Schwarzschild hole may
use three reflection planes. Independent x/y reflections are not valid for a
nonzero z spin.

Disk orientation is separately represented by `initial_data.disk_axis1/2/3`
and persisted in physical-ID metadata. The reduced r-theta torus owner accepts
only the aligned +z source axis. The direct 3D Cartesian fixed-Kerr torus may
use any disk axis parallel or antiparallel to the nonzero black-hole spin;
the pointwise torus and Kerr providers are both arbitrary-axis. A relative
black-hole/disk tilt is not stationary circular equilibrium and therefore
rejects pending an explicit nonequilibrium or quasiequilibrium 3D owner.

The orientation contract therefore has three independent axial quantities:
black-hole spin, stellar rotation axis, and disk axis. Aligned axisymmetric ID
may reduce to `r theta`; a genuinely tilted composition is full 3D even though
the target evolution interface and stored ADM/fluid field schema are unchanged.
The portable materializer may rigidly rotate and translate an entire isolated
reduced solution, but it cannot change those axes relative to one another.

## Next solver layer

The next constraint-solved black-hole step is the spinning,
Kerr--Schild-seeded conformally curved XCTS extension of the present
Cook--Pfeiffer Schwarzschild shell. Exact Kerr--Schild remains the
fixed-background and vacuum control. Bowen--York puncture data are not the
primary route. True evolution excision and full-grid filled evolution may
share the exterior solution, but have separate interior ownership and
validation gates.

True excision is two contracts, not one flag: the ID solve owns an apparent-
horizon inner boundary, while evolution owns an active-domain mask, one-sided
stencils/prolongation near the mask, and a causally pure-outflow inner boundary.
The diagnostic candidate mask now exists, but `black_hole_interior = excision`
still rejects until the evolution owners and characteristic/outflow validation
exist. A found horizon is a prerequisite, not by itself authorization to
discard cells.
