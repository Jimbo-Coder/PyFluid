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

## Analytic static trumpet control

`spacetime.initial_data = static_trumpet` selects the (R_0=M) member of
Dennison and Baumgarte's analytic Schwarzschild trumpet family. It is a
spinless, horizon-penetrating puncture with coordinate horizon radius (M)
and areal horizon radius (2M). The only supported interior declaration is
`spacetime.black_hole_interior = puncture`; it is neither the Kerr fill nor
excision. The current owner is 3D Cartesian and supports uniform/block-AMR
materialization, full or reflection-reduced domains, portable physical-ID
save/reload, fixed-background test-fluid disks, and evolved vacuum Z4c.

The physical ADM metric diverges at the puncture, but the evolved conformal
state has finite radial limits. Initialization therefore writes the analytic
Z4c state directly instead of passing the puncture through ADM conversion.
The trace-free conformal extrinsic curvature has a direction-dependent limit;
its spherical average supplies the one exact nodal puncture value. The
puncture must lie on a Cartesian mesh face in every direction, which keeps
fixed-background cell-centered metric sampling away from the singular point
and makes reflection planes exact. Analytic ADM identities, finite-difference
metric derivatives, and the complete generated stationary RHS away from the
puncture are regression checked.

`spacetime.lapse_gauge = trumpet` implements the lapse condition compatible
with the static solution,

`dt(alpha) = beta^i d_i(alpha) - alpha (1-alpha) K`,

and `spacetime.shift_gauge = frozen` retains the analytic shift. This is a
diagnostic gauge pair, not the selected production default. In the coarse
`dx=0.5M`, CFL-0.9, 16-step comparison with KO strength 0.32, the matched pair
changes mean coordinate horizon radius by only 1.37 percent, but ends with a
5.10 percent proper-area error and larger exterior constraint norms. The same
ID under the ordinary advective 1+log/Gamma-driver pair changes coordinate
radius by 10.10 percent but ends at 0.13 percent proper-area error. Both runs
complete with zero C2P fallback or atmosphere reset. These are short
regression measurements, not a resolution-converged universal gauge ranking;
the ordinary moving-puncture pair remains the supplied evolution input.

The focused black-hole gate also rematerializes a saved trumpet artifact from
`32^3` to `24^3` and matches full versus octant one-step constraint extrema to
roundoff. The equilibrium-torus gate builds a fixed-background constant-(l)
Schwarzschild disk with resolved support `7.141--19.480 M`, baryonic mass
`0.13195`, and zero recovery. Nonzero trumpet spin, a filled trumpet interior,
and true excision remain rejected rather than silently mapped to another
method.

## Constraint-solved spinning horizon shell

`pyfluid.ic = black_hole`, `spacetime.initial_data = xcts`, and
`initial_data.solve_only = 1` select the first native constraint-solved
black-hole owner. It is an origin-centered vacuum hole on a spherical shell
whose inner face is the quasi-equilibrium apparent horizon `r=M/2`. The
Cook--Pfeiffer conditions give the conformal-factor MOTS Robin condition, use
zero horizon lapse as the slicing choice, and set the tangential orthonormal
shift to `beta_hat_phi=Omega_r r_H sin(theta)`. `Omega_r` is a horizon-boundary
coefficient, not the physical spin. `spacetime.dimensionless_spin3` is instead
the target signed Christodoulou spin; the reduced owner requires the other two
spin components to vanish. The atmosphere is only the stored fluid-floor
convention and is excluded from the vacuum elliptic sources.

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

Nonzero spin starts from that exact spin-zero solution and continues in the
horizon rotation coefficient while measuring `J_H/M_H^2` on every accepted
throat. It stops as soon as the measured physical spin brackets the target,
then calibrates the coefficient; using a Kerr estimate all the way to its
nominal coefficient can needlessly cross a nonlinear fold. The mapped
`256x32` `chi_H=0.3` control gives `J_H=0.32381635`,
`M_H=1.03893597`, `chi_H=0.29999997`, and residuals
`2.19e-12/5.57e-12/1.39e-13`. The demonstrated high-spin regression reaches
`chi_H=0.67499489` for target `0.675`; target `0.7` presently reaches a
nonfinite nonlinear RHS before bracketing and is an open numerical-owner
limit, not a physical Kerr or quasi-equilibrium limit.

The native throat summary uses that solved spherical geometry to persist area,
irreducible mass, proper equatorial and polar circumferences, and the same
Kerr-assumption mass/spin convention as the runtime apparent-horizon tracker.
Because the native solve is axisymmetric, it additionally evaluates the
standard quasi-local angular momentum with the exact axial Killing vector and
persists the Christodoulou mass
`M_BH=Mirr sqrt(1+J_H^2/(4 Mirr^4))` and signed spin `J_H/M_BH^2`.
On the changed `256x32` Schwarzschild shell, area and Kerr-mass relative errors
are `1.56e-3` and `8.81e-4`; doubling radial resolution from the `128x16` seed
improves both by about four. The inferred zero-spin error decreases from
`5.67e-2` to `2.83e-2`; this slower-looking quantity is the expected
ill-conditioning of the inverse Kerr relation near exactly zero spin. Matched
2D and periodic `r theta phi` storage use the same reduced physical operator
and agree to roundoff; the full-storage `chi_H=0.3` control has `256x32x4`
cells and no phi variation. The Schwarzschild controls give `J_H=0` and
Christodoulou spin zero exactly.

The same shell now owns aligned, unmagnetized, self-gravitating disks.
`initial_data.disk_gravity = coupled` rebuilds the disk Euler state and its
energy, momentum, and stress-trace sources inside the XCTS fixed point;
`test_fluid` leaves the solved hole geometry unchanged. The standard gate
requires a nonzero coupled matter-iteration count, measurable geometric
backreaction relative to the test-fluid model, finite throat data, preserved
disk metadata and proper-volume baryonic mass after Cartesian materialization,
and a clean one-step fixed-XCTS evolution. Physical-spin calibration is nested
outside the settled matter fixed point. The zero-target control converges to
`chi_H=-7.50e-7` instead of retaining the disk-induced
`chi_H=-1.31e-3` from a zero rotation coefficient; the target-`0.3` control
converges to `chi_H=0.30004288` with Hamiltonian/lapse/momentum residuals
`6.74e-12/1.06e-11/6.28e-13`.

This is still an exterior solve-only owner. A Cartesian evolution artifact
uses a finite smooth interior fill below the throat and retains the solved XCTS
geometry outside it. That is useful for short fixed-spacetime controls, but it
is neither excision nor a constraint-solved complete interior, and the current
massive-disk evidence is not an evolved-Z4c/full-NR claim. The vacuum
`chi_H=0.3` artifact has also been materialized onto a `64^3` Cartesian grid
and completed one CFL-0.9 RK4 evolved-Z4c step with zero recovery or atmosphere
reset. Its whole-domain constraint norms are dominated by the deliberately
constraint-violating interior fill, so this is a routing and finite-evolution
smoke, not an exterior-accuracy or stability claim. Target-grid
constraint polishing, a constraint-improved complete interior, and true
excision remain open. `N x M x 1` and larger periodic-phi storage are
dimensional-degeneracy checks of the same owner, not additional methods in the
catalog.

The current constant-`ell` stress family uses `ell=4.25`, requested
`r_in=6 M`, and varies the gamma-two polytropic scale. On matched adaptive
`320x80` shells, the largest clean eight-step CFL-0.9 Cartesian control has
disk baryonic mass `M0=0.309183`, throat area `54.2473`, and hence
`M0/Mirr=0.29762`; it reaches `t=2.102 M` with zero C2P fallback or atmosphere
reset. The next tested model has `M0=0.313796` and `M0/Mirr=0.30160`; it
finishes eight steps but requires eight atmosphere resets at step one, so it
is not accepted as clean. This brackets the tested numerical handoff boundary,
not a physical maximum disk mass or long-time stability threshold. The
under-relaxed fixed-point update and source resolution must be matched before
comparing adjacent models: the coarser midpoint produced 24 resets, while its
matched-resolution re-solve produced none.

A Rayleigh-stable q-power continuation improves that tested handoff boundary.
The conventional angular-velocity exponent `q_Omega=1.85`, meaning
`Omega ~ varpi^-q_Omega`, maps to the generated catalog parameter
`rotation_power_q=2/q_Omega=1.0810810810810811`; these two q conventions must
not be interchanged. With `A=6 M`, `K=100`, and requested `r_in=6 M`, saved
coupled artifacts now continue the same family by increasing `Omega_c` while
preserving the converged matter state. On the adaptive `768x160` shell, a
physical-spin-zero re-solve at `Omega_c=0.157/M` reaches `M0=1.075252`, throat
area `56.92796`, `Mirr=1.064211`, quasi-local `J_H=9.12e-6`, and
Christodoulou `M_BH=1.064211`, hence `M0/M_BH=1.010375`. The measured signed
horizon spin is `8.05e-6`; the Hamiltonian/lapse/momentum residuals are
`1.56e-10/1.50e-10/5.18e-12`. Fourth-order materialization onto an
equatorial-half `160x160x80` Cartesian grid retains these metadata and has
maximum conserved/primitive round-trip errors `5.41e-12/6.38e-11`. It then
completes eight CFL-0.9 SSPRK3/PWL/HLL fixed-XCTS steps to `t=2.030 M` with
zero C2P fallback, atmosphere reset, or reconstruction/Riemann correction.
The raw open-domain mass drift is `1.54e-5`. This is the first constructed and
short-evolved point whose mass-ratio-one claim uses the physical
Christodoulou denominator rather than `Mirr`; it is not a long-time stability
result or a physical maximum-mass claim.

Continuing that same adaptive `768x160` family with smaller accepted angular-
velocity steps reaches `Omega_c=0.300/M`, `M0=4.624213`, throat
`Mirr=1.403521`, and hence the explicitly labeled `M0/Mirr=3.294723`, with
Hamiltonian/lapse/momentum residuals
`2.03e-10/1.48e-10/2.56e-11`. This is the largest constructed point checked in
that continuation, not a physical endpoint and not an evolved-stability
claim. Direct larger jumps can leave the nonlinear basin even when an
intermediate saved model reaches the same target; the shared continuation-step
control now applies to disk angular velocity as well as stellar rotation.

## Horizon diagnostics and finder boundary

For comparisons with spinning-hole literature, `Mirr=sqrt(A_H/(16 pi))` is
not generally the quoted black-hole mass. Shibata's circumference measure is
`M_C=C_e/(4 pi)`; the Christodoulou measure is
`M_BH=Mirr sqrt(1+J_H^2/(4 Mirr^4))`. The apparent-horizon tracker reports
proper equatorial and polar circumferences and, assuming the surface obeys the
vacuum-Kerr area/circumference relations, `M_C` as `M_H^Kerr` and
`|chi_H^Kerr|`. These horizon-derived values are catalog diagnostics and are
automatically appended to `scalars.tsv` when horizon finding is enabled. The
implemented relations are `M_H^Kerr=C_e/(4 pi)` and
`|chi_H^Kerr|=sqrt(1-[A_H/(8 pi (M_H^Kerr)^2)-1]^2)`; the polar circumference
is retained as an independent shape diagnostic. The
Kerr qualifier is deliberate: the proper area, circumferences, and `Mirr`
remain the assumption-free geometric outputs for a distorted dynamical
horizon.

Constraint-solved spherical BH and BH-disk artifacts persist the corresponding
`black_hole_throat_*` quantities directly at solve/continuation cadence,
including exact-axisymmetry `J_H`, Christodoulou mass, and signed spin. The
runtime diagnostic catalog consumes the live apparent-horizon tracker and
emits `black_hole_horizon_*` quantities at evolution cadence. It integrates
the three flat-coordinate rotation generators about the measured centroid and
therefore labels the resulting quantities
`black_hole_horizon_coordinate_angular_momentum*` and
`black_hole_horizon_coordinate_christodoulou_*`. On a centered or translated
aligned Kerr horizon the relevant generator is the axial Killing vector; the
analytic `a/M=0.6` control measures signed spin with absolute error
`1.87e-4` at 16 polar points. These are two
deliberately distinct owners: the native throat summary does not require
rematerializing the ID into Cartesian space, while the runtime values follow
the evolved found surface. A generic distorted 3D horizon still needs a
Cook--Whiting approximate-Killing-vector solve for a coordinate-independent
spin; the current catalog keeps the coordinate qualifier rather than silently
claiming that stronger result.

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
system, SSPRK3, sixth-order pole-parity angular differences, one radial metric
interpolation frozen across each pseudo-time RK step, scale-aware residuals,
coarse-to-fine angular continuation, and residual-tested over-relaxation.
Independent angular-point kernels use an isolated work-sized OpenMP team;
serial norm/diagnostic passes retain deterministic summation order. The finder
is one separately compiled 3D library, so it neither enables AMReX OpenMP nor
enters the reconstruction/Riemann template product. The resulting surface is
broadcast.

On the analytic `a/M=0.6` Kerr control, the local two-thread profile solves the
routine 16-polar-point surface in about `0.258 s`. Repeated strict 32-point
solves take `6.81--6.82 s`, versus a repeated one-worker median of `11.89 s`,
while preserving the same 18,150 iterations, residual, radius error, and area
error. Build with
`PYFLUID_AMREX_HORIZON_OPENMP=OFF` when an OpenMP runtime is unavailable; the
mathematical serial path remains the same.

The principal runtime controls are `horizon.find_int` (default 8),
`radial_points` (48), `polar_points` (32, with twice as many azimuthal points),
`coarsest_polar_points` (8), the radial search bounds and center, `cfl` (0.98),
`eta_times_mass` (7), the L-infinity/L2 expansion tolerances, and the iteration
limit. Every successful search appends `apparent_horizon.tsv` and writes the
radial shape under `apparent_horizon/ah########.tsv`. A restart restores the
latest compatible shape at or before its checkpoint step. An exact-step shape
is immediately valid for masks; an older shape is only an initial guess and
forces a fresh search before output or AMR use. Current diagnostics include
expansion residuals, proper area, irreducible mass, circumferences,
coordinate-rotation angular momentum/spin, area centroid, and
minimum/maximum/mean coordinate radius.

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
center in 3D Cartesian stored spacetime. The AMR adapter maps sample points
through configured Cartesian reflection planes before interpolation, applies
rank-two parity to `gamma_ij`, `gamma^ij`, and `K_ij`, and applies the extra
derivative-index parity to `d_k gamma_ij`. It also uses parity-filled nodal
ghosts for centered metric derivatives at a reflection plane. On the matched
`32^3`/`32x32x16` Kerr control, the full and equatorial-half finder residuals,
area, irreducible mass, and radius extrema agree to roundoff. Multiple
horizons, moving-center tracking, approximate-Killing-vector spin, an
AHFinderDirect-style Newton polishing backend, and true evolution excision
remain later capabilities.

The coupled XCTS BH-disk evolution artifact has a distinct live-finder
boundary. Its native spherical source enforces an apparent-horizon throat at
`r=M/2`, but the current finite Cartesian interior fill does not yet provide a
surface to which the nonlinear live finder converges after materialization,
even with fixed local refinement through `dx=0.0625 M`. Source-throat area can
still be integrated from the authoritative native ADM artifact. This is an
interior-fill/materialized-horizon compatibility limitation, not the repaired
reflection-sampling bug.

Saved evolved geometries can be re-found without taking another evolution
step:

```console
uv run python pyfluid_amrex/tools/audit_kerr_horizon_convergence.py
```

The optional `--evolve-resolution` branch runs the aligned `a/M=0.5` filled-
Kerr control to `t=2.5 M` at finest spacings `0.5`, `0.25`, and `0.125 M`,
writes the final checkpoint, and re-finds every final geometry with the same
surface settings. The measured final proper-area errors are 7.65, 2.06, and
0.62 percent, while the changes from each grid's measured initial area are
5.55, 2.08, and 0.29 percent. This demonstrates early-time evolution-grid
convergence; it does not establish the same rate through the existing
`t=10.69 M` routine-resolution run. Re-finding that run's step-64 checkpoint
at 16, 24, and 32 polar points leaves 4.67--4.85 percent area error, so its
material drift is not a loose finder-tolerance artifact.

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

A rotating disk supplies its own axial angular momentum even when the central
hole is Schwarzschild. Its Cartesian reflection plane must pass through the
disk center and have normal parallel to the disk axis. Consequently, an
aligned z-axis BH disk may use the z-equatorial half-domain, but it may not
inherit the x/y reflections of a spinless isolated hole. Common solver startup
enforces these black-hole, star, and disk conditions for fresh ID, loaded ID,
and evolution rather than only while running an elliptic ID owner.

Equatorial reduction still runs the complete 3D Cartesian operator and retains
arbitrary azimuthal structure, including odd-m modes. It removes only modes
that are odd across the disk midplane. It is therefore appropriate for aligned
equilibrium checks, early accretion, and deliberately equatorially symmetric
instability studies. A tilted disk, warp/corrugation mode, vertical asymmetry,
or unrestricted turbulent/dynamo evolution requires the full vertical domain.
Axisymmetric 2D evolution would additionally remove every nonzero azimuthal
mode and is a different physical restriction.

The focused fixed-Kerr torus regression compares `32^3` with
`32 x 32 x 16` through two complete CFL-0.9 SSPRK3 steps. Symmetry-restored
mass, energy, and axial angular momentum agree within `8.2e-10` relative,
maximum density within `4.5e-9`, and both paths use zero recovery/reset. Cell
count is exactly halved; this tiny run improves measured wall time only
`0.444 -> 0.378 s` (`1.18x`) because startup, MPI, and diagnostics dominate.
Larger production grids should be benchmarked before assigning a realized
speedup from the ideal twofold cell reduction.

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

The aligned conformally flat Cook--Pfeiffer owner now covers physical spin and
self-gravitating disks. The next constraint-solved black-hole step is a
Kerr--Schild-seeded conformally curved XCTS extension and, separately, a full
3D nonaxisymmetric owner for genuine relative tilt. Exact Kerr--Schild remains
the fixed-background and vacuum control. Bowen--York puncture data are not the
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
