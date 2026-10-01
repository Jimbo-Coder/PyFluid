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
- `excision` makes interior fluid cells inactive and supplies a one-sided
  Valencia inner flux. Fixed Kerr uses the horizon-fraction mask. Evolved Z4c
  instead uses the fixed coordinate sphere set by `black_hole_fill_radius`,
  with exterior derivative/transfer closures and live characteristic and
  horizon-containment checks. Finite storage below the cut is not an evolved
  physical interior. See the short-run validation boundary below; evolved
  excision is not yet a long-accretion production claim.

These values expose three useful ID-to-evolution interior policies:

1. `excision`, which omits the interior and supplies a causally outflowing
   inner boundary;
2. `puncture`, including either wormhole-puncture or direct-trumpet initial
   slices on a complete grid; and
3. `filled`, which smoothly completes an exterior solution on the complete
   evolution grid after the exterior ID solve.

The third policy is the user-facing smooth-fill option; `turduckening` is only
the established literature name for this operation. It is not a different
method or a useful configuration term. For constraint-solved XCTS/COCAL data,
the fill belongs strictly to materialization/handoff and must not feed back
into the exterior elliptic solve. A wormhole puncture and a direct trumpet are
two initial slices within the puncture policy: moving-puncture gauge normally
drives the former toward the latter. Future direct spinning-trumpet data should
therefore remain `black_hole_interior = puncture`, not add a `trumpet` interior
policy.

Fixed versus dynamically evolved spacetime is an orthogonal choice. Smooth
full-grid and puncture/trumpet policies support evolved, matter-coupled Z4c.
Fixed-background Kerr/Schwarzschild is a valuable test-fluid limit and
comparison baseline, not a fourth interior method. Evolved excision has its
own spacetime boundary; its validation must not be inferred from working
fixed-background hydro excision.

### Evolved excision: current validation boundary

`inputs/kerr_schild_coupled_excision_3d.in` is a short coupled control with
harmonic lapse, frozen shift, RK4, and CFL 0.9. One fixed spherical worldtube
owns fluid and Z4c inactivity. The boundary must remain inside the found AH
and have outgoing physical and gauge characteristics; an incoming mode aborts
instead of silently imposing missing boundary data. There is no moving-cut
controller. In particular, moving-puncture gauges are not automatically
compatible with a chosen cut radius.

Generated one-sided derivatives and negative-semidefinite masked KO close
the spacetime operator. Centering, cell-to-face metric interpolation, and
nodal AMR interpolation use exterior samples. Fluid reconstruction reduces
to its cell-centered trace where its wider stencil crosses the cut; the
inner face uses the exterior state only. Inactive primitive-cache entries
are finite scratch, without C2P or any conserved-state reset. They are not
physical atmosphere injection. Canonical CT magnetic storage is retained.

Four-step coupled hydro and divergence-free CT-RMHD controls pass at CFL 0.9,
including MP5/HLLD with finite-state poisoning enabled. Restart and
AMR/subcycling/remake controls also pass; recovery/reset counts are zero and
baryon balance closes below `6e-17` relative. A compact hybrid-EoS disk reaches
four SSPRK3 steps (`t=0.6 M`) with coupled metric evolution and zero recovery.
That disk starts from test-fluid Kerr data, not a self-gravitating constraint
solve, so it is a routing/robustness control rather than a physical equilibrium
or accretion result. The baryon ledger
remains valid in dynamical geometry (`external_mass_budget_valid`); a
fluid-only energy ledger does not (`external_budget_valid=0`). Linear
operator and transfer convergence tests are separate from nonlinear
long-time stability: coarse longer controls still lose horizon tracking.
The resolved control reaches 32 steps (`t=1.2 M`) without recovery, but its
tiny outer domain has growing exterior constraint error; a long
self-gravitating disk evolution remains unvalidated. Surface ADM
and horizon diagnostics are preferable to interpreting an uncorrected
whole-volume ADM integral across an omitted interior.

The opt-in `tools/validate_black_hole_excision_convergence.py` separates
resolution and outer extent at matched time. Its legacy exterior norms exclude
dx-dependent stencil layers and must not be mistaken for fixed-region norms.
The opt-in `--fixed-regions` measurements use the same coordinate box of
half-extent `3.5 M`, exclude Euclidean radius below `2.5 M`, and split near/far
at `3 M`, intersected with the represented `z >= 0` half-domain. They reuse
the canonical runtime constraint calculation, not a Python reconstruction.

Current spin-0.5 vacuum Kerr controls at CFL `0.9`, `t=1.2 M`, give:

| Domain half-extent | Spacing | Near H RMS | Far H RMS |
| --- | --- | --- | --- |
| 4 M | 0.25 M | 2.49010e-4 | 2.22820e-3 |
| 4 M | 0.125 M | 1.45880e-4 | 1.71826e-3 |
| 6 M | 0.25 M | 1.80489e-4 | 3.34900e-5 |
| 6 M | 0.125 M | 1.04306e-5 | 2.05846e-6 |

The larger-box pair recovers near/far H RMS orders `4.11/4.02` and
momentum-constraint Linf orders `3.97/3.98`. At fixed fine spacing, moving
the boundary reduces near/far H RMS by `14x/835x` with identical measured
nodes and volumes; this is not global-volume dilution. The smaller-box near
region is fourth-order at `t=0.6 M` before late contamination arrives.
All four controls have zero recovery/interior interventions and roundoff
baryon-budget closure. These results identify boundary contamination in this
small-box experiment, not a failed initial-data solve or a need to reduce CFL.
They do not establish long-time or self-gravitating-disk accuracy, and `6 M`
is not a generally adequate production boundary radius. AH search/angular
error must also be reduced before demanding fine-grid area convergence.
Cycle 372 records the exact summaries and commands; reproduce with:

```sh
uv run python pyfluid_amrex/tools/validate_black_hole_excision_convergence.py \
  --cases extent4_n32 extent4_n64 extent6_n48 extent6_n96 \
  --fixed-regions --plot --label boundary_localization
```

The pure-wave stationary-Kerr boundary characterization isolates a continuum-model
error: exact stationary data still receive a nonzero `A11` boundary RHS as
the derivative spacing tends to zero. At radius `4 M`, halving the derivative
spacing from `0.125 M` to `0.0625 M` to `0.03125 M` gives
`0.006797`, `0.006839`, `0.006850`, rather than zero. The outgoing-wave
assumption does not describe that stationary background. The ETK-style
non-wave correction removes that pointwise plateau: at radius `4 M`,
`dx=0.0625/0.03125 M` gives `5.1155e-4/2.5440e-4`. However, the matched
evolved small-box control still loses the expected constraint convergence.
Pointwise consistency is not a full evolution-convergence result; the
fixed-region/domain-distance checks above distinguish these errors.
A corner-compatible constraint-aware outer closure remains open.

### Outer spacetime boundaries

`spacetime.outer_boundary=extrapolate` (default) fills open nodal ghosts with
six-point Lagrange extrapolation and leaves the ordinary RHS on valid boundary
nodes. This is sixth-order extrapolation (quintic exactness), so a second
derivative can retain fourth-order consistency. A globally shorter direction
uses all of its distinct nodes; box decomposition does not shorten the stencil.
`sommerfeld` uses the same ghosts, then supplies an approximate outgoing RHS
for Khat, Theta, contracted Gamma, and A. Its speeds follow the selected
lapse/shift gauge; frozen-gauge Khat/Gamma retain their bulk equations. Other
metric/gauge equations are not replaced. The non-wave correction uses
`spacetime.sommerfeld_power=3` by default, following ETK NewRad: subtract the
radiative model from the neighboring interior MOL RHS and carry that residual
to the boundary with an `r^-n` falloff. It infers the background from current
evolved data, without an analytic Kerr provider or a stored initial snapshot.
Set `sommerfeld_power=-1` for the former pure-wave comparison. Nonnegative
values select the residual's falloff exponent. This is not a constraint-preserving
boundary, and it cannot make a too-small outer domain physically adequate.
The configured spacetime origin must lie strictly inside every open face
(by more than one cell for the non-wave correction, whose donor must have
nonzero radius and lie in the bulk); reflection planes through it remain allowed.

One compiled owner applies the condition at every stage and AMR level.
`tools/validate_z4c_boundary.py` checks flat-space preservation, gauge noise,
reflection, and coupled excision, with optional `--include-amr-restart`.
Add `--include-filled-evolution` for matched-time MP5/LLF live-gauge controls
using SSPRK3 and RK4; they reuse the black-hole validator and scalar checks.
The matched BH scan selects Sommerfeld by default; use `--outer-boundary
extrapolate` for the bulk-RHS control. A run marked PASS there means its
runtime/scalar checks passed, not that its resolution sequence converged.

### Outer fluid boundaries

`pyfluid.bc=outflow` extrapolates ghost data; it does not prohibit incoming
matter. In a shifted chart the relevant velocity is `alpha*v^i-beta^i`.
`diode` uses that transport velocity, then supplies one interior-derived
primitive state to both Riemann traces at each selected outer face. This
retains pressure and magnetic stresses while preventing numerical baryon
influx to roundoff. Merely clipping a ghost velocity does not provide that
guarantee. The same face owner serves RHD/RMHD, the available reconstructions
and Riemann solvers, and candidate correction. Other boundary types and
Newtonian kernels are unchanged. `tools/validate_relativistic_diode.py` owns
the short shifted-boundary evolution controls.

This fluid condition is independent of the spacetime outer boundary and
cannot fix incoming gravitational/gauge/constraint error. It is also separate
from the inner excision surface.

The local COCAL/ETK BHT WIP provides a useful comparison for improving the
filled option. Its smooth/legacy paths split `K_ij` into trace and trace-free
parts, extrapolate the trace-free part, project it trace-free against the final
filled metric, damp `tr(K)` toward the center, and remove interior hydro. That
is a careful turduckening construction rather than simple component blending,
but it still neither solves the constraints nor supplies an evolution excision
boundary. For analytic fixed Kerr, deriving `K_ij` from the completed
stationary lapse/shift/spatial metric is the stronger consistency target. For
an exterior numerical XCTS/COCAL artifact without a complete stationary
four-metric, trace-split extrapolation is a bounded compatibility option whose
exterior independence and constraint propagation must be validated.

Smoothing may be used as an initial guess for another global constraint solve,
but that operation is not a local repair. Elliptic constraint projection can
change the exterior fields and therefore the measured horizon, ADM mass, and
spin. Solving only inside the horizon while requiring both values and smooth
derivative matching to a frozen exterior is instead a nontrivial localized
constraint-gluing problem. The production rule is consequently:

- an excision artifact solves the constraints on its exterior domain and never
  materializes the omitted interior as active evolution cells;
- a puncture/trumpet artifact solves the constraints with its puncture topology
  represented in the conformal variables; and
- a smooth-filled artifact explicitly admits interior constraint error,
  unless a separately named global projection is run and the resulting black
  hole is re-diagnosed as a new solution.

The last choice can be a useful experiment, but it must not silently convert a
requested Kerr/COCAL hole into a different constraint-solved object. A future
localized gluing owner would be a separate research method rather than an XCTS
polish flag.

After a smooth fill, the ordinary evolved-spacetime path converts the completed
ADM fields to regular Z4c variables and evolves them with the selected
moving-puncture gauge and Z4c constraint damping. This is a settling evolution,
not a second constraint solve and not a separate hidden preprocessing method.
The desired outcome is that fill error remains horizon-contained or is damped;
outward propagation into the physical exterior is a measured error, not the
mechanism on which the method relies. Acceptance requires comparison across
fill radii/methods and resolution using exterior-only Hamiltonian/momentum/Z4c
norms, apparent-horizon area and physical mass/spin, lapse, and exterior matter
observables.

Coordinate horizon radius is gauge dependent and may shrink or grow while the
physical horizon area converges. A direct-trumpet slice should reduce the large
wormhole-to-trumpet coordinate transition, but cannot replace area, mass, spin,
constraint, and convergence checks.

`spacetime.black_hole_matter` selects one runtime matter owner without adding a
template product:

- `evolve` is ordinary complete-grid RHD/RMHD evolution;
- `bounded` is available for `filled` and `puncture` geometries. Inside the
  strictly nested `horizon.excision_fraction` mask it caps Lorentz factor and
  specific internal energy, optionally caps density, and limits RMHD
  magnetization by adding floor matter while retaining canonical CT face `B`.
  It then recomputes pressure through the selected EoS and converts primitives
  back to conserved variables. Every bound and terminal interior reset is
  counted separately; an interior reset is still a warning; and
- `excision` is inactive-cell matter ownership and requires the matching
  `black_hole_interior = excision` geometry. C2P is
  bypassed inside, active-domain diagnostics and CFL omit those cells, AMR does
  not tag the deep interior, and the one-sided inner flux closes the exterior
  baryon ledger.

The bounded ceilings are
`spacetime.black_hole_max_lorentz_factor`,
`spacetime.black_hole_max_specific_internal_energy`,
`spacetime.black_hole_max_magnetization`, and the disabled-by-default
`spacetime.black_hole_max_density`. They are horizon-contained robustness
controls, not permission to treat exterior atmosphere resets as harmless.

The filled Kerr control uses ordinary advective `one_plus_log`,
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
filled-Kerr gauge. These historical gauge comparisons predate the Cycle 366
shift-upwind sign correction and require remeasurement before ranking gauges.
Gauge selection is independent of interior policy, but excision additionally
requires the selected gauge's characteristics to leave the active domain.

The Cycle 369 autopsy corrects the earlier attribution of a step-three failure
to high-order reconstruction: that control omitted the live gauges and used
frozen lapse/shift. PWC and halved CFL also fail. Stage checks find a spatial
metric that is no longer positive definite near the fill core, before
projection or fluid recovery produces NaNs. A wider fill or live lapse alone
can instead first produce negative cell-centered chi/lapse through signed
high-order interpolation. Neither nodal floors nor a fluid fallback repairs
that geometry contract.

Before the Cycle-371 fill-geometry correction, at CFL `0.9`, MP5/LLF with
both one-plus-log and Gamma-driver reached
`t=1.2 M` at `32^3` and `64^3` without C2P fallback, atmosphere reset, or
interior bounds/reset. Exterior Hamiltonian Linf decreases from `0.01211`
to `0.004577`; the coarse RK4 control also reaches the target cleanly.
Exterior Hamiltonian L2 instead changes from `0.0008103` to `0.0009650`;
the diagnostic's excluded near-horizon stencil layer also shrinks on the
finer grid, so this pair is not a fixed-region convergence measurement.
These historical comparisons use KO `0.1`, not the historical `0.32` above.
They establish short robustness for that earlier fill, not a converged long accretion
solution or a constraint-consistent stationary fill.
These successful fill controls use **vacuum** Z4c evolution with a test fluid:
live lapse/shift, dynamically evolved geometry, and matter backreaction are
three distinct choices. They do not show that stress-energy coupling is needed
to stabilize the fill, nor that frozen gauge is intrinsically invalid. A
stationary exact exterior can use frozen gauge; the arbitrary filled interior
is not a stationary solution of that same system.

The current analytic fill differentiates its lapse/shift/spatial-metric blend
and derives `K_ij` from the stationary ADM identity. The former zero-derivative
and independently weighted-K completion was inconsistent with a fixed
four-metric. The new completion is kinematically consistent, not constraint
satisfying: evolving it with the vacuum equations still changes the interior.
Pointwise derivative/ADM checks must therefore be supplemented by fresh
evolution checks, not inherited from the earlier fill.

Positive-scalar transfers interpolate `log(chi)` and `log(lapse)` with the
selected polynomial weights, then exponentiate. This applies to physical
node/cell/face centering; tensor fields retain their ordinary representation.
Outside-domain finite-difference extensions retain the raw boundary polynomial,
not log extrapolation. Invalid physical sources are not repaired. On the final
Cycle-371 source, SSPRK3 and RK4 reach `t=1.2 M` at CFL 0.9 with zero C2P
fallback, atmosphere reset, or interior intervention, and baryon-budget error
below `4e-16`. The boundary/fill/AMR/restart matrix passes 14/14 and the fresh
Z4c quick gate passes 13/13. The nodal lapse
does reach its selected floor. This repairs the observed negative transferred
lapse, not the arbitrary interior's vacuum-constraint violations.

The focused black-hole gate evolves the filled data at CFL 0.9 without C2P
fallback or atmosphere reset, includes a 16-step moving-puncture-gauge stress,
reloads its physical-ID artifact on a changed Cartesian grid, and exercises an
exact exterior fixed-background control.
Matched full/equatorial Kerr domains at two resolutions and a matched
full/octant Schwarzschild domain agree in minimum lapse and constraint extrema
to roundoff after one step. The gate includes fixed-Kerr and short coupled
excision, incoming-gauge rejection, incompatible physical-ID spin, and
incompatible reflection-plane rejection.

The focused interior-policy comparison uses the same `32^3`, `a/M=0.5`,
CFL-0.9 fixed Kerr exterior. With no repeated horizon solve, bounded full-grid
and excision sustain median `0.309` and `0.305` Mcells/s; excision advances
`3.37 M/s` versus `3.20 M/s` after inactive cells are removed from the CFL
maximum. RHD
and CT-RMHD excision close baryon mass to roundoff, and the RMHD control has
`divB_L1=8.45e-19` with zero recovery/reset. With the finder run every step,
the fixed-excision horizon area and Kerr mass change only `2.53e-5` and
`1.13e-5` relative over three steps (finder tolerance); the evolved smooth
fill changes them by `2.68` and `1.18` percent by `t=0.237 M` while its
coordinate radius changes `1.12` percent. The 16-step direct-trumpet control
ends at `t=1.390 M` with area `50.200 M^2` (`0.13` percent below exact) and
Kerr mass `0.9948 M`, while its coordinate radius grows by 10.1 percent. These
are coarse short-run method controls, not a long-time accuracy ranking.

The fixed-excision gate also seeds a two-level ratio-2 hierarchy, advances the
fine level twice per coarse step, and permits synchronized regridding every
step. The first regrid remakes level 1 from 110,592 to 32,768 cells around the
tracked horizon. After three coarse steps the exterior baryon-ledger residual
is `4.86e-17` relative, the coarse curved-background energy/source residual is
`3.14e-4`, and C2P fallbacks, atmosphere resets, interior bounds, and interior
resets are all zero. Initial hierarchy tagging precedes the step-zero horizon
search and therefore retains ordinary initial tags until a surface exists;
startup still constructs every level's mask before physical projection,
preflight, CFL evaluation, or evolution.
On restart, a saved horizon surface is a search seed, not a current finder
result. Startup refreshes an enabled finder once even when the checkpoint
step is off `horizon.find_int`; ordinary evolution retains that cadence. The
off-cadence excision regression restarts step 1 with `find_int=4`, refreshes
the surface before scalar output, and closes the baryon ledger to `2.56e-16`
with zero recovery/reset.

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
The compressed artifact has a `64^3` base plus a ratio-2 near-hole seed and
resolved support `14.098--25.981 M`; its current on-disk size is 99 MB. The
fixed-Kerr production evolution uses true matter excision and a ratio-4/2
target. Its near-hole box is a seed rather than a fixed complete hierarchy, so
the compact-object metric also resolves the disk after artifact load. A
24-step CFL-0.9 run reaches `t=52.489 M`; adaptive regrids grow the layout from
196,608 to 409,600 cells, the maximum density decreases 3.60 percent, the
horizon baryon flux grows from `3.10e-9` to `4.92e-8`, and the exterior baryon
ledger closes to `1.86e-13`. A temporary diagnostic pass identified seven late
below-floor projections; they are ordinary floor enforcement, not failed-C2P
atmosphere resets. C2P fallback and atmosphere-reset counts are zero. Horizon
area and Kerr mass change by `2.53e-5` and `1.13e-5` relative (finder
tolerance). The corresponding dynamically evolved spacetime still needs Z4c
excision.

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
diagnostic gauge pair, not the selected production default. The earlier coarse
gauge ranking is stale after subsequent numerical changes. Cycle 373 reruns
the `dx=0.5M`, CFL-0.9, 16-step controls with KO strength 0.32: the matched pair
changes mean coordinate horizon radius by 4.99 percent and ends with a
7.42 percent proper-area error; ordinary advective 1+log/Gamma-driver changes
radius by 23.35 percent and ends with a 7.89 percent proper-area error. Both
complete with zero C2P fallback or atmosphere reset, but the unchanged
2-percent final-area regression bound fails. The ordinary-gauge horizon
history is byte-identical under AMReX 26.08 and 26.09. This is an existing
accuracy issue to investigate, not an AMReX-update regression or a validated
current gauge ranking. The ordinary moving-puncture pair remains the supplied
evolution input.

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

The opt-in `horizon_flux` scalar group adds
`black_hole_horizon_baryon_accretion_rate`. It evaluates the Valencia baryon
current on the found radial graph,

```text
F_D^i = D (alpha v^i - beta^i),
dot(M_0,H) = - integral_H F_D^i dS_i,
```

so inward flow is positive. The executor uses the evolved densitized baryon
density `D`, generated metric/current algebra, the generic finest-complete AMR
surface interpolation plan, and reflection parity when the Cartesian domain
omits horizon sectors. This is the instantaneous flux through the coordinate
two-surface on one slice. A rapidly moving horizon requires an explicit
world-tube/surface-velocity term before it can close an exact moving-boundary
baryon budget. The sign and Valencia-current convention follow the
[Einstein Toolkit Outflow documentation](https://einsteintoolkit.org/thornguide/EinsteinAnalysis/Outflow/documentation.html).

This baryon budget is distinct from the black-hole energy budget. Baryon
number crosses the horizon; it is not converted one-for-one into curvature.
Horizon area, Christodoulou mass, and spin respond to the total matter-energy,
angular-momentum, electromagnetic, and gravitational fluxes. Consequently,
accreted rest mass need not equal the increase in horizon mass. A proper
dynamic-excision validation requires both the exterior baryon balance and
convergent horizon area/mass/spin response, together with exterior constraints
and outer matter/radiation fluxes. Fixed-background accretion deliberately
omits this geometric backreaction and is only a test-fluid comparison.

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

`horizon.abort_on_failure` defaults to `1`, preserving fail-fast behavior for
runs that start with a known black hole. Set it to `0` for collapse problems
that intentionally arm the finder before a horizon exists. In that mode a
failed search is logged and retried at `horizon.find_int`; no horizon surface,
mask, or positive horizon scalar is fabricated before a converged search
succeeds. The scalar catalog reports `black_hole_horizon_found=0` before
formation and `1` afterward. Once any surface has been accepted or restored,
a later failed search remains fatal so stale horizon geometry cannot silently
drive diagnostics or refinement.

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

The reproducible simulation-derived accretion cutaway uses

```console
uv run python pyfluid_amrex/tools/run_case.py \
  pyfluid_amrex/inputs/bh_accretion_octant_movie_3d.in --ranks 3
uv run python pyfluid_amrex/tools/plot_bh_accretion_octant_movie.py \
  runs/current/black_hole_disk/bh_accretion_octant_movie_clean
```

The evolution keeps the complete azimuthal domain and only reflects across the
equator. The renderer then displays the positive-coordinate octant as a
cutaway, overlays the live apparent horizon (or exact Kerr fallback) and finest
AMR boxes, and reports the hole diagnostics and matter `|C1|/C0`, `|C2|/C0`.
The input is a fixed
Kerr--Schild `a/M=0.5` control with a test-fluid q-power torus. It now selects
the same true matter-excision, live-horizon, and seed-plus-automatic-AMR
ownership as the maintained full-domain control. Its previous filled-hole
movie remains historical output and must not be cited as validation of the
updated input until that 32-step rendering run is repeated. This remains an
accretion/evolution and visualization control, not evidence for a
self-gravitating massive-disk or PPI endpoint.

The longer maintained control is `bh_accretion_long_excision_3d.in`: disk
`dx=0.5 M`, horizon seed `dx=0.25 M`, outer boundary `64 M`, and full azimuth
with equatorial reflection. It reaches `t=550 M` in 300 CFL-0.9 steps with
zero C2P fallback, atmosphere reset, interior bound, or interior reset. The
final hierarchy has 501,760 cells (11.15 percent of its cell-budget guard).
This fixed-Kerr test-fluid run is not evolved-spacetime excision or a
self-gravitating massive-disk result.

Report disk time as `t/P_c`, with `P_c=2*pi/abs(Omega_peak)` fixed from the
initial unperturbed disk density maximum. For saved Valencia velocity, use
coordinate transport velocity `alpha*v-beta`, not `v` alone:
`Omega_peak=(alpha*v-beta).(axis cross position)/|axis cross position|^2`,
where position is relative to the disk origin. The sampled physical ID gives
`P_c=527.3409577286847 M`, so the completed run covers `1.04297 P_c`. This is
a characteristic orbital clock, not a claim that fluid at every radius has
completed that many orbits. Retain the reference sample, source artifact,
angular velocity, and period in run-local `orbital_normalization.tsv`; the
renderer reads its `period` column automatically, or accepts the explicit
`--characteristic-period` override. Movie frames retain `t/M` as well.

The companion diagnostic plot uses the runtime `multipoles.tsv` and
`scalars.tsv`: `|C1|/C0` decreases from `7.49e-3` to `5.30e-3`, while
`|C2|/C0` reaches `1.73e-4`. The atmosphere-excluded `C0` ends 1.21 percent
below its initial value. Its late upturn accompanies measured net boundary
influx; do not interpret this finite-domain control as isolated disk mass
loss or evidence for PPI growth. The whole-history maximum normalized
baryon-ledger residual is `9.33e-11`; the final checkpoint continuation closes
to roundoff. Render the retained output with

```console
uv run python pyfluid_amrex/tools/plot_bh_accretion_octant_movie.py \
  runs/current/black_hole_disk/bh_accretion_long_excision \
  --output runs/current/black_hole_disk/bh_accretion_long_excision/bh_accretion_long_excision.mp4 \
  --radial-extent 40 --vertical-extent 20 --max-pixels 320 --fps 6
```

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
The fixed-Kerr RHD/RMHD owner supplies inactive matter cells, one-sided
diode fluxes, excision-aware candidate repairs, CT-compatible magnetic
ownership, CFL/diagnostic/AMR masking, and an inner conservation ledger.
The evolved-Z4c fixed-worldtube owner now also supplies its characteristic
guard, one-sided derivatives, masked dissipation and exterior transfers;
short coupled, AMR and restart controls pass. Moving cuts and long-time
exterior convergence remain unvalidated. A found horizon is a prerequisite,
not by itself authorization to discard spacetime cells.
