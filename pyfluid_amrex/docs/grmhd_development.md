# Relativistic HD And MHD Development

## Scope

RHD and RMHD extend the existing AMReX solver through one generated
relativistic fluid-system bundle. They do not introduce another AMR driver,
reconstruction implementation, Riemann interface, time integrator, output
format, or checkpoint path.

`PhysicalSystem` is the stable identity:

| Identifier | Fluid sector | Magnetic terms | Spacetime sector |
| --- | --- | --- | --- |
| `HD` | Newtonian | no | none |
| `MHD` | Newtonian | yes | none |
| `RHD` | Valencia | no | fixed or evolved |
| `RMHD` | Valencia | yes | fixed or evolved |

`SpacetimeEvolution` is the second stable identity. Newtonian `HD/MHD` require
`none`; relativistic `RHD/RMHD` require `fixed` or `evolved`. Fixed Minkowski is
special-relativistic HD/MHD, fixed curved geometry is Cowling evolution, and
evolved geometry schedules the spacetime sector. The metric provider is present
for both relativistic systems. In Cowling mode
it supplies immutable lapse, shift, spatial metric, inverse metric, determinant,
and derivatives. In dynamical mode the spacetime sector evolves the same fields.
The relativistic fluid kernels do not distinguish those ownership modes.

## Reference Implementations

- [Einstein Toolkit AsterX](https://github.com/EinsteinToolkit/AsterX) is the
  primary RHD/RMHD reference for Valencia conserved variables, primitive and
  conserved conversion, fluxes, characteristic-speed estimates, source terms,
  stress-energy, atmosphere handling, and recovery diagnostics. The local
  checkout is `/Users/mrizzo/phys_research/ETK_CarpetX/AsterX_dev/AsterX`.
- [Einstein Toolkit Cottonmouth](https://github.com/EinsteinToolkit/Cottonmouth)
  is the generated BSSN/Z4c reference and an example of turning high-level
  spacetime equations into optimized Cactus kernels.
- SpacetimeX/Z4c is the primary local CarpetX reference for evolved spacetime
  variables, gauge, scheduling, synchronization, and AMR behavior. The local
  checkout is `/Users/mrizzo/phys_research/ETK_CarpetX/SpacetimeX/Z4c`.
- [GRHayL](https://github.com/GRHayL/GRHayL) is the primary independent
  reference for generated RMHD flux/source expressions, EOS integration, and
  the full recovery-method catalog. The local checkout is
  `/Users/mrizzo/phys_research/ETK_Carpet/Cactus/repos/GRHayL/GRHayL`.
- [GRHayLET](https://github.com/GRHayL/GRHayLET) and IllinoisGRMHD provide the
  Einstein Toolkit integration reference for recovery ordering, atmosphere,
  diagnostics, and evolution scheduling.
- [RePrimAnd](https://github.com/wokast/RePrimAnd), developed by Wolfgang
  Kastaun, is an additional modern C++ primitive-recovery and EOS reference.
  The local checkout is
  `/Users/mrizzo/phys_research/ETK_CarpetX/CactusX/repos/RePrimAnd`.

References define equations and validation targets. PyFluid keeps its own
single AMReX ownership, field-schema, code-generation, and validation contracts.

### Local coupled-evolution reference snapshot

The matter-coupling design was checked against these local ETK revisions:

- AsterX `7e70efa181cec5936be375ae8e7c146aaa3d7085`, especially
  `AsterX/AsterX/schedule.ccl` and `AsterX/AsterX/src/tmunu.cxx`;
- CarpetX `375f6dbd85ca7d19680229bf75e41b3be7046239`, especially
  `CarpetX/ODESolvers/src/odesolvers_solve.cxx`,
  `CarpetX/ODESolvers/src/odesolvers_solve_subcycling.cxx`, and
  `CarpetX/TmunuBaseX/schedule.ccl`;
- SpacetimeX `e89fe1d3ee47adaf881f051c502fac61e4a4c0f9`, especially
  `SpacetimeX/Z4c/schedule.ccl` and `SpacetimeX/Z4c/src/z4c_vars.hxx`.

ETK's `ODESolvers` owns one stage loop for all evolved groups. Its stage
post-processing recovers AsterX primitives, clears and rebuilds the
vertex-centered `TmunuBaseX` fields, and then evaluates the fluid and Z4c RHS
groups from that common stage state. `TmunuBaseX` is a useful multi-thorn
aggregation interface, but it is not an additional evolved sector. Z4c
projects the covariant tensor into the standard Eulerian energy, momentum, and
spatial-stress sources.

PyFluid follows the same numerical contract without adopting the Cactus
aggregation storage. The generated RHD/RMHD bundle directly forms the
Eulerian projections from the recovered cell-centered stage state and the
stage metric, then the common centering operator transfers those stage-local
sources to Z4c vertices. Optional output of a covariant stress-energy tensor
must reuse the same generated definition; it must not introduce a second
matter-coupling path.

## Generated Relativistic Fluid Bundle

The symbolic source uses indexed 3+1 objects with explicit variance and checked
Einstein contraction. RHD and RMHD are generated from the same equations;
magnetic contributions are included when the physical-system capability is
enabled.

Generate, in this order:

1. Valencia primitive and conserved layouts plus densitization conventions.
2. Primitive-to-conserved expressions.
3. Conservative fluxes projected onto an arbitrary face direction.
4. Stress-energy projections and geometric source terms.
5. Conserved-to-primitive residuals and analytic Jacobians.
6. Signal-speed bounds and optional characteristic transforms.
7. RMHD magnetic fluxes and CT electric fields from the same ideal-MHD
   stress-energy definition.

Generated expressions own tensor algebra and analytic differentiation. The C++
runtime owns iteration control, bracketing, admissibility decisions, atmosphere
policy, reconstruction, Riemann fan selection, AMR, and scheduling.

## Conserved-To-Primitive Policy

C2P is one ordered policy, not independent evolution branches. Candidate
solvers consume the same conserved state, metric, EOS, tolerances, and result
type. Each candidate returns primitives plus a reasoned status. Shared
admissibility checks enforce finiteness, EOS bounds, positive density/pressure,
and subluminal velocity. The validation corpus additionally recomputes P2C and
checks the full conserved residual; production does not pay for that redundant
P2C evaluation on every recovery.

The measured production ideal-energy RMHD policy is:

1. RePrimAnd, the seed-free bracketed master-root method;
2. Palenzuela1D, an independent bracketed Newton/bisection fallback.

Both residuals and primitive reconstructions are generated from the same
ideal-gas RMHD equations. On the maintained regular corpus, RePrimAnd is about
2.4 times faster than Palenzuela1D and is at least as robust. The fallback is
entered only after RePrimAnd returns a reasoned failure; adding another
candidate does not add work to successful earlier candidates.

The complete implemented candidate catalog is Noble2D, Palenzuela1D,
RePrimAnd, Noble1D entropy, Palenzuela1D entropy, and Font1D. They are not
interchangeable candidates for every state: Noble2D requires a prior primitive
seed, the entropy variants require an evolved entropy variable, and Font1D is a
cold-EOS projection that does not preserve a general ideal-gas energy state.
The ideal-energy production policy therefore allocates none of those optional
inputs. RHD uses its smaller hydrodynamic recovery specialization.
Initial atmosphere, evolved-state admissibility, C2P fallback, and failed-state
repair are separate contracts:

- initial atmosphere is a configured ambient density/thermodynamic floor added
  by initial-data construction before P2C;
- admissibility validates every accepted evolved conserved state and every
  recovered primitive state;
- the ordered C2P policy tries numerical recovery methods without silently
  changing the conserved state;
- exhaustion of every C2P method means the conserved state is invalid for the
  selected EOS. Validation aborts with diagnostics. Any future production
  repair/floor policy is explicit, counted, and outside the C2P solver.

For composition-dependent EOSs, `Y_e` is an evolved conserved composition
field. Temperature is recovered primitive/cache data rather than an independent
conserved variable. An entropy field is allocated only when an entropy-based
recovery/evolution option requires it. The ideal-gas RHD slice allocates none of
these optional fields.

Each recovery result reports the number of attempted methods, winning method,
residual, total iteration count, and terminal status. These are available to
focused diagnostics without adding atomic counters to the production cell
loop. Successful Newtonian runs compile no C2P cascade and pay no relativistic
recovery cost.

`c2p_policy.hpp` is the single source of truth for the shared
method/status/result vocabulary, capability requirements, and ideal-energy
ordering. No method is marked implemented until its residual, iteration, and
admissibility path is validated.

Every recovery method must pass the same two-sided corpus invariant for each
supported EOS and metric regime:

```text
C2P(P2C(primitive)) == primitive
P2C(C2P(conserved)) == conserved
```

The corpus includes deterministic noisy but admissible states spanning density,
pressure/internal energy, velocity, magnetic-field strength and orientation,
metric conditioning, and states near physical bounds. Comparisons use scaled
component residuals and also record the P2C residual returned by the shared C2P
validator. Barotropic conserved states are sampled on their EOS manifold because
energy is not an independent primitive degree of freedom. Newtonian HD/MHD must
pass this same contract to roundoff; iterative RHD/RMHD tolerances are set from
the requested recovery residual and verified against an independent reference.

## Evolution Sectors

The fluid sector owns RHD/RMHD conserved state and evaluates its RHS using a
read-only metric view at the requested stage time. The spacetime sector owns
metric/gauge state and evaluates its RHS using stress-energy at the requested
stage time.

The first implementation uses one synchronized timestep. The interface must
carry stage time explicitly so a later multirate method can interpolate sector
state and advance fluid and spacetime on different clocks without changing
physics kernels. Cowling evolution is the same interface with no spacetime
advance.

All explicit methods use the existing shared RK coefficient tables and stage
algebra. A coupled GRHD/GRMHD stage constructs every active sector at one stage
time, fills its ghosts, builds the cell/vertex coupling views, evaluates every
active RHS, and applies one common stage update. There are no fluid-specific
and spacetime-specific copies of Euler, SSPRK3, or RK4.

AMR level subcycling and physics-sector multirate evolution are separate
policies over this same stage interface. A slower spacetime cadence cannot be
implemented by skipping ordinary-RK spacetime stages while feeding stale
metric or matter data to the fluid. A production multirate policy must provide
the metric at every requested fluid stage time and accumulate a
time-consistent matter source for the spacetime macro-step. Regridding,
checkpointing, and synchronized output occur only when both sector clocks and
the affected AMR levels align. The fully coupled cadence-one method is the
reference against which every slower cadence is converged and timed.

## Vertical Validation

Build the implementation through executable vertical slices:

1. Minkowski RHD primitive-to-conserved-to-primitive round trips and analytic
   Jacobian checks.
2. Minkowski RHD waves and shock tubes through the existing reconstruction,
   Riemann, timestep, AMR, and subcycling machinery.
3. Fixed curved metrics and hydrostatic equilibrium in Cowling mode.
4. Minkowski RMHD round trips, waves, Brio-Wu-style references, and CT divB.
5. Fixed curved-metric RMHD.
6. Evolved spacetime gauge/constraint waves without matter.
7. Coupled RHD, then RMHD, with synchronized sectors.
8. Multirate sector evolution only after synchronized coupling converges and
   conserves as expected.

WIP slices remain executable and are tracked in `review_running_cycle.md` and
the support matrix. Do not add temporary fallbacks or startup prohibitions to
hide incomplete work.

Current fixed-spacetime status:

- ideal RHD and RMHD P2C, directional fluxes, and causal speed bounds are
  generated from the indexed Valencia systems;
- RHD and RMHD use the existing component reconstruction, HLL-family Riemann,
  timestep, AMR/subcycling, output, and checkpoint machinery;
- the RMHD standard gate covers randomized 3+1 identities, zero-field RHD
  reduction, cyclic-axis covariance, two-sided noisy-state recovery, exact
  free streams, second-order entropy-wave convergence, a Balsara-1 numerical
  reference, 2D CT refinement, L2 aligned AMR/subcycling with zero divB, and
  bitwise-identical synchronized restart;
- magnetic C2P uses the ordered RePrimAnd/Palenzuela1D policy. The corpus
  validates every implemented candidate under its capability contract and
  forces the production fallback on a deterministic off-manifold state.
  Exhaustion aborts rather than changing the conserved state;
- fixed Minkowski, Rindler, and isotropic-Schwarzschild metrics supply lapse,
  shift, spatial metric, inverse metric, determinant, and analytic spatial
  derivatives at every RK stage. Valencia geometric sources are generated
  from the contravariant stress-energy tensor and metric derivatives;
- Rindler and isotropic-Schwarzschild hydrostatic equilibria converge under
  refinement. A three-level Rindler control covers subcycling and exact
  checkpoint restart, a variable-determinant Schwarzschild control exercises
  RMHD, and a two-dimensional Rindler AMR+CT+subcycling control preserves
  magnetic divergence.

## Signal-Speed Policy

The production RHD/RMHD path uses generated local causal characteristic bounds,
matching AsterX and GRHayL practice. A hard `[-1, 1]` light-speed fan is safe in
units where `c=1`, but it is not the default: it reduces simulation time per
wall time by approximately `max(|lambda_local|)` and adds the same avoidable
HLL/LLF dissipation when the physical waves are slower than light. It becomes
nearly equivalent only for ultrarelativistic or strongly magnetized states.
The generated discriminant is clamped at zero against roundoff before taking
its square root; noncausal or nonfinite recovered states still fail recovery.
On the controlled RMHD entropy benchmark, whose maximum characteristic speed is
`0.857c`, strict light-speed stepping improves simulation-time/real-time by
`6.5%` because it avoids a C2P speed pass. The measured break-even is about
`0.80c`; local bounds win below it and remain the robust general policy.

Current evolved-spacetime status:

- vacuum Z4c, coupled RHD, and coupled RMHD use the same generated Z4c RHS and
  the same sector-aware Euler, corrector, midpoint, SSPRK3, and RK4 stage
  algebra. Frozen/Cowling and cadence-one coupled evolution are supported;
  physics-sector multirate evolution remains deliberately unimplemented;
- Eulerian matter projections and Valencia metric/source views are rebuilt
  from the common stage state. Deep four-level AMR/subcycling restart is exact
  within `1e-12` for coupled RHD, and a weak coupled RMHD four-level smoke
  conserves baryon mass to roundoff;
- vertex-centered spacetime FillPatch and level construction use one
  fourth-order tensor-product nodal interpolater. This reduced the maintained
  smooth gauge-wave AMR error by roughly two orders of magnitude relative to
  AMReX's bilinear nodal interpolater without the severe 3D cost of a tested
  sixth-order alternative;
- the internal TOV provider solves a one-dimensional areal-radius problem,
  transforms it to isotropic radius, and samples the same physical-coordinate
  view into both fluid and spacetime fields on every AMR level. Single and
  piecewise cold polytropes are supported, with optional hybrid thermal
  pressure for evolution;
- the 3D TOV Hamiltonian L2 norm decreases from `5.99e-5` at N=24 to
  `2.19e-5` at N=48, with surface-limited observed orders of 1.33 and 1.55. A
  short two-level coupled TOV evolution preserves baryon mass to about
  `1e-11`; its AMR constraint norm remains a tracked interface/surface
  validation rather than a claim of long-term stellar equilibrium.

The current initial-data implementation is an internal analytic/radial-table
TOV provider. The physical-coordinate sampling boundary is intended to accept
later COCAL/KEH or other grid data, but no arbitrary-coordinate external-grid
reader or physics-aware interpolation has yet been implemented. Rotating-star,
black-hole, and binary initial data therefore remain future work.

Validation commands are:

```bash
uv run python pyfluid_amrex/tools/validate_fixed_spacetime.py \
  --level standard --force --ranks 3
uv run python pyfluid_amrex/tools/validate_z4c.py \
  --level standard --force --ranks 3
```
