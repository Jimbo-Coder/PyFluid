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

The next implementation boundary is evolved spacetime. The same read-only
stage-time metric interface can receive Cottonmouth-like generated metric
fields without duplicating the RHD/RMHD system. Fixed-spacetime validation is:

```bash
uv run python pyfluid_amrex/tools/validate_fixed_spacetime.py \
  --level standard --force --ranks 3
```
