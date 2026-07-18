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
for both relativistic systems. In Cowling mode it supplies immutable lapse,
shift, spatial metric, inverse metric, determinant, and derivatives. Analytic
and sampled fixed metrics both materialize one canonical immutable Z4c state on
the AMR hierarchy. Analytic data are resampled exactly when a level is created
or remade; sampled providers use their interpolation policy there. Fixed
spacetime allocates no old, stage, RHS, or dense-output state and receives no
spacetime update. In dynamical mode the spacetime sector evolves the same
canonical fields.
The relativistic fluid kernels do not distinguish those ownership modes.
`spacetime.matter_coupling = vacuum` is the orthogonal evolved-spacetime
control: gravity evolves without fluid backreaction while the fluid still
responds to the metric. It is not a second Cowling/fixed branch.

## EOS And Initial-Data Ownership

The selected EOS is one immutable run property consumed by initialization,
P2C/C2P, fluxes, sources, boundaries, diagnostics, and every RK stage. HD, MHD,
and RHD support `ideal_gas`, `polytrope`, and `hybrid_polytrope`. RMHD currently
supports `ideal_gas` only because its maintained RePrimAnd/Palenzuela recovery
residuals are generated for that closure; unsupported RMHD EOS choices are
rejected before allocation rather than silently using another EOS.

An initial-data provider normally supplies one self-consistent fluid and
spacetime solution in physical coordinates. A provider advertises those groups
independently so analytic fixed metrics need not fabricate fluid data. The fluid
group is mandatory for every physical system. Storage is selected by
`pyfluid.physics` and `spacetime.evolution`, never by an initialization fallback:
RHD allocates no magnetic state and RMHD allocates magnetic state. Evolved
spacetime owns Z4c stage/RHS storage; fixed spacetime owns only the immutable
canonical Z4c state needed for coupling, AMR, output, and restart.

Active optional groups may select provider data or one mathematically valid
neutral value:

| Parameter | Values | Neutral value |
| --- | --- | --- |
| `initial_data.magnetic` | `id`, `zero`, `none` | zero magnetic vector |
| `initial_data.spacetime` | `id`, `flat`, `none` | flat geometry and curvature |
| `initial_data.lapse` | `id`, `one`, `none` | unit lapse |
| `initial_data.shift` | `id`, `zero`, `none` | zero shift |

`none` is valid only when the selected physical system does not allocate that
group. The defaults are `id`, preserving a provider's coupled solution. Gauge
choices remain independent runtime evolution policies; they do not rewrite the
initial data.

Constraint and equilibrium solvers produce one canonical solution on their
natural layout. The field schema determines the evolution centering of every
component, and one generated order-2/4/6/8 topology operator converts between
cell, vertex, face, and edge layouts. The hierarchy adapter then performs the
topology-appropriate coarse-fine synchronization. This keeps staggering at the
initial-data/evolution boundary instead of coupling CTT, XCTS, TOV, or future
KEH solvers to AMReX evolution storage.

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
- [Aurrekoetxea, Clough, and Lim, "CTTK: A new method to solve the initial
  data constraints in numerical relativity"](https://arxiv.org/abs/2207.03125)
  is the reference for the CTTK alternative to conventional CTT. It chooses
  the conformal factor, obtains the spatially varying mean curvature
  algebraically from the Hamiltonian constraint, and iterates that result
  with a conformal-vector momentum solve. Its Chombo multigrid implementation
  and AMR convergence tests are useful validation references, but the paper
  does not prescribe cell- versus vertex-centered storage.

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
- after each relativistic RK stage, a successfully recovered primitive below
  the configured density or pressure floor is projected to the explicit
  atmosphere state before the next RHS evaluation. Magnetic fields are retained
  by RMHD atmosphere projection. C2P exhaustion, nonfinite data, and invalid
  states above the atmosphere floor still abort with diagnostics; the atmosphere
  policy does not turn arbitrary recovery failure into a fallback solution.

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

The first production GR/RHD problem ladder is:

1. Long-lived stable TOV in Cowling mode, then with evolved spacetime; compare
   baryon mass, central-density oscillations, radial-mode frequency, and
   convergent Hamiltonian/momentum constraints.
2. Vacuum Schwarzschild/puncture evolution to validate black-hole gauge and
   boundary behavior independently of fluid recovery.
3. Stationary Michel/Bondi accretion on a fixed Schwarzschild background,
   using its analytic transonic solution to validate curved-spacetime GRHD
   fluxes, sources, and boundaries.
4. Unstable-star migration or collapse only after the stable-star and
   stationary-accretion gates pass. Expensive GRMHD tests follow after the RHD
   hot path is profiled and optimized.

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
- fixed TOV data use that same Cowling mode. A four-level
  SSPRK3/subcycling control changes the stored metric by only `2.3e-16`, changes
  maximum density by `2.4e-6` over four coarse steps, and restarts exactly within
  `1e-12`. The corresponding evolved-spacetime control changes maximum density
  by about `9.3e-4` over its shorter CFL-limited interval.

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

Initial-data construction and evolution meet only at the physical-coordinate
provider and canonical field schema. `initial_data/elliptic.hpp` is the first
solver-side service: it accepts an AMR hierarchy plus solution/RHS views and
performs composite AMReX MLMG scalar or conformal-vector solves without owning
evolution state. Physical boundaries use coefficient fields in the single
coordinate-derivative form `A f + B d_d f = C`; periodicity comes from the
hierarchy geometry. The scalar operator supports general spatially varying
Robin data. Coefficient-defined Dirichlet data are lowered to AMReX's native
form for both operators; genuine Robin data for `MLTensorOp` are rejected
because its tensor cross-term boundary fill does not consistently implement
Robin conditions. Generated multilevel controls cover a general scalar Robin
problem, the nonlinear time-symmetric CTT Hamiltonian equation, the CTT
momentum operator, and recovery of isotropic TOV conformal data. Centered fields
with `f = f_inf + amplitude/r^p` use the standard amplitude-independent
condition `p(f-f_inf) + r d_r f = 0`; the TOV conformal correction uses its
`p=1`, `f_inf=0` instance rather than sampled finite-radius boundary data.
Later XCTS residuals, Jacobian actions, and nonlinear iteration belong in this
same initial-data layer; evolution continues to consume only the resulting
provider.

CTTK is a formulation option inside this same layer, not another
materialization path. It is particularly relevant when the requested physical
matter profile should remain fixed: conventional conformal rescaling can make
the physical source reconstructed after the solve differ from the source that
was specified. CTTK instead fixes the conformal factor and solves algebraically
for a spatially varying mean curvature, then iterates the momentum solve; its
hybrid form retains an elliptic conformal-factor correction for non-matter
sources such as Bowen-York data. This does not replace CTT/XCTS generally:
pure CTTK requires a nonnegative algebraic Hamiltonian radicand, couples the
Hamiltonian and momentum sectors even for zero initial momentum density, and
does not by itself construct quasiequilibrium lapse and shift.

The paper does not resolve the centering choice. PyFluid keeps finite-volume
fluid state as cell averages and never solves a duplicate vertex-centered
hydro system. The production target is instead one staggered solve on the
evolution hierarchy: CTT, CTTK, and XCTS spacetime unknowns live directly on
Z4c vertices, while fluid state remains on cells and CT magnetic state remains
on faces. Matter-to-vertex and metric-to-cell projections are stage-local
coupling views, not a post-solve initial-data remap. The current cell-centered
CTT solve plus schema-driven transfer remains a validation control until the
native nodal solve matches its AMR and boundary coverage, after which it is
removed rather than retained as a second production implementation.

AMReX provides a composite `MLNodeLaplacian` for the scalar nodal equations.
The conformal vector Laplacian is lowered to scalar nodal Poisson equations
through its Helmholtz-style decomposition, generated from the same high-level
momentum operator. The initial-data backend must supply the same public
`A f + B d_d f = C` boundary contract to nodal and cell operators; homogeneous
Dirichlet substitution is a validation control, not a replacement for the
asymptotic Robin condition.

Small analytic or elliptic problems may execute in process. Expensive
KEH/COCAL, black-hole, or binary solves may execute as separate programs and
write an initial-data artifact, but must materialize through the same provider
boundary. The current physical provider backends are fixed analytic spacetime
and the internal radial-table TOV solution. No arbitrary-coordinate external
grid reader, nonlinear XCTS solve, or physics-aware interpolation is yet
implemented, so rotating-star, black-hole, and binary data remain future work.

Validation commands are:

```bash
uv run python pyfluid_amrex/tools/validate_fixed_spacetime.py \
  --level standard --force --ranks 3
uv run python pyfluid_amrex/tools/validate_z4c.py \
  --level standard --force --ranks 3
```
