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
P2C/C2P, fluxes, sources, boundaries, diagnostics, and every RK stage.
`EOSParams.method` is its only runtime identity. HD, MHD, RHD, and RMHD use one
`EOSRuntime` production bundle for `ideal_gas`, `polytrope`,
`hybrid_polytrope`, and cold `compose` plus the configured gamma-law thermal
extension. The launch interfaces do not retain a second EoS selector and adding
an EoS does not add an evolution/reconstruction template product.

Analytic EoS ownership is under `include/pyfluid_amrex/eos/` and `src/eos/`.
Custom polytropic segments and the named Read-fit presets (`sly`, `apr4`, `h4`,
and `ms1`) are assembled there into the same continuous runtime representation.
The current CompOSE model is a one-parameter cold beta-equilibrium barotrope
plus a gamma-law thermal extension. It is not a finite-temperature or
composition-dependent table. That eventual implementation belongs in this
module and must extend the thermodynamic state without introducing a second
dispatch path in initial data, C2P, or evolution kernels.

RMHD recovery accepts that one runtime EoS view. At the canonical C2P entry,
ideal gas selects the algebraically equivalent generated ideal
RePrimAnd/Palenzuela chain once; hybrid, polytropic, and CompOSE closures use
the generic runtime thermodynamic residuals. This measured boundary selection
avoids paying method dispatch at every root iteration without creating a
per-EoS evolution, reconstruction, or factory product.

An initial-data provider normally supplies one self-consistent fluid and
spacetime solution in physical coordinates. A provider advertises those groups
independently so analytic fixed metrics need not fabricate fluid data. The fluid
group is mandatory for every physical system. Storage is selected by
`pyfluid.physics` and `spacetime.evolution`, never by an initialization fallback:
RHD allocates no magnetic state and RMHD allocates magnetic state. Evolved
spacetime owns Z4c stage/RHS storage; fixed spacetime owns only the immutable
canonical Z4c state needed for coupling, AMR, output, and restart.

Newtonian analytic profiles may be reused as shapes for explicit fixed-Minkowski
RHD/RMHD tests after the relativistic provider performs the selected EoS and
P2C conversion. They are not constraint-consistent coupled-GR initial data.
Selecting evolved spacetime therefore requires a provider that supplies a
self-consistent fluid and spacetime solution, such as TOV plus CTT/CTTK/XCTS;
the runtime never silently combines a Newtonian profile with flat or unrelated
spacetime data.

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
component, and one generated catalogued-order topology operator converts between
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

## AMR Indicators

`schema/methods.toml` owns a small runtime AMR-policy catalog. It selects
semantic indicators rather than exposing one independent switch per evolved
component:

- `lohner` preserves the broad flow-study maximum over density, pressure,
  velocity, magnetic structure when present, and stored `chi - 1` when
  present;
- `matter` follows density-supported matter only and ignores cells within ten
  configured density floors;
- `spacetime` follows stored `chi - 1` only and skips fluid primitive recovery;
- `compact_object` combines the spacetime indicator with density structure
  above ten thousand configured floors. The stricter support threshold keeps
  low-density numerical-atmosphere halos from owning compact-object AMR;
  select `matter` explicitly when the low-density flow itself is the target.

The directional maximum of the selected indicators feeds the unchanged AMReX
hierarchy. `compact_object` is the normal stellar GRHD/GRMHD choice. `spacetime`
is appropriate for vacuum compact-object geometry, while `lohner` remains an
explicit shock, instability, and ideal-flow choice. A magnetic/shear science
target should become another small documented policy only after a matched
validation demonstrates its benefit; adding a new evolved or diagnostic field
must never silently make it an AMR driver.

This distinction is supported by production-code evidence rather than a claim
that `chi` is universally sufficient. Vacuum numerical-relativity codes refine
on conformal geometry or truncation error, whereas GRMHD merger work uses
objective-specific density, magnetic-field, or shear criteria and explicitly
guards against refining low-density unphysical regions. Broad all-field
tagging is therefore a useful stress/flow policy, not a realistic universal
full-GRMHD default.

## Curvilinear Relativity Boundary

Fixed-spacetime RHD/RMHD can reuse the coordinate-independent generated
Valencia stress/flux algebra, coordinate metric provider, and mapped face
measures. They cannot reuse the Newtonian curvilinear momentum source or the
Cartesian determinant/partial-metric source unchanged. The first curvilinear
relativistic target therefore lowers Valencia through an explicit
reference-metric owner: scalar conserved fluxes use physical cell/face
measures, mixed momentum fluxes add the generated covariant-basis connection,
and spacetime sources use reference-covariant metric derivatives. Isotropic
pressure is separated before the basis correction so constant pressure is
balanced by construction. The current runtime slice is single-level spherical
RHD with active axes `r` or `r theta` and analytic fixed Minkowski or centered
isotropic-Schwarzschild spacetime. Schwarzschild cells must remain outside
r=M/2. `validate_reference_metric_rhd.py` covers static balance, radial
expansion, Schwarzschild hydrostatic convergence and dimensional consistency.
AMR, RMHD, stored fixed spacetime and full spherical 3D remain unsupported
in this analytic fixed-metric slice. The separate evolved fCCZ4 path below
now supplies stored reference-metric GRHD geometry. These are
evolution restrictions, not restrictions on all curvilinear initial-data solves.

Curvilinear AMR needs an explicit coarse-fine contract for reference-basis
momentum, geometric source balance, volume/face measures and stage-time metric
data. The existing Cartesian FillPatch/reflux/average-down structure should
remain the driver; removing configuration rejections alone cannot supply the
missing numerical contract. Magnetic evolution additionally needs compatible
edge EMFs and face transfer. Keep these limitations visible until the intended
operators and conservation/convergence checks exist.

Evolved Z4c in cylindrical or spherical coordinates is not obtained by adding
Jacobian factors to the Cartesian kernels. The current formulation uses
Cartesian partial derivatives and the Cartesian conformal-determinant
condition. A robust extension requires an explicitly derived reference-metric
formulation, generated tensor-basis terms, parity/origin regularization, and
matching nodal AMR transfer and boundary contracts. SphericalNR's published
BSSN/fCCZ4 equations provide a reference, not a drop-in relabeling of our
Z4c system. A separately generated reference-metric spacetime system is a
valid extension; it need not rewrite the working Cartesian Z4c system.
Formulation-specific catalog entries may declare state-field dependencies,
basis conventions, equation groups, and supported gauges. They reference the
shared GF catalog rather than declaring fields again. The actual tensor
equations remain readable symbolic sources, not opaque TOML expressions.

For this implementation, keep the tensor equations and coordinate geometry
as high-level symbolic owners. Derive basis-rescaled expressions and genuine
symmetry reductions there, retaining all physical vector/tensor components;
do not write separate component equations for each executable dimension.
Emit geometry, constraints, gauge, and RHS groups with shared subexpressions
and private compiled owners under the existing compile-cost conventions.
Coordinate selection must not multiply the reconstruction/EoS/gauge template
product. Review symbolic identities and independent physical tests rather
than repeatedly expanding generated C++ into agent context. Reuse tensor
algebra, physical metric/stress-energy interfaces, and diagnostics where the
mathematics is shared; do not force different formulation states to have
identical component semantics.

Parallel ownership belongs in AMReX orchestration, not generated equations.
Ordinary local stencils can use distributed boxes; origin/pole exchange and
any angular filter need explicit communication contracts. Do not require
complete angular shells on every rank. Before choosing filtering, coarsening,
or a multipatch alternative, compare physical-time throughput at matched
error, conservation, and retained angular modes. Start with spherical 1D and
axisymmetric 2D physical controls; these do not establish full-3D polar
scaling. See the curvilinear-evolution references in `../references.md`.

The generated implementation now starts in
`codegen/pyfluid_codegen/spacetime_geometry.py`. Its input is a symmetric
tensor in the orthonormal reference basis and coordinate partial derivatives
of those components. It emits coordinate-basis metric derivatives and
orthonormal reference-covariant first/second derivatives. The second derivative
includes the connection acting on the derivative index. The same source emits
Cartesian, cylindrical and spherical kernels; all three tensor directions
remain present in reduced-dimensional inputs. These are temporary pointwise
arrays (`metric_jet.hpp`), not new evolved grid functions.

The generated connection difference implements SphericalNR Eq. (21) using
reference-covariant metric derivatives. Shared Christoffel/Ricci algebra lives
in `metric_differential.py`; Cartesian Z4c consumes the same expressions as
before that extraction. The new `spacetime` generation family is separate
from reconstruction and equation-of-state kernels.

`reference_metric_evolution.py` now owns the first evolution group: the
conformal metric and `chi=exp(-4 phi)` coordinate-time RHS, including shift
Lie derivatives and the conformal divergence. Inputs use reference-covariant
derivatives in a local orthonormal basis. Physical `K` is explicit; it must
not be confused with Cartesian Z4c's `K_hat`. The kernel retains the
conformal-determinant gradient in the shift divergence rather than silently
assuming its constraint is satisfied. Checks recover the ADM physical-metric
equation and frame covariance with a non-diagonal metric and nonzero shift.
The formulation kernel has no coordinate-family selection. Chart derivatives,
patch maps, and exchange stay outside it; the infrastructure conventions record
the future mixed-chart contract.

The complete fCCZ4 equation owner is now `fccz4_equations.py`, with separate
geometry, RHS, gauge and constraint generated leaves. Select it explicitly
with `spacetime.evolution=evolved` and `spacetime.formulation=fccz4`; Cartesian
Z4c remains the default. `fccz4_runtime.cpp` connects these equations to the
shared conformal field catalog. Its adapters decode physical `K=K_hat+2Theta`
and interpret the connection slots as reference Lambda. Restart metadata
rejects a change of formulation. Runtime uses the existing integrated
Gamma-driver; the separate auxiliary-B point kernel does not add a B field
to evolution storage.

The stored-metric GRHD bridge uses reference-covariant metric derivatives,
physical angular CFL factors, and the same FV face/volume measures in both
high-order divergence and admissibility fallback. Diagnostic symmetry factors
belong to `metric_reductions.hpp`, where cell totals and boundary-face budgets
share one normalization. TOV metric sampling uses a shared C2 quintic Hermite
interpolant with radial derivatives obtained from the TOV equations; matter
sampling remains linear on the radial table.

Current evidence is deliberately bounded. Cartesian gauge-wave propagation
retains fourth order for both formulations, and the existing fixed-hierarchy
Cartesian fCCZ4 AMR/subcycling/restart control passes. The opt-in command
`validate_z4c.py --only spherical_gauge_wave --evolution-formulation fccz4
--ranks 3 --force` checks all conformal fields in a regular 3D spherical shell
against an independent Cartesian plane-wave solution, transformed into the
local frame. All nonzero-error fields converge at orders 3.84--3.94 on the
finest pair. This compares a fixed physical interior away from open radial
and polar boundaries, with a full periodic azimuth; it does not test polar
regularity. A short spherical Minkowski evolution is stationary to roundoff.
The coupled TOV-shell mass ledger closes to roundoff, but its global evolved
constraint errors do not yet show clean convergence.

Origin/pole regularity and curvilinear radiative boundaries remain missing.
Current fCCZ4 domains therefore use regular charts away from these coordinate
singularities and extrapolation boundaries. Curved RMHD/CT and mixed-chart
multipatch evolution remain unsupported. Evolved fCCZ4 can enter the common
AMR driver. A fixed partial 3D spherical hierarchy now conserves mass to
roundoff with subcycling after aligning register and RHS flux representations.
This isolated control does not certify adaptive evolution: an unbuffered
vacuum interface still develops inadmissible energy after reflux. A four-cell
buffer passes a bounded placement control but does not resolve that defect.
A reduced `r phi` field jet keeps
theta derivatives of basis factors; it is not a general closed equatorial
physical model. Full-star and star-plus-disk production readiness needs those
boundary, magnetic, AMR and long-duration physical validations.

`spacetime.derivative_order` selects 2/4/6/8/10 for Cartesian Z4c and
regular-chart fCCZ4 (default 4). The same selection controls bulk derivatives,
connection completion, constraints, KO dissipation and stencil/ghost reach.
Physical extrapolation uses order+2 nodes when the domain has sufficient
extent. Spacetime AMR prolongation remains a separate `prolongation_order`
selection. Both use the numerical catalog rather than independent lists.
This is spatial stencil support, not tenth-order full GRMHD accuracy: RK time
order, matter centering/transfer, finite-volume reconstruction and the chosen
boundary algorithm can set lower limits. The existing approximate Sommerfeld
operator retains its independent second-order boundary gradients. The common
elliptic ID operators retain second-order interiors; their boundary
interpolation and provider materialization orders are separate settings.
Specialized horizon, radiation and surface-observable geometry gathers still
use fourth-order derivatives. The evolution apparent-horizon tracker requires
3D Cartesian geometry; shared field metadata does not imply every diagnostic
supports every chart or follows the bulk derivative-order selection.

The current readiness boundary is:

| Capability | Evidence | Still required |
| --- | --- | --- |
| Generated regular-chart fCCZ4 | Independent exact equations and frame covariance; regular-shell gauge propagation at orders 2/4/6/8/10 | Broader gauge and long-time nonlinear evolutions |
| Spherical `r`, `r theta`, `r phi`, `r theta phi` projections | Coupled TOV-shell one-step differences below 5e-12 | Long-time symmetry preservation and physically justified reductions |
| Regular-shell GRHD | Fixed-metric balance/convergence and evolved-metric mass budgets | Full stellar surface and temperature evolution |
| Spherical AMR | Partial 3D shell reflux conserves mass; buffered radial controls pass | Admissibility at unbuffered vacuum interfaces; adaptive convergence and restart |
| Origin and polar axis | Explicitly rejected by the regular-chart runtime | Tensor parity, regularity and matching ghost/transfer operators |
| Curved RMHD/CT | Pointwise geometry and Cartesian magnetic controls only | Curved face/edge evolution, divergence control and AMR synchronization |
| Mixed-chart multipatch evolution | Equation/geometry/transfer ownership defined | Tensor exchange and conservative fluid/CT interfaces |

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
admissibility checks enforce finiteness, EOS bounds, positive density,
nonnegative pressure,
and subluminal velocity. The validation corpus additionally recomputes P2C and
checks the full conserved residual; production does not pay for that redundant
P2C evaluation on every recovery.

The production energy-evolving RMHD policy is:

1. RePrimAnd, the seed-free bracketed master-root method;
2. Palenzuela1D, an independent bracketed Newton/bisection fallback.

The generator owns the EoS-independent recovery kinematics. Each trial calls
the same runtime closure used by P2C, fluxes, and sources for pressure and
enthalpy; primitive construction therefore cannot silently revert to ideal
gas. On the maintained ideal-gas corpus, RePrimAnd is about 2.4 times faster
than Palenzuela1D and is at least as robust. The fallback is entered only after
RePrimAnd returns a reasoned failure; adding another candidate does not add
work to successful earlier candidates.

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
  by initial-data construction before P2C. CTT, CTTK, and XCTS build their
  matter sources from this same floored primitive state, so the constraint
  solve and the state handed to evolution contain identical atmosphere matter;
- admissibility validates every accepted evolved conserved state and every
  recovered primitive state;
- the ordered C2P policy tries numerical recovery methods without silently
  changing the conserved state;
- the runtime atmosphere owner projects under-density states and can attempt
  atmosphere replacement after C2P exhaustion. Failed-recovery replacements
  increment reset counters and produce runtime warnings; bounded black-hole
  interior replacements have their own counter. RMHD replacement retains
  finite magnetic fields rather than repairing them. Recovery still reports
  failure if the replacement cannot be recovered. These are changes to the
  conserved state, not additional root-solving candidates.

The RHD candidate flux limiter must close its correction mask before installing
fluxes: changing a shared face can invalidate the neighboring candidate. It
rechecks mixed-flux candidates using a double-buffered, monotonically growing
binary mask and a global changed-cell reduction. The low-order candidate is
terminal; if that candidate is itself inadmissible, the existing projection
policy still applies. This prevents one-pass mask propagation errors without
turning a local energy projection into an unreported conservation claim.

Initial-data flooring and runtime atmosphere projection use the same EoS
thermodynamic closure, but they are distinct numerical policies. Initial data
constructs the ambient state; runtime projection repairs an evolved state that
falls outside the admissible C2P range and is not a second background
superposition. A healthy stellar evolution should apply runtime repair rarely
after initialization. Radially graded initial atmospheres therefore require a
separate initial profile and a compatible, lower recovery floor; an ID-only
profile that the next C2P pass overwrites is not supported. A true zero-density
vacuum needs an explicit zero-state extension of the fluid numerics because
the positive-density Valencia C2P map is singular there; setting configured
floors to zero does not create a supported vacuum evolution.

### Vacuum/Atmosphere Exterior Policy

Keep one exterior-state policy selecting vacuum or atmosphere, with the same
fluid recovery and conservative flux-correction owners. Start with GRHD;
nonzero magnetic fields require a separate physical exterior closure. This
policy is not `spacetime.matter_coupling=vacuum`, which only disables fluid
backreaction on evolved geometry.

Recognize canonical zero matter before density-dividing EoS/C2P expressions.
It is a valid state, not a failed recovery or a reason to freeze the metric.
Partially occupied finite-volume cells still carry matter, and face fluxes
must allow fluid to expand into empty cells. For nonzero matter, retain the
ordered recovery policy. A last-resort state replacement, when permitted,
uses the selected exterior construction but remains a counted/warned reset;
do not label failed nonzero matter as ordinary vacuum. Share that final
construction rather than adding parallel fallback ladders or a template axis.

The production GRHD implementation follows these independently testable
stages:

1. make P2C, flux, stress-energy, and geometric-source algebra finite and
   exactly zero for canonical zero matter without changing positive-density
   equations;
2. give canonical zero conserved matter an explicit successful recovery
   result before EoS or density-dividing C2P work, while keeping invalid
   nonzero states and counted resets distinct;
3. carry one exterior policy through initial data, constraint sources,
   materialization, boundaries, AMR transfer, output, and restart metadata;
4. validate one-sided fluid-vacuum interfaces, expansion into vacuum,
   stationary stellar surfaces, and AMR conservation before enabling the
   runtime option.

All four stages are implemented for RHD behind
`pyfluid.relativistic_exterior = vacuum`; atmosphere remains the default. The
policy is a compact runtime value, not a template product. It is immutable
restart/physical-ID identity, and legacy physical-ID artifacts without the key
mean atmosphere. Exact zero conserved matter recovers successfully to one
canonical zero primitive, zero--zero faces have exactly zero flux, and ordinary
Riemann fluxes transport matter across one-sided matter--vacuum faces.

The maintained validation boundary covers uniform expansion, dynamically
regridded subcycled AMR expansion at CFL 0.9, bitwise restart, policy-mismatch
rejection, physical-ID save/load and changed-grid materialization, and a
buffered two-level 3D octant TOV surface through two CFL-0.9 steps. These cases
preserve exact vacuum without C2P fallback or reset and conserve baryon mass to
roundoff. A longer deliberately coarse TOV stress can still create tiny
low-density tail states after more steps; long atmosphere-free stellar
evolution therefore remains a convergence/stress target rather than a claim
of this gate.

Atmosphere behavior remains unchanged. The initial-data constraint sources,
saved state, materializer, and evolution must agree on the exterior policy;
removing an atmosphere only after solving constraints is not that contract.
Nonzero magnetic fields remain an atmosphere or explicit electromagnetic-
exterior problem rather than GRHD matter vacuum, so configuration rejects
vacuum for RMHD.

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
- fixed TOV data use that same Cowling mode. The maintained two-level
  SSPRK3/subcycling matrix keeps the stored metric fixed to roundoff, limits the
  four-step maximum-density change to about `5.3e-3`, and closes the finite-box
  rest-mass ledger at `4e-15`. The corresponding evolved-spacetime control
  changes maximum density by about `1.2e-3`, keeps Hamiltonian L2 below `1e-4`,
  and closes the rest-mass ledger to roundoff.
- the TOV contract is unmagnetized equilibrium. An imposed field is useful for
  RMHD stress tests but is not a self-consistent magnetized star; that requires
  the future rotating/magnetized equilibrium provider coupled to the constraint
  solver.

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
- the internal `TovSolution` integrates the one-dimensional stellar equations
  adaptively in log specific enthalpy, reaches the exact zero-pressure surface,
  transforms to isotropic radius, and samples the same physical-coordinate
  profile into fluid and spacetime fields on every AMR level. The same module
  directly produces EoS sequences and standard stellar observables without an
  evolution; see `docs/tov.md`;
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
CTT, hybrid CTTK, and XCTS residuals, Jacobian actions, and nonlinear
iterations all live in this initial-data layer; evolution consumes only the
materialized canonical fields.

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
grid reader, rotating-fluid equilibrium solve, black-hole solve, or
physics-aware interpolation is yet implemented, so rotating-star, black-hole,
and binary data remain future work.

Validation commands are:

```bash
uv run python pyfluid_amrex/tools/validate_fixed_spacetime.py \
  --level standard --force --ranks 3
uv run python pyfluid_amrex/tools/validate_z4c.py \
  --level standard --force --ranks 3
```

## Stellar surface thermal accuracy

A cold equilibrium star should be assessed using its thermal excess relative to
its initialized state, separating stellar interior, surface, and atmosphere.
The current hybrid/Compose cold-plus-gamma-law closure does not define a unique
physical temperature. `thermal_entropy_proxy` is solver-authored output:
`K_th = (p - p_cold(rho))/rho^gamma_th`; it is not evolved entropy. For the
single hybrid polytrope with equal cold and thermal gammas,
`K_th/K_cold = p_th/p_cold` without a second primitive recovery.
`tools/report_stellar_thermal.py` reports signed values and changes for fixed
coordinate shells of a single-level Cartesian TOV run. It requires t=0 output,
optionally from a matching initial-only run, and rejects other EOS/mesh cases.
Its volume means are not proper-mass averages or tracked fluid elements.

Dual-energy methods evolve an additional internal-energy or entropy quantity
and choose/synchronize thermodynamics where total-energy recovery loses thermal
accuracy; shocks must still produce physical entropy. See
[Cosmos++ (Anninos et al. 2005)](https://arxiv.org/abs/astro-ph/0509254).
Entropy-based flux limiting is a separate discretization strategy; a recent
[hybrid-EOS BAM study (Doulis 2026)](https://arxiv.org/abs/2608.16945)
addresses artificial surface heating with that approach. Neither method is
implemented here by the diagnostic addition, and neither should substitute for
correct conservative-stage accounting.

Cycle 406's two short 3D Cowling probes already show initial surface thermal
excess and mixed subsequent heating/cooling. They do not establish thermal
convergence or long-term stability. The atmosphere pressure is not the cold
pressure at its prescribed density, so large exterior ratios are not evidence
of stellar heating. Before selecting a new method, compare initialization-only
thermal error, several dynamical times of TOV evolution, spatial resolution,
atmosphere sensitivity, and EOS-consistent thermal excess, then extend to RNS.
Retain shock-heating validation to ensure any thermal switch preserves genuine
shock heating. A future finite-temperature EOS test should additionally record
EOS-consistent temperature and composition, not an inferred gamma-law value.

## Single-object readiness before multipatch and binaries

Use a supported-combination matrix, not an exhaustive Cartesian product of
all input options. Distinguish initial-data construction, handoff, fixed-metric
evolution, coupled evolution, AMR/restart and long-duration physical accuracy.
A spherical XCTS solve does not establish spherical Z4c evolution. A successful
single-object spacetime test does not establish matter or magnetic coupling.

The current priority is a reliable full-GR single-patch backbone: establish
long-duration TOV and isolated-BH convergence, validate star-plus-disk handoff
and evolution, and close the scoped precision/backend gates in
`precision_and_metal.md`. Known Newtonian self-gravity AMR/restart failures
remain tracked separately under the user's full-GR priority. Extend the
reference-metric evolution/AMR operators with their own physical gates. Work on
unsupported coordinate formulations should not silently expand the support
matrix, and support for every optional algorithm is not a necessary prerequisite
for a well-defined Cartesian binary calculation.

NS-plus-disk initial data already supports test-fluid and coupled disk gravity
and XCTS materialization into Cartesian evolution. The standard torus gate
includes only two steps for the coupled stellar case. This establishes a
construction/handoff control, not a stable magnetized NS-plus-disk production
configuration. A Cycle 410 64^3, CFL 0.2 extension fails at t=3.71 with
extrapolated spacetime boundaries; changing only to Sommerfeld reaches t=5
without recovery repairs. The example now selects that radiative condition
explicitly. EOS choices, rotation, atmosphere, magnetization and long-term
metric response each require relevant evidence.

Thornburg-style angular patches are a useful later geometry target. They avoid
angular coordinate poles, but do not by themselves supply conservative fluid
interface fluxes, CT synchronization, tensor-basis transfer, interpolation at
stage times, or unique ownership of overlapping cells. Those contracts must be
solved before interpreting successful patch communication as physical accuracy.

For binary-oriented grid work, Cycle431 recommends a Cartesian near zone with
regular angular outer patches as the first geometry target. It can retain
nodal Cartesian spacetime and avoids a spherical coordinate origin. Begin
with vacuum wave transmission across stationary patch interfaces, then add
the conservative matter and magnetic interface contracts. BlackHoles@Home's
overlapping vacuum-grid layout is a design reference, not those contracts.

A cell-centered spacetime option is a separate route for full-sphere and
reduced-symmetry evolution. It changes discrete sampling, derivatives,
boundaries, centering transfers and AMR/checkpoint semantics, not the continuum
Einstein equations. Moving radial/polar samples off singular points still
requires regular ghost connectivity, tensor parity and angular CFL treatment.
Hydro stays cell-centered and face/edge CT need not change centering. Neither
this option nor completed origin treatment is a prerequisite for a supported
Cartesian binary calculation.

Particle work should start from a physical purpose. Passive tracers could
record surface heating and ejecta histories without backreaction. Radiation
packets need transport and matter coupling; SPH is another fluid discretization;
kinetic particles need a kinetic model and conservative coupling to any fluid
region. AMReX's storage/redistribution machinery does not supply those models.
