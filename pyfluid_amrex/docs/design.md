# PyFluid AMReX Design

## Purpose And Status

PyFluid AMReX is the production finite-volume backend for Newtonian and
relativistic HD/MHD. AMReX manages hierarchy, parallelism, and I/O; the same
infrastructure supports flat spacetime, fixed curved metrics, and coupled Z4c
evolution. Physical equations and numerical methods can therefore evolve
without rewriting hierarchy, output, or validation infrastructure.

Supported means a path has a focused validation gate and uses the standard
finite-volume contracts below. A parsed option or an input file alone does not
make a path supported.

| Area | Current status |
| --- | --- |
| Single-patch Cartesian HD/MHD | Production numerical path |
| Cartesian SRHD/SRMHD and fixed-curved GRHD/GRMHD | Production for focused validated combinations |
| Coupled evolved-Z4c matter | Validated vertical slice; long stellar evolution remains an explicit gate |
| TOV initial data | Direct 3D Cartesian and portable spherical-1D solve/materialization paths validated |
| XCTS rotating-star initial data | 3D Cartesian path implemented for the documented EoS, rotation-law, and symmetry boundaries; the reduced `r theta` scalar elliptic foundation is validated, while the toroidal shift and coupled axisymmetric solve remain open |
| Reusable physical-ID and sequence artifacts | Same-chart and schema-v2 cross-dimension/cross-coordinate handoff implemented through one materialization interface |
| Native 1D, 2D, 3D | Production build modes |
| AMR, reflux, average-down, restart | Production infrastructure |
| AMR subcycling | Production for validated Cartesian cases |
| CT with explicit resistivity | Production for validated Cartesian cases |
| Hydro implicit viscosity | Available; use its focused validation |
| Cartesian periodic implicit CT resistivity | Available in 2D/3D on one AMR level; use its focused validation |
| Multilevel or curvilinear implicit CT resistivity | Unsupported by the AMReX 25.11 `MLCurlCurl` backend |
| Homogeneous Cartesian multipatch | Hydro-only finite-volume path; use multipatch gate |
| Cylindrical/spherical single patch | Coordinate-general kernel path; validate the exact method combination |
| Mixed Cartesian-curvilinear shell layouts | Geometry/maps and sampling only; seam evolution is experimental |
| Gravity and moving solids | Isolated feature paths; validate before combining with other experimental work |

Unsupported combinations abort during configuration. The solver must not silently
change reconstruction, Riemann solver, time integrator, geometry, or physics.

## Layering

The backend has three layers.

1. Infrastructure is stable and rarely changed:
   AMReX lifecycle, level allocation, FillPatch, regridding, flux registers,
   restriction, checkpoints, output, profiling, scalar reductions, and test
   helpers.

2. Physics and numerics own equations:
   EOS, initial data, reconstruction, Riemann solvers, RHS/source terms,
   diffusion, CT, gravity, mesh geometry, and solid geometry. They expose
   small APIs and do not independently manage AMR storage or output layouts.

3. Orchestration wires infrastructure and physics:
   AmrSolver owns level state and calls physics kernels through stage-aware
   interfaces. Scripts call shared run/report/validation helpers rather than
   duplicating solver setup.

Keep these boundaries. A new physics feature should not invent another output
writer, scalar format, AMR hierarchy, or source-stage loop.

## EOS And Code Units

HD, MHD, and RHD use the same generated ideal-gas, cold-polytrope, and hybrid
thermal-polytrope closures. A polytropic run may contain up to eight density
segments. The input supplies the first `K`, every density break, and each later
adiabatic index; the solver derives later `K` values and internal-energy
offsets so pressure and cold specific energy remain continuous.

Custom EOS inputs are code-unit quantities. In `P = K rho^Gamma`, each
`pyfluid.polytropic_density_breaks` entry has the same code units as `rho`, and
`pyfluid.polytropic_k` has units `P_unit / rho_unit^Gamma`. Consequently a
change of density or pressure unit requires transforming both the breaks and
the leading `K`; the dimensionless adiabatic indices do not change. Relativistic
runs additionally use `c = G = 1`.

`pyfluid.eos_preset` selects a named cold piecewise-polytrope fit. `custom`
uses the run-supplied code-unit segments. `sly`, `apr4`, `h4`, and `ms1` use
the common crust and three-piece core fits from
[Read et al. (2009)](https://arxiv.org/abs/0812.2163); the EoS
module explicitly converts the published CGS coefficients into
`G = c = M_sun = 1` once during configuration. Named presets and custom
segments produce the same immutable `EOSParams` value consumed by all runtime
kernels, so there is no preset-specific evolution branch. Future tabulated
EoS support belongs behind this same module boundary and must declare its unit
system explicitly.

RMHD evolution currently accepts only `ideal_gas`. Cold and hybrid
polytropes remain available to the supported nonmagnetized relativistic and
stellar paths; adding them to RMHD requires a consistent magnetic C2P, signal-
speed, atmosphere, and evolution validation boundary.

## Initial-Data Solve And Handoff

Initial-data solving and evolution are separate owners. A solver produces
portable cell-centred physical fluid primitives and nodal 3+1 ADM geometry;
the common materializer constructs the target hierarchy's conserved fluid and
Z4c state. `initial_data.solve_only=1` terminates after preflight and artifact
write. This mode neither selects nor instantiates an evolution reconstruction,
Riemann, or timestep algorithm beyond the existing bundle used to enter the
shared initialization lifecycle.

The schema-v2 artifact describes its source chart at runtime. It does not add
coordinate or dimension template products. Exact hierarchy caches remain a
lossless same-layout acceleration. A different target dimension, coordinate,
resolution, or physical extent samples the portable physical fields, transforms
catalog-declared tensors through orthonormal Cartesian components, completes
the physical exterior, and then runs the normal target preflight. Evolution
remains three-dimensional Cartesian for GRHD/GRMHD in the current production
contract.

Spherical one-dimensional TOV is the validated reduced solve. The uniform
cell-centred spherical `r theta` scalar elliptic operator now has manufactured
origin, axis/equator, Dirichlet, Robin, refinement, and MPI-decomposition
coverage. Circular axisymmetric XCTS still needs its physical azimuthal
spin-1 shift operator and source/tensor reconstruction; that symmetry-reduced
operator is preferable to instantiating an unused general three-vector solve.
A zero-spin TOV equivalence gate follows before nonzero uniform rotation.

## Public Interfaces And Compilation

The executable-facing solver API is intentionally small: `amr_solver.hpp`
exposes the AMR run function directly. The concrete `AmrSolver` class and its
base are private backend machinery under `src/amr/`; code outside the AMR
implementation must not include those headers or depend on their storage
layout. `AmrSolverBase` owns the hierarchy,
state storage, FillPatch, reflux, restriction, output, diagnostics, and restart
lifecycle. The private derived solver owns per-stage RHS, reconstruction,
Riemann, diffusion, CT, and time-integration template instantiations. A narrow
set of hierarchy-repair and gravity-diagnostic hooks connects the two; those
hooks run only during hierarchy changes or diagnostics, never inside cell
kernels.

The source tree and CMake groups express the same ownership rather than
pretending that a static library automatically reduces template rebuilds:

- `src/driver/`: executable and stable run entry points;
- `src/framework/`: configuration, canonical fields, checkpoints, sector
  storage, and profiling;
- `src/amr/`: hierarchy lifecycle, synchronization, diagnostics reductions,
  and output orchestration;
- `src/numerics/`: compiled reconstruction/flux dispatch, centering transfer,
  CT registers, and implicit operators;
- `src/initial_data/`: providers, materialization, elliptic solves, and TOV;
- `src/relativity/`: C2P/cache dispatch, metric coupling, and Z4c runtime;
- `src/kernel_bundles/`: explicit ownership of unique state-layout/CT/EOS
  specializations;
- `tests/`, `benchmarks/`, and `tools/cpp/`: non-production executables.

The local full-release preset uses `ccache` as a compiler launcher. This is a
machine-level acceleration and does not change generated source, compiler
flags, linked code, or supported methods. Time representative builds with
`tools/benchmark_compile.py`; it records the target, parallelism, elapsed time,
cache deltas, revision, and dirty state in `runs/current/compile_history.tsv`
with a bounded recent mirror. Cold compile time and cache-hit time are distinct
metrics and must not be compared as if they measured the same operation.

Reconstruction, Riemann, EOS, and cell-flux kernels remain inline templates
where AMReX CPU/GPU lambdas require their definitions. Moving those headers into
nominal libraries would not isolate rebuilds and could prevent device inlining.
The useful compiled boundary is an evolution-kernel bundle. Its key contains
only properties that change template instantiation: component count, CT
topology, and EOS dispatch policy. Runtime dispatch selects a fully specialized
bundle outside cell loops, and that bundle calls the shared numerical
algorithms. Physical systems with the same compile-time signature reuse one
bundle; system identity remains explicit runtime data and is never inferred
from component count.

The evolution lifecycle follows the same boundary. It is compiled once per
state/EOS/CT bundle and receives one immutable operation table selected before
the time loop. Timestep, RK, diffusion, CT, subcycling, FillPatch, and regrid
entries point directly to the existing specialized member templates. Indirect
selection therefore occurs only at stage or coarse-step boundaries, never
inside AMReX cell loops, and output/diagnostic/regrid orchestration is not
instantiated for every runtime method combination.

Compiler dependency files verify the boundary: only the lightweight driver and
the sources under `src/kernel_bundles/` include the complete derived solver.
Thus an equation or reconstruction edit rebuilds kernel owners, not hierarchy,
output, checkpoint, metric, or multipatch sources. A shared
numerical-interface edit still correctly rebuilds every consumer.

Every physical system, including future systems, must supply one coherent
physics contract with:

1. a conserved/primitive state layout and field requirements;
2. primitive-to-conserved and conserved-to-primitive operations;
3. directional physical fluxes and maximum signal speeds;
4. optional characteristic transforms used by shared reconstruction;
5. source, parabolic, and CT hooks that are absent unless the system requires
   them.

AMR, reconstruction policies, Riemann fan selection, time integrators,
diagnostics, output, and checkpointing consume this contract. They are not
reimplemented per physical system. System identity must be explicit in this
contract; future code must not infer RHD, MHD, or RMHD solely from the number
of state components.

`PhysicalSystem` is the single stable identifier: `HD`, `MHD`, `RHD`, and
`RMHD`. Its capability table states whether the fluid is relativistic,
magnetized, owns a spacetime sector, and supports CT. Component layouts remain
separate traits because an identifier is not a state-vector size.

Evolution scheduling is sector based. All four systems have a fluid sector.
RHD and RMHD additionally consume a spacetime metric. `SpacetimeEvolution` is
`none`, `fixed`, or `evolved`: Newtonian systems require `none`, relativistic
systems require `fixed` or `evolved`. A fixed metric/Cowling run
uses the relativistic fluid bundle with one immutable canonical metric field
set materialized from an analytic or sampled initial-data provider; a
dynamical spacetime run schedules both sectors. Fixed analytic data are
resampled exactly on hierarchy changes and allocate no evolution time levels.
Magnetism augments the fluid sector and
does not become a third independently scheduled sector. Fluid kernels consume
one metric-provider interface whether its fields are fixed or evolved, allowing
a future multirate driver to advance fluid and spacetime on different clocks.
For evolved spacetime, matter coupling is independent of storage/evolution
ownership: `coupled` feeds fluid stress-energy to Z4c, while `vacuum` evolves
Z4c without fluid backreaction. Neither changes the meaning of fixed/Cowling.

When a new system needs a compile-time signature not already present, add one
kernel owner and one registry entry. If its signature matches an existing
bundle, reuse that entry rather than compiling a nominally system-specific
copy. Changing a genuinely shared reconstruction algorithm correctly rebuilds
every bundle that instantiates it. Do not add virtual calls inside cell loops or
duplicate the AMR driver to obtain compile isolation.

## State And Grid Functions

`schema/fields.toml` is the sole declaration source for named grid functions.
It owns field and component names, mathematical labels, tensor shape,
centering, symmetry, diagnostic kind, evolution role, requirements, checkpoint
ownership, and output ownership. `field_catalog.py` is the strict loader,
model, and validator; it is not a second declaration table. The generator
emits public C++ component IDs, runtime descriptors, plotting JSON, and the
field tables in the equation reference. `fields.hpp` defines only C++
descriptor types and the AMReX-facing query API. Evolution, checkpointing,
output, diagnostics, and plotting consume generated views of the same schema.

`codegen/pyfluid_codegen/diagnostic_catalog.py` distinguishes runtime scalar
reductions from field-output and postprocessed quantities. Runtime reductions
are written to `scalars.tsv`; postprocessing reads those values and requested
GF data rather than maintaining another physical reduction implementation.
Diagnostic names and labels are generated into C++ and JSON. Adding a plotted
GF label or scalar column in an independent plotting table is not allowed.

`codegen/pyfluid_codegen/equation_catalog.py` indexes the high-level physical
systems built by `physical_system.py`. The same symbolic conservation laws and
EOS objects feed optimized C++ emission and the generated
`docs/equations.md`/`docs/equations.tex` reference. Generated component
expansions are review artifacts, not separately maintained equations.

The physical state has three evolution roles:

- Main state: conserved HD variables and MHD variables when active.
- Evolution auxiliary: fields required or cached by the solver, such as
  face-centered CT magnetic fields or gravity data.
- Non-backreacting extra: conserved tracers such as passive dye.

Output has separate semantics:

- Stored state: already-evolved data written directly.
- Mixed auxiliary: can use live solver storage or be computed at output time.
- Pure auxiliary: computed only when requested for output.

Do not make output-only fields solver state. Do not give individual output
formats their own field names or centering conventions.

Hydro allocates no magnetic state. Vector components are runtime directions:
mom1..mom3 and b1..b3; their physical interpretation comes from mesh.axes.

## Coordinates And Geometry

The canonical coordinate orders are:

- Cartesian: x y z
- Cylindrical: r phi z
- Spherical: r theta phi

A lower-dimensional run chooses an ordered subset through mesh.axes. Geometry
is a provider of coordinate maps, physical locations, Jacobians, metric
information, physical vector conversion, and finite-volume divergence data.
Physics kernels receive geometry data; they do not hand-code coordinate-specific
component algebra.

Coordinate-independent expressions belong in the SymPy code generator whenever
that reduces algebra duplication or makes a metric/source term less error-prone.
Runtime code dispatches the selected geometry and applies the generated kernel.
Cartesian fast paths are acceptable only when they remove mathematically trivial
metric work without creating a second physical implementation.

A mixed patch layout has three distinct concepts:

- Patch layout: logical patch maps and context extents.
- Owned domain: cells evolved exactly once.
- Physical union boundary: faces whose opposite side is outside the fluid
  domain.

Overlap provides ghost/interpolation context. It is not a duplicated evolved
domain unless a conservative overset reconciliation method is implemented and
validated.

## Finite Volume And AMR

Every fluid RK evaluation follows the same order; hierarchy synchronization
occurs after an accepted level step:

1. Prepare the stage state, metric coupling, and physical/coarse-fine ghosts.
2. Reconstruct primitive states and solve face Riemann problems.
3. Compute conservative flux divergence and CT EMFs when active.
4. Add stage-local source terms through the source interface.
5. Combine stages with the selected time integrator.
6. Reflux coarse-fine mismatches, average down covered state, synchronize
   canonical CT face fields into cell state, and apply accepted-state checks.

AMR uses the standard finite-volume contract:

- FillPatch/prolongation fills fine ghosts and initializes new fine data.
- Reflux corrects coarse cells next to coarse-fine interfaces.
- Average-down replaces covered coarse state with fine averages.
- CT additionally averages face fields down and refluxes edge EMFs before
  synchronizing cell-centered magnetic fields.

These are different operations. Do not replace reflux with average-down, or
treat interpolation as a conservation correction.

Coarse-fine state transfer has one method contract. `amr.state_interp` selects
the spatial interpolater used for new levels, remade levels, ordinary
FillPatch, subcycled FillPatch, and coarse-state repair seeds. Coordinate
family, configured refinement ratio, and subcycling must not silently replace it with a
different algorithm. A supported interpolater must accept the complete active
`IntVect` refinement ratio and preserve the appropriate cell measure; reject
an unsupported combination before allocating levels. Newly allocated and
ghost state must be completely initialized by that contract and the physical
boundary operator. Nonfinite values are errors, not input to a neighbor-copy
repair. Linear and quartic conservative interpolation preserve closed-system
totals when adaptive boxes move. Coupled linear conservative interpolation is
the robust production default: AMReX's base kernel shares its linear slope
limiter across the full conserved state so interpolation does not independently
break component relations. That linear coupling alone does not preserve the
nonlinear relativistic admissible set, and the stock `protected_interp` hook is
not invoked by the FillPatch/level-remake paths used here. For Cartesian,
analytic fixed-spacetime, CT-off RHD/RMHD, PyFluid therefore post-limits each
parent family's interpolated deviations with one C2P-admissible factor shared
by every child and every conserved component. A conservative base then retains
the parent average exactly; an already admissible interpolation is unchanged,
while nonfinite output or a non-admissible coarse anchor is a transfer error
rather than an atmosphere repair. Curvilinear basis transfer, stored/evolved-
spacetime, and CT-on relativistic transfer need their metric/face-field-aware
owner and are not covered by this cell-state limiter. The component-limited
linear method is an explicit alternative. Conservative quartic interpolation
is a smooth-data choice and
is not positivity preserving. Non-conservative quadratic and quartic
interpolation remain explicit experimental choices.

AMReX can request a fine region that contains only part of one coarse parent's
child family. A family-coupled limiter therefore expands to the complete
sibling-aligned region, performs base interpolation and admissibility limiting
in private scratch, and copies only the requested subset. Limiting only the
visible children makes the answer FAB/ghost-decomposition dependent and does
not preserve the stated family-average contract.

Following reflux, average-down, face-field synchronization, or regrid, the
last hierarchy owner refreshes metric coupling and performs relativistic
acceptance before exposing the state. Newly created and remade fine levels
then keep current and old-time storage consistent with that accepted state.

RHD applies one source-aware conservative flux limiter to the complete
forward-Euler candidate at each RK stage. It blends each shared face between
the selected high-order flux and PWC LLF only as much as required to preserve
positive densitized rest mass and the Valencia admissibility margin.

RMHD separately predicts each concrete RK candidate and replaces shared faces
adjacent to cells that fail the configured candidate/recovery checks with the
selected terminal low-order flux, normally PWC. The guard is a robustness
policy, not an invariant-domain or nonoscillation proof. If accepted-state C2P
still exhausts its recovery paths, the atmosphere reset is counted and warned
as a distinct last resort. For either conservative correction, the final
shared-face flux is the single value consumed by the RHS, boundary ledger, and
reflux registers; accepted-state atmosphere projection is not a replacement
for that conservation step.

Advanced RMHD HLLC/HLLD uses one dedicated compiled owner rather than
instantiating the heavy fan algebra in every generic flux owner. HLLD solves
the five-wave flat-SRMHD fan and nonlinear total-pressure closure in a local
face-normal orthonormal frame, then transforms the sampled flux back to the
coordinate Valencia frame. An unordered, inadmissible, nonfinite, or
nonconverged fan uses the explicit counted HLL limit. HLLD is the production
choice; HLLC remains a contact/algebra control. This validation boundary is not
an all-regime positivity or entropy proof, especially for strongly magnetized
intermediate states and the coupled CT/source discretization.
The reported HLL fallback value counts solver evaluations, including discarded
candidate trials and shared-face/FAB duplication, rather than unique final
faces; zero is meaningful, while a positive magnitude is diagnostic only.

The production AMR tagger follows HD/MHD physical state structure with physical
gradients and cell sizes. Passive scalars never drive tags. max_level,
min_feature_length, and cell budgets are safety limits; the flow metric should
normally end refinement. Curvilinear origin guards are topology/resource guards,
not a substitute for the metric.

Coordinate-equivalence validation holds a common seeded hierarchy after
initialization, so it measures the coordinate operators rather than different
numbers of CFL-limited regrids. Dynamic curvilinear regridding is checked by a
separate freestream/regrid gate.

## Sources, Diffusion, And CT

RHS assembly is stage-local and source-agnostic:

- The driver prepares read-only stage data.
- Flux/divergence kernels assemble the conservative RHS.
- The source interface adds any compatible source RHS.
- Features requiring conservative coarse-fine correction have an explicit AMR
  synchronization hook.

Viscosity is a conservative flux divergence. Explicit and implicit updates use
the same physical face-flux kernel; explicit updates apply that divergence
directly, while implicit updates register the converged-time face flux without
applying it twice. Both enter the AMR flux registers and reflux coarse-fine
mismatches. CT resistivity is the corresponding edge-EMF contribution. These
explicit operators are generated from the same coordinate-independent physics
across every supported dimension and coordinate family. The implicit CT update
uses the same generated resistive EMF and CT curl update, but its linear solve
is limited to one-level Cartesian 2D/3D by AMReX 25.11 `MLCurlCurl`.
Fluid viscosity and diffusion are currently Newtonian-only; configuration
rejects RHD/RMHD combinations rather than silently applying those operators.

CT uses face-centered magnetic fields and edge-centered EMFs. Face `B` is the
canonical magnetic state, and the matching face-divergence operator is the
only divb diagnostic. Cell-centered `B` must be synchronized from the
corresponding canonical face state before reconstruction and physical
acceptance.

The current implementation accumulates EMFs with the same RK weights used by
the fluid update and applies one face curl update after the multistage fluid
step. Final face-to-cell synchronization plus accepted-state recovery is a
necessary backstop, but this remains a step-boundary CT path: later fluid RK
stages do not yet reconstruct from a same-stage evolved face field. Full
stage-coupled CT must advance canonical face `B` at every RK stage and use that
same field for reconstruction, C2P, edge-EMF reflux, and face average-down.

Fixed gravity is a stage-local source with cached potential/acceleration data.
Periodic self-gravity solves the multilevel Poisson problem at synchronized
hierarchy times and uses the Athena-style conservative source coupling. On a
fixed single-level hierarchy its total fluid-plus-potential energy closes to
roundoff. Regridding changes the discrete gravitational energy and is therefore
validated as a convergent hierarchy-transfer error, not hidden by a global
energy correction. Self-gravity with time subcycling is experimental until it
shows both convergent agreement with a matched non-subcycled hierarchy and a
useful simulation-time throughput gain.

## Input, Output, And Diagnostics

The configuration registry owns registered method and AMR defaults, parsing,
accepted string options, descriptions, and restart mutability. `config.hpp`
retains structural parsing and combination validation. Inputs are reproducible
configurations; the C++ executable writes the canonical merged AMReX parameter
table to `run.in` for wrapped and raw runs alike.

The supported configuration path is compiled defaults plus one AMReX input
file. Input values replace defaults. The production executable accepts exactly
that one file; trailing parameter mutations are rejected. Normal launchers
expose execution and reporting controls only, while validation tools generate a
concrete input file before launch. Unused input keys always abort, so the
accepted parameter surface remains explicit rather than accumulating
compatibility aliases.

Each run has one folder. The only field roots are:

    <run_dir>/
      scalars.tsv
      amr_levels.tsv
      stateGF/
      auxGF/
      checkpoints/
      run.in
      run.log
      report.tsv

Scalars are AMReX runtime reductions. Python reads them for reports; it does not
replace them with a separate core-total computation. Generic postprocessing,
physics-specific postprocessing, and movies consume the same run directory.

## Code Generation

codegen/ owns high-level symbolic definitions and emits checked C++ headers.
Generated headers are an implementation product, not a second source of truth.
When adding vector/tensor physics:

1. Express the coordinate-independent equation in the generator.
2. Generate coordinate/dimension-specific kernels.
3. Call the generated kernel through the common physics interface.
4. Validate Cartesian equivalence and the intended curvilinear case.

Reconstruction and Riemann solvers stay handwritten numerical algorithms unless
generation clearly reduces a real algebraic risk.

For RHD/RMHD, the first generated consumer establishes the indexed symbolic
API. Add spatial/spacetime index variance, metric contraction, and 3+1
projection only as required by that consumer; do not predeclare unused tensor
ranks. The intended first vertical slice is generated Valencia primitive and
conserved variables, both conversion directions, one directional flux, and its
numeric validation. This slice defines the physical-system bundle before the
full source and characteristic algebra is added.

The detailed implementation order, reference codes, C2P policy, sector
scheduling contract, and validation ladder are maintained in
`docs/grmhd_development.md`.

## Extension Checklist

Before declaring a new feature complete:

1. Declare named state or auxiliary GFs in `schema/fields.toml`; do not add a
   second field table in C++ or Python.
2. Declare supported method identity in `schema/methods.toml` when applicable,
   and keep the numerical formula in its labeled C++ or symbolic owner.
3. Extend the runtime configuration registry and combination validator, then
   regenerate and verify derived catalog artifacts.
4. Use existing RHS, source, AMR, output, and checkpoint interfaces.
5. Add a focused analytic, reference, or invariance validation.
6. Run scalar validation and compare feature on/off where meaningful.
7. Run the standard suite and the performance benchmark if a hot path changed.
8. Document the exact support boundary in the README and this file.

Do not introduce another fallback implementation merely to keep an unsupported
combination running. Reject it clearly until it has one correct implementation
and a test.

During active feature development, do not add short-lived startup guards or
fallbacks merely to hide an incomplete WIP branch. Record its status in the
running cycle and support matrix, keep building the intended implementation,
and test each usable vertical slice. Permanent configuration rejection is for a
design that is genuinely unsupported, not for code expected to be completed in
the next development steps.
