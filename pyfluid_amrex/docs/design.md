# PyFluid AMReX Design

## Purpose And Status

PyFluid AMReX is the production Newtonian HD/MHD solver. It is a finite-volume
code with AMReX-managed hierarchy, parallelism, and I/O. The implementation is
organized so that physical equations and numerical methods can evolve without
rewriting hierarchy, output, or validation infrastructure.

Supported means a path has a focused validation gate and uses the standard
finite-volume contracts below. A parsed option or an input file alone does not
make a path supported.

| Area | Current status |
| --- | --- |
| Single-patch Cartesian HD/MHD | Production numerical path |
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

## State And Grid Functions

fields.hpp is the sole schema for grid functions. It owns field name,
components, centering, diagnostic kind, evolution role, and requirements.
Writers and plotting tools consume that schema.

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

Every accepted stage follows the same order:

1. Fill physical and coarse-fine ghosts.
2. Reconstruct primitive states and solve face Riemann problems.
3. Compute conservative flux divergence and CT EMFs when active.
4. Add stage-local source terms through the source interface.
5. Combine stages with the selected time integrator.
6. Reflux coarse-fine flux mismatches and average down covered coarse state.

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
totals when adaptive boxes move. Conservative quartic interpolation is the
higher-order production default; non-conservative
quadratic and quartic interpolation remain explicit experimental choices.

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

CT uses face-centered magnetic fields and edge-centered EMFs. The matching
face-divergence operator is the only divb diagnostic. Cell-centered B is
synchronized from faces for reconstruction and Riemann solves.

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

## Extension Checklist

Before declaring a new feature complete:

1. Add or extend state and auxiliary fields in fields.hpp.
2. Define configuration validation in config.hpp.
3. Use existing RHS, source, AMR, output, and checkpoint interfaces.
4. Add a focused analytic, reference, or invariance validation.
5. Run scalar validation and compare feature on/off where meaningful.
6. Run the standard suite and the performance benchmark if a hot path changed.
7. Document the exact support boundary in the README and this file.

Do not introduce another fallback implementation merely to keep an unsupported
combination running. Reject it clearly until it has one correct implementation
and a test.
