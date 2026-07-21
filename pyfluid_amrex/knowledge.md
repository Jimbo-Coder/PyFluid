# PyFluid AMReX Working Knowledge

This is the compact, maintained state snapshot for an agent entering the AMReX
backend after a context reset. It is a router and status record, not a second
design specification. Read `../AGENTS.md`, this file,
`references.md`, and the tail of `review_running_cycle.md` before changing the
backend. When this file disagrees with current source or a newer continuity
cycle, verify the source and update this snapshot.

Last reconciled: 2026-07-21, branch `main`, for the integration state recorded
in Cycle 202. That state was assembled from the extensive AMReX, Z4c,
relativistic initial-data, diagnostic, validation, and documentation work based
on `d78a0acc6a27`; use the eventual Cycle 202 commit rather than that older base
when reproducing the baseline.

## Production Boundary

- `pyfluid_amrex/` is the production C++/AMReX implementation. Python/JAX and
  Python/Numba are references unless a request explicitly includes them.
- One coordinate-independent physical implementation serves every executable
  dimension. Dimension branches are for AMReX topology, storage, centering, or
  an explicitly documented unsupported feature.
- Fluid conserved state is cell-centered. CT magnetic state is face-centered.
  Evolved Z4c state is vertex-centered. Stage, RHS, coupling, and transfer
  arrays are scratch, not checkpointed physical fields.
- The input file is the simulation configuration. Runners execute and report;
  they do not override simulation parameters.
- Production AMR uses the selected state interpolater consistently for level
  creation, remake, FillPatch, subcycling, and repair seeds. Fluid state is
  refluxed and averaged down; CT uses edge-EMF reflux and face average-down;
  nodal spacetime uses its nodal synchronization contract and is not refluxed.

## Canonical Ownership

- Runtime configuration contract: `include/pyfluid_amrex/config.hpp`.
  Parsing, supported-value catalogs, validation, and derived host policy:
  `src/framework/config.cpp`, `config_options.hpp`, and `config_runtime.cpp`.
- Current grid-function declaration source: data-only `schema/fields.toml`.
  `codegen/pyfluid_codegen/field_catalog.py` strictly loads and validates it,
  then exposes immutable models and derived views. Generated IDs, descriptors,
  labels, centerings, output, and plotting views consume that validated model.
  See `docs/catalog_schema.md`.
- Requested state and auxiliary GF snapshots normally share one resolved GF
  output cadence: an omitted auxiliary write interval inherits
  `pyfluid.plot_int`, and
  the resolved auxiliary compute cadence equals that output cadence. Only an
  explicit compute override separates them. Explicit auxiliary write and
  compute overrides, including cached-only operation, remain available;
  scalars and checkpoints remain independent. Each writer constructs one
  private ordered auxiliary layout per centering from generated catalog
  descriptors; names, component IDs, counts, and fill dispatch therefore do
  not form a second local registry.
- Critical catalog identity should live in versioned, data-only TOML rather
  than executable declaration tuples. Python remains the strict loader/model,
  tensor expander, symbolic implementation, and C++/JSON/document emitter. A
  canonical GF is any named field that is evolved, retained between operations,
  selectable for output, or checkpoint eligible; it must exist in the field
  schema before production code can request it. Unnamed stage/RHS/elliptic work
  arrays are scratch, not GFs. Runtime code may implement an exhaustive
  generated-ID fill operation, but may not introduce a raw-string field registry.
- Runtime and postprocessed diagnostic schema:
  `codegen/pyfluid_codegen/diagnostic_catalog.py`. Runtime reductions own
  discrete budgets; postprocessing owns analytic comparisons and derived
  output views.
- High-level equations and generated kernels: `codegen/pyfluid_codegen/` and
  `codegen/generate_kernels.py`. Generated C++ headers are outputs, not edit
  targets. The browsable generated equation catalog is `docs/equations.md` and
  `docs/equations.tex`.
- AMR hierarchy and lifecycle: `src/amr/`. Numerical factories and transfer:
  `src/numerics/`. Initial-data providers and materialization:
  `src/initial_data/`. Relativistic recovery, sources, signal speeds, and Z4c
  runtime: `src/relativity/`. Point flux ownership: `src/flux_kernels/`.
- Large host orchestration belongs in compiled `.cpp` owners. Header-only code
  is reserved for small value types and pointwise CPU/GPU kernels whose
  inlining is part of the numerical implementation. Runtime selection happens
  once on the host, outside cell loops.

## Current Validation Boundary

The latest recorded evidence is not one monolithic green result:

- Cycle 175 was the last recorded uninterrupted full suite: 46/46 gates. It
  predates substantial relativistic, diagnostic, transfer, and atmosphere
  work and is historical evidence, not validation of the present dirty tree.
- Cycle 192 recorded all production dimensions, architecture, generated
  physics, focused RHD/RMHD/TOV checks, the complete quick suite, and fresh
  runtime performance ratios of 0.93--0.99 of the frozen baselines.
- Cycle 193 changed the partial-AMR CTT matter handoff afterward. It recorded
  architecture, generated-kernel consistency, all production dimensions, the
  six-check focused CTT gate, and the three-check evolved TOV/subcycling gate.
  A requested three-rank rerun was not executable, and Cycle 193 did not record
  a complete quick or standard suite.
- Cycle 200 subsequently moved GF declarations to `schema/fields.toml` and
  removed dead flux-dispatch templates without changing active kernels. It
  recorded generated-kernel consistency, architecture, all production
  dimensions, the complete quick suite, and a three-rank WENO-Z/HLLC runtime
  comparison. This is the newest broad validation of the current dirty tree;
  it is still not a standard or full suite.
- Cycle 202 integrates the accumulated tree, cadence/catalog output cleanup,
  and validation retiering. Architecture, generated physics (including the
  atmosphere profiles), all production dimensions, both focused auxiliary
  cadence cases, the `2.64 s` quick gate, and all 42 standard gates pass. The
  roughly 90-second exhaustive multipatch executable is preserved in full and
  was not rerun by standard. Consult the end of Cycle 202 for the post-commit
  compile and runtime baseline rather than inferring those from this gate.

Never collapse compile, focused validation, smoke, standard/full suite,
performance, and MPI-rank evidence into one claim.

## Open Numerical And Physics Boundaries

- Partial-AMR CTT initial data are finite and convergent after the common
  matter-handoff repair, but the normalized coarse-fine Hamiltonian band
  remains roughly constant under the recorded refinement. This is an exposed
  second-order multilevel-elliptic/fourth-order pointwise-diagnostic mismatch,
  not a completed high-order coarse-fine derivative closure.
- Rotating relativistic equilibrium/RNS matter data are not implemented. Do
  not imitate rotation by adding azimuthal velocity to spherical TOV data.
- Fixed-spacetime and evolved curvilinear relativistic evolution remain
  pending. Evolved curvilinear Z4c requires a generated reference-metric
  formulation, parity/origin regularity, nodal AMR transfer, and matching
  boundaries; it is not a Jacobian-only change.
- The radiation pipeline has surface interpolation, Weyl/Psi4 generation,
  multipoles, restart-aware histories, and postprocessing controls. Honest
  extraction-radius convergence toward null infinity still needs a compact
  isolated radiative source and multiple physical radii.
- Generic spherical accretion, fluid/EM energy-flux, and magnetic-flux
  diagnostics remain pending. Flat/power-law initial atmospheres and the shared
  allocation-free physical-coordinate geometry provider are implemented; the
  continuity log records their focused validation boundary.

## AthenaK Reference Boundary

- A local AthenaK source tree is available at
  `/Users/mrizzo/phys_research/athenak/athenak`, with CPU and MPI executables at
  `build/src/athena` and `build_mpi/src/athena`. The source tree is not itself
  a Git checkout, but representative source, test, and build files were
  byte-matched on 2026-07-21 to AthenaK commit
  `5f1993109bcb2e5d588ba41b4efc897408e9959a` from
  `https://github.com/IAS-Astrophysics/athenak.git`. Recheck provenance before
  publishing or after the local tree changes.
- AthenaK is a valuable independent reference for Newtonian, SR, fixed-GR,
  dynamical-GRMHD, AMR, CT-MHD, Z4c, wave, shock-tube, puncture, and radiation
  algorithms and test setups. It is not PyFluid's source of truth and its
  numerical output is a code-to-code reference, not an analytic or exact
  solution.
- For comparisons, first match equations, EoS/units, initial data, domain,
  boundaries, resolution, reconstruction/Riemann method, CFL, time integrator,
  output time, and AMR policy. Start with a uniform-grid control, compare
  physical fields/diagnostics on a common sampling, then add AMR. Record both
  revisions, dirty states, build configurations, inputs, rank counts, and
  output paths.
- Keep AthenaK use read-only unless a request explicitly includes changes to
  that checkout. Prefer an independent comparison over copying implementation.
  If code is adapted, review and preserve the BSD-3-Clause attribution and
  keep PyFluid's generated catalogs and ownership rules intact.
- The 2026-07-21 optimized CPU build-artifact comparison found the same broad
  optimization class in both projects (`-O3 -DNDEBUG -mcpu=native`) but very
  different generated-code volume. AthenaK's one runtime-dimensional binary
  linked about 13.5 MiB across 144 application objects. PyFluid's three
  `AMREX_SPACEDIM` binaries linked about 165.2 MiB; eight reconstruction-owned
  flux bundles contributed about 133.3 MiB. Treat reconstruction/layout/EoS/
  direction specialization cloning as the first compile-time target, while
  preserving separate AMReX dimension executables and benchmarking any shift
  from compile-time to runtime selection.
- If current measurement later justifies a dispatch rewrite, the accepted
  direction is an immutable host-built operation plan with approximately
  additive method ownership, not the current Cartesian template product and
  not an unmeasured per-cell virtual dispatch. Run-wide choices should be
  resolved at the coarsest practical boundary. Retain a compile-time policy
  only when AMReX storage requires it or a focused benchmark demonstrates a
  meaningful throughput benefit.
- Compiler caching plus method-local flux owners makes normal narrow iteration
  fast by reusing unrelated optimized objects. Treat the large specialization
  graph primarily as cold-build, broad-invalidation, CI/new-toolchain, disk,
  and possible code-size/runtime debt. Do not rewrite the dispatch unless a
  current cold/cache-miss matrix or runtime evidence demonstrates enough value;
  always report cold builds and cache-hit rebuilds separately.
- AthenaK's hydro/MHD flux owners keep Riemann choice explicitly instantiated
  but switch the uniform reconstruction method inside the Kokkos kernel. This
  makes method growth closer to additive, at the price of a uniform device
  branch. Treat that and a separate-method-launcher design as bounded benchmark
  candidates, not a source-level recipe.
- AthenaK's numerical-relativity task list builds a dependency graph once at
  startup: optional Z4c, matter, gauge, boundary, refinement, and diagnostic
  operations are added according to available modules and then topologically
  ordered. The useful PyFluid lesson is a small immutable host operation plan
  for future GR composition, while AMReX hierarchy lifecycle remains owned by
  the existing solver. Do not import a second scheduler.
- AthenaK's ordinary build also compiles only one selected problem generator.
  That reduces work compared with PyFluid's built-in initial-data set, but a
  feature-stripped validation binary conflicts with PyFluid's production-build
  contract. Keep it as explanatory evidence, not an optimization to copy.
- AthenaK's current Z4c tests are useful independent formulations and
  normalization references, but they are much narrower than PyFluid's existing
  gauge, robust-stability, AMR/interface, stellar, and radiation gates. Adapt
  test ideas only where they add an independent physical boundary; do not add
  duplicate linear-wave tests for breadth alone.

## Review And Change Discipline

The local ETK/CarpetX comparison explains how a large option surface avoids a
single template Cartesian product:

- CCL separates parameter identity (`param.ccl`), GF/group identity
  (`interface.ccl`), and conditional operation scheduling (`schedule.ccl`).
  Many apparent branches are independently compiled thorn routines that are
  activated/composed once, not alternatives inside one fused kernel.
- Hot AsterX code still uses uniform runtime choices. Its seven reconstruction
  methods and LxF/HLLE flux choice are enum switches in device code, while only
  direction and EoS object type are templated in the inspected flux owner.
- Modularity does not make the complete build small: the inspected ET_CarpetX
  configuration contains 85 thorn archives totaling about 402 MiB. AsterX is a
  16 MiB archive and its largest flux object is about 4.7 MiB. The advantage is
  additive thorn/owner compilation and effective incremental rebuilds, not a
  free clean build or zero runtime dispatch cost.
- The PyFluid analogue is a small immutable operation plan: compile dimension
  and required storage topology; schedule optional operations at the host
  level; use uniform runtime method enums where measured acceptable; reserve
  per-cell branches for genuinely data-dependent physics.

- Source-checked external review findings live in `review_backlog.md`. Raw
  Spark notes under `../spark_amrex_review/` are review input, not accepted
  requirements.
- Reproduce a claimed defect before refactoring around it. Prefer the smallest
  correction and its focused gate. Do not introduce a registry, metadata table,
  helper object, or generic iterator merely because a branch or loop repeats;
  retain it only when it reduces ownership drift or a measured maintenance,
  compile-time, or runtime cost.
- Do not repair nonfinite transfer output, weaken a tolerance, add a temporary
  unsupported-combination guard, or silently change a physical default to
  quarantine an active bug.
- Update `review_running_cycle.md` during every AMReX request. Update this file
  only when a durable ownership rule, validation boundary, or open limitation
  changes. Put historical detail and exact measurements in the continuity
  cycle rather than growing this snapshot indefinitely.
