# PyFluid AMReX Source-Checked Review Backlog

This file turns external review notes into a small implementation-ready queue.
An item enters the queue only after its path and claim are checked against the
current source. Raw suggestions remain linked for provenance and do not become
architecture policy by repetition.

Last source check: 2026-08-27, cleanup cycle on `main` after `23daf38`.

Status meanings:

- `ACCEPT`: concrete current issue with a bounded correction and verification.
- `DEFER`: plausible cleanup, but value or design is not established.
- `STALE`: already prevented or superseded in current source.
- `REJECT`: conflicts with current ownership/performance policy or lacks a
  concrete benefit.

## Completed Outcomes And Active Queue

### 1. Give state and auxiliary GFs one ordinary output cadence (`DONE`, M)

Source: `src/amr/amr_solver_auxiliary.cpp`, declaration in
`src/amr/amr_solver_base.hpp`, configuration in `src/framework/config.cpp`, and
the state/auxiliary cadence construction in
`src/amr/solver_sections/evolve_core.inc.hpp`.

`should_refresh_mhd_aux_fields(int step, bool write_aux_gfs)` explicitly
discards `write_aux_gfs`, but removing that parameter alone would preserve a
needlessly fragmented user contract. The normal path should require one GF
snapshot cadence: when the auxiliary cadence is not explicitly overridden,
requested `auxGF` snapshots follow the normal `stateGF` cadence. The resolved
auxiliary compute cadence then equals the resolved auxiliary output cadence
exactly. Keep explicit auxiliary write and cached/live derivative compute
cadences for advanced runs; only an explicit compute override separates the
two. An explicit cached-only mode must remain possible. Scalars and checkpoints
retain their separate cadence and restart lifecycles.

Preferred narrow implementation: retain the current public knobs, distinguish
an omitted override from an explicitly disabled cadence while parsing, resolve
the ordinary auxiliary write cadence from the state plot cadence, and resolve
cached auxiliary refresh exactly from the write cadence unless explicitly overridden.
Pure output auxiliaries should still be computed only when written. Remove the
unused boolean as part of this change, after the resolved cadence contract is
covered; do not silently make every auxiliary a stored evolution field.

Verification: focused state-plus-aux output controls for inherited cadence,
explicit independent write cadence, explicit independent compute cadence,
cached-only operation, initial-only output, restart append, and mixed
centerings. Check written steps and values, then run architecture, all
production dimensions, and quick because this changes shared run behavior.

Implemented in Cycle 202. Configuration parsing now resolves an omitted
auxiliary write interval from `pyfluid.plot_int` and an omitted compute interval
from that resolved write interval. Explicit values, including `-1` and
cached-only operation, retain their old meaning. The writer remains responsible
for a fresh edge-field value on output steps; the evolution-side refresh skips
those steps so the derivative is not computed twice. Focused inherited and
cached-only cases, exact mixed-centering names, architecture, generated-code,
all-dimension build, and the restored quick gate pass. Restart and deliberately
independent write/compute cadences retain their existing focused/full coverage.

### 2. Derive each auxiliary output layout from the GF catalog (`DONE`, M)

Source: `src/amr/output.hpp`, `aux_output_names`,
`aux_output_component_count`, the single-/multilevel writers, and
`codegen/pyfluid_codegen/field_catalog.py`.

The catalog already owns GF names, mathematical labels, tensor components,
roles, requirements, and centerings. Preserve that single declaration point.
Have generated/runtime descriptors produce one ordered output layout per
centering after filtering the requested fields; its entries supply names,
component IDs, centering, and fill ownership, and its size supplies the
component count. Construct it once in each writer. `output.hpp` remains the
AMReX/HDF5 executor and must not grow a second auxiliary registry or locally
duplicated name/component/centering policy.

Verification: focused single-level and multilevel auxiliary-output cases with
mixed centerings; compare component names/order and output values before and
after; run generated-code consistency, architecture, all production dimensions,
and quick because the catalog is shared infrastructure.

Implemented in Cycle 202. `output.hpp` now builds one private ordered layout per
requested centering from generated field descriptors and reuses it for names,
source component IDs, allocation size, fill dispatch, and both single- and
multilevel output. The layout introduces no field identity or metadata outside
`schema/fields.toml`. The focused multilevel MHD case verifies exact `j,e`
component order on every edge centering.

### 3. Move critical catalog identity to data-only TOML (`DONE` for implemented stages)

Source: `codegen/pyfluid_codegen/field_catalog.py`,
`codegen/pyfluid_codegen/diagnostic_catalog.py`, configuration/method catalogs,
and `docs/catalog_schema.md`.

`schema/fields.toml` now owns every field declaration, component, stable ID,
centering, role, requirement, set membership, and output/checkpoint attribute.
The Python field module is its strict loader/model/validator and emitter; its
former declaration tuples are removed. Generated field/component IDs,
descriptors, JSON, names, centerings, and documentation remained byte-identical
through the migration.

`schema/diagnostics.toml` now independently owns diagnostic identity, labels,
groups, stable column ordering, requirements, and component structure.
`schema/methods.toml` owns only the method families that have a clear
declarative boundary, and `schema/physical_id.toml` owns the artifact manifest
vocabulary. Their Python modules are strict loaders/models/validators rather
than second declaration sources. Symbolic mathematics, reduction algorithms,
coupled validation, and physical compatibility remain code. Continue staged
family-by-family migrations only when they delete a real duplicate; do not
introduce an omnibus catalog or custom parser.

### 4. Measure dispatch growth before restructuring it (`ACCEPT` measurement only; refactor `DEFER`)

Source: `src/numerics/flux_factory.cpp`,
`src/numerics/flux_factory_internal.hpp`, `src/flux_kernels/*.cpp`, and the
Newtonian/relativistic reconstruction and Riemann owners.

The current factory selects reconstruction on the host, but each of its eight
owners then instantiates layout x EoS x direction x topology and the Riemann
choices below that boundary. Existing optimized objects show that those eight
owners contribute about 133.3 MiB, or 80.7 percent of the three linked PyFluid
object sets. Supporting another independent option must not multiply every
unrelated policy axis.

Compiler caching and method-local owners substantially mitigate this during
ordinary development: an edit confined to one owner can reuse every unrelated
optimized object, and a no-op/cache-hit build is already fast. The large object
graph is therefore cold-build, new-toolchain/CI, broad-header invalidation,
disk, and possible instruction-cache debt rather than a standing claim that
normal incremental work is slow or that runtime is degraded.

If measurement justifies a change, the target contract is to build and validate
one immutable flux plan from `RunConfig`
before evolution. Keep compile-time specialization only for storage/topology
facts required by AMReX or for a hot numerical axis with measured benefit.
Represent uniform run-wide choices at the coarsest practical runtime boundary,
and organize method implementations so adding one reconstruction or solver has
additive rather than Cartesian-product compile cost. Reject invalid combinations
while constructing the plan, not inside the cell loop.

Do not prescribe a naive virtual/function-pointer call at every face or blindly
move every switch into `ParallelFor`. Benchmark at least two bounded designs:
an AthenaK-style uniform reconstruction switch at tile/row granularity in one
fused kernel, and separate method launchers calling a shared non-templated
backend. Compare cold compile time, object/code size, CPU throughput, and a GPU
build/run when available. The acceptance goal is no statistically meaningful
runtime regression for the production default while code size and cold compile
growth become approximately additive. Literal zero cost for an unlimited
number of algorithms is not a defensible requirement.

Before a production rewrite, record a current all-dimension cold build, a
representative method-local miss, a common/generated-header invalidation, and a
cache-hit rebuild. Retain the existing dispatch if cached iteration is healthy
and no CI, binary-size, memory, or runtime evidence justifies the risk.

AthenaK source inspection in Cycle 202 sharpens, but does not change, this
decision. Its hydro/MHD flux owners explicitly instantiate Riemann solvers while
selecting the uniform reconstruction enum inside the Kokkos kernel, which keeps
those axes closer to additive. Its CMake also compiles only the selected problem
generator. Both are plausible reasons for the fast local clean build, but the
first introduces a uniform device branch and the second would create a
feature-reduced ordinary binary if copied literally. Neither is accepted for
PyFluid without the compile/runtime matrix above.

## Checked But Not Accepted

### Plotfile metadata append failure (`DEFER`, opportunistic XS/S)

Source: `src/amr/output.hpp`, `append_plotfile_metadata_on_io_rank`.

The HDF5 sidecar and native AMReX `Header` paths return silently if the metadata
file cannot be reopened. In a normal run the directory and primary output were
just created, so the remaining window is unusual: permissions or mount state
changed, quota/disk failure occurred, or the target was externally damaged
between data output and metadata append. If this code is touched, replace the
silent return with a checked error containing the target path and output kind.
Do not prioritize it ahead of cadence/catalog consolidation, and do not treat
`create_directories` as another silent failure: the overload in use throws.

### Partial `mesh.patch_counts` can escape parsing (`STALE`)

Spark correctly observed that `configured_patch_counts` treats any nonzero
entry as configured, but current `src/framework/config.cpp` validation aborts
unless every active-direction count is positive whenever any count is set:
`mesh.patch_counts must be either omitted or positive in every active
direction`. Normal solver construction validates configuration before the
multipatch layout is built. A focused config regression could improve coverage,
but there is no current escaped-partial-count defect to repair.

### CT initializer registry/map (`DEFER`)

`initialize_ct_fields` repeats `time == 0.0` and has a readable IC chain. A
single time guard may simplify it, but a runtime map of initializers would add
registration machinery and can obscure compile ownership. Revisit only with a
new IC or demonstrated branch drift; preserve the fallback from cell-centered
magnetic state and the post-initialization boundary/synchronization order.

### Patch-union domain-to-level cache (`DEFER`)

`apply_boundary` scans the small active level set only when patch-union mode is
enabled. Cache only after profiling shows material cost and after defining
invalidation across level creation, remake, restart, and regrid. A lookup
optimization is not currently a correctness finding.

### Checkpoint CT helper split and spacetime coarse-source resolver (`DEFER`)

Both suggestions may improve readability, but the order of boundary fill,
face-field restoration/reallocation, cell synchronization, time clamping, and
RK dense-state selection is numerical behavior. Require a narrowly specified
invariant and restart/subcycling regression before changing either sequence.

### Broad table/registry/iterator conversions (`REJECT` as a blanket plan)

The earlier Spark passes proposed schema tables for configuration, factory
registries, generic dimension iterators, storage descriptors, and broad CMake
macros across many files. Cycles 173--175 already replaced major ownership and
catalog structures, and several raw notes name pre-migration paths. Review any
remaining repetition locally. Accept a table or helper only when it preserves
host-side selection, specialized point kernels, generated sources of truth,
and measured compile/runtime behavior while deleting real duplicated policy.

The six items labeled high-confidence/high-impact in Spark's broader final pass
were checked separately:

- `src/framework/config.cpp` already begins with registered string/scalar
  parameter tables for method and AMR options. Complex arrays and coupled
  compatibility checks remain explicit. A wholesale heterogeneous schema
  conversion is therefore `STALE` as a description of the current design and
  `DEFER` for any smaller local parsing duplication.
- `src/numerics/flux_factory.cpp` is a 78-line host-side switch with build
  guards and method-local compiled owners. Its private template dispatch keeps
  topology, layout, EoS, and direction specialization outside cell loops. A
  runtime registry would not be simpler and is `REJECT` without a new method or
  measured ownership problem.
- `tools/validation_common.py` already centralizes `mpi_command`, `run_mpi`,
  and `run_amrex`. Remaining callers should use those helpers, but the claim
  that command assembly first needs a new common builder is `STALE`.
- `src/initial_data/initial_data_state.cpp` is still large, but it already has
  per-IC initializer functions, one final IC dispatch, and common
  relativistic/Newtonian materialization boundaries. A structured per-cell
  payload could increase kernel branching or compile ownership. Further split
  work is `DEFER` until a concrete duplicated formula or incremental-build
  problem is identified.
- `src/numerics/hierarchy_interpolation.cpp` is a focused 257-line owner. Its
  dimension-specific coordinate assembly is visible, but transfer semantics
  were recently correctness-critical. A generic envelope is `DEFER` until it
  deletes an actual policy duplicate and passes the full interpolation/AMR
  controls.
- `src/numerics/ct_edge_flux_register.cpp` contains real 2D/3D topology
  differences inside one owner. Directional factoring is `DEFER`, not a
  cleanup-first task: require exact face-field/reflux equivalence, rotated-axis
  controls, AMR+CT restart, and divB/conservation evidence.

## AMReX 26.10 optional integration review (Cycle458)

- `MEASURED / DEFER production adoption`: `MLMG::setNoGpuSync(true)` uses a
  single-stream solve region and synchronizes before returning. The warmed
  Metal FP32 cell-centered periodic Helmholtz control is 7.7--9.3x faster
  for 32/64-cell one-box and 64-cell sixteen-box layouts; both orders retain
  identical error, mean error and residual. Before enabling it, measure the
  actual diffusion/gravity operators on/off, including nodal/edge and larger
  box layouts. No production call site has changed. Exact probe source,
  flags and logs: `runs/current/infrastructure_audit/cycle458/`.
- `DEFER`: new sparse AMG/Krylov requires an actual sparse-matrix operator.
  Current structured MLMG consumers do not justify a new assembly route.
- `AVAILABLE / DEFER case tuning`: upstream whole-direction refinement and
  unsplit-box controls may suit thin/slab anisotropic layouts. They preserve
  the finite-volume transfer contract and do not establish new CT/refinement
  support; evaluate with the existing physical-error and resource gates.

## Comparative compilation source review (Cycle463)

- `SOURCE-CHECKED / DEFER measurement`: local AthenaK builds broad hydro/MHD,
  radiation and NR sources in one executable. Its hydro flux owner specializes
  Riemann selection but keeps reconstruction, passive count and dimensional
  activation runtime. Our `rhs.hpp` retains reconstruction-by-Riemann and
  layout/topology specializations. Investigate one uniform-choice factorization
  under the300s owner deadline; retain fused states and require numerical and
  runtime acceptance. Fewer template axes is a hypothesis for smaller expanded
  code, not a measured build-speed improvement.
- `SOURCE-CHECKED / DEFER`: public AthenaK main stages reconstruction and
  Riemann work through global face-state buffers, unlike the older local fused
  team-scratch snapshot. This illustrates a compile/memory/launch tradeoff;
  changing our fused ownership solely to ease compilation is not accepted.
  Source: https://github.com/IAS-Astrophysics/athenak/blob/main/src/hydro/hydro_fluxes.cpp
- `ALREADY PRESENT`: `cmake/PyFluidSharedObjects.cmake` shares compatible
  multiply-used implementation objects per dimension. A generic suggestion to
  introduce shared libraries first is stale. Investigate actual duplicate work,
  transitive includes and broad initial-data/spacetime dependencies instead;
  retain full-feature integration and dimension-compatible AMReX objects.
- `SOURCE-CHECKED / DEFER`: local SpacetimeX_mfrk Z4c fixes derivative order4
  and separates derivative buffers from pointwise RHS algebra. SpECTRE's own
  build guide uses compiled libraries and system-specific executables, and
  documents compile/memory costs of forced inlining. These are bounded source
  examples, not evidence of cheap complete NR/CUDA builds. Preserve the user's
  fixed-choice option and measure ownership/inlining costs before changing them.
  Source: https://spectre-code.org/spectre_build_system.html

Exact local paths, source hashes, comparison limits and proposed probes:
`runs/current/infrastructure_audit/cycle463/report.md`. The local AthenaK
snapshot has no Git identity and its historical build metadata is optimized
CPU-only, with obsolete source paths. No compile was run for this comparison;
SpEC source was not reviewed. See `docs/build_cost_roadmap.md` for the plan.

## Spherical collaborator source review (Cycle 475)

- `SOURCE-CHECKED / IMPLEMENTATION PENDING`: spherical fluid origin ghosts
  with active theta omit the antipodal theta reflection. Current
  `boundary.hpp` only enables its nonlocal origin phase with active phi and
  leaves theta unchanged. `rho=1+0.1*z` gives a direct source-index
  counterexample without a vector convention assumption. Follow the current
  cycle for runtime evidence, compiled-owner repair and MPI/AMR limits.
- `SOURCE-CHECKED / DESIGN`: collaborator cell-shifted point samples support
  the previously proposed startup spacetime-centering policy. Preserve one
  generated physics path and explicitly distinguish point-field AMR transfers
  from finite-volume averaging. Parity ghost fills alone do not implement
  singular vertex equations; global radial remapping is not conservative AMR.
- Other source capabilities and rejected direct adaptations are recorded in
  `docs/baumgarte_spherical_review.md` and its pinned branch map. The review
  inventories all branch sources and reads selected numerical owners; it does
  not certify every historical formula or reproduce collaborator runs. The
  collaborator repository is strictly read-only under `../AGENTS.md`.

## Separate pre-existing flux findings (Cycle 478)

- `SOURCE-DEMONSTRATED MECHANISM / RUNTIME REPRODUCTION PENDING`: traced 2D
  mapped-hydro external walls pass an empty passive vector to
  `add_mapped_patch_flux_correction<NComp>`
  (`src/numerics/mapped_patch_interfaces.hpp:611-619,775-783`). For
  `NComp>NVAR_HYDRO`, both insertion and accumulation index that vector
  unconditionally (`:390-392,409-411`). This is an out-of-bounds mechanism,
  retained in the frozen Cycle478 control, separate from the optional-tracer
  factoring. The direct boundary writer explicitly sets passive flux to zero
  (`:579-582`). Reproduce with a tiny traced mapped layout containing an
  external reflective face under bounds checking/ASan; require finite zero
  wall dye flux, unchanged physical fluxes, and equal/opposite internal-seam
  corrections before using this branch as an equivalence oracle. No runtime
  failure, repair or complete mapped-tracer acceptance is claimed. Provenance:
  `runs/current/infrastructure_audit/cycle477/tracer_assessment.json`.
- `SOURCE-CHECKED DIVERGENCE / RUNTIME REPRODUCTION PENDING`: the all-direction
  analytic-metric RHD/RMHD calls in `src/numerics/rhs.hpp:656-671,697-714`
  omit the resolved diode argument, selecting the empty default in
  `src/relativity/{rhd_system.hpp:504,rmhd_system.hpp:321}`. The directional
  calls pass it (`rhs.hpp:914-924,947-960`). With profiling enabled,
  `profile_detail=flux` selects directional dispatch while basic profiling
  selects all-direction dispatch for fixed metrics
  (`src/amr/solver_sections/rhs_assembly.inc.hpp:301-402`;
  `src/framework/profiling.cpp:55-57`). Thus the source permits a profiling
  choice to change a diode face's reconstruction/transport treatment; the
  numerical impact depends on the boundary state. This omission predates
  Cycle478's tracer changes. Reproduce fixed-Minkowski RHD and cheap-solver
  RMHD with a nonperiodic diode and incoming face transport; compare basic
  versus flux profiling, requiring identical face fluxes, state/scalar
  histories and the intended diode baryon-flux sign. Stored-metric dispatch
  and separately routed advanced RMHD are outside this exact mechanism.
  No reproducer, fix, numerical magnitude or broad acceptance exists yet.

## Raw Review Provenance

- `../spark_amrex_review/2026-07-20_final_refinement.txt`: concrete final
  refinement; primary external input for this backlog.
- `../spark_amrex_review/spark_amrex_review_final_pass.txt`: broader prioritized
  maintainability pass.
- `../spark_amrex_review/spark_amrex_review_notes.txt`: long brainstorming and
  repeated review waves; useful for search, not an accepted roadmap.

When an item is implemented, record its exact change and validation in the
current continuity cycle, then retain only a compact outcome here when it is
needed to interpret later Spark claims. Move a short result to `knowledge.md`
only if it changes a numerical fact or open limitation; durable ownership
belongs in `docs/infrastructure_conventions.md`.
