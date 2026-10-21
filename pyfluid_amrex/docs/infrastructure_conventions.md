# PyFluid AMReX Infrastructure Conventions

This document owns durable infrastructure and artifact contracts. Numerical
methods and physics belong in their topic documents; coding style belongs in
`code_conventions.md`; dated measurements and decisions belong in
`review_running_cycle.md`.

## Continuity Files

- `references.md` is a router to authoritative local sources and external
  references. It does not own design decisions.
- `knowledge.md` records compact learned facts, numerical boundaries, and
  AMReX or physics subtleties that materially affect future work.
- `review_running_cycle.md` records dated work state, commands, measurements,
  failures, and validation evidence.
- This file records stable infrastructure ownership and artifact semantics.

## Conservative Stage Accounting

`schema/time_integrators.toml` is the sole declaration of time-method IDs and
rational stage coefficients. The strict loader derives stage times, final flux
weights and projection weights and checks nonlinear RK order conditions through
order four. Generated IDs stay separate from coefficient tables; pointwise
physics must not include the tables. Main stage updates, CT staging, boundary
stage times, integrated flux/source budgets and SSPRK3 gravity carry factors
consume these descriptors. Keep the AMReX lifecycle/storage calls explicit.
Dense-time interpolation and physics-specific corrections require their own
mathematical validation; the RK catalog does not certify them automatically.

A direct state projection is a numerical source even when primitive recovery
succeeds. Record its contribution to the accepted state, not merely to the next
intermediate candidate. For SSPRK3 the first intermediate-state projection
survives with weight 1/6, the second with 2/3; a mutation of the base state has
unit weight. Intermediate midpoint/RK4 projections affect fluxes but have no
direct state contribution. Physical source-free conservation and closure of
boundary-plus-source bookkeeping are separate validation claims.

## Spacetime accuracy contract

`spacetime.derivative_order` is the single selection for bulk spacetime
stencils, connection completion, constraints and KO. Derive ghost reach from
the generated stencil contract, including one-sided excision reach. Constraint
validity masks and physical-boundary exclusions must use the selected radius.
Order selection dispatches the differential operator, not separate copies of
the equations. Keep initial-data solve accuracy, materialization/prolongation,
fluid reconstruction, time integration and physical boundary algorithms
explicitly separate; a high-order stencil does not upgrade those methods.

Cartesian and reference-metric connection completion share the metric-only
first-derivative gather, with stencil-order dispatch outside the device launch.
The full evolution gather reuses the same component assembly. Excision keeps
its existing stencil-side selection and reach; a domain-sided diagnostic
derivative is not an interchangeable replacement for that policy.

Constraint diagnostics select `Z4cDerivativePurpose::Constraints` in that same
assembly: metric first/second derivatives and curvature first derivatives,
without gauge or upwind transport stencils. Select Cartesian versus
reference-metric equations once on the host before the diagnostic launch. Keep
the order dispatch confined to the differential operator so each order does not
instantiate another equation kernel. The symbolic dependency check covers Cartesian and reference-metric
constraints, including their intermediate geometry; update this contract if
the equations acquire new inputs. Excision side selection, validity masks and
diagnostic volume/ownership rules remain responsibilities of their existing
owners, independent of which derivative subset is evaluated.

Numerical kernel generation depends on both `schema/numerics.toml` and its
loader/emitter. Generated leaves assert their actual emitted order set rather
than trusting only a separate catalog header: an incomplete rebuild must not
silently admit a new order into a zero-return fallback. Host orchestration and
point-kernel dispatch consume the same catalog but retain appropriate host and
device call boundaries.

## Reference-metric evolution and future patch interfaces

Reference-metric equation owners consume scalar/vector/tensor values and
reference-covariant derivatives in a local orthonormal frame. They do not
select a coordinate family or perform patch lookup, ghost exchange, AMR
transfer, or MPI calls. Chart geometry owns maps, Jacobians, basis factors
and their derivatives; the AMReX layer owns storage and communication.
The current generated operators assume a time-independent reference metric
and basis. Moving maps need their time derivatives in the evolution contract.

The fCCZ4 runtime adapter maps every component and derivative index between
runtime axis order and the canonical chart order. It explicitly decodes
`K=K_hat+2Theta` and the reference Lambda interpretation of the common
connection slots. Equation kernels do not infer these meanings from buffer
positions. Checkpoint metadata must preserve the formulation and chart/basis
semantics. The generated auxiliary-B gauge kernel is separate from the current
integrated Gamma-driver runtime and does not imply an extra evolved field.

Cell-state radial origin continuation has one independently compiled owner in
`src/numerics/origin_boundary.cpp`. Uniform levels gather a bounded radial
band; partial levels supply that same band through the existing
`FillPatchNLevels` owner, using its selected state interpolater and time slices.
Ordinary local physical-BC refreshes preserve these nonlocal origin ghosts.
Do not add another angular map or a separate coarse-fine donor algorithm in a
physics/reconstruction owner. Staggered CT fields and spacetime point fields
need their own centering-aware transfer and collapsed-topology validation.

Divergence, flux-register weighting and reflux share
`uses_cartesian_momentum_fluxes()` for the momentum representation. Newtonian
full-chart HD may integrate Cartesian momentum through physical face vectors;
Valencia uses local reference-basis fluxes with scalar curvilinear face areas.
Do not infer the representation from dimension or absence of magnetic fields.

FV fallback corrections use the same physical face areas and cell volumes as
the primary divergence. Reduced-domain diagnostic normalization is a separate
policy in `metric_reductions.hpp`: multiply cell and boundary-face integrals
by the same symmetry factor. Do not put full-sphere diagnostic factors into
local evolution fluxes. Constraint L2 weighting uses the chart's nodal dual
volume; it must not relabel that reference volume as physical proper volume.

Mixed-coordinate patches must represent the same geometric reference metric
in their respective charts for connection differences to transform as tensors.
Assigning independently chosen reference connections to each patch changes
the conformal connection variable's meaning and requires an explicit field
conversion, not just rotation. In a frame change, every index of a
reference-covariant derivative transforms, including derivative indices.
Raw partial derivatives of frame components require the differentiated basis
and chain rule instead. Constructing destination derivatives after exchanging
tensor values is a separate contract from transferring derivative jets.

Keep spacetime tensor exchange separate from finite-volume/CT synchronization.
Cell-integrated conserved quantities, face magnetic fluxes and edge EMF
integrals need matching physical measures, orientations and conservative
interface operators. A correct pointwise tensor rotation does not establish
those contracts. Reduced-dimensional patches also need compatible physical
symmetry assumptions; matching their counts of active axes is insufficient.
These are extension boundaries; mixed-coordinate runtime evolution is not yet
implemented by the reference-metric kernels.

Relativistic accepted-state checking uses the shared non-mutating recovery
and conserved round-trip contract. Initial-data preflight additionally tests
repeatability of recovered primitives. Keep these explicit check modes:
repeated primitive inversion near a cold admissibility boundary can be
ill-conditioned even when the conserved round trip is accurate. Measure conserved momentum round trips against the
whole vector norm: raising/lowering can cancel a nearly zero component.
Keep density and energy separately scaled so that cold-energy errors remain
visible. Do not duplicate Newtonian strict-positive-pressure
checks for an EOS/recovery path that admits finite cold matter. Numerical
normalization in generated kernels belongs in ordered symbolic bindings;
residuals and derivatives consume those same bindings. Avoid expanding
normalized recovery algebra into powers of density during code generation.

## Production And Source Ownership

- `pyfluid_amrex/` is the production C++/AMReX path. Python/JAX and
  Python/Numba are reference paths unless a task explicitly selects them.
- `dependencies/amrex.toml` owns the exact official AMReX release and resolved
  source commit. `tools/manage_amrex.py` checks only by default. Its explicit
  update path builds source/build/install under versioned `.deps` directories,
  verifies MPI, HDF5, linear solvers, and all dimensions, then switches the
  stable install link. Dependency management must not build PyFluid. CMake
  resolves the stable package hint into real imported include/library paths
  and tracks the pin as a configure dependency; normal dependency rebuilding
  must not delete unchanged generated artifacts. Retire obsolete managed
  trees after successful activation, preserving still-active companions and
  locally modified/unverified sources, not a permanent archive. Do not make
  presets chase a mutable source checkout.
- Scalar precision and execution backend are independent build policies.
  Keep CPU FP64, CPU FP32, and experimental GPU builds in separate directories
  with matching AMReX installations. Handwritten real-valued arithmetic uses
  `amrex::Real` and typed `_rt` literals; generated expressions follow the
  same rule. Retain intentional wider host arithmetic only where its owner
  requires it. Output metadata reports the build's real mantissa bits.
  Do not infer precision from filenames or globally weaken validation budgets.
  The opt-in precision comparison reuses existing physical oracles and declares
  separate arithmetic and physical-error limits. Performance runs validate each
  repeat before overwriting its outputs, retaining compact per-repeat scalar
  results alongside timing medians.
- Managed CUDA installations select `AMReX_GPU_RDC=OFF`; application targets
  inherit that policy through AMReX's CMake helper. Pointwise device callees
  belong to their kernel's translation unit, while ordinary host modules stay
  separately compiled. This avoids repeated template-kernel registration with
  the validated CUDA 13 toolchain. Cross-translation-unit device calls would
  require an explicit ownership/build-policy change and fresh validation.
  Keep default sanitizer API reporting enabled; do not suppress registration
  diagnostics or maintain backend-specific copies of the equations.
  Evaluate backend changes by numerical correctness, instrumentation, startup,
  steady-state throughput, compile cost and code size together. A smaller
  binary or a clean warning log alone does not establish an improvement.
- Selected short, unit-scale precision controls use
  `validation_common.precision_scalar_policy`, identified as
  `short_run_roundoff_v1`. Conservation accounting retains its 64-epsilon
  allowance and existing FP64 bounds. Runtime `divb_l1` is the volume integral
  of absolute divergence: compare `divb_l1*h/(V*B_ref)` with 64 epsilons in
  either precision. Declare the characteristic field independently of the
  observed error. `cartesian_divb_scale` uses the periodic box volume and
  smallest solver-reported active directional width over saved times.
  On AMR this is a conservative global bound, not a cellwise normalized norm;
  curvilinear, symmetry-multiplied or masked domains need their own scale.
  Per-run result tables retain raw diagnostics, scales, effective tolerances,
  and precision metadata; rolling history descriptions retain policy identity
  and divergence scale/tolerance. Unknown/mismatched precision and invalid
  scales fail. Physical accuracy, convergence, recovery, positivity, and
  stability requirements remain independent; exact-zero contracts stay exact.
- FP32 production owners compile with Clang floating promotion/narrowing errors.
  This includes generated coefficient catalogs, overloads and stage-time
  arithmetic, not only field storage. Keep intentional wider host work explicit
  (`WallTime` for profiling), and inspect device IR when changing a backend.
  Device FP64 must fail on this Metal route rather than be silently demoted.
  Retain precise device math and validate compiler changes with a fresh JIT cache.
- RK state blends are affine: the exact coefficient catalog enforces unit
  total state weight, and `time_integration.hpp` adds the stage difference
  and RHS increment before the base state. Accepted states, limiter candidates
  and CT stage fields must use the same arithmetic. Preserve a constant state
  exactly when the RHS vanishes; general linear combinations retain their
  separate contract. Do not introduce precision-specific evolution copies.
- Runtime integral/ledger reductions share `ReductionSum` from
  `reduction_sum.hpp`: two Real values for compensated FP32, ordinary Real for
  FP64. Retain the correction when combining local partial sums; host MPI
  reduction remains the hierarchy owner's operation. The compiled diagnostic
  component-sum owner is `src/numerics/diagnostic_reductions.cpp`. Python may
  inspect saved fields to diagnose rounding but must not replace runtime totals.
- Array4 is a view, not a host-access guarantee. Grid statistics that read
  solver fields must use AMReX device-capable reductions; detailed host-only
  inspection requires an explicit completed copy. Directional AMR tag counts
  and strengths use the same indicator on CPU/GPU and transfer only reduced
  values for MPI aggregation.
- Experimental Metal dependency/compiler patches are pinned in
  `dependencies/metal.toml`, isolated from official CPU sources, and prepared
  by `tools/manage_metal.py`. Verify exact patched source identity before use;
  never reset/clean an unexpected dependency checkout. The selected GPU
  profile must fail without its requested device. Keep device transfers and
  asynchronous buffer lifetimes in AMReX storage/launch owners, not physics.
  The Metal runtime must declare indirect USM resources to the command encoder,
  including allocations reached through Array4/BoxIndexer metadata. Sharing an
  allocation can hide missing residency/hazard declarations; the storage check
  deliberately separates field data from metadata. Keep this rule below physics.
  Unified frontend compilation also requires the compiler's target-selection
  mechanism; a preprocessor device-pass macro is insufficient. Validate host
  and device iteration ownership before relying on box-loop backend kernels.
  See `precision_and_metal.md` for setup and the current evidence boundary.
- The checked-in Apple Silicon presets and local machine profiles explicitly
  set `CMAKE_OSX_ARCHITECTURES=arm64`. CMake's implicit Darwin detection queries
  `sysctl`, which a sandbox may deny; otherwise reconfiguration can silently
  add/remove `-arch arm64` and invalidate every compile/cache key. Keep this
  host-specific choice in those presets/profiles, not portable solver CMake.
- Host concurrency is a machine/build concern below the numerical ownership
  graph. Production kernels target AMReX; machine profiles select MPI and an
  AMReX-supported host-thread backend when one is useful. Do not call Grand
  Central Dispatch or another operating-system scheduler from physics, ID,
  diagnostic, or timestep code. The optional local MPI+OpenMP companion uses
  the same source pin and generated production matrix as the MPI-only default;
  it differs only at the managed AMReX install, CMake preset, and machine
  profile. Host threading belongs at AMReX's coarse tile/task boundary and must
  demonstrate numerical equivalence and useful task grain before its profile
  becomes a default for a workload.
- The input file is the simulation configuration. Runners execute and report;
  they do not override simulation parameters.
- `tools/input_parameters.py` owns Python interpretation of self-contained
  AMReX assignment files. Reporting consumes decoded values; input
  materialization retains serialized quoting. AMReX owns expression evaluation,
  includes, and conditional preprocessing; Python rejects unresolved directives
  instead of guessing their effect. Before cleaning output, the runner checks
  executable and read-dependency availability and rejects overlap with restart,
  physical-ID, or EoS input artifacts, including symlink paths. Use a separate
  output directory or explicitly retain existing output for such runs.
- `include/pyfluid_amrex/config.hpp` owns the runtime configuration contract;
  parsing, supported choices, validation, and derived policy live under
  `src/framework/`.
- `schema/fields.toml` is the only declaration source for named grid fields,
  components, labels, centerings, roles, and persistence/output ownership.
  Python strictly loads and validates it and emits generated C++/JSON/document
  views. Runtime code must not create a second raw-string field registry.
- `schema/diagnostics.toml` is the only declaration source for runtime and
  postprocess diagnostic identity, mathematical labels, scalar/vector shape,
  canonical column ordering, applicability, groups, and dependencies. Python
  strictly loads and validates it, expands component families, and emits the
  generated C++/JSON views. Symbolic integrands and runtime reduction
  algorithms remain owned by their generated-equation and C++ executors.
  A runtime surface integral also declares its surface target there. Generic
  surface geometry and hierarchy interpolation remain compiled executor
  services; the catalog must not become a string-encoded quadrature algorithm.
- `schema/methods.toml` owns supported-method identity and dispatch metadata,
  including reconstruction names/codes/stencil radii/characteristic forms,
  rotation-law names, and the runtime AMR indicator policies and their default
  thresholds. An AMR policy selects a small set of semantic
  density/flow/magnetic/spacetime indicators; it does not create a template
  product or infer a driver from every stored GF.
  Its native initial-data-owner records declare natural
  coordinate/axis layouts, dimensions, grid kinds, broad object/disk support,
  relative-tilt support, and reflection class. Coupled physical restrictions
  remain explicit host validation; owner records do not generate a solver
  Cartesian product. Numerical formulas and generated equations remain in
  labeled C++ or symbolic Python sources rather than TOML expression strings.
  The catalog describes native nonlinear solve owners, not every validation
  projection. In particular, an `N x M x 1` run is the 3D operator on a
  one-cell invariant direction and is a required dimensional-degeneracy check
  against the corresponding native `N x M` reduction when that symmetry is
  valid; it is not a separate physical owner or a new compiled method product.
- `schema/numerics.toml` owns the small shared identity of supported spatial
  accuracy families. SymPy derives derivative, upwind, dissipation, staggered
  point-interpolation, and uniform-Cartesian point/cell-average coefficients
  from that list; TOML does not store coefficient tables. Catalog presence
  advertises a reusable stencil family, not whole-solver convergence. Each
  elliptic, hyperbolic, AMR-transfer, CT, and boundary owner must separately
  expose and validate the orders it actually supports.
- `schema/physical_id.toml` owns stable physical-ID manifest keys, their value
  classes and sections, and the runtime component role/kind/gravity vocabulary.
  Generated C++ validates artifact publication/loading and generated JSON is
  the Python reporting view. `fields.toml` still exclusively owns GF identity
  and centering, while `methods.toml` still owns solve-method capabilities;
  the manifest catalog does not duplicate either source or encode conditional
  physics compatibility.
- Each compiled reconstruction value owns one flux-kernel translation unit.
  Its CMake feature switch must remove that owner from the source list, not
  merely hide its factory case. Candidate correction composes selected
  reconstruction methods at runtime; it is not another template axis.
- A large pointwise algorithm that would multiply through every generic
  reconstruction owner belongs in one dedicated compiled owner. Keep its
  public header to a small host entry, keep device-callable mathematics in one
  private algorithm header, and prevent unintended inlining when measured
  object growth shows cloned bodies. Instantiate only runtime-valid method
  combinations; configuration must reject misleading unsupported variants.
- Runtime reduction identity comes from `schema/diagnostics.toml`; generated
  integrands come from the symbolic diagnostic-equation owner. Runtime AMReX
  reductions own discrete totals; postprocessing owns analytic comparisons and
  derived presentation.
- Independent fields at the same communication phase may be passed together to
  AMReX's vector `FillBoundary` or `FillBoundaryAndSync` APIs so their exchanges
  can overlap. They must share the hierarchy periodicity and compatible FabArray
  type. Do not batch across a physical-boundary update, coarse-fine transfer, or
  any other dependency that requires one field's completed ghosts first.
- Diagnostic and integral owners may fuse per-cell sum/min/max work locally
  when profiling supports it and may pack same-operator MPI values when that
  reduces collectives without changing ownership. Do not assume that one large
  heterogeneous reducer beats AMReX's specialized `MultiFab` reductions on
  CPU; retain the faster measured path. A local reducer is not permission to
  omit the explicit hierarchy collective or to mix semantically different
  cadences.
- Validate interpolation-specific refinement-ratio contracts before hierarchy
  evolution. CT face-divergence-preserving transfer accepts ratios 2 and 4;
  conservative quartic cell-state interpolation remains ratio-2-only. Keep CT
  ratios isotropic until edge-EMF reflux/register behavior has direct
  anisotropic validation.
- The symbolic C++ printer owns fixed-integer power spelling. It expands the
  smallest powers directly and emits `amrex::Math::powi<N>` for other integer
  exponents; generated headers own the corresponding AMReX math include.
- Fluid conserved state is cell-centered, CT magnetic state is face-centered,
  and evolved Z4c state is vertex-centered. Named persistent fields belong in
  the field schema; RHS, RK-stage, elliptic-residual, transfer, and other local
  work arrays are scratch.
- Large host orchestration belongs in compiled `.cpp` owners. Header-only code
  is for small value types and pointwise CPU/GPU kernels where inlining is part
  of the implementation. Runtime method choice occurs outside cell loops.
- Newtonian acceleration, heating and gravity source application share the
  compiled `src/numerics/sources.cpp` owner. Its declaration-only private header
  carries `SourceStageContext`; hydro/MHD/passive layouts use the same conserved
  density, momentum and energy roles here and do not need separate source
  instantiations. Keep gravity-field construction and the gravity source's
  energy-coupling contract in their existing shared owners.
- Profiling observes the selected operation plan; it must not alter physical
  constraints passed to a launch. Keep resolved diode/excision policy identical
  between all-direction and directional flux interfaces and protect that contract
  with numerical comparisons rather than relying on caller-default arguments.
- The signal-speed factory selects layout and diffusion timestep policy on the
  host at runtime. Keep separate hydro/MHD/relativistic pointwise kernels and
  the measured ideal-gas EoS fast path. Tracer presence and diffusion policy
  must not multiply these launch bodies; neither changes their cell equations.
- Subcycling uses the same immutable host operation-plan pattern as single-rate
  evolution. Select the specialized time integrator and diffusion operations
  once, then pass them through the shared recursive level advance. Timestep
  and diffusion choices must not multiply the recursive AMR orchestration.
- RMHD candidate limiting shares candidate assembly, recovery and mask closure
  across correction Riemann solvers. Dispatch only the face-flux launches by
  solver; select PWL/PWC face states before one common Riemann call. Keep the
  shared-face closure, correction counters and conservative update order intact.
  `src/relativity/rmhd_correction_flux.hpp` is the declaration-only host launch
  contract; private per-method owners include the shared pointwise launch
  implementation. Keep each device kernel and its callees together for non-RDC
  CUDA. A hydro-only configuration owns no RMHD correction launches; LLF/HLL
  remain internal fallback dependencies of an enabled MHD correction path even
  when those primary methods are disabled.
- Identical implementation sources used by the executable and its utilities or
  tests share CMake object owners within one dimension/backend. Entry points
  stay private; a small check must not pull in unrelated implementation files.
  The isolated horizon-finder OpenMP owner remains separate. Normal dimension
  build presets build that solver only; comparison tools are explicit targets.
- Full and focused Mac CPU builds use the Ninja executor through their CMake
  presets; the local machine profile selects the same preset, including ccache
  and compile-database ownership. Generator choice does not change physics or
  dependency features. Existing Make caches require an explicit metadata
  migration: retain objects/cache, compare compiler commands, and measure the
  first trusted replay separately from a steady no-op. Do not use `--fresh` or
  a clean-first build to change executors. Optional companion/GPU trees retain
  their explicitly selected executors until independently migrated.
- Architecture checks protect dependencies and ownership, not file length,
  suffix preferences, whitespace, or exact qualifier spelling. Retain behavioral
  fixtures for real failure modes; an independent numerical oracle earns its
  place by testing a contract that existing checks do not establish.
- The quick architecture gate follows transitive project includes to protect
  declaration-only factories/configuration and method-local reconstruction
  ownership. It reports the include chain on failure. This is a conservative
  all-feature source check, not a replacement for CMake's configured dependency
  graph. Intentional shared PPM machinery and the single advanced-RMHD owner
  are explicit exceptions; new exceptions require a real ownership reason.
- Measure expensive owners with `tools/profile_compile.py` using the configured
  compilation database and unchanged production flags. Profiles disable cache,
  isolate compiler outputs from the build tree, and retain compact logs/timings
  rather than another object cache. `--trace` selects Clang phase totals or
  NVCC device-tool process spans. Nested phases and parallel processes overlap;
  do not add them to estimate wall time. Use `tools/benchmark_compile.py` for
  whole-target generation, compilation and link cost; select its time budget
  explicitly. A missed timing goal is recorded as `slow`, distinct from a
  compilation failure. Both tools default to a separate 300 s hard deadline
  (`--timeout`); expiry terminates the command's process group and records
  timeout rather than allowing an unbounded build. `--allow-slow` cannot
  suppress a timeout. Start investigations with one owner and a bounded probe;
  justify longer integration runs from measured scope first. Never weaken
  optimization or remove methods merely to meet a development timing goal.
- Complete Newtonian face solvers may use ordinary `inline` so the compiler
  can share large bodies across passive and wall callers. Ordinary hydro PWC
  retains direct forced inlining because the added call cost measurably slows
  its cheap loop. Both call the same reconstruction/positivity and Riemann
  implementations. The current path fuses reconstruction and solving without
  persistent face states. Replacing that boundary needs matched compiler,
  memory and runtime evidence; fusion is not a permanent backend constraint.
  This is a measured exception, not
  permission to force-inline every pointwise routine or to outline every
  cheap one. Validate runtime as well as compile time when moving the boundary.
- Shared diagnostic interfaces include generated IDs and value storage, not
  descriptor tables or visitors. Both are emitted from `schema/diagnostics.toml`;
  metadata consumers include the catalog directly. Display-label/group-membership
  changes must not rewrite the value interface or rebuild unrelated AMR owners.
- Vector kernels are emitted by equation responsibility under
  `kernels_generated/vector/`: calculus, reference-metric connection,
  viscous, EMF, magnetic scalars, CT, and finite-volume divergence. Their single grouping owner is
  `VECTOR_KERNEL_GROUPS` in `codegen/pyfluid_codegen/vector_equations.py`.
  `vector_ops.hpp` is an aggregate validation include, never a production
  dependency. EoS includes only the magnetic-scalar leaf; CFL selection uses
  `diffusion_timestep.hpp`, not the viscous RHS. Vector-only source edits use
  their own CMake generation owner. Keep write-if-changed semantics for every
  leaf so an unchanged group does not invalidate its consumers.
- Reference-metric Valencia MultiFab orchestration belongs to the compiled,
  non-template `src/relativity/reference_metric_valencia.cpp` owner. Its public
  header is declaration-only. Do not put the loop in `flux_divergence.hpp` or
  a reconstruction bundle: that couples a geometry edit to every
  reconstruction product without buying runtime specialization.
- Analytic fixed-spacetime selection is metric ownership, not merely a storage
  optimization. Once selected, initialization, primitive recovery, fluxes,
  CFL, sources, admissibility, boundaries, scalar diagnostics, and auxiliary
  primitive output must evaluate the analytic metric in the runtime reference
  basis; they must not consume the redundant stored Cartesian Z4c copy.
  Generated diagnostic requirements expose Cartesian Z4c constraints only on
  Cartesian meshes. Coordinate-aware pointwise accessors retain an explicit
  Cartesian fast path so RMHD and Cartesian RHD do not pay for a basis
  transformation; do not express coordinate family as another reconstruction
  template-product axis.
- `EOSParams.method` is the sole evolution-time EoS identity. The production
  bundle instantiates `EOSRuntime` once for each storage/topology signature;
  flux, signal-speed, primitive-cache, atmosphere, source, boundary, and C2P
  launches carry that same compact value and must not carry a second method
  integer. Adding an EoS therefore extends the closure owner rather than the
  evolution-bundle or reconstruction template product. Introduce a specialized
  compiled hot path only after a focused benchmark establishes a material gain
  and keep its numerical contract equivalent to the runtime owner.
- The measured ideal-gas exceptions are primitive-cache/CFL launch selection
  and RMHD C2P boundary selection. A host launch checks the canonical
  `EOSParams.method` once for cache/CFL kernels. The canonical RMHD recovery
  entry checks it once per recovery and reuses the existing ideal
  RePrimAnd/Palenzuela algebra; every nonideal method uses `EOSRuntime`. The
  latter raised measured C2P throughput from `1.342` to `1.725` million
  recoveries/s and application throughput by 7.3%.
  Neither exception restores the four-EoS evolution/reconstruction product.
  Do not extend these selections without a similarly isolated measurement and
  an equivalent runtime-owner control.
- `pyfluid.relativistic_exterior` is the sole physical exterior identity for
  relativistic matter. One runtime `RelativisticExteriorState` carries either
  the existing positive atmosphere closure or canonical exact vacuum through
  initial data, recovery/projection, face fluxes, AMR transfer, physical-ID
  materialization, output, and restart. Do not infer vacuum from zero-valued
  EoS floors, add a reconstruction/EoS template axis, or create a second
  fallback ladder. Valid canonical zero is not a C2P failure; replacing a
  failed nonzero state remains counted and warned. The policy is immutable
  checkpoint and physical-ID identity, and legacy physical-ID artifacts imply
  atmosphere. RMHD rejects vacuum until an electromagnetic exterior closure
  exists.
- A tabulated EoS is loaded and validated once into immutable managed storage;
  pointwise CPU/GPU consumers receive the same small pointer/value view. Adding
  a table backend must not instantiate unsupported physics, layout, topology,
  or characteristic products. The table manifest states provenance and
  continuation policy, while runtime compatibility uses a fingerprint of the
  normalized physics data rather than a file path.
- Boundary value data and stable kind identity live in `boundary_data.hpp` and
  `boundary_kind.hpp`. The public `boundary.hpp` provides launch declarations
  and small forwarding wrappers; it must not import configuration, field
  descriptors, EoS formulas or generated numerical kernels. Configuration and
  schema parity resolve in `src/framework/config_runtime.cpp`. Pointwise
  inflow/analytic helpers and typed FillPatch callbacks live in
  `boundary_point.hpp`, included explicitly by device consumers. Whole-array
  state/patch-union fills and registered component-layout instantiations have
  one per-dimension compiled owner in `src/numerics/boundary.cpp`. Keep kernel
  device callees visible there for the non-RDC GPU contract. The separate
  `origin_boundary.cpp` retains hierarchy/nonlocal origin topology ownership.
- The standard suite runs the generated-equation numerical contracts after
  the CMake code-generation dependency has established freshness. Exact
  byte-for-byte regeneration of every expensive SymPy artifact belongs in the
  full suite; do not make ordinary solver iteration repeat that exhaustive
  check.
- Catalog emission and symbolic physics-kernel emission are separate CMake
  dependency owners. Catalog emitters and catalog-only schemas must not
  invalidate the expensive physics-kernel provenance; symbolic kernels retain only
  the catalog inputs they actually consume, such as the field schema used to
  define conserved components.
  Diagnostic, Z4c and relativistic-fluid symbolic kernels also have separate
  freshness records. Their actual shared equation dependencies still propagate changes
  (for example, diagnostics consume Z4c equations); unchanged generated text
  keeps its timestamp. Do not rejoin these into an unconditional all-family
  generation step during ordinary builds.
- CMake declares each generation family's inputs and complete outputs once.
  All build profiles share its ignored source-owned provenance under
  `pyfluid_amrex/.cache/codegen/`; dimension, precision and backend do not own
  separate symbolic outputs. Cheap content checks run before importing SymPy.
  Changed inputs, missing outputs, changed output contents, or changed CSE
  options require genuine generation. Provenance also records the running
  Python implementation/version and installed SymPy/mpmath versions, read
  through package metadata before symbolic imports. Whole-project dependency
  files do not invalidate symbolic output for unrelated package changes;
  CMake invokes `uv run --locked` to reject stale resolution without rewriting
  the lock. A missing or older provenance record requires genuine generation,
  never a bootstrap from timestamps. A per-family OS lock has a 300-second wait limit and
  rechecks freshness after acquisition. Writers stage the complete family,
  verify its declared outputs and unchanged inputs, atomically replace changed
  files, and publish provenance last. Unchanged text retains its timestamp.
  Preserve independent family dependencies and this publication contract when
  extending the generator; do not add a profile-local freshness registry.
- Orthogonal coordinate geometry is declared once in
  `codegen/pyfluid_codegen/coordinates.py` as coordinates and scale factors.
  The symbolic tensor/finite-volume owners derive metric measures, differential
  identities, and curvature sources from that frame while the physical
  conservation law remains coordinate independent. Add a coordinate or
  projection by extending those high-level owners and their generated tests;
  do not hand-copy the HD/MHD equations for each chart. Solver algorithms that
  are not yet expressed through this symbolic layer, notably the current XCTS
  elliptic discretizations, remain explicit numerical owners rather than being
  described as generated merely because their materialization is generic.

## AMR And Output Infrastructure

- `GravityFields` owns potential/gradient allocation, fixed-field cache validity,
  and Poisson operator setup. Consumers receive const field pointers. The
  hierarchy driver supplies the valid-density RHS and owns stage timing;
  gravity energy/flux registers remain integrator state. The compiled owner
  does not receive the solver object or mutate fluid state.

- `src/amr/regrid_lifecycle.{hpp,cpp}` owns regrid attempt bookkeeping and
  aligned cadence/restart clocks. AMReX callbacks report changed levels; a
  completed attempt returns the first changed level or no change. The driver
  owns transfer/synchronization and decides when it is legal to regrid.
  Descendant attempts do not consume the whole-hierarchy cadence. This compiled
  owner has no AMReX, field-storage, configuration, or generated-kernel dependency.

- Repository MPI launches go through `validation_common.py`. Each launcher
  owns a new POSIX process group containing its ranks; a timeout, fatal-rank
  marker, nonzero launcher exit, or runner exception/interruption terminates
  and reaps that complete group while retaining combined rank output. Do not
  add raw `subprocess.run` MPI call sites that can orphan ranks after the
  launcher exits.

- One coordinate-independent physical implementation serves every executable
  dimension. Dimension branches are limited to storage, centering, topology,
  or an explicitly documented unsupported feature.
- One selected coarse-fine state interpolater serves level creation, regrid,
  FillPatch, subcycling, and repair seeds. Fluid state is refluxed and averaged
  down; CT uses edge-EMF reflux and face average-down; nodal spacetime follows
  its nodal synchronization contract and is not refluxed.
- Automatic AMR tagging uses one runtime policy declared in
  `schema/methods.toml`, including the policy's default threshold.
  Compact-object policies suppress floor-level matter noise and select density
  support and/or strong-field `abs(chi - 1)` support explicitly; they do not
  chase small outgoing gauge tails. Broad Lohner curvature tagging remains
  opt-in for shocks and instability studies.
- `amr.initial_refine_fixed_levels = 0` makes an initial box a startup seed
  that is unioned with the selected automatic metric. Use that contract when
  a physical-ID target needs guaranteed near-object resolution plus separate
  matter-support refinement. A positive fixed-level count deliberately makes
  the box the complete initial hierarchy and suppresses automatic tags at the
  startup instant; do not use it merely to seed a horizon if a disk or star
  must also be resolved.
- A physics-system admissibility restriction augments that selected
  interpolater in its state-transfer owner; it is not a second interpolater or
  a post-transfer atmosphere repair. Keep relativistic conserved components
  and a coarse parent's child family coupled so a conservative base retains its
  parent average. AMReX may request only part of one child family for a FAB or
  ghost region, so the owner must interpolate and limit the complete
  sibling-aligned family in private scratch before copying the requested
  subset. A partial-region call must not change the resulting child values.
- Face-centered magnetic fields are canonical when CT is active. Each RK stage
  constructs its face field from the saved step-start field and the appropriate
  stage EMF combination. Primitive recovery/reconstruction substitutes that
  face average without mutating the stage momentum or energy. A per-level
  ownership mask admits valid, same-level, and periodic ghost context; AMReX
  FillPatch cell state remains authoritative in uncovered coarse-fine ghosts.
  After the RK-integrated face update, synchronize cell-centered magnetic
  components and apply final physical acceptance. The latter is a backstop,
  not a substitute for stage-consistent face evolution.
- After hierarchy synchronization or regrid, refresh the metric coupling and
  perform final relativistic acceptance before exposing or writing the state.
  Newly created or remade levels must keep current and old-time storage
  semantically consistent after that acceptance step.
- Z4c centering transfers select one positive-scalar representation from
  `z4c_centering_representation.hpp`: logarithms of chi/lapse are interpolated
  and exponentiated; every other component uses the identity. Storage and
  checkpoint values remain ordinary physical variables. The generic
  `centering_representation.hpp` and `centering_transfer.hpp` know no Z4c
  component IDs. Accepted and candidate nodal-to-cell transfers select the
  same representation, order, and excision region; ordinary and exterior
  face readers use that same representation. Physical point targets, including
  MPI/periodic ghosts, use the representation. Outside-domain targets preserve
  the raw polynomial extension owned by the boundary operator: wide derivative
  ghosts are not automatically admissible pointwise metrics. Actual metric
  consumers must still pass the existing signature checks. This changes neither the
  selected polynomial stencil nor its order, and is not a failed-transfer
  repair. Invalid sources and tensor-signature failures still fail checks.
  Array-view adapters captured by GPU kernels must own their lightweight
  views by value, not references to host-stack handles.
- `eulerian_matter_access.hpp` owns component packing/unpacking of the shared
  Eulerian matter state. Consumers needing only access must not include the
  heavier fluid-to-spacetime coupling kernels just to load that state.
- CPU-tiled numerical kernels launch over `MFIter::tilebox`; `validbox` is
  whole-FAB metadata and must not be used as the `ParallelFor` region inside a
  tiled loop. This also applies when the iterated `MultiFab` is face-centered:
  `tilebox` owns each tiled face once. Kernels that write ghost regions use
  `MFIter::growntilebox` so adjacent internal tiles do not write overlapping
  grown boxes. Device-side temporary FABs use `The_Async_Arena` or retain
  AMReX `Elixir` ownership through asynchronous kernels and copies. Ordinary
  arena deallocation does not acquire stream ordering merely because a FAB
  was consumed by a GPU kernel. CT edge-integral temporaries select the async
  arena at allocation; register kernels retain their ordinary field views.
  The architecture check rejects a
  whole-FAB numerical launch inside `TilingIfNotGPU`; the 2D/3D boundary-
  tiling gate covers ordinary, patch-union, relativistic, and CT ghost fills
  with forced small tiles.
- AMReX's all-node physical-boundary path does not apply `BCRec` as the
  cell/face path does. `Z4cFillPatchBoundary` therefore owns both polynomial
  extrapolation and catalog-parity reflection. Tensor-product corner fills
  read physical-domain nodes, not physical ghosts being written concurrently;
  thin reflected directions fold repeatedly into the domain. Six-point
  Lagrange weights share `interpolation_weights.hpp` with nodal AMR transfer;
  globally shorter lines use their available nodes. Evaluate tensor products
  one direction at a time to avoid unnecessary corner-weight multiplication.
  The existing nodal halo supplies the same stencil across small boundary
  boxes; do not lower the order according to MPI decomposition.
  Reflection-odd valid-plane values are explicitly
  zeroed. Constraint masks admit reflected stencils but exclude outer ones.
- Z4c ghost filling and an outgoing RHS condition are separate operations.
  `z4c_outer_boundary.cpp` owns face selection, boundary records, and the
  optional `spacetime.outer_boundary=sommerfeld` dispatch after bulk RHS/KO
  assembly, shared by every stage and level. `extrapolate` retains the bulk
  RHS. Neither option changes the fluid boundary. Principal lapse/shift speeds
  share `z4c_characteristics.hpp` with the excision guard and local CFL
  frequency calculation; runtime choices do not add a template-product axis.
  Sommerfeld is approximate, not a constraint-preserving boundary.
- Validate rotating-object reflection compatibility once during common solver
  startup, before either ID or evolution. In Cartesian coordinates every
  reflection/symmetry plane must pass through the affected object's center and
  its normal must follow each nonzero rotation/disk axis; black-hole spin has
  the corresponding axial-vector rule. A spinless isolated hole may use an
  octant, but attaching a rotating disk leaves only the disk-equatorial plane.
  This policy applies equally to fresh analytic data, constraint-solved data,
  and loaded physical-ID artifacts.
- State and auxiliary GF snapshots use generated field descriptors. Auxiliary
  output normally inherits the state GF cadence; explicit compute/write
  overrides remain available. Scalar diagnostics and checkpoints retain their
  independent cadences.
- Relativistic diode boundaries own both ghost transport and the physical-face
  trace pair in `relativistic_diode.hpp`. The face uses one interior-derived
  primitive with no incoming coordinate transport; the selected generated
  flux/Riemann owner retains pressure and magnetic stresses. Normal fluxes and
  every candidate-correction tier consume that same boundary contract. Do not
  clamp only the final mass flux, zero the entire flux, or leave a low-order
  correction path with a different boundary condition. The comparatively large
  causal velocity completion is shared/no-inline rather than cloned into the
  reconstruction template matrix. This condition does not make a boundary
  nonreflecting and is independent of the spacetime boundary.
- Shared TSV writers escape embedded NUL, tab, carriage-return, and newline
  characters rather than corrupting row structure. Canonical validation result
  paths belong to the complete declared matrix; filtered or exploratory runs
  write distinct result names and must not overwrite canonical evidence.
- Ignored output has three retention tiers. `runs/current/` is reproducible
  working state and may be compacted after an exact inventory;
  `runs/baselines/` stores named bounded validation/performance evidence; and
  `runs/final_done/` stores deliberately curated finished media, compact run
  reports, and expensive accepted reusable physical-ID artifacts. Preserve
  canonical artifact prerequisites referenced by tracked inputs and bounded
  recent histories. Raw GF snapshots, intermediate checkpoints, failed probes,
  duplicate movies, and superseded scan points do not become durable merely
  because they are large. A prune must state exact retained roots, deleted
  size, and recoverability before removing data.

## Shared Validation Contracts

Validate shared implementations, not the Cartesian product of every object,
EoS, disk, coordinate family, and mesh option. Keep four distinct layers:

1. **Algebra and geometry:** compare generated continuum operators with an
   independent reference for each coordinate family. The Cartesian embeddings
   and chain-rule oracle in `tests/test_coordinate_contracts.py` check the
   scale-factor operators for arbitrary local scalar, vector, and tensor
   derivatives. These identities include symmetry reductions by setting field
   derivatives to zero, while retaining moving-basis terms and three physical
   components. Executable axis mappings still need dimensional-consistency
   tests. Current symbolic coverage is regular Euclidean orthogonal charts,
   not coordinate singularities or arbitrary curved-spacetime operators.
2. **Discrete operators:** each numerical owner needs manufactured/exact
   convergence checks including its boundary closure, centering, and supported
   order. Pole/origin regularity, mapped/adaptive meshes, and coarse-fine
   interfaces need their own focused controls. A correct continuum identity
   does not validate a stencil or its assembled solve.
3. **Field handoff:** test the shared representation contracts (tensor basis,
   reflection, centering, save/load, and same/different grids) with synthetic
   fields. Coordinate changes are algebraic; interpolation is approximate and
   must converge. Use existing field and method catalogs, not a parallel
   registry of test-only capabilities.
4. **Physical owners:** retain equilibrium, constraints, admissibility,
   conservation, and nonlinear-coupling checks for the actual ID/evolution
   algorithms. A few independent physical cross-chart anchors per algorithm
   family check assembly and boundary assumptions; repeating every anchor for
   every parameter product is unnecessary unless a separate implementation or
   demonstrated failure mode warrants it.

"Once" means one maintained regression contract per shared implementation,
rerun when its dependencies change, not permanent certification. Sources and
oracles generated from the same formula can share a bug; keep independent
geometry/analytic references. Different discrete grids need not agree bitwise
at finite resolution, but must approach the same physical solution. Do not
retire failure-driven regressions merely because symbolic checks pass. Add
these gates to the existing suite; no new testing framework or compiled
coordinate-product axis is required. The symbolic coordinate gate runs in
standard/full, not in the sub-three-second quick suite.

## Independent Numerical References

External codes are comparison baselines, not production initial-data or
physics implementations. A paired comparison records an equivalence statement
for equations, units, EoS, state conventions, domain, boundaries, initial data,
and final time; reports unavoidable discretization differences; begins with
uniform-grid controls; and compares common physical points or integral
diagnostics rather than storage ordering. Preserve each code's input, build
identity, stdout, resolution, conservation/constraint results, and wall time
under `runs/current/comparisons/<code>/<case>/`.

An external numerical reference complements but does not replace PyFluid's own
analytic, conservation, restart, AMR, and performance gates.

## Reusable Initial Data

A reusable initial-data artifact is not an evolution checkpoint. It stores a
provider-independent physical core after the initial-data solve:

- cell-centered fluid primitives;
- nodal ADM spatial metric, extrinsic curvature, lapse, and shift;
- face-centered magnetic data when present; and
- source hierarchy, EoS and physical parameters, diagnostics, schema version,
  and provenance.

Artifact publication is one transaction. All ranks write portable fields and
optional exact caches into a hidden sibling staging directory; the I/O rank
writes the complete manifest, including solved component diagnostics, checks it
against the generated metadata catalog, and atomically renames the directory
to its final path. A sequence index may append a model only after that rename
and the following rank barrier. A visible artifact is therefore complete;
`Header.meta` must never be amended after publication. A leftover
`.pending` sibling identifies an interrupted transaction and is rejected
explicitly rather than guessed or silently reused.

The manifest may carry indexed runtime component records. Today these adapt
the existing single central TOV/RNS/black-hole owner and optional disk owner;
they add no object-count template axis and do not create another field layout.
Component origins and axes are transformed with the same runtime rigid map as
the portable fields and are preserved when a completed artifact is re-saved.
A future N-object builder may request component poses, free-data seeds, matter
sources, and support from the same runtime descriptions, but arbitrary sums of
accepted GR fields are only labeled initial guesses/test fields. One global
constraint/fluid owner must accept a self-gravitating multi-object ID before it
is published as physical initial data. Deprecated manifest spellings may
remain declared as read-only compatibility aliases; the current writer emits
only the canonical key.

For a generated stellar magnetic seed, the edge-centered vector potential is
transient construction scratch rather than a persistent GF. The owner takes its
discrete curl with the production CT topology, normalizes against the current
metric, and stores only the schema-owned canonical face magnetic fields plus
seed identity/parameters in metadata. A coupled constraint solve invokes this
owner after each matter update and before assembling RMHD stress-energy. This
makes the gravitational constraints consistent with the seeded field; it does
not imply magnetohydrostatic Euler/Grad--Shafranov force balance. The current
pressure-poloidal owner is deliberately restricted to fresh, single-level,
3D Cartesian, solve-only stellar XCTS. AMR requires one edge-potential
prolongation/synchronization owner before it may be admitted.

The artifact records `magnetic_force_balance=not_solved` for this seeded
route. This is an explicit semantic boundary: the gravitational constraints
include the magnetic stress, but no magnetized Euler or Grad--Shafranov
force-balance equation has been solved.

The separate `poloidal_grad_shafranov` owner is the force-balanced stellar
route. It solves the pure-poloidal conformally-flat vector Grad--Shafranov
equation together with the magnetic Bernoulli first integral inside the same
stellar/XCTS fixed point. Its catalog parameters select
`M(A_phi)=k_pol(A_phi+xi A_phi^2/2)` and nonlinear relaxation; the toroidal
current is zero. Cell-centered and edge-centered potentials are private solver
scratch. The persistent CT owner is the Valencia density
`mathcal B^i=sqrt(gamma) B^i`; portable primitive fields contain physical
Eulerian `B^i`. The shared field catalog keeps the representation-neutral
label `B^i`; the route-specific artifact metadata is authoritative. Artifacts
record `magnetic_force_balance=grad_shafranov` and
`magnetic_face_representation=valencia_densitized`. Schema-v1/v2 RMHD artifacts
lack the latter identity and are rejected rather than guessed.

Schema-v3 magnetic construction metadata is a strict continuation contract:
the seed/equilibrium names must exist in the generated method catalog, the
force-balance and face-representation identities must match the selected
physical system, and every construction parameter must be present and finite.
An exact-layout RMHD solve-only load may reserialize a saved magnetic artifact
without rerunning its constructor. That path must load the portable primitive
core as well as any exact evolution caches and preserve the source magnetic
metadata and reflection closure. Exact caches accelerate restoration; they
never replace the portable physical fields.

The first force-balanced owner is deliberately fresh, single-level, 3D
Cartesian, cold-barotropic TOV/RNS, coupled-XCTS, and solve-only. It supports
an equatorial reflection plane normal to its magnetic/rotation axis. A
transverse Cartesian reflection is not a symmetry of an oriented axial
magnetic moment and is rejected. AMR, curvilinear Grad--Shafranov solves,
toroidal/mixed fields, magnetized disks, and changed-grid CT materialization
need their own validated discrete owners; catalog presence does not imply
those combinations. A constructed magnetic artifact is also rejected when a
load changes the source IC, continues TOV into RNS, or composes a new torus:
those operations change matter/geometry and require a target-grid magnetic
equilibrium owner rather than retaining stale force balance.

The container is deliberately a superset. In addition to the portable
physical core, it may contain optional consumer-specific sections such as
exact-layout evolution caches and sequence-continuation seeds. Evolution may
discard continuation-only fields, while a sequence continuation may discard
evolution-only caches. These fields are accelerators, not the physical
interchange contract, and their roles must be explicit in the artifact
metadata. Uniform Cartesian artifacts may carry exact-layout caches for
lossless direct evolution loading. Reduced mapped artifacts deliberately omit
those evolution caches: an exact same-layout continuation restores their
portable primitives and nodal ADM fields without interpolation, while a
different target uses the common materializer.

TOV, rotating-star, binary, and external providers must all produce this same
field contract. The artifact must not encode timestepper, reconstruction,
subcycling, output cadence, or other mutable evolution choices.

Black-hole providers use the same contract. The exterior ADM fields, the
horizon-interior policy (`exterior`, complete-grid `filled`, or true
`excision`), mass, origin, and Cartesian spin vector are artifact identity.
An interior fill is data on an active cell and must not be reinterpreted as an
excision mask. Conversely, true excision requires explicit active-domain and
inner-boundary ownership in both the ID solver and evolution hierarchy.
Black-hole spin, stellar rotation axis, and disk axis remain separate metadata;
this preserves aligned reduced solves and lets the full-3D owner retain a
physical relative star/hole--disk tilt through the same materializer.

Do not encode puncture, trumpet, smooth fill, and excision as four independent
compile-time axes. Expose three runtime ID-to-evolution interior policies:
true excision, complete-grid puncture/trumpet, and complete-grid smooth fill.
The underlying active-domain ownership remains only excised versus complete
grid. A trumpet is a puncture initial slice, not another interior policy. For
an exterior XCTS/COCAL solve, smooth filling is a materialization/handoff step
after the solve and must never modify the saved exterior solution or its
continuation state. Generated catalogs may enumerate supported combinations,
but each combination must route through these shared owners rather than
instantiate another physics implementation.

Fixed-background versus dynamically evolved spacetime is a separate runtime
axis. Prefer evolved, matter-coupled spacetime for production science; retain
fixed backgrounds as analytic/test-fluid limits and require convergence toward
that limit as the matter backreaction is reduced.

The native spherical Schwarzschild-XCTS owner has both reduced `r theta` and
full-storage periodic `r theta phi` layouts. In 3D, the Hamiltonian and lapse
use the genuine spherical scalar operator while the momentum equation remains
the axisymmetric toroidal sector. This is an ordinary 3D storage/operator
compatibility owner for aligned data, including the `N_phi=1` degeneracy; it is
not the general three-component nonaxisymmetric spherical vector XCTS system.
Changed-grid vacuum shell solves reconstruct their analytic conformal iterate
from mass and apparent-horizon boundary data rather than treating interpolated
Z4c fields as authoritative conformal unknowns.

An apparent-horizon surface is a diagnostic object sampled from the hierarchy,
not a second spacetime field layout and not an implicit excision mask. The
surface solver owns its angular/volume work grid and may persist its radial
shape for tracking; it consumes Cartesian metric and extrinsic-curvature data
through the common hierarchy interpolation boundary. Exterior-only constraint
diagnostics must require the complete derivative stencil, not only its center,
to lie outside the selected horizon. Catalog-owned `apparent_horizon_mask` and
`excision_mask` auxiliary fields are derived views of that surface. They may
drive diagnostic output; the surface's signed-distance view may drive a
geometric AMR tagging band. They do not share solid-obstacle semantics and
must not alter active-cell, flux, FillPatch, or boundary ownership by
themselves. Only an explicitly selected evolution owner may convert a found
surface into inactive cells and inner-boundary closures. The fixed-Kerr matter
excision owner does so at runtime: it owns the active mask, one-sided fluid
flux, zero interior RHS, recovery/CFL/diagnostic/AMR exclusion, and inner
conservation ledger. Evolved-Z4c excision instead shares one fixed coordinate
sphere (`spacetime_excision_region`) across matter, derivative/KO closures,
centering and face-metric transfers, and nodal AMR interpolation. The AH is
a containment check, not a moving-boundary controller. Incoming physical or
gauge characteristics reject the configuration. A runtime derivative policy
feeds the existing generated Z4c RHS; excision is not another template product
for the complete physics equations. Inactive primitive caches contain finite
scratch without C2P or conserved-state edits; exterior fluid stencils must
not consume it. Restart metadata fixes the interior owner and cut radius.
The baryon-ledger validity flag is separate from the joint fluid-energy flag:
evolving geometry invalidates the latter but not baryon conservation.

Debug Z4c stage checks belong in the shared stage/coupling owner, not in each
integrator or reconstruction. Check finiteness around projection, ghost fill,
centering transfer, and RHS completion; check metric positive definiteness
before projection and in the transferred fluid metric halo. The nodal
projection owns its explicit lapse/chi floors, so its precheck must allow
those pending floors. Extra excision derivative-only scratch must be finite,
but is not itself a physical metric. These checks never repair data and add
no grid sweeps when the debug options are disabled.

The optional radiative outer RHS is one compiled stage owner. Its non-wave
correction reads an interior bulk-RHS donor displaced off every incident open
face, including edges/corners. Never read an RHS value concurrently being
replaced by another boundary kernel. The current nodal donor is local to its
FAB; no extra full-volume state, RHS synchronization, or analytic background
is needed. Frozen-gauge and other unowned components keep their bulk RHS.

State output treats immutable fixed spacetime as run context: write one
multilevel fixed-spacetime snapshot with the first fluid plot, while evolved
spacetime is written at every fluid-plot cadence. Checkpoints continue to
carry every restart-required spacetime field in either case.
AMReX may request the initial hierarchy's tags before the step-zero
apparent-horizon search has produced a surface. In that pre-surface phase,
retain the configured initial seed tags instead of inventing or asserting a
mask. The startup owner must still find the horizon and construct masks on all
created levels before matter projection, preflight, CFL evaluation, or the
first evolution stage; every runtime regrid then consumes the tracked mask.

An event horizon is a history-consuming postprocessing object, not a
slice-local auxiliary GF. Its future offline owner consumes a time-ordered,
schema-declared Cartesian spacetime history and evolves a null surface
backward. A general multiple-hole representation must permit topology change;
surface meshes and generator histories are derived artifacts and cannot be
used as evolution masks. Do not register an event-horizon runtime capability
until the history writer, temporal interpolation contract, backward operator,
and analytic validation gate exist together.

Optional equilibrium matter components use that same physical field contract;
they do not create a parallel artifact or materializer. The construction owner
may compose a central object and one or more disjoint matter supports before
writing the combined primitive and ADM fields. Component identity, rotation-law
parameters, support, and component diagnostics are retained as metadata, while
system-wide ADM/baryonic/angular-momentum diagnostics remain explicitly
system-wide. A gravity-disabled `test_fluid` component means its stress-energy
is omitted from the constraint solve, not that the stored rest density is zero.
Every self-gravitating component must participate in the coupled matter/metric
fixed point; adding it after convergence is only the test-fluid contract.
`equilibrium_torus` means the central-object and disk axes admit the stationary
circular first integral. `tilted_torus` means a deliberately nonstationary,
constraint-consistent circular seed: the fluid is approximately stationary and
axisymmetric about its own axis while the coupled spacetime is fully 3D. It
requires the native 3D Cartesian coupled-XCTS owner. The portable artifact may
then be evaluated on Cartesian or full `r theta phi` targets. A 3D curvilinear
target may not continue the nonlinear matter solve or add a torus until a
native curvilinear elliptic owner exists.

Initial-data centering transfer treats periodic and reflection ghost values as
valid stencil support. Bounded one-sided closure is reserved for true physical
nonreflection faces. Otherwise a reduced-domain matter source can solve a
different discrete elliptic problem even when its reported residual is tiny.
Keep this policy in the initial-data transfer owner rather than broadening the
generic evolution-centering template surface.

The portable physical-ID reader applies the same rule to a reflected source
artifact. It fills one parity-aware source ghost layer for cell-, face-, and
node-centered fields and explicitly authorizes the hierarchy interpolation
stencil to cross only metadata-declared reflection planes. A reflected
half-cell strip is source support, not atmosphere or asymptotic exterior.
Materialization preserves the saved physical fields; it does not manufacture
unresolved scales or project the interpolated fields back onto the constraints.
For evolution on a finer target, solve the source ID at least as finely as the
target's physically important region or perform an explicit target-grid
constraint polish once such an owner is selected and validated.

`initial_data.output_file` writes this artifact at step zero.
`initial_data.input_file` loads it. `initial_data.solve_only=1` performs the
configured initial-data construction, runs the normal relativistic preflight,
writes the artifact, and returns before allocating evolution flux-register
work. For a loaded spherical-shell XCTS black hole, `solve_only=1` always
re-solves the configured constraints even on an identical layout; ordinary
evolution loading (`solve_only=0`) retains the exact-cache fast path. This
prevents an area-only historical artifact from being copied and mislabeled as
a newly constraint-solved physical-spin model. The source chart is part of the
artifact rather than an evolution requirement: the current schema-v3 records
source dimension, coordinate family, ordered active axes, periodic/reflection
closure, and `mesh_orthonormal` tensor basis.
Physics and physically relevant EoS data remain mandatory. On an identical
hierarchy, optional exact caches start evolution directly. Otherwise the
reader maps physical target points into the source chart, interpolates the
portable core, transforms catalog-declared scalars, polar/axial vectors, and
symmetric tensors through an orthonormal Cartesian basis, reconstructs
conserved and Z4c state, and applies the target preflight. For a saved
rotating-star artifact with the same rotation law and a larger configured
target angular velocity, the same materialized state continues XCTS; a
matching target starts evolution without another XCTS solve. The ordinary
continuation identity includes the differential-rotation length and, for the
q-power family, its exponent. The artifact must also restore the converged
Euler first-integral injection energy; recomputing it from the original TOV
seed changes the nonlinear problem.
An aligned coupled equilibrium torus follows the same rule for a nonconstant
rotation law: unchanged law/EoS/geometry parameters plus a larger same-sign
disk angular velocity reuse the saved matter and spacetime as the next XCTS
seed. A matching disk configuration materializes or evolves without another
solve. The artifact records the native black-hole throat area, irreducible
mass, proper equatorial/polar circumferences, and Kerr-assumption horizon mass
and spin. Axisymmetric native throats additionally record quasi-local `J_H`,
Christodoulou mass, and signed dimensionless spin. Reports retain `M0/Mirr`,
circumference-derived `M0/M_C`, and Christodoulou `M0/M_BH` as separate
quantities. Every saved continuation model owns
these diagnostics for its accepted geometry; it must not inherit them from
the seed artifact. Keep the Kerr qualifier because a generic distorted
horizon need not obey the vacuum-Kerr relations.
The q-power catalog parameter follows
`j=A^2 Omega[(Omega_c/Omega)^q-1]`; it is not the conventional radial disk
shear exponent. In an asymptotically flat region with `1 <= q < 2`, the latter
is `q_Omega=2/q` in `Omega proportional to R^-q_Omega`. Persist and compare the
catalog parameter, while reports may derive the conventional exponent for
interpretation.
Rematerialization must carry forward the complete rotating-star continuation
summary (injection energy, surface diagnostics, charges, and spin) and disk
component summary, not only the fields. Otherwise a valid evolution artifact
silently ceases to be a valid sequence-continuation artifact.

For a tabulated EoS, both physical-ID and checkpoint metadata retain the
backend/version, normalized-table fingerprint, point counts, causality
truncation flag, baryon-mass normalization, and thermal-extension parameters.
The source directory is provenance only: relocating byte-equivalent table
files must preserve compatibility. A changed normalized table, continuation
contract, or thermal gamma must fail before materialization or restart. Do not
substitute a display model name, DOI, archive checksum, or path for the actual
physics fingerprint.

The ordinary
in-process sequence accepts an explicit strictly ordered
`initial_data.rotation_continuation_targets` list and retains live fields
between targets without artifact I/O. A multi-target rotating sequence writes
every accepted target by default; `initial_data.save_sequence=0` disables
this. A one-model TOV or RNS construction is not automatically saved. A
one-target continuation from `initial_data.input_file` is saved because it is
part of a persisted sequence. `initial_data.sequence_output_dir` overrides
the automatic location and also explicitly requests output for a one-model
rotating solve.

The continuation step count is shared by stellar and disk rotation. Loading a
compatible artifact initializes each active component from its saved angular
velocity, linearly advances changed components to their configured targets,
and writes both angular velocities into every sequence index and artifact.
This keeps one live-field continuation owner and avoids a separate disk-only
solver path.

An initial-data solver may use the coordinates and dimension natural to its
symmetry, independently of the evolution hierarchy. Artifact metadata must
describe the physical coordinates of the stored samples exactly: a transformed
solver coordinate must not masquerade as `theta`, `r`, or another canonical
axis. A spherical axisymmetric artifact therefore stores a genuine ordered
`r theta` lattice even when the solver uses a weighted angular operator.
Matching source and target points may take the exact-layout path; different
spacing, extent, dimension, or coordinate family uses the same materializer
and the field's declared origin, exterior, and reflection closure.

Keep three representations independent:

1. the solver-private ID representation, which may be uniform, mapped,
   multidomain, modal, block-adaptive, or symmetry-reduced;
2. the portable physical-ID representation, which stores enough chart,
   hierarchy/mapping, basis, centering, and exterior metadata to evaluate the
   physical fields; and
3. the evolution hierarchy, which owns its own Cartesian resolution, extent,
   AMR, storage, and final primitive-to-conserved conversion.

The handoff is an evaluation contract at physical target points, not an AMReX
box-layout contract. A materializer must not depend on the ID solver matrix,
iteration storage, logical spacing, or refinement bookkeeping. Conversely, an
ID solver must not allocate evolution flux/register state. The portable schema
supports uniform AMReX levels and versioned explicit coordinate faces.
`coordinate_layout=explicit_faces_v1` stores every nonuniform coordinate face
used to evaluate cell-centred and nodal fields; logical AMReX boxes remain
storage indices and never masquerade as uniform physical spacing. A future
modal or block-adaptive representation must likewise version and store its
evaluation coordinates or coefficients before production.

For relativistic scratch construction with stored spacetime, the final fluid
conserved state is formed only after the authoritative nodal ADM/Z4c data have
been centered onto the evolution cells. Initial fluid primitives may be sampled
earlier, but they must be resampled/preserved as primitives and converted
against that final cell metric. Converting against a placeholder analytic
metric and later replacing the stored spacetime violates the physical-ID
handoff even if weak-curvature tests happen to recover it.

`initial_data.radial_grid=adaptive` selects the shared stellar radial-map
owner. TOV uses spherical `r`; circular RNS uses spherical `r theta`. The map
keeps regular uniform spacing through a conservative enclosing matter region
and a geometrically stretched, adequately resolved gravity exterior. It is
fixed during one nonlinear solve and rotation sequence so live fields never
need an in-solve regrid. Uniform remains the explicit work-precision control.
This is star-adapted mapped spacing, not residual-driven AMR: binaries and
multiple compact objects still need a block/multidomain design.
`initial_data.adaptive_source_radius_factor` and
`initial_data.adaptive_exterior_fraction` are bounded solver-private policy
controls; the saved explicit faces, rather than either knob, are authoritative
for continuation and materialization. The conventional RNS inputs keep the
uniform source ball through 1.8 times the spherical seed radius and reserve one
third of radial cells for the gravity exterior. TOV retains the 1.5 and
one-third defaults. Changing either map for a reduced solve-only RNS load
forces a new discrete XCTS solve even when angular velocity is unchanged;
same-map loading remains a direct sequence continuation.

Angular resolution remains one uniform `theta` count on the rectangular
separable grid; it is not AMReX angular AMR. Matched near-shedding solves make
64 cells the conventional sequence/exploration choice, 128 a practical
endpoint grid, and 256 the angular-reference control. A radius-dependent mode
cap or coarse exterior angular domain must own a genuine modal/multidomain
interface and preserve the coupled residual; do not represent it as a
nonuniform theta coordinate or silently change `amr.n_cell` during a solve.

Matter support and gravity extent need not be the same computational domain.
An ID method may stop its matter representation at the enthalpy-zero surface
while continuing metric fields through exterior and compactified domains. The
portable/evolution materializer owns the requested exterior atmosphere and
asymptotically flat completion. This separation does not license a reset of an
invalid interior solution or omission of surface matching conditions.

Cell-, node-, face-, and edge-centered arrays may share one hierarchy and
coordinate-map descriptor, but they are different degrees of freedom and are
not aliases of a single base array. Their index extents, ownership at shared
faces/edges, interpolation, synchronization, and conservation identities
remain distinct. Nodal derivatives need valid neighbor data across AMReX box
boundaries, and portable nodal fields must be synchronized to one value on
overlapping valid nodes before output. A coordinate/basis change is algebraic;
these centering, parity, axis, origin, and stellar-surface contracts are the
nontrivial part of reduced spherical-to-Cartesian materialization.
Remove duplicated topology/metadata and temporary caches where safe;
do not collapse physically distinct centerings into one storage view.

Interpolation ownership includes target points outside the source sample
hull. A cell-centred radial source has a half-cell gap at the regular origin;
clamping there is interior regular interpolation and must not create
atmosphere. The same rule holds for the interior half-cell strip adjoining a
declared reflection plane, such as the RNS equator; a finer target must not
turn that regular strip into physical exterior. Beyond the saved outer radius,
the loader clamps only the source sample and applies the selected physical
exterior at the original target point: exact isotropic-Schwarzschild exterior
for spherical TOV ADM data,
component-appropriate asymptotically flat falloff for general ADM data, and
the configured target relativistic exterior for TOV/RNS fluid. It must not
copy a source-boundary atmosphere value or use exact floating-point equality
to classify stellar matter support.

A periodic cell-centred direction has no physical half-cell exterior at its
coordinate seam. The portable evaluator wraps both interpolation neighbors
across that seam, including when explicit coordinate faces are stored, and it
must not clamp the query or apply an exterior/atmosphere policy there.
Node-centred periodic data may include the duplicated endpoint and follow their
own nodal extent. Every newly portable periodic chart needs a seam-focused
interpolation check; a phi-invariant source is a particularly sharp control
because any seam-localized matter loss is unambiguously infrastructure error.

Automatic sequence output is organized under the run directory by EoS,
rotation law, and fixed central density. Named EoS presets and differential
law parameters are recorded in the path and `sequence.meta`; `sequence.tsv`
schema v5 indexes accepted targets together with the converged injection
energy, surface diagnostics, ADM/baryonic masses, angular momentum, and spin.
Model names include prescribed `omega`, base-grid resolution, and hierarchy
level count. Every model is the same losslessly
compressed, directly evolvable physical-ID artifact—not a sequence-only file.
Artifact metadata, including `eos_preset`, remains the compatibility authority;
directory and model names are discovery aids.

The artifact also records `spacetime_initial_data`. A loaded fixed-Minkowski
artifact may use the analytic flat-spacetime kernels only when both that source
metadata and the target configuration select Minkowski. Curved, constraint-
solved, or legacy artifacts without this metadata retain the stored-metric
path.

Fresh analytic data and stored artifacts enter evolution through the same
configured-ID materialization call. The stored-artifact implementation owns
exact-cache selection, hierarchy interpolation, exterior completion, and state
reconstruction; callers must not implement separate "sequence load" and
"evolution load" remappers.

AMReX constructs its scratch hierarchy before a stored artifact is available
to state-based tagging. After the first artifact load, evolution therefore
regrids the target hierarchy and rematerializes every changed layout directly
from the artifact until a regrid is a no-op. Do not leave a newly created fine
level as prolonged coarse initial data when the portable source can evaluate
that level directly. This convergence is startup-only; runtime regrids retain
the ordinary FillPatch/prolongation finite-volume contract.

For a same-chart periodic source, each periodic target extent must contain an
integer number of saved periods. For a nonperiodic larger domain, physical
fluid data use the configured relativistic exterior and ADM data use the
physical exterior described above. This is an initialization remap, not a new
constraint solve: interpolated data can have larger pointwise constraints than
the saved grid. A continued rotating XCTS solve does solve the remapped target
again; a direct evolution start does not.

A finer target grid does not recreate information missing from the saved
solution. It can reduce evolution truncation error, but a genuinely
higher-resolution initial-data solution requires a higher-resolution solve.
The field identities and centerings in the physical core are declared once in
`schema/fields.toml`. Checkpoint/restart remains a separate time-evolution
contract with step, time, hierarchy, and run-control state. Both formats are
valid evolution inputs, but a checkpoint resumes its saved time and hierarchy,
whereas a physical-ID artifact initializes a new run at step zero through this
materialization interface.

Evolution, physical-ID remapping, and the shared CTT/CTTK/XCTS elliptic layer
use tensor identity from the field catalog. Scalar, polar-vector, axial-vector,
and symmetric-tensor identities are distinct; in particular a magnetic field
is axial, so its component normal to a Cartesian symmetry plane is even and
its tangential components are odd. Physical-ID metadata records saved
reflection faces. A target outside one of those faces is mirrored into the
saved sector, interpolated on its declared centering, transformed in physical
Cartesian components, then expressed in the target basis. Points beyond the
saved outer extent still use the normal physical exterior completion.

A reduced source dimension removes coordinate dependence, not physical tensor
components. Portable `Vector3` and `SymmetricTensor3` fields retain all three
orthonormal physical components declared in `schema/fields.toml`; an
axisymmetric `r theta` source can therefore store a nonzero azimuthal velocity
or shift even though `phi` is not an active mesh axis. The materializer owns
the basis transformation to the target chart. Do not truncate portable field
rank to `AMREX_SPACEDIM` or create a reduced-dimension duplicate field schema.

The Cartesian conformal-constraint owner follows the same rule inside the
elliptic solve. `cartesian_physical_axes` maps each active runtime direction to
canonical physical `x/y/z`; derivatives along omitted axes are exactly zero,
while the CTT/CTTK/XCTS vector and symmetric-tensor unknowns remain
three-dimensional physical fields. The physical owner remains the full `x y z`
stellar problem. Its catalog `solver_axis_layouts` only advertises ordered
one-, two-, and three-dimensional Cartesian numerical projections for
manufactured checks, invariant problems, and future explicitly validated
reduced owners. Do not instantiate separate dimension-specific physics or call
a planar projection a spherical star. The 3D cell-centered vector solve stays
on the direct three-component MLTensor path; only 2D packs active physical
components, and 1D uses the analytically reduced scalar coefficients.

Cartesian ID vector and scalar first derivatives share a centering-aware
boundary closure. At physical outer vertices, use the generated one-sided
stencil on solved values, not constant-extrapolated nodal ghosts. At genuine
reflection/periodic faces, and for cell-centered face-BC fields, use the
corresponding ghost data with centered differences. Keep this choice outside
the physics equations; it does not change evolution boundary handling. The
nonzero manufactured CTT/XCTS convergence gate includes boundary vertices and
all physical components (`validate_reduced_initial_data.py
--cartesian-matter-convergence-only`); the cheaper projection gate remains the
standard-suite slice.

The validated reduced physical-ID handoffs are spherical one-dimensional TOV
and spherical two-dimensional circular-RNS XCTS to three-dimensional Cartesian
RHD. Uniform and explicit-face radial artifacts use the same evaluator. The
coordinate-equivalence ladder compares (1) native reduced and full-storage
source fields, (2) both sources materialized onto the identical Cartesian
hierarchy, and (3) at least one complete identical evolution step. An
independent solver in another chart is a separate work-precision comparison:
its errors should decrease under refinement, but it is not expected to agree
to roundoff with a different discrete elliptic operator. An
exact-layout mapped RNS artifact can seed the next rotation target without
interpolation; the resulting model remains independently usable for Cartesian
materialization. Same-layout Cartesian magnetized artifacts retain their
canonical face fields. Cross-chart or changed-grid CT remains deliberately
rejected until face-centered divergence-preserving magnetic transfer has an
owner.

The spherical Schwarzschild-XCTS owner is an exterior horizon shell. Reduced
`r theta` and full-storage `r theta phi` shells must agree as coordinate
projections and may exercise the portable Cartesian evaluator, but this alone
does not define production full-grid evolution: the target still needs an
explicit constraint-solved black-hole interior fill or a genuine excision
boundary. Do not relabel a uniform-fluid materialization probe as a reusable
black-hole artifact or silently treat horizon clamping as that missing policy.

Object pose is a materialization choice, not an elliptic-solver coordinate
choice. The artifact persists stellar rotation axis and center separately from
black-hole spin/origin and disk axis. One proper rigid transform maps target
points back into the source object's Cartesian frame before chart
interpolation, then rotates catalog-declared polar/axial vectors and symmetric
tensors into the target frame. A nonidentity pose disables exact evolution
caches and uses the portable core. Relative axes of composed components must
remain compatible under that one transform; changing their relative tilt is a
new physical ID problem, not a materializer option.

Elliptic boundary types are component-specific: on a reflection plane the
normal component of a polar vector is odd while its tangential components are
even. Explicit reflection/type metadata is authoritative. The compatibility
path that infers legacy Robin data from a nonzero stored coefficient is used
only when no explicit reflection boundary is present.

For a centered nonrotating TOV model, independent x/y/z reflections permit a
true octant solve. A star rotating about z is not invariant under independent
x or y polar-vector reflections: its azimuthal velocity and shift change with
a coupled rotation, not ordinary plane parity. Nonzero rotating-star initial
data therefore reject x/y reflection boundaries and support the z-equatorial
reflection only. Surface rays must reconstruct the omitted reflected
direction before evaluating polar/equatorial geometry. Do not describe a
nonzero rotating Cartesian half-domain as an octant.
