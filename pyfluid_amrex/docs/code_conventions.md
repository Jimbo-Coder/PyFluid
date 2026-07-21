# PyFluid AMReX Code Conventions

These rules keep the production solver predictable for human and AI
maintainers. `AGENTS.md` owns workflow details; this file owns implementation
decisions.

## One Contract Per Operation

- Implement one production path for one numerical operation. Dimension,
  coordinate, AMR level, and subcycling differences are inputs to that path,
  not reasons to duplicate it.
- A selected method means that method. Do not silently replace reconstruction,
  interpolation, Riemann solving, timestepping, diffusion, or CT because a
  branch is inconvenient.
- Generalize an implementation to all intended inputs. If a combination is
  genuinely outside the design, reject it once at configuration validation and
  document the support boundary. A bug in an intended combination does not make
  that combination unsupported.

## Fail Honestly

- Do not copy neighboring values over NaNs, clamp an invalid result into a
  passing test, or add a fallback merely to keep evolution running.
- Every relativistic evolution runs the shared startup preflight after initial
  data, AMR synchronization, centering transfer, and boundary filling. The
  preflight reads conserved state without repair and verifies both
  `P2C(C2P(U))` and a second `C2P` idempotence pass on every active cell before
  iteration zero. Initial-data providers must not carry private substitutes.
- Development failures belong in `review_running_cycle.md` with the failing
  command and evidence. Fix the underlying implementation; do not add a
  temporary startup prohibition for a branch currently being repaired.
- Do not commit temporary guards, bypasses, aliases, or replacement algorithms
  whose only purpose is to quarantine an active bug. Keep the failure visible,
  reduce it to the smallest reproducer, and repair the shared operation.
- Physical floors and positivity methods are explicit numerical methods, not
  error concealment. They need diagnostics and validation.

## Ownership And Sources Of Truth

- Register each runtime option once. The input file selects supported values;
  runners do not mutate simulation parameters.
- `config.hpp` owns the canonical `RunConfig`, stable configuration value
  types, and declarations only. Supported-value catalogs, parsing,
  validation, and derived host policy live under `src/framework/`; do not
  inline them into the public contract or import pointwise physics headers
  merely to expose a configuration data type.
- Define each grid function once in data-only `schema/fields.toml`, consumed by
  the strict `codegen/pyfluid_codegen/field_catalog.py` loader/model. Evolution,
  checkpointing, output, diagnostics, plotting, and equation generation consume
  generated C++/JSON views. Do not add a handwritten component ID, centering
  table, output label, or spacetime field list.
- Keep critical identity/metadata catalogs declarative and versioned. Python
  owns strict loading, validation, tensor/component expansion, symbolic
  mathematics, and artifact emission rather than being the only declaration
  text. Any named field that is evolved, retained between operations,
  user-selectable, output-visible, or checkpoint eligible is a GF and must have
  a generated ID and descriptor before C++ can request it.
  Locally named RHS, RK-stage, elliptic-residual, and transfer work arrays are
  temporary storage rather than GFs and do not enter the catalog.
- Parse user field names through the generated catalog and pass generated field
  IDs/descriptors thereafter. Field allocation, lookup, checkpoint, output,
  and plotting must not accept a second raw-string registry. A handwritten fill
  implementation may switch exhaustively on the generated ID, but it must not
  define identity, components, labels, centering, or ownership again.
- Define runtime reductions and postprocessed diagnostics once in
  `codegen/pyfluid_codegen/diagnostic_catalog.py`. Runtime code owns discrete
  conservation reductions; postprocessing owns analytic comparisons and
  derived views of output GFs.
- Migrate declarative diagnostic identity/group metadata and supported-method
  metadata to separate TOML catalogs only through byte-equivalent staged
  conversions. Keep symbolic integrands, algorithms, coupled compatibility
  checks, and physical validation in Python/C++; do not encode them as opaque
  expression strings or invent a general schema language.
- Coordinate transforms, metric factors, and vector/tensor operations each have
  one shared implementation. Initial data, evolution, and diagnostics reuse it.
- Generated kernels come from one high-level equation source. Do not retain a
  handwritten production duplicate after equivalence is established.
- Header-only code is reserved for small value types and pointwise CPU/GPU
  kernels whose inlining is part of the numerical implementation. Put host
  orchestration, runtime policy selection, AMR launches, diagnostics, and
  other substantial implementation in independently compiled `.cpp` owners
  behind declaration-only public headers.
- Emit large generated systems by stable equation responsibility. Relativistic
  core conversion, flux/wave speed, sources, and C2P recovery are separate
  generated groups; Z4c geometry, constraints, gauge, and evolution are
  separate groups. Umbrella headers are aggregate checks only. Production
  modules include the exact generated groups they own.
- C++ mathematical constants live in `constants.hpp`; generated kernels and
  handwritten code consume those typed values. Python tooling uses `math` or
  NumPy constants. Numeric input files remain serialized numeric data.
- Review complex systems through their high-level symbolic source, generated
  diff, and validation corpus. Do not consume maintainer context by manually
  expanding generated equations to compare them component by component.
- Keep generated pointwise physics out of the template-heavy fluid driver.
  Compiled sector orchestration may select policies and move AMReX data, but it
  must not duplicate physical equations.
- Own evolution template instantiations by their compile-time signature, not by
  nominal physical-system names. The bundle key is component count, CT
  topology, and EOS dispatch policy; systems that share that key share the
  compiled bundle. Add a bundle only when a new signature is genuinely needed.
- Runtime method selection belongs in a declaration-only compiled factory.
  Build and validate one immutable operation plan on the host. Do not form the
  Cartesian template product of independent run parameters: adding one method
  should have approximately additive compile/code-size cost. Keep a choice
  compile-time only when storage/topology requires it or a focused benchmark
  establishes a material hot-path benefit. Uniform runtime choices belong at
  the coarsest practical launch, tile, or row boundary; do not add divergent or
  repeated policy discovery inside a cell loop.
- Keep fused operations fused at their numerical boundary. Reconstruction and
  the Riemann solve share reconstructed face states, so their production
  ownership is one flux-kernel registry rather than two materialized stages.
  Formula headers remain method-local so changing MP5, TENO-5, or WENO-Z does
  not invalidate unrelated methods.
- Public factory headers contain request types and declarations. Heavy kernel
  headers and template dispatch live in private implementation owners. Do not
  restore umbrella solver headers to broadly included class interfaces merely
  to make a transitive include available.

## Evolution Sectors And Grid Fields

- Fluid and spacetime are first-class evolution sectors on one AMR hierarchy.
  Fluid conserved state is cell-centered. Evolved Z4c state is vertex-centered.
  Neither sector is an auxiliary field of the other.
- An evolution sector owns its accepted, previous, next, scratch, RHS, and
  time-integrator stage storage. Only accepted state is physical/checkpoint
  data; RHS and stage arrays are solver work storage, not auxiliary GFs.
- Fixed spacetime is a read-only metric provider, not an evolved field set.
  Evolved spacetime is the complete mandatory Z4c field set. Constraints,
  cached derivatives, and optional output quantities remain derived scratch or
  diagnostic fields and must not be checkpointed as evolved state.
- Z4c stores full values, not Minkowski-subtracted perturbations. Derive
  conformal/physical geometry once through the generated pointwise evaluator.
  Evolution, initial-data checks, and postprocessing diagnostics call the same
  generated constraint function; they must not carry separate Hamiltonian,
  momentum, Z, Theta, or algebraic-constraint formulas.
- Grid-level Z4c consumers use the shared pointwise bridge for component
  loading and derivatives. Geometric derivatives are centered; shift
  advection uses the common shifted-upwind derivative. Component maps and
  stencil assembly must not be repeated in initial data, evolution, or output.
- Runtime orchestration selects generated lapse and shift policies outside the
  point loop. The complete generated Z4c RHS is one compile-time-policy entry
  point; do not reproduce its geometry/gauge assembly in the timestepper.
- Keep metric-derived Ricci used for constraints distinct from the Z4c
  evolution Ricci that contains the evolved contracted connection. Equality of
  those two is a validation identity on the Gamma-constraint surface, not a
  reason to alias their implementations.
- Matter enters Z4c through explicit Eulerian projections. Vacuum supplies a
  zero-valued projection object; do not generate or maintain a second vacuum
  RHS implementation.
- Dynamical matter coupling is one strongly coupled method-of-lines stage, not
  a fluid step followed by a spacetime step. At each RK stage, recover the
  cell-centered relativistic fluid state in the current metric, generate the
  stress-energy tensor and Eulerian matter projections, transfer those
  projections directly to vertices, and evaluate the existing generated Z4c
  RHS. Transfer the stage metric and required derivatives directly to cells
  once for the generated Valencia flux/source RHS.
- Metric views, stress-energy projections, and centering transfers are
  stage-local coupling scratch. They are neither evolved fields nor checkpoint
  data. Fluid conserved state is refluxed; vertex-centered spacetime state is
  restricted/synchronized through its nodal AMR contract and is not refluxed.
- Subcycling supplies both sectors at the requested stage time through the
  shared evolution-sector coarse-time interpolation. Regridding and
  checkpointing occur only at synchronized sector times.
- Every solver-created AMR level samples the same initial-data provider for all
  active sectors. Level-zero, initial fine-level, restart, and regrid code must
  not call a provider-free initializer when provider data are active.
- Initial data are sampled in physical coordinates. Analytic providers and
  external grid-data providers share this sampling boundary; provider-specific
  interpolation does not belong in the fluid or spacetime evolution kernels.
- Initial-data elliptic/nonlinear solvers own no evolution state. They produce
  analytic or sampled provider data on the canonical field schema; evolution
  must not depend on a particular TOV, XCTS, KEH, or external-data backend.
- Solve each coupled initial-data problem once on its natural canonical
  centering. The initial-data hierarchy adapter materializes the centering
  declared by each evolution GF through the one generated order-2/4/6/8
  topology transfer, then applies the AMR synchronization contract for that
  target topology. Do not solve independent cell-, vertex-, face-, or
  edge-centered copies of the same constraints.
- Relativistic coupled initial data use one staggered hierarchy matching the
  evolution schema: finite-volume fluid unknowns are cell-centered,
  Z4c/CTT/CTTK/XCTS spacetime unknowns are vertex-centered, and CT magnetic
  unknowns are face-centered. Inter-centering matter and metric views used
  inside the coupled equations are shared stage-local operators; there is no
  final initial-data-to-evolution remap of a solved field.
- Cell/vertex/face/edge conversions are source and destination masks supplied
  to one tensor-product operator, not separately maintained `c2v`, `v2c`,
  `c2f`, or `v2f` kernels. Analytic providers may bypass interpolation only by
  evaluating the same canonical field exactly at the requested target points.
- Elliptic physical boundaries use one coefficient-field contract,
  `A f + B d_d f = C`, where `d_d` is the increasing-coordinate derivative.
  Periodicity remains a geometry property. Dirichlet and Neumann data are
  coefficient choices, not separate caller interfaces.
- Centered asymptotic fields `f = f_inf + amplitude/r^p` use the standard
  amplitude-independent Robin condition `p(f-f_inf) + r d_r f = 0`. On a
  Cartesian face this is represented through the same coefficient contract;
  do not sample an exact finite-radius solution or add a formulation-specific
  boundary path.
- Value type, centering, symmetry, ghost width, checkpoint ownership, and
  output ownership come from the field schema. Do not maintain a second list of
  spacetime variables in evolution, checkpoint, or plotting code.

## AMR

- Level creation, level remake, FillPatch, subcycled FillPatch, and repair seeds
  use the same selected coarse-fine transfer contract.
- Vertex-centered spacetime state uses the project fourth-order nodal transfer
  for level creation, remake, and FillPatch. AMReX's bilinear nodal transfer is
  insufficient for the fourth-order Z4c spatial operator.
- Reflux, average-down, prolongation, and physical boundary filling are distinct
  operations and must not substitute for one another.
- Automatic AMR is successful when the physical metric stops refinement with
  headroom. Maximum level, minimum feature length, radial guards, and cell
  budgets are resource protection, not convergence criteria.
- Short smoke tests do not establish dynamic AMR correctness. Regridding
  changes require long moving-feature tests, conservation, causal quiescence,
  and hierarchy diagnostics.
- Regrid descendants only when their parent and child times are synchronized.
  A descendant-only regrid must preserve the active parent flux register and
  reset only rebuilt descendants; otherwise retain globally synchronized
  regridding.
- `amr.regrid_mode=synchronized` evaluates `amr.regrid_int` in accepted
  level-zero steps. `amr.regrid_mode=aligned` evaluates it in accepted steps
  on the current finest level, then rebuilds from the lowest level whose
  descendants are synchronized. These cadences have different units; compare
  them at similar measured regrid counts, not by assigning the same interval.
- AMReX level callbacks are the source of truth for whether a scheduled regrid
  changed the hierarchy. An unchanged attempt must return before rebuilding
  registers, resetting clocks or caches, averaging down, or refreshing state.
  Changed-hierarchy maintenance starts at the first modified interface and
  covers only the synchronization chain affected by that change.
- Aligned subcycling may rebuild synchronized descendants between coarse
  steps, but it must also consume the same finest-step cadence at level-zero
  synchronization points. Descendant attempts must never prevent periodic
  whole-hierarchy tagging from discovering newly separated coarse features.
- Keep the measured adaptive Lohner/Richardson policy in
  `adaptive_richardson.md`. It remains a semiproduction design until a generic
  reversible hierarchy advance exists; do not substitute an offline shadow-run
  harness for a runtime estimator.

## Conservation Accounting

- Evolution owns cumulative numerical boundary fluxes, explicit source
  injection, and mechanical wall work, using the same RK/subcycling weights as
  the state update. Do not reconstruct those discrete budgets in Python.
- Scalar diagnostics reduce and write the ledger with the conserved totals.
  Shared postprocessing applies policy, checks closure, and plots residuals.
- Mark a ledger unsupported when its conserved quantity has a different update
  contract, such as fluid-plus-potential self-gravity energy. Do not report a
  partial budget as a valid closure check.

## Changes

- Prefer deleting an obsolete path over preserving compatibility aliases.
- Keep patches narrow, then run the smallest relevant regression. After shared
  infrastructure changes, run the quick suite; after hot-path changes, also run
  the benchmark.
- Never weaken a test to accommodate a failure without a physical or numerical
  reason recorded in the test declaration.
