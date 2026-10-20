# Complete feature support with practical build cost

Status: Cycles461--463 established the plan; Cycle467 tested isolated wall-kernel
variants on Mac and CUDA. Cycle468 adopts the ordered wall call and runtime
source/CFL ownership described below. Cycle469 extends sharing to subcycling
and limiter orchestration. Compiler flags and supported methods remain unchanged.
See the measured decisions below. The user wants all supported features to remain available while
ordinary development builds approach30--60s. Also minimize a genuinely fresh
complete build, subject to numerical correctness and runtime performance;
cache reuse alone does not satisfy that second goal. Smaller development configurations
are allowed as a workflow option; they must not replace complete integration
coverage or conceal the cost of an expensive single kernel.

## What already exists

CMake selects separate backend/precision dependencies and1D/2D/3D executables.
It has compile switches for hydro/MHD/CT, reconstruction variants, Riemann
solvers, time integration, diffusion and subcycling. Reconstruction source
owners are separate, and the host flux factory selects an enabled method.
The internal flux dispatch shares runtime EOS identity and specializes the
necessary component layouts/directions/topologies. Do not introduce an EOS
variant for every model name: this would reverse existing sharing.

Current code conventions already require additive method ownership rather than
a Cartesian template product of independent runtime parameters. They also keep
reconstruction and the Riemann operation fused at the numerical boundary.
Libraries alone do not reduce template instantiations or device optimization;
a monolithic kernel renamed as a library is still an expensive kernel.

## Proposed workflow

1. Keep a full build with every maintained feature for baseline/integration
   validation. Preserve an existing complete executable while developing a
   separate focused configuration. No unrelated working build is invalidated.
2. Create a small maintained set of focused development presets from existing
   feature switches, selected for the task's method set, dimension, backend and
   precision. Example: one-dimensional FP32 CUDA RMHD with PWL/HLL/SSPRK3,
   alongside whatever control methods that investigation needs. Keep physical
   coefficients, CFL, resolutions, output, IC values and ordinary EOS parameters
   runtime. Compiling only one method is useful only when explicitly selected;
   changing a normal input should not unexpectedly launch a long compiler job.
3. Expose the exact compiled capabilities in a generated build manifest and
   startup report. The runner should identify a compatible existing executable
   or report the missing capability and the explicit build command. Reject
   unavailable input choices before evolution, never substitute a different
   method. Reuse the existing option declarations/build switches instead of
   creating a second physics registry. Record build identity alongside run.in.
4. Organize reusable compiled owners into static/object libraries where that
   removes repeated compilation or narrows dependencies: common host framework,
   state/topology bundles, fused flux-method owners, elliptic/initial-data and
   spacetime owners. Share each compatible object across drivers/tests. Objects
   tied to AMReX dimension, precision, GPU architecture or compile definitions
   cannot simply be shared across those configurations; cache them separately.
5. Dispatch coarse operations on the host through the existing factory/operation
   plan. Keep structural template choices where required for storage/topology
   or a measured hot-path benefit. Avoid multiplying each kernel by independent
   EOS, IC, output, timestep and feature policies. A runtime scalar value does
   not inherently create a code variant. A runtime switch referencing many
   templated versions does retain those instantiations; an explicitly restricted
   build can prune them. Fixing a build to one compile-time choice really does
   produce only that chosen implementation and can reduce compilation/code size.
   The explosion warning applies when a full build instantiates all combinations,
   not when a parameter is fixed to one value. Separate build identities and
   compatible module reuse should manage the resulting executable variants.

Choose static/object modules before proposing dynamically loaded GPU plugins.
Dynamic modules introduce device linkage, registration, ABI and deployment
requirements, particularly with the validated non-RDC CUDA contract. Host
polymorphism at an operation boundary can be acceptable; per-cell device virtual
calls are not the proposed solution. Preserve fused fluxes rather than staging
reconstructed states solely to simplify compilation.

## Measurements before implementation

- Use the retained NVCC owner traces, then inspect the transitive include graph,
  instantiated device functions and resource reports. Measure expensive owners
  under the existing300s hard deadline; no full CUDA rebuild as a first probe.
- Compare the smallest representative full/focused owner configurations before
  promising profile speedups. A single PWC owner was still roughly97--136s in
  previous probes; restricting the catalog alone may not achieve the edit goal.
- Measure clean compilation, one physics-owner edit, a common-header edit,
  driver/diagnostic edits, cache-hit rebuilding and no-op rebuilding separately.
  Record compiler/device optimization/assembly/link phases, binary size and
  startup. A cache hit does not establish acceptable changed-kernel latency.
- Verify numerically identical enabled operations between focused and full
  configurations, with existing analytic/reference/scalar/AMR/CT/restart gates.
  Preserve optimization, precise math and actual production numerical methods.
- Promote only changes that reduce the affected compile/rebuild work while
  preserving numerical acceptance and runtime performance. Document remaining
  expensive changes honestly. Run broad integration checks at checkpoints
  rather than rebuilding every configuration for each routine edit.

## GPU compilation cost: measured evidence and next diagnostic

GPU compilation has additional host/device passes, preprocessing and binary
embedding. It has no established universal minute-per-file penalty. Good
runtime performance and clear source organization do not ensure small compiler
work: templates, inlining and generated algebra can produce large device IR;
optimization, register allocation and scheduling then process that expanded code.

Retained Cycle452/454 exact production-command probes, not new measurements:

| Source owner/configuration | Wall time | Device compiler (cicc) | Device assembler (ptxas) |
| --- | ---: | ---: | ---: |
| Diagnostics owner | 13.682s | See retained trace | See retained trace |
| 1D PWC owner | 135.790s | 63.617s | 56.446s |
| Same PWC owner, two optimizer/assembler threads | 96.860s | 44.728s | 35.378s |

The PWC baseline spends about88% of wall time in those two device phases;
its107 emitted device functions are not107 distinct physics kernels. This is
expanded compiler work rather than evidence of a fixed per-file launch tax.
Parallel compilation changes resource records, so its runtime acceptance is
still pending. Diagnostics is also substantive code, not a measurement of the
minimum CUDA compilation overhead.

When resumed, compare a trivial kernel, a light owner with the actual dependency
headers, one selected real specialization, and that owner's complete permitted
set under identical architecture/precision/optimization flags. Record phases,
emitted functions and IR size. This separates setup/header cost, single-kernel
complexity and variant multiplication before deciding whether to prune versions,
narrow includes, alter inlining/ownership or change compiler parallelism. Each
probe retains the existing hard deadline. Cycle467 subsequently measured a
focused hydro/PWC owner and the complete PWC owner; the trivial/header-only
decomposition and other expensive owners remain unmeasured.

## Cycle467 measured decisions

Isolated production-flag, uncached full1D PWC replays against AMReX26.10:
baseline Mac11.454s/CUDA180.084s; sharing the optional-tracer wall kernels
8.878/137.572s; selecting wall left/right states before a single solver call
9.922/153.271s. The latter is a small shared-source factoring and the preferred
first integration candidate. These are owner measurements, not full-build
savings, and independent ablation savings are not additive.

The tested runtime Riemann switch saved only8.2% in the focused CUDA owner and
made cheap mixed-wall LLF/HLL cases roughly8times slower. Its common kernel
retains64registers/1320stack bytes, versus32--37registers/zero stack in separate
cheap kernels. A no-inline wall helper also regressed these cases severely.
Uniform runtime selection does not guarantee cheap resource allocation; retain
specialization across materially different heavy algorithms unless measured
otherwise. Smaller choices such as optional tracers can share more successfully,
but this probe showed about8% lower untraced HLLC wall throughput on Mac.

Tracer sharing and single-call factoring match all338140 checked flux values
bitwise on both machines. Both pass six paired existing Mac wall evolution
cases with identical non-timing scalar histories and the selected standard
no-wall performance gate. This is ideal-gas/PWC evidence, not acceptance of
all reconstructions, EOS choices, dimensions or CUDA evolution. Cycle468 adopts
the ordered call; tracer sharing and the separate expensive3D limiter remain
pending. Exact scopes,
single/repeated compile samples, runtime exclusions, resources and reproducible
artifacts: `runs/current/infrastructure_audit/cycle467/report.md`.

## Cycle468 first implementation pass

- Source terms: remove unused component-count parameters and give acceleration,
  heating and gravity application one compiled `src/numerics/sources.cpp` owner.
  The numerical equations still use the same density/momentum/energy fields in
  every layout. Editing this implementation reaches one source owner per
  dimension instead of the seven old consumers of its implementation header.
  The measured individual evolution bundle barely changes compile time; this
  is chiefly an edit-scope improvement, not a demonstrated cold-build saving.
- CFL: layout/tracer and diffusion selection are host runtime choices. Preserve
  separate pointwise hydro/MHD/relativistic kernels and the measured ideal-gas
  specialization. Uncached1D owner samples change Mac2.006 to1.644s and CUDA
  37.562 to34.202s; the equations and per-cell arithmetic are unchanged.
- Wall calls: adopt the measured ordered-state/single-call factoring. Keep the
  tracer-sharing and combined-Riemann experiments isolated; their runtime
  penalties still prevent blanket adoption.
- Other mechanisms: `EvolutionKernelEntry` and `EvolutionOperations` already
  provide host function-pointer dispatch. Virtual classes, `std::function` and
  variant visitors can organize host selection, but substituting them for a
  switch does not remove the kernel variants being selected. No new dispatch
  framework or per-cell indirect call is warranted by this pass.

All-dimensional Mac production build,11 selected numerical/architecture gates,
15 build-infrastructure checks and all4 performance gates pass. Seventeen
paired cases have identical physical scalar histories, including the unchanged
existing self-gravity AMR energy-drift failure. The new CUDA source-owner
boundary passes a fresh link/run and80 pointwise cases with860160 bitwise-equal
values. Full current CUDA application acceptance remains open. Exact commands,
data, fixture corrections and validation limits are under
`runs/current/infrastructure_audit/cycle468/`.

Fresh complete-build comparisons must retain the same promised feature set and
hardware targets. A faster focused build is a separate result, not evidence that
the full build got faster. Avoid repeated instantiation, reduce expanded work,
reuse compatible ownership within a clean build, and measure resource-limited
parallelism; do not obtain a timing win by weakening arithmetic or runtime work.

## Cycle469 second implementation pass

Subcycling now passes independently selected time-integration and diffusion
operations through one recursive AMR loop per structural bundle. The five by
three method product remains only in the small operation-plan construction;
it no longer instantiates fifteen recursive loops. The measured hydro object
contains one recursive loop, one level-step implementation and one timestep
helper instead of fifteen, fifteen and three respectively.

The RMHD limiter shares candidate assembly, recovery and mask closure across
Riemann methods. Its PWL/PWC choice selects face states before one expensive
Riemann call. The Mac 3D owner compiles in 39.015 s versus 46.755 s, with object
size 5,205,072 versus 6,099,072 bytes. The intermediate shared-orchestration-only
candidate takes 42.339 s. These are uncached owner measurements, not full-build
percentages. Both complete 3D CUDA probes, and the old complete 1D CUDA probe,
hit the 300 s deadline. The new complete 1D CUDA owner finishes in 231.263 s,
establishing a lower bound of 23% on its saving; the exact old time remains
unknown. Per-method controls and
resource/runtime evidence are recorded in Cycle469 rather than raising that
limit or weakening production compiler flags.

All-dimensional Mac integration and 35 paired scalar-history comparisons pass
apart from the identical pre-existing self-gravity AMR energy-budget failure.
The maintained analytic/reference gates and canonical performance checks pass.
All 32 scoped CUDA limiter cases match bitwise. HLL timing shows occasional
slow executions in both old/new binaries; that comparison is inconclusive.
All 187 hydro-bundle CUDA device resource records are unchanged. See
`runs/current/infrastructure_audit/cycle469/` and the continuity cycle for
raw targeted timings, CUDA evidence and remaining acceptance boundaries.

## Comparative source review

Cycle463 source review is recorded in `review_backlog.md`, with exact paths,
hashes and evidence under `runs/current/infrastructure_audit/cycle463/`.
Local AthenaK has broad hydro/MHD/radiation/NR scope, runtime reconstruction
inside fused team-scratch fluxes, runtime component counts and one executable
for dimensional layouts. Public main instead stages face states between kernels.
These give concrete alternatives to our retained numerical specialization
combinations; their compile/runtime tradeoffs remain unmeasured. Local
SpacetimeX Z4c fixes one derivative order and stages derivative assembly before
pointwise algebra. SpECTRE illustrates system-specific executables and compiled
library ownership, not proven cheap complete CUDA compilation.

Compatible implementation-object sharing already exists in
`cmake/PyFluidSharedObjects.cmake`. Extend measured reuse where missing rather
than creating another generic layer. First isolate actual expensive numerical
variants and dependency boundaries. Preserve fused production fluxes; a runtime
choice at a uniform coarse boundary is a candidate to benchmark. Fixed-choice
focused builds remain useful but must retain separate full-feature coverage.
No matched external compile timing has been collected; historical local
AthenaK build metadata is CPU-only and refers to obsolete source paths.

## Precision interaction

See [precision_roadmap.md](precision_roadmap.md). FP32/FP64 are existing build
policies; mixed precision should be an explicit measured operation policy, not
a new precision dimension instantiated across every method/layout/backend.
Lower-width formats require representation/capability work, not extra enum names.
Backend and precision variants keep distinct artifacts and numerical identities.
Build capabilities must not replace checkpoint semantic compatibility rules:
unchanged storage/layout semantics should not be rejected solely because a
build contains a different unrelated method set.

The target is one shared solver and complete maintained feature catalog,
with a convenient full executable and reusable focused configurations. One-time
complete compilation still has a cost; the goal is to avoid paying it repeatedly
and to reduce the measured cost of the affected source owners themselves.
