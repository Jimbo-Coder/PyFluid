# PyFluid AMReX Working Knowledge

This is the compact state snapshot for an agent entering the production AMReX
backend after a context reset. It records learned numerical and physics facts,
current evidence boundaries, and open limitations. It is not a design
specification, bibliography, or work log.

Read `../AGENTS.md`, this file, `references.md`,
`docs/infrastructure_conventions.md`, and the tail of
`review_running_cycle.md` before changing the backend. Current source and a
newer continuity cycle win if they disagree with this snapshot.

Last full reconciliation: 2026-08-27, branch `main`, through Cycle 344's consolidated
Cycle 322--343 integration tree, CPU tile-ownership hardening, and preserved
performance baseline. `review_running_cycle.md` retains the detailed sequence;
this file records only the resulting current facts and evidence boundaries.
Scoped dependency, build and validation-boundary refresh: 2026-09-01, Cycles 373--380.

## Scope And Evidence

- Cycles469 and474 share timestep/diffusion orchestration and Riemann-independent
  RMHD limiter work. Correction face launches now have independent method owners;
  preliminary PWL and selected-face work share the same method kernel. The
  32-case 3D CPU correction fixture is bitwise equal with matching active counters.
  Bounded CUDA owner compilation is established, but the new full device fixture
  and current complete CUDA application remain unaccepted. Earlier CUDA HLL
  timing variability was inconclusive. Consult the current cycle for exact
  source/timing boundaries; the existing self-gravity AMR energy failure is open.

- Cycle 414 connects the complete generated fCCZ4 equations, reference-frame
  adapters, diagnostics and stored-metric GRHD bridge on regular charts.
  Cartesian and spherical gauge propagation, bounded spherical symmetry
  projections, and a partial-hierarchy spherical mass-conservation control
  pass. Origin/pole regularity, curved RMHD/CT, mixed-chart exchange and
  long-duration stellar validation remain open. Unbuffered vacuum-interface
  AMR reflux can still produce inadmissible energy; see the active cycle.

- Cycle 411 isolates the short self-gravity AMR/subcycling energy failure:
  disabling regridding has no effect; lower CFL and a finer base grid reduce
  the error, while tightening Poisson tolerance does not. This supports a
  temporal/spatial coupling investigation, not a completed conservation fix.
  The deep restart discrepancy decreases below its unchanged 1e-12 field
  threshold with Poisson tolerance 1e-11, but both restart controls still fail
  physical energy/momentum budgets. No production tolerance or gate was changed.

- Cycle 410 verifies the existing NS+disk construction/handoff gate and exposes
  an evolution setup limitation: a 64^3 evolved-spacetime run with frozen gauge,
  no KO dissipation and extrapolated outer boundaries develops nonfinite corner
  states at t=3.71. Changing only the outer condition to Sommerfeld reaches t=5
  (49 steps at CFL 0.2) with zero recovery resets and baryon drift <8e-10. The
  stellar disk evolution example now selects Sommerfeld explicitly. This is
  short-run evidence, not long-term equilibrium or all-configuration support;
  density changes ~1.2% and constraint norms still need convergence assessment.

- Cycle 409 adds exact finite-amplitude relativistic CP Alfven convergence:
  1D and axis-aligned 2D CT, PWL/HLL/SSPRK3, CFL 0.9, one period. The initializer
  uses relativistic inertia for RMHD; Newtonian arithmetic remains unchanged.
  All six cases pass, with maximum-error orders 2.21/2.20 and finest normalized
  L1 below 8e-4. This does not cover oblique propagation, curved GRMHD or stars.
  The fresh 28-row RMHD standard gate also passes, including exact Balsara 1--5.

- Exact finite-pressure SRHD shock heating uses conserved-profile convergence
  and independent analytic inflow budgets. Cycle 406 fixed SSPRK3 projection
  accounting; Cycle 408 fixes the RHD limiter's missing shared-face mask closure.
  W=10 now has zero projection-added energy and analytic energy error <=1.4e-14
  through 2048 cells in these probes, with matching one/three-rank profiles.
  W=100 retains its passing sequence. The original W=10 128--512 profile-order
  gate still fails (density/energy orders 0.576/0.288); the finer 512--2048 probe
  gives orders 0.660/0.688. Neither thresholds nor atmosphere floors were changed.
  Low-order terminal candidates can still require projection in other problems;
  zero recovery/reset counters alone are not conservation evidence.

- Cycle 404's CPU gravity-owner extraction preserves the checked scalar
  histories, but self-gravity AMR/subcycling has pre-existing conservation-gate
  failures. The deep self-gravity restart control also fails its checkpoint
  tolerance and scalar budgets on both pre- and post-extraction binaries.
  Restart equivalence alone is not physical validation. Exact evidence and
  current performance-validation limits are recorded in Cycle 404.

- `docs/infrastructure_conventions.md` owns durable source, catalog, AMR, and
  artifact contracts. `references.md` routes to implementation and external
  literature. `review_running_cycle.md` owns dated commands, measurements,
  failures, and decisions.
- The production implementation is C++/AMReX. Python research tools are
  baselines, offline operators, or postprocessing unless a task explicitly
  promotes and validates a production implementation.
- The native framework pin is AMReX 26.10 commit
  `015c14b65715b277fbc54706ebe7455b196d2254` (Cycle 456). The tracked manifest
  owns this identity; the build consumes a stable link to a versioned install. The
  check-by-default dependency helper updates only on explicit request, builds
  AMReX independently of PyFluid, and retires obsolete managed releases after
  successful activation. Still-active companions are not obsolete.
- The 26.10 consumer review finds relevant FillPatch/component-offset,
  periodic reflux, GPU reduction/launch, HDF5 conversion and finite-aware
  MLMG fixes without a required solver-policy migration. Cycle456's Mac CPU
  production, exact/reference AMR, implicit diffusion, elliptic and exact
  restart controls pass. New sparse AMG, gridding controls and asynchronous
  MLMG synchronization remain opt-in candidates, not adopted features.
- Cycle458's same-source 26.09/26.10 CPU comparison preserves all 265
  selected Sod, Brio-Wu AMR+CT and diffusion accuracy metrics. The refreshed
  Mac M4 CPU reference is `benchmarks/baseline_2026_10_01.md`; throughput
  changes range from -2.39% (RMHD) to +3.55% (WENO-Z). Thirteen selected
  standard gates pass, but the full standard suite still fails the known
  self-gravity subcycling energy check on both releases. A small Metal
  cell-centered MLMG probe measures 7.7--9.3x faster solves with no-sync and
  identical errors/residuals; actual production operators are not validated
  for that policy, so it remains disabled.
- The experimental Metal 26.10 patch retains upstream RNG/overflow/C++20
  fixes and uses a direct subgroup reduction for ordinary SYCL block sizes.
  The upstream multi-round fold returned zero in native and compensated
  Metal controls; the corrected library reproduces both prior values and
  passes the maintained state/reduction/launch check. Larger-count folding
  remains upstream, and the exact compiler-lowering cause is unproven.
  The refreshed 1D Metal application passes all 18 maintained CPU64/Metal32
  precision comparisons; 2D/3D Metal applications were not rebuilt this cycle.
  WSL dependencies are updated, but WSL application rebuilds/tests are
  deferred; prior executables are not evidence for the new pin.
- A stable dependency symlink does not invalidate Make's timestamp checks
  when it is redirected to an older install. CMake must resolve imported
  include/library paths to the actual install, and run banners must agree with
  the selected pin;
  matching manifests, links, and CMake configuration alone do not prove
  executable provenance. `CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS` handles this
  without clean-first or deleting generated kernels. A tiny compiled fixture
  verifies automatic upgrade/downgrade rebuilding and unchanged generated
  artifacts. macOS Make 3.81 compares whole-second timestamps; rapid synthetic
  rebuild tests must separate writes across that granularity.
- The build-ownership cleanup is verified for all three CPU MPI executables,
  focused diagnostic/restart checks and the four performance controls. The
  MPI+OpenMP graph configures but was not rebuilt in that pass. Fast individual
  recompile/no-op measurements do not establish a universal cold-build bound;
  ccache forced recaches are distinct from both cache misses and cache hits.
- Compile cost is not synonymous with template parsing. Current Clang profiles
  put almost all of the expensive generic MP5 owner's time in optimization and
  machine-code generation, while orchestration/diagnostic owners are much
  cheaper. Header isolation protects incremental rebuild reach but cannot by
  itself fix repeated optimization of large instantiated flux bodies. Allowing
  the compiler to share complete Newtonian face solvers reduces that repeated
  work, but doing so indiscriminately slows cheap PWC hydro. Preserve its
  measured direct-inline path using the same reconstruction/positivity and
  Riemann owners. The heavy MP5 local-edit target remains unmet; outlining
  needs matched runtime controls, including walls and mapped patches, not just
  smaller objects. These measurements are CPU MPI, not GPU evidence.
- Cycles467--468 distinguish useful sharing from merging heavy algorithms.
  Ordering wall states before one solver call reduces repeated optimization
  while retaining specialized Riemann kernels. A tested uniform runtime wall
  solver switch retained the expensive register/stack footprint and strongly
  regressed cheap CUDA solvers. The original uniform-tracer experiment is
  superseded by Cycle478's separate scalar-only launches. Production source terms now share one
  compiled owner; CFL layout/diffusion selection is runtime outside the
  unchanged pointwise kernels. Mac all-dimensional build, scoped numerical
  and four performance gates pass; CUDA source-owner link/pointwise equality
  passes, but the current full WSL application refresh remains deferred.
- Optional Newtonian tracers reuse the base five/eight-component physics kernel
  and reconstruct scalar flux independently from its stored mass flux. Component
  views remove the tracer-layout template axis. Reuse wall tile classification,
  skip dye reads on zero-area/solid faces, and retain per-overlap mapped upwinding;
  net seam mass flux cannot recover tracer counterflow. Keeping scalar host
  traversal out of the fluid launcher avoids a measured CPU optimizer regression.
  Some traced wall cases remain slower; scoped CPU equality is not GPU acceptance.
- Flux owners consume resolved value parameters rather than `RunConfig` or field
  descriptor metadata. This limits invalidation without reducing most optimizer
  work by itself. Shared code generation checks content plus the actual symbolic
  runtime environment, independently of build profile. Warm reuse is cheap;
  genuine diagnostic generation and complete feature builds remain expensive.
- A generated header's size is not its rebuild cost: one broadly imported
  helper can connect unrelated formula edits to expensive flux instantiations.
  The old vector umbrella reached reconstruction via magnetic energy in EoS;
  vector leaves and an independent generation stamp now isolate that path.
  Measure a forced-cold leaf dependency rebuild separately from no-op/cache-hit
  timing and the one-time cost of reorganizing the includes.
- AMReX 26.09 requires CMake >=3.25. Its tensor-operator update-flag and bulk
  viscosity fixes do not establish an XCTS speedup: the current cell-vector
  wrapper constructs/solves once and Cartesian XCTS uses a separate nodal
  operator. The new `compensatedAdd` is a caller-owned scalar Kahan helper,
  not a replacement for MultiFab/MPI reductions. FFT, EB, particle and GPU
  additions remain inactive in the maintained CPU builds; no new solver or
  template axis is needed for this release.
- CPU FP64 remains the default; CPU FP32 now has an independent AMReX
  installation, all-dimensional production build, preset and machine profile.
  Precision selects `amrex::Real` without another physics implementation.
  Native/HDF5 output preserves its stored precision, and runtime metadata
  records 24/53 mantissa bits. See `docs/precision_and_metal.md` for commands.
- Short CPU FP32/FP64 comparisons pass 18 RHD/RMHD entropy and gauge-wave
  controls with explicit scalar and physical accuracy budgets. FP32 HDF5
  restarts reproduce fluid and spacetime fields exactly in the bounded
  entropy/gauge controls. This does not certify long stellar evolution,
  thermal recovery, elliptic solves, all AMR combinations or every legacy
  compiled test in FP32. Gauge lapse meets the 1e-6 absolute target, but
  short-run constraint maxima reach roughly 6e-4--3e-3 at the tested FP32
  resolutions: refinement can amplify storage roundoff.
- Final Cycle432 CPU benchmarks find FP32/FP64 speed ratios of 1.08/1.12/1.04
  for the canonical 2D PWC, 3D PWC and 3D WENO hydro cases, and 1.92 for uniform
  RMHD. Real arrays halve storage, but runtime benefit is workload-dependent.
  The Cycle426 fine-grid FP32 error growth was largely RK background-rounding
  bias. Shared affine stage updates preserve zero-RHS constant states exactly.
  Cycle428 N1024 RHD/RMHD errors fall from 4.4--4.5e-5 to 3.46e-6/4.56e-6;
  existing accuracy and conservation budgets pass. FP64 retains 1.76e-6/1.77e-6.
  FP32 still approaches a roundoff plateau; refinement does not guarantee
  indefinitely better accuracy. Limiter candidates and CT use the same blend.
- FP32 arithmetic is checked beyond array storage: Clang rejects implicit
  floating widening/narrowing in production owners, and captured device IR
  has no FP64/wider arithmetic. This IR evidence covers exercised kernels;
  host clocks/formatting and Python reference calculations remain wider.
  Cycle451 extends the audit to GCC/NVCC: the symbolic C++ printer selects
  typed math overloads, correcting real GCC widening from unqualified C99
  calls. CUDA vendor float trigonometric routines contain internal FP64
  argument-reduction instructions; typed inputs/outputs do not mean every
  instruction in a vendor implementation is FP32. Static counts are not costs.
  Generated RK weights and AMR thresholds use Real. FP32 recovery's root
  bracket is eight epsilons; 512 epsilons was too loose for the unchanged
  conserved round-trip acceptance bound on smooth RMHD states.
- Generated derivative sums evaluate antisymmetric sample differences and
  zero-moment rebased sums before multiplying by rounded coefficients. Mixed
  derivatives use cross differences. Constant fields then differentiate to
  exactly zero in both precisions, including one-sided derivatives, without
  changing the catalogued coefficients/orders. Boundary test roundoff must
  account for cancellation of donor values, not only the derivative size.
- Metal uses an isolated experimental AMReX SYCL -> AdaptiveCpp SSCP route,
  with exact compiler/source pins and reviewed patches. The M4 device probe
  and complete 1D/2D/3D production builds pass. Short RHD/RMHD entropy and gauge
  lapse comparisons pass against fresh FP64 runs after correcting compiler
  control-flow reuse and FP32 recovery. The final Cycle432 matrix passes
  18/18, and the 2D/3D Alfvén accuracy/convergence controls pass6/6. Gauge constraints remain a numerical
  limitation: the resolved N64 Metal Hamiltonian maximum is about 0.0125,
  versus 0.0031 on CPU FP32 and 1.1e-7 on FP64. Evaluating identical saved Metal
  fields on CPU reproduces that norm in FP32 and FP64; widening the diagnostic
  cannot restore the evolved state. Full FP32 NR accuracy is not certified.
  Metal entropy/gauge HDF5 restarts reproduce final fields bitwise, and the
  poisoned-state debug controls pass. Validation uses one GPU rank. Final moving Newtonian CT refinement and
  the existing 3D ratio-four CT/reflux topology control pass after the
  target-selection and temporary-lifetime corrections; these are scoped
  controls, not certification of all AMR/relativistic configurations.
  The compatible local toolchain uses LLVM20 and SDK26.5; SDK27 fails LLVM20
  math headers. One GPU rank is selected until
  device identity/sharing is audited. HIP remains unvalidated; the scoped
  native CUDA evidence is recorded below.
- Device portability includes host/device ownership, not just annotations:
  checkpoint unpacking needs an explicit host-to-device copy and temporary
  lifetime; conservation ledgers and active-state debug checks use AMReX
  reductions over device arrays. Detailed failure formatting copies data to
  the host instead of dereferencing GPU storage or using device varargs.
  Keep dependency/compiler fixes separate from pointwise physics. FP32
  storage cannot retain perturbations below its local relative resolution;
  wider arithmetic cannot recover discarded data, and this Metal route has
  no native device FP64. Cycle428 traced the larger-grid failure to missing
  Metal declarations for indirectly accessed buffers. The runtime now prepares
  residency/hazard tracking from the same live-allocation inventory as pointer
  translation. A separated-allocation regression fails even at 256 cells before
  the patch and passes through 32768 afterward: this was not a RAM-capacity
  limit. Corrected full waves pass through 262144 cells. At that size the GPU's
  later evolution reaches 1.314 Mcells/s versus 0.550 CPU FP32 and 0.285 FP64
  on the same 16 GiB M4 (one GPU rank versus three CPU ranks). Startup still
  dominated the original short 64-step process: 169 seconds GPU versus 31 CPU
  FP32. Cycle429 isolates inherited one-second compilation postcheck sleeps:
  the unchanged executable with `ACPP_METAL_JIT_POSTCHECK_MS=0` passes in
  16.6--17.0 seconds with a populated source cache, or 43 seconds with a fresh
  application cache. This establishes a bounded end-to-end GPU gain on reuse;
  Cycle432 adopts this setting after four fresh/reused-cache debug-enabled
  hydro/gauge restart pairs preserve final fluid/spacetime fields bitwise. The older
  256-cell result reflects
  severe small-workload overhead; it is not a capacity limit. Full measurements
  and remaining profiling/coverage boundaries are in the precision document.
- Selected multidimensional GPU controls now pass, but throughput benefits
  depend on layout and operator cost. The accepted 64³ CT comparison is
  effectively equal end-to-end on reused-cache Metal and three-rank CPU FP32
  (about 48.5 s each); fresh-cache Metal costs about 251 s. Later Metal
  throughput is 19% higher. GPU solver peak RSS is much larger than its
  260 MiB main-arena use; the tested grids fit 16 GiB without establishing
  a capacity ceiling. Cycle433 supersedes the 512² FP32 absolute-divB
  rejections with the shared dimensionally scaled arithmetic policy:
  `divb_l1*h/(V*B_ref) <= 64*epsilon`. Runtime divB L1 is a volume integral.
  All 12 saved 2D/3D repeats pass; normalized errors are 0.30–0.37 epsilons
  across FP64, CPU FP32 and Metal. The original records remain unchanged.
  This periodic Cartesian scale uses the smallest reported active width;
  on AMR it is a conservative global bound, not a local constraint norm.
  Physical accuracy, convergence and stability remain separate requirements.
  See the precision document for
  matched inputs, memory definitions and the full measurement boundary.
- Cycle437 separates RMHD candidate recovery from face reconstruction/Riemann
  work and reuses tile-local face fluxes. Metal 2D MP5/HLL CT-AMR and shock
  controls with both PWC and PWL/PWC guards now pass; matched CPU64/CPU32 shock
  fields remain bitwise unchanged. CPU 3D CT-AMR passes in both precisions.
  Cycles438--439 close the reproduced Metal 3D CT-AMR compiler failures:
  connection initialization uses shared metric first derivatives with host
  order dispatch, and optional geometric tagging launches separately from the
  fluid indicator. Constraints use the shared derivative subset and select
  their formulation on the host. The final Metal 3D analytic CT-AMR gate
  passes; matched CPU/Metal fields and constraint histories are unchanged
  from Cycle438. CPU Cartesian order-family and regular spherical evolution
  convergence pass; Metal spherical evidence covers startup only. The retained
  Cycle439 pipeline failure is from an intermediate shader. Cycle445 reproduces
  it from retained MSL alone: library creation passes, then Apple's pipeline
  compiler fails. Holding the entry-point name fixed and changing only its
  dynamic formulation flag to either constant makes compilation pass; restoring
  the dynamic flag restores failure. This supports the existing host formulation
  selection without identifying the faulty compiler pass. Two independent fresh
  application-cache 3D CT-AMR repeats pass with exact prior fields and constraint
  histories; OS-managed Metal caches remain intact. A fresh 3D MP5 Metal restart
  remains outside that evidence. The optional backend still follows the pinned
  third-party WarpX reference route with local adaptations, documented in
  `docs/precision_and_metal.md`. These controls do not certify all GR/backend
  combinations or establish a steady-state speedup.
- Native AMReX CUDA on WSL uses the same physics sources, generated equations
  and precision selection. Cycle451 completes all-dimensional FP32 production
  builds and the maintained precision, 3D MP5 CT-AMR and exact restart controls
  on one RTX 3070 Ti. GCC CPU32 also passes the precision gate. Two MPI ranks
  sharing the RTX exercise host-staged transfers; independent devices,
  GPU-aware MPI, inter-node communication and scaling remain unvalidated.
- CUDA portability requires valid launch ownership and captures throughout
  the call chain, named factory types, and device-readable component tables.
  These are shared infrastructure changes. Final Mac CPU32/Metal precision
  controls pass; final Metal 3D CT-AMR fields and constraint histories exactly
  preserve Cycle439. Retaining ideal/runtime EOS host selection through RMHD
  signal frequency reduces its ideal shader from 1.49 MB to 279 KB and resolves
  the reproduced intermediate pipeline failure without a Metal equation fork.
- Matched FP32 RMHD-wave timing at 262144 cells and 64 steps gives Metal 2.42x
  solver throughput and 2.00x end-to-end speed over three M4 CPU ranks with
  reused caches. CUDA gives 6.89x throughput and 2.47x end-to-end speed over
  eight Ryzen ranks, and 3.00x end-to-end over Metal. Cheap short hydro remains
  startup-dominated on GPUs. These are workload-specific measurements, not
  multi-device scaling; WSL CPU code is initially portable rather than native
  tuned. Compile/startup cost and benchmark inputs are recorded in Cycle451.
- The 3D six-step PWL restart control reaches its one-million-cell AMR budget
  on CPU64, CPU32 and CUDA32 before completion. This remains a failed control
  needing an error/resource study; budget exhaustion alone does not prove an
  instability or tagging bug. The passing MP5 restart does not certify PWL.
  The checkpoint comparison utility must initialize AMReX without parsing its
  positional directory arguments as an input file; Linux exposed that defect.
- Cycle452 resolves duplicate CUDA kernel-entry registrations by selecting
  AMReX non-RDC compilation with local launch ownership; the failure and fix
  reproduce in a two-file CUDA-only control. Default memcheck/initcheck/
  synccheck and racecheck pass actual 1D RMHD evolution without suppression;
  an invalid-write control still fails. All-dimensional builds, the maintained
  precision gate and fresh 3D CT-AMR pass. This candidate's exact restart and
  3D sanitizer follow-up remain incomplete. Cold CUDA compilation is still
  impractically expensive; the retained single-owner trace is dominated by
  device optimization and assembly. Do not equate a no-op rebuild with a
  practical cold/edit workflow or claim broad backend acceptance from these
  bounded controls.
- EOS CGS conversion must preserve intermediate representability as well as
  final values. Staged pressure conversion avoids FP32 c^4 overflow; Read
  piecewise polytropes are built in code units to avoid overflowing CGS powers.
  All four presets and 28 segments pass continuity and physical pressure
  anchors in both precisions. Maintained CompOSE table/recovery checks pass
  in FP64; this does not certify FP32 cold-surface thermal accuracy.
- CFL-location diagnostics use masked AMReX reductions with deterministic
  exact-maximum selection. The Metal checker and CPU64/CPU32/Metal hierarchy
  on/off controls pass without changing physical scalar histories. Solid-mask
  host Array4 work remains explicitly deferred.
- FP32 multidimensional integral diagnostics need compensated summation:
  `ReductionSum` retains a two-Real residual pair through AMReX reductions;
  FP64 remains ordinary Real. This corrects apparent 3D Alfvén conservation
  drift without widening the device arithmetic or replacing runtime totals.
  Cold Valencia energy and general-EOS recovery use shared rationalized
  expressions to avoid rest-mass cancellation. A representable TOV center
  offset plus explicitly selected FP32 ID tolerances make the stellar startup
  round trip pass at about 1.7e-7.
- FP32 evolved TOV and filled-hole controls still reject invalid outer-corner
  ghost metrics. The tested tensor extrapolation plus centering stencil has
  an absolute weight sum~2.1e8; promoting the stored FP32 donors does not fix
  it. This is a boundary-conditioning limit requiring a properly validated
  closure, separate from the existing periodic gauge constraint limitation.
  Matched FP64 TOV t=10 and filled-hole t=2.4M controls complete, but coarse
  TOV surface thermal-pressure excess and growing exterior BH constraints
  preclude a general long-time/converged-accuracy claim. Cycle432 retains the
  thermal-proxy shell table; it is not a Kelvin-temperature diagnostic.
- AMReX's vector `FillBoundary` and `FillBoundaryAndSync` helpers overlap
  independent same-phase exchanges without changing field centering or
  ownership. Batch fields only when they share periodicity and no field depends
  on the completed ghosts of another; keep numerically dependent phases
  separate. The periodic CT face path needs one such exchange, not two.
- `amrex::Reducer` can fuse heterogeneous sum/min/max operations into one local
  traversal, but it is not an automatic CPU optimization and does not replace
  the hierarchy owner's MPI reduction. A 17-value scalar-diagnostic reducer
  regressed the clean three-rank CPU path; direct specialized `MultiFab`
  reductions plus removal of unused integrand components were faster. Use a
  reducer for a measured local/GPU benefit, and pack same-operator MPI values
  when that preserves semantics and reduces collectives. Generated fixed
  integer powers use `amrex::Math::powi<N>` once the exponent is outside the
  printer's small explicit-product cases.
- Deterministic MLMG, variable-coefficient `MLCurlCurl`, FFT/OpenBC solvers,
  particle/EB additions, and device-vector reductions are not blanket upgrades.
  Enable them only for their actual contract: reproducible GPU multigrid,
  spatially varying resistivity, a compatible linear Cartesian operator,
  particle/embedded-boundary physics, or a measured device-vector hotspot.
- A compile, focused executable, smoke run, standard suite, full suite,
  performance benchmark, and MPI-rank comparison are different evidence.
  Report them separately and attach claims to the exact revision or dirty tree.
- Cycle 373's AMReX 26.09 update covers 52 standard gates across scoped reruns:
  51 pass; the short static-trumpet horizon-area accuracy gate remains failed.
  Its entire horizon history is byte-identical under verified 26.08 and 26.09
  builds of the same PyFluid source. This is a pre-existing numerical limit,
  not a dependency regression, and not a passing complete standard suite.
  Three stale test fixtures were corrected without changing evolution or
  loosening tolerances; commands and evidence are in the continuity cycle.
- The current scalar-policy input reader retains surrounding string quotes
  from canonical `run.in`. This can misclassify quoted `"none"` as diffusion;
  use an explicit physically appropriate policy when validating canonical
  inputs until semantic decoding is fixed. Closed and diffusive policies
  currently have identical checks, but other quote-sensitive branches do not.
- Cycle 284's final cache-hot quick suite passed 3/3 in 1.776 s. Its preserved
  standard baseline passed 42/42 gates with 135.305 s summed gate time, and all
  four runtime performance controls passed their recorded warn thresholds. The
  archive is `runs/baselines/20260812_212858_cycle284_final`. This is broad
  standard-suite CPU evidence, not a full-suite or GPU validation result.
- Cycle 296's equilibrium-torus implementation tree passes the expanded 44/44
  standard suite with 178.07 s summed gate time. Its focused full matrix is
  separate evidence under `runs/current/equilibrium_torus_validation/`; this
  rolling result is not a preserved performance baseline.
- Cycle 299's Kerr-foundation tree passes the expanded 46/46 standard suite
  with 191.96 s summed gate time and the four production throughput controls.
  Its 12-case black-hole result is rolling evidence under
  `runs/current/black_hole_validation/`, not a preserved baseline or a true-
  excision validation.
- Cycle 300's focused black-hole gate contains 13 accepted/negative cases,
  including exterior-only constraint convergence and a 16-step CFL-0.9
  moving-puncture-gauge stress. The complete tree passes the 46/46 standard
  suite in 207.68 s summed gate time. Its rolling artifacts are not a long-
  timescale equilibrium result or an apparent-horizon-finder validation.
- Cycle 301's dirty-tree focused black-hole gate contains 14 reported cases.
  Its new spinning-Kerr case covers the independent finder, dynamic horizon-
  band AMR, two nested output masks on both levels, persisted shape history,
  and restart restoration. This is focused CPU/MPI evidence, not yet a new
  archived standard or performance baseline.
- Preserved baselines under `runs/baselines/` and rolling TSV histories are
  routing evidence, not proof for later source. Re-run the smallest relevant
  gate before relying on an old result.

## Production Numerical Snapshot

- `spacetime.derivative_order` selects 2/4/6/8/10 (default 4) for
  Cartesian Z4c and regular-chart fCCZ4 bulk derivatives, connection completion,
  constraints, KO and halo reach. Both formulations pass the three-resolution
  Cartesian gauge control at every order; the full 22-field spherical gauge
  comparison passes at every order on a regular shell. High spatial order does
  not set full evolution order: RK4, fluid reconstruction, matter transfer and
  boundary algorithms can limit it. Approximate Sommerfeld gradients remain
  second order. Direct TOV uses adaptive Dormand--Prince 5(4) in log enthalpy;
  the general Cartesian and reduced-spherical elliptic interiors remain second
  order. Origin/pole evolution and Cartesian-only AH tracking are still open.
- `schema/numerics.toml` owns supported spatial orders. Exact SymPy moments
  generate derivative/KO/centering coefficients and smooth uniform-Cartesian
  point/cell-average operators. Generated leaf assertions retain their own
  order snapshot so a stale kernel cannot silently admit a new catalog order.
  Nodal prolongation independently selects a catalogued order using
  `spacetime.prolongation_order` (default 4). Tenth-order derivatives together
  with tenth-order prolongation pass the existing Cartesian four-level
  subcycling/restart control in both formulations; this does not establish
  tenth-order AMR convergence or adaptive spherical readiness.
- `PhysicalSystem` explicitly distinguishes HD, MHD, RHD, and RMHD. SRHD,
  SRMHD, fixed curved GRHD/GRMHD, and evolved-Z4c matter use the shared
  relativistic system rather than separate flat-space physics implementations.
  Analytic Minkowski is a host-selected metric specialization.
- `EOSParams.method` is the sole production EoS identity. HD/MHD/RHD/RMHD
  instantiate one `EOSRuntime` evolution owner; flux, signal speed, primitive
  cache, atmosphere, sources, boundaries, and RMHD C2P consume the same compact
  closure view. Current support is ideal gas, cold polytrope, hybrid
  polytrope, and cold CompOSE plus gamma-law thermal pressure. The latter is
  not a finite-temperature or composition-dependent CompOSE implementation.
- Two measured ideal-gas exceptions avoid repeated runtime closure dispatch
  without adding an EoS evolution/reconstruction product. A host check selects
  ideal primitive-cache and CFL kernels; the canonical RMHD C2P entry selects
  the already-maintained ideal RePrimAnd/Palenzuela algebra once per recovery.
  Every nonideal method continues through `EOSRuntime`. Fully generic per-cell
  dispatch regressed the simple 2D hydro controls to `0.73--0.81` of baseline.
  In the RMHD recovery corpus, throughput is `1.342` million recoveries/s with
  repeated runtime closure calls versus `1.725` after selection. Boundary
  selection restores 7.3% end-to-end throughput, from `1.892` to `2.030`
  Mcells/s.
- Production reconstruction acts on primitive variables. For relativistic
  velocity it reconstructs `z^i = W v^i`, with metric-consistent boundary
  handling. The validated production configuration for discontinuous RMHD
  remains component MP5, SSPRK3, the candidate correction, and CFL 0.9; the
  low-order registered executable defaults are startup-safe, not this accuracy
  recommendation.
- The RMHD candidate guard predicts each concrete RK candidate, replaces
  implicated shared faces with the configured terminal low-order flux, and
  records correction counts. Direct MP5 to PWC is the selected general policy.
  PWL-MC before PWC remains an optional comparison, not the default.
- A post-update recovery fallback and an atmosphere reset are not the same as
  a conservative face correction. Both are counted across ranks and warned
  once per affected step. Ordinary prescribed atmosphere cells are not reset
  events. An evolution-time reset is a serious last resort, not a routine
  reconstruction branch or a reason to add an atmosphere mass ledger. Current
  run-total counters restart from zero after checkpoint restart, so their
  cumulative labels mean since process restart rather than whole logical run.
- MP5/PWC has the best general error/robustness balance in the current exact
  SRMHD shock matrix. PPM4/PWC is useful when lower shock ringing matters more
  than minimum L1 error. Unlimited parabolic reconstruction is a smooth-data
  reference and is not a discontinuous-flow production choice.
- `pwl` means monotonized-central limited PWL, not raw two-argument MINMOD.
  MC is less dissipative than strict MINMOD and reduces to a constant state
  when its bounded slope vanishes.
- SSPRK3 is the discontinuous-RMHD default because the stage-candidate guard is
  materially more robust with it than with non-SSP RK4 on extreme tubes. RK4
  remains appropriate for smooth problems.
- HLLD is the production RMHD Riemann solver. HLLC is a contact-restoring
  algebra/control path with known multidimensional strong-shock sensitivity.
  Multidimensional HLLC now uses a reconstruction-stencil-bounded
  total-pressure ratio in the local orthonormal face frame to select its HLL
  parent at strong grid-aligned shocks. It is inactive in 1D, contacts,
  uniform flow, curved-spacetime smokes, and the smooth AMR control. The
  rotated Balsara-1 CT defect is `4.07e-8` for stabilized HLLC versus
  `3.65e-11` for HLLD; do not promote HLLC over HLLD.

## RMHD HLLD Boundary

- Production HLLD implements the Mignone--Ugliano--Bodo five-wave SRMHD fan in
  one face-normal orthonormal frame. It solves the nonlinear total-pressure
  closure, samples the fan at the shift-driven interface, and transforms the
  Valencia flux back to the coordinate frame.
- Nonfinite, unordered, inadmissible, or nonconverged HLLD constructions use
  the explicit HLL limit. The current counter records solver evaluations,
  including discarded high-order trials, repeated corrected-face evaluations,
  and duplicate nodal FAB instances; zero is meaningful, but a positive value
  is not a count of unique final selected faces. This is a documented robust
  limit, not an unreported fallback.
- Exact contacts, rotational waves, zero-normal-field degeneracy, random fan
  audits, fixed curved backgrounds, evolved Z4c, uniform CT, and AMR+CT provide
  useful validation. Across the exact Balsara tubes HLLD's improvement over HLL
  is real but modest; a complete-wave fan cannot represent a subcell compound
  structure by itself.
- The current validation is not an all-regime invariant-domain, positivity, or
  entropy proof. Strong-magnetization intermediate thermodynamics and the
  coupled CT/source/AMR discretization remain audit boundaries. Do not narrow
  required signal bounds or tune the pressure closure to exact tube speeds.
- Standard entropy-stable RMHD derivations often use a Powell-like source.
  Their proof does not transfer to the present conservative staggered-CT path
  by changing a face flux alone.

## Exact And Analytic Reference Semantics

- A finite-volume profile comparison matches numerical conserved cell averages
  to exact conserved cell averages. For nonlinear systems, integrate the
  conserved map; do not average primitives or evaluate the conserved map at an
  averaged primitive. Recover exact primitives through the same C2P
  interpretation used for numerical cell averages.
- A fixed continuum exact solution is the physical authority in plots. Its
  finite-volume projection legitimately changes with cell width and phase and
  is the primary discrete error reference, but it must not be plotted or
  described as a resolution-dependent continuum solution.
- A supplementary norm may integrate the difference between the fixed
  continuum profile and the numerical piecewise-constant representation. It
  includes unavoidable within-cell representation error and is not the
  primary finite-volume order metric.
- Tabulated Riemann data require discontinuity-aware integration. Repeated
  coordinates encode jumps and must not be smoothed by ordinary interpolation.
  Reject nontrivial coordinate reversal. Normalize errors using the complete
  physical profile and compare TV on the same mesh.
- The Giacomazzo--Rezzolla files under `../exact_balsara/` are independent
  exact SRMHD references. Their pressure column is total pressure, so validation
  subtracts comoving magnetic pressure before constructing Valencia state.
  Their checked-in Balsara 2--5 generator used decimal `1.6666` although the
  published tests specify `5/3`; retain that provenance caveat.
- A discontinuous center-sampled IC can use a continuum finite-volume
  reference only when its jumps lie on cell faces. For translated discrete
  data, compare against a conservative translation of the initialized cell
  averages rather than a newly center-sampled continuum profile.
- Smooth analytic providers that initialize primitives at centers validate that
  discrete initialization. Do not silently relabel them high-order FV
  projections; changing initialization requires changing the analytic
  reference consistently.

## Balsara And Reconstruction Lessons

- The Balsara-3 dense feature is a material contact followed closely by another
  MHD front. Its physical width is about `0.011396` on the standard `[-1,1]`
  domain, so it spans less than one cell at N=128 and only a few cells through
  N=512. A homogeneous-cell FV state cannot represent both boundaries and the
  intermediate plateau while it is subcell.
- More diffusion is not a shell-resolution mechanism. PWC, strict MINMOD, and
  the tested density THINC/BVD variants broadened or lowered the shell. The
  demonstrated remedies are adequate resolution, state-following AMR, or a
  genuinely validated multi-interface in-cell representation.
- Production MP5/SSPRK3/HLLD converges monotonically on uniform Balsara-3
  refinement. State-following AMR recovers the separate structures with far
  fewer active cells than a uniform finest mesh while maintaining scalar and
  recovery gates.
  This is a focused one-dimensional result, not blanket AMR superiority.
- Balsara 4 is not one generic ringing mode. Much of its alternating structure
  is a cluster of physical transverse waves whose amplitude and phase are
  smeared. A scalar shock detector, entropy minimum, temporal second difference,
  or independent component limiter did not reliably separate it from legitimate
  Balsara-3/5 fan evolution.
- A continuous high-order sensor needs an exact smooth plateau. A weight merely
  close to one can accumulate damping over many reconstructions. Shared state
  order is safer than independently selected characteristic order when strong
  MHD/RMHD components remain coupled.
- Offline physical-wave carriers, mode-selective limiting, THINC/BVD,
  constrained-fan, and ADER-Prim/MOOD/CWENO experiments are useful Pareto or
  falsification controls. None currently dominates the production method over
  HD, MHD, SRHD, and SRMHD held-out tests. Do not add one to the production
  catalog from a favorable aggregate Balsara score.
- The closed standalone reconstruction-research drivers were removed after
  their conclusions were distilled here. Git history retains those experiments;
  do not recreate or restore that prototype matrix without a new falsifiable
  hypothesis and a clear route into the common HD-to-GRMHD production owner.
- When evaluating a hybrid, compare the same-grid exact/reference, pure high
  method, pure low method, and hybrid over complete accepted timesteps. Include
  activation and reset counts. One Euler candidate is a causal diagnostic, not
  a method ranking.

## AMR Transfer And CT

- AMR follows the standard finite-volume contract: FillPatch/prolongation,
  flux-register reflux, average-down of covered cells, and for CT edge-EMF
  reflux plus face average-down. These operations are not interchangeable.
- PyFluid already routes all face components together through AMReX's
  `face_divfree_interp`. AMReX 26.07 generalized that interpolation so fine
  ghost-cell divergence matches a spatially varying underlying coarse-cell
  divergence and supports a refinement ratio of 2 or 4 in every direction.
  PyFluid rejects other CT ratios before evolution and still requires isotropic
  CT ratios because its edge-EMF reflux/register path lacks anisotropic
  evidence. Conservative quartic state interpolation remains ratio-2-only;
  ratio 4 uses coupled-conservative state transfer. Keep the existing AMR+CT
  validation before changing transfer ownership; the upstream improvement is
  not permission to add a second CT prolongation path.
- Full GRMHD does not have one universal "tag every evolved field" policy.
  Production evidence separates geometry/vacuum tracking, matter support,
  object-following boxes, and targeted magnetic/shear science criteria. The
  catalog therefore keeps `lohner` as an explicit broad flow-study policy,
  uses floor-guarded density for `matter`, strong-field support
  `abs(chi - 1)` for `spacetime`, and their union for `compact_object`.
  `spacetime` and `compact_object` default to threshold 0.15 and deliberately
  do not follow a small outgoing gauge tail. Broad `lohner` still measures
  conformal-factor curvature, with the dimensionless Minkowski scale one in
  its smoothness floor. Adding a diagnostic GF never makes it an AMR driver
  implicitly; `chi` alone is not a universal matter criterion.
- A floor-relative support guard of only `10 rho_floor` is still too permissive
  for compact-object accretion: the q-power Kerr-disk run promoted an expanding
  `rho ~ 1e-11` numerical-atmosphere halo until level 2 requested 3.94 million
  cells. `compact_object` therefore uses `10^4 rho_floor`, which is about
  0.4 percent of that disk's peak density; the explicitly broader `matter`
  policy retains `10 rho_floor` when low-density flow is itself the target.
- Conservative component averages do not guarantee relativistic
  admissibility. A coarse parent can produce children whose exact component
  average is correct while one child recovers with negative pressure.
- For Cartesian analytic fixed spacetime with CT off, the production transfer
  guard applies
  `U_f = U_c + theta (U_f_raw - U_c)` with one maximal C2P-admissible `theta`
  shared by every child and every conserved component. A conservative base
  therefore retains the parent average, and an already admissible family is
  unchanged. A nonfinite child or inadmissible parent is an error, not input to
  an atmosphere repair.
- AMReX can request only part of a coarse parent's child family because FAB and
  ghost regions are split independently. A sibling-coupled limiter must expand
  to a sibling-aligned region, run the base interpolation and limiter on the
  complete family in private scratch, then copy only the requested region.
  Limiting only the visible subset makes results decomposition-dependent and
  invalidates the family-conservation argument.
- CT-on transfer needs a metric-aware limiter with canonical face `B` held
  fixed. Stored or evolved spacetime needs the corresponding stage metric; a
  Minkowski surrogate is not a valid admissibility test.
- Face-centered magnetic fields are the canonical CT state. Euler, corrector,
  midpoint, SSPRK3, and RK4 now construct the appropriate intermediate face
  field from the step-start field and stage EMF. Stage primitive recovery and
  reconstruction use its face averages without rewriting conservative
  momentum/energy. Valid, same-level, and periodic ghost cells use canonical
  faces; uncovered coarse-fine ghosts retain the AMReX FillPatch cell state.
  The final update reapplies the RK-integrated EMF to the saved start field,
  then synchronizes cell `B` and performs accepted-state recovery.
- After reflux, average-down, final face-field synchronization, and regrid, the
  hierarchy refreshes metric coupling and applies relativistic acceptance at
  the last state owner. Newly created or remade fine levels then copy the
  accepted state consistently into both current and old-time storage.
- CPU-tiled numerical kernels must launch over `tilebox`; `validbox` is the
  whole FAB and is only metadata inside a tiled loop. Face-centered
  `tilebox()` also supplies unique owned faces. Four Newtonian face-flux
  owners, four masked flux/update owners, and the patch-wall flux owner once
  launched over `validbox`; with the default 3D `32x8x8` CPU tiles, a `32^3`
  FAB could be recomputed 16 times. Correcting those owners raises the
  `64^3`, 16-step MPI-only control from `5.26` to `26.13` Mcells/s and the
  OpenMP control from `8.29` to `40.19`, without changing its reported physics
  diagnostics. Grown ghost writers use `growntilebox`; manually growing every
  internal tile creates overlapping OpenMP writes. Device-side temporary FABs
  must retain an `Elixir` through asynchronous interpolation/copy chains.
- The default local AMReX build remains MPI-only, while a separately managed,
  explicitly selected MPI+OpenMP companion uses the same AMReX pin, physics
  matrix, and numerical code. In the `320x320` 2D spherical control, two ranks
  by four threads gives `10.67` Mcells/s versus `9.21` for four MPI ranks; two
  by five is within about one percent of that optimum. In the corrected
  `96^3`, PWL/HLLC/SSPRK3 control, `1x1/1x4/1x10/2x4/2x5` give
  `2.52/8.16/10.50/9.81/10.49--10.61` Mcells/s. The explicit local OpenMP
  profile therefore uses two ranks by five threads as the cross-dimensional
  compromise. Keep reduced ID on the MPI-only profile unless its custom dense
  loops gain coarse parallel ownership: merely enabling AMReX OpenMP adds
  overhead there.
- AMReX's static outer-`MFIter` threading is the useful host boundary. Mapping
  each inner `ParallelFor` to synchronous GCD is too fine-grained. Dynamic
  `MFIter` scheduling is exact but is `5--7%` slower in the same uniform-tile
  control, so static scheduling remains selected. The local LLVM OpenMP runtime
  does not expose reliable macOS affinity; let the OS place workers rather than
  encoding Apple core IDs.
- Apple `powermetrics --show-process-amp` can attribute a process's scheduled
  CPU time to performance cores directly; efficiency-core time is total CPU
  time minus that value. In the corrected sustained 3D control, median PyFluid
  occupancy is `3.93P+0.02E` for `1x4`, `3.66P+3.80E` for `1x10`,
  `3.86P+2.38E` for `2x4`, and `3.81P+3.39E` for `2x5`; active clocks remain
  about `3.94/2.8 GHz`. With E cores worth roughly `0.35--0.40` P cores on this
  integer-heavy workload, all-core throughput is consistent with the actually
  scheduled weighted capacity. Static AMReX/OpenMP therefore uses the E cores
  usefully; lack of full six-E-core occupancy and memory/barrier costs explain
  the remaining gap to a theoretical `4P+6E` limit better than an affinity or
  thermal failure.
- Apple's M6 CPU has 2 super, 4 performance, and 6 efficiency cores. The
  corrected tile-owned OpenMP path should let macOS schedule PyFluid across all
  three classes, but the existing M4 `2x5` profile is not an M6 optimum by
  construction. Benchmark `1x12`, `2x6`, `3x4`, and the retained `2x5` control
  with power attribution before selecting an M6 profile. Static equal-tile
  scheduling can wait on slower workers; revisit dynamic/guided tile ownership
  only if measured M6 heterogeneity outweighs the M4 dynamic-scheduling cost.
- AMReX 26.09 still asserts that `MLCurlCurl` has one AMR level, and its 1D
  `MLNodeLaplacian::reflux` still aborts. The corresponding implicit-CT and 1D
  nodal-solve restrictions remain real; do not remove them based only on the
  version update.
- The present transfer limiter has CPU evidence. Retrieving a device failure
  flag after every interpolater call creates a likely GPU synchronization
  point; batch status at a FillPatch owner before claiming GPU performance
  neutrality.
- Runtime configuration rejects diffusion and viscosity for RHD/RMHD, so a
  separate relativistic post-diffusion recovery path is not currently
  reachable. Revisit accepted-state ownership when relativistic diffusion is
  introduced.

## Stellar Initial Data And Evolution

- The self-contained rotating-star path supports 3D and reduced axisymmetric
  spherical `r theta` XCTS with uniform, relativistic j-constant, and
  relativistic q-power rotation for cold polytropic/hybrid-polytropic EoS. It
  continues a TOV seed through one coupled under-relaxed matter/constraint
  iteration. Zero angular velocity bypasses the rotating update and reproduces
  the same-EoS TOV endpoint.
- The generated rotation-law catalog also owns the two post-merger
  `uryu_plus_13` and `uryu_j_envelope_p1` laws. Their generated angular
  velocity, derivative, first-integral identity, sign, and j-constant limiting
  controls pass pointwise. The maintained adaptive `r theta` gate also solves
  a nonzero `uryu_plus_13` model and checks `uryu_j_envelope_p1` in its large-B
  j-constant limit, including artifact identity and physical-diagnostic sanity.
  These isolated ID checks are not a validated Uryu sequence, independent
  physical-reference comparison, or Uryu evolution test.
- A zero-temperature barotrope has one independent thermodynamic variable.
  Scaling pressure at fixed density is therefore off the cold EoS unless a
  separate thermal degree of freedom is introduced. PyFluid's
  `cold_pressure_scale` instead uses the scaled pressure as a target, inverts
  the same cold EoS for a new density, and rebuilds internal energy, enthalpy,
  sound speed, and conserved state; its baryon mass changes. It is a small
  evolution-start perturbation with no constraint re-solve, and reusable
  equilibrium artifacts are written before it is applied.
- Font-style controlled collapse is deliberately off the cold curve: increase
  density with the radial cosine seed, retain the equilibrium pressure, and
  recompute specific internal energy through a hybrid-polytrope or Compose EoS.
  Cold-projecting the denser state instead adds pressure support and changes
  the intended problem. During a static spherical XCTS polish, preserve both
  densitized baryon density `psi^6 rho` and specific internal energy, then
  restore pressure from the runtime EoS after each conformal correction.
- Automatic AMR has two independent controls: the physical refinement target
  says what spacing the state needs, while `amr.regrid_int` says how often a
  new hierarchy may be considered. For subcycled full GR, synchronized mode
  counts accepted coarse steps. Keep hard `max_level` above the physical
  target when validation requires emergency headroom; otherwise reaching the
  hard cap can masquerade as successful adaptive refinement.
- Existing nonzero-spin evidence is reduced-grid continuation and short
  evolution, not a resolution-converged mass-shedding limit, long-term
  equilibrium, or constant-baryon-mass sequence. The surface-gradient
  indicator required ghost exchange across FABs; results predating that fix
  are not physical endpoint evidence.
- The signed surface-gradient ratio diagnoses the equatorial mass-shedding
  cusp. It is neither a linear measure of remaining angular velocity nor a
  radial/axisymmetric stability criterion; the latter needs an appropriate
  fixed-baryon-mass sequence, turning-point analysis, and time evolution.
- At fixed central density `rho_c=1.28e-3`, independent 128x32, 256x64, and
  384x96 SLy sequences converge through `Omega=0.032`. Their
  `chi_ms=0.18439/0.17968/0.17868` and the global charges converge cleanly.
  The 384x96 continuation remains physical at `Omega=0.0329`, with
  `chi_ms=0.05144`, `M_ADM=1.51433`, `M_b=1.65248`, `J=1.53834`, and
  `J/M^2=0.67083`; at `0.033` the matter iteration reaches tolerance but the
  surface diagnostic is invalid, so no model is accepted. By contrast, the
  256x64 and 128x32 models accept `0.033` with `chi_ms=0.00685/0.03978`.
  This brackets the refined numerical endpoint in `0.0329 < Omega < 0.033`
  while demonstrating that the tiny coarse equatorial slope is not a
  resolution-converged physical mass-shedding frequency. Direct cross-grid
  near-limit continuation is also more fragile than an independent sequence.
- The identical N=24, small-spin XCTS solve takes 171.37/95.61/67.69 seconds
  on one/two/three local MPI ranks, for 89.6/84.4 percent efficiency at two/
  three ranks. This establishes useful local spatial scaling, not large-grid or
  multi-node efficiency.
- Matter, ADM-volume, and ADM-surface angular momentum should agree under
  refinement. ADM--Komar disagreement on the tested compact domains is largely
  a finite-radius extraction effect; do not tune the elliptic solve to force
  equality at one outer radius.
- A centered nonrotating TOV model can use x/y/z octant reflection. A star
  rotating about z can use z-equatorial reflection, but ordinary x/y plane
  parity is not a symmetry of its azimuthal velocity and shift.
- CTT and CTTK can reuse the rotating matter layer, but still need a validated
  stationarity-compatible free-data closure before being called rotating
  equilibrium methods. A deliberately nonzero-mean-curvature CTTK slice is not
  the zero-spin XCTS/TOV endpoint.
- The first magnetized physical-ID owner is a fresh single-level 3D Cartesian
  RMHD+CT stellar XCTS solve-only path. It constructs an edge-centered
  pressure-confined vector potential, takes the existing CT discrete curl,
  stores canonical face fields, normalizes the maximum comoving field, and
  includes RMHD magnetic energy, momentum, and stress in every constraint
  fixed-point update. At requested `b_max=1e-4`, the SLy control has artifact
  `divB=4.52e-21`, constraint residual `5.83e-16`, and a finite
  `8.65e-6` ADM response relative to the matched unmagnetized solve. This is
  constraint-consistent magnetic seeding, not magnetohydrostatic force balance.
- The separate `poloidal_grad_shafranov` owner solves the pure-poloidal CFC
  vector Grad--Shafranov equation and the magnetic Bernoulli first integral
  inside the coupled stellar/XCTS fixed point. Its free function is
  `M(A_phi)=k_pol(A_phi+xi A_phi^2/2)` with zero toroidal current. The Cartesian
  unknown is `A_phi e_phi/varpi`; transient cell/edge potentials remain solver
  scratch, while the artifact owns only `sqrt(gamma) B^i` on CT faces and
  physical `B^i` in portable primitives. A final metric-aware sync makes exact
  same-grid evolution caches agree with face averages to roundoff.
- An oriented poloidal field admits a Cartesian reflection only through the
  plane normal to its magnetic axis (the usual equatorial half-domain). A
  transverse plane reverses the axial magnetic moment and is not a symmetry;
  reject it rather than treating a dipole as an unmagnetized TOV octant.
  Cold/hybrid/CompOSE RMHD artifacts now pass the common runtime C2P/P2C
  preflight and may enter evolution on an exact layout. Cross-grid and
  cross-chart CT materialization remain unsupported.
- A schema-v3 constructed magnetic artifact can be loaded and reserialized on
  its exact layout without rerunning the magnetic equilibrium. The reader must
  restore portable primitives even when exact evolution caches exist and must
  preserve every magnetic construction parameter. The fresh nonlinear SLy TOV
  control takes about 75 seconds on three local ranks while its exact reload
  takes 0.17 seconds and preserves authoritative fields exactly; its derived
  evolution-fluid cache agrees to `1.59e-17` relative. Changed-grid transfer,
  source-IC continuation, and torus composition remain rejected because each
  needs a target-grid force-balance owner. Exact magnetic reload also preserves
  the saved reflection closure; changing an equatorial symmetry plane into an
  ordinary boundary is rejected.
- A forced physical-state check refreshes the live RMHD primitive cache from
  conserved state and can change cell-centered magnetic primitives at
  roundoff. An unchanged exact-layout solve-only reserialization therefore
  restores the already-validated portable primitive core immediately before
  writing. Continuation, changed-grid, rigid-transform, and added-component
  paths instead serialize their newly materialized primitives.
- The same pure-poloidal Grad--Shafranov owner is validated with the bundled
  cold CompOSE SLY4 table: the focused `24^3` control has XCTS residual
  `6.26e-15`, Grad--Shafranov residual `1.90e-6`, Bernoulli residual
  `5.55e-7`, and CT divergence `4.57e-20`. Its exact-layout t=0 evolution
  handoff uses the same runtime CompOSE RMHD recovery owner; long stellar
  equilibrium remains a separate evolution validation boundary.
- On the coarse `24^3`, radius-16 Cartesian RNS control, the finite-domain ADM
  diagnostic gives `M_ADM > M_b` by about 1.5 percent in both the magnetized
  and matched unmagnetized models. The nonlinear poloidal field changes that
  ADM value by only `5.5e-5` relative. Do not use the mass ordering from this
  coarse diagnostic as a magnetic-equilibrium acceptance test; use the XCTS,
  Grad--Shafranov, Bernoulli, CT, and matched-response checks, and require a
  larger-domain/refined ADM extraction before interpreting binding energy.
- The reusable physical-ID container and sequence behavior are defined in
  `docs/infrastructure_conventions.md`. Materialization onto an evolution grid
  is interpolation, not a new constraint solve, so finer evolution cannot
  recreate unresolved ID information. A changed-grid reduced solve-only RNS
  load re-solves XCTS even at unchanged angular velocity; a same-map load keeps
  the direct sequence path.
- Physical-ID manifest keys and runtime component vocabulary are declared in
  `schema/physical_id.toml`; generated C++ rejects unknown/incomplete current
  manifests and generated JSON feeds reporting. Artifact visibility is atomic:
  fields, exact caches, solved star/disk diagnostics, and component records are
  completed under a hidden `.pending` sibling before a same-directory rename.
  Fresh 256x64 RNS, aligned coupled/test-fluid disk, Kerr--disk, reduced TOV,
  and black-hole round trips pass with no surviving pending directories. The
  component records describe provenance and poses only; they do not license
  superposition of accepted self-gravitating fields.
- Physical-ID schema v3 makes the source chart independent of the target
  executable: dimension, coordinate family, ordered active axes,
  periodic/reflection closure, and orthonormal tensor basis are runtime
  metadata. Scalars, polar/axial vectors, and symmetric tensors transform
  through Cartesian physical components; this adds one compiled materializer
  owner rather than a coordinate/formulation/EoS template axis. Exact
  same-layout caches remain optional accelerators. Stellar rotation axis and
  center are persisted; one proper rigid transform may rotate/translate an
  entire isolated source during portable materialization and disables an
  otherwise matching exact cache. It cannot change relative component tilt.
- Native single-object ID ownership is cataloged once in `schema/methods.toml`:
  natural dimension/chart/axes, broad object and disk capabilities, grid kind,
  relative tilt, and reflection class. The generated descriptors are host-only
  policy and do not create a coordinate/object/formulation template product.
  Physical combinations that are not a simple capability remain explicit C++
  validation. A full `r theta phi` materialized artifact is not by itself
  evidence for a native curvilinear nonlinear continuation owner. The native
  3D spherical scalar operator and axisymmetric black-hole XCTS shell are now
  separate explicit owners; neither is a general nonaxisymmetric three-vector
  spherical XCTS solve.
- Axisymmetric stellar, aligned-torus, and Schwarzschild-XCTS owners now have
  both reduced `r theta` and full-storage `r theta phi` routes. The 3D owner
  solves the same separable physical `r theta` operator once and broadcasts it
  over periodic phi storage; it does not invite nonexistent nonaxisymmetric
  modes into the linear solve. At adaptive `96x24`, the 2D, `Nphi=1`, and
  `Nphi=4` RNS artifacts agree to `1.71e-15` in stored fields,
  `4.34e-14` in global diagnostics, and `1.67e-23` in phi invariance. The
  `96x24x4` rotating solve completes in about 1.3 seconds on three local ranks
  with constraint residuals below `3.3e-13` and zero recovery.
- Periodic cell-centred physical-ID interpolation must wrap its stencil across
  the coordinate seam. Treating the half-cell strips around phi=0/2pi as a
  nonperiodic interpolation exterior changed only six 16-cubed Cartesian RNS
  matter cells, but the maximum density error was `5.29e-4` because those
  cells were replaced by atmosphere. With periodic index wrapping, matched
  native-2D and `96x24x4` sources materialize onto 16-cubed Cartesian data to
  `2.59e-15` normalized field error and remain within `1.66e-15` through one
  complete SSPRK3 step at CFL 0.9, with zero recovery/reset.
- Independent Cartesian and reduced-spherical XCTS are work-precision
  controls, not roundoff-equivalent storage paths. For the same SLy,
  central-density, omega=0.001 model on radius 16, refining the direct
  Cartesian solve from 16-cubed to 24-cubed decreases every tracked fluid/ADM
  difference against the spherical owner; the worst ratio is `0.774`, but the
  24-cubed pressure difference is still `9.67e-2`. The reduced spherical owner
  remains the efficient production ID route; this coarse Cartesian comparison
  is convergence evidence, not high-accuracy agreement.
- Matched `96x24` and `96x24x4` adaptive Schwarzschild-XCTS exterior shells
  agree to `1.09e-15` in fluid/ADM fields with zero phi variation, and both
  pass the generic Cartesian materializer. This validates the coordinate
  projection and basis handoff only. Constraint-solved black-hole evolution
  still lacks an explicit full-grid interior-fill owner or true excision, so a
  shell-to-full-grid probe must not be promoted as reusable production BH ID.
- Applying generic 3D MLMG to an origin-containing, phi-redundant axisymmetric
  stellar solve excites unused azimuthal modes and diverges for `Nphi>1`, even
  for a static TOV and exact radial seeding. The separable owner is the
  physical and faster remedy. The generic full-spherical operator remains
  validated for smooth regular nonaxisymmetric manufactured solutions and
  shell domains. A general nonaxisymmetric spherical three-component XCTS
  vector operator remains open; tilted matter stays with full 3D Cartesian
  XCTS.
- `initial_data.solve_only=1` performs the normal ID materialization and
  relativistic C2P/P2C preflight, writes the portable artifact, and exits before
  evolution. Validated reduced paths include spherical 1D TOV and spherical
  2D circular RNS to Cartesian 3D.
  At source `N_r=256/512/1024`, the maximum normalized density error against a
  direct 32-cubed TOV decreases `7.58e-4/3.52e-5/1.16e-5`; pressure, diagonal
  metric, and lapse errors also decrease monotonically. An `N_r=1024` source
  materialized onto `[-40,40]^3` has maximum tracked error `1.25e-5`, the
  regular origin density differs by `9.89e-6`, and a three-step CFL-0.9 loaded
  evolution has mass drift `3.51e-12` with zero recovery.
- The shared star-adapted radial map uses uniform physical spacing through an
  enclosing matter region and a geometric gravity exterior, with one third of
  radial cells reserved for the exterior. At `N_r=256`, adaptive TOV-to-32^3
  errors are density `4.98e-5`, pressure `9.24e-5`, diagonal metric `8.90e-6`,
  and lapse `4.77e-6`: all are 2.9--15.2 times smaller than uniform `N_r=256`
  and within 1.2--1.4 times uniform `N_r=512`. This is a demonstrated
  work-precision point, not residual-driven AMR.
- A 256x64 reduced SLy RNS artifact materializes onto a full three-rank 32^3
  Cartesian hierarchy with maximum primitive and conserved round-trip errors
  `7.16e-13` and `1.79e-13`. Constraint-solved data must perform the final P2C
  only after nodal ADM fields have been reconstructed at authoritative cell
  centering; using a provisional metric can make otherwise valid cold surface
  primitives fail C2P.
- A saved adaptive `128x32` +z RNS rigidly rematerialized about +x and translated
  to `(1,-2,0.5)` on a correspondingly shifted `16^3` Cartesian grid preserves
  the sorted density/speed profiles to `1.50e-11` absolute, gives exactly zero
  velocity along the new axis, passes C2P/P2C at `9.03e-14`, and uses zero
  recovery. Pointwise controls separately cover tensor rotation and the
  antiparallel-axis construction.
- Adaptive 256x64 small-spin RNS uses one quarter the cells of uniform 512x128,
  runs in `2.115 s` versus `7.467 s`, and differs in ADM mass by `6.75e-6`
  relative (`J/M^2` by `5.68e-4`). Cartesian 32^3 materialization retains
  primitive/conserved round-trip errors `4.31e-13/1.93e-13`. A saved mapped
  128x32 model remapped to a 96x24 adaptive grid and continued to the next
  rotation target agrees with a matched direct tight-tolerance solve to at
  most `8.08e-6` in global charges; explicit radial faces therefore support
  same/different-grid sequence continuation and evolution handoff.
- A cell-centred radial source's half-cell gap at `r=0`, and the corresponding
  strip adjoining a declared reflection plane such as the RNS equator, are
  regular interior interpolation rather than stellar exterior. Finer target
  grids must not turn either strip into physical exterior. Beyond a spherical
  TOV source radius, fluid uses the configured target relativistic exterior
  and ADM geometry uses the exact isotropic-Schwarzschild exterior inferred
  from the saved outer metric.
- The cell-centred spherical `r theta` scalar and azimuthal spin-1 operators
  own origin, axis/equator reflection, radial Robin closure, coupled XCTS
  sources, longitudinal-tensor reconstruction, matter iteration, and global
  diagnostics without adding a coordinate/formulation template axis. Their
  manufactured solutions converge at second order and agree across one and
  three ranks. Cross-chart CT remains rejected until face-centred
  divergence-preserving transfer is implemented.
- A nodal spherical solve makes every angular point at `r=0` a duplicate
  physical degree of freedom. Use a cell-centred radial origin for the reduced
  scalar operator. Stock point-smoothed Cartesian multigrid was not robust for
  the strongly weighted spherical-axis operator tested here; the separable
  solve diagonalizes the self-adjoint angular operator and solves independent
  radial tridiagonal systems without adding a coordinate/template product.
- A circular stationary axisymmetric RNS has only the physical azimuthal
  shift and momentum component. That field is divergence-free, so the flat
  conformal-vector equation reduces to the toroidal spin-1 scalar Laplacian,
  rather than a general coupled three-vector solve. A manufactured `l=1+3`
  field converges at order `1.98048`, with `1.09e-6` fine-grid error and
  `5.64e-11` residual, and agrees across one and three ranks. Portable vectors
  retain three physical components even on a two-dimensional source mesh.
- The separable spherical operator, matter/source update, diagnostics, ADM
  integrals, centering, portable evaluator, and same-grid continuation all
  consume the same explicit radial faces. The first Cycle-290 map failed
  because it starved the gravity exterior; reserving one third of cells there
  fixes that ownership/performance tradeoff. The adaptive 256x64 sequence
  reaches `Omega=0.0329` in `57.95 s` with `chi_ms=0.05083` and
  `J/M^2=0.670946`, close to uniform 384x96 (`129.26 s`, `0.05144`,
  `0.670831`); continuation to `0.033` again loses a resolved enthalpy surface.
  Thus `0.0329 < Omega < 0.033` survives an independent mapped-grid sequence.
  Surface-fitted/multidomain refinement and residual-controlled angular-mode
  truncation remain promising isolated-star upgrades; general block-adaptive
  ID remains the appropriate later route for binaries.
- Explicit-face layout identity includes the coordinate faces, not only AMReX
  index boxes and physical bounds. Otherwise a requested remap can silently
  retain the source map. Rotating artifacts persist an enclosing
  `matter_outer_radius`; different-grid materialization evaluates the exact
  target atmosphere outside that support so tiny power-law-atmosphere
  interpolation error cannot create disconnected matter near the light
  cylinder. Nodal ADM temporaries need ghost-filled derivatives and final
  `OverrideSync`: overlapping nodal boxes formerly disagreed only in
  `K_(r phi)`, by order unity at box boundaries.
- The conventional reduced RNS map now keeps uniform source spacing to 1.8
  times the nonrotating seed radius and one third of cells in the gravity
  exterior; both are bounded runtime controls, while saved explicit faces are
  authoritative. At `Omega=0.0329`, same-map re-solves at `512x128`, `768x192`,
  and `1024x256` take `10.6/26.2/45.2 s`. Their mass-shedding indicators are
  `0.049055/0.048556/0.048369`; bulk charges and raw surface slopes show
  observed unequal-grid orders about `1.74--1.99`. The `1024x256` model has
  `M_ADM=1.513929`, `M_b=1.651972`, `J=1.538163`, and `J/M^2=0.671106`.
  Its losslessly compressed portable artifact is about 20 MiB and materializes
  onto a full three-rank Cartesian `32^3` hierarchy in `1.18 s`, with maximum
  primitive/conserved preflight errors `8.82e-13/1.12e-14`. This is a
  high-resolution numerical sequence point, not yet a physical mass-shedding
  limit.
- At `Omega=0.03295`, matched tight-tolerance `1024x256` and independently
  refined `1536x192` models give `chi_ms=0.0293406/0.0292918`; angular momentum
  differs by `5.5e-7` relative and the other bulk charges by roughly `1e-5`.
  The respective re-solves take `31.56/43.89 s`. Both `1024x128` and
  `1536x192` accept `Omega=0.032975`, with `chi_ms=0.0126918/0.0116318`, and
  both reach the `1e-7` matter tolerance before the resolved enthalpy-surface
  guard rejects `Omega=0.0329875`. The fixed-central-density numerical
  mass-shedding bracket is therefore
  `0.032975 < Omega_limit < 0.0329875` on two grids. This is not a radial-
  stability turning point; that requires an appropriate constant-baryon-mass
  or constant-angular-momentum sequence.
- At `Omega=0.02968`, an adaptive `1024x128` uniform SLy model is about 90
  percent of the lower endpoint bracket and has `chi_ms=0.34204`,
  `M_ADM=1.42917`, `M_b=1.55769`, `J=1.19629`, and `J/M^2=0.58569`.
  Its roughly 14 MiB artifact is the practical fast-rotation evolution seed;
  endpoint bisection is calibration, not the normal solve path.
- A direct literature sanity check anchors the reduced solver to the published
  fixed-central-density uniform `N=1`, `K=100` BU sequence rather than only to
  its own residuals. At adaptive `512x64`, the near-zero BU0 probe, BU4, and
  near-shedding BU9 differ from the tabulated ADM masses by
  `2.10e-4/2.22e-4/1.89e-4` relative. Their quasi-isotropic coordinate
  equatorial-radius errors are `3.30e-3/6.48e-5/9.11e-3`, and absolute
  `r_p/r_e` errors are `0/1.41e-3/4.42e-3`. BU9 has
  `chi_ms=0.01990` and `J/M^2=0.63682`. This is strong evidence that the
  current IDs are physically recognizable equilibria across the uniform
  sequence. It is not a precision same-equation comparison: the reference is
  full quasi-isotropic GR, while PyFluid currently solves conformally flat
  XCTS at finite outer radius.
- The reduced `r theta` XCTS owner can compose either a TOV or rotating-star
  central object with an optional unmagnetized barotropic equilibrium torus.
  Constant `ell=-u_phi/u_t` is a distinct disk law; uniform, j-constant, and
  q-power disks reuse one runtime-selected circular Euler-integral/root owner
  with the star without forming a new template product. `test_fluid` omits the
  torus stress-energy; `coupled` iterates both matter components with XCTS.
- A coupled adaptive `256x64` SLy TOV-plus-disk control with `ell=5.2` and
  requested inner radius 8 has disk baryonic mass `0.103830`, angular momentum
  `0.524952`, resolved support `8.474--33.915`, and elliptic residuals below
  `1.5e-12` in the scalar equations and `5.3e-14` in momentum. A matched
  `512x96` solve changes disk maximum density, outer radius, and baryonic mass
  by at most 1.65 percent. One- and three-rank disk diagnostics agree to the
  focused gate tolerance. This establishes a resolved self-gravitating XCTS
  model, not equivalence to a full quasi-isotropic KEH solution.
- The same disk component composes with a small-spin RNS in both gravity modes.
  Physical-ID metadata retain the disk identity, support, component mass and
  angular momentum through reduced-to-Cartesian rematerialization. A two-step
  `32^3` evolved-Z4c smoke at CFL 0.9 has rest-mass drift below `1e-10` with
  zero recovery. Direct 3D fixed-Kerr constant-l and j-constant/q-power
  artifacts also solve, reload, and evolve with zero recovery; misaligned Kerr
  spin/disk axes reject because they are not stationary circular equilibrium.
  The catalog q-power law is
  `j=A^2 Omega[(Omega_c/Omega)^q-1]`: `q=1` is j-constant and, for an
  asymptotically flat metric with `1 <= q < 2`, its outer angular-velocity
  exponent is `q_Omega=2/q`. Do not confuse the catalog exponent with the
  conventional `Omega proportional to R^-q_Omega` disk exponent.
- A bounded differential fixed-Kerr control uses catalog `q=1.05`
  (`q_Omega=1.905`), `Omega_c=0.1355`, `A=6 M`, and requested `r_in=10 M`.
  Its 85 MiB compressed `64^3` artifact has resolved support
  `14.098--25.981 M`, density maximum at `19.092 M`, maximum velocity `0.315`,
  baryonic mass `8.693e-3`, and zero recovery. Direct rematerialization onto a
  ratio-4/2 AMR box twice as wide reaches `t=51.753 M` in 24 CFL-0.9 steps
  with zero recovery/reset. It peaks and ends at 355,840 cells, 47.4 percent
  of its guard versus 2.10M for uniform `dx=1 M`, retains one level of
  headroom, and has corrected mass/energy residuals `4.28e-13/3.13e-5`; its
  maximum density decreases 3.58 percent. Larger changes in the catalog
  exponent require retuning `Omega_c`: the old fixed `Omega_c=0.15`, `A=6`,
  `r_in=10 M` q=1.1/1.25/1.5 probes reached the Cartesian boundary, but this
  is a parameter-family rejection, not evidence that q>1 disks are impossible.
  The atmosphere inside the filled hole reaches a large Lorentz factor late,
  so this remains a fixed-background test-fluid control, not a long-lived
  accretion production run. Long-term star/disk or black-hole/disk equilibrium,
  magnetized tori, cusp/overflow diagnostics, and cross-method comparison
  remain open.
- An equatorially reflected full-azimuth movie control rematerializes that
  q-power artifact onto a `32x32x16`, ratio-4/2 AMR hierarchy and evolves to
  `t=62.3269 M` in 32 CFL-0.9 PWL/HLL/SSPRK3 steps. It finishes with 221,184
  cells, one unused level, zero C2P fallback, zero atmosphere reset, and an
  external mass-ledger residual `9.23e-13`. The positive octant is a rendering
  cutaway only; the evolution retains odd azimuthal modes. A finer
  disk-`dx=0.4 M`/horizon-`dx=0.2 M` MP5/HLLE stress keeps the fixed Kerr
  horizon invariant but begins filled-hole atmosphere resets at step 20 and
  accumulates 5,482 by `t=24.19 M`. Do not promote that fine branch as a clean
  accretion run: it is direct evidence that refinement alone does not replace
  a better complete-grid interior treatment or true excision.
- True matter excision extends the fixed-Kerr q-power control to `t=550 M`
  with disk/horizon spacing `0.5/0.25 M`, CFL 0.9, and no C2P fallback,
  atmosphere reset, or interior intervention. Its fixed initial orbital
  period at the saved density maximum is `P_c=527.341 M`, obtained from
  coordinate transport velocity `alpha*v-beta`; the run covers only
  `1.043 P_c`, not hundreds of disk orbits. Use that same clock throughout
  evolution. The atmosphere-excluded `|C1|/C0` decreases `0.00749 -> 0.00530`,
  `|C2|/C0` reaches `1.73e-4`, and `C0` changes by `-1.21%`. The late mass
  upturn includes measured open-boundary influx, so this is neither a PPI
  detection nor an isolated-disk convergence result. Evolved-spacetime
  excision and long self-gravitating accretion remain unvalidated.
- BH-disk mass ratios must be labeled by gravity ownership and denominator.
  The axisymmetric apparent-horizon-shell owner now couples an aligned disk's
  energy, momentum, and stress trace into every XCTS matter/constraint
  iteration. In the `ell=4.25`, requested-`r_in=6`, gamma-two stress family,
  matched adaptive `320x80` source models bracket the clean eight-step
  CFL-0.9 fixed-XCTS handoff: `K=12.5` has `M_0=0.309183`, throat
  `M_irr=1.038853`, and `M_0/M_irr=0.297620` and evolves to `t=2.102 M` with
  zero fallback/reset; `K=10` has `M_0=0.313796`, `M_irr=1.040426`, and
  `M_0/M_irr=0.301603` but needs eight atmosphere resets at step one. The
  latter is not a clean endpoint. `spacetime.mass=1` is the bare/throat scale,
  not the backreacted horizon mass; its corresponding raw ratios are
  `0.309183/0.313796`. These are a numerical short-handoff bracket, not a
  physical mass limit or long-time stability result. The Cartesian run keeps
  the disk-sourced XCTS exterior fixed and uses a finite smooth interior fill;
  it is not an evolved-Z4c/full-NR massive-disk evolution. The largest
  constructed fixed-Kerr test-fluid torus still has the larger nominal raw
  ratio `0.5151`, but its stress-energy does not backreact and must not be
  compared as a self-gravitating result.
- Moving from marginal constant specific angular momentum
  (`Omega ~ varpi^-2`) to a Rayleigh-stable q-power law raises the demonstrated
  clean coupled BH-disk handoff boundary. The conventional `q_Omega=1.85` law
  is catalog `rotation_power_q=2/q_Omega=1.0810810810810811`. Saved coupled
  artifacts can continue this family at larger same-sign `Omega_c` while
  retaining the converged matter state. At `A=6 M`, `Omega_c=0.157/M`,
  `K=100`, and requested `r_in=6 M`, a physical-spin-zero re-solve on the
  adaptive `768x160` shell gives `M_0=1.075252`, `M_irr=1.064211`,
  quasi-local `J_H=9.12e-6`, Christodoulou `M_BH=1.064211`, and
  `M_0/M_BH=1.010375`; `chi_H=8.05e-6` and
  `H/L/M=1.56e-10/1.50e-10/5.18e-12`. Its equatorial-half `160x160x80`
  Cartesian materialization preserves those physical metadata with maximum
  conserved/primitive round-trip errors `5.41e-12/6.38e-11`, then completes
  eight CFL-0.9 fixed-XCTS steps to `t=2.030 M` with zero C2P fallback,
  atmosphere reset, or reconstruction/Riemann correction and raw open-domain
  mass drift `1.54e-5`. This is a short clean handoff at physical mass ratio
  one, not a long-time stability or physical maximum-mass result.
- The same adaptive `768x160` q-power BH-disk family has a clean constructed
  continuation through `Omega_c=0.300/M`, `M_0=4.624213`, throat
  `M_irr=1.403521`, and `M_0/M_irr=3.294723`, with
  `H/L/M=2.03e-10/1.48e-10/2.56e-11`. The point required smaller continuation
  increments and 426 coupled matter iterations; failed larger jumps were
  nonlinear-basin failures, not evidence of a physical endpoint. A `384x80`
  re-solve of the heavy source also fails, so that grid is below the observed
  convergence floor for this regime. These are initial-data facts only; no
  dynamical stability has been established beyond the clean evolved
  `M_0/M_BH=1.010375` model.
- In the Shibata et al. high-mass BH-disk benchmark, `M_BH,0` is inferred from
  apparent-horizon area and circumferential radii under vacuum-Kerr relations.
  It is not `M_irr` alone for a spinning hole. Native spherical BH IDs and the
  live apparent-horizon tracker now both retain area, `M_irr`, proper
  equatorial/polar circumferences, and the explicitly Kerr-inferred horizon
  mass/spin. The circumference quantity is Shibata's `M_C=C_e/(4 pi)`, not a
  distinct assumption-free mass. Native axisymmetric throats also integrate
  the exact-axial quasi-local `J_H` and retain Christodoulou
  `M_BH=M_irr sqrt(1+J_H^2/(4 M_irr^4))`; reports expose `M_0/M_irr`,
  `M_0/M_C`, and `M_0/M_BH` separately. Each saved
  BH-disk continuation step recomputes and owns these values. Existing
  artifacts written before this addition, including the `Omega_c=0.300`
  stress point, retain only area/`M_irr` and must not be silently relabeled.
  The `256x32` Schwarzschild shell has area/Kerr-mass relative errors
  `1.56e-3/8.81e-4`, improving by about four from `128x16`; its inferred
  zero-spin error is `2.83e-2` because inversion is ill-conditioned near zero.
  The same Schwarzschild control gives quasi-local `J_H=0` and Christodoulou
  spin zero exactly. The live tracker integrates the three flat-coordinate
  rotation generators about its measured centroid; on aligned Kerr this gives
  signed spin to `1.87e-4` at 16 polar points, but the catalog keeps a
  `coordinate_` qualifier. A generic distorted 3D horizon still needs a
  Cook--Whiting approximate-Killing-vector solve for coordinate-independent
  `J_H` and Christodoulou quantities.
- Apparent-horizon hierarchy sampling must transform geometry, not merely map
  coordinates. After mirroring a sample through Cartesian planes,
  `gamma_ij`, `gamma^ij`, and `K_ij` acquire one sign per tensor index and
  `d_k gamma_ij` acquires the derivative-index sign too. Metric derivatives at
  a symmetry boundary must use the parity-filled nodal ghosts; a generic
  one-sided physical-boundary stencil left a visible full/half finder gap.
  With both corrections, matched `32^3` and `32x32x16` Kerr searches agree in
  expansion norms, area, irreducible mass, and radii to roundoff.
- The native spherical coupled-BH XCTS throat and its finite Cartesian
  interior fill do not yet have equivalent live-finder behavior. The source
  throat area is well-defined, but the nonlinear Cartesian finder leaves its
  radial volume on the materialized artifact even after fixed local AMR
  reaches `dx=0.0625 M`. Treat this as an interior-fill/materialized-horizon
  compatibility limitation, separate from reflection sampling and from the
  clean fixed-XCTS fluid handoff.
- Source resolution still matters independently of nonlinear convergence: the
  `K=12.5` `256x64` source produced 24 first-step atmosphere resets after
  Cartesian handoff, while the `320x80` re-solve at the same physical defect
  scale produced none. The former strongly damped iteration counts are not a
  current performance boundary; the solver now reports the relaxation-
  independent defect and the retained heavy endpoint converges in tens of
  iterations with a suitable runtime relaxation.
- A relative star/hole--disk tilt removes the shared axial Killing symmetry.
  Tsokaros--Uryu--Shapiro and the later Tsokaros--Ruiz evolutions handle this
  honestly: assume approximate time stationarity and circular axisymmetry only
  about the disk axis, solve that Euler first integral together with the full
  gravitational elliptic system, and call the tilted result a self-consistent
  non-equilibrium starting point. KEH is their Poisson solver, not what makes
  the tilted fluid an equilibrium. They obtain massive models by continuing
  from a massless disk while targeting increasing disk maximum density.
- A rotating disk does not inherit all Cartesian reflections of a spinless
  central object: its axial angular momentum permits only the equatorial plane
  normal to the disk axis. Equatorial-half Cartesian evolution remains fully
  3D in-plane and retains arbitrary azimuthal m, but suppresses vertically odd
  warp/corrugation and symmetry-breaking modes. A matched fixed-Kerr torus
  `32^3` versus `32x32x16` pair completes two CFL-0.9 SSPRK3 steps with zero
  recovery/reset and symmetry-restored mass, energy, and axial angular
  momentum agreement below `8.2e-10` relative. The cell count halves, while
  the tiny measured wall time improves only `0.444 -> 0.378 s` because fixed
  overhead dominates.
- PyFluid now names that boundary explicitly. `equilibrium_torus` requires a
  genuinely aligned stationary composition; `tilted_torus` is a full-3D
  Cartesian, coupled-XCTS circular seed with independent stellar and disk
  rotation laws. The bounded `24^3` SLy closure at a 30-degree relative tilt,
  stellar `Omega=0.001`, disk `ell=5.8`, and requested `r_in=10` converges in
  52 matter iterations with Hamiltonian/lapse residuals near `2e-15`, momentum
  residual `9.84e-13`, causal disk speed `0.421`, ordered support
  `10.30--15.41`, and zero recovery. Its disk baryonic mass is only
  `1.41e-7`, so it validates orientation/coupling infrastructure, not a
  strongly self-gravitating tilted-disk regime.
- The native nonaxisymmetric elliptic owner remains 3D Cartesian. A completed
  tilted artifact materializes through the portable physical-ID interface on
  a full `r theta phi` chart and back to Cartesian with finite preflight error
  and zero recovery; 3D curvilinear continuation or disk insertion rejects
  before it can masquerade as a native spherical XCTS solve. Rematerialization
  must retain both disk metadata and rotating-star injection/surface/charge
  metadata so the result remains valid for either evolution or later sequence
  continuation.
- A two-level Cartesian coupled-XCTS disk solve with a `12^3` base and a fixed
  factor-two refined central box reproduces the matched uniform `24^3` support,
  maximum density, disk mass, and angular momentum within `5e-7` relative,
  then materializes onto a uniform `16^3` evolution grid with zero recovery and
  `1.85e-9` one-step mass drift. It takes `134.15 s` versus `91.52 s` for the
  tiny uniform control, so this proves adaptive ownership and handoff, not a
  work-precision speedup; useful Cartesian ID AMR needs larger problems or a
  tighter refined region.
- A reflected cell-centered initial-data hierarchy must let the cell-to-node
  transfer consume its parity-filled ghost cells. Treating the reflection plane
  as an ordinary bounded face selects a one-sided nodal closure and changes the
  discrete matter source even while the elliptic residual converges. With the
  corrected transfer, matched full and z-half `24^3`/`24x24x12` Cartesian TOV
  fixed-point updates agree to `4.7e-16` absolute, and aligned torus support,
  maximum density, reconstructed baryonic mass, and projected angular momentum
  agree within `6e-8` relative.
- Saved reduced-domain data need the same parity extension during
  rematerialization. Clamping an octant cell center to the coarse-grid
  interpolation hull lowered the expanded TOV central density by 5.15 percent
  and raised Hamiltonian Linf from `3.01e-4` to `4.92e-3`, despite the native
  full/octant solves agreeing. A centering-general parity ghost plus an
  explicitly reflected interpolation stencil restores octant-to-full scalar
  agreement to `2.64e-5` and Hamiltonian Linf to `3.01e-4`. The matched RNS
  full/equatorial-half scalar and surface errors are `2.57e-4` and `8.30e-7`,
  with zero recovery.
- Upsampling cannot improve the saved ID's differentiability. A `48^3` base
  plus one refined level materialized from the octant artifact preserves
  maximum density/minimum lapse to `1.72e-16` and passes preflight with zero
  recovery, directly exercising the reflected half-cell ghost. Its fourth-
  order target-grid Hamiltonian Linf is nevertheless `9.69e-2` because a
  piecewise-linear source is being differentiated on a grid twice as fine.
  This is a valid field-transfer control, not evolution-ready high-resolution
  constraint evidence. Solve the source at least as finely as the intended
  evolution region until a target-grid constraint-polish owner is validated.
- Differential rotation is validated over useful ranges, not to a physical
  endpoint. Q-power `q=1`, `A=10` reproduces j-constant fields and diagnostics
  to roundoff. Q-power `q=2` converges at `A=10`, `Omega=0.03` and at `A=5`,
  `Omega=0.0201`; held-out `q=1.5/3` branches reach `A=10`, `Omega=0.03` with
  `0.005` continuation spacing; j-constant `A=10` continues through
  `Omega=0.068` with reduced matter relaxation. A q=1.5 jump from `0.01` to
  `0.0201` was still decreasing at `2.14e-6` when the 128-iteration cap fired.
  Continuation-step, relaxation, or iteration-cap failure is a nonlinear basin
  boundary and must not be labeled mass shedding.
- A persisted rotating continuation must restore its converged Euler
  first-integral injection energy as well as rotation law, differential length,
  and q exponent. Falling back to the TOV injection constant can make the first
  continued XCTS right-hand side nonfinite even when the saved fields are
  valid.
- A matched `1024`-radial angular scan at `Omega=0.0329/0.03295` establishes
  distinct resolution roles. Relative to 256 theta cells, 48 cells are about
  5.9 times faster and retain ADM mass/spin within `3.5e-5/4.9e-5`, but their
  near-shedding `chi_ms` is 11.8 percent high. Counts 64/96/128 are about
  4.6/3.2/2.3 times faster and reduce the `chi_ms` error to 6.6/2.7/1.34
  percent. Every 48--128 case independently rejects `Omega=0.033` through the
  same resolved-surface guard. Use 64 for sequence discovery, 128 for endpoint
  work, and 256 as the angular-reference control.
- Offline parity-aware angular projection of the `1024x256` exterior gives
  maximum global ADM-field errors at the source-ball boundary of roughly
  `8.9e-6/1.8e-6/4.4e-7/1.1e-7` for `4/8/16/32` angular cells. Four cells are
  too aggressive near the star. Actual radius-dependent angular reduction
  needs a modal/multidomain interface and coupled residual test; COCAL keeps a
  uniform theta grid and therefore does not validate this algorithm for us.
- On finer same-domain physical-ID loads, outer target cell centers can lie in
  the half-cell strip outside the source cell-center interpolation hull. The
  reader must apply the field's physical exterior there. For TOV/RNS fluid it
  regenerates the configured atmosphere at the target point; copying the last
  source atmosphere value can fabricate matter support. This is distinct from
  the regular radial-origin gap above. Atmosphere/support classification also
  needs a roundoff tolerance rather than strict equality.
- Short rotating-star evolution is currently cleaner with PWL than MP5 at the
  surface. Do not weaken rest-mass gates to promote MP5; a conservative stellar
  surface treatment and longer evolution evidence are still required.
- The production-default MP5/HLL/SSPRK3 saved-RNS harness now passes eight-step
  CFL-0.9 `32^3/48^3` controls from the adaptive `96x24` source with zero
  recovery/reset, rest-mass drift `3.24e-9/1.42e-9`, and angular-momentum
  change `0.797/0.368` percent. A two-step midpoint restart agrees with the
  uninterrupted final checkpoint to maximum field difference `1.52e-19`, and
  a separate level-one AMR start passes. This is still short-evolution and
  does not erase the longer stellar-surface boundary above.
- The coupled three-dimensional TOV AMR smoke passes its two SSPRK3 steps at
  the production CFL 0.9 with physical checks, one active fine level, and
  rest-mass drift `1.64e-10`. The old input-level CFL 0.2 was not required for
  this path; this remains a startup smoke rather than a long-stability claim.
- The `24^3` magnetized SLy artifact passes exact-layout RMHD startup preflight
  at `2.33e-12/4.00e-12` conserved/primitive error with zero repair. Four-step
  experiments at CFL 0.9 nevertheless produce many surface atmosphere resets
  with both PWC/Euler and MP5/SSPRK3 because the star spans only a few cells.
  Treat this as an underresolved long-evolution boundary, not an EoS handoff
  failure and not a reason to weaken reset accounting.
- Native CompOSE support currently means one-parameter cold beta-equilibrium
  barotropes plus an explicitly configured gamma-law thermal extension; it is
  not finite-temperature or composition-dependent CompOSE. The immutable
  table view is shared by TOV/RNS construction, cold enthalpy
  inversion, relativistic pressure/energy closure, signal speeds, atmosphere
  projection, C2P/P2C, physical-ID materialization, and RHD/RMHD evolution.
  The RMHD contract is validated for the present cold-plus-thermal closure; it
  does not imply general finite-temperature or composition-dependent support.
- The cold reader uses the CompOSE `Q1=p/n_b` and
  `Q7=e/(n_b m_n)-1` quantities in `G=c=M_sun=1`, retains the tabulated total
  energy density, selects one table-specific baryon-mass normalization so the
  first point joins a vacuum polytrope with `h -> 1`, and uses a
  constant-sound-speed continuation above the retained causal prefix. The
  required `table.toml` states this physics/provenance contract. SLY4 and DD2
  retain their full source tables; APR retains 148 of 172 points before the
  causal continuation.
- The bundled SLY4/APR/DD2 TOV sequences agree with the providers' independent
  mass-radius files to maximum radius errors `0.6009/0.3273/0.1402` percent on
  the checked stable overlap; maximum-mass errors are below 1.5 percent. A
  `256x64` adaptive SLY4 RNS solves, saves, rematerializes to Cartesian, and
  evolves three CFL-0.9 SSPRK3/PWL steps with zero recovery. A `128x32` source
  passed ID/preflight but caused 48 first-step atmosphere resets after `32^3`
  materialization, so it is not the production handoff floor for this model.
  Physical-ID and checkpoint compatibility use the path-independent normalized
  table fingerprint and thermal gamma; the source directory is provenance,
  not identity.
- At fixed central density `rho_c=1.28e-3`, the cold SLY4 uniform sequence
  continues cleanly through `Omega=0.032`. A `256x64` model has
  `chi_ms=0.174849` and `J/M^2=0.644786`; a changed-grid `512x96` re-solve has
  `0.174208` and `0.644985`. Baryonic mass and angular momentum differ by
  `1.90e-4` and `2.15e-5` relative. One- and three-rank `512x96` diagnostics
  agree to roundoff, while the three-rank solve is 2.26 times faster on this
  Mac. Both reduced grids reject `Omega=0.033` only after the matter update
  reaches tolerance and the resolved enthalpy-zero surface is lost. Thus
  `0.032 < Omega_limit < 0.033` is a bounded tabulated-EoS numerical
  mass-shedding bracket, not a turning-point or constant-baryon-mass result.
- The `512x96`, `Omega=0.032` SLY4 artifact materializes directly onto `32^3`
  and `48^3` Cartesian grids and evolves to matched time `t=3.5` with CFL 0.9,
  SSPRK3/PWL/HLL, and zero recovery. Rest-mass drifts are
  `5.61e-9/5.51e-9`; angular-momentum changes are `0.837/0.511` percent and
  maximum-density changes are `0.837/0.423` percent. At the initial Cartesian
  state, Komar surface/volume mass differs by `5.4e-4`, matter/ADM-volume
  angular momentum by `3.3e-5`, and matter/ADM-surface angular momentum by
  `7.0e-4`. The known finite-radius ADM-surface/Komar gap is 5.5 percent and
  must not be tuned away. This interval is only 1.8 percent of a rotation;
  long equilibrium evidence remains open.
- Exact ingoing Cartesian Kerr--Schild ADM data are now available for an
  arbitrary subextremal Cartesian spin vector. The same physical provider
  seeds stored fixed-spacetime and evolved-Z4c consumers; the generic flux
  templates do not contain a Kerr branch. A complete-grid `filled` policy is
  exact outside a transition radius inside the horizon and regular at the
  center, but its arbitrary interior violates the constraints. Whole-domain
  Hamiltonian/momentum norms are therefore not exterior-vacuum error measures.
  Catalog-owned exterior Hamiltonian L2/Linf and momentum Linf now require the
  complete fourth-order stencil to lie at `r_KS>r_+`. In the one-step exact
  Kerr controls, N=32 to N=64 improves those norms by factors 3.33/4.91/4.09.
  Fixed-Kerr RHD/RMHD matter excision now owns inactive cells, a one-sided
  Valencia diode flux, zero interior RHS/C2P, excision-aware RHD/RMHD
  candidate repair, CFL/diagnostic/AMR masking, canonical CT face fields, and
  the inner mass/energy ledger. Evolved-Z4c excision now has a fixed-worldtube
  owner and short validation described below; moving cuts and long-time
  exterior convergence remain unvalidated.
- The analytic Kerr `filled` control C2-blends lapse, shift, and spatial metric
  to a regular Minkowski core. First derivatives include the blend's product
  rule; inverse metric/determinant and `K_ij=(D_i beta_j+D_j beta_i)/(2 alpha)`
  are recomputed from that completed geometry. Exact and filled paths share
  the kinematic completion and ADM component packing. This fixes the former
  zero derivatives/independently scaled Kerr K inconsistency, but does NOT
  solve the constraints or make the arbitrary interior a stationary solution
  of the vacuum Einstein equations. Stationary ADM identity and derivative
  tests are not an evolution-stability proof. Fixed-Kerr production accretion
  should prefer a causally validated interior excision boundary.
- Black-hole ID-to-evolution handoff exposes three useful interior policies:
  excision, complete-grid puncture/trumpet, and smooth fill. Underneath, only
  excision changes the active-domain topology; puncture and filled data both
  occupy a complete evolution grid. Wormhole puncture and direct trumpet are
  different initial slices of the puncture policy: ordinary moving-puncture
  gauge drives the wormhole end toward a trumpet, whereas direct trumpet data
  start close to that late-time geometry. For an exterior XCTS/COCAL solve,
  smoothing occurs only after the solve during target-grid materialization and
  must not feed back into the exterior elliptic solution. `Turduckening` is the
  literature synonym for this smooth fill, not a separate configuration or
  solver method.
- Fixed versus evolved spacetime is orthogonal to the interior policy. The
  production goal is dynamically evolved, matter-coupled Z4c wherever the
  combination is mathematically applicable. Fixed Kerr/Schwarzschild remains
  a test-fluid/reference limit; evolved solutions should approach it as matter
  backreaction tends to zero.
- The local WIP COCAL/ETK BHT `smooth` and `IDpar` fills split `K_ij` into trace
  and trace-free pieces, extrapolate and reproject the trace-free piece against
  the final filled metric, damp `tr(K)` inward, and zero interior hydro. They
  do not preserve `tr(K)` unchanged, solve the constraints, or derive the
  completed curvature from a stationary four-metric. Treat them as independent
  algorithm provenance for a trace-split turduckening option, not as validated
  production truth.
- A smooth fill can seed a subsequent global CTT/CTTK/XCTS constraint solve,
  but the elliptic correction is nonlocal: it generally changes the exterior,
  horizon, mass, and spin, and it does not by itself retain black-hole puncture
  topology. Freezing the exterior while solving only the interior also demands
  a genuine matching/gluing boundary problem, not the existing target-grid
  constraint polish. Therefore a constraint-resolved production contract uses
  either an exterior solve plus evolution excision, or a complete-grid
  puncture/trumpet formulation whose singular/topological part is built into
  the conformal ansatz. A trace-split fill may remain a validated compatibility
  option; an optional global projection must be named experimental and must
  remeasure the exterior charges and horizon rather than claiming to preserve
  the supplied black hole.
- The normal post-fill operation is not an additional elliptic repair: convert
  the completed ADM data to regular Z4c variables and evolve with
  moving-puncture gauge plus Z4c constraint damping. Fill error should remain
  horizon-contained or damp; outward leakage is an error to measure. Validate
  independence under fill radius/method and refinement using exterior-only
  constraints, horizon area/mass/spin, lapse, and exterior matter. Coordinate
  horizon radius may shrink or grow purely through gauge, so it is never a
  physical-size diagnostic by itself.
- AsterX supplies optional lapse-threshold limits on density, internal energy,
  and `Wv` specifically for recovery trouble inside a black hole; ETK GRHydro
  has a separate hydro-excision reconstruction path. These are provenance for
  one optional horizon-contained matter-stabilization owner, not for silently
  accepting general atmosphere resets. Such actions must be counted and must
  vanish from the exterior solution under refinement.
- Production black-hole matter handling does not generally mean freezing a
  stale hydro state. True hydro excision makes safely interior cells inactive,
  avoids C2P there, and closes the exterior budget through a pure-outflow inner
  flux. Complete-grid puncture/fill methods instead commonly bound recovered
  primitives in a region safely nested inside the apparent horizon: EoS-
  consistent density/pressure or internal-energy floors and ceilings plus a
  Lorentz-factor ceiling, with a magnetization ceiling for MHD when needed.
- On the matched `32^3`, `a/M=0.5`, three-step CFL-0.9 fixed-Kerr controls,
  three no-plot repeats give median bounded/excised RHD throughput
  `0.309/0.305` Mcells/s. Masked CFL increases excision simulated-time
  throughput to `3.37` from `3.20 M/s`;
  excluding the inactive atmosphere from CFL is therefore part of excision
  ownership. RHD and divergence-free RMHD+CT excision close baryon mass to
  roundoff (`2.6e-16/6.9e-17` relative); RMHD has `divB_L1=8.45e-19`, zero
  C2P/reset/candidate correction, and `2.03 M/s`. A forced one-step density
  ceiling records 560 strictly interior bounds and no reset, proving that the
  bounded path is active rather than a no-op. These are short coarse controls,
  not long accretion or evolved-spacetime validation.
- Fixed-Kerr matter excision passes a two-level ratio-2 AMR control with
  spatial subcycling and a runtime fine-level remake from 110,592 to 32,768
  cells. After three coarse steps its exterior baryon budget closes to
  `4.86e-17` relative, the coordinate-energy/source residual is `3.14e-4`,
  and recovery, atmosphere-reset, interior-bound, and interior-reset counts
  are zero. `InitFromScratch` legitimately requests initial tags before the
  step-zero horizon exists; retain ordinary seed tags in that phase, then
  require the tracked mask before physical projection, preflight, CFL,
  evolution, and runtime regridding.
  Recompute conserved variables and stress-energy after any projection. For
  CT MHD, prefer raising the matter floor to reduce magnetization rather than
  independently editing face-centered magnetic fields. Every such operation
  is a nonconservative numerical intervention, must be counted separately from
  successful C2P fallback, and may never legalize exterior recovery failure.
- Recovery ownership now distinguishes inactive excision, bounded interior,
  and ordinary active fluid. Inactive cells must bypass both projection C2P
  and temporary primitive-cache C2P. The latter was a real missing path:
  evolving the metric with nonzero B made inactive conserved data unsuitable
  for recovery, yielding NaNs that reached neighboring stencils. Cache
  scratch must be finite without conserved-state repair, and exterior
  reconstruction/metric transfers must exclude inactive samples. Successful
  C2P fallbacks leave the evolved conservative state intact; terminal active
  atmosphere resets remain counted nonconservative failures.
- Z4c shift advection appears as `+beta^i partial_i u` on the RHS. The generic
  advection selector therefore needs transport velocity `-beta`, not `beta`.
  The old sign anti-damped grid-scale modes; the corrected order-2/4/6/8
  operators damp them for both shift signs. Earlier strong-shift/gauge drift
  comparisons need revalidation; this is not evidence that every old failure
  had the same cause.
- Evolved excision currently uses a fixed spherical cut with outgoing
  physical/gauge characteristic and AH-containment guards, generated exterior
  derivatives, masked negative-semidefinite KO, and exterior centering/nodal
  AMR transfers. Short coupled HD, restart, and AMR/subcycling controls close
  the exterior baryon ledger to roundoff without C2P/reset. A coarse longer
  control still loses its AH: long accretion and moving cuts remain open.
  Evolving geometry does not break densitized baryon conservation, but does
  invalidate a fluid-only coordinate-energy balance. A uniform physical B on
  a curved slice need not have zero divergence of densitized CT B; test its
  preservation separately from a divergence-free magnetic ID control.
  The fresh divergence-free coupled CT control has `divB_L1=2.03e-19`,
  zero recovery/reset, and baryon error below `2.4e-17` after four CFL-0.9
  steps. MP5/HLLD and a four-step coupled hybrid-EoS disk control also pass;
  the latter starts from test-fluid ID and is not a self-gravitating
  equilibrium or long-accretion validation.
- A relativistic `outflow` ghost copy is not a no-inflow condition: the
  baryon transport is `alpha*v^i-beta^i`, so zero Eulerian velocity can still
  carry atmosphere inward in a shifted chart. The completed fixed-Kerr disk
  movie has measured net boundary influx; this alone does not establish
  contamination of the disk or its modes. The corrected `diode` uses
  coordinate transport in ghosts and one common interior-derived state on
  both physical-boundary Riemann traces, including candidate correction.
  Ghost clipping alone is insufficient: a two-state HLL solve can still
  import mass through diffusion/averaging. Preserve pressure/magnetic stresses
  rather than zeroing the entire flux. The old movie selected `outflow`,
  not `diode`, so it is not retrospectively fixed by this change.
- Constant-copy nodal ghosts introduce a derivative discontinuity at an open
  Z4c boundary. Cubic extrapolation greatly reduces the measured Kerr error;
  the optional, gauge-aware Sommerfeld RHS reduces it further (about 25--31x
  versus the original small-box controls). Legacy dx-dependent exterior
  norms plateau at fixed small outer extent. Fixed coordinate-region controls
  now localize the loss of evolved convergence to boundary contamination:
  near-BH H_RMS is fourth-order through t=0.6M but loses that order by1.2M
  on R4. Moving the boundary to R6 at the same dx=0.125 and measuring exactly
  the same nodes/volumes lowers final near/far H_RMS by14x/835x. The matched
  R6 coarse/fine pair restores H_RMS order4.05 and momentum-Linf order3.97
  at1.2M, with no interventions at CFL0.9 (spin0.5 vacuum Kerr control).
  The AH area response barely changes. This is short exterior convergence,
  not long-time or massive-disk validation; R6 is not a universal safe radius.
  Do not confuse global L2 dilution with improvement. These runtime norms
  retain the established owned-node coordinate-volume quadrature, not proper
  volume or exact cut-cell integration; sampled volume varies with dx.
  Extrapolation is not radiative, and Sommerfeld is not constraint-preserving;
  a constraint-aware outer closure remains open. Applying the pure-wave
  Sommerfeld operator to exact stationary Kerr data gives a nonzero A RHS
  that persists as the derivative spacing decreases: the stationary
  background does not obey its assumed outgoing-wave falloff. Refinement
  cannot remove this continuum boundary-model error. The ETK NewRad-style
  interior-RHS correction removes the pointwise stationary plateau, but the
  matched evolved BH constraint norms still do not decrease with resolution.
  Do not infer evolved constraint preservation from a consistent point test.
  Sixth-order ghost extrapolation is needed to preserve fourth-order
  second-derivative consistency; it is separate from the physical outgoing
  boundary condition. Fixed AH angular/search tolerances also limit fine-grid
  area comparisons.
- A nodal direction with only one cell may require repeated reflection to
  map a ghost into physical nodes. A single mirror can read an opposite-side
  unfilled ghost; sequential fills can hide this race. Poisoned-ghost checks
  now cover this together with mixed reflection/extrapolation corners.
- Z4c gauge speeds are state-dependent: one-plus-log has squared Eulerian
  speed `2/alpha`, trumpet `(1-alpha)/alpha`, and the fastest live-shift mode
  `4*mu_S/(3*chi)`. Fixed sqrt(2)/2 multipliers can underestimate the strong-
  field bound (or overestimate weak-field shift speeds). CFL, outgoing BC,
  and excision now share these formulas. This repairs speed selection, not
  a proof of nonlinear stability at any specified CFL.
- Generic full-surface interpolation currently does not fold quadrature
  points across evolution reflection planes. Full-sphere radiation/scalar
  extraction can therefore reject a half/octant domain even though evolution
  is valid. The baryon-flux surface path already folds points and restores
  vector parity; general tensor/pseudotensor extraction still needs that
  shared treatment. This is a diagnostic gap, not an evolution limitation.
- The former step-three MP5/LLF filled-Kerr failure was a **frozen-gauge**
  control, not evidence that MP5 causes the failure or that all fills fail.
  PWC and halved CFL fail too. The nodal spatial metric first loses positive
  definiteness near the core; algebraic projection then produces NaNs before
  fluid recovery fails. With both one-plus-log lapse and Gamma-driver shift,
  pre-Cycle-371 matched CFL-0.9 controls reached `t=1.2 M` at `32^3` and `64^3` with no
  recovery/interior interventions; exterior H_Linf improves by 2.65x. This
  is short robustness evidence, not convergence: H_L2 slightly worsens, and
  the exterior diagnostic's stencil-exclusion layer changes with resolution.
  Live-lapse/frozen-shift and wider-fill/frozen-gauge controls expose another
  mechanism: signed high-order vertex-to-cell interpolation can make lapse
  or chi negative despite positive nodal values. Nodal floors do not ensure
  admissible transferred geometry; do not hide this with transfer floors.
- Positive Z4c scalars now use log-space interpolation for chi/lapse in
  physical nodal-to-cell and ordinary/exterior cell-to-face transfers. Raw
  polynomial physical-boundary ghosts are finite-difference extensions, not
  necessarily admissible pointwise metrics. Outside-domain centering targets
  preserve that raw extension; MPI/periodic ghosts representing physical
  points use the same log policy as valid cells. A trial log-space ghost
  extrapolation overflowed in wide AMR/excision corner storage and was removed.
  Actual point-metric consumers remain subject to signature checks. Selected
  polynomial weights/order are unchanged; smooth positive data retain the
  formal order. This is not a bound/floor limiter, source repair, or guarantee
  of tensor positive definiteness, and nonrepresentable outputs remain errors.
  Final corrected-fill controls reach t=1.2 M with SSPRK3/RK4 at CFL 0.9
  without fluid/interior interventions and with roundoff baryon closure.
  The boundary/fill/AMR/restart matrix passes14/14 and Z4c quick13/13.
  The lapse reaches its selected nodal floor.
  Kinematic consistency and robustness do not imply improved exterior
  constraint accuracy. The direct log-interpolation cost is not yet isolated
  or optimized; flat-space throughput gates do not measure stored-GR-metric
  transfer costs. Detailed results remain in Cycle 371.
- Smooth-boundary Z4c constraint/radiation conditions cannot simply be pasted
  onto Cartesian faces with a bisector normal at corners. Hilditch 1212.2901
  explicitly uses spherical outer patches to avoid corners. The Cycle-371
  face/bisector trial passed flat/manufactured checks but failed curved-BH
  outer ghost metrics. Six-point extrapolation multiplies a boundary-only
  perturbation by 56 at the third ghost node, or 56^3 at a triple corner;
  this can amplify incompatible face data, independently of MPI races.
  A radial normal on a box is an oblique approximate closure, not the paper's
  surface-normal condition. The failed trial is not a production option;
  corner-compatible constraint-preserving boundaries remain open.
- The constraint-solved black-hole owner is a Cook--Pfeiffer
  quasi-equilibrium exterior on a finite spherical shell. Spin zero reproduces
  isotropic Schwarzschild at `r=M/2`; aligned nonzero spin imposes
  `beta_hat_phi=Omega_r r_H sin(theta)` and calibrates that free boundary
  coefficient against the exact-axisymmetry throat `J_H/M_H^2`. Physical-spin
  continuation must stop when measured spin brackets the target: continuing
  all the way to a Kerr estimate of `Omega_r` can cross a nonlinear fold
  because the coefficient is not the physical spin. At `256x32`, target
  `chi_H=0.3` gives `0.29999997` with `H/L/M` residuals
  `2.19e-12/5.57e-12/1.39e-13`; the demonstrated branch reaches target
  `0.675` at `0.67499489`, while target `0.7` currently encounters a nonfinite
  nonlinear RHS and is not a supported physical-limit claim. Matched reduced
  `r theta` and periodic full-storage `r theta phi` spinning fields agree to
  roundoff. The same aligned owner supports self-gravitating disks and nested
  physical-spin/matter fixed points. On the coupled constant-ell control,
  target zero gives `chi_H=-7.50e-7` after removing the disk-induced
  `-1.31e-3` spin of a zero boundary coefficient, while target `0.3` gives
  `0.30004288` with constraint residuals at or below `1.06e-11`. It remains an
  exterior solve: Cartesian full-grid evolution still uses a finite
  constraint-violating interior fill. A `64^3` target-`0.3` artifact completes
  one CFL-0.9 evolved-Z4c step without recovery/reset, which validates routing
  and finite evolution but not exterior convergence or stability. Dynamic
  evolved-Z4c excision now has short fixed-worldtube controls, not long
  self-gravitating accretion validation; fixed-Kerr matter excision is implemented.
  `N x M x 1` is a dimensional-degeneracy test, not a separate owner.
- The tested complete-grid Kerr evolution uses ordinary advective 1+log,
  gamma-driver shift, RK4, KO strength 0.32, and positive chi/lapse floors of
  `1e-10`. The resolved `a/M=0.5` AMR/horizon control reaches `t=10.690 M` in
  64 CFL-0.9 steps with zero fluid recovery/reset. All 15 runtime regrids are
  no-ops at 98,304 cells with two unused levels. Its mean coordinate horizon
  radius grows 26.7 percent rather than shrinking, while proper area ends 4.9
  percent above exact. Zero-step checkpoint re-finds at 16/24/32 polar points
  leave the step-64 area error at 4.67--4.85 percent, so finder tolerance and
  angular discretization do not explain that drift. At matched `t=2.5 M`,
  halving the finest evolution spacing `0.5 -> 0.25 -> 0.125 M` reduces final
  proper-area error `7.65 -> 2.06 -> 0.62` percent (observed order about
  `1.8/1.7`) and area change `5.55 -> 2.08 -> 0.29` percent. Thus coordinate-
  radius growth is gauge motion and the early proper-area growth is a strongly
  convergent filled-interior evolution error; this is not yet a fine-grid
  convergence result through `t=10.7 M`. A practical `a/M=0.7` control reaches
  `t=2.476 M` with the same bounded hierarchy and final area 2.6 percent above
  exact.
  `a/M=0.8` is sensitive to the arbitrary filled interior: widening its C2
  fill transition extends survival from about `0.15 M` to `1.19 M`, but area
  falls 4 percent and exterior constraints grow. This is an interior/excision
  boundary, not a physical Kerr spin limit. The earlier amplitude-
  one/time-five Gaussian `slow_start` lapse fails by step 54. Calibrating it to
  strength `0.6` and width `20` reaches 64 steps with zero recovery, but its
  final exterior Hamiltonian/momentum norms are roughly 3.3 to 5.1 times
  larger than the matched ordinary-1+log control. It remains optional rather
  than the current recommendation.
- The analytic spinless (R_0=M) Schwarzschild trumpet is a separate 3D
  Cartesian puncture owner, not another Kerr interior policy. Its physical ADM
  metric is singular at the puncture, so evolved initialization writes the
  finite conformal Z4c state directly; the one direction-dependent
  `A_tilde_ij` puncture value is its spherical average. The puncture is
  mesh-face aligned, allowing exact reflection planes while keeping fixed-
  background cell centers off the singular point. Analytic identities,
  fourth-order stationary-RHS convergence away from the puncture, `32^3` to
  `24^3` physical-ID reload, full/octant one-step extrema, and a constant-l
  test-fluid torus have passed earlier controls. The earlier coarse-grid gauge
  ranking is now stale: Cycle 373's current-source `dx=0.5M`, CFL 0.9, KO 0.32,
  16-step rerun gives ordinary 1+log/Gamma-driver 7.89 percent final horizon-area
  error and 23.35 percent coordinate-radius change; the matched trumpet/frozen
  pair gives 7.42 percent area error and 4.99 percent radius change. Both have
  zero recovery/reset, but the unchanged 2 percent moving-puncture area gate
  FAILS. The ordinary-gauge horizon history is byte-identical under AMReX
  26.08 and 26.09, so this is an existing numerical validation gap, not an
  update regression. Do not promote a current gauge ranking from these runs.
- `apparent_horizon_expansion()` owns the Cartesian level-set evaluation of
  `Theta=D_i s^i+K_ij s^i s^j-K`; flat-sphere and exact Kerr-horizon pointwise
  checks fix its sign and geometry convention. The first full backend now uses
  solver-private BHaHAHA-style damped-wave/SSPRK3 relaxation, sixth-order pole-
  parity angular differences, one radial metric interpolation frozen across
  each pseudo-time SSPRK3 step, coarse-to-fine angular continuation, and
  hierarchy-interpolated Cartesian `gamma_ij,gamma^ij,d_k gamma_ij,K_ij`.
  Independent angular points use isolated host OpenMP with work-sized teams;
  cheap norm/diagnostic reductions remain serial and deterministic. This does
  not enable AMReX OpenMP or alter fluid evolution kernels. The local machine
  profile supplies two sustained workers. At fixed analytic Kerr data, the
  routine 16-polar-point solve takes about `0.258 s`; repeated strict 32-point
  solves take `6.81--6.82 s` versus a repeated one-worker median of `11.89 s`
  (`1.74x`). Iterations, residuals, radius error, and area error are thread
  invariant. The IO rank solves and broadcasts the star-shaped surface.
  Shape/residual/area/irreducible-mass/centroid histories persist, and
  checkpoint restart restores the latest shape. Separate
  catalog-owned AH and inner candidate masks support output and AMR tagging;
  the explicitly selected fixed-Kerr matter-excision owner may additionally
  consume the inner mask, while the mask alone never changes ownership. It is
  not an event horizon. AHFinderDirect-style Newton polishing, moving centers,
  multiple horizons, and quasilocal spin remain open.
- An event horizon is nonlocal in time. The clean future owner is offline: it
  consumes a schema-declared, time-ordered Cartesian lapse/shift/spatial-metric
  history and integrates an outgoing null surface backward from a late settled
  seed. A single radial graph is acceptable for an initial stationary-hole
  control but cannot represent general BBH topology change; a volume level set
  and derived surface/generator products are the general boundary. Sparse
  checkpoints or a live AH mask are not substitutes for this history contract.
- A scratch relativistic fluid state must be converted from physical
  primitives only after the final stored cell metric is available. Initializing
  conserved variables against a placeholder metric and then replacing nodal
  spacetime caused order-one preflight inconsistency for strong Kerr data;
  resampling the configured primitives and finalizing against the centered
  metric reduced the mismatch to roundoff. This ownership rule applies to all
  nontrivial stored-spacetime ID, not only black holes.
- `InitFromScratch` cannot tag from a physical-ID artifact before that artifact
  is loaded. Evolution starts from a saved ID by alternating target-hierarchy
  regrid and direct artifact rematerialization until the layout is a no-op.
  This prevents new fine levels from retaining prolonged coarse ID when native
  source resolution is available. The focused fixed-Kerr torus regression
  requires one changed layout, one no-op, a refined step-zero hierarchy, and
  spare level headroom.
- AMReX does not apply `BCRec` reflection in its all-node
  `GpuBndryFuncFab` path; it constant-extrapolates and invokes the user
  boundary functor. The former empty Z4c functor therefore made reflection
  runs resolution-convergent but not symmetry-equivalent. Explicit nodal
  mirror/parity closure makes matched full/equatorial Kerr and
  full/Schwarzschild-octant minimum lapse plus Hamiltonian/momentum extrema
  agree to roundoff after a CFL-0.9 step. Symmetry-aware constraint stencils
  must also regard reflected ghosts as valid; otherwise reduced domains hide
  precisely the near-plane interior where a filled hole violates constraints.

## Other Open Boundaries

- Matter vacuum is compatible with curved, evolving spacetime: zero matter
  supplies zero stress-energy, not a Minkowski metric. Write the fluid
  enthalpy density as `rho h = rho (1 + eps) + p`, not
  `rho * (1 + eps + p/rho)`: the former is algebraically identical for matter
  and makes generated P2C, flux, stress-energy, and source kernels exactly
  zero at canonical vacuum. The RHD runtime now owns one explicit exterior
  policy, default atmosphere or exact vacuum, across initial data, EoS,
  recovery, faces, AMR transfer, physical-ID materialization, output, and
  restart. Exact zero is a successful recovery state; a failed nonzero state
  remains a counted reset. RMHD vacuum remains unsupported because nonzero
  magnetic fields require an electrovacuum or force-free exterior closure.
- A low-order admissibility flux must use the same physical spectral-radius
  contract as timestep selection. Hard-coding a unit minimum signal speed can
  violate the fallback flux's own positivity CFL in subsonic flow. With local-
  CFL AMR subcycling, each accepted fine substep must also recompute the next
  allowed fine timestep as the wave speed changes; choosing every substep from
  the parent interval's initial state can violate CFL later in that interval.
- A relativistic physical-ID materializer must use the shared EoS limit when
  converting energy density to specific internal energy. Locally evaluating
  `energy_density / rho` creates a NaN in canonical vacuum that a uniform
  startup projection can hide but coarse--fine interpolation correctly
  exposes. Stored curved stellar spacetime must likewise select the stored-
  metric evolution path: a legacy target label of `minkowski` does not permit
  analytic flat kernels when a TOV or rotating-star provider supplied Z4c
  fields.
- Sharp stellar surfaces on AMR need enough physics-tag buffer that the
  coarse--fine interface remains in a smooth region. In the maintained coarse
  octant TOV gate, an unbuffered interface refluxed tiny negative conserved
  matter into uncovered coarse cells even though every fine RK stage was
  admissible; four tag-buffer cells keep the surface refined and the two-step
  CFL-0.9 gate clean. This is an AMR placement contract, not permission to
  repair nonfinite interpolation or weaken conservative reflux.
- Reference-metric spherical evolution is distinct from our spherical ID
  solver and current Cartesian Z4c evolution. SphericalNR's older filter
  restriction is not a fundamental MPI limitation; see its later parallel
  filtering reference. Axisymmetric reduction retains azimuthal tensor/fluid
  components but excludes nonaxisymmetric m=1/m=2 dynamics. Angular resolution
  reduction must not silently damp the physical modes being measured.
- SphericalNR is a published implementation reference but not presently an
  inspectable public code dependency: its RIT software page provides no source
  route, and current literature describes the framework as closed-source.
  Public NRPy does expose related reference-metric symbolic equations,
  rescaling, regularity tests, and generators. Use those sources to check
  algebra and implementation patterns while keeping PyFluid's equations and
  discretization independently owned.
- A reference-metric Valencia discretization has two equivalent continuum
  lowerings but they are not equally conservative after discretization.
  Splitting vector divergence into Cartesian coordinate-volume fluxes plus a
  cell-centered reference-connection source loses roundoff baryon
  conservation. Integrating scalar conserved fluxes with exact physical
  cell/face measures retains the generalized-Stokes conservation law; mixed
  momentum fluxes additionally need the generated covariant-basis connection
  correction. Remove the isotropic pressure tensor before that correction and
  represent its gradient with the same face/volume geometry so a constant
  pressure state is zero term-by-term. The existing Newtonian curvilinear
  source is not a Valencia reference-metric source.
- The bounded production reference-metric GRHD path is single-level and uses
  one spherical patch with active axes `r` or `r theta`. Fixed Minkowski and
  centered isotropic Schwarzschild are enabled; the Schwarzschild shell must
  lie strictly outside `r=M/2`. Initialization, C2P, reconstruction metrics,
  face fluxes, CFL, geometric sources, candidate limiting, boundaries, and
  fluid/aux diagnostics all use the same local orthonormal metric owner. The
  redundant stored Cartesian Z4c copy is not a numerical metric provider, and
  Cartesian-only Z4c constraint diagnostics are not advertised on this mesh.
  A nonlinear fixed-Minkowski `r`/`r theta` projection agrees to `4.663e-15`;
  the Schwarzschild HSE projection agrees to `1.599e-14`, and its fixed-shell
  errors converge at orders `1.997` and `1.979` from 32 through 128 radial
  cells. This does not validate AMR reflux, RMHD, Rindler/displaced/trumpet
  metrics, stored/evolved spacetime, or full spherical 3D; those combinations
  remain outside that analytic fixed-metric path. The evolved fCCZ4 slice
  and its separate evidence are described below.
- The generated fCCZ4 system now evolves regular charts through the shared
  conformal storage, with physical K=K_hat+2Theta and reference Lambda in the
  connection slots. Formulation metadata protects checkpoint compatibility.
  Cartesian gauge waves retain fourth order; the fixed-hierarchy Cartesian
  AMR/subcycling/restart control passes. The regular 3D spherical plane-wave
  gate compares all conformal fields to an independent analytic solution and
  reaches orders 3.84--3.94 on its finest pair. This is short-time interior
  propagation with periodic phi, not origin/pole or full-star validation.
  Curved RMHD/CT and mixed-chart evolution remain unsupported, and curved
  AMR has only isolated conservation controls; vacuum-interface admissibility
  and long-term stellar stability remain open. Runtime keeps the
  integrated Gamma-driver; the auxiliary-B point kernel is not an evolved GF.
- RHD flux fallback must use the same lower/upper face-area-to-volume weights
  as the high-order FV divergence. Using dt/dx on a curved mesh breaks their
  conservative equivalence when the fallback activates. Reduced diagnostic
  cell volumes and boundary areas also need the same symmetry factor.
- Curved RHD flux registers must use the same local Valencia momentum basis
  and scalar face areas as the RHS. Newtonian full-chart HD may use Cartesian
  momentum and physical face vectors; one shared runtime predicate selects
  this representation for divergence, register weighting and reflux.
  The partial spherical 3D shell control catches a mass-conservation defect
  in the former mismatched treatment. It does not validate vacuum interfaces:
  unbuffered radial AMR still creates negative internal energy after reflux.
  Moving the interface with a four-cell buffer passes a bounded control but
  is not a general admissibility-preserving synchronization method.
- Ideal RHD recovery retains dimensionless numerical intermediates in its
  generated residual and derivative. The rationalized W-1 expression avoids
  cold-state cancellation; positive-density zero-pressure matter is accepted
  by the shared relativistic physical checker. Negative internal energy is
  rejected, including the low-density controls; no transfer repair is implied.
- The TOV radial metric sampler uses quintic Hermite segments with ODE-derived
  first/second derivatives, shared by CPU and device views. Linear metric
  sampling can create growing second-derivative constraint errors despite
  accurate point values. Matter remains linearly sampled. Initial shell
  Hamiltonian errors improve, but evolved global shell constraints have not
  established convergence; do not infer full-star equilibrium from this fix.
- A Cartesian analytic fixed-spacetime provider can remain the single physics
  owner for orthogonal reference-metric Valencia. Evaluate it at the physical
  Cartesian point, then rotate the metric, inverse metric, shift, and
  extrinsic curvature into the local orthonormal reference frame. Cartesian
  partial derivatives are Cartesian covariant derivatives; rotating the lapse
  gradient, mixed shift derivative, and rank-three spatial-metric derivative
  therefore gives the reference-covariant derivatives required by the same
  generated Valencia source algebra. Do not instead differentiate the
  rotating-basis components as ordinary scalars. The determinant in the
  orthonormal frame is the reference-determinant ratio. Compiled 1D/2D/3D
  controls verify conserved-state and geometric-source covariance for
  Rindler, isotropic Schwarzschild, and the stationary trumpet. The first
  complete runtime lowering uses this representation only for centered
  isotropic Schwarzschild in the bounded spherical GRHD path above; the wider
  curved matrix remains a pointwise validation set, not runtime support.
- Reference-metric Valencia characteristic speeds are physical orthonormal
  speeds. Their CFL frequency must divide by physical directional spacing
  `h_i delta q_i`, not raw angular coordinate width. The bounded spherical RHD
  path now uses the shared mesh physical-spacing owner; the maintained
  one-step `r`/`r theta` projection remains at `4.663e-15` with theta spread
  `5.329e-15` after that correction.
- A coordinate-aware metric accessor in a relativistic face loop must retain
  an explicit Cartesian fast path. Routing Cartesian RMHD through the general
  basis transformation reduced the canonical one-dimensional benchmark from
  `2.03` to `1.31 Mcells/s`; bypassing the transformation for
  `MeshCartesian` restored `1.99 Mcells/s` without a new template axis. Keep
  this performance control whenever the reference-metric accessor changes.
- Force-free electrodynamics describes magnetically dominated plasma with
  negligible inertia, not empty electrovacuum. GRMHD/FFE matching needs a
  consistent electromagnetic update and stress-energy coupling; this can be
  achieved with coupled regional solvers and need not be a monolithic solver.
  Shared magnetic storage is one implementation choice, not a physical
  requirement. Distance or
  weak gravity alone does not justify replacing outer GRMHD by Newtonian
  matter or freezing spacetime: relativistic outflows and gravitational waves
  may still matter. These extensions are research directions, not implemented
  capabilities; sources are routed in `references.md`.
- Staged simulation handoff is distinct from live mixed-equation coupling.
  Published examples transfer settled-remnant data to fixed-metric GRMHD or
  magnetized outer ejecta to SRMHD. Neither establishes a Newtonian limit.
  A snapshot remap also does not supply continued inner-engine boundary data
  or feedback from the outer run. Coordinate/time conventions, EoS closure,
  magnetic divergence, and transferred fluxes are separate checks.
- Coordinate identities need independent geometry checks, not only mutually
  consistent formulas. A vector-gradient connection error can be purely
  antisymmetric: its trace and symmetric viscous strain remain correct while
  rigid-rotation vorticity inferred from that gradient is wrong. Likewise,
  transposed tensor-divergence connection terms are indistinguishable on
  symmetric stresses. The Cartesian-chain-rule contract now checks arbitrary
  scalar/vector/general-tensor jets in regular Cartesian/cylindrical/spherical
  charts and catches both cases. This does not establish discrete boundary,
  axis/pole, interpolation, or nonlinear-solve accuracy.
- Cartesian CTT/CTTK/XCTS now uses one physical three-component operator in
  1D, 2D, and 3D executables. Active runtime axes map to canonical physical
  `x/y/z`; omitted derivatives are zero. For the cell-centered conformal vector
  operator, AMReX's 2D `MLTensorOp` with unit shear and zero bulk viscosity
  retains the physical trace-free `+1/3 grad(div W)` coefficient, while the 1D
  active component reduces to `(4/3) d2 W = S` and inactive components to
  scalar Poisson. Three-rank nonzero manufactured CTT/XCTS matter solves
  converge at second order in both cell/vertex centering in all dimensions,
  including the physical vector and trace-free tensor, with axis-permuted
  error norms agreeing to roundoff. Vacuum CTT/CTTK/hybrid/XCTS smoke checks
  also pass. The manufactured matter is not an EoS equilibrium or an energy-
  condition test; nonzero reduced CTTK convergence and a physical cross-chart
  comparison remain unvalidated. The catalog physical Cartesian owner remains
  `x y z` and lists numerical projections separately.
- Constant-extrapolated nodal outer ghosts are not valid derivative data:
  centered differences there halve the boundary slope and prevent the
  conformal extrinsic-curvature error from converging. Cartesian ID first
  derivatives use the generated second-order one-sided closure at physical
  boundary vertices, while cell-centered face-BC ghosts and genuine
  reflection/periodic ghosts retain centered differences. Include boundary
  vertices in convergence norms; interior-only norms can hide this defect.
- The shared CTT/CTTK/XCTS elliptic solve is second-order. Raising the operator,
  coarse-fine closure, and boundary treatment together is required before
  claiming fourth-order elliptic accuracy; changing only interpolation or a
  diagnostic stencil is insufficient.
- The existing reduced spherical speed comes from separability: one cached
  angular eigendecomposition turns each scalar/toroidal Poisson solve into
  independent tridiagonal radial mode solves on a fixed spherical map. A true
  surface-fitted map `r = r(s,theta)` makes metric/Jacobian coefficients
  angle-dependent and couples those modes. Generated XCTS sources, EoS and
  Euler first integrals, continuation, diagnostics, physical-ID artifacts, and
  Cartesian materialization remain reusable, but multidomain bases/transforms,
  regularity, free-surface updates, compactification, inter-domain matching,
  coupled linear/nonlinear solves, and their MPI ownership are a new elliptic
  backend rather than a nearly free subclass.
- The separable owner's dense angular projection/reconstruction and residual
  evaluation are hot inside long matter fixed points. Cached angular
  eigenvectors include their weighted projection/reconstruction views, and CPU
  builds operate directly on resident `MultiFab` data rather than staging host
  copies. On the `368x80`, 6,459-iteration coupled BH-disk polish these
  algebra-preserving changes reduced wall time from `195.759 s` to `176.978 s`
  (9.59%); resulting geometry/fluid fields differed from the baseline by at
  most `2.84e-14/4.16e-16`.
- Matter convergence is now the undamped fixed-point defect: divide only the
  relaxed hydro update by `matter_relaxation`, then take the maximum with
  geometry, magnetic, and externally reported equilibrium defects. Thus the
  tolerance no longer weakens when a user selects stronger damping. On the
  retained `368x80` high-mass q-power BH-disk endpoint, relaxation `0.5`
  reaches a `8.41e-9` defect in 33 iterations and `2.67 s`, versus 6,459
  iterations and `176.978 s` for the old `0.0015` control. Its artifact is
  byte-identical to the same `0.5` trajectory stopped with the manually scaled
  old criterion. Relaxation `0.8` takes only 20 iterations here, but `1.0`
  excites a long alternating-support mode and takes 431; retain the runtime
  knob rather than embedding this endpoint's optimum.
- At `736x160`, the same solve takes 36 iterations and about `12.2--12.9 s` on
  three local ranks. Matched one/two/three-rank times are
  `27.81/15.45/12.2--12.9 s`, about 90-percent two-rank and 72--76-percent
  three-rank efficiency; rank artifacts agree to `2.0e-14` geometry and
  `8.8e-15` fluid absolute error. The remaining reduced-ID scaling owner is
  the dense `O(N_r N_theta^2)` forward/inverse angular transform plus two
  full-plane MPI assemblies per field. Batching the small radial-boundary
  collectives and caching its tiny setup were exact but performance-neutral
  and were removed. A material next improvement requires a different fast
  transform/distributed backend or validated threaded/accelerator execution,
  not more setup micro-optimization.
- Packed convergence norms must use AMReX's current subcommunicator, matching
  `MultiFab::norm0` rather than assuming all world ranks participate. A consumed
  linear-solve candidate can double as relaxed-update scratch, but preserve
  subtraction/offset/scaling order: conformal scalar candidates represent
  `f - 1`, whereas vector candidates represent the full field. Repopulate the
  candidate before the next solve. Fewer allocations and collectives alone do
  not establish a measurable throughput or multi-node scaling improvement.
- A KEH backend is not the next speed optimization for the present reduced
  XCTS equations. The current cached angular diagonalization plus independent
  tridiagonal radial solves is linear in radial resolution per angular mode;
  a direct KEH radial Green-function convolution is quadratic unless given a
  separate fast implementation. KEH's real advantages are natural free-space
  multipole closure and access to a full quasi-isotropic/CST rotating-star
  formulation. First compare an offline free-space multipole outer solution
  with the finite-radius Robin result; implement a second backend only if that
  exposes a material physical or work-precision limit.
- Partial-AMR CTT data are finite and convergent, but the normalized
  coarse-fine Hamiltonian band exposes a second-order multilevel-elliptic versus
  fourth-order pointwise-diagnostic mismatch.
- Cartesian multipatch evolution owns unique cells. Patch overlap is ghost and
  interpolation context, not duplicated evolved volume. Conservative overset
  reconciliation remains unimplemented.
- BlackHoles@Home's published binary-grid example is a useful future design
  reference: small stretched Cartesian near-hole patches, surrounding
  overlapping stretched spherical patches, and a shared spherical wave zone.
  Its historical site also discusses changing the arrangement for merger.
  BHaHAHA is the associated horizon finder: it samples host ADM fields onto
  its own centered spherical volume/shell and uses radial-spoke interpolation
  during surface relaxation. Keep evolution-patch exchange distinct from
  horizon-search sampling. Adopting vacuum overlapping grids would not supply
  the conservative fluid/CT seam contract required for GRMHD; reported memory
  savings are not a matched-resolution PyFluid benchmark. See references.md.
- Cylindrical and spherical single-patch kernels exist for validated method
  combinations. Mixed Cartesian-curvilinear seam evolution remains
  experimental. Do not infer production support from maps or sampling alone.
- Radiation infrastructure includes catalog-driven surface, volume, and
  multipole reductions plus Weyl/Psi4 paths. Missing quantities should extend
  those catalogs, not create separate output implementations.
- A star-shaped found horizon can use the same surface executor as a coordinate
  sphere when its sampled radius supplies the oriented radial-graph area
  vector. The Valencia baryon current is
  `F_D^i = D(alpha v^i-beta^i)` and PyFluid reports `-integral F_D^i dS_i`,
  positive inward. This is an instantaneous coordinate-surface rate; a moving
  horizon world tube needs a surface-velocity correction for an exact closed
  baryon budget.
- Baryon rest mass is not converted one-for-one into curvature. The conserved
  baryon current crosses the horizon, while the black hole's area,
  Christodoulou mass, and spin respond to matter-energy/angular-momentum plus
  gravitational and electromagnetic fluxes. Binding, thermal, kinetic, and
  magnetic energy and radiation make `Delta M_H` differ from accreted
  `Delta M_0`. For excision, require an exterior baryon balance using the inner
  and outer fluxes, pure-outflow characteristic ownership, convergent horizon
  mass/spin/area growth, and exterior constraint independence. Fixed-background
  accretion cannot close the coupled black-hole energy budget by construction;
  it is only the test-fluid limit.
- After checkpoint load or prebuilt-hierarchy synchronization, average-down can
  change a covered coarse relativistic conserved state even though every
  active fine child remains recoverable. Reapply the normal final
  relativistic acceptance after fluid and spacetime average-down and before
  CFL recovery. The collapse checkpoint at `t=75` exposed this as a covered
  level-0 C2P failure; the corrected restart reaches `t=75.6` with no runtime
  recovery fallback or atmosphere reset.
- Reusable metric/candidate scratch and longer stellar runs are open runtime
  work. Compile caching makes narrow iteration fast, but cold builds and broad
  header invalidation remain relevant; measure cold and cache-hit builds
  separately.
- Cycle 284's isolated empty-tree production build took 544.32 s with 231
  cache misses; an unchanged production rebuild took 0.66 s. Template pruning
  reduced measured object products, but a genuinely cold build remains much
  slower than the normal incremental-development path.

## Independent References And Comparisons

- AthenaK, AsterX, ETK/CarpetX, COCAL, RNS/FUKA, and public exact solvers are
  comparison and method references. They are not PyFluid production initial
  data or implementations unless a request explicitly changes that boundary.
- A code comparison starts by matching equations, EoS/units, state convention,
  domain, boundaries, resolution, reconstruction, Riemann solver, CFL,
  timestepper, output time, and AMR policy. Start uniform, compare common
  physical quantities, then add AMR.
- The matched AthenaK/AsterX evidence is case-specific. PyFluid is competitive
  on the checked Balsara controls; it has not established broad superiority.
  Preserve both revisions, inputs, builds, rank counts, outputs, conservation,
  constraints, and wall times.
- AthenaK demonstrates additive owners, uniform runtime choices, and a
  startup-built operation graph. The useful PyFluid lesson is a small immutable
  host operation plan, not a second scheduler or a per-cell virtual interface.
- AMReX and Kokkos are different layers. Replacing AMReX with Kokkos would also
  require hierarchy, FillPatch, reflux, restriction, regridding, checkpoint,
  I/O, and elliptic infrastructure. Consider Kokkos only for a measured isolated
  hotspot with a clean data boundary.

- A physical-ID AMR box with `initial_refine_fixed_levels > 0` is the complete
  startup hierarchy and suppresses automatic tags at the initial instant. For
  a black-hole disk, use `initial_refine_fixed_levels = 0`: the box guarantees
  near-hole resolution while the compact-object metric independently resolves
  disk support after materialization. The q-power torus exposed this boundary:
  a fixed horizon-only hierarchy left the torus at `dx=4`, caused 16
  density-floor projections and a `7.21e-5` baryon residual on the first coarse
  step; the combined seed/tag hierarchy had zero projections and a
  `5.94e-18` residual at the same CFL 0.9.
- Keep ordinary density-floor enforcement distinct from an atmosphere reset:
  the latter means a failed primitive recovery was replaced by atmosphere.
  The conservation ledger, not a second reset meaning, exposes material mass
  added by floor enforcement. The maintained fixed-Kerr q-power disk
  reaches `t=52.489 M` in 24 CFL-0.9 steps with a live horizon, true matter
  excision, adaptive ratio-4/2 AMR, exterior baryon residual `1.86e-13`, and
  zero C2P fallback or atmosphere reset. It is a functional
  dynamics/accretion control; evolved-Z4c interior ownership and a longer
  self-gravitating run remain open before a production claim.

## Spherical reference and centering boundary (Cycle 475--476)

- The read-only collaborator repository `yuheng-guo/Baumgarte_Spherical_Code`
  has 18 pinned branch tips in the Cycle475 review. The principal reviewed
  source is `2ee5550396be2c56fc1c54f9c2ac859edeb9f096`, rather than the much
  older default `main`. Detailed findings, branch differences, exact source
  locations and reading coverage are in `docs/baumgarte_spherical_review.md`
  and `docs/baumgarte_spherical_branches.md`. Source is guidance, not independent
  numerical validation; its code/tests were not run. `AGENTS.md` owns the
  persistent prohibition on modifying the collaborator repository.
- Its reference-rescaled spacetime point samples are half-cell shifted in r
  and theta. Tensor parity plus origin/pole coordinate identifications allow
  bulk equations to avoid singular valid nodes. This supports revisiting
  cell-centered spacetime, not removing the present vertex-fCCZ4 regular-chart
  guard. One generated equation path and a startup-resolved layout/sampling
  contract can avoid another template axis. Co-location removes some coupling
  transfers but does not equate spacetime point samples with FV fluid averages
  or eliminate face metrics and point-field AMR interpolation.
- Its `Regrid` remaps/shrinks one radial grid at fixed point count; it supplies
  no AMReX coarse-fine conservation/CT method. The latest GPU port is documented
  for an axisymmetric, equator-symmetric, magnetized-dust model. Thermal
  magnetized TOV lies on a distinct branch with unresolved historical
  stability evidence. Do not import recovery resets, surface smoothing,
  characteristic substitutions, numerical tolerances or formula-name aliases.
- Other convention traps: current HWM EM rescaling differs from stale header
  comments and standalone Maxwell; magnetic normalization includes 4*pi;
  stored K/Theta and full versus reference-subtracted conformal tensors need
  explicit conversion. Current qK waveform columns copy ordinary Psi4, whose
  projection is real, even-l, m=0 and directly time integrated. They are not
  independent tetrad or general waveform validation.
- Source review identifies an existing fluid-origin topology defect with
  active spherical theta: the fill mirrors r and optionally shifts phi, but
  misses theta -> pi-theta. A smooth equator-asymmetric scalar distinguishes
  this from existing radially symmetric origin tests. Runtime reproduction
  and repair are pending; this is separate from vertex-spacetime regularity.
- Current `EvolutionSectorStorage::average_down_level` selects nodal injection
  versus FV averaging from centering alone. A centered spacetime extension
  must first make point-sample transfer semantics explicit. No selectable
  centered-spacetime evolution capability is implemented by this review.

## Next Milestones

1. Extend the now-spinning, aligned Cook--Pfeiffer XCTS BH/BH-disk owner to a
   Kerr--Schild-seeded conformally curved solve, then add the full-3D
   nonaxisymmetric owner needed for a genuinely tilted black-hole disk. Keep
   exact exterior, filled full-grid controls, fixed-Kerr matter excision, and
   evolved-spacetime excision separate. Extend the latter's new short
   fixed-worldtube boundary/AMR controls to longer coupled accretion and
   exterior convergence; moving-mask control is not implemented. The
   stellar 3D tilt and artifact-handoff infrastructure is already in place.
2. Extend the saved fast analytic and CompOSE RNS models from the present short
   two-resolution evolutions to a meaningful fraction of a rotation, tracking
   baryon mass, angular momentum, constraints, central density, and modes.
   Extend the same work-precision history to the coupled TOV/RNS-plus-disk
   controls before calling them dynamical equilibria. Then construct constant-
   baryon-mass sequences before quoting a radial-stability/turning-point limit.
   Prototype an exterior modal domain only if it beats the 64/128-theta
   controls.
3. Extend the now stage-coupled canonical face-`B` path from the passing
   short/focused CT, Brio-Wu AMR, and SRMHD gates to long multidimensional
   GRMHD evolutions. Retain final post-sync recovery as a backstop and require
   zero atmosphere resets before promoting a long production case.
4. Run a matched PyFluid/AthenaK work-precision comparison: uniform Balsara and
   multidimensional Orszag--Tang first, with AMR only after the uniform
   equivalence statement is sound. This calibrates performance and error; it is
   not another reconstruction search.
5. Add true finite-temperature/composition-dependent CompOSE only when a
   physical evolution milestone needs it. Do not relabel the current cold
   barotrope plus gamma-law thermal extension as finite-temperature support.

## Review Discipline

- Source-check external review findings before accepting them. Raw review prose
  is provenance, not an implementation plan.
- Prefer the smallest owner-correct fix and a focused reproducer. Do not repair
  nonfinite transfer output, weaken a physical tolerance, silently change a
  default, or add a temporary fallback to conceal an intended combination.
- Treat any numerical improvement as a Pareto point until held-out HD, MHD,
  SRHD, and SRMHD tests improve without regressions. One-step or offline
  evidence is not production validation.
- Update `review_running_cycle.md` with exact work state and measurements.
  Update this file only when a durable learned fact, current evidence boundary,
  or open limitation changes.
