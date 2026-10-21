# PyFluid AMReX Reference Router

Use this file to find the authoritative source quickly after a context reset.
It intentionally links rather than restating the detailed contracts.

## Required Entry Sequence

1. Read repository instructions in `../AGENTS.md`.
2. Read the maintained state snapshot in `knowledge.md`.
3. Read at least the last active/completed cycles in
   `review_running_cycle.md`; append one cycle for the current request.
4. Read `docs/code_conventions.md` before implementation.
5. Read `docs/infrastructure_conventions.md` when changing ownership, catalogs,
   file formats, or persistent artifacts.
6. Route to the task-specific source and document below. Verify current source
   before relying on a dated result.

## Topic To Authority

| Topic | Primary implementation/source of truth | Explanatory document |
| --- | --- | --- |
| Architecture, ownership, AMR, coordinates | `src/`, `include/pyfluid_amrex/` | `docs/design.md` |
| Relativistic coarse-fine admissibility and partial sibling regions | `src/amr/state_interpolation.{hpp,cpp}`, `tests/state_interpolation_check.cpp`, `tools/validate_rmhd_balsara3_amr_prolongation.py` | `docs/design.md`, Cycles 283--284 |
| Cell-state spherical/cylindrical origin continuation | `src/numerics/origin_boundary.cpp`, `src/amr/amr_solver_fillpatch.cpp`, `tests/origin_boundary_check.cpp`, `tools/validate_origin_boundary.py` | `docs/baumgarte_spherical_review.md`, Cycle480; staggered CT/spacetime origin acceptance remains separate |
| CPU tiled numerical/ghost ownership and device scratch lifetime | `src/numerics/{rhs,patch_wall_flux}.hpp`, `include/pyfluid_amrex/flux_update.hpp`, `include/pyfluid_amrex/{boundary,relativistic_boundary,ct}.hpp`, `src/relativity/relativistic_atmosphere.hpp`, `src/amr/state_interpolation.cpp`, `tests/boundary_tiling_check.cpp`, `tools/{check_architecture,validate_boundary_tiling}.py` | `docs/infrastructure_conventions.md`, Cycles 284 and 343 |
| Coding and review policy | current source | `docs/code_conventions.md` |
| Runtime configuration | `include/pyfluid_amrex/config.hpp`, `src/framework/config*.{cpp,hpp}` | `README.md`, configuration rules |
| Grid fields, components, centerings | `schema/fields.toml`, loaded by `codegen/pyfluid_codegen/field_catalog.py` | `docs/catalog_schema.md`, generated field sections in `docs/equations.md` |
| Reconstruction, rotation-law, and AMR-policy identity | `schema/methods.toml`, loaded by `codegen/pyfluid_codegen/method_catalog.py`; formulas under `include/pyfluid_amrex/reconstruction/`, AMR evaluation in `include/pyfluid_amrex/amr_indicator.hpp`, and shared circular ID policy under `include/pyfluid_amrex/initial_data/circular_rotation.hpp` | `docs/catalog_schema.md`, `docs/design.md`, `docs/tov.md` |
| Diagnostics and reductions | `schema/diagnostics.toml`, strict loader `codegen/pyfluid_codegen/diagnostic_catalog.py`, runtime executor `src/amr/amr_solver_diagnostics.cpp`, and shared `src/numerics/diagnostic_reductions.cpp` | `docs/diagnostics.md` |
| Precision acceptance and integrated divB normalization | `tools/validation_common.py` (`precision_scalar_policy`, `cartesian_divb_scale`); `src/amr/amr_solver_diagnostics.cpp` (volume-weighted L1); `src/numerics/ct_diagnostics.hpp` (face-flux divergence) | `docs/precision_and_metal.md`; Cycle 433 |
| Generated equations | labeled symbolic sources under `codegen/pyfluid_codegen/` | `docs/equations.md`, `docs/equations.tex` |
| Shared coordinate/operator verification | `tests/test_coordinate_contracts.py`, `codegen/pyfluid_codegen/{coordinates,tensor_algebra}.py`, `tools/validate_amrex_suite.py` | `docs/infrastructure_conventions.md#shared-validation-contracts`; Cycle 386 |
| Relativistic HD/MHD and Z4c | generated kernels, `src/relativity/{rhd_system,rmhd_system,relativistic_atmosphere}.hpp`, `include/pyfluid_amrex/relativistic_boundary.hpp`, and `src/amr/amr_solver_spacetime*.cpp` | `docs/grmhd_development.md` |
| Runtime EoS and relativistic C2P | `include/pyfluid_amrex/eos/equation_of_state.hpp`, `include/pyfluid_amrex/{rhd_c2p,rmhd_c2p}.hpp`, generated `relativistic/c2p_*.hpp`, `tests/{generated_flux_compile_check,tabulated_eos_check}.cpp` | RePrimAnd, Spritz, and recovery-comparison references below; `docs/grmhd_development.md` |
| RMHD HLLC/HLLD fan and closure | `include/pyfluid_amrex/{rmhd_riemann,rmhd_advanced_flux}.hpp`, `src/relativity/rmhd_advanced_flux.cpp`, `tests/rmhd_riemann_check.cpp`, and `tools/validate_rmhd_advanced_riemann.py` | Mignone--Bodo and Mignone--Ugliano--Bodo references below; Cycles 262 and 284 |
| CT stage and AMR ownership | `src/amr/solver_sections/{ct_evolution,time_integrators,evolve_core,subcycling}.inc.hpp`, `src/numerics/ct_edge_flux_register.cpp`, and `tools/validate_ct_divb.py` | `docs/design.md`, current CT boundary in `knowledge.md` |
| Stellar surface thermal diagnostic | `tools/report_stellar_thermal.py`, `schema/fields.toml` thermal entropy proxy, `eos/equation_of_state.hpp` | `docs/grmhd_development.md`, Cycle 406; hybrid thermal excess is not physical temperature |
| Time integration and budget coefficients | `schema/time_integrators.toml`, `codegen/pyfluid_codegen/time_integrator_catalog.py`, generated time-integrator descriptors, `src/amr/solver_sections/time_integrators.inc.hpp` | `codegen/README.md`; [Butcher, Practical Runge-Kutta methods (2009)](https://doi.org/10.1017/S1446181109000030); Cycle 407 |
| Nonlinear relativistic CP Alfven wave | `tools/validate_rmhd_alfven.py`, `tools/relativistic_validation.py`; RMHD branch of `initialize_circular_alfven_x` | [ECHO section 4.1, Eq. 85](https://arxiv.org/html/0704.3206#S4.SS1); unit-amplitude 1D and axis-aligned 2D CT, Cycle 409 |
| Exact SRHD shock heating | `tools/relativistic_validation.py`, `tools/validate_rhd_shock_heating.py` | [Meliani et al. (2007), Section 3.1.2](https://arxiv.org/abs/astro-ph/0701434); finite-pressure conservation jumps, symmetric-stream variant; Cycles 405--408. Shared-face candidate closure: `src/relativity/relativistic_flux_limiter.cpp` |
| Exact SRMHD Riemann profiles | tracked `../exact_balsara/solution_b{1..5}.dat`, read by `tools/validate_rmhd.py` | Giacomazzo--Rezzolla exact solver and paper below |
| TOV/EoS/stellar initial data | `include/pyfluid_amrex/eos/`, `src/eos/`, `src/initial_data/`, `schema/methods.toml` | `docs/tov.md` |
| Magnetized stellar initial data | `include/pyfluid_amrex/initial_data/magnetic.hpp`, `src/initial_data/initial_data_magnetic.cpp`, `src/initial_data/rotating_star_initial_data.cpp`, `schema/methods.toml`, `tools/validate_magnetized_initial_data.py`, `tools/validate_magnetohydrostatic_initial_data.py` | Pili--Bucciantini--Del Zanna below; `docs/infrastructure_conventions.md`; Cycles 308--309 |
| Native cold CompOSE tables | `eos/compose/`, `src/eos/tabulated.cpp`, `include/pyfluid_amrex/eos/tabulated*.hpp`, `tests/tabulated_eos_check.cpp`, `tools/validate_compose_eos.py` | CompOSE manual and dataset records below; Cycle 297 |
| Equilibrium torus/disk initial data | `include/pyfluid_amrex/initial_data/equilibrium_torus.hpp`, `src/initial_data/equilibrium_torus_initial_data.cpp`, `schema/methods.toml`, `tools/validate_equilibrium_torus.py`, and `inputs/kerr_schild_q_power_torus_{id,evolution}_3d.in` | `docs/tov.md`, `docs/infrastructure_conventions.md`, Cycles 296 and 321 |
| Kerr black-hole initial data and interior policy | `include/pyfluid_amrex/initial_data/kerr_schild.hpp`, `src/initial_data/initial_data_provider.cpp`, `tools/validate_black_hole.py`, and `inputs/kerr_schild_{,high_spin_}horizon_amr_3d.in` | `docs/black_holes.md`, Cycles 299--301 and 321 |
| Apparent-horizon geometry, finder, tracking, and masks | `include/pyfluid_amrex/apparent_horizon.hpp`, `include/pyfluid_amrex/apparent_horizon_finder.hpp`, `src/relativity/apparent_horizon_finder.cpp`, `src/relativity/apparent_horizon_tracker.cpp`, `src/amr/amr_solver_horizon.cpp` | `docs/black_holes.md`, Cycles 300--301 |
| Portable physical-ID solve/save/materialization, including explicit radial faces | `src/initial_data/physical_id.cpp`, `src/initial_data/physical_id_interpolation.cpp`, `src/initial_data/initial_data_elliptic.cpp`, `src/framework/checkpoint.cpp`, `tests/physical_id_interpolation_check.cpp`, `tools/validate_reduced_initial_data.py` | `docs/infrastructure_conventions.md`, Cycles 287 and 291 |
| Axisymmetric spherical ID scalar/toroidal elliptic operators | `src/initial_data/initial_data_elliptic.cpp`, `include/pyfluid_amrex/initial_data/elliptic.hpp`, `tests/initial_data_elliptic_check.cpp` | `docs/infrastructure_conventions.md`, Cycles 288--289 |
| Conformal ID iteration, Cartesian operator and dimensional projections | `src/initial_data/{initial_data_elliptic,initial_data_ctt}.cpp`, `include/pyfluid_amrex/initial_data/{elliptic,ctt}.hpp`, `schema/methods.toml`, `tests/{initial_data_elliptic_check,cartesian_id_convergence}.cpp`, `tools/validate_reduced_initial_data.py` | `docs/infrastructure_conventions.md`, Cycles 384–385 and 390 |
| Infrastructure and artifact ownership | current implementation and field schema | `docs/infrastructure_conventions.md` |
| Regrid bookkeeping and aligned clocks | `src/amr/regrid_lifecycle.{hpp,cpp}`; AMReX callback integration in `src/amr/amr_solver_levels.cpp` | `docs/infrastructure_conventions.md`; Cycle 403 |
| Gravity field lifetime and Poisson setup | `include/pyfluid_amrex/gravity_fields.hpp`, `src/amr/{gravity_fields,amr_solver_gravity}.cpp` | `docs/infrastructure_conventions.md`; Cycle 404 |
| Adaptive Richardson status | AMR estimator and validation tools | `docs/adaptive_richardson.md` |
| Machines, MPI, build | `machines/*.toml`, `tools/validation_common.py` | `docs/install.md`, `docs/machines.md` |
| Compile ownership and cost evidence | `tools/build_dependencies.py`, `tools/profile_compile.py`, `tools/benchmark_compile.py`, configured `compile_commands.json` | `docs/install.md#keeping-edit-cost-local`; Clang `-ftime-trace` and [NVCC device traces](https://developer.nvidia.com/blog/optimizing-compile-times-for-cuda-c/); Cycles 380 and 452 |
| Read-only spherical collaborator source and centered-layout comparison | pinned `yuheng-guo/Baumgarte_Spherical_Code` branches, default reviewed commit `2ee5550396be2c56fc1c54f9c2ac859edeb9f096`; local read-only bare/source views under `runs/current/infrastructure_audit/cycle475/` | `docs/baumgarte_spherical_review.md`, `docs/baumgarte_spherical_branches.md`; persistent access policy in `../AGENTS.md`; Cycles 475--476 |
| CUDA device linkage and template launch ownership | `tools/manage_amrex.py`, `CMakeLists.txt` (`setup_target_for_cuda_compilation`) | [NVCC compilation modes](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda), [NVIDIA template stub linkage](https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/); `docs/machines.md`, Cycle 452 |
| CUDA runtime and compile profiling | `tools/profile_compile.py`, `machines/wsl_cuda.toml`, Cycle452 artifacts | [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/), [CUDA Compile Time Advisor](https://docs.nvidia.com/cuda/archive/13.4.1/cuda-compile-time-advisor/index.html); `docs/machines.md` records the tested WSL counter-access boundary |
| Cottonmouth generation and kernel boundaries | [EinsteinEngine Z4c recipe](https://github.com/max-morris/EinsteinEngine/blob/939a208c2e87a7d14445c0ab816d6f24637d1701/recipes/Cottonmouth/Z4c.py), [CarpetX emitter](https://github.com/max-morris/EinsteinEngine/blob/939a208c2e87a7d14445c0ab816d6f24637d1701/EinsteinEngine/generators/cpp_carpetx_generator.py) | `codegen/README.md`; Cycle 438 compares CSE, ordering, temporary reuse and explicit loop splits without importing a second physics implementation |
| Generated vector compile isolation | `codegen/pyfluid_codegen/vector_equations.py`, `include/pyfluid_amrex/kernels_generated/vector/`, `include/pyfluid_amrex/diffusion_timestep.hpp`, `tests/test_build_infrastructure.py` | `docs/infrastructure_conventions.md`; Cycle 387 |
| Fused face-solver compile/runtime tradeoff | `include/pyfluid_amrex/riemann.hpp`, `tests/test_build_infrastructure.py`, `tools/{profile_compile,benchmark_pyfluid}.py` | `docs/infrastructure_conventions.md`; Cycle 388 |
| External review triage | current source | `review_backlog.md` |
| Independent AthenaK comparison | `/Users/mrizzo/phys_research/athenak/athenak` | local `README.md`, test suite, and inputs |

Generated C++ and JSON files are derived views. Change the owning declarative
schema, strict catalog loader, or labeled equation source; regenerate and
verify consistency rather than hand-editing generated artifacts.

## Validation Ladder

Use the smallest gate that can falsify the change, then widen according to
risk. Normal commands use `UV_CACHE_DIR=.uv-cache` when the global uv cache is
not writable.

For the distinction between code verification, calculation accuracy, and
physical validation, see [NASA's verification assessment guide](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html).
It discusses exact/manufactured references, grid/time/iterative convergence,
and why agreement between two codes is insufficient by itself.

```bash
# Discover exact current names rather than guessing a stale case.
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/validate_feature_matrix.py --list-cases
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/validate_amrex_suite.py --list-steps

# Generated/source-of-truth and architecture changes.
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/validate_codegen_flux.py
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/check_architecture.py

# Targeted feature, then quick/standard integration as warranted.
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/validate_feature_matrix.py --only <case>
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/validate_amrex_suite.py --level quick
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/validate_amrex_suite.py

# Performance-sensitive production changes.
UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.matplotlib uv run python pyfluid_amrex/tools/benchmark_pyfluid.py

# Focused relativistic reconstruction/recovery, CT/AMR, and restart coverage.
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/validate_rmhd.py --level standard --ranks 3 --force
```

Normal MPI jobs use three ranks and must go through helpers in
`tools/validation_common.py`. A sandbox socket-launch failure is environmental,
not a solver result. Record the exact command, rank count, build target,
run/log path, and whether the evidence is compile-only, focused numerical,
quick, standard/full, performance, or restart/MPI-specific.

For an ordinary named input, use `tools/run_case.py`; for an already completed
run, use `tools/report_run.py`; for an ad-hoc raw run, use
`tools/validate_run_scalars.py` unless the declared policy is intentionally
nonconservative. Runtime `scalars.tsv` reductions remain the source of truth
for core totals and ledgers.

For AMR interpolation call-path comparisons, the installed AMReX 26.10 headers
are under `.deps/amrex-mpi-lsolvers-hdf5/include/`; route first to
`AMReX_FillPatchUtil{,_I}.H`, `AMReX_Interpolater.H`,
`AMReX_InterpFaceRegister.H`, and the dimension-specific interpolation
kernels. The matching source contract for face-divergence-preserving ratios 2
and 4 is in `.deps/amrex-26.10/CHANGES.md` and
`.deps/amrex-26.10/Src/AmrCore/AMReX_Interpolater.cpp`. The exact official
release pin is in `dependencies/amrex.toml`, and
`tools/manage_amrex.py` checks or updates the local versioned install. AthenaK's
corresponding componentwise prolongation is
under `/Users/mrizzo/phys_research/athenak/athenak/src/mesh/` in
`prolongation.hpp` and `mesh_refinement.cpp`. AsterX declares conserved-group
prolongation in
`/Users/mrizzo/phys_research/ETK_CarpetX/CactusX/repos/AsterX/AsterX/interface.ccl`.
These are behavior/provenance references, not sources to import.

## Durable Evidence Locations

- Active task state and exact prior results: `review_running_cycle.md`.
- Compact current rules and limitations: `knowledge.md`.
- Accepted external-review queue: `review_backlog.md`.
- Rolling validation/performance summaries: `runs/current/*history*.tsv`.
- Preserved known-good snapshots: `runs/baselines/` created by
  `tools/capture_baseline_run.py`.
- Raw Spark review provenance: `../spark_amrex_review/`. These dated notes may
  contain stale paths, already-fixed claims, or unmeasured style suggestions.

Do not use this router as evidence that a gate passed. Use the continuity cycle,
the generated TSV/log, and the current source revision/dirty state.

## Apple Silicon Platform Reference

- Apple,
  [M6 announcement](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/):
  official 12-core CPU topology of 2 super, 4 performance, and 6 efficiency
  cores. This establishes hardware identity, not PyFluid scaling efficiency.

## AMReX Framework Release

- AMReX 26.10, [official release](https://github.com/AMReX-Codes/amrex/releases/tag/26.10),
  [26.09..26.10 source delta](https://github.com/AMReX-Codes/amrex/compare/26.09...26.10):
  current framework pin in `dependencies/amrex.toml`. Local `.deps/amrex-26.10/CHANGES.md`
  routes FillPatch/component-offset and periodic flux-register fixes, GPU/SYCL
  launch/reduction fixes, HDF5 conversion, finite-aware MLMG and FP32 thresholds,
  sparse AMG and optional gridding/synchronization features. Scoped source review
  and actual build/validation status are in Cycle456 and its report.
- CMake,
  [package symlink resolution](https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS.html):
  built-in real-path resolution for imported package locations (since 3.14).
  Also available locally with `cmake --help-variable CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS`.
  Cycle 375 tests this through compiled upgrade/downgrade behavior, rather than
  assuming the cached package-hint spelling proves executable provenance.
- AMReX 26.09,
  [official release](https://github.com/AMReX-Codes/amrex/releases/tag/26.09),
  [complete 26.08..26.09 delta](https://github.com/AMReX-Codes/amrex/compare/26.08...26.09):
  prior framework pin (Cycle373). The 26.09 changelog routes
  tensor-operator state fixes, hook/MPI edge cases, scalar Kahan accumulation,
  FFT callbacks, EB/STL changes, and GPU/CMake modernization. CMake 3.25 is
  required; see `Tools/CMake/AMReXConfig.cmake.in` and `CMakeLists.txt` in that
  source checkout. The exact resolved commit is tracked in
  `dependencies/amrex.toml`.
- AMReX 26.08,
  [official release](https://github.com/AMReX-Codes/amrex/releases/tag/26.08):
  previously adopted linear-solver
  fixes, flux-register lifecycle fixes, HDF5/tooling fixes, and safer repeated
  layout/boundary definitions.
- AMReX 26.07,
  [official release](https://github.com/AMReX-Codes/amrex/releases/tag/26.07):
  PR #5252 introduced general divergence-preserving face interpolation with
  per-direction refinement ratios 2 or 4, already present in the prior 26.08
  pin. See the 26.07 section of the local `CHANGES.md` and
  `Src/AmrCore/AMReX_Interpolater.cpp`.
- AMReX 26.03--26.06 release notes:
  [26.03](https://github.com/AMReX-Codes/amrex/releases/tag/26.03) introduces
  the heterogeneous `Reducer`,
  [26.04](https://github.com/AMReX-Codes/amrex/releases/tag/26.04) adds vector
  `FillBoundary`/`FillBoundaryAndSync`, and
  [26.06](https://github.com/AMReX-Codes/amrex/releases/tag/26.06) adds runtime
  support for fixed-integer `amrex::Math::powi`. These are the upstream API
  sources for Cycle 319's adoption; the PyFluid owners and tests remain the
  evidence for local semantics and validation.
- AMReX,
  [official download/version documentation](https://amrex-codes.github.io/amrex/docs_html/GettingStarted.html):
  stable releases use monthly `YY.MM` tags. PyFluid follows stable tags, not
  the mutable development branch.
- AMReX,
  [official GPU documentation](https://amrex-codes.github.io/amrex/docs_html/GPU.html):
  the maintained upstream GPU backends are CUDA, HIP, and SYCL. Metal is not a
  native AMReX backend.
- AMReX issue
  [#5151, Apple Silicon GPU support](https://github.com/AMReX-Codes/amrex/issues/5151):
  public discussion of Metal and the AdaptiveCpp route. The issue is open but
  labeled `wontfix`, with no upstream implementation milestone.
- Lulzx,
  [WarpX Metal proof of concept](https://github.com/Lulzx/warpx-metal):
  patch-based `WarpX -> AMReX SYCL -> AdaptiveCpp SSCP -> Metal` implementation,
  validation notes, known limitations, and CPU/GPU benchmarks. The snapshot
  audited in Cycle349 pinned AMReX 26.06 and used single precision with MPI
  disabled. The README revisited in Cycle422 reports additional field-level
  PIC validation and runtime fixes; those author-reported results are routing
  evidence, not a validated PyFluid dependency or GRMHD result.
  The current `dependencies/metal.toml` retains the reference's AdaptiveCpp
  base and six patches adopted in Cycle424, plus eight local compiler/runtime
  patches. The adapted AMReX26.10 patch and redistribution license are in
  `dependencies/patches/`. Setup and evidence boundaries are
  documented in `docs/precision_and_metal.md` and the continuity cycle.
  Its pinned [benchmark results](https://github.com/Lulzx/warpx-metal/blob/7cade298257e9ffac9540549065120ab2d462c7e/benchmarks/RESULTS.md)
  report March 2026 M4 Pro PIC timings, with smaller cases slower on GPU and
  1.39x speedup for the 128-cubed case. These are historical author-reported
  results for different physics and hardware, not a current PyFluid target.
  The neighboring `scripts/09-profile-metal.sh` provides a warmed Metal System
  Trace workflow; timing and tracing runs are separate. Cycle427 source-checks
  these practices against our current implementation.
- AdaptiveCpp,
  [Metal backend documentation](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-metal.md):
  the development branch documents experimental SYCL-to-Metal support in the
  SSCP compilation flow. Device `double` is unsupported; planned software
  emulation is not native FP64 hardware. This matters when designing mixed
  precision: CPU FP64 recovery/reductions cannot simply become Metal kernels.
- Local `dependencies/patches/adaptivecpp-metal-builtin-identity.patch`:
  specializes inline builtin operation names before LLVM CFG optimization and
  carries the operation as a callee attribute. `LLVMToMetal.cpp` owns lowering;
  `Emitter.cpp` consumes the identity. Cycle432 preserves minimal distinct-name,
  duplicate-string and dynamic-name controls plus the full 2D/3D Alfvén results.
  `adaptivecpp-metal-scalarize-vectors.patch` subsequently lowers vectorized
  tag-byte fills with LLVM's ScalarizerPass before emission; the paired
  four-byte splat reproducer isolates this separate AMR compiler failure.
- Local `dependencies/patches/amrex-26.10-adaptivecpp.patch` and AdaptiveCpp's
  `sycl/libkernel/sscp/sscp_backend.hpp`: source owners for unified-pass target
  selection. AMReX's `GpuRange` must partition box-loop work by the SYCL work
  item; Cycle432's exact cell-visit reproducer exposes host range semantics
  accidentally running on every GPU thread.
- AMReX `Src/Base/AMReX_GpuReduce.H` and the same local AMReX patch:
  Cycle456 isolates a Metal zero-sum regression in the 26.10 multi-round
  subgroup fold. Ordinary block sizes use the equivalent direct subgroup
  reduction; larger counts retain the upstream multi-round path. Minimal
  native/compensated controls and maintained precision checks own acceptance;
  the exact compiler-lowering cause remains unproven.
- Apple, [indirect resource residency](https://developer.apple.com/documentation/metal/tracking-the-resource-residency-of-argument-buffers)
  and [compute useResource](https://developer.apple.com/documentation/Metal/MTLComputeCommandEncoder/useResource(_:usage:)):
  resources reached through nested pointers require explicit residency and
  hazard declarations. The pinned local indirect-residency patch consumes the
  same allocation inventory as pointer translation; Cycle428 retains the
  separated-allocation failure and repaired controls.
- Apple, [Metal math mode](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode)
  and [fast-math option](https://developer.apple.com/documentation/metal/mtlcompileoptions/fastmathenabled):
  the default fast-math behavior permits transformations inappropriate for
  assuming strict IEEE semantics. The pinned local runtime explicitly selects
  safe math and precise floating-point functions; benchmark that actual mode.
- Apple, [binary archives](https://developer.apple.com/documentation/metal/metal-binary-archives)
  and [recommended GPU working-set size](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize):
  routes for compiled-pipeline reuse and memory-capacity assessment. Cycle429
  isolates the pinned runtime's deliberate JIT postcheck sleeps independently
  of these mechanisms; it does not enable the optional binary archive.
- LLVM, [floating-point classification intrinsic](https://llvm.org/docs/LangRef.html#llvm-is-fpclass-intrinsic):
  reference for the ten IEEE class bits translated by the local Metal patch.
  Cycle425 exercises all ten classes on the M4 and inspects the compiled IR.

## Relativistic AMR Policy References

- B. Daszuta et al., *Numerical relativity simulations of compact binaries:
  comparison of cell- and vertex-centered adaptive meshes*,
  [arXiv:2406.09139](https://arxiv.org/html/2406.09139v1): direct same-code
  Z4c comparison. Section II.1 distinguishes point samples at cell centers
  from FV averages, and explains why 2:1 vertex AMR permits injection while
  centered point fields require interpolation at every coarse/fine target.
  Their Cartesian BBH timing favors vertices by about 20%; their non-magnetic
  BNS timing favors cell centers by about 17%, because fewer fluid/spacetime
  transfers offset more expensive AMR operators. These are external workload
  measurements, not a PyFluid performance prediction or spherical-origin test.
- M. Daszuta et al., *GR-Athena++: puncture evolutions on vertex-centered
  oct-tree adaptive mesh refinement*,
  [GRMHD methods and BNS application](https://arxiv.org/abs/2311.04989): the
  targeted BNS runs refine on a maximum magnetic-field threshold, while the
  shear-based policy adds a maximum-density condition specifically to avoid
  over-refining unphysical low-density regions. The thresholds are
  experimental, and the shear policy can create more blocks without matching
  all magnetic amplification of a static finest grid. This is direct evidence
  for small objective-specific GRMHD policies, not an all-GF maximum.
- T. Andrade et al., *GRChombo: numerical relativity with adaptive mesh
  refinement*, [arXiv:1503.03436](https://arxiv.org/abs/1503.03436): vacuum
  black-hole examples use conformal-factor/geometry-driven refinement. This
  supports `chi`-like vacuum tagging but does not establish `chi` alone as a
  universal GRMHD matter criterion.
- The BAM BNS implementation described in
  [arXiv:1104.4751](https://arxiv.org/abs/1104.4751): compact objects are
  followed by nested moving refinement boxes. This is the object-tracking
  control against which purely state-triggered tagging should be interpreted.
- E. Schnetter et al., *Lessons for adaptive mesh refinement in numerical
  relativity*, [DOI 10.1088/1361-6382/ac6fa9](https://doi.org/10.1088/1361-6382/ac6fa9):
  conformal-geometry derivatives and physical-resolution criteria are useful
  vacuum/geometry estimators; resource limits are not substitutes for a
  physical refinement objective.

## Rotating-Star Physics References

### CompOSE cold-EoS sources

- CompOSE, *CompOSE Reference Manual v3.00*,
  [official PDF](https://compose.obspm.fr/download/pdf/manual_v3.00.pdf): file
  schemas, units, thermodynamic quantity definitions, and the one-parameter
  cold beta-equilibrium export contract. In particular, `Q1=p/n_b` and
  `Q7=e/(n_b m_n)-1` are the two required `eos.thermo` quantities used by the
  native reader.
- RG(SLY4), [CompOSE model 134](https://compose.obspm.fr/eos/134),
  [Zenodo DOI 10.5281/zenodo.14808591](https://doi.org/10.5281/zenodo.14808591):
  bundled under `eos/compose/sly4/` with archive MD5 and provider references in
  `table.toml` and `SOURCE_README.txt`.
- APR(APR), [CompOSE model 68](https://compose.obspm.fr/eos/68),
  [Zenodo DOI 10.5281/zenodo.14508034](https://doi.org/10.5281/zenodo.14508034):
  bundled under `eos/compose/apr/`. The native causal audit retains 148 of 172
  points before applying the declared constant-sound-speed continuation.
- GPPVA(DD2) unified inner-crust/core,
  [CompOSE model 217](https://compose.obspm.fr/eos/217),
  [Zenodo DOI 10.5281/zenodo.14808942](https://doi.org/10.5281/zenodo.14808942):
  bundled under `eos/compose/dd2/`.
- The three downloaded providers declare CC-BY-4.0 redistribution. The bundled
  raw tables are external reference data, not generated PyFluid solutions;
  preserve their manifests, source READMEs, archive checksums, and DOI
  attribution when updating them.

- E. Gourgoulhon, *An introduction to the theory of rotating relativistic
  stars*, [arXiv:1003.5015](https://arxiv.org/abs/1003.5015): stationary-star
  global charges, ADM--Komar equality, GRV2/GRV3 virial identities, surface
  geometry, and mass shedding.
- K. Uryu et al., *Modeling differential rotations of compact stars in
  equilibriums*, Phys. Rev. D 96, 103011 (2017),
  [arXiv:1709.02643](https://arxiv.org/abs/1709.02643): integrable relativistic
  differential-rotation laws, including the q-power family whose q=1 member is
  the j-constant law, plus spheroidal/toroidal equilibrium sequences.
- P. Iosif and N. Stergioulas, *Equilibrium sequences of differentially
  rotating stars with post-merger-like rotational profiles*,
  [arXiv:2011.10612](https://arxiv.org/abs/2011.10612): Uryu-law sequence
  choices and comparison with the established KEH law.
- P. L. Espino et al., *Dynamical generation of stable hypermassive neutron
  stars from spinning neutron star collisions*,
  [arXiv:1906.08786](https://arxiv.org/abs/1906.08786): global matter-mode
  convention `C_m = integral sqrt(-g) rho_0 u^0 exp(i m phi) d^3x` and the
  small `m=1,2` density perturbation used by the catalog. Its separate
  fixed-density pressure depletion assumes a gamma-law evolution and is not
  the cold-consistent pressure-target inversion implemented for cold PyFluid
  equilibria.
- J. S. Read et al., *Constraints on a phenomenologically parameterized
  neutron-star equation of state*,
  [arXiv:0812.2163](https://arxiv.org/abs/0812.2163): the piecewise-polytrope
  parametrization and fitted EoS families used by the local SLy/APR4/H4/MS1
  presets.

## Rotating-Star Numerical Method References

- H. Komatsu, Y. Eriguchi, and I. Hachisu, *Rapidly rotating general
  relativistic stars -- I. Numerical method and its application to uniformly
  rotating polytropes*, MNRAS 237, 355 (1989),
  [DOI 10.1093/mnras/237.2.355](https://doi.org/10.1093/mnras/237.2.355):
  foundational KEH self-consistent-field/Green-function formulation. It is a
  reference for free-space integral closure and full quasi-isotropic rotating
  stars, not evidence that a second PyFluid backend would accelerate the
  present separable radial-mode XCTS solve.
- N. Stergioulas and J. L. Friedman, *Comparing Models of Rapidly Rotating
  Relativistic Stars Constructed by Two Numerical Methods*,
  [arXiv:astro-ph/9411032](https://arxiv.org/abs/astro-ph/9411032): direct KEH
  versus Butterworth--Ipser comparison and provenance for the public RNS code.

- S. Bonazzola, E. Gourgoulhon, and J.-A. Marck,
  *Numerical approach for high precision 3-D relativistic star models*,
  [arXiv:astro-ph/9803086](https://arxiv.org/abs/astro-ph/9803086):
  surface-fitted multidomain spectral construction; place physical
  discontinuities such as the stellar surface on domain boundaries so each
  domain contains smooth fields.
- P. Grandclement, S. Bonazzola, E. Gourgoulhon, and J.-A. Marck,
  *A multi-domain spectral method for scalar and vectorial Poisson equations
  with non-compact sources*,
  [arXiv:gr-qc/0003072](https://arxiv.org/abs/gr-qc/0003072): spherical
  multidomain Chebyshev expansions in `r` or `1/r`, spherical harmonics, and
  noncompact decaying elliptic sources. This is the main route for replacing a
  large coarsened finite-radius exterior by a controlled compactified one.
- N. Bucciantini and L. Del Zanna,
  *GRMHD in axisymmetric dynamical spacetimes: the X-ECHO code*,
  [arXiv:1010.3532](https://arxiv.org/abs/1010.3532): axisymmetric spherical
  XCFC/XNS reference using spherical-harmonic decomposition and a banded
  radial solve on the same numerical grid as the matter calculation.
- *A new code for computing differentially rotating neutron stars*,
  [arXiv:2601.05176](https://arxiv.org/abs/2601.05176): current FUKA/KADATH
  rotating-star reference spanning quasi-isotropic and XCTS formulations,
  spectral accuracy, differential rotation, and modern sequence diagnostics.
- N. Stergioulas, T. A. Apostolatos, and J. A. Font,
  *Nonlinear pulsations in differentially rotating neutron stars*,
  [arXiv:astro-ph/0312648](https://arxiv.org/abs/astro-ph/0312648): Table 1
  gives the fixed-central-density `N=1`, `K=100` uniform BU sequence and
  differential B sequence used by the standalone physical-ID literature
  check. The tabulated models use full quasi-isotropic GR, so agreement with
  conformally flat XCTS is a physical sanity check rather than a same-equation
  code comparison.
- J. A. Font et al., *Three-dimensional general relativistic hydrodynamics II:
  long-term dynamics of single relativistic stars*,
  [arXiv:gr-qc/0110047](https://arxiv.org/abs/gr-qc/0110047): unstable TOV
  migration and controlled collapse reference. Section IV.D uses a roughly
  one-percent radial density seed proportional to `cos(pi r / (2 R))`, retains
  the equilibrium pressure while recomputing specific internal energy, and
  follows apparent-horizon formation in a 3D octant. Local implementation
  cross-checks are `ETK_Carpet/Cactus/repos/einsteininitialdata/TOVSolver/src/tov.c`
  and `ETK_CarpetX/AsterX_dev/TOVSolverX/src/tov.cxx` under the shared
  `phys_research` root.
- T. Nozawa, N. Stergioulas, E. Gourgoulhon, and Y. Eriguchi,
  *Construction of highly accurate models of rotating neutron stars*,
  [arXiv:gr-qc/9804048](https://arxiv.org/abs/gr-qc/9804048): classic
  cross-code comparison of 25 polytropic and 22 realistic-EoS equilibria and
  the provenance for treating smooth rotating-star models as quantitative
  benchmark data rather than trusting convergence alone.
- XNS numerical approach,
  [project documentation](https://www.arcetri.inaf.it/science/ahead/XNS/code.html):
  hierarchically solves Poisson-like metric equations, rebuilds sources from
  the preceding iterate, and repeats the coupled field/matter loop to
  convergence. The companion
  [XNS guide](https://www.arcetri.inaf.it/science/ahead/XNS/files/Guide.pdf)
  documents separate field and Bernoulli/matter damping.
- A. G. Pili, N. Bucciantini, and L. Del Zanna,
  *Axisymmetric equilibrium models for magnetized neutron stars in General
  Relativity under the Conformally Flat Condition*,
  [arXiv:1401.4308](https://arxiv.org/abs/1401.4308): primary axisymmetric XCFC
  equilibrium and spherical-harmonic Poisson-solve reference for XNS. Equations
  20--23 give the magnetic Bernoulli integral, pure-poloidal vector
  Grad--Shafranov equation for `A_phi/(r sin(theta))`, and free magnetization
  function used by PyFluid's independent Cartesian implementation.
- Local AthenaK pressure-poloidal construction:
  `/Users/mrizzo/phys_research/athenak/athenak/src/pgen/dyn_grmhd/elliptica/elliptica.cpp`
  routes the pressure-confined vector-potential prescription, discrete curl,
  comoving-field normalization, and its AMR/divergence cautions. It is an
  independent implementation reference, not imported solution data.
- Local Einstein Toolkit construction:
  `/Users/mrizzo/phys_research/ETK_Carpet/Cactus/repos/wvuthorns_diagnostics/Seed_Magnetic_Fields_BNS/`
  contains the staggered magnetic-seed source, parameter interface,
  documentation, and regression. Use it to cross-check centering and
  normalization semantics rather than duplicating its GF declarations.
- LORENE `Star_rot::equilibrium`,
  [class reference](https://lorene.obspm.fr/Refguide/classLorene_1_1Star__rot.html):
  documents staged rotation, separate main/Poisson relaxation, enthalpy-change
  convergence, and continued angular-velocity increases toward the Keplerian
  limit.
- Public RNS reference implementation,
  [rns-alberta/rns](https://github.com/rns-alberta/rns): constructs a spherical
  model at the requested central density before spinning it to the requested
  frequency. Use only for independent comparisons and algorithm provenance,
  not imported production data.
- Local COCAL sequence guidance:
  `/Users/mrizzo/phys_research/ETK_Carpet/pcocal_illinois/Cocal/TIPS/README.txt`
  explicitly reuses each converged 3D solution as the initial guess for the
  next model. `TIPS/README_rnspar.txt` documents constant-rest-mass sequence
  controls and 1D/3D converged-data inputs. This is an independent provenance
  route for the implemented persisted sequence-continuation semantics.
- Local COCAL radial-grid construction:
  `/Users/mrizzo/phys_research/ETK_Hypatia/COCAL/COCAL_ID/src/Cocal/coordinate_grav_r.f90`
  (`grid_r`) uses a uniform inner interval followed by a geometrically growing
  exterior. It is a useful finite-volume stretching control, not an assumed
  optimum for PyFluid's axisymmetric XCTS solver.
- The companion COCAL angular construction
  `/Users/mrizzo/phys_research/ETK_Hypatia/COCAL/COCAL_ID/src/Cocal/coordinate_grav_theta.f90`
  keeps theta uniform through the full radial domain; it is not precedent for
  radius-dependent angular coarsening. Representative local RNS parameter
  files use `120x24x48` or `192x48x48` gravity grids, a surface radius normalized
  to one, a uniform-inner transition near `1.25`, and extremely distant
  geometrically stretched outer radii. COCAL's KEH equations and domains differ
  from PyFluid XCTS, so these counts are comparison guidance rather than copied
  production parameters.

## Black-Hole Initial-Data And Interior References

- J. D. Brown et al., *Turduckening black holes: an analytical and
  computational study*,
  [arXiv:0809.3533](https://arxiv.org/abs/0809.3533): constraint-propagation
  and three-dimensional numerical evidence for smoothing otherwise singular
  black-hole interiors before complete-grid BSSN-type evolution. It also
  records the important limitation that gauge modes can carry fill-dependent
  differences outward. Use it to validate a named turduckening option, not to
  relabel stuffing as exact data or excision.
- Z. B. Etienne et al., *Filling the holes: Evolving excised binary black
  hole initial data with puncture techniques*,
  [arXiv:0707.2083](https://arxiv.org/abs/0707.2083): fills the excised
  interiors of CTS data with smooth constraint-violating fields and compares
  complete-grid moving-puncture evolutions. This is direct evidence that a
  carefully validated fill can leave exterior results nearly unchanged; it is
  not a method for making the filled interior constraint satisfying.
- J. Corvino and R. M. Schoen, *On the Asymptotics for the Vacuum Einstein
  Constraint Equations*,
  [arXiv:gr-qc/0301071](https://arxiv.org/abs/gr-qc/0301071): proves a genuine
  constraint-gluing construction that can preserve interior data while making
  the far exterior exactly Kerr. Localized constraint gluing is conceptually
  distinct from smoothing followed by an ordinary global XCTS projection and
  is not currently implemented in PyFluid.
- M. Hannam et al., *Geometry and Regularity of Moving Punctures*,
  [arXiv:gr-qc/0606099](https://arxiv.org/abs/gr-qc/0606099), and
  *Wormholes and trumpets: the Schwarzschild spacetime for the moving-puncture
  generation*, [arXiv:0804.0628](https://arxiv.org/abs/0804.0628): standard
  wormhole puncture data evolve toward a cylindrical/trumpet end under the
  moving-puncture gauge, while direct stationary trumpet data avoid much of
  that initial gauge transition. A trumpet is therefore a puncture slicing/
  initial-data choice, not a third active-domain treatment alongside puncture
  and excision.
- K. A. Dennison and T. W. Baumgarte, *A Simple Family of Analytical
  Trumpet Slices of the Schwarzschild Spacetime*,
  [arXiv:1403.5484](https://arxiv.org/abs/1403.5484): the (R_0=M)
  horizon-penetrating, spatially isotropic static trumpet used by PyFluid's
  analytic spinless puncture provider. The local independently maintained
  NRPy+ formula reference is
  `/Users/mrizzo/phys_research/ETK_Hypatia/Cactus/repos/nrpytutorial/BSSN/StaticTrumpet.py`.
- I. Ruchlin, Z. B. Etienne, and T. W. Baumgarte, *SENR/NRPy+: Numerical
  Relativity in Singular Curvilinear Coordinate Systems*,
  [arXiv:1712.07658](https://arxiv.org/abs/1712.07658): equation 67 supplies
  the lapse condition analytically compatible with the static trumpet. The
  local gauge notebook at
  `/Users/mrizzo/phys_research/ETK_Hypatia/Cactus/repos/nrpytutorial/Tutorial-BSSN_time_evolution-BSSN_gauge_RHSs.ipynb`
  is the formula-routing reference; PyFluid independently generates its Z4c
  gauge kernel and validates the complete stationary RHS away from the
  puncture.
- G. B. Cook and H. P. Pfeiffer, *Excision boundary conditions for black-hole
  initial data*, [arXiv:gr-qc/0407078](https://arxiv.org/abs/gr-qc/0407078):
  quasi-equilibrium apparent-horizon inner boundary conditions for CTS/XCTS.
  This is the source for PyFluid's constraint-solved spinning horizon shell;
  its initial-data horizon boundary is not
  interchangeable with an evolution excision boundary.
- M. Caudill et al., *Circular orbits and spin in black-hole initial data*,
  [arXiv:gr-qc/0605053](https://arxiv.org/abs/gr-qc/0605053): calibrates the
  freely specified tangential horizon-shift rotation against quasi-local spin
  in quasi-equilibrium excision data. This is the interpretation route for
  PyFluid's physical-spin target rather than identifying the boundary
  coefficient with `J_H/M_H^2`.
- G. B. Cook and B. F. Whiting, *Approximate Killing Vectors on S^2*,
  [arXiv:0706.0199](https://arxiv.org/abs/0706.0199): variational elliptic
  construction of approximate axial Killing vectors on distorted horizon
  two-surfaces. This is the required next reference for replacing PyFluid's
  explicitly coordinate-rotation evolved spin with a coordinate-independent
  generic-horizon diagnostic.
- Z. B. Etienne et al., *Relativistic simulations of black-hole-neutron-star
  mergers: Effects of black-hole spin*,
  [Phys. Rev. D 76, 101503](https://doi.org/10.1103/PhysRevD.76.101503):
  smooth constraint-violating black-hole interior data evolved on a complete
  puncture grid. Use as provenance for an explicitly named filled/turduckened
  interior, never as evidence that a fill is exact or excised.
- I. Ruchlin et al., *Puncture Initial Data for Black-Hole Binaries with High
  Spins and High Boosts*,
  [arXiv:1410.8607](https://arxiv.org/abs/1410.8607): comparison route for
  high-spin puncture initial data and modified moving-puncture lapse choices.
- D. Hilditch et al., *Compact binary evolutions with the Z4c formulation*,
  [arXiv:1212.2901](https://arxiv.org/abs/1212.2901): standard Z4c,
  moving-puncture gauge, constraint-damping, and black-hole evolution control.
  Eqs. (26)--(30) give approximate Sommerfeld conditions; Eqs. (17)--(25)
  describe the distinct constraint-preserving treatment. Boundary-generated
  nonconverging constraint error is discussed explicitly. Consult the gauge
  and boundary-geometry assumptions before adapting the latter to Kerr.
  Sec. III also specifies sixth-order Lagrange ghost extrapolation before
  evaluating the bulk finite-difference and dissipation operators. Its
  "Spherical patches for the wave zone" subsection explicitly notes that
  the spherical outer boundary avoids corners; this matters when adapting
  the surface-normal CP equations to Cartesian boundaries.
- M. Alcubierre et al., *Gauge conditions for long term numerical black hole
  evolutions without excision*,
  [arXiv:gr-qc/0206072](https://arxiv.org/abs/gr-qc/0206072), Sec. VI,
  and [ETK NewRad](https://einsteintoolkit.org/thornguide/EinsteinEvolve/NewRad/documentation.html):
  radiative boundaries with a non-wave `h(t)/r^n` contribution inferred from
  the interior method-of-lines RHS; `n=3` is the published empirical choice.
  Source-check the differential signs against the outgoing-wave ansatz and
  implementation, not the inconsistent signs in some documentation equations.
  Local implementation: `/Users/mrizzo/phys_research/ETK_Hypatia/Hypatia/Cactus/repos/einsteinevolve/NewRad/src/newrad.cc`.
  This corrects non-wave boundary transients; it is not constraint-preserving.
- Z. Cao and D. Hilditch, *Numerical stability of the Z4c formulation of
  general relativity*, [arXiv:1111.2177](https://arxiv.org/abs/1111.2177):
  gauge conventions and principal physical/lapse/shift characteristic
  speeds, including the conformal-factor dependence of the shift modes.
  AthenaK's local `src/utils/finite_diff.hpp::Lx` is the source-check route
  for the sign of the one-sided stencil in RHS `+beta*d_i u` advection.
- AthenaK's local `src/bvals/physics/z4c_bcs.cpp` and
  `src/z4c/z4c_Sbc.cpp` separate polynomial ghost extrapolation from a
  Sommerfeld RHS outer boundary. They provide the polynomial ghost and
  second-order boundary-derivative reference; the Khat speed there is tied
  to AthenaK's lapse gauge, not a universal constant for PyFluid gauges.
  The hydro diode is a separate matter-flux condition. Local source root:
  `/Users/mrizzo/phys_research/athenak/athenak`.
- M. Alcubierre and B. Bruegmann, *Simple excision of a black hole in 3+1
  numerical relativity*,
  [arXiv:gr-qc/0008067](https://arxiv.org/abs/gr-qc/0008067): early long-lived
  finite-difference Schwarzschild evolution with an excised interior.
- M. D. Duez, S. L. Shapiro, and H.-J. Yo, *Relativistic Hydrodynamic
  Evolutions with Black Hole Excision*,
  [arXiv:gr-qc/0401076](https://arxiv.org/abs/gr-qc/0401076): hydrodynamic
  excision tests including relativistic Bondi accretion and collapse. This is
  the most direct validation route for PyFluid's future fixed-Kerr fluid
  excision owner.
- Z. B. Etienne et al., *IllinoisGRMHD: An Open-Source, User-Friendly GRMHD
  Code for Dynamical Spacetimes*,
  [arXiv:1501.07276](https://arxiv.org/abs/1501.07276): complete-grid
  black-hole GRMHD provenance. Section 3.4 documents conservative-state
  admissibility repair, a Lorentz-factor ceiling, a cold-pressure floor, and
  a pressure ceiling in low-density or deep-horizon cells. Use these as
  bounded-interior comparison policy, not as conservative excision.
- A. Chael, *Hybrid GRMHD and force-free simulations of black hole accretion*,
  [MNRAS 532, 3198 (2024)](https://doi.org/10.1093/mnras/stae1692): explains
  why ordinary GRMHD primitive recovery becomes ill-conditioned at high
  magnetization, records the conventional density-floor/magnetization-ceiling
  treatment, and develops a GRMHD-to-GRFFE alternative. This is a future
  high-magnetization comparison route, not justification for altering CT
  face fields during a fluid reset.
- A. Ashtekar and B. Krishnan, *Dynamical Horizons: Energy, Angular Momentum,
  Fluxes and Balance Laws*,
  [arXiv:gr-qc/0207080](https://arxiv.org/abs/gr-qc/0207080): nonlinear local
  matter and gravitational-radiation flux laws relating dynamical-horizon
  area, energy, and angular momentum changes. This is the physical reference
  for checking a dynamically coupled accreting horizon; baryon-number flux is
  a separate conservation law and is not numerically identical to horizon-mass
  growth.
- D. A. Hemberger et al., *Dynamical Excision Boundaries in Spectral
  Evolutions of Binary Black Hole Spacetimes*,
  [arXiv:1211.6079](https://arxiv.org/abs/1211.6079): demonstrates why a
  dynamical spacetime excision surface must be controlled to remain a pure-
  outflow characteristic boundary. It is implementation guidance, not
  evidence that an AMReX/Z4c active mask alone supplies that boundary.
- Local COCAL/ETK BHT fill comparison:
  `/Users/mrizzo/phys_research/ETK_Hypatia/COCAL/COCAL_ID/src/Cocal/cocal_id_bht_fill.f90`
  on the local `wip/cocal-id-bht` branch provides `smooth` and legacy `IDpar`
  completions. It extrapolates the trace-free part of `K_ij`, projects that
  part trace-free with respect to the filled metric, damps `tr(K)` inward, and
  zeroes hydro inside. This is useful independent implementation provenance,
  but the local branch is WIP and the fill neither solves the constraints nor
  derives `K_ij` from a complete stationary four-metric.
- J. Thornburg, *A Fast Apparent-Horizon Finder for 3-Dimensional Cartesian
  Grids in Numerical Relativity*,
  [arXiv:gr-qc/0306056](https://arxiv.org/abs/gr-qc/0306056): star-shaped
  six-patch angular surface, fourth-order finite differences, and direct
  Newton solution of the apparent-horizon equation. The local ETK checkout at
  `/Users/mrizzo/phys_research/ETK_CarpetX/SpacetimeX/AHFinderDirect` is an
  implementation reference only; PyFluid does not import its GPL source.
- Z. B. Etienne et al., *BHaHAHA: A Fast, Robust Apparent Horizon Finder
  Library for Numerical Relativity*,
  [arXiv:2505.15912](https://arxiv.org/abs/2505.15912): infrastructure-
  independent damped-wave relaxation, coarse-to-fine angular continuation,
  over-relaxation, and threaded surface finding. The BSD-2-Clause
  implementation under
  `/Users/mrizzo/phys_research/ETK_Hypatia/Cactus/repos/einsteinanalysis/ET_BHaHAHA`
  is the local source reference for parallel-loop ownership, pseudo-time-stage
  metric reuse, and tracking optimizations. PyFluid independently implements
  the method against its ADM geometry contract; no source is imported.
- BlackHoles@Home,
  [binary-grid technical description](https://blackholesathome.org/blackholesathome_homepage-en_US.html):
  the historical April2021 demonstration combines small SinhCartesian AMR
  patches around each hole, overlapping SinhSpherical near-zone patches, and
  one SinhSpherical wave-zone patch. The later update describes changing the
  grid arrangement for merger. This is the binary evolution-grid reference;
  BHaHAHA is the related horizon-finder library, not that grid infrastructure.
  The published memory comparison uses different resolutions and domains;
  it is not a matched PyFluid performance prediction.
- Einstein Toolkit `Outflow`,
  [thorn documentation](https://einsteintoolkit.org/thornguide/EinsteinAnalysis/Outflow/documentation.html):
  derives the Valencia rest-mass current
  `sqrt(gamma) D [alpha v^i - beta^i]`, its oriented surface integral, and the
  accumulated flux convention for coordinate spheres or apparent horizons.
  This is the sign/formulation reference for PyFluid's instantaneous found-AH
  baryon accretion rate; PyFluid's generic radial-graph quadrature and AMR
  interpolation are independent implementations.
- P. Diener, *A new general purpose event horizon finder for 3D numerical
  spacetimes*,
  [arXiv:gr-qc/0305039](https://arxiv.org/abs/gr-qc/0305039): evolves a complete
  null level-set surface backward through an already computed 3D spacetime and
  naturally admits topology change. This is the primary reference for a
  future offline PyFluid event-horizon owner; it is not an extension of the
  slice-local apparent-horizon relaxation.
- M. I. Cohen, H. P. Pfeiffer, and M. A. Scheel, *Revisiting event horizon
  finders*,
  [DOI 10.1088/0264-9381/26/3/035005](https://doi.org/10.1088/0264-9381/26/3/035005):
  comparative evidence for backward null-geodesic and null-surface methods,
  late-time seeding, generator histories, and merger tests. Use it to validate
  method choices against analytic Kerr and binary controls rather than
  assuming one representation dominates every postprocessing task.
- M. K. Bhattacharyya et al., *Implementation of the dual foliation generalized
  harmonic gauge formulation with application to spherical black hole
  excision*,
  [arXiv:2101.12094](https://arxiv.org/abs/2101.12094): comparison route for
  causally outflowing black-hole excision boundaries in a different evolution
  formulation. It does not make a Z4c active-mask boundary automatic.
- The analytic seed implemented locally is ingoing Cartesian Kerr--Schild
  data with an arbitrary Cartesian spin vector. Fixed and evolved consumers
  materialize the same ADM/Z4c fields; the flux templates do not own a second
  Kerr formula.

## Equilibrium Torus And Self-Gravitating Disk References

- S. S. Komissarov, *Magnetized Tori around Kerr Black Holes: Analytic
  Solutions with a Toroidal Magnetic Field*,
  [arXiv:astro-ph/0601678](https://arxiv.org/abs/astro-ph/0601678): stationary,
  axisymmetric, barotropic constant-specific-angular-momentum tori. PyFluid
  currently uses the unmagnetized first integral and defines
  `ell=-u_phi/u_t`; it does not import a reference solution.
- Y. Kim, J. Kim, H. I. Kim, and H. M. Lee, *General relativistic
  self-gravitating equilibrium disks around rotating neutron stars*,
  [arXiv:2406.00945](https://arxiv.org/abs/2406.00945): extended-KEH
  self-consistent-field evidence that neutron-star/disk equilibria can include
  disk rest masses of order `0.1--1 M_sun`. This is a physical and numerical
  comparison route for PyFluid's coupled XCTS approximation, not a method to
  import.
- J. Karkowski et al., *Self-gravitating axially symmetric disks in
  general-relativistic rotation*,
  [arXiv:1802.02848](https://arxiv.org/abs/1802.02848): integrable
  general-relativistic Keplerian rotation laws for self-gravitating disks.
  Route future nonconstant disk laws through the separate generated disk-law
  catalog rather than aliasing stellar rotation laws.
- M. Shibata, K. Kiuchi, S. Fujibayashi, and Y. Sekiguchi,
  *Alternative possibility of GW190521: Gravitational waves from high-mass
  black hole-disk systems*,
  [arXiv:2101.05440](https://arxiv.org/abs/2101.05440): full numerical-
  relativity evolutions of `15--50 M_sun` disks around an initially
  `50 M_sun` spinning hole, spanning `M_disk/M_BH,0 = 0.3--1`. The reported
  black-hole mass and spin are inferred from apparent-horizon area and proper
  circumferential radii under the vacuum-Kerr functional relation; for the
  paper's spinning holes `M_BH,0` is therefore not the irreducible mass alone.
  PyFluid persists both the assumption-free `M0/Mirr` and the explicitly
  Kerr-inferred horizon-mass ratio for newly solved BH/BH-disk artifacts and
  live found horizons; older area-only artifacts retain the `M0/Mirr` label.
  The reported
  one-arm instability grows in less than one orbital period once the ratio is
  about `0.6` or larger. This is the primary high-mass evolution benchmark for
  a future coupled PyFluid BH-disk sequence; a fixed-background torus at the
  same nominal mass ratio is not physically comparable.
- M. Shibata, *Rotating black hole surrounded by self-gravitating torus in the
  puncture framework*,
  [Phys. Rev. D 76, 064035 (2007)](https://doi.org/10.1103/PhysRevD.76.064035):
  primary definitions of horizon `J_H`, area, meridional/equatorial proper
  circumferences, `M_C=C_e/(4 pi)`, irreducible mass, and the Christodoulou
  `M_BH`. It compares three candidate mass measures for a nonisolated
  axisymmetric hole and is the definition source for PyFluid's native
  BH/BH-disk throat diagnostics. A generic dynamical 3D use still requires a
  valid approximate axial Killing vector.
- A. Tsokaros, K. Uryu, and S. L. Shapiro, *Complete initial value spacetimes
  containing black holes in general relativity: Application to black
  hole-disk systems*,
  [arXiv:1810.02825](https://arxiv.org/abs/1810.02825): complete
  three-dimensional self-gravitating black-hole/disk initial data. The method
  assumes approximate time stationarity and axisymmetry about the *disk* axis,
  solves the circular Euler first integral with the gravitational elliptic
  system, and explicitly states that a tilted system is not an equilibrium.
  Its sequence fixes the disk inner point and maximum density, determines the
  fluid constants during the coupled iteration, and continues from a light
  disk toward larger disk masses. COCAL uses a nonuniform-radial `660x48x48`
  spherical KEH grid and a non-conformally-flat Kerr--Schild-matched gravity
  formulation; these are comparison targets, not imported PyFluid data.
- A. Tsokaros, M. Ruiz, S. L. Shapiro, and V. Paschalidis,
  *Self-gravitating disks around rapidly spinning, tilted black holes:
  General-relativistic simulations*,
  [arXiv:2209.04454](https://arxiv.org/abs/2209.04454),
  [DOI 10.1103/PhysRevD.106.104010](https://doi.org/10.1103/PhysRevD.106.104010):
  evolution of the COCAL-style construction at `0`, `45`, `90`, and `180`
  degree black-hole tilts. The fluid is treated as approximately stationary
  and axisymmetric about the disk axis because the disk is well separated from
  the tilted hole; the models use a Gamma-`4/3`, constant-`ell` torus and grow a
  self-gravitating sequence from a massless disk. This is direct provenance
  for naming a tilted circular model a constraint-consistent non-equilibrium
  seed and then measuring its precession/warping during evolution, not for
  labeling it stationary equilibrium.

## Local AthenaK Reference

Usable local paths:

- source and inputs: `/Users/mrizzo/phys_research/athenak/athenak`;
- CPU executable: `/Users/mrizzo/phys_research/athenak/athenak/build/src/athena`;
- MPI executable:
  `/Users/mrizzo/phys_research/athenak/athenak/build_mpi/src/athena`;
- upstream identity used for the current local snapshot:
  `IAS-Astrophysics/athenak` commit
  `5f1993109bcb2e5d588ba41b4efc897408e9959a` (2026-07-08);
- license: BSD-3-Clause.

The existing CPU build is arm64, double precision, Kokkos Serial, and
MPI-disabled; `build/src/athena -h` completed successfully on 2026-07-21. The
existing MPI build is arm64, Kokkos Serial plus MPI, and links Homebrew Open
MPI. Its executable presence was checked, but no MPI launch was counted as
validation in this documentation cycle.

The local source directory has no `.git` metadata. Its `README.md`, root
`CMakeLists.txt`, Z4c RHS source, and 2D AMR Z4c linear-wave test matched the
recorded upstream commit on 2026-07-21. This is enough to identify the present
reference, but recheck representative hashes or restore a clean Git checkout
before making publication-grade provenance claims.

High-value source/test routes include:

- Newtonian HD/MHD: `src/hydro/`, `src/mhd/`, `src/reconstruct/`,
  `src/*/rsolvers/`, `inputs/hydro/`, `inputs/mhd/`, and
  `tst/test_suite/nr/`;
- SR and fixed-GR HD/MHD: `src/eos/`, relativistic Riemann solvers, and
  `tst/test_suite/sr/` plus `tst/test_suite/gr/`;
- coupled spacetime/matter: `src/dyn_grmhd/` and
  `tst/test_suite/dyngrmhd/`;
- dynamical-GRMHD robustness: `src/dyn_grmhd/dyn_grmhd_fofc.cpp` predicts the
  actual RK-stage candidate and replaces only faces adjacent to C2P/floor-
  flagged cells; `src/eos/ideal_grhyd.cpp` performs the test and records floor,
  velocity-ceiling, recovery-failure, and FOFC counters;
- explicit atmosphere/floor policy:
  `src/eos/primitive-solver/reset_floor.hpp` and
  `src/eos/ideal_c2p_hyd.hpp`;
- runtime GR task composition: `src/tasklist/numerical_relativity.hpp`,
  `src/tasklist/numerical_relativity.cpp`, `src/z4c/z4c_tasks.cpp`, and
  `src/dyn_grmhd/dyn_grmhd.cpp`;
- compile/runtime numerical dispatch: `src/hydro/hydro_fluxes.cpp`,
  `src/mhd/mhd_fluxes.cpp`, their `rsolvers/` directories, and the selected
  problem-generator logic in the root and `src/pgen/` CMake files;
- generic scalar PPM: `src/reconstruct/ppm.hpp`; `PPM4` is the original
  monotonic Colella--Woodward interpolation, while `PPMX` uses the
  Colella--Sekora smooth-extremum limiter. PyFluid implements its own compact
  uniform-grid formulas under `include/pyfluid_amrex/reconstruction/` and uses
  this file as an independent algorithm and comparison reference;
- Z4c, constraints, gauge, AMR, Weyl/wave extraction, and punctures:
  `src/z4c/`, `inputs/z4c/`, and `tst/test_suite/z4c/`;
- shared numerical references: linear waves, shock tubes, field loops,
  Orszag-Tang, diffusion, Bondi/monopole, and TOV utilities under `inputs/`,
  `tst/test_suite/`, and `src/utils/tov/`.

## Exact SRMHD Riemann Reference

- B. Giacomazzo and L. Rezzolla, *The exact solution of the Riemann problem in
  relativistic magnetohydrodynamics*, J. Fluid Mech. 562, 223 (2006),
  [paper](https://doi.org/10.1017/S0022112006001145).
- The authors' GPL-3.0
  [Exact_Riemann_Solver](https://github.com/bgiacoma/Exact_Riemann_Solver)
  emits `solution.dat` columns `x, rho, total pressure, vx, vy, vz, By, Bz`.
  For nonzero normal magnetic field its README documents nonlinear iterations
  and the need for suitable initial guesses.
- `../exact_balsara/solution_b1.dat` through `solution_b5.dat` are tracked
  outputs from that solver, introduced by commit `3aa8217`. The validation
  reader selects the final iteration block, reconstructs gas pressure from
  total pressure and comoving magnetic pressure, converts the exact primitive
  profile independently to PyFluid's flat Valencia conserved variables, and
  compares conserved cell averages at the published times. Repeated
  coordinates encode jumps; a redundant zero-strength Balsara-1 wave sample is
  removed only because its states are identical, and any nontrivial coordinate
  reversal is rejected. These files are independent validation data, not
  production flux, reconstruction, or recovery code.
  The public solver constructs a nonuniform output grid after solving the wave
  states: discontinuities are emitted at their exact coordinates and
  rarefactions receive `nx/5` samples. Its current `initialdata.f90` spells
  Balsara 2--5's nominal `5/3` adiabatic index as `1.6666`; the paper's table
  uses `5/3`. A temporary Cycle-247 regeneration verified this discrepancy is
  small but nonzero and that output-density sensitivity is larger for Balsara
  3. See Cycle 247 for measurements and failed dense-case boundaries rather
  than treating the tracked plot tables as infinite-resolution functions.

## Relativistic Primitive-Recovery References

- A. Poudel, W. Tichy, B. Bruegmann, and T. Dietrich, *Increasing the
  Accuracy of Binary Neutron Star Simulations with an improved Vacuum
  Treatment*, [arXiv:2009.06617](https://arxiv.org/abs/2009.06617):
  atmosphere-free GRHD in BAM, including TOV, migration, collapse, and BNS
  tests. Section II.C describes vacuum-aware recovery and interface states;
  it still uses numerical safeguards and resets. Its CFL 0.25 tests do not
  establish our CFL 0.9 stability or justify simply setting a density floor
  to zero.
- J. V. Kalinani et al., *Jet-environment interaction after delayed collapse
  in binary neutron star mergers*,
  [arXiv:2505.09426v3](https://arxiv.org/abs/2505.09426v3): published
  dynamical GRMHD with a very low radially decreasing density floor. The
  earlier title mentioned no artificial atmosphere limitations; this remains
  an atmosphere method, not true matter vacuum. Useful for magnetized
  atmosphere comparisons, not a blanket prescription for our floor profile.
- W. Kastaun, J. V. Kalinani, and R. Ciolfi, *Robust recovery of primitive
  variables in relativistic ideal magnetohydrodynamics*,
  [arXiv:2005.01821](https://arxiv.org/abs/2005.01821), derives the seed-free
  one-dimensional master root, admissibility bounds, and general
  thermodynamic-closure requirements used as the primary RePrimAnd reference.
  The independent local C++ implementation is routed in
  `docs/grmhd_development.md`.
- R. Cipolletta et al., *Spritz: General Relativistic Magnetohydrodynamics with
  Neutrinos*, [arXiv:2107.10620](https://arxiv.org/abs/2107.10620), documents
  a production GRMHD recovery hierarchy, tabulated-EoS integration, atmosphere
  policy, and validation corpus. Use it to compare recovery ownership and
  diagnostics, not as source to import.
- D. M. Siegel et al., *Recovery schemes for primitive variables in general-
  relativistic magnetohydrodynamics*,
  [arXiv:1712.07538](https://arxiv.org/abs/1712.07538), compares multidimensional
  and one-dimensional recovery methods over broad magnetization and EoS
  regimes. It is the independent robustness/performance comparison route for
  the maintained RePrimAnd then Palenzuela ordering.

## Curvilinear Evolution And Mixed-Regime References

- Generated spacetime geometry: `codegen/pyfluid_codegen/spacetime_geometry.py`
  owns the rescaled tensor chain rule, reference-covariant tensor derivatives,
  and connection difference; `metric_differential.py` owns shared metric
  Christoffel/Ricci algebra. SphericalNR Eqs. (14)--(21) are the continuum
  reference. `fccz4_equations.py` owns the complete reference-metric fCCZ4
  equations and physical ADM constraints from Section II.1, Eqs. (14)--(37);
  `fccz4_kernels.py` emits the compiled point kernels and
  `src/relativity/fccz4_runtime.cpp` is the AMReX runtime adapter.
  `reference_metric_evolution.py` implements the conformal metric/factor
  kinematics from Section II.1 and Eq. (33), with `chi=exp(-4 phi)`. Its basis-covariance
  contract is independent of patch selection and communication.

- P. J. Montero, T. W. Baumgarte, and E. Müller, *General relativistic
  hydrodynamics in curvilinear coordinates*,
  [arXiv:1309.7808](https://arxiv.org/abs/1309.7808): derives the
  reference-metric Valencia variables, vector and mixed-tensor covariant
  divergences, and the pressure-balance advantage over the original
  determinant-densitized spherical equations. This is the primary continuum
  GRHD reference for the generated matter geometry.
- V. Mewes et al., *Numerical relativity in spherical coordinates: A new
  dynamical spacetime and general relativistic MHD evolution framework for
  the Einstein Toolkit*,
  [arXiv:2002.06225](https://arxiv.org/abs/2002.06225): SphericalNR's
  reference-metric BSSN/fCCZ4 and Valencia GRMHD equations, rescaled tensor
  components, regularity, and coupled tests. fCCZ4 is not an alias for Z4c.
  Its orthonormal finite-volume implementation moves reference geometry into
  cell-centered sources and explicitly notes that this loses roundoff baryon
  conservation. PyFluid instead retains the equivalent generalized-Stokes
  physical-volume form for scalar conserved fluxes and uses generated basis
  connection terms for momentum.
  The [RIT SphericalNR software page](https://ccrg.rit.edu/content/software/sphericalnr)
  describes the implementation and its boundary treatment but does not expose
  a source download. Current public-code audits should therefore treat the
  SphericalNR implementation itself as unavailable, not as an Einstein Toolkit
  thorn that can be imported or inspected.
- SphericalNR origin/layout implementation: Mewes et al. (2018),
  [arXiv:1802.09625](https://arxiv.org/pdf/1802.09625), Section II.B,
  Eqs. (15)--(20): spacetime sampling is half-cell shifted in r and theta,
  vertex sampled in periodic phi. The 2020 paper above supplies the corrected
  south-pole mapping and rescaled tensor parities in Section III/Table I;
  its magnetic vector potential and derived magnetic field are cell-centered.
  This is not PyFluid's current vertex-spacetime/face-CT arrangement.
- Olivares et al. (2019), [arXiv:1906.10795](https://arxiv.org/pdf/1906.10795),
  Sections 3.1 and 4.5/Fig. 6: BHAC face-flux/edge-circulation CT, polar
  connectivity including refinement interfaces, and vanishing contributions
  on collapsed faces/edges. These establish staggered polar-axis techniques,
  not a full dynamical-spacetime spherical-origin implementation.
- [NRPy public source](https://github.com/nrpy/nrpy): inspectable SymPy-based
  reference-metric equation and code-generation infrastructure descended from
  the generator family used for SphericalNR's spacetime kernels. Relevant
  routes are `nrpy/equations`, `nrpy/infrastructures`, and `nrpy/tests`; use it
  for independent algebra/regularization comparisons, not as a substitute for
  the unavailable SphericalNR GRHydro modifications.
- L. Ji et al., *Ameliorating the Courant-Friedrichs-Lewy condition in
  spherical coordinates: A double FFT filter method for general relativistic
  MHD in dynamical spacetimes*,
  [arXiv:2305.01537](https://arxiv.org/abs/2305.01537): replaces the older
  filter's phi-decomposition restriction. Section II.2 describes ring-local
  communicator redistribution and scaling through 4096 cores. Filtering has
  conservation and mode-damping tradeoffs, not just execution cost.
- V. Paschalidis and S. L. Shapiro, *A new scheme for matching general
  relativistic ideal magnetohydrodynamics to its force-free limit*,
  [arXiv:1310.3274](https://arxiv.org/abs/1310.3274): matched ideal-MHD
  interior and force-free exterior, including a rotating-star magnetosphere
  application. For evolving magnetization-selected transitions in accretion,
  see the Chael reference under black-hole interior references above.
- H. Cho et al., *Multi-Zone Modeling of Black Hole Accretion and Feedback
  in 3D GRMHD: Bridging Vast Spatial and Temporal Scales*,
  [arXiv:2405.13887](https://arxiv.org/abs/2405.13887): scale-separated
  multi-zone GRMHD toward dynamical steady state. This retains GRMHD; it does
  not validate a dynamical-Einstein interior matched to Newtonian or frozen
  outer evolution.
- F. G. Lopez Armengol et al., *Handing off the outcome of binary neutron
  star mergers for accurate and long-term post-merger simulations*,
  [arXiv:2112.09817](https://arxiv.org/abs/2112.09817): HandOff transfers
  dynamical-spacetime IllinoisGRMHD data to spherical-like HARM3D grids with
  a fixed numerical metric after the remnant becomes nearly stationary.
  Both stages retain GRMHD; the paper validates remapping and continued disk
  evolution, not a Newtonian limit or arbitrary early metric freezing.
- A. Pavan et al., *Jet-environment interplay in magnetized binary neutron
  star mergers*, [arXiv:2211.10135](https://arxiv.org/abs/2211.10135):
  imports magnetized GRMHD merger environments into PLUTO's special-
  relativistic MHD solver outside an excised central region, with boundary
  data informed by the merger and a separately prescribed jet. Section 2
  and appendices discuss mapping, magnetic divergence, and EoS mismatch.
  This is a GR-to-SR exterior handoff reference, not full engine transfer
  or nonrelativistic MHD evidence.

## Modern Shock-Capturing Research Routes

Use these as method and comparison references, not imported production
implementations:

- A. Suresh and H. T. Huynh, *Accurate Monotonicity-Preserving Schemes with
  Runge--Kutta Time Stepping*,
  [NASA record and paper](https://ntrs.nasa.gov/citations/19970010128):
  original MP5 construction and its high-order monotonicity-preserving bounds.
- P. Colella and P. R. Woodward, *The Piecewise Parabolic Method (PPM) for Gas-
  Dynamical Simulations*,
  [paper](https://doi.org/10.1016/0021-9991(84)90143-8): original monotonic PPM,
  flattening, and density contact-steepening construction. The local AthenaK
  and AsterX routes above are implementation comparisons, not imported source.
- P. Colella and M. D. Sekora, *A limiter for PPM that preserves accuracy at
  smooth extrema*,
  [paper](https://crd.lbl.gov/assets/pubs_presos/AMCS/ANAG/ColellaSekora.pdf);
  R. Borges et al., *An improved weighted essentially non-oscillatory scheme
  for hyperbolic conservation laws*,
  [WENO-Z](https://doi.org/10.1016/j.jcp.2007.11.038); L. Fu et al.,
  *A family of high-order targeted ENO schemes for compressible-fluid
  simulations*,
  [TENO](https://doi.org/10.1016/j.jcp.2015.10.037); and J. Zhu and C.-W. Shu,
  *A new type of multi-resolution WENO schemes*,
  [paper](https://doi.org/10.1016/j.jcp.2018.09.003): primary routes for
  smooth-extremum, nonlinear-weight, stencil-rejection, and hierarchical
  polynomial mechanisms.
- U. S. Fjordholm, S. Mishra, and E. Tadmor, *Arbitrarily high-order accurate
  entropy stable essentially nonoscillatory schemes for systems of
  conservation laws*,
  [TeCNO](https://doi.org/10.1137/110836961), and U. S. Fjordholm and D. Ray,
  *A sign preserving WENO reconstruction method*,
  [SP-WENO](https://arxiv.org/abs/1510.09038): the precise role of a
  reconstructed entropy-variable jump with the sign property. The sign
  property is not a generic guarantee for component-reconstructed Godunov
  fluxes.
- J.-L. Guermond et al., *Second-order invariant domain preserving
  approximation of the Euler equations using convex limiting*,
  [paper](https://arxiv.org/abs/1710.00417): invariant-domain constraints and
  convex correction from a robust low-order update. It motivates admissibility
  providers but does not by itself specify a sharp complete-wave flux.
- E. Tadmor, *The numerical viscosity of entropy stable schemes for systems
  of conservation laws*,
  [paper](https://doi.org/10.1090/S0025-5718-1987-0890255-3), and R. Abgrall,
  P. Öffner, and H. Ranocha, *Reinterpretation and Extension of Entropy
  Correction Terms...*, [arXiv:1908.04556](https://arxiv.org/abs/1908.04556):
  primary routes for entropy-residual viscosity and minimum-norm constrained
  corrections. They establish the correction principle; they do not establish
  the present audited-HLLD composition.
- E. F. Toro, M. Spruce, and W. Speares, *Restoration of the contact surface
  in the HLL-Riemann solver*,
  [HLLC](https://doi.org/10.1016/0021-9991(94)90166-X), and T. Miyoshi and
  K. Kusano, *A multi-state HLL approximate Riemann solver for ideal
  magnetohydrodynamics*,
  [HLLD](https://doi.org/10.1016/j.jcp.2005.02.017): baseline contact and
  contact/Alfven wave-fan constructions.
- A. Mignone and G. Bodo, *An HLLC Riemann solver for relativistic flows. I.
  Hydrodynamics*, [arXiv:astro-ph/0506414](https://arxiv.org/abs/astro-ph/0506414),
  and *II. Magnetohydrodynamics*,
  [arXiv:astro-ph/0601640](https://arxiv.org/abs/astro-ph/0601640):
  relativistic HLLC fan derivations and degeneracy boundaries.
- G. Mattia and A. Mignone, *A comparison of approximate non-linear Riemann
  solvers for Relativistic MHD*,
  [arXiv:2111.09369](https://arxiv.org/abs/2111.09369): matched HLL, HLLC,
  HLLD, HLLEM, and GFORCE accuracy/robustness evidence. HLLD is strongest for
  weak/moderate magnetization; GFORCE is a robust less-diffusive HLL
  alternative at strong magnetization.
- A. Mignone, M. Ugliano, and G. Bodo, *A five-wave HLL Riemann solver for
  relativistic MHD*, [arXiv:0811.1483](https://arxiv.org/abs/0811.1483):
  the nonlinear total-pressure closure and jump conditions for RMHD HLLD.
- L. Antón et al., *Numerical 3+1 general relativistic magnetohydrodynamics: a
  local characteristic approach*,
  [arXiv:astro-ph/0506063](https://arxiv.org/abs/astro-ph/0506063): the
  equivalence-principle construction that applies an SRMHD Riemann solver in a
  local orthonormal frame for 3+1 GRMHD.
- C. J. White, J. M. Stone, and C. F. Gammie, *An Extension of the Athena++
  Code Framework for GRMHD Based on Advanced Riemann Solvers and
  Staggered-Mesh Constrained Transport*,
  [arXiv:1511.00943](https://arxiv.org/abs/1511.00943): an independent GRMHD
  HLLC/HLLD plus staggered-CT implementation and validation route.
- K. Kiuchi et al., *Implementation of advanced Riemann solvers in a
  neutrino-radiation magnetohydrodynamics code in numerical relativity and its
  application to a binary neutron star merger*,
  [arXiv:2205.04487](https://arxiv.org/abs/2205.04487): numerical-relativity
  HLLC/HLLD and advanced-CT tests in Minkowski and black-hole spacetimes.
- K. Derigs et al., *Ideal GLM-MHD: About the entropy consistent nine-wave
  magnetic field divergence diminishing ideal magnetohydrodynamics
  equations*, [entropy-stable MHD route](https://arxiv.org/abs/1509.09311),
  and B. Schmidtmann and A. R. Winters, *Hybrid Entropy Stable HLL-Type
  Riemann Solvers for Hyperbolic Conservation Laws*,
  [paper](https://arxiv.org/abs/1607.06240). Verify the equation/source
  formulation before transferring an entropy proof to CT MHD or RMHD.
- S. Simon and J. C. Mandal,
  [HLLC-ADC](https://arxiv.org/abs/1803.04954) and
  [HLLC-SWM](https://arxiv.org/abs/1803.04922): pressure/shock-sensitive
  control of contact antidiffusion for multidimensional shock stability.
  These are the closest collision with a pressure/compression entropy budget.
- A. I. Sheikh et al., *The CRONOS code for astrophysical magnetohydrodynamics*,
  [DOI 10.3847/1538-4365/aabe75](https://doi.org/10.3847/1538-4365/aabe75):
  a strong-shock detector locally selects HLL to suppress carbuncle modes.
  Together with HLLC-ADC/SWM, this supports PyFluid's multidimensional
  pressure-jump HLL parent switch; it does not determine the present threshold.
- A. Mignone et al., *A robust HLLC-type Riemann solver for relativistic flows*,
  [MNRAS 496, 2493](https://academic.oup.com/mnras/article/496/2/2493/5857183):
  relativistic HLL/HLLC blending provides an independent relativistic example
  of retaining contact resolution away from shock-instability regions.
- X. Zhang et al., *A Provably Positive-Preserving HLLD Riemann Solver for
  Ideal Magnetohydrodynamics. Part I: The One-Dimensional Case*,
  [paper](https://doi.org/10.4208/cicp.OA-2024-0068), and F. Zhang,
  A. Lani, and S. Poedts, *On physical consistency of HLL-type Riemann solvers
  for ideal MHD*, [preprint](https://arxiv.org/abs/2507.10420) and
  [published energy-consistency version](https://doi.org/10.1016/j.jcp.2026.114724):
  source routes for ordered positive fan states and magnetic-energy
  consistency, not production code to copy.
- K. Takahashi and S. Yamada, *Exact Riemann solver for ideal
  magnetohydrodynamics that can handle all types of intermediate shocks and
  switch-on/off waves*,
  [arXiv:1310.2330](https://arxiv.org/abs/1310.2330), and *Regular and
  non-regular solutions of the Riemann problem in ideal
  magnetohydrodynamics*,
  [arXiv:1210.5584](https://arxiv.org/abs/1210.5584): primary wave-curve,
  degeneracy, and nonuniqueness sources. Rankine--Hugoniot plus the ordinary
  entropy condition does not define one universal ideal-MHD fan; the
  Brio--Wu data admit one regular solution and uncountably many nonregular
  solutions.
- M. Torrilhon, *Uniqueness conditions for Riemann problems of ideal
  magnetohydrodynamics*,
  [paper](https://doi.org/10.1017/S0022377803002186), and *Non-uniform
  convergence of finite volume schemes for Riemann problems of ideal
  magnetohydrodynamics*,
  [paper](https://doi.org/10.1016/S0021-9991(03)00347-4): primary sources for
  the non-strict/non-convex ideal-MHD fan, large-angle tangential-field test,
  and pseudo-convergence boundary. A finite-resolution numerical fan that
  looks stable is not necessarily the regular solution.
- K. Xu et al., *Numerical path preserving Roe scheme for ideal MHD Riemann
  problem: Complete elimination of pseudo-convergence*,
  [paper](https://doi.org/10.1016/j.jcp.2024.113621): identifies straight-line
  averaging of Alfvén rotations in the transverse-magnetic plane as the
  corruption mechanism and constructs a Roe correction whose numerical path
  follows the constant-`|Bt|` arc. This is direct support for retaining an
  accepted rotational wave as latent physical structure; it is not a source
  implementation to import and does not by itself solve general fan birth,
  interactions, CT, or AMR.
- F. Zhang, *An improved estimate of the intermediate internal energy in the
  energy-consistent HLLD scheme*,
  [arXiv:2607.10293](https://arxiv.org/abs/2607.10293): July-2026 evidence that
  admissible-looking standard HLLD algebra can still need internally
  consistent intermediate thermodynamics under strong magnetic fields.
- O. Zanotti, F. Fambri, and M. Dumbser, *Solving the relativistic
  magnetohydrodynamics equations with ADER discontinuous Galerkin methods,
  a posteriori subcell limiting and adaptive mesh refinement*,
  [arXiv:1504.07458](https://arxiv.org/abs/1504.07458): RMHD evidence for
  accepting a high-order candidate in smooth cells and replacing troubled
  cells a posteriori. The DG/subcell implementation is not a direct fit for
  the finite-volume AMReX path, but the MOOD-style policy is relevant.
- M. Dumbser, O. Zanotti, R. Loubere, and S. Diot, *A posteriori subcell
  limiting of the discontinuous Galerkin finite element method for hyperbolic
  conservation laws*,
  [arXiv:1406.7416](https://arxiv.org/abs/1406.7416): original local
  space-time ADER plus MOOD/subcell precedent. It establishes that checking a
  completed high-order candidate and recomputing troubled regions is an
  existing method family; it does not validate PyFluid's particular finite-
  volume DMP or MP5/PWL/PWC ladder.
- N. Deppe et al., *A positivity-preserving adaptive-order finite-difference
  scheme for GRMHD*,
  [arXiv:2306.04755](https://arxiv.org/abs/2306.04755), with the source-checked
  [SpECTRE PPAO implementation](https://github.com/sxs-collaboration/spectre/blob/develop/src/NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp):
  primary reference for using the relative highest-polynomial-mode power to
  reduce order a priori. Its discrete order cascade and finite-difference
  flux differencing are useful controls, not a validated replacement for
  PyFluid's finite-volume candidate guard.
- R. Bourgeois and D. Lee, *GP-MOOD: A positive-preserving high-order finite
  volume method for hyperbolic conservation laws*,
  [arXiv:2110.08683](https://arxiv.org/abs/2110.08683): primary finite-volume
  precedent for accepting a high-order candidate and cascading local order
  a posteriori. It supports the guard architecture, but not any particular
  PyFluid DMP variable, tolerance, or RMHD/CT correction policy.
- V. Berta, A. Mignone, M. Bugli, and G. Mattia, *A 4th-order accurate finite
  volume method for ideal classical and special relativistic MHD based on
  pointwise reconstructions*,
  [arXiv:2310.11831](https://arxiv.org/abs/2310.11831): primary source for the
  five-point discontinuity detector and local order-reduction strategy audited
  in Cycle 264. Detection of a nonsmooth stencil does not by itself distinguish
  a physical compound fan from a numerical oscillation.
- L. Del Zanna, O. Zanotti, N. Bucciantini, and P. Londrillo, *ECHO: an
  Eulerian Conservative High Order scheme for general relativistic
  magnetohydrodynamics and magnetodynamics*,
  [arXiv:0704.3206](https://arxiv.org/abs/0704.3206): primary GRMHD precedent
  for high-order reconstruction with UCT in a generic spacetime. ECHO is a
  conservative finite-difference formulation, so it establishes feasibility
  but is not a drop-in finite-volume order switch for PyFluid.
- K. Wu and H. Tang, *On physical-constraints-preserving schemes for special
  relativistic magnetohydrodynamics with a general equation of state*,
  [arXiv:1709.05838](https://arxiv.org/abs/1709.05838): rigorous
  finite-volume/DG admissibility results for Lax--Friedrichs-type fluxes and
  the required discrete divergence-free condition. PCP prevents unphysical
  states; it is not a nonoscillatory-profile theorem.
- J. Duan and H. Tang, *High-order accurate entropy stable nodal
  discontinuous Galerkin schemes for the ideal special relativistic
  magnetohydrodynamics*,
  [arXiv:1911.03825](https://arxiv.org/abs/1911.03825): an RMHD entropy-stable
  construction for a modified Powell-like system. Its entropy inequality does
  not transfer unchanged to the present conservative staggered-CT update.
- C. Pan et al., *A high-fidelity finite volume scheme for ideal
  magnetohydrodynamics equations using boundary variation diminishing
  algorithm*, [arXiv:2404.13015](https://arxiv.org/abs/2404.13015): recent
  nonrelativistic MHD plus CT evidence for polynomial/THINC BVD selection.
  Treat it as a prototype direction requiring independent SRMHD validation.
- Z. Sun, S. Inaba, and F. Xiao, *Boundary Variation Diminishing
  reconstruction: a new approach to improve Godunov scheme*,
  [arXiv:1602.00814](https://arxiv.org/abs/1602.00814), and X. Deng et al.,
  *A fifth-order shock capturing scheme with two-stage boundary variation
  diminishing algorithm*,
  [arXiv:1811.01379](https://arxiv.org/abs/1811.01379): original in-cell
  jump-like BVD motivation and the P4 plus two-THINC cascade. These are the
  primary definitions behind the local MP5-THINC research precursor.
- E. Pimentel-García et al., *In-cell Discontinuous Reconstruction
  path-conservative methods...*,
  [arXiv:2105.00424](https://arxiv.org/abs/2105.00424), and E. Abreu et al.,
  *High-order DR.MOOD...*,
  [arXiv:2407.02931](https://arxiv.org/abs/2407.02931): exact isolated-shock
  capture by explicitly representing a discontinuity inside a cell. Their
  current equation sets are not a drop-in conservative MHD or SRMHD method.
- A. Harten, *ENO schemes with subcell resolution*,
  [paper](https://doi.org/10.1016/0021-9991(89)90226-X): the original
  polynomial-plus-in-cell-jump construction used to audit whether a
  discontinuity can be recovered from neighboring finite-volume averages.
  The construction is a representation tool, not by itself an MHD wave
  admissibility or causal-front creation rule.
- N. Aguillon, *A Non-dissipative Reconstruction Scheme for the Compressible
  Euler Equations*,
  [arXiv:1403.7497](https://arxiv.org/abs/1403.7497): conservative in-cell
  shock reconstruction and crossing-time flux integration that is exact for
  pure shocks. Its full-Euler results are prospective, contact detection is
  explicitly imperfect, and rarefactions use a separate background scheme;
  it is the closest source precedent for the causal-front prototype, not a
  complete MHD/relativistic implementation.
- G. I. Montecinos, E. F. Toro, and L. O. Müller, *A generalized Riemann
  problem-based compact reconstruction method for finite volume schemes*,
  [arXiv:2602.21911](https://arxiv.org/abs/2602.21911): a compact
  space-time route that carries interface GRP data between steps and builds a
  high-order polynomial from current cell averages plus that history. The
  February-2026 paper reports one-dimensional advection/Euler tests, not MHD
  or relativistic evidence, but it is directly relevant to the missing
  time-integrated-background representation in the physical-front prototype.
- H. Zhang et al., *An Adaptive Reconstruction Method for Arbitrary
  High-Order Accuracy Using Discontinuity Feedback*,
  [arXiv:2410.05489](https://arxiv.org/abs/2410.05489): a recent
  low-to-high-order alternative in which a discontinuity-feedback factor is
  both detector and reconstruction input. It supports continuous,
  scale-independent adaptation but does not establish the physical-wave
  purity construction tested here.
- Y. Chen and K. Wu, *A Physical-Constraint-Preserving Finite Volume WENO
  Method for Special Relativistic Hydrodynamics on Unstructured Meshes*,
  [arXiv:2207.09385](https://arxiv.org/abs/2207.09385), and M. Dumbser et al.,
  *ADER-WENO finite volume schemes with space-time adaptive mesh refinement*,
  [SRHD/SRMHD formulation](https://arxiv.org/abs/1312.7784): primary routes
  for relativistic admissibility, scaling-invariant nonlinear weights, and
  high-order finite-volume extension. A Newtonian positivity check does not
  substitute for density/pressure/subluminal relativistic PCP constraints.
- D. S. Balsara et al., *Efficient, high accuracy ADER-WENO schemes for
  hydrodynamics and divergence-free magnetohydrodynamics*,
  [arXiv:0811.2200](https://arxiv.org/abs/0811.2200): primary
  space-time-predictor and divergence-free-MHD precedent. It motivates
  integrating a high-order background flux in time instead of treating a
  spatial reconstruction as the complete discontinuity update.
- D. S. Balsara et al., *Efficient Implementation of ADER Schemes for Euler
  and Magnetohydrodynamical Flows on Structured Meshes -- Comparison with
  Runge-Kutta Methods*,
  [arXiv:1006.2146](https://arxiv.org/abs/1006.2146): optimized structured-
  mesh Euler/MHD evidence that ADER can be competitive with, and in that
  implementation almost twice as fast as, comparable SSPRK schemes. It is a
  performance precedent only; the current Python research wall times cannot
  predict a generated AMReX implementation.
- I. Cravero, M. Semplice, and G. Visconti, *Optimal definition of the
  nonlinear weights in multidimensional Central WENOZ reconstructions*,
  [arXiv:1811.08688](https://arxiv.org/abs/1811.08688),
  [paper](https://doi.org/10.1137/18M1228232): primary CWENO-Z construction
  and global-smoothness-indicator analysis. It is the reconstruction basis for
  the offline one-polynomial-per-cell precursor; it does not establish a
  coupled system sensor or an admissibility policy by itself.
- M. Dumbser et al., *ADER-WENO finite volume schemes with space-time adaptive
  mesh refinement*,
  [arXiv:1212.3585](https://arxiv.org/abs/1212.3585): a conservative
  one-step finite-volume route coupling WENO data to a local space-time DG
  predictor, including Euler and MHD with AMR. This is the primary general
  precedent for a coherent space-time background; it does not supply exact
  persistent front topology.
- O. Zanotti and M. Dumbser, *Efficient conservative ADER schemes based on
  WENO reconstruction and space-time predictor in primitive variables*,
  [arXiv:1511.04728](https://arxiv.org/abs/1511.04728): the directly relevant
  HD/RHD/MHD/RMHD precedent for reconstructing conservative averages to point
  primitives, evolving one local primitive space-time polynomial, and then
  applying a conservative flux update. Equations 40--44 define the directly
  implemented research precursor: tensor-product Gauss-Legendre Lagrange
  space-time degrees of freedom, a weak primitive PDE integrated by parts in
  time, and a cell-local discrete Picard solve. It supports a complete
  ADER/GRP background, not a second-order Hancock flux spliced into
  Runge--Kutta stages.
- L. Anton et al., *Relativistic Magnetohydrodynamics: Renormalized
  eigenvectors and full wave decomposition Riemann solver*,
  [arXiv:0912.4692](https://arxiv.org/abs/0912.4692): primary reference for
  complete left/right RMHD eigenvectors that remain regular in degenerate
  physical states. Use it when replacing the offline numerical
  eigensystem/real-Schur reconstruction basis with generated production
  formulas; the paper does not validate PyFluid's Schur limiting fallback.
- L. Anton et al., *Numerical 3+1 general relativistic magnetohydrodynamics: a
  local characteristic approach*,
  [arXiv:astro-ph/0506063](https://arxiv.org/abs/astro-ph/0506063): primary
  reference for applying regular SRMHD characteristic information in a local
  orthonormal frame within GRMHD. It is the routing source for the future
  flat-space predictor-to-GR tetrad step, not evidence that CT/source/AMR
  coupling follows automatically.
- J. Li and Z. Du, *A Two-Stage Fourth Order Time-Accurate Discretization for
  Lax-Wendroff Type Flow Solvers. I. Hyperbolic Conservation Laws*,
  [arXiv:1512.03664](https://arxiv.org/abs/1512.03664), and Y. Yuan and
  H. Tang, *Two-stage fourth-order accurate time discretizations for 1D and
  2D special relativistic hydrodynamics*,
  [arXiv:1712.05546](https://arxiv.org/abs/1712.05546): primary GRP-based
  routes to fourth-order time accuracy, including an analytic SRHD GRP.
  They support deriving time-dependent physical flux rather than adding more
  Runge--Kutta sampling of static face traces.
- E. Gaburro and M. Dumbser, *A posteriori subcell finite volume limiter for
  general PNPM schemes: applications from gasdynamics to relativistic
  magnetohydrodynamics*,
  [arXiv:2010.04853](https://arxiv.org/abs/2010.04853): a broad
  HD/MHD/SRHD/SRMHD route for retaining a high-order space-time candidate and
  replacing only troubled subcells. It does not supply an exact persistent
  physical-front degree of freedom.
- J. Duan, P. Chandrashekar, and C. Klingenberg, *Active flux for ideal
  magnetohydrodynamics: A positivity-preserving scheme with the
  Godunov--Powell source term*,
  [arXiv:2506.04857](https://arxiv.org/abs/2506.04857): recent evidence for
  evolving interface point values as additional degrees of freedom instead
  of reconstructing every face state anew. Its multidimensional MHD
  formulation uses a different divergence treatment and is not a direct CT
  implementation route.
- Y. Ding and K. Wu, *High-order positivity-preserving finite volume methods
  for ideal MHD with locally divergence-free reconstruction*,
  [paper](https://doi.org/10.1137/23M1562081): direct projection of magnetic
  interface values into a discrete-divergence-free space. This is a close
  lexical collision for "trace projection," but its constraint and objective
  differ from paired oscillation control.
- C. Cao et al., *Convex oscillation suppression for high-order discontinuous
  Galerkin methods*, [paper](https://doi.org/10.1016/j.jcp.2026.114755):
  parameter-independent entropy-distance damping of a cell polynomial. It is
  a high-risk broad novelty collision, not the same paired finite-volume
  PWL/parabolic/MP5 trace polygon.
- K. Wu and C.-W. Shu, *Entropy Symmetrization and High-Order Accurate Entropy
  Stable Numerical Schemes for Relativistic MHD Equations*,
  [arXiv:1907.07467](https://arxiv.org/abs/1907.07467): entropy stability for
  a symmetrized Godunov--Powell-type RMHD system. This changes the equation
  discretization and is not a drop-in reconstruction or Riemann-solver swap
  for the present CT formulation.
- C. Huang, A. S. Sebastian, and V. Viswanathan, *Learning second-order TVD
  flux limiters using differentiable solvers*,
  [arXiv:2503.09625](https://arxiv.org/abs/2503.09625): a constrained learned
  limiter represented as a convex combination inside the classical TVD
  admissible family. This supports offline constrained coefficient/search
  experiments; it is not evidence for deploying an unconstrained neural
  reconstruction in production RMHD.

## Local AsterX Robust-Reconstruction Reference

The local AsterX checkout is under
`/Users/mrizzo/phys_research/ETK_CarpetX/CactusX/repos/AsterX`. For the narrow
reconstruction question, use:

- J. V. Kalinani et al., *AsterX: a new open-source GPU-accelerated GRMHD code
  for dynamical spacetimes*, Class. Quantum Grav. 42, 025016 (2025),
  [journal](https://doi.org/10.1088/1361-6382/ad9c11),
  [arXiv:2406.11669](https://arxiv.org/abs/2406.11669). Section 4.1 and
  figure 3 compare Balsara 1--5 against the Giacomazzo--Rezzolla exact solver.
  The published curves show only rest-mass density and `By`: tests 1, 2, 4,
  and 5 use third-order PPM, while the extreme-pressure test 3 deliberately
  uses more dissipative second-order MINMOD. The paper describes WENO-Z and
  MP5 as implemented but still under testing, so this figure is not evidence
  for their published Balsara behavior or for an RK-stage FOFC guard.
- `AsterX/src/fluxes.cxx`: component reconstruction, selectable `z_vec = Wv`
  reconstruction, face-metric conversion, and per-face-side low-order fallback
  when reconstructed thermodynamic variables are nonpositive;
- `ReconX/src/mp5.hxx`: the MP5 formula;
- `ReconX/param.ccl`: independent primary and low-order method parameters;
- `AsterX/src/con2prim.cxx` and `Con2PrimFactory/src/`: atmosphere, velocity
  ceiling, and ordered recovery behavior.

### Multipatch follow-up (Cycle 410)

- [Focused CapyrX architecture review](docs/capyrx_architecture_review.md): pinned
  October 2026 interpolation/cache/centering source and derivative interfaces;
  lessons for later PyFluid geometry/transfer ownership, not conservative FV/CT
  seam acceptance.
- [ET 2026_05 release notes](https://www.einsteintoolkit.org/about/releases/ET_2026_05_announcement.html): NewRadX gained CapyrX multipatch support.
- [CarpetX developer notes, January 2026](https://lists.einsteintoolkit.org/pipermail/carpetx-developers/2026-January/000538.html): split-phase interpolation work for CapyrX and its possible CCE use. This is infrastructure provenance, not evidence of conservative GRMHD interface coupling.
- [Thornburg (2004), Black Hole Excision with Multiple Grid Patches](https://arxiv.org/abs/gr-qc/0404059): angular patches, tensor bases and interpatch boundary design for black-hole evolution.
- [Pollney et al. (2011), High accuracy binary black hole simulations with an extended wave zone](https://arxiv.org/abs/0910.3803): Cartesian near zone with nonsingular adapted outer patches; a binary-grid design reference, not a conservative GRMHD seam implementation.

### Particle scope references (Cycle 431)

- [AMReX particle documentation](https://amrex-codes.github.io/amrex/docs_html/Particle.html): containers, AMR/MPI redistribution, mesh interpolation/deposition and output. The pinned local documentation is `.deps/amrex-26.10/Docs/sphinx_documentation/source/Particle.rst`; current CPU and Metal builders select `AMReX_PARTICLES=OFF`. CMake has a separate `AMReX_PARTICLES_PRECISION` policy, defaulting to mesh precision in this release, which must remain explicit in any future Metal particle audit.
- [Price, Smoothed Particle Hydrodynamics and Magnetohydrodynamics](https://arxiv.org/abs/1012.1885): particle discretization of fluid/MHD equations, with its own stability, dissipation and magnetic-divergence issues. SPH does not remove the fluid closure assumption.
- [Ryan, Dolence and Gammie, bhlight](https://arxiv.org/abs/1505.05119): GR radiation MHD using Monte Carlo photon transport; a reference for a physical radiation-packet method beyond particle storage.
- [Parfrey, Philippov and Cerutti, First-Principles Plasma Simulations of Black-Hole Jet Launching](https://arxiv.org/abs/1810.03613): collisionless GR plasma and pair supply in Kerr magnetospheres. This illustrates the different physics of kinetic particles, not an automatic fallback for a troubled fluid cell.
