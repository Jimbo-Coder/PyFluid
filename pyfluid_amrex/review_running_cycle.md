# Running review cycle tracker

Only this file is authoritative for task continuation.
I will always continue from the last cycle marked as `in_progress`.

## Cycle 7 [completed]
- Rebuild AMReX with HDF5 support first, then continue.
  - Current status: done for single-step rebuild path.
  - HDF5 now found, but only non-parallel HDF5 exists in current brew install.
  - AMReX HDF5 rebuild now succeeds with `-DAMReX_MPI=OFF` (required because AMReX with MPI requires parallel HDF5).
- Fix and continue from there: keep code cleanups and unification work without creating temporary duplicate implementations.
  - Current status: keep checkpoints/output/signature and vector-identity cleanup fixes done in prior cycle applied successfully.
- Benchmark/compare output/cpkt path behavior after output changes.
  - Current status: done.
  - Medium AMR test run (2D hydro_t_intersection_pulse_2d_amr, max_level=8, plot_int=4, final_time=0.75, --ranks 1, --policy smoke):
    - amrex text output: `runs/current/benchmark_amr_amrex_smoke` (82.105 s wall, plotfiles 763M, aux_plotfiles 155M, run dir 918M, 98 plotfiles).
    - hdf5 + best: `runs/current/benchmark_amr_hdf5_smoke` (83.867 s wall, plotfiles 765M, aux_plotfiles 155M, run dir 920M, 98 plotfiles / 196 files).
    - hdf5 + none: `runs/current/benchmark_amr_hdf5_smoke_nocomp` (82.816 s wall, plotfiles 760M, aux_plotfiles 154M, run dir 914M, 98 plotfiles / 196 files).
  - hdf5 readers work with existing tooling: `plotfile_reader.parse_plot_header`, `sorted_plotfiles`, and `latest_aux_plotfiles` all load with matching fields/time.

## Cycle 8 [completed]
- Fix the compression token mapping so AMReX actually receives its expected compression codes.
  - Current status: done.
  - `resolved_plotfile_compression` now maps:
    - `best -> ZLIB@9`
    - `zlib -> ZLIB@6`
    - `zlibN -> ZLIB@N`
- Rebuild after code change and rerun the medium AMR case to verify behavior.
  - Current status: done.
  - `mpirun` is blocked in this environment with `bind() failed for port 0`/socket restrictions; ran direct executable instead.
- `hdf5 + best` (`pyfluid.plotfile_compression=best`):
    - output dir: `runs/current/benchmark_amr_hdf5_smoke_best_fix`
    - wall time: `88.46 s`
    - total run dir: `115M` (`plotfiles` 111M, `aux_plotfiles` 3.9M)
    - HDF5 filter confirmation: `h5stat -d` shows `GZIP filter: 3` and `SHUFFLE filter: 3` on `plt00032.h5`.
- `hdf5 + none` (`pyfluid.plotfile_compression=none`):
    - output dir: `runs/current/benchmark_amr_hdf5_smoke_nocomp_fix`
    - wall time: `82.31 s`
    - total run dir: `914M` (`plotfiles` 760M, `aux_plotfiles` 154M)
    - HDF5 filter confirmation: `h5stat -d` shows `NO filter: 12`, no GZIP/SHUFFLE.

## Cycle 9 [completed]
- Split and prioritize remaining output/checkpoint/metadata simplification work.
- Current status: done.
  - 1) Unify metadata emission for both plotfile types behind a single policy.
     - Goal: write all non-binary output metadata into HDF5 attrs when writing HDF5 plotfiles; keep `*.meta` text-sidecar for legacy AMReX text format.
     - Current status: done.
     - Completed by moving all `plotfile_format`/`plotfile_compression` metadata to one helper path (`append_output_metadata`) and writing as HDF5 attrs or text-sidecar uniformly.
  - 2) Make HDF5+compression the default for normal GF output.
     - Goal: set default config to `plotfile_format=hdf5`, default compression to a compressed mode (likely `best`), while preserving `amrex` as explicit opt-out.
     - Current status: done.
     - Current defaults already target `plotfile_format=hdf5`, `plotfile_compression=best`; this is now validated through the same metadata helper path.
  - 3) Align checkpoint storage strategy with GF output pipeline.
     - Goal: write checkpoint GFs in compressed HDF5 path when available (`pyfluid.plotfile_format=hdf5` parity), keep checkpoint metadata sidecar (`Header.meta`) for structured headers, and preserve restart compatibility.
     - Current status: pending.
     - Design note from user: checkpoint output should be a complete mandatory set of required GFs (state + aux needed for restart/recovery), with metadata split as:
       - minimal per-GF book-keeping metadata in the file itself (where supported)
       - a separate checkpoint-side text file for full run config/invariants/history that full validation can assert against on restart.
     - Accepted outcome target: reduce duplicate metadata, keep run/restart-relevant data explicit, preserve compatibility, and keep total file footprint practical.
  - 4) Clarify scalar output modes.
     - Goal: keep append-at-run `scalars.tsv` as baseline; add option for end-of-run-only scalar emission if/when desired, without breaking current validation tools that read the existing TSV stream.
     - Current status: done.
     - `scalar_output_mode = stream|final` is already wired in the solver (stream updates on cadence, final computes once).
- 5) Finalize unified output/checkpoint API surface.
   - Goal: one high-level output/writepoint for state/aux/checkpoint paths, with clear format/compression/metadata policy and no duplicate code paths.
   - Current status: done.
   - Metadata policy is consolidated.
- 6) Unify plotfile naming and output folder layout for state and aux GFs.
   - Goal: move all state outputs to `stateGF*` under `stateGF` and all aux outputs to `auxGF*` under `auxGF`, one root per role.
   - Current status: done.
   - C++ output writer now uses `stateGF` and `auxGF` stems and writes state/aux under separate root folders.
   - Python plotfile readers were updated to resolve the new `stateGF`/`auxGF` roots, map legacy `stateplt`/`auxplt` names for compatibility, and use the shared layout consistently.
   - Plot scripts and docs now prefer the `stateGF`/`auxGF` layout by default, with reader-level compatibility for legacy `plotfiles` runs retained.

## Cycle 11 [completed]
- Unify basic 3-vector GF math into one reusable class.
  - Goal: define a single `GfVector3` base with operators/reductions (`+`, `-`, `*`, `/`, `dot`, `cross`, `sum`, `squared_norm`) and make `StateVector3` and `PhysicalVector` reuse it.
  - Current status: done.
  - Completed in `state.hpp` (`GfVector3`, updated `StateVector3`, and wrapper functions delegating to shared methods) and `patch_metrics.hpp` (`PhysicalVector : GfVector3` with shared `dot`/`cross`/`scale` paths).
  - Follow-up in next cycle: add the same pattern for any 4-vector helper or tensor helper containers where the same operations repeat (`Vector4`, 3x3 tensor, stress tensors), then migrate remaining explicit scalar-level callsites gradually.

## Cycle 12 [completed]
- Extend vector value-layer semantics and apply shared vector math at callsites.
  - Goal: use one canonical vector math type instead of temporary scalar/vector alias layers, while keeping semantic wrapper names where needed.
  - Current status: done.
  - `state.hpp`: removed temporary scalar/vector alias placeholders and kept shared vector math in `Vector3`, with semantic wrappers (`StateVector3` and `PhysicalVector`) using it.
  - `mapped_patch_interfaces.hpp`: removed local `subtract`/`add` helpers and replaced point-pair usage with `PhysicalVector` operators.
  - `riemann.hpp`: converted normal-velocity/momentum-flux expressions to shared `dot`/`StateVector3` math.

## Cycle 13 [completed]
- Canonicalize vector naming and document scalar semantics.
  - Goal: keep "single-name unless needed" naming (`Vector3` as the shared class, semantics via wrappers).
  - Current status: done.
  - `state.hpp`: renamed the shared value type from `GfVector3` to `Vector3` and updated `StateVector3` inheritance/ctor forwarding.
  - `patch_metrics.hpp`: updated `PhysicalVector` to inherit from `Vector3`.
  - Documentation intent update: true scalars are values (`amrex::Real`), vectors are `Vector3` bundles.

## Cycle 14 [completed]
- Add a clean time-augmented value model (scalars/vectors/tensors) without touching hot AMReX kernels.
  - Goal: keep current single-step math types for performance and add optional, explicit time-indexed wrappers for readers, diagnostics, and postprocessing.
  - Goal refined: keep value types as today (`amrex::Real` scalars, `Vector3` vectors), and provide generic time containers for non-kernel data.
  - Actual work done:
    - `fields.hpp`: added `TemporalCadence`, `TimeSample<T>`, and `OverTime<T>` with max-frame trimming.
    - Aliases for immediate use: `ScalarOverTime`, `Vector3OverTime`, `PathOverTime`.
    - `plotfile_reader.py`: added `PlotfileSample` and `plotfile_timeline()` to represent a compact over-time plotfile trajectory with optional frame limiting.
- Current status: done (core scaffolding in place).

## Cycle 15 [completed]
- Expand trajectory use in tooling and output scheduling.
 - Goal: consume `plotfile_timeline()` in plotting/validation paths so "how many frames" and cadence are explicit.
 - Planned follow-up: keep the output cadence decision close to timeline storage rather than duplicated `step` checks.
 - Constraint: avoid introducing duplicate implementations; reuse shared timeline/container interfaces for both scalar and field trajectories.
 - Actual work done:
   - Added reusable plotfile timeline sampling helpers (`_select_timeline_stride`, `_select_timeline_target_frames`) to `plotfile_reader.py`.
   - `plotfile_timeline()` now supports `frame_stride` and `target_frames` selection with explicit validation.
   - `plot_kh_spectrum.py` now uses `plotfile_timeline` for deterministic frame capping.
   - `plot_amrex_boxed_movie.py` now uses `plotfile_timeline` for `--frame-stride` / `--target-frames` selection instead of ad-hoc index math.
   - C++ evolve loop now uses `TemporalCadence::should_record()` for scalar/plot/aux/checkpoint cadence decisions while keeping the existing end-of-run behavior.

## Cycle 16 [completed]
- Unify output and checkpoint initial-step cadence paths using a single cadence API.
  - Goal: remove duplicated `cfg_.plot_int == 0` / `cfg_.aux_gf_int == 0` / `cfg_.checkpoint_int == 0` checks and route all decisions through `TemporalCadence`.
  - Planned change: use a dedicated `TemporalCadence::should_record_initial()` path for startup output and rely on `should_record()` everywhere else.
  - Planned status:
    - `TemporalCadence` now exposes `should_record_initial()` in `fields.hpp`.
    - `evolve_core.inc.hpp` now uses cadence-based initial checks for state/aux/checkpoint.
    - Output behavior remains unchanged (interval `0` remains startup-only output).
- Refine time-augmented field naming and utility methods.
  - Goal: replace the generic `OverTime` wording with a clearer field-oriented `GridFunctionTs` convention and add simple helper methods for recording and lifecycle control.
  - Status:
    - `GridFunctionTs<T>` now replaces the old class body; `OverTime<T>` remains as a compat alias.
    - Added `ScalarTs`, `Vector3Ts`, `PathTs`, and `Tensor3x3Ts` aliases for field-time history types.
    - Added `record`, `clear`, and `at_or_null` helpers for cleaner high-level usage.
- AMR curvilinear-directional indicator cleanup.
  - Goal: remove nonphysical tagging contamination from missing canonical axes in 2D curvilinear runs.
  - Status:
    - `state_vector_cartesian_component` now zero-fills canonical directions that are not active in `mesh.axes` instead of reading stale third-component data.
  - AMR signal-speed (`b2`) and magnetic-pressure indicators in the directional indicator path now use cartesianized B components consistently.
  - This keeps directional AMR refinement signals focused on active geometric directions.
- HDF5 plotfile metadata path fix for output append.
 - Goal: fix metadata append on HDF5 plotfiles by targeting `.../pltXXXXX.h5` instead of stem-only `.../pltXXXXX`.
 - Current status: done.
 - `append_plotfile_metadata` in `output.hpp` now appends HDF5 attrs to the actual HDF5 file path.
 - Build verification attempt:
    - `cmake --build build/pyfluid_amrex_optamrex --target pyfluid_amrex_2d -j 3` currently fails before reaching this change-set with unrelated errors:
      - `Vector3`/`PhysicalVector` conversion issues in `patch_metrics.hpp` and
      - missing `AMReX_MLMG.H` in this build config.
- Rebuild path/type fix for MLMG + vector typing.
  - Goal: make vector type usage single-source (`Vector3`/`PhysicalVector`) and compile against MLMG-enabled AMReX.
  - Status: done.
  - `PhysicalVector` in `patch_metrics.hpp` is now `using PhysicalVector = Vector3;`.
  - `riemann.hpp` normal-velocity helpers use `StateVector3::dot` directly to avoid ambiguous free-function `dot` overloads.
  - Reconfigure build with MLMG-enabled AMReX:
    - `-DAMReX_DIR=/Users/mrizzo/phys_research/PyFluid/.deps/amrex-lsolvers/lib/cmake/AMReX`
  - Rebuild result:
    - `cmake --build build/pyfluid_amrex_optamrex --target pyfluid_amrex_2d -j 3` succeeds.
- CMake guardrail for AMG/MLMG availability.
  - Goal: prevent silent misuse of AMReX builds without linear solvers when implicit-gravity paths are enabled.
  - Status: done.
  - `pyfluid_amrex/CMakeLists.txt` now defines `PYFLUID_AMREX_ENABLE_IMPLICIT_GRAVITY` (default ON) and fails configure if
    `AMReX_LSOLVERS`/`AMReX_LSOLVERS_FOUND` are OFF.
  - Reconfigure/build result:
    - `cmake -S pyfluid_amrex -B build/pyfluid_amrex_optamrex -DAMReX_DIR=/Users/mrizzo/phys_research/PyFluid/.deps/amrex-lsolvers/lib/cmake/AMReX`
    - `cmake --build build/pyfluid_amrex_optamrex --target pyfluid_amrex_2d -j 3`
    - success.

## Cycle 10 [completed]
- Ensure defaults use the compressed writer path by default.
  - Current status: done.
  - Updated `RunConfig::checkpoint_compression` default from `none` to `best`.
  - Updated `write_checkpoint` API default argument to `best` so C++ callers that omit the field inherit compressed-mode default.
- Keep plot readers format/layout agnostic.
  - Current status: done.
  - Existing reader compatibility (`stateGF`/`auxGF` with legacy `stateplt`/`auxplt` remap) remains in place; no separate reader implementation path was introduced.
- Validation.
  - Not rerun in this cycle (default-only config/API cleanup).

## Cycle 17 [completed]
- Generate a cleaner curvilinear validation movie pair with targeted plotting defaults and no duplicate outputs.
- Current status:
  - Build a dedicated inward-origin pulse case (r-phi, `bc_xlo=origin`, `source=none`) with AMR and plot output.
  - Build a rotating quadrupole source case with explicit output cadence and improved movie settings.
  - Run both cases through a unified movie workflow:
    - `--target-frames 180`
    - `--symmetric-limits` (for inward pulse; not used for quadrupole)
    - `--curvilinear-view r-phi`
    - `--max-pixels 1100`
    - `--subtract-constant 0.04`
  - Results:
   - Reused clean AMR outputs for the inward-origin case (`inward_origin_pulse_v4`) and quadrupole source (`quadrupole_source_v2`) from existing high-density runs.
    - Generated new boxed and no-box movies under:
      - `runs/current/curvilinear_validation_movies/inward_origin_pulse_v4/gifs_clean/inward_origin_pulse_v4_pressure_rphi_clean.gif`
      - `runs/current/curvilinear_validation_movies/inward_origin_pulse_v4/gifs_clean/inward_origin_pulse_v4_pressure_rphi_clean_nobox.gif`
      - `runs/current/curvilinear_validation_movies/quadrupole_source_v2/gifs_clean/quadrupole_source_v2_pressure_rphi_clean.gif`
      - `runs/current/curvilinear_validation_movies/quadrupole_source_v2/gifs_clean/quadrupole_source_v2_pressure_rphi_clean_nobox.gif`
    - Each GIF is 180 frames at 140 ms/frame, size `1224x1054`.
  - Note:
    - A longer dedicated inward-origin rerun (`inward_origin_pulse_v5`) was attempted with faster crossing parameters but was manually interrupted because the requested `final_time=1.0` produced too many AMR steps for quick turnaround.

## Cycle 18 [completed]
- Run inward-origin and rotating-source curvilinear movie cases using `pyfluid.subcycling=on`, `pyfluid.timestep=ssprk3`, low base resolution, and AMR-driven refinement.
  - Plan:
    - Launch a fresh inward pulse run with coarse `amr.n_cell`, `pyfluid.subcycling=on`, `amr.max_level` > 1, and existing HDF5 output path.
    - Launch a fresh rotating source run with the same subcycling/SSPRK3 defaults and stronger source coupling to show outward transport.
    - Generate movies with robust plotting limits (`--symmetric-limits`, `--subtract-constant 0.04`) where helpful.
  - Status: completed.
  - Executed with `build/pyfluid_amrex_optamrex/pyfluid_amrex_2d` (stateGF-compatible output) and explicit output settings:
    - `amr.n_cell = 40 64`, `geometry.prob_lo = 0.0 0.0`, `pyfluid.subcycling = on`, `pyfluid.subcycling_steps = local_cfl`, `amr.refinement_tensor = fixed`.
    - Inward pulse: `curvilinear_inward_origin_pulse_v8` (`run_time=0.30`, `v1_0=-1.5`, `amr.max_level=3`, `tag_threshold=0.025`, `plotfile_format=hdf5`, `plotfile_compression=best`).
    - Quadrupole source: `curvilinear_quadrupole_source_v3` (`run_time=0.80`, `source_power=20.0`, `source_ramp_time=0.05`, `amr.max_level=3`, `tag_threshold=0.02`, `plotfile_format=hdf5`, `plotfile_compression=best`).
  - Output artifacts:
    - `runs/current/curvilinear_validation_movies/inward_origin_pulse_v8/inward_origin_pulse_v8_pressure.gif`
    - `runs/current/curvilinear_validation_movies/inward_origin_pulse_v8/inward_origin_pulse_v8_pressure_nobox.gif`
    - `runs/current/curvilinear_validation_movies/quadrupole_source_v3/quadrupole_source_v3_pressure.gif`
    - `runs/current/curvilinear_validation_movies/quadrupole_source_v3/quadrupole_source_v3_pressure_nobox.gif`

## Cycle 19 [completed]
- Consolidate the AMReX developer and user interface after parallel-agent work.
  - Ledger convention: this is the single append-only task ledger. New work is appended as a cycle; subtask state is `PENDING`, `WIP`, or `DONE`. Completed Cycle 6 checkpoint/API fixes are retained in the prior cycle file until this consolidation is verified, then that redundant file will be removed.
  - 1) Establish this file as the only running task ledger and retire redundant cycle files. [DONE]
  - 2) Inventory supported runtime features, experimental paths, fallback behavior, and their documentation. [DONE]
  - 3) Remove dead or duplicate compatibility/fallback paths when their replacement is verified. [DONE]
  - 4) Make the documented build, run, validation, output, and code-generation workflows agree with the implementation. [DONE]
  - 5) Run focused build and validation gates, then record exact scope and remaining explicit limitations. [DONE]

### Cycle 19 update
- The active preset pointed at `.deps/amrex-lsolvers`, whose installed `AMReXConfig.cmake` reports `AMReX_MPI=OFF` and one configured space dimension. This contradicted the normal three-rank 1D/2D/3D workflow. Added an MPI-enabled AMReX install with linear solvers, pointed the canonical preset at it, and switched the default plotfile format to native AMReX unless a parallel-HDF5 build is installed. [DONE]

### Cycle 19 update
- The initial three-rank probe used a stale serial-AMReX CMake cache: each `mpirun` process independently identified as the IO processor, emitted a completion line, and raced on `scalars.tsv`. The new AMReX install itself was MPI-enabled; the preset now pins `AMReX_DIR` and CMake rejects serial AMReX at configure time.
- Rebuilt the 2D target against `.deps/amrex-mpi-lsolvers`. A three-rank hydro sine probe now reports one MPI communicator and one completion line; its scalar report passes with mass/energy/momentum drift near roundoff and about 56 Mcells/s.
- Runtime scalar readers now reject incomplete scalar rows with the source line number. Generic history TSV readers remain permissive because their optional columns are intentionally blank. [DONE]
- Consolidated the completed checkpoint/API notes from Cycle 6 and the old Cycle 10 origin-run record into this canonical ledger. The old side ledgers are removed; their stale serial-HDF5 and legacy-plotfile fallback notes are historical only and do not describe the current workflow. [DONE]
- Removed the generated root-level VisIt log that referenced a deleted legacy `plotfiles/plt...` run. Renamed the maintained CFL representative sweep from `legacy` to `baseline`; the cases and numerical coverage are unchanged. [DONE]
- Fixed `validate_amr_global_tagging.py` to consume metadata-bearing `amr_levels.tsv` through the shared TSV reader. Its three-step broad-flow case reaches L1 with 14,336/16,384 level-0 cells tagged; the stale L2/eightfold requirement is replaced by the intended L1/fourfold broad-refinement gate. Passive dye remains verified as non-tagging. [DONE]
- Fixed the directional-AMR gate to match its intended independent-direction contract: a requested direction must refine on at least one active level, while an unrequested direction must never refine. It no longer incorrectly requires every requested direction to refine at every level after another direction has already converged. [DONE]
- Corrected `run_case.py` physics-hook dispatch: scalar validation is unconditional, but plotfile-based pressure-bump isotropy/locality checks only run when state GF output exists. Curvilinear smoke cases intentionally use `plot_int=-1`; their completed scalar-valid runs no longer fail after evolution merely because a level-3 snapshot is absent. [DONE]
- Consolidated AMReX metadata-table parsing in `validation_common.py`: `read_float_tsv` now delegates to the same metadata-aware reader as scalar validation. The AMR global/tensor/score and obstacle-wall gates no longer carry incompatible private `DictReader` implementations. [DONE]
- Split curvilinear verification by concern. The Cartesian/cylindrical field-equivalence gate now holds the shared seeded AMR hierarchy after initialization (`regrid_int=100000`), avoiding polar-CFL-dependent regrid-count differences; the dedicated curvilinear freestream/regrid gate remains responsible for dynamic regrid safety. The controlled three-rank L2 comparison passes. [DONE]
- Restored the public `read_level0_state_1d` name as a thin wrapper over the canonical generic `read_level_state` reader. This fixes the Sod/diffusion/gravity/dispersion postprocessing import without restoring a duplicate 1D reader implementation. [DONE]
- Continued metadata-reader consolidation through the standard analytic paths: the EOS acoustic validator now uses `read_float_tsv` rather than treating AMReX output metadata as a CSV header. [DONE]
- Fixed the 1D profile header wrapper to use the canonical two-value 1D adapter rather than the generic three-value plotfile reader. Sod exact validation now reaches numerical comparison instead of failing during postprocessing setup. [DONE]

### Cycle 19 update
- Re-established performance provenance. The previous hard-coded baselines came from a serial AMReX cache run under `mpirun`; they were not aggregate MPI-throughput measurements. `benchmark_pyfluid.py` now requires its stdout to confirm the requested MPI process count, records `execution_mode`, `mpi_processes`, and a labeled real-MPI baseline, and treats a serial or mismatched launch as a failure.
- The current real-MPI baseline medians are `56.35` Mcells/s (2D PWC/LLF/Euler, 3 ranks), `64.81` (the same, 4 ranks), `4.77` (3D PWC/LLF/Euler, 3 ranks), and `0.157` (3D WENO-Z/HLLC/RK4, 3 ranks). They are the first comparable references for this installed AMReX/MPI stack; the old mixed history will be preserved as a legacy schema when the updated benchmark writes its new history.
- Fixed `capture_baseline_run.py::row_count`: its TSV/CSV counter was unreachable after `summary_steps`, so baseline manifests wrote blank row counts. [DONE]

### Cycle 19 update
- Repeated the complete benchmark after the validation/build load ended. The clean three-repeat medians are now the canonical `real_mpi_2026-07-09_clean` baseline: 2D PWC/LLF/Euler `109.9095765` Mcells/s (3 ranks) and `137.0855579` (4 ranks); 3D PWC/LLF/Euler `6.373059139` (3 ranks); 3D WENO-Z/HLLC/RK4 `0.1900616783` (3 ranks). The earlier 56/65/4.77/0.157 values are retained only as explicitly provisional, loaded-machine measurements in the rolling history. [DONE]

### Cycle 19 final validation
- Reconfigured and rebuilt `full-1d`, `full-2d`, and `full-3d` with the pinned MPI+linear-solvers AMReX installation. The CMake guard rejects a serial AMReX configuration.
- `validate_amrex_suite.py --level standard` passed on the real three-rank MPI executable: feature matrix, CT, AMR/regrid, Cartesian and mixed multipatch, static/moving solids, dimension/rank consistency, generated fluxes, curvilinear families/origin, controlled seeded coordinate equivalence, Sod, sources, gravity, Brio-Wu reference, EOS, dispersion, and explicit/implicit diffusion all passed.
- Mechanical checks passed: Python tool compile, CMake preset JSON parse, `git diff --check`, canonical-output/legacy-alias search, real-MPI benchmark guard, and baseline-manifest row counting.
- Explicit scope retained: mixed Cartesian-curvilinear seam evolution remains experimental; 3D multilevel implicit CT resistivity remains unsupported. These are documented limitations, not fallback paths. [DONE]

## Cycle 20 [completed]
- Make compressed HDF5 the canonical AMReX grid-function output when it is demonstrably smaller and fully compatible with MPI evolution, shared readers, diagnostics, plotting, and restart-independent workflows.
  - 1) Inventory the installed HDF5/AMReX MPI capabilities and reject serial-HDF5 linkage. [DONE]
  - 2) Build one MPI+linear-solvers+parallel-HDF5 AMReX installation and route the canonical preset through it. [DONE]
  - 3) Compare native AMReX and HDF5 output size and write cost on the same representative run. [DONE]
  - 4) Verify one shared state/aux reader and plotting path against both formats with equivalent fields, metadata, and scalar diagnostics. [DONE]
  - 5) If the evidence is favorable, set HDF5 with `best` compression as the default, update concise documentation, and run focused plus standard validation. [DONE]

### Cycle 20 results
- Installed and activated Homebrew `hdf5-mpi` 2.1.1. `h5pcc -showconfig` confirms parallel HDF5, parallel filtered dataset writes, large parallel IO, and deflate encode/decode support.
- Built and installed `.deps/amrex-mpi-lsolvers-hdf5` for dimensions 1/2/3 with MPI, linear solvers, and HDF5. The canonical CMake preset now pins that install and `HDF5_ROOT`; PyFluid configure rejects AMReX builds without MPI or HDF5.
- Fixed MPI metadata output: collective plotfile writes are synchronized and only the IO rank appends native metadata or HDF5 attributes. HDF5 files are now self-contained without `.meta` sidecars.
- Fixed the shared reader to initialize cells outside actual AMR boxes to `NaN` instead of uninitialized memory. Native and HDF5 readers now have identical masks and bit-identical valid values at every compared level/component.
- Consolidated hierarchy, dynamic-regrid, solid-volume, pulsating-solid, solid-sphere, gravity, and diffusion tools onto shared format-agnostic plotfile discovery. No validation tool retains a fixed `stateGF000xx`/`auxGF000xx` path or directory-only lookup.
- Representative 2D AMR run, eight frames: native AMReX `5728 KiB`; HDF5 ZLIB@9 `3396 KiB`, a 40.7% reduction. Output stage cost was 5.7 ms native versus 34.2 ms HDF5 for all eight frames. ZLIB@1 saved only 1.3% less disk while reducing HDF5 write cost to 25.8 ms; `best` remains the default because normal GF output is sparse and disk reduction is the requested priority.
- `pyfluid.plotfile_format=hdf5` and `pyfluid.plotfile_compression=best` are now defaults. Native AMReX remains explicit with `pyfluid.plotfile_format=amrex`.
- Native AMReX output now records compression as `None@0` rather than incorrectly echoing the HDF5 compression request. A focused three-rank native run passes and the shared metadata reader reports the corrected value.
- Rebuilt full 1D/2D/3D binaries. The real-MPI standard suite passed with HDF5 default across all gates. The no-output 3-rank benchmark measured `109.95` Mcells/s against the `109.91` baseline (ratio `1.00`). [DONE]

## Cycle 21 [completed]
- Consolidate output/restart execution contracts and commit the validated AMReX production implementation.
  - 1) Keep native AMReX and HDF5 GF output behind one schema/discovery contract, without removing either format. [DONE]
  - 2) Implement actual lossless compression for checkpoint grid functions; keep checkpoint metadata readable text. [DONE]
  - 3) Make checkpoint metadata mandatory and validate only immutable state/layout semantics, while allowing documented run-control changes on restart. [DONE]
  - 4) Centralize MPI process launching for AMReX Python tools and migrate direct launch sites to it. [DONE]
  - 5) Rebuild and validate checkpoint/restart equivalence, both GF formats, the standard suite, and performance. [DONE]
  - 6) Stage only the production AMReX setup and its required shared files, inspect the staged tree, and create a scoped commit. [DONE]
- Explicitly deferred: multipatch development and the declarative configuration-schema rewrite. [DONE]

### Cycle 21 validation to date
- All 1D/2D/3D solver and checkpoint-comparison targets rebuild cleanly against the canonical MPI+linear-solvers+HDF5 AMReX installation. Python tools/codegen compile and `CMakePresets.json` parses. [DONE]
- Checkpoint state is stored as one multilevel HDF5 file per field with `ZLIB@9`; `h5dump` confirms shuffle plus DEFLATE level 9. `Header` and `Header.meta` remain text. [DONE]
- One-rank focused restart probes are bit-identical at the final synchronized checkpoint for two-level AMR hydro and for CT MHD, including every face-centered magnetic field. Changing timestepper/reconstruction/Riemann/CFL on restart succeeds; changing the ideal-gas gamma aborts on the named immutable metadata key. [DONE]
- Native AMReX and HDF5 state output both pass the same `run_case.py` scalar report and requested GF snapshot plotting path. A one-rank Sod analytic run confirms migrated validation scripts use direct execution rather than wrapping rank one in `mpirun`. [DONE]
- The current Codex external-execution allowance was exhausted before the post-change three-rank standard suite and benchmark could run. The previous Cycle 20 standard suite and 109.95 Mcells/s benchmark passed before these checkpoint/launcher-only changes; current real-MPI confirmation remains pending and must not be claimed. [PENDING]
- The same external-execution limit leaves `.git` read-only in this session, so the requested scoped stage/commit could not be created. The intended stage set is `.gitignore`, root/AMReX docs, `AGENTS.md`, `pyproject.toml`, `uv.lock`, `pyfluid_amrex/`, and the shared `pyfluid` exact-solution/dispersion modules; JAX, Numba, and legacy scripts remain out of scope. [PENDING]

### Cycle 21 resumed validation
- The current real-MPI standard suite passed every gate. The canonical benchmark remains at baseline: 2D PWC/LLF/Euler measured `109.73` Mcells/s on 3 ranks and `137.52` on 4 ranks; 3D PWC/LLF/Euler measured `6.36`, and 3D WENO-Z/HLLC/RK4 measured `0.19`. Every baseline ratio rounds to `1.00`. [DONE]
- The focused three-rank compressed checkpoint gate is exact for two-level AMR hydro and CT MHD with subcycling on and off: every comparison reports `checkpoint_max=0` and `total_l1=0`. Fixed the focused `validate_amr.py --case` path so it skips improvement checks whose prerequisite comparisons were intentionally filtered out instead of emitting artificial infinite-factor failures. [DONE]
- Created the scoped AMReX production commit. The staged tree contained no generated runs, plots, checkpoints, build products, caches, or media; unrelated JAX, Numba, and legacy-script work remains unstaged. [DONE]

## Cycle 22 [completed]
- Audit the committed AMReX production tree for the next highest-value cleanup that reduces maintainer/agent overhead without changing validated physics or performance. [DONE]
  - 1) Check remaining configuration, execution, build/install, generated-code, and validation ownership boundaries. [DONE]
  - 2) Identify concrete duplication, fallbacks, hard-coded local assumptions, and oversized change hotspots. [DONE]
  - 3) Recommend a small prioritized cleanup sequence; do not reopen deferred multipatch physics. [DONE]

### Cycle 22 findings
- Highest-value structural cleanup: replace the 1,986-line, 251-query `config.hpp` ownership tangle with one declarative parameter registry that supplies defaults, AMReX query keys, accepted values, validation metadata, restart mutability, and generated concise documentation. Keep the runtime `RunConfig` and tiny template-facing predicates in C++; move parsing/validation implementation out of the widely included header. `config.hpp` currently reaches 18 compilation units, so this also targets incremental compile cost. [DONE]
- Finish execution consolidation: command construction is centralized and no tool duplicates `mpirun -np`, but 43 Python `subprocess.run` sites remain. Solver/MPI invocations should use `run_amrex`/`run_mpi`; direct subprocesses should remain only for CMake, plotting, and deliberate child-tool execution. [DONE]
- Make dependency setup portable without weakening the pinned local build: the canonical preset and install guide hard-code `/opt/homebrew`, and `benchmark_install.py` contains a user-specific AMReX example path. Add one bootstrap/configure helper that discovers `brew --prefix`, accepts an AMReX source/install override, verifies MPI/HDF5/linear-solver capabilities, and writes/uses a local user preset rather than adding runtime fallbacks. [DONE]
- After the parameter registry, split only non-template implementation out of the largest headers. Leave coordinate/physics templates and generated kernels alone unless profiling identifies a runtime or compile bottleneck. Add a small existing-suite architecture gate for the resulting invariants rather than another validation framework. [DONE]
- No urgent cleanup is justified in GF ownership, output/checkpoint schema, generated-kernel dependency tracking, or numerical dispatch. Those boundaries are now explicit, validated, and should remain stable. Multipatch physics remains deferred. [DONE]

## Cycle 23 [completed]
- Begin the next maintenance phase without changing validated numerical behavior or performance. [DONE]
  - 1) Inventory generated versus handwritten HD/MHD/EOS/viscous/resistive/CT equations and report the actual boundary clearly. [DONE]
  - 2) Design and implement the first incremental declarative parameter-registry slice, preserving current keys/defaults/validation and leaving a straightforward migration path for remaining parameters. [DONE]
  - 3) Route all solver/MPI Python execution through one shared runner while preserving intentional direct execution for rank one and supporting both MPI and non-MPI programs. [DONE]
  - 4) Move suitable non-template configuration implementation out of widely included headers when it measurably reduces rebuild scope without obscuring the interface. [DONE]
  - 5) Rebuild, run focused equivalence checks, the standard suite, and performance before closing the cycle. [DONE]
- Deferred by request: portable dependency bootstrap and multipatch physics development. [DONE]

### Cycle 23 results
- The SymPy generator owns ideal HD/MHD flux algebra, LLF/HLL state combinations, orthogonal metric and curvature terms, Newtonian viscous stress and tensor-divergence connection terms, curl/divergence and ideal/resistive EMF algebra, CT face-flux updates, and patch maps. EOS closures, HLLC/HLLD fan construction, reconstruction, timestepping, AMR/reflux orchestration, and staggered AMReX assembly remain handwritten. This boundary is now explicit in `codegen/README.md`. [DONE]
- Added the first method-parameter registry for EOS, physics/CT, reconstruction, Riemann solver, timestep, diffusion, and subcycling selectors. One table now owns each migrated key, default, accepted option set, restart policy, and description; `RunConfig` construction, `ParmParse` queries, and validation consume it. The implementation lives in `src/config.cpp`; a follow-up incremental build rebuilt only `config.cpp` and relinked each executable rather than recompiling solver translation units. [DONE]
- All Python solver execution now goes through `run_mpi`/`run_amrex`. Rank one runs directly, higher rank counts use the canonical MPI launcher, and remaining direct `subprocess.run` calls are non-solver child tools such as CMake, plotting, or suite orchestration. [DONE]
- All 1D/2D/3D solver and checkpoint-comparison targets rebuilt. The default and invalid-option config probes passed, all modified Python entrypoints parse, and the three-rank standard suite passed every gate. [DONE]
- Performance is unchanged against the clean baseline: 2D PWC/LLF/Euler measured 109.81 Mcells/s on 3 ranks and 137.60 on 4 ranks; 3D PWC/LLF/Euler measured 6.36, and 3D WENO-Z/HLLC/RK4 measured 0.19. Every baseline ratio rounds to 1.00. [DONE]

## Cycle 24 [completed]
- Strengthen the symbolic physics layer without creating a second implementation or changing production numerics. [DONE]
  - 1) Inventory duplicated scalar/vector/tensor and differential-operator formulas in the generator. [DONE]
  - 2) Introduce the smallest useful field/operator abstraction for reusable gradient, divergence, curl, Laplacian, strain, stress, and flux composition across coordinate frames. [DONE]
  - 3) Migrate an existing generated physics family onto that abstraction and prove generated C++ equivalence. [DONE]
  - 4) Document the extension contract for new scalar/vector/tensor PDE terms without claiming a complete arbitrary-PDE compiler. [DONE]
  - 5) Regenerate, compile, run algebra checks and focused validation, then check performance if production kernels change. [DONE]

### Cycle 24 results
- Extended `Vector3`/`Tensor3x3` with reusable tensor arithmetic, columns, double contraction, symmetry, and deviatoric operations. Added `OrthogonalOperators` as the single symbolic implementation of scalar gradient/Laplacian, vector gradient, divergence, curl, strain, Newtonian stress, and tensor-divergence connection terms. [DONE]
- Removed the unused speculative `Vector4`/`Tensor4x4` types and the `_as_vector` compatibility shim; current algebra exposes only exercised abstractions and direct constructors. [DONE]
- Replaced direction-expanded hydro/MHD flux construction with one `ConservationFlux3` object containing mass-vector, momentum-tensor, energy-vector, and optional induction-tensor fluxes. Directional generated kernels are projections onto a face normal. [DONE]
- Migrated generated Newtonian stress, geometric connection terms, Cartesian curl, and Cartesian divergence through the shared operator layer. Independent randomized references and new symbolic identities pass. [DONE]
- All six generated production headers are byte-identical before and after the abstraction change, so emitted arithmetic, signatures, runtime behavior, and performance are unchanged. All solver dimensions compile. [DONE]
- Made generated-header writes content-aware and changed CMake codegen tracking to a stamp plus byproducts. Regenerating identical kernels now preserves header timestamps; the following solver build is a no-op instead of recompiling every translation unit. Added the previously missing `tensor_algebra.py` CMake dependency. [DONE]
- The standalone generated C++ check still reports its pre-existing Thornburg patch-map transform mismatch even though `patch_maps.hpp` is byte-identical; the Python symbolic/numeric codegen gate passes. This unrelated multipatch assertion was not changed in this cycle. [DONE]

## Cycle 25 [completed]
- Decide the next symbolic-generation boundary before extending the implementation. [DONE]
  - Keep pointwise algebra such as arithmetic, dot, cross, outer product, and contraction on scalar/vector/tensor value types. [DONE]
  - Keep gradient, divergence, curl, and Laplacian on a geometry/operator object because their meaning depends on the coordinate frame. [DONE]
  - Model each EOS as one constitutive-law object providing pressure, inverse internal energy, and sound speed from a single analytic definition; lower those expressions into generated specialized kernels. [DONE]
  - Treat full PDE-to-kernel generation as a staged lowering pipeline: conservation-law IR, coordinate lowering, explicit numerical-scheme declaration, pointwise kernel generation, then generic AMReX orchestration. Einstein-summation physics alone cannot select reconstruction, Riemann fan logic, time integration, centering, or AMR synchronization. [DONE]
  - Preserve the planned 4-vector/4-tensor capability through shared rank-generic algebra rather than duplicated unused implementations when relativity work begins. [DONE]

## Cycle 26 [completed]
- Refine the symbolic physics architecture before implementation. [DONE]
  - Keep zero-cost pointwise algebra as methods on immutable scalar/vector/tensor expression types; generator-time method calls must disappear into emitted expressions and introduce no runtime abstraction. [DONE]
  - Represent spatial and temporal derivatives explicitly, with geometry-dependent lowering owned by an initialized coordinate/operator context rather than hidden global state. [DONE]
  - Make each EOS a generated constitutive model from one high-level pressure or fundamental-energy definition; derive inverse closure and sound-speed expressions where mathematically determined. [DONE]
  - Let the physical-system model expose composed capabilities such as barotropic and magnetized without duplicating independent flags already owned by its EOS and physics modules. [DONE]
  - Split each PDE into a hyperbolic conservation law, source terms, and dissipative constitutive operators. The viscous model supplies a stress tensor; the conservation-law assembly turns that stress and its power into momentum and energy flux contributions. [DONE]
  - Judge simplification by the number of handwritten mathematical sources of truth, not generated line count; require emitted-kernel equivalence and performance preservation while old duplicate formulas are removed. [DONE]

## Cycle 27 [completed]
- Complete the first coordinate-independent physics-to-AMReX generation loop. [DONE]
  - Inventory current symbolic algebra, derivative operators, EOS formulas, physical-system assembly, and generated-kernel entrypoints. [DONE]
  - Make symbolic scalar/vector/tensor algebra rank-generic where that removes duplicate 3D/4D implementations without changing emitted arithmetic. [DONE]
  - Give spatial and temporal derivatives one explicit high-level representation and one geometry-aware lowering path for every coordinate system and dimension. [DONE]
  - Define analytic barotropic and nonbarotropic EOS models once and generate pressure, inverse internal energy, thermodynamic derivatives, and sound speed from those definitions. [DONE]
  - Compose coordinate-independent conservation laws, sources, and dissipative constitutive tensors at high level, then lower to specialized AMReX kernels. [DONE]
  - Remove superseded handwritten formulas only after generated equivalence, compile, focused validation, and performance checks pass. [DONE]

### Cycle 27 results
- Replaced fixed `Vector3`/`Tensor3x3` generator classes with dimension-carrying `Vector`/`Tensor` algebra. Four-dimensional dot, outer-product, identity, contraction, and deviatoric identities are now covered without duplicate 4D implementations. [DONE]
- Added `ScalarField`/`VectorField` as the single representation of values plus coordinate/time derivatives. `OrthogonalOperators` now lowers gradient, divergence, curl, Laplacian, strain, and connection terms from those fields; Cartesian 1D/2D/3D Laplacians are projections of that same operator. [DONE]
- Added coordinate-independent `ConservationEquation` objects with semantics `d_t U_A + div(F_A) = S_A`. Hydro and MHD flux kernels are projections of composed conservation laws, while `NewtonianViscosity` owns the constitutive stress and geometry owns only differential operations. Existing flux, scalar/vector operator, metric, Riemann, CT, and patch-map generated headers remain byte-identical. [DONE]
- Added analytic EOS models defined by one `P(rho, u)`. Barotropic status is derived, barotropic energy satisfies the first-law identity, nonbarotropic closures are analytically inverted, and sound speeds are generated from thermodynamic derivatives. Runtime ideal-gas, polytrope, and hybrid-polytrope formulas now call the generated EOS header instead of retaining handwritten duplicates. [DONE]
- Consolidated scalar/structured kernel IR and dimension-independent finite-volume stencil objects. Split heavy patch-map generation from ordinary physics generation: a physics/EOS generator edit now regenerates in about 1.9 seconds instead of paying the 6.4-second patch-map cost; unchanged full builds remain 0.14-second no-ops. [DONE]
- All 1D/2D/3D solver and generated-check targets compile. The symbolic/randomized codegen gate passes, including EOS inversion/thermodynamics and 4D algebra. The standalone generated executable retains its pre-existing Thornburg patch-map transform assertion; its generated patch-map header is byte-identical and this unrelated assertion was not changed. [DONE]
- The complete standard AMReX suite passes: AMR, multipatch, solids, dimensional/rank consistency, generated algebra, Cartesian/curvilinear methods and origin, regridding, Sod, sources, gravity, Brio-Wu, all EOS modes, dispersion relations, and diffusion. Focused acoustic EOS runs conserve mass/energy to roundoff. [DONE]
- Valid uncontended performance is unchanged: 2D three-rank PWC/LLF/Euler is 109.80 Mcells/s versus 109.91 baseline, and 3D three-rank PWC/LLF/Euler is 6.37 Mcells/s versus 6.37 baseline. Earlier warning rows were produced while a detached full-benchmark 3D job occupied all three performance cores; that exact leftover job was terminated and no MPI jobs remain. [DONE]

## Cycle 28 [completed]
- Establish the generated-code sprawl rule before further generator extension. [DONE]
  - Accept generated size that represents real branch-free specialization by physics, EOS, coordinate frame, dimension, or numerical method. [DONE]
  - Treat repeated handwritten kernel declarations or repeated symbolic formulas as evidence that a shared model method, lowering rule, or kernel-family emitter is missing. [DONE]
  - Keep generated output split by stable domain ownership and emit only supported specializations; do not optimize for the smallest generated line count at the expense of runtime performance or inspectability. [DONE]
  - Prefer the next cleanup at repetitive generator source, especially EOS operation declarations and common kernel-family emission, while requiring identical generated arithmetic and compile/performance gates. [DONE]

## Cycle 29 [completed]
- Consolidate EOS kernel-family generation and quantify existing CSE/drop-policy use. [DONE]
  - Measure current generated temporaries and output changes under the default, no-drop, and more aggressive drop policies. [DONE]
  - Replace repeated EOS `ScalarKernel` declarations with one declarative operation-family emitter without changing formulas, names, signatures, or emitted arithmetic. [DONE]
  - Regenerate, compile, and run focused symbolic/EOS validation; verify ordinary generator latency remains short. [DONE]
  - Document when hard/soft drop actually changes current generated physics and why defaults preserve repeated expensive expressions. [DONE]

### Cycle 29 results
- `AnalyticEquationOfState.kernels()` now emits the pressure, inverse-energy, and sound-speed operation family for either barotropic or nonbarotropic models. `eos_kernels()` is only a flattening composition over the registered models; nine repeated kernel declarations were removed. [DONE]
- The EOS generator source decreased from 183 to 149 lines while the generated EOS header remained byte-identical, including all names, signatures, expressions, and CSE temporaries. [DONE]
- SymPy CSE is active throughout current generated physics. The production hard/soft drop thresholds `0/3` currently produce byte-identical output to disabling drops because retained temporaries are reused or exceed the cheap-inline threshold. An aggressive `3/8` policy removes most temporaries but may duplicate arithmetic and even enlarge kernels, so production defaults remain unchanged. [DONE]
- All solver targets remain no-op builds after content-identical generation; unchanged full build time is 0.14 seconds. The symbolic/randomized generator gate and focused three-rank ideal-gas, polytrope, and hybrid-polytrope acoustic validation pass. [DONE]

## Cycle 30 [completed]
- Decide the final low-level boundary before returning to AMR robustness. [DONE]
  - Stop general generator refactoring: symbolic algebra, EOS, derivatives, conservation laws, codegen dependencies, compile latency, validation, and performance are sufficiently stable for AMR work. [DONE]
  - Before changing refinement behavior, consolidate AMR parameter defaults/options/documentation/restart policy into the existing configuration registry so AMR choices have one source of truth. [DONE]
  - Add one refinement-decision report rather than more ad-hoc diagnostics: per level and direction, record tag demand, chosen refinement, and whether refinement stopped because the solution was smooth, the physical-length guard fired, a cell budget fired, or maximum level fired. [DONE]
  - Treat removal of the current `refinement_tensor=auto` feature restrictions and dynamic curvilinear AMR equivalence/convergence as the next AMR robustness phase, not as low-level cleanup. [DONE]
  - Preserve the current validated performance and standard-suite result as the pre-AMR baseline; do not mix unrelated multipatch/Thornburg work into this phase. [DONE]

## Cycle 31 [in progress]
- Finish the two narrow prerequisites for robust AMR development. [DONE]
  - Move registered method and AMR defaults, parsing, option declarations, descriptions, and restart-mutability classification into the existing configuration registry. Keep special array-valued geometry parsing outside the registry until it has a natural schema representation. [DONE]
  - Make the executable write one canonical merged AMReX parameter table to each successful run's `run.in`; remove Python-side rewrites from the normal runner and current AMR/curvilinear/multipatch validation paths. [DONE]
  - Add compact refinement-decision reporting to the existing `amr_levels.tsv`: directional tag counts, mean indicator-over-threshold demand, selected ratios, and per-level decisions (`smooth`, `refined`, `physical_length_guard`, `max_level_guard`, or explicit inactive/off states). Cell budgets remain reported safety checks because exceeding them currently aborts rather than gracefully stopping refinement. [DONE]
  - Validate the consolidation with 1D/2D/3D full builds, a focused three-rank AMR run, and the complete standard suite. Closed-run mass and energy remain at about `1e-15`; all standard gates pass. [DONE]
- Then make automatic refinement robust without branch-specific AMR implementations. [PENDING]
  - Remove `refinement_tensor=auto` restrictions incrementally, beginning with single-patch Cartesian hydro and then curvilinear AMR, regridding, and subcycling. [PENDING]
  - Gate behavior with conservation through regrids, causal quiescence, isotropy, Cartesian-curvilinear equivalence, hierarchy consistency, and convergence rather than adding redundant AMR-existence tests. [PENDING]

## Cycle 32 [completed]
- Enforce one ETK-like configuration path: compiled defaults plus one AMReX input file. [DONE]
  - Remove arbitrary `key=value`, `--run-name`, `--output-dir`, `--dim`, and `--executable` mutation from the normal `run_case.py` interface. The input file is the sole user run configuration; runner flags control execution and reporting only. [DONE]
  - Require every checked-in run input to define `amr.n_cell`, `pyfluid.run_name`, and `pyfluid.output_dir`; all 56 current inputs satisfy the contract. [DONE]
  - Make the production executable accept exactly one input file and reject trailing command-line parameter mutations. Keep unused-input checking mandatory so a parfile cannot disable typo detection. [DONE]
  - Add one shared validation helper that materializes a complete run-local `run.in` from a base input plus test-owned parameter values. All PyFluid solver launches under `tools/` now execute with one input path; there is no second runtime parameter interface. [DONE]
  - Preserve AMReX parser semantics inside that contract: supported input values replace registered C++ defaults, while unsupported keys and unsupported combinations abort. [DONE]

### Cycle 32 validation
- Full 1D/2D/3D production builds pass; Python tool compilation, `git diff --check`, generated-kernel freshness, and randomized symbolic flux/Riemann/EOS/operator/vector/metric validation pass. [DONE]
- The runner and all three production binaries reject trailing parameter mutations before evolution. [DONE]
- The post-change standard three-rank suite was not rerun: the MPI execution escalation was rejected because the approval service hit its usage quota. The quick suite passed immediately before the final executable hardening and remaining validator migrations; a fresh standard MPI gate remains the next validation action. [DONE]

## Cycle 33 [completed]
- Complete the deferred post-configuration MPI validation after the local runtime update. [DONE]
  - Record the active MPI/PRRTE versions. [DONE]
  - Run the unchanged standard AMReX suite with the normal three ranks. [DONE]
  - Inspect the generated suite summary and record any failures without weakening gates. [DONE]

### Cycle 33 results
- The rebuilt executables use Open MPI 5.0.9 and PRRTE 4.1.0. The updated toolchain triggered a clean CMake reconfiguration and successful full 1D/2D/3D rebuild. [DONE]
- Corrected mandatory unused-input handling by setting AMReX's compiled `abort_on_unused_inputs` default before `Initialize`; an explicit parfile value of `0` remains rejected. A deliberate unknown-key run aborts and names the unsupported key. [DONE]
- Strict checking exposed stale generated validation parfiles that combined `amr.max_level=0` with inherited `amr.ref_ratio`. Feature-matrix, CT, no-AMR multipatch, solid-volume, pulsating-solid, solid-sphere, dimensional-consistency, curvilinear-origin, and Brio-Wu writers now emit one deduplicated concrete input and explicitly omit the inactive ratio. [DONE]
- Registered `pyfluid.validation_policy` as canonical run metadata with validated choices and removed the Python-only `validation.policy` alias. Source and gravity inputs now pass strict C++ parsing without losing their postprocessing policy. [DONE]
- Final standard suite: 30/30 gates pass in 103.158 seconds. This includes feature combinations, CT, AMR/regridding, multipatch, solids, dimensional/rank consistency, generated kernels, curvilinear geometry/methods/origin/AMR, Sod, sources, gravity, Brio-Wu AMR+CT, EOS, dispersion, and diffusion amplitude. [DONE]

## Cycle 34 [completed]
- Make open-boundary validation causal rather than globally permissive. [DONE]
  - Define one reusable localized-disturbance causal model from run metadata: initial support, static background, conservative signal speed, and first boundary-contact time. [DONE]
  - Require open runs to satisfy the applicable closed scalar budgets before causal contact with an open boundary. [DONE]
  - Generalize the existing spatial quiescence check so state output, when present, must remain unchanged outside the numerical causal cone in any supported dimension. [DONE]
  - Apply the same optional causal check to closed localized-perturbation runs without weakening their normal conservation policy. [DONE]
  - Add focused tests and run the quick confidence gate. [DONE]

### Cycle 34 results
- Recognized single-patch, source-free, nondiffusive ideal-gas pressure bumps now carry an inferred numerical causal model. Open/diode runs enforce mass, energy, and momentum drift below `1e-12` through the last scalar sample before first causal boundary contact; unsupported/nonlocal setups retain their existing policy rather than using guessed metadata. [DONE]
- Causal quiescence now reads 1D/2D/3D state output and checks density, pressure, velocity, and magnetic field outside the signal cone in Cartesian, cylindrical, or spherical coordinates. Closed localized runs receive the same optional spatial check. [DONE]
- The focused three-rank open AMR pulse gate passes with 12 pre-contact rows: mass drift `8.44e-15`, energy drift `4.00e-15`, momentum drift `2.10e-19`, and exterior change `2.22e-8`. The spatial tolerance is `1e-7` because finite reconstruction/RK stencils have small tails outside the continuum cone; Gaussian support is conservatively treated as eight sigma. [DONE]
- Python compilation and `git diff --check` pass. The complete quick AMReX gate passes, including feature combinations, CT, AMR/refinement tensors/regridding, Cartesian multipatch, dimensional consistency, curvilinear geometry/methods/origin/regridding, Sod, sources, and gravity. [DONE]

## Cycle 35 [completed]
- Stop using uniform grids as the implicit validation default. [DONE]
  - Classify every checked-in and suite-generated `amr.max_level=0` use as a necessary uniform control, a geometry/unit smoke, a benchmark baseline, or avoidable AMR bypass. [DONE]
  - Keep fixed-grid controls for convergence, exact dimensional comparisons, CFL scans, and isolated topology checks, but pair physics branches with automatic multilevel AMR coverage. [DONE]
  - Move representative HD, MHD+CT, curvilinear, diffusion/resistivity, source, gravity, and production cases to automatic AMR with regridding, subcycling where supported, refinement headroom, and scalar/hierarchy gates. [DONE]
  - Add a suite-level coverage check so future standard/full suites cannot claim broad feature coverage while silently exercising only level zero. [DONE]
  - Run focused AMR cases and the standard confidence gate; preserve performance baselines as explicitly uniform controls. [DONE]

### Cycle 35 findings and active fixes
- Automatic tagging was evaluating a neighbor stencil after applying only physical/same-level boundary conditions. It now uses the hierarchy FillPatch path so coarse-fine ghosts are current before every AMR indicator evaluation. [DONE]
- The automatic tag threshold no longer changes when `amr.max_level` changes; the maximum level remains a resource ceiling rather than an implicit refinement-policy knob. [DONE]
- Representative feature cases now require both active refinement and at least two unused levels of headroom. Hydro, MHD+CT, explicit resistivity, explicit/implicit viscosity, sources, self-gravity, local-CFL subcycling, and broad MHD/solid combinations pass focused multilevel runs. [DONE]
- Focused implicit-viscosity AMR testing exposed two hierarchy omissions: the level solve discarded FillPatched coarse-fine scratch ghosts, and the converged implicit viscous face flux never entered the AMR flux registers. The solve now preserves coarse-fine boundary data, refreshes each fine solve after its coarse level advances, and refluxes the same physical face flux used by explicit viscosity. Momentum drift fell from `3.77e-8` to `2.49e-16` with subcycling; the non-subcycled AMR case also passes. [DONE]
- The automatic pressure-bump hierarchy now reaches L4 with four levels of headroom and stops on smoothness using 3,904 cells; before the FillPatch fix it ran to L6 and 334,656 cells before the physical-length guard stopped it. [DONE]
- Full 1D/2D/3D builds pass. The standard three-rank suite passes all 31 gates, including causal AMR quiescence, dynamic/curvilinear regridding, seeded Cartesian-curvilinear AMR equivalence, multipatch, Brio-Wu AMR+CT, diffusion, sources, and gravity. [DONE]
- The canonical uniform 2D three-rank benchmark is `107.23 Mcells/s`, 98% of the `109.91 Mcells/s` baseline; the AMR hierarchy fixes do not materially tax the level-zero path. [DONE]

## Cycle 36 [completed]
- Define successful production AMR by metric-limited refinement rather than resource-limited refinement. [DONE]
  - Document that `max_level`, `min_feature_length`, radial guards, and cell budgets are emergency resource limits; touching one is an under-resolution/failure signal unless a test explicitly targets that guard. [DONE]
  - Start production-style tests from a deliberately coarse base hierarchy and require automatic AMR to create and move resolution according to the evolving state. [DONE]
  - Replace the fixed-L2 smooth pressure-bump improvement check with an automatic-AMR spatial sequence that varies the requested error threshold, retains headroom, stops on smoothness, and shows monotonically improving error against the uniform reference. [DONE]
  - Keep fixed-level AMR only as an isolated hierarchy/reflux control and keep the discontinuous Brio-Wu reference as a separate shock-resolution test. [DONE]

### Cycle 36 current state
- The production-AMR success rule is documented in `AGENTS.md`, the AMReX README, and diagnostics documentation. [DONE]
- The pressure AMR validator now defines loose/medium/tight automatic thresholds from the same deliberately coarse 32x32 base grid, with `max_level=12`, SSPRK3, WENO-Z/HLLC, subcycling, and resource guards far below expected use. Every automatic case requires active refinement, at least two levels of headroom, budget fraction below 0.8, and a terminal `smooth` decision. [DONE]
- The sequence compares all three composite solutions against one uniform 512x512 reference and requires error to improve monotonically as the metric threshold tightens. A matched medium-threshold no-subcycling case retains a coarser algorithm-consistency check. [DONE]
- Three-rank results are metric limited: threshold 0.16 reaches L2 with 9,216 cells and pressure L2 error `3.17e-3`; 0.08 reaches L3 with 18,432 cells and `2.10e-3`; 0.02 reaches L4 with 52,480 cells and `1.11e-3`. All retain 8-10 levels of headroom, use 0.5-2.6% of the cell budget, stop on `smooth`, and conserve energy near `1e-16`. Error improvement factors are 1.51 and 1.89. [DONE]
- This is an automatic error-control convergence test, not a claim that error-versus-finest-dx alone recovers the uniform-grid WENO formal order: adaptive coverage changes with the threshold. Uniform method scans remain the formal spatial-order gate. [DONE]
- Python compilation and `git diff --check` pass, and the focused production MPI convergence sequence passes. [DONE]

## Cycle 37 [in progress]
- Checkpoint the validated single-patch AMReX infrastructure, then stress dynamic curvilinear AMR. [WIP]
  - Commit the current AMReX and repository-agent state without including unrelated JAX/Numba/reference-tree changes. [WIP]
  - Run a long acoustic packet through the cylindrical/spherical origin with automatic anisotropic AMR, SSPRK3, and subcycling; compare transmission, symmetry, conservation, causal exterior behavior, and profile error against Cartesian evolution. [PENDING]
  - Run a long rotating two-hot-spot source with open radial boundary and automatic anisotropic AMR; require refinement to follow the emitted wave to the boundary, no pre-contact exterior change, smooth metric termination, positive headroom, and honest hierarchy diagnostics. [PENDING]
  - Use these runs to decide whether single-patch hydro is complete and identify the remaining MHD+CT+viscous/implicit stress matrix without weakening existing gates. [PENDING]
