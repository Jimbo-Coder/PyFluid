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

## Cycle 31 [completed]
- Finish the two narrow prerequisites for robust AMR development. [DONE]
  - Move registered method and AMR defaults, parsing, option declarations, descriptions, and restart-mutability classification into the existing configuration registry. Keep special array-valued geometry parsing outside the registry until it has a natural schema representation. [DONE]
  - Make the executable write one canonical merged AMReX parameter table to each successful run's `run.in`; remove Python-side rewrites from the normal runner and current AMR/curvilinear/multipatch validation paths. [DONE]
  - Add compact refinement-decision reporting to the existing `amr_levels.tsv`: directional tag counts, mean indicator-over-threshold demand, selected ratios, and per-level decisions (`smooth`, `refined`, `physical_length_guard`, `max_level_guard`, or explicit inactive/off states). Cell budgets remain reported safety checks because exceeding them currently aborts rather than gracefully stopping refinement. [DONE]
  - Validate the consolidation with 1D/2D/3D full builds, a focused three-rank AMR run, and the complete standard suite. Closed-run mass and energy remain at about `1e-15`; all standard gates pass. [DONE]
- The proposed dynamic `refinement_tensor=auto` phase was explicitly resolved
  by Cycles 39--40: AMReX refinement ratios remain fixed hierarchy-construction
  data, while automatic tagging/reboxing remains supported. The unsupported
  ratio-mutation experiment and its dedicated tests were removed rather than
  hidden behind a runtime guard. [DONE]

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

## Cycle 37 [completed]
- Checkpoint the validated single-patch AMReX infrastructure, then stress dynamic curvilinear AMR. [DONE]
  - Commit the current AMReX and repository-agent state without including unrelated JAX/Numba/reference-tree changes. Commit `b366fb7`. [DONE]
  - Run a long acoustic packet through the cylindrical/spherical origin with automatic anisotropic AMR, SSPRK3, and subcycling; compare transmission, symmetry, conservation, causal exterior behavior, and profile error against Cartesian evolution. [DONE]
    - Cartesian automatic anisotropic AMR with subcycling reaches L2 and passes scalar/causal checks. Cylindrical `r phi` reaches L2 but produces active-cell NaNs during the first subcycled step. Isolate fixed-vs-auto refinement and subcycling before changing the solver. [WIP]
    - Removed the duplicated non-subcycling PWC FillPatch path and nonfinite neighbor-copy repair. One configured AMReX interpolater now serves level creation, remake, FillPatch, subcycling, and repair seeds. Conservative anisotropic FillPatch remains finite. [DONE]
    - Long dynamic anisotropic regridding changes L2 from `(2,2)` to `(2,1)` near `t=0.118` and AMReX rejects the resulting proper-nesting transaction. A full conservative hierarchy rebuild avoids the abort but spuriously cascades to L12, so that experiment was reverted. Cycle 40 resolved the design by removing runtime ratio mutation and retaining fixed ratios with dynamic tagging/reboxing. [DONE]
    - Runtime curvilinear momentum diagnostics now rotate local basis components into physical Cartesian momentum through one shared coordinate transform. [DONE]
  - The long rotating-source demonstration moved to Cycle 48 and completed
    there with its finite-stencil causal-cone limitation recorded. [DONE]
  - Later curvilinear, MHD+CT, diffusion, dimensional, and AMR gates supersede
    this historical coverage-planning bullet. [DONE]

## Cycle 38 [completed]
- Make the no-fallback development policy durable for future maintainers. [DONE]
  - Keep implementation conventions in `docs/code_conventions.md` and link them from the repository agent instructions. [DONE]
  - State explicitly that a bug in an intended branch does not make the branch unsupported: record the reproducer in this cycle log and fix the shared operation instead of adding a temporary startup guard, bypass, alias, or replacement algorithm. [DONE]
  - Confirm the failed dynamic anisotropic hierarchy-rebuild experiment is reverted to the bounded transaction; retain the unresolved ratio-change failure as Cycle 37 work rather than concealing it. [DONE]

## Cycle 39 [completed]
- Clarify the current anisotropic AMR and subcycling support boundary. [DONE]
  - Static anisotropic refinement and Cartesian automatic anisotropic AMR with subcycling have passed focused checks. [DONE]
  - The unresolved failure is a dynamic refinement-ratio change in an existing multilevel hierarchy: the long cylindrical case changes L2 from `(2,2)` to `(2,1)`, after which AMReX rejects the proper-nesting transaction. [DONE]
  - Keep the failure recorded and use fixed isotropic ratios for long production physics runs until hierarchy migration across dynamic anisotropic ratios is implemented and validated. [DONE]

## Cycle 40 [completed]
- Determine whether fully automatic anisotropic AMR can be made robust with subcycling inside the standard AMReX hierarchy contract. [DONE]
  - Reduce the long cylindrical `(2,2)` to `(2,1)` failure to the smallest dynamic-ratio reproducer and locate the exact AMReX proper-nesting invariant being violated. [DONE]
    - AMReX treats `ref_ratio` as hierarchy-construction data and exposes no supported runtime setter; its header explicitly notes that changing it requires rebuilding Geometry objects. PyFluid's in-place mutation leaves `MakeNewGrids` with a declared fine level whose new `BoxArray` is empty, triggering `AmrMesh::MakeNewGrids: how did this happen?`. [DONE]
    - This is not specifically a subcycling defect. Subcycling adds time-factor and synchronization state that a migration must also rebuild, but dynamic ratio mutation is outside the normal AMReX regrid contract even without subcycling. [DONE]
  - Do not add a second hierarchy-migration manager. Remove runtime tensor mutation and retain AMReX-owned dynamic tagging/reboxing with configured fixed ratios. [DONE]
  - Dynamic ratio-change validation is no longer applicable because the unsupported mode and its dedicated tests were removed rather than hidden behind a guard. [DONE]
  - Run the quick suite and performance benchmark after the shared hierarchy path is stable. [DONE]

## Cycle 41 [completed]
- Distinguish automatic tagging, anisotropic refinement tensors, and time subcycling. [DONE]
  - Production automatic isotropic AMR already uses `amr.tag_metric=auto` with `amr.refinement_tensor=fixed`; it dynamically creates, removes, and reboxes levels while keeping configured spatial ratios fixed. [DONE]
  - That production path supports both `pyfluid.subcycling=off` and `on`. Subcycling changes level timesteps, not the tagging metric or spatial tensor. [DONE]
  - Static anisotropic tensors can likewise subcycle. The unresolved experiment is specifically `amr.refinement_tensor=auto` changing the tensor of an already-existing hierarchy; omitting subcycling does not make that unsupported AMReX mutation correct. [DONE]

## Cycle 42 [completed]
- State the production support boundary for automatic anisotropic refinement. [DONE]
  - Runtime mutation of an existing AMReX hierarchy's directional refinement ratios is not supported. [DONE]
  - Automatic state-based tagging with fixed isotropic or fixed anisotropic ratios remains supported with subcycling on or off. [DONE]

## Cycle 43 [completed]
- Return production refinement to the standard AMReX AMR contract. [DONE]
  - Remove the incomplete runtime `refinement_tensor=auto` mutation path and its branch-specific hierarchy reconstruction. [DONE]
  - Preserve automatic physical-state tagging, dynamic level creation/removal/reboxing, directional indicator diagnostics, standard fixed isotropic refinement ratios, reflux, average-down, and subcycling. [DONE]
  - Remove experimental tensor-only configuration/tests/documentation so the interface does not imply unsupported directional adaptivity. [DONE]
  - Unify state transfer under the selected AMReX interpolater. The later Cycle 44 audit found and corrected swapped quartic operator labels; conservative quartic remains the production default. [DONE]
  - Full 1D/2D/3D builds and the complete quick suite pass. The canonical benchmark remains at baseline: 2D three-rank `109.47 Mcells/s` versus `109.91`, 3D PWC `6.36` versus `6.37`, and 3D WENO-Z `0.19` versus `0.19`. [DONE]

## Cycle 44 [completed]
- Explain and isolate the higher-order AMR state-transfer conservation failure. [DONE]
  - PyFluid's quartic labels were reversed: `amrex::quartic_interp` is `CellConservativeQuartic`, while `amrex::cell_quartic_interp` is the non-conservative `CellQuartic`. `quartic-conservative` had therefore selected the wrong AMReX object. [DONE]
  - Correct the mapping so `quartic` selects `cell_quartic_interp` and `quartic-conservative` selects `quartic_interp`; retain conservative quartic as the higher-order production default. [DONE]
  - The corrected moving-cloud case conserves mass to `6.65e-16`, energy to `1.27e-15`, and momentum to `1.78e-15`, while the level-1 centroid moves from x=0.28125 to x=0.46875. [DONE]
  - Full 1D/2D/3D builds and the complete quick suite pass with the corrected mapping. [DONE]

## Cycle 45 [completed]
- Record the supported AMR state-transfer choices after correcting their labels. [DONE]
  - Conservative choices are linear (`conservative`), protected linear (`protected`, 2D/3D only), and conservative quartic (`quartic-conservative`, the production default). [DONE]
  - `quadratic` and `quartic` select AMReX's non-conservative interpolation operators; no conservative method above quartic is currently exposed. [DONE]

## Cycle 46 [completed]
- Quantify support, accuracy, conservation, and cost of conservative AMR state transfer. [DONE]
  - Make support boundaries explicit before AMReX interpolation: protected conservative is 2D/3D only, and conservative quartic requires refinement ratio 2 in every active direction. Linear conservative accepts the normal supported dimensions and integer ratios. [DONE]
  - Extend the existing moving-cloud regrid validator with a selectable conservative interpolater, analytic density L2 error, and longer-run control rather than adding another test tool. [DONE]
  - At 80 steps all three methods conserve and differ by only about 0.1% in density L2 error. At 800 steps: linear gives `2.93544e-3` at `5.79 Mcells/s`, protected gives `2.92532e-3` at `5.80 Mcells/s`, and conservative quartic gives `2.96146e-3` at `5.77 Mcells/s`. [DONE]
  - This test shows no meaningful transfer-cost difference and no quartic accuracy advantage for moving-cloud advection. Quartic retains a theoretical smooth-prolongation advantage but needs a dedicated smooth comparison before it is preferred on empirical grounds. [DONE]
  - Full 1D/2D/3D builds pass after adding the support checks; Python compilation and `git diff --check` pass. [DONE]

## Cycle 47 [completed]
- Clarify the intended AMR transfer regime. [DONE]
  - Conservative quartic targets smooth high-order prolongation; it is not monotonicity preserving and may overshoot near steep gradients or discontinuities. [DONE]
  - Protected conservative interpolation is the appropriate transfer choice when suppressing coarse-fine overshoot is more important than smooth-region polynomial order. [DONE]

## Cycle 48 [completed]
- Produce two production-style curvilinear AMR demonstrations. [DONE]
  - Configure a cylindrical `r phi` acoustic/density packet whose first radial center is `dr/2`, crosses the origin without an inner hole, and continues outward with automatic state tagging, SSPRK3, subcycling, protected conservative transfer, and metric-limited refinement. The first attempt correctly refined to L7 but exposed excessive origin CFL cost from an over-resolved angular base; rerun with coarser angular sampling and automatic AMR retained. [DONE]
  - Run long enough to validate origin transmission, conservation, causal behavior, hierarchy movement, level headroom, and AMR efficiency; render honest pressure-perturbation movies with and without boxes. The run passes causal quiescence at `8.85e-12`, stops smoothly at L2 with eight levels of headroom and 0.55% budget use, and shows the inward lobe crossing and re-emerging from `r=0`. [DONE]
  - Configure and run a rotating two-hot-spot source with the same standard AMR path and open outer radial boundary; require refinement to follow emitted waves through the domain without pre-causal artifacts. [DONE]
    - Replace the separable radial/angular ring profile outright with rotating physical-space compact disks. The source is now `rotating_compact_heating`, uses one `source_sigma`, and has exact support at three sigma. The final 3-rank run reaches L2 with eight levels of headroom and remains far below budget. [DONE]
    - The strict causal-exterior validator measures `1.67e-6` outside a `1.2` times background-acoustic cone, above its `1e-7` tolerance; it passes with a factor-two cone. The demonstration remains recorded with that limitation and is not claimed strictly causal-clean. [DONE]
  - Render density movies with and without boxes, retain run-local scalar/hierarchy reports, and report exact output paths. [DONE]

## Cycle 49 [completed]
- Evaluate protected conservative interpolation as the default during curvilinear movie work. [DONE]
  - Do not make it the universal default yet: AMReX supports protected interpolation only in 2D/3D, and silently selecting another 1D method would violate the one-method contract. [DONE]
  - Use protected interpolation explicitly for the two requested 2D stress runs, where monotonic regridding is more valuable than smooth quartic prolongation. [DONE]
  - Retain conservative quartic as the registered default pending a deliberate universal-default decision or validated 1D protected implementation. [DONE]

## Cycle 50 [completed]
- Replace the origin pressure-bump demonstration with a controlled one-way acoustic characteristic packet and audit AMR coverage. [DONE]
  - Keep coordinate families explicit: Cartesian, cylindrical, and spherical, with lower-dimensional ordered slices of those families; do not introduce a separate generic curvilinear runtime family. [DONE]
  - Add one coordinate-independent physical-space `acoustic_packet_x` initial-data definition. Initialize the linear right-going acoustic eigenvector in Cartesian physical space and project velocity into each mesh basis. [DONE]
  - Evolve matching Cartesian and cylindrical `r phi` cases, compare the translated packet in a causally uncontaminated central region, and quantify numerical damping/error. [DONE]
  - Measure whether all above-threshold cells and the strongest-gradient subset are covered by the next AMR level after regridding; distinguish tagging, clustering, cadence, and visualization issues without adding a budget fallback or flow-specific AMR rule. [DONE]
  - Render honest cylindrical density-perturbation movies with and without boxes only after the scalar, pointwise-equivalence, origin-transmission, and AMR-coverage checks pass. [DONE]
  - Replace the planar packet with one physical-space right-going acoustic eigenvector modulated by an explicit transverse Gaussian envelope, so the polar outer boundary is causally separated from the origin test. [DONE]
  - Replace the old level-0 piecewise-constant antipodal fallback with a canonical full-angular origin shell sampled by AMReX `FillPatchNLevels`; this supports sparse/non-nested fine layouts and preserves the selected hierarchy interpolater. [DONE]
  - Add a debug-only reconstruction-input contract covering required ghosts. It exposed wrong-FAB/component reads and zero-density corner ghosts during development; the corrected origin shell passes it throughout SSPRK3 subcycling. [DONE]
  - The coarse-base `16 x 8` cylindrical run crosses the packet center through the origin by `t=0.26`, remains metric-limited at L2 with eight levels of headroom and 0.24% budget use, and covers 100% of measured packet-gradient samples at every output. Cartesian remains L1 because its base physical spacing differs. [DONE]
  - Record the remaining limitation honestly: tighter tolerance (`0.10` or below) drives isotropic polar refinement to L7-L8 and collapses origin CFL even with large level/budget headroom. This is not dropped tagging; robust automatic directional refinement is not supported by the standard fixed-ratio AMReX hierarchy. [DONE]
  - Rebuild all dimensions, run the quick suite, rerun the validated packet after removing diagnostic copies, and add packet damping/profile postprocessing. [DONE]
  - The final `16 x 16` base run uses a two-cell tag buffer, PWL/HLLC/SSPRK3, protected conservative transfer, and subcycling. Every sampled packet-gradient strip is refined above base at every output. The hierarchy reaches L3 only during the origin crossing, then removes those levels; it retains at least seven levels of headroom and uses at most 0.17% of the configured cell budget. [DONE]
  - Correct the coverage diagnostic: a packet region needs coverage by the next finer level, not necessarily by the globally finest level when unrelated stronger structure receives deeper refinement. Report both fractions but gate the former. [DONE]
  - The final centerline comparison gives cylindrical-versus-Cartesian density L2 `9.67e-5` and peak-amplitude ratio `1.0235`. Both retain about 55% of the initial centerline amplitude because the transversely localized 2D packet diffracts; the ideal rigid 1D translation is context, not the exact 2D solution. [DONE]
  - Broad refined annuli on the deliberately coarse polar base are explained by the tag buffer, periodic angular dilation, minimum box granularity, and fixed isotropic AMReX ratios. This is not dropped tagging. A two-cell buffer is the smallest general choice consistent with the PWL stencil and two-step regrid cadence. [DONE]

## Cycle 51 [completed]
- Restore physical conservation during curvilinear AMR regridding. [DONE]
  - A closed-wall packet control conserves mass/energy at roundoff on one cylindrical level, but dynamic AMR produces stepwise mass/energy changes only when the fine layout changes. Subcycling off still reproduces the defect, isolating it to mapped coarse/fine remapping rather than fluxes, origin evolution, reflux, or level-time synchronization. [DONE]
  - Replace Cartesian arithmetic conservation during level creation/remake with a local physical-volume-weighted conservative prolongation projection. Preserve the selected interpolater's reconstructed shape while making every refined coarse-cell average equal its physical-volume-weighted parent value; do not add a global after-step scalar correction. [DONE]
    - The projection is implemented for curvilinear and mapped-patch level creation/remake and is a no-op for Cartesian single-patch evolution. Full 1D/2D/3D compilation passes. [DONE]
  - Rebuild all dimensions, rerun the closed-wall packet with subcycling off/on, then rerun the quick suite. Cylindrical mass/energy drift is `6.4e-15`/`6.6e-15` without subcycling and `6.2e-15`/`6.0e-15` with subcycling; the complete quick suite passes. [DONE]

## Cycle 52 [completed]
- Add a narrow curvilinear subcycling equivalence gate and deliver the acoustic-packet origin-crossing movie. [DONE]
  - Validate the pending physical-volume-conservative curvilinear remap first. [DONE]
  - Compare subcycling off/on on the same short fixed cylindrical hierarchy at a synchronized endpoint; require direct composite-GF agreement rather than duplicating broad existing subcycling tests. [DONE]
    - The no-subcycling run takes 52 synchronized level steps; spatial subcycling takes 26 coarse and 52 fine steps. Final pressure and density relative L2 differences are `5.40e-9` and `3.23e-9`. Nontrivial SSPRK3 subcycling is truncation-equivalent, not bitwise identical. [DONE]
    - Add the focused mode to `validate_subcycling_benefit.py` and the normal quick/standard/full suite as `curvilinear_subcycling_equivalence`; the suite-dispatched gate passes in 1.24 s. [DONE]
  - Render the validated packet crossing the origin with AMR boxes following the state, plus a no-box view. Use blocking factor 4 on the deliberately coarse `16 x 16` base so clustering can follow localized sectors; the run remains metric-limited and conservative. [DONE]
    - A `16 x 32` angular-base probe reaches L7 and costs 211 s because polar-origin CFL shrinks rapidly. It offers poor cost/benefit and is not the selected demonstration. [DONE]
  - Report the rotating quadrupole-source status and the exact remaining issue. The previous Gaussian source had nonzero tails at the open boundary at `t=0`, invalidating causal outflow accounting. The shared source is now compactly tapered from `2.5 sigma` to `3 sigma`; its final long 3-rank rerun remains separate Cycle 48 work. [DONE]

## Cycle 53 [completed]
- Tighten the curvilinear packet and rotating-source demonstrations. [DONE]
  - Replace the separable ring/angle heating profile with genuinely compact circular sources defined in physical Cartesian space and rotating antipodally. Remove the misleading ring source name and angular-width parameter rather than retaining aliases. [DONE]
  - Write and render the solver-authored `t=0` acoustic-packet state and hierarchy so its initial direction, resolution, and AMR coverage can be inspected honestly. The demonstrated packet starts at L2, places every sampled gradient-strip point on L2, and resolves a peak `rho-1=9.84e-3` for requested amplitude `1e-2`. [DONE]
  - Add a general metric-weighted AMR coverage score using the existing physical-state tag metric, physical cell volumes, and actual composite hierarchy levels; do not make passive tracers drive the score. Bound diagnostic sampling independently of the finest logical grid so a failed deep hierarchy cannot exhaust memory during postprocessing. [DONE]
  - Clarify packet semantics in the implementation: positive `acoustic_transverse_sigma` is a localized 2D packet that diffracts; zero selects a planar `+x` eigenpacket. The planar cylindrical probe currently runs to L10 at initialization and is therefore not a valid metric-limited production case. [DONE]
  - Rebuild 1D/2D/3D and run the normal quick suite. All builds and every quick gate pass. [DONE]

## Cycle 54 [completed]
- Replace the diffracting acoustic-origin demonstration with a rigidly translating Gaussian entropy packet. [DONE]
  - Reuse the coordinate-independent `density_cloud` IC with constant pressure and uniform physical `+x` velocity; do not add a duplicate packet initializer. [DONE]
  - Extend the existing Cartesian/cylindrical equivalence validator with the exact translated Gaussian profile and AMR-interest diagnostics. [DONE]
  - Fix the shared coarse-fine state-transfer contract: a forced cylindrical fine level preserves valid uniform Cartesian velocity but componentwise interpolation of local-basis momentum creates false gradients in fine ghosts (`max amr_tag_metric=0.804`). Route every selected AMReX state interpolater through one mesh-basis-aware wrapper that interpolates conserved vectors in Cartesian physical components. Apply the same representation to metric-conservative prolongation and average-down; the final forced-level metric is `6.1e-12` on fine cells and `2.3e-14` on coarse cells. [DONE]
  - A one-level uniform physical `+x` control accumulates momentum error `1.35e-3` by `t=0.02`, proving the evolved AMR runaway is a geometric finite-volume defect rather than tagging. For curvilinear layouts that span physical space, use one mapped finite-volume formulation: reconstruct Cartesian physical vectors, solve along physical face normals, use closing face-area vectors, conservatively diverge the physical flux, and project the momentum RHS back to the stored mesh basis. The corrected one-level momentum drift is `2.0e-17`. [DONE]
  - Make weighted reflux use the same closing face-area measure and project its Cartesian momentum correction into the coarse cell basis. The automatic AMR packet run improves from mass/energy drift `1.6e-4`/`3.1e-4` to `1.8e-14`/`1.7e-14`. [DONE]
  - Promote the existing curvilinear regrid freestream gate from a static zero-velocity state to a uniform physical Cartesian `+x` flow represented by the shared `density_cloud` IC with zero amplitude. [DONE]
    - The evolved state and scalar budgets remain at roundoff, but the first promoted-gate run exposed a postprocessing error: causal quiescence compared stored local `r/phi` velocity components directly with the physical Cartesian IC velocity. Compare velocity in physical Cartesian components without changing the solver path. The corrected three-level regrid gate measures `1.35e-14` maximum physical-state change. [DONE]
    - Keep MHD/CT on its original AMReX state interpolater: wrapping an already supported MHD transfer path raised the cylindrical AMR+CT energy drift from `1.98e-12` to `2.22e-8`. The exact original pointer restores the historical result while the basis-aware wrapper remains confined to mapped curvilinear hydro. [DONE]
  - Start from a coarse cylindrical `r phi` base, require metric-limited AMR with headroom, evolve the packet through `r=0` using SSPRK3 and subcycling, and quantify the origin error against Cartesian and exact solutions. [DONE]
    - The packet center translates from `x=-0.25` to the exact `x=+0.25` at `t=0.5`. Cylindrical density L2 error against the exact translated Gaussian is `2.05e-3`, versus `2.39e-3` for the Cartesian control; cylindrical-versus-Cartesian L2 is `1.83e-3`. Pressure remains constant within `5.2e-14`. [DONE]
    - Cylindrical mass, energy, and momentum drift are `1.81e-14`, `1.67e-14`, and `4.62e-14`. Metric-weighted AMR coverage remains effectively one, the hierarchy has eight levels of final headroom, and only 0.205% of the configured cell budget is used. [DONE]
    - Record the remaining limitation honestly: the selected 16-sector base visibly facets the packet, while a 32-sector probe drives isotropic polar refinement to L8 before the crossing. Deep origin refinement remains unnecessarily expensive even though the selected metric-limited run is conservative and stable. [DONE]
  - Render only honest curved physical-space density movies and frames, with and without AMR boxes. [DONE]
  - Rebuild 1D/2D/3D and run the normal quick suite. Every quick gate passes, including cylindrical/spherical methods, curvilinear origin, subcycling equivalence, and the promoted regrid freestream gate. [DONE]

## Cycle 55 [completed]
- Audit why the translating packet looks only partly refined and why the rotating compact-heating demonstration is visibly inadequate. [DONE]
  - The entropy packet is a valid but modest AMR demonstration: its `sigma=0.08` profile is smooth at L2 away from the origin and briefly requires L3 while crossing it. Metric-weighted coverage is effectively one, so demanding additional levels without tightening the physical error target would over-refine rather than improve the stated tolerance. [DONE]
  - The rotating-source movie is not a valid production demonstration. It contains only seven frames over `t=0.8`; its `sigma=0.06` source is represented by about 1.2 radial base cells and fewer than one useful angular cell on the `24 x 16` base; and the explicitly selected `tag_threshold=0.30` leaves only a handful of tags. The hierarchy reaches L2 briefly and then returns to L1. [DONE]
  - Identify the structural curvilinear limitation separately from the poor run setup: standard AMReX refinement doubles both `r` and `phi`. Tightening the state threshold can therefore drive deep isotropic polar refinement and a severe origin CFL cost even when the emitted structure is primarily radial. [DONE]
  - Durable direction: retain one state-based error metric for evolved waves, derive the error buffer from stencil width/regrid cadence rather than per-run tuning, and consider a generic RHS/source truncation indicator so compact physical forcing is resolved before its coarse-grid imprint is smeared. Do not add a rotating-source-specific tagging rule. [DONE]

## Cycle 56 [completed]
- Replace flow-tuned AMR thresholds with a flow-independent sampling/accuracy contract and determine the honest path for curvilinear anisotropy. [DONE]
  - Verify AMReX's supported refinement-ratio and box-clustering model, especially whether local automatic direction-by-direction refinement can be represented within one hierarchy. AMReX accepts an `IntVect` refinement ratio per level, but every region on that level shares it; one hierarchy level cannot independently choose radial refinement in one region and angular refinement in another. [DONE]
  - Assess a recursive initial-grid builder that samples arbitrary ICs and active sources before deposition, refines until a universal cells-per-feature criterion is met, and reports failure when the cell budget cannot satisfy it. This can use AMReX's existing grid generation; the missing piece is a direct fine-probe/representation-error tagger that cannot miss subcell IC or source structure. [DONE]
  - Replace the user-facing Lohner threshold with one global accuracy contract, such as target cells per inferred local feature or normalized high/low-order representation error. Some accuracy target is mathematically unavoidable, but it must be dimensionless, amplitude-independent, and shared across flows rather than tuned per IC. [DONE]
  - Distinguish improvements possible within AMReX from a hierarchy/backend limitation that would require multipatch, multiple hierarchies, or another AMR data structure. Recommended order: implement universal sampling/error tags and automatic initial hierarchy first; use a Cartesian center plus curved outer patches to remove the polar singularity; then benchmark whether isotropic outer-patch AMR is acceptable before considering a custom anisotropic block hierarchy. [DONE]

## Cycle 57 [completed]
- Reassess whether cells-per-feature is the right robust estimator, whether standard AMReX can provide fully automatic local anisotropy, and whether PyFluid is using AMReX's intended responsibilities. [DONE]
  - Do not select cells-per-feature by assertion. Compare the existing Lohner estimator, a local Richardson/shadow-refinement estimator, and a multiresolution reconstruction-detail estimator at equal active-cell budgets on smooth, discontinuous, radial, and nonsymmetric flows; choose by solution error and cost. [DONE]
  - Standard `AmrCore` supports a direction-dependent `IntVect` refinement ratio per level, but not independently selected refinement directions per region on the same level. Rectangular clustering changes patch shape, not the cell refinement ratio. Fully automatic local anisotropy therefore requires a nonstandard/custom hierarchy rather than another tagging rule. [DONE]
  - PyFluid uses AMReX for its intended hierarchy mechanics: tag clustering, regridding, proper nesting, FillPatch, reflux, average-down, distribution, and subcycling. The application is still responsible for defining meaningful error tags; AMReX does not supply a universal PDE error estimator. [DONE]
  - Defer the automatic initial-grid builder until the estimator comparison identifies one shared accuracy contract, then use that same estimator for direct IC/source sampling and evolved-state adaptation. [DONE]

## Cycle 58 [completed]
- Establish common AMR error-estimation practice and scope the work required for truly local anisotropic refinement on AMReX. [DONE]
  - Distinguish common production tagging practice from more rigorous Richardson, truncation-error, and multiresolution estimators. Richardson-type truncation estimates are foundational Berger-Oliger AMR, but production AMReX applications commonly use cheaper state/gradient/problem indicators; multiresolution detail is a practical intermediate estimator with published error-per-cell advantages over scaled gradients. [DONE]
  - Inventory which `AmrCore` assumptions would have to change for different regions at one effective resolution to choose different refinement directions. `AmrCore` is a chain of levels, each with one `Geometry`, spacing vector, and parent refinement `IntVect`. Local anisotropy creates a partial-order patch hierarchy and requires new neighbor/connectivity, FillPatch, restriction/prolongation, reflux, subcycling, regrid, restart, and CT edge-reflux logic. It is possible using AMReX boxes/MultiFabs as storage, but is effectively a new AMR backend rather than a small `AmrCore` extension. [DONE]
  - Recommend a conservative prototype that uses curvature only to nominate candidates and a local coarse/fine update comparison to decide refinement. Apply the expensive estimator only to buffered candidate patches, normalize error in physical/characteristic variables, and select refinement by predicted error reduction per wall-time under the global budget. [DONE]
  - Record required policy questions before implementation: shock refinement needs a physical/numerical dissipation stopping rule; error norms and variable scaling must be fixed; coarsening needs hysteresis; buffers should derive from stencil width and signal travel; and validation must compare error at equal cost against uniform fine references. [DONE]

## Cycle 59 [completed]
- Select one evidence-based AMR design for PyFluid rather than presenting an open-ended estimator menu. [DONE]
  - Compare foundational Richardson, Lohner, multiresolution finite-volume, and anisotropic block-AMR methods against PyFluid's shared FV/WENO/Riemann/RHS and AMReX hierarchy contracts. Select conservative two-level multiresolution detail as the production estimator; retain local Richardson/shadow evolution as an offline calibration and validation reference, not the default runtime tagger. [DONE]
  - Use curvature/Lohner only as a diagnostic and optional cheap candidate prefilter if profiling shows multiresolution cost matters. Do not combine multiple heuristic taggers by default. [DONE]
  - Fix implementation defaults: derive dimensionless detail from cell-average prediction errors in normalized primitive fields (`log rho`, `log p`, velocity scaled by local signal speed, and magnetic field scaled by thermal/magnetic pressure); combine by local max norm; exclude passive tracers; derive tag buffers from reconstruction radius plus maximum signal travel before the next regrid; refine/coarsen with hysteresis. [DONE]
  - Bootstrap the initial hierarchy by directly sampling IC and active-source cell averages on virtual dyadic children and applying the same detail estimator recursively. During evolution, evaluate source detail alongside state detail only for sources that backreact. [DONE]
  - Treat true inviscid discontinuities explicitly: no smoothness/error estimator converges at a mathematical shock. Physically diffusive runs stop at the resolved dissipation scale; ideal shock runs must report that refinement ended at the resource/resolution guard rather than claiming metric convergence. [DONE]
  - Select refinement under resource constraints by descending predicted error contribution per measured subcycled wall-time cost; if the requested error cannot fit, mark the run unresolved instead of silently accepting it. [DONE]
  - User-level scientific choices are reduced to requested relative representation error, resource budget, and physical dissipation scales. Estimator formula, variable normalization, hysteresis, buffering, and hierarchy construction are infrastructure defaults. [DONE]
  - Defer a custom locally anisotropic hierarchy. First implement the estimator offline, validate predicted detail against uniform-fine and Richardson references, wire it into standard AMReX tagging, add automatic initial hierarchy construction, and measure the remaining isotropic penalty on radial and nonsymmetric benchmarks. [DONE]

## Cycle 60 [completed]
- Constrain multiresolution AMR to the canonical transfer operators and define resource-aware refinement under the Mac mini memory limit. [DONE]
  - Reuse exactly one selected coarse-to-fine state transfer and one metric-conservative fine-to-coarse restriction for evolution, multiresolution prediction/detail, initial hierarchy construction, and validation; do not add analysis-only transfer formulas. Expose these as the shared hierarchy-transfer service instead of duplicating formulas in the tagger. [DONE]
  - Build the automatic initial hierarchy through the normal AMReX recursive tagging/grid-construction path, using direct IC/source child averages only to evaluate representation error. This is an initial-time error provider for normal `ErrorEst`/grid generation, not a second grid builder. [DONE]
  - Define a global error contribution and predicted error-reduction-per-subcycled-wall-time score so limited resources are assigned to the regions that most reduce total error, accounting for AMReX box/buffer overhead. Use physical-volume-weighted squared normalized detail for global contribution, retain a local maximum-error guard, and rank candidate boxes by predicted reduction divided by measured cost. [DONE]
  - Estimate reduction order from detail decay across existing levels and use first order when unavailable or nonsmooth. Estimate cost from clustered child cells, level substeps, and measured branch throughput rather than raw tag count. [DONE]
  - Reconcile requested error, total cell/memory budget, max-level safety, physical dissipation scales, and inviscid numerical-viscosity behavior without silently claiming metric convergence. Requested representation error is the accuracy objective; memory/cell budget is the hard resource constraint; max level remains a high emergency guard; physical dissipation supplies a meaningful smallest scale when present. Inviscid shocks may remain budget-limited and must be reported as unresolved rather than smoothness-converged. [DONE]
  - Coarsening reuses the same detail estimate: remove a fine region only when its predicted parent representation satisfies the target with an internal safety margin. This supplies standard anti-thrashing hysteresis without a user-facing coarsening parameter. [DONE]

## Cycle 61 [completed]
- Audit `fhermet/machmallow` as a small external AMR-fluid reference without assuming its AI-written implementation is correct. [DONE]
  - Clone tag `v1.0.0` (`43b6ecc`) to `/tmp/machmallow` and inspect hierarchy representation, tagging, grid construction, transfer, reflux, subcycling, dimensional support, and validation. [DONE]
  - The implementation is a deliberately narrow 2D Cartesian fp32 mini-AMR: fixed square blocks, isotropic ratio 2, relative density/velocity threshold tags, one uniform base grid, and no MPI, curvilinear geometry, local anisotropy, MHD, or implicit solve. It does not address PyFluid's estimator or polar-efficiency problem. [DONE]
  - Its recursive Berger-Colella ordering, per-level regrid cadence, time-blended fine ghosts, conservative restriction/reflux, explicit nesting invariant, unified-memory CPU-bookkeeping/GPU-kernel split, and discriminating reflux-on/off tests are useful compact references. [DONE]
  - Do not copy its architecture: two-level/multilevel and CPU/GPU paths are separately implemented; AMR configuration mixes physics and hierarchy options; tagging is flow-tuned; `stepCounts_` is hard-sized to 16 despite arbitrary-depth claims; and documented three-level Sod behavior has derefinement churn and near-full fine coverage. [DONE]
  - Build and run the CPU `sod_amr` and `ml_amr` gates locally. Both pass. Two-level Sod matches uniform-fine L1 (`2.1672e-3` versus `2.1709e-3`) at 63% of uniform cell-step work; reflux-off leaks 6279 times more mass than reflux-on. Three-level Sod is 1.43 times the uniform-fine error with `8.97e-7` mass drift; periodic KH drift is `2.39e-8` in fp32. [DONE]

## Cycle 62 [completed]
- Decide what, if anything, PyFluid should adapt from the machmallow audit and clarify its refinement terminology. [DONE]
  - Machmallow's ratio 2 means spacing is halved in every active direction: refinement vector `(2,2)` and four fine children per coarse cell in 2D. It does not implement selectable `(3,3)`, directional `(2,1)`, or locally automatic anisotropic ratios. [DONE]
  - Select the Woodward-Colella Mach-10 double-Mach-reflection problem as the worthwhile addition: adapt the published IC and time-dependent/segmented boundary conditions into PyFluid's existing config and validation infrastructure, not machmallow's solver or AMR classes. [DONE]
  - Useful validation ideas already largely represented in PyFluid are explicit reflux-on/off discrimination, composite error versus equivalent uniform-fine work, nesting checks, and per-level regrid accounting. Do not duplicate these merely because the compact reference presents them clearly. [DONE]
  - Record the curvilinear conclusion: isotropic mapped AMR can be correct but is structurally inefficient for radially thin, angularly smooth waves because every radial refinement also refines angle and worsens the origin CFL. Cartesian AMR is the robust general default; efficient radial far fields ultimately need mixed Cartesian/curved patches or a locally anisotropic hierarchy rather than more flow-tuned isotropic tags. [DONE]

## Cycle 63 [completed]
- Clarify the canonical double-Mach-reflection top boundary. [DONE]
  - The inclined incident shock intersects and moves along the top boundary during the standard test, so its ghost state is the exact time-dependent pre/post-shock continuation. Ordinary outflow would truncate and distort the incident shock. The right boundary is the transmissive/outflow boundary. [DONE]

## Cycle 64 [completed]
- Explain physically why double-Mach reflection needs the analytic top boundary. [DONE]
  - The finite rectangle clips an otherwise continuing planar incident shock. At the top edge, the shock intersection moves from `x = 1/6 + 1/sqrt(3)` at `t=0` to `x = 1/6 + (1 + 20t)/sqrt(3)`, while post-shock fluid has negative vertical velocity and therefore enters through the top. The boundary must prescribe post-shock state left of that intersection and pre-shock state right of it; zero-gradient outflow cannot represent this mixed moving inflow. [DONE]

## Cycle 65 [completed]
- Add the canonical Woodward-Colella Mach-10 double-Mach-reflection problem as a demanding Cartesian AMR validation and compare PyFluid's behavior with the useful machmallow evidence. [DONE]
  - Implement the planar-shock IC and moving segmented boundaries through shared initial-data/config/boundary-time infrastructure, with no DMR-specific evolution path. The left boundary supplies post-shock flow, the lower boundary switches from fixed post-shock inflow to reflection at the wall start, the top follows the exact moving shock intersection, and the right is outflow. [DONE]
  - Validate the small 3-rank AMR+subcycling case, physical-state floors, active fine levels, analytic open-boundary mass budget, and standard-suite integration. Shared 1D/2D/3D builds and the existing quick AMReX suite pass. [DONE]
  - Compare coarse, single-rate AMR, subcycled AMR, equivalent uniform-fine, and a finer reference. At `t=0.01`, subcycled AMR uses `87552/557056 = 15.7%` of the equivalent-uniform cell updates while matching its sampled errors (`rho`: `5.685e-3` versus `5.674e-3`; pressure: `1.340e-2` versus `1.337e-2`). [DONE]
  - Run the production visualization to `t=0.2`: `9.768 Mcells/s`, `6.063e-3` simulation-time/real-time, physical floors intact, 7.7% total cell-budget use, and analytic mass-budget relative error `7.32e-4`. The inviscid discontinuity reaches the configured level ceiling and is reported resolution-limited rather than metric-converged. [DONE]
  - Produce run-local density and schlieren movies, final frames, scalar report, and AMR hierarchy report. Fix the hierarchy plotter to consume the canonical metadata-prefixed TSV through the shared reader. [DONE]

## Cycle 66 [completed]
- Audit whether the double-Mach-reflection hierarchy actually follows the right-going incident shock and clarify PyFluid versus machmallow AMR infrastructure coverage. [DONE]
  - Confirm the original movie was numerically wrong, not merely visually ambiguous: the incident shock was covered only through `L1` at `t=0.2`, while the left reflected/contact structure reached `L5`. Direct evaluation showed the shock indicator remained about `0.99` against a `0.1` threshold, excluding estimator blindness. [DONE]
  - Fix the shared subcycled regrid cadence. The outer evolution counter was already the level-0 step, but `should_regrid` divided it by the cumulative finest-level time factor; `regrid_int=4` with five refinements therefore rebuilt every 128 level-0 steps. It now has the same level-0-step meaning with and without subcycling. [DONE]
  - Add a DMR regression that requires the analytic incident front to remain covered by the configured finest level. The strengthened quick case runs to `t=0.03` with three levels and passes; the corrected `t=0.2` production run covers the front at `L5` in all 19 saved frames. [DONE]
  - Refresh the quantitative reference: subcycled AMR uses `101632/557056 = 18.2%` of equivalent-uniform cell updates with nearly identical sampled errors (`rho`: `5.677e-3` versus `5.674e-3`; pressure: `1.3375e-2` versus `1.3366e-2`). The corrected production run evolves `746752` active cells at `10.64 Mcells/s`, uses 37.3% of its budget, and has analytic open-boundary mass-budget error `2.78e-3`. [DONE]
  - Validate the shared AMR path: dynamic regrid, global tagging, curvilinear subcycling equivalence, and curvilinear freestream regrid pass; 1D/2D/3D builds pass. The broad quick feature matrix passes 16/17 cases. Correct regridding exposes a separate pulsing-cut-cell AMR/subcycling composite mass defect (`1.16e-7` after pre-regrid conservation capture, tolerance `1e-8`); do not hide or weaken this failure. [DONE]
  - Infrastructure comparison: both codes provide recursive conservative FV AMR, time-subcycled levels, time-interpolated coarse/fine ghost data, restriction, reflux, proper nesting, dynamic regridding, checkpointing, and hierarchy diagnostics. PyFluid additionally uses AMReX multi-box MPI infrastructure and supports 1D/2D/3D, HD/MHD+CT, multiple numerical methods/EOS, diffusion, curvilinear coordinates, and broader validation. Machmallow additionally rebuilds descendants on each parent level's own cadence and has a Metal/GPU fixed-block pool/live renderer; PyFluid currently rebuilds only at synchronized level-0 times. [DONE]

## Cycle 67 [completed]
- Clarify conservation for the open double-Mach-reflection problem and strengthen its analytic boundary-budget validation. [DONE]
  - Compare measured mass and energy changes with the exact continuum inflow/outflow integrals before downstream right-boundary contact. [DONE]
  - At `t=0.2`, measured/analytic mass changes are `22.799586/22.863071` (`2.777e-3` relative) and energy changes are `1937.934329/1943.361006` (`2.792e-3` relative). These are continuum boundary-budget truncation errors, not roundoff conservation. [DONE]
  - Add the energy budget to the existing DMR mass-budget regression. The strengthened quick AMR+subcycling case passes with mass error `1.259e-3`, energy error `1.417e-3`, and incident-shock coverage through configured `L3`. [DONE]
  - Record the missing stronger diagnostic: accumulate the actual numerical physical-boundary fluxes with the same RK/subcycling weights used by evolution. The composite total minus this numerical flux budget should then close to roundoff; comparing only with the analytic continuum boundary integral cannot isolate discrete conservation from boundary truncation error. [DONE]

## Cycle 68 [completed]
- Clarify the corrected regrid cadence relative to machmallow and define the general external-conservation diagnostic direction. [DONE]
  - PyFluid's outer evolution counter is the level-0 step. The removed subcycling conversion divided that counter by the cumulative finest-level factor before applying `regrid_int`; with five refinements and `regrid_int=4`, the hierarchy rebuilt every 128 level-0 steps instead of every 4. [DONE]
  - Machmallow avoids this exact error with a separate step counter per level and `regridFrom_(level+1)` on each parent level's cadence. PyFluid now correctly rebuilds every configured level-0 synchronization interval, but does not yet implement machmallow's finer per-parent-level remesh cadence. [DONE]
  - Generalize open/driven accounting as a runtime external-conservation ledger: numerical physical-boundary fluxes, source injection/removal, and wall work use the evolution's actual RK/subcycling weights. Report unexplained residuals separately from optional continuum analytic-budget comparisons. [DONE]

## Cycle 69 [completed]
- Inventory machmallow infrastructure advantages and adapt the generally useful pieces without duplicating PyFluid's AMReX machinery. [DONE]
  - Map machmallow hierarchy scheduling, ghost interpolation, reflux/restriction, nesting checks, regrid tests, checkpoint state, and CPU/GPU organization to current PyFluid equivalents. [DONE]
  - Retain PyFluid/AMReX implementations of time-interpolated FillPatch, restriction, reflux, hierarchy construction, restart metadata, cell budgets, and regrid profiling; these already equal or exceed the corresponding narrow 2D fixed-block mechanisms. Do not copy duplicate CPU/GPU AMR classes, fixed square blocks, fp32 state, flow-tuned tags, or the hard-sized 16-level counter. [DONE]
  - Select machmallow's per-parent-level remesh cadence and explicit post-regrid nesting invariant as useful adaptations. The cadence keeps tag-buffer width scale-invariant; the invariant should be a debug/validation contract on top of AMReX rather than another hierarchy implementation. [DONE]
  - Design and implement a shared runtime external-conservation ledger for mass and energy: actual numerical physical-boundary flux, prescribed source injection/removal, moving-wall work, and unexplained residual. [DONE]
  - Evaluate and, if clean across supported features, implement per-parent-level regrid cadence for subcycled AMR; otherwise document the exact compatibility blocker and retain corrected synchronized regridding. AMReX's full `Amr` driver does use per-level `level_count` counters, but PyFluid is an `AmrCore` application with custom flux/edge-flux registers and source/solid caches. Its current regrid wrapper assumes a globally synchronized hierarchy, recreates every register, and resets all level times/steps. Calling it inside recursive subcycling would discard an in-flight parent register and mishandle moving-solid composite conservation. Retain corrected synchronized regridding until regridding is factored into a level-base operation that preserves the parent register and resets only descendants. [DONE]
  - Add focused validation that distinguishes analytic continuum-budget error from discrete conservation closure and preserves existing AMR/feature behavior. [DONE]

## Cycle 70 [completed]
- Clarify synchronized versus per-parent-level regridding, verify AMReX proper-nesting enforcement, audit AMR/subcycling consistency, and reassess a clean FP32 build option. [DONE]
  - Verify the current PyFluid regrid call schedule and identify exactly what Cycle 66 fixed versus what remains unimplemented. Cycle 66 corrected synchronized level-0 cadence; independent parent-level regridding remains unimplemented. [DONE]
  - Inspect the installed AMReX source for proper-nesting checks/enforcement rather than assuming the contract. `AmrMesh::MakeNewGrids` constructs proper-nesting domains from `n_proper`, removes inadmissible tags, projects descendants onto parents, and emits disjoint fine grids; do not duplicate hierarchy construction in PyFluid. [DONE]
  - Inventory shared regrid/subcycling coverage across timesteppers, CT, diffusion, gravity, solids, coordinates, and restart, including current failing gates. The focused AMR gates pass, and the complete AMR feature matrix passes 28/28 across all five timesteppers, hydro/MHD+CT, explicit resistivity, explicit/implicit diffusion, local-CFL subcycling, gravity/regridding, and static/moving solids. The broader combination matrix remains 16/17 because pulsating cut-cell geometry with AMR+subcycling has `1.156e-7` mass drift against `1e-8`. [DONE]
  - Determine whether FP32 can be a separate compile-time AMReX configuration with no runtime branching or source duplication, and define the required accuracy/performance gates. Build a separate single-precision AMReX successfully and make generated kernels emit precision-typed AMReX literals. A full PyFluid FP32 build still encounters roughly 2700 handwritten unsuffixed floating literals across 66 files; converting those cleanly is a broad precision-portability pass, so float64 remains production. The float64 generated-kernel equivalence gate passes and the canonical 3-rank hydro benchmark remains `109.13 Mcells/s` versus `109.91` baseline. [DONE]

## Cycle 71 [completed]
- Improve DMR boundary visualization, compare PyFluid regrid synchronization with CarpetX, and add general external mass/energy accounting. [DONE]
  - Add an honest postprocessing overlay marking the DMR lower reflecting wall/wedge start without introducing an evolution obstacle or mask. [DONE]
  - Inspect CarpetX and its subcycling development structure for useful synchronization/regrid ordering that can improve PyFluid without duplicating AMReX hierarchy mechanics. [DONE]
  - Add runtime cumulative numerical physical-boundary mass/energy flux and source mass/energy injection using the same RK/subcycling weights as evolution, retaining wall work as the mechanical source term. [DONE]
  - Report external-budget closure in `scalars.tsv` and validate closed, open DMR, and driven/heating cases with AMR and subcycling. [DONE]

## Cycle 72 [completed]
- Finalize conservation-ledger ownership, simplify the DMR boundary overlay, and state the exact level-synchronization support boundary. [DONE]
  - Keep numerical boundary/source/work accumulation in evolution, scalar reduction and serialization in diagnostics, and policy/closure checks plus plots in shared Python postprocessing. [DONE]
  - Remove the DMR wall text while retaining the honest fixed wall-start marker. [DONE]
  - Audit PyFluid against Liwei's CarpetX subcycling schedule: level-local clocks and parent/child reflux plus average-down synchronization are implemented; independent descendant regridding at intermediate parent synchronization points is not. [DONE]
  - Retain globally synchronized regridding for now. Independent descendant regridding is an optimization, not a missing state-sync operation, and must not be enabled until a level-base regrid preserves the parent flux/CT/gravity registers and rebuilds only descendants. [DONE]

## Cycle 73 [completed]
- Diagnose the DMR movie and measured throughput regression, then prototype lowest-aligned-level regridding as a selectable AMR cadence. [DONE]
  - Find and fix the generated DMR GIF path defect: repo-relative paths were incorrectly prefixed by `output_dir`, creating a duplicated nested path. Make `--gif` an ordinary path and have the high-level movie runner pass the complete run-local path. [DONE]
  - A/B benchmark the canonical periodic source-free path with conservation-ledger call sites compiled in and out. Five-run medians are `108.57 Mcells/s` with the ledger and `108.04 Mcells/s` with its direct-path call compiled out, so the earlier `100.8 Mcells/s` sample was timing noise rather than ledger overhead. [DONE]
  - Add a single runtime AMR regrid cadence option for synchronized level-0 regridding versus lowest-aligned descendant regridding. Preserve active parent registers and rebuild/reset only descendants. [DONE]
  - Reject the initial per-level-counter prototype after it caused 42 regrids versus 5 synchronized regrids in 12 coarse steps. Replace it with CarpetX's model: one finest-time event counter, one due flag, and one regrid from the lowest level whose descendants have caught up. [DONE]
  - Validate state agreement, conservation, AMR hierarchy behavior, and simulation-time/real-time for both regrid cadences. Hydro, local-CFL subcycling, MHD+CT, explicit and implicit diffusion, self-gravity, and checkpoint/restart pass; restart reproduces the uninterrupted state exactly and the CT case keeps divB at `5.13e-15`. Matched short tests do not show a speedup: aligned regridding removes stale cells earlier in some flows but can spend 5-27% of wall time rebuilding descendants. Keep synchronized as the default and aligned as an opt-in scheduling policy. [DONE]
  - Rebuild 1D/2D/3D and rerun the performance gate. The canonical 3-rank path is `109.31 Mcells/s`, 0.99 of its frozen baseline; the conservation ledger and dormant aligned-regrid option do not regress the default path. The quick suite remains 16/17 with only the pre-existing pulsating cut-cell AMR/subcycling conservation defect (`1.156e-7`). [DONE]

## Cycle 74 [completed]
- Explain and verify apparently detached AMR boxes in the corrected DMR movie. [DONE]
  - Compare rendered boxes with the state/tag metric, tag-buffer growth, proper-nesting support, and synchronized regrid lag. At `t=0.0907`, apparently detached level-2 boxes contain weak pressure/velocity gradients hidden by the `rho=1..16` color scale, while roundoff-uniform neighboring boxes are parent/proper-nesting support. Four-cell tag growth, eight-cell blocking, rectangular clustering, and up to four coarse steps of synchronized-regrid lag explain the visible halo. The boxes are intentional conservative hierarchy overhead, not spurious fluid state, though this DMR setup is not minimal-box optimized. [DONE]

## Cycle 75 [completed]
- Clarify the production AMR indicator fields and render box-free linear/log-density DMR movies from existing snapshots. [DONE]
  - Verify the hydro and MHD state variables contributing to automatic tagging. [DONE]
  - Render numbered linear-density and log-density movies without AMR outlines. Both contain 19 frames at 180 ms/frame and 1224x1054 output. The retained log-density result uses the honest `log10(rho)=0..1.21` physical range rather than percentile clipping the unshocked `rho=1` state. [DONE]

## Cycle 76 [completed]
- Clarify whether production AMR combines curvature tagging with an independent regrid-error estimate. [DONE]
  - Production `amr.tag_metric=auto` uses the metric-aware normalized Lohner curvature indicator over the selected hydro/MHD state fields. Regrid cadence only determines when tags are reevaluated and the hierarchy rebuilt; `grid_eff`, `n_error_buf`, proper nesting, and budgets shape or constrain the result. No independent Richardson/truncation-error estimator currently contributes to tags. [DONE]

## Cycle 77 [completed]
- Freeze the current production AMReX state with a scoped commit and a maintained validation/performance baseline snapshot. [DONE]
  - Keep JAX/Numba/reference-path changes unstaged; commit only `AGENTS.md` and the coherent `pyfluid_amrex` production changes as `1053014` (`Harden AMReX transfers and conservation validation`). [DONE]
  - Capture 14 passing production validation gates and the standard performance benchmark under `runs/baselines/20260712_024631_amr_transfer_conservation_20260712/`. The snapshot is tied to `1053014`; the canonical 3-rank result is `109.11 Mcells/s`, `0.2538` simulation-time/real-time, and 0.993 of its frozen performance baseline. All four tracked benchmark cases pass. [DONE]

## Cycle 78 [completed]
- Add the validated double-Mach-reflection result to the canonical compact finished-run collection. [DONE]
  - Curate boxed, linear-density, and honest log-density movies plus scalar/AMR plots under `runs/final_done/media/double_mach_reflection/`. [DONE]
  - Preserve compact configuration, logs, scalar/AMR TSVs, report, and profile under `runs/final_done/compact_runs/double_mach_reflection_2d_amr/` without copying raw HDF5 state snapshots. [DONE]

## Cycle 79 [completed]
- Keep curated DMR output self-describing and consistent with the one-run/one-folder output convention. [DONE]
  - Use the solver-authored canonical `run.in` as simulation metadata; do not create a per-run prose descriptor. [DONE]
  - Consolidate DMR movies, plots, logs, configuration, and TSV reports under `runs/final_done/double_mach_reflection_2d_amr/` instead of separate media and compact-run trees. [DONE]

## Cycle 80 [completed]
- Explain why lowest-aligned descendant regridding has not outperformed synchronized level-zero regridding in current tests. [DONE]
  - Alignment is an eligible synchronization point, not a separate error-demand trigger. The current aligned cadence counts finest accepted steps, so the same numeric `regrid_int` causes many more hierarchy rebuilds than synchronized mode. In the pressure-bump probe this produced 21 versus 5 regrids; aligned AMR reduced final cells from 20224 to 14208, but regridding consumed 67.6% of wall time versus 31.2% and simulation-time/real-time fell from 0.949 to 0.601. A matched-frequency DMR probe used 8 regrids with 4.7% regrid overhead but had little stale hierarchy to remove, so it also showed no net gain. Aligned regridding can win only when saved descendant evolution exceeds tagging, grid generation, redistribution, allocation, prolongation, and cache/register rebuild costs. [DONE]

## Cycle 81 [completed]
- Recheck aligned-regrid policy against the local Liwei CarpetX subcycling branch rather than inferring behavior from its synchronization contract. [DONE]
  - CarpetX increments a global driver iteration for each subcycling evolution event, applies `regrid_every` to that counter, finds the lowest contiguous active/aligned level as `min_active_level`, and calls `amrcore->regrid(min_active_level, time)`. Its `CarpetXRegrid::regrid_error` field and threshold produce tags inside AMReX `ErrorEst`; they do not cheaply suppress a scheduled regrid when topology is unchanged. Production/example intervals are commonly 10, 16, 32, or 128, while static cases disable regridding or use 100000. PyFluid's aligned scheduling model is structurally similar, but the initial interval-2/8 probes were too aggressive and too short to establish benefit. [DONE]

## Cycle 82 [completed]
- Verify whether AMReX and CarpetX already avoid expensive work when a scheduled regrid leaves the hierarchy unchanged. [DONE]
  - AMReX 25.11 compares each generated `BoxArray` with the existing level and skips `RemakeLevel` when unchanged. CarpetX additionally records `level_modified` in its level callbacks and runs post-regrid scheduling only over the first/last actually modified levels. PyFluid currently calls `define_flux_registers`, resets descendant clocks and caches, average-downs, refreshes boundaries/solids, and copies old state after every scheduled regrid regardless of whether AMReX modified a level. Therefore aligned mode is numerically functional but not yet performance-equivalent to CarpetX; this unconditional wrapper work is a concrete likely source of its measured slowdown. [DONE]

## Cycle 83 [completed]
- Make shared PyFluid regridding change-aware so aligned scheduling is as restrictive as the AMReX/CarpetX contract. [DONE]
  - Track levels modified by AMReX callbacks and make unchanged scheduled regrids return before descendant register/cache/state maintenance. A fixed-hierarchy probe records 8 attempts, 7 no-ops, one modified level, and roundoff conservation. [DONE]
  - Restrict changed-hierarchy maintenance to the minimal safe affected range while preserving reflux, CT, gravity, solids, restart, and all timesteppers. Rebuild registers and reset clocks from the first modified level; retain average-down and boundary/cache synchronization across the affected ancestor chain. Remove the redundant pre-regrid hierarchy boundary/CT fill because ErrorEst and level callbacks own their required FillPatch operations. [DONE]
  - Validate no-op and changed regrids, then benchmark synchronized and aligned modes at comparable physical cadences. Focused hydro, MHD+CT, and implicit-diffusion aligned gates pass. Add `hydro_amr_aligned_noop_contract` to keep the optimization observable. A three-run matched pressure-bump comparison uses synchronized interval 2 versus aligned interval 16: 14 versus 8 changed attempts, stable regrid costs of 0.0496 s versus 0.0262 s, and about 1.51 versus 2.24 simulation-time/real-time. The standard performance gate passes at 0.95-0.99 of baseline across all four cases. The quick feature matrix remains 16/17 before adding the new passing contract, with only the pre-existing pulsating cut-cell AMR/subcycling conservation failure. [DONE]

## Cycle 84 [completed]
- Measure the combined benefit of time subcycling and aligned descendant regridding against a no-subcycling hierarchy at matched finest-step regrid cadence. [DONE]
  - Run repeated no-subcycling/synchronized and subcycling/aligned pressure-bump cases with identical physics, AMR accuracy settings, final time, and regrid interval measured in finest-step units. At the same numeric interval 16, no-subcycling performs 12 regrids and has median simulation-time/real-time `0.9774`; aligned subcycling performs 8 regrids and reaches `2.2383`, a `2.29x` gain. [DONE]
  - Repeat with the no-subcycling interval raised to 24 so both policies perform eight regrids. Its median simulation-time/real-time is `1.2087`, making aligned subcycling `1.85x` faster at equal regrid count. No-subcycling reports about `50.95 Mcells/s` versus aligned subcycling `36.58 Mcells/s`; the lower aligned cell-update throughput is expected because it deliberately avoids updates and advances physical time faster. Both conserve mass and energy to roundoff. [DONE]

## Cycle 85 [completed]
- Promote efficient aligned subcycling for dynamic AMR and evaluate a production-quality representation-error tagger before changing the AMR metric. [DONE]
  - Audit and update registered defaults so AMR runs use time subcycling plus aligned regridding unless an input explicitly selects synchronized or no-subcycling behavior. Registered defaults are now `subcycling=on` and `regrid_mode=aligned`; all 27 existing validated AMR inputs explicitly retain synchronized cadence until individually retuned because their intervals were selected in coarse-step units. An omitted-default AMR probe records aligned/on in canonical `run.in`, passes conservation/causal checks, and the single-level 3-rank performance guard remains 0.99 of baseline. [DONE]
  - Add an equal-cost estimator comparison for current normalized Lohner curvature versus Richardson/shadow references on smooth, discontinuous, radial, and nonsymmetric flows. Source audit: AMReX delegates tags to the application; FLASH production defaults to normalized Lohner curvature; Berger-Oliger and Chombo document Richardson/truncation estimators. The 2016 Colella mapped-FV paper discusses truncation error at refinement boundaries but does not define a runtime truncation tagger. Implement `compare_amr_estimators.py`: its coarse control is compared directly with level 0 of a full-domain two-level AMR shadow run, so reflux and restriction are performed by the production solver. Every constituent run passes shared scalar validation. At a 10% tag budget, mean Richardson-error-energy capture is 0.665 for Lohner, 0.782 for Richardson, 0.745 for union, 0.778 for Richardson-heavy weighting, and 0.730 for candidate/sentinel sampling over 20-29% of the domain. [DONE]
  - Evaluate robustness/cost before changing production. Full Richardson shadow evolution costs 4.6-8.7 times the coarse evolution in the current small hydro cases. Spatial Lohner gating recovers most shock benefit cheaply but cannot discover the smooth-wave error Lohner missed; deterministic 6-25% sentinels do not fix that robustly. Select occasional global unbiased Richardson with Lohner on intervening regrids as the runtime prototype; do not promote union/weighted complexity or a spatially gated shadow branch from this evidence. [DONE]
  - Cycle 89's four-resolution cost/error study rejected the shadow-based
    temporal hybrid as noncompetitive, so no optional runtime branch was added. [DONE]
  - Production remained on Lohner tagging; later AMR, quick, and performance
    cycles supersede the proposed post-change gate because no tagger change was
    retained. [DONE]

## Cycle 86 [completed]
- Stress the AMR estimator comparison with noisy multiscale flow and select a robust automatic Richardson/hybrid policy without exposing a tag-budget tuning knob. [DONE]
  - Add a deterministic broadband hydro flow evolved by the production solver, and validate every coarse/shadow constituent run through the shared scalar checks. The thin multimode KH case evolves 65 SSPRK3 steps and makes the default Lohner detector tag 63% of the coarse domain. [DONE]
  - Compare Lohner, full Richardson, and simple hybrid policies by representation-error capture versus actual estimator/refinement work. Keep fixed tag fractions analysis-only. Full Richardson remains the Pareto accuracy reference; same-event OR/weighted hybrids spend Richardson's full cost and over-tag relative to its direct error map. [DONE]
  - Evaluate fixed dimensionless error tolerances and temporal Richardson calibration so the production interface does not require per-flow budget tuning. At a 1% normalized error tolerance, full Richardson selects 9.2% of cells on average across five flows versus default Lohner's 28.5%; Lohner still misses eight Sod violation cells. On the noisy KH time sequence, a cached Richardson map plus automatically calibrated Lohner and a two-cell analysis buffer selects 31.6% rather than default Lohner's 61.8%, catches every violation at both later times, and leaves maximum unrefined error below 0.009. [DONE]
  - Recommend or reject a runtime prototype from the expanded evidence; do not change the production tagger until the policy is robust and focused gates pass. Recommend an optional temporal prototype with a fixed normalized Richardson tolerance, pure Richardson tags on calibration events, and calibrated Lohner between events. Reject tag-budget controls, same-event unions, arbitrary score weights, and spatial Lohner-gated Richardson. Production remains unchanged until real AMReX box growth, cadence cost, subcycling, and conservation are measured. [DONE]

## Cycle 87 [completed]
- Quantify the optimistic benefit ceiling of Richardson tagging before selecting a production hybrid. [DONE]
  - Reuse the five validated coarse/shadow evolutions and sweep equal selected-cell fractions from 1% through 50%; no production parameter or solver behavior changes. [DONE]
  - Richardson captures 2.24x as much representation-error energy at 1% selected cells, 1.85x at 5%, 1.38x at 10%, and 1.17x at 20% averaged across the five flows. The advantage falls to 1.05x by 30% because Lohner eventually tags most relevant cells. [DONE]
  - The selective-flow ceiling is much larger than the aggregate: at 10%, noisy KH improves from 22.4% to 85.1% captured error and Sod from 69.9% to 96.9%. These are optimistic cellwise bounds before AMReX clustering, buffers, and estimator runtime cost. [DONE]

## Cycle 88 [completed]
- Remove circular estimator scoring before finalizing Richardson or a hybrid AMR policy. [DONE]
  - Add an independent full-domain third-resolution truth run. Use coarse-to-medium discrepancy only as the Richardson tag score and medium-to-fine discrepancy only as the holdout error target. All coarse, medium, and fine constituent evolutions pass shared scalar validation. [DONE]
  - Re-evaluate Lohner, Richardson, and simple hybrids at equal selected-cell fractions on all smooth, shock, radial, advected, and noisy flows. At 10%, holdout capture is 0.543 for Lohner and 0.710 for Richardson; noisy KH is 0.255 versus 0.819 and Sod is 0.887 versus 0.977. [DONE]
  - Determine whether Lohner contributes complementary predictive information on the independent holdout; reject weighting or cadence choices that only optimize the in-sample Richardson map. Same-event union and weights are not robust improvements at selective budgets: pure Richardson leads the aggregate through 15%, while Richardson-heavy weighting gains only 0.003 at 20%. Lohner becomes complementary mainly after selecting enough cells that the efficiency advantage is already small. [DONE]
  - Retain production Lohner unchanged until cross-resolution evidence and actual adaptive AMReX runtime tests support a replacement. The noisy temporal holdout still favors cached Richardson plus calibrated Lohner: with a two-cell analysis buffer it selects 31.6% and catches every independent holdout violation versus default Lohner selecting 61.8%. This is the preferred runtime prototype, not yet a production default. [DONE]

## Cycle 89 [completed]
- Improve production-style Lohner AMR with a robust, efficient adaptive Richardson ladder while avoiding per-flow budgets, fixed blend weights, and cadence tuning. [DONE]
  - Define one shared estimator ladder: global Lohner, Richardson over block-clustered biased/cached regions, error-driven threshold calibration and region growth, unbiased discovery, and full-domain saturation when regional work reaches break-even. No tag fraction, fixed blend weight, or manually selected regional extent is part of the policy. [DONE]
  - Prototype AMReX-block and halo costs against the independent coarse-to-medium estimator and medium-to-fine holdout on all five flows. Ordinary Lohner regions require 73.8% of the domain on average because distributed smooth waves, rings, and KH layers touch nearly every eight-cell block. Lohner-local-maxima seeding reduces localized-cloud work but correctly saturates to global Richardson on the tiny smooth/radial and distributed KH grids. Sparse probes add cost without improving holdout capture in these snapshots. [DONE]
  - Exercise cache and causal-age behavior on the noisy KH time sequence; derive escalation from observed error and movement rather than fixed iteration counts. Cached Richardson plus calibrated Lohner remains the effective cheap tier. A current regional Richardson pass over block-aware noisy-KH candidates already covers 92-100% of the domain, so the break-even rule should skip regional work and either reuse cache or run global calibration. The cached two-cell analysis buffer catches every independent holdout violation between calibration times. [DONE]
  - Implement only a policy that survives accuracy-versus-cost analysis. Do not add a second shadow evolution to production: it is too expensive and would duplicate stateful timestepper side effects. The hierarchy-native coarse-prediction/restricted-fine estimator was retained as a future research idea, not an active production task. [DONE]
    - Reference decision: SAMRAI `StandardTagAndInitialize` supports Richardson, gradient detection, and refine boxes individually or in combination; it deliberately applies Richardson first and gradient detection second, selects criteria by cycle/time, advances both the hierarchy level and a coarsened temporary level through the shared level integrator, then restores pre-estimate state. Follow this generic regrid-advance workspace pattern rather than adding physics-specific estimator advances. [DONE]
    - Condense the runtime policy experiment to one nested schedule with only three candidate global calibration periods (12, 24, and 48 regrid events): Lohner every event; regional spatial defect over normal Lohner/cache at period/6; lower-threshold Lohner candidates at period/4; expanded cached/lower-threshold regions at period/2; full temporal Richardson at the global period. Coincident events execute only the broadest due check, and regional work saturates to global at measured AMReX box-cost break-even. [DONE]
    - The held-out error/cost study below rejected every shadow schedule; no
      nested calibration parameters or losing production policies were retained. [DONE]
    - Make the AMR driver selection explicit before adding new estimators: replace the ambiguous supported value `amr.tag_metric=auto` with `amr.tag_metric=lohner` across C++, maintained inputs, and validation tools. Keep `none` and `lohner` as the only accepted values until `richardson` and `hybrid` actually exist; do not retain an `auto` alias. All 1D/2D/3D builds pass, the obsolete value aborts with the supported set, and the quick suite remains at its known 16/17 state with only the pre-existing pulsating cut-cell AMR/subcycling conservation defect. [DONE]
    - Strengthen the policy comparison to four independent resolutions and four temporal policies over smooth sine, radial pressure bump, advected cloud, and noisy KH. Coarse-to-medium defines the Richardson predictor, medium-to-eight-times-finer defines held-out error, and the four-times/eight-times-finer difference checks reference convergence. The reference-error energy is 3.5% of the held-out error on average, so the target is materially finer rather than circular. At a 0.3% normalized error tolerance, Lohner selects 30.0% of cells, captures 99.7% of held-out error energy, and has 0.910 worst recall. The stride-2 cached/calibrated hybrid selects 22.6%, captures 99.8%, has 0.913 worst recall, but costs an estimated 4.87x because of full-domain shadow calibration. Stride 4 costs 3.41x and is slightly worse than Lohner; global Richardson costs 7.81x for only a marginal gain. Reject the shadow-based hybrid as noncompetitive. [DONE]
      - Keep Sod in the existing three-resolution and exact-solution analyses. Its additional full-domain level-3 reference exposed six nonphysical cells from quartic-conservative transfer at CFL 0.9 by step 7; do not hide that distinct transfer/stability defect by lowering CFL or silently changing interpolaters inside this estimator comparison. [DONE]
    - A hierarchy-native synchronization defect remains a possible future
      estimator, but unbiased discovery is unsolved and production remains
      `lohner`; this is not part of the current GR queue. [DONE]
  - No estimator default changed. Later AMR accuracy, subcycling, conservation,
    hierarchy, and performance gates therefore validate the retained Lohner
    production path rather than an unimplemented Richardson branch. [DONE]

## Cycle 90 [completed]
- Interpret the apparently saturated Lohner score honestly and isolate the level-3 full-domain Sod failure with quartic-conservative coarse-fine transfer. [DONE]
  - Separate error-energy capture from violation recall and equal-budget selectivity so a dominant-cell L2 score is not treated as broad AMR correctness. Lohner's 99.75% aggregate held-out L2 capture occurs while selecting 30.0% of cells; it does not contradict its 54.3% capture at an equal 10% budget or its selective misses on Sod/noisy KH. The shadow hybrid does not justify its complexity or cost. [DONE]
  - Reproduce the Sod defect while varying only hierarchy depth, state interpolater, and subcycling at CFL 0.9. Level 2 spatial subcycling passes; level 3 spatial subcycling fails identically at step 7 with both conservative-linear and quartic-conservative transfer; level 3 without subcycling passes; level 3 spatial passes at CFL 0.8 and 0.7; level 3 local-CFL subcycling passes at CFL 0.9; and level 4 local-CFL also passes. This is not a quartic prolongation defect. The fixed spatial schedule assumes exactly the geometric 2:1 temporal ratios while the strong shock develops faster fine-level waves during a parent step. Local-CFL responds with level step counts `4,11,24,50` instead of `4,8,16,32`. [DONE]
  - Make full-domain estimator shadow/truth/reference runs use the already-supported local-CFL subcycling schedule. Also invalidate cached constituent runs whenever their materialized input changes. The clean four-resolution rerun retains the estimator conclusion: Lohner capture 99.750%, stride-2 hybrid 99.782% at 5.93x estimated runtime, stride-4 hybrid 99.579% at 4.09x, and global Richardson 99.803% at 9.57x. [DONE]
  - Add `hydro_sod_deep_amr_subcycling_local_cfl` to the maintained feature matrix: four full refinement levels, SSPRK3/PWL/HLLC, quartic-conservative transfer, CFL 0.9, and physical checks every accepted substep. It passes through `t=0.016` with level step counts `4,11,24,50,102`, mass and energy drift at roundoff, and positive density/pressure. Momentum is intentionally excluded from the scalar invariant policy because unequal end pressures exert a net force on the finite Sod domain. No shared transfer defect was confirmed; retain spatial subcycling as the faster fixed-ratio option and use local-CFL or a lower CFL for rapidly changing strong shocks. [DONE]

## Cycle 91 [completed]
- Quantify hybrid improvement specifically where Lohner is selective, and make local-CFL substep selection the robust omitted-parameter default. [DONE]
  - Report equal-10%-budget gap recovery per flow rather than relying on aggregate L2 capture. Distinguish the cheap regional candidate from global Richardson-weighted blends that retain full estimator cost. [DONE]
  - Change only the registered `pyfluid.subcycling_steps` default from `spatial` to `local_cfl`; preserve explicit input choices and retain `spatial` as the faster fixed-ratio option. [DONE]
  - Validate the omitted-default contract, deep Sod regression, focused local-CFL feature cases, and canonical performance guard. The canonical `run.in` records `local_cfl`; hydro SSPRK3, hydro RK4, MHD+CT SSPRK3, and deep Sod local-CFL cases pass 4/4. The standard performance gate passes all four cases: canonical 3-rank 2D hydro is `108.78 Mcells/s` (`0.990x` baseline), 4-rank 2D is `131.55` (`0.960x`), 3D PWC is `5.97` (`0.936x`), and 3D WENO-Z/RK4 is `0.182` (`0.958x`). Single-level performance is unchanged by the AMR-only default. [DONE]

## Cycle 92 [completed]
- Reduce the Richardson/Lohner design to an optional adaptive global-calibration policy with no user cadence tuning. [DONE]
  - Preserve pure Lohner as an explicit zero-Richardson mode. Define one adaptive mode that runs Lohner continuously, caches global Richardson results, triggers early on rapid cheap-indicator change, and lengthens/shortens calibration age from observed Richardson-only misses. The controller interval spans every regrid event through exponential backoff and a dormant state; explicit `off` is permanently dormant. [DONE]
  - Validate cadence decisions against an independently finer reference on smooth, shock, advected, and rapidly developing nonlinear flows. Over 36 held-out samples, adaptive global calibration performs nine calibrations and two early triggers, captures 99.87% of violation error with 0.926 worst recall, and has an estimated 1.22x runtime versus 8.40x for global Richardson every event. Pure Lohner captures 99.36% with 0.591 worst recall. Smooth sine and advected cloud back off from 32 to 64 events; KH backs off to 64 after a 0.49% cache-missed error observation; Sod triggers early at events 24 and 48 but both checks are clean and the interval grows to 128. [DONE]
  - Correct analysis cost accounting: independently restarted temporal runs are cumulative, while a runtime calibration advances only one regrid interval. Amortize shadow overhead by elapsed regrid opportunities. Separate requested inputs as `requested.in` from solver-authored canonical `run.in` so expanded defaults do not invalidate every cached run. [DONE]
  - Do not add a production interface prematurely. PyFluid does not yet have a generic side-effect-safe global Richardson evolution operator. Remove the sprawling offline harness and scratch outputs, retain pure Lohner as the production state tagger, and preserve the measured semiproduction policy and promotion contract in `docs/adaptive_richardson.md`. [DONE]

## Cycle 93 [completed]
- Finalize the current AMReX state for production, remove discarded estimator experiments, validate and freeze performance, commit the coherent implementation, then produce a high-resolution compressed-output AMR movie. [DONE]
  - Audit the complete dirty AMReX diff and preserve unrelated JAX/Numba/reference-path work untouched. The coherent AMReX diff contains callback-based change-aware regridding, explicit Lohner naming, robust subcycling defaults, maintained-input compatibility pins, and regression coverage; unrelated JAX/Numba files remain unstaged. [DONE]
  - Keep only runtime-supported, validated AMR/subcycling behavior. Remove the discarded Richardson analysis harness and scratch runs; retain `none` and `lohner` as the only production taggers. Registered defaults are aligned regridding, subcycling on, and local-CFL fine-level steps; fixed spatial ratios remain an explicit faster option. [DONE]
  - Rebuild all dimensions, run relevant standard/full gates and performance benchmark, then capture a named baseline. All 1D/2D/3D targets build. Full AMR, reference, and analytic gates pass. The first benchmark passes all four performance guards. The broad standard gate remains 16/17 only because of the pre-existing pulsating moving-cut-cell mass/ledger defect; do not weaken that check. [DONE]
  - Commit only coherent AMReX production files. Commit the initial cleanup/defaults as `8bc7fea`; preserve unrelated JAX/Numba files unstaged. [DONE]
  - Run and postprocess one high-resolution, high-frame-count structured-flow AMR case using local-CFL aligned subcycling and compressed HDF5 output. The first honest rendering exposed that descendant aligned attempts consumed the cadence without any whole-hierarchy rebuild: 422/422 attempts were no-ops and the hierarchy stayed frozen around the initial shock. Add a restart-persistent base cadence, retain descendant regrids, and strengthen the moving-cloud regression to require matching synchronized/aligned box motion at equal physical time. Fix hierarchy-weighted movie limits so AMReX box splitting cannot bias colors. [DONE]
  - Rerun DMR from a `64x16` base with an effective `4096x1024` finest resolution, level 6 active, six levels of headroom, local-CFL aligned subcycling, and 192 losslessly compressed HDF5 snapshots. The hierarchy follows the Mach stem and reflected/slip-line structure, peaks at 24% of the 5M-cell budget, closes the prescribed open-boundary ledger to `2.06e-13` mass and `1.76e-13` energy, and runs at `10.85 Mcells/s` and `2.76e-4` simulation-time/real-time. Render 160-frame density movies with and without AMR boxes. [DONE]
  - Rebuild all dimensions after the aligned-cadence fix, rerun the full AMR/restart gate, and rerun performance. The AMR gate is green; the benchmark passes at `107.95`, `127.37`, `5.95`, and `0.182 Mcells/s` for the four maintained cases. Commit the correction as `98b5be2` and freeze the final AMR/restart plus performance baseline under `runs/baselines/20260712_211256_aligned_dynamic_amr_production_20260712/`; the frozen benchmark records `108.19`, `126.15`, `5.95`, and `0.182 Mcells/s`. [DONE]

## Cycle 94 [completed]
- Audit the accepted high-resolution DMR result for honest open-boundary conservation and AMR feature coverage rather than relying on the scalar summary or visual appeal. [DONE]
  - Reconstruct prescribed mass/energy exchange from runtime ledger columns and verify that augmented totals close at every output, not only at the final row. The discrete boundary-flux ledger closes to `2.06e-13` mass and `1.76e-13` energy relative over the full run. An independent integral of the prescribed DMR boundary states differs from the measured changes by `2.69e-3` mass and `2.70e-3` energy; this is finite-volume boundary accuracy, not roundoff. [DONE]
  - Quantify which visible boxes contain Lohner tags versus tag buffers, proper-nesting support, or stale cadence, and identify any genuinely detached refinement. The all-level overlay is visually dominated by necessary coarse nesting/buffer boxes and repeats colors across levels. The final hierarchy is three coarse steps past its last global regrid and retains a small set (`2.06e-4` of sampled area) of uncovered L5 cells above the `0.10` tag threshold. Regridding itself costs only `0.32%` of wall time, but changing cadence remains a production tradeoff because it also changes hierarchy churn and flux work. [DONE]
  - Compare density-gradient/curvature structure with finest-level coverage to measure missed small eddies and determine whether the issue is tagging, clustering, regrid cadence, or rendering. On an independent `1024x256` composite sample, L5+ covers effectively all of the top 5% density, pressure, and vorticity features. L6 covers `94.5%`, `93.4%`, and `92.3%` respectively; for the top 1%, L6 coverage is `98.7%`, `98.6%`, and `98.7%`. The strongest small structures are refined, while weaker structures and cadence lag remain visible limitations. [DONE]
  - Fix only confirmed general defects, rerun focused validation, and revise the curated movie if needed. Consolidate the independent DMR analytic boundary budget into shared postprocessing, tighten the dedicated validation gate from 5% to 1%, pass the focused DMR test, and add a curated L5-L6-only box movie plus `physics_postprocess.tsv`. The broad quick gate remains 16/17 solely because of the pre-existing pulsating moving-cut-cell mass/ledger defect. Do not claim the long-time nonlinear morphology is reference-validated until a same-time uniform or independently finer comparison is run. [DONE]

## Cycle 95 [completed]
- Harden aligned-subcycling AMR regressions around the cadence-starvation failure without weakening its performance benefit. [DONE]
  - Upgrade the moving-feature regression from one to multiple fine levels and require every active refined level to move with the feature. The prior L1-only test was structurally incapable of reproducing descendant-event starvation. The corrected regression places the resource floor below L2 and verifies both L1 and L2 centroids through the full physical evolution. [DONE]
  - Compare aligned and synchronized final accuracy, hierarchy motion, and simulation-time throughput without making a noisy microbenchmark the sole correctness gate. Both modes move L1/L2 to `x=0.5`; aligned/synchronized density L2 error is `0.995855`, and aligned simulation-time throughput is `1.375x`. Record the ratio and retain a noise-tolerant `0.90x` regression floor while keeping accuracy and hierarchy motion strict. [DONE]
  - Exercise checkpoint/restart persistence of aligned cadence counters in an existing AMR restart check. Convert the pressure-bump subcycling restart control to aligned mode with a cadence interrupted by the checkpoint; final checkpoint comparison is bitwise exact (`checkpoint_max=0`, `total_l1=0`). [DONE]
  - Run focused aligned dynamic/restart validation and confirm maintained performance remains within its existing baseline guard. The strengthened dynamic test and aligned restart check pass. No evolution kernel changed in this cycle; the post-fix canonical performance baseline remains the one captured after commit `98b5be2`. [DONE]

## Cycle 96 [completed]
- Stress aligned subcycling at L7+, settle the DMR Lohner-versus-hybrid policy, and extend the high-resolution DMR movie through domain traversal. [DONE]
  - Add a maintained L7 aligned/local-CFL hierarchy test with two levels of headroom and a strict all-level conserved-GF uniform-state check. The first blocking-factor-4 probe exposed coarse-fine interface errors that global conservation missed: uniform density/pressure errors appeared from L4 and reached about 2% by L7. This was not a subcycling defect. AMReX `ProperlyNested` shows that quartic interpolation with this ghost width requires a larger blocking factor; blocking factor 8 preserves every field through L7 to roundoff. Add the general pre-evolution `ProperlyNested` contract rather than changing `n_proper` or repairing bad transfer output. The retained standard-suite L7 gate passes in 0.52 s at three ranks. [DONE]
  - Permit ten levels in the production-style automatic KH comparison while retaining physical stopping criteria and resource guards. The Lohner metric stops cleanly at L5 with five levels of headroom and only 3.71% of the cell budget. Aligned local-CFL subcycling is `1.376x` faster in simulation-time/real-time than no subcycling at equal final time, with pressure/rho relative L2 differences `4.04e-7`/`1.50e-7`. The matched dynamic L2 synchronized/aligned regression independently measures `1.379x` aligned speedup with a `0.995855x` density-error ratio. [DONE]
  - Run the DMR-specific fine-reference suite. Pure Lohner L2 AMR gives density/pressure relative L1 errors `0.007142`/`0.014505`, effectively identical to equivalent uniform-grid `0.007153`/`0.014526`, while using 18.5% of its cell updates. Together with the L6 audit covering over 98% of the strongest 1% density, pressure, and vorticity features, this rejects Richardson overhead for DMR. Keep pure Lohner as the DMR production policy; retain the documented adaptive hybrid only for future flows with demonstrated Lohner misses. [DONE]
  - Make the analytic DMR boundary integral restart-interval aware and refuse an analytic total-budget claim after the shock reaches the solution-dependent outflow boundary. Before contact, the focused DMR gate passes the tightened 1% analytic budget check. The extended runtime finite-volume boundary ledger remains applicable and closes to `3.38e-13` mass and `3.89e-13` energy. [DONE]
  - Restart the accepted L6 DMR run at `t=0.20` and evolve through `t=0.33` on three ranks. The extension takes 1196 s, sustains `10.67 Mcells/s`, reaches L6 with six levels of headroom, and peaks at 18.1% of the 5M-cell budget. Combine 329 real compressed snapshots and render no-box plus L5-L6 box density movies; curate compact outputs under `runs/final_done/double_mach_reflection_2d_amr_power_extended/`. [DONE]
  - Rebuild 1D/2D/3D, pass the focused L7 and DMR gates, and run the quick suite through all 17 cases. Its only failure remains the pre-existing pulsating moving-cut-cell conservation defect. The performance benchmark passes at `108.40`, `130.24`, `6.09`, and `0.18 Mcells/s`, or `0.99`, `0.95`, `0.96`, and `0.96` of the maintained baselines. [DONE]

## Cycle 97 [completed]
- Discuss aligned-subcycling performance, a single robust arbitrary-level AMR renderer, and the remaining DMR hierarchy/visual questions without changing code. [DONE]
  - Keep comparison denominators explicit: aligned subcycling is `1.85x` faster than no subcycling at equal regrid count and `2.29x` at the same numeric cadence in the pressure-bump study, while aligned versus synchronized subcycling is about `1.38x` in the maintained moving-feature and deep-KH comparisons. [DONE]
  - Consolidate future AMR rendering around one hierarchy-aware composite path: finest valid state wins, covered coarse data is omitted, arbitrary levels are supported, and box display distinguishes refinement-region boundaries from raw AMReX allocation boxes. [DONE]
  - Treat the DMR box noise as a mixture of real weak pressure/velocity structure, tag buffers, proper nesting, box clustering, and finite regrid cadence. Existing quantitative coverage shows L6 contains over 98% of the strongest 1% density, pressure, and vorticity features, but the diagonal striping and long-time morphology still require same-time finer/reference comparisons before being called physical. [DONE]

## Cycle 98 [completed]
- Discuss a kuibit-style lazy AMR sampling interface and a sharper DMR numerical-method comparison without changing production code. [DONE]
  - Use one lazy hierarchy representation with two views: native composite tiles as the authoritative data view, and a requested physical-region/uniform-resolution sampler that reads only intersecting level chunks and chooses the finest valid donor. [DONE]
  - Keep nearest-neighbor sampling as the default uniform visualization, expose multilinear interpolation only as a clearly labeled presentation option, and prefer physical area/volume averaging when downsampling to pixels coarser than the available simulation cells. [DONE]
  - For DMR, retain HLLC as the sharp current hydro Riemann solver and compare PWL, component WENO-Z, characteristic WENO-Z, and a general troubled-face component/characteristic policy at equal AMR accuracy and physical time. Treat a shock-stable HLLC/HLLE blend as a general carbuncle-control candidate only if the comparison identifies grid-aligned shock instability. [DONE]

## Cycle 99 [completed]
- Consolidate 2D AMR visualization and perform an evidence-driven DMR reconstruction study before considering a hybrid method. [DONE]
  - Add `amr_plot_data.py` as the hierarchy source of truth for native tiles and optional uniform nearest, linear, or area-weighted presentation views. HDF5 crop requests filter boxes before reading payload data; native movie rendering remains the default and does not resample. [DONE]
  - Route the snapshot, multivariable, comparison, validation, and regular single-patch movie paths through the shared hierarchy/component semantics. Preserve the established public wrappers while removing 326 lines of duplicated implementations. [DONE]
  - Add focused DMR-gate checks for arbitrary active levels, exact finest-over-coarse precedence, cropped/full-view equality, and finite optional resampling. The L3 focused gate passes with zero finest and crop disagreement; an independent cropped L6 probe exposes all seven levels. [DONE]
  - Extend the DMR validator with matched PWL, component WENO-Z, characteristic WENO-Z, and existing level-hybrid runs at `t=0.10`, all using HLLC/SSPRK3/CFL 0.4 and the same L2 AMR policy, plus a uniform `512x128` characteristic reference. Record density/pressure L1/L2/gradient error, vorticity error, checkerboard residual, work, Mcells/s, and simulation-time throughput. [DONE]
  - Characteristic WENO-Z gives an aggregate error ratio of `0.8809` versus component WENO-Z and improves every retained error metric, at `0.881x` its simulation-time throughput. The existing level policy (component on L0-L1, characteristic on L2) reproduces full characteristic density to `5.33e-9` mean difference and is `1.027x` faster than full characteristic. Keep it opt-in; the modest gain does not justify another local sensor/kernel. [DONE]
  - Pass Python compilation/diff checks, native movie rendering, the focused DMR L3 renderer gate, and the full matched reconstruction study. The broad quick gate remains 16/17 solely because of the known pulsating moving-cut-cell mass/ledger defect; no evolution kernel changed in this cycle, so the maintained solver performance baseline remains applicable. [DONE]

## Cycle 100 [completed]
- Diagnose the visible DMR stair stepping and diagonal bands without hiding them through interpolation. [DONE]
  - Replace the reconstruction validator's nearest-neighbor enlargement to the `512x128` reference canvas with physical area averaging onto the common candidate resolution. The regenerated plot now compares honest `256x64` finite-volume cell averages with interpolation disabled. Apply the same resolution-matched reference policy to coarse/uniform/AMR error metrics. [DONE]
  - Inspect the standard DMR boundaries: fixed post-shock left inflow, fixed post-shock lower segment for `x < 1/6`, reflecting lower wall to its right, moving analytic top shock, and right outflow are implemented consistently. Overlaying L2 boxes shows that the diagonal bands do not follow coarse-fine interfaces. [DONE]
  - Run a matched `256x64` uniform PWL control through `t=0.1`. In the striped lower-left region, AMR-versus-uniform density L1 is `5.47e-4`, while AMR/uniform versus the restricted fine characteristic reference is `2.93e-1`; the AMR discrepancy is only about `0.19%` of the shared method error. The bands are reconstruction/startup shock-capturing error, not an AMR or boundary defect. The remaining left staircase is the native cell-average representation of a curved discontinuity. [DONE]
  - Rerun the matched reconstruction study and AMR/reference gate at `t=0.1`; both pass. Characteristic and level-hybrid WENO-Z reduce the corrected aggregate error to `0.8377x` component WENO-Z and strongly suppress the bands. No evolution kernel or boundary behavior changed. [DONE]

## Cycle 101 [completed]
- Clean reconstruction policy placement, evaluate a PWL/characteristic-WENO level policy, and audit remaining symbolic-codegen leverage without widening physics scope. [DONE]
  - Move the general per-level reconstruction selector out of solid geometry and into the central private solver interface shared by RHS assembly and solid-state stencil preparation. Remove the per-call string test by resolving the optional fine method once during construction; no reconstruction method-selection policy remains duplicated. [DONE]
  - Compare PWL-to-characteristic-WENO, component-WENO-to-characteristic-WENO, and full characteristic WENO on the matched DMR case through `t=0.1`. PWL-to-characteristic WENO-Z reproduces the full-characteristic aggregate error (`0.8377x` component WENO-Z) and runs at `0.1617` simulation-time/real-time versus `0.1428` for full characteristic, a `1.13x` gain. Promote the existing general level policy in the production DMR input without adding a DMR-specific solver branch, and remove the obsolete component-hybrid run artifact. [DONE]
  - Inventory generated and handwritten physics kernels against the shared symbolic algebra layer. Ideal HD/MHD fluxes, EOS closures, LLF/HLL combinations, orthogonal geometry sources, viscous stress, curl/divergence, resistive EMF/CT, and patch metrics already use generated kernels. Regeneration is content-stable. Characteristic projection/recovery is the next coherent codegen candidate, but requires shared generated characteristic types plus round-trip and performance gates; do not mix that larger migration into this cleanup. [DONE]
  - Rebuild 1D/2D/3D, pass `validate_codegen_flux.py`, pass the final DMR reconstruction sweep, and run the quick suite. The quick suite remains 16/17 solely because of the tracked pulsating moving-cut-cell conservation defect. The maintained benchmark passes at `109.38`, `131.28`, `6.09`, and `0.18 Mcells/s`; the cleanup introduces no measurable production regression. [DONE]

## Cycle 102 [completed]
- Generate the two analytic characteristic transforms, simplify the production DMR reconstruction choice, and preserve the existing reconstruction/runtime contract. [DONE]
  - Replace the production DMR level hybrid with full characteristic WENO-Z; retain component WENO-Z and PWL as explicit validation/performance controls rather than hidden solver fallbacks. Keep the generic coarse/fine policy covered by the DMR comparison but do not use it in the standard movie input. [DONE]
  - Add one generated characteristic-kernel family for 5-wave HD projection/recovery and 7-wave reduced-MHD projection/recovery. Move the shared data types into `characteristics.hpp`; keep the EOS-aware, degeneracy-safe MHD basis constructor and WENO numerical algorithm handwritten. Cache the basis velocity denominator once instead of recomputing it for all six stencil states. [DONE]
  - Add independent reference projections plus 128 randomized projection/recovery round trips for both systems to the canonical codegen gate. Regenerate content-stably, build 1D/2D/3D and the C++ codegen-check target, and pass focused hydro characteristic-WENO, ideal-MHD characteristic-WENO/HLLD/CT, and polytropic-MHD characteristic-WENO/HLLD/CT runs. The standalone C++ check executable retains its documented unrelated Thornburg patch-map mismatch. [DONE]
  - Rerun the matched DMR study: generated characteristic transforms preserve the `0.8377x` component-WENO aggregate error and run at `0.1413` simulation-time/real-time versus the immediately preceding handwritten `0.1428`, within timing noise. The maintained benchmark passes at `108.94`, `130.18`, `5.98`, and `0.18 Mcells/s`; no non-characteristic regression appears. The quick suite remains 16/17 solely because of the tracked pulsating moving-cut-cell conservation defect. [DONE]

## Cycle 103 [completed]
- Solidify the generated parabolic/elliptic operators and gravity validation without widening this cycle to relativistic physics. [DONE]
  - Inventory explicit/implicit viscosity, MHD resistivity, CT, elliptic-solver ownership, EOS coverage, and gravity validation; identify duplicated or non-generated physics algebra. Explicit and implicit viscosity share the generated stress/operator kernels; explicit and implicit CT resistivity share the generated resistive EMF and CT curl update. [DONE]
  - Route explicit and implicit viscosity and resistivity through shared generated constitutive/operator definitions wherever AMReX centering permits, preserving one AMR transfer and synchronization contract. Add single-level Cartesian implicit resistive CT through AMReX `MLCurlCurl`; retain the generated RHS/update operator and explicitly reject multilevel/curvilinear use because AMReX 25.11 does not provide that backend. [DONE]
  - Verify CT plus resistivity and every supported EOS across relevant dimensions, coordinates, AMR, subcycling, and restart combinations; document genuinely unsupported topology rather than masking it. Generated/codegen, explicit/implicit Fourier decay, six EOS/physics combinations, deep L4-L5 AMR viscosity/resistive CT, and 1D/2D/3D builds pass. Explicit resistive CT reaches L5 with four levels of headroom and `divb=6.76e-15`; implicit CT is intentionally limited by AMReX 25.11 `MLCurlCurl` to one-level periodic Cartesian 2D/3D. A checkpoint/restart gate is bitwise exact for cell state and face-centered B. [DONE]
  - Strengthen fixed/self-gravity elliptic, source-stage, AMR, restart, and total-energy validation. Periodic Poisson, RK stage coupling, fixed-field/energy, HSE, and homogeneous-Neumann smoke gates pass. A matched L4 self-gravity comparison at `t=0.02` gives `2.47e-7` non-subcycled and `1.46e-6` subcycled total-energy drift; conservative physical-space comparison gives maximum relative L2 `4.46e-4` across rho/pressure. A restart control evolves from L3 to metric-limited L6 with four levels of headroom and agrees to `4.26e-14` maximum state difference. Subcycling gives no speed benefit in this solve-dominated case and remains experimental. [DONE]
  - Run focused performance gates and assess the additional projection/constraint interface required for incompressible flow. The maintained benchmark passes at `109.29`, `129.70`, `5.99`, and `0.18 Mcells/s`, or `0.99`, `0.95`, `0.94`, and `0.96` of baseline. Incompressible flow is not another EOS: it needs a separate physical-system module with an advective predictor, pressure Poisson solve, and velocity projection; the elliptic infrastructure is reusable, but that implementation is outside this cycle. [DONE]
  - Run the normal confidence gate after the focused tests. The quick suite is 17/18; its only failure is the pre-existing pulsating moving-cut-cell mass/external-ledger defect (`1.16e-7` mass drift), unchanged by this cycle. [DONE]
  - Evaluate conservative hierarchy transfer for self-gravity without targeting the diagnostic. A temporary `E_fluid + E_potential` transfer increased the three-step L4 gravity-energy drift from `7.276e-7` to `8.746e-7` and doubled state storage; remove it. Keep the native conservative state transfer and the old global residual correction removed. Treat AMR hierarchy-change energy error as a convergence quantity. [DONE]
  - Make the final self-gravity AMR/subcycling acceptance test 2D and exercise at least L3, preferably L4-L7, with repeated substeps/regrids and remaining level headroom. L1 localization checks are not acceptance coverage. A matched uniform `256x256` control conserves gravity total energy to `2.16e-14`. At matched `t=0.02`, native L4 AMR gives `2.47e-7` non-subcycled and `1.46e-6` subcycled energy drift with four levels of headroom; subcycled/non-subcycled rho and pressure agree to maximum relative L2 `4.46e-4`. The first subcycled L4 run exposed a test-only `1e-12` MLMG tolerance below the five-level periodic solve floor (`1.43e-12`); the registered `1e-10` tolerance converges. Keep subcycled self-gravity experimental because it is less conservative here and provides no measured throughput gain. [DONE]

## Cycle 104 [completed]
- Freeze the coherent AMReX parabolic/elliptic state with a commit and a performance snapshot that reports both cell throughput and simulation-time throughput. [DONE]
  - Audit the dirty tree and stage only coherent AMReX production, validation, documentation, and generated-kernel changes; preserve unrelated JAX/Numba/reference work. [DONE]
  - Capture the maintained benchmark plus representative AMR subcycling comparisons with both Mcells/s and simulation-time/real-time. Freeze the sector gates and benchmark under `runs/baselines/20260713_131150_parabolic_elliptic_subcycling_20260713/`, including the raw standard-L3 and deep-auto-L5 subcycling TSVs. Standard KH keeps Mcells/s effectively unchanged (`2.902` to `2.909`) while simulation-time throughput improves `1.136x`. Deep-auto KH changes Mcells/s from `1.912` to `1.863` while aligned local-CFL improves simulation-time throughput `1.349x`; the two-flow mean physical-time speedup is `1.242x`. [DONE]
  - Reconfirm the explicit/implicit viscosity and resistivity decay-rate evidence, combined damping result, and the ideal/nonresistive status of Brio-Wu reference tests. Analytic Fourier-amplitude errors are `6.74e-6`/`4.62e-5` for explicit/implicit hydro viscosity, `8.89e-5`/`7.23e-5` for 2D explicit/implicit CT resistivity, and `8.20e-5` for 3D implicit CT resistivity. Combined viscous-resistive Alfvén damping agrees with `exp[-0.5(nu+eta)k^2t]` to `4.51e-3` explicit and `5.24e-3` implicit. Deep AMR/subcycling, CT/divB, curvilinear explicit smoke, EOS combinations, and restart cover integration, but an AMR parabolic convergence study and nonlinear viscous exact solution remain stronger future tests. Brio-Wu is an ideal-MHD numerical-reference test with diffusion disabled and `nu=eta=0`. [DONE]
  - Commit the validated coherent AMReX change set while leaving unrelated reference implementations unstaged. [DONE]

## Cycle 105 [completed]
- Clarify the current AMR support boundary of the implicit parabolic sector from the implemented solver paths and maintained validation cases. [DONE]
  - Hydro implicit viscosity supports multilevel AMR with and without subcycling through level-local implicit solves, shared generated viscous fluxes, FillPatch coarse-fine boundary data, flux-register reflux, and average-down. It is not a single composite multilevel implicit solve. [DONE]
  - Implicit CT resistivity remains restricted to one-level periodic Cartesian 2D/3D because AMReX 25.11 `MLCurlCurl` lacks the required multilevel backend. Explicit viscosity and explicit CT resistivity retain the full validated AMR/subcycling path. [DONE]

## Cycle 106 [completed]
- Assess the next development direction after consolidating the Cartesian AMReX solver and encountering diminishing returns from curvilinear multipatch AMR. [DONE]
  - Pause mixed curvilinear multipatch evolution unless a concrete science problem requires it; isotropic AMReX refinement weakens the expected radial-efficiency benefit and mixed-patch conservative seam evolution remains a substantial independent project. [DONE]
  - Treat incompressible flow as a separate projection-based physical system, not an EOS toggle. It can reuse AMR, MLMG, I/O, diagnostics, and generated differential operators, but requires a predictor, pressure projection, divergence constraint, boundary treatment, and AMR synchronization. [DONE]
  - The lowest-risk next phase is deeper Cartesian single-patch validation and production runs, followed by a deliberate choice between nonrelativistic multiphysics and a new relativistic physical-system module. A tabulated EOS is a comparatively contained extension useful to either direction. [DONE]

## Cycle 107 [completed]
- Produce a bounded Cartesian HD/MHD capstone and close the remaining AMR parabolic validation gap before beginning relativistic-system work. [DONE]
  - Audit existing production inputs and rendering infrastructure. Retain the validated DMR evolution as the HD capstone/control; no maintained non-ideal current-sheet/reconnection IC exists. [DONE]
  - Add a generic analytic periodic current-sheet MHD initial condition and focused smoke/physics validation without introducing a feature-specific evolution branch. The IC builds a discretely divergence-free CT field from one analytic vector potential; the focused AMR+CT+resistive feature gate passes. [DONE]
  - Run a semiproduction explicit-resistive CT evolution with aligned local-CFL AMR, characteristic WENO-Z, HLLD, SSPRK3, compressed HDF5 state/aux output, and three MPI ranks. The low-eta 1024-square control reached L5 but was physical-length-limited; retain it only as the high-resolution movie/control. A corrected base-16, nu=2e-4, eta=1e-3 probe reached metric-limited L8 with four levels of headroom and 23.3% of the total cell budget. Update the maintained input to that physics-resolved setup. [DONE]
  - Validate conservation, divB, AMR tracking/headroom, current-sheet physics, and both throughput metrics; render run-local density/current movies with and without AMR boxes. The control conserves mass/energy to 8.33e-14/5.24e-14, has divB=1.92e-12, converts 3.42% of magnetic energy by t=0.6, and runs at 2.125 Mcells/s and 5.97e-4 simulation-time/real-time. The metric-limited L8 probe conserves mass/energy to 2.12e-14/1.34e-14 with divB=5.99e-13. Generalize the renderer to discover cell/face/edge-centered GF timelines by schema component; density and live edge-centered j3 movies render successfully with and without boxes. [DONE]
  - Add or run an AMR parabolic convergence comparison and freeze the capstone outputs if all focused gates pass. A two-level fixed-interface shear-wave control covers explicit/implicit viscosity, SSPRK3, subcycling, reflux, and quartic conservative transfer at N=32,64,128,256. Explicit reaches order 1.95 (second-order spatial path); backward-Euler split implicit reaches order 1.24 (expected first-order temporal path). Full 1D/2D/3D builds and the focused current-sheet gate pass. The quick suite remains 17/18 with only the pre-existing pulsating moving-cut-cell mass-ledger failure. [DONE]
  - Capture the maintained performance snapshot after the CT regrid correction and new IC dispatch. The four maintained cases pass at 109.03, 130.50, 5.99, and 0.18 Mcells/s, or 0.99, 0.95, 0.94, and 0.96 of the frozen baseline. [DONE]

## Cycle 108 [completed]
- Verify and update the compact showpiece archive after the Cartesian capstone. [DONE]
  - Confirm `runs/final_done/` is the canonical AMReX curated folder; retain `pyfluid/final_done/` only as the older reference-path archive. [DONE]
  - Add the four resistive-current-sheet density/current movies, scalar plot, compact long-run records, metric-limited L8 acceptance records, and AMR parabolic convergence artifacts without copying raw HDF5 grid data. [DONE]
  - Document the distinction between the physical-resolution-limited long movie control and the metric-limited AMR acceptance probe. [DONE]

## Cycle 109 [completed]
- Audit why the resistive-current-sheet capstone is less visually structured than the archived short MHD KH run. [DONE]
  - Compare representative final frames, plotting scales, initial conditions, dissipation, perturbations, AMR histories, and evolution times. [DONE]
  - Conclude that the current-sheet run is a clean two-mode tearing/conservation control rather than a turbulence showcase: zero initial shear, one imposed wave number, exact periodic symmetry, two identical sheets, finite explicit dissipation, and insufficient nonlinear time suppress a plasmoid/eddy cascade. [DONE]
  - Record that linear signed-j3 rendering further hides weak filamentary structure, whereas the archived KH movie uses log10 current magnitude and a strongly shear-unstable state. Keep the numerical validation, but do not present the current-sheet control as the strongest visual capstone. [DONE]

## Cycle 110 [completed]
- Produce a genuinely high-structure Cartesian non-ideal MHD capstone. [DONE]
  - Run a validated single-sheet magnetic KH problem with characteristic WENO-Z, HLLD, SSPRK3, CT, aligned local-CFL subcycling, explicit viscosity/resistivity, compressed HDF5, and deep AMR. The t=1.2 run completes in 248.8 s and exposes filamentary current structure, but its broad single shear layer remains a weak visual capstone. [DONE]
  - Add canonical Cartesian Orszag-Tang initial data plus a discretely divergence-free CT face field from one analytic vector potential. Add a focused characteristic-WENO/HLLD/CT feature-matrix case and a generic cell-state-derived `current3` plotting diagnostic. [DONE]
  - Diagnose the first deep-AMR/CFL-0.9 attempt honestly: it forms negative-pressure cells at t=0.293651 as strong shocks develop. A uniform 256-square control at CFL 0.5 reaches t=0.4 with positive states and roundoff conservation. Orszag-Tang has domain-wide structure, so an almost full AMR hierarchy is slower than the equivalent uniform grid; use a 512-square uniform production mesh rather than pretending AMR helps this case. [DONE]
  - Complete the 512-square non-ideal MHD capstone through t=1.0 on three ranks with 5,075 SSPRK3 steps and 113 compressed frames. It takes 672.8 s, sustains 2.008 Mcells/s and 1.503e-3 simulation-time/real-time, and finishes with mass/energy/momentum drift 3.46e-14/7.28e-14/4.21e-16 and divB 8.73e-13. [DONE]
  - Render density, pressure, and smooth log10 |J3| movies, inspect the nonlinear final state, and curate checksum-verified media plus compact records under `runs/final_done/`. Build 1D/2D/3D, pass the focused new-IC gate, and retain the quick suite's sole pre-existing pulsating moving-cut-cell ledger failure. The maintained benchmark passes at 106.65, 126.29, 5.99, and 0.18 Mcells/s. [DONE]

## Cycle 111 [completed]
- Produce mobile-safe copies of the curated Orszag-Tang movies after local 13-25 MB GIF links fail in the Codex mobile app. [DONE]
  - Retain all 113 simulation frames and the 7.91-second duration while reducing the final mobile raster to 480x413 and the palette to 64 colors. The density, pressure, and log-current copies are 4.7, 4.4, and 6.9 MB and were attached directly. [DONE]

## Cycle 112 [completed]
- Produce a higher-structure, AMR-relevant non-ideal MHD capstone without diagnostic-driven refinement or presentation. [DONE]
  - Add one pressure-balanced central reconnecting shear-layer initial condition with CT-consistent magnetic faces and localized multimode tearing/KH perturbations. The implementation is a normal MHD initial-data provider; it introduces no feature-specific evolution branch. [DONE]
  - Fix two general CT hierarchy defects exposed by the new case. Every exact-time-zero AMR level now receives the analytic divergence-free CT face field instead of interpolating an unnecessarily noisy fine field from coarse data. Compressed-checkpoint restart now preserves valid face fields and rebuilds omitted coarse-fine/physical face ghosts through the standard divergence-preserving CT FillPatch path before any resistive curl. [DONE]
  - Make same-directory restart diagnostics append only mutually consistent histories that do not extend beyond the selected checkpoint; insert the checkpoint row when scalar and checkpoint cadences differ. Strengthen the maintained CT restart control to L3 aligned subcycling with explicit resistivity. The uninterrupted/restarted maximum field difference is `3.06e-12` after two continued coarse steps. [DONE]
  - Run the production configuration at CFL `0.9` through `t=1.5` on three MPI ranks with characteristic WENO-Z, HLLD, SSPRK3, CT, aligned local-CFL subcycling, and `nu=eta=5e-5`. It remains physical (`min(rho)=0.468`, `min(p)=0.165`) with `divb=5.77e-13`. Open-boundary mass and energy ledger residuals are `8.29e-14` and `8.14e-7`; raw total changes are boundary transfer, not closed-system drift. The late hierarchy uses L6 of an allowed L12 and 40.3% of the cell budget, but reaches the configured `5e-4` physical-resolution floor, so it is not claimed as metric-limited. [DONE]
  - Retain 74 compressed state/current snapshots and checkpoints in the 4.5-GB run folder. Render density and true edge-centered `log10(abs(j3))` movies with and without AMR boxes. The current movies show multiple thin folded layers and island-like substructure materially richer than the prior laminar sheet control; curate the four suffix-003 movies plus compact reports under `runs/final_done/`. [DONE]
  - Build 1D/2D/3D, pass Brio-Wu AMR+CT (`2.13x` improvement over uniform64; rotated agreement about `4e-15`), and pass the focused L3 resistive CT restart gate. The quick suite remains 17/18 solely because of the pre-existing pulsating moving-cut-cell ledger defect. The maintained benchmark passes at `106.54`, `127.42`, `6.01`, and `0.18 Mcells/s`, or 0.93-0.97 of the frozen baseline. [DONE]

## Cycle 113 [completed]
- Make the shared AMR movie renderer produce compact, high-quality output without per-run visual tuning. [DONE]
  - Size figures from the physical domain aspect and attach the colorbar to the data axes so it never creates a mostly-empty canvas. Wide reconnecting-shear, 4:1 DMR, and square pressure-bump frames all retain equal physical aspect with the colorbar exactly matching the data-axis height. [DONE]
  - Preserve continuous high-DPI RGB master frames, use one fixed movie-wide normalization anchored to a configured initial-state reference when applicable, and replace per-frame adaptive GIF palettes with one stable colormap-rich palette. Derived/log fields keep one physical movie-wide scale rather than dividing by a zero or nonuniform initial field. [DONE]
  - Let the renderer discover state versus auxiliary GF ownership from the run directory, add full-color H.264 MP4 output, and cap only the compatibility GIF to the existing pixel budget after rendering. Rerender all 74 retained reconnecting-shear snapshots with and without AMR boxes: MP4 outputs are 0.95-4.7 MB and 1000-pixel GIFs are 7.8-13 MB, down from 21-31 MB at uncapped high-DPI size. Curate both formats under `runs/final_done/media/reconnecting_shear_layer/`. [DONE]
  - Pass Python compilation, direct run-root state/edge-aux discovery, two-frame GIF/MP4 assembly, square/wide-domain visual inspection, and the normal quick suite. The suite remains 17/18 solely because of the tracked pulsating moving-cut-cell mass-ledger defect (`1.156e-7`), unrelated to plotting. [DONE]

## Cycle 114 [completed]
- Correct the shared movie presentation policy after the first compact-renderer pass over-clips the reconnecting-shear fields. [DONE]
  - Define one shared math-mode label source for standard state and auxiliary fields, including `P`, `rho`, vector subscripts, magnitudes, normalized references, MHD fields, gravity fields, and derived diagnostics. Use it in native AMR movies and the general slice plotter. [DONE]
  - Remove the concentration-centering heuristic and replace it with asymmetric movie-wide limits. Linear fields suppress at most the lowest 0.1% tail and retain the 99.99% high tail; log-magnitudes suppress the uninformative lowest 5% while retaining the same high tail. Preserve each sampled tile's exact extrema without giving them percentile weight, then include nearby extrema automatically while giving isolated extrema small headroom. On the retained run this expands density from about `0.87-1.13` to `0.64-1.32` and log-current from about `-2.9-1.0` to `-2.91-2.25`. [DONE]
  - Remove GIF support and its Pillow/palette/downsampling implementation entirely. MP4 is the sole movie product; RGB PNG masters remain available with `--keep-frames`. Remove all superseded reconnecting-shear GIFs and clipped suffix-004 MP4s, then curate only four corrected suffix-005 MP4s totaling 9.4 MB. [DONE]
  - Compile all modified Python tools; render and inspect full 74-frame density/current movies with and without boxes; verify MP4 frame count, math labels, run-root field discovery, wide DMR scaling, and normalized pressure rendering. The quick suite remains 17/18 solely because of the tracked pulsating moving-cut-cell mass-ledger defect (`1.156e-7`), unrelated to plotting. [DONE]

## Cycle 115 [completed]
- Quantify how physically developed and computationally efficient the reconnecting-shear capstone is. [DONE]
  - The run reaches L6 at `dx=4.8828125e-4`, with 1.210M active cells, 40.3% of its cell budget, and six levels of headroom. Its finest level is stopped by the configured `5e-4` physical-length guard while the state metric still requests refinement, so the macroscopic instability remains early but its strongest sheets are already locally resolution-limited. [DONE]
  - Profile the post-restart interval: flux/reconstruction/Riemann work is 68.0% of wall time, explicit viscosity/resistivity 8.7%, regridding 7.7%, CT 4.7%, and output only 0.6%. The full run costs about 47 minutes and 5.4 billion active-cell advances; frequent compressed output is not the bottleneck. [DONE]
  - Measure the maintained JAX baselines directly. At 128-square, WENO-Z/RK4/LLF/CT gives `1.926 Mcells/s`; WENO-Z/SSPRK3/LLF/CT plus diffusion gives `1.681 Mcells/s`. The production AMReX run gives `1.953 Mcells/s` despite characteristic WENO-Z, HLLD, CT, explicit viscosity/resistivity, and AMR. At equivalent finest resolution, AMR plus subcycling avoids roughly 12-14x of the late uniform-grid work; the old JAX CT path also remains a performance reference rather than a correctness reference. [DONE]
  - Prefer a postprocessed dissipation-resolution diagnostic over mass/volume at the smallest scale: compute kinetic and magnetic dissipation spectra plus dissipation-weighted local-scale CDFs from `2 rho nu S:S` and `eta J^2`, and report the fraction of dissipation in the finest resolved octave and the cutoff relative to Nyquist. Statistical saturation over several turnover times, with spectra rolling off before the grid cutoff, is the useful completion criterion. [DONE]

## Cycle 116 [completed]
- Replace the conservative 2D AMR cell-budget assumption with a measured Mac mini capacity estimate. [DONE]
  - Separate hierarchy allocation from unique composite coverage. The retained L6 reconnecting-shear hierarchy contains 1,209,600 per-level valid cells but 907,712 unique composite cells after subtracting fine-covered coarse cells; only 577,792 are on L6. Its 2:1 movie raster also makes many fine cells subpixel, so it does not visually resemble a uniform million-cell image. [DONE]
  - Run controlled three-rank MHD/CT/characteristic-WENO-Z/SSPRK3 memory probes without field output. Natural continuation reaches L7 with 2,185,216 hierarchy cells at 2.58 GiB sampled aggregate RSS and L8 with 2,840,064 cells at 3.04 GiB. Fixed-hierarchy load tests complete at 5,851,136, 9,906,176, and 11,184,128 cells. The largest case has a 2.39-GiB maximum rank RSS, so three times that value gives a conservative 7.16-GiB aggregate upper bound, with no swap and roundoff conservation. [DONE]
  - Use 12,000,000 total hierarchy cells as the normal 2D production guard on this 16-GiB Mac mini. An 11.18M-cell full-domain hierarchy is demonstrated for evolution allocations; 12M still leaves several GiB for compressed output/checkpoint buffers, the OS, and concurrent work. This remains a resource guard, not a refinement target. [DONE]
  - Update the reconnecting-shear capstone input to the measured 12M total guard and a nonbinding 4096-square per-level guard. [DONE]
  - The retained run was not constrained by its old 3M budget: it stopped at the configured `5e-4` feature-length floor. A continued run must lower that floor to create L7/L8; measured probes reach L7 at `2.5e-4` and L8 at `1.25e-4`, while keeping max-level headroom. [DONE]

## Cycle 117 [completed]
- Prepare a longer reconnecting-shear continuation that uses the measured Mac mini capacity without an artificial physical-resolution stop. [DONE]
  - Raise the hierarchy resource guard to 12M cells while retaining operating-system and output headroom. The 11.18M-cell load test completed in 7.16 GiB by the conservative three-rank upper bound. [DONE]
  - Keep `min_feature_length` only as an emergency physical-resolution guard. An L9 probe at `6.25e-5` reached 3.014M cells, 25.1% of budget, and required 349 seconds for one coarse step because of 512-way fine-level subcycling; use `1e-5` with L15 headroom so normal refinement is metric- or resource-limited instead. [DONE]
  - Audit the existing level-staged reconstruction support. Its generic two-method transition is sufficient, but deep-level work dominates this hierarchy, so staging PWL below characteristic WENO-Z would save only a few percent. Retain one full characteristic WENO-Z path rather than adding a three-stage branch. [DONE]
  - Retain explicit `nu=eta=5e-5`: the strongest retained current sheets are only a few L6 cells wide, so deeper AMR can expose more structure, while zero resistivity would turn reconnection into a grid- and reconstruction-dependent numerical effect. [DONE]
  - Smoke-test the retained L6 checkpoint on three MPI ranks with the production input's L15 headroom, `1e-5` emergency floor, 12M-cell guard, characteristic WENO-Z, HLLD, SSPRK3, CT, and explicit dissipation. Restart loading completes without modifying the retained run. Do not launch the long continuation implicitly: the measured L9 coarse step already costs 349 seconds, so the requested duration must be chosen with that wall-time cost visible. [DONE]

## Cycle 118 [completed]
- Make the reconnecting-shear physical-length guard effectively unreachable in normal production and account for the measured deep-AMR work. [DONE]
  - Lower `amr.min_feature_length` to `1e-6`. With the 64-cell base over the length-2 axis, L15 has `dx=9.54e-7`; therefore the physical floor is effectively coincident with the emergency maximum level and well below useful movie resolution. The 12M-cell resource guard should intervene first. [DONE]
  - Profile the 349-second L9 coarse step: flux/reconstruction/Riemann work is 56.2%, regridding 19.9%, explicit viscosity/resistivity 7.8%, boundary fill 5.2%, CT 4.0%, divergence 2.6%, average-down 1.7%, and primitive caching 1.2%. Cell-count times subcycle factor places about 87.5% of stage work on L7-L9, so characteristic WENO-Z on the deepest levels is the main cost, but not the only one. [DONE]

## Cycle 119 [completed]
- Add production-representative performance coverage and a compact periodic advection medley. [DONE]
  - Inventory the shared initial-data, validation, benchmark, launcher, and output-reader paths; extend them rather than adding parallel implementations. [DONE]
  - Add one exact periodic Euler contact-advection initial condition containing sinusoidal, square, triangular, and Gaussian features with clean segment joins. [DONE]
  - Add a cheap reconstruction comparison that reports feature-wise errors, global norms, conservation, and both throughput metrics after one or more periods. After two periods at N=256, WENO-Z gives global normalized L1 error 1.50e-2 versus 4.07e-2 for PWL and 3.18e-1 for PWC. Component and characteristic WENO-Z agree to roundoff on the pure contact; component is 13.6% faster in simulation-time/real-time. All methods conserve mass and energy to about 5-6e-14. [DONE]
  - Extend rank scaling to deep AMR plus aligned subcycling with production WENO-Z/component and characteristic-WENO-Z cases; track speedup, parallel efficiency, hierarchy equivalence, Mcells/s, and simulation-time/real-time. Both cases reach identical per-method L5 hierarchies across 1/2/3/4 ranks. Three-rank efficiency is 87.9% for component and 90.8% for characteristic WENO-Z; four-rank efficiency is 79.2% and 82.4%. [DONE]
  - Add one minimal machine-profile build/run/submit interface for reproducible local and HPC execution without bypassing the canonical MPI runner. Machine TOML supplies build paths, launcher syntax, environment, and optional scheduler command; simulation parameters remain exclusively in the run input. [DONE]
  - Run focused builds, validation, and bounded performance probes. The 1D/2D/3D executables build. The maintained benchmark remains at 109.17/129.89 Mcells/s for 3/4-rank 2D PWC, 5.99 Mcells/s for 3D PWC, and 0.18 Mcells/s for 3D WENO-Z/RK4, all 0.94-0.99 of the frozen baseline. The quick suite remains 17/18 solely because of the pre-existing pulsating moving-cut-cell mass-ledger failure at 1.156e-7. [DONE]

## Cycle 120 [completed]
- Determine a coupled reconstruction test that distinguishes component from characteristic WENO-Z, and audit whether splitting the AMReX state can improve hot-path performance. [DONE]
  - Confirm that the current contact-advection medley contains only an entropy/contact family, so component and characteristic WENO-Z should agree. The existing Double Mach reconstruction scan supplies the coupled-shock counterexample: characteristic WENO-Z has an aggregate high-resolution-reference error ratio of 0.838 relative to component WENO-Z. [DONE]
  - Identify a cheap periodic extension: prescribe distinct, overlapping small-amplitude medley profiles in the left-acoustic, entropy, and right-acoustic characteristic families, then compare return error after the acoustic families traverse the periodic domain. Retain Double Mach or Shu-Osher as the strong nonlinear discriminator. [DONE]
  - Audit storage and accesses. AMReX `BaseFab`/`Array4` is already component-major, with each component occupying one contiguous cell block. The solver uses one state/primitive/flux `MultiFab` per centering and templates hydro versus MHD component counts; face magnetic fields, edge fields, gravity, and other differently owned data are already separate. [DONE]
  - Do not split the coupled conserved state into per-variable `MultiFab`s: it would not improve payload locality for WENO stencil streams, while it would multiply box metadata, ghost fills, FillPatch/interpolation, reflux, and kernel traversal. If profiling justifies layout work later, test a flux-kernel-local AoSoA/stencil staging experiment rather than changing persistent AMR state ownership. [DONE]

## Cycle 121 [completed]
- Add a coupled wave-crossing medley, remove Python from the scheduled HPC run plane, and evaluate focused WENO-Z optimizations. [DONE]
  - Keep the exact contact-advection medley and add a separate periodic hydro characteristic-family medley with distinct overlapping left-acoustic, entropy, and right-acoustic profiles. Score family-wise return errors, global state errors, conservation, Mcells/s, and simulation-time/real-time for component and characteristic WENO-Z. At amplitude 0.1, characteristic reconstruction reduces aggregate fine-reference error by 17.2% relative to component WENO-Z while costing 11.0% throughput; both conserve mass and energy to about 1.4e-14. [DONE]
  - Compare the current machine helper with local SimFactory machine, optionlist, runscript, and submit-script responsibilities. Retain a compact PyFluid interface, but generate a plain-shell scheduled runscript that requires no Python or `uv` on compute nodes. The dry-run script passes shell syntax and invokes MPI plus the dimension-selected AMReX executable directly; Python remains a login-side build, launch, and reporting convenience. [DONE]
  - Profile/audit component and characteristic WENO-Z stencil work, implement only general local reuse that preserves the single reconstruction implementation, then A/B accuracy and speed on the coupled medley and maintained benchmarks. [DONE]
    - Compile-time characteristic-component selection is performance-neutral because Clang already folds the literal branches. Hoist density/sound/MHD-basis reciprocals into one generated per-face basis: this removes repeated divisions explicitly but is also performance-neutral on the current CPU, so treat it as compiler/GPU portability cleanup rather than a Mac speed claim. [DONE]
    - Evaluate the published WENO-Z power-1 candidate without retaining a second production branch. It reduces component mixed-characteristic error from 1.629e-2 to 1.235e-2 and exact-contact error from 1.50e-2 to 1.36e-2, but characteristic projection adds only another 2.3% and power 1 loses fifth order at smooth first-order critical points. Restore the power-2 default, which preserves fifth order there and gives the stronger characteristic/component discriminator. [DONE]
  - Build and run focused validation before updating documentation and closing the cycle. Generated-kernel validation and 1D/2D/3D builds pass. The final power-2 medleys pass, including the maintained 17.2% characteristic accuracy gain. The deep L5 production benchmark reaches 1.958 Mcells/s and 7.941e-3 simulation-time/s for component WENO-Z versus 1.676 Mcells/s and 6.689e-3 simulation-time/s for characteristic WENO-Z, with 92.6% and 93.4% three-rank efficiency. The maintained benchmark passes at 107.71, 128.80, 5.97, and 0.18 Mcells/s, or 0.94-0.98 of baseline. The quick suite remains 17/18 solely because of the pre-existing pulsating moving-cut-cell ledger failure. [DONE]

## Cycle 122 [completed]
- Add MP5 and TENO5 as compact five-point reconstruction alternatives and measure them against WENO-Z. [DONE]
  - Extend the single reconstruction selection interface with component and characteristic MP5/TENO5 variants; do not duplicate physics-, dimension-, or coordinate-specific implementations. One compile-time five-point policy now feeds the existing hydro/MHD, component/characteristic, mapped-patch, and positivity paths. [DONE]
  - Use RK4 medley runs to reduce temporal contamination and compare feature-wise error, characteristic-family error, conservation, Mcells/s, and simulation-time/real-time. On the exact two-period contact medley, MP5 component lowers normalized L1 error from WENO-Z component's 1.480e-2 to 1.254e-2 and runs at 7.97 versus 6.18 Mcells/s. On the coupled-family medley, MP5 component gives the lowest aggregate error, 1.043e-2, versus 1.351e-2 for characteristic WENO-Z; characteristic MP5 changes that by only 0.2% while costing throughput. [DONE]
  - Verify fifth-order smooth convergence and robust shock behavior before making any default recommendation. All six WENO-Z/MP5/TENO5 component and characteristic variants recover fifth order through N=256. Exact Sod passes for every variant; MP5 component reduces total L1 error by 28% and is 21% faster than component WENO-Z. In the 2D AMR Double Mach comparison, MP5 component reduces aggregate fine-reference error by 35% and is 8% faster than component WENO-Z; MP5 characteristic gains only another 3.5% accuracy while losing 18% throughput. MHD+CT feature cases pass for all four new variants. [DONE]
  - Build all executable dimensions, run focused validation and performance checks, document measured tradeoffs, and close the cycle. The 1D/2D/3D executables build. The maintained benchmark passes at 103.46, 126.28, 5.97, and 0.18 Mcells/s, or 0.92-0.96 of the frozen baseline. The quick suite remains 17/18 solely because of the pre-existing pulsating moving-cut-cell mass-ledger failure at 1.156e-7. Retain WENO-Z as the compatibility default for now; MP5 component is the measured accuracy/performance recommendation for new hydro runs. [DONE]

## Cycle 123 [completed]
- Close the reconstruction comparison with an MHD-specific discriminator and prepare the symbolic layer for generated GRMHD. [DONE]
  - Inventory the existing MHD wave initial data, reconstruction validation, AMR/subcycling controls, and symbolic PDE-to-kernel boundary. The existing generated seven-wave projection/recovery is the correct source for a mixed-mode IC; the remaining GRMHD gap is indexed variance/metric algebra rather than another runtime code path. [DONE]
  - Add one compact mixed MHD-mode medley through shared initial-data and validation infrastructure; compare component and characteristic WENO-Z/MP5/TENO5 using RK4. The IC uses the generated inverse MHD characteristic transform on a nondegenerate oblique-field background and excites all seven families with offset compact profiles. [DONE]
  - Exercise the same problem with AMR plus subcycling and verify conservation, hierarchy use, solution agreement, and both throughput metrics. The L3 MP5 control conserves mass/energy to roundoff, keeps `divB=0`, and subcycling agrees with single-rate evolution to `3.21e-4` relative to the perturbation amplitude. It is a compatibility control rather than a speed case because structure occupies most of the periodic domain. [DONE]
  - Tighten and document the symbolic abstractions actually required to express GRMHD in high-level indexed form without adding an unexercised second solver implementation. Extract the characteristic basis into one shared module and document the checked index/variance, metric, 3+1 projection, generated residual/Jacobian, and validation contract. Defer an indexed tensor implementation until the first generated GRMHD equation consumes it. [DONE]
  - Run focused generated-kernel checks, builds, validation, and performance gates; close the cycle with measured recommendations and known limits. Generated flux/Riemann/EOS/characteristic/operator validation passes, as do fresh 1D/2D/3D builds. On the seven-family MHD medley, MP5 component lowers aggregate normalized error by 20.5% and is 5.9% faster than component WENO-Z; MP5 characteristic lowers error by another 13.4% at a 21% throughput cost, making MP5 component the default accuracy/performance recommendation and MP5 characteristic the accuracy choice. All uniform runs conserve mass/energy to roundoff with zero one-dimensional magnetic divergence. The L3 AMR control reaches all requested levels and remains conservative; aligned subcycling is 7% slower in simulation-time/real-time for this nearly domain-filling 1D problem, so it is retained as a compatibility check rather than a speed claim. The quick suite remains 17/18 solely because of the pre-existing pulsating moving-cut-cell ledger failure (`1.156e-7`). [DONE]

## Cycle 124 [completed]
- Solidify AMReX module and compilation boundaries before adding generated GRMHD. [DONE]
  - Audit the current AMR driver, physics/numerics headers, generated-kernel dependencies, and CMake targets; identify real rebuild coupling rather than adding cosmetic libraries. The actual coupling is the concrete AMR class header embedding every evolution template; ordinary CMake static-library grouping would not isolate those header dependencies. [DONE]
  - Replace the public exposure of the concrete `AmrSolver` implementation with one small run interface and an internal driver header. Preserve behavior and ownership. [DONE]
  - Define and document the shared physical-system/kernel-bundle contract for HD, MHD, and future GRMHD, including what remains header-only for GPU inlining and what can be compiled independently. Group CMake sources by driver, AMR, and runtime ownership without changing compilation semantics. [DONE]
  - Centralize stable `HD`/`MHD`/`GRHD`/`GRMHD` identifiers, capabilities, and fluid/spacetime evolution-sector semantics. Keep component layout separate and retain one shared fluid/numerics path; magnetism augments the fluid sector while fixed-metric Cowling evolution simply leaves the spacetime sector unscheduled. [DONE]
  - Record the generated relativistic-fluid architecture, AsterX/Cottonmouth/SpacetimeX reference responsibilities, ordered C2P policy, Cowling/evolved-metric sector contract, and executable validation ladder. WIP vertical slices remain runnable instead of accumulating temporary guards or fallback implementations. [DONE]
  - Add GRHayL/GRHayLET/IllinoisGRMHD and Kastaun's RePrimAnd as independent equation, integration, EOS, and recovery references. Seed the C2P policy with the GRHayL ordering Noble2D, Palenzuela1D, Noble1D entropy, Palenzuela1D entropy, and Font1D; benchmark RePrimAnd before assigning its production position. [DONE]
  - Split hierarchy/storage lifecycle into `AmrSolverBase` and keep per-stage RHS/reconstruction/Riemann/time-integration templates in the private derived evolution shard. Hierarchy repair and gravity diagnostics cross one narrow lifecycle-hook boundary and never dispatch inside cell kernels. Remove the unnecessary `gravity.hpp -> rhs.hpp` dependency and declare the actual mask dependency directly. Fresh compiler dependency files show that `amr_solver_driver.cpp` is the only current AMR source depending on `reconstruction.hpp` in every executable dimension. [DONE]
  - Verify generated kernels, fresh and no-op builds, quick validation, and maintained performance before closing the cycle. Generated-kernel validation and fresh 1D/2D/3D builds pass; the all-dimension no-op build takes 0.48 s. The quick suite remains 17/18 solely because of the tracked pulsating moving-cut-cell mass-ledger defect (`1.156e-7`). Maintained benchmarks pass at 105.10, 130.02, 5.95, and 0.18 Mcells/s, or 0.93-0.96 of the frozen baseline. [DONE]

## Cycle 125 [completed]
- Add the first generated Valencia relativistic-fluid vertical slice. [DONE]
  - Audit the existing symbolic tensor algebra, physical-system model, generated-kernel emitter, and AsterX/GRHayL Valencia conventions; choose one explicit densitization and velocity convention. [DONE]
  - Express GRHD primitive-to-conserved variables and directional conservative fluxes from high-level indexed 3+1 equations, then emit one optimized C++ kernel family without direction-specific handwritten formulas. [DONE]
  - Validate Minkowski analytic identities, Newtonian limits, arbitrary spatial metrics, generated-header reproducibility, and C++ compilation before exposing the bundle to runtime dispatch. Randomized positive-definite metric comparisons pass against an independent numeric Valencia reference in all directions; the Newtonian energy limit converges at the expected fourth-order remainder; generated headers are reproducible; and compile checks pass in 1D/2D/3D. [DONE]
  - Define the shared C2P candidate result/status interface and ordered policy metadata for Noble2D, Palenzuela1D, Noble1D entropy, Palenzuela1D entropy, Font1D, and the separately benchmarked RePrimAnd candidate; do not implement terminal atmosphere as a recovery method. [DONE]
  - Regenerate and build all AMReX executable dimensions. The no-op solver build remains 0.5 s. The runtime quick gate remains 17/18 solely because of the pre-existing pulsating moving-cut-cell mass-ledger failure (`mass_rel_drift=1.156e-7`, `external_mass_rel_error=4.588e-7`); this cycle does not touch the runtime evolution path. While executing the aggregate generated C++ checker, correct its stale Thornburg reference formula to test the existing blended square-to-sphere map rather than an unrelated pure spherical radius. [DONE]

## Cycle 126 [completed]
- Establish one two-sided primitive/conserved round-trip validation contract before implementing relativistic C2P. [DONE]
  - Exercise existing Newtonian HD and MHD P2C/C2P in both orders over deterministic messy admissible states and every analytic EOS. The corpus independently perturbs conserved momentum and energy for nonbarotropic EOSs; polytropic states remain on the barotropic EOS manifold. [DONE]
  - Define a symmetric scaled maximum component residual and a precision-aware `256 epsilon` acceptance bound. The worst Newtonian residual is `3.492e-14` for hybrid-EOS pressure recovery after subtracting magnetic energy about 400 times larger; all other conserved round trips are at or below `1.19e-14`. The same two-sided residual definition is the validation contract for iterative GRHD/GRMHD recovery. [DONE]
  - Add the GRHD and GRMHD recovery corpus requirements to the development contract; do not claim a relativistic round trip until an actual C2P method supplies it. [DONE]
  - Build and run the focused generated C++ gate in 1D, 2D, and 3D. All pass. [DONE]

## Cycle 127 [completed]
- Begin the ideal relativistic-hydrodynamics implementation on a fixed Minkowski spacetime. [DONE]
  - Rename the relativistic fluid identities to RHD/RMHD and define the valid none/fixed/evolved spacetime compatibility matrix without changing Newtonian runtime behavior. Internal enums, state types, generated functions, compile checks, and documentation now use one RHD/RMHD convention without aliases. `HD/MHD` accept only `none`; `RHD/RMHD` accept `fixed` or `evolved`. [DONE]
  - Separate initial atmosphere floors, evolved-state admissibility, ordered C2P fallback, and terminal failed-recovery repair in the design contract. Initial ambient atmosphere belongs to initial-data construction; C2P methods never change the conserved state; exhaustion means invalid conserved data; validation aborts and any future production repair is explicit and separately counted. `Y_e`, temperature, and entropy ownership are also recorded without allocating unused ideal-gas fields. [DONE]
  - Extend the generated EOS/P2C algebra with ideal-gas relativistic enthalpy and sound speed. One analytic EOS object now emits pressure, specific enthalpy, and causal sound-speed kernels for ideal RHD. [DONE]
  - Generate a one-dimensional SRHD recovery residual/Jacobian, implement one bracketed recovery method behind the common C2P result interface, and validate both round-trip directions over a noisy admissible corpus. SymPy emits the Palenzuela `x=hW` residual, analytic derivative, and primitive reconstruction. C++ owns a bracket-preserving Newton/bisection iteration with no atmosphere fallback. Cold, hot, high-Lorentz, and independently perturbed conserved states pass both round-trip directions; the generated derivative agrees with centered finite differences. [DONE]
  - Build all generated-kernel checks and document the remaining runtime evolution work. Symbolic/randomized generation validation passes and the generated C++ gate builds and runs in 1D, 2D, and 3D. The relativistic kernels remain outside runtime evolution, so Newtonian performance and behavior are unchanged. [DONE]

## Cycle 128 [completed]
- Wire the validated ideal-gas RHD bundle into a fixed-Minkowski AMReX evolution vertical slice. [DONE]
  - Audit state layout, primitive cache, reconstruction, Riemann, signal-speed, RHS, diagnostics, and configuration dispatch boundaries; choose the smallest shared-system extension without duplicating the AMR driver. [DONE]
  - Add generated SRHD signal speeds and a shared RHD system bundle using component reconstruction and a robust HLL-family flux. [DONE]
  - Register fixed-Minkowski ideal RHD configuration and state fields from one schema while leaving HD/MHD allocation and hot kernels unchanged. [DONE]
    - Remove divergent stage implementations found by the first runtime probe: both direct-flux and materialized-RHS stages now use one system-aware primitive-cache entry point, and RHD uses the same conservative Cartesian flux divergence as HD/MHD. [DONE]
  - Evolve a uniform state, smooth relativistic wave, and shock tube through the existing timestep/AMR machinery; validate round trips, conservation, convergence/reference behavior, and restart. Uniform free stream is exact, the entropy wave converges at second order, the shock remains physical, L2 AMR improves error by 2.39x over the common-grid N=16 control, and synchronized AMR/subcycling restart agrees exactly. [DONE]
  - Build all dimensions, run focused and quick regression gates, measure Newtonian performance, and document the exact remaining RMHD boundary. Fresh 1D/2D/3D builds and generated-kernel validation pass. The standard RHD gate passes exact 1D/2D/3D free streams, second-order smooth convergence, a physical shock reference, AMR improvement, and exact synchronized restart. Newtonian benchmarks remain at 0.93-0.99 of baseline. The aggregate quick gate remains 17/18 solely because of the pre-existing pulsating moving-cut-cell mass-ledger failure. RMHD state/flux generation and magnetic C2P remain unimplemented. [DONE]

## Cycle 129 [completed]
- Add the first generated ideal-RMHD vertical slice on fixed Minkowski spacetime. [DONE]
  - Derive and emit Valencia RMHD primitive-to-conserved variables, directional fluxes, and causal signal-speed bounds from one high-level indexed formulation. [DONE]
  - Validate generated identities, zero-field reduction to RHD, and arbitrary-direction rotational consistency before runtime dispatch. Independent randomized 3+1 comparisons, zero-field RHD reduction, and cyclic-axis covariance pass. [DONE]
  - Implement one robust magnetic C2P method behind the existing ordered recovery interface and extend the two-sided noisy-state round-trip corpus. A strict bracket-preserving Palenzuela1D recovery handles weak/strong-field, cold/hot, high-Lorentz, and off-manifold states without changing conserved data. [DONE]
  - Register RMHD state/schema and evolve uniform, smooth-wave, shock/reference, CT, AMR/subcycling, and restart controls without changing HD/MHD/RHD hot paths. Uniform 1D and 2D CT free streams are exact; entropy-mode convergence is second order; the Balsara-1-style shock preserves normal B; the 2D CT shock improves monotonically toward an N=1024 reference; L2 protected-transfer CT AMR with aligned local-CFL subcycling has zero divB; and synchronized restart is bitwise identical. [DONE]
  - Build all dimensions, run focused/common regression gates, measure existing-system performance, and assess the remaining fallback/curved-metric boundary. Generated Python and compiled 1D/2D/3D corpora pass, as do focused RHD/RMHD aggregate gates and both standard relativistic suites. Existing Newtonian benchmarks remain at 0.95-0.98 of the frozen baseline (108.11/130.63 Mcells/s in 2D and 6.03/0.18 Mcells/s in 3D). The aggregate quick gate still stops at the tracked pulsating moving-cut-cell mass-ledger defect before reaching relativistic steps. RMHD has one validated Palenzuela1D recovery; the remaining ordered methods, RePrimAnd placement, fixed curved metrics/geometric sources, and evolved spacetime are explicitly not yet implemented. [DONE]

## Cycle 130 [completed]
- Finish the production ideal-RMHD C2P policy and solidify the fixed-Minkowski RMHD infrastructure before generated spacetime evolution. [DONE]
  - Audit GRHayL, AsterX, and RePrimAnd recovery equations, admissibility rules, EOS requirements, and signal-speed policies; retain only methods that add distinct robustness to the supported state. Noble2D requires a seed, entropy methods require evolved entropy, and Font1D is a cold-EOS projection, so none is falsely exposed as an ideal-energy candidate. [DONE]
  - Implement one ordered RMHD recovery driver with shared post-recovery validation, bounded candidate reports, and no conserved-state mutation or hidden atmosphere fallback. Production uses Palenzuela1D then an out-of-line RePrimAnd cold fallback; full P2C residual validation is compile-time enabled in the corpus rather than repeated in the hot loop. [DONE]
  - Add and force-test the applicable energy recovery candidates over regular and pathological states; measure candidate success, residuals, iterations, and fallback cost. Both methods pass independently, and a deterministic off-manifold state forces Palenzuela failure followed by validated RePrimAnd success. The ordered fallback costs 1.8% versus direct Palenzuela in a controlled C2P-heavy timing. [DONE]
  - Compare the generated adaptive SRMHD fast-wave bound with a strict light-speed bound for accuracy, dissipation, timestep, Mcells/s, and simulation-time/real-time before selecting the production policy. Strict light speed is 6.5% faster in simulation-time/real-time for the measured 0.857c entropy wave but loses below the approximately 0.80c break-even; retain local causal bounds as the robust default and avoid another permanent knob. [DONE]
  - Consolidate RMHD configuration, primitive caching, diagnostics, CT, AMR/subcycling, restart, and generated-kernel boundaries; document the exact fixed-metric and evolved-spacetime handoff. Runtime call sites use one ordered recovery entry point, diagnostics consume the common result, CT/restart remain on the shared AMReX path, and the next boundary is generated fixed-curved-metric Valencia sources through a read-only stage-time metric provider. [DONE]
  - Build all dimensions and run generated, RMHD, common regression, and maintained performance gates. Final 1D/2D/3D builds and generated C++ corpus pass. Standard RHD and RMHD gates pass convergence, references, AMR/CT/subcycling, and bitwise restart. Existing Newtonian benchmarks pass at 0.91-0.98 of the frozen baseline. [DONE]

## Cycle 131 [completed]
- Complete the recovery-candidate infrastructure and fixed-curved-metric RHD/RMHD sector before generated spacetime evolution. [DONE]
  - Audit indexed tensor use and AsterX/GRHayL/RePrimAnd ownership of entropy, temperature, and electron fraction. AsterX evolves entropy for dual-energy recovery while temperature and electron fraction are finite-temperature/composition EOS coordinates; none belongs in the current ideal-gas state. Add checked variance-aware `Vector3`, `Vector4`, `Tensor3`, and `Tensor4` symbolic types for metric contractions and generated stress-energy work. [DONE]
  - Implement the remaining recovery candidates behind explicit seed, entropy, and cold-EOS capability contracts without allocating unused ideal-gas state. Noble2D, Palenzuela1D, RePrimAnd, both entropy candidates, and Font1D now share one result/validation contract while retaining their actual input requirements. [DONE]
  - Benchmark each applicable C2P candidate directly for success, iterations, residual, and throughput; separately retain the ordered-policy overhead measurement. RePrimAnd is 2.4x faster than Palenzuela1D on the regular corpus; seeded Noble2D is much faster when a valid prior primitive exists but is less robust on the pathological corpus. The production ideal-energy policy therefore uses RePrimAnd then Palenzuela1D, while seed/entropy/cold candidates remain explicit capability paths. [DONE]
  - Generalize the generated Valencia P2C, C2P, flux, and geometric-source path to an arbitrary read-only fixed 3+1 metric supplied at stage time. The same metric-aware primitive cache, initial P2C, face flux, causal speed, C2P, and generated stress-energy source path now serves fixed Minkowski, Rindler, and isotropic-Schwarzschild metrics. Physical ghosts are recovered using the interior metric and re-densitized using the ghost metric, so reflecting and inflow boundaries preserve the physical primitive state without copying metric-dependent conserved densities. [DONE]
  - Validate curved-space round trips, free streams or equilibria where applicable, AMR/subcycling, restart, and unchanged Minkowski/Newtonian behavior and performance. Generated algebra and compiled checks pass in every dimension. Rindler HSE approaches second order at N=128; isotropic-Schwarzschild HSE converges above first order. A three-level Rindler control gives a `7.88e-7` subcycling/single-rate difference and checkpoint restart agrees within `1e-12`. Variable-determinant Schwarzschild RMHD and three-level 2D Rindler RMHD+CT+subcycling controls pass scalar/divB checks. Maintained Newtonian performance is 107.55/130.92 Mcells/s in 2D and 5.94/0.18 Mcells/s in 3D, or 0.93-0.98 of the frozen baseline. The aggregate quick suite remains 17/18 solely because of the pre-existing pulsating moving-cut-cell ledger failure. [DONE]
  - Commit the completed coherent AMReX state without staging unrelated reference-backend work. Stage and verify only `pyfluid_amrex/`; leave unrelated JAX, Python-reference, Numba, and top-level script changes untouched. [DONE]

## Cycle 132 [completed]
- Fix the design boundary for generated spacetime evolution before implementation. [DONE]
  - Review AGPLv3 implications for using Cottonmouth as a reference rather than copying its implementation. Private local study creates no publication requirement; independently generate the published equations and resolve PyFluid licensing before distribution. [DONE]
  - Audit AsterX entropy evolution. Treat evolved entropy as an optional conserved fluid capability, `D_entropy = dens * entropy`, advected by the same transport velocity and used only by explicitly selected recovery candidates; do not silently repair conserved energy or allocate it when disabled. [DONE]
  - Keep fluid and spacetime fields on one AMR hierarchy while making centering a field-schema property. Valencia fields remain cell-centered; spacetime fields are vertex-centered; generated coupling operators perform the required vertex-to-cell, cell-to-vertex, and derivative transfers. Do not create a second independent gravity hierarchy. [DONE]
  - Define one symbolic derivative interface lowered by compile-time order, input/output centering, and stencil bias. Scalar stencils are implemented once; vector and tensor derivatives map componentwise while preserving index metadata. Centered first/second derivatives, mixed derivatives, upwind shift advection, Kreiss-Oliger dissipation, and centering interpolation are distinct policies under the same interface. [DONE]
  - Keep lapse and shift as generated gauge policies. Start with frozen, harmonic, 1+log, and the referenced slow-start lapse; frozen and Gamma-driver shift; avoid duplicating formulation equations for gauge choices. [DONE]

## Cycle 133 [completed]
- Refine the grid-field, centering, and differential-operator contract. [DONE]
  - Keep scalar, vector, and tensor value types independent of storage. A `GridField<ValueType>` binds one value type to the shared AMR hierarchy, component names, per-component centering, ghost contract, symmetry, and evolution/output ownership from the single field schema. [DONE]
  - Permit uniform centering for ordinary tensor fields and explicit per-component centering for staggered vectors such as face-centered magnetic fields and edge-centered electric/current fields. Do not duplicate declarations in output, checkpoint, or solver code. [DONE]
  - Expose one derivative API for all grid-field value types. Lower raw logical partial derivatives through one generated finite-difference policy; build physical gradient/divergence/curl and covariant derivatives from the coordinate/metric provider; satisfy physical and coarse-fine boundaries through the field ghost-validity contract rather than embedding boundary conditions in physics formulas. [DONE]
  - Minimize coupling interpolation by transferring directly from source centering to each required destination centering once per stage, fusing derivative/interpolation work where profitable, and reusing a stage-local metric view. Fixed metrics may cache these views for the whole run. [DONE]
  - Before public distribution, audit PyFluid code generation for copied EinsteinEngine/Cottonmouth implementation rather than assuming equation-level inspiration creates source-code provenance. Preserve notices and compatible licensing for retained copied code, or independently re-express it from primary equations. [DONE]

## Cycle 134 [completed]
- Add the first independently generated vacuum Z4c spacetime vertical slice on the unified AMReX field/operator infrastructure. [DONE]
  - Audit the current field registry, centered storage, derivative helpers, generated equation layer, AMR lifecycle, and runtime system dispatch. Define the smallest migration that serves existing HD/MHD/RHD/RMHD and vertex-centered spacetime fields without a second hierarchy or duplicate hot path. The current state uses one AMR hierarchy but a three-component auxiliary descriptor and output-local centering map; the spacetime sector therefore needs a first-class nodal state container beside, not inside, auxiliary fields. [DONE]
  - Generalize the single field schema to arbitrary scalar/vector/tensor component counts, per-component centerings including vertices, symmetry/parity, ghost requirements, evolution ownership, restart, and output metadata. Remove the three-component ceiling, add vertex topology, and centralize cell/vertex/face/edge AMReX index types and output suffixes. Replace the fluid solver's parallel lifecycle vectors with one first-class `EvolutionSectorStorage`; the same owner now represents cell-centered fluid state and, only when evolved spacetime is selected, the separate 22-component nodal Z4c state. RHS and RK stage storage are sector work arrays, not aux fields. Fresh 1D/2D/3D builds, a fixed-metric RHD runtime step, and maintained performance at 0.94-0.99 of baseline pass. Nodal hierarchy transfer/checkpoint/timestep wiring remains under the dedicated runtime subtask below. [DONE]
  - Introduce one generated differential-operator contract for all GFs and coordinate systems: centered first/second and mixed derivatives, upwind advection, KO dissipation, and direct centering transfers. Reuse it from existing nonrelativistic operators where behavior is identical. The generated 2/4/6/8-order centered, upwind, mixed, KO, and cell/face/vertex transfer contract compiles and passes its 1D/2D/3D polynomial checks; Newtonian gravity interpolation and CT current derivatives consume the common generated operators. [DONE]
  - Express vacuum Z4c state, constraints, gauge policies, and RHS in independent high-level indexed equations; generate fourth-order Cartesian kernels with optional frozen/harmonic/1+log/slow-start lapse and frozen/Gamma-driver shift. Do not add entropy evolution. The generator now emits conformal/physical metric geometry, true diagnostic Ricci, evolved-Gamma Z4c Ricci, Hamiltonian/momentum/Z/Theta/algebraic constraints, the complete geometric RHS with explicit matter projections, and the requested gauge policies. One compile-time gauge-selected `evaluate_z4c_rhs` assembles the complete pointwise evolution result, so runtime orchestration will not reproduce geometry or add per-point gauge switches. Centered geometric derivatives and shifted-upwind advection derivatives use the common generated operator contract; the corrected order-N second derivative is the standard order-N centered stencil rather than a mislabeled higher-order stencil. Generated and complete solver builds pass in 1D/2D/3D. [DONE]
    - Define full-value Z4c state conventions and one reusable generated pointwise geometry/constraint evaluator. Evolution, initial-data validation, and diagnostics must consume this evaluator rather than duplicate constraint formulas; matter projections are explicit inputs with a vacuum zero-source wrapper. `evaluate_z4c_constraints` is the pure state/derivative API, while `evaluate_z4c_constraints_at<Order, Dim>` is the common nodal-grid entry point that owns component loading and derivative gathering. A compiled corpus checks all 22 load/store mappings, centered/upwind/mixed derivatives in every executable dimension, Minkowski constraints, manufactured curved geometry, Gamma-constraint Ricci agreement, matter coefficients, and gauges. [DONE]
  - Register vertex-centered spacetime state/RHS on the existing AMR hierarchy and run Minkowski, gauge-wave, linear-wave, robust-stability, AMR/subcycling, checkpoint/restart, and constraint tests before matter coupling. The standard 15-check gate passes: every timestepper preserves Minkowski to about `1.1e-15`; the gauge wave converges at fourth order; the linear wave converges at fourth order; scaled robust-stability constraints remain bounded across N=32/64; L3 RK4 subcycling agrees with single-rate evolution to `1.04e-4` relative while both remain within `2.11e-6` of the analytic lapse; and synchronized restart is exact within `1e-12`. [DONE]
    - Finish runtime integration without expanding Z4c algebra in handwritten code: move generic copy/cache/transfer operations behind the evolution-sector owner, use AMReX nodal interpolation and restriction, and keep initial data, constraints, gauges, and RHS generated from high-level equations. Generic copy/linear-combination/add-scaled and RK4 dense coarse-time interpolation now belong to `EvolutionSectorStorage` and are used by both fluid and spacetime. [DONE]
    - Consolidate the resulting fluid/spacetime stage and hierarchy lifecycle, remove superseded helpers rather than retaining fallback paths, and keep the no-spacetime production fast path unchanged. Spacetime stage orchestration is now a normal compiled AMR source rather than an implementation include in the template-heavy fluid driver. Editing it rebuilds and links the 1D target in 3.65 s instead of rebuilding the roughly one-minute driver translation unit. [DONE]
  - Build all dimensions, run Newtonian and relativistic regression/performance gates, remove superseded helpers rather than retaining fallbacks, document provenance, and close the cycle. Generated algebra and fresh 1D/2D/3D builds pass. The shared fluid RK4 AMR/subcycling regression passes after dense-output consolidation. The standard Z4c gate passes all 15 checks. The quick suite remains 17/18 only because of the pre-existing pulsating moving-cut-cell ledger defect (`mass_rel_drift=1.156e-7`). Maintained benchmarks pass at 108.91/128.58 Mcells/s in 2D and 6.03/0.18 Mcells/s in 3D, or 0.94-0.99 of the frozen baseline. Matter projections are an explicit generated interface but evolved fluid-spacetime coupling is not implemented in this vacuum vertical slice. [DONE]

## Cycle 135 [completed]
- Temporarily remove the known-broken pulsating moving-obstacle case from the aggregate feature and AMReX gates without adding skip/fallback machinery. Keep its input and solver implementation intact for direct reproduction when moving-cut-cell conservation work resumes; static-obstacle and moving-piston coverage remain active. [DONE]

## Cycle 136 [completed]
- Consolidate mathematical constants behind one typed AMReX source of truth and make generated and handwritten kernels consume it. [DONE]
  - Replace duplicated C++ pi literals with `pyfluid_amrex::constants::pi` or `two_pi`, and make the symbolic C++ printer emit the same constant. [DONE]
  - Replace duplicated Python validation literals with `math.pi`/`numpy.pi` and leave serialized input-file values numeric. [DONE]
  - Regenerate every kernel family and run focused generated/Z4c/build validation. Generated algebra passes, the Z4c quick gate passes all 11 checks, and the complete 1D/2D/3D production targets build. [DONE]
  - Record the coupled fluid-spacetime method-of-lines implementation plan without duplicating the generated Valencia or Z4c equations. [DONE]

## Cycle 137 [completed]
- Audit handwritten and symbolic AMReX code for duplicated named mathematical or physical constants beyond pi. Consolidate only universal values; leave model coefficients, tolerances, and serialized run data at their ownership sites. [DONE]
  - Remove the duplicate `PyFluidHalfPi`/`TwoPi`/`FourPi` coordinate constants and the radial-origin two-pi literal. `constants.hpp` now derives `half_pi`, `two_pi`, `four_pi`, and `radians_per_degree` from its single stored `pi`. [DONE]
  - Replace the opaque KH Gaussian-derivative normalization decimal with its local mathematical definition `exp(1/2)`; do not globalize problem-specific coefficients or unrelated numerical tolerances/floors. [DONE]
  - Build complete 1D/2D/3D targets and pass the generated-kernel corpus. The 52-case curvilinear method gate passes 48 cases and independently exposes four existing numerical failures: one `1.074e-10` energy-ledger miss against `1e-10`, plus three curvilinear AMR+CT `divB` failures. Do not misattribute or hide those failures in this constants-only cleanup. [DONE]

## Cycle 138 [completed]
- Add strongly coupled RHD/RMHD and evolved-Z4c matter evolution through the existing shared time integrators, with ETK AsterX/TmunuBaseX/ODESolvers and SpacetimeX/Z4c as scheduling and projection references. [DONE]
  - Record the exact reference revisions and the chosen stage-ordering contract in `docs/grmhd_development.md`; keep generated Valencia and Z4c equations independent of scheduler details. [DONE]
  - Upgrade the single generated cell/vertex transfer contract to the supported spatial orders and use one selected order consistently for stage metric and matter projections. [DONE]
  - Replace sequential fluid-then-vacuum-spacetime stepping with one sector-aware RK stage scheduler. Every active sector must evaluate its RHS from the same stage-time state; fixed-spacetime and vacuum paths remain specializations of this contract rather than separate integrators. The shared Euler/Heun/midpoint/SSPRK3/RK4 stage algebra now advances both sectors, and the old standalone vacuum integrators are removed. [DONE]
  - Implement generated RHD/RMHD Eulerian matter projections as stage-local coupling views, transfer them to the vertex-centered Z4c sector, and preserve the existing fluid reflux and nodal spacetime synchronization contracts. Generated Z4c-to-Valencia metric/derivative views, Valencia-to-Eulerian projections, fourth-order centering transfers, evolved-metric C2P/flux/source/CFL evaluation, and one metric-provider physical-boundary algorithm are wired. [DONE]
  - Defer multirate gravity until a published method is selected. Frozen/fixed spacetime and cadence-one fully coupled evolution are the supported policies; do not expose stale-source step skipping as multirate evolution. [DONE]
  - Validate vacuum equivalence, fixed-metric regression, projection identities, coupled weak-field behavior, deep AMR/subcycling, and synchronized restart before TOV initial data. The standard 22-check gate passes all vacuum controls, four-level AMR/subcycling, exact synchronized vacuum and coupled-RHD restart, and a four-level coupled-RMHD smoke with roundoff baryon-mass drift. [DONE]
  - Add one coordinate/evolution-grid-independent initial-data provider. The internal one-dimensional TOV solve now transforms areal to isotropic radius and supplies one physical-coordinate view to both fluid and spacetime initialization on every AMR level. Single and piecewise cold polytropes plus hybrid thermal evolution are supported. Arbitrary-coordinate external-grid sampling, COCAL/KEH, RNS, black-hole, and binary data remain future provider backends. [DONE]
  - Replace AMReX bilinear nodal prolongation with one fourth-order tensor-product spacetime transfer used by level creation, remake, FillPatch, and subcycling. This reduces the smooth deep-AMR gauge-wave error by about two orders of magnitude; a sixth-order experiment improved that smooth control further but provided no TOV benefit and reduced 3D throughput by about 3.7x, so it was removed rather than retained as another branch. [DONE]
  - Validate the internal TOV provider in 3D. Interior Hamiltonian L2 norms converge from `5.995e-5` at N=24 to `2.185e-5` at N=48 with surface-limited orders 1.33 and 1.55; a piecewise-polytrope TOV smoke is finite; and a short two-level coupled SSPRK3/subcycling evolution retains mass to about `1e-11` without Hamiltonian-L2 growth. [DONE]
  - Preserve the nonrelativistic CT initialization contract while adding spacetime support. Every newly initialized level now FillPatches its cell state before constructing staggered magnetic faces so coarse-fine ghosts cannot contaminate initial `divB`. The focused uniform-field AMR control improves from `6.94e-1` to `6.94e-18` initially and remains at `3.01e-15` after three SSPRK3 subcycled steps; the production multipatch MHD+CT gate passes. [DONE]
  - Run final generated, 1D/2D/3D build, nonrelativistic regression, and maintained performance gates; document the exact supported boundary and close the cycle. Python generation and compiled 1D/2D/3D corpora pass, complete production targets build, and the 22-check coupled Z4c/TOV gate passes. Maintained rates are 109.75/129.62 Mcells/s for 2D at 3/4 ranks, 5.95 Mcells/s for 3D PWC, and 0.18 Mcells/s for 3D WENO-Z/RK4, or 0.93-1.00 of the frozen baseline. The aggregate quick gate passes every case except the independently tracked spherical wall-ledger residual `1.074e-10` against `1.000e-10`, unchanged from Cycle 137; do not weaken or hide it in this coupled-GR cycle. [DONE]

## Cycle 139 [completed]
- Make relativistic EOS evolution and initial-data ownership consistent before expanding GRMHD. [DONE]
  - Route RHD through one EOS-aware evolution dispatch so initialization, C2P, fluxes, sources, boundaries, diagnostics, and every RK stage use the selected ideal-gas, polytropic, or hybrid EOS. The piecewise/hybrid TOV regression now advances two coupled steps, retains baryon mass to `9.34e-12`, and does not grow its Hamiltonian L2 norm. RMHD remains explicitly ideal-gas-only because its maintained recovery residuals use that closure; no unsupported EOS silently falls back. [DONE]
  - Define one initial-data provider contract that normally supplies a self-consistent fluid and spacetime solution. Group policies select provider data or a mathematically valid neutral override (`zero` magnetic field, flat spacetime, unit lapse, zero shift); `none` is rejected when the selected system allocates that group. Storage remains a physical-system/spacetime capability decision. The focused gate verifies exact RHD magnetic omission, exact RMHD zero-field initialization, and exact neutral Z4c initialization. [DONE]
  - Consolidate the resulting interfaces and document the first production GR/RHD validation ladder: long-lived TOV in Cowling then coupled spacetime, vacuum Schwarzschild/puncture, and stationary relativistic Bondi accretion before expensive GRMHD tests. Gauge remains a runtime evolution policy independent of initial-data ownership. [DONE]
  - Generated checks and complete 1D/2D/3D builds pass. The standard coupled Z4c/TOV/RMHD gate passes all 24 checks. The aggregate quick gate passes every component except the unchanged spherical 3D wall-ledger residual `1.074e-10` against `1.000e-10`. Maintained rates are 107.46/127.30 Mcells/s in 2D at 3/4 ranks, 5.95 Mcells/s for 3D PWC, and 0.18 Mcells/s for 3D WENO-Z/RK4, or 0.93-0.98 of the frozen baseline. [DONE]

## Cycle 140 [completed]
- Verify and solidify piecewise-polytropic EOS support for Newtonian HD/MHD without adding another EOS implementation. [DONE]
  - Confirm the shared EOS dispatch, segment continuity construction, barotropic energy synchronization, CT, explicit resistivity, and implicit resistivity all consume the same piecewise EOS parameters. The production implementation already supports up to eight segments through one generated EOS path. [DONE]
  - Upgrade the Newtonian feature cases from one-segment polytropes to true two-segment piecewise cold and hybrid polytropes. Finite-amplitude HD and MHD sine states cross the density break; CT plus explicit and implicit resistivity remain covered. [DONE]
  - Define the proper cold-barotrope scalar contract once in shared validation: mass, momentum, physical states, divB, and performance remain required, while total energy is not treated as an independently conserved variable because it is reconstructed from the EOS manifold. Hybrid thermal polytropes retain the conservative-energy check. [DONE]
  - Document the code-unit contract for density breaks and polytropic constants. Generated EOS validation passes, all six focused HD/MHD EOS cases pass on three MPI ranks, and the 13-gate aggregate quick suite passes. [DONE]

## Cycle 141 [completed]
- Complete provider-backed fixed-spacetime (Cowling) evolution and begin the generic relativistic initial-data boundary. [DONE]
  - Make `spacetime.evolution = fixed` the single Cowling policy: analytic metrics remain pointwise and allocation-free, while provider-backed metrics use stored fields initialized from the same provider as coupled evolution, projected to fluid stages, checkpointed/restarted/output, but never receive a spacetime RHS. Fixed TOV now uses this contract, including one mutable primitive-recovery path that applies atmosphere projection without duplicating C2P work. [DONE]
  - Add maximum density and its relative change to the standard runtime scalar diagnostics so hydrostatic-equilibrium violations and stellar oscillations are directly measurable. The 65-column scalar schema and checkpoint reference state now carry both values. [DONE]
  - Factor the TOV-specific sampling view into a general physical-coordinate initial-data provider contract capable of analytic or sampled Cartesian/polar data, without adding a second fluid or spacetime initialization path. State initialization, hierarchy creation/remake, FillPatch, and Z4c initialization now consume one provider view; TOV is the first backend. [DONE]
  - Add focused Cowling-versus-coupled TOV validation, synchronized restart, generated/build gates, and performance regression checks; document the remaining external-grid, Michel/Bondi, puncture, and radiative-boundary work honestly. The four-level subcycled Cowling control changes maximum density by `2.39e-6`, preserves its stored metric to `2.22e-16`, and restarts exactly within `1e-12`; the evolved control changes maximum density by `9.30e-4`. The standard 27-check Z4c/TOV/RMHD gate, generated corpus, and complete 1D/2D/3D builds pass. The aggregate quick gate passes all components except the unchanged spherical wall-ledger residual `1.074e-10` against `1.000e-10`. Maintained Newtonian rates are 108.92/128.58 Mcells/s in 2D and 5.97/0.18 Mcells/s in 3D, or 0.94-0.99 of the frozen baseline. Validation also caught and fixed an invalid pressure-triggered atmosphere replacement in cold stellar envelopes; the coupled 3D TOV mass drift is restored from `2.71e-6` to `1.36e-11`. [DONE]

## Cycle 142 [completed]
- Unify fixed analytic and provider-backed spacetime behind canonical stored fields without burdening evolved spacetime or Newtonian paths. [DONE]
  - Make the initial-data provider independently advertise fluid and spacetime data, and materialize analytic fixed metrics through the same physical-coordinate sampling contract as TOV. [DONE]
  - Re-evaluate immutable spacetime exactly on hierarchy creation/remake; retain normal AMR transfer for evolved fluid and spacetime state. [DONE]
  - Allocate only one canonical state for immutable sectors; reserve old/stage/RHS/dense-output storage for sectors that evolve. [DONE]
  - Cache immutable cell-centered coupling data across stages and invalidate it only when hierarchy/state materialization changes. [DONE]
  - Remove the old pointwise fixed-metric physical-boundary fallback. Fresh, remade, and restarted levels now refresh one canonical cell metric view and apply the same stored-metric boundary conversion used at every RHS stage. [DONE]
  - Build all dimensions and run focused fixed-metric, TOV Cowling/restart/AMR, generated, quick, and performance gates. Complete 1D/2D/3D targets and generated algebra pass. All 10 fixed-spacetime checks and all 27 Z4c/TOV/RMHD checks pass, including three/four-level AMR, CT, subcycling, and exact restart. The aggregate quick gate passes every case except the unchanged spherical wall-energy residual `1.074e-10` against `1.000e-10`. Maintained Newtonian rates are 107.28/128.27 Mcells/s in 2D and 5.97/0.18 Mcells/s in 3D, or 0.94-0.98 of the frozen baseline. [DONE]

## Cycle 143 [completed]
- Plan the first visible fixed-Minkowski SRHD/RMHD evolution and the constraint-consistent coupled-GR initial-data sector without changing runtime behavior. [DONE]
  - Use a short unmagnetized relativistic Kelvin-Helmholtz run as the first visual SRHD exercise; defer RMHD until the hydrodynamic AMR and recovery behavior is visible and measured. [DONE]
  - Keep fluid AMR driven by normalized physical primitive structure and add spacetime refinement through a separately measured constraint/curvature contribution combined by the common AMR policy, rather than tagging raw Z4c components or introducing a second hierarchy. [DONE]
  - Build the coupled initial-data solver as a provider backend: generated conformal constraint residuals and Jacobian pieces, one AMReX multilevel elliptic solve, then canonical spacetime materialization and Valencia P2C. Start with CTT/XCTS controls and TOV recovery before considering KEH/waveless rotating or binary data. [DONE]

## Cycle 144 [completed]
- Refine the relativistic AMR and initial-data design from production references before implementation. [DONE]
  - Keep the first visible test token-cheap: a short fixed-Minkowski SRHD Kelvin-Helmholtz run with scalar validation, AMR level history, and one state movie. [DONE]
  - Do not combine raw fluid, geometry, and constraint indicators. Each contributor must first report a dimensionless error-budget ratio with common semantics; only then may the shared AMR policy combine them. Treat constraint violations as a safety/validation channel until convergence tests show that using them for tagging improves solution error rather than merely chasing numerical noise. [DONE]
  - For RMHD, extend the existing physical-variable Lohner estimator with invariant magnetic structure rather than conserved-coordinate components: comoving magnetic pressure/magnetization and, after validation, a current-sheet sensor. Keep divergence error diagnostic rather than a normal refinement driver. [DONE]
  - Keep initial-data solving outside the evolution algorithm behind one provider interface. Supply both an in-process library path for analytic/small solves and a separate executable/artifact path for expensive spectral or multilevel elliptic solves; both materialize the same canonical field schema on the evolution hierarchy. [DONE]

## Cycle 145 [completed]
- Start the relativistic validation and initial-data-solver staircase without widening into full XCTS/KEH development. [DONE]
  - Audit and extend the common physical-variable AMR indicator so RHD and RMHD use normalized relativistic primitive and invariant magnetic channels rather than Newtonian-only selection or raw mixed-sector maxima. RHD/RMHD tagging and AMR-metric output now recover primitives from the canonical stored spacetime view; RMHD adds generated comoving magnetic pressure and magnetization channels. A magnetic-only discontinuity reaches the requested L2 hierarchy while uniform hydro primitives contribute no tags. [DONE]
  - Add one compact fixed-Minkowski SRHD Kelvin-Helmholtz run and automated conservation/reference checks using the normal runner and report infrastructure. Both subcycled and non-subcycled cases independently reach L3 with five levels of headroom and conserve mass/energy at `1e-15`; their equal-time conserved hierarchies agree to `3.53e-6` relative L2. Local-CFL subcycling improves simulation-time/wall-time from `0.0211` to `0.0301` in the focused run. [DONE]
  - Establish the initial-data elliptic problem/solver/provider boundary with generated residual support and one manufactured multilevel solve, keeping evolution dependent only on the provider contract. The isolated AMReX MLMG service owns no evolution state; generated manufactured Poisson checks pass in 1D/2D/3D with residuals `5.3e-12`--`6.4e-12` and L-infinity errors below `6.2e-4`. Full CTT/XCTS/KEH equations remain future provider backends. [DONE]
  - Build all affected targets and run focused relativistic, AMR, generated-kernel, and regression gates before closing the cycle. Complete 1D/2D/3D production targets, standalone elliptic checks, generated-header consistency, all 13 quick Z4c/RHD/RMHD checks, and the focused SRHD/RMHD AMR checks pass. The aggregate quick gate retains its pre-existing spherical 3D wall-ledger miss (`1.074e-10` versus `1.000e-10`). Maintained 3-rank 2D and 3D rates are 0.86--0.94 of the frozen baseline; the current unbound macOS 4-rank sample is anomalously low at 0.53 and is recorded as a warning rather than hidden or attributed to this AMR-only change. [DONE]

## Cycle 146 [completed]
- Begin the physical CTT-to-XCTS initial-data staircase and analytic-spacetime gallery while preserving one provider/materialization path for fixed and evolved spacetime. [DONE]
  - Generate and validate the conformally flat, time-symmetric CTT Hamiltonian residual/Jacobian pieces; use the existing multilevel elliptic service rather than a formulation-specific solver copy. The manufactured nonlinear solve now passes on two AMR levels in 1D/2D/3D. [DONE]
  - Exercise the first nonlinear CTT solve against the regular isotropic TOV provider, then add the CTT momentum/vector-potential sector before any XCTS lapse/shift equations. The two-level TOV recovery reaches `4.15e-4` L-infinity conformal-factor error, while the generated 3D momentum control reaches `6.42e-6`. [DONE]
  - Later provider/materialization cycles consolidated fixed and evolved
    analytic spacetime on canonical fields and validated the shared path. [DONE]
  - Black-hole excision/turduckening was deliberately deferred and is not an
    unfinished part of the now-completed regular-domain CTT/XCTS staircase. [DONE]
  - Cycles 156--176 completed the generated, elliptic, provider, formulation,
    materialization, and relativistic regression gates. [DONE]

## Cycle 147 [completed]
- Give the initial-data elliptic sector one general, readable Robin boundary contract without formulation-specific caller paths. [DONE]
  - Represent every nonperiodic physical boundary as coefficient fields in the coordinate-derivative form `A f + B d_d f = C`; derive periodicity from AMReX geometry and convert low-face coordinate derivatives to AMReX outward-normal convention once inside the backend. [DONE]
  - Use the same coefficient API for spatially varying Robin and Dirichlet data. Lower the exact `B=0` case to native AMReX Dirichlet internally so scalar and conformal-vector operators share the public contract; reject genuine Robin for `MLTensorOp` because its tensor cross-term fill does not implement Robin consistently. [DONE]
  - Validate a spatially varying Robin manufactured problem, nonlinear CTT Hamiltonian solve, generated CTT momentum solve, and isotropic-TOV recovery on two AMR levels. Three-rank 1D/2D/3D checks pass with scalar Robin L-infinity errors `3.98e-4`--`5.98e-4`, CTT Hamiltonian errors `3.58e-6`--`6.07e-6`, 3D momentum error `6.42e-6`, and TOV conformal-factor error `4.15e-4`; generated-kernel consistency also passes. [DONE]
  - Run the aggregate quick regression gate. Every maintained feature, relativistic, AMR, dimensional, multipatch, cylindrical, and spherical case passes except the unchanged spherical 3D wall-ledger residual `1.074e-10` against `1.000e-10`; no tolerance or diagnostic was weakened. [DONE]

## Cycle 148 [completed]
- Add the standard centered asymptotic power-law Robin boundary used by fields of the form `f_inf + amplitude/r^p`, without requiring the amplitude or a formulation-specific boundary path. [DONE]
  - Fill the common `A/B/C` coefficient fields from the physical face coordinates and domain center. The direct coefficient identity passes in 1D/2D/3D at `0`--`2.22e-16`; corner conversion now selects the same physical face as coefficient construction. [DONE]
  - Exercise the `p=1`, `f_inf=0` boundary on the isotropic-TOV conformal correction instead of sampling exact finite-radius data. The two-level TOV solve converges with `4.61e-4` L-infinity error; all focused 1D/2D/3D elliptic controls and generated-kernel consistency pass. [DONE]

## Cycle 149 [completed]
- Audit the current CTT and multilevel initial-data handoff status before extending the solver. [DONE]
  - The composite AMReX elliptic service and conformally flat, time-symmetric Hamiltonian solve are functional on a supplied multilevel hierarchy. The manufactured conformal-vector operator is validated separately, but general CTT still needs coupled momentum/Hamiltonian iteration, extrinsic-curvature construction, and canonical-provider materialization. [DONE]
  - Prefer solving directly on the initial evolution hierarchy when practical: it removes an avoidable grid-to-grid interpolation and lets evolution inherit the constraint-solved state. Keep physical-coordinate provider sampling for analytic or external initial data, and allow normal evolution AMR to regrid after startup. [DONE]

## Cycle 150 [completed]
- Fix the initial-data materialization contract for analytic, sampled-grid, and hierarchy-native sources. [DONE]
  - Use one provider-to-hierarchy interface: analytic providers evaluate directly at every target location, sampled-grid providers interpolate from their original data once to each target location, and hierarchy-native providers hand off canonical fields directly when source and destination layouts match. Do not create three evolution initialization implementations. [DONE]
  - Permit the initial hierarchy builder to start from a coarse representation, tag/build AMR levels, then rematerialize from the original provider on every level. Analytic data must not be prolonged from an avoidably coarse sample; finite-resolution source grids cannot create information beyond their source resolution. [DONE]

## Cycle 151 [completed]
- Refine the initial-data-to-evolution handoff and initial AMR-settling policy. [DONE]
  - When initialized canonical GFs already have the evolution hierarchy and schema, hand them directly to evolution regardless of whether their provenance was analytic, sampled, or elliptically solved. Analytic resampling is an optional quality improvement only for cells created during hierarchy construction or regridding. [DONE]
  - Before the first evolution step, run the normal production tagging/regrid/transfer path at `t=0` until the hierarchy is unchanged. Populate new levels with the one supported centering-aware conservative transfer, or rematerialize from an available exact provider and average down. Report hierarchy changes; fail only when settling cannot complete because a resource guard, unsupported transfer, or iteration bound prevents the requested refinement. [DONE]

## Cycle 152 [completed]
- Simplify the initial hierarchy contract and supersede Cycle 151's proposed startup settling loop. [DONE]
  - Materialize initial data once on the configured evolution hierarchy, independent of provider provenance. Immediately before iteration zero, perform a read-only proposal through the normal AMR tagging/grid-generation policy. If the proposed hierarchy differs, abort with the requested level/coverage change so the initial-grid configuration can be corrected and rerun. [DONE]
  - Do not regrid, rematerialize analytic data, or iterate hierarchy construction inside the IC module. Initial-grid construction remains preprocessing; the IC module only materializes canonical fields, and evolution receives those same fields after the preflight passes. [DONE]

## Cycle 153 [completed]
- Clarify analytic spacetime materialization and storage ownership. [DONE]
  - Analytic providers retain compiled pointwise evaluators, not a second hierarchy of functions or every derived expression as a GF. At initialization they evaluate only the canonical formulation fields required by the schema; constraint residuals and other derived quantities are computed as diagnostics unless explicitly designated as evolution caches. [DONE]
  - Evolved spacetime uses the analytic evaluator only to initialize iteration zero. Fixed analytic spacetime may reuse it when new AMR levels are created, then stores one immutable canonical metric state without evolved time levels. [DONE]

## Cycle 154 [completed]
- Set the pre-evolution regrid policy from provider capability rather than provider names. [DONE]
  - A fully initialized state may expose exact rematerialization on a changed hierarchy. If every required field supports it, the iteration-zero preflight may apply the proposed regrid, rematerialize the canonical fields, and recheck before evolution. [DONE]
  - If any required initial field exists only as sampled/hierarchy-native GF data, a proposed iteration-zero hierarchy change is a hard configuration error with the requested level/coverage reported. Constraint-solved fields count as GF-only unless the complete solve is explicitly rerunnable on the new hierarchy. [DONE]

## Cycle 155 [completed]
- Refine initial regrid policy around the source/evolution hierarchy relationship. [DONE]
  - When the IC source hierarchy differs from the evolution hierarchy, target-grid construction already requires interpolation; permit the iteration-zero AMR preflight to change that target hierarchy and materialize through the same provider interface. [DONE]
  - When IC and evolution hierarchies match, preserve the no-interpolation contract by default: a proposed initial regrid is an error. Provide one explicit opt-in to allow it through the normal transfer/materialization path, emit a prominent warning that the matched-grid guarantee was lost, and rerun normal state/constraint validation before evolution. [DONE]

## Cycle 156 [completed]
- Route fluid and spacetime initial data through one materialization contract, then extend the physical CTT-to-XCTS staircase without parallel initialization implementations. [DONE]
  - Consolidate the shared primitive fluid sample and runtime magnetic/spacetime/lapse/shift group selections so Newtonian, fixed-spacetime relativistic, and coupled initial data consume one source of truth. [DONE]
  - Complete coupled conformally flat CTT momentum/Hamiltonian iteration from generated high-level equations, materialize canonical Z4c fields, and construct Valencia conserved state from the solved metric. The smooth acoustic-packet control converges in two matter and three CTT iterations with elliptic residuals below `1.3e-17`. [DONE]
  - The production iteration-zero preflight now validates the materialized
    hierarchy/state before evolution through the common solver path. [DONE]
  - Validate smooth coupled CTT controls before enabling additional ordinary hydro initial-data families; do not claim discontinuous or specialized source/obstacle cases until each is physically defined and tested. Density-cloud convergence and the nonzero-momentum acoustic packet pass; discontinuous/open/specialized profiles remain compatibility-only until separately validated. [DONE]
  - Build and run generated, elliptic, relativistic, AMR, scalar, and performance gates before beginning XCTS lapse/shift equations; TOV recovery follows XCTS, and rotating-star construction remains a later KEH/waveless backend. All dimensions build, generated checks and five focused CTT controls pass, and the aggregate quick gate has only the unchanged spherical wall-ledger miss (`1.074e-10` versus `1.000e-10`). Newtonian benchmark ratios are `0.93`--`0.99` of the frozen baseline. [DONE]
  - Treat initial-data centering as one schema-driven adapter boundary: providers and elliptic solvers produce canonical fields on their natural centering, then shared generated 2/4/6/8-order topology transfers materialize the cell/vertex/face/edge evolution fields and synchronize their AMR hierarchy. Do not solve separate copies of the ID equations for each evolution centering. Later centering-transfer and RMHD initial-data cycles completed the shared topology path and its boundary closure. [DONE]
  - Replace the relativistic IC whitelist and repeated hydro/MHD checks with one IC descriptor table that declares physical-system compatibility; retain geometry/EOS checks only where they describe the problem itself. [DONE]
  - Add a smooth nonzero-momentum ordinary-fluid profile to the coupled CTT gate so the shared Newtonian-profile-to-Valencia adapter and the CTT momentum sector are exercised together. [DONE]
  - Repair the compact CTT AMR handoff by synchronizing the composite cell hierarchy before cell-to-vertex materialization, then require solution improvement over the coarse uniform control without masking the evolution-side constraint at coarse-fine interfaces. Fine-level `chi` error is `0.397` of the coarse control; the raw fourth-order Hamiltonian stencil reports the retained interface penalty at `1.94` times the uniform coarse residual. [DONE]

## Cycle 157 [completed]
- Review CTTK as an initial-data reference and settle what it does and does not imply for the cell-fluid/vertex-spacetime handoff. [DONE]
  - Add Aurrekoetxea, Clough, and Lim (arXiv:2207.03125) to the GRMHD reference inventory. Record CTTK as a formulation option that fixes the conformal factor, solves the Hamiltonian constraint algebraically for spatially varying mean curvature, and iterates the coupled momentum solve; retain conventional CTT/XCTS as separate supported targets. [DONE]
  - Do not use the paper as evidence for a centering choice: it specifies no cell/vertex contract. Keep finite-volume hydro cell-centered and the current cell-centered CTT-to-schema transfer as the validated baseline; require an end-to-end AMR and boundary comparison before preferring a nodal elliptic solve for vertex-centered Z4c fields. [DONE]

## Cycle 158 [completed]
- Establish the production staggered initial-data architecture, then implement CTT, CTTK, and XCTS as formulations over that one hierarchy while reducing avoidable rebuild scope. [DONE]
  - Inventory the current cell-centered elliptic operators, topology transfers, formulation algebra, and CMake dependency graph. AMReX supplies a composite nodal scalar Laplacian but no equivalent coupled conformal-vector operator or general nodal Robin coefficient path; lower the vector operator to generated scalar Poisson solves and preserve one public Robin contract. The generator currently has one all-family stamp and one all-header compile-check translation unit. [DONE]
  - Supersede Cycle 157's conservative centering hold: the production endpoint is one staggered solve with cell-centered finite-volume matter and vertex-centered CTT/CTTK/XCTS unknowns. The cell-centered CTT path is a temporary validation control, not a second production formulation. [DONE]
  - Add one native vertex-centered scalar/vector elliptic contract and migrate CTT without duplicating the formulation or materialization path. The common nodal Robin contract now uses an MLMG-preconditioned matrix-free boundary Schur solve; AMReX-owned fine valid nodes remain authoritative through FillPatch, and coupled CTT directly materializes canonical vertex Z4c before constructing cell-centered Valencia state. Native scalar, vector, coupled matter, two-level boundary, and TOV controls pass in 1D/2D/3D. [DONE]
  - Add CTTK and XCTS formulation policies over the same cell-fluid/vertex-spacetime state and common nonlinear iteration infrastructure. Reuse the generated constraint algebra, vector solve, extrinsic-curvature construction, Robin backend, and canonical materializer; later formulation cycles completed all three branches and their distinct physical controls. [DONE]
  - Split generated and formulation-local compilation dependencies where this reduces incremental rebuild time without adding runtime branches or duplicate kernels. Cycle 159 isolated initial-data generation and implementation ownership; broader Z4c/relativistic specialization remains future work. [DONE]
  - Cycles 159--176 validated generated equations, manufactured elliptic
    problems, AMR boundaries, coupled matter, final evolution-centered
    constraints, complete builds, and incremental ownership. [DONE]

## Cycle 159 [completed]
- Reduce AMReX incremental compile time before continuing XCTS development, without changing runtime behavior or introducing parallel APIs. [DONE]
  - Measure the current dependency fanout for configuration, generated kernels, and initial-data implementation changes. The old initial-data implementation header reached 18 production translation units through the public umbrella header; generated Z4c and relativistic-fluid kernels still reach 17 and 16 translation units respectively. [DONE]
  - Move implementation-only config validation and initial-data provider materialization out of broadly included headers. Configuration parsing/validation now physically lives in `config.cpp`, and the public initial-data header contains declarations only; the 2,700-line state initializer and provider primitive materializer each have one owning translation unit. After the one-time header transition, a config implementation edit rebuilds all three dimensions in 4.12 s. [DONE]
  - Split generated-kernel dependencies only where a formulation-local edit currently rebuilds unrelated physics sectors. Initial-data/CTT/XCTS generation now has its own precise CMake stamp: an unchanged formulation edit regenerates and checks all production dimensions in 0.86 s instead of spending 13.32 s regenerating every physics family. Broader Z4c/relativistic headers remain the next high-cost compile boundary. [DONE]
  - Rebuild 1D/2D/3D targets, run focused regressions, and record before/after incremental timings. All production dimensions build. A one-time all-dimension rebuild after the public-header change took 728.72 s; touching the large state-initializer implementation afterward rebuilt all three dimensions in 8.21 s (1D alone: 2.36 s). The restored two-level 3-rank XCTS runtime gate passes on 46,592 cells with Hamiltonian L2 `1.24e-3` and Linf `1.48e-2`; the generated-kernel consistency gate passes. The normal quick suite passes every case except the unchanged spherical wall-ledger tolerance miss (`1.074e-10` versus `1.000e-10`). [DONE]

## Cycle 160 [completed]
- Reduce the excessive clean-build critical path without changing the runtime method matrix or duplicating solver implementations. [DONE]
  - Confirm the bottleneck from build artifacts and CPU utilization. The 3D `amr_solver_driver.cpp.o` is 86 MB while the next object is 0.74 MB; the 721 s rebuild consumed only 801 s of CPU time, showing that one template Cartesian product serializes the build. [DONE]
  - Test sharding the common evolution template graph at the runtime timestep-policy boundary. Reject it: one-dimensional wall time increased from roughly 149 s to 246 s and object code grew from one shared driver to five 24.7 MB shards because common template work was repeated. Restore the single implementation. [DONE]
  - Add one aggregate production target so Make schedules the existing dimension-specialized drivers concurrently. Rebuilding the three dominant driver objects and linking all dimensions takes 462.91 s versus 721.08 s sequentially, a 36% wall-time reduction with no solver-code duplication. This remains excessive: the next structural target is a non-template evolution orchestration loop calling fully specialized kernel bundles outside cell loops, so timestep/diffusion/subcycling policies do not replicate the complete evolution driver. [DONE]

## Cycle 161 [completed]
- Reduce the evolution-driver template Cartesian product while preserving one numerical implementation, runtime feature selection, readability, and production kernel performance. [DONE]
  - Inventory which template parameters affect orchestration versus AMReX point kernels and establish the narrow compiled bundle boundary. The synchronized output/regrid/diagnostic loop was repeated for every timestep, diffusion, and subcycling combination even though those policies only select coarse-step operations. [DONE]
  - Select one immutable `EvolutionOperations` bundle before evolution and call its specialized timestep, RK, CT, diffusion, subcycling, FillPatch, and regrid operations outside cell loops. Compile the common evolution lifecycle once per state/EOS/CT bundle; retain the existing specialized numerical templates as the sole implementations. Remove unnecessary adapters by pointing directly at existing member templates. [DONE]
  - Reject two non-improvements rather than retaining extra policy: runtime RK stage branching left the 1D compile unchanged (`106.74` versus `106.13` s), and `-O2` only reduced a direct compile to `103.55` s while weakening optimization. Restore compile-time RK specialization and keep `-O3`. [DONE]
  - Measure compile and runtime behavior. The 1D critical driver rebuild fell from roughly `149` s to `106` s (29%); its object is 23 MB. The 3D object fell from 86 MB to 73 MB. Concurrent dominant-driver rebuilds for 1D/2D/3D now take `403`--`420` s versus `462.91` s before the operation bundle and `721.08` s sequentially before parallel scheduling. A Clang time trace attributes 106.4 of 109.9 s to the optimization/backend and only 3.5 s to the frontend; the remaining compile cost is the reconstruction/Riemann/EOS/direction kernel matrix, not header parsing. [DONE]
  - Validate behavior and performance. All production dimensions build; the focused matrix passes all five timesteppers, explicit/implicit diffusion, explicit/implicit resistive CT, deep L7 aligned subcycling, and MHD CT subcycling (11/11). Runtime benchmark ratios remain 0.94--0.99 of the frozen baseline. The quick gate passes all cases except the unchanged spherical 3D wall-ledger edge (`1.074e-10` versus `1.000e-10`). [DONE]

## Cycle 162 [completed]
- Define the next compile-isolation boundary without weakening optimization, duplicating numerical methods, or complicating the runtime interface. [DONE]
  - Keep fused reconstruction/Riemann/EOS/direction cell kernels and move their explicit instantiations out of `amr_solver_driver.cpp` into generated, independently compiled kernel-family translation units. Expose one small host-side registry selected outside cell loops. [DONE]
  - Group generated translation units by physical system and reconstruction family, not by every Cartesian-product leaf, so clean builds can parallelize while source count remains manageable. Keep one shared handwritten reconstruction/Riemann API and generate wrappers plus explicit instantiation ownership. [DONE]
  - Split method implementation headers only where dependency measurements show that editing one reconstruction family currently rebuilds unrelated families. Do not add runtime branches inside cell kernels, temporary face-state passes, lowered optimization, PCH policy, or duplicate development/production implementations as compile-time workarounds. [DONE]
  - Preserve the existing input/config interface and full production target. Focused build targets may compile one physical-system kernel module for development, while the normal production aggregate still builds and links every supported module. [DONE]

## Cycle 163 [completed]
- Move expensive evolution-kernel instantiation ownership out of the monolithic driver while preserving one solver, one numerical implementation, and the complete runtime method matrix. [DONE]
  - Map unique template bundles before splitting. HD and RHD share the five-component hydro layout specialization; MHD and RMHD share magnetic-layout specializations. Use six unique state-layout/CT ownership units (hydro, hydro+passive, MHD, MHD+CT, MHD+passive, MHD+passive+CT) rather than four nominal physics files that would duplicate compilation. [DONE]
  - Add one generic compiled-bundle registry keyed only by template-instantiation properties (`num_components`, CT topology, and EOS dispatch policy), plus six ownership sources under `src/kernel_bundles/`. Physical-system identity remains explicit runtime data; HD/RHD and MHD/RMHD reuse matching compiled specializations without sharing or duplicating their physical equations. The driver now performs validation and one host-side callback selection outside the evolution loop. [DONE]
  - Measure clean critical-path time, optimizer/object distribution, and incremental ownership before retaining or further splitting the largest bundle. The 1D critical build falls from 106 s to 63 s. A one-time all-dimension bundle rebuild takes 348 s versus 403--420 s before the split. A no-content 1D magnetic-bundle rebuild compiles one owner and relinks in 26.64 s. Intermediate bundle objects total 42/88/123 MB in 1D/2D/3D, while linked executables are 19/40/56 MB; retain this measured object-space tradeoff because it removes the serial optimizer bottleneck and gives family-local ownership without runtime code duplication. [DONE]
  - Rebuild all dimensions and run the focused method matrix, quick gate, and performance benchmark. All dimensions link; the focused matrix passes 16/16 across every current bundle key, five timesteppers, passive layouts, explicit/implicit parabolic methods, RHD/RMHD, deep L7 aligned subcycling, and CT subcycling. The quick gate has only the unchanged spherical wall-ledger edge (`1.074e-10` versus `1.000e-10`). Runtime benchmark ratios remain 0.94--0.99 of the frozen baseline. [DONE]

## Cycle 164 [completed]
- Add the final low-risk compile-time tooling improvements before returning to physics development. [DONE]
  - Enable compiler caching in the Mac full-release preset without making ccache a numerical or repository-wide build dependency. The launcher uses an ignored workspace-local cache and explicitly identifies the Homebrew MPI wrapper as Clang. A 1D magnetic-bundle compile takes 25.63 s cold and 0.32 s from cache. [DONE]
  - Add `tools/benchmark_compile.py`, one compact wrapper around ordinary CMake targets that records target, jobs, elapsed time, status, cache deltas, revision, and dirty state in `runs/current/compile_history.tsv` plus its bounded 40-run mirror. Document that cold and cache-hit timings are distinct measurements. [DONE]
  - Confirm ordinary dependency builds still compile only changed ownership units, then rebuild and run a focused smoke gate. Touching one magnetic owner compiles exactly its 1D/2D/3D objects: the two uncached remaining dimensions build and link in 76.77 s, while a repeated all-dimension build takes 0.61 s with three cache hits. The compact HD/MHD/RHD/RMHD feature gate passes 5/5. [DONE]

## Cycle 165 [completed]
- Audit the remaining clean-build and future-development compile-time opportunities after enabling compiler caching. [DONE]
  - Confirm that the remaining cold-build cost is generated code optimization rather than parsing or linking. Every one of the 18 dimension/layout bundle objects currently depends on the monolithic generated Z4c, relativistic-fluid, reconstruction, and Riemann headers; split generated headers by stable equation family to narrow incremental rebuilds, while preserving one high-level generator and one numerical implementation. [DONE]
  - Quantify repeated backend work before proposing another split. In 1D, 775 weak symbols occur in all six layout bundles, 1,889 symbols occur in at least two, and the bundle objects contain 5,003 redundant symbol copies. Move bundle-invariant AMReX launch wrappers and diagnostics to one owning translation unit per dimension, starting with one measured family and retaining the change only if cold CPU time and object size fall without runtime regression. [DONE]
  - Reject compile-time shortcuts that compromise production behavior or merely relocate cost: lower optimization, broad unity builds, PCH/modules for an optimizer-bound workload, finer state/EOS source multiplication, and LTO as a development-build workaround. Keep focused object targets plus ccache for normal iteration and reserve the aggregate production target for integration gates. [DONE]

## Cycle 166 [completed]
- Modularize compile ownership around stable numerical responsibilities while preserving one runtime interface, one implementation of each method, and fully specialized point kernels. [DONE]
  - Measure generated-header fanout and repeated weak-symbol ownership by subsystem before moving code. Establish cold compile time, object size, and runtime baselines for each retained extraction. The old six 1D bundles contained 41 MiB of objects and all consumed reconstruction/Riemann/RHD/RMHD implementation headers. [DONE]
  - Extract bundle-invariant AMR hierarchy/transfer, diagnostics/repair, and outer AMReX launch operations into declaration-only interfaces with one implementation owner per executable dimension. Do not move runtime selection into cell loops. `ErrorEst` now has one source owner; output cadence is independent of the output implementation; primitive-cache/C2P, atmosphere, signal-speed, spacetime timestep, and relativistic-source launches use narrow compiled factories. Large orchestration remains in compiled owners; only small AMReX/GPU point kernels remain inline. [DONE]
  - Split generated relativistic-fluid and Z4c outputs by stable equation family so changing C2P, flux/source, geometry, constraints, or spacetime evolution invalidates only its consumers. Keep one high-level symbolic source and remove monolithic generated includes from exact consumers. [DONE]
    - Emit relativistic `core`, `flux`, `sources`, `c2p_rhd`, and `c2p_rmhd` groups plus Z4c `geometry`, `constraints`, `gauge`, and `evolution` groups. Keep umbrella headers only for aggregate compile/consistency checks; production headers include precise generated groups. Z4c state access, derivatives, constraints, and evolution have separate pointwise interfaces. Touching generated constraints rebuilds only the three dimension-specific constraint owners; touching generated evolution rebuilds only the three evolution owners. Each cache-backed all-dimension rebuild completes in about 0.45 s. [DONE]
  - Establish narrow reconstruction, Riemann, C2P, and AMR policy factory/registry boundaries selected outside hot loops. The fused flux registry owns host reconstruction and Riemann selection while preserving one specialized point kernel. WENO-Z, MP5, and TENO-5 formulas have method-local headers; a cold MP5 edit rebuilds exactly its component/characteristic owners plus link in 19.20 s. C2P/cache and signal-speed dispatch have compiled host factories. [DONE]
  - Rebuild all dimensions, compare cold and cache-hit compile timings and duplicate object symbols, then run focused method, quick-suite, and performance gates before completing the cycle. All dimensions build; current 1D bundle objects total 6.9 MiB versus 41 MiB before extraction and no longer depend on reconstruction, Riemann, RHS, or RHD/RMHD flux-system headers. The remaining repeated weak symbols are specialized inline point-kernel/template machinery retained for production performance, not duplicate physics implementations. The complete HD/MHD reconstruction matrix passes 16/16 and the Riemann matrix passes 8/8; RHD/RMHD AMR and subcycling controls pass 3/3; Z4c passes 13/13; generated-code consistency passes. The aggregate quick gate retains only the unchanged spherical AMR-subcycling ledger edge (`1.074e-10` versus `1.000e-10`). Runtime benchmark ratios are 0.92--0.98 of the frozen baseline, and an unchanged production build is subsecond with cache. [DONE]

## Cycle 167 [completed]
- Audit and remove remaining module-boundary leaks after the generated-family split, preserving one implementation and specialized hot kernels. [DONE]
  - Measure public-header fanout and classify each broad dependency as schema, inline point-kernel requirement, or implementation leakage. `AmrSolverBase` textually owned about 2,950 lines of diagnostics, conservation, gravity synchronization, flux-register, solid-geometry, and FillPatch implementation. Move the non-template bodies to one compiled owner per dimension; retain only the four subcycling-dependent FillPatch templates and small debug declarations in the class contract. [DONE]
  - Extract the shared scalar reconstruction contract so MP5, TENO-5, WENO-Z, RHD, and RMHD consumers do not import unrelated characteristic formulas. Keep Riemann and mapped-patch flux formulas together because the mapped interface directly instantiates the same fused Riemann kernels; no duplicate contract or adapter was added. [DONE]
  - Remove stale umbrella and transitive includes from the AMR base. Diagnostics, initial-data state kernels, and signal-speed kernels are source-private implementation headers with exactly one owner. Checkpoint, CT, mapped-fill, patch-metric, primitive-cache, and solid-conservation dependencies are included by the compiled or evolution-template owner that actually uses them. Generated umbrella headers have no production consumers. [DONE]
  - Verify incremental ownership. Touching diagnostics, solid geometry, or FillPatch implementation rebuilds exactly one 1D owner plus the link, each in about 0.4--0.5 s with cache. The solver-visible AMR contract is now 753 base lines plus 229 lines of required declarations/templates, while 6,674 lines of substantial implementation are private to nine compiled owners. [DONE]
  - Rebuild all dimensions and run generated, focused method, relativistic/Z4c, quick-suite, deep-AMR/CT/solid, and performance-sensitive gates. All dimensions build; generated consistency passes; all 24 HD/MHD reconstruction and Riemann cases pass; RHD/RMHD retain second-order convergence; Z4c passes 13/13; deep L7 aligned AMR, CT aligned regrid, and both half-obstacle orientations pass 4/4. The aggregate quick gate retains only the unchanged spherical 3D subcycling ledger edge (`1.074e-10` versus `1.000e-10`). Confirmed benchmark medians are 0.92--0.96 of the frozen baseline; the initial 4-rank outlier was host/MPI contention and reran at 0.92. [DONE]

## Cycle 168 [completed]
- Perform a read-only architecture audit of the complete production AMReX implementation before any further refactoring. [DONE]
  - Inventory source/header size, ownership, directory cohesion, trivial forwarding translation units, and single-consumer implementation headers. The tree has 59 files directly under `src/`, 92 directly under the public include root, 6,674 lines in nine single-consumer implementation headers, 4,226 lines of class-body solver sections, and 6,620 lines of checks/benchmarks mixed into `src/`. [DONE]
  - Inspect dependency direction, generated/private/public boundaries, CMake ownership, and the remaining `AmrSolverBase` responsibilities. `AmrSolverBase` remains a 753-line dependency hub with approximately 254 method declarations and 77 data-member declarations; CMake source groups provide compile ownership but the filesystem and class ownership do not reflect the documented layers. Generated equation-family ownership and specialized flux/kernel bundle boundaries are materially better and should be preserved. [DONE]
  - Rate the current architecture and produce a concise, prioritized cleanup plan that preserves numerical behavior and measured compile/runtime performance. Overall maintainability is approximately 5.5/10: strong numerical contracts and validation, weak navigation, encapsulation, configuration cohesion, and repository organization. No solver code was changed and no numerical tests were needed for this read-only audit. [DONE]

## Cycle 169 [completed]
- Define the architecture decisions needed before the AMReX cleanup begins. [DONE]
  - Keep one indexed symbolic equation catalog, organized by physical sector, and generate both C++ kernels and human-readable equation documentation from that source. Emission backends may remain internally split by artifact family behind one generator interface; users should not need to inspect emitter machinery to review equations. [DONE]
  - Recommend an ETK-light framework: one stable AMR hierarchy and lifecycle coordinator, typed field/config registries, explicit evolution stages, and independently owned physics/numerics modules, without implementing a dynamic thorn/plugin scheduler solely for architectural resemblance. [DONE]
  - Convert code conventions from prose-only guidance into enforceable checks for public/private boundaries, source placement, wrapper-only files, implementation-header consumers, and dependency direction. Preserve input/output/checkpoint contracts and numerical behavior while allowing internal APIs to change. [DONE]

## Cycle 170 [completed]
- Record the approved architecture constraints for the forthcoming AMReX cleanup. [DONE]
  - Build one indexed high-level symbolic equation catalog with labeled equations and generated human-readable documentation; retain modular optimized C++ emission behind one generator interface. [DONE]
  - Use an ETK-light framework with one implementation of shared field, centering, derivative, vector/tensor, transfer, and lifecycle operations. Keep conventions explicit, consistent, and enforceable without adding a dynamic plugin system. [DONE]
  - Preserve multipatch, moving-solid, and Newtonian self-gravity behavior in their proper modules without quarantine or deletion; leave unfinished integration clearly documented for later adaptation rather than adding fallback paths. [DONE]
  - Preserve external run contracts while allowing internal C++ APIs and source organization to change. Create a checkpoint commit before structural cleanup, then gate each phase on numerical, compile-time, and runtime baselines. [DONE]

## Cycle 171 [completed]
- Clarify equation-catalog and compatibility terminology before implementation. [DONE]
  - The labeled, human-readable symbolic equation definitions are the sole source of truth. Optimized C++ and any browsable rendered index are generated views, never separately maintained equation copies. [DONE]
  - External compatibility means avoiding accidental changes to existing input parameter names, checkpoint restart semantics, output GF names/centerings, and run tooling while reorganizing internal C++. It is a default cleanup constraint, not a prohibition against deliberate interface improvements. [DONE]

## Cycle 172 [completed]
- Correct the cleanup policy and equation-catalog presentation requirements. [DONE]
  - PyFluid owns every consumer and has no backward-compatibility requirement. Repository-wide convention changes may immediately update inputs, checkpoints, outputs, tools, tests, and documentation; remove the superseded path rather than retaining compatibility adapters, stale formats, aliases, or fallbacks. Temporary breakage during a coherent migration is acceptable. [DONE]
  - Present the symbolic source of truth as a labeled, paper-like catalog grouped by physical system and sector. The catalog remains executable symbolic input for code generation and also emits a complete Markdown/LaTeX/PDF equation reference containing the high-level PDE systems and labeled derived kernels. [DONE]

## Cycle 173 [completed]
- Replace the accumulated AMReX organization with an enforceable ETK-light framework and canonical field/equation catalogs. [DONE]
  - Freeze the currently validated AMReX state in a scoped baseline commit before structural replacement. Baseline commit: `dd314c5`. [DONE]
  - Define grid-function scalars/vectors/tensors, components, labels, centerings, roles, and physical-system requirements once in a high-level catalog consumed by C++ generation and equation documentation. Fluid, magnetic, face-centered CT, passive, Z4c, and auxiliary fields now come from `field_catalog.py`; the generated C++ descriptors, component IDs, JSON labels, and Markdown/LaTeX field reference are derived views. This also removed a real hydro-passive component-5 ambiguity in origin parity metadata. [DONE]
  - Consolidate production source ownership, remove wrapper-only and single-consumer implementation-header indirection, and separate production, generated, validation, and benchmark code. Production sources now live under `driver`, `framework`, `amr`, `numerics`, `initial_data`, `relativity`, `flux_kernels`, and `kernel_bundles`; checks, benchmarks, and tools are outside `src`. The architecture gate rejects root-level production files, `_impl.hpp` wrappers, non-instantiation pass-through sources, duplicated generated IDs, synthesized face-field names, and missing generated catalog artifacts. [DONE]
  - Consolidate runtime and postprocessed diagnostics around the same field schema and one lifecycle contract. Runtime scalar columns come from `diagnostic_catalog.py`; field output and plotting consume generated field metadata; postprocessed diagnostics declare their GF dependencies and no longer own duplicate GF labels. [DONE]
  - Rebuild all dimensions and run focused numerical, compile-time, and runtime performance gates before committing the completed cleanup. All production dimensions build. Generated consistency, architecture, hydro-passive, six-level MHD CT/resistive/auxiliary, and CT AMR-subcycling restart checks pass; restart maximum field difference is `3.06e-12` against `1e-10`. The quick suite passes every gate except the unchanged spherical 3D AMR-subcycling ledger edge (`1.074e-10` versus `1.000e-10`; 30/31 spherical cases pass). A no-content all-dimension production build is `0.53 s`; runtime benchmark throughput remains `0.94`--`0.99` of the frozen baseline. [DONE]

## Cycle 174 [completed]
- Continue simplifying the AMReX backend after the ownership/catalog migration; reduce interfaces and code, not merely relocate it. [DONE]
  - Inventory handwritten public-header exposure, include fanout, backend-only contracts, textual solver sections, and remaining duplicate/dead APIs. Necessary GPU point kernels are distinct from accidental public implementation; `config.hpp` and `AmrSolverBase` remain the two largest dependency hubs. [DONE]
  - Shrink the public contract to stable configuration, field, physical-system, and run interfaces. `AmrSolverBase`, `AmrSolver`, all solver sections, output/derived diagnostics, AMR transfer, source kernels, relativistic point systems, and other single-owner backend headers now live under their owning `src/` modules. The architecture gate prohibits those headers from leaking back into the shared include root. [DONE]
  - Reduce `AmrSolverBase` and textual solver-section coupling through composition or narrow compiled operations where this deletes declarations and dependencies without introducing pass-through objects. Seven profiler forwarding methods and the four-line AMR-tagging declaration section are removed. Further composition was rejected for this pass because the remaining solver sections are substantive template families rather than wrappers; adding pass-through objects would increase code and indirection without changing ownership. [DONE]
  - Prune duplicate configuration/diagnostic/output helpers and dead options, retaining one supported behavior and one source of truth for each operation. The public configuration-registry wrapper and unused Z4c include aggregator are removed. Host-only option catalogs now live in private `src/framework/config_options.hpp`; all host-side config derivation and parsing helpers are consolidated in one compiled `config_runtime.cpp`. Public `config.hpp` fell from 1,118 to 394 lines and now owns only `RunConfig`, stable types, and declarations. EOS and fixed-spacetime parameter schemas are separate canonical value types, so the config contract no longer imports 946 lines of pointwise physics implementation. A private option-catalog edit rebuilds exactly `config.cpp`, `config_runtime.cpp`, and the link in 0.32 s rather than invalidating numerical kernels. CMake source groups now mirror framework, numerics, flux-kernel, initial-data, and relativity ownership. [DONE]
  - Rebuild all dimensions and rerun architecture, generated consistency, quick numerical, compile, and performance gates before committing. All production dimensions build; architecture and generated physics checks pass; a no-content production build is 0.34 s. The quick suite reproduces only the unchanged spherical 3D AMR-subcycling energy edge (`1.074e-10` versus `1.000e-10`; the other 30 spherical cases pass). Benchmark ratios are 1.00, 0.94, 0.94, and 0.96 of the frozen baseline across the standard 2D/3D cases. Prior cleanup commit: `dd153ba`; this cycle is recorded by the containing scoped commit. [DONE]

## Cycle 175 [completed]
- Confirm the complete validation state after the final AMReX ownership cleanup before returning to relativistic development. [DONE]
  - The uninterrupted full AMReX suite passes all 46 gates in 803.85 s: architecture, 111/111 feature combinations, RHD, RMHD, fixed spacetime, performance/rank scaling, AMR/subcycling/restart, Cartesian and curvilinear methods, CT, diffusion, gravity, sources, EOS, analytic/reference problems, and dimensional/rank consistency. [DONE]
  - Fix a latent RMHD multi-FAB heap overread exposed by the full gate. Fourth-order Z4c upwind derivatives require three ghost cells, but spacetime coupling storage had hardcoded two; all spacetime storage widths now derive from the canonical field descriptors. The exact two-rank Balsara reproducer and the complete RMHD standard gate pass, including exact restart and three-level AMR+CT. [DONE]
  - Repair stale validation contracts without adding solver exceptions: the resistive-current-sheet case now owns a consistent cell budget, curved RMHD uses a general RMHD IC rather than an RHD-only equilibrium, solid refinement satisfies the canonical quartic transfer stencil, scalar readers use the shared metadata-aware TSV path, and centered auxiliary fields use one shared field-aware reader. The global external-ledger tolerance is 2e-10, still 50 times stricter than the ordinary conservation gate; spherical methods pass 31/31. [DONE]
  - Move the `AmrSolverBase` constructor from the common implementation header into the existing AMR level/storage source. This one-time integration rebuild was broad, but subsequent ownership-policy edits no longer invalidate every kernel bundle. A no-content all-dimension production build is 0.46 s. Runtime benchmark ratios are 0.99, 0.91, 0.94, and 0.96 of the frozen baseline; all performance gates pass. Relativistic development can resume on the cleaned boundaries. [DONE]

## Cycle 176 [completed]
- Resume relativistic stellar initial-data development without weakening the cleaned module boundaries. [DONE]
  - Inventory the existing TOV, CTT, CTTK, XCTS, EoS, materialization, and validation ownership; distinguish implemented behavior from incomplete RNS work. TOV, CTT, CTTK, and XCTS solvers exist; rotating-fluid equilibrium/RNS matter data do not yet exist and must precede a meaningful rotating CTT/CTTK/XCTS comparison. [DONE]
  - Establish one dedicated EoS module containing analytic and piecewise-polytropic definitions, named realistic PWP presets, and the future tabulated-EoS boundary; keep every supported consumer on that source of truth. Public EoS definitions now live under `include/pyfluid_amrex/eos/`, host assembly under `src/eos/`, and `config.hpp` no longer imports the EoS service. Read-fit SLy, APR4, H4, and MS1 presets use explicit CGS-to-`G=c=M_sun=1` conversion and the same continuous up-to-eight-segment runtime representation as custom PWPs. RMHD remains ideal-gas-only until its C2P contract is generalized; no preset-specific exception was introduced. [DONE]
  - Validate spherical TOV materialization through CTT, CTTK, and XCTS before adding rotation, then define the smallest shared extension needed for rotating-star/RNS data rather than adding solver-specific branches. One common staggered TOV validator now runs all three formulations. CTT, hybrid CTTK, and XCTS converge in two matter iterations with relative matter updates below `1e-10`; elliptic Hamiltonian/lapse residuals are below `2.1e-14` and the CTTK momentum residual is `5.5e-13`. The deliberately coarse sampled-grid Z4c norms remain separately reported at `O(1e-3)`. [DONE]
  - Fix provider-state floor projection so every floored RHD primitive recomputes `eps` and pressure through the selected EoS before P2C. This removed real C2P failures in the named-PWP stellar atmosphere without adding a preset-specific path. The end-to-end hybrid-SLy TOV smoke reports `M=1.2099`, `R_areal=7.97` and passes accepted-state validation. [DONE]
  - Gate each step on architecture, focused relativistic/initial-data tests, incremental compile time, and existing nonrelativistic behavior. All dimensions build; architecture, generated EoS checks in 1D/2D/3D, focused elliptic/CTT/CTTK/XCTS, conformal-TOV comparison, realistic-EoS TOV, and the complete quick suite pass. No-op all-dimension compile is `0.46 s`; runtime benchmark ratios are `0.90`--`0.99` of the frozen baseline and all performance gates pass. [DONE]

## Cycle 177 [pending]
- Add rotating relativistic stellar equilibrium data through one provider consumed by the existing CTT/CTTK/XCTS materializer. [PENDING]
  - Select and document the equilibrium formulation and rotation-law inputs before coding; do not approximate an RNS star by assigning azimuthal velocity to a spherical TOV solution. [PENDING]
  - Extend the stellar provider with the minimum axisymmetric field representation needed by the equilibrium solve, then validate the zero-rotation limit against the existing TOV provider before comparing rotating constraint formulations. [PENDING]

## Cycle 178 [completed]
- Consolidate diagnostics into one generated, field-catalog-aware system shared by initial data, evolution, and offline analysis. [DONE]
  - Inventory the current scalar catalog, hardcoded AMR reductions, constraint kernels, interpolation support, and output contracts. [DONE]
  - Define diagnostics and their scalar/vector/tensor integrands once at high level, with individual selection and ETK-like named groups. [DONE]
  - Use shared volume, extrema/norm, and surface executors rather than diagnostic-specific loops; retain fused kernels where performance requires them. Generated integrands now feed the existing fused conserved reduction and one shared spacetime hierarchy executor with composite masks and nodal ownership. [DONE]
  - Add first GR diagnostics: total rest mass, Hamiltonian norms, and ADM mass/angular momentum in surface and constraint-assisted volume forms. The volume forms use the Hamiltonian and momentum constraints to replace curvature divergences with matter and algebraic geometry terms; they are independent of the surface quadrature. [DONE]
  - Keep the ADM convergence gate an isolated uniform-hierarchy quadrature test. ADM surface-volume disagreement decreases from 0.95% at 16^3 to 0.24% at 32^3, and the finest constraint-assisted mass is within 2.5% of the TOV provider mass. [DONE]
  - Validate analytic/reference values, runtime/initial-data agreement, generated artifacts, compile time, numerical gates, and performance before completion. Generated consistency, architecture, focused Z4c/ADM, all-dimension production build, and the complete quick suite pass. Standard runtime benchmarks remain 0.93--0.99 of the frozen baseline. [DONE]

## Cycle 179 [completed]
- Diagnose the reported partial-AMR CTT initial-data failure before claiming composite CTT initial data. [DONE]
  - The multilevel elliptic solve and staggered materialization are coherent: the MLMG residual is `8.43e-17`, the spherical ADM surface mass is `6.6884e-4` versus `6.7418e-4` on the uniform control, and genuine fine-only nodes improve over coarse sampling by a factor `0.508`. The previous near-zero-monopole result came from a validator that sampled only fine nodes coincident with coarse nodes. [DONE]
  - Keep the remaining coarse-fine derivative error visible rather than calling it repaired. Fourth-order constraint stencils crossing the coarse-fine boundary dominate the full Hamiltonian norm (`L2=1.18249e-4`, `Linf=1.29427e-3`), while the same-level interior is `L2=6.78211e-6`, `Linf=4.23869e-5`. Runtime diagnostics and the CTT validator now report full, same-level-interior, and coarse-fine contributions separately. A higher-order coarse-fine derivative closure remains future numerical work. [DONE]

## Cycle 180 [completed]
- Upgrade the TOV implementation into one reusable enthalpy-based stellar-solution module for direct EoS science, evolution materialization, and constraint-solver validation. [DONE]
  - Replace floor-terminated fixed-radius integration with a robust enthalpy-coordinate solve whose physical surface is the zero-pressure endpoint, while retaining one exact Schwarzschild exterior and one evolution sampling path. [DONE]
  - Define one `TovSolution` containing radial profiles and standard observables: gravitational, baryonic, proper, and Komar masses; circumferential/isotropic radii; compactness, binding energy, surface redshift, and tidal response where the EoS contract permits it. [DONE]
  - Add direct sequence output and plotting across central density and named/custom EoS choices without launching an evolution, using the same solver and observable definitions as initial-data materialization. [DONE]
  - Extend the generated diagnostics catalog with the generally meaningful stellar/GR volume and surface quantities, reusing constraint-assisted forms rather than introducing naive divergence copies. [DONE]
  - Strengthen direct TOV and CTT/CTTK/XCTS checks against the same observables; keep the known fragmented partial-AMR CTT defect explicit and separate. The direct Gamma=2 solution has tolerance sensitivity near `1e-12`, independently integrated Komar/gravitational masses agree to `2.2e-13`, and named SLy/APR4/H4/MS1 sequence controls pass. At the deliberately coarse 24^3 staggered materialization, CTT/XCTS proper and baryonic masses agree with the direct solution within `0.58%`; XCTS Komar volume agrees within `0.48%` and Komar surface/volume agree within `0.14%`. Hybrid CTTK is correctly validated as different nonzero-mean-curvature free data rather than falsely required to reproduce the time-symmetric TOV masses. [DONE]
  - Rebuild all dimensions, run focused stellar/EoS/Z4c and standard gates, record compile/runtime impact, and document the validated limits. All production dimensions build; generated consistency, direct TOV/elliptic, realistic-EoS TOV, the complete three-formulation stellar matrix, architecture, and every quick-suite gate pass. Standard runtime benchmark ratios remain `0.93`--`0.99` of the frozen baseline. The then-open partial-AMR CTT report remained Cycle 179 work and was not hidden by these uniform/full-domain checks. [DONE]

## Cycle 181 [completed]
- Add the one-layer Gamma=2 isentropic ideal-gas/polytrope to the direct TOV EoS comparison without duplicating the EoS implementation. [DONE]
  - Carry a human-readable EoS label in sequence metadata so custom curves remain self-describing in shared plots. [DONE]
  - Regenerate and verify the combined sequence plot and focused build. The sequence target builds, all five 80-point sequences complete, the custom curve reaches `Mmax=1.637272` at `R=7.626780`, and the combined plot renders with the metadata label. [DONE]

## Cycle 182 [completed]
- Harden nonrotating relativistic-star evolution and diagnostics before starting RNS/KEH work. [DONE]
  - Audit which TOV, CTT, CTTK, and XCTS results are materialization checks versus physically faithful evolutions; retain the fragmented partial-AMR CTT defect as explicit unfinished work. [DONE]
  - Run short 3D Cartesian Cowling and coupled-Z4c TOV evolutions from the same initial data, with runtime rest mass, ADM/Komar masses, density stability, and constraint diagnostics. [DONE]
  - Consolidate any missing stellar-evolution validation into the existing Z4c/TOV harness rather than adding a parallel script. [DONE]
  - Keep GR diagnostic integrands coordinate-independent and later generalize the shared volume/surface quadrature executors to curvilinear geometry; do not expand each ADM/Komar integrand separately. Relativistic evolution remains Cartesian-only in the current cycle. [DONE]
  - Audit relativistic reuse of Newtonian initial conditions and document the boundary between fixed-Minkowski SR tests and constraint-consistent coupled GR initial data. [DONE]
  - Re-run focused and quick gates, then report remaining cleanup and physics gaps before RNS. [DONE]
  - Project accepted below-floor relativistic states at every common RK-stage entry and flatten reconstruction only on faces touching the explicit atmosphere. C2P exhaustion remains fatal. The focused two-level 3D TOV matrix passes: Cowling changes maximum density by `5.28e-3`; evolved Z4c preserves baryon mass to `2.01e-11`, changes maximum density by `1.22e-3`, changes constraint-assisted ADM volume mass by `1.29e-4`, and keeps Hamiltonian L2 below `1e-4`. The outer Cartesian-box ADM surface value is reported but is not treated as an evolved conservation law. [DONE]
  - RHD KH with and without AMR subcycling and RMHD invariant-field AMR pass. All production dimensions build and the complete quick suite passes, including curvilinear geometry/origin/subcycling/regrid gates. The no-op production build is `0.43 s`; standard runtime ratios are `1.00`, `0.95`, `0.95`, and `0.96` of the frozen baseline. The then-open partial-AMR CTT report remained Cycle 179; curvilinear relativistic evolution and interior spherical extraction were future work. [DONE]

## Cycle 183 [completed]
- Complete the next relativistic diagnostics and AMR infrastructure layer without duplicating physics by coordinate system. [DONE]
  - Define the GR total-energy contract: baryon mass is a matter conservation law, while ADM mass is the isolated-system total energy at spatial infinity. A finite-radius evolved ledger requires matter flux plus gravitational-wave/Bondi flux through the same extraction surface; no fictitious local gravitational-energy density was introduced. [DONE]
  - Build one interpolated spherical-surface executor, later extensible to deformed parametric surfaces, that constructs geometry once and evaluates all requested generated scalar/vector/tensor integrands. It maps physical points into Cartesian, cylindrical, or spherical mesh coordinates, selects the finest complete AMR interpolation cell, and contracts scalar or vector surface charges through one implementation. [DONE]
  - Diagnose the partial-AMR CTT report at the hierarchy/source/materialization boundary. Cycle 179 records the corrected fine-node comparison and the separately visible coarse-fine derivative error; no transfer repair, diagnostic mask, or weakened tolerance was added. [DONE]
  - Review production GRMHD AMR practice and implement one scale-free fluid aggregate plus one scale-free spacetime aggregate. Their directional maximum feeds the existing single hierarchy error field, following the CarpetX/AsterX ownership model without making every evolved component an independent driver. [DONE]
  - Audit generated curvilinear RHD/RMHD/Z4c support and record the implementation boundary. Fixed-spacetime curvilinear Valencia can reuse the generated physical flux/source equations with mapped measures and basis transforms. Evolved curvilinear Z4c requires a generated reference-metric formulation, parity/origin regularity, nodal AMR transfer, and matching boundary conditions; it is not a Jacobian-only extension of the Cartesian formulation. [DONE]
  - Rebuild all production dimensions and run the surface interpolation check, architecture and generated-code gates, focused CTT, TOV, RHD/RMHD AMR checks, and the complete quick suite. All pass. Standard Newtonian point kernels are unchanged; a new performance snapshot was not taken because the new surface work is opt-in and the spacetime aggregate is inactive for Newtonian runs. [DONE]

## Cycle 184 [pending]
- Extend relativistic evolution to curvilinear single-patch meshes through one generated coordinate-independent implementation. [PENDING]
  - First add fixed-spacetime cylindrical/spherical RHD using generated Valencia equations, mapped finite-volume measures, and physical-to-mesh basis transforms; validate against Cartesian Minkowski controls before adding magnetic state. [PENDING]
  - Then add fixed-spacetime RMHD and its CT geometry through the same mapped contracts. Do not add coordinate-specific fluid equations. [PENDING]
  - Treat evolved curvilinear spacetime as a separate reference-metric Z4c formulation generated from the same tensor source, with explicit origin/parity, nodal AMR transfer, and boundary validation. [PENDING]

## Cycle 185 [in progress]
- Round out spherical extraction with one generated gravitational-radiation and multipole pipeline. [WIP]
  - Extend the shared parametric sphere with angular quadrature data and retain its center/radius as immutable geometry. The extraction tetrad is constructed after hierarchy interpolation at Gauss points, not as a second surface sampler or a singular grid field. [DONE]
  - Generate electric/magnetic Weyl and `Psi4` from the existing Z4c geometry, matter projections, and tensor algebra. Grid kernels produce regular E/B tensors; the generated surface kernel uses the documented ETK/Baker-Campanelli-Lousto-style `phi,r,theta` orthonormal frame and `Psi4=(E-iB)_ij mbar^i mbar^j`. [DONE]
  - Add one spin-weighted spherical-harmonic reducer for requested `(l,m)` modes and one compact long-form `multipoles.tsv` history rather than hardcoded mode columns. Real radiation reductions use the generated scalar history; restart output requires aligned scalar and multipole histories. [DONE]
  - Postprocess instantaneous matter-energy flux and `Psi4` modes into news, strain, GW luminosity, cumulative radiated energy, and an explicitly finite-radius ADM balance. Missing ADM data remain blank, finite-time integration constants are documented, and normal `run_case.py` workflows invoke the postprocessor automatically. [DONE]
  - Validate spherical geometry, AMR interpolation, spin-weighted harmonic orthogonality and injected-mode recovery, generated Minkowski E/B/Psi4, finite matter current, stationary vacuum extraction, checkpoint/restart append semantics, synthetic news/strain/energy integration, and a nonzero evolved linear gravitational wave. The full mode-vector quadrature change decreases from `5.86e-3` at 10-to-20 polar points to `2.24e-3` at 20-to-40; stationary `Psi4` L2 remains below `8.3e-33`. Architecture, all-dimension production build, generated-kernel validation, and the complete quick suite pass. [DONE]
  - Extraction-radius convergence toward null infinity still requires a compact isolated radiative source and multiple physically meaningful radii. The current plane-wave control validates nonzero extraction and surface resolution but is not an honest `r Psi4` radius-extrapolation test. [PENDING]
  - Consolidate surface diagnostics by moving real radiation reductions into the generated scalar history, retaining one generalized complex `multipoles.tsv`, and adding raw azimuthal Fourier surface modes `C_m[f] = integral f exp(-i m phi) dA` for catalog-selected scalar fields. Runtime reductions are ephemeral: they are emitted at the scalar cadence and discarded; normalized amplitudes and phases remain postprocessing products. The restart check writes exactly three aligned scalar/multipole samples, uniform `rho` and `pressure` have `C_0=4 pi r^2` with higher modes at roundoff, and the generalized strict TSV schema uses `ell=none` for non-spherical modes. Surface/multipole MPI, radiation restart/evolution, postprocessing, architecture, generated physics, all-dimension production build, and the complete quick suite pass. [DONE]

## Cycle 186 [in progress]
- Complete typed global-charge and GRMHD runtime diagnostics without creating parallel output lifecycles. [WIP]
  - Keep one `scalars.tsv` row cadence while teaching the generated diagnostic catalog that related columns are scalar, vector, or tensor components. Do not add a separately synchronized `vectors.tsv`; expose whole-vector selection and plotting through catalog metadata. [DONE]
  - Add generated ADM linear momentum surface and constraint-assisted volume vectors from translational asymptotic generators, alongside the existing generated rotational ADM angular momentum vectors. [DONE]
  - Replace the Newtonian-only interpretation of `magnetic_energy` on RMHD runs with generated relativistic diagnostics: comoving `b_squared`, enthalpy magnetization, plasma beta, Lorentz factor, and an explicitly defined Eulerian electromagnetic-energy volume integral. Add extrema/reductions only where physically meaningful. [DONE]
  - Add generic spherical mass-accretion, fluid/EM energy-flux, and magnetic-flux diagnostics after validating their sign, metric measure, and closed-surface contracts. Track C2P fallback/failure and atmosphere injection separately as algorithmic health diagnostics. [PENDING]

## Cycle 187 [completed]
- Separate initial ambient atmosphere from runtime C2P recovery policy, and validate Cartesian reflection-symmetry evolution for relativistic stars. [DONE]
  - Keep one EoS thermodynamic floor-state closure, but do not force initial atmosphere construction and runtime recovery to share a spatial profile or application policy. Confirm CTT/CTTK/XCTS sources use the actual floored initial primitive state. [DONE]
  - Cycle 189 implemented the graded initial atmosphere as an initial-data
    policy separate from runtime C2P recovery. [DONE]
  - Correct scalar/vector parity factors for GR volume charges on half, quadrant, and octant Cartesian domains. Keep spherical extraction unavailable until the surface sampler can reconstruct omitted symmetry sectors with field parity. Scalar masses use the domain-volume factor, polar ADM momentum uses reflection parity, and axial ADM angular momentum uses the corresponding axial parity. [DONE]
  - Add a matched-resolution full-domain versus octant TOV control covering initial data, short Cowling evolution, AMR, constraints, rest mass, and GR volume diagnostics. The two-level octant run agrees with the full-domain initial volume charges to at worst `1.32e-5`, preserves every symmetry-forbidden ADM momentum/angular-momentum component at zero, and passes its short Cowling evolution gate. [DONE]

## Cycle 188 [completed]
- Improve the common centering-transfer closure near physical and symmetry boundaries without adding diagnostic- or physics-specific interpolation paths. [DONE]
  - Inventory every `c2v`, `v2c`, cell/face, and related averaging stencil and identify the single implementation used by initial data, evolution, and diagnostics. All MultiFab transfers route through `transfer_grid_field_centering`; generated tensor-product kernels own the topology algebra. [DONE]
  - Reproduce the boundary-order degradation with a focused polynomial control before changing the stencil. The centered kernel was correct, but valid boundary destinations assumed mathematically valid source ghosts; initial-data physical ghost callbacks were no-ops, so boundary values could lose order or read undefined extensions. [DONE]
  - Implement the highest consistent one-sided closure supported by available data, then validate interior order, boundary order, and existing relativistic initial-data consumers. The generated bounded transfer shifts the same order-`2/4/6/8` Lagrange stencil inside nonperiodic physical domains, leaves periodic and interior fast paths centered, and uses filled ghosts only for destination ghost cells. Polynomial exactness now covers every centering pair, supported dimension, order, and low/high boundary. Generated-kernel checks, the focused full/octant TOV evolution matrix, and the complete cell/vertex CTT/CTTK/XCTS elliptic test pass. [DONE]

## Cycle 189 [completed]
- Add explicit flat and convergent power-law initial atmospheres without coupling their spatial profile to runtime C2P recovery. [DONE]
  - Define one initial-data atmosphere policy with a constant inner state, optional radial power-law tail, and a minimum state identical to the lower runtime recovery floor. Apply it before relativistic P2C and elliptic matter-source construction. [DONE]
  - Generated physical checks cover flat compatibility, power-law scaling,
    ideal/cold EoS closure, and the exact runtime-floor endpoint. Maintained
    coupled, Cowling, octant, and XCTS TOV inputs exercise the power-law profile
    above a lower recovery floor. [DONE]

## Cycle 190 [completed]
- Consolidate derived physical-coordinate geometry behind one host/device provider API rather than storing or recomputing ad hoc radius fields. [DONE]
  - Define Cartesian spherical and cylindrical radius operations once and route initial data, fixed spacetime, gravity, solid geometry, and coordinate conversion through them. Do not allocate derived-radius GFs. [DONE]
  - The shared header is used by initial data, fixed spacetime, gravity, solid
    geometry, coordinate locations, and patch maps; generated physical checks
    and the all-dimension production build pass. [DONE]

## Cycle 191 [completed]
- Make relativistic primitive/conserved inversion coverage rigorous enough to catch cold, low-density stellar-surface failures. [DONE]
  - Extend the existing generated-physics gate across logarithmic density and thermal-energy ranges, static through relativistic velocities, flat and nontrivial spatial metrics, and ideal/polytropic/hybrid EoS choices. Check both `C2P(P2C(P))` and `P2C(C2P(U))` with scale-aware component tolerances. The three EoS matrices pass with worst conserved errors `2.22e-10`, `2.58e-8`, and `4.04e-10`. [DONE]
  - Require invalid conservative states to fail explicitly rather than being repaired. Negative density, nonpositive `D+tau`, and nonfinite states return `InvalidConservedState`. [DONE]
  - Run one const, repair-free C2P/P2C preflight over every active relativistic cell after initial hierarchy materialization and before iteration zero. The two-level low-atmosphere TOV hierarchy passes over 609664 cells with conserved and primitive idempotence errors below `3.7e-14`. [DONE]

## Cycle 192 [completed]
- Diagnose the low-atmosphere coupled-TOV failure now localized beyond initialization. [DONE]
  - The shared startup preflight passes over 609664 active cells, but the first no-subcycling evolution fails. With ghost-inclusive finite/physical contracts enabled, the first invalid state is now localized before the initial RHS: level-2 coarse-fine FillPatch produces negative conserved density (`D=-5.18741e-9`) in stellar-surface ghosts at `t=0`. This excludes initial data, active-cell C2P/P2C, fluxes, and RK updates as the first cause. [DONE]
  - Quartic conservative interpolation is not positivity preserving at the stellar surface. Ambiguous transfer names were replaced with explicit `component-conservative` and `coupled-conservative` choices, and the coupled full-state limiter is the production default. All level creation, regrid, FillPatch, subcycling, and repair seeds continue through the single selector. [DONE]
  - The post-transfer failure was a source-aware forward-Euler admissibility defect, not another interpolation failure. One conservative RHD flux limiter now blends final shared faces against PWC LLF, and the RHS, physical-boundary ledger, and reflux register all consume those same final fluxes. The limiter receives the actual RK stage span rather than stale AMR level timestep storage. [DONE]
  - Runtime stage validation now distinguishes C2P admissibility from the stricter startup two-sided C2P/P2C identity check. Strict identity remains a preflight requirement; accepted RK inputs require finite, admissible states and successful C2P without treating cold-surface roundoff as a physical failure. [DONE]
  - Initial atmosphere and runtime recovery are separate contracts. Evolution now owns one precomputed flat thermodynamic floor state; it no longer samples the initial graded atmosphere in projection or face kernels. Runtime replacement is density-triggered, so a pressure floor cannot delete otherwise valid conserved mass and momentum. [DONE]
  - Canonical `run.in` replay no longer serializes an unset automatic `amr.min_feature_length` as explicit zero. A fresh replay reproduces the original hierarchy and every non-performance scalar exactly; the focused TOV validator rejects recurrence of the unset-to-explicit conversion. [DONE]
  - The two-level SSPRK3/subcycled TOV matrix passes for Cowling, evolved Z4c, and octant symmetry. External rest-mass ledger errors are at most `4.0e-15`; raw finite-box mass changes remain separately visible. Cowling/evolved maximum-density changes are `5.28e-3` and `1.22e-3`, and Hamiltonian L2 stays below `7.2e-5`. [DONE]
  - RHD KH with and without AMR subcycling, RMHD invariant-field AMR, architecture, generated physics, all production dimensions, and the complete quick suite pass. [DONE]
  - The apparent Newtonian performance collapse was Open MPI 5 binding local ranks to Apple logical cores after the host update. The canonical local profile now uses `--bind-to none`; fresh three-repeat ratios are `0.99`, `0.94`, `0.93`, and `0.96` of the frozen baselines. [DONE]

## Cycle 193 [completed]
- Repair the partial-AMR CTT matter handoff and characterize the remaining coarse-fine derivative truncation without adding a CTT- or diagnostic-specific transfer. [DONE]
  - Trace the common nodal ghost-fill and fourth-order GF derivative contract used by CTT constraints, runtime Z4c constraints, ADM diagnostics, and Z4c evolution. The elliptic solve converges to `2.86e-16`; the nonfinite constraint values instead came from 632 nonfinite entries in derived vertex matter. [DONE]
  - Derive Eulerian matter only on valid cells, synchronize the derived cell hierarchy through the selected conservative AMR interpolater, then use the one common fourth-order cell-to-vertex transfer. This removes C2P evaluation on interpolated coarse-fine ghost states without repairing or mutating the fluid state. [DONE]
  - Keep one nodal AMR interpolater with exact injection at coincident coarse nodes and bounded cubic interpolation elsewhere. Remove the redundant spacetime interpolater wrapper. A generated one-sided fourth-order diagnostic experiment was rejected and removed because it worsened the normalized coarse-fine Hamiltonian contribution rather than improving the elliptic solution. [DONE]
  - Strengthen the focused partial-AMR CTT control with 16^3 and 32^3 base grids. The global Hamiltonian L2 decreases from `1.18206e-4` to `7.11698e-5` and the same-level interior from `6.77610e-6` to `2.83378e-6`; fine `chi` improves over coarse interpolation by a factor `0.518`. The normalized coarse-fine band remains `1.49e-4` to `1.57e-4`, explicitly exposing the second-order multilevel elliptic/fourth-order pointwise-diagnostic mismatch. [DONE]
  - Architecture, generated-kernel consistency, all production dimensions, the six-check focused CTT gate, and the three-check evolved TOV/subcycling gate pass. A requested three-rank rerun was not executable because external MPI-socket approval was unavailable; this is not counted as solver validation. [DONE]

## Cycle 194 [completed]
- Make AMReX agent continuity durable across context compaction by consolidating the current source-of-truth map, active limitations, validation boundaries, and external Spark review triage. [DONE]
  - Preserve the existing dirty checkout and recover current state from `AGENTS.md`, the documentation set, recent continuity cycles, and the three read-only Spark review artifacts. [DONE]
  - Check Spark's concrete final-pass findings and its six claimed high-impact areas against the current source. Three small items remain accepted: fail on lost plotfile metadata, remove one unused auxiliary-cadence argument without changing semantics, and build auxiliary names/count once per centering. The partial patch-count bug and missing validation-command builder are stale; broad config/factory registries are rejected as blanket work; initial-data, hierarchy-interpolation, and CT-register factoring remain deferred until a concrete ownership or correctness benefit is demonstrated. [DONE]
  - Register the local AthenaK source/build as an independent, provenance-tracked code-to-code reference. Record its usable source and executable paths, matched upstream commit, license boundary, high-value test routes, and a comparison protocol that does not mislabel numerical output as analytic. [DONE]
  - Add a compact durable knowledge file and reference/router file, then teach `AGENTS.md` and the AMReX README to use them without duplicating the full design documents. [DONE]
  - Verify all new cross-references and report the implementation-ready review queue without claiming source or numerical validation that was not run. All routed paths exist; tracked and untracked documentation has no whitespace errors; feature-case and suite-step discovery succeed; `check_architecture.py` passes; and AthenaK's CPU executable reports a valid double-precision serial configuration. No PyFluid solver source, compile, MPI launch, or numerical gate was part of this documentation/review cycle. [DONE]

## Cycle 195 [completed]
- Reclassify the Spark auxiliary-output findings from user triage and explain the large AthenaK/PyFluid clean-build difference from the current build graphs. [DONE]
  - Demote metadata append failure to opportunistic defensive hardening. The remaining failure window requires an output target that became unavailable or unwritable between creating the output and reopening its metadata sink; it is worth failing honestly if touched, but is not a likely normal-run defect or a first implementation priority. [DONE]
  - Accept one ordinary grid-function output cadence with advanced overrides: state and auxiliary snapshots should normally inherit the same requested cadence, while explicit auxiliary write and cached-derivative compute cadences remain available. Scalars and checkpoints retain independent lifecycles. Exact sentinel/input migration is an implementation decision and must preserve an explicit cached-only mode. [DONE]
  - Accept catalog-owned auxiliary output layout. `field_catalog.py` already owns GF identity, components, role, requirements, and centering; generated/runtime descriptors should be filtered once into each output layout, and `output.hpp` should only execute that layout rather than define a second name/component/centering policy. [DONE]
  - Compare existing optimized CPU build artifacts without cleaning the dirty tree. Both codes use `-O3 -DNDEBUG -mcpu=native`; the difference is not a debug-versus-release build. AthenaK links 144 application objects totaling about 13.5 MiB into one runtime-dimensional executable. PyFluid links 64 objects per AMReX dimension, about 27.9/57.5/79.8 MiB for 1D/2D/3D, or 165.2 MiB total. Its eight reconstruction-owned flux bundles account for about 133.3 MiB (80.7 percent) by repeatedly instantiating layout, EoS, direction, and solver paths. [DONE]
  - Record that the historical roughly five-minute figure is real but not a current cold-build measurement: Cycle 163 recorded 348 s for a full all-dimension build after a first split, versus 403--420 s before it; later ownership work improved incremental targets, but no clean current-tree all-dimension timing was taken in this read-only build-graph audit. [DONE]
  - Identify the feasible AthenaK lesson without pretending AMReX dimensions can be unified cheaply: keep the three `AMREX_SPACEDIM` executables, but reduce reconstruction/layout/EoS cross-product cloning and retain compile-time specialization only where a focused runtime benchmark earns it. A reduced-method development preset is an immediate optional speed path; a production restructuring requires compile and runtime benchmarks. [DONE]
  - Update the maintained knowledge snapshot and source-checked backlog, then verify documentation whitespace and the architecture ownership gate. Both pass. No solver source, compilation, cache invalidation, MPI launch, or numerical run was performed. [DONE]

## Cycle 196 [completed]
- Refine the accepted auxiliary cadence, decide the canonical GF declaration form, and define a scalable parameter-dispatch target that avoids both template cross-product growth and inner-loop runtime branching. [DONE]
  - Make the ordinary auxiliary compute cadence exactly inherit the resolved auxiliary output cadence. Preserve a separate compute cadence only when explicitly requested, including cached-only operation. [DONE]
  - Audit whether `field_catalog.py` is already authoritative across generated IDs/descriptors, allocation, output, checkpoint, diagnostics, and user selection; distinguish canonical physical GFs from unnamed temporary AMReX work arrays. It already generates field/component IDs, C++ descriptors, JSON/equation views, selectable auxiliary names, evolution field sets, and checkpoint/output metadata. Retain it as the CCL-like schema; close edge enforcement rather than adding another DSL. [DONE]
  - Define the enforceable boundary: every named evolved, retained, output-selectable, or checkpoint-eligible field requires a generated catalog ID/descriptor. Stage, RHS, transfer, and elliptic work MultiFabs remain unnamed scratch. C++ fill implementations switch on generated IDs but cannot define new field identity or metadata. [DONE]
  - Compare compile-time and runtime dispatch boundaries against the current flux owners and AthenaK, then record a design that scales to many configuration combinations without compiling their Cartesian product. The target is an immutable host-built plan, additive method ownership, coarse-grained uniform selection, and compile-time specialization only for required layout/topology or measured hot-path wins. [DONE]
  - Record that unlimited alternatives cannot have literally zero compile, code-size, and runtime cost simultaneously. Require no statistically meaningful default-runtime regression while measuring cold compile, object size, CPU throughput, and GPU behavior where available; benchmark AthenaK-style tile/row selection against separate launchers with a shared non-templated backend before changing the production factory. [DONE]
  - Update the durable knowledge, code-convention, and source-checked backlog contracts. `validate_codegen_flux.py` and `check_architecture.py` pass with the repository-local uv cache; documentation whitespace checks pass. No solver source, compile, MPI launch, or numerical run was performed. [DONE]

## Cycle 197 [completed]
- Move the durable catalog design from executable Python declarations toward CCL-like data-only text, and clarify AthenaK's selective specialization lesson. [DONE]
  - Select standard, versioned TOML rather than a custom `.ccl` grammar: it is comment-friendly, explicitly typed, and parsed by Python 3.13 `tomllib` without another dependency or maintained parser. [DONE]
  - Define `schema/fields.toml` as the first migration target. Python retains strict models, validation, tensor/component expansion, symbolic equations, and artifact emission; the first stage must reproduce generated IDs, descriptors, JSON, documentation, output, and checkpoint metadata byte-for-byte. [DONE]
  - Separate declarative identity from executable science. Later `diagnostics.toml` and `methods.toml` may own names, labels, groups, codes, requirements, and implementation references, while symbolic integrands, algorithms, coupled compatibility, and physical validation remain Python/C++. [DONE]
  - Record the specialization target: AthenaK specializes selected hot algorithms such as the Riemann solver while keeping reconstruction and dimensional activation as uniform runtime choices. PyFluid should establish its own small specialization budget by measurement rather than templating every independent parameter. [DONE]
  - Add `docs/catalog_schema.md`, update the durable router/knowledge/conventions/backlog, and verify path presence, documentation whitespace, and the architecture gate. All pass. No schema migration, generated artifact, solver source, compile, MPI launch, or numerical run was performed. [DONE]

## Cycle 198 [completed]
- Reclassify the template specialization problem in light of effective compiler caching and method-local incremental ownership. [DONE]
  - Confirm the distinction: the optimized Cartesian-product object graph makes cold builds and broad invalidations expensive, but cache hits and narrow method-local edits reuse unrelated objects and can provide an acceptable normal development loop. [DONE]
  - Accept only a current compile-matrix measurement as the next action and defer a production dispatch rewrite. Measure all-dimension cold, representative method-local miss, common/generated-header invalidation, and cache-hit builds separately; proceed only if CI, disk/memory, binary size, runtime, or developer latency evidence earns the risk. [DONE]
  - Update durable knowledge and the source-checked backlog. No production source, build/cache mutation, MPI launch, or numerical run was performed. [DONE]

## Cycle 199 [completed]
- Explain from the local ETK/CarpetX source and build artifacts how a large branch surface avoids PyFluid's current template Cartesian product. [DONE]
  - Inspect AsterX/ReconX CCL and hot flux/RHS owners. `param.ccl` declares seven reconstruction choices; AsterX converts the selected string to one enum and `reconstruct.hxx` switches on that enum in device code. `fluxes.cxx` likewise switches LxF/HLLE at runtime and templates only three directions and the selected EoS object type. [DONE]
  - Inspect Cactus modularity. `interface.ccl` declares GF groups and metadata, `schedule.ccl` conditionally schedules routines, compiled thorns are separate archives, and runtime activation/composition prevents most optional capabilities from becoming alternatives in one numerical kernel. [DONE]
  - Quantify the boundary without calling ETK intrinsically small: the local ET_CarpetX configuration has 85 thorn archives totaling about 402 MiB; `libthorn_AsterX.a` is about 16 MiB and its largest `fluxes.cxx.o` is about 4.7 MiB. The gain is additive owners, coarse scheduling, uniform runtime switches, and narrow incremental rebuilds, not a free clean build. [DONE]
  - Record the PyFluid lesson as four branch classes: compile dimension/required storage topology; host-schedule optional operations; use uniform run-wide enums for algorithm choices when benchmarks permit; keep inner-loop divergence for genuinely cell-dependent physics. No PyFluid or ETK source, build, cache, MPI launch, or numerical run was changed. [DONE]

## Cycle 200 [completed]
- Implement a first source pass that moves GF identity into data-only TOML and reduces template overhead only where the current source proves it unnecessary. [DONE]
  - Preserve the extensive dirty tree and inspect overlap before editing. `field_catalog.py` is currently unmodified; the flux factory/RHS and CMake owners contain active user changes that must be patched narrowly rather than replaced. [DONE]
  - Record the pre-change cache-hit production build at `0.52 s` with no compiler invocations. The focused three-rank WENO-Z/HLLC benchmark did finish and append its history row: the three-repeat median is `0.1799753057` Mcells/s, `0.9469` of the frozen baseline, with flux fraction `0.9932`. [DONE]
  - Move all 35 field declarations and 65 component entries, set membership, stable component IDs, centerings, roles, requirements, and output/checkpoint attributes to `schema/fields.toml`. `field_catalog.py` is now only the strict model/loader/validator and derived-view provider; it rejects unknown keys, wrong TOML types, invalid enum values, wrong tensor/component counts, duplicates, and unresolved references. The full generated-kernel comparison confirms byte-equivalent artifacts. [DONE]
  - Audit the active flux-template matrix before changing specialization. A Riemann/EoS/layout collapse would cross Newtonian, RHD, RMHD, mapped-patch, solid-wall, and directional launch contracts, so no unmeasured production-axis rewrite was retained. Remove four dead template helpers instead: the obsolete second reconstruction dispatch surface plus unused all-direction hydro/MHD wrappers, 236 lines with no call sites. Active object sizes remain exactly unchanged at `139770152` bytes across the 24 flux objects, as expected for dead declarations. [DONE]
  - Measure broad invalidation honestly. Rebuilding all 24 flux owners after the shared-header edit takes `265.49 s` with 24 cache misses; the immediately repeated production build takes `0.51 s` with no compiler invocations. This confirms both the roughly four-to-five-minute broad-header cost and the effective cached edit loop. [DONE]
  - Preserve runtime behavior. The matching post-change three-rank WENO-Z/HLLC median is `0.1823231666` Mcells/s, `1.30%` above the same-tree pre-change result and within ordinary run noise; flux fraction remains `0.9936`. No runtime speedup is claimed from deleting unreachable templates. [DONE]
  - Keep auxiliary cadence inheritance and catalog-derived output-layout cleanup as the already accepted independent backlog item. The field-schema migration does not require a behavior/default change, so it was not silently bundled into this first pass. [DONE]
  - Validate with `check_architecture.py`, the full generated-kernel comparison, the all-dimension production build, and the complete quick AMReX suite. All pass. Documentation whitespace checks also pass. [DONE]

## Cycle 201 [completed]
- Audit whether broad AthenaK comparisons or additional compile/layout/agent cleanup should precede renewed GR development. [DONE]
  - Treat broad code-to-code reruns as mostly redundant where PyFluid already has an analytic solution and convergence gate. AthenaK is most valuable at nonlinear or coupled boundaries without an exact evolved solution, and as a source of independent test design rather than a numerical oracle. [DONE]
  - Prioritize matched comparisons by the next GR feature: identical COCAL RNS/BNS initial-data sampling and derived-state materialization for rotating-star work; TOV oscillation/constraint histories for stellar evolution; Psi4 sign, normalization, and mode content for radiation; and puncture gauge/constraint/horizon histories only after puncture support exists. Require a uniform-grid equivalence declaration before comparing AMR implementations. [DONE]
  - Do not promote the existing `COCAL_ID_AK/compare_smoke` plots to rigorous validation. That script explicitly compares AthenaK mass against an unscaled ETK `rho.sum` and reports `drift_diff_unscaled_pct`; it is smoke/provenance evidence. A PyFluid comparison should instead use matched COCAL tables, units, coordinates, primitive/conserved definitions, proper volume measures, and pointwise plus global diagnostics. [DONE]
  - Defer another production flux-template rewrite. Cycle 200 confirms a `265.49 s` broad-header invalidation but a `0.51 s` cached rebuild, unchanged emitted flux-object size, and unchanged runtime. Use the existing full-method per-dimension presets or individual owner targets while iterating, then rebuild all dimensions at integration points; do not introduce a feature-stripped validation binary. [DONE]
  - Repair validation tiering before the next long GR cycle. The current `--level quick` run is no longer the documented sub-three-second smoke: `quick_feature_matrix` took `14.85 s`, `multipatch_cartesian` took `93.29 s`, and the curvilinear gates add tens of seconds. Preserve those gates in standard/full while restoring a genuinely tiny quick falsification path. [DONE]
  - Keep the accepted auxiliary cadence inheritance and catalog-derived output layout as the one worthwhile narrow layout cleanup before many more GR diagnostics/fields are added. Defer diagnostics/method TOML migration, generic registries, PCH/unity-build experiments, and other speculative compile machinery until a measured problem requires them. [DONE]
  - Reconcile the continuity queue before selecting the next GR implementation. The running log still contains numerous historical `in progress`/`pending` cycles; explicitly close, supersede, or rank the genuinely active GR items so a compacted agent cannot mistake old work for the current priority. [DONE]
  - This was a read-only source/artifact audit apart from this continuity entry. No production source, build, MPI launch, AthenaK run, or numerical comparison was changed or executed. [DONE]

## Cycle 202 [completed]
- Stabilize the accumulated AMReX/GR working tree before renewed GR development, then commit and capture a reproducible baseline. [DONE]
  - Preserve the current `main` working tree at `d78a0acc6a27`; it contains 124 tracked-file changes plus new AMReX, GR, diagnostic, validation, continuity, and reference-path files accumulated across Cycles 177--201. Audit and validate this state as one intended integration rather than discarding or silently splitting overlapping work. [DONE]
  - Restore a genuinely small quick validation tier while preserving expensive multipatch, production multipatch, curvilinear, and feature coverage in standard/full or focused tools. Quick now contains architecture plus two all-dimension feature probes and passes in `2.64 s`; the unchanged roughly 90-second multipatch executable remains full-only, while production multipatch, curvilinear, dimensional, analytic, and broader feature gates remain standard/full or focused. [DONE]
  - Implement ordinary GF output-cadence inheritance and one catalog-derived auxiliary output layout while preserving explicit write/compute overrides and cached-only operation. Parsing resolves omitted auxiliary output from `plot_int` and omitted compute from that result; explicit values retain their meaning. Each writer builds one private centering layout from generated descriptors. The inherited two-level `j,e` case writes the same two steps as state output with exact `j1,e1`, `j2,e2`, and `j3,e3` headers, and the explicit cached-only current-sheet case passes. [DONE]
  - Reconcile historical pending/WIP continuity entries into a compact current GR queue without deleting their evidence. Cycles 31, 37, 48, 85, 89, 146, 156, 158, 187, 189, and 190 are now explicitly completed or superseded. The active order is: Cycle 177 rotating equilibrium/RNS first; Cycle 186 generic spherical fluid/EM/magnetic fluxes and C2P health next; Cycle 185 honest extraction-radius convergence after a compact source exists; Cycle 184 curvilinear relativistic evolution as the largest later formulation project. [DONE]
  - Review AthenaK source and tests for concrete patterns PyFluid can adapt without duplicating analytic gates or importing incompatible ownership. Its flux owners explicitly instantiate Riemann solvers but select uniform reconstruction in the Kokkos kernel; its numerical-relativity task graph composes optional work once at startup; and its build selects one problem generator. These explain additive compile behavior and offer benchmark/design references, but a device branch, second scheduler, or feature-reduced production binary is not adopted without measurement. AthenaK's three current Z4c test modules are narrower than PyFluid's existing gate, so no duplicate analytic test was added. Knowledge, references, and backlog now route the exact source owners. [DONE]
  - Run focused output/restart checks, restored quick, broader integration gates, all-dimension compile, and runtime/compile benchmarks in proportion to the accumulated change set. The final all-dimension production build, architecture, generated physics, inherited-output case, explicit cached-only case, `2.64 s` quick gate, and all 42 standard gates pass. The standard gate includes production multipatch, radiation, dimensional/rank consistency, relativistic, curvilinear, AMR, analytic/reference, EoS, and diffusion coverage; its generated executable check is the dominant `129.87 s` stage. Python syntax compilation and `git diff --check` pass. The post-commit production cache-hit build is `0.57 s` with zero compiler invocations; this is not a cold-build measurement. [DONE]
  - Commit the validated intended tree as `e5e0574` (`Integrate AMReX GR infrastructure and validation`) and capture `runs/baselines/20260721_150242_pre_gr_stabilized` from that clean commit. The archived standard gate passes 42/42. The four three-/four-rank performance cases pass at `108.27`, `125.40`, `5.969`, and `0.1824` Mcells/s, or `0.985`, `0.915`, `0.937`, and `0.960` of the frozen clean baseline. No push was performed. [DONE]

## Cycle 203 [completed]
- Refine the post-TOV roadmap around tabulated EoS, rotating-star construction,
  existing diagnostics infrastructure, and genuinely deferred work. [DONE]
  - The initial proposal to import COCAL RNS data as a production provider was
    rejected by the user in Cycle 204. External stationary solutions remain
    baseline comparisons only; PyFluid will solve its own equilibrium. [DONE]
  - CTT, CTTK, and XCTS remain alternative choices of constraint variables and
    free data. Cycle 204 selects an in-house common rotating-fluid model with
    rotating XCTS as the first stationary construction rather than an imported
    full solution. [DONE]
  - Narrow Cycle 186: the catalog, generated GR integrands, composite volume
    reducer, spherical surface/interpolation service, and multipole/Fourier
    reducer already exist. Missing work is limited to mass-accretion, separated
    fluid/EM energy flux, magnetic flux, and explicit C2P/atmosphere algorithm
    counters. The generic geometry/reduction framework must not be duplicated. [DONE]
  - The Weyl/Psi4 algebra, harmonics, injected modes, stationary control, and a
    nonzero evolved wave are already validated. The remaining radiation check is
    retarded-time alignment and `r Psi4` convergence over several extraction
    radii for an isolated outgoing source; AthenaK's multi-radius output is a
    useful feature reference, not a different Weyl algorithm. [DONE]
  - Deprioritize curvilinear relativistic evolution. Retain the partial-AMR CTT
    result as a visible coarse-fine accuracy limitation, not an unfixed
    nonfinite/elliptic-solve bug: the matter handoff is repaired and the global
    solution converges, while a second-order multilevel solve and fourth-order
    pointwise constraint stencil leave a nonconvergent normalized interface
    band. Revisit only through a shared higher-order transfer/derivative or
    discrete-operator-consistent diagnostic, never a CTT-specific mask. [DONE]
  - Keep compilation policy empirical: preserve compile-time specialization
    only for required storage/topology or demonstrated runtime benefit; remove
    specialization that inflates broad rebuild/code size without measurable
    performance. Current cache-hit iteration is fast, so no speculative dispatch
    rewrite precedes physics work. [DONE]
  - Preferred sequence, superseded only in its former import mechanism by Cycle
    204: finish TOV edges; add a cold beta-equilibrated CompOSE table shared by
    initial data and evolution; construct and validate PyFluid's own uniform and
    differential rotating-star sequences; proceed to BNS, then BH/BBH. [DONE]

## Cycle 204 [completed]
- Make self-contained relativistic initial-data construction an explicit project
  boundary: external solutions are comparisons, not production inputs. [DONE]
  - PyFluid must solve its own TOV, rotating-star, BNS, BH, and BBH initial data
    as those capabilities are added. COCAL, RNS/FUKA, AthenaK, ETK, and similar
    codes may supply matched baselines and convergence comparisons only. [DONE]
  - Selected CompOSE/TEOS tables may be imported as physical EoS data, with
    provenance, unit conversion, range checks, interpolation validation, and
    checkpoint compatibility recorded. The same table implementation must serve
    initial data and evolution. [DONE]
  - Build one formulation-independent rotating-fluid layer: enthalpy/Euler first
    integral, surface condition, uniform rotation, KEH/j-constant differential
    rotation, and later additional laws. Couple it first to the existing XCTS
    nonlinear infrastructure so lapse, shift, conformal geometry, and matter are
    solved together. CTT and CTTK reuse the same matter/rotation definitions as
    explicitly different constraint formulations; they are not separate fluid
    implementations. [DONE]
  - Validate the zero-spin endpoint against the direct TOV solver at matched EoS
    and central density before continuing a sequence in angular velocity or
    angular momentum. Require pointwise matter/geometry agreement, global masses
    and radius, vanishing angular momentum/deformation, constraints, and short
    evolution stability. [DONE]
  - Use external RNS/COCAL solutions only after the internal zero-spin and
    self-convergence gates pass, then compare uniform-rotation sequences,
    mass-shedding behavior, and differential-rotation families without tuning
    PyFluid to reproduce one external discretization. [DONE]

## Cycle 205 [completed]
- Clarify whether CTT and CTTK can construct rotating-neutron-star data rather
  than implying that rotating equilibrium is exclusive to XCTS. [DONE]
  - CTT, CTTK, and XCTS can all solve the Einstein constraints with rotating
    matter and nonzero momentum/extrinsic-curvature sources. The formulation
    choice does not prohibit rotation. [DONE]
  - Distinguish a rotating constraint solution from a stationary fluid star.
    An RNS equilibrium additionally requires the relativistic Euler first
    integral, a rotation law and surface solve, and gravitational free data or
    time-vector conditions consistent with stationarity. Satisfying only the
    Hamiltonian and momentum constraints does not guarantee that the star will
    avoid an initial readjustment. [DONE]
  - A CTT backend can be coupled to the shared fluid-equilibrium iteration and
    can provide a conformally flat, commonly maximal-slicing rotating-star
    branch. It does not solve a lapse, so stationarity information not fixed by
    the constraints must be supplied consistently rather than inferred from a
    unit-lapse materialization. [DONE]
  - CTTK can likewise carry rotating matter, but PyFluid's current hybrid CTTK
    option algebraically selects spatially varying nonzero mean curvature from
    the matter source. That option intentionally does not reproduce the
    time-symmetric TOV solution at zero spin and therefore is not a stationary
    RNS construction by default. Label it equilibrium only after adding and
    validating stationarity-compatible free data; a maximal `K=0` endpoint
    would instead approach the CTT family. [DONE]
  - Keep XCTS first because its coupled conformal-factor, lapse, and shift solve
    makes quasiequilibrium conditions the most direct initial implementation,
    not because CTT or CTTK are incapable of rotation. Implement the Euler and
    rotation-law layer once, then expose formulation backends without separate
    fluid algorithms. [DONE]
  - This clarification changes durable documentation only. No solver source,
    compilation, MPI launch, or numerical validation was performed. [DONE]

## Cycle 206 [completed]
- Implement PyFluid's first self-contained rotating-neutron-star equilibrium
  path without importing an external solution or requiring a KEH solver. [DONE]
  - Preserve direct TOV as the spherical seed and exact zero-spin control. Add
    a data-only, human-readable rotation-law catalog and generate the C++ method
    identity/descriptor view from it rather than declaring method names in both
    Python and C++. `schema/methods.toml` now declares the uniform law;
    `method_catalog.py` strictly validates it and generation emits
    `generated/method_catalog.hpp`. [DONE]
  - Implement one shared relativistic Euler first-integral update for circular
    barotropic flow. Couple uniform rotation first to the existing outer
    matter/constraint iteration. The generated kernel evaluates
    `ln h = ln C - ln alpha - 1/2 ln(1-v^2)` and the hierarchy update forms
    `v^i=(Omega phi^i+beta^i)/alpha`, retains the requested central enthalpy,
    under-relaxes matter, rebuilds stress-energy, and rejects superluminal flow
    or matter at the physical boundary. [DONE]
  - Use XCTS for the first nonzero-spin stationary construction because its
    lapse and shift are solved consistently. Keep the rotation-law and matter
    update formulation-independent, but reject an equilibrium label for CTT or
    the current CTTK free-data branch until their additional stationarity
    closure is implemented. Configuration also rejects nonzero rotation outside
    the 3D executable, a cold polytropic/hybrid-polytropic EoS, or XCTS. [DONE]
  - Add strict configuration checks, generated-kernel/catalog consistency,
    zero-spin TOV equivalence, a small nonzero-spin solve, and proportional
    compile/runtime validation. Record any convergence or surface-closure
    limitation honestly. [DONE]
    - `UV_CACHE_DIR=.uv-cache uv run python
      pyfluid_amrex/tools/validate_codegen_flux.py` passed, including exact
      regeneration of the new catalog and Euler kernel. [DONE]
    - `UV_CACHE_DIR=.uv-cache uv run python
      pyfluid_amrex/tools/check_architecture.py` passed. [DONE]
    - The 3D `pyfluid_amrex_initial_data_elliptic_check_3d` target built and its
      one-rank MPI execution passed the generated uniform-rotation identity at
      `6.94e-17`, then completed the existing nodal/cell Poisson, CTT, CTTK,
      TOV-CTT, and TOV-XCTS controls. [DONE]
    - The final cache-warm integration build of `pyfluid_amrex_1d`, `_2d`,
      `_3d`, and the focused elliptic target completed successfully. Its tool
      wall time was about nine seconds; this is not a cold-compile benchmark.
      [DONE]
    - `benchmark_compile.py --target pyfluid_amrex_production --label
      rotating_star_catalog_cache_warm` recorded a fully up-to-date no-op in
      `0.47 s` with zero cache hits and zero misses. This measures build-system
      no-op overhead only, not a cold or cache-restored compile. [DONE]
    - The new opt-in `validate_z4c.py --only rotating_star --ranks 3 --force`
      gate passed two checks. The zero-spin `rotating_star` and `tov` paths had
      maximum normalized scalar difference exactly `0.0`. At `Omega=0.001`,
      the N=24 uniform-grid XCTS solve reached coupled update `7.6944e-6`,
      elliptic residual maximum `6.5725e-13`, `vmax=0.0248711`, axial matter
      angular momentum `0.0486005`, and zero boundary density excess. The
      retained PWL/HLL/SSPRK3 run then completed three evolved-spacetime steps
      at the production CFL `0.9` through `t=1.34284`: rest-mass drift was
      `8.5237e-10`, maximum-density change `1.0509e-2`, final axial angular
      momentum `0.0486915`, Hamiltonian L2 `1.9716e-4`, Hamiltonian Linf
      `4.5593e-3`, and momentum Linf `5.8298e-4`. The source artifact is
      `runs/current/z4c_validation/z4c_validation.tsv`. [DONE]
    - The first PWL/HLL evolution probe exposed nonrecoverable surface cells at
      step 3 even at CFL `0.2`, and at step 1 at CFL `0.9`. PWC/HLL completed
      three steps at CFL `0.9`, isolating the defect to high-order surface
      admissibility rather than timestep selection. The generic RHD
      invariant-domain flux limiter had allowed high-order candidates only a
      roundoff-sized margin under the input stage metric; the coupled RK stage
      then accepted them under the evolved metric. It now retains ten percent
      of the safe low-order LLF candidate's margin. The exact PWL/CFL-0.9 case
      then completed without nonfinite cells. [DONE]
    - Python already defined the appropriate `dynamic_spacetime` scalar policy,
      but the C++ accepted-name list omitted it. The list now accepts the same
      policy name, so evolved spacetime checks enforce rest mass and physical
      states without incorrectly demanding conservation of fluid energy alone.
      [DONE]
    - Both existing `rhd_kh_amr_no_subcycling` and
      `rhd_kh_amr_subcycling` feature cases pass after the generic limiter
      change. `validate_amrex_suite.py --level quick` again passed architecture
      and the quick feature matrix in `2.47 s`. The subsequent standard suite
      passed all 42 gates, including RHD/RMHD matrices, rank reproducibility,
      code generation, dimensional/curvilinear, AMR, multipatch, analytic, and
      reference controls. No full/stability suite or broad runtime-performance
      benchmark was run for this physics addition. [DONE]
    - Exploratory N=16 solves failed in the nodal Robin Schur solve after the
      first matter update, and a fixed injection-energy iteration depleted the
      star. The retained N=24 path instead updates injection energy to preserve
      central enthalpy. The three-step evolution is a short finite-state control,
      not long-term stability, a self-convergent spin sequence, deformation,
      mass shedding, differential rotation, or AMR rotating equilibrium. [DONE]
    - The central-enthalpy closure currently identifies the stellar center with
      the minimum helical effective potential. This is a deliberate small-spin
      spheroidal assumption; high-spin, toroidal, off-center, and mass-shedding
      families need explicit topology/surface controls before validation.
      [DONE]
  - Audit the requested fourth-order general elliptic solve. AMReX
    `MLLinOp::setMaxOrder(4)` explicitly controls coarse/fine interpolation,
    while `MLABecLap_3D_K.H::mlabeclap_adotx` remains the nearest-neighbor
    second-order interior stencil. Rename the internal option to
    `boundary_interpolation_order` and document the distinction. [DONE]
    - A true fourth-order general solver was not faked with an interior-only
      stencil. It requires matched scalar, nodal, and conformal-vector
      operators, two ghost layers, fourth-order Dirichlet/Robin closures, and
      AMR restriction/prolongation/coarse-fine consistency. A staged
      fourth-order defect-correction operator using MLMG as the second-order
      preconditioner remains a plausible next numerical project after a
      uniform-grid manufactured-solution prototype. [DONE]
  - Keep tabulated CompOSE/TEOS support as the next physics stage; do not mix a
    new table interpolation/inversion contract into the first rotating
    polytrope implementation. No TEOS source was changed in this cycle. [DONE]
  - Create one local checkpoint commit containing the catalog, rotating-star,
    limiter, validation, and continuity changes after the final gates. Nothing
    was pushed. [DONE]

## Cycle 207 [completed]
- Re-audit the rotating-star RHD surface failure and the Cycle 206 limiter
  change against component MP5 and established production GRMHD practice. [WIP]
  - Treat the fixed `0.1` retained-low-order-margin fraction as an unproven
    heuristic. It has been removed; no arbitrary retained-margin fraction is
    present in the current limiter. [DONE]
  - Run the same small-spin XCTS evolution with component primitive MP5 at CFL
    `0.9`, then determine whether the failure is reconstruction-specific or a
    coupled-stage admissibility defect. [DONE]
    - The original MP5 probe failed on the first step with 2168 nonfinite cells.
      A focused reconstructed-flux check localized the first failure to a face
      where the cell-centered velocity was causal under its own metric but had
      `v^2 approximately 1.0187` when reused unchanged with the face metric.
      RHD now reconstructs all three components of `z^i = W v^i` with the
      selected component method and converts `z^i` to `v^i` under the face
      metric. Reconstructed MP5 fluxes are finite after this repair. [DONE]
    - Exact stage algebra exposed a second issue: the admissibility check had
      tested a forward-Euler-like candidate under the input metric, although
      the coupled fluid and Z4c update is recovered under the projected metric
      of the actual RK candidate. The limiter now receives the concrete base,
      input, and increment coefficients for Euler, corrector, midpoint,
      SSPRK3, and RK4, and evaluates the candidate in its matching projected
      cell metric. [DONE]
    - With binary first-order correction and minimal energy-floor projection,
      the PWL/HLL/SSPRK3 checkpoint control completed three steps at CFL `0.9`.
      Scalar validation passed with rest-mass drift `8.523694827e-10`, fluid
      energy drift `5.35814591e-5`, final time `1.342840673`, and about
      `0.05442 Mcells/s`. This is a short evolved-spacetime control, not a
      long-term rotating-star result. [DONE]
    - The matching component-MP5 run also completed all three CFL-0.9 steps
      without a nonfinite state, but scalar validation correctly failed:
      rest-mass drift was `9.345033624e-8` against the existing `1e-8` gate.
      Temporary accounting localized the dominant first-stage density change
      to atmosphere projection (`3.8386e-7` in the raw densitized sum), so the
      remaining problem is surface density-floor injection, not a reason to
      lower CFL or globally replace MP5. The input remains PWL and the mass gate
      remains unchanged. [DONE]
  - Prefer a stage-consistent physical-state test and principled first-order
    flux correction, or a mathematically justified positivity limiter, over an
    arbitrary safety fraction. Record the current XCTS/CTT/CTTK and rotation-law
    validation boundary without overstating a three-step small-spin smoke. [WIP]
    - Local AsterX confirms MP5 plus explicit safeguards: it supports `z_vec`
      reconstruction and substitutes a configured lower-order method only for
      the invalid reconstructed face side. Local AthenaK's dynamical-GRMHD FOFC
      predicts the actual RK candidate, asks C2P whether floors, a velocity
      ceiling, or failure would occur, replaces only adjoining faces with
      first-order LLF/HLLE, and records counters. The current PyFluid direction
      follows this localized structure; it does not introduce a global MP5
      fallback. [DONE]
    - The current RHD implementation uses binary adjacent-face LLF correction.
      Accepted-stage atmosphere handling preserves densitized density and
      momentum when only an energy admissibility floor is needed, raising
      `tau` minimally before retrying C2P. Density-floor replacement remains
      explicit and is the measured source of the MP5 mass-gate failure. [DONE]
    - Current integration evidence: `pyfluid_amrex_production` built in all
      dimensions; `validate_amrex_suite.py --level quick` passed architecture
      and the quick feature matrix in `3.03584 s`; `git diff --check` passed.
      The focused three-rank `rhd_kh_amr_no_subcycling` case passed, and the
      matching subcycling invocation passed both its no-subcycling dependency
      and `rhd_kh_amr_subcycling`. The first sandboxed MPI attempt failed only
      because local sockets were unavailable; both socket-capable reruns passed.
      These are cache-warm/incremental checks, not a cold compile benchmark.
      No standard/full/stability suite or runtime performance benchmark has
      been run for the current WIP. [DONE]
    - Cycle 208 extended the same `Wv` contract to RMHD and added compact
      per-step counts for secondary RMHD C2P use and terminal atmosphere
      resets. The requested KISS policy explicitly excludes a repair
      mass/energy ledger. Reusable candidate-metric scratch, MP5 stellar-surface
      treatment, and coupled RMHD candidate FOFC remain possible future work,
      not production requirements established by this cycle. [DONE]
    - XCTS remains the first stationary backend because it solves lapse and
      shift alongside the constraints and shared Euler first integral. CTT can
      support the same rotating matter after adding a consistent lapse/slicing
      stationarity closure. The current deliberately nonzero-`K` CTTK free-data
      branch is not the TOV zero-spin endpoint; it needs a maximal/stationary
      branch before it is called RNS equilibrium. Uniform rotation is the only
      implemented law. Zero spin is an exact TOV-path identity; `Omega=0.001`
      initialization converges, but no spin sequence, resolution convergence,
      deformation, mass shedding, differential rotation, or long evolution is
      yet validated. [DONE]
## Cycle 208 — Relativistic reconstruction and recovery consistency (completed)

- Apply the established spatial-four-velocity reconstruction contract
  consistently to production GRHD and GRMHD for every selected component
  reconstruction method, without changing Newtonian HD/MHD behavior. RMHD now
  reconstructs `z^i = W v^i` under each cell metric and maps it through the
  face metric exactly as RHD does. [DONE]
- Curved reflecting boundaries initially exposed a real regression:
  `schwarzschild_rmhd_1d` mass drift rose to `4.5483e-7`. The external mass
  ledger residual remained at roundoff, localizing the change to wall flux.
  Reflecting ghosts had still negated coordinate `v^i`; unequal source/ghost
  metrics therefore made reconstructed `Wv` non-antisymmetric. The boundary
  now reflects `Wv` and maps it through the ghost metric. The mass drift returns
  to `3.2256e-16`, and the complete fixed-spacetime RHD/RMHD suite passes.
  [DONE]
- Keep recovery observability deliberately small. Evolution counts successful
  secondary RMHD C2P use and terminal atmosphere resets after C2P exhaustion,
  reduces counts across ranks, warns once per affected step, and prints totals
  at completion. Ordinary atmosphere cells are not failures. GPU builds
  accumulate into persistent device counters and synchronize once per step.
  No repair mass/energy ledger or new diagnostic schema was added. [DONE]
- `validate_rmhd.py` now runs a compact smooth-wave sweep over PWC, PWL,
  WENO-Z, MP5, and TENO-5 and requires zero fallback/reset events. The complete
  three-rank RMHD suite passes: entropy-wave order is `2.0642` at N=256,
  Balsara normal-field spread is zero, CT reference improvement is `1.6369`,
  AMR improvement is `2.8441`, AMR+CT divB L1 is `1.6069e-15`, and restart is
  identical within `1e-12`. [DONE]
- The focused RHD KH single-rate/subcycling pair passes. Exact zero-spin/TOV
  scalar equivalence remains zero, and the N=24 `Omega=0.001` XCTS control
  passes three PWL/HLL/SSPRK3 steps at CFL `0.9` with rest-mass drift
  `8.5237e-10` and zero recovery events. [DONE]
- Architecture, generated-kernel consistency, all production dimensions, the
  `2.53 s` quick gate, and every standard gate pass. Canonical performance
  cases pass at `0.99`, `0.92`, `0.93`, and `0.96` of their frozen baselines.
  The first standard invocation failed before executing cases because sandbox
  MPI sockets were unavailable; the identical socket-capable rerun passed.
  A fresh 42/42 standard plus performance run is archived at
  `runs/baselines/20260721_201750_relativistic_reconstruction_recovery/`.
  No full or stability suite was run. [DONE]
- The CT stage layout was inspected but not changed. AthenaK advances face
  fields with explicit RK stage coefficients; PyFluid's corresponding concern
  needs a dedicated temporal-order reproducer before it is treated as a defect
  or used to justify a redesign. RMHD stage-candidate FOFC remains separate
  future work rather than an unvalidated change in this pass. [DONE]

## Cycle 209 — Uniform-rotation charge and range probes (completed)

- Rechecked the current nonzero-spin XCTS boundary from source. The uniform
  Euler/matter update is shared by CTT, CTTK, and XCTS materialization, but the
  configuration guard admits nonzero spin only for XCTS because the present
  CTT/CTTK branches do not close a stationary lapse/shift/slicing system.
  No formulation restriction or source code was changed. [DONE]
- Ran ID-only three-rank `Omega=0.001` models with global GR diagnostics.
  At N=24, matter `J3=0.0486005` and volume ADM `J3=0.0485737` agree to
  `5.53e-4`, giving volume `J/M^2=0.02005`. Surface ADM `J3=0.0285794` is
  41.2 percent below the volume charge, while surface and volume ADM masses
  differ by 2.37 percent. [DONE]
- The N=32 repeat with 48 polar quadrature points failed the unchanged
  `1e-5` matter-update tolerance at the checked-in 24-iteration cap
  (`3.83e-5`). Raising only the cap to 40 converged in 30 iterations at
  `9.93e-6`. Matter/volume ADM angular-momentum disagreement improved to
  `1.67e-4`, but surface/volume disagreement remained 40.1 percent. Surface
  spin is therefore not accepted as validated; radius/domain/resolution tests
  are required before attributing the discrepancy to one implementation.
  [DONE]
- Higher-spin N=24 ID probes retained relaxation `0.2` and tolerance `1e-5`:
  `Omega=0.010` converged in 43 iterations with `M_ADM=1.58685`,
  `J_ADM=0.516237`, `J/M^2=0.2050`, maximum velocity `0.24938`, and
  matter/volume-ADM `J` disagreement `5.35e-4`; `Omega=0.015` converged in 45
  iterations with `M_ADM=1.62996`, `J_ADM=0.838370`, `J/M^2=0.3156`, maximum
  velocity `0.37550`, and disagreement `5.14e-4`. [DONE]
- `Omega=0.0175` and `0.020` abort on the first rotating-matter update because
  matter reaches the physical boundary. This brackets the current
  fixed-domain/central-enthalpy implementation limit between `0.015` and
  `0.0175`; it is not a measured Kepler or mass-shedding limit. The successful
  runs are under `runs/current/z4c_validation/rotating_star_spincheck_*`.
  [DONE]
- Stronger production acceptance should use a continued uniform sequence with
  at least two resolutions, independent matter/ADM volume/ADM surface charges,
  `J/M_ADM^2`, ADM--Komar and surface--volume agreement, constraint convergence,
  baryon-mass control, equatorial/polar radii, and an equatorial mass-shedding
  indicator. Uryu differential-rotation laws are a subsequent extension of the
  same generated Euler-integral interface, after the uniform sequence and
  charge boundary are sound. [DONE]

## Cycle 210 — XCTS spin charge and connected-surface repair (completed)

- Diagnosed the roughly 40 percent ADM-surface angular-momentum deficit before
  changing quadrature. Moving the N=24 extraction radius from 10 to 12.8 left
  surface `J3` essentially unchanged (`0.0285362` versus `0.0285794`) while
  volume `J3=0.0485737`; this ruled out extraction radius as the dominant
  error. A trial componentwise `r^-2` Robin shift boundary made the nodal
  conformal-vector solve fail on the second matter iteration and was fully
  reverted. [DONE]
- Corrected the generated XCTS shift matter source. The evolved Eulerian
  momentum projection is covariant `S_i`; with conformal flatness, raising the
  index changes `16 pi alpha psi^4 S^i` to
  `16 pi (alpha psi) S_i / psi`. The old symbolic source instead used
  `16 pi (alpha psi)^2 S_i / psi^4`, suppressing frame dragging. At N=24 and
  `Omega=0.001`, the corrected ID has matter `J3=0.0442704`, volume ADM
  `J3=0.0442470`, and surface ADM `J3=0.0438890`: matter/volume differ by
  `5.3e-4` and surface/volume by `8.1e-3`, replacing the old 41 percent
  discrepancy. [DONE]
- Repaired the higher-spin matter topology. The central enthalpy constraint now
  interpolates the helical effective potential at the configured rotation
  center rather than using a global domain minimum. A one-ghost-cell matter
  mask admits target matter only adjacent to the preceding central component,
  allowing the stellar surface to move while excluding disconnected
  positive-enthalpy lobes near finite-domain corners. Timelike and maximum-
  velocity diagnostics now inspect that continued matter support rather than
  irrelevant vacuum corners. A preliminary expanding-radius envelope was
  tested, produced repeated coarse shell activations, and was replaced rather
  than retained. [DONE]
- The formerly immediate-failing N=24 `Omega=0.0175` model converged in 45
  matter iterations with matter/volume/surface `J3` equal to
  `0.923138/0.922702/0.915539` and matter `vmax=0.19619`. The N=24
  `Omega=0.020` model has one coarse surface-cell activation at iteration 42,
  then decays with the configured 0.2 relaxation and converges in 77 iterations
  at `8.744e-6`. Its volume `M_ADM=1.695691`, matter/volume/surface `J3` are
  `1.123234/1.122723/1.114193`, and volume `J/M^2=0.3905`. This demonstrates
  operation at that point only; it is not a resolution-converged sequence or a
  Kepler/mass-shedding limit. [DONE]
- Raised the rotating-matter maximum-iteration default and checked-in XCTS
  example ceiling from 32/24 to 96. This ceiling does not force extra work:
  the small-spin gate still converges in 24 iterations. The focused validator
  now requests global GR diagnostics, requires matter/volume `J` agreement
  within 1 percent and surface/volume agreement within 3 percent, checks
  `0 < J_ADM/M_ADM^2 < 1`, and records the charges and spin. The three-rank
  `validate_z4c.py --only rotating_star --force` gate passes both reported
  checks; its small-spin evolution retains rest-mass drift `8.52e-10` at CFL
  `0.9`. [DONE]

## Cycle 211 — Rotation continuation, ID scaling, and saved-ID boundary (completed)

- Added an explicit angular-velocity continuation count to rotating-star
  initial data. Each stage is a fully converged matter/XCTS solve and retains
  the preceding fields as its seed; the default remains one direct stage.
  `Omega=0.025` converged with three stages after the direct solve failed.
  `Omega=0.02625` converged with five stages and 207 total matter iterations,
  yielding volume `M_ADM=1.864969`, matter/volume/surface `J3` of
  `1.920608/1.919866/1.906503`, and `J/M_ADM^2=0.5520`. The independent
  angular-momentum comparisons are within 0.039 and 0.696 percent. [DONE]
- The N=24 `Omega=0.02625` result remains a numerical reach probe, not a
  Kepler limit: volume ADM--Komar mass differs by about 4.9 percent and no
  equatorial mass-shedding indicator or resolution sequence exists yet.
  Direct `Omega=0.025` and larger jumps failed, while staged continuation
  separated nonlinear-basin failure from physical-limit evidence. [DONE]
- The existing Read-style SLy piecewise-polytrope path is compatible with the
  rotating solver. N=24 `Omega=0.001` converged directly in 23 iterations;
  matter/volume and surface/volume `J3` differed by 0.076 and 1.17 percent.
  N=24 `Omega=0.025` converged through three stages in 31/34/35 iterations,
  with volume `M_ADM=1.533368`, `J3=1.022967`, `J/M^2=0.4351`, and the two
  charge discrepancies 0.065 and 1.12 percent. Its ADM--Komar gap remains 4.53
  percent. The lower iteration count than the Gamma=2 model is not an EoS
  algorithm comparison because the stellar models differ. [DONE]
- Added coordinate equatorial/polar matter extents and their ratio to the
  rotating-star summary and focused gate. They are compact deformation sanity
  checks, not a mass-shedding proxy. The physical equatorial enthalpy-gradient
  or `Omega/Omega_K` diagnostic remains open. The focused N=24 small-spin gate
  records coordinate radii `7.60117/7.33333`, ratio `1.03652`, and passes.
  [DONE]
- Added a controlled 1/2/3-rank initial-data benchmark that records wall time,
  seconds per matter iteration, speedup, and parallel efficiency for an
  identical N=24 small-spin solve with 8-cell boxes. One, two, and three ranks
  took `399.686`, `221.863`, and `159.385` seconds for the same 24 matter
  iterations. Speedups are `1.802` and `2.508`, giving 90.1 and 83.6 percent
  efficiency. Final updates and elliptic residuals agree across ranks. Results
  are in `runs/current/initial_data_scaling/results.tsv`. [DONE]
- Rejected the draft step-zero-checkpoint wrapper as the stable saved-ID
  architecture. A reusable ID must contain provider-independent physical
  fields before evolution conversion, with its own hierarchy, EoS, physical
  parameters, diagnostics, and provenance. Loading is one target-hierarchy
  materialization: matching cells copy exactly; changed grids use field-aware
  interpolation/asymptotic exterior seeding followed by a target-grid
  constraint solve. This artifact boundary is now owned by
  `docs/infrastructure_conventions.md`; the current checkpoint is only a
  temporary exact-evolution cache. [DONE]
- Implement the provider-independent physical-field writer and target-grid
  materializer described by that contract. No checkpoint alias or partial
  writer is retained in this cycle. [PENDING]
- Reorganized continuity ownership explicitly: `knowledge.md` contains learned
  physics/numerical/AMReX subtleties and current evidence boundaries;
  `references.md` routes to local and external sources and explains what they
  contain/how to access them; infrastructure and artifact conventions live in
  `docs/infrastructure_conventions.md`; dated evidence stays in this cycle.
  [DONE]
- Integration evidence for this tree: architecture and generated-kernel checks
  pass, `pyfluid_amrex_production` builds in all dimensions, the focused
  three-rank rotating-star gate passes both checks, and the complete quick gate
  passes with the feature matrix in `2.97467 s`. Python syntax and
  `git diff --check` pass. These are incremental builds and focused/quick
  evidence, not a cold compile, standard/full/stability suite, or broad runtime
  benchmark. [DONE]

## Cycle 212 — Uniform mass shedding, elliptic efficiency, and q-law rotation (completed)

- Use the existing SLy piecewise-polytrope and fixed-central-enthalpy closure
  for the first uniform-rotation sequence. Push angular-velocity continuation
  beyond the current coarse N=24 reach, but identify a physical endpoint only
  from a source-verified equatorial mass-shedding diagnostic plus resolution
  evidence, never from nonlinear failure alone. The implemented signed
  Gourgoulhon-style ratio averages interpolated outward `d ln(h)/dr` over two
  polar and four equatorial rays; spherical stars approach one and a cusp
  approaches zero. The first absolute-value version was corrected because it
  could hide a post-cusp sign reversal. [DONE]
- The N=24 fixed-central-density SLy sequence records: `Omega=0.001`,
  23 matter iterations and `chi_ms=0.99257`; `0.025`, 100 iterations and
  magnitude `0.28881`; `0.026`, 107 and magnitude `0.27937`; directly signed
  `0.028`, 158 and `0.25972`; directly signed `0.029`, 177 and `0.32299`;
  and `0.030`, 163 and pre-sign-correction magnitude `0.36020`. Matter/volume
  angular momentum differs by at most `7.6e-4` in these recorded rows and
  surface/volume by about 1.1--1.2 percent. The signed N=24 indicator is already
  non-monotone between `0.028` and `0.029`; no mass-shedding frequency is
  claimed, and more coarse bisection is rejected in favor of resolution
  evidence. [DONE]
- Profile the nested matter/XCTS solve and remove measured elliptic setup,
  allocation, or repeated-operator overhead without weakening residual or
  physical gates. Hoisted scratch allocations plus reuse of one nodal MLMG
  operator changed the identical three-rank N=24 solve from `159.385` to
  `159.621` seconds while preserving 24 matter iterations, 15 final coupled
  iterations, `7.590684236e-6` update, and residuals, but changed the
  surface-gradient diagnostic from about `0.99970` to `0.98548`. The entire
  experiment was reverted because it bought no measured speed and was not
  physically identical. [DONE]
- Add only the requested q-power/j-constant differential-rotation families to
  `schema/methods.toml`, with generated first integrals and focused zero-law or
  limiting controls. Do not add the broader Uryu post-merger profiles in this
  cycle. The relativistic law solves `u^t u_phi = j(Omega)` pointwise; q-power
  `q=1` equals j-constant to roundoff. Positive/negative and negative
  noninteger-q unit residuals are below `1e-15`. N=24 SLy `Omega=0.001`,
  `A=5` full-ID runs pass for j-constant and q-power `q=2` in 23 matter
  iterations. Their matter/volume J errors are `9.56e-4` and `8.58e-4`, and
  surface/volume errors are 1.37 and 1.27 percent. N=16 uniform and j-constant
  SLy controls fail identically after the first constraint solve, establishing
  that grid as below the usable resolution rather than a law-specific defect.
  [DONE]
- Fixed an independent diagnostic dependency exposed by the differential
  probe. `global_gr`-only volume reductions used vertex matter that was
  refreshed only when the `constraints` group was also selected; stale data
  produced volume mass near 3017 while the surface mass was 1.397. Volume GR
  diagnostics now explicitly refresh matter. The identical q-power rerun with
  only coordinate angular momentum plus `global_gr` gives surface/volume ADM
  masses `1.39737/1.36616` and matter/volume/surface J
  `0.0198317/0.0198147/0.0195623`. [DONE]
- Run focused rotating-star, generated/architecture, all-dimension build,
  quick integration, and initial-data performance checks; update the compact
  knowledge boundary and commit locally only after the evidence is recorded.
  The focused three-rank elliptic/TOV/CTT/CTTK/XCTS executable passes after the
  final negative-spin hardening. Generated consistency and architecture pass;
  all production dimensions build; the focused three-rank rotating gate passes
  exact zero-spin TOV equivalence plus the small-spin control with signed
  `chi_ms=0.999696`, CFL `0.9` rest-mass drift `8.52e-10`, and both charge
  comparisons below one percent. The complete quick gate passes in `3.11 s`,
  and all 42 standard gates pass. The four canonical runtime benchmarks pass
  at `0.92--0.99` of frozen baselines. [DONE]
- The central-config/header integration rebuild took `156.57 s` with two cache
  hits and 103 misses; its immediate no-op rebuild took `0.61 s`. This is a
  dirty-tree broad-invalidation/cache-miss measurement, not a clean cold build.
  It quantifies the remaining central-config fanout without justifying a config
  redesign inside this physics change. No full or stability suite was run.
  [DONE]
## Cycle 213 — EOS-aware entropy output and resolved rotation-limit continuation (DONE)

- Replace the misleading single-gamma `entropy` auxiliary output with one
  EOS-aware thermal entropy proxy. Ideal gas uses total pressure and its
  adiabatic index; hybrid piecewise-polytropic matter uses
  `p_th = p - p_cold(rho)` and `Gamma_th`; a strictly cold barotrope has no
  independent thermal entropy and returns zero. Keep the GF identity and
  documentation honest and add focused formula/output coverage. The field is
  now cataloged only as `thermal_entropy_proxy`; generated descriptors and
  equations were regenerated. The focused ideal-gas HDF5 run writes one
  component with all 512 values exactly equal to
  `3/2^1.4 = 1.136787424882799`. Ideal, cold-polytrope, and hybrid-PWP formula
  checks compile and emit no mismatch in the generated check executable.
  [DONE]
- Continue the fixed-central-density SLy uniform-rotation sequence at higher
  resolution. Use the signed surface-enthalpy-gradient indicator plus
  independent matter/ADM volume/ADM surface charges; do not infer a physical
  endpoint from nonlinear failure or another non-monotone coarse-grid sample.
  Run expensive points serially and first confirm that no old solver jobs are
  active. No old solver job was present. Initial N=32 attempts exposed
  decomposition-dependent garbage at the rotation center: the temporary
  effective-potential and surface-`ln h` fields had zero ghosts while linear
  interpolation crossed internal FAB boundaries. Both now allocate and
  exchange one ghost layer. `HierarchyInterpolationPlan` collectively rejects
  a crossing stencil without sufficient storage, and the focused three-rank
  regression checks both exact interpolation at an eight-FAB junction and
  rejection of the zero-ghost control. [DONE]
- The corrected N=32 SLy sequence passes at `Omega=0.001`, `0.028`, and
  `0.032`. The `0.028` model has `chi_ms=0.404505` and
  `J_ADM/M_ADM^2=0.504573`. The six-stage `0.032` solve takes `2831.43 s`,
  converges in 219 total matter iterations to relative update `8.91e-6`, and
  gives `chi_ms=0.230266`, coordinate axis ratio `1.83922`, and
  `J_ADM/M_ADM^2=0.594246`. Its Hamiltonian/lapse/momentum residuals are
  `6.02e-15`, `1.12e-14`, and `5.72e-13`; matter/volume J differs by
  `1.75e-4` and surface/volume by `5.77e-3`. This is a strongly flattened,
  still-subcritical solution, not a bracketed or resolution-converged
  mass-shedding endpoint. Old N=24 surface-indicator values are superseded;
  their independent charge comparisons remain usable. [DONE]
- Run the smallest relevant generated/catalog, focused, build, and quick gates;
  update `knowledge.md` with only durable numerical findings and commit the
  completed state if the evidence is sound. Generated consistency,
  architecture, Python syntax, the focused three-rank surface and
  elliptic/TOV/CTT/CTTK/XCTS executables, the actual auxiliary-HDF5 run, and
  the two-check rotating-star gate pass. The rotating gate retains CFL `0.9`,
  exact zero-spin equivalence, small-spin `chi_ms=0.999696`, dimensionless spin
  `0.0182646`, and rest-mass drift `8.52e-10`. All production dimensions build;
  the complete quick gate passes in `3.02 s`, all 42 standard gates pass, and
  the four canonical runtime benchmarks are `0.92--0.99` of their frozen
  baselines. The production integration rebuild takes `375.94 s` with one
  cache hit and 156 misses; its no-op rebuild takes `0.61 s`. The generated
  compile-check executable still exits on the pre-existing high-order
  centering-transfer mismatch after the new EoS checks, so it is not recorded
  as a whole-executable pass. No full or stability suite was run. [DONE]

## Cycle 214 — Upper uniform-rotation branch and infrastructure stress (DONE)

- Continue the corrected N=32 fixed-central-density SLy uniform sequence above
  `Omega=0.032`, serially and with no competing solver jobs. Start at
  `Omega=0.034` with seven fully converged continuation stages. Accept a model
  only if the signed mass-shedding indicator remains positive, the flow is
  timelike and subextremal, constraints converge, matter/volume/surface charges
  agree, and the stellar surface remains inside the physical domain. Solver
  failure alone is not a mass-shedding endpoint. The seven-stage
  `Omega=0.034` run reached the final continuation stage after steps 1--6
  converged. Its final iterate diverged: maximum velocity rose from `0.410` to
  `0.761`, relative matter updates remained `0.05--0.12`, and the nodal Robin
  Schur residual reached `4.00e38` versus tolerance `5.26e34`. The mandatory
  residual guard aborted all ranks; boundary density excess remained zero.
  This is a cleanly rejected nonlinear/elliptic failure, not a mass-shedding
  measurement. The narrowed seven-stage `Omega=0.033` point passes after
  `3609.84 s` and 261 total matter iterations with final update `8.68e-6`,
  signed `chi_ms=0.0817565`, `J_ADM/M_ADM^2=0.619983`, and coordinate axis
  ratio `1.96889`. Matter/volume J differs by `1.70e-4`, surface/volume by
  `5.73e-3`, the maximum residual is `7.20e-13`, and boundary density excess
  is zero. The eight-stage `Omega=0.0335` run converged through stage 7, then
  reproduced the `0.034` final-stage failure: maximum velocity rose from
  `0.404` to `0.718`, relative updates stayed `0.05--0.12`, and the Robin
  Schur residual reached `3.41e83`; boundary excess remained zero. The
  eight-stage `Omega=0.03325` bisection likewise converged through stage 7,
  then ran away at the final target: maximum velocity rose from `0.400` to
  `0.812`, the relative update reached `0.180`, and the Robin Schur residual
  reached `2.55e53` against tolerance `1.29e47`. Boundary excess remained
  zero. The current nonlinear basin is therefore bracketed by the valid
  `0.033` point and failed `0.03325` attempt; it is not a physical
  mass-shedding bracket. [DONE]
- Audit the ADM--Komar discrepancy rather than changing the formulas. The
  implemented Komar volume and surface expressions include the lapse, shift,
  stress, momentum, and extrinsic-curvature terms. At fixed `dx=1`, expanding
  the zero-spin domain changes the extraction radius from 12.8 to 19.2: the
  Komar-surface/ADM-surface difference falls from 4.39 to 2.94 percent while
  the Komar surface mass changes by only 0.009 percent. A leading `1/R`
  extrapolation gives ADM surface mass `1.20627`, consistent with the Komar
  value near `1.207`. The discrepancy is finite-radius ADM contamination.
  The control also exposed that scalar plotting assumed standard mass/energy
  columns; it now renders intentionally sparse diagnostic selections with
  explicit `not selected` panels and passes a global-GR-only TSV probe. [DONE]

## Cycle 215 — Self-consistent-field XCTS acceleration (DONE)

- Check established rotating-star solvers before changing the nonlinear
  method. Local COCAL explicitly reuses each converged 3D solution as the next
  sequence seed. XNS iterates one hierarchical set of Poisson-like field
  solves with sources from the preceding field/matter iterate, and LORENE
  documents staged rotation with separate main/Poisson relaxation. PyFluid
  already uses AMReX MLMG for every spatial elliptic subsolve; the measured
  waste was nesting a fully converged frozen-matter XCTS fixed point inside
  every matter update. [DONE]
- Replace only that nonlinear nesting with a conservative inexact
  self-consistent-field schedule. Accurate XCTS solves settle dimensionless
  updates above `sqrt(matter_tolerance)` and, once within
  `10*matter_tolerance`, permanently polish the endpoint. The intermediate
  regime uses two ordinary relaxed XCTS sweeps. Acceptance still requires the
  original matter tolerance, a converged strict XCTS result, finite/timelike
  matter, interior stellar support, and the existing elliptic residual guards.
  Aggressive undamped and late-polish prototypes were rejected after they
  failed the known N=32 upper model; their speedups are not accepted results.
  [DONE]
- The final policy reduces the identical N=24, 8-cell-box, three-rank
  `Omega=0.001` benchmark from `159.62 s` to `66.81 s` (2.39x), with 22 matter
  updates, a strict 14-iteration final XCTS solve, and residuals below
  `9.34e-13`. The decisive seven-stage N=32 `Omega=0.033` comparison passes in
  `2267.96 s` instead of `3609.84 s` (1.59x). It gives
  `chi_ms=0.081731`, `J_ADM/M_ADM^2=0.619966`, the identical `8.86002/4.5`
  coordinate radii, matter/volume J difference `1.70e-4`, surface/volume
  difference `5.73e-3`, and residuals below `7.90e-13`. These agree with the
  old strict-nested fixed point within the configured matter tolerance. [DONE]
- Architecture, generated consistency, Python syntax, sparse scalar plotting,
  all production dimensions, the final two-check three-rank rotating gate,
  the complete quick gate (`3.27 s`), and the default standard suite pass. The
  focused gate retains exact zero-spin equivalence, CFL `0.9`, and zero
  recovery events. The final individual materialization-object rebuild took
  about `4.9 s`; the cached integration relink took `0.94 s`. No full or stability
  suite was run. Reusable physical-ID save/load remains the next established
  sequence optimization; the current step-zero checkpoint is not that
  artifact. [DONE]

## Cycle 216 — Reusable rotating-ID sequences and symmetry boundary (WIP)

- Generalize the existing in-process angular-velocity continuation into an
  explicit ordered target list. A normal initial-data solve remains the
  one-target case; accepted targets retain their live matter and XCTS fields
  as the seed for the next target without checkpoint I/O or reconstruction.
  Preserve the existing scalar target parameters, equations, tolerances, and
  final acceptance gates. The explicit `0.0005, 0.001` N=24 control converged
  in 41 total matter iterations with final residuals below `7.15e-13`; omitting
  the list preserves the existing equal-fraction behavior exactly. [DONE]
- Add one versioned, losslessly compressed, provider-independent PyFluid
  physical-ID artifact with text metadata. Its core field contract covers all
  current initial data: Newtonian fluid primitives, optional CT face magnetic
  fields, and—when present—relativistic fluid primitives plus physical ADM
  metric, extrinsic curvature, lapse, and shift. Flat Minkowski hydro is a
  deliberately redundant round-trip control. Optional exact evolution caches
  accelerate same-grid loading and seed reconstruction but do not redefine
  the physical interchange format. The artifact can initialize
  evolution directly and must not encode timestepper, reconstruction,
  cadence, or other mutable run controls. The implementation writes the
  physical core plus optional exact evolution caches through the checkpoint-
  proven parallel HDF5 field I/O; text metadata owns schema, hierarchy,
  centering-relevant layout, physics, EoS, and source parameters. Field names,
  components, labels, roles, and centerings are declared only in
  `schema/fields.toml`. [DONE]
- Validate a one-target solve against the current path, a chained target
  against a fresh-from-TOV control, flat-HD and relativistic direct-load
  round trips, and direct same-grid evolution loading against the original
  materialized state. Measure the sequence speedup before changing elliptic
  symmetry. A two-level flat-HD artifact reproduces every step-zero scalar and
  one complete RK4 step exactly except wall-clock throughput columns. A
  flat-Minkowski RMHD+CT artifact reproduces its primitive/conserved preflight
  and all step-zero scalars exactly while exercising ADM and every active face
  field. An N=24 `Omega=0.001` RNS artifact reloads in seconds, reproduces all
  step-zero scalars and C2P/P2C preflight values exactly, and emits no XCTS
  solve. Continuing that artifact to `Omega=0.0011` converges in 21 matter
  iterations with residuals below `5.90e-13` and writes the next artifact.
  This small step did not materially reduce matter iterations versus the
  22-iteration fresh small-spin solve, so no persisted-seed speedup is claimed.
  Cross-grid interpolation/exterior materialization remains explicitly
  unimplemented; exact loads reject hierarchy mismatch. Architecture,
  catalog regeneration, all three production dimensions, the final `2.98 s` quick
  gate, and the final 42/42 standard gate pass. The standard RMHD validator now
  permanently requires exact fresh/direct-load physical scalar rows through
  one evolution step for a flat-Minkowski RMHD+CT artifact. The corrected
  persisted XCTS seed reconstructs the conformal extrinsic tensor from Z4c as
  `A_conformal = psi^6 a_tilde`; the focused `0.001 -> 0.0011` continuation
  still passes with the same 21 matter iterations and residuals. Canonical
  runtime cases are `0.92--0.99` of their frozen baselines. The final
  production cache-hit build is `0.49 s`; no new cold compile was measured.
  [DONE]
- Assess aligned, centered Cartesian reflection/octant symmetry only after the
  reusable sequence path is sound. Do not conflate discrete component parity
  boundaries with a true two-dimensional axisymmetric cylindrical XCTS
  formulation. [PENDING]

## Cycle 217 — Grid-independent physical-ID materialization (DONE)

- Correct the physical-ID abstraction: an artifact is a hierarchy-backed
  initial-data provider, not an exact-layout evolution restore. Route both
  evolution startup and persisted sequence continuation through one
  configured-ID materialization call. Preserve exact evolution caches only as
  a same-layout optimization. The artifact is deliberately a superset:
  evolution and sequence consumers ignore optional fields they do not need.
  [DONE]

## Cycle 218 — Persisted live sequences and symmetry audit (DONE)

- Save every accepted target by default for a genuine multi-target rotating-ID
  sequence, with `initial_data.save_sequence=0` as the explicit disable knob.
  Do not automatically save ordinary one-model TOV or RNS construction. Write
  the common directly evolvable physical-ID artifact while retaining the live
  conformal and matter fields for the next target. Keep
  `initial_data.output_file` as the optional singular final artifact and write
  a compact sequence index rather than inventing a second file format. [DONE]
- Audit symmetry across analytic providers, stored artifacts, elliptic solves,
  evolution boundaries, CT, and diagnostics. TOV octant evolution already has
  a focused full-versus-octant charge check, catalog-driven fluid/Z4c parity,
  CT symmetry handling, and volume-charge factors. The common elliptic solver
  currently assigns Robin or Dirichlet data to every nonperiodic face, so a
  fresh RNS/XCTS reflection/octant solve is not supported. Stored full-domain
  data can be geometrically restricted to a contained octant target through
  the common materializer, although that direction has no focused test;
  octant-to-full parity reconstruction and spherical GR surface reconstruction
  are not implemented. [DONE]
- Add focused sequence save/load/continue/evolve checks, rebuild all production
  dimensions, and run architecture, generated, quick, and proportionate
  relativistic gates before committing. The finalized three-rank focused gate
  passes four checks: zero-spin TOV equivalence, live two-target SLy sequence,
  exact final-model evolution load, and persisted intermediate-model
  continuation. Its two N=24 artifacts are 18.6 MiB each with ZLIB@9; the final
  exact-load scalar error is zero, its CFL 0.9 step has no recovery events, and
  the independently continued target differs by at most `1.9514e-3` in the
  selected global scalars at the configured `1e-5` matter tolerance. All
  production dimensions build, quick passes, the pre-final-metadata standard
  gate passes 42/42 in `244.38 s`, and the final RMHD standard physical-ID gate
  passes exact and cross-grid loading. Precommit runtime ratios are
  `0.93--0.99` of the frozen baseline and the integration cache-hit build is
  `0.21 s` with no compiler invocations. Implementation commit `504b4b8` is
  captured from a clean worktree under
  `runs/baselines/20260722_211331_rotating_id_sequences/`; its fresh standard
  suite passes 42/42 and all four archived performance cases pass at
  `0.91--0.99` of the frozen baseline. No full or stability suite was run.
  [DONE]
- Support materialization onto the same physical grid, finer or coarser grids,
  and target domains larger than the saved source. Interpolate the saved
  physical primitives and ADM fields on their catalog centerings; use the
  appropriate physical exterior state rather than extending evolution-cache
  values. Preserve the CT divergence contract when magnetic face fields are
  remapped. Exact layouts read cached conserved/Z4c fields. Other layouts read
  source-hierarchy primitives, nodal ADM data, and CT faces; reconstruct target
  conserved and Z4c state; use asymptotically flat ADM falloff outside a
  nonperiodic source; and project logical CT faces only if interpolation creates
  measurable divergence. Periodic targets require an integer number of saved
  periods. A direct remap does not re-solve constraints; rotating continuation
  does run XCTS again on the target hierarchy. [DONE]
- Add focused exact, resolution-change, and larger-domain controls before
  widening validation. Record separately what is source-inspected, compiled,
  and executed. The permanent RMHD standard gate now evolves exact, 8x8
  coarser, and 40x40 two-period finer/larger loads; the exact fresh/load scalar
  rows match, uniform extensive densities scale with volume, and CT divergence
  remains below `1e-13`. Its independent AMR+CT checkpoint restart remains
  bitwise exact. A manual RNS control materialized an N=24 `[-16,16]^3`
  `Omega=0.001` artifact onto N=32 `[-20,20]^3`, continued to `Omega=0.0011`
  in 22 matter updates with XCTS residuals below `6.52e-13`, wrote a new
  artifact, then loaded that artifact without XCTS and completed one CFL `0.9`
  evolved-spacetime step. That step had zero C2P fallbacks/atmosphere resets
  and scalar validation passed with mass drift `7.95e-11`. Architecture,
  generated operators, Python syntax, all production dimensions, the `2.48 s`
  quick gate, and all 42 standard gates (`244.50 s`) pass. Runtime benchmark
  ratios on the precommit run are `0.92--0.97` of frozen baselines; the
  integration cache-hit build is `0.52 s` with no compiler invocations and is
  not a cold-build result. Implementation commit `3e22a40` is captured cleanly
  under `runs/baselines/20260722_175457_physical_id_materialization/`: its
  standard suite and performance commands return zero and its three-repeat
  runtime medians are `0.90--0.96` of the frozen baselines. No full or
  stability suite was run.
  [DONE]

## Cycle 219 — Physical sequence organization (DONE)

- Organize automatically persisted rotating-star sequences by EoS, rotation
  law, and fixed central density beneath the configured run directory. Include
  named EoS presets and differential-law parameters in human-readable path or
  text metadata, while keeping each artifact's `Header.meta` authoritative and
  preserving `initial_data.sequence_output_dir` as an explicit override. Named
  fits now record and validate `eos_preset`; inactive custom polytrope knobs do
  not control compatibility. Legacy named-EoS artifacts without the preset are
  rejected because their physical EoS cannot be verified. [DONE]

## Cycle 220 — Informative sequence model names (DONE)

- Include the prescribed angular velocity, base-grid resolution, and hierarchy
  level count in each persisted sequence model name. Label the prescribed
  quantity as `omega`, not dimensionless spin, because `J_ADM/M_ADM^2` is a
  measured property of the converged model. Keep the exact box hierarchy and
  physical diagnostics in artifact metadata and the sequence index. Names use
  parser-safe hyphen separators because AMReX treats `=` inside an unquoted
  input path as a parameter delimiter. [DONE]

## Cycle 221 — Octant elliptic ID, SRMHD exact references, and flat/GR audit (DONE)

- Extend the shared CTT/CTTK/XCTS elliptic boundary contract with centered
  Cartesian reflection planes and component parity. Validate zero-spin TOV
  and nonzero uniform-rotation RNS octants against matched full-domain solves,
  including constraints, masses, angular momentum, surface geometry, and short
  CFL 0.9 evolution. Keep spherical surface extraction limitations explicit
  until omitted sectors are reconstructed. Scalar/polar-vector/tensor parity
  is centralized in the field catalog, magnetic fields use axial-vector
  parity, the nodal and cell elliptic ghost paths touch reflection faces only,
  and ordinary outer Robin/Dirichlet behavior is preserved. The manufactured
  reflection-plus-Robin Poisson control has `1.786e-3` L-infinity error at N=24;
  the complete CTT/CTTK/XCTS elliptic executable passes. A nonrotating star
  supports x/y/z octant symmetry. A star rotating about z supports z-equatorial
  symmetry but rejects ordinary x/y reflection because that is not a physical
  symmetry of azimuthal velocity/shift. [DONE]
- Solidify reusable-ID symmetry behavior: distinguish contained full-to-octant
  restriction from octant-to-full parity reconstruction, implement the latter
  only through catalog/tensor parity rather than scalar clamping, and add
  focused round trips if it can preserve the physical-field contract cleanly.
  Reflection faces are persisted in `Header.meta`; coordinates mirror into the
  stored sector before interpolation and catalog parity is applied afterward.
  The focused AMR TOV control records a `1.339725e-2` full/octant difference at
  matched finest spacing, while expanding the octant artifact to the full
  domain changes selected scalars by only `1.784626e-6`; its reconstructed
  constraints remain finite. The focused gate is full-tier because the fresh
  full AMR XCTS solve takes about 397 seconds on one local rank. [DONE]
- Add self-contained 1D SRMHD Balsara shock-tube comparisons. Verify which
  published cases have exact relativistic MHD Riemann solutions, implement or
  use an independent exact postprocessor without importing production physics,
  and keep exact/reference labels honest. The tracked Giacomazzo--Rezzolla
  Balsara 1--5 profiles are validation data only. Their total pressure is
  converted independently to gas pressure and flat Valencia conserved fields.
  At N=512 with MP5/HLL/SSPRK3 and CFL 0.9, maximum normalized component L1
  errors are `0.004413`, `0.006203`, `0.011040`, `0.013123`, and `0.003643`.
  Every normal magnetic field is constant to `1e-12`; all cases have zero C2P
  fallbacks and at most three counted atmosphere resets. [DONE]
- Audit compile-time and runtime specialization for SRHD/SRMHD versus
  GRHD/GRMHD. Quantify the flat-spacetime cost, compare AthenaK's separation,
  and simplify only boundaries exposed by measurements without duplicating
  physics implementations. PyFluid previously sent flat RHD/RMHD through
  stored Z4c metric loads and stored-metric derivative sources. One
  host-selected flag now sends verified Minkowski configurations to the
  existing analytic fixed-metric kernels; physical-ID metadata preserves this
  choice across exact loads, while curved and solved ADM data remain generic.
  Three-repeat one-rank medians on 4096 cells for 128 MP5/HLL/SSPRK3 steps rise
  from `0.5695` to `0.7466` Mcell/s for SRHD (31.1 percent) and from `0.1053`
  to `0.1075` Mcell/s for SRMHD (2.1 percent). AthenaK likewise composes common
  SR/GR modules rather than using a different flat RMHD algorithm. [DONE]
- Finish with generated/architecture checks, all production dimensions,
  focused relativistic gates, quick/standard validation, performance and
  compile measurements, local commits, and a clean named baseline.
  Architecture and generated consistency pass; all production dimensions
  build; complete one-rank RHD, RMHD, and fixed-curved-spacetime standard gates
  pass; the new focused symmetry gate passes on fresh runs; and the complete
  three-rank quick gate passes. The production cache-hit build is `0.39 s`.
  The complete three-rank standard gate and canonical MPI performance benchmark
  were not executable because the external-socket approval was rejected by the
  account usage limit until 2026-07-28; do not count that as solver failure or
  as validation. The coarse N=24 RNS full/z-half comparison differs by
  4.99 percent in its worst global scalar but has identical radii, strict XCTS
  residuals, CFL 0.9 MP5 evolution, and zero recovery events. Its raw rest-mass
  drift (`9.68e-8` full, `6.92e-8` half) exceeds the generic dynamic-spacetime
  `1e-8` policy, so the focused test explicitly bounds it at `2e-7` and remains
  a smoke, not long-evolution conservation evidence. Final review added a
  mixed-parity cell-centered conformal-vector control. It exposed and fixed
  two boundary-lowering defects: tensor operators now receive per-component
  boundary types, and explicit even reflection coefficients cannot trigger
  the legacy coefficient-only Robin inference. The complete elliptic
  executable then passed through CTT, CTTK, and XCTS; the reflected cell-vector
  L-infinity error is `6.46e-6` and the reflected nodal-vector error is
  `5.61e-5`. All production dimensions rebuilt after the final source cleanup
  in `3.54 s` with three misses; the immediate no-op rebuild took `0.49 s`. Final
  architecture, Python syntax, and diff-whitespace checks pass. Fresh one-rank
  reruns of the octant TOV (`55.86 s`) and z-half RNS (`91.77 s`) reproduce
  their pre-review step-zero scalars and physical diagnostics exactly; the RNS
  solve and three CFL 0.9 steps complete with zero recovery events, while its
  expected `6.92e-8` mass drift still fails the generic `1e-8` policy and passes
  only the explicitly documented `2e-7` focused-smoke bound. The named baseline
  and canonical MPI benchmark remain unavailable for the stated external
  execution limit rather than being represented by cached artifacts. [DONE]

## Cycle 222 — Balsara conservation audit and plots (DONE)

- Re-run the five exact 1D SRMHD Balsara comparisons at N=512 with
  MP5/HLL/SSPRK3 and CFL 0.9, report conservation without confusing the
  established open-domain policy with a closed problem, and add concise plots
  for exact profiles and refinement evidence. All five exact-profile gates
  reproduce their Cycle 221 normalized component L1 errors. Balsara 1 has
  roundoff mass and energy drift (`1.38e-15`, `1.51e-15`) and no recovery
  events. Cases 2--5 have substantial raw total changes because their finite
  open boxes do not constitute closed conservation controls; cases 2--4 also
  record two, three, and two terminal atmosphere resets. The plotted Balsara 4
  MP5 solution follows the exact wave structure but visibly rings in the
  strongest post-shock regions, consistent with treating its `0.0131` maximum
  normalized L1 as bounded agreement rather than a finished robustness result.
  [DONE]
- Make the convergence label exact. The current Balsara 1 CT refinement uses
  uniform 2D grids N=128, 256, and 512 against an N=1024 numerical reference;
  it is not an AMR convergence test. Errors decrease from `5.536e-3` through
  `3.382e-3` to `1.445e-3`, with interval rates `0.71` and `1.23`, which is
  sensible shock-dominated near-first-order behavior. The separate smooth
  RMHD entropy-wave AMR+CT gate remains the actual RMHD AMR evidence. [DONE]
- Probe whether the open boundaries alone explain the non-roundoff totals by
  doubling Balsara 2 to a periodic domain containing two causally separated
  Riemann problems at the same spacing. MP5 reproduces two atmosphere resets,
  mass drift `9.14e-4`, and energy drift `2.01e-4`; PWL has two resets and
  drifts `1.81e-4` and `1.54e-4`, while PWC has no reset and conserves mass to
  `1.89e-15`. This diagnostic configuration is not retained in the standard
  gate because it deliberately fails today. Source inspection confirms that
  the stage-candidate conservative first-order flux correction is invoked only
  for RHD. A CT-consistent RMHD candidate limiter is therefore the main open
  production issue exposed by this audit; do not mislabel post-update
  atmosphere projection as conservative reconstruction fallback. [DONE]

## Cycle 223 — Balsara reconstruction scan and named baseline (DONE)

- Run all five exact Balsara problems at N=512, HLL/SSPRK3, and CFL 0.9 with
  every production component reconstructor: PWC, PWL, WENO-Z, MP5, and TENO5.
  Persist exact-profile maximum normalized L1, global-range overshoot, total
  variation ratio, recovery counts, scalar drifts, throughput, and run paths
  in one scan TSV and generate overlays. No high-order method dominates. MP5
  has the lowest L1 on Balsara 1 and 5; TENO5 is lowest on 2; WENO-Z is lowest
  on 3; and PWL is lowest on 4. PWC is substantially more diffusive but is the
  only method with zero atmosphere resets across all five. [DONE]
- The hoped-for non-ringing WENO result does not occur on the extreme
  counter-streaming Balsara 4 problem. WENO-Z develops an off-scale central
  spike (`0.298` maximum normalized L1, `45.4` normalized global overshoot,
  and `182.8` maximum component TV ratio) despite zero terminal resets. PWL is
  the best completed high-order result there (`0.00890` L1, zero resets) but
  still rings (`4.31` maximum TV ratio). MP5 has `0.0131` L1, `6.89` maximum
  TV ratio, and two resets; TENO5 has `0.0279` L1, `7.47` maximum TV ratio,
  and 31 resets. Keep WENO-Z's off-scale result visible in a separate plot so
  it cannot flatten the useful method comparison. [DONE]
- Capture the clean committed Cycle 221/222 state at Git `47ae7ba` in
  `runs/baselines/20260725_221243_elliptic_symmetry_srmhd/`. All 42 standard
  gates pass with `274.79 s` summed gate time. The canonical MPI benchmark
  also returns zero: current/frozen throughput ratios are `0.983`, `0.932`,
  `0.945`, and `0.961` for its four cases. This baseline predates the
  reconstruction-scan tool only; the scan does not change production code.
  [DONE]
