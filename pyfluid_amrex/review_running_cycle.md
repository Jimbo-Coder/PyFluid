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
  - Commit the current AMReX and repository-agent state without including unrelated JAX/Numba/reference-tree changes. Commit `b366fb7`. [DONE]
  - Run a long acoustic packet through the cylindrical/spherical origin with automatic anisotropic AMR, SSPRK3, and subcycling; compare transmission, symmetry, conservation, causal exterior behavior, and profile error against Cartesian evolution. [WIP]
    - Cartesian automatic anisotropic AMR with subcycling reaches L2 and passes scalar/causal checks. Cylindrical `r phi` reaches L2 but produces active-cell NaNs during the first subcycled step. Isolate fixed-vs-auto refinement and subcycling before changing the solver. [WIP]
    - Removed the duplicated non-subcycling PWC FillPatch path and nonfinite neighbor-copy repair. One configured AMReX interpolater now serves level creation, remake, FillPatch, subcycling, and repair seeds. Conservative anisotropic FillPatch remains finite. [DONE]
    - Long dynamic anisotropic regridding changes L2 from `(2,2)` to `(2,1)` near `t=0.118` and AMReX rejects the resulting proper-nesting transaction. A full conservative hierarchy rebuild avoids the abort but spuriously cascades to L12, so that experiment was reverted. Dynamic anisotropic ratio changes remain unresolved; use fixed isotropic ratios for the long origin/source physics gates. [WIP]
    - Runtime curvilinear momentum diagnostics now rotate local basis components into physical Cartesian momentum through one shared coordinate transform. [DONE]
  - Run a long rotating two-hot-spot source with open radial boundary and automatic anisotropic AMR; require refinement to follow the emitted wave to the boundary, no pre-contact exterior change, smooth metric termination, positive headroom, and honest hierarchy diagnostics. [PENDING]
  - Use these runs to decide whether single-patch hydro is complete and identify the remaining MHD+CT+viscous/implicit stress matrix without weakening existing gates. [PENDING]

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

## Cycle 48 [in progress]
- Produce two production-style curvilinear AMR demonstrations. [WIP]
  - Configure a cylindrical `r phi` acoustic/density packet whose first radial center is `dr/2`, crosses the origin without an inner hole, and continues outward with automatic state tagging, SSPRK3, subcycling, protected conservative transfer, and metric-limited refinement. The first attempt correctly refined to L7 but exposed excessive origin CFL cost from an over-resolved angular base; rerun with coarser angular sampling and automatic AMR retained. [WIP]
  - Run long enough to validate origin transmission, conservation, causal behavior, hierarchy movement, level headroom, and AMR efficiency; render honest pressure-perturbation movies with and without boxes. The run passes causal quiescence at `8.85e-12`, stops smoothly at L2 with eight levels of headroom and 0.55% budget use, and shows the inward lobe crossing and re-emerging from `r=0`. [DONE]
  - Configure and run a rotating two-hot-spot source with the same standard AMR path and open outer radial boundary; require refinement to follow emitted waves through the domain without pre-causal artifacts. [WIP]
    - Replace the separable radial/angular ring profile outright with rotating physical-space compact disks. The source is now `rotating_compact_heating`, uses one `source_sigma`, and has exact support at three sigma. The final 3-rank run reaches L2 with eight levels of headroom and remains far below budget. [DONE]
    - The strict causal-exterior validator still measures `1.67e-6` outside a `1.2` times background-acoustic cone, above its `1e-7` tolerance; it passes with a factor-two cone. Distinguishing finite-stencil numerical-domain tails from an underestimated heated-flow signal speed remains before calling this demonstration strictly causal-clean. [WIP]
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

## Cycle 85 [in progress]
- Promote efficient aligned subcycling for dynamic AMR and evaluate a production-quality representation-error tagger before changing the AMR metric. [WIP]
  - Audit and update registered defaults so AMR runs use time subcycling plus aligned regridding unless an input explicitly selects synchronized or no-subcycling behavior. Registered defaults are now `subcycling=on` and `regrid_mode=aligned`; all 27 existing validated AMR inputs explicitly retain synchronized cadence until individually retuned because their intervals were selected in coarse-step units. An omitted-default AMR probe records aligned/on in canonical `run.in`, passes conservation/causal checks, and the single-level 3-rank performance guard remains 0.99 of baseline. [DONE]
  - Add an equal-cost estimator comparison for current normalized Lohner curvature versus Richardson/shadow references on smooth, discontinuous, radial, and nonsymmetric flows. Source audit: AMReX delegates tags to the application; FLASH production defaults to normalized Lohner curvature; Berger-Oliger and Chombo document Richardson/truncation estimators. The 2016 Colella mapped-FV paper discusses truncation error at refinement boundaries but does not define a runtime truncation tagger. Implement `compare_amr_estimators.py`: its coarse control is compared directly with level 0 of a full-domain two-level AMR shadow run, so reflux and restriction are performed by the production solver. Every constituent run passes shared scalar validation. At a 10% tag budget, mean Richardson-error-energy capture is 0.665 for Lohner, 0.782 for Richardson, 0.745 for union, 0.778 for Richardson-heavy weighting, and 0.730 for candidate/sentinel sampling over 20-29% of the domain. [DONE]
  - Evaluate robustness/cost before changing production. Full Richardson shadow evolution costs 4.6-8.7 times the coarse evolution in the current small hydro cases. Spatial Lohner gating recovers most shock benefit cheaply but cannot discover the smooth-wave error Lohner missed; deterministic 6-25% sentinels do not fix that robustly. Select occasional global unbiased Richardson with Lohner on intervening regrids as the runtime prototype; do not promote union/weighted complexity or a spatially gated shadow branch from this evidence. [DONE]
  - Implement and validate the selected temporal hybrid as an optional runtime branch before considering it for production. [PENDING]
  - Run focused AMR gates, the quick suite, and performance regression after any default or production-tagger change. [PENDING]

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

## Cycle 89 [in progress]
- Improve production-style Lohner AMR with a robust, efficient adaptive Richardson ladder while avoiding per-flow budgets, fixed blend weights, and cadence tuning. [WIP]
  - Define one shared estimator ladder: global Lohner, Richardson over block-clustered biased/cached regions, error-driven threshold calibration and region growth, unbiased discovery, and full-domain saturation when regional work reaches break-even. No tag fraction, fixed blend weight, or manually selected regional extent is part of the policy. [DONE]
  - Prototype AMReX-block and halo costs against the independent coarse-to-medium estimator and medium-to-fine holdout on all five flows. Ordinary Lohner regions require 73.8% of the domain on average because distributed smooth waves, rings, and KH layers touch nearly every eight-cell block. Lohner-local-maxima seeding reduces localized-cloud work but correctly saturates to global Richardson on the tiny smooth/radial and distributed KH grids. Sparse probes add cost without improving holdout capture in these snapshots. [DONE]
  - Exercise cache and causal-age behavior on the noisy KH time sequence; derive escalation from observed error and movement rather than fixed iteration counts. Cached Richardson plus calibrated Lohner remains the effective cheap tier. A current regional Richardson pass over block-aware noisy-KH candidates already covers 92-100% of the domain, so the break-even rule should skip regional work and either reuse cache or run global calibration. The cached two-cell analysis buffer catches every independent holdout violation between calibration times. [DONE]
  - Implement only a policy that survives accuracy-versus-cost analysis. Do not add a second shadow evolution to production: it is too expensive and would duplicate stateful timestepper side effects. Evaluate the hierarchy-native coarse-prediction/restricted-fine estimator at existing synchronization points instead. [WIP]
    - Reference decision: SAMRAI `StandardTagAndInitialize` supports Richardson, gradient detection, and refine boxes individually or in combination; it deliberately applies Richardson first and gradient detection second, selects criteria by cycle/time, advances both the hierarchy level and a coarsened temporary level through the shared level integrator, then restores pre-estimate state. Follow this generic regrid-advance workspace pattern rather than adding physics-specific estimator advances. [DONE]
    - Condense the runtime policy experiment to one nested schedule with only three candidate global calibration periods (12, 24, and 48 regrid events): Lohner every event; regional spatial defect over normal Lohner/cache at period/6; lower-threshold Lohner candidates at period/4; expanded cached/lower-threshold regions at period/2; full temporal Richardson at the global period. Coincident events execute only the broadest due check, and regional work saturates to global at measured AMReX box-cost break-even. [DONE]
    - Calibrate nested Lohner thresholds from the latest full temporal Richardson result; do not expose threshold fractions, regional extents, or separate cadences as production parameters. Select one schedule by held-out error versus wall time, then remove the losing experimental policies. [PENDING]
    - Make the AMR driver selection explicit before adding new estimators: replace the ambiguous supported value `amr.tag_metric=auto` with `amr.tag_metric=lohner` across C++, maintained inputs, and validation tools. Keep `none` and `lohner` as the only accepted values until `richardson` and `hybrid` actually exist; do not retain an `auto` alias. All 1D/2D/3D builds pass, the obsolete value aborts with the supported set, and the quick suite remains at its known 16/17 state with only the pre-existing pulsating cut-cell AMR/subcycling conservation defect. [DONE]
    - Strengthen the policy comparison to four independent resolutions and four temporal policies over smooth sine, radial pressure bump, advected cloud, and noisy KH. Coarse-to-medium defines the Richardson predictor, medium-to-eight-times-finer defines held-out error, and the four-times/eight-times-finer difference checks reference convergence. The reference-error energy is 3.5% of the held-out error on average, so the target is materially finer rather than circular. At a 0.3% normalized error tolerance, Lohner selects 30.0% of cells, captures 99.7% of held-out error energy, and has 0.910 worst recall. The stride-2 cached/calibrated hybrid selects 22.6%, captures 99.8%, has 0.913 worst recall, but costs an estimated 4.87x because of full-domain shadow calibration. Stride 4 costs 3.41x and is slightly worse than Lohner; global Richardson costs 7.81x for only a marginal gain. Reject the shadow-based hybrid as noncompetitive. [DONE]
      - Keep Sod in the existing three-resolution and exact-solution analyses. Its additional full-domain level-3 reference exposed six nonphysical cells from quartic-conservative transfer at CFL 0.9 by step 7; do not hide that distinct transfer/stability defect by lowering CFL or silently changing interpolaters inside this estimator comparison. [DONE]
    - The viable next prototype is hierarchy-native: preserve the ordinary coarse prediction immediately before reflux, compare it with the restricted fine state after synchronization, and cache that normalized discrepancy over covered coarse cells. This reuses every existing timestepper/physics path and pays no second full-domain evolution. It still needs unbiased discovery outside the current fine footprint before it can replace Lohner, so production remains `lohner`. [PENDING]
  - Validate AMR accuracy, subcycling, conservation, hierarchy behavior, Mcells/s, and simulation-time/real-time before changing any default. [PENDING]

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
  - Do not add a production interface. Although the adaptive policy analysis is promising, PyFluid does not yet have a generic side-effect-safe global Richardson evolution operator. Remove the experimental harness and scratch outputs; retain pure Lohner as the only production state tagger and preserve the measured conclusions here for a future implementation. [DONE]

## Cycle 93 [in progress]
- Finalize the current AMReX state for production, remove discarded estimator experiments, validate and freeze performance, commit the coherent implementation, then produce a high-resolution compressed-output AMR movie. [WIP]
  - Audit the complete dirty AMReX diff and preserve unrelated JAX/Numba/reference-path work untouched. The coherent AMReX diff contains callback-based change-aware regridding, explicit Lohner naming, robust subcycling defaults, maintained-input compatibility pins, and regression coverage; unrelated JAX/Numba files remain unstaged. [DONE]
  - Keep only runtime-supported, validated AMR/subcycling behavior. Remove the discarded Richardson analysis harness and scratch runs; retain `none` and `lohner` as the only production taggers. Registered defaults are aligned regridding, subcycling on, and local-CFL fine-level steps; fixed spatial ratios remain an explicit faster option. [DONE]
  - Rebuild all dimensions, run relevant standard/full gates and performance benchmark, then capture a named baseline. All 1D/2D/3D targets build. The full AMR gate passes dynamic/global tagging, curvilinear equivalence and freestream, smooth AMR convergence, subcycling agreement, AMR+CT, and exact restart recovery. Full reference and analytic gates pass DMR, Brio-Wu AMR+CT, Sod exact, causal boundaries, sources, gravity, EoS acoustics, dispersion, and diffusion decay. The benchmark passes all four performance guards: 3-rank 2D `108.78 Mcells/s`, 4-rank 2D `131.55`, 3D PWC `5.97`, and 3D WENO-Z/RK4 `0.182`. The broad standard gate remains 16/17 only because of the pre-existing pulsating moving-cut-cell mass/ledger defect; do not weaken that check. [WIP]
  - Commit only coherent AMReX production files. [PENDING]
  - Run and postprocess one high-resolution, high-frame-count structured-flow AMR case using local-CFL aligned subcycling and compressed HDF5 output. [PENDING]
