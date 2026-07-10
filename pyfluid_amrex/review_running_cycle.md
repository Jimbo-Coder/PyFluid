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
