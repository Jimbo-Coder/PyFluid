This repo is a Newtonian HD/MHD solver. The AMReX/C++ path is the production
implementation. Python/JAX and Python/Numba are reference/baseline paths unless
the user explicitly asks to work on them.

Implementation conventions are defined in `pyfluid_amrex/docs/code_conventions.md`.

## Code Priorities

- Keep solver logic importable from modules; scripts should call shared code.
- Do not spend development time on JAX/Numba parity unless it is needed for a baseline comparison.
- Keep initial data in `pyfluid/initial_data.py`; simulation classes decide when to apply it.
- Use primitive-variable reconstruction for production runs unless a test explicitly selects conserved variables.
- Keep non-magnetized runs on hydro state vectors; do not allocate magnetic fields for hydro-only runs.
- Favor clear module boundaries: options/configs, initial data, reconstruction, Riemann solvers, RHS, timestepping, diffusion, diagnostics, plotting.
- Do not maintain separate physics implementations for different dimensions. Define coordinate-independent physics once, generate coordinate/dimension-specific kernels from that source where practical, and treat lower-dimensional runs as projections of the same operator. Dimension-specific branches are acceptable only for AMReX storage/centering/topology or a clearly documented unsupported feature.
- Keep AMReX AMR on the standard finite-volume contract: FillPatch/prolongation for ghosts/new fine data, flux-register reflux for coarse-fine conservation, average-down/restriction for covered coarse cells, and edge-EMF reflux plus face average-down for CT MHD.
- Use one explicitly selected coarse-fine state interpolater for level creation, regridding, FillPatch, subcycling, and repair seeds. Do not override it by coordinate system, refinement ratio, or subcycling mode, and do not repair nonfinite transfer output. Reject genuinely unsupported designs before evolution; record and fix bugs in intended combinations rather than temporarily prohibiting them.
- For Cartesian multipatch, production finite-volume evolution uses unique owned cells. Patch overlap is only patch context for ghost/interpolation work until a conservative duplicated-overlap/overset reconciliation method is implemented and validated.

## Style

- Use standard, boring C++ naming: types and structs in `UpperCamelCase`, functions and local variables in `lower_snake_case`, constants/enumerators in the existing project style, and private data members with a trailing `_`.
- Do not give a local variable the same name as a helper function or class method. Prefer precise names such as `coordinate_kind`, `active_axes`, `patch_system_kind`, and `num_components`.
- Keep Python tools and the SymPy generator PEP 8-style: modules, functions, and variables in `lower_snake_case`; classes in `UpperCamelCase`.
- Generated C++ should follow the same naming conventions as handwritten C++ unless a mathematical name is clearer and local to the generated kernel.

## Coordinate And Component Conventions

- `mesh.coordinate` selects the coordinate family. `mesh.axes` selects the active runtime axes for the executable dimension.
- Canonical 3D axis orders are Cartesian `x y z`, cylindrical `r phi z`, and spherical `r theta phi`.
- Lower-dimensional `mesh.axes` values must preserve the canonical order. Supported finite-volume layouts are ordered subsets of the canonical 3D order: Cartesian `x`, `y`, `z`, `x y`, `x z`, `y z`, `x y z`; cylindrical `r`, `phi`, `z`, `r phi`, `r z`, `phi z`, `r phi z`; and spherical `r`, `theta`, `phi`, `r theta`, `r phi`, `theta phi`, `r theta phi`.
- Conserved vector components are runtime-direction components, not permanent physical-axis names. Use `MOM1/MOM2/MOM3`, `B1/B2/B3`, `mom1/mom2/mom3`, and `b1/b2/b3`; interpret them through `mesh.axes`.
- Run parameters for vector quantities use `1/2/3` suffixes such as `v1_0`, `b2_left`, `inflow_v3`, `source_a1`, and `solid_velocity_2`. Do not add new solver-state or user-facing vector names with `x/y/z` suffixes.
- Define AMReX grid functions once in `pyfluid_amrex/schema/fields.toml`: name, mathematical label, tensor shape, components, role, requirements, and centering. `field_catalog.py` is the strict loader/model/validator, not a second declaration source. Generated C++ descriptors, output writers, diagnostics, and plotting tools consume that schema; do not duplicate GF names, component IDs, labels, or centerings locally.

## Numerical Work

- Default to CPU execution on this Mac mini.
- Use 3 MPI ranks for normal AMReX jobs unless benchmarking rank scaling.
- MPI requires local sockets and cannot run inside the Codex sandbox. Use the
  approved socket-capable execution path for MPI validation; a sandbox MPI
  launcher failure is an environment limitation, not a solver result.
- Python AMReX tools must construct MPI launches through
  `validation_common.py` (`mpi_prefix`, `mpi_command`, `run_mpi`, or
  `run_amrex`); do not duplicate `mpirun -np` command assembly.
- Default CFL is 0.9 unless a stability scan or deliberately conservative validation run says otherwise.
- Validate changes with small reproducible runs before long simulations.
- `pyfluid.physical_check_int=0` is the normal production default. Use `pyfluid.physical_check_int=1` for development, validation, or new run setups so accepted states fail immediately when rho/pressure become nonphysical. For deeper AMReX debugging, enable `pyfluid.debug_poison=1`, `pyfluid.debug_check_finite=1`, and/or `pyfluid.debug_check_physical=1` on small runs.
- After performance-sensitive AMReX changes, run `uv run python pyfluid_amrex/tools/benchmark_pyfluid.py`; use `--allow-warn` only when intentionally collecting current numbers despite a regression.
- Track compile-sensitive changes with `uv run python pyfluid_amrex/tools/benchmark_compile.py --target <target> --label <label>`. Record cold compiles and cache-hit rebuilds separately; do not present a cache hit as clean-build performance. Use an individual kernel object target while iterating and `pyfluid_amrex_production` at integration points.
- Do not claim broad correctness without scalar diagnostics, conservation checks, and a relevant analytic or high-resolution comparison.

## Validation

- Individual runs must write scalar diagnostics and report mass/energy/momentum drift, divb when MHD is active, floors, timestep, AMR level/budget fractions, Mcells/s, and simulation-time/real-time. Closed problems should conserve to roundoff unless the run is explicitly open, driven, diffusive, or otherwise non-conservative.
- Scalar diagnostics are runtime AMReX reductions written to `scalars.tsv`; Python postprocessing reads them after the run and should not recompute the core totals as a replacement for runtime diagnostics.
- New AMReX validation scripts should use `pyfluid_amrex/tools/validation_common.py` (`check_scalars` or `scalar_validation_row`) for scalar sanity, policy inference, and history rows instead of reimplementing `scalars.tsv` checks. Prefer readable declarations such as `check_scalars(rows, momentum=False)` when a conserved quantity intentionally does not apply.
- Analytic or exact comparisons belong in postprocessing, not in the evolution path. Use them whenever the IC has a known solution, and keep the plotted comparison visibly interpretable.
- Do not label MHD shock-tube comparisons analytic until an exact NR MHD Riemann solver is actually wired in. High-resolution Brio-Wu-style references are reference/regression tests, not exact analytic validation.
- Brio-Wu AMR+CT is part of the standard AMReX reference gate; `uv run python pyfluid_amrex/tools/validate_brio_wu_reference.py --include-amr-ct` can also be run directly to check AMR shock-tube improvement, CT normal-field preservation, and rotated x/y consistency against a finer numerical reference.
- Do not add redundant AMR tests just to prove AMR exists. Existing AMR gates cover smooth AMR improvement, subcycling/no-subcycling agreement, restart, Brio-Wu AMR+CT, rotated consistency, and KH reference behavior. Add a narrower reflux-specific regression only when changing coarse-fine flux/register code or when conservation across refinement boundaries is suspect.
- Production-style AMR tests start from a deliberately coarse base grid and must stop because the physical error metric is satisfied. `amr.max_level`, `amr.min_feature_length`, radial guards, and cell budgets are emergency resource limits; reaching one is not successful AMR unless that guard is the explicit subject of the test. Require positive level headroom, a budget safely below its cap, and refinement that follows evolving state. Fixed-level hierarchies remain valid only as isolated interpolation/reflux controls.
- Sod shock is a normal analytic validation, not only a CFL stress case. Keep the exact-profile/integral check in the AMReX suite, and use the nonlinear CFL scan only to probe stability margins.
- Scans/convergence tests go under `plots/current/scans/`; they should show the expected order over the meaningful x/y range and still satisfy the individual-run scalar checks for each branch.
- Feature tests compare feature on/off when that is physically meaningful, track the performance delta, and still run the individual scalar checks. AMR/subcycling comparisons may use coarser equivalence checks when the grid hierarchy changes.
- For ad-hoc AMReX runs, check scalar sanity with `uv run python pyfluid_amrex/tools/validate_run_scalars.py <run_dir>` unless the run intentionally opts into a non-conservative policy such as open, driven, diffusive, or smoke.
- For targeted AMReX coverage, use `uv run python pyfluid_amrex/tools/validate_feature_matrix.py --only <case>`. Use `--suite <name>` for compact branch sweeps and `--list-cases` to discover names.
- For the normal AMReX suite, use `uv run python pyfluid_amrex/tools/validate_amrex_suite.py`. Keep `--level quick` as a sub-3-second smoke and default `--level standard` as a fast confidence gate, not an exhaustive harness. Use `--level full` only when restart/AMR depth, rank scaling, full feature sweeps, or heavier stress checks are needed.
- CFL-edge/stability sweeps are opt-in: `uv run python pyfluid_amrex/tools/validate_amrex_suite.py --level stability`. They include a linear acoustic Nyquist-mode amplification scan over timesteppers, the same amplification scan for one AMR subcycling timestepper, a nonlinear Sod-shock CFL scan, plus pass/fail stress scans for explicit/implicit diffusion, subcycling, AMR metric settings, and MHD CT on/off. The pass/fail stress scans fail only if runs at or below CFL 0.9 fail.
- The AMReX suite can also be sliced with `--kind feature`, `--kind performance`, `--kind analytic`, `--kind reference`, `--kind amr`, `--kind stability`, or `--only <step>`. Use `--list-steps` to see the concrete steps.
- Suite gates should be TSV/log oriented and skip ordinary plot generation; standalone validation tools may still generate human-facing plots by default. Move slow plots, full performance sweeps, and known-heavy stress tests out of standard unless they are the change being worked on.
- Keep dimensional degeneracy and axis-preference covered with `validate_dimensional_consistency.py`: 1D vs 2D `Nx1`/`1xN`, 2D vs 3D `NxNx1`/`Nx1xN`/`1xNxN`, rotated smooth bumps, and periodic diagonal Sod shocks in every active coordinate plane plus the 3D body diagonal (`xy`, `xz`, `yz`, `xyz` where applicable) should agree after squeezing inactive axes and rotating/reflecting vector components into the reference basis.
- Running validation summaries append to `runs/current/validation_history.tsv`; performance and rank-scaling summaries append to `runs/current/performance_history.tsv`.
- Shared validation history also writes bounded compact mirrors: `validation_history_recent.tsv` keeps the last 10 append batches, and `performance_history_recent.tsv` keeps the last 40. Ordinary suite scratch outputs should overwrite/delete themselves on rerun; only compact histories and deliberately generated scan/performance artifacts should accumulate.
- When the current code state is worth preserving as a baseline, run `uv run python pyfluid_amrex/tools/capture_baseline_run.py --label <name>`. This executes one selected validation suite pass, optionally one benchmark pass, and stores only that run's summary/logs under `runs/baselines/`; rolling histories remain rolling histories.
- Ordinary completed runs should go through `uv run python pyfluid_amrex/tools/report_run.py <run_dir>` for scalar validation and scalar plots. Add `--gf-plots` only when level-2 state/aux GF snapshot plots are wanted. IC-specific postprocessing, such as moving-wall work, gravity total energy, oblique-shock angles, diffusion amplitudes, Sod exact overlays, or KH spectra, should be thin scripts that consume the same run directory and shared helpers.
- Normal ad-hoc AMReX runs should use `uv run python pyfluid_amrex/tools/run_case.py <input.in>`. The input file is the only simulation configuration: registered C++ defaults apply when a supported parameter is omitted, and values in the input file replace those defaults. The runner controls execution/reporting only; it does not accept simulation-parameter overrides. It overwrites the configured run folder by default, writes `run.log`, consumes the solver-authored canonical `run.in`, calls `report_run.py`, and runs lightweight scalar-budget physics hooks. Use raw `mpirun` only when intentionally bypassing automatic reporting. Add `--gf-plots` only when level-2 GF snapshots are requested.

## Outputs

- Keep generated artifacts repo-local and ignored by git: `runs/`, `plots/`, `checkpoints/`.
- Grid-function output defaults to parallel HDF5 with `pyfluid.plotfile_compression=best`; use `pyfluid.plotfile_format=amrex` only when native AMReX plotfiles are explicitly needed.
- Checkpoint GFs use losslessly compressed parallel HDF5 while `Header` and
  `Header.meta` remain text. Restart metadata is mandatory and should reject
  incompatible state/layout semantics, not mutable run-control choices.
- Put scan/convergence/scaling plots under `plots/current/scans/`.
- Put ordinary single-run artifacts under `runs/current/<ic_name>/<run_name>/`: logs, scalar TSVs, AMReX plotfiles, checkpoints, scalar plots, movies, and run-specific postprocessing. Each normal input file explicitly sets `pyfluid.run_name` and `pyfluid.output_dir`; the runner does not derive or override them.
- Keep `plots/current/` for scan/convergence/scaling plots and deliberately curated copies, not normal run output.
- Use stable per-run output folders and overwrite ordinary run plots by default; opt into numbered/archive outputs only for explicit comparisons or mobile/UI cache workarounds.
- Each normal run directory should contain the runtime artifacts plus the immediate validation products: `run.log`, `run.in`, `scalars.tsv`, optional `stateGF/`, optional `auxGF/`, `report.tsv`, stable report plots, and `physics_postprocess.tsv` when a lightweight physics hook applies. Stable run-local outputs overwrite on rerun; rolling histories append. GF snapshot plots and heavier level-3 physics plots are request-only.
- Default diagnostics should stay compact: mass, momentum, energy, divb, floors, timestep, throughput, and diffusion residuals when diffusion is enabled.
- Passive scalars are optional conserved tracers for evolution/output only. They must not drive AMR tagging; the AMR metric should follow hydro/MHD state structure using physical gradients and physical cell sizes.
- Auxiliary GFs use one descriptor/output path with two separate classifications: diagnostic/output kind and evolution ownership. Diagnostic kinds are stored state, pure aux computed only at output cadence, and mixed aux that may use live storage or compute temporarily. Evolution kinds are main state, aux state needed/cached by the solver, and non-backreacting extra fields such as passive dye. Do not store pure diagnostic aux fields as solver state.

## Continuity

- After a context reset, read `pyfluid_amrex/knowledge.md`,
  `pyfluid_amrex/references.md`,
  `pyfluid_amrex/docs/infrastructure_conventions.md`, and the tail of
  `pyfluid_amrex/review_running_cycle.md` before changing AMReX. Treat the
  knowledge file as a maintained snapshot: current source and newer continuity
  cycles win when they disagree.
- Keep active AMReX work in `pyfluid_amrex/review_running_cycle.md`. Append one
  cycle per user request, use explicit `PENDING`/`WIP`/`DONE` states, and read
  the file tail before resuming after any interruption or compaction.
- Put source-checked external review findings in
  `pyfluid_amrex/review_backlog.md`. Raw review prose is provenance, not an
  accepted implementation plan; mark stale or rejected claims explicitly.
- Update `pyfluid_amrex/knowledge.md` only for compact learned physics,
  numerical, and AMReX facts plus current validation boundaries and open
  limitations. Put durable ownership and artifact rules in
  `pyfluid_amrex/docs/infrastructure_conventions.md`. Keep exact commands,
  failures, measurements, and historical detail in the continuity cycle and
  run logs. Keep `pyfluid_amrex/references.md` as a router to authoritative
  local sources and external literature rather than a design document.
