# PyFluid AMReX

This is PyFluid's production Newtonian HD/MHD backend. Python/JAX and Numba
remain reference baselines. AMReX owns parallel mesh management, AMR, MPI,
plotfiles, and checkpoints; PyFluid owns the physical state, numerical methods,
diagnostics, and validation.

## Normal Workflow

Configure once, then build the dimension being changed. The full preset keeps
every supported method available at runtime.

```bash
cd pyfluid_amrex
cmake --preset full-release
cmake --build --preset full-2d
```

Run an input through the standard wrapper:

```bash
uv run python pyfluid_amrex/tools/run_case.py \
  pyfluid_amrex/inputs/hydro_sine_2d.in
```

`run_case.py` infers the executable dimension, uses three MPI ranks by default,
replaces the selected run directory, writes `run.log`, validates
the scalar budget, and generates the scalar report. Add `--gf-plots` only when
a state or auxiliary snapshot is wanted. Raw `mpirun` is for focused low-level
debugging because it bypasses automatic reporting. The C++ executable always
writes the canonical merged AMReX parameter table to `<run_dir>/run.in`, so
raw and wrapped runs retain the same configuration record.

Every input file is a complete run configuration and explicitly names its run
and output directory. Supported input values replace registered C++ defaults;
there is no second launcher-level simulation-parameter interface. Registered method and AMR defaults live in
[`config.cpp`](src/config.cpp); structural parsing and combination validation
live in [`config.hpp`](include/pyfluid_amrex/config.hpp). Grid-function names,
components, centering, and output roles live in
[`fields.hpp`](include/pyfluid_amrex/fields.hpp).

## Supported Numerical Surface

The regular production path is single-patch Cartesian HD or MHD in native 1D,
2D, or 3D. It provides hydro-only state allocation for hydro runs, primitive
PWC/PWL/WENO-Z reconstruction, LLF/HLL/HLLE/HLLC/HLLD Riemann solvers,
Euler/midpoint/SSPRK3/RK4 time integration, adaptive CFL stepping, AMR,
reflux/average-down, checkpoint/restart, and optional AMR subcycling.

CT MHD uses cell-centered `B` for reconstruction/Riemann solves plus active
face-centered `B` for the constrained update. Explicit viscosity and CT
resistivity use the AMR flux/EMF path. Hydro implicit viscosity and the
validated 2D periodic implicit CT-resistivity path are available; 3D multilevel
implicit CT resistivity is rejected until a composite edge solve exists.

These paths remain explicitly experimental or limited:

- Cylindrical and spherical single-patch coordinates: validate the exact
  coordinate/method combination before treating it as a production result.
- Homogeneous Cartesian multipatch: hydro finite-volume ownership and AMR are
  supported; overlap is ghost/interpolation context, not duplicated evolution.
- Mixed Cartesian-curvilinear shell layouts: mapping and sampling are present,
  but conservative finite-volume evolution across nonaligned seams is not yet
  production-supported.
- Gravity and moving solids: use their focused validation gates; do not mix
  them with experimental coordinate/multipatch work without a new validation.

Unsupported combinations abort during configuration. They never silently fall
back to a different method.

## Configuration Rules

- `geometry.*` and `amr.*` describe the AMReX domain and hierarchy.
- `mesh.*` selects coordinate family, active axes, and patch topology.
- `pyfluid.*` selects physics, initial data, numerical method, boundaries,
  sources, output cadence, and diagnostics.
- Canonical axis orders are Cartesian `x y z`, cylindrical `r phi z`, and
  spherical `r theta phi`. Lower-dimensional `mesh.axes` is an ordered subset
  of its canonical 3D order.
- Vector parameters and state components use runtime directions: `v1`, `v2`,
  `v3`, `b1`, `b2`, `b3`, `mom1`, `mom2`, `mom3`.
- Normal runs use `pyfluid.cfl = 0.9`, `pyfluid.physical_check_int = 0`, and
  three MPI ranks. New setups use `pyfluid.physical_check_int = 1` until their
  scalar and physics checks pass.
- `pyfluid.run_name` and `pyfluid.output_dir` are explicit in every checked-in
  run input. `run_case.py` does not infer, alias, or override simulation
  parameters.
- The production executable accepts exactly one input file. Trailing
  `key=value` command-line mutations are unsupported.
- `amrex.abort_on_unused_inputs = 1` is mandatory, so misspelled or
  unsupported input keys fail instead of being silently ignored.
- AMR should stop because the physics metric is smooth enough. `max_level`,
  `min_feature_length`, and cell budgets are safety limits, not physics knobs.
  The default automatic tag threshold is independent of `max_level`. Passive
  scalars never tag AMR. A production-style run starts from a deliberately
  coarse base grid and lets the metric create and move resolution. Reaching a
  safety limit means the run is under-resolved or the metric is misconfigured;
  it is not a successful refinement result. On available hardware, production
  setups commonly allow roughly 10-15 levels as headroom while cell budgets
  remain the hard memory guard; the flow metric should select far fewer levels.

## Output Contract

Each ordinary run uses one directory:

```text
runs/current/<ic_name>/<run_name>/
  run.in
  run.log
  scalars.tsv
  amr_levels.tsv                 # AMR runs
  stateGF/stateGF00000.h5        # default; native AMReX is optional
  auxGF/auxGF00000.h5
  checkpoints/chk00000/
  report.tsv
  <run_name>_scalars.png
```

`stateGF/` and `auxGF/` are the only plotfile roots. Compressed parallel HDF5 is
the default; native AMReX plotfiles remain available with
`pyfluid.plotfile_format=amrex`. Scalars are C++
runtime reductions; Python postprocessing reads them and does not replace them
with a second total calculation. See [`docs/diagnostics.md`](docs/diagnostics.md)
for the field and scalar contract.

Checkpoints use readable text `Header`/`Header.meta` files plus losslessly
compressed multilevel HDF5 field files. Restart requires the current metadata
schema and matching state interpretation: dimension, domain geometry,
coordinate axes, active refinement ratios, physics/EOS parameters, CT layout,
and passive-component count. Run-control choices such as CFL, reconstruction,
Riemann solver, timestepper, subcycling, diffusion/source settings, output
cadence, diagnostics, and a larger `max_level` may change after restart.

## Validation And Performance

```bash
# quick confidence gate
uv run python pyfluid_amrex/tools/validate_amrex_suite.py --level quick

# standard gate after solver or infrastructure changes
uv run python pyfluid_amrex/tools/validate_amrex_suite.py --level standard

# performance comparison after a hot-path change
uv run python pyfluid_amrex/tools/benchmark_pyfluid.py
```

The suite writes compact TSV history under `runs/current/`; ordinary temporary
outputs are overwritten. Preserve a known-good state with:

```bash
uv run python pyfluid_amrex/tools/capture_baseline_run.py --label <label>
```

Run-specific postprocessing stays outside evolution. Generic reporting is
[`report_run.py`](tools/report_run.py); exact overlays, spectra, movies, and
specialized physical checks are thin scripts that consume the same run folder.

## Repository Map

- [`include/pyfluid_amrex/`](include/pyfluid_amrex): solver, physics, field
  schema, and AMR interfaces.
- [`src/`](src): AMReX hierarchy, output, diagnostics, checkpoints, and
  multipatch orchestration.
- [`codegen/`](codegen): SymPy sources and generated kernels.
- [`inputs/`](inputs): reproducible named configurations.
- [`tools/`](tools): shared run, report, benchmark, validation, plotting, and
  comparison commands.
- [`docs/design.md`](docs/design.md): developer architecture and limitations.
- [`docs/install.md`](docs/install.md): local AMReX build notes.

Active work is tracked in
[`review_running_cycle.md`](review_running_cycle.md). Append one cycle at a
time and verify a subtask before marking it done.
