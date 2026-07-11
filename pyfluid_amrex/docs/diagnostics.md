# AMReX Diagnostics Contract

Diagnostics are scalar reductions and timing records. They do not write AMReX
plotfiles and they do not manage checkpoints.

## Scalar TSV

Each run writes one tab-separated scalar file:

```text
runs/current/<ic_name>/<run_name>/scalars.tsv
```

The first row is the exact runtime header written from
`ScalarDiagnosticColumns` in `diagnostics.hpp`. It includes compact columns for:

- step, time, and dt;
- conserved totals and relative drifts;
- physical-state floors;
- MHD divergence and diffusion residuals when active;
- AMR hierarchy health;
- active-cell throughput and simulation-time throughput.

Rules:

- The current implementation writes a fixed schema. Hydro-only runs write zero
  magnetic/diffusion values.
- `pyfluid.scalar_int` controls runtime scalar cadence. The default is
  `1`, so every accepted coarse step writes a scalar row. `0` writes the
  initial row only. `-1` currently still creates `scalars.tsv` with the
  initial accepted state, but writes no later cadence rows.
- Every AMReX/MPI reduction must be called by every rank in the same order.
  File writes are rank-0 only, but scalar computation itself is collective.
- `divb_l1` must use the same discrete divergence implied by the CT
  representation.
- `mcells_per_s` is average cell-update throughput since the previous
  diagnostic sample.
- `passive_mass` and `passive_mass_rel_drift` are active when
  `pyfluid.passive_scalars > 0`; passive tracers are conserved quantities in
  closed, non-diffusive domains.
- `sim_time_per_s` is average simulation time advanced per wall-clock second
  since the previous diagnostic sample. This is the main cost metric for
  comparing AMR/subcycling/diffusion choices at fixed physical target time.
- `amr_active_finest_level`, `amr_finest_headroom`,
  `amr_total_budget_fraction`, and `amr_peak_level_budget_fraction` mirror the
  compact hierarchy-health fields from `amr_levels.tsv` so ordinary scalar
  reports immediately show whether a run is max-level or budget limited.
- The diagnostic cadence should default to a small, cheap number for validation
  and be configurable for production runs.

## AMR Level TSV

AMR runs also write one tab-separated hierarchy-health file:

```text
runs/current/<ic_name>/<run_name>/amr_levels.tsv
```

Important columns:

```text
active_finest_level
max_level
finest_headroom
touches_max_level
total_budget_fraction
peak_level_budget_fraction
cells_L*
steps_L*
latest_auto_dir*_mean_over_threshold_L*
refinement_decision_L*
```

Rules:

- `touches_max_level = 1` means the run may be limited by `amr.max_level`
  rather than the refinement metric.
- Budget fractions near `1` mean the hierarchy is resource-limited.
- `latest_auto_dir*_mean_over_threshold_L*` reports directional tagging demand
  normalized by `amr.tag_threshold`; values above one requested refinement.
- `refinement_decision_L*` summarizes each active level as `smooth`, `refined`,
  `physical_length_guard`, `max_level_guard`, or `tagged_no_fine_level`.
  `tagging_off` and `inactive` are explicit rather than inferred from zeros.
- Cell budgets are safety checks that abort when exceeded; they are not
  mislabeled as graceful refinement decisions.
- Healthy production AMR should usually have positive headroom and budget
  fractions safely below `1`, unless a deliberate stress test says otherwise.
- The expected terminal decision on every active finest region is `smooth`.
  `physical_length_guard`, `max_level_guard`, a radial guard, or a cell-budget
  abort is a resource-limited result and must not be reported as converged AMR.
- Representative feature-matrix AMR cases fail unless refinement activates and
  retains at least two levels of headroom. Uniform cases remain explicit
  controls for convergence, dimensional equivalence, CFL scans, and benchmarks;
  they do not count as AMR feature coverage.
- `pyfluid_amrex/tools/plot_amrex_amr_levels.py` turns this TSV into the
  standard AMR/subcycling diagnostic plot.

## Field Output

Grid-function output is optional and separate from diagnostics:

```text
runs/current/<ic_name>/<run_name>/stateGF/
```

Auxiliary outputs are written under `runs/current/<ic_name>/<run_name>/auxGF/`.

Field output should support 1D, 2D, and 3D runs. Write only active and useful
fields for the active dimension:

- Hydro: do not write magnetic fields.
- 1D: write line data or a 1D AMReX plotfile when supported by the build.
- 2D: write 2D fields only; do not require a fake one-cell z direction.
- 3D: write full 3D fields.
- MHD: include the variables needed to inspect the run and reconstruct derived
  quantities in postprocessing.
- `pyfluid.aux_gfs` selects optional auxiliary grid functions. Common choices
  are `j`, `e`, `divb`, `magnetic_energy`, `mach`, `plasma_beta`, `entropy`,
  `amr_tag_metric`, `solid_volume_fraction`, `solid_swept_volume`, and
  `solid_phi`. `pyfluid.aux_gf_compute_int` controls live cached derivatives
  such as `j`/`e`; `pyfluid.aux_gf_int` controls plotfile output. Compute-only
  aux requests are valid only for cached derivatives.
- Pure diagnostic GFs such as `mach`, `plasma_beta`, `entropy`, `vorticity`,
  `magnetic_energy`, and `amr_tag_metric` are never stored as solver state.
  They are computed only at aux-output cadence when requested.
- `solid_swept_volume` is positive where the analytic solid has consumed fluid
  volume since `pyfluid.solid_swept_reference_time`, and negative where it has
  uncovered fluid volume.
- Scalar output includes `angular_mom1`, `angular_mom2`, and
  `angular_mom3`, computed as `r x momentum` over valid fluid cells. These
  are the primary compact diagnostics for rotating-wall and future gear runs.
- `j` and `e` are cached derivative fields for nonzero resistivity or when
  `pyfluid.aux_gf_compute_int >= 0` explicitly requests a compute cadence. If
  output requests `j`/`e` without live storage, the plotfile writer computes
  temporary fields at output time.

Thin 3D output with one cell in z may be useful for tool compatibility, but it
should be an explicit output/run mode rather than the default 2D representation.

## Checkpoints

Checkpoints are separate from diagnostics and plotfiles:

```text
runs/current/<ic_name>/<run_name>/checkpoints/
```

The checkpoint cadence, restart file, and recovery behavior belong to the
checkpoint module, not the diagnostics module.

Checkpoint state and CT face fields use losslessly compressed multilevel HDF5;
`Header` and `Header.meta` remain plain text. Metadata is mandatory on restart.
It protects immutable state/layout semantics while leaving ordinary run-control
parameters free to change. The Python tools use the shared MPI launcher in
`validation_common.py`; rank-one runs execute directly and multi-rank runs use
the same `mpirun -np N` contract as `run_case.py`.

## Postprocessing

Python postprocessing reads `scalars.tsv` and optional AMReX plotfiles after the
run. Plots, exact overlays, and movies should not be required during the solver
step.

For ordinary ad-hoc runs, prefer the shared run wrapper:

```bash
uv run python pyfluid_amrex/tools/run_case.py pyfluid_amrex/inputs/hydro_sine_2d.in
```

It runs the AMReX executable, overwrites the configured run directory by
default, writes `run.log`, then calls the generic report and lightweight
physics hooks. The executable itself writes canonical `run.in`. The default run directory is
`runs/current/<ic_name>/<run_name>/`; each checked-in input sets its explicit
`pyfluid.output_dir`. The runner has no output-directory or simulation-parameter
override path. A raw executable launch still accepts exactly one input file;
using raw `mpirun` only bypasses automatic post-run validation.

For already completed runs, use the shared report command:

```bash
uv run python pyfluid_amrex/tools/report_run.py runs/current/<ic_name>/<run_name>
```

It applies the standard scalar sanity policy, prints one compact status line,
writes a TSV row, and generates the scalar report plot in the run directory.
Level-2 state/aux GF snapshot plots are opt-in with `--gf-plots`. Report plots
overwrite stable filenames by default; use `--numbered-plots` only for ad-hoc
comparisons or UI cache workarounds.

The open-boundary policy is not globally permissive for recognized localized
pressure perturbations. Before the numerical signal cone can reach an outflow
or diode boundary, mass, energy, and momentum must satisfy the closed-run
budget at `1e-12` on a uniform grid or `1e-11` on a multilevel hierarchy. The
report records the estimated contact time and number of
pre-contact samples. When state output exists, the lightweight physics hooks
also write `causal_quiescence.tsv` and check that density, pressure, velocity,
and magnetic field remain unchanged outside the cone. Gaussian bumps use an
eight-sigma numerical support because they are not mathematically compact;
finite-stencil exterior tails are checked at `1e-7`. Closed localized runs use
the same spatial hook while retaining their normal all-time conservation gate.

Ordinary runs should follow this lifecycle:

1. The AMReX executable writes `scalars.tsv`, optional state plotfiles, optional
   aux plotfiles, and optional checkpoints under one run directory.
2. `report_run.py` performs the common scalar validation and scalar plot for
   that run directory. Add `--gf-plots` when state/aux snapshots are wanted.
3. Lightweight physics hooks may add scalar-budget checks, such as moving-wall
   `PdV` work or gravity total energy, and write `physics_postprocess.tsv` under
   the run directory.
4. Heavier physics postprocessing, such as oblique-shock angle fits, Sod exact
   overlays, diffusion amplitude decay, KH spectra, convergence scans, and
   movies, stays in focused scripts. These scripts should call shared
   scalar/report helpers and should write their summaries under `runs/current/`
   and their plots under the same run directory.

Specialized validators should keep unique physics checks local but use the same
scalar/report helpers where possible. New ICs should not duplicate scalar
parsing or generic 1D/2D/3D plotting; add only the small physics-specific
postprocessing needed for that IC.

For KH runs, `pyfluid_amrex/tools/plot_kh_spectrum.py` samples the finest
available AMR data onto a regular diagnostic canvas and writes kinetic-energy
spectra, enstrophy spectra, vorticity measures, and scalar drift plots. The
summary TSV and plots stay in the run directory.
