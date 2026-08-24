# AMReX Diagnostics Contract

Diagnostics are scalar reductions and timing records. They do not write AMReX
plotfiles and they do not manage checkpoints.

## Scalar TSV

Each run writes one tab-separated scalar file:

```text
runs/current/<ic_name>/<run_name>/scalars.tsv
```

The first row is the exact applicable runtime header generated from
`codegen/pyfluid_codegen/diagnostic_catalog.py`. The generated C++ view is
`generated/diagnostic_catalog.hpp`; plotting and reports read the matching
generated JSON metadata. Each entry carries a logical family, scalar/vector/
tensor kind, component index, requirements, and mathematical label. Vector and
tensor components remain ordinary columns in this one file; there is no second
history file to synchronize. Selecting a family such as
`adm_momentum_surface` enables every component. The schema includes compact
columns for:

- step, time, and dt;
- conserved totals and relative drifts;
- physical-state floors;
- MHD divergence and diffusion residuals when active;
- AMR hierarchy health;
- active-cell throughput and simulation-time throughput.

Rules:

- The selected groups are filtered by generated applicability requirements.
  Hydro runs therefore omit MHD-only columns, runs without a spacetime omit
  Z4c constraints, and inactive diffusion/passive/gravity columns are not
  emitted as misleading zeros. Cowling runs retain constraints because a
  fixed metric and its matter can still be constraint-consistent or violating.
- `pyfluid.scalar_int` controls runtime scalar cadence. The default is
  `1`, so every accepted coarse step writes a scalar row. `0` writes the
  initial row only. `-1` currently still creates `scalars.tsv` with the
  initial accepted state, but writes no later cadence rows.
- Every AMReX/MPI reduction must be called by every rank in the same order.
  File writes are rank-0 only, but scalar computation itself is collective.
- `divb_l1`, `divb_l2`, and `divb_linf` use the same discrete divergence
  implied by the CT representation.
- `newtonian_magnetic_energy` is the Newtonian integral of `B^2/2`. RMHD does
  not reuse that name or formula: it emits the generated Eulerian
  electromagnetic energy, comoving `b_squared`, enthalpy magnetization,
  plasma beta, and Lorentz-factor extrema.
- `mcells_per_s` is average cell-update throughput since the previous
  diagnostic sample.
- `passive_mass` and `passive_mass_rel_drift` are active when
  `pyfluid.passive_scalars > 0`; passive tracers are conserved quantities in
  closed, non-diffusive domains.
- `max_rho` is the maximum primitive rest-mass density for RHD/RMHD and the
  maximum conserved density for Newtonian HD/MHD. `max_rho_rel_change` measures
  it against the initial or checkpoint-persisted reference, making stellar
  equilibrium drift and oscillation amplitude visible without postprocessing.
- `min_lapse` and `relativistic_atmosphere_volume_fraction` are enabled by the
  standard group when applicable. The latter is the composite-AMR physical
  volume at the configured atmosphere divided by total composite volume; it is
  an accepted-state health diagnostic, not a C2P fallback counter.
- `sim_time_per_s` is average simulation time advanced per wall-clock second
  since the previous diagnostic sample. This is the main cost metric for
  comparing AMR/subcycling/diffusion choices at fixed physical target time.
- `amr_active_finest_level`, `amr_finest_headroom`,
  `amr_total_budget_fraction`, and `amr_peak_level_budget_fraction` mirror the
  compact hierarchy-health fields from `amr_levels.tsv` so ordinary scalar
  reports immediately show whether a run is max-level or budget limited.
- The opt-in `global_gr` group evaluates ADM mass, linear momentum, and angular momentum in
  surface and constraint-assisted volume forms, proper matter energy, Komar
  surface and stationary volume masses, and baryonic-minus-ADM binding energy.
  These use the shared hierarchy reduction executor and generated integrands.
  The ADM volume expressions use the constraints to remove curvature
  divergences; they are not naive divergence-theorem copies of the surface
  quadrature.
- The high-level formulas in
  `codegen/pyfluid_codegen/diagnostic_equations.py` generate the point kernels
  and the Diagnostic Integrands sections of `docs/equations.md` and
  `docs/equations.tex`. Those records are the human-readable equation catalog;
  runtime code must not carry a second handwritten definition.
- General relativity has no local, gauge-invariant gravitational-energy
  density to add pointwise to the fluid energy. For an isolated asymptotically
  flat system, ADM mass is the global total energy at spatial infinity. The
  finite-radius radiation ledger below combines the ADM surface estimate with
  matter and gravitational-wave fluxes through the same extraction sphere; it
  is a convergence diagnostic, not an exact Bondi balance at finite radius.
- Diagnostic integrands are defined once in a physical tensor basis and must
  not be expanded separately for Cartesian, cylindrical, or spherical meshes.
  Coordinate support belongs to the quadrature executors. A volume executor
  supplies the physical point, mapped cell measure, and composite-AMR ownership
  mask. A surface executor supplies a parametric physical surface, its tangents,
  oriented area element, and interpolation of the requested fields from the
  hierarchy. Coordinate singularities are handled by the map and measure, not
  by diagnostic-specific branches.
- The existing Newtonian volume reductions already follow this geometry-aware
  contract. `global_gr` now builds one interior spherical surface using
  Gauss-Legendre polar quadrature and uniform azimuthal quadrature, maps every
  physical point into the active mesh coordinates, selects the finest complete
  AMR interpolation cell, and evaluates all requested generated flux bundles
  through one nodal hierarchy plan. The same surface contraction accepts one
  physical flux vector per scalar charge or one flux tensor per vector charge,
  so ADM mass, Komar mass, and ADM angular momentum do not own separate
  quadrature loops. `diagnostics.surface_radius < 0` selects an
  automatically inscribed Cartesian sphere; `diagnostics.surface_center1/2/3`,
  `diagnostics.surface_radius`, and `diagnostics.surface_polar_points` select an
  explicit surface. The geometry/interpolation service supports single-patch
  Cartesian, cylindrical, and spherical coordinate maps. Relativistic
  integrands remain restricted to Cartesian evolution until the reference-metric
  curvilinear Z4c formulation is implemented; the integrands are not duplicated.
- Cartesian `symmetry` boundaries reconstruct omitted half, quadrant, or octant
  sectors for volume diagnostics using field parity. Scalar charges receive one
  factor of two per reduced axis; polar vectors and axial vectors use their
  respective reflection parity. Interior spherical extraction is intentionally
  unavailable on a reduced domain until the surface sampler can reconstruct
  every omitted sector with the same catalog parity metadata.
- The diagnostic cadence should default to a small, cheap number for validation
  and be configurable for production runs.

## Gravitational Radiation

`diagnostics.multipole_l_max = 0` disables radiation extraction. Values from 2
through 16 enable one extraction at the scalar cadence. Real reductions
(`extraction_radius`, `matter_energy_flux`, and `psi4_l2`) use the normal
`scalars.tsv` lifecycle, while complex mode coefficients use:

```text
multipoles.tsv
```

The extraction uses the same `ParametricSurface` and finest-complete AMR
interpolation plan as the ADM surface diagnostics. The runtime first computes
the regular electric and magnetic Weyl tensors on the nodal hierarchy,
interpolates those tensors and the physical metric to Gauss-sphere points, and
only then constructs the angular tetrad and contracts `Psi4`. This ordering
avoids introducing a coordinate-axis tetrad singularity into otherwise regular
grid data.

The generated convention is

```text
Psi4 = (E_ij - i B_ij) mbar^i mbar^j,
mbar = (e_theta - i e_phi) / sqrt(2).
```

The `(e_r, e_theta, e_phi)` frame is orthonormalized with the interpolated
physical spatial metric and outward coordinate radial direction. Mode rows are
the coefficients of spin-weight `-2` harmonics and are stored in generalized
long form with explicit `field`, `basis`, `spin_weight`, `ell`, and `m` columns.
No mode set is hardcoded into the scalar schema.

## Azimuthal Surface Modes

`diagnostics.azimuthal_mode_fields` accepts a comma-separated set of scalar
fields and `diagnostics.azimuthal_mode_m_max` selects the largest nonnegative
mode. `none` disables the diagnostic. `rho` and `pressure` select physical
primitive fields consistently for HD, MHD, RHD, and RMHD; scalar stored fluid
and spacetime GFs may be selected by their field-catalog names. At the same
surface and scalar cadence, the runtime writes

```text
C_m[f](r) = integral over S_r of f exp(-i m phi) dA
```

as `basis=azimuthal_fourier_exp_minus_imphi` rows in `multipoles.tsv`. Only
`m >= 0` is stored
because real fields satisfy `C_-m = conjugate(C_m)`. Runtime stores the raw
complex coefficients, including `C_0`; normalized amplitudes `|C_m|/|C_0|`,
phases, growth rates, and pattern speeds are postprocessed quantities. This is
a shell diagnostic, distinct from the related volume-integrated density-mode
diagnostic used for global bar and disk instabilities. The common long-form
schema records `ell=none` for these azimuthal Fourier rows because `ell` applies
only to spherical-harmonic modes.

## Global Matter Modes

`diagnostics.matter_mode_m_max` enables the related volume diagnostic; `-1`
disables it. The configured origin and axis are

```text
diagnostics.matter_mode_origin1/2/3
diagnostics.matter_mode_axis1/2/3
```

The runtime computes

```text
C_m = integral D exp(+i m phi) d^3x,
D = sqrt(gamma) rho W = sqrt(-g) rho u^0,
```

over composite valid AMR cells and excludes runtime atmosphere. Rows are
written to `multipoles.tsv` with
`basis=azimuthal_fourier_exp_plus_imphi`, `field=rest_mass`, and no extraction
radius. `C_0` is therefore the atmosphere-excluded baryon mass used to
normalize `|C_m|/C_0`; the same reduction also writes `matter_mode_mass` and
the three matter-barycenter columns to `scalars.tsv`. Surface and volume modes
use opposite documented exponential signs and are not interchangeable.

A single reflection plane is supported only when it is the equatorial plane
through the configured origin and normal to the mode axis. The missing half is
then reconstructed by its factor of two. Additional reflection planes would
constrain nonaxisymmetric modes and are rejected instead of silently reporting
partial-domain amplitudes. The focused full-domain/equatorial and three-rank
control is

```bash
uv run python pyfluid_amrex/tools/validate_matter_modes.py --force --ranks 3
```

Run

```text
uv run python pyfluid_amrex/tools/postprocess_radiation.py <run_dir>
```

to produce `radiation_modes_postprocess.tsv` and `radiation_balance.tsv` in the
same run directory. Normal `run_case.py` workflows invoke this postprocessor
automatically through the existing lightweight physics-hook layer. The
postprocessor uses
`Psi4 = d^2(h_plus - i h_cross)/dt^2`, integrates by the trapezoidal rule with
news and strain set to zero at the first available sample, and computes

```text
L_GW = r^2/(16 pi) sum_lm |N_lm|^2.
```

Those finite-time integration constants are not drift removal. Production
waveforms may need fixed-frequency integration or an independently justified
early-time baseline. The finite-radius balance is emitted only when
`adm_mass_surface` exists in `scalars.tsv`; missing ADM or flux terms are left
blank rather than replaced by zero. Extraction-radius convergence and gauge
dependence must be assessed before interpreting the result as radiation at
null infinity.

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

- `amr_tag_metric` is a catalog-declared pure auxiliary GF and evaluates the
  currently selected runtime policy. It is not a hidden second tagger. The
  level-indexed hierarchy columns below remain in `amr_levels.tsv` because
  their shape depends on the runtime hierarchy rather than a static GF/scalar
  component catalog.
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
  are `j`, `e`, `divb`, `magnetic_energy`, `mach`, `plasma_beta`,
  `thermal_entropy_proxy`,
  `amr_tag_metric`, `solid_volume_fraction`, `solid_swept_volume`, and
  `solid_phi`. By default, aux output inherits `pyfluid.plot_int`, and live
  cached derivatives such as `j`/`e` use that same resolved cadence.
  `pyfluid.aux_gf_int` and `pyfluid.aux_gf_compute_int` remain explicit
  overrides; explicitly setting either to `-1` disables that path. Compute-only
  aux requests are valid only for cached derivatives.
- Pure diagnostic GFs such as `mach`, `plasma_beta`,
  `thermal_entropy_proxy`, `vorticity`,
  `magnetic_energy`, and `amr_tag_metric` are never stored as solver state.
  They are computed only at aux-output cadence when requested.
- `thermal_entropy_proxy` is `p/rho^Gamma` for an ideal gas. For a hybrid
  piecewise-polytropic EoS it is
  `(p - p_cold(rho))/rho^Gamma_th`, so changes in cold-polytrope segment do not
  masquerade as entropy production. A strictly cold barotrope returns zero
  because entropy is not an independent variable in that model. This is an
  adiabatic/thermal proxy, not absolute physical entropy; a tabulated EoS must
  eventually supply its own thermodynamic entropy.
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

Every diagnostic has one declared evaluation mode in
`codegen/pyfluid_codegen/diagnostic_catalog.py`:

- `runtime_reduction`: an MPI reduction written by the solver to `scalars.tsv`;
- `field_output`: a stored or requested GF written through the canonical field
  schema;
- `postprocess`: a derived quantity computed from named GF dependencies after
  the run.

Do not recompute runtime conservation totals from plot data, and do not add a
second plotting-only name or label table. A diagnostic that can be evaluated
either from live storage or at output time remains one field descriptor with a
mixed diagnostic kind; the storage choice is not a second field.

All 2D snapshots, movie frames, and analysis samplers use
`tools/amr_plot_data.py` as the hierarchy source of truth. Native tile rendering
is the default for movies: it draws original cells in coarse-to-fine order and
does not invent resolution. Uniform views are explicit presentation products;
they may use nearest-neighbor, linear, or area-weighted tile sampling, with the
finest available level taking precedence. HDF5 crop requests read only boxes
that intersect the requested physical region. Do not use an interpolated
uniform view as a replacement for native data in conservation diagnostics.

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
