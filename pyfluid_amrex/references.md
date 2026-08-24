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
| CPU tiled ghost ownership and device scratch lifetime | `include/pyfluid_amrex/{boundary,relativistic_boundary,ct}.hpp`, `src/relativity/relativistic_atmosphere.hpp`, `src/amr/state_interpolation.cpp`, `tests/boundary_tiling_check.cpp`, `tools/validate_boundary_tiling.py` | `docs/infrastructure_conventions.md`, Cycle 284 |
| Coding and review policy | current source | `docs/code_conventions.md` |
| Runtime configuration | `include/pyfluid_amrex/config.hpp`, `src/framework/config*.{cpp,hpp}` | `README.md`, configuration rules |
| Grid fields, components, centerings | `schema/fields.toml`, loaded by `codegen/pyfluid_codegen/field_catalog.py` | `docs/catalog_schema.md`, generated field sections in `docs/equations.md` |
| Reconstruction, rotation-law, and AMR-policy identity | `schema/methods.toml`, loaded by `codegen/pyfluid_codegen/method_catalog.py`; formulas under `include/pyfluid_amrex/reconstruction/`, AMR evaluation in `include/pyfluid_amrex/amr_indicator.hpp`, and shared circular ID policy under `include/pyfluid_amrex/initial_data/circular_rotation.hpp` | `docs/catalog_schema.md`, `docs/design.md`, `docs/tov.md` |
| Diagnostics and reductions | `codegen/pyfluid_codegen/diagnostic_catalog.py`, `src/amr/amr_solver_diagnostics.cpp` | `docs/diagnostics.md` |
| Generated equations | labeled symbolic sources under `codegen/pyfluid_codegen/` | `docs/equations.md`, `docs/equations.tex` |
| Relativistic HD/MHD and Z4c | generated kernels, `src/relativity/{rhd_system,rmhd_system,relativistic_atmosphere}.hpp`, `include/pyfluid_amrex/relativistic_boundary.hpp`, and `src/amr/amr_solver_spacetime*.cpp` | `docs/grmhd_development.md` |
| Runtime EoS and relativistic C2P | `include/pyfluid_amrex/eos/equation_of_state.hpp`, `include/pyfluid_amrex/{rhd_c2p,rmhd_c2p}.hpp`, generated `relativistic/c2p_*.hpp`, `tests/{generated_flux_compile_check,tabulated_eos_check}.cpp` | RePrimAnd, Spritz, and recovery-comparison references below; `docs/grmhd_development.md` |
| RMHD HLLC/HLLD fan and closure | `include/pyfluid_amrex/{rmhd_riemann,rmhd_advanced_flux}.hpp`, `src/relativity/rmhd_advanced_flux.cpp`, `tests/rmhd_riemann_check.cpp`, and `tools/validate_rmhd_advanced_riemann.py` | Mignone--Bodo and Mignone--Ugliano--Bodo references below; Cycles 262 and 284 |
| CT stage and AMR ownership | `src/amr/solver_sections/{ct_evolution,time_integrators,evolve_core,subcycling}.inc.hpp`, `src/numerics/ct_edge_flux_register.cpp`, and `tools/validate_ct_divb.py` | `docs/design.md`, current CT boundary in `knowledge.md` |
| Exact SRMHD Riemann profiles | tracked `../exact_balsara/solution_b{1..5}.dat`, read by `tools/validate_rmhd.py` | Giacomazzo--Rezzolla exact solver and paper below |
| SRMHD symptom onset, hybrid attribution, and prototype applicability | `tools/diagnose_rmhd_balsara_onset.py`, `tools/audit_srmhd_physical_wave_balsara.py`, their TSVs under `runs/current/rmhd_validation/`, and `plots/current/scans/rmhd_balsara*_profiles.png` plus `rmhd_balsara_hybrid_timelines.png` and `rmhd_balsara_hybrid_profile_evolution.png` | `docs/unified_shock_capturing_research.md`, Cycles 252--258 |
| Coherent space-time operator attribution | `tools/research_spacetime_operator_ablation.py`, `tools/research_mhd_spacetime_operator_ablation.py`, and `tools/research_srmhd_spacetime_operator_ablation.py` | `docs/unified_shock_capturing_research.md`, Cycles 273--274 |
| TOV/EoS/stellar initial data | `include/pyfluid_amrex/eos/`, `src/eos/`, `src/initial_data/`, `schema/methods.toml` | `docs/tov.md` |
| Magnetized stellar initial data | `include/pyfluid_amrex/initial_data/magnetic.hpp`, `src/initial_data/initial_data_magnetic.cpp`, `src/initial_data/rotating_star_initial_data.cpp`, `schema/methods.toml`, `tools/validate_magnetized_initial_data.py`, `tools/validate_magnetohydrostatic_initial_data.py` | Pili--Bucciantini--Del Zanna below; `docs/infrastructure_conventions.md`; Cycles 308--309 |
| Native cold CompOSE tables | `eos/compose/`, `src/eos/tabulated.cpp`, `include/pyfluid_amrex/eos/tabulated*.hpp`, `tests/tabulated_eos_check.cpp`, `tools/validate_compose_eos.py` | CompOSE manual and dataset records below; Cycle 297 |
| Equilibrium torus/disk initial data | `include/pyfluid_amrex/initial_data/equilibrium_torus.hpp`, `src/initial_data/equilibrium_torus_initial_data.cpp`, `schema/methods.toml`, `tools/validate_equilibrium_torus.py` | `docs/tov.md`, `docs/infrastructure_conventions.md`, Cycle 296 |
| Kerr black-hole initial data and interior policy | `include/pyfluid_amrex/initial_data/kerr_schild.hpp`, `src/initial_data/initial_data_provider.cpp`, `tools/validate_black_hole.py` | `docs/black_holes.md`, Cycles 299--300 |
| Apparent-horizon geometry, finder, tracking, and masks | `include/pyfluid_amrex/apparent_horizon.hpp`, `include/pyfluid_amrex/apparent_horizon_finder.hpp`, `src/relativity/apparent_horizon_finder.cpp`, `src/relativity/apparent_horizon_tracker.cpp`, `src/amr/amr_solver_horizon.cpp` | `docs/black_holes.md`, Cycles 300--301 |
| Portable physical-ID solve/save/materialization, including explicit radial faces | `src/initial_data/physical_id.cpp`, `src/initial_data/physical_id_interpolation.cpp`, `src/initial_data/initial_data_elliptic.cpp`, `src/framework/checkpoint.cpp`, `tests/physical_id_interpolation_check.cpp`, `tools/validate_reduced_initial_data.py` | `docs/infrastructure_conventions.md`, Cycles 287 and 291 |
| Axisymmetric spherical ID scalar/toroidal elliptic operators | `src/initial_data/initial_data_elliptic.cpp`, `include/pyfluid_amrex/initial_data/elliptic.hpp`, `tests/initial_data_elliptic_check.cpp` | `docs/infrastructure_conventions.md`, Cycles 288--289 |
| Infrastructure and artifact ownership | current implementation and field schema | `docs/infrastructure_conventions.md` |
| Adaptive Richardson status | AMR estimator and validation tools | `docs/adaptive_richardson.md` |
| Machines, MPI, build | `machines/*.toml`, `tools/validation_common.py` | `docs/install.md`, `docs/machines.md` |
| External review triage | current source | `review_backlog.md` |
| Independent AthenaK comparison | `/Users/mrizzo/phys_research/athenak/athenak` | local `README.md`, test suite, and inputs |

Generated C++ and JSON files are derived views. Change the owning declarative
schema, strict catalog loader, or labeled equation source; regenerate and
verify consistency rather than hand-editing generated artifacts.

## Validation Ladder

Use the smallest gate that can falsify the change, then widen according to
risk. Normal commands use `UV_CACHE_DIR=.uv-cache` when the global uv cache is
not writable.

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

For AMR interpolation call-path comparisons, the installed AMReX 25.11 headers
are under `.deps/amrex-mpi-lsolvers-hdf5/include/`; route first to
`AMReX_FillPatchUtil_I.H`, `AMReX_Interpolater.H`, and
`AMReX_Interp_1D_C.H`. AthenaK's corresponding componentwise prolongation is
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

## Relativistic AMR Policy References

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

- G. B. Cook and H. P. Pfeiffer, *Excision boundary conditions for black-hole
  initial data*, [arXiv:gr-qc/0407078](https://arxiv.org/abs/gr-qc/0407078):
  quasi-equilibrium apparent-horizon inner boundary conditions for CTS/XCTS.
  This is the source for PyFluid's constraint-solved Schwarzschild shell and
  planned spinning extension; its initial-data horizon boundary is not
  interchangeable with an evolution excision boundary.
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
  over-relaxation, and threaded surface finding. This is the method reference
  for PyFluid's independent first full finder; no implementation code is
  imported.
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
