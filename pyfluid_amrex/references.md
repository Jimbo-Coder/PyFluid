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
| Coding and review policy | current source | `docs/code_conventions.md` |
| Runtime configuration | `include/pyfluid_amrex/config.hpp`, `src/framework/config*.{cpp,hpp}` | `README.md`, configuration rules |
| Grid fields, components, centerings | `schema/fields.toml`, loaded by `codegen/pyfluid_codegen/field_catalog.py` | `docs/catalog_schema.md`, generated field sections in `docs/equations.md` |
| Rotation-law identity | `schema/methods.toml`, loaded by `codegen/pyfluid_codegen/method_catalog.py` | `docs/catalog_schema.md`, `docs/tov.md` |
| Diagnostics and reductions | `codegen/pyfluid_codegen/diagnostic_catalog.py`, `src/amr/amr_solver_diagnostics.cpp` | `docs/diagnostics.md` |
| Generated equations | labeled symbolic sources under `codegen/pyfluid_codegen/` | `docs/equations.md`, `docs/equations.tex` |
| Relativistic HD/MHD and Z4c | generated kernels, `src/relativity/{rhd_system,rmhd_system,relativistic_atmosphere}.hpp`, `include/pyfluid_amrex/relativistic_boundary.hpp`, and `src/amr/amr_solver_spacetime*.cpp` | `docs/grmhd_development.md` |
| TOV/EoS/stellar initial data | `include/pyfluid_amrex/eos/`, `src/eos/`, `src/initial_data/`, `schema/methods.toml` | `docs/tov.md` |
| Infrastructure and artifact ownership | current implementation and field schema | `docs/infrastructure_conventions.md` |
| Adaptive Richardson status | AMR estimator and validation tools | `docs/adaptive_richardson.md` |
| Machines, MPI, build | `machines/*.toml`, `tools/validation_common.py` | `docs/install.md`, `docs/machines.md` |
| External review triage | current source | `review_backlog.md` |
| Independent AthenaK comparison | `/Users/mrizzo/phys_research/athenak/athenak` | local `README.md`, test suite, and inputs |

Generated C++ and JSON files are derived views. Change their current Python
catalog, later declarative schema, or equation source; regenerate and verify
consistency rather than hand-editing generated artifacts.

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

## Rotating-Star Physics References

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
- J. S. Read et al., *Constraints on a phenomenologically parameterized
  neutron-star equation of state*,
  [arXiv:0812.2163](https://arxiv.org/abs/0812.2163): the piecewise-polytrope
  parametrization and fitted EoS families used by the local SLy/APR4/H4/MS1
  presets.

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
- Z4c, constraints, gauge, AMR, Weyl/wave extraction, and punctures:
  `src/z4c/`, `inputs/z4c/`, and `tst/test_suite/z4c/`;
- shared numerical references: linear waves, shock tubes, field loops,
  Orszag-Tang, diffusion, Bondi/monopole, and TOV utilities under `inputs/`,
  `tst/test_suite/`, and `src/utils/tov/`.

## Local AsterX Robust-Reconstruction Reference

The local AsterX checkout is under
`/Users/mrizzo/phys_research/ETK_CarpetX/CactusX/repos/AsterX`. For the narrow
reconstruction question, use:

- `AsterX/src/fluxes.cxx`: component reconstruction, selectable `z_vec = Wv`
  reconstruction, face-metric conversion, and per-face-side low-order fallback
  when reconstructed thermodynamic variables are nonpositive;
- `ReconX/src/mp5.hxx`: the MP5 formula;
- `ReconX/param.ccl`: independent primary and low-order method parameters;
- `AsterX/src/con2prim.cxx` and `Con2PrimFactory/src/`: atmosphere, velocity
  ceiling, and ordered recovery behavior.
