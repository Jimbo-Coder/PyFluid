# PyFluid AMReX Reference Router

Use this file to find the authoritative source quickly after a context reset.
It intentionally links rather than restating the detailed contracts.

## Required Entry Sequence

1. Read repository instructions in `../AGENTS.md`.
2. Read the maintained state snapshot in `knowledge.md`.
3. Read at least the last active/completed cycles in
   `review_running_cycle.md`; append one cycle for the current request.
4. Read `docs/code_conventions.md` before implementation.
5. Route to the task-specific source and document below. Verify current source
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

Treat AthenaK output as an independent numerical reference. A useful paired
comparison should:

1. write a short equivalence declaration for equations, normalization, EoS,
   primitive/conserved conventions, domain, boundary conditions, IC, and final
   time;
2. match numerical choices where possible, but report irreducible differences
   such as block AMR versus AMReX, prolongation/reflux policy, CT topology, and
   floor/recovery rules;
3. run uniform-grid controls first, then matched refinement studies;
4. resample both outputs to common physical points or compare scalar/integral
   diagnostics rather than raw storage ordering;
5. report each code's conservation, constraints/divB, errors, wall time, and
   resolution separately; and
6. store the exact AthenaK input, stdout, build/provenance record, and comparison
   products under `runs/current/comparisons/athenak/<case>/` rather than relying
   on an ephemeral terminal invocation.

AthenaK's fast compile/run loop is useful for generating an independent
high-resolution numerical reference, but it does not replace PyFluid's own
analytic, conservation, restart, AMR, and performance gates.

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

AsterX is therefore not evidence that raw MP5 should run without safeguards.
It is evidence for preserving MP5 as the primary component reconstructor while
making any low-order substitution local and explicit. AthenaK's stage-candidate
FOFC is the more directly reusable design for preventing an accepted update
from reaching C2P in a state that requires repair.
