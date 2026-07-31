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
| Reconstruction and rotation-law identity | `schema/methods.toml`, loaded by `codegen/pyfluid_codegen/method_catalog.py`; formulas under `include/pyfluid_amrex/reconstruction/` | `docs/catalog_schema.md`, `docs/tov.md` |
| Diagnostics and reductions | `codegen/pyfluid_codegen/diagnostic_catalog.py`, `src/amr/amr_solver_diagnostics.cpp` | `docs/diagnostics.md` |
| Generated equations | labeled symbolic sources under `codegen/pyfluid_codegen/` | `docs/equations.md`, `docs/equations.tex` |
| Relativistic HD/MHD and Z4c | generated kernels, `src/relativity/{rhd_system,rmhd_system,relativistic_atmosphere}.hpp`, `include/pyfluid_amrex/relativistic_boundary.hpp`, and `src/amr/amr_solver_spacetime*.cpp` | `docs/grmhd_development.md` |
| Exact SRMHD Riemann profiles | tracked `../exact_balsara/solution_b{1..5}.dat`, read by `tools/validate_rmhd.py` | Giacomazzo--Rezzolla exact solver and paper below |
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

## Rotating-Star Numerical Method References

- XNS numerical approach,
  [project documentation](https://www.arcetri.inaf.it/science/ahead/XNS/code.html):
  hierarchically solves Poisson-like metric equations, rebuilds sources from
  the preceding iterate, and repeats the coupled field/matter loop to
  convergence. The companion
  [XNS guide](https://www.arcetri.inaf.it/science/ahead/XNS/files/Guide.pdf)
  documents separate field and Bernoulli/matter damping.
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
  controls and 1D/3D converged-data inputs. This is the immediate reference for
  the planned reusable physical-ID sequence seed.

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

## Modern Shock-Capturing Research Routes

Use these as method and comparison references, not imported production
implementations:

- A. Suresh and H. T. Huynh, *Accurate Monotonicity-Preserving Schemes with
  Runge--Kutta Time Stepping*,
  [NASA record and paper](https://ntrs.nasa.gov/citations/19970010128):
  original MP5 construction and its high-order monotonicity-preserving bounds.
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
- X. Zhang et al., *A Provably Positive-Preserving HLLD Riemann Solver for
  Ideal Magnetohydrodynamics. Part I: The One-Dimensional Case*,
  [paper](https://doi.org/10.4208/cicp.OA-2024-0068), and F. Zhang,
  A. Lani, and S. Poedts, *On physical consistency of HLL-type Riemann solvers
  for ideal MHD*, [preprint](https://arxiv.org/abs/2507.10420) and
  [published energy-consistency version](https://doi.org/10.1016/j.jcp.2026.114724):
  source routes for ordered positive fan states and magnetic-energy
  consistency, not production code to copy.
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
