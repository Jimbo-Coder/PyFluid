# PyFluid AMReX Working Knowledge

This is the compact, maintained state snapshot for an agent entering the AMReX
backend after a context reset. It is a router and status record, not a second
design specification. Read `../AGENTS.md`, this file,
`references.md`, and the tail of `review_running_cycle.md` before changing the
backend. When this file disagrees with current source or a newer continuity
cycle, verify the source and update this snapshot.

Last reconciled: 2026-07-22, branch `main`, through the local Cycle 221
elliptic-symmetry/SRMHD pass. Use the commit
containing this snapshot plus the exact newer continuity-cycle evidence rather
than inferring it from an earlier baseline.

## Scope

Infrastructure ownership, catalog rules, and artifact semantics live in
`docs/infrastructure_conventions.md`; source routes live in `references.md`.
This file keeps learned numerical/physics facts, current evidence boundaries,
and open limitations that materially affect future implementation.

## Current Validation Boundary

The latest recorded evidence is not one monolithic green result:

- Cycle 175 was the last recorded uninterrupted full suite: 46/46 gates. It
  predates substantial relativistic, diagnostic, transfer, and atmosphere
  work and is historical evidence, not validation of the present dirty tree.
- Cycle 192 recorded all production dimensions, architecture, generated
  physics, focused RHD/RMHD/TOV checks, the complete quick suite, and fresh
  runtime performance ratios of 0.93--0.99 of the frozen baselines.
- Cycle 193 changed the partial-AMR CTT matter handoff afterward. It recorded
  architecture, generated-kernel consistency, all production dimensions, the
  six-check focused CTT gate, and the three-check evolved TOV/subcycling gate.
  A requested three-rank rerun was not executable, and Cycle 193 did not record
  a complete quick or standard suite.
- Cycle 200 subsequently moved GF declarations to `schema/fields.toml` and
  removed dead flux-dispatch templates without changing active kernels. It
  recorded generated-kernel consistency, architecture, all production
  dimensions, the complete quick suite, and a three-rank WENO-Z/HLLC runtime
  comparison. This is the newest broad validation of the current dirty tree;
  it is still not a standard or full suite.
- Cycle 202 integrates the accumulated tree, cadence/catalog output cleanup,
  and validation retiering. Architecture, generated physics (including the
  atmosphere profiles), all production dimensions, both focused auxiliary
  cadence cases, the `2.64 s` quick gate, and all 42 standard gates pass. The
  roughly 90-second exhaustive multipatch executable is preserved in full and
  was not rerun by standard. Consult the end of Cycle 202 for the post-commit
  compile and runtime baseline rather than inferring those from this gate.
- Cycle 206 adds the first uniform-rotation XCTS matter/constraint continuation
  and the rotation-law TOML catalog. Generated/architecture checks, all
  production dimensions, the focused elliptic/CTT/TOV executable, exact
  zero-spin scalar equivalence, one N=24 `Omega=0.001` initialization, three
  PWL/HLL evolved-spacetime steps at CFL `0.9`, the focused RHD KH AMR pair, and
  all 42 standard gates pass. This is short finite-state evidence only: no
  long-term stability, rotating sequence, convergence, AMR rotation,
  full/stability suite, or fresh broad runtime-performance result was recorded.
- Cycle 208 makes `z^i = W v^i` the shared RHD/RMHD component-reconstruction
  contract, including metric-consistent reflecting ghosts. All production
  dimensions, generated and architecture checks, the focused RHD AMR pair,
  exact zero-spin plus three-step small-spin XCTS control, the full RMHD and
  fixed-spacetime suites, the `2.53 s` quick gate, and all standard gates pass.
  The canonical performance cases are `0.92--0.99` of their frozen baselines.
  The named 42/42 standard plus performance artifact is
  `runs/baselines/20260721_201750_relativistic_reconstruction_recovery/`.
  No full or stability suite was run.
- Cycle 210 corrects the XCTS shift matter source for the covariant Eulerian
  momentum used by the code, anchors the uniform Euler first integral at an
  interpolated rotation center, and continues only the matter component
  connected to the previous TOV/matter iterate. The focused rotating gate now
  requires agreement among matter, ADM-volume, and ADM-surface angular
  momentum and records `J_ADM/M_ADM^2`. N=24 ID-only models converge through
  `Omega=0.020`; this is a range probe, not a mass-shedding sequence or a
  resolution-converged spin limit.
- Cycle 211 adds fully converged angular-velocity continuation stages, compact
  coordinate surface extents, and a controlled 1/2/3-rank ID benchmark. The
  focused three-rank rotating-star gate, architecture and generated-kernel
  checks, all production dimensions, and the complete quick suite pass. This
  remains coarse N=24 uniform-grid evidence; no standard/full/stability suite
  or resolution-converged rotating sequence was run.
- Cycle 212 adds generated relativistic j-constant and q-power Euler integrals,
  a signed equatorial/polar surface-enthalpy-gradient diagnostic, and a
  fixed-central-density sequence driver with independent charge gates. N=24
  SLy j-constant (`A=5`) and q-power (`A=5`, `q=2`) small-spin models pass;
  matter/volume angular momentum agrees within `9.6e-4` and surface/volume
  within 1.4 percent. The signed uniform SLy sequence is directly checked at
  `Omega=0.028` and `0.029`, but its indicator is already non-monotone at this
  coarse resolution, so no mass-shedding endpoint is claimed. The focused
  elliptic/differential-law executable passes. Broad integration evidence is
  recorded at the end of Cycle 212 rather than assumed here. Architecture,
  generated consistency, all production dimensions, the focused rotating and
  elliptic tests, the complete quick gate, all 42 standard gates, and the
  canonical runtime benchmark pass. No full or stability suite was run.
- Cycle 213 makes the auxiliary thermal-entropy proxy EoS-aware and fixes two
  rotating-star hierarchy interpolation inputs that lacked exchanged ghost
  data at internal FAB boundaries. Generated consistency, architecture, all
  production dimensions, focused three-rank surface, elliptic, rotating-star,
  and actual auxiliary-HDF5 controls, the complete quick gate, all 42 standard
  gates, and the canonical runtime benchmark pass. The production integration
  rebuild took `375.94 s` with one cache hit and 156 misses; its immediate
  no-op rebuild took `0.61 s`. The generated compile-check executable still
  exits on its pre-existing high-order centering-transfer mismatch; the new
  EoS proxy checks emit no mismatch before that unrelated failure. No full or
  stability suite was run.
- Cycle 214 reaches a valid N=32 SLy uniform model at `Omega=0.033` and
  cleanly rejects `0.03325`, `0.0335`, and `0.034` as nonlinear/Robin failures,
  not physical mass-shedding measurements. A fixed-spacing domain-radius
  control pins the several-percent ADM--Komar difference on these small boxes
  to finite-radius ADM contamination: the surface discrepancy falls from
  4.39 to 2.94 percent as the extraction radius grows from 12.8 to 19.2, while
  the Komar surface mass changes by only 0.009 percent.
- Cycle 215 replaces the nested fully converged frozen-matter XCTS solve with
  a conservative self-consistent-field schedule: accurate XCTS solves settle
  large changes and certify the endpoint, while two relaxed XCTS sweeps are
  used only in the intermediate update regime. The identical N=24/three-rank
  small-spin benchmark falls from `159.62 s` to `66.81 s`; the known N=32
  `Omega=0.033` model falls from `3609.84 s` to `2267.96 s` and reproduces its
  radii and global charges within the configured matter tolerance. Generated
  consistency, architecture, Python syntax, sparse diagnostic plotting, the
  focused rotating gate, quick gate, and all standard gates pass. No full or
  stability suite was run.
- Cycle 216 adds explicit in-process angular-velocity targets plus one common
  physical-ID artifact for Newtonian fluid, TOV/RNS, RHD, and RMHD states. Its
  catalog-defined physical core contains primitives, ADM fields when present,
  and CT face fields when present; exact evolution caches support lossless
  same-hierarchy direct loading. A saved rotating XCTS model can seed a larger
  target with the same rotation law. Flat HD one-step and flat RMHD+CT
  round trips preserve all physical scalar diagnostics exactly, and an N=24
  RNS direct load reproduces its step-zero diagnostics while skipping XCTS.
  Cycle 217 generalizes the artifact into a source-hierarchy materializer:
  compatible coarser, finer, and larger target grids reconstruct conserved and
  Z4c state from the physical core, while identical layouts retain the exact
  cache path. Periodic remaps require an integer number of saved periods;
  nonperiodic ADM exteriors use asymptotically flat falloff. A remap is not a
  new constraint solve, so direct cross-grid evolution must retain that
  distinction; rotating continuation does solve XCTS again on the target.
  The reader rejects incompatible dimension, coordinates/axes, periodicity,
  physics, or physically relevant EoS metadata. RMHD standard permanently
  covers exact, coarser, and finer/larger one-step loads. A cross-grid RNS
  artifact continued XCTS and its resulting artifact then evolved one CFL
  `0.9` step with zero recovery events. The final quick and 42/42 standard
  gates pass. The clean implementation-commit baseline is
  `runs/baselines/20260722_175457_physical_id_materialization/`; its standard
  suite and performance commands return zero, with runtime ratios 0.90--0.96
  of frozen baselines. Persisted XCTS seeds reconstruct
  `A_conformal = psi^6 a_tilde` before continuing rotation.
- A live multi-target XCTS sequence can materialize and losslessly compress
  each accepted model without reloading it or discarding the conformal fields
  used by the next target. The focused SLy N=24 control writes two 19 MiB
  artifacts, exact-loads the final model with zero scalar discrepancy, evolves
  it one CFL 0.9 step with no recovery events, and continues the saved
  intermediate model to the same target within 0.20 percent in selected global
  scalars at the configured `1e-5` matter tolerance. Named EoS identity must be
  recorded as `eos_preset`; generic `eos=hybrid_polytrope` plus unused custom
  polytrope knobs does not identify SLy versus APR4. The clean implementation
  baseline for commit `504b4b8` is
  `runs/baselines/20260722_211331_rotating_id_sequences/`; standard passes
  42/42 and its four performance cases pass at 0.91--0.99 of reference.
- The shared elliptic layer now has component-wise Cartesian reflection
  boundaries, including mixed Robin/Neumann/Dirichlet nodal solves. Fresh
  zero-spin XCTS TOV octants and nonzero z-equatorial RNS half-domains execute;
  nonzero x/y rotating-star reflection is rejected because it is not a true
  plane symmetry. Stored reflected sectors parity-expand through the field
  catalog, with axial-vector parity for magnetic fields, and rotating surface
  rays reconstruct the omitted z sector.

Never collapse compile, focused validation, smoke, standard/full suite,
performance, and MPI-rank evidence into one claim.

## Open Numerical And Physics Boundaries

- Partial-AMR CTT initial data are finite and convergent after the common
  matter-handoff repair, but the normalized coarse-fine Hamiltonian band
  remains roughly constant under the recorded refinement. This is an exposed
  second-order multilevel-elliptic/fourth-order pointwise-diagnostic mismatch,
  not a completed high-order coarse-fine derivative closure.
- The self-contained rotating-star path is implemented in 3D for uniform,
  relativistic j-constant, and relativistic q-power rotation with a cold
  polytropic/hybrid-polytropic EoS and XCTS. It continues a direct TOV seed
  with generated relativistic Euler first integrals and a
  coupled under-relaxed matter/constraint iteration; zero angular velocity
  bypasses that update and retains the exact TOV control. The first executed
  nonzero gate is only a small-spin uniform-grid initialization plus three
  evolved-spacetime steps, not an RNS sequence, mass-shedding result,
  convergence study, or long-term stability claim. The current
  central-enthalpy closure evaluates the helical effective potential at the
  interpolated rotation center. Each relaxed update admits target matter only
  next to the previous matter component, so disconnected positive-enthalpy
  outer lobes are not materialized. This is appropriate for continuation of a
  centered spheroidal TOV seed; toroidal families need an explicit alternative
  topology seed rather than bypassing the connected-component contract.
  CTT and CTTK must reuse this matter layer, but still need a validated
  stationarity-compatible free-data closure before being labeled equilibrium.
  COCAL, RNS/FUKA, AthenaK, and ETK solutions are comparison baselines only.
  The zero-spin limit of each equilibrium branch must match the same-EoS TOV
  solution, including point fields, masses, radius, vanishing shift/extrinsic
  curvature/angular momentum, and evolution. The current deliberately nonzero-
  mean-curvature CTTK control is a different slice and is excluded from that
  equivalence unless its free data are changed to the equilibrium endpoint.
- Prior N=24 uniform-rotation charge probes remain useful charge evidence, but
  their Cycle-212 surface-gradient indicators are invalid. The temporary
  effective-potential and `ln h` MultiFabs had no ghost cells, so a linear
  interpolation stencil crossing an internal FAB boundary read outside the
  selected FAB. This box-layout dependence, not a demonstrated physical
  surface-resolution floor, caused the non-monotone N=24 indicator. Both
  fields now allocate and exchange one ghost layer; the hierarchy
  interpolation service also rejects a crossing stencil without sufficient
  storage before any rank-local interpolation or collective reduction.
- Corrected N=32 fixed-central-density SLy uniform models converge through
  `Omega=0.033`. The `0.032` and `0.033` signed mass-shedding indicators are
  `0.230266` and `0.081731`; their dimensionless ADM spins are `0.594246` and
  `0.619966`. At `Omega=0.033`, the equatorial/polar coordinate radii are
  `8.86002/4.5`, matter/volume angular momentum differs by `1.70e-4`,
  surface/volume by `5.73e-3`, and final XCTS residuals remain below
  `7.90e-13`. Attempts at `0.03325`, `0.0335`, and `0.034` fail in the final
  nonlinear/Robin solve before producing a valid physical indicator. They
  bracket the current solver basin, not the mass-shedding endpoint. A second
  resolution and a valid point closer to the cusp remain necessary.
  The prior roughly 40 percent surface deficit came from an incorrect
  conformal/lapse weight in the generated XCTS shift matter source, not the
  surface quadrature. The focused small-spin gate now enforces both independent
  charge comparisons. The 96-iteration ceiling is non-forcing: small spin
  still exits in 24 iterations, while coarse high-spin surface-cell activation
  can require more than 64. No equatorial/polar-radius convergence,
  physical maximum-spin claim exists yet.
- Uniform-star mass shedding is a physical surface condition: the outward
  equatorial enthalpy gradient approaches zero as the fluid angular velocity
  reaches the prograde circular-geodesic frequency at the equator. Nonlinear
  solver failure, finite-domain contact, or a timelike-flow guard does not
  measure that limit. A credible sequence therefore needs equatorial/polar
  radii, a surface-gradient or `Omega/Omega_K` indicator, independent charges,
  constraints, and at least two resolutions.
- The implemented Gourgoulhon-style indicator is the signed ratio of outward
  equatorial to polar `ln h` derivatives. Taking absolute derivatives is
  incorrect because it can hide a sign reversal past a cusp. On a coarse grid,
  cubic root/derivative interpolation still cannot recover information absent
  from the resolved surface; require monotonic behavior and resolution
  agreement rather than treating the smallest sampled value as the endpoint.
- The original N=24 uniform-grid XCTS solve with 8-cell boxes and nested
  frozen-matter constraint solves scaled from 399.69 seconds on one MPI rank
  to 159.39 seconds on three, or 83.6 percent parallel efficiency. MLMG was
  therefore already effective; the dominant waste was nonlinear nesting, not
  a missing multigrid method or MPI scaling collapse. The conservative
  self-consistent-field schedule reduces the identical three-rank solve to
  `66.81 s` while retaining the strict final XCTS solve. On the strongly
  deformed N=32 `Omega=0.033` sequence it reduces wall time from `3609.84 s`
  to `2267.96 s`, a 1.59-fold speedup. Reusing a saved converged physical ID as
  the next sequence seed remains the larger established opportunity.
- Hoisting fixed-point scratch allocations and reusing a nodal MLMG operator
  changed the identical three-rank N=24 solve from `159.39` to `159.62`
  seconds, with identical iteration counts, updates, and residuals, but also
  shifted the surface-gradient diagnostic from about `0.9997` to `0.9855`.
  The patch was therefore reverted rather than retained as an unmeasured and
  numerically non-identical optimization.
  AMReX's current nodal/cell initial-data Laplacians remain second order; its
  boundary interpolation-order knob does not create a fourth-order interior
  operator.
- `global_gr` volume diagnostics depend on current vertex-centered matter even
  when the separate constraint diagnostic group is not selected. Cycle 212
  fixed that dependency; before the fix, a `global_gr`-only row could read
  stale allocation contents and report plausible or catastrophic charges by
  accident. Keep a direct `global_gr`-only matter-coupled control.
- The Read-style SLy piecewise-polytrope implementation feeds the same cold
  barotropic rotating-star inversion successfully: N=24 models converge at
  `Omega=0.001` directly and at `Omega=0.025` with three continuation stages.
  This establishes EoS-path compatibility, not that piecewise polytropes
  improve convergence; the SLy and Gamma=2 stars are different physical
  models and their iteration counts are not a controlled EoS-algorithm test.
- Production RHD and RMHD face states reconstruct thermodynamic primitives and
  all three components of spatial four-velocity `z^i = W v^i`. The selected component
  method, including MP5, remains the high-order method; mapping `z^i` through
  the face metric avoids making a cell velocity superluminal merely by
  evaluating it in a different metric. RHD also has stage-candidate
  first-order flux correction: only faces adjoining a cell whose actual RK
  candidate violates the conserved density/admissibility floors are replaced
  with piecewise-constant LLF fluxes. This follows AthenaK's local-fallback
  structure rather than globally downgrading reconstruction. The current
  stage-candidate correction is RHD-only; RMHD retains its local face-state
  admissibility limit and ordered C2P policy. The N=512 exact Balsara
  reconstruction scan shows why this remains a production limitation: no one
  high-order method is uniformly robust. MP5 is most accurate on Balsara 1 and
  5 but resets on 2--4; PWL is the best Balsara 4 high-order control but still
  rings; WENO-Z has an off-scale Balsara 4 spike; and TENO5 resets 31 times on
  Balsara 4. PWC is diffusive but completes all five without a reset.
- During evolution, successful use of a secondary RMHD C2P method and terminal
  atmosphere resets after C2P exhaustion are counted across ranks and warned
  once per affected step. Ordinary atmosphere cells are not counted as
  failures. A terminal RMHD reset preserves finite magnetic fields; a
  nonfinite magnetic state remains fatal. The smooth five-method RMHD sweep,
  fixed-curved-metric controls, and small-spin RHD control all record zero
  recovery events.
- The current small-spin XCTS PWL control passes three evolved-spacetime steps
  at CFL 0.9 with rest-mass drift `8.52e-10`. Component MP5 now remains finite
  for the same three steps, but its `9.35e-8` rest-mass drift fails the existing
  `1e-8` gate because density-floor projection injects mass in surface cells.
  Keep the checked-in rotating-star input on PWL and do not weaken that gate;
  MP5 rotating-star evolution needs a conservative surface treatment before it
  is a validated production default. The accepted compact policy is to count
  and warn on recovery fallback/reset, not to add a special atmosphere
  mass/energy ledger. Reusable candidate-metric scratch and longer rotating-
  star validation remain open performance/numerical work.
- Native CompOSE/tabulated EoS support is not implemented. Add a cold,
  beta-equilibrated one-dimensional barotrope first and use the same loaded,
  unit-checked table for direct TOV, rotating-star initial data, C2P/P2C,
  signal speeds, atmosphere closure, and evolution. A cold-plus-thermal
  extension is a bounded next step; a general finite-temperature
  `(n_b,T,Y_e)` table also requires composition evolution and a larger
  inversion/validity contract and should not be conflated with the first stage.
  Importing selected CompOSE/TEOS table data is permitted; importing another
  code's TOV/RNS/BNS solution for production initialization is not.
- The output-only `thermal_entropy_proxy` is `p/rho^Gamma` for an ideal gas,
  `(p-p_cold)/rho^Gamma_th` for a hybrid cold-plus-thermal EoS, and zero for a
  strictly cold barotrope with no independent entropy. It is not absolute
  physical entropy; a tabulated EoS must provide its own thermodynamic
  definition. Do not use a piecewise-polytrope segment gamma on total pressure,
  because cold-segment changes then masquerade as entropy production.
- Fixed-spacetime and evolved curvilinear relativistic evolution remain
  pending but are explicitly deprioritized. Evolved curvilinear Z4c requires a generated reference-metric
  formulation, parity/origin regularity, nodal AMR transfer, and matching
  boundaries; it is not a Jacobian-only change.
- The radiation pipeline has surface interpolation, Weyl/Psi4 generation,
  multipoles, restart-aware histories, and postprocessing controls. Honest
  extraction-radius convergence toward null infinity still needs a compact
  isolated radiative source and multiple physical radii.
- Diagnostic infrastructure is already shared: generated catalog/integrands,
  composite-AMR volume reductions, spherical hierarchy interpolation,
  parametric-surface quadrature, and general multipole/Fourier reduction. The
  remaining additions are specific quantities: spherical mass-accretion rate,
  separated fluid and electromagnetic energy fluxes, and magnetic flux. The
  existing catalog
  already includes total finite-radius matter energy flux and atmosphere volume
  fraction. Recovery fallback/reset counts are compact runtime warnings rather
  than new scalar GFs. Flat/power-law initial atmospheres and the allocation-
  free physical geometry provider are implemented.

## Elliptic Symmetry And Flat Relativistic Kernels

- Centered Cartesian elliptic reflection planes are mixed Neumann/Dirichlet
  boundaries: even components have zero normal derivative and odd components
  vanish on the plane; outer faces retain their normal Robin/Dirichlet data.
  Nodal ghost synchronization must modify reflection ghosts only and preserve
  the elliptic operator's outer-boundary ghost state.
- A centered nonrotating TOV model can use x/y/z octant symmetry. A nonzero
  model rotating about z can use z-equatorial symmetry, but ordinary x/y
  reflection parity is physically invalid for its azimuthal velocity and
  shift. The configuration validator rejects that combination.
- Saved physical-ID reflection faces are part of source geometry metadata.
  Octant-to-full materialization reflects coordinates before interpolation and
  applies catalog parity afterward. Magnetic fields use axial-vector parity,
  unlike velocity and momentum.
- SRHD/SRMHD and fixed-GR HD/MHD share the same generated Valencia physics.
  Fixed Minkowski runs now bypass stored Z4c metric loads and the fourth-order
  stored-metric source derivative pass. A physical-ID artifact records its
  spacetime class, so a Minkowski source loaded into a Minkowski target retains
  the analytic path; fixed curved metrics and solved ADM data retain the
  general metric path. This is a host-selected metric specialization, not a
  duplicate SR physics implementation. On the recorded 4096-cell, 128-step
  MP5/HLL/SSPRK3 controls it improves SRHD throughput by 31.1 percent and SRMHD
  throughput by 2.1 percent.
- The tracked Giacomazzo--Rezzolla profiles under `../exact_balsara/` provide
  exact 1D SRMHD comparisons for Balsara 1--5. Their pressure column is total
  pressure; validation must subtract half the comoving magnetic magnitude
  before independently constructing the flat Valencia conserved state.
- At matched finest spacing, the current coarse TOV XCTS full/octant control
  differs by 1.34 percent in its worst selected global scalar; parity-expanding
  the saved octant artifact changes those scalars by only `1.78e-6`. The N=24
  uniform RNS full/z-half control differs by 4.99 percent in its worst global
  scalar while central density/lapse and surface radii agree much more closely.
  Treat these as coarse nodal-boundary truncation measurements, not exact
  sector equivalence.
- Three-step MP5 RNS controls at CFL 0.9 complete with zero recovery events,
  but their raw rest-mass drifts are `9.68e-8` (full) and `6.92e-8` (z-half),
  above the generic dynamic-spacetime policy's `1e-8` threshold. The focused
  ID-symmetry gate treats them as short smoke runs with an explicit `2e-7`
  bound; this is not a long-evolution conservation validation.

## AthenaK Reference Boundary

- A local AthenaK source tree is available at
  `/Users/mrizzo/phys_research/athenak/athenak`, with CPU and MPI executables at
  `build/src/athena` and `build_mpi/src/athena`. The source tree is not itself
  a Git checkout, but representative source, test, and build files were
  byte-matched on 2026-07-21 to AthenaK commit
  `5f1993109bcb2e5d588ba41b4efc897408e9959a` from
  `https://github.com/IAS-Astrophysics/athenak.git`. Recheck provenance before
  publishing or after the local tree changes.
- AthenaK is a valuable independent reference for Newtonian, SR, fixed-GR,
  dynamical-GRMHD, AMR, CT-MHD, Z4c, wave, shock-tube, puncture, and radiation
  algorithms and test setups. It is not PyFluid's source of truth and its
  numerical output is a code-to-code reference, not an analytic or exact
  solution.
- For relativistic reconstruction and recovery, inspect
  `src/dyn_grmhd/dyn_grmhd_fofc.cpp`, `src/dyn_grmhd/dyn_grmhd_fluxes.cpp`,
  `src/eos/ideal_grhyd.cpp`, and `src/eos/primitive-solver/reset_floor.hpp`.
  AthenaK predicts the concrete RK-stage candidate, asks C2P whether floors, a
  velocity ceiling, or recovery failure would be needed, replaces the faces
  adjoining flagged cells with first-order LLF/HLLE, and records event
  counters. Its floor policy is an explicit conserved-state mutation after the
  conservative flux correction, not part of the nominal reconstructor.
- For comparisons, first match equations, EoS/units, initial data, domain,
  boundaries, resolution, reconstruction/Riemann method, CFL, time integrator,
  output time, and AMR policy. Start with a uniform-grid control, compare
  physical fields/diagnostics on a common sampling, then add AMR. Record both
  revisions, dirty states, build configurations, inputs, rank counts, and
  output paths.
- Keep AthenaK use read-only unless a request explicitly includes changes to
  that checkout. Prefer an independent comparison over copying implementation.
  If code is adapted, review and preserve the BSD-3-Clause attribution and
  keep PyFluid's generated catalogs and ownership rules intact.
- The 2026-07-21 optimized CPU build-artifact comparison found the same broad
  optimization class in both projects (`-O3 -DNDEBUG -mcpu=native`) but very
  different generated-code volume. AthenaK's one runtime-dimensional binary
  linked about 13.5 MiB across 144 application objects. PyFluid's three
  `AMREX_SPACEDIM` binaries linked about 165.2 MiB; eight reconstruction-owned
  flux bundles contributed about 133.3 MiB. Treat reconstruction/layout/EoS/
  direction specialization cloning as the first compile-time target, while
  preserving separate AMReX dimension executables and benchmarking any shift
  from compile-time to runtime selection.
- If current measurement later justifies a dispatch rewrite, the accepted
  direction is an immutable host-built operation plan with approximately
  additive method ownership, not the current Cartesian template product and
  not an unmeasured per-cell virtual dispatch. Run-wide choices should be
  resolved at the coarsest practical boundary. Retain a compile-time policy
  only when AMReX storage requires it or a focused benchmark demonstrates a
  meaningful throughput benefit.
- Compiler caching plus method-local flux owners makes normal narrow iteration
  fast by reusing unrelated optimized objects. Treat the large specialization
  graph primarily as cold-build, broad-invalidation, CI/new-toolchain, disk,
  and possible code-size/runtime debt. Do not rewrite the dispatch unless a
  current cold/cache-miss matrix or runtime evidence demonstrates enough value;
  always report cold builds and cache-hit rebuilds separately.
- AthenaK's hydro/MHD flux owners keep Riemann choice explicitly instantiated
  but switch the uniform reconstruction method inside the Kokkos kernel. This
  makes method growth closer to additive, at the price of a uniform device
  branch. Treat that and a separate-method-launcher design as bounded benchmark
  candidates, not a source-level recipe.
- AthenaK's numerical-relativity task list builds a dependency graph once at
  startup: optional Z4c, matter, gauge, boundary, refinement, and diagnostic
  operations are added according to available modules and then topologically
  ordered. The useful PyFluid lesson is a small immutable host operation plan
  for future GR composition, while AMReX hierarchy lifecycle remains owned by
  the existing solver. Do not import a second scheduler.
- AthenaK's ordinary build also compiles only one selected problem generator.
  That reduces work compared with PyFluid's built-in initial-data set, but a
  feature-stripped validation binary conflicts with PyFluid's production-build
  contract. Keep it as explanatory evidence, not an optimization to copy.
- AthenaK's current Z4c tests are useful independent formulations and
  normalization references, but they are much narrower than PyFluid's existing
  gauge, robust-stability, AMR/interface, stellar, and radiation gates. Adapt
  test ideas only where they add an independent physical boundary; do not add
  duplicate linear-wave tests for breadth alone.

## Review And Change Discipline

The local ETK/CarpetX comparison explains how a large option surface avoids a
single template Cartesian product:

- CCL separates parameter identity (`param.ccl`), GF/group identity
  (`interface.ccl`), and conditional operation scheduling (`schedule.ccl`).
  Many apparent branches are independently compiled thorn routines that are
  activated/composed once, not alternatives inside one fused kernel.
- Hot AsterX code still uses uniform runtime choices. Its seven reconstruction
  methods and LxF/HLLE flux choice are enum switches in device code, while only
  direction and EoS object type are templated in the inspected flux owner.
- Modularity does not make the complete build small: the inspected ET_CarpetX
  configuration contains 85 thorn archives totaling about 402 MiB. AsterX is a
  16 MiB archive and its largest flux object is about 4.7 MiB. The advantage is
  additive thorn/owner compilation and effective incremental rebuilds, not a
  free clean build or zero runtime dispatch cost.
- The PyFluid analogue is a small immutable operation plan: compile dimension
  and required storage topology; schedule optional operations at the host
  level; use uniform runtime method enums where measured acceptable; reserve
  per-cell branches for genuinely data-dependent physics.

- Source-checked external review findings live in `review_backlog.md`. Raw
  Spark notes under `../spark_amrex_review/` are review input, not accepted
  requirements.
- Reproduce a claimed defect before refactoring around it. Prefer the smallest
  correction and its focused gate. Do not introduce a registry, metadata table,
  helper object, or generic iterator merely because a branch or loop repeats;
  retain it only when it reduces ownership drift or a measured maintenance,
  compile-time, or runtime cost.
- Do not repair nonfinite transfer output, weaken a tolerance, add a temporary
  unsupported-combination guard, or silently change a physical default to
  quarantine an active bug.
- Update `review_running_cycle.md` during every AMReX request. Update this file
  only when a durable ownership rule, validation boundary, or open limitation
  changes. Put historical detail and exact measurements in the continuity
  cycle rather than growing this snapshot indefinitely.
