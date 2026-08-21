# PyFluid AMReX Working Knowledge

This is the compact state snapshot for an agent entering the production AMReX
backend after a context reset. It records learned numerical and physics facts,
current evidence boundaries, and open limitations. It is not a design
specification, bibliography, or work log.

Read `../AGENTS.md`, this file, `references.md`,
`docs/infrastructure_conventions.md`, and the tail of
`review_running_cycle.md` before changing the backend. Current source and a
newer continuity cycle win if they disagree with this snapshot.

Last reconciled: 2026-08-14, branch `main`, through Cycle 296's optional
equilibrium-torus matter component. Cycle 284's preserved standard/performance
baseline remains the latest broad integration baseline.

## Scope And Evidence

- `docs/infrastructure_conventions.md` owns durable source, catalog, AMR, and
  artifact contracts. `references.md` routes to implementation and external
  literature. `review_running_cycle.md` owns dated commands, measurements,
  failures, and decisions.
- The production implementation is C++/AMReX. Python research tools are
  baselines, offline operators, or postprocessing unless a task explicitly
  promotes and validates a production implementation.
- A compile, focused executable, smoke run, standard suite, full suite,
  performance benchmark, and MPI-rank comparison are different evidence.
  Report them separately and attach claims to the exact revision or dirty tree.
- Cycle 284's final cache-hot quick suite passed 3/3 in 1.776 s. Its preserved
  standard baseline passed 42/42 gates with 135.305 s summed gate time, and all
  four runtime performance controls passed their recorded warn thresholds. The
  archive is `runs/baselines/20260812_212858_cycle284_final`. This is broad
  standard-suite CPU evidence, not a full-suite or GPU validation result.
- Cycle 296's equilibrium-torus implementation tree passes the expanded 44/44
  standard suite with 178.07 s summed gate time. Its focused full matrix is
  separate evidence under `runs/current/equilibrium_torus_validation/`; this
  rolling result is not a preserved performance baseline.
- Preserved baselines under `runs/baselines/` and rolling TSV histories are
  routing evidence, not proof for later source. Re-run the smallest relevant
  gate before relying on an old result.

## Production Numerical Snapshot

- `PhysicalSystem` explicitly distinguishes HD, MHD, RHD, and RMHD. SRHD,
  SRMHD, fixed curved GRHD/GRMHD, and evolved-Z4c matter use the shared
  relativistic system rather than separate flat-space physics implementations.
  Analytic Minkowski is a host-selected metric specialization.
- Production reconstruction acts on primitive variables. For relativistic
  velocity it reconstructs `z^i = W v^i`, with metric-consistent boundary
  handling. The validated production configuration for discontinuous RMHD
  remains component MP5, SSPRK3, the candidate correction, and CFL 0.9; the
  low-order registered executable defaults are startup-safe, not this accuracy
  recommendation.
- The RMHD candidate guard predicts each concrete RK candidate, replaces
  implicated shared faces with the configured terminal low-order flux, and
  records correction counts. Direct MP5 to PWC is the selected general policy.
  PWL-MC before PWC remains an optional comparison, not the default.
- A post-update recovery fallback and an atmosphere reset are not the same as
  a conservative face correction. Both are counted across ranks and warned
  once per affected step. Ordinary prescribed atmosphere cells are not reset
  events. An evolution-time reset is a serious last resort, not a routine
  reconstruction branch or a reason to add an atmosphere mass ledger. Current
  run-total counters restart from zero after checkpoint restart, so their
  cumulative labels mean since process restart rather than whole logical run.
- MP5/PWC has the best general error/robustness balance in the current exact
  SRMHD shock matrix. PPM4/PWC is useful when lower shock ringing matters more
  than minimum L1 error. Unlimited parabolic reconstruction is a smooth-data
  reference and is not a discontinuous-flow production choice.
- `pwl` means monotonized-central limited PWL, not raw two-argument MINMOD.
  MC is less dissipative than strict MINMOD and reduces to a constant state
  when its bounded slope vanishes.
- SSPRK3 is the discontinuous-RMHD default because the stage-candidate guard is
  materially more robust with it than with non-SSP RK4 on extreme tubes. RK4
  remains appropriate for smooth problems.
- HLLD is the production RMHD Riemann solver. HLLC is a contact-restoring
  algebra/control path with known multidimensional strong-shock sensitivity;
  do not promote it by weakening HLLD's rotated or CT gates.

## RMHD HLLD Boundary

- Production HLLD implements the Mignone--Ugliano--Bodo five-wave SRMHD fan in
  one face-normal orthonormal frame. It solves the nonlinear total-pressure
  closure, samples the fan at the shift-driven interface, and transforms the
  Valencia flux back to the coordinate frame.
- Nonfinite, unordered, inadmissible, or nonconverged HLLD constructions use
  the explicit HLL limit. The current counter records solver evaluations,
  including discarded high-order trials, repeated corrected-face evaluations,
  and duplicate nodal FAB instances; zero is meaningful, but a positive value
  is not a count of unique final selected faces. This is a documented robust
  limit, not an unreported fallback.
- Exact contacts, rotational waves, zero-normal-field degeneracy, random fan
  audits, fixed curved backgrounds, evolved Z4c, uniform CT, and AMR+CT provide
  useful validation. Across the exact Balsara tubes HLLD's improvement over HLL
  is real but modest; a complete-wave fan cannot represent a subcell compound
  structure by itself.
- The current validation is not an all-regime invariant-domain, positivity, or
  entropy proof. Strong-magnetization intermediate thermodynamics and the
  coupled CT/source/AMR discretization remain audit boundaries. Do not narrow
  required signal bounds or tune the pressure closure to exact tube speeds.
- Standard entropy-stable RMHD derivations often use a Powell-like source.
  Their proof does not transfer to the present conservative staggered-CT path
  by changing a face flux alone.

## Exact And Analytic Reference Semantics

- A finite-volume profile comparison matches numerical conserved cell averages
  to exact conserved cell averages. For nonlinear systems, integrate the
  conserved map; do not average primitives or evaluate the conserved map at an
  averaged primitive. Recover exact primitives through the same C2P
  interpretation used for numerical cell averages.
- A fixed continuum exact solution is the physical authority in plots. Its
  finite-volume projection legitimately changes with cell width and phase and
  is the primary discrete error reference, but it must not be plotted or
  described as a resolution-dependent continuum solution.
- A supplementary norm may integrate the difference between the fixed
  continuum profile and the numerical piecewise-constant representation. It
  includes unavoidable within-cell representation error and is not the
  primary finite-volume order metric.
- Tabulated Riemann data require discontinuity-aware integration. Repeated
  coordinates encode jumps and must not be smoothed by ordinary interpolation.
  Reject nontrivial coordinate reversal. Normalize errors using the complete
  physical profile and compare TV on the same mesh.
- The Giacomazzo--Rezzolla files under `../exact_balsara/` are independent
  exact SRMHD references. Their pressure column is total pressure, so validation
  subtracts comoving magnetic pressure before constructing Valencia state.
  Their checked-in Balsara 2--5 generator used decimal `1.6666` although the
  published tests specify `5/3`; retain that provenance caveat.
- A discontinuous center-sampled IC can use a continuum finite-volume
  reference only when its jumps lie on cell faces. For translated discrete
  data, compare against a conservative translation of the initialized cell
  averages rather than a newly center-sampled continuum profile.
- Smooth analytic providers that initialize primitives at centers validate that
  discrete initialization. Do not silently relabel them high-order FV
  projections; changing initialization requires changing the analytic
  reference consistently.

## Balsara And Reconstruction Lessons

- The Balsara-3 dense feature is a material contact followed closely by another
  MHD front. Its physical width is about `0.011396` on the standard `[-1,1]`
  domain, so it spans less than one cell at N=128 and only a few cells through
  N=512. A homogeneous-cell FV state cannot represent both boundaries and the
  intermediate plateau while it is subcell.
- More diffusion is not a shell-resolution mechanism. PWC, strict MINMOD, and
  the tested density THINC/BVD variants broadened or lowered the shell. The
  demonstrated remedies are adequate resolution, state-following AMR, or a
  genuinely validated multi-interface in-cell representation.
- Production MP5/SSPRK3/HLLD converges monotonically on uniform Balsara-3
  refinement. State-following AMR recovers the separate structures with far
  fewer active cells than a uniform finest mesh while maintaining scalar and
  recovery gates.
  This is a focused one-dimensional result, not blanket AMR superiority.
- Balsara 4 is not one generic ringing mode. Much of its alternating structure
  is a cluster of physical transverse waves whose amplitude and phase are
  smeared. A scalar shock detector, entropy minimum, temporal second difference,
  or independent component limiter did not reliably separate it from legitimate
  Balsara-3/5 fan evolution.
- A continuous high-order sensor needs an exact smooth plateau. A weight merely
  close to one can accumulate damping over many reconstructions. Shared state
  order is safer than independently selected characteristic order when strong
  MHD/RMHD components remain coupled.
- Offline physical-wave carriers, mode-selective limiting, THINC/BVD,
  constrained-fan, and ADER-Prim/MOOD/CWENO experiments are useful Pareto or
  falsification controls. None currently dominates the production method over
  HD, MHD, SRHD, and SRMHD held-out tests. Do not add one to the production
  catalog from a favorable aggregate Balsara score.
- When evaluating a hybrid, compare the same-grid exact/reference, pure high
  method, pure low method, and hybrid over complete accepted timesteps. Include
  activation and reset counts. One Euler candidate is a causal diagnostic, not
  a method ranking.

## AMR Transfer And CT

- AMR follows the standard finite-volume contract: FillPatch/prolongation,
  flux-register reflux, average-down of covered cells, and for CT edge-EMF
  reflux plus face average-down. These operations are not interchangeable.
- Conservative component averages do not guarantee relativistic
  admissibility. A coarse parent can produce children whose exact component
  average is correct while one child recovers with negative pressure.
- For Cartesian analytic fixed spacetime with CT off, the production transfer
  guard applies
  `U_f = U_c + theta (U_f_raw - U_c)` with one maximal C2P-admissible `theta`
  shared by every child and every conserved component. A conservative base
  therefore retains the parent average, and an already admissible family is
  unchanged. A nonfinite child or inadmissible parent is an error, not input to
  an atmosphere repair.
- AMReX can request only part of a coarse parent's child family because FAB and
  ghost regions are split independently. A sibling-coupled limiter must expand
  to a sibling-aligned region, run the base interpolation and limiter on the
  complete family in private scratch, then copy only the requested region.
  Limiting only the visible subset makes results decomposition-dependent and
  invalidates the family-conservation argument.
- CT-on transfer needs a metric-aware limiter with canonical face `B` held
  fixed. Stored or evolved spacetime needs the corresponding stage metric; a
  Minkowski surrogate is not a valid admissibility test.
- Face-centered magnetic fields are the canonical CT state. RK-weighted EMF
  accumulation is required, but it is not by itself full stage coupling. The
  current path accumulates stage EMFs and applies one face curl update after the
  multistage fluid evolution; later fluid stages therefore do not yet
  reconstruct from the same-stage canonical face field.
- Synchronizing cell `B` from faces and re-running accepted-state C2P after the
  final face update is a necessary final-state backstop. It does not replace
  the remaining owner-correct task: advance canonical face `B` at every RK
  stage, synchronize it before reconstruction/physical acceptance, and use the
  same stage flux/EMF in fluid, CT, and AMR ledgers.
- After reflux, average-down, final face-field synchronization, and regrid, the
  hierarchy refreshes metric coupling and applies relativistic acceptance at
  the last state owner. Newly created or remade fine levels then copy the
  accepted state consistently into both current and old-time storage.
- CPU tiled kernels that write grown ghost regions must use AMReX's
  `growntilebox`; manually growing every internal tile creates overlapping
  OpenMP writes. Device-side temporary FABs must retain an `Elixir` through
  asynchronous interpolation/copy chains.
- The local AMReX build has no OpenMP support. Forced-small-tile 2D/3D boundary
  regressions validate output and ownership semantics, but they do not execute
  the former writes concurrently; an OpenMP-enabled run remains useful.
- The present transfer limiter has CPU evidence. Retrieving a device failure
  flag after every interpolater call creates a likely GPU synchronization
  point; batch status at a FillPatch owner before claiming GPU performance
  neutrality.
- Runtime configuration rejects diffusion and viscosity for RHD/RMHD, so a
  separate relativistic post-diffusion recovery path is not currently
  reachable. Revisit accepted-state ownership when relativistic diffusion is
  introduced.

## Stellar Initial Data And Evolution

- The self-contained rotating-star path supports 3D and reduced axisymmetric
  spherical `r theta` XCTS with uniform, relativistic j-constant, and
  relativistic q-power rotation for cold polytropic/hybrid-polytropic EoS. It
  continues a TOV seed through one coupled under-relaxed matter/constraint
  iteration. Zero angular velocity bypasses the rotating update and reproduces
  the same-EoS TOV endpoint.
- Existing nonzero-spin evidence is reduced-grid continuation and short
  evolution, not a resolution-converged mass-shedding limit, long-term
  equilibrium, or constant-baryon-mass sequence. The surface-gradient
  indicator required ghost exchange across FABs; results predating that fix
  are not physical endpoint evidence.
- The signed surface-gradient ratio diagnoses the equatorial mass-shedding
  cusp. It is neither a linear measure of remaining angular velocity nor a
  radial/axisymmetric stability criterion; the latter needs an appropriate
  fixed-baryon-mass sequence, turning-point analysis, and time evolution.
- At fixed central density `rho_c=1.28e-3`, independent 128x32, 256x64, and
  384x96 SLy sequences converge through `Omega=0.032`. Their
  `chi_ms=0.18439/0.17968/0.17868` and the global charges converge cleanly.
  The 384x96 continuation remains physical at `Omega=0.0329`, with
  `chi_ms=0.05144`, `M_ADM=1.51433`, `M_b=1.65248`, `J=1.53834`, and
  `J/M^2=0.67083`; at `0.033` the matter iteration reaches tolerance but the
  surface diagnostic is invalid, so no model is accepted. By contrast, the
  256x64 and 128x32 models accept `0.033` with `chi_ms=0.00685/0.03978`.
  This brackets the refined numerical endpoint in `0.0329 < Omega < 0.033`
  while demonstrating that the tiny coarse equatorial slope is not a
  resolution-converged physical mass-shedding frequency. Direct cross-grid
  near-limit continuation is also more fragile than an independent sequence.
- The identical N=24, small-spin XCTS solve takes 171.37/95.61/67.69 seconds
  on one/two/three local MPI ranks, for 89.6/84.4 percent efficiency at two/
  three ranks. This establishes useful local spatial scaling, not large-grid or
  multi-node efficiency.
- Matter, ADM-volume, and ADM-surface angular momentum should agree under
  refinement. ADM--Komar disagreement on the tested compact domains is largely
  a finite-radius extraction effect; do not tune the elliptic solve to force
  equality at one outer radius.
- A centered nonrotating TOV model can use x/y/z octant reflection. A star
  rotating about z can use z-equatorial reflection, but ordinary x/y plane
  parity is not a symmetry of its azimuthal velocity and shift.
- CTT and CTTK can reuse the rotating matter layer, but still need a validated
  stationarity-compatible free-data closure before being called rotating
  equilibrium methods. A deliberately nonzero-mean-curvature CTTK slice is not
  the zero-spin XCTS/TOV endpoint.
- The reusable physical-ID container and sequence behavior are defined in
  `docs/infrastructure_conventions.md`. Materialization onto an evolution grid
  is interpolation, not a new constraint solve, so finer evolution cannot
  recreate unresolved ID information. A changed-grid reduced solve-only RNS
  load re-solves XCTS even at unchanged angular velocity; a same-map load keeps
  the direct sequence path.
- Physical-ID schema v2 makes the source chart independent of the target
  executable: dimension, coordinate family, ordered active axes,
  periodic/reflection closure, and orthonormal tensor basis are runtime
  metadata. Scalars, polar/axial vectors, and symmetric tensors transform
  through Cartesian physical components; this adds one compiled materializer
  owner rather than a coordinate/formulation/EoS template axis. Exact
  same-layout caches remain optional accelerators.
- `initial_data.solve_only=1` performs the normal ID materialization and
  relativistic C2P/P2C preflight, writes the portable artifact, and exits before
  evolution. Validated reduced paths include spherical 1D TOV and spherical
  2D circular RNS to Cartesian 3D.
  At source `N_r=256/512/1024`, the maximum normalized density error against a
  direct 32-cubed TOV decreases `7.58e-4/3.52e-5/1.16e-5`; pressure, diagonal
  metric, and lapse errors also decrease monotonically. An `N_r=1024` source
  materialized onto `[-40,40]^3` has maximum tracked error `1.25e-5`, the
  regular origin density differs by `9.89e-6`, and a three-step CFL-0.9 loaded
  evolution has mass drift `3.51e-12` with zero recovery.
- The shared star-adapted radial map uses uniform physical spacing through an
  enclosing matter region and a geometric gravity exterior, with one third of
  radial cells reserved for the exterior. At `N_r=256`, adaptive TOV-to-32^3
  errors are density `4.98e-5`, pressure `9.24e-5`, diagonal metric `8.90e-6`,
  and lapse `4.77e-6`: all are 2.9--15.2 times smaller than uniform `N_r=256`
  and within 1.2--1.4 times uniform `N_r=512`. This is a demonstrated
  work-precision point, not residual-driven AMR.
- A 256x64 reduced SLy RNS artifact materializes onto a full three-rank 32^3
  Cartesian hierarchy with maximum primitive and conserved round-trip errors
  `7.16e-13` and `1.79e-13`. Constraint-solved data must perform the final P2C
  only after nodal ADM fields have been reconstructed at authoritative cell
  centering; using a provisional metric can make otherwise valid cold surface
  primitives fail C2P.
- Adaptive 256x64 small-spin RNS uses one quarter the cells of uniform 512x128,
  runs in `2.115 s` versus `7.467 s`, and differs in ADM mass by `6.75e-6`
  relative (`J/M^2` by `5.68e-4`). Cartesian 32^3 materialization retains
  primitive/conserved round-trip errors `4.31e-13/1.93e-13`. A saved mapped
  128x32 model remapped to a 96x24 adaptive grid and continued to the next
  rotation target agrees with a matched direct tight-tolerance solve to at
  most `8.08e-6` in global charges; explicit radial faces therefore support
  same/different-grid sequence continuation and evolution handoff.
- A cell-centred radial source's half-cell gap at `r=0`, and the corresponding
  strip adjoining a declared reflection plane such as the RNS equator, are
  regular interior interpolation rather than stellar exterior. Finer target
  grids must not turn either strip into atmosphere. Beyond a spherical TOV
  source radius, fluid uses the configured target atmosphere and ADM geometry
  uses the exact isotropic-Schwarzschild exterior inferred from the saved
  outer metric.
- The cell-centred spherical `r theta` scalar and azimuthal spin-1 operators
  own origin, axis/equator reflection, radial Robin closure, coupled XCTS
  sources, longitudinal-tensor reconstruction, matter iteration, and global
  diagnostics without adding a coordinate/formulation template axis. Their
  manufactured solutions converge at second order and agree across one and
  three ranks. Cross-chart CT remains rejected until face-centred
  divergence-preserving transfer is implemented.
- A nodal spherical solve makes every angular point at `r=0` a duplicate
  physical degree of freedom. Use a cell-centred radial origin for the reduced
  scalar operator. Stock point-smoothed Cartesian multigrid was not robust for
  the strongly weighted spherical-axis operator tested here; the separable
  solve diagonalizes the self-adjoint angular operator and solves independent
  radial tridiagonal systems without adding a coordinate/template product.
- A circular stationary axisymmetric RNS has only the physical azimuthal
  shift and momentum component. That field is divergence-free, so the flat
  conformal-vector equation reduces to the toroidal spin-1 scalar Laplacian,
  rather than a general coupled three-vector solve. A manufactured `l=1+3`
  field converges at order `1.98048`, with `1.09e-6` fine-grid error and
  `5.64e-11` residual, and agrees across one and three ranks. Portable vectors
  retain three physical components even on a two-dimensional source mesh.
- The separable spherical operator, matter/source update, diagnostics, ADM
  integrals, centering, portable evaluator, and same-grid continuation all
  consume the same explicit radial faces. The first Cycle-290 map failed
  because it starved the gravity exterior; reserving one third of cells there
  fixes that ownership/performance tradeoff. The adaptive 256x64 sequence
  reaches `Omega=0.0329` in `57.95 s` with `chi_ms=0.05083` and
  `J/M^2=0.670946`, close to uniform 384x96 (`129.26 s`, `0.05144`,
  `0.670831`); continuation to `0.033` again loses a resolved enthalpy surface.
  Thus `0.0329 < Omega < 0.033` survives an independent mapped-grid sequence.
  Surface-fitted/multidomain refinement and residual-controlled angular-mode
  truncation remain promising isolated-star upgrades; general block-adaptive
  ID remains the appropriate later route for binaries.
- Explicit-face layout identity includes the coordinate faces, not only AMReX
  index boxes and physical bounds. Otherwise a requested remap can silently
  retain the source map. Rotating artifacts persist an enclosing
  `matter_outer_radius`; different-grid materialization evaluates the exact
  target atmosphere outside that support so tiny power-law-atmosphere
  interpolation error cannot create disconnected matter near the light
  cylinder. Nodal ADM temporaries need ghost-filled derivatives and final
  `OverrideSync`: overlapping nodal boxes formerly disagreed only in
  `K_(r phi)`, by order unity at box boundaries.
- The conventional reduced RNS map now keeps uniform source spacing to 1.8
  times the nonrotating seed radius and one third of cells in the gravity
  exterior; both are bounded runtime controls, while saved explicit faces are
  authoritative. At `Omega=0.0329`, same-map re-solves at `512x128`, `768x192`,
  and `1024x256` take `10.6/26.2/45.2 s`. Their mass-shedding indicators are
  `0.049055/0.048556/0.048369`; bulk charges and raw surface slopes show
  observed unequal-grid orders about `1.74--1.99`. The `1024x256` model has
  `M_ADM=1.513929`, `M_b=1.651972`, `J=1.538163`, and `J/M^2=0.671106`.
  Its losslessly compressed portable artifact is about 20 MiB and materializes
  onto a full three-rank Cartesian `32^3` hierarchy in `1.18 s`, with maximum
  primitive/conserved preflight errors `8.82e-13/1.12e-14`. This is a
  high-resolution numerical sequence point, not yet a physical mass-shedding
  limit.
- At `Omega=0.03295`, matched tight-tolerance `1024x256` and independently
  refined `1536x192` models give `chi_ms=0.0293406/0.0292918`; angular momentum
  differs by `5.5e-7` relative and the other bulk charges by roughly `1e-5`.
  The respective re-solves take `31.56/43.89 s`. Both `1024x128` and
  `1536x192` accept `Omega=0.032975`, with `chi_ms=0.0126918/0.0116318`, and
  both reach the `1e-7` matter tolerance before the resolved enthalpy-surface
  guard rejects `Omega=0.0329875`. The fixed-central-density numerical
  mass-shedding bracket is therefore
  `0.032975 < Omega_limit < 0.0329875` on two grids. This is not a radial-
  stability turning point; that requires an appropriate constant-baryon-mass
  or constant-angular-momentum sequence.
- At `Omega=0.02968`, an adaptive `1024x128` uniform SLy model is about 90
  percent of the lower endpoint bracket and has `chi_ms=0.34204`,
  `M_ADM=1.42917`, `M_b=1.55769`, `J=1.19629`, and `J/M^2=0.58569`.
  Its roughly 14 MiB artifact is the practical fast-rotation evolution seed;
  endpoint bisection is calibration, not the normal solve path.
- A direct literature sanity check anchors the reduced solver to the published
  fixed-central-density uniform `N=1`, `K=100` BU sequence rather than only to
  its own residuals. At adaptive `512x64`, the near-zero BU0 probe, BU4, and
  near-shedding BU9 differ from the tabulated ADM masses by
  `2.10e-4/2.22e-4/1.89e-4` relative. Their quasi-isotropic coordinate
  equatorial-radius errors are `3.30e-3/6.48e-5/9.11e-3`, and absolute
  `r_p/r_e` errors are `0/1.41e-3/4.42e-3`. BU9 has
  `chi_ms=0.01990` and `J/M^2=0.63682`. This is strong evidence that the
  current IDs are physically recognizable equilibria across the uniform
  sequence. It is not a precision same-equation comparison: the reference is
  full quasi-isotropic GR, while PyFluid currently solves conformally flat
  XCTS at finite outer radius.
- The reduced `r theta` XCTS owner can compose either a TOV or rotating-star
  central object with an optional unmagnetized barotropic constant-`ell`
  equilibrium torus, where `ell=-u_phi/u_t`. `test_fluid` solves the central
  metric first and adds the torus without its stress-energy; it does not mean
  zero torus rest density. `coupled` iterates the stellar and torus first
  integrals with the XCTS fields and includes both matter sources. The disk law
  is a separate generated catalog from stellar `j_constant` and q-power laws.
- A coupled adaptive `256x64` SLy TOV-plus-disk control with `ell=5.2` and
  requested inner radius 8 has disk baryonic mass `0.103830`, angular momentum
  `0.524952`, resolved support `8.474--33.915`, and elliptic residuals below
  `1.5e-12` in the scalar equations and `5.3e-14` in momentum. A matched
  `512x96` solve changes disk maximum density, outer radius, and baryonic mass
  by at most 1.65 percent. One- and three-rank disk diagnostics agree to the
  focused gate tolerance. This establishes a resolved self-gravitating XCTS
  model, not equivalence to a full quasi-isotropic KEH solution.
- The same disk component composes with a small-spin RNS in both gravity modes.
  Physical-ID metadata retain the disk identity, support, component mass and
  angular momentum through reduced-to-Cartesian rematerialization. A two-step
  `32^3` evolved-Z4c smoke at CFL 0.9 has rest-mass drift below `1e-10` with
  zero recovery. Long-term star/disk equilibrium, nonconstant disk laws,
  magnetized tori, central black holes, cusp/overflow diagnostics, and
  cross-method comparison remain open.
- Differential rotation is validated over useful ranges, not to a physical
  endpoint. Q-power `q=1`, `A=10` reproduces j-constant fields and diagnostics
  to roundoff. Q-power `q=2` converges at `A=10`, `Omega=0.03` and at `A=5`,
  `Omega=0.0201`; held-out `q=1.5/3` branches reach `A=10`, `Omega=0.03` with
  `0.005` continuation spacing; j-constant `A=10` continues through
  `Omega=0.068` with reduced matter relaxation. A q=1.5 jump from `0.01` to
  `0.0201` was still decreasing at `2.14e-6` when the 128-iteration cap fired.
  Continuation-step, relaxation, or iteration-cap failure is a nonlinear basin
  boundary and must not be labeled mass shedding.
- A persisted rotating continuation must restore its converged Euler
  first-integral injection energy as well as rotation law, differential length,
  and q exponent. Falling back to the TOV injection constant can make the first
  continued XCTS right-hand side nonfinite even when the saved fields are
  valid.
- A matched `1024`-radial angular scan at `Omega=0.0329/0.03295` establishes
  distinct resolution roles. Relative to 256 theta cells, 48 cells are about
  5.9 times faster and retain ADM mass/spin within `3.5e-5/4.9e-5`, but their
  near-shedding `chi_ms` is 11.8 percent high. Counts 64/96/128 are about
  4.6/3.2/2.3 times faster and reduce the `chi_ms` error to 6.6/2.7/1.34
  percent. Every 48--128 case independently rejects `Omega=0.033` through the
  same resolved-surface guard. Use 64 for sequence discovery, 128 for endpoint
  work, and 256 as the angular-reference control.
- Offline parity-aware angular projection of the `1024x256` exterior gives
  maximum global ADM-field errors at the source-ball boundary of roughly
  `8.9e-6/1.8e-6/4.4e-7/1.1e-7` for `4/8/16/32` angular cells. Four cells are
  too aggressive near the star. Actual radius-dependent angular reduction
  needs a modal/multidomain interface and coupled residual test; COCAL keeps a
  uniform theta grid and therefore does not validate this algorithm for us.
- On finer same-domain physical-ID loads, outer target cell centers can lie in
  the half-cell strip outside the source cell-center interpolation hull. The
  reader must apply the field's physical exterior there. For TOV/RNS fluid it
  regenerates the configured atmosphere at the target point; copying the last
  source atmosphere value can fabricate matter support. This is distinct from
  the regular radial-origin gap above. Atmosphere/support classification also
  needs a roundoff tolerance rather than strict equality.
- Short rotating-star evolution is currently cleaner with PWL than MP5 at the
  surface. Do not weaken rest-mass gates to promote MP5; a conservative stellar
  surface treatment and longer evolution evidence are still required.
- Native CompOSE support currently means one-parameter cold beta-equilibrium
  barotropes plus an explicitly configured gamma-law thermal extension; it is
  not finite-temperature or composition-dependent CompOSE. The immutable
  table view is shared by TOV/RNS construction, cold enthalpy
  inversion, relativistic pressure/energy closure, signal
  speeds, atmosphere projection, C2P/P2C, physical-ID materialization, and RHD
  evolution. Native CompOSE evolution is intentionally RHD-only; RMHD remains
  ideal-gas-only until its C2P and advanced Riemann contracts support a general
  EoS.
- The cold reader uses the CompOSE `Q1=p/n_b` and
  `Q7=e/(n_b m_n)-1` quantities in `G=c=M_sun=1`, retains the tabulated total
  energy density, selects one table-specific baryon-mass normalization so the
  first point joins a vacuum polytrope with `h -> 1`, and uses a
  constant-sound-speed continuation above the retained causal prefix. The
  required `table.toml` states this physics/provenance contract. SLY4 and DD2
  retain their full source tables; APR retains 148 of 172 points before the
  causal continuation.
- The bundled SLY4/APR/DD2 TOV sequences agree with the providers' independent
  mass-radius files to maximum radius errors `0.6009/0.3273/0.1402` percent on
  the checked stable overlap; maximum-mass errors are below 1.5 percent. A
  `256x64` adaptive SLY4 RNS solves, saves, rematerializes to Cartesian, and
  evolves three CFL-0.9 SSPRK3/PWL steps with zero recovery. A `128x32` source
  passed ID/preflight but caused 48 first-step atmosphere resets after `32^3`
  materialization, so it is not the production handoff floor for this model.
  Physical-ID and checkpoint compatibility use the path-independent normalized
  table fingerprint and thermal gamma; the source directory is provenance,
  not identity.
- At fixed central density `rho_c=1.28e-3`, the cold SLY4 uniform sequence
  continues cleanly through `Omega=0.032`. A `256x64` model has
  `chi_ms=0.174849` and `J/M^2=0.644786`; a changed-grid `512x96` re-solve has
  `0.174208` and `0.644985`. Baryonic mass and angular momentum differ by
  `1.90e-4` and `2.15e-5` relative. One- and three-rank `512x96` diagnostics
  agree to roundoff, while the three-rank solve is 2.26 times faster on this
  Mac. Both reduced grids reject `Omega=0.033` only after the matter update
  reaches tolerance and the resolved enthalpy-zero surface is lost. Thus
  `0.032 < Omega_limit < 0.033` is a bounded tabulated-EoS numerical
  mass-shedding bracket, not a turning-point or constant-baryon-mass result.
- The `512x96`, `Omega=0.032` SLY4 artifact materializes directly onto `32^3`
  and `48^3` Cartesian grids and evolves to matched time `t=3.5` with CFL 0.9,
  SSPRK3/PWL/HLL, and zero recovery. Rest-mass drifts are
  `5.61e-9/5.51e-9`; angular-momentum changes are `0.837/0.511` percent and
  maximum-density changes are `0.837/0.423` percent. At the initial Cartesian
  state, Komar surface/volume mass differs by `5.4e-4`, matter/ADM-volume
  angular momentum by `3.3e-5`, and matter/ADM-surface angular momentum by
  `7.0e-4`. The known finite-radius ADM-surface/Komar gap is 5.5 percent and
  must not be tuned away. This interval is only 1.8 percent of a rotation;
  long equilibrium evidence remains open.

## Other Open Boundaries

- The shared CTT/CTTK/XCTS elliptic solve is second-order. Raising the operator,
  coarse-fine closure, and boundary treatment together is required before
  claiming fourth-order elliptic accuracy; changing only interpolation or a
  diagnostic stencil is insufficient.
- The existing reduced spherical speed comes from separability: one cached
  angular eigendecomposition turns each scalar/toroidal Poisson solve into
  independent tridiagonal radial mode solves on a fixed spherical map. A true
  surface-fitted map `r = r(s,theta)` makes metric/Jacobian coefficients
  angle-dependent and couples those modes. Generated XCTS sources, EoS and
  Euler first integrals, continuation, diagnostics, physical-ID artifacts, and
  Cartesian materialization remain reusable, but multidomain bases/transforms,
  regularity, free-surface updates, compactification, inter-domain matching,
  coupled linear/nonlinear solves, and their MPI ownership are a new elliptic
  backend rather than a nearly free subclass.
- A KEH backend is not the next speed optimization for the present reduced
  XCTS equations. The current cached angular diagonalization plus independent
  tridiagonal radial solves is linear in radial resolution per angular mode;
  a direct KEH radial Green-function convolution is quadratic unless given a
  separate fast implementation. KEH's real advantages are natural free-space
  multipole closure and access to a full quasi-isotropic/CST rotating-star
  formulation. First compare an offline free-space multipole outer solution
  with the finite-radius Robin result; implement a second backend only if that
  exposes a material physical or work-precision limit.
- Partial-AMR CTT data are finite and convergent, but the normalized
  coarse-fine Hamiltonian band exposes a second-order multilevel-elliptic versus
  fourth-order pointwise-diagnostic mismatch.
- Cartesian multipatch evolution owns unique cells. Patch overlap is ghost and
  interpolation context, not duplicated evolved volume. Conservative overset
  reconciliation remains unimplemented.
- Cylindrical and spherical single-patch kernels exist for validated method
  combinations. Mixed Cartesian-curvilinear seam evolution remains
  experimental. Do not infer production support from maps or sampling alone.
- Radiation infrastructure includes catalog-driven surface, volume, and
  multipole reductions plus Weyl/Psi4 paths. Missing quantities should extend
  those catalogs, not create separate output implementations.
- Reusable metric/candidate scratch and longer stellar runs are open runtime
  work. Compile caching makes narrow iteration fast, but cold builds and broad
  header invalidation remain relevant; measure cold and cache-hit builds
  separately.
- Cycle 284's isolated empty-tree production build took 544.32 s with 231
  cache misses; an unchanged production rebuild took 0.66 s. Template pruning
  reduced measured object products, but a genuinely cold build remains much
  slower than the normal incremental-development path.

## Independent References And Comparisons

- AthenaK, AsterX, ETK/CarpetX, COCAL, RNS/FUKA, and public exact solvers are
  comparison and method references. They are not PyFluid production initial
  data or implementations unless a request explicitly changes that boundary.
- A code comparison starts by matching equations, EoS/units, state convention,
  domain, boundaries, resolution, reconstruction, Riemann solver, CFL,
  timestepper, output time, and AMR policy. Start uniform, compare common
  physical quantities, then add AMR.
- The matched AthenaK/AsterX evidence is case-specific. PyFluid is competitive
  on the checked Balsara controls; it has not established broad superiority.
  Preserve both revisions, inputs, builds, rank counts, outputs, conservation,
  constraints, and wall times.
- AthenaK demonstrates additive owners, uniform runtime choices, and a
  startup-built operation graph. The useful PyFluid lesson is a small immutable
  host operation plan, not a second scheduler or a per-cell virtual interface.
- AMReX and Kokkos are different layers. Replacing AMReX with Kokkos would also
  require hierarchy, FillPatch, reflux, restriction, regridding, checkpoint,
  I/O, and elliptic infrastructure. Consider Kokkos only for a measured isolated
  hotspot with a clean data boundary.

## Next Milestones

1. Extend the saved fast analytic and CompOSE RNS models from the present short
   two-resolution evolutions to a meaningful fraction of a rotation, tracking
   baryon mass, angular momentum, constraints, central density, and modes.
   Extend the same work-precision history to the coupled TOV/RNS-plus-disk
   controls before calling them dynamical equilibria. Then construct constant-
   baryon-mass sequences before quoting a radial-stability/turning-point limit.
   Prototype an exterior modal domain only if it beats the 64/128-theta
   controls.
2. Make canonical face-`B` ownership stage-coupled before relying on long
   production GRMHD evolution. Retain the final post-sync projection as a
   backstop rather than calling it the stage-coupling solution.
3. Run a matched PyFluid/AthenaK work-precision comparison: uniform Balsara and
   multidimensional Orszag--Tang first, with AMR only after the uniform
   equivalence statement is sound. This calibrates performance and error; it is
   not another reconstruction search.
4. Add true finite-temperature/composition-dependent CompOSE only when a
   physical evolution milestone needs it. Do not relabel the current cold
   barotrope plus gamma-law thermal extension as finite-temperature support.

## Review Discipline

- Source-check external review findings before accepting them. Raw review prose
  is provenance, not an implementation plan.
- Prefer the smallest owner-correct fix and a focused reproducer. Do not repair
  nonfinite transfer output, weaken a physical tolerance, silently change a
  default, or add a temporary fallback to conceal an intended combination.
- Treat any numerical improvement as a Pareto point until held-out HD, MHD,
  SRHD, and SRMHD tests improve without regressions. One-step or offline
  evidence is not production validation.
- Update `review_running_cycle.md` with exact work state and measurements.
  Update this file only when a durable learned fact, current evidence boundary,
  or open limitation changes.
