# PyFluid AMReX Working Knowledge

This is the compact state snapshot for an agent entering the production AMReX
backend after a context reset. It records learned numerical and physics facts,
current evidence boundaries, and open limitations. It is not a design
specification, bibliography, or work log.

Read `../AGENTS.md`, this file, `references.md`,
`docs/infrastructure_conventions.md`, and the tail of
`review_running_cycle.md` before changing the backend. Current source and a
newer continuity cycle win if they disagree with this snapshot.

Last reconciled: 2026-08-13, branch `main`, through Cycle 288's axisymmetric
spherical elliptic foundation. Cycle 284's preserved standard/performance
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

- The self-contained rotating-star path supports 3D uniform, relativistic
  j-constant, and relativistic q-power rotation with cold
  polytropic/hybrid-polytropic EoS and XCTS. It continues a TOV seed through a
  coupled under-relaxed matter/constraint iteration. Zero angular velocity
  bypasses the rotating update and must reproduce the same-EoS TOV endpoint.
- Existing nonzero-spin evidence is coarse continuation and short evolution,
  not a resolution-converged mass-shedding limit, long-term equilibrium, or
  constant-baryon-mass sequence. The surface-gradient indicator required ghost
  exchange across FABs; results predating that fix are not physical endpoint
  evidence.
- The signed surface-gradient ratio diagnoses the equatorial mass-shedding
  cusp. It is neither a linear measure of remaining angular velocity nor a
  radial/axisymmetric stability criterion; the latter needs an appropriate
  fixed-baryon-mass sequence, turning-point analysis, and time evolution.
- At fixed central density, the fresh SLy uniform `Omega=0.028` N=24/32/40
  solves give equatorial `d ln(h)/dr = -0.0138293/-0.0134070/-0.0132684` and
  `J_ADM/M_ADM^2 = 0.496214/0.504568/0.508643`. The equatorial cusp trend and
  charges are converging, but the polar slopes
  `-0.0393502/-0.0331447/-0.0372987` and therefore
  `chi_ms = 0.351441/0.404501/0.355733` are nonmonotonic. This is strong
  evidence for a regular high-spin XCTS solution, not a resolution-converged
  endpoint; retain and inspect both raw slopes whenever the ratio is used.
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
  `docs/infrastructure_conventions.md`. A remapped artifact is initialization,
  not a new constraint solve; finer evolution cannot recreate unresolved ID
  information. A continued XCTS target does solve again on the target grid.
- Physical-ID schema v2 makes the source chart independent of the target
  executable: dimension, coordinate family, ordered active axes,
  periodic/reflection closure, and orthonormal tensor basis are runtime
  metadata. Scalars, polar/axial vectors, and symmetric tensors transform
  through Cartesian physical components; this adds one compiled materializer
  owner rather than a coordinate/formulation/EoS template axis. Exact
  same-layout caches remain optional accelerators.
- `initial_data.solve_only=1` performs the normal ID materialization and
  relativistic C2P/P2C preflight, writes the portable artifact, and exits before
  evolution. The validated reduced path is spherical 1D TOV to Cartesian 3D.
  At source `N_r=256/512/1024`, the maximum normalized density error against a
  direct 32-cubed TOV decreases `7.58e-4/3.52e-5/1.16e-5`; pressure, diagonal
  metric, and lapse errors also decrease monotonically. An `N_r=1024` source
  materialized onto `[-40,40]^3` has maximum tracked error `1.25e-5`, the
  regular origin density differs by `9.89e-6`, and a three-step CFL-0.9 loaded
  evolution has mass drift `3.51e-12` with zero recovery.
- A cell-centred radial source's half-cell gap at `r=0` is regular interior
  interpolation, not stellar exterior. Beyond a spherical TOV source radius,
  fluid uses the configured target atmosphere and ADM geometry uses the exact
  isotropic-Schwarzschild exterior inferred from the saved outer metric.
- The portable materializer does not make rotating XCTS axisymmetric by
  itself. The uniform cell-centred spherical `r theta` scalar Poisson operator
  now handles origin, axis/equator reflection, and finite-radius Dirichlet or
  Robin closure with a separable weighted angular solve. Its regular
  manufactured `P2+P4` solution converges at second order and gives the same
  result across one and three ranks. Full spherical `r theta` RNS still needs
  the conformal-vector/XCTS operator and zero-spin equivalence. Cross-chart CT
  is likewise rejected until face-centred divergence-preserving transfer is
  implemented.
- A nodal spherical solve makes every angular point at `r=0` a duplicate
  physical degree of freedom. Use a cell-centred radial origin for the reduced
  scalar operator. Stock point-smoothed Cartesian multigrid was not robust for
  the strongly weighted spherical-axis operator tested here; the separable
  solve diagonalizes the self-adjoint angular operator and solves independent
  radial tridiagonal systems without adding a coordinate/template product.
- A circular stationary axisymmetric RNS has only the physical azimuthal
  shift and momentum component. That field is divergence-free, so the flat
  conformal-vector equation reduces to the toroidal spin-1 scalar Laplacian,
  rather than a general coupled three-vector solve. Its axis/origin parity,
  inactive-phi storage, source terms, and longitudinal-tensor reconstruction
  still need a manufactured contract before it is a production XCTS path.
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
- Native CompOSE/tabulated-EoS support is not implemented. It must enter through
  the shared immutable EoS interface with one explicit unit convention and be
  validated consistently in TOV/RNS construction, P2C/C2P, signal speeds,
  atmosphere closure, and evolution.

## Other Open Boundaries

- The shared CTT/CTTK/XCTS elliptic solve is second-order. Raising the operator,
  coarse-fine closure, and boundary treatment together is required before
  claiming fourth-order elliptic accuracy; changing only interpolation or a
  diagnostic stencil is insufficient.
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

1. Implement and manufacture-test the spherical `r theta` scalar elliptic
   operator, then the conformal vector/XCTS operator and axis/equator
   regularity. Require zero-spin agreement with the spherical TOV source before
   continuing uniform RNS rotation and re-evaluating the surface diagnostic.
2. Make canonical face-`B` ownership stage-coupled before relying on long
   production GRMHD evolution. Retain the final post-sync projection as a
   backstop rather than calling it the stage-coupling solution.
3. Run a matched PyFluid/AthenaK work-precision comparison: uniform Balsara and
   multidimensional Orszag--Tang first, with AMR only after the uniform
   equivalence statement is sound. This calibrates performance and error; it is
   not another reconstruction search.
4. Add a two-resolution time-dependent TOV oscillation/constraint gate, then
   evolve a saved uniform-RNS artifact with PWL while tracking rest mass,
   angular momentum, constraints, central density, and mode content.
5. Add CompOSE/tabulated EoS after the stellar evolution boundary is stable.

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
