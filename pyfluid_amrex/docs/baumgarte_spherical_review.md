# Baumgarte spherical reference review

Status: broad source survey complete; targeted numerical review below, 2026-10-20.
The companion coverage inventory explicitly distinguishes detailed reading,
interface survey and inventory-only artifacts. This is not a line-by-line audit
of every historical source variant. No collaborator program,
build script, hook, test, binary, or archive has been executed. No remote write
has been attempted. The bare reference clone has its push URL disabled by the
parent agent. This is a source-grounded reference note, not adopted PyFluid methodology
or independent scientific validation. Adaptations require their own checks.

Repository: [Baumgarte_Spherical_Code](https://github.com/yuheng-guo/Baumgarte_Spherical_Code).
All collaborator access is read-only under the persistent rule in
[AGENTS.md](../../AGENTS.md#read-only-collaborator-reference). Local reference
objects and the source view are non-writable; remote pushing is disabled.
This does not modify GitHub account permissions.

Default source: [pinned tree 2ee5550](https://github.com/yuheng-guo/Baumgarte_Spherical_Code/tree/2ee5550396be2c56fc1c54f9c2ac859edeb9f096).
Bare filenames and line numbers below refer to `SourceFiles/` at that commit
unless another branch is named. See the [branch map](baumgarte_spherical_branches.md)
for all 18 pinned heads and distinct research paths. Detailed inventories and
read-depth evidence are retained under `runs/current/infrastructure_audit/cycle475/`.

## Immediate conclusions

1. The reference contains a useful implementation of rescaled reference-metric
   BSSN/Z4-style gravity on one mapped spherical grid, parity fills that pass
   through the origin and poles, many initial-data providers, coupled
   axisymmetric hydro/electromagnetism, wave extraction, and a recent Kokkos
   port. It is substantially broader than its old `main` branch.
2. Its `Regrid` is global radial remapping/shrinking at fixed point count. It is
   not an AMR hierarchy: there is no coarse/fine FillPatch, flux register,
   reflux, edge-EMF reflux, or CT hierarchy solution to transfer to PyFluid.
3. Its physical r/theta samples are shifted half a logical cell away from the
   origin and polar axes. Its parity closures therefore do not supply the
   missing singular-node equations for PyFluid's vertex-centered fCCZ4 state.
   Do not simply remove PyFluid's regular-chart rejection.
4. The highest-value near-term work is independent tests of tensor parity and
   frame/derivative semantics, followed by an explicit centering/regularity
   design. Smooth mapped radial grids are a useful alternative for bounded
   spherical experiments, but do not replace conservative AMR or justify
   changing evolution policy silently.

## Reproducible source identity and coverage

The inventory in `reference_inventory.json` records every path, blob SHA,
byte size and branch tip in the 18 fetched heads. The initial snapshot has
1,014 distinct paths and 1,152 distinct blobs over all heads. `branch_pins.tsv`
is the parent-owned branch/commit pin list. Default evidence below is
`yguo/dev/output-validation` at
`2ee5550396be2c56fc1c54f9c2ac859edeb9f096`; line numbers are from
`git show <commit>:<path> | nl -ba`, not a mutable checkout.

| Branch | Pinned tip | Review role |
|---|---|---|
| main | fd5d52b17ee35c8ca0ad91bc9336a583c5244c65 | old April base; original grid, spacetime, hydro, ID and elliptic owners |
| rescale_Bi | d223176705190512edc79759249860aff8d7e35f | rescaled magnetic variables history |
| MHD_rescaled_Bi_smooth_surface | 44b47ceab374532118322045c7850c020eca246e | interior MHD/surface experiments |
| excision | 2cde81f0326397e32996d31a8c89fecb7fbe6b41 | excision-aware derivatives |
| particle_surface_ex | 83517644bd312015d70ae4bf2ea320c8c957492a | particle surface and alternate curvature files |
| moving_grid | ad028c2f46d0c037d72370fef18cee21128138d2 | moving exterior electromagnetic grid |
| EM_ICN_only_theta_dep_EM_ex_BCs | 933b1b5705c074a085fdd10b28ea1d3f162715c9 | angularly dependent electromagnetic surface BCs |
| interior_only_B_field | cee7c7aa510c2ddd9dfcaceab1d6c5ba98f76fb6 | confined field experiment |
| better_Psi4_tetrad | 5ae037be7e5118418ad99bef039a05a9a24d1d6e | wave extraction/tetrad changes |
| better_Psi4_tetrad_anvil | d18209471843f8b31eb9fb9e75754bf448592008 | additional invariant/source modes, Weyl-index change |
| commit_7a6a1e9_13_July | a40f826d49c797ebc40e9d4953c74def22bc40c3 | pinned historical executable/source snapshot |
| version_10_Aug_2026 | cd36f7f50ca8ca5ff185444d716fe9cc4e490640 | historical physics base |
| fix_E_osc_issue_2 | 87519aa3051055e9f48668d357c370007cb1bbd6 | electromagnetic oscillation experiments |
| version_27_Aug_2026 | bf80c622f40e8441f4caf7c566fda5db36393291 | late-August integration |
| Magnetized_TOV | eda44cc851e12eeef6a561891c1c140eec4ea0f7 | unique magnetized TOV provider and recovery fix |
| yguo/dev/reference-profile | d03a81f4f4907ba7496bd7b4a4611f659833a370 | Kokkos/device port and performance documentation |
| yguo/dev/output-control | fd4c85786cd885e4a497f7bbd8254455b6a42388 | output schedule/HDF5 changes |
| yguo/dev/output-validation | 2ee5550396be2c56fc1c54f9c2ac859edeb9f096 | latest source and timing documentation |

The latest head contains 435 files: 257 under `SourceFiles`, 84 under `tools`,
46 under `Build`, 19 under `docs`, 18 under `code`, and 11 at other roots.
Inventory is complete; source reading is selective and organized by numerical
owner. Historical saved `.C.*` alternatives are provenance, not simultaneously
compiled implementations. Binaries, `.tgz` archives, generated CMake products
and raw data are inventoried but not executed or treated as validation evidence.

## Coordinates, basis, regularization and topology

`SourceFiles/Grid.h:6–7` states the design: uniform logical x,y with physical
r(x), theta(y). `Grid.h:446–449,535–549,569–573` uses half-cell shifted logical
samples in r, theta and phi. Equatorial symmetry halves the theta domain;
axisymmetry fixes one interior phi point (`Grid.h:166–169`). The underlying
three physical components remain in the state.

The default analytic radial map is a normalized combination of tanh and sinh,
with analytic first and second map derivatives (`Grid.h:614–641`). The
optional targeted map prescribes positive physical spacing Delta(r), with
smooth transitions from stellar to wave to outer regions, integrates
dr/dx=C Delta(r), and normalizes to the outer radius
(`Grid.h:645–728`). It logs star/wave spacing and maximum adjacent spacing
change (`Grid.h:467–518`). `SetCustomRadialGrid` instead estimates map
derivatives with second-order differences and rejects nonmonotone spacing
(`Grid.h:333–367`). High-order field stencils do not automatically improve
those map derivatives.

The metric split is explicitly documented in `Curvature.C:9–40`:
bar(gamma)=eta+epsilon, with epsilon_rr=h_rr,
epsilon_rtheta=r h_rt, epsilon_rphi=r sin(theta) h_rp,
epsilon_thetatheta=r^2 h_tt, epsilon_thetaphi=r^2 sin(theta) h_tp,
epsilon_phiphi=r^2 sin^2(theta) h_pp. A_tilde follows the same pattern;
Lambda^i=(lam_r,lam_t/r,lam_p/(r sin(theta))). `Curvature.C:76–111`
includes the basis/connection terms when differentiating these fields.
The determinant stored by curvature excludes the reference r^4 sin^2(theta)
factor (`Curvature.C:153–179`). These are representation contracts, not
small-radius clipping prescriptions.

`State.h:97–160` declares center/axis/equator parity per rescaled field.
Examples: scalar (+,+,+), radial vector (-,+,+), polar vector (+,-,-),
azimuthal vector (-,-,+); h_rt (-,-,-), h_rp (+,-,+), h_tp (-,+,-).
`gridfunction.h:27–78` implements the coordinate identifications: negative-r
ghosts reflect theta and shift phi by pi; pole reflection shifts phi by pi;
equatorial reflection uses its separate sign. With equatorial symmetry, the
origin mapping composes center and equator signs. Ghost fill order is radial,
then theta, then phi because later phases read earlier phases
(`gridfunction.h:336–367`). These transformations are an excellent
independent test oracle for our schema-driven tensor handling, but PyFluid
must derive component identity from its own catalog.

PyFluid currently rejects singular nodes in
`src/relativity/fccz4_runtime.cpp:54–69`. Its fCCZ4 spacetime is vertex centered,
whereas the reference avoids sampling singular points. A production extension
requires a decision among a regular staggered spacetime discretization,
singular-node analytic limits with degree-of-freedom identification, or a
patch layout avoiding the singular chart. Parity ghosts alone are insufficient.

## Spacetime equations and time integration

`State.h:7,32–42,89` ties the rescaling to Baumgarte–Montero–Cordero-Carrion–
Mueller (2013), stores 25 spacetime/gauge fields including phi, K, Theta,
Lambda and auxiliary Gamma-driver B. `Manager.h:65–69` selects Eulerian versus
Lagrangian determinant handling and BSSN versus a code-labeled Z4c switch.
Do not equate this label with PyFluid's generated fCCZ4 formulation by name.
Map every K/Theta/Lambda and damping convention first.

`Compute_RHS.C:196–259` retains comments from an older PIRK L2/L3 split, but
current integration dispatch explicitly selects RK4, RK3, RK2 or ICN
(`Integrate.C:35–43`). Thus those comments do not establish that the current
code uses PIRK. Lagrangian metric rescaling is explicit at accepted-step
boundaries (`Integrate.C:513–516,608–611,690–693`). Constraint damping includes
constant and nonlinear terms, e.g. kappa_11+kappa_12 Theta^2
(`Compute_RHS.C:229–245`) and connection-constraint damping
(`Compute_RHS.C:437–448`). These should be documented as reference choices,
not imported as a PyFluid formula change.

Spatial derivative order is compile-selected 4/6/8 in r/theta
(`Stencil.h:94–106`). Excision-aware derivatives downgrade near the inner cut
and use one-sided formulas (`Stencil.h:108–238`). Surface-aware derivative
experiments exist separately (`Stencil.h:240–...`); they are not evidence that
every curvature stencil avoids matter-surface discontinuities. Phi central
derivatives remain a distinct fourth-order implementation even when the
r/theta order is higher (`Stencil.h:525–530,586–592`). KO selects order 5 or 7
and checks ghost reach (`Stencil.h:967–985,1033–1046`).

The active timestep uses the minimum dr and r*dtheta over radial samples
(`Manager.h:279–298`); it is not a demonstrated full three-dimensional
characteristic-speed CFL policy. The call to recompute timestep every
accepted loop is commented out (`Integrate.C:22`). PyFluid should retain its
own physical characteristic-speed and AMR substep contracts.

## Mesh adaptation and conservation boundary

`Regrid.C:60–92` interpolates all dynamical fields onto one new radial grid,
then replaces the map. `Grid.h:402–429` changes r_max by one global factor;
there are no concurrent fine levels. Criteria are self-similar shrink or a
lapse/matter gradient (`Regrid.C:148–174`). Before/after mass and momentum
diagnostics exist (`Regrid.C:24–35,128–139`), but a diagnostic does not make
this interpolation conservative. No source-grounded AMR reflux method was
found. Keep PyFluid's standard FV/CT hierarchy contract unchanged.

## Matter and electromagnetic scope

The latest `Hydro_with_Maxwell.h:394` describes its intended model as
relativistic hydrodynamics in the interior with Maxwell equations in vacuum
outside. `Hydro_with_Maxwell_State.h:28–42` stores five fluid conserved fields
and just a_p/e_p. Its state-header comments at lines 41–42 describe the older
Maxwell scaling and are stale for the HWM evolution: the current equations
instead use A_phi=r sin^2(theta) a_p and E^phi=e_p/r
(`Hydro_with_Maxwell.C:24–33,1232–1240,3707`). The standalone `Maxwell.C`
owner retains a different scaling, so identical member names are not a shared
physical representation. Any adaptation must follow actual conversions and
equations rather than copying these comments.
The derived poloidal magnetic field is not a general three-component
face-centered CT GRMHD state. `HWMBlocks.h:65–66` explicitly uses
B=(Br,Btheta,0) in its local state helper. Its scalar potential/curl structure
and surface matching can guide a future axisymmetric electrovacuum research
owner, but do not establish arbitrary-field RMHD, force-free exterior physics,
AMR CT, or nonaxisymmetric support.

The surface owner uses particles, theta-dependent radius/velocity arrays and
an optional atmosphere-based surface; `Hydro_with_Maxwell.h:58–63,91–122,
267–277` declares these policies. There is a distinct moving exterior grid
with ICN update (`Hydro_with_Maxwell.h:433–455`). SurfaceMask classifies points
using per-ray or mean stellar radius (`HWMBlocks.h:299–350`) and specialized
derivatives treat the surface (`HWMBlocks.h:360–...`). These are additional
free-boundary/discretization contracts, not a generic cure for FV
vacuum-interface reflux.

## Initial-data, elliptic, diagnostics and infrastructure routes

`GR.C:257–309` and `InData.h:23–27,766–789` route analytic/ODE/elliptic initial
data: linear and gauge waves, flat space, Schwarzschild/trumpets, Kerr and
Kerr–Schild, Bowen–York and Brill–Lindquist, Brill/Teukolsky/Shibata waves,
TOV/RNS, Bondi, OS collapse, magnetized OS, critical scalar/fluid collapse,
radiation and shock problems. Unique magnetized TOV work is on `Magnetized_TOV`
and must be reviewed separately from the later Kokkos lineage.

`ConstraintSolver.h`, `EllSolver2D.h`, `EllSolver3D.h`,
`FlatEllSolver3D2ndOrder.h`, `VecLaplaceSolver.h` provide host initial-data
elliptic owners based on Epetra/AztecOO. `WaveExtraction.h`,
`ZerilliExtraction.h`, `EnergyFluxIntegrals.h`, `SurfaceIntegralDiagnostics.h`,
`HorizonFinder.{h,C}` and `Diagnostics.{h,C}` are the routes for curvature/
constraints, surface charges, horizons, Psi4/strain/mode extraction and EM
energy flux. Detailed formulas and branch-specific limitations remain under
review; their existence alone is not a validation claim.

The recent Kokkos port keeps point helpers on Views/plain structs, virtual and
allocating operations outside device kernels, and compares the port to frozen
CPU results (`docs/kokkos-port/ROADMAP.md:15–27`). It explicitly limits GPU
acceptance to matter type 8; other matter models and photons retain unvalidated
host paths (`ROADMAP.md:148–153`). Its bitwise/ref-drift gates establish
transcription/equivalence within their tested decks, not continuum accuracy.
The Kinnersley tetrad is explicitly rejected on this path
(`ROADMAP.md:153`). GPU timing/throughput numbers in those docs are
collaborator-recorded measurements and have not been reproduced here.

## Full owner map and additional lessons

| Numerical owner | Sources at the latest pin | What the implementation offers | Adaptation boundary |
|---|---|---|---|
| Reference geometry and tensors | Grid.h, Stencil.h, tensors.h, Curvature.C/.h | half-shifted mapped spherical grid; reference-covariant metric derivatives; rescaled tensor components; inline fixed-size tensors | keep PyFluid chart algebra and generated kernels; compare every tensor/derivative index independently |
| Evolution/RK | Manager.h, Compute_RHS.C, Integrate.C, Initialize.C | spacetime/matter RK staging, algebraic determinant/trace corrections, ICN exterior step | retain our single RK catalog and accepted-state projection budgets |
| Gauge | Slicing.h, Gauge.h, Bona_Masso_function.h | geodesic, 1+log, advective 1+log, harmonic and Bona–Masso lapse; Gamma/Jena/advective/covariant/self-similar shifts | auxiliary-B and integrated-driver variants have different state/equation semantics |
| Hydro | Hydro.C/.h, Hydro_State.h, Hydro_Aux.h, HydroBlocks.h, Approximate_Riemann_Solver.h, Slope_Limiter.h, Fluxes.h | primitive reconstruction, MC/minmod and PPM-r options, LLF/HLLE fluxes, gamma-law helpers and recovery | spherical source/flux splitting is not our exact-area conservative FV lowering |
| Hydro+EM | Hydro_with_Maxwell.C/.h, HWMBlocks.h, HWMRecovery.h, HWMSurface.h and associated state/aux | axisymmetric poloidal magnetic state, dust recovery, stellar surface particles, moving electrovacuum shell and matching | separate EM scaling from standalone Maxwell; no general CT/AMR claim |
| Electrovacuum | Maxwell.C/.h, Maxwell_State.h, Maxwell_Aux.h | reduced A_phi/E^phi equations and metric sources | retain actual scaling/assumed azimuthal symmetry; no FFE closure |
| Scalar matter | ScalarField.C/.h, Scalar_State.h, Scalar_Aux.h | scalar-field wave evolution, conjugate momentum, ADM source and central-collapse diagnostics | host paths remain outside the documented GPU-port acceptance |
| Radiation matter | RadHydro.C/.h, RadHydro_State.h, RadHydro_Aux.h, GasRad.h | coupled gas/radiation conserved energy and momenta; mixture EoS with dimensional conversion | does not establish a general neutrino transport closure or existing PyFluid capability |
| EoS/recovery | EOS.h, EOS_safe.h, GasRad.h, HydroBlocks.h, RootFind.h, brent.h | polytropic, gamma-law, piecewise-polytrope, ideal-gas and gas/radiation models; bracketed roots | ported HWM device state is gamma-law and dust-specialized despite broader host interfaces |
| Initial data | InData.h and INDATA/* | broad analytic/ODE/elliptic library and common physical-point provider | formulas, units, gauges and regularity vary by provider; not all are manufactured exact evolved solutions |
| Elliptic constraints | ConstraintSolver.h, FlatEllSolver3D2ndOrder.h, EllSolver2D.h, EllSolver3D.h, VecLaplaceSolver.h | Newton-like Hamiltonian correction and sparse spherical scalar/vector operators through Epetra/AztecOO | old EllSolver3D explicitly hardcodes two ghosts; not AMReX MLMG/AMR infrastructure |
| Horizons | HorizonFinder.C/.h, surfacefunction.h | star-shaped h(theta,phi), surface projection, elliptic expansion iteration, area/spin/momentum/tendicity/vorticity | remains a separate sampling/elliptic owner; does not establish moving excision causality |
| Charges/constraints | Diagnostics.C/.h, DiagnosticsBlocks.h, SurfaceIntegralDiagnostics.h, Monitor.h | Hamiltonian/momentum/connection constraints; multiple-radius ADM charges; invariants; deterministic signed maximum reduction | masks, weights, normalization and hemisphere factors differ; do not compare labels without definitions |
| Radiation extraction | WaveExtraction.h, ZerilliExtraction.h, ExtractionBlocks.h, EnergyFluxIntegrals.h | even m=0 Psi4/metric/Zerilli modes, direct strain integration, EM/GW surface fluxes and cumulative energy | mode/tetrad/phase/sign/areal-radius/integration conventions must be mapped explicitly |
| Tracers | Particles.h, Photons.h | fluid/surface trajectories, interior-only velocity interpolation, photon swarms | particle flow is not grid ownership; photon path has no documented device acceptance |
| Output/restart | dumper.C/.h, OutputInput.h, OutputSchedule.h, OutputHDF5.h, Profiles.h, CheckPoint.h | independent rays/slices/profiles/monitor/waves/checkpoint/horizon schedules; HDF5 event metadata; archive/restart input inclusion | preserve PyFluid's schema, compressed parallel output and versioned restart semantics |
| Performance/port gates | Kernels.h, Stencil.h, Interp.h, LoopTimer.h, CodeProfile.h, SourceFiles/tests/*, tools/delta/* | plain-value device captures, state-free interpolation search, reusable local point helpers, phase timers, kernel label coverage, CPU/GPU equivalence | portable point operations can be adapted independently; no matched backend compile comparison, fast-math or drift fallback acceptance |
| Older independent EM prototype | code/MagOS.C, code/em.C/.h, code/ex_em.C/.h and support files | reduced radial surface/excision EM integrator and energy checks | separate old executable with obsolete C++ headers, raw arrays and distinct coordinate assumptions; no modern integration evidence |

### Time, normalization and magnetic conventions

The field equations and horizon-radius formulas use geometrized quantities
(for example `INDATA/Bondi_Solution.h:76–83,107` uses a 2M horizon and critical
accretion formulas); this is an inference from the equations, not a universal
repository-wide units declaration. `GasRad.h:24–56,82–104` explicitly supplies
cgs conversions and a gas/radiation length scale. Magnetic stresses include
1/(4 pi), and four-field magnetic components include its square root
(`HWMBlocks.h:69`, `Hydro_with_Maxwell.C:1656,1841`). PyFluid's absorbed-field
normalization must be converted before any cross-code profile comparison.

`t` is coordinate time; `tau_c` is separately integrated central proper time
and appears in diagnostics and restart metadata (`Integrate.C` RK methods;
`CheckPoint.h:20–24,71–74`). Output cadence unit `M` is the simulation-time
coordinate used by the deck; copying a numeric interval between differently
normalized mass/time systems is not a physical comparison.

### High order means several distinct orders

The default latest CMake target explicitly defines EIGHTHORDER, EQSYMMETRY and
AXISYMMETRY (`CMakeLists.txt:182–214`). The source contains some nonaxisymmetric
branches, but its latest build/test evidence is an axisymmetric equatorial
subset. Bulk r/theta central operators can be eighth order, while:

- upwind r/theta transport uses the same sixth-order formulas under both
  SIXTHORDER and EIGHTHORDER (`Stencil.h:717–781`);
- phi central/transport formulas remain fourth order when axisymmetry is off
  (`Stencil.h:525–530,586–592,782–792`);
- ordinary interpolation is six points for the high-order builds and four
  points otherwise (`Stencil.h:1033–1040`), with special eight-point Zerilli
  interpolation (`ExtractionBlocks.h:230–245`);
- custom map derivatives are second order; surface interpolation, physical
  boundaries, fluid reconstruction and temporal integration have separate
  orders.

This is a useful warning against advertising the complete scheme with one
stencil-order label. PyFluid's current independent stencil/gauge/constraint,
fluid, transfer and ID accuracy contracts should remain explicit.

### Surface regularization changes equations/discrete budgets

The HWM owner factors known magnetic angular structure into Bbarr and
Bbartheta (`Hydro_with_Maxwell.C:1232–1240`). Factoring a canceling 1/r
combination analytically can reduce cancellation error, but the factorization
depends on the assumed field topology. It is not a generic magnetic-variable
rescaling and must not silently restrict retained physical modes.

Surface profile projection uses a small even Legendre basis; the historical
adjustable `lmax` only selects 1, 2 or 3 even modes (0,2,4) rather than every
mode through an arbitrary integer (`fix_E_osc_issue_2` pin,
`Hydro_with_Maxwell.C:805–905`). The latest owner carries the same surface
filter family. That is an explicit low-pass surface model. It cannot be used
to claim the physical l=6/8 dynamics are preserved simply because those
wave-extraction outputs exist.

Recovery fallback is also explicit model intervention: it extrapolates failed
points from interior states, resets remaining failures to atmosphere, and
sets pressure to zero in the surface band (`HWMRecovery.h:3–16,47–159`). The
latest main loop itself specializes h=1 and returns eps=p=0
(`Hydro_with_Maxwell.C:1838–1853,1931–1934`). These paths are useful examples
of preserving ordered dependencies during a GPU port, but not a template for
PyFluid's conservative admissibility treatment. `HWMRecovery.h:11–16` records
that an earlier OpenMP loop raced on those dependencies; the port instead
serializes radius within independent angular rows.

`particle_surface_ex` has a different research branch: a smooth source window
(`Hydro_with_Maxwell.C:45–52`), density cut-cell redistribution, and explicit
surface-curvature projection at relaxation weight 0.1
(`Curvature.C:66–113,879–885`). Those operations modify source/curvature or
redistribute density and require their own conservation/convergence study.
The branch's own commit subjects include failed attempts and an untested
cut-cell method. Do not port them as accepted fixes for our AMR interface.

### Elliptic and horizon limits

`ConstraintSolver.h:75–81` explicitly rejects nonzero a_rr rather than
supporting general nonzero extrinsic curvature, then applies underrelaxed
Hamiltonian corrections (`:102–125`) and optional precollapsed lapse
(`:131–139`). The limitation is in this generic helper; other ID classes have
their own solvers. `EllSolver3D.h:6,85,102–104` labels itself old with two
hardcoded ghost zones. The second-order FlatEllSolver3D2ndOrder owner is
distinct. Epetra MPI maps distribute the sparse solve; these sources do not
provide AMReX-like distributed evolution ownership and halo exchange.

`HorizonFinder.C:98–143` iterates a star-shaped surface with projected fields,
an elliptic correction and xi=0.5 mixing. `Excision.h:107–120` activates a
fixed radial cut by time or stellar-surface crossing, and `:123–145` fills its
interior by constant/linear extrapolation. Source presence does not establish
that every characteristic points out of the evolved exterior at that cut.
Retain PyFluid's separate horizon tracking, excision characteristic and
one-sided derivative contracts.

### Constraints, waveform definitions and independent comparison limits

The Hamiltonian source uses K+trace(A)+2Theta and separately stores a local
denominator from the quadratic sum of K/R/rho contributions
(`Diagnostics.C:395–419`). Momentum derivatives likewise include K+2Theta
(`:523–543`). Thus its state member `K` is not automatically physical K in
the Z4-enabled path. In contrast, the current Weyl assembly reconstructs K_ij
with `S.K` at `:1109–1114,1226–1227`; whether that is intended for nonzero
Theta requires a formulation-specific audit. Do not accept the waveform
assembly as a universally verified Z4/fCCZ4 oracle.

The coordinate tetrad uses metric Gram–Schmidt
(`DiagnosticsBlocks.h:18–49`). Psi4 has an explicit overall sign change to
the convention intended to match metric eta (`Diagnostics.C:1416–1428`).
The qK route is currently unavailable in this source: the active routine
rejects `use_Kinnersley_tetrad` (`:1022–1027`), initializes qk_ok=false
(`:1350`), and writes qK fields as copies of ordinary Psi4 (`:1427–1428`).
Equal qK and ordinary output columns therefore do not validate tetrad
equivalence. Earlier branch implementations remain historical references.

Projection in `ExtractionBlocks.h:23–32,69–118` only extracts real m=0,
even l=2,4,6,8 modes; its quadrature multiplies by 2 pi and optionally by
equatorial factor 2. It also computes an areal-radius diagnostic. Strain is
obtained by sequential direct time sums
(`WaveExtraction.h:251–264`), and luminosity by
r_areal^2/(16 pi) sum hdot_l0^2 (`:322–333`). This is not FFI or a general
complex all-m waveform pipeline. Zero-frequency drift, initial integration
constants and finite-radius/gauge effects still matter.

`SurfaceIntegralDiagnostics.h:56–76` clamps requested surface radii to saved
grid indices and computes an axisymmetric area; its recorded used radius
must accompany comparisons. The northern-hemisphere z momentum in the
equatorial setup (`codex_11_Aug_2026.txt:17–30`, source route
`Diagnostics.C:819–875`) is not a nonzero full-sphere conserved linear
momentum. PyFluid scalar conservation gates should not adopt those labels
without the corresponding domain/weight definition.

### GPU/build lessons without framework replacement

`tensors.h:18–30,48–74` stores small vectors/tensors inline, avoiding the
original heap allocation. `Interp.h:5–11` removes a shared mutable index-search
starting guess, uses fixed scratch and shares Neville interpolation. Both
are clear examples of separating mathematical point operations from storage
and launch ownership, consistent with PyFluid's current architecture direction.

Field-list kernels batch three ordered ghost phases and array operations
(`gridfunction.h` GFList owners; `ROADMAP.md:158`) rather than instantiating a
new physics algorithm. The port docs report roughly halved launch counts;
these are collaborator measurements, not PyFluid performance predictions.
`LoopTimer.h:3–11,17–46` fences timer readings and separates evolution from
output and other work. The output docs show why timing only kernels can
misrepresent end-to-end runtime. Our compile ledger should similarly keep
single-owner latency, total compile work and complete-build timing separate.

The source's reference build disables contraction/fast math and its production
build enables aggressive flags (`CMakeLists.txt:15–25,235–272`). Its GPU
comparison tolerances are calibrated against a different CPU build and have
explicit noise/printed-digit fallbacks (`tools/delta/compare_runs.py:1–45`).
Do not import those tolerances, weaken PyFluid arithmetic, or replace analytic
and scalar-conservation gates with transcription equivalence. CUDA/compiler
restrictions in `Kernels.h:13–21` and the docs are specific to this Kokkos
header-lambda implementation; they are not a general AMReX CUDA requirement.

The newest independent output scheduling is restart-derivable from step and
the interval (t_prev,t] (`OutputSchedule.h:3–6,38–70`), warns when a step
passes multiple output times, and has per-output schedules. This is a
conceptual reference if PyFluid needs phase-dependent output cadence; it does
not justify a second input catalog or a parallel field registry.

## Proposed PyFluid work order

1. Preserve the now-pinned external reference and route it in references.md
   with a clear read-only/provenance boundary.
2. Add a small independent tensor parity/basis test set using smooth Cartesian
   scalar/vector/symmetric-tensor fields sampled through the spherical map.
   Exercise origin antipodes, both poles, equator, every component and
   derivative index. This can run without a C++ compile.
3. Record a centering/regularity architecture decision before enabling full
   origin/pole fCCZ4 evolution. Benchmark any alternative in a narrow numerical
   owner rather than multiplying the complete physics build matrix.
4. Extend discrete mapped-operator manufactured tests through the chosen
   boundary closure; test map derivative accuracy independently from field
   stencils. Keep physical spacing in CFL and volume/area metrics in FV fluxes.
5. Validate uniform-grid spherical vacuum/matter problems first; retain
   analytic constraints, scalar budgets and actual mode amplitudes. Add
   dynamic AMR only after the same state/interface owner is sound uniformly.
6. Treat axisymmetric electrovacuum coupling and waveform/tetrad adaptations
   as separate research capabilities with their own equations and tests.

No solver formula, centering, AMR transfer, damping parameter, tolerance,
physics capability or supported-domain check has been changed by this review.

## Demonstrated PyFluid candidate: missing theta reflection at fluid origin

The parent identified and this review independently source-checked a concrete
topology defect in the existing cell-centered fluid boundary owner. For a
spherical point, (-r,theta,phi) is the same physical location as
(r,pi-theta,phi+pi). `boundary.hpp:1163–1167` triggers the nonlocal antipodal
phase only if phi is active; `boundary.hpp:1212–1227` mirrors only the radial
index and shifts phi, leaving theta unchanged. With active `r theta`, that
phase does not run, and the ordinary radial reflection also keeps theta.
`config.cpp:2589–2602` allows spherical origin boundaries without imposing
equatorial symmetry. `fields.cpp:216–250` supplies the expected spherical
local vector sign transformation but cannot correct source-point identity.

The smooth positive axisymmetric scalar rho=1+0.1 z demonstrates the defect
without any vector convention ambiguity: at r=-0.125 and theta=pi/6, analytic
rho is 0.9891746824526945; the current same-theta positive-r source is
1.0108253175473054, an error of 0.02165063509461096. This source-index
counterexample is saved in `origin_source_counterexample.json`. It is not a
runtime solver result. Existing origin validation primarily covers cylindrical
`r phi` or radially symmetric spherical states and does not distinguish this
case. Full `0..pi` theta domains require a nonlocal polar reflection across
box/MPI ownership; half-theta equatorial domains require explicitly composed
equatorial symmetry. A local FAB index reversal is not sufficient generally.

Cycle479 subsequently compiles the real boundary owner and reproduces all eight
2D/3D full/half-theta HD/MHD failures over multiple boxes and three ranks. Values
remain finite, but full-sphere scalar error reaches 0.0373194 and vector parity
also fails. This is direct boundary evidence, not evolution, CT or partial-AMR
acceptance. Evidence is under `runs/current/infrastructure_audit/cycle479/origin/`.

The existing `AmrSolverBase::complete_radial_origin_boundary` already materializes
a positive-radius donor shell through `FillPatchNLevels` with the selected state
interpolater and time slices. Retain that owner for missing fine-level donors;
correct its shared angular map together with the uniform boundary owner. The
complete repair remains pending. This defect is independent of unsupported
vertex-centered spacetime singular nodes.

## Focused architecture comparison: centered spacetime

This revisits the already documented Cycles 417–423 proposal and the user's
current request to remove unnecessary vertex-to-cell work without creating a
second solver or a large template matrix. It is grounded in the current
`EvolutionSectorStorage`, `SpacetimeCouplingStorage` and stage coupling sources.
This review makes no source edits to those owners.

### What the collaborator's centering actually solves

Its r and theta point samples are half a logical cell from the coordinate
origin/axes (`Grid.h:446–449,535–549`). Thus no evolved spacetime point sits
exactly at r=0 or sin(theta)=0. Reference rescaling plus correctly identified
ghost points allows the same bulk finite-difference equations to be evaluated
at every valid point. There are no physical-node angular aliases at the origin
or poles that need to be projected to one common tensor value. This is an
established coherent layout choice, and it avoids the extra singular-node
Taylor/limit machinery considered for retained vertex sampling.

It does not by itself solve small-radius/angular CFL limits, cancellation of
large geometric terms, tensor parity, ghost communication, interpolation
accuracy, fluid face geometry or AMR. Its own default timestep is limited by
r*dtheta (`Manager.h:279–298`), its moving EM shell has an additional map and
matching discretization, and its `Regrid` never creates a coarse/fine hierarchy.
The documented latest GPU acceptance is axisymmetric plus equatorial symmetry;
it is not evidence of full 3D origin/pole passage.

Co-locating spacetime point values with fluid cell centers can remove the
spacetime-vertex-to-cell state transfer and the cell-matter-to-vertex transfer.
It cannot make a point value a finite-volume average. Metric reconstruction to
fluid faces remains necessary, and matter projections formed from FV averages
are still an approximation to a pointwise source. Centered evolution must
validate that coupling accuracy rather than assuming exact identity from
array co-location.

### Current PyFluid ownership that must change coherently

| Current owner and exact source | Present assumption | Resolved centered behavior |
|---|---|---|
| schema/fields.toml; field_catalog.py component model | conformal spacetime centering is Vertex in the sole schema | same field identities with one setup-resolved centering; preserve point-sample meaning |
| EvolutionSectorStorage::configure, evolution_sector_storage.cpp:13–73 | derives common centering and copies the schema field set | validate one resolved field set/layout; use it for all stage storage and public field descriptors |
| EvolutionSectorStorage::allocate_level, :148–201 | converts cell boxes using the stored common centering | existing generic allocation already supports a centered IndexType |
| EvolutionSectorStorage::average_down_level, :441–462 | Vertex implies nodal restriction; anything else implies FV average_down | dispatch from sampling/transfer meaning as well as topology; centered spacetime needs point restriction/prolongation |
| SpacetimeCouplingStorage::allocate_level, spacetime_coupling_storage.cpp:24–46 | always allocates cell metric, cell matter and vertex_matter | derive source/target layouts; avoid allocating unused converted buffers when state/source already co-locate |
| amr_solver_spacetime_evolution.cpp:82–149 | stage ghost fill/project, then unconditional V2C into cell_spacetime | same stage checks, direct read view of stage cell spacetime when selected; convert only when layouts differ |
| amr_solver_spacetime_evolution.cpp:205–313 | forms cell matter, fills it, then unconditional C2V into vertex_matter | form/fill the same Eulerian matter; bind it directly when the spacetime target is cell centered |
| ReferenceChartGeometry::at, reference_chart.hpp:55–60 | default nodal=true | pass resolved sample location explicitly; geometry map stays independent of field layout |
| reference_chart.hpp:63–82, spacetime masks/diagnostics | nodal dual volumes and nodal domains/validity | derive domains, duplicate ownership, boundary planes and integration weights from resolved layout |
| initial data, output and restart consumers | use the schema's existing vertex representation | consume the same resolved descriptors; materialization evaluates target sample points and metadata records layout/sampling semantics |

The present explicit `vertex_matter()` API name reveals the hardcoded target;
rename its semantic role to spacetime-target matter only as part of resolving
that contract. Do not keep a vertex-shaped buffer and relabel its use at
individual call sites. Likewise an existing `cell_spacetime` cache should not
force a duplicate copy of an already centered stage state. Prefer returning a
const view/reference of the correct owner while retaining explicit stage and
halo lifetime validity. No new physics state should be introduced by a cache.

### First coherent implementation stage

Implement a small infrastructure stage before exposing `spacetime.centering`:

1. Declare the needed sampling meaning in the existing field schema/model,
   distinguishing spacetime point samples from fluid conserved cell averages.
   Resolve its default centering and numerical transfer policy once during
   sector setup. Keep field names, component IDs and default Vertex declarations
   in the existing catalog; do not add a parallel capability/field registry.
2. Pass the resolved field set/layout through existing sector storage and
   coupling allocation. Use the actual resolved IndexType for domains,
   chart sample locations, output descriptors, validity regions and integration
   weights. At this first stage the selected behavior remains the existing
   vertex baseline, so its outputs should remain identical.
3. Make restriction/prolongation choose the point-field versus FV contract
   explicitly. Initially verify existing vertex point restriction and fluid
   average-down remain unchanged; implement a centered point-field owner before
   enabling the public cell option. Reuse the existing interpolation-weight and
   topology machinery, but do not route centered spacetime through hydro's
   conservative average-down merely because both are cell centered.
4. Make stage coupling query source/target layout once on the host and select
   direct access versus the existing transfer. Keep numerical conversion in
   the one current transfer owner, and preserve scalar positivity representation
   where actual interpolation is performed. Centered arrays should bind
   directly with explicit stage/halo validity, avoiding two redundant buffers.
5. Only then expose the startup-only vertex|cell selection, with vertex default,
   once allocation, initialization, FillPatch/regrid/restriction, physical
   boundaries, constraints, output and restart all honor the same selection.
   Validate Cartesian vacuum/coupled uniform controls, AMR/restart and physical
   accuracy before adding the separate spherical origin/pole topology stage.

This is one layout/transfer refactor, not two equation paths. The generated
Z4c/fCCZ4 point kernels, derivative formulas, RK catalog, primitive recovery,
Valencia equations and all Riemann solvers remain single owners. A startup
enum/host dispatch or uniform location offset does not need a new centering
template parameter propagated through reconstruction, Riemann methods or
recursive AMR orchestration. Keep only genuinely topology-specific AMReX
calls in the corresponding launch/communication owners. Use the active focused
development profile to measure each changed owner; integrate complete builds
at milestones rather than recompiling every method for a metadata edit.

### Variable mapping before a cross-code comparison

| Reference state | PyFluid interpretation to verify |
|---|---|
| phi with exp(4 phi) conformal factor | chi=exp(-4 phi) when cosmological scale factor is one; convert the complete metric and derivatives if a(t) is active |
| h_ij perturbation relative to flat spherical reference metric | PyFluid stores full conformal tensor components in its declared reference frame; in the corresponding orthonormal frame the diagonal is 1+h_ii, not h_ii |
| reference rescaled a_ij | compare A_tilde only after conformal factor, r/sin factors, physical frame and trace-projection conventions agree |
| stored K and Theta | confirm K_hat versus physical K=K_hat+2Theta in every evolution/constraint/Weyl consumer; collaborator constraints add 2Theta but some Weyl assembly uses K alone |
| lam_r,lam_t,lam_p | reference Lambda coordinate components are lam_r,lam_t/r,lam_p/(r sin theta); PyFluid connection slots have formulation-dependent Lambda/Gamma meaning and runtime-axis permutations |
| separate B_r,B_t,B_p Gamma-driver auxiliary state | PyFluid runtime currently uses an integrated driver and has no corresponding extra evolved GF; names do not imply state equivalence |
| 1/(4 pi) magnetic stresses | convert to PyFluid's magnetic normalization before conserved/source comparisons |

The collaborator calls its switch BSSN/Z4c, while PyFluid has independently
owned Cartesian Z4c and reference-metric fCCZ4 equations. Matching names,
rescaled storage patterns or cell locations does not establish identical
formulations. Compare the pointwise equations and conventions before asking
whether the resulting simulations agree.
