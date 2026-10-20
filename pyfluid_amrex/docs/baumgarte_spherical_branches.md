# Baumgarte spherical branch and evidence map

Companion to [the source review](baumgarte_spherical_review.md).
All remote access and local reference source remain read-only under `AGENTS.md`.

Read-only snapshot of 2026-10-20. Commit identities are in `branch_pins.tsv` and
the full path/blob inventory is in `reference_inventory.json`. This map records
implementation history and research directions, not independent run validation.
No reference program or test was executed.

`reference_source_coverage.tsv` contains 1,487 path/blob entries (1,014 distinct
paths, 1,152 distinct blobs over 18 heads). There are 67 distinct blobs with
selected numerical source sections read in this review; other entries explicitly
identify interface/owner survey, historical alternative or excluded generated
artifact coverage. Neither that inventory nor branch enumeration is a claim of
a complete formula audit of all historical variants.

## Branch ancestry

The latest reviewed head is `yguo/dev/output-validation` at
`2ee5550396be2c56fc1c54f9c2ac859edeb9f096`.
The following 11 other heads are ancestors of that tip: `main`, `rescale_Bi`,
`MHD_rescaled_Bi_smooth_surface`, `excision`,
`EM_ICN_only_theta_dep_EM_ex_BCs`, `better_Psi4_tetrad`,
`better_Psi4_tetrad_anvil`, `version_10_Aug_2026`, `version_27_Aug_2026`,
`yguo/dev/reference-profile`, `yguo/dev/output-control`.

Six heads have commits absent from latest. Their exact commit lists and changed
core paths are saved in `reference_branch_delta.json`. The count is a Git
ancestry fact; some equivalent changes reached latest through other commits,
so a non-ancestor tip does not prove its feature is absent.

| Branch | Distinct commits outside latest | Useful source content | Boundary and adaptation decision |
|---|---:|---|---|
| main | 0 | original mapped spherical reference-metric state, derivative/ghost/elliptic/ID library | old snapshot; use later fixed owners for learning current behavior |
| rescale_Bi | 0 | factored magnetic angular components and flux/source cancellation work; history contains central magnetic-source/index corrections | cancellation depends on field ansatz; no general GRMHD/CT/AMR certification |
| MHD_rescaled_Bi_smooth_surface | 0 | option to turn interior MHD coupling on/off, magnetic flux/source alternatives | useful A/B idea; flags do not show independently validated physical closure |
| excision | 0 | fixed radial excision object, safe/one-sided derivative selection and KO lower-safe variants | inspect ghost reach and characteristic directions separately; no horizon-following causal proof |
| particle_surface_ex | 18 | surface particles, interior-only trajectory interpolation, density cut-cell redistribution, ADM source smoothing, curvature-surface projection | explicit research branch with failed/untested experiments in its own history; do not treat smoothing/projection as accepted physics fixes |
| moving_grid | 3 | separate moving exterior EM grid, stellar surface matching, wave cadence and constraint monitors | most important moving-grid lineage also appears later; source fixes cover RK copies, surface cell placement, matching each substage and restart |
| EM_ICN_only_theta_dep_EM_ex_BCs | 0 | theta-dependent surface BCs, split ICN/RK exterior evolution and angular EM terms | time/angle matching choices are model-specific; must retain actual field scaling |
| interior_only_B_field | 9 | confined interior field research using a_r,a_t,a_p and an additional toroidal magnetic representation | broader potential ansatz is distinct from latest two-field HWM; parity/scaling comments differ, no verified CT or long-time result |
| better_Psi4_tetrad | 0 | metric-normalized tetrad, quasi-Kinnersley experiment, Zerilli/Moncrief source modes, independent EM flux diagnostics | later active GPU owner rejects qK and copies qK columns; old implementation is a formula reference, not current accepted route |
| better_Psi4_tetrad_anvil | 0 | Weyl derivative-index change, eta modes, RNS reader and surface diagnostics | source change titles are hypotheses unless independently checked; audit physical K vs stored K with Theta before cross-code use |
| commit_7a6a1e9_13_July | 1 | named historical executable for source commit 7a6a1e9 | unique commit adds an executable only; binary not executed or treated as scientific evidence |
| version_10_Aug_2026 | 0 | historical integrated gravity/OS/RNS/wave/source snapshot | latest Kokkos branch descends from this family; do not replace current source with archived variant |
| fix_E_osc_issue_2 | 7 | adjustable even Legendre surface filter for a_p/e_p; older oscillation-debug lineage | lmax maps to modes 0/2/4, so filtering removes physical surface content; history itself notes restrictive behavior at stronger B; no blanket stability result |
| version_27_Aug_2026 | 0 | later integration of surface filtering and physics fixes | branch name is not a capability proof; inspect active formulas and new code separately |
| Magnetized_TOV | 9 | unique TOV+magnetic ID provider, general thermal Palenzuela closure, z=D*x recovery correction and magnetosonic-bound experiment | August handoff reports unresolved B+pressure origin instability; later fixes exist but no independently inspected resolution evidence; do not call equilibrium GRMHD support |
| yguo/dev/reference-profile | 0 | Kokkos OpenMP/CUDA port, fixed-size tensor/scratch, field-list kernels, branch coverage and phase profiling | acceptance recorded mainly for dust magnetized OS with axisymmetry/equatorial symmetry; other matter paths/GPU tetrad excluded |
| yguo/dev/output-control | 0 | runtime output schedules, separate ray/slice lists, HDF5 event output, independently scheduled checkpoints/particles | conceptual cadence/metadata reference; preserve PyFluid catalogs and restart semantics |
| yguo/dev/output-validation | — | latest fenced compute/output/other timings and accompanying output-deck evidence | collaborator-recorded performance, no local remeasurement and no implication for PyFluid compile/runtime cost |

## Non-ancestor source details

### Magnetized_TOV at eda44cc851e12eeef6a561891c1c140eec4ea0f7

`SourceFiles/INDATA/TOV_Magnetized.h:4–8` explicitly calls itself a first trial:
a spherical TOV background for density/lapse/conformal factor with the
magnetic prescription from magnetized OS. Constructor controls include central
density, magnetic amplitude, optional pressure depletion and optional
Hamiltonian correction (`:97–109`). It builds an ordinary spherical TOV ODE
solution (`:111–116`) and optionally derives the EM threshold radius from
the TOV density (`:119–130`). `:156–175` applies the optional constraint solve.
This is not a full self-consistent magnetohydrostatic force-balance solver.

The branch's recovery solves x-h(rho,eps)W=0 with EoS cold-energy floors
(`SourceFiles/Hydro_with_Maxwell.C:2218–2269`). Commit eda44cc corrects
z=D*x*W to z=D*x (`:2288–2294,2317–2333`), consistent with x=hW and
D=rho W. That is a valuable variable-definition lesson. It does not validate
all brackets, floor interventions, fallback or strong-field states.

Commit 87479be disables quartic magnetosonic roots behind a literal diagnostic
switch and returns a robust bound instead. That is an isolation experiment,
not evidence that this choice resolved the underlying central instability.
The August handoff (`codex_28_Aug_2026_magnetized_TOV_handoff.txt:8–27`)
reports spurious radial velocity only for nonzero magnetic field plus
pressure, with flux/source cancellation and characteristic speeds as suspects.
Treat those as collaborator observations/hypotheses. The source's later
recovery fix is confirmed; independent numerical resolution is unverified.

### particle_surface_ex at 83517644bd312015d70ae4bf2ea320c8c957492a

Source-level features include a quintic source taper
(`SourceFiles/Hydro_with_Maxwell.C:45–52`), particle-based cut-cell density
redistribution, and a surface curvature projection using a shared surface value
and minmod slopes with relaxation 0.1 (`SourceFiles/Curvature.C:66–113`). These
are interventions requiring their own physical/discrete budgets. The commit
subjects expressly include failed surface-aware attempts and a cut-cell method
not yet inspected/tested; source presence is the evidence boundary here.

The extra `SourceFiles_surface_particles_theta_tracking/` directory has 165
paths. Hash comparison against every `SourceFiles/` blob finds only three
novel blobs: Hydro_with_Maxwell.C/.h and Particles.h. Their diffs show an older
surface-only prototype that lacks later excision/cut-cell/interior-only
interpolation refinements. The other 162 paths are identical to blobs already
reachable through ordinary source paths. That archive should be routed as
history rather than independently promoted as another solver implementation.

### interior_only_B_field at cee7c7aa510c2ddd9dfcaceab1d6c5ba98f76fb6

`SourceFiles/Hydro_with_Maxwell.C:24–43` documents a different potential
ansatz with a_r,a_t,a_p and a factored toroidal component. The state stores
all three potential fields (`Hydro_with_Maxwell_State.h:29,63–67`), and the
evolution uses their derivatives (`Hydro_with_Maxwell.C:3020–3037,
3259–3277`). Potential-value and rescaled-component comments overlap and
must be reconciled with actual formulas before use. It remains a reduced
ansatz, not three-dimensional arbitrary-potential CT or divergence-preserving
AMR transfer. No numerical acceptance was inferred from the branch.

### fix_E_osc_issue_2 at 87519aa3051055e9f48668d357c370007cb1bbd6

`SourceFiles/Hydro_with_Maxwell.C:814–832` allows disabling filtering for
negative lmax; nonnegative choices retain one to three even modes, and
`:895–897` reconstructs c0+c2 P2+c4 P4 and its angular derivative. Filtering
therefore changes the boundary model and removes modes. It is not merely an
accuracy/performance setting. The latest source includes the same filter family
despite differing ancestry; no cherry-pick is recommended.

## Best routes for a later comparison

1. Start with independent smooth Cartesian fields projected into spherical
   components and test exact ghost-point identity/parity. This review already
   exposed the missing theta reflection in PyFluid's fluid origin fill.
2. Compare mapped derivatives and reference-covariant metric jets at regular
   points with explicit conventions. Retain arbitrary off-diagonal tensors and
   derivative-index rotations, not merely radial or symmetric fields.
3. Run uniform-grid regularity/manufactured convergence before a star, and a
   known spherical equilibrium/dust limit before thermal magnetic evolution.
   Do not infer a validated full-star origin/pole method from shell evolution.
4. Compare actual local characteristic speeds, finite-volume measures,
   conservative boundary/source totals and recovered physical variables. A
   frozen-source bitwise comparison establishes transcription only.
5. Any wave comparison must first match K/Theta, Weyl sign, tetrad,
   magnetic normalization, symmetry/mode set, extraction radius/time and
   direct integration versus FFI. Existing qK duplicate columns are not an
   independent comparison.
