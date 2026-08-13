# Unified Shock-Capturing Research

Status: research design and falsification record. No method in this document is
a validated production default, and no novelty claim is made.

## Objective

The target is one reconstruction and one Riemann policy for Newtonian HD/MHD,
RHD, and the local face problems in GRMHD. "One" means a common mathematical
operator with physics providers, not one scalar formula that ignores the
different admissible sets and wave families. The desired operator should:

- retain fifth-order smooth accuracy and smooth extrema;
- contract continuously through robust lower polynomial modes when the high
  modes are inadmissible;
- preserve coupled contact, shear, Alfven, slow, and fast wave structure;
- preserve positive density and pressure and subluminal relativistic velocity;
- keep the normal magnetic field single-valued at a CT face;
- approach a robust HLL-type limit continuously at degeneracies;
- provide a defensible entropy statement where the evolved equation system
  actually admits one; and
- avoid a separately selected runtime reconstruction/fallback ladder.

Godunov's order barrier means that no linear scheme can be simultaneously
higher than first order and monotone. "Best everywhere" therefore cannot mean
zero dissipation, zero overshoot, maximum discontinuity sharpness, and
unlimited order under every norm. The research target is a demonstrably better
Pareto front with explicit invariants and no hidden terminal repair in normal
operation.

## Current Production Boundary

The current strongest verified RMHD policy remains PPM4 with the RK-stage
candidate DMP/PWC correction at factor 1.05 when low ringing is the priority.
On exact Balsara 4 at 1024 cells it has maximum normalized all-component L1
`2.797194e-3`, overshoot `4.524195e-3`, TV ratio `1.4913`, 2236 PWC
corrections, and zero atmosphere resets. A research candidate must beat this
and the existing MP5 accuracy controls on held-out cases before it is added to
the production method catalog. The L1 value uses the Cycle-247
discontinuity-aware average of the exact conserved state over each numerical
cell; the older point-sampled comparison reported `3.200159e-3`.

The external comparison is narrower than that production statement. On a
matched N=1024 Balsara-4 run at CFL 0.9, PyFluid MP5/PWC beat AthenaK PPM4 in
the chosen conserved all-component L1/overshoot/TV metrics. On a separate
matched 50-cell Balsara-4 control, PyFluid PPM4/PWC and AsterX PPM each won
some primitive-density/`By` metrics and were only comparable overall. Neither
experiment covered the narrow Balsara-3 density shell shown below. AsterX's
published Balsara-3 plot uses MINMOD and reports only primitive `rho` and
`By`; no matched AthenaK/AsterX Balsara-3 conserved-state comparison exists.
Accordingly, "competitive on matched Balsara 4" must not be restated as
"equal to or better than those codes overall."

## Course Correction: Production Causality First

The broad fan-lifecycle work below is now parked as mechanism and negative-
control evidence. It is not the immediate implementation roadmap. The
production trigger was two separate flat-SRMHD symptoms, and they must not be
collapsed into one reconstruction problem:

- Balsara 4 is an accuracy/ringing case. At N=256, PPM4/HLL/SSPRK3, CFL 0.9,
  and the PWC candidate guard at DMP factor 1.05, the final conserved-
  cell-average metrics are L1 `9.805072e-3`, overshoot `9.955165e-3`, and TV
  ratio `1.327840`, with 574 PWC corrections and no recovery event.
- Balsara 3 is an admissibility/recovery case. At N=128, unguarded MP5 reaches
  three atmosphere resets. The guarded control has zero resets; its purpose is
  not to rank ringing on Balsara 4.

`tools/diagnose_rmhd_balsara_onset.py` writes every accepted state and, when
`pyfluid.debug_stage_output = 1`, the three raw SSPRK3 candidate states plus
the post-projection accepted state. The option is off by default, requires
SSPRK3, and currently supports one uniform AMR level. It changes no flux,
candidate, recovery, or accepted-state operation.

The first-step causal replay gives a sharper boundary than final profiles:

- At the initial aligned Balsara-4 jump, one Euler stage is identical for
  global PPM4, guarded PPM4, guarded PPM4 with LLF correction, and global PWC.
  The first SSPRK3 stage and the stage-2 convex candidate have no measured
  global overshoot or plateau reversal. The first surviving variation appears
  in stage 3. Unguarded PPM4 has overshoot `1.536317e-3` and TV ratio
  `1.242834`; the PWC guard removes the global overshoot and lowers TV to
  `1.138395`, but leaves a normalized `mom2` plateau reversal of
  `9.07898e-4`. Global PWC and guarded MP5 have stage-3 TV ratios
  `1.118963` and `1.104608`. Atmosphere projection changes none of these
  stage-3 states.
- In unguarded Balsara 3, accepted step 2 begins with a stage-1 candidate whose
  overshoot is `0.805053` and TV ratio is `2.489339`; the standalone offline
  recovery also fails in one cell. The projection before the next stage
  records the production atmosphere reset. The guarded stage-1 candidate has
  roundoff global overshoot, unit TV, and no reset. This is direct evidence
  that the existing conservative guard addresses the recovery symptom.

`tools/audit_srmhd_physical_wave_balsara.py` then applies the standalone
two-plateau/full-Rankine--Hugoniot detector to those production stage states.
It rejects both initial compound jumps and every guarded Balsara-3/4 stage.
The only nominal acceptance is a far-field, unguarded Balsara-4 face after
stage 3 has already created the artifact; its relative conserved jump is only
`6.18e-15`. Thus the present isolated-front carrier is inactive before the
production failure and cannot be promoted as its repair. Exact Euler fans,
the invariant-subsystem MHD fan, persistent HLLD controls, and physical-wave
oracles remain useful falsification harnesses only.

Do not call the production PPM4/PWC or MP5/PWC candidate guard the
physical-wave prototype. The former is a conservative high-order/PWC flux
replay selected from a concrete RK candidate and has no persistent subcell
front. The latter is a standalone Python mechanism with a strict
two-plateau/full-RH subcell detector and crossing-time carrier. It is not shown
in the production profile plots because the audit above fails the required
causal activation test.

### Hybrid comparison contract

Every proposed hybrid `A/B` must be shown beside the same exact or independent
reference, pure `A`, and pure `B`, with identical grid, CFL, timestep, Riemann
solver, output time, and scoring. Report the hybrid's activation count as well.
If a pure branch fails, record the failure or last valid state rather than
silently omitting it. Without this four-way matrix, one cannot tell whether the
hybrid preserves `A`, merely reproduces `B`, or improves on either constituent.
Use stage-local matrices only to locate causality. Rank a hybrid with the full
accepted evolution, because a guard may intentionally reduce to pure `B` while
the discontinuity is unresolved and diverge from `B` only after structure
spreads and `A` becomes admissible again.

For the current saved runs, the multi-time spatial view uses Balsara-4 steps
10/28/57 and Balsara-3 steps 5/14/29. It plots the exact FV average, pure `A`,
pure `B`, and the existing `A/B` guard in the self-similar coordinate `x/t`.
The methods' saved times agree to `3.1e-10` in Balsara 4 and within
`4.4e-5` in Balsara 3; the final times are identical.

Density must be shown alongside the component used to localize the first
artifact. Its final-time result prevents an overly favorable reading of the
aggregate metrics:

| Case and density metric | pure high order | global PWC | high order/PWC guard |
| --- | ---: | ---: | ---: |
| Balsara 4 normalized L1 | `1.10882e-2` | `2.02877e-2` | `9.80507e-3` |
| Balsara 4 TV ratio | `1.49961` | `0.76277` | `1.32784` |
| Balsara 3 normalized L1 | `7.85467e-3` | `1.09283e-2` | `9.92183e-3` |
| Balsara 3 maximum `D` | `8.0785` | `2.9060` | `5.2175` |

The exact Balsara-3 FV density reaches `20.7419` in its narrow right feature.
Raw MP5 has the lowest density L1 of the three but incurs three atmosphere
resets and a density overshoot; the guard removes those failures while
smearing the peak more. The PWC branch is still more diffusive. This is a real
robustness trade, but none of the three is a clean high-fidelity solution at
this resolution.

That failure has two distinct parts: subcell representation and flux-family
diffusion.
The exact primitive table identifies the left edge at `x=0.381160560` as a
contact: density jumps from `0.0979823515` to `10.1039427`, while total
pressure, all velocities, and both transverse magnetic fields remain equal to
the table precision. The high-density state terminates at another MHD jump at
`x=0.392556640`. Its width `0.011396080` is only `0.729` of one N=128 cell on
`[-1,1]`; it spans `1.459`, `2.917`, and `5.835` cells at N=256, 512, and
1024. No conventional homogeneous-cell FV reconstruction can store two
interfaces and the state between them inside one cell. The N=128 figure is
therefore appropriate for locating recovery but not for judging contact
sharpness.

At N=1024 the feature becomes representable, but the two-wave RMHD flux used
in the pre-HLLD runs remains a bottleneck. Saved guarded WENO-Z, MP5, and TENO5
runs reach maximum conserved density `27.6303`, `28.1006`, and `28.1327`
against the exact FV `34.0283`. This is partial resolution, not a sharp shell.
It motivated the controlled contact-resolving-flux experiment below; no flux
can by itself recover a shell narrower than one homogeneous cell.

Cycle 262 closes that implementation gap. Production AMReX now provides the
Mignone--Bodo HLLC fan and the Mignone--Ugliano--Bodo five-wave HLLD fan, with
an audited HLL inadmissibility limit. One local orthonormal face frame applies
the flat-SRMHD solve to fixed or evolved GR metrics. HLLD preserves the exact
isolated contact and rotational control, passes CT/AMR and curved-spacetime
gates, and uses no HLL limit in the five N=512 Balsara runs. Its Balsara-3
N=1024 all-component error improves only `0.004952 -> 0.004934` relative to
HLL, and its conserved-density peak is `26.728` versus exact `34.0283`.
Therefore complete-wave flux diffusion was real, but it was not the sole shell
limitation. HLLC is retained only as a contact/algebra control because its
contact antidiffusion shows a bounded rotated strong-shock transverse defect;
HLLD is the production multidimensional RMHD choice.

The high-resolution profile matrix reinforces the separation. At N=512,
exact FV `D_max=34.0283`; MP5/PWC reaches `19.5114`, while the removed
HARM5/PWC and SP-MP5/PWC reconstruction prototypes reach only
`14.2114/14.5365`. At N=1024, MP5/PWC reaches `28.1006` and HARM5/PWC only
`24.2244`. Raw MP5 is both less accurate in the peak (`17.0764/20.3745`) and
invalid as a clean comparator because it records 3/35 atmosphere resets at
N=512/1024. The candidate guard earns its robustness role but not a
sharp-contact claim. The persistent subcell carrier produces no Balsara curve:
its detector never activates causally on these compound fans, so plotting a
modified profile for it would fabricate an effect it did not have.

### Status of the distinct research objects

| Object | What is established | What is not established |
| --- | --- | --- |
| Production candidate guard | Conservative shared-face replay prevents the present raw recovery failures and is less diffusive than global PWC | Clean nonoscillatory Balsara profiles or a new universal method |
| Production RMHD HLLD | Contact/rotational fan, HLL limit, curved GR frame, CT/AMR, and exact-profile gates pass | A cure for under-resolved compound shells or a positivity proof at arbitrary magnetization |
| Mode-selective and THINC/BVD reconstruction studies | Useful Newtonian Pareto and falsification controls | A dominant SRMHD/GRMHD reconstruction |
| Persistent physical-front carrier | Near-exact isolated recognized contacts, shocks, and rotations | Timely birth on the present compound Balsara fans; production CT/AMR/GRMHD |
| Exact/causal fan and invariant-subsystem controls | Show what preserving complete space-time wave information can achieve | An oracle-free general MHD/SRMHD method |
| WCEM/COWE | Research specifications informed by rejected simpler candidates | An implemented or validated solver |

Accordingly, HLLD is now a production Riemann solver, but there is still no
successful novel reconstruction/Riemann prototype that solves the motivating
Balsara-3/4 profile problem. The guard remains a safety baseline, and the other
research objects must not be described as solving that symptom.

### Cycle 270 coupled finite-volume candidate

`tools/research_coupled_fv_reconstruction.py` is the final bounded prototype
from this research pass. It is one reconstruction core with system providers:

1. reconstruct an average-preserving quartic conserved polynomial in each
   cell;
2. recover primitives at three Gauss points and quadrature-average them rather
   than treating `C2P(Ubar)` as a primitive cell average;
3. use one vector-wide nonlinear/high-mode order, with an exact smooth plateau,
   for a continuous PPM4-to-MP5 face correction;
4. reconstruct velocity through spatial four-velocity, keep the normal
   magnetic field single-valued, and use local characteristic variables where
   the finite-volume representation remains well conditioned; and
5. for SRMHD, apply the existing factor-1.05 density/energy DMP and conservative
   shared-face PWC correction to every concrete SSPRK3 candidate. No atmosphere
   reset, local time subdivision, analytic exact state, or problem label enters
   the operator.

The exact smooth plateau is essential. Before it, a nominal weight near but not
equal to one accumulated measurable damping over many reconstructions. With
it, the candidate matches characteristic MP5 on the N=32/64 HD entropy wave
(`4.1893e-6/1.6248e-7`) and circular Alfven wave
(`2.7585e-5/2.5314e-6`). Per-wave sensing, binary PPAO selection, a PWL-first
face hierarchy, and an evolved-low-order DMP reference were independently
tested and rejected: each restored a known diffusion/oscillation failure.

The complete promotion matrix falsifies universal dominance:

| N=64, CFL 0.4 | coupled FV5 | best relevant control | result |
| --- | ---: | ---: | --- |
| HD Sod L1 | `1.6378e-2` | characteristic MP5 `1.2189e-2` | 34 percent worse |
| HD Lax L1 | `2.3167e-2` | characteristic MP5 `2.0355e-2` | 14 percent worse |
| HD blast L1 | `4.5672e-2` | characteristic MP5 `3.9464e-2` | 16 percent worse |
| NRMHD Brio--Wu reference L1 | `1.1975e-2` | characteristic MP5 `9.4047e-3` | 27 percent worse |
| SRMHD Balsara 3 L1 | `2.6325e-2` | MP5 `2.1640e-2` | 22 percent worse |
| SRMHD Balsara 4 L1 | `2.0921e-2` | PPM4 `2.7325e-2`; MP5 fails | 23 percent better |

The SRMHD robustness result strengthens at N=128. At CFL 0.4, Balsara 4
improves from PPM4 L1/overshoot/TV `1.6923e-2/0.2847/3.314` to
`1.2162e-2/0.0383/1.385`, while MP5 fails at `t=0.125`. At CFL 0.9, the
candidate completes both Balsara 3 and 4; PPM4 and MP5 both fail early on
Balsara 3, and Balsara-4 MP5 `2.2489e-2/0.3012/3.735` is worse than the
candidate `1.3607e-2/0.0573/1.549`. However, the guarded candidate's N=128
Balsara-3 L1 is `1.9740e-2` at CFL 0.4 versus MP5 `1.2000e-2`. The guard buys
the intended robustness by suppressing part of the narrow physical shell.

Therefore this is the best coherent SRMHD prototype from the pass, not a new
all-regime default. It remains outside the AMReX method catalog. Porting it
would add a costly numerical characteristic basis and primitive quadrature
while regressing simpler HD/MHD gates; optimization cannot repair that
algorithmic failure. The reusable findings are the exact smooth plateau,
representation-conditioned characteristic basis, and whole-state/shared-face
coupling. Production retains MP5/PWC for general accuracy and PPM4/PWC when
reduced RMHD ringing is the priority.

### Cycle 264 causal result

The remaining degradation is not one defect and is not primarily missing
Riemann waves. In Balsara 4, the first unguarded Forward-Euler candidate at the
two initial-interface cells gives `D=42.476` to the shown precision, equal to
the exact finite-volume value; its error is about `3.0e-8` of the error from
holding the stage input. The current factor-1.05 density/energy DMP rejects
those cells anyway because the exact system solution creates a new compression
extremum. Later reconstructions of the partially developed fan generate the
alternating defect and eventually unphysical candidates. In Balsara 3, the
same component DMP rejects beneficial early fan evolution and broadens the
narrow compound shell.

Changing the SSPRK bookkeeping does not repair this. A DMP applied to the
reconstructed Forward-Euler state is at least as restrictive, while including
the accepted-step base in the neighborhood bounds changes almost none of the
relevant flags. CT is absent from these one-dimensional runs, and HLLD changes
the magnitude but not the onset category. A Berta-style five-point detector
with its flag halo catches every DMP cell in both difficult cases; it detects
nonsmooth structure but cannot say whether that structure is physical. The
standard PPM shock sensor catches only a subset of Balsara-4 cells and misses
the early HLL Balsara-3 flags. The initial physical-specific-entropy minimum
also accepts the first complete Balsara-4 step despite its small surviving
zigzag, so admissibility or a scalar entropy floor alone is insufficient.

Component ablations expose a useful but non-promotable Pareto direction.
Checking only `tau` improves most HLL/HLLD L1 errors and sharply reduces the
Balsara-3 corrections. It also lets the HLLD Balsara-4 N=1024 density overshoot
grow from `0.01022` with the full guard to `0.07165`. Raising the density factor
or enabling density only under the PPM shock sensor has the same failure mode.
The full density/energy PWC guard therefore remains production behavior; none
of these experiments is a replacement.

The next prototype should not enumerate GRMHD wave families. It should retain
at least two accepted states and judge the current completed high-order
timestep candidate from that short space-time history, using physical
admissibility plus a discrete entropy/residual condition derived for the actual
SRMHD/CT formulation. Stage-local logic remains only the immediate
admissibility backstop. A rejection must replay the current step
conservatively through shared face fluxes and, in MHD dimensions, compatible
edge EMFs; it should not accept an artifact and attempt to erase it later. A
same-cell temporal DMP is also insufficient because a moving physical front is
naturally nonsmooth in time. This is a finite-volume MOOD/flux-correction
research direction, not permission to import a DG subcell implementation or
to claim entropy stability from a Powell-system paper. The existing WCEM
characteristic design is deferred unless this characteristic-free route is
falsified by a specific mechanism.

## Reconstruction Candidate: Wave-Constrained Entropy-Modal Reconstruction

The provisional reconstruction is abbreviated WCEM. The name is only a
working label.

For each face, first transform the six-cell primitive stencil into a local
orthonormal frame and a scaled physical-wave basis. In systems with a
compatible convex entropy pair this should be an entropy-scaled
characteristic basis. The relativistic magnetic case needs separate treatment
described below.

For each wave family `k`, form a conservative finite-volume polynomial in a
cell-average-preserving modal basis,

```text
P_k(xi) = a0_k + a1_k phi1(xi) + ... + a4_k phi4(xi).
```

The unrestricted coefficients reproduce a fifth-order MP/WENO-quality face
value. The accepted coefficients are the closest coefficients to that target
under a small face-local constrained problem:

```text
minimize  sum_k,m wm_k (a_mk - a_mk_high)^2

subject to
  sign([v_k]_face) = sign(Delta v_k_cell),
  |[v_k]_face| <= C_k |Delta v_k_cell|,
  U(q_left), U(q_right) are admissible,
  B_normal,left = B_normal,right = B_normal,CT,
  1 >= theta_1k >= theta_2k >= ... >= theta_4k >= 0.
```

Here `v_k` denotes a scaled entropy/wave variable and each modal coefficient is
`a_mk = theta_mk a_mk_high`. The ordered weights remove the highest polynomial
mode first. The same expression therefore reaches a robust parabola, PWL, and
PWC as constraint boundaries, without selecting separate reconstruction
methods. Families that become degenerate are constrained as one subspace, not
through singular individual eigenvectors.

The sign condition is valuable only when paired with an entropy-conservative
flux and dissipation written in the reconstructed entropy-variable jump. It is
not a proof for ordinary Godunov fluxes applied to independently reconstructed
primitive components.

### Why the basis must be coupled

Two simpler candidates were implemented and rejected:

1. A shared fourth-difference sensor blended MP5 toward PPM4. It improved some
   512-cell RMHD profiles but on held-out Balsara 4 at 1024 cells produced L1
   `3.470219e-3`, overshoot `1.340847e-2`, and TV `1.5555`, all worse than
   PPM4 except for no terminal reset.
2. A minimal joint-face sign projection applied to MP5 traces was excellent
   for scalar advection: it was indistinguishable from MP5 on the smooth sine
   test and reduced the 128-cell square-wave overshoot from
   `1.9367e-4` to `2.7659e-5` for a 0.67 percent L1 increase. Applied
   componentwise to RMHD primitives and spatial four-velocity, however, it
   raised Balsara-4 512-cell overshoot/TV to `8.3054e-2/2.1007` versus
   PPM4's `2.5258e-2/1.4149`.

The second failure is decisive evidence that a scalar sign property does not
preserve a coupled RMHD wave. The production candidate must project a shared
physical-wave subspace and then map one coherent left/right state back to
primitives.

### Mode-selective physical-wave prototype

`tools/research_wave_reconstruction.py` now implements the missing
mode-selective step for one-dimensional ideal-gas Euler. Each genuinely
nonlinear acoustic family has one nested trace formula,

```text
q_face = q_PWL
       + theta_lower  (q_parabolic - q_PWL)
       + theta_higher (q_MP5 - q_parabolic),

0 <= theta_higher <= theta_lower <= 1.
```

The two coefficients are the Euclidean projection of the full target `(1, 1)`
onto a face-local feasible polygon. Its constraints are the existing MP5
one-sided admissible intervals and a joint trace jump between zero and the
neighboring cell-average wave jump. PWL-MC reaches PWC where its slopes vanish.
The linearly degenerate entropy/contact family retains MP5. A whole-state
positivity scale is shared by both traces, followed by a final positivity/jump
closure against floating-point drift.

A lexicographic objective was tested and rejected because it over-selected the
common parabola and lost strong-shock accuracy. The Euclidean projection has
no fitted shock coefficient. It is inactive on the periodic entropy wave:
characteristic MP5 and the hierarchy have identical density L1 errors
`4.18930e-6, 1.62476e-7, 8.93779e-9` at 32, 64, and 128 cells after the
face-CFL audit.

After enforcing reconstructed-face signal speeds at every SSPRK stage, the
audited 256-cell comparison is:

| Case | Method | summed normalized L1 | max overshoot | max TV ratio |
| --- | --- | ---: | ---: | ---: |
| Sod | characteristic MP5 | `3.61173e-3` | `2.74858e-3` | `1.007409` |
| Sod | coefficient hierarchy | `3.62721e-3` | `2.46989e-3` | `1.007372` |
| Lax | characteristic MP5 | `6.25535e-3` | `1.65163e-2` | `1.026281` |
| Lax | coefficient hierarchy | `6.23747e-3` | `1.63894e-2` | `1.024594` |
| Blast | characteristic MP5 | `9.50178e-3` | `2.82300e-2` | `1.029063` |
| Blast | coefficient hierarchy | `9.73949e-3` | `2.79788e-2` | `1.028767` |

The hierarchy lowers overshoot and TV on all three tubes and improves Lax L1
by 0.29 percent. It costs 0.43 percent L1 on Sod and 2.50 percent on blast.
These Cycle-247 values compare exact conserved cell averages with numerical
cell averages; the earlier point-profile scores were not finite-volume
consistent. This remains a Pareto control, not strict L1 dominance.

### Ideal-MHD extension

`tools/research_mhd_wave_reconstruction.py` carries the identical hierarchy
into the generated seven-wave ideal-MHD basis. Entropy and Alfvén families
retain MP5; fast and slow families receive the nested projection. The reduced
reconstruction uses one face-normal magnetic field for both traces.

Two thousand random and near-degenerate characteristic round trips have
maximum relative error `4.49e-13`. On the exact circular Alfvén wave,
characteristic MP5 and the hierarchy are identical: transverse normalized L1
is `2.75847e-5` and `2.53134e-6` at 32 and 64 cells, normal-field spread is
zero, and conserved-integral errors are at roundoff.

Brio--Wu is a numerical-reference test, not an exact MHD solution. The
reconstructed-face CFL audit overturns the earlier all-metric improvement
claim. At 128 cells against a 512-cell characteristic-MP5/HLLD reference:

| Method | aggregate L1 | transverse L1 | max overshoot | max TV ratio |
| --- | ---: | ---: | ---: | ---: |
| characteristic MP5 | `5.92358e-3` | `2.35654e-3` | `1.74145e-2` | `1.173892` |
| coefficient hierarchy | `6.01583e-3` | `2.56811e-3` | `1.48098e-2` | `1.172509` |
| balanced hierarchy | `5.99658e-3` | `2.38803e-3` | `1.48495e-2` | `1.172321` |

Both hierarchies reduce overshoot and TV, but MP5 retains lower L1. The
balanced objective substantially recovers the coefficient hierarchy's
transverse error, yet still costs 1.23 percent aggregate and 1.34 percent
transverse L1 against MP5. More MHD tubes, rotated waves, CT, and AMR are
required before promotion.

### Wave-trace metric experiment

The coefficient projection was also generalized to minimize the error in the
two returned scalar wave traces. A normalized combined objective,

```text
||delta theta||^2
  + 2 ||A delta theta||^2 / ||A||_F^2,
```

matches the mean eigenvalue of the two-dimensional wave-trace metric to the
coefficient metric and is invariant to uniform rescaling of the wave. This is
a **wave-trace metric**, not a coupled primitive, conserved, or entropy metric:
each characteristic family is still optimized independently.

The cell-average-corrected audit gives a narrower result. At 256 cells the
combined metric improves the coefficient hierarchy's Sod L1/overshoot/TV and
Lax overshoot/TV, while its Lax L1 is effectively tied but higher by
`2.0e-8`. On blast it improves overshoot
`2.79788e-2 -> 2.78641e-2` and TV `1.028767 -> 1.028692`, but L1 rises from
`9.73949e-3` to `1.02657e-2`, a 5.40 percent cost rather than the old
point-profile estimate of 0.53 percent. At 128-cell Brio--Wu it improves
aggregate/transverse L1 and TV relative to the coefficient objective, while
overshoot rises by 0.27 percent; that fine-to-coarse MHD comparison was already
conservative and is unchanged. Trace-only and differently normalized
objectives also change the ranking. Retain this as a useful Pareto control, not
a universal replacement or an entropy-metric claim.

The final two-sided trace contract now survives the terminal positivity step.
One common scaling is applied to both traces, and a floating-point closure
projects farther toward the two cell averages only if the mapped physical
traces no longer satisfy positivity or the nonlinear-wave jump interval.
Twenty thousand log-uniform HD stencils and twenty thousand MHD stencils pass
for each coefficient and combined metric with zero nonfinite/nonpositive
traces, zero jump violations, and maximum normalized violation `2.853e-14`.
`tools/research_reconstruction_invariants.py` is the repeatable falsification
harness.

### RMHD entropy limitation

The conservative ideal RMHD equations used with CT do not directly admit the
convex thermodynamic entropy pair required by the standard entropy-stable
construction. Published RMHD entropy symmetrization adds a
Godunov--Powell-type source. That is a formulation change, not a drop-in
reconstruction option.

The first WCEM implementation should therefore:

- use entropy-scaled variables for HD, ideal MHD with its compatible
  divergence treatment, and RHD;
- use local characteristic subspaces plus admissibility, normal-field, and
  Rankine--Hugoniot constraints for the existing conservative CT RMHD path;
- make no RMHD entropy-stability claim unless the symmetrizing source and CT
  discretization are derived and validated together.

### Local MP5-THINC and physical-wave subcell prototype

`tools/research_subcell_reconstruction.py` implements an order-independent
local precursor of cascade BVD. It offers MP5 or P4 and conservative THINC
profiles with the published Euler steepness pair `beta=(1.1, 1.6)`.
Wavewise selection minimizes boundary variation separately in each
characteristic field; shared selection chooses one profile for the complete
characteristic state. The THINC quadrature audit preserves its supplied cell
average to `6.03e-15`.

P4-THINC is rejected in this operator because it produces substantially larger
Euler overshoot and TV. MP5-THINC is a useful but not universal control. At 32
cells its wavewise form lowers exact-profile aggregate L1 relative to
characteristic MP5 on Sod, Lax, blast, and reverse blast, and lowers a moving
contact L1 from `3.1006e-2` to `1.5933e-2`. It is exactly identical to MP5 on
the 16- and 32-cell smooth entropy wave. The colliding-shock tube and the MHD
tests expose the cost of indiscriminate sharpening: L1 can improve while
overshoot and TV worsen.

The Cycle-247 conserved-cell-average rerun preserves that conclusion. At
N=32, wavewise MP5-THINC changes summed L1 from
`2.70394e-2/3.66389e-2/6.36557e-2/6.27026e-2` to
`2.30437e-2/3.02472e-2/5.84587e-2/5.98763e-2` for
Sod/Lax/blast/reverse-blast, but worsens colliding-shock L1 from
`1.18189e-2` to `1.34963e-2`. The contact rerun initializes and compares
off-grid discontinuities as exact cell averages; the older center-sampled
contact scores were `4.7462e-2/3.0593e-2`.

The physical-wave hybrid therefore assigns the candidates by wave class:

```text
nonlinear acoustic / fast / slow:
    balanced PWL-MC -> parabola -> MP5 hierarchy

linearly degenerate contact / entropy / Alfven:
    MP5 <-> THINC subcell candidate selected by boundary variation
```

The MHD form retains one normal magnetic field at every face. A continuous
wave-purity factor,

```text
purity_j = activity(linear wave j)
           / (activity(linear wave j) + activity(nonlinear waves)),
```

also tests whether a linearly-degenerate component is isolated or embedded in
a compound shock. The square-root form and a purity-adaptive THINC steepness
are Pareto experiments, not fitted production formulas.

At 32 cells versus an eight-times-finer characteristic-MP5 reference, the
purity-adaptive hybrid improves all four reported error metrics over the
balanced hierarchy:

| MHD tube | method | aggregate L1 | transverse L1 | max overshoot | max TV ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Brio--Wu | balanced MP5 | `1.72919e-2` | `7.50805e-3` | `4.25628e-2` | `1.163526` |
| Brio--Wu | adaptive hybrid | `1.71603e-2` | `7.46165e-3` | `4.24460e-2` | `1.162719` |
| Dai--Woodward | balanced MP5 | `1.40437e-2` | `1.60693e-2` | `3.07651e-3` | `0.994860` |
| Dai--Woodward | adaptive hybrid | `1.35155e-2` | `1.51481e-2` | `2.79280e-3` | `0.994851` |

The 64-cell Brio--Wu check prevents a broad winning claim. Characteristic MP5
still has the lowest aggregate L1 (`9.40473e-3`). The adaptive hybrid improves
balanced-MP5 L1, transverse L1, and overshoot but raises its TV ratio from
`1.140908` to `1.141213`; the gentler fixed-beta hybrid improves all four
balanced-MP5 metrics but remains behind characteristic MP5 in L1 and TV. All
tested smooth circular-Alfven results are identical to characteristic MP5,
1,000 broad random MHD stencils have no inadmissible traces, and the normal
magnetic jump is zero. These are one-dimensional Newtonian research results
with numerical MHD references, not CT, AMR, SRMHD, cost, or production
validation.

The first rescaled-state audit found a concrete homogeneity defect in the
baseline MP5 provider: its unlimited-candidate test uses
`epsilon * max(1, scale^2)`. Rescaling ideal-MHD primitive states as
`rho,p -> zeta(rho,p)` and `B -> sqrt(zeta) B` can therefore change the
selected trace. A standalone MP5 variant using `epsilon * scale^2` reduces
the maximum 200-stencil BVD reconstruction mismatch from `1.67e1` to
`2.81e-14`; its 32-cell Brio--Wu and Dai--Woodward metrics remain essentially
the same. The production `mp5.hpp` has the same absolute unit-scale test, but
this research cycle does not change it. The full physical-wave hybrid also
inherits absolute tolerances from the older balanced hierarchy and is not yet
a homogeneous SR candidate.

The current Python form deliberately recomputes neighboring candidates at
each face. A 32-face microbenchmark measures the adaptive hybrid at `24.95x`
characteristic MP5 and `2.83x` the balanced hierarchy; scale-invariant
wavewise BVD is `17.41x` MP5. These ratios are useful evidence that the
research implementation shape is not production-ready, not a C++ throughput
prediction. A C++ experiment must compute each cell candidate once, reuse it
at both faces, and then measure the complete timestep.

The SR extension is architectural rather than automatic. It should replace
the Newtonian eigenbasis and positive-density/pressure scale with a
relativistic characteristic provider and a convex physical-constraint
projection that also enforces subluminal velocity. SRMHD must additionally
retain the one-normal-field/CT contract. Do not apply scalar THINC directly to
relativistic primitive components and call it the same method.

### Time-integrated physical-front result

Cycle 248 tests a stronger interpretation of the in-cell hypothesis. A
standard spatial trace is insufficient even when it recovers the correct
plateau state: SSPRK stages still average fluxes at different numerical
profiles, and a moving discontinuity is diffused when it crosses a face. The
new research operator therefore retains characteristic/component MP5 in
ordinary cells but, for one accepted physical wave, integrates the plateau
flux on each affected face over the exact substep crossing time.

Acceptance is deliberately strict:

1. two constant plateau states bound the candidate cell;
2. one common fraction of their conserved states reproduces every component
   of the cell average;
3. both endpoints are physically admissible, including subluminal velocity in
   the relativistic systems and one common normal magnetic field in MHD; and
4. one speed satisfies the complete Rankine--Hugoniot flux jump.

This changes the isolated-wave result qualitatively. The trace-only ablation
is close to ordinary MP5, whereas the time-integrated operator transports the
tested fronts in one mixed cell:

| system and CFL-0.9 control | baseline total L1 | physical-front total L1 |
| --- | ---: | ---: |
| NRHD contact, N=32 | `1.00072e-2` | `1.65597e-13` |
| NRHD normal shock, N=32 | `1.15160e-2` | `1.80850e-15` |
| NRMHD contact, N=32 | `9.97088e-3` | `4.00519e-13` |
| NRMHD oblique fast shock, N=32 | `4.38486e-2` | `1.25649e-15` |
| NRMHD rotational discontinuity, N=32 | `4.98718e-2` | `1.16012e-12` |
| SRHD contact / shock, N=16 | `1.34642e-2 / 1.02568e-1` | `2.52576e-15 / 1.69348e-15` |
| SRMHD contact / oblique shock, N=8 | `2.80856e-2 / 2.44601e-1` | `1.33747e-15 / 5.27361e-14` |

The smooth NRHD entropy, NRMHD circular-Alfven, SRHD entropy, and SRMHD entropy
controls remain on the baseline path to roundoff. The NRMHD and SRMHD tests
retain exactly one normal magnetic field. A hostile detector audit constructs
1,000 unrelated but admissible conserved mixtures in each of NRHD, NRMHD,
SRHD, and SRMHD: all 4,000 mixtures invert successfully, none passes the full jump
test, and all 64 exact-contact controls pass.

This is not yet the requested all-regime reconstruction. Brio--Wu and
Dai--Woodward start as compound Riemann fans, so the strict memoryless detector
does not activate and returns exactly MP5+HLLD. Retaining the complete initial
HLLD fan is not a solution: at N=32 and CFL 0.9 its aggregate L1 is
`1.33361e-1/8.38784e-2`, versus `1.47037e-2/1.10285e-2` for MP5+HLLD. The
individual HLLD intermediate jumps have physical-flux Rankine--Hugoniot
residuals from roughly `2.5e-2` to `8.5e-1`; they are useful approximate flux
states, not an exact fan that can be persisted as physical subcell structure.

An offline nonlinear projection can move broadened transition endpoints onto
the exact MHD jump manifold while preserving the complete window total.
Condensing those projected waves is still mixed: Brio--Wu L1 and TV worsen;
Dai--Woodward L1 improves by about one percent, but maximum overshoot rises
from `1.65489e-2` to `2.16524e-2` and TV crosses above one. This falsifies
final-profile sharpening as the missing universal rule. A physical front must
be a causal degree of freedom carried through time, not inferred and sharpened
after numerical diffusion has already altered its location and neighbors.

The resulting research direction is a two-representation finite-volume
operator:

```text
smooth / rarefaction / interacting background:
    high-order conservative space-time polynomial or GRP predictor

independently verified shock / contact / rotational front:
    persistent conservative subcell jump + time-integrated face flux
```

The unresolved work is front creation from a compound fan, wave interactions
and deletion, multidimensional front geometry, CT coupling, AMR transfer, and
an efficient cached implementation. SRHD/SRMHD show that the local algebra is
not the blocker. GRMHD should only apply the validated SRMHD front operation in
a face-local orthonormal tetrad while integrating metric/source variation
consistently; local flat Rankine--Hugoniot consistency alone does not make a
curved-spacetime wave globally exact.

### Causal complete-fan and interaction result

Cycle 249 tests the next lifecycle object in
`tools/research_causal_fan_bundle.py` and
`tools/research_causal_fan_interaction.py`. The first harness solves an exact
Euler Riemann problem from the supplied endpoint states, persists its shock
and contact trajectories, and lets characteristic MP5/HLLC own only
rarefaction interiors. It is still oracle-assisted because the fan birth
location is supplied, but it carries the solution causally rather than
sharpening a final profile.

Across N=16, 32, and 64 at CFL 0.4 and 0.9, this representation improves
aggregate L1, maximum overshoot, and maximum TV ratio over MP5/HLLC for Sod,
Lax, blast, reverse blast, and a held-out two-shock problem. At N=32 and CFL
0.9:

| case | MP5/HLLC L1 | causal fan L1 | causal overshoot | causal max TV |
| --- | ---: | ---: | ---: | ---: |
| Sod | `3.17441e-2` | `4.30192e-3` | `3.59e-16` | `1.000000` |
| Lax | `4.03541e-2` | `9.90492e-3` | `7.26e-16` | `1.018971` |
| blast | `6.58278e-2` | `6.76072e-3` | `4.91e-16` | `1.013531` |
| reverse blast | `5.29490e-2` | `2.77318e-3` | `5.88e-16` | `1.000000` |
| two shocks | `1.57149e-2` | `1.68e-16` | `6.41e-17` | `1.000000` |

Moving the initial interface to `x=0.487` so it cuts a cell preserves the
advantage and candidate conserved integrals to roundoff. During this audit,
the shared exact Euler reference was found to hardcode a left-rarefaction
density and right-shock density. Canonical Sod hid the defect. Star density
now selects the shock or isentropic formula independently on each side, and
the exact-reference gate covers left/right shock-rarefaction, two-shock, and
two-rarefaction domain balances.

The interaction harness launches two exact fans toward one another, computes
their first shock-collision event, deletes the incident fronts, and solves a
new child Riemann fan from the two post-shock states. It stops before that
child fan reaches the old contacts, so the tested event graph is exact and
bounded. At N=32 and CFL 0.9:

| ownership policy | total L1 | max overshoot | max TV |
| --- | ---: | ---: | ---: |
| MP5/HLLC | `5.02687e-2` | `2.65897e-2` | `1.040089` |
| exact fronts, MP5 rarefactions | `9.19792e-3` | `2.66522e-2` | `1.041781` |
| primitive-linear complete fan | `2.75452e-2` | `5.00606e-2` | `1.100121` |
| physical complete fan | `1.18111e-9` | `5.67743e-9` | `1.000000014` |

At N=64 and CFL 0.9 the physical complete-fan L1 is `7.78e-14`. All candidate
forms are globally conservative because they replace one face flux, but only
the actual rarefaction wave curve removes the ownership artifact. A
primitive-linear surrogate is decisively worse. This supports a coherent
local space-time Riemann/GRP element, not an arbitrary interpolation between
wave endpoints.

`tools/research_causal_mhd_fan_interaction.py` carries the same event through
an exact ideal-NRMHD invariant subsystem. With `gamma=2`, `Bx=0`, zero
transverse velocity, and `By=k rho`, the MHD conserved state and flux map
exactly to Euler with effective pressure `p + By^2/2`. The harness audits that
mapping for every incident/star/child state, then evolves the mapped problem
with characteristic MP5/HLLD or the complete physical-fan flux. Over N=16--64,
CFL 0.4/0.9, and magnetic-mass ratio `k=0.1`, the causal result has total
primitive L1 `5.4e-16`--`5.4e-11`, conserved errors near roundoff, zero normal
field spread, and maximum TV within `1e-9` of one. MP5/HLLD gives
`2.36e-2`--`8.36e-2` L1. N=32/CFL-0.9 checks at `k=0.03` and `0.13` retain
the same result; the latter makes 84.5 percent of the low-pressure state's
effective pressure magnetic.

These are the strongest mechanism tests so far, but not a production method
or universal claim. The complete Euler fan uses analytic Riemann information
and supplied birth events. The MHD result covers perpendicular fast structure
only. General birth from cell-average data, nonzero-normal-field
slow/Alfven/compound fans, repeated event topology, shock-smooth interaction,
multidimensional geometry, CT/AMR transfer, relativistic wave curves, and
cost remain open.

### NRMHD path selection and latent Alfvén state

General ideal NRMHD changes the question in two ways. First, the exact Riemann
problem need not be unique: regular and nonregular intermediate/compound-wave
solutions can satisfy the usual weak and entropy conditions for the same
data. Second, even a uniquely chosen rotational discontinuity does not survive
ordinary homogeneous finite-volume projection. Its physical path keeps
`|Bt|` fixed while rotating in the `(By,Bz)` plane; the componentwise cell
average follows the chord instead of that arc.

The half-turn control makes the state error explicit. Both endpoints have
`|Bt|=0.5` and `p=1`, but their equal conserved mixture inverts to
`|Bt|=0` and `p=7/6`. The conserved total energy is intact; the homogeneous
primitive inverse has reclassified unresolved tangential magnetic and kinetic
energy as heat. This is the large-angle Alfvén averaging mechanism identified
independently by Xu et al.'s numerical-path-preserving Roe analysis.

The old `physical_wave_space_time` prototype did not fully honor its own
design: it inferred the same wave again from homogeneous primitives every
step. It carries the half-turn to about `1e-12` at N=16 and at N=32/CFL 0.4,
then loses it at N=32/CFL 0.9; at N=64 it also loses the 90-degree rotation.
That resolution reversal is a detector/projection failure, not physical
convergence.

`persistent_physical_wave_space_time` instead verifies the complete jump once,
retains the two endpoint states, speed, and position, and integrates the
physical flux over every face-crossing interval. It never asks the chord
average to rediscover the arc. Contacts, the independently constructed
oblique fast shock, and 90/180-degree rotations cross many cells at N=16, 32,
and 64 and CFL 0.4/0.9 with:

- total L1 `1.2e-16`--`1.2e-14`;
- exactly one transition cell and maximum TV equal to one to roundoff;
- conserved-integral errors at roundoff; and
- exactly constant normal magnetic field.

For the N=32/CFL-0.9 half-turn, characteristic MP5/HLLD has total L1
`7.8404e-2`, maximum overshoot `1.3010e-1`, and maximum TV ratio `1.7710`;
the persistent element has `2.2190e-15`, `2.6645e-15`, and `1.0`.
The persistent 24-case matrix takes about 0.26 seconds in the Python harness
because propagation is analytic after birth. The comparison is mechanism and
amortization evidence, not a production C++ throughput benchmark.

A second standalone probe evolves Brio--Wu and the coplanar/near-coplanar
Torrilhon data with a conservative fourth-order centered ideal flux plus
explicit viscosity, resistivity, and heat conduction. Balanced,
viscosity-dominant, and resistivity-dominant paths are normalized to the same
maximum diffusion length. Through 2048 cells they remain admissible, preserve
normal field, and close their discrete boundary-flux balance to about
`1.6e-13`; path and resolution differences shrink together. This does not
establish a universal dissipative limit. The near-coplanar control retains the
known slow pseudo-convergence, so a small brute-force local PDE solve can cache
the wrong apparent fan.

The resulting method boundary is sharper:

```text
smooth region:
    high-order cell averages + MP5/space-time predictor

accepted nonsmooth region:
    physical wave/fan path as latent state
    + cheap time-integrated propagation between events

birth/collision:
    state-derived wave-curve solve with an explicit admissibility rule
```

This is more than switching reconstruction or Riemann formulas, but it remains
inside conservative finite volume: the cell average is still the conserved
ledger, while the latent fan supplies information that averaging destroys.
The isolated result does not yet solve general fan birth, rarefactions,
nonunique-solution policy, interactions, CT edge ownership, AMR transfer, or
relativistic/curved-spacetime extension.

### State-derived birth and complete fan ownership

The isolated NRMHD carrier now has a deliberately narrow state-derived birth
path. Two real plateau cells on each side plus either one exact conserved
mixture or a face-aligned jump supply the endpoint states and position; no
test-case endpoint, speed, or location is passed to the detector. The fit is
performed in a local Galilean frame with nondimensional conserved and flux
components. Complete Rankine--Hugoniot closure is followed by a separate
physical-family classification: contact, nonzero-normal-field Alfvén rotation
with the Walén relation and constant transverse-field radius, or an extremal
fast Lax shock with entropy increase.

The detector is intentionally allowed to say "unresolved." Fixed
total-energy-ULP allowances do not separate physical equality from a small but
distinct pressure jump after an ill-conditioned conserved-to-primitive round
trip. A two-million-contact scan found exact resolved-by-the-old-policy cases
requiring five ULPs, while a separate perturbation scan found that the
four-ULP policy accepted 83.6 percent of represented pressure jumps in its
screened set. The detector therefore admits no ULP exception: both pressure
recovery and bulk-boost uncertainty must be no larger than the nominal
`3e-10` fit tolerance. That screen only decides whether the invariant is
resolvable; its uncertainty is not added back onto the `3e-9` physical
equality tolerance. Storage roundoff is componentwise: a large transverse
field or velocity cannot relax the normal-field or normal-velocity invariant.
A clean fast shock at bulk velocity 200 remains accepted; at velocity 300 the
clean and corrupted shocks both return no birth.
The clean/corrupt high-Mach contact and Alfvén non-discrimination pairs also
both reject. A `p=1e-12` contact with no kinetic or magnetic cancellation
accepts, while the same pressure embedded in finite velocity/field and the
`Bn=1e6` Alfvén control are explicitly unresolved. Recovering those regimes
requires a more informative local or dual-energy ledger, not a looser
predicate. Under this strict policy, 11,386 of 100,000 broadly drawn exact
contacts lie inside both caps and every one fits/classifies; the worst plus
the first 128 also pass full birth. Deliberate in-cap pressure scans accept
zero of 2,000 contacts and zero of 1,947 Alfvén states outside the nominal
equality tolerance. The post-fix hidden-component check is deterministic, not
yet randomized. Other controls reject zero-normal-field pseudo-Alfvén jumps,
multiple candidates, missing boundary plateaus, nonadmissible mixtures,
broken scaling, and unsupported periodic one-element topology.

Once born, the carrier retains the endpoints, speed, position, and Alfvén
path metadata rather than re-inferring them from the homogeneous mixed cell.
Contacts, fast shocks in both directions, and 90/180-degree rotations pass 30
N=16/32/64, CFL-0.4/0.9 cases with maximum total L1 `1.573e-14`, maximum
overshoot `7.922e-15`, exactly one transition cell, unit TV, constant normal
field, and conserved errors at roundoff. Thirty-two Alfvén controls cover both
normal-field signs, both Walén signs, and rotations through a half-turn. Six
smooth circular-Alfvén controls differ from the untouched MP5/HLLD path by at
most `1.932e-14`. A 10,000-pair audit in each of NRHD, NRMHD, SRHD, and SRMHD
finds zero unrelated-state accepts while accepting all 16 exact contacts per
system. The relativistic rows audit their existing narrow detectors; they do
not extend this NRMHD carrier to relativity.

State-derived endpoints alone are not sufficient for a compound fan. On the
off-grid N=64 Sod control, exact shocks/contact with an MP5 rarefaction lowers
L1 from `1.7551e-2` to `4.3889e-3` at CFL 0.4 and removes overshoot, but raises
the TV ratio from `1.011879` to `1.020994`. This ownership seam persists at
CFL 0.9. When the complete physical fan, including the rarefaction curve,
owns the face-time flux, all 60 N=16/32/64, CFL-0.4/0.9, aligned/off-grid Euler
controls have maximum L1 `1.524e-15`, maximum overshoot `2.334e-15`, unit TV,
and conserved errors at roundoff. That mode calls the exact Euler Riemann
solution recovered from the detected endpoints. It validates the latent
space-time representation and event bookkeeping, not the predictive accuracy
or cost of a practical rarefaction provider.

The two-fan NRMHD collision also derives both initial fan origins and endpoint
states from finite-volume data before solving its incident and child fans.
Across N=16--64 and CFL 0.4/0.9, the candidate has maximum L1 `5.427e-11`,
maximum overshoot `1.638e-10`, TV within `9.5e-10` of one, roundoff
conservation, and zero normal-field spread; `By/rho=0.03` and `0.13` controls
retain the result. The fan solver is still exact-Euler-assisted inside the
restricted `Bx=0`, `By=k rho`, `gamma=2` invariant subsystem.

The resulting conclusion is narrower and more useful than "choose a better
limiter": a nonsmooth finite-volume average is an incomplete state for a
curved physical wave path, and splitting one physical fan between unrelated
space-time operators creates a measurable seam. The next provider must solve
and cache a complete regular-MHD wave path, including rarefactions, while
making an explicit regular/nonregular selection. No production C++, CT, AMR,
SRMHD, or GRMHD path has been changed by this experiment.

## Riemann Candidate: Constraint-Optimized Wave Enrichment

The provisional Riemann policy is abbreviated COWE. The robust version is
specified first; an analytic fast version is a later reduction of the same
constraints.

At a face, transform both states to the local orthonormal frame. Construct an
entropy-certified LLF/HLL flux `F0` and a set of corrections `Ck`, one for each
physical wave subspace:

```text
F(theta) = F0 + sum_k theta_k Ck,       0 <= theta_k <= 1.
```

The target corrections may be supplied by HLLC for HD/RHD, HLLD for MHD/RMHD,
or eventually by an entropy-scaled path-integral flux. The outer algorithm is
the same for every physics system. It maximizes resolved-wave content under:

- ordered, finite intermediate fan states;
- positive density and thermal/internal energy in every fan state;
- subluminal velocity and a valid primitive inverse in RHD/RMHD;
- the two-point entropy inequality where the equation formulation supports
  it;
- a minimum shock entropy-production budget so contact/Alfven antidiffusion
  cannot cancel all robust dissipation inside a compressive nonlinear wave;
- common normal magnetic field and magnetic-energy consistency;
- Rankine--Hugoniot residual bounds for each recovered wave;
- continuous grouped limits when `B_normal`, slow, Alfven, or contact speeds
  coalesce; and
- all-Mach scaling that removes excessive acoustic dissipation without
  reducing nonlinear-shock dissipation.

The robust research form can solve this low-dimensional constrained problem
per face. The admissible fan set is not assumed convex. The efficient form
should derive analytic upper bounds for the grouped `theta_k`; it must
reproduce the robust result on a large random state corpus before replacing
it.

For GRMHD, the solver acts only on the special-relativistic local face problem.
The tetrad transforms fluxes and states back to the coordinate basis. Curvature
sources remain outside the Riemann problem.

### Fan-hierarchy prototype result

The following Cycle-242 result is retained as falsification history. Its
independently weighted MHD inner fan was subsequently shown not to preserve
the contact Rankine--Hugoniot relation and has been replaced by the common-fan
construction below.

`tools/research_constrained_riemann.py` preserves the first scalar-correction
falsification control and originally tested a nested fan hierarchy:

```text
entropy-certified LLF
  + theta_outer   (HLL - LLF)
  + theta_contact (contact fan - HLL)
  + theta_Alfven  (HLLD - contact fan),

1 >= theta_outer >= theta_contact >= theta_Alfven >= 0.
```

Each coefficient is independently maximized while retaining the two-point
entropy budget and admissibility of both worst-local-CFL one-sided updates.
The implementation now inspects both HLLC star states and all HLLD star/double
star states. An invalid inner fan is removed without discarding an admissible
outer fan.

Those two constraints alone were inactive for all 20,000 broad random HD
pairs and all 20,000 broad random MHD pairs. This is not a successful new
solver; it proves that a scalar HLL-to-HLLC/HLLD blend with only those checks
is normally a no-op. Adding a dimensionless compressive pressure/total-pressure
entropy budget makes the prototype nontrivial (mean accepted target weights
0.904 for HD and 0.959 for MHD, while stationary contacts retain weight one),
but this overlaps established HLLC antidiffusion-control ideas and is not a
novelty claim.

On 20,000 broad random pairs with no added compressive entropy budget, every
safe update is admissible and every target flux is finite. All HD HLLC fans are
admissible. In MHD, 878 HLLD target fans (4.39 percent) contain an inadmissible
intermediate conserved state even though the returned flux is finite. The fan
hierarchy detects those cases; its mean accepted outer/contact/Alfvén weights
are `0.999991/0.988508/0.988508`. It recovers stationary HD contacts, the
stationary MHD contact, and an exact stationary rotational discontinuity to
`2.78e-17` or better.

This is a meaningful robustness result but not a winning default flux. With
the mode-selective reconstruction at 64 cells, the no-budget fan hierarchy is
indistinguishable from HLLC on Sod and Lax and slightly worse on the Euler
blast. On Brio--Wu it lowers maximum overshoot from `2.1723e-2` to
`2.0991e-2`, but raises aggregate L1 from `9.4958e-3` to `9.6533e-3` and TV.
On a 16-cell smooth circular Alfvén wave it raises transverse L1 from
`4.5696e-4` to `5.8685e-4`. The compressive-budget variant does not remove
that tradeoff. Keep the hierarchy as a robust fan-audit/fallback prototype;
do not promote it until the entropy control is smooth-consistent or a better
entropy-compatible resolved target is derived.

### Audited common-fan target and rank-one entropy repair

The independently weighted MHD contact and Alfvén corrections were
mathematically rejected. The HLLD outer-star pair is not generally adjacent
across the contact, so it does not satisfy the contact Rankine--Hugoniot
relation by itself. Unequal contact and Alfvén weights therefore create a
nonphysical synthetic fan even when every interpolated state is positive.

The corrected prototype uses one common full-fan weight:

```text
F_target = F_HLL + theta_fan (F_HLLD - F_HLL).
```

The same `theta_fan` interpolates every single- and double-star state from the
HLL intermediate state. Convexity of the ideal-HD/MHD admissible set then
supports a bisection bound, while the common affine weight preserves every
accepted Rankine--Hugoniot relation. Unordered fans, nonfinite fluxes, an
inadmissible HLL state, or a normalized fan RH residual above `1e-10` reduce
the target to HLL/entropy-certified LLF rather than accepting a malformed
inner fan.

After that audit, a positive Tadmor residual is corrected along the state jump,

```text
F_entropy = F_target - beta [U],
beta = max(0, ([v]^T F_target - [psi]) / ([v]^T [U])).
```

This is the minimum correction in the secant entropy metric, not a new general
entropy-stabilization principle. A final maximal blend toward the
entropy-certified LLF flux is available only if the corrected one-sided
updates are inadmissible. Every reconstructed face now contributes its
physical signal speed to the standalone SSPRK3 harness; a flux-specific
contract may only raise that bound. Nonfinite/negative bounds are rejected,
later-stage increases tighten the timestep with one-percent headroom, and a
finite but nonphysical final stage is retried rather than committed. The local
repair audit is defined at CFL `0.45`, so the research CLI rejects
`entropy_repair` evolution above that value instead of implying a stronger
guarantee.

In 20,000 broad random pairs, all accepted HD/MHD target fans are admissible,
ordered, and have maximum normalized RH residual `6.803e-11`; all final local
updates satisfy the entropy and admissibility checks. The raw MHD HLLD target
has 879 inadmissible and 2330 unordered fans, and the common accepted fan
weight averages `0.880429`. The rank-one entropy repair and terminal LLF blend
are inactive on this distribution, so those mechanisms remain an evidence
gap rather than a validated benefit. Exact stationary HD/MHD contacts and a
stationary rotational discontinuity are retained to roundoff.

The literature scan found every ingredient separately: Tadmor entropy
viscosity, optimization-based entropy corrections, invariant-domain/LLF
limiting, positive relaxation fans, BVD/trace objectives, and recent
energy-consistent HLLD variants. It did not locate the exact paired-trace
hierarchy or the exact `audit full fan -> common fan contraction -> rank-one
repair -> LLF update bound` composition. That supports continued investigation
of a possibly useful synthesis, not a novelty claim.

## Literature Collision Map

The design deliberately intersects several established method families:

| Mechanism | Established nearby work | Remaining candidate distinction |
| --- | --- | --- |
| Fifth-order nonlinear polynomial | MP5, WENO-Z, TENO, multi-resolution WENO | modal constraints are solved jointly in a physical-wave basis |
| Smooth-extremum preservation | Colella--Sekora PPM | no componentwise parabola clipping; extrema survive through modal/sign constraints |
| Sign property | ENO, SP-WENO, TeCNO | fifth-order MP-quality target plus vector wave-subspace projection |
| Bound preservation | invariant-domain/convex limiting, positivity-preserving FV | face modes and Riemann wave corrections share one admissibility provider |
| Boundary variation | BVD/THINC | a constraint, not discrete candidate selection |
| Robust complete-wave flux | HLLC/HLLD, HLL3R/HLL5R | maximum independently admissible wave enrichment from one robust fan |
| Shock stability | HLLC-ADC/SWM and hybrid HLL solvers | entropy-budgeted per-wave constraints generalized to MHD and relativistic providers |
| Low-Mach accuracy | all-speed HLLD modifications | all-Mach target inside the same constrained wave optimization |
| MHD positivity/energy | positive HLLD and magnetic-energy-consistent HLL variants | explicit energy consistency is a constraint shared with CT |

Until a broader source search shows otherwise and the full method is written
mathematically, the synthesis may be useful but must not be described as
novel.

## Validation and Promotion Matrix

Coefficient choices, if any, must be fixed on a training subset and then frozen
before held-out tests.

1. **Algebra and random states**
   - constants, linear/quadratic/quartic cell averages, mirror and negation
     symmetry;
   - exact contact, shear, Alfven, and stationary discontinuities;
   - at least one million log-uniform HD/MHD/RHD/RMHD state pairs;
   - every intermediate state admissible and every enabled entropy/RH
     inequality checked.
2. **Smooth accuracy**
   - scalar advection medley and critical points;
   - HD entropy/acoustic/shear modes;
   - circular Alfven and all MHD characteristic modes;
   - RHD/RMHD entropy and linear waves;
   - measured fifth order before time error dominates.
3. **Exact one-dimensional discontinuities**
   - Sod and strong HD shock tubes;
   - exact RHD shock tubes;
   - all five exact SRMHD Balsara problems at 128--2048 cells;
   - scalar conservation checks and no atmosphere reset.
4. **Multidimensional robustness**
   - Quirk odd-even shock, carbuncle, double Mach reflection, blast waves;
   - rotated x/y/z and diagonal consistency;
   - Orszag--Tang, rotor, field loop, current sheets, low-beta blast;
   - relativistic shock extrusion and magnetized blast.
5. **CT and AMR**
   - normal-field preservation, edge-EMF consistency, reflux and average-down;
   - Brio--Wu AMR+CT and smooth AMR improvement;
   - no loss of divergence control or rank/rotation consistency.
6. **Cost**
   - cold build and cache-hit compile separately;
   - face throughput, full timestep throughput, GPU register pressure;
   - robust constrained optimizer versus analytic fast form.

A production method must improve the aggregate L1/overshoot/TV/reset Pareto
front, not merely one plotted component. Exact-solution and scalar diagnostics
remain mandatory.

## Immediate Research Sequence

1. Completed in Cycle 262: derive standalone flat-SRMHD HLLC/HLLD from the
   published jump conditions, retain audited HLL limits, validate contacts,
   rotations, zero-normal-field degeneracy, random fans, exact Balsara profiles,
   CT/AMR, and curved-spacetime local-frame fluxes. HLLD is the production
   solver; HLLC remains the contact-only control. The resolved N=1024 result
   confirms a modest flux benefit but rejects a flux-only explanation of the
   narrow Balsara-3 shell.
2. Completed in Cycle 264: replay the first Balsara-4 and Balsara-3 stages from
   identical saved states. The component DMP rejects legitimate physical
   extrema before Balsara-4 ringing, while SSP bookkeeping, generic
   discontinuity detection, PPM shock sensing, entropy-floor checks, and
   density/energy split guards do not provide a safe separating trigger.
3. Completed in Cycle 266: match the offline operator to all three saved
   production SSPRK3 stages, compare reconstructed and exact physical traces,
   and replay a two-history temporal detector through one complete next
   timestep. The replay matches production to `2.00e-13`, but marked-face PWL
   increases L1 and TV, and the detector also fires on an exact Balsara-5 fan.
   The simple characteristic-free temporal repair is rejected.
4. Treat Balsara 3 as a representation-limited narrow shell and Balsara 4 as a
   sharper physical compound fan plus amplitude/phase error, not as one generic
   ringing syndrome. The focused base-N=128 AMR control reaches local N=1024
   spacing with 536 peak cells and slightly beats uniform N=1024 L1 for both
   HLL and HLLD. Use adaptive resolution as the current direct remedy; do not
   invent a scalar detector to erase valid compound structure.
5. If new reconstruction research resumes, start from locally coupled system
   information and test complete production timesteps immediately. Direct
   velocity/rapidity coordinates are neutral, adaptive transformed pressure
   reverses its one-step gain in evolution, and freely fitted HLLD signal speeds
   are unsafe. A characteristic-subspace method must be audited from NRHD
   through GRMHD; an entropy-variable route must first derive the Powell/source,
   CT, and AMR formulation together rather than borrowing an RMHD entropy claim.
6. Keep the exact Euler fan, mapped `Bx=0, By=k rho` interaction, regularized
   micro-solve, approximate-HLLD persistence, and THINC/BVD work as labeled
   controls. Do not extend them toward production unless a production causal
   trace specifically calls for the mechanism they isolate.
7. Advance to multidimensional CT/AMR and local-tetrad GRMHD only after one
   flat-space mechanism survives the NRHD--NRMHD--SRHD--SRMHD reduction with
   conservation, smooth-wave inactivity, phase-shift, resolution, and CFL
   0.4/0.9 controls.
8. Completed in Cycle 270: the coupled finite-volume candidate preserves
   smooth characteristic-MP5 accuracy and substantially improves difficult
   Balsara-4 robustness, including at CFL 0.9. It does not survive the full
   reduction: HD shocks, Brio--Wu, and Balsara 3 are materially more diffuse
   than characteristic MP5. Preserve the exact sensor plateau,
   representation-conditioning, and shared-state/shared-face lessons, but do
   not port or optimize the complete method until a new mechanism removes
   those cross-regime regressions.
9. Completed in Cycle 271: local physical-specific-entropy minima and
   prior-state numerical-characteristic hulls do not separate the Balsara-4
   defect from legitimate Balsara-3 or exact Balsara-fan evolution. A
   conservative continuous shared-face MP5-to-PWC blend was then compared
   directly with the current binary correction through complete N=64,
   CFL-0.4/0.9 Balsara-3/4 evolutions. It is nearly neutral on Balsara 3 and
   substantially worse on Balsara 4 at CFL 0.4; at CFL 0.9 it remains a mixed
   tradeoff. Reject the added continuous limiter and retain the simpler binary
   production guard.
10. Completed in Cycle 272: a conservative Hancock/subcell time-slab profile
    confirms that coherent time evolution can improve Balsara 3/4 at CFL 0.9,
    but it regresses Balsara 3 at CFL 0.4 and local RK splices create ownership
    seams. Reject the low-order splice and use it only as evidence for a
    complete high-order space-time background.
11. Cycle 273 supplies that background offline from HD through SRMHD:
    primitive CWENO-Z, a weak local space-time DG predictor, time-integrated
    HLLC/HLLD, and an a-posteriori MP5/PWL/PWC hierarchy. Flat-SRMHD smooth
    entropy and all seven linearized families recover fifth order at CFL 0.9;
    exact Balsara worst-component L1 improves for all five N=64 cases and TV
    improves in all five. The remaining Balsara-2 defect is localized to a
    strong material contact. A magnetization-gated material-characteristic /
    fourth-difference / PPM3 density correction reduces its undershoot by
    `7.15%` at N=64/CFL 0.9 and is exactly inactive at `B=0`, but its CFL-0.4
    and current-production-operator controls remain mixed. A newly explicit
    NRMHD MP5 control likewise leaves a real Pareto boundary: coherent is
    `6.44%` worse in Brio--Wu total L1 while substantially less oscillatory,
    but `12.0%` better in Dai--Woodward total L1. Keep the contact mechanism
    as a separately selected offline ablation rather than part of the default
    coherent core or an isolated production limiter. The complete operator is
    still not production-ready:
    multidimensional CT/AMR and curved-GRMHD gates are absent, and the
    numerical eigensystem must become a generated, device-suitable regular
    basis. Continue from this candidate rather than reopening rejected scalar
    detectors or time-operator splices.
12. Cycle 274 separates that candidate into matched spatial and temporal
    controls and removes the novelty ambiguity. Its double reconstruction,
    primitive local space-time DG/Picard solve, and conservative one-step
    update are the established ADER-Prim design; its a-posteriori correction
    is MOOD-family logic. The candidate is therefore a PyFluid implementation
    and evaluation of that family, not a new reconstruction method. The core
    is still substantive: at N=64/CFL 0.9, replacing SSPRK3 with the local
    predictor for the same characteristic-CWENO spatial polynomial lowers
    exact Balsara 1--5 worst-component L1 by
    `18.9%/46.6%/39.1%/1.4%/31.7%`. Disabling the DMP worsens the guarded
    local-predictor result by `7--25%` in those five cases, although secondary
    overshoot/TV metrics remain Pareto. The MP5-parabola/local-predictor branch
    is sharp but only third-order on smooth waves. Unlimited-quartic/MOOD and
    point-MP5 trace-anchor alternatives were falsified rather than retained;
    central-weight and WENO-Z-exponent scans also show no cross-case dominance.
    Keep the three ablation harnesses as the reproducible boundary, keep the
    magnetized contact correction opt-in, and do not modify production until
    multidimensional CT/AMR and curved-GRMHD gates plus a generated basis and
    an actual C++ performance benchmark exist.
    The CFL-0.4 control narrows this further: fixed-CWENO local-time improves
    Balsara 1/2/3/5 L1 by `8.5--24.7%` but worsens Balsara 4 L1 by `13.8%`
    while lowering its TV. In Balsara 4, disabling the DMP recovers part of
    that L1 but raises TV. Treat the candidate as a strong CFL-0.9 Pareto point,
    not an all-CFL result.
