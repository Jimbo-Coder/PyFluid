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
`3.200159e-3`, overshoot `4.524195e-3`, TV ratio `1.4913`, 2236 PWC
corrections, and zero atmosphere resets. A research candidate must beat this
and the existing MP5 accuracy controls on held-out cases before it is added to
the production method catalog.

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

### Newtonian physical-wave precursor

`tools/research_wave_reconstruction.py` tests the first system-level
precursor on one-dimensional ideal-gas Euler. It projects a primitive stencil
into the local contact and acoustic basis, continuously moves limited PWL
traces toward MP5, and enforces the wave-jump constraints either wavewise, on
the acoustic group, or with one face weight. A whole-state positivity scaling
is the only terminal trace constraint.

The constraint is inactive on the periodic entropy wave: characteristic MP5
and the wavewise form have identical density L1 errors
`4.1934e-6, 1.6301e-7, 9.0039e-9` at 32, 64, and 128 cells, or measured
orders 4.69 and 4.18 with SSPRK3 time error present. It is active on exact
shock tubes, but no grouping is yet Pareto dominant. At 128 cells:

| Case | Method | summed normalized L1 | max overshoot | max TV ratio |
| --- | --- | ---: | ---: | ---: |
| Sod | characteristic MP5 | `9.8691e-3` | `4.4275e-3` | `1.00963` |
| Sod | wavewise constraint | `1.0006e-2` | `4.6451e-3` | `1.01008` |
| Lax | characteristic MP5 | `2.0742e-2` | `1.8373e-2` | `1.02680` |
| Lax | wavewise constraint | `2.0747e-2` | `1.3427e-2` | `1.01915` |
| Blast | characteristic MP5 | `2.5796e-2` | `2.9313e-2` | `1.03004` |
| Blast | wavewise constraint | `2.5481e-2` | `2.6207e-2` | `1.02680` |

The N=256 held-out tests preserve the pattern. Wavewise limiting improves Lax
L1/overshoot/TV from `9.8050e-3/1.6347e-2/1.02632` to
`9.7278e-3/1.1626e-2/1.01824`. On the blast it lowers overshoot from
`2.8574e-2` to `2.4832e-2` and TV from `1.02943` to `1.02563`, but raises L1
from `1.4393e-2` to `1.4526e-2`. Sod regresses in all three metrics. Limiting
only genuinely compressive acoustic waves improves Sod but loses the
strong-shock gains. This is a useful mechanism result, not a winning method:
the next version needs mode-selective constraints inside each physical
subspace rather than a single high-versus-PWL coefficient.

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

### Prototype result

`tools/research_constrained_riemann.py` implements the first scalar-correction
falsification harness for ideal-gas HD and MHD. It constructs a minimally
entropy-certified LLF flux, targets HLLC/HLLD, and limits their difference by
two-point entropy and worst-local-CFL admissibility.

Those two constraints alone were inactive for all 20,000 broad random HD
pairs and all 20,000 broad random MHD pairs. This is not a successful new
solver; it proves that a scalar HLL-to-HLLC/HLLD blend with only those checks
is normally a no-op. Adding a dimensionless compressive pressure/total-pressure
entropy budget makes the prototype nontrivial (mean accepted target weights
0.904 for HD and 0.959 for MHD, while stationary contacts retain weight one),
but this overlaps established HLLC antidiffusion-control ideas and is not a
novelty claim.

The next Riemann prototype must expose separate contact, shear/Alfven, and
nonlinear corrections and test the intermediate fan itself. A scalar blend
will not be promoted.

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

1. Replace the scalar high-versus-PWL coefficient in the Newtonian HD
   precursor with an entropy-scaled, mode-selective constrained projection and
   compare against MP5, WENO-Z, PPM4, SP-WENO, and the recorded precursor.
2. Extend the same projection to the existing generated Newtonian MHD
   characteristic basis and CT normal-field contract.
3. Implement the per-wave COWE optimizer with HLLC/HLLD targets and
   multidimensional shock probes.
4. Derive and test RHD HLLC and RMHD HLLD providers in a local orthonormal
   frame. Do not make them production methods before random-fan and exact
   relativistic gates pass.
5. Only then combine WCEM and COWE; keep attribution runs with one side fixed.
