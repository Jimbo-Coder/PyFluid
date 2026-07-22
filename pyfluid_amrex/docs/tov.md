# TOV Solutions And Sequences

`TovSolution` is the single owner of nonrotating relativistic stellar
solutions. Evolution initial data and direct EoS studies consume the same
radial profile and observables; there is no second evolution-specific TOV
integration.

## Solver Contract

The solver integrates in decreasing log specific enthalpy using the regular
center variables `u = r^2` and `v = m/r`. An adaptive Dormand-Prince 5(4)
method terminates at zero log enthalpy, so the stellar surface is the physical
zero-pressure endpoint rather than a density floor. The exterior is the exact
Schwarzschild solution in isotropic coordinates.

`TovSolution` stores the radial profile and:

- gravitational, baryonic, proper, and Komar masses;
- areal and isotropic radii;
- compactness, binding energy, and surface redshift;
- the quadrupolar Love number and dimensionless tidal deformability when the
  low-density EoS permits a regular surface tidal limit.

The mass observables are independent radial integrals. In particular, the
Komar mass is not copied from the gravitational mass. Their agreement is a
solver validation for a static solution. Tidal observables are currently
available when the surface polytropic exponent is at most two; unsupported
surface behavior is reported explicitly rather than assigned a fabricated
value.

The relevant run parameters are:

```text
pyfluid.tov_central_density
pyfluid.tov_absolute_tolerance
pyfluid.tov_relative_tolerance
pyfluid.tov_max_steps
```

The maximum step count is a safety bound. Accuracy is controlled by the two
adaptive tolerances.

## Evolution Atmosphere

Relativistic initial data use one explicit ambient-atmosphere policy before
primitive-to-conserved conversion and before CTT/CTTK/XCTS matter sources are
formed:

```text
initial_data.atmosphere = flat | power_law
initial_data.atmosphere_density
initial_data.atmosphere_pressure
initial_data.atmosphere_transition_radius
initial_data.atmosphere_power
```

Negative density/pressure values inherit `pyfluid.rho_floor` and
`pyfluid.pressure_floor`. A negative transition radius means four times the
TOV isotropic surface radius; non-TOV power-law data must provide a positive
radius. The power-law state is constant inside that radius and decays outside
it. Its exponent must exceed three so its unfloored three-dimensional mass
tail is integrable. Pressure follows the corresponding low-density isentropic
law and is then projected through the selected EoS, including its cold floor.

The initial atmosphere and runtime C2P recovery are separate policies. The
graded tail terminates at the exact runtime EoS floor state, so recovery does
not immediately rewrite it. Because the current Valencia evolution still has
a nonzero density floor, the asymptotic domain eventually contains a uniform
floor; strict infinite-domain mass convergence requires future vacuum-capable
C2P. Lowering the recovery floor moves that negligible plateau outward.

Evolution precomputes one constant thermodynamic recovery state from the EoS
floors. Reconstruction, C2P recovery, and the RHD admissibility limiter consume
that state directly; they do not reevaluate the spatial initial-atmosphere
profile. Runtime atmosphere replacement is density-triggered. A pressure floor
is enforced through the EoS closure and does not reset an otherwise valid cell's
conserved density or momentum.

## Direct EoS Sequences

The sequence executable does not initialize or evolve an AMR hierarchy:

```bash
cmake --build build/pyfluid_amrex --target pyfluid_amrex_tov_sequence -j4
build/pyfluid_amrex/pyfluid_amrex_tov_sequence \
  pyfluid_amrex/inputs/tov_sequence.in

uv run python pyfluid_amrex/tools/plot_tov_sequences.py \
  runs/current/tov/sequences/custom.tsv \
  runs/current/tov/sequences/sly.tsv \
  runs/current/tov/sequences/apr4.tsv \
  runs/current/tov/sequences/h4.tsv \
  runs/current/tov/sequences/ms1.tsv \
  --output runs/current/tov/sequences/eos_comparison.png
```

Each TSV row contains the complete observable set and a turning-point stable
branch flag. The plotting tool shows mass-radius, mass-central-density,
tidal-deformability, and binding-fraction comparisons. Custom piecewise
polytropes use the same EoS configuration as evolution runs.
For a one-layer cold EoS, `P=K rho^Gamma` is also the isentropic ideal-gas
equilibrium relation. A thermal ideal-gas law alone does not select a unique
TOV sequence without an entropy or temperature profile.

## Constraint Initial Data

CTT and XCTS materializations are compared to the direct TOV gravitational,
baryonic, proper, and binding masses. XCTS additionally validates Komar
surface-volume agreement because it solves the lapse. CTT does not solve a
lapse, so a Komar identity is not imposed on its unit-lapse materialization.

The current hybrid CTTK option selects nonzero mean-curvature free data from
the matter source. It therefore constructs a different constraint-satisfying
star, not the time-symmetric TOV spacetime. Its validation requires elliptic
convergence, ADM surface-volume consistency, finite stellar diagnostics, and
zero angular momentum; it is intentionally not compared to the TOV masses.

Partial-AMR CTT materialization derives Eulerian matter only on valid cells,
synchronizes that derived cell hierarchy with the normal conservative AMR
transfer, and then performs the common cell-to-vertex transfer. This avoids
evaluating C2P on interpolated coarse-fine ghost states. The focused control
requires the composite and same-level-interior Hamiltonian norms to converge
across two resolutions. A narrow coarse-fine contribution remains visible:
the multilevel elliptic operator is second order while the continuum
constraint diagnostic uses fourth-order pointwise derivatives. Uniform-grid
or interior convergence must not be used to hide that interface truncation
layer.

`InitialDataEllipticOptions::boundary_interpolation_order=4` requests AMReX's
fourth-order interpolation at coarse/fine boundaries. It does not
change the cell-centered or nodal interior Laplacian, which remains second
order. A genuinely fourth-order general initial-data solver requires a
fourth-order scalar and conformal-vector operator, two-layer ghost data,
matching Dirichlet/Robin closures, and AMR restriction/prolongation/interface
consistency. An interior-only fourth-order stencil would remain globally
second order when a second-order boundary or coarse/fine layer dominates.

## Rotating Stars

`pyfluid.ic = rotating_star` continues the direct TOV seed through a
coupled matter/XCTS iteration. The shared circular-flow update uses

```text
v^i = (Omega (axis x (x - center))^i + beta^i) / alpha
ln h = ln C - ln alpha - 1/2 ln(1 - gamma_ij v^i v^j)
```

for uniform angular velocity. The implemented differential families use

```text
j = u^t u_phi
j_constant: j(Omega) = A^2 (Omega_c - Omega)
q_power:    j(Omega) = A^2 Omega [(Omega_c/Omega)^q - 1]
ln h = ln C - ln alpha - 1/2 ln(1 - gamma_ij v^i v^j)
       - integral_(Omega_c)^Omega j(Omega') dOmega'
```

so `q_power` with `q = 1` is exactly the j-constant law. Rotation-law identity
and required parameters are declared once in `schema/methods.toml`; the Euler
integrals are generated kernels and the AMReX hierarchy update is owned by
`src/initial_data/rotating_star_initial_data.cpp`. The iteration updates the
injection energy to retain the requested central enthalpy, under-relaxes the
matter variables, rebuilds stress-energy, and resolves XCTS lapse, shift, and
conformal geometry. It rejects non-timelike helical flow and matter touching a
physical boundary. The final summary reports coordinate equatorial and polar
surface extents about the configured axis as deformation sanity checks. It also
reports the signed mass-shedding indicator

```text
chi_ms = (d ln(h) / dr)_equator / (d ln(h) / dr)_pole,
```

evaluated at interpolated surface roots and averaged over opposite axial and
equatorial rays. It approaches one in the spherical limit and zero at an
equatorial cusp. A negative value is supercritical, not an accepted equilibrium.
The indicator must be checked at multiple resolutions before quoting a physical
mass-shedding frequency.

Both the rotation-center effective potential and the surface `ln h` field are
filled through one exchanged ghost layer before arbitrary-point interpolation.
This is required when a linear interpolation stencil crosses an internal AMReX
FAB boundary; otherwise the result depends on box decomposition and may read
outside valid storage. `HierarchyInterpolationPlan` rejects a crossing stencil
that lacks storage for this layer, while the caller remains responsible for
calling `FillBoundary` before interpolation.

The current input controls are:

```text
initial_data.rotation_law = uniform
initial_data.angular_velocity
initial_data.differential_rotation_length
initial_data.rotation_power_q
initial_data.rotation_axis1/2/3
initial_data.rotation_center1/2/3
initial_data.matter_relaxation
initial_data.matter_relative_tolerance
initial_data.matter_max_iterations
initial_data.rotation_continuation_steps
```

`rotation_continuation_steps` defaults to one. Values above one solve a
sequence of fully converged intermediate angular velocities, retaining the
matter and XCTS fields between steps; this is useful when a large direct jump
from the spherical TOV seed leaves the nonlinear basin of attraction.

Measure spatial MPI scaling with an identical small-spin solve using:

```bash
UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.matplotlib uv run python \
  pyfluid_amrex/tools/benchmark_initial_data.py --force
```

The tool records wall time, seconds per matter iteration, speedup, and parallel
efficiency in `runs/current/initial_data_scaling/results.tsv`.

Run fixed-central-density SLy sequences, including charge agreement and the
mass-shedding indicator, with:

```bash
UV_CACHE_DIR=.uv-cache uv run python \
  pyfluid_amrex/tools/scan_rotating_star_sequence.py \
  --eos-preset sly --rotation-law uniform --omega 0.025 --continuation-steps 3
```

Use `--rotation-law j_constant --differential-rotation-length A` or
`--rotation-law q_power --differential-rotation-length A --rotation-power-q q`
for the implemented differential families. Sequence rows accumulate under
`runs/current/rotating_star_sequence/`; a completed executable is still marked
`FAIL_PHYSICS` when its independent charges disagree and `SUPERCRITICAL` when
the signed surface indicator is nonpositive. Each new row also records and
requires the subextremal volume diagnostic
`dimensionless_spin = J_ADM/M_ADM^2`.

The corrected N=32 fixed-central-density SLy uniform sequence currently reaches
`Omega=0.032`. At `Omega=0.028` and `0.032`, respectively,
`chi_ms = 0.404505` and `0.230266`, while
`J_ADM/M_ADM^2 = 0.504573` and `0.594246`. The upper model has coordinate
axis ratio `1.83922`; matter/volume and surface/volume angular momentum agree
to `1.75e-4` and `5.77e-3`. It is strongly rotating but remains subcritical.
These points do not bracket or resolution-converge the mass-shedding endpoint.
Earlier N=24 surface-indicator values produced without exchanged interpolation
ghost data are superseded and must not be used for an endpoint estimate.

Reusable stellar initial data follow the separate physical-field artifact
contract in `infrastructure_conventions.md`, not the evolution-checkpoint
contract. The writer/loader is not implemented yet; a step-zero checkpoint is
only a temporary exact-evolution cache.

The opt-in focused gate is intentionally outside quick/standard because the
three nonlinear 3D controls are comparatively slow:

```bash
UV_CACHE_DIR=.uv-cache uv run python pyfluid_amrex/tools/validate_z4c.py \
  --only rotating_star --ranks 3 --force
```

Zero angular velocity deliberately bypasses the matter continuation, retaining
the direct TOV path as the exact control. The first executed nonzero control is
a 3D, cold hybrid-polytrope, uniform-grid XCTS initialization at
`Omega=0.001`, followed by three PWL/HLL/SSPRK3 evolved-spacetime steps at CFL
`0.9`. This focused gate is a finite-state smoke, not a long-term stability
result. Uniform, j-constant, and q-power rotation are accepted with XCTS. The
shared matter layer can later serve CTT/CTTK, but those formulations still need
a validated stationarity closure before receiving the equilibrium label.

The short-evolution gate also protects the generic RHD stage-candidate flux
correction. Only faces adjoining a cell whose concrete RK candidate violates
the conserved admissibility floors are replaced by piecewise-constant LLF
fluxes. The gate uses the normal production CFL `0.9`; lowering CFL is not an
accepted substitute for reconstruction/admissibility correctness.

The fixed-central-enthalpy update evaluates the helical effective potential at
the configured rotation center. Each relaxed update grows or shrinks only the
matter component connected to the preceding centered TOV seed, excluding
distant positive-enthalpy lobes. This supports centered spheroidal sequences;
toroidal families require an explicit alternative topology seed. A physical
mass-shedding endpoint still needs resolution convergence of the implemented
surface-gradient indicator. The first differential-law points establish the
Euler-integral/configuration path and charge agreement, not a differential-
rotation sequence or endpoint. KEH is not required by these XCTS branches.

## Evolution Status

The focused three-dimensional two-level matrix evolves the same star in
Cowling and coupled-Z4c modes with SSPRK3 and AMR subcycling. It requires a
finite, C2P-admissible state at every RK input, a fixed metric to roundoff in
Cowling mode, maximum-density changes below one percent, and a finite-box rest
mass ledger residual below `1e-12`. Physical atmosphere outflow is recorded by
the boundary ledger rather than mislabeled as conservation error.

This is a self-consistent **unmagnetized** star. Adding a magnetic field to a
TOV profile produces RMHD initial data but not magnetohydrostatic equilibrium.
A self-consistent magnetized star requires a rotating/magnetized equilibrium
matter solve coupled to the constraint equations; it is not supplied by the
current unmagnetized uniform-rotation XCTS path.
