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
matter solve coupled to the constraint equations (the planned RNS/KEH/XCTS
layer); it is not part of the current TOV contract.
