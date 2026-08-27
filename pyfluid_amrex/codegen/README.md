# PyFluid Kernel Code Generation

This directory is reserved for generated physics kernels and the symbolic code
that emits them. The production AMReX solver should call ordinary inline C++
kernels; it should not depend on Python at runtime.

## Direction

Use a small repo-local SymPy generator for long algebraic kernels:

- coordinate metrics and Jacobians;
- analytic equations of state and thermodynamic derivatives;
- multipatch local/global coordinate maps and patch-map Jacobians;
- finite-volume metric factors and geometric source terms;
- vector-calculus identities in coordinate/unit-vector bases;
- shared algebraic Riemann flux formulas such as LLF/HLL component updates;
- characteristic systems and eigenvalue/eigenvector formulas;
- future GRMHD or mapped-coordinate source terms.

The model is NRPy/EinsteinEngine/Kranc-like, but smaller:

- equations are declared symbolically in Python;
- common-subexpression elimination is applied during generation;
- generated kernels use a small drop policy after CSE: hard-drop inlines only
  expression-free temporaries by default, while soft-drop inlines cheap
  single-use temporaries so repeated physics terms are still computed once;
- emitted C++ has stable function signatures and lives under
  `pyfluid_amrex/include/pyfluid_amrex/kernels_generated/`;
- generated files are reviewed like source; CMake regenerates them only when a
  generator dependency changes, and Python is never used at solver runtime.

## Expression Optimization

Every kernel runs SymPy common-subexpression elimination before C++ emission.
The optional drop policy controls which resulting temporaries are inlined:

- hard drop inlines every temporary at or below its operation threshold;
- soft drop inlines only cheap single-use temporaries;
- repeated or expensive expressions remain named temporaries so arithmetic is
  not duplicated.

The default hard/soft thresholds are `0/3`. For the current generated families,
those defaults are byte-identical to disabling drops: existing CSE temporaries
are either reused or cost more than three operations. More aggressive settings
remove most temporaries but can duplicate work and can enlarge some kernels, so
they are opt-in generation experiments rather than production defaults.

## Rules

- Generated kernels must be side-effect free pointwise functions.
- AMR, MPI, FillPatch, refluxing, CT synchronization, output, and diagnostics
  remain hand-written AMReX infrastructure.
- Coordinate algebra, vector/tensor identities, finite-volume metric factors,
  and long physical source/flux expressions should be generated when the
  generated form is clearer or less error-prone than handwritten C++.
- Short orchestration, branch selection, reconstruction, Riemann solver
  dispatch, time integration, and AMReX data movement should stay handwritten
  unless codegen removes real duplication without hiding the finite-volume
  contract.
- Generated code must not encode run-specific parameters except through
  explicit function arguments.
- Generated Cartesian flux, Riemann algebra, scalar operator, metric, vector,
  EOS, and patch-map kernels are the production source of truth once they pass the
  algebra, compile, validation, and performance gates. Keep separate
  handwritten references only in validation code, not in the runtime path.

## Maintainer Workflow

Work at the highest validated layer. Change the symbolic equation, indexed
tensor operation, EOS, or differential policy; regenerate; then inspect the
generated diff and run the algebra/compiled/runtime gates. Do not manually
re-expand a generated physical system into runtime C++ to review or repair it.
Generated headers may be large because they are build artifacts, but the
high-level source, emitter, and validation corpus must remain small enough to
reason about without expanding the full equations in maintainer context.

AMReX stage scheduling is compiled separately from the template-heavy fluid
driver. It selects generated pointwise policies and moves sector-owned data; it
must not reproduce tensor geometry, constraints, gauges, sources, or RHS
expressions. A generated Z4c change should rebuild the spacetime module, not
reinstantiate the fluid evolution driver.

## Current Boundary

The generator owns pointwise algebra that benefits from symbolic tensor and
coordinate manipulation:

- ideal HD and MHD physical fluxes;
- ideal-gas, barotropic-polytrope, and hybrid-polytrope closures, inverse
  internal-energy relations, and adiabatic sound speeds;
- LLF and HLL state-combination formulas;
- HD and MHD characteristic projection/recovery transforms;
- orthogonal-coordinate metric factors and geometric momentum sources;
- Newtonian viscous stress and tensor-divergence connection terms;
- curl, divergence, ideal/resistive EMF, and CT face-flux updates;
- local/global patch maps and their Jacobians;
- densitized Valencia RHD/RMHD primitive-to-conserved variables, directional
  conservative fluxes, local causal wave bounds, and ideal-energy C2P
  residual/reconstruction kernels for a runtime 3+1 spatial metric.

The generator represents a conservation equation as
`d_t U_A + div(F_A) = S_A`, where scalar/vector conserved quantities determine
the rank of their coordinate-independent flux. Hydro and MHD are compositions
of those equations rather than separate direction-expanded formulas. It does
not yet emit an arbitrary complete solver: characteristic basis and wave
estimates, HLLC/HLLD fan logic, reconstruction, time integration, AMR/reflux
operations, and staggered AMReX data assembly remain numerical or topology
modules.

## Coordinate Strategy

The generator uses orthogonal coordinate frames with physical components in the
local unit-vector basis. Cartesian kernels are emitted as direct specialized
functions and stay free of runtime geometry branches. Cylindrical and spherical
kernels should be emitted as separate coordinate-specialized functions that
take metric/scale-factor terms as explicit arguments, so the eventual mapped
coordinate solver can select the specialization at compile/config setup time
instead of branching inside each cell update.

## Physics Extension Contract

Build new physics through the same staged symbolic path:

1. Represent physical components with dimension-carrying `Vector` and `Tensor` values; use their
   arithmetic, dot, cross, outer-product, contraction, transpose, symmetric,
   and deviatoric operations instead of expanding components by hand.
2. Wrap values and their coordinate/time derivatives in `ScalarField` or
   `VectorField`. Use one `OrthogonalOperators` instance to lower gradient,
   divergence, curl, Laplacian, strain, and connection terms for the selected
   coordinate frame.
3. Define an analytic EOS once as `P(rho, u)`. The EOS model derives its
   barotropic property, inverse energy closure, thermodynamic derivatives, and
   adiabatic sound speed before emitting specialized kernels.
4. Express hyperbolic physics as `ConservationEquation` objects in
   `physical_system.py`. Constitutive modules such as `NewtonianViscosity`
   return physical tensors; the conservation system decides how those tensors
   enter momentum and energy equations.
5. Use the dimension-independent finite-volume objects in `discretization.py`
   for numerical lowering. Lower-dimensional kernels are projections with
   inactive derivatives, not separate physical formulas.
6. Lower only side-effect-free expressions into generated kernels. Keep
   reconstruction, Riemann fan decisions, AMR data movement, and time staging
   in the numerical layer.

This separation supports reusable coordinate-independent equations without
making symbolic field objects responsible for grid allocation, centering,
boundary conditions, or time storage.

## Relativistic MHD Extension Boundary

RMHD should extend this symbolic layer through the 3+1 Valencia equations,
not through direction-expanded handwritten kernels. The first RMHD change
must introduce indexed objects together with a generated equation that uses
them; do not add speculative `Vector4` or rank-four containers without a
production consumer.

The indexed layer distinguishes spatial from spacetime indices and records
variance. Checked `Vector3`, `Vector4`, `Tensor3`, and `Tensor4` objects provide
contraction, metric raising/lowering, tensor products, and 3+1 projections
using runtime metric fields (`alpha`, `beta^i`, `gamma_ij`, `gamma^ij`, and
`sqrt_gamma`). Invalid variance and dimension combinations fail during
generation. These are symbolic generation-time objects, not runtime AMReX
containers: generated C++ receives flat scalar structs so index abstraction
adds no cell-kernel overhead. AMReX remains responsible for metric storage,
centering, ghost validity, and update cadence.

Generate the following pointwise algebra from one high-level physical system:

- primitive-to-conserved and conserved residual/Jacobian expressions;
- four-velocity, magnetic four-vector, stress-energy tensor, and their 3+1
  projections;
- Valencia fluxes and metric-derivative source terms;
- induction/EMF expressions used by the existing staggered CT infrastructure;
- EOS-dependent closure pieces and characteristic speed estimates.

Keep nonlinear root-iteration control, admissibility/repair policy,
reconstruction, Riemann-fan branch selection, time integration, AMR, and CT
topology handwritten. The generator may emit residuals and analytic Jacobians
for primitive recovery, but it must not own the iteration or failure policy.

The fixed-spacetime vertical slice emits RHD/RMHD primitive-to-conserved,
directional flux, causal wave-speed, C2P residual/reconstruction, stress-energy,
and Valencia metric-source kernels from the indexed systems. The C++ runtime
owns bracketed iteration and the ordered recovery policy. Minkowski, Rindler,
and isotropic-Schwarzschild providers use the same generated fluid kernels.
Algebraic identities, analytic derivatives, two-sided round trips, zero-field
reduction, rotational covariance, equilibria, evolution, AMR/CT/subcycling,
and restart are validation gates. Evolved spacetime changes metric ownership
and scheduling, not the relativistic fluid equations.

## Z4c Extension Boundary

Z4c uses full field values rather than perturbations. Minkowski data are
`chi=alpha=1`, `gamma_tilde_ij=delta_ij`, with every other component zero.
The high-level equations independently implement the published Z4c system;
Cottonmouth and SpacetimeX are architecture and cross-check references, not
production source copied into this generator.

The generated pointwise API is intentionally reusable:

1. `z4c_conformal_metric` derives physical and inverse metrics.
2. `z4c_connections` computes the conformal and physical connections needed by
   every evolution RHS.
3. `z4c_diagnostic_geometry` independently computes the true Ricci tensor only
   when constraints or diagnostics request it.
4. `evaluate_z4c_constraints` computes Hamiltonian, momentum, Z, Theta, and
   algebraic constraints from the same state/derivative inputs used by initial
   data validation and postprocessing.
5. `z4c_evolution_geometry` uses the evolved contracted connection in the Z4c
   Ricci expression. It is kept distinct from diagnostic geometry because they
   agree only when the Gamma constraint is satisfied.
6. `z4c_geometric_rhs` consumes the reusable geometry objects plus explicit
   Eulerian matter projections. Vacuum passes the zero-valued matter object.
7. `evaluate_z4c_rhs` is the complete side-effect-free evolution entry point.
   Gauge choices are runtime-uniform values so adding a gauge does not
   multiply the large generated geometric RHS template product.

`evaluate_z4c_constraints_at<Order, Dim>` is the AMReX-grid entry point. It
loads the canonical nodal state, gathers centered geometric derivatives and
shift-upwind advection derivatives through the shared generated operator
contract, then calls `evaluate_z4c_constraints`. Initial-data checks, runtime
diagnostics, and postprocessing adapters should use that entry point rather
than assemble their own component or stencil maps.
`evaluate_z4c_rhs_at<Order, Dim>` applies the same
loading and derivative contract to the full evolution RHS.

Gauge RHS kernels are separate policies: harmonic, 1+log, stationary-trumpet,
slow-start 1+log, and Gamma-driver shift. Frozen gauge means a zero RHS selected
by the runtime-uniform selector; it does not need a duplicate generated
equation. Slow-start parameters and stage time are explicit arguments.

These functions remain side-effect free. Vertex storage, AMR synchronization,
algebraic-constraint projection, boundary conditions, and time integration
belong to the common runtime infrastructure. The primary
equation references are arXiv:1212.2901 for Z4c and arXiv:2404.01137 for the
slow-start lapse modification.

## Workflow

Generate explicitly before compiling generated kernels:

```bash
uv run python pyfluid_amrex/codegen/generate_kernels.py
```

or through CMake:

```bash
cmake --build build/pyfluid_amrex --target pyfluid_amrex_generate_kernels
```

The normal solver target depends on the generated headers that are wired into
production, but Python codegen is still a build-time step only. Compile and
algebra checks are separate:

```bash
uv run python pyfluid_amrex/tools/validate_codegen_flux.py
cmake --build build/pyfluid_amrex --target pyfluid_amrex_codegen_check_2d
```
