# PyFluid AMReX Infrastructure Conventions

This document owns durable infrastructure and artifact contracts. Numerical
methods and physics belong in their topic documents; coding style belongs in
`code_conventions.md`; dated measurements and decisions belong in
`review_running_cycle.md`.

## Continuity Files

- `references.md` is a router to authoritative local sources and external
  references. It does not own design decisions.
- `knowledge.md` records compact learned facts, numerical boundaries, and
  AMReX or physics subtleties that materially affect future work.
- `review_running_cycle.md` records dated work state, commands, measurements,
  failures, and validation evidence.
- This file records stable infrastructure ownership and artifact semantics.

## Production And Source Ownership

- `pyfluid_amrex/` is the production C++/AMReX path. Python/JAX and
  Python/Numba are reference paths unless a task explicitly selects them.
- The input file is the simulation configuration. Runners execute and report;
  they do not override simulation parameters.
- `include/pyfluid_amrex/config.hpp` owns the runtime configuration contract;
  parsing, supported choices, validation, and derived policy live under
  `src/framework/`.
- `schema/fields.toml` is the only declaration source for named grid fields,
  components, labels, centerings, roles, and persistence/output ownership.
  Python strictly loads and validates it and emits generated C++/JSON/document
  views. Runtime code must not create a second raw-string field registry.
- `schema/methods.toml` owns supported-method identity such as rotation-law
  names. Generated equations remain in labeled symbolic Python sources rather
  than TOML expression strings.
- Runtime reduction identity/integrands come from the diagnostic catalog.
  Runtime AMReX reductions own discrete totals; postprocessing owns analytic
  comparisons and derived presentation.
- Fluid conserved state is cell-centered, CT magnetic state is face-centered,
  and evolved Z4c state is vertex-centered. Named persistent fields belong in
  the field schema; RHS, RK-stage, elliptic-residual, transfer, and other local
  work arrays are scratch.
- Large host orchestration belongs in compiled `.cpp` owners. Header-only code
  is for small value types and pointwise CPU/GPU kernels where inlining is part
  of the implementation. Runtime method choice occurs outside cell loops.

## AMR And Output Infrastructure

- One coordinate-independent physical implementation serves every executable
  dimension. Dimension branches are limited to storage, centering, topology,
  or an explicitly documented unsupported feature.
- One selected coarse-fine state interpolater serves level creation, regrid,
  FillPatch, subcycling, and repair seeds. Fluid state is refluxed and averaged
  down; CT uses edge-EMF reflux and face average-down; nodal spacetime follows
  its nodal synchronization contract and is not refluxed.
- State and auxiliary GF snapshots use generated field descriptors. Auxiliary
  output normally inherits the state GF cadence; explicit compute/write
  overrides remain available. Scalar diagnostics and checkpoints retain their
  independent cadences.

## Independent Numerical References

External codes are comparison baselines, not production initial-data or
physics implementations. A paired comparison records an equivalence statement
for equations, units, EoS, state conventions, domain, boundaries, initial data,
and final time; reports unavoidable discretization differences; begins with
uniform-grid controls; and compares common physical points or integral
diagnostics rather than storage ordering. Preserve each code's input, build
identity, stdout, resolution, conservation/constraint results, and wall time
under `runs/current/comparisons/<code>/<case>/`.

An external numerical reference complements but does not replace PyFluid's own
analytic, conservation, restart, AMR, and performance gates.

## Reusable Initial Data

A reusable initial-data artifact is not an evolution checkpoint. It stores a
provider-independent physical core after the initial-data solve:

- cell-centered fluid primitives;
- nodal ADM spatial metric, extrinsic curvature, lapse, and shift;
- face-centered magnetic data when present; and
- source hierarchy, EoS and physical parameters, diagnostics, schema version,
  and provenance.

The same artifact may also contain exact same-hierarchy evolution caches.
Those caches are accelerators, not the physical interchange contract: they
permit lossless direct evolution loading and can seed another XCTS rotation
target without reconstructing conserved or Z4c state from the physical core.

TOV, rotating-star, binary, and external providers must all produce this same
field contract. The artifact must not encode timestepper, reconstruction,
subcycling, output cadence, or other mutable evolution choices.

`initial_data.output_file` writes this artifact at step zero.
`initial_data.input_file` loads it. On an identical hierarchy, matching
physics, EoS, component layout, and centering are mandatory and the exact
caches start evolution directly. For a saved rotating-star artifact with the
same rotation law and a larger configured target angular velocity, the same
input automatically continues XCTS from the saved physical state; a matching
target starts evolution without another XCTS solve. The ordinary in-process
sequence accepts an explicit strictly ordered
`initial_data.rotation_continuation_targets` list and retains live fields
between targets without artifact I/O.

Cross-hierarchy interpolation and asymptotic exterior materialization remain
the next part of this contract, not a current capability. The present reader
rejects a different dimension, domain, hierarchy, centering, physics system,
or physically relevant EoS definition rather than silently interpolating.

A finer target grid does not recreate information missing from the saved
solution. It can reduce evolution truncation error, but a genuinely
higher-resolution initial-data solution requires a higher-resolution solve.
The field identities and centerings in the physical core are declared once in
`schema/fields.toml`. Checkpoint/restart remains a separate time-evolution
contract with step, time, and run-control state.
