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
- `schema/methods.toml` owns supported-method identity and dispatch metadata,
  including reconstruction names/codes/stencil radii/characteristic forms and
  rotation-law names. Its native initial-data-owner records declare natural
  coordinate/axis layouts, dimensions, grid kinds, broad object/disk support,
  relative-tilt support, and reflection class. Coupled physical restrictions
  remain explicit host validation; owner records do not generate a solver
  Cartesian product. Numerical formulas and generated equations remain in
  labeled C++ or symbolic Python sources rather than TOML expression strings.
  The catalog describes native nonlinear solve owners, not every validation
  projection. In particular, an `N x M x 1` run is the 3D operator on a
  one-cell invariant direction and is a required dimensional-degeneracy check
  against the corresponding native `N x M` reduction when that symmetry is
  valid; it is not a separate physical owner or a new compiled method product.
- Each compiled reconstruction value owns one flux-kernel translation unit.
  Its CMake feature switch must remove that owner from the source list, not
  merely hide its factory case. Candidate correction composes selected
  reconstruction methods at runtime; it is not another template axis.
- A large pointwise algorithm that would multiply through every generic
  reconstruction owner belongs in one dedicated compiled owner. Keep its
  public header to a small host entry, keep device-callable mathematics in one
  private algorithm header, and prevent unintended inlining when measured
  object growth shows cloned bodies. Instantiate only runtime-valid method
  combinations; configuration must reject misleading unsupported variants.
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
- A tabulated EoS is loaded and validated once into immutable managed storage;
  pointwise CPU/GPU consumers receive the same small pointer/value view. Adding
  a table backend must not instantiate unsupported physics, layout, topology,
  or characteristic products. Preserve specialized pointwise kernels where a
  measured generic dispatch regresses throughput, and prune the added backend
  at compile time to its validated capability boundary. The table manifest
  states provenance and continuation policy, while runtime compatibility uses
  a fingerprint of the normalized physics data rather than a file path.
- The standard suite runs the generated-equation numerical contracts after
  the CMake code-generation dependency has established freshness. Exact
  byte-for-byte regeneration of every expensive SymPy artifact belongs in the
  full suite; do not make ordinary solver iteration repeat that exhaustive
  check.

## AMR And Output Infrastructure

- Repository MPI launches go through `validation_common.py`. Each launcher
  owns a new POSIX process group containing its ranks; a timeout, fatal-rank
  marker, nonzero launcher exit, or runner exception/interruption terminates
  and reaps that complete group while retaining combined rank output. Do not
  add raw `subprocess.run` MPI call sites that can orphan ranks after the
  launcher exits.

- One coordinate-independent physical implementation serves every executable
  dimension. Dimension branches are limited to storage, centering, topology,
  or an explicitly documented unsupported feature.
- One selected coarse-fine state interpolater serves level creation, regrid,
  FillPatch, subcycling, and repair seeds. Fluid state is refluxed and averaged
  down; CT uses edge-EMF reflux and face average-down; nodal spacetime follows
  its nodal synchronization contract and is not refluxed.
- A physics-system admissibility restriction augments that selected
  interpolater in its state-transfer owner; it is not a second interpolater or
  a post-transfer atmosphere repair. Keep relativistic conserved components
  and a coarse parent's child family coupled so a conservative base retains its
  parent average. AMReX may request only part of one child family for a FAB or
  ghost region, so the owner must interpolate and limit the complete
  sibling-aligned family in private scratch before copying the requested
  subset. A partial-region call must not change the resulting child values.
- Face-centered magnetic fields are canonical when CT is active. Before an
  accepted relativistic state is exposed to another RK stage or AMR level,
  synchronize its cell-centered magnetic components from the corresponding
  canonical face state and apply physical acceptance with that same field.
  A final post-sync recovery/projection is a backstop, not a substitute for
  stage-consistent face-field evolution.
- After hierarchy synchronization or regrid, refresh the metric coupling and
  perform final relativistic acceptance before exposing or writing the state.
  Newly created or remade levels must keep current and old-time storage
  semantically consistent after that acceptance step.
- CPU-tiled kernels that write ghost regions use `MFIter::growntilebox` so
  adjacent internal tiles do not write overlapping grown boxes. Device-side
  temporary FABs retain AMReX `Elixir` ownership through asynchronous kernels
  and copies. The 2D/3D boundary-tiling gate covers ordinary, patch-union,
  relativistic, and CT ghost fills with forced small tiles.
- AMReX's all-node physical-boundary path constant-extrapolates ghosts and
  delegates reflection to the user functor; `BCRec` is not applied there as it
  is for cell/face data. Nodal Z4c reflection therefore mirrors about the
  boundary node with catalog parity, composes parity at corners, and explicitly
  zeros reflection-odd values on the valid boundary plane. Constraint stencil
  masks treat those reflected ghosts as same-level data while continuing to
  exclude one-sided outflow-boundary stencils.
- State and auxiliary GF snapshots use generated field descriptors. Auxiliary
  output normally inherits the state GF cadence; explicit compute/write
  overrides remain available. Scalar diagnostics and checkpoints retain their
  independent cadences.
- Shared TSV writers escape embedded NUL, tab, carriage-return, and newline
  characters rather than corrupting row structure. Canonical validation result
  paths belong to the complete declared matrix; filtered or exploratory runs
  write distinct result names and must not overwrite canonical evidence.

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

For a generated stellar magnetic seed, the edge-centered vector potential is
transient construction scratch rather than a persistent GF. The owner takes its
discrete curl with the production CT topology, normalizes against the current
metric, and stores only the schema-owned canonical face magnetic fields plus
seed identity/parameters in metadata. A coupled constraint solve invokes this
owner after each matter update and before assembling RMHD stress-energy. This
makes the gravitational constraints consistent with the seeded field; it does
not imply magnetohydrostatic Euler/Grad--Shafranov force balance. The current
pressure-poloidal owner is deliberately restricted to fresh, single-level,
3D Cartesian, solve-only stellar XCTS. AMR requires one edge-potential
prolongation/synchronization owner before it may be admitted.

The artifact records `magnetic_force_balance=not_solved` for this seeded
route. This is an explicit semantic boundary: the gravitational constraints
include the magnetic stress, but no magnetized Euler or Grad--Shafranov
force-balance equation has been solved.

The separate `poloidal_grad_shafranov` owner is the force-balanced stellar
route. It solves the pure-poloidal conformally-flat vector Grad--Shafranov
equation together with the magnetic Bernoulli first integral inside the same
stellar/XCTS fixed point. Its catalog parameters select
`M(A_phi)=k_pol(A_phi+xi A_phi^2/2)` and nonlinear relaxation; the toroidal
current is zero. Cell-centered and edge-centered potentials are private solver
scratch. The persistent CT owner is the Valencia density
`mathcal B^i=sqrt(gamma) B^i`; portable primitive fields contain physical
Eulerian `B^i`. The shared field catalog keeps the representation-neutral
label `B^i`; the route-specific artifact metadata is authoritative. Artifacts
record `magnetic_force_balance=grad_shafranov` and
`magnetic_face_representation=valencia_densitized`. Schema-v1/v2 RMHD artifacts
lack the latter identity and are rejected rather than guessed.

The first force-balanced owner is deliberately fresh, single-level, 3D
Cartesian, cold-barotropic TOV/RNS, coupled-XCTS, and solve-only. It supports
an equatorial reflection plane normal to its magnetic/rotation axis. A
transverse Cartesian reflection is not a symmetry of an oriented axial
magnetic moment and is rejected. AMR, curvilinear Grad--Shafranov solves,
toroidal/mixed fields, magnetized disks, and changed-grid CT materialization
need their own validated discrete owners; catalog presence does not imply
those combinations.

The container is deliberately a superset. In addition to the portable
physical core, it may contain optional consumer-specific sections such as
exact-layout evolution caches and sequence-continuation seeds. Evolution may
discard continuation-only fields, while a sequence continuation may discard
evolution-only caches. These fields are accelerators, not the physical
interchange contract, and their roles must be explicit in the artifact
metadata. Uniform Cartesian artifacts may carry exact-layout caches for
lossless direct evolution loading. Reduced mapped artifacts deliberately omit
those evolution caches: an exact same-layout continuation restores their
portable primitives and nodal ADM fields without interpolation, while a
different target uses the common materializer.

TOV, rotating-star, binary, and external providers must all produce this same
field contract. The artifact must not encode timestepper, reconstruction,
subcycling, output cadence, or other mutable evolution choices.

Black-hole providers use the same contract. The exterior ADM fields, the
horizon-interior policy (`exterior`, complete-grid `filled`, or true
`excision`), mass, origin, and Cartesian spin vector are artifact identity.
An interior fill is data on an active cell and must not be reinterpreted as an
excision mask. Conversely, true excision requires explicit active-domain and
inner-boundary ownership in both the ID solver and evolution hierarchy.
Black-hole spin, stellar rotation axis, and disk axis remain separate metadata;
this preserves aligned reduced solves and lets the full-3D owner retain a
physical relative star/hole--disk tilt through the same materializer.

The native spherical Schwarzschild-XCTS owner has both reduced `r theta` and
full-storage periodic `r theta phi` layouts. In 3D, the Hamiltonian and lapse
use the genuine spherical scalar operator while the momentum equation remains
the axisymmetric toroidal sector. This is an ordinary 3D storage/operator
compatibility owner for aligned data, including the `N_phi=1` degeneracy; it is
not the general three-component nonaxisymmetric spherical vector XCTS system.
Changed-grid vacuum shell solves reconstruct their analytic conformal iterate
from mass and apparent-horizon boundary data rather than treating interpolated
Z4c fields as authoritative conformal unknowns.

An apparent-horizon surface is a diagnostic object sampled from the hierarchy,
not a second spacetime field layout and not an implicit excision mask. The
surface solver owns its angular/volume work grid and may persist its radial
shape for tracking; it consumes Cartesian metric and extrinsic-curvature data
through the common hierarchy interpolation boundary. Exterior-only constraint
diagnostics must require the complete derivative stencil, not only its center,
to lie outside the selected horizon. Catalog-owned `apparent_horizon_mask` and
`excision_mask` auxiliary fields are derived views of that surface. They may
drive diagnostic output; the surface's signed-distance view may drive a
geometric AMR tagging band. They do not share solid-obstacle semantics and
must not alter active-cell, flux, FillPatch, or boundary ownership. Only an
explicitly validated evolution owner may
convert a found surface into inactive cells and inner-boundary closures.

An event horizon is a history-consuming postprocessing object, not a
slice-local auxiliary GF. Its future offline owner consumes a time-ordered,
schema-declared Cartesian spacetime history and evolves a null surface
backward. A general multiple-hole representation must permit topology change;
surface meshes and generator histories are derived artifacts and cannot be
used as evolution masks. Do not register an event-horizon runtime capability
until the history writer, temporal interpolation contract, backward operator,
and analytic validation gate exist together.

Optional equilibrium matter components use that same physical field contract;
they do not create a parallel artifact or materializer. The construction owner
may compose a central object and one or more disjoint matter supports before
writing the combined primitive and ADM fields. Component identity, rotation-law
parameters, support, and component diagnostics are retained as metadata, while
system-wide ADM/baryonic/angular-momentum diagnostics remain explicitly
system-wide. A gravity-disabled `test_fluid` component means its stress-energy
is omitted from the constraint solve, not that the stored rest density is zero.
Every self-gravitating component must participate in the coupled matter/metric
fixed point; adding it after convergence is only the test-fluid contract.
`equilibrium_torus` means the central-object and disk axes admit the stationary
circular first integral. `tilted_torus` means a deliberately nonstationary,
constraint-consistent circular seed: the fluid is approximately stationary and
axisymmetric about its own axis while the coupled spacetime is fully 3D. It
requires the native 3D Cartesian coupled-XCTS owner. The portable artifact may
then be evaluated on Cartesian or full `r theta phi` targets. A 3D curvilinear
target may not continue the nonlinear matter solve or add a torus until a
native curvilinear elliptic owner exists.

Initial-data centering transfer treats periodic and reflection ghost values as
valid stencil support. Bounded one-sided closure is reserved for true physical
nonreflection faces. Otherwise a reduced-domain matter source can solve a
different discrete elliptic problem even when its reported residual is tiny.
Keep this policy in the initial-data transfer owner rather than broadening the
generic evolution-centering template surface.

The portable physical-ID reader applies the same rule to a reflected source
artifact. It fills one parity-aware source ghost layer for cell-, face-, and
node-centered fields and explicitly authorizes the hierarchy interpolation
stencil to cross only metadata-declared reflection planes. A reflected
half-cell strip is source support, not atmosphere or asymptotic exterior.
Materialization preserves the saved physical fields; it does not manufacture
unresolved scales or project the interpolated fields back onto the constraints.
For evolution on a finer target, solve the source ID at least as finely as the
target's physically important region or perform an explicit target-grid
constraint polish once such an owner is selected and validated.

`initial_data.output_file` writes this artifact at step zero.
`initial_data.input_file` loads it. `initial_data.solve_only=1` performs the
configured initial-data construction, runs the normal relativistic preflight,
writes the artifact, and returns before allocating evolution flux-register
work. The source chart is part of the artifact rather than an evolution
requirement: the current schema-v3 records source dimension, coordinate family, ordered
active axes, periodic/reflection closure, and `mesh_orthonormal` tensor basis.
Physics and physically relevant EoS data remain mandatory. On an identical
hierarchy, optional exact caches start evolution directly. Otherwise the
reader maps physical target points into the source chart, interpolates the
portable core, transforms catalog-declared scalars, polar/axial vectors, and
symmetric tensors through an orthonormal Cartesian basis, reconstructs
conserved and Z4c state, and applies the target preflight. For a saved
rotating-star artifact with the same rotation law and a larger configured
target angular velocity, the same materialized state continues XCTS; a
matching target starts evolution without another XCTS solve. The ordinary
continuation identity includes the differential-rotation length and, for the
q-power family, its exponent. The artifact must also restore the converged
Euler first-integral injection energy; recomputing it from the original TOV
seed changes the nonlinear problem.
Rematerialization must carry forward the complete rotating-star continuation
summary (injection energy, surface diagnostics, charges, and spin) and disk
component summary, not only the fields. Otherwise a valid evolution artifact
silently ceases to be a valid sequence-continuation artifact.

For a tabulated EoS, both physical-ID and checkpoint metadata retain the
backend/version, normalized-table fingerprint, point counts, causality
truncation flag, baryon-mass normalization, and thermal-extension parameters.
The source directory is provenance only: relocating byte-equivalent table
files must preserve compatibility. A changed normalized table, continuation
contract, or thermal gamma must fail before materialization or restart. Do not
substitute a display model name, DOI, archive checksum, or path for the actual
physics fingerprint.

The ordinary
in-process sequence accepts an explicit strictly ordered
`initial_data.rotation_continuation_targets` list and retains live fields
between targets without artifact I/O. A multi-target rotating sequence writes
every accepted target by default; `initial_data.save_sequence=0` disables
this. A one-model TOV or RNS construction is not automatically saved. A
one-target continuation from `initial_data.input_file` is saved because it is
part of a persisted sequence. `initial_data.sequence_output_dir` overrides
the automatic location and also explicitly requests output for a one-model
rotating solve.

An initial-data solver may use the coordinates and dimension natural to its
symmetry, independently of the evolution hierarchy. Artifact metadata must
describe the physical coordinates of the stored samples exactly: a transformed
solver coordinate must not masquerade as `theta`, `r`, or another canonical
axis. A spherical axisymmetric artifact therefore stores a genuine ordered
`r theta` lattice even when the solver uses a weighted angular operator.
Matching source and target points may take the exact-layout path; different
spacing, extent, dimension, or coordinate family uses the same materializer
and the field's declared origin, exterior, and reflection closure.

Keep three representations independent:

1. the solver-private ID representation, which may be uniform, mapped,
   multidomain, modal, block-adaptive, or symmetry-reduced;
2. the portable physical-ID representation, which stores enough chart,
   hierarchy/mapping, basis, centering, and exterior metadata to evaluate the
   physical fields; and
3. the evolution hierarchy, which owns its own Cartesian resolution, extent,
   AMR, storage, and final primitive-to-conserved conversion.

The handoff is an evaluation contract at physical target points, not an AMReX
box-layout contract. A materializer must not depend on the ID solver matrix,
iteration storage, logical spacing, or refinement bookkeeping. Conversely, an
ID solver must not allocate evolution flux/register state. The portable schema
supports uniform AMReX levels and versioned explicit coordinate faces.
`coordinate_layout=explicit_faces_v1` stores every nonuniform coordinate face
used to evaluate cell-centred and nodal fields; logical AMReX boxes remain
storage indices and never masquerade as uniform physical spacing. A future
modal or block-adaptive representation must likewise version and store its
evaluation coordinates or coefficients before production.

For relativistic scratch construction with stored spacetime, the final fluid
conserved state is formed only after the authoritative nodal ADM/Z4c data have
been centered onto the evolution cells. Initial fluid primitives may be sampled
earlier, but they must be resampled/preserved as primitives and converted
against that final cell metric. Converting against a placeholder analytic
metric and later replacing the stored spacetime violates the physical-ID
handoff even if weak-curvature tests happen to recover it.

`initial_data.radial_grid=adaptive` selects the shared stellar radial-map
owner. TOV uses spherical `r`; circular RNS uses spherical `r theta`. The map
keeps regular uniform spacing through a conservative enclosing matter region
and a geometrically stretched, adequately resolved gravity exterior. It is
fixed during one nonlinear solve and rotation sequence so live fields never
need an in-solve regrid. Uniform remains the explicit work-precision control.
This is star-adapted mapped spacing, not residual-driven AMR: binaries and
multiple compact objects still need a block/multidomain design.
`initial_data.adaptive_source_radius_factor` and
`initial_data.adaptive_exterior_fraction` are bounded solver-private policy
controls; the saved explicit faces, rather than either knob, are authoritative
for continuation and materialization. The conventional RNS inputs keep the
uniform source ball through 1.8 times the spherical seed radius and reserve one
third of radial cells for the gravity exterior. TOV retains the 1.5 and
one-third defaults. Changing either map for a reduced solve-only RNS load
forces a new discrete XCTS solve even when angular velocity is unchanged;
same-map loading remains a direct sequence continuation.

Angular resolution remains one uniform `theta` count on the rectangular
separable grid; it is not AMReX angular AMR. Matched near-shedding solves make
64 cells the conventional sequence/exploration choice, 128 a practical
endpoint grid, and 256 the angular-reference control. A radius-dependent mode
cap or coarse exterior angular domain must own a genuine modal/multidomain
interface and preserve the coupled residual; do not represent it as a
nonuniform theta coordinate or silently change `amr.n_cell` during a solve.

Matter support and gravity extent need not be the same computational domain.
An ID method may stop its matter representation at the enthalpy-zero surface
while continuing metric fields through exterior and compactified domains. The
portable/evolution materializer owns the requested exterior atmosphere and
asymptotically flat completion. This separation does not license a reset of an
invalid interior solution or omission of surface matching conditions.

Cell-, node-, face-, and edge-centered arrays may share one hierarchy and
coordinate-map descriptor, but they are different degrees of freedom and are
not aliases of a single base array. Their index extents, ownership at shared
faces/edges, interpolation, synchronization, and conservation identities
remain distinct. Nodal derivatives need valid neighbor data across AMReX box
boundaries, and portable nodal fields must be synchronized to one value on
overlapping valid nodes before output. A coordinate/basis change is algebraic;
these centering, parity, axis, origin, and stellar-surface contracts are the
nontrivial part of reduced spherical-to-Cartesian materialization.
Remove duplicated topology/metadata and temporary caches where safe;
do not collapse physically distinct centerings into one storage view.

Interpolation ownership includes target points outside the source sample
hull. A cell-centred radial source has a half-cell gap at the regular origin;
clamping there is interior regular interpolation and must not create
atmosphere. The same rule holds for the interior half-cell strip adjoining a
declared reflection plane, such as the RNS equator; a finer target must not
turn that regular strip into physical exterior. Beyond the saved outer radius,
the loader clamps only the source
sample and applies the selected physical exterior at the original target
point: exact isotropic-Schwarzschild exterior for spherical TOV ADM data,
component-appropriate asymptotically flat falloff for general ADM data, and
the configured target atmosphere for TOV/RNS fluid. It must not copy a
source-boundary atmosphere value or use exact floating-point equality to
classify stellar matter support.

Automatic sequence output is organized under the run directory by EoS,
rotation law, and fixed central density. Named EoS presets and differential
law parameters are recorded in the path and `sequence.meta`; `sequence.tsv`
schema v5 indexes accepted targets together with the converged injection
energy, surface diagnostics, ADM/baryonic masses, angular momentum, and spin.
Model names include prescribed `omega`, base-grid resolution, and hierarchy
level count. Every model is the same losslessly
compressed, directly evolvable physical-ID artifact—not a sequence-only file.
Artifact metadata, including `eos_preset`, remains the compatibility authority;
directory and model names are discovery aids.

The artifact also records `spacetime_initial_data`. A loaded fixed-Minkowski
artifact may use the analytic flat-spacetime kernels only when both that source
metadata and the target configuration select Minkowski. Curved, constraint-
solved, or legacy artifacts without this metadata retain the stored-metric
path.

Fresh analytic data and stored artifacts enter evolution through the same
configured-ID materialization call. The stored-artifact implementation owns
exact-cache selection, hierarchy interpolation, exterior completion, and state
reconstruction; callers must not implement separate "sequence load" and
"evolution load" remappers.

For a same-chart periodic source, each periodic target extent must contain an
integer number of saved periods. For a nonperiodic larger domain, physical
fluid data use the configured atmosphere and ADM data use the physical
exterior described above. This is an initialization remap, not a new
constraint solve: interpolated data can have larger pointwise constraints than
the saved grid. A continued rotating XCTS solve does solve the remapped target
again; a direct evolution start does not.

A finer target grid does not recreate information missing from the saved
solution. It can reduce evolution truncation error, but a genuinely
higher-resolution initial-data solution requires a higher-resolution solve.
The field identities and centerings in the physical core are declared once in
`schema/fields.toml`. Checkpoint/restart remains a separate time-evolution
contract with step, time, hierarchy, and run-control state. Both formats are
valid evolution inputs, but a checkpoint resumes its saved time and hierarchy,
whereas a physical-ID artifact initializes a new run at step zero through this
materialization interface.

Evolution, physical-ID remapping, and the shared CTT/CTTK/XCTS elliptic layer
use tensor identity from the field catalog. Scalar, polar-vector, axial-vector,
and symmetric-tensor identities are distinct; in particular a magnetic field
is axial, so its component normal to a Cartesian symmetry plane is even and
its tangential components are odd. Physical-ID metadata records saved
reflection faces. A target outside one of those faces is mirrored into the
saved sector, interpolated on its declared centering, transformed in physical
Cartesian components, then expressed in the target basis. Points beyond the
saved outer extent still use the normal physical exterior completion.

A reduced source dimension removes coordinate dependence, not physical tensor
components. Portable `Vector3` and `SymmetricTensor3` fields retain all three
orthonormal physical components declared in `schema/fields.toml`; an
axisymmetric `r theta` source can therefore store a nonzero azimuthal velocity
or shift even though `phi` is not an active mesh axis. The materializer owns
the basis transformation to the target chart. Do not truncate portable field
rank to `AMREX_SPACEDIM` or create a reduced-dimension duplicate field schema.

The validated reduced physical-ID handoffs are spherical one-dimensional TOV
and spherical two-dimensional circular-RNS XCTS to three-dimensional Cartesian
RHD. Uniform and explicit-face radial artifacts use the same evaluator. An
exact-layout mapped RNS artifact can seed the next rotation target without
interpolation; the resulting model remains independently usable for Cartesian
materialization. Same-layout Cartesian magnetized artifacts retain their
canonical face fields. Cross-chart or changed-grid CT remains deliberately
rejected until face-centered divergence-preserving magnetic transfer has an
owner.

Object pose is a materialization choice, not an elliptic-solver coordinate
choice. The artifact persists stellar rotation axis and center separately from
black-hole spin/origin and disk axis. One proper rigid transform maps target
points back into the source object's Cartesian frame before chart
interpolation, then rotates catalog-declared polar/axial vectors and symmetric
tensors into the target frame. A nonidentity pose disables exact evolution
caches and uses the portable core. Relative axes of composed components must
remain compatible under that one transform; changing their relative tilt is a
new physical ID problem, not a materializer option.

Elliptic boundary types are component-specific: on a reflection plane the
normal component of a polar vector is odd while its tangential components are
even. Explicit reflection/type metadata is authoritative. The compatibility
path that infers legacy Robin data from a nonzero stored coefficient is used
only when no explicit reflection boundary is present.

For a centered nonrotating TOV model, independent x/y/z reflections permit a
true octant solve. A star rotating about z is not invariant under independent
x or y polar-vector reflections: its azimuthal velocity and shift change with
a coupled rotation, not ordinary plane parity. Nonzero rotating-star initial
data therefore reject x/y reflection boundaries and support the z-equatorial
reflection only. Surface rays must reconstruct the omitted reflected
direction before evaluating polar/equatorial geometry. Do not describe a
nonzero rotating Cartesian half-domain as an octant.
