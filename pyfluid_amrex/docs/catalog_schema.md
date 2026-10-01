# Declarative Catalog Schema

Status: the field, diagnostic, physical-ID manifest, and selected method stages
are implemented. `schema/fields.toml` is authoritative for GF declarations,
`schema/diagnostics.toml` for diagnostic identity and ordering,
`schema/physical_id.toml` for artifact metadata, and `schema/methods.toml` for
the supported method families already migrated there.

## Purpose

Critical solver identity should be visible in small, data-only text files that
can be reviewed without executing Python or reading generated C++. Python owns
schema loading, validation, tensor/component expansion, symbolic equations, and
artifact generation; it should not remain the only place where a GF, diagnostic,
or supported method is declared.

Use standard TOML rather than a custom `.ccl` grammar. Python 3.13 provides
`tomllib`, TOML supports comments and explicit scalar/array/table types, and a
standard format avoids maintaining another parser. The intended layout is:

```text
pyfluid_amrex/schema/
  fields.toml
  diagnostics.toml
  methods.toml
```

Migrate one catalog at a time. Do not create an omnibus file or move symbolic
mathematics into strings merely to make it declarative.

## Fields

`fields.toml` is the single declaration point for every canonical GF: any named
field that is evolved, retained between operations, user-selectable,
output-visible, checkpoint eligible, or part of the reusable physical
initial-data contract. Solver-local RHS, RK-stage, transfer, elliptic residual,
and other work arrays remain scratch and are not GFs.

A representative entry is:

```toml
schema_version = 1

[[field]]
key = "magnetic_field"
name = "magnetic_field"
label = "B^i"
value_kind = "Vector3"
role = "EvolvedState"
evolution_kind = "MainState"
sector = "Fluid"
requires_mhd = true

[[field.components]]
name = "b1"
label = "B_{1}"
cpp_id = "B1"
symmetry = "Vector1"

# b2 and b3 are declared explicitly in the same way.
```

The loader converts explicit TOML components into immutable Python model
objects and validates the declared tensor shape, names, types, set membership,
and references. Components are explicit so names, stable IDs, labels, mixed
centerings, and symmetry metadata have no hidden Python defaults. Generated
artifacts include field
and component IDs, C++ descriptors, JSON views, output/checkpoint layouts,
plotting metadata, and browsable equation/field documentation.

After parsing a user-provided field name, production C++ passes only a generated
ID or descriptor. Allocation, lookup, checkpoint, output, diagnostics, and
plotting cannot accept a second raw-string field registry. A handwritten C++
fill operation may switch exhaustively on a generated ID, but cannot redefine
field identity, components, labels, centering, or ownership.

## Diagnostics And Methods

`diagnostics.toml` owns diagnostic identity, labels, kinds, component structure,
groups, selection names, surface targets, and references to
integrand/reducer implementations. Every `surface_integral` declares exactly
one `surface` target; nonsurface operations reject that key. Current runtime
targets are the configured `diagnostic_surface` and the found
`apparent_horizon`.
The actual symbolic integrands and runtime reduction algorithms stay in Python
or C++; the text catalog names them and validation requires every reference to
resolve exactly once. Contiguous vector declarations expand into ordinary
`scalars.tsv` columns in declared order. `diagnostic_catalog.py` is the strict
loader/model/validator and derives group membership; it is not a second
declaration table.

`methods.toml` owns supported runtime method names, stable codes, build-option
names, broad capability requirements, and implementation entry identifiers.
Its implemented initial-data families are `rotation_law`,
`disk_rotation_law`, and `initial_data_owner`. The strict loader emits separate
C++ enums and descriptors consumed by configuration and the corresponding
matter factories. Stellar entries are uniform, relativistic j-constant, and
relativistic q-power rotation; the disk entries are constant
`ell=-u_phi/u_t`, uniform, j-constant, and q-power. Native-owner entries declare
the natural physical coordinate/axis layout, dimension, broad central-object
and disk capabilities, grid kinds, relative-tilt support, and reflection
class. `solver_axis_layouts` independently declares valid coordinate
projections of the shared numerical operator. It does not turn a planar or
translation-invariant projection into a spherical/axisymmetric physical
equilibrium owner. Coupled
restrictions that are not a simple capability—such as stationarity, matter
overlap, or Kerr spin/disk alignment—remain explicit C++ validation. These
identities and required parameters live in TOML. Initial-data enums are emitted
into a dedicated generated header so extending them does not invalidate the
reconstruction owners. The relativistic Euler integrals, torus first integral,
elliptic operators, and nonlinear iterations remain generated/C++ science
code. Future method families
should migrate independently rather than turning this catalog into an
unvalidated omnibus registry. Factories consume generated descriptors. Coupled
numerical compatibility and physical validation remain explicit code when they
cannot be represented as a simple declarative requirement.

Configuration parameters may receive a similar catalog later, but only after a
source audit separates simple name/type/default metadata from coupled parsing
and validation. This migration must not turn complex arrays or physical
compatibility rules into an opaque schema language.

## Migration And Verification

1. The strict versioned field loader and immutable models are implemented.
2. All field declarations now live in `schema/fields.toml`; the migration
   preserved generated IDs, descriptors, JSON, and documentation byte-for-byte.
3. `field_catalog.py` now exposes parsed/validated models and derived helper
   views without declaration tuples.
4. Architecture checks require every declarative catalog and its generated
   public views, and reject duplicate production GF identity.
5. The independent stellar and disk rotation-law identities now live in
   `schema/methods.toml`; extend either only with a matching implementation and
   focused configuration/equation test.
6. Native single-object ID owner capabilities now live beside those laws and
   generate host-only physical-layout/grid/symmetry policy plus explicit
   numerical solver projections. They do not generate per-dimension solver
   variants, redefine the physical owner, or authorize every Cartesian product
   of their broad capabilities.
7. Diagnostic identity, ordering, groups, value shapes, requirements, and
   postprocess dependencies now live in `schema/diagnostics.toml`; the staged
   conversion preserves the generated C++/JSON contract.
8. Repeat independently for other method families after their ownership and
   validation boundary is clear.

Every migration stage must preserve catalog ordering, component IDs, centerings,
checkpoint metadata, output names, and numerical behavior. A format migration
does not authorize solver or default changes.
