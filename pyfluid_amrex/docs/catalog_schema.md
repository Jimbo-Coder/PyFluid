# Declarative Catalog Schema

Status: the field stage is implemented. `schema/fields.toml` is authoritative
for GF declarations. The method stage has begun narrowly with rotation-law
identity in `schema/methods.toml`; diagnostics and the remaining method
families remain later staged migrations.

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
output-visible, or checkpoint eligible. Solver-local RHS, RK-stage, transfer,
elliptic residual, and other work arrays remain scratch and are not GFs.

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
groups, selection names, and references to integrand/reducer implementations.
The actual symbolic integrands and runtime reduction algorithms stay in Python
or C++; the text catalog names them and validation requires every reference to
resolve exactly once.

`methods.toml` owns supported runtime method names, stable codes, build-option
names, broad capability requirements, and implementation entry identifiers.
Its first implemented family is `rotation_law`: the strict loader emits the C++
enum and descriptors consumed by configuration and the rotating-star factory.
The relativistic Euler integral and nonlinear iteration remain generated/C++
science code. Future method families should migrate independently rather than
turning this first catalog into an unvalidated omnibus registry. Factories
consume generated descriptors. Coupled numerical compatibility and physical
validation remain explicit code when they cannot be represented as a simple
declarative requirement.

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
4. Strengthen architecture checks so new production GF strings or descriptors
   outside generated views fail.
5. The first independent method slice, rotation-law identity, now lives in
   `schema/methods.toml`; extend it only with a matching implementation and
   focused configuration/equation test.
6. Repeat independently for diagnostics and other method families after their
   ownership and validation boundary is clear.

Every migration stage must preserve catalog ordering, component IDs, centerings,
checkpoint metadata, output names, and numerical behavior. A format migration
does not authorize solver or default changes.
