from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tomllib
from typing import Any, TypeVar

import sympy as sp


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "fields.toml"
SCHEMA_VERSION = 1


class Centering(str, Enum):
    CELL = "Cell"
    VERTEX = "Vertex"
    FACE1 = "Face1"
    FACE2 = "Face2"
    FACE3 = "Face3"
    EDGE1 = "Edge1"
    EDGE2 = "Edge2"
    EDGE3 = "Edge3"


class ValueKind(str, Enum):
    SCALAR = "Scalar"
    VECTOR3 = "Vector3"
    VECTOR4 = "Vector4"
    SYMMETRIC_TENSOR3 = "SymmetricTensor3"
    TENSOR3 = "Tensor3"
    TENSOR4 = "Tensor4"


class Symmetry(str, Enum):
    SCALAR = "Scalar"
    VECTOR1 = "Vector1"
    VECTOR2 = "Vector2"
    VECTOR3 = "Vector3"
    AXIAL_VECTOR1 = "AxialVector1"
    AXIAL_VECTOR2 = "AxialVector2"
    AXIAL_VECTOR3 = "AxialVector3"
    TENSOR11 = "Tensor11"
    TENSOR12 = "Tensor12"
    TENSOR13 = "Tensor13"
    TENSOR22 = "Tensor22"
    TENSOR23 = "Tensor23"
    TENSOR33 = "Tensor33"


class FieldRole(str, Enum):
    EVOLVED_STATE = "EvolvedState"
    INITIAL_DATA_STATE = "InitialDataState"
    EVOLUTION_AUXILIARY = "EvolutionAuxiliary"
    CACHED_DERIVATIVE = "CachedDerivative"
    OUTPUT_ONLY_AUXILIARY = "OutputOnlyAuxiliary"
    SOLVER_SCRATCH = "SolverScratch"


class DiagnosticKind(str, Enum):
    STORED_STATE = "StoredState"
    PURE_AUXILIARY = "PureAuxiliary"
    MIXED_AUXILIARY = "MixedAuxiliary"


class EvolutionKind(str, Enum):
    MAIN_STATE = "MainState"
    AUXILIARY_STATE = "AuxiliaryState"
    NON_BACKREACTING_EXTRA = "NonBackreactingExtra"
    NONE = "None"


@dataclass(frozen=True)
class FieldComponent:
    name: str
    label: str
    cpp_id: str | None = None
    centering: Centering = Centering.CELL
    symmetry: Symmetry = Symmetry.SCALAR

    def cpp_symbol(self, owner: str) -> str:
        return f"{owner}.{self.name}"


@dataclass(frozen=True)
class GridField:
    name: str
    label: str
    value_kind: ValueKind
    components: tuple[FieldComponent, ...]
    role: FieldRole
    evolution_kind: EvolutionKind
    diagnostic_kind: DiagnosticKind = DiagnosticKind.STORED_STATE
    sector: str = "None"
    ghost_width: int = 0
    checkpoint: bool = True
    output: bool = True
    aux_id: str | None = None
    requires_mhd: bool = False
    requires_solid_geometry: bool = False
    requires_gravity: bool = False
    min_dimension: int = 1


EnumType = TypeVar("EnumType", bound=Enum)

_ROOT_KEYS = {"schema_version", "field_sets", "special_fields", "field"}
_FIELD_SET_KEYS = {
    "fluid_newtonian",
    "fluid_relativistic",
    "z4c",
    "auxiliary",
    "physical_fluid",
    "adm_initial_data",
}
_SPECIAL_FIELD_KEYS = {
    "magnetic",
    "ct_magnetic",
    "passive_scalar",
    "physical_magnetic",
    "physical_passive_scalar",
}
_FIELD_KEYS = {
    "key",
    "name",
    "label",
    "value_kind",
    "components",
    "role",
    "evolution_kind",
    "diagnostic_kind",
    "sector",
    "ghost_width",
    "checkpoint",
    "output",
    "aux_id",
    "requires_mhd",
    "requires_solid_geometry",
    "requires_gravity",
    "min_dimension",
}
_COMPONENT_KEYS = {"name", "label", "cpp_id", "centering", "symmetry"}


def _reject_unknown_keys(record: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")


def _enum_value(enum_type: type[EnumType], value: object, context: str) -> EnumType:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{context} must be one of: {choices}") from error


def _required_string(record: dict[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a nonempty string")
    return value


def _optional_string(
    record: dict[str, Any], key: str, default: str, context: str
) -> str:
    value = record.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a nonempty string")
    return value


def _optional_bool(
    record: dict[str, Any], key: str, default: bool, context: str
) -> bool:
    value = record.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return value


def _load_component(record: dict[str, Any], context: str) -> FieldComponent:
    _reject_unknown_keys(record, _COMPONENT_KEYS, context)
    cpp_id = record.get("cpp_id")
    if cpp_id is not None and (not isinstance(cpp_id, str) or not cpp_id):
        raise ValueError(f"{context}.cpp_id must be a nonempty string when present")
    return FieldComponent(
        name=_required_string(record, "name", context),
        label=_required_string(record, "label", context),
        cpp_id=cpp_id,
        centering=_enum_value(
            Centering, record.get("centering", Centering.CELL.value), f"{context}.centering"
        ),
        symmetry=_enum_value(
            Symmetry, record.get("symmetry", Symmetry.SCALAR.value), f"{context}.symmetry"
        ),
    )


def _load_field(record: dict[str, Any], index: int) -> tuple[str, GridField]:
    context = f"field[{index}]"
    _reject_unknown_keys(record, _FIELD_KEYS, context)
    key = _required_string(record, "key", context)
    component_records = record.get("components")
    if not isinstance(component_records, list) or not component_records:
        raise ValueError(f"{context}.components must be a nonempty array of tables")
    if not all(isinstance(component, dict) for component in component_records):
        raise ValueError(f"{context}.components entries must be tables")
    components = tuple(
        _load_component(component, f"{context}.components[{component_index}]")
        for component_index, component in enumerate(component_records)
    )
    ghost_width = record.get("ghost_width", 0)
    min_dimension = record.get("min_dimension", 1)
    if type(ghost_width) is not int or ghost_width < 0:
        raise ValueError(f"{context}.ghost_width must be a nonnegative integer")
    if type(min_dimension) is not int or min_dimension not in (1, 2, 3):
        raise ValueError(f"{context}.min_dimension must be 1, 2, or 3")
    aux_id = record.get("aux_id")
    if aux_id is not None and (not isinstance(aux_id, str) or not aux_id):
        raise ValueError(f"{context}.aux_id must be a nonempty string when present")
    value_kind = _enum_value(
        ValueKind, record.get("value_kind"), f"{context}.value_kind"
    )
    expected_component_counts = {
        ValueKind.SCALAR: 1,
        ValueKind.VECTOR3: 3,
        ValueKind.VECTOR4: 4,
        ValueKind.SYMMETRIC_TENSOR3: 6,
        ValueKind.TENSOR3: 9,
        ValueKind.TENSOR4: 16,
    }
    expected_component_count = expected_component_counts[value_kind]
    if len(components) != expected_component_count:
        raise ValueError(
            f"{context}.components has {len(components)} entries; "
            f"{value_kind.value} requires {expected_component_count}"
        )
    component_names = [component.name for component in components]
    if len(component_names) != len(set(component_names)):
        raise ValueError(f"{context}.components contains duplicate names")
    return key, GridField(
        name=_required_string(record, "name", context),
        label=_required_string(record, "label", context),
        value_kind=value_kind,
        components=components,
        role=_enum_value(FieldRole, record.get("role"), f"{context}.role"),
        evolution_kind=_enum_value(
            EvolutionKind, record.get("evolution_kind"), f"{context}.evolution_kind"
        ),
        diagnostic_kind=_enum_value(
            DiagnosticKind,
            record.get("diagnostic_kind", DiagnosticKind.STORED_STATE.value),
            f"{context}.diagnostic_kind",
        ),
        sector=_optional_string(record, "sector", "None", context),
        ghost_width=ghost_width,
        checkpoint=_optional_bool(record, "checkpoint", True, context),
        output=_optional_bool(record, "output", True, context),
        aux_id=aux_id,
        requires_mhd=_optional_bool(record, "requires_mhd", False, context),
        requires_solid_geometry=_optional_bool(
            record, "requires_solid_geometry", False, context
        ),
        requires_gravity=_optional_bool(record, "requires_gravity", False, context),
        min_dimension=min_dimension,
    )


def _load_string_table(
    schema: dict[str, Any], key: str, expected_keys: set[str]
) -> dict[str, Any]:
    table = schema.get(key)
    if not isinstance(table, dict):
        raise ValueError(f"{key} must be a table")
    _reject_unknown_keys(table, expected_keys, key)
    missing = sorted(expected_keys - set(table))
    if missing:
        raise ValueError(f"{key} is missing keys: {', '.join(missing)}")
    return table


def _field_tuple(
    names: object, fields_by_key: dict[str, GridField], context: str
) -> tuple[GridField, ...]:
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{context} must be a nonempty array of field keys")
    if len(names) != len(set(names)):
        raise ValueError(f"{context} contains duplicate field keys")
    missing = [name for name in names if name not in fields_by_key]
    if missing:
        raise ValueError(f"{context} references unknown fields: {', '.join(missing)}")
    return tuple(fields_by_key[name] for name in names)


def _load_schema() -> tuple[
    dict[str, GridField], dict[str, tuple[GridField, ...]], dict[str, GridField]
]:
    with SCHEMA_PATH.open("rb") as stream:
        schema = tomllib.load(stream)
    _reject_unknown_keys(schema, _ROOT_KEYS, "fields schema")
    schema_version = schema.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported field schema version {schema_version!r}; "
            f"expected {SCHEMA_VERSION}"
        )

    field_records = schema.get("field")
    if not isinstance(field_records, list) or not field_records:
        raise ValueError("field must be a nonempty array of tables")
    fields_by_key: dict[str, GridField] = {}
    for index, record in enumerate(field_records):
        if not isinstance(record, dict):
            raise ValueError(f"field[{index}] must be a table")
        key, field = _load_field(record, index)
        if key in fields_by_key:
            raise ValueError(f"duplicate field key {key}")
        fields_by_key[key] = field
    field_names = [field.name for field in fields_by_key.values()]
    if len(field_names) != len(set(field_names)):
        raise ValueError("field names must be unique")

    field_set_records = _load_string_table(schema, "field_sets", _FIELD_SET_KEYS)
    field_sets = {
        name: _field_tuple(field_set_records[name], fields_by_key, f"field_sets.{name}")
        for name in _FIELD_SET_KEYS
    }
    special_records = _load_string_table(schema, "special_fields", _SPECIAL_FIELD_KEYS)
    special_fields: dict[str, GridField] = {}
    for name in _SPECIAL_FIELD_KEYS:
        field_key = special_records[name]
        if not isinstance(field_key, str) or field_key not in fields_by_key:
            raise ValueError(f"special_fields.{name} references an unknown field")
        special_fields[name] = fields_by_key[field_key]

    referenced = {
        key
        for names in field_set_records.values()
        for key in names
    } | set(special_records.values())
    unreferenced = sorted(set(fields_by_key) - referenced)
    if unreferenced:
        raise ValueError(f"unreferenced field declarations: {', '.join(unreferenced)}")
    return fields_by_key, field_sets, special_fields


_FIELDS_BY_KEY, _FIELD_SETS, _SPECIAL_FIELDS = _load_schema()

FLUID_NEWTONIAN_FIELDS = _FIELD_SETS["fluid_newtonian"]
FLUID_RELATIVISTIC_FIELDS = _FIELD_SETS["fluid_relativistic"]
Z4C_FIELDS = _FIELD_SETS["z4c"]
AUXILIARY_FIELDS = _FIELD_SETS["auxiliary"]
PHYSICAL_FLUID_FIELDS = _FIELD_SETS["physical_fluid"]
ADM_INITIAL_DATA_FIELDS = _FIELD_SETS["adm_initial_data"]
MAGNETIC_FIELD = _SPECIAL_FIELDS["magnetic"]
CT_MAGNETIC_FIELD = _SPECIAL_FIELDS["ct_magnetic"]
PASSIVE_SCALAR_FIELD = _SPECIAL_FIELDS["passive_scalar"]
PHYSICAL_MAGNETIC_FIELD = _SPECIAL_FIELDS["physical_magnetic"]
PHYSICAL_PASSIVE_SCALAR_FIELD = _SPECIAL_FIELDS["physical_passive_scalar"]


def component_names(fields: tuple[GridField, ...]) -> tuple[str, ...]:
    return tuple(component.name for field in fields for component in field.components)


NEWTONIAN_HYDRO_COMPONENT_NAMES = component_names(FLUID_NEWTONIAN_FIELDS)
RELATIVISTIC_HYDRO_COMPONENT_NAMES = component_names(FLUID_RELATIVISTIC_FIELDS)
MAGNETIC_COMPONENT_NAMES = component_names((MAGNETIC_FIELD,))
NEWTONIAN_MHD_COMPONENT_NAMES = (
    *NEWTONIAN_HYDRO_COMPONENT_NAMES,
    *MAGNETIC_COMPONENT_NAMES,
)
RELATIVISTIC_MHD_COMPONENT_NAMES = (
    *RELATIVISTIC_HYDRO_COMPONENT_NAMES,
    *MAGNETIC_COMPONENT_NAMES,
)
MOMENTUM_COMPONENT_NAMES = component_names((_FIELDS_BY_KEY["momentum"],))


def all_fields() -> tuple[GridField, ...]:
    return (
        *FLUID_NEWTONIAN_FIELDS,
        *FLUID_RELATIVISTIC_FIELDS,
        MAGNETIC_FIELD,
        CT_MAGNETIC_FIELD,
        PASSIVE_SCALAR_FIELD,
        *PHYSICAL_FLUID_FIELDS,
        PHYSICAL_MAGNETIC_FIELD,
        PHYSICAL_PASSIVE_SCALAR_FIELD,
        *ADM_INITIAL_DATA_FIELDS,
        *Z4C_FIELDS,
        *AUXILIARY_FIELDS,
    )


def component_symbols(
    fields: tuple[GridField, ...],
    *,
    owner: str,
) -> dict[str, sp.Symbol]:
    """Map canonical component IDs to symbols used by generated C++ kernels."""
    symbols: dict[str, sp.Symbol] = {}
    for field in fields:
        for component in field.components:
            if component.cpp_id is None:
                continue
            if component.cpp_id in symbols:
                raise ValueError(f"duplicate component ID {component.cpp_id}")
            symbols[component.cpp_id] = sp.Symbol(component.cpp_symbol(owner))
    return symbols


def validate_field_catalog() -> None:
    auxiliary_ids = [field.aux_id for field in AUXILIARY_FIELDS]
    if any(aux_id is None for aux_id in auxiliary_ids):
        raise ValueError("every auxiliary field requires an identifier")
    if len(auxiliary_ids) != len(set(auxiliary_ids)):
        raise ValueError("auxiliary field identifiers must be unique")
    auxiliary_component_names = component_names(AUXILIARY_FIELDS)
    if len(auxiliary_component_names) != len(set(auxiliary_component_names)):
        raise ValueError("auxiliary field component names must be unique")
    for family in (FLUID_NEWTONIAN_FIELDS, FLUID_RELATIVISTIC_FIELDS, Z4C_FIELDS):
        family_component_names = component_names(family)
        if len(family_component_names) != len(set(family_component_names)):
            raise ValueError("component names must be unique within a field set")
        if any(component.cpp_id is None for field in family for component in field.components):
            raise ValueError("evolved field components require C++ identifiers")


validate_field_catalog()
