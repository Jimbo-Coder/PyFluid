from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "methods.toml"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RotationLaw:
    name: str
    cpp_id: str
    label: str
    description: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class ReconstructionMethod:
    name: str
    cpp_id: str
    label: str
    description: str
    stencil_radius: int
    characteristic: bool


def _reject_unknown_keys(
    record: dict[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")


def _required_string(record: dict[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a nonempty string")
    return value


def _load_rotation_law(record: dict[str, Any], index: int) -> RotationLaw:
    context = f"rotation_law[{index}]"
    _reject_unknown_keys(
        record,
        {"name", "cpp_id", "label", "description", "parameters"},
        context,
    )
    parameters = record.get("parameters")
    if (
        not isinstance(parameters, list)
        or not parameters
        or not all(isinstance(parameter, str) and parameter for parameter in parameters)
    ):
        raise ValueError(f"{context}.parameters must be a nonempty string array")
    if len(parameters) != len(set(parameters)):
        raise ValueError(f"{context}.parameters contains duplicates")
    return RotationLaw(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        parameters=tuple(parameters),
    )


def _load_reconstruction(
    record: dict[str, Any], index: int
) -> ReconstructionMethod:
    context = f"reconstruction[{index}]"
    _reject_unknown_keys(
        record,
        {
            "name",
            "cpp_id",
            "label",
            "description",
            "stencil_radius",
            "characteristic",
        },
        context,
    )
    stencil_radius = record.get("stencil_radius")
    if type(stencil_radius) is not int or stencil_radius < 0:
        raise ValueError(f"{context}.stencil_radius must be a nonnegative integer")
    characteristic = record.get("characteristic")
    if type(characteristic) is not bool:
        raise ValueError(f"{context}.characteristic must be a boolean")
    if characteristic and stencil_radius < 1:
        raise ValueError(
            f"{context} characteristic reconstruction needs a nonzero stencil"
        )
    return ReconstructionMethod(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        stencil_radius=stencil_radius,
        characteristic=characteristic,
    )


def _validate_unique_ids(records: tuple[Any, ...], context: str) -> None:
    names = [record.name for record in records]
    cpp_ids = [record.cpp_id for record in records]
    if len(names) != len(set(names)):
        raise ValueError(f"{context} names must be unique")
    if len(cpp_ids) != len(set(cpp_ids)):
        raise ValueError(f"{context} cpp_id values must be unique")


def _load_schema() -> tuple[
    tuple[ReconstructionMethod, ...], tuple[RotationLaw, ...]
]:
    with SCHEMA_PATH.open("rb") as stream:
        schema = tomllib.load(stream)
    _reject_unknown_keys(
        schema,
        {"schema_version", "reconstruction", "rotation_law"},
        "methods schema",
    )
    schema_version = schema.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported methods schema version {schema_version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    reconstruction_records = schema.get("reconstruction")
    if not isinstance(reconstruction_records, list) or not reconstruction_records:
        raise ValueError("reconstruction must be a nonempty array of tables")
    reconstruction_methods = tuple(
        _load_reconstruction(record, index)
        for index, record in enumerate(reconstruction_records)
        if isinstance(record, dict)
    )
    if len(reconstruction_methods) != len(reconstruction_records):
        raise ValueError("reconstruction entries must be tables")
    _validate_unique_ids(reconstruction_methods, "reconstruction")

    rotation_records = schema.get("rotation_law")
    if not isinstance(rotation_records, list) or not rotation_records:
        raise ValueError("rotation_law must be a nonempty array of tables")
    laws = tuple(
        _load_rotation_law(record, index)
        for index, record in enumerate(rotation_records)
        if isinstance(record, dict)
    )
    if len(laws) != len(rotation_records):
        raise ValueError("rotation_law entries must be tables")
    _validate_unique_ids(laws, "rotation_law")
    return reconstruction_methods, laws


RECONSTRUCTION_METHODS, ROTATION_LAWS = _load_schema()
