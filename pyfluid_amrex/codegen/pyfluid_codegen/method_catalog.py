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


def _load_schema() -> tuple[RotationLaw, ...]:
    with SCHEMA_PATH.open("rb") as stream:
        schema = tomllib.load(stream)
    _reject_unknown_keys(schema, {"schema_version", "rotation_law"}, "methods schema")
    schema_version = schema.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported methods schema version {schema_version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    records = schema.get("rotation_law")
    if not isinstance(records, list) or not records:
        raise ValueError("rotation_law must be a nonempty array of tables")
    laws = tuple(
        _load_rotation_law(record, index)
        for index, record in enumerate(records)
        if isinstance(record, dict)
    )
    if len(laws) != len(records):
        raise ValueError("rotation_law entries must be tables")
    names = [law.name for law in laws]
    cpp_ids = [law.cpp_id for law in laws]
    if len(names) != len(set(names)):
        raise ValueError("rotation_law names must be unique")
    if len(cpp_ids) != len(set(cpp_ids)):
        raise ValueError("rotation_law cpp_id values must be unique")
    return laws


ROTATION_LAWS = _load_schema()
