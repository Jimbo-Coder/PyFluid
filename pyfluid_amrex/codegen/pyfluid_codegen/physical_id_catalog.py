from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "physical_id.toml"
SCHEMA_VERSION = 1
VALUE_TYPES = {
    "string",
    "integer",
    "real",
    "boolean",
    "integer_vector",
    "real_vector",
}


@dataclass(frozen=True)
class NamedDescription:
    name: str
    description: str


@dataclass(frozen=True)
class MetadataKey:
    name: str
    cpp_id: str
    value_type: str
    section: str
    required: bool
    description: str


@dataclass(frozen=True)
class DynamicKey:
    prefix: str
    suffix: str
    index_count: int
    value_type: str
    section: str
    description: str


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


def _load_named(record: dict[str, Any], context: str) -> NamedDescription:
    _reject_unknown_keys(record, {"name", "description"}, context)
    return NamedDescription(
        name=_required_string(record, "name", context),
        description=_required_string(record, "description", context),
    )


def _load_metadata_key(record: dict[str, Any], index: int) -> MetadataKey:
    context = f"metadata_key[{index}]"
    _reject_unknown_keys(
        record,
        {"name", "cpp_id", "value_type", "section", "required", "description"},
        context,
    )
    value_type = _required_string(record, "value_type", context)
    if value_type not in VALUE_TYPES:
        raise ValueError(f"{context}.value_type is unsupported: {value_type}")
    required = record.get("required")
    if not isinstance(required, bool):
        raise ValueError(f"{context}.required must be a boolean")
    return MetadataKey(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        value_type=value_type,
        section=_required_string(record, "section", context),
        required=required,
        description=_required_string(record, "description", context),
    )


def _load_dynamic_key(record: dict[str, Any], index: int) -> DynamicKey:
    context = f"dynamic_key[{index}]"
    _reject_unknown_keys(
        record,
        {
            "prefix",
            "suffix",
            "index_count",
            "value_type",
            "section",
            "description",
        },
        context,
    )
    value_type = _required_string(record, "value_type", context)
    if value_type not in VALUE_TYPES:
        raise ValueError(f"{context}.value_type is unsupported: {value_type}")
    index_count = record.get("index_count")
    if not isinstance(index_count, int) or isinstance(index_count, bool) \
            or index_count < 1:
        raise ValueError(f"{context}.index_count must be a positive integer")
    suffix = record.get("suffix")
    if not isinstance(suffix, str):
        raise ValueError(f"{context}.suffix must be a string")
    return DynamicKey(
        prefix=_required_string(record, "prefix", context),
        suffix=suffix,
        index_count=index_count,
        value_type=value_type,
        section=_required_string(record, "section", context),
        description=_required_string(record, "description", context),
    )


def _load_schema() -> tuple[
    tuple[NamedDescription, ...],
    tuple[NamedDescription, ...],
    tuple[NamedDescription, ...],
    tuple[DynamicKey, ...],
    tuple[MetadataKey, ...],
]:
    with SCHEMA_PATH.open("rb") as stream:
        document = tomllib.load(stream)
    _reject_unknown_keys(
        document,
        {
            "schema_version",
            "component_role",
            "component_kind",
            "component_gravity",
            "dynamic_key",
            "metadata_key",
        },
        "physical_id",
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"physical-ID schema_version must be {SCHEMA_VERSION}"
        )
    roles = tuple(
        _load_named(record, f"component_role[{index}]")
        for index, record in enumerate(document.get("component_role", []))
    )
    kinds = tuple(
        _load_named(record, f"component_kind[{index}]")
        for index, record in enumerate(document.get("component_kind", []))
    )
    gravities = tuple(
        _load_named(record, f"component_gravity[{index}]")
        for index, record in enumerate(document.get("component_gravity", []))
    )
    dynamic_keys = tuple(
        _load_dynamic_key(record, index)
        for index, record in enumerate(document.get("dynamic_key", []))
    )
    metadata_keys = tuple(
        _load_metadata_key(record, index)
        for index, record in enumerate(document.get("metadata_key", []))
    )
    for label, values in (
        ("component_role", roles),
        ("component_kind", kinds),
        ("component_gravity", gravities),
    ):
        names = [value.name for value in values]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{label} names must be nonempty and unique")
    names = [key.name for key in metadata_keys]
    cpp_ids = [key.cpp_id for key in metadata_keys]
    patterns = [(key.prefix, key.suffix) for key in dynamic_keys]
    if not names or len(names) != len(set(names)):
        raise ValueError("metadata key names must be nonempty and unique")
    if len(cpp_ids) != len(set(cpp_ids)):
        raise ValueError("metadata key cpp_id values must be unique")
    if len(patterns) != len(set(patterns)):
        raise ValueError("dynamic metadata patterns must be unique")
    return roles, kinds, gravities, dynamic_keys, metadata_keys


(
    COMPONENT_ROLES,
    COMPONENT_KINDS,
    COMPONENT_GRAVITIES,
    DYNAMIC_KEYS,
    METADATA_KEYS,
) = _load_schema()
