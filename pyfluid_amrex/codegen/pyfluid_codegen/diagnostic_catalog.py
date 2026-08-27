from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tomllib
from typing import Any


class DiagnosticEvaluation(str, Enum):
    RUNTIME_REDUCTION = "runtime_reduction"
    FIELD_OUTPUT = "field_output"
    POSTPROCESS = "postprocess"


class DiagnosticOperation(str, Enum):
    COORDINATE = "coordinate"
    VOLUME_INTEGRAL = "volume_integral"
    SURFACE_INTEGRAL = "surface_integral"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    LINF_NORM = "linf_norm"
    DERIVED = "derived"
    COUNTER = "counter"
    PERFORMANCE = "performance"
    FIELD = "field"


class DiagnosticStorage(str, Enum):
    REAL = "real"
    INTEGER = "integer"
    LONG = "long"


class DiagnosticValueKind(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"


@dataclass(frozen=True)
class ComplexModeDiagnostic:
    name: str
    label: str
    basis: str
    domain: str
    integrand: str
    requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class Diagnostic:
    name: str
    label: str
    evaluation: DiagnosticEvaluation
    operation: DiagnosticOperation = DiagnosticOperation.DERIVED
    storage: DiagnosticStorage = DiagnosticStorage.REAL
    groups: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    default: str = "0.0"
    family: str = ""
    value_kind: DiagnosticValueKind = DiagnosticValueKind.SCALAR
    component_index: int = 0
    component_count: int = 1


_SCHEMA_VERSION = 1
_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "schema" / "diagnostics.toml"
)


def _reject_unknown_keys(
    table: dict[str, Any], allowed: set[str], context: str
) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(
            f"{context} contains unknown keys: {', '.join(sorted(unknown))}"
        )


def _require_string(table: dict[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _string_tuple(
    table: dict[str, Any], key: str, context: str
) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(
            f"{context}.{key} must be an array of non-empty strings"
        )
    return tuple(value)


def _table_list(
    root: dict[str, Any], key: str
) -> tuple[dict[str, Any], ...]:
    value = root.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(
            f"diagnostics.toml {key} must be an array of tables"
        )
    return tuple(value)


def _enum_value(enum_type: type[Enum], value: str, context: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{context} must be one of: {choices}") from error


def _load_runtime_diagnostic(
    table: dict[str, Any], index: int
) -> tuple[Diagnostic, ...]:
    context = f"runtime[{index}]"
    kind = table.get("kind", DiagnosticValueKind.SCALAR.value)
    if not isinstance(kind, str):
        raise ValueError(f"{context}.kind must be a string")
    value_kind = _enum_value(
        DiagnosticValueKind, kind, f"{context}.kind"
    )
    if value_kind is DiagnosticValueKind.TENSOR:
        raise ValueError(
            f"{context}.kind tensor is not supported by scalars.tsv"
        )

    common_keys = {
        "kind",
        "operation",
        "storage",
        "groups",
        "dependencies",
        "requirements",
        "default",
    }
    identity_keys = (
        {"name", "label"}
        if value_kind is DiagnosticValueKind.SCALAR
        else {"family", "names", "labels"}
    )
    _reject_unknown_keys(table, common_keys | identity_keys, context)

    operation = _enum_value(
        DiagnosticOperation,
        _require_string(table, "operation", context),
        f"{context}.operation",
    )
    storage_name = table.get("storage", DiagnosticStorage.REAL.value)
    if not isinstance(storage_name, str) or not storage_name:
        raise ValueError(f"{context}.storage must be a non-empty string")
    storage = _enum_value(
        DiagnosticStorage, storage_name, f"{context}.storage"
    )
    groups = _string_tuple(table, "groups", context)
    dependencies = _string_tuple(table, "dependencies", context)
    requirements = _string_tuple(table, "requirements", context)
    default_value = table.get("default")
    if default_value is None:
        default = (
            "0.0" if storage is DiagnosticStorage.REAL else "0"
        )
    elif isinstance(default_value, str) and default_value:
        default = default_value
    else:
        raise ValueError(f"{context}.default must be a non-empty string")

    if value_kind is DiagnosticValueKind.SCALAR:
        name = _require_string(table, "name", context)
        return (
            Diagnostic(
                name=name,
                label=_require_string(table, "label", context),
                evaluation=DiagnosticEvaluation.RUNTIME_REDUCTION,
                operation=operation,
                storage=storage,
                groups=groups,
                dependencies=dependencies,
                requirements=requirements,
                default=default,
                family=name,
            ),
        )

    family = _require_string(table, "family", context)
    names = _string_tuple(table, "names", context)
    labels = _string_tuple(table, "labels", context)
    if not names or len(names) != len(labels):
        raise ValueError(
            f"{context} vector names and labels must have the same positive length"
        )
    component_count = len(names)
    return tuple(
        Diagnostic(
            name=name,
            label=label,
            evaluation=DiagnosticEvaluation.RUNTIME_REDUCTION,
            operation=operation,
            storage=storage,
            groups=groups,
            dependencies=dependencies,
            requirements=requirements,
            default=default,
            family=family,
            value_kind=DiagnosticValueKind.VECTOR,
            component_index=component_index,
            component_count=component_count,
        )
        for component_index, (name, label) in enumerate(
            zip(names, labels, strict=True)
        )
    )


def _load_complex_mode(
    table: dict[str, Any], index: int
) -> ComplexModeDiagnostic:
    context = f"complex_mode[{index}]"
    _reject_unknown_keys(
        table,
        {
            "name",
            "label",
            "basis",
            "domain",
            "integrand",
            "requirements",
        },
        context,
    )
    return ComplexModeDiagnostic(
        name=_require_string(table, "name", context),
        label=_require_string(table, "label", context),
        basis=_require_string(table, "basis", context),
        domain=_require_string(table, "domain", context),
        integrand=_require_string(table, "integrand", context),
        requirements=_string_tuple(table, "requirements", context),
    )


def _load_postprocess_diagnostic(
    table: dict[str, Any], index: int
) -> Diagnostic:
    context = f"postprocess[{index}]"
    _reject_unknown_keys(
        table, {"name", "label", "operation", "dependencies"}, context
    )
    return Diagnostic(
        name=_require_string(table, "name", context),
        label=_require_string(table, "label", context),
        evaluation=DiagnosticEvaluation.POSTPROCESS,
        operation=_enum_value(
            DiagnosticOperation,
            _require_string(table, "operation", context),
            f"{context}.operation",
        ),
        dependencies=_string_tuple(table, "dependencies", context),
    )


def _load_catalog() -> tuple[
    tuple[Diagnostic, ...],
    tuple[ComplexModeDiagnostic, ...],
    tuple[Diagnostic, ...],
]:
    with _CATALOG_PATH.open("rb") as catalog_file:
        root = tomllib.load(catalog_file)
    if not isinstance(root, dict):
        raise ValueError("diagnostics.toml root must be a table")
    _reject_unknown_keys(
        root,
        {"schema_version", "runtime", "complex_mode", "postprocess"},
        "diagnostics.toml",
    )
    if root.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            "diagnostics.toml schema_version must be "
            f"{_SCHEMA_VERSION}, got {root.get('schema_version')!r}"
        )

    runtime = tuple(
        diagnostic
        for index, table in enumerate(_table_list(root, "runtime"))
        for diagnostic in _load_runtime_diagnostic(table, index)
    )
    complex_modes = tuple(
        _load_complex_mode(table, index)
        for index, table in enumerate(_table_list(root, "complex_mode"))
    )
    postprocess = tuple(
        _load_postprocess_diagnostic(table, index)
        for index, table in enumerate(_table_list(root, "postprocess"))
    )
    if not runtime or not postprocess:
        raise ValueError(
            "diagnostics.toml runtime and postprocess catalogs must not be empty"
        )
    return runtime, complex_modes, postprocess


(
    RUNTIME_SCALAR_DIAGNOSTICS,
    COMPLEX_MODE_DIAGNOSTICS,
    POSTPROCESS_DIAGNOSTICS,
) = _load_catalog()


# "standard" preserves the compact scalars.tsv contract. Global GR charges,
# horizon integrals, and radiation diagnostics remain opt-in through groups.
_RUNTIME_GROUP_NAMES = tuple(
    dict.fromkeys(
        group
        for diagnostic in RUNTIME_SCALAR_DIAGNOSTICS
        for group in diagnostic.groups
    )
)
DIAGNOSTIC_GROUPS = {
    "standard": tuple(
        diagnostic.name
        for diagnostic in RUNTIME_SCALAR_DIAGNOSTICS
        if "global_gr" not in diagnostic.groups
        and "radiation" not in diagnostic.groups
        and "horizon" not in diagnostic.groups
    ),
    **{
        group: tuple(
            diagnostic.name
            for diagnostic in RUNTIME_SCALAR_DIAGNOSTICS
            if group in diagnostic.groups
        )
        for group in _RUNTIME_GROUP_NAMES
    },
}


def validate_diagnostic_catalog() -> None:
    diagnostics = (*RUNTIME_SCALAR_DIAGNOSTICS, *POSTPROCESS_DIAGNOSTICS)
    names = [diagnostic.name for diagnostic in diagnostics]
    if len(names) != len(set(names)):
        raise ValueError("diagnostic names must be unique")
    family_shapes: dict[str, tuple[DiagnosticValueKind, int]] = {}
    family_components: dict[str, set[int]] = {}
    for diagnostic in diagnostics:
        family = diagnostic.family or diagnostic.name
        shape = (diagnostic.value_kind, diagnostic.component_count)
        if family in family_shapes and family_shapes[family] != shape:
            raise ValueError(
                f"diagnostic family {family} has inconsistent shape"
            )
        family_shapes[family] = shape
        family_components.setdefault(family, set()).add(
            diagnostic.component_index
        )
    for family, (_, component_count) in family_shapes.items():
        expected = set(range(component_count))
        if family_components[family] != expected:
            raise ValueError(
                f"diagnostic family {family} has incomplete components"
            )
    runtime_names = {
        diagnostic.name for diagnostic in RUNTIME_SCALAR_DIAGNOSTICS
    }
    for group, members in DIAGNOSTIC_GROUPS.items():
        unknown = set(members) - runtime_names
        if unknown:
            raise ValueError(
                f"diagnostic group {group} contains unknown members: "
                f"{sorted(unknown)}"
            )
    mode_names = [
        diagnostic.name for diagnostic in COMPLEX_MODE_DIAGNOSTICS
    ]
    if len(mode_names) != len(set(mode_names)):
        raise ValueError("complex-mode diagnostic names must be unique")


validate_diagnostic_catalog()
