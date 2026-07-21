"""Concise math labels shared by PyFluid plotting tools."""

from __future__ import annotations

import json
import re
from pathlib import Path


_FIELD_CATALOG = Path(__file__).resolve().parents[1] / "generated" / "field_catalog.json"
with _FIELD_CATALOG.open() as catalog_file:
    _CATALOG = json.load(catalog_file)

_FIELD_SYMBOLS = _CATALOG["component_labels"]
_DIAGNOSTIC_SYMBOLS = {
    diagnostic["name"]: diagnostic["label"]
    for diagnostic in _CATALOG["diagnostics"]
}
_DIAGNOSTIC_FAMILIES: dict[str, list[dict[str, object]]] = {}
for _diagnostic in _CATALOG["diagnostics"]:
    _DIAGNOSTIC_FAMILIES.setdefault(
        str(_diagnostic["family"]), []
    ).append(_diagnostic)
for _components in _DIAGNOSTIC_FAMILIES.values():
    _components.sort(key=lambda component: int(component["component_index"]))
_DERIVED_SYMBOLS = {
    "dye": "C",
}

_VECTOR_PREFIXES = {
    "v": "v",
    "mom": r"\rho v",
    "b": "B",
    "j": "J",
    "e": "E",
    "current": "J",
    "vorticity": r"\omega",
    "grad_phi": r"\partial\Phi",
    "solid_aperture": "A",
}


def component_math(component: str) -> str:
    """Return a math-mode expression without surrounding dollar signs."""
    if component.startswith("abs_"):
        return rf"\left| {component_math(component.removeprefix('abs_'))} \right|"
    if component.endswith("_mag"):
        prefix = component.removesuffix("_mag")
        symbol = _VECTOR_PREFIXES.get(prefix, rf"\mathrm{{{prefix.replace('_', r'\_')}}}")
        return rf"\left| \mathbf{{{symbol}}} \right|"
    if component in _FIELD_SYMBOLS:
        return _FIELD_SYMBOLS[component]
    if component in _DIAGNOSTIC_SYMBOLS:
        return _DIAGNOSTIC_SYMBOLS[component]
    if component in _DERIVED_SYMBOLS:
        return _DERIVED_SYMBOLS[component]
    match = re.fullmatch(r"([a-z_]+)([123])", component)
    if match:
        prefix, direction = match.groups()
        symbol = _VECTOR_PREFIXES.get(prefix)
        if symbol is not None:
            return rf"({symbol})_{{{direction}}}" if prefix in {"mom", "grad_phi"} else rf"{symbol}_{{{direction}}}"
    escaped = component.replace("_", r"\_")
    return rf"\mathrm{{{escaped}}}"


def diagnostic_family_components(family: str) -> tuple[str, ...]:
    """Return catalog-ordered scalar columns belonging to one typed family."""
    return tuple(
        str(component["name"])
        for component in _DIAGNOSTIC_FAMILIES.get(family, ())
    )


def field_label(
    component: str,
    *,
    log10: bool = False,
    normalization_reference: str | None = None,
    subtract_constant: float | None = None,
) -> str:
    """Return a complete Matplotlib mathtext label for a plotted field."""
    expression = component_math(component)
    if normalization_reference is not None:
        expression = rf"\frac{{{expression}}}{{{component_math(normalization_reference)}_0}}"
    if subtract_constant is not None:
        expression = rf"{expression} - {subtract_constant:g}"
    if log10:
        expression = rf"\log_{{10}}\!\left({expression}\right)"
    return f"${expression}$"
