"""Concise math labels shared by PyFluid plotting tools."""

from __future__ import annotations

import re


_SCALAR_SYMBOLS = {
    "rho": r"\rho",
    "pressure": "P",
    "energy": "E",
    "dye": "C",
    "dye_mass": r"\rho C",
    "divb": r"\nabla\!\cdot\!\mathbf{B}",
    "magnetic_energy": r"E_B",
    "mach": r"\mathcal{M}",
    "plasma_beta": r"\beta",
    "entropy": "K",
    "amr_tag_metric": r"\eta_{\mathrm{AMR}}",
    "phi": r"\Phi",
    "solid_mask": r"\chi_s",
    "solid_volume_fraction": r"f_s",
    "solid_swept_volume": r"\Delta V_s",
    "solid_phi": r"\phi_s",
    "schlieren": r"\left|\nabla\rho\right|",
    "velocity_mag": r"\left|\mathbf{v}\right|",
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
    if component in _SCALAR_SYMBOLS:
        return _SCALAR_SYMBOLS[component]
    match = re.fullmatch(r"([a-z_]+)([123])", component)
    if match:
        prefix, direction = match.groups()
        symbol = _VECTOR_PREFIXES.get(prefix)
        if symbol is not None:
            return rf"({symbol})_{{{direction}}}" if prefix in {"mom", "grad_phi"} else rf"{symbol}_{{{direction}}}"
    escaped = component.replace("_", r"\_")
    return rf"\mathrm{{{escaped}}}"


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
