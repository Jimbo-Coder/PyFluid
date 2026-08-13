"""Offline material-contact correction pieces for coherent FV prototypes.

This module is research-only.  It combines three independent requirements:
the local jump must be dominated by the material characteristic, its density
profile must be unresolved on a five-cell stencil, and the replacement trace
comes from limited-slope PPM with classical density contact steepening.
System adapters supply the characteristic-purity calculation and may supply a
physical activation that makes the correction vanish in an inapplicable
limit, such as zero magnetization.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


Array = np.ndarray
CharacteristicFraction = Callable[[Array, Array], float]
ActivationWeight = Callable[[Array, Array], float]

# These are frozen research values, not registered numerical parameters.  The
# five-cell threshold separates the audited N >= 8 smooth waves from unresolved
# jumps, while the high purity power suppresses finite-amplitude shocks that
# project weakly onto a locally linearized material eigenvector.
DENSITY_ACTIVITY_TOLERANCE = 1.0e-10
DISCONTINUITY_ONSET = 0.2
DISCONTINUITY_WIDTH = 0.3
MATERIAL_PURITY_POWER = 12


def density_discontinuity_strength(density: Array) -> float:
    """Separate an unresolved jump from a resolved five-cell smooth profile."""

    scale = max(float(np.max(np.abs(density))), 1.0)
    if np.ptp(density) <= DENSITY_ACTIVITY_TOLERANCE * scale:
        return 0.0
    fourth_difference = float(
        density[0]
        - 4.0 * density[1]
        + 6.0 * density[2]
        - 4.0 * density[3]
        + density[4]
    )
    total_jump = float(np.sum(np.abs(np.diff(density))))
    indicator = abs(fourth_difference) / max(
        total_jump, 256.0 * np.finfo(float).eps * scale
    )
    return float(
        np.clip(
            (indicator - DISCONTINUITY_ONSET) / DISCONTINUITY_WIDTH,
            0.0,
            1.0,
        )
    )


def _limited_centered_slope(
    left: float,
    center: float,
    right: float,
) -> float:
    backward = center - left
    forward = right - center
    if backward * forward <= 0.0:
        return 0.0
    centered = 0.5 * (right - left)
    return math.copysign(
        min(abs(centered), 2.0 * abs(backward), 2.0 * abs(forward)),
        centered,
    )


def _ppm3_contact_cell(density: Array) -> tuple[float, float]:
    """Return limited PPM3 density traces with classical contact steepening."""

    slope_left = _limited_centered_slope(*density[:3])
    slope_center = _limited_centered_slope(*density[1:4])
    slope_right = _limited_centered_slope(*density[2:5])
    lower = 0.5 * (density[1] + density[2]) + (
        slope_left - slope_center
    ) / 6.0
    upper = 0.5 * (density[2] + density[3]) + (
        slope_center - slope_right
    ) / 6.0

    jump = float(density[3] - density[1])
    second_left = float(density[2] - 2.0 * density[1] + density[0])
    second_right = float(density[4] - 2.0 * density[3] + density[2])
    eta = 0.0
    if (
        second_left * second_right < 0.0
        and abs(jump) > 0.01 * min(abs(density[3]), abs(density[1]))
    ):
        eta_tilde = -(second_right - second_left) / (6.0 * jump)
        eta = float(np.clip(20.0 * (eta_tilde - 0.05), 0.0, 1.0))
    lower = (1.0 - eta) * lower + eta * (
        density[1] + 0.5 * slope_left
    )
    upper = (1.0 - eta) * upper + eta * (
        density[3] - 0.5 * slope_right
    )

    if (upper - density[2]) * (density[2] - lower) <= 0.0:
        return float(density[2]), float(density[2])
    difference = upper - lower
    curvature = 6.0 * difference * (
        density[2] - 0.5 * (lower + upper)
    )
    if curvature > difference * difference:
        lower = 3.0 * density[2] - 2.0 * upper
    elif curvature < -difference * difference:
        upper = 3.0 * density[2] - 2.0 * lower
    return float(lower), float(upper)


def ppm3_contact_density_pair(
    stencil: Array,
    density_component: int = 0,
) -> tuple[float, float]:
    density = stencil[:, density_component]
    _left_lower, left_upper = _ppm3_contact_cell(density[:5])
    right_lower, _right_upper = _ppm3_contact_cell(density[1:])
    return left_upper, right_lower


def density_trace_weights(
    stencil: Array,
    characteristic_fraction: CharacteristicFraction,
    density_component: int = 0,
    activation_weight: ActivationWeight | None = None,
) -> tuple[float, float]:
    """Return left/right correction weights from independent physical tests."""

    contact_weight = max(
        characteristic_fraction(stencil[cell], stencil[cell + 1])
        ** MATERIAL_PURITY_POWER
        * (
            1.0
            if activation_weight is None
            else float(
                np.clip(
                    activation_weight(stencil[cell], stencil[cell + 1]),
                    0.0,
                    1.0,
                )
            )
        )
        for cell in range(5)
    )
    density = stencil[:, density_component]
    return (
        contact_weight * density_discontinuity_strength(density[:5]),
        contact_weight * density_discontinuity_strength(density[1:]),
    )
