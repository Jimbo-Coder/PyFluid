#!/usr/bin/env python3
"""Extend time-integrated in-cell physical waves to ideal Newtonian MHD.

This research harness keeps the existing characteristic MP5 reconstruction,
HLLD flux, and SSPRK3 smooth-region update.  It recognizes only an isolated
MHD discontinuity whose neighboring states are constant, whose conserved cell
average has one common subcell fraction, whose normal magnetic field agrees,
and whose complete state/flux jump satisfies Rankine--Hugoniot.

For an accepted wave, the face flux is integrated over the exact crossing time
during the step.  This is a one-dimensional precursor: it tests the coupled
MHD algebra and the normal-field invariant, not multidimensional CT or AMR.

Course-correction status: retain this isolated-wave carrier as a bounded
mechanism control only.  The current standalone detector does not activate
before the production Balsara-3 recovery or Balsara-4 ringing event, so it is
not a candidate production fix without a new oracle-free, pre-artifact birth
result.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_mhd_wave_reconstruction as mhd  # noqa: E402
import research_subcell_mhd_reconstruction as subcell_mhd  # noqa: E402
from research_constrained_riemann import (  # noqa: E402
    mhd_conserved,
    mhd_flux,
    mhd_hlld_flux,
)
from research_physical_wave_subcell import (  # noqa: E402
    PhysicalWave,
    scaled_residual,
)


Array = np.ndarray


# A physical-wave carrier must not be born from primitives whose pressure was
# recovered by subtracting nearly equal conserved-energy terms beyond the
# detector's nominal resolution. More permissive ULP allowances cannot
# separate exact contacts from distinctly represented pressure jumps.
PHYSICAL_WAVE_RELATIVE_TOLERANCE = 3.0e-10
MAX_RELATIVE_PRESSURE_RECOVERY_UNCERTAINTY = (
    PHYSICAL_WAVE_RELATIVE_TOLERANCE
)

# The same rule bounds the roundoff allowance introduced by a bulk Galilean
# boost. Above it, the state stays on the baseline FV path.
MAX_BOOST_CONDITIONING_UNCERTAINTY = PHYSICAL_WAVE_RELATIVE_TOLERANCE


@dataclasses.dataclass(frozen=True)
class Result:
    case: str
    method: str
    cells: int
    cfl: float
    completed: bool
    total_l1: float
    transverse_l1: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    transition_cells: int
    maximum_divergence_proxy: float
    mass_error: float
    momentum_error: float
    energy_error: float


@dataclasses.dataclass(frozen=True)
class WindowFit:
    start: int
    stop: int
    speed: float
    jump_residual: float
    aggregate_residual: float
    profile_residual: float
    plateau_residual: float


@dataclasses.dataclass(frozen=True)
class PhysicalWavePath:
    family: str
    transverse_field_radius: float | None = None
    transverse_rotation_angle: float | None = None
    walen_sign: int | None = None


@dataclasses.dataclass(frozen=True)
class PhysicalWaveBirth:
    wave: PhysicalWave
    path: PhysicalWavePath
    cell: int
    position: float
    location_kind: str


@dataclasses.dataclass(frozen=True)
class MhdNormalization:
    density_scale: float
    velocity_scale: float
    pressure_scale: float
    magnetic_scale: float
    velocity_origin: Array


@dataclasses.dataclass(frozen=True)
class MhdMixtureFit:
    left_fraction: float
    mixture_residual: float
    normalization: MhdNormalization
    left_state: Array
    right_state: Array


def maximum_componentwise_residual(
    value: Array,
    reference: Array,
    *scales: Array,
    scale_floor: float = np.finfo(float).tiny,
) -> float:
    """Return the strict worst relative residual of any component."""
    scale = np.zeros_like(np.asarray(value, dtype=float))
    for candidate in (value, reference, *scales):
        scale = np.maximum(scale, np.abs(np.asarray(candidate, dtype=float)))
    scale = np.maximum(scale, scale_floor)
    return float(
        np.max(
            np.abs(
                np.asarray(value, dtype=float)
                - np.asarray(reference, dtype=float)
            )
            / scale
        )
    )


def components_agree(
    value: Array,
    reference: Array,
    *scales: Array,
    relative_tolerance: float,
    roundoff_scale: Array | float,
    roundoff_factor: float = 256.0,
) -> bool:
    """Compare each component with relative and conditioning-aware floors."""
    value_array = np.asarray(value, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    physical_scale = np.maximum(np.abs(value_array), np.abs(reference_array))
    for candidate in scales:
        physical_scale = np.maximum(
            physical_scale,
            np.abs(np.asarray(candidate, dtype=float)),
        )
    tolerance = (
        relative_tolerance * physical_scale
        + roundoff_factor
        * np.finfo(float).eps
        * np.abs(np.asarray(roundoff_scale, dtype=float))
    )
    return bool(np.all(np.abs(value_array - reference_array) <= tolerance))


def primitive_component_roundoff_scales(
    *primitive_states: Array,
) -> Array:
    states = np.atleast_2d(np.asarray(primitive_states, dtype=float))
    component_scale = np.maximum(
        np.max(np.abs(states), axis=0),
        np.finfo(float).tiny,
    )
    # Coupled conserved updates and characteristic transforms can place
    # ordinary arithmetic roundoff into an analytically zero component.
    # Giving that component a subnormal tolerance makes an exact physical
    # plateau fail equality after one otherwise roundoff-accurate step.  Keep
    # this floor componentwise: scaling it by an unrelated large component
    # would hide a resolved normal-field or normal-velocity mismatch.
    return np.maximum(component_scale, 1.0)


def primitive_components_agree(
    value: Array,
    reference: Array,
    *scale_states: Array,
    gamma: float,
    relative_tolerance: float,
    indices: Array | list[int] | tuple[int, ...] | None = None,
    pressure_from_conserved: bool = False,
    roundoff_factor: float = 256.0,
) -> bool:
    all_states = (value, reference, *scale_states)
    roundoff_scales = primitive_component_roundoff_scales(*all_states)
    if pressure_from_conserved:
        for state in all_states:
            primitive = np.asarray(state, dtype=float)
            roundoff_scales[4] = max(
                roundoff_scales[4],
                float(primitive[0] * np.dot(primitive[1:4], primitive[1:4])),
                float(np.dot(primitive[5:8], primitive[5:8])),
            )
    if indices is None:
        selected = np.arange(8)
    else:
        selected = np.asarray(indices, dtype=int)
    return components_agree(
        np.asarray(value, dtype=float)[selected],
        np.asarray(reference, dtype=float)[selected],
        *(np.asarray(state, dtype=float)[selected] for state in scale_states),
        relative_tolerance=relative_tolerance,
        roundoff_scale=roundoff_scales[selected],
        roundoff_factor=roundoff_factor,
    )


def maximum_pressure_recovery_uncertainty(
    gamma: float,
    *primitive_states: Array,
    roundoff_factor: float = 32.0,
) -> float:
    """Estimate relative pressure uncertainty from conserved-energy loss."""
    maximum_uncertainty = 0.0
    for primitive_state in primitive_states:
        primitive = np.asarray(primitive_state, dtype=float)
        density = float(primitive[0])
        pressure = float(primitive[4])
        kinetic_energy = 0.5 * density * float(
            np.dot(primitive[1:4], primitive[1:4])
        )
        magnetic_energy = 0.5 * float(
            np.dot(primitive[5:8], primitive[5:8])
        )
        internal_energy = pressure / (gamma - 1.0)
        total_energy = internal_energy + kinetic_energy + magnetic_energy
        cancellation_scale = (
            abs(total_energy) + kinetic_energy + magnetic_energy
        )
        relative_uncertainty = (
            roundoff_factor
            * np.finfo(float).eps
            * (gamma - 1.0)
            * cancellation_scale
            / max(abs(pressure), np.finfo(float).tiny)
        )
        maximum_uncertainty = max(
            maximum_uncertainty,
            relative_uncertainty,
        )
    return maximum_uncertainty


def pressure_invariant_agrees(
    left: Array,
    right: Array,
    gamma: float,
    *,
    relative_tolerance: float,
) -> bool:
    """Compare pressure only when conserved-energy recovery resolves it."""
    recovery_uncertainty = maximum_pressure_recovery_uncertainty(
        gamma,
        left,
        right,
    )
    if recovery_uncertainty > relative_tolerance:
        return False
    pressure_scale = max(
        abs(float(left[4])),
        abs(float(right[4])),
        np.finfo(float).tiny,
    )
    tolerance = relative_tolerance * pressure_scale
    return abs(float(left[4]) - float(right[4])) <= tolerance


def mhd_primitive_normalization(
    gamma: float,
    *primitive_states: Array,
) -> MhdNormalization:
    states = np.atleast_2d(np.asarray(primitive_states, dtype=float))
    velocity_origin = np.mean(states[:, 1:4], axis=0)
    local_states = states.copy()
    local_states[:, 1:4] -= velocity_origin
    density_scale = max(float(np.max(states[:, 0])), np.finfo(float).tiny)
    velocity_scale = max(
        float(np.max(np.abs(local_states[:, 1:4]))),
        np.finfo(float).tiny,
    )
    for state in local_states:
        density = max(float(state[0]), np.finfo(float).tiny)
        pressure = max(float(state[4]), 0.0)
        velocity_scale = max(
            velocity_scale,
            math.sqrt(gamma * pressure / density),
            float(np.linalg.norm(state[5:8])) / math.sqrt(density),
        )
    pressure_scale = density_scale * velocity_scale**2
    magnetic_scale = math.sqrt(density_scale) * velocity_scale
    return MhdNormalization(
        density_scale=density_scale,
        velocity_scale=velocity_scale,
        pressure_scale=pressure_scale,
        magnetic_scale=magnetic_scale,
        velocity_origin=velocity_origin,
    )


def mhd_conserved_scales(
    normalization: MhdNormalization,
) -> Array:
    density_scale = normalization.density_scale
    velocity_scale = normalization.velocity_scale
    pressure_scale = normalization.pressure_scale
    magnetic_scale = normalization.magnetic_scale
    return np.array(
        (
            density_scale,
            density_scale * velocity_scale,
            density_scale * velocity_scale,
            density_scale * velocity_scale,
            pressure_scale,
            magnetic_scale,
            magnetic_scale,
            magnetic_scale,
        )
    )


def mhd_flux_scales(
    normalization: MhdNormalization,
) -> Array:
    density_scale = normalization.density_scale
    velocity_scale = normalization.velocity_scale
    pressure_scale = normalization.pressure_scale
    magnetic_scale = normalization.magnetic_scale
    return np.array(
        (
            density_scale * velocity_scale,
            pressure_scale,
            pressure_scale,
            pressure_scale,
            pressure_scale * velocity_scale,
            magnetic_scale * velocity_scale,
            magnetic_scale * velocity_scale,
            magnetic_scale * velocity_scale,
        )
    )


def mhd_boost_conditioning_uncertainty(
    normalization: MhdNormalization,
) -> float:
    boost_ratio = (
        float(np.linalg.norm(normalization.velocity_origin))
        / normalization.velocity_scale
    )
    return 2.0 * np.finfo(float).eps * (1.0 + boost_ratio**2)


def fit_mhd_conserved_mixture(
    average: Array,
    left: Array,
    right: Array,
    gamma: float,
    *,
    mixture_tolerance: float,
    allow_endpoint: bool = False,
) -> MhdMixtureFit | None:
    """Fit one dimensionless conserved mixture with conditioned C2P closure."""
    endpoints = np.array((left, right), dtype=float)
    observed_average = np.asarray(average, dtype=float)[None, :]
    if (
        not mhd.primitive_is_admissible(endpoints)
        or not mhd.primitive_is_admissible(observed_average)
    ):
        return None
    if (
        maximum_pressure_recovery_uncertainty(
            gamma,
            average,
            left,
            right,
        )
        > MAX_RELATIVE_PRESSURE_RECOVERY_UNCERTAINTY
    ):
        # The original total-energy state no longer contains enough pressure
        # information to distinguish a physical wave from a nearby corrupt
        # state.  Growing the detector tolerance here creates false births.
        return None
    normalization = mhd_primitive_normalization(gamma, left, right)
    boost_conditioning_uncertainty = mhd_boost_conditioning_uncertainty(
        normalization
    )
    if (
        boost_conditioning_uncertainty
        > MAX_BOOST_CONDITIONING_UNCERTAINTY
    ):
        return None
    conserved_scales = mhd_conserved_scales(normalization)
    local_endpoints = endpoints.copy()
    local_endpoints[:, 1:4] -= normalization.velocity_origin
    local_average = observed_average.copy()
    local_average[:, 1:4] -= normalization.velocity_origin
    left_state, right_state = (
        mhd.conserved_array(local_endpoints, gamma) / conserved_scales
    )
    average_state = (
        mhd.conserved_array(local_average, gamma)[0] / conserved_scales
    )
    state_jump = left_state - right_state
    jump_norm2 = float(np.dot(state_jump, state_jump))
    state_scale = max(
        float(np.dot(left_state, left_state)),
        float(np.dot(right_state, right_state)),
        np.finfo(float).tiny,
    )
    if jump_norm2 <= 256.0 * np.finfo(float).eps**2 * state_scale:
        return None
    ledger_components = np.array((0, 5, 6, 7))
    fit_jump = state_jump[ledger_components]
    fit_jump_norm2 = float(np.dot(fit_jump, fit_jump))
    if fit_jump_norm2 > 256.0 * np.finfo(float).eps**2 * state_scale:
        fit_average = average_state[ledger_components]
        fit_right = right_state[ledger_components]
    else:
        fit_jump = state_jump
        fit_jump_norm2 = jump_norm2
        fit_average = average_state
        fit_right = right_state
    left_fraction = float(
        np.dot(fit_average - fit_right, fit_jump) / fit_jump_norm2
    )
    endpoint_tolerance = 256.0 * np.finfo(float).eps
    if allow_endpoint:
        if not (
            -endpoint_tolerance
            <= left_fraction
            <= 1.0 + endpoint_tolerance
        ):
            return None
    elif not (
        endpoint_tolerance < left_fraction < 1.0 - endpoint_tolerance
    ):
        return None
    left_fraction = min(max(left_fraction, 0.0), 1.0)
    fitted_average = (
        left_fraction * left_state
        + (1.0 - left_fraction) * right_state
    )
    mixture_residual = scaled_residual(
        average_state - fitted_average,
        average_state,
        fitted_average,
        state_jump,
    )
    conditioned_mixture_tolerance = (
        mixture_tolerance
        + boost_conditioning_uncertainty
    )
    if mixture_residual > conditioned_mixture_tolerance:
        return None
    fitted_primitive = mhd.primitive_from_conserved(
        (fitted_average * conserved_scales)[None, :],
        gamma,
    )[0]
    fitted_primitive[1:4] += normalization.velocity_origin
    if (
        not mhd.primitive_is_admissible(fitted_primitive[None, :])
        or not primitive_components_agree(
            average,
            fitted_primitive,
            left,
            right,
            gamma=gamma,
            relative_tolerance=mixture_tolerance,
            pressure_from_conserved=True,
            roundoff_factor=32.0,
        )
    ):
        return None
    return MhdMixtureFit(
        left_fraction=left_fraction,
        mixture_residual=mixture_residual,
        normalization=normalization,
        left_state=left_state,
        right_state=right_state,
    )


def fit_physical_mhd_wave(
    average: Array,
    left: Array,
    right: Array,
    gamma: float,
    *,
    mixture_tolerance: float = PHYSICAL_WAVE_RELATIVE_TOLERANCE,
    jump_tolerance: float = PHYSICAL_WAVE_RELATIVE_TOLERANCE,
    allow_endpoint: bool = False,
) -> PhysicalWave | None:
    if not primitive_components_agree(
        left,
        right,
        gamma=gamma,
        relative_tolerance=3.0e-10,
        indices=(5,),
    ):
        return None
    mixture = fit_mhd_conserved_mixture(
        average,
        left,
        right,
        gamma,
        mixture_tolerance=mixture_tolerance,
        allow_endpoint=allow_endpoint,
    )
    if mixture is None:
        return None
    left_state = mixture.left_state
    right_state = mixture.right_state
    state_jump = left_state - right_state
    jump_norm2 = float(np.dot(state_jump, state_jump))
    flux_scales = mhd_flux_scales(mixture.normalization)
    local_left = np.asarray(left, dtype=float).copy()
    local_right = np.asarray(right, dtype=float).copy()
    local_left[1:4] -= mixture.normalization.velocity_origin
    local_right[1:4] -= mixture.normalization.velocity_origin
    left_flux = mhd_flux(local_left, gamma) / flux_scales
    right_flux = mhd_flux(local_right, gamma) / flux_scales
    flux_jump = left_flux - right_flux
    normalized_speed = float(np.dot(state_jump, flux_jump) / jump_norm2)
    jump_residual = maximum_componentwise_residual(
        flux_jump,
        normalized_speed * state_jump,
        scale_floor=1.0,
    )
    conditioned_jump_tolerance = (
        jump_tolerance
        + mhd_boost_conditioning_uncertainty(mixture.normalization)
    )
    if (
        not np.isfinite(normalized_speed)
        or jump_residual > conditioned_jump_tolerance
    ):
        return None
    speed = (
        normalized_speed * mixture.normalization.velocity_scale
        + mixture.normalization.velocity_origin[0]
    )
    return PhysicalWave(
        left=np.asarray(left, dtype=float).copy(),
        right=np.asarray(right, dtype=float).copy(),
        speed=speed,
        left_fraction=mixture.left_fraction,
        mixture_residual=mixture.mixture_residual,
        jump_residual=jump_residual,
    )


def local_primitive(
    primitive: Array,
    cell: int,
    periodic: bool,
) -> Array:
    return np.asarray(
        mhd.stencil_value(primitive, cell, periodic),
        dtype=float,
    )


def states_agree(left: Array, right: Array, gamma: float) -> bool:
    return primitive_components_agree(
        left,
        right,
        gamma=gamma,
        relative_tolerance=3.0e-10,
    )


def mhd_characteristic_speeds(primitive: Array, gamma: float) -> Array:
    rho = float(primitive[0])
    normal_velocity = float(primitive[1])
    pressure = float(primitive[4])
    normal_b = float(primitive[5])
    transverse_b2 = float(primitive[6] ** 2 + primitive[7] ** 2)
    sound2 = gamma * pressure / rho
    magnetic2 = (normal_b**2 + transverse_b2) / rho
    normal_alfven2 = normal_b**2 / rho
    discriminant = max(
        (sound2 + magnetic2) ** 2
        - 4.0 * sound2 * normal_alfven2,
        0.0,
    )
    fast2 = 0.5 * (
        sound2 + magnetic2 + math.sqrt(discriminant)
    )
    slow2 = 0.5 * (
        sound2 + magnetic2 - math.sqrt(discriminant)
    )
    alfven = math.sqrt(max(normal_alfven2, 0.0))
    fast = math.sqrt(max(fast2, 0.0))
    slow = math.sqrt(max(slow2, 0.0))
    return np.array(
        (
            normal_velocity - fast,
            normal_velocity - alfven,
            normal_velocity - slow,
            normal_velocity,
            normal_velocity + slow,
            normal_velocity + alfven,
            normal_velocity + fast,
        )
    )


def classify_physical_mhd_wave(
    wave: PhysicalWave,
    gamma: float,
) -> PhysicalWavePath | None:
    """Accept a contact, canonical Alfvén endpoint path, or fast Lax shock."""
    left = wave.left
    right = wave.right
    primitive_scales = primitive_component_roundoff_scales(left, right)
    speed_scale = max(
        abs(wave.speed - left[1]),
        abs(wave.speed - right[1]),
        abs(left[5]) / math.sqrt(left[0]),
        abs(right[5]) / math.sqrt(right[0]),
        math.sqrt(gamma * left[4] / left[0]),
        math.sqrt(gamma * right[4] / right[0]),
        np.finfo(float).tiny,
    )
    tolerance = speed_scale * (
        3.0e-9 + 256.0 * np.finfo(float).eps
    )

    if (
        primitive_components_agree(
            left,
            right,
            gamma=gamma,
            relative_tolerance=3.0e-9,
            indices=(1, 2, 3, 5, 6, 7),
        )
        and pressure_invariant_agrees(
            left,
            right,
            gamma,
            relative_tolerance=3.0e-9,
        )
        and abs(wave.speed - left[1]) <= tolerance
        and abs(wave.speed - right[1]) <= tolerance
    ):
        return PhysicalWavePath(family="contact")

    transverse_left = np.asarray(left[6:8], dtype=float)
    transverse_right = np.asarray(right[6:8], dtype=float)
    radius_left = float(np.linalg.norm(transverse_left))
    radius_right = float(np.linalg.norm(transverse_right))
    transverse_velocity_jump = right[2:4] - left[2:4]
    transverse_field_jump = right[6:8] - left[6:8]
    walen_signs = []
    for walen_sign in (-1, 1):
        speed_left = (
            left[1]
            - walen_sign * left[5] / math.sqrt(left[0])
        )
        speed_right = (
            right[1]
            - walen_sign * right[5] / math.sqrt(right[0])
        )
        walen_matches = components_agree(
            transverse_velocity_jump,
            walen_sign
            * transverse_field_jump
            / math.sqrt(0.5 * (left[0] + right[0])),
            relative_tolerance=3.0e-9,
            roundoff_scale=primitive_scales[2:4],
        )
        if (
            abs(wave.speed - speed_left) <= tolerance
            and abs(wave.speed - speed_right) <= tolerance
            and walen_matches
        ):
            walen_signs.append(walen_sign)
    radius_scale = max(radius_left, radius_right, np.finfo(float).tiny)
    radius_tolerance = (
        3.0e-9 * radius_scale
        + 256.0
        * np.finfo(float).eps
        * max(float(np.max(primitive_scales[6:8])), np.finfo(float).tiny)
    )
    normal_b_scale = max(abs(left[5]), abs(right[5]), np.finfo(float).tiny)
    normal_b_tolerance = (
        3.0e-9 * normal_b_scale
        + 256.0 * np.finfo(float).eps * primitive_scales[5]
    )
    if (
        primitive_components_agree(
            left,
            right,
            gamma=gamma,
            relative_tolerance=3.0e-9,
            indices=(0, 1, 5),
        )
        and pressure_invariant_agrees(
            left,
            right,
            gamma,
            relative_tolerance=3.0e-9,
        )
        and abs(0.5 * (left[5] + right[5])) > normal_b_tolerance
        and abs(radius_left - radius_right) <= radius_tolerance
        and radius_left > radius_tolerance
        and float(np.linalg.norm(transverse_left - transverse_right))
        > radius_tolerance
        and len(walen_signs) == 1
    ):
        cross = (
            transverse_left[0] * transverse_right[1]
            - transverse_left[1] * transverse_right[0]
        )
        dot = float(np.dot(transverse_left, transverse_right))
        rotation_angle = math.atan2(cross, dot)
        if abs(abs(rotation_angle) - math.pi) <= 3.0e-9:
            # The two signed half-turn arcs have identical endpoints.  Do not
            # let roundoff in sin(pi) masquerade as physical path selection.
            rotation_angle = math.pi
        return PhysicalWavePath(
            family=(
                "alfven_minus"
                if wave.speed < 0.5 * (left[1] + right[1])
                else "alfven_plus"
            ),
            transverse_field_radius=0.5 * (radius_left + radius_right),
            transverse_rotation_angle=rotation_angle,
            walen_sign=walen_signs[0],
        )

    left_speeds = mhd_characteristic_speeds(left, gamma)
    right_speeds = mhd_characteristic_speeds(right, gamma)
    lax_families: list[int] = []
    for family in (0, 6):
        if not (
            left_speeds[family] > wave.speed + tolerance
            and wave.speed > right_speeds[family] + tolerance
        ):
            continue
        slower_exits = (
            family == 0
            or left_speeds[family - 1] < wave.speed - tolerance
        )
        faster_exits = (
            family == 6
            or wave.speed < right_speeds[family + 1] - tolerance
        )
        if family == 0:
            downstream = right
            upstream = left
        else:
            downstream = left
            upstream = right
        log_pressure_ratio = math.log(downstream[4] / upstream[4])
        log_density_ratio = math.log(downstream[0] / upstream[0])
        log_entropy_increase = (
            log_pressure_ratio - gamma * log_density_ratio
        )
        entropy_roundoff = (
            256.0
            * np.finfo(float).eps
            * (
                1.0
                + abs(log_pressure_ratio)
                + gamma * abs(log_density_ratio)
            )
        )
        entropy_increases = log_entropy_increase > entropy_roundoff
        if slower_exits and faster_exits and entropy_increases:
            lax_families.append(family)
    if len(lax_families) == 1:
        return PhysicalWavePath(
            family=(
                "fast_minus_shock"
                if lax_families[0] == 0
                else "fast_plus_shock"
            )
        )
    return None


def fit_cell_wave(
    primitive: Array,
    cell: int,
    gamma: float,
    periodic: bool,
) -> PhysicalWave | None:
    left_outer = local_primitive(primitive, cell - 2, periodic)
    left = local_primitive(primitive, cell - 1, periodic)
    average = local_primitive(primitive, cell, periodic)
    right = local_primitive(primitive, cell + 1, periodic)
    right_outer = local_primitive(primitive, cell + 2, periodic)
    if not states_agree(left_outer, left, gamma):
        return None
    if not states_agree(right, right_outer, gamma):
        return None
    return fit_physical_mhd_wave(average, left, right, gamma)


def fit_face_wave(
    primitive: Array,
    left_cell: int,
    gamma: float,
    periodic: bool,
) -> PhysicalWave | None:
    left_outer = local_primitive(primitive, left_cell - 1, periodic)
    left = local_primitive(primitive, left_cell, periodic)
    right = local_primitive(primitive, left_cell + 1, periodic)
    right_outer = local_primitive(primitive, left_cell + 2, periodic)
    if not states_agree(left_outer, left, gamma):
        return None
    if not states_agree(right, right_outer, gamma):
        return None
    return fit_physical_mhd_wave(
        left,
        left,
        right,
        gamma,
        allow_endpoint=True,
    )


def find_single_physical_wave_birth(
    primitive: Array,
    gamma: float,
    periodic: bool,
) -> PhysicalWaveBirth | None:
    """Infer one isolated nonperiodic jump from finite-volume state alone."""
    if periodic:
        # A periodic piecewise-constant state necessarily contains a return
        # jump.  This one-element prototype does not yet own paired topology.
        return None
    cells = len(primitive)
    dx = 1.0 / cells
    births: list[PhysicalWaveBirth] = []
    for cell in range(2, cells - 2):
        wave = fit_cell_wave(
            primitive,
            cell,
            gamma,
            periodic,
        )
        if wave is None:
            continue
        path = classify_physical_mhd_wave(wave, gamma)
        if path is None:
            continue
        position = (cell + wave.left_fraction) * dx
        if periodic:
            position %= 1.0
        elif not (cell * dx < position < (cell + 1) * dx):
            continue
        births.append(
            PhysicalWaveBirth(
                wave=wave,
                path=path,
                cell=cell,
                position=position,
                location_kind="mixed_cell",
            )
        )

    for face in range(2, cells - 1):
        wave = fit_face_wave(
            primitive,
            face - 1,
            gamma,
            periodic=False,
        )
        if wave is None:
            continue
        path = classify_physical_mhd_wave(wave, gamma)
        if path is None:
            continue
        births.append(
            PhysicalWaveBirth(
                wave=wave,
                path=path,
                cell=face,
                position=face * dx,
                location_kind="face",
            )
        )
    if len(births) != 1:
        return None
    return births[0]


def physical_wave_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    periodic = bool(physical_wave_mp5.periodic)
    left_wave = fit_cell_wave(
        primitive,
        left_cell,
        gamma,
        periodic,
    )
    right_wave = fit_cell_wave(
        primitive,
        left_cell + 1,
        gamma,
        periodic,
    )
    if left_wave is not None and right_wave is None:
        return left_wave.right.copy(), left_wave.right.copy()
    if right_wave is not None and left_wave is None:
        return right_wave.left.copy(), right_wave.left.copy()
    if left_wave is not None and right_wave is not None:
        if states_agree(left_wave.right, right_wave.left, gamma):
            common = 0.5 * (left_wave.right + right_wave.left)
            common[5] = left_wave.right[5]
            return common.copy(), common.copy()
        return mhd.characteristic_mp5(primitive, left_cell, gamma)
    face_wave = fit_face_wave(
        primitive,
        left_cell,
        gamma,
        periodic,
    )
    if face_wave is not None:
        upwind = face_wave.left if face_wave.speed >= 0.0 else face_wave.right
        return upwind.copy(), upwind.copy()
    return mhd.characteristic_mp5(primitive, left_cell, gamma)


physical_wave_mp5.periodic = False


def rhs_with_fluxes(
    state: Array,
    dx: float,
    gamma: float,
    periodic: bool,
) -> tuple[Array, float, Array]:
    primitive = mhd.primitive_from_conserved(state, gamma)
    if not mhd.primitive_is_admissible(primitive):
        return (
            np.full_like(state, np.nan),
            math.inf,
            np.full((len(state) + 1, 8), np.nan),
        )
    physical_wave_mp5.periodic = periodic
    fluxes = np.empty((len(state) + 1, 8))
    required_speed = 0.0
    for face in range(len(state) + 1):
        left, right = physical_wave_mp5(primitive, face - 1, gamma)
        if not mhd.primitive_is_admissible(np.array((left, right))):
            return np.full_like(state, np.nan), math.inf, fluxes
        face_flux = mhd_hlld_flux(left, right, gamma)
        if not np.all(np.isfinite(face_flux)):
            return np.full_like(state, np.nan), math.inf, fluxes
        fluxes[face] = face_flux
        required_speed = max(
            required_speed,
            abs(left[1]) + mhd.mhd_fast_speed(left, gamma),
            abs(right[1]) + mhd.mhd_fast_speed(right, gamma),
        )
    if periodic:
        fluxes[-1] = fluxes[0]
    return (
        -(fluxes[1:] - fluxes[:-1]) / dx,
        required_speed,
        fluxes,
    )


def physical_flux_overrides(
    primitive: Array,
    dx: float,
    dt: float,
    gamma: float,
    periodic: bool,
) -> dict[int, Array]:
    cells = len(primitive)
    overrides: dict[int, Array] = {}

    def face_index(face: int) -> int | None:
        if periodic:
            return face % cells
        return face if 0 <= face <= cells else None

    def add_override(face: int, flux: Array) -> None:
        resolved = face_index(face)
        if resolved is None:
            return
        existing = overrides.get(resolved)
        if existing is None:
            overrides[resolved] = flux.copy()
            return
        tolerance = 3.0e-10 * max(
            float(np.linalg.norm(existing)),
            float(np.linalg.norm(flux)),
            1.0,
        )
        if np.linalg.norm(existing - flux) > tolerance:
            overrides.pop(resolved, None)

    def add_plateau_face(face: int, state: Array, flux: Array) -> None:
        left_state = local_primitive(primitive, face - 1, periodic)
        right_state = local_primitive(primitive, face, periodic)
        if states_agree(left_state, state, gamma) and states_agree(
            right_state,
            state,
            gamma,
        ):
            add_override(face, flux)

    fitted_cells: set[int] = set()
    for cell in range(cells):
        wave = fit_cell_wave(primitive, cell, gamma, periodic)
        if wave is None or classify_physical_mhd_wave(wave, gamma) is None:
            continue
        fitted_cells.add(cell)
        left_flux = mhd_flux(wave.left, gamma)
        right_flux = mhd_flux(wave.right, gamma)
        left_average_flux = left_flux
        right_average_flux = right_flux
        if wave.speed > 0.0:
            crossing_time = (1.0 - wave.left_fraction) * dx / wave.speed
            if crossing_time < dt:
                crossed = min(max(1.0 - crossing_time / dt, 0.0), 1.0)
                right_average_flux = (
                    (1.0 - crossed) * right_flux + crossed * left_flux
                )
        elif wave.speed < 0.0:
            crossing_time = wave.left_fraction * dx / (-wave.speed)
            if crossing_time < dt:
                crossed = min(max(1.0 - crossing_time / dt, 0.0), 1.0)
                left_average_flux = (
                    (1.0 - crossed) * left_flux + crossed * right_flux
                )
        add_override(cell, left_average_flux)
        add_override(cell + 1, right_average_flux)
        for offset in range(1, 4):
            add_plateau_face(cell - offset, wave.left, left_flux)
            add_plateau_face(cell + 1 + offset, wave.right, right_flux)

    for left_cell in range(-1 if not periodic else 0, cells):
        left_index = left_cell % cells
        right_index = (left_cell + 1) % cells
        if left_index in fitted_cells or right_index in fitted_cells:
            continue
        wave = fit_face_wave(primitive, left_cell, gamma, periodic)
        if wave is None or classify_physical_mhd_wave(wave, gamma) is None:
            continue
        upwind = wave.left if wave.speed >= 0.0 else wave.right
        add_override(left_cell + 1, mhd_flux(upwind, gamma))
        left_flux = mhd_flux(wave.left, gamma)
        right_flux = mhd_flux(wave.right, gamma)
        for offset in range(1, 4):
            add_plateau_face(
                left_cell + 1 - offset,
                wave.left,
                left_flux,
            )
            add_plateau_face(
                left_cell + 1 + offset,
                wave.right,
                right_flux,
            )
    if periodic and 0 in overrides:
        overrides[cells] = overrides[0].copy()
    return overrides


def evolve_space_time(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
) -> tuple[Array, bool]:
    state = mhd.conserved_array(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    while time < final_time:
        current = mhd.primitive_from_conserved(state, gamma)
        if not mhd.primitive_is_admissible(current):
            return current, False
        physical_speeds = np.array(
            [
                abs(row[1]) + mhd.mhd_fast_speed(row, gamma)
                for row in current
            ]
        )
        rhs0, speed0, flux0 = rhs_with_fluxes(
            state,
            dx,
            gamma,
            periodic,
        )
        if not np.isfinite(speed0) or not np.all(np.isfinite(rhs0)):
            return current, False
        dt = min(
            final_time - time,
            cfl * dx / max(float(np.max(physical_speeds)), speed0),
        )
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1, flux1 = rhs_with_fluxes(
                stage1,
                dx,
                gamma,
                periodic,
            )
            if not np.isfinite(speed1):
                dt *= 0.5
                continue
            reduced_dt = min(
                dt,
                final_time - time,
                cfl * dx / max(speed0, speed1),
            )
            if reduced_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * reduced_dt
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, speed2, flux2 = rhs_with_fluxes(
                stage2,
                dx,
                gamma,
                periodic,
            )
            if not np.isfinite(speed2):
                dt *= 0.5
                continue
            reduced_dt = min(
                dt,
                final_time - time,
                cfl * dx / max(speed0, speed1, speed2),
            )
            if reduced_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * reduced_dt
                continue
            integrated_flux = (
                flux0 / 6.0 + flux1 / 6.0 + 2.0 * flux2 / 3.0
            )
            for face, flux in physical_flux_overrides(
                current,
                dx,
                dt,
                gamma,
                periodic,
            ).items():
                integrated_flux[face] = flux
            candidate = state - dt * (
                integrated_flux[1:] - integrated_flux[:-1]
            ) / dx
            candidate_primitive = mhd.primitive_from_conserved(
                candidate,
                gamma,
            )
            if not mhd.primitive_is_admissible(candidate_primitive):
                dt *= 0.5
                continue
            state = candidate
            break
        else:
            return mhd.primitive_from_conserved(state, gamma), False
        time += dt
    final = mhd.primitive_from_conserved(state, gamma)
    return final, mhd.primitive_is_admissible(final)


def evolve_persistent_wave(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    left: Array,
    right: Array,
    speed: float,
    initial_position: float,
) -> tuple[Array, bool]:
    """Carry one accepted physical jump without re-detecting its FV average."""
    fitted_wave = fit_physical_mhd_wave(
        left,
        left,
        right,
        gamma,
        allow_endpoint=True,
    )
    if fitted_wave is None or not math.isclose(
        fitted_wave.speed,
        speed,
        rel_tol=3.0e-10,
        abs_tol=3.0e-10,
    ):
        raise ValueError(
            "persistent wave must satisfy the complete physical jump"
        )
    state = mhd.conserved_array(primitive, gamma)
    cells = len(state)
    dx = 1.0 / cells
    faces = np.linspace(0.0, 1.0, cells + 1)
    left_flux = mhd_flux(left, gamma)
    right_flux = mhd_flux(right, gamma)
    signal_speed = max(
        abs(left[1]) + mhd.mhd_fast_speed(left, gamma),
        abs(right[1]) + mhd.mhd_fast_speed(right, gamma),
    )
    position = initial_position
    time_value = 0.0
    while time_value < final_time:
        dt = min(
            final_time - time_value,
            cfl * dx / signal_speed,
        )
        if speed > 0.0:
            crossing_time = (faces - position) / speed
            left_fraction = np.clip(
                (dt - crossing_time) / dt,
                0.0,
                1.0,
            )
        elif speed < 0.0:
            crossing_time = (faces - position) / speed
            left_fraction = np.clip(crossing_time / dt, 0.0, 1.0)
        else:
            left_fraction = (faces < position).astype(float)
        integrated_flux = (
            left_fraction[:, None] * left_flux[None, :]
            + (1.0 - left_fraction[:, None]) * right_flux[None, :]
        )
        candidate = state - dt * (
            integrated_flux[1:] - integrated_flux[:-1]
        ) / dx
        candidate_primitive = mhd.primitive_from_conserved(
            candidate,
            gamma,
        )
        if not mhd.primitive_is_admissible(candidate_primitive):
            return candidate_primitive, False
        state = candidate
        position += speed * dt
        time_value += dt
    final = mhd.primitive_from_conserved(state, gamma)
    return final, mhd.primitive_is_admissible(final)


def evolve_born_persistent_wave(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
) -> tuple[Array, bool]:
    """Birth one verified jump from the FV state, then carry it causally."""
    birth = find_single_physical_wave_birth(
        primitive,
        gamma,
        periodic,
    )
    if birth is None:
        return mhd.evolve(
            primitive,
            final_time,
            cfl,
            gamma,
            mhd.characteristic_mp5,
            mhd_hlld_flux,
            periodic=periodic,
        )
    return evolve_persistent_wave(
        primitive,
        final_time,
        cfl,
        gamma,
        birth.wave.left,
        birth.wave.right,
        birth.wave.speed,
        birth.position,
    )


def discontinuity_cell_averages(
    cells: int,
    position: float,
    left: Array,
    right: Array,
    gamma: float,
) -> Array:
    edges = np.linspace(0.0, 1.0, cells + 1)
    left_state, right_state = mhd.conserved_array(
        np.array((left, right)),
        gamma,
    )
    averages = np.empty((cells, 8))
    for cell, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        left_fraction = min(
            max((position - lower) / (upper - lower), 0.0),
            1.0,
        )
        averages[cell] = (
            left_fraction * left_state
            + (1.0 - left_fraction) * right_state
        )
    return mhd.primitive_from_conserved(averages, gamma)


def require_physical_wave_birth(
    *,
    case: str,
    cells: int,
    gamma: float,
    left: Array,
    right: Array,
    speed: float,
    position: float,
    expected_family: str,
) -> PhysicalWaveBirth:
    initial = discontinuity_cell_averages(
        cells,
        position,
        left,
        right,
        gamma,
    )
    birth = find_single_physical_wave_birth(
        initial,
        gamma,
        periodic=False,
    )
    if birth is None:
        raise RuntimeError(f"{case} did not produce one physical-wave birth")
    if (
        birth.path.family != expected_family
        or not primitive_components_agree(
            birth.wave.left,
            left,
            gamma=gamma,
            relative_tolerance=3.0e-10,
            pressure_from_conserved=True,
        )
        or not primitive_components_agree(
            birth.wave.right,
            right,
            gamma=gamma,
            relative_tolerance=3.0e-10,
            pressure_from_conserved=True,
        )
        or not math.isclose(
            birth.wave.speed,
            speed,
            rel_tol=3.0e-10,
            abs_tol=3.0e-10,
        )
        or not math.isclose(
            birth.position,
            position,
            rel_tol=0.0,
            abs_tol=3.0e-13,
        )
    ):
        raise RuntimeError(f"{case} physical-wave birth was misidentified")
    return birth


def validate_alfven_birth_family(gamma: float) -> int:
    angles = tuple(
        math.radians(value)
        for value in (-179.999, -90.0, -15.0, 15.0, 90.0, 179.999)
    ) + (-math.pi, math.pi)
    controls = 0
    rho = 1.0
    pressure = 1.0
    normal_velocity = 0.2
    radius = 0.5
    offset_velocity = np.array((0.1, -0.2))
    for normal_b in (-1.0, 1.0):
        for walen_sign in (-1, 1):
            left_transverse_b = np.array((radius, 0.0))
            left_transverse_v = (
                offset_velocity
                + walen_sign * left_transverse_b / math.sqrt(rho)
            )
            left = np.array(
                (
                    rho,
                    normal_velocity,
                    *left_transverse_v,
                    pressure,
                    normal_b,
                    *left_transverse_b,
                )
            )
            speed = (
                normal_velocity
                - walen_sign * normal_b / math.sqrt(rho)
            )
            expected_family = (
                "alfven_minus"
                if speed < normal_velocity
                else "alfven_plus"
            )
            for angle in angles:
                right_transverse_b = radius * np.array(
                    (math.cos(angle), math.sin(angle))
                )
                right_transverse_v = (
                    offset_velocity
                    + walen_sign
                    * right_transverse_b
                    / math.sqrt(rho)
                )
                right = np.array(
                    (
                        rho,
                        normal_velocity,
                        *right_transverse_v,
                        pressure,
                        normal_b,
                        *right_transverse_b,
                    )
                )
                birth = require_physical_wave_birth(
                    case="alfven_family",
                    cells=32,
                    gamma=gamma,
                    left=left,
                    right=right,
                    speed=speed,
                    position=0.333,
                    expected_family=expected_family,
                )
                recovered_angle = birth.path.transverse_rotation_angle
                if (
                    birth.path.walen_sign != walen_sign
                    or birth.path.transverse_field_radius is None
                    or not math.isclose(
                        birth.path.transverse_field_radius,
                        radius,
                        rel_tol=3.0e-10,
                        abs_tol=3.0e-10,
                    )
                    or recovered_angle is None
                    or abs(
                        math.atan2(
                            math.sin(recovered_angle - angle),
                            math.cos(recovered_angle - angle),
                        )
                    )
                    > 3.0e-10
                ):
                    raise RuntimeError(
                        "Alfven physical-path metadata was misidentified"
                    )
                controls += 1
    return controls


def physical_window_fits(
    primitive: Array,
    gamma: float,
    maximum_width: int = 12,
) -> list[WindowFit]:
    """Rank local multi-cell transitions by one-wave consistency."""
    state = mhd.conserved_array(primitive, gamma)
    fits: list[WindowFit] = []
    for start in range(2, len(state) - 3):
        maximum_stop = min(start + maximum_width, len(state) - 2)
        for stop in range(start + 1, maximum_stop + 1):
            left = primitive[start - 1]
            right = primitive[stop]
            wave = fit_physical_mhd_wave(
                left,
                left,
                right,
                gamma,
                allow_endpoint=True,
                jump_tolerance=math.inf,
            )
            if wave is None:
                continue
            left_state, right_state = mhd.conserved_array(
                np.array((left, right)),
                gamma,
            )
            state_jump = left_state - right_state
            jump_norm2 = float(np.dot(state_jump, state_jump))
            zone = state[start:stop]
            zone_mean = np.mean(zone, axis=0)
            left_fraction = float(
                np.dot(zone_mean - right_state, state_jump) / jump_norm2
            )
            fitted_mean = (
                left_fraction * left_state
                + (1.0 - left_fraction) * right_state
            )
            aggregate_residual = scaled_residual(
                zone_mean - fitted_mean,
                zone_mean,
                fitted_mean,
                state_jump,
            )
            profile_residual = 0.0
            for cell_state in zone:
                fraction = float(
                    np.dot(cell_state - right_state, state_jump) / jump_norm2
                )
                fitted_state = (
                    fraction * left_state
                    + (1.0 - fraction) * right_state
                )
                profile_residual = max(
                    profile_residual,
                    scaled_residual(
                        cell_state - fitted_state,
                        cell_state,
                        fitted_state,
                        state_jump,
                    ),
                )
            plateau_residual = max(
                scaled_residual(
                    state[start - 2] - left_state,
                    state[start - 2],
                    left_state,
                ),
                scaled_residual(
                    state[stop + 1] - right_state,
                    state[stop + 1],
                    right_state,
                ),
            )
            if -0.05 <= left_fraction <= 1.05:
                fits.append(
                    WindowFit(
                        start=start,
                        stop=stop,
                        speed=wave.speed,
                        jump_residual=wave.jump_residual,
                        aggregate_residual=aggregate_residual,
                        profile_residual=profile_residual,
                        plateau_residual=plateau_residual,
                    )
                )
    return sorted(
        fits,
        key=lambda fit: (
            max(
                fit.jump_residual,
                fit.aggregate_residual,
                fit.profile_residual,
                fit.plateau_residual,
            ),
            fit.stop - fit.start,
        ),
    )


def condense_physical_windows(
    primitive: Array,
    gamma: float,
    tolerance: float = 2.0e-3,
) -> tuple[Array, int]:
    """Sharpen isolated RH-consistent windows while preserving their totals."""
    original_state = mhd.conserved_array(primitive, gamma)
    condensed_state = original_state.copy()
    occupied: set[int] = set()
    accepted = 0
    for fit in physical_window_fits(primitive, gamma):
        score = max(
            fit.jump_residual,
            fit.aggregate_residual,
            fit.profile_residual,
            fit.plateau_residual,
        )
        cells = set(range(fit.start, fit.stop))
        if score > tolerance or occupied.intersection(cells):
            continue
        left_state = original_state[fit.start - 1]
        right_state = original_state[fit.stop]
        state_jump = left_state - right_state
        jump_norm2 = float(np.dot(state_jump, state_jump))
        zone_total = np.sum(
            original_state[fit.start : fit.stop],
            axis=0,
        )
        zone_cells = fit.stop - fit.start
        left_equivalent = float(
            np.dot(
                zone_total - zone_cells * right_state,
                state_jump,
            )
            / jump_norm2
        )
        if not (0.0 <= left_equivalent <= zone_cells):
            continue
        full_left = min(int(math.floor(left_equivalent)), zone_cells)
        fraction = left_equivalent - full_left
        replacement = np.empty((zone_cells, 8))
        replacement[:] = right_state
        replacement[:full_left] = left_state
        mixed_index: int | None = None
        if fraction > 256.0 * np.finfo(float).eps and full_left < zone_cells:
            mixed_index = full_left
            replacement[mixed_index] = (
                fraction * left_state + (1.0 - fraction) * right_state
            )
        closure = zone_total - np.sum(replacement, axis=0)
        closure_index = (
            mixed_index
            if mixed_index is not None
            else min(full_left, zone_cells - 1)
        )
        replacement[closure_index] += closure
        replacement_primitive = mhd.primitive_from_conserved(
            replacement,
            gamma,
        )
        if not mhd.primitive_is_admissible(replacement_primitive):
            continue
        normal_b_scale = max(
            float(np.max(np.abs(replacement_primitive[:, 5]))),
            1.0,
        )
        if (
            np.ptp(replacement_primitive[:, 5])
            > 3.0e-10 * normal_b_scale
        ):
            continue
        condensed_state[fit.start : fit.stop] = replacement
        occupied.update(cells)
        accepted += 1
    condensed = mhd.primitive_from_conserved(condensed_state, gamma)
    if not mhd.primitive_is_admissible(condensed):
        return primitive.copy(), 0
    conservation_error = float(
        np.max(np.abs(np.sum(condensed_state - original_state, axis=0)))
    )
    conservation_scale = max(
        float(np.max(np.abs(np.sum(original_state, axis=0)))),
        1.0,
    )
    if conservation_error > 2.0e-12 * conservation_scale:
        raise RuntimeError("window condensation violated conservation")
    return condensed, accepted


def oblique_fast_shock_states(
    gamma: float,
    shock_speed: float = 2.5,
) -> tuple[Array, Array, float]:
    """Construct one nontrivial oblique fast shock from the complete RH system."""
    upstream = np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 0.5, 0.0))
    upstream_state = mhd.conserved_array(upstream[None, :], gamma)[0]
    upstream_flux = mhd_flux(upstream, gamma)
    active = np.array((0, 1, 2, 3, 4, 6, 7))
    residual_scale = np.maximum(
        np.abs(upstream_flux - shock_speed * upstream_state),
        1.0,
    )

    def primitive_from_unknown(unknown: Array) -> Array:
        return np.array(
            (
                math.exp(unknown[0]),
                unknown[1],
                unknown[2],
                unknown[3],
                math.exp(unknown[4]),
                upstream[5],
                unknown[5],
                unknown[6],
            )
        )

    def jump_residual(unknown: Array) -> Array:
        downstream = primitive_from_unknown(unknown)
        downstream_state = mhd.conserved_array(
            downstream[None, :],
            gamma,
        )[0]
        residual = (
            mhd_flux(downstream, gamma)
            - upstream_flux
            - shock_speed * (downstream_state - upstream_state)
        ) / residual_scale
        return residual[active]

    sound = math.sqrt(gamma * upstream[4] / upstream[0])
    mach = shock_speed / sound
    density_ratio = (
        (gamma + 1.0) * mach * mach
        / ((gamma - 1.0) * mach * mach + 2.0)
    )
    pressure_ratio = (
        2.0 * gamma * mach * mach - (gamma - 1.0)
    ) / (gamma + 1.0)
    initial_guess = np.array(
        (
            math.log(density_ratio),
            shock_speed * (1.0 - 1.0 / density_ratio),
            -0.1,
            0.0,
            math.log(pressure_ratio),
            upstream[6] * density_ratio,
            0.0,
        )
    )
    solution = least_squares(
        jump_residual,
        initial_guess,
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=5000,
    )
    downstream = primitive_from_unknown(solution.x)
    if (
        not solution.success
        or float(np.max(np.abs(jump_residual(solution.x)))) > 2.0e-11
        or np.linalg.norm(downstream - upstream) < 1.0e-4
    ):
        raise RuntimeError("failed to construct the oblique MHD fast shock")
    fitted = fit_physical_mhd_wave(
        downstream,
        downstream,
        upstream,
        gamma,
        allow_endpoint=True,
    )
    if fitted is None or not math.isclose(
        fitted.speed,
        shock_speed,
        rel_tol=3.0e-11,
        abs_tol=3.0e-11,
    ):
        raise RuntimeError("oblique MHD shock fails the fitted-wave contract")
    return downstream, upstream, shock_speed


def result_from_exact(
    *,
    case: str,
    method: str,
    cells: int,
    cfl: float,
    completed: bool,
    evolved: Array,
    exact: Array,
    gamma: float,
) -> Result:
    errors = []
    overshoots = []
    tv_ratios = []
    for component in range(8):
        exact_component = exact[:, component]
        evolved_component = evolved[:, component]
        scale = max(
            float(np.ptp(exact_component)),
            float(np.max(np.abs(exact_component))),
            1.0,
        )
        errors.append(
            float(np.mean(np.abs(evolved_component - exact_component)) / scale)
        )
        overshoots.append(
            max(
                0.0,
                float(np.max(evolved_component) - np.max(exact_component)),
                float(np.min(exact_component) - np.min(evolved_component)),
            )
            / scale
        )
        exact_tv = mhd.total_variation(exact_component)
        evolved_tv = mhd.total_variation(evolved_component)
        tv_ratios.append(
            evolved_tv / exact_tv
            if exact_tv > 1.0e-12 * scale
            else evolved_tv / scale
        )
    endpoint_scale = max(
        float(np.linalg.norm(exact[-1] - exact[0])),
        1.0e-14,
    )
    endpoint_distance = np.minimum(
        np.linalg.norm(evolved - exact[0], axis=1),
        np.linalg.norm(evolved - exact[-1], axis=1),
    ) / endpoint_scale
    exact_integrals = np.mean(mhd.conserved_array(exact, gamma), axis=0)
    final_integrals = np.mean(mhd.conserved_array(evolved, gamma), axis=0)
    integral_error = np.abs(final_integrals - exact_integrals)
    return Result(
        case=case,
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        total_l1=sum(errors),
        transverse_l1=sum(errors[index] for index in (2, 3, 6, 7)),
        maximum_overshoot=max(overshoots),
        maximum_tv_ratio=max(tv_ratios),
        transition_cells=int(np.count_nonzero(endpoint_distance > 3.0e-8)),
        maximum_divergence_proxy=float(np.ptp(evolved[:, 5])),
        mass_error=float(integral_error[0]),
        momentum_error=float(np.max(integral_error[1:4])),
        energy_error=float(integral_error[4]),
    )


def discontinuity_result(
    *,
    case: str,
    method: str,
    cells: int,
    cfl: float,
    gamma: float,
    left: Array,
    right: Array,
    speed: float,
    initial_position: float,
    final_time: float,
) -> Result:
    initial = discontinuity_cell_averages(
        cells,
        initial_position,
        left,
        right,
        gamma,
    )
    if method == "physical_wave_space_time":
        evolved, completed = evolve_space_time(
            initial,
            final_time,
            cfl,
            gamma,
            periodic=False,
        )
    elif method == "persistent_physical_wave_space_time":
        evolved, completed = evolve_persistent_wave(
            initial,
            final_time,
            cfl,
            gamma,
            left,
            right,
            speed,
            initial_position,
        )
    elif method == "born_persistent_physical_wave_space_time":
        evolved, completed = evolve_born_persistent_wave(
            initial,
            final_time,
            cfl,
            gamma,
            periodic=False,
        )
    elif method == "mp5_hlld":
        evolved, completed = mhd.evolve(
            initial,
            final_time,
            cfl,
            gamma,
            mhd.characteristic_mp5,
            mhd_hlld_flux,
            periodic=False,
        )
    else:
        raise ValueError(f"unknown isolated-wave method: {method}")
    exact = discontinuity_cell_averages(
        cells,
        initial_position + speed * final_time,
        left,
        right,
        gamma,
    )
    return result_from_exact(
        case=case,
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        evolved=evolved,
        exact=exact,
        gamma=gamma,
    )


def print_results(results: list[Result]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(Result))
    print("\t".join(fields))
    for result in results:
        values = []
        for field in fields:
            value = getattr(result, field)
            if isinstance(value, float):
                values.append(f"{value:.12e}")
            else:
                values.append(str(value))
        print("\t".join(values))


def validate_persistent_results(results: list[Result]) -> None:
    for result in results:
        if result.method not in (
            "persistent_physical_wave_space_time",
            "born_persistent_physical_wave_space_time",
        ):
            continue
        if (
            not result.completed
            or result.total_l1 > 5.0e-10
            or result.maximum_overshoot > 5.0e-10
            or abs(result.maximum_tv_ratio - 1.0) > 5.0e-9
            or result.transition_cells != 1
            or result.maximum_divergence_proxy > 5.0e-13
            or max(
                result.mass_error,
                result.momentum_error,
                result.energy_error,
            )
            > 5.0e-11
        ):
            raise RuntimeError(
                "persistent physical-wave control failed: "
                f"{result.case}, N={result.cells}, CFL={result.cfl:g}"
            )


def shock_tube_comparison(
    *,
    case: str,
    initial: Callable[[int], Array],
    gamma: float,
    final_time: float,
    cells: int,
    cfl: float,
    reference_factor: int,
) -> list[mhd.Result]:
    reference, reference_completed = mhd.evolve(
        initial(cells * reference_factor),
        final_time,
        cfl,
        gamma,
        mhd.characteristic_mp5,
        mhd_hlld_flux,
        periodic=False,
    )
    if not reference_completed:
        raise RuntimeError(f"fine {case} reference failed")
    exact = mhd.coarsen_conserved(reference, cells, gamma)
    baseline, baseline_completed = mhd.evolve(
        initial(cells),
        final_time,
        cfl,
        gamma,
        mhd.characteristic_mp5,
        mhd_hlld_flux,
        periodic=False,
    )
    candidate, candidate_completed = evolve_space_time(
        initial(cells),
        final_time,
        cfl,
        gamma,
        periodic=False,
    )
    fan_memory = hlld_fan_cell_averages(
        initial(cells),
        final_time,
        gamma,
    )
    condensed, condensed_windows = condense_physical_windows(
        candidate,
        gamma,
    )
    return [
        mhd.result_from_exact(
            f"{case}_reference",
            "mp5_hlld",
            baseline,
            exact,
            baseline_completed,
            gamma,
        ),
        mhd.result_from_exact(
            f"{case}_reference",
            "physical_wave_space_time",
            candidate,
            exact,
            candidate_completed,
            gamma,
        ),
        mhd.result_from_exact(
            f"{case}_reference",
            "single_hlld_fan_memory",
            fan_memory,
            exact,
            mhd.primitive_is_admissible(fan_memory),
            gamma,
        ),
        mhd.result_from_exact(
            f"{case}_reference",
            f"physical_window_condensed_{condensed_windows}",
            condensed,
            exact,
            mhd.primitive_is_admissible(condensed),
            gamma,
        ),
    ]


def hlld_fan_states(
    initial: Array,
    gamma: float,
) -> tuple[tuple[Array, ...], Array]:
    """Recover the ordered conserved states and speeds of one HLLD fan."""
    left = initial[0].copy()
    right = initial[-1].copy()
    fan_states: list[Array] = []
    fan_speeds: list[float] = []
    flux = mhd_hlld_flux(
        left,
        right,
        gamma,
        fan_state_sink=fan_states,
        fan_speed_sink=fan_speeds,
    )
    if (
        not np.all(np.isfinite(flux))
        or len(fan_states) not in (2, 4)
        or len(fan_speeds) not in (3, 5)
        or np.any(np.diff(fan_speeds) < 0.0)
    ):
        raise RuntimeError("initial HLLD fan is unavailable or unordered")
    left_state = mhd_conserved(left, gamma)
    right_state = mhd_conserved(right, gamma)
    if len(fan_states) == 4:
        state_ls, state_rs, state_ld, state_rd = fan_states
        states = (
            left_state,
            state_ls,
            state_ld,
            state_rd,
            state_rs,
            right_state,
        )
    else:
        state_ls, state_rs = fan_states
        states = (left_state, state_ls, state_rs, right_state)
    return states, np.asarray(fan_speeds)


def hlld_fan_physical_residuals(
    initial: Array,
    gamma: float,
) -> Array:
    """Measure every HLLD intermediate jump using the physical MHD flux."""
    states, speeds = hlld_fan_states(initial, gamma)
    primitive = mhd.primitive_from_conserved(np.asarray(states), gamma)
    if not mhd.primitive_is_admissible(primitive):
        raise RuntimeError("recovered HLLD fan is not physically admissible")
    residuals = np.empty(len(speeds))
    for index, speed in enumerate(speeds):
        state_jump = states[index + 1] - states[index]
        flux_jump = (
            mhd_flux(primitive[index + 1], gamma)
            - mhd_flux(primitive[index], gamma)
        )
        residuals[index] = scaled_residual(
            flux_jump - speed * state_jump,
            flux_jump,
            speed * state_jump,
        )
    return residuals


def hlld_fan_cell_averages(
    initial: Array,
    time: float,
    gamma: float,
    discontinuity: float = 0.5,
) -> Array:
    """Average one retained HLLD fan without repeated FV projection."""
    states, fan_speeds = hlld_fan_states(initial, gamma)
    boundaries = discontinuity + fan_speeds * time
    cells = len(initial)
    edges = np.linspace(0.0, 1.0, cells + 1)
    averages = np.empty((cells, 8))
    for cell, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        split = np.concatenate(
            (
                [lower],
                boundaries[
                    (boundaries > lower) & (boundaries < upper)
                ],
                [upper],
            )
        )
        integral = np.zeros(8)
        for segment_lower, segment_upper in zip(
            split[:-1],
            split[1:],
            strict=True,
        ):
            midpoint = 0.5 * (segment_lower + segment_upper)
            region = int(np.searchsorted(boundaries, midpoint, side="right"))
            integral += (
                segment_upper - segment_lower
            ) * states[region]
        averages[cell] = integral / (upper - lower)
    return mhd.primitive_from_conserved(averages, gamma)


def print_mhd_results(results: list[mhd.Result]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(mhd.Result))
    print("shock_tube\t" + "\t".join(fields))
    for result in results:
        values = []
        for field in fields:
            value = getattr(result, field)
            if isinstance(value, float):
                values.append(f"{value:.12e}")
            else:
                values.append(str(value))
        print("shock_tube\t" + "\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32,))
    parser.add_argument("--cfl", type=float, nargs="+", default=(0.9,))
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--include-shock-tubes", action="store_true")
    parser.add_argument("--scan-physical-windows", action="store_true")
    parser.add_argument("--skip-smooth-control", action="store_true")
    parser.add_argument("--reference-factor", type=int, default=4)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "mp5_hlld",
            "physical_wave_space_time",
            "persistent_physical_wave_space_time",
            "born_persistent_physical_wave_space_time",
        ),
        default=(
            "mp5_hlld",
            "physical_wave_space_time",
            "persistent_physical_wave_space_time",
            "born_persistent_physical_wave_space_time",
        ),
    )
    args = parser.parse_args()

    contact_left = np.array(
        (2.0, 0.5, 0.1, -0.2, 1.0, 0.7, 0.3, -0.4)
    )
    contact_right = contact_left.copy()
    contact_right[0] = 1.0

    rotational_left = np.array(
        (1.0, 1.5, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0)
    )
    rotational_right = np.array(
        (1.0, 1.5, 0.0, 0.5, 1.0, 1.0, 0.0, 0.5)
    )
    rotational_fit = fit_physical_mhd_wave(
        rotational_left,
        rotational_left,
        rotational_right,
        args.gamma,
        allow_endpoint=True,
    )
    if rotational_fit is None:
        raise RuntimeError("rotational-discontinuity control is not physical")
    rotational_speed = rotational_fit.speed
    half_turn_left = np.array(
        (1.0, 1.5, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0)
    )
    half_turn_right = np.array(
        (1.0, 1.5, -0.5, 0.0, 1.0, 1.0, -0.5, 0.0)
    )
    half_turn_fit = fit_physical_mhd_wave(
        half_turn_left,
        half_turn_left,
        half_turn_right,
        args.gamma,
        allow_endpoint=True,
    )
    if half_turn_fit is None:
        raise RuntimeError(
            "half-turn rotational-discontinuity control is not physical"
        )
    half_turn_speed = half_turn_fit.speed
    half_turn_mixed_state = np.mean(
        mhd.conserved_array(
            np.array((half_turn_left, half_turn_right)),
            args.gamma,
        ),
        axis=0,
    )
    half_turn_mixed = mhd.primitive_from_conserved(
        half_turn_mixed_state[None, :],
        args.gamma,
    )[0]
    endpoint_bt = math.hypot(half_turn_left[6], half_turn_left[7])
    mixed_bt = math.hypot(half_turn_mixed[6], half_turn_mixed[7])
    if (
        mixed_bt > 1.0e-14
        or half_turn_mixed[4] <= half_turn_left[4]
    ):
        raise RuntimeError("half-turn chord-average corruption control failed")
    print(
        "half_turn_chord_average"
        f"\tendpoint_bt={endpoint_bt:.12e}"
        f"\tmixed_bt={mixed_bt:.12e}"
        f"\tendpoint_pressure={half_turn_left[4]:.12e}"
        f"\tmixed_pressure={half_turn_mixed[4]:.12e}"
    )
    fast_shock_left, fast_shock_right, fast_shock_speed = (
        oblique_fast_shock_states(args.gamma)
    )
    fast_minus_left = fast_shock_right.copy()
    fast_minus_right = fast_shock_left.copy()
    fast_minus_left[[1, 5]] *= -1.0
    fast_minus_right[[1, 5]] *= -1.0
    fast_minus_speed = -fast_shock_speed
    weak_mach = 1.0001
    weak_shock_speed = weak_mach * math.sqrt(args.gamma)
    weak_compression = (
        (args.gamma + 1.0) * weak_mach**2
        / ((args.gamma - 1.0) * weak_mach**2 + 2.0)
    )
    weak_pressure_ratio = (
        2.0 * args.gamma * weak_mach**2 - (args.gamma - 1.0)
    ) / (args.gamma + 1.0)
    weak_fast_right = np.array(
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    )
    weak_fast_left = np.array(
        (
            weak_compression,
            weak_shock_speed * (1.0 - 1.0 / weak_compression),
            0.0,
            0.0,
            weak_pressure_ratio,
            0.0,
            0.0,
            0.0,
        )
    )
    # This boost remains resolvable at the nominal detector tolerance; the
    # rejection controls below exercise nearby under-resolved states.
    boost_velocity = 2.0e2
    boosted_fast_left = fast_shock_left.copy()
    boosted_fast_right = fast_shock_right.copy()
    boosted_fast_left[1] += boost_velocity
    boosted_fast_right[1] += boost_velocity
    boosted_fast_speed = fast_shock_speed + boost_velocity

    birth_controls = (
        (
            "moving_mhd_contact",
            contact_left,
            contact_right,
            0.5,
            "contact",
        ),
        (
            "moving_oblique_fast_shock",
            fast_shock_left,
            fast_shock_right,
            fast_shock_speed,
            "fast_plus_shock",
        ),
        (
            "moving_rotational_discontinuity",
            rotational_left,
            rotational_right,
            rotational_speed,
            "alfven_minus",
        ),
        (
            "moving_half_turn_rotational_discontinuity",
            half_turn_left,
            half_turn_right,
            half_turn_speed,
            "alfven_minus",
        ),
        (
            "moving_oblique_fast_minus_shock",
            fast_minus_left,
            fast_minus_right,
            fast_minus_speed,
            "fast_minus_shock",
        ),
        (
            "weak_hydro_fast_shock",
            weak_fast_left,
            weak_fast_right,
            weak_shock_speed,
            "fast_plus_shock",
        ),
        (
            "galilean_boosted_fast_shock",
            boosted_fast_left,
            boosted_fast_right,
            boosted_fast_speed,
            "fast_plus_shock",
        ),
    )
    for birth_position in (0.217, 0.5):
        for case, left, right, speed, expected_family in birth_controls:
            birth = require_physical_wave_birth(
                case=case,
                cells=32,
                gamma=args.gamma,
                left=left,
                right=right,
                speed=speed,
                position=birth_position,
                expected_family=expected_family,
            )
            print(
                "physical_wave_birth"
                f"\tcase={case}"
                f"\tfamily={birth.path.family}"
                f"\tlocation={birth.location_kind}"
                f"\tcell={birth.cell}"
                f"\tposition={birth.position:.12e}"
                f"\tspeed={birth.wave.speed:.12e}"
            )

    face_evolution_results = []
    for case, left, right, speed, _ in (
        birth_controls[0],
        birth_controls[1],
        birth_controls[3],
        birth_controls[4],
    ):
        face_evolution_results.append(
            discontinuity_result(
                case=f"{case}_face_aligned",
                method="born_persistent_physical_wave_space_time",
                cells=32,
                cfl=0.9,
                gamma=args.gamma,
                left=left,
                right=right,
                speed=speed,
                initial_position=0.5,
                final_time=0.11 if "shock" in case else 0.6,
            )
        )
    validate_persistent_results(face_evolution_results)
    print(
        "physical_wave_face_evolution"
        f"\tcases={len(face_evolution_results)}"
        f"\tmaximum_l1={max(result.total_l1 for result in face_evolution_results):.12e}"
        "\tmaximum_overshoot="
        f"{max(result.maximum_overshoot for result in face_evolution_results):.12e}"
    )

    perturbed_rotation = rotational_right.copy()
    perturbed_rotation[4] *= 1.0001
    perturbed_initial = discontinuity_cell_averages(
        32,
        0.217,
        rotational_left,
        perturbed_rotation,
        args.gamma,
    )
    zero_normal_left = np.array(
        (1.0, 0.2, 0.6, -0.2, 1.0, 0.0, 0.5, 0.0)
    )
    zero_normal_right = np.array(
        (1.0, 0.2, 0.1, 0.3, 1.0, 0.0, 0.0, 0.5)
    )
    zero_normal_initial = discontinuity_cell_averages(
        32,
        0.217,
        zero_normal_left,
        zero_normal_right,
        args.gamma,
    )
    low_beta_left = np.array(
        (1.0, 1.5, 0.5, 0.0, 1.0, 1.0e6, 0.5, 0.0)
    )
    low_beta_right = np.array(
        (1.0, 1.5, -0.5, 0.0, 1.0, 1.0e6, -0.5, 0.0)
    )
    low_beta_initial = discontinuity_cell_averages(
        32,
        0.217,
        low_beta_left,
        low_beta_right,
        args.gamma,
    )
    low_beta_corruption = discontinuity_cell_averages(
        32,
        0.217,
        low_beta_left,
        low_beta_right,
        args.gamma,
    )
    low_beta_corruption[6, 4] += 0.05
    unresolved_low_pressure_left = np.array(
        (2.0, 0.5, 0.1, -0.2, 1.0e-12, 0.75, 0.3, -0.4)
    )
    unresolved_low_pressure_right = unresolved_low_pressure_left.copy()
    unresolved_low_pressure_right[0] = 1.0
    unresolved_low_pressure = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_low_pressure_left,
        unresolved_low_pressure_right,
        args.gamma,
    )
    resolved_low_pressure_left = np.array(
        (2.0, 0.0, 0.0, 0.0, 1.0e-12, 0.0, 0.0, 0.0)
    )
    resolved_low_pressure_right = resolved_low_pressure_left.copy()
    resolved_low_pressure_right[0] = 1.0
    resolved_low_pressure_birth = require_physical_wave_birth(
        case="resolved_low_pressure_contact",
        cells=32,
        gamma=args.gamma,
        left=resolved_low_pressure_left,
        right=resolved_low_pressure_right,
        speed=0.0,
        position=0.217,
        expected_family="contact",
    )
    low_pressure_corruption = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_low_pressure_left,
        unresolved_low_pressure_right,
        args.gamma,
    )
    low_pressure_corruption[6, 4] += 1.0e-12
    low_pressure_plateau = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_low_pressure_left,
        unresolved_low_pressure_right,
        args.gamma,
    )
    low_pressure_plateau[4, 4] *= 2.0
    nonadmissible_average = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_low_pressure_left,
        unresolved_low_pressure_right,
        args.gamma,
    )
    nonadmissible_average[6, 4] = -unresolved_low_pressure_left[4]
    hidden_invariant_left = np.array(
        (1.0, 1.5, 1.0e6, 0.0, 1.0, 1.0e6, 1.0e6, 0.0)
    )
    hidden_invariant_right = np.array(
        (1.0, 1.5, -1.0e6, 0.0, 1.0005, 1.0e6, -1.0e6, 0.0)
    )
    hidden_invariant = discontinuity_cell_averages(
        32,
        0.217,
        hidden_invariant_left,
        hidden_invariant_right,
        args.gamma,
    )
    scaled_non_rh_left = fast_shock_left.copy()
    scaled_non_rh_right = fast_shock_right.copy()
    for state in (scaled_non_rh_left, scaled_non_rh_right):
        state[0] *= 1.0e-12
        state[4] *= 1.0e-12
        state[5:8] *= 1.0e-6
    scaled_non_rh_left[4] *= 1.1
    scaled_non_rh = discontinuity_cell_averages(
        32,
        0.217,
        scaled_non_rh_left,
        scaled_non_rh_right,
        args.gamma,
    )
    boosted_non_rh_left = boosted_fast_left.copy()
    boosted_non_rh_left[4] *= 1.1
    boosted_non_rh = discontinuity_cell_averages(
        32,
        0.217,
        boosted_non_rh_left,
        boosted_fast_right,
        args.gamma,
    )
    unresolved_boost_velocity = 3.0e2
    unresolved_boost_left = fast_shock_left.copy()
    unresolved_boost_right = fast_shock_right.copy()
    unresolved_boost_left[1] += unresolved_boost_velocity
    unresolved_boost_right[1] += unresolved_boost_velocity
    unresolved_boost = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_boost_left,
        unresolved_boost_right,
        args.gamma,
    )
    unresolved_boost_corrupt_left = unresolved_boost_left.copy()
    unresolved_boost_corrupt_left[4] *= 1.1
    unresolved_boost_corrupt = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_boost_corrupt_left,
        unresolved_boost_right,
        args.gamma,
    )
    unresolved_contact_left = np.array(
        (2.0, 2.0e4, 0.1, -0.2, 1.0, 0.7, 0.3, -0.4)
    )
    unresolved_contact_right = unresolved_contact_left.copy()
    unresolved_contact_right[0] = 1.0
    unresolved_contact_clean = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_contact_left,
        unresolved_contact_right,
        args.gamma,
    )
    unresolved_contact_right[4] += 1.5e-7
    unresolved_contact_corrupt = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_contact_left,
        unresolved_contact_right,
        args.gamma,
    )
    unresolved_alfven_left = np.array(
        (1.0, 2.50015e4, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0)
    )
    unresolved_alfven_right = np.array(
        (1.0, 2.50015e4, 0.0, 0.5, 1.0, 1.0, 0.0, 0.5)
    )
    unresolved_alfven_clean = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_alfven_left,
        unresolved_alfven_right,
        args.gamma,
    )
    unresolved_alfven_right[4] += 1.5e-7
    unresolved_alfven_corrupt = discontinuity_cell_averages(
        32,
        0.217,
        unresolved_alfven_left,
        unresolved_alfven_right,
        args.gamma,
    )
    resolved_corrupt_contact_left = np.array(
        (2.0, 0.0, 0.0, 0.0, 1.0, 250.0, 0.0, 0.0)
    )
    resolved_corrupt_contact_right = resolved_corrupt_contact_left.copy()
    resolved_corrupt_contact_right[0] = 1.0
    resolved_corrupt_contact_right[4] = 1.000000003297
    resolved_corrupt_contact = discontinuity_cell_averages(
        32,
        0.217,
        resolved_corrupt_contact_left,
        resolved_corrupt_contact_right,
        args.gamma,
    )
    resolved_corrupt_alfven_left = np.array(
        (1.0, 0.0, 102.65, 0.0, 1.0, 205.3, 102.65, 0.0)
    )
    resolved_corrupt_alfven_right = np.array(
        (
            1.0,
            0.0,
            0.0,
            102.65,
            1.0000000032987,
            205.3,
            0.0,
            102.65,
        )
    )
    resolved_corrupt_alfven = discontinuity_cell_averages(
        32,
        0.217,
        resolved_corrupt_alfven_left,
        resolved_corrupt_alfven_right,
        args.gamma,
    )
    hidden_normal_b_left = np.array(
        (2.0, 0.0, 0.0, 0.0, 1.0e8, 0.0, 1.0e6, 0.0)
    )
    hidden_normal_b_right = np.array(
        (1.0, 0.0, 0.0, 0.0, 1.0e8, 5.0e-8, 1.0e6, 0.0)
    )
    hidden_normal_b = discontinuity_cell_averages(
        32,
        0.217,
        hidden_normal_b_left,
        hidden_normal_b_right,
        args.gamma,
    )
    hidden_normal_velocity_left = np.array(
        (2.0, 0.0, 1.0e6, 0.0, 1.0e8, 0.0, 0.0, 0.0)
    )
    hidden_normal_velocity_right = np.array(
        (1.0, 5.0e-8, 1.0e6, 0.0, 1.0e8, 0.0, 0.0, 0.0)
    )
    hidden_normal_velocity = discontinuity_cell_averages(
        32,
        0.217,
        hidden_normal_velocity_left,
        hidden_normal_velocity_right,
        args.gamma,
    )
    boundary_front = discontinuity_cell_averages(
        32,
        1.0 / 32.0 + 0.007,
        contact_left,
        contact_right,
        args.gamma,
    )
    _, smooth_birth_control = mhd.circular_alfven_initial(32)
    endpoint_states = mhd.conserved_array(
        np.array((contact_left, contact_right)),
        args.gamma,
    )
    double_state = np.repeat(endpoint_states[0][None, :], 32, axis=0)
    double_state[7] = 0.35 * endpoint_states[0] + 0.65 * endpoint_states[1]
    double_state[8:23] = endpoint_states[1]
    double_state[23] = 0.65 * endpoint_states[1] + 0.35 * endpoint_states[0]
    double_birth_control = mhd.primitive_from_conserved(
        double_state,
        args.gamma,
    )
    rejection_controls = (
        find_single_physical_wave_birth(
            perturbed_initial,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            smooth_birth_control,
            args.gamma,
            periodic=True,
        ),
        find_single_physical_wave_birth(
            double_birth_control,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            zero_normal_initial,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            low_beta_initial,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            low_beta_corruption,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            unresolved_low_pressure,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            low_pressure_corruption,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            low_pressure_plateau,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            nonadmissible_average,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            hidden_invariant,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            scaled_non_rh,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            boosted_non_rh,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            unresolved_boost,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            unresolved_boost_corrupt,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            unresolved_contact_clean,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            unresolved_contact_corrupt,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            unresolved_alfven_clean,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            unresolved_alfven_corrupt,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            resolved_corrupt_contact,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            resolved_corrupt_alfven,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            hidden_normal_b,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            hidden_normal_velocity,
            args.gamma,
            periodic=False,
        ),
        find_single_physical_wave_birth(
            boundary_front,
            args.gamma,
            periodic=False,
        ),
    )
    rejection_control_names = (
        "perturbed_rotation",
        "smooth_periodic_wave",
        "multiple_fronts",
        "zero_normal_field",
        "unresolved_low_beta",
        "corrupt_low_beta",
        "unresolved_low_pressure",
        "corrupt_low_pressure",
        "low_pressure_plateau",
        "nonadmissible_average",
        "hidden_invariant",
        "scaled_non_rh",
        "boosted_non_rh",
        "unresolved_boost",
        "corrupt_unresolved_boost",
        "unresolved_contact",
        "corrupt_unresolved_contact",
        "unresolved_alfven",
        "corrupt_unresolved_alfven",
        "resolved_corrupt_contact",
        "resolved_corrupt_alfven",
        "hidden_normal_field",
        "hidden_normal_velocity",
        "boundary_front",
    )
    failed_rejections = [
        name
        for name, control in zip(
            rejection_control_names,
            rejection_controls,
            strict=True,
        )
        if control is not None
    ]
    if failed_rejections:
        raise RuntimeError(
            "physical-wave birth rejection control failed: "
            + ", ".join(failed_rejections)
        )
    alfven_birth_controls = validate_alfven_birth_family(args.gamma)
    print(
        "physical_wave_birth_rejections"
        "\tnon_rh=passed\tsmooth=passed\tmultiple_candidates=passed"
        "\tzero_normal_alfven=passed\tlow_beta_unresolved=passed"
        "\tlow_beta_corruption=passed"
        "\tlow_pressure_unresolved=passed"
        "\tlow_pressure_corruption=passed\tplateau_conditioning=passed"
        "\tnonadmissible_average=passed"
        "\thidden_invariant=passed\tscaling_symmetry=passed"
        "\tgalilean_symmetry=passed"
        "\tunresolved_boost_clean=passed"
        "\tunresolved_boost_corrupt=passed"
        "\tunresolved_contact_pair=passed"
        "\tunresolved_alfven_pair=passed"
        "\tresolved_pressure_jump_contact=passed"
        "\tresolved_pressure_jump_alfven=passed"
        "\tcomponentwise_normal_b=passed"
        "\tcomponentwise_normal_velocity=passed"
        "\tpressure_uncertainty_ceiling="
        f"{MAX_RELATIVE_PRESSURE_RECOVERY_UNCERTAINTY:.3e}"
        f"\tboost_uncertainty_ceiling={MAX_BOOST_CONDITIONING_UNCERTAINTY:.3e}"
        "\tboundary_plateau=passed\tperiodic=unsupported"
        f"\talfven_path_controls={alfven_birth_controls}"
    )
    print(
        "physical_wave_resolved_acceptance"
        f"\tlow_pressure={resolved_low_pressure_birth.path.family}"
        f"\tboosted_fast_velocity={boost_velocity:.1f}"
    )

    results: list[Result] = []
    for cells in args.cells:
        for cfl in args.cfl:
            for method in args.methods:
                results.append(
                    discontinuity_result(
                        case="moving_mhd_contact",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=args.gamma,
                        left=contact_left,
                        right=contact_right,
                        speed=0.5,
                        initial_position=0.233,
                        final_time=0.6,
                    )
                )
                results.append(
                    discontinuity_result(
                        case="moving_oblique_fast_shock",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=args.gamma,
                        left=fast_shock_left,
                        right=fast_shock_right,
                        speed=fast_shock_speed,
                        initial_position=0.217,
                        final_time=0.2,
                    )
                )
                results.append(
                    discontinuity_result(
                        case="moving_rotational_discontinuity",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=args.gamma,
                        left=rotational_left,
                        right=rotational_right,
                        speed=rotational_speed,
                        initial_position=0.217,
                        final_time=0.6,
                    )
                )
                results.append(
                    discontinuity_result(
                        case="moving_oblique_fast_minus_shock",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=args.gamma,
                        left=fast_minus_left,
                        right=fast_minus_right,
                        speed=fast_minus_speed,
                        initial_position=0.783,
                        final_time=0.2,
                    )
                )
                results.append(
                    discontinuity_result(
                        case="moving_half_turn_rotational_discontinuity",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=args.gamma,
                        left=half_turn_left,
                        right=half_turn_right,
                        speed=half_turn_speed,
                        initial_position=0.217,
                        final_time=0.6,
                    )
                )
            if args.skip_smooth_control:
                continue
            _, smooth_initial = mhd.circular_alfven_initial(cells)
            smooth_baseline, baseline_completed = mhd.evolve(
                smooth_initial,
                final_time=1.0,
                cfl=cfl,
                gamma=5.0 / 3.0,
                reconstruct=mhd.characteristic_mp5,
                riemann_flux=mhd_hlld_flux,
                periodic=True,
            )
            smooth_candidate, candidate_completed = evolve_space_time(
                smooth_initial,
                final_time=1.0,
                cfl=cfl,
                gamma=5.0 / 3.0,
                periodic=True,
            )
            smooth_born, born_completed = evolve_born_persistent_wave(
                smooth_initial,
                final_time=1.0,
                cfl=cfl,
                gamma=5.0 / 3.0,
                periodic=True,
            )
            smooth_difference = float(
                max(
                    np.max(np.abs(smooth_candidate - smooth_baseline)),
                    np.max(np.abs(smooth_born - smooth_baseline)),
                )
            )
            if (
                not baseline_completed
                or not candidate_completed
                or not born_completed
                or smooth_difference > 3.0e-12
            ):
                raise RuntimeError(
                    "physical-wave path changed the smooth MHD control: "
                    f"{smooth_difference:.12e}"
                )
            print(
                "smooth_circular_alfven"
                f"\tcells={cells}"
                f"\tcfl={cfl:g}"
                f"\tmaximum_difference={smooth_difference:.12e}"
            )
    validate_persistent_results(results)
    print_results(results)
    if args.include_shock_tubes:
        shock_tube_results: list[mhd.Result] = []
        for cells in args.cells:
            for cfl in args.cfl:
                shock_tube_results.extend(
                    shock_tube_comparison(
                        case="brio_wu",
                        initial=mhd.brio_wu_initial,
                        gamma=2.0,
                        final_time=0.1,
                        cells=cells,
                        cfl=cfl,
                        reference_factor=args.reference_factor,
                    )
                )
                shock_tube_results.extend(
                    shock_tube_comparison(
                        case="dai_woodward",
                        initial=subcell_mhd.dai_woodward_initial,
                        gamma=5.0 / 3.0,
                        final_time=0.08,
                        cells=cells,
                        cfl=cfl,
                        reference_factor=args.reference_factor,
                    )
                )
        print_mhd_results(shock_tube_results)
        for case, initial, gamma in (
            ("brio_wu", mhd.brio_wu_initial, 2.0),
            (
                "dai_woodward",
                subcell_mhd.dai_woodward_initial,
                5.0 / 3.0,
            ),
        ):
            residuals = hlld_fan_physical_residuals(
                initial(args.cells[0]),
                gamma,
            )
            print(
                "hlld_physical_jump_audit"
                f"\tcase={case}"
                f"\tminimum={np.min(residuals):.12e}"
                f"\tmaximum={np.max(residuals):.12e}"
                "\tresiduals="
                + ",".join(f"{residual:.12e}" for residual in residuals)
            )
    if args.scan_physical_windows:
        for case, initial, gamma, final_time in (
            ("brio_wu", mhd.brio_wu_initial, 2.0, 0.1),
            (
                "dai_woodward",
                subcell_mhd.dai_woodward_initial,
                5.0 / 3.0,
                0.08,
            ),
        ):
            for cells in args.cells:
                evolved, completed = mhd.evolve(
                    initial(cells),
                    final_time,
                    args.cfl[0],
                    gamma,
                    mhd.characteristic_mp5,
                    mhd_hlld_flux,
                    periodic=False,
                )
                if not completed:
                    raise RuntimeError(f"{case} window-scan evolution failed")
                fits = physical_window_fits(evolved, gamma)
                print(f"physical_window_scan\t{case}\tcells={cells}")
                for fit in fits[:8]:
                    print(
                        "physical_window"
                        f"\tstart={fit.start}"
                        f"\tstop={fit.stop}"
                        f"\tspeed={fit.speed:.8e}"
                        f"\trh={fit.jump_residual:.8e}"
                        f"\taggregate={fit.aggregate_residual:.8e}"
                        f"\tprofile={fit.profile_residual:.8e}"
                        f"\tplateau={fit.plateau_residual:.8e}"
                    )


if __name__ == "__main__":
    main()
