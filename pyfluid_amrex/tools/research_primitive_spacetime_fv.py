#!/usr/bin/env python3
"""Falsify a coherent primitive-variable space-time FV candidate in HD.

This research harness is intentionally separate from the AMReX method
catalog.  It combines one cell-local CWENO-Z polynomial, a primitive-variable
space-time predictor, and one time-integrated HLLC flux per shared face.
A posteriori corrections replace both faces of a troubled cell with the next
complete operator in the hierarchy

    CWENO-Z predictor -> characteristic MP5 predictor
                      -> PWL-MC predictor -> PWC Godunov.

Every accepted update is therefore a single conservative flux difference;
there are no cell-local Runge--Kutta splices.  The first promotion gate is
deliberately only Newtonian HD: formal smooth order, exact Euler shock tubes,
admissibility, and CFL 0.9.  MHD and relativistic adapters are out of scope
until this operator wins that gate.

Both an explicit collocation precursor and a weak local space-time DG/Picard
predictor are available.  The ADER-Prim path first reconstructs conserved cell
averages to point values, converts those center values to primitives, and then
reconstructs the primitive point samples.  This remains a falsification
harness, not a production implementation or a novelty claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research_modal_reconstruction import (  # noqa: E402
    mc_slope,
    minmod,
    minmod4,
    mp5_left,
    ppm4_left,
)
import research_physical_wave_subcell as physical_subcell  # noqa: E402
from research_wave_reconstruction import (  # noqa: E402
    SHOCK_CASES,
    Result as ShockResult,
    ShockCase,
    characteristic_mp5,
    characteristic_basis as hydro_characteristic_basis,
    comparison_scale,
    component_reconstruction,
    conserved_from_primitive,
    evolve as evolve_ssprk3,
    from_characteristics,
    hllc_flux,
    primitive_from_conserved,
    primitive_is_admissible,
    positive_trace,
    shock_result,
    stencil_value,
    to_characteristics,
    total_variation,
)


Array = np.ndarray


EXACT_CASES = (
    *SHOCK_CASES,
    ShockCase(
        "contact",
        (1.0, 1.0, 1.0),
        (0.125, 1.0, 1.0),
        0.1,
    ),
)


@dataclasses.dataclass(frozen=True)
class PredictorParameters:
    """Dimensionless controls for the offline operator."""

    central_linear_weight: float = 0.85
    wenoz_exponent: int = 2
    dmp_relative_tolerance: float = 0.01
    dmp_absolute_tolerance: float = 1.0e-10
    smooth_curvature_ratio: float = 0.05
    maximum_correction_iterations: int = 8
    characteristic_reconstruction: bool = True
    anchor_mp5_traces: bool = True
    contact_anchor_strength: float = 1.0
    acoustic_anchor_strength: float = 1.0
    compressive_acoustic_floor: float = 0.4
    acoustic_shock_scale: float = 0.2
    mp5_parabolic_primary: bool = False
    predictor_substeps: int = 1
    predictor_kind: str = "collocation"
    picard_iterations: int = 8
    picard_tolerance: float = math.inf
    double_reconstruction: bool = True
    physical_fronts: bool = False
    sensor_relative_activity_floor: float = 0.0


@dataclasses.dataclass
class EvolutionStats:
    steps: int = 0
    rejected_steps: int = 0
    high_cells: int = 0
    mp5_cells: int = 0
    pwl_cells: int = 0
    pwc_cells: int = 0
    physical_front_faces: int = 0
    minimum_order: int = 3


@dataclasses.dataclass(frozen=True)
class SmoothResult:
    method: str
    cells: int
    cfl: float
    completed: bool
    density_l1: float
    observed_order: float
    maximum_overshoot: float
    tv_ratio: float
    mass_error: float
    steps: int
    mp5_cell_steps: int
    pwl_cell_steps: int
    pwc_cell_steps: int


@dataclasses.dataclass(frozen=True)
class SymmetryResult:
    case: str
    method: str
    cells: int
    completed: bool
    maximum_normalized_l1: float
    maximum_absolute_error: float


@dataclasses.dataclass(frozen=True)
class SmoothEulerResult:
    method: str
    cells: int
    cfl: float
    completed: bool
    maximum_l1: float
    observed_order: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    mass_error: float
    momentum_error: float
    energy_error: float
    steps: int
    mp5_cell_steps: int
    pwl_cell_steps: int
    pwc_cell_steps: int


@dataclasses.dataclass(frozen=True)
class StepResult:
    state: Array | None
    maximum_speed: float
    cell_orders: Array
    physical_front_faces: int = 0


@dataclasses.dataclass(frozen=True)
class LocalPrediction:
    boundary_stages: tuple[Array, ...]
    time_weights: Array
    valid_cells: Array
    maximum_speed: float


def candidate_name(parameters: PredictorParameters) -> str:
    if parameters.mp5_parabolic_primary:
        return f"mp5_parabolic_{parameters.predictor_kind}"
    basis = (
        "characteristic"
        if parameters.characteristic_reconstruction
        else "component"
    )
    anchor = ""
    if parameters.anchor_mp5_traces:
        contact = f"{parameters.contact_anchor_strength:g}".replace(".", "p")
        acoustic = f"{parameters.acoustic_anchor_strength:g}".replace(".", "p")
        floor = f"{parameters.compressive_acoustic_floor:g}".replace(
            ".", "p"
        )
        shock_scale = f"{parameters.acoustic_shock_scale:g}".replace(
            ".", "p"
        )
        anchor = f"_mp5c{contact}a{acoustic}f{floor}s{shock_scale}"
    representation = (
        "_aderprim" if parameters.double_reconstruction else "_proxyprim"
    )
    name = (
        f"cweno_z_{basis}{anchor}{representation}_"
        f"{parameters.predictor_kind}"
    )
    if parameters.physical_fronts:
        name += "_physical_fronts"
    return name


def _cell_average_matrix(offsets: Array, degree: int) -> Array:
    matrix = np.empty((len(offsets), degree + 1))
    for row, offset in enumerate(offsets):
        lower = float(offset) - 0.5
        upper = float(offset) + 0.5
        for power in range(degree + 1):
            matrix[row, power] = (
                upper ** (power + 1) - lower ** (power + 1)
            ) / (power + 1)
    return matrix


def _smoothness_matrix(degree: int) -> Array:
    matrix = np.zeros((degree + 1, degree + 1))
    for derivative in range(1, degree + 1):
        for first in range(derivative, degree + 1):
            first_factor = math.factorial(first) / math.factorial(
                first - derivative
            )
            for second in range(derivative, degree + 1):
                power = first + second - 2 * derivative
                if power % 2:
                    continue
                integral = 2.0 * 0.5 ** (power + 1) / (power + 1)
                second_factor = math.factorial(second) / math.factorial(
                    second - derivative
                )
                matrix[first, second] += (
                    first_factor * second_factor * integral
                )
    return matrix


def _differentiation_matrix(nodes: Array) -> Array:
    vandermonde = np.vander(nodes, len(nodes), increasing=True)
    derivative = np.zeros_like(vandermonde)
    for power in range(1, len(nodes)):
        derivative[:, power] = power * nodes ** (power - 1)
    return derivative @ np.linalg.inv(vandermonde)


QUARTIC_OFFSETS = np.arange(-2, 3, dtype=float)
QUADRATIC_OFFSETS = (
    np.arange(-2, 1, dtype=float),
    np.arange(-1, 2, dtype=float),
    np.arange(0, 3, dtype=float),
)
QUARTIC_INVERSE = np.linalg.inv(_cell_average_matrix(QUARTIC_OFFSETS, 4))
QUADRATIC_INVERSES = tuple(
    np.linalg.inv(_cell_average_matrix(offsets, 2))
    for offsets in QUADRATIC_OFFSETS
)
POINT_QUARTIC_INVERSE = np.linalg.inv(
    np.vander(QUARTIC_OFFSETS, 5, increasing=True)
)
POINT_QUADRATIC_INVERSES = tuple(
    np.linalg.inv(np.vander(offsets, 3, increasing=True))
    for offsets in QUADRATIC_OFFSETS
)
POINT_RIGHT_FACE_WEIGHTS = np.asarray(
    [0.5**power for power in range(5)]
) @ POINT_QUARTIC_INVERSE
SMOOTHNESS_MATRIX = _smoothness_matrix(4)
SPACE_NODES = 0.5 * np.array(
    (-1.0, -math.sqrt(3.0 / 7.0), 0.0, math.sqrt(3.0 / 7.0), 1.0)
)
SPACE_EVALUATION = np.vander(SPACE_NODES, 5, increasing=True)
SPACE_DERIVATIVE = _differentiation_matrix(SPACE_NODES)
RK4_TIME_WEIGHTS = np.array(
    (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
)
_gauss_nodes, GAUSS_WEIGHTS_01 = np.polynomial.legendre.leggauss(5)
GAUSS_SPACE_NODES = 0.5 * _gauss_nodes
GAUSS_TIME_NODES = 0.5 * (_gauss_nodes + 1.0)
GAUSS_WEIGHTS_01 = 0.5 * GAUSS_WEIGHTS_01
GAUSS_SPACE_EVALUATION = np.vander(
    GAUSS_SPACE_NODES, 5, increasing=True
)
GAUSS_SPACE_DERIVATIVE = _differentiation_matrix(GAUSS_SPACE_NODES)
GAUSS_TIME_DERIVATIVE = _differentiation_matrix(GAUSS_TIME_NODES)


def _lagrange_values(nodes: Array, point: float) -> Array:
    return np.asarray([point**power for power in range(len(nodes))]) @ (
        np.linalg.inv(np.vander(nodes, len(nodes), increasing=True))
    )


GAUSS_SPACE_LEFT = _lagrange_values(GAUSS_SPACE_NODES, -0.5)
GAUSS_SPACE_RIGHT = _lagrange_values(GAUSS_SPACE_NODES, 0.5)
GAUSS_TIME_INITIAL = _lagrange_values(GAUSS_TIME_NODES, 0.0)
GAUSS_TIME_FINAL = _lagrange_values(GAUSS_TIME_NODES, 1.0)
GAUSS_TIME_WEAK_MATRIX = (
    np.outer(GAUSS_TIME_FINAL, GAUSS_TIME_FINAL)
    - GAUSS_TIME_DERIVATIVE.T @ np.diag(GAUSS_WEIGHTS_01)
)
GAUSS_TIME_WEAK_INVERSE = np.linalg.inv(GAUSS_TIME_WEAK_MATRIX)


def _cell_index(index: int, cells: int, periodic: bool) -> int:
    if periodic:
        return index % cells
    return min(cells - 1, max(0, index))


def _stencil(
    values: Array,
    center: int,
    offsets: range | tuple[int, ...] | Array,
    periodic: bool,
) -> Array:
    return np.asarray(
        [
            values[_cell_index(center + int(offset), len(values), periodic)]
            for offset in offsets
        ]
    )


def _polynomial_average(coefficients: Array) -> Array:
    return (
        coefficients[0]
        + coefficients[2] / 12.0
        + coefficients[4] / 80.0
    )


def _candidate_smoothness(coefficients: Array) -> Array:
    return np.einsum(
        "pc,pq,qc->c",
        coefficients,
        SMOOTHNESS_MATRIX,
        coefficients,
    )


def cweno_z_coefficients(
    averages: Array,
    periodic: bool,
    parameters: PredictorParameters,
    *,
    point_samples: bool = False,
) -> Array:
    """Return one degree-four CWENO-Z polynomial per cell.

    Nonlinear weights are shared by all primitive components.  Component
    indicators are normalized by their domain variation before summation, so
    changing units cannot silently give one field ownership of the stencil.
    """

    cells, components = averages.shape
    central_weight = parameters.central_linear_weight
    if not 0.0 < central_weight < 1.0:
        raise ValueError("central CWENO linear weight must lie in (0, 1)")
    side_weight = (1.0 - central_weight) / 3.0
    linear_weights = np.array(
        (central_weight, side_weight, side_weight, side_weight)
    )
    variation = np.ptp(averages, axis=0)
    amplitude = np.max(np.abs(averages), axis=0)
    constant_components = variation <= (
        256.0 * np.finfo(float).eps * np.maximum(amplitude, 1.0)
    )
    dimensionless_activity = variation / np.maximum(amplitude, 1.0)
    activity_scale = float(np.max(dimensionless_activity))
    sensor_inactive = constant_components | (
        dimensionless_activity
        <= parameters.sensor_relative_activity_floor * activity_scale
    )
    scales = np.maximum(
        variation,
        64.0 * np.finfo(float).eps * np.maximum(amplitude, 1.0),
    )
    result = np.empty((cells, 5, components))
    epsilon = 128.0 * np.finfo(float).eps

    for cell in range(cells):
        five = _stencil(averages, cell, QUARTIC_OFFSETS, periodic)
        optimal = (
            POINT_QUARTIC_INVERSE if point_samples else QUARTIC_INVERSE
        ) @ five
        lower = []
        quadratic_inverses = (
            POINT_QUADRATIC_INVERSES
            if point_samples
            else QUADRATIC_INVERSES
        )
        for offsets, inverse in zip(
            QUADRATIC_OFFSETS, quadratic_inverses, strict=True
        ):
            coefficients = np.zeros((5, components))
            coefficients[:3] = inverse @ _stencil(
                averages, cell, offsets, periodic
            )
            lower.append(coefficients)
        candidates = np.empty((4, 5, components))
        candidates[1:] = lower
        candidates[0] = (
            optimal
            - side_weight * (lower[0] + lower[1] + lower[2])
        ) / central_weight
        component_indicators = np.asarray(
            [_candidate_smoothness(candidate) for candidate in candidates]
        )
        normalized = component_indicators / (scales * scales)
        # C2P roundoff in an analytically constant field must not own the
        # shared nonlinear stencil.  Merely zeroing that field's final
        # polynomial is too late: its normalized indicator can otherwise
        # perturb every genuinely varying component through the common
        # weights.
        normalized[:, sensor_inactive] = 0.0
        indicators = np.sum(normalized, axis=1)
        tau = float(np.sum(np.abs(normalized[1] - normalized[3])))
        ratios = tau / (epsilon + indicators)
        alpha = linear_weights * (
            1.0 + ratios ** parameters.wenoz_exponent
        )
        weights = alpha / np.sum(alpha)
        coefficients = np.einsum("s,spc->pc", weights, candidates)
        if point_samples:
            coefficients[0] = averages[cell]
        else:
            coefficients[0] += averages[cell] - _polynomial_average(
                coefficients
            )
        result[cell] = coefficients
    if np.any(constant_components):
        result[:, :, constant_components] = 0.0
        result[:, 0, constant_components] = averages[:, constant_components]
    return result


def characteristic_cweno_z_coefficients(
    averages: Array,
    gamma: float,
    periodic: bool,
    parameters: PredictorParameters,
    *,
    point_samples: bool = False,
) -> Array:
    """Build cell-local CWENO-Z polynomials wave by wave.

    A frozen primitive Euler basis is chosen in each target cell.  Each wave
    receives its own nonlinear weights, so an acoustic shock does not
    automatically collapse the entropy/contact polynomial.  The resulting
    wave polynomial is mapped back to primitive variables before the common
    positivity and space-time predictor stages.
    """

    cells, components = averages.shape
    if components != 3:
        raise ValueError("the Euler characteristic adapter expects 3 fields")
    central_weight = parameters.central_linear_weight
    side_weight = (1.0 - central_weight) / 3.0
    linear_weights = np.array(
        (central_weight, side_weight, side_weight, side_weight)
    )
    result = np.empty((cells, 5, components))
    epsilon = 128.0 * np.finfo(float).eps
    for cell in range(cells):
        five_primitive = _stencil(
            averages, cell, QUARTIC_OFFSETS, periodic
        )
        rho_ref = averages[cell, 0]
        sound = math.sqrt(gamma * averages[cell, 2] / rho_ref)
        five = to_characteristics(five_primitive, rho_ref, sound)
        optimal = (
            POINT_QUARTIC_INVERSE if point_samples else QUARTIC_INVERSE
        ) @ five
        lower = []
        quadratic_inverses = (
            POINT_QUADRATIC_INVERSES
            if point_samples
            else QUADRATIC_INVERSES
        )
        for offsets, inverse in zip(
            QUADRATIC_OFFSETS, quadratic_inverses, strict=True
        ):
            primitive_stencil = _stencil(
                averages, cell, offsets, periodic
            )
            waves = to_characteristics(
                primitive_stencil, rho_ref, sound
            )
            coefficients = np.zeros((5, components))
            coefficients[:3] = inverse @ waves
            lower.append(coefficients)
        candidates = np.empty((4, 5, components))
        candidates[1:] = lower
        candidates[0] = (
            optimal
            - side_weight * (lower[0] + lower[1] + lower[2])
        ) / central_weight
        indicators = np.asarray(
            [_candidate_smoothness(candidate) for candidate in candidates]
        )
        variation = np.ptp(five, axis=0)
        amplitude = np.max(np.abs(five), axis=0)
        constant = variation <= (
            256.0 * np.finfo(float).eps * np.maximum(amplitude, 1.0)
        )
        scales = np.maximum(
            variation,
            64.0
            * np.finfo(float).eps
            * np.maximum(amplitude, 1.0),
        )
        normalized = indicators / (scales * scales)
        normalized[:, constant] = 0.0
        tau = np.abs(normalized[1] - normalized[3])
        alpha = linear_weights[:, np.newaxis] * (
            1.0
            + (
                tau[np.newaxis, :] / (epsilon + normalized)
            )
            ** parameters.wenoz_exponent
        )
        weights = alpha / np.sum(alpha, axis=0, keepdims=True)
        wave_coefficients = np.einsum(
            "sc,spc->pc", weights, candidates
        )
        if np.any(constant):
            wave_coefficients[:, constant] = 0.0
            wave_coefficients[0, constant] = five[2, constant]
        primitive_coefficients = np.empty_like(wave_coefficients)
        for power, wave_coefficients_at_power in enumerate(
            wave_coefficients
        ):
            primitive_coefficients[power] = from_characteristics(
                wave_coefficients_at_power, rho_ref, sound
            )
        if point_samples:
            primitive_coefficients[0] = averages[cell]
        else:
            primitive_coefficients[0] += averages[
                cell
            ] - _polynomial_average(primitive_coefficients)
        result[cell] = primitive_coefficients
    return result


def point_mp5_left(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
    vpp: float,
    _vppp: float,
) -> float:
    """MP5 bounds applied to the fifth-order point interpolant."""

    values = np.asarray((vmm, vm, v0, vp, vpp))
    unlimited = float(POINT_RIGHT_FACE_WEIGHTS @ values)
    alpha = 4.0
    scale = max(float(np.max(np.abs(values))), 1.0)
    monotone = v0 + minmod(vp - v0, alpha * (v0 - vm))
    if (unlimited - v0) * (unlimited - monotone) <= 1.0e-12 * (
        scale * scale
    ):
        return unlimited
    dm = vmm - 2.0 * vm + v0
    d0 = vm - 2.0 * v0 + vp
    dp = v0 - 2.0 * vp + vpp
    d_m4_left = minmod4(4.0 * dm - d0, 4.0 * d0 - dm, dm, d0)
    d_m4_right = minmod4(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp)
    upper_limit = v0 + alpha * (v0 - vm)
    average = 0.5 * (v0 + vp)
    median = average - 0.5 * d_m4_right
    large_curvature = v0 + 0.5 * (v0 - vm) + (4.0 / 3.0) * d_m4_left
    lower_bound = max(
        min(v0, vp, median),
        min(v0, upper_limit, large_curvature),
    )
    upper_bound = min(
        max(v0, vp, median),
        max(v0, upper_limit, large_curvature),
    )
    return unlimited + minmod(
        lower_bound - unlimited, upper_bound - unlimited
    )


def characteristic_point_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
    periodic: bool,
) -> tuple[Array, Array]:
    stencil = np.asarray(
        [
            stencil_value(primitive, left_cell + offset, periodic)
            for offset in range(-2, 4)
        ]
    )
    rho_ref, sound = hydro_characteristic_basis(
        stencil[2], stencil[3], gamma
    )
    waves = to_characteristics(stencil, rho_ref, sound)
    left_waves = np.asarray(
        [point_mp5_left(*waves[:, wave]) for wave in range(3)]
    )
    right_waves = np.asarray(
        [point_mp5_left(*waves[::-1, wave]) for wave in range(3)]
    )
    left = from_characteristics(left_waves, rho_ref, sound)
    right = from_characteristics(right_waves, rho_ref, sound)
    return positive_trace(left, stencil[2]), positive_trace(right, stencil[3])


def anchor_characteristic_mp5_traces(
    coefficients: Array,
    averages: Array,
    gamma: float,
    periodic: bool,
    *,
    point_samples: bool,
    contact_strength: float,
    acoustic_strength: float,
    compressive_acoustic_floor: float,
    acoustic_shock_scale: float,
) -> Array:
    """Match both cell traces to MP5 without changing the cell average.

    For trace corrections ``delta_left`` and ``delta_right``, the quadratic
    basis functions ``-1/4 - xi + 3 xi^2`` and
    ``-1/4 + xi + 3 xi^2`` have zero cell average and Kronecker values at the
    two faces.  Smooth MP5 and CWENO traces differ only at high order, while
    near a discontinuity this gives the sharper proven MP5 limiter ownership
    of the actual Riemann data.
    """

    result = coefficients.copy()
    characteristic_mp5.periodic = periodic

    def scaled_delta(
        delta: Array,
        first: Array,
        second: Array,
    ) -> Array:
        rho_ref, sound = hydro_characteristic_basis(first, second, gamma)
        waves = to_characteristics(
            delta[np.newaxis, :], rho_ref, sound
        )[0]
        first_sound = math.sqrt(gamma * first[2] / first[0])
        second_sound = math.sqrt(gamma * second[2] / second[0])
        speed_scale = max(first_sound, second_sound, 1.0e-14)
        compression_minus = max(
            0.0,
            (first[1] - first_sound) - (second[1] - second_sound),
        ) / speed_scale
        compression_plus = max(
            0.0,
            (first[1] + first_sound) - (second[1] + second_sound),
        ) / speed_scale

        def acoustic_strength_for(compression: float) -> float:
            if acoustic_shock_scale <= 0.0:
                return acoustic_strength
            return compressive_acoustic_floor + (
                acoustic_strength - compressive_acoustic_floor
            ) / (1.0 + (compression / acoustic_shock_scale) ** 4)

        waves *= np.asarray(
            (
                contact_strength,
                acoustic_strength_for(compression_minus),
                acoustic_strength_for(compression_plus),
            )
        )
        return from_characteristics(waves, rho_ref, sound)

    for cell in range(len(averages)):
        if point_samples:
            _left_neighbor, left_trace = characteristic_point_mp5(
                averages, cell - 1, gamma, periodic
            )
            right_trace, _right_neighbor = characteristic_point_mp5(
                averages, cell, gamma, periodic
            )
        else:
            _left_neighbor, left_trace = characteristic_mp5(
                averages, cell - 1, gamma
            )
            right_trace, _right_neighbor = characteristic_mp5(
                averages, cell, gamma
            )
        current_left = np.asarray(
            [(-0.5) ** power for power in range(5)]
        ) @ result[cell]
        current_right = np.asarray(
            [0.5**power for power in range(5)]
        ) @ result[cell]
        left_neighbor = _cell_index(cell - 1, len(averages), periodic)
        right_neighbor = _cell_index(cell + 1, len(averages), periodic)
        delta_left = scaled_delta(
            left_trace - current_left,
            averages[left_neighbor],
            averages[cell],
        )
        delta_right = scaled_delta(
            right_trace - current_right,
            averages[cell],
            averages[right_neighbor],
        )
        correction_sum = delta_left + delta_right
        if point_samples:
            # ``-xi + 2 xi^2`` and ``xi + 2 xi^2`` vanish at the known
            # center point and have Kronecker values at the two faces.
            result[cell, 1] += delta_right - delta_left
            result[cell, 2] += 2.0 * correction_sum
        else:
            result[cell, 0] -= 0.25 * correction_sum
            result[cell, 1] += delta_right - delta_left
            result[cell, 2] += 3.0 * correction_sum
    return result


def pwl_coefficients(averages: Array, periodic: bool) -> Array:
    cells, components = averages.shape
    result = np.zeros((cells, 5, components))
    result[:, 0] = averages
    for cell in range(cells):
        values = _stencil(averages, cell, (-1, 0, 1), periodic)
        result[cell, 1] = np.asarray(
            [
                mc_slope(
                    values[0, component],
                    values[1, component],
                    values[2, component],
                )
                for component in range(components)
            ]
        )
    return result


def pwc_coefficients(averages: Array) -> Array:
    result = np.zeros((len(averages), 5, averages.shape[1]))
    result[:, 0] = averages
    return result


def characteristic_mp5_parabolic_coefficients(
    averages: Array,
    gamma: float,
    periodic: bool,
    *,
    point_samples: bool,
) -> Array:
    """Return average-preserving parabolas with characteristic MP5 traces."""

    result = np.zeros((len(averages), 5, averages.shape[1]))
    characteristic_mp5.periodic = periodic
    for cell in range(len(averages)):
        _left_neighbor, left_trace = characteristic_mp5(
            averages, cell - 1, gamma
        )
        right_trace, _right_neighbor = characteristic_mp5(
            averages, cell, gamma
        )
        curvature_factor = 2.0 if point_samples else 3.0
        curvature = curvature_factor * (
            left_trace + right_trace - 2.0 * averages[cell]
        )
        result[cell, 0] = (
            averages[cell]
            if point_samples
            else averages[cell] - curvature / 12.0
        )
        result[cell, 1] = right_trace - left_trace
        result[cell, 2] = curvature
    return result


def ader_point_primitives(
    state: Array,
    gamma: float,
    periodic: bool,
    parameters: PredictorParameters,
) -> tuple[Array, Array]:
    """Apply the first ADER-Prim reconstruction and one center C2P."""

    conserved_polynomial = cweno_z_coefficients(
        state,
        periodic,
        parameters,
        point_samples=False,
    )
    raw_point_state = conserved_polynomial[:, 0]
    point_state = raw_point_state.copy()
    point_primitive = primitive_from_conserved(point_state, gamma)
    success = (
        np.all(np.isfinite(point_primitive), axis=1)
        & (point_primitive[:, 0] > 0.0)
        & (point_primitive[:, 2] > 0.0)
    )
    scale = np.ones(len(state))
    for _ in range(32):
        if np.all(success):
            return point_primitive, scale
        scale[~success] *= 0.5
        point_state[~success] = state[~success] + scale[
            ~success, np.newaxis
        ] * (raw_point_state[~success] - state[~success])
        point_primitive = primitive_from_conserved(point_state, gamma)
        success = (
            np.all(np.isfinite(point_primitive), axis=1)
            & (point_primitive[:, 0] > 0.0)
            & (point_primitive[:, 2] > 0.0)
        )
    base_primitive = primitive_from_conserved(state, gamma)
    point_primitive[~success] = base_primitive[~success]
    scale[~success] = 0.0
    return point_primitive, scale


def _positive_polynomial(
    coefficients: Array,
    averages: Array,
) -> tuple[Array, Array]:
    """Scale density/pressure modes toward their positive cell averages."""

    result = coefficients.copy()
    valid = np.ones(len(result), dtype=bool)
    for cell in range(len(result)):
        nodal = SPACE_EVALUATION @ result[cell]
        theta = 1.0
        for component in (0, 2):
            floor = 1.0e-13 * max(1.0, abs(averages[cell, component]))
            if averages[cell, component] <= floor:
                valid[cell] = False
                theta = 0.0
                continue
            for value in nodal[:, component]:
                correction = value - averages[cell, component]
                if not np.isfinite(value):
                    theta = 0.0
                elif value < floor and correction < 0.0:
                    theta = min(
                        theta,
                        (averages[cell, component] - floor) / (-correction),
                    )
        theta = max(0.0, min(1.0, theta))
        result[cell, 0] = averages[cell] + theta * (
            result[cell, 0] - averages[cell]
        )
        result[cell, 1:] *= theta
        valid[cell] &= theta > 0.0 or primitive_is_admissible(
            averages[cell : cell + 1]
        )
    return result, valid


def primitive_predictor_rhs(
    values: Array,
    derivative_matrix: Array,
    dx: float,
    gamma: float,
) -> Array:
    derivatives = np.einsum(
        "ab,nbc->nac", derivative_matrix, values
    ) / dx
    rho = values[..., 0]
    velocity = values[..., 1]
    pressure = values[..., 2]
    rho_x = derivatives[..., 0]
    velocity_x = derivatives[..., 1]
    pressure_x = derivatives[..., 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return -np.stack(
            (
                velocity * rho_x + rho * velocity_x,
                velocity * velocity_x + pressure_x / rho,
                velocity * pressure_x + gamma * pressure * velocity_x,
            ),
            axis=-1,
        )


def _stage_validity(stages: tuple[Array, ...]) -> Array:
    valid = np.ones(stages[0].shape[0], dtype=bool)
    for stage in stages:
        valid &= np.all(np.isfinite(stage), axis=(1, 2))
        valid &= np.all(stage[..., 0] > 0.0, axis=1)
        valid &= np.all(stage[..., 2] > 0.0, axis=1)
    return valid


def local_predict(
    coefficients: Array,
    averages: Array,
    dt: float,
    dx: float,
    gamma: float,
    substeps: int,
) -> LocalPrediction:
    if substeps < 1:
        raise ValueError("the local predictor needs at least one substep")
    positive, polynomial_valid = _positive_polynomial(
        coefficients, averages
    )
    current = np.einsum("qp,npc->nqc", SPACE_EVALUATION, positive)
    stage_list = []
    substep_dt = dt / substeps
    for _ in range(substeps):
        first = primitive_predictor_rhs(
            current, SPACE_DERIVATIVE, dx, gamma
        )
        middle_first = current + 0.5 * substep_dt * first
        second = primitive_predictor_rhs(
            middle_first, SPACE_DERIVATIVE, dx, gamma
        )
        middle_second = current + 0.5 * substep_dt * second
        third = primitive_predictor_rhs(
            middle_second, SPACE_DERIVATIVE, dx, gamma
        )
        final_stage = current + substep_dt * third
        fourth = primitive_predictor_rhs(
            final_stage, SPACE_DERIVATIVE, dx, gamma
        )
        stage_list.extend((current, middle_first, middle_second, final_stage))
        current = current + (substep_dt / 6.0) * (
            first + 2.0 * second + 2.0 * third + fourth
        )
    stages = tuple(stage_list)
    valid = polynomial_valid & _stage_validity(stages)
    maximum_speed = 0.0
    for stage in stages:
        safe_rho = np.maximum(stage[..., 0], 1.0e-300)
        safe_pressure = np.maximum(stage[..., 2], 0.0)
        speeds = np.abs(stage[..., 1]) + np.sqrt(
            gamma * safe_pressure / safe_rho
        )
        finite = speeds[np.isfinite(speeds)]
        if finite.size:
            maximum_speed = max(maximum_speed, float(np.max(finite)))
    return LocalPrediction(
        boundary_stages=stages,
        time_weights=np.tile(
            RK4_TIME_WEIGHTS / substeps, substeps
        ),
        valid_cells=valid,
        maximum_speed=maximum_speed,
    )


def lsdg_predict(
    coefficients: Array,
    averages: Array,
    dt: float,
    dx: float,
    gamma: float,
    iterations: int,
    tolerance: float,
) -> LocalPrediction:
    """Solve the 1D primitive local space-time DG weak predictor."""

    if iterations < 1:
        raise ValueError("the LSDG predictor needs a Picard iteration")
    positive, polynomial_valid = _positive_polynomial(
        coefficients, averages
    )
    initial = np.einsum(
        "qp,npc->nqc", GAUSS_SPACE_EVALUATION, positive
    )
    values = np.repeat(initial[:, :, np.newaxis, :], 5, axis=2)
    residual = np.full(len(values), math.inf)
    for _ in range(iterations):
        nonconservative_product = np.empty_like(values)
        for time_node in range(5):
            rhs = primitive_predictor_rhs(
                values[:, :, time_node],
                GAUSS_SPACE_DERIVATIVE,
                dx,
                gamma,
            )
            nonconservative_product[:, :, time_node] = -dt * rhs
        weak_rhs = (
            initial[:, :, np.newaxis, :]
            * GAUSS_TIME_INITIAL[np.newaxis, np.newaxis, :, np.newaxis]
            - nonconservative_product
            * GAUSS_WEIGHTS_01[np.newaxis, np.newaxis, :, np.newaxis]
        )
        next_values = np.einsum(
            "lk,nakc->nalc", GAUSS_TIME_WEAK_INVERSE, weak_rhs
        )
        residual = np.max(
            np.abs(next_values - values)
            / np.maximum(1.0, np.abs(next_values)),
            axis=(1, 2, 3),
        )
        values = next_values
    left = np.einsum("q,nqtc->ntc", GAUSS_SPACE_LEFT, values)
    right = np.einsum("q,nqtc->ntc", GAUSS_SPACE_RIGHT, values)
    boundary_stages = tuple(
        np.stack((left[:, time_node], right[:, time_node]), axis=1)
        for time_node in range(5)
    )
    valid = polynomial_valid & _stage_validity(boundary_stages)
    valid &= residual <= tolerance
    valid &= np.all(np.isfinite(values), axis=(1, 2, 3))
    valid &= np.all(values[..., 0] > 0.0, axis=(1, 2))
    valid &= np.all(values[..., 2] > 0.0, axis=(1, 2))
    maximum_speed = 0.0
    for stage in boundary_stages:
        safe_rho = np.maximum(stage[..., 0], 1.0e-300)
        safe_pressure = np.maximum(stage[..., 2], 0.0)
        speeds = np.abs(stage[..., 1]) + np.sqrt(
            gamma * safe_pressure / safe_rho
        )
        finite = speeds[np.isfinite(speeds)]
        if finite.size:
            maximum_speed = max(maximum_speed, float(np.max(finite)))
    return LocalPrediction(
        boundary_stages=boundary_stages,
        time_weights=GAUSS_WEIGHTS_01,
        valid_cells=valid,
        maximum_speed=maximum_speed,
    )


def _boundary_traces(
    stage: Array,
    face: int,
    periodic: bool,
    boundary_primitives: tuple[Array, Array] | None,
) -> tuple[Array, Array]:
    cells = len(stage)
    if periodic:
        return stage[(face - 1) % cells, -1], stage[face % cells, 0]
    if face == 0:
        if boundary_primitives is not None:
            return boundary_primitives[0], boundary_primitives[0]
        return stage[0, 0], stage[0, 0]
    if face == cells:
        if boundary_primitives is not None:
            return boundary_primitives[1], boundary_primitives[1]
        return stage[-1, -1], stage[-1, -1]
    return stage[face - 1, -1], stage[face, 0]


def time_integrated_fluxes(
    stages: tuple[Array, ...],
    time_weights: Array,
    gamma: float,
    periodic: bool,
    boundary_primitives: tuple[Array, Array] | None = None,
) -> tuple[Array, Array, Array]:
    if len(stages) != len(time_weights):
        raise ValueError("predictor stages and time weights disagree")
    cells = len(stages[0])
    fluxes = np.empty((cells + 1, 3))
    valid = np.ones(cells + 1, dtype=bool)
    face_speeds = np.zeros(cells + 1)
    for face in range(cells + 1):
        flux = np.zeros(3)
        for weight, stage in zip(time_weights, stages, strict=True):
            left, right = _boundary_traces(
                stage,
                face,
                periodic,
                boundary_primitives,
            )
            if not primitive_is_admissible(np.asarray((left, right))):
                valid[face] = False
                continue
            face_flux = hllc_flux(left, right, gamma)
            if not np.all(np.isfinite(face_flux)):
                valid[face] = False
                continue
            flux += weight * face_flux
            for primitive in (left, right):
                speed = abs(primitive[1]) + math.sqrt(
                    gamma * primitive[2] / primitive[0]
                )
                face_speeds[face] = max(face_speeds[face], speed)
        fluxes[face] = flux
    if periodic:
        fluxes[-1] = fluxes[0]
        valid[-1] = valid[0]
        face_speeds[-1] = face_speeds[0]
    return fluxes, valid, face_speeds


def _face_orders(cell_orders: Array, periodic: bool) -> Array:
    cells = len(cell_orders)
    result = np.empty(cells + 1, dtype=int)
    for face in range(cells + 1):
        if periodic:
            result[face] = min(
                cell_orders[(face - 1) % cells], cell_orders[face % cells]
            )
        elif face == 0:
            result[face] = cell_orders[0]
        elif face == cells:
            result[face] = cell_orders[-1]
        else:
            result[face] = min(cell_orders[face - 1], cell_orders[face])
    if periodic:
        result[-1] = result[0]
    return result


def _relaxed_dmp_failures(
    old_primitive: Array,
    candidate_primitive: Array,
    cell_orders: Array,
    periodic: bool,
    parameters: PredictorParameters,
) -> Array:
    cells = len(old_primitive)
    scale = np.maximum(np.ptp(old_primitive, axis=0), 1.0)
    bad = np.zeros(cells, dtype=bool)
    for cell in range(cells):
        # The DMP decides whether the primary high-order polynomial is
        # trusted. Once a cell has been recomputed with the already-limited
        # characteristic MP5 parabola, admissibility owns the remaining
        # hierarchy; repeatedly applying the same DMP would mechanically skip
        # that complete fallback and land on PWL.
        if cell_orders[cell] <= 2:
            continue
        neighborhood = _stencil(
            old_primitive, cell, (-1, 0, 1), periodic
        )
        lower = np.min(neighborhood, axis=0)
        upper = np.max(neighborhood, axis=0)
        tolerance = np.maximum(
            parameters.dmp_absolute_tolerance * scale,
            parameters.dmp_relative_tolerance * (upper - lower),
        )
        value = candidate_primitive[cell]
        component_bad = (value < lower - tolerance) | (
            value > upper + tolerance
        )
        if np.any(component_bad):
            five = _stencil(
                old_primitive, cell, (-2, -1, 0, 1, 2), periodic
            )
            curvatures = np.asarray(
                (
                    five[2] - 2.0 * five[1] + five[0],
                    five[3] - 2.0 * five[2] + five[1],
                    five[4] - 2.0 * five[3] + five[2],
                )
            )
            curvature_scale = np.max(np.abs(curvatures), axis=0)
            same_positive = np.all(curvatures > 0.0, axis=0)
            same_negative = np.all(curvatures < 0.0, axis=0)
            balanced_curvature = np.min(
                np.abs(curvatures), axis=0
            ) >= 0.125 * curvature_scale
            fourth_difference = np.abs(
                curvatures[0]
                - 2.0 * curvatures[1]
                + curvatures[2]
            )
            curvature_sum = (
                np.abs(curvatures[0])
                + 2.0 * np.abs(curvatures[1])
                + np.abs(curvatures[2])
            )
            slowly_varying_curvature = fourth_difference <= (
                parameters.smooth_curvature_ratio * curvature_sum
                + parameters.dmp_absolute_tolerance * scale
            )
            smooth_curvature_region = (
                curvature_scale
                > parameters.dmp_absolute_tolerance * scale
            ) & (
                (same_positive | same_negative)
                & balanced_curvature
                & slowly_varying_curvature
            )
            component_bad &= ~smooth_curvature_region
        bad[cell] = bool(np.any(component_bad))
    return bad


def spacetime_step(
    state: Array,
    dt: float,
    dx: float,
    gamma: float,
    periodic: bool,
    parameters: PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None = None,
) -> StepResult:
    primitive = primitive_from_conserved(state, gamma)
    if not primitive_is_admissible(primitive):
        return StepResult(None, math.inf, np.zeros(len(state), dtype=int))
    reconstruction_primitive = primitive
    if parameters.double_reconstruction:
        reconstruction_primitive, _center_scales = ader_point_primitives(
            state, gamma, periodic, parameters
        )
    if parameters.mp5_parabolic_primary:
        high_coefficients = characteristic_mp5_parabolic_coefficients(
            primitive,
            gamma,
            periodic,
            point_samples=False,
        )
        high_primitive = primitive
    else:
        high_coefficients = (
            characteristic_cweno_z_coefficients(
                reconstruction_primitive,
                gamma,
                periodic,
                parameters,
                point_samples=parameters.double_reconstruction,
            )
            if parameters.characteristic_reconstruction
            else cweno_z_coefficients(
                reconstruction_primitive,
                periodic,
                parameters,
                point_samples=parameters.double_reconstruction,
            )
        )
        high_primitive = reconstruction_primitive
    if (
        parameters.anchor_mp5_traces
        and not parameters.mp5_parabolic_primary
    ):
        high_coefficients = anchor_characteristic_mp5_traces(
            high_coefficients,
            reconstruction_primitive,
            gamma,
            periodic,
            point_samples=parameters.double_reconstruction,
            contact_strength=parameters.contact_anchor_strength,
            acoustic_strength=parameters.acoustic_anchor_strength,
            compressive_acoustic_floor=(
                parameters.compressive_acoustic_floor
            ),
            acoustic_shock_scale=parameters.acoustic_shock_scale,
        )
    coefficient_sets = (
        pwc_coefficients(primitive),
        pwl_coefficients(primitive, periodic),
        characteristic_mp5_parabolic_coefficients(
            primitive,
            gamma,
            periodic,
            point_samples=False,
        ),
        high_coefficients,
    )
    center_sets = (primitive, primitive, primitive, high_primitive)
    if parameters.predictor_kind == "collocation":
        predictions = [
            local_predict(
                coefficients,
                centers,
                dt,
                dx,
                gamma,
                parameters.predictor_substeps,
            )
            for coefficients, centers in zip(
                coefficient_sets, center_sets, strict=True
            )
        ]
    elif parameters.predictor_kind == "lsdg":
        predictions = [
            lsdg_predict(
                coefficients,
                centers,
                dt,
                dx,
                gamma,
                parameters.picard_iterations,
                parameters.picard_tolerance,
            )
            for coefficients, centers in zip(
                coefficient_sets, center_sets, strict=True
            )
        ]
    else:
        raise ValueError(
            f"unknown local predictor {parameters.predictor_kind!r}"
        )
    flux_data = [
        time_integrated_fluxes(
            prediction.boundary_stages,
            prediction.time_weights,
            gamma,
            periodic,
            boundary_primitives,
        )
        for prediction in predictions
    ]
    fallback_speed = max(
        float(np.max(flux_result[2])) for flux_result in flux_data
    )
    physical_overrides = (
        physical_subcell.physical_flux_overrides(
            primitive,
            dx,
            dt,
            gamma,
            periodic,
        )
        if parameters.physical_fronts
        else {}
    )
    physical_front_faces = len(physical_overrides)
    if periodic and 0 in physical_overrides and len(state) in physical_overrides:
        physical_front_faces -= 1
    cell_orders = np.full(len(state), 3, dtype=int)
    for cell in range(len(state)):
        while (
            cell_orders[cell] > 0
            and not predictions[cell_orders[cell]].valid_cells[cell]
        ):
            cell_orders[cell] -= 1

    for _ in range(parameters.maximum_correction_iterations):
        face_orders = _face_orders(cell_orders, periodic)
        selected_fluxes = np.empty_like(flux_data[0][0])
        selected_speeds = np.empty(len(face_orders))
        for face, requested_order in enumerate(face_orders):
            order = int(requested_order)
            while order > 0 and not flux_data[order][1][face]:
                order -= 1
            if not flux_data[order][1][face]:
                return StepResult(None, fallback_speed, cell_orders)
            selected_fluxes[face] = flux_data[order][0][face]
            selected_speeds[face] = flux_data[order][2][face]
        for face, flux in physical_overrides.items():
            selected_fluxes[face] = flux
        maximum_speed = float(np.max(selected_speeds))
        candidate = state - (dt / dx) * (
            selected_fluxes[1:] - selected_fluxes[:-1]
        )
        candidate_primitive = primitive_from_conserved(candidate, gamma)
        physical = (
            np.all(np.isfinite(candidate_primitive), axis=1)
            & (candidate_primitive[:, 0] > 0.0)
            & (candidate_primitive[:, 2] > 0.0)
        )
        bad = ~physical
        if np.all(physical):
            bad |= _relaxed_dmp_failures(
                primitive,
                candidate_primitive,
                cell_orders,
                periodic,
                parameters,
            )
        reducible = bad & (cell_orders > 0)
        if not np.any(bad) or not np.any(reducible):
            if np.all(physical):
                return StepResult(
                    candidate,
                    maximum_speed,
                    cell_orders,
                    physical_front_faces,
                )
            return StepResult(None, maximum_speed, cell_orders)
        cell_orders[reducible] -= 1
    return StepResult(None, fallback_speed, cell_orders)


def evolve_spacetime(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
    parameters: PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None = None,
) -> tuple[Array, bool, EvolutionStats]:
    state = conserved_from_primitive(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    stats = EvolutionStats()
    speed_estimate = 0.0
    while time < final_time:
        current = primitive_from_conserved(state, gamma)
        if not primitive_is_admissible(current):
            return current, False, stats
        speeds = np.abs(current[:, 1]) + np.sqrt(
            gamma * current[:, 2] / current[:, 0]
        )
        step_speed = max(float(np.max(speeds)), speed_estimate)
        dt = min(final_time - time, cfl * dx / step_speed)
        accepted: StepResult | None = None
        for _ in range(12):
            trial = spacetime_step(
                state,
                dt,
                dx,
                gamma,
                periodic,
                parameters,
                boundary_primitives,
            )
            if (
                trial.state is not None
                and np.isfinite(trial.maximum_speed)
                and dt * trial.maximum_speed <= cfl * dx * (1.0 + 1.0e-12)
            ):
                accepted = trial
                break
            if (
                np.isfinite(trial.maximum_speed)
                and trial.maximum_speed > 0.0
            ):
                speed_limited_dt = 0.99 * cfl * dx / trial.maximum_speed
                dt = (
                    speed_limited_dt
                    if trial.state is not None
                    else min(dt * 0.5, speed_limited_dt)
                )
            else:
                dt *= 0.5
            stats.rejected_steps += 1
        if accepted is None or accepted.state is None:
            return primitive_from_conserved(state, gamma), False, stats
        state = accepted.state
        time += dt
        speed_estimate = accepted.maximum_speed
        stats.steps += 1
        stats.high_cells += int(np.count_nonzero(accepted.cell_orders == 3))
        stats.mp5_cells += int(np.count_nonzero(accepted.cell_orders == 2))
        stats.pwl_cells += int(np.count_nonzero(accepted.cell_orders == 1))
        stats.pwc_cells += int(np.count_nonzero(accepted.cell_orders == 0))
        stats.physical_front_faces += accepted.physical_front_faces
        stats.minimum_order = min(
            stats.minimum_order, int(np.min(accepted.cell_orders))
        )
    final = primitive_from_conserved(state, gamma)
    return final, primitive_is_admissible(final), stats


def entropy_cell_averages(cells: int, time: float = 0.0) -> Array:
    dx = 1.0 / cells
    centers = (np.arange(cells) + 0.5) * dx
    average_factor = math.sin(math.pi * dx) / (math.pi * dx)
    return np.column_stack(
        (
            1.0
            + 0.2
            * average_factor
            * np.sin(2.0 * math.pi * (centers - 0.5 * time)),
            np.full(cells, 0.5),
            np.ones(cells),
        )
    )


def acoustic_simple_wave_cell_averages(
    cells: int,
    time: float,
    gamma: float,
    amplitude: float = 0.02,
    quadrature_order: int = 16,
) -> Array:
    """Cell-average an exact pre-shock right-going Euler simple wave."""

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    edges = np.linspace(0.0, 1.0, cells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_width = 0.5 / cells
    points = centers[:, np.newaxis] + half_width * nodes
    sound_0 = math.sqrt(gamma)
    lagrangian = np.mod(points - sound_0 * time, 1.0)
    nonlinear_shift = 0.5 * (gamma + 1.0) * amplitude * time
    labels = lagrangian.copy()
    for _ in range(12):
        phase = 2.0 * math.pi * labels
        residual = (
            labels + nonlinear_shift * np.sin(phase) - lagrangian
        )
        derivative = 1.0 + nonlinear_shift * 2.0 * math.pi * np.cos(
            phase
        )
        labels -= residual / derivative
    velocity = amplitude * np.sin(2.0 * math.pi * labels)
    sound = sound_0 + 0.5 * (gamma - 1.0) * velocity
    density = (sound / sound_0) ** (2.0 / (gamma - 1.0))
    pressure = density**gamma
    point_primitive = np.stack((density, velocity, pressure), axis=-1)
    point_state = conserved_from_primitive(point_primitive, gamma)
    average_state = 0.5 * np.einsum(
        "q,cqk->ck",
        weights,
        point_state,
    )
    return primitive_from_conserved(average_state, gamma)


def _smooth_euler_row(
    method: str,
    cells: int,
    cfl: float,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
    stats: EvolutionStats | None,
) -> SmoothEulerResult:
    component_l1 = []
    overshoots = []
    tv_ratios = []
    for component in range(3):
        exact_component = exact[:, component]
        evolved_component = evolved[:, component]
        scale = comparison_scale(exact_component)
        component_l1.append(
            float(np.mean(np.abs(evolved_component - exact_component)))
            / scale
        )
        overshoots.append(
            max(
                0.0,
                float(np.max(evolved_component) - np.max(exact_component)),
                float(np.min(exact_component) - np.min(evolved_component)),
            )
            / scale
        )
        exact_tv = total_variation(exact_component)
        if exact_tv <= 1.0e-12 * scale:
            tv_ratios.append(
                1.0
                + total_variation(evolved_component - exact_component)
                / scale
            )
        else:
            tv_ratios.append(
                total_variation(evolved_component) / exact_tv
            )
    exact_integrals = np.mean(conserved_from_primitive(exact, gamma), axis=0)
    evolved_integrals = np.mean(
        conserved_from_primitive(evolved, gamma), axis=0
    )
    integral_error = np.abs(evolved_integrals - exact_integrals)
    return SmoothEulerResult(
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        maximum_l1=max(component_l1),
        observed_order=math.nan,
        maximum_overshoot=max(overshoots),
        maximum_tv_ratio=max(tv_ratios),
        mass_error=float(integral_error[0]),
        momentum_error=float(integral_error[1]),
        energy_error=float(integral_error[2]),
        steps=0 if stats is None else stats.steps,
        mp5_cell_steps=0 if stats is None else stats.mp5_cells,
        pwl_cell_steps=0 if stats is None else stats.pwl_cells,
        pwc_cell_steps=0 if stats is None else stats.pwc_cells,
    )


def run_acoustic_simple_wave(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    parameters: PredictorParameters,
    final_time: float,
    amplitude: float,
) -> list[SmoothEulerResult]:
    baseline_methods = {
        "ppm4_ssprk3": component_reconstruction(ppm4_left),
        "mp5_characteristic_ssprk3": characteristic_mp5,
    }
    rows: list[SmoothEulerResult] = []
    previous_errors: dict[str, tuple[int, float]] = {}
    for cells in cells_values:
        initial = acoustic_simple_wave_cell_averages(
            cells, 0.0, gamma, amplitude
        )
        exact = acoustic_simple_wave_cell_averages(
            cells, final_time, gamma, amplitude
        )
        current_rows = []
        for method, reconstruct in baseline_methods.items():
            evolved, completed = evolve_ssprk3(
                initial,
                final_time,
                cfl,
                gamma,
                reconstruct,
                hllc_flux,
                periodic=True,
            )
            current_rows.append(
                _smooth_euler_row(
                    method,
                    cells,
                    cfl,
                    evolved,
                    exact,
                    completed,
                    gamma,
                    None,
                )
            )
        evolved, completed, stats = evolve_spacetime(
            initial,
            final_time,
            cfl,
            gamma,
            periodic=True,
            parameters=parameters,
        )
        current_rows.append(
            _smooth_euler_row(
                candidate_name(parameters),
                cells,
                cfl,
                evolved,
                exact,
                completed,
                gamma,
                stats,
            )
        )
        for row in current_rows:
            previous = previous_errors.get(row.method)
            order = math.nan
            if previous is not None and row.maximum_l1 > 0.0:
                previous_cells, previous_error = previous
                order = math.log(previous_error / row.maximum_l1) / math.log(
                    cells / previous_cells
                )
            rows.append(dataclasses.replace(row, observed_order=order))
            previous_errors[row.method] = (cells, row.maximum_l1)
    return rows


def _smooth_row(
    method: str,
    cells: int,
    cfl: float,
    evolved: Array,
    exact: Array,
    completed: bool,
    stats: EvolutionStats | None,
) -> SmoothResult:
    error = float(np.mean(np.abs(evolved[:, 0] - exact[:, 0])))
    return SmoothResult(
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        density_l1=error,
        observed_order=math.nan,
        maximum_overshoot=max(
            0.0,
            float(np.max(evolved[:, 0]) - np.max(exact[:, 0])),
            float(np.min(exact[:, 0]) - np.min(evolved[:, 0])),
        ),
        tv_ratio=total_variation(evolved[:, 0])
        / total_variation(exact[:, 0]),
        mass_error=abs(float(np.mean(evolved[:, 0])) - 1.0),
        steps=0 if stats is None else stats.steps,
        mp5_cell_steps=0 if stats is None else stats.mp5_cells,
        pwl_cell_steps=0 if stats is None else stats.pwl_cells,
        pwc_cell_steps=0 if stats is None else stats.pwc_cells,
    )


def run_entropy(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    parameters: PredictorParameters,
    final_time: float,
) -> list[SmoothResult]:
    baseline_methods = {
        "ppm4_ssprk3": component_reconstruction(ppm4_left),
        "mp5_ssprk3": component_reconstruction(mp5_left),
        "mp5_characteristic_ssprk3": characteristic_mp5,
    }
    rows: list[SmoothResult] = []
    previous_errors: dict[str, tuple[int, float]] = {}
    for cells in cells_values:
        initial = entropy_cell_averages(cells)
        exact = entropy_cell_averages(cells, final_time)
        current_rows = []
        for method, reconstruct in baseline_methods.items():
            evolved, completed = evolve_ssprk3(
                initial,
                final_time=final_time,
                cfl=cfl,
                gamma=gamma,
                reconstruct=reconstruct,
                riemann_flux=hllc_flux,
                periodic=True,
            )
            current_rows.append(
                _smooth_row(
                    method,
                    cells,
                    cfl,
                    evolved,
                    exact,
                    completed,
                    None,
                )
            )
        evolved, completed, stats = evolve_spacetime(
            initial,
            final_time=final_time,
            cfl=cfl,
            gamma=gamma,
            periodic=True,
            parameters=parameters,
        )
        current_rows.append(
            _smooth_row(
                candidate_name(parameters),
                cells,
                cfl,
                evolved,
                exact,
                completed,
                stats,
            )
        )
        for row in current_rows:
            previous = previous_errors.get(row.method)
            order = math.nan
            if previous is not None and row.density_l1 > 0.0:
                previous_cells, previous_error = previous
                order = math.log(previous_error / row.density_l1) / math.log(
                    cells / previous_cells
                )
            rows.append(dataclasses.replace(row, observed_order=order))
            previous_errors[row.method] = (cells, row.density_l1)
    return rows


def run_shocks(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    parameters: PredictorParameters,
    requested_cases: tuple[str, ...],
) -> tuple[list[ShockResult], dict[tuple[str, int], EvolutionStats]]:
    baseline_methods = {
        "ppm4_ssprk3": component_reconstruction(ppm4_left),
        "mp5_characteristic_ssprk3": characteristic_mp5,
    }
    rows: list[ShockResult] = []
    candidate_stats: dict[tuple[str, int], EvolutionStats] = {}
    for cells in cells_values:
        x = (np.arange(cells) + 0.5) / cells
        for case in EXACT_CASES:
            if case.name not in requested_cases:
                continue
            initial = np.empty((cells, 3))
            initial[x < 0.5] = case.left
            initial[x >= 0.5] = case.right
            boundary_primitives = (
                np.asarray(case.left),
                np.asarray(case.right),
            )
            for method, reconstruct in baseline_methods.items():
                evolved, completed = evolve_ssprk3(
                    initial,
                    case.final_time,
                    cfl,
                    gamma,
                    reconstruct,
                    hllc_flux,
                    periodic=False,
                    boundary_primitives=boundary_primitives,
                )
                rows.append(
                    shock_result(
                        case,
                        method,
                        cells,
                        evolved,
                        completed,
                        gamma,
                    )
                )
            evolved, completed, stats = evolve_spacetime(
                initial,
                case.final_time,
                cfl,
                gamma,
                periodic=False,
                parameters=parameters,
                boundary_primitives=boundary_primitives,
            )
            rows.append(
                shock_result(
                    case,
                    candidate_name(parameters),
                    cells,
                    evolved,
                    completed,
                    gamma,
                )
            )
            candidate_stats[(case.name, cells)] = stats
    return rows, candidate_stats


def _mirrored_case(case: ShockCase) -> ShockCase:
    return ShockCase(
        f"{case.name}_mirrored",
        (case.right[0], -case.right[1], case.right[2]),
        (case.left[0], -case.left[1], case.left[2]),
        case.final_time,
    )


def _initial_step(case: ShockCase, cells: int) -> Array:
    x = (np.arange(cells) + 0.5) / cells
    initial = np.empty((cells, 3))
    initial[x < 0.5] = case.left
    initial[x >= 0.5] = case.right
    return initial


def _symmetry_row(
    case: ShockCase,
    method: str,
    cells: int,
    original: Array,
    mirrored: Array,
    completed: bool,
) -> SymmetryResult:
    reflected = mirrored[::-1].copy()
    reflected[:, 1] *= -1.0
    difference = np.abs(original - reflected)
    scales = np.asarray(
        [comparison_scale(original[:, component]) for component in range(3)]
    )
    normalized_l1 = np.mean(difference, axis=0) / scales
    return SymmetryResult(
        case=case.name,
        method=method,
        cells=cells,
        completed=completed,
        maximum_normalized_l1=float(np.max(normalized_l1)),
        maximum_absolute_error=float(np.max(difference)),
    )


def run_symmetry(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    parameters: PredictorParameters,
    requested_cases: tuple[str, ...],
) -> list[SymmetryResult]:
    baseline_methods = {
        "ppm4_ssprk3": component_reconstruction(ppm4_left),
        "mp5_characteristic_ssprk3": characteristic_mp5,
    }
    rows: list[SymmetryResult] = []
    for cells in cells_values:
        for case in EXACT_CASES:
            if case.name not in requested_cases:
                continue
            mirror = _mirrored_case(case)
            initial = _initial_step(case, cells)
            mirrored_initial = _initial_step(mirror, cells)
            boundaries = (
                np.asarray(case.left),
                np.asarray(case.right),
            )
            mirrored_boundaries = (
                np.asarray(mirror.left),
                np.asarray(mirror.right),
            )
            for method, reconstruct in baseline_methods.items():
                original, original_completed = evolve_ssprk3(
                    initial,
                    case.final_time,
                    cfl,
                    gamma,
                    reconstruct,
                    hllc_flux,
                    periodic=False,
                    boundary_primitives=boundaries,
                )
                mirrored, mirrored_completed = evolve_ssprk3(
                    mirrored_initial,
                    mirror.final_time,
                    cfl,
                    gamma,
                    reconstruct,
                    hllc_flux,
                    periodic=False,
                    boundary_primitives=mirrored_boundaries,
                )
                rows.append(
                    _symmetry_row(
                        case,
                        method,
                        cells,
                        original,
                        mirrored,
                        original_completed and mirrored_completed,
                    )
                )
            original, original_completed, _ = evolve_spacetime(
                initial,
                case.final_time,
                cfl,
                gamma,
                periodic=False,
                parameters=parameters,
                boundary_primitives=boundaries,
            )
            mirrored, mirrored_completed, _ = evolve_spacetime(
                mirrored_initial,
                mirror.final_time,
                cfl,
                gamma,
                periodic=False,
                parameters=parameters,
                boundary_primitives=mirrored_boundaries,
            )
            rows.append(
                _symmetry_row(
                    case,
                    candidate_name(parameters),
                    cells,
                    original,
                    mirrored,
                    original_completed and mirrored_completed,
                )
            )
    return rows


def print_rows(rows: list[object]) -> None:
    if not rows:
        return
    fields = tuple(field.name for field in dataclasses.fields(rows[0]))
    print("\t".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = getattr(row, field)
            if isinstance(value, float):
                values.append(f"{value:.12e}")
            else:
                values.append(str(value))
        print("\t".join(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32, 64, 128))
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--entropy-final-time", type=float, default=2.0)
    parser.add_argument("--acoustic-final-time", type=float, default=0.2)
    parser.add_argument("--acoustic-amplitude", type=float, default=0.02)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "entropy",
            "acoustic",
            "sod",
            "lax",
            "blast",
            "contact",
        ),
        default=("entropy", "sod"),
    )
    parser.add_argument(
        "--symmetry-check",
        action="store_true",
        help="also compare each requested shock against its reflection",
    )
    parser.add_argument("--central-linear-weight", type=float, default=0.85)
    parser.add_argument("--wenoz-exponent", type=int, default=2)
    parser.add_argument("--dmp-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--predictor-substeps", type=int, default=1)
    parser.add_argument(
        "--predictor-kind",
        choices=("collocation", "lsdg"),
        default="collocation",
    )
    parser.add_argument("--picard-iterations", type=int, default=8)
    parser.add_argument("--picard-tolerance", type=float, default=math.inf)
    parser.add_argument(
        "--component-cweno",
        action="store_true",
        help="use shared component weights instead of wavewise weights",
    )
    parser.add_argument(
        "--no-mp5-anchor",
        action="store_true",
        help="leave CWENO traces unanchored for an ablation run",
    )
    parser.add_argument("--contact-anchor-strength", type=float, default=1.0)
    parser.add_argument("--acoustic-anchor-strength", type=float, default=1.0)
    parser.add_argument("--compressive-acoustic-floor", type=float, default=0.4)
    parser.add_argument("--acoustic-shock-scale", type=float, default=0.2)
    parser.add_argument(
        "--mp5-parabolic-primary",
        action="store_true",
        help="ablate CWENO and use the MP5-trace parabola everywhere",
    )
    parser.add_argument(
        "--single-reconstruction",
        action="store_true",
        help="ablate ADER-Prim and reconstruct C2P(Ubar) directly",
    )
    parser.add_argument(
        "--physical-fronts",
        action="store_true",
        help="enable strict Rankine-Hugoniot subcell front ownership",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cfl <= 0.0:
        raise ValueError("--cfl must be positive")
    if args.gamma <= 1.0:
        raise ValueError("--gamma must exceed one")
    if args.entropy_final_time <= 0.0 or args.acoustic_final_time <= 0.0:
        raise ValueError("smooth-wave final times must be positive")
    if args.acoustic_amplitude <= 0.0:
        raise ValueError("--acoustic-amplitude must be positive")
    minimum_map_derivative = 1.0 - (
        math.pi
        * (args.gamma + 1.0)
        * args.acoustic_amplitude
        * args.acoustic_final_time
    )
    if minimum_map_derivative <= 0.0:
        raise ValueError("the acoustic simple wave has already formed a shock")
    if any(cells < 8 for cells in args.cells):
        raise ValueError("all grids need at least eight cells")
    if not 0.0 < args.central_linear_weight < 1.0:
        raise ValueError("--central-linear-weight must lie between zero and one")
    if args.wenoz_exponent < 1:
        raise ValueError("--wenoz-exponent must be positive")
    if args.dmp_relative_tolerance < 0.0:
        raise ValueError("--dmp-relative-tolerance must be nonnegative")
    if args.predictor_substeps < 1 or args.picard_iterations < 1:
        raise ValueError("predictor iteration counts must be positive")
    if args.picard_tolerance <= 0.0:
        raise ValueError("--picard-tolerance must be positive")
    strengths = (
        args.contact_anchor_strength,
        args.acoustic_anchor_strength,
        args.compressive_acoustic_floor,
    )
    if any(strength < 0.0 or strength > 1.0 for strength in strengths):
        raise ValueError("anchor strengths must lie between zero and one")
    if args.compressive_acoustic_floor > args.acoustic_anchor_strength:
        raise ValueError(
            "--compressive-acoustic-floor cannot exceed acoustic strength"
        )
    if args.acoustic_shock_scale < 0.0:
        raise ValueError("--acoustic-shock-scale must be nonnegative")
    parameters = PredictorParameters(
        central_linear_weight=args.central_linear_weight,
        wenoz_exponent=args.wenoz_exponent,
        dmp_relative_tolerance=args.dmp_relative_tolerance,
        characteristic_reconstruction=not args.component_cweno,
        anchor_mp5_traces=not args.no_mp5_anchor,
        contact_anchor_strength=args.contact_anchor_strength,
        acoustic_anchor_strength=args.acoustic_anchor_strength,
        compressive_acoustic_floor=args.compressive_acoustic_floor,
        acoustic_shock_scale=args.acoustic_shock_scale,
        mp5_parabolic_primary=args.mp5_parabolic_primary,
        predictor_substeps=args.predictor_substeps,
        predictor_kind=args.predictor_kind,
        picard_iterations=args.picard_iterations,
        picard_tolerance=args.picard_tolerance,
        double_reconstruction=not args.single_reconstruction,
        physical_fronts=args.physical_fronts,
    )
    cells_values = tuple(sorted(set(args.cells)))
    if "entropy" in args.cases:
        print("[smooth entropy wave]")
        print_rows(
            run_entropy(
                cells_values,
                args.cfl,
                args.gamma,
                parameters,
                args.entropy_final_time,
            )
        )
    if "acoustic" in args.cases:
        print("[smooth acoustic simple wave]")
        print_rows(
            run_acoustic_simple_wave(
                cells_values,
                args.cfl,
                args.gamma,
                parameters,
                args.acoustic_final_time,
                args.acoustic_amplitude,
            )
        )
    shock_cases = tuple(
        case for case in args.cases if case not in ("entropy", "acoustic")
    )
    if shock_cases:
        rows, stats = run_shocks(
            cells_values,
            args.cfl,
            args.gamma,
            parameters,
            shock_cases,
        )
        print("\n[exact Euler shock tubes]")
        print_rows(rows)
        print("\n[candidate correction counts]")
        for key, value in sorted(stats.items()):
            print(
                key[0],
                key[1],
                f"steps={value.steps}",
                f"rejected={value.rejected_steps}",
                f"mp5_cell_steps={value.mp5_cells}",
                f"pwl_cell_steps={value.pwl_cells}",
                f"pwc_cell_steps={value.pwc_cells}",
                f"physical_front_faces={value.physical_front_faces}",
                f"minimum_order={value.minimum_order}",
            )
        if args.symmetry_check:
            print("\n[reflection symmetry]")
            print_rows(
                run_symmetry(
                    cells_values,
                    args.cfl,
                    args.gamma,
                    parameters,
                    shock_cases,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
