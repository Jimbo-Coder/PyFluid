#!/usr/bin/env python3
"""Test the coherent primitive space-time operator on ideal MHD.

This is the next offline rung after ``research_primitive_spacetime_fv.py``.
It reuses that tool's CWENO-Z reconstruction and weak local space-time
collocation, supplies the one-dimensional ideal-MHD primitive equations, and
uses one time-integrated HLLD flux per shared face.  The current gates cover a
cell-averaged circular Alfvén wave, complete conservative fallback, and an
optional strict Rankine--Hugoniot physical-front correction.  CT, AMR,
relativistic systems, and production C++ remain out of scope until the
standalone Newtonian MHD matrix passes.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import research_contact_correction as contact_correction  # noqa: E402
import research_mhd_wave_reconstruction as mhd  # noqa: E402
import research_physical_wave_subcell_mhd as physical_mhd  # noqa: E402
import research_primitive_spacetime_fv as spacetime  # noqa: E402
import research_subcell_mhd_reconstruction as shock_mhd  # noqa: E402
from pyfluid.exact_solutions import circular_alfven_wave_exact  # noqa: E402
from research_constrained_riemann import (  # noqa: E402
    mhd_fast_speed,
    mhd_hlld_flux,
)


Array = np.ndarray
MHD_ACOUSTIC_SHOCK_SCALE = 0.05


@dataclasses.dataclass(frozen=True)
class MhdStepResult:
    state: Array | None
    maximum_speed: float
    cell_orders: Array
    physical_front_faces: int = 0


@dataclasses.dataclass
class MhdEvolutionStats:
    steps: int = 0
    cfl_retries: int = 0
    high_cells: int = 0
    mp5_cells: int = 0
    pwl_cells: int = 0
    pwc_cells: int = 0
    physical_front_faces: int = 0
    minimum_order: int = 3


@dataclasses.dataclass(frozen=True)
class SmoothMhdResult:
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
    normal_b_spread: float
    steps: int
    cfl_retries: int
    mp5_cell_steps: int
    pwl_cell_steps: int
    pwc_cell_steps: int
    physical_front_faces: int


def characteristic_cweno_z_coefficients(
    averages: Array,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    *,
    point_samples: bool,
) -> Array:
    """Build one MHD wavewise CWENO-Z polynomial with constant Bn."""

    cells, components = averages.shape
    if components != 8:
        raise ValueError("the ideal-MHD adapter expects eight primitives")
    central_weight = parameters.central_linear_weight
    side_weight = (1.0 - central_weight) / 3.0
    linear_weights = np.asarray(
        (central_weight, side_weight, side_weight, side_weight)
    )
    result = np.zeros((cells, 5, components))
    epsilon = 128.0 * np.finfo(float).eps
    quadratic_inverses = (
        spacetime.POINT_QUADRATIC_INVERSES
        if point_samples
        else spacetime.QUADRATIC_INVERSES
    )
    quartic_inverse = (
        spacetime.POINT_QUARTIC_INVERSE
        if point_samples
        else spacetime.QUARTIC_INVERSE
    )
    for cell in range(cells):
        primitive_five = spacetime._stencil(
            averages,
            cell,
            spacetime.QUARTIC_OFFSETS,
            periodic,
        )
        normal_b = float(averages[cell, 5])
        basis = mhd.characteristic_basis(
            averages[cell], averages[cell], gamma, normal_b
        )
        waves_five = mhd.to_characteristics(
            primitive_five[:, mhd.REDUCED], basis
        )
        optimal = quartic_inverse @ waves_five
        lower = []
        for offsets, inverse in zip(
            spacetime.QUADRATIC_OFFSETS,
            quadratic_inverses,
            strict=True,
        ):
            primitive_stencil = spacetime._stencil(
                averages, cell, offsets, periodic
            )
            waves = mhd.to_characteristics(
                primitive_stencil[:, mhd.REDUCED], basis
            )
            coefficients = np.zeros((5, 7))
            coefficients[:3] = inverse @ waves
            lower.append(coefficients)
        candidates = np.empty((4, 5, 7))
        candidates[1:] = lower
        candidates[0] = (
            optimal
            - side_weight * (lower[0] + lower[1] + lower[2])
        ) / central_weight
        indicators = np.asarray(
            [
                spacetime._candidate_smoothness(candidate)
                for candidate in candidates
            ]
        )
        variation = np.ptp(waves_five, axis=0)
        amplitude = np.max(np.abs(waves_five), axis=0)
        constant = variation <= (
            256.0 * np.finfo(float).eps * np.maximum(amplitude, 1.0)
        )
        scales = np.maximum(
            variation,
            64.0 * np.finfo(float).eps * np.maximum(amplitude, 1.0),
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
            wave_coefficients[0, constant] = waves_five[2, constant]
        for power, coefficients_at_power in enumerate(wave_coefficients):
            result[cell, power, mhd.REDUCED] = mhd.from_characteristics(
                coefficients_at_power, basis
            )
        result[cell, 0, 5] = normal_b
        if point_samples:
            result[cell, 0] = averages[cell]
        else:
            result[cell, 0] += averages[cell] - spacetime._polynomial_average(
                result[cell]
            )
    return result


def characteristic_point_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
    periodic: bool,
) -> tuple[Array, Array]:
    stencil = np.asarray(
        [
            mhd.stencil_value(
                primitive,
                left_cell + offset,
                periodic,
            )
            for offset in range(-2, 4)
        ]
    )
    normal_b = 0.5 * (stencil[2, 5] + stencil[3, 5])
    basis = mhd.characteristic_basis(
        stencil[2], stencil[3], gamma, normal_b
    )
    waves = mhd.to_characteristics(stencil[:, mhd.REDUCED], basis)
    left_waves = np.asarray(
        [
            spacetime.point_mp5_left(*waves[:, wave])
            for wave in range(7)
        ]
    )
    right_waves = np.asarray(
        [
            spacetime.point_mp5_left(*waves[::-1, wave])
            for wave in range(7)
        ]
    )
    left_reduced = mhd.positive_trace(
        mhd.from_characteristics(left_waves, basis),
        stencil[2, mhd.REDUCED],
    )
    right_reduced = mhd.positive_trace(
        mhd.from_characteristics(right_waves, basis),
        stencil[3, mhd.REDUCED],
    )
    left = np.insert(left_reduced, 5, normal_b)
    right = np.insert(right_reduced, 5, normal_b)
    return left, right


def _material_characteristic_fraction(
    first: Array,
    second: Array,
    gamma: float,
) -> float:
    normal_b = 0.5 * (first[5] + second[5])
    basis = mhd.characteristic_basis(first, second, gamma, normal_b)
    amplitudes = mhd.to_characteristics(
        (second - first)[np.asarray(mhd.REDUCED)][np.newaxis, :],
        basis,
    )[0]
    total = float(np.sum(np.abs(amplitudes)))
    if total <= 256.0 * np.finfo(float).eps:
        return 0.0
    return float(np.clip(abs(amplitudes[3]) / total, 0.0, 1.0))


def _magnetic_activation(first: Array, second: Array) -> float:
    center = 0.5 * (first + second)
    alfven_speed2 = float(np.dot(center[5:8], center[5:8])) / max(
        float(center[0]), np.finfo(float).tiny
    )
    return float(np.clip(alfven_speed2, 0.0, 1.0))


def characteristic_mp5_contact_corrected(
    primitive: Array,
    left_cell: int,
    gamma: float,
    periodic: bool,
) -> tuple[Array, Array]:
    """Apply the offline material-contact correction to MHD MP5 traces."""

    mhd.characteristic_mp5.periodic = periodic
    left, right = mhd.characteristic_mp5(primitive, left_cell, gamma)
    stencil = np.asarray(
        [
            mhd.stencil_value(primitive, left_cell + offset, periodic)
            for offset in range(-2, 4)
        ]
    )
    density_left, density_right = (
        contact_correction.ppm3_contact_density_pair(stencil)
    )
    left_weight, right_weight = contact_correction.density_trace_weights(
        stencil,
        lambda first, second: _material_characteristic_fraction(
            first, second, gamma
        ),
        activation_weight=_magnetic_activation,
    )
    left[0] += left_weight * (density_left - left[0])
    right[0] += right_weight * (density_right - right[0])
    reduced = np.asarray(mhd.REDUCED)
    left[reduced] = mhd.positive_trace(
        left[reduced], stencil[2, reduced]
    )
    right[reduced] = mhd.positive_trace(
        right[reduced], stencil[3, reduced]
    )
    return left, right


def anchor_mp5_traces(
    coefficients: Array,
    point_primitive: Array,
    gamma: float,
    periodic: bool,
    compressive_floor: float,
    shock_scale: float,
) -> Array:
    """Give MP5 ownership of both traces while preserving center points."""

    result = coefficients.copy()
    left_evaluation = np.asarray([(-0.5) ** power for power in range(5)])
    right_evaluation = np.asarray([0.5**power for power in range(5)])

    def scaled_delta(delta: Array, first: Array, second: Array) -> Array:
        normal_b = 0.5 * (first[5] + second[5])
        face_basis = mhd.characteristic_basis(
            first, second, gamma, normal_b
        )
        waves = mhd.to_characteristics(
            delta[np.asarray(mhd.REDUCED)][np.newaxis, :], face_basis
        )[0]
        first_basis = mhd.characteristic_basis(
            first, first, gamma, normal_b
        )
        second_basis = mhd.characteristic_basis(
            second, second, gamma, normal_b
        )
        first_speeds = np.asarray(
            (
                first[1] - first_basis.fast,
                first[1] - first_basis.slow,
                first[1] + first_basis.slow,
                first[1] + first_basis.fast,
            )
        )
        second_speeds = np.asarray(
            (
                second[1] - second_basis.fast,
                second[1] - second_basis.slow,
                second[1] + second_basis.slow,
                second[1] + second_basis.fast,
            )
        )
        speed_scale = max(first_basis.fast, second_basis.fast, 1.0e-14)
        compression = np.maximum(0.0, first_speeds - second_speeds)
        if shock_scale <= 0.0:
            strengths = np.ones(4)
        else:
            strengths = compressive_floor + (
                1.0 - compressive_floor
            ) / (1.0 + (compression / (shock_scale * speed_scale)) ** 4)
        waves[np.asarray((0, 2, 4, 6))] *= strengths
        reduced = mhd.from_characteristics(waves, face_basis)
        return np.insert(reduced, 5, 0.0)

    for cell in range(len(point_primitive)):
        _left_neighbor, left_trace = characteristic_point_mp5(
            point_primitive, cell - 1, gamma, periodic
        )
        right_trace, _right_neighbor = characteristic_point_mp5(
            point_primitive, cell, gamma, periodic
        )
        current_left = left_evaluation @ result[cell]
        current_right = right_evaluation @ result[cell]
        left_neighbor = spacetime._cell_index(
            cell - 1, len(point_primitive), periodic
        )
        right_neighbor = spacetime._cell_index(
            cell + 1, len(point_primitive), periodic
        )
        delta_left = scaled_delta(
            left_trace - current_left,
            point_primitive[left_neighbor],
            point_primitive[cell],
        )
        delta_right = scaled_delta(
            right_trace - current_right,
            point_primitive[cell],
            point_primitive[right_neighbor],
        )
        correction_sum = delta_left + delta_right
        result[cell, 1] += delta_right - delta_left
        result[cell, 2] += 2.0 * correction_sum
        result[cell, 1:, 5] = 0.0
        result[cell, 0, 5] = point_primitive[cell, 5]
    return result


def characteristic_mp5_parabolic_coefficients(
    averages: Array,
    gamma: float,
    periodic: bool,
    *,
    contact_correction_enabled: bool = False,
) -> Array:
    """Build one average-preserving parabola from MHD MP5 traces."""

    result = np.zeros((len(averages), 5, 8))
    mhd.characteristic_mp5.periodic = periodic
    for cell in range(len(averages)):
        if contact_correction_enabled:
            _left_neighbor, left_trace = (
                characteristic_mp5_contact_corrected(
                    averages, cell - 1, gamma, periodic
                )
            )
            right_trace, _right_neighbor = (
                characteristic_mp5_contact_corrected(
                    averages, cell, gamma, periodic
                )
            )
        else:
            _left_neighbor, left_trace = mhd.characteristic_mp5(
                averages, cell - 1, gamma
            )
            right_trace, _right_neighbor = mhd.characteristic_mp5(
                averages, cell, gamma
            )
        curvature = 3.0 * (
            left_trace + right_trace - 2.0 * averages[cell]
        )
        result[cell, 0] = averages[cell] - curvature / 12.0
        result[cell, 1] = right_trace - left_trace
        result[cell, 2] = curvature
        result[cell, 1:, 5] = 0.0
        result[cell, 0, 5] = averages[cell, 5]
    return result


def _relaxed_dmp_failures(
    old_primitive: Array,
    candidate_primitive: Array,
    cell_orders: Array,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
) -> Array:
    cells = len(old_primitive)
    scale = np.maximum(np.ptp(old_primitive, axis=0), 1.0)
    variation = np.ptp(old_primitive, axis=0)
    amplitude = np.max(np.abs(old_primitive), axis=0)
    dimensionless_activity = variation / np.maximum(amplitude, 1.0)
    activity_scale = float(np.max(dimensionless_activity))
    active_components = dimensionless_activity > (
        parameters.sensor_relative_activity_floor * activity_scale
    )
    bad = np.zeros(cells, dtype=bool)
    for cell in range(cells):
        if cell_orders[cell] <= 2:
            continue
        neighborhood = spacetime._stencil(
            old_primitive, cell, (-1, 0, 1), periodic
        )
        lower = np.min(neighborhood, axis=0)
        upper = np.max(neighborhood, axis=0)
        tolerance = np.maximum(
            parameters.dmp_absolute_tolerance * scale,
            parameters.dmp_relative_tolerance * (upper - lower),
        )
        component_bad = (candidate_primitive[cell] < lower - tolerance) | (
            candidate_primitive[cell] > upper + tolerance
        )
        component_bad &= active_components
        if np.any(component_bad):
            five = spacetime._stencil(
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
            same_sign = np.all(curvatures > 0.0, axis=0) | np.all(
                curvatures < 0.0, axis=0
            )
            balanced = np.min(np.abs(curvatures), axis=0) >= (
                0.125 * curvature_scale
            )
            fourth_difference = np.abs(
                curvatures[0] - 2.0 * curvatures[1] + curvatures[2]
            )
            curvature_sum = (
                np.abs(curvatures[0])
                + 2.0 * np.abs(curvatures[1])
                + np.abs(curvatures[2])
            )
            slowly_varying = fourth_difference <= (
                parameters.smooth_curvature_ratio * curvature_sum
                + parameters.dmp_absolute_tolerance * scale
            )
            smooth_curvature = (
                curvature_scale
                > parameters.dmp_absolute_tolerance * scale
            ) & same_sign & balanced & slowly_varying
            component_bad &= ~smooth_curvature
        bad[cell] = bool(np.any(component_bad))
    return bad


def _positive_polynomial(
    coefficients: Array,
    centers: Array,
) -> tuple[Array, Array]:
    result = coefficients.copy()
    valid = np.ones(len(result), dtype=bool)
    for cell in range(len(result)):
        nodal = spacetime.SPACE_EVALUATION @ result[cell]
        theta = 1.0
        for component in (0, 4):
            floor = 1.0e-13 * max(1.0, abs(centers[cell, component]))
            if centers[cell, component] <= floor:
                valid[cell] = False
                theta = 0.0
                continue
            for value in nodal[:, component]:
                correction = value - centers[cell, component]
                if not np.isfinite(value):
                    theta = 0.0
                elif value < floor and correction < 0.0:
                    theta = min(
                        theta,
                        (centers[cell, component] - floor) / (-correction),
                    )
        theta = max(0.0, min(1.0, theta))
        result[cell, 0] = centers[cell] + theta * (
            result[cell, 0] - centers[cell]
        )
        result[cell, 1:] *= theta
        valid[cell] &= theta > 0.0 or mhd.primitive_is_admissible(
            centers[cell : cell + 1]
        )
    return result, valid


def primitive_predictor_rhs(
    values: Array,
    derivative_matrix: Array,
    dx: float,
    gamma: float,
) -> Array:
    """Return ``-A(V) dV/dx`` for 1D ideal MHD with constant Bn."""

    derivatives = np.einsum(
        "ab,nbc->nac", derivative_matrix, values
    ) / dx
    rho = values[..., 0]
    vn = values[..., 1]
    pressure = values[..., 4]
    bn = values[..., 5]
    bt1 = values[..., 6]
    bt2 = values[..., 7]
    rho_x = derivatives[..., 0]
    vn_x = derivatives[..., 1]
    vt1_x = derivatives[..., 2]
    vt2_x = derivatives[..., 3]
    pressure_x = derivatives[..., 4]
    bt1_x = derivatives[..., 6]
    bt2_x = derivatives[..., 7]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.stack(
            (
                -vn * rho_x - rho * vn_x,
                -vn * vn_x
                - (pressure_x + bt1 * bt1_x + bt2 * bt2_x) / rho,
                -vn * vt1_x + bn * bt1_x / rho,
                -vn * vt2_x + bn * bt2_x / rho,
                -vn * pressure_x - gamma * pressure * vn_x,
                np.zeros_like(rho),
                -vn * bt1_x - bt1 * vn_x + bn * vt1_x,
                -vn * bt2_x - bt2 * vn_x + bn * vt2_x,
            ),
            axis=-1,
        )


def _stage_validity(stages: tuple[Array, ...]) -> Array:
    valid = np.ones(stages[0].shape[0], dtype=bool)
    for stage in stages:
        valid &= np.all(np.isfinite(stage), axis=(1, 2))
        valid &= np.all(stage[..., 0] > 0.0, axis=1)
        valid &= np.all(stage[..., 4] > 0.0, axis=1)
        normal_b = stage[..., 5]
        valid &= np.ptp(normal_b, axis=1) <= 1.0e-12 * np.maximum(
            1.0, np.max(np.abs(normal_b), axis=1)
        )
    return valid


def lsdg_predict(
    coefficients: Array,
    centers: Array,
    dt: float,
    dx: float,
    gamma: float,
    iterations: int,
) -> spacetime.LocalPrediction:
    positive, polynomial_valid = _positive_polynomial(
        coefficients, centers
    )
    initial = np.einsum(
        "qp,npc->nqc", spacetime.GAUSS_SPACE_EVALUATION, positive
    )
    values = np.repeat(initial[:, :, np.newaxis, :], 5, axis=2)
    for _ in range(iterations):
        nonconservative_product = np.empty_like(values)
        for time_node in range(5):
            rhs = primitive_predictor_rhs(
                values[:, :, time_node],
                spacetime.GAUSS_SPACE_DERIVATIVE,
                dx,
                gamma,
            )
            nonconservative_product[:, :, time_node] = -dt * rhs
        weak_rhs = (
            initial[:, :, np.newaxis, :]
            * spacetime.GAUSS_TIME_INITIAL[
                np.newaxis, np.newaxis, :, np.newaxis
            ]
            - nonconservative_product
            * spacetime.GAUSS_WEIGHTS_01[
                np.newaxis, np.newaxis, :, np.newaxis
            ]
        )
        values = np.einsum(
            "lk,nakc->nalc",
            spacetime.GAUSS_TIME_WEAK_INVERSE,
            weak_rhs,
        )
    left = np.einsum(
        "q,nqtc->ntc", spacetime.GAUSS_SPACE_LEFT, values
    )
    right = np.einsum(
        "q,nqtc->ntc", spacetime.GAUSS_SPACE_RIGHT, values
    )
    boundary_stages = tuple(
        np.stack((left[:, time_node], right[:, time_node]), axis=1)
        for time_node in range(5)
    )
    valid = polynomial_valid & _stage_validity(boundary_stages)
    valid &= np.all(np.isfinite(values), axis=(1, 2, 3))
    maximum_speed = 0.0
    for stage in boundary_stages:
        for primitive in stage.reshape(-1, 8):
            if mhd.primitive_is_admissible(primitive[np.newaxis, :]):
                maximum_speed = max(
                    maximum_speed,
                    abs(primitive[1]) + mhd_fast_speed(primitive, gamma),
                )
    return spacetime.LocalPrediction(
        boundary_stages=boundary_stages,
        time_weights=spacetime.GAUSS_WEIGHTS_01,
        valid_cells=valid,
        maximum_speed=maximum_speed,
    )


def point_primitives(
    state: Array,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
) -> Array:
    conserved_polynomial = spacetime.cweno_z_coefficients(
        state,
        periodic,
        parameters,
        point_samples=False,
    )
    raw = conserved_polynomial[:, 0]
    point_state = raw.copy()
    point_primitive = mhd.primitive_from_conserved(point_state, gamma)
    success = (
        np.all(np.isfinite(point_primitive), axis=1)
        & (point_primitive[:, 0] > 0.0)
        & (point_primitive[:, 4] > 0.0)
    )
    scale = np.ones(len(state))
    for _ in range(32):
        if np.all(success):
            return point_primitive
        scale[~success] *= 0.5
        point_state[~success] = state[~success] + scale[
            ~success, np.newaxis
        ] * (raw[~success] - state[~success])
        point_primitive = mhd.primitive_from_conserved(point_state, gamma)
        success = (
            np.all(np.isfinite(point_primitive), axis=1)
            & (point_primitive[:, 0] > 0.0)
            & (point_primitive[:, 4] > 0.0)
        )
    base = mhd.primitive_from_conserved(state, gamma)
    point_primitive[~success] = base[~success]
    return point_primitive


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
    prediction: spacetime.LocalPrediction,
    gamma: float,
    periodic: bool,
    boundary_primitives: tuple[Array, Array] | None,
) -> tuple[Array, Array, Array]:
    cells = len(prediction.boundary_stages[0])
    fluxes = np.empty((cells + 1, 8))
    valid = np.ones(cells + 1, dtype=bool)
    speeds = np.zeros(cells + 1)
    for face in range(cells + 1):
        flux = np.zeros(8)
        for weight, stage in zip(
            prediction.time_weights,
            prediction.boundary_stages,
            strict=True,
        ):
            left, right = _boundary_traces(
                stage, face, periodic, boundary_primitives
            )
            if not mhd.primitive_is_admissible(np.asarray((left, right))):
                valid[face] = False
                continue
            face_flux = mhd_hlld_flux(left, right, gamma)
            if not np.all(np.isfinite(face_flux)):
                valid[face] = False
                continue
            flux += weight * face_flux
            speeds[face] = max(
                speeds[face],
                abs(left[1]) + mhd_fast_speed(left, gamma),
                abs(right[1]) + mhd_fast_speed(right, gamma),
            )
        fluxes[face] = flux
    if periodic:
        fluxes[-1] = fluxes[0]
        valid[-1] = valid[0]
        speeds[-1] = speeds[0]
    return fluxes, valid, speeds


def spacetime_step(
    state: Array,
    dt: float,
    dx: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None = None,
    *,
    contact_correction_enabled: bool = False,
) -> MhdStepResult:
    primitive = mhd.primitive_from_conserved(state, gamma)
    if not mhd.primitive_is_admissible(primitive):
        return MhdStepResult(
            None, math.inf, np.zeros(len(state), dtype=int)
        )
    center_primitive = point_primitives(
        state, gamma, periodic, parameters
    )
    if parameters.mp5_parabolic_primary:
        coefficients = characteristic_mp5_parabolic_coefficients(
            primitive,
            gamma,
            periodic,
            contact_correction_enabled=contact_correction_enabled,
        )
        high_primitive = primitive
    else:
        coefficients = characteristic_cweno_z_coefficients(
            center_primitive,
            gamma,
            periodic,
            parameters,
            point_samples=True,
        )
        high_primitive = center_primitive
    if parameters.anchor_mp5_traces and not parameters.mp5_parabolic_primary:
        coefficients = anchor_mp5_traces(
            coefficients,
            center_primitive,
            gamma,
            periodic,
            parameters.compressive_acoustic_floor,
            parameters.acoustic_shock_scale,
        )
    coefficient_sets = (
        spacetime.pwc_coefficients(primitive),
        spacetime.pwl_coefficients(primitive, periodic),
        characteristic_mp5_parabolic_coefficients(
            primitive,
            gamma,
            periodic,
            contact_correction_enabled=contact_correction_enabled,
        ),
        coefficients,
    )
    center_sets = (primitive, primitive, primitive, high_primitive)
    predictions = [
        lsdg_predict(
            candidate_coefficients,
            candidate_centers,
            dt,
            dx,
            gamma,
            parameters.picard_iterations,
        )
        for candidate_coefficients, candidate_centers in zip(
            coefficient_sets, center_sets, strict=True
        )
    ]
    flux_data = [
        time_integrated_fluxes(
            prediction,
            gamma,
            periodic,
            boundary_primitives,
        )
        for prediction in predictions
    ]
    physical_overrides = (
        physical_mhd.physical_flux_overrides(
            primitive,
            dx,
            dt,
            gamma,
            periodic,
        )
        if parameters.physical_fronts
        else {}
    )
    fallback_speed = max(
        float(np.max(flux_result[2])) for flux_result in flux_data
    )
    cell_orders = np.full(len(state), 3, dtype=int)
    for cell in range(len(state)):
        while (
            cell_orders[cell] > 0
            and not predictions[cell_orders[cell]].valid_cells[cell]
        ):
            cell_orders[cell] -= 1
    for _ in range(parameters.maximum_correction_iterations):
        face_orders = spacetime._face_orders(cell_orders, periodic)
        selected_fluxes = np.empty_like(flux_data[0][0])
        selected_speeds = np.empty(len(face_orders))
        for face, requested_order in enumerate(face_orders):
            order = int(requested_order)
            while order > 0 and not flux_data[order][1][face]:
                order -= 1
            if not flux_data[order][1][face]:
                return MhdStepResult(None, fallback_speed, cell_orders)
            selected_fluxes[face] = flux_data[order][0][face]
            selected_speeds[face] = flux_data[order][2][face]
        effective_physical_faces: set[int] = set()
        for face, flux in physical_overrides.items():
            difference = float(np.linalg.norm(flux - selected_fluxes[face]))
            flux_scale = max(
                float(np.linalg.norm(flux)),
                float(np.linalg.norm(selected_fluxes[face])),
                1.0,
            )
            if difference > 256.0 * np.finfo(float).eps * flux_scale:
                effective_physical_faces.add(
                    face % len(state) if periodic else face
                )
            selected_fluxes[face] = flux
        physical_front_faces = len(effective_physical_faces)
        maximum_speed = float(np.max(selected_speeds))
        candidate = state - (dt / dx) * (
            selected_fluxes[1:] - selected_fluxes[:-1]
        )
        candidate_primitive = mhd.primitive_from_conserved(candidate, gamma)
        physical = (
            np.all(np.isfinite(candidate_primitive), axis=1)
            & (candidate_primitive[:, 0] > 0.0)
            & (candidate_primitive[:, 4] > 0.0)
        )
        normal_b_scale = max(float(np.max(np.abs(candidate[:, 5]))), 1.0)
        normal_b_valid = (
            np.ptp(candidate[:, 5]) <= 1.0e-12 * normal_b_scale
        )
        bad = ~physical
        if np.all(physical) and normal_b_valid:
            bad |= _relaxed_dmp_failures(
                primitive,
                candidate_primitive,
                cell_orders,
                periodic,
                parameters,
            )
        elif not normal_b_valid:
            bad[:] = True
        reducible = bad & (cell_orders > 0)
        if not np.any(bad) or not np.any(reducible):
            if np.all(physical) and normal_b_valid:
                return MhdStepResult(
                    candidate,
                    maximum_speed,
                    cell_orders,
                    physical_front_faces,
                )
            return MhdStepResult(None, maximum_speed, cell_orders)
        cell_orders[reducible] -= 1
    return MhdStepResult(None, fallback_speed, cell_orders)


def evolve_spacetime(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    parameters: spacetime.PredictorParameters,
    periodic: bool = True,
    boundary_primitives: tuple[Array, Array] | None = None,
    *,
    contact_correction_enabled: bool = True,
) -> tuple[Array, bool, MhdEvolutionStats]:
    state = mhd.conserved_array(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    stats = MhdEvolutionStats()
    speed_estimate = 0.0
    while time < final_time:
        current = mhd.primitive_from_conserved(state, gamma)
        if not mhd.primitive_is_admissible(current):
            return current, False, stats
        cell_speed = max(
            abs(row[1]) + mhd_fast_speed(row, gamma) for row in current
        )
        dt = min(
            final_time - time,
            cfl * dx / max(cell_speed, speed_estimate),
        )
        accepted: MhdStepResult | None = None
        for _ in range(12):
            trial = spacetime_step(
                state,
                dt,
                dx,
                gamma,
                periodic,
                parameters,
                boundary_primitives,
                contact_correction_enabled=contact_correction_enabled,
            )
            if (
                trial.state is not None
                and np.isfinite(trial.maximum_speed)
                and dt * trial.maximum_speed
                <= cfl * dx * (1.0 + 1.0e-12)
            ):
                accepted = trial
                break
            if np.isfinite(trial.maximum_speed) and trial.maximum_speed > 0.0:
                speed_dt = 0.99 * cfl * dx / trial.maximum_speed
                dt = (
                    speed_dt
                    if trial.state is not None
                    else min(dt * 0.5, speed_dt)
                )
            else:
                dt *= 0.5
            stats.cfl_retries += 1
        if accepted is None or accepted.state is None:
            return current, False, stats
        state = accepted.state
        speed_estimate = accepted.maximum_speed
        time += dt
        stats.steps += 1
        stats.high_cells += int(np.count_nonzero(accepted.cell_orders == 3))
        stats.mp5_cells += int(np.count_nonzero(accepted.cell_orders == 2))
        stats.pwl_cells += int(np.count_nonzero(accepted.cell_orders == 1))
        stats.pwc_cells += int(np.count_nonzero(accepted.cell_orders == 0))
        stats.physical_front_faces += accepted.physical_front_faces
        stats.minimum_order = min(
            stats.minimum_order, int(np.min(accepted.cell_orders))
        )
    final = mhd.primitive_from_conserved(state, gamma)
    return final, mhd.primitive_is_admissible(final), stats


def circular_alfven_cell_averages(
    cells: int,
    time: float,
    gamma: float,
    amplitude: float,
    quadrature_order: int = 16,
) -> Array:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    centers = (np.arange(cells) + 0.5) / cells
    points = centers[:, np.newaxis] + 0.5 * nodes / cells
    values = circular_alfven_wave_exact(
        points.reshape(-1),
        time,
        rho0=1.0,
        pressure0=1.0,
        bx0=1.0,
        amplitude=amplitude,
        wave_number=1.0,
        direction=1.0,
    )
    rho, pressure, vx, vy, vz, bx, by, bz = values
    primitive = np.column_stack(
        (rho, vx, vy, vz, pressure, bx, by, bz)
    )
    point_state = mhd.conserved_array(primitive, gamma).reshape(cells, -1, 8)
    average_state = 0.5 * np.einsum(
        "q,cqk->ck", weights, point_state
    )
    return mhd.primitive_from_conserved(average_state, gamma)


def _result(
    method: str,
    cells: int,
    cfl: float,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
    stats: MhdEvolutionStats | None,
) -> SmoothMhdResult:
    component_l1 = []
    overshoots = []
    tv_ratios = []
    density_scale = max(float(np.max(np.abs(exact[:, 0]))), 1.0e-14)
    velocity_scale = max(
        float(np.max(np.linalg.norm(exact[:, 1:4], axis=1))),
        1.0e-14,
    )
    pressure_scale = max(float(np.max(np.abs(exact[:, 4]))), 1.0e-14)
    magnetic_scale = max(
        float(np.max(np.linalg.norm(exact[:, 5:8], axis=1))),
        1.0e-14,
    )
    component_scales = np.asarray(
        (
            density_scale,
            velocity_scale,
            velocity_scale,
            velocity_scale,
            pressure_scale,
            magnetic_scale,
            magnetic_scale,
            magnetic_scale,
        )
    )
    for component in range(8):
        exact_component = exact[:, component]
        evolved_component = evolved[:, component]
        component_scale = max(
            float(np.ptp(exact_component)),
            float(component_scales[component]),
        )
        component_l1.append(
            float(np.mean(np.abs(evolved_component - exact_component)))
            / component_scale
        )
        overshoots.append(
            max(
                0.0,
                float(np.max(evolved_component) - np.max(exact_component)),
                float(np.min(exact_component) - np.min(evolved_component)),
            )
            / component_scale
        )
        exact_tv = mhd.total_variation(exact_component)
        if exact_tv <= 1.0e-12 * component_scale:
            tv_ratios.append(
                1.0
                + mhd.total_variation(evolved_component - exact_component)
                / component_scale
            )
        else:
            tv_ratios.append(
                mhd.total_variation(evolved_component) / exact_tv
            )
    exact_integrals = np.mean(mhd.conserved_array(exact, gamma), axis=0)
    evolved_integrals = np.mean(mhd.conserved_array(evolved, gamma), axis=0)
    integral_error = np.abs(evolved_integrals - exact_integrals)
    return SmoothMhdResult(
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        maximum_l1=max(component_l1),
        observed_order=math.nan,
        maximum_overshoot=max(overshoots),
        maximum_tv_ratio=max(tv_ratios),
        mass_error=float(integral_error[0]),
        momentum_error=float(np.max(integral_error[1:4])),
        energy_error=float(integral_error[4]),
        normal_b_spread=float(np.ptp(evolved[:, 5])),
        steps=0 if stats is None else stats.steps,
        cfl_retries=0 if stats is None else stats.cfl_retries,
        mp5_cell_steps=0 if stats is None else stats.mp5_cells,
        pwl_cell_steps=0 if stats is None else stats.pwl_cells,
        pwc_cell_steps=0 if stats is None else stats.pwc_cells,
        physical_front_faces=(
            0 if stats is None else stats.physical_front_faces
        ),
    )


def run(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    amplitude: float,
    parameters: spacetime.PredictorParameters,
) -> list[SmoothMhdResult]:
    rows: list[SmoothMhdResult] = []
    previous: dict[str, tuple[int, float]] = {}
    for cells in cells_values:
        initial = circular_alfven_cell_averages(
            cells, 0.0, gamma, amplitude
        )
        exact = circular_alfven_cell_averages(
            cells, 1.0, gamma, amplitude
        )
        baseline, baseline_completed = mhd.evolve(
            initial,
            1.0,
            cfl,
            gamma,
            mhd.characteristic_mp5,
            mhd_hlld_flux,
            periodic=True,
        )
        current = [
            _result(
                "mp5_characteristic_ssprk3_hlld",
                cells,
                cfl,
                baseline,
                exact,
                baseline_completed,
                gamma,
                None,
            )
        ]
        candidate, completed, stats = evolve_spacetime(
            initial, 1.0, cfl, gamma, parameters, periodic=True
        )
        current.append(
            _result(
                "cweno_z_aderprim_lsdg_hlld",
                cells,
                cfl,
                candidate,
                exact,
                completed,
                gamma,
                stats,
            )
        )
        for row in current:
            prior = previous.get(row.method)
            order = math.nan
            if prior is not None and row.maximum_l1 > 0.0:
                previous_cells, previous_error = prior
                order = math.log(previous_error / row.maximum_l1) / math.log(
                    cells / previous_cells
                )
            rows.append(dataclasses.replace(row, observed_order=order))
            previous[row.method] = (cells, row.maximum_l1)
    return rows


def run_shock_tube_references(
    cells_values: tuple[int, ...],
    cfl: float,
    parameters: spacetime.PredictorParameters,
    reference_factor: int,
) -> list[mhd.Result]:
    """Compare the contact correction with the same coherent MHD operator."""

    specifications = (
        ("brio_wu", mhd.brio_wu_initial, 2.0, 0.1),
        (
            "dai_woodward",
            shock_mhd.dai_woodward_initial,
            5.0 / 3.0,
            0.08,
        ),
    )
    rows: list[mhd.Result] = []
    for case, initial_data, gamma, final_time in specifications:
        for cells in cells_values:
            reference, reference_completed = mhd.evolve(
                initial_data(cells * reference_factor),
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
            initial = initial_data(cells)
            boundaries = (initial[0].copy(), initial[-1].copy())
            baseline, baseline_completed = mhd.evolve(
                initial,
                final_time,
                cfl,
                gamma,
                mhd.characteristic_mp5,
                mhd_hlld_flux,
                periodic=False,
            )
            rows.append(
                mhd.result_from_exact(
                    f"{case}_reference",
                    "mp5_characteristic_ssprk3_hlld",
                    baseline,
                    exact,
                    baseline_completed,
                    gamma,
                )
            )
            for method, correction_enabled in (
                ("coherent_baseline", False),
                ("coherent_contact_corrected", True),
            ):
                evolved, completed, _stats = evolve_spacetime(
                    initial,
                    final_time,
                    cfl,
                    gamma,
                    parameters,
                    periodic=False,
                    boundary_primitives=boundaries,
                    contact_correction_enabled=correction_enabled,
                )
                rows.append(
                    mhd.result_from_exact(
                        f"{case}_reference",
                        method,
                        evolved,
                        exact,
                        completed,
                        gamma,
                    )
                )
    return rows


def print_rows(rows: list[SmoothMhdResult]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(SmoothMhdResult))
    print("\t".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = getattr(row, field)
            values.append(f"{value:.12e}" if isinstance(value, float) else str(value))
        print("\t".join(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--picard-iterations", type=int, default=8)
    parser.add_argument("--physical-fronts", action="store_true")
    parser.add_argument("--shock-tubes", action="store_true")
    parser.add_argument("--reference-factor", type=int, default=8)
    parser.add_argument(
        "--sensor-relative-activity-floor",
        type=float,
        default=1.0e-7,
    )
    args = parser.parse_args()
    if any(cells < 8 for cells in args.cells):
        raise ValueError("all grids need at least eight cells")
    if args.cfl <= 0.0 or args.gamma <= 1.0 or args.amplitude <= 0.0:
        raise ValueError("CFL, gamma, and amplitude are outside their domains")
    if args.picard_iterations < 1:
        raise ValueError("--picard-iterations must be positive")
    if args.reference_factor < 2:
        raise ValueError("--reference-factor must be at least two")
    if not 0.0 <= args.sensor_relative_activity_floor < 1.0:
        raise ValueError("the sensor activity floor must lie in [0, 1)")
    parameters = spacetime.PredictorParameters(
        predictor_kind="lsdg",
        picard_iterations=args.picard_iterations,
        acoustic_shock_scale=MHD_ACOUSTIC_SHOCK_SCALE,
        physical_fronts=args.physical_fronts,
        sensor_relative_activity_floor=args.sensor_relative_activity_floor,
    )
    print("[cell-averaged circular Alfven wave]")
    print_rows(
        run(
            tuple(sorted(set(args.cells))),
            args.cfl,
            args.gamma,
            args.amplitude,
            parameters,
        )
    )
    if args.shock_tubes:
        print("[MHD shock tubes against fine numerical references]")
        mhd.print_results(
            run_shock_tube_references(
                tuple(sorted(set(args.cells))),
                args.cfl,
                parameters,
                args.reference_factor,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
