#!/usr/bin/env python3
"""Carry the coherent primitive space-time FV prototype into SRHD.

This standalone adapter reuses the common CWENO-Z polynomial and weak local
space-time DG machinery from ``research_primitive_spacetime_fv.py``.  Its
system-specific pieces are the flat-Valencia conserved map, C2P recovery,
HLLC flux with HLLE fallback, exact primitive quasilinear Jacobian, causal
admissibility, and the strict isolated physical-front carrier.  It is a
falsification harness, not a production AMReX method.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_physical_wave_subcell_srhd as srhd  # noqa: E402
import research_primitive_spacetime_fv as spacetime  # noqa: E402


Array = np.ndarray
STRONG_SHOCK_DMP_TOLERANCE = 0.1
DMP_COMPRESSION_SCALE = 0.5


@dataclasses.dataclass(frozen=True)
class StepResult:
    state: Array | None
    maximum_speed: float
    cell_orders: Array
    physical_front_faces: int = 0


@dataclasses.dataclass
class EvolutionStats:
    steps: int = 0
    cfl_retries: int = 0
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
    maximum_l1: float
    observed_order: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    mass_error: float
    momentum_error: float
    energy_error: float
    steps: int
    cfl_retries: int
    mp5_cell_steps: int
    pwl_cell_steps: int
    pwc_cell_steps: int
    physical_front_faces: int


def point_primitives(
    state: Array,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
) -> tuple[Array, Array]:
    """Apply the first ADER-Prim conserved reconstruction and one C2P."""

    conserved_polynomial = spacetime.cweno_z_coefficients(
        state,
        periodic,
        parameters,
        point_samples=False,
    )
    raw_point_state = conserved_polynomial[:, 0]
    point_state = raw_point_state.copy()
    point_primitive = srhd.primitive_from_conserved(point_state, gamma)
    success = np.asarray(
        [srhd.primitive_is_admissible(row[None, :]) for row in point_primitive]
    )
    scale = np.ones(len(state))
    for _ in range(32):
        if np.all(success):
            return point_primitive, scale
        scale[~success] *= 0.5
        point_state[~success] = state[~success] + scale[
            ~success, np.newaxis
        ] * (raw_point_state[~success] - state[~success])
        point_primitive = srhd.primitive_from_conserved(point_state, gamma)
        success = np.asarray(
            [
                srhd.primitive_is_admissible(row[None, :])
                for row in point_primitive
            ]
        )
    base_primitive = srhd.primitive_from_conserved(state, gamma)
    point_primitive[~success] = base_primitive[~success]
    scale[~success] = 0.0
    return point_primitive, scale


def primitive_flux_jacobian(primitive: Array, gamma: float) -> Array:
    """Return the exact primitive quasilinear Jacobian for 1D SRHD."""

    rho, velocity, pressure = np.asarray(primitive, dtype=float)
    if (
        not np.isfinite(rho + velocity + pressure)
        or rho <= 0.0
        or pressure <= 0.0
        or abs(velocity) >= 1.0
    ):
        return np.full((3, 3), np.nan)
    adiabatic_factor = gamma / (gamma - 1.0)
    lorentz = 1.0 / math.sqrt(1.0 - velocity * velocity)
    lorentz2 = lorentz * lorentz
    lorentz3 = lorentz2 * lorentz
    lorentz4 = lorentz2 * lorentz2
    enthalpy_density = rho + adiabatic_factor * pressure
    conserved_jacobian = np.array(
        (
            (lorentz, rho * velocity * lorentz3, 0.0),
            (
                lorentz2 * velocity,
                enthalpy_density
                * (1.0 + velocity * velocity)
                * lorentz4,
                adiabatic_factor * lorentz2 * velocity,
            ),
            (
                lorentz2 - lorentz,
                2.0 * enthalpy_density * velocity * lorentz4
                - rho * velocity * lorentz3,
                adiabatic_factor * lorentz2 - 1.0,
            ),
        )
    )
    flux_jacobian = np.array(
        (
            (lorentz * velocity, rho * lorentz3, 0.0),
            (
                lorentz2 * velocity * velocity,
                2.0 * enthalpy_density * velocity * lorentz4,
                adiabatic_factor * lorentz2 * velocity * velocity + 1.0,
            ),
            (
                velocity * (lorentz2 - lorentz),
                enthalpy_density
                * (1.0 + velocity * velocity)
                * lorentz4
                - rho * lorentz3,
                adiabatic_factor * lorentz2 * velocity,
            ),
        )
    )
    return np.linalg.solve(conserved_jacobian, flux_jacobian)


def characteristic_basis(
    left: Array,
    right: Array,
    gamma: float,
) -> tuple[Array, Array]:
    """Build a frozen numerical primitive eigensystem at one face/cell."""

    reference = 0.5 * (np.asarray(left) + np.asarray(right))
    jacobian = primitive_flux_jacobian(reference, gamma)
    eigenvalues, right_vectors = np.linalg.eig(jacobian)
    if (
        np.max(np.abs(np.imag(eigenvalues))) > 1.0e-11
        or np.max(np.abs(np.imag(right_vectors))) > 1.0e-11
    ):
        raise RuntimeError("SRHD primitive eigensystem became complex")
    order = np.argsort(np.real(eigenvalues))
    right_basis = np.real(right_vectors[:, order])
    for wave in range(3):
        right_basis[:, wave] /= max(
            float(np.linalg.norm(right_basis[:, wave])),
            np.finfo(float).tiny,
        )
    left_basis = np.linalg.inv(right_basis)
    return left_basis, right_basis


def characteristic_cweno_z_coefficients(
    primitive: Array,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    *,
    point_samples: bool,
) -> Array:
    """Reconstruct SRHD primitive point data in frozen local waves."""

    cells, components = primitive.shape
    central_weight = parameters.central_linear_weight
    side_weight = (1.0 - central_weight) / 3.0
    linear_weights = np.array(
        (central_weight, side_weight, side_weight, side_weight)
    )
    result = np.empty((cells, 5, components))
    epsilon = 128.0 * np.finfo(float).eps
    for cell in range(cells):
        five_primitive = spacetime._stencil(
            primitive, cell, spacetime.QUARTIC_OFFSETS, periodic
        )
        left_basis, right_basis = characteristic_basis(
            primitive[cell], primitive[cell], gamma
        )
        five = five_primitive @ left_basis.T
        optimal = (
            spacetime.POINT_QUARTIC_INVERSE
            if point_samples
            else spacetime.QUARTIC_INVERSE
        ) @ five
        lower = []
        quadratic_inverses = (
            spacetime.POINT_QUADRATIC_INVERSES
            if point_samples
            else spacetime.QUADRATIC_INVERSES
        )
        for offsets, inverse in zip(
            spacetime.QUADRATIC_OFFSETS,
            quadratic_inverses,
            strict=True,
        ):
            stencil = spacetime._stencil(
                primitive, cell, offsets, periodic
            )
            coefficients = np.zeros((5, components))
            coefficients[:3] = inverse @ (stencil @ left_basis.T)
            lower.append(coefficients)
        candidates = np.empty((4, 5, components))
        candidates[1:] = lower
        candidates[0] = (
            optimal - side_weight * (lower[0] + lower[1] + lower[2])
        ) / central_weight
        indicators = np.asarray(
            [
                spacetime._candidate_smoothness(candidate)
                for candidate in candidates
            ]
        )
        variation = np.ptp(five, axis=0)
        amplitude = np.max(np.abs(five), axis=0)
        dimensionless_activity = variation / np.maximum(amplitude, 1.0)
        activity_scale = float(np.max(dimensionless_activity))
        inactive = variation <= (
            256.0 * np.finfo(float).eps * np.maximum(amplitude, 1.0)
        )
        inactive |= dimensionless_activity <= (
            parameters.sensor_relative_activity_floor * activity_scale
        )
        scales = np.maximum(
            variation,
            64.0 * np.finfo(float).eps * np.maximum(amplitude, 1.0),
        )
        normalized = indicators / (scales * scales)
        normalized[:, inactive] = 0.0
        tau = np.abs(normalized[1] - normalized[3])
        alpha = linear_weights[:, np.newaxis] * (
            1.0
            + (
                tau[np.newaxis, :] / (epsilon + normalized)
            )
            ** parameters.wenoz_exponent
        )
        weights = alpha / np.sum(alpha, axis=0, keepdims=True)
        wave_coefficients = np.einsum("sc,spc->pc", weights, candidates)
        if np.any(inactive):
            wave_coefficients[:, inactive] = 0.0
            wave_coefficients[0, inactive] = five[2, inactive]
        coefficients = wave_coefficients @ right_basis.T
        if point_samples:
            coefficients[0] = primitive[cell]
        else:
            coefficients[0] += primitive[cell] - spacetime._polynomial_average(
                coefficients
            )
        result[cell] = coefficients
    return result


def characteristic_point_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
    periodic: bool,
) -> tuple[Array, Array]:
    stencil = np.asarray(
        [
            spacetime.stencil_value(
                primitive, left_cell + offset, periodic
            )
            for offset in range(-2, 4)
        ]
    )
    left_basis, right_basis = characteristic_basis(
        stencil[2], stencil[3], gamma
    )
    waves = stencil @ left_basis.T
    left_waves = np.asarray(
        [spacetime.point_mp5_left(*waves[:, wave]) for wave in range(3)]
    )
    right_waves = np.asarray(
        [
            spacetime.point_mp5_left(*waves[::-1, wave])
            for wave in range(3)
        ]
    )
    left = left_waves @ right_basis.T
    right = right_waves @ right_basis.T
    left_scale = srhd.admissible_pair_scale(left, stencil[2])
    right_scale = srhd.admissible_pair_scale(right, stencil[3])
    return (
        stencil[2] + left_scale * (left - stencil[2]),
        stencil[3] + right_scale * (right - stencil[3]),
    )


def anchor_mp5_traces(
    coefficients: Array,
    point_primitive: Array,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
) -> Array:
    """Give causal characteristic MP5 ownership of the two face traces."""

    result = coefficients.copy()
    left_evaluation = np.asarray([(-0.5) ** power for power in range(5)])
    right_evaluation = np.asarray([0.5**power for power in range(5)])

    def scaled_delta(delta: Array, first: Array, second: Array) -> Array:
        left_basis, right_basis = characteristic_basis(first, second, gamma)
        waves = left_basis @ delta
        first_speeds = np.sort(
            np.real(np.linalg.eigvals(primitive_flux_jacobian(first, gamma)))
        )
        second_speeds = np.sort(
            np.real(np.linalg.eigvals(primitive_flux_jacobian(second, gamma)))
        )
        speed_scale = max(
            float(np.max(np.abs(first_speeds))),
            float(np.max(np.abs(second_speeds))),
            1.0e-14,
        )
        compression_minus = max(
            0.0, first_speeds[0] - second_speeds[0]
        ) / speed_scale
        compression_plus = max(
            0.0, first_speeds[2] - second_speeds[2]
        ) / speed_scale

        def acoustic_strength(compression: float) -> float:
            if parameters.acoustic_shock_scale <= 0.0:
                return parameters.acoustic_anchor_strength
            return parameters.compressive_acoustic_floor + (
                parameters.acoustic_anchor_strength
                - parameters.compressive_acoustic_floor
            ) / (
                1.0
                + (
                    compression / parameters.acoustic_shock_scale
                )
                ** 4
            )

        waves *= np.asarray(
            (
                acoustic_strength(compression_minus),
                parameters.contact_anchor_strength,
                acoustic_strength(compression_plus),
            )
        )
        return right_basis @ waves

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
        result[cell, 1] += delta_right - delta_left
        result[cell, 2] += 2.0 * (delta_left + delta_right)
    return result


def characteristic_mp5_parabolic_coefficients(
    primitive: Array,
    gamma: float,
    periodic: bool,
) -> Array:
    """Return average-preserving parabolas with causal SRHD MP5 faces."""

    result = np.zeros((len(primitive), 5, primitive.shape[1]))
    for cell in range(len(primitive)):
        _left_neighbor, left_trace = characteristic_point_mp5(
            primitive, cell - 1, gamma, periodic
        )
        right_trace, _right_neighbor = characteristic_point_mp5(
            primitive, cell, gamma, periodic
        )
        curvature = 3.0 * (
            left_trace + right_trace - 2.0 * primitive[cell]
        )
        result[cell, 0] = primitive[cell] - curvature / 12.0
        result[cell, 1] = right_trace - left_trace
        result[cell, 2] = curvature
    return result


def _admissible_polynomial(
    coefficients: Array,
    centers: Array,
) -> tuple[Array, Array]:
    """Scale all modes together until every predictor node is causal."""

    result = coefficients.copy()
    valid = np.ones(len(result), dtype=bool)
    for cell in range(len(result)):
        original = result[cell].copy()
        theta = 1.0
        accepted = False
        for _ in range(54):
            trial = original.copy()
            trial[0] = centers[cell] + theta * (
                original[0] - centers[cell]
            )
            trial[1:] *= theta
            nodal = spacetime.SPACE_EVALUATION @ trial
            if srhd.primitive_is_admissible(nodal):
                result[cell] = trial
                accepted = True
                break
            theta *= 0.5
        if not accepted:
            result[cell] = 0.0
            result[cell, 0] = centers[cell]
            valid[cell] = srhd.primitive_is_admissible(
                centers[cell : cell + 1]
            )
    return result, valid


def primitive_predictor_rhs(
    values: Array,
    derivative_matrix: Array,
    dx: float,
    gamma: float,
) -> Array:
    """Evaluate ``-dU/dV^-1 dF/dV V_x`` for flat-Valencia SRHD."""

    derivatives = np.einsum(
        "ab,nbc->nac", derivative_matrix, values
    ) / dx
    result = np.full_like(values, np.nan)
    for cell in range(values.shape[0]):
        for node in range(values.shape[1]):
            primitive_jacobian = primitive_flux_jacobian(
                values[cell, node], gamma
            )
            if not np.all(np.isfinite(primitive_jacobian)):
                continue
            result[cell, node] = -primitive_jacobian @ derivatives[cell, node]
    return result


def _stage_validity(stages: tuple[Array, ...]) -> Array:
    valid = np.ones(stages[0].shape[0], dtype=bool)
    for stage in stages:
        valid &= np.asarray(
            [srhd.primitive_is_admissible(cell) for cell in stage]
        )
    return valid


def _adaptive_dmp_failures(
    old_primitive: Array,
    candidate_primitive: Array,
    cell_orders: Array,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    gamma: float,
) -> Array:
    """Relax the old-state hull only for a strong converging SRHD wave."""

    bad = spacetime._relaxed_dmp_failures(
        old_primitive,
        candidate_primitive,
        cell_orders,
        periodic,
        parameters,
    )
    scale = np.maximum(np.ptp(old_primitive, axis=0), 1.0)
    for cell in np.flatnonzero(bad & (cell_orders > 2)):
        neighborhood = spacetime._stencil(
            old_primitive, int(cell), (-1, 0, 1), periodic
        )
        left_speeds = np.sort(
            np.real(
                np.linalg.eigvals(
                    primitive_flux_jacobian(neighborhood[0], gamma)
                )
            )
        )
        right_speeds = np.sort(
            np.real(
                np.linalg.eigvals(
                    primitive_flux_jacobian(neighborhood[2], gamma)
                )
            )
        )
        speed_scale = max(
            float(np.max(np.abs(left_speeds))),
            float(np.max(np.abs(right_speeds))),
            1.0e-14,
        )
        compression = float(
            np.max(np.maximum(0.0, left_speeds - right_speeds))
        ) / speed_scale
        transition = DMP_COMPRESSION_SCALE
        compression_power = (compression / transition) ** 4
        relative_tolerance = parameters.dmp_relative_tolerance + (
            STRONG_SHOCK_DMP_TOLERANCE
            - parameters.dmp_relative_tolerance
        ) * compression_power / (1.0 + compression_power)
        lower = np.min(neighborhood, axis=0)
        upper = np.max(neighborhood, axis=0)
        tolerance = np.maximum(
            parameters.dmp_absolute_tolerance * scale,
            relative_tolerance * (upper - lower),
        )
        value = candidate_primitive[cell]
        if np.all(
            (value >= lower - tolerance) & (value <= upper + tolerance)
        ):
            bad[cell] = False
    return bad


def lsdg_predict(
    coefficients: Array,
    centers: Array,
    dt: float,
    dx: float,
    gamma: float,
    iterations: int,
    tolerance: float,
) -> spacetime.LocalPrediction:
    """Solve the primitive SRHD local weak space-time predictor."""

    positive, polynomial_valid = _admissible_polynomial(
        coefficients, centers
    )
    initial = np.einsum(
        "qp,npc->nqc", spacetime.GAUSS_SPACE_EVALUATION, positive
    )
    values = np.repeat(initial[:, :, np.newaxis, :], 5, axis=2)
    residual = np.full(len(values), math.inf)
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
        next_values = np.einsum(
            "lk,nakc->nalc", spacetime.GAUSS_TIME_WEAK_INVERSE, weak_rhs
        )
        residual = np.max(
            np.abs(next_values - values)
            / np.maximum(1.0, np.abs(next_values)),
            axis=(1, 2, 3),
        )
        values = next_values
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
    valid &= residual <= tolerance
    valid &= np.all(np.isfinite(values), axis=(1, 2, 3))
    maximum_speed = 0.0
    for stage in boundary_stages:
        for primitive in stage.reshape(-1, 3):
            if not srhd.primitive_is_admissible(primitive[None, :]):
                continue
            maximum_speed = max(
                maximum_speed,
                *(abs(speed) for speed in srhd.signal_speeds(primitive, gamma)),
            )
    return spacetime.LocalPrediction(
        boundary_stages=boundary_stages,
        time_weights=spacetime.GAUSS_WEIGHTS_01,
        valid_cells=valid,
        maximum_speed=maximum_speed,
    )


def time_integrated_fluxes(
    prediction: spacetime.LocalPrediction,
    gamma: float,
    periodic: bool,
    boundary_primitives: tuple[Array, Array] | None,
) -> tuple[Array, Array, Array]:
    cells = len(prediction.boundary_stages[0])
    fluxes = np.empty((cells + 1, 3))
    valid = np.ones(cells + 1, dtype=bool)
    face_speeds = np.zeros(cells + 1)
    for face in range(cells + 1):
        flux = np.zeros(3)
        for weight, stage in zip(
            prediction.time_weights,
            prediction.boundary_stages,
            strict=True,
        ):
            left, right = spacetime._boundary_traces(
                stage, face, periodic, boundary_primitives
            )
            if not srhd.primitive_is_admissible(np.asarray((left, right))):
                valid[face] = False
                continue
            face_flux = srhd.hllc_flux(left, right, gamma)
            if not np.all(np.isfinite(face_flux)):
                valid[face] = False
                continue
            flux += weight * face_flux
            for primitive in (left, right):
                face_speeds[face] = max(
                    face_speeds[face],
                    *(
                        abs(speed)
                        for speed in srhd.signal_speeds(primitive, gamma)
                    ),
                )
        fluxes[face] = flux
    if periodic:
        fluxes[-1] = fluxes[0]
        valid[-1] = valid[0]
        face_speeds[-1] = face_speeds[0]
    return fluxes, valid, face_speeds


def spacetime_step(
    state: Array,
    dt: float,
    dx: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None = None,
) -> StepResult:
    primitive = srhd.primitive_from_conserved(state, gamma)
    if not srhd.primitive_is_admissible(primitive):
        return StepResult(None, math.inf, np.zeros(len(state), dtype=int))
    reconstruction_primitive, _center_scales = point_primitives(
        state, gamma, periodic, parameters
    )
    high_coefficients = characteristic_cweno_z_coefficients(
        reconstruction_primitive,
        gamma,
        periodic,
        parameters,
        point_samples=True,
    )
    if parameters.anchor_mp5_traces:
        high_coefficients = anchor_mp5_traces(
            high_coefficients,
            reconstruction_primitive,
            gamma,
            periodic,
            parameters,
        )
    coefficient_sets = (
        spacetime.pwc_coefficients(primitive),
        spacetime.pwl_coefficients(primitive, periodic),
        characteristic_mp5_parabolic_coefficients(
            primitive, gamma, periodic
        ),
        high_coefficients,
    )
    center_sets = (
        primitive,
        primitive,
        primitive,
        reconstruction_primitive,
    )
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
    flux_data = [
        time_integrated_fluxes(
            prediction, gamma, periodic, boundary_primitives
        )
        for prediction in predictions
    ]
    physical_overrides = (
        srhd.physical_flux_overrides(
            primitive, dx, dt, gamma, periodic
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
                return StepResult(None, fallback_speed, cell_orders)
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
        maximum_speed = float(np.max(selected_speeds))
        candidate = state - (dt / dx) * (
            selected_fluxes[1:] - selected_fluxes[:-1]
        )
        candidate_primitive = srhd.primitive_from_conserved(candidate, gamma)
        physical = np.asarray(
            [
                srhd.primitive_is_admissible(row[None, :])
                for row in candidate_primitive
            ]
        )
        bad = ~physical
        if np.all(physical):
            bad |= _adaptive_dmp_failures(
                primitive,
                candidate_primitive,
                cell_orders,
                periodic,
                parameters,
                gamma,
            )
        reducible = bad & (cell_orders > 0)
        if not np.any(bad) or not np.any(reducible):
            if np.all(physical):
                return StepResult(
                    candidate,
                    maximum_speed,
                    cell_orders,
                    len(effective_physical_faces),
                )
            return StepResult(None, maximum_speed, cell_orders)
        cell_orders[reducible] -= 1
    return StepResult(None, fallback_speed, cell_orders)


def evolve_spacetime(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    parameters: spacetime.PredictorParameters,
    periodic: bool = True,
    boundary_primitives: tuple[Array, Array] | None = None,
) -> tuple[Array, bool, EvolutionStats]:
    state = srhd.conserved_array(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    stats = EvolutionStats()
    speed_estimate = 0.0
    while time < final_time:
        current = srhd.primitive_from_conserved(state, gamma)
        if not srhd.primitive_is_admissible(current):
            return current, False, stats
        cell_speed = max(
            abs(speed)
            for row in current
            for speed in srhd.signal_speeds(row, gamma)
        )
        dt = min(
            final_time - time,
            cfl * dx / max(cell_speed, speed_estimate),
        )
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
                    else min(0.5 * dt, speed_dt)
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
    final = srhd.primitive_from_conserved(state, gamma)
    return final, srhd.primitive_is_admissible(final), stats


def entropy_cell_averages(
    cells: int,
    time: float,
    gamma: float,
    velocity: float = 0.3,
    amplitude: float = 0.2,
) -> Array:
    centers = (np.arange(cells) + 0.5) / cells
    cell_width = 1.0 / cells
    averaging_factor = math.sin(math.pi * cell_width) / (
        math.pi * cell_width
    )
    density = 1.0 + amplitude * averaging_factor * np.sin(
        2.0 * math.pi * (centers - velocity * time)
    )
    primitive = np.column_stack(
        (
            density,
            np.full(cells, velocity),
            np.ones(cells),
        )
    )
    # With constant velocity and pressure, the Valencia conserved variables
    # are affine in density, so evaluating them at the averaged density is the
    # exact finite-volume cell average.
    state = srhd.conserved_array(primitive, gamma)
    return srhd.primitive_from_conserved(state, gamma)


def _result(
    method: str,
    cells: int,
    cfl: float,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
    stats: EvolutionStats | None,
) -> SmoothResult:
    component_l1 = []
    overshoots = []
    tv_ratios = []
    for component in range(3):
        scale = max(float(np.ptp(exact[:, component])), 1.0)
        component_l1.append(
            float(np.mean(np.abs(evolved[:, component] - exact[:, component])))
            / scale
        )
        overshoots.append(
            max(
                0.0,
                float(np.max(evolved[:, component]) - np.max(exact[:, component])),
                float(np.min(exact[:, component]) - np.min(evolved[:, component])),
            )
            / scale
        )
        exact_tv = spacetime.total_variation(exact[:, component])
        error_tv = spacetime.total_variation(
            evolved[:, component] - exact[:, component]
        )
        tv_ratios.append(
            spacetime.total_variation(evolved[:, component]) / exact_tv
            if exact_tv > 1.0e-12 * scale
            else 1.0 + error_tv / scale
        )
    exact_integral = np.mean(srhd.conserved_array(exact, gamma), axis=0)
    evolved_integral = np.mean(srhd.conserved_array(evolved, gamma), axis=0)
    integral_error = np.abs(evolved_integral - exact_integral)
    return SmoothResult(
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
        cfl_retries=0 if stats is None else stats.cfl_retries,
        mp5_cell_steps=0 if stats is None else stats.mp5_cells,
        pwl_cell_steps=0 if stats is None else stats.pwl_cells,
        pwc_cell_steps=0 if stats is None else stats.pwc_cells,
        physical_front_faces=(
            0 if stats is None else stats.physical_front_faces
        ),
    )


def run_entropy(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    parameters: spacetime.PredictorParameters,
) -> list[SmoothResult]:
    rows: list[SmoothResult] = []
    previous: dict[str, tuple[int, float]] = {}
    for cells in cells_values:
        initial = entropy_cell_averages(cells, 0.0, gamma)
        exact = entropy_cell_averages(cells, 1.0, gamma)
        baseline, baseline_completed = srhd.evolve(
            initial,
            final_time=1.0,
            cfl=cfl,
            gamma=gamma,
            physical_space_time=False,
            periodic=True,
            riemann_flux=srhd.hllc_flux,
        )
        candidate, candidate_completed, stats = evolve_spacetime(
            initial, 1.0, cfl, gamma, parameters, periodic=True
        )
        current = (
            _result(
                "component_mp5_ssprk3_hllc",
                cells,
                cfl,
                baseline,
                exact,
                baseline_completed,
                gamma,
                None,
            ),
            _result(
                "cweno_z_aderprim_lsdg_hllc",
                cells,
                cfl,
                candidate,
                exact,
                candidate_completed,
                gamma,
                stats,
            ),
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


def print_rows(rows: list[SmoothResult]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(SmoothResult))
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
    parser.add_argument("--picard-iterations", type=int, default=8)
    parser.add_argument("--physical-fronts", action="store_true")
    args = parser.parse_args()
    if any(cells < 8 for cells in args.cells):
        raise ValueError("all grids need at least eight cells")
    if args.cfl <= 0.0 or args.gamma <= 1.0:
        raise ValueError("CFL and gamma are outside their domains")
    if args.picard_iterations < 1:
        raise ValueError("--picard-iterations must be positive")
    parameters = spacetime.PredictorParameters(
        predictor_kind="lsdg",
        picard_iterations=args.picard_iterations,
        characteristic_reconstruction=True,
        anchor_mp5_traces=True,
        physical_fronts=args.physical_fronts,
        sensor_relative_activity_floor=1.0e-7,
    )
    print("[cell-averaged SRHD entropy wave]")
    print_rows(
        run_entropy(
            tuple(sorted(set(args.cells))),
            args.cfl,
            args.gamma,
            parameters,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
