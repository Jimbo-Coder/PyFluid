#!/usr/bin/env python3
"""Test a coherent physical-wave reconstruction outside production.

This research harness tests two concrete precursors of the WCEM design.  At
each Euler face it projects a six-cell primitive stencil into a locally
linearized acoustic/contact basis.  Either one common coefficient or one
coefficient per physical wave blends the traces from limited PWL toward MP5:

    w_side = w_side,pwl + theta (w_side,mp5 - w_side,pwl).

The largest ``theta`` keeps each selected reconstructed wave jump aligned with
and no larger than its neighboring cell-average jump.  The common form
preserves the relative wave correction; the wavewise form avoids limiting a
contact because one acoustic family is troubled.  A nonlinear-wave form
constrains only the genuinely nonlinear acoustic families and retains MP5 on
the linearly degenerate contact.  Limited PWL reaches PWC automatically where
its MC slopes vanish.  A final whole-state scaling toward the adjacent cell
average enforces positive density and pressure.

This is a falsification harness, not a production method or a novelty claim.
It compares smooth entropy-wave transport and exact Euler shock tubes against
component MP5, characteristic MP5, PWL, and PPM4 controls.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyfluid.exact_solutions import (  # noqa: E402
    sod_conserved_integrals,
    sod_exact_solution,
    sod_wave_breakpoints,
)
from exact_reference import cell_average_newtonian_hydro  # noqa: E402
from research_constrained_riemann import (  # noqa: E402
    entropy_repaired_hydro_flux,
    hierarchical_hydro_flux,
    hydro_hllc_flux,
)
from research_modal_reconstruction import (  # noqa: E402
    interval_weight,
    mc_slope,
    mp5_left,
    mp5_left_with_bounds,
    parabolic_right_interface,
    ppm4_left,
)


Array = np.ndarray
FaceReconstruction = Callable[[Array, int, float], tuple[Array, Array]]
RiemannFlux = Callable[
    [Array, Array, float],
    Array | tuple[Array, float],
]


@dataclasses.dataclass(frozen=True)
class ShockCase:
    name: str
    left: tuple[float, float, float]
    right: tuple[float, float, float]
    final_time: float


@dataclasses.dataclass(frozen=True)
class Result:
    case: str
    method: str
    cells: int
    completed: bool
    total_l1: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    mass_error: float
    momentum_error: float
    energy_error: float
    minimum_rho: float
    minimum_pressure: float


SHOCK_CASES = (
    ShockCase("sod", (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2),
    ShockCase("lax", (0.445, 0.698, 3.528), (0.5, 0.0, 0.571), 0.13),
    ShockCase("blast", (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.012),
)


def conserved_from_primitive(primitive: Array, gamma: float) -> Array:
    rho = primitive[..., 0]
    velocity = primitive[..., 1]
    pressure = primitive[..., 2]
    return np.stack(
        (
            rho,
            rho * velocity,
            pressure / (gamma - 1.0) + 0.5 * rho * velocity * velocity,
        ),
        axis=-1,
    )


def primitive_from_conserved(state: Array, gamma: float) -> Array:
    rho = state[..., 0]
    velocity = state[..., 1] / rho
    pressure = (gamma - 1.0) * (
        state[..., 2] - 0.5 * state[..., 1] * velocity
    )
    return np.stack((rho, velocity, pressure), axis=-1)


def primitive_is_admissible(primitive: Array) -> bool:
    return bool(
        np.all(np.isfinite(primitive))
        and np.all(primitive[..., 0] > 0.0)
        and np.all(primitive[..., 2] > 0.0)
    )


def stencil_value(
    primitive: Array,
    cell: int,
    periodic: bool,
) -> tuple[Array, float]:
    if periodic:
        return primitive[cell % len(primitive)]
    return primitive[min(len(primitive) - 1, max(0, cell))]


def reconstruct_scalar_pair(
    values: Array,
    method: Callable[..., float],
) -> tuple[float, float]:
    left = method(*values)
    right = method(*values[::-1])
    return left, right


def component_reconstruction(
    method: Callable[..., float],
) -> FaceReconstruction:
    def reconstruct(
        primitive: Array,
        left_cell: int,
        _gamma: float,
    ) -> tuple[Array, Array]:
        periodic = bool(reconstruct.periodic)
        stencil = np.array(
            [
                stencil_value(primitive, left_cell + offset, periodic)
                for offset in range(-2, 4)
            ]
        )
        left = np.empty(3)
        right = np.empty(3)
        for component in range(3):
            left[component], right[component] = reconstruct_scalar_pair(
                stencil[:, component],
                method,
            )
        return positive_trace(left, stencil[2]), positive_trace(
            right,
            stencil[3],
        )

    reconstruct.periodic = False
    return reconstruct


def characteristic_basis(
    left: Array,
    right: Array,
    gamma: float,
) -> tuple[float, float]:
    rho = 0.5 * (left[0] + right[0])
    pressure = 0.5 * (left[2] + right[2])
    sound = math.sqrt(gamma * pressure / rho)
    return rho, sound


def to_characteristics(
    primitive: Array,
    rho_ref: float,
    sound: float,
) -> Array:
    rho, velocity, pressure = np.moveaxis(primitive, -1, 0)
    pressure_mode = pressure / (sound * sound)
    velocity_mode = rho_ref * velocity / sound
    return np.stack(
        (
            rho - pressure_mode,
            0.5 * (pressure_mode - velocity_mode),
            0.5 * (pressure_mode + velocity_mode),
        ),
        axis=-1,
    )


def from_characteristics(
    waves: Array,
    rho_ref: float,
    sound: float,
) -> Array:
    entropy, sound_minus, sound_plus = waves
    return np.array(
        [
            entropy + sound_minus + sound_plus,
            sound * (sound_plus - sound_minus) / rho_ref,
            sound * sound * (sound_minus + sound_plus),
        ]
    )


def positive_trace_weight(trace: Array, cell: Array) -> float:
    theta = 1.0
    for component in (0, 2):
        floor = 1.0e-13 * max(1.0, cell[component])
        correction = trace[component] - cell[component]
        if trace[component] < floor and correction < 0.0:
            theta = min(
                theta,
                (cell[component] - floor) / (-correction),
            )
    return max(0.0, min(1.0, theta))


def positive_trace(trace: Array, cell: Array) -> Array:
    theta = positive_trace_weight(trace, cell)
    return cell + theta * (trace - cell)


def hierarchical_weights(
    constraints: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """Project the full high-order target onto the feasible mode polygon."""
    def feasible(candidate: tuple[float, float]) -> bool:
        theta_lower, theta_higher = candidate
        return all(
            a * theta_lower + b * theta_higher
            <= c + 128.0 * np.finfo(float).eps * max(1.0, abs(c))
            for a, b, c in constraints
        )

    if feasible((1.0, 1.0)):
        return 1.0, 1.0

    candidates = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    for a, b, c in constraints:
        norm2 = a * a + b * b
        if norm2 > 0.0:
            distance_factor = (a + b - c) / norm2
            candidates.append(
                (
                    1.0 - distance_factor * a,
                    1.0 - distance_factor * b,
                )
            )
    for first_index, first in enumerate(constraints):
        a0, b0, c0 = first
        for a1, b1, c1 in constraints[first_index + 1 :]:
            determinant = a0 * b1 - a1 * b0
            if abs(determinant) <= 1.0e-14 * max(
                1.0,
                abs(a0),
                abs(b0),
                abs(a1),
                abs(b1),
            ):
                continue
            theta_lower = (c0 * b1 - c1 * b0) / determinant
            theta_higher = (a0 * c1 - a1 * c0) / determinant
            candidates.append((theta_lower, theta_higher))

    feasible_candidates = [candidate for candidate in candidates if feasible(candidate)]
    if not feasible_candidates:
        return 0.0, 0.0
    best = min(
        feasible_candidates,
        key=lambda candidate: (
            (1.0 - candidate[0]) ** 2 + (1.0 - candidate[1]) ** 2,
            -candidate[0],
            -candidate[1],
        ),
    )
    return (
        max(0.0, min(1.0, best[0])),
        max(0.0, min(1.0, best[1])),
    )


def hierarchical_trace_weights(
    constraints: list[tuple[float, float, float]],
    trace_corrections: Array,
) -> tuple[float, float]:
    """Project the full target in the left/right scalar wave-trace metric."""
    target = np.ones(2)

    def constraint_tolerance(
        constraint: tuple[float, float, float],
    ) -> float:
        a, b, c = constraint
        return 256.0 * np.finfo(float).eps * max(
            1.0,
            abs(a),
            abs(b),
            abs(c),
        )

    def feasible(candidate: Array) -> bool:
        return all(
            a * candidate[0] + b * candidate[1]
            <= c + constraint_tolerance((a, b, c))
            for a, b, c in constraints
        )

    if feasible(target):
        return 1.0, 1.0

    vertices = [
        np.array((0.0, 0.0)),
        np.array((1.0, 0.0)),
        np.array((1.0, 1.0)),
    ]
    for first_index, first in enumerate(constraints):
        a0, b0, c0 = first
        for a1, b1, c1 in constraints[first_index + 1 :]:
            determinant = a0 * b1 - a1 * b0
            if abs(determinant) <= 1.0e-14 * max(
                1.0,
                abs(a0),
                abs(b0),
                abs(a1),
                abs(b1),
            ):
                continue
            vertices.append(
                np.array(
                    (
                        (c0 * b1 - c1 * b0) / determinant,
                        (a0 * c1 - a1 * c0) / determinant,
                    )
                )
            )
    feasible_vertices = [
        vertex for vertex in vertices if feasible(vertex)
    ]
    if not feasible_vertices:
        return 0.0, 0.0

    candidates = list(feasible_vertices)
    for constraint in constraints:
        a, b, c = constraint
        tolerance = constraint_tolerance(constraint)
        edge_vertices = [
            vertex
            for vertex in feasible_vertices
            if abs(a * vertex[0] + b * vertex[1] - c) <= tolerance
        ]
        if len(edge_vertices) < 2:
            continue
        endpoints = max(
            (
                (first, second)
                for first_index, first in enumerate(edge_vertices)
                for second in edge_vertices[first_index + 1 :]
            ),
            key=lambda pair: float(np.dot(pair[1] - pair[0], pair[1] - pair[0])),
        )
        start, end = endpoints
        direction = end - start
        direction_norm2 = float(np.dot(direction, direction))
        if direction_norm2 <= np.finfo(float).eps:
            continue
        metric_direction = trace_corrections @ direction
        denominator = float(np.dot(metric_direction, metric_direction))
        if denominator > np.finfo(float).eps * max(
            1.0,
            direction_norm2,
            float(np.sum(trace_corrections * trace_corrections)),
        ):
            metric_offset = trace_corrections @ (start - target)
            fraction = -float(
                np.dot(metric_direction, metric_offset)
            ) / denominator
        else:
            fraction = -float(
                np.dot(direction, start - target)
            ) / direction_norm2
        fraction = max(0.0, min(1.0, fraction))
        candidates.append(start + fraction * direction)

    def objective(candidate: Array) -> tuple[float, float, float, float]:
        difference = candidate - target
        trace_error = trace_corrections @ difference
        return (
            float(np.dot(trace_error, trace_error)),
            float(np.dot(difference, difference)),
            -float(candidate[0]),
            -float(candidate[1]),
        )

    best = min(candidates, key=objective)
    return (
        max(0.0, min(1.0, float(best[0]))),
        max(0.0, min(1.0, float(best[1]))),
    )


def append_interval_constraints(
    constraints: list[tuple[float, float, float]],
    base: float,
    lower_correction: float,
    higher_correction: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> None:
    constraints.append(
        (
            lower_correction,
            higher_correction,
            upper - base + tolerance,
        )
    )
    constraints.append(
        (
            -lower_correction,
            -higher_correction,
            base - lower + tolerance,
        )
    )


def hierarchical_scalar_pair(
    values: Array,
    objective: str = "coefficient",
) -> tuple[float, float, float, float]:
    """Reconstruct one physical wave with nested PWL/parabolic/MP5 modes."""
    robust_left = values[2] + 0.5 * mc_slope(
        values[1],
        values[2],
        values[3],
    )
    robust_right = values[3] - 0.5 * mc_slope(
        values[2],
        values[3],
        values[4],
    )
    cell_jump = values[3] - values[2]
    jump_lower = min(0.0, cell_jump)
    jump_upper = max(0.0, cell_jump)
    robust_jump = robust_right - robust_left
    projected_jump = min(jump_upper, max(jump_lower, robust_jump))
    jump_change = projected_jump - robust_jump
    robust_left -= 0.5 * jump_change
    robust_right += 0.5 * jump_change

    parabolic = parabolic_right_interface(
        values[1],
        values[2],
        values[3],
        values[4],
    )
    high_left, left_lower, left_upper = mp5_left_with_bounds(*values)
    high_right, right_lower, right_upper = mp5_left_with_bounds(
        *values[::-1]
    )
    left_lower = min(left_lower, robust_left)
    left_upper = max(left_upper, robust_left)
    right_lower = min(right_lower, robust_right)
    right_upper = max(right_upper, robust_right)

    left_lower_correction = parabolic - robust_left
    right_lower_correction = parabolic - robust_right
    left_higher_correction = high_left - parabolic
    right_higher_correction = high_right - parabolic
    jump_lower_correction = right_lower_correction - left_lower_correction
    jump_higher_correction = right_higher_correction - left_higher_correction
    scale = max(1.0, *(abs(value) for value in values))
    tolerance = 128.0 * np.finfo(float).eps * scale
    constraints: list[tuple[float, float, float]] = [
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (0.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
    ]
    append_interval_constraints(
        constraints,
        robust_left,
        left_lower_correction,
        left_higher_correction,
        left_lower,
        left_upper,
        tolerance,
    )
    append_interval_constraints(
        constraints,
        robust_right,
        right_lower_correction,
        right_higher_correction,
        right_lower,
        right_upper,
        tolerance,
    )
    append_interval_constraints(
        constraints,
        projected_jump,
        jump_lower_correction,
        jump_higher_correction,
        jump_lower,
        jump_upper,
        tolerance,
    )
    if objective == "coefficient":
        theta_lower, theta_higher = hierarchical_weights(constraints)
    elif objective in ("trace", "balanced"):
        trace_corrections = np.array(
            (
                (left_lower_correction, left_higher_correction),
                (right_lower_correction, right_higher_correction),
            )
        )
        if objective == "balanced":
            trace_norm = float(
                np.sqrt(np.sum(trace_corrections * trace_corrections))
            )
            if trace_norm > np.finfo(float).eps:
                # Match the mean eigenvalue of the two-dimensional trace
                # metric to the coefficient metric.  This keeps the combined
                # objective invariant to a uniform rescaling of the wave.
                trace_corrections = np.vstack(
                    (
                        np.eye(2),
                        math.sqrt(2.0) * trace_corrections / trace_norm,
                    )
                )
            else:
                trace_corrections = np.eye(2)
        theta_lower, theta_higher = hierarchical_trace_weights(
            constraints,
            trace_corrections,
        )
    else:
        raise ValueError(f"unknown hierarchy objective {objective!r}")
    accepted_left = (
        robust_left
        + theta_lower * left_lower_correction
        + theta_higher * left_higher_correction
    )
    accepted_right = (
        robust_right
        + theta_lower * right_lower_correction
        + theta_higher * right_higher_correction
    )
    return accepted_left, accepted_right, theta_lower, theta_higher


def hierarchical_wave_reconstruction(
    primitive: Array,
    left_cell: int,
    gamma: float,
    limit_contact: bool,
    objective: str,
    periodic: bool,
) -> tuple[Array, Array]:
    stencil = np.array(
        [
            stencil_value(primitive, left_cell + offset, periodic)
            for offset in range(-2, 4)
        ]
    )
    rho_ref, sound = characteristic_basis(stencil[2], stencil[3], gamma)
    waves = to_characteristics(stencil, rho_ref, sound)
    accepted_left = np.empty(3)
    accepted_right = np.empty(3)

    for wave in range(3):
        values = waves[:, wave]
        if wave == 0 and not limit_contact:
            accepted_left[wave] = mp5_left(*values)
            accepted_right[wave] = mp5_left(*values[::-1])
            continue
        accepted_left[wave], accepted_right[wave], _, _ = (
            hierarchical_scalar_pair(values, objective)
        )

    left_trace = from_characteristics(accepted_left, rho_ref, sound)
    right_trace = from_characteristics(accepted_right, rho_ref, sound)
    theta = min(
        positive_trace_weight(left_trace, stencil[2]),
        positive_trace_weight(right_trace, stencil[3]),
    )

    constrained_waves = (0, 1, 2) if limit_contact else (1, 2)
    cell_jump = waves[3] - waves[2]

    def physical_pair(weight: float) -> tuple[Array, Array]:
        if weight <= 0.0:
            return stencil[2].copy(), stencil[3].copy()
        return (
            stencil[2] + weight * (left_trace - stencil[2]),
            stencil[3] + weight * (right_trace - stencil[3]),
        )

    def pair_feasible(weight: float) -> bool:
        left, right = physical_pair(weight)
        if (
            not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or left[0] <= 0.0
            or left[2] <= 0.0
            or right[0] <= 0.0
            or right[2] <= 0.0
        ):
            return False
        trace_waves = to_characteristics(
            np.array((left, right)),
            rho_ref,
            sound,
        )
        trace_jump = trace_waves[1] - trace_waves[0]
        for wave in constrained_waves:
            lower = min(0.0, cell_jump[wave])
            upper = max(0.0, cell_jump[wave])
            tolerance = 128.0 * np.finfo(float).eps * max(
                1.0,
                abs(lower),
                abs(upper),
                abs(trace_jump[wave]),
            )
            if not (
                lower - tolerance
                <= trace_jump[wave]
                <= upper + tolerance
            ):
                return False
        return True

    if not pair_feasible(theta):
        lower = 0.0
        upper = theta
        for _ in range(52):
            middle = 0.5 * (lower + upper)
            if pair_feasible(middle):
                lower = middle
            else:
                upper = middle
        theta = lower
    return physical_pair(theta)


def hierarchical_wave_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return hierarchical_wave_reconstruction(
        primitive,
        left_cell,
        gamma,
        limit_contact=True,
        objective="coefficient",
        periodic=bool(hierarchical_wave_mp5.periodic),
    )


hierarchical_wave_mp5.periodic = False


def hierarchical_nonlinear_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return hierarchical_wave_reconstruction(
        primitive,
        left_cell,
        gamma,
        limit_contact=False,
        objective="coefficient",
        periodic=bool(hierarchical_nonlinear_mp5.periodic),
    )


hierarchical_nonlinear_mp5.periodic = False


def trace_metric_nonlinear_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return hierarchical_wave_reconstruction(
        primitive,
        left_cell,
        gamma,
        limit_contact=False,
        objective="trace",
        periodic=bool(trace_metric_nonlinear_mp5.periodic),
    )


trace_metric_nonlinear_mp5.periodic = False


def balanced_metric_nonlinear_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return hierarchical_wave_reconstruction(
        primitive,
        left_cell,
        gamma,
        limit_contact=False,
        objective="balanced",
        periodic=bool(balanced_metric_nonlinear_mp5.periodic),
    )


balanced_metric_nonlinear_mp5.periodic = False


def characteristic_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    periodic = bool(characteristic_mp5.periodic)
    stencil = np.array(
        [
            stencil_value(primitive, left_cell + offset, periodic)
            for offset in range(-2, 4)
        ]
    )
    rho_ref, sound = characteristic_basis(stencil[2], stencil[3], gamma)
    waves = to_characteristics(stencil, rho_ref, sound)
    left_waves = np.empty(3)
    right_waves = np.empty(3)
    for wave in range(3):
        left_waves[wave], right_waves[wave] = reconstruct_scalar_pair(
            waves[:, wave],
            mp5_left,
        )
    return positive_trace(
        from_characteristics(left_waves, rho_ref, sound),
        stencil[2],
    ), positive_trace(
        from_characteristics(right_waves, rho_ref, sound),
        stencil[3],
    )


characteristic_mp5.periodic = False


def wave_limited_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
    weight_policy: str,
    periodic: bool,
) -> tuple[Array, Array]:
    stencil = np.array(
        [
            stencil_value(primitive, left_cell + offset, periodic)
            for offset in range(-2, 4)
        ]
    )
    rho_ref, sound = characteristic_basis(stencil[2], stencil[3], gamma)
    waves = to_characteristics(stencil, rho_ref, sound)
    robust_left = np.empty(3)
    robust_right = np.empty(3)
    high_left = np.empty(3)
    high_right = np.empty(3)
    for wave in range(3):
        values = waves[:, wave]
        robust_left[wave] = values[2] + 0.5 * mc_slope(
            values[1],
            values[2],
            values[3],
        )
        robust_right[wave] = values[3] - 0.5 * mc_slope(
            values[2],
            values[3],
            values[4],
        )
        high_left[wave], high_right[wave] = reconstruct_scalar_pair(
            values,
            mp5_left,
        )

    theta = np.ones(3)
    cell_jumps = waves[3] - waves[2]
    for wave in range(3):
        cell_jump = cell_jumps[wave]
        robust_jump = robust_right[wave] - robust_left[wave]
        high_jump = high_right[wave] - high_left[wave]
        lower = min(0.0, cell_jump)
        upper = max(0.0, cell_jump)
        scale = max(
            1.0,
            abs(waves[2, wave]),
            abs(waves[3, wave]),
        )
        tolerance = 64.0 * np.finfo(float).eps * scale
        projected_robust_jump = min(
            upper,
            max(lower, robust_jump),
        )
        jump_correction = projected_robust_jump - robust_jump
        robust_left[wave] -= 0.5 * jump_correction
        robust_right[wave] += 0.5 * jump_correction
        robust_jump = projected_robust_jump
        theta[wave] = interval_weight(
            theta[wave],
            robust_jump,
            high_jump - robust_jump,
            lower,
            upper,
            tolerance,
        )
    if weight_policy == "shared":
        theta.fill(float(np.min(theta)))
    elif weight_policy == "acoustic":
        acoustic_theta = min(theta[1], theta[2])
        theta[1:] = acoustic_theta
    elif weight_policy == "nonlinear":
        theta[0] = 1.0
    elif weight_policy == "compressive":
        theta[0] = 1.0
        if stencil[2, 1] <= stencil[3, 1]:
            theta[1:] = 1.0
    elif weight_policy == "genuine_compression":
        theta[0] = 1.0
        if cell_jumps[1] <= 0.0:
            theta[1] = 1.0
        if cell_jumps[2] >= 0.0:
            theta[2] = 1.0
    elif weight_policy != "wavewise":
        raise ValueError(f"unknown weight policy {weight_policy!r}")

    left_waves = robust_left + theta * (high_left - robust_left)
    right_waves = robust_right + theta * (high_right - robust_right)
    return positive_trace(
        from_characteristics(left_waves, rho_ref, sound),
        stencil[2],
    ), positive_trace(
        from_characteristics(right_waves, rho_ref, sound),
        stencil[3],
    )


def coherent_wave_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return wave_limited_mp5(
        primitive,
        left_cell,
        gamma,
        weight_policy="shared",
        periodic=bool(coherent_wave_mp5.periodic),
    )


coherent_wave_mp5.periodic = False


def wavewise_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return wave_limited_mp5(
        primitive,
        left_cell,
        gamma,
        weight_policy="wavewise",
        periodic=bool(wavewise_mp5.periodic),
    )


wavewise_mp5.periodic = False


def acoustic_grouped_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return wave_limited_mp5(
        primitive,
        left_cell,
        gamma,
        weight_policy="acoustic",
        periodic=bool(acoustic_grouped_mp5.periodic),
    )


acoustic_grouped_mp5.periodic = False


def nonlinear_wave_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return wave_limited_mp5(
        primitive,
        left_cell,
        gamma,
        weight_policy="nonlinear",
        periodic=bool(nonlinear_wave_mp5.periodic),
    )


nonlinear_wave_mp5.periodic = False


def compressive_wave_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return wave_limited_mp5(
        primitive,
        left_cell,
        gamma,
        weight_policy="compressive",
        periodic=bool(compressive_wave_mp5.periodic),
    )


compressive_wave_mp5.periodic = False


def genuine_compression_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return wave_limited_mp5(
        primitive,
        left_cell,
        gamma,
        weight_policy="genuine_compression",
        periodic=bool(genuine_compression_mp5.periodic),
    )


genuine_compression_mp5.periodic = False


def hllc_flux(left: Array, right: Array, gamma: float) -> Array:
    left_full = np.array([left[0], left[1], 0.0, 0.0, left[2]])
    right_full = np.array([right[0], right[1], 0.0, 0.0, right[2]])
    full_flux = hydro_hllc_flux(left_full, right_full, gamma)
    return full_flux[[0, 1, 4]]


def hierarchical_flux(
    left: Array,
    right: Array,
    gamma: float,
    shock_budget_scale: float,
) -> Array:
    left_full = np.array([left[0], left[1], 0.0, 0.0, left[2]])
    right_full = np.array([right[0], right[1], 0.0, 0.0, right[2]])
    result = hierarchical_hydro_flux(
        left_full,
        right_full,
        gamma,
        shock_budget_scale,
    )
    return result.flux[[0, 1, 4]]


def entropy_repair_flux(
    left: Array,
    right: Array,
    gamma: float,
) -> tuple[Array, float]:
    left_full = np.array([left[0], left[1], 0.0, 0.0, left[2]])
    right_full = np.array([right[0], right[1], 0.0, 0.0, right[2]])
    result = entropy_repaired_hydro_flux(left_full, right_full, gamma)
    return result.flux[[0, 1, 4]], result.safe_speed


def rhs(
    state: Array,
    dx: float,
    gamma: float,
    reconstruct: FaceReconstruction,
    riemann_flux: RiemannFlux,
    periodic: bool,
) -> tuple[Array, float]:
    primitive = primitive_from_conserved(state, gamma)
    if not primitive_is_admissible(primitive):
        return np.full_like(state, np.nan), math.inf
    reconstruct.periodic = periodic
    fluxes = np.empty((len(state) + 1, 3))
    required_speed = 0.0
    for face in range(len(state) + 1):
        left, right = reconstruct(primitive, face - 1, gamma)
        if not primitive_is_admissible(np.array((left, right))):
            return np.full_like(state, np.nan), math.inf
        endpoint_speeds = (
            abs(left[1]) + math.sqrt(gamma * left[2] / left[0]),
            abs(right[1]) + math.sqrt(gamma * right[2] / right[0]),
        )
        if any(
            not np.isfinite(speed) or speed < 0.0
            for speed in endpoint_speeds
        ):
            return np.full_like(state, np.nan), math.inf
        face_speed = max(endpoint_speeds)
        evaluation = riemann_flux(left, right, gamma)
        if isinstance(evaluation, tuple):
            face_flux, flux_speed = evaluation
            if not np.isfinite(flux_speed) or flux_speed < 0.0:
                return np.full_like(state, np.nan), math.inf
            face_speed = max(face_speed, float(flux_speed))
        else:
            face_flux = evaluation
        if not np.all(np.isfinite(face_flux)):
            return np.full_like(state, np.nan), math.inf
        fluxes[face] = face_flux
        required_speed = max(required_speed, face_speed)
    if periodic:
        fluxes[-1] = fluxes[0]
    return -(fluxes[1:] - fluxes[:-1]) / dx, required_speed


def evolve(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    reconstruct: FaceReconstruction,
    riemann_flux: RiemannFlux,
    periodic: bool,
) -> tuple[Array, bool]:
    state = conserved_from_primitive(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    while time < final_time:
        current = primitive_from_conserved(state, gamma)
        if not primitive_is_admissible(current):
            return current, False
        speeds = np.abs(current[:, 1]) + np.sqrt(
            gamma * current[:, 2] / current[:, 0]
        )
        rhs0, required_speed = rhs(
            state,
            dx,
            gamma,
            reconstruct,
            riemann_flux,
            periodic,
        )
        if (
            not np.isfinite(required_speed)
            or not np.all(np.isfinite(rhs0))
        ):
            return current, False
        step_speed = max(float(np.max(speeds)), required_speed)
        dt = min(final_time - time, cfl * dx / step_speed)
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, stage1_speed = rhs(
                stage1,
                dx,
                gamma,
                reconstruct,
                riemann_flux,
                periodic,
            )
            if not np.isfinite(stage1_speed):
                dt *= 0.5
                continue
            reduced_dt = min(
                dt,
                final_time - time,
                cfl * dx / max(step_speed, stage1_speed),
            )
            if reduced_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * reduced_dt
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, stage2_speed = rhs(
                stage2,
                dx,
                gamma,
                reconstruct,
                riemann_flux,
                periodic,
            )
            if not np.isfinite(stage2_speed):
                dt *= 0.5
                continue
            reduced_dt = min(
                dt,
                final_time - time,
                cfl * dx / max(step_speed, stage1_speed, stage2_speed),
            )
            if reduced_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * reduced_dt
                continue
            candidate = (1.0 / 3.0) * state + (2.0 / 3.0) * (
                stage2 + dt * rhs2
            )
            if not np.all(np.isfinite(candidate)):
                dt *= 0.5
                continue
            candidate_primitive = primitive_from_conserved(candidate, gamma)
            if not primitive_is_admissible(candidate_primitive):
                dt *= 0.5
                continue
            state = candidate
            break
        else:
            return primitive_from_conserved(state, gamma), False
        if not np.all(np.isfinite(state)):
            return primitive_from_conserved(state, gamma), False
        time += dt
    final = primitive_from_conserved(state, gamma)
    return final, primitive_is_admissible(final)


def normalized_l1(values: Array, exact: Array) -> float:
    scale = max(float(np.max(exact) - np.min(exact)), 1.0e-14)
    return float(np.mean(np.abs(values - exact)) / scale)


def total_variation(values: Array) -> float:
    return float(np.sum(np.abs(np.diff(values))))


def shock_result(
    case: ShockCase,
    method: str,
    cells: int,
    evolved: Array,
    completed: bool,
    gamma: float,
) -> Result:
    cell_edges = np.linspace(0.0, 1.0, cells + 1)
    rho_exact, pressure_exact, velocity_exact = (
        cell_average_newtonian_hydro(
            lambda points: sod_exact_solution(
                points,
                case.final_time,
                gamma,
                case.left,
                case.right,
            ),
            cell_edges,
            gamma,
            breakpoints=sod_wave_breakpoints(
                case.final_time,
                gamma,
                case.left,
                case.right,
            ),
        )
    )
    exact_values = np.column_stack(
        (rho_exact, velocity_exact, pressure_exact),
    )
    errors = [
        normalized_l1(evolved[:, component], exact_values[:, component])
        for component in range(3)
    ]
    overshoots = []
    tv_ratios = []
    for component in range(3):
        exact = exact_values[:, component]
        values = evolved[:, component]
        scale = max(float(np.max(exact) - np.min(exact)), 1.0e-14)
        overshoots.append(
            max(
                0.0,
                float(np.max(values) - np.max(exact)),
                float(np.min(exact) - np.min(values)),
            )
            / scale
        )
        tv_ratios.append(
            total_variation(values)
            / max(total_variation(exact), 1.0e-14)
        )
    exact_integrals = sod_conserved_integrals(
        case.final_time,
        gamma,
        case.left,
        case.right,
    )
    state = conserved_from_primitive(evolved, gamma)
    integrals = np.mean(state, axis=0)
    return Result(
        case=case.name,
        method=method,
        cells=cells,
        completed=completed,
        total_l1=sum(errors),
        maximum_overshoot=max(overshoots),
        maximum_tv_ratio=max(tv_ratios),
        mass_error=abs(integrals[0] - exact_integrals["mass"]),
        momentum_error=abs(integrals[1] - exact_integrals["momx"]),
        energy_error=abs(integrals[2] - exact_integrals["energy"]),
        minimum_rho=float(np.nanmin(evolved[:, 0])),
        minimum_pressure=float(np.nanmin(evolved[:, 2])),
    )


def entropy_wave_result(
    method: str,
    cells: int,
    cfl: float,
    gamma: float,
    reconstruct: FaceReconstruction,
    riemann_flux: RiemannFlux,
) -> Result:
    x = (np.arange(cells) + 0.5) / cells
    primitive = np.column_stack(
        (
            1.0 + 0.2 * np.sin(2.0 * math.pi * x),
            np.full(cells, 0.5),
            np.ones(cells),
        )
    )
    evolved, completed = evolve(
        primitive,
        final_time=2.0,
        cfl=cfl,
        gamma=gamma,
        reconstruct=reconstruct,
        riemann_flux=riemann_flux,
        periodic=True,
    )
    density_l1 = float(np.mean(np.abs(evolved[:, 0] - primitive[:, 0])))
    return Result(
        case="entropy_wave",
        method=method,
        cells=cells,
        completed=completed,
        total_l1=density_l1,
        maximum_overshoot=max(
            0.0,
            float(np.max(evolved[:, 0]) - 1.2),
            float(0.8 - np.min(evolved[:, 0])),
        ),
        maximum_tv_ratio=total_variation(evolved[:, 0])
        / total_variation(primitive[:, 0]),
        mass_error=abs(float(np.mean(evolved[:, 0])) - 1.0),
        momentum_error=math.nan,
        energy_error=math.nan,
        minimum_rho=float(np.min(evolved[:, 0])),
        minimum_pressure=float(np.min(evolved[:, 2])),
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


def write_results(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(field.name for field in dataclasses.fields(Result))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for result in results:
            writer.writerow(dataclasses.asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(128, 256))
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("entropy_wave", "sod", "lax", "blast"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
    )
    parser.add_argument(
        "--riemann",
        nargs="+",
        choices=(
            "hllc",
            "hierarchical0",
            "hierarchical1",
            "entropy_repair",
        ),
        default=("hllc",),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cfl <= 0.0:
        parser.error("--cfl must be positive")
    if "entropy_repair" in args.riemann and args.cfl > 0.45:
        parser.error(
            "entropy_repair is locally audited only through CFL 0.45"
        )

    known_cases = {"entropy_wave", *(case.name for case in SHOCK_CASES)}
    unknown = sorted(set(args.cases) - known_cases)
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")

    methods: dict[str, FaceReconstruction] = {
        "pwl": component_reconstruction(
            lambda vmm, vm, v0, vp, vpp, vppp: (
                v0 + 0.5 * mc_slope(vm, v0, vp)
            )
        ),
        "ppm4": component_reconstruction(ppm4_left),
        "mp5_component": component_reconstruction(mp5_left),
        "mp5_characteristic": characteristic_mp5,
        "coherent_wave_mp5": coherent_wave_mp5,
        "wavewise_mp5": wavewise_mp5,
        "acoustic_grouped_mp5": acoustic_grouped_mp5,
        "nonlinear_wave_mp5": nonlinear_wave_mp5,
        "compressive_wave_mp5": compressive_wave_mp5,
        "genuine_compression_mp5": genuine_compression_mp5,
        "hierarchical_wave_mp5": hierarchical_wave_mp5,
        "hierarchical_nonlinear_mp5": hierarchical_nonlinear_mp5,
        "trace_metric_nonlinear_mp5": trace_metric_nonlinear_mp5,
        "balanced_metric_nonlinear_mp5": balanced_metric_nonlinear_mp5,
    }
    if args.methods != ["all"]:
        unknown_methods = sorted(set(args.methods) - set(methods))
        if unknown_methods:
            parser.error(f"unknown methods: {', '.join(unknown_methods)}")
        methods = {name: methods[name] for name in args.methods}

    results: list[Result] = []
    riemann_methods: dict[str, RiemannFlux] = {
        "hllc": hllc_flux,
        "hierarchical0": (
            lambda left, right, gamma: hierarchical_flux(
                left,
                right,
                gamma,
                0.0,
            )
        ),
        "hierarchical1": (
            lambda left, right, gamma: hierarchical_flux(
                left,
                right,
                gamma,
                1.0,
            )
        ),
        "entropy_repair": entropy_repair_flux,
    }
    riemann_methods = {
        name: riemann_methods[name] for name in args.riemann
    }
    for cells in args.cells:
        for method, reconstruct in methods.items():
            for riemann_name, riemann_flux in riemann_methods.items():
                result_name = (
                    method
                    if riemann_name == "hllc"
                    else f"{method}+{riemann_name}"
                )
                if "entropy_wave" in args.cases:
                    results.append(
                        entropy_wave_result(
                            result_name,
                            cells,
                            args.cfl,
                            args.gamma,
                            reconstruct,
                            riemann_flux,
                        )
                    )
                for case in SHOCK_CASES:
                    if case.name not in args.cases:
                        continue
                    x = (np.arange(cells) + 0.5) / cells
                    primitive = np.empty((cells, 3))
                    primitive[x < 0.5] = case.left
                    primitive[x >= 0.5] = case.right
                    evolved, completed = evolve(
                        primitive,
                        case.final_time,
                        args.cfl,
                        args.gamma,
                        reconstruct,
                        riemann_flux,
                        periodic=False,
                    )
                    results.append(
                        shock_result(
                            case,
                            result_name,
                            cells,
                            evolved,
                            completed,
                            args.gamma,
                        )
                    )
    print_results(results)
    if args.output is not None:
        write_results(args.output, results)
        print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
