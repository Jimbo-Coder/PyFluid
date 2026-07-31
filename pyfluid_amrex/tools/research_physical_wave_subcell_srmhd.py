#!/usr/bin/env python3
"""Test time-integrated isolated physical waves in flat-spacetime RMHD.

This standalone precursor extends the physical-wave mechanism through
relativistic MHD conserved variables, nonlinear primitive recovery, the
subluminal constraint, and one common normal magnetic field.  Smooth cells use
component MP5 with a common admissibility scale; accepted isolated waves obey
the complete flat-Valencia Rankine--Hugoniot relation.

The implementation is intentionally small and slow.  Primitive recovery uses
a constrained nonlinear solve and the Riemann baseline is a robust HLLE bound.
This is evidence about the local space-time mechanism, not a production RMHD
solver, multidimensional CT result, or general Balsara-tube improvement.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_wave_reconstruction as base  # noqa: E402
from research_physical_wave_subcell import (  # noqa: E402
    PhysicalWave,
    scaled_residual,
)


Array = np.ndarray


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
    normal_b_spread: float
    minimum_rho: float
    minimum_pressure: float
    maximum_speed: float
    mass_error: float
    momentum_error: float
    energy_error: float


def primitive_is_admissible(primitive: Array) -> bool:
    values = np.asarray(primitive)
    velocity2 = np.sum(values[..., 1:4] ** 2, axis=-1)
    return bool(
        np.all(np.isfinite(values))
        and np.all(values[..., 0] > 0.0)
        and np.all(values[..., 4] > 0.0)
        and np.all(velocity2 < 1.0)
    )


def conserved_array(primitive: Array, gamma: float) -> Array:
    values = np.asarray(primitive, dtype=float)
    rho = values[..., 0]
    velocity = values[..., 1:4]
    pressure = values[..., 4]
    magnetic = values[..., 5:8]
    velocity2 = np.sum(velocity * velocity, axis=-1)
    lorentz = 1.0 / np.sqrt(1.0 - velocity2)
    enthalpy = 1.0 + gamma * pressure / ((gamma - 1.0) * rho)
    magnetic2 = np.sum(magnetic * magnetic, axis=-1)
    magnetic_velocity = np.sum(magnetic * velocity, axis=-1)
    comoving2 = magnetic2 / (lorentz * lorentz) + magnetic_velocity**2
    z_value = rho * enthalpy * lorentz * lorentz
    dens = rho * lorentz
    momentum = (
        (z_value + magnetic2)[..., None] * velocity
        - magnetic_velocity[..., None] * magnetic
    )
    tau = (
        z_value
        + magnetic2
        - pressure
        - 0.5 * comoving2
        - dens
    )
    return np.concatenate(
        (
            dens[..., None],
            momentum,
            tau[..., None],
            magnetic,
        ),
        axis=-1,
    )


def flux(primitive: Array, gamma: float) -> Array:
    value = np.asarray(primitive, dtype=float)
    state = conserved_array(value[None, :], gamma)[0]
    rho = value[0]
    velocity = value[1:4]
    pressure = value[4]
    magnetic = value[5:8]
    velocity2 = float(np.dot(velocity, velocity))
    lorentz = 1.0 / math.sqrt(1.0 - velocity2)
    enthalpy = 1.0 + gamma * pressure / ((gamma - 1.0) * rho)
    magnetic2 = float(np.dot(magnetic, magnetic))
    magnetic_velocity = float(np.dot(magnetic, velocity))
    alpha_b0 = lorentz * magnetic_velocity
    comoving = magnetic / lorentz + alpha_b0 * velocity
    comoving2 = magnetic2 / (lorentz * lorentz) + magnetic_velocity**2
    total_pressure = pressure + 0.5 * comoving2
    result = np.empty(8)
    result[0] = state[0] * velocity[0]
    for component in range(3):
        result[1 + component] = (
            state[1 + component] * velocity[0]
            + (total_pressure if component == 0 else 0.0)
            - comoving[component] * magnetic[0] / lorentz
        )
    result[4] = (
        state[4] * velocity[0]
        + total_pressure * velocity[0]
        - alpha_b0 * magnetic[0] / lorentz
    )
    for component in range(3):
        result[5 + component] = (
            magnetic[component] * velocity[0]
            - velocity[component] * magnetic[0]
        )
    return result


def velocity_from_unconstrained(unconstrained: Array) -> Array:
    norm2 = float(np.dot(unconstrained, unconstrained))
    return unconstrained / math.sqrt(1.0 + norm2)


def unconstrained_from_velocity(velocity: Array) -> Array:
    velocity2 = float(np.dot(velocity, velocity))
    return velocity / math.sqrt(max(1.0 - velocity2, 1.0e-14))


def primitive_from_conserved(state: Array, gamma: float) -> Array:
    values = np.asarray(state, dtype=float)
    result = np.full(values.shape, np.nan)
    previous: Array | None = None
    for index in np.ndindex(values.shape[:-1]):
        target = values[index]
        magnetic = target[5:8]
        if (
            not np.all(np.isfinite(target))
            or target[0] <= 0.0
            or not np.all(np.isfinite(magnetic))
        ):
            continue
        if previous is not None and primitive_is_admissible(previous[None, :]):
            guess_primitive = previous
        else:
            energy_scale = max(target[4] + target[0], 1.0e-8)
            velocity_guess = target[1:4] / max(
                energy_scale + float(np.dot(magnetic, magnetic)),
                float(np.linalg.norm(target[1:4])) * 1.05,
                1.0e-8,
            )
            speed = float(np.linalg.norm(velocity_guess))
            if speed >= 0.8:
                velocity_guess *= 0.8 / speed
            guess_primitive = np.concatenate(
                (
                    [max(target[0], 1.0e-8)],
                    velocity_guess,
                    [max((gamma - 1.0) * target[4], 1.0e-6)],
                    magnetic,
                )
            )
        unknown0 = np.concatenate(
            (
                [math.log(guess_primitive[0])],
                unconstrained_from_velocity(guess_primitive[1:4]),
                [math.log(guess_primitive[4])],
            )
        )
        scale = np.maximum(np.abs(target[:5]), 1.0)

        def primitive(unknown: Array) -> Array:
            return np.concatenate(
                (
                    [math.exp(unknown[0])],
                    velocity_from_unconstrained(unknown[1:4]),
                    [math.exp(unknown[4])],
                    magnetic,
                )
            )

        def residual(unknown: Array) -> Array:
            candidate = primitive(unknown)
            predicted = conserved_array(candidate[None, :], gamma)[0]
            return (predicted[:5] - target[:5]) / scale

        solution = least_squares(
            residual,
            unknown0,
            xtol=2.0e-12,
            ftol=2.0e-12,
            gtol=2.0e-12,
            max_nfev=300,
        )
        candidate = primitive(solution.x)
        if (
            solution.success
            and np.max(np.abs(residual(solution.x))) <= 3.0e-9
            and primitive_is_admissible(candidate[None, :])
        ):
            result[index] = candidate
            previous = candidate
    return result


def fast_speed_bound(primitive: Array, gamma: float) -> float:
    rho = float(primitive[0])
    pressure = float(primitive[4])
    velocity = primitive[1:4]
    magnetic = primitive[5:8]
    velocity2 = float(np.dot(velocity, velocity))
    lorentz = 1.0 / math.sqrt(1.0 - velocity2)
    enthalpy = 1.0 + gamma * pressure / ((gamma - 1.0) * rho)
    sound2 = gamma * pressure / (rho * enthalpy)
    magnetic2 = float(np.dot(magnetic, magnetic))
    magnetic_velocity = float(np.dot(magnetic, velocity))
    comoving2 = magnetic2 / (lorentz * lorentz) + magnetic_velocity**2
    alfven2 = comoving2 / (rho * enthalpy + comoving2)
    fast2 = min(sound2 + alfven2 - sound2 * alfven2, 1.0)
    return math.sqrt(max(fast2, 0.0))


def signal_speeds(primitive: Array, gamma: float) -> tuple[float, float]:
    velocity = float(primitive[1])
    speed = fast_speed_bound(primitive, gamma)
    return (
        (velocity - speed) / (1.0 - velocity * speed),
        (velocity + speed) / (1.0 + velocity * speed),
    )


def hlle_flux(left: Array, right: Array, gamma: float) -> Array:
    left_state, right_state = conserved_array(
        np.array((left, right)),
        gamma,
    )
    left_flux = flux(left, gamma)
    right_flux = flux(right, gamma)
    left_speeds = signal_speeds(left, gamma)
    right_speeds = signal_speeds(right, gamma)
    speed_left = min(0.0, left_speeds[0], right_speeds[0])
    speed_right = max(0.0, left_speeds[1], right_speeds[1])
    if speed_left >= 0.0:
        return left_flux
    if speed_right <= 0.0:
        return right_flux
    return (
        speed_right * left_flux
        - speed_left * right_flux
        + speed_left * speed_right * (right_state - left_state)
    ) / (speed_right - speed_left)


def fit_physical_wave(
    average: Array,
    left: Array,
    right: Array,
    gamma: float,
    *,
    mixture_tolerance: float = 5.0e-9,
    jump_tolerance: float = 5.0e-9,
    allow_endpoint: bool = False,
) -> PhysicalWave | None:
    endpoints = np.array((left, right), dtype=float)
    if not primitive_is_admissible(endpoints):
        return None
    normal_scale = max(abs(left[5]), abs(right[5]), 1.0)
    if abs(left[5] - right[5]) > 5.0e-9 * normal_scale:
        return None
    left_state, right_state = conserved_array(endpoints, gamma)
    average_state = conserved_array(np.asarray(average)[None, :], gamma)[0]
    state_jump = left_state - right_state
    jump_norm2 = float(np.dot(state_jump, state_jump))
    scale = max(
        float(np.dot(left_state, left_state)),
        float(np.dot(right_state, right_state)),
        np.finfo(float).tiny,
    )
    if jump_norm2 <= 256.0 * np.finfo(float).eps**2 * scale:
        return None
    fraction = float(
        np.dot(average_state - right_state, state_jump) / jump_norm2
    )
    endpoint_tolerance = 1024.0 * np.finfo(float).eps
    if allow_endpoint:
        if not -endpoint_tolerance <= fraction <= 1.0 + endpoint_tolerance:
            return None
    elif not endpoint_tolerance < fraction < 1.0 - endpoint_tolerance:
        return None
    fraction = min(max(fraction, 0.0), 1.0)
    fitted_average = fraction * left_state + (1.0 - fraction) * right_state
    mixture_residual = scaled_residual(
        average_state - fitted_average,
        average_state,
        fitted_average,
        state_jump,
    )
    if mixture_residual > mixture_tolerance:
        return None
    flux_jump = flux(left, gamma) - flux(right, gamma)
    speed = float(np.dot(state_jump, flux_jump) / jump_norm2)
    jump_residual = scaled_residual(
        flux_jump - speed * state_jump,
        flux_jump,
        speed * state_jump,
    )
    if not np.isfinite(speed) or jump_residual > jump_tolerance:
        return None
    return PhysicalWave(
        left=np.asarray(left).copy(),
        right=np.asarray(right).copy(),
        speed=speed,
        left_fraction=fraction,
        mixture_residual=mixture_residual,
        jump_residual=jump_residual,
    )


def local_primitive(primitive: Array, cell: int, periodic: bool) -> Array:
    return np.asarray(base.stencil_value(primitive, cell, periodic))


def states_agree(left: Array, right: Array, gamma: float) -> bool:
    left_state, right_state = conserved_array(np.array((left, right)), gamma)
    return scaled_residual(
        left_state - right_state,
        left_state,
        right_state,
    ) <= 5.0e-9


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
    return fit_physical_wave(average, left, right, gamma)


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
    return fit_physical_wave(
        left,
        left,
        right,
        gamma,
        allow_endpoint=True,
    )


def component_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    periodic = bool(component_mp5.periodic)
    stencil = np.array(
        [
            local_primitive(primitive, left_cell + offset, periodic)
            for offset in range(-2, 4)
        ]
    )
    left = np.empty(8)
    right = np.empty(8)
    for component in range(8):
        left[component], right[component] = base.reconstruct_scalar_pair(
            stencil[:, component],
            base.mp5_left,
        )
    left[5] = right[5] = 0.5 * (stencil[2, 5] + stencil[3, 5])
    left_center = stencil[2]
    right_center = stencil[3]

    def feasible(weight: float) -> bool:
        pair = np.array(
            (
                left_center + weight * (left - left_center),
                right_center + weight * (right - right_center),
            )
        )
        return primitive_is_admissible(pair)

    weight = 1.0
    if not feasible(weight):
        lower = 0.0
        upper = 1.0
        for _ in range(52):
            middle = 0.5 * (lower + upper)
            if feasible(middle):
                lower = middle
            else:
                upper = middle
        weight = lower
    return (
        left_center + weight * (left - left_center),
        right_center + weight * (right - right_center),
    )


component_mp5.periodic = False


def physical_wave_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    periodic = bool(physical_wave_mp5.periodic)
    left_wave = fit_cell_wave(primitive, left_cell, gamma, periodic)
    right_wave = fit_cell_wave(primitive, left_cell + 1, gamma, periodic)
    if left_wave is not None and right_wave is None:
        return left_wave.right.copy(), left_wave.right.copy()
    if right_wave is not None and left_wave is None:
        return right_wave.left.copy(), right_wave.left.copy()
    if left_wave is not None and right_wave is not None:
        if states_agree(left_wave.right, right_wave.left, gamma):
            common = 0.5 * (left_wave.right + right_wave.left)
            common[5] = left_wave.right[5]
            return common.copy(), common.copy()
        return component_mp5(primitive, left_cell, gamma)
    face_wave = fit_face_wave(primitive, left_cell, gamma, periodic)
    if face_wave is not None:
        upwind = face_wave.left if face_wave.speed >= 0.0 else face_wave.right
        return upwind.copy(), upwind.copy()
    component_mp5.periodic = periodic
    return component_mp5(primitive, left_cell, gamma)


physical_wave_mp5.periodic = False


def rhs_with_fluxes(
    state: Array,
    dx: float,
    gamma: float,
    reconstruct,
    periodic: bool,
) -> tuple[Array, float, Array]:
    primitive = primitive_from_conserved(state, gamma)
    if not primitive_is_admissible(primitive):
        return (
            np.full_like(state, np.nan),
            math.inf,
            np.full((len(state) + 1, 8), np.nan),
        )
    reconstruct.periodic = periodic
    fluxes = np.empty((len(state) + 1, 8))
    required_speed = 0.0
    for face in range(len(state) + 1):
        left, right = reconstruct(primitive, face - 1, gamma)
        if not primitive_is_admissible(np.array((left, right))):
            return np.full_like(state, np.nan), math.inf, fluxes
        fluxes[face] = hlle_flux(left, right, gamma)
        required_speed = max(
            required_speed,
            *(abs(speed) for speed in signal_speeds(left, gamma)),
            *(abs(speed) for speed in signal_speeds(right, gamma)),
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

    def add(face: int, value: Array) -> None:
        resolved = face_index(face)
        if resolved is not None:
            overrides[resolved] = value.copy()

    def add_plateau(face: int, state: Array, value: Array) -> None:
        if states_agree(
            local_primitive(primitive, face - 1, periodic),
            state,
            gamma,
        ) and states_agree(
            local_primitive(primitive, face, periodic),
            state,
            gamma,
        ):
            add(face, value)

    fitted_cells: set[int] = set()
    for cell in range(cells):
        wave = fit_cell_wave(primitive, cell, gamma, periodic)
        if wave is None:
            continue
        fitted_cells.add(cell)
        left_flux = flux(wave.left, gamma)
        right_flux = flux(wave.right, gamma)
        left_average = left_flux
        right_average = right_flux
        if wave.speed > 0.0:
            crossing = (1.0 - wave.left_fraction) * dx / wave.speed
            if crossing < dt:
                crossed = min(max(1.0 - crossing / dt, 0.0), 1.0)
                right_average = (
                    (1.0 - crossed) * right_flux + crossed * left_flux
                )
        elif wave.speed < 0.0:
            crossing = wave.left_fraction * dx / (-wave.speed)
            if crossing < dt:
                crossed = min(max(1.0 - crossing / dt, 0.0), 1.0)
                left_average = (
                    (1.0 - crossed) * left_flux + crossed * right_flux
                )
        add(cell, left_average)
        add(cell + 1, right_average)
        for offset in range(1, 4):
            add_plateau(cell - offset, wave.left, left_flux)
            add_plateau(cell + 1 + offset, wave.right, right_flux)
    for left_cell in range(-1 if not periodic else 0, cells):
        if (
            left_cell % cells in fitted_cells
            or (left_cell + 1) % cells in fitted_cells
        ):
            continue
        wave = fit_face_wave(primitive, left_cell, gamma, periodic)
        if wave is None:
            continue
        upwind = wave.left if wave.speed >= 0.0 else wave.right
        add(left_cell + 1, flux(upwind, gamma))
        for offset in range(1, 4):
            add_plateau(
                left_cell + 1 - offset,
                wave.left,
                flux(wave.left, gamma),
            )
            add_plateau(
                left_cell + 1 + offset,
                wave.right,
                flux(wave.right, gamma),
            )
    if periodic and 0 in overrides:
        overrides[cells] = overrides[0].copy()
    return overrides


def evolve(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    *,
    physical_space_time: bool,
    periodic: bool,
) -> tuple[Array, bool]:
    state = conserved_array(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    reconstruct = physical_wave_mp5 if physical_space_time else component_mp5
    while time < final_time:
        current = primitive_from_conserved(state, gamma)
        if not primitive_is_admissible(current):
            return current, False
        rhs0, speed0, flux0 = rhs_with_fluxes(
            state,
            dx,
            gamma,
            reconstruct,
            periodic,
        )
        if not np.isfinite(speed0) or not np.all(np.isfinite(rhs0)):
            return current, False
        dt = min(final_time - time, cfl * dx / speed0)
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1, flux1 = rhs_with_fluxes(
                stage1,
                dx,
                gamma,
                reconstruct,
                periodic,
            )
            if not np.isfinite(speed1):
                dt *= 0.5
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, speed2, flux2 = rhs_with_fluxes(
                stage2,
                dx,
                gamma,
                reconstruct,
                periodic,
            )
            if not np.isfinite(speed2):
                dt *= 0.5
                continue
            reduced = min(
                dt,
                final_time - time,
                cfl * dx / max(speed0, speed1, speed2),
            )
            if reduced < dt * (1.0 - 1.0e-13):
                dt = 0.99 * reduced
                continue
            integrated_flux = (
                flux0 / 6.0 + flux1 / 6.0 + 2.0 * flux2 / 3.0
            )
            if physical_space_time:
                for face, face_flux in physical_flux_overrides(
                    current,
                    dx,
                    dt,
                    gamma,
                    periodic,
                ).items():
                    integrated_flux[face] = face_flux
            candidate = state - dt * (
                integrated_flux[1:] - integrated_flux[:-1]
            ) / dx
            candidate_primitive = primitive_from_conserved(candidate, gamma)
            if not primitive_is_admissible(candidate_primitive):
                dt *= 0.5
                continue
            state = candidate
            break
        else:
            return primitive_from_conserved(state, gamma), False
        time += dt
    final = primitive_from_conserved(state, gamma)
    return final, primitive_is_admissible(final)


def oblique_shock_states(
    gamma: float,
    shock_speed: float = 0.9,
) -> tuple[Array, Array, float]:
    upstream = np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, 0.0))
    upstream_state = conserved_array(upstream[None, :], gamma)[0]
    upstream_flux = flux(upstream, gamma)
    active = np.array((0, 1, 2, 3, 4, 6, 7))
    scale = np.maximum(
        np.abs(upstream_flux - shock_speed * upstream_state),
        1.0,
    )

    def primitive(unknown: Array) -> Array:
        return np.concatenate(
            (
                [math.exp(unknown[0])],
                velocity_from_unconstrained(unknown[1:4]),
                [math.exp(unknown[4]), upstream[5], unknown[5], unknown[6]],
            )
        )

    def residual(unknown: Array) -> Array:
        downstream = primitive(unknown)
        jump = (
            flux(downstream, gamma)
            - upstream_flux
            - shock_speed
            * (conserved_array(downstream[None, :], gamma)[0] - upstream_state)
        ) / scale
        return jump[active]

    initial = np.concatenate(
        (
            [math.log(2.75)],
            unconstrained_from_velocity(np.array((0.65, -0.03, 0.0))),
            [math.log(6.0), 0.7, 0.0],
        )
    )
    solution = least_squares(
        residual,
        initial,
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=8000,
    )
    downstream = primitive(solution.x)
    if (
        not solution.success
        or np.max(np.abs(residual(solution.x))) > 2.0e-10
        or np.linalg.norm(downstream - upstream) < 1.0e-4
    ):
        raise RuntimeError("failed to construct the oblique RMHD shock")
    fitted = fit_physical_wave(
        downstream,
        downstream,
        upstream,
        gamma,
        allow_endpoint=True,
    )
    if fitted is None:
        raise RuntimeError("constructed RMHD shock fails its wave contract")
    return downstream, upstream, shock_speed


def discontinuity_averages(
    cells: int,
    position: float,
    left: Array,
    right: Array,
    gamma: float,
) -> Array:
    left_state, right_state = conserved_array(
        np.array((left, right)),
        gamma,
    )
    edges = np.linspace(0.0, 1.0, cells + 1)
    averages = np.empty((cells, 8))
    for cell, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        fraction = min(max((position - lower) / (upper - lower), 0.0), 1.0)
        averages[cell] = fraction * left_state + (1.0 - fraction) * right_state
    return primitive_from_conserved(averages, gamma)


def result(
    *,
    case: str,
    method: str,
    cells: int,
    cfl: float,
    gamma: float,
    left: Array,
    right: Array,
    speed: float,
    final_time: float,
    physical_space_time: bool,
) -> Result:
    initial_position = 0.217
    initial = discontinuity_averages(
        cells,
        initial_position,
        left,
        right,
        gamma,
    )
    evolved, completed = evolve(
        initial,
        final_time,
        cfl,
        gamma,
        physical_space_time=physical_space_time,
        periodic=False,
    )
    exact = discontinuity_averages(
        cells,
        initial_position + speed * final_time,
        left,
        right,
        gamma,
    )
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
        exact_tv = base.total_variation(exact_component)
        evolved_tv = base.total_variation(evolved_component)
        tv_ratios.append(
            evolved_tv / exact_tv
            if exact_tv > 1.0e-12 * scale
            else evolved_tv / scale
        )
    endpoint_scale = max(float(np.linalg.norm(left - right)), 1.0e-14)
    endpoint_distance = np.minimum(
        np.linalg.norm(evolved - left, axis=1),
        np.linalg.norm(evolved - right, axis=1),
    ) / endpoint_scale
    exact_integrals = np.mean(conserved_array(exact, gamma), axis=0)
    final_integrals = np.mean(conserved_array(evolved, gamma), axis=0)
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
        transition_cells=int(np.count_nonzero(endpoint_distance > 3.0e-7)),
        normal_b_spread=float(np.ptp(evolved[:, 5])),
        minimum_rho=float(np.nanmin(evolved[:, 0])),
        minimum_pressure=float(np.nanmin(evolved[:, 4])),
        maximum_speed=float(
            np.nanmax(np.linalg.norm(evolved[:, 1:4], axis=1))
        ),
        mass_error=float(integral_error[0]),
        momentum_error=float(np.max(integral_error[1:4])),
        energy_error=float(integral_error[4]),
    )


def print_results(results: list[Result]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(Result))
    print("\t".join(fields))
    for item in results:
        values = []
        for field in fields:
            value = getattr(item, field)
            if isinstance(value, float):
                values.append(f"{value:.12e}")
            else:
                values.append(str(value))
        print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(8,))
    parser.add_argument("--cfl", type=float, nargs="+", default=(0.9,))
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    args = parser.parse_args()

    contact_left = np.array(
        (2.0, 0.5, 0.1, -0.1, 1.0, 0.5, 0.3, -0.2)
    )
    contact_right = contact_left.copy()
    contact_right[0] = 1.0
    shock_left, shock_right, shock_speed = oblique_shock_states(args.gamma)
    results: list[Result] = []
    for cells in args.cells:
        for cfl in args.cfl:
            for method, physical_space_time in (
                ("component_mp5_hlle", False),
                ("physical_wave_space_time", True),
            ):
                results.append(
                    result(
                        case="moving_rmhd_contact",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=args.gamma,
                        left=contact_left,
                        right=contact_right,
                        speed=0.5,
                        final_time=0.6,
                        physical_space_time=physical_space_time,
                    )
                )
                results.append(
                    result(
                        case="moving_oblique_rmhd_shock",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=args.gamma,
                        left=shock_left,
                        right=shock_right,
                        speed=shock_speed,
                        final_time=0.6,
                        physical_space_time=physical_space_time,
                    )
                )
            x = (np.arange(cells) + 0.5) / cells
            smooth_initial = np.column_stack(
                (
                    1.0 + 0.1 * np.sin(2.0 * math.pi * x),
                    np.full(cells, 0.3),
                    np.full(cells, 0.05),
                    np.zeros(cells),
                    np.ones(cells),
                    np.full(cells, 0.5),
                    np.full(cells, 0.2),
                    np.full(cells, -0.1),
                )
            )
            smooth_baseline, baseline_completed = evolve(
                smooth_initial,
                final_time=1.0,
                cfl=cfl,
                gamma=args.gamma,
                physical_space_time=False,
                periodic=True,
            )
            smooth_candidate, candidate_completed = evolve(
                smooth_initial,
                final_time=1.0,
                cfl=cfl,
                gamma=args.gamma,
                physical_space_time=True,
                periodic=True,
            )
            smooth_difference = float(
                np.max(np.abs(smooth_candidate - smooth_baseline))
            )
            if (
                not baseline_completed
                or not candidate_completed
                or smooth_difference > 3.0e-10
            ):
                raise RuntimeError(
                    "physical-wave path changed the smooth SRMHD control: "
                    f"{smooth_difference:.12e}"
                )
            print(
                "smooth_srmhd_entropy"
                f"\tcells={cells}"
                f"\tcfl={cfl:g}"
                f"\tmaximum_difference={smooth_difference:.12e}"
            )
    print_results(results)


if __name__ == "__main__":
    main()
