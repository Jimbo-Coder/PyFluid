#!/usr/bin/env python3
"""Test time-integrated physical waves in one-dimensional SRHD.

This research precursor carries the Newtonian in-cell mechanism across the
first relativistic boundary.  Smooth cells use component MP5 with one common
admissibility scale.  An isolated discontinuity is accepted only when one
subcell fraction reproduces the Valencia conserved average and one speed
satisfies every relativistic Rankine--Hugoniot equation.

The method integrates the accepted physical flux through a face over the exact
crossing time.  It is not yet a characteristic SRHD reconstruction, a general
shock-tube method, or an SRMHD/GRMHD result.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_wave_reconstruction as base  # noqa: E402
from research_physical_wave_subcell import (  # noqa: E402
    PhysicalWave,
    scaled_residual,
)


Array = np.ndarray
component_mp5 = base.component_reconstruction(base.mp5_left)


@dataclasses.dataclass(frozen=True)
class Result:
    case: str
    method: str
    cells: int
    cfl: float
    completed: bool
    total_l1: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    transition_cells: int
    minimum_rho: float
    minimum_pressure: float
    maximum_speed: float
    mass_error: float
    momentum_error: float
    energy_error: float


def primitive_is_admissible(primitive: Array) -> bool:
    values = np.asarray(primitive)
    return bool(
        np.all(np.isfinite(values))
        and np.all(values[..., 0] > 0.0)
        and np.all(values[..., 2] > 0.0)
        and np.all(np.abs(values[..., 1]) < 1.0)
    )


def conserved_array(primitive: Array, gamma: float) -> Array:
    values = np.asarray(primitive, dtype=float)
    rho = values[..., 0]
    velocity = values[..., 1]
    pressure = values[..., 2]
    lorentz = 1.0 / np.sqrt(1.0 - velocity * velocity)
    enthalpy = 1.0 + gamma * pressure / ((gamma - 1.0) * rho)
    dens = rho * lorentz
    momentum = rho * enthalpy * lorentz * lorentz * velocity
    tau = rho * enthalpy * lorentz * lorentz - pressure - dens
    return np.stack((dens, momentum, tau), axis=-1)


def primitive_from_conserved(state: Array, gamma: float) -> Array:
    values = np.asarray(state, dtype=float)
    result = np.full(values.shape, np.nan)
    for index in np.ndindex(values.shape[:-1]):
        dens, momentum, tau = values[index]
        energy = tau + dens
        if (
            not np.isfinite(dens + momentum + tau)
            or dens <= 0.0
            or energy <= abs(momentum)
        ):
            continue

        def pressure_residual(pressure: float) -> float:
            total = energy + pressure
            velocity = momentum / total
            if abs(velocity) >= 1.0:
                return -math.inf
            lorentz = 1.0 / math.sqrt(1.0 - velocity * velocity)
            rho = dens / lorentz
            enthalpy = total / (dens * lorentz)
            eos_pressure = (
                (gamma - 1.0) * rho * (enthalpy - 1.0) / gamma
            )
            return pressure - eos_pressure

        lower = max(
            0.0,
            abs(momentum) - energy + 16.0 * np.finfo(float).eps,
        )
        upper = max(energy, abs(momentum), 1.0)
        lower_residual = pressure_residual(lower)
        upper_residual = pressure_residual(upper)
        for _ in range(80):
            if (
                np.isfinite(lower_residual)
                and np.isfinite(upper_residual)
                and lower_residual * upper_residual <= 0.0
            ):
                break
            upper *= 2.0
            upper_residual = pressure_residual(upper)
        else:
            continue
        try:
            pressure = brentq(
                pressure_residual,
                lower,
                upper,
                xtol=2.0e-14,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=200,
            )
        except ValueError:
            continue
        total = energy + pressure
        velocity = momentum / total
        lorentz = 1.0 / math.sqrt(1.0 - velocity * velocity)
        rho = dens / lorentz
        result[index] = (rho, velocity, pressure)
    return result


def flux(primitive: Array, gamma: float) -> Array:
    state = conserved_array(np.asarray(primitive)[None, :], gamma)[0]
    velocity = float(primitive[1])
    pressure = float(primitive[2])
    return np.array(
        (
            state[0] * velocity,
            state[1] * velocity + pressure,
            state[1] - state[0] * velocity,
        )
    )


def signal_speeds(primitive: Array, gamma: float) -> tuple[float, float]:
    rho, velocity, pressure = primitive
    enthalpy = 1.0 + gamma * pressure / ((gamma - 1.0) * rho)
    sound2 = gamma * pressure / (rho * enthalpy)
    sound = math.sqrt(min(max(sound2, 0.0), 1.0))
    return (
        (velocity - sound) / (1.0 - velocity * sound),
        (velocity + sound) / (1.0 + velocity * sound),
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


def hllc_flux(left: Array, right: Array, gamma: float) -> Array:
    """Return the Mignone--Bodo SRHD HLLC flux, with HLLE fallback."""

    primitive_pair = np.asarray((left, right), dtype=float)
    if not primitive_is_admissible(primitive_pair):
        return np.full(3, np.nan)
    conserved = conserved_array(primitive_pair, gamma)
    physical_flux = np.asarray(
        (flux(left, gamma), flux(right, gamma))
    )
    # The HLLC derivation uses total energy E=tau+D and its flux S.
    state = conserved.copy()
    state[:, 2] += state[:, 0]
    energy_flux = physical_flux.copy()
    energy_flux[:, 2] += energy_flux[:, 0]
    left_speeds = signal_speeds(left, gamma)
    right_speeds = signal_speeds(right, gamma)
    speed_left = min(left_speeds[0], right_speeds[0])
    speed_right = max(left_speeds[1], right_speeds[1])
    if 0.0 <= speed_left:
        return physical_flux[0]
    if speed_right <= 0.0:
        return physical_flux[1]
    inverse_width = 1.0 / (speed_right - speed_left)
    hll_state = (
        speed_right * state[1]
        - speed_left * state[0]
        + energy_flux[0]
        - energy_flux[1]
    ) * inverse_width
    hll_flux = (
        speed_right * energy_flux[0]
        - speed_left * energy_flux[1]
        + speed_left * speed_right * (state[1] - state[0])
    ) * inverse_width
    quadratic = float(hll_flux[2])
    linear = float(-hll_flux[1] - hll_state[2])
    constant = float(hll_state[1])
    coefficient_scale = max(
        abs(quadratic), abs(linear), abs(constant), 1.0
    )
    if abs(quadratic) <= 1.0e-14 * coefficient_scale:
        contact_speed = (
            -constant / linear
            if abs(linear) > 1.0e-14 * coefficient_scale
            else 0.0
        )
    else:
        discriminant = linear * linear - 4.0 * quadratic * constant
        if not np.isfinite(discriminant) or discriminant < 0.0:
            return hlle_flux(left, right, gamma)
        contact_speed = (
            -linear - math.sqrt(discriminant)
        ) / (2.0 * quadratic)
    if (
        not np.isfinite(contact_speed)
        or not speed_left < contact_speed < speed_right
        or abs(contact_speed) >= 1.0
    ):
        return hlle_flux(left, right, gamma)
    pressure_star = hll_flux[1] - hll_flux[2] * contact_speed
    if not np.isfinite(pressure_star) or pressure_star <= 0.0:
        return hlle_flux(left, right, gamma)

    def star_flux(side: int, wave_speed: float) -> tuple[Array, Array] | None:
        denominator = wave_speed - contact_speed
        if abs(denominator) <= 1.0e-14:
            return None
        primitive = primitive_pair[side]
        star = np.empty(3)
        star[0] = (
            (wave_speed - primitive[1])
            * state[side, 0]
            / denominator
        )
        star[2] = (
            wave_speed * state[side, 2]
            - state[side, 1]
            + pressure_star * contact_speed
        ) / denominator
        star[1] = (star[2] + pressure_star) * contact_speed
        flux_star = energy_flux[side] + wave_speed * (
            star - state[side]
        )
        if (
            not np.all(np.isfinite(star))
            or not np.all(np.isfinite(flux_star))
            or star[0] <= 0.0
            or star[2] <= 0.0
        ):
            return None
        return star, flux_star

    left_star = star_flux(0, speed_left)
    right_star = star_flux(1, speed_right)
    if left_star is None or right_star is None:
        return hlle_flux(left, right, gamma)
    selected = left_star[1] if contact_speed >= 0.0 else right_star[1]
    return np.asarray((selected[0], selected[1], selected[2] - selected[0]))


def fit_physical_wave(
    average: Array,
    left: Array,
    right: Array,
    gamma: float,
    *,
    mixture_tolerance: float = 3.0e-10,
    jump_tolerance: float = 3.0e-10,
    allow_endpoint: bool = False,
) -> PhysicalWave | None:
    endpoints = np.array((left, right), dtype=float)
    if not primitive_is_admissible(endpoints):
        return None
    left_state, right_state = conserved_array(endpoints, gamma)
    average_state = conserved_array(
        np.asarray(average)[None, :],
        gamma,
    )[0]
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
    endpoint_tolerance = 256.0 * np.finfo(float).eps
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
    ) <= 3.0e-10


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


def admissible_pair_scale(
    trace: Array,
    center: Array,
) -> float:
    if primitive_is_admissible(trace[None, :]):
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(52):
        middle = 0.5 * (lower + upper)
        candidate = center + middle * (trace - center)
        if primitive_is_admissible(candidate[None, :]):
            lower = middle
        else:
            upper = middle
    return lower


def smooth_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    component_mp5.periodic = bool(smooth_mp5.periodic)
    left, right = component_mp5(primitive, left_cell, gamma)
    left_center = local_primitive(
        primitive,
        left_cell,
        bool(smooth_mp5.periodic),
    )
    right_center = local_primitive(
        primitive,
        left_cell + 1,
        bool(smooth_mp5.periodic),
    )
    weight = min(
        admissible_pair_scale(left, left_center),
        admissible_pair_scale(right, right_center),
    )
    return (
        left_center + weight * (left - left_center),
        right_center + weight * (right - right_center),
    )


smooth_mp5.periodic = False


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
            return common.copy(), common.copy()
        return smooth_mp5(primitive, left_cell, gamma)
    face_wave = fit_face_wave(primitive, left_cell, gamma, periodic)
    if face_wave is not None:
        upwind = face_wave.left if face_wave.speed >= 0.0 else face_wave.right
        return upwind.copy(), upwind.copy()
    smooth_mp5.periodic = periodic
    return smooth_mp5(primitive, left_cell, gamma)


physical_wave_mp5.periodic = False


def rhs_with_fluxes(
    state: Array,
    dx: float,
    gamma: float,
    reconstruct,
    periodic: bool,
    riemann_flux=hlle_flux,
) -> tuple[Array, float, Array]:
    primitive = primitive_from_conserved(state, gamma)
    if not primitive_is_admissible(primitive):
        return (
            np.full_like(state, np.nan),
            math.inf,
            np.full((len(state) + 1, 3), np.nan),
        )
    reconstruct.periodic = periodic
    fluxes = np.empty((len(state) + 1, 3))
    required_speed = 0.0
    for face in range(len(state) + 1):
        left, right = reconstruct(primitive, face - 1, gamma)
        if not primitive_is_admissible(np.array((left, right))):
            return np.full_like(state, np.nan), math.inf, fluxes
        fluxes[face] = riemann_flux(left, right, gamma)
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
    riemann_flux=hlle_flux,
) -> tuple[Array, bool]:
    state = conserved_array(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    reconstruct = physical_wave_mp5 if physical_space_time else smooth_mp5
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
            riemann_flux,
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
            riemann_flux,
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
            riemann_flux,
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


def relativistic_shock_states(
    gamma: float,
    shock_speed: float = 0.9,
) -> tuple[Array, Array, float]:
    upstream = np.array((1.0, 0.0, 1.0))
    upstream_state = conserved_array(upstream[None, :], gamma)[0]
    upstream_flux = flux(upstream, gamma)
    scale = np.maximum(
        np.abs(upstream_flux - shock_speed * upstream_state),
        1.0,
    )

    def primitive(unknown: Array) -> Array:
        return np.array(
            (
                math.exp(unknown[0]),
                math.tanh(unknown[1]),
                math.exp(unknown[2]),
            )
        )

    def residual(unknown: Array) -> Array:
        downstream = primitive(unknown)
        return (
            flux(downstream, gamma)
            - upstream_flux
            - shock_speed
            * (conserved_array(downstream[None, :], gamma)[0] - upstream_state)
        ) / scale

    solution = least_squares(
        residual,
        np.array((math.log(5.0), math.atanh(0.7), math.log(10.0))),
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=5000,
    )
    downstream = primitive(solution.x)
    if (
        not solution.success
        or np.max(np.abs(residual(solution.x))) > 2.0e-11
        or np.linalg.norm(downstream - upstream) < 1.0e-4
    ):
        raise RuntimeError("failed to construct the relativistic shock")
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
    averages = np.empty((cells, 3))
    for cell, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        fraction = min(max((position - lower) / (upper - lower), 0.0), 1.0)
        averages[cell] = fraction * left_state + (1.0 - fraction) * right_state
    return primitive_from_conserved(averages, gamma)


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
    for component in range(3):
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
        maximum_overshoot=max(overshoots),
        maximum_tv_ratio=max(tv_ratios),
        transition_cells=int(np.count_nonzero(endpoint_distance > 3.0e-8)),
        minimum_rho=float(np.min(evolved[:, 0])),
        minimum_pressure=float(np.min(evolved[:, 2])),
        maximum_speed=float(np.max(np.abs(evolved[:, 1]))),
        mass_error=float(integral_error[0]),
        momentum_error=float(integral_error[1]),
        energy_error=float(integral_error[2]),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(16,))
    parser.add_argument("--cfl", type=float, nargs="+", default=(0.9,))
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    args = parser.parse_args()

    contact_left = np.array((2.0, 0.5, 1.0))
    contact_right = np.array((1.0, 0.5, 1.0))
    shock_left, shock_right, shock_speed = relativistic_shock_states(
        args.gamma
    )
    results: list[Result] = []
    for cells in args.cells:
        for cfl in args.cfl:
            for method, physical_space_time in (
                ("component_mp5_hlle", False),
                ("physical_wave_space_time", True),
            ):
                results.append(
                    discontinuity_result(
                        case="moving_relativistic_contact",
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
                    discontinuity_result(
                        case="moving_relativistic_shock",
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
                    1.0 + 0.2 * np.sin(2.0 * math.pi * x),
                    np.full(cells, 0.3),
                    np.ones(cells),
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
                    "physical-wave path changed the smooth SRHD control: "
                    f"{smooth_difference:.12e}"
                )
            print(
                "smooth_srhd_entropy"
                f"\tcells={cells}"
                f"\tcfl={cfl:g}"
                f"\tmaximum_difference={smooth_difference:.12e}"
            )
    print_results(results)


if __name__ == "__main__":
    main()
