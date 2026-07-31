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


def fit_physical_mhd_wave(
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
    if not mhd.primitive_is_admissible(endpoints):
        return None
    normal_b_scale = max(abs(left[5]), abs(right[5]), 1.0)
    if abs(left[5] - right[5]) > 3.0e-10 * normal_b_scale:
        return None
    endpoint_state = mhd.conserved_array(endpoints, gamma)
    average_state = mhd.conserved_array(
        np.asarray(average, dtype=float)[None, :],
        gamma,
    )[0]
    left_state, right_state = endpoint_state
    state_jump = left_state - right_state
    jump_norm2 = float(np.dot(state_jump, state_jump))
    scale = max(
        float(np.dot(left_state, left_state)),
        float(np.dot(right_state, right_state)),
        np.finfo(float).tiny,
    )
    if jump_norm2 <= 256.0 * np.finfo(float).eps**2 * scale:
        return None
    left_fraction = float(
        np.dot(average_state - right_state, state_jump) / jump_norm2
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
        endpoint_tolerance
        < left_fraction
        < 1.0 - endpoint_tolerance
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
    if mixture_residual > mixture_tolerance:
        return None
    left_flux = mhd_flux(left, gamma)
    right_flux = mhd_flux(right, gamma)
    flux_jump = left_flux - right_flux
    speed = float(np.dot(state_jump, flux_jump) / jump_norm2)
    jump_residual = scaled_residual(
        flux_jump - speed * state_jump,
        flux_jump,
        speed * state_jump,
    )
    if not np.isfinite(speed) or jump_residual > jump_tolerance:
        return None
    return PhysicalWave(
        left=np.asarray(left, dtype=float).copy(),
        right=np.asarray(right, dtype=float).copy(),
        speed=speed,
        left_fraction=left_fraction,
        mixture_residual=mixture_residual,
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
    left_state, right_state = mhd.conserved_array(
        np.array((left, right)),
        gamma,
    )
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
        if wave is None:
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
        if wave is None:
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
        if result.method != "persistent_physical_wave_space_time":
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
        ),
        default=(
            "mp5_hlld",
            "physical_wave_space_time",
            "persistent_physical_wave_space_time",
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
            smooth_difference = float(
                np.max(np.abs(smooth_candidate - smooth_baseline))
            )
            if (
                not baseline_completed
                or not candidate_completed
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
