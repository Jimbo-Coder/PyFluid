#!/usr/bin/env python3
"""Probe conservative in-cell physical waves for Newtonian Euler flow.

This standalone research harness is deliberately narrower than a production
reconstruction method.  It asks whether a cell average that lies on one
Rankine--Hugoniot segment between two locally constant states can be represented
by that physical discontinuity instead of by a polynomial.

Characteristic MP5 remains the default.  A physical-wave override is accepted
only when

* the two cells on either side are locally constant,
* one common subcell fraction reproduces every conserved component,
* the endpoint states are admissible, and
* one speed satisfies the complete Rankine--Hugoniot relation.

The spatial-only ablation supplies the same physical state on both sides of
each bounding face, so the existing HLLC flux sees no artificial face Riemann
problem.  The complete candidate additionally replaces the SSPRK3-integrated
flux at affected faces with the exact plateau-flux average over a subcell
crossing.  Comparing the two isolates the need for a space-time representation
without changing the smooth-region operator.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_wave_reconstruction as base  # noqa: E402
from exact_reference import cell_average_newtonian_hydro  # noqa: E402


Array = np.ndarray


@dataclasses.dataclass(frozen=True)
class PhysicalWave:
    left: Array
    right: Array
    speed: float
    left_fraction: float
    mixture_residual: float
    jump_residual: float


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
    mass_error: float
    momentum_error: float
    energy_error: float


def euler_flux(primitive: Array, gamma: float) -> Array:
    state = base.conserved_from_primitive(
        np.asarray(primitive, dtype=float)[None, :],
        gamma,
    )[0]
    velocity = float(primitive[1])
    pressure = float(primitive[2])
    return np.array(
        (
            state[1],
            state[1] * velocity + pressure,
            velocity * (state[2] + pressure),
        )
    )


def scaled_residual(value: Array, *references: Array) -> float:
    denominator = max(
        *(float(np.linalg.norm(reference)) for reference in references),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(value) / denominator)


def fit_physical_wave(
    average: Array,
    left: Array,
    right: Array,
    gamma: float,
    *,
    mixture_tolerance: float = 2.0e-10,
    jump_tolerance: float = 2.0e-10,
    allow_endpoint: bool = False,
) -> PhysicalWave | None:
    """Fit one conservative Euler discontinuity to a cell average."""
    endpoints = np.array((left, right), dtype=float)
    if not base.primitive_is_admissible(endpoints):
        return None
    endpoint_state = base.conserved_from_primitive(endpoints, gamma)
    average_state = base.conserved_from_primitive(
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

    left_flux = euler_flux(left, gamma)
    right_flux = euler_flux(right, gamma)
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
        base.stencil_value(primitive, cell, periodic),
        dtype=float,
    )


def states_agree(left: Array, right: Array, gamma: float) -> bool:
    left_state, right_state = base.conserved_from_primitive(
        np.array((left, right)),
        gamma,
    )
    return scaled_residual(
        left_state - right_state,
        left_state,
        right_state,
    ) <= 2.0e-10


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


def physical_wave_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    """Use exact physical boundary states around one fitted in-cell wave."""
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
            return common.copy(), common.copy()
        return base.characteristic_mp5(primitive, left_cell, gamma)

    face_wave = fit_face_wave(
        primitive,
        left_cell,
        gamma,
        periodic,
    )
    if face_wave is not None:
        upwind = face_wave.left if face_wave.speed >= 0.0 else face_wave.right
        return upwind.copy(), upwind.copy()
    return base.characteristic_mp5(primitive, left_cell, gamma)


physical_wave_mp5.periodic = False


def rhs_with_fluxes(
    state: Array,
    dx: float,
    gamma: float,
    reconstruct: base.FaceReconstruction,
    periodic: bool,
) -> tuple[Array, float, Array]:
    """Evaluate the research flux divergence and retain every face flux."""
    primitive = base.primitive_from_conserved(state, gamma)
    if not base.primitive_is_admissible(primitive):
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
        pair = np.array((left, right))
        if not base.primitive_is_admissible(pair):
            return np.full_like(state, np.nan), math.inf, fluxes
        endpoint_speeds = (
            abs(left[1]) + math.sqrt(gamma * left[2] / left[0]),
            abs(right[1]) + math.sqrt(gamma * right[2] / right[0]),
        )
        face_flux = base.hllc_flux(left, right, gamma)
        if not np.all(np.isfinite(face_flux)):
            return np.full_like(state, np.nan), math.inf, fluxes
        fluxes[face] = face_flux
        required_speed = max(required_speed, *endpoint_speeds)
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
    """Integrate isolated physical-wave fluxes over one complete timestep."""
    cells = len(primitive)
    overrides: dict[int, Array] = {}

    def face_index(face: int) -> int | None:
        if periodic:
            return face % cells
        if 0 <= face <= cells:
            return face
        return None

    def add_override(face: int, flux: Array) -> None:
        resolved_face = face_index(face)
        if resolved_face is None:
            return
        existing = overrides.get(resolved_face)
        if existing is None:
            overrides[resolved_face] = flux.copy()
            return
        tolerance = 2.0e-10 * max(
            float(np.linalg.norm(existing)),
            float(np.linalg.norm(flux)),
            1.0,
        )
        if np.linalg.norm(existing - flux) > tolerance:
            overrides.pop(resolved_face, None)

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
        left_flux = euler_flux(wave.left, gamma)
        right_flux = euler_flux(wave.right, gamma)
        left_average_flux = left_flux
        right_average_flux = right_flux
        if wave.speed > 0.0:
            crossing_time = (1.0 - wave.left_fraction) * dx / wave.speed
            if crossing_time < dt:
                crossed_fraction = min(max(1.0 - crossing_time / dt, 0.0), 1.0)
                right_average_flux = (
                    (1.0 - crossed_fraction) * right_flux
                    + crossed_fraction * left_flux
                )
        elif wave.speed < 0.0:
            crossing_time = wave.left_fraction * dx / (-wave.speed)
            if crossing_time < dt:
                crossed_fraction = min(max(1.0 - crossing_time / dt, 0.0), 1.0)
                left_average_flux = (
                    (1.0 - crossed_fraction) * left_flux
                    + crossed_fraction * right_flux
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
        wave = fit_face_wave(
            primitive,
            left_cell,
            gamma,
            periodic,
        )
        if wave is None:
            continue
        upwind = wave.left if wave.speed >= 0.0 else wave.right
        add_override(left_cell + 1, euler_flux(upwind, gamma))
        left_flux = euler_flux(wave.left, gamma)
        right_flux = euler_flux(wave.right, gamma)
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


def evolve_space_time_physical_waves(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
) -> tuple[Array, bool]:
    """SSPRK3 in smooth cells plus time-integrated isolated-wave fluxes."""
    state = base.conserved_from_primitive(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    while time < final_time:
        current = base.primitive_from_conserved(state, gamma)
        if not base.primitive_is_admissible(current):
            return current, False
        speeds = np.abs(current[:, 1]) + np.sqrt(
            gamma * current[:, 2] / current[:, 0]
        )
        rhs0, speed0, flux0 = rhs_with_fluxes(
            state,
            dx,
            gamma,
            physical_wave_mp5,
            periodic,
        )
        if not np.isfinite(speed0) or not np.all(np.isfinite(rhs0)):
            return current, False
        dt = min(
            final_time - time,
            cfl * dx / max(float(np.max(speeds)), speed0),
        )
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1, flux1 = rhs_with_fluxes(
                stage1,
                dx,
                gamma,
                physical_wave_mp5,
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
                physical_wave_mp5,
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
            candidate_primitive = base.primitive_from_conserved(
                candidate,
                gamma,
            )
            if not base.primitive_is_admissible(candidate_primitive):
                dt *= 0.5
                continue
            state = candidate
            break
        else:
            return base.primitive_from_conserved(state, gamma), False
        time += dt
    final = base.primitive_from_conserved(state, gamma)
    return final, base.primitive_is_admissible(final)


def normal_shock_states(
    gamma: float,
    mach: float,
    shock_speed: float,
) -> tuple[Array, Array]:
    """Return downstream/upstream states for one right-moving normal shock."""
    if mach <= 1.0:
        raise ValueError("normal shock Mach number must exceed one")
    upstream_rho = 1.0
    upstream_pressure = 1.0
    upstream_sound = math.sqrt(
        gamma * upstream_pressure / upstream_rho
    )
    upstream_relative_velocity = -mach * upstream_sound
    density_ratio = (
        (gamma + 1.0) * mach * mach
        / ((gamma - 1.0) * mach * mach + 2.0)
    )
    pressure_ratio = (
        2.0 * gamma * mach * mach - (gamma - 1.0)
    ) / (gamma + 1.0)
    downstream_rho = density_ratio * upstream_rho
    downstream_pressure = pressure_ratio * upstream_pressure
    downstream_relative_velocity = (
        upstream_relative_velocity / density_ratio
    )
    downstream = np.array(
        (
            downstream_rho,
            shock_speed + downstream_relative_velocity,
            downstream_pressure,
        )
    )
    upstream = np.array(
        (
            upstream_rho,
            shock_speed + upstream_relative_velocity,
            upstream_pressure,
        )
    )
    fitted = fit_physical_wave(
        downstream,
        downstream,
        upstream,
        gamma,
        allow_endpoint=True,
    )
    if fitted is None or not math.isclose(
        fitted.speed,
        shock_speed,
        rel_tol=2.0e-12,
        abs_tol=2.0e-12,
    ):
        raise RuntimeError("constructed normal shock fails its jump relation")
    return downstream, upstream


def discontinuity_cell_averages(
    cells: int,
    position: float,
    left: Array,
    right: Array,
    gamma: float,
) -> Array:
    edges = np.linspace(0.0, 1.0, cells + 1)

    def exact(points: Array) -> tuple[Array, Array, Array]:
        left_mask = points < position
        return (
            np.where(left_mask, left[0], right[0]),
            np.where(left_mask, left[2], right[2]),
            np.where(left_mask, left[1], right[1]),
        )

    density, pressure, velocity = cell_average_newtonian_hydro(
        exact,
        edges,
        gamma,
        breakpoints=(position,),
        quadrature_order=2,
    )
    return np.column_stack((density, velocity, pressure))


def result_for_discontinuity(
    *,
    case: str,
    method: str,
    cells: int,
    cfl: float,
    completed: bool,
    evolved: Array,
    exact: Array,
    initial: Array,
    gamma: float,
) -> Result:
    component_errors = []
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
        component_errors.append(
            float(
                np.mean(
                    np.abs(evolved_component - exact_component)
                )
                / scale
            )
        )
        overshoots.append(
            max(
                0.0,
                float(np.max(evolved_component) - np.max(exact_component)),
                float(np.min(exact_component) - np.min(evolved_component)),
            )
            / scale
        )
        tv_ratios.append(
            (
                base.total_variation(evolved_component)
                / base.total_variation(exact_component)
                if base.total_variation(exact_component) > 1.0e-12 * scale
                else base.total_variation(evolved_component) / scale
            )
        )
    exact_left = exact[0]
    exact_right = exact[-1]
    normalized_distance = np.minimum(
        np.linalg.norm(evolved - exact_left, axis=1)
        / max(np.linalg.norm(exact_right - exact_left), 1.0e-14),
        np.linalg.norm(evolved - exact_right, axis=1)
        / max(np.linalg.norm(exact_right - exact_left), 1.0e-14),
    )
    transition_cells = int(np.count_nonzero(normalized_distance > 2.0e-8))
    exact_integrals = np.mean(
        base.conserved_from_primitive(exact, gamma),
        axis=0,
    )
    final_integrals = np.mean(
        base.conserved_from_primitive(evolved, gamma),
        axis=0,
    )
    integral_error = np.abs(final_integrals - exact_integrals)
    return Result(
        case=case,
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        total_l1=sum(component_errors),
        maximum_overshoot=max(overshoots),
        maximum_tv_ratio=max(tv_ratios),
        transition_cells=transition_cells,
        mass_error=float(integral_error[0]),
        momentum_error=float(integral_error[1]),
        energy_error=float(integral_error[2]),
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
    reconstruct: base.FaceReconstruction,
    periodic: bool,
    space_time: bool = False,
) -> Result:
    initial = discontinuity_cell_averages(
        cells,
        initial_position,
        left,
        right,
        gamma,
    )
    if space_time:
        evolved, completed = evolve_space_time_physical_waves(
            initial,
            final_time=final_time,
            cfl=cfl,
            gamma=gamma,
            periodic=periodic,
        )
    else:
        evolved, completed = base.evolve(
            initial,
            final_time=final_time,
            cfl=cfl,
            gamma=gamma,
            reconstruct=reconstruct,
            riemann_flux=base.hllc_flux,
            periodic=periodic,
        )
    final_position = initial_position + speed * final_time
    exact = discontinuity_cell_averages(
        cells,
        final_position,
        left,
        right,
        gamma,
    )
    return result_for_discontinuity(
        case=case,
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        evolved=evolved,
        exact=exact,
        initial=initial,
        gamma=gamma,
    )


def smooth_entropy_error(
    cells: int,
    cfl: float,
    gamma: float,
    reconstruct: base.FaceReconstruction,
    *,
    space_time: bool = False,
) -> tuple[bool, float]:
    if space_time:
        x = (np.arange(cells) + 0.5) / cells
        primitive = np.column_stack(
            (
                1.0 + 0.2 * np.sin(2.0 * math.pi * x),
                np.full(cells, 0.5),
                np.ones(cells),
            )
        )
        evolved, completed = evolve_space_time_physical_waves(
            primitive,
            final_time=2.0,
            cfl=cfl,
            gamma=gamma,
            periodic=True,
        )
        return completed, float(
            np.mean(np.abs(evolved[:, 0] - primitive[:, 0]))
        )
    result = base.entropy_wave_result(
        method="probe",
        cells=cells,
        cfl=cfl,
        gamma=gamma,
        reconstruct=reconstruct,
        riemann_flux=base.hllc_flux,
    )
    return result.completed, result.total_l1


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
    parser.add_argument("--cells", type=int, nargs="+", default=(32,))
    parser.add_argument("--cfl", type=float, nargs="+", default=(0.9,))
    parser.add_argument("--gamma", type=float, default=1.4)
    args = parser.parse_args()

    gamma = args.gamma
    shock_speed = 1.25
    shock_left, shock_right = normal_shock_states(
        gamma,
        mach=2.0,
        shock_speed=shock_speed,
    )
    contact_left = np.array((2.0, 0.5, 1.0))
    contact_right = np.array((1.0, 0.5, 1.0))
    methods: tuple[
        tuple[str, base.FaceReconstruction, bool],
        ...,
    ] = (
        ("mp5", base.characteristic_mp5, False),
        ("physical_wave_mp5", physical_wave_mp5, False),
        ("physical_wave_space_time", physical_wave_mp5, True),
    )
    results: list[Result] = []
    for cells in args.cells:
        for cfl in args.cfl:
            for method, reconstruct, space_time in methods:
                results.append(
                    discontinuity_result(
                        case="moving_contact",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=gamma,
                        left=contact_left,
                        right=contact_right,
                        speed=0.5,
                        initial_position=0.233,
                        final_time=0.6,
                        reconstruct=reconstruct,
                        periodic=False,
                        space_time=space_time,
                    )
                )
                results.append(
                    discontinuity_result(
                        case="moving_normal_shock",
                        method=method,
                        cells=cells,
                        cfl=cfl,
                        gamma=gamma,
                        left=shock_left,
                        right=shock_right,
                        speed=shock_speed,
                        initial_position=0.217,
                        final_time=0.36,
                        reconstruct=reconstruct,
                        periodic=False,
                        space_time=space_time,
                    )
                )
            mp5_completed, mp5_entropy = smooth_entropy_error(
                cells,
                cfl,
                gamma,
                base.characteristic_mp5,
            )
            physical_completed, physical_entropy = smooth_entropy_error(
                cells,
                cfl,
                gamma,
                physical_wave_mp5,
            )
            space_time_completed, space_time_entropy = smooth_entropy_error(
                cells,
                cfl,
                gamma,
                physical_wave_mp5,
                space_time=True,
            )
            if not (
                mp5_completed
                and physical_completed
                and space_time_completed
            ):
                raise RuntimeError("smooth entropy-wave evolution failed")
            entropy_tolerance = 2.0e-13 * max(1.0, mp5_entropy)
            if abs(physical_entropy - mp5_entropy) > entropy_tolerance:
                raise RuntimeError(
                    "physical-wave path changed the smooth MP5 control: "
                    f"{physical_entropy:.16e} versus {mp5_entropy:.16e}"
                )
            if abs(space_time_entropy - mp5_entropy) > entropy_tolerance:
                raise RuntimeError(
                    "space-time physical-wave path changed smooth MP5: "
                    f"{space_time_entropy:.16e} versus {mp5_entropy:.16e}"
                )
            print(
                "smooth_entropy"
                f"\tcells={cells}"
                f"\tcfl={cfl:g}"
                f"\tmp5_l1={mp5_entropy:.12e}"
                f"\tphysical_l1={physical_entropy:.12e}"
                f"\tspace_time_l1={space_time_entropy:.12e}"
            )
    print_results(results)


if __name__ == "__main__":
    main()
