#!/usr/bin/env python3
"""Probe dissipative solution selection in the Brio--Wu Riemann problem.

Ideal MHD Riemann data do not always select a unique entropy solution.  This
standalone research harness therefore asks a narrower question before a
general exact-fan implementation is attempted: does an explicitly stated
viscous, resistive, heat-conducting regularization approach one fan, and does
that fan depend on the ratios of the dissipative coefficients?

The microproblem is evolved with a conservative fourth-order centered ideal
flux and physical diffusive face fluxes.  The discontinuity is initially
smoothed on the same vanishing length scale as the transport coefficients.
This is deliberately not production code, an exact Riemann solver, or a
validation reference.  It is a falsification experiment for the proposition
that an ``exact ideal-MHD fan`` alone defines a universal persistent element.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclasses.dataclass(frozen=True)
class Regularization:
    name: str
    viscosity_ratio: float
    resistivity_ratio: float
    thermal_ratio: float


@dataclasses.dataclass(frozen=True)
class RunResult:
    case: str
    regularization: str
    cells: int
    completed: bool
    steps: int
    runtime_seconds: float
    minimum_density: float
    minimum_pressure: float
    maximum_normal_b_spread: float
    balance_residual: float


def primitive_to_conserved(primitive: Array, gamma: float) -> Array:
    rho = primitive[..., 0]
    velocity = primitive[..., 1:4]
    pressure = primitive[..., 4]
    magnetic = primitive[..., 5:8]
    momentum = rho[..., None] * velocity
    energy = (
        pressure / (gamma - 1.0)
        + 0.5 * rho * np.sum(velocity * velocity, axis=-1)
        + 0.5 * np.sum(magnetic * magnetic, axis=-1)
    )
    return np.concatenate(
        (
            rho[..., None],
            momentum,
            energy[..., None],
            magnetic,
        ),
        axis=-1,
    )


def conserved_to_primitive(state: Array, gamma: float) -> Array:
    rho = state[..., 0]
    velocity = state[..., 1:4] / rho[..., None]
    magnetic = state[..., 5:8]
    kinetic = 0.5 * np.sum(state[..., 1:4] * velocity, axis=-1)
    magnetic_energy = 0.5 * np.sum(magnetic * magnetic, axis=-1)
    pressure = (gamma - 1.0) * (
        state[..., 4] - kinetic - magnetic_energy
    )
    return np.concatenate(
        (
            rho[..., None],
            velocity,
            pressure[..., None],
            magnetic,
        ),
        axis=-1,
    )


def ideal_flux(primitive: Array, gamma: float) -> Array:
    state = primitive_to_conserved(primitive, gamma)
    rho = primitive[..., 0]
    vx = primitive[..., 1]
    velocity = primitive[..., 1:4]
    pressure = primitive[..., 4]
    magnetic = primitive[..., 5:8]
    bx = magnetic[..., 0]
    magnetic2 = np.sum(magnetic * magnetic, axis=-1)
    total_pressure = pressure + 0.5 * magnetic2
    velocity_dot_b = np.sum(velocity * magnetic, axis=-1)
    flux = np.empty_like(state)
    flux[..., 0] = rho * vx
    flux[..., 1] = rho * vx * vx + total_pressure - bx * bx
    flux[..., 2] = rho * vx * velocity[..., 1] - bx * magnetic[..., 1]
    flux[..., 3] = rho * vx * velocity[..., 2] - bx * magnetic[..., 2]
    flux[..., 4] = (
        (state[..., 4] + total_pressure) * vx
        - bx * velocity_dot_b
    )
    flux[..., 5] = 0.0
    flux[..., 6] = magnetic[..., 1] * vx - bx * velocity[..., 1]
    flux[..., 7] = magnetic[..., 2] * vx - bx * velocity[..., 2]
    return flux


def fast_speed(primitive: Array, gamma: float) -> Array:
    rho = primitive[..., 0]
    pressure = primitive[..., 4]
    magnetic = primitive[..., 5:8]
    sound2 = gamma * pressure / rho
    magnetic2 = np.sum(magnetic * magnetic, axis=-1) / rho
    normal2 = magnetic[..., 0] ** 2 / rho
    discriminant = np.maximum(
        (sound2 + magnetic2) ** 2 - 4.0 * sound2 * normal2,
        0.0,
    )
    return np.sqrt(
        np.maximum(
            0.5 * (sound2 + magnetic2 + np.sqrt(discriminant)),
            0.0,
        )
    )


def admissible(primitive: Array) -> bool:
    return bool(
        np.all(np.isfinite(primitive))
        and np.all(primitive[..., 0] > 0.0)
        and np.all(primitive[..., 4] > 0.0)
    )


def pad_constant(values: Array, width: int = 2) -> Array:
    return np.pad(values, ((width, width), (0, 0)), mode="edge")


def face_interpolate(values: Array) -> Array:
    """Fourth-order centered values on all physical faces."""
    padded = pad_constant(values)
    cells = len(values)
    faces = np.arange(cells + 1) + 1
    return (
        -padded[faces - 1]
        + 7.0 * padded[faces]
        + 7.0 * padded[faces + 1]
        - padded[faces + 2]
    ) / 12.0


def face_gradient(values: Array, dx: float) -> Array:
    """Fourth-order centered gradients on all physical faces."""
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    padded = pad_constant(values)
    cells = len(values)
    faces = np.arange(cells + 1) + 1
    gradient = (
        padded[faces - 1]
        - 27.0 * padded[faces]
        + 27.0 * padded[faces + 1]
        - padded[faces + 2]
    ) / (24.0 * dx)
    return gradient[:, 0] if squeeze else gradient


def diffusive_flux(
    primitive: Array,
    dx: float,
    epsilon: float,
    gamma: float,
    regularization: Regularization,
) -> Array:
    face_primitive = face_interpolate(primitive)
    velocity_gradient = face_gradient(primitive[:, 1:4], dx)
    transverse_b_gradient = face_gradient(primitive[:, 6:8], dx)
    temperature_gradient = face_gradient(
        primitive[:, 4] / primitive[:, 0],
        dx,
    )

    density = np.maximum(face_primitive[:, 0], np.finfo(float).tiny)
    dynamic_viscosity = (
        density * epsilon * regularization.viscosity_ratio
    )
    resistivity = epsilon * regularization.resistivity_ratio
    conductivity = (
        density
        * epsilon
        * regularization.thermal_ratio
        / (gamma - 1.0)
    )
    stress = dynamic_viscosity[:, None] * velocity_gradient
    stress[:, 0] *= 4.0 / 3.0

    flux = np.zeros((len(primitive) + 1, 8))
    flux[:, 1:4] = stress
    flux[:, 6:8] = resistivity * transverse_b_gradient
    flux[:, 4] = (
        np.sum(face_primitive[:, 1:4] * stress, axis=1)
        + conductivity * temperature_gradient
        + resistivity
        * np.sum(
            face_primitive[:, 6:8] * transverse_b_gradient,
            axis=1,
        )
    )
    return flux


def rhs(
    state: Array,
    dx: float,
    epsilon: float,
    gamma: float,
    regularization: Regularization,
) -> tuple[Array, Array]:
    primitive = conserved_to_primitive(state, gamma)
    if not admissible(primitive):
        return np.full_like(state, np.nan), np.full((2, 8), np.nan)
    face_ideal_flux = face_interpolate(ideal_flux(primitive, gamma))
    face_diffusive_flux = diffusive_flux(
        primitive,
        dx,
        epsilon,
        gamma,
        regularization,
    )
    total_flux = face_ideal_flux - face_diffusive_flux
    return (
        -(total_flux[1:] - total_flux[:-1]) / dx,
        total_flux[[0, -1]],
    )


def smoothed_riemann_initial(
    cells: int,
    smoothing_cells: float,
    left: Array,
    right: Array,
    x_min: float,
    x_max: float,
    discontinuity: float,
) -> tuple[Array, Array]:
    dx = (x_max - x_min) / cells
    x = x_min + (np.arange(cells) + 0.5) * dx
    width = smoothing_cells * dx
    weight = 0.5 * (
        1.0 - np.tanh((x - discontinuity) / width)
    )
    primitive = (
        weight[:, None] * left[None, :]
        + (1.0 - weight[:, None]) * right[None, :]
    )
    return x, primitive


def riemann_case(
    case: str,
    cells: int,
    smoothing_cells: float,
) -> tuple[Array, Array, float, float, float]:
    if case == "brio_wu":
        gamma = 2.0
        x_min = 0.0
        x_max = 1.0
        discontinuity = 0.5
        left = np.array(
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        )
        right = np.array(
            (0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
        )
        default_final_time = 0.1
    elif case in {"torrilhon_pi", "torrilhon_3"}:
        gamma = 5.0 / 3.0
        x_min = -1.0
        x_max = 1.5
        discontinuity = 0.0
        angle = math.pi if case == "torrilhon_pi" else 3.0
        left = np.array(
            (1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        )
        right = np.array(
            (
                0.2,
                0.0,
                0.0,
                0.0,
                0.2,
                1.0,
                math.cos(angle),
                math.sin(angle),
            )
        )
        default_final_time = 0.4
    else:
        raise ValueError(f"unknown Riemann case: {case}")
    x, primitive = smoothed_riemann_initial(
        cells,
        smoothing_cells,
        left,
        right,
        x_min,
        x_max,
        discontinuity,
    )
    return x, primitive, gamma, x_max - x_min, default_final_time


def evolve(
    case: str,
    cells: int,
    requested_final_time: float | None,
    cfl: float,
    epsilon_cells: float,
    smoothing_cells: float,
    regularization: Regularization,
) -> tuple[Array, Array, float, RunResult]:
    x, initial, gamma, domain_length, default_final_time = riemann_case(
        case,
        cells,
        smoothing_cells,
    )
    final_time = (
        default_final_time
        if requested_final_time is None
        else requested_final_time
    )
    dx = domain_length / cells
    state = primitive_to_conserved(initial, gamma)
    initial_integral = dx * np.sum(state, axis=0)
    maximum_ratio = max(
        4.0 * regularization.viscosity_ratio / 3.0,
        regularization.resistivity_ratio,
        regularization.thermal_ratio,
    )
    epsilon = epsilon_cells * dx / maximum_ratio
    maximum_diffusivity = epsilon * maximum_ratio
    time_value = 0.0
    steps = 0
    integrated_boundary_flux = np.zeros(8)
    started = time.perf_counter()
    completed = True
    while time_value < final_time:
        primitive = conserved_to_primitive(state, gamma)
        if not admissible(primitive):
            completed = False
            break
        speed = float(
            np.max(np.abs(primitive[:, 1]) + fast_speed(primitive, gamma))
        )
        dt_hyperbolic = cfl * dx / speed
        dt_diffusive = (
            math.inf
            if maximum_diffusivity == 0.0
            else 0.16 * dx * dx / maximum_diffusivity
        )
        dt = min(final_time - time_value, dt_hyperbolic, dt_diffusive)
        stage_rhs, stage_boundary_flux = rhs(
            state,
            dx,
            epsilon,
            gamma,
            regularization,
        )
        stage1 = state + dt * stage_rhs
        stage_rhs1, stage_boundary_flux1 = rhs(
            stage1,
            dx,
            epsilon,
            gamma,
            regularization,
        )
        stage2 = 0.75 * state + 0.25 * (stage1 + dt * stage_rhs1)
        stage_rhs2, stage_boundary_flux2 = rhs(
            stage2,
            dx,
            epsilon,
            gamma,
            regularization,
        )
        candidate = (1.0 / 3.0) * state + (2.0 / 3.0) * (
            stage2 + dt * stage_rhs2
        )
        candidate_primitive = conserved_to_primitive(candidate, gamma)
        if not admissible(candidate_primitive):
            completed = False
            break
        state = candidate
        stage_net_flux = stage_boundary_flux[1] - stage_boundary_flux[0]
        stage_net_flux1 = (
            stage_boundary_flux1[1] - stage_boundary_flux1[0]
        )
        stage_net_flux2 = (
            stage_boundary_flux2[1] - stage_boundary_flux2[0]
        )
        integrated_boundary_flux += dt * (
            stage_net_flux / 6.0
            + stage_net_flux1 / 6.0
            + 2.0 * stage_net_flux2 / 3.0
        )
        time_value += dt
        steps += 1

    elapsed = time.perf_counter() - started
    primitive = conserved_to_primitive(state, gamma)
    expected_integral = initial_integral - integrated_boundary_flux
    balance_scale = max(
        float(np.max(np.abs(expected_integral))),
        1.0,
    )
    balance_residual = float(
        np.max(np.abs(dx * np.sum(state, axis=0) - expected_integral))
        / balance_scale
    )
    result = RunResult(
        case=case,
        regularization=regularization.name,
        cells=cells,
        completed=completed and math.isclose(time_value, final_time),
        steps=steps,
        runtime_seconds=elapsed,
        minimum_density=float(np.min(primitive[:, 0])),
        minimum_pressure=float(np.min(primitive[:, 4])),
        maximum_normal_b_spread=float(np.ptp(primitive[:, 5])),
        balance_residual=balance_residual,
    )
    return x, primitive, gamma, result


def coarsen_conserved(
    primitive: Array,
    target_cells: int,
    gamma: float,
) -> Array:
    state = primitive_to_conserved(primitive, gamma)
    ratio = len(state) // target_cells
    if ratio * target_cells != len(state):
        raise ValueError("fine resolution must be divisible by target")
    coarse = state.reshape(target_cells, ratio, 8).mean(axis=1)
    return conserved_to_primitive(coarse, gamma)


def normalized_l1(left: Array, right: Array) -> float:
    component_scale = np.maximum(np.ptp(right, axis=0), 1.0e-13)
    return float(np.mean(np.abs(left - right) / component_scale))


def print_run_results(results: list[RunResult]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(RunResult))
    print("\t".join(fields))
    for result in results:
        values = []
        for field in fields:
            value = getattr(result, field)
            values.append(
                f"{value:.12e}" if isinstance(value, float) else str(value)
            )
        print("\t".join(values))


def parse_regularization(value: str) -> Regularization:
    try:
        name, ratios = value.split("=", maxsplit=1)
        viscosity, resistivity, thermal = (
            float(item) for item in ratios.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "regularization must be NAME=VISCOSITY,RESISTIVITY,THERMAL"
        ) from error
    if min(viscosity, resistivity, thermal) <= 0.0:
        raise argparse.ArgumentTypeError(
            "regularization ratios must all be positive"
        )
    return Regularization(name, viscosity, resistivity, thermal)


def save_profiles(
    path: Path,
    profiles: dict[tuple[str, str, int], tuple[Array, Array]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Array] = {}
    for (case, name, cells), (x, primitive) in profiles.items():
        prefix = f"{case}_{name}_{cells}"
        arrays[f"{prefix}_x"] = x
        arrays[f"{prefix}_primitive"] = primitive
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("brio_wu", "torrilhon_pi", "torrilhon_3"),
        default=("brio_wu",),
    )
    parser.add_argument("--cells", type=int, nargs="+", default=(256, 512))
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--epsilon-cells", type=float, default=1.5)
    parser.add_argument("--smoothing-cells", type=float, default=2.0)
    parser.add_argument(
        "--regularization",
        action="append",
        type=parse_regularization,
        default=None,
        help="NAME=VISCOSITY,RESISTIVITY,THERMAL; may be repeated",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.cells) < 16:
        parser.error("--cells values must be at least 16")
    if any(cells <= 0 for cells in args.cells):
        parser.error("--cells values must be positive")
    if args.final_time is not None and args.final_time <= 0.0:
        parser.error("--final-time must be positive")
    if args.cfl <= 0.0:
        parser.error("--cfl must be positive")
    if args.epsilon_cells <= 0.0 or args.smoothing_cells <= 0.0:
        parser.error("--epsilon-cells and --smoothing-cells must be positive")
    regularizations = args.regularization or (
        Regularization("balanced", 1.0, 1.0, 1.0),
        Regularization("viscous", 4.0, 1.0, 1.0),
        Regularization("resistive", 1.0, 4.0, 1.0),
    )

    profiles: dict[tuple[str, str, int], tuple[Array, Array]] = {}
    gammas: dict[str, float] = {}
    run_results = []
    for case in args.cases:
        for regularization in regularizations:
            for cells in sorted(set(args.cells)):
                x, primitive, gamma, result = evolve(
                    case,
                    cells,
                    args.final_time,
                    args.cfl,
                    args.epsilon_cells,
                    args.smoothing_cells,
                    regularization,
                )
                profiles[(case, regularization.name, cells)] = (
                    x,
                    primitive,
                )
                gammas[case] = gamma
                run_results.append(result)
    print_run_results(run_results)

    resolutions = sorted(set(args.cells))
    finest = resolutions[-1]
    print("comparison\tleft\tright\tnormalized_l1")
    for case in args.cases:
        gamma = gammas[case]
        for regularization in regularizations:
            finest_profile = profiles[
                (case, regularization.name, finest)
            ][1]
            for cells in resolutions[:-1]:
                coarse_reference = coarsen_conserved(
                    finest_profile,
                    cells,
                    gamma,
                )
                error = normalized_l1(
                    profiles[(case, regularization.name, cells)][1],
                    coarse_reference,
                )
                print(
                    f"resolution:{case}\t{regularization.name}:{cells}\t"
                    f"{regularization.name}:{finest}\t{error:.12e}"
                )
        baseline = regularizations[0]
        baseline_profile = profiles[(case, baseline.name, finest)][1]
        for regularization in regularizations[1:]:
            error = normalized_l1(
                profiles[(case, regularization.name, finest)][1],
                baseline_profile,
            )
            print(
                f"path:{case}\t{regularization.name}:{finest}\t"
                f"{baseline.name}:{finest}\t{error:.12e}"
            )

    if args.output is not None:
        save_profiles(args.output, profiles)
    if not all(result.completed for result in run_results):
        raise SystemExit("one or more regularized microproblems failed")
    if max(result.balance_residual for result in run_results) > 2.0e-10:
        raise SystemExit("regularized microproblem violated conservation")
    if (
        max(result.maximum_normal_b_spread for result in run_results)
        > 2.0e-12
    ):
        raise SystemExit("regularized microproblem changed normal field")


if __name__ == "__main__":
    main()
