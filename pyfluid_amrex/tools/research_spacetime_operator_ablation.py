#!/usr/bin/env python3
"""Separate spatial reconstruction from local space-time prediction in HD.

This research-only harness compares the four operators needed to attribute the
Cycle-273 result without changing production AMReX:

* characteristic MP5 with SSPRK3;
* characteristic CWENO-Z/ADER-Prim face traces with SSPRK3;
* an MP5-derived parabola with the local space-time DG predictor; and
* characteristic CWENO-Z/ADER-Prim with the same local predictor.

The CWENO+SSPRK3 branch deliberately performs the same conserved-average to
point-primitive conversion and primitive polynomial reconstruction as the
space-time branch at every Runge--Kutta stage.  Its only intended difference is
the temporal operator.  Failed Runge--Kutta candidates retry the whole step;
there is no local reconstruction fallback in that branch.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_primitive_spacetime_fv as spacetime  # noqa: E402
import research_wave_reconstruction as hydro  # noqa: E402


Array = np.ndarray


@dataclasses.dataclass
class RunStats:
    steps: int = 0
    rejected_steps: int = 0


@dataclasses.dataclass(frozen=True)
class AblationRow:
    case: str
    method: str
    cells: int
    cfl: float
    completed: bool
    error: float
    overshoot: float
    tv_ratio: float
    mass_error: float
    momentum_error: float
    energy_error: float
    steps: int
    rejected_steps: int
    local_correction_fraction: float
    wall_seconds: float


def _cweno_face_fluxes(
    state: Array,
    dx: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None,
) -> tuple[Array, float]:
    """Return a method-of-lines RHS from the ADER-Prim spatial polynomial."""

    primitive = hydro.primitive_from_conserved(state, gamma)
    if not hydro.primitive_is_admissible(primitive):
        return np.full_like(state, np.nan), math.inf
    point_primitive, _center_scales = spacetime.ader_point_primitives(
        state,
        gamma,
        periodic,
        parameters,
    )
    coefficients = spacetime.characteristic_cweno_z_coefficients(
        point_primitive,
        gamma,
        periodic,
        parameters,
        point_samples=True,
    )
    coefficients, polynomial_valid = spacetime._positive_polynomial(
        coefficients,
        point_primitive,
    )
    if not np.all(polynomial_valid):
        return np.full_like(state, np.nan), math.inf

    left_evaluation = np.asarray([(-0.5) ** power for power in range(5)])
    right_evaluation = np.asarray([0.5**power for power in range(5)])
    cell_left = np.einsum("p,npc->nc", left_evaluation, coefficients)
    cell_right = np.einsum("p,npc->nc", right_evaluation, coefficients)
    cell_left = np.asarray(
        [
            hydro.positive_trace(trace, center)
            for trace, center in zip(
                cell_left,
                point_primitive,
                strict=True,
            )
        ]
    )
    cell_right = np.asarray(
        [
            hydro.positive_trace(trace, center)
            for trace, center in zip(
                cell_right,
                point_primitive,
                strict=True,
            )
        ]
    )

    fluxes = np.empty((len(state) + 1, 3))
    maximum_speed = 0.0
    for face in range(len(fluxes)):
        if not periodic and boundary_primitives is not None and face == 0:
            left = right = boundary_primitives[0]
        elif (
            not periodic
            and boundary_primitives is not None
            and face == len(state)
        ):
            left = right = boundary_primitives[1]
        else:
            left_cell = spacetime._cell_index(
                face - 1,
                len(state),
                periodic,
            )
            right_cell = spacetime._cell_index(
                face,
                len(state),
                periodic,
            )
            left = cell_right[left_cell]
            right = cell_left[right_cell]
        if not hydro.primitive_is_admissible(np.asarray((left, right))):
            return np.full_like(state, np.nan), math.inf
        maximum_speed = max(
            maximum_speed,
            abs(float(left[1]))
            + math.sqrt(gamma * float(left[2]) / float(left[0])),
            abs(float(right[1]))
            + math.sqrt(gamma * float(right[2]) / float(right[0])),
        )
        fluxes[face] = hydro.hllc_flux(left, right, gamma)
    if periodic:
        fluxes[-1] = fluxes[0]
    return -(fluxes[1:] - fluxes[:-1]) / dx, maximum_speed


def evolve_cweno_ssprk3(
    initial: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None = None,
) -> tuple[Array, bool, RunStats]:
    """Advance the matched CWENO spatial operator with SSPRK3."""

    state = hydro.conserved_from_primitive(initial, gamma)
    dx = 1.0 / len(state)
    current_time = 0.0
    stats = RunStats()
    while current_time < final_time:
        primitive = hydro.primitive_from_conserved(state, gamma)
        if not hydro.primitive_is_admissible(primitive):
            return primitive, False, stats
        cell_speed = float(
            np.max(
                np.abs(primitive[:, 1])
                + np.sqrt(gamma * primitive[:, 2] / primitive[:, 0])
            )
        )
        rhs0, face_speed = _cweno_face_fluxes(
            state,
            dx,
            gamma,
            periodic,
            parameters,
            boundary_primitives,
        )
        if not np.isfinite(face_speed) or not np.all(np.isfinite(rhs0)):
            return primitive, False, stats
        dt = min(
            final_time - current_time,
            cfl * dx / max(cell_speed, face_speed),
        )
        accepted = False
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1 = _cweno_face_fluxes(
                stage1,
                dx,
                gamma,
                periodic,
                parameters,
                boundary_primitives,
            )
            if not np.isfinite(speed1) or not np.all(np.isfinite(rhs1)):
                dt *= 0.5
                stats.rejected_steps += 1
                continue
            allowed_dt = cfl * dx / max(cell_speed, face_speed, speed1)
            if allowed_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * allowed_dt
                stats.rejected_steps += 1
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, speed2 = _cweno_face_fluxes(
                stage2,
                dx,
                gamma,
                periodic,
                parameters,
                boundary_primitives,
            )
            if not np.isfinite(speed2) or not np.all(np.isfinite(rhs2)):
                dt *= 0.5
                stats.rejected_steps += 1
                continue
            allowed_dt = cfl * dx / max(
                cell_speed,
                face_speed,
                speed1,
                speed2,
            )
            if allowed_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * allowed_dt
                stats.rejected_steps += 1
                continue
            candidate = (1.0 / 3.0) * state + (2.0 / 3.0) * (
                stage2 + dt * rhs2
            )
            candidate_primitive = hydro.primitive_from_conserved(
                candidate,
                gamma,
            )
            if not hydro.primitive_is_admissible(candidate_primitive):
                dt *= 0.5
                stats.rejected_steps += 1
                continue
            state = candidate
            accepted = True
            break
        if not accepted:
            return hydro.primitive_from_conserved(state, gamma), False, stats
        current_time += dt
        stats.steps += 1
    final = hydro.primitive_from_conserved(state, gamma)
    return final, hydro.primitive_is_admissible(final), stats


def _integral_errors(
    initial: Array,
    evolved: Array,
    gamma: float,
) -> tuple[float, float, float]:
    initial_state = hydro.conserved_from_primitive(initial, gamma)
    final_state = hydro.conserved_from_primitive(evolved, gamma)
    errors = np.abs(np.mean(final_state, axis=0) - np.mean(initial_state, axis=0))
    scales = np.maximum(np.ptp(initial_state, axis=0), 1.0)
    normalized = errors / scales
    return tuple(float(value) for value in normalized)


def _smooth_row(
    case: str,
    method: str,
    cells: int,
    cfl: float,
    initial: Array,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
    steps: int,
    rejected_steps: int,
    local_correction_fraction: float,
    wall_seconds: float,
) -> AblationRow:
    scale = max(float(np.ptp(exact[:, 0])), 1.0)
    error = float(np.mean(np.abs(evolved[:, 0] - exact[:, 0])) / scale)
    overshoot = max(
        0.0,
        float(np.max(evolved[:, 0]) - np.max(exact[:, 0])) / scale,
        float(np.min(exact[:, 0]) - np.min(evolved[:, 0])) / scale,
    )
    exact_tv = hydro.total_variation(exact[:, 0])
    tv_ratio = hydro.total_variation(evolved[:, 0]) / exact_tv
    mass, momentum, energy = _integral_errors(initial, evolved, gamma)
    return AblationRow(
        case,
        method,
        cells,
        cfl,
        completed,
        error,
        overshoot,
        tv_ratio,
        mass,
        momentum,
        energy,
        steps,
        rejected_steps,
        local_correction_fraction,
        wall_seconds,
    )


def _shock_row(
    case: hydro.ShockCase,
    method: str,
    cells: int,
    cfl: float,
    initial: Array,
    evolved: Array,
    completed: bool,
    gamma: float,
    steps: int,
    rejected_steps: int,
    local_correction_fraction: float,
    wall_seconds: float,
) -> AblationRow:
    result = hydro.shock_result(
        case,
        method,
        cells,
        evolved,
        completed,
        gamma,
    )
    return AblationRow(
        case.name,
        method,
        cells,
        cfl,
        completed,
        result.total_l1,
        result.maximum_overshoot,
        result.maximum_tv_ratio,
        result.mass_error,
        result.momentum_error,
        result.energy_error,
        steps,
        rejected_steps,
        local_correction_fraction,
        wall_seconds,
    )


def _spacetime_fraction(stats: spacetime.EvolutionStats, cells: int) -> float:
    total = stats.steps * cells
    corrected = stats.mp5_cells + stats.pwl_cells + stats.pwc_cells
    return corrected / max(total, 1)


def _run_method(
    method: str,
    initial: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None,
) -> tuple[Array, bool, int, int, float, float]:
    start = time.perf_counter()
    if method == "mp5_ssprk3":
        evolved, completed = hydro.evolve(
            initial,
            final_time,
            cfl,
            gamma,
            hydro.characteristic_mp5,
            hydro.hllc_flux,
            periodic,
            boundary_primitives,
        )
        steps = rejected = 0
        correction_fraction = 0.0
    elif method == "cweno_ssprk3":
        evolved, completed, stats = evolve_cweno_ssprk3(
            initial,
            final_time,
            cfl,
            gamma,
            periodic,
            parameters,
            boundary_primitives,
        )
        steps = stats.steps
        rejected = stats.rejected_steps
        correction_fraction = 0.0
    elif method in ("mp5_lsdg", "cweno_lsdg"):
        method_parameters = dataclasses.replace(
            parameters,
            mp5_parabolic_primary=method == "mp5_lsdg",
        )
        evolved, completed, stats = spacetime.evolve_spacetime(
            initial,
            final_time,
            cfl,
            gamma,
            periodic,
            method_parameters,
            boundary_primitives,
        )
        steps = stats.steps
        rejected = stats.rejected_steps
        correction_fraction = _spacetime_fraction(stats, len(initial))
    else:
        raise ValueError(f"unknown ablation method {method!r}")
    return (
        evolved,
        completed,
        steps,
        rejected,
        correction_fraction,
        time.perf_counter() - start,
    )


def run(args: argparse.Namespace) -> list[AblationRow]:
    parameters = spacetime.PredictorParameters(
        predictor_kind="lsdg",
        picard_iterations=args.picard_iterations,
        anchor_mp5_traces=False,
        double_reconstruction=True,
    )
    methods = ("mp5_ssprk3", "cweno_ssprk3", "mp5_lsdg", "cweno_lsdg")
    rows: list[AblationRow] = []
    for cells in sorted(set(args.cells)):
        if "entropy" in args.cases:
            initial = spacetime.entropy_cell_averages(cells)
            exact = spacetime.entropy_cell_averages(cells, args.entropy_time)
            for method in methods:
                values = _run_method(
                    method,
                    initial,
                    args.entropy_time,
                    args.cfl,
                    args.gamma,
                    True,
                    parameters,
                    None,
                )
                rows.append(
                    _smooth_row(
                        "entropy",
                        method,
                        cells,
                        args.cfl,
                        initial,
                        values[0],
                        exact,
                        values[1],
                        args.gamma,
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                    )
                )
        for case in spacetime.EXACT_CASES:
            if case.name not in args.cases:
                continue
            x = (np.arange(cells) + 0.5) / cells
            initial = np.empty((cells, 3))
            initial[x < 0.5] = case.left
            initial[x >= 0.5] = case.right
            boundaries = (np.asarray(case.left), np.asarray(case.right))
            for method in methods:
                values = _run_method(
                    method,
                    initial,
                    case.final_time,
                    args.cfl,
                    args.gamma,
                    False,
                    parameters,
                    boundaries,
                )
                rows.append(
                    _shock_row(
                        case,
                        method,
                        cells,
                        args.cfl,
                        initial,
                        values[0],
                        values[1],
                        args.gamma,
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                    )
                )
    return rows


def print_rows(rows: list[AblationRow]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(AblationRow))
    print("\t".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = getattr(row, field)
            values.append(f"{value:.12e}" if isinstance(value, float) else str(value))
        print("\t".join(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--entropy-time", type=float, default=2.0)
    parser.add_argument("--picard-iterations", type=int, default=8)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("entropy", "sod", "lax", "blast", "contact"),
        default=("entropy", "sod", "lax", "blast", "contact"),
    )
    args = parser.parse_args()
    if any(cells < 8 for cells in args.cells):
        raise ValueError("all grids need at least eight cells")
    if args.cfl <= 0.0 or args.gamma <= 1.0:
        raise ValueError("CFL must be positive and gamma must exceed one")
    if args.entropy_time <= 0.0 or args.picard_iterations < 1:
        raise ValueError("time and Picard iteration count must be positive")
    print_rows(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
