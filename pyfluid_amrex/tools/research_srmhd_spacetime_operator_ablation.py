#!/usr/bin/env python3
"""Attribute the SRMHD coherent operator to space, time, and correction.

The offline comparison holds the production C2P/HLLD oracle fixed and runs:

* component MP5 with SSPRK3;
* characteristic CWENO-Z/ADER-Prim traces with SSPRK3;
* a component-MP5 parabola with the local space-time DG predictor; and
* characteristic CWENO-Z/ADER-Prim with that local predictor.

The CWENO Runge--Kutta branch rebuilds the same conserved-average to
point-primitive polynomial at every stage and has no local reconstruction
fallback.  ``--disable-dmp`` removes only the relaxed DMP from the local
space-time branches; admissibility correction remains.  No production AMReX
source is exercised or changed by this research harness.
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
import research_srmhd_spacetime_fv as coherent  # noqa: E402
import research_physical_wave_subcell_srmhd as srmhd  # noqa: E402


Array = np.ndarray


@dataclasses.dataclass(frozen=True)
class AblationRow:
    case: str
    method: str
    cells: int
    cfl: float
    completed: bool
    maximum_l1: float
    observed_order: float
    density_l1: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    maximum_integral_error: float
    normal_b_spread: float
    steps: int
    cfl_retries: int
    riemann_fallbacks: int
    effective_correction_fraction: float
    wall_seconds: float


def _cweno_faces(
    state: Array,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None,
) -> tuple[Array, Array, Array] | None:
    recovery = coherent.recover_conserved(state, gamma)
    if not np.all(recovery.success):
        return None
    point_primitive, _center_scales = coherent.point_primitives(
        state,
        gamma,
        periodic,
        parameters,
    )
    coefficients = coherent.characteristic_cweno_z_coefficients(
        point_primitive,
        gamma,
        periodic,
        parameters,
        point_samples=True,
    )
    coefficients, valid = coherent._admissible_polynomial(
        coefficients,
        point_primitive,
    )
    if not np.all(valid):
        return None

    left_weights = np.asarray([(-0.5) ** power for power in range(5)])
    right_weights = np.asarray([0.5**power for power in range(5)])
    cell_left = np.einsum("p,npc->nc", left_weights, coefficients)
    cell_right = np.einsum("p,npc->nc", right_weights, coefficients)
    faces = len(state) + 1
    left = np.empty((faces, 8))
    right = np.empty_like(left)
    for face in range(faces):
        if periodic:
            left[face] = cell_right[(face - 1) % len(state)]
            right[face] = cell_left[face % len(state)]
        elif face == 0 and boundary_primitives is not None:
            left[face] = right[face] = boundary_primitives[0]
        elif face == len(state) and boundary_primitives is not None:
            left[face] = right[face] = boundary_primitives[1]
        elif face == 0:
            left[face] = right[face] = cell_left[0]
        elif face == len(state):
            left[face] = right[face] = cell_right[-1]
        else:
            left[face] = cell_right[face - 1]
            right[face] = cell_left[face]
        normal_b = 0.5 * (left[face, 5] + right[face, 5])
        left[face, 5] = normal_b
        right[face, 5] = normal_b
    if not srmhd.primitive_is_admissible(np.concatenate((left, right))):
        return None
    return left, right, recovery.primitive


def _cweno_rhs(
    state: Array,
    dx: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None,
) -> tuple[Array | None, float, int]:
    face_data = _cweno_faces(
        state,
        gamma,
        periodic,
        parameters,
        boundary_primitives,
    )
    if face_data is None:
        return None, math.inf, 0
    left, right, primitive = face_data
    fluxes, fallbacks = coherent.hlld_fluxes(left, right, gamma)
    maximum_speed = max(
        abs(signal)
        for row in np.concatenate((primitive, left, right))
        for signal in srmhd.signal_speeds(row, gamma)
    )
    counted = fallbacks[:-1] if periodic else fallbacks
    return (
        -(fluxes[1:] - fluxes[:-1]) / dx,
        maximum_speed,
        int(np.count_nonzero(counted)),
    )


def evolve_cweno_ssprk3(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None = None,
    domain_length: float = 1.0,
) -> tuple[Array, bool, coherent.EvolutionStats]:
    state, _flux = coherent.state_and_flux(primitive, gamma)
    dx = domain_length / len(state)
    current_time = 0.0
    stats = coherent.EvolutionStats()
    while current_time < final_time:
        rhs0, speed0, fallback0 = _cweno_rhs(
            state,
            dx,
            gamma,
            periodic,
            parameters,
            boundary_primitives,
        )
        if rhs0 is None:
            recovery = coherent.recover_conserved(state, gamma)
            return recovery.primitive, False, stats
        dt = min(final_time - current_time, cfl * dx / speed0)
        accepted = False
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1, fallback1 = _cweno_rhs(
                stage1,
                dx,
                gamma,
                periodic,
                parameters,
                boundary_primitives,
            )
            if rhs1 is None:
                dt *= 0.5
                stats.cfl_retries += 1
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, speed2, fallback2 = _cweno_rhs(
                stage2,
                dx,
                gamma,
                periodic,
                parameters,
                boundary_primitives,
            )
            if rhs2 is None:
                dt *= 0.5
                stats.cfl_retries += 1
                continue
            allowed_dt = cfl * dx / max(speed0, speed1, speed2)
            if allowed_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * allowed_dt
                stats.cfl_retries += 1
                continue
            candidate = state / 3.0 + 2.0 * (
                stage2 + dt * rhs2
            ) / 3.0
            recovery = coherent.recover_conserved(candidate, gamma)
            if not np.all(recovery.success) or not srmhd.primitive_is_admissible(
                recovery.primitive
            ):
                dt *= 0.5
                stats.cfl_retries += 1
                continue
            state = candidate
            stats.riemann_fallbacks += fallback0 + fallback1 + fallback2
            accepted = True
            break
        if not accepted:
            recovery = coherent.recover_conserved(state, gamma)
            return recovery.primitive, False, stats
        current_time += dt
        stats.steps += 1
    recovery = coherent.recover_conserved(state, gamma)
    return recovery.primitive, bool(np.all(recovery.success)), stats


def _effective_correction_fraction(
    stats: coherent.EvolutionStats,
    cells: int,
    *,
    mp5_primary: bool,
) -> float:
    corrected = stats.pwl_cells + stats.pwc_cells
    if not mp5_primary:
        corrected += stats.mp5_cells
    return corrected / max(stats.steps * cells, 1)


def _run_method(
    method: str,
    initial: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None = None,
    domain_length: float = 1.0,
) -> tuple[Array, bool, coherent.EvolutionStats, float, float]:
    start = time.perf_counter()
    if method == "component_mp5_ssprk3":
        evolved, completed, stats = coherent.evolve_mp5_ssprk3(
            initial,
            final_time,
            cfl,
            gamma,
            periodic,
            boundary_primitives,
            domain_length,
        )
        correction_fraction = 0.0
    elif method == "characteristic_cweno_ssprk3":
        evolved, completed, stats = evolve_cweno_ssprk3(
            initial,
            final_time,
            cfl,
            gamma,
            periodic,
            parameters,
            boundary_primitives,
            domain_length,
        )
        correction_fraction = 0.0
    elif method in ("component_mp5_lsdg", "characteristic_cweno_lsdg"):
        mp5_primary = method == "component_mp5_lsdg"
        method_parameters = dataclasses.replace(
            parameters,
            characteristic_reconstruction=not mp5_primary,
            mp5_parabolic_primary=mp5_primary,
        )
        evolved, completed, stats = coherent.evolve_spacetime(
            initial,
            final_time,
            cfl,
            gamma,
            method_parameters,
            periodic,
            boundary_primitives,
            domain_length,
            contact_correction_enabled=False,
        )
        correction_fraction = _effective_correction_fraction(
            stats,
            len(initial),
            mp5_primary=mp5_primary,
        )
    else:
        raise ValueError(f"unknown ablation method {method!r}")
    return (
        evolved,
        completed,
        stats,
        correction_fraction,
        time.perf_counter() - start,
    )


def _smooth_row(
    method: str,
    cells: int,
    cfl: float,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
    stats: coherent.EvolutionStats,
    correction_fraction: float,
    wall_seconds: float,
    observed_order: float,
) -> AblationRow:
    result = coherent._result(
        method,
        cells,
        cfl,
        evolved,
        exact,
        completed,
        gamma,
        stats,
    )
    integral_error = max(
        result.mass_error,
        result.momentum_error,
        result.energy_error,
    )
    return AblationRow(
        "entropy",
        method,
        cells,
        cfl,
        completed,
        result.maximum_l1,
        observed_order,
        result.maximum_l1,
        result.maximum_overshoot,
        result.maximum_tv_ratio,
        integral_error,
        result.normal_b_spread,
        stats.steps,
        stats.cfl_retries,
        stats.riemann_fallbacks,
        correction_fraction,
        wall_seconds,
    )


def _shock_row(
    method: str,
    case: int,
    cells: int,
    cfl: float,
    evolved: Array,
    completed: bool,
    gamma: float,
    exact_profile: Array,
    exact_average: Array,
    stats: coherent.EvolutionStats,
    correction_fraction: float,
    wall_seconds: float,
) -> AblationRow:
    result = coherent._shock_result(
        method,
        case,
        cells,
        cfl,
        evolved,
        completed,
        gamma,
        exact_profile,
        exact_average,
        stats,
    )
    return AblationRow(
        f"balsara_{case}",
        method,
        cells,
        cfl,
        completed,
        result.maximum_l1,
        math.nan,
        result.density_l1,
        result.maximum_overshoot,
        result.maximum_tv_ratio,
        result.maximum_integral_error,
        result.normal_b_spread,
        stats.steps,
        stats.cfl_retries,
        stats.riemann_fallbacks,
        correction_fraction,
        wall_seconds,
    )


def run(args: argparse.Namespace) -> list[AblationRow]:
    parameters = spacetime.PredictorParameters(
        predictor_kind="lsdg",
        picard_iterations=args.picard_iterations,
        central_linear_weight=args.central_linear_weight,
        wenoz_exponent=args.wenoz_exponent,
        characteristic_reconstruction=True,
        anchor_mp5_traces=False,
        sensor_relative_activity_floor=1.0e-7,
        dmp_relative_tolerance=(
            1.0e6 if args.disable_dmp else args.dmp_relative_tolerance
        ),
        dmp_absolute_tolerance=(1.0e6 if args.disable_dmp else 1.0e-10),
    )
    methods = tuple(args.methods)
    rows: list[AblationRow] = []
    previous: dict[str, tuple[int, float]] = {}
    if args.entropy:
        for cells in sorted(set(args.cells)):
            initial = coherent.entropy_cell_averages(
                cells,
                0.0,
                args.gamma,
            )
            exact = coherent.entropy_cell_averages(
                cells,
                1.0,
                args.gamma,
            )
            for method in methods:
                values = _run_method(
                    method,
                    initial,
                    1.0,
                    args.cfl,
                    args.gamma,
                    True,
                    parameters,
                )
                order = math.nan
                prior = previous.get(method)
                result = coherent._result(
                    method,
                    cells,
                    args.cfl,
                    values[0],
                    exact,
                    values[1],
                    args.gamma,
                    values[2],
                )
                if prior is not None and result.maximum_l1 > 0.0:
                    prior_cells, prior_error = prior
                    order = math.log(
                        prior_error / result.maximum_l1
                    ) / math.log(cells / prior_cells)
                previous[method] = (cells, result.maximum_l1)
                rows.append(
                    _smooth_row(
                        method,
                        cells,
                        args.cfl,
                        values[0],
                        exact,
                        values[1],
                        args.gamma,
                        values[2],
                        values[3],
                        values[4],
                        order,
                    )
                )

    for case in args.balsara_cases:
        spec = coherent._balsara_spec(case)
        gamma = float(spec["gamma"])
        final_time = float(spec["time"])
        exact_profile = coherent.read_last_exact_balsara_profile(
            coherent.EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
        )
        for cells in sorted(set(args.balsara_cells)):
            initial, boundaries = coherent._balsara_initial(spec, cells)
            edges = np.linspace(-1.0, 1.0, cells + 1)
            exact_average = coherent.exact_rmhd_cell_averages(
                exact_profile,
                normal_magnetic=float(boundaries[0][5]),
                gamma=gamma,
                cell_edges=edges,
            )
            for method in methods:
                values = _run_method(
                    method,
                    initial,
                    final_time,
                    args.cfl,
                    gamma,
                    False,
                    parameters,
                    boundaries,
                    2.0,
                )
                rows.append(
                    _shock_row(
                        method,
                        case,
                        cells,
                        args.cfl,
                        values[0],
                        values[1],
                        gamma,
                        exact_profile,
                        exact_average,
                        values[2],
                        values[3],
                        values[4],
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
            values.append(
                f"{value:.12e}" if isinstance(value, float) else str(value)
            )
        print("\t".join(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--picard-iterations", type=int, default=8)
    parser.add_argument("--central-linear-weight", type=float, default=0.85)
    parser.add_argument("--wenoz-exponent", type=int, default=2)
    parser.add_argument("--dmp-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--disable-dmp", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "component_mp5_ssprk3",
            "characteristic_cweno_ssprk3",
            "component_mp5_lsdg",
            "characteristic_cweno_lsdg",
        ),
        default=(
            "component_mp5_ssprk3",
            "characteristic_cweno_ssprk3",
            "component_mp5_lsdg",
            "characteristic_cweno_lsdg",
        ),
    )
    parser.add_argument(
        "--entropy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--balsara-cases", type=int, nargs="+", default=())
    parser.add_argument("--balsara-cells", type=int, nargs="+", default=(64,))
    args = parser.parse_args()
    if any(cells < 8 for cells in (*args.cells, *args.balsara_cells)):
        raise ValueError("all grids need at least eight cells")
    if args.cfl <= 0.0 or args.gamma <= 1.0:
        raise ValueError("CFL and gamma are outside their domains")
    if args.picard_iterations < 1 or args.dmp_relative_tolerance < 0.0:
        raise ValueError("iteration count or DMP tolerance is invalid")
    if not 0.0 < args.central_linear_weight < 1.0:
        raise ValueError("the central linear weight must lie in (0, 1)")
    if args.wenoz_exponent < 1:
        raise ValueError("the WENO-Z exponent must be positive")
    if any(case not in range(1, 6) for case in args.balsara_cases):
        raise ValueError("Balsara cases are numbered 1 through 5")
    print_rows(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
