#!/usr/bin/env python3
"""Separate MHD spatial reconstruction from local time prediction.

This research-only harness is the ideal-MHD counterpart of
``research_spacetime_operator_ablation.py``.  It compares four conservative
operators while keeping HLLD and all point/conserved conversion contracts
fixed:

* characteristic MP5 with SSPRK3;
* characteristic CWENO-Z/ADER-Prim face traces with SSPRK3;
* an MP5-derived parabola with the local space-time DG predictor; and
* characteristic CWENO-Z/ADER-Prim with that local predictor.

The CWENO+SSPRK3 branch reconstructs the same point-primitive polynomial at
every Runge--Kutta stage.  It has whole-step retry but no local fallback.  The
space-time branches retain the existing a-posteriori correction ladder so the
reported correction fraction can distinguish core behavior from repair.
Production AMReX code is intentionally out of scope.
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

import research_mhd_spacetime_fv as coherent  # noqa: E402
import research_mhd_wave_reconstruction as mhd  # noqa: E402
import research_primitive_spacetime_fv as spacetime  # noqa: E402
import research_subcell_mhd_reconstruction as shock_mhd  # noqa: E402
from research_constrained_riemann import (  # noqa: E402
    mhd_fast_speed,
    mhd_hlld_flux,
)


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
    total_l1: float
    observed_order: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    mass_error: float
    momentum_error: float
    energy_error: float
    normal_b_spread: float
    steps: int
    rejected_steps: int
    effective_correction_fraction: float
    wall_seconds: float


def _cweno_rhs(
    state: Array,
    dx: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
) -> tuple[Array, float]:
    """Return a method-of-lines RHS from the ADER-Prim MHD polynomial."""

    primitive = mhd.primitive_from_conserved(state, gamma)
    if not mhd.primitive_is_admissible(primitive):
        return np.full_like(state, np.nan), math.inf
    point_primitive = coherent.point_primitives(
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
    coefficients, valid = coherent._positive_polynomial(
        coefficients,
        point_primitive,
    )
    if not np.all(valid):
        return np.full_like(state, np.nan), math.inf

    left_weights = np.asarray([(-0.5) ** power for power in range(5)])
    right_weights = np.asarray([0.5**power for power in range(5)])
    cell_left = np.einsum("p,npc->nc", left_weights, coefficients)
    cell_right = np.einsum("p,npc->nc", right_weights, coefficients)
    cell_left = np.asarray(
        [
            mhd.positive_trace(trace, center)
            for trace, center in zip(
                cell_left,
                point_primitive,
                strict=True,
            )
        ]
    )
    cell_right = np.asarray(
        [
            mhd.positive_trace(trace, center)
            for trace, center in zip(
                cell_right,
                point_primitive,
                strict=True,
            )
        ]
    )

    fluxes = np.empty((len(state) + 1, 8))
    maximum_speed = 0.0
    for face in range(len(fluxes)):
        if periodic:
            left_cell = (face - 1) % len(state)
            right_cell = face % len(state)
        elif face == 0:
            left_cell = right_cell = 0
        elif face == len(state):
            left_cell = right_cell = len(state) - 1
        else:
            left_cell = face - 1
            right_cell = face
        left = cell_right[left_cell].copy()
        right = cell_left[right_cell].copy()
        normal_b = 0.5 * (left[5] + right[5])
        left[5] = normal_b
        right[5] = normal_b
        if not mhd.primitive_is_admissible(np.asarray((left, right))):
            return np.full_like(state, np.nan), math.inf
        maximum_speed = max(
            maximum_speed,
            abs(float(left[1])) + mhd_fast_speed(left, gamma),
            abs(float(right[1])) + mhd_fast_speed(right, gamma),
        )
        face_flux = mhd_hlld_flux(left, right, gamma)
        if not np.all(np.isfinite(face_flux)):
            return np.full_like(state, np.nan), math.inf
        fluxes[face] = face_flux
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
) -> tuple[Array, bool, RunStats]:
    """Advance matched MHD CWENO face traces with SSPRK3."""

    state = mhd.conserved_array(initial, gamma)
    dx = 1.0 / len(state)
    current_time = 0.0
    stats = RunStats()
    while current_time < final_time:
        primitive = mhd.primitive_from_conserved(state, gamma)
        if not mhd.primitive_is_admissible(primitive):
            return primitive, False, stats
        cell_speed = max(
            abs(float(row[1])) + mhd_fast_speed(row, gamma)
            for row in primitive
        )
        rhs0, speed0 = _cweno_rhs(
            state,
            dx,
            gamma,
            periodic,
            parameters,
        )
        if not np.isfinite(speed0) or not np.all(np.isfinite(rhs0)):
            return primitive, False, stats
        dt = min(
            final_time - current_time,
            cfl * dx / max(cell_speed, speed0),
        )
        accepted = False
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1 = _cweno_rhs(
                stage1,
                dx,
                gamma,
                periodic,
                parameters,
            )
            if not np.isfinite(speed1) or not np.all(np.isfinite(rhs1)):
                dt *= 0.5
                stats.rejected_steps += 1
                continue
            allowed_dt = cfl * dx / max(cell_speed, speed0, speed1)
            if allowed_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * allowed_dt
                stats.rejected_steps += 1
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, speed2 = _cweno_rhs(
                stage2,
                dx,
                gamma,
                periodic,
                parameters,
            )
            if not np.isfinite(speed2) or not np.all(np.isfinite(rhs2)):
                dt *= 0.5
                stats.rejected_steps += 1
                continue
            allowed_dt = cfl * dx / max(
                cell_speed,
                speed0,
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
            candidate_primitive = mhd.primitive_from_conserved(
                candidate,
                gamma,
            )
            if not mhd.primitive_is_admissible(candidate_primitive):
                dt *= 0.5
                stats.rejected_steps += 1
                continue
            state = candidate
            accepted = True
            break
        if not accepted:
            return mhd.primitive_from_conserved(state, gamma), False, stats
        current_time += dt
        stats.steps += 1
    final = mhd.primitive_from_conserved(state, gamma)
    return final, mhd.primitive_is_admissible(final), stats


def _effective_correction_fraction(
    stats: coherent.MhdEvolutionStats,
    cells: int,
    *,
    mp5_primary: bool,
) -> float:
    total = stats.steps * cells
    corrected = stats.pwl_cells + stats.pwc_cells
    if not mp5_primary:
        corrected += stats.mp5_cells
    return corrected / max(total, 1)


def _run_method(
    method: str,
    initial: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
) -> tuple[Array, bool, int, int, float, float]:
    start = time.perf_counter()
    if method == "mp5_ssprk3":
        evolved, completed = mhd.evolve(
            initial,
            final_time,
            cfl,
            gamma,
            mhd.characteristic_mp5,
            mhd_hlld_flux,
            periodic,
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
        )
        steps = stats.steps
        rejected = stats.rejected_steps
        correction_fraction = 0.0
    elif method in ("mp5_lsdg", "cweno_lsdg"):
        mp5_primary = method == "mp5_lsdg"
        method_parameters = dataclasses.replace(
            parameters,
            mp5_parabolic_primary=mp5_primary,
        )
        evolved, completed, stats = coherent.evolve_spacetime(
            initial,
            final_time,
            cfl,
            gamma,
            method_parameters,
            periodic=periodic,
            contact_correction_enabled=False,
        )
        steps = stats.steps
        rejected = stats.cfl_retries
        correction_fraction = _effective_correction_fraction(
            stats,
            len(initial),
            mp5_primary=mp5_primary,
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    return (
        evolved,
        completed,
        steps,
        rejected,
        correction_fraction,
        time.perf_counter() - start,
    )


def _row_from_result(
    result: mhd.Result,
    cfl: float,
    observed_order: float,
    steps: int,
    rejected_steps: int,
    correction_fraction: float,
    wall_seconds: float,
) -> AblationRow:
    return AblationRow(
        case=result.case,
        method=result.method,
        cells=result.cells,
        cfl=cfl,
        completed=result.completed,
        total_l1=result.total_l1,
        observed_order=observed_order,
        maximum_overshoot=result.maximum_overshoot,
        maximum_tv_ratio=result.maximum_tv_ratio,
        mass_error=result.mass_error,
        momentum_error=result.momentum_error,
        energy_error=result.energy_error,
        normal_b_spread=result.normal_b_spread,
        steps=steps,
        rejected_steps=rejected_steps,
        effective_correction_fraction=correction_fraction,
        wall_seconds=wall_seconds,
    )


def _shock_specifications() -> dict[
    str,
    tuple[object, float, float],
]:
    return {
        "brio_wu": (mhd.brio_wu_initial, 2.0, 0.1),
        "dai_woodward": (
            shock_mhd.dai_woodward_initial,
            5.0 / 3.0,
            0.08,
        ),
    }


def run(args: argparse.Namespace) -> list[AblationRow]:
    dmp_relative_tolerance = (
        1.0e6 if args.disable_dmp else args.dmp_relative_tolerance
    )
    dmp_absolute_tolerance = 1.0e6 if args.disable_dmp else 1.0e-10
    parameters = spacetime.PredictorParameters(
        predictor_kind="lsdg",
        picard_iterations=args.picard_iterations,
        anchor_mp5_traces=False,
        double_reconstruction=True,
        physical_fronts=False,
        acoustic_shock_scale=coherent.MHD_ACOUSTIC_SHOCK_SCALE,
        dmp_relative_tolerance=dmp_relative_tolerance,
        dmp_absolute_tolerance=dmp_absolute_tolerance,
        sensor_relative_activity_floor=args.sensor_relative_activity_floor,
    )
    methods = ("mp5_ssprk3", "cweno_ssprk3", "mp5_lsdg", "cweno_lsdg")
    rows: list[AblationRow] = []
    prior_smooth: dict[str, tuple[int, float]] = {}
    for cells in sorted(set(args.cells)):
        if "circular_alfven" in args.cases:
            initial = coherent.circular_alfven_cell_averages(
                cells,
                0.0,
                args.gamma,
                args.amplitude,
            )
            exact = coherent.circular_alfven_cell_averages(
                cells,
                1.0,
                args.gamma,
                args.amplitude,
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
                result = mhd.result_from_exact(
                    "circular_alfven",
                    method,
                    values[0],
                    exact,
                    values[1],
                    args.gamma,
                )
                order = math.nan
                prior = prior_smooth.get(method)
                if prior is not None and result.total_l1 > 0.0:
                    prior_cells, prior_error = prior
                    order = math.log(prior_error / result.total_l1) / math.log(
                        cells / prior_cells
                    )
                prior_smooth[method] = (cells, result.total_l1)
                rows.append(
                    _row_from_result(
                        result,
                        args.cfl,
                        order,
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                    )
                )

        specifications = _shock_specifications()
        for case in args.cases:
            if case == "circular_alfven":
                continue
            initial_data, gamma, final_time = specifications[case]
            fine_initial = initial_data(cells * args.reference_factor)
            reference, reference_completed = mhd.evolve(
                fine_initial,
                final_time,
                args.cfl,
                gamma,
                mhd.characteristic_mp5,
                mhd_hlld_flux,
                periodic=False,
            )
            if not reference_completed:
                raise RuntimeError(f"fine {case} reference failed")
            exact = mhd.coarsen_conserved(reference, cells, gamma)
            initial = initial_data(cells)
            for method in methods:
                values = _run_method(
                    method,
                    initial,
                    final_time,
                    args.cfl,
                    gamma,
                    False,
                    parameters,
                )
                result = mhd.result_from_exact(
                    f"{case}_reference",
                    method,
                    values[0],
                    exact,
                    values[1],
                    gamma,
                )
                rows.append(
                    _row_from_result(
                        result,
                        args.cfl,
                        math.nan,
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
            values.append(
                f"{value:.12e}" if isinstance(value, float) else str(value)
            )
        print("\t".join(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--picard-iterations", type=int, default=8)
    parser.add_argument("--reference-factor", type=int, default=8)
    parser.add_argument("--dmp-relative-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--sensor-relative-activity-floor",
        type=float,
        default=1.0e-7,
    )
    parser.add_argument(
        "--disable-dmp",
        action="store_true",
        help="disable only the relaxed DMP; admissibility correction remains",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("circular_alfven", "brio_wu", "dai_woodward"),
        default=("circular_alfven", "brio_wu", "dai_woodward"),
    )
    args = parser.parse_args()
    if any(cells < 8 for cells in args.cells):
        raise ValueError("all grids need at least eight cells")
    if args.cfl <= 0.0 or args.gamma <= 1.0 or args.amplitude <= 0.0:
        raise ValueError("CFL, gamma, and amplitude are outside their domains")
    if args.picard_iterations < 1 or args.reference_factor < 2:
        raise ValueError("iteration count and reference factor are too small")
    if args.dmp_relative_tolerance < 0.0:
        raise ValueError("the DMP relative tolerance cannot be negative")
    if not 0.0 <= args.sensor_relative_activity_floor < 1.0:
        raise ValueError("the sensor activity floor must lie in [0, 1)")
    print_rows(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
