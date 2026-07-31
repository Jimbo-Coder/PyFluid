#!/usr/bin/env python3
"""Test a complete causal fan in an exact NRMHD invariant subsystem.

For gamma=2, zero normal field and transverse velocity, and ``By = k rho``,
one-dimensional ideal MHD is exactly Euler flow with effective pressure equal
to gas pressure plus magnetic pressure.  This maps the symmetric shock
collision and fan-rebirth problem into a genuinely magnetized fast-wave test.
It exercises MHD conservation, induction, primitive recovery, HLLD, and fan
interaction, but not nonzero normal field, Alfvén/slow waves, CT, or AMR.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_constrained_riemann import (  # noqa: E402
    mhd_flux,
    mhd_hlld_flux,
)
import research_causal_fan_interaction as interaction  # noqa: E402
import research_mhd_wave_reconstruction as mhd  # noqa: E402
import research_physical_wave_subcell_mhd as physical_mhd  # noqa: E402
import research_physical_wave_subcell as physical_hydro  # noqa: E402
import research_wave_reconstruction as hydro  # noqa: E402


Array = np.ndarray
TIME_NODES, TIME_WEIGHTS = np.polynomial.legendre.leggauss(8)


def effective_to_mhd(
    effective: Array,
    magnetic_mass_ratio: float,
) -> Array:
    rho, velocity, effective_pressure = effective
    transverse_magnetic = magnetic_mass_ratio * rho
    gas_pressure = effective_pressure - 0.5 * transverse_magnetic**2
    if gas_pressure <= 0.0:
        raise RuntimeError("effective state maps to nonpositive gas pressure")
    return np.asarray(
        (
            rho,
            velocity,
            0.0,
            0.0,
            gas_pressure,
            0.0,
            transverse_magnetic,
            0.0,
        )
    )


def map_effective_averages(
    effective: Array,
    magnetic_mass_ratio: float,
) -> Array:
    mapped = np.asarray(
        [
            effective_to_mhd(row, magnetic_mass_ratio)
            for row in effective
        ]
    )
    if not mhd.primitive_is_admissible(mapped):
        raise RuntimeError("mapped MHD cell averages are inadmissible")
    return mapped


def validate_invariant_subsystem(
    problem: interaction.InteractionProblem,
    gamma: float,
    magnetic_mass_ratio: float,
) -> None:
    states = (
        problem.left,
        problem.middle,
        problem.right,
        problem.left_fan.star_left,
        problem.left_fan.star_right,
        problem.right_fan.star_left,
        problem.right_fan.star_right,
        problem.child_fan.star_left,
        problem.child_fan.star_right,
    )
    for effective in states:
        mapped = effective_to_mhd(effective, magnetic_mass_ratio)
        effective_conserved = hydro.conserved_from_primitive(
            effective[np.newaxis, :],
            gamma,
        )[0]
        effective_flux = physical_hydro.euler_flux(effective, gamma)
        expected_conserved = np.asarray(
            (
                effective_conserved[0],
                effective_conserved[1],
                0.0,
                0.0,
                effective_conserved[2],
                0.0,
                magnetic_mass_ratio * effective_conserved[0],
                0.0,
            )
        )
        expected_flux = np.asarray(
            (
                effective_flux[0],
                effective_flux[1],
                0.0,
                0.0,
                effective_flux[2],
                0.0,
                magnetic_mass_ratio * effective_flux[0],
                0.0,
            )
        )
        actual_conserved = mhd.conserved_array(
            mapped[np.newaxis, :],
            gamma,
        )[0]
        actual_flux = mhd_flux(mapped, gamma)
        scale = max(
            float(np.max(np.abs(expected_conserved))),
            float(np.max(np.abs(expected_flux))),
            1.0,
        )
        error = max(
            float(np.max(np.abs(actual_conserved - expected_conserved))),
            float(np.max(np.abs(actual_flux - expected_flux))),
        )
        if error > 3.0e-13 * scale:
            raise RuntimeError(
                "effective-Euler/MHD invariant map failed: "
                f"scaled error={error / scale:.3e}"
            )


def rhs_with_fluxes(
    state: Array,
    dx: float,
    gamma: float,
) -> tuple[Array, float, Array]:
    primitive = mhd.primitive_from_conserved(state, gamma)
    if not mhd.primitive_is_admissible(primitive):
        return (
            np.full_like(state, np.nan),
            math.inf,
            np.full((len(state) + 1, 8), np.nan),
        )
    mhd.characteristic_mp5.periodic = False
    fluxes = np.empty((len(state) + 1, 8))
    required_speed = 0.0
    for face_index in range(len(state) + 1):
        left, right = mhd.characteristic_mp5(
            primitive,
            face_index - 1,
            gamma,
        )
        if not mhd.primitive_is_admissible(np.asarray((left, right))):
            return np.full_like(state, np.nan), math.inf, fluxes
        face_flux = mhd_hlld_flux(left, right, gamma)
        if not np.all(np.isfinite(face_flux)):
            return np.full_like(state, np.nan), math.inf, fluxes
        fluxes[face_index] = face_flux
        required_speed = max(
            required_speed,
            abs(left[1]) + mhd.mhd_fast_speed(left, gamma),
            abs(right[1]) + mhd.mhd_fast_speed(right, gamma),
        )
    return (
        -(fluxes[1:] - fluxes[:-1]) / dx,
        required_speed,
        fluxes,
    )


def exact_face_flux(
    *,
    face_position: float,
    time: float,
    dt: float,
    problem: interaction.InteractionProblem,
    gamma: float,
    magnetic_mass_ratio: float,
) -> Array:
    integrated = np.zeros(8)
    cuts = interaction.crossing_times(
        problem,
        face_position,
        time,
        dt,
    )
    for lower, upper in zip(cuts[:-1], cuts[1:], strict=True):
        midpoint = 0.5 * (lower + upper)
        half_duration = 0.5 * (upper - lower)
        quadrature_times = midpoint + half_duration * TIME_NODES
        for quadrature_time, weight in zip(
            quadrature_times,
            TIME_WEIGHTS,
            strict=True,
        ):
            effective_state = interaction.exact_state_at(
                problem,
                face_position,
                float(quadrature_time),
                gamma,
            )
            mapped_state = effective_to_mhd(
                effective_state,
                magnetic_mass_ratio,
            )
            integrated += (
                half_duration
                * weight
                * mhd_flux(mapped_state, gamma)
            )
    return integrated / dt


def evolve(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    problem: interaction.InteractionProblem,
    magnetic_mass_ratio: float,
) -> tuple[Array, bool]:
    state = mhd.conserved_array(primitive, gamma)
    cells = len(state)
    dx = 1.0 / cells
    time = 0.0
    while time < final_time:
        current = mhd.primitive_from_conserved(state, gamma)
        if not mhd.primitive_is_admissible(current):
            return current, False
        rhs0, speed0, flux0 = rhs_with_fluxes(state, dx, gamma)
        if not np.isfinite(speed0) or not np.all(np.isfinite(rhs0)):
            return current, False
        dt = min(final_time - time, cfl * dx / speed0)
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1, flux1 = rhs_with_fluxes(stage1, dx, gamma)
            if not np.isfinite(speed1):
                dt *= 0.5
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, speed2, flux2 = rhs_with_fluxes(stage2, dx, gamma)
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
            for face_index in range(cells + 1):
                integrated_flux[face_index] = exact_face_flux(
                    face_position=face_index * dx,
                    time=time,
                    dt=dt,
                    problem=problem,
                    gamma=gamma,
                    magnetic_mass_ratio=magnetic_mass_ratio,
                )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--cfl", type=float, nargs="+", default=(0.4, 0.9))
    parser.add_argument("--magnetic-mass-ratio", type=float, default=0.1)
    args = parser.parse_args()
    gamma = 2.0
    problem = interaction.make_symmetric_problem(gamma)
    validate_invariant_subsystem(
        problem,
        gamma,
        args.magnetic_mass_ratio,
    )
    final_time = 0.5 * (
        problem.interaction_time + problem.next_interaction_time
    )
    print(
        "event"
        f"\tinteraction_time={problem.interaction_time:.12e}"
        f"\tnext_interaction_time={problem.next_interaction_time:.12e}"
        f"\tfinal_time={final_time:.12e}"
        f"\tmagnetic_mass_ratio={args.magnetic_mass_ratio:.12e}"
    )
    results: list[physical_mhd.Result] = []
    for cells in args.cells:
        effective_initial = interaction.cell_averages(
            problem,
            cells,
            0.0,
            gamma,
        )
        effective_exact = interaction.cell_averages(
            problem,
            cells,
            final_time,
            gamma,
        )
        initial = map_effective_averages(
            effective_initial,
            args.magnetic_mass_ratio,
        )
        exact = map_effective_averages(
            effective_exact,
            args.magnetic_mass_ratio,
        )
        for cfl in args.cfl:
            baseline, baseline_completed = mhd.evolve(
                initial,
                final_time,
                cfl,
                gamma,
                mhd.characteristic_mp5,
                mhd_hlld_flux,
                periodic=False,
            )
            candidate, candidate_completed = evolve(
                initial,
                final_time,
                cfl,
                gamma,
                problem,
                args.magnetic_mass_ratio,
            )
            baseline_result = physical_mhd.result_from_exact(
                case="perpendicular_fast_shock_collision",
                method="mp5_hlld",
                cells=cells,
                cfl=cfl,
                completed=baseline_completed,
                evolved=baseline,
                exact=exact,
                gamma=gamma,
            )
            candidate_result = physical_mhd.result_from_exact(
                case="perpendicular_fast_shock_collision",
                method="causal_full_fan",
                cells=cells,
                cfl=cfl,
                completed=candidate_completed,
                evolved=candidate,
                exact=exact,
                gamma=gamma,
            )
            magnetic_ratio_error = float(
                np.max(
                    np.abs(
                        candidate[:, 6] / candidate[:, 0]
                        - args.magnetic_mass_ratio
                    )
                )
            )
            if (
                not candidate_result.completed
                or candidate_result.total_l1 >= baseline_result.total_l1
                or candidate_result.maximum_overshoot
                > baseline_result.maximum_overshoot + 5.0e-12
                or candidate_result.maximum_tv_ratio
                > baseline_result.maximum_tv_ratio + 5.0e-12
                or candidate_result.maximum_divergence_proxy > 5.0e-12
                or magnetic_ratio_error > 5.0e-12
                or max(
                    candidate_result.mass_error,
                    candidate_result.momentum_error,
                    candidate_result.energy_error,
                )
                > 5.0e-10
            ):
                raise RuntimeError(
                    "complete causal MHD fan failed its interaction gate"
                )
            results.extend((baseline_result, candidate_result))
    physical_mhd.print_results(results)


if __name__ == "__main__":
    main()
