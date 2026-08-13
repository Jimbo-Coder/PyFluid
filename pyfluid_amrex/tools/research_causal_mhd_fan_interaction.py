#!/usr/bin/env python3
"""Test a state-derived two-fan lifecycle in an exact NRMHD subsystem.

For gamma=2, zero normal field and transverse velocity, and ``By = k rho``,
one-dimensional ideal MHD is exactly Euler flow with effective pressure equal
to gas pressure plus magnetic pressure.  This maps the symmetric shock
collision and fan-rebirth problem into a genuinely magnetized fast-wave test.
Two initial fan origins and endpoint states are recovered from finite-volume
data with strict plateau/mixture checks.  The initial and child Euler fans are
then solved from those recovered states, so this is explicitly
exact-Euler-solver-assisted, not an oracle-free general MHD Riemann provider.
It exercises MHD conservation, induction, primitive recovery, HLLD, and one
causal interaction, but not nonzero normal field, Alfvén/slow waves, CT, AMR,
or a general regularity selector.

Course-correction status: parked as an exact-subsystem interaction control.
It does not establish a general NRHD-to-GRMHD reconstruction path and must not
be promoted into production.  Production work resumes from saved, stage-local
SRMHD flux causality before any broader fan-lifecycle implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
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


@dataclasses.dataclass(frozen=True)
class MhdFanBirth:
    left: Array
    right: Array
    position: float
    location_kind: str


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


def mhd_to_effective(
    primitive: Array,
    magnetic_mass_ratio: float,
) -> Array:
    """Invert the restricted perpendicular-field invariant map."""
    primitive = np.asarray(primitive, dtype=float)
    if not mhd.primitive_is_admissible(primitive[np.newaxis, :]):
        raise ValueError("inferred MHD fan endpoint is inadmissible")
    rho, velocity, vy, vz, pressure, bx, by, bz = primitive
    scale = max(abs(magnetic_mass_ratio * rho), 1.0)
    if (
        max(abs(vy), abs(vz), abs(bx), abs(bz)) > 3.0e-10
        or abs(by - magnetic_mass_ratio * rho) > 3.0e-10 * scale
    ):
        raise ValueError("state is outside the perpendicular MHD subsystem")
    return np.asarray((rho, velocity, pressure + 0.5 * by * by))


def find_two_mhd_fan_births(
    primitive: Array,
    gamma: float,
) -> tuple[MhdFanBirth, MhdFanBirth] | None:
    """Infer exactly two isolated Riemann origins from finite-volume state."""
    cells = len(primitive)
    dx = 1.0 / cells
    births: list[MhdFanBirth] = []
    for cell in range(2, cells - 2):
        left_outer, left, average, right, right_outer = primitive[
            cell - 2 : cell + 3
        ]
        if not physical_mhd.states_agree(left_outer, left, gamma):
            continue
        if not physical_mhd.states_agree(right, right_outer, gamma):
            continue
        mixture = physical_mhd.fit_mhd_conserved_mixture(
            average,
            left,
            right,
            gamma,
            mixture_tolerance=2.0e-10,
        )
        if mixture is None:
            continue
        births.append(
            MhdFanBirth(
                left=np.asarray(left, dtype=float).copy(),
                right=np.asarray(right, dtype=float).copy(),
                position=(cell + mixture.left_fraction) * dx,
                location_kind="mixed_cell",
            )
        )
    for face in range(2, cells - 1):
        left_outer, left, right, right_outer = primitive[face - 2 : face + 2]
        if not physical_mhd.states_agree(left_outer, left, gamma):
            continue
        if not physical_mhd.states_agree(right, right_outer, gamma):
            continue
        if physical_mhd.states_agree(left, right, gamma):
            continue
        births.append(
            MhdFanBirth(
                left=np.asarray(left, dtype=float).copy(),
                right=np.asarray(right, dtype=float).copy(),
                position=face * dx,
                location_kind="face",
            )
        )
    if len(births) != 2:
        return None
    first, second = sorted(births, key=lambda birth: birth.position)
    if first.position >= second.position:
        return None
    return first, second


def build_state_derived_problem(
    primitive: Array,
    gamma: float,
    magnetic_mass_ratio: float,
) -> tuple[interaction.InteractionProblem, tuple[MhdFanBirth, MhdFanBirth]] | None:
    """Solve the restricted exact-Euler fan lifecycle from two inferred births."""
    births = find_two_mhd_fan_births(primitive, gamma)
    if births is None:
        return None
    left_birth, right_birth = births
    try:
        left = mhd_to_effective(left_birth.left, magnetic_mass_ratio)
        middle_left = mhd_to_effective(left_birth.right, magnetic_mass_ratio)
        middle_right = mhd_to_effective(right_birth.left, magnetic_mass_ratio)
        right = mhd_to_effective(right_birth.right, magnetic_mass_ratio)
    except ValueError:
        return None
    if not physical_mhd.states_agree(
        left_birth.right, right_birth.left, gamma
    ):
        return None
    middle = 0.5 * (middle_left + middle_right)
    left_fan = interaction.fan.exact_fan_bundle(left, middle, gamma)
    right_fan = interaction.fan.exact_fan_bundle(middle, right, gamma)
    closing_speed = left_fan.right_head_speed - right_fan.left_head_speed
    if (
        not left_fan.right_is_shock
        or not right_fan.left_is_shock
        or closing_speed <= 0.0
    ):
        return None
    interaction_time = (
        right_birth.position - left_birth.position
    ) / closing_speed
    left_collision_position = (
        left_birth.position + left_fan.right_head_speed * interaction_time
    )
    right_collision_position = (
        right_birth.position + right_fan.left_head_speed * interaction_time
    )
    if not math.isclose(
        left_collision_position,
        right_collision_position,
        rel_tol=0.0,
        abs_tol=2.0e-12,
    ):
        return None
    interaction_position = 0.5 * (
        left_collision_position + right_collision_position
    )
    # This child is deliberately exact-Euler-solver-assisted in the invariant
    # subsystem. It is not a general NRMHD Riemann or regularity selector.
    child_fan = interaction.fan.exact_fan_bundle(
        left_fan.star_right,
        right_fan.star_left,
        gamma,
    )
    left_denominator = left_fan.contact_speed - child_fan.left_head_speed
    right_denominator = child_fan.right_head_speed - right_fan.contact_speed
    if left_denominator <= 0.0 or right_denominator <= 0.0:
        return None
    left_next = (
        interaction_position
        - child_fan.left_head_speed * interaction_time
        - left_birth.position
    ) / left_denominator
    right_next = (
        right_birth.position
        - interaction_position
        + child_fan.right_head_speed * interaction_time
    ) / right_denominator
    next_interaction_time = min(left_next, right_next)
    if next_interaction_time <= interaction_time:
        return None
    return (
        interaction.InteractionProblem(
            left=left,
            middle=middle,
            right=right,
            left_position=left_birth.position,
            right_position=right_birth.position,
            left_fan=left_fan,
            right_fan=right_fan,
            interaction_time=interaction_time,
            interaction_position=interaction_position,
            child_fan=child_fan,
            next_interaction_time=next_interaction_time,
        ),
        births,
    )


def validate_birth_rejections(initial: Array, gamma: float) -> None:
    """Reject smooth, damaged, and over-specified initial fan data."""
    constant = np.repeat(initial[:1], len(initial), axis=0)
    if find_two_mhd_fan_births(constant, gamma) is not None:
        raise RuntimeError("constant MHD data produced two fan births")
    damaged = initial.copy()
    damaged[len(initial) * 7 // 20, 7] = 1.0e-3
    if find_two_mhd_fan_births(damaged, gamma) is not None:
        raise RuntimeError("damaged MHD data produced two fan births")
    left, middle, right = initial[0], initial[len(initial) // 2], initial[-1]
    ambiguous = np.empty_like(initial)
    quarter = len(initial) // 4
    ambiguous[:quarter] = left
    ambiguous[quarter : 2 * quarter] = middle
    ambiguous[2 * quarter : 3 * quarter] = right
    ambiguous[3 * quarter :] = middle
    if find_two_mhd_fan_births(ambiguous, gamma) is not None:
        raise RuntimeError("ambiguous MHD data produced two fan births")


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
    reference_problem = interaction.make_symmetric_problem(gamma)
    validate_invariant_subsystem(
        reference_problem,
        gamma,
        args.magnetic_mass_ratio,
    )
    results: list[physical_mhd.Result] = []
    for cells in args.cells:
        effective_initial = interaction.cell_averages(
            reference_problem,
            cells,
            0.0,
            gamma,
        )
        initial = map_effective_averages(
            effective_initial,
            args.magnetic_mass_ratio,
        )
        validate_birth_rejections(initial, gamma)
        derived = build_state_derived_problem(
            initial,
            gamma,
            args.magnetic_mass_ratio,
        )
        if derived is None:
            raise RuntimeError("failed to derive two MHD fan births")
        problem, births = derived
        if (
            not math.isclose(
                births[0].position,
                reference_problem.left_position,
                rel_tol=0.0,
                abs_tol=3.0e-13,
            )
            or not math.isclose(
                births[1].position,
                reference_problem.right_position,
                rel_tol=0.0,
                abs_tol=3.0e-13,
            )
            or not math.isclose(
                problem.interaction_time,
                reference_problem.interaction_time,
                rel_tol=0.0,
                abs_tol=3.0e-13,
            )
        ):
            raise RuntimeError("state-derived MHD event recovery is incorrect")
        validate_invariant_subsystem(
            problem,
            gamma,
            args.magnetic_mass_ratio,
        )
        final_time = 0.5 * (
            problem.interaction_time + problem.next_interaction_time
        )
        effective_exact = interaction.cell_averages(
            reference_problem,
            cells,
            final_time,
            gamma,
        )
        exact = map_effective_averages(
            effective_exact,
            args.magnetic_mass_ratio,
        )
        print(
            "state_derived_event"
            f"\tcells={cells}"
            f"\tleft_origin={births[0].position:.12e}"
            f"\tright_origin={births[1].position:.12e}"
            f"\tleft_kind={births[0].location_kind}"
            f"\tright_kind={births[1].location_kind}"
            f"\tinteraction_time={problem.interaction_time:.12e}"
            f"\tnext_interaction_time={problem.next_interaction_time:.12e}"
            f"\tfinal_time={final_time:.12e}"
            "\tchild=exact_euler_solver_assisted"
        )
        print(
            "state_derived_birth_rejections"
            "\tconstant=passed\tdamaged=passed\tambiguous=passed"
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
                method=(
                    "causal_state_derived_two_fans_"
                    "exact_euler_child"
                ),
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
