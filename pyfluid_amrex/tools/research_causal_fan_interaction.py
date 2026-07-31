#!/usr/bin/env python3
"""Test causal deletion and rebirth when two exact Euler fans interact.

Two symmetric blast interfaces launch shocks into a low-pressure middle
state.  The two incoming shocks meet at an analytically known event.  Their
obsolete fronts are clipped there and a new exact Riemann fan is born from
the two incident post-shock states.  The original outer waves continue until
the new shocks reach their contacts, which defines the next unresolved event.

Exact physical flux owns constant regions and shock crossings.  The existing
characteristic-MP5/HLLC flux continues to own rarefaction interiors.  This is
a deliberately bounded front-lifecycle experiment, not a general interaction
solver.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyfluid.exact_solutions import sod_exact_solution  # noqa: E402

from exact_reference import cell_average_newtonian_hydro  # noqa: E402
import research_causal_fan_bundle as fan  # noqa: E402
import research_physical_wave_subcell as physical  # noqa: E402
import research_wave_reconstruction as base  # noqa: E402


Array = np.ndarray
TIME_NODES, TIME_WEIGHTS = np.polynomial.legendre.leggauss(8)


@dataclasses.dataclass(frozen=True)
class InteractionProblem:
    left: Array
    middle: Array
    right: Array
    left_position: float
    right_position: float
    left_fan: fan.FanBundle
    right_fan: fan.FanBundle
    interaction_time: float
    interaction_position: float
    child_fan: fan.FanBundle
    next_interaction_time: float


def make_symmetric_problem(gamma: float) -> InteractionProblem:
    left = np.asarray((1.0, 0.0, 100.0))
    middle = np.asarray((1.0, 0.0, 0.01))
    right = left.copy()
    left_position = 0.35
    right_position = 0.65
    left_fan = fan.exact_fan_bundle(left, middle, gamma)
    right_fan = fan.exact_fan_bundle(middle, right, gamma)
    closing_speed = (
        left_fan.right_head_speed - right_fan.left_head_speed
    )
    if (
        not left_fan.right_is_shock
        or not right_fan.left_is_shock
        or closing_speed <= 0.0
    ):
        raise RuntimeError("test problem does not launch two closing shocks")
    interaction_time = (
        right_position - left_position
    ) / closing_speed
    left_collision_position = (
        left_position
        + left_fan.right_head_speed * interaction_time
    )
    right_collision_position = (
        right_position
        + right_fan.left_head_speed * interaction_time
    )
    if not math.isclose(
        left_collision_position,
        right_collision_position,
        rel_tol=0.0,
        abs_tol=2.0e-14,
    ):
        raise RuntimeError("incident shock trajectories do not meet")
    interaction_position = 0.5 * (
        left_collision_position + right_collision_position
    )
    child_fan = fan.exact_fan_bundle(
        left_fan.star_right,
        right_fan.star_left,
        gamma,
    )
    left_denominator = (
        left_fan.contact_speed - child_fan.left_head_speed
    )
    right_denominator = (
        child_fan.right_head_speed - right_fan.contact_speed
    )
    left_next = (
        interaction_position
        - child_fan.left_head_speed * interaction_time
        - left_position
    ) / left_denominator
    right_next = (
        right_position
        - interaction_position
        + child_fan.right_head_speed * interaction_time
    ) / right_denominator
    next_interaction_time = min(left_next, right_next)
    if next_interaction_time <= interaction_time:
        raise RuntimeError("new fan has no valid post-collision interval")
    return InteractionProblem(
        left=left,
        middle=middle,
        right=right,
        left_position=left_position,
        right_position=right_position,
        left_fan=left_fan,
        right_fan=right_fan,
        interaction_time=interaction_time,
        interaction_position=interaction_position,
        child_fan=child_fan,
        next_interaction_time=next_interaction_time,
    )


def initial_state(problem: InteractionProblem, position: float) -> Array:
    if position < problem.left_position:
        return problem.left
    if position < problem.right_position:
        return problem.middle
    return problem.right


def inside_fan(
    bundle: fan.FanBundle,
    position: float,
    origin: float,
    age: float,
) -> bool:
    if age <= 0.0:
        return False
    lower = origin + bundle.left_head_speed * age
    upper = origin + bundle.right_head_speed * age
    return lower <= position <= upper


def constant_state_at(
    problem: InteractionProblem,
    position: float,
    time: float,
) -> Array | None:
    """Return a constant exact state, or None in a rarefaction."""
    if time >= problem.interaction_time:
        child_age = time - problem.interaction_time
        if inside_fan(
            problem.child_fan,
            position,
            problem.interaction_position,
            child_age,
        ):
            return problem.child_fan.constant_state(
                (position - problem.interaction_position) / child_age
            )
    if position <= problem.interaction_position:
        selected_fan = problem.left_fan
        origin = problem.left_position
    else:
        selected_fan = problem.right_fan
        origin = problem.right_position
    if inside_fan(selected_fan, position, origin, time):
        return selected_fan.constant_state((position - origin) / time)
    return initial_state(problem, position)


def exact_state_at(
    problem: InteractionProblem,
    position: float,
    time: float,
    gamma: float,
) -> Array:
    """Return the exact primitive state before the second interaction."""
    if time >= problem.interaction_time:
        child_age = time - problem.interaction_time
        if inside_fan(
            problem.child_fan,
            position,
            problem.interaction_position,
            child_age,
        ):
            rho, pressure, velocity = sod_exact_solution(
                np.asarray((position,)),
                child_age,
                gamma,
                problem.child_fan.left,
                problem.child_fan.right,
                problem.interaction_position,
            )
            return np.asarray((rho[0], velocity[0], pressure[0]))
    if position <= problem.interaction_position:
        selected_fan = problem.left_fan
        origin = problem.left_position
    else:
        selected_fan = problem.right_fan
        origin = problem.right_position
    if inside_fan(selected_fan, position, origin, time):
        rho, pressure, velocity = sod_exact_solution(
            np.asarray((position,)),
            time,
            gamma,
            selected_fan.left,
            selected_fan.right,
            origin,
        )
        return np.asarray((rho[0], velocity[0], pressure[0]))
    return initial_state(problem, position)


def linear_bundle_state(
    bundle: fan.FanBundle,
    similarity: float,
) -> Array:
    state = bundle.constant_state(similarity)
    if state is not None:
        return state
    if similarity <= bundle.left_tail_speed:
        fraction = (
            similarity - bundle.left_head_speed
        ) / (bundle.left_tail_speed - bundle.left_head_speed)
        return (
            (1.0 - fraction) * bundle.left
            + fraction * bundle.star_left
        )
    fraction = (
        similarity - bundle.right_tail_speed
    ) / (bundle.right_head_speed - bundle.right_tail_speed)
    return (
        (1.0 - fraction) * bundle.star_right
        + fraction * bundle.right
    )


def linear_state_at(
    problem: InteractionProblem,
    position: float,
    time: float,
) -> Array:
    """Return a cheap primitive-linear approximation inside rarefactions."""
    if time >= problem.interaction_time:
        child_age = time - problem.interaction_time
        if inside_fan(
            problem.child_fan,
            position,
            problem.interaction_position,
            child_age,
        ):
            return linear_bundle_state(
                problem.child_fan,
                (position - problem.interaction_position) / child_age,
            )
    if position <= problem.interaction_position:
        selected_fan = problem.left_fan
        origin = problem.left_position
    else:
        selected_fan = problem.right_fan
        origin = problem.right_position
    if inside_fan(selected_fan, position, origin, time):
        return linear_bundle_state(
            selected_fan,
            (position - origin) / time,
        )
    return initial_state(problem, position)


def crossing_times(
    problem: InteractionProblem,
    position: float,
    time: float,
    dt: float,
) -> list[float]:
    cuts = [time, time + dt]
    if time < problem.interaction_time < time + dt:
        cuts.append(problem.interaction_time)
    for origin, start_time, bundle in (
        (problem.left_position, 0.0, problem.left_fan),
        (problem.right_position, 0.0, problem.right_fan),
        (
            problem.interaction_position,
            problem.interaction_time,
            problem.child_fan,
        ),
    ):
        for speed in bundle.boundary_speeds():
            if abs(speed) <= np.finfo(float).tiny:
                continue
            crossing = start_time + (position - origin) / speed
            if max(time, start_time) < crossing < time + dt:
                cuts.append(crossing)
    return sorted(set(cuts))


def blended_interaction_flux(
    *,
    baseline_flux: Array,
    face_position: float,
    time: float,
    dt: float,
    problem: InteractionProblem,
    gamma: float,
    rarefaction_mode: str,
) -> tuple[Array, float]:
    integrated = np.zeros(3)
    exact_duration = 0.0
    cuts = crossing_times(problem, face_position, time, dt)
    for lower, upper in zip(cuts[:-1], cuts[1:], strict=True):
        midpoint = 0.5 * (lower + upper)
        state = constant_state_at(problem, face_position, midpoint)
        duration = upper - lower
        if state is None:
            if rarefaction_mode != "baseline":
                half_duration = 0.5 * duration
                quadrature_times = (
                    midpoint + half_duration * TIME_NODES
                )
                for quadrature_time, weight in zip(
                    quadrature_times,
                    TIME_WEIGHTS,
                    strict=True,
                ):
                    if rarefaction_mode == "exact":
                        quadrature_state = exact_state_at(
                            problem,
                            face_position,
                            float(quadrature_time),
                            gamma,
                        )
                    else:
                        quadrature_state = linear_state_at(
                            problem,
                            face_position,
                            float(quadrature_time),
                        )
                    integrated += (
                        half_duration
                        * weight
                        * physical.euler_flux(quadrature_state, gamma)
                    )
            else:
                integrated += duration * baseline_flux
        else:
            integrated += duration * physical.euler_flux(state, gamma)
        if state is not None or rarefaction_mode != "baseline":
            exact_duration += duration
    return integrated / dt, exact_duration / dt


def evolve(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    problem: InteractionProblem,
    *,
    rarefaction_mode: str = "baseline",
) -> tuple[Array, bool, fan.BundleStats]:
    if rarefaction_mode not in {"baseline", "linear", "exact"}:
        raise ValueError(f"unsupported rarefaction mode {rarefaction_mode}")
    state = base.conserved_from_primitive(primitive, gamma)
    cells = len(state)
    dx = 1.0 / cells
    time = 0.0
    exact_face_time = 0.0
    total_face_time = 0.0
    while time < final_time:
        current = base.primitive_from_conserved(state, gamma)
        if not base.primitive_is_admissible(current):
            return current, False, fan.BundleStats(0.0, 1.0)
        rhs0, speed0, flux0 = physical.rhs_with_fluxes(
            state,
            dx,
            gamma,
            base.characteristic_mp5,
            False,
        )
        if not np.isfinite(speed0) or not np.all(np.isfinite(rhs0)):
            return current, False, fan.BundleStats(0.0, 1.0)
        dt = min(final_time - time, cfl * dx / speed0)
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1, flux1 = physical.rhs_with_fluxes(
                stage1,
                dx,
                gamma,
                base.characteristic_mp5,
                False,
            )
            if not np.isfinite(speed1):
                dt *= 0.5
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, speed2, flux2 = physical.rhs_with_fluxes(
                stage2,
                dx,
                gamma,
                base.characteristic_mp5,
                False,
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
            step_exact = 0.0
            for face_index in range(cells + 1):
                integrated_flux[face_index], exact_fraction = (
                    blended_interaction_flux(
                        baseline_flux=integrated_flux[face_index],
                        face_position=face_index * dx,
                        time=time,
                        dt=dt,
                        problem=problem,
                        gamma=gamma,
                        rarefaction_mode=rarefaction_mode,
                    )
                )
                step_exact += exact_fraction
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
            exact_face_time += dt * step_exact
            total_face_time += dt * (cells + 1)
            break
        else:
            return (
                base.primitive_from_conserved(state, gamma),
                False,
                fan.BundleStats(0.0, 1.0),
            )
        time += dt
    final = base.primitive_from_conserved(state, gamma)
    exact_fraction = exact_face_time / max(
        total_face_time,
        np.finfo(float).tiny,
    )
    return (
        final,
        base.primitive_is_admissible(final),
        fan.BundleStats(exact_fraction, 1.0 - exact_fraction),
    )


def cell_averages(
    problem: InteractionProblem,
    cells: int,
    time: float,
    gamma: float,
) -> Array:
    edges = np.linspace(0.0, 1.0, cells + 1)
    breakpoints = [problem.left_position, problem.right_position]
    for origin, age, bundle in (
        (problem.left_position, time, problem.left_fan),
        (problem.right_position, time, problem.right_fan),
        (
            problem.interaction_position,
            max(0.0, time - problem.interaction_time),
            problem.child_fan,
        ),
    ):
        breakpoints.extend(
            origin + speed * age for speed in bundle.boundary_speeds()
        )

    def primitive(points: Array) -> tuple[Array, Array, Array]:
        states = np.asarray(
            [
                exact_state_at(problem, float(point), time, gamma)
                for point in points
            ]
        )
        return states[:, 0], states[:, 2], states[:, 1]

    rho, pressure, velocity = cell_average_newtonian_hydro(
        primitive,
        edges,
        gamma,
        breakpoints=breakpoints,
    )
    return np.column_stack((rho, velocity, pressure))


def result(
    *,
    method: str,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
) -> base.Result:
    errors = []
    overshoots = []
    tv_ratios = []
    for component in range(3):
        values = evolved[:, component]
        reference = exact[:, component]
        scale = max(float(np.ptp(reference)), 1.0e-14)
        errors.append(float(np.mean(np.abs(values - reference))) / scale)
        overshoots.append(
            max(
                0.0,
                float(np.max(values) - np.max(reference)),
                float(np.min(reference) - np.min(values)),
            )
            / scale
        )
        tv_ratios.append(
            base.total_variation(values)
            / max(base.total_variation(reference), 1.0e-14)
        )
    integral_error = np.abs(
        np.mean(base.conserved_from_primitive(evolved, gamma), axis=0)
        - np.mean(base.conserved_from_primitive(exact, gamma), axis=0)
    )
    return base.Result(
        case="symmetric_shock_collision",
        method=method,
        cells=len(evolved),
        completed=completed,
        total_l1=sum(errors),
        maximum_overshoot=max(overshoots),
        maximum_tv_ratio=max(tv_ratios),
        mass_error=float(integral_error[0]),
        momentum_error=float(integral_error[1]),
        energy_error=float(integral_error[2]),
        minimum_rho=float(np.min(evolved[:, 0])),
        minimum_pressure=float(np.min(evolved[:, 2])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--cfl", type=float, nargs="+", default=(0.9,))
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument(
        "--final-time",
        type=float,
        help="must fall after shock collision and before the next interaction",
    )
    args = parser.parse_args()
    problem = make_symmetric_problem(args.gamma)
    default_final_time = 0.5 * (
        problem.interaction_time + problem.next_interaction_time
    )
    final_time = (
        default_final_time if args.final_time is None else args.final_time
    )
    if not (
        problem.interaction_time
        < final_time
        < problem.next_interaction_time
    ):
        raise ValueError(
            "final time must be inside the one-rebirth validity interval "
            f"({problem.interaction_time:.12g}, "
            f"{problem.next_interaction_time:.12g})"
        )
    print(
        "event"
        f"\tinteraction_time={problem.interaction_time:.12e}"
        f"\tnext_interaction_time={problem.next_interaction_time:.12e}"
        f"\tfinal_time={final_time:.12e}"
    )
    print(
        "\t".join(
            (
                *(field.name for field in dataclasses.fields(base.Result)),
                "exact_face_time_fraction",
                "rarefaction_face_time_fraction",
            )
        )
    )
    for cells in args.cells:
        initial = cell_averages(problem, cells, 0.0, args.gamma)
        exact = cell_averages(problem, cells, final_time, args.gamma)
        for cfl in args.cfl:
            baseline, baseline_completed = base.evolve(
                initial,
                final_time,
                cfl,
                args.gamma,
                base.characteristic_mp5,
                base.hllc_flux,
                periodic=False,
            )
            candidate, candidate_completed, stats = evolve(
                initial,
                final_time,
                cfl,
                args.gamma,
                problem,
            )
            full_fan, full_fan_completed, full_fan_stats = evolve(
                initial,
                final_time,
                cfl,
                args.gamma,
                problem,
                rarefaction_mode="exact",
            )
            linear_fan, linear_fan_completed, linear_fan_stats = evolve(
                initial,
                final_time,
                cfl,
                args.gamma,
                problem,
                rarefaction_mode="linear",
            )
            baseline_result = result(
                method=f"mp5_hllc_cfl={cfl:.8g}",
                evolved=baseline,
                exact=exact,
                completed=baseline_completed,
                gamma=args.gamma,
            )
            candidate_result = result(
                method=f"causal_rebirth_cfl={cfl:.8g}",
                evolved=candidate,
                exact=exact,
                completed=candidate_completed,
                gamma=args.gamma,
            )
            linear_result = result(
                method=f"causal_linear_fan_cfl={cfl:.8g}",
                evolved=linear_fan,
                exact=exact,
                completed=linear_fan_completed,
                gamma=args.gamma,
            )
            full_fan_result = result(
                method=f"causal_full_fan_cfl={cfl:.8g}",
                evolved=full_fan,
                exact=exact,
                completed=full_fan_completed,
                gamma=args.gamma,
            )
            if (
                not full_fan_result.completed
                or full_fan_result.total_l1 >= baseline_result.total_l1
                or full_fan_result.maximum_overshoot
                > baseline_result.maximum_overshoot + 5.0e-12
                or full_fan_result.maximum_tv_ratio
                > baseline_result.maximum_tv_ratio + 5.0e-12
                or max(
                    full_fan_result.mass_error,
                    full_fan_result.momentum_error,
                    full_fan_result.energy_error,
                )
                > 5.0e-10
            ):
                raise RuntimeError(
                    "complete causal fan failed its interaction gate"
                )
            if linear_result.total_l1 <= full_fan_result.total_l1:
                raise RuntimeError(
                    "primitive-linear rarefaction falsification disappeared"
                )
            fan.print_result(
                baseline_result,
                None,
            )
            fan.print_result(
                candidate_result,
                stats,
            )
            fan.print_result(
                linear_result,
                linear_fan_stats,
            )
            fan.print_result(
                full_fan_result,
                full_fan_stats,
            )


if __name__ == "__main__":
    main()
