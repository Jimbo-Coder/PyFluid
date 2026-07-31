#!/usr/bin/env python3
"""Test a causal multi-front Euler fan over an MP5/HLLC background.

An exact Riemann fan is initialized once from the shock-tube discontinuity.
Its shocks, contact, constant star regions, and outer plateau regions are then
carried causally in physical space.  At each face and timestep, the candidate
uses exact physical flux while that face lies in a constant region or is
crossed by a tracked discontinuity.  If the face lies inside a rarefaction,
the ordinary SSPRK3-integrated characteristic-MP5/HLLC flux retains ownership.

This is intentionally an oracle-assisted mechanism test.  It asks whether the
multi-front representation removes the one-front trade found by
``research_riemann_seeded_fronts.py``.  It does not solve front birth from
general evolving data or interactions between unrelated fans.
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

from pyfluid.exact_solutions import solve_sod_star_state  # noqa: E402

import research_physical_wave_subcell as physical  # noqa: E402
import research_riemann_seeded_fronts as seeded  # noqa: E402
import research_wave_reconstruction as base  # noqa: E402


Array = np.ndarray
HELD_OUT_CASES = (
    base.ShockCase(
        "reverse_blast",
        (1.0, 0.0, 0.01),
        (1.0, 0.0, 100.0),
        0.012,
    ),
    base.ShockCase(
        "colliding_shocks",
        (5.99924, 19.5975, 460.894),
        (5.99242, -6.19633, 46.095),
        0.006,
    ),
)
ALL_CASES = (*base.SHOCK_CASES, *HELD_OUT_CASES)


@dataclasses.dataclass(frozen=True)
class FanBundle:
    left: Array
    star_left: Array
    star_right: Array
    right: Array
    contact_speed: float
    left_is_shock: bool
    left_head_speed: float
    left_tail_speed: float
    right_is_shock: bool
    right_tail_speed: float
    right_head_speed: float

    def boundary_speeds(self) -> tuple[float, ...]:
        return (
            self.left_head_speed,
            self.left_tail_speed,
            self.contact_speed,
            self.right_tail_speed,
            self.right_head_speed,
        )

    def constant_state(self, similarity: float) -> Array | None:
        """Return the exact constant-region state, or None in a rarefaction."""
        if self.left_is_shock:
            if similarity < self.left_head_speed:
                return self.left
        else:
            if similarity < self.left_head_speed:
                return self.left
            if similarity <= self.left_tail_speed:
                return None
        if similarity < self.contact_speed:
            return self.star_left
        if self.right_is_shock:
            if similarity < self.right_head_speed:
                return self.star_right
            return self.right
        if similarity < self.right_tail_speed:
            return self.star_right
        if similarity <= self.right_head_speed:
            return None
        return self.right


@dataclasses.dataclass(frozen=True)
class BundleStats:
    exact_face_time_fraction: float
    rarefaction_face_time_fraction: float


def exact_fan_bundle(
    left: Array,
    right: Array,
    gamma: float,
) -> FanBundle:
    pressure_star, velocity_star = solve_sod_star_state(left, right, gamma)
    rho_left, velocity_left, pressure_left = left
    rho_right, velocity_right, pressure_right = right
    sound_left = math.sqrt(gamma * pressure_left / rho_left)
    sound_right = math.sqrt(gamma * pressure_right / rho_right)
    gamma_ratio = (gamma - 1.0) / (gamma + 1.0)

    def star_density(
        rho: float,
        pressure: float,
    ) -> float:
        ratio = pressure_star / pressure
        if pressure_star > pressure:
            return rho * (ratio + gamma_ratio) / (
                1.0 + gamma_ratio * ratio
            )
        return rho * ratio ** (1.0 / gamma)

    star_left = np.array(
        (
            star_density(rho_left, pressure_left),
            velocity_star,
            pressure_star,
        )
    )
    star_right = np.array(
        (
            star_density(rho_right, pressure_right),
            velocity_star,
            pressure_star,
        )
    )
    if pressure_star > pressure_left:
        left_head = velocity_left - sound_left * math.sqrt(
            (gamma + 1.0)
            * pressure_star
            / (2.0 * gamma * pressure_left)
            + (gamma - 1.0) / (2.0 * gamma)
        )
        left_tail = left_head
        left_is_shock = True
    else:
        sound_star_left = sound_left * (
            pressure_star / pressure_left
        ) ** ((gamma - 1.0) / (2.0 * gamma))
        left_head = velocity_left - sound_left
        left_tail = velocity_star - sound_star_left
        left_is_shock = False
    if pressure_star > pressure_right:
        right_head = velocity_right + sound_right * math.sqrt(
            (gamma + 1.0)
            * pressure_star
            / (2.0 * gamma * pressure_right)
            + (gamma - 1.0) / (2.0 * gamma)
        )
        right_tail = right_head
        right_is_shock = True
    else:
        sound_star_right = sound_right * (
            pressure_star / pressure_right
        ) ** ((gamma - 1.0) / (2.0 * gamma))
        right_tail = velocity_star + sound_star_right
        right_head = velocity_right + sound_right
        right_is_shock = False
    bundle = FanBundle(
        left=np.asarray(left, dtype=float),
        star_left=star_left,
        star_right=star_right,
        right=np.asarray(right, dtype=float),
        contact_speed=velocity_star,
        left_is_shock=left_is_shock,
        left_head_speed=left_head,
        left_tail_speed=left_tail,
        right_is_shock=right_is_shock,
        right_tail_speed=right_tail,
        right_head_speed=right_head,
    )
    for wave in seeded.exact_riemann_discontinuities(left, right, gamma):
        if wave.jump_residual > 3.0e-10:
            raise RuntimeError("fan bundle contains an inconsistent jump")
    return bundle


def blended_bundle_flux(
    *,
    baseline_flux: Array,
    face_position: float,
    discontinuity_position: float,
    time: float,
    dt: float,
    bundle: FanBundle,
    gamma: float,
) -> tuple[Array, float]:
    """Integrate exact constant regions and defer rarefactions to baseline."""
    distance = face_position - discontinuity_position
    cuts = [time, time + dt]
    for speed in bundle.boundary_speeds():
        if abs(speed) <= np.finfo(float).tiny:
            continue
        crossing = distance / speed
        if time < crossing < time + dt:
            cuts.append(crossing)
    cuts = sorted(set(cuts))
    integrated = np.zeros(3)
    exact_duration = 0.0
    for lower, upper in zip(cuts[:-1], cuts[1:], strict=True):
        midpoint = 0.5 * (lower + upper)
        if midpoint <= np.finfo(float).tiny:
            midpoint = max(0.5 * upper, np.finfo(float).tiny)
        similarity = distance / midpoint
        state = bundle.constant_state(similarity)
        duration = upper - lower
        if state is None:
            integrated += duration * baseline_flux
        else:
            integrated += duration * physical.euler_flux(state, gamma)
            exact_duration += duration
    return integrated / dt, exact_duration / dt


def evolve(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    bundle: FanBundle,
    *,
    discontinuity_position: float = 0.5,
) -> tuple[Array, bool, BundleStats]:
    state = base.conserved_from_primitive(primitive, gamma)
    cells = len(state)
    dx = 1.0 / cells
    time = 0.0
    exact_face_time = 0.0
    total_face_time = 0.0
    while time < final_time:
        current = base.primitive_from_conserved(state, gamma)
        if not base.primitive_is_admissible(current):
            return current, False, BundleStats(0.0, 1.0)
        rhs0, speed0, flux0 = physical.rhs_with_fluxes(
            state,
            dx,
            gamma,
            base.characteristic_mp5,
            False,
        )
        if not np.isfinite(speed0) or not np.all(np.isfinite(rhs0)):
            return current, False, BundleStats(0.0, 1.0)
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
            for face in range(cells + 1):
                integrated_flux[face], exact_fraction = blended_bundle_flux(
                    baseline_flux=integrated_flux[face],
                    face_position=face * dx,
                    discontinuity_position=discontinuity_position,
                    time=time,
                    dt=dt,
                    bundle=bundle,
                    gamma=gamma,
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
                BundleStats(0.0, 1.0),
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
        BundleStats(exact_fraction, 1.0 - exact_fraction),
    )


def print_result(result: base.Result, stats: BundleStats | None) -> None:
    values = []
    for field in dataclasses.fields(base.Result):
        value = getattr(result, field.name)
        values.append(f"{value:.12e}" if isinstance(value, float) else str(value))
    if stats is None:
        values.extend(("0.000000000000e+00", "0.000000000000e+00"))
    else:
        values.extend(
            (
                f"{stats.exact_face_time_fraction:.12e}",
                f"{stats.rarefaction_face_time_fraction:.12e}",
            )
        )
    print("\t".join(values))


def require_improvement(
    baseline: base.Result,
    candidate: base.Result,
) -> None:
    tolerance = 5.0e-12
    if not candidate.completed:
        raise RuntimeError(f"{candidate.case}: causal fan did not complete")
    if candidate.total_l1 >= baseline.total_l1:
        raise RuntimeError(f"{candidate.case}: causal fan did not improve L1")
    if (
        candidate.maximum_overshoot
        > baseline.maximum_overshoot + tolerance
        or candidate.maximum_tv_ratio
        > baseline.maximum_tv_ratio + tolerance
    ):
        raise RuntimeError(
            f"{candidate.case}: causal fan left the baseline Pareto front"
        )
    if max(
        candidate.mass_error,
        candidate.momentum_error,
        candidate.energy_error,
    ) > 5.0e-10:
        raise RuntimeError(
            f"{candidate.case}: causal fan violated conservation"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32,))
    parser.add_argument("--cfl", type=float, nargs="+", default=(0.9,))
    parser.add_argument(
        "--positions",
        type=float,
        nargs="+",
        default=(0.5,),
        help="initial Riemann-interface positions in the unit domain",
    )
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(case.name for case in ALL_CASES),
        default=tuple(case.name for case in ALL_CASES),
    )
    args = parser.parse_args()
    print(
        "\t".join(
            (
                *(field.name for field in dataclasses.fields(base.Result)),
                "exact_face_time_fraction",
                "rarefaction_face_time_fraction",
            )
        )
    )
    selected_cases = tuple(
        case for case in ALL_CASES if case.name in args.cases
    )
    for cells in args.cells:
        for cfl in args.cfl:
            for position in args.positions:
                if not 0.0 < position < 1.0:
                    raise ValueError("fan positions must lie inside the domain")
                for case in selected_cases:
                    initial = physical.discontinuity_cell_averages(
                        cells,
                        position,
                        np.asarray(case.left),
                        np.asarray(case.right),
                        args.gamma,
                    )
                    baseline, baseline_completed = base.evolve(
                        initial,
                        case.final_time,
                        cfl,
                        args.gamma,
                        base.characteristic_mp5,
                        base.hllc_flux,
                        periodic=False,
                    )
                    candidate, completed, stats = evolve(
                        initial,
                        case.final_time,
                        cfl,
                        args.gamma,
                        exact_fan_bundle(
                            np.asarray(case.left),
                            np.asarray(case.right),
                            args.gamma,
                        ),
                        discontinuity_position=position,
                    )
                    label = f"x0={position:.8g}"
                    baseline_result = base.shock_result(
                        case,
                        f"mp5_hllc_{label}",
                        cells,
                        baseline,
                        baseline_completed,
                        args.gamma,
                        position,
                    )
                    candidate_result = base.shock_result(
                        case,
                        f"causal_exact_fan_bundle_{label}",
                        cells,
                        candidate,
                        completed,
                        args.gamma,
                        position,
                    )
                    require_improvement(baseline_result, candidate_result)
                    print_result(baseline_result, None)
                    print_result(candidate_result, stats)


if __name__ == "__main__":
    main()
