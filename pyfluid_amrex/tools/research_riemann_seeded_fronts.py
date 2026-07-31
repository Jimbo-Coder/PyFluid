#!/usr/bin/env python3
"""Test exact-Riemann wave birth for the physical-front NRHD prototype.

Cycle 248 only recognized a discontinuity after two locally constant plateau
states surrounded one mixed cell.  This experiment follows Aguillon's
full-Euler discontinuous-reconstruction idea more closely: solve the exact
Riemann problem between the neighboring cell averages, extract its physical
left shock, contact, and right shock, and ask whether the current conserved
cell average lies on one of those exact wave segments.

Two policies are compared:

* ``interior`` accepts only a front already lying strictly inside a cell.
* ``endpoint_seed`` may create a front at a cell face from an exact Riemann
  fan.  This is more aggressive and exposes whether one-wave birth is
  consistent when the same fan also contains other waves.

Characteristic MP5, HLLC, SSPRK3, and the CFL calculation remain fixed.  Only
the complete-timestep flux at faces affected by an accepted physical wave is
replaced with its crossing-time average.  This is a research falsification
harness, not a production reconstruction.
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
import research_wave_reconstruction as base  # noqa: E402


Array = np.ndarray


@dataclasses.dataclass(frozen=True)
class EvolutionStats:
    accepted_cell_steps: int
    maximum_simultaneous_fronts: int


def exact_riemann_discontinuities(
    left: Array,
    right: Array,
    gamma: float,
) -> tuple[physical.PhysicalWave, ...]:
    """Return the shock/contact discontinuities in one exact Euler fan."""
    endpoints = np.array((left, right), dtype=float)
    if not base.primitive_is_admissible(endpoints):
        return ()
    rho_left, velocity_left, pressure_left = left
    rho_right, velocity_right, pressure_right = right
    sound_left = math.sqrt(gamma * pressure_left / rho_left)
    sound_right = math.sqrt(gamma * pressure_right / rho_right)
    vacuum_limit = 2.0 * (sound_left + sound_right) / (gamma - 1.0)
    if velocity_right - velocity_left >= vacuum_limit:
        return ()
    pressure_star, velocity_star = solve_sod_star_state(
        left,
        right,
        gamma,
    )
    if not np.isfinite(pressure_star + velocity_star) or pressure_star <= 0.0:
        return ()
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
    candidates: list[physical.PhysicalWave] = []
    pressure_tolerance = 128.0 * np.finfo(float).eps
    if pressure_star > pressure_left * (1.0 + pressure_tolerance):
        wave = physical.fit_physical_wave(
            left,
            left,
            star_left,
            gamma,
            allow_endpoint=True,
        )
        if wave is not None:
            candidates.append(wave)
    contact_scale = max(
        abs(star_left[0]),
        abs(star_right[0]),
        np.finfo(float).tiny,
    )
    if abs(star_left[0] - star_right[0]) > 3.0e-12 * contact_scale:
        wave = physical.fit_physical_wave(
            star_left,
            star_left,
            star_right,
            gamma,
            allow_endpoint=True,
        )
        if wave is not None:
            candidates.append(wave)
    if pressure_star > pressure_right * (1.0 + pressure_tolerance):
        wave = physical.fit_physical_wave(
            star_right,
            star_right,
            right,
            gamma,
            allow_endpoint=True,
        )
        if wave is not None:
            candidates.append(wave)
    return tuple(candidates)


def fit_riemann_cell_wave(
    primitive: Array,
    cell: int,
    gamma: float,
    periodic: bool,
    *,
    allow_endpoint: bool,
) -> physical.PhysicalWave | None:
    left = physical.local_primitive(primitive, cell - 1, periodic)
    average = physical.local_primitive(primitive, cell, periodic)
    right = physical.local_primitive(primitive, cell + 1, periodic)
    accepted: list[physical.PhysicalWave] = []
    for candidate in exact_riemann_discontinuities(left, right, gamma):
        fitted = physical.fit_physical_wave(
            average,
            candidate.left,
            candidate.right,
            gamma,
            allow_endpoint=allow_endpoint,
        )
        if fitted is not None:
            accepted.append(fitted)
    if not accepted:
        return None

    def rank(wave: physical.PhysicalWave) -> tuple[float, float]:
        endpoint_distance = min(
            wave.left_fraction,
            1.0 - wave.left_fraction,
        )
        state_jump = base.conserved_from_primitive(
            np.array((wave.left, wave.right)),
            gamma,
        )
        strength = float(np.linalg.norm(state_jump[0] - state_jump[1]))
        return endpoint_distance, strength

    return max(accepted, key=rank)


def physical_flux_overrides(
    primitive: Array,
    dx: float,
    dt: float,
    gamma: float,
    periodic: bool,
    *,
    allow_endpoint: bool,
) -> tuple[dict[int, Array], int]:
    """Integrate every accepted Riemann-seeded cell wave over one step."""
    cells = len(primitive)
    overrides: dict[int, Array] = {}
    conflicts: set[int] = set()
    accepted = 0

    def face_index(face: int) -> int | None:
        if periodic:
            return face % cells
        return face if 0 <= face <= cells else None

    def add(face: int, value: Array) -> None:
        resolved = face_index(face)
        if resolved is None or resolved in conflicts:
            return
        previous = overrides.get(resolved)
        if previous is None:
            overrides[resolved] = value.copy()
            return
        tolerance = 3.0e-10 * max(
            float(np.linalg.norm(previous)),
            float(np.linalg.norm(value)),
            1.0,
        )
        if np.linalg.norm(previous - value) > tolerance:
            overrides.pop(resolved, None)
            conflicts.add(resolved)

    def add_plateau(face: int, state: Array, value: Array) -> None:
        left = physical.local_primitive(primitive, face - 1, periodic)
        right = physical.local_primitive(primitive, face, periodic)
        if physical.states_agree(left, state, gamma) and physical.states_agree(
            right,
            state,
            gamma,
        ):
            add(face, value)

    for cell in range(cells):
        wave = fit_riemann_cell_wave(
            primitive,
            cell,
            gamma,
            periodic,
            allow_endpoint=allow_endpoint,
        )
        if wave is None:
            continue
        accepted += 1
        left_flux = physical.euler_flux(wave.left, gamma)
        right_flux = physical.euler_flux(wave.right, gamma)
        left_average = left_flux
        right_average = right_flux
        if wave.speed > 0.0:
            crossing = (1.0 - wave.left_fraction) * dx / wave.speed
            if crossing < dt:
                crossed = min(max(1.0 - crossing / dt, 0.0), 1.0)
                right_average = (
                    (1.0 - crossed) * right_flux + crossed * left_flux
                )
        elif wave.speed < 0.0:
            crossing = wave.left_fraction * dx / (-wave.speed)
            if crossing < dt:
                crossed = min(max(1.0 - crossing / dt, 0.0), 1.0)
                left_average = (
                    (1.0 - crossed) * left_flux + crossed * right_flux
                )
        add(cell, left_average)
        add(cell + 1, right_average)
        for offset in range(1, 4):
            add_plateau(cell - offset, wave.left, left_flux)
            add_plateau(cell + 1 + offset, wave.right, right_flux)
    if periodic and 0 in overrides:
        overrides[cells] = overrides[0].copy()
    return overrides, accepted


def evolve(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    *,
    periodic: bool,
    allow_endpoint: bool,
) -> tuple[Array, bool, EvolutionStats]:
    state = base.conserved_from_primitive(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    accepted_cell_steps = 0
    maximum_fronts = 0
    while time < final_time:
        current = base.primitive_from_conserved(state, gamma)
        if not base.primitive_is_admissible(current):
            return current, False, EvolutionStats(
                accepted_cell_steps,
                maximum_fronts,
            )
        rhs0, speed0, flux0 = physical.rhs_with_fluxes(
            state,
            dx,
            gamma,
            base.characteristic_mp5,
            periodic,
        )
        if not np.isfinite(speed0) or not np.all(np.isfinite(rhs0)):
            return current, False, EvolutionStats(
                accepted_cell_steps,
                maximum_fronts,
            )
        dt = min(final_time - time, cfl * dx / speed0)
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1, flux1 = physical.rhs_with_fluxes(
                stage1,
                dx,
                gamma,
                base.characteristic_mp5,
                periodic,
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
                periodic,
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
            overrides, accepted = physical_flux_overrides(
                current,
                dx,
                dt,
                gamma,
                periodic,
                allow_endpoint=allow_endpoint,
            )
            for face, face_flux in overrides.items():
                integrated_flux[face] = face_flux
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
            accepted_cell_steps += accepted
            maximum_fronts = max(maximum_fronts, accepted)
            break
        else:
            return (
                base.primitive_from_conserved(state, gamma),
                False,
                EvolutionStats(accepted_cell_steps, maximum_fronts),
            )
        time += dt
    final = base.primitive_from_conserved(state, gamma)
    return (
        final,
        base.primitive_is_admissible(final),
        EvolutionStats(accepted_cell_steps, maximum_fronts),
    )


def shock_initial(case: base.ShockCase, cells: int) -> Array:
    x = (np.arange(cells) + 0.5) / cells
    return np.where(
        (x < 0.5)[:, None],
        np.asarray(case.left)[None, :],
        np.asarray(case.right)[None, :],
    )


def print_result(result: base.Result, stats: EvolutionStats | None) -> None:
    values = []
    for field in dataclasses.fields(base.Result):
        value = getattr(result, field.name)
        values.append(f"{value:.12e}" if isinstance(value, float) else str(value))
    if stats is None:
        values.extend(("0", "0"))
    else:
        values.extend(
            (
                str(stats.accepted_cell_steps),
                str(stats.maximum_simultaneous_fronts),
            )
        )
    print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32,))
    parser.add_argument("--cfl", type=float, nargs="+", default=(0.9,))
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(case.name for case in base.SHOCK_CASES),
        default=tuple(case.name for case in base.SHOCK_CASES),
    )
    parser.add_argument("--skip-smooth", action="store_true")
    args = parser.parse_args()
    fields = [field.name for field in dataclasses.fields(base.Result)]
    print(
        "\t".join(
            (
                *fields,
                "accepted_cell_steps",
                "maximum_simultaneous_fronts",
            )
        )
    )
    selected_cases = tuple(
        case for case in base.SHOCK_CASES if case.name in args.cases
    )
    for cells in args.cells:
        for cfl in args.cfl:
            for case in selected_cases:
                initial = shock_initial(case, cells)
                baseline, baseline_completed = base.evolve(
                    initial,
                    case.final_time,
                    cfl,
                    args.gamma,
                    base.characteristic_mp5,
                    base.hllc_flux,
                    periodic=False,
                )
                print_result(
                    base.shock_result(
                        case,
                        "mp5_hllc",
                        cells,
                        baseline,
                        baseline_completed,
                        args.gamma,
                    ),
                    None,
                )
                for method, allow_endpoint in (
                    ("riemann_interior_front", False),
                    ("riemann_endpoint_seed", True),
                ):
                    candidate, completed, stats = evolve(
                        initial,
                        case.final_time,
                        cfl,
                        args.gamma,
                        periodic=False,
                        allow_endpoint=allow_endpoint,
                    )
                    print_result(
                        base.shock_result(
                            case,
                            method,
                            cells,
                            candidate,
                            completed,
                            args.gamma,
                        ),
                        stats,
                    )

            if args.skip_smooth:
                continue
            x = (np.arange(cells) + 0.5) / cells
            smooth = np.column_stack(
                (
                    1.0 + 0.2 * np.sin(2.0 * math.pi * x),
                    np.full(cells, 0.5),
                    np.ones(cells),
                )
            )
            smooth_baseline, baseline_completed = base.evolve(
                smooth,
                2.0,
                cfl,
                args.gamma,
                base.characteristic_mp5,
                base.hllc_flux,
                periodic=True,
            )
            for allow_endpoint in (False, True):
                smooth_candidate, completed, stats = evolve(
                    smooth,
                    2.0,
                    cfl,
                    args.gamma,
                    periodic=True,
                    allow_endpoint=allow_endpoint,
                )
                maximum_difference = float(
                    np.max(np.abs(smooth_candidate - smooth_baseline))
                )
                if (
                    not baseline_completed
                    or not completed
                    or maximum_difference > 3.0e-11
                ):
                    raise RuntimeError(
                        "Riemann-seeded front changed the smooth path: "
                        f"{maximum_difference:.12e}"
                    )
                print(
                    "smooth_path_audit"
                    f"\tcells={cells}"
                    f"\tcfl={cfl:g}"
                    f"\tallow_endpoint={int(allow_endpoint)}"
                    f"\tmaximum_difference={maximum_difference:.12e}"
                    f"\taccepted_cell_steps={stats.accepted_cell_steps}"
                )


if __name__ == "__main__":
    main()
