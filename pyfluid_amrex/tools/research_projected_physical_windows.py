#!/usr/bin/env python3
"""Project broadened NRMHD transitions onto a conservative physical wave.

This offline experiment probes the missing activation step in the
time-integrated physical-front prototype.  It starts from an evolved
MP5+HLLD profile, identifies a nearly one-wave window, and jointly adjusts its
two plateau states, wave speed, and occupied fraction so that

* one Rankine--Hugoniot speed fits every ideal-MHD conserved component, and
* one convex combination of the endpoint states reproduces the window total.

The projected window is then condensed to plateau cells plus at most one mixed
cell.  This is a diagnostic of the representation, not an evolution method:
postprocessing with the final numerical profile is unavailable during a real
timestep and cannot establish causality, stability, or convergence.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_physical_wave_subcell_mhd as physical  # noqa: E402


Array = np.ndarray


@dataclasses.dataclass(frozen=True)
class ProjectedWindow:
    start: int
    stop: int
    left: Array
    right: Array
    speed: float
    left_fraction: float
    jump_residual: float
    mixture_residual: float
    endpoint_shift: float


def encoded_primitive(primitive: Array) -> Array:
    return np.array(
        (
            math.log(primitive[0]),
            primitive[1],
            primitive[2],
            primitive[3],
            math.log(primitive[4]),
            primitive[6],
            primitive[7],
        )
    )


def decoded_primitive(encoded: Array, normal_b: float) -> Array:
    return np.array(
        (
            math.exp(encoded[0]),
            encoded[1],
            encoded[2],
            encoded[3],
            math.exp(encoded[4]),
            normal_b,
            encoded[5],
            encoded[6],
        )
    )


def component_scale(*values: Array) -> Array:
    stacked = np.abs(np.stack(values))
    scale = np.max(stacked, axis=0)
    common = max(float(np.max(scale)), np.finfo(float).tiny)
    return np.maximum(scale, 1.0e-8 * common)


def project_window(
    primitive: Array,
    fit: physical.WindowFit,
    gamma: float,
) -> ProjectedWindow | None:
    state = physical.mhd.conserved_array(primitive, gamma)
    left_initial = primitive[fit.start - 1].copy()
    right_initial = primitive[fit.stop].copy()
    normal_b = 0.5 * (left_initial[5] + right_initial[5])
    left_initial[5] = normal_b
    right_initial[5] = normal_b
    left_state, right_state = physical.mhd.conserved_array(
        np.array((left_initial, right_initial)),
        gamma,
    )
    zone_mean = np.mean(state[fit.start : fit.stop], axis=0)
    state_jump = left_state - right_state
    jump_norm2 = float(np.dot(state_jump, state_jump))
    if jump_norm2 <= np.finfo(float).tiny:
        return None
    left_fraction = float(
        np.dot(zone_mean - right_state, state_jump) / jump_norm2
    )
    if not 0.0 < left_fraction < 1.0:
        return None

    left_encoded = encoded_primitive(left_initial)
    right_encoded = encoded_primitive(right_initial)
    endpoint_jump = np.abs(left_encoded - right_encoded)
    endpoint_scale = np.maximum(
        np.maximum(np.abs(left_encoded), np.abs(right_encoded)),
        1.0,
    )
    half_width = 0.38 * endpoint_jump + 0.01 * endpoint_scale
    lower = np.concatenate(
        (
            left_encoded - half_width,
            right_encoded - half_width,
            [fit.speed - max(abs(fit.speed), 1.0), 1.0e-4],
        )
    )
    upper = np.concatenate(
        (
            left_encoded + half_width,
            right_encoded + half_width,
            [fit.speed + max(abs(fit.speed), 1.0), 1.0 - 1.0e-4],
        )
    )
    initial = np.concatenate(
        (
            left_encoded,
            right_encoded,
            [fit.speed, left_fraction],
        )
    )
    initial = np.minimum(np.maximum(initial, lower), upper)
    initial_left_flux = physical.mhd_flux(left_initial, gamma)
    initial_right_flux = physical.mhd_flux(right_initial, gamma)
    state_scale = component_scale(left_state, right_state, zone_mean)
    flux_scale = component_scale(
        initial_left_flux,
        initial_right_flux,
        fit.speed * state_jump,
    )

    def unpack(unknown: Array) -> tuple[Array, Array, float, float]:
        return (
            decoded_primitive(unknown[:7], normal_b),
            decoded_primitive(unknown[7:14], normal_b),
            float(unknown[14]),
            float(unknown[15]),
        )

    def residual(unknown: Array) -> Array:
        left, right, speed, fraction = unpack(unknown)
        endpoints = physical.mhd.conserved_array(
            np.array((left, right)),
            gamma,
        )
        left_cons, right_cons = endpoints
        mixture = fraction * left_cons + (1.0 - fraction) * right_cons
        jump = (
            physical.mhd_flux(left, gamma)
            - physical.mhd_flux(right, gamma)
            - speed * (left_cons - right_cons)
        )
        return np.concatenate(
            (
                jump / flux_scale,
                (mixture - zone_mean) / state_scale,
            )
        )

    solution = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=4000,
    )
    left, right, speed, fraction = unpack(solution.x)
    endpoint_states = physical.mhd.conserved_array(
        np.array((left, right)),
        gamma,
    )
    projected_jump = endpoint_states[0] - endpoint_states[1]
    flux_jump = (
        physical.mhd_flux(left, gamma)
        - physical.mhd_flux(right, gamma)
    )
    jump_residual = physical.scaled_residual(
        flux_jump - speed * projected_jump,
        flux_jump,
        speed * projected_jump,
    )
    projected_mean = (
        fraction * endpoint_states[0]
        + (1.0 - fraction) * endpoint_states[1]
    )
    mixture_residual = physical.scaled_residual(
        projected_mean - zone_mean,
        projected_mean,
        zone_mean,
        projected_jump,
    )
    initial_endpoint_states = np.array((left_state, right_state))
    endpoint_shift = physical.scaled_residual(
        endpoint_states - initial_endpoint_states,
        endpoint_states,
        initial_endpoint_states,
        np.array((state_jump, state_jump)),
    )
    initial_separation = float(np.linalg.norm(state_jump))
    projected_separation = float(np.linalg.norm(projected_jump))
    if (
        not solution.success
        or not physical.mhd.primitive_is_admissible(np.array((left, right)))
        or not 0.0 < fraction < 1.0
        or jump_residual > 3.0e-7
        or mixture_residual > 3.0e-7
        or endpoint_shift > 0.08
        or projected_separation < 0.25 * initial_separation
    ):
        return None
    return ProjectedWindow(
        start=fit.start,
        stop=fit.stop,
        left=left,
        right=right,
        speed=speed,
        left_fraction=fraction,
        jump_residual=jump_residual,
        mixture_residual=mixture_residual,
        endpoint_shift=endpoint_shift,
    )


def condense_projected_windows(
    primitive: Array,
    gamma: float,
    maximum_raw_score: float = 1.0e-2,
) -> tuple[Array, list[ProjectedWindow]]:
    original_state = physical.mhd.conserved_array(primitive, gamma)
    result_state = original_state.copy()
    candidates: list[ProjectedWindow] = []
    for fit in physical.physical_window_fits(primitive, gamma):
        raw_score = max(
            fit.jump_residual,
            fit.aggregate_residual,
            fit.profile_residual,
            fit.plateau_residual,
        )
        if raw_score > maximum_raw_score:
            continue
        projected = project_window(primitive, fit, gamma)
        if projected is not None:
            candidates.append(projected)
        if len(candidates) >= 64:
            break
    candidates.sort(
        key=lambda item: (
            max(
                item.jump_residual,
                item.mixture_residual,
                item.endpoint_shift,
            ),
            item.stop - item.start,
        )
    )

    accepted: list[ProjectedWindow] = []
    occupied: set[int] = set()
    for candidate in candidates:
        exclusion_zone = set(
            range(
                max(candidate.start - 1, 0),
                min(candidate.stop + 1, len(primitive)),
            )
        )
        if occupied.intersection(exclusion_zone):
            continue
        zone_cells = candidate.stop - candidate.start
        left_state, right_state = physical.mhd.conserved_array(
            np.array((candidate.left, candidate.right)),
            gamma,
        )
        left_equivalent = candidate.left_fraction * zone_cells
        full_left = min(int(math.floor(left_equivalent)), zone_cells)
        mixed_fraction = left_equivalent - full_left
        replacement = np.empty((zone_cells, 8))
        replacement[:] = right_state
        replacement[:full_left] = left_state
        mixed_index = min(full_left, zone_cells - 1)
        if mixed_fraction > 256.0 * np.finfo(float).eps:
            replacement[mixed_index] = (
                mixed_fraction * left_state
                + (1.0 - mixed_fraction) * right_state
            )
        zone_total = np.sum(
            original_state[candidate.start : candidate.stop],
            axis=0,
        )
        replacement[mixed_index] += zone_total - np.sum(replacement, axis=0)
        replacement_primitive = physical.mhd.primitive_from_conserved(
            replacement,
            gamma,
        )
        if not physical.mhd.primitive_is_admissible(replacement_primitive):
            continue
        result_state[candidate.start : candidate.stop] = replacement
        occupied.update(exclusion_zone)
        accepted.append(candidate)

    result = physical.mhd.primitive_from_conserved(result_state, gamma)
    if not physical.mhd.primitive_is_admissible(result):
        return primitive.copy(), []
    conservation_error = float(
        np.max(np.abs(np.sum(result_state - original_state, axis=0)))
    )
    conservation_scale = max(
        float(np.max(np.abs(np.sum(original_state, axis=0)))),
        1.0,
    )
    if conservation_error > 2.0e-12 * conservation_scale:
        raise RuntimeError("projected condensation violated conservation")
    return result, accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--reference-factor", type=int, default=4)
    args = parser.parse_args()

    comparisons: list[physical.mhd.Result] = []
    for case, initial, gamma, final_time in (
        ("brio_wu", physical.mhd.brio_wu_initial, 2.0, 0.1),
        (
            "dai_woodward",
            physical.subcell_mhd.dai_woodward_initial,
            5.0 / 3.0,
            0.08,
        ),
    ):
        reference, reference_completed = physical.mhd.evolve(
            initial(args.cells * args.reference_factor),
            final_time,
            args.cfl,
            gamma,
            physical.mhd.characteristic_mp5,
            physical.mhd_hlld_flux,
            periodic=False,
        )
        baseline, baseline_completed = physical.mhd.evolve(
            initial(args.cells),
            final_time,
            args.cfl,
            gamma,
            physical.mhd.characteristic_mp5,
            physical.mhd_hlld_flux,
            periodic=False,
        )
        if not reference_completed or not baseline_completed:
            raise RuntimeError(f"{case} evolution failed")
        exact = physical.mhd.coarsen_conserved(
            reference,
            args.cells,
            gamma,
        )
        projected, windows = condense_projected_windows(baseline, gamma)
        comparisons.extend(
            (
                physical.mhd.result_from_exact(
                    f"{case}_reference",
                    "mp5_hlld",
                    baseline,
                    exact,
                    True,
                    gamma,
                ),
                physical.mhd.result_from_exact(
                    f"{case}_reference",
                    f"projected_physical_windows_{len(windows)}",
                    projected,
                    exact,
                    physical.mhd.primitive_is_admissible(projected),
                    gamma,
                ),
            )
        )
        for window in windows:
            print(
                "projected_window"
                f"\tcase={case}"
                f"\tstart={window.start}"
                f"\tstop={window.stop}"
                f"\tspeed={window.speed:.12e}"
                f"\trh={window.jump_residual:.12e}"
                f"\tmixture={window.mixture_residual:.12e}"
                f"\tshift={window.endpoint_shift:.12e}"
            )
    physical.print_mhd_results(comparisons)


if __name__ == "__main__":
    main()
