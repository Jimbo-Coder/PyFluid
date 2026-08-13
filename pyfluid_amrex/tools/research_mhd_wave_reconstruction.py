#!/usr/bin/env python3
"""Test mode-selective MP5 reconstruction in ideal-MHD physical waves.

This standalone research harness carries the Euler hierarchy from
``research_wave_reconstruction.py`` into the generated seven-wave ideal-MHD
characteristic basis.  Entropy and Alfvén families retain their MP5 traces.
Fast and slow families use the same nested

    PWL-MC -> centered parabola -> MP5

feasible-polygon projection as the Euler prototype.  A face has one normal
magnetic field, so the reduced characteristic reconstruction cannot create
two incompatible normal-field traces.

The harness checks characteristic round trips, a smooth exact circular
Alfvén wave, and a Brio-Wu tube against a finer numerical reference.  It is a
falsification tool, not a production reconstruction method or an exact MHD
Riemann validation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyfluid.exact_solutions import circular_alfven_wave_exact  # noqa: E402
from research_constrained_riemann import (  # noqa: E402
    entropy_repaired_mhd_flux,
    hierarchical_mhd_flux,
    mhd_conserved,
    mhd_fast_speed,
    mhd_hlld_flux,
)
from research_modal_reconstruction import mp5_left  # noqa: E402
from research_wave_reconstruction import (  # noqa: E402
    hierarchical_scalar_pair,
    reconstruct_scalar_pair,
    stencil_value,
)


Array = np.ndarray
FaceReconstruction = Callable[[Array, int, float], tuple[Array, Array]]
RiemannFlux = Callable[
    [Array, Array, float],
    Array | tuple[Array, float],
]
REDUCED = (0, 1, 2, 3, 4, 6, 7)
NONLINEAR_WAVES = frozenset((0, 2, 4, 6))


@dataclasses.dataclass(frozen=True)
class MHDCharacteristicBasis:
    rho: float
    sqrt_rho: float
    sound: float
    sound2: float
    fast: float
    slow: float
    alpha_f: float
    alpha_s: float
    beta_t1: float
    beta_t2: float
    normal_sign: float
    velocity_denom: float


@dataclasses.dataclass(frozen=True)
class Result:
    case: str
    method: str
    cells: int
    completed: bool
    total_l1: float
    transverse_l1: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    mass_error: float
    momentum_error: float
    energy_error: float
    normal_b_spread: float
    minimum_rho: float
    minimum_pressure: float


def primitive_from_conserved(state: Array, gamma: float) -> Array:
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


def conserved_array(primitive: Array, gamma: float) -> Array:
    return np.stack(
        [mhd_conserved(row, gamma) for row in np.asarray(primitive)],
        axis=0,
    )


def primitive_is_admissible(primitive: Array) -> bool:
    return bool(
        np.all(np.isfinite(primitive))
        and np.all(primitive[..., 0] > 0.0)
        and np.all(primitive[..., 4] > 0.0)
    )


def characteristic_basis(
    left: Array,
    right: Array,
    gamma: float,
    normal_b: float,
) -> MHDCharacteristicBasis:
    rho = 0.5 * (left[0] + right[0])
    pressure = 0.5 * (left[4] + right[4])
    bt1 = 0.5 * (left[6] + right[6])
    bt2 = 0.5 * (left[7] + right[7])
    sound2 = gamma * pressure / rho
    sound = math.sqrt(sound2)
    magnetic2 = (normal_b * normal_b + bt1 * bt1 + bt2 * bt2) / rho
    normal2 = normal_b * normal_b / rho
    discriminant = max(
        (sound2 + magnetic2) ** 2 - 4.0 * sound2 * normal2,
        0.0,
    )
    root = math.sqrt(discriminant)
    fast2 = max(0.5 * (sound2 + magnetic2 + root), 0.0)
    slow2 = max(0.5 * (sound2 + magnetic2 - root), 0.0)
    fast = math.sqrt(fast2)
    slow = math.sqrt(slow2)

    transverse_b = math.hypot(bt1, bt2)
    beta_t1 = 1.0 / math.sqrt(2.0)
    beta_t2 = beta_t1
    if transverse_b > 1.0e-14:
        beta_t1 = bt1 / transverse_b
        beta_t2 = bt2 / transverse_b

    alpha_f = 1.0
    alpha_s = 0.0
    denominator = fast2 - slow2
    if denominator > 1.0e-14:
        alpha_f = math.sqrt(max((sound2 - slow2) / denominator, 0.0))
        alpha_s = math.sqrt(max((fast2 - sound2) / denominator, 0.0))
    velocity_denom = alpha_f * alpha_f * fast2
    velocity_denom += alpha_s * alpha_s * slow2
    if velocity_denom <= 1.0e-14:
        velocity_denom = sound2
    return MHDCharacteristicBasis(
        rho=rho,
        sqrt_rho=math.sqrt(rho),
        sound=sound,
        sound2=sound2,
        fast=fast,
        slow=slow,
        alpha_f=alpha_f,
        alpha_s=alpha_s,
        beta_t1=beta_t1,
        beta_t2=beta_t2,
        normal_sign=-1.0 if normal_b < 0.0 else 1.0,
        velocity_denom=velocity_denom,
    )


def to_characteristics(
    reduced: Array,
    basis: MHDCharacteristicBasis,
) -> Array:
    rho, vn, vt1, vt2, pressure, bt1, bt2 = np.moveaxis(
        reduced,
        -1,
        0,
    )
    compressive = pressure / (basis.sound2 * basis.rho)
    b_parallel = (
        basis.beta_t1 * bt1 + basis.beta_t2 * bt2
    ) / (basis.sound * basis.sqrt_rho)
    fast_sum = basis.alpha_f * compressive + basis.alpha_s * b_parallel
    slow_sum = basis.alpha_s * compressive - basis.alpha_f * b_parallel
    v_parallel = basis.normal_sign * (
        basis.beta_t1 * vt1 + basis.beta_t2 * vt2
    )
    fast_diff = (
        basis.alpha_f * basis.fast * vn
        - basis.alpha_s * basis.slow * v_parallel
    ) / basis.velocity_denom
    slow_diff = (
        basis.alpha_s * basis.slow * vn
        + basis.alpha_f * basis.fast * v_parallel
    ) / basis.velocity_denom
    alfven_sum = (
        basis.beta_t2 * bt1 - basis.beta_t1 * bt2
    ) / basis.sqrt_rho
    alfven_diff = basis.normal_sign * (
        basis.beta_t2 * vt1 - basis.beta_t1 * vt2
    )
    return np.stack(
        (
            0.5 * (fast_sum - fast_diff),
            0.5 * (alfven_sum + alfven_diff),
            0.5 * (slow_sum - slow_diff),
            rho - pressure / basis.sound2,
            0.5 * (slow_sum + slow_diff),
            0.5 * (alfven_sum - alfven_diff),
            0.5 * (fast_sum + fast_diff),
        ),
        axis=-1,
    )


def from_characteristics(
    waves: Array,
    basis: MHDCharacteristicBasis,
) -> Array:
    (
        fast_minus,
        alfven_minus,
        slow_minus,
        entropy,
        slow_plus,
        alfven_plus,
        fast_plus,
    ) = waves
    fast_sum = fast_minus + fast_plus
    fast_diff = fast_plus - fast_minus
    slow_sum = slow_minus + slow_plus
    slow_diff = slow_plus - slow_minus
    alfven_sum = alfven_minus + alfven_plus
    alfven_diff = alfven_minus - alfven_plus
    rho = (
        basis.alpha_f * basis.rho * fast_sum
        + basis.alpha_s * basis.rho * slow_sum
        + entropy
    )
    vn = (
        basis.alpha_f * basis.fast * fast_diff
        + basis.alpha_s * basis.slow * slow_diff
    )
    vt1 = basis.normal_sign * (
        -basis.alpha_s
        * basis.slow
        * basis.beta_t1
        * fast_diff
        + basis.alpha_f
        * basis.fast
        * basis.beta_t1
        * slow_diff
        + basis.beta_t2 * alfven_diff
    )
    vt2 = basis.normal_sign * (
        -basis.alpha_s
        * basis.slow
        * basis.beta_t2
        * fast_diff
        + basis.alpha_f
        * basis.fast
        * basis.beta_t2
        * slow_diff
        - basis.beta_t1 * alfven_diff
    )
    pressure = basis.rho * basis.sound2 * (
        basis.alpha_f * fast_sum + basis.alpha_s * slow_sum
    )
    bt1 = basis.sqrt_rho * (
        basis.alpha_s * basis.sound * basis.beta_t1 * fast_sum
        - basis.alpha_f * basis.sound * basis.beta_t1 * slow_sum
        + basis.beta_t2 * alfven_sum
    )
    bt2 = basis.sqrt_rho * (
        basis.alpha_s * basis.sound * basis.beta_t2 * fast_sum
        - basis.alpha_f * basis.sound * basis.beta_t2 * slow_sum
        - basis.beta_t1 * alfven_sum
    )
    return np.array((rho, vn, vt1, vt2, pressure, bt1, bt2))


def positive_trace_weight(trace: Array, cell: Array) -> float:
    theta = 1.0
    for component in (0, 4):
        floor = 1.0e-13 * max(1.0, cell[component])
        correction = trace[component] - cell[component]
        if trace[component] < floor and correction < 0.0:
            theta = min(
                theta,
                (cell[component] - floor) / (-correction),
            )
    return max(0.0, min(1.0, theta))


def positive_trace(trace: Array, cell: Array) -> Array:
    theta = positive_trace_weight(trace, cell)
    return cell + theta * (trace - cell)


def reconstruct_characteristic(
    primitive: Array,
    left_cell: int,
    gamma: float,
    hierarchy_objective: str | None,
    periodic: bool,
) -> tuple[Array, Array]:
    stencil = np.array(
        [
            stencil_value(primitive, left_cell + offset, periodic)
            for offset in range(-2, 4)
        ]
    )
    normal_b = 0.5 * (stencil[2, 5] + stencil[3, 5])
    basis = characteristic_basis(
        stencil[2],
        stencil[3],
        gamma,
        normal_b,
    )
    reduced = stencil[:, REDUCED]
    waves = to_characteristics(reduced, basis)
    left_waves = np.empty(7)
    right_waves = np.empty(7)
    for wave in range(7):
        values = waves[:, wave]
        if hierarchy_objective is not None and wave in NONLINEAR_WAVES:
            left_waves[wave], right_waves[wave], _, _ = (
                hierarchical_scalar_pair(values, hierarchy_objective)
            )
        else:
            left_waves[wave], right_waves[wave] = reconstruct_scalar_pair(
                values,
                mp5_left,
            )
    left_reduced = from_characteristics(left_waves, basis)
    right_reduced = from_characteristics(right_waves, basis)
    if hierarchy_objective is None:
        left_reduced = positive_trace(left_reduced, reduced[2])
        right_reduced = positive_trace(right_reduced, reduced[3])
    else:
        theta = min(
            positive_trace_weight(left_reduced, reduced[2]),
            positive_trace_weight(right_reduced, reduced[3]),
        )
        cell_jump = waves[3] - waves[2]

        def physical_pair(weight: float) -> tuple[Array, Array]:
            if weight <= 0.0:
                return reduced[2].copy(), reduced[3].copy()
            return (
                reduced[2] + weight * (left_reduced - reduced[2]),
                reduced[3] + weight * (right_reduced - reduced[3]),
            )

        def pair_feasible(weight: float) -> bool:
            left, right = physical_pair(weight)
            if (
                not np.all(np.isfinite(left))
                or not np.all(np.isfinite(right))
                or left[0] <= 0.0
                or left[4] <= 0.0
                or right[0] <= 0.0
                or right[4] <= 0.0
            ):
                return False
            trace_waves = to_characteristics(
                np.array((left, right)),
                basis,
            )
            trace_jump = trace_waves[1] - trace_waves[0]
            for wave in NONLINEAR_WAVES:
                lower = min(0.0, cell_jump[wave])
                upper = max(0.0, cell_jump[wave])
                tolerance = 128.0 * np.finfo(float).eps * max(
                    1.0,
                    abs(lower),
                    abs(upper),
                    abs(trace_jump[wave]),
                )
                if not (
                    lower - tolerance
                    <= trace_jump[wave]
                    <= upper + tolerance
                ):
                    return False
            return True

        if not pair_feasible(theta):
            lower = 0.0
            upper = theta
            for _ in range(52):
                middle = 0.5 * (lower + upper)
                if pair_feasible(middle):
                    lower = middle
                else:
                    upper = middle
            theta = lower
        left_reduced, right_reduced = physical_pair(theta)
    left = np.insert(left_reduced, 5, normal_b)
    right = np.insert(right_reduced, 5, normal_b)
    return left, right


def characteristic_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return reconstruct_characteristic(
        primitive,
        left_cell,
        gamma,
        hierarchy_objective=None,
        periodic=bool(characteristic_mp5.periodic),
    )


characteristic_mp5.periodic = False


def hierarchical_nonlinear_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return reconstruct_characteristic(
        primitive,
        left_cell,
        gamma,
        hierarchy_objective="coefficient",
        periodic=bool(hierarchical_nonlinear_mp5.periodic),
    )


hierarchical_nonlinear_mp5.periodic = False


def trace_metric_nonlinear_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return reconstruct_characteristic(
        primitive,
        left_cell,
        gamma,
        hierarchy_objective="trace",
        periodic=bool(trace_metric_nonlinear_mp5.periodic),
    )


trace_metric_nonlinear_mp5.periodic = False


def balanced_metric_nonlinear_mp5(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return reconstruct_characteristic(
        primitive,
        left_cell,
        gamma,
        hierarchy_objective="balanced",
        periodic=bool(balanced_metric_nonlinear_mp5.periodic),
    )


balanced_metric_nonlinear_mp5.periodic = False


def entropy_repair_flux(
    left: Array,
    right: Array,
    gamma: float,
) -> tuple[Array, float]:
    result = entropy_repaired_mhd_flux(left, right, gamma)
    return result.flux, result.safe_speed


def rhs(
    state: Array,
    dx: float,
    gamma: float,
    reconstruct: FaceReconstruction,
    riemann_flux: RiemannFlux,
    periodic: bool,
) -> tuple[Array, float]:
    primitive = primitive_from_conserved(state, gamma)
    if not primitive_is_admissible(primitive):
        return np.full_like(state, np.nan), math.inf
    reconstruct.periodic = periodic
    fluxes = np.empty((len(state) + 1, 8))
    required_speed = 0.0
    for face in range(len(state) + 1):
        left, right = reconstruct(primitive, face - 1, gamma)
        if not primitive_is_admissible(np.array((left, right))):
            return np.full_like(state, np.nan), math.inf
        endpoint_speeds = (
            abs(left[1]) + mhd_fast_speed(left, gamma),
            abs(right[1]) + mhd_fast_speed(right, gamma),
        )
        if any(
            not np.isfinite(speed) or speed < 0.0
            for speed in endpoint_speeds
        ):
            return np.full_like(state, np.nan), math.inf
        face_speed = max(endpoint_speeds)
        evaluation = riemann_flux(left, right, gamma)
        if isinstance(evaluation, tuple):
            face_flux, flux_speed = evaluation
            if not np.isfinite(flux_speed) or flux_speed < 0.0:
                return np.full_like(state, np.nan), math.inf
            face_speed = max(face_speed, float(flux_speed))
        else:
            face_flux = evaluation
        if not np.all(np.isfinite(face_flux)):
            return np.full_like(state, np.nan), math.inf
        fluxes[face] = face_flux
        required_speed = max(required_speed, face_speed)
    if periodic:
        fluxes[-1] = fluxes[0]
    return -(fluxes[1:] - fluxes[:-1]) / dx, required_speed


def evolve(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    reconstruct: FaceReconstruction,
    riemann_flux: RiemannFlux,
    periodic: bool,
) -> tuple[Array, bool]:
    state = conserved_array(primitive, gamma)
    dx = 1.0 / len(state)
    time = 0.0
    while time < final_time:
        current = primitive_from_conserved(state, gamma)
        if not primitive_is_admissible(current):
            return current, False
        speeds = np.array(
            [
                abs(row[1]) + mhd_fast_speed(row, gamma)
                for row in current
            ]
        )
        rhs0, required_speed = rhs(
            state,
            dx,
            gamma,
            reconstruct,
            riemann_flux,
            periodic,
        )
        if (
            not np.isfinite(required_speed)
            or not np.all(np.isfinite(rhs0))
        ):
            return current, False
        step_speed = max(float(np.max(speeds)), required_speed)
        dt = min(final_time - time, cfl * dx / step_speed)
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, stage1_speed = rhs(
                stage1,
                dx,
                gamma,
                reconstruct,
                riemann_flux,
                periodic,
            )
            if not np.isfinite(stage1_speed):
                dt *= 0.5
                continue
            reduced_dt = min(
                dt,
                final_time - time,
                cfl * dx / max(step_speed, stage1_speed),
            )
            if reduced_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * reduced_dt
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, stage2_speed = rhs(
                stage2,
                dx,
                gamma,
                reconstruct,
                riemann_flux,
                periodic,
            )
            if not np.isfinite(stage2_speed):
                dt *= 0.5
                continue
            reduced_dt = min(
                dt,
                final_time - time,
                cfl * dx / max(step_speed, stage1_speed, stage2_speed),
            )
            if reduced_dt < dt * (1.0 - 1.0e-13):
                dt = 0.99 * reduced_dt
                continue
            candidate = (1.0 / 3.0) * state + (2.0 / 3.0) * (
                stage2 + dt * rhs2
            )
            if not np.all(np.isfinite(candidate)):
                dt *= 0.5
                continue
            candidate_primitive = primitive_from_conserved(candidate, gamma)
            if not primitive_is_admissible(candidate_primitive):
                dt *= 0.5
                continue
            state = candidate
            break
        else:
            return primitive_from_conserved(state, gamma), False
        if not np.all(np.isfinite(state)):
            return primitive_from_conserved(state, gamma), False
        time += dt
    final = primitive_from_conserved(state, gamma)
    return final, primitive_is_admissible(final)


def total_variation(values: Array) -> float:
    return float(np.sum(np.abs(np.diff(values))))


def normalized_l1(values: Array, exact: Array) -> float:
    scale = max(float(np.max(exact) - np.min(exact)), 1.0e-14)
    return float(np.mean(np.abs(values - exact)) / scale)


def result_from_exact(
    case: str,
    method: str,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
) -> Result:
    errors = [
        normalized_l1(evolved[:, component], exact[:, component])
        for component in range(8)
        if np.ptp(exact[:, component]) > 1.0e-13
    ]
    transverse_errors = [
        normalized_l1(evolved[:, component], exact[:, component])
        for component in (2, 3, 6, 7)
        if np.ptp(exact[:, component]) > 1.0e-13
    ]
    overshoots = []
    tv_ratios = []
    for component in range(8):
        scale = float(np.ptp(exact[:, component]))
        if scale <= 1.0e-13:
            continue
        overshoots.append(
            max(
                0.0,
                float(np.max(evolved[:, component]) - np.max(exact[:, component])),
                float(np.min(exact[:, component]) - np.min(evolved[:, component])),
            )
            / scale
        )
        tv_ratios.append(
            total_variation(evolved[:, component])
            / max(total_variation(exact[:, component]), 1.0e-14)
        )
    exact_integrals = np.mean(conserved_array(exact, gamma), axis=0)
    final_integrals = np.mean(conserved_array(evolved, gamma), axis=0)
    return Result(
        case=case,
        method=method,
        cells=len(evolved),
        completed=completed,
        total_l1=float(np.mean(errors)),
        transverse_l1=(
            float(np.mean(transverse_errors)) if transverse_errors else 0.0
        ),
        maximum_overshoot=max(overshoots, default=0.0),
        maximum_tv_ratio=max(tv_ratios, default=1.0),
        mass_error=abs(final_integrals[0] - exact_integrals[0]),
        momentum_error=float(
            np.max(np.abs(final_integrals[1:4] - exact_integrals[1:4]))
        ),
        energy_error=abs(final_integrals[4] - exact_integrals[4]),
        normal_b_spread=float(np.ptp(evolved[:, 5])),
        minimum_rho=float(np.min(evolved[:, 0])),
        minimum_pressure=float(np.min(evolved[:, 4])),
    )


def circular_alfven_initial(cells: int) -> tuple[Array, Array]:
    x = (np.arange(cells) + 0.5) / cells
    values = circular_alfven_wave_exact(
        x,
        0.0,
        rho0=1.0,
        pressure0=1.0,
        bx0=1.0,
        amplitude=0.1,
        wave_number=1.0,
        direction=1.0,
    )
    rho, pressure, vx, vy, vz, bx, by, bz = values
    primitive = np.column_stack(
        (rho, vx, vy, vz, pressure, bx, by, bz)
    )
    return x, primitive


def circular_alfven_result(
    method: str,
    cells: int,
    cfl: float,
    reconstruct: FaceReconstruction,
    riemann_flux: RiemannFlux,
) -> Result:
    gamma = 5.0 / 3.0
    x, initial = circular_alfven_initial(cells)
    final_time = 1.0
    evolved, completed = evolve(
        initial,
        final_time,
        cfl,
        gamma,
        reconstruct,
        riemann_flux,
        periodic=True,
    )
    values = circular_alfven_wave_exact(
        x,
        final_time,
        rho0=1.0,
        pressure0=1.0,
        bx0=1.0,
        amplitude=0.1,
        wave_number=1.0,
        direction=1.0,
    )
    rho, pressure, vx, vy, vz, bx, by, bz = values
    exact = np.column_stack(
        (rho, vx, vy, vz, pressure, bx, by, bz)
    )
    return result_from_exact(
        "circular_alfven",
        method,
        evolved,
        exact,
        completed,
        gamma,
    )


def brio_wu_initial(cells: int) -> Array:
    x = (np.arange(cells) + 0.5) / cells
    primitive = np.empty((cells, 8))
    left = x < 0.5
    primitive[:, 0] = np.where(left, 1.0, 0.125)
    primitive[:, 1:4] = 0.0
    primitive[:, 4] = np.where(left, 1.0, 0.1)
    primitive[:, 5] = 0.75
    primitive[:, 6] = np.where(left, 1.0, -1.0)
    primitive[:, 7] = 0.0
    return primitive


def coarsen_conserved(
    primitive: Array,
    target_cells: int,
    gamma: float,
) -> Array:
    state = conserved_array(primitive, gamma)
    ratio = len(state) // target_cells
    if ratio * target_cells != len(state):
        raise ValueError("reference resolution must divide target resolution")
    coarse = state.reshape(target_cells, ratio, 8).mean(axis=1)
    return primitive_from_conserved(coarse, gamma)


def brio_wu_results(
    methods: dict[str, FaceReconstruction],
    riemann_methods: dict[str, RiemannFlux],
    cells: int,
    cfl: float,
    reference_factor: int,
) -> list[Result]:
    gamma = 2.0
    final_time = 0.1
    reference_initial = brio_wu_initial(cells * reference_factor)
    reference, reference_completed = evolve(
        reference_initial,
        final_time,
        cfl,
        gamma,
        characteristic_mp5,
        mhd_hlld_flux,
        periodic=False,
    )
    if not reference_completed:
        raise RuntimeError("fine Brio-Wu reference failed")
    exact = coarsen_conserved(reference, cells, gamma)
    results = []
    for method, reconstruct in methods.items():
        for riemann_name, riemann_flux in riemann_methods.items():
            result_name = (
                method
                if riemann_name == "hlld"
                else f"{method}+{riemann_name}"
            )
            initial = brio_wu_initial(cells)
            evolved, completed = evolve(
                initial,
                final_time,
                cfl,
                gamma,
                reconstruct,
                riemann_flux,
                periodic=False,
            )
            results.append(
                result_from_exact(
                    "brio_wu_reference",
                    result_name,
                    evolved,
                    exact,
                    completed,
                    gamma,
                )
            )
    return results


def characteristic_round_trip(samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    maximum_error = 0.0
    for sample in range(samples):
        if sample < 4:
            normal_b = (0.0, 1.0e-16, 1.0, -1.0)[sample]
            transverse = 0.0 if sample < 2 else 1.0e-16
        else:
            normal_b = rng.uniform(-2.0, 2.0)
            transverse = 10.0 ** rng.uniform(-12.0, 0.5)
        primitive = np.array(
            [
                10.0 ** rng.uniform(-2.0, 1.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                10.0 ** rng.uniform(-2.0, 1.0),
                normal_b,
                transverse,
                -0.37 * transverse,
            ]
        )
        basis = characteristic_basis(
            primitive,
            primitive,
            5.0 / 3.0,
            normal_b,
        )
        reduced = primitive[list(REDUCED)]
        recovered = from_characteristics(
            to_characteristics(reduced, basis),
            basis,
        )
        scale = np.maximum(1.0, np.abs(reduced))
        error = float(np.max(np.abs(recovered - reduced) / scale))
        maximum_error = max(maximum_error, error)
    return maximum_error


def print_results(results: list[Result]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(Result))
    print("\t".join(fields))
    for result in results:
        values = []
        for field in fields:
            value = getattr(result, field)
            values.append(f"{value:.12e}" if isinstance(value, float) else str(value))
        print("\t".join(values))


def write_results(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(field.name for field in dataclasses.fields(Result))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for result in results:
            writer.writerow(dataclasses.asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("circular_alfven", "brio_wu"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("mp5_characteristic", "hierarchical_nonlinear_mp5"),
    )
    parser.add_argument(
        "--riemann",
        nargs="+",
        choices=(
            "hlld",
            "hierarchical0",
            "hierarchical1",
            "entropy_repair",
        ),
        default=("hlld",),
    )
    parser.add_argument("--reference-factor", type=int, default=4)
    parser.add_argument("--round-trip-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cfl <= 0.0:
        parser.error("--cfl must be positive")
    if "entropy_repair" in args.riemann and args.cfl > 0.45:
        parser.error(
            "entropy_repair is locally audited only through CFL 0.45"
        )

    methods: dict[str, FaceReconstruction] = {
        "mp5_characteristic": characteristic_mp5,
        "hierarchical_nonlinear_mp5": hierarchical_nonlinear_mp5,
        "trace_metric_nonlinear_mp5": trace_metric_nonlinear_mp5,
        "balanced_metric_nonlinear_mp5": balanced_metric_nonlinear_mp5,
    }
    unknown_methods = sorted(set(args.methods) - set(methods))
    if unknown_methods:
        parser.error(f"unknown methods: {', '.join(unknown_methods)}")
    methods = {name: methods[name] for name in args.methods}
    riemann_methods: dict[str, RiemannFlux] = {
        "hlld": mhd_hlld_flux,
        "hierarchical0": (
            lambda left, right, gamma: hierarchical_mhd_flux(
                left,
                right,
                gamma,
                0.0,
            ).flux
        ),
        "hierarchical1": (
            lambda left, right, gamma: hierarchical_mhd_flux(
                left,
                right,
                gamma,
                1.0,
            ).flux
        ),
        "entropy_repair": entropy_repair_flux,
    }
    riemann_methods = {
        name: riemann_methods[name] for name in args.riemann
    }
    unknown_cases = sorted(
        set(args.cases) - {"circular_alfven", "brio_wu"}
    )
    if unknown_cases:
        parser.error(f"unknown cases: {', '.join(unknown_cases)}")

    round_trip_error = characteristic_round_trip(
        args.round_trip_samples,
        args.seed,
    )
    print(f"maximum_characteristic_round_trip_error={round_trip_error:.12e}")
    if round_trip_error > 5.0e-12:
        raise RuntimeError("MHD characteristic round trip exceeded tolerance")

    results: list[Result] = []
    for cells in args.cells:
        if "circular_alfven" in args.cases:
            for method, reconstruct in methods.items():
                for riemann_name, riemann_flux in riemann_methods.items():
                    result_name = (
                        method
                        if riemann_name == "hlld"
                        else f"{method}+{riemann_name}"
                    )
                    results.append(
                        circular_alfven_result(
                            result_name,
                            cells,
                            args.cfl,
                            reconstruct,
                            riemann_flux,
                        )
                    )
        if "brio_wu" in args.cases:
            results.extend(
                brio_wu_results(
                    methods,
                    riemann_methods,
                    cells,
                    args.cfl,
                    args.reference_factor,
                )
            )
    print_results(results)
    if args.output is not None:
        write_results(args.output, results)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
