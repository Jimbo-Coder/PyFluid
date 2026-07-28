#!/usr/bin/env python3
"""Explore a continuous hierarchical modal reconstruction outside production.

This is a research harness, not a validation gate.  It compares the scalar
part of existing production reconstructions with three rejected candidate
histories.  The sensor-modal branch writes the right interface value from cell
``i`` as

    q_face = q_i + delta_linear + theta_h * delta_high,

The constrained-face branch instead starts from PPM4 and admits the largest
MP5 modal correction that keeps both traces inside the PPM4/PWC envelope and
does not increase their boundary variation.  The sign-projected branch makes
the smallest joint correction to the MP5 traces that gives their jump the
same sign as the cell-average jump.  All three are retained here as
falsification controls; none is a production candidate.

The script intentionally tests only scalar linear advection and static stencil
properties.  Passing it is necessary but nowhere near sufficient for promotion
to the production hydro/MHD/relativistic reconstruction path.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Callable

import numpy as np


Array = np.ndarray
Reconstruction = Callable[
    [float, float, float, float, float, float],
    float,
]


def minmod(a: float, b: float) -> float:
    if a * b <= 0.0:
        return 0.0
    return min(a, b) if a > 0.0 else max(a, b)


def minmod4(a: float, b: float, c: float, d: float) -> float:
    if a > 0.0 and b > 0.0 and c > 0.0 and d > 0.0:
        return min(a, b, c, d)
    if a < 0.0 and b < 0.0 and c < 0.0 and d < 0.0:
        return max(a, b, c, d)
    return 0.0


def mc_slope(qm: float, q0: float, qp: float) -> float:
    centered = 0.5 * (qp - qm)
    limited = minmod(2.0 * (q0 - qm), 2.0 * (qp - q0))
    return minmod(centered, limited)


def pwl_left(
    _vmm: float,
    vm: float,
    v0: float,
    vp: float,
    _vpp: float,
    _vppp: float,
) -> float:
    return v0 + 0.5 * mc_slope(vm, v0, vp)


def unlimited_fifth_left(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
    vpp: float,
) -> float:
    return (2.0 * vmm - 13.0 * vm + 47.0 * v0 + 27.0 * vp - 3.0 * vpp) / 60.0


def mp5_left_with_bounds(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
    vpp: float,
    _vppp: float = 0.0,
) -> tuple[float, float, float]:
    alpha = 4.0
    epsilon = 1.0e-12
    unlimited = unlimited_fifth_left(vmm, vm, v0, vp, vpp)
    monotone = v0 + minmod(vp - v0, alpha * (v0 - vm))
    scale = max(abs(vmm), abs(vm), abs(v0), abs(vp), abs(vpp))
    accept_unlimited = (unlimited - v0) * (
        unlimited - monotone
    ) <= epsilon * max(1.0, scale * scale)

    dm = vmm - 2.0 * vm + v0
    d0 = vm - 2.0 * v0 + vp
    dp = v0 - 2.0 * vp + vpp
    d_m4_left = minmod4(4.0 * dm - d0, 4.0 * d0 - dm, dm, d0)
    d_m4_right = minmod4(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp)
    upper_limit = v0 + alpha * (v0 - vm)
    average = 0.5 * (v0 + vp)
    median = average - 0.5 * d_m4_right
    large_curvature = v0 + 0.5 * (v0 - vm) + (4.0 / 3.0) * d_m4_left
    lower_bound = max(
        min(v0, vp, median),
        min(v0, upper_limit, large_curvature),
    )
    upper_bound = min(
        max(v0, vp, median),
        max(v0, upper_limit, large_curvature),
    )
    limited = unlimited + minmod(
        lower_bound - unlimited,
        upper_bound - unlimited,
    )
    value = unlimited if accept_unlimited else limited
    return value, min(lower_bound, value), max(upper_bound, value)


def mp5_left(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
    vpp: float,
    _vppp: float = 0.0,
) -> float:
    value, _, _ = mp5_left_with_bounds(vmm, vm, v0, vp, vpp)
    return value


def wenoz_left(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
    vpp: float,
    _vppp: float,
) -> float:
    epsilon = 1.0e-40
    d0 = vmm - 2.0 * vm + v0
    e0 = vmm - 4.0 * vm + 3.0 * v0
    d1 = vm - 2.0 * v0 + vp
    e1 = vm - vp
    d2 = v0 - 2.0 * vp + vpp
    e2 = 3.0 * v0 - 4.0 * vp + vpp
    beta0 = (13.0 / 12.0) * d0 * d0 + 0.25 * e0 * e0
    beta1 = (13.0 / 12.0) * d1 * d1 + 0.25 * e1 * e1
    beta2 = (13.0 / 12.0) * d2 * d2 + 0.25 * e2 * e2
    tau5 = abs(beta0 - beta2)
    r0 = tau5 / (epsilon + beta0)
    r1 = tau5 / (epsilon + beta1)
    r2 = tau5 / (epsilon + beta2)
    a0 = 0.1 * (1.0 + r0 * r0)
    a1 = 0.6 * (1.0 + r1 * r1)
    a2 = 0.3 * (1.0 + r2 * r2)
    p0 = (2.0 * vmm - 7.0 * vm + 11.0 * v0) / 6.0
    p1 = (-vm + 5.0 * v0 + 2.0 * vp) / 6.0
    p2 = (2.0 * v0 + 5.0 * vp - vpp) / 6.0
    return (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2)


def normalized_fourth_difference(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
    vpp: float,
) -> float:
    second_m = vmm - 2.0 * vm + v0
    second_0 = vm - 2.0 * v0 + vp
    second_p = v0 - 2.0 * vp + vpp
    numerator = abs(second_m - 2.0 * second_0 + second_p)
    denominator = abs(second_m) + 2.0 * abs(second_0) + abs(second_p)
    scale = max(abs(vmm), abs(vm), abs(v0), abs(vp), abs(vpp), 1.0)
    return numerator / (denominator + 32.0 * np.finfo(float).eps * scale)


def modal_weight(sensor: float, threshold: float, exponent: int) -> float:
    ratio = sensor / threshold
    return 1.0 / (1.0 + ratio**exponent)


@dataclasses.dataclass(frozen=True)
class ModalParameters:
    high_threshold: float
    exponent: int = 4


def modal_left(parameters: ModalParameters) -> Reconstruction:
    def reconstruct(
        vmm: float,
        vm: float,
        v0: float,
        vp: float,
        vpp: float,
        _vppp: float,
    ) -> float:
        sensor = normalized_fourth_difference(vmm, vm, v0, vp, vpp)
        theta_high = modal_weight(
            sensor, parameters.high_threshold, parameters.exponent
        )
        linear = pwl_left(vmm, vm, v0, vp, vpp, 0.0)
        high = mp5_left(vmm, vm, v0, vp, vpp)
        return linear + theta_high * (high - linear)

    return reconstruct


def clamp_between(value: float, first: float, second: float) -> float:
    return max(min(first, second), min(max(first, second), value))


def parabolic_right_interface(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
) -> float:
    return (7.0 * (vm + v0) - (vmm + vp)) / 12.0


def ppm4_left(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
    vpp: float,
    _vppp: float = 0.0,
) -> float:
    left = clamp_between(
        parabolic_right_interface(vmm, vm, v0, vp), vm, v0
    )
    right = clamp_between(
        parabolic_right_interface(vm, v0, vp, vpp), v0, vp
    )
    right_delta = right - v0
    left_delta = left - v0
    if right_delta * left_delta >= 0.0:
        return v0
    if abs(right_delta) >= 2.0 * abs(left_delta):
        right = v0 - 2.0 * left_delta
    return right


def interval_weight(
    weight: float,
    base: float,
    correction: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    if correction > 0.0:
        weight = min(weight, (upper - base + tolerance) / correction)
    elif correction < 0.0:
        weight = min(weight, (lower - base - tolerance) / correction)
    return max(0.0, min(1.0, weight))


def constrained_face_left(
    vmm: float,
    vm: float,
    vl: float,
    vr: float,
    vp: float,
    vpp: float,
) -> float:
    robust_left = ppm4_left(vmm, vm, vl, vr, vp)
    robust_right = ppm4_left(vpp, vp, vr, vl, vm)
    high_left = mp5_left(vmm, vm, vl, vr, vp)
    high_right = mp5_left(vpp, vp, vr, vl, vm)
    lower = min(vl, vr, robust_left, robust_right)
    upper = max(vl, vr, robust_left, robust_right)
    scale = max(1.0, abs(vl), abs(vr), abs(robust_left), abs(robust_right))
    tolerance = 32.0 * np.finfo(float).eps * scale
    weight = 1.0
    weight = interval_weight(
        weight,
        robust_left,
        high_left - robust_left,
        lower,
        upper,
        tolerance,
    )
    weight = interval_weight(
        weight,
        robust_right,
        high_right - robust_right,
        lower,
        upper,
        tolerance,
    )
    robust_jump = robust_right - robust_left
    high_jump = high_right - high_left
    allowed_jump = abs(robust_jump) + tolerance
    weight = interval_weight(
        weight,
        robust_jump,
        high_jump - robust_jump,
        -allowed_jump,
        allowed_jump,
        tolerance,
    )
    return robust_left + weight * (high_left - robust_left)


def sign_projected_mp5_left(
    vmm: float,
    vm: float,
    vl: float,
    vr: float,
    vp: float,
    vpp: float,
) -> float:
    """Return the left trace after a minimal joint sign projection.

    MP5 supplies both traces.  Their mean is retained while their jump is
    projected onto the interval with the same sign as the cell-average jump
    and no greater magnitude.  The correction is fifth-order small wherever
    the two MP5 traces are already consistent.
    """
    left = mp5_left(vmm, vm, vl, vr, vp)
    right = mp5_left(vpp, vp, vr, vl, vm)
    cell_jump = vr - vl
    trace_jump = right - left
    lower = min(0.0, cell_jump)
    upper = max(0.0, cell_jump)
    projected_jump = min(upper, max(lower, trace_jump))
    return left - 0.5 * (projected_jump - trace_jump)


def reconstruct_faces(values: Array, reconstruct: Reconstruction) -> Array:
    cells = len(values)
    faces = np.empty(cells)
    for i in range(cells):
        faces[i] = reconstruct(
            values[(i - 2) % cells],
            values[(i - 1) % cells],
            values[i],
            values[(i + 1) % cells],
            values[(i + 2) % cells],
            values[(i + 3) % cells],
        )
    return faces


def rhs(values: Array, dx: float, reconstruct: Reconstruction) -> Array:
    flux = reconstruct_faces(values, reconstruct)
    return -(flux - np.roll(flux, 1)) / dx


def ssprk3_step(
    values: Array,
    dt: float,
    dx: float,
    reconstruct: Reconstruction,
) -> Array:
    stage1 = values + dt * rhs(values, dx, reconstruct)
    stage2 = 0.75 * values + 0.25 * (
        stage1 + dt * rhs(stage1, dx, reconstruct)
    )
    return (1.0 / 3.0) * values + (2.0 / 3.0) * (
        stage2 + dt * rhs(stage2, dx, reconstruct)
    )


def evolve_periodic(
    initial: Array,
    final_time: float,
    cfl: float,
    reconstruct: Reconstruction,
) -> Array:
    values = initial.copy()
    dx = 1.0 / len(values)
    steps = math.ceil(final_time / (cfl * dx))
    dt = final_time / steps
    for _ in range(steps):
        values = ssprk3_step(values, dt, dx, reconstruct)
    return values


def sine_cell_averages(cells: int) -> Array:
    dx = 1.0 / cells
    centers = (np.arange(cells) + 0.5) * dx
    factor = math.sin(math.pi * dx) / (math.pi * dx)
    return factor * np.sin(2.0 * math.pi * centers)


def square_cell_averages(cells: int) -> Array:
    dx = 1.0 / cells
    edges = np.arange(cells + 1) * dx
    left = np.maximum(edges[:-1], 0.2)
    right = np.minimum(edges[1:], 0.6)
    return np.maximum(0.0, right - left) / dx


def total_variation(values: Array) -> float:
    return float(np.sum(np.abs(values - np.roll(values, 1))))


@dataclasses.dataclass(frozen=True)
class AdvectionResult:
    method: str
    profile: str
    cells: int
    l1: float
    linf: float
    undershoot: float
    overshoot: float
    total_variation: float


def advection_result(
    method: str,
    profile: str,
    initial: Array,
    evolved: Array,
) -> AdvectionResult:
    error = np.abs(evolved - initial)
    lower = float(np.min(initial))
    upper = float(np.max(initial))
    return AdvectionResult(
        method=method,
        profile=profile,
        cells=len(initial),
        l1=float(np.mean(error)),
        linf=float(np.max(error)),
        undershoot=max(0.0, lower - float(np.min(evolved))),
        overshoot=max(0.0, float(np.max(evolved)) - upper),
        total_variation=total_variation(evolved),
    )


def run_advection_suite(
    methods: dict[str, Reconstruction],
    cells_values: tuple[int, ...],
    cfl: float,
) -> list[AdvectionResult]:
    results: list[AdvectionResult] = []
    for cells in cells_values:
        for profile, initial in (
            ("sine", sine_cell_averages(cells)),
            ("square", square_cell_averages(cells)),
        ):
            for name, reconstruct in methods.items():
                evolved = evolve_periodic(initial, 1.0, cfl, reconstruct)
                results.append(
                    advection_result(name, profile, initial, evolved)
                )
    return results


def print_results(results: list[AdvectionResult]) -> None:
    fields = tuple(field.name for field in dataclasses.fields(AdvectionResult))
    print("\t".join(fields))
    for result in results:
        print(
            "\t".join(
                str(getattr(result, field))
                if field in {"method", "profile", "cells"}
                else f"{getattr(result, field):.12e}"
                for field in fields
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--cells", type=int, nargs="+", default=(64, 128, 256))
    parser.add_argument("--high-threshold", type=float, default=0.1)
    parser.add_argument("--exponent", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = ModalParameters(
        high_threshold=args.high_threshold,
        exponent=args.exponent,
    )
    methods = {
        "pwl": pwl_left,
        "wenoz": wenoz_left,
        "mp5": mp5_left,
        "ppm4": ppm4_left,
        "sensor_modal_rejected": modal_left(parameters),
        "constrained_face": constrained_face_left,
        "sign_projected_mp5": sign_projected_mp5_left,
    }
    print_results(run_advection_suite(methods, tuple(args.cells), args.cfl))


if __name__ == "__main__":
    main()
