#!/usr/bin/env python3
"""Prototype entropy/admissibility-constrained wave enrichment.

This is a research harness, not a production Riemann solver or validation
gate.  It tests one reusable part of a proposed solver:

    F(theta) = F_safe + theta * (F_resolved - F_safe),  0 <= theta <= 1.

``F_safe`` is a local Lax--Friedrichs flux whose dissipation is increased only
as much as needed to satisfy the two-point Tadmor entropy inequality.
``F_resolved`` is HLLC for ideal-gas Euler and HLLD for ideal MHD.  The
largest theta is selected subject to the entropy inequality and admissibility
of both one-sided, worst-local-CFL finite-volume updates.  The eventual method
will replace the single theta with independently constrained physical-wave
corrections; this scalar prototype exists to falsify the constraint mechanism
before that substantially larger implementation.

The MHD entropy variables and potential correspond to the conservative
one-dimensional ideal-MHD face problem with a common normal magnetic field.
They are not a claim that the multidimensional CT/source discretization is
entropy stable.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclasses.dataclass(frozen=True)
class ProjectionResult:
    flux: Array
    theta: float
    theta_entropy: float
    theta_admissible: float
    safe_speed: float
    safe_entropy_residual: float
    target_entropy_residual: float
    projected_entropy_residual: float
    required_entropy_residual: float
    safe_admissible: bool
    target_finite: bool


def hydro_conserved(primitive: Array, gamma: float) -> Array:
    rho, vx, vy, vz, pressure = primitive
    kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    energy = pressure / (gamma - 1.0) + kinetic
    return np.array([rho, rho * vx, rho * vy, rho * vz, energy])


def hydro_flux(primitive: Array, gamma: float) -> Array:
    state = hydro_conserved(primitive, gamma)
    _, vx, vy, vz, pressure = primitive
    return np.array(
        [
            state[0] * vx,
            state[1] * vx + pressure,
            state[2] * vx,
            state[3] * vx,
            (state[4] + pressure) * vx,
        ]
    )


def hydro_admissible(state: Array, gamma: float) -> bool:
    if not np.all(np.isfinite(state)) or state[0] <= 0.0:
        return False
    kinetic = 0.5 * np.dot(state[1:4], state[1:4]) / state[0]
    pressure = (gamma - 1.0) * (state[4] - kinetic)
    return bool(pressure > 0.0 and np.isfinite(pressure))


def hydro_entropy_pair(primitive: Array, gamma: float) -> tuple[Array, float]:
    rho, vx, vy, vz, pressure = primitive
    beta = rho / (2.0 * pressure)
    entropy = math.log(pressure) - gamma * math.log(rho)
    velocity2 = vx * vx + vy * vy + vz * vz
    variables = np.array(
        [
            (gamma - entropy) / (gamma - 1.0) - beta * velocity2,
            2.0 * beta * vx,
            2.0 * beta * vy,
            2.0 * beta * vz,
            -2.0 * beta,
        ]
    )
    return variables, rho * vx


def hydro_max_speed(primitive: Array, gamma: float) -> float:
    rho, vx, _, _, pressure = primitive
    sound = math.sqrt(gamma * pressure / rho)
    return abs(vx) + sound


def hydro_hllc_flux(left: Array, right: Array, gamma: float) -> Array:
    state_l = hydro_conserved(left, gamma)
    state_r = hydro_conserved(right, gamma)
    flux_l = hydro_flux(left, gamma)
    flux_r = hydro_flux(right, gamma)
    sound_l = math.sqrt(gamma * left[4] / left[0])
    sound_r = math.sqrt(gamma * right[4] / right[0])
    speed_l = min(left[1] - sound_l, right[1] - sound_r)
    speed_r = max(left[1] + sound_l, right[1] + sound_r)
    if speed_l >= 0.0:
        return flux_l
    if speed_r <= 0.0:
        return flux_r

    left_factor = left[0] * (speed_l - left[1])
    right_factor = right[0] * (speed_r - right[1])
    denominator = left_factor - right_factor
    if abs(denominator) <= 1.0e-14 * max(
        1.0, abs(left_factor), abs(right_factor)
    ):
        return np.full(5, np.nan)
    contact = (
        right[4]
        - left[4]
        + left_factor * left[1]
        - right_factor * right[1]
    ) / denominator

    def star(primitive: Array, state: Array, speed: float) -> Array:
        rho, vx, vy, vz, pressure = primitive
        star_density = rho * (speed - vx) / (speed - contact)
        star_pressure = pressure + rho * (speed - vx) * (contact - vx)
        star_energy = (
            (speed - vx) * state[4]
            - pressure * vx
            + star_pressure * contact
        ) / (speed - contact)
        return np.array(
            [
                star_density,
                star_density * contact,
                star_density * vy,
                star_density * vz,
                star_energy,
            ]
        )

    if contact >= 0.0:
        star_l = star(left, state_l, speed_l)
        return flux_l + speed_l * (star_l - state_l)
    star_r = star(right, state_r, speed_r)
    return flux_r + speed_r * (star_r - state_r)


def mhd_conserved(primitive: Array, gamma: float) -> Array:
    rho, vx, vy, vz, pressure, bx, by, bz = primitive
    kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    magnetic = 0.5 * (bx * bx + by * by + bz * bz)
    energy = pressure / (gamma - 1.0) + kinetic + magnetic
    return np.array(
        [rho, rho * vx, rho * vy, rho * vz, energy, bx, by, bz]
    )


def mhd_flux(primitive: Array, gamma: float) -> Array:
    state = mhd_conserved(primitive, gamma)
    rho, vx, vy, vz, pressure, bx, by, bz = primitive
    magnetic = 0.5 * (bx * bx + by * by + bz * bz)
    total_pressure = pressure + magnetic
    velocity_dot_b = vx * bx + vy * by + vz * bz
    return np.array(
        [
            rho * vx,
            rho * vx * vx + total_pressure - bx * bx,
            rho * vx * vy - bx * by,
            rho * vx * vz - bx * bz,
            (state[4] + total_pressure) * vx - bx * velocity_dot_b,
            0.0,
            by * vx - bx * vy,
            bz * vx - bx * vz,
        ]
    )


def mhd_admissible(state: Array, gamma: float) -> bool:
    if not np.all(np.isfinite(state)) or state[0] <= 0.0:
        return False
    kinetic = 0.5 * np.dot(state[1:4], state[1:4]) / state[0]
    magnetic = 0.5 * np.dot(state[5:8], state[5:8])
    pressure = (gamma - 1.0) * (state[4] - kinetic - magnetic)
    return bool(pressure > 0.0 and np.isfinite(pressure))


def mhd_entropy_pair(primitive: Array, gamma: float) -> tuple[Array, float]:
    rho, vx, vy, vz, pressure, bx, by, bz = primitive
    beta = rho / (2.0 * pressure)
    entropy = math.log(pressure) - gamma * math.log(rho)
    velocity2 = vx * vx + vy * vy + vz * vz
    magnetic2 = bx * bx + by * by + bz * bz
    velocity_dot_b = vx * bx + vy * by + vz * bz
    variables = np.array(
        [
            (gamma - entropy) / (gamma - 1.0) - beta * velocity2,
            2.0 * beta * vx,
            2.0 * beta * vy,
            2.0 * beta * vz,
            -2.0 * beta,
            2.0 * beta * bx,
            2.0 * beta * by,
            2.0 * beta * bz,
        ]
    )
    potential = (
        rho * vx
        + beta * vx * magnetic2
        - 2.0 * beta * bx * velocity_dot_b
    )
    return variables, potential


def mhd_fast_speed(primitive: Array, gamma: float) -> float:
    rho, _, _, _, pressure, bx, by, bz = primitive
    sound2 = gamma * pressure / rho
    magnetic2_over_rho = (bx * bx + by * by + bz * bz) / rho
    normal2_over_rho = bx * bx / rho
    discriminant = max(
        (sound2 + magnetic2_over_rho) ** 2
        - 4.0 * sound2 * normal2_over_rho,
        0.0,
    )
    return math.sqrt(
        0.5 * (sound2 + magnetic2_over_rho + math.sqrt(discriminant))
    )


def mhd_max_speed(primitive: Array, gamma: float) -> float:
    return abs(primitive[1]) + mhd_fast_speed(primitive, gamma)


def mhd_hlld_flux(left_in: Array, right_in: Array, gamma: float) -> Array:
    left = left_in.copy()
    right = right_in.copy()
    normal_b = 0.5 * (left[5] + right[5])
    left[5] = normal_b
    right[5] = normal_b
    state_l = mhd_conserved(left, gamma)
    state_r = mhd_conserved(right, gamma)
    flux_l = mhd_flux(left, gamma)
    flux_r = mhd_flux(right, gamma)
    speed_l = min(
        left[1] - mhd_fast_speed(left, gamma),
        right[1] - mhd_fast_speed(right, gamma),
    )
    speed_r = max(
        left[1] + mhd_fast_speed(left, gamma),
        right[1] + mhd_fast_speed(right, gamma),
    )
    if speed_l >= 0.0:
        return flux_l
    if speed_r <= 0.0:
        return flux_r

    def total_pressure(primitive: Array) -> float:
        return primitive[4] + 0.5 * np.dot(primitive[5:8], primitive[5:8])

    factor_l = left[0] * (speed_l - left[1])
    factor_r = right[0] * (speed_r - right[1])
    denominator = factor_r - factor_l
    if abs(denominator) <= 1.0e-14 * max(
        1.0, abs(factor_l), abs(factor_r)
    ):
        return np.full(8, np.nan)
    contact = (
        factor_r * right[1]
        - factor_l * left[1]
        + total_pressure(left)
        - total_pressure(right)
    ) / denominator
    pressure_star = total_pressure(left) + factor_l * (contact - left[1])

    def star(
        primitive: Array,
        state: Array,
        speed: float,
    ) -> tuple[Array, Array]:
        rho, vx, vy, vz, pressure, bx, by, bz = primitive
        density_star = rho * (speed - vx) / (speed - contact)
        denom = rho * (speed - vx) * (speed - contact) - bx * bx
        if abs(denom) <= 1.0e-14 * max(
            1.0, abs(rho * (speed - vx) * (speed - contact)), bx * bx
        ):
            velocity_y = vy
            velocity_z = vz
            magnetic_y = by
            magnetic_z = bz
        else:
            velocity_factor = bx * (contact - vx) / denom
            velocity_y = vy - by * velocity_factor
            velocity_z = vz - bz * velocity_factor
            magnetic_factor = (
                rho * (speed - vx) ** 2 - bx * bx
            ) / denom
            magnetic_y = by * magnetic_factor
            magnetic_z = bz * magnetic_factor
        velocity_dot_b = vx * bx + vy * by + vz * bz
        star_velocity_dot_b = (
            contact * bx
            + velocity_y * magnetic_y
            + velocity_z * magnetic_z
        )
        energy_star = (
            (speed - vx) * state[4]
            - total_pressure(primitive) * vx
            + pressure_star * contact
            + bx * (velocity_dot_b - star_velocity_dot_b)
        ) / (speed - contact)
        primitive_star = np.array(
            [
                density_star,
                contact,
                velocity_y,
                velocity_z,
                pressure,
                bx,
                magnetic_y,
                magnetic_z,
            ]
        )
        state_star = np.array(
            [
                density_star,
                density_star * contact,
                density_star * velocity_y,
                density_star * velocity_z,
                energy_star,
                bx,
                magnetic_y,
                magnetic_z,
            ]
        )
        return primitive_star, state_star

    primitive_ls, state_ls = star(left, state_l, speed_l)
    primitive_rs, state_rs = star(right, state_r, speed_r)
    flux_ls = flux_l + speed_l * (state_ls - state_l)
    flux_rs = flux_r + speed_r * (state_rs - state_r)
    abs_b = abs(normal_b)
    if abs_b <= 1.0e-14:
        return flux_ls if contact >= 0.0 else flux_rs
    if primitive_ls[0] <= 0.0 or primitive_rs[0] <= 0.0:
        return np.full(8, np.nan)

    sqrt_l = math.sqrt(primitive_ls[0])
    sqrt_r = math.sqrt(primitive_rs[0])
    inverse_sum = 1.0 / (sqrt_l + sqrt_r)
    sign_b = -1.0 if normal_b < 0.0 else 1.0
    speed_l_star = contact - abs_b / sqrt_l
    speed_r_star = contact + abs_b / sqrt_r
    velocity_y = (
        sqrt_l * primitive_ls[2]
        + sqrt_r * primitive_rs[2]
        + (primitive_rs[6] - primitive_ls[6]) * sign_b
    ) * inverse_sum
    velocity_z = (
        sqrt_l * primitive_ls[3]
        + sqrt_r * primitive_rs[3]
        + (primitive_rs[7] - primitive_ls[7]) * sign_b
    ) * inverse_sum
    magnetic_y = (
        sqrt_l * primitive_rs[6]
        + sqrt_r * primitive_ls[6]
        + sqrt_l
        * sqrt_r
        * (primitive_rs[2] - primitive_ls[2])
        * sign_b
    ) * inverse_sum
    magnetic_z = (
        sqrt_l * primitive_rs[7]
        + sqrt_r * primitive_ls[7]
        + sqrt_l
        * sqrt_r
        * (primitive_rs[3] - primitive_ls[3])
        * sign_b
    ) * inverse_sum
    dot_l = (
        contact * normal_b
        + primitive_ls[2] * primitive_ls[6]
        + primitive_ls[3] * primitive_ls[7]
    )
    dot_r = (
        contact * normal_b
        + primitive_rs[2] * primitive_rs[6]
        + primitive_rs[3] * primitive_rs[7]
    )
    dot_double = (
        contact * normal_b + velocity_y * magnetic_y + velocity_z * magnetic_z
    )
    energy_l = state_ls[4] - sqrt_l * (dot_l - dot_double) * sign_b
    energy_r = state_rs[4] + sqrt_r * (dot_r - dot_double) * sign_b
    state_ld = np.array(
        [
            primitive_ls[0],
            primitive_ls[0] * contact,
            primitive_ls[0] * velocity_y,
            primitive_ls[0] * velocity_z,
            energy_l,
            normal_b,
            magnetic_y,
            magnetic_z,
        ]
    )
    state_rd = np.array(
        [
            primitive_rs[0],
            primitive_rs[0] * contact,
            primitive_rs[0] * velocity_y,
            primitive_rs[0] * velocity_z,
            energy_r,
            normal_b,
            magnetic_y,
            magnetic_z,
        ]
    )
    flux_ld = flux_ls + speed_l_star * (state_ld - state_ls)
    flux_rd = flux_rs + speed_r_star * (state_rd - state_rs)
    if speed_l_star >= 0.0:
        return flux_ls
    if contact >= 0.0:
        return flux_ld
    if speed_r_star > 0.0:
        return flux_rd
    return flux_rs


def entropy_certified_llf(
    state_l: Array,
    state_r: Array,
    flux_l: Array,
    flux_r: Array,
    variables_l: Array,
    variables_r: Array,
    potential_l: float,
    potential_r: float,
    physical_speed: float,
) -> tuple[Array, float, float]:
    state_jump = state_r - state_l
    variable_jump = variables_r - variables_l
    average_flux = 0.5 * (flux_l + flux_r)
    convexity = float(np.dot(variable_jump, state_jump))
    central_residual = float(
        np.dot(variable_jump, average_flux) - (potential_r - potential_l)
    )
    entropy_speed = 0.0
    if convexity > 0.0:
        entropy_speed = max(0.0, 2.0 * central_residual / convexity)
    safe_speed = max(physical_speed, entropy_speed)
    safe_flux = average_flux - 0.5 * safe_speed * state_jump
    safe_residual = float(
        np.dot(variable_jump, safe_flux) - (potential_r - potential_l)
    )
    return safe_flux, safe_speed, safe_residual


def project_flux(
    state_l: Array,
    state_r: Array,
    flux_l: Array,
    flux_r: Array,
    safe_flux: Array,
    resolved_flux: Array,
    variables_l: Array,
    variables_r: Array,
    potential_l: float,
    potential_r: float,
    safe_speed: float,
    entropy_budget_fraction: float,
    admissible,
    gamma: float,
) -> ProjectionResult:
    variable_jump = variables_r - variables_l
    potential_jump = potential_r - potential_l

    def entropy_residual(flux: Array) -> float:
        return float(np.dot(variable_jump, flux) - potential_jump)

    safe_residual = entropy_residual(safe_flux)
    target_finite = bool(np.all(np.isfinite(resolved_flux)))
    if not target_finite:
        resolved_flux = safe_flux.copy()
    target_residual = entropy_residual(resolved_flux)
    theta_entropy = 1.0
    residual_delta = target_residual - safe_residual
    required_residual = entropy_budget_fraction * min(safe_residual, 0.0)
    if target_residual > required_residual:
        if safe_residual > required_residual or residual_delta <= 0.0:
            theta_entropy = 0.0
        else:
            theta_entropy = min(
                1.0,
                max(
                    0.0,
                    (required_residual - safe_residual) / residual_delta,
                ),
            )

    inverse_dx_speed = 0.45 / max(safe_speed, 1.0e-14)

    def updates(theta: float) -> tuple[Array, Array]:
        flux = safe_flux + theta * (resolved_flux - safe_flux)
        next_l = state_l - inverse_dx_speed * (flux - flux_l)
        next_r = state_r - inverse_dx_speed * (flux_r - flux)
        return next_l, next_r

    low_l, low_r = updates(0.0)
    safe_admissible = admissible(low_l, gamma) and admissible(low_r, gamma)
    theta_admissible = 0.0
    if safe_admissible:
        high_l, high_r = updates(theta_entropy)
        if admissible(high_l, gamma) and admissible(high_r, gamma):
            theta_admissible = theta_entropy
        else:
            lower = 0.0
            upper = theta_entropy
            for _ in range(52):
                middle = 0.5 * (lower + upper)
                middle_l, middle_r = updates(middle)
                if admissible(middle_l, gamma) and admissible(middle_r, gamma):
                    lower = middle
                else:
                    upper = middle
            theta_admissible = lower
    theta = min(theta_entropy, theta_admissible)
    flux = safe_flux + theta * (resolved_flux - safe_flux)
    projected_residual = safe_residual + theta * residual_delta
    return ProjectionResult(
        flux=flux,
        theta=theta,
        theta_entropy=theta_entropy,
        theta_admissible=theta_admissible,
        safe_speed=safe_speed,
        safe_entropy_residual=safe_residual,
        target_entropy_residual=target_residual,
        projected_entropy_residual=projected_residual,
        required_entropy_residual=required_residual,
        safe_admissible=safe_admissible,
        target_finite=target_finite,
    )


def constrained_hydro_flux(
    left: Array,
    right: Array,
    gamma: float,
    shock_budget_scale: float,
) -> ProjectionResult:
    state_l = hydro_conserved(left, gamma)
    state_r = hydro_conserved(right, gamma)
    flux_l = hydro_flux(left, gamma)
    flux_r = hydro_flux(right, gamma)
    variables_l, potential_l = hydro_entropy_pair(left, gamma)
    variables_r, potential_r = hydro_entropy_pair(right, gamma)
    physical_speed = max(
        hydro_max_speed(left, gamma), hydro_max_speed(right, gamma)
    )
    safe_flux, safe_speed, _ = entropy_certified_llf(
        state_l,
        state_r,
        flux_l,
        flux_r,
        variables_l,
        variables_r,
        potential_l,
        potential_r,
        physical_speed,
    )
    pressure_strength = abs(right[4] - left[4]) / max(
        right[4], left[4]
    )
    compression_strength = min(
        1.0,
        max(0.0, left[1] - right[1])
        / max(
            hydro_max_speed(left, gamma)
            + hydro_max_speed(right, gamma),
            1.0e-14,
        ),
    )
    entropy_budget_fraction = min(
        1.0,
        shock_budget_scale
        * math.sqrt(pressure_strength * compression_strength),
    )
    return project_flux(
        state_l,
        state_r,
        flux_l,
        flux_r,
        safe_flux,
        hydro_hllc_flux(left, right, gamma),
        variables_l,
        variables_r,
        potential_l,
        potential_r,
        safe_speed,
        entropy_budget_fraction,
        hydro_admissible,
        gamma,
    )


def constrained_mhd_flux(
    left_in: Array,
    right_in: Array,
    gamma: float,
    shock_budget_scale: float,
) -> ProjectionResult:
    left = left_in.copy()
    right = right_in.copy()
    normal_b = 0.5 * (left[5] + right[5])
    left[5] = normal_b
    right[5] = normal_b
    state_l = mhd_conserved(left, gamma)
    state_r = mhd_conserved(right, gamma)
    flux_l = mhd_flux(left, gamma)
    flux_r = mhd_flux(right, gamma)
    variables_l, potential_l = mhd_entropy_pair(left, gamma)
    variables_r, potential_r = mhd_entropy_pair(right, gamma)
    physical_speed = max(
        mhd_max_speed(left, gamma), mhd_max_speed(right, gamma)
    )
    safe_flux, safe_speed, _ = entropy_certified_llf(
        state_l,
        state_r,
        flux_l,
        flux_r,
        variables_l,
        variables_r,
        potential_l,
        potential_r,
        physical_speed,
    )
    total_pressure_l = left[4] + 0.5 * np.dot(left[5:8], left[5:8])
    total_pressure_r = right[4] + 0.5 * np.dot(right[5:8], right[5:8])
    pressure_strength = abs(total_pressure_r - total_pressure_l) / max(
        total_pressure_r, total_pressure_l
    )
    compression_strength = min(
        1.0,
        max(0.0, left[1] - right[1])
        / max(
            mhd_max_speed(left, gamma) + mhd_max_speed(right, gamma),
            1.0e-14,
        ),
    )
    entropy_budget_fraction = min(
        1.0,
        shock_budget_scale
        * math.sqrt(pressure_strength * compression_strength),
    )
    return project_flux(
        state_l,
        state_r,
        flux_l,
        flux_r,
        safe_flux,
        mhd_hlld_flux(left, right, gamma),
        variables_l,
        variables_r,
        potential_l,
        potential_r,
        safe_speed,
        entropy_budget_fraction,
        mhd_admissible,
        gamma,
    )


def random_hydro_primitive(rng: np.random.Generator) -> Array:
    rho = 10.0 ** rng.uniform(-4.0, 4.0)
    pressure = 10.0 ** rng.uniform(-5.0, 5.0)
    velocity = rng.normal(0.0, 3.0, size=3)
    return np.array([rho, *velocity, pressure])


def random_mhd_pair(
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    normal_b = math.copysign(
        10.0 ** rng.uniform(-4.0, 3.0),
        rng.uniform(-1.0, 1.0),
    )

    def primitive() -> Array:
        rho = 10.0 ** rng.uniform(-4.0, 4.0)
        pressure = 10.0 ** rng.uniform(-6.0, 5.0)
        velocity = rng.normal(0.0, 2.0, size=3)
        transverse_b = rng.normal(0.0, 1.0, size=2) * (
            10.0 ** rng.uniform(-4.0, 3.0)
        )
        return np.array(
            [
                rho,
                velocity[0],
                velocity[1],
                velocity[2],
                pressure,
                normal_b,
                transverse_b[0],
                transverse_b[1],
            ]
        )

    return primitive(), primitive()


def summarize(
    physics: str,
    results: list[ProjectionResult],
) -> dict[str, str | int | float]:
    theta = np.array([result.theta for result in results])
    return {
        "physics": physics,
        "samples": len(results),
        "safe_inadmissible": sum(not result.safe_admissible for result in results),
        "target_nonfinite": sum(not result.target_finite for result in results),
        "theta_mean": float(np.mean(theta)),
        "theta_min": float(np.min(theta)),
        "theta_full_fraction": float(np.mean(theta >= 1.0 - 1.0e-12)),
        "theta_zero_fraction": float(np.mean(theta <= 1.0e-12)),
        "entropy_limited_fraction": float(
            np.mean(
                [
                    result.theta_entropy < 1.0 - 1.0e-12
                    for result in results
                ]
            )
        ),
        "admissibility_limited_fraction": float(
            np.mean(
                [
                    result.theta_admissible
                    < result.theta_entropy - 1.0e-12
                    for result in results
                ]
            )
        ),
        "max_safe_entropy_residual": float(
            max(result.safe_entropy_residual for result in results)
        ),
        "max_normalized_entropy_violation": float(
            max(
                max(
                    0.0,
                    result.projected_entropy_residual
                    - result.required_entropy_residual,
                )
                / max(
                    1.0,
                    abs(result.projected_entropy_residual),
                    abs(result.required_entropy_residual),
                )
                for result in results
            )
        ),
    }


def contact_checks(
    gamma: float,
    shock_budget_scale: float,
) -> list[dict[str, str | float]]:
    checks: list[dict[str, str | float]] = []
    for velocity in (0.0, 0.7, -1.3):
        left = np.array([1.0, velocity, 0.2, -0.1, 1.0])
        right = np.array([7.0, velocity, 0.2, -0.1, 1.0])
        result = constrained_hydro_flux(
            left,
            right,
            gamma,
            shock_budget_scale,
        )
        target = hydro_hllc_flux(left, right, gamma)
        checks.append(
            {
                "physics": "hydro",
                "case": f"contact_v{velocity:g}",
                "theta": result.theta,
                "target_flux_error": float(np.max(np.abs(result.flux - target))),
            }
        )

    left_mhd = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, -0.2])
    right_mhd = np.array([4.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, -0.2])
    result_mhd = constrained_mhd_flux(
        left_mhd,
        right_mhd,
        gamma,
        shock_budget_scale,
    )
    target_mhd = mhd_hlld_flux(left_mhd, right_mhd, gamma)
    checks.append(
        {
            "physics": "mhd",
            "case": "stationary_contact",
            "theta": result_mhd.theta,
            "target_flux_error": float(
                np.max(np.abs(result_mhd.flux - target_mhd))
            ),
        }
    )
    return checks


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress a constrained HLLC/HLLD wave-enrichment prototype."
    )
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=872341)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument(
        "--shock-budget-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the compressive entropy budget; zero reproduces the "
            "entropy/admissibility-only falsification control"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/current/research/constrained_riemann_random_scan.tsv"
        ),
    )
    args = parser.parse_args()
    if args.shock_budget_scale < 0.0:
        parser.error("--shock-budget-scale must be nonnegative")

    rng = np.random.default_rng(args.seed)
    hydro_results: list[ProjectionResult] = []
    mhd_results: list[ProjectionResult] = []
    for _ in range(args.samples):
        hydro_results.append(
            constrained_hydro_flux(
                random_hydro_primitive(rng),
                random_hydro_primitive(rng),
                args.gamma,
                args.shock_budget_scale,
            )
        )
        left_mhd, right_mhd = random_mhd_pair(rng)
        mhd_results.append(
            constrained_mhd_flux(
                left_mhd,
                right_mhd,
                args.gamma,
                args.shock_budget_scale,
            )
        )

    summary_rows = [
        summarize("hydro", hydro_results),
        summarize("mhd", mhd_results),
    ]
    for row in summary_rows:
        row["shock_budget_scale"] = args.shock_budget_scale
    check_rows = contact_checks(args.gamma, args.shock_budget_scale)
    for row in check_rows:
        row["shock_budget_scale"] = args.shock_budget_scale
    write_rows(args.output, summary_rows)
    check_output = args.output.with_name(
        args.output.stem + "_contacts" + args.output.suffix
    )
    write_rows(check_output, check_rows)
    for row in summary_rows:
        print(
            f"{row['physics']}: samples={row['samples']} "
            f"theta_mean={row['theta_mean']:.6f} "
            f"full={row['theta_full_fraction']:.3f} "
            f"zero={row['theta_zero_fraction']:.3f} "
            f"safe_inadmissible={row['safe_inadmissible']} "
            f"target_nonfinite={row['target_nonfinite']}"
        )
    for row in check_rows:
        print(
            f"{row['physics']} {row['case']}: theta={row['theta']:.12g} "
            f"target_flux_error={row['target_flux_error']:.3e}"
        )
    print(f"wrote {args.output}")
    print(f"wrote {check_output}")

    if any(int(row["safe_inadmissible"]) != 0 for row in summary_rows):
        raise RuntimeError(
            "entropy-certified LLF baseline failed the local admissibility probe"
        )
    if any(
        float(row["max_normalized_entropy_violation"]) > 5.0e-12
        for row in summary_rows
    ):
        raise RuntimeError(f"projected flux violated entropy budget: {summary_rows}")
    for row in check_rows:
        if float(row["target_flux_error"]) > 1.0e-10:
            raise RuntimeError(f"contact recovery failed: {row}")


if __name__ == "__main__":
    main()
