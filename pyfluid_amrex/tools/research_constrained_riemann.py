#!/usr/bin/env python3
"""Prototype audited fan contraction and entropy/admissibility repair.

This is a research harness, not a production Riemann solver or validation
gate.  It retains the original scalar LLF-to-HLLC/HLLD projection and the
subsequently falsified independently weighted fan hierarchy as controls.  The
current candidate audits an HLLC/HLLD target, contracts every intermediate
state and flux toward one HLL state with a common admissible weight, applies a
state-jump Tadmor correction if needed, and finally bounds local one-sided
updates by an entropy-certified LLF flux.

``F_safe`` is a local Lax--Friedrichs flux whose dissipation is increased only
as much as needed to satisfy the two-point Tadmor entropy inequality.  The
common-fan and repair stages remain prototypes pending held-out evolution,
multidimensional CT, relativistic, and cost validation.

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


@dataclasses.dataclass(frozen=True)
class HierarchicalProjectionResult:
    flux: Array
    theta_outer: float
    theta_contact: float
    theta_alfven: float
    safe_speed: float
    safe_entropy_residual: float
    projected_entropy_residual: float
    required_entropy_residual: float
    safe_admissible: bool
    projected_admissible: bool
    targets_finite: bool
    target_fan_admissible: bool


@dataclasses.dataclass(frozen=True)
class EntropyRepairResult:
    flux: Array
    entropy_speed: float
    admissibility_weight: float
    contact_weight: float
    alfven_weight: float
    safe_speed: float
    safe_entropy_residual: float
    target_entropy_residual: float
    repaired_entropy_residual: float
    safe_admissible: bool
    repaired_admissible: bool
    target_finite: bool
    target_fan_admissible: bool
    target_fan_speed_ordered: bool
    target_fan_rh_residual: float
    accepted_fan_admissible: bool
    accepted_fan_speed_ordered: bool
    accepted_fan_rh_residual: float
    target_fan_depth: int


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


def hydro_hllc_flux(
    left: Array,
    right: Array,
    gamma: float,
    fan_state_sink: list[Array] | None = None,
    fan_speed_sink: list[float] | None = None,
) -> Array:
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

    star_l = star(left, state_l, speed_l)
    star_r = star(right, state_r, speed_r)
    if fan_state_sink is not None:
        fan_state_sink.extend((star_l, star_r))
    if fan_speed_sink is not None:
        fan_speed_sink.extend((speed_l, contact, speed_r))
    if contact >= 0.0:
        return flux_l + speed_l * (star_l - state_l)
    return flux_r + speed_r * (star_r - state_r)


def hydro_hll_flux(left: Array, right: Array, gamma: float) -> Array:
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
    return (
        speed_r * flux_l
        - speed_l * flux_r
        + speed_l * speed_r * (state_r - state_l)
    ) / (speed_r - speed_l)


def hydro_hll_state(
    left: Array,
    right: Array,
    gamma: float,
) -> Array | None:
    state_l = hydro_conserved(left, gamma)
    state_r = hydro_conserved(right, gamma)
    flux_l = hydro_flux(left, gamma)
    flux_r = hydro_flux(right, gamma)
    sound_l = math.sqrt(gamma * left[4] / left[0])
    sound_r = math.sqrt(gamma * right[4] / right[0])
    speed_l = min(left[1] - sound_l, right[1] - sound_r)
    speed_r = max(left[1] + sound_l, right[1] + sound_r)
    if speed_l >= 0.0 or speed_r <= 0.0:
        return None
    return (
        speed_r * state_r
        - speed_l * state_l
        - (flux_r - flux_l)
    ) / (speed_r - speed_l)


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


def mhd_hlld_flux(
    left_in: Array,
    right_in: Array,
    gamma: float,
    fan_level: str = "alfven",
    fan_state_sink: list[Array] | None = None,
    fan_speed_sink: list[float] | None = None,
) -> Array:
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
    if fan_state_sink is not None:
        fan_state_sink.extend((state_ls, state_rs))
    flux_ls = flux_l + speed_l * (state_ls - state_l)
    flux_rs = flux_r + speed_r * (state_rs - state_r)
    if fan_level == "contact":
        if fan_speed_sink is not None:
            fan_speed_sink.extend((speed_l, contact, speed_r))
        return flux_ls if contact >= 0.0 else flux_rs
    if fan_level != "alfven":
        raise ValueError(f"unknown HLLD fan level {fan_level!r}")
    abs_b = abs(normal_b)
    if abs_b <= 1.0e-14:
        if fan_speed_sink is not None:
            fan_speed_sink.extend((speed_l, contact, speed_r))
        return flux_ls if contact >= 0.0 else flux_rs
    if primitive_ls[0] <= 0.0 or primitive_rs[0] <= 0.0:
        if fan_speed_sink is not None:
            fan_speed_sink.extend((speed_l, contact, speed_r))
        return np.full(8, np.nan)

    sqrt_l = math.sqrt(primitive_ls[0])
    sqrt_r = math.sqrt(primitive_rs[0])
    inverse_sum = 1.0 / (sqrt_l + sqrt_r)
    sign_b = -1.0 if normal_b < 0.0 else 1.0
    speed_l_star = contact - abs_b / sqrt_l
    speed_r_star = contact + abs_b / sqrt_r
    if fan_speed_sink is not None:
        fan_speed_sink.extend(
            (speed_l, speed_l_star, contact, speed_r_star, speed_r)
        )
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
    if fan_state_sink is not None:
        fan_state_sink.extend((state_ld, state_rd))
    flux_ld = flux_ls + speed_l_star * (state_ld - state_ls)
    flux_rd = flux_rs + speed_r_star * (state_rd - state_rs)
    if speed_l_star >= 0.0:
        return flux_ls
    if contact >= 0.0:
        return flux_ld
    if speed_r_star > 0.0:
        return flux_rd
    return flux_rs


def mhd_hll_flux(left_in: Array, right_in: Array, gamma: float) -> Array:
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
    return (
        speed_r * flux_l
        - speed_l * flux_r
        + speed_l * speed_r * (state_r - state_l)
    ) / (speed_r - speed_l)


def mhd_hll_state(
    left_in: Array,
    right_in: Array,
    gamma: float,
) -> Array | None:
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
    if speed_l >= 0.0 or speed_r <= 0.0:
        return None
    return (
        speed_r * state_r
        - speed_l * state_l
        - (flux_r - flux_l)
    ) / (speed_r - speed_l)


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


def maximum_admissible_state_weight(
    base: Array,
    target: Array,
    admissible,
    gamma: float,
) -> float:
    if admissible(target, gamma):
        return 1.0
    if not admissible(base, gamma):
        return 0.0
    lower = 0.0
    upper = 1.0
    for _ in range(52):
        middle = 0.5 * (lower + upper)
        candidate = base + middle * (target - base)
        if admissible(candidate, gamma):
            lower = middle
        else:
            upper = middle
    return lower


def add_weighted_correction(
    base: Array,
    target: Array,
    reference: Array,
    weight: float,
) -> Array:
    """Add a correction without allowing an inactive NaN branch to leak."""
    if weight <= 0.0:
        return base.copy()
    return base + weight * (target - reference)


def fan_speeds_ordered(speeds: list[float]) -> bool:
    if not speeds:
        return True
    if not all(np.isfinite(speed) for speed in speeds):
        return False
    tolerance = 128.0 * np.finfo(float).eps * max(
        1.0,
        *(abs(speed) for speed in speeds),
    )
    return all(
        right >= left - tolerance
        for left, right in zip(speeds[:-1], speeds[1:], strict=True)
    )


def normalized_rh_residual(
    residual: Array,
    *scales: Array,
) -> float:
    scale = max(
        1.0,
        *(float(np.max(np.abs(value))) for value in scales),
    )
    return float(np.max(np.abs(residual))) / scale


def hydro_fan_rh_residual(
    left: Array,
    right: Array,
    gamma: float,
    states: list[Array],
    speeds: list[float],
) -> float:
    if not states and not speeds:
        return 0.0
    if len(states) != 2 or len(speeds) != 3:
        return math.inf
    state_l = hydro_conserved(left, gamma)
    state_r = hydro_conserved(right, gamma)
    flux_l = hydro_flux(left, gamma)
    flux_r = hydro_flux(right, gamma)
    speed_l, contact, speed_r = speeds
    state_ls, state_rs = states
    flux_ls = flux_l + speed_l * (state_ls - state_l)
    flux_rs = flux_r + speed_r * (state_rs - state_r)
    return normalized_rh_residual(
        flux_rs - flux_ls - contact * (state_rs - state_ls),
        flux_ls,
        flux_rs,
        contact * state_ls,
        contact * state_rs,
    )


def mhd_fan_rh_residual(
    left: Array,
    right: Array,
    gamma: float,
    states: list[Array],
    speeds: list[float],
) -> float:
    if not states and not speeds:
        return 0.0
    if len(states) not in (2, 4) or len(speeds) not in (3, 5):
        return math.inf
    state_l = mhd_conserved(left, gamma)
    state_r = mhd_conserved(right, gamma)
    flux_l = mhd_flux(left, gamma)
    flux_r = mhd_flux(right, gamma)
    speed_l = speeds[0]
    contact = speeds[len(speeds) // 2]
    speed_r = speeds[-1]
    state_ls, state_rs = states[:2]
    flux_ls = flux_l + speed_l * (state_ls - state_l)
    flux_rs = flux_r + speed_r * (state_rs - state_r)
    if len(states) == 2:
        state_left_contact = state_ls
        state_right_contact = state_rs
        flux_left_contact = flux_ls
        flux_right_contact = flux_rs
    else:
        speed_l_star, _, speed_r_star = speeds[1:4]
        state_left_contact, state_right_contact = states[2:4]
        flux_left_contact = flux_ls + speed_l_star * (
            state_left_contact - state_ls
        )
        flux_right_contact = flux_rs + speed_r_star * (
            state_right_contact - state_rs
        )
    return normalized_rh_residual(
        flux_right_contact
        - flux_left_contact
        - contact * (state_right_contact - state_left_contact),
        flux_left_contact,
        flux_right_contact,
        contact * state_left_contact,
        contact * state_right_contact,
    )


def hydro_weighted_fan_rh_residual(
    left: Array,
    right: Array,
    gamma: float,
    hll_state: Array | None,
    states: list[Array],
    speeds: list[float],
    contact_weight: float,
) -> float:
    if hll_state is None or not states or not speeds:
        return 0.0
    if len(states) != 2 or len(speeds) != 3:
        return math.inf
    state_l = hydro_conserved(left, gamma)
    state_r = hydro_conserved(right, gamma)
    flux_l = hydro_flux(left, gamma)
    flux_r = hydro_flux(right, gamma)
    speed_l, contact, speed_r = speeds
    state_ls, state_rs = states
    hll_flux = hydro_hll_flux(left, right, gamma)
    flux_ls = flux_l + speed_l * (state_ls - state_l)
    flux_rs = flux_r + speed_r * (state_rs - state_r)
    accepted_state_l = add_weighted_correction(
        hll_state,
        state_ls,
        hll_state,
        contact_weight,
    )
    accepted_state_r = add_weighted_correction(
        hll_state,
        state_rs,
        hll_state,
        contact_weight,
    )
    accepted_flux_l = add_weighted_correction(
        hll_flux,
        flux_ls,
        hll_flux,
        contact_weight,
    )
    accepted_flux_r = add_weighted_correction(
        hll_flux,
        flux_rs,
        hll_flux,
        contact_weight,
    )
    return normalized_rh_residual(
        accepted_flux_r
        - accepted_flux_l
        - contact * (accepted_state_r - accepted_state_l),
        accepted_flux_l,
        accepted_flux_r,
        contact * accepted_state_l,
        contact * accepted_state_r,
    )


def mhd_weighted_fan_rh_residual(
    left: Array,
    right: Array,
    gamma: float,
    hll_state: Array | None,
    states: list[Array],
    speeds: list[float],
    contact_weight: float,
    alfven_weight: float,
) -> float:
    if hll_state is None or not states or not speeds:
        return 0.0
    if len(states) not in (2, 4) or len(speeds) not in (3, 5):
        return math.inf
    state_l = mhd_conserved(left, gamma)
    state_r = mhd_conserved(right, gamma)
    flux_l = mhd_flux(left, gamma)
    flux_r = mhd_flux(right, gamma)
    speed_l = speeds[0]
    contact = speeds[len(speeds) // 2]
    speed_r = speeds[-1]
    state_ls, state_rs = states[:2]
    hll_flux = mhd_hll_flux(left, right, gamma)
    flux_ls = flux_l + speed_l * (state_ls - state_l)
    flux_rs = flux_r + speed_r * (state_rs - state_r)
    accepted_state_l = add_weighted_correction(
        hll_state,
        state_ls,
        hll_state,
        contact_weight,
    )
    accepted_state_r = add_weighted_correction(
        hll_state,
        state_rs,
        hll_state,
        contact_weight,
    )
    accepted_flux_l = add_weighted_correction(
        hll_flux,
        flux_ls,
        hll_flux,
        contact_weight,
    )
    accepted_flux_r = add_weighted_correction(
        hll_flux,
        flux_rs,
        hll_flux,
        contact_weight,
    )
    if len(states) == 4 and len(speeds) == 5 and alfven_weight > 0.0:
        speed_l_star = speeds[1]
        speed_r_star = speeds[3]
        state_ld, state_rd = states[2:4]
        flux_ld = flux_ls + speed_l_star * (state_ld - state_ls)
        flux_rd = flux_rs + speed_r_star * (state_rd - state_rs)
        accepted_state_l = add_weighted_correction(
            accepted_state_l,
            state_ld,
            state_ls,
            alfven_weight,
        )
        accepted_state_r = add_weighted_correction(
            accepted_state_r,
            state_rd,
            state_rs,
            alfven_weight,
        )
        accepted_flux_l = add_weighted_correction(
            accepted_flux_l,
            flux_ld,
            flux_ls,
            alfven_weight,
        )
        accepted_flux_r = add_weighted_correction(
            accepted_flux_r,
            flux_rd,
            flux_rs,
            alfven_weight,
        )
    return normalized_rh_residual(
        accepted_flux_r
        - accepted_flux_l
        - contact * (accepted_state_r - accepted_state_l),
        accepted_flux_l,
        accepted_flux_r,
        contact * accepted_state_l,
        contact * accepted_state_r,
    )


def project_hierarchical_flux(
    state_l: Array,
    state_r: Array,
    flux_l: Array,
    flux_r: Array,
    safe_flux: Array,
    fan_fluxes: tuple[Array, ...],
    variables_l: Array,
    variables_r: Array,
    potential_l: float,
    potential_r: float,
    safe_speed: float,
    entropy_budget_fraction: float,
    admissible,
    gamma: float,
    target_fan_admissible: bool,
    maximum_weights: tuple[float, ...] | None = None,
) -> HierarchicalProjectionResult:
    """Add outer, contact, and Alfvén fan corrections independently."""
    variable_jump = variables_r - variables_l
    potential_jump = potential_r - potential_l

    def entropy_residual(flux: Array) -> float:
        return float(np.dot(variable_jump, flux) - potential_jump)

    safe_residual = entropy_residual(safe_flux)
    required_residual = entropy_budget_fraction * min(safe_residual, 0.0)
    inverse_dx_speed = 0.45 / max(safe_speed, 1.0e-14)

    def feasible(flux: Array) -> bool:
        residual = entropy_residual(flux)
        tolerance = 128.0 * np.finfo(float).eps * max(
            1.0,
            abs(residual),
            abs(required_residual),
        )
        if residual > required_residual + tolerance:
            return False
        next_l = state_l - inverse_dx_speed * (flux - flux_l)
        next_r = state_r - inverse_dx_speed * (flux_r - flux)
        return admissible(next_l, gamma) and admissible(next_r, gamma)

    safe_admissible = feasible(safe_flux)
    current = safe_flux.copy()
    previous_endpoint = safe_flux
    weights: list[float] = []
    previous_weight = 1.0
    targets_finite = True
    for level, target in enumerate(fan_fluxes):
        if not np.all(np.isfinite(target)):
            targets_finite = False
            target = previous_endpoint
        correction = target - previous_endpoint
        upper = previous_weight
        if maximum_weights is not None:
            upper = min(upper, maximum_weights[level])
        if feasible(current + upper * correction):
            weight = upper
        else:
            lower = 0.0
            for _ in range(52):
                middle = 0.5 * (lower + upper)
                if feasible(current + middle * correction):
                    lower = middle
                else:
                    upper = middle
            weight = lower
        current += weight * correction
        weights.append(weight)
        previous_weight = weight
        previous_endpoint = target

    while len(weights) < 3:
        weights.append(math.nan)
    return HierarchicalProjectionResult(
        flux=current,
        theta_outer=weights[0],
        theta_contact=weights[1],
        theta_alfven=weights[2],
        safe_speed=safe_speed,
        safe_entropy_residual=safe_residual,
        projected_entropy_residual=entropy_residual(current),
        required_entropy_residual=required_residual,
        safe_admissible=safe_admissible,
        projected_admissible=feasible(current),
        targets_finite=targets_finite,
        target_fan_admissible=target_fan_admissible,
    )


def repair_entropy_residual(
    state_l: Array,
    state_r: Array,
    flux_l: Array,
    flux_r: Array,
    safe_flux: Array,
    target_flux: Array,
    variables_l: Array,
    variables_r: Array,
    potential_l: float,
    potential_r: float,
    safe_speed: float,
    admissible,
    gamma: float,
    target_fan_admissible: bool,
    target_fan_speed_ordered: bool,
    target_fan_rh_residual: float,
    accepted_fan_admissible: bool,
    accepted_fan_speed_ordered: bool,
    accepted_fan_rh_residual: float,
    target_fan_depth: int,
    contact_weight: float,
    alfven_weight: float,
) -> EntropyRepairResult:
    """Apply the smallest state-jump viscosity that closes the entropy defect."""
    variable_jump = variables_r - variables_l
    state_jump = state_r - state_l
    potential_jump = potential_r - potential_l

    def entropy_residual(flux: Array) -> float:
        return float(np.dot(variable_jump, flux) - potential_jump)

    inverse_dx_speed = 0.45 / max(safe_speed, 1.0e-14)

    def updates_admissible(flux: Array) -> bool:
        next_l = state_l - inverse_dx_speed * (flux - flux_l)
        next_r = state_r - inverse_dx_speed * (flux_r - flux)
        return admissible(next_l, gamma) and admissible(next_r, gamma)

    safe_residual = entropy_residual(safe_flux)
    safe_admissible = updates_admissible(safe_flux)
    target_finite = bool(np.all(np.isfinite(target_flux)))
    if not target_finite:
        target_flux = safe_flux.copy()
        target_fan_depth = 0
    target_residual = entropy_residual(target_flux)
    convexity = float(np.dot(variable_jump, state_jump))
    entropy_speed = 0.0
    repaired_flux = target_flux.copy()
    residual_tolerance = 128.0 * np.finfo(float).eps * max(
        1.0,
        abs(target_residual),
        abs(potential_jump),
    )
    if target_residual > residual_tolerance:
        if convexity > 0.0 and np.isfinite(convexity):
            entropy_speed = target_residual / convexity
            repaired_flux -= entropy_speed * state_jump
        else:
            repaired_flux = safe_flux.copy()
            target_fan_depth = 0

    admissibility_weight = 1.0
    if not updates_admissible(repaired_flux):
        lower = 0.0
        upper = 1.0
        for _ in range(52):
            middle = 0.5 * (lower + upper)
            candidate = safe_flux + middle * (repaired_flux - safe_flux)
            if updates_admissible(candidate):
                lower = middle
            else:
                upper = middle
        admissibility_weight = lower
        repaired_flux = safe_flux + lower * (repaired_flux - safe_flux)
    repaired_residual = entropy_residual(repaired_flux)
    repaired_admissible = updates_admissible(repaired_flux)
    return EntropyRepairResult(
        flux=repaired_flux,
        entropy_speed=entropy_speed,
        admissibility_weight=admissibility_weight,
        contact_weight=contact_weight,
        alfven_weight=alfven_weight,
        safe_speed=safe_speed,
        safe_entropy_residual=safe_residual,
        target_entropy_residual=target_residual,
        repaired_entropy_residual=repaired_residual,
        safe_admissible=safe_admissible,
        repaired_admissible=repaired_admissible,
        target_finite=target_finite,
        target_fan_admissible=target_fan_admissible,
        target_fan_speed_ordered=target_fan_speed_ordered,
        target_fan_rh_residual=target_fan_rh_residual,
        accepted_fan_admissible=accepted_fan_admissible,
        accepted_fan_speed_ordered=accepted_fan_speed_ordered,
        accepted_fan_rh_residual=accepted_fan_rh_residual,
        target_fan_depth=target_fan_depth,
    )


def entropy_repaired_hydro_flux(
    left: Array,
    right: Array,
    gamma: float,
) -> EntropyRepairResult:
    state_l = hydro_conserved(left, gamma)
    state_r = hydro_conserved(right, gamma)
    flux_l = hydro_flux(left, gamma)
    flux_r = hydro_flux(right, gamma)
    variables_l, potential_l = hydro_entropy_pair(left, gamma)
    variables_r, potential_r = hydro_entropy_pair(right, gamma)
    physical_speed = max(
        hydro_max_speed(left, gamma),
        hydro_max_speed(right, gamma),
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
    hll_flux = hydro_hll_flux(left, right, gamma)
    fan_states: list[Array] = []
    fan_speeds: list[float] = []
    hllc_flux = hydro_hllc_flux(
        left,
        right,
        gamma,
        fan_state_sink=fan_states,
        fan_speed_sink=fan_speeds,
    )
    fan_admissible = all(
        hydro_admissible(state, gamma) for state in fan_states
    )
    full_fan_speed_ordered = fan_speeds_ordered(fan_speeds)
    full_fan_rh_residual = hydro_fan_rh_residual(
        left,
        right,
        gamma,
        fan_states,
        fan_speeds,
    )
    hll_state = hydro_hll_state(left, right, gamma)
    hll_fan_admissible = (
        hll_state is None or hydro_admissible(hll_state, gamma)
    )
    fan_audit_passed = (
        hll_fan_admissible
        and full_fan_speed_ordered
        and full_fan_rh_residual <= 1.0e-10
        and np.all(np.isfinite(hllc_flux))
    )
    contact_weight = 1.0
    if not fan_audit_passed:
        contact_weight = 0.0
    elif hll_state is not None and fan_states:
        contact_weight = min(
            maximum_admissible_state_weight(
                hll_state,
                state,
                hydro_admissible,
                gamma,
            )
            for state in fan_states
        )
    accepted_fan_admissible = True
    if fan_audit_passed and hll_state is not None and fan_states:
        accepted_fan_admissible = all(
            hydro_admissible(
                add_weighted_correction(
                    hll_state,
                    state,
                    hll_state,
                    contact_weight,
                ),
                gamma,
            )
            for state in fan_states
        )
    if hll_fan_admissible:
        target_flux = add_weighted_correction(
            hll_flux,
            hllc_flux,
            hll_flux,
            contact_weight,
        )
    else:
        target_flux = safe_flux.copy()
    return repair_entropy_residual(
        state_l,
        state_r,
        flux_l,
        flux_r,
        safe_flux,
        target_flux,
        variables_l,
        variables_r,
        potential_l,
        potential_r,
        safe_speed,
        hydro_admissible,
        gamma,
        hll_fan_admissible and fan_admissible,
        full_fan_speed_ordered,
        full_fan_rh_residual,
        accepted_fan_admissible,
        contact_weight <= 0.0 or full_fan_speed_ordered,
        hydro_weighted_fan_rh_residual(
            left,
            right,
            gamma,
            hll_state,
            fan_states,
            fan_speeds,
            contact_weight,
        ),
        (
            0
            if not hll_fan_admissible or hll_state is None
            else (2 if contact_weight > 0.0 else 1)
        ),
        contact_weight,
        math.nan,
    )


def entropy_repaired_mhd_flux(
    left_in: Array,
    right_in: Array,
    gamma: float,
) -> EntropyRepairResult:
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
        mhd_max_speed(left, gamma),
        mhd_max_speed(right, gamma),
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
    hll_flux = mhd_hll_flux(left, right, gamma)
    fan_states: list[Array] = []
    fan_speeds: list[float] = []
    hlld_flux = mhd_hlld_flux(
        left,
        right,
        gamma,
        fan_state_sink=fan_states,
        fan_speed_sink=fan_speeds,
    )
    fan_admissible = all(
        mhd_admissible(state, gamma) for state in fan_states
    )
    full_fan_speed_ordered = fan_speeds_ordered(fan_speeds)
    full_fan_rh_residual = mhd_fan_rh_residual(
        left,
        right,
        gamma,
        fan_states,
        fan_speeds,
    )
    hll_state = mhd_hll_state(left, right, gamma)
    hll_fan_admissible = (
        hll_state is None or mhd_admissible(hll_state, gamma)
    )
    fan_weight = 1.0
    fan_audit_passed = (
        hll_fan_admissible
        and full_fan_speed_ordered
        and full_fan_rh_residual <= 1.0e-10
        and np.all(np.isfinite(hlld_flux))
    )
    if not fan_audit_passed:
        fan_weight = 0.0
    elif hll_state is not None and fan_states:
        fan_weight = min(
            maximum_admissible_state_weight(
                hll_state,
                state,
                mhd_admissible,
                gamma,
            )
            for state in fan_states
        )
    accepted_fan_admissible = True
    if fan_audit_passed and hll_state is not None and fan_states:
        accepted_fan_admissible = all(
            mhd_admissible(
                add_weighted_correction(
                    hll_state,
                    state,
                    hll_state,
                    fan_weight,
                ),
                gamma,
            )
            for state in fan_states
        )
    if hll_fan_admissible:
        target_flux = add_weighted_correction(
            hll_flux,
            hlld_flux,
            hll_flux,
            fan_weight,
        )
        fan_depth = 0 if hll_state is None else 1
        if hll_state is not None and fan_weight > 0.0:
            fan_depth = 3 if len(fan_states) >= 4 else 2
    else:
        target_flux = safe_flux.copy()
        fan_depth = 0
    return repair_entropy_residual(
        state_l,
        state_r,
        flux_l,
        flux_r,
        safe_flux,
        target_flux,
        variables_l,
        variables_r,
        potential_l,
        potential_r,
        safe_speed,
        mhd_admissible,
        gamma,
        hll_fan_admissible and fan_admissible,
        full_fan_speed_ordered,
        full_fan_rh_residual,
        accepted_fan_admissible,
        fan_weight <= 0.0 or full_fan_speed_ordered,
        mhd_weighted_fan_rh_residual(
            left,
            right,
            gamma,
            hll_state,
            fan_states,
            fan_speeds,
            fan_weight,
            fan_weight,
        ),
        fan_depth,
        fan_weight,
        fan_weight,
    )


def hierarchical_hydro_flux(
    left: Array,
    right: Array,
    gamma: float,
    shock_budget_scale: float,
) -> HierarchicalProjectionResult:
    state_l = hydro_conserved(left, gamma)
    state_r = hydro_conserved(right, gamma)
    flux_l = hydro_flux(left, gamma)
    flux_r = hydro_flux(right, gamma)
    variables_l, potential_l = hydro_entropy_pair(left, gamma)
    variables_r, potential_r = hydro_entropy_pair(right, gamma)
    physical_speed = max(
        hydro_max_speed(left, gamma),
        hydro_max_speed(right, gamma),
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
        right[4],
        left[4],
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
    hll_flux = hydro_hll_flux(left, right, gamma)
    hllc_states: list[Array] = []
    hllc_speeds: list[float] = []
    hllc_flux = hydro_hllc_flux(
        left,
        right,
        gamma,
        fan_state_sink=hllc_states,
        fan_speed_sink=hllc_speeds,
    )
    fan_admissible = all(
        hydro_admissible(state, gamma) for state in hllc_states
    )
    contact_weight = 1.0
    hll_state = hydro_hll_state(left, right, gamma)
    hll_outer_weight = 1.0
    if hll_state is not None and not hydro_admissible(hll_state, gamma):
        hll_outer_weight = 0.0
        contact_weight = 0.0
    elif not fan_speeds_ordered(hllc_speeds):
        contact_weight = 0.0
    elif hll_state is not None and hllc_states:
        contact_weight = min(
            maximum_admissible_state_weight(
                hll_state,
                state,
                hydro_admissible,
                gamma,
            )
            for state in hllc_states
        )
    return project_hierarchical_flux(
        state_l,
        state_r,
        flux_l,
        flux_r,
        safe_flux,
        (
            hll_flux,
            hllc_flux,
        ),
        variables_l,
        variables_r,
        potential_l,
        potential_r,
        safe_speed,
        entropy_budget_fraction,
        hydro_admissible,
        gamma,
        hll_outer_weight > 0.0 and fan_admissible,
        (hll_outer_weight, contact_weight),
    )


def hierarchical_mhd_flux(
    left_in: Array,
    right_in: Array,
    gamma: float,
    shock_budget_scale: float,
) -> HierarchicalProjectionResult:
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
        mhd_max_speed(left, gamma),
        mhd_max_speed(right, gamma),
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
        total_pressure_r,
        total_pressure_l,
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
    hll_flux = mhd_hll_flux(left, right, gamma)
    hlld_states: list[Array] = []
    hlld_speeds: list[float] = []
    hlld_flux = mhd_hlld_flux(
        left,
        right,
        gamma,
        fan_state_sink=hlld_states,
        fan_speed_sink=hlld_speeds,
    )
    fan_admissible = all(
        mhd_admissible(state, gamma) for state in hlld_states
    )
    hll_state = mhd_hll_state(left, right, gamma)
    hll_outer_weight = 1.0
    full_fan_speed_ordered = fan_speeds_ordered(hlld_speeds)
    full_fan_rh_residual = mhd_fan_rh_residual(
        left,
        right,
        gamma,
        hlld_states,
        hlld_speeds,
    )
    fan_weight = 1.0
    if hll_state is not None and not mhd_admissible(hll_state, gamma):
        hll_outer_weight = 0.0
        fan_weight = 0.0
    elif (
        not full_fan_speed_ordered
        or full_fan_rh_residual > 1.0e-10
        or not np.all(np.isfinite(hlld_flux))
    ):
        fan_weight = 0.0
    elif hll_state is not None and hlld_states:
        fan_weight = min(
            maximum_admissible_state_weight(
                hll_state,
                state,
                mhd_admissible,
                gamma,
            )
            for state in hlld_states
        )
    return project_hierarchical_flux(
        state_l,
        state_r,
        flux_l,
        flux_r,
        safe_flux,
        (
            hll_flux,
            hlld_flux,
        ),
        variables_l,
        variables_r,
        potential_l,
        potential_r,
        safe_speed,
        entropy_budget_fraction,
        mhd_admissible,
        gamma,
        (
            hll_outer_weight > 0.0
            and fan_admissible
            and full_fan_speed_ordered
            and full_fan_rh_residual <= 1.0e-10
        ),
        (hll_outer_weight, fan_weight),
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
        "projection": "scalar",
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


def summarize_hierarchical(
    physics: str,
    results: list[HierarchicalProjectionResult],
) -> dict[str, str | int | float]:
    outer = np.array([result.theta_outer for result in results])
    contact = np.array([result.theta_contact for result in results])
    alfven = np.array([result.theta_alfven for result in results])
    finite_alfven = alfven[np.isfinite(alfven)]
    return {
        "physics": physics,
        "projection": "hierarchical",
        "samples": len(results),
        "safe_inadmissible": sum(
            not result.safe_admissible for result in results
        ),
        "projected_inadmissible": sum(
            not result.projected_admissible for result in results
        ),
        "target_nonfinite": sum(
            not result.targets_finite for result in results
        ),
        "target_fan_inadmissible": sum(
            not result.target_fan_admissible for result in results
        ),
        "theta_outer_mean": float(np.mean(outer)),
        "theta_contact_mean": float(np.mean(contact)),
        "theta_alfven_mean": (
            float(np.mean(finite_alfven))
            if len(finite_alfven)
            else math.nan
        ),
        "theta_outer_full_fraction": float(
            np.mean(outer >= 1.0 - 1.0e-12)
        ),
        "theta_contact_full_fraction": float(
            np.mean(contact >= 1.0 - 1.0e-12)
        ),
        "theta_alfven_full_fraction": (
            float(np.mean(finite_alfven >= 1.0 - 1.0e-12))
            if len(finite_alfven)
            else math.nan
        ),
        "theta_outer_zero_fraction": float(
            np.mean(outer <= 1.0e-12)
        ),
        "max_nested_weight_violation": float(
            max(
                0.0,
                float(np.max(contact - outer)),
                (
                    float(np.max(finite_alfven - contact))
                    if len(finite_alfven)
                    else 0.0
                ),
                float(-np.min(outer)),
                float(np.max(outer) - 1.0),
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


def summarize_entropy_repair(
    physics: str,
    results: list[EntropyRepairResult],
) -> dict[str, str | int | float]:
    entropy_speed = np.array([result.entropy_speed for result in results])
    relative_speed = np.array(
        [
            result.entropy_speed / max(result.safe_speed, 1.0e-14)
            for result in results
        ]
    )
    admissibility_weight = np.array(
        [result.admissibility_weight for result in results]
    )
    effective_entropy_speed = entropy_speed * admissibility_weight
    contact_weight = np.array(
        [result.contact_weight for result in results]
    )
    alfven_weight = np.array(
        [result.alfven_weight for result in results]
    )
    target_fan_depth = np.array(
        [result.target_fan_depth for result in results]
    )
    return {
        "physics": physics,
        "projection": "entropy_repair",
        "samples": len(results),
        "safe_inadmissible": sum(
            not result.safe_admissible for result in results
        ),
        "projected_inadmissible": sum(
            not result.repaired_admissible for result in results
        ),
        "target_nonfinite": sum(
            not result.target_finite for result in results
        ),
        "target_fan_inadmissible": sum(
            not result.target_fan_admissible for result in results
        ),
        "target_fan_unordered": sum(
            not result.target_fan_speed_ordered for result in results
        ),
        "maximum_target_fan_rh_residual": float(
            max(result.target_fan_rh_residual for result in results)
        ),
        "accepted_fan_inadmissible": sum(
            not result.accepted_fan_admissible for result in results
        ),
        "accepted_fan_unordered": sum(
            not result.accepted_fan_speed_ordered for result in results
        ),
        "maximum_accepted_fan_rh_residual": float(
            max(result.accepted_fan_rh_residual for result in results)
        ),
        "entropy_repair_fraction": float(np.mean(entropy_speed > 0.0)),
        "mean_relative_entropy_speed": float(np.mean(relative_speed)),
        "max_relative_entropy_speed": float(np.max(relative_speed)),
        "mean_effective_entropy_speed": float(
            np.mean(effective_entropy_speed)
        ),
        "maximum_effective_entropy_speed": float(
            np.max(effective_entropy_speed)
        ),
        "admissibility_blend_fraction": float(
            np.mean(admissibility_weight < 1.0 - 1.0e-12)
        ),
        "mean_admissibility_weight": float(
            np.mean(admissibility_weight)
        ),
        "mean_contact_weight": float(np.mean(contact_weight)),
        "minimum_contact_weight": float(np.min(contact_weight)),
        "mean_alfven_weight": (
            float(np.nanmean(alfven_weight))
            if np.any(np.isfinite(alfven_weight))
            else math.nan
        ),
        "minimum_alfven_weight": (
            float(np.nanmin(alfven_weight))
            if np.any(np.isfinite(alfven_weight))
            else math.nan
        ),
        "mean_target_fan_depth": float(np.mean(target_fan_depth)),
        "max_normalized_entropy_violation": float(
            max(
                max(0.0, result.repaired_entropy_residual)
                / max(
                    1.0,
                    abs(result.repaired_entropy_residual),
                    abs(result.safe_entropy_residual),
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


def hierarchical_contact_checks(
    gamma: float,
    shock_budget_scale: float,
) -> list[dict[str, str | float]]:
    checks: list[dict[str, str | float]] = []
    for velocity in (0.0, 0.7, -1.3):
        left = np.array([1.0, velocity, 0.2, -0.1, 1.0])
        right = np.array([7.0, velocity, 0.2, -0.1, 1.0])
        result = hierarchical_hydro_flux(
            left,
            right,
            gamma,
            shock_budget_scale,
        )
        target = hydro_hllc_flux(left, right, gamma)
        checks.append(
            {
                "physics": "hydro",
                "projection": "hierarchical",
                "case": f"contact_v{velocity:g}",
                "theta_outer": result.theta_outer,
                "theta_contact": result.theta_contact,
                "theta_alfven": result.theta_alfven,
                "target_flux_error": float(
                    np.max(np.abs(result.flux - target))
                ),
            }
        )

    left_mhd = np.array(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, -0.2]
    )
    right_mhd = np.array(
        [4.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, -0.2]
    )
    result_mhd = hierarchical_mhd_flux(
        left_mhd,
        right_mhd,
        gamma,
        shock_budget_scale,
    )
    target_mhd = mhd_hlld_flux(left_mhd, right_mhd, gamma)
    checks.append(
        {
            "physics": "mhd",
            "projection": "hierarchical",
            "case": "stationary_contact",
            "theta_outer": result_mhd.theta_outer,
            "theta_contact": result_mhd.theta_contact,
            "theta_alfven": result_mhd.theta_alfven,
            "target_flux_error": float(
                np.max(np.abs(result_mhd.flux - target_mhd))
            ),
        }
    )

    left_alfven = np.array(
        [1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    )
    right_alfven = np.array(
        [1.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.5]
    )
    result_alfven = hierarchical_mhd_flux(
        left_alfven,
        right_alfven,
        gamma,
        shock_budget_scale,
    )
    physical_flux = mhd_flux(left_alfven, gamma)
    physical_flux_error = float(
        np.max(np.abs(mhd_flux(right_alfven, gamma) - physical_flux))
    )
    checks.append(
        {
            "physics": "mhd",
            "projection": "hierarchical",
            "case": "stationary_rotational_discontinuity",
            "theta_outer": result_alfven.theta_outer,
            "theta_contact": result_alfven.theta_contact,
            "theta_alfven": result_alfven.theta_alfven,
            "target_flux_error": float(
                np.max(np.abs(result_alfven.flux - physical_flux))
            ),
            "physical_flux_jump": physical_flux_error,
        }
    )
    return checks


def entropy_repair_checks(
    gamma: float,
) -> list[dict[str, str | float]]:
    checks: list[dict[str, str | float]] = []
    for velocity in (0.0, 0.7, -1.3):
        left = np.array([1.0, velocity, 0.2, -0.1, 1.0])
        right = np.array([7.0, velocity, 0.2, -0.1, 1.0])
        result = entropy_repaired_hydro_flux(left, right, gamma)
        target = hydro_hllc_flux(left, right, gamma)
        checks.append(
            {
                "physics": "hydro",
                "projection": "entropy_repair",
                "case": f"contact_v{velocity:g}",
                "entropy_speed": result.entropy_speed,
                "target_fan_depth": result.target_fan_depth,
                "target_flux_error": float(
                    np.max(np.abs(result.flux - target))
                ),
            }
        )

    left_contact = np.array(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, -0.2]
    )
    right_contact = np.array(
        [4.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, -0.2]
    )
    contact = entropy_repaired_mhd_flux(
        left_contact,
        right_contact,
        gamma,
    )
    checks.append(
        {
            "physics": "mhd",
            "projection": "entropy_repair",
            "case": "stationary_contact",
            "entropy_speed": contact.entropy_speed,
            "target_fan_depth": contact.target_fan_depth,
            "target_flux_error": float(
                np.max(
                    np.abs(
                        contact.flux
                        - mhd_hlld_flux(
                            left_contact,
                            right_contact,
                            gamma,
                        )
                    )
                )
            ),
        }
    )

    left_alfven = np.array(
        [1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    )
    right_alfven = np.array(
        [1.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.5]
    )
    alfven = entropy_repaired_mhd_flux(
        left_alfven,
        right_alfven,
        gamma,
    )
    checks.append(
        {
            "physics": "mhd",
            "projection": "entropy_repair",
            "case": "stationary_rotational_discontinuity",
            "entropy_speed": alfven.entropy_speed,
            "target_fan_depth": alfven.target_fan_depth,
            "target_flux_error": float(
                np.max(
                    np.abs(
                        alfven.flux - mhd_flux(left_alfven, gamma)
                    )
                )
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
        "--projection",
        choices=(
            "scalar",
            "hierarchical",
            "entropy_repair",
            "both",
            "all",
        ),
        default="all",
    )
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
    hierarchical_hydro_results: list[HierarchicalProjectionResult] = []
    hierarchical_mhd_results: list[HierarchicalProjectionResult] = []
    repaired_hydro_results: list[EntropyRepairResult] = []
    repaired_mhd_results: list[EntropyRepairResult] = []
    for _ in range(args.samples):
        left_hydro = random_hydro_primitive(rng)
        right_hydro = random_hydro_primitive(rng)
        left_mhd, right_mhd = random_mhd_pair(rng)
        if args.projection in ("scalar", "both", "all"):
            hydro_results.append(
                constrained_hydro_flux(
                    left_hydro,
                    right_hydro,
                    args.gamma,
                    args.shock_budget_scale,
                )
            )
            mhd_results.append(
                constrained_mhd_flux(
                    left_mhd,
                    right_mhd,
                    args.gamma,
                    args.shock_budget_scale,
                )
            )
        if args.projection in ("hierarchical", "both", "all"):
            hierarchical_hydro_results.append(
                hierarchical_hydro_flux(
                    left_hydro,
                    right_hydro,
                    args.gamma,
                    args.shock_budget_scale,
                )
            )
            hierarchical_mhd_results.append(
                hierarchical_mhd_flux(
                    left_mhd,
                    right_mhd,
                    args.gamma,
                    args.shock_budget_scale,
                )
            )
        if args.projection in ("entropy_repair", "all"):
            repaired_hydro_results.append(
                entropy_repaired_hydro_flux(
                    left_hydro,
                    right_hydro,
                    args.gamma,
                )
            )
            repaired_mhd_results.append(
                entropy_repaired_mhd_flux(
                    left_mhd,
                    right_mhd,
                    args.gamma,
                )
            )

    summary_rows: list[dict[str, str | int | float]] = []
    check_rows: list[dict[str, str | float]] = []
    if args.projection in ("scalar", "both", "all"):
        summary_rows.extend(
            (
                summarize("hydro", hydro_results),
                summarize("mhd", mhd_results),
            )
        )
        check_rows.extend(
            contact_checks(args.gamma, args.shock_budget_scale)
        )
    if args.projection in ("hierarchical", "both", "all"):
        summary_rows.extend(
            (
                summarize_hierarchical(
                    "hydro",
                    hierarchical_hydro_results,
                ),
                summarize_hierarchical(
                    "mhd",
                    hierarchical_mhd_results,
                ),
            )
        )
        check_rows.extend(
            hierarchical_contact_checks(
                args.gamma,
                args.shock_budget_scale,
            )
        )
    if args.projection in ("entropy_repair", "all"):
        summary_rows.extend(
            (
                summarize_entropy_repair(
                    "hydro",
                    repaired_hydro_results,
                ),
                summarize_entropy_repair(
                    "mhd",
                    repaired_mhd_results,
                ),
            )
        )
        check_rows.extend(entropy_repair_checks(args.gamma))
    for row in summary_rows:
        row["shock_budget_scale"] = args.shock_budget_scale
    for row in check_rows:
        row["shock_budget_scale"] = args.shock_budget_scale
    write_rows(args.output, summary_rows)
    check_output = args.output.with_name(
        args.output.stem + "_contacts" + args.output.suffix
    )
    write_rows(check_output, check_rows)
    for row in summary_rows:
        if row["projection"] == "scalar":
            print(
                f"{row['physics']} scalar: samples={row['samples']} "
                f"theta_mean={row['theta_mean']:.6f} "
                f"full={row['theta_full_fraction']:.3f} "
                f"zero={row['theta_zero_fraction']:.3f} "
                f"safe_inadmissible={row['safe_inadmissible']} "
                f"target_nonfinite={row['target_nonfinite']}"
            )
        elif row["projection"] == "hierarchical":
            print(
                f"{row['physics']} hierarchical: samples={row['samples']} "
                f"outer={row['theta_outer_mean']:.6f} "
                f"contact={row['theta_contact_mean']:.6f} "
                f"alfven={row['theta_alfven_mean']:.6f} "
                f"safe_inadmissible={row['safe_inadmissible']} "
                f"projected_inadmissible={row['projected_inadmissible']} "
                f"target_nonfinite={row['target_nonfinite']} "
                f"fan_inadmissible={row['target_fan_inadmissible']}"
            )
        else:
            print(
                f"{row['physics']} entropy_repair: "
                f"samples={row['samples']} "
                f"repair={row['entropy_repair_fraction']:.3f} "
                f"admissibility_blend="
                f"{row['admissibility_blend_fraction']:.3f} "
                f"contact={row['mean_contact_weight']:.6f} "
                f"alfven={row['mean_alfven_weight']:.6f} "
                f"mean_target_depth="
                f"{row['mean_target_fan_depth']:.3f} "
                f"projected_inadmissible={row['projected_inadmissible']} "
                f"fan_inadmissible={row['target_fan_inadmissible']} "
                f"fan_unordered={row['target_fan_unordered']} "
                f"fan_rh={row['maximum_target_fan_rh_residual']:.3e} "
                f"accepted_fan_inadmissible="
                f"{row['accepted_fan_inadmissible']} "
                f"accepted_fan_unordered="
                f"{row['accepted_fan_unordered']} "
                f"accepted_fan_rh="
                f"{row['maximum_accepted_fan_rh_residual']:.3e}"
            )
    for row in check_rows:
        if row.get("projection") == "hierarchical":
            print(
                f"{row['physics']} {row['case']}: "
                f"weights={row['theta_outer']:.12g}/"
                f"{row['theta_contact']:.12g}/"
                f"{row['theta_alfven']:.12g} "
                f"target_flux_error={row['target_flux_error']:.3e}"
            )
        elif row.get("projection") == "entropy_repair":
            print(
                f"{row['physics']} {row['case']}: "
                f"entropy_speed={row['entropy_speed']:.3e} "
                f"target_fan_depth={row['target_fan_depth']} "
                f"target_flux_error={row['target_flux_error']:.3e}"
            )
        else:
            print(
                f"{row['physics']} {row['case']}: "
                f"theta={row['theta']:.12g} "
                f"target_flux_error={row['target_flux_error']:.3e}"
            )
    print(f"wrote {args.output}")
    print(f"wrote {check_output}")

    if any(int(row["safe_inadmissible"]) != 0 for row in summary_rows):
        raise RuntimeError(
            "entropy-certified LLF baseline failed the local admissibility probe"
        )
    if any(
        int(row.get("projected_inadmissible", 0)) != 0
        for row in summary_rows
    ):
        raise RuntimeError(
            f"projected flux produced invalid updates: {summary_rows}"
        )
    if any(
        int(row.get("accepted_fan_inadmissible", 0)) != 0
        or int(row.get("accepted_fan_unordered", 0)) != 0
        or float(row.get("maximum_accepted_fan_rh_residual", 0.0))
        > 1.0e-9
        for row in summary_rows
    ):
        raise RuntimeError(f"accepted fan audit failed: {summary_rows}")
    if any(
        float(row.get("max_nested_weight_violation", 0.0)) > 5.0e-12
        for row in summary_rows
    ):
        raise RuntimeError(
            f"hierarchical wave weights lost nesting: {summary_rows}"
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
