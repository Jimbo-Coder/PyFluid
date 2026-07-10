"""Linear HD/MHD and Jeans dispersion helpers for validation plots."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JeansMode:
    branch: str
    omega: float
    growth_rate: float
    omega2: float


def sound_speed_ideal_gas(gamma: float, pressure: float, rho: float) -> float:
    return math.sqrt(gamma * pressure / rho)


def hydro_acoustic_frequencies(k: np.ndarray, sound_speed: float, velocity: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return k * (velocity - sound_speed), k * (velocity + sound_speed)


def jeans_mode(k: float, sound_speed: float, rho0: float, gravity_constant: float) -> JeansMode:
    omega2 = sound_speed * sound_speed * k * k - 4.0 * math.pi * gravity_constant * rho0
    if omega2 >= 0.0:
        return JeansMode("stable", math.sqrt(omega2), 0.0, omega2)
    return JeansMode("unstable", 0.0, math.sqrt(-omega2), omega2)


def jeans_wave_exact(
    x: np.ndarray,
    time_value: float,
    gamma: float,
    rho0: float,
    pressure0: float,
    amplitude: float,
    wave_number: float,
    gravity_constant: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, JeansMode]:
    k = 2.0 * math.pi * wave_number
    cs = sound_speed_ideal_gas(gamma, pressure0, rho0)
    mode = jeans_mode(k, cs, rho0, gravity_constant)
    phase = k * x
    if mode.branch == "stable":
        theta = phase - mode.omega * time_value
        drho = amplitude * np.sin(theta)
        vx = (mode.omega * amplitude / (rho0 * k)) * np.sin(theta)
    else:
        growth = math.exp(mode.growth_rate * time_value)
        drho = amplitude * growth * np.sin(phase)
        vx = (mode.growth_rate * amplitude * growth / (rho0 * k)) * np.cos(phase)
    rho = rho0 + drho
    pressure = pressure0 + cs * cs * drho
    return rho, pressure, vx, mode


def ideal_mhd_wave_speeds(
    rho0: float,
    pressure0: float,
    gamma: float,
    magnetic_field: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> dict[str, float]:
    b = np.asarray(magnetic_field, dtype=float)
    n = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    n = n / norm
    cs2 = gamma * pressure0 / rho0
    va2 = float(np.dot(b, b)) / rho0
    can2 = float(np.dot(b, n) ** 2) / rho0
    disc = max((cs2 + va2) * (cs2 + va2) - 4.0 * cs2 * can2, 0.0)
    fast2 = 0.5 * (cs2 + va2 + math.sqrt(disc))
    slow2 = 0.5 * (cs2 + va2 - math.sqrt(disc))
    return {
        "slow": math.sqrt(max(slow2, 0.0)),
        "alfven": math.sqrt(max(can2, 0.0)),
        "fast": math.sqrt(max(fast2, 0.0)),
        "sound": math.sqrt(cs2),
    }


def ideal_mhd_frequencies(
    k: np.ndarray,
    rho0: float,
    pressure0: float,
    gamma: float,
    magnetic_field: tuple[float, float, float],
    direction: tuple[float, float, float],
    velocity_parallel: float = 0.0,
) -> dict[str, np.ndarray]:
    speeds = ideal_mhd_wave_speeds(rho0, pressure0, gamma, magnetic_field, direction)
    return {
        "fast-": k * (velocity_parallel - speeds["fast"]),
        "alfven-": k * (velocity_parallel - speeds["alfven"]),
        "slow-": k * (velocity_parallel - speeds["slow"]),
        "entropy/contact": k * velocity_parallel,
        "slow+": k * (velocity_parallel + speeds["slow"]),
        "alfven+": k * (velocity_parallel + speeds["alfven"]),
        "fast+": k * (velocity_parallel + speeds["fast"]),
    }


def _mhd_characteristic_speed(
    family: str,
    rho0: float,
    pressure0: float,
    gamma: float,
    magnetic_field: tuple[float, float, float],
) -> float:
    speeds = ideal_mhd_wave_speeds(rho0, pressure0, gamma, magnetic_field, (1.0, 0.0, 0.0))
    if family == "fast_plus":
        return speeds["fast"]
    if family == "fast_minus":
        return -speeds["fast"]
    if family == "alfven_plus":
        return speeds["alfven"]
    if family == "alfven_minus":
        return -speeds["alfven"]
    if family == "slow_plus":
        return speeds["slow"]
    if family == "slow_minus":
        return -speeds["slow"]
    if family == "entropy":
        return 0.0
    raise ValueError(f"unsupported MHD wave family {family!r}")


def linear_wave_x_exact(
    x: np.ndarray,
    time_value: float,
    *,
    physics: str,
    family: str,
    gamma: float,
    rho0: float,
    pressure0: float,
    amplitude: float,
    wave_number: float,
    v1_0: float = 0.0,
    v2_0: float = 0.0,
    v3_0: float = 0.0,
    b1_0: float = 0.0,
    b2_0: float = 0.0,
    b3_0: float = 0.0,
) -> tuple[dict[str, np.ndarray], float]:
    k = 2.0 * math.pi * wave_number
    cs = sound_speed_ideal_gas(gamma, pressure0, rho0)
    values = {
        "rho": np.full_like(x, rho0, dtype=float),
        "vx": np.full_like(x, v1_0, dtype=float),
        "vy": np.full_like(x, v2_0, dtype=float),
        "vz": np.full_like(x, v3_0, dtype=float),
        "pressure": np.full_like(x, pressure0, dtype=float),
        "b1": np.full_like(x, b1_0, dtype=float),
        "b2": np.full_like(x, b2_0, dtype=float),
        "b3": np.full_like(x, b3_0, dtype=float),
    }

    if physics == "hydro":
        if family == "acoustic_plus":
            c = cs
        elif family == "acoustic_minus":
            c = -cs
        elif family in ("entropy", "shear_y", "shear_z"):
            c = 0.0
        else:
            raise ValueError(f"unsupported hydro wave family {family!r}")
        profile = np.sin(k * (x - (v1_0 + c) * time_value))
        if family in ("acoustic_plus", "acoustic_minus"):
            values["rho"] += amplitude * profile
            values["vx"] += c * amplitude * profile / rho0
            values["pressure"] += cs * cs * amplitude * profile
        elif family == "entropy":
            values["rho"] += amplitude * profile
        elif family == "shear_y":
            values["vy"] += amplitude * profile
        elif family == "shear_z":
            values["vz"] += amplitude * profile
        return values, k * (v1_0 + c)

    if physics != "mhd":
        raise ValueError(f"unsupported physics {physics!r}")

    c = _mhd_characteristic_speed(family, rho0, pressure0, gamma, (b1_0, b2_0, b3_0))
    profile = np.sin(k * (x - (v1_0 + c) * time_value))
    if family in ("fast_plus", "fast_minus", "slow_plus", "slow_minus"):
        speeds = ideal_mhd_wave_speeds(rho0, pressure0, gamma, (b1_0, b2_0, b3_0), (1.0, 0.0, 0.0))
        cabs = speeds["fast"] if family.startswith("fast") else speeds["slow"]
        can2 = b1_0 * b1_0 / rho0
        denom = cabs * cabs - can2
        if abs(denom) < 1.0e-14:
            raise ValueError("degenerate magnetosonic eigenvector")
        values["vx"] += amplitude * profile
        values["rho"] += rho0 * amplitude * profile / c
        values["pressure"] += rho0 * cs * cs * amplitude * profile / c
        values["b2"] += b2_0 * c * amplitude * profile / denom
        values["b3"] += b3_0 * c * amplitude * profile / denom
        values["vy"] += -b1_0 * b2_0 * amplitude * profile / (rho0 * denom)
        values["vz"] += -b1_0 * b3_0 * amplitude * profile / (rho0 * denom)
    elif family in ("alfven_plus", "alfven_minus"):
        bt = math.sqrt(b2_0 * b2_0 + b3_0 * b3_0)
        if bt > 0.0:
            ey = -b3_0 / bt
            ez = b2_0 / bt
        else:
            ey = 1.0
            ez = 0.0
        values["b2"] += amplitude * ey * profile
        values["b3"] += amplitude * ez * profile
        values["vy"] += -b1_0 * amplitude * ey * profile / (rho0 * c)
        values["vz"] += -b1_0 * amplitude * ez * profile / (rho0 * c)
    elif family == "entropy":
        values["rho"] += amplitude * profile
    else:
        raise ValueError(f"unsupported MHD wave family {family!r}")
    return values, k * (v1_0 + c)
