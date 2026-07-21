#!/usr/bin/env python3
"""Integrate runtime Psi4 multipoles and assemble a finite-radius energy ledger."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile

import numpy as np

from validation_common import read_tsv, write_tsv


def cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=values.dtype)
    if values.size > 1:
        result[1:] = np.cumsum(
            0.5 * (values[1:] + values[:-1]) * np.diff(times)
        )
    return result


def finite_float(row: dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid {field!r} value") from error
    if not math.isfinite(value):
        raise ValueError(f"{path}: nonfinite {field!r} value")
    return value


def read_mode_histories(path: Path) -> tuple[np.ndarray, float, dict[tuple[int, int], np.ndarray]]:
    all_rows = read_tsv(path, require_complete=True)
    rows = [
        row for row in all_rows
        if row.get("field") == "psi4"
        and row.get("basis") == "spin_weighted_spherical_harmonic"
    ]
    if not rows:
        raise ValueError(f"{path}: no Psi4 spherical-harmonic rows")
    radii = {finite_float(row, "radius", path) for row in rows}
    if len(radii) != 1:
        raise ValueError(f"{path}: expected one extraction radius, found {sorted(radii)}")
    radius = radii.pop()
    grouped: dict[tuple[int, int], dict[float, complex]] = {}
    all_times: set[float] = set()
    for row in rows:
        if int(row["spin_weight"]) != -2:
            raise ValueError(f"{path}: Psi4 modes must have spin weight -2")
        time = finite_float(row, "time", path)
        mode = (int(row["ell"]), int(row["m"]))
        value = complex(
            finite_float(row, "real", path), finite_float(row, "imag", path)
        )
        if time in grouped.setdefault(mode, {}):
            raise ValueError(f"{path}: duplicate mode {mode} at time {time}")
        grouped[mode][time] = value
        all_times.add(time)
    times = np.asarray(sorted(all_times), dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path}: times must be strictly increasing")
    histories: dict[tuple[int, int], np.ndarray] = {}
    expected_times = set(times.tolist())
    for mode, values in sorted(grouped.items()):
        if set(values) != expected_times:
            raise ValueError(f"{path}: incomplete history for mode {mode}")
        histories[mode] = np.asarray([values[time] for time in times], dtype=complex)
    return times, radius, histories


def read_radiation_history(path: Path, times: np.ndarray, radius: float) -> np.ndarray:
    rows = read_tsv(path, require_complete=True)
    by_time = {finite_float(row, "time", path): row for row in rows}
    if any(time not in by_time for time in times):
        raise ValueError(f"{path}: scalar history does not cover every Psi4 sample")
    selected = [by_time[time] for time in times]
    row_radii = np.asarray([
        finite_float(row, "extraction_radius", path) for row in selected
    ])
    if not np.allclose(row_radii, radius):
        raise ValueError(f"{path}: scalar and multipole extraction radii differ")
    return np.asarray([
        finite_float(row, "matter_energy_flux", path) for row in selected
    ])


def optional_adm_mass(run_dir: Path, times: np.ndarray) -> np.ndarray | None:
    path = run_dir / "scalars.tsv"
    rows = read_tsv(path, require_complete=True)
    if not rows or "adm_mass_surface" not in rows[0]:
        return None
    scalar_times = np.asarray([finite_float(row, "time", path) for row in rows])
    masses = np.asarray([finite_float(row, "adm_mass_surface", path) for row in rows])
    if times[0] < scalar_times[0] or times[-1] > scalar_times[-1]:
        raise ValueError(f"{path}: ADM surface history does not cover radiation times")
    return np.interp(times, scalar_times, masses)


def postprocess(run_dir: Path) -> tuple[Path, Path]:
    times, radius, psi4 = read_mode_histories(run_dir / "multipoles.tsv")
    matter_flux = read_radiation_history(run_dir / "scalars.tsv", times, radius)
    news = {mode: cumulative_trapezoid(values, times) for mode, values in psi4.items()}
    strain = {mode: cumulative_trapezoid(values, times) for mode, values in news.items()}
    luminosity = radius * radius * sum(
        np.abs(values) ** 2 for values in news.values()
    ) / (16.0 * math.pi)
    cumulative_matter = cumulative_trapezoid(matter_flux, times)
    cumulative_gw = cumulative_trapezoid(luminosity, times)
    adm_mass = optional_adm_mass(run_dir, times)

    mode_rows: list[dict[str, object]] = []
    for mode in sorted(psi4):
        ell, m = mode
        for index, time in enumerate(times):
            mode_rows.append({
                "time": time,
                "radius": radius,
                "ell": ell,
                "m": m,
                "psi4_real": psi4[mode][index].real,
                "psi4_imag": psi4[mode][index].imag,
                "news_real": news[mode][index].real,
                "news_imag": news[mode][index].imag,
                "strain_real": strain[mode][index].real,
                "strain_imag": strain[mode][index].imag,
            })
    mode_path = run_dir / "radiation_modes_postprocess.tsv"
    write_tsv(
        mode_path,
        mode_rows,
        [
            "time", "radius", "ell", "m", "psi4_real", "psi4_imag",
            "news_real", "news_imag", "strain_real", "strain_imag",
        ],
    )

    balance_rows: list[dict[str, object]] = []
    for index, time in enumerate(times):
        row: dict[str, object] = {
            "time": time,
            "radius": radius,
            "matter_energy_flux": matter_flux[index],
            "gw_luminosity": luminosity[index],
            "cumulative_matter_energy": cumulative_matter[index],
            "cumulative_gw_energy": cumulative_gw[index],
            "adm_mass_surface": "",
            "finite_radius_balance": "",
            "finite_radius_balance_change": "",
        }
        if adm_mass is not None:
            balance = adm_mass[index] + cumulative_matter[index] + cumulative_gw[index]
            initial_balance = adm_mass[0]
            row.update({
                "adm_mass_surface": adm_mass[index],
                "finite_radius_balance": balance,
                "finite_radius_balance_change": balance - initial_balance,
            })
        balance_rows.append(row)
    balance_path = run_dir / "radiation_balance.tsv"
    write_tsv(
        balance_path,
        balance_rows,
        [
            "time", "radius", "matter_energy_flux", "gw_luminosity",
            "cumulative_matter_energy", "cumulative_gw_energy",
            "adm_mass_surface", "finite_radius_balance",
            "finite_radius_balance_change",
        ],
    )
    return mode_path, balance_path


def self_test() -> None:
    times = np.asarray([0.0, 1.0, 2.0])
    radius = 3.0
    expected_news = times
    expected_strain = np.asarray([0.0, 0.5, 2.0])
    expected_luminosity = radius * radius * times * times / (16.0 * math.pi)
    expected_matter = 2.0 * times
    expected_gw = cumulative_trapezoid(expected_luminosity, times)
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        write_tsv(
            run_dir / "multipoles.tsv",
            [
                {
                    "step": index,
                    "time": time,
                    "radius": radius,
                    "field": "psi4",
                    "basis": "spin_weighted_spherical_harmonic",
                    "spin_weight": -2,
                    "ell": 2,
                    "m": 2,
                    "real": 1.0,
                    "imag": 0.0,
                }
                for index, time in enumerate(times)
            ],
            [
                "step", "time", "radius", "field", "basis", "spin_weight",
                "ell", "m", "real", "imag",
            ],
        )
        write_tsv(
            run_dir / "scalars.tsv",
            [
                {
                    "step": index,
                    "time": time,
                    "extraction_radius": radius,
                    "matter_energy_flux": 2.0,
                    "psi4_l2": 1.0,
                    "adm_mass_surface": 10.0 - expected_matter[index] - expected_gw[index],
                }
                for index, time in enumerate(times)
            ],
            [
                "step", "time", "extraction_radius", "matter_energy_flux",
                "psi4_l2", "adm_mass_surface",
            ],
        )
        mode_path, balance_path = postprocess(run_dir)
        mode_rows = read_tsv(mode_path, require_complete=True)
        balance_rows = read_tsv(balance_path, require_complete=True)
        np.testing.assert_allclose(
            [float(row["news_real"]) for row in mode_rows], expected_news
        )
        np.testing.assert_allclose(
            [float(row["strain_real"]) for row in mode_rows], expected_strain
        )
        np.testing.assert_allclose(
            [float(row["gw_luminosity"]) for row in balance_rows], expected_luminosity
        )
        np.testing.assert_allclose(
            [float(row["finite_radius_balance"]) for row in balance_rows], 10.0
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("PASS radiation postprocessing self-test")
        return 0
    if args.run_dir is None:
        parser.error("run_dir is required unless --self-test is used")
    mode_path, balance_path = postprocess(args.run_dir)
    print(f"wrote {mode_path}")
    print(f"wrote {balance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
