#!/usr/bin/env python3
"""Validate spherical interpolation, Psi4 modes, restart history, and balance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

from postprocess_radiation import postprocess, self_test
from validation_common import (
    ROOT,
    SMOKE_POLICY,
    build_cached_target,
    materialize_run_input,
    read_tsv,
    repo_relative,
    run_amrex,
    run_mpi,
    scalar_validation_row,
    write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/z4c_minkowski_radiation_3d.in"
RUN_ROOT = ROOT / "runs/current/radiation_validation"
RUN_DIR = RUN_ROOT / "minkowski_restart"
SURFACE_CHECK = ROOT / "build/pyfluid_amrex/pyfluid_amrex_surface_check"


def run_checked(
    input_path: Path,
    run_dir: Path,
    ranks: int,
    timeout: int | None,
    log_name: str,
) -> None:
    completed = run_amrex(
        dim=3, input_path=input_path, ranks=ranks, timeout=timeout
    )
    (run_dir / log_name).write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"radiation evolution failed; see {run_dir / log_name}")


def require_finite_small(rows: list[dict[str, str]], fields: tuple[str, ...], tolerance: float) -> None:
    for row in rows:
        for field in fields:
            value = float(row[field])
            if not math.isfinite(value) or abs(value) > tolerance:
                raise RuntimeError(f"{field}={value:.3e} exceeds {tolerance:.3e}")


def validate(args: argparse.Namespace) -> Path:
    if args.force and RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    build_cached_target(SURFACE_CHECK.name)
    surface = run_mpi(SURFACE_CHECK, ranks=args.ranks, timeout=args.timeout)
    (RUN_DIR / "surface_check.log").write_text(surface.stdout)
    if surface.returncode != 0:
        raise RuntimeError("surface/multipole check failed")
    self_test()

    base_input = materialize_run_input(
        BASE_INPUT,
        RUN_ROOT / "base.in",
        {
            "pyfluid.run_name": "minkowski_restart",
            "pyfluid.output_dir": repo_relative(RUN_DIR),
            "pyfluid.max_step": "1",
            "pyfluid.checkpoint_int": "1",
            "pyfluid.scalar_diagnostics": "standard,global_gr",
        },
    )
    run_checked(base_input, RUN_DIR, args.ranks, args.timeout, "base.log")
    checkpoint = RUN_DIR / "checkpoints/chk00001"
    if not checkpoint.exists():
        raise RuntimeError(f"missing restart checkpoint {checkpoint}")

    restart_input = materialize_run_input(
        base_input,
        RUN_ROOT / "restart.in",
        {
            "pyfluid.max_step": "2",
            "pyfluid.restart_file": repo_relative(checkpoint),
        },
    )
    run_checked(restart_input, RUN_DIR, args.ranks, args.timeout, "restart.log")

    scalar = scalar_validation_row(
        RUN_DIR,
        case="z4c_minkowski_radiation_restart",
        input_file=RUN_DIR / "run.in",
        checks=SMOKE_POLICY,
        mhd=False,
        dim=3,
        ranks=args.ranks,
        kind="feature",
    )
    if scalar["status"] != "PASS":
        raise RuntimeError(f"radiation scalar validation failed: {scalar['reason']}")

    radiation_rows = read_tsv(RUN_DIR / "scalars.tsv", require_complete=True)
    mode_rows = read_tsv(RUN_DIR / "multipoles.tsv", require_complete=True)
    if [int(row["step"]) for row in radiation_rows] != [0, 1, 2]:
        raise RuntimeError("restart radiation history is duplicated or incomplete")
    psi4_rows = [row for row in mode_rows if row["field"] == "psi4"]
    azimuthal_rows = [
        row
        for row in mode_rows
        if row["basis"] == "azimuthal_fourier_exp_minus_imphi"
    ]
    expected_psi4_modes = sum(2 * ell + 1 for ell in range(2, 5))
    expected_azimuthal_modes = 2 * 5
    expected_modes_per_step = expected_psi4_modes + expected_azimuthal_modes
    if len(mode_rows) != 3 * expected_modes_per_step:
        raise RuntimeError("restart multipole history is duplicated or incomplete")
    require_finite_small(
        radiation_rows, ("matter_energy_flux", "psi4_l2"), 1.0e-25
    )
    require_finite_small(psi4_rows, ("real", "imag"), 1.0e-25)
    surface_area = 4.0 * math.pi * 0.6**2
    for row in azimuthal_rows:
        expected = surface_area if int(row["m"]) == 0 else 0.0
        value = complex(float(row["real"]), float(row["imag"]))
        if abs(value - expected) > 2.0e-12 * surface_area:
            raise RuntimeError(
                f"uniform {row['field']} azimuthal mode m={row['m']} is {value}, "
                f"expected {expected}"
            )

    _mode_path, balance_path = postprocess(RUN_DIR)
    balance_rows = read_tsv(balance_path, require_complete=True)
    require_finite_small(
        balance_rows,
        (
            "matter_energy_flux",
            "gw_luminosity",
            "cumulative_matter_energy",
            "cumulative_gw_energy",
            "adm_mass_surface",
            "finite_radius_balance",
            "finite_radius_balance_change",
        ),
        1.0e-24,
    )

    wave_modes: list[dict[tuple[int, int], complex]] = []
    wave_l2: list[float] = []
    for polar_points in (10, 20, 40):
        wave_dir = RUN_ROOT / f"linear_wave_p{polar_points}"
        wave_dir.mkdir(parents=True, exist_ok=True)
        wave_input = materialize_run_input(
            BASE_INPUT,
            RUN_ROOT / f"linear_wave_p{polar_points}.in",
            {
                "pyfluid.run_name": f"linear_wave_p{polar_points}",
                "pyfluid.output_dir": repo_relative(wave_dir),
                "amr.n_cell": "24 24 24",
                "amr.max_grid_size": "12",
                "spacetime.initial_data": "linear_wave",
                "spacetime.amplitude": "1.0e-4",
                "spacetime.wave_number": "0.5",
                "spacetime.lapse_gauge": "frozen",
                "diagnostics.surface_polar_points": str(polar_points),
                "pyfluid.max_step": "1",
                "pyfluid.checkpoint_int": "-1",
            },
        )
        run_checked(wave_input, wave_dir, args.ranks, args.timeout, "run.log")
        wave_radiation = read_tsv(wave_dir / "scalars.tsv", require_complete=True)
        wave_rows = read_tsv(wave_dir / "multipoles.tsv", require_complete=True)
        wave_psi4_rows = [row for row in wave_rows if row["field"] == "psi4"]
        final_step = max(int(row["step"]) for row in wave_psi4_rows)
        modes = {
            (int(row["ell"]), int(row["m"])): complex(
                float(row["real"]), float(row["imag"])
            )
            for row in wave_psi4_rows
            if int(row["step"]) == final_step
        }
        wave_modes.append(modes)
        wave_l2.append(float(wave_radiation[-1]["psi4_l2"]))
    if min(wave_l2) <= 1.0e-8:
        raise RuntimeError("linear gravitational-wave extraction is unexpectedly zero")
    mode_keys = sorted(wave_modes[0])
    reference_scale = math.sqrt(sum(
        abs(wave_modes[-1][mode]) ** 2 for mode in mode_keys
    ))
    mode_resolution_differences = [
        math.sqrt(sum(
            abs(wave_modes[right][mode] - wave_modes[right - 1][mode]) ** 2
            for mode in mode_keys
        )) / max(reference_scale, 1.0e-30)
        for right in range(1, len(wave_modes))
    ]
    if mode_resolution_differences[1] >= mode_resolution_differences[0]:
        raise RuntimeError(
            "linear-wave multipoles do not converge with surface resolution: "
            f"successive relative differences={mode_resolution_differences}"
        )
    result_path = RUN_ROOT / "radiation_validation.tsv"
    write_tsv(
        result_path,
        [{
            "status": "PASS",
            "case": "z4c_minkowski_radiation_restart",
            "samples": len(radiation_rows),
            "modes_per_sample": expected_modes_per_step,
            "max_psi4_l2": max(float(row["psi4_l2"]) for row in radiation_rows),
            "linear_wave_psi4_l2": wave_l2[-1],
            "surface_resolution_relative_difference": mode_resolution_differences[-1],
            "run_dir": repo_relative(RUN_DIR),
        }],
        [
            "status", "case", "samples", "modes_per_sample", "max_psi4_l2",
            "linear_wave_psi4_l2", "surface_resolution_relative_difference",
            "run_dir",
        ],
    )
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args()
    result = validate(args)
    print(f"PASS radiation validation: {repo_relative(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
