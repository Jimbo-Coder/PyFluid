#!/usr/bin/env python3
"""Validate global matter C_m modes and equatorial-reflection reconstruction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

from validation_common import (
    ROOT,
    materialize_run_input,
    read_tsv,
    repo_relative,
    run_amrex,
    write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/matter_modes_uniform_3d.in"
RUN_ROOT = ROOT / "runs/current/matter_mode_validation"


def run_case(
    name: str,
    overrides: dict[str, str],
    ranks: int,
    timeout: int | None,
) -> tuple[Path, list[dict[str, str]], list[dict[str, str]]]:
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        BASE_INPUT,
        RUN_ROOT / f"{name}.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            **overrides,
        },
    )
    completed = run_amrex(
        dim=3,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"matter-mode run failed; see {run_dir / 'run.log'}")
    scalars = read_tsv(run_dir / "scalars.tsv", require_complete=True)
    modes = read_tsv(run_dir / "multipoles.tsv", require_complete=True)
    return run_dir, scalars, modes


def require_uniform_modes(
    scalars: list[dict[str, str]],
    modes: list[dict[str, str]],
    expected_barycenter: tuple[float, float, float],
) -> None:
    expected_modes = {0: 1.0 + 0.0j, 1: 0.0j, 2: 0.0j}
    steps = sorted({int(row["step"]) for row in modes})
    if steps != [0, 1]:
        raise RuntimeError(f"unexpected matter-mode steps: {steps}")
    for step in steps:
        rows = [row for row in modes if int(row["step"]) == step]
        if len(rows) != len(expected_modes):
            raise RuntimeError(f"step {step} has {len(rows)} mode rows")
        for row in rows:
            if row["field"] != "rest_mass":
                raise RuntimeError(f"unexpected mode field {row['field']}")
            if row["basis"] != "azimuthal_fourier_exp_plus_imphi":
                raise RuntimeError(f"unexpected mode basis {row['basis']}")
            if row["radius"] != "none" or row["ell"] != "none":
                raise RuntimeError("volume mode was mislabeled as a surface mode")
            mode = int(row["m"])
            value = complex(float(row["real"]), float(row["imag"]))
            if abs(value - expected_modes[mode]) > 2.0e-13:
                raise RuntimeError(
                    f"uniform C_{mode}={value}, expected {expected_modes[mode]}"
                )
    for row in scalars:
        values = {
            "matter_mode_mass": 1.0,
            "matter_barycenter1": expected_barycenter[0],
            "matter_barycenter2": expected_barycenter[1],
            "matter_barycenter3": expected_barycenter[2],
        }
        for field, expected in values.items():
            value = float(row[field])
            if not math.isfinite(value) or abs(value - expected) > 2.0e-13:
                raise RuntimeError(f"{field}={value}, expected {expected}")


def validate(args: argparse.Namespace) -> Path:
    if args.force and RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    full_dir, full_scalars, full_modes = run_case(
        "uniform_full",
        {},
        args.ranks,
        args.timeout,
    )
    require_uniform_modes(full_scalars, full_modes, (0.5, 0.5, 0.5))

    reduced_dir, reduced_scalars, reduced_modes = run_case(
        "uniform_equatorial_half",
        {
            "geometry.prob_lo": "0.0 0.0 0.0",
            "geometry.prob_hi": "1.0 1.0 0.5",
            "geometry.is_periodic": "1 1 0",
            "pyfluid.bc": "outflow",
            "pyfluid.bc_xlo": "periodic",
            "pyfluid.bc_xhi": "periodic",
            "pyfluid.bc_ylo": "periodic",
            "pyfluid.bc_yhi": "periodic",
            "pyfluid.bc_zlo": "symmetry",
            "diagnostics.matter_mode_origin3": "0.0",
        },
        args.ranks,
        args.timeout,
    )
    require_uniform_modes(reduced_scalars, reduced_modes, (0.5, 0.5, 0.0))

    full_values = [
        complex(float(row["real"]), float(row["imag"])) for row in full_modes
    ]
    reduced_values = [
        complex(float(row["real"]), float(row["imag"]))
        for row in reduced_modes
    ]
    equivalence_error = max(
        abs(left - right) for left, right in zip(full_values, reduced_values)
    )
    if equivalence_error > 3.0e-13:
        raise RuntimeError(
            "full and equatorial-half mode histories disagree: "
            f"{equivalence_error:.3e}"
        )

    result_path = RUN_ROOT / "matter_modes.tsv"
    write_tsv(
        result_path,
        [{
            "status": "PASS",
            "case": "uniform_global_matter_modes",
            "ranks": args.ranks,
            "samples": len(full_scalars),
            "maximum_reduced_equivalence_error": equivalence_error,
            "full_run_dir": repo_relative(full_dir),
            "reduced_run_dir": repo_relative(reduced_dir),
        }],
        [
            "status",
            "case",
            "ranks",
            "samples",
            "maximum_reduced_equivalence_error",
            "full_run_dir",
            "reduced_run_dir",
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
    print(f"PASS matter-mode validation: {repo_relative(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
