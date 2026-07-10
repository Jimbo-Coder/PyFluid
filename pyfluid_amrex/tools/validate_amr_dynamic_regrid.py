#!/usr/bin/env python3
"""Validate that adaptive AMR boxes move with a moving state feature."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation_common import (
    ROOT,
    VALIDATION_HISTORY_FIELDS,
    append_validation_history,
    repo_relative,
    run_amrex,
    scalar_validation_row,
    write_tsv,
)

from plot_amr_hierarchy import plotfile_level_boxes
from plotfile_reader import sorted_plotfiles


RUN_ROOT = ROOT / "runs/current/amr_dynamic_regrid"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
BASE_INPUT = ROOT / "pyfluid_amrex/inputs/hydro_sine_2d.in"


COMMON_OVERRIDES = {
    "pyfluid.run_name": "amr_dynamic_density_cloud",
    "pyfluid.output_dir": str(RUN_ROOT / "density_cloud"),
    "pyfluid.ic": "density_cloud",
    "pyfluid.rho0": "1.0",
    "pyfluid.pressure0": "1.0",
    "pyfluid.amplitude": "0.25",
    "pyfluid.bump_sigma": "0.035",
    "pyfluid.bump_x0": "0.25",
    "pyfluid.bump_y0": "0.5",
    "pyfluid.v1_0": "2.0",
    "pyfluid.v2_0": "0.0",
    "pyfluid.v3_0": "0.0",
    "pyfluid.passive_scalars": "0",
    "pyfluid.recon": "pwl",
    "pyfluid.riemann": "hllc",
    "pyfluid.timestep": "ssprk3",
    "pyfluid.ng": "2",
    "pyfluid.cfl": "0.9",
    "pyfluid.max_step": "80",
    "pyfluid.final_time": "-1",
    "pyfluid.subcycling": "off",
    "pyfluid.scalar_int": "8",
    "pyfluid.plot_int": "8",
    "pyfluid.checkpoint_int": "-1",
    "pyfluid.physical_check_int": "1",
    "amr.n_cell": "64 64",
    "amr.max_level": "1",
    "amr.ref_ratio": "2",
    "amr.max_grid_size": "32",
    "amr.blocking_factor": "8",
    "amr.regrid_int": "4",
    "amr.tag_metric": "auto",
}


def args_from(values: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in values.items()]


def write_input(path: Path, params: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for key in sorted(params):
            f.write(f"{key} = {params[key]}\n")


def plotfiles(run_dir: Path) -> list[Path]:
    plot_dir = run_dir / "stateGF"
    files = sorted_plotfiles(run_dir)
    if len(files) < 3:
        raise RuntimeError(f"expected at least three plotfiles under {plot_dir}")
    return files


def level_centroid_x(plotfile: Path, level: int) -> float:
    boxes = [box for box in plotfile_level_boxes(plotfile, 2) if box.level == level]
    if not boxes:
        raise RuntimeError(f"{plotfile} has no level-{level} AMR boxes")
    total_width = 0.0
    weighted_x = 0.0
    for box in boxes:
        xlo, xhi = box.bounds[0]
        ylo, yhi = box.bounds[1]
        area = (xhi - xlo) * (yhi - ylo)
        total_width += area
        weighted_x += 0.5 * (xlo + xhi) * area
    return weighted_x / total_width


def dynamic_regrid_reason(run_dir: Path, min_shift: float) -> tuple[str, str]:
    files = plotfiles(run_dir)
    first = level_centroid_x(files[0], 1)
    middle = level_centroid_x(files[len(files) // 2], 1)
    last = level_centroid_x(files[-1], 1)
    shift = last - first
    if shift < min_shift:
        return (
            "FAIL",
            f"level-1 box centroid moved {shift:.6g}, expected >= {min_shift:.6g} "
            f"(first={first:.6g}, middle={middle:.6g}, last={last:.6g})",
        )
    if abs(middle - first) < 1.0e-12 and abs(last - middle) < 1.0e-12:
        return "FAIL", "level-1 AMR box centroid was exactly unchanged across plotfiles"
    return (
        "PASS",
        f"level-1 box centroid moved from {first:.6g} to {last:.6g} "
        f"(middle={middle:.6g})",
    )


def run_case(ranks: int, timeout: int | None, min_shift: float) -> dict[str, object]:
    run_dir = RUN_ROOT / "density_cloud"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_file = run_dir / "run.in"
    write_input(input_file, COMMON_OVERRIDES)
    result = run_amrex(
        dim=2,
        input_path=BASE_INPUT,
        ranks=ranks,
        overrides=args_from(COMMON_OVERRIDES),
        timeout=timeout,
    )
    log_path = run_dir / "run.log"
    log_path.write_text("$ " + " ".join(result.command) + "\n" + result.stdout)

    if result.returncode != 0:
        return {
            "kind": "amr",
            "suite": "amr_dynamic_regrid",
            "step": "density_cloud",
            "case": "amr_dynamic_density_cloud",
            "label": "amr_dynamic_density_cloud",
            "status": "FAIL",
            "reason": "AMReX run failed",
            "returncode": result.returncode,
            "seconds": f"{result.seconds:.6g}",
            "dim": "2",
            "input": repo_relative(BASE_INPUT),
            "policy": "closed",
            "ranks": ranks,
            "log": repo_relative(log_path),
            "tail": " | ".join(result.stdout.splitlines()[-6:]),
            "description": "advected density cloud should drag AMR boxes",
            "run_dir": repo_relative(run_dir),
        }

    scalar_row = scalar_validation_row(
        run_dir=run_dir,
        case="amr_dynamic_density_cloud",
        input_file=input_file,
        checks="closed",
        dim=2,
        ranks=ranks,
        base_input=repo_relative(BASE_INPUT),
        kind="amr",
    )
    status = str(scalar_row["status"])
    reason = str(scalar_row["reason"])
    if status == "PASS":
        status, reason = dynamic_regrid_reason(run_dir, min_shift)

    row = {
        **scalar_row,
        "status": status,
        "reason": reason,
        "returncode": result.returncode,
        "seconds": f"{result.seconds:.6g}",
        "kind": "amr",
        "suite": "amr_dynamic_regrid",
        "step": "density_cloud",
        "log": repo_relative(log_path),
        "tail": " | ".join(result.stdout.splitlines()[-6:]),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--min-shift", type=float, default=0.12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and RUN_ROOT.exists():
        import shutil

        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    row = run_case(args.ranks, args.timeout, args.min_shift)
    write_tsv(SUMMARY, [row], VALIDATION_HISTORY_FIELDS)
    append_validation_history(HISTORY, [row])
    print(
        f"{row['status']} amr_dynamic_regrid: {row['reason']} "
        f"Mcells/s={float(row['avg_mcells_per_s']):.3g} "
        f"sim_time/s={float(row['avg_sim_time_per_s']):.3g}"
    )
    return 0 if row["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
