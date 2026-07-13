#!/usr/bin/env python3
"""Validate that adaptive AMR boxes move with a moving state feature."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from validation_common import (
    ROOT,
    VALIDATION_HISTORY_FIELDS,
    append_validation_history,
    materialize_run_input,
    repo_relative,
    run_amrex,
    scalar_validation_row,
    write_tsv,
)

from plot_amr_hierarchy import plotfile_level_boxes
from plotfile_reader import (
    latest_plotfile,
    parse_plot_bounds,
    parse_plot_header,
    read_level_state,
    sorted_plotfiles,
)


RUN_ROOT = ROOT / "runs/current/amr_dynamic_regrid"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
BASE_INPUT = ROOT / "pyfluid_amrex/inputs/hydro_sine_2d.in"
CLOUD_SIGMA = 0.035
MIN_ALIGNED_THROUGHPUT_RATIO = 0.90


COMMON_OVERRIDES = {
    "pyfluid.ic": "density_cloud",
    "pyfluid.rho0": "1.0",
    "pyfluid.pressure0": "1.0",
    "pyfluid.amplitude": "0.25",
    "pyfluid.bump_sigma": f"{CLOUD_SIGMA:.17g}",
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
    "pyfluid.final_time": "0.12",
    "pyfluid.subcycling": "off",
    "pyfluid.checkpoint_int": "-1",
    "pyfluid.physical_check_int": "1",
    "amr.n_cell": "64 64",
    "amr.max_level": "2",
    "amr.ref_ratio": "2 2",
    "amr.max_grid_size": "32",
    "amr.blocking_factor": "8",
    "amr.regrid_int": "4",
    "amr.tag_metric": "lohner",
    "amr.min_feature_length": "0.001",
}


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
    level_counts = [
        max((box.level for box in plotfile_level_boxes(path, 2)), default=0)
        for path in files
    ]
    finest_common_level = min(level_counts)
    if finest_common_level < 2:
        return "FAIL", f"moving-cloud hierarchy reached only L{finest_common_level}; expected at least L2"

    movements = []
    failures = []
    for level in range(1, finest_common_level + 1):
        first = level_centroid_x(files[0], level)
        middle = level_centroid_x(files[len(files) // 2], level)
        last = level_centroid_x(files[-1], level)
        shift = last - first
        movements.append(f"L{level} {first:.6g}->{last:.6g} (mid={middle:.6g})")
        if shift < min_shift:
            failures.append(f"L{level} centroid shift {shift:.6g} < {min_shift:.6g}")
        if abs(middle - first) < 1.0e-12 and abs(last - middle) < 1.0e-12:
            failures.append(f"L{level} centroid was exactly unchanged")
    return ("FAIL" if failures else "PASS", "; ".join([*failures, *movements]))


def density_l2_error(run_dir: Path) -> float:
    plotfile = latest_plotfile(run_dir, require=True)
    names, ndim, time_value = parse_plot_header(plotfile)
    if ndim != 2:
        raise RuntimeError(f"expected 2D density-cloud output, got {ndim}D")
    state = read_level_state(plotfile, names, ndim, 0)
    problo, probhi = parse_plot_bounds(plotfile)
    nx, ny = state.shape[1:]
    x = problo[0] + (np.arange(nx) + 0.5) * (probhi[0] - problo[0]) / nx
    y = problo[1] + (np.arange(ny) + 0.5) * (probhi[1] - problo[1]) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    center_x = (0.25 + 2.0 * time_value - problo[0]) % (probhi[0] - problo[0]) + problo[0]
    center_y = 0.5
    dx = (xx - center_x + 0.5 * (probhi[0] - problo[0])) % (probhi[0] - problo[0]) - 0.5 * (probhi[0] - problo[0])
    dy = (yy - center_y + 0.5 * (probhi[1] - problo[1])) % (probhi[1] - problo[1]) - 0.5 * (probhi[1] - problo[1])
    exact = 1.0 + 0.25 * np.exp(-0.5 * (dx * dx + dy * dy) / (CLOUD_SIGMA * CLOUD_SIGMA))
    error = state[names.index("rho")] - exact
    return math.sqrt(float(np.mean(error * error)))


def run_case(
    ranks: int,
    timeout: int | None,
    min_shift: float,
    state_interp: str,
    max_step: int,
    regrid_mode: str,
) -> dict[str, object]:
    method_label = state_interp.replace("-", "_")
    run_name = f"amr_dynamic_density_cloud_{regrid_mode}_{method_label}"
    run_dir = RUN_ROOT / regrid_mode / method_label
    run_dir.mkdir(parents=True, exist_ok=True)
    input_file = materialize_run_input(
        BASE_INPUT,
        run_dir / "run.in",
        {
            **COMMON_OVERRIDES,
            "pyfluid.run_name": run_name,
            "pyfluid.output_dir": str(run_dir),
            "amr.state_interp": state_interp,
            "amr.regrid_mode": regrid_mode,
            "pyfluid.subcycling": "on" if regrid_mode == "aligned" else "off",
            "pyfluid.subcycling_steps": "local_cfl",
            "pyfluid.max_step": str(max_step),
            "pyfluid.scalar_int": str(min(8, max(1, max_step // 10))),
            "pyfluid.plot_int": str(min(8, max(1, max_step // 10))),
        },
    )
    result = run_amrex(
        dim=2,
        input_path=input_file,
        ranks=ranks,
        timeout=timeout,
    )
    log_path = run_dir / "run.log"
    log_path.write_text("$ " + " ".join(result.command) + "\n" + result.stdout)

    if result.returncode != 0:
        return {
            "kind": "amr",
            "suite": "amr_dynamic_regrid",
            "step": "density_cloud",
            "case": run_name,
            "label": run_name,
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
        case=run_name,
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
    rho_l2 = density_l2_error(run_dir) if status == "PASS" else math.nan
    if status == "PASS":
        reason += f"; rho_l2={rho_l2:.6e}"

    row = {
        **scalar_row,
        "status": status,
        "reason": reason,
        "returncode": result.returncode,
        "seconds": f"{result.seconds:.6g}",
        "kind": "amr",
        "suite": "amr_dynamic_regrid",
        "step": "density_cloud",
        "regrid_mode": regrid_mode,
        "state_interp": state_interp,
        "rho_l2_error": rho_l2,
        "log": repo_relative(log_path),
        "tail": " | ".join(result.stdout.splitlines()[-6:]),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--min-shift", type=float, default=0.12)
    parser.add_argument("--max-step", type=int, default=200)
    parser.add_argument(
        "--state-interp",
        choices=("conservative", "protected", "quartic-conservative"),
        default="quartic-conservative",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and RUN_ROOT.exists():
        import shutil

        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    rows = [
        run_case(args.ranks, args.timeout, args.min_shift, args.state_interp, args.max_step, mode)
        for mode in ("synchronized", "aligned")
    ]
    by_mode = {str(row["regrid_mode"]): row for row in rows}
    synchronized = by_mode["synchronized"]
    aligned = by_mode["aligned"]
    synchronized_error = float(synchronized["rho_l2_error"])
    aligned_error = float(aligned["rho_l2_error"])
    accuracy_ratio = aligned_error / synchronized_error if synchronized_error > 0.0 else math.nan
    speedup = float(aligned["avg_sim_time_per_s"]) / float(synchronized["avg_sim_time_per_s"])
    aligned["aligned_accuracy_ratio"] = accuracy_ratio
    aligned["aligned_sim_time_speedup"] = speedup
    comparison_failures = []
    if not math.isfinite(accuracy_ratio) or accuracy_ratio > 1.05:
        comparison_failures.append(f"aligned/synchronized rho L2 ratio {accuracy_ratio:.3f} > 1.05")
    if speedup < MIN_ALIGNED_THROUGHPUT_RATIO:
        comparison_failures.append(
            f"aligned simulation-time throughput ratio {speedup:.3f}x < {MIN_ALIGNED_THROUGHPUT_RATIO:.2f}x"
        )
    if comparison_failures:
        aligned["status"] = "FAIL"
        aligned["reason"] = "; ".join([str(aligned.get("reason", "")), *comparison_failures])

    write_tsv(
        SUMMARY,
        rows,
        [
            *VALIDATION_HISTORY_FIELDS,
            "regrid_mode",
            "state_interp",
            "rho_l2_error",
            "aligned_accuracy_ratio",
            "aligned_sim_time_speedup",
        ],
    )
    append_validation_history(HISTORY, rows)
    for row in rows:
        print(
            f"{row['status']} amr_dynamic_regrid {row['case']}: {row['reason']} "
            f"Mcells/s={float(row['avg_mcells_per_s']):.3g} "
            f"sim_time/s={float(row['avg_sim_time_per_s']):.3g}"
        )
    print(f"aligned comparison: rho_L2={accuracy_ratio:.6f}x sim_time/s={speedup:.3f}x")
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
