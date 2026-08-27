#!/usr/bin/env python3
"""Re-find saved Kerr horizons without advancing the evolution state."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

from validation_common import (
    ROOT,
    build_cached_target,
    materialize_run_input,
    read_tsv,
    repo_relative,
    run_amrex,
    write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/kerr_schild_horizon_amr_3d.in"
DEFAULT_RUN = ROOT / "runs/current/black_hole_long/kerr_schild_horizon_amr_3d"
DEFAULT_OUTPUT = ROOT / "runs/current/black_hole_long/horizon_convergence"
CHECKPOINT_STEP = re.compile(r"chk(\d+)$")
RECOVERY = re.compile(
    r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)
FINDER_CASES = (
    ("p16", 48, 16, 5.0e-4),
    ("p24", 72, 24, 1.0e-4),
    ("p32", 96, 32, 2.0e-5),
)
EVOLUTION_CASES = (
    ("dx0p5", 0.5, 1),
    ("dx0p25", 0.25, 2),
    ("dx0p125", 0.125, 3),
)


def checkpoint_step(checkpoint: Path) -> int:
    match = CHECKPOINT_STEP.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(f"checkpoint name does not encode a step: {checkpoint}")
    return int(match.group(1))


def exact_kerr_area(spin: float) -> float:
    if not math.isfinite(spin) or abs(spin) >= 1.0:
        raise ValueError("dimensionless Kerr spin must satisfy abs(spin) < 1")
    return 8.0 * math.pi * (1.0 + math.sqrt(1.0 - spin * spin))


def refind_checkpoint(
    checkpoint: Path,
    output_root: Path,
    *,
    ranks: int,
    timeout: int | None,
    spin: float,
    finder_cases: tuple[tuple[str, int, int, float], ...] = FINDER_CASES,
) -> list[dict[str, object]]:
    step = checkpoint_step(checkpoint)
    rows: list[dict[str, object]] = []
    exact_area = exact_kerr_area(spin)
    for case_name, radial_points, polar_points, tolerance in finder_cases:
        run_name = f"step{step:05d}_{case_name}"
        run_dir = output_root / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        input_path = materialize_run_input(
            BASE_INPUT,
            run_dir / "input.in",
            {
                "pyfluid.run_name": run_name,
                "pyfluid.output_dir": repo_relative(run_dir),
                "pyfluid.restart_file": repo_relative(checkpoint),
                "pyfluid.max_step": str(step),
                "spacetime.dimensionless_spin3": f"{spin:.17g}",
                "pyfluid.scalar_int": str(max(1, step)),
                "pyfluid.plot_int": "-1",
                "pyfluid.aux_gf_int": "-1",
                "pyfluid.checkpoint_int": "-1",
                "horizon.find_int": "1",
                "horizon.radial_points": str(radial_points),
                "horizon.polar_points": str(polar_points),
                "horizon.coarsest_polar_points": "8",
                "horizon.theta_linf_times_mass_tolerance": f"{tolerance:.17g}",
                "horizon.theta_l2_times_mass_tolerance": f"{tolerance:.17g}",
                "horizon.max_iterations_per_level": "50000",
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
            raise RuntimeError(
                f"{run_name} failed; see {repo_relative(run_dir / 'run.log')}"
            )
        horizon_rows = read_tsv(run_dir / "apparent_horizon.tsv")
        if len(horizon_rows) != 1 or int(horizon_rows[0]["step"]) != step:
            raise RuntimeError(f"{run_name} did not write exactly one surface")
        horizon = horizon_rows[0]
        area = float(horizon["area"])
        rows.append(
            {
                "checkpoint_step": step,
                "case": case_name,
                "radial_points": radial_points,
                "polar_points": polar_points,
                "tolerance": tolerance,
                "iterations": int(horizon["iterations"]),
                "theta_linf_times_mass": float(
                    horizon["theta_linf_times_mass"]
                ),
                "theta_l2_times_mass": float(horizon["theta_l2_times_mass"]),
                "area": area,
                "area_relative_error": (area - exact_area) / exact_area,
                "irreducible_mass": float(horizon["irreducible_mass"]),
                "minimum_radius": float(horizon["minimum_radius"]),
                "maximum_radius": float(horizon["maximum_radius"]),
                "mean_radius": float(horizon["mean_radius"]),
                "run_dir": repo_relative(run_dir),
            }
        )
    return rows


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = list((run_dir / "checkpoints").glob("chk*"))
    if not checkpoints:
        raise RuntimeError(f"no checkpoint was written under {run_dir}")
    return max(checkpoints, key=checkpoint_step)


def run_evolution_convergence(
    output_root: Path,
    *,
    ranks: int,
    timeout: int | None,
    spin: float,
) -> list[dict[str, object]]:
    exact_area = exact_kerr_area(spin)
    rows: list[dict[str, object]] = []
    for case_name, finest_dx, expected_level in EVOLUTION_CASES:
        run_dir = output_root / "evolution" / case_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        input_path = materialize_run_input(
            BASE_INPUT,
            run_dir / "input.in",
            {
                "pyfluid.run_name": case_name,
                "pyfluid.output_dir": repo_relative(run_dir),
                "pyfluid.final_time": "2.5",
                "pyfluid.max_step": "64",
                "pyfluid.plot_int": "-1",
                "pyfluid.aux_gf_int": "-1",
                "pyfluid.checkpoint_int": "100000",
                "spacetime.dimensionless_spin3": f"{spin:.17g}",
                "amr.min_feature_length": f"{finest_dx:.17g}",
                "amr.total_cell_budget": "600000",
                "amr.level_cell_budget": "300000",
                "horizon.find_int": "1000",
                "horizon.radial_points": "72",
                "horizon.polar_points": "24",
                "horizon.coarsest_polar_points": "8",
                "horizon.theta_linf_times_mass_tolerance": "1.0e-4",
                "horizon.theta_l2_times_mass_tolerance": "1.0e-4",
                "horizon.max_iterations_per_level": "50000",
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
            raise RuntimeError(
                f"{case_name} failed; see {repo_relative(run_dir / 'run.log')}"
            )
        recovery = RECOVERY.findall(completed.stdout)
        if not recovery or tuple(int(value) for value in recovery[-1]) != (0, 0):
            raise RuntimeError(f"{case_name} used relativistic recovery")
        initial_horizons = read_tsv(run_dir / "apparent_horizon.tsv")
        if len(initial_horizons) != 1 or int(initial_horizons[0]["step"]) != 0:
            raise RuntimeError(f"{case_name} did not write its initial horizon")
        scalar_rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
        amr_rows = read_tsv(run_dir / "amr_levels.tsv", require_complete=True)
        final_amr = amr_rows[-1]
        actual_level = int(final_amr["active_finest_level"])
        actual_dx = float(final_amr["finest_dx"])
        if actual_level != expected_level or not math.isclose(
            actual_dx, finest_dx, rel_tol=0.0, abs_tol=1.0e-14
        ):
            raise RuntimeError(
                f"{case_name} did not realize its requested hierarchy: "
                f"level={actual_level}, dx={actual_dx:.17g}"
            )
        if int(final_amr["touches_max_level"]) != 0 or any(
            float(row["total_budget_fraction"]) >= 1.0 for row in amr_rows
        ):
            raise RuntimeError(f"{case_name} exhausted its AMR headroom or budget")
        final_checkpoint = latest_checkpoint(run_dir)
        final_step = checkpoint_step(final_checkpoint)
        final_rows = refind_checkpoint(
            final_checkpoint,
            run_dir / "final_refind",
            ranks=ranks,
            timeout=timeout,
            spin=spin,
            finder_cases=(FINDER_CASES[1],),
        )
        final_horizon = next(row for row in final_rows if row["case"] == "p24")
        initial_area = float(initial_horizons[0]["area"])
        final_area = float(final_horizon["area"])
        rows.append(
            {
                "case": case_name,
                "finest_dx": finest_dx,
                "final_step": final_step,
                "final_time": float(scalar_rows[-1]["time"]),
                "total_cells": int(final_amr["total_cells"]),
                "peak_total_cells": max(
                    int(row["total_cells"]) for row in amr_rows
                ),
                "active_finest_level": actual_level,
                "initial_area": initial_area,
                "initial_area_relative_error": (initial_area - exact_area)
                / exact_area,
                "final_area": final_area,
                "final_area_relative_error": (final_area - exact_area) / exact_area,
                "area_change": (final_area - initial_area) / initial_area,
                "final_mean_radius": float(final_horizon["mean_radius"]),
                "final_theta_linf_times_mass": float(
                    final_horizon["theta_linf_times_mass"]
                ),
                "run_dir": repo_relative(run_dir),
            }
        )
    for index, row in enumerate(rows):
        row["final_area_observed_order"] = math.nan
        row["area_change_observed_order"] = math.nan
        if index == 0:
            continue
        previous = rows[index - 1]
        row["final_area_observed_order"] = math.log2(
            abs(float(previous["final_area_relative_error"]))
            / abs(float(row["final_area_relative_error"]))
        )
        row["area_change_observed_order"] = math.log2(
            abs(float(previous["area_change"]))
            / abs(float(row["area_change"]))
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        help="checkpoint folders; defaults to steps 32 and 64 of the long run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--spin", type=float, default=0.5)
    parser.add_argument(
        "--evolve-resolution",
        action="store_true",
        help="run a dx=0.5, 0.25, 0.125 evolution comparison to t=2.5 M",
    )
    args = parser.parse_args()

    build_cached_target("pyfluid_amrex_3d")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.evolve_resolution:
        if args.checkpoints:
            parser.error(
                "explicit checkpoints cannot be combined with "
                "--evolve-resolution"
            )
        rows = run_evolution_convergence(
            args.output,
            ranks=args.ranks,
            timeout=args.timeout,
            spin=args.spin,
        )
        write_tsv(
            args.output / "evolution_convergence.tsv",
            rows,
            tuple(rows[0]),
        )
        for row in rows:
            print(
                f"dx={row['finest_dx']} cells={row['total_cells']} "
                f"initial_error={100.0 * row['initial_area_relative_error']:.4f}% "
                f"final_error={100.0 * row['final_area_relative_error']:.4f}% "
                f"area_change={100.0 * row['area_change']:.4f}%"
            )
        return 0

    checkpoints = args.checkpoints or [
        DEFAULT_RUN / "checkpoints/chk00032",
        DEFAULT_RUN / "checkpoints/chk00064",
    ]
    for checkpoint in checkpoints:
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)

    rows = [
        row
        for checkpoint in checkpoints
        for row in refind_checkpoint(
            checkpoint,
            args.output,
            ranks=args.ranks,
            timeout=args.timeout,
            spin=args.spin,
        )
    ]
    fields = tuple(rows[0])
    write_tsv(args.output / "horizon_convergence.tsv", rows, fields)
    for row in rows:
        print(
            f"step={row['checkpoint_step']} {row['case']} "
            f"MTheta_Linf={row['theta_linf_times_mass']:.3e} "
            f"area_error={100.0 * row['area_relative_error']:.4f}% "
            f"mean_radius={row['mean_radius']:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
