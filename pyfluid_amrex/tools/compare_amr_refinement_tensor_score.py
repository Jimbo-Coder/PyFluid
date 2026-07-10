#!/usr/bin/env python3
"""Compare fixed isotropic AMR against automatic directional refinement tensors."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from validation_common import ROOT, read_tsv, repo_relative, run_amrex, write_tsv


RUN_ROOT = ROOT / "runs/current/amr_refinement_tensor_score"
SUMMARY = RUN_ROOT / "summary.tsv"


COMMON = {
    "mesh.coordinate": "cartesian",
    "mesh.axes": "x y",
    "geometry.prob_lo": "0.0 0.0",
    "geometry.prob_hi": "1.0 1.0",
    "geometry.is_periodic": "1 1",
    "amr.n_cell": "16 16",
    "amr.max_level": "4",
    "amr.max_grid_size": "32",
    "amr.blocking_factor": "4",
    "amr.regrid_int": "1",
    "amr.grid_eff": "0.80",
    "amr.tag_metric": "auto",
    "amr.tag_threshold": "0.08",
    "amr.total_cell_budget": "800000",
    "pyfluid.ic": "sin_wave",
    "pyfluid.rho0": "1.0",
    "pyfluid.pressure0": "1.0",
    "pyfluid.amplitude": "0.08",
    "pyfluid.wave_number": "4.0",
    "pyfluid.v1_0": "0.0",
    "pyfluid.v2_0": "0.0",
    "pyfluid.v3_0": "0.0",
    "pyfluid.physics": "hydro",
    "pyfluid.ct": "off",
    "pyfluid.diffusion": "none",
    "pyfluid.gravity": "none",
    "pyfluid.solid_geometry": "off",
    "pyfluid.subcycling": "off",
    "pyfluid.recon": "pwc",
    "pyfluid.riemann": "llf",
    "pyfluid.timestep": "euler",
    "pyfluid.ng": "2",
    "pyfluid.cfl": "0.9",
    "pyfluid.max_step": "2",
    "pyfluid.final_time": "-1",
    "pyfluid.scalar_int": "1",
    "pyfluid.plot_int": "-1",
    "pyfluid.checkpoint_int": "-1",
    "pyfluid.physical_check_int": "1",
    "pyfluid.bc": "periodic",
}


@dataclass(frozen=True)
class ScoreCase:
    name: str
    wave_modes: tuple[float, float]
    useful_dirs: tuple[int, ...]
    description: str


CASES = (
    ScoreCase(
        "x_only",
        (1.0, 0.0),
        (1,),
        "feature varies only along direction 1; best case for anisotropic AMR",
    ),
    ScoreCase(
        "x3_y1",
        (3.0, 1.0),
        (1, 2),
        "feature varies in both directions but has stronger direction-1 content",
    ),
    ScoreCase(
        "diagonal_xy",
        (1.0, 1.0),
        (1, 2),
        "feature varies comparably in both directions; anisotropy should not over-specialize",
    ),
)


FIELDS = (
    "case",
    "branch",
    "status",
    "seconds",
    "description",
    "run_dir",
    "active_finest_level",
    "max_level",
    "total_cells",
    "base_cells",
    "cell_cost",
    "useful_resolution_gain",
    "smooth_resolution_gain",
    "score",
    "stop_reason",
    "touches_max_level",
    "resolution_limited_by_max_level",
    "finest_tags",
    "finest_dx1",
    "finest_dx2",
    "min_feature_length1",
    "min_feature_length2",
    "ref_ratio1",
    "ref_ratio2",
    "reason",
)


def args_from(values: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in values.items()]


def read_amr_rows(run_dir: Path) -> list[dict[str, str]]:
    return read_tsv(run_dir / "amr_levels.tsv", require_complete=True)


def parse_float(row: dict[str, str], name: str, default: float = math.nan) -> float:
    value = row.get(name, "")
    if value == "":
        return default
    return float(value)


def parse_int(row: dict[str, str], name: str, default: int = 0) -> int:
    value = row.get(name, "")
    if value == "":
        return default
    return int(float(value))


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    product = 1.0
    for value in values:
        product *= value
    return product ** (1.0 / len(values))


def ratio_history(row: dict[str, str], direction: int, active_level: int) -> str:
    return ",".join(
        str(parse_int(row, f"ref_ratio{direction}_L{level}", 1))
        for level in range(active_level)
    )


def stop_reason(row: dict[str, str], active_level: int) -> tuple[str, float]:
    max_level = parse_int(row, "max_level")
    touches_max = parse_int(row, "touches_max_level")
    limited_by_max = parse_int(row, "resolution_limited_by_max_level")
    finest_tags = parse_float(row, f"latest_auto_tags_L{active_level}", 0.0)
    if limited_by_max:
        return "max_level_before_cutoff", finest_tags
    if touches_max and finest_tags > 0.0:
        return "max_level_with_active_tags", finest_tags
    min_ratios = [
        parse_float(row, "finest_dx1_over_min_feature_length1", -1.0),
        parse_float(row, "finest_dx2_over_min_feature_length2", -1.0),
    ]
    enabled = [ratio for ratio in min_ratios if ratio >= 0.0]
    if enabled and all(ratio <= 1.0 for ratio in enabled):
        return "min_feature_cutoff", finest_tags
    if finest_tags <= 0.0:
        return "smoothness_metric", finest_tags
    return "active_tags_below_max", finest_tags


def score_row(case: ScoreCase, branch: str, run_dir: Path, seconds: float) -> dict[str, object]:
    rows = read_amr_rows(run_dir)
    final = rows[-1]
    active_level = parse_int(final, "active_finest_level")
    max_level = parse_int(final, "max_level")
    total_cells = parse_float(final, "total_cells")
    n_cell = [int(value) for value in COMMON["amr.n_cell"].split()]
    base_cells = math.prod(n_cell)
    base_dx = [1.0 / n_cell[0], 1.0 / n_cell[1]]
    finest_dx = [parse_float(final, "finest_dx1"), parse_float(final, "finest_dx2")]
    gains = [base_dx[direction] / finest_dx[direction] for direction in range(2)]
    useful = [gains[direction - 1] for direction in case.useful_dirs]
    smooth = [gains[direction] for direction in range(2) if direction + 1 not in case.useful_dirs]
    useful_gain = geometric_mean(useful)
    smooth_gain = geometric_mean(smooth)
    cell_cost = total_cells / base_cells
    score = useful_gain / (cell_cost * smooth_gain)
    reason, finest_tags = stop_reason(final, active_level)
    return {
        "case": case.name,
        "branch": branch,
        "status": "PASS",
        "seconds": seconds,
        "description": case.description,
        "run_dir": repo_relative(run_dir),
        "active_finest_level": active_level,
        "max_level": max_level,
        "total_cells": int(total_cells),
        "base_cells": base_cells,
        "cell_cost": cell_cost,
        "useful_resolution_gain": useful_gain,
        "smooth_resolution_gain": smooth_gain,
        "score": score,
        "stop_reason": reason,
        "touches_max_level": parse_int(final, "touches_max_level"),
        "resolution_limited_by_max_level": parse_int(final, "resolution_limited_by_max_level"),
        "finest_tags": finest_tags,
        "finest_dx1": finest_dx[0],
        "finest_dx2": finest_dx[1],
        "min_feature_length1": parse_float(final, "min_feature_length1", -1.0),
        "min_feature_length2": parse_float(final, "min_feature_length2", -1.0),
        "ref_ratio1": ratio_history(final, 1, active_level),
        "ref_ratio2": ratio_history(final, 2, active_level),
        "reason": "",
    }


def branch_overrides(branch: str, max_level: int, max_refinement_ratio: int) -> dict[str, str]:
    if branch == "fixed_iso":
        return {
            "amr.refinement_tensor": "fixed",
            "amr.ref_ratio": " ".join("2" for _ in range(max_level)),
        }
    if branch == "auto_tensor":
        return {
            "amr.refinement_tensor": "auto",
            "amr.max_refinement_ratio": str(max_refinement_ratio),
        }
    raise ValueError(f"unknown branch {branch}")


def run_case(
    case: ScoreCase,
    branch: str,
    ranks: int,
    timeout: int | None,
    common: dict[str, str],
    max_refinement_ratio: int,
) -> dict[str, object]:
    run_name = f"{case.name}_{branch}"
    run_dir = RUN_ROOT / run_name
    values = dict(common)
    values.update(branch_overrides(branch, int(common["amr.max_level"]), max_refinement_ratio))
    values.update(
        {
            "pyfluid.run_name": run_name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "pyfluid.wave_mode_1": f"{case.wave_modes[0]:.17g}",
            "pyfluid.wave_mode_2": f"{case.wave_modes[1]:.17g}",
        }
    )
    result = run_amrex(
        dim=2,
        input_path=ROOT / "pyfluid_amrex/inputs/hydro_sine_2d.in",
        ranks=ranks,
        overrides=args_from(values),
        timeout=timeout,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.log").open("w") as f:
        f.write("$ " + " ".join(result.command) + "\n")
        f.write(f"# wall_seconds = {result.seconds:.6g}\n")
        f.write(result.stdout)
        if result.stdout and not result.stdout.endswith("\n"):
            f.write("\n")
    with (run_dir / "run.in").open("w") as f:
        for key in sorted(values):
            f.write(f"{key} = {values[key]}\n")
    if result.returncode != 0:
        return {
            "case": case.name,
            "branch": branch,
            "status": "FAIL",
            "seconds": result.seconds,
            "description": case.description,
            "run_dir": repo_relative(run_dir),
            "reason": " | ".join(result.stdout.splitlines()[-6:]),
        }
    return score_row(case, branch, run_dir, result.seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--only-case", action="append", choices=[case.name for case in CASES])
    parser.add_argument("--tag-threshold", type=float, default=float(COMMON["amr.tag_threshold"]))
    parser.add_argument("--max-level", type=int, default=int(COMMON["amr.max_level"]))
    parser.add_argument("--max-refinement-ratio", type=int, default=4)
    parser.add_argument("--total-cell-budget", type=int, default=int(COMMON["amr.total_cell_budget"]))
    args = parser.parse_args()

    common = dict(COMMON)
    common["amr.tag_threshold"] = f"{args.tag_threshold:.17g}"
    common["amr.max_level"] = str(args.max_level)
    common["amr.total_cell_budget"] = str(args.total_cell_budget)
    selected_cases = [case for case in CASES if not args.only_case or case.name in args.only_case]
    branches = ("fixed_iso", "auto_tensor")
    rows = [
        run_case(case, branch, args.ranks, args.timeout, common, args.max_refinement_ratio)
        for case in selected_cases
        for branch in branches
    ]
    failures: list[str] = []
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(SUMMARY, rows, FIELDS)
    print(f"wrote {repo_relative(SUMMARY)}")
    for case in selected_cases:
        fixed = next(row for row in rows if row["case"] == case.name and row["branch"] == "fixed_iso")
        auto = next(row for row in rows if row["case"] == case.name and row["branch"] == "auto_tensor")
        if fixed["status"] != "PASS" or auto["status"] != "PASS":
            print(f"FAIL {case.name}: fixed={fixed['status']} auto={auto['status']}")
            failures.append(f"{case.name}: branch failed")
            continue
        improvement = float(auto["score"]) / float(fixed["score"])
        if case.name == "x_only" and improvement < 2.0:
            failures.append(f"{case.name}: expected auto score at least 2x fixed, got {improvement:.3g}")
        if improvement < 0.9:
            failures.append(f"{case.name}: auto score degraded below 90% of fixed, got {improvement:.3g}")
        print(
            f"{case.name}: score auto/fixed={improvement:.3g} "
            f"fixed={float(fixed['score']):.4g} auto={float(auto['score']):.4g} "
            f"fixed_stop={fixed['stop_reason']} auto_stop={auto['stop_reason']} "
            f"fixed_ref=({fixed['ref_ratio1']};{fixed['ref_ratio2']}) "
            f"auto_ref=({auto['ref_ratio1']};{auto['ref_ratio2']})"
        )
    failed = [row for row in rows if row["status"] != "PASS"]
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failed or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
