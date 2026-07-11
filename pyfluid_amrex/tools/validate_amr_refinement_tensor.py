#!/usr/bin/env python3
"""Validate automatic directional AMR refinement tensors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from validation_common import (
    OPEN_POLICY,
    ROOT,
    VALIDATION_HISTORY_FIELDS,
    append_validation_history,
    materialize_run_input,
    read_tsv,
    repo_relative,
    run_amrex,
    scalar_validation_row,
    write_tsv,
)


RUN_ROOT = ROOT / "runs/current/amr_refinement_tensor"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"


COMMON = {
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
    "amr.max_level": "3",
    "amr.refinement_tensor": "auto",
    "amr.tag_metric": "auto",
    "amr.tag_threshold": "0.015",
    "amr.max_refinement_ratio": "2",
    "amr.max_grid_size": "32",
    "amr.blocking_factor": "4",
    "amr.regrid_int": "1",
    "amr.grid_eff": "0.80",
    "amr.total_cell_budget": "500000",
}


@dataclass(frozen=True)
class TensorCase:
    name: str
    dim: int
    overrides: dict[str, str]
    refined_dirs: tuple[int, ...]
    smooth_dirs: tuple[int, ...]
    expected_refined_ratio: int = 2


def wave_mode_overrides(dim: int, refined_dirs: tuple[int, ...]) -> dict[str, str]:
    refined = set(refined_dirs)
    return {
        f"pyfluid.wave_mode_{direction}": "1.0" if direction in refined else "0.0"
        for direction in range(1, dim + 1)
    }


def make_tensor_case(
    name: str,
    dim: int,
    base: dict[str, str],
    refined_dirs: tuple[int, ...],
) -> TensorCase:
    overrides = dict(base)
    overrides.update(wave_mode_overrides(dim, refined_dirs))
    smooth_dirs = tuple(direction for direction in range(1, dim + 1) if direction not in refined_dirs)
    return TensorCase(name, dim, overrides, refined_dirs, smooth_dirs)


def make_tensor_case_set(
    prefix: str,
    dim: int,
    base: dict[str, str],
    axis_names: tuple[str, ...],
    refined_sets: tuple[tuple[int, ...], ...],
) -> tuple[TensorCase, ...]:
    cases: list[TensorCase] = []
    for refined_dirs in refined_sets:
        label = "_".join(axis_names[direction - 1] for direction in refined_dirs)
        if len(refined_dirs) == 1:
            name = f"{prefix}_{label}_only_{dim}d"
        else:
            name = f"{prefix}_{label}_{dim}d"
        cases.append(make_tensor_case(name, dim, base, refined_dirs))
    return tuple(cases)


THREE_D_REFINED_SETS = (
    (1,),
    (2,),
    (3,),
    (1, 2),
    (1, 3),
    (2, 3),
    (1, 2, 3),
)


BASE_3D = {
    "amr.n_cell": "8 8 8",
    "amr.max_level": "2",
    "amr.total_cell_budget": "800000",
    "pyfluid.wave_number": "3.0",
}

CARTESIAN_3D = {
    **BASE_3D,
    "mesh.coordinate": "cartesian",
    "mesh.axes": "x y z",
    "geometry.prob_lo": "0.0 0.0 0.0",
    "geometry.prob_hi": "1.0 1.0 1.0",
    "geometry.is_periodic": "1 1 1",
    "pyfluid.bc": "periodic",
}

CYLINDRICAL_3D = {
    **BASE_3D,
    "mesh.coordinate": "cylindrical",
    "mesh.axes": "r phi z",
    "geometry.prob_lo": "0.35 0.0 0.0",
    "geometry.prob_hi": "1.35 6.283185307179586 1.0",
    "geometry.is_periodic": "0 1 1",
    "pyfluid.bc": "outflow",
    "pyfluid.bc_ylo": "periodic",
    "pyfluid.bc_yhi": "periodic",
    "pyfluid.bc_zlo": "periodic",
    "pyfluid.bc_zhi": "periodic",
}

SPHERICAL_3D = {
    **BASE_3D,
    "mesh.coordinate": "spherical",
    "mesh.axes": "r theta phi",
    "geometry.prob_lo": "0.45 0.45 0.0",
    "geometry.prob_hi": "1.45 1.25 6.283185307179586",
    "geometry.is_periodic": "0 0 1",
    "pyfluid.bc": "outflow",
    "pyfluid.bc_zlo": "periodic",
    "pyfluid.bc_zhi": "periodic",
}


CASES = (
    TensorCase(
        "cartesian_x_only",
        2,
        {
            "mesh.coordinate": "cartesian",
            "mesh.axes": "x y",
            "geometry.prob_lo": "0.0 0.0",
            "geometry.prob_hi": "1.0 1.0",
            "geometry.is_periodic": "1 1",
            "amr.n_cell": "16 16",
            "pyfluid.bc": "periodic",
            "pyfluid.wave_mode_1": "1.0",
            "pyfluid.wave_mode_2": "0.0",
        },
        (1,),
        (2,),
    ),
    TensorCase(
        "cartesian_y_only",
        2,
        {
            "mesh.coordinate": "cartesian",
            "mesh.axes": "x y",
            "geometry.prob_lo": "0.0 0.0",
            "geometry.prob_hi": "1.0 1.0",
            "geometry.is_periodic": "1 1",
            "amr.n_cell": "16 16",
            "pyfluid.bc": "periodic",
            "pyfluid.wave_mode_1": "0.0",
            "pyfluid.wave_mode_2": "1.0",
        },
        (2,),
        (1,),
    ),
    TensorCase(
        "cylindrical_r_only",
        2,
        {
            "mesh.coordinate": "cylindrical",
            "mesh.axes": "r phi",
            "geometry.prob_lo": "0.35 0.0",
            "geometry.prob_hi": "1.35 6.283185307179586",
            "geometry.is_periodic": "0 1",
            "amr.n_cell": "16 16",
            "pyfluid.bc": "outflow",
            "pyfluid.bc_ylo": "periodic",
            "pyfluid.bc_yhi": "periodic",
            "pyfluid.wave_mode_1": "1.0",
            "pyfluid.wave_mode_2": "0.0",
        },
        (1,),
        (2,),
    ),
    TensorCase(
        "cylindrical_phi_only",
        2,
        {
            "mesh.coordinate": "cylindrical",
            "mesh.axes": "r phi",
            "geometry.prob_lo": "0.35 0.0",
            "geometry.prob_hi": "1.35 6.283185307179586",
            "geometry.is_periodic": "0 1",
            "amr.n_cell": "16 16",
            "pyfluid.bc": "outflow",
            "pyfluid.bc_ylo": "periodic",
            "pyfluid.bc_yhi": "periodic",
            "pyfluid.wave_mode_1": "0.0",
            "pyfluid.wave_mode_2": "1.0",
        },
        (2,),
        (1,),
    ),
    TensorCase(
        "spherical_r_only",
        2,
        {
            "mesh.coordinate": "spherical",
            "mesh.axes": "r theta",
            "geometry.coord_sys": "2",
            "geometry.prob_lo": "0.45 0.45",
            "geometry.prob_hi": "1.45 1.25",
            "geometry.is_periodic": "0 0",
            "amr.n_cell": "16 16",
            "pyfluid.bc": "outflow",
            "pyfluid.wave_mode_1": "1.0",
            "pyfluid.wave_mode_2": "0.0",
        },
        (1,),
        (2,),
    ),
    TensorCase(
        "spherical_theta_only",
        2,
        {
            "mesh.coordinate": "spherical",
            "mesh.axes": "r theta",
            "geometry.coord_sys": "2",
            "geometry.prob_lo": "0.45 0.45",
            "geometry.prob_hi": "1.45 1.25",
            "geometry.is_periodic": "0 0",
            "amr.n_cell": "16 16",
            "pyfluid.bc": "outflow",
            "pyfluid.wave_mode_1": "0.0",
            "pyfluid.wave_mode_2": "1.0",
        },
        (2,),
        (1,),
    ),
    TensorCase(
        "cartesian_x_strong_ratio4",
        2,
        {
            "mesh.coordinate": "cartesian",
            "mesh.axes": "x y",
            "geometry.prob_lo": "0.0 0.0",
            "geometry.prob_hi": "1.0 1.0",
            "geometry.is_periodic": "1 1",
            "amr.n_cell": "16 16",
            "amr.max_refinement_ratio": "4",
            "amr.tag_threshold": "0.002",
            "pyfluid.bc": "periodic",
            "pyfluid.amplitude": "0.20",
            "pyfluid.wave_mode_1": "1.0",
            "pyfluid.wave_mode_2": "0.0",
        },
        (1,),
        (2,),
        expected_refined_ratio=4,
    ),
    TensorCase(
        "cylindrical_r_strong_ratio4",
        2,
        {
            "mesh.coordinate": "cylindrical",
            "mesh.axes": "r phi",
            "geometry.prob_lo": "0.35 0.0",
            "geometry.prob_hi": "1.35 6.283185307179586",
            "geometry.is_periodic": "0 1",
            "amr.n_cell": "16 16",
            "amr.max_refinement_ratio": "4",
            "amr.tag_threshold": "0.002",
            "pyfluid.bc": "outflow",
            "pyfluid.bc_ylo": "periodic",
            "pyfluid.bc_yhi": "periodic",
            "pyfluid.amplitude": "0.20",
            "pyfluid.wave_mode_1": "1.0",
            "pyfluid.wave_mode_2": "0.0",
        },
        (1,),
        (2,),
        expected_refined_ratio=4,
    ),
    *make_tensor_case_set(
        "cartesian",
        3,
        CARTESIAN_3D,
        ("x", "y", "z"),
        THREE_D_REFINED_SETS,
    ),
    *make_tensor_case_set(
        "cylindrical",
        3,
        CYLINDRICAL_3D,
        ("r", "phi", "z"),
        THREE_D_REFINED_SETS,
    ),
    *make_tensor_case_set(
        "spherical",
        3,
        SPHERICAL_3D,
        ("r", "theta", "phi"),
        THREE_D_REFINED_SETS,
    ),
)


def read_amr_rows(run_dir: Path) -> list[dict[str, str]]:
    rows = read_tsv(run_dir / "amr_levels.tsv", require_complete=True)
    if not rows:
        raise RuntimeError(f"missing AMR rows in {run_dir / 'amr_levels.tsv'}")
    return rows


def read_initial_amr_row(run_dir: Path) -> dict[str, str]:
    return read_amr_rows(run_dir)[0]


def finest_dx(row: dict[str, str], direction: int) -> float:
    return float(row[f"finest_dx{direction}"])


def direction_tags(row: dict[str, str], direction: int) -> float:
    max_level = int(row["max_level"])
    values = [
        float(row.get(f"latest_auto_dir{direction}_tags_L{lev}", 0.0))
        for lev in range(max_level + 1)
    ]
    return max(values)


def ref_ratio(row: dict[str, str], direction: int, level: int) -> int:
    return int(row.get(f"ref_ratio{direction}_L{level}", 1.0))


def check_tensor(case: TensorCase, row: dict[str, str]) -> tuple[str, str]:
    reasons: list[str] = []
    active_level = int(row["active_finest_level"])
    if active_level < 1:
        reasons.append(f"active_finest_level={active_level} < 1")

    for direction in case.refined_dirs:
        if direction_tags(row, direction) <= 0.0:
            reasons.append(f"dir{direction} was never tagged")
    for level in range(active_level):
        for direction in case.smooth_dirs:
            ratio = ref_ratio(row, direction, level)
            if ratio != 1:
                reasons.append(f"ref_ratio{direction}_L{level}={ratio}, expected 1")

    # Directional AMR may resolve one selected direction at an early level and
    # refine only another direction on later levels. Require each selected
    # direction to refine somewhere, rather than imposing an artificial tensor
    # product at every level.
    for direction in case.refined_dirs:
        ratios = [ref_ratio(row, direction, level) for level in range(active_level)]
        if case.expected_refined_ratio not in ratios:
            reasons.append(
                f"dir{direction} never used refinement ratio "
                f"{case.expected_refined_ratio}"
            )

    if reasons:
        return "FAIL", "; ".join(reasons)
    return "PASS", ""


def run_case(case: TensorCase, ranks: int, timeout: int | None) -> dict[str, object]:
    run_dir = RUN_ROOT / case.name
    values = dict(COMMON)
    values.update(case.overrides)
    values["pyfluid.run_name"] = case.name
    values["pyfluid.output_dir"] = repo_relative(run_dir)

    base_input = ROOT / f"pyfluid_amrex/inputs/hydro_sine_{case.dim}d.in"
    resolved_input = materialize_run_input(base_input, run_dir / "run.in", values)
    result = run_amrex(
        dim=case.dim,
        input_path=resolved_input,
        ranks=ranks,
        timeout=timeout,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.log").open("w") as f:
        f.write("$ " + " ".join(result.command) + "\n")
        f.write(f"# wall_seconds = {result.seconds:.6g}\n")
        f.write(result.stdout)
        if result.stdout and not result.stdout.endswith("\n"):
            f.write("\n")
    row: dict[str, object]
    if result.returncode != 0:
        row = {
            "kind": "amr",
            "case": case.name,
            "label": case.name,
            "status": "FAIL",
            "reason": " | ".join(result.stdout.splitlines()[-6:]),
            "returncode": result.returncode,
            "seconds": result.seconds,
            "dim": case.dim,
            "ranks": ranks,
            "run_dir": repo_relative(run_dir),
            "input": repo_relative(base_input),
            "policy": OPEN_POLICY.name,
        }
        return row
    if not resolved_input.exists():
        raise RuntimeError(f"solver did not write canonical run configuration: {resolved_input}")

    row = scalar_validation_row(
        run_dir,
        case=case.name,
        input_file=resolved_input,
        checks=OPEN_POLICY,
        mhd=False,
        dim=case.dim,
        ranks=ranks,
        base_input=repo_relative(base_input),
        kind="amr",
    )
    row["returncode"] = result.returncode
    row["seconds"] = result.seconds

    amr_row = read_initial_amr_row(run_dir)
    tensor_status, tensor_reason = check_tensor(case, amr_row)
    if row["status"] != "PASS":
        tensor_status = "FAIL"
        tensor_reason = f"{row.get('reason', '')}; {tensor_reason}".strip("; ")
    row["status"] = tensor_status
    row["reason"] = tensor_reason
    row["active_finest_level"] = int(amr_row["active_finest_level"])
    row["total_cells"] = int(amr_row["total_cells"])
    for direction in range(1, case.dim + 1):
        row[f"finest_dx{direction}"] = amr_row[f"finest_dx{direction}"]
        row[f"max_dir{direction}_tags"] = direction_tags(amr_row, direction)
        active_level = int(amr_row["active_finest_level"])
        ratios = [str(ref_ratio(amr_row, direction, level)) for level in range(active_level)]
        row[f"ref_ratio{direction}"] = ",".join(ratios)
    row["run_dir"] = repo_relative(run_dir)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--include-3d", action="store_true")
    args = parser.parse_args()

    selected = [case for case in CASES if args.include_3d or case.dim != 3]
    rows = [run_case(case, args.ranks, args.timeout) for case in selected]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = [
        "active_finest_level",
        "total_cells",
        "finest_dx1",
        "finest_dx2",
        "finest_dx3",
        "max_dir1_tags",
        "max_dir2_tags",
        "max_dir3_tags",
        "ref_ratio1",
        "ref_ratio2",
        "ref_ratio3",
    ]
    summary_fields = VALIDATION_HISTORY_FIELDS + extra_fields
    write_tsv(SUMMARY, rows, summary_fields)
    append_validation_history(HISTORY, rows)
    print(f"wrote {repo_relative(SUMMARY)}")
    failed = [row for row in rows if row["status"] != "PASS"]
    for row in rows:
        print(
            f"{row['status']} {row['case']}: L{row.get('active_finest_level', '?')} "
            f"cells={row.get('total_cells', '?')} reason={row.get('reason', '')}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
