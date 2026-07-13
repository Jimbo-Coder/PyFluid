#!/usr/bin/env python3
"""Validate filling of newly uncovered cells from a shrinking solid circle."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from validation_common import materialize_run_input, parameters_from_assignments, run_mpi

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_2d_amr.in"
DEFAULT_OUTPUT = ROOT / "runs" / "current" / "pulsating_vacuum_fill_validation"
DEFAULT_PLOT_DIR = ROOT / "plots" / "current" / "IC" / "pulsating_circle"

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
STEPS_RE = re.compile(r"steps=(\d+)")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")

RADIUS = 0.22
AMPLITUDE = 0.08
PHASE = 0.5 * math.pi
FINAL_TIME = 0.5


@dataclass(frozen=True)
class Case:
    name: str
    overrides: tuple[str, ...]
    step_limit: int
    mass_tol: float
    min_rho_floor: float
    min_pressure_floor: float
    require_amr: bool = False


CASES = (
    Case(
        name="single_level_shrink",
        overrides=(
            "amr.n_cell=128 128",
            "amr.max_level=0",
        ),
        step_limit=800,
        mass_tol=1.0e-12,
        min_rho_floor=1.0e-3,
        min_pressure_floor=1.0e-3,
    ),
    Case(
        name="amr_shrink",
        overrides=(
            "amr.n_cell=64 64",
            "amr.max_level=2",
            "amr.ref_ratio=2 2",
            "amr.regrid_int=4",
            "amr.grid_eff=0.8",
            "amr.n_error_buf=3 3",
            "amr.tag_metric=lohner",
            "amr.solid_refine_band_cells=2.0",
            "amr.solid_curvature_error=0.0005",
            "amr.level_cell_budget_side=384",
            "amr.total_cell_budget=250000",
            "amr.state_interp=conservative",
        ),
        step_limit=4000,
        mass_tol=1.0e-11,
        min_rho_floor=1.0e-3,
        min_pressure_floor=1.0e-3,
        require_amr=True,
    ),
)

COMMON_OVERRIDES = (
    "geometry.prob_lo=0.0 0.0",
    "geometry.prob_hi=1.0 1.0",
    "geometry.is_periodic=0 0",
    "amr.max_grid_size=32",
    "amr.blocking_factor=8",
    "pyfluid.physics=hydro",
    "pyfluid.ct=off",
    "pyfluid.ic=uniform",
    "pyfluid.rho0=1.0",
    "pyfluid.pressure0=1.0",
    "pyfluid.v1_0=0.0",
    "pyfluid.v2_0=0.0",
    "pyfluid.v3_0=0.0",
    "pyfluid.gamma=1.6666666666666667",
    "pyfluid.bc=reflecting",
    "pyfluid.ng=2",
    "pyfluid.recon=pwl",
    "pyfluid.riemann=hllc",
    "pyfluid.timestep=ssprk3",
    "pyfluid.cfl=0.9",
    "pyfluid.diffusion=none",
    "pyfluid.subcycling=off",
    "pyfluid.solid_geometry=circle",
    "pyfluid.solid_motion=pulse",
    "pyfluid.solid_wall_condition=slip",
    "pyfluid.solid_center_x=0.5",
    "pyfluid.solid_center_y=0.5",
    f"pyfluid.solid_radius={RADIUS:.17g}",
    f"pyfluid.solid_radius_amplitude={AMPLITUDE:.17g}",
    "pyfluid.solid_radius_frequency=1.0",
    f"pyfluid.solid_radius_phase={PHASE:.17g}",
    "pyfluid.solid_volume_fraction_samples_per_dim=4",
    "pyfluid.solid_swept_reference_time=0.0",
    "pyfluid.max_step=100000",
    f"pyfluid.final_time={FINAL_TIME:.17g}",
    "pyfluid.scalar_int=32",
    "pyfluid.plot_int=100000",
    "pyfluid.checkpoint_int=-1",
    "pyfluid.aux_gfs=solid_mask,solid_volume_fraction",
    "pyfluid.aux_gf_compute_int=-1",
    "pyfluid.aux_gf_int=100000",
)


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def next_plot_path(directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = directory / f"{stem}_{index:03d}.png"
        if not path.exists():
            return path
        index += 1


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        for row in reader:
            if any(row.get(field) in (None, "") for field in fields):
                break
            rows.append({field: float(row[field]) for field in fields})
    return rows


def read_active_finest_level(run_dir: Path) -> int:
    path = run_dir / "amr_levels.tsv"
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return 0
    return max(int(float(row.get("active_finest_level", "0"))) for row in rows)


def exact_area_delta() -> float:
    initial_radius = RADIUS + AMPLITUDE * math.sin(PHASE)
    final_phase = PHASE + 2.0 * math.pi * FINAL_TIME
    final_radius = RADIUS + AMPLITUDE * math.sin(final_phase)
    return math.pi * (final_radius * final_radius - initial_radius * initial_radius)


def run_case(case: Case, args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.output_dir / case.name
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        args.input,
        run_dir / "run.in",
        parameters_from_assignments(
            [f"pyfluid.output_dir={repo_relative(run_dir)}", *COMMON_OVERRIDES, *case.overrides]
        ),
    )
    completed = run_mpi(
        args.executable,
        [input_path],
        ranks=args.ranks,
    )
    rows = read_rows(run_dir / "scalars.tsv") if (run_dir / "scalars.tsv").exists() else []
    final = rows[-1] if rows else {}
    failures: list[str] = []
    steps_match = STEPS_RE.search(completed.stdout)
    average_match = AVERAGE_RE.search(completed.stdout)
    sim_time_match = SIM_TIME_RE.search(completed.stdout)
    steps = int(steps_match.group(1)) if steps_match else -1
    active_finest_level = read_active_finest_level(run_dir)

    if completed.returncode != 0:
        failures.append(f"returncode={completed.returncode}")
    if not rows:
        failures.append("missing scalar output")
    else:
        if abs(final["time"] - FINAL_TIME) > 1.0e-12:
            failures.append(f"final time={final['time']:.17g} != {FINAL_TIME:.17g}")
        if final["mass_rel_drift"] > case.mass_tol:
            failures.append(f"mass drift={final['mass_rel_drift']:.3e} > {case.mass_tol:.3e}")
        if final["min_rho"] <= case.min_rho_floor or not math.isfinite(final["min_rho"]):
            failures.append(f"min rho={final['min_rho']:.3e} <= {case.min_rho_floor:.3e}")
        if final["min_pressure"] <= case.min_pressure_floor or not math.isfinite(final["min_pressure"]):
            failures.append(f"min pressure={final['min_pressure']:.3e} <= {case.min_pressure_floor:.3e}")
    if steps <= 0 or steps > case.step_limit:
        failures.append(f"steps={steps} outside (0, {case.step_limit}]")
    if case.require_amr and active_finest_level < 1:
        failures.append(f"active finest level={active_finest_level} < 1")

    return {
        "case": case,
        "status": "pass" if not failures else "fail",
        "failures": "; ".join(failures),
        "rows": rows,
        "run_dir": run_dir,
        "steps": steps,
        "active_finest_level": active_finest_level,
        "mcells_s": float(average_match.group(1)) if average_match else math.nan,
        "sim_time_per_s": float(sim_time_match.group(1)) if sim_time_match else math.nan,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-8:]),
    }


def make_plot(results: list[dict[str, object]], output: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    fig.suptitle(
        "Pulsating circle vacuum-fill check: shrink/uncover annulus, "
        "hydro 2D, PWL/HLLC/SSPRK3, CFL=0.9"
    )
    for result in results:
        rows = result["rows"]
        if not rows:
            continue
        label = result["case"].name
        t = np.asarray([row["time"] for row in rows])
        mass = np.asarray([row["mass_rel_drift"] for row in rows])
        min_rho = np.asarray([row["min_rho"] for row in rows])
        min_p = np.asarray([row["min_pressure"] for row in rows])
        dt = np.asarray([row["dt"] for row in rows])
        wall_residual = np.asarray([abs(row["energy_minus_wall_work"]) for row in rows])
        markers = {"marker": "o"} if len(rows) <= 20 else {}

        axs[0, 0].semilogy(t, np.maximum(mass, 1.0e-18), label=label, linewidth=2.0, **markers)
        axs[0, 1].plot(t, min_rho, label=f"{label} rho", linewidth=2.0, **markers)
        axs[0, 1].plot(t, min_p, label=f"{label} P", linewidth=2.0, linestyle="--", **markers)
        axs[1, 0].plot(t, dt, label=label, linewidth=2.0, **markers)
        axs[1, 1].semilogy(t, np.maximum(wall_residual, 1.0e-18), label=label, linewidth=2.0, **markers)

    axs[0, 0].set_title("Mass Drift")
    axs[0, 1].set_title("Positive Floors")
    axs[1, 0].set_title("Timestep")
    axs[1, 1].set_title("|E - E0 - Wall Work|")
    for ax in axs.flat:
        ax.set_xlabel("time")
        ax.grid(True, which="major", alpha=0.25)
        ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_results(results: list[dict[str, object]], output: Path, plot_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "status",
                "steps",
                "active_finest_level",
                "mass_rel_drift",
                "min_rho",
                "min_pressure",
                "mcells_s",
                "sim_time_per_s",
                "exact_solid_area_delta",
                "plot",
                "run_dir",
                "failures",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for result in results:
            rows = result["rows"]
            final = rows[-1] if rows else {}
            writer.writerow(
                {
                    "case": result["case"].name,
                    "status": result["status"],
                    "steps": result["steps"],
                    "active_finest_level": result["active_finest_level"],
                    "mass_rel_drift": f"{final.get('mass_rel_drift', math.nan):.17e}",
                    "min_rho": f"{final.get('min_rho', math.nan):.17e}",
                    "min_pressure": f"{final.get('min_pressure', math.nan):.17e}",
                    "mcells_s": f"{result['mcells_s']:.8e}",
                    "sim_time_per_s": f"{result['sim_time_per_s']:.8e}",
                    "exact_solid_area_delta": f"{exact_area_delta():.17e}",
                    "plot": repo_relative(plot_path),
                    "run_dir": repo_relative(result["run_dir"]),
                    "failures": result["failures"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_OUTPUT / "results.tsv")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = [run_case(case, args) for case in CASES]
    plot_path = next_plot_path(args.plot_dir, "pulsating_vacuum_fill")
    make_plot(results, plot_path)
    write_results(results, args.results, plot_path)

    failed = False
    for result in results:
        rows = result["rows"]
        final = rows[-1] if rows else {}
        print(
            f"{result['status'].upper()} {result['case'].name}: "
            f"steps={result['steps']}, mass={final.get('mass_rel_drift', math.nan):.3e}, "
            f"min_rho={final.get('min_rho', math.nan):.3e}, "
            f"min_P={final.get('min_pressure', math.nan):.3e}, "
            f"{result['mcells_s']:.2f} Mcells/s"
        )
        if result["status"] != "pass":
            failed = True
            print(result["failures"])
            print(result["stdout_tail"])
    print(f"plot {repo_relative(plot_path)}")
    print(f"results {repo_relative(args.results)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
