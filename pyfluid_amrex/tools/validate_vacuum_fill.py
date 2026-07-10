#!/usr/bin/env python3
"""Validate atmosphere/near-vacuum fill from a half-domain pressure jump."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from validation_common import mpi_prefix

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pyfluid_amrex" / "tools"))
from plot_amrex_boxed_movie import composite_slice  # noqa: E402
from plotfile_reader import latest_plotfile  # noqa: E402


DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_2d_amr.in"
DEFAULT_OUTPUT = ROOT / "runs" / "current" / "vacuum_fill_validation"
DEFAULT_PLOT_DIR = ROOT / "plots" / "current" / "IC" / "near_vacuum_half"

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
STEPS_RE = re.compile(r"steps=(\d+)")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")

RHO_FLOOR = 1.0e-12
PRESSURE_FLOOR = 1.0e-12
FINAL_TIME = 0.12


@dataclass(frozen=True)
class Case:
    name: str
    label: str
    recon: str
    ng: int
    rho_atmosphere: float
    pressure_atmosphere: float
    amr: bool
    step_limit: int
    mass_tol: float


CASES = (
    Case(
        name="pwl_single_harsh",
        label="PWL harsh",
        recon="pwl",
        ng=2,
        rho_atmosphere=1.0e-6,
        pressure_atmosphere=1.0e-8,
        amr=False,
        step_limit=900,
        mass_tol=1.0e-4,
    ),
    Case(
        name="pwl_amr_harsh",
        label="PWL AMR harsh",
        recon="pwl",
        ng=2,
        rho_atmosphere=1.0e-6,
        pressure_atmosphere=1.0e-8,
        amr=True,
        step_limit=1500,
        mass_tol=1.0e-4,
    ),
    Case(
        name="wenoz_amr_moderate",
        label="WENO-Z AMR moderate",
        recon="wenoz",
        ng=3,
        rho_atmosphere=1.0e-4,
        pressure_atmosphere=1.0e-6,
        amr=True,
        step_limit=1500,
        mass_tol=1.0e-4,
    ),
)


COMMON_OVERRIDES = (
    "geometry.prob_lo=0.0 0.0",
    "geometry.prob_hi=1.0 1.0",
    "geometry.is_periodic=0 1",
    "amr.max_grid_size=64",
    "amr.blocking_factor=8",
    "pyfluid.physics=hydro",
    "pyfluid.ct=off",
    "pyfluid.ic=sod_x",
    "pyfluid.shock_x0=0.5",
    "pyfluid.rho_left=1.0",
    "pyfluid.pressure_left=1.0",
    "pyfluid.v1_left=0.0",
    "pyfluid.v2_left=0.0",
    "pyfluid.v3_left=0.0",
    "pyfluid.v1_right=0.0",
    "pyfluid.v2_right=0.0",
    "pyfluid.v3_right=0.0",
    f"pyfluid.rho_floor={RHO_FLOOR:.17g}",
    f"pyfluid.pressure_floor={PRESSURE_FLOOR:.17g}",
    "pyfluid.bc=outflow",
    "pyfluid.bc_xlo=outflow",
    "pyfluid.bc_xhi=outflow",
    "pyfluid.bc_ylo=periodic",
    "pyfluid.bc_yhi=periodic",
    "pyfluid.riemann=hllc",
    "pyfluid.timestep=ssprk3",
    "pyfluid.cfl=0.9",
    "pyfluid.diffusion=none",
    "pyfluid.subcycling=off",
    "pyfluid.max_step=100000",
    f"pyfluid.final_time={FINAL_TIME:.17g}",
    "pyfluid.scalar_int=20",
    "pyfluid.plot_int=100000",
    "pyfluid.checkpoint_int=-1",
    "pyfluid.aux_gfs=none",
    "pyfluid.aux_gf_compute_int=-1",
    "pyfluid.aux_gf_int=-1",
)

SINGLE_OVERRIDES = (
    "amr.n_cell=256 64",
    "amr.max_level=0",
)

AMR_OVERRIDES = (
    "amr.n_cell=128 32",
    "amr.max_level=2",
    "amr.ref_ratio=2 2",
    "amr.regrid_int=4",
    "amr.grid_eff=0.8",
    "amr.n_error_buf=3 3",
    "amr.tag_metric=auto",
    "amr.state_interp=conservative",
    "amr.level_cell_budget_side=384",
    "amr.total_cell_budget=250000",
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


def run_case(case: Case, args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.output_dir / case.name
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *mpi_prefix(args.ranks),
        str(args.executable),
        str(args.input),
        f"pyfluid.output_dir={repo_relative(run_dir)}",
        *COMMON_OVERRIDES,
        *(AMR_OVERRIDES if case.amr else SINGLE_OVERRIDES),
        f"pyfluid.recon={case.recon}",
        f"pyfluid.ng={case.ng}",
        f"pyfluid.rho_right={case.rho_atmosphere:.17g}",
        f"pyfluid.pressure_right={case.pressure_atmosphere:.17g}",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
        check=False,
    )
    rows = read_rows(run_dir / "scalars.tsv") if (run_dir / "scalars.tsv").exists() else []
    final = rows[-1] if rows else {}
    steps_match = STEPS_RE.search(completed.stdout)
    average_match = AVERAGE_RE.search(completed.stdout)
    sim_time_match = SIM_TIME_RE.search(completed.stdout)
    steps = int(steps_match.group(1)) if steps_match else -1
    active_finest_level = read_active_finest_level(run_dir)
    failures: list[str] = []
    if completed.returncode != 0:
        failures.append(f"returncode={completed.returncode}")
    if not rows:
        failures.append("missing scalar output")
    else:
        if abs(final["time"] - FINAL_TIME) > 1.0e-12:
            failures.append(f"final time={final['time']:.17g} != {FINAL_TIME:.17g}")
        if not math.isfinite(final["mass_rel_drift"]) or final["mass_rel_drift"] > case.mass_tol:
            failures.append(f"mass drift={final['mass_rel_drift']:.3e} > {case.mass_tol:.3e}")
        if not math.isfinite(final["min_rho"]) or final["min_rho"] <= RHO_FLOOR:
            failures.append(f"min rho={final['min_rho']:.3e} <= {RHO_FLOOR:.3e}")
        if not math.isfinite(final["min_pressure"]) or final["min_pressure"] <= PRESSURE_FLOOR:
            failures.append(f"min pressure={final['min_pressure']:.3e} <= {PRESSURE_FLOOR:.3e}")
    if steps <= 0 or steps > case.step_limit:
        failures.append(f"steps={steps} outside (0, {case.step_limit}]")
    if case.amr and active_finest_level < 1:
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
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-10:]),
    }


def make_scalar_plot(results: list[dict[str, object]], output: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    fig.suptitle(
        "Near-vacuum half-domain fill: sod_x atmosphere, hydro 2D, HLLC/SSPRK3, CFL=0.9"
    )
    for result in results:
        rows = result["rows"]
        if not rows:
            continue
        case = result["case"]
        label = case.label
        t = np.asarray([row["time"] for row in rows])
        markers = {"marker": "o"} if len(rows) <= 20 else {}
        axs[0, 0].semilogy(
            t,
            np.maximum([row["mass_rel_drift"] for row in rows], 1.0e-18),
            label=label,
            linewidth=2.0,
            **markers,
        )
        axs[0, 1].semilogy(t, [row["min_rho"] for row in rows], label=f"{label} rho", linewidth=2.0, **markers)
        axs[0, 1].semilogy(
            t,
            [row["min_pressure"] for row in rows],
            label=f"{label} P",
            linewidth=2.0,
            linestyle="--",
            **markers,
        )
        axs[1, 0].plot(t, [row["dt"] for row in rows], label=label, linewidth=2.0, **markers)
        axs[1, 1].plot(t, [row["mcells_per_s"] for row in rows], label=label, linewidth=2.0, **markers)
    axs[0, 0].set_title("Mass Drift")
    axs[0, 1].set_title("Positive Floors")
    axs[1, 0].set_title("Timestep")
    axs[1, 1].set_title("Throughput")
    axs[1, 1].set_ylabel("Mcells/s")
    for ax in axs.flat:
        ax.set_xlabel("time")
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_density_plot(results: list[dict[str, object]], output: Path) -> None:
    plot_results = [result for result in results if latest_plotfile(result["run_dir"]) is not None]
    if not plot_results:
        return
    fig, axs = plt.subplots(1, len(plot_results), figsize=(5.1 * len(plot_results), 4.5), constrained_layout=True)
    if len(plot_results) == 1:
        axs = [axs]
    fig.suptitle("Near-vacuum half-domain final density: true saved AMReX plotfiles")
    for ax, result in zip(axs, plot_results):
        case = result["case"]
        image, _boxes, time_value, _ndim, _names, xlim, ylim = composite_slice(
            latest_plotfile(result["run_dir"]),
            "rho",
            5.0 / 3.0,
            2,
            0.5,
            True,
            None,
            1000,
            (0.0, 1.0),
            (0.0, 1.0),
        )
        im = ax.imshow(
            image,
            origin="lower",
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title(f"{case.label}\nlog10 rho, t={time_value:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="log10 rho")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_results(results: list[dict[str, object]], output: Path, scalar_plot: Path, density_plot: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "status",
                "recon",
                "rho_atmosphere",
                "pressure_atmosphere",
                "amr",
                "steps",
                "active_finest_level",
                "mass_rel_drift",
                "min_rho",
                "min_pressure",
                "mcells_s",
                "sim_time_per_s",
                "scalar_plot",
                "density_plot",
                "run_dir",
                "failures",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for result in results:
            case = result["case"]
            rows = result["rows"]
            final = rows[-1] if rows else {}
            writer.writerow(
                {
                    "case": case.name,
                    "status": result["status"],
                    "recon": case.recon,
                    "rho_atmosphere": f"{case.rho_atmosphere:.17e}",
                    "pressure_atmosphere": f"{case.pressure_atmosphere:.17e}",
                    "amr": int(case.amr),
                    "steps": result["steps"],
                    "active_finest_level": result["active_finest_level"],
                    "mass_rel_drift": f"{final.get('mass_rel_drift', math.nan):.17e}",
                    "min_rho": f"{final.get('min_rho', math.nan):.17e}",
                    "min_pressure": f"{final.get('min_pressure', math.nan):.17e}",
                    "mcells_s": f"{result['mcells_s']:.8e}",
                    "sim_time_per_s": f"{result['sim_time_per_s']:.8e}",
                    "scalar_plot": repo_relative(scalar_plot),
                    "density_plot": repo_relative(density_plot),
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
    scalar_plot = next_plot_path(args.plot_dir, "near_vacuum_half_scalars")
    density_plot = next_plot_path(args.plot_dir, "near_vacuum_half_density")
    make_scalar_plot(results, scalar_plot)
    make_density_plot(results, density_plot)
    write_results(results, args.results, scalar_plot, density_plot)

    failed = False
    for result in results:
        case = result["case"]
        rows = result["rows"]
        final = rows[-1] if rows else {}
        print(
            f"{result['status'].upper()} {case.name}: recon={case.recon}, "
            f"rho_atm={case.rho_atmosphere:.1e}, P_atm={case.pressure_atmosphere:.1e}, "
            f"steps={result['steps']}, mass={final.get('mass_rel_drift', math.nan):.3e}, "
            f"min_rho={final.get('min_rho', math.nan):.3e}, "
            f"min_P={final.get('min_pressure', math.nan):.3e}, "
            f"{result['mcells_s']:.2f} Mcells/s"
        )
        if result["status"] != "pass":
            failed = True
            print(result["failures"])
            print(result["stdout_tail"])
    print(f"scalar_plot {repo_relative(scalar_plot)}")
    print(f"density_plot {repo_relative(density_plot)}")
    print(f"results {repo_relative(args.results)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
