#!/usr/bin/env python3
"""Validate discrete PdV work for a pulsating circular solid boundary."""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation_common import next_numbered_path, read_float_tsv, repo_relative, run_amrex, write_tsv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_2d_amr.in"
DEFAULT_OUTPUT = ROOT / "runs" / "current" / "pulsating_wall_work_validation"
DEFAULT_PLOT_DIR = ROOT / "plots" / "current" / "IC" / "pulsating_circle"


@dataclass(frozen=True)
class Case:
    name: str
    work_tol: float
    overrides: tuple[str, ...]


CASES = (
    Case(
        name="single_level",
        work_tol=2.0e-4,
        overrides=(
            "amr.n_cell=128 128",
            "amr.max_level=0",
        ),
    ),
    Case(
        name="amr_level2",
        work_tol=2.0e-3,
        overrides=(
            "amr.n_cell=64 64",
            "amr.max_level=2",
            "amr.ref_ratio=2 2",
            "amr.regrid_int=4",
            "amr.grid_eff=0.8",
            "amr.n_error_buf=3 3",
            "amr.tag_metric=none",
            "amr.solid_refine_band_cells=2.0",
            "amr.solid_curvature_error=0.0005",
            "amr.level_cell_budget_side=384",
            "amr.total_cell_budget=250000",
            "amr.state_interp=conservative",
        ),
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
    "pyfluid.solid_radius=0.18",
    "pyfluid.solid_radius_amplitude=0.04",
    "pyfluid.solid_radius_frequency=1.0",
    "pyfluid.solid_volume_fraction_samples_per_dim=4",
    "pyfluid.solid_swept_reference_time=0.0",
    "pyfluid.max_step=100000",
    "pyfluid.final_time=1.0",
    "pyfluid.scalar_int=8",
    "pyfluid.plot_int=-1",
    "pyfluid.checkpoint_int=-1",
    "pyfluid.aux_gfs=none",
    "pyfluid.aux_gf_compute_int=-1",
    "pyfluid.aux_gf_int=-1",
)


def run_case(case: Case, args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.output_dir / case.name
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"pyfluid.output_dir={repo_relative(run_dir)}",
        *COMMON_OVERRIDES,
        *case.overrides,
    ]
    completed = run_amrex(dim=2, input_path=args.input, ranks=args.ranks, executable=args.executable, overrides=overrides)
    scalar_path = run_dir / "scalars.tsv"
    rows = read_float_tsv(scalar_path, stop_on_incomplete=True) if scalar_path.exists() else []
    final = rows[-1] if rows else {}
    failures: list[str] = []
    max_work_error = math.nan
    if completed.returncode != 0:
        failures.append(f"returncode={completed.returncode}")
    if not rows:
        failures.append("missing scalar output")
    else:
        energy_change = final["energy"] - rows[0]["energy"]
        max_work_error = max(row["wall_work_rel_error"] for row in rows)
        if final["mass_rel_drift"] > args.mass_tol:
            failures.append(f"mass_rel_drift={final['mass_rel_drift']:.3e} > {args.mass_tol:.3e}")
        if final["wall_work"] <= 0.0:
            failures.append(f"wall_work={final['wall_work']:.3e} did not increase")
        if energy_change <= 0.0:
            failures.append(f"energy_change={energy_change:.3e} did not increase")
        if max_work_error > case.work_tol:
            failures.append(f"max_wall_work_rel_error={max_work_error:.3e} > {case.work_tol:.3e}")
        if final["min_rho"] <= 0.0:
            failures.append(f"min_rho={final['min_rho']:.3e}")
        if final["min_pressure"] <= 0.0:
            failures.append(f"min_pressure={final['min_pressure']:.3e}")
    return {
        "case": case,
        "run_dir": run_dir,
        "scalars": scalar_path,
        "rows": rows,
        "stdout": completed.stdout,
        "max_wall_work_rel_error": max_work_error,
        "status": "pass" if not failures else "fail",
        "failures": "; ".join(failures),
    }


def positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(values), 1.0e-18)


def make_plot(results: list[dict[str, object]], output: Path) -> None:
    fig, axs = plt.subplots(len(results), 3, figsize=(15, 4.1 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axs = np.asarray([axs])
    fig.suptitle("Pulsating circular wall PdV work: hydro 2D, PWL/HLLC/SSPRK3, CFL=0.9, reflecting BC")
    for row, result in enumerate(results):
        case = result["case"]
        rows = result["rows"]
        t = np.asarray([entry["time"] for entry in rows])
        energy0 = rows[0]["energy"]
        energy_change = np.asarray([entry["energy"] - energy0 for entry in rows])
        wall_work = np.asarray([entry["wall_work"] for entry in rows])
        residual = np.asarray([entry["energy_minus_wall_work"] for entry in rows])
        rel_error = np.asarray([entry["wall_work_rel_error"] for entry in rows])
        mass = np.asarray([entry["mass_rel_drift"] for entry in rows])
        markers = {"marker": "o"} if len(rows) <= 20 else {}

        axs[row, 0].plot(t, energy_change, label="energy change", linewidth=2.3, zorder=2, **markers)
        axs[row, 0].plot(t, wall_work, label="wall work", linewidth=2.4, linestyle=":", zorder=5, **markers)
        axs[row, 0].set_title(f"{case.name}: Energy Budget")
        axs[row, 0].set_xlabel("time")
        axs[row, 0].legend()

        axs[row, 1].semilogy(t, positive(residual), label="|E - E0 - work|", linewidth=2.0, **markers)
        axs[row, 1].semilogy(t, positive(rel_error), label="relative error", linewidth=2.0, **markers)
        axs[row, 1].set_title("PdV Work Residual")
        axs[row, 1].set_xlabel("time")
        axs[row, 1].legend()

        axs[row, 2].semilogy(t, np.maximum(mass, 1.0e-18), label="mass", linewidth=2.0, **markers)
        axs[row, 2].plot(t, [entry["min_rho"] for entry in rows], label="min rho", linewidth=2.0, **markers)
        axs[row, 2].plot(t, [entry["min_pressure"] for entry in rows], label="min P", linewidth=2.0, **markers)
        axs[row, 2].set_title("Mass / Floors")
        axs[row, 2].set_xlabel("time")
        axs[row, 2].legend()

    for ax in axs.flat:
        ax.grid(True, which="major", alpha=0.25)
        try:
            ax.ticklabel_format(useOffset=False)
        except AttributeError:
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_results(results: list[dict[str, object]], plot_path: Path, output: Path) -> None:
    rows = []
    for result in results:
        case = result["case"]
        scalar_rows = result["rows"]
        final = scalar_rows[-1] if scalar_rows else {}
        rows.append(
            {
                "case": case.name,
                "status": result["status"],
                "mass_rel_drift": f"{final.get('mass_rel_drift', math.nan):.17e}",
                "energy_change": f"{(final.get('energy', math.nan) - scalar_rows[0]['energy']) if scalar_rows else math.nan:.17e}",
                "wall_work": f"{final.get('wall_work', math.nan):.17e}",
                "energy_minus_wall_work": f"{final.get('energy_minus_wall_work', math.nan):.17e}",
                "wall_work_rel_error": f"{final.get('wall_work_rel_error', math.nan):.17e}",
                "max_wall_work_rel_error": f"{result['max_wall_work_rel_error']:.17e}",
                "min_rho": f"{final.get('min_rho', math.nan):.17e}",
                "min_pressure": f"{final.get('min_pressure', math.nan):.17e}",
                "plot": repo_relative(plot_path),
                "scalars": repo_relative(result["scalars"]),
                "failures": result["failures"],
            }
        )
    write_tsv(
        output,
        rows,
        [
                "case",
                "status",
                "mass_rel_drift",
                "energy_change",
                "wall_work",
                "energy_minus_wall_work",
                "wall_work_rel_error",
                "max_wall_work_rel_error",
                "min_rho",
                "min_pressure",
                "plot",
                "scalars",
                "failures",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mass-tol", type=float, default=5.0e-13)
    args = parser.parse_args()

    results = [run_case(case, args) for case in CASES]
    plot_path = next_numbered_path(args.plot_dir, "pulsating_wall_work")
    if all(result["rows"] for result in results):
        make_plot(results, plot_path)
    results_path = args.output_dir / "results.tsv"
    write_results(results, plot_path, results_path)

    for result in results:
        case = result["case"]
        rows = result["rows"]
        if rows:
            final = rows[-1]
            print(
                f"{result['status'].upper()} {case.name}: "
                f"mass={final['mass_rel_drift']:.2e}, "
                f"dE={final['energy'] - rows[0]['energy']:.6e}, "
                f"work={final['wall_work']:.6e}, "
                f"rel_error={final['wall_work_rel_error']:.2e}, "
                f"max_rel_error={result['max_wall_work_rel_error']:.2e}, "
                f"min_rho={final['min_rho']:.3e}, min_P={final['min_pressure']:.3e}"
            )
        else:
            print(f"{result['status'].upper()} {case.name}: {result['failures']}")
        if result["status"] != "pass":
            print(result["failures"])
    print(f"plot {repo_relative(plot_path)}")
    print(f"wrote {repo_relative(results_path)}")
    return 0 if all(result["status"] == "pass" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
