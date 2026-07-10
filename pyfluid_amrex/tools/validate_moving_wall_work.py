#!/usr/bin/env python3
"""Validate discrete PdV work for the conservative moving-wall piston path."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation_common import next_numbered_path, read_float_tsv, repo_relative, run_amrex, write_tsv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "hydro_moving_wall_smush_2d.in"
DEFAULT_OUTPUT = ROOT / "runs" / "current" / "moving_wall_work_validation"
DEFAULT_PLOT_DIR = ROOT / "plots" / "current" / "IC" / "hydro_moving_wall_smush_2d"


def run_case(args: argparse.Namespace):
    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"pyfluid.output_dir={repo_relative(args.output_dir)}",
        "geometry.prob_lo=0.0 0.0",
        "geometry.prob_hi=1.0 1.0",
        "geometry.is_periodic=0 0",
        "amr.n_cell=256 128",
        "amr.max_level=0",
        "amr.max_grid_size=64",
        "amr.blocking_factor=16",
        "pyfluid.physics=hydro",
        "pyfluid.ct=off",
        "pyfluid.ic=uniform",
        "pyfluid.rho0=1.0",
        "pyfluid.pressure0=1.0",
        "pyfluid.v1_0=0.0",
        "pyfluid.v2_0=0.0",
        "pyfluid.v3_0=0.0",
        "pyfluid.gamma=1.4",
        "pyfluid.cfl=0.9",
        "pyfluid.recon=pwl",
        "pyfluid.riemann=hllc",
        "pyfluid.timestep=ssprk3",
        "pyfluid.bc=reflecting",
        "pyfluid.bc_xlo=reflecting",
        "pyfluid.bc_xhi=reflecting",
        "pyfluid.bc_ylo=reflecting",
        "pyfluid.bc_yhi=reflecting",
        "pyfluid.diffusion=none",
        "pyfluid.subcycling=off",
        "pyfluid.solid_geometry=boxes",
        "pyfluid.solid_motion=translate",
        "pyfluid.solid_wall_condition=slip",
        "pyfluid.solid_box_lo=-1.0 0.0",
        "pyfluid.solid_box_hi=0.0 1.0",
        f"pyfluid.solid_velocity_1={args.wall_speed}",
        f"pyfluid.final_time={args.displacement / args.wall_speed}",
        "pyfluid.max_step=100000",
        "pyfluid.scalar_int=8",
        "pyfluid.plot_int=-1",
        "pyfluid.checkpoint_int=-1",
        "pyfluid.aux_gfs=none",
        "pyfluid.aux_gf_compute_int=-1",
        "pyfluid.aux_gf_int=-1",
    ]
    return run_amrex(dim=2, input_path=args.input, ranks=args.ranks, executable=args.executable, overrides=overrides)


def make_plot(rows: list[dict[str, float]], output: Path, title: str) -> None:
    t = np.asarray([row["time"] for row in rows])
    energy0 = rows[0]["energy"]
    energy_change = np.asarray([row["energy"] - energy0 for row in rows])
    wall_work = np.asarray([row["wall_work"] for row in rows])
    residual = np.asarray([abs(row["energy_minus_wall_work"]) for row in rows])
    rel_error = np.asarray([row["wall_work_rel_error"] for row in rows])
    mass = np.asarray([max(row["mass_rel_drift"], 1.0e-18) for row in rows])
    markers = {"marker": "o"} if len(rows) <= 20 else {}

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle(title)
    axs[0].plot(t, energy_change, label="energy change", linewidth=2.0, **markers)
    axs[0].plot(t, wall_work, label="wall work", linewidth=2.0, linestyle="--", **markers)
    axs[0].set_title("Energy Budget")
    axs[0].set_xlabel("time")
    axs[0].legend()

    axs[1].semilogy(t, np.maximum(residual, 1.0e-18), label="|E - E0 - work|", linewidth=2.0, **markers)
    axs[1].semilogy(t, np.maximum(rel_error, 1.0e-18), label="relative error", linewidth=2.0, **markers)
    axs[1].set_title("PdV Work Residual")
    axs[1].set_xlabel("time")
    axs[1].legend()

    axs[2].semilogy(t, mass, label="mass", linewidth=2.0, **markers)
    axs[2].set_title("Mass Conservation")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("relative change")
    axs[2].legend()
    for ax in axs:
        ax.grid(True, which="major", alpha=0.25)
        try:
            ax.ticklabel_format(useOffset=False)
        except AttributeError:
            pass
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wall-speed", type=float, default=2.0)
    parser.add_argument("--displacement", type=float, default=0.12)
    parser.add_argument("--mass-tol", type=float, default=5.0e-13)
    parser.add_argument("--work-tol", type=float, default=5.0e-12)
    args = parser.parse_args()

    completed = run_case(args)
    if completed.returncode != 0:
        print(completed.stdout)
        return completed.returncode

    scalar_path = args.output_dir / "scalars.tsv"
    rows = read_float_tsv(scalar_path)
    final = rows[-1]
    failures: list[str] = []
    if final["mass_rel_drift"] > args.mass_tol:
        failures.append(f"mass_rel_drift={final['mass_rel_drift']:.3e} > {args.mass_tol:.3e}")
    if final["wall_work"] <= 0.0:
        failures.append(f"wall_work={final['wall_work']:.3e} did not increase")
    if final["wall_work_rel_error"] > args.work_tol:
        failures.append(f"wall_work_rel_error={final['wall_work_rel_error']:.3e} > {args.work_tol:.3e}")

    plot_path = next_numbered_path(args.plot_dir, "moving_wall_work")
    make_plot(
        rows,
        plot_path,
        (
            "Moving-wall PdV work: IC=uniform piston, PWL/HLLC/SSPRK3, "
            f"reflecting BC, vx={args.wall_speed:g}, displacement={args.displacement:g}"
        ),
    )

    result_path = args.output_dir / "results.tsv"
    write_tsv(result_path, [{
        "status": "pass" if not failures else "fail",
        "mass_rel_drift": f"{final['mass_rel_drift']:.17e}",
        "energy_change": f"{final['energy'] - rows[0]['energy']:.17e}",
        "wall_work": f"{final['wall_work']:.17e}",
        "energy_minus_wall_work": f"{final['energy_minus_wall_work']:.17e}",
        "wall_work_rel_error": f"{final['wall_work_rel_error']:.17e}",
        "plot": repo_relative(plot_path),
        "scalars": repo_relative(scalar_path),
        "failures": "; ".join(failures),
    }], [
        "status",
        "mass_rel_drift",
        "energy_change",
        "wall_work",
        "energy_minus_wall_work",
        "wall_work_rel_error",
        "plot",
        "scalars",
        "failures",
    ])

    print(
        f"{'PASS' if not failures else 'FAIL'} moving_wall_work: "
        f"mass={final['mass_rel_drift']:.2e}, "
        f"dE={final['energy'] - rows[0]['energy']:.6e}, "
        f"work={final['wall_work']:.6e}, "
        f"rel_error={final['wall_work_rel_error']:.2e}"
    )
    print(f"plot {repo_relative(plot_path)}")
    print(f"wrote {repo_relative(result_path)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
