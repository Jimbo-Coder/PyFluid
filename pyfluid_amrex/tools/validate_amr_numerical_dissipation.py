#!/usr/bin/env python3
"""Measure numerical damping versus AMR minimum feature length."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from validation_common import materialize_run_input, run_mpi

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amrex_boxed_movie import composite_slice, level_count, plotfile_step  # noqa: E402
from plotfile_reader import latest_plotfile  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "build/pyfluid_amrex/pyfluid_amrex_2d"
INPUT = ROOT / "pyfluid_amrex/inputs/hydro_sine_2d.in"


@dataclass(frozen=True)
class SweepPoint:
    min_feature_length: float
    output_dir: Path


def parse_tsv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
           return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f, delimiter="\t")]


def fourier_amplitude_density(plotfile: Path, max_pixels: int) -> tuple[float, float]:
    image, _boxes, time_value, _ndim, _names, _xlim, _ylim = composite_slice(
           plotfile,
           "rho",
           5.0 / 3.0,
           2,
           0.5,
           False,
           None,
           max_pixels,
           (0.0, 1.0),
           (0.0, 1.0),
    )
    if not np.all(np.isfinite(image)):
           finite = np.isfinite(image)
           if not np.any(finite):
               raise ValueError(f"composite density has no finite values for {plotfile}")
           image = np.where(finite, image, np.nanmean(image))
    ny, nx = image.shape
    x = (np.arange(nx, dtype=float) + 0.5) / nx
    y = (np.arange(ny, dtype=float) + 0.5) / ny
    xx, yy = np.meshgrid(x, y)
    perturbation = image - float(np.mean(image))
    coefficient = np.mean(perturbation * np.exp(-2.0j * np.pi * (xx + yy)))
    return 2.0 * abs(coefficient), time_value


def active_finest_summary(output_dir: Path) -> dict[str, float]:
    rows = parse_tsv(output_dir / "amr_levels.tsv")
    return rows[-1]


def run_case(args: argparse.Namespace, point: SweepPoint) -> dict[str, float | str]:
    overrides = {
           "pyfluid.output_dir": str(point.output_dir),
           "pyfluid.max_step": "100000",
           "pyfluid.final_time": f"{args.final_time:.17g}",
           "pyfluid.ng": "2",
           "pyfluid.cfl": "0.9",
           "pyfluid.gamma": "1.6666666666666667",
           "pyfluid.ic": "sin_wave",
           "pyfluid.rho0": "1.0",
           "pyfluid.pressure0": "1.0",
           "pyfluid.amplitude": f"{args.amplitude:.17g}",
           "pyfluid.v1_0": "0.2",
           "pyfluid.v2_0": "-0.1",
           "pyfluid.v3_0": "0.0",
           "pyfluid.recon": args.recon,
           "pyfluid.riemann": args.riemann,
           "pyfluid.timestep": args.timestep,
           "pyfluid.bc": "periodic",
           "pyfluid.diffusion": "none",
           "pyfluid.subcycling": "on",
           "pyfluid.subcycling_steps": "spatial",
           "pyfluid.plot_int": "100000",
           "pyfluid.scalar_int": "8",
           "pyfluid.checkpoint_int": "-1",
           "pyfluid.aux_gfs": "none",
           "pyfluid.aux_gf_int": "-1",
           "pyfluid.aux_gf_compute_int": "-1",
           "amr.n_cell": f"{args.base_n} {args.base_n}",
           "amr.max_level": str(args.max_level),
           "amr.ref_ratio": " ".join(["2"] * args.max_level),
           "amr.max_grid_size": str(args.max_grid_size),
           "amr.blocking_factor": str(args.blocking_factor),
           "amr.regrid_int": "4",
           "amr.grid_eff": "0.85",
           "amr.n_error_buf": " ".join([str(args.error_buffer)] * args.max_level),
           "amr.tag_metric": "auto",
           "amr.tag_threshold": f"{args.tag_threshold:.17g}",
           "amr.state_interp": "quartic",
           "amr.min_feature_length": f"{point.min_feature_length:.17g}",
           "amr.level_cell_budget_side": str(args.level_cell_budget_side),
           "amr.total_cell_budget": str(args.total_cell_budget),
    }
    input_path = materialize_run_input(INPUT, point.output_dir / "run.in", overrides)
    result = run_mpi(SOLVER, [input_path], ranks=args.ranks)
    if result.returncode != 0:
        raise RuntimeError(
            f"run failed for min_feature_length={point.min_feature_length:g}\n"
            f"output:\n{result.stdout}"
        )

    levels = active_finest_summary(point.output_dir)
    finest_dx = float(levels["finest_dx"])
    plotfile = latest_plotfile(point.output_dir, require=True)
    nlevels = level_count(plotfile)
    max_pixels = args.base_n * (2 ** (nlevels - 1))
    amplitude, time_value = fourier_amplitude_density(plotfile, max_pixels)
    ratio = max(amplitude / args.amplitude, 1.0e-300)
    k2 = (2.0 * math.pi) ** 2 * 2.0
    nu_num = max(0.0, -math.log(ratio) / (k2 * time_value))
    ell_num = math.sqrt(nu_num * time_value) if nu_num > 0.0 else 0.0
    scalars = parse_tsv(point.output_dir / "scalars.tsv")[-1]
    return {
        "min_feature_length": point.min_feature_length,
        "finest_dx": finest_dx,
        "finest_dx_over_min_feature_length": float(levels["finest_dx_over_min_feature_length"]),
        "active_finest_level": float(levels["active_finest_level"]),
        "touches_max_level": float(levels["touches_max_level"]),
        "resolution_limited_by_max_level": float(levels["resolution_limited_by_max_level"]),
        "time": time_value,
        "amplitude": amplitude,
        "amplitude_ratio": ratio,
        "nu_num": nu_num,
        "ell_num": ell_num,
        "mcells_per_s": float(scalars["mcells_per_s"]),
        "sim_time_per_s": float(scalars["sim_time_per_s"]),
        "plotfile": str(plotfile.relative_to(ROOT)),
        "stdout_last": next(
            (line for line in result.stdout.strip().splitlines() if line.startswith("PyFluid AMReX")),
            result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "",
        ),
    }


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = [
        "min_feature_length",
        "finest_dx",
        "finest_dx_over_min_feature_length",
        "active_finest_level",
        "touches_max_level",
        "resolution_limited_by_max_level",
        "time",
        "amplitude",
        "amplitude_ratio",
        "nu_num",
        "ell_num",
        "mcells_per_s",
        "sim_time_per_s",
        "plotfile",
        "stdout_last",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def positive_fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return math.nan
    slope, _intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope)


def plot_summary(path: Path, rows: list[dict[str, float | str]], args: argparse.Namespace) -> None:
    finest_dx = np.asarray([float(row["finest_dx"]) for row in rows])
    min_length = np.asarray([float(row["min_feature_length"]) for row in rows])
    nu_num = np.asarray([float(row["nu_num"]) for row in rows])
    ell_num = np.asarray([float(row["ell_num"]) for row in rows])
    amp_ratio = np.asarray([float(row["amplitude_ratio"]) for row in rows])
    active_level = np.asarray([float(row["active_finest_level"]) for row in rows])
    limited = np.asarray([float(row["resolution_limited_by_max_level"]) for row in rows])
    slope = positive_fit_slope(finest_dx, nu_num)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    title = (
        "AMR numerical dissipation: hydro sine entropy wave, "
        f"{args.recon.upper()}/{args.riemann.upper()}/{args.timestep.upper()}, CFL=0.9"
    )
    fig.suptitle(title)

    axs[0, 0].loglog(min_length, finest_dx, marker="o", linewidth=2.2, label="measured finest dx")
    axs[0, 0].loglog(min_length, min_length, linestyle="--", color="0.25", label="dx = min length")
    axs[0, 0].set_title("AMR Cutoff Control")
    axs[0, 0].set_xlabel("min feature length")
    axs[0, 0].set_ylabel("finest dx")
    axs[0, 0].invert_xaxis()
    axs[0, 0].invert_yaxis()
    axs[0, 0].legend()

    label = "inferred nu_num" if math.isnan(slope) else f"inferred nu_num, slope={slope:.2f}"
    axs[0, 1].loglog(finest_dx, np.maximum(nu_num, 1.0e-300), marker="s", linewidth=2.2, label=label)
    if np.any(nu_num > 0.0):
        anchor = int(np.argmax(finest_dx))
        reference_power = 2 if math.isnan(slope) else max(1, min(6, int(round(slope))))
        reference = nu_num[anchor] * (finest_dx / finest_dx[anchor]) ** reference_power
        axs[0, 1].loglog(
            finest_dx,
            np.maximum(reference, 1.0e-300),
            linestyle="--",
            color="0.25",
            label=f"dx^{reference_power} reference",
        )
    axs[0, 1].set_title("Effective Numerical Viscosity")
    axs[0, 1].set_xlabel("finest dx")
    axs[0, 1].set_ylabel("nu_num")
    axs[0, 1].invert_xaxis()
    axs[0, 1].legend()

    axs[1, 0].loglog(finest_dx, np.maximum(ell_num, 1.0e-300), marker="^", linewidth=2.2, label="sqrt(nu_num t)")
    axs[1, 0].loglog(finest_dx, finest_dx, linestyle="--", color="0.25", label="finest dx")
    axs[1, 0].set_title("Damping Length At Final Time")
    axs[1, 0].set_xlabel("finest dx")
    axs[1, 0].set_ylabel("length")
    axs[1, 0].invert_xaxis()
    axs[1, 0].legend()

    axs[1, 1].semilogx(finest_dx, amp_ratio, marker="o", linewidth=2.2, label="A/A0")
    axs[1, 1].semilogx(finest_dx, active_level, marker="s", linewidth=2.0, label="active level")
    axs[1, 1].semilogx(finest_dx, limited, marker="x", linewidth=1.8, label="max-level limited")
    axs[1, 1].set_title("Run Health")
    axs[1, 1].set_xlabel("finest dx")
    axs[1, 1].invert_xaxis()
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.grid(True, which="major", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--base-n", type=int, default=32)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--final-time", type=float, default=0.5)
    parser.add_argument("--amplitude", type=float, default=5.0e-2)
    parser.add_argument("--tag-threshold", type=float, default=1.0e-7)
    parser.add_argument("--error-buffer", type=int, default=4)
    parser.add_argument("--max-grid-size", type=int, default=64)
    parser.add_argument("--blocking-factor", type=int, default=8)
    parser.add_argument("--level-cell-budget-side", type=int, default=1024)
    parser.add_argument("--total-cell-budget", type=int, default=900000)
    parser.add_argument("--recon", default="pwl")
    parser.add_argument("--riemann", default="hllc")
    parser.add_argument("--timestep", default="ssprk3")
    parser.add_argument(
        "--min-feature-lengths",
        nargs="+",
        type=float,
        default=[1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0, 1.0 / 512.0],
    )
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/current/amr_numerical_dissipation")
    parser.add_argument("--summary", type=Path, default=ROOT / "runs/current/amr_numerical_dissipation_sweep.tsv")
    parser.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "plots/current/scans/amr_numerical_dissipation_015.png",
    )
    parser.add_argument(
        "--allow-max-limited",
        action="store_true",
        help="write outputs but do not fail if a sweep point is capped by max_level before meeting its tolerance",
    )
    args = parser.parse_args()

    rows: list[dict[str, float | str]] = []
    for min_length in args.min_feature_lengths:
        tag = f"minlen_{min_length:.8f}".replace(".", "p")
        point = SweepPoint(min_length, args.run_root / tag)
        row = run_case(args, point)
        rows.append(row)
        print(
            f"min_length={min_length:.8g} finest_dx={row['finest_dx']:.8g} "
            f"nu_num={row['nu_num']:.6e} A/A0={row['amplitude_ratio']:.8f}"
        )
    write_summary(args.summary, rows)
    plot_summary(args.plot, rows, args)
    print(f"wrote {args.summary.relative_to(ROOT)}")
    print(f"wrote {args.plot.relative_to(ROOT)}")
    max_limited = [row for row in rows if float(row["resolution_limited_by_max_level"]) > 0.5]
    if max_limited and not args.allow_max_limited:
        limited_values = ", ".join(f"{float(row['min_feature_length']):.8g}" for row in max_limited)
        raise SystemExit(
            "AMR sweep is max-level limited before satisfying its tolerance for "
            f"min_feature_length={limited_values}; increase max_level/budget or loosen the tolerance"
        )


if __name__ == "__main__":
    main()
