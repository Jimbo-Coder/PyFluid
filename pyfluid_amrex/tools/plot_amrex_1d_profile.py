#!/usr/bin/env python3
"""Plot a 1D PyFluid AMReX plotfile profile."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyfluid.exact_solutions import (
    circular_alfven_wave_exact,
    resistive_mhd_wave_exact,
    sod_exact_solution,
    sod_wave_breakpoints,
)
from exact_reference import cell_average_newtonian_hydro
from plotfile_reader import parse_plot_header_1d
from plotfile_reader import read_level0_state_1d as _read_level0_state_1d


def parse_plot_header(plotfile: Path) -> tuple[list[str], float]:
    return parse_plot_header_1d(plotfile)


def read_level0_state_1d(plotfile: Path, names: list[str]) -> np.ndarray:
    return _read_level0_state_1d(plotfile, names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotfile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="AMReX 1D profile")
    parser.add_argument("--gamma", default=1.4, type=float)
    parser.add_argument("--xlo", default=0.0, type=float)
    parser.add_argument("--xhi", default=1.0, type=float)
    parser.add_argument("--sod", action="store_true")
    parser.add_argument("--circular-alfven", action="store_true")
    parser.add_argument("--resistive-wave", action="store_true")
    parser.add_argument("--show-initial", action="store_true")
    parser.add_argument("--shock-x0", default=0.5, type=float)
    parser.add_argument("--rho-left", default=1.0, type=float)
    parser.add_argument("--pressure-left", default=1.0, type=float)
    parser.add_argument("--v1-left", dest="v1_left", default=0.0, type=float)
    parser.add_argument("--rho-right", default=0.125, type=float)
    parser.add_argument("--pressure-right", default=0.1, type=float)
    parser.add_argument("--v1-right", dest="v1_right", default=0.0, type=float)
    parser.add_argument("--rho0", default=1.0, type=float)
    parser.add_argument("--pressure0", default=1.0, type=float)
    parser.add_argument("--b1_0", default=1.0, type=float)
    parser.add_argument("--resistivity", default=0.0, type=float)
    parser.add_argument("--alfven-amplitude", default=1.0e-3, type=float)
    parser.add_argument("--wave-number", default=1.0, type=float)
    parser.add_argument("--alfven-direction", default=1.0, type=float)
    args = parser.parse_args()

    names, time_value = parse_plot_header(args.plotfile)
    state = read_level0_state_1d(args.plotfile, names)
    name_to_index = {name: n for n, name in enumerate(names)}

    rho = state[name_to_index["rho"]]
    momx = state[name_to_index["mom1"]]
    energy = state[name_to_index["energy"]]
    vx = momx / rho
    if "b1" in name_to_index:
        bx = state[name_to_index["b1"]]
        by = state[name_to_index["b2"]]
        bz = state[name_to_index["b3"]]
        momy = state[name_to_index["mom2"]]
        momz = state[name_to_index["mom3"]]
        vy = momy / rho
        vz = momz / rho
        pressure = (args.gamma - 1.0) * (
            energy
            - 0.5 * rho * (vx * vx + vy * vy + vz * vz)
            - 0.5 * (bx * bx + by * by + bz * bz)
        )
    else:
        by = bz = vy = vz = None
        pressure = (args.gamma - 1.0) * (energy - 0.5 * rho * vx * vx)

    ncell = rho.size
    dx = (args.xhi - args.xlo) / ncell
    x = args.xlo + (np.arange(ncell) + 0.5) * dx

    exact = None
    if args.sod:
        left = (args.rho_left, args.v1_left, args.pressure_left)
        right = (args.rho_right, args.v1_right, args.pressure_right)
        exact = cell_average_newtonian_hydro(
            lambda points: sod_exact_solution(
                points,
                time_value,
                args.gamma,
                left,
                right,
                x0=args.shock_x0,
            ),
            np.linspace(args.xlo, args.xhi, ncell + 1),
            args.gamma,
            breakpoints=sod_wave_breakpoints(
                time_value,
                args.gamma,
                left,
                right,
                x0=args.shock_x0,
                domain_lo=args.xlo,
                domain_hi=args.xhi,
            ),
        )
    if args.circular_alfven:
        exact = circular_alfven_wave_exact(
            x,
            time_value,
            args.rho0,
            args.pressure0,
            args.b1_0,
            args.alfven_amplitude,
            args.wave_number,
            args.alfven_direction,
        )
        initial = (
            circular_alfven_wave_exact(
                x,
                0.0,
                args.rho0,
                args.pressure0,
                args.b1_0,
                args.alfven_amplitude,
                args.wave_number,
                args.alfven_direction,
            )
            if args.show_initial
            else None
        )
    elif args.resistive_wave:
        exact = resistive_mhd_wave_exact(
            x,
            time_value,
            args.gamma,
            args.pressure0,
            args.alfven_amplitude,
            args.resistivity,
            args.wave_number,
        )
        initial = (
            resistive_mhd_wave_exact(
                x,
                0.0,
                args.gamma,
                args.pressure0,
                args.alfven_amplitude,
                args.resistivity,
                args.wave_number,
            )
            if args.show_initial
            else None
        )
    else:
        initial = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    title_lines = textwrap.wrap(args.title, width=82) or [args.title]
    title_lines.append(f"t={time_value:.5f}, nx={ncell}")
    fig.suptitle("\n".join(title_lines), fontsize=12)
    if args.circular_alfven:
        if by is None or bz is None:
            raise ValueError("--circular-alfven requires an MHD plotfile with b2/b3")
        pressure_error = pressure - exact[1]
        series = [(by, "b2"), (bz, "b3"), (pressure_error, "P error")]
    elif args.resistive_wave:
        if by is None or bz is None:
            raise ValueError("--resistive-wave requires an MHD plotfile with b2/b3")
        series = [(by, "b2"), (bz, "b3"), (pressure - args.pressure0, "P - P0")]
    else:
        series = [(rho, "rho"), (pressure, "P"), (vx, "v1")]
    for ax, (values, label) in zip(axs, series, strict=True):
        ax.plot(x, values, color="tab:blue", marker="o", markersize=2.5, linewidth=1.2, label="num")
        if exact is not None:
            if args.circular_alfven:
                exact_values = {"P error": np.zeros_like(x), "b2": exact[6], "b3": exact[7]}[label]
            elif args.resistive_wave:
                exact_values = {"b2": exact[0], "b3": exact[1], "P - P0": exact[2] - args.pressure0}[label]
            else:
                exact_values = {"rho": exact[0], "P": exact[1], "v1": exact[2]}[label]
            ax.plot(x, exact_values, color="black", linestyle=":", linewidth=2.0, label="exact")
        if initial is not None:
            if args.resistive_wave:
                initial_values = {"b2": initial[0], "b3": initial[1], "P - P0": initial[2] - args.pressure0}[label]
            else:
                initial_values = {"P error": np.zeros_like(x), "b2": initial[6], "b3": initial[7]}[label]
            ax.plot(x, initial_values, color="0.45", linestyle="--", linewidth=1.6, label="initial")
        ax.set_ylabel(label)
        ax.grid(True, which="major", alpha=0.25)
        ax.legend()
    axs[-1].set_xlabel("x")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
