#!/usr/bin/env python3
"""Planar SRHD shock heating: exact jumps, inflow budgets and mesh convergence.

Two opposing streams replace the reflecting wall by symmetry at x=0.5.
This tests the same shock-heating jump, but does not validate wall boundaries.
The full-suite gate uses W=10,100 and N=128,256,512 at CFL=0.9. No plots are
produced. Results (including failures) are written before returning failure.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil

import numpy as np

from plotfile_reader import latest_plotfile, parse_plot_header_1d, read_composite_state_1d
from relativistic_validation import shock_heating_cell_averages, shock_heating_solution
from validation_common import (
    OPEN_POLICY, ROOT, append_validation_history, materialize_run_input,
    read_scalar_rows, repo_relative, run_amrex, scalar_validation_row, write_tsv,
)

RUN_ROOT = ROOT / "runs/current/rhd_shock_heating"
SUMMARY = RUN_ROOT / "summary.tsv"
COMPONENTS = ("dens", "mom1", "tau")
# Discontinuous solutions converge at first order in L1. Require >=0.6 on
# the full resolution sequence, <=3% normalized error on the finest mesh,
# and analytic inflow budgets to 1e-9 at every resolution. Adjacent-mesh
# orders are reported, but shock/cell alignment can make them oscillate.
ERROR_TOLERANCE = 0.03
MINIMUM_ORDER = 0.6
BUDGET_TOLERANCE = 1.0e-9


def check_oracle() -> None:
    """Check independent cold-limit formulas and finite-pressure jump residuals."""
    for lorentz in (2.0, 10.0, 100.0):
        cold = shock_heating_solution(lorentz, pressure=0.0)
        expected_density = 4.0 * lorentz + 3.0
        expected_speed = lorentz * cold["velocity"] / (3.0 * (lorentz + 1.0))
        np.testing.assert_allclose(cold["downstream_dens"], expected_density, rtol=2e-14)
        np.testing.assert_allclose(cold["shock_speed"], expected_speed, rtol=2e-14)
        solution = shock_heating_solution(lorentz)
        speed, velocity = solution["shock_speed"], solution["velocity"]
        np.testing.assert_allclose(
            speed * (solution["downstream_tau"] + solution["downstream_dens"]
                     - solution["upstream_tau"] - lorentz),
            solution["upstream_mom1"], rtol=2e-14,
        )
        averages = shock_heating_cell_averages(127, 0.2, solution)
        np.testing.assert_allclose(
            np.mean(averages["dens"]), lorentz * (1.0 + 0.4 * velocity), rtol=2e-14,
        )


def run_case(lorentz: float, cells: int, args: argparse.Namespace) -> dict:
    name = f"w{lorentz:g}_n{cells}"
    run_dir = RUN_ROOT / name
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    solution = shock_heating_solution(lorentz)
    row = {"case": name, "lorentz": lorentz, "cells": cells,
           "status": "FAIL", "reason": "", "run_dir": repo_relative(run_dir)}
    parameters = {
        "pyfluid.run_name": name, "pyfluid.output_dir": repo_relative(run_dir),
        "amr.n_cell": str(cells), "amr.max_grid_size": "128",
        "pyfluid.rho_left": "1", "pyfluid.rho_right": "1",
        "pyfluid.pressure_left": "1e-6", "pyfluid.pressure_right": "1e-6",
        "pyfluid.v1_left": repr(solution["velocity"]),
        "pyfluid.v1_right": repr(-solution["velocity"]),
        "pyfluid.gamma": repr(4.0 / 3.0),
        "pyfluid.final_time": "0.2", "pyfluid.scalar_int": "16",
        "pyfluid.aux_gfs": "none", "pyfluid.plot_int": "1000000",
    }
    try:
        input_path = materialize_run_input(
            ROOT / "pyfluid_amrex/inputs/rhd_sod_x_1d.in",
            run_dir / "input.in", parameters,
        )
        if args.force or latest_plotfile(run_dir) is None:
            completed = run_amrex(dim=1, input_path=input_path,
                                  ranks=args.ranks, timeout=args.timeout)
            (run_dir / "run.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError("solver failed; see run.log")
        scalar = scalar_validation_row(
            run_dir, case=name, checks=OPEN_POLICY, mhd=False,
            dim=1, ranks=args.ranks, kind="analytic",
        )
        append_validation_history(ROOT / "runs/current/validation_history.tsv", [scalar])
        if scalar["status"] != "PASS":
            raise RuntimeError(f"scalar sanity: {scalar['reason']}")
        recovery = re.findall(
            r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)",
            (run_dir / "run.log").read_text(),
        )
        if not recovery or any(int(value) for match in recovery for value in match):
            raise RuntimeError("missing recovery summary or nonzero fallback/reset count")
        plotfile = latest_plotfile(run_dir, require=True)
        names, final_time = parse_plot_header_1d(plotfile)
        if not math.isclose(final_time, 0.2, rel_tol=0.0, abs_tol=1e-13):
            raise RuntimeError(f"stopped at t={final_time}, expected 0.2")
        state = read_composite_state_1d(plotfile, names)
        if state.shape[1] != cells:
            raise RuntimeError("unexpected output mesh")
        exact = shock_heating_cell_averages(cells, final_time, solution)
        for component in COMPONENTS:
            scale = max(abs(solution[f"upstream_{component}"]),
                        abs(solution.get(f"downstream_{component}", 0.0)))
            row[f"{component}_l1"] = float(np.mean(
                np.abs(state[names.index(component)] - exact[component])) / scale)
        # Independent analytic boundary influx; core totals come from C++
        # scalars, not sums of plotfile fields or solver-authored budgets.
        mass_errors, energy_errors, momentum_errors = [], [], []
        scalar_rows = read_scalar_rows(run_dir)
        for scalar_row in scalar_rows:
            time = float(scalar_row["time"])
            expected_mass = lorentz * (1.0 + 2.0 * solution["velocity"] * time)
            expected_energy = solution["upstream_tau"] + 2.0 * time * (
                solution["upstream_mom1"] - lorentz * solution["velocity"])
            mass_errors.append(abs(float(scalar_row["mass"]) / expected_mass - 1.0))
            energy_errors.append(abs(float(scalar_row["energy"]) / expected_energy - 1.0))
            momentum_errors.append(abs(float(scalar_row["mom1"])) / solution["upstream_mom1"])
        row.update(
            source_energy_change=float(scalar_rows[-1]["source_energy_change"]),
            external_energy_rel_error=max(abs(float(item["external_energy_rel_error"]))
                                          for item in scalar_rows),
        )
        row["budget_accounting_status"] = (
            "PASS" if row["external_energy_rel_error"] <= BUDGET_TOLERANCE else "FAIL"
        )
        row.update(mass_budget_error=max(mass_errors), energy_budget_error=max(energy_errors),
                   momentum_budget_error=max(momentum_errors), shock_speed=solution["shock_speed"])
        if any(not math.isfinite(float(value)) for value in row.values()
               if isinstance(value, (float, int))):
            raise RuntimeError("nonfinite comparison metric")
        if row["budget_accounting_status"] != "PASS":
            raise RuntimeError("runtime boundary/source accounting residual exceeds 1e-9")
        if max(row[f"{field}_budget_error"] for field in ("mass", "energy", "momentum")) > BUDGET_TOLERANCE:
            raise RuntimeError("analytic inflow budget error exceeds 1e-9")
        row["status"] = "PASS"
    except (RuntimeError, ValueError, OSError) as error:
        row["reason"] = str(error)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--force", action="store_true",
                        help="rerun instead of postprocessing existing outputs")
    args = parser.parse_args()
    check_oracle()
    rows = []
    for lorentz in (10.0, 100.0):
        previous = None
        sequence = []
        for cells in (128, 256, 512):
            row = run_case(lorentz, cells, args)
            if previous is not None:
                for component in COMPONENTS:
                    key = f"{component}_l1"
                    if key in row and key in previous and row[key] > 0:
                        order = math.log2(previous[key] / row[key])
                        row[f"{component}_order"] = order
            rows.append(row)
            sequence.append(row)
            previous = row
            write_tsv(SUMMARY, rows, list(dict.fromkeys(key for item in rows for key in item)))
            print(f"{row['status']} {row['case']}: {row['reason'] or 'budgets pass; profile errors recorded'}", flush=True)
        # Assess convergence over a factor four in resolution; do not mistake
        # the phase of a moving discontinuity relative to one mesh for order.
        coarse, fine = sequence[0], sequence[-1]
        for component in COMPONENTS:
            key = f"{component}_l1"
            if key in coarse and key in fine and fine[key] > 0:
                order = math.log(coarse[key] / fine[key], fine["cells"] / coarse["cells"])
                fine[f"{component}_sequence_order"] = order
                if order < MINIMUM_ORDER or fine[key] > ERROR_TOLERANCE:
                    fine["status"] = "FAIL"
                    fine["reason"] += f" {component}: finest L1={fine[key]:.4g}, sequence order={order:.3f}"
        write_tsv(SUMMARY, rows, list(dict.fromkeys(key for item in rows for key in item)))
        print(f"{fine['status']} W={lorentz:g} resolution sequence: {fine['reason'] or 'converged'}", flush=True)
    append_validation_history(ROOT / "runs/current/validation_history.tsv",
                              [{"kind": "analytic", **row} for row in rows])
    print(f"wrote {repo_relative(SUMMARY)}")
    return int(any(row["status"] != "PASS" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
