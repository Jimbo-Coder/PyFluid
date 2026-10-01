#!/usr/bin/env python3
"""Exact finite-amplitude relativistic CP Alfven convergence, ECHO section 4.1.

Uses a unit box (one wavelength), rho=p=B_normal=B_transverse=1, gamma=4/3.
The multidimensional CT control has eight cells in each transverse direction; it tests CT
coupling, not oblique propagation. No comparison is a Newtonian wave oracle.
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np

from machine_common import load_machine
from plotfile_reader import (
    latest_plotfile, parse_plot_header, read_level_state, read_scalars_metadata,
)
from relativistic_validation import circular_alfven_conserved, circular_alfven_speed
from validation_common import (
    ROOT, append_validation_history, cartesian_divb_scale, materialize_run_input,
    precision_scalar_policy, real_epsilon, repo_relative, run_amrex, scalar_checks,
    scalar_validation_row, write_tsv,
)
from validate_rmhd import relativistic_recovery_counts

RUN_ROOT = ROOT / "runs/current/rmhd_alfven"


def run_case(cells: int, dim: int, args: argparse.Namespace) -> dict:
    name = f"cp_{dim}d_n{cells}"
    out = args.output_dir / name
    shape = (cells,) + (8,) * (dim - 1)
    row = {"case": name, "kind": "analytic", "cells": cells, "dim": dim,
           "status": "FAIL", "reason": "", "run_dir": repo_relative(out)}
    if args.force and out.exists():
        shutil.rmtree(out)
    period = 1.0 / circular_alfven_speed()
    parameters = {
        "pyfluid.run_name": name, "pyfluid.output_dir": repo_relative(out),
        "geometry.prob_lo": " ".join(["0"] * dim),
        "geometry.prob_hi": " ".join(["1"] * dim),
        "geometry.is_periodic": " ".join(["1"] * dim),
        "amr.n_cell": " ".join(map(str, shape)),
        "amr.max_grid_size": "32", "pyfluid.ic": "circular_alfven_x",
        "pyfluid.rho0": "1", "pyfluid.pressure0": "1", "pyfluid.b1_0": "1",
        "pyfluid.b2_0": "0", "pyfluid.b3_0": "0",
        "pyfluid.alfven_amplitude": "1", "pyfluid.wave_number": "1",
        "pyfluid.gamma": repr(4.0 / 3.0), "pyfluid.final_time": repr(period),
        "pyfluid.ct": "off" if dim == 1 else "on",
        "pyfluid.aux_gfs": "none", "pyfluid.aux_gf_compute_int": "-1",
        "pyfluid.aux_gf_int": "-1",
        "amrex.the_arena_init_size": str(8 * 1024 * 1024),
    }
    try:
        inp = materialize_run_input(ROOT / "pyfluid_amrex/inputs/rmhd_entropy_wave_1d.in",
                                    out / "input.in", parameters)
        if args.force or latest_plotfile(out) is None:
            result = run_amrex(dim=dim, input_path=inp, ranks=args.ranks,
                               machine=args.machine, timeout=args.timeout)
            (out / "run.log").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError("solver failed; see run.log")
        bits = int(read_scalars_metadata(out).get("real_mantissa_bits", "0"))
        roundoff = 64 * real_epsilon(bits)
        checks = precision_scalar_policy(
            scalar_checks(f"alfven_{bits}"), bits,
            # The declared wave has unit normal/transverse field amplitudes.
            divb_scale=cartesian_divb_scale(out, field_scale=1.0),
        )
        scalar = scalar_validation_row(out, case=name, mhd=True, dim=dim,
                                       ranks=args.ranks, kind="analytic", checks=checks)
        row.update(scalar)
        row.update(machine=args.machine.name, mantissa_bits=bits,
                   conservation_tolerance=checks.mass_tol)
        append_validation_history(ROOT / "runs/current/validation_history.tsv", [scalar])
        if scalar["status"] != "PASS":
            raise RuntimeError(f"scalar checks: {scalar['reason']}")
        if any(relativistic_recovery_counts(out)):
            raise RuntimeError("wave required a recovery fallback or atmosphere reset")
        plot = latest_plotfile(out, require=True)
        names, ndim, time = parse_plot_header(plot)
        if ndim != dim or not math.isclose(time, period, rel_tol=max(1e-12, roundoff)):
            raise RuntimeError("wrong output dimension or final time")
        state = read_level_state(plot, names, ndim, 0)
        if state.shape[1:] != shape or not np.all(np.isfinite(state)):
            raise RuntimeError("unexpected mesh or nonfinite conserved output")
        exact = circular_alfven_conserved(cells, time)
        errors = []
        for component, reference in exact.items():
            actual = state[names.index(component)]
            target = reference.reshape((cells,) + (1,) * (dim - 1))
            error = float(np.mean(abs(actual - target))) / max(1.0, float(np.max(abs(reference))))
            row[f"{component}_l1"] = error
            errors.append(error)
        row["max_l1"] = max(errors)
        row["status"] = "PASS"
    except (RuntimeError, ValueError, OSError) as error:
        row["status"] = "FAIL"
        row["reason"] = str(error)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--machine", help="machine profile; otherwise PYFLUID_MACHINE or local")
    parser.add_argument("--ranks", type=int, help="defaults to the selected machine profile")
    parser.add_argument("--dims", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    args.machine = load_machine(args.machine)
    if args.ranks is None:
        args.ranks = args.machine.default_ranks
    if args.ranks < 1 or args.timeout < 1:
        parser.error("ranks and timeout must be positive")
    rows = []
    for dim in dict.fromkeys(args.dims):
        sequence = [run_case(n, dim, args) for n in (32, 64, 128)]
        coarse, fine = sequence[0], sequence[-1]
        if "max_l1" in coarse and "max_l1" in fine:
            order = math.log(coarse["max_l1"] / fine["max_l1"], 4.0)
            fine["sequence_order"] = order
            for component in ("mom2", "mom3", "b2", "b3"):
                component_order = math.log(coarse[f"{component}_l1"] / fine[f"{component}_l1"], 4.0)
                fine[f"{component}_order"] = component_order
                if component_order < 1.7:
                    fine["status"] = "FAIL"
                    fine["reason"] += f" {component} order={component_order:.3f}"
            if order < 1.7 or fine["max_l1"] > 0.01:
                fine["status"] = "FAIL"
                fine["reason"] += f" convergence order={order:.3f}, finest error={fine['max_l1']:.3e}"
        rows.extend(sequence)
        write_tsv(args.output_dir / "summary.tsv", rows,
                  list(dict.fromkeys(key for row in rows for key in row)))
        for row in sequence:
            print(row, flush=True)
    append_validation_history(ROOT / "runs/current/validation_history.tsv", rows)
    return int(any(row["status"] != "PASS" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
