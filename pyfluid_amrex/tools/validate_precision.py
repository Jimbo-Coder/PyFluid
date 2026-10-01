#!/usr/bin/env python3
"""Compare FP32 and FP64 using the existing relativistic wave oracles.

This opt-in gate checks a precision/backend contract, not the full physics
suite. Scalar budgets allow 64 FP32 epsilons on these unit-scale, short runs;
analytic error and convergence requirements remain independent of that budget.
Gauge constraint norms are reported, without certifying FP32 NR constraint accuracy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from machine_common import load_machine
from plotfile_reader import read_scalars_metadata
from relativistic_validation import entropy_density_error
from validate_rhd import require_clean_relativistic_recovery
from validate_z4c import final_constraint_max, gauge_wave_lapse_error
from validation_common import (
    ROOT,
    append_validation_history,
    cartesian_divb_scale,
    materialize_run_input,
    precision_scalar_policy,
    real_epsilon,
    repo_relative,
    run_amrex,
    scalar_checks,
    scalar_validation_row,
    write_tsv,
)


INPUTS = ROOT / "pyfluid_amrex/inputs"
FP32_EPSILON = real_epsilon(24)


def run_wave(
    run_dir: Path, machine: str, bits: int, physics: str, cells: int,
    *, amplitude: float, final_time: float, ranks: int, timeout: int = 120,
) -> dict[str, object]:
    gauge = physics == "gauge"
    base = "z4c_gauge_wave_1d.in" if gauge else f"{physics}_entropy_wave_1d.in"
    parameters = {
        "pyfluid.run_name": run_dir.name,
        "pyfluid.output_dir": repo_relative(run_dir),
        "amr.n_cell": str(cells),
        "amr.max_grid_size": str(min(cells, 64)),
        # These tiny controls do not need AMReX's default GPU preallocation
        # of most device memory. The standard arena can still grow if needed.
        "amrex.the_arena_init_size": str(8 * 1024 * 1024),
        "pyfluid.final_time": str(final_time),
        "pyfluid.max_step": "100000",
        "pyfluid.scalar_int": "1",
        "pyfluid.plot_int": "1000000",
        "pyfluid.checkpoint_int": "-1",
        "pyfluid.physical_check_int": "1",
    }
    if gauge:
        parameters.update({
            "spacetime.amplitude": str(amplitude),
            "spacetime.ko_dissipation": "0",
            "pyfluid.cfl": "0.25",
        })
    input_path = materialize_run_input(INPUTS / base, run_dir / "input.in", parameters)
    result = run_amrex(
        dim=1, input_path=input_path, machine=machine, ranks=ranks, timeout=timeout,
    )
    (run_dir / "run.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"solver failed; see {repo_relative(run_dir / 'run.log')}")
    actual_bits = int(read_scalars_metadata(run_dir).get("real_mantissa_bits", "0"))
    if actual_bits != bits:
        raise RuntimeError(f"expected {bits} mantissa bits, output reports {actual_bits}")

    roundoff = 64 * real_epsilon(bits)
    checks = precision_scalar_policy(
        scalar_checks(f"precision_{bits}", energy=not gauge, momentum=not gauge,
                      divb=physics == "rmhd"),
        bits,
        # The entropy input has a constant background B1=0.75.
        divb_scale=(cartesian_divb_scale(run_dir, field_scale=0.75)
                    if physics == "rmhd" else None),
    )
    row = scalar_validation_row(
        run_dir, checks=checks, dim=1, ranks=ranks, mhd=physics == "rmhd",
        kind="precision", case=run_dir.name,
    )
    if abs(float(row["time"]) - final_time) > max(1e-12, 4 * roundoff * final_time):
        raise RuntimeError("run did not reach the requested physical time")
    require_clean_relativistic_recovery(run_dir)
    row.update({
        "machine": machine, "mantissa_bits": bits, "cells": cells,
        "conservation_tolerance": checks.mass_tol,
        "ledger_tolerance": checks.external_tol,
    })
    if gauge:
        row["analytic_error"] = gauge_wave_lapse_error(run_dir, amplitude, 1.0)
        row["constraint_linf"] = final_constraint_max(run_dir)
        # One percent of the weak wave; 0.01 percent of the resolved wave.
        # Both are absolute lapse L2 <= 1e-6 and apply to both precisions.
        row["physical_error_limit"] = 1e-6
    else:
        row["analytic_error"] = entropy_density_error(run_dir)
        row["physical_error_limit"] = 0.02 * (32 / cells) ** 1.8
    if not math.isfinite(float(row["analytic_error"])) or (
        float(row["analytic_error"]) > float(row["physical_error_limit"])
    ):
        row["status"] = "FAIL"
        row["reason"] = f"{row['reason']}; analytic error exceeds physical budget"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-machine", default="local")
    parser.add_argument("--candidate-machine", default="local_fp32")
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--candidate-ranks", type=int,
                        help="defaults to the candidate machine profile")
    parser.add_argument("--timeout", type=int, default=120,
                        help="per-run seconds, including first-use GPU kernel compilation")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/current/precision_validation")
    args = parser.parse_args()
    if args.candidate_ranks is None:
        args.candidate_ranks = load_machine(args.candidate_machine).default_ranks
    if args.ranks < 1 or args.candidate_ranks < 1 or args.timeout < 1:
        parser.error("rank counts and timeout must be positive")
    cases = [(physics, cells, 0.1, 0.2)
             for physics in ("rhd", "rmhd") for cells in (32, 64, 128)]
    cases += [("gauge", cells, 0.01, 0.2) for cells in (32, 64)]
    cases += [("gauge", 64, 0.0001, 0.02)]
    rows = []
    previous_errors: dict[tuple[str, int, float], float] = {}
    for physics, cells, amplitude, final_time in cases:
        pair = []
        case = f"{physics}_n{cells}_a{amplitude:g}"
        for label, machine, bits in (
            ("fp64", args.reference_machine, 53),
            ("fp32", args.candidate_machine, 24),
        ):
            run_dir = args.output_dir / label / case
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                row = run_wave(
                    run_dir, machine, bits, physics, cells, amplitude=amplitude,
                    final_time=final_time,
                    ranks=args.ranks if label == "fp64" else args.candidate_ranks,
                    timeout=args.timeout,
                )
                key = (physics, bits, amplitude)
                error = float(row["analytic_error"])
                if key in previous_errors:
                    order = math.log2(previous_errors[key] / error)
                    row["order"] = order
                    expected = 3.5 if physics == "gauge" else 1.8
                    # FP32 gauge waves may reach the storage-roundoff plateau.
                    resolved = bits == 53 or physics != "gauge" or error > 8 * FP32_EPSILON
                    if resolved and order < expected:
                        row["status"] = "FAIL"
                        row["reason"] = f"{row['reason']}; convergence order {order:.3f} < {expected}"
                previous_errors[key] = error
            except (RuntimeError, ValueError, OSError) as exc:
                row = {"kind": "precision", "case": case, "machine": machine,
                       "status": "FAIL", "reason": str(exc),
                       "run_dir": repo_relative(run_dir), "mantissa_bits": bits}
            pair.append(row)
            rows.append(row)
        if all("analytic_error" in row for row in pair):
            # Permit five percent of the reference discretization error plus
            # 16 storage epsilons, expressed in the oracle's normalization.
            normalization = amplitude if physics != "gauge" else 1.0
            limit = 1.05 * float(pair[0]["analytic_error"]) + 16 * FP32_EPSILON / normalization
            pair[1]["reference_error_limit"] = limit
            if float(pair[1]["analytic_error"]) > limit:
                pair[1]["status"] = "FAIL"
                pair[1]["reason"] = f"{pair[1]['reason']}; FP32 error exceeds reference allowance"
        for row in pair:
            print(f"{row['machine']} {case}: {row['status']} {row.get('reason', '')}",
                  flush=True)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_tsv(args.output_dir / "precision.tsv", rows, fields)
    append_validation_history(ROOT / "runs/current/validation_history.tsv", rows)
    failures = sum(row["status"] != "PASS" for row in rows)
    print(f"precision comparisons: {len(rows) - failures}/{len(rows)} pass")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
