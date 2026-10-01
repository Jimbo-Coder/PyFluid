#!/usr/bin/env python3
"""Small outgoing-Z4c boundary routing and preservation controls.

Pointwise outgoing-wave convergence lives in boundary_tiling_check.cpp.
Matched BH accuracy scans are separate, opt-in, and are not inferred here.
"""

from __future__ import annotations

import argparse
import math

from validate_black_hole import (
    COUPLED_EXCISION_INPUT,
    RUN_ROOT,
    run_case,
    validate_excision_amr_subcycling,
    validate_excision_off_cadence_restart,
    validate_filled_evolution,
)
from validation_common import (
    ROOT,
    append_validation_history,
    materialize_run_input,
    read_key_values,
    read_scalar_rows,
    repo_relative,
    run_amrex,
    write_tsv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--include-amr-restart", action="store_true")
    parser.add_argument("--include-filled-evolution", action="store_true")
    parser.add_argument("--outer-boundary",
                        choices=("extrapolate", "sommerfeld"),
                        default="sommerfeld")
    args = parser.parse_args()
    prefix = "z4c_boundary" if args.outer_boundary == "sommerfeld" \
        else f"z4c_boundary_{args.outer_boundary}"
    suffix = "_amr_restart" if args.include_amr_restart else ""
    if args.include_filled_evolution:
        suffix += "_filled"
    summary = RUN_ROOT / f"{prefix}{suffix}.tsv"
    # Never leave an earlier PASS looking current if a fresh run aborts.
    write_tsv(summary, [{
        "case": prefix, "status": "INCOMPLETE",
        "reason": "Current invocation has not completed all requested checks; see run logs",
    }], ["case", "status", "reason"])
    common = {
        "spacetime.outer_boundary": args.outer_boundary,
        "pyfluid.max_step": "4",
        "pyfluid.cfl": "0.9",
        "pyfluid.scalar_diagnostics": "standard,constraints",
        "pyfluid.debug_poison": "1",
        "pyfluid.debug_check_finite": "1",
        "pyfluid.plot_int": "-1",
        # The fixture's full-sphere extraction is unrelated to boundary tests
        # and would extend outside the octant used by the reflection control.
        "diagnostics.multipole_l_max": "0",
        "diagnostics.azimuthal_mode_fields": "none",
    }
    flat_input = ROOT / "pyfluid_amrex/inputs/z4c_minkowski_radiation_3d.in"
    cases = (
        ("z4c_boundary_flat_rk4", {"pyfluid.timestep": "rk4"}),
        ("z4c_boundary_flat_ssprk3", {"pyfluid.timestep": "ssprk3"}),
        ("z4c_boundary_flat_pure_wave", {"spacetime.sommerfeld_power": "-1"}),
        ("z4c_boundary_flat_live_lapse", {
            "spacetime.lapse_gauge": "one_plus_log",
        }),
        ("z4c_boundary_flat_live_shift", {
            "spacetime.lapse_gauge": "harmonic",
            "spacetime.shift_gauge": "gamma_driver",
        }),
        ("z4c_boundary_reflection", {
            "geometry.prob_lo": "0 0 0",
            "pyfluid.bc_xlo": "symmetry", "pyfluid.bc_ylo": "symmetry",
            "pyfluid.bc_zlo": "symmetry",
        }),
        ("z4c_boundary_gauge_noise", {
            "spacetime.initial_data": "robust_stability",
            "spacetime.amplitude": "1e-10",
            "spacetime.lapse_gauge": "one_plus_log",
            "spacetime.shift_gauge": "gamma_driver",
            "spacetime.ko_dissipation": "0.1",
        }),
        ("z4c_boundary_gauge_noise_split", {
            "spacetime.initial_data": "robust_stability",
            "spacetime.amplitude": "1e-10",
            "spacetime.lapse_gauge": "one_plus_log",
            "spacetime.shift_gauge": "gamma_driver",
            "spacetime.ko_dissipation": "0.1",
            "amr.max_grid_size": "4",
            "amr.blocking_factor": "4",
        }),
    )
    results, scalars = [], []
    for case_name, overrides in cases:
        name = case_name.replace("z4c_boundary", prefix, 1)
        row, scalar = run_case(
            name, {**common, **overrides}, ranks=args.ranks,
            timeout=args.timeout, base_input=flat_input,
            require_bh_exterior_constraints=False,
        )
        tolerance = 1e-6 if "noise" in name else 1e-12
        if row["hamiltonian_linf"] > tolerance:
            raise RuntimeError(f"{name}: flat-space constraint preservation failed")
        if row["interior_bounds"] != 0 or row["interior_resets"] != 0:
            raise RuntimeError(f"{name}: unexpected interior intervention")
        if name.endswith("noise_split"):
            original = read_scalar_rows(RUN_ROOT / f"{prefix}_gauge_noise")
            split = read_scalar_rows(RUN_ROOT / name)
            for coarse_row, split_row in zip(original, split, strict=True):
                for field in ("min_lapse", "z4c_hamiltonian_linf", "z4c_theta_linf"):
                    if not math.isclose(float(coarse_row[field]), float(split_row[field]),
                                        rel_tol=1e-10, abs_tol=2e-13):
                        raise RuntimeError(f"{name}: boundary correction depends on block layout: {field}")
        if case_name in ("z4c_boundary_flat_live_lapse", "z4c_boundary_flat_live_shift"):
            # Flat geometry isolates the actual CFL dispatch. Strong-field
            # alpha/chi dependence is checked by the pointwise C++ gate.
            run_dir = ROOT / row["run_dir"]
            config = read_key_values(run_dir / "run.in")
            low = [float(value) for value in config["geometry.prob_lo"].split()]
            high = [float(value) for value in config["geometry.prob_hi"].split()]
            cells = [int(value) for value in config["amr.n_cell"].split()]
            speed = math.sqrt(2.0 if name.endswith("lapse") else 4.0 / 3.0)
            frequency = speed * sum(
                count / (hi - lo) for count, lo, hi in zip(cells, low, high, strict=True)
            )
            expected_dt = float(config["pyfluid.cfl"]) / frequency
            for state in read_scalar_rows(run_dir)[1:]:
                if not math.isclose(float(state["dt"]), expected_dt, rel_tol=1e-10):
                    raise RuntimeError(f"{name}: gauge CFL disagrees with principal speed")
            row["expected_dt"] = expected_dt
        results.append(row)
        scalars.append(scalar)
        print(f"PASS {name}: H_Linf={row['hamiltonian_linf']:.6e}", flush=True)

    coupled_input = materialize_run_input(
        COUPLED_EXCISION_INPUT, RUN_ROOT / f"{prefix}_coupled.in", common)
    row, scalar = run_case(f"{prefix}_coupled", common, ranks=args.ranks,
                           timeout=args.timeout, base_input=coupled_input)
    if row["interior_bounds"] != 0 or row["interior_resets"] != 0:
        raise RuntimeError("outgoing coupled boundary used interior interventions")
    results.append(row)
    scalars.append(scalar)
    print(f"PASS {prefix}_coupled", flush=True)

    if args.outer_boundary != "extrapolate":
        rejected_dir = RUN_ROOT / f"{prefix}_bad_origin"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        rejected_input = materialize_run_input(flat_input, rejected_dir / "input.in", {
            **common, "spacetime.origin1": "-1",
            "pyfluid.output_dir": repo_relative(rejected_dir),
        })
        rejected = run_amrex(dim=3, input_path=rejected_input, ranks=args.ranks,
                             timeout=args.timeout)
        (rejected_dir / "run.log").write_text(rejected.stdout)
        if rejected.returncode == 0 or "strictly inside every open face" not in rejected.stdout:
            raise RuntimeError("outgoing spherical BC accepted an origin on an open face")
        results.append({"case": f"{prefix}_bad_origin", "status": "PASS"})
        print(f"PASS {prefix}_bad_origin: rejected at setup", flush=True)

    if args.include_amr_restart:
        for validator, name in (
            (validate_excision_amr_subcycling, f"{prefix}_coupled_amr"),
            (validate_excision_off_cadence_restart, f"{prefix}_coupled_restart"),
        ):
            row, scalar = validator(ranks=args.ranks, timeout=args.timeout,
                                    base_input=coupled_input, name=name)
            results.append(row)
            scalars.append(scalar)
            print(f"PASS {name}", flush=True)
    if args.include_filled_evolution:
        filled_results, filled_scalars = validate_filled_evolution(
            ranks=args.ranks, timeout=args.timeout,
            outer_boundary=args.outer_boundary,
            name_prefix="kerr_schild_filled_moving" if args.outer_boundary == "sommerfeld"
            else f"{prefix}_filled_moving",
        )
        results.extend(filled_results)
        scalars.extend(filled_scalars)
        for row in filled_results:
            print(f"PASS {row['case']}: t={row['time']:.3f}", flush=True)
    for row in results:
        row["outer_boundary"] = args.outer_boundary
    fields = list(dict.fromkeys(key for row in results for key in row))
    write_tsv(summary, results, fields)
    append_validation_history(ROOT / "runs/current/validation_history.tsv", scalars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
