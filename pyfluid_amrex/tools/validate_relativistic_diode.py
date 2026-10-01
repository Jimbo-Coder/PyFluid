#!/usr/bin/env python3
"""Check no-inflow numerical baryon flux on shifted full-grid boundaries."""

from __future__ import annotations

import argparse

from validate_black_hole import COUPLED_EXCISION_INPUT, EXCISION_INPUT, RUN_ROOT, run_case
from validation_common import ROOT, append_validation_history, read_tsv, write_tsv


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/kerr_schild_bounded_fixed_evolution_3d.in"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    magnetic = {
        "pyfluid.physics": "rmhd", "pyfluid.ct": "on",
        "pyfluid.ic": "sod_plane",
        "pyfluid.shock_nx": "0", "pyfluid.shock_ny": "0",
        "pyfluid.shock_nz": "1",
        "pyfluid.rho_left": "1e-4", "pyfluid.rho_right": "1e-4",
        "pyfluid.pressure_left": "1e-6", "pyfluid.pressure_right": "1e-6",
        "pyfluid.b3_left": "1e-6", "pyfluid.b3_right": "1e-6",
        "pyfluid.b3_0": "1e-6",
    }
    cases = {
        "outflow_control": {"pyfluid.bc": "outflow"},
        "rhd_fixed": {},
        "rhd_fixed_ssprk3": {
            "pyfluid.recon": "mp5",
            "pyfluid.riemann": "llf",
            "pyfluid.timestep": "ssprk3",
        },
        # The separate coarse filled/evolved MP5 probe fails with either
        # outflow or diode (Cycle 367). Isolate the boundary regression using
        # the already validated evolved-excision interior, not a hidden floor
        # change or reduced CFL. Its ledger includes the inner surface too.
        "rhd_coupled_excision": {},
        "rmhd_flat": {
            **magnetic,
            "spacetime.initial_data": "minkowski",
            "spacetime.fixed_metric": "minkowski",
            "spacetime.black_hole_matter": "evolve",
            "spacetime.dimensionless_spin3": "0",
            "spacetime.black_hole_interior": "exterior",
            "spacetime.black_hole_fill_radius": "-1",
            "horizon.enabled": "0",
        },
        "rmhd_hlld_excision": {
            **magnetic,
            "pyfluid.recon": "mp5", "pyfluid.riemann": "hlld",
            # Exercise the candidate path too, not only a high-order flux.
            "pyfluid.rmhd_candidate_guard": "pwl-pwc",
            "pyfluid.rmhd_candidate_dmp": "1",
            "pyfluid.rmhd_candidate_dmp_factor": "1",
        },
    }
    results, scalar_rows = [], []
    for label, parameters in cases.items():
        name = "diode_" + label
        base_input = {
            "rhd_coupled_excision": COUPLED_EXCISION_INPUT,
            "rmhd_hlld_excision": EXCISION_INPUT,
        }.get(label, BASE_INPUT)
        result, scalar = run_case(
            name, {"pyfluid.max_step": "3", "pyfluid.plot_int": "-1",
                   "pyfluid.bc": "diode", **parameters},
            ranks=args.ranks, timeout=args.timeout,
            base_input=base_input,
            require_bh_exterior_constraints=label != "rmhd_flat",
        )
        records = read_tsv(RUN_ROOT / name / "scalars.tsv", require_complete=True)
        boundary = [float(record["boundary_mass_change"]) for record in records]
        result["boundary_mass_change"] = boundary[-1]
        result["external_mass_rel_error"] = scalar["external_mass_rel_error"]
        if label == "outflow_control":
            if boundary[-1] <= 1e-8:
                raise RuntimeError("outflow control did not demonstrate shift-driven mass influx")
        elif max(boundary) > 1e-14:
            raise RuntimeError(f"{name} admitted baryon mass through its numerical boundary flux")
        if float(scalar["external_mass_rel_error"]) > 1e-12:
            raise RuntimeError(f"{name} failed baryon closure")
        if result["interior_resets"] != 0 or result["interior_bounds"] != 0:
            raise RuntimeError(f"{name} needed an interior intervention")
        if label == "rmhd_hlld_excision":
            log_lines = (RUN_ROOT / name / "run.log").read_text().splitlines()
            recovery_line = next(line for line in reversed(log_lines)
                                 if line.startswith("Relativistic recovery "))
            counts = dict(token.split("=", 1) for token in recovery_line.split()[2:])
            corrections = sum(int(counts[key]) for key in
                              ("rmhd_pwl_corrections", "rmhd_pwc_corrections"))
            result["candidate_corrections"] = corrections
            if corrections == 0:
                raise RuntimeError("diode RMHD control did not exercise candidate correction")
        if label.startswith("rmhd"):
            result["divb_linf"] = records[-1]["divb_linf"]
            if max(abs(float(record["divb_linf"])) for record in records) > 1e-14:
                raise RuntimeError("diode RMHD control lost its divergence-free CT field")
        results.append(result)
        scalar_rows.append(scalar)
        write_tsv(RUN_ROOT / "diode_validation.tsv", results,
                  list(dict.fromkeys(key for row in results for key in row)))
        print(f"PASS {name}: boundary_mass_change={boundary[-1]:.6e}", flush=True)
    append_validation_history(ROOT / "runs/current/validation_history.tsv", scalar_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
