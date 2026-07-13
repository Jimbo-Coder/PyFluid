#!/usr/bin/env python3
"""Validate causal scalar budgets and exterior quiescence for an open AMR run."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from postprocess_common import physics_validation_rows
from validation_common import (
   OPEN_POLICY,
   ROOT,
   append_validation_history,
   materialize_run_input,
   repo_relative,
   run_amrex,
   scalar_validation_row,
   write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/pressure_bump_2d_amr.in"
RUN_DIR = ROOT / "runs/current/causal_policy/open_pressure_bump_amr"
SUMMARY = RUN_DIR.parent / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=120)
   args = parser.parse_args()

   if args.force and RUN_DIR.exists():
      shutil.rmtree(RUN_DIR)
   parameters = {
      "pyfluid.run_name": RUN_DIR.name,
      "pyfluid.output_dir": repo_relative(RUN_DIR),
      "geometry.prob_lo": "0.0 0.0",
      "geometry.prob_hi": "1.0 1.0",
      "geometry.is_periodic": "0 0",
      "amr.n_cell": "24 24",
      "amr.max_level": "8",
      "amr.ref_ratio": "2 2",
      "amr.regrid_int": "1",
      "amr.tag_metric": "lohner",
      "amr.min_feature_length": "0.001",
      "amr.total_cell_budget": "1000000",
      "pyfluid.validation_policy": "open",
      "pyfluid.bc": "outflow",
      "pyfluid.ic": "pressure_bump",
      "pyfluid.rho0": "1.0",
      "pyfluid.pressure0": "1.0",
      "pyfluid.amplitude": "0.1",
      "pyfluid.bump_sigma": "0.02",
      "pyfluid.bump_x0": "0.5",
      "pyfluid.bump_y0": "0.5",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.ng": "2",
      "pyfluid.cfl": "0.9",
      "pyfluid.max_step": "64",
      "pyfluid.final_time": "0.08",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "1",
      "pyfluid.physical_check_int": "1",
   }
   input_path = materialize_run_input(BASE_INPUT, RUN_DIR / "input.in", parameters)
   result = run_amrex(dim=2, input_path=input_path, ranks=args.ranks, timeout=args.timeout)
   RUN_DIR.mkdir(parents=True, exist_ok=True)
   (RUN_DIR / "run.log").write_text(result.stdout)

   if result.returncode == 0:
      row = scalar_validation_row(
         RUN_DIR,
         case=RUN_DIR.name,
         input_file=RUN_DIR / "run.in",
         checks=OPEN_POLICY,
         mhd=False,
         dim=2,
         ranks=args.ranks,
         base_input=repo_relative(BASE_INPUT),
         kind="causal",
      )
      row["returncode"] = result.returncode
      row["seconds"] = result.seconds
      active_level = int(float(row.get("amr_active_finest_level", 0)))
      headroom = int(float(row.get("amr_finest_headroom", 0)))
      hierarchy_failures = []
      if active_level < 1:
         hierarchy_failures.append("automatic AMR never refined above level zero")
      if headroom < 2:
         hierarchy_failures.append(f"automatic AMR has only {headroom} level(s) of headroom")
      if hierarchy_failures:
         row["status"] = "FAIL"
         row["reason"] = "; ".join(hierarchy_failures)
      causal_rows = [entry for entry in physics_validation_rows(RUN_DIR) if entry["hook"] == "causal_quiescence"]
      if not causal_rows:
         row["status"] = "FAIL"
         row["reason"] = "causal exterior-quiescence hook did not run"
      elif causal_rows[0]["status"] != "PASS":
         row["status"] = "FAIL"
         row["reason"] = causal_rows[0]["reason"]
      row["causal_exterior_change"] = causal_rows[0]["value"] if causal_rows else float("nan")
   else:
      row = {
         "kind": "causal",
         "case": RUN_DIR.name,
         "label": RUN_DIR.name,
         "status": "FAIL",
         "reason": "solver failed",
         "returncode": result.returncode,
         "seconds": result.seconds,
         "dim": 2,
         "ranks": args.ranks,
         "run_dir": repo_relative(RUN_DIR),
      }

   fields = [
      "kind", "case", "status", "reason", "returncode", "seconds", "dim", "ranks",
      "causal_boundary_contact_time", "causal_precontact_rows", "causal_mass_rel_drift",
      "causal_energy_rel_drift", "causal_momentum_abs_drift", "causal_exterior_change",
      "amr_active_finest_level", "amr_finest_headroom", "amr_total_budget_fraction",
      "avg_mcells_per_s", "avg_sim_time_per_s", "run_dir",
   ]
   write_tsv(SUMMARY, [row], fields)
   append_validation_history(HISTORY, [row])
   print(
      f"{row['status']} causal policy: precontact_rows={row.get('causal_precontact_rows', 0)} "
      f"contact_time={float(row.get('causal_boundary_contact_time', float('nan'))):.3e} "
      f"exterior={float(row.get('causal_exterior_change', float('nan'))):.3e} "
      f"AMR=L{row.get('amr_active_finest_level', '?')}+{row.get('amr_finest_headroom', '?')} "
      f"run={repo_relative(RUN_DIR)}"
   )
   return 0 if row["status"] == "PASS" else 1


if __name__ == "__main__":
   raise SystemExit(main())
