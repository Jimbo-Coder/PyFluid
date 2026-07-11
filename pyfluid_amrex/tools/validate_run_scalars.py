#!/usr/bin/env python3
"""Check scalar diagnostics for one or more completed AMReX run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation_common import (
   SCALAR_POLICIES,
   append_validation_history,
   parse_float,
   repo_relative,
   scalar_validation_row,
   write_tsv,
)


ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "runs/current/validation_history.tsv"
SUMMARY = ROOT / "runs/current/scalar_sanity.tsv"


def validate_run(
   run_dir: Path,
   *,
   input_file: Path | None,
   policy_override: str,
   mhd_override: str,
) -> dict[str, object]:
   return scalar_validation_row(
      run_dir,
      input_file=input_file,
      policy=None if policy_override == "auto" else policy_override,
      mhd=None if mhd_override == "auto" else mhd_override == "yes",
   )


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
   run_dirs = [Path(path) for path in args.run_dir]
   for pattern in args.glob or []:
      run_dirs.extend(Path(path).parent for path in ROOT.glob(pattern))
   unique = []
   seen = set()
   for run_dir in run_dirs:
      path = run_dir.resolve()
      if path not in seen:
         unique.append(path)
         seen.add(path)
   return unique


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("run_dir", nargs="*", help="Completed run directory containing scalars.tsv.")
   parser.add_argument("--glob", action="append", help="Repo-relative glob for scalars.tsv files to check.")
   parser.add_argument("--input", type=Path, help="Input file used for a single run.")
   parser.add_argument("--policy", choices=["auto", *SCALAR_POLICIES.keys()], default="auto")
   parser.add_argument("--mhd", choices=("auto", "yes", "no"), default="auto")
   parser.add_argument("--output", type=Path, default=SUMMARY)
   parser.add_argument("--append-history", action=argparse.BooleanOptionalAction, default=True)
   args = parser.parse_args()

   run_dirs = resolve_run_dirs(args)
   if not run_dirs:
      raise SystemExit("provide at least one run_dir or --glob")
   if args.input is not None and len(run_dirs) != 1:
      raise SystemExit("--input is only valid for a single run_dir")

   rows = [
      validate_run(run_dir, input_file=args.input, policy_override=args.policy, mhd_override=args.mhd)
      for run_dir in run_dirs
   ]
   fields = [
      "case",
      "status",
      "reason",
      "policy",
      "description",
      "input",
      "run_dir",
      "steps",
      "time",
      "min_rho",
      "min_pressure",
      "mass_rel_drift",
      "energy_rel_drift",
      "gravity_total_energy_rel_drift",
      "momentum_abs_drift",
      "divb_l1",
      "causal_boundary_contact_time",
      "causal_precontact_rows",
      "causal_mass_rel_drift",
      "causal_energy_rel_drift",
      "causal_momentum_abs_drift",
      "avg_mcells_per_s",
      "avg_sim_time_per_s",
      "diffusion_residual",
      "diffusion_iterations",
   ]
   output = args.output if args.output.is_absolute() else ROOT / args.output
   write_tsv(output, rows, fields)
   if args.append_history:
      append_validation_history(HISTORY, rows)

   failed = [row for row in rows if row["status"] != "PASS"]
   for row in rows:
      reason = f" ({row['reason']})" if row["reason"] else ""
      has_gravity_total = "gravity_total_energy_rel_drift" in row and row["gravity_total_energy_rel_drift"] != ""
      energy_label = "gravity_total" if has_gravity_total else "fluid"
      energy_value = parse_float(
         row["gravity_total_energy_rel_drift"] if has_gravity_total else row["energy_rel_drift"]
      )
      print(
         f"{row['status']} {row['case']}: policy={row['policy']} "
         f"mass={parse_float(row['mass_rel_drift']):.3e} "
         f"{energy_label}_energy={energy_value:.3e} "
         f"divb={parse_float(row['divb_l1'], 0.0):.3e}{reason}"
      )
   print(f"wrote {repo_relative(output)}")
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
