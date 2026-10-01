#!/usr/bin/env python3
"""Evaluate one completed AMReX run and generate compact human-facing plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from plotting_common import (
   default_aux_components,
   default_snapshot_components,
   latest_plotfile,
   numbered_plot_path,
   plot_multivar_snapshot,
   plot_scalar_report,
   stable_plot_path,
)
from plot_amrex_slice import parse_plot_header
from validation_common import (
   ROOT,
   SCALAR_POLICIES,
   VALIDATION_HISTORY_FIELDS,
   append_validation_history,
   has_gravity_total_energy,
   parse_float,
   read_scalar_rows,
   repo_relative,
   scalar_validation_row,
   write_tsv,
)


MULTIRUN_SUMMARY = ROOT / "runs/current/run_report.tsv"


def parse_component_list(value: str) -> list[str]:
   value = value.strip()
   if not value or value == "none":
      return []
   return [token.strip() for token in value.split(",") if token.strip()]


def default_plot_dir(run_dir: Path) -> Path:
   return run_dir


def component_argument(value: str, names: list[str], *, aux: bool) -> list[str]:
   if value == "auto":
      return default_aux_components(names) if aux else default_snapshot_components(names)
   return parse_component_list(value)


def report_plot_path(args: argparse.Namespace, plot_dir: Path, stem: str) -> Path:
   return numbered_plot_path(plot_dir, stem) if args.numbered_plots else stable_plot_path(plot_dir, stem)


def remove_stale_stable_plot(args: argparse.Namespace, plot_dir: Path, stem: str) -> None:
   if args.numbered_plots:
      return
   path = stable_plot_path(plot_dir, stem)
   if path.exists():
      path.unlink()


def print_row(row: dict[str, object], plot_paths: Sequence[Path]) -> None:
   reason = f" ({row['reason']})" if row.get("reason") else ""
   energy_key = str(row.get("report_energy_key", "energy_rel_drift"))
   energy_label = "gravity_total" if energy_key == "gravity_total_energy_rel_drift" else "fluid"
   energy_value = parse_float(row.get(energy_key))
   plots = ", ".join(repo_relative(path) for path in plot_paths) if plot_paths else "none"
   amr_note = ""
   amr_level = parse_float(row.get("amr_active_finest_level"))
   if amr_level == amr_level:
      amr_budget = parse_float(row.get("amr_total_budget_fraction"), -1.0)
      budget_note = f" budget={amr_budget:.3e}" if amr_budget >= 0.0 else ""
      amr_note = f" AMR=L{amr_level:.0f}{budget_note}"
   passive_note = ""
   passive_drift = parse_float(row.get("passive_mass_rel_drift"))
   if passive_drift == passive_drift and passive_drift > 0.0:
      passive_note = f" passive={passive_drift:.3e}"
   causal_note = ""
   causal_contact = parse_float(row.get("causal_boundary_contact_time"))
   if causal_contact == causal_contact:
      causal_note = (
         f" causal_precontact={int(parse_float(row.get('causal_precontact_rows'), 0.0))}"
         f"@t<{causal_contact:.3e}"
      )
   ledger_note = ""
   if parse_float(row.get("external_budget_valid"), 0.0) > 0.5:
      ledger_note = (
         f" ledger(m={parse_float(row.get('external_mass_rel_error')):.3e},"
         f"E={parse_float(row.get('external_energy_rel_error')):.3e})"
      )
   elif parse_float(row.get("external_mass_budget_valid"), 0.0) > 0.5:
      ledger_note = f" baryon_ledger={parse_float(row.get('external_mass_rel_error')):.3e}"
   print(
      f"{row['status']} {row['case']}: policy={row['policy']} "
      f"mass={parse_float(row.get('mass_rel_drift')):.3e} "
      f"{energy_label}_energy={energy_value:.3e} "
      f"mom={parse_float(row.get('momentum_abs_drift')):.3e} "
      f"divb={parse_float(row.get('divb_l1'), 0.0):.3e} "
      f"Mcells/s={parse_float(row.get('avg_mcells_per_s')):.3e} "
      f"sim_time/s={parse_float(row.get('avg_sim_time_per_s')):.3e}"
      f"{amr_note}{passive_note}{causal_note}{ledger_note}{reason} "
      f"plots={plots}"
   )


def report_one(args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, object], list[Path]]:
   run_dir = run_dir.resolve()
   row = scalar_validation_row(
      run_dir,
      input_file=args.input,
      policy=None if args.policy == "auto" else args.policy,
      mhd=None if args.mhd == "auto" else args.mhd == "yes",
      kind="run_report",
   )
   row = dict(row)
   scalar_rows = read_scalar_rows(run_dir)
   row["report_energy_key"] = "gravity_total_energy_rel_drift" if has_gravity_total_energy(scalar_rows) else "energy_rel_drift"
   plot_paths: list[Path] = []
   plot_dir = (args.plot_dir.resolve() if args.plot_dir else default_plot_dir(run_dir))

   if not args.skip_plots:
      remove_stale_stable_plot(args, plot_dir, f"{run_dir.name}_snapshot")
      remove_stale_stable_plot(args, plot_dir, f"{run_dir.name}_aux_snapshot")

      scalar_path = run_dir / "scalars.tsv"
      if scalar_path.exists():
         scalar_plot = report_plot_path(args, plot_dir, f"{run_dir.name}_scalars")
         plot_scalar_report(scalar_path, scalar_plot, title=f"{run_dir.name}: scalar diagnostics")
         plot_paths.append(scalar_plot)

      state_plotfile = latest_plotfile(run_dir, "stateGF")
      if args.gf_plots and state_plotfile is not None:
         names, ndim, _time_value = parse_plot_header(state_plotfile)
         if ndim >= 2:
            components = component_argument(args.components, names, aux=False)
            if components:
               snapshot = report_plot_path(args, plot_dir, f"{run_dir.name}_snapshot")
               rendered = plot_multivar_snapshot(
                  state_plotfile,
                  snapshot,
                  components=components,
                  title=f"{run_dir.name}: state snapshot",
                  gamma=args.gamma,
                  ref_ratio=args.ref_ratio,
                  max_pixels=args.max_pixels,
                  slice_axis=args.slice_axis,
                  slice_fraction=args.slice_fraction,
                  show_boxes=args.show_boxes,
               )
               if rendered:
                  plot_paths.append(snapshot)

      aux_plotfile = latest_plotfile(run_dir, "auxGF")
      if args.gf_plots and aux_plotfile is not None:
         names, ndim, _time_value = parse_plot_header(aux_plotfile)
         if ndim >= 2:
            components = component_argument(args.aux_components, names, aux=True)
            if components:
               snapshot = report_plot_path(args, plot_dir, f"{run_dir.name}_aux_snapshot")
               rendered = plot_multivar_snapshot(
                  aux_plotfile,
                  snapshot,
                  components=components,
                  title=f"{run_dir.name}: auxiliary GF snapshot",
                  gamma=args.gamma,
                  ref_ratio=args.ref_ratio,
                  max_pixels=args.max_pixels,
                  slice_axis=args.slice_axis,
                  slice_fraction=args.slice_fraction,
                  show_boxes=args.show_boxes,
               )
               if rendered:
                  plot_paths.append(snapshot)

   row["plots"] = ",".join(repo_relative(path) for path in plot_paths)
   return row, plot_paths


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("run_dir", nargs="+", type=Path)
   parser.add_argument("--input", type=Path, help="Input file for a single run.")
   parser.add_argument("--policy", choices=["auto", *SCALAR_POLICIES.keys()], default="auto")
   parser.add_argument("--mhd", choices=("auto", "yes", "no"), default="auto")
   parser.add_argument("--output", type=Path)
   parser.add_argument("--plot-dir", type=Path, help="Plot output directory. Defaults to <run_dir>.")
   parser.add_argument("--skip-plots", action="store_true")
   parser.add_argument("--gf-plots", action="store_true", help="Also generate level-2 state/aux GF snapshot plots from plotfiles.")
   parser.add_argument("--numbered-plots", action="store_true", help="Append _001-style suffixes instead of overwriting report plots.")
   parser.add_argument("--append-history", action=argparse.BooleanOptionalAction, default=True)
   parser.add_argument("--components", default="auto", help="Comma-separated state components, auto, or none.")
   parser.add_argument("--aux-components", default="auto", help="Comma-separated aux components, auto, or none.")
   parser.add_argument("--show-boxes", action="store_true")
   parser.add_argument("--ref-ratio", type=int, default=2)
   parser.add_argument("--max-pixels", type=int, default=1000)
   parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
   parser.add_argument("--slice-fraction", type=float, default=0.5)
   parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
   args = parser.parse_args()

   if args.input is not None and len(args.run_dir) != 1:
      raise SystemExit("--input is only valid for a single run_dir")
   rows: list[dict[str, object]] = []
   failed = False
   for run_dir in args.run_dir:
      row, plots = report_one(args, run_dir)
      rows.append(row)
      failed = failed or row["status"] != "PASS"
      print_row(row, plots)

   if args.output is None:
      output = args.run_dir[0].resolve() / "report.tsv" if len(args.run_dir) == 1 else MULTIRUN_SUMMARY
   else:
      output = args.output if args.output.is_absolute() else ROOT / args.output
   fields = [
      *VALIDATION_HISTORY_FIELDS,
      "run_dir",
      "plots",
   ]
   write_tsv(output, rows, fields)
   if args.append_history:
      append_validation_history(ROOT / "runs/current/validation_history.tsv", rows)
   print(f"wrote {repo_relative(output)}")
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
