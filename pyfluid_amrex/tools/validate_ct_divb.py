#!/usr/bin/env python3
"""Validate that constrained transport controls magnetic divergence."""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from validation_common import (
   append_validation_history,
   average_positive,
   materialize_run_input,
   next_numbered_path,
   parameters_from_assignments,
   parse_float,
   read_tsv,
   repo_relative,
   run_amrex,
   write_tsv,
)


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build/pyfluid_amrex/pyfluid_amrex_2d"
INPUT = ROOT / "pyfluid_amrex/inputs/kelvin_helmholtz_mhd_2d_single_sheet_amr_ct.in"
RUN_ROOT = ROOT / "runs/current/ct_divb_compare"
PLOT_ROOT = ROOT / "plots/current/scans"
RESULTS = RUN_ROOT / "ct_divb_compare.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"


@dataclass(frozen=True)
class Case:
   name: str
   label: str
   ct: str


CASES = (
   Case("ct_on", "CT on", "on"),
   Case("ct_off", "CT off", "off"),
)


COMMON_OVERRIDES = (
   "pyfluid.final_time=-1",
   "pyfluid.max_step=64",
   "pyfluid.scalar_int=1",
   "pyfluid.plot_int=-1",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.aux_gf_int=-1",
   "pyfluid.diffusion=none",
   "pyfluid.viscosity=0",
   "pyfluid.resistivity=0",
   "amr.n_cell=64 64",
   "amr.max_level=0",
)


def run_case(case: Case, args: argparse.Namespace) -> dict[str, object]:
   output_dir = RUN_ROOT / case.name
   if args.force and output_dir.exists():
      shutil.rmtree(output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)
   overrides = [
      f"pyfluid.output_dir={repo_relative(output_dir)}",
      f"pyfluid.ct={case.ct}",
      *COMMON_OVERRIDES,
   ]
   input_path = materialize_run_input(
      INPUT,
      output_dir / "run.in",
      parameters_from_assignments(overrides),
      remove_parameters=("amr.ref_ratio",),
   )
   completed = run_amrex(dim=2, input_path=input_path, ranks=args.ranks, executable=EXE)
   (output_dir / "run.log").write_text(completed.stdout)
   if completed.returncode != 0:
      return {
         "case": case.name,
         "label": case.label,
         "status": "FAIL",
         "reason": f"run failed; see {repo_relative(output_dir / 'run.log')}",
         "run_dir": repo_relative(output_dir),
      }

   rows = read_tsv(output_dir / "scalars.tsv", require_complete=True)
   divb = [parse_float(row.get("divb_l1")) for row in rows]
   finite_divb = [value for value in divb if math.isfinite(value)]
   final = rows[-1] if rows else {}
   return {
      "case": case.name,
      "label": case.label,
      "status": "PASS",
      "reason": "",
      "ranks": args.ranks,
      "steps": final.get("step", ""),
      "time": final.get("time", ""),
      "final_divb_l1": finite_divb[-1] if finite_divb else math.nan,
      "max_divb_l1": max(finite_divb) if finite_divb else math.nan,
      "mass_rel_drift": final.get("mass_rel_drift", ""),
      "energy_rel_drift": final.get("energy_rel_drift", ""),
      "avg_mcells_per_s": average_positive(rows, "mcells_per_s"),
      "avg_sim_time_per_s": average_positive(rows, "sim_time_per_s"),
      "run_dir": repo_relative(output_dir),
   }


def plot_divb(output: Path) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
   for case in CASES:
      rows = read_tsv(RUN_ROOT / case.name / "scalars.tsv", require_complete=True)
      t = [parse_float(row.get("time")) for row in rows]
      divb = [max(parse_float(row.get("divb_l1")), 1.0e-18) for row in rows]
      style = "-" if case.ct == "on" else "--"
      ax.semilogy(t, divb, style, marker="o", markersize=3.0, linewidth=1.8, label=case.label)
   ax.set_xlabel("time")
   ax.set_ylabel("divb L1")
   ax.set_title("MHD KH 2D divb control: WENO-Z/HLLD/SSPRK3, CFL=0.9, periodic/outflow BC")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend()
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def add_status(rows: list[dict[str, object]], args: argparse.Namespace) -> bool:
   by_case = {str(row["case"]): row for row in rows}
   failures = []
   ct_on = parse_float(by_case["ct_on"].get("max_divb_l1"))
   ct_off = parse_float(by_case["ct_off"].get("max_divb_l1"))
   if ct_on > args.ct_tol:
      failures.append(f"CT max divb {ct_on:.3e} > {args.ct_tol:.3e}")
   if ct_off < args.noct_min_divb:
      failures.append(f"no-CT max divb {ct_off:.3e} < {args.noct_min_divb:.3e}")
   if ct_off / max(ct_on, 1.0e-300) < args.min_ratio:
      failures.append(f"no-CT/CT divb ratio too small: {ct_off / max(ct_on, 1.0e-300):.3e}")
   status = "PASS" if not failures else "FAIL"
   reason = "; ".join(failures)
   for row in rows:
      if row["status"] == "PASS":
         row["status"] = status
         row["reason"] = reason
   return not failures


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--ct-tol", type=float, default=1.0e-10)
   parser.add_argument("--noct-min-divb", type=float, default=1.0e-5)
   parser.add_argument("--min-ratio", type=float, default=1.0e6)
   parser.add_argument("--skip-plot", action="store_true", help="write TSV/history only")
   args = parser.parse_args()

   rows = [run_case(case, args) for case in CASES]
   ok = all(row["status"] == "PASS" for row in rows)
   if ok:
      ok = add_status(rows, args)
      plot_path = None
      if not args.skip_plot:
         plot_path = next_numbered_path(PLOT_ROOT, "mhd_ct_divb_compare")
         plot_divb(plot_path)
      for row in rows:
         row["plot"] = repo_relative(plot_path) if plot_path is not None else ""
   else:
      for row in rows:
         row["plot"] = ""

   fields = [
      "case",
      "label",
      "status",
      "reason",
      "ranks",
      "steps",
      "time",
      "final_divb_l1",
      "max_divb_l1",
      "mass_rel_drift",
      "energy_rel_drift",
      "avg_mcells_per_s",
      "avg_sim_time_per_s",
      "plot",
      "run_dir",
   ]
   write_tsv(RESULTS, rows, fields)
   append_validation_history(HISTORY, [{"kind": "ct_divb", **row} for row in rows])

   by_case = {str(row["case"]): row for row in rows}
   print(
      f"{'PASS' if ok else 'FAIL'} ct_divb: "
      f"CT max={parse_float(by_case['ct_on'].get('max_divb_l1')):.3e}, "
      f"no-CT max={parse_float(by_case['ct_off'].get('max_divb_l1')):.3e}"
   )
   print(f"wrote {repo_relative(RESULTS)}")
   if ok and by_case["ct_on"]["plot"]:
      print(f"wrote {by_case['ct_on']['plot']}")
   return 0 if ok else 1


if __name__ == "__main__":
   raise SystemExit(main())
