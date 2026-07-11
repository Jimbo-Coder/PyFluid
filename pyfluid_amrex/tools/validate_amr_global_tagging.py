#!/usr/bin/env python3
"""Check that AMR can refine broad flow structure within budget."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from validation_common import (
   CLOSED_POLICY,
   VALIDATION_HISTORY_FIELDS,
   append_validation_history,
   materialize_run_input,
   read_tsv,
   repo_relative,
   run_amrex,
   scalar_validation_row,
   write_tsv,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "pyfluid_amrex/inputs/hydro_sine_2d.in"
RUN_ROOT = ROOT / "runs/current/amr_global_tagging"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"


COMMON_OVERRIDES = {
   "pyfluid.max_step": "3",
   "pyfluid.final_time": "-1",
   "pyfluid.scalar_int": "1",
   "pyfluid.plot_int": "-1",
   "pyfluid.checkpoint_int": "-1",
   "pyfluid.ng": "3",
   "pyfluid.recon": "pwl",
   "pyfluid.riemann": "hllc",
   "pyfluid.timestep": "ssprk3",
   "pyfluid.cfl": "0.9",
   "amr.max_level": "2",
   "amr.ref_ratio": "2 2",
   "amr.regrid_int": "1",
   "amr.tag_metric": "auto",
   "amr.tag_threshold": "0.02",
   "amr.initial_refine": "none",
   "amr.max_grid_size": "32",
   "amr.blocking_factor": "8",
   "amr.level_cell_budget_side": "512",
   "amr.total_cell_budget": "500000",
}


@dataclass(frozen=True)
class TagCase:
   name: str
   overrides: dict[str, str]
   min_level: int
   min_cell_ratio: float
   max_budget_fraction: float = 0.95


CASES = (
   TagCase(
      name="flow_sine_global",
      overrides={
         "pyfluid.ic": "sin_wave",
         "pyfluid.amplitude": "0.1",
         "pyfluid.wave_number": "8",
         "pyfluid.passive_scalars": "0",
      },
      # Three coarse steps exercise broad automatic tagging, not a full
      # two-stage hierarchy build. Require the broad L1 refinement it can
      # actually establish in that short gate.
      min_level=1,
      min_cell_ratio=4.0,
   ),
   TagCase(
      name="passive_ignored_by_amr",
      overrides={
         "pyfluid.ic": "sin_wave",
         "pyfluid.amplitude": "0.0",
         "pyfluid.wave_number": "8",
         "pyfluid.passive_scalars": "1",
         "pyfluid.passive_profile": "sin_wave",
         "pyfluid.passive_background": "0.5",
         "pyfluid.passive_amplitude": "0.45",
      },
      min_level=0,
      min_cell_ratio=1.0,
   ),
)


def final_level_summary(run_dir: Path) -> dict[str, float]:
   rows = read_tsv(run_dir / "amr_levels.tsv", require_complete=True)
   if not rows:
      raise ValueError(f"missing rows in {run_dir / 'amr_levels.tsv'}")
   final = rows[-1]
   return {
      "total_cells": float(final["total_cells"]),
      "active_finest_level": float(final["active_finest_level"]),
      "budget_fraction": float(final["total_budget_fraction"]),
      "peak_level_budget_fraction": float(final["peak_level_budget_fraction"]),
   }


def run_case(case: TagCase, ranks: int, timeout: int | None) -> dict[str, object]:
   run_dir = RUN_ROOT / case.name
   values = dict(COMMON_OVERRIDES)
   values.update(case.overrides)
   values.update({
      "pyfluid.run_name": case.name,
      "pyfluid.output_dir": repo_relative(run_dir),
   })
   run_input = materialize_run_input(INPUT, run_dir / "run.in", values)

   result = run_amrex(
      dim=2,
      input_path=run_input,
      ranks=ranks,
      timeout=timeout,
   )
   run_dir.mkdir(parents=True, exist_ok=True)
   with (run_dir / "run.log").open("w") as f:
      f.write("$ " + " ".join(result.command) + "\n")
      f.write(f"# wall_seconds = {result.seconds:.6g}\n")
      f.write(result.stdout)
      if result.stdout and not result.stdout.endswith("\n"):
         f.write("\n")
   if result.returncode != 0:
      tail = " | ".join(result.stdout.splitlines()[-6:])
      return {
         "kind": "amr",
         "case": case.name,
         "label": case.name,
         "status": "FAIL",
         "reason": tail,
         "returncode": result.returncode,
         "seconds": result.seconds,
         "dim": "2",
         "input": repo_relative(INPUT),
         "policy": CLOSED_POLICY.name,
         "ranks": ranks,
         "run_dir": repo_relative(run_dir),
      }
   if not run_input.exists():
      raise RuntimeError(f"solver did not write canonical run configuration: {run_input}")

   row = scalar_validation_row(
      run_dir,
      case=case.name,
      input_file=run_input,
      checks=CLOSED_POLICY,
      mhd=False,
      dim=2,
      ranks=ranks,
      base_input=repo_relative(INPUT),
      kind="amr",
   )
   row["returncode"] = result.returncode
   row["seconds"] = result.seconds
   row.update(final_level_summary(run_dir))
   return row


def base_cells(row: dict[str, object]) -> float:
   cells = float(row.get("total_cells", "nan"))
   level = int(float(row.get("active_finest_level", "nan")))
   return cells / (4.0 ** level) if level >= 0 else float("nan")


def check_case(case: TagCase, row: dict[str, object], base_cell_count: float) -> None:
   failures: list[str] = []
   if row.get("status") != "PASS":
      failures.append("run or scalar check failed")
   level = float(row.get("active_finest_level", "nan"))
   cells = float(row.get("total_cells", "nan"))
   ratio = cells / base_cell_count if base_cell_count > 0.0 else float("nan")
   budget = float(row.get("budget_fraction", "nan"))

   if level < case.min_level:
      failures.append(f"active_finest_level={level:g} < {case.min_level}")
   if ratio < case.min_cell_ratio:
      failures.append(f"cell ratio={ratio:.3g} < {case.min_cell_ratio:g}")
   if budget > case.max_budget_fraction:
      failures.append(f"budget fraction={budget:.3g} > {case.max_budget_fraction:g}")

   if failures:
      row["status"] = "FAIL"
      existing = str(row.get("reason", ""))
      row["reason"] = "; ".join(part for part in (existing, "; ".join(failures)) if part)
   row["description"] = f"cell_ratio_vs_L0={ratio:.6g}"


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int)
   args = parser.parse_args()

   if args.force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   rows = [run_case(case, args.ranks, args.timeout) for case in CASES]
   base_cell_count = base_cells(rows[1])
   for case, row in zip(CASES, rows, strict=True):
      check_case(case, row, base_cell_count)

   write_tsv(SUMMARY, rows, VALIDATION_HISTORY_FIELDS)
   append_validation_history(HISTORY, rows)

   for row in rows:
      reason = f" reason={row['reason']}" if row.get("reason") else ""
      print(f"{row['status']} {row['case']}{reason}")
   print(f"summary={repo_relative(SUMMARY)}")
   return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
   raise SystemExit(main())
