#!/usr/bin/env python3
"""Scan SRMHD candidate-correction policies against exact Balsara solutions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scan_rmhd_balsara_reconstruction import exact_comparison
from validate_rmhd import (
   EXACT_BALSARA_CASES,
   INPUTS,
   RUN_ROOT,
   relativistic_recovery_counts,
   rmhd_candidate_correction_counts,
   run_case,
)
from validation_common import (
   ROOT,
   read_scalar_rows,
   repo_relative,
   write_tsv,
)


METHODS = ("pwl", "wenoz", "mp5", "teno5")
GUARDS = ("off", "pwc", "pwl-pwc")
RESULTS = RUN_ROOT / "balsara_candidate_guard_scan.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_balsara_candidate_guard_convergence.png"
BEST_POLICY_PLOT = (
   ROOT / "plots/current/scans/rmhd_balsara_best_policy_convergence.png"
)


def run_scan(args: argparse.Namespace) -> list[dict[str, object]]:
   specs = {
      int(spec["case"]): spec
      for spec in EXACT_BALSARA_CASES
      if int(spec["case"]) in args.cases
   }
   rows: list[dict[str, object]] = []
   for cells in args.cells:
      for case in args.cases:
         spec = specs[case]
         for method in args.methods:
            for guard in args.guards:
               overrides = {
                  "geometry.prob_lo": "-1.0",
                  "geometry.prob_hi": "1.0",
                  "amr.n_cell": str(cells),
                  "amr.max_grid_size": "128",
                  "pyfluid.gamma": repr(spec["gamma"]),
                  "pyfluid.final_time": repr(spec["time"]),
                  "pyfluid.max_step": "10000",
                  "pyfluid.ng": (
                     "3" if method in {"wenoz", "mp5", "teno5"} else "2"
                  ),
                  "pyfluid.cfl": "0.9",
                  "pyfluid.recon": method,
                  "pyfluid.rmhd_candidate_guard": guard,
                  "pyfluid.scalar_int": "64",
                  "pyfluid.plot_int": "1000000",
               }
               for side, state in (
                  ("left", spec["left"]),
                  ("right", spec["right"]),
               ):
                  for name, value in zip(
                     (
                        "rho",
                        "pressure",
                        "v1",
                        "v2",
                        "v3",
                        "b1",
                        "b2",
                        "b3",
                     ),
                     state,
                     strict=True,
                  ):
                     overrides[f"pyfluid.{name}_{side}"] = repr(value)
               run_name = (
                  f"guard_scan_balsara{case}_{method}_{guard}_n{cells}"
               )
               run_dir, scalar = run_case(
                  run_name,
                  INPUTS / "rmhd_balsara1_1d.in",
                  overrides,
                  args,
               )
               _, _, _, maximum_l1, overshoot, tv_ratio = exact_comparison(
                  case, spec, run_dir
               )
               fallback_count, reset_count = relativistic_recovery_counts(
                  run_dir
               )
               pwl_count, pwc_count = rmhd_candidate_correction_counts(
                  run_dir
               )
               scalar_rows = read_scalar_rows(run_dir)
               rows.append({
                  "case": case,
                  "method": method,
                  "guard": guard,
                  "cells": cells,
                  "maximum_normalized_component_l1": maximum_l1,
                  "maximum_normalized_global_overshoot": overshoot,
                  "maximum_component_tv_ratio": tv_ratio,
                  "c2p_fallbacks": fallback_count,
                  "atmosphere_resets": reset_count,
                  "rmhd_pwl_corrections": pwl_count,
                  "rmhd_pwc_corrections": pwc_count,
                  "mass_rel_drift": scalar["mass_rel_drift"],
                  "energy_rel_drift": scalar["energy_rel_drift"],
                  "mcells_per_s": scalar_rows[-1]["mcells_per_s"],
                  "run_dir": repo_relative(run_dir),
               })
               print(
                  f"Balsara {case} {method}/{guard} N={cells}: "
                  f"L1={maximum_l1:.6e} overshoot={overshoot:.6e} "
                  f"TV={tv_ratio:.4f} corrections={pwl_count}/{pwc_count} "
                  f"resets={reset_count}"
               )
   return rows


def plot_scan(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
   columns = min(3, len(args.cases))
   num_rows = math.ceil(len(args.cases) / columns)
   figure, axes = plt.subplots(
      num_rows,
      columns,
      figsize=(4.8 * columns, 3.8 * num_rows),
      constrained_layout=True,
      squeeze=False,
   )
   colors = {
      "pwl": "#009E73",
      "wenoz": "#CC79A7",
      "mp5": "#0072B2",
      "teno5": "#D55E00",
   }
   styles = {"off": ":", "pwc": "--", "pwl-pwc": "-"}
   display_names = {
      "pwl": "PWL-MC",
      "wenoz": "WENO-Z",
      "mp5": "MP5",
      "teno5": "TENO5",
   }
   for index, case in enumerate(args.cases):
      axis = axes[index // columns, index % columns]
      for method in args.methods:
         for guard in args.guards:
            selected = sorted(
               (
                  row for row in rows
                  if int(row["case"]) == case
                  and row["method"] == method
                  and row["guard"] == guard
               ),
               key=lambda row: int(row["cells"]),
            )
            axis.loglog(
               [int(row["cells"]) for row in selected],
               [
                  float(row["maximum_normalized_component_l1"])
                  for row in selected
               ],
               marker="o",
               color=colors[method],
               linestyle=styles[guard],
               label=f"{display_names[method]} / {guard}",
            )
      axis.set_title(f"Balsara {case}")
      axis.set_xlabel("cells")
      axis.set_ylabel("maximum normalized component L1")
      axis.grid(which="both", alpha=0.22)
   for index in range(len(args.cases), num_rows * columns):
      axes[index // columns, index % columns].set_visible(False)
   axes[0, 0].legend(frameon=False, fontsize=7)
   figure.suptitle(
      "SRMHD candidate correction — HLL/SSPRK3, CFL 0.9"
   )
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PLOT, dpi=180)
   plt.close(figure)


def plot_best_policy(rows: list[dict[str, object]]) -> None:
   selections = (
      ("pwl", "off", "PWL-MC", "#009E73", ":"),
      ("mp5", "off", "MP5", "#999999", ":"),
      ("mp5", "pwl-pwc", "MP5 + PWL-MC/PWC", "#D55E00", "--"),
      ("mp5", "pwc", "MP5 + PWC", "#0072B2", "-"),
   )
   figure, axes = plt.subplots(
      2,
      3,
      figsize=(13.5, 7.2),
      constrained_layout=True,
      squeeze=False,
   )
   for index, case in enumerate(range(1, 6)):
      axis = axes[index // 3, index % 3]
      for method, guard, label, color, style in selections:
         selected = sorted(
            (
               row for row in rows
               if int(row["case"]) == case
               and row["method"] == method
               and row["guard"] == guard
            ),
            key=lambda row: int(row["cells"]),
         )
         if not selected:
            continue
         axis.loglog(
            [int(row["cells"]) for row in selected],
            [
               float(row["maximum_normalized_component_l1"])
               for row in selected
            ],
            marker="o",
            color=color,
            linestyle=style,
            label=label,
         )
      axis.set_title(f"Balsara {case}")
      axis.set_xlabel("cells")
      axis.set_ylabel("maximum normalized component L1")
      axis.grid(which="both", alpha=0.22)
   axes[1, 2].set_visible(False)
   axes[0, 0].legend(frameon=False, fontsize=8)
   figure.suptitle(
      "Best SRMHD correction policies — HLL/SSPRK3, CFL 0.9"
   )
   BEST_POLICY_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(BEST_POLICY_PLOT, dpi=180)
   plt.close(figure)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--case",
      dest="cases",
      action="append",
      type=int,
      choices=tuple(range(1, 6)),
      help="Balsara case; repeatable (default: 3 and 4)",
   )
   parser.add_argument(
      "--cells",
      action="append",
      type=int,
      help="resolution; repeatable (default: 128, 256, 512)",
   )
   parser.add_argument(
      "--method",
      dest="methods",
      action="append",
      choices=METHODS,
      help="reconstruction; repeatable (default: all)",
   )
   parser.add_argument(
      "--guard",
      dest="guards",
      action="append",
      choices=GUARDS,
      help="candidate correction; repeatable (default: all)",
   )
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   args.cases = sorted(set(args.cases or [3, 4]))
   args.cells = sorted(set(args.cells or [128, 256, 512]))
   args.methods = list(dict.fromkeys(args.methods or METHODS))
   args.guards = list(dict.fromkeys(args.guards or GUARDS))
   rows = run_scan(args)
   write_tsv(
      RESULTS,
      rows,
      [
         "case",
         "method",
         "guard",
         "cells",
         "maximum_normalized_component_l1",
         "maximum_normalized_global_overshoot",
         "maximum_component_tv_ratio",
         "c2p_fallbacks",
         "atmosphere_resets",
         "rmhd_pwl_corrections",
         "rmhd_pwc_corrections",
         "mass_rel_drift",
         "energy_rel_drift",
         "mcells_per_s",
         "run_dir",
      ],
   )
   plot_scan(args, rows)
   plot_best_policy(rows)
   print(f"wrote {repo_relative(RESULTS)}")
   print(f"wrote {repo_relative(PLOT)}")
   print(f"wrote {repo_relative(BEST_POLICY_PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
