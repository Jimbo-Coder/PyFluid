#!/usr/bin/env python3
"""Scan SRMHD candidate-correction policies against exact Balsara solutions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scan_rmhd_balsara_reconstruction import (
   COMPONENT_NAMES,
   PLOT_COMPONENTS,
   exact_comparison,
)
from validate_rmhd import (
   EXACT_BALSARA_CASES,
   EXACT_BALSARA_ROOT,
   INPUTS,
   RUN_ROOT,
   exact_rmhd_conserved,
   read_last_exact_balsara_profile,
   relativistic_recovery_counts,
   rmhd_candidate_correction_counts,
   run_case,
)
from validation_common import (
   ROOT,
   read_scalar_rows,
   read_tsv,
   repo_relative,
   write_tsv,
)


METHODS = ("pwl", "wenoz", "mp5", "teno5", "ppm4", "ppmx", "ppm3")
METHOD_CHOICES = METHODS + (
   "parabolic",
   "wenoz-char",
   "mp5-char",
   "teno5-char",
   "parabolic-char",
   "ppm4-char",
   "ppmx-char",
   "ppm3-char",
)
GUARDS = ("off", "pwc", "pwl-pwc")
TIMESTEPS = ("ssprk3", "rk4")
CANDIDATE_RIEMANN_METHODS = ("same", "hll", "llf")


def artifact_paths(label: str) -> tuple[Path, Path, Path]:
   suffix = f"_{label}" if label else ""
   results = RUN_ROOT / f"balsara_candidate_guard_scan{suffix}.tsv"
   plot = (
      ROOT
      / "plots/current/scans"
      / f"rmhd_balsara_candidate_guard_convergence{suffix}.png"
   )
   best_policy_plot = (
      ROOT
      / "plots/current/scans"
      / f"rmhd_balsara_best_policy_convergence{suffix}.png"
   )
   return results, plot, best_policy_plot


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
               for timestep in args.timesteps:
                  for candidate_riemann in args.candidate_riemann_methods:
                     if guard == "off" and candidate_riemann != "same":
                        continue
                     overrides = {
                        "geometry.prob_lo": "-1.0",
                        "geometry.prob_hi": "1.0",
                        "amr.n_cell": str(cells),
                        "amr.max_grid_size": "128",
                        "pyfluid.gamma": repr(spec["gamma"]),
                        "pyfluid.final_time": repr(spec["time"]),
                        "pyfluid.max_step": "10000",
                        "pyfluid.ng": "2" if method == "pwl" else "3",
                        "pyfluid.cfl": "0.9",
                        "pyfluid.recon": method,
                        "pyfluid.timestep": timestep,
                        "pyfluid.rmhd_candidate_guard": guard,
                        "pyfluid.rmhd_candidate_riemann": candidate_riemann,
                        "pyfluid.rmhd_candidate_dmp_factor": repr(
                           args.dmp_factor
                        ),
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
                        f"guard_scan_balsara{case}_{method}_{guard}_"
                        f"{candidate_riemann}_{timestep}_"
                        f"dmp{args.dmp_factor:g}_n{cells}"
                     )
                     run_dir, scalar = run_case(
                        run_name,
                        INPUTS / "rmhd_balsara1_1d.in",
                        overrides,
                        args,
                     )
                     _, _, _, maximum_l1, overshoot, tv_ratio = (
                        exact_comparison(case, spec, run_dir)
                     )
                     fallback_count, reset_count = (
                        relativistic_recovery_counts(run_dir)
                     )
                     pwl_count, pwc_count = (
                        rmhd_candidate_correction_counts(run_dir)
                     )
                     scalar_rows = read_scalar_rows(run_dir)
                     rows.append({
                        "case": case,
                        "method": method,
                        "guard": guard,
                        "candidate_riemann": candidate_riemann,
                        "timestep": timestep,
                        "dmp_factor": args.dmp_factor,
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
                        f"Balsara {case} {method}/{guard}/"
                        f"{candidate_riemann}/{timestep} N={cells}: "
                        f"L1={maximum_l1:.6e} overshoot={overshoot:.6e} "
                        f"TV={tv_ratio:.4f} "
                        f"corrections={pwl_count}/{pwc_count} "
                        f"resets={reset_count}"
                     )
   return rows


def plot_scan(
   args: argparse.Namespace,
   rows: list[dict[str, object]],
   plot_path: Path,
) -> None:
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
      "parabolic": "#E69F00",
      "ppm4": "#56B4E9",
      "ppmx": "#000000",
      "ppm3": "#F0E442",
      "wenoz-char": "#CC79A7",
      "mp5-char": "#0072B2",
      "teno5-char": "#D55E00",
      "parabolic-char": "#E69F00",
      "ppm4-char": "#56B4E9",
      "ppmx-char": "#000000",
      "ppm3-char": "#F0E442",
   }
   styles = {"off": ":", "pwc": "--", "pwl-pwc": "-"}
   candidate_styles = {"same": "-", "hll": "-", "llf": "--"}
   markers = {"ssprk3": "o", "rk4": "s"}
   display_names = {
      "pwl": "PWL-MC",
      "wenoz": "WENO-Z",
      "mp5": "MP5",
      "teno5": "TENO5",
      "parabolic": "parabolic",
      "ppm4": "PPM4",
      "ppmx": "PPMX",
      "ppm3": "PPM3",
      "wenoz-char": "WENO-Z char",
      "mp5-char": "MP5 char",
      "teno5-char": "TENO5 char",
      "parabolic-char": "parabolic char",
      "ppm4-char": "PPM4 char",
      "ppmx-char": "PPMX char",
      "ppm3-char": "PPM3 char",
   }
   for index, case in enumerate(args.cases):
      axis = axes[index // columns, index % columns]
      for method in args.methods:
         for guard in args.guards:
            for timestep in args.timesteps:
               for candidate_riemann in args.candidate_riemann_methods:
                  selected = sorted(
                     (
                        row for row in rows
                        if int(row["case"]) == case
                        and row["method"] == method
                        and row["guard"] == guard
                        and row["timestep"] == timestep
                        and row["candidate_riemann"] == candidate_riemann
                     ),
                     key=lambda row: int(row["cells"]),
                  )
                  axis.loglog(
                     [int(row["cells"]) for row in selected],
                     [
                        float(row["maximum_normalized_component_l1"])
                        for row in selected
                     ],
                     marker=markers[timestep],
                     color=colors[method],
                     linestyle=(
                        candidate_styles[candidate_riemann]
                        if len(args.candidate_riemann_methods) > 1
                        else styles[guard]
                     ),
                     label=(
                        f"{display_names[method]} / {guard} / "
                        f"{candidate_riemann} / {timestep}"
                     ),
                  )
      axis.set_title(f"Balsara {case}")
      axis.set_xlabel("cells")
      axis.set_ylabel("maximum normalized component L1")
      axis.grid(which="both", alpha=0.22)
   for index in range(len(args.cases), num_rows * columns):
      axes[index // columns, index % columns].set_visible(False)
   axes[0, 0].legend(frameon=False, fontsize=7)
   correction_label = "/".join(
      method.upper() for method in args.candidate_riemann_methods
   )
   timestep_label = "/".join(method.upper() for method in args.timesteps)
   figure.suptitle(
      "SRMHD candidate correction — primary HLL, correction "
      f"{correction_label}, {timestep_label}, CFL 0.9"
   )
   plot_path.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(plot_path, dpi=180)
   plt.close(figure)


def plot_balsara4_profiles(
   args: argparse.Namespace,
   rows: list[dict[str, object]],
   plot_path: Path,
) -> bool:
   if 4 not in args.cases or not rows:
      return False
   cells = max(int(row["cells"]) for row in rows if int(row["case"]) == 4)
   spec = next(
      spec for spec in EXACT_BALSARA_CASES if int(spec["case"]) == 4
   )
   exact_profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / "solution_b4.dat"
   )
   exact_state = exact_rmhd_conserved(
      exact_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
   )
   figure, axes = plt.subplots(
      1,
      len(PLOT_COMPONENTS),
      figsize=(13.5, 3.2),
      constrained_layout=True,
   )
   for column, (name, label) in enumerate(PLOT_COMPONENTS):
      component = COMPONENT_NAMES.index(name)
      axes[column].plot(
         exact_profile[:, 0],
         exact_state[component],
         color="black",
         linewidth=1.8,
         label="exact",
      )
      axes[column].set_title(label)
      axes[column].set_xlabel("x")
      axes[column].grid(alpha=0.22)

   for row in rows:
      if int(row["case"]) != 4 or int(row["cells"]) != cells:
         continue
      run_dir = ROOT / str(row["run_dir"])
      centers, numerical, _, _, _, _ = exact_comparison(4, spec, run_dir)
      method = str(row["method"])
      label = (
         f"{method} / {row['guard']} / "
         f"{row['candidate_riemann']} / {row['timestep']}"
      )
      for column, (name, _) in enumerate(PLOT_COMPONENTS):
         component = COMPONENT_NAMES.index(name)
         axes[column].plot(
            centers,
            numerical[component],
            linewidth=1.1,
            label=label,
         )
   axes[0].set_ylabel(f"t = {float(spec['time']):g}")
   axes[0].legend(frameon=False, fontsize=7)
   figure.suptitle(
      f"Exact Balsara 4 profiles — N={cells}, CFL 0.9"
   )
   plot_path.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(plot_path, dpi=180)
   plt.close(figure)
   return True


def plot_best_policy(
   rows: list[dict[str, object]],
   plot_path: Path,
) -> None:
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
               and row["candidate_riemann"] == "same"
               and row["timestep"] == "ssprk3"
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
   plot_path.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(plot_path, dpi=180)
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
      choices=METHOD_CHOICES,
      help="reconstruction; repeatable (default: all)",
   )
   parser.add_argument(
      "--guard",
      dest="guards",
      action="append",
      choices=GUARDS,
      help="candidate correction; repeatable (default: all)",
   )
   parser.add_argument(
      "--timestep",
      dest="timesteps",
      action="append",
      choices=TIMESTEPS,
      help="timestepper; repeatable (default: ssprk3)",
   )
   parser.add_argument(
      "--candidate-riemann",
      dest="candidate_riemann_methods",
      action="append",
      choices=CANDIDATE_RIEMANN_METHODS,
      help="candidate-correction Riemann solver; repeatable (default: same)",
   )
   parser.add_argument(
      "--label",
      default="",
      help="artifact suffix for a focused scan",
   )
   parser.add_argument(
      "--dmp-factor",
      type=float,
      default=1.05,
      help="relaxed candidate DMP factor (default: 1.05)",
   )
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--force", action="store_true")
   parser.add_argument(
      "--plot-only",
      action="store_true",
      help="regenerate plots from the labeled TSV without running cases",
   )
   args = parser.parse_args()
   args.cases = sorted(set(args.cases or [3, 4]))
   args.cells = sorted(set(args.cells or [128, 256, 512]))
   args.methods = list(dict.fromkeys(args.methods or METHODS))
   args.guards = list(dict.fromkeys(args.guards or GUARDS))
   args.timesteps = list(dict.fromkeys(args.timesteps or ["ssprk3"]))
   args.candidate_riemann_methods = list(
      dict.fromkeys(args.candidate_riemann_methods or ["same"])
   )
   results_path, plot_path, best_policy_plot_path = artifact_paths(args.label)
   if args.plot_only:
      rows = read_tsv(results_path)
      for row in rows:
         row.setdefault("candidate_riemann", "same")
         row.setdefault("timestep", "ssprk3")
         # Legacy scan rows predate the explicit column and used the former
         # production default.
         row.setdefault("dmp_factor", 1.2)
   else:
      rows = run_scan(args)
      write_tsv(
         results_path,
         rows,
         [
            "case",
            "method",
            "guard",
            "candidate_riemann",
            "timestep",
            "dmp_factor",
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
   plot_scan(args, rows, plot_path)
   profile_plot_path = plot_path.with_name(
      plot_path.name.replace("_convergence", "_profiles")
   )
   wrote_profile_plot = plot_balsara4_profiles(
      args,
      rows,
      profile_plot_path,
   )
   wrote_best_policy_plot = any(
      row["candidate_riemann"] == "same"
      and row["timestep"] == "ssprk3"
      and row["method"] in {"pwl", "mp5"}
      for row in rows
   )
   if wrote_best_policy_plot:
      plot_best_policy(rows, best_policy_plot_path)
   print(f"wrote {repo_relative(results_path)}")
   print(f"wrote {repo_relative(plot_path)}")
   if wrote_profile_plot:
      print(f"wrote {repo_relative(profile_plot_path)}")
   if wrote_best_policy_plot:
      print(f"wrote {repo_relative(best_policy_plot_path)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
