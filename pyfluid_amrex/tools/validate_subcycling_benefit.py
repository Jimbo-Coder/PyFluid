#!/usr/bin/env python3
"""Validate that AMR subcycling preserves the solution while improving speed."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from validation_common import mpi_prefix
from validation_common import append_performance_history, write_tsv
from plotfile_reader import latest_plotfile


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build/pyfluid_amrex/pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex/inputs/kelvin_helmholtz_mhd_2d_movie_ct.in"
DEEP_AUTO_INPUT = ROOT / "pyfluid_amrex/inputs/kelvin_helmholtz_mhd_2d_single_sheet_amr_ct.in"
RUN_ROOT = ROOT / "runs/current/subcycling_benefit"
PLOT_ROOT = ROOT / "plots/current/scans"
RESULTS = RUN_ROOT / "subcycling_benefit.tsv"
PERFORMANCE_HISTORY = ROOT / "runs/current/performance_history.tsv"

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")


@dataclass(frozen=True)
class SubcyclingCase:
   name: str
   label: str
   overrides: tuple[str, ...]


CASES = (
   SubcyclingCase(
      name="nosub",
      label="AMR no subcycling",
      overrides=("pyfluid.subcycling=off",),
   ),
   SubcyclingCase(
      name="spatial",
      label="AMR spatial subcycling",
      overrides=("pyfluid.subcycling=on", "pyfluid.subcycling_steps=spatial"),
   ),
)

LOCAL_CFL_CASE = SubcyclingCase(
   name="local_cfl",
   label="AMR local-CFL subcycling",
   overrides=("pyfluid.subcycling=on", "pyfluid.subcycling_steps=local_cfl"),
)


COMMON_OVERRIDES = (
   "pyfluid.final_time=0.03",
   "pyfluid.max_step=100000",
   "pyfluid.scalar_int=8",
   "pyfluid.plot_int=100000",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.aux_gf_int=-1",
)

DEEP_AUTO_OVERRIDES = (
   "pyfluid.final_time=0.015",
   "pyfluid.max_step=100000",
   "pyfluid.scalar_int=4",
   "pyfluid.plot_int=100000",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.aux_gf_int=-1",
   "amr.n_cell=24 24",
   "amr.max_level=6",
   "amr.ref_ratio=2 2 2 2 2 2",
   "amr.regrid_int=2",
   "amr.n_error_buf=4 4 4 4 4 4",
   "amr.max_grid_size=32",
   "amr.blocking_factor=8",
   "amr.level_cell_budget_side=1024",
   "amr.total_cell_budget=3000000",
   "amr.initial_refine=none",
   "amr.initial_refine_levels=0",
   "amr.initial_refine_fixed_levels=0",
   "amr.tag_metric=auto",
   "amr.tag_threshold=0.035",
)


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def parse_float(text: str) -> float:
   try:
      return float(text)
   except (TypeError, ValueError):
      return float("nan")


def read_last_tsv_row(path: Path) -> dict[str, str]:
   with path.open(newline="") as f:
      rows = list(csv.DictReader(f, delimiter="\t"))
   if not rows:
      raise RuntimeError(f"{path} has no data rows")
   return rows[-1]


def step_count(row: dict[str, str], level: int) -> float:
   return parse_float(row.get(f"steps_L{level}", "nan"))


def run_input(args: argparse.Namespace) -> Path:
   return DEEP_AUTO_INPUT if args.deep_auto_amr else DEFAULT_INPUT


def run_root(args: argparse.Namespace) -> Path:
   return RUN_ROOT / "deep_auto" if args.deep_auto_amr else RUN_ROOT


def results_path(args: argparse.Namespace) -> Path:
   return run_root(args) / "subcycling_benefit.tsv" if args.deep_auto_amr else RESULTS


def common_overrides(args: argparse.Namespace) -> tuple[str, ...]:
   return DEEP_AUTO_OVERRIDES if args.deep_auto_amr else COMMON_OVERRIDES


def subcycling_cases(args: argparse.Namespace) -> tuple[SubcyclingCase, ...]:
   return (*CASES, LOCAL_CFL_CASE) if args.deep_auto_amr else CASES


def candidate_case_name(args: argparse.Namespace) -> str:
   return "local_cfl" if args.deep_auto_amr else "spatial"


def plot_stem(args: argparse.Namespace) -> str:
   return "kh_mhd_deep_auto_subcycling" if args.deep_auto_amr else "kh_mhd_subcycling"


def run_label(args: argparse.Namespace) -> str:
   if args.deep_auto_amr:
      return "KH MHD auto-AMR deep subcycling benefit, WENO-Z/HLLD/SSPRK3/CT, CFL=0.9"
   return "KH MHD AMR subcycling benefit, WENO-Z/HLLD/SSPRK3/CT, CFL=0.9"


def run_case(case: SubcyclingCase, args: argparse.Namespace) -> dict[str, object]:
   output_dir = run_root(args) / case.name
   if args.force and output_dir.exists():
      shutil.rmtree(output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)
   command = [
      *mpi_prefix(args.ranks),
      str(EXE),
      str(run_input(args)),
      f"pyfluid.output_dir={repo_relative(output_dir)}",
      *common_overrides(args),
      *case.overrides,
   ]
   completed = subprocess.run(
      command,
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
      check=False,
   )
   (output_dir / "run.log").write_text(completed.stdout)
   if completed.returncode != 0:
      raise RuntimeError(f"{case.name} failed; see {output_dir / 'run.log'}")

   scalar = read_last_tsv_row(output_dir / "scalars.tsv")
   levels = read_last_tsv_row(output_dir / "amr_levels.tsv")
   finest = int(parse_float(levels.get("active_finest_level", levels.get("max_level", "0"))))
   coarse_steps = step_count(levels, 0)
   fine_steps = step_count(levels, finest)
   sim_time_match = SIM_TIME_RE.search(completed.stdout)
   average_match = AVERAGE_RE.search(completed.stdout)
   return {
      "case": case.name,
      "label": case.label,
      "status": "PASS",
      "ranks": args.ranks,
      "mcells_s": parse_float(average_match.group(1)) if average_match else float("nan"),
      "sim_time_per_s": parse_float(sim_time_match.group(1)) if sim_time_match else float("nan"),
      "final_time": parse_float(scalar.get("time", "nan")),
      "energy_rel_drift": parse_float(scalar.get("energy_rel_drift", "nan")),
      "divb_l1": parse_float(scalar.get("divb_l1", "nan")),
      "active_finest_level": finest,
      "coarse_steps": coarse_steps,
      "fine_steps": fine_steps,
      "fine_per_coarse_steps": fine_steps / coarse_steps if coarse_steps > 0.0 else float("nan"),
      "plotfile": repo_relative(latest_plotfile(output_dir, require=True)),
      "run_dir": repo_relative(output_dir),
   }


def parse_comparison_output(text: str, component: str) -> dict[str, float]:
   parsed: dict[str, float] = {}
   for token in text.split():
      if "=" not in token:
         continue
      key, value = token.split("=", 1)
      parsed[f"{component}_{key}"] = parse_float(value)
   return parsed


def compare_component(
   reference: Path,
   candidate: Path,
   component: str,
   output: Path,
   title: str,
) -> dict[str, float]:
   completed = subprocess.run(
      [
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/compare_amrex_plotfiles.py",
         "--reference",
         repo_relative(reference),
         "--candidate",
         repo_relative(candidate),
         "--component",
         component,
         "--output",
         repo_relative(output),
         "--title",
         title,
      ],
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=True,
   )
   return parse_comparison_output(completed.stdout, component)


def write_plot(rows: list[dict[str, object]], output: Path, title: str) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   labels = [str(row["label"]) for row in rows]
   sim_rates = [float(row["sim_time_per_s"]) for row in rows]
   ratios = [float(row["fine_per_coarse_steps"]) for row in rows]
   fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
   colors = ["0.45", "tab:blue", "tab:green"][: len(rows)]
   axs[0].bar(labels, sim_rates, color=colors)
   axs[0].set_ylabel("simulation time / wall second")
   axs[0].set_title("Performance")
   axs[0].tick_params(axis="x", rotation=12)
   axs[0].grid(True, axis="y", alpha=0.25)

   axs[1].bar(labels, ratios, color=colors)
   axs[1].set_ylabel("finest-level steps / coarse steps")
   axs[1].set_title("Level time stepping")
   axs[1].tick_params(axis="x", rotation=12)
   axs[1].grid(True, axis="y", alpha=0.25)
   fig.suptitle(title)
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=170)
   plt.close(fig)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument(
      "--deep-auto-amr",
      action="store_true",
      help="Use the slower pure-auto AMR probe that reaches many refinement levels.",
   )
   parser.add_argument("--min-speedup", type=float)
   parser.add_argument("--max-pressure-l2", type=float)
   parser.add_argument("--max-rho-l2", type=float)
   parser.add_argument("--min-active-finest-level", type=int)
   args = parser.parse_args()
   if args.min_speedup is None:
      args.min_speedup = 1.20 if args.deep_auto_amr else 1.10
   if args.max_pressure_l2 is None:
      args.max_pressure_l2 = 1.0e-4 if args.deep_auto_amr else 5.0e-4
   if args.max_rho_l2 is None:
      args.max_rho_l2 = 1.0e-4 if args.deep_auto_amr else 5.0e-4
   if args.min_active_finest_level is None:
      args.min_active_finest_level = 6 if args.deep_auto_amr else 3

   rows = [run_case(case, args) for case in subcycling_cases(args)]
   by_case = {str(row["case"]): row for row in rows}
   nosub = by_case["nosub"]
   candidate_name = candidate_case_name(args)
   candidate = by_case[candidate_name]

   comparison_output = PLOT_ROOT / "kh_mhd_subcycling_vs_nosub_001.png"
   comparison = {}
   for component in ("pressure", "rho"):
      comparison.update(
         compare_component(
            ROOT / str(nosub["plotfile"]),
            ROOT / str(candidate["plotfile"]),
            component,
            comparison_output.with_name(f"{plot_stem(args)}_{candidate_name}_vs_nosub_{component}_001.png"),
            f"{run_label(args)}: {candidate_name} subcycling vs no subcycling",
         )
      )

   speedup = float(candidate["sim_time_per_s"]) / float(nosub["sim_time_per_s"])
   failures = []
   if int(candidate["active_finest_level"]) < args.min_active_finest_level:
      failures.append(
         f"active_finest_level={candidate['active_finest_level']}<{args.min_active_finest_level}"
      )
   if float(nosub["fine_per_coarse_steps"]) > 1.1:
      failures.append(f"no-sub fine/coarse steps={nosub['fine_per_coarse_steps']:.3g} should be near 1")
   if candidate_name == "spatial" and float(candidate["fine_per_coarse_steps"]) < 2.0 ** args.min_active_finest_level:
      failures.append(f"subcycling fine/coarse steps={candidate['fine_per_coarse_steps']:.3g} too small")
   if candidate_name != "spatial" and float(candidate["fine_per_coarse_steps"]) <= 1.1:
      failures.append(f"subcycling fine/coarse steps={candidate['fine_per_coarse_steps']:.3g} too small")
   if speedup < args.min_speedup:
      failures.append(f"speedup={speedup:.3g}<{args.min_speedup:.3g}")
   if comparison["pressure_rel_l2"] > args.max_pressure_l2:
      failures.append(f"pressure_rel_l2={comparison['pressure_rel_l2']:.3e}>{args.max_pressure_l2:.3e}")
   if comparison["rho_rel_l2"] > args.max_rho_l2:
      failures.append(f"rho_rel_l2={comparison['rho_rel_l2']:.3e}>{args.max_rho_l2:.3e}")

   summary = {
      "case": "subcycling_deep_auto_benefit" if args.deep_auto_amr else "subcycling_benefit",
      "label": run_label(args),
      "status": "PASS" if not failures else "FAIL",
      "ranks": args.ranks,
      "mcells_s": candidate["mcells_s"],
      "sim_time_per_s": candidate["sim_time_per_s"],
      "baseline_mcells_s": nosub["mcells_s"],
      "ratio": speedup,
      "warn_fraction": args.min_speedup,
      "returncode": 0 if not failures else 1,
      "speedup_vs_1rank": "",
      "efficiency_vs_1rank": "",
      "sim_time_speedup_vs_1rank": speedup,
      "sim_time_efficiency_vs_1rank": "",
      "dominant_stage": "",
      "dominant_stage_fraction": "",
      "timestep_fraction": "",
      "flux_fraction": "",
      "run_dir": repo_relative(run_root(args)),
   }
   for row in rows:
      if row["case"] == candidate_name:
         row["speedup_vs_nosub"] = speedup
         row.update(comparison)
      elif row["case"] == "nosub":
         row["speedup_vs_nosub"] = 1.0
      else:
         row["speedup_vs_nosub"] = float(row["sim_time_per_s"]) / float(nosub["sim_time_per_s"])
      row["overall_status"] = summary["status"]
      row["reason"] = "; ".join(failures)
   fields = [
      "case",
      "label",
      "overall_status",
      "reason",
      "ranks",
      "mcells_s",
      "sim_time_per_s",
      "speedup_vs_nosub",
      "final_time",
      "energy_rel_drift",
      "divb_l1",
      "active_finest_level",
      "coarse_steps",
      "fine_steps",
      "fine_per_coarse_steps",
      "pressure_rel_l2",
      "rho_rel_l2",
      "plotfile",
      "run_dir",
   ]
   results = results_path(args)
   write_tsv(results, rows, fields)
   append_performance_history(PERFORMANCE_HISTORY, [{"kind": "subcycling_benefit", **summary}])
   plot_path = PLOT_ROOT / f"{plot_stem(args)}_benefit_001.png"
   write_plot(rows, plot_path, run_label(args))

   print(
      f"{summary['status']} subcycling_benefit: speedup={speedup:.3f}, "
      f"pressure_l2={comparison['pressure_rel_l2']:.3e}, "
      f"rho_l2={comparison['rho_rel_l2']:.3e}, "
      f"mode={candidate_name}, "
      f"active_finest_level={candidate['active_finest_level']}, "
      f"fine/coarse steps={candidate['fine_per_coarse_steps']:.1f}"
   )
   print(f"wrote {repo_relative(results)}")
   print(f"wrote {repo_relative(plot_path)}")
   if failures:
      for failure in failures:
         print(f"FAIL: {failure}")
      return 1
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
