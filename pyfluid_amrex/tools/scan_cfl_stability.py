#!/usr/bin/env python3
"""Empirically scan CFL stability for representative AMReX solver branches."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation_common import append_validation_history, run_mpi


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "pyfluid_amrex"
INPUTS = ROOT / "pyfluid_amrex" / "inputs"
DEFAULT_RUN_ROOT = ROOT / "runs" / "current" / "cfl_stability"
DEFAULT_PLOT_DIR = ROOT / "plots" / "current" / "scans"
DEFAULT_HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"
DEFAULT_CFL_VALUES = (0.80, 0.90, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20)


@dataclass(frozen=True)
class ScanCase:
   name: str
   label: str
   suite: str
   dim: int
   base_input: str
   overrides: dict[str, str] = field(default_factory=dict)
   description: str = ""


def hydro_timestep_cases() -> list[ScanCase]:
   cases = []
   for timestep in ("euler", "corrector", "midpoint", "ssprk3", "rk4"):
      cases.append(ScanCase(
         name=f"hydro_sine_pwl_hllc_{timestep}",
         label=f"HD sine PWL/HLLC/{timestep}",
         suite="timesteppers",
         dim=2,
         base_input="hydro_sine_2d.in",
         overrides={
            "amr.n_cell": "96 96",
            "amr.max_level": "0",
            "pyfluid.max_step": "64",
            "pyfluid.final_time": "-1.0",
            "pyfluid.ng": "2",
            "pyfluid.recon": "pwl",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": timestep,
            "pyfluid.diffusion": "none",
         },
         description="smooth hydro branch for timestep CFL behavior",
      ))
   return cases


def diffusion_stability_cases() -> list[ScanCase]:
   hydro_common = {
      "amr.n_cell": "96 96",
      "amr.max_level": "0",
      "pyfluid.max_step": "48",
      "pyfluid.final_time": "-1.0",
      "pyfluid.ng": "2",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.viscosity": "1.0e-4",
   }
   mhd_common = {
      "amr.n_cell": "64 64",
      "amr.max_level": "0",
      "pyfluid.max_step": "48",
      "pyfluid.final_time": "-1.0",
      "pyfluid.ng": "2",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hlld",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.viscosity": "0.0",
      "pyfluid.resistivity": "1.0e-3",
      "pyfluid.ct": "on",
   }
   return [
      ScanCase(
         "hydro_diffusion_explicit",
         "HD explicit viscosity",
         "diffusion",
         2,
         "hydro_sine_2d.in",
         {**hydro_common, "pyfluid.diffusion": "explicit"},
         "explicit viscous diffusion; dt should include advective and diffusion limits",
      ),
      ScanCase(
         "hydro_diffusion_implicit",
         "HD implicit viscosity",
         "diffusion",
         2,
         "hydro_sine_2d.in",
         {
            **hydro_common,
            "pyfluid.diffusion": "implicit",
            "pyfluid.diffusion_max_iterations": "500",
         },
         "implicit viscous diffusion with same flow as explicit case",
      ),
      ScanCase(
         "mhd_resistivity_explicit_ct",
         "MHD explicit resistivity/CT",
         "diffusion",
         2,
         "resistive_wave_x_2d_ct.in",
         {**mhd_common, "pyfluid.diffusion": "explicit"},
         "explicit resistive CT induction update",
      ),
   ]


def subcycling_cases() -> list[ScanCase]:
   common = {
      "amr.n_cell": "48 48",
      "amr.max_level": "2",
      "amr.ref_ratio": "2 2",
      "amr.regrid_int": "2",
      "amr.tag_metric": "lohner",
      "amr.state_interp": "quartic-conservative",
      "pyfluid.max_step": "32",
      "pyfluid.final_time": "-1.0",
      "pyfluid.ng": "2",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.diffusion": "none",
   }
   return [
      ScanCase(
         "hydro_amr_no_subcycling",
         "HD AMR no subcycling",
         "subcycling",
         2,
         "pressure_bump_2d_amr.in",
         {**common, "pyfluid.subcycling": "off"},
         "AMR pressure bump using global timestep on all levels",
      ),
      ScanCase(
         "hydro_amr_spatial_subcycling",
         "HD AMR spatial subcycling",
         "subcycling",
         2,
         "pressure_bump_2d_amr.in",
         {**common, "pyfluid.subcycling": "on", "pyfluid.subcycling_steps": "spatial"},
         "AMR pressure bump using refinement-ratio subcycling",
      ),
      ScanCase(
         "hydro_amr_local_cfl_subcycling",
         "HD AMR local-CFL subcycling",
         "subcycling",
         2,
         "pressure_bump_2d_amr.in",
         {**common, "pyfluid.subcycling": "on", "pyfluid.subcycling_steps": "local_cfl"},
         "AMR pressure bump using local-CFL subcycling",
      ),
   ]


def amr_metric_cases() -> list[ScanCase]:
   common = {
      "amr.n_cell": "48 48",
      "amr.max_level": "2",
      "amr.ref_ratio": "2 2",
      "amr.regrid_int": "2",
      "amr.state_interp": "quartic-conservative",
      "pyfluid.max_step": "32",
      "pyfluid.final_time": "-1.0",
      "pyfluid.ng": "2",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.subcycling": "on",
      "pyfluid.subcycling_steps": "spatial",
      "pyfluid.diffusion": "none",
   }
   return [
      ScanCase(
         "hydro_amr_metric_none",
         "HD AMR metric=none",
         "amr_metrics",
         2,
         "pressure_bump_2d_amr.in",
         {**common, "amr.tag_metric": "none"},
         "no automatic flow tagging; checks the AMR driver without refinement metric",
      ),
      ScanCase(
         "hydro_amr_metric_auto",
         "HD AMR metric=auto",
         "amr_metrics",
         2,
         "pressure_bump_2d_amr.in",
         {**common, "amr.tag_metric": "lohner", "amr.tag_threshold": "0.03"},
         "default jack-of-all-trades AMR metric",
      ),
      ScanCase(
         "hydro_amr_metric_auto_tight",
         "HD AMR metric=auto tight",
         "amr_metrics",
         2,
         "pressure_bump_2d_amr.in",
         {**common, "amr.tag_metric": "lohner", "amr.tag_threshold": "0.01"},
         "tighter auto metric threshold to catch refinement-driven stability issues",
      ),
   ]


def ct_cases() -> list[ScanCase]:
   common = {
      "amr.n_cell": "64 64",
      "amr.max_level": "0",
      "pyfluid.max_step": "64",
      "pyfluid.final_time": "-1.0",
      "pyfluid.ng": "2",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hlld",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.diffusion": "none",
      "pyfluid.resistivity": "0.0",
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_compute_int": "-1",
      "pyfluid.aux_gf_int": "-1",
   }
   return [
      ScanCase(
         "mhd_wave_ct_on",
         "MHD wave CT on",
         "ct",
         2,
         "resistive_wave_x_2d_ct.in",
         {**common, "pyfluid.ct": "on"},
         "ideal-MHD wave with constrained transport",
      ),
      ScanCase(
         "mhd_wave_ct_off",
         "MHD wave CT off",
         "ct",
         2,
         "resistive_wave_x_2d_ct.in",
         {**common, "pyfluid.ct": "off"},
         "same MHD wave with CT disabled; useful for divb/stability comparison",
      ),
   ]


def nonlinear_shock_cases() -> list[ScanCase]:
   common = {
      "amr.n_cell": "512",
      "amr.max_level": "0",
      "pyfluid.max_step": "10000",
      "pyfluid.final_time": "0.20",
      "pyfluid.ng": "2",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.diffusion": "none",
      "pyfluid.scalar_int": "8",
   }
   cases = [
      ScanCase(
         name=f"hydro_sod_pwl_hllc_{timestep}",
         label=f"HD Sod PWL/HLLC/{timestep}",
         suite="nonlinear",
         dim=1,
         base_input="sod_x_1d.in",
         overrides={**common, "pyfluid.timestep": timestep},
         description="nonlinear Sod shock/contact CFL behavior",
      )
      for timestep in ("euler", "corrector", "midpoint", "ssprk3", "rk4")
   ]
   cases.append(ScanCase(
      name="hydro_sod_strong100_pwl_hllc_ssprk3",
      label="HD Sod 100:1 PWL/HLLC/SSPRK3",
      suite="nonlinear",
      dim=1,
      base_input="sod_x_1d.in",
      overrides={
         **common,
         "pyfluid.timestep": "ssprk3",
         "pyfluid.rho_left": "100.0",
         "pyfluid.pressure_left": "100.0",
         "pyfluid.rho_right": "1.0",
         "pyfluid.pressure_right": "1.0",
      },
      description="stronger 100:1 Sod shock to stress positivity and nonlinear CFL",
   ))
   return cases


BASELINE_CASES = (
   ScanCase(
      name="hydro_sine_pwc_llf_euler",
      label="HD sine PWC/LLF/Euler",
      suite="baseline",
      dim=2,
      base_input="hydro_sine_2d.in",
      overrides={
         "amr.n_cell": "128 128",
         "pyfluid.max_step": "96",
         "pyfluid.final_time": "-1.0",
         "pyfluid.ng": "1",
         "pyfluid.recon": "pwc",
         "pyfluid.riemann": "llf",
         "pyfluid.timestep": "euler",
      },
      description="baseline first-order hydro branch",
   ),
   ScanCase(
      name="hydro_sod_pwl_hllc_ssprk3",
      label="HD Sod PWL/HLLC/SSPRK3",
      suite="baseline",
      dim=1,
      base_input="sod_x_1d.in",
      overrides={
         "amr.n_cell": "512",
         "pyfluid.max_step": "10000",
         "pyfluid.final_time": "0.20",
         "pyfluid.ng": "2",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.timestep": "ssprk3",
      },
      description="shock/contact hydro branch",
   ),
   ScanCase(
      name="hydro_sine_wenoz_hllc_rk4",
      label="HD sine WENO-Z/HLLC/RK4",
      suite="baseline",
      dim=2,
      base_input="hydro_sine_2d.in",
      overrides={
         "amr.n_cell": "128 128",
         "pyfluid.max_step": "96",
         "pyfluid.final_time": "-1.0",
         "pyfluid.ng": "3",
         "pyfluid.recon": "wenoz",
         "pyfluid.riemann": "hllc",
         "pyfluid.timestep": "rk4",
      },
      description="high-order smooth hydro branch; may tolerate CFL above one",
   ),
   ScanCase(
      name="mhd_wave_pwl_hlld_ssprk3_ct",
      label="MHD wave PWL/HLLD/SSPRK3/CT",
      suite="baseline",
      dim=2,
      base_input="resistive_wave_x_2d_ct.in",
      overrides={
         "amr.n_cell": "64 64",
         "pyfluid.max_step": "96",
         "pyfluid.final_time": "-1.0",
         "pyfluid.ng": "2",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.diffusion": "none",
         "pyfluid.resistivity": "0.0",
         "pyfluid.aux_gfs": "none",
         "pyfluid.aux_gf_compute_int": "-1",
         "pyfluid.aux_gf_int": "-1",
      },
      description="ideal-MHD CT branch",
   ),
   ScanCase(
      name="hydro_amr_localcfl_pwc_hllc_ssprk3",
      label="HD AMR local-CFL PWC/HLLC/SSPRK3",
      suite="baseline",
      dim=2,
      base_input="hydro_beam_obstacle_2d_amr.in",
      overrides={
         "pyfluid.max_step": "48",
         "pyfluid.final_time": "-1.0",
         "pyfluid.scalar_int": "4",
         "pyfluid.plot_int": "-1",
         "pyfluid.checkpoint_int": "-1",
         "pyfluid.aux_gfs": "none",
         "pyfluid.aux_gf_compute_int": "-1",
         "pyfluid.aux_gf_int": "-1",
         "pyfluid.subcycling": "on",
         "pyfluid.subcycling_steps": "local_cfl",
      },
      description="AMR obstacle branch with stable local-CFL subcycling",
   ),
)


def suite_cases() -> dict[str, list[ScanCase]]:
   return {
      "baseline": list(BASELINE_CASES),
      "timesteppers": hydro_timestep_cases(),
      "diffusion": diffusion_stability_cases(),
      "subcycling": subcycling_cases(),
      "amr_metrics": amr_metric_cases(),
      "ct": ct_cases(),
      "nonlinear": nonlinear_shock_cases(),
   }


CASES = tuple(case for cases in suite_cases().values() for case in cases)


def exe_for_dim(dim: int) -> Path:
   return BUILD / f"pyfluid_amrex_{dim}d"


def next_output_path(directory: Path, stem: str) -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   existing = sorted(directory.glob(f"{stem}_*.png"))
   if not existing:
      return directory / f"{stem}_001.png"
   try:
      index = int(existing[-1].stem.rsplit("_", 1)[-1]) + 1
   except ValueError:
      index = 1
   return directory / f"{stem}_{index:03d}.png"


def case_by_name() -> dict[str, ScanCase]:
   return {case.name: case for case in CASES}


def selected_cases(suites: list[str] | None, names: str) -> list[ScanCase]:
   by_suite = suite_cases()
   if not suites or "all" in suites:
      cases = list(CASES)
   else:
      cases = []
      for suite in suites:
         cases.extend(by_suite[suite])

   if names == "all":
      return cases

   all_by_name = case_by_name()
   wanted = [raw.strip() for raw in names.split(",") if raw.strip()]
   selected = []
   for name in wanted:
      if name not in all_by_name:
         raise ValueError(f"unknown case {name}; choices: {', '.join(all_by_name)}")
      selected.append(all_by_name[name])
   selected_names = {case.name for case in cases}
   filtered = sorted(name for name in wanted if name not in selected_names)
   if filtered:
      raise ValueError(f"case(s) not in selected suite(s): {', '.join(filtered)}")
   return selected


def parse_cfl_values(text: str) -> list[float]:
   values = [float(raw.strip()) for raw in text.split(",") if raw.strip()]
   if not values:
      raise ValueError("at least one CFL value is required")
   if any(value <= 0.0 for value in values):
      raise ValueError("CFL values must be positive")
   return values


def write_case_input(case: ScanCase, cfl: float, run_root: Path, force: bool) -> tuple[Path, Path]:
   tag = f"{case.name}_cfl{cfl:.4g}".replace(".", "p")
   output_dir = run_root / tag
   if force and output_dir.exists():
      shutil.rmtree(output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)
   input_dir = run_root / "inputs"
   input_dir.mkdir(parents=True, exist_ok=True)

   overrides = {
      "pyfluid.run_name": tag,
      "pyfluid.output_dir": str(output_dir.relative_to(ROOT)),
      "pyfluid.cfl": f"{cfl:.17g}",
      "pyfluid.scalar_int": "4",
      "pyfluid.plot_int": "-1",
      "pyfluid.checkpoint_int": "-1",
   }
   overrides.update(case.overrides)
   lines = (INPUTS / case.base_input).read_text().splitlines()
   lines.append("")
   lines.append("# scan_cfl_stability.py overrides")
   for key, value in overrides.items():
      lines.append(f"{key} = {value}")
   input_path = input_dir / f"{tag}.in"
   input_path.write_text("\n".join(lines) + "\n")
   return input_path, output_dir


def scalar_rows(output_dir: Path) -> list[dict[str, str]]:
   scalar_file = output_dir / "scalars.tsv"
   if not scalar_file.exists():
      return []
   with scalar_file.open(newline="") as f:
      return list(csv.DictReader(f, delimiter="\t"))


def rows_are_finite(rows: list[dict[str, str]]) -> bool:
   if not rows:
      return False
   checked = ("mass", "energy", "min_rho", "min_pressure", "mcells_per_s", "sim_time_per_s")
   for row in rows:
      for key in checked:
         try:
            value = float(row.get(key) if row.get(key) is not None else "nan")
         except (TypeError, ValueError):
            return False
         if not math.isfinite(value):
            return False
      if float(row["min_rho"]) <= 0.0 or float(row["min_pressure"]) <= 0.0:
         return False
   return True


def average_positive(rows: list[dict[str, str]], key: str) -> str:
   values = []
   for row in rows:
      try:
         value = float(row.get(key) if row.get(key) is not None else "nan")
      except (TypeError, ValueError):
         continue
      if math.isfinite(value) and value > 0.0:
         values.append(value)
   return "" if not values else f"{sum(values) / len(values):.12g}"


def run_case(case: ScanCase, cfl: float, run_root: Path, ranks: int, timeout: int, force: bool) -> dict[str, str]:
   exe = exe_for_dim(case.dim)
   input_path, output_dir = write_case_input(case, cfl, run_root, force)
   if not exe.exists():
      return {
         "case": case.name,
         "label": case.label,
         "cfl": f"{cfl:.8g}",
         "status": "FAIL",
         "reason": f"missing executable {exe.relative_to(ROOT)}",
      }
   proc = run_mpi(
      exe,
      [input_path],
      ranks=ranks,
      timeout=timeout,
   )
   rows = scalar_rows(output_dir)
   ok = proc.returncode == 0 and rows_are_finite(rows)
   reason = "" if ok else "\n".join(proc.stdout.splitlines()[-5:]).replace("\t", " ").replace("\x00", "")
   last = rows[-1] if rows else {}
   return {
      "case": case.name,
      "label": case.label,
      "suite": case.suite,
      "dim": str(case.dim),
      "input": case.base_input,
      "cfl": f"{cfl:.8g}",
      "status": "PASS" if ok else "FAIL",
      "reason": reason,
      "steps": last.get("step", ""),
      "time": last.get("time", ""),
      "min_rho": last.get("min_rho", ""),
      "min_pressure": last.get("min_pressure", ""),
      "mass_rel_drift": last.get("mass_rel_drift", ""),
      "energy_rel_drift": last.get("energy_rel_drift", ""),
      "divb_l1": last.get("divb_l1", ""),
      "avg_mcells_per_s": average_positive(rows, "mcells_per_s"),
      "avg_sim_time_per_s": average_positive(rows, "sim_time_per_s"),
      "description": case.description,
   }


def write_results(rows: list[dict[str, str]], path: Path) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   fields = [
      "case",
      "label",
      "suite",
      "dim",
      "input",
      "cfl",
      "status",
      "reason",
      "steps",
      "time",
      "min_rho",
      "min_pressure",
      "mass_rel_drift",
      "energy_rel_drift",
      "divb_l1",
      "avg_mcells_per_s",
      "avg_sim_time_per_s",
      "description",
   ]
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, str]]:
   with path.open(newline="") as f:
      return list(csv.DictReader(f, delimiter="\t"))


def plot_results(rows: list[dict[str, str]], output: Path) -> None:
   labels = []
   grouped: dict[str, list[dict[str, str]]] = {}
   for row in rows:
      case = row["case"]
      grouped.setdefault(case, []).append(row)
      if row["label"] not in labels:
         labels.append(row["label"])

   fig, ax = plt.subplots(figsize=(11.5, max(4.0, 0.55 * len(grouped) + 1.5)), constrained_layout=True)
   cmap = {"PASS": "#2ca02c", "FAIL": "#d62728"}
   y_ticks = []
   y_labels = []
   x_values = [float(row["cfl"]) for row in rows]
   x_text = max(x_values) + 0.04 * (max(x_values) - min(x_values))
   for y, (case, case_rows) in enumerate(grouped.items()):
      case_rows = sorted(case_rows, key=lambda row: float(row["cfl"]))
      x = np.array([float(row["cfl"]) for row in case_rows])
      stable = np.array([1 if row["status"] == "PASS" else 0 for row in case_rows])
      colors = [cmap[row["status"]] for row in case_rows]
      ax.scatter(x, np.full_like(x, y), c=colors, s=80, edgecolors="black", linewidths=0.5, zorder=3)
      summary = []
      if np.any(stable == 1):
         max_pass = x[stable == 1].max()
         summary.append(f"pass <= {max_pass:g}")
      if np.any(stable == 0):
         min_fail = x[stable == 0].min()
         summary.append(f"fail >= {min_fail:g}")
      ax.text(x_text, y, ", ".join(summary), ha="left", va="center", fontsize=8)
      y_ticks.append(y)
      y_labels.append(case_rows[0]["label"])
   ax.axvline(1.0, color="black", linestyle=":", linewidth=1.6)
   ax.set_xlabel("CFL")
   ax.set_yticks(y_ticks, y_labels)
   ax.set_title("AMReX CFL stability scan: green=pass, red=fail")
   ax.set_xlim(min(x_values) - 0.04 * (max(x_values) - min(x_values)), x_text + 0.22 * (max(x_values) - min(x_values)))
   ax.grid(True, axis="x", which="major", alpha=0.25)
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def append_stability_history(rows: list[dict[str, str]], history: Path) -> None:
   append_validation_history(history, [{"kind": "cfl_stability", **row} for row in rows])


def safe_cfl_failures(rows: list[dict[str, str]], required_cfl: float) -> list[dict[str, str]]:
   return [
      row for row in rows
      if float(row["cfl"]) <= required_cfl and row["status"] != "PASS"
   ]


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--suite",
      action="append",
      choices=["all", *suite_cases().keys()],
      help="stability suite to run; may repeat; default is all suites",
   )
   parser.add_argument("--cases", default="all", help="Comma-separated case names or all")
   parser.add_argument("--cfl-values", default=",".join(f"{v:g}" for v in DEFAULT_CFL_VALUES))
   parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
   parser.add_argument("--results", type=Path)
   parser.add_argument("--plot", type=Path)
   parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--require-pass-cfl", type=float, default=0.9)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--plot-only", action="store_true")
   parser.add_argument("--list-cases", action="store_true")
   args = parser.parse_args()

   try:
      cases = selected_cases(args.suite, args.cases)
   except ValueError as exc:
      raise SystemExit(str(exc)) from exc
   if not cases:
      raise SystemExit("no cases selected")
   if args.list_cases:
      for case in cases:
         print(f"{case.name}\t{case.suite}\t{case.label}")
      return 0

   run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
   results_path = args.results if args.results is not None else run_root / "cfl_stability.tsv"
   results_path = results_path if results_path.is_absolute() else ROOT / results_path
   if args.plot_only:
      rows = read_results(results_path)
   else:
      rows = []
      for case in cases:
         for cfl in parse_cfl_values(args.cfl_values):
            row = run_case(case, cfl, run_root, args.ranks, args.timeout, args.force)
            rows.append(row)
            print(f"{row['case']} CFL={row['cfl']} {row['status']}")
      write_results(rows, results_path)
      print(f"wrote {results_path.relative_to(ROOT)}")

   plot_path = args.plot
   if plot_path is None:
      plot_path = next_output_path(DEFAULT_PLOT_DIR, "cfl_stability")
   plot_path = plot_path if plot_path.is_absolute() else ROOT / plot_path
   plot_results(rows, plot_path)
   print(f"wrote {plot_path.relative_to(ROOT)}")
   if not args.plot_only:
      history_path = args.history if args.history.is_absolute() else ROOT / args.history
      append_stability_history(rows, history_path)
      print(f"appended {history_path.relative_to(ROOT)}")

   failures = safe_cfl_failures(rows, args.require_pass_cfl)
   if failures:
      print(f"{len(failures)} run(s) failed at CFL <= {args.require_pass_cfl:g}")
      return 1
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
