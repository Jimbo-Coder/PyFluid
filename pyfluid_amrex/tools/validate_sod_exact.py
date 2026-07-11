#!/usr/bin/env python3
"""Validate AMReX hydro shock tubes against the exact Sod solution."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyfluid.exact_solutions import sod_conserved_integrals, sod_exact_solution  # noqa: E402
from plotfile_reader import latest_plotfile as discover_latest_plotfile
from plot_amrex_1d_profile import parse_plot_header, read_level0_state_1d  # noqa: E402
from validation_common import append_validation_history, next_numbered_path, read_scalar_rows, run_mpi, write_tsv  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "pyfluid_amrex" / "inputs" / "sod_x_1d.in"
EXECUTABLE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_1d"
RUN_ROOT = ROOT / "runs" / "current" / "sod_exact_validation"
PLOT_DIR = ROOT / "plots" / "current" / "IC" / "sod_x"
HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"
RIEMANN_CHOICES = ("llf", "hll", "hlle", "hllc")
LADDER_RIEMANN_MIN_IMPROVEMENT = 1.25
LADDER_PWL_MIN_IMPROVEMENT = 2.0
LADDER_WENOZ_MAX_REGRESSION = 1.10


@dataclass(frozen=True)
class SodCase:
   riemann: str
   recon: str
   timestep: str
   nx: int
   final_time: float
   cfl: float

   @property
   def name(self) -> str:
      return f"sod_{self.recon}_{self.riemann}_{self.timestep}_nx{self.nx}"

   @property
   def label(self) -> str:
      return f"Sod {self.recon.upper()}/{self.riemann.upper()}/{self.timestep.upper()}, nx={self.nx}"


@dataclass(frozen=True)
class LadderPoint:
   key: str
   riemann: str
   recon: str
   timestep: str
   check_against: str
   min_improvement: float
   description: str


LADDER_POINTS = (
   LadderPoint(
      "pwc_llf_euler",
      "llf",
      "pwc",
      "euler",
      "",
      0.0,
      "diffusive baseline",
   ),
   LadderPoint(
      "pwc_hllc_euler",
      "hllc",
      "pwc",
      "euler",
      "pwc_llf_euler",
      LADDER_RIEMANN_MIN_IMPROVEMENT,
      "contact-resolving HLLC should improve over LLF at fixed PWC/Euler",
   ),
   LadderPoint(
      "pwl_hllc_ssprk3",
      "hllc",
      "pwl",
      "ssprk3",
      "pwc_hllc_euler",
      LADDER_PWL_MIN_IMPROVEMENT,
      "PWL reconstruction should sharply improve the exact profile over PWC",
   ),
   LadderPoint(
      "wenoz_hllc_rk4",
      "hllc",
      "wenoz",
      "rk4",
      "pwl_hllc_ssprk3",
      1.0 / LADDER_WENOZ_MAX_REGRESSION,
      "WENO-Z should not regress the global shock-tube L1 metric",
   ),
)


def parse_list(text: str, choices: tuple[str, ...]) -> list[str]:
   if text == "all":
      return list(choices)
   values = [raw.strip() for raw in text.split(",") if raw.strip()]
   unknown = sorted(set(values) - set(choices))
   if unknown:
      raise ValueError(f"unknown value(s): {', '.join(unknown)}; choices: {', '.join(choices)}")
   return values


def write_case_input(case: SodCase, args: argparse.Namespace) -> tuple[Path, Path]:
   run_dir = args.run_root / case.name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_dir = args.run_root / "inputs"
   input_dir.mkdir(parents=True, exist_ok=True)

   ng = 3 if case.recon.startswith("wenoz") else (2 if case.recon == "pwl" else 1)
   overrides = {
      "pyfluid.run_name": case.name,
      "pyfluid.output_dir": str(run_dir.relative_to(ROOT)),
      "amr.n_cell": str(case.nx),
      "amr.max_level": "0",
      "pyfluid.final_time": f"{case.final_time:.17g}",
      "pyfluid.max_step": "100000",
      "pyfluid.cfl": f"{case.cfl:.17g}",
      "pyfluid.ng": str(ng),
      "pyfluid.recon": case.recon,
      "pyfluid.riemann": case.riemann,
      "pyfluid.timestep": case.timestep,
      "pyfluid.scalar_int": "8",
      "pyfluid.plot_int": "100000",
      "pyfluid.checkpoint_int": "-1",
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_int": "-1",
      "pyfluid.aux_gf_compute_int": "-1",
   }
   lines = INPUT.read_text().splitlines()
   lines.append("")
   lines.append("# validate_sod_exact.py overrides")
   for key, value in overrides.items():
      lines.append(f"{key} = {value}")
   input_path = input_dir / f"{case.name}.in"
   input_path.write_text("\n".join(lines) + "\n")
   return input_path, run_dir


def run_solver(case: SodCase, args: argparse.Namespace) -> tuple[int, str, Path]:
   input_path, run_dir = write_case_input(case, args)
   completed = run_mpi(
      EXECUTABLE,
      [input_path],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   return completed.returncode, completed.stdout, run_dir


def pressure_from_state(state: np.ndarray, names: list[str], gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   index = {name: n for n, name in enumerate(names)}
   rho = state[index["rho"]]
   vx = state[index["mom1"]] / rho
   pressure = (gamma - 1.0) * (state[index["energy"]] - 0.5 * rho * vx * vx)
   return rho, pressure, vx


def normalized_l1(values: np.ndarray, exact: np.ndarray) -> float:
   scale = max(float(np.max(exact) - np.min(exact)), 1.0e-300)
   return float(np.mean(np.abs(values - exact)) / scale)


def scalar_error(last: dict[str, str], exact: dict[str, float]) -> tuple[float, float, float]:
   mass = abs(float(last.get("mass", "nan")) - exact["mass"]) / max(abs(exact["mass"]), 1.0e-300)
   energy = abs(float(last.get("energy", "nan")) - exact["energy"]) / max(abs(exact["energy"]), 1.0e-300)
   mom = abs(float(last.get("mom1", "nan")) - exact["momx"])
   return mass, energy, mom


def collect_result(case: SodCase, run_dir: Path, args: argparse.Namespace) -> dict[str, object]:
   plotfile = discover_latest_plotfile(run_dir)
   if plotfile is None:
      raise FileNotFoundError(f"no plotfiles found under {run_dir}")
   names, time_value = parse_plot_header(plotfile)
   state = read_level0_state_1d(plotfile, names)
   rho, pressure, vx = pressure_from_state(state, names, args.gamma)
   nx = rho.size
   dx = 1.0 / nx
   x = (np.arange(nx, dtype=float) + 0.5) * dx
   left = (args.rho_left, args.v1_left, args.pressure_left)
   right = (args.rho_right, args.v1_right, args.pressure_right)
   rho_exact, pressure_exact, vx_exact = sod_exact_solution(x, time_value, args.gamma, left, right, args.shock_x0)
   exact_integrals = sod_conserved_integrals(time_value, args.gamma, left, right, args.shock_x0)
   scalar_rows = read_scalar_rows(run_dir)
   if not scalar_rows:
      raise RuntimeError(f"missing scalars.tsv rows under {run_dir}")
   last = scalar_rows[-1]
   mass_error, energy_error, mom_error = scalar_error(last, exact_integrals)

   rho_l1 = normalized_l1(rho, rho_exact)
   pressure_l1 = normalized_l1(pressure, pressure_exact)
   v1_l1 = normalized_l1(vx, vx_exact)
   total_l1 = rho_l1 + pressure_l1 + v1_l1
   failures = []
   if rho_l1 > args.rho_l1_tol:
      failures.append(f"rho_l1={rho_l1:.3e} > {args.rho_l1_tol:.3e}")
   if pressure_l1 > args.pressure_l1_tol:
      failures.append(f"pressure_l1={pressure_l1:.3e} > {args.pressure_l1_tol:.3e}")
   if v1_l1 > args.v1_l1_tol:
      failures.append(f"v1_l1={v1_l1:.3e} > {args.v1_l1_tol:.3e}")
   if mass_error > args.scalar_tol:
      failures.append(f"mass_exact_rel_error={mass_error:.3e} > {args.scalar_tol:.3e}")
   if energy_error > args.scalar_tol:
      failures.append(f"energy_exact_rel_error={energy_error:.3e} > {args.scalar_tol:.3e}")
   if mom_error > args.momentum_tol:
      failures.append(f"mom_exact_abs_error={mom_error:.3e} > {args.momentum_tol:.3e}")

   plot_path = None
   if not args.skip_plots:
      plot_path = write_profile_plot(
         case,
         x,
         time_value,
         (rho, pressure, vx),
         (rho_exact, pressure_exact, vx_exact),
         args.plot_dir,
      )
   return {
      "case": case.name,
      "label": case.label,
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(failures),
      "riemann": case.riemann,
      "recon": case.recon,
      "timestep": case.timestep,
      "nx": nx,
      "time": f"{time_value:.17g}",
      "rho_l1": rho_l1,
      "pressure_l1": pressure_l1,
      "v1_l1": v1_l1,
      "total_l1": total_l1,
      "mass_exact_rel_error": mass_error,
      "energy_exact_rel_error": energy_error,
      "mom_exact_abs_error": mom_error,
      "mass_rel_drift": last.get("mass_rel_drift", ""),
      "energy_rel_drift": last.get("energy_rel_drift", ""),
      "avg_mcells_per_s": average_scalars(scalar_rows, "mcells_per_s"),
      "avg_sim_time_per_s": average_scalars(scalar_rows, "sim_time_per_s"),
      "plot": str(plot_path.relative_to(ROOT)) if plot_path is not None else "",
      "run_dir": str(run_dir.relative_to(ROOT)),
   }


def row_error(row: dict[str, object]) -> float:
   try:
      return float(row["total_l1"])
   except (KeyError, TypeError, ValueError):
      return float("nan")


def apply_ladder_checks(rows: list[dict[str, object]]) -> None:
   by_key = {f"{row.get('recon')}_{row.get('riemann')}_{row.get('timestep')}": row for row in rows}
   for point in LADDER_POINTS:
      row = by_key.get(point.key)
      if row is None:
         continue
      row["ladder_description"] = point.description
      if not point.check_against:
         row["comparison_case"] = ""
         row["improvement_vs_comparison"] = ""
         row["required_improvement"] = ""
         continue
      baseline = by_key.get(point.check_against)
      row["comparison_case"] = point.check_against
      row["required_improvement"] = point.min_improvement
      if baseline is None:
         row["improvement_vs_comparison"] = ""
         mark_ladder_failure(row, f"missing ladder baseline {point.check_against}")
         continue
      base_error = row_error(baseline)
      current_error = row_error(row)
      improvement = base_error / current_error if math.isfinite(base_error) and math.isfinite(current_error) and current_error > 0.0 else float("nan")
      row["improvement_vs_comparison"] = improvement
      if not math.isfinite(improvement) or improvement < point.min_improvement:
         mark_ladder_failure(
            row,
            f"method improvement {improvement:.3g} < required {point.min_improvement:.3g} vs {point.check_against}",
         )


def mark_ladder_failure(row: dict[str, object], reason: str) -> None:
   existing = str(row.get("reason", ""))
   row["status"] = "FAIL"
   row["reason"] = reason if not existing else f"{existing}; {reason}"


def average_scalars(rows: list[dict[str, str]], key: str) -> float:
   values = []
   for row in rows:
      try:
         value = float(row.get(key, "nan"))
      except ValueError:
         continue
      if math.isfinite(value) and value > 0.0:
         values.append(value)
   return float("nan") if not values else sum(values) / len(values)


def write_profile_plot(
   case: SodCase,
   x: np.ndarray,
   time_value: float,
   numerical: tuple[np.ndarray, np.ndarray, np.ndarray],
   exact: tuple[np.ndarray, np.ndarray, np.ndarray],
   plot_dir: Path,
) -> Path:
   plot_dir.mkdir(parents=True, exist_ok=True)
   path = next_numbered_path(plot_dir, f"{case.name}_exact_overlay")
   fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.2), sharex=True, constrained_layout=True)
   fig.suptitle(f"{case.label}: exact Sod overlay, t={time_value:.5f}")
   labels = ("rho", "P", "v1")
   for ax, label, num, ref in zip(axes, labels, numerical, exact, strict=True):
      ax.plot(x, num, color="tab:blue", marker="o", markersize=2.0, linewidth=1.0, label="num")
      ax.plot(x, ref, color="black", linestyle=":", linewidth=1.8, label="exact")
      ax.set_ylabel(label)
      ax.grid(True, which="major", alpha=0.25)
      ax.legend()
   axes[-1].set_xlabel("x")
   fig.savefig(path, dpi=180)
   plt.close(fig)
   return path


def write_ladder_plot(rows: list[dict[str, object]], plot_dir: Path) -> Path:
   ordered = []
   for point in LADDER_POINTS:
      for row in rows:
         if row.get("recon") == point.recon and row.get("riemann") == point.riemann and row.get("timestep") == point.timestep:
            ordered.append(row)
            break
   if not ordered:
      raise ValueError("no Sod ladder rows to plot")

   labels = [f"{row['recon'].upper()}\n{row['riemann'].upper()}\n{row['timestep'].upper()}" for row in ordered]
   x = np.arange(len(ordered))
   plot_dir.mkdir(parents=True, exist_ok=True)
   path = next_numbered_path(plot_dir, "sod_method_ladder_errors")
   fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
   series = [
      ("rho", "rho_l1"),
      ("P", "pressure_l1"),
      ("v1", "v1_l1"),
      ("total", "total_l1"),
   ]
   for label, key in series:
      values = np.asarray([float(row[key]) for row in ordered], dtype=float)
      ax.semilogy(x, values, marker="o", linewidth=2.0, label=label)
   nx = ordered[0].get("nx", "")
   time_value = ordered[0].get("time", "")
   ax.set_title(f"Sod exact method ladder: normalized L1 error, nx={nx}, t={float(time_value):.3g}")
   ax.set_ylabel("normalized L1 error")
   ax.set_xticks(x)
   ax.set_xticklabels(labels)
   ax.grid(True, which="major", alpha=0.25)
   ax.legend()
   fig.savefig(path, dpi=180)
   plt.close(fig)
   return path


def write_results(rows: list[dict[str, object]], output: Path) -> None:
   fields = [
      "case",
      "label",
      "status",
      "reason",
      "riemann",
      "recon",
      "timestep",
      "nx",
      "time",
      "rho_l1",
      "pressure_l1",
      "v1_l1",
      "total_l1",
      "comparison_case",
      "improvement_vs_comparison",
      "required_improvement",
      "ladder_description",
      "mass_exact_rel_error",
      "energy_exact_rel_error",
      "mom_exact_abs_error",
      "mass_rel_drift",
      "energy_rel_drift",
      "avg_mcells_per_s",
      "avg_sim_time_per_s",
      "plot",
      "run_dir",
   ]
   write_tsv(output, rows, fields)


def append_history(rows: list[dict[str, object]]) -> None:
   history_rows = []
   for row in rows:
      history_rows.append({
         "kind": "analytic",
         "suite": "sod_exact",
         "case": row.get("case", ""),
         "label": row.get("label", ""),
         "status": row.get("status", "FAIL"),
         "reason": row.get("reason", ""),
         "time": row.get("time", ""),
         "mass_rel_drift": row.get("mass_rel_drift", ""),
         "energy_rel_drift": row.get("energy_rel_drift", ""),
         "avg_mcells_per_s": row.get("avg_mcells_per_s", ""),
         "avg_sim_time_per_s": row.get("avg_sim_time_per_s", ""),
         "tail": row.get("plot", ""),
      })
   append_validation_history(HISTORY, history_rows)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--riemann", default="hllc", help="Comma-separated Riemann solvers or all")
   parser.add_argument("--recon", default="pwl")
   parser.add_argument("--timestep", default="ssprk3")
   parser.add_argument("--nx", type=int, default=512)
   parser.add_argument("--final-time", type=float, default=0.2)
   parser.add_argument("--cfl", type=float, default=0.9)
   parser.add_argument("--gamma", type=float, default=1.4)
   parser.add_argument("--shock-x0", type=float, default=0.5)
   parser.add_argument("--rho-left", type=float, default=1.0)
   parser.add_argument("--pressure-left", type=float, default=1.0)
   parser.add_argument("--v1-left", dest="v1_left", type=float, default=0.0)
   parser.add_argument("--rho-right", type=float, default=0.125)
   parser.add_argument("--pressure-right", type=float, default=0.1)
   parser.add_argument("--v1-right", dest="v1_right", type=float, default=0.0)
   parser.add_argument("--rho-l1-tol", type=float, default=4.0e-2)
   parser.add_argument("--pressure-l1-tol", type=float, default=4.0e-2)
   parser.add_argument("--v1-l1-tol", dest="v1_l1_tol", type=float, default=7.0e-2)
   parser.add_argument("--scalar-tol", type=float, default=1.0e-8)
   parser.add_argument("--momentum-tol", type=float, default=1.0e-4)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
   parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
   parser.add_argument("--output", type=Path, default=RUN_ROOT / "sod_exact.tsv")
   parser.add_argument("--method-ladder", action="store_true", help="Run the fixed Sod method-improvement ladder")
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--skip-plots", action="store_true", help="write exact-comparison TSV/history only")
   args = parser.parse_args()
   args.run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
   args.plot_dir = args.plot_dir if args.plot_dir.is_absolute() else ROOT / args.plot_dir
   args.output = args.output if args.output.is_absolute() else ROOT / args.output

   if args.method_ladder:
      cases = [SodCase(point.riemann, point.recon, point.timestep, args.nx, args.final_time, args.cfl) for point in LADDER_POINTS]
   else:
      cases = [SodCase(riemann, args.recon, args.timestep, args.nx, args.final_time, args.cfl) for riemann in parse_list(args.riemann, RIEMANN_CHOICES)]

   rows: list[dict[str, object]] = []
   for case in cases:
      try:
         returncode, stdout, run_dir = run_solver(case, args)
         if returncode != 0:
            rows.append({
               "case": case.name,
               "label": case.label,
               "status": "FAIL",
               "reason": "solver failed: " + " | ".join(stdout.splitlines()[-6:]),
               "riemann": case.riemann,
               "recon": case.recon,
               "timestep": case.timestep,
               "nx": args.nx,
               "run_dir": str(run_dir.relative_to(ROOT)),
            })
            continue
         rows.append(collect_result(case, run_dir, args))
      except Exception as exc:  # keep the validation TSV informative on postprocess failures
         rows.append({
            "case": case.name,
            "label": case.label,
            "status": "FAIL",
            "reason": str(exc),
            "riemann": case.riemann,
            "recon": case.recon,
            "timestep": case.timestep,
            "nx": args.nx,
         })
      print(f"{case.name}: {rows[-1]['status']} {rows[-1].get('reason', '')}", flush=True)

   if args.method_ladder:
      apply_ladder_checks(rows)
      if not args.skip_plots:
         ladder_plot = write_ladder_plot(rows, args.plot_dir)
         print(f"wrote {ladder_plot.relative_to(ROOT)}")

   write_results(rows, args.output)
   append_history(rows)
   print(f"wrote {args.output.relative_to(ROOT)}")
   print(f"appended {HISTORY.relative_to(ROOT)}")
   failed = [row for row in rows if row["status"] != "PASS"]
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
