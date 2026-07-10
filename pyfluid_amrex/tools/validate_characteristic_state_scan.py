#!/usr/bin/env python3
"""Validate nonlinear HD/MHD characteristic speeds by sweeping background states."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyfluid.dispersion import ideal_mhd_wave_speeds, sound_speed_ideal_gas  # noqa: E402
from validate_dispersion_relations import DispersionCase, collect_result, run_case  # noqa: E402
from validation_common import (  # noqa: E402
   CLOSED_POLICY,
   ROOT,
   append_validation_history,
   scalar_validation_row,
   write_tsv,
)


RUN_ROOT = ROOT / "runs/current/characteristic_state_scan"
SUMMARY = RUN_ROOT / "summary.tsv"
PLOT_ROOT = ROOT / "plots/current/scans"
HISTORY = ROOT / "runs/current/validation_history.tsv"


@dataclass(frozen=True)
class ScanCase:
   case: DispersionCase
   group: str
   parameter: float
   parameter_label: str


def next_output_path(directory: Path, stem: str) -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   used = []
   for path in directory.glob(f"{stem}_*.png"):
      suffix = path.stem.removeprefix(f"{stem}_")
      if suffix.isdigit():
         used.append(int(suffix))
   return directory / f"{stem}_{max(used, default=0) + 1:03d}.png"


def safe_name(value: float) -> str:
   return f"{value:.4g}".replace("-", "m").replace(".", "p")


def mhd_background_from_angle(theta_deg: float, bmag: float = 1.35) -> tuple[float, float, float]:
   theta = math.radians(theta_deg)
   return bmag * math.cos(theta), bmag * math.sin(theta), 0.0


def mhd_background_from_bmag(bmag: float, theta_deg: float = 45.0) -> tuple[float, float, float]:
   theta = math.radians(theta_deg)
   return bmag * math.cos(theta), bmag * math.sin(theta), 0.0


def build_scan_cases(args: argparse.Namespace) -> list[ScanCase]:
   out: list[ScanCase] = []
   nx = args.nx
   final_time = args.final_time
   amplitude = args.amplitude

   for ratio in args.hd_pressure_ratios:
      pressure = ratio
      for family, label, field in (
         ("acoustic_plus", "HD acoustic +", "rho"),
         ("acoustic_minus", "HD acoustic -", "rho"),
         ("entropy", "HD entropy/contact", "rho"),
         ("shear_y", "HD shear y", "vy"),
      ):
         name = f"hd_{family}_pr_{safe_name(ratio)}"
         out.append(ScanCase(
            DispersionCase(
               name,
               label,
               "hydro",
               "linear_wave_x",
               family,
               field,
               nx=nx,
               final_time=final_time,
               amplitude=amplitude,
               pressure0=pressure,
               v1_0=args.v1_0,
            ),
            "hd_pressure",
            ratio,
            "P/rho",
         ))

   for theta in args.mhd_angles:
      bx, by, bz = mhd_background_from_angle(theta, args.mhd_bmag)
      for family, label, field in (
         ("fast_plus", "MHD fast +", "vx"),
         ("alfven_plus", "MHD Alfven +", "b3"),
         ("slow_plus", "MHD slow +", "vx"),
         ("entropy", "MHD entropy/contact", "rho"),
      ):
         name = f"mhd_angle_{family}_{safe_name(theta)}"
         out.append(ScanCase(
            DispersionCase(
               name,
               label,
               "mhd",
               "linear_wave_x",
               family,
               field,
               nx=nx,
               final_time=final_time,
               amplitude=amplitude,
               v1_0=args.v1_0,
               b1_0=bx,
               b2_0=by,
               b3_0=bz,
               riemann="hlld",
            ),
            "mhd_angle",
            theta,
            "theta_deg",
         ))

   for bmag in args.mhd_bmags:
      bx, by, bz = mhd_background_from_bmag(bmag, args.mhd_bscan_angle)
      for family, label, field in (
         ("fast_plus", "MHD fast +", "vx"),
         ("alfven_plus", "MHD Alfven +", "b3"),
         ("slow_plus", "MHD slow +", "vx"),
         ("entropy", "MHD entropy/contact", "rho"),
      ):
         name = f"mhd_bmag_{family}_{safe_name(bmag)}"
         out.append(ScanCase(
            DispersionCase(
               name,
               label,
               "mhd",
               "linear_wave_x",
               family,
               field,
               nx=nx,
               final_time=final_time,
               amplitude=amplitude,
               v1_0=args.v1_0,
               b1_0=bx,
               b2_0=by,
               b3_0=bz,
               riemann="hlld",
            ),
            "mhd_bmag",
            bmag,
            "|B|",
         ))
   return out


def speed_from_result(scan: ScanCase, result: dict[str, object]) -> tuple[float, float]:
   case = scan.case
   k = 2.0 * math.pi * float(result["mode_number"])
   theory = float(result["theory_rate"]) / k
   measured = float(result["measured_rate"]) / k
   return theory, measured


def validate_scan_case(scan: ScanCase, args: argparse.Namespace) -> dict[str, object]:
   case = scan.case
   run_dir, input_path, returncode, elapsed, stdout = run_case(case, args)
   row = scalar_validation_row(
      run_dir,
      case=case.name,
      input_file=input_path,
      policy=CLOSED_POLICY,
      mhd=case.physics == "mhd",
      dim=1,
      ranks=args.ranks,
      base_input=str(input_path.relative_to(ROOT)),
      kind="characteristic_state_scan",
   )
   row["returncode"] = returncode
   row["seconds"] = f"{elapsed:.6g}"
   row["group"] = scan.group
   row["parameter"] = scan.parameter
   row["parameter_label"] = scan.parameter_label
   row["family"] = case.family
   row["display_label"] = case.label
   if returncode != 0:
      row["status"] = "FAIL"
      row["reason"] = f"solver returned {returncode}: {' | '.join(stdout.splitlines()[-4:])}"
      return row

   result = collect_result(case, run_dir)
   theory_speed, measured_speed = speed_from_result(scan, result)
   speed_abs_error = abs(measured_speed - theory_speed)
   speed_rel_error = speed_abs_error / max(abs(theory_speed), 1.0)
   row.update({
      "theory_speed": theory_speed,
      "measured_speed": measured_speed,
      "speed_abs_error": speed_abs_error,
      "speed_rel_error": speed_rel_error,
      "field_error": result["field_error"],
      "rate_abs_error": result["rate_abs_error"],
      "rate_rel_error": result["rate_rel_error"],
      "rho0": case.rho0,
      "pressure0": case.pressure0,
      "b1_0": case.b1_0,
      "b2_0": case.b2_0,
      "b3_0": case.b3_0,
      "run_dir": run_dir.relative_to(ROOT),
   })

   failures = []
   if row["status"] != "PASS":
      failures.append(str(row["reason"]))
   if float(result["field_error"]) > args.field_error_tol:
      failures.append(f"field_error={result['field_error']:.3e} > {args.field_error_tol:.3e}")
   if speed_abs_error > args.speed_abs_error_tol and speed_rel_error > args.speed_rel_error_tol:
      failures.append(f"speed_abs_error={speed_abs_error:.3e}, speed_rel_error={speed_rel_error:.3e}")
   if failures:
      row["status"] = "FAIL"
      row["reason"] = "; ".join(failures)
   else:
      row["status"] = "PASS"
      row["reason"] = f"speed num/theory={measured_speed:.6g}/{theory_speed:.6g}, field_error={result['field_error']:.3e}"
   return row


def group_rows(rows: list[dict[str, object]], group: str, label: str) -> list[dict[str, object]]:
   selected = [row for row in rows if row["group"] == group and row["display_label"] == label]
   return sorted(selected, key=lambda row: float(row["parameter"]))


def plot_results(rows: list[dict[str, object]], output: Path, args: argparse.Namespace) -> None:
   fig, axs = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
   fig.suptitle("AMReX nonlinear-state characteristic speed validation: WENO-Z char/RK4, CFL=0.9")
   colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

   ax = axs[0, 0]
   x = np.linspace(min(args.hd_pressure_ratios), max(args.hd_pressure_ratios), 300)
   cs = np.sqrt(args.gamma * x)
   ax.plot(x, args.v1_0 + cs, color=colors[0], linewidth=2.4, label="HD acoustic + exact")
   ax.plot(x, args.v1_0 - cs, color=colors[3], linewidth=2.4, label="HD acoustic - exact")
   ax.plot(x, np.full_like(x, args.v1_0), color=colors[1], linewidth=2.2, label="HD entropy/shear exact")
   for label, marker, color in (
      ("HD acoustic +", "o", colors[0]),
      ("HD acoustic -", "D", colors[3]),
      ("HD entropy/contact", "s", colors[1]),
      ("HD shear y", "^", colors[2]),
   ):
      selected = group_rows(rows, "hd_pressure", label)
      ax.plot(
         [float(row["parameter"]) for row in selected],
         [float(row["measured_speed"]) for row in selected],
         linestyle=":",
         marker=marker,
         color=color,
         markeredgecolor="black",
         label=f"{label} num",
      )
   ax.set_title("HD: characteristic speeds vs state")
   ax.set_xlabel("P/rho")
   ax.set_ylabel("lambda")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend(fontsize=8)

   ax = axs[0, 1]
   theta = np.linspace(min(args.mhd_angles), max(args.mhd_angles), 300)
   exact = {"MHD fast +": [], "MHD Alfven +": [], "MHD slow +": [], "MHD entropy/contact": []}
   for value in theta:
      speeds = ideal_mhd_wave_speeds(1.0, 1.0, args.gamma, mhd_background_from_angle(value, args.mhd_bmag), (1.0, 0.0, 0.0))
      exact["MHD fast +"].append(args.v1_0 + speeds["fast"])
      exact["MHD Alfven +"].append(args.v1_0 + speeds["alfven"])
      exact["MHD slow +"].append(args.v1_0 + speeds["slow"])
      exact["MHD entropy/contact"].append(args.v1_0)
   for idx, label in enumerate(exact):
      ax.plot(theta, exact[label], color=colors[idx], linewidth=2.4, label=f"{label} exact")
      selected = group_rows(rows, "mhd_angle", label)
      ax.plot(
         [float(row["parameter"]) for row in selected],
         [float(row["measured_speed"]) for row in selected],
         linestyle=":",
         marker=["o", "s", "^", "P"][idx],
         color=colors[idx],
         markeredgecolor="black",
         label=f"{label} num",
      )
   ax.set_title(f"MHD: speeds vs angle, |B|={args.mhd_bmag:g}")
   ax.set_xlabel("angle between B and propagation direction [deg]")
   ax.set_ylabel("lambda")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend(fontsize=7)

   ax = axs[1, 0]
   bmag = np.linspace(min(args.mhd_bmags), max(args.mhd_bmags), 300)
   exact = {"MHD fast +": [], "MHD Alfven +": [], "MHD slow +": [], "MHD entropy/contact": []}
   for value in bmag:
      speeds = ideal_mhd_wave_speeds(1.0, 1.0, args.gamma, mhd_background_from_bmag(value, args.mhd_bscan_angle), (1.0, 0.0, 0.0))
      exact["MHD fast +"].append(args.v1_0 + speeds["fast"])
      exact["MHD Alfven +"].append(args.v1_0 + speeds["alfven"])
      exact["MHD slow +"].append(args.v1_0 + speeds["slow"])
      exact["MHD entropy/contact"].append(args.v1_0)
   for idx, label in enumerate(exact):
      ax.plot(bmag, exact[label], color=colors[idx], linewidth=2.4, label=f"{label} exact")
      selected = group_rows(rows, "mhd_bmag", label)
      ax.plot(
         [float(row["parameter"]) for row in selected],
         [float(row["measured_speed"]) for row in selected],
         linestyle=":",
         marker=["o", "s", "^", "P"][idx],
         color=colors[idx],
         markeredgecolor="black",
         label=f"{label} num",
      )
   ax.set_title(f"MHD: speeds vs field strength, theta={args.mhd_bscan_angle:g} deg")
   ax.set_xlabel("|B|")
   ax.set_ylabel("lambda")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend(fontsize=7)

   ax = axs[1, 1]
   labels = []
   speed_errors = []
   field_errors = []
   for group in ("hd_pressure", "mhd_angle", "mhd_bmag"):
      for label in dict.fromkeys(str(row["display_label"]) for row in rows if row["group"] == group):
         selected = group_rows(rows, group, label)
         labels.append(f"{group.replace('_', ' ')}\n{label.replace(' +', '+')}")
         speed_errors.append(max(abs(float(row["speed_abs_error"])) for row in selected))
         field_errors.append(max(abs(float(row["field_error"])) for row in selected))
   xloc = np.arange(len(labels))
   ax.semilogy(xloc, np.maximum(speed_errors, 1.0e-16), marker="o", linewidth=1.8, label="speed abs error")
   ax.semilogy(xloc, np.maximum(field_errors, 1.0e-16), marker="s", linewidth=1.8, label="field L2 error")
   ax.set_title("All state-sweep branches")
   ax.set_xticks(xloc)
   ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
   ax.grid(True, which="major", axis="y", alpha=0.25)
   ax.legend(fontsize=8)

   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def write_summary(rows: list[dict[str, object]]) -> None:
   fields = [
      "case",
      "group",
      "display_label",
      "parameter_label",
      "parameter",
      "status",
      "reason",
      "theory_speed",
      "measured_speed",
      "speed_abs_error",
      "speed_rel_error",
      "field_error",
      "rate_abs_error",
      "rate_rel_error",
      "rho0",
      "pressure0",
      "b1_0",
      "b2_0",
      "b3_0",
      "avg_mcells_per_s",
      "avg_sim_time_per_s",
      "run_dir",
   ]
   write_tsv(SUMMARY, rows, fields)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--cfl", type=float, default=0.9)
   parser.add_argument("--nx", type=int, default=256)
   parser.add_argument("--final-time", type=float, default=0.08)
   parser.add_argument("--amplitude", type=float, default=1.0e-6)
   parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
   parser.add_argument("--v1_0", type=float, default=0.1)
   parser.add_argument("--hd-pressure-ratios", type=float, nargs="+", default=[0.15, 0.3, 0.6, 1.0, 1.8, 3.0])
   parser.add_argument("--mhd-angles", type=float, nargs="+", default=[5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0])
   parser.add_argument("--mhd-bmag", type=float, default=1.35)
   parser.add_argument("--mhd-bmags", type=float, nargs="+", default=[0.2, 0.35, 0.6, 1.0, 1.7, 2.6, 3.6])
   parser.add_argument("--mhd-bscan-angle", type=float, default=45.0)
   parser.add_argument("--field-error-tol", type=float, default=2.0e-2)
   parser.add_argument("--speed-abs-error-tol", type=float, default=2.0e-3)
   parser.add_argument("--speed-rel-error-tol", type=float, default=2.0e-3)
   parser.add_argument("--skip-plots", action="store_true")
   args = parser.parse_args()
   args.run_root = RUN_ROOT

   if args.force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   rows: list[dict[str, object]] = []
   status = 0
   for scan in build_scan_cases(args):
      row = validate_scan_case(scan, args)
      rows.append(row)
      if row["status"] != "PASS":
         status = 1
      print(f"{row['case']}: {row['status']} {row['reason']}", flush=True)

   write_summary(rows)
   append_validation_history(HISTORY, rows)
   if status == 0 and not args.skip_plots:
      plot_path = next_output_path(PLOT_ROOT, "characteristic_state_scan")
      plot_results(rows, plot_path, args)
      print(f"wrote {plot_path.relative_to(ROOT)}")
   print(f"wrote {SUMMARY.relative_to(ROOT)}")
   return status


if __name__ == "__main__":
   raise SystemExit(main())
