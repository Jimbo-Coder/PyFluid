#!/usr/bin/env python3
"""Validate radial-origin finite-volume fills for curvilinear runs.

This is a focused staircase for the coordinate origin: PWC/PWL/WENO without
AMR, then AMR, then optional deeper AMR/subcycling stress.  It exists because
origin ghost fills are a topology condition, not a reconstruction fallback.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plotfile_reader import latest_plotfile
from plot_amrex_slice import parse_plot_bounds, parse_plot_header, read_level_state
from validation_common import (
   OPEN_POLICY,
   ROOT,
   ScalarPolicy,
   append_validation_history,
   materialize_run_input,
   repo_relative,
   run_amrex,
   scalar_validation_row,
   select_named_cases,
   write_tsv,
)


RUN_ROOT = ROOT / "runs/current/curvilinear_origin"
CASE_INPUTS = RUN_ROOT / "inputs"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
BASE_INPUT = ROOT / "pyfluid_amrex/inputs/cylindrical_spiral_radiation_rphi_2d_amr.in"


@dataclass(frozen=True)
class OriginCase:
   name: str
   branch: str
   kind: str = "run"
   policy: ScalarPolicy = OPEN_POLICY
   overrides: dict[str, str] = field(default_factory=dict)


COMMON = {
   "geometry.prob_lo": "0.0 0.0",
   "pyfluid.bc_xlo": "origin",
   "pyfluid.scalar_int": "8",
   "pyfluid.plot_int": "-1",
   "pyfluid.checkpoint_int": "-1",
   "pyfluid.aux_gfs": "none",
   "pyfluid.aux_gf_int": "-1",
   "pyfluid.aux_gf_compute_int": "-1",
   "pyfluid.physical_check_int": "1",
   "pyfluid.debug_check_finite": "1",
}


def configure_run_root(path: Path) -> None:
   global RUN_ROOT, CASE_INPUTS, SUMMARY
   RUN_ROOT = path if path.is_absolute() else ROOT / path
   CASE_INPUTS = RUN_ROOT / "inputs"
   SUMMARY = RUN_ROOT / "summary.tsv"


def recon_ng(recon: str) -> str:
   return "3" if recon in ("wenoz", "wenoz-char") else ("2" if recon == "pwl" else "1")


def origin_cases(level: str) -> list[OriginCase]:
   cases: list[OriginCase] = []
   for recon in ("pwc", "pwl", "wenoz"):
      cases.append(OriginCase(
         name=f"origin_{recon}_l0",
         branch="recon_l0",
         overrides={
            **COMMON,
            "amr.max_level": "0",
            "pyfluid.recon": recon,
            "pyfluid.ng": recon_ng(recon),
            "pyfluid.riemann": "llf",
            "pyfluid.timestep": "euler",
            "pyfluid.final_time": "0.20",
         },
      ))

   for recon in ("pwc", "pwl"):
      cases.append(OriginCase(
         name=f"origin_{recon}_amr1",
         branch="recon_amr",
         overrides={
            **COMMON,
            "amr.max_level": "1",
            "pyfluid.recon": recon,
            "pyfluid.ng": recon_ng(recon),
            "pyfluid.riemann": "llf",
            "pyfluid.timestep": "euler",
            "pyfluid.final_time": "0.20",
         },
      ))

   cases.append(OriginCase(
      name="origin_wenoz_hllc_ssprk3_amr1",
      branch="production_amr",
      overrides={
         **COMMON,
         "amr.max_level": "1",
         "pyfluid.recon": "wenoz",
         "pyfluid.ng": "3",
         "pyfluid.riemann": "hllc",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.final_time": "0.35",
      },
   ))

   cases.append(OriginCase(
      name="origin_wenoz_hllc_ssprk3_amr1_subcycling",
      branch="subcycling",
      overrides={
         **COMMON,
         "amr.max_level": "1",
         "amr.ref_ratio": "2 2 2 2 2 2",
         "pyfluid.subcycling": "on",
         "pyfluid.recon": "wenoz",
         "pyfluid.ng": "3",
         "pyfluid.riemann": "hllc",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.final_time": "0.35",
      },
   ))

   cases.append(OriginCase(
      name="origin_cartesian_advection_l0",
      branch="origin_transmission",
      kind="transmission",
   ))

   if level == "full":
      cases.append(OriginCase(
         name="origin_wenoz_hllc_ssprk3_amr3",
         branch="deep_amr",
         overrides={
            **COMMON,
            "amr.max_level": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.ng": "3",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "ssprk3",
            "pyfluid.final_time": "0.25",
         },
      ))
      cases.append(OriginCase(
         name="origin_wenoz_hllc_ssprk3_amr3_subcycling",
         branch="deep_subcycling",
         overrides={
            **COMMON,
            "amr.max_level": "3",
            "amr.ref_ratio": "2 2 2 2 2 2",
            "pyfluid.subcycling": "on",
            "pyfluid.recon": "wenoz",
            "pyfluid.ng": "3",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "ssprk3",
            "pyfluid.final_time": "0.12",
         },
      ))
   return cases


def write_case_input(case: OriginCase, force: bool) -> Path:
   run_dir = RUN_ROOT / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   CASE_INPUTS.mkdir(parents=True, exist_ok=True)

   values = {
      "pyfluid.run_name": case.name,
      "pyfluid.output_dir": repo_relative(run_dir),
      **case.overrides,
   }
   path = CASE_INPUTS / f"{case.name}.in"
   remove_parameters = ("amr.ref_ratio",) if case.overrides.get("amr.max_level") == "0" else ()
   return materialize_run_input(
      BASE_INPUT,
      path,
      values,
      remove_parameters=remove_parameters,
   )


def failure_row(case: OriginCase, reason: str, ranks: int) -> dict[str, object]:
   return {
      "kind": "curvilinear_origin",
      "case": case.name,
      "label": case.name,
      "status": "FAIL",
      "reason": reason,
      "dim": "2",
      "input": repo_relative(BASE_INPUT),
      "policy": case.policy.name,
      "ranks": str(ranks),
      "branch": case.branch,
      "run_dir": repo_relative(RUN_ROOT / case.name),
      "log": repo_relative(RUN_ROOT / case.name / "run.log"),
   }


def write_input_file(path: Path, values: dict[str, str]) -> None:
   lines = [f"{key} = {value}" for key, value in values.items()]
   path.parent.mkdir(parents=True, exist_ok=True)
   path.write_text("\n".join(lines) + "\n")


def transmission_common_values(run_name: str, run_dir: Path) -> dict[str, str]:
   return {
      "pyfluid.run_name": run_name,
      "pyfluid.output_dir": repo_relative(run_dir),
      "mesh.patch_system": "single",
      "geometry.coord_sys": "0",
      "amr.max_level": "0",
      "amr.max_grid_size": "64",
      "amr.blocking_factor": "8",
      "amr.tag_metric": "none",
      "pyfluid.physics": "hydro",
      "pyfluid.ct": "off",
      "pyfluid.ic": "cartesian_advection",
      "pyfluid.eos": "ideal_gas",
      "pyfluid.rho0": "1.0",
      "pyfluid.pressure0": "1.0",
      "pyfluid.v1_0": "0.5",
      "pyfluid.v2_0": "0.0",
      "pyfluid.v3_0": "0.0",
      "pyfluid.passive_scalars": "1",
      "pyfluid.passive_profile": "gaussian",
      "pyfluid.passive_background": "0.0",
      "pyfluid.passive_amplitude": "1.0",
      "pyfluid.passive_sigma": "0.035",
      "pyfluid.max_step": "100000",
      "pyfluid.final_time": "0.75",
      "pyfluid.ng": "2",
      "pyfluid.cfl": "0.9",
      "pyfluid.gamma": "1.6666666666666667",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.bc": "outflow",
      "pyfluid.bc_ylo": "periodic",
      "pyfluid.bc_yhi": "periodic",
      "pyfluid.diffusion": "none",
      "pyfluid.subcycling": "off",
      "pyfluid.scalar_int": "32",
      "pyfluid.plot_int": "100000",
      "pyfluid.checkpoint_int": "-1",
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_int": "-1",
      "pyfluid.aux_gf_compute_int": "-1",
      "pyfluid.physical_check_int": "1",
      "pyfluid.debug_check_finite": "1",
   }


def transmission_inputs(case: OriginCase, force: bool) -> tuple[Path, Path, Path, Path]:
   run_dir = RUN_ROOT / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   cart_run = run_dir / "cartesian"
   cyl_run = run_dir / "cylindrical"
   cart_run.mkdir(parents=True, exist_ok=True)
   cyl_run.mkdir(parents=True, exist_ok=True)

   cart_input = CASE_INPUTS / f"{case.name}_cartesian.in"
   cyl_input = CASE_INPUTS / f"{case.name}_cylindrical.in"
   common_cart = transmission_common_values(f"{case.name}_cartesian", cart_run)
   common_cyl = transmission_common_values(f"{case.name}_cylindrical", cyl_run)
   write_input_file(cart_input, {
      **common_cart,
      "mesh.coordinate": "cartesian",
      "mesh.axes": "x y",
      "geometry.prob_lo": "-0.6 -0.0375",
      "geometry.prob_hi": "0.6 0.0375",
      "geometry.is_periodic": "0 1",
      "amr.n_cell": "128 8",
      "pyfluid.passive_x0": "-0.18",
      "pyfluid.passive_y0": "0.0",
   })
   write_input_file(cyl_input, {
      **common_cyl,
      "mesh.coordinate": "cylindrical",
      "mesh.axes": "r phi",
      "geometry.prob_lo": "0.0 0.0",
      "geometry.prob_hi": "0.6 6.283185307179586",
      "geometry.is_periodic": "0 1",
      "amr.n_cell": "64 64",
      "pyfluid.bc_xlo": "origin",
      "pyfluid.bc_xhi": "outflow",
      "pyfluid.passive_x0": "0.18",
      "pyfluid.passive_y0": "3.141592653589793",
   })
   return cart_input, cyl_input, cart_run, cyl_run


def level0_dye(plotfile: Path) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...], float]:
   names, ndim, time_value = parse_plot_header(plotfile)
   state = read_level_state(plotfile, names, ndim, 0)
   if "rho" not in names or "dye_mass" not in names:
      raise ValueError(f"{plotfile} does not contain rho and dye_mass")
   rho = state[names.index("rho")]
   dye = state[names.index("dye_mass")] / rho
   problo, probhi = parse_plot_bounds(plotfile)
   return dye, problo, probhi, time_value


def cartesian_line_profile(plotfile: Path) -> tuple[np.ndarray, np.ndarray, float]:
   dye, problo, probhi, time_value = level0_dye(plotfile)
   nx, ny = dye.shape[:2]
   x = np.linspace(problo[0] + 0.5 * (probhi[0] - problo[0]) / nx, probhi[0] - 0.5 * (probhi[0] - problo[0]) / nx, nx)
   y = np.linspace(problo[1] + 0.5 * (probhi[1] - problo[1]) / ny, probhi[1] - 0.5 * (probhi[1] - problo[1]) / ny, ny)
   return x, dye[:, int(np.argmin(np.abs(y)))], time_value


def cylindrical_origin_line_profile(plotfile: Path) -> tuple[np.ndarray, np.ndarray, float]:
   dye, problo, probhi, time_value = level0_dye(plotfile)
   nr, nphi = dye.shape[:2]
   r = np.linspace(problo[0] + 0.5 * (probhi[0] - problo[0]) / nr, probhi[0] - 0.5 * (probhi[0] - problo[0]) / nr, nr)
   phi = np.linspace(problo[1] + 0.5 * (probhi[1] - problo[1]) / nphi, probhi[1] - 0.5 * (probhi[1] - problo[1]) / nphi, nphi)
   periodic_delta_0 = np.minimum(np.abs(phi), np.abs((phi - probhi[1]) + problo[1]))
   j0 = int(np.argmin(periodic_delta_0))
   jpi = int(np.argmin(np.abs(phi - np.pi)))
   x = np.concatenate((-r[::-1], r))
   profile = np.concatenate((dye[::-1, jpi], dye[:, j0]))
   return x, profile, time_value


def plot_transmission_profile(
   output: Path,
   cart_x: np.ndarray,
   cart_profile: np.ndarray,
   cyl_x: np.ndarray,
   cyl_profile: np.ndarray,
   expected_peak: float,
   time_value: float,
) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   output.parent.mkdir(parents=True, exist_ok=True)
   fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
   ax.plot(cart_x, cart_profile, label="Cartesian ref", linewidth=2.0)
   ax.plot(cyl_x, cyl_profile, label="cyl r-phi through origin", linestyle=":", linewidth=2.0)
   ax.axvline(expected_peak, color="black", linestyle="--", linewidth=1.2, label="expected peak")
   ax.set_xlabel("x")
   ax.set_ylabel("dye")
   ax.set_title(f"origin transmission, t={time_value:.3f}")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend()
   fig.savefig(output, dpi=180)
   plt.close(fig)


def row_float(row: dict[str, object], key: str) -> float:
   value = row.get(key, "")
   try:
      return float(value)
   except (TypeError, ValueError):
      return float("nan")


def max_finite(values: list[float]) -> str:
   finite = [value for value in values if np.isfinite(value)]
   return f"{max(finite):.8e}" if finite else ""


def run_transmission_case(case: OriginCase, ranks: int, force: bool, timeout: int, make_plots: bool) -> dict[str, object]:
   cart_input, cyl_input, cart_run, cyl_run = transmission_inputs(case, force)
   run_dir = RUN_ROOT / case.name
   cart_result = run_amrex(dim=2, input_path=cart_input, ranks=ranks, timeout=timeout)
   cyl_result = run_amrex(dim=2, input_path=cyl_input, ranks=ranks, timeout=timeout)
   (cart_run / "run.log").write_text(cart_result.stdout)
   (cyl_run / "run.log").write_text(cyl_result.stdout)

   if cart_result.returncode != 0:
      tail = " | ".join(cart_result.stdout.splitlines()[-8:])
      return failure_row(case, f"cartesian returncode={cart_result.returncode}: {tail}", ranks)
   if cyl_result.returncode != 0:
      tail = " | ".join(cyl_result.stdout.splitlines()[-8:])
      return failure_row(case, f"cylindrical returncode={cyl_result.returncode}: {tail}", ranks)
   for completed_run in (cart_run, cyl_run):
      if not (completed_run / "run.in").exists():
         raise RuntimeError(f"solver did not write canonical run configuration: {completed_run / 'run.in'}")

   cart_scalar = scalar_validation_row(
      cart_run,
      case=f"{case.name}_cartesian",
      input_file=cart_run / "run.in",
      policy=case.policy,
      mhd=False,
      dim=2,
      ranks=ranks,
      base_input=repo_relative(cart_input),
      kind="curvilinear_origin",
   )
   cyl_scalar = scalar_validation_row(
      cyl_run,
      case=f"{case.name}_cylindrical",
      input_file=cyl_run / "run.in",
      policy=case.policy,
      mhd=False,
      dim=2,
      ranks=ranks,
      base_input=repo_relative(cyl_input),
      kind="curvilinear_origin",
   )

   cart_x, cart_profile, cart_time = cartesian_line_profile(latest_plotfile(cart_run, require=True))
   cyl_x, cyl_profile, cyl_time = cylindrical_origin_line_profile(latest_plotfile(cyl_run, require=True))
   if abs(cart_time - cyl_time) > 1.0e-12:
      return failure_row(case, f"plotfile times differ: cart={cart_time:g} cyl={cyl_time:g}", ranks)

   reference = np.interp(cyl_x, cart_x, cart_profile)
   active = np.abs(cyl_x) < 0.45
   error = cyl_profile[active] - reference[active]
   norm = np.sqrt(np.mean(reference[active] ** 2))
   l2_rel = float(np.sqrt(np.mean(error ** 2)) / max(norm, 1.0e-14))
   linf = float(np.max(np.abs(error)))
   cart_peak = float(cart_x[int(np.argmax(cart_profile))])
   cyl_peak = float(cyl_x[int(np.argmax(cyl_profile))])
   expected_peak = -0.18 + 0.5 * cyl_time
   peak_dx = abs(cyl_peak - cart_peak)
   expected_dx = abs(cyl_peak - expected_peak)
   status = "PASS" if l2_rel < 0.12 and peak_dx < 0.04 and expected_dx < 0.05 else "FAIL"
   reason = (
      f"l2_rel={l2_rel:.3e}; linf={linf:.3e}; "
      f"peak_dx={peak_dx:.3e}; expected_dx={expected_dx:.3e}"
   )
   scalar_failures = [
      row for row in (cart_scalar, cyl_scalar)
      if row.get("status") != "PASS"
   ]
   if scalar_failures:
      status = "FAIL"
      scalar_reason = " | ".join(str(row.get("reason", "")) for row in scalar_failures)
      reason = f"{reason}; scalar check failed: {scalar_reason}"
   if make_plots:
      plot_transmission_profile(
         run_dir / "origin_transmission_profile.png",
         cart_x,
         cart_profile,
         cyl_x,
         cyl_profile,
         expected_peak,
         cyl_time,
      )
   return {
      "kind": "curvilinear_origin",
      "case": case.name,
      "label": case.name,
      "status": status,
      "reason": reason,
      "dim": "2",
      "input": f"{repo_relative(cart_input)},{repo_relative(cyl_input)}",
      "policy": case.policy.name,
      "ranks": str(ranks),
      "branch": case.branch,
      "steps": cyl_scalar.get("steps", ""),
      "time": f"{cyl_time:.8g}",
      "mass_rel_drift": max_finite([row_float(cart_scalar, "mass_rel_drift"), row_float(cyl_scalar, "mass_rel_drift")]),
      "energy_rel_drift": max_finite([row_float(cart_scalar, "energy_rel_drift"), row_float(cyl_scalar, "energy_rel_drift")]),
      "momentum_abs_drift": max_finite([row_float(cart_scalar, "momentum_abs_drift"), row_float(cyl_scalar, "momentum_abs_drift")]),
      "avg_mcells_per_s": cyl_scalar.get("avg_mcells_per_s", ""),
      "avg_sim_time_per_s": cyl_scalar.get("avg_sim_time_per_s", ""),
      "run_dir": repo_relative(run_dir),
      "log": f"{repo_relative(cart_run / 'run.log')},{repo_relative(cyl_run / 'run.log')}",
      "l2_rel": f"{l2_rel:.8e}",
      "linf_abs": f"{linf:.8e}",
      "peak_dx": f"{peak_dx:.8e}",
      "expected_dx": f"{expected_dx:.8e}",
   }


def run_case(case: OriginCase, ranks: int, force: bool, timeout: int) -> dict[str, object]:
   input_path = write_case_input(case, force)
   result = run_amrex(dim=2, input_path=input_path, ranks=ranks, timeout=timeout)
   run_dir = RUN_ROOT / case.name
   (run_dir / "run.log").write_text(result.stdout)

   if result.returncode != 0:
      tail = " | ".join(result.stdout.splitlines()[-8:])
      return failure_row(case, f"returncode={result.returncode}: {tail}", ranks)
   if not (run_dir / "run.in").exists():
      raise RuntimeError(f"solver did not write canonical run configuration: {run_dir / 'run.in'}")

   row = scalar_validation_row(
      run_dir,
      case=case.name,
      input_file=run_dir / "run.in",
      policy=case.policy,
      mhd=False,
      dim=2,
      ranks=ranks,
      base_input=repo_relative(BASE_INPUT),
      kind="curvilinear_origin",
   )
   row["branch"] = case.branch
   row["log"] = repo_relative(run_dir / "run.log")
   return row


def write_summary(rows: list[dict[str, object]]) -> None:
   write_tsv(
      SUMMARY,
      rows,
      [
         "case",
         "status",
         "reason",
         "branch",
         "dim",
         "input",
         "policy",
         "ranks",
         "steps",
         "time",
         "mass_rel_drift",
         "energy_rel_drift",
         "momentum_abs_drift",
         "avg_mcells_per_s",
         "avg_sim_time_per_s",
         "l2_rel",
         "linf_abs",
         "peak_dx",
         "expected_dx",
         "run_dir",
         "log",
      ],
   )


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("quick", "full"), default="quick")
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--list-cases", action="store_true")
   parser.add_argument("--only", action="append", default=[], help="Run only the named case; may be repeated.")
   parser.add_argument("--plots", action="store_true", help="Write optional human-facing profile plots for comparison cases.")
   parser.add_argument(
      "--run-root",
      type=Path,
      default=RUN_ROOT,
      help="Run/output directory. Relative paths are interpreted from the repository root.",
   )
   args = parser.parse_args()

   os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
   os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
   configure_run_root(args.run_root)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   cases = origin_cases(args.level)
   if args.list_cases:
      for case in cases:
         print(case.name)
      return 0
   if args.only:
      try:
         cases = select_named_cases(cases, args.only)
      except ValueError as exc:
         print(str(exc), file=sys.stderr)
         return 2

   rows: list[dict[str, object]] = []
   for index, case in enumerate(cases, start=1):
      print(f"[{index:02d}/{len(cases):02d}] {case.name}", flush=True)
      if case.kind == "transmission":
         row = run_transmission_case(case, args.ranks, args.force, args.timeout, args.plots)
      else:
         row = run_case(case, args.ranks, args.force, args.timeout)
      rows.append(row)
      print(f"   {row['status']}: {row.get('reason', '')}", flush=True)
      write_summary(rows)

   append_validation_history(HISTORY, rows)
   failed = [row for row in rows if row["status"] != "PASS"]
   print(f"summary: {len(rows) - len(failed)} passed, {len(failed)} failed")
   print(f"wrote {repo_relative(SUMMARY)}")
   print(f"appended {repo_relative(HISTORY)}")
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
