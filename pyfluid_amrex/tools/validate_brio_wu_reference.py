#!/usr/bin/env python3
"""Validate Brio-Wu MHD shock tubes against a finer numerical reference."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pyfluid_amrex" / "tools"))

from plot_amrex_boxed_movie import composite_slice  # noqa: E402
from plotfile_reader import latest_plotfile  # noqa: E402
from validation_common import mpi_prefix
from validation_common import (  # noqa: E402
   OPEN_POLICY,
   append_validation_history,
   read_scalar_rows,
   repo_relative,
   scalar_validation_row,
   write_tsv,
)


EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
BASE_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_2d_amr.in"
RUN_ROOT = ROOT / "runs" / "current" / "brio_wu_reference"
INPUT_ROOT = RUN_ROOT / "inputs"
RESULTS = RUN_ROOT / "results.tsv"
HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"
PLOT_ROOT = ROOT / "plots" / "current" / "IC" / "brio_wu"


@dataclass(frozen=True)
class RunSpec:
   name: str
   direction: str
   n_cell: tuple[int, int]
   max_level: int
   tag_threshold: float


COMMON = (
   "geometry.prob_lo=0.0 0.0",
   "geometry.prob_hi=1.0 1.0",
   "pyfluid.physics=mhd",
   "pyfluid.ct=on",
   "pyfluid.ic=sod_plane",
   "pyfluid.max_step=10000",
   "pyfluid.final_time=0.06",
   "pyfluid.cfl=0.9",
   "pyfluid.gamma=2.0",
   "pyfluid.ng=2",
   "pyfluid.recon=pwl",
   "pyfluid.riemann=hlld",
   "pyfluid.timestep=ssprk3",
   "pyfluid.diffusion=none",
   "pyfluid.viscosity=0.0",
   "pyfluid.resistivity=0.0",
   "pyfluid.scalar_int=4",
   "pyfluid.plot_int=10000",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_int=-1",
   "pyfluid.aux_gf_compute_int=-1",
   "amr.ref_ratio=2 2",
   "amr.max_grid_size=32",
   "amr.blocking_factor=8",
   "amr.regrid_int=2",
   "amr.n_error_buf=4 4",
   "amr.tag_metric=auto",
   "amr.state_interp=quartic-conservative",
   "amr.initial_refine=none",
   "amr.initial_refine_levels=0",
   "amr.initial_refine_fixed_levels=0",
)


SPECS = (
   RunSpec("x_uniform64", "x", (64, 8), 0, 0.02),
   RunSpec("x_uniform256", "x", (256, 8), 0, 0.02),
   RunSpec("y_uniform64", "y", (8, 64), 0, 0.02),
)

AMR_CT_SPECS = (
   RunSpec("x_amr64_l2", "x", (64, 8), 2, 0.02),
   RunSpec("y_amr64_l2", "y", (8, 64), 2, 0.02),
)


COMPONENTS = ("rho", "pressure", "v1", "v2", "b1", "b2")
ROTATE_Y_TO_X = {
   "rho": "rho",
   "pressure": "pressure",
   "v1": "v2",
   "v2": "v1",
   "b1": "b2",
   "b2": "b1",
}


def direction_overrides(direction: str) -> tuple[str, ...]:
   if direction == "x":
      return (
         "geometry.is_periodic=0 1",
         "pyfluid.bc=outflow",
         "pyfluid.bc_xlo=outflow",
         "pyfluid.bc_xhi=outflow",
         "pyfluid.bc_ylo=periodic",
         "pyfluid.bc_yhi=periodic",
         "pyfluid.shock_x0=0.5",
         "pyfluid.shock_y0=0.5",
         "pyfluid.shock_nx=1.0",
         "pyfluid.shock_ny=0.0",
         "pyfluid.b1_left=0.75",
         "pyfluid.b2_left=1.0",
         "pyfluid.b3_left=0.0",
         "pyfluid.b1_right=0.75",
         "pyfluid.b2_right=-1.0",
         "pyfluid.b3_right=0.0",
      )
   if direction == "y":
      return (
         "geometry.is_periodic=1 0",
         "pyfluid.bc=outflow",
         "pyfluid.bc_xlo=periodic",
         "pyfluid.bc_xhi=periodic",
         "pyfluid.bc_ylo=outflow",
         "pyfluid.bc_yhi=outflow",
         "pyfluid.shock_x0=0.5",
         "pyfluid.shock_y0=0.5",
         "pyfluid.shock_nx=0.0",
         "pyfluid.shock_ny=1.0",
         "pyfluid.b1_left=1.0",
         "pyfluid.b2_left=0.75",
         "pyfluid.b3_left=0.0",
         "pyfluid.b1_right=-1.0",
         "pyfluid.b2_right=0.75",
         "pyfluid.b3_right=0.0",
      )
   raise ValueError(direction)


def state_overrides() -> tuple[str, ...]:
   return (
      "pyfluid.rho_left=1.0",
      "pyfluid.pressure_left=1.0",
      "pyfluid.v1_left=0.0",
      "pyfluid.v2_left=0.0",
      "pyfluid.v3_left=0.0",
      "pyfluid.rho_right=0.125",
      "pyfluid.pressure_right=0.1",
      "pyfluid.v1_right=0.0",
      "pyfluid.v2_right=0.0",
      "pyfluid.v3_right=0.0",
   )


def run_dir(spec: RunSpec) -> Path:
   return RUN_ROOT / spec.name


def write_input(spec: RunSpec) -> Path:
   output_dir = run_dir(spec)
   input_dir = INPUT_ROOT
   input_dir.mkdir(parents=True, exist_ok=True)
   output_dir.mkdir(parents=True, exist_ok=True)
   lines = BASE_INPUT.read_text().splitlines()
   lines.append("")
   lines.append("# validate_brio_wu_reference.py overrides")
   lines.append(f"pyfluid.run_name=brio_wu_{spec.name}")
   lines.append(f"pyfluid.output_dir={repo_relative(output_dir)}")
   lines.extend(COMMON)
   lines.extend(direction_overrides(spec.direction))
   lines.extend(state_overrides())
   lines.append(f"amr.n_cell={spec.n_cell[0]} {spec.n_cell[1]}")
   lines.append(f"amr.max_level={spec.max_level}")
   lines.append(f"amr.tag_threshold={spec.tag_threshold:.17g}")
   path = input_dir / f"{spec.name}.in"
   path.write_text("\n".join(lines) + "\n")
   return path


def run_spec(spec: RunSpec, ranks: int, force: bool, timeout: int) -> tuple[int, Path, Path, str]:
   output_dir = run_dir(spec)
   if force and output_dir.exists():
      shutil.rmtree(output_dir)
   input_path = write_input(spec)
   if not force:
      try:
         latest_plotfile(output_dir, require=True)
         return 0, output_dir, input_path, ""
      except FileNotFoundError:
         pass
   command = [*mpi_prefix(ranks), str(EXE), str(input_path)]
   completed = subprocess.run(
      command,
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      timeout=timeout,
      env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
      check=False,
   )
   output_dir.mkdir(parents=True, exist_ok=True)
   (output_dir / "run.log").write_text(completed.stdout)
   return completed.returncode, output_dir, input_path, "\n".join(completed.stdout.splitlines()[-8:])


def profile(plotfile: Path, component: str, direction: str, max_pixels: int) -> np.ndarray:
   image, _boxes, _time, _ndim, _names, _xlim, _ylim = composite_slice(
      plotfile, component, 2.0, 2, 0.5, False, None, max_pixels, (0.0, 1.0), (0.0, 1.0)
   )
   if direction == "x":
      return np.nanmean(image, axis=0)
   return np.nanmean(image, axis=1)


def compare_profiles(
   reference: Path,
   candidate: Path,
   *,
   reference_direction: str,
   candidate_direction: str,
   max_pixels: int,
) -> dict[str, object]:
   row: dict[str, object] = {
      "max_rel_l2": 0.0,
      "max_rel_linf": 0.0,
      "worst_component": "",
   }
   for component in COMPONENTS:
      candidate_component = ROTATE_Y_TO_X[component] if candidate_direction == "y" else component
      ref = profile(reference, component, reference_direction, max_pixels)
      cand = profile(candidate, candidate_component, candidate_direction, max_pixels)
      if ref.shape != cand.shape:
         raise ValueError(f"{component}: profile shapes differ {ref.shape} != {cand.shape}")
      mask = np.isfinite(ref) & np.isfinite(cand)
      if not np.any(mask):
         raise ValueError(f"{component}: no finite profile overlap")
      diff = cand - ref
      scale = max(float(np.sqrt(np.mean(ref[mask] ** 2))), 1.0e-300)
      rel_l2 = float(np.sqrt(np.mean(diff[mask] ** 2))) / scale
      rel_linf = float(np.max(np.abs(diff[mask]))) / scale
      row[f"{component}_rel_l2"] = rel_l2
      row[f"{component}_rel_linf"] = rel_linf
      if rel_linf >= float(row["max_rel_linf"]):
         row["worst_component"] = component
      row["max_rel_l2"] = max(float(row["max_rel_l2"]), rel_l2)
      row["max_rel_linf"] = max(float(row["max_rel_linf"]), rel_linf)
   return row


def average_positive(rows: list[dict[str, str]], key: str) -> float:
   values = []
   for row in rows:
      try:
         value = float(row.get(key, "nan"))
      except ValueError:
         value = math.nan
      if math.isfinite(value) and value > 0.0:
         values.append(value)
   return sum(values) / len(values) if values else math.nan


def plot_profiles(plotfiles: dict[str, Path], output: Path, max_pixels: int) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   x = np.linspace(0.0, 1.0, max_pixels, endpoint=False) + 0.5 / max_pixels
   fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True, sharex=True)
   panels = (("rho", "rho"), ("pressure", "P"), ("b2", "By"), ("v1", "vx"))
   styles = (
      ("x_uniform256", "reference", "k", ":"),
      ("x_uniform64", "uniform64", "tab:orange", "-"),
      ("x_amr64_l2", "AMR L2 CT", "tab:blue", "--"),
   )
   for ax, (component, label) in zip(axes.ravel(), panels):
      for name, run_label, color, linestyle in styles:
         if name not in plotfiles:
            continue
         values = profile(plotfiles[name], component, "x", max_pixels)
         ax.plot(x, values, color=color, linestyle=linestyle, linewidth=1.7, label=run_label)
      ax.set_ylabel(label)
      ax.grid(True, which="major", alpha=0.25)
   axes[-1, 0].set_xlabel("x")
   axes[-1, 1].set_xlabel("x")
   axes[0, 0].legend()
   fig.suptitle("Brio-Wu MHD shock reference: PWL/HLLD/SSPRK3/CT, CFL=0.9")
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=170)
   plt.close(fig)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--max-pixels", type=int, default=256)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--coarse-l2-tol", type=float, default=3.5e-1)
   parser.add_argument("--amr-l2-tol", type=float, default=2.5e-1)
   parser.add_argument("--rotation-l2-tol", type=float, default=1.0e-10)
   parser.add_argument("--amr-rotation-l2-tol", type=float, default=5.0e-3)
   parser.add_argument(
      "--include-amr-ct",
      action="store_true",
      help="include the currently strict AMR+CT Brio-Wu stress probe",
   )
   parser.add_argument("--skip-plot", action="store_true")
   args = parser.parse_args()

   if args.force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   plotfiles: dict[str, Path] = {}
   rows: list[dict[str, object]] = []
   specs = SPECS + (AMR_CT_SPECS if args.include_amr_ct else ())
   for spec in specs:
      code, output_dir, input_path, tail = run_spec(spec, args.ranks, args.force, args.timeout)
      if code != 0:
         rows.append({
            "case": spec.name,
            "kind": "run",
            "status": "FAIL",
            "reason": tail,
            "ranks": args.ranks,
            "run_dir": repo_relative(output_dir),
         })
         write_tsv(RESULTS, rows, ["case", "kind", "status", "reason", "ranks", "run_dir"])
         append_validation_history(HISTORY, [{"kind": "brio_wu_reference", **row} for row in rows])
         return 1
      plotfiles[spec.name] = latest_plotfile(output_dir, require=True)
      scalar = scalar_validation_row(
         output_dir,
         case=spec.name,
         input_file=input_path,
         checks=OPEN_POLICY,
         mhd=True,
         dim=2,
         ranks=args.ranks,
         base_input="generated Brio-Wu sod_plane",
         kind="brio_wu_scalar",
      )
      rows.append(scalar)

   comparisons = (
      ("x_uniform64_vs_x_uniform256", "x_uniform256", "x_uniform64", "x", args.coarse_l2_tol),
      ("y_uniform64_rotated_vs_x_uniform64", "x_uniform64", "y_uniform64", "y", args.rotation_l2_tol),
   )
   amr_comparisons = (
      ("x_amr64_l2_vs_x_uniform256", "x_uniform256", "x_amr64_l2", "x", args.amr_l2_tol),
      ("y_amr64_l2_rotated_vs_x_amr64_l2", "x_amr64_l2", "y_amr64_l2", "y", args.amr_rotation_l2_tol),
   )
   comparison_rows = []
   for name, reference, candidate, candidate_direction, tolerance in comparisons + (amr_comparisons if args.include_amr_ct else ()):
      values = compare_profiles(
         plotfiles[reference],
         plotfiles[candidate],
         reference_direction="x",
         candidate_direction=candidate_direction,
         max_pixels=args.max_pixels,
      )
      failures = []
      if float(values["max_rel_l2"]) > tolerance:
         failures.append(f"max_rel_l2={float(values['max_rel_l2']):.3e} > {tolerance:.3e}")
      row = {
         "case": name,
         "kind": "comparison",
         "status": "PASS" if not failures else "FAIL",
         "reason": "; ".join(failures),
         "ranks": args.ranks,
         "reference": reference,
         "candidate": candidate,
         "tolerance_l2": tolerance,
         **values,
      }
      comparison_rows.append(row)
      rows.append(row)

   if args.include_amr_ct:
      coarse = next(row for row in comparison_rows if row["case"] == "x_uniform64_vs_x_uniform256")
      amr = next(row for row in comparison_rows if row["case"] == "x_amr64_l2_vs_x_uniform256")
      improvement = float(coarse["max_rel_l2"]) / max(float(amr["max_rel_l2"]), 1.0e-300)
      rows.append({
         "case": "amr_improves_over_uniform64",
         "kind": "improvement",
         "status": "PASS" if improvement >= 1.1 else "FAIL",
         "reason": "" if improvement >= 1.1 else f"improvement={improvement:.3e} < 1.1",
         "ranks": args.ranks,
         "improvement_factor": improvement,
      })

   if not args.skip_plot:
      plot_profiles(plotfiles, PLOT_ROOT / "brio_wu_reference_profiles_001.png", args.max_pixels)

   fields = sorted({key for row in rows for key in row})
   write_tsv(RESULTS, rows, fields)
   append_validation_history(HISTORY, [{"kind": "brio_wu_reference", **row} for row in rows])
   failed = [row for row in rows if row.get("status") != "PASS"]
   for row in rows:
      if row.get("kind") == "comparison":
         print(
            f"{row['status']:>4} {row['case']}: "
            f"max_rel_l2={float(row['max_rel_l2']):.3e} "
            f"max_rel_linf={float(row['max_rel_linf']):.3e} "
            f"worst={row['worst_component']}"
         )
      elif row.get("kind") == "improvement":
         print(f"{row['status']:>4} {row['case']}: factor={float(row['improvement_factor']):.3e}")
   print(f"wrote {repo_relative(RESULTS)}")
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
