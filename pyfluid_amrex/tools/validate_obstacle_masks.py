#!/usr/bin/env python3
"""Validate obstacle-mask robustness and measure skipped-cell throughput.

The cases below are the single place to define this obstacle mask sweep. They
all run a static uniform hydro state, so mass/energy should stay fixed to
roundoff while the solver still exercises the same masked evolution kernels.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from validation_common import materialize_run_input, parameters_from_assignments, run_mpi


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "forward_step_hydro_2d_obstacle.in"
DEFAULT_RUN_ROOT = ROOT / "runs" / "current" / "obstacle_mask_validation"
DEFAULT_RESULTS = DEFAULT_RUN_ROOT / "results.tsv"
DEFAULT_PLOT = ROOT / "plots" / "current" / "scans" / "obstacle_mask_performance_004.png"
DEFAULT_N_CELL = (1024, 512)

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
LAYOUT_AVERAGE_RE = re.compile(r"layout_average=([0-9.eE+-]+)\s+layout_Mcells/s")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")


@dataclass(frozen=True)
class ObstacleCase:
   name: str
   label: str
   box_lo: tuple[tuple[float, float], ...]
   box_hi: tuple[tuple[float, float], ...]
   min_speedup: float
   description: str


def checkerboard_boxes(nx: int = 4, ny: int = 4) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
   lo: list[tuple[float, float]] = []
   hi: list[tuple[float, float]] = []
   dx = 3.0 / nx
   dy = 1.0 / ny
   for j in range(ny):
      for i in range(nx):
         if (i + j) % 2 == 0:
            lo.append((i * dx, j * dy))
            hi.append(((i + 1) * dx, (j + 1) * dy))
   return tuple(lo), tuple(hi)


CHECKER_LO, CHECKER_HI = checkerboard_boxes()

CASES = (
   ObstacleCase(
      name="empty",
      label="empty mask",
      box_lo=((10.0, 10.0),),
      box_hi=((11.0, 11.0),),
      min_speedup=0.98,
      description="inactive off-domain box; baseline no-solid behavior",
   ),
   ObstacleCase(
      name="bottom_half",
      label="bottom half solid",
      box_lo=((0.0, 0.0),),
      box_hi=((3.0, 0.5),),
      min_speedup=1.40,
      description="large contiguous obstacle should skip most work in covered FABs",
   ),
   ObstacleCase(
      name="left_half",
      label="left half solid",
      box_lo=((0.0, 0.0),),
      box_hi=((1.5, 1.0),),
      min_speedup=1.40,
      description="same covered fraction as bottom_half; catches orientation-dependent MPI imbalance",
   ),
   ObstacleCase(
      name="small_center",
      label="small obstacle",
      box_lo=((1.35, 0.40),),
      box_hi=((1.65, 0.60),),
      min_speedup=0.85,
      description="small obstacle should not cause a major overhead regression",
   ),
   ObstacleCase(
      name="checkerboard",
      label="4x4 checkerboard",
      box_lo=CHECKER_LO,
      box_hi=CHECKER_HI,
      min_speedup=0.80,
      description="pathological mixed-mask layout stresses robustness more than speed",
   ),
)


def repo_relative(path: Path) -> Path:
   absolute = path if path.is_absolute() else ROOT / path
   return absolute.resolve().relative_to(ROOT)


def flatten_boxes(boxes: tuple[tuple[float, float], ...]) -> str:
   return " ".join(f"{value:.16g}" for box in boxes for value in box)


def read_last_scalar(path: Path) -> dict[str, float]:
   with path.open(newline="") as f:
      rows = list(csv.DictReader(f, delimiter="\t"))
   if not rows:
      raise RuntimeError(f"{path} is empty")
   return {key: float(value) for key, value in rows[-1].items()}


def read_profile_summary(run_dir: Path) -> dict[str, float | str]:
   profile_path = run_dir / "profile.tsv"
   if not profile_path.exists():
      return {
         "dominant_stage": "",
         "dominant_stage_fraction": math.nan,
         "flux_fraction": math.nan,
         "primitive_cache_fraction": math.nan,
         "update_fraction": math.nan,
      }

   with profile_path.open(newline="") as f:
      rows = list(csv.DictReader(f, delimiter="\t"))

   fraction_key = "fraction_of_run_wall" if rows and "fraction_of_run_wall" in rows[0] else "fraction"
   fractions: dict[str, float] = {}
   for row in rows:
      stage = row.get("stage", "")
      if not stage:
         continue
      try:
         fractions[stage] = float(row.get(fraction_key, "nan"))
      except ValueError:
         fractions[stage] = math.nan

   finite = {stage: value for stage, value in fractions.items() if math.isfinite(value)}
   dominant_stage = ""
   dominant_fraction = math.nan
   if finite:
      dominant_stage, dominant_fraction = max(finite.items(), key=lambda item: item[1])

   return {
      "dominant_stage": dominant_stage,
      "dominant_stage_fraction": dominant_fraction,
      "flux_fraction": fractions.get("flux", fractions.get("fused_rhs", math.nan)),
      "primitive_cache_fraction": fractions.get("primitive_cache", math.nan),
      "update_fraction": fractions.get("update", math.nan),
   }


def run_case(
   case: ObstacleCase,
   executable: Path,
   input_file: Path,
   run_root: Path,
   ranks: int,
   n_cell: tuple[int, int],
   force: bool,
) -> dict[str, object]:
   run_dir = run_root / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)

   input_path = materialize_run_input(
      input_file,
      run_dir / "run.in",
      parameters_from_assignments(
         [
            f"pyfluid.output_dir={repo_relative(run_dir)}",
            f"amr.n_cell={n_cell[0]} {n_cell[1]}",
            "amr.max_level=0",
            "amr.max_grid_size=64",
            "amr.blocking_factor=16",
            "pyfluid.max_step=128",
            "pyfluid.final_time=-1",
            "pyfluid.cfl=0.9",
            "pyfluid.rho0=1.4",
            "pyfluid.pressure0=1.0",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
            "pyfluid.bc=reflecting",
            "pyfluid.bc_xlo=reflecting",
            "pyfluid.bc_xhi=reflecting",
            "pyfluid.bc_ylo=reflecting",
            "pyfluid.bc_yhi=reflecting",
            "pyfluid.recon=pwc",
            "pyfluid.riemann=hllc",
            "pyfluid.timestep=euler",
            "pyfluid.solid_geometry=boxes",
            f"pyfluid.solid_box_lo={flatten_boxes(case.box_lo)}",
            f"pyfluid.solid_box_hi={flatten_boxes(case.box_hi)}",
            "pyfluid.scalar_int=128",
            "pyfluid.plot_int=-1",
            "pyfluid.aux_gfs=none",
            "pyfluid.aux_gf_int=-1",
            "pyfluid.aux_gf_compute_int=-1",
            "pyfluid.checkpoint_int=-1",
            "pyfluid.profile=on",
            "pyfluid.profile_detail=basic",
         ]
      ),
   )
   completed = run_mpi(
      executable,
      [input_path],
      ranks=ranks,
   )

   average_match = AVERAGE_RE.search(completed.stdout)
   layout_average_match = LAYOUT_AVERAGE_RE.search(completed.stdout)
   sim_time_match = SIM_TIME_RE.search(completed.stdout)
   mcells_s = float(average_match.group(1)) if average_match else math.nan
   layout_mcells_s = float(layout_average_match.group(1)) if layout_average_match else math.nan
   sim_time_per_s = float(sim_time_match.group(1)) if sim_time_match else math.nan
   scalar_path = run_dir / "scalars.tsv"
   final = read_last_scalar(scalar_path) if scalar_path.exists() else {}
   profile = read_profile_summary(run_dir)

   row: dict[str, object] = {
      "case": case.name,
      "label": case.label,
      "description": case.description,
      "ranks": ranks,
      "n_cell": f"{n_cell[0]}x{n_cell[1]}",
      "boxes": len(case.box_lo),
      "mcells_s": mcells_s,
      "layout_mcells_s": layout_mcells_s,
      "sim_time_per_s": sim_time_per_s,
      "returncode": completed.returncode,
      "run_dir": repo_relative(run_dir),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
      **profile,
   }
   for key in (
      "mass_rel_drift",
      "energy_rel_drift",
      "mom1",
      "mom2",
      "mom3",
      "min_rho",
      "min_pressure",
   ):
      row[key] = final.get(key, math.nan)
   return row


def add_status(rows: list[dict[str, object]]) -> None:
   baseline = next(row for row in rows if row["case"] == "empty")
   baseline_layout_mcells = float(baseline["layout_mcells_s"])
   for row, case in zip(rows, CASES, strict=True):
      mcells_s = float(row["mcells_s"])
      layout_mcells_s = float(row["layout_mcells_s"])
      layout_speedup = layout_mcells_s / baseline_layout_mcells if baseline_layout_mcells > 0.0 else math.nan
      row["layout_speedup_vs_empty"] = layout_speedup
      failures = []
      if int(row["returncode"]) != 0:
         failures.append(f"returncode={row['returncode']}")
      if not math.isfinite(mcells_s):
         failures.append("missing active throughput")
      if not math.isfinite(layout_mcells_s):
         failures.append("missing layout throughput")
      for key in ("mass_rel_drift", "energy_rel_drift"):
         value = float(row[key])
         if not math.isfinite(value) or value > 1.0e-11:
            failures.append(f"{key}={value:.3e}")
      for key, floor in (("min_rho", 1.0e-12), ("min_pressure", 1.0e-12)):
         value = float(row[key])
         if not math.isfinite(value) or value <= floor:
            failures.append(f"{key}={value:.3e}")
      if math.isfinite(layout_speedup) and layout_speedup < case.min_speedup:
         failures.append(f"layout_speedup={layout_speedup:.3f} < {case.min_speedup:.3f}")
      row["min_speedup"] = case.min_speedup
      row["status"] = "pass" if not failures else "fail"
      row["failures"] = "; ".join(failures)


def write_results(rows: list[dict[str, object]], output: Path) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "case",
      "label",
      "status",
      "ranks",
      "n_cell",
      "boxes",
      "mcells_s",
      "layout_mcells_s",
      "sim_time_per_s",
      "layout_speedup_vs_empty",
      "min_speedup",
      "mass_rel_drift",
      "energy_rel_drift",
      "mom1",
      "mom2",
      "mom3",
      "min_rho",
      "min_pressure",
      "dominant_stage",
      "dominant_stage_fraction",
      "flux_fraction",
      "primitive_cache_fraction",
      "update_fraction",
      "run_dir",
      "description",
      "failures",
      "tail",
   ]
   with output.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      for row in rows:
         writer.writerow(row)


def write_plot(rows: list[dict[str, object]], output: Path, n_cell: tuple[int, int], ranks: int) -> bool:
   mpl_config = ROOT / ".matplotlib"
   mpl_config.mkdir(exist_ok=True)
   os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
   try:
      import matplotlib.pyplot as plt
   except ImportError:
      return False

   output.parent.mkdir(parents=True, exist_ok=True)
   labels = [str(row["label"]) for row in rows]
   active_speeds = [float(row["mcells_s"]) for row in rows]
   layout_speeds = [float(row["layout_mcells_s"]) for row in rows]
   layout_speedups = [float(row["layout_speedup_vs_empty"]) for row in rows]

   fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
   x = list(range(len(rows)))
   width = 0.38
   axes[0].bar([value - width / 2 for value in x], active_speeds, width=width, color="tab:blue", label="active")
   axes[0].bar([value + width / 2 for value in x], layout_speeds, width=width, color="tab:orange", label="layout")
   axes[0].set_xticks(list(x), labels, rotation=18, ha="right")
   axes[0].set_ylabel("Mcells/s")
   axes[0].set_title("Obstacle mask throughput")
   axes[0].legend()
   axes[0].grid(True, axis="y", alpha=0.25)

   axes[1].bar(x, layout_speedups, color="tab:green")
   axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.5)
   axes[1].set_xticks(list(x), labels, rotation=18, ha="right")
   axes[1].set_ylabel("layout speedup vs empty mask")
   axes[1].set_title("Skipped-cell speedup")
   axes[1].grid(True, axis="y", alpha=0.25)

   fig.suptitle(
      f"Obstacle masks: 2D hydro, {n_cell[0]}x{n_cell[1]}, ranks={ranks}, "
      "static uniform IC, reflecting BC, PWC/HLLC/Euler, CFL=0.9, max_level=0"
   )
   fig.savefig(output, dpi=160)
   plt.close(fig)
   return True


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
   parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
   parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--n-cell", type=int, nargs=2, default=DEFAULT_N_CELL, metavar=("NX", "NY"))
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--allow-fail", action="store_true", help="write outputs and return success even if a case fails")
   parser.add_argument("--skip-plot", action="store_true", help="write TSV only")
   args = parser.parse_args()

   if len(CHECKER_LO) > 32:
      raise SystemExit("checkerboard case exceeds MaxSolidBoxes=32")
   if args.force and args.run_root.exists():
      shutil.rmtree(args.run_root)

   n_cell = (int(args.n_cell[0]), int(args.n_cell[1]))
   rows = [
      run_case(case, args.executable, args.input, args.run_root, args.ranks, n_cell, args.force)
      for case in CASES
   ]
   add_status(rows)
   write_results(rows, args.output)
   wrote_plot = False if args.skip_plot else write_plot(rows, args.plot_output, n_cell, args.ranks)

   for row in rows:
      print(
         f"{str(row['status']).upper():>4} {row['label']}: "
         f"{float(row['mcells_s']):.2f} active Mcells/s, "
         f"{float(row['layout_mcells_s']):.2f} layout Mcells/s, "
         f"layout_speedup={float(row['layout_speedup_vs_empty']):.2f}, "
         f"sim_time/s={float(row['sim_time_per_s']):.4g}, "
         f"mass={float(row['mass_rel_drift']):.2e}, "
         f"energy={float(row['energy_rel_drift']):.2e}, "
         f"top={row['dominant_stage']}:{float(row['dominant_stage_fraction']):.2f}"
      )
      if row["status"] != "pass":
         print(f"     {row['failures']}")
   print(f"wrote {repo_relative(args.output)}")
   if wrote_plot:
      print(f"wrote {repo_relative(args.plot_output)}")
   elif not args.skip_plot:
      print("skipped plot: matplotlib is not available in this Python environment")

   failed = [row for row in rows if row["status"] != "pass"]
   return 0 if args.allow_fail or not failed else 1


if __name__ == "__main__":
   raise SystemExit(main())
