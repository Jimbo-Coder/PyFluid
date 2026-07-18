#!/usr/bin/env python3
"""Validate sampled 3D solid-sphere geometry outputs."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from validation_common import materialize_run_input, parameters_from_assignments, run_mpi


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pyfluid_amrex" / "tools"))
from plotfile_reader import plotfile_at_step, read_aux_field, sorted_aux_plotfiles  # noqa: E402


DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_3d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_3d_amr.in"
DEFAULT_RUN_ROOT = ROOT / "runs" / "current" / "solid_sphere_geometry_validation"
DEFAULT_RESULTS = DEFAULT_RUN_ROOT / "results.tsv"

N_CELL = 40
RADIUS = 0.23
CENTER = 0.5
SAMPLES_PER_DIM = 4
PULSE_AMPLITUDE = 0.04
PULSE_FREQUENCY = 1.0


@dataclass(frozen=True)
class Case:
   name: str
   motion: str
   amplitude: float
   frequency: float
   phase: float
   reference_time: float


CASES = (
   Case("static_sphere", "static", 0.0, 0.0, 0.0, 0.0),
   Case("pulse_expanded_sphere", "pulse", PULSE_AMPLITUDE, PULSE_FREQUENCY, 0.5 * math.pi, -0.25),
)


BASE_OVERRIDES = (
   "geometry.prob_lo=0.0 0.0 0.0",
   "geometry.prob_hi=1.0 1.0 1.0",
   "geometry.is_periodic=0 0 0",
   f"amr.n_cell={N_CELL} {N_CELL} {N_CELL}",
   "amr.max_level=0",
   "amr.max_grid_size=20",
   "amr.blocking_factor=4",
   "pyfluid.physics=hydro",
   "pyfluid.ct=off",
   "pyfluid.ic=uniform",
   "pyfluid.rho0=1.0",
   "pyfluid.pressure0=1.0",
   "pyfluid.bc=reflecting",
   "pyfluid.solid_geometry=circle",
   "pyfluid.solid_motion=static",
   "pyfluid.solid_wall_condition=slip",
   f"pyfluid.solid_center_x={CENTER:.17g}",
   f"pyfluid.solid_center_y={CENTER:.17g}",
   f"pyfluid.solid_center_z={CENTER:.17g}",
   f"pyfluid.solid_radius={RADIUS:.17g}",
   f"pyfluid.solid_volume_fraction_samples_per_dim={SAMPLES_PER_DIM}",
   "pyfluid.solid_swept_reference_time=0.0",
   "pyfluid.max_step=0",
   "pyfluid.scalar_int=0",
   "pyfluid.plot_int=-1",
   "pyfluid.aux_gfs=solid_volume_fraction,solid_swept_volume,solid_aperture1,solid_aperture2,solid_aperture3,solid_phi",
   "pyfluid.aux_gf_int=0",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.checkpoint_int=-1",
)


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def sphere_volume_from_aux_plotfile(plotfile: Path) -> dict[str, float]:
   aux_dir = plotfile.parent
   _volume_fraction, ndim = read_aux_field(aux_dir, "solid_volume_fraction")
   if ndim != 3:
      raise RuntimeError(f"expected 3D aux plotfile, got ndim={ndim}")
   cell_volume = 1.0 / float(N_CELL**3)
   values: dict[str, float] = {}
   for name in ("solid_volume_fraction", "solid_aperture1", "solid_aperture2", "solid_aperture3"):
      field, _ = read_aux_field(aux_dir, name)
      values[name] = float((1.0 - field).sum() * cell_volume)
   swept_volume, _ = read_aux_field(aux_dir, "solid_swept_volume")
   values["solid_swept_volume"] = float(swept_volume.sum() * cell_volume)
   return values


def sphere_radius_at(case: Case, time: float) -> float:
   if case.motion != "pulse":
      return RADIUS
   return RADIUS + case.amplitude * math.sin(case.phase + 2.0 * math.pi * case.frequency * time)


def run_case(case: Case, executable: Path, input_file: Path, run_root: Path, ranks: int, force: bool) -> dict[str, object]:
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
            *BASE_OVERRIDES,
            f"pyfluid.solid_motion={case.motion}",
            f"pyfluid.solid_radius_amplitude={case.amplitude:.17g}",
            f"pyfluid.solid_radius_frequency={case.frequency:.17g}",
            f"pyfluid.solid_radius_phase={case.phase:.17g}",
            f"pyfluid.solid_swept_reference_time={case.reference_time:.17g}",
         ]
      ),
      remove_parameters=("amr.ref_ratio",),
   )
   completed = run_mpi(
      executable,
      [input_path],
      ranks=ranks,
   )
   plotfile = plotfile_at_step(run_dir, 0, "auxGF")
   volumes = sphere_volume_from_aux_plotfile(plotfile) if plotfile is not None else {
      "solid_volume_fraction": math.nan,
      "solid_aperture1": math.nan,
      "solid_aperture2": math.nan,
      "solid_aperture3": math.nan,
      "solid_swept_volume": math.nan,
   }
   radius = sphere_radius_at(case, 0.0)
   reference_radius = sphere_radius_at(case, case.reference_time)
   exact_volume = 4.0 * math.pi * radius**3 / 3.0
   exact_swept_volume = exact_volume - 4.0 * math.pi * reference_radius**3 / 3.0
   volume_keys = ("solid_volume_fraction", "solid_aperture1", "solid_aperture2", "solid_aperture3")
   errors = {key: abs(volumes[key] - exact_volume) for key in volume_keys}
   swept_error = abs(volumes["solid_swept_volume"] - exact_swept_volume)
   failures: list[str] = []
   if completed.returncode != 0:
      failures.append(f"returncode={completed.returncode}")
   for key in volume_keys:
      error = errors[key]
      if not math.isfinite(error) or error > 1.0e-3:
         failures.append(f"{key} error={error:.3e} > 1e-3")
   if not math.isfinite(swept_error) or swept_error > 1.5e-3:
      failures.append(f"solid_swept_volume error={swept_error:.3e} > 1.5e-3")
   return {
      "case": case.name,
      "status": "pass" if not failures else "fail",
      "radius": radius,
      "exact_volume": exact_volume,
      "solid_volume": volumes["solid_volume_fraction"],
      "aperture_x_volume": volumes["solid_aperture1"],
      "aperture_y_volume": volumes["solid_aperture2"],
      "aperture_z_volume": volumes["solid_aperture3"],
      "solid_volume_error": errors["solid_volume_fraction"],
      "aperture_x_error": errors["solid_aperture1"],
      "aperture_y_error": errors["solid_aperture2"],
      "aperture_z_error": errors["solid_aperture3"],
      "swept_volume": volumes["solid_swept_volume"],
      "exact_swept_volume": exact_swept_volume,
      "swept_volume_error": swept_error,
      "run_dir": repo_relative(run_dir),
      "failures": "; ".join(failures),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def write_results(rows: list[dict[str, object]], output: Path) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "case",
      "status",
      "radius",
      "exact_volume",
      "solid_volume",
      "aperture_x_volume",
      "aperture_y_volume",
      "aperture_z_volume",
      "swept_volume",
      "exact_swept_volume",
      "solid_volume_error",
      "aperture_x_error",
      "aperture_y_error",
      "aperture_z_error",
      "swept_volume_error",
      "run_dir",
      "failures",
      "tail",
   ]
   with output.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
   parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--allow-fail", action="store_true")
   args = parser.parse_args()

   if args.force and args.run_root.exists():
      shutil.rmtree(args.run_root)
   rows = [run_case(case, args.executable, args.input, args.run_root, args.ranks, args.force) for case in CASES]
   write_results(rows, args.results)
   for row in rows:
      print(
         f"{str(row['status']).upper():>4} {row['case']}: "
         f"volume={float(row['solid_volume']):.8f}, "
         f"ap=({float(row['aperture_x_volume']):.8f}, {float(row['aperture_y_volume']):.8f}, {float(row['aperture_z_volume']):.8f}), "
         f"exact={float(row['exact_volume']):.8f}, "
         f"swept={float(row['swept_volume']):.8f}, "
         f"exact_swept={float(row['exact_swept_volume']):.8f}, "
         f"errors=({float(row['solid_volume_error']):.3e}, {float(row['aperture_x_error']):.3e}, "
         f"{float(row['aperture_y_error']):.3e}, {float(row['aperture_z_error']):.3e}, "
         f"{float(row['swept_volume_error']):.3e})"
      )
      if row["status"] != "pass":
         print(f"     {row['failures']}")
   print(f"wrote {repo_relative(args.results)}")
   failed = [row for row in rows if row["status"] != "pass"]
   return 0 if args.allow_fail or not failed else 1


if __name__ == "__main__":
   raise SystemExit(main())
