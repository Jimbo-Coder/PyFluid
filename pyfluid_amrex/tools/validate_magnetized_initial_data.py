#!/usr/bin/env python3
"""Validate the constraint-consistent pressure-poloidal RMHD ID owner."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np

from plotfile_reader import parse_plot_header, read_level_state
from validation_common import (
   ROOT,
   materialize_run_input,
   repo_relative,
   run_amrex,
   write_tsv,
)


BASE_INPUT = (
   ROOT / "pyfluid_amrex/inputs/tov_xcts_magnetized_poloidal_id_3d.in"
)
RUN_ROOT = ROOT / "runs/current/magnetized_initial_data_validation"
RESULTS = RUN_ROOT / "magnetized_initial_data.tsv"
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MAGNETIC = re.compile(
   rf"pressure-poloidal magnetic ID: updates=(\d+) "
   rf"maximum_pressure=({FLOAT}) normalization=({FLOAT}) "
   rf"maximum_comoving_field=({FLOAT}) divb_linf=({FLOAT})"
)
XCTS = re.compile(
   rf"xcts initial data: matter_iterations=(\d+) .*?"
   rf"hamiltonian_residual=({FLOAT}) lapse_residual=({FLOAT}) "
   rf"momentum_residual=({FLOAT})"
)


def read_metadata(artifact: Path) -> dict[str, str]:
   result: dict[str, str] = {}
   for raw_line in (artifact / "Header.meta").read_text().splitlines():
      line = raw_line.strip()
      if line.startswith("#") and "=" in line:
         key, value = line[1:].split("=", 1)
         result[key.strip()] = value.strip()
   return result


def read_face(path: Path) -> np.ndarray:
   names, dimension, _ = parse_plot_header(path)
   state = read_level_state(path, names, dimension, 0)
   if dimension != 3 or state.shape[0] != 1:
      raise RuntimeError(f"unexpected face-field shape {state.shape}")
   return state[0]


def read_state(path: Path) -> np.ndarray:
   names, dimension, _ = parse_plot_header(path)
   return read_level_state(path, names, dimension, 0)


def validate(*, ranks: int, timeout: int | None, force: bool) -> dict[str, object]:
   if force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   run_dir = RUN_ROOT / "sly_tov_pressure_poloidal"
   if run_dir.exists():
      shutil.rmtree(run_dir)
   artifact = run_dir / "physical_id"
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "input.in",
      {
         "pyfluid.run_name": run_dir.name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "initial_data.output_file": repo_relative(artifact),
      },
   )
   start = time.perf_counter()
   completed = run_amrex(
      dim=3, input_path=input_path, ranks=ranks, timeout=timeout
   )
   seconds = time.perf_counter() - start
   (run_dir / "run.log").write_text(completed.stdout)
   if completed.returncode != 0:
      raise RuntimeError(
         f"magnetized ID run failed; see {repo_relative(run_dir / 'run.log')}"
      )
   magnetic_matches = MAGNETIC.findall(completed.stdout)
   solve = XCTS.search(completed.stdout)
   if not magnetic_matches or solve is None:
      raise RuntimeError("magnetized ID run omitted its magnetic or XCTS summary")
   updates, maximum_pressure, normalization, maximum_field, logged_divergence = (
      magnetic_matches[-1]
   )
   constraint_residual = max(abs(float(solve.group(index))) for index in (2, 3, 4))
   if constraint_residual > 1.0e-10:
      raise RuntimeError(
         f"magnetized XCTS residual is {constraint_residual:.3e}"
      )
   metadata = read_metadata(artifact)
   expected_metadata = {
      "physics": "rmhd",
      "eos": "hybrid_polytrope",
      "has_ct": "1",
      "magnetic_seed": "pressure_poloidal",
      "magnetic_force_balance": "not_solved",
      "source_ic": "tov",
      "spacetime_initial_data": "xcts",
   }
   for key, expected in expected_metadata.items():
      if metadata.get(key) != expected:
         raise RuntimeError(
            f"magnetized artifact metadata {key}={metadata.get(key)!r}"
         )
   requested_field = float(metadata["magnetic_seed_maximum_comoving_field"])
   if not math.isclose(
      float(maximum_field), requested_field, rel_tol=1.0e-12, abs_tol=0.0
   ):
      raise RuntimeError("magnetized ID did not retain its requested normalization")

   b1 = read_face(artifact / "face_b1.h5")
   b2 = read_face(artifact / "face_b2.h5")
   b3 = read_face(artifact / "face_b3.h5")
   if b1.shape != (25, 24, 24) or b2.shape != (24, 25, 24) \
         or b3.shape != (24, 24, 25):
      raise RuntimeError(
         f"unexpected CT face shapes {b1.shape}, {b2.shape}, {b3.shape}"
      )
   spacing = 72.0 / 24.0
   divergence = (
      np.diff(b1, axis=0) + np.diff(b2, axis=1) + np.diff(b3, axis=2)
   ) / spacing
   artifact_divergence = float(np.max(np.abs(divergence)))
   if not any(np.any(field != 0.0) for field in (b1, b2, b3)):
      raise RuntimeError("magnetized artifact face field is identically zero")
   if artifact_divergence > 1.0e-14 or float(logged_divergence) > 1.0e-14:
      raise RuntimeError(
         "magnetized artifact violates the discrete CT divergence contract"
      )

   control_dir = RUN_ROOT / "sly_tov_unmagnetized_control"
   if control_dir.exists():
      shutil.rmtree(control_dir)
   control_artifact = control_dir / "physical_id"
   control_dir.mkdir(parents=True)
   control_input = materialize_run_input(
      BASE_INPUT,
      control_dir / "input.in",
      {
         "pyfluid.run_name": control_dir.name,
         "pyfluid.output_dir": repo_relative(control_dir),
         "initial_data.output_file": repo_relative(control_artifact),
         "initial_data.magnetic_seed": "none",
         "initial_data.magnetic": "zero",
         "pyfluid.physics": "rhd",
         "pyfluid.ct": "off",
      },
   )
   control = run_amrex(
      dim=3, input_path=control_input, ranks=ranks, timeout=timeout
   )
   (control_dir / "run.log").write_text(control.stdout)
   if control.returncode != 0:
      raise RuntimeError(
         "matched unmagnetized XCTS control failed; see "
         f"{repo_relative(control_dir / 'run.log')}"
      )
   magnetized_geometry = read_state(artifact / "adm_geometry.h5")
   control_geometry = read_state(control_artifact / "adm_geometry.h5")
   if magnetized_geometry.shape != control_geometry.shape:
      raise RuntimeError("magnetized and unmagnetized ADM controls changed layout")
   geometry_response = float(
      np.max(np.abs(magnetized_geometry - control_geometry))
   )
   if not 1.0e-8 < geometry_response < 1.0e-3:
      raise RuntimeError(
         "weak magnetic seed did not produce its expected finite coupled-XCTS "
         f"response: {geometry_response:.3e}"
      )

   rejection_dir = RUN_ROOT / "reject_evolution_time_seed"
   if rejection_dir.exists():
      shutil.rmtree(rejection_dir)
   rejection_dir.mkdir(parents=True)
   rejection_input = materialize_run_input(
      BASE_INPUT,
      rejection_dir / "input.in",
      {
         "pyfluid.run_name": rejection_dir.name,
         "pyfluid.output_dir": repo_relative(rejection_dir),
         "initial_data.output_file": repo_relative(
            rejection_dir / "physical_id"
         ),
         "initial_data.solve_only": "0",
      },
   )
   rejected = run_amrex(
      dim=3, input_path=rejection_input, ranks=ranks, timeout=timeout
   )
   (rejection_dir / "run.log").write_text(rejected.stdout)
   expected_rejection = (
      "magnetic initial-data seeds currently require a fresh 3D Cartesian "
      "RMHD+CT stellar coupled-XCTS solve-only artifact without a disk"
   )
   if rejected.returncode == 0 or expected_rejection not in rejected.stdout:
      raise RuntimeError("evolution-time magnetic seed did not fail honestly")

   artifact_bytes = sum(
      path.stat().st_size for path in artifact.rglob("*") if path.is_file()
   )
   return {
      "case": run_dir.name,
      "status": "PASS",
      "ranks": ranks,
      "wall_seconds": seconds,
      "matter_iterations": int(solve.group(1)),
      "magnetic_updates": int(updates),
      "maximum_pressure": float(maximum_pressure),
      "normalization": float(normalization),
      "maximum_comoving_field": float(maximum_field),
      "logged_divb_linf": float(logged_divergence),
      "artifact_divb_linf": artifact_divergence,
      "geometry_response_linf": geometry_response,
      "constraint_residual": constraint_residual,
      "artifact_bytes": artifact_bytes,
      "run_dir": repo_relative(run_dir),
   }


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=180)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   row = validate(ranks=args.ranks, timeout=args.timeout, force=args.force)
   fields = tuple(row)
   write_tsv(RESULTS, [row], fields)
   print(
      f"PASS {row['case']} H/M={row['constraint_residual']:.3e} "
      f"divb={row['artifact_divb_linf']:.3e} "
      f"bmax={row['maximum_comoving_field']:.3e} "
      f"dgeometry={row['geometry_response_linf']:.3e} "
      f"artifact={row['artifact_bytes']}B"
   )
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
