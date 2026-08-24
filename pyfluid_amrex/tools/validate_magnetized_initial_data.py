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
RECOVERY = re.compile(
   r"Relativistic recovery .*?c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)
PREFLIGHT = re.compile(
   rf"Evolution preflight C2P/P2C: active_cells=(\d+) "
   rf"max_conserved_error=({FLOAT}) max_primitive_error=({FLOAT})"
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
      "magnetic_face_representation": "valencia_densitized",
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

   handoff_dir = RUN_ROOT / "sly_tov_pressure_poloidal_handoff"
   if handoff_dir.exists():
      shutil.rmtree(handoff_dir)
   handoff_dir.mkdir(parents=True)
   handoff_input = materialize_run_input(
      BASE_INPUT,
      handoff_dir / "input.in",
      {
         "pyfluid.run_name": handoff_dir.name,
         "pyfluid.output_dir": repo_relative(handoff_dir),
         "initial_data.input_file": repo_relative(artifact),
         "initial_data.output_file": repo_relative(
            handoff_dir / "physical_id"
         ),
         "initial_data.magnetic_seed": "none",
         "initial_data.solve_only": "0",
         "pyfluid.max_step": "0",
         "pyfluid.final_time": "-1",
         "pyfluid.scalar_int": "1",
         "pyfluid.plot_int": "-1",
      },
   )
   handoff = run_amrex(
      dim=3, input_path=handoff_input, ranks=ranks, timeout=timeout
   )
   (handoff_dir / "run.log").write_text(handoff.stdout)
   if handoff.returncode != 0:
      raise RuntimeError(
         "loaded magnetized stellar handoff failed; see "
         f"{repo_relative(handoff_dir / 'run.log')}"
      )
   preflight = PREFLIGHT.search(handoff.stdout)
   if preflight is None:
      raise RuntimeError("loaded magnetized handoff omitted its C2P/P2C preflight")
   active_cells = int(preflight.group(1))
   handoff_conserved_error = float(preflight.group(2))
   handoff_primitive_error = float(preflight.group(3))
   if active_cells != 24**3 or max(
      handoff_conserved_error, handoff_primitive_error
   ) > 1.0e-10:
      raise RuntimeError(
         "loaded magnetized stellar handoff failed its C2P/P2C contract: "
         f"cells={active_cells} conserved={handoff_conserved_error:.3e} "
         f"primitive={handoff_primitive_error:.3e}"
      )
   recovery_matches = RECOVERY.findall(handoff.stdout)
   if not recovery_matches:
      raise RuntimeError("loaded magnetized handoff omitted its C2P summary")
   handoff_fallbacks, handoff_resets = (
      int(value) for value in recovery_matches[-1]
   )
   if handoff_fallbacks != 0 or handoff_resets != 0:
      raise RuntimeError(
         "loaded magnetized stellar handoff required C2P repair: "
         f"{handoff_fallbacks}/{handoff_resets}"
      )

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
      "handoff_active_cells": active_cells,
      "handoff_conserved_error": handoff_conserved_error,
      "handoff_primitive_error": handoff_primitive_error,
      "handoff_c2p_fallbacks": handoff_fallbacks,
      "handoff_atmosphere_resets": handoff_resets,
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
      f"handoff={row['handoff_conserved_error']:.3e}/"
      f"{row['handoff_primitive_error']:.3e} "
      f"artifact={row['artifact_bytes']}B"
   )
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
