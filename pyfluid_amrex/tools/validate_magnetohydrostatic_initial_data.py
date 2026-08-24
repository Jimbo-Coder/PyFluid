#!/usr/bin/env python3
"""Validate the coupled poloidal Grad-Shafranov stellar-ID owner."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np

from plotfile_reader import parse_plot_header, read_level_state
from physical_id_metadata import read_physical_id_metadata
from validation_common import (
   ROOT,
   materialize_run_input,
   repo_relative,
   run_amrex,
   write_tsv,
)


BASE_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/tov_xcts_magnetohydrostatic_poloidal_id_3d.in"
)
RNS_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/rotating_star_xcts_magnetohydrostatic_poloidal_id_3d.in"
)
RUN_ROOT = ROOT / "runs/current/magnetohydrostatic_initial_data_validation"
RESULTS = RUN_ROOT / "magnetohydrostatic_initial_data.tsv"
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MAGNETIC = re.compile(
   rf"poloidal Grad-Shafranov magnetic ID: updates=(\d+) "
   rf"relative_potential_update=({FLOAT}) linear_residual=({FLOAT}) "
   rf"equation_relative_residual=({FLOAT}) bernoulli_residual=({FLOAT}) "
   rf"maximum_comoving_field=({FLOAT}) divb_linf=({FLOAT})"
)
XCTS = re.compile(
   rf"xcts initial data: matter_iterations=(\d+) .*?"
   rf"hamiltonian_residual=({FLOAT}) lapse_residual=({FLOAT}) "
   rf"momentum_residual=({FLOAT})"
)
ROTATING = re.compile(
   rf"rotating star: omega=({FLOAT}) .*?maximum_velocity=({FLOAT}) .*?"
   rf"mass_shedding_indicator=({FLOAT}) .*?adm_mass=({FLOAT}) .*?"
   rf"baryonic_mass=({FLOAT}) .*?angular_momentum=({FLOAT}) .*?"
   rf"dimensionless_spin=({FLOAT})"
)


read_metadata = read_physical_id_metadata


def read_state(path: Path) -> np.ndarray:
   names, dimension, _ = parse_plot_header(path)
   return read_level_state(path, names, dimension, 0)


def read_face(path: Path) -> np.ndarray:
   state = read_state(path)
   if state.shape[0] != 1:
      raise RuntimeError(f"unexpected face-field shape {state.shape}")
   return state[0]


ARTIFACT_FIELDS = (
   "physical_fluid.h5",
   "adm_geometry.h5",
   "evolution_fluid_cache.h5",
   "evolution_spacetime_cache.h5",
   "face_b1.h5",
   "face_b2.h5",
   "face_b3.h5",
)


def artifact_field_differences(
   first: Path, second: Path
) -> dict[str, tuple[float, float]]:
   differences: dict[str, tuple[float, float]] = {}
   for name in ARTIFACT_FIELDS:
      first_state = read_state(first / name)
      second_state = read_state(second / name)
      if first_state.shape != second_state.shape:
         raise RuntimeError(
            f"round-trip field {name} changed shape from "
            f"{first_state.shape} to {second_state.shape}"
         )
      absolute = float(np.max(np.abs(first_state - second_state)))
      scale = max(float(np.max(np.abs(first_state))), 1.0e-300)
      differences[name] = (absolute, absolute / scale)
   return differences


def run_case(
   name: str,
   *,
   ranks: int,
   timeout: int | None,
   parameters: dict[str, str] | None = None,
   base_input: Path = BASE_INPUT,
) -> tuple[Path, Path, str, float]:
   run_dir = RUN_ROOT / name
   if run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True)
   artifact = run_dir / "physical_id"
   values = {
      "pyfluid.run_name": name,
      "pyfluid.output_dir": repo_relative(run_dir),
      "initial_data.output_file": repo_relative(artifact),
   }
   if parameters:
      values.update(parameters)
   input_path = materialize_run_input(
      base_input, run_dir / "input.in", values
   )
   start = time.perf_counter()
   completed = run_amrex(
      dim=3, input_path=input_path, ranks=ranks, timeout=timeout
   )
   seconds = time.perf_counter() - start
   (run_dir / "run.log").write_text(completed.stdout)
   if completed.returncode != 0:
      raise RuntimeError(
         f"{name} failed; see {repo_relative(run_dir / 'run.log')}"
      )
   return run_dir, artifact, completed.stdout, seconds


def maximum_constraint_residual(text: str) -> tuple[int, float]:
   solve = XCTS.search(text)
   if solve is None:
      raise RuntimeError("magnetohydrostatic run omitted its XCTS summary")
   residual = max(abs(float(solve.group(index))) for index in (2, 3, 4))
   return int(solve.group(1)), residual


def cell_average_faces(faces: tuple[np.ndarray, ...]) -> np.ndarray:
   return np.stack(
      (
         0.5 * (faces[0][:-1, :, :] + faces[0][1:, :, :]),
         0.5 * (faces[1][:, :-1, :] + faces[1][:, 1:, :]),
         0.5 * (faces[2][:, :, :-1] + faces[2][:, :, 1:]),
      ),
      axis=0,
   )


def validate_equilibrium(
   *, ranks: int, timeout: int | None
) -> dict[str, object]:
   name = "sly_tov_nonlinear_poloidal_grad_shafranov"
   run_dir, artifact, text, seconds = run_case(
      name, ranks=ranks, timeout=timeout
   )
   matches = MAGNETIC.findall(text)
   if not matches:
      raise RuntimeError("magnetohydrostatic run omitted its magnetic summary")
   (
      updates,
      potential_update,
      linear_residual,
      equation_residual,
      bernoulli_residual,
      maximum_field,
      logged_divergence,
   ) = matches[-1]
   matter_iterations, constraint_residual = maximum_constraint_residual(text)
   limits = {
      "relative potential update": (float(potential_update), 1.1e-6),
      "linear Grad-Shafranov residual": (float(linear_residual), 1.0e-12),
      "independent Grad-Shafranov residual": (float(equation_residual), 3.0e-6),
      "magnetic Bernoulli residual": (float(bernoulli_residual), 1.1e-6),
      "XCTS constraint residual": (constraint_residual, 1.0e-10),
      "logged CT divergence": (float(logged_divergence), 1.0e-14),
   }
   for label, (value, limit) in limits.items():
      if not math.isfinite(value) or value > limit:
         raise RuntimeError(f"{label}={value:.3e} exceeds {limit:.3e}")
   if not 2 < int(updates) <= 96:
      raise RuntimeError(f"unexpected magnetic iteration count {updates}")
   if not 1.0e-5 < float(maximum_field) < 1.0e-3:
      raise RuntimeError(
         f"nonlinear equilibrium has implausible bmax={maximum_field}"
      )

   metadata = read_metadata(artifact)
   expected_metadata = {
      "schema_version": "3",
      "physics": "rmhd",
      "eos": "hybrid_polytrope",
      "has_ct": "1",
      "magnetic_seed": "none",
      "magnetic_equilibrium": "poloidal_grad_shafranov",
      "magnetic_force_balance": "grad_shafranov",
      "magnetic_face_representation": "valencia_densitized",
      "source_ic": "tov",
      "spacetime_initial_data": "xcts",
   }
   for key, expected in expected_metadata.items():
      if metadata.get(key) != expected:
         raise RuntimeError(
            f"magnetohydrostatic artifact metadata {key}="
            f"{metadata.get(key)!r}"
         )
   if not math.isclose(
      float(metadata["magnetic_equilibrium_poloidal_magnetization"]),
      1.0e-2,
   ) or not math.isclose(
      float(metadata["magnetic_equilibrium_nonlinear_poloidal"]), 50.0
   ):
      raise RuntimeError("magnetohydrostatic artifact lost its current law")

   faces = tuple(
      read_face(artifact / f"face_b{direction}.h5")
      for direction in range(1, 4)
   )
   expected_shapes = ((25, 24, 24), (24, 25, 24), (24, 24, 25))
   if tuple(field.shape for field in faces) != expected_shapes:
      raise RuntimeError(
         f"unexpected CT face shapes {tuple(field.shape for field in faces)}"
      )
   spacing = 32.0 / 24.0
   divergence = (
      np.diff(faces[0], axis=0)
      + np.diff(faces[1], axis=1)
      + np.diff(faces[2], axis=2)
   ) / spacing
   artifact_divergence = float(np.max(np.abs(divergence)))
   if artifact_divergence > 1.0e-14:
      raise RuntimeError(
         f"artifact CT divergence is {artifact_divergence:.3e}"
      )

   face_average = cell_average_faces(faces)
   evolution = read_state(artifact / "evolution_fluid_cache.h5")
   if evolution.shape[0] < 8:
      raise RuntimeError("magnetohydrostatic artifact lost its RMHD cache")
   handoff_error = float(np.max(np.abs(evolution[5:8] - face_average)))
   handoff_scale = float(np.max(np.abs(face_average)))
   handoff_relative_error = handoff_error / max(handoff_scale, 1.0e-300)
   if handoff_relative_error > 2.0e-15:
      raise RuntimeError(
         "densitized CT face-to-evolution handoff mismatch: "
         f"relative error={handoff_relative_error:.3e}"
      )

   magnetic = evolution[5:8]
   swapped = np.swapaxes(magnetic, 1, 2)
   symmetry_error = max(
      float(np.max(np.abs(magnetic[0] - swapped[1]))),
      float(np.max(np.abs(magnetic[1] - swapped[0]))),
      float(np.max(np.abs(magnetic[2] - swapped[2]))),
   )
   symmetry_relative_error = symmetry_error / max(
      float(np.max(np.abs(magnetic))), 1.0e-300
   )
   if symmetry_relative_error > 2.0e-12:
      raise RuntimeError(
         "axisymmetric magnetic field lost x/y exchange symmetry: "
         f"relative error={symmetry_relative_error:.3e}"
      )

   artifact_bytes = sum(
      path.stat().st_size for path in artifact.rglob("*") if path.is_file()
   )
   return {
      "case": name,
      "status": "PASS",
      "ranks": ranks,
      "wall_seconds": seconds,
      "matter_iterations": matter_iterations,
      "magnetic_updates": int(updates),
      "relative_potential_update": float(potential_update),
      "linear_residual": float(linear_residual),
      "equation_relative_residual": float(equation_residual),
      "bernoulli_residual": float(bernoulli_residual),
      "maximum_comoving_field": float(maximum_field),
      "constraint_residual": constraint_residual,
      "artifact_divb_linf": artifact_divergence,
      "handoff_relative_error": handoff_relative_error,
      "symmetry_relative_error": symmetry_relative_error,
      "reflection_relative_error": math.nan,
      "rank_relative_error": math.nan,
      "zero_limit_geometry_error": math.nan,
      "maximum_velocity": 0.0,
      "dimensionless_spin": 0.0,
      "artifact_bytes": artifact_bytes,
      "run_dir": repo_relative(run_dir),
   }


def validate_exact_round_trip(
   source_artifact: Path, *, ranks: int, timeout: int | None
) -> dict[str, object]:
   name = "sly_tov_grad_shafranov_exact_round_trip"
   run_dir, artifact, text, seconds = run_case(
      name,
      ranks=ranks,
      timeout=timeout,
      parameters={
         "initial_data.input_file": repo_relative(source_artifact),
         "initial_data.magnetic_equilibrium": "none",
         "initial_data.magnetic_equilibrium_poloidal_magnetization": "0.0",
         "initial_data.magnetic_equilibrium_nonlinear_poloidal": "0.0",
         "initial_data.magnetic_equilibrium_relaxation": "0.5",
      },
   )
   required_log = (
      "loaded physical initial data:",
      "completed initial-data-only solve and artifact write",
   )
   has_preflight = (
      "Evolution preflight C2P/P2C:" in text
      or (
         "Physical-ID preflight: finite non-ideal RMHD primitive and conserved fields"
         in text
      )
   )
   if any(fragment not in text for fragment in required_log) \
         or not has_preflight:
      raise RuntimeError("magnetic physical-ID round trip missed the C++ reader")
   if MAGNETIC.search(text) is not None:
      raise RuntimeError("exact magnetic reload unnecessarily reran equilibrium")
   source_metadata = read_metadata(source_artifact)
   round_trip_metadata = read_metadata(artifact)
   if source_metadata != round_trip_metadata:
      differing = sorted(
         key for key in source_metadata.keys() | round_trip_metadata.keys()
         if source_metadata.get(key) != round_trip_metadata.get(key)
      )
      raise RuntimeError(
         "magnetic physical-ID round trip changed metadata: "
         + ", ".join(differing)
      )
   field_differences = artifact_field_differences(
      source_artifact, artifact
   )
   for field, (absolute, relative) in field_differences.items():
      limit = 2.0e-15 if field == "evolution_fluid_cache.h5" else 0.0
      if relative > limit:
         raise RuntimeError(
            f"magnetic physical-ID round trip changed {field} by "
            f"absolute={absolute:.3e}, relative={relative:.3e}"
         )
   field_difference = max(
      relative for _, relative in field_differences.values()
   )
   return {
      "case": name,
      "status": "PASS",
      "ranks": ranks,
      "wall_seconds": seconds,
      "matter_iterations": 0,
      "magnetic_updates": 0,
      "relative_potential_update": 0.0,
      "linear_residual": 0.0,
      "equation_relative_residual": 0.0,
      "bernoulli_residual": 0.0,
      "maximum_comoving_field": math.nan,
      "constraint_residual": math.nan,
      "artifact_divb_linf": 0.0,
      "handoff_relative_error": field_difference,
      "symmetry_relative_error": math.nan,
      "reflection_relative_error": math.nan,
      "rank_relative_error": math.nan,
      "zero_limit_geometry_error": math.nan,
      "maximum_velocity": 0.0,
      "dimensionless_spin": 0.0,
      "artifact_bytes": sum(
         path.stat().st_size for path in artifact.rglob("*") if path.is_file()
      ),
      "run_dir": repo_relative(run_dir),
   }


def validate_zero_limit(
   *, ranks: int, timeout: int | None
) -> dict[str, object]:
   common = {
      "initial_data.magnetic_equilibrium_poloidal_magnetization": "0.0",
      "initial_data.magnetic_equilibrium_nonlinear_poloidal": "0.0",
   }
   name = "sly_tov_grad_shafranov_zero_limit"
   run_dir, artifact, text, seconds = run_case(
      name, ranks=ranks, timeout=timeout, parameters=common
   )
   matter_iterations, constraint_residual = maximum_constraint_residual(text)
   if MAGNETIC.search(text) is not None:
      raise RuntimeError("zero magnetization unnecessarily ran a magnetic solve")
   faces = tuple(
      read_face(artifact / f"face_b{direction}.h5")
      for direction in range(1, 4)
   )
   if any(np.any(field != 0.0) for field in faces):
      raise RuntimeError("zero-magnetization limit produced a magnetic field")

   control_name = "sly_tov_unmagnetized_xcts_control"
   _, control_artifact, control_text, _ = run_case(
      control_name,
      ranks=ranks,
      timeout=timeout,
      parameters={
         "initial_data.magnetic_equilibrium": "none",
         "initial_data.magnetic": "zero",
         "pyfluid.physics": "rhd",
         "pyfluid.ct": "off",
      },
   )
   _, control_constraint = maximum_constraint_residual(control_text)
   geometry = read_state(artifact / "adm_geometry.h5")
   control_geometry = read_state(control_artifact / "adm_geometry.h5")
   fluid = read_state(artifact / "physical_fluid.h5")[:5]
   control_fluid = read_state(control_artifact / "physical_fluid.h5")[:5]
   zero_limit_error = max(
      float(np.max(np.abs(geometry - control_geometry))),
      float(np.max(np.abs(fluid - control_fluid))),
   )
   if zero_limit_error > 5.0e-15:
      raise RuntimeError(
         f"magnetic zero limit differs from hydrodynamics by {zero_limit_error:.3e}"
      )
   if max(constraint_residual, control_constraint) > 1.0e-10:
      raise RuntimeError("zero-limit XCTS control missed its residual gate")
   return {
      "case": name,
      "status": "PASS",
      "ranks": ranks,
      "wall_seconds": seconds,
      "matter_iterations": matter_iterations,
      "magnetic_updates": 0,
      "relative_potential_update": 0.0,
      "linear_residual": 0.0,
      "equation_relative_residual": 0.0,
      "bernoulli_residual": 0.0,
      "maximum_comoving_field": 0.0,
      "constraint_residual": constraint_residual,
      "artifact_divb_linf": 0.0,
      "handoff_relative_error": 0.0,
      "symmetry_relative_error": 0.0,
      "reflection_relative_error": math.nan,
      "rank_relative_error": math.nan,
      "zero_limit_geometry_error": zero_limit_error,
      "maximum_velocity": 0.0,
      "dimensionless_spin": 0.0,
      "artifact_bytes": sum(
         path.stat().st_size
         for path in artifact.rglob("*")
         if path.is_file()
      ),
      "run_dir": repo_relative(run_dir),
   }


def validate_reflection_and_rank(
   *, ranks: int, timeout: int | None
) -> dict[str, object]:
   parameters = {
      "geometry.prob_lo": "-16.0 -16.0 0.0",
      "geometry.prob_hi": "16.0 16.0 16.0",
      "amr.n_cell": "24 24 12",
      "amr.max_grid_size": "12",
      "pyfluid.bc_zlo": "symmetry",
   }
   name = "sly_tov_nonlinear_poloidal_equatorial_half"
   run_dir, artifact, text, seconds = run_case(
      name, ranks=ranks, timeout=timeout, parameters=parameters
   )
   magnetic = MAGNETIC.search(text)
   if magnetic is None:
      raise RuntimeError("equatorial-half solve omitted its magnetic summary")
   matter_iterations, constraint_residual = maximum_constraint_residual(text)
   for group, limit, label in (
      (2, 1.1e-6, "potential update"),
      (3, 1.0e-12, "linear residual"),
      (4, 3.0e-6, "equation residual"),
      (5, 1.1e-6, "Bernoulli residual"),
      (7, 1.0e-14, "CT divergence"),
   ):
      value = float(magnetic.group(group))
      if not math.isfinite(value) or value > limit:
         raise RuntimeError(
            f"equatorial-half magnetic {label}={value:.3e} exceeds {limit:.3e}"
         )
   if constraint_residual > 1.0e-10:
      raise RuntimeError(
         "equatorial-half magnetic XCTS residual exceeds tolerance"
      )
   metadata = read_metadata(artifact)
   if metadata.get("reflection_lo") != "0,0,1" \
         or metadata.get("reflection_hi") != "0,0,0":
      raise RuntimeError(
         "equatorial-half artifact lost its reflection contract"
      )

   full_artifact = (
      RUN_ROOT / "sly_tov_nonlinear_poloidal_grad_shafranov/physical_id"
   )
   full_evolution = read_state(
      full_artifact / "evolution_fluid_cache.h5"
   )
   half_evolution = read_state(artifact / "evolution_fluid_cache.h5")
   full_half = full_evolution[:, :, :, 12:]
   reflection_error = float(
      np.max(np.abs(half_evolution - full_half))
   ) / max(float(np.max(np.abs(full_half))), 1.0e-300)
   if reflection_error > 2.0e-10:
      raise RuntimeError(
         "equatorial-half and full-domain fields disagree: "
         f"relative error={reflection_error:.3e}"
      )

   serial_name = "sly_tov_nonlinear_poloidal_equatorial_half_rank1"
   _, serial_artifact, _, _ = run_case(
      serial_name, ranks=1, timeout=timeout, parameters=parameters
   )
   serial_evolution = read_state(
      serial_artifact / "evolution_fluid_cache.h5"
   )
   rank_error = float(
      np.max(np.abs(serial_evolution - half_evolution))
   ) / max(float(np.max(np.abs(half_evolution))), 1.0e-300)
   if rank_error > 2.0e-13:
      raise RuntimeError(
         f"one/three-rank equatorial-half disagreement={rank_error:.3e}"
      )

   faces = tuple(
      read_face(artifact / f"face_b{direction}.h5")
      for direction in range(1, 4)
   )
   face_average = cell_average_faces(faces)
   handoff_error = float(
      np.max(np.abs(half_evolution[5:8] - face_average))
   ) / max(float(np.max(np.abs(face_average))), 1.0e-300)
   if handoff_error > 2.0e-15:
      raise RuntimeError("equatorial-half CT handoff is not exact")

   return {
      "case": name,
      "status": "PASS",
      "ranks": ranks,
      "wall_seconds": seconds,
      "matter_iterations": matter_iterations,
      "magnetic_updates": int(magnetic.group(1)),
      "relative_potential_update": float(magnetic.group(2)),
      "linear_residual": float(magnetic.group(3)),
      "equation_relative_residual": float(magnetic.group(4)),
      "bernoulli_residual": float(magnetic.group(5)),
      "maximum_comoving_field": float(magnetic.group(6)),
      "constraint_residual": constraint_residual,
      "artifact_divb_linf": float(magnetic.group(7)),
      "handoff_relative_error": handoff_error,
      "symmetry_relative_error": math.nan,
      "reflection_relative_error": reflection_error,
      "rank_relative_error": rank_error,
      "zero_limit_geometry_error": math.nan,
      "maximum_velocity": 0.0,
      "dimensionless_spin": 0.0,
      "artifact_bytes": sum(
         path.stat().st_size
         for path in artifact.rglob("*")
         if path.is_file()
      ),
      "run_dir": repo_relative(run_dir),
   }


def validate_rotating_equilibrium(
   *, ranks: int, timeout: int | None
) -> dict[str, object]:
   name = "sly_uniform_rns_nonlinear_poloidal_grad_shafranov"
   run_dir, artifact, text, seconds = run_case(
      name,
      ranks=ranks,
      timeout=timeout,
      base_input=RNS_INPUT,
   )
   magnetic = MAGNETIC.search(text)
   rotating = ROTATING.search(text)
   if magnetic is None or rotating is None:
      raise RuntimeError("magnetized RNS omitted its equilibrium diagnostics")
   matter_iterations, constraint_residual = maximum_constraint_residual(text)
   for group, limit, label in (
      (2, 1.1e-6, "potential update"),
      (3, 1.0e-12, "linear residual"),
      (4, 3.0e-6, "equation residual"),
      (5, 1.1e-6, "Bernoulli residual"),
      (7, 1.0e-14, "CT divergence"),
   ):
      value = float(magnetic.group(group))
      if not math.isfinite(value) or value > limit:
         raise RuntimeError(
            f"magnetized RNS {label}={value:.3e} exceeds {limit:.3e}"
         )
   if constraint_residual > 1.0e-10:
      raise RuntimeError("magnetized RNS XCTS residual exceeds tolerance")
   omega = float(rotating.group(1))
   maximum_velocity = float(rotating.group(2))
   shedding = float(rotating.group(3))
   adm_mass = float(rotating.group(4))
   baryonic_mass = float(rotating.group(5))
   angular_momentum = float(rotating.group(6))
   dimensionless_spin = float(rotating.group(7))
   if not math.isclose(omega, 5.0e-3, rel_tol=0.0, abs_tol=1.0e-14) \
         or not 0.01 < maximum_velocity < 1.0 \
         or not 0.0 < shedding <= 1.1 \
         or not adm_mass > 0.0 \
         or not baryonic_mass > 0.0 \
         or not angular_momentum > 0.0 \
         or not 1.0e-3 < dimensionless_spin < 0.8:
      raise RuntimeError(
         "magnetized RNS diagnostics are not a physical rotating model: "
         f"omega={omega}, vmax={maximum_velocity}, shedding={shedding}, "
         f"M={adm_mass}, Mb={baryonic_mass}, J={angular_momentum}, "
         f"chi={dimensionless_spin}"
      )
   metadata = read_metadata(artifact)
   if metadata.get("source_ic") != "rotating_star" \
         or metadata.get("rotation_law") != "uniform" \
         or not math.isclose(float(metadata["angular_velocity"]), omega):
      raise RuntimeError("magnetized RNS artifact lost its rotation identity")

   faces = tuple(
      read_face(artifact / f"face_b{direction}.h5")
      for direction in range(1, 4)
   )
   spacing = 32.0 / 24.0
   divergence = (
      np.diff(faces[0], axis=0)
      + np.diff(faces[1], axis=1)
      + np.diff(faces[2], axis=2)
   ) / spacing
   artifact_divergence = float(np.max(np.abs(divergence)))
   evolution = read_state(artifact / "evolution_fluid_cache.h5")
   face_average = cell_average_faces(faces)
   handoff_error = float(
      np.max(np.abs(evolution[5:8] - face_average))
   ) / max(float(np.max(np.abs(face_average))), 1.0e-300)
   if artifact_divergence > 1.0e-14 or handoff_error > 2.0e-15:
      raise RuntimeError("magnetized RNS violated its CT artifact contract")
   return {
      "case": name,
      "status": "PASS",
      "ranks": ranks,
      "wall_seconds": seconds,
      "matter_iterations": matter_iterations,
      "magnetic_updates": int(magnetic.group(1)),
      "relative_potential_update": float(magnetic.group(2)),
      "linear_residual": float(magnetic.group(3)),
      "equation_relative_residual": float(magnetic.group(4)),
      "bernoulli_residual": float(magnetic.group(5)),
      "maximum_comoving_field": float(magnetic.group(6)),
      "constraint_residual": constraint_residual,
      "artifact_divb_linf": artifact_divergence,
      "handoff_relative_error": handoff_error,
      "symmetry_relative_error": math.nan,
      "reflection_relative_error": math.nan,
      "rank_relative_error": math.nan,
      "zero_limit_geometry_error": math.nan,
      "maximum_velocity": maximum_velocity,
      "dimensionless_spin": dimensionless_spin,
      "artifact_bytes": sum(
         path.stat().st_size
         for path in artifact.rglob("*")
         if path.is_file()
      ),
      "run_dir": repo_relative(run_dir),
   }


def validate_compose_equilibrium(
   *, ranks: int, timeout: int | None
) -> dict[str, object]:
   name = "compose_sly4_tov_linear_poloidal_grad_shafranov"
   run_dir, artifact, text, seconds = run_case(
      name,
      ranks=ranks,
      timeout=timeout,
      parameters={
         "pyfluid.eos": "compose",
         "pyfluid.eos_preset": "custom",
         "pyfluid.eos_table": "pyfluid_amrex/eos/compose/sly4",
         "pyfluid.thermal_gamma": "1.8",
         "initial_data.magnetic_equilibrium_nonlinear_poloidal": "0.0",
      },
   )
   magnetic = MAGNETIC.search(text)
   if magnetic is None:
      raise RuntimeError("CompOSE magnetic TOV omitted its equilibrium summary")
   matter_iterations, constraint_residual = maximum_constraint_residual(text)
   for group, limit, label in (
      (2, 1.1e-6, "potential update"),
      (3, 1.0e-12, "linear residual"),
      (4, 3.0e-6, "equation residual"),
      (5, 1.1e-6, "Bernoulli residual"),
      (7, 1.0e-14, "CT divergence"),
   ):
      value = float(magnetic.group(group))
      if not math.isfinite(value) or value > limit:
         raise RuntimeError(
            f"CompOSE magnetic {label}={value:.3e} exceeds {limit:.3e}"
         )
   if constraint_residual > 1.0e-10:
      raise RuntimeError("CompOSE magnetic XCTS residual exceeds tolerance")
   metadata = read_metadata(artifact)
   expected = {
      "eos": "compose",
      "eos_backend": "compose_cold_barotrope_v1",
      "magnetic_equilibrium": "poloidal_grad_shafranov",
      "magnetic_force_balance": "grad_shafranov",
      "magnetic_face_representation": "valencia_densitized",
   }
   for key, value in expected.items():
      if metadata.get(key) != value:
         raise RuntimeError(
            f"CompOSE magnetic metadata {key}={metadata.get(key)!r}"
         )
   if not metadata.get("eos_table_fingerprint"):
      raise RuntimeError("CompOSE magnetic artifact omitted its EoS fingerprint")
   faces = tuple(
      read_face(artifact / f"face_b{direction}.h5")
      for direction in range(1, 4)
   )
   face_average = cell_average_faces(faces)
   evolution = read_state(artifact / "evolution_fluid_cache.h5")
   handoff_error = float(
      np.max(np.abs(evolution[5:8] - face_average))
   ) / max(float(np.max(np.abs(face_average))), 1.0e-300)
   if handoff_error > 2.0e-15:
      raise RuntimeError("CompOSE magnetic CT handoff is not exact")
   return {
      "case": name,
      "status": "PASS",
      "ranks": ranks,
      "wall_seconds": seconds,
      "matter_iterations": matter_iterations,
      "magnetic_updates": int(magnetic.group(1)),
      "relative_potential_update": float(magnetic.group(2)),
      "linear_residual": float(magnetic.group(3)),
      "equation_relative_residual": float(magnetic.group(4)),
      "bernoulli_residual": float(magnetic.group(5)),
      "maximum_comoving_field": float(magnetic.group(6)),
      "constraint_residual": constraint_residual,
      "artifact_divb_linf": float(magnetic.group(7)),
      "handoff_relative_error": handoff_error,
      "symmetry_relative_error": math.nan,
      "reflection_relative_error": math.nan,
      "rank_relative_error": math.nan,
      "zero_limit_geometry_error": math.nan,
      "maximum_velocity": 0.0,
      "dimensionless_spin": 0.0,
      "artifact_bytes": sum(
         path.stat().st_size for path in artifact.rglob("*") if path.is_file()
      ),
      "run_dir": repo_relative(run_dir),
   }


def validate_invalid_composition(*, ranks: int, timeout: int | None) -> None:
   name = "reject_seed_plus_equilibrium"
   run_dir = RUN_ROOT / name
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "initial_data.output_file": repo_relative(run_dir / "physical_id"),
         "initial_data.magnetic_seed": "pressure_poloidal",
         "initial_data.magnetic_seed_maximum_comoving_field": "1.0e-4",
      },
   )
   completed = run_amrex(
      dim=3, input_path=input_path, ranks=ranks, timeout=timeout
   )
   (run_dir / "run.log").write_text(completed.stdout)
   expected = (
      "a kinematic magnetic seed and a magnetohydrostatic magnetic "
      "equilibrium cannot be selected together"
   )
   if completed.returncode == 0 or expected not in completed.stdout:
      raise RuntimeError("seed/equilibrium composition did not fail honestly")

   transverse_name = "reject_transverse_magnetic_reflection"
   transverse_dir = RUN_ROOT / transverse_name
   transverse_dir.mkdir(parents=True, exist_ok=True)
   transverse_input = materialize_run_input(
      BASE_INPUT,
      transverse_dir / "input.in",
      {
         "pyfluid.run_name": transverse_name,
         "pyfluid.output_dir": repo_relative(transverse_dir),
         "initial_data.output_file": repo_relative(
            transverse_dir / "physical_id"
         ),
         "geometry.prob_lo": "0.0 -16.0 -16.0",
         "geometry.prob_hi": "16.0 16.0 16.0",
         "pyfluid.bc_xlo": "symmetry",
      },
   )
   transverse = run_amrex(
      dim=3, input_path=transverse_input, ranks=ranks, timeout=timeout
   )
   (transverse_dir / "run.log").write_text(transverse.stdout)
   transverse_expected = (
      "an oriented poloidal magnetic field may use a Cartesian reflection "
      "plane only normal to its magnetic axis"
   )
   if transverse.returncode == 0 \
         or transverse_expected not in transverse.stdout:
      raise RuntimeError("transverse magnetic reflection did not fail honestly")


def corrupted_artifact(
   source: Path, name: str, key: str, value: str | None
) -> Path:
   destination = RUN_ROOT / "invalid_artifacts" / name
   if destination.exists():
      shutil.rmtree(destination)
   destination.parent.mkdir(parents=True, exist_ok=True)
   shutil.copytree(source, destination)
   metadata_path = destination / "Header.meta"
   prefix = f"# {key}="
   lines = metadata_path.read_text().splitlines()
   matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
   if len(matches) != 1:
      raise RuntimeError(f"cannot corrupt unique metadata key {key}")
   if value is None:
      del lines[matches[0]]
   else:
      lines[matches[0]] = prefix + value
   metadata_path.write_text("\n".join(lines) + "\n")
   return destination


def expect_rejected_load(
   name: str,
   artifact: Path,
   expected: str,
   *,
   ranks: int,
   timeout: int | None,
   parameters: dict[str, str] | None = None,
) -> None:
   run_dir = RUN_ROOT / name
   if run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True)
   values = {
      "pyfluid.run_name": name,
      "pyfluid.output_dir": repo_relative(run_dir),
      "initial_data.input_file": repo_relative(artifact),
      "initial_data.output_file": repo_relative(run_dir / "physical_id"),
      "initial_data.magnetic_equilibrium": "none",
      "initial_data.magnetic_equilibrium_poloidal_magnetization": "0.0",
      "initial_data.magnetic_equilibrium_nonlinear_poloidal": "0.0",
      "initial_data.magnetic_equilibrium_relaxation": "0.5",
   }
   if parameters:
      values.update(parameters)
   input_path = materialize_run_input(BASE_INPUT, run_dir / "input.in", values)
   completed = run_amrex(
      dim=3, input_path=input_path, ranks=ranks, timeout=timeout
   )
   (run_dir / "run.log").write_text(completed.stdout)
   if completed.returncode == 0 or expected not in completed.stdout:
      raise RuntimeError(
         f"{name} did not reject with {expected!r}; see "
         f"{repo_relative(run_dir / 'run.log')}"
      )


def validate_invalid_artifacts(
   source_artifact: Path,
   reflected_artifact: Path,
   *,
   ranks: int,
   timeout: int | None,
) -> None:
   bad_representation = corrupted_artifact(
      source_artifact,
      "bad_magnetic_face_representation",
      "magnetic_face_representation",
      "physical",
   )
   expect_rejected_load(
      "reject_bad_magnetic_face_representation",
      bad_representation,
      "metadata mismatch for 'magnetic_face_representation'",
      ranks=ranks,
      timeout=timeout,
   )
   unknown_method = corrupted_artifact(
      source_artifact,
      "unknown_magnetic_equilibrium",
      "magnetic_equilibrium",
      "unknown_equilibrium",
   )
   expect_rejected_load(
      "reject_unknown_magnetic_equilibrium",
      unknown_method,
      "magnetic equilibrium is not present in the method catalog",
      ranks=ranks,
      timeout=timeout,
   )
   missing_parameter = corrupted_artifact(
      source_artifact,
      "missing_magnetic_equilibrium_relaxation",
      "magnetic_equilibrium_relaxation",
      None,
   )
   expect_rejected_load(
      "reject_missing_magnetic_equilibrium_parameter",
      missing_parameter,
      "metadata missing catalog-required key "
      "'magnetic_equilibrium_relaxation'",
      ranks=ranks,
      timeout=timeout,
   )
   missing_component_kind = corrupted_artifact(
      source_artifact,
      "missing_component_kind",
      "component_0_kind",
      None,
   )
   expect_rejected_load(
      "reject_missing_component_kind",
      missing_component_kind,
      "physical-ID component record missing key 'component_0_kind'",
      ranks=ranks,
      timeout=timeout,
   )
   malformed_component_origin = corrupted_artifact(
      source_artifact,
      "malformed_component_origin",
      "component_0_origin",
      "0,0",
   )
   expect_rejected_load(
      "reject_malformed_component_origin",
      malformed_component_origin,
      "component_0_origin' must contain three reals",
      ranks=ranks,
      timeout=timeout,
   )
   out_of_range_component = corrupted_artifact(
      source_artifact,
      "out_of_range_component",
      "component_count",
      "0",
   )
   expect_rejected_load(
      "reject_out_of_range_component",
      out_of_range_component,
      "physical-ID component record has unexpected key",
      ranks=ranks,
      timeout=timeout,
   )
   expect_rejected_load(
      "reject_changed_grid_magnetic_equilibrium",
      source_artifact,
      "require an exact target layout",
      ranks=ranks,
      timeout=timeout,
      parameters={
         "amr.n_cell": "20 20 20",
         "amr.max_grid_size": "10",
         "amr.blocking_factor": "2",
      },
   )
   expect_rejected_load(
      "reject_magnetic_tov_to_rns_continuation",
      source_artifact,
      "cannot yet be continued or recomposed",
      ranks=ranks,
      timeout=timeout,
      parameters={
         "pyfluid.ic": "rotating_star",
         "initial_data.angular_velocity": "5.0e-3",
      },
   )
   expect_rejected_load(
      "reject_magnetic_torus_composition",
      source_artifact,
      "cannot yet be continued or recomposed",
      ranks=ranks,
      timeout=timeout,
      parameters={
         "initial_data.disk": "equilibrium_torus",
         "initial_data.disk_gravity": "coupled",
         "initial_data.disk_specific_angular_momentum": "5.2",
         "initial_data.disk_inner_radius": "10.0",
      },
   )
   expect_rejected_load(
      "reject_changed_magnetic_reflection_contract",
      reflected_artifact,
      "metadata mismatch for 'reflection_lo'",
      ranks=ranks,
      timeout=timeout,
      parameters={
         "geometry.prob_lo": "-16.0 -16.0 0.0",
         "geometry.prob_hi": "16.0 16.0 16.0",
         "amr.n_cell": "24 24 12",
         "amr.max_grid_size": "12",
         "pyfluid.bc_zlo": "outflow",
      },
   )


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   if args.force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   equilibrium = validate_equilibrium(ranks=args.ranks, timeout=args.timeout)
   source_artifact = (
      RUN_ROOT / "sly_tov_nonlinear_poloidal_grad_shafranov/physical_id"
   )
   round_trip = validate_exact_round_trip(
      source_artifact, ranks=args.ranks, timeout=args.timeout
   )
   rows = [
      equilibrium,
      round_trip,
      validate_reflection_and_rank(ranks=args.ranks, timeout=args.timeout),
      validate_rotating_equilibrium(ranks=args.ranks, timeout=args.timeout),
      validate_zero_limit(ranks=args.ranks, timeout=args.timeout),
      validate_compose_equilibrium(ranks=args.ranks, timeout=args.timeout),
   ]
   validate_invalid_composition(ranks=args.ranks, timeout=args.timeout)
   reflected_artifact = (
      RUN_ROOT / "sly_tov_nonlinear_poloidal_equatorial_half/physical_id"
   )
   validate_invalid_artifacts(
      source_artifact,
      reflected_artifact,
      ranks=args.ranks,
      timeout=args.timeout,
   )
   write_tsv(RESULTS, rows, tuple(rows[0]))
   reflected = rows[2]
   rotating = rows[3]
   zero_limit = rows[4]
   compose = rows[5]
   print(
      "PASS magnetohydrostatic_initial_data "
      f"H/M={equilibrium['constraint_residual']:.3e} "
      f"GS={equilibrium['equation_relative_residual']:.3e} "
      f"Bernoulli={equilibrium['bernoulli_residual']:.3e} "
      f"divB={equilibrium['artifact_divb_linf']:.3e} "
      f"handoff={equilibrium['handoff_relative_error']:.3e} "
      f"symmetry={equilibrium['symmetry_relative_error']:.3e} "
      f"reflection={reflected['reflection_relative_error']:.3e} "
      f"rank={reflected['rank_relative_error']:.3e} "
      f"rns_chi={rotating['dimensionless_spin']:.3e} "
      f"zero_limit={zero_limit['zero_limit_geometry_error']:.3e} "
      f"round_trip={round_trip['handoff_relative_error']:.3e} "
      f"compose_GS={compose['equation_relative_residual']:.3e}"
   )
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
