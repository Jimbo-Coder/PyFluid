#!/usr/bin/env python3
"""Validate reduced-coordinate physical ID and 3D Cartesian handoff."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

from plotfile_reader import parse_plot_header, read_level_state
from validation_common import (
   ELLIPTIC_POLICY,
   ROOT,
   append_validation_history,
   build_cached_target,
   materialize_run_input,
   read_scalar_rows,
   repo_relative,
   run_amrex,
   run_mpi,
   scalar_validation_row,
   write_tsv,
)


SOURCE_INPUT = ROOT / "pyfluid_amrex/inputs/tov_spherical_id_1d.in"
DIRECT_INPUT = ROOT / "pyfluid_amrex/inputs/tov_direct_id_3d.in"
MATERIALIZED_INPUT = ROOT / "pyfluid_amrex/inputs/tov_from_spherical_id_3d.in"
RNS_SOURCE_INPUT = (
   ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_spherical_id_2d.in"
)
RNS_MATERIALIZED_INPUT = (
   ROOT / "pyfluid_amrex/inputs/rotating_star_from_spherical_id_3d.in"
)
RNS_FULL_STORAGE_INPUT = (
   ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_spherical_id_3d.in"
)
RUN_ROOT = ROOT / "runs/current/reduced_initial_data_validation"
RESULTS = RUN_ROOT / "reduced_initial_data.tsv"
FULL_RESULTS = RUN_ROOT / "reduced_initial_data_full.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
INTERPOLATION_CHECK = "pyfluid_amrex_physical_id_interpolation_check"
AXISYMMETRIC_ELLIPTIC_CHECK = (
   "pyfluid_amrex_initial_data_elliptic_check_2d"
)
RECOVERY_PATTERN = re.compile(
   r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)
PREFLIGHT_PATTERN = re.compile(
   r"Evolution preflight C2P/P2C: active_cells=(\d+) "
   r"max_conserved_error=([0-9.eE+-]+) "
   r"max_primitive_error=([0-9.eE+-]+)"
)


def run_id_case(
   name: str,
   dim: int,
   base_input: Path,
   overrides: dict[str, str],
   args: argparse.Namespace,
) -> tuple[Path, Path, float]:
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   artifact = run_dir / "physical_id"
   input_path = materialize_run_input(
      base_input,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "initial_data.output_file": repo_relative(artifact),
         **overrides,
      },
   )
   log_path = run_dir / "run.log"
   seconds = 0.0
   if args.force or not artifact.is_dir():
      completed = run_amrex(
         dim=dim,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      log_path.write_text(completed.stdout)
      seconds = completed.seconds
      if completed.returncode != 0:
         raise RuntimeError(
            f"{name} failed; see {repo_relative(log_path)}"
         )
   if not log_path.is_file() or not artifact.is_dir():
      raise RuntimeError(f"{name} lacks its log or physical-ID artifact")
   log_text = log_path.read_text()
   preflight = PREFLIGHT_PATTERN.findall(log_text)
   if not preflight:
      raise RuntimeError(f"{name} did not run the relativistic ID preflight")
   recovery = RECOVERY_PATTERN.findall(log_text)
   if not recovery or tuple(int(value) for value in recovery[-1]) != (0, 0):
      raise RuntimeError(f"{name} required relativistic recovery during ID")
   return run_dir, artifact, seconds


def read_level_components(path: Path) -> list[np.ndarray]:
   with h5py.File(path, "r") as handle:
      group = handle["level_0"]
      data = np.asarray(group["data:datatype=0"])
      offsets = np.asarray(group["data:offsets=0"], dtype=np.int64)
      boxes = np.asarray(group["boxes"])
      num_components = int(
         np.asarray(handle.attrs["num_components"]).reshape(-1)[0]
      )
      pieces: list[list[tuple[tuple[int, ...], np.ndarray]]] = [
         [] for _ in range(num_components)
      ]
      for box, begin, end in zip(boxes, offsets[:-1], offsets[1:]):
         key = tuple(int(value) for value in box)
         block = data[begin:end].reshape(num_components, -1)
         for component in range(num_components):
            pieces[component].append((key, block[component].copy()))
   return [
      np.concatenate([values for _, values in sorted(component)])
      for component in pieces
   ]


def maximum_overlapping_node_disagreement(path: Path) -> float:
   """Return the largest disagreement where nodal HDF5 boxes overlap."""
   with h5py.File(path, "r") as handle:
      dimension = int(
         np.asarray(handle.attrs["dim"]).reshape(-1)[0]
      )
      num_components = int(
         np.asarray(handle.attrs["num_components"]).reshape(-1)[0]
      )
      group = handle["level_0"]
      data = np.asarray(group["data:datatype=0"])
      offsets = np.asarray(group["data:offsets=0"], dtype=np.int64)
      boxes = np.asarray(group["boxes"])
      box_values = np.asarray(
         [tuple(int(value) for value in box) for box in boxes], dtype=int
      )
      lower = np.min(box_values[:, :dimension], axis=0)
      upper = np.max(box_values[:, dimension:], axis=0)
      extents = tuple((upper - lower + 1).tolist())
      dense = np.full((num_components, *extents), np.nan)
      maximum = 0.0
      for bounds, begin, end in zip(
         box_values, offsets[:-1], offsets[1:]
      ):
         box_lower = bounds[:dimension]
         box_upper = bounds[dimension:]
         box_extents = tuple((box_upper - box_lower + 1).tolist())
         block = data[begin:end].reshape(
            (num_components, *reversed(box_extents))
         )
         block = block.transpose(
            (0, *range(dimension, 0, -1))
         )
         slices = tuple(
            slice(
               int(box_lower[direction] - lower[direction]),
               int(box_upper[direction] - lower[direction] + 1),
            )
            for direction in range(dimension)
         )
         destination = dense[(slice(None), *slices)]
         overlap = np.isfinite(destination)
         if np.any(overlap):
            maximum = max(
               maximum,
               float(np.max(np.abs(destination[overlap] - block[overlap]))),
            )
         destination[...] = block
   return maximum


def read_cell_component(
   path: Path,
   index: tuple[int, int, int],
   component: int,
) -> float:
   with h5py.File(path, "r") as handle:
      group = handle["level_0"]
      data = np.asarray(group["data:datatype=0"])
      offsets = np.asarray(group["data:offsets=0"], dtype=np.int64)
      boxes = np.asarray(group["boxes"])
      num_components = int(
         np.asarray(handle.attrs["num_components"]).reshape(-1)[0]
      )
      if component < 0 or component >= num_components:
         raise RuntimeError("physical-ID component index is out of range")
      for box, begin, end in zip(boxes, offsets[:-1], offsets[1:]):
         lower = tuple(int(box[direction]) for direction in range(3))
         upper = tuple(int(box[direction + 3]) for direction in range(3))
         if not all(
            lower[direction] <= index[direction] <= upper[direction]
            for direction in range(3)
         ):
            continue
         extents = tuple(
            upper[direction] - lower[direction] + 1
            for direction in range(3)
         )
         points = int(np.prod(extents))
         local = index[0] - lower[0]
         local += extents[0] * (index[1] - lower[1])
         local += extents[0] * extents[1] * (index[2] - lower[2])
         value_index = int(begin) + component * points + local
         if value_index >= int(end):
            raise RuntimeError("physical-ID box/component layout is inconsistent")
         return float(data[value_index])
   raise RuntimeError(f"physical-ID cell {index} is not present")


def relative_component_error(
   candidate: list[np.ndarray],
   reference: list[np.ndarray],
   component: int,
) -> tuple[float, float]:
   difference = np.abs(candidate[component] - reference[component])
   scale = max(float(np.max(np.abs(reference[component]))), 1.0e-14)
   return float(np.max(difference) / scale), float(np.mean(difference) / scale)


def compare_artifacts(
   candidate: Path,
   reference: Path,
) -> dict[str, float]:
   fluid = read_level_components(candidate / "physical_fluid.h5")
   fluid_reference = read_level_components(reference / "physical_fluid.h5")
   adm = read_level_components(candidate / "adm_geometry.h5")
   adm_reference = read_level_components(reference / "adm_geometry.h5")
   rho_max, rho_l1 = relative_component_error(fluid, fluid_reference, 0)
   pressure_max, pressure_l1 = relative_component_error(
      fluid, fluid_reference, 4
   )
   gamma_errors = [
      relative_component_error(adm, adm_reference, component)[0]
      for component in (0, 3, 5)
   ]
   lapse_max, lapse_l1 = relative_component_error(
      adm, adm_reference, 12
   )
   inactive_absolute = max(
      float(np.max(np.abs(adm[component] - adm_reference[component])))
      for component in (1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15)
   )
   return {
      "rho_max": rho_max,
      "rho_l1": rho_l1,
      "pressure_max": pressure_max,
      "pressure_l1": pressure_l1,
      "gamma_max": max(gamma_errors),
      "lapse_max": lapse_max,
      "lapse_l1": lapse_l1,
      "inactive_absolute": inactive_absolute,
   }


def require_portable_source(artifact: Path, dimension: int) -> None:
   metadata = (artifact / "Header.meta").read_text()
   required = (
      "# schema_version=2",
      f"# amrex_spacedim={dimension}",
      "# component_basis=mesh_orthonormal",
      "# has_exact_caches=0",
   )
   if any(item not in metadata for item in required):
      raise RuntimeError("reduced source artifact lacks portable chart metadata")
   expected = {"Header", "Header.meta", "physical_fluid.h5", "adm_geometry.h5"}
   actual = {path.name for path in artifact.iterdir() if path.is_file()}
   if actual != expected:
      raise RuntimeError(
         "reduced source artifact should contain only portable physical fields: "
         f"expected={sorted(expected)} actual={sorted(actual)}"
      )


def read_artifact_metadata(artifact: Path) -> dict[str, str]:
   values: dict[str, str] = {}
   for raw_line in (artifact / "Header.meta").read_text().splitlines():
      line = raw_line.removeprefix("#").strip()
      if "=" not in line:
         continue
      key, value = line.split("=", 1)
      values[key.strip()] = value.strip()
   return values


def dense_artifact_field(path: Path) -> np.ndarray:
   names, dimension, _ = parse_plot_header(path)
   return read_level_state(path, names, dimension, 0)


def normalized_maximum_difference(
   candidate: np.ndarray,
   reference: np.ndarray,
) -> float:
   scale = max(float(np.max(np.abs(reference))), 1.0e-14)
   return float(np.max(np.abs(candidate - reference)) / scale)


def validate_full_storage_rns(
   args: argparse.Namespace,
) -> list[dict[str, object]]:
   common_2d = {
      "amr.n_cell": "96 24",
      "amr.max_grid_size": "32",
      "amr.blocking_factor": "1",
      "initial_data.radial_grid": "adaptive",
      "initial_data.angular_velocity": "0.001",
   }
   _, reduced_artifact, reduced_seconds = run_id_case(
      "rns_axisymmetric_2d_96x24",
      2,
      RNS_SOURCE_INPUT,
      common_2d,
      args,
   )
   require_portable_source(reduced_artifact, 2)

   full_artifacts: dict[int, Path] = {}
   full_seconds: dict[int, float] = {}
   for azimuthal_cells in (1, 4):
      _, artifact, seconds = run_id_case(
         f"rns_axisymmetric_3d_96x24x{azimuthal_cells}",
         3,
         RNS_FULL_STORAGE_INPUT,
         {
            "amr.n_cell": f"96 24 {azimuthal_cells}",
            "amr.max_grid_size": "32",
            "amr.blocking_factor": "1",
            "initial_data.radial_grid": "adaptive",
            "initial_data.angular_velocity": "0.001",
         },
         args,
      )
      require_portable_source(artifact, 3)
      metadata = read_artifact_metadata(artifact)
      if metadata.get("mesh_coordinate") != "spherical" \
            or metadata.get("mesh_axes") != "r theta phi" \
            or metadata.get("periodicity") != "0,0,1":
         raise RuntimeError(
            "full-storage RNS artifact lost its spherical owner identity"
         )
      full_artifacts[azimuthal_cells] = artifact
      full_seconds[azimuthal_cells] = seconds

   reduced_fluid = dense_artifact_field(
      reduced_artifact / "physical_fluid.h5"
   )
   reduced_adm = dense_artifact_field(reduced_artifact / "adm_geometry.h5")
   full_fields = {
      azimuthal_cells: (
         dense_artifact_field(artifact / "physical_fluid.h5"),
         dense_artifact_field(artifact / "adm_geometry.h5"),
      )
      for azimuthal_cells, artifact in full_artifacts.items()
   }
   maximum_phi_variation = 0.0
   maximum_field_difference = 0.0
   for fluid, adm in full_fields.values():
      maximum_phi_variation = max(
         maximum_phi_variation,
         normalized_maximum_difference(
            fluid, np.repeat(fluid[..., :1], fluid.shape[-1], axis=-1)
         ),
         normalized_maximum_difference(
            adm, np.repeat(adm[..., :1], adm.shape[-1], axis=-1)
         ),
      )
      maximum_field_difference = max(
         maximum_field_difference,
         normalized_maximum_difference(fluid[..., 0], reduced_fluid),
         normalized_maximum_difference(adm[..., 0], reduced_adm),
      )

   diagnostic_fields = (
      "maximum_density",
      "maximum_velocity",
      "injection_energy",
      "equatorial_radius",
      "polar_radius",
      "mass_shedding_indicator",
      "adm_mass",
      "baryonic_mass",
      "angular_momentum",
      "dimensionless_spin",
   )
   reduced_metadata = read_artifact_metadata(reduced_artifact)
   maximum_diagnostic_difference = 0.0
   for artifact in full_artifacts.values():
      metadata = read_artifact_metadata(artifact)
      for field in diagnostic_fields:
         reference = float(reduced_metadata[field])
         candidate = float(metadata[field])
         maximum_diagnostic_difference = max(
            maximum_diagnostic_difference,
            abs(candidate - reference) / max(abs(reference), 1.0e-14),
         )

   if maximum_phi_variation > 5.0e-14 \
         or maximum_field_difference > 2.0e-12 \
         or maximum_diagnostic_difference > 2.0e-11:
      raise RuntimeError(
         "axisymmetric full-storage RNS does not reduce to the native 2D "
         "owner: "
         f"phi={maximum_phi_variation:.6e}, "
         f"fields={maximum_field_difference:.6e}, "
         f"diagnostics={maximum_diagnostic_difference:.6e}"
      )
   return [{
      "kind": "initial_data",
      "case": "rns_axisymmetric_2d_to_3d_nphi1_nphi4",
      "status": "PASS",
      "metric": "maximum_normalized_field_difference",
      "value": maximum_field_difference,
      "maximum_phi_variation": maximum_phi_variation,
      "maximum_diagnostic_difference": maximum_diagnostic_difference,
      "source_seconds": reduced_seconds
         + full_seconds[1] + full_seconds[4],
      "run_dir": repo_relative(full_artifacts[4].parent),
   }]


def validate_reduced_differential_rns(
   args: argparse.Namespace,
) -> list[dict[str, object]]:
   common = {
      "amr.n_cell": "96 24",
      "amr.max_grid_size": "48",
      "initial_data.radial_grid": "adaptive",
      "initial_data.angular_velocity": "0.0201",
      "initial_data.differential_rotation_length": "10",
   }
   artifacts: dict[str, Path] = {}
   seconds: dict[str, float] = {}
   for name, law, power, length_b in (
      ("j_constant", "j_constant", "1", "10"),
      ("q_power_q1", "q_power", "1", "10"),
      ("q_power_q2", "q_power", "2", "10"),
      ("uryu_plus_13", "uryu_plus_13", "1", "10"),
      ("uryu_j_limit", "uryu_j_envelope_p1", "1", "1e7"),
   ):
      _, artifact, elapsed = run_id_case(
         f"rns_differential_{name}_96x24",
         2,
         RNS_SOURCE_INPUT,
         {
            **common,
            "initial_data.rotation_law": law,
            "initial_data.rotation_power_q": power,
            "initial_data.differential_rotation_length_b": length_b,
         },
         args,
      )
      require_portable_source(artifact, 2)
      artifacts[name] = artifact
      seconds[name] = elapsed

   metadata = {
      name: read_artifact_metadata(artifact)
      for name, artifact in artifacts.items()
   }
   if metadata["j_constant"].get("rotation_law") != "j_constant" \
         or metadata["q_power_q1"].get("rotation_law") != "q_power" \
         or metadata["q_power_q2"].get("rotation_law") != "q_power" \
         or metadata["q_power_q1"].get("rotation_power_q") != "1" \
         or metadata["q_power_q2"].get("rotation_power_q") != "2" \
         or metadata["uryu_plus_13"].get("rotation_law") \
            != "uryu_plus_13" \
         or metadata["uryu_j_limit"].get("rotation_law") \
            != "uryu_j_envelope_p1" \
         or float(metadata["uryu_plus_13"].get(
            "differential_rotation_length_b", "nan"
         )) != 10.0 \
         or float(metadata["uryu_j_limit"].get(
            "differential_rotation_length_b", "nan"
         )) != 1.0e7:
      raise RuntimeError("differential RNS artifacts lost their law identity")

   identity_fields = (
      "maximum_density",
      "maximum_velocity",
      "injection_energy",
      "equatorial_radius",
      "polar_radius",
      "equatorial_log_enthalpy_gradient",
      "polar_log_enthalpy_gradient",
      "mass_shedding_indicator",
      "adm_mass",
      "baryonic_mass",
      "angular_momentum",
      "dimensionless_spin",
   )
   q1_identity_error = max(
      abs(
         float(metadata["q_power_q1"][field])
         - float(metadata["j_constant"][field])
      ) / max(abs(float(metadata["j_constant"][field])), 1.0e-14)
      for field in identity_fields
   )
   if q1_identity_error > 1.0e-11:
      raise RuntimeError(
         "q-power q=1 does not reproduce j-constant RNS: "
         f"maximum relative diagnostic error={q1_identity_error:.6e}"
      )

   uryu_j_limit_error = max(
      abs(
         float(metadata["uryu_j_limit"][field])
         - float(metadata["j_constant"][field])
      ) / max(abs(float(metadata["j_constant"][field])), 1.0e-14)
      for field in identity_fields
   )
   if uryu_j_limit_error > 1.0e-7:
      raise RuntimeError(
         "Uryu j-envelope did not approach j-constant as B increased: "
         f"maximum relative diagnostic error={uryu_j_limit_error:.6e}"
      )

   q2 = metadata["q_power_q2"]
   q2_values = {
      key: float(q2[key])
      for key in (
         "maximum_velocity",
         "injection_energy",
         "mass_shedding_indicator",
         "adm_mass",
         "baryonic_mass",
         "angular_momentum",
         "dimensionless_spin",
      )
   }
   q2_j_difference = abs(
      q2_values["angular_momentum"]
      - float(metadata["j_constant"]["angular_momentum"])
   ) / abs(float(metadata["j_constant"]["angular_momentum"]))
   if not all(np.isfinite(value) for value in q2_values.values()) \
         or q2_values["maximum_velocity"] <= 0.0 \
         or q2_values["injection_energy"] <= 0.0 \
         or not 0.0 < q2_values["mass_shedding_indicator"] < 2.0 \
         or q2_values["adm_mass"] <= 0.0 \
         or q2_values["baryonic_mass"] <= 0.0 \
         or q2_values["angular_momentum"] <= 0.0 \
         or not 0.0 < q2_values["dimensionless_spin"] < 1.0 \
         or q2_j_difference < 1.0e-3:
      raise RuntimeError(
         "q-power q=2 did not produce a distinct physical RNS: "
         f"values={q2_values}, J difference={q2_j_difference:.6e}"
      )

   uryu_plus = metadata["uryu_plus_13"]
   uryu_plus_values = {
      key: float(uryu_plus[key])
      for key in (
         "maximum_velocity",
         "injection_energy",
         "mass_shedding_indicator",
         "adm_mass",
         "baryonic_mass",
         "angular_momentum",
         "dimensionless_spin",
      )
   }
   uryu_plus_j_difference = abs(
      uryu_plus_values["angular_momentum"]
      - float(metadata["j_constant"]["angular_momentum"])
   ) / abs(float(metadata["j_constant"]["angular_momentum"]))
   if not all(np.isfinite(value) for value in uryu_plus_values.values()) \
         or uryu_plus_values["maximum_velocity"] <= 0.0 \
         or uryu_plus_values["injection_energy"] <= 0.0 \
         or not 0.0 < uryu_plus_values["mass_shedding_indicator"] < 2.0 \
         or uryu_plus_values["adm_mass"] <= 0.0 \
         or uryu_plus_values["baryonic_mass"] <= 0.0 \
         or uryu_plus_values["angular_momentum"] <= 0.0 \
         or not 0.0 < uryu_plus_values["dimensionless_spin"] < 1.0 \
         or uryu_plus_j_difference < 1.0e-3:
      raise RuntimeError(
         "Uryu plus p=1 q=3 did not produce a distinct physical RNS: "
         f"values={uryu_plus_values}, "
         f"J difference={uryu_plus_j_difference:.6e}"
      )

   return [{
      "kind": "initial_data",
      "case": "rns_j_constant_q_power_q1_identity_96x24",
      "status": "PASS",
      "metric": "maximum_relative_diagnostic_error",
      "value": q1_identity_error,
      "source_seconds": seconds["j_constant"] + seconds["q_power_q1"],
      "run_dir": repo_relative(artifacts["q_power_q1"].parent),
   }, {
      "kind": "initial_data",
      "case": "rns_q_power_q2_useful_regime_96x24",
      "status": "PASS",
      "metric": "mass_shedding_indicator",
      "value": q2_values["mass_shedding_indicator"],
      "mass_shedding_indicator": q2_values["mass_shedding_indicator"],
      "dimensionless_spin": q2_values["dimensionless_spin"],
      "source_seconds": seconds["q_power_q2"],
      "run_dir": repo_relative(artifacts["q_power_q2"].parent),
   }, {
      "kind": "initial_data",
      "case": "rns_uryu_j_envelope_j_constant_limit_96x24",
      "status": "PASS",
      "metric": "maximum_relative_diagnostic_error",
      "value": uryu_j_limit_error,
      "source_seconds": seconds["uryu_j_limit"],
      "run_dir": repo_relative(artifacts["uryu_j_limit"].parent),
   }, {
      "kind": "initial_data",
      "case": "rns_uryu_plus_13_useful_regime_96x24",
      "status": "PASS",
      "metric": "mass_shedding_indicator",
      "value": uryu_plus_values["mass_shedding_indicator"],
      "mass_shedding_indicator": uryu_plus_values[
         "mass_shedding_indicator"
      ],
      "dimensionless_spin": uryu_plus_values["dimensionless_spin"],
      "source_seconds": seconds["uryu_plus_13"],
      "run_dir": repo_relative(artifacts["uryu_plus_13"].parent),
   }]


def validate_reduced_rns_handoff(
   args: argparse.Namespace,
) -> list[dict[str, object]]:
   _, source_artifact, source_seconds = run_id_case(
      "rns_adaptive_source_128x32",
      2,
      RNS_SOURCE_INPUT,
      {
         "amr.n_cell": "128 32",
         "amr.max_grid_size": "64",
         "initial_data.radial_grid": "adaptive",
         "initial_data.angular_velocity": "0.001",
      },
      args,
   )
   require_portable_source(source_artifact, 2)
   metadata = read_artifact_metadata(source_artifact)
   if metadata.get("mesh_coordinate") != "spherical" \
         or metadata.get("mesh_axes") != "r theta" \
         or metadata.get("eos_preset") != "sly" \
         or metadata.get("rotation_law") != "uniform" \
         or metadata.get("differential_rotation_length") != "10" \
         or metadata.get("rotation_power_q") != "1" \
         or metadata.get("rotation_axis1") != "0" \
         or metadata.get("rotation_axis2") != "0" \
         or metadata.get("rotation_axis3") != "1" \
         or metadata.get("coordinate_layout") != "explicit_faces_v1":
      raise RuntimeError("reduced RNS artifact lost its chart or EoS identity")
   radial_faces = np.fromstring(
      metadata.get("coordinate_faces_0_0", ""), sep=","
   )
   if radial_faces.size != 129 \
         or not np.all(np.diff(radial_faces) > 0.0) \
         or radial_faces[0] != 0.0 \
         or radial_faces[-1] != 32.0:
      raise RuntimeError("adaptive RNS artifact has an invalid radial map")
   required_diagnostics = {
      key: float(metadata[key])
      for key in (
         "mass_shedding_indicator",
         "injection_energy",
         "adm_mass",
         "baryonic_mass",
         "angular_momentum",
         "dimensionless_spin",
      )
   }
   if not all(np.isfinite(value) for value in required_diagnostics.values()):
      raise RuntimeError("reduced RNS artifact has nonfinite global diagnostics")
   if not 0.9 < required_diagnostics["mass_shedding_indicator"] < 1.1 \
         or required_diagnostics["injection_energy"] <= 0.0 \
         or required_diagnostics["adm_mass"] <= 0.0 \
         or required_diagnostics["baryonic_mass"] <= 0.0 \
         or required_diagnostics["angular_momentum"] <= 0.0 \
         or required_diagnostics["dimensionless_spin"] <= 0.0:
      raise RuntimeError(
         f"reduced small-spin RNS diagnostics are unphysical: {required_diagnostics}"
      )
   matter_outer_radius = float(metadata.get("matter_outer_radius", "nan"))
   if not 0.0 < matter_outer_radius < radial_faces[-1]:
      raise RuntimeError("reduced RNS artifact lost its matter-support bound")
   nodal_disagreement = maximum_overlapping_node_disagreement(
      source_artifact / "adm_geometry.h5"
   )
   if nodal_disagreement > 2.0e-13:
      raise RuntimeError(
         "reduced RNS nodal geometry is not single-valued across HDF5 boxes: "
         f"maximum disagreement={nodal_disagreement:.6e}"
      )

   _, materialized_artifact, materialization_seconds = run_id_case(
      "rns_materialized_16cubed",
      3,
      RNS_MATERIALIZED_INPUT,
      {
         "geometry.prob_lo": "-16.0 -16.0 -16.0",
         "geometry.prob_hi": "16.0 16.0 16.0",
         "amr.n_cell": "16 16 16",
         "amr.max_grid_size": "16",
         "initial_data.input_file": repo_relative(source_artifact),
         "initial_data.radial_grid": "uniform",
         "initial_data.angular_velocity": "0.001",
      },
      args,
   )
   target_metadata = read_artifact_metadata(materialized_artifact)
   if target_metadata.get("amrex_spacedim") != "3" \
         or target_metadata.get("mesh_coordinate") != "cartesian" \
         or target_metadata.get("mesh_axes") != "x y z" \
         or target_metadata.get("source_ic") != "rotating_star":
      raise RuntimeError("materialized RNS artifact lost its target identity")
   fluid = read_level_components(materialized_artifact / "physical_fluid.h5")
   if len(fluid) < 5 or not all(np.all(np.isfinite(component)) for component in fluid):
      raise RuntimeError("materialized RNS fluid is incomplete or nonfinite")
   velocity_max = float(np.max(np.sqrt(fluid[1] ** 2 + fluid[2] ** 2 + fluid[3] ** 2)))
   if not velocity_max > 1.0e-5:
      raise RuntimeError("materialized rotating star lost its azimuthal velocity")

   rotated_run, rotated_artifact, rotated_seconds = run_id_case(
      "rns_materialized_x_axis_16cubed",
      3,
      RNS_MATERIALIZED_INPUT,
      {
         "geometry.prob_lo": "-15.0 -18.0 -15.5",
         "geometry.prob_hi": "17.0 14.0 16.5",
         "amr.n_cell": "16 16 16",
         "amr.max_grid_size": "16",
         "initial_data.input_file": repo_relative(source_artifact),
         "initial_data.radial_grid": "uniform",
         "initial_data.angular_velocity": "0.001",
         "initial_data.rotation_axis1": "1.0",
         "initial_data.rotation_axis2": "0.0",
         "initial_data.rotation_axis3": "0.0",
         "initial_data.rotation_center1": "1.0",
         "initial_data.rotation_center2": "-2.0",
         "initial_data.rotation_center3": "0.5",
      },
      args,
   )
   rotated_metadata = read_artifact_metadata(rotated_artifact)
   if rotated_metadata.get("rotation_axis1") != "1" \
         or rotated_metadata.get("rotation_axis2") != "0" \
         or rotated_metadata.get("rotation_axis3") != "0" \
         or rotated_metadata.get("rotation_center1") != "1" \
         or rotated_metadata.get("rotation_center2") != "-2" \
         or rotated_metadata.get("rotation_center3") != "0.5" \
         or "rigidly transformed physical initial data" not in \
            (rotated_run / "run.log").read_text():
      raise RuntimeError("rotated RNS materialization lost its axis identity")
   rotated_fluid = read_level_components(
      rotated_artifact / "physical_fluid.h5"
   )
   rotated_speed = np.sqrt(
      rotated_fluid[1] ** 2
      + rotated_fluid[2] ** 2
      + rotated_fluid[3] ** 2
   )
   reference_speed = np.sqrt(fluid[1] ** 2 + fluid[2] ** 2 + fluid[3] ** 2)
   rotation_covariance_error = max(
      float(np.max(np.abs(np.sort(rotated_fluid[0], axis=None)
                          - np.sort(fluid[0], axis=None)))),
      float(np.max(np.abs(np.sort(rotated_speed, axis=None)
                          - np.sort(reference_speed, axis=None)))),
   )
   if rotation_covariance_error > 2.0e-11 \
         or float(np.max(np.abs(rotated_fluid[1]))) > 2.0e-13:
      raise RuntimeError(
         "reduced RNS rigid-axis materialization is not rotation covariant: "
         f"error={rotation_covariance_error:.6e}"
      )

   continuation_overrides = {
      "amr.n_cell": "96 24",
      "amr.max_grid_size": "48",
      "initial_data.radial_grid": "adaptive",
      "initial_data.angular_velocity": "0.0011",
      "initial_data.matter_relative_tolerance": "1.0e-7",
      "initial_data.matter_max_iterations": "128",
   }
   same_omega_run, same_omega_artifact, same_omega_seconds = run_id_case(
      "rns_adaptive_same_omega_resolved_96x24",
      2,
      RNS_SOURCE_INPUT,
      {
         "amr.n_cell": "96 24",
         "amr.max_grid_size": "48",
         "initial_data.radial_grid": "adaptive",
         "initial_data.angular_velocity": "0.001",
         "initial_data.input_file": repo_relative(source_artifact),
         "initial_data.matter_relative_tolerance": "1.0e-7",
         "initial_data.matter_max_iterations": "128",
      },
      args,
   )
   if (
      "re-solving rotating physical initial data on the changed r-theta grid"
      not in (same_omega_run / "run.log").read_text()
   ):
      raise RuntimeError("changed-grid same-omega RNS load skipped its XCTS solve")
   same_omega_metadata = read_artifact_metadata(same_omega_artifact)
   if (
      same_omega_metadata.get("coordinate_faces_0_0")
      == metadata.get("coordinate_faces_0_0")
   ):
      raise RuntimeError("same-omega RNS re-solve did not change its radial map")
   _, direct_artifact, direct_seconds = run_id_case(
      "rns_adaptive_direct_96x24",
      2,
      RNS_SOURCE_INPUT,
      continuation_overrides,
      args,
   )
   _, continued_artifact, continued_seconds = run_id_case(
      "rns_adaptive_continued_96x24",
      2,
      RNS_SOURCE_INPUT,
      {
         **continuation_overrides,
         "initial_data.input_file": repo_relative(source_artifact),
      },
      args,
   )
   direct_metadata = read_artifact_metadata(direct_artifact)
   continued_metadata = read_artifact_metadata(continued_artifact)
   if direct_metadata.get("coordinate_faces_0_0") \
         != continued_metadata.get("coordinate_faces_0_0"):
      raise RuntimeError("adaptive RNS continuation changed its radial map")
   continuation_errors = {}
   for key in (
      "injection_energy", "adm_mass", "baryonic_mass", "angular_momentum"
   ):
      direct_value = float(direct_metadata[key])
      continued_value = float(continued_metadata[key])
      continuation_errors[key] = abs(continued_value - direct_value) \
         / abs(direct_value)
   if max(continuation_errors.values()) > 2.0e-5:
      raise RuntimeError(
         "adaptive RNS continuation disagrees with a matched direct solve: "
         f"{continuation_errors}"
      )
   return [{
      "kind": "initial_data",
      "case": "rns_adaptive_128x32_to_cartesian_16cubed",
      "status": "PASS",
      "metric": "materialized_maximum_velocity",
      "value": velocity_max,
      "mass_shedding_indicator": required_diagnostics[
         "mass_shedding_indicator"
      ],
      "dimensionless_spin": required_diagnostics["dimensionless_spin"],
      "nodal_overlap_disagreement": nodal_disagreement,
      "source_seconds": source_seconds,
      "materialization_seconds": materialization_seconds,
      "run_dir": repo_relative(materialized_artifact.parent),
   }, {
      "kind": "initial_data",
      "case": "rns_adaptive_rotated_x_axis_128x32_to_16cubed",
      "status": "PASS",
      "metric": "rotation_covariance_error",
      "value": rotation_covariance_error,
      "mass_shedding_indicator": required_diagnostics[
         "mass_shedding_indicator"
      ],
      "dimensionless_spin": required_diagnostics["dimensionless_spin"],
      "source_seconds": source_seconds,
      "materialization_seconds": rotated_seconds,
      "run_dir": repo_relative(rotated_artifact.parent),
   }, {
      "kind": "initial_data",
      "case": "rns_adaptive_saved_sequence_continuation_128x32_to_96x24",
      "status": "PASS",
      "metric": "maximum_relative_global_diagnostic_difference",
      "value": max(continuation_errors.values()),
      "source_seconds": direct_seconds,
      "materialization_seconds": continued_seconds,
      "run_dir": repo_relative(continued_artifact.parent),
   }, {
      "kind": "initial_data",
      "case": "rns_adaptive_same_omega_different_grid_resolve",
      "status": "PASS",
      "metric": "matter_outer_radius",
      "value": float(same_omega_metadata["matter_outer_radius"]),
      "source_seconds": same_omega_seconds,
      "run_dir": repo_relative(same_omega_run),
   }]


def run_evolution(
   source_artifact: Path,
   args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
   name = "tov_loaded_evolution"
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      MATERIALIZED_INPUT,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "initial_data.input_file": repo_relative(source_artifact),
         "initial_data.solve_only": "0",
         "pyfluid.max_step": "3",
         "pyfluid.ng": "2",
         "pyfluid.recon": "pwl",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.scalar_int": "1",
      },
      remove_parameters=("initial_data.output_file",),
   )
   log_path = run_dir / "run.log"
   if args.force or not (run_dir / "scalars.tsv").is_file():
      completed = run_amrex(
         dim=3,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      log_path.write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(
            f"{name} failed; see {repo_relative(log_path)}"
         )
   scalar = scalar_validation_row(
      run_dir,
      case=name,
      input_file=run_dir / "run.in",
      checks=ELLIPTIC_POLICY,
      mhd=False,
      dim=3,
      ranks=args.ranks,
      kind="initial_data",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"loaded TOV evolution failed: {scalar['reason']}")
   rows = read_scalar_rows(run_dir)
   if len(rows) != 4:
      raise RuntimeError("loaded TOV evolution did not retain all four scalar rows")
   initial_mass = float(rows[0]["mass"])
   final_mass = float(rows[-1]["mass"])
   mass_drift = abs(final_mass - initial_mass) / initial_mass
   match = RECOVERY_PATTERN.search(log_path.read_text())
   if match is None or int(match.group(1)) != 0 or int(match.group(2)) != 0:
      raise RuntimeError("loaded TOV evolution required relativistic recovery")
   return {
      "kind": "initial_data",
      "case": name,
      "status": "PASS",
      "metric": "three_step_mass_drift",
      "value": mass_drift,
      "run_dir": repo_relative(run_dir),
   }, scalar


def validate_errors(errors: dict[int, dict[str, float]], level: str) -> None:
   final = errors[max(errors)]
   limits = {
      "rho_max": 8.0e-5 if level == "standard" else 3.0e-5,
      "pressure_max": 8.0e-5 if level == "standard" else 3.0e-5,
      "gamma_max": 1.0e-5 if level == "standard" else 4.0e-6,
      "lapse_max": 6.0e-6 if level == "standard" else 2.0e-6,
      "inactive_absolute": 2.0e-12,
   }
   failed = {
      key: (final[key], limit)
      for key, limit in limits.items()
      if final[key] > limit
   }
   if failed:
      raise RuntimeError(f"reduced TOV handoff exceeded error limits: {failed}")
   if level == "full":
      resolutions = sorted(errors)
      for key in ("rho_max", "pressure_max", "gamma_max", "lapse_max"):
         values = [errors[resolution][key] for resolution in resolutions]
         if not all(right < left for left, right in zip(values, values[1:])):
            raise RuntimeError(
               f"reduced TOV {key} did not decrease with source resolution: "
               f"{values}"
            )


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("standard", "full"), default="standard")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=180)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   interpolation_check = build_cached_target(INTERPOLATION_CHECK)
   interpolation_result = run_mpi(
      interpolation_check,
      ranks=1,
      timeout=args.timeout,
   )
   (RUN_ROOT / "interpolation_check.log").write_text(
      interpolation_result.stdout
   )
   if interpolation_result.returncode != 0:
      raise RuntimeError("portable physical-ID basis/reflection check failed")

   elliptic_check = build_cached_target(AXISYMMETRIC_ELLIPTIC_CHECK)
   elliptic_result = run_mpi(
      elliptic_check,
      ranks=args.ranks,
      timeout=args.timeout,
   )
   (RUN_ROOT / "axisymmetric_elliptic_check.log").write_text(
      elliptic_result.stdout
   )
   if (
      elliptic_result.returncode != 0
      or "axisymmetric_spherical_poisson" not in elliptic_result.stdout
      or "axisymmetric_spherical_toroidal_vector_poisson"
         not in elliptic_result.stdout
   ):
      raise RuntimeError("axisymmetric spherical elliptic check failed")

   resolutions = (512,) if args.level == "standard" else (256, 512, 1024)
   source_artifacts: dict[int, Path] = {}
   run_seconds: dict[int, float] = {}
   for resolution in resolutions:
      _, artifact, seconds = run_id_case(
         f"source_n{resolution}",
         1,
         SOURCE_INPUT,
         {
            "amr.n_cell": str(resolution),
            "amr.max_grid_size": str(min(128, resolution)),
            "initial_data.radial_grid": "uniform",
         },
         args,
      )
      require_portable_source(artifact, 1)
      source_artifacts[resolution] = artifact
      run_seconds[resolution] = seconds

   _, direct_artifact, _ = run_id_case(
      "direct_3d",
      3,
      DIRECT_INPUT,
      {},
      args,
   )
   errors: dict[int, dict[str, float]] = {}
   rows: list[dict[str, object]] = [
      {
         "kind": "initial_data",
         "case": "portable_id_basis_and_equatorial_reflection",
         "status": "PASS",
         "metric": "focused_cpp_check",
         "value": 0.0,
         "run_dir": repo_relative(RUN_ROOT),
      },
      {
         "kind": "initial_data",
         "case": "axisymmetric_spherical_scalar_and_toroidal_elliptic",
         "status": "PASS",
         "metric": "focused_cpp_check",
         "value": 0.0,
         "run_dir": repo_relative(RUN_ROOT),
      },
   ]
   for resolution in resolutions:
      _, materialized_artifact, seconds = run_id_case(
         f"materialized_n{resolution}_3d",
         3,
         MATERIALIZED_INPUT,
         {
            "initial_data.input_file": repo_relative(
               source_artifacts[resolution]
            ),
         },
         args,
      )
      error = compare_artifacts(materialized_artifact, direct_artifact)
      errors[resolution] = error
      rows.append({
         "kind": "initial_data",
         "case": f"tov_spherical_n{resolution}_to_cartesian_3d",
         "status": "PASS",
         "metric": "maximum_normalized_physical_error",
         "value": max(
            error["rho_max"],
            error["pressure_max"],
            error["gamma_max"],
            error["lapse_max"],
         ),
         "rho_max": error["rho_max"],
         "pressure_max": error["pressure_max"],
         "gamma_max": error["gamma_max"],
         "lapse_max": error["lapse_max"],
         "source_seconds": run_seconds[resolution],
         "materialization_seconds": seconds,
         "run_dir": repo_relative(materialized_artifact.parent),
      })
   validate_errors(errors, args.level)

   _, adaptive_tov_artifact, adaptive_tov_seconds = run_id_case(
      "tov_adaptive_source_n256",
      1,
      SOURCE_INPUT,
      {
         "amr.n_cell": "256",
         "amr.max_grid_size": "128",
         "initial_data.radial_grid": "adaptive",
      },
      args,
   )
   require_portable_source(adaptive_tov_artifact, 1)
   adaptive_tov_metadata = read_artifact_metadata(adaptive_tov_artifact)
   adaptive_tov_faces = np.fromstring(
      adaptive_tov_metadata.get("coordinate_faces_0_0", ""), sep=","
   )
   if adaptive_tov_metadata.get("coordinate_layout") != "explicit_faces_v1" \
         or adaptive_tov_faces.size != 257 \
         or not np.all(np.diff(adaptive_tov_faces) > 0.0):
      raise RuntimeError("adaptive TOV artifact has an invalid radial map")
   _, adaptive_tov_materialized, adaptive_materialization_seconds = run_id_case(
      "tov_adaptive_n256_to_cartesian_3d",
      3,
      MATERIALIZED_INPUT,
      {
         "initial_data.input_file": repo_relative(adaptive_tov_artifact),
      },
      args,
   )
   adaptive_tov_error = compare_artifacts(
      adaptive_tov_materialized, direct_artifact
   )
   adaptive_tov_limits = {
      "rho_max": 8.0e-5,
      "pressure_max": 1.2e-4,
      "gamma_max": 1.2e-5,
      "lapse_max": 7.0e-6,
      "inactive_absolute": 2.0e-12,
   }
   adaptive_tov_failed = {
      key: (adaptive_tov_error[key], limit)
      for key, limit in adaptive_tov_limits.items()
      if adaptive_tov_error[key] > limit
   }
   if adaptive_tov_failed:
      raise RuntimeError(
         "adaptive TOV handoff exceeded error limits: "
         f"{adaptive_tov_failed}"
      )
   rows.append({
      "kind": "initial_data",
      "case": "tov_adaptive_spherical_n256_to_cartesian_3d",
      "status": "PASS",
      "metric": "maximum_normalized_physical_error",
      "value": max(
         adaptive_tov_error["rho_max"],
         adaptive_tov_error["pressure_max"],
         adaptive_tov_error["gamma_max"],
         adaptive_tov_error["lapse_max"],
      ),
      "rho_max": adaptive_tov_error["rho_max"],
      "pressure_max": adaptive_tov_error["pressure_max"],
      "gamma_max": adaptive_tov_error["gamma_max"],
      "lapse_max": adaptive_tov_error["lapse_max"],
      "source_seconds": adaptive_tov_seconds,
      "materialization_seconds": adaptive_materialization_seconds,
      "run_dir": repo_relative(adaptive_tov_materialized.parent),
   })
   rows.extend(validate_full_storage_rns(args))
   rows.extend(validate_reduced_rns_handoff(args))
   rows.extend(validate_reduced_differential_rns(args))

   if args.level == "full":
      finest_source = source_artifacts[max(resolutions)]
      large_target = {
         "geometry.prob_lo": "-40.0 -40.0 -40.0",
         "geometry.prob_hi": "40.0 40.0 40.0",
         "amr.n_cell": "24 24 24",
         "amr.max_grid_size": "24",
      }
      _, direct_large_artifact, _ = run_id_case(
         "direct_large_domain_3d",
         3,
         DIRECT_INPUT,
         large_target,
         args,
      )
      _, materialized_large_artifact, seconds = run_id_case(
         "materialized_large_domain_3d",
         3,
         MATERIALIZED_INPUT,
         {
            **large_target,
            "initial_data.input_file": repo_relative(finest_source),
         },
         args,
      )
      large_error = compare_artifacts(
         materialized_large_artifact, direct_large_artifact
      )
      large_limits = {
         "rho_max": 3.0e-5,
         "pressure_max": 3.0e-5,
         "gamma_max": 4.0e-6,
         "lapse_max": 2.0e-6,
         "inactive_absolute": 2.0e-12,
      }
      large_failed = {
         key: (large_error[key], limit)
         for key, limit in large_limits.items()
         if large_error[key] > limit
      }
      if large_failed:
         raise RuntimeError(
            "large-domain TOV handoff exceeded error limits: "
            f"{large_failed}"
         )
      rows.append({
         "kind": "initial_data",
         "case": "tov_spherical_n1024_to_larger_cartesian_domain",
         "status": "PASS",
         "metric": "maximum_normalized_physical_error",
         "value": max(
            large_error["rho_max"],
            large_error["pressure_max"],
            large_error["gamma_max"],
            large_error["lapse_max"],
         ),
         "rho_max": large_error["rho_max"],
         "pressure_max": large_error["pressure_max"],
         "gamma_max": large_error["gamma_max"],
         "lapse_max": large_error["lapse_max"],
         "materialization_seconds": seconds,
         "run_dir": repo_relative(materialized_large_artifact.parent),
      })

      # The odd target has a cell exactly at the origin.  It distinguishes
      # regular radial interpolation from true outer-domain continuation.
      origin_target = {
         "geometry.prob_lo": "-15.5 -15.5 -15.5",
         "geometry.prob_hi": "15.5 15.5 15.5",
         "amr.n_cell": "31 31 31",
         "amr.max_grid_size": "31",
         "amr.blocking_factor": "1",
      }
      _, direct_origin_artifact, _ = run_id_case(
         "direct_origin_cell_3d",
         3,
         DIRECT_INPUT,
         origin_target,
         args,
      )
      _, materialized_origin_artifact, seconds = run_id_case(
         "materialized_origin_cell_3d",
         3,
         MATERIALIZED_INPUT,
         {
            **origin_target,
            "initial_data.input_file": repo_relative(finest_source),
         },
         args,
      )
      origin_error = compare_artifacts(
         materialized_origin_artifact, direct_origin_artifact
      )
      origin_index = (15, 15, 15)
      direct_central_density = read_cell_component(
         direct_origin_artifact / "physical_fluid.h5", origin_index, 0
      )
      materialized_central_density = read_cell_component(
         materialized_origin_artifact / "physical_fluid.h5", origin_index, 0
      )
      central_density_error = abs(
         materialized_central_density - direct_central_density
      ) / direct_central_density
      if central_density_error > 2.0e-5:
         raise RuntimeError(
            "regular radial-origin materialization was treated as exterior: "
            f"central_density_error={central_density_error}"
         )
      rows.append({
         "kind": "initial_data",
         "case": "tov_spherical_regular_origin_to_cartesian_cell",
         "status": "PASS",
         "metric": "central_cell_normalized_density_error",
         "value": central_density_error,
         "rho_max": origin_error["rho_max"],
         "pressure_max": origin_error["pressure_max"],
         "gamma_max": origin_error["gamma_max"],
         "lapse_max": origin_error["lapse_max"],
         "materialization_seconds": seconds,
         "run_dir": repo_relative(materialized_origin_artifact.parent),
      })

   evolution_row, scalar_row = run_evolution(
      source_artifacts[max(resolutions)], args
   )
   rows.append(evolution_row)
   result_fields = [
      "kind",
      "case",
      "status",
      "metric",
      "value",
      "rho_max",
      "pressure_max",
      "gamma_max",
      "lapse_max",
      "source_seconds",
      "materialization_seconds",
      "mass_shedding_indicator",
      "dimensionless_spin",
      "maximum_phi_variation",
      "maximum_diagnostic_difference",
      "run_dir",
   ]
   results_path = RESULTS if args.level == "standard" else FULL_RESULTS
   write_tsv(results_path, rows, result_fields)
   append_validation_history(HISTORY, [*rows, scalar_row])
   print(f"wrote {repo_relative(results_path)}")
   for resolution in resolutions:
      error = errors[resolution]
      print(
         f"N={resolution}: rho={error['rho_max']:.6e} "
         f"pressure={error['pressure_max']:.6e} "
         f"gamma={error['gamma_max']:.6e} "
         f"lapse={error['lapse_max']:.6e}"
      )
   print("PASS reduced-coordinate stellar physical-ID handoff")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
