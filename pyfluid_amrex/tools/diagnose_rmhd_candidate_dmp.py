#!/usr/bin/env python3
"""Audit the RMHD candidate DMP and state-derived trouble detectors.

The production candidate guard currently tests each blended Runge--Kutta
candidate against a density/energy neighborhood taken only from the current
stage input.  For SSPRK3 stages two and three, however, that candidate is a
convex blend of the accepted step base and a forward-Euler update of the stage
input.  This offline diagnostic reconstructs that forward-Euler update from
saved unguarded stage states and compares three classifications:

* the current blended-candidate/current-input DMP;
* an SSP-consistent forward-Euler/current-input DMP; and
* a weaker blended-candidate bound using both base and input neighborhoods.

The same saved stage inputs are also classified with the five-point
derivative-ratio detector used by Berta et al.  That detector is evaluated
separately with hydrodynamic primitives and with magnetic components included,
both with a one-cell flag halo.  Each candidate is additionally checked against
the initial physical-specific-entropy minimum.  This is only a necessary
entropy-solution check, not a discrete numerical-entropy inequality.

It changes no evolution state and uses no exact solution to select cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import research_physical_wave_subcell_srmhd as physical
from diagnose_rmhd_balsara_onset import (
   COMPONENT_NAMES,
   PROFILE_BY_NAME,
   RUN_ROOT,
   case_spec,
   stage_output_paths,
)
from plotfile_reader import (
   parse_plot_bounds,
   parse_plot_header_1d,
   read_composite_state_1d,
)
from validation_common import repo_relative, write_tsv


DEFAULT_PROFILES = (
   "b4_ssprk3_stage_trace_ppm4_unguarded",
   "b4_ssprk3_stage_trace_ppm4_unguarded_hlld",
   "b3_ssprk3_stage_trace_mp5_unguarded",
   "b3_ssprk3_stage_trace_mp5_unguarded_hlld",
)
RESULTS = RUN_ROOT / "balsara_candidate_dmp_audit.tsv"
STAGE_BLEND = {
   "ssprk3_stage1_candidate": (0.0, 1.0),
   "ssprk3_stage2_candidate": (0.75, 0.25),
   "ssprk3_stage3_candidate": (1.0 / 3.0, 2.0 / 3.0),
}


def conserved_state(path: Path) -> tuple[np.ndarray, float]:
   names, time = parse_plot_header_1d(path)
   output = read_composite_state_1d(path, names)
   state = np.vstack([output[names.index(name)] for name in COMPONENT_NAMES])
   return state, time


def initial_state(path: Path, case: int) -> np.ndarray:
   names, _time = parse_plot_header_1d(path)
   output = read_composite_state_1d(path, names)
   cells = output.shape[1]
   prob_lo, prob_hi = parse_plot_bounds(path)
   edges = np.linspace(prob_lo[0], prob_hi[0], cells + 1)
   spec = case_spec(case)

   def prototype_primitive(state: tuple[float, ...]) -> np.ndarray:
      rho, pressure, v1, v2, v3, b1, b2, b3 = state
      return np.asarray((rho, v1, v2, v3, pressure, b1, b2, b3))

   endpoints = physical.conserved_array(
      np.vstack((
         prototype_primitive(spec["left"]),
         prototype_primitive(spec["right"]),
      )),
      float(spec["gamma"]),
   )
   widths = np.diff(edges)
   left_width = np.maximum(
      np.minimum(edges[1:], 0.0) - edges[:-1], 0.0
   )
   left_fraction = np.minimum(left_width / widths, 1.0)
   return (
      left_fraction[:, np.newaxis] * endpoints[0]
      + (1.0 - left_fraction[:, np.newaxis]) * endpoints[1]
   ).T


def neighborhood_bounds(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   """Return radius-one bounds for every interior cell and component."""

   neighbors = np.stack((state[:, :-2], state[:, 1:-1], state[:, 2:]))
   return np.min(neighbors, axis=0), np.max(neighbors, axis=0)


def dmp_reasons(
   candidate: np.ndarray,
   lower_source: np.ndarray,
   upper_source: np.ndarray,
   factor: float,
) -> dict[str, np.ndarray]:
   dens = COMPONENT_NAMES.index("dens")
   tau = COMPONENT_NAMES.index("tau")
   interior = candidate[:, 1:-1]
   return {
      "dens_low": interior[dens] < lower_source[dens] / factor,
      "dens_high": interior[dens] > factor * upper_source[dens],
      "tau_low": interior[tau] < lower_source[tau] / factor,
      "tau_high": interior[tau] > factor * upper_source[tau],
   }


def combined_mask(reasons: dict[str, np.ndarray]) -> np.ndarray:
   mask = np.zeros_like(next(iter(reasons.values())), dtype=bool)
   for reason in reasons.values():
      mask |= reason
   return mask


def first_cell(mask: np.ndarray) -> int | str:
   indices = np.flatnonzero(mask)
   return int(indices[0] + 1) if indices.size else ""


def cell_list(mask: np.ndarray) -> str:
   return ",".join(str(int(index + 1)) for index in np.flatnonzero(mask))


def thermodynamic_diagnostics(
   state: np.ndarray,
   gamma: float,
   entropy_floor: float,
   entropy_tolerance: float,
) -> tuple[int, int, float]:
   """Return recovery failures, entropy-floor violations, and max deficit."""

   primitive = physical.primitive_from_conserved(state.T, gamma)
   valid = (
      np.all(np.isfinite(primitive), axis=1)
      & (primitive[:, 0] > 0.0)
      & (primitive[:, 4] > 0.0)
   )
   failures = int(np.count_nonzero(~valid))
   entropy = np.full(len(primitive), np.nan)
   entropy[valid] = np.log(
      primitive[valid, 4] / primitive[valid, 0] ** gamma
   )
   violations = int(np.count_nonzero(
      valid & (entropy < entropy_floor - entropy_tolerance)
   ))
   if np.any(valid):
      max_deficit = max(0.0, entropy_floor - float(np.nanmin(entropy)))
   else:
      max_deficit = float("inf")
   return failures, violations, max_deficit


def ppm_shock_flattening(primitive: np.ndarray) -> np.ndarray:
   """Return the standard pressure/compression PPM flattening coefficient."""

   pressure = primitive[:, 4]
   normal_velocity = primitive[:, 1]
   coefficient = np.ones(len(primitive))
   for cell in range(2, len(primitive) - 2):
      pressure_jump = pressure[cell + 1] - pressure[cell - 1]
      wide_pressure_jump = pressure[cell + 2] - pressure[cell - 2]
      compression = normal_velocity[cell - 1] > normal_velocity[cell + 1]
      strong_jump = abs(pressure_jump) > (
         0.33 * min(pressure[cell - 1], pressure[cell + 1])
      )
      if (
         compression
         and strong_jump
         and abs(wide_pressure_jump) >= 1.0e-7
      ):
         coefficient[cell] = max(
            0.0,
            1.0 - 10.0 * max(
               0.0, pressure_jump / wide_pressure_jump - 0.75
            ),
         )
   return coefficient


def five_point_derivative_indicator(
   values: np.ndarray,
   reference: np.ndarray,
) -> np.ndarray:
   """Return the Berta five-point odd/even derivative ratio in one axis."""

   indicator = np.zeros(len(values))
   q_mm = values[:-4]
   q_m = values[1:-3]
   q_i = values[2:-2]
   q_p = values[3:-1]
   q_pp = values[4:]
   d0_q = np.abs(reference[2:-2])
   d1_q = np.abs(0.5 * (q_p - q_m))
   d2_q = np.abs(q_p - 2.0 * q_i + q_m)
   d3_q = np.abs(0.5 * (q_pp - 2.0 * q_p + 2.0 * q_m - q_mm))
   d4_q = np.abs(q_pp - 4.0 * q_p + 6.0 * q_i - 4.0 * q_m + q_mm)
   eta_odd = d3_q / (d0_q + d1_q + d3_q + 1.0e-22)
   eta_even = d4_q / (d0_q + d2_q + d4_q + 1.0e-22)
   indicator[2:-2] = np.maximum(eta_odd, eta_even)
   return indicator


def derivative_trouble_indicators(
   primitive: np.ndarray,
   gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
   """Return hydro-only and hydro-plus-magnetic detector amplitudes."""

   rho = primitive[:, 0]
   pressure = primitive[:, 4]
   enthalpy = 1.0 + gamma * pressure / ((gamma - 1.0) * rho)
   sound_speed = np.sqrt(np.maximum(gamma * pressure / (rho * enthalpy), 0.0))
   hydro = np.maximum(
      five_point_derivative_indicator(rho, rho),
      five_point_derivative_indicator(pressure, pressure),
   )
   for component in range(1, 4):
      hydro = np.maximum(
         hydro,
         five_point_derivative_indicator(
            primitive[:, component], sound_speed
         ),
      )

   magnetic = hydro.copy()
   magnetic_norm = np.linalg.norm(primitive[:, 5:8], axis=1)
   for component in range(5, 8):
      magnetic = np.maximum(
         magnetic,
         five_point_derivative_indicator(
            primitive[:, component], magnetic_norm
         ),
      )
   invalid = ~np.all(np.isfinite(primitive), axis=1)
   hydro[invalid] = np.inf
   magnetic[invalid] = np.inf
   return hydro, magnetic


def flag_with_halo(indicator: np.ndarray, threshold: float) -> np.ndarray:
   """Flag threshold crossings and their immediate cell neighbors."""

   direct = indicator > threshold
   halo = direct.copy()
   halo[1:] |= direct[:-1]
   halo[:-1] |= direct[1:]
   return halo


def audit_stage(
   *,
   profile_name: str,
   step: int,
   stage: str,
   path: Path,
   base: np.ndarray,
   stage_input: np.ndarray,
   factor: float,
   detector_threshold: float,
   entropy_floor: float,
   entropy_tolerance: float,
) -> tuple[dict[str, object], np.ndarray]:
   candidate, time = conserved_state(path)
   base_weight, fe_weight = STAGE_BLEND[stage]
   forward_euler = (candidate - base_weight * base) / fe_weight

   input_min, input_max = neighborhood_bounds(stage_input)
   base_min, base_max = neighborhood_bounds(base)
   current_reasons = dmp_reasons(
      candidate, input_min, input_max, factor
   )
   forward_euler_reasons = dmp_reasons(
      forward_euler, input_min, input_max, factor
   )
   union_reasons = dmp_reasons(
      candidate,
      np.minimum(input_min, base_min),
      np.maximum(input_max, base_max),
      factor,
   )
   current = combined_mask(current_reasons)
   forward_euler_mask = combined_mask(forward_euler_reasons)
   union = combined_mask(union_reasons)

   profile = PROFILE_BY_NAME[profile_name]
   spec = case_spec(profile.case)
   stage_primitive = physical.primitive_from_conserved(
      stage_input.T, float(spec["gamma"])
   )
   flattening = ppm_shock_flattening(stage_primitive)
   flattened = flattening[1:-1] < 1.0 - 32.0 * np.finfo(float).eps
   fully_flattened = flattening[1:-1] <= 32.0 * np.finfo(float).eps
   hydro_indicator, magnetic_indicator = derivative_trouble_indicators(
      stage_primitive, float(spec["gamma"])
   )
   hydro_direct = hydro_indicator[1:-1] > detector_threshold
   magnetic_direct = magnetic_indicator[1:-1] > detector_threshold
   hydro_halo = flag_with_halo(
      hydro_indicator, detector_threshold
   )[1:-1]
   magnetic_halo = flag_with_halo(
      magnetic_indicator, detector_threshold
   )[1:-1]
   candidate_thermo = thermodynamic_diagnostics(
      candidate, float(spec["gamma"]), entropy_floor, entropy_tolerance
   )
   forward_euler_thermo = thermodynamic_diagnostics(
      forward_euler, float(spec["gamma"]), entropy_floor, entropy_tolerance
   )
   row: dict[str, object] = {
      "profile": profile_name,
      "case": profile.case,
      "riemann": profile.riemann,
      "step": step,
      "stage": stage,
      "time": time,
      "current_dmp_cells": int(np.count_nonzero(current)),
      "ssp_fe_dmp_cells": int(np.count_nonzero(forward_euler_mask)),
      "base_input_union_dmp_cells": int(np.count_nonzero(union)),
      "shared_current_fe_cells": int(
         np.count_nonzero(current & forward_euler_mask)
      ),
      "current_only_cells": int(
         np.count_nonzero(current & ~forward_euler_mask)
      ),
      "ssp_fe_only_cells": int(
         np.count_nonzero(forward_euler_mask & ~current)
      ),
      "current_dmp_flattened_cells": int(
         np.count_nonzero(current & flattened)
      ),
      "current_dmp_fully_flattened_cells": int(
         np.count_nonzero(current & fully_flattened)
      ),
      "hydro_detector_cells": int(np.count_nonzero(hydro_direct)),
      "hydro_detector_halo_cells": int(np.count_nonzero(hydro_halo)),
      "magnetic_detector_cells": int(np.count_nonzero(magnetic_direct)),
      "magnetic_detector_halo_cells": int(np.count_nonzero(magnetic_halo)),
      "current_dmp_hydro_detector_cells": int(
         np.count_nonzero(current & hydro_direct)
      ),
      "current_dmp_hydro_halo_cells": int(
         np.count_nonzero(current & hydro_halo)
      ),
      "current_dmp_magnetic_detector_cells": int(
         np.count_nonzero(current & magnetic_direct)
      ),
      "current_dmp_magnetic_halo_cells": int(
         np.count_nonzero(current & magnetic_halo)
      ),
      "max_hydro_indicator": float(np.nanmax(hydro_indicator)),
      "max_magnetic_indicator": float(np.nanmax(magnetic_indicator)),
      "ssp_fe_dmp_flattened_cells": int(
         np.count_nonzero(forward_euler_mask & flattened)
      ),
      "current_cell_indices": cell_list(current),
      "current_shock_cell_indices": cell_list(current & flattened),
      "current_hydro_halo_cell_indices": cell_list(current & hydro_halo),
      "current_magnetic_halo_cell_indices": cell_list(
         current & magnetic_halo
      ),
      "first_current_cell": first_cell(current),
      "first_ssp_fe_cell": first_cell(forward_euler_mask),
      "candidate_c2p_failures": candidate_thermo[0],
      "candidate_entropy_floor_violations": candidate_thermo[1],
      "candidate_max_entropy_deficit": candidate_thermo[2],
      "ssp_fe_c2p_failures": forward_euler_thermo[0],
      "ssp_fe_entropy_floor_violations": forward_euler_thermo[1],
      "ssp_fe_max_entropy_deficit": forward_euler_thermo[2],
      "run_dir": repo_relative(path.parents[1]),
   }
   for label, reasons in (
      ("current", current_reasons),
      ("ssp_fe", forward_euler_reasons),
   ):
      for reason, mask in reasons.items():
         row[f"{label}_{reason}_cells"] = int(np.count_nonzero(mask))
   return row, candidate


def audit_profile(
   profile_name: str,
   factor: float,
   detector_threshold: float,
   entropy_tolerance: float,
) -> list[dict[str, object]]:
   profile = PROFILE_BY_NAME[profile_name]
   run_dir = RUN_ROOT / f"onset_{profile.name}_n{profile.cells}"
   paths = [
      item for item in stage_output_paths(run_dir)
      if item[1] in STAGE_BLEND
   ]
   if not paths:
      raise FileNotFoundError(
         f"missing SSPRK3 stage trace under {repo_relative(run_dir)}"
      )

   rows: list[dict[str, object]] = []
   base = initial_state(paths[0][2], profile.case)
   initial_primitive = physical.primitive_from_conserved(
      base.T, float(case_spec(profile.case)["gamma"])
   )
   initial_entropy = np.log(
      initial_primitive[:, 4]
      / initial_primitive[:, 0] ** float(case_spec(profile.case)["gamma"])
   )
   entropy_floor = float(np.nanmin(initial_entropy))
   current_step = paths[0][0]
   stage_input = base
   final_candidate = base
   for step, stage, path in paths:
      if step != current_step:
         base = final_candidate
         stage_input = base
         current_step = step
      row, candidate = audit_stage(
         profile_name=profile_name,
         step=step,
         stage=stage,
         path=path,
         base=base,
         stage_input=stage_input,
         factor=factor,
         detector_threshold=detector_threshold,
         entropy_floor=entropy_floor,
         entropy_tolerance=entropy_tolerance,
      )
      rows.append(row)
      print(
         f"{profile_name} step={step} {stage}: "
         f"current/SSP-FE/union={row['current_dmp_cells']}/"
         f"{row['ssp_fe_dmp_cells']}/"
         f"{row['base_input_union_dmp_cells']} "
         f"current-only/FE-only={row['current_only_cells']}/"
         f"{row['ssp_fe_only_cells']} "
         f"current+shock={row['current_dmp_flattened_cells']}"
         f" detector-halo(hydro/mhd)="
         f"{row['current_dmp_hydro_halo_cells']}/"
         f"{row['current_dmp_magnetic_halo_cells']}"
      )
      stage_input = candidate
      final_candidate = candidate
   return rows


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--profile",
      dest="profiles",
      action="append",
      choices=DEFAULT_PROFILES,
      help="saved unguarded stage trace to audit; repeatable (default: all)",
   )
   parser.add_argument("--dmp-factor", type=float, default=1.05)
   parser.add_argument("--detector-threshold", type=float, default=0.005)
   parser.add_argument("--entropy-tolerance", type=float, default=1.0e-8)
   args = parser.parse_args()
   if args.dmp_factor < 1.0:
      parser.error("--dmp-factor must be at least 1")
   if args.detector_threshold < 0.0:
      parser.error("--detector-threshold must be nonnegative")
   if args.entropy_tolerance < 0.0:
      parser.error("--entropy-tolerance must be nonnegative")

   profiles = list(dict.fromkeys(args.profiles or DEFAULT_PROFILES))
   rows: list[dict[str, object]] = []
   for profile_name in profiles:
      rows.extend(
         audit_profile(
            profile_name,
            args.dmp_factor,
            args.detector_threshold,
            args.entropy_tolerance,
         )
      )

   columns = [
      "profile", "case", "riemann", "step", "stage", "time",
      "current_dmp_cells", "ssp_fe_dmp_cells",
      "base_input_union_dmp_cells", "shared_current_fe_cells",
      "current_only_cells", "ssp_fe_only_cells", "first_current_cell",
      "current_dmp_flattened_cells",
      "current_dmp_fully_flattened_cells",
      "hydro_detector_cells", "hydro_detector_halo_cells",
      "magnetic_detector_cells", "magnetic_detector_halo_cells",
      "current_dmp_hydro_detector_cells",
      "current_dmp_hydro_halo_cells",
      "current_dmp_magnetic_detector_cells",
      "current_dmp_magnetic_halo_cells",
      "max_hydro_indicator", "max_magnetic_indicator",
      "ssp_fe_dmp_flattened_cells", "first_ssp_fe_cell",
      "current_cell_indices", "current_shock_cell_indices",
      "current_hydro_halo_cell_indices",
      "current_magnetic_halo_cell_indices",
      "candidate_c2p_failures",
      "candidate_entropy_floor_violations",
      "candidate_max_entropy_deficit",
      "ssp_fe_c2p_failures",
      "ssp_fe_entropy_floor_violations",
      "ssp_fe_max_entropy_deficit",
      "current_dens_low_cells", "current_dens_high_cells",
      "current_tau_low_cells", "current_tau_high_cells",
      "ssp_fe_dens_low_cells", "ssp_fe_dens_high_cells",
      "ssp_fe_tau_low_cells", "ssp_fe_tau_high_cells", "run_dir",
   ]
   write_tsv(RESULTS, rows, columns)
   print(f"wrote {repo_relative(RESULTS)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
