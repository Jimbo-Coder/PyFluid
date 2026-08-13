#!/usr/bin/env python3
"""Test whether the isolated-wave SRMHD prototype activates before Balsara artifacts.

This tool is deliberately offline.  It reads production SSPRK3 debug-stage
states, applies the standalone research C2P and strict two-plateau/full-RH
detector, and records any accepted cell or face wave.  Exact Balsara data do
not select a candidate.  A detector that remains inactive through the first
artifact cannot explain or repair that artifact, even if it is exact on its
isolated-wave controls.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import research_physical_wave_subcell_srmhd as physical
from diagnose_rmhd_balsara_onset import (
   COMPONENT_NAMES,
   PROFILE_BY_NAME,
   RUN_ROOT,
   case_spec,
   exact_average_at_time,
   stage_output_paths,
)
from diagnose_rmhd_operator import (
   primitive_state_and_flux,
   recover_conserved,
)
from plotfile_reader import (
   parse_plot_bounds,
   parse_plot_header_1d,
   read_composite_state_1d,
)
from validate_rmhd import EXACT_BALSARA_ROOT, read_last_exact_balsara_profile
from validation_common import repo_relative, write_tsv


DEFAULT_PROFILES = (
   "b4_ssprk3_stage_trace_ppm4_guarded",
   "b4_ssprk3_stage_trace_ppm4_unguarded",
   "b4_ssprk3_stage_trace_pwc",
   "b4_ssprk3_stage_trace_mp5_guarded",
   "b3_ssprk3_stage_trace_mp5_unguarded",
   "b3_ssprk3_stage_trace_mp5_guarded",
)
RESULTS = RUN_ROOT / "balsara_physical_wave_activation.tsv"
HARTEN_RESULTS = RUN_ROOT / "balsara_harten_subcell_audit.tsv"
HARTEN_CANDIDATES = RUN_ROOT / "balsara_harten_subcell_candidates.tsv"


@dataclasses.dataclass(frozen=True)
class HartenFit:
   cell: int
   degree: int
   left_fraction: float
   mixture_residual: float
   left_state: np.ndarray
   right_state: np.ndarray
   component_scale: np.ndarray
   normalized_jump: float
   interior_distance: float
   endpoint_recoverable: bool = False
   speed: float = np.nan
   jump_residual: float = np.inf
   maximum_jump_residual: float = np.inf
   maximum_jump_component: str = ""


def cell_average_matrix(offsets: np.ndarray, degree: int) -> np.ndarray:
   """Map polynomial coefficients to unit-cell averages at ``offsets``."""
   matrix = np.empty((len(offsets), degree + 1))
   for row, offset in enumerate(offsets):
      lower = offset - 0.5
      upper = offset + 0.5
      for power in range(degree + 1):
         matrix[row, power] = (
            upper ** (power + 1) - lower ** (power + 1)
         ) / (power + 1)
   return matrix


def side_polynomial(
   conserved: np.ndarray,
   cell: int,
   degree: int,
   side: str,
) -> np.ndarray:
   if side == "left":
      indices = np.arange(cell - degree - 1, cell)
   elif side == "right":
      indices = np.arange(cell + 1, cell + degree + 2)
   else:
      raise ValueError(f"unsupported side {side}")
   offsets = indices.astype(float) - cell
   matrix = cell_average_matrix(offsets, degree)
   return np.linalg.solve(matrix, conserved[indices])


def polynomial_integral(
   coefficients: np.ndarray,
   lower: float,
   upper: float,
) -> np.ndarray:
   result = np.zeros(coefficients.shape[1])
   for power, coefficient in enumerate(coefficients):
      result += coefficient * (
         upper ** (power + 1) - lower ** (power + 1)
      ) / (power + 1)
   return result


def harten_cell_fit(
   conserved: np.ndarray,
   cell: int,
   degree: int,
) -> HartenFit:
   """Fit one shared Harten-style jump between smooth side polynomials."""
   left = side_polynomial(conserved, cell, degree, "left")
   right = side_polynomial(conserved, cell, degree, "right")
   first = cell - degree - 1
   last = cell + degree + 2
   window = conserved[first:last]
   common_scale = max(float(np.max(np.abs(window))), np.finfo(float).tiny)
   scale = np.maximum(
      np.maximum(np.max(np.abs(window), axis=0), np.ptp(window, axis=0)),
      64.0 * np.finfo(float).eps * common_scale,
   )

   def mixed_average(left_fraction: float) -> np.ndarray:
      location = -0.5 + left_fraction
      return (
         polynomial_integral(left, -0.5, location)
         + polynomial_integral(right, location, 0.5)
      )

   def objective(left_fraction: float) -> float:
      residual = (mixed_average(left_fraction) - conserved[cell]) / scale
      return float(np.dot(residual, residual))

   bounded = minimize_scalar(
      objective,
      bounds=(0.0, 1.0),
      method="bounded",
      options={"xatol": 2.0e-13, "maxiter": 128},
   )
   candidates = (0.0, float(bounded.x), 1.0)
   left_fraction = min(candidates, key=objective)
   location = -0.5 + left_fraction
   mixture = mixed_average(left_fraction)
   mixture_residual = float(
      np.max(np.abs(mixture - conserved[cell]) / scale)
   )
   powers = np.asarray([location**power for power in range(degree + 1)])
   left_state = powers @ left
   right_state = powers @ right
   return HartenFit(
      cell=cell,
      degree=degree,
      left_fraction=left_fraction,
      mixture_residual=mixture_residual,
      left_state=left_state,
      right_state=right_state,
      component_scale=scale,
      normalized_jump=float(
         np.max(np.abs(right_state - left_state) / scale)
      ),
      interior_distance=min(left_fraction, 1.0 - left_fraction),
   )


def score_harten_fits(
   fits: list[HartenFit],
   gamma: float,
) -> list[HartenFit]:
   if not fits:
      return []
   endpoints = np.column_stack([
      endpoint
      for fit in fits
      for endpoint in (fit.left_state, fit.right_state)
   ])
   recovery = recover_conserved(endpoints, gamma)
   recoverable_pairs = [
      index for index in range(len(fits))
      if recovery.success[2 * index] and recovery.success[2 * index + 1]
   ]
   if recoverable_pairs:
      recoverable_indices = np.asarray([
         endpoint
         for index in recoverable_pairs
         for endpoint in (2 * index, 2 * index + 1)
      ])
      recovered_state, flux = primitive_state_and_flux(
         recovery.primitive[:, recoverable_indices],
         gamma,
      )
      pair_slot = {
         fit_index: slot for slot, fit_index in enumerate(recoverable_pairs)
      }
   else:
      recovered_state = np.empty((endpoints.shape[0], 0))
      flux = np.empty_like(recovered_state)
      pair_slot = {}
   scored: list[HartenFit] = []
   for index, fit in enumerate(fits):
      if index not in pair_slot:
         scored.append(fit)
         continue
      slot = pair_slot[index]
      left_index = 2 * slot
      right_index = left_index + 1
      state_jump = (
         recovered_state[:, right_index] - recovered_state[:, left_index]
      )
      flux_jump = flux[:, right_index] - flux[:, left_index]
      scaled_state_jump = state_jump / fit.component_scale
      scaled_flux_jump = flux_jump / fit.component_scale
      jump_norm2 = float(np.dot(scaled_state_jump, scaled_state_jump))
      if jump_norm2 <= np.finfo(float).tiny:
         scored.append(dataclasses.replace(fit, endpoint_recoverable=True))
         continue
      speed = float(
         np.dot(scaled_state_jump, scaled_flux_jump) / jump_norm2
      )
      scaled_speed_jump = speed * scaled_state_jump
      residual = scaled_flux_jump - scaled_speed_jump
      common_scale = max(
         float(np.linalg.norm(scaled_flux_jump)),
         float(np.linalg.norm(scaled_speed_jump)),
         np.finfo(float).tiny,
      )
      jump_residual = float(np.linalg.norm(residual) / common_scale)
      component_denominator = np.maximum(
         np.maximum(np.abs(scaled_flux_jump), np.abs(scaled_speed_jump)),
         64.0 * np.finfo(float).eps * common_scale,
      )
      scored.append(dataclasses.replace(
         fit,
         endpoint_recoverable=True,
         speed=speed,
         jump_residual=jump_residual,
         maximum_jump_residual=float(
            np.max(np.abs(residual) / component_denominator)
         ),
         maximum_jump_component=COMPONENT_NAMES[int(np.argmax(
            np.abs(residual) / component_denominator
         ))],
      ))
   return scored


def harten_fits(
   conserved: np.ndarray,
   degree: int,
   gamma: float,
) -> list[HartenFit]:
   first = degree + 1
   stop = len(conserved) - degree - 1
   return score_harten_fits(
      [harten_cell_fit(conserved, cell, degree) for cell in range(first, stop)],
      gamma,
   )


def harten_summary_rows(
   *,
   profile_name: str,
   step: int,
   stage: str,
   time: float,
   source: str,
   conserved: np.ndarray,
   degrees: tuple[int, ...],
   prob_lo: float,
   prob_hi: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   profile = PROFILE_BY_NAME[profile_name]
   gamma = float(case_spec(profile.case)["gamma"])
   rows: list[dict[str, object]] = []
   candidate_rows: list[dict[str, object]] = []
   dx = (prob_hi - prob_lo) / len(conserved)
   for degree in degrees:
      fits = harten_fits(conserved, degree, gamma)
      recoverable = [fit for fit in fits if fit.endpoint_recoverable]
      finite = [
         fit for fit in recoverable
         if np.isfinite(fit.maximum_jump_residual)
      ]
      interior = [fit for fit in finite if fit.interior_distance > 1.0e-8]
      nontrivial = [
         fit for fit in interior
         if fit.normalized_jump > 1.0e-3 and abs(fit.speed) <= 1.0
      ]
      strong = [fit for fit in nontrivial if fit.normalized_jump > 1.0e-1]
      best = min(
         nontrivial,
         key=lambda fit: max(
            fit.mixture_residual, fit.maximum_jump_residual
         ),
         default=None,
      )
      best_strong = min(
         strong,
         key=lambda fit: max(
            fit.mixture_residual, fit.maximum_jump_residual
         ),
         default=None,
      )
      for fit in fits:
         candidate_rows.append({
            "profile": profile.name,
            "case": profile.case,
            "source": source,
            "step": step,
            "stage": stage,
            "time": time,
            "degree": degree,
            "cell": fit.cell,
            "left_fraction": fit.left_fraction,
            "coordinate": prob_lo + (fit.cell + fit.left_fraction) * dx,
            "interior_distance": fit.interior_distance,
            "normalized_jump": fit.normalized_jump,
            "mixture_residual": fit.mixture_residual,
            "endpoint_recoverable": int(fit.endpoint_recoverable),
            "speed": fit.speed,
            "causal_speed": int(np.isfinite(fit.speed) and abs(fit.speed) <= 1.0),
            "jump_residual": fit.jump_residual,
            "maximum_jump_residual": fit.maximum_jump_residual,
            "maximum_jump_component": fit.maximum_jump_component,
            "joint_residual": max(
               fit.mixture_residual, fit.maximum_jump_residual
            ),
         })
      rows.append({
         "profile": profile.name,
         "case": profile.case,
         "source": source,
         "step": step,
         "stage": stage,
         "time": time,
         "degree": degree,
         "fits": len(fits),
         "recoverable_fits": len(recoverable),
         "interior_fits": len(interior),
         "nontrivial_fits": len(nontrivial),
         "strong_fits": len(strong),
         "minimum_mixture_residual": min(
            (fit.mixture_residual for fit in fits), default=np.inf
         ),
         "minimum_jump_residual": min(
            (fit.jump_residual for fit in finite), default=np.inf
         ),
         "minimum_maximum_jump_residual": min(
            (fit.maximum_jump_residual for fit in finite), default=np.inf
         ),
         "joint_1e2": sum(
            max(fit.mixture_residual, fit.maximum_jump_residual) <= 1.0e-2
            for fit in nontrivial
         ),
         "joint_1e3": sum(
            max(fit.mixture_residual, fit.maximum_jump_residual) <= 1.0e-3
            for fit in nontrivial
         ),
         "strong_joint_1e2": sum(
            max(fit.mixture_residual, fit.maximum_jump_residual) <= 1.0e-2
            for fit in strong
         ),
         "best_joint_residual": (
            max(best.mixture_residual, best.maximum_jump_residual)
            if best is not None else np.inf
         ),
         "best_cell": best.cell if best is not None else "",
         "best_left_fraction": (
            best.left_fraction if best is not None else ""
         ),
         "best_speed": best.speed if best is not None else "",
         "best_coordinate": (
            prob_lo + (best.cell + best.left_fraction) * dx
            if best is not None else ""
         ),
         "best_normalized_jump": (
            best.normalized_jump if best is not None else ""
         ),
         "best_maximum_jump_residual": (
            best.maximum_jump_residual if best is not None else ""
         ),
         "best_maximum_jump_component": (
            best.maximum_jump_component if best is not None else ""
         ),
         "best_strong_joint_residual": (
            max(
               best_strong.mixture_residual,
               best_strong.maximum_jump_residual,
            )
            if best_strong is not None else np.inf
         ),
         "best_strong_cell": (
            best_strong.cell if best_strong is not None else ""
         ),
         "best_strong_coordinate": (
            prob_lo + (best_strong.cell + best_strong.left_fraction) * dx
            if best_strong is not None else ""
         ),
      })
   return rows, candidate_rows


def prototype_primitive(state: tuple[float, ...]) -> np.ndarray:
   rho, pressure, v1, v2, v3, b1, b2, b3 = state
   return np.asarray((rho, v1, v2, v3, pressure, b1, b2, b3))


def initial_row(profile_name: str) -> dict[str, object]:
   profile = PROFILE_BY_NAME[profile_name]
   spec = case_spec(profile.case)
   left = prototype_primitive(spec["left"])
   right = prototype_primitive(spec["right"])
   candidate = physical.fit_physical_wave(
      left,
      left,
      right,
      float(spec["gamma"]),
      allow_endpoint=True,
   )
   return {
      "profile": profile.name,
      "case": profile.case,
      "step": 0,
      "stage": "initial_face",
      "time": 0.0,
      "offline_c2p_failures": 0,
      "cell_wave_accepts": 0,
      "face_wave_accepts": int(candidate is not None),
      "first_accept_location": "initial_face" if candidate is not None else "",
      "first_accept_index": "",
      "first_accept_speed": candidate.speed if candidate is not None else "",
      "first_accept_relative_jump": (
         relative_conserved_jump(candidate, float(spec["gamma"]))
         if candidate is not None else ""
      ),
      "maximum_mixture_residual": (
         candidate.mixture_residual if candidate is not None else ""
      ),
      "maximum_jump_residual": (
         candidate.jump_residual if candidate is not None else ""
      ),
      "run_dir": "",
   }


def relative_conserved_jump(
   candidate: physical.PhysicalWave,
   gamma: float,
) -> float:
   endpoints = physical.conserved_array(
      np.asarray((candidate.left, candidate.right)), gamma
   )
   scale = max(
      float(np.linalg.norm(endpoints[0])),
      float(np.linalg.norm(endpoints[1])),
      np.finfo(float).tiny,
   )
   return float(np.linalg.norm(endpoints[0] - endpoints[1]) / scale)


def read_stage_conserved(path: Path) -> tuple[np.ndarray, float]:
   names, time = parse_plot_header_1d(path)
   output = read_composite_state_1d(path, names)
   conserved = np.vstack([
      output[names.index(name)] for name in COMPONENT_NAMES
   ]).T
   return conserved, time


def stage_row(
   profile_name: str,
   step: int,
   stage: str,
   path: Path,
) -> dict[str, object]:
   profile = PROFILE_BY_NAME[profile_name]
   spec = case_spec(profile.case)
   conserved, time = read_stage_conserved(path)
   primitive = physical.primitive_from_conserved(
      conserved, float(spec["gamma"])
   )
   c2p_failures = int(np.count_nonzero(~np.all(np.isfinite(primitive), axis=1)))

   cell_candidates: list[tuple[int, physical.PhysicalWave]] = []
   face_candidates: list[tuple[int, physical.PhysicalWave]] = []
   if c2p_failures == 0:
      for cell in range(2, len(primitive) - 2):
         candidate = physical.fit_cell_wave(
            primitive, cell, float(spec["gamma"]), periodic=False
         )
         if candidate is not None:
            cell_candidates.append((cell, candidate))
      for left_cell in range(1, len(primitive) - 2):
         candidate = physical.fit_face_wave(
            primitive, left_cell, float(spec["gamma"]), periodic=False
         )
         if candidate is not None:
            face_candidates.append((left_cell + 1, candidate))
   candidates = [
      candidate for _index, candidate in (*cell_candidates, *face_candidates)
   ]
   first_location = ""
   first_index: int | str = ""
   first_speed: float | str = ""
   first_relative_jump: float | str = ""
   if cell_candidates:
      first_location = "mixed_cell"
      first_index, first_candidate = cell_candidates[0]
      first_speed = first_candidate.speed
      first_relative_jump = relative_conserved_jump(
         first_candidate, float(spec["gamma"])
      )
   elif face_candidates:
      first_location = "face"
      first_index, first_candidate = face_candidates[0]
      first_speed = first_candidate.speed
      first_relative_jump = relative_conserved_jump(
         first_candidate, float(spec["gamma"])
      )
   return {
      "profile": profile.name,
      "case": profile.case,
      "step": step,
      "stage": stage,
      "time": time,
      "offline_c2p_failures": c2p_failures,
      "cell_wave_accepts": len(cell_candidates),
      "face_wave_accepts": len(face_candidates),
      "first_accept_location": first_location,
      "first_accept_index": first_index,
      "first_accept_speed": first_speed,
      "first_accept_relative_jump": first_relative_jump,
      "maximum_mixture_residual": (
         max(candidate.mixture_residual for candidate in candidates)
         if candidates else ""
      ),
      "maximum_jump_residual": (
         max(candidate.jump_residual for candidate in candidates)
         if candidates else ""
      ),
      "run_dir": repo_relative(path.parents[1]),
   }


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--profile",
      dest="profiles",
      action="append",
      choices=DEFAULT_PROFILES,
      help="saved stage trace to audit; repeatable (default: all)",
   )
   parser.add_argument(
      "--harten-degree",
      type=int,
      nargs="+",
      default=(0, 1, 2),
      help="smooth-side polynomial degrees for the Harten audit",
   )
   parser.add_argument("--skip-physical-wave", action="store_true")
   args = parser.parse_args()
   profiles = list(dict.fromkeys(args.profiles or DEFAULT_PROFILES))
   degrees = tuple(dict.fromkeys(args.harten_degree))
   if any(degree < 0 or degree > 4 for degree in degrees):
      raise ValueError("--harten-degree values must lie in 0..4")

   rows: list[dict[str, object]] = []
   harten_rows: list[dict[str, object]] = []
   harten_candidates: list[dict[str, object]] = []
   exact_profiles: dict[int, np.ndarray] = {}
   for profile_name in profiles:
      profile = PROFILE_BY_NAME[profile_name]
      spec = case_spec(profile.case)
      run_dir = RUN_ROOT / f"onset_{profile.name}_n{profile.cells}"
      paths = stage_output_paths(run_dir)
      if not paths:
         raise FileNotFoundError(
            f"missing SSPRK3 stage trace under {repo_relative(run_dir)}"
         )
      if not args.skip_physical_wave:
         rows.append(initial_row(profile_name))
      exact_profile = exact_profiles.setdefault(
         profile.case,
         read_last_exact_balsara_profile(
            EXACT_BALSARA_ROOT / f"solution_b{profile.case}.dat"
         ),
      )
      for step, stage, path in paths:
         conserved, time = read_stage_conserved(path)
         prob_lo, prob_hi = parse_plot_bounds(path)
         if not args.skip_physical_wave:
            row = stage_row(profile_name, step, stage, path)
            rows.append(row)
            print(
               f"{profile_name} step={step} {stage}: "
               f"c2p_failures={row['offline_c2p_failures']} "
               f"cell/face accepts={row['cell_wave_accepts']}/"
               f"{row['face_wave_accepts']}"
            )
         numerical_rows, numerical_candidates = harten_summary_rows(
            profile_name=profile_name,
            step=step,
            stage=stage,
            time=time,
            source="numerical",
            conserved=conserved,
            degrees=degrees,
            prob_lo=float(prob_lo[0]),
            prob_hi=float(prob_hi[0]),
         )
         cell_edges = np.linspace(prob_lo[0], prob_hi[0], len(conserved) + 1)
         exact_conserved = exact_average_at_time(
            exact_profile,
            spec,
            time,
            cell_edges,
         ).T
         exact_rows, exact_candidates = harten_summary_rows(
            profile_name=profile_name,
            step=step,
            stage=stage,
            time=time,
            source="exact",
            conserved=exact_conserved,
            degrees=degrees,
            prob_lo=float(prob_lo[0]),
            prob_hi=float(prob_hi[0]),
         )
         harten_rows.extend(numerical_rows)
         harten_rows.extend(exact_rows)
         harten_candidates.extend(numerical_candidates)
         harten_candidates.extend(exact_candidates)
         for harten_row in numerical_rows:
            matching_exact = next(
               row for row in exact_rows
               if row["degree"] == harten_row["degree"]
            )
            print(
               f"{profile_name} step={step} {stage} "
               f"degree={harten_row['degree']}: "
               f"best numerical/exact joint="
               f"{harten_row['best_joint_residual']:.3e}/"
               f"{matching_exact['best_joint_residual']:.3e} "
               f"joint1e-2={harten_row['joint_1e2']}/"
               f"{matching_exact['joint_1e2']}"
            )

   if not args.skip_physical_wave:
      write_tsv(
         RESULTS,
         rows,
         [
            "profile", "case", "step", "stage", "time",
            "offline_c2p_failures", "cell_wave_accepts", "face_wave_accepts",
            "first_accept_location", "first_accept_index", "first_accept_speed",
            "first_accept_relative_jump",
            "maximum_mixture_residual", "maximum_jump_residual", "run_dir",
         ],
      )
      accepted = sum(
         int(row["cell_wave_accepts"]) + int(row["face_wave_accepts"])
         for row in rows
      )
      print(f"accepted prototype stage observations={accepted}")
      print(f"wrote {repo_relative(RESULTS)}")
   write_tsv(HARTEN_RESULTS, harten_rows, list(harten_rows[0]))
   write_tsv(
      HARTEN_CANDIDATES,
      harten_candidates,
      list(harten_candidates[0]),
   )
   print(f"wrote {repo_relative(HARTEN_RESULTS)}")
   print(f"wrote {repo_relative(HARTEN_CANDIDATES)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
