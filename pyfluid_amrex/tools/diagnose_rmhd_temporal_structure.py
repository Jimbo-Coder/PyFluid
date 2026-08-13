#!/usr/bin/env python3
"""Audit temporal structure in the difficult Balsara 3/4 SRMHD fans.

This is an offline diagnostic and rough-prototype harness.  It compares saved
accepted AMReX states with time-matched exact finite-volume averages, including
the primitive state recovered from each cell average.  A deliberately simple
two-history detector asks whether a spatial second difference grows faster
than the largest neighboring second difference in either prior accepted state.

The detector is first applied to exact Balsara 1--5 histories.  Its only
numerical intervention is a complete SSPRK3 replay from a saved Balsara-4
state: both traces of every marked shared face use production PWL-MC rather
than PPM4.  Exact data never enter the replay or the evolution code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diagnose_rmhd_balsara_onset import (  # noqa: E402
   case_spec,
   exact_average_at_time,
)
from diagnose_rmhd_operator import (  # noqa: E402
   exact_primitives_at,
   primitive_state_and_flux,
   read_saved_state,
   reconstruct_faces,
   recover_conserved,
   rhs_from_faces,
)
from plotfile_reader import parse_plot_header_1d  # noqa: E402
from scan_rmhd_balsara_reconstruction import COMPONENT_NAMES  # noqa: E402
from validate_rmhd import (  # noqa: E402
   EXACT_BALSARA_ROOT,
   RUN_ROOT,
   read_last_exact_balsara_profile,
)
from validation_common import ROOT, repo_relative, write_tsv  # noqa: E402


STRUCTURE_RESULTS = RUN_ROOT / "balsara_temporal_structure.tsv"
DETECTOR_RESULTS = RUN_ROOT / "balsara_temporal_detector.tsv"
PROTOTYPE_RESULTS = RUN_ROOT / "balsara_temporal_pwl_prototype.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_balsara_temporal_structure.png"
PRIMITIVE_NAMES = ("rho", "v1", "v2", "v3", "pressure", "b1", "b2", "b3")
B4_RUN = RUN_ROOT / "onset_b4_ssprk3_stage_trace_ppm4_unguarded_n256"
GROWTH_THRESHOLD = 3.0
ACTIVE_AMPLITUDE = 1.0e-3


def exact_history(
   case: int,
   cells: int,
   times: list[float],
) -> tuple[list[np.ndarray], np.ndarray, dict[str, object], np.ndarray]:
   spec = case_spec(case)
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   edges = np.linspace(-1.0, 1.0, cells + 1)
   history = [
      exact_average_at_time(profile, spec, time, edges) for time in times
   ]
   return history, edges, spec, profile


def production_initial_state(
   spec: dict[str, object],
   profile: np.ndarray,
   edges: np.ndarray,
) -> np.ndarray:
   centers = 0.5 * (edges[:-1] + edges[1:])
   primitive = exact_primitives_at(profile, spec, 0.0, centers)
   conserved, _ = primitive_state_and_flux(primitive, float(spec["gamma"]))
   return conserved


def recovered_primitive(state: np.ndarray, gamma: float) -> np.ndarray:
   recovery = recover_conserved(state, gamma)
   if not np.all(recovery.success):
      failed = np.flatnonzero(~recovery.success)
      raise RuntimeError(f"C2P failed in cells {failed[:12].tolist()}")
   return recovery.primitive


def representation_rows(
   *,
   case: int,
   step: int,
   time: float,
   numerical: np.ndarray,
   exact: np.ndarray,
   spec: dict[str, object],
   profile: np.ndarray,
   edges: np.ndarray,
) -> list[dict[str, object]]:
   gamma = float(spec["gamma"])
   numerical_primitive = recovered_primitive(numerical, gamma)
   exact_primitive = recovered_primitive(exact, gamma)
   centers = 0.5 * (edges[:-1] + edges[1:])
   exact_point_primitive = exact_primitives_at(
      profile, spec, time, centers
   )
   exact_point_conserved, _ = primitive_state_and_flux(
      exact_point_primitive, gamma
   )
   rows: list[dict[str, object]] = []
   for representation, names, candidate, reference, point_reference in (
      (
         "conserved",
         COMPONENT_NAMES,
         numerical,
         exact,
         exact_point_conserved,
      ),
      (
         "recovered_primitive",
         PRIMITIVE_NAMES,
         numerical_primitive,
         exact_primitive,
         exact_point_primitive,
      ),
   ):
      scale = np.maximum(
         np.max(np.abs(point_reference), axis=1),
         np.ptp(point_reference, axis=1),
      )
      scale = np.maximum(scale, 1.0e-14)
      error = np.abs(candidate - reference) / scale[:, np.newaxis]
      candidate_tv = np.sum(np.abs(np.diff(candidate, axis=1)), axis=1)
      reference_tv = np.sum(np.abs(np.diff(reference, axis=1)), axis=1)
      candidate_highpass = np.sum(
         np.abs(np.diff(candidate, n=2, axis=1)), axis=1
      )
      reference_highpass = np.sum(
         np.abs(np.diff(reference, n=2, axis=1)), axis=1
      )
      physical_min = np.min(point_reference, axis=1)
      physical_max = np.max(point_reference, axis=1)
      overshoot = np.maximum(
         np.maximum(physical_min[:, np.newaxis] - candidate, 0.0),
         np.maximum(candidate - physical_max[:, np.newaxis], 0.0),
      ) / scale[:, np.newaxis]
      for component, name in enumerate(names):
         rows.append({
            "case": case,
            "step": step,
            "time": time,
            "representation": representation,
            "component": name,
            "normalized_l1": float(np.mean(error[component])),
            "normalized_linf": float(np.max(error[component])),
            "tv_ratio_to_exact_fv": (
               float(candidate_tv[component] / reference_tv[component])
               if reference_tv[component] > 1.0e-14 * scale[component]
               else 1.0
            ),
            "highpass_ratio_to_exact_fv": (
               float(
                  candidate_highpass[component]
                  / reference_highpass[component]
               )
               if reference_highpass[component] > 1.0e-14 * scale[component]
               else 1.0
            ),
            "normalized_physical_overshoot": float(
               np.max(overshoot[component])
            ),
         })
   return rows


def second_difference(state: np.ndarray) -> np.ndarray:
   return np.abs(state[:, :-2] - 2.0 * state[:, 1:-1] + state[:, 2:])


def neighbor_maximum(values: np.ndarray, radius: int) -> np.ndarray:
   padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
   return np.maximum.reduce([
      padded[:, offset : offset + values.shape[1]]
      for offset in range(2 * radius + 1)
   ])


def temporal_growth(
   previous_two: tuple[np.ndarray, np.ndarray],
   current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   current_highpass = second_difference(current)
   prior_envelope = np.maximum(
      # At CFL 0.9, a causal front can move at most one cell per accepted
      # step.  The older state therefore needs a two-cell spatial envelope.
      neighbor_maximum(second_difference(previous_two[0]), 2),
      neighbor_maximum(second_difference(previous_two[1]), 1),
   )
   scale = np.maximum(
      np.max(np.abs(current), axis=1), np.ptp(current, axis=1)
   )
   scale = np.maximum(scale, 1.0e-14)
   active = current_highpass > ACTIVE_AMPLITUDE * scale[:, np.newaxis]
   denominator = np.maximum(prior_envelope, 1.0e-12 * scale[:, np.newaxis])
   ratio = np.where(active, current_highpass / denominator, 0.0)
   marked = active & (ratio > GROWTH_THRESHOLD)
   return ratio, active, marked


def detector_rows(
   *,
   case: int,
   source: str,
   histories: list[np.ndarray],
   times: list[float],
   gamma: float,
) -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
   rows: list[dict[str, object]] = []
   marked_by_step: dict[int, set[int]] = {}
   for representation, names, states in (
      ("conserved", COMPONENT_NAMES, histories),
      (
         "recovered_primitive",
         PRIMITIVE_NAMES,
         [recovered_primitive(state, gamma) for state in histories],
      ),
   ):
      for history_index in range(2, len(states)):
         ratio, active, marked = temporal_growth(
            (states[history_index - 2], states[history_index - 1]),
            states[history_index],
         )
         marked_by_step.setdefault(history_index, set()).update(
            (np.flatnonzero(np.any(marked, axis=0)) + 1).tolist()
         )
         for component, name in enumerate(names):
            component_marked = np.flatnonzero(marked[component]) + 1
            rows.append({
               "case": case,
               "source": source,
               "representation": representation,
               "step": history_index,
               "time": times[history_index],
               "component": name,
               "maximum_active_growth_ratio": float(
                  np.max(ratio[component]) if np.any(active[component]) else 0.0
               ),
               "marked_cells": int(component_marked.size),
               "cell_indices": ",".join(str(cell) for cell in component_marked),
            })
   return rows, {
      step: np.asarray(sorted(cells), dtype=int)
      for step, cells in marked_by_step.items()
   }


def rhs_with_optional_pwl_faces(
   state: np.ndarray,
   *,
   gamma: float,
   dx: float,
   marked_cells: np.ndarray,
) -> np.ndarray:
   primitive = recovered_primitive(state, gamma)
   left, right = reconstruct_faces(primitive, "ppm4", gamma)
   if marked_cells.size:
      pwl_left, pwl_right = reconstruct_faces(primitive, "pwl", gamma)
      face_mask = np.zeros(state.shape[1] + 1, dtype=bool)
      face_mask[marked_cells] = True
      face_mask[marked_cells + 1] = True
      left[:, face_mask] = pwl_left[:, face_mask]
      right[:, face_mask] = pwl_right[:, face_mask]
   rhs, fallback_count = rhs_from_faces(left, right, gamma, dx, "hll")
   if fallback_count:
      raise RuntimeError("HLL unexpectedly reported a fallback")
   return rhs


def ssprk3_step(
   state: np.ndarray,
   *,
   gamma: float,
   dx: float,
   dt: float,
   marked_cells: np.ndarray,
) -> np.ndarray:
   first_rhs = rhs_with_optional_pwl_faces(
      state, gamma=gamma, dx=dx, marked_cells=marked_cells
   )
   first = state + dt * first_rhs
   second_rhs = rhs_with_optional_pwl_faces(
      first, gamma=gamma, dx=dx, marked_cells=marked_cells
   )
   second = 0.75 * state + 0.25 * (first + dt * second_rhs)
   third_rhs = rhs_with_optional_pwl_faces(
      second, gamma=gamma, dx=dx, marked_cells=marked_cells
   )
   return (1.0 / 3.0) * state + (2.0 / 3.0) * (
      second + dt * third_rhs
   )


def prototype_metrics(
   label: str,
   candidate: np.ndarray,
   exact: np.ndarray,
   saved: np.ndarray,
) -> dict[str, object]:
   scale = np.maximum(np.max(np.abs(exact), axis=1), np.ptp(exact, axis=1))
   scale = np.maximum(scale, 1.0e-14)
   error = np.abs(candidate - exact) / scale[:, np.newaxis]
   replay_error = np.abs(candidate - saved) / scale[:, np.newaxis]
   candidate_tv = np.sum(np.abs(np.diff(candidate, axis=1)), axis=1)
   exact_tv = np.sum(np.abs(np.diff(exact, axis=1)), axis=1)
   active = exact_tv > 1.0e-14 * scale
   tv_ratio = np.ones_like(exact_tv)
   tv_ratio[active] = candidate_tv[active] / exact_tv[active]
   worst_l1 = int(np.argmax(np.mean(error, axis=1)))
   worst_linf = int(np.argmax(np.max(error, axis=1)))
   worst_tv = int(np.argmax(tv_ratio))
   return {
      "method": label,
      "maximum_normalized_l1": float(np.max(np.mean(error, axis=1))),
      "l1_component": COMPONENT_NAMES[worst_l1],
      "maximum_normalized_linf": float(np.max(error)),
      "linf_component": COMPONENT_NAMES[worst_linf],
      "maximum_tv_ratio": float(tv_ratio[worst_tv]),
      "tv_component": COMPONENT_NAMES[worst_tv],
      "maximum_replay_difference": float(np.max(replay_error)),
   }


def saved_b4_history() -> tuple[list[np.ndarray], list[float], list[Path]]:
   paths = [
      B4_RUN / "debugStage" / f"ssprk3_acceptedGF{step:05d}.h5"
      for step in range(1, 5)
   ]
   missing = [path for path in paths if not path.is_file()]
   if missing:
      raise FileNotFoundError(f"missing saved B4 accepted state {missing[0]}")
   times = [float(parse_plot_header_1d(path)[1]) for path in paths]
   return [read_saved_state(path) for path in paths], times, paths


def make_plot(
   exact: np.ndarray,
   saved: np.ndarray,
   candidate: np.ndarray,
   gamma: float,
   detector: list[dict[str, object]],
) -> None:
   exact_primitive = recovered_primitive(exact, gamma)
   saved_primitive = recovered_primitive(saved, gamma)
   candidate_primitive = recovered_primitive(candidate, gamma)
   cells = exact.shape[1]
   x = -1.0 + (np.arange(cells) + 0.5) * 2.0 / cells
   figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
   for axis, component, name in (
      (axes[0], 0, r"$\rho$"),
      (axes[1], 2, r"$v_2$"),
   ):
      axis.plot(x, exact_primitive[component], color="black", label="exact FV")
      axis.plot(x, saved_primitive[component], color="#4c78a8", label="PPM4")
      axis.plot(
         x,
         candidate_primitive[component],
         color="#f58518",
         linestyle="--",
         label="temporal PWL replay",
      )
      axis.set(xlim=(-0.06, 0.06), xlabel="x", ylabel=name)
      axis.grid(alpha=0.25)
   axes[0].legend(frameon=False, fontsize=8)
   exact_maxima = []
   numerical_maxima = []
   for case in range(1, 6):
      exact_rows = [
         row for row in detector
         if row["source"] == "exact_fv" and int(row["case"]) == case
         and int(row["step"]) == 3
      ]
      exact_maxima.append(max(float(row["maximum_active_growth_ratio"])
                              for row in exact_rows))
      numerical_rows = [
         row for row in detector
         if row["source"] == "amrex" and int(row["case"]) == case
         and int(row["step"]) == 3
      ]
      numerical_maxima.append(
         max((float(row["maximum_active_growth_ratio"])
              for row in numerical_rows), default=np.nan)
      )
   axes[2].bar(np.arange(1, 6) - 0.16, exact_maxima, width=0.32,
               color="black", alpha=0.7, label="exact FV")
   axes[2].bar(np.arange(1, 6) + 0.16, numerical_maxima, width=0.32,
               color="#4c78a8", label="AMReX")
   axes[2].axhline(GROWTH_THRESHOLD, color="#e45756", linestyle="--",
                   label="prototype threshold")
   axes[2].set(
      xlabel="Balsara case",
      ylabel="maximum temporal growth ratio",
      xticks=np.arange(1, 6),
      title="Step-three detector control",
   )
   axes[2].grid(axis="y", alpha=0.25)
   axes[2].legend(frameon=False, fontsize=8)
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PLOT, dpi=180)
   plt.close(figure)


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--skip-plot", action="store_true")
   return parser.parse_args()


def main() -> int:
   args = parse_args()
   numerical_history, numerical_times, numerical_paths = saved_b4_history()
   dt = numerical_times[-1] - numerical_times[-2]
   exact_states, edges, spec, profile = exact_history(
      4, 256, [0.0, *numerical_times]
   )
   gamma = float(spec["gamma"])
   initial = production_initial_state(spec, profile, edges)
   numerical_with_initial = [initial, *numerical_history]
   all_times = [0.0, *numerical_times]

   structure_rows: list[dict[str, object]] = []
   for step, (time, numerical, exact) in enumerate(zip(
      numerical_times, numerical_history, exact_states[1:], strict=True
   ), start=1):
      structure_rows.extend(representation_rows(
         case=4,
         step=step,
         time=time,
         numerical=numerical,
         exact=exact,
         spec=spec,
         profile=profile,
         edges=edges,
      ))

   detector: list[dict[str, object]] = []
   for case in range(1, 6):
      exact_case, _, case_specification, _ = exact_history(
         case, 256, all_times
      )
      rows, _ = detector_rows(
         case=case,
         source="exact_fv",
         histories=exact_case,
         times=all_times,
         gamma=float(case_specification["gamma"]),
      )
      detector.extend(rows)
   numerical_rows, numerical_marks = detector_rows(
      case=4,
      source="amrex",
      histories=numerical_with_initial,
      times=all_times,
      gamma=gamma,
   )
   detector.extend(numerical_rows)
   # The decision must precede the step it modifies.  Use the completed
   # step-three history to replay step four; never use step-four output to
   # choose its own correction.
   marked_cells = numerical_marks.get(3, np.empty(0, dtype=int))

   start = numerical_history[-2]
   saved_final = numerical_history[-1]
   exact_final = exact_states[-1]
   baseline = ssprk3_step(
      start,
      gamma=gamma,
      dx=2.0 / 256,
      dt=dt,
      marked_cells=np.empty(0, dtype=int),
   )
   candidate = ssprk3_step(
      start,
      gamma=gamma,
      dx=2.0 / 256,
      dt=dt,
      marked_cells=marked_cells,
   )
   prototype_rows = []
   for label, state in (
      ("saved_ppm4", saved_final),
      ("offline_ppm4_replay", baseline),
      ("temporal_marked_pwl", candidate),
   ):
      prototype_rows.append({
         "case": 4,
         "start_step": 3,
         "end_step": 4,
         "start_time": numerical_times[-2],
         "end_time": numerical_times[-1],
         "marked_cells": marked_cells.size,
         "cell_indices": ",".join(str(cell) for cell in marked_cells),
         "source_path": repo_relative(numerical_paths[-1]),
         **prototype_metrics(label, state, exact_final, saved_final),
      })

   STRUCTURE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
   write_tsv(STRUCTURE_RESULTS, structure_rows, list(structure_rows[0]))
   write_tsv(DETECTOR_RESULTS, detector, list(detector[0]))
   write_tsv(PROTOTYPE_RESULTS, prototype_rows, list(prototype_rows[0]))
   if not args.skip_plot:
      make_plot(exact_final, saved_final, candidate, gamma, detector)

   print(f"wrote {repo_relative(STRUCTURE_RESULTS)}")
   print(f"wrote {repo_relative(DETECTOR_RESULTS)}")
   print(f"wrote {repo_relative(PROTOTYPE_RESULTS)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(PLOT)}")
   print(f"temporal detector marked B4 cells {marked_cells.tolist()}")
   for row in prototype_rows:
      print(
         f"{row['method']}: L1={row['maximum_normalized_l1']:.6e} "
         f"Linf={row['maximum_normalized_linf']:.6e} "
         f"TV={row['maximum_tv_ratio']:.6e} "
         f"replay={row['maximum_replay_difference']:.3e}"
      )
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
