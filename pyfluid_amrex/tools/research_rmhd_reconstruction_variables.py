#!/usr/bin/env python3
"""Scan characteristic-free RMHD reconstruction variable choices offline.

The production scheme reconstructs density, pressure, magnetic components,
and spatial four-velocity components.  This bounded experiment changes only
the algebraic variables supplied to the same PPM4/MP5 formulas and the same
production HLL/HLLD flux oracle.  Exact Balsara finite-volume states score the
one-step operator.  A smaller set of velocity-coordinate candidates is also
available through ``--evolution-only`` for four complete offline SSPRK3
timesteps; exact data are used only for post-step scoring.  That chained
oracle is recovery-basin sensitive on the strongest cases and is a rough
screen, not a substitute for a production evolution.
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diagnose_rmhd_balsara_onset import (  # noqa: E402
   case_spec,
   exact_average_at_time,
)
from diagnose_rmhd_operator import (  # noqa: E402
   PRIMITIVE_NAMES,
   _limited_face,
   exact_primitives_at,
   exact_rhs,
   normalized_error_metrics,
   primitive_state_and_flux,
   recover_conserved,
   rhs_from_faces,
   scalar_left,
)
from scan_rmhd_balsara_reconstruction import COMPONENT_NAMES  # noqa: E402
from validate_rmhd import (  # noqa: E402
   EXACT_BALSARA_ROOT,
   RUN_ROOT,
   read_last_exact_balsara_profile,
)
from validation_common import ROOT, repo_relative, write_tsv  # noqa: E402


RESULTS = RUN_ROOT / "balsara_reconstruction_variable_scan.tsv"
EVOLUTION_RESULTS = RUN_ROOT / "balsara_reconstruction_variable_evolution.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_reconstruction_variable_scan.png"
VARIANTS = (
   "linear_pressure",
   "log_density",
   "log_pressure",
   "pressure_power_0p25",
   "pressure_power_0p5",
   "pressure_power_0p75",
   "shifted_log_pressure_1",
   "shifted_log_pressure_3",
   "shifted_log_pressure_10",
   "shifted_log_pressure_30",
   "shifted_log_pressure_100",
   "shifted_log_pressure_3_log_density",
   "shifted_log_pressure_10_log_density",
   "shifted_log_pressure_30_log_density",
   "shifted_log_pressure_10_shifted_log_density_10",
   "shifted_log_pressure_10_shifted_log_density_30",
   "shifted_log_pressure_30_shifted_log_density_10",
   "shifted_log_pressure_30_shifted_log_density_30",
   "adaptive_shifted_log_pressure_10_ratio_10_power_4",
   "adaptive_shifted_log_pressure_10_ratio_30_power_4",
   "adaptive_shifted_log_pressure_10_ratio_100_power_4",
   "adaptive_shifted_log_pressure_10_ratio_30_power_6",
   "specific_pressure",
   "log_specific_pressure",
   "entropy_proxy",
   "log_entropy_proxy",
   "log_thermo",
   "normal_rapidity_log_pressure",
   "three_velocity",
   "normal_rapidity",
   "rapidity_vector",
)
EVOLUTION_VARIANTS = (
   "linear_pressure",
   "three_velocity",
   "normal_rapidity",
   "rapidity_vector",
)


def pressure_power(variant: str) -> float | None:
   if not variant.startswith("pressure_power_"):
      return None
   return float(variant.removeprefix("pressure_power_").replace("p", "."))


def shifted_log_scale(variant: str) -> float | None:
   if not variant.startswith("shifted_log_pressure_"):
      return None
   suffix = variant.removeprefix("shifted_log_pressure_")
   return float(suffix.split("_", 1)[0])


def shifted_log_density_scale(variant: str) -> float | None:
   marker = "_shifted_log_density_"
   if marker not in variant:
      return None
   return float(variant.rsplit(marker, 1)[1])


def adaptive_shifted_log_parameters(
   variant: str,
) -> tuple[float, float, float] | None:
   prefix = "adaptive_shifted_log_pressure_"
   if not variant.startswith(prefix):
      return None
   tokens = variant.removeprefix(prefix).split("_")
   if len(tokens) != 5 or tokens[1] != "ratio" or tokens[3] != "power":
      raise ValueError(f"malformed adaptive transform {variant}")
   return float(tokens[0]), float(tokens[2]), float(tokens[4])


def encode_variables(
   primitive: np.ndarray,
   variant: str,
   gamma: float,
) -> np.ndarray:
   values = primitive.copy()
   velocity = primitive[1:4]
   velocity_squared = np.sum(velocity * velocity, axis=0)
   lorentz = 1.0 / np.sqrt(1.0 - velocity_squared)
   if variant in {"normal_rapidity", "normal_rapidity_log_pressure"}:
      values[1] = np.arctanh(
         np.clip(velocity[0], -1.0 + 1.0e-14, 1.0 - 1.0e-14)
      )
      values[2:4] = lorentz * velocity[1:3]
   elif variant == "three_velocity":
      values[1:4] = velocity
   elif variant == "rapidity_vector":
      speed = np.sqrt(velocity_squared)
      rapidity = np.arctanh(np.clip(speed, 0.0, 1.0 - 1.0e-14))
      factor = np.divide(
         rapidity,
         speed,
         out=np.ones_like(speed),
         where=speed > 1.0e-14,
      )
      values[1:4] = velocity * factor
   else:
      values[1:4] = lorentz * velocity
   if variant in {"log_density", "log_thermo"} or variant.endswith(
      "_log_density"
   ):
      values[0] = np.log(primitive[0])
   if variant in {
      "log_pressure", "log_thermo", "normal_rapidity_log_pressure"
   }:
      values[4] = np.log(primitive[4])
   exponent = pressure_power(variant)
   if exponent is not None:
      values[4] = primitive[4] ** exponent
   if variant == "specific_pressure":
      values[4] = primitive[4] / primitive[0]
   elif variant == "log_specific_pressure":
      values[4] = np.log(primitive[4] / primitive[0])
   elif variant == "entropy_proxy":
      values[4] = primitive[4] / primitive[0] ** gamma
   elif variant == "log_entropy_proxy":
      values[4] = np.log(primitive[4] / primitive[0] ** gamma)
   return values


def decode_variables(
   values: np.ndarray,
   variant: str,
   gamma: float,
) -> np.ndarray:
   primitive = values.copy()
   if variant in {"log_density", "log_thermo"} or variant.endswith(
      "_log_density"
   ):
      primitive[0] = np.exp(values[0])
   if variant in {
      "log_pressure", "log_thermo", "normal_rapidity_log_pressure"
   }:
      primitive[4] = np.exp(values[4])
   exponent = pressure_power(variant)
   if exponent is not None:
      primitive[4] = np.maximum(values[4], 0.0) ** (1.0 / exponent)
   if variant == "specific_pressure":
      primitive[4] = values[4] * primitive[0]
   elif variant == "log_specific_pressure":
      primitive[4] = np.exp(values[4]) * primitive[0]
   elif variant == "entropy_proxy":
      primitive[4] = values[4] * primitive[0] ** gamma
   elif variant == "log_entropy_proxy":
      primitive[4] = np.exp(values[4]) * primitive[0] ** gamma
   if variant in {"normal_rapidity", "normal_rapidity_log_pressure"}:
      normal_velocity = np.tanh(values[1])
      transverse_four_velocity_squared = values[2] ** 2 + values[3] ** 2
      lorentz = np.sqrt(
         (1.0 + transverse_four_velocity_squared)
         / (1.0 - normal_velocity**2)
      )
      primitive[1] = normal_velocity
      primitive[2] = values[2] / lorentz
      primitive[3] = values[3] / lorentz
   elif variant == "three_velocity":
      primitive[1:4] = values[1:4]
   elif variant == "rapidity_vector":
      rapidity = np.sqrt(np.sum(values[1:4] ** 2, axis=0))
      speed = np.tanh(rapidity)
      factor = np.divide(
         speed,
         rapidity,
         out=np.ones_like(rapidity),
         where=rapidity > 1.0e-14,
      )
      primitive[1:4] = values[1:4] * factor
   else:
      lorentz = np.sqrt(1.0 + np.sum(values[1:4] ** 2, axis=0))
      primitive[1:4] = values[1:4] / lorentz
   return primitive


def reconstruct_variant(
   primitive: np.ndarray,
   method: str,
   gamma: float,
   variant: str,
) -> tuple[np.ndarray, np.ndarray]:
   variables = encode_variables(primitive, variant, gamma)
   num_cells = primitive.shape[1]
   padded = np.pad(variables, ((0, 0), (3, 3)), mode="edge")
   left_variables = np.empty((8, num_cells + 1))
   right_variables = np.empty((8, num_cells + 1))
   for face_index in range(num_cells + 1):
      left_center = face_index + 2
      for component in range(8):
         shift_factor = (
            shifted_log_scale(variant)
            if component == 4
            else shifted_log_density_scale(variant)
            if component == 0
            else None
         )
         adaptive_parameters = (
            adaptive_shifted_log_parameters(variant)
            if component == 4
            else None
         )
         if shift_factor is not None or adaptive_parameters is not None:
            left_stencil = primitive[
               component,
               np.clip(
                  np.arange(face_index - 3, face_index + 2),
                  0,
                  num_cells - 1,
               ),
            ]
            right_stencil = primitive[
               component,
               np.clip(
                  np.arange(face_index + 2, face_index - 3, -1),
                  0,
                  num_cells - 1,
               ),
            ]
            active_shift_factor = (
               shift_factor
               if shift_factor is not None
               else adaptive_parameters[0]
            )
            left_shift = active_shift_factor * float(np.min(left_stencil))
            right_shift = active_shift_factor * float(np.min(right_stencil))
            left_shifted = (
               np.exp(scalar_left(
                  np.log(left_stencil + left_shift), method
               )) - left_shift
            )
            right_shifted = (
               np.exp(scalar_left(
                  np.log(right_stencil + right_shift), method
               )) - right_shift
            )
            if adaptive_parameters is None:
               left_variables[component, face_index] = left_shifted
               right_variables[component, face_index] = right_shifted
            else:
               _, activation_ratio, activation_power = adaptive_parameters
               left_ratio = float(np.max(left_stencil) / np.min(left_stencil))
               right_ratio = float(np.max(right_stencil) / np.min(right_stencil))
               left_strength = (
                  np.log(left_ratio) / np.log(activation_ratio)
               ) ** activation_power
               right_strength = (
                  np.log(right_ratio) / np.log(activation_ratio)
               ) ** activation_power
               left_weight = left_strength / (1.0 + left_strength)
               right_weight = right_strength / (1.0 + right_strength)
               left_linear = scalar_left(left_stencil, method)
               right_linear = scalar_left(right_stencil, method)
               left_variables[component, face_index] = (
                  (1.0 - left_weight) * left_linear
                  + left_weight * left_shifted
               )
               right_variables[component, face_index] = (
                  (1.0 - right_weight) * right_linear
                  + right_weight * right_shifted
               )
         else:
            left_variables[component, face_index] = scalar_left(
               padded[component, left_center - 2 : left_center + 3], method
            )
            right_variables[component, face_index] = scalar_left(
               padded[component, left_center + 3 : left_center - 2 : -1],
               method,
            )
   left = decode_variables(left_variables, variant, gamma)
   right = decode_variables(right_variables, variant, gamma)
   for face_index in range(num_cells + 1):
      center_l = primitive[:, min(max(face_index - 1, 0), num_cells - 1)]
      center_r = primitive[:, min(max(face_index, 0), num_cells - 1)]
      left[:, face_index] = _limited_face(
         left[:, face_index], center_l, gamma
      )
      right[:, face_index] = _limited_face(
         right[:, face_index], center_r, gamma
      )
   return left, right


def scan_state(
   *,
   case: int,
   cells: int,
   time: float,
   dt: float,
   age: str,
   method: str,
) -> list[dict[str, object]]:
   spec = case_spec(case)
   gamma = float(spec["gamma"])
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   edges = np.linspace(-1.0, 1.0, cells + 1)
   conserved = exact_average_at_time(profile, spec, time, edges)
   recovery = recover_conserved(conserved, gamma)
   if not np.all(recovery.success):
      raise RuntimeError(
         f"B{case} N={cells} exact averages failed production C2P"
      )
   reference_rhs = exact_rhs(profile, spec, time, edges, dt)
   exact_left = exact_primitives_at(
      profile, spec, time, edges, side="left"
   )
   exact_right = exact_primitives_at(
      profile, spec, time, edges, side="right"
   )
   trace_scale = np.maximum(
      np.max(np.abs(np.hstack((exact_left, exact_right))), axis=1),
      np.ptp(np.hstack((exact_left, exact_right)), axis=1),
   )
   trace_scale = np.maximum(trace_scale, 1.0e-14)
   rows: list[dict[str, object]] = []
   for variant in VARIANTS:
      left, right = reconstruct_variant(
         recovery.primitive, method, gamma, variant
      )
      left_error = np.abs(left - exact_left) / trace_scale[:, np.newaxis]
      right_error = np.abs(right - exact_right) / trace_scale[:, np.newaxis]
      face_linf = np.maximum(
         np.max(left_error, axis=1), np.max(right_error, axis=1)
      )
      for riemann in ("hll", "hlld"):
         rhs, fallback_count = rhs_from_faces(
            left, right, gamma, 2.0 / cells, riemann
         )
         metrics = normalized_error_metrics(
            rhs, reference_rhs, conserved, dt
         )
         rows.append({
            "case": case,
            "cells": cells,
            "age": age,
            "time": time,
            "dt": dt,
            "reconstruction": method,
            "riemann": riemann,
            "variables": variant,
            "riemann_fallbacks": fallback_count,
            **metrics,
            "density_face_linf": face_linf[0],
            "normal_velocity_face_linf": face_linf[1],
            "tangential_velocity_face_linf": float(np.max(face_linf[2:4])),
            "pressure_face_linf": face_linf[4],
            "magnetic_face_linf": float(np.max(face_linf[5:8])),
            "worst_face_component": PRIMITIVE_NAMES[int(np.argmax(face_linf))],
            "worst_face_linf": float(np.max(face_linf)),
         })
   return rows


def improvement_summary(rows: list[dict[str, object]]) -> None:
   keys = ("case", "cells", "age", "reconstruction", "riemann")
   grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
   for row in rows:
      key = tuple(row[name] for name in keys)
      grouped.setdefault(key, {})[str(row["variables"])] = row
   for candidate in VARIANTS[1:]:
      ratios = np.asarray([
         float(group[candidate]["normalized_step_linf"])
         / float(group["linear_pressure"]["normalized_step_linf"])
         for group in grouped.values()
      ])
      print(
         f"{candidate}: wins={np.count_nonzero(ratios < 1.0)}/{ratios.size} "
         f"geomean={np.exp(np.mean(np.log(ratios))):.4f} "
         f"best={np.min(ratios):.4f} worst={np.max(ratios):.4f}"
      )


def variant_rhs(
   state: np.ndarray,
   *,
   gamma: float,
   dx: float,
   method: str,
   riemann: str,
   variant: str,
) -> tuple[np.ndarray | None, int]:
   recovery = recover_conserved(state, gamma)
   if not np.all(recovery.success):
      return None, 0
   left, right = reconstruct_variant(
      recovery.primitive, method, gamma, variant
   )
   return rhs_from_faces(left, right, gamma, dx, riemann)


def evolve_variant(
   *,
   case: int,
   cells: int,
   steps: int,
   method: str,
   riemann: str,
   variant: str,
) -> dict[str, object]:
   spec = case_spec(case)
   gamma = float(spec["gamma"])
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   edges = np.linspace(-1.0, 1.0, cells + 1)
   centers = 0.5 * (edges[:-1] + edges[1:])
   dx = 2.0 / cells
   dt = 0.9 * dx
   initial_primitive = exact_primitives_at(profile, spec, 0.0, centers)
   state, _ = primitive_state_and_flux(initial_primitive, gamma)
   fallback_count = 0
   failed_stage = ""
   for step in range(1, steps + 1):
      first_rhs, fallbacks = variant_rhs(
         state,
         gamma=gamma,
         dx=dx,
         method=method,
         riemann=riemann,
         variant=variant,
      )
      fallback_count += fallbacks
      if first_rhs is None:
         failed_stage = f"{step}:stage1"
         break
      first = state + dt * first_rhs
      second_rhs, fallbacks = variant_rhs(
         first,
         gamma=gamma,
         dx=dx,
         method=method,
         riemann=riemann,
         variant=variant,
      )
      fallback_count += fallbacks
      if second_rhs is None:
         failed_stage = f"{step}:stage2"
         break
      second = 0.75 * state + 0.25 * (first + dt * second_rhs)
      third_rhs, fallbacks = variant_rhs(
         second,
         gamma=gamma,
         dx=dx,
         method=method,
         riemann=riemann,
         variant=variant,
      )
      fallback_count += fallbacks
      if third_rhs is None:
         failed_stage = f"{step}:stage3"
         break
      state = (1.0 / 3.0) * state + (2.0 / 3.0) * (
         second + dt * third_rhs
      )
   common = {
      "case": case,
      "cells": cells,
      "steps": steps,
      "time": steps * dt,
      "dt": dt,
      "reconstruction": method,
      "riemann": riemann,
      "variables": variant,
      "status": "C2P_FAILURE" if failed_stage else "OK",
      "failed_stage": failed_stage,
      "riemann_fallbacks": fallback_count,
   }
   if failed_stage:
      return {
         **common,
         "maximum_normalized_l1": "",
         "l1_component": "",
         "maximum_normalized_linf": "",
         "linf_component": "",
         "maximum_tv_ratio": "",
         "tv_component": "",
      }
   exact = exact_average_at_time(profile, spec, steps * dt, edges)
   scale = np.maximum(np.max(np.abs(exact), axis=1), np.ptp(exact, axis=1))
   scale = np.maximum(scale, 1.0e-14)
   error = np.abs(state - exact) / scale[:, np.newaxis]
   component_l1 = np.mean(error, axis=1)
   component_linf = np.max(error, axis=1)
   numerical_tv = np.sum(np.abs(np.diff(state, axis=1)), axis=1)
   exact_tv = np.sum(np.abs(np.diff(exact, axis=1)), axis=1)
   active = exact_tv > 1.0e-14 * scale
   tv_ratio = np.ones_like(exact_tv)
   tv_ratio[active] = numerical_tv[active] / exact_tv[active]
   worst_l1 = int(np.argmax(component_l1))
   worst_linf = int(np.argmax(component_linf))
   worst_tv = int(np.argmax(tv_ratio))
   return {
      **common,
      "maximum_normalized_l1": float(component_l1[worst_l1]),
      "l1_component": COMPONENT_NAMES[worst_l1],
      "maximum_normalized_linf": float(component_linf[worst_linf]),
      "linf_component": COMPONENT_NAMES[worst_linf],
      "maximum_tv_ratio": float(tv_ratio[worst_tv]),
      "tv_component": COMPONENT_NAMES[worst_tv],
   }


def evolution_rows() -> list[dict[str, object]]:
   return [
      evolve_variant(
         case=case,
         cells=256,
         steps=4,
         method=method,
         riemann=riemann,
         variant=variant,
      )
      for case in range(1, 6)
      for method in ("ppm4", "mp5")
      for riemann in ("hll", "hlld")
      for variant in EVOLUTION_VARIANTS
   ]


def evolution_summary(rows: list[dict[str, object]]) -> None:
   keys = ("case", "cells", "steps", "reconstruction", "riemann")
   grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
   for row in rows:
      key = tuple(row[name] for name in keys)
      grouped.setdefault(key, {})[str(row["variables"])] = row
   for candidate in EVOLUTION_VARIANTS[1:]:
      ratios = []
      failures = 0
      baseline_failures = 0
      for group in grouped.values():
         if group["linear_pressure"]["status"] != "OK":
            baseline_failures += 1
            continue
         row = group[candidate]
         if row["status"] != "OK":
            failures += 1
            continue
         ratios.append(
            float(row["maximum_normalized_l1"])
            / float(group["linear_pressure"]["maximum_normalized_l1"])
         )
      values = np.asarray(ratios)
      print(
         f"four-step {candidate}: wins={np.count_nonzero(values < 1.0)}/"
         f"{values.size} failures={failures} "
         f"baseline_failures={baseline_failures} "
         f"geomean={np.exp(np.mean(np.log(values))):.4f} "
         f"best={np.min(values):.4f} worst={np.max(values):.4f}"
      )


def make_plot(rows: list[dict[str, object]]) -> None:
   candidate_name = "adaptive_shifted_log_pressure_10_ratio_30_power_6"
   keys = ("case", "cells", "age", "reconstruction", "riemann")
   grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
   for row in rows:
      key = tuple(row[name] for name in keys)
      grouped.setdefault(key, {})[str(row["variables"])] = row
   figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
   colors = {"hll": "#4c78a8", "hlld": "#f58518"}
   markers = {"early": "o", "final": "s"}
   for key, group in grouped.items():
      baseline = float(group["linear_pressure"]["normalized_step_linf"])
      candidate = float(group[candidate_name]["normalized_step_linf"])
      riemann = str(key[4])
      age = str(key[2])
      axes[0].scatter(
         baseline,
         candidate,
         color=colors[riemann],
         marker=markers[age],
         alpha=0.75,
      )
      axes[1].scatter(
         int(key[0]),
         candidate / baseline,
         color=colors[riemann],
         marker=markers[age],
         alpha=0.75,
      )
   all_errors = [
      float(row["normalized_step_linf"])
      for row in rows
      if row["variables"] in {"linear_pressure", candidate_name}
   ]
   lower = max(1.0e-4, min(all_errors) * 0.7)
   upper = max(all_errors) * 1.3
   axes[0].plot((lower, upper), (lower, upper), "k--", linewidth=1)
   axes[0].set(
      xscale="log",
      yscale="log",
      xlim=(lower, upper),
      ylim=(lower, upper),
      xlabel="linear-pressure worst one-step defect",
      ylabel="adaptive-pressure worst one-step defect",
      title="Every exact-state operator sample",
   )
   axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
   axes[1].set(
      xlabel="Balsara case",
      ylabel="adaptive-pressure / linear-pressure defect",
      title="Below one is an improvement",
      xticks=np.arange(1, 6),
   )
   for axis in axes:
      axis.grid(alpha=0.25)
   # Compact manual legend; every combination is already visible in the data.
   for label, color in colors.items():
      axes[0].scatter([], [], color=color, marker="o", label=label.upper())
   for label, marker in markers.items():
      axes[0].scatter([], [], color="gray", marker=marker, label=label)
   axes[0].legend(frameon=False, fontsize=8)
   figure.suptitle(
      "Offline exact-state screen; complete AMReX evolution rejected this candidate",
      fontsize=11,
   )
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PLOT, dpi=180)
   plt.close(figure)


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--skip-plot", action="store_true")
   parser.add_argument("--evolution-only", action="store_true")
   return parser.parse_args()


def main() -> int:
   args = parse_args()
   if args.evolution_only:
      complete_step_rows = evolution_rows()
      EVOLUTION_RESULTS.parent.mkdir(parents=True, exist_ok=True)
      write_tsv(
         EVOLUTION_RESULTS,
         complete_step_rows,
         list(complete_step_rows[0]),
      )
      evolution_summary(complete_step_rows)
      print(f"wrote {repo_relative(EVOLUTION_RESULTS)}")
      return 0
   rows: list[dict[str, object]] = []
   for case in range(1, 6):
      spec = case_spec(case)
      for cells, age, time in (
         (256, "early", 4.0 * 0.9 * (2.0 / 256)),
         (512, "final", float(spec["time"])),
      ):
         dt = 0.9 * (2.0 / cells)
         for method in ("ppm4", "mp5"):
            rows.extend(scan_state(
               case=case,
               cells=cells,
               time=time,
               dt=dt,
               age=age,
               method=method,
            ))
   RESULTS.parent.mkdir(parents=True, exist_ok=True)
   write_tsv(RESULTS, rows, list(rows[0]))
   improvement_summary(rows)
   if not args.skip_plot:
      make_plot(rows)
   print(f"wrote {repo_relative(RESULTS)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
