#!/usr/bin/env python3
"""Decompose difficult Balsara errors through the production SRMHD operator.

This tool is deliberately offline.  It uses cell-averaged exact conserved
states, the production C++ conservative-to-primitive recovery, the scalar
reconstruction formulas used by the AMReX path, and the production C++
HLL/HLLC/HLLD fluxes.  It never supplies exact data to an evolution run.

The main comparison distinguishes three representations of the same exact
finite-volume state:

* ``c2p_average`` is the production representation: recover one primitive
  state from each exact conserved cell average;
* ``exact_center`` is a diagnostic upper bound using the exact primitive at
  the cell center before applying the same reconstruction and Riemann flux;
* ``pointwise_roundtrip`` is a rough, solution-independent fourth-order
  average-to-point-to-average conversion prototype.

The first Balsara-4 step is also replayed against saved AMReX stage output so
that a mismatch in this diagnostic cannot silently be interpreted as physics.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diagnose_rmhd_balsara_onset import (  # noqa: E402
   case_spec,
   exact_average_at_time,
)
from plotfile_reader import (  # noqa: E402
   parse_plot_bounds,
   parse_plot_header_1d,
   read_composite_state_1d,
)
from scan_rmhd_balsara_reconstruction import COMPONENT_NAMES  # noqa: E402
from validate_rmhd import (  # noqa: E402
   EXACT_BALSARA_ROOT,
   RUN_ROOT,
   read_last_exact_balsara_profile,
)
from validation_common import BUILD, ROOT, repo_relative, write_tsv  # noqa: E402


ORACLE = BUILD / "pyfluid_amrex_rmhd_riemann_check"
RESULTS = RUN_ROOT / "balsara_operator_autopsy.tsv"
INTERVENTIONS = RUN_ROOT / "balsara_operator_interventions.tsv"
FACE_ERRORS = RUN_ROOT / "balsara_face_trace_errors.tsv"
SUBCELL_RESULTS = RUN_ROOT / "balsara_subcell_prototype.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_balsara_operator_autopsy.png"
PRIMITIVE_NAMES = ("rho", "v1", "v2", "v3", "pressure", "b1", "b2", "b3")
REPRESENTATIONS = ("c2p_average", "exact_center", "pointwise_roundtrip")


@dataclasses.dataclass(frozen=True)
class RecoveryBatch:
   primitive: np.ndarray
   success: np.ndarray
   conserved_residual: np.ndarray


@dataclasses.dataclass(frozen=True)
class FluxBatch:
   hll: np.ndarray
   hllc: np.ndarray
   hlld: np.ndarray
   hllc_fallback: np.ndarray
   hlld_fallback: np.ndarray


@dataclasses.dataclass(frozen=True)
class OperatorResult:
   rhs: np.ndarray
   primitive: np.ndarray
   left_face: np.ndarray
   right_face: np.ndarray
   riemann_fallbacks: int
   point_c2p_fallbacks: int


def _run_oracle(mode: str, rows: list[str]) -> np.ndarray:
   if not ORACLE.is_file():
      raise FileNotFoundError(
         f"missing {ORACLE}; build target pyfluid_amrex_rmhd_riemann_check"
      )
   completed = subprocess.run(
      [str(ORACLE), mode],
      input="".join(rows),
      text=True,
      capture_output=True,
      check=False,
   )
   if completed.returncode != 0:
      raise RuntimeError(
         f"RMHD operator oracle failed ({mode}):\n{completed.stderr}"
      )
   if not completed.stdout.strip():
      return np.empty((0, 0))
   table = np.loadtxt(completed.stdout.splitlines(), ndmin=2)
   expected_ids = np.arange(table.shape[0], dtype=int)
   if not np.array_equal(table[:, 0].astype(int), expected_ids):
      raise RuntimeError(f"RMHD operator oracle reordered rows in {mode}")
   return table


def recover_conserved(conserved: np.ndarray, gamma: float) -> RecoveryBatch:
   rows = [
      f"{index} {gamma:.17g} "
      + " ".join(f"{value:.17g}" for value in conserved[:, index])
      + "\n"
      for index in range(conserved.shape[1])
   ]
   table = _run_oracle("--c2p-table", rows)
   # Oracle primitive order is rho, eps, p, v1, v2, v3, b1, b2, b3.
   primitive = table[:, (7, 10, 11, 12, 9, 13, 14, 15)].T
   return RecoveryBatch(
      primitive=primitive,
      success=table[:, 1].astype(bool),
      conserved_residual=table[:, 6],
   )


def primitive_state_and_flux(
   primitive: np.ndarray,
   gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
   rows = [
      f"{index} {gamma:.17g} "
      + " ".join(
         f"{value:.17g}"
         for value in primitive[[0, 4, 1, 2, 3, 5, 6, 7], index]
      )
      + "\n"
      for index in range(primitive.shape[1])
   ]
   table = _run_oracle("--primitive-table", rows)
   return table[:, 1:9].T, table[:, 9:17].T


def riemann_fluxes(
   left: np.ndarray,
   right: np.ndarray,
   gamma: float,
) -> FluxBatch:
   rows = []
   oracle_order = (0, 4, 1, 2, 3, 5, 6, 7)
   for index in range(left.shape[1]):
      values = np.concatenate((
         left[list(oracle_order), index],
         right[list(oracle_order), index],
      ))
      rows.append(
         f"{index} {gamma:.17g} "
         + " ".join(f"{value:.17g}" for value in values)
         + "\n"
      )
   table = _run_oracle("--flux-table", rows)
   return FluxBatch(
      hll=table[:, 1:9].T,
      hllc=table[:, 9:17].T,
      hllc_fallback=table[:, 17].astype(bool),
      hlld=table[:, 18:26].T,
      hlld_fallback=table[:, 26].astype(bool),
   )


def select_riemann_flux(
   batch: FluxBatch,
   riemann: str,
) -> tuple[np.ndarray, int]:
   if riemann == "hll":
      return batch.hll, 0
   if riemann == "hllc":
      return batch.hllc, int(np.count_nonzero(batch.hllc_fallback))
   if riemann == "hlld":
      return batch.hlld, int(np.count_nonzero(batch.hlld_fallback))
   raise ValueError(f"unknown Riemann method {riemann}")


def rhs_from_faces(
   left: np.ndarray,
   right: np.ndarray,
   gamma: float,
   dx: float,
   riemann: str,
) -> tuple[np.ndarray, int]:
   flux, fallback_count = select_riemann_flux(
      riemann_fluxes(left, right, gamma), riemann
   )
   return -(flux[:, 1:] - flux[:, :-1]) / dx, fallback_count


def minmod(first: np.ndarray, second: np.ndarray) -> np.ndarray:
   same_sign = first * second > 0.0
   magnitude = np.minimum(np.abs(first), np.abs(second))
   return np.where(same_sign, np.copysign(magnitude, first), 0.0)


def minmod4(first: float, second: float, third: float, fourth: float) -> float:
   values = np.asarray((first, second, third, fourth))
   if np.all(values > 0.0):
      return float(np.min(values))
   if np.all(values < 0.0):
      return float(np.max(values))
   return 0.0


def scalar_left(stencil: np.ndarray, method: str) -> float:
   vmm, vm, v0, vp, vpp = stencil
   if method == "ppm4":
      left = np.clip(
         (7.0 * (vm + v0) - (vmm + vp)) / 12.0,
         min(vm, v0),
         max(vm, v0),
      )
      right = np.clip(
         (7.0 * (v0 + vp) - (vm + vpp)) / 12.0,
         min(v0, vp),
         max(v0, vp),
      )
      right_delta = right - v0
      left_delta = left - v0
      if right_delta * left_delta >= 0.0:
         return float(v0)
      if abs(right_delta) >= 2.0 * abs(left_delta):
         right = v0 - 2.0 * left_delta
      return float(right)
   if method == "mp5":
      alpha = 4.0
      unlimited = (
         2.0 * vmm - 13.0 * vm + 47.0 * v0 + 27.0 * vp - 3.0 * vpp
      ) / 60.0
      monotone = v0 + float(minmod(np.asarray(vp - v0), np.asarray(alpha * (v0 - vm))))
      scale = float(np.max(np.abs(stencil)))
      if (unlimited - v0) * (unlimited - monotone) <= (
         1.0e-12 * max(1.0, scale * scale)
      ):
         return float(unlimited)
      dm = vmm - 2.0 * vm + v0
      d0 = vm - 2.0 * v0 + vp
      dp = v0 - 2.0 * vp + vpp
      d_left = minmod4(4.0 * dm - d0, 4.0 * d0 - dm, dm, d0)
      d_right = minmod4(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp)
      upper_limit = v0 + alpha * (v0 - vm)
      average = 0.5 * (v0 + vp)
      median = average - 0.5 * d_right
      large_curvature = v0 + 0.5 * (v0 - vm) + (4.0 / 3.0) * d_left
      lower_bound = max(
         min(v0, vp, median),
         min(v0, upper_limit, large_curvature),
      )
      upper_bound = min(
         max(v0, vp, median),
         max(v0, upper_limit, large_curvature),
      )
      return float(
         unlimited
         + minmod(
            np.asarray(lower_bound - unlimited),
            np.asarray(upper_bound - unlimited),
         )
      )
   raise ValueError(f"unsupported five-point reconstruction: {method}")


def reconstruction_variables(primitive: np.ndarray) -> np.ndarray:
   velocity_squared = np.sum(primitive[1:4] ** 2, axis=0)
   if np.any(velocity_squared >= 1.0):
      raise RuntimeError("reconstruction received a superluminal center state")
   lorentz = 1.0 / np.sqrt(1.0 - velocity_squared)
   return np.vstack((
      primitive[0],
      lorentz * primitive[1],
      lorentz * primitive[2],
      lorentz * primitive[3],
      primitive[4:],
   ))


def primitive_from_reconstruction_variables(values: np.ndarray) -> np.ndarray:
   lorentz = np.sqrt(1.0 + np.sum(values[1:4] ** 2, axis=0))
   return np.vstack((
      values[0],
      values[1] / lorentz,
      values[2] / lorentz,
      values[3] / lorentz,
      values[4:],
   ))


def _limited_face(
   face: np.ndarray,
   center: np.ndarray,
   gamma: float,
   density_floor: float = 1.0e-12,
   pressure_floor: float = 1.0e-12,
) -> np.ndarray:
   if not np.all(np.isfinite(face[:5])):
      limited = center.copy()
      limited[5:] = face[5:]
      return limited
   theta = 1.0
   for component, floor in ((0, density_floor), (4, pressure_floor)):
      if face[component] < floor:
         denominator = center[component] - face[component]
         candidate = (
            (center[component] - floor) / denominator
            if denominator > 0.0 and center[component] > floor
            else 0.0
         )
         theta = min(theta, max(0.0, min(1.0, candidate)))
   velocity_squared = float(np.dot(face[1:4], face[1:4]))
   velocity_limit_squared = 1.0 - 128.0 * np.finfo(float).eps
   if not np.isfinite(velocity_squared) or velocity_squared >= velocity_limit_squared:
      delta = face[1:4] - center[1:4]
      quadratic = float(np.dot(delta, delta))
      linear = 2.0 * float(np.dot(center[1:4], delta))
      constant = float(np.dot(center[1:4], center[1:4])) - velocity_limit_squared
      discriminant = linear * linear - 4.0 * quadratic * constant
      causal_theta = 0.0
      if quadratic > 0.0 and discriminant >= 0.0 and np.isfinite(discriminant):
         causal_theta = (-linear + np.sqrt(discriminant)) / (2.0 * quadratic)
      theta = min(theta, max(0.0, min(1.0, causal_theta)))
   limited = face.copy()
   limited[:5] = center[:5] + theta * (face[:5] - center[:5])
   enthalpy = 1.0 + gamma / (gamma - 1.0) * limited[4] / limited[0]
   sound_speed_squared = gamma * limited[4] / (limited[0] * enthalpy)
   if (
      limited[0] <= 0.0
      or limited[4] < 0.0
      or not 0.0 <= sound_speed_squared <= 1.0
      or float(np.dot(limited[1:4], limited[1:4])) >= 1.0
   ):
      limited[:5] = center[:5]
   return limited


def reconstruct_faces(
   primitive: np.ndarray,
   method: str,
   gamma: float,
   *,
   variables: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
   num_cells = primitive.shape[1]
   reconstructed = (
      reconstruction_variables(primitive) if variables is None else variables
   )
   padded = np.pad(reconstructed, ((0, 0), (3, 3)), mode="edge")
   left_variables = np.empty((8, num_cells + 1))
   right_variables = np.empty((8, num_cells + 1))
   for face_index in range(num_cells + 1):
      left_cell = face_index - 1
      right_cell = face_index
      for component in range(8):
         if method == "pwc":
            left_variables[component, face_index] = padded[
               component, left_cell + 3
            ]
            right_variables[component, face_index] = padded[
               component, right_cell + 3
            ]
         elif method == "pwl":
            left_index = left_cell + 3
            right_index = right_cell + 3
            qlm, ql, qr, qrp = padded[
               component, (left_index - 1, left_index, right_index, right_index + 1)
            ]
            centered_l = 0.5 * (qr - qlm)
            limited_l = float(minmod(
               np.asarray(2.0 * (ql - qlm)),
               np.asarray(2.0 * (qr - ql)),
            ))
            slope_l = float(minmod(np.asarray(centered_l), np.asarray(limited_l)))
            centered_r = 0.5 * (qrp - ql)
            limited_r = float(minmod(
               np.asarray(2.0 * (qr - ql)),
               np.asarray(2.0 * (qrp - qr)),
            ))
            slope_r = float(minmod(np.asarray(centered_r), np.asarray(limited_r)))
            left_variables[component, face_index] = ql + 0.5 * slope_l
            right_variables[component, face_index] = qr - 0.5 * slope_r
         else:
            center = left_cell + 3
            left_variables[component, face_index] = scalar_left(
               padded[component, center - 2 : center + 3], method
            )
            right_variables[component, face_index] = scalar_left(
               padded[component, center + 3 : center - 2 : -1], method
            )
   left = primitive_from_reconstruction_variables(left_variables)
   right = primitive_from_reconstruction_variables(right_variables)
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


def exact_profile_to_primitive(
   values: np.ndarray,
   normal_magnetic: float,
   gamma: float,
) -> np.ndarray:
   rho = values[:, 0]
   total_pressure = values[:, 1]
   velocity = values[:, 2:5]
   magnetic = np.column_stack((
      np.full(values.shape[0], normal_magnetic), values[:, 5], values[:, 6]
   ))
   velocity_squared = np.sum(velocity * velocity, axis=1)
   lorentz = 1.0 / np.sqrt(1.0 - velocity_squared)
   comoving_magnetic_squared = (
      np.sum(magnetic * magnetic, axis=1) / lorentz**2
      + np.sum(magnetic * velocity, axis=1) ** 2
   )
   pressure = total_pressure - 0.5 * comoving_magnetic_squared
   if np.any(pressure <= 0.0):
      raise RuntimeError("sampled exact profile has nonpositive gas pressure")
   return np.vstack((
      rho,
      velocity[:, 0],
      velocity[:, 1],
      velocity[:, 2],
      pressure,
      magnetic[:, 0],
      magnetic[:, 1],
      magnetic[:, 2],
   ))


def tabulated_samples(
   profile: np.ndarray,
   coordinates: np.ndarray,
   *,
   side: str = "center",
) -> np.ndarray:
   """Sample a repeated-coordinate exact table without crossing jumps."""

   abscissa = profile[:, 0]
   result = np.empty((coordinates.size, profile.shape[1] - 1))
   tolerance = 64.0 * np.finfo(float).eps
   for sample_index, coordinate in enumerate(coordinates):
      if coordinate <= abscissa[0]:
         result[sample_index] = profile[0, 1:]
         continue
      if coordinate >= abscissa[-1]:
         result[sample_index] = profile[-1, 1:]
         continue
      first = int(np.searchsorted(abscissa, coordinate, side="left"))
      last = int(np.searchsorted(abscissa, coordinate, side="right"))
      exact_match = (
         first < abscissa.size
         and abs(abscissa[first] - coordinate)
         <= tolerance * max(1.0, abs(coordinate))
      )
      if exact_match:
         if side == "left":
            result[sample_index] = profile[first, 1:]
         elif side == "right":
            result[sample_index] = profile[last - 1, 1:]
         else:
            result[sample_index] = 0.5 * (
               profile[first, 1:] + profile[last - 1, 1:]
            )
         continue
      upper = first
      lower = first - 1
      weight = (coordinate - abscissa[lower]) / (
         abscissa[upper] - abscissa[lower]
      )
      result[sample_index] = (
         (1.0 - weight) * profile[lower, 1:] + weight * profile[upper, 1:]
      )
   return result


def exact_primitives_at(
   profile: np.ndarray,
   spec: dict[str, object],
   time: float,
   coordinates: np.ndarray,
   *,
   side: str = "center",
) -> np.ndarray:
   if time <= 16.0 * np.finfo(float).eps:
      left = np.asarray(spec["left"], dtype=float)
      right = np.asarray(spec["right"], dtype=float)
      states = np.where(
         (coordinates < 0.0)[:, np.newaxis], left, right
      )
      return states[:, (0, 2, 3, 4, 1, 5, 6, 7)].T
   final_coordinates = coordinates * float(spec["time"]) / time
   values = tabulated_samples(profile, final_coordinates, side=side)
   return exact_profile_to_primitive(
      values,
      float(spec["left"][5]),
      float(spec["gamma"]),
   )


def pointwise_roundtrip_variables(
   conserved_average: np.ndarray,
   gamma: float,
   base_primitive: np.ndarray,
) -> tuple[np.ndarray, int]:
   """Prototype fourth-order FV average/point primitive conversion."""

   padded = np.pad(conserved_average, ((0, 0), (1, 1)), mode="edge")
   laplacian = padded[:, 2:] - 2.0 * padded[:, 1:-1] + padded[:, :-2]
   point_conserved = conserved_average - laplacian / 24.0
   point_recovery = recover_conserved(point_conserved, gamma)
   failed = ~point_recovery.success
   point_primitive = point_recovery.primitive.copy()
   point_primitive[:, failed] = base_primitive[:, failed]
   point_variables = reconstruction_variables(point_primitive)
   padded_variables = np.pad(point_variables, ((0, 0), (1, 1)), mode="edge")
   variable_laplacian = (
      padded_variables[:, 2:]
      - 2.0 * padded_variables[:, 1:-1]
      + padded_variables[:, :-2]
   )
   average_variables = point_variables + variable_laplacian / 24.0
   return average_variables, int(np.count_nonzero(failed))


def operator_rhs(
   conserved: np.ndarray,
   gamma: float,
   dx: float,
   method: str,
   riemann: str,
   representation: str,
   *,
   exact_center: np.ndarray | None = None,
) -> OperatorResult:
   recovery = recover_conserved(conserved, gamma)
   if not np.all(recovery.success):
      failed = np.flatnonzero(~recovery.success)
      raise RuntimeError(f"production C2P failed in cells {failed[:12].tolist()}")
   primitive = recovery.primitive
   variables = None
   point_fallbacks = 0
   if representation == "exact_center":
      if exact_center is None:
         raise ValueError("exact_center representation needs an exact state")
      primitive = exact_center
   elif representation == "pointwise_roundtrip":
      variables, point_fallbacks = pointwise_roundtrip_variables(
         conserved, gamma, primitive
      )
   elif representation != "c2p_average":
      raise ValueError(f"unknown representation {representation}")
   left, right = reconstruct_faces(
      primitive, method, gamma, variables=variables
   )
   rhs, fallback_count = rhs_from_faces(left, right, gamma, dx, riemann)
   return OperatorResult(
      rhs=rhs,
      primitive=primitive,
      left_face=left,
      right_face=right,
      riemann_fallbacks=fallback_count,
      point_c2p_fallbacks=point_fallbacks,
   )


def exact_rhs(
   exact_profile: np.ndarray,
   spec: dict[str, object],
   time: float,
   cell_edges: np.ndarray,
   dt: float,
) -> np.ndarray:
   delta = min(1.0e-4 * dt, 0.2 * time)
   delta = max(delta, 256.0 * np.finfo(float).eps)
   before = exact_average_at_time(
      exact_profile, spec, time - delta, cell_edges
   )
   after = exact_average_at_time(
      exact_profile, spec, time + delta, cell_edges
   )
   return (after - before) / (2.0 * delta)


def normalized_error_metrics(
   candidate: np.ndarray,
   reference: np.ndarray,
   state: np.ndarray,
   dt: float,
) -> dict[str, object]:
   scale = np.maximum(np.max(np.abs(state), axis=1), np.ptp(state, axis=1))
   scale = np.maximum(scale, 1.0e-14)
   defect = dt * (candidate - reference) / scale[:, np.newaxis]
   component_l1 = np.mean(np.abs(defect), axis=1)
   component_linf = np.max(np.abs(defect), axis=1)
   alternating = 0.5 * np.abs(defect[:, 1:] - defect[:, :-1])
   worst_l1 = int(np.argmax(component_l1))
   worst_linf = int(np.argmax(component_linf))
   return {
      "normalized_step_l1": float(component_l1[worst_l1]),
      "step_l1_component": COMPONENT_NAMES[worst_l1],
      "normalized_step_linf": float(component_linf[worst_linf]),
      "step_linf_component": COMPONENT_NAMES[worst_linf],
      "maximum_alternating_defect": float(np.max(alternating)),
      "alternating_component": COMPONENT_NAMES[
         int(np.unravel_index(np.argmax(alternating), alternating.shape)[0])
      ],
   }


def exact_trace_rhs(
   exact_profile: np.ndarray,
   spec: dict[str, object],
   time: float,
   cell_edges: np.ndarray,
) -> tuple[np.ndarray, int]:
   gamma = float(spec["gamma"])
   left = exact_primitives_at(
      exact_profile, spec, time, cell_edges, side="left"
   )
   right = exact_primitives_at(
      exact_profile, spec, time, cell_edges, side="right"
   )
   _, flux_left = primitive_state_and_flux(left, gamma)
   _, flux_right = primitive_state_and_flux(right, gamma)
   discontinuous = np.max(np.abs(left - right), axis=0) > 1.0e-11
   flux = 0.5 * (flux_left + flux_right)
   dx = float(cell_edges[1] - cell_edges[0])
   return -(flux[:, 1:] - flux[:, :-1]) / dx, int(np.count_nonzero(discontinuous))


def operator_rows(
   case: int,
   cells: int,
   time: float,
   dt: float,
   method: str,
   riemann_methods: tuple[str, ...],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
   spec = case_spec(case)
   gamma = float(spec["gamma"])
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   cell_edges = np.linspace(-1.0, 1.0, cells + 1)
   cell_centers = 0.5 * (cell_edges[:-1] + cell_edges[1:])
   conserved = exact_average_at_time(profile, spec, time, cell_edges)
   center = exact_primitives_at(profile, spec, time, cell_centers)
   reference_rhs = exact_rhs(profile, spec, time, cell_edges, dt)
   trace_rhs, aligned_discontinuities = exact_trace_rhs(
      profile, spec, time, cell_edges
   )
   rows = [{
      "case": case,
      "cells": cells,
      "time": time,
      "dt": dt,
      "reconstruction": "exact_trace",
      "riemann": "physical",
      "representation": "exact_trace",
      "riemann_fallbacks": 0,
      "point_c2p_fallbacks": 0,
      "aligned_discontinuities": aligned_discontinuities,
      **normalized_error_metrics(
         trace_rhs, reference_rhs, conserved, dt
      ),
   }]
   defects: dict[str, np.ndarray] = {
      "exact_trace": dt * (trace_rhs - reference_rhs),
   }
   for riemann in riemann_methods:
      for representation in REPRESENTATIONS:
         result = operator_rhs(
            conserved,
            gamma,
            2.0 / cells,
            method,
            riemann,
            representation,
            exact_center=center,
         )
         label = f"{representation}_{riemann}"
         defects[label] = dt * (result.rhs - reference_rhs)
         rows.append({
            "case": case,
            "cells": cells,
            "time": time,
            "dt": dt,
            "reconstruction": method,
            "riemann": riemann,
            "representation": representation,
            "riemann_fallbacks": result.riemann_fallbacks,
            "point_c2p_fallbacks": result.point_c2p_fallbacks,
            "aligned_discontinuities": aligned_discontinuities,
            **normalized_error_metrics(
               result.rhs, reference_rhs, conserved, dt
            ),
         })
   return rows, defects


def face_trace_error_rows(
   *,
   case: int,
   cells: int,
   time: float,
   method: str,
) -> list[dict[str, object]]:
   spec = case_spec(case)
   gamma = float(spec["gamma"])
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   edges = np.linspace(-1.0, 1.0, cells + 1)
   conserved = exact_average_at_time(profile, spec, time, edges)
   recovered = recover_conserved(conserved, gamma)
   if not np.all(recovered.success):
      raise RuntimeError("exact cell averages unexpectedly failed production C2P")
   left, right = reconstruct_faces(recovered.primitive, method, gamma)
   exact_left = exact_primitives_at(profile, spec, time, edges, side="left")
   exact_right = exact_primitives_at(profile, spec, time, edges, side="right")
   exact_scale = np.maximum(
      np.max(np.abs(np.hstack((exact_left, exact_right))), axis=1),
      np.ptp(np.hstack((exact_left, exact_right)), axis=1),
   )
   exact_scale = np.maximum(exact_scale, 1.0e-14)
   rows: list[dict[str, object]] = []
   for side, candidate, exact in (
      ("left", left, exact_left),
      ("right", right, exact_right),
   ):
      error = (candidate - exact) / exact_scale[:, np.newaxis]
      alternating = 0.5 * np.abs(error[:, 1:] - error[:, :-1])
      for component, name in enumerate(PRIMITIVE_NAMES):
         rows.append({
            "case": case,
            "cells": cells,
            "time": time,
            "reconstruction": method,
            "side": side,
            "component": name,
            "normalized_l1": float(np.mean(np.abs(error[component]))),
            "normalized_linf": float(np.max(np.abs(error[component]))),
            "maximum_alternating_error": float(np.max(alternating[component])),
         })
   return rows


def intervention_rows(
   *,
   case: int,
   cells: int,
   time: float,
   dt: float,
   method: str,
   riemann_methods: tuple[str, ...],
) -> list[dict[str, object]]:
   """Replace selected numerical traces by exact traces to assign cause."""

   spec = case_spec(case)
   gamma = float(spec["gamma"])
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   edges = np.linspace(-1.0, 1.0, cells + 1)
   centers = 0.5 * (edges[:-1] + edges[1:])
   dx = 2.0 / cells
   conserved = exact_average_at_time(profile, spec, time, edges)
   center = exact_primitives_at(profile, spec, time, centers)
   reference_rhs = exact_rhs(profile, spec, time, edges, dt)
   base = operator_rhs(
      conserved,
      gamma,
      dx,
      method,
      riemann_methods[0],
      "c2p_average",
      exact_center=center,
   )
   exact_left = exact_primitives_at(profile, spec, time, edges, side="left")
   exact_right = exact_primitives_at(profile, spec, time, edges, side="right")
   groups = {
      "none": (),
      "rho": (0,),
      "pressure": (4,),
      "thermodynamic": (0, 4),
      "normal_velocity": (1,),
      "tangential_velocity": (2, 3),
      "velocity": (1, 2, 3),
      "tangential_magnetic": (6, 7),
      "transverse_mhd": (2, 3, 6, 7),
      "velocity_magnetic": (1, 2, 3, 5, 6, 7),
      "all": tuple(range(8)),
   }
   rows: list[dict[str, object]] = []
   for riemann in riemann_methods:
      base_rhs, base_fallbacks = rhs_from_faces(
         base.left_face, base.right_face, gamma, dx, riemann
      )
      base_metrics = normalized_error_metrics(
         base_rhs, reference_rhs, conserved, dt
      )
      base_linf = float(base_metrics["normalized_step_linf"])
      for group, components in groups.items():
         left = base.left_face.copy()
         right = base.right_face.copy()
         if components:
            index = list(components)
            left[index] = exact_left[index]
            right[index] = exact_right[index]
         velocity_squared_l = np.sum(left[1:4] ** 2, axis=0)
         velocity_squared_r = np.sum(right[1:4] ** 2, axis=0)
         if np.any(velocity_squared_l >= 1.0) or np.any(velocity_squared_r >= 1.0):
            rows.append({
               "case": case,
               "cells": cells,
               "time": time,
               "dt": dt,
               "reconstruction": method,
               "riemann": riemann,
               "exact_trace_group": group,
               "status": "SUPERLUMINAL_MIXED_TRACE",
               "riemann_fallbacks": "",
               "linf_ratio_to_base": "",
            })
            continue
         rhs, fallbacks = rhs_from_faces(left, right, gamma, dx, riemann)
         metrics = normalized_error_metrics(rhs, reference_rhs, conserved, dt)
         rows.append({
            "case": case,
            "cells": cells,
            "time": time,
            "dt": dt,
            "reconstruction": method,
            "riemann": riemann,
            "exact_trace_group": group,
            "status": "OK",
            "riemann_fallbacks": fallbacks,
            "linf_ratio_to_base": (
               float(metrics["normalized_step_linf"]) / base_linf
               if base_linf > 0.0 else 0.0
            ),
            **metrics,
         })
      if base_fallbacks:
         raise RuntimeError("base intervention flux unexpectedly changed fallback count")
   return rows


def local_subcell_faces(
   conserved: np.ndarray,
   primitive: np.ndarray,
   left: np.ndarray,
   right: np.ndarray,
   gamma: float,
   *,
   maximum_side_to_jump: float = 0.35,
   maximum_mixture_residual: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
   """Represent one locally isolated, unresolved jump inside a cell.

   This is a deliberately rough general prototype.  It assumes neither a wave
   family nor an exact solution.  A cell is eligible only when two cells on
   each side look like plateaus compared with the cross-cell jump, and when
   its conserved average is accurately represented by a convex mixture of
   those plateau states.  The inferred mixture fraction locates the jump;
   only the two traces emitted by that cell are replaced.
   """

   candidates: list[dict[str, object]] = []
   num_cells = conserved.shape[1]
   for cell in range(2, num_cells - 2):
      window = conserved[:, cell - 2 : cell + 3]
      scale = np.maximum(
         np.ptp(window, axis=1),
         1.0e-3 * np.max(np.abs(window), axis=1),
      )
      scale = np.maximum(scale, 1.0e-12)

      def norm(value: np.ndarray) -> float:
         return float(np.sqrt(np.mean((value / scale) ** 2)))

      state_l = 0.5 * (conserved[:, cell - 2] + conserved[:, cell - 1])
      state_r = 0.5 * (conserved[:, cell + 1] + conserved[:, cell + 2])
      jump_vector = state_l - state_r
      jump = norm(jump_vector)
      if jump <= 1.0e-8:
         continue
      side_variation = max(
         norm(conserved[:, cell - 1] - conserved[:, cell - 2]),
         norm(conserved[:, cell + 2] - conserved[:, cell + 1]),
      )
      side_ratio = side_variation / jump
      if side_ratio > maximum_side_to_jump:
         continue
      weighted_jump = jump_vector / scale
      denominator = float(np.dot(weighted_jump, weighted_jump))
      weighted_offset = (conserved[:, cell] - state_r) / scale
      fraction_l = float(np.dot(weighted_offset, weighted_jump) / denominator)
      if not 0.02 < fraction_l < 0.98:
         continue
      mixture = fraction_l * state_l + (1.0 - fraction_l) * state_r
      residual = norm(conserved[:, cell] - mixture) / jump
      if residual > maximum_mixture_residual:
         continue
      score = jump * (1.0 - side_ratio / maximum_side_to_jump) * (
         1.0 - residual / maximum_mixture_residual
      )
      candidates.append({
         "cell": cell,
         "state_l": state_l,
         "state_r": state_r,
         "fraction_l": fraction_l,
         "side_ratio": side_ratio,
         "mixture_residual": residual,
         "score": score,
      })

   # Adjacent detections describe the same unresolved structure more often
   # than two independent jumps.  Keep the strongest local representation.
   retained: list[dict[str, object]] = []
   for candidate in sorted(candidates, key=lambda item: -float(item["score"])):
      cell = int(candidate["cell"])
      if any(abs(cell - int(other["cell"])) <= 1 for other in retained):
         continue
      retained.append(candidate)
   retained.sort(key=lambda item: int(item["cell"]))
   if not retained:
      return left.copy(), right.copy(), []

   endpoint_states = np.column_stack([
      state
      for candidate in retained
      for state in (candidate["state_l"], candidate["state_r"])
   ])
   endpoints = recover_conserved(endpoint_states, gamma)
   modified_left = left.copy()
   modified_right = right.copy()
   details: list[dict[str, float]] = []
   for index, candidate in enumerate(retained):
      if not endpoints.success[2 * index] or not endpoints.success[2 * index + 1]:
         continue
      cell = int(candidate["cell"])
      primitive_l = endpoints.primitive[:, 2 * index]
      primitive_r = endpoints.primitive[:, 2 * index + 1]
      # The cell's left trace is the right state at face ``cell``; its right
      # trace is the left state at face ``cell + 1``.
      modified_right[:, cell] = primitive_l
      modified_left[:, cell + 1] = primitive_r
      details.append({
         "cell": float(cell),
         "fraction_l": float(candidate["fraction_l"]),
         "side_ratio": float(candidate["side_ratio"]),
         "mixture_residual": float(candidate["mixture_residual"]),
         "score": float(candidate["score"]),
      })
   return modified_left, modified_right, details


def subcell_prototype_rows(
   *,
   case: int,
   cells: int,
   time: float,
   dt: float,
   method: str,
   riemann_methods: tuple[str, ...],
) -> list[dict[str, object]]:
   spec = case_spec(case)
   gamma = float(spec["gamma"])
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   edges = np.linspace(-1.0, 1.0, cells + 1)
   centers = 0.5 * (edges[:-1] + edges[1:])
   conserved = exact_average_at_time(profile, spec, time, edges)
   center = exact_primitives_at(profile, spec, time, centers)
   reference_rhs = exact_rhs(profile, spec, time, edges, dt)
   base = operator_rhs(
      conserved,
      gamma,
      2.0 / cells,
      method,
      riemann_methods[0],
      "c2p_average",
      exact_center=center,
   )
   rows: list[dict[str, object]] = []
   for side_threshold in (0.20, 0.35, 0.50):
      left, right, details = local_subcell_faces(
         conserved,
         base.primitive,
         base.left_face,
         base.right_face,
         gamma,
         maximum_side_to_jump=side_threshold,
      )
      for riemann in riemann_methods:
         base_rhs, _ = rhs_from_faces(
            base.left_face, base.right_face, gamma, 2.0 / cells, riemann
         )
         candidate_rhs, fallbacks = rhs_from_faces(
            left, right, gamma, 2.0 / cells, riemann
         )
         base_metrics = normalized_error_metrics(
            base_rhs, reference_rhs, conserved, dt
         )
         candidate_metrics = normalized_error_metrics(
            candidate_rhs, reference_rhs, conserved, dt
         )
         rows.append({
            "case": case,
            "cells": cells,
            "time": time,
            "dt": dt,
            "reconstruction": method,
            "riemann": riemann,
            "maximum_side_to_jump": side_threshold,
            "marked_cells": len(details),
            "cell_indices": ",".join(str(int(item["cell"])) for item in details),
            "maximum_mixture_residual": max(
               (item["mixture_residual"] for item in details), default=0.0
            ),
            "riemann_fallbacks": fallbacks,
            "linf_ratio_to_base": (
               float(candidate_metrics["normalized_step_linf"])
               / float(base_metrics["normalized_step_linf"])
            ),
            **candidate_metrics,
         })
   return rows


def read_saved_state(path: Path) -> np.ndarray:
   names, _ = parse_plot_header_1d(path)
   state = read_composite_state_1d(path, names)
   return np.vstack([state[names.index(name)] for name in COMPONENT_NAMES])


def replay_validation_rows() -> list[dict[str, object]]:
   case = 4
   cells = 256
   spec = case_spec(case)
   gamma = float(spec["gamma"])
   run_dir = RUN_ROOT / "onset_b4_ssprk3_stage_trace_ppm4_unguarded_n256"
   first_path = run_dir / "debugStage/ssprk3_stage1_candidateGF00001.h5"
   second_path = run_dir / "debugStage/ssprk3_stage2_candidateGF00001.h5"
   third_path = run_dir / "debugStage/ssprk3_stage3_candidateGF00001.h5"
   accepted_path = run_dir / "debugStage/ssprk3_acceptedGF00001.h5"
   if not all(
      path.is_file()
      for path in (first_path, second_path, third_path, accepted_path)
   ):
      return []
   _, dt = parse_plot_header_1d(first_path)
   bounds_lo, bounds_hi = parse_plot_bounds(first_path)
   edges = np.linspace(bounds_lo[0], bounds_hi[0], cells + 1)
   centers = 0.5 * (edges[:-1] + edges[1:])
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / "solution_b4.dat"
   )
   # Use the exact production algebra for the piecewise-constant initial
   # state.  The independent NumPy Valencia map differs by O(1e-11) in this
   # W~22 state, enough to cross a later C2P basin boundary during a chained
   # replay even though each individual stage operator agrees.
   initial_primitive = exact_primitives_at(profile, spec, 0.0, centers)
   initial, _ = primitive_state_and_flux(initial_primitive, gamma)
   saved_first = read_saved_state(first_path)
   saved_second = read_saved_state(second_path)
   saved_third = read_saved_state(third_path)
   saved_accepted = read_saved_state(accepted_path)
   exact_final = exact_average_at_time(profile, spec, dt, edges)
   rows: list[dict[str, object]] = []
   stage_states = (initial, saved_first, saved_second)
   stage_weights = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)
   base_rhs = [
      operator_rhs(
         state,
         gamma,
         2.0 / cells,
         "ppm4",
         "hll",
         "c2p_average",
      ).rhs
      for state in stage_states
   ]
   for representation in ("c2p_average", "pointwise_roundtrip"):
      rhs = [
         operator_rhs(
            state,
            gamma,
            2.0 / cells,
            "ppm4",
            "hll",
            representation,
         ).rhs
         for state in stage_states
      ]
      stage_candidates = (
         initial + dt * rhs[0],
         0.75 * initial + 0.25 * (saved_first + dt * rhs[1]),
         (1.0 / 3.0) * initial
         + (2.0 / 3.0) * (saved_second + dt * rhs[2]),
      )
      for stage, candidate, saved, path in (
         ("stage1_local", stage_candidates[0], saved_first, first_path),
         ("stage2_local", stage_candidates[1], saved_second, second_path),
         ("stage3_local", stage_candidates[2], saved_third, third_path),
      ):
         difference = candidate - saved
         scale = np.maximum(np.max(np.abs(saved), axis=1), 1.0e-14)
         normalized = np.abs(difference) / scale[:, np.newaxis]
         maximum = np.unravel_index(np.argmax(normalized), normalized.shape)
         rows.append({
            "case": case,
            "cells": cells,
            "time": dt,
            "dt": dt,
            "reconstruction": "ppm4",
            "riemann": "hll",
            "representation": representation,
            "stage": stage,
            "comparison": "saved_production_stage",
            "maximum_relative_replay_error": float(np.max(normalized)),
            "replay_component": COMPONENT_NAMES[maximum[0]],
            "maximum_absolute_replay_error": float(np.max(np.abs(difference))),
            "saved_path": repo_relative(path),
         })
      linearized = saved_accepted.copy()
      for weight, candidate_rhs, production_rhs in zip(
         stage_weights, rhs, base_rhs
      ):
         linearized += dt * weight * (candidate_rhs - production_rhs)
      difference = linearized - exact_final
      scale = np.maximum(np.max(np.abs(exact_final), axis=1), 1.0e-14)
      normalized = np.abs(difference) / scale[:, np.newaxis]
      maximum = np.unravel_index(np.argmax(normalized), normalized.shape)
      rows.append({
         "case": case,
         "cells": cells,
         "time": dt,
         "dt": dt,
         "reconstruction": "ppm4",
         "riemann": "hll",
         "representation": representation,
         "stage": "accepted_frozen_linearized",
         "comparison": "exact_cell_average",
         "maximum_relative_replay_error": float(np.max(normalized)),
         "replay_component": COMPONENT_NAMES[maximum[0]],
         "maximum_absolute_replay_error": float(np.max(np.abs(difference))),
         "saved_path": repo_relative(accepted_path),
      })
   return rows


def make_plot(
   rows: list[dict[str, object]],
   defect_sets: list[tuple[str, int, np.ndarray]],
) -> None:
   selected = [
      row for row in rows
      if row.get("representation") in (*REPRESENTATIONS, "exact_trace")
   ]
   if not selected:
      return
   fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
   labels = []
   values = []
   for row in selected:
      if int(row["case"]) == 4 and row["riemann"] in {
         "physical", "hll", "hlld"
      }:
         label = (
            "exact physical trace"
            if row["representation"] == "exact_trace"
            else f"{row['representation']}\n{row['riemann']}"
         )
         labels.append(label)
         values.append(float(row["normalized_step_linf"]))
   axes[0].bar(np.arange(len(values)), values)
   axes[0].set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
   axes[0].set_yscale("log")
   axes[0].set_ylabel("normalized one-step operator defect")
   axes[0].set_title("Balsara 4: exact-state operator split")
   axes[0].grid(True, axis="y", alpha=0.25)

   if defect_sets:
      label, cells, defect = defect_sets[0]
      x = -1.0 + (np.arange(cells) + 0.5) * (2.0 / cells)
      component = int(np.argmax(np.max(np.abs(defect), axis=1)))
      axes[1].plot(x, defect[component], label=label)
      axes[1].set_xlim(-0.12, 0.12)
      axes[1].set_xlabel("x")
      axes[1].set_ylabel(f"one-step defect in {COMPONENT_NAMES[component]}")
      axes[1].set_title("Worst localized Balsara-4 defect")
      axes[1].grid(True, alpha=0.25)
      axes[1].legend()
   fig.tight_layout()
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(PLOT, dpi=180)
   plt.close(fig)


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--riemann", nargs="+", choices=("hll", "hllc", "hlld"),
      default=("hll", "hlld"),
   )
   parser.add_argument("--skip-replay", action="store_true")
   parser.add_argument("--skip-plot", action="store_true")
   return parser.parse_args()


def main() -> int:
   args = parse_args()
   cases = (
      # The B4 time matches the fourth accepted N=256 step, after the
      # alternating defect has become visible in the onset trace.
      (4, 256, 0.02812507535346939, 0.0070312688819208395, "ppm4"),
      # B3 is sampled both before and after its narrow shell becomes resolved.
      (3, 128, 0.028275215528439496, 0.014137607764219748, "mp5"),
      (3, 512, 0.4, 0.9 * (2.0 / 512), "mp5"),
      (3, 1024, 0.4, 0.9 * (2.0 / 1024), "mp5"),
   )
   rows: list[dict[str, object]] = []
   interventions: list[dict[str, object]] = []
   face_errors: list[dict[str, object]] = []
   subcell_rows: list[dict[str, object]] = []
   defect_sets: list[tuple[str, int, np.ndarray]] = []
   for case, cells, time, dt, method in cases:
      case_rows, defects = operator_rows(
         case,
         cells,
         time,
         dt,
         method,
         tuple(args.riemann),
      )
      rows.extend(case_rows)
      if case == 4:
         defect_sets.append((
            "production representation + HLL",
            cells,
            defects["c2p_average_hll"],
         ))
      if case == 4 or (case == 3 and cells == 1024):
         interventions.extend(intervention_rows(
            case=case,
            cells=cells,
            time=time,
            dt=dt,
            method=method,
            riemann_methods=tuple(args.riemann),
         ))
         face_errors.extend(face_trace_error_rows(
            case=case,
            cells=cells,
            time=time,
            method=method,
         ))
         subcell_rows.extend(subcell_prototype_rows(
            case=case,
            cells=cells,
            time=time,
            dt=dt,
            method=method,
            riemann_methods=tuple(args.riemann),
         ))
   RESULTS.parent.mkdir(parents=True, exist_ok=True)
   write_tsv(RESULTS, rows, list(rows[0]))
   write_tsv(INTERVENTIONS, interventions, list(interventions[0]))
   write_tsv(FACE_ERRORS, face_errors, list(face_errors[0]))
   write_tsv(SUBCELL_RESULTS, subcell_rows, list(subcell_rows[0]))
   replay_rows = [] if args.skip_replay else replay_validation_rows()
   replay_path = RESULTS.with_name("balsara_operator_replay.tsv")
   if replay_rows:
      write_tsv(replay_path, replay_rows, list(replay_rows[0]))
   if not args.skip_plot:
      make_plot(rows, defect_sets)

   print(f"wrote {repo_relative(RESULTS)}")
   print(f"wrote {repo_relative(INTERVENTIONS)}")
   print(f"wrote {repo_relative(FACE_ERRORS)}")
   print(f"wrote {repo_relative(SUBCELL_RESULTS)}")
   if replay_rows:
      print(f"wrote {repo_relative(replay_path)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(PLOT)}")
   for row in rows:
      if row["representation"] in REPRESENTATIONS:
         print(
            f"B{row['case']} N={row['cells']} {row['representation']} "
            f"{row['riemann']}: Linf={row['normalized_step_linf']:.3e} "
            f"alt={row['maximum_alternating_defect']:.3e} "
            f"fallbacks={row['riemann_fallbacks']}"
         )
   for row in replay_rows:
      print(
         f"replay {row['representation']} {row['stage']}: "
         f"relative={row['maximum_relative_replay_error']:.3e} "
         f"absolute={row['maximum_absolute_replay_error']:.3e}"
      )
   for case in (4, 3):
      selected = [
         row for row in interventions
         if int(row["case"]) == case
         and row["riemann"] == args.riemann[0]
         and row["status"] == "OK"
      ]
      best = min(selected, key=lambda row: float(row["linf_ratio_to_base"]))
      print(
         f"B{case} best exact-trace intervention: "
         f"{best['exact_trace_group']} ratio={best['linf_ratio_to_base']:.3e}"
      )
      candidates = [
         row for row in subcell_rows
         if int(row["case"]) == case and row["riemann"] == args.riemann[0]
      ]
      best_subcell = min(
         candidates, key=lambda row: float(row["linf_ratio_to_base"])
      )
      print(
         f"B{case} best local-subcell prototype: "
         f"ratio={best_subcell['linf_ratio_to_base']:.3e} "
         f"marked={best_subcell['marked_cells']} "
         f"cells={best_subcell['cell_indices']}"
      )
   return 0


if __name__ == "__main__":
   sys.exit(main())
