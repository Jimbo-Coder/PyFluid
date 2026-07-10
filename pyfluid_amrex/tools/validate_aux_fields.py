#!/usr/bin/env python3
"""Validate AMReX auxiliary j/e plotfiles against cell-centered formulas.

This tool is intentionally narrow: it checks uniform periodic 2D/3D plotfiles.
AMR plotfiles need ghost-fill context to recompute derivatives exactly from
disk, so use this for algebra/path validation and separate AMR hierarchy/visual
checks for multilevel artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plot_amrex_slice import parse_plot_header, read_level_state


def central_diff(values: np.ndarray, axis: int, spacing: float) -> np.ndarray:
   return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing)


def component_map(names: list[str]) -> dict[str, int]:
   return {name: idx for idx, name in enumerate(names)}


def require_components(names: dict[str, int], required: list[str], label: str) -> None:
   missing = [name for name in required if name not in names]
   if missing:
      raise ValueError(f"{label} is missing required components: {', '.join(missing)}")


def relative_error(candidate: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
   diff = candidate - expected
   scale = max(float(np.max(np.abs(expected))), 1.0e-30)
   return (
      float(np.mean(np.abs(diff)) / scale),
      float(np.sqrt(np.mean(diff * diff)) / scale),
      float(np.max(np.abs(diff)) / scale),
   )


def absolute_linf(candidate: np.ndarray, expected: np.ndarray) -> float:
   return float(np.max(np.abs(candidate - expected)))


def expected_aux_fields(state: np.ndarray, names: dict[str, int], eta: float) -> dict[str, np.ndarray]:
   ndim = state.ndim - 1
   require_components(
      names,
      ["rho", "mom1", "mom2", "mom3", "b1", "b2", "b3"],
      "state plotfile",
   )
   shape = state.shape[1:]
   spacing = [1.0 / n for n in shape]

   rho = state[names["rho"]]
   vx = state[names["mom1"]] / rho
   vy = state[names["mom2"]] / rho
   vz = state[names["mom3"]] / rho
   bx = state[names["b1"]]
   by = state[names["b2"]]
   bz = state[names["b3"]]

   dbx_dx = central_diff(bx, 0, spacing[0])
   dby_dx = central_diff(by, 0, spacing[0])
   dbz_dx = central_diff(bz, 0, spacing[0])

   if ndim >= 2:
      dbx_dy = central_diff(bx, 1, spacing[1])
      dbz_dy = central_diff(bz, 1, spacing[1])
   else:
      dbx_dy = np.zeros_like(bx)
      dbz_dy = np.zeros_like(bx)

   if ndim == 3:
      dbx_dz = central_diff(bx, 2, spacing[2])
      dby_dz = central_diff(by, 2, spacing[2])
   else:
      dbx_dz = np.zeros_like(bx)
      dby_dz = np.zeros_like(bx)

   jx = dbz_dy - dby_dz
   jy = dbx_dz - dbz_dx
   jz = dby_dx - dbx_dy

   return {
      "j1": jx,
      "j2": jy,
      "j3": jz,
      "e1": -(vy * bz - vz * by) + eta * jx,
      "e2": -(vz * bx - vx * bz) + eta * jy,
      "e3": -(vx * by - vy * bx) + eta * jz,
      "magnetic_energy": 0.5 * (bx * bx + by * by + bz * bz),
   }


def validate(args: argparse.Namespace) -> int:
   state_names, ndim, state_time = parse_plot_header(args.state_plotfile)
   aux_names, aux_ndim, aux_time = parse_plot_header(args.aux_plotfile)
   if ndim != aux_ndim:
      raise ValueError(f"dimension mismatch: state ndim={ndim}, aux ndim={aux_ndim}")
   if ndim not in (2, 3):
      raise ValueError("aux validation supports 2D or 3D plotfiles")
   if abs(state_time - aux_time) > args.time_tolerance:
      raise ValueError(f"time mismatch: state t={state_time}, aux t={aux_time}")

   state = read_level_state(args.state_plotfile, state_names, ndim, args.level)
   aux = read_level_state(args.aux_plotfile, aux_names, ndim, args.level)
   if state.shape[1:] != aux.shape[1:]:
      raise ValueError(f"shape mismatch: state {state.shape[1:]}, aux {aux.shape[1:]}")

   state_idx = component_map(state_names)
   aux_idx = component_map(aux_names)
   expected = expected_aux_fields(state, state_idx, args.resistivity)

   failures = []
   for name, values in expected.items():
      if name not in aux_idx:
         continue
      candidate = aux[aux_idx[name]]
      rel_l1, rel_l2, rel_linf = relative_error(candidate, values)
      abs_linf = absolute_linf(candidate, values)
      print(
         f"{name}: rel_l1={rel_l1:.3e} rel_l2={rel_l2:.3e} "
         f"rel_linf={rel_linf:.3e} abs_linf={abs_linf:.3e}"
      )
      if rel_linf > args.relative_tolerance and abs_linf > args.absolute_tolerance:
         failures.append(name)

   if "divb" in aux_idx:
      divb_linf = float(np.max(np.abs(aux[aux_idx["divb"]])))
      print(f"divb: abs_linf={divb_linf:.3e}")
      if args.ct_divb and divb_linf > args.absolute_tolerance:
         failures.append("divb")

   if failures:
      print("FAILED: " + ", ".join(failures), file=sys.stderr)
      return 1
   return 0


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--state-plotfile", required=True, type=Path)
   parser.add_argument("--aux-plotfile", required=True, type=Path)
   parser.add_argument("--resistivity", type=float, default=0.0)
   parser.add_argument("--level", type=int, default=0)
   parser.add_argument("--relative-tolerance", type=float, default=1.0e-12)
   parser.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
   parser.add_argument("--time-tolerance", type=float, default=1.0e-14)
   parser.add_argument("--ct-divb", action="store_true", help="require aux divb to be at roundoff")
   args = parser.parse_args()
   return validate(args)


if __name__ == "__main__":
   sys.exit(main())
