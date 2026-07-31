#!/usr/bin/env python3
"""Validate finite-volume sampling of analytic and discontinuous references."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyfluid.exact_solutions import (  # noqa: E402
   sod_conserved_integrals,
   sod_exact_solution,
   sod_wave_breakpoints,
)
from exact_reference import (
   cell_average_newtonian_hydro,
   cell_average_function,
   cell_average_tabulated,
   cell_average_tabulated_mapped,
   conservatively_shift_periodic_cell_averages,
)  # noqa: E402
from validate_rmhd import (
   EXACT_BALSARA_CASES,
   exact_rmhd_cell_averages,
   read_last_exact_balsara_profile,
)  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def require_close(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
   error = float(np.max(np.abs(actual - expected)))
   scale = max(1.0, float(np.max(np.abs(expected))))
   if error > 2.0e-13 * scale:
      raise RuntimeError(f"{label}: error={error:.3e}")


def euler_conserved_and_flux(
   primitive: tuple[float, float, float],
   gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
   rho, velocity, pressure = primitive
   energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity * velocity
   return (
      np.asarray((rho, rho * velocity, energy)),
      np.asarray(
         (
            rho * velocity,
            rho * velocity * velocity + pressure,
            velocity * (energy + pressure),
         )
      ),
   )


def main() -> int:
   step_x = np.asarray((0.0, 0.5, 0.5, 1.0))
   step_values = np.asarray((1.0, 1.0, 3.0, 3.0))
   step_edges = np.asarray((-0.25, 0.25, 0.75, 1.25))
   require_close(
      "duplicate-coordinate jump and constant exterior",
      cell_average_tabulated(step_x, step_values, step_edges),
      np.asarray((1.0, 2.0, 3.0)),
   )

   vector_values = np.vstack((step_values, 2.0 * step_values - 1.0))
   require_close(
      "vector tabulated profile",
      cell_average_tabulated(step_x, vector_values, step_edges),
      np.asarray(((1.0, 2.0, 3.0), (1.0, 3.0, 5.0))),
   )

   periodic_values = np.asarray((0.0, 1.0, 2.0, 3.0))
   require_close(
      "conservative periodic cell-average translation",
      conservatively_shift_periodic_cell_averages(periodic_values, 1.25),
      (
         0.75 * np.roll(periodic_values, 1)
         + 0.25 * np.roll(periodic_values, 2)
      ),
   )

   nonlinear_edges = np.linspace(0.0, 1.0, 5)
   nonlinear_average = cell_average_tabulated_mapped(
      np.asarray((0.0, 1.0)),
      np.asarray(((0.0, 1.0),)),
      nonlinear_edges,
      lambda samples: samples * samples,
      quadrature_order=2,
   )
   nonlinear_lo = nonlinear_edges[:-1]
   nonlinear_hi = nonlinear_edges[1:]
   require_close(
      "nonlinear tabulated transform",
      nonlinear_average[0],
      (nonlinear_hi**3 - nonlinear_lo**3)
      / (3.0 * (nonlinear_hi - nonlinear_lo)),
   )

   polynomial_edges = np.linspace(-1.0, 1.0, 9)
   polynomial_average = cell_average_function(
      lambda x: x**4 - 2.0 * x**2 + 3.0,
      polynomial_edges,
      quadrature_order=4,
   )
   lo = polynomial_edges[:-1]
   hi = polynomial_edges[1:]
   polynomial_integral = (
      (hi**5 - lo**5) / 5.0
      - 2.0 * (hi**3 - lo**3) / 3.0
      + 3.0 * (hi - lo)
   )
   require_close(
      "polynomial quadrature",
      polynomial_average,
      polynomial_integral / (hi - lo),
   )

   discontinuous_average = cell_average_function(
      lambda x: np.where(x < 0.1, -2.0, 4.0),
      np.asarray((-0.4, 0.4)),
      breakpoints=(0.1,),
      quadrature_order=2,
   )
   require_close(
      "callable jump split",
      discontinuous_average,
      np.asarray((0.25,)),
   )

   sod_points = sod_wave_breakpoints(
      0.2,
      1.4,
      (1.0, 0.0, 1.0),
      (0.125, 0.0, 0.1),
      x0=-0.1,
      domain_lo=-2.0,
      domain_hi=3.0,
   )
   if (
      sod_points[0] != -2.0
      or sod_points[-1] != 3.0
      or tuple(sorted(sod_points)) != sod_points
   ):
      raise RuntimeError("Sod breakpoints do not honor the requested domain")

   euler_cases = (
      (
         "left rarefaction and right shock",
         (1.0, 0.0, 1.0),
         (0.125, 0.0, 0.1),
         0.2,
      ),
      (
         "left shock and right rarefaction",
         (1.0, 0.0, 0.01),
         (1.0, 0.0, 100.0),
         0.012,
      ),
      (
         "two shocks",
         (5.99924, 19.5975, 460.894),
         (5.99242, -6.19633, 46.095),
         0.006,
      ),
      (
         "two rarefactions",
         (1.0, -1.0, 1.0),
         (1.0, 1.0, 1.0),
         0.05,
      ),
   )
   euler_edges = np.linspace(0.0, 1.0, 257)
   euler_gamma = 1.4
   for label, left, right, final_time in euler_cases:
      conserved_left, flux_left = euler_conserved_and_flux(
         left,
         euler_gamma,
      )
      conserved_right, flux_right = euler_conserved_and_flux(
         right,
         euler_gamma,
      )
      balance_integrals = (
         0.5 * (conserved_left + conserved_right)
         + final_time * (flux_left - flux_right)
      )
      averages = cell_average_newtonian_hydro(
         lambda points, left=left, right=right, final_time=final_time: (
            sod_exact_solution(
               points,
               final_time,
               euler_gamma,
               left,
               right,
            )
         ),
         euler_edges,
         euler_gamma,
         breakpoints=sod_wave_breakpoints(
            final_time,
            euler_gamma,
            left,
            right,
         ),
      )
      rho, pressure, velocity = averages
      conserved_means = np.asarray(
         (
            np.mean(rho),
            np.mean(rho * velocity),
            np.mean(
               pressure / (euler_gamma - 1.0)
               + 0.5 * rho * velocity * velocity
            ),
         )
      )
      exact_integrals = sod_conserved_integrals(
         final_time,
         euler_gamma,
         left,
         right,
      )
      integral_vector = np.asarray(
         (
            exact_integrals["mass"],
            exact_integrals["momx"],
            exact_integrals["energy"],
         )
      )
      require_close(
         f"Euler {label} domain balance",
         integral_vector,
         balance_integrals,
      )
      require_close(
         f"Euler {label} conserved cell averages",
         conserved_means,
         integral_vector,
      )

   for spec in EXACT_BALSARA_CASES:
      case = int(spec["case"])
      profile = read_last_exact_balsara_profile(
         ROOT / "exact_balsara" / f"solution_b{case}.dat"
      )
      if np.any(np.diff(profile[:, 0]) < 0.0):
         raise RuntimeError(f"Balsara {case} profile remains out of order")
      averages = []
      for cells in (32, 64):
         averages.append(exact_rmhd_cell_averages(
            profile,
            normal_magnetic=float(spec["left"][5]),
            gamma=float(spec["gamma"]),
            cell_edges=np.linspace(-1.0, 1.0, cells + 1),
         ))
      require_close(
         f"Balsara {case} domain integral under repartition",
         np.mean(averages[0], axis=1),
         np.mean(averages[1], axis=1),
      )
   print("exact reference cell-average checks passed")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
