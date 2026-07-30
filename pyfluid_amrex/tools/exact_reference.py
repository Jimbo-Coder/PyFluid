"""Finite-volume sampling helpers for analytic and tabulated references."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math

import numpy as np


Array = np.ndarray


def conservatively_shift_periodic_cell_averages(
   values: Array,
   shift_in_cells: float,
) -> Array:
   """Translate periodic piecewise-constant cell averages exactly."""

   averages = np.asarray(values, dtype=float)
   if averages.ndim < 1 or not np.all(np.isfinite(averages)):
      raise ValueError("periodic cell averages must be finite")
   if not np.isfinite(shift_in_cells):
      raise ValueError("periodic shift must be finite")
   whole_cells = math.floor(shift_in_cells)
   fraction = shift_in_cells - whole_cells
   return (
      (1.0 - fraction) * np.roll(averages, whole_cells, axis=-1)
      + fraction * np.roll(averages, whole_cells + 1, axis=-1)
   )


def cell_average_newtonian_hydro(
   primitive_solution: Callable[[Array], tuple[Array, Array, Array]],
   cell_edges: Array,
   gamma: float,
   *,
   breakpoints: Iterable[float] = (),
   quadrature_order: int = 32,
) -> tuple[Array, Array, Array]:
   """Return primitives recovered from exact conserved cell averages."""

   if gamma <= 1.0:
      raise ValueError("gamma must exceed one")

   def conserved(points: Array) -> Array:
      rho, pressure, velocity = primitive_solution(points)
      return np.vstack((
         rho,
         rho * velocity,
         pressure / (gamma - 1.0) + 0.5 * rho * velocity * velocity,
      ))

   averaged = cell_average_function(
      conserved,
      cell_edges,
      breakpoints=breakpoints,
      quadrature_order=quadrature_order,
   )
   rho = averaged[0]
   velocity = averaged[1] / rho
   pressure = (gamma - 1.0) * (
      averaged[2] - 0.5 * averaged[1] * averaged[1] / rho
   )
   return rho, pressure, velocity


def _validated_cell_edges(cell_edges: Array) -> Array:
   edges = np.asarray(cell_edges, dtype=float)
   if (
      edges.ndim != 1
      or edges.size < 2
      or not np.all(np.isfinite(edges))
      or np.any(np.diff(edges) <= 0.0)
   ):
      raise ValueError("cell edges must be a finite strictly increasing vector")
   return edges


def cell_average_function(
   function: Callable[[Array], Array],
   cell_edges: Array,
   *,
   breakpoints: Iterable[float] = (),
   quadrature_order: int = 32,
) -> Array:
   """Average a scalar/vector function over cells, splitting known jumps."""

   edges = _validated_cell_edges(cell_edges)
   if quadrature_order < 1:
      raise ValueError("quadrature order must be positive")
   split_points = np.asarray(tuple(breakpoints), dtype=float)
   if split_points.ndim != 1 or not np.all(np.isfinite(split_points)):
      raise ValueError("breakpoints must be finite")
   split_points = np.unique(split_points)
   nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
   integrated: Array | None = None

   for cell in range(edges.size - 1):
      cell_lo = edges[cell]
      cell_hi = edges[cell + 1]
      inside = split_points[
         (split_points > cell_lo) & (split_points < cell_hi)
      ]
      segments = np.concatenate(([cell_lo], inside, [cell_hi]))
      cell_integral: Array | None = None
      for segment_lo, segment_hi in zip(
         segments[:-1], segments[1:], strict=True
      ):
         center = 0.5 * (segment_lo + segment_hi)
         half_width = 0.5 * (segment_hi - segment_lo)
         points = center + half_width * nodes
         values = np.asarray(function(points), dtype=float)
         if values.shape == points.shape:
            values = values[np.newaxis, :]
         if (
            values.ndim < 2
            or values.shape[-1] != points.size
            or not np.all(np.isfinite(values))
         ):
            raise ValueError(
               "reference function must return finite values with sample axis last"
            )
         segment_integral = half_width * np.sum(
            values * weights,
            axis=-1,
         )
         cell_integral = (
            segment_integral
            if cell_integral is None
            else cell_integral + segment_integral
         )
      if cell_integral is None:
         raise RuntimeError("cell integration produced no segments")
      if integrated is None:
         integrated = np.empty((*cell_integral.shape, edges.size - 1))
      integrated[..., cell] = cell_integral / (cell_hi - cell_lo)

   if integrated is None:
      raise RuntimeError("cell integration produced no cells")
   return integrated[0] if integrated.shape[:-1] == (1,) else integrated


def cell_average_tabulated(
   coordinates: Array,
   values: Array,
   cell_edges: Array,
) -> Array:
   """Exactly average a discontinuous piecewise-linear tabulated profile.

   Repeated coordinates encode zero-width jumps. Values outside the tabulated
   coordinate range are completed with the nearest endpoint state, matching
   the constant exterior states emitted by the exact RMHD Riemann solver.
   The sample coordinate is the last axis of ``values``.
   """

   profile = np.asarray(values, dtype=float)
   scalar = profile.ndim == 1
   if scalar:
      profile = profile[np.newaxis, :]
   averaged = cell_average_tabulated_mapped(
      coordinates,
      profile,
      cell_edges,
      lambda samples: samples,
      quadrature_order=2,
   )
   return averaged[0] if scalar else averaged


def cell_average_tabulated_mapped(
   coordinates: Array,
   values: Array,
   cell_edges: Array,
   transform: Callable[[Array], Array],
   *,
   quadrature_order: int = 8,
) -> Array:
   """Average a nonlinear transform of a piecewise-linear tabulated state."""

   x = np.asarray(coordinates, dtype=float)
   profile = np.asarray(values, dtype=float)
   edges = _validated_cell_edges(cell_edges)
   if quadrature_order < 1:
      raise ValueError("quadrature order must be positive")
   if (
      x.ndim != 1
      or x.size < 2
      or not np.all(np.isfinite(x))
      or np.any(np.diff(x) < 0.0)
   ):
      raise ValueError("profile coordinates must be finite and nondecreasing")
   if (
      profile.ndim < 1
      or profile.shape[-1] != x.size
      or not np.all(np.isfinite(profile))
   ):
      raise ValueError(
         "profile values must be finite with the sample coordinate last"
      )

   cell_widths = np.diff(edges)
   integrals: list[Array | None] = [None] * cell_widths.size
   covered = np.zeros(cell_widths.size)
   nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)

   def add_segment(
      segment_lo: float,
      segment_hi: float,
      state_lo: Array,
      state_hi: Array,
   ) -> None:
      lo = np.maximum(edges[:-1], segment_lo)
      hi = np.minimum(edges[1:], segment_hi)
      for cell in np.flatnonzero(hi > lo):
         overlap_lo = lo[cell]
         overlap_hi = hi[cell]
         center = 0.5 * (overlap_lo + overlap_hi)
         half_width = 0.5 * (overlap_hi - overlap_lo)
         points = center + half_width * nodes
         fraction = (points - segment_lo) / (segment_hi - segment_lo)
         samples = (
            state_lo[..., np.newaxis] * (1.0 - fraction)
            + state_hi[..., np.newaxis] * fraction
         )
         mapped = np.asarray(transform(samples), dtype=float)
         if mapped.shape == points.shape:
            mapped = mapped[np.newaxis, :]
         if (
            mapped.ndim < 2
            or mapped.shape[-1] != points.size
            or not np.all(np.isfinite(mapped))
         ):
            raise ValueError(
               "profile transform must return finite values with sample axis last"
            )
         contribution = half_width * np.sum(mapped * weights, axis=-1)
         integrals[cell] = (
            contribution
            if integrals[cell] is None
            else integrals[cell] + contribution
         )
         covered[cell] += overlap_hi - overlap_lo

   if edges[0] < x[0]:
      add_segment(
         edges[0],
         min(edges[-1], x[0]),
         profile[..., 0],
         profile[..., 0],
      )

   for sample in range(x.size - 1):
      segment_lo = x[sample]
      segment_hi = x[sample + 1]
      if segment_hi <= segment_lo:
         continue
      add_segment(
         segment_lo,
         segment_hi,
         profile[..., sample],
         profile[..., sample + 1],
      )

   if edges[-1] > x[-1]:
      add_segment(
         max(edges[0], x[-1]),
         edges[-1],
         profile[..., -1],
         profile[..., -1],
      )

   tolerance = (
      256.0
      * np.finfo(float).eps
      * max(1.0, abs(edges[0]), abs(edges[-1]))
   )
   if np.max(np.abs(covered - cell_widths)) > tolerance:
      raise ValueError(
         "tabulated profile does not cover the requested cells"
      )
   if any(integral is None for integral in integrals):
      raise RuntimeError("tabulated profile integration missed a cell")
   return np.stack(
      [integral for integral in integrals if integral is not None],
      axis=-1,
   ) / cell_widths
