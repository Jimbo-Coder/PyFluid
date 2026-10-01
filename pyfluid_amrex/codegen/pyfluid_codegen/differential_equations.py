from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .numerical_catalog import SPATIAL_ORDERS


@dataclass(frozen=True)
class FiniteDifferenceStencil:
   order: int
   offsets: tuple[int, ...]
   weights: tuple[sp.Rational, ...]


def _weights(derivative: int, offsets: tuple[int, ...]) -> tuple[sp.Rational, ...]:
   table = sp.finite_diff_weights(derivative, offsets, 0)
   return tuple(sp.Rational(weight) for weight in table[derivative][-1])


def centered_stencils(derivative: int) -> tuple[FiniteDifferenceStencil, ...]:
   stencils = []
   for order in SPATIAL_ORDERS:
      radius = order // 2
      offsets = tuple(range(-radius, radius + 1))
      stencils.append(FiniteDifferenceStencil(order, offsets, _weights(derivative, offsets)))
   return tuple(stencils)


def one_sided_stencils(derivative: int = 1) -> tuple[
   tuple[FiniteDifferenceStencil, FiniteDifferenceStencil], ...
]:
   stencils = []
   for order in SPATIAL_ORDERS:
      # A one-sided derivative of degree d needs order+d points.  In
      # particular, the fourth-order second derivative needs six, not five.
      radius = order + derivative - 1
      backward_offsets = tuple(range(-radius, 1))
      forward_offsets = tuple(range(0, radius + 1))
      stencils.append(
         (
            FiniteDifferenceStencil(
               order, backward_offsets, _weights(derivative, backward_offsets)
            ),
            FiniteDifferenceStencil(
               order, forward_offsets, _weights(derivative, forward_offsets)
            ),
         )
      )
   return tuple(stencils)


def forward_difference_weights(degree: int) -> tuple[sp.Rational, ...]:
   """Unscaled forward-difference factor used by negative-semidefinite KO."""
   return tuple(
      sp.Rational((-1) ** (degree - index) * sp.binomial(degree, index))
      for index in range(degree + 1)
   )


def upwind_stencils() -> tuple[tuple[FiniteDifferenceStencil, FiniteDifferenceStencil], ...]:
   stencils = []
   for order in SPATIAL_ORDERS:
      radius = order // 2
      # Shift the centered stencil by one point in the upwind direction.
      # This is less biased than a fully one-sided stencil and requires only
      # order/2+1 ghost points.
      backward_offsets = tuple(range(-radius - 1, radius))
      forward_offsets = tuple(range(-radius + 1, radius + 2))
      stencils.append(
         (
            FiniteDifferenceStencil(order, backward_offsets, _weights(1, backward_offsets)),
            FiniteDifferenceStencil(order, forward_offsets, _weights(1, forward_offsets)),
         )
      )
   return tuple(stencils)


def ko_stencils() -> tuple[FiniteDifferenceStencil, ...]:
   stencils = []
   for order in SPATIAL_ORDERS:
      radius = order // 2 + 1
      offsets = tuple(range(-radius, radius + 1))
      raw = tuple(
         sp.Rational((-1) ** (offset + radius) * sp.binomial(2 * radius, offset + radius))
         for offset in offsets
      )
      scale = sp.Rational((-1) ** (radius + 1), 2 ** (2 * radius))
      stencils.append(FiniteDifferenceStencil(order, offsets, tuple(scale * value for value in raw)))
   return tuple(stencils)


def centering_stencils() -> tuple[FiniteDifferenceStencil, ...]:
   """Point interpolation between staggered uniform lattices, not FV averages."""
   stencils = []
   for order in SPATIAL_ORDERS:
      offsets = tuple(range(1 - order // 2, 1 + order // 2))
      table = sp.finite_diff_weights(0, offsets, sp.Rational(1, 2))
      stencils.append(FiniteDifferenceStencil(order, offsets, tuple(table[0][-1])))
   return tuple(stencils)


def average_stencils(*, to_average: bool) -> tuple[FiniteDifferenceStencil, ...]:
   """Uniform Cartesian point <-> unit-cell-average polynomial functionals.

   These are linear smooth-region operators. They are not a positivity limiter
   or a conservative remap, and cannot act directly on curvilinear averages.
   Symmetry supplies one extra exact odd moment, so order p uses p-1 points.
   """
   stencils = []
   half = sp.Rational(1, 2)
   for order in SPATIAL_ORDERS:
      radius = order // 2 - 1
      offsets = tuple(range(-radius, radius + 1))
      def moment(offset: int, degree: int, averaged: bool) -> sp.Rational:
         if averaged:
            return ((offset + half) ** (degree + 1)
                    - (offset - half) ** (degree + 1)) / (degree + 1)
         return sp.Rational(offset) ** degree
      matrix = sp.Matrix([
         [moment(offset, degree, not to_average) for offset in offsets]
         for degree in range(len(offsets))
      ])
      target = sp.Matrix([
         moment(0, degree, to_average) for degree in range(len(offsets))
      ])
      weights = tuple(matrix.inv() * target)
      stencils.append(FiniteDifferenceStencil(order, offsets, weights))
   return tuple(stencils)
