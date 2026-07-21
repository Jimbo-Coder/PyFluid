from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


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
   for order in (2, 4, 6, 8):
      radius = order // 2
      offsets = tuple(range(-radius, radius + 1))
      stencils.append(FiniteDifferenceStencil(order, offsets, _weights(derivative, offsets)))
   return tuple(stencils)


def one_sided_stencils() -> tuple[
   tuple[FiniteDifferenceStencil, FiniteDifferenceStencil], ...
]:
   stencils = []
   for order in (2, 4, 6, 8):
      backward_offsets = tuple(range(-order, 1))
      forward_offsets = tuple(range(0, order + 1))
      stencils.append(
         (
            FiniteDifferenceStencil(
               order, backward_offsets, _weights(1, backward_offsets)
            ),
            FiniteDifferenceStencil(
               order, forward_offsets, _weights(1, forward_offsets)
            ),
         )
      )
   return tuple(stencils)


def upwind_stencils() -> tuple[tuple[FiniteDifferenceStencil, FiniteDifferenceStencil], ...]:
   stencils = []
   for order in (2, 4, 6, 8):
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
   for order in (2, 4, 6, 8):
      radius = order // 2 + 1
      offsets = tuple(range(-radius, radius + 1))
      raw = tuple(
         sp.Rational((-1) ** (offset + radius) * sp.binomial(2 * radius, offset + radius))
         for offset in offsets
      )
      scale = sp.Rational((-1) ** (radius + 1), 2 ** (2 * radius))
      stencils.append(FiniteDifferenceStencil(order, offsets, tuple(scale * value for value in raw)))
   return tuple(stencils)
