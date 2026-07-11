from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class CenteredDiffusionStencil:
   """Dimension-independent centered finite-volume diffusion stencil."""

   center: sp.Expr
   neighbors: tuple[tuple[sp.Expr, sp.Expr], ...]
   coefficients: tuple[sp.Expr, ...]

   def __post_init__(self) -> None:
      if len(self.neighbors) != len(self.coefficients):
         raise ValueError("each neighbor pair requires one directional coefficient")

   def laplacian(self) -> sp.Expr:
      return sum(
         coefficient * (lower - 2 * self.center + upper)
         for (lower, upper), coefficient in zip(
            self.neighbors,
            self.coefficients,
            strict=True,
         )
      )

   def implicit_relaxed(self, base: sp.Expr) -> sp.Expr:
      neighbor_sum = sum(
         coefficient * (lower + upper)
         for (lower, upper), coefficient in zip(
            self.neighbors,
            self.coefficients,
            strict=True,
         )
      )
      return (base + neighbor_sum) / (1 + 2 * sum(self.coefficients))

   def implicit_residual(self, base: sp.Expr) -> sp.Expr:
      return self.center - base - self.laplacian()


@dataclass(frozen=True)
class FiniteVolumeDivergence:
   face_fluxes: tuple[tuple[sp.Expr, sp.Expr], ...]
   volume: sp.Expr

   def value(self) -> sp.Expr:
      return sum(upper - lower for upper, lower in self.face_fluxes) / self.volume

