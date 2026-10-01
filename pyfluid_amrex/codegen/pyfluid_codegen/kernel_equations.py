from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def differentiate_with_bindings(
   expression: sp.Expr,
   variable: sp.Symbol,
   bindings: tuple[tuple[sp.Symbol, sp.Expr], ...],
) -> sp.Expr:
   """Total derivative through ordered numerical intermediates.

   Retain the intermediates in the result instead of substituting and
   expanding them. This preserves the same normalization in a function and
   its derivative, including when a binding depends on the differentiation
   variable (for example a Lorentz factor in primitive recovery).
   """
   derivatives = {variable: sp.Integer(1)}

   def total_derivative(value: sp.Expr) -> sp.Expr:
      return sum(
         sp.diff(value, symbol) * derivatives.get(symbol, sp.Integer(0))
         for symbol in value.free_symbols
      )

   for symbol, value in bindings:
      derivatives[symbol] = total_derivative(value)
   return total_derivative(expression)


@dataclass(frozen=True)
class ScalarKernel:
   name: str
   arguments: tuple[str | tuple[str, str], ...]
   expression: sp.Expr
   # Ordered numerical intermediates kept outside algebraic CSE. Use for
   # dimensionless normalization or other evaluation-order-sensitive algebra.
   bindings: tuple[tuple[sp.Symbol, sp.Expr], ...] = ()


@dataclass(frozen=True)
class StructKernel:
   name: str
   return_type: str
   arguments: tuple[tuple[str, str], ...]
   outputs: tuple[str, ...]
   expressions: tuple[sp.Expr, ...]
   bindings: tuple[tuple[sp.Symbol, sp.Expr], ...] = ()
