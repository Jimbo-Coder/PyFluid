from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ScalarKernel:
   name: str
   arguments: tuple[str | tuple[str, str], ...]
   expression: sp.Expr


@dataclass(frozen=True)
class StructKernel:
   name: str
   return_type: str
   arguments: tuple[tuple[str, str], ...]
   outputs: tuple[str, ...]
   expressions: tuple[sp.Expr, ...]
