from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class OrthogonalFrame:
   name: str
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr]
   cartesian_position: tuple[sp.Expr, sp.Expr, sp.Expr] | None = None

   @property
   def volume_scale(self) -> sp.Expr:
      h1, h2, h3 = self.scale_factors
      return h1 * h2 * h3


def coordinate_symbols() -> dict[str, sp.Symbol]:
   names = "x y z r theta phi"
   return {name: symbol for name, symbol in zip(names.split(), sp.symbols(names), strict=True)}


def cartesian_frame() -> OrthogonalFrame:
   s = coordinate_symbols()
   return OrthogonalFrame(
      name="cartesian",
      coordinates=(s["x"], s["y"], s["z"]),
      scale_factors=(sp.Integer(1), sp.Integer(1), sp.Integer(1)),
      cartesian_position=(s["x"], s["y"], s["z"]),
   )


def cylindrical_frame() -> OrthogonalFrame:
   s = coordinate_symbols()
   return OrthogonalFrame(
      name="cylindrical",
      coordinates=(s["r"], s["phi"], s["z"]),
      scale_factors=(sp.Integer(1), s["r"], sp.Integer(1)),
      cartesian_position=(s["r"] * sp.cos(s["phi"]), s["r"] * sp.sin(s["phi"]), s["z"]),
   )


def spherical_frame() -> OrthogonalFrame:
   s = coordinate_symbols()
   return OrthogonalFrame(
      name="spherical",
      coordinates=(s["r"], s["theta"], s["phi"]),
      scale_factors=(sp.Integer(1), s["r"], s["r"] * sp.sin(s["theta"])),
      cartesian_position=(s["r"] * sp.sin(s["theta"]) * sp.cos(s["phi"]),
                          s["r"] * sp.sin(s["theta"]) * sp.sin(s["phi"]),
                          s["r"] * sp.cos(s["theta"])),
   )


def orthogonal_frames() -> tuple[OrthogonalFrame, ...]:
   return (cartesian_frame(), cylindrical_frame(), spherical_frame())
