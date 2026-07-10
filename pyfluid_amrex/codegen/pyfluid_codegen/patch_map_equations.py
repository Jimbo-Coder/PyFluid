from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class PatchMapKernel:
   name: str
   return_type: str
   arguments: tuple[tuple[str, str], ...]
   expressions: tuple[sp.Expr, ...]


def patch_map_symbols() -> dict[str, sp.Symbol]:
   names = "q1 q2 q3 g1 g2 g3 o1 o2 o3 primary_sign radial_inner radial_outer"
   return {name: symbol for name, symbol in zip(names.split(), sp.symbols(names), strict=True)}


def _point_kernel(name: str, expressions: tuple[sp.Expr, sp.Expr, sp.Expr], args: tuple[str, ...]) -> PatchMapKernel:
   return PatchMapKernel(
      name=name,
      return_type="PatchPoint",
      arguments=tuple(("amrex::Real", arg) for arg in args),
      expressions=expressions,
   )


def _jacobian_kernel(name: str, expressions: tuple[sp.Expr, ...], args: tuple[str, ...]) -> PatchMapKernel:
   return PatchMapKernel(
      name=name,
      return_type="PatchJacobian",
      arguments=tuple(("amrex::Real", arg) for arg in args),
      expressions=expressions,
   )


def _jacobian(expressions: tuple[sp.Expr, sp.Expr, sp.Expr], coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol]) -> tuple[sp.Expr, ...]:
   derivatives: list[sp.Expr] = []
   for expr in expressions:
      for coordinate in coordinates:
         derivative = sp.diff(expr, coordinate)
         for angle in coordinates:
            derivative = derivative.subs(sp.cos(angle) ** -2, 1 + sp.tan(angle) ** 2)
         derivatives.append(sp.simplify(derivative))
   return tuple(derivatives)


def _shell_primary(tangents: tuple[sp.Expr, ...], radius: sp.Symbol) -> sp.Expr:
   angular_norm = sp.sqrt(1 + sum(tangent**2 for tangent in tangents))
   return sp.simplify(radius / angular_norm)


def _thornburg_shell_global(axis: int) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   s = patch_map_symbols()
   tangent_a = sp.tan(s["q1"])
   tangent_b = sp.tan(s["q2"])
   angular_norm = sp.sqrt(1 + tangent_a**2 + tangent_b**2)
   radial_fraction = (s["q3"] - s["radial_inner"]) / (s["radial_outer"] - s["radial_inner"])
   blend = sp.simplify(1 - radial_fraction + radial_fraction / angular_norm)
   primary = s["primary_sign"] * s["q3"] * blend
   if axis == 0:
      centered = (primary, primary * tangent_b, primary * tangent_a)
   elif axis == 1:
      centered = (primary * tangent_b, primary, primary * tangent_a)
   else:
      centered = (primary * tangent_b, primary * tangent_a, primary)
   return tuple(sp.simplify(origin + component) for origin, component in zip((s["o1"], s["o2"], s["o3"]), centered, strict=True))


def _thornburg_shell_local(axis: int) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   s = patch_map_symbols()
   centered = (s["g1"] - s["o1"], s["g2"] - s["o2"], s["g3"] - s["o3"])
   if axis == 0:
      tangent_a = centered[2] / centered[0]
      tangent_b = centered[1] / centered[0]
   elif axis == 1:
      tangent_a = centered[2] / centered[1]
      tangent_b = centered[0] / centered[1]
   else:
      tangent_a = centered[1] / centered[2]
      tangent_b = centered[0] / centered[2]
   physical_radius = sp.sqrt(centered[0] ** 2 + centered[1] ** 2 + centered[2] ** 2)
   angular_norm = sp.sqrt(1 + tangent_a**2 + tangent_b**2)
   radial_denominator = s["radial_outer"] - s["radial_inner"]
   quadratic = (1 - angular_norm) / radial_denominator
   linear = angular_norm - quadratic * s["radial_inner"]
   radial_coordinate = sp.simplify(
      2 * physical_radius
      / (linear + sp.sqrt(linear**2 + 4 * quadratic * physical_radius))
   )
   return (sp.atan(tangent_a), sp.atan(tangent_b), radial_coordinate)


def _cylindrical_shell_global(axis: int) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   s = patch_map_symbols()
   tangent_a = sp.tan(s["q1"])
   angular_norm = sp.sqrt(1 + tangent_a**2)
   radial_fraction = (s["q3"] - s["radial_inner"]) / (s["radial_outer"] - s["radial_inner"])
   blend = sp.simplify(1 - radial_fraction + radial_fraction / angular_norm)
   primary = s["primary_sign"] * s["q3"] * blend
   if axis == 0:
      centered = (primary, primary * tangent_a, s["q2"])
   else:
      centered = (primary * tangent_a, primary, s["q2"])
   return tuple(sp.simplify(origin + component) for origin, component in zip((s["o1"], s["o2"], s["o3"]), centered, strict=True))


def _cylindrical_shell_local(axis: int) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   s = patch_map_symbols()
   centered = (s["g1"] - s["o1"], s["g2"] - s["o2"], s["g3"] - s["o3"])
   if axis == 0:
      tangent_a = centered[1] / centered[0]
   else:
      tangent_a = centered[0] / centered[1]
   physical_radius = sp.sqrt(centered[0] ** 2 + centered[1] ** 2)
   angular_norm = sp.sqrt(1 + tangent_a**2)
   radial_denominator = s["radial_outer"] - s["radial_inner"]
   quadratic = (1 - angular_norm) / radial_denominator
   linear = angular_norm - quadratic * s["radial_inner"]
   radial_coordinate = sp.simplify(
      2 * physical_radius
      / (linear + sp.sqrt(linear**2 + 4 * quadratic * physical_radius))
   )
   return (sp.atan(tangent_a), centered[2], radial_coordinate)


def patch_map_kernels() -> tuple[PatchMapKernel, ...]:
   s = patch_map_symbols()
   kernels: list[PatchMapKernel] = [
      _point_kernel(
         "patch_identity_local_to_global",
         (s["o1"] + s["q1"], s["o2"] + s["q2"], s["o3"] + s["q3"]),
         ("q1", "q2", "q3", "o1", "o2", "o3"),
      ),
      _point_kernel(
         "patch_identity_global_to_local",
         (s["g1"] - s["o1"], s["g2"] - s["o2"], s["g3"] - s["o3"]),
         ("g1", "g2", "g3", "o1", "o2", "o3"),
      ),
      _jacobian_kernel(
         "patch_identity_jacobian",
         (sp.Integer(1), sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(1), sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(1)),
         (),
      ),
   ]
   for axis in range(3):
      global_expr = _thornburg_shell_global(axis)
      centered_global_expr = tuple(
         sp.simplify(expr - origin)
         for expr, origin in zip(global_expr, (s["o1"], s["o2"], s["o3"]), strict=True)
      )
      kernels.append(
         _point_kernel(
            f"thornburg_shell_local_to_global_axis{axis}",
            global_expr,
            (
               "q1",
               "q2",
               "q3",
               "o1",
               "o2",
               "o3",
               "primary_sign",
               "radial_inner",
               "radial_outer",
            ),
         )
      )
      kernels.append(
         _point_kernel(
            f"thornburg_shell_global_to_local_axis{axis}",
            _thornburg_shell_local(axis),
            (
               "g1",
               "g2",
               "g3",
               "o1",
               "o2",
               "o3",
               "primary_sign",
               "radial_inner",
               "radial_outer",
            ),
         )
      )
      kernels.append(
         _jacobian_kernel(
            f"thornburg_shell_jacobian_axis{axis}",
            _jacobian(centered_global_expr, (s["q1"], s["q2"], s["q3"])),
            ("q1", "q2", "q3", "primary_sign", "radial_inner", "radial_outer"),
         )
      )
   for axis in range(2):
      global_expr = _cylindrical_shell_global(axis)
      centered_global_expr = tuple(
         sp.simplify(expr - origin)
         for expr, origin in zip(global_expr, (s["o1"], s["o2"], s["o3"]), strict=True)
      )
      kernels.append(
         _point_kernel(
            f"cylindrical_shell_local_to_global_axis{axis}",
            global_expr,
            (
               "q1",
               "q2",
               "q3",
               "o1",
               "o2",
               "o3",
               "primary_sign",
               "radial_inner",
               "radial_outer",
            ),
         )
      )
      kernels.append(
         _point_kernel(
            f"cylindrical_shell_global_to_local_axis{axis}",
            _cylindrical_shell_local(axis),
            (
               "g1",
               "g2",
               "g3",
               "o1",
               "o2",
               "o3",
               "primary_sign",
               "radial_inner",
               "radial_outer",
            ),
         )
      )
      kernels.append(
         _jacobian_kernel(
            f"cylindrical_shell_jacobian_axis{axis}",
            _jacobian(centered_global_expr, (s["q1"], s["q2"], s["q3"])),
            ("q1", "q2", "q3", "primary_sign", "radial_inner", "radial_outer"),
         )
      )
   return tuple(kernels)
