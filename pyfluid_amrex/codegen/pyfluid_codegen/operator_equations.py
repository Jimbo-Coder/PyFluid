from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ScalarKernel:
   name: str
   arguments: tuple[str, ...]
   expression: sp.Expr


def scalar_symbols() -> dict[str, sp.Symbol]:
   names = (
      "base center xm xp ym yp zm zp inv_dx2 inv_dy2 inv_dz2 "
      "cx cy cz coeff_dt lap left right coefficient inv_spacing neighbor_sum coeff_sum "
      "face_area volume spacing"
   )
   return {name: symbol for name, symbol in zip(names.split(), sp.symbols(names), strict=True)}


def laplacian_equation(dim: int) -> ScalarKernel:
   s = scalar_symbols()
   expression = (s["xm"] - 2 * s["center"] + s["xp"]) * s["inv_dx2"]
   if dim >= 2:
      expression += (s["ym"] - 2 * s["center"] + s["yp"]) * s["inv_dy2"]
   if dim == 3:
      expression += (s["zm"] - 2 * s["center"] + s["zp"]) * s["inv_dz2"]
   return ScalarKernel(
      name=f"laplacian_cartesian_{dim}d",
      arguments=("center", "xm", "xp", "ym", "yp", "zm", "zp", "inv_dx2", "inv_dy2", "inv_dz2"),
      expression=expression,
   )


def implicit_relaxed_equation(dim: int) -> ScalarKernel:
   s = scalar_symbols()
   numerator = s["base"] + s["cx"] * (s["xm"] + s["xp"])
   denom = 1 + 2 * s["cx"]
   if dim >= 2:
      numerator += s["cy"] * (s["ym"] + s["yp"])
      denom += 2 * s["cy"]
   if dim == 3:
      numerator += s["cz"] * (s["zm"] + s["zp"])
      denom += 2 * s["cz"]
   return ScalarKernel(
      name=f"implicit_relaxed_cartesian_{dim}d",
      arguments=("base", "xm", "xp", "ym", "yp", "zm", "zp", "cx", "cy", "cz"),
      expression=numerator / denom,
   )


def implicit_residual_equation(dim: int) -> ScalarKernel:
   s = scalar_symbols()
   expression = s["center"] - s["base"] + s["cx"] * (2 * s["center"] - s["xm"] - s["xp"])
   if dim >= 2:
      expression += s["cy"] * (2 * s["center"] - s["ym"] - s["yp"])
   if dim == 3:
      expression += s["cz"] * (2 * s["center"] - s["zm"] - s["zp"])
   return ScalarKernel(
      name=f"implicit_residual_cartesian_{dim}d",
      arguments=("base", "center", "xm", "xp", "ym", "yp", "zm", "zp", "cx", "cy", "cz"),
      expression=expression,
   )


def implicit_relaxed_variable_equation() -> ScalarKernel:
   s = scalar_symbols()
   return ScalarKernel(
      name="implicit_relaxed_variable_coeff",
      arguments=("base", "neighbor_sum", "coeff_sum"),
      expression=(s["base"] + s["neighbor_sum"]) / (1 + s["coeff_sum"]),
   )


def implicit_residual_variable_equation() -> ScalarKernel:
   s = scalar_symbols()
   return ScalarKernel(
      name="implicit_residual_variable_coeff",
      arguments=("base", "center", "neighbor_sum", "coeff_sum"),
      expression=s["center"] - s["base"] + s["coeff_sum"] * s["center"] - s["neighbor_sum"],
   )


def explicit_diffusion_update_equation() -> ScalarKernel:
   s = scalar_symbols()
   return ScalarKernel(
      name="explicit_diffusion_update",
      arguments=("center", "coeff_dt", "lap"),
      expression=s["center"] + s["coeff_dt"] * s["lap"],
   )


def diffusive_flux_equation() -> ScalarKernel:
   s = scalar_symbols()
   return ScalarKernel(
      name="diffusive_flux",
      arguments=("left", "right", "coefficient", "inv_spacing"),
      expression=-s["coefficient"] * (s["right"] - s["left"]) * s["inv_spacing"],
   )


def finite_volume_laplacian_face_weight_equation() -> ScalarKernel:
   s = scalar_symbols()
   return ScalarKernel(
      name="finite_volume_laplacian_face_weight",
      arguments=("face_area", "volume", "spacing"),
      expression=s["face_area"] / (s["volume"] * s["spacing"]),
   )


def scalar_operator_kernels() -> tuple[ScalarKernel, ...]:
   kernels: list[ScalarKernel] = [
      explicit_diffusion_update_equation(),
      diffusive_flux_equation(),
      finite_volume_laplacian_face_weight_equation(),
      implicit_relaxed_variable_equation(),
      implicit_residual_variable_equation(),
   ]
   for dim in (1, 2, 3):
      kernels.append(laplacian_equation(dim))
      kernels.append(implicit_relaxed_equation(dim))
      kernels.append(implicit_residual_equation(dim))
   return tuple(kernels)
