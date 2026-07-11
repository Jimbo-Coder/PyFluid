from __future__ import annotations

import sympy as sp

from .coordinates import cartesian_frame
from .discretization import CenteredDiffusionStencil
from .kernel_equations import ScalarKernel
from .tensor_algebra import OrthogonalOperators, ScalarField, Tensor, Vector


def scalar_symbols() -> dict[str, sp.Symbol]:
   names = (
      "base center xm xp ym yp zm zp inv_dx2 inv_dy2 inv_dz2 "
      "cx cy cz coeff_dt lap left right coefficient inv_spacing neighbor_sum coeff_sum "
      "face_area volume spacing"
   )
   return {name: symbol for name, symbol in zip(names.split(), sp.symbols(names), strict=True)}


def _cartesian_stencil(
   dim: int,
   symbols: dict[str, sp.Symbol],
   coefficients: tuple[str, str, str],
) -> CenteredDiffusionStencil:
   if dim not in (1, 2, 3):
      raise ValueError("dimensions must be 1, 2, or 3")
   neighbor_names = (("xm", "xp"), ("ym", "yp"), ("zm", "zp"))
   return CenteredDiffusionStencil(
      center=symbols["center"],
      neighbors=tuple(
         (symbols[lower], symbols[upper]) for lower, upper in neighbor_names[:dim]
      ),
      coefficients=tuple(symbols[name] for name in coefficients[:dim]),
   )


def laplacian_equation(dim: int) -> ScalarKernel:
   s = scalar_symbols()
   stencil = _cartesian_stencil(dim, s, ("inv_dx2", "inv_dy2", "inv_dz2"))
   active_second_derivatives = tuple(
      coefficient * (lower - 2 * s["center"] + upper)
      for (lower, upper), coefficient in zip(
         stencil.neighbors,
         stencil.coefficients,
         strict=True,
      )
   )
   second_derivatives = active_second_derivatives + (sp.Integer(0),) * (3 - dim)
   hessian = Tensor.from_rows(tuple(
      tuple(second_derivatives[row] if row == col else sp.Integer(0) for col in range(3))
      for row in range(3)
   ))
   field = ScalarField(s["center"], Vector.zero(3), hessian)
   frame = cartesian_frame()
   expression = OrthogonalOperators(frame.scale_factors, frame.coordinates).laplacian(field)
   return ScalarKernel(
      name=f"laplacian_cartesian_{dim}d",
      arguments=("center", "xm", "xp", "ym", "yp", "zm", "zp", "inv_dx2", "inv_dy2", "inv_dz2"),
      expression=expression,
   )


def implicit_relaxed_equation(dim: int) -> ScalarKernel:
   s = scalar_symbols()
   stencil = _cartesian_stencil(dim, s, ("cx", "cy", "cz"))
   return ScalarKernel(
      name=f"implicit_relaxed_cartesian_{dim}d",
      arguments=("base", "xm", "xp", "ym", "yp", "zm", "zp", "cx", "cy", "cz"),
      expression=stencil.implicit_relaxed(s["base"]),
   )


def implicit_residual_equation(dim: int) -> ScalarKernel:
   s = scalar_symbols()
   stencil = _cartesian_stencil(dim, s, ("cx", "cy", "cz"))
   return ScalarKernel(
      name=f"implicit_residual_cartesian_{dim}d",
      arguments=("base", "center", "xm", "xp", "ym", "yp", "zm", "zp", "cx", "cy", "cz"),
      expression=stencil.implicit_residual(s["base"]),
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
