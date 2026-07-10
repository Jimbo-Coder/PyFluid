from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .coordinates import (
   OrthogonalFrame,
   cartesian_frame,
   cylindrical_frame,
   spherical_frame,
)
from .operator_equations import ScalarKernel
from .tensor_algebra import (
   Vector3,
   _as_vector,
   orthogonal_mhd_stress_curvature_sources,
   orthogonal_velocity_curvature_sources,
)


def metric_symbols() -> dict[str, sp.Symbol]:
   names = (
      "dx dy dz r theta phi z rlo rhi theta_lo theta_hi z_width phi_width "
      "rho vr vtheta vphi vz br btheta bphi bz inv_r cot_theta"
   )
   return {name: symbol for name, symbol in zip(names.split(), sp.symbols(names), strict=True)}


@dataclass(frozen=True)
class FiniteVolumeSpec:
   name: str
   frame: OrthogonalFrame
   active_axes: tuple[int, ...]
   face_names: tuple[str, ...]


def _axis_integral(expr: sp.Expr, coordinate: sp.Symbol) -> sp.Expr:
   s = metric_symbols()
   if coordinate == s["r"]:
      return sp.integrate(expr, (coordinate, s["rlo"], s["rhi"]))
   if coordinate == s["theta"]:
      return sp.integrate(expr, (coordinate, s["theta_lo"], s["theta_hi"]))
   if coordinate == s["z"]:
      return sp.integrate(expr, (coordinate, 0, s["z_width"]))
   if coordinate == s["phi"]:
      return sp.integrate(expr, (coordinate, 0, s["phi_width"]))
   raise ValueError(f"unsupported coordinate {coordinate}")


def _integrate_active_axes(expr: sp.Expr, frame: OrthogonalFrame, axes: tuple[int, ...]) -> sp.Expr:
   result = expr
   for axis in axes:
      result = _axis_integral(result, frame.coordinates[axis])
   return sp.simplify(result)


def _face_measure_expression(spec: FiniteVolumeSpec, active_axis_index: int) -> sp.Expr:
   axis = spec.active_axes[active_axis_index]
   expr = spec.frame.volume_scale / spec.frame.scale_factors[axis]
   axes_to_integrate = tuple(other for other in spec.active_axes if other != axis)
   expr = _integrate_active_axes(expr, spec.frame, axes_to_integrate)
   return sp.simplify(expr)


def _arguments_for(expr: sp.Expr) -> tuple[str, ...]:
   preferred = ("rlo", "rhi", "r", "theta_lo", "theta_hi", "theta", "z_width", "phi_width")
   symbols = {str(symbol) for symbol in expr.free_symbols}
   return tuple(name for name in preferred if name in symbols)


def _finite_volume_specs() -> tuple[FiniteVolumeSpec, ...]:
   return (
      FiniteVolumeSpec(
         name="cylindrical_rz",
         frame=cylindrical_frame(),
         active_axes=(0, 2),
         face_names=("radial", "axial"),
      ),
      FiniteVolumeSpec(
         name="spherical_rtheta",
         frame=spherical_frame(),
         active_axes=(0, 1),
         face_names=("radial", "theta"),
      ),
      FiniteVolumeSpec(
         name="spherical_rphi",
         frame=spherical_frame(),
         active_axes=(0, 2),
         face_names=("radial", "azimuthal"),
      ),
      FiniteVolumeSpec(
         name="cylindrical_phiz",
         frame=cylindrical_frame(),
         active_axes=(1, 2),
         face_names=("azimuthal", "axial"),
      ),
      FiniteVolumeSpec(
         name="spherical_thetaphi",
         frame=spherical_frame(),
         active_axes=(1, 2),
         face_names=("theta", "azimuthal"),
      ),
      FiniteVolumeSpec(
         name="cylindrical_rphiz",
         frame=cylindrical_frame(),
         active_axes=(0, 1, 2),
         face_names=("radial", "azimuthal", "axial"),
      ),
      FiniteVolumeSpec(
         name="spherical_rthetaphi",
         frame=spherical_frame(),
         active_axes=(0, 1, 2),
         face_names=("radial", "theta", "azimuthal"),
      ),
   )


def _orthonormal_velocity_curvature_sources(
   frame: OrthogonalFrame,
   density: sp.Expr,
   velocities: Vector3,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   return tuple(
      sp.trigsimp(sp.simplify(source))
      for source in orthogonal_velocity_curvature_sources(
         frame.scale_factors,
         frame.coordinates,
         density,
         velocities,
      )
   )


def _orthogonal_mhd_stress_curvature_sources(
   frame: OrthogonalFrame,
   density: sp.Expr,
   velocities: Vector3,
   magnetic: Vector3,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   return tuple(
      sp.trigsimp(sp.simplify(source))
      for source in orthogonal_mhd_stress_curvature_sources(
         frame.scale_factors,
         frame.coordinates,
         density,
         velocities,
         magnetic,
      )
   )


def _lower_curvature_source(expr: sp.Expr) -> sp.Expr:
   s = metric_symbols()
   expr = sp.trigsimp(sp.simplify(expr))
   expr = expr.subs(1 / sp.tan(s["theta"]), s["cot_theta"])
   expr = expr.subs(1 / s["r"], s["inv_r"])
   return sp.simplify(expr)


def _curvature_kernels(
   frame: OrthogonalFrame,
   kernel_prefix: str,
   component_names: tuple[str, str, str],
   expressions: tuple[sp.Expr, sp.Expr, sp.Expr],
   arguments: tuple[str, ...],
) -> tuple[ScalarKernel, ...]:
   return tuple(
      ScalarKernel(
         name=f"{frame.name}_{kernel_prefix}_{component_name}",
         arguments=arguments,
         expression=_lower_curvature_source(expression),
      )
      for component_name, expression in zip(component_names, expressions, strict=True)
   )


def velocity_curvature_source_kernels() -> tuple[ScalarKernel, ...]:
   s = metric_symbols()
   cyl_frame = cylindrical_frame()
   sph_frame = spherical_frame()
   cylindrical_frame_sources = _orthonormal_velocity_curvature_sources(
      cyl_frame,
      s["rho"],
      _as_vector((s["vr"], s["vphi"], s["vz"])),
   )
   spherical_frame_sources = _orthonormal_velocity_curvature_sources(
      sph_frame,
      s["rho"],
      _as_vector((s["vr"], s["vtheta"], s["vphi"])),
   )
   component_names: tuple[str, str, str] = ("mom1", "mom2", "mom3")
   return (
      *_curvature_kernels(
         cyl_frame,
         "momentum_curvature_source",
         component_names,
         cylindrical_frame_sources,
         ("rho", "vr", "vphi", "vz", "inv_r"),
      ),
      *_curvature_kernels(
         sph_frame,
         "momentum_curvature_source",
         component_names,
         spherical_frame_sources,
         ("rho", "vr", "vtheta", "vphi", "inv_r", "cot_theta"),
      ),
   )


def mhd_stress_curvature_source_kernels() -> tuple[ScalarKernel, ...]:
   s = metric_symbols()
   cyl_frame = cylindrical_frame()
   sph_frame = spherical_frame()
   vr = s["vr"]
   vphi = s["vphi"]
   vz = s["vz"]
   br = s["br"]
   bphi = s["bphi"]
   bz = s["bz"]
   v_cyl = _as_vector((vr, vphi, vz))
   b_cyl = _as_vector((br, bphi, bz))
   cylindrical_stress_sources = _orthogonal_mhd_stress_curvature_sources(
      cyl_frame, s["rho"], v_cyl, b_cyl
   )
   v_sph = _as_vector((s["vr"], s["vtheta"], s["vphi"]))
   b_sph = _as_vector((s["br"], s["btheta"], s["bphi"]))
   spherical_stress_sources = _orthogonal_mhd_stress_curvature_sources(
      sph_frame, s["rho"], v_sph, b_sph
   )
   component_names: tuple[str, str, str] = ("mom1", "mom2", "mom3")
   return (
      *_curvature_kernels(
         cyl_frame,
         "mhd_stress_curvature_source",
         component_names,
         cylindrical_stress_sources,
         ("rho", "vr", "vphi", "vz", "br", "bphi", "bz", "inv_r"),
      ),
      *_curvature_kernels(
         sph_frame,
         "mhd_stress_curvature_source",
         component_names,
         spherical_stress_sources,
         ("rho", "vr", "vtheta", "vphi", "br", "btheta", "bphi", "inv_r", "cot_theta"),
      ),
   )


def cartesian_measure_kernels() -> tuple[ScalarKernel, ...]:
   s = metric_symbols()
   lengths = ("dx", "dy", "dz")
   length_exprs = (s["dx"], s["dy"], s["dz"])
   axes = ("x", "y", "z")
   kernels: list[ScalarKernel] = [
      ScalarKernel(
         name="cartesian_cell_volume",
         arguments=lengths,
         expression=length_exprs[0] * length_exprs[1] * length_exprs[2],
      ),
   ]
   for axis_index, axis in enumerate(axes):
      face_axes = tuple(index for index in range(len(lengths)) if index != axis_index)
      kernels.append(
         ScalarKernel(
            name=f"cartesian_face_area_{axis}",
            arguments=tuple(lengths[index] for index in face_axes),
            expression=sp.prod(length_exprs[index] for index in face_axes),
         )
      )
      kernels.append(
         ScalarKernel(
            name=f"cartesian_edge_length_{axis}",
            arguments=(lengths[axis_index],),
            expression=length_exprs[axis_index],
         )
      )
   return tuple(kernels)


def coordinate_scale_factor_kernels() -> tuple[ScalarKernel, ...]:
   return (
      *_frame_scale_kernels(cartesian_frame(), ("x", "y", "z")),
      *_frame_scale_kernels(cylindrical_frame(), ("r", "phi", "z")),
      *_frame_scale_kernels(spherical_frame(), ("r", "theta", "phi")),
   )


def _frame_scale_kernels(
   frame: OrthogonalFrame,
   axis_names: tuple[str, str, str],
) -> tuple[ScalarKernel, ...]:
   kernels: list[ScalarKernel] = []
   for axis_name, scale_factor in zip(axis_names, frame.scale_factors, strict=True):
      kernels.append(
         ScalarKernel(
            name=f"{frame.name}_scale_{axis_name}",
            arguments=("r", "theta", "phi"),
            expression=sp.simplify(scale_factor),
         )
      )
   kernels.append(
      ScalarKernel(
         name=f"{frame.name}_volume_scale",
         arguments=("r", "theta", "phi"),
         expression=sp.simplify(frame.volume_scale),
      )
   )
   return tuple(kernels)


def spherical_radial_measure_kernels() -> tuple[ScalarKernel, ...]:
   return _radial_shell_measure_kernels("spherical", shell_volume_power=3)


def cylindrical_radial_measure_kernels() -> tuple[ScalarKernel, ...]:
   return _radial_shell_measure_kernels("cylindrical", shell_volume_power=2)


def _radial_shell_measure_kernels(
   frame_name: str,
   shell_volume_power: int,
) -> tuple[ScalarKernel, ...]:
   s = metric_symbols()
   surface_power = shell_volume_power - 1
   shell_volume_scale = (s["rhi"] ** shell_volume_power - s["rlo"] ** shell_volume_power) / sp.Integer(
      shell_volume_power
   )
   face_area_scale_delta = s["rhi"] ** surface_power - s["rlo"] ** surface_power
   return (
      ScalarKernel(
         name=f"{frame_name}_radial_shell_volume_scale",
         arguments=("rlo", "rhi"),
         expression=shell_volume_scale,
      ),
      ScalarKernel(
         name=f"{frame_name}_radial_face_area_scale",
         arguments=("r",),
         expression=s["r"] ** surface_power,
      ),
      ScalarKernel(
         name=f"{frame_name}_radial_pressure_source_factor",
         arguments=("rlo", "rhi"),
         expression=face_area_scale_delta / shell_volume_scale,
      ),
   )


def curvilinear_finite_volume_kernels() -> tuple[ScalarKernel, ...]:
   s = metric_symbols()
   kernels: list[ScalarKernel] = [
      ScalarKernel(
         name="spherical_polar_weight",
         arguments=("theta_lo", "theta_hi"),
         expression=_axis_integral(sp.sin(s["theta"]), s["theta"]),
      ),
   ]
   pressure_coordinate_bounds = {
      s["r"]: (s["rlo"], s["rhi"]),
      s["theta"]: (s["theta_lo"], s["theta_hi"]),
   }
   for spec in _finite_volume_specs():
      cell = _integrate_active_axes(spec.frame.volume_scale, spec.frame, spec.active_axes)
      kernels.append(
         ScalarKernel(
            name=f"{spec.name}_cell_volume_scale",
            arguments=_arguments_for(cell),
            expression=cell,
         )
      )
      for active_axis_index, face_name in enumerate(spec.face_names):
         face = _face_measure_expression(spec, active_axis_index)
         kernels.append(
            ScalarKernel(
               name=f"{spec.name}_{face_name}_face_area_scale",
               arguments=_arguments_for(face),
               expression=face,
            )
         )
         axis = spec.active_axes[active_axis_index]
         coordinate = spec.frame.coordinates[axis]
         if coordinate in pressure_coordinate_bounds:
            lo, hi = pressure_coordinate_bounds[coordinate]
            pressure_source = sp.simplify((face.subs(coordinate, hi) - face.subs(coordinate, lo)) / cell)
            kernel_name = f"{spec.name}_{face_name}_pressure_source_factor"
            kernels.append(
               ScalarKernel(
                  name=kernel_name,
                  arguments=_arguments_for(pressure_source),
                  expression=pressure_source,
               )
            )
   return tuple(kernels)


def edge_length_kernels() -> tuple[ScalarKernel, ...]:
   s = metric_symbols()
   specs = (
      (
         "cylindrical_rphiz_radial_edge_length_scale",
         ("rlo", "rhi"),
         s["rhi"] - s["rlo"],
      ),
      (
         "cylindrical_rphiz_axial_edge_length_scale",
         ("z_width",),
         s["z_width"],
      ),
      (
         "cylindrical_rphiz_azimuthal_edge_length_scale",
         ("r", "phi_width"),
         s["r"] * s["phi_width"],
      ),
      (
         "spherical_rthetaphi_radial_edge_length_scale",
         ("rlo", "rhi"),
         s["rhi"] - s["rlo"],
      ),
      (
         "spherical_rthetaphi_theta_edge_length_scale",
         ("r", "theta_lo", "theta_hi"),
         s["r"] * (s["theta_hi"] - s["theta_lo"]),
      ),
      (
         "spherical_rthetaphi_azimuthal_edge_length_scale",
         ("r", "theta", "phi_width"),
         s["r"] * sp.sin(s["theta"]) * s["phi_width"],
      ),
   )
   return tuple(
      ScalarKernel(
         name=name,
         arguments=arguments,
         expression=expression,
      )
      for name, arguments, expression in specs
   )


def metric_kernels() -> tuple[ScalarKernel, ...]:
   return (
      *cartesian_measure_kernels(),
      *coordinate_scale_factor_kernels(),
      *cylindrical_radial_measure_kernels(),
      *spherical_radial_measure_kernels(),
      *curvilinear_finite_volume_kernels(),
      *edge_length_kernels(),
      *velocity_curvature_source_kernels(),
      *mhd_stress_curvature_source_kernels(),
   )
