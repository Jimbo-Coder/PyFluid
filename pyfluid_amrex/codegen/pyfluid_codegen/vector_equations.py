from __future__ import annotations

import sympy as sp

from .discretization import FiniteVolumeDivergence
from .kernel_equations import ScalarKernel
from .physical_system import NewtonianViscosity
from .tensor_algebra import (
   OrthogonalOperators,
   Tensor,
   Vector,
   VectorField,
   replace_coordinates,
)
from .coordinates import OrthogonalFrame, cartesian_frame, cylindrical_frame, spherical_frame


def vector_symbols() -> dict[str, sp.Symbol]:
   names = (
      "daz_dq2 day_dq3 dax_dq3 daz_dq1 day_dq1 dax_dq2 "
      "dax_dq1 day_dq2 daz_dq3 "
      "inv_h2h3 inv_h3h1 inv_h1h2 inv_volume_scale "
      "d_h3_a3_dq2 d_h2_a2_dq3 d_h1_a1_dq3 d_h3_a3_dq1 "
      "d_h2_a2_dq1 d_h1_a1_dq2 "
      "d_h2h3_a1_dq1 d_h3h1_a2_dq2 d_h1h2_a3_dq3 "
      "vx vy vz bx by bz eta j ideal h_tangent d_scaled_b denominator pressure floor "
      "face_b face_area face_flux edge_e edge_length dt "
      "ez_hi ez_lo ex_hi ex_lo ey_hi ey_lo "
      "mu q1 q2 q3 v1 v2 v3 "
      "dv1_dq1 dv1_dq2 dv1_dq3 dv2_dq1 dv2_dq2 dv2_dq3 dv3_dq1 dv3_dq2 dv3_dq3 "
      "t11 t12 t13 t21 t22 t23 t31 t32 t33"
   )
   return {name: symbol for name, symbol in zip(names.split(), sp.symbols(names), strict=True)}


def _pairwise_difference_kernels(
   kernel_specs: tuple[tuple[str, str, str, str | None], ...],
) -> tuple[ScalarKernel, ...]:
   s = vector_symbols()
   kernels: list[ScalarKernel] = []
   for name, plus_symbol, minus_symbol, scale_factor in kernel_specs:
      arguments: tuple[str, ...] = (plus_symbol, minus_symbol)
      expression = s[plus_symbol] - s[minus_symbol]
      if scale_factor is not None:
         arguments = (scale_factor, *arguments)
         expression = s[scale_factor] * expression
      kernels.append(
         ScalarKernel(
            name=name,
            arguments=arguments,
            expression=expression,
         )
      )
   return tuple(kernels)


def cartesian_curl_kernels() -> tuple[ScalarKernel, ...]:
   s = vector_symbols()
   derivatives = Tensor.from_rows(
      (
         (s["dax_dq1"], s["dax_dq2"], s["dax_dq3"]),
         (s["day_dq1"], s["day_dq2"], s["day_dq3"]),
         (s["daz_dq1"], s["daz_dq2"], s["daz_dq3"]),
      )
   )
   frame = cartesian_frame()
   field = VectorField(Vector.zero(3), derivatives)
   curl = OrthogonalOperators(frame.scale_factors, frame.coordinates).curl(field)
   specs = (
      ("curl_x_cartesian", ("daz_dq2", "day_dq3"), curl.component(0)),
      ("curl_y_cartesian", ("dax_dq3", "daz_dq1"), curl.component(1)),
      ("curl_z_cartesian", ("day_dq1", "dax_dq2"), curl.component(2)),
   )
   return tuple(ScalarKernel(name=name, arguments=arguments, expression=expression) for name, arguments, expression in specs)


def cartesian_divergence_kernel() -> ScalarKernel:
   s = vector_symbols()
   derivatives = Tensor.from_rows(
      (
         (s["dax_dq1"], sp.Integer(0), sp.Integer(0)),
         (sp.Integer(0), s["day_dq2"], sp.Integer(0)),
         (sp.Integer(0), sp.Integer(0), s["daz_dq3"]),
      )
   )
   frame = cartesian_frame()
   return ScalarKernel(
      name="divergence_cartesian",
      arguments=("dax_dq1", "day_dq2", "daz_dq3"),
      expression=OrthogonalOperators(frame.scale_factors, frame.coordinates).divergence(
         VectorField(Vector.zero(3), derivatives)
      ),
   )


def orthogonal_curl_kernels() -> tuple[ScalarKernel, ...]:
   return _pairwise_difference_kernels(
      (
         ("curl_orthogonal_1", "d_h3_a3_dq2", "d_h2_a2_dq3", "inv_h2h3"),
         ("curl_orthogonal_2", "d_h1_a1_dq3", "d_h3_a3_dq1", "inv_h3h1"),
         ("curl_orthogonal_3", "d_h2_a2_dq1", "d_h1_a1_dq2", "inv_h1h2"),
      )
   )


def orthogonal_divergence_kernel() -> ScalarKernel:
   s = vector_symbols()
   return ScalarKernel(
      name="divergence_orthogonal",
      arguments=("inv_volume_scale", "d_h2h3_a1_dq1", "d_h3h1_a2_dq2", "d_h1h2_a3_dq3"),
      expression=s["inv_volume_scale"]
      * (s["d_h2h3_a1_dq1"] + s["d_h3h1_a2_dq2"] + s["d_h1h2_a3_dq3"]),
   )


def _finite_volume_divergence_kernel(dimensions: int) -> ScalarKernel:
   symbols = (
      "flux_x_hi flux_x_lo flux_y_hi flux_y_lo flux_z_hi flux_z_lo volume"
   )
   q = {name: symbol for name, symbol in zip(symbols.split(), sp.symbols(symbols), strict=True)}
   axes = ("x", "y", "z")
   if dimensions not in (1, 2, 3):
      raise ValueError("dimensions must be 1, 2, or 3")
   arguments: list[str] = []
   face_fluxes: list[tuple[sp.Expr, sp.Expr]] = []
   for axis in axes[:dimensions]:
      arguments.append(f"flux_{axis}_hi")
      arguments.append(f"flux_{axis}_lo")
      face_fluxes.append((q[f"flux_{axis}_hi"], q[f"flux_{axis}_lo"]))
   arguments.append("volume")
   return ScalarKernel(
      name=f"finite_volume_divergence_{dimensions}d",
      arguments=tuple(arguments),
      expression=FiniteVolumeDivergence(tuple(face_fluxes), q["volume"]).value(),
   )


def _orthogonal_frame_runtime_coordinates(
   q1: sp.Symbol, q2: sp.Symbol, q3: sp.Symbol
) -> tuple[tuple[OrthogonalFrame, tuple[sp.Symbol, sp.Symbol, sp.Symbol]], ...]:
   runtime = (q1, q2, q3)
   return (
      (cartesian_frame(), runtime),
      (cylindrical_frame(), runtime),
      (spherical_frame(), runtime),
   )


def ideal_emf_kernels() -> tuple[ScalarKernel, ...]:
   s = vector_symbols()
   velocity = Vector(s["vx"], s["vy"], s["vz"])
   magnetic = Vector(s["bx"], s["by"], s["bz"])
   emf = magnetic.cross(velocity)
   return (
      ScalarKernel(
         name="ideal_emf_x",
         arguments=("vy", "vz", "by", "bz"),
         expression=emf.component(0),
      ),
      ScalarKernel(
         name="ideal_emf_y",
         arguments=("vz", "vx", "bz", "bx"),
         expression=emf.component(1),
      ),
      ScalarKernel(
         name="ideal_emf_z",
         arguments=("vx", "vy", "bx", "by"),
         expression=emf.component(2),
      ),
   )


def electromagnetic_scalar_kernels() -> tuple[ScalarKernel, ...]:
   s = vector_symbols()
   magnetic = Vector(s["bx"], s["by"], s["bz"])
   return (
      ScalarKernel(
         name="resistive_emf",
         arguments=("eta", "j"),
         expression=s["eta"] * s["j"],
      ),
      ScalarKernel(
         name="total_emf",
         arguments=("ideal", "eta", "j"),
         expression=s["ideal"] + s["eta"] * s["j"],
      ),
      ScalarKernel(
         name="resistive_emf_scaled_gradient",
         arguments=("eta", "h_tangent", "d_scaled_b", "denominator"),
         expression=s["eta"] * s["h_tangent"] * s["d_scaled_b"] / s["denominator"],
      ),
      ScalarKernel(
         name="magnetic_energy_density_from_components",
         arguments=("bx", "by", "bz"),
         expression=sp.Rational(1, 2) * magnetic.norm_squared(),
      ),
      ScalarKernel(
         name="plasma_beta_from_pressure",
         arguments=("pressure", "floor", "bx", "by", "bz"),
         expression=s["pressure"] / sp.Max(s["floor"], sp.Rational(1, 2) * magnetic.norm_squared()),
      ),
   )


def constrained_transport_kernels() -> tuple[ScalarKernel, ...]:
   s = vector_symbols()
   return (
      ScalarKernel(
         name="magnetic_face_flux",
         arguments=("face_b", "face_area"),
         expression=s["face_b"] * s["face_area"],
      ),
      ScalarKernel(
         name="magnetic_face_component",
         arguments=("face_flux", "face_area"),
         expression=s["face_flux"] / s["face_area"],
      ),
      ScalarKernel(
         name="emf_line_integral",
         arguments=("edge_e", "edge_length"),
         expression=s["edge_e"] * s["edge_length"],
      ),
      *_ct_flux_update_kernels(
         (
            ("ct_flux_update_2d_dir0", (("ez", -1),)),
            ("ct_flux_update_2d_dir1", (("ez", 1),)),
            ("ct_flux_update_3d_dir0", (("ez", -1), ("ey", 1))),
            ("ct_flux_update_3d_dir1", (("ex", -1), ("ez", 1))),
            ("ct_flux_update_3d_dir2", (("ey", -1), ("ex", 1))),
         )
      ),
      _finite_volume_divergence_kernel(1),
      _finite_volume_divergence_kernel(2),
      _finite_volume_divergence_kernel(3),
   )


def _ct_flux_update_kernels(
   update_specs: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
) -> tuple[ScalarKernel, ...]:
   s = vector_symbols()
   kernels: list[ScalarKernel] = []
   for name, edge_terms in update_specs:
      arguments: list[str] = ["face_flux", "dt"]
      correction = sp.Integer(0)
      for edge_axis, coefficient in edge_terms:
         hi = f"{edge_axis}_hi"
         lo = f"{edge_axis}_lo"
         arguments.extend((hi, lo))
         correction += sp.Integer(coefficient) * (s[hi] - s[lo])
      kernels.append(
         ScalarKernel(
            name=name,
            arguments=tuple(arguments),
            expression=s["face_flux"] + s["dt"] * correction,
         )
      )
   return tuple(kernels)


def _viscous_stress_components(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   velocities: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   derivatives: tuple[tuple[sp.Symbol, sp.Symbol, sp.Symbol], ...],
   mu: sp.Symbol,
) -> tuple[tuple[sp.Expr, sp.Expr, sp.Expr], ...]:
   velocity_field = VectorField(Vector(*velocities), Tensor.from_rows(derivatives))
   operators = OrthogonalOperators(scale_factors, coordinates)
   return NewtonianViscosity(mu).stress(velocity_field, operators).as_rows()


def viscous_stress_kernels() -> tuple[ScalarKernel, ...]:
   """Newtonian shear stress in orthonormal components for orthogonal coordinates.

   The high-level definition is tau_ij = 2 mu (e_ij - delta_ij div(v)/3),
   with e_ij the symmetric covariant velocity gradient. Coordinates enter only
   through the orthogonal scale factors h_i.
   """
   s = vector_symbols()
   velocities = (s["v1"], s["v2"], s["v3"])
   derivatives = (
      (s["dv1_dq1"], s["dv1_dq2"], s["dv1_dq3"]),
      (s["dv2_dq1"], s["dv2_dq2"], s["dv2_dq3"]),
      (s["dv3_dq1"], s["dv3_dq2"], s["dv3_dq3"]),
   )
   runtime_coordinates = (s["q1"], s["q2"], s["q3"])
   kernels: list[ScalarKernel] = []
   for frame, frame_coordinates in _orthogonal_frame_runtime_coordinates(*runtime_coordinates):
      scale_factors = replace_coordinates(
         frame.scale_factors,
         frame.coordinates,
         frame_coordinates,
      )
      stress = _viscous_stress_components(
         scale_factors,
         frame_coordinates,
         velocities,
         derivatives,
         s["mu"],
      )
      for normal in range(3):
         for component in range(3):
            kernels.append(
               ScalarKernel(
                  name=f"{frame.name}_viscous_stress_{normal + 1}{component + 1}",
                  arguments=(
                     "mu",
                     "q1",
                     "q2",
                     "q3",
                     "v1",
                     "v2",
                     "v3",
                     "dv1_dq1",
                     "dv1_dq2",
                     "dv1_dq3",
                     "dv2_dq1",
                     "dv2_dq2",
                     "dv2_dq3",
                     "dv3_dq1",
                     "dv3_dq2",
                     "dv3_dq3",
                  ),
                  expression=stress[normal][component],
               )
            )
   return tuple(kernels)


def _tensor_divergence_connection_sources(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   tensor: Tensor,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   """Connection terms missing from FV divergence of orthonormal tensor fluxes.

   The finite-volume divergence supplies
   1/(h1 h2 h3) d_qj[(h1 h2 h3 / h_j) T_ji].  Physical vector components
   also need the basis-derivative terms below.
   """
   return OrthogonalOperators(scale_factors, coordinates).tensor_divergence_connection(tensor).as_tuple()


def tensor_divergence_connection_kernels() -> tuple[ScalarKernel, ...]:
   s = vector_symbols()
   tensor = Tensor(
      Vector(s["t11"], s["t12"], s["t13"]),
      Vector(s["t21"], s["t22"], s["t23"]),
      Vector(s["t31"], s["t32"], s["t33"]),
   )
   runtime_coordinates = (s["q1"], s["q2"], s["q3"])
   kernels: list[ScalarKernel] = []
   for frame, frame_coordinates in _orthogonal_frame_runtime_coordinates(*runtime_coordinates):
      scale_factors = replace_coordinates(
         frame.scale_factors,
         frame.coordinates,
         frame_coordinates,
      )
      sources = _tensor_divergence_connection_sources(
         scale_factors,
         frame_coordinates,
         tensor,
      )
      for component, expression in enumerate(sources):
         kernels.append(
            ScalarKernel(
               name=f"{frame.name}_tensor_divergence_connection_{component + 1}",
               arguments=("q1", "q2", "q3", "t11", "t12", "t13", "t21", "t22", "t23", "t31", "t32", "t33"),
               expression=expression,
            )
         )
   return tuple(kernels)


def vector_operator_kernels() -> tuple[ScalarKernel, ...]:
   return (
      *cartesian_curl_kernels(),
      cartesian_divergence_kernel(),
      *orthogonal_curl_kernels(),
      orthogonal_divergence_kernel(),
      *viscous_stress_kernels(),
      *tensor_divergence_connection_kernels(),
      *ideal_emf_kernels(),
      *electromagnetic_scalar_kernels(),
      *constrained_transport_kernels(),
   )
