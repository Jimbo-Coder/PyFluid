from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .kernel_equations import ScalarKernel, StructKernel
from .metric_differential import (
   inverse_metric_derivatives as _inverse_metric_derivatives,
   metric_christoffel as _christoffel,
   metric_ricci_tensor as _ricci_tensor,
)
from .tensor_algebra import symmetric_tensor3_symbols


DIMENSION = 3
SYMMETRIC_COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _symmetric_matrix(prefix: str) -> sp.Matrix:
   return symmetric_tensor3_symbols(prefix)


def _symmetric_derivatives(prefix: str) -> list[sp.Matrix]:
   return [
      sp.Matrix(
         DIMENSION,
         DIMENSION,
         lambda row, column: sp.Symbol(
            f"{prefix}[{direction}][{SYMMETRIC_COMPONENTS.index(tuple(sorted((row, column))))}]"
         ),
      )
      for direction in range(DIMENSION)
   ]


def _symmetric_second_derivatives(prefix: str) -> list[list[sp.Matrix]]:
   return [
      [
         sp.Matrix(
            DIMENSION,
            DIMENSION,
            lambda row, column: sp.Symbol(
               f"{prefix}[{first}][{second}]"
               f"[{SYMMETRIC_COMPONENTS.index(tuple(sorted((row, column))))}]"
            ),
         )
         for second in range(DIMENSION)
      ]
      for first in range(DIMENSION)
   ]


def _symmetric_outputs(prefix: str) -> tuple[str, ...]:
   return tuple(f"{prefix}{first + 1}{second + 1}" for first, second in SYMMETRIC_COMPONENTS)


def _symmetric_expressions(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
   return tuple(matrix[first, second] for first, second in SYMMETRIC_COMPONENTS)


def _inverse_symmetric_3x3(matrix: sp.Matrix) -> tuple[sp.Expr, sp.Matrix]:
   a, b, c = matrix[0, 0], matrix[0, 1], matrix[0, 2]
   d, e, f = matrix[1, 1], matrix[1, 2], matrix[2, 2]
   determinant = a * (d * f - e**2) - b * (b * f - c * e) + c * (b * e - c * d)
   adjugate = sp.Matrix((
      (d * f - e**2, c * e - b * f, b * e - c * d),
      (c * e - b * f, a * f - c**2, b * c - a * e),
      (b * e - c * d, b * c - a * e, a * d - b**2),
   ))
   return determinant, adjugate / determinant


@dataclass(frozen=True)
class Z4cPhysicalGeometryExpressions:
   chi: sp.Expr
   gamma_tilde: sp.Matrix
   inverse_gamma_tilde: sp.Matrix
   gamma: sp.Matrix
   inverse_gamma: sp.Matrix
   d_gamma: tuple[sp.Matrix, ...]
   dd_gamma: tuple[tuple[sp.Matrix, ...], ...]
   extrinsic_curvature: sp.Matrix
   d_extrinsic_curvature: tuple[sp.Matrix, ...]
   trace_extrinsic_curvature: sp.Expr
   d_trace_extrinsic_curvature: tuple[sp.Expr, ...]


def z4c_physical_geometry_expressions() -> Z4cPhysicalGeometryExpressions:
   """Return the physical 3-geometry once for evolution and diagnostics codegen."""
   chi = sp.Symbol("state.chi", positive=True)
   gamma_tilde = _state_metric()
   _, inverse_gamma_tilde = _inverse_symmetric_3x3(gamma_tilde)
   d_chi = tuple(sp.Symbol(f"derivatives.d_chi[{direction}]") for direction in range(DIMENSION))
   dd_chi = tuple(
      tuple(sp.Symbol(f"derivatives.dd_chi[{first}][{second}]") for second in range(DIMENSION))
      for first in range(DIMENSION)
   )
   d_gamma_tilde = tuple(_symmetric_derivatives("derivatives.d_gamma_tilde"))
   dd_gamma_tilde = tuple(
      tuple(row) for row in _symmetric_second_derivatives("derivatives.dd_gamma_tilde")
   )
   gamma = gamma_tilde / chi
   inverse_gamma = chi * inverse_gamma_tilde
   d_gamma = tuple(
      d_gamma_tilde[direction] / chi
      - gamma_tilde * d_chi[direction] / chi**2
      for direction in range(DIMENSION)
   )
   dd_gamma = tuple(
      tuple(
         dd_gamma_tilde[first][second] / chi
         - d_gamma_tilde[first] * d_chi[second] / chi**2
         - d_gamma_tilde[second] * d_chi[first] / chi**2
         - gamma_tilde * dd_chi[first][second] / chi**2
         + 2 * gamma_tilde * d_chi[first] * d_chi[second] / chi**3
         for second in range(DIMENSION)
      )
      for first in range(DIMENSION)
   )
   a_tilde = _symmetric_matrix("state.a_tilde")
   d_a_tilde = tuple(_symmetric_derivatives("derivatives.d_a_tilde"))
   trace_extrinsic_curvature = sp.Symbol("state.k_hat") + 2 * sp.Symbol("state.theta")
   d_trace_extrinsic_curvature = tuple(
      sp.Symbol(f"derivatives.d_k_hat[{direction}]")
      + 2 * sp.Symbol(f"derivatives.d_theta[{direction}]")
      for direction in range(DIMENSION)
   )
   extrinsic_curvature = a_tilde / chi + gamma * trace_extrinsic_curvature / 3
   d_extrinsic_curvature = tuple(
      d_a_tilde[direction] / chi
      - a_tilde * d_chi[direction] / chi**2
      + d_gamma[direction] * trace_extrinsic_curvature / 3
      + gamma * d_trace_extrinsic_curvature[direction] / 3
      for direction in range(DIMENSION)
   )
   return Z4cPhysicalGeometryExpressions(
      chi=chi,
      gamma_tilde=gamma_tilde,
      inverse_gamma_tilde=inverse_gamma_tilde,
      gamma=gamma,
      inverse_gamma=inverse_gamma,
      d_gamma=d_gamma,
      dd_gamma=dd_gamma,
      extrinsic_curvature=extrinsic_curvature,
      d_extrinsic_curvature=d_extrinsic_curvature,
      trace_extrinsic_curvature=trace_extrinsic_curvature,
      d_trace_extrinsic_curvature=d_trace_extrinsic_curvature,
   )


def z4c_physical_christoffel(
   geometry: Z4cPhysicalGeometryExpressions,
) -> list[list[list[sp.Expr]]]:
   """Return physical Christoffel symbols from the canonical Z4c geometry."""
   return _christoffel(geometry.inverse_gamma, list(geometry.d_gamma))


def _state_metric() -> sp.Matrix:
   return _symmetric_matrix("state.gamma_tilde")


def z4c_conformal_metric_kernel() -> StructKernel:
   chi = sp.Symbol("state.chi", positive=True)
   gamma_tilde = _state_metric()
   determinant, inverse_gamma_tilde = _inverse_symmetric_3x3(gamma_tilde)
   physical_metric = gamma_tilde / chi
   inverse_physical_metric = chi * inverse_gamma_tilde
   return StructKernel(
      name="z4c_conformal_metric",
      return_type="Z4cConformalMetric",
      arguments=(("const Z4cState&", "state"),),
      outputs=(
         "determinant",
         *_symmetric_outputs("inv_gamma_tilde"),
         *_symmetric_outputs("gamma"),
         *_symmetric_outputs("inv_gamma"),
         "sqrt_det_gamma",
      ),
      expressions=(
         determinant,
         *_symmetric_expressions(inverse_gamma_tilde),
         *_symmetric_expressions(physical_metric),
         *_symmetric_expressions(inverse_physical_metric),
         sp.sqrt(determinant / chi**3),
      ),
   )


def z4c_spacetime_metric_kernel() -> StructKernel:
   chi = sp.Symbol("state.chi", positive=True)
   gamma_tilde = _state_metric()
   determinant, inverse_gamma_tilde = _inverse_symmetric_3x3(gamma_tilde)
   physical_metric = gamma_tilde / chi
   inverse_physical_metric = chi * inverse_gamma_tilde
   return StructKernel(
      name="z4c_spacetime_metric",
      return_type="SpacetimeMetric",
      arguments=(("const Z4cState&", "state"),),
      outputs=(
         "lapse",
         "shift1",
         "shift2",
         "shift3",
         *_symmetric_outputs("gamma"),
         *_symmetric_outputs("inv_gamma"),
         "sqrt_det_gamma",
      ),
      expressions=(
         sp.Symbol("state.lapse"),
         *(sp.Symbol(f"state.shift{component + 1}") for component in range(3)),
         *_symmetric_expressions(physical_metric),
         *_symmetric_expressions(inverse_physical_metric),
         sp.sqrt(determinant / chi**3),
      ),
   )


def z4c_spacetime_metric_derivatives_kernel() -> StructKernel:
   geometry = z4c_physical_geometry_expressions()
   return StructKernel(
      name="z4c_spacetime_metric_derivatives",
      return_type="SpacetimeMetricDerivatives",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
      ),
      outputs=(
         "d1_lapse",
         "d2_lapse",
         "d3_lapse",
         *(f"d{direction + 1}_shift{component + 1}"
           for component in range(3) for direction in range(3)),
         *(f"d{direction + 1}_gamma{first + 1}{second + 1}"
           for direction in range(3) for first, second in SYMMETRIC_COMPONENTS),
         *_symmetric_outputs("extrinsic"),
      ),
      expressions=(
         *(sp.Symbol(f"derivatives.d_lapse[{direction}]") for direction in range(3)),
         *(sp.Symbol(f"derivatives.d_shift[{direction}][{component}]")
           for component in range(3) for direction in range(3)),
         *(geometry.d_gamma[direction][first, second]
           for direction in range(3) for first, second in SYMMETRIC_COMPONENTS),
         *_symmetric_expressions(geometry.extrinsic_curvature),
      ),
   )


def z4c_algebraic_projection_kernel() -> StructKernel:
   gamma_tilde = _state_metric()
   determinant, inverse_gamma_tilde = _inverse_symmetric_3x3(gamma_tilde)
   scale = determinant ** -sp.Rational(1, 3)
   projected_gamma_tilde = scale * gamma_tilde

   a_tilde = _symmetric_matrix("state.a_tilde")
   scaled_a_tilde = scale * a_tilde
   projected_inverse_gamma_tilde = inverse_gamma_tilde / scale
   trace_a_tilde = sum(
      projected_inverse_gamma_tilde[first, second]
      * scaled_a_tilde[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   projected_a_tilde = (
      scaled_a_tilde
      - sp.Rational(1, 3) * projected_gamma_tilde * trace_a_tilde
   )

   return StructKernel(
      name="z4c_project_algebraic_constraints",
      return_type="Z4cState",
      arguments=(("const Z4cState&", "state"),),
      outputs=(
         "chi",
         *_symmetric_outputs("gamma_tilde"),
         "k_hat",
         *_symmetric_outputs("a_tilde"),
         "gamma_tilde1",
         "gamma_tilde2",
         "gamma_tilde3",
         "theta",
         "lapse",
         "shift1",
         "shift2",
         "shift3",
      ),
      expressions=(
         scale * sp.Symbol("state.chi"),
         *_symmetric_expressions(projected_gamma_tilde),
         sp.Symbol("state.k_hat"),
         *_symmetric_expressions(projected_a_tilde),
         sp.Symbol("state.gamma_tilde1"),
         sp.Symbol("state.gamma_tilde2"),
         sp.Symbol("state.gamma_tilde3"),
         sp.Symbol("state.theta"),
         sp.Symbol("state.lapse"),
         sp.Symbol("state.shift1"),
         sp.Symbol("state.shift2"),
         sp.Symbol("state.shift3"),
      ),
   )


def _metric_from_argument(prefix: str, component_prefix: str) -> sp.Matrix:
   return _symmetric_matrix(f"{prefix}.{component_prefix}")


def z4c_connections_kernel() -> StructKernel:
   chi = sp.Symbol("state.chi", positive=True)
   gamma_tilde = _state_metric()
   inverse_gamma_tilde = _metric_from_argument("metric", "inv_gamma_tilde")
   inverse_physical_metric = _metric_from_argument("metric", "inv_gamma")
   d_chi = [sp.Symbol(f"derivatives.d_chi[{direction}]") for direction in range(DIMENSION)]
   d_gamma_tilde = _symmetric_derivatives("derivatives.d_gamma_tilde")

   d_physical_metric = [
      d_gamma_tilde[direction] / chi
      - gamma_tilde * d_chi[direction] / chi**2
      for direction in range(DIMENSION)
   ]
   conformal_christoffel = _christoffel(inverse_gamma_tilde, d_gamma_tilde)
   physical_christoffel = _christoffel(inverse_physical_metric, d_physical_metric)
   contracted_conformal = [
      sum(
         inverse_gamma_tilde[first, second]
         * conformal_christoffel[upper][first][second]
         for first in range(DIMENSION)
         for second in range(DIMENSION)
      )
      for upper in range(DIMENSION)
   ]
   connection_outputs: list[str] = []
   connection_expressions: list[sp.Expr] = []
   physical_connection_outputs: list[str] = []
   physical_connection_expressions: list[sp.Expr] = []
   for upper in range(DIMENSION):
      for first, second in SYMMETRIC_COMPONENTS:
         connection_outputs.append(
            f"christoffel_tilde{upper + 1}_{first + 1}{second + 1}"
         )
         connection_expressions.append(conformal_christoffel[upper][first][second])
         physical_connection_outputs.append(
            f"christoffel{upper + 1}_{first + 1}{second + 1}"
         )
         physical_connection_expressions.append(physical_christoffel[upper][first][second])
   return StructKernel(
      name="z4c_connections",
      return_type="Z4cConnections",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("const Z4cConformalMetric&", "metric"),
      ),
      outputs=(
         *connection_outputs,
         "contracted_christoffel_tilde1",
         "contracted_christoffel_tilde2",
         "contracted_christoffel_tilde3",
         *physical_connection_outputs,
      ),
      expressions=(
         *connection_expressions,
         *contracted_conformal,
         *physical_connection_expressions,
      ),
   )


def z4c_diagnostic_geometry_kernel() -> StructKernel:
   geometry = z4c_physical_geometry_expressions()
   inverse_physical_metric = _metric_from_argument("metric", "inv_gamma")
   ricci = _ricci_tensor(
      inverse_physical_metric,
      list(geometry.d_gamma),
      [list(row) for row in geometry.dd_gamma],
      _connections_physical_christoffel(),
   )
   return StructKernel(
      name="z4c_diagnostic_geometry",
      return_type="Z4cDiagnosticGeometry",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("const Z4cConformalMetric&", "metric"),
         ("const Z4cConnections&", "connections"),
      ),
      outputs=(*_symmetric_outputs("ricci"), "ricci_scalar"),
      expressions=(
         *_symmetric_expressions(ricci),
         sum(
            inverse_physical_metric[first, second] * ricci[first, second]
            for first in range(DIMENSION)
            for second in range(DIMENSION)
         ),
      ),
   )


def _connections_conformal_christoffel() -> list[list[list[sp.Expr]]]:
   christoffel = [
      [[sp.Integer(0) for _ in range(DIMENSION)] for _ in range(DIMENSION)]
      for _ in range(DIMENSION)
   ]
   for upper in range(DIMENSION):
      for first in range(DIMENSION):
         for second in range(DIMENSION):
            low, high = sorted((first, second))
            christoffel[upper][first][second] = sp.Symbol(
               f"connections.christoffel_tilde{upper + 1}_{low + 1}{high + 1}"
            )
   return christoffel


def _connections_physical_christoffel() -> list[list[list[sp.Expr]]]:
   christoffel = [
      [[sp.Integer(0) for _ in range(DIMENSION)] for _ in range(DIMENSION)]
      for _ in range(DIMENSION)
   ]
   for upper in range(DIMENSION):
      for first in range(DIMENSION):
         for second in range(DIMENSION):
            low, high = sorted((first, second))
            christoffel[upper][first][second] = sp.Symbol(
               f"connections.christoffel{upper + 1}_{low + 1}{high + 1}"
            )
   return christoffel


def z4c_constraints_kernel() -> StructKernel:
   chi = sp.Symbol("state.chi", positive=True)
   inverse_gamma_tilde = _metric_from_argument("metric", "inv_gamma_tilde")
   a_tilde = _symmetric_matrix("state.a_tilde")
   d_gamma_tilde = _symmetric_derivatives("derivatives.d_gamma_tilde")
   d_a_tilde = _symmetric_derivatives("derivatives.d_a_tilde")
   inverse_derivatives = _inverse_metric_derivatives(inverse_gamma_tilde, d_gamma_tilde)
   a_tilde_up = inverse_gamma_tilde * a_tilde * inverse_gamma_tilde
   d_a_tilde_up = [
      inverse_derivatives[direction] * a_tilde * inverse_gamma_tilde
      + inverse_gamma_tilde * d_a_tilde[direction] * inverse_gamma_tilde
      + inverse_gamma_tilde * a_tilde * inverse_derivatives[direction]
      for direction in range(DIMENSION)
   ]
   christoffel = _connections_conformal_christoffel()
   gamma_tilde_vector = [sp.Symbol(f"state.gamma_tilde{index + 1}") for index in range(DIMENSION)]
   contracted_christoffel = [
      sp.Symbol(f"connections.contracted_christoffel_tilde{index + 1}")
      for index in range(DIMENSION)
   ]
   total_curvature = sp.Symbol("state.k_hat") + 2 * sp.Symbol("state.theta")
   d_total_curvature = [
      sp.Symbol(f"derivatives.d_k_hat[{direction}]")
      + 2 * sp.Symbol(f"derivatives.d_theta[{direction}]")
      for direction in range(DIMENSION)
   ]
   d_chi = [sp.Symbol(f"derivatives.d_chi[{direction}]") for direction in range(DIMENSION)]
   momentum_covariant = [sp.Symbol(f"matter.momentum{index + 1}") for index in range(DIMENSION)]
   momentum_constraints: list[sp.Expr] = []
   for upper in range(DIMENSION):
      value = sum(d_a_tilde_up[second][upper, second] for second in range(DIMENSION))
      value += sum(
         christoffel[upper][first][second] * a_tilde_up[first, second]
         for first in range(DIMENSION)
         for second in range(DIMENSION)
      )
      value -= sp.Rational(2, 3) * sum(
         inverse_gamma_tilde[upper, direction] * d_total_curvature[direction]
         for direction in range(DIMENSION)
      )
      value -= sp.Rational(3, 2) * sum(
         a_tilde_up[upper, direction] * d_chi[direction] / chi
         for direction in range(DIMENSION)
      )
      value -= 8 * sp.pi * sum(
         inverse_gamma_tilde[upper, direction] * momentum_covariant[direction]
         for direction in range(DIMENSION)
      )
      momentum_constraints.append(value)
   trace_a_tilde = sum(
      inverse_gamma_tilde[first, second] * a_tilde[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   a_tilde_squared = sum(
      a_tilde[first, second] * a_tilde_up[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   return StructKernel(
      name="z4c_constraints_from_geometry",
      return_type="Z4cConstraints",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("const Z4cConformalMetric&", "metric"),
         ("const Z4cConnections&", "connections"),
         ("const Z4cDiagnosticGeometry&", "diagnostic_geometry"),
         ("const Z4cMatter&", "matter"),
      ),
      outputs=(
         "theta",
         "z_tilde1",
         "z_tilde2",
         "z_tilde3",
         "hamiltonian",
         "momentum1",
         "momentum2",
         "momentum3",
         "log_conformal_determinant",
         "trace_a_tilde",
      ),
      expressions=(
         sp.Symbol("state.theta"),
         *(sp.Rational(1, 2) * (gamma_tilde_vector[index] - contracted_christoffel[index]) for index in range(DIMENSION)),
         sp.Symbol("diagnostic_geometry.ricci_scalar")
         - a_tilde_squared
         + sp.Rational(2, 3) * total_curvature**2
         - 16 * sp.pi * sp.Symbol("matter.rho"),
         *momentum_constraints,
         sp.log(sp.Symbol("metric.determinant", positive=True)),
         trace_a_tilde,
      ),
   )


def z4c_evolution_geometry_kernel() -> StructKernel:
   chi = sp.Symbol("state.chi", positive=True)
   gamma_tilde = _state_metric()
   inverse_gamma_tilde = _metric_from_argument("metric", "inv_gamma_tilde")
   inverse_physical_metric = _metric_from_argument("metric", "inv_gamma")
   d_chi = [sp.Symbol(f"derivatives.d_chi[{direction}]") for direction in range(DIMENSION)]
   dd_chi = [
      [sp.Symbol(f"derivatives.dd_chi[{first}][{second}]") for second in range(DIMENSION)]
      for first in range(DIMENSION)
   ]
   d_gamma_tilde = _symmetric_derivatives("derivatives.d_gamma_tilde")
   dd_gamma_tilde = _symmetric_second_derivatives("derivatives.dd_gamma_tilde")
   d_gamma_vector = [
      [sp.Symbol(f"derivatives.d_gamma_tilde_vector[{direction}][{component}]") for component in range(DIMENSION)]
      for direction in range(DIMENSION)
   ]
   christoffel = _connections_conformal_christoffel()
   contracted = [
      sp.Symbol(f"connections.contracted_christoffel_tilde{component + 1}")
      for component in range(DIMENSION)
   ]
   first_kind = [
      [
         [
            sp.Rational(1, 2)
            * (
               d_gamma_tilde[second][first, third]
               + d_gamma_tilde[third][first, second]
               - d_gamma_tilde[first][second, third]
            )
            for third in range(DIMENSION)
         ]
         for second in range(DIMENSION)
      ]
      for first in range(DIMENSION)
   ]
   conformal_hessian_chi = sp.Matrix(
      DIMENSION,
      DIMENSION,
      lambda first, second: dd_chi[first][second]
      - sum(christoffel[upper][first][second] * d_chi[upper] for upper in range(DIMENSION)),
   )
   conformal_laplacian_chi = sum(
      inverse_gamma_tilde[first, second] * conformal_hessian_chi[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   conformal_gradient_chi_squared = sum(
      inverse_gamma_tilde[first, second] * d_chi[first] * d_chi[second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   ricci_chi = sp.Matrix(
      DIMENSION,
      DIMENSION,
      lambda first, second:
         conformal_hessian_chi[first, second] / (2 * chi)
         + gamma_tilde[first, second] * conformal_laplacian_chi / (2 * chi)
         - d_chi[first] * d_chi[second] / (4 * chi**2)
         - 3 * gamma_tilde[first, second] * conformal_gradient_chi_squared / (4 * chi**2),
   )
   ricci_tilde = sp.zeros(DIMENSION)
   for first in range(DIMENSION):
      for second in range(DIMENSION):
         value = -sp.Rational(1, 2) * sum(
            inverse_gamma_tilde[left, right] * dd_gamma_tilde[left][right][first, second]
            for left in range(DIMENSION)
            for right in range(DIMENSION)
         )
         value += sp.Rational(1, 2) * sum(
            gamma_tilde[upper, first] * d_gamma_vector[second][upper]
            + gamma_tilde[upper, second] * d_gamma_vector[first][upper]
            for upper in range(DIMENSION)
         )
         value += sp.Rational(1, 2) * sum(
            contracted[upper]
            * (first_kind[first][second][upper] + first_kind[second][first][upper])
            for upper in range(DIMENSION)
         )
         value += sum(
            inverse_gamma_tilde[left, right]
            * (
               christoffel[upper][left][first] * first_kind[second][upper][right]
               + christoffel[upper][left][second] * first_kind[first][upper][right]
               + christoffel[upper][first][right] * first_kind[upper][left][second]
            )
            for left in range(DIMENSION)
            for right in range(DIMENSION)
            for upper in range(DIMENSION)
         )
         ricci_tilde[first, second] = value
   ricci = ricci_chi + ricci_tilde
   return StructKernel(
      name="z4c_evolution_geometry",
      return_type="Z4cEvolutionGeometry",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("const Z4cConformalMetric&", "metric"),
         ("const Z4cConnections&", "connections"),
      ),
      outputs=(*_symmetric_outputs("ricci"), "ricci_scalar"),
      expressions=(
         *_symmetric_expressions(ricci),
         sum(
            inverse_physical_metric[first, second] * ricci[first, second]
            for first in range(DIMENSION)
            for second in range(DIMENSION)
         ),
      ),
   )


def z4c_geometric_rhs_kernel() -> StructKernel:
   chi = sp.Symbol("state.chi", positive=True)
   lapse = sp.Symbol("state.lapse", positive=True)
   theta = sp.Symbol("state.theta")
   k_hat = sp.Symbol("state.k_hat")
   total_curvature = k_hat + 2 * theta
   gamma_tilde = _state_metric()
   inverse_gamma_tilde = _metric_from_argument("metric", "inv_gamma_tilde")
   physical_metric = _metric_from_argument("metric", "gamma")
   inverse_physical_metric = _metric_from_argument("metric", "inv_gamma")
   a_tilde = _symmetric_matrix("state.a_tilde")
   a_tilde_up = inverse_gamma_tilde * a_tilde * inverse_gamma_tilde
   a_tilde_mixed = inverse_gamma_tilde * a_tilde
   a_tilde_squared = sum(
      a_tilde[first, second] * a_tilde_up[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   shift = [sp.Symbol(f"state.shift{component + 1}") for component in range(DIMENSION)]
   d_shift = [
      [sp.Symbol(f"derivatives.d_shift[{direction}][{component}]") for component in range(DIMENSION)]
      for direction in range(DIMENSION)
   ]
   dd_shift = [
      [
         [sp.Symbol(f"derivatives.dd_shift[{first}][{second}][{component}]") for component in range(DIMENSION)]
         for second in range(DIMENSION)
      ]
      for first in range(DIMENSION)
   ]
   d_chi = [sp.Symbol(f"derivatives.d_chi[{direction}]") for direction in range(DIMENSION)]
   advection_chi = [
      sp.Symbol(f"derivatives.advection_chi[{direction}]")
      for direction in range(DIMENSION)
   ]
   advection_gamma_tilde = _symmetric_derivatives(
      "derivatives.advection_gamma_tilde"
   )
   advection_a_tilde = _symmetric_derivatives("derivatives.advection_a_tilde")
   d_k_hat = [sp.Symbol(f"derivatives.d_k_hat[{direction}]") for direction in range(DIMENSION)]
   advection_k_hat = [
      sp.Symbol(f"derivatives.advection_k_hat[{direction}]")
      for direction in range(DIMENSION)
   ]
   advection_gamma_vector = [
      [
         sp.Symbol(
            f"derivatives.advection_gamma_tilde_vector[{direction}][{component}]"
         )
         for component in range(DIMENSION)
      ]
      for direction in range(DIMENSION)
   ]
   d_theta = [sp.Symbol(f"derivatives.d_theta[{direction}]") for direction in range(DIMENSION)]
   advection_theta = [
      sp.Symbol(f"derivatives.advection_theta[{direction}]")
      for direction in range(DIMENSION)
   ]
   d_lapse = [sp.Symbol(f"derivatives.d_lapse[{direction}]") for direction in range(DIMENSION)]
   dd_lapse = [
      [sp.Symbol(f"derivatives.dd_lapse[{first}][{second}]") for second in range(DIMENSION)]
      for first in range(DIMENSION)
   ]
   physical_christoffel = _connections_physical_christoffel()
   conformal_christoffel = _connections_conformal_christoffel()
   contracted_conformal = [
      sp.Symbol(f"connections.contracted_christoffel_tilde{component + 1}")
      for component in range(DIMENSION)
   ]
   evolved_gamma = [sp.Symbol(f"state.gamma_tilde{component + 1}") for component in range(DIMENSION)]
   ricci = _symmetric_matrix("evolution_geometry.ricci")
   hessian_lapse = sp.Matrix(
      DIMENSION,
      DIMENSION,
      lambda first, second: dd_lapse[first][second]
      - sum(physical_christoffel[upper][first][second] * d_lapse[upper] for upper in range(DIMENSION)),
   )
   laplacian_lapse = sum(
      inverse_physical_metric[first, second] * hessian_lapse[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   stress = _symmetric_matrix("matter.stress")
   stress_trace = sum(
      inverse_physical_metric[first, second] * stress[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   momentum = [sp.Symbol(f"matter.momentum{component + 1}") for component in range(DIMENSION)]
   divergence_shift = sum(d_shift[direction][direction] for direction in range(DIMENSION))
   shift_advection_chi = sum(
      shift[direction] * advection_chi[direction]
      for direction in range(DIMENSION)
   )
   chi_rhs = shift_advection_chi + sp.Rational(2, 3) * chi * (
      lapse * total_curvature - divergence_shift
   )
   gamma_rhs = sp.zeros(DIMENSION)
   a_rhs = sp.zeros(DIMENSION)
   source_tensor = -hessian_lapse + lapse * (ricci - 8 * sp.pi * stress)
   source_trace = sum(
      inverse_physical_metric[first, second] * source_tensor[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   for first in range(DIMENSION):
      for second in range(DIMENSION):
         gamma_rhs[first, second] = (
            -2 * lapse * a_tilde[first, second]
            + sum(
               shift[direction] * advection_gamma_tilde[direction][first, second]
               for direction in range(DIMENSION)
            )
            + sum(
               gamma_tilde[direction, first] * d_shift[second][direction]
               + gamma_tilde[direction, second] * d_shift[first][direction]
               for direction in range(DIMENSION)
            )
            - sp.Rational(2, 3) * gamma_tilde[first, second] * divergence_shift
         )
         a_rhs[first, second] = (
            chi * (
               source_tensor[first, second]
               - sp.Rational(1, 3) * physical_metric[first, second] * source_trace
            )
            + lapse * (
               total_curvature * a_tilde[first, second]
               - 2 * sum(a_tilde_mixed[direction, first] * a_tilde[direction, second] for direction in range(DIMENSION))
            )
            + sum(
               shift[direction] * advection_a_tilde[direction][first, second]
               for direction in range(DIMENSION)
            )
            + sum(
               a_tilde[direction, first] * d_shift[second][direction]
               + a_tilde[direction, second] * d_shift[first][direction]
               for direction in range(DIMENSION)
            )
            - sp.Rational(2, 3) * a_tilde[first, second] * divergence_shift
         )
   kappa1 = sp.Symbol("kappa1")
   kappa2 = sp.Symbol("kappa2")
   k_hat_rhs = (
      -laplacian_lapse
      + lapse * (a_tilde_squared + sp.Rational(1, 3) * total_curvature**2)
      + 4 * sp.pi * lapse * (stress_trace + sp.Symbol("matter.rho"))
      + lapse * kappa1 * (1 - kappa2) * theta
      + sum(
         shift[direction] * advection_k_hat[direction]
         for direction in range(DIMENSION)
      )
   )
   gamma_rhs_vector: list[sp.Expr] = []
   for upper in range(DIMENSION):
      value = -2 * sum(a_tilde_up[upper, direction] * d_lapse[direction] for direction in range(DIMENSION))
      value += 2 * lapse * (
         sum(
            conformal_christoffel[upper][first][second] * a_tilde_up[first, second]
            for first in range(DIMENSION)
            for second in range(DIMENSION)
         )
         - sp.Rational(3, 2) * sum(
            a_tilde_up[upper, direction] * d_chi[direction] / chi
            for direction in range(DIMENSION)
         )
         - sp.Rational(1, 3) * sum(
            inverse_gamma_tilde[upper, direction]
            * (2 * d_k_hat[direction] + d_theta[direction])
            for direction in range(DIMENSION)
         )
         - 8 * sp.pi * sum(
            inverse_gamma_tilde[upper, direction] * momentum[direction]
            for direction in range(DIMENSION)
         )
      )
      value += sum(
         inverse_gamma_tilde[first, second] * dd_shift[first][second][upper]
         for first in range(DIMENSION)
         for second in range(DIMENSION)
      )
      value += sp.Rational(1, 3) * sum(
         inverse_gamma_tilde[upper, first] * dd_shift[first][second][second]
         for first in range(DIMENSION)
         for second in range(DIMENSION)
      )
      value += sum(
         shift[direction] * advection_gamma_vector[direction][upper]
         for direction in range(DIMENSION)
      )
      value -= sum(contracted_conformal[direction] * d_shift[direction][upper] for direction in range(DIMENSION))
      value += sp.Rational(2, 3) * contracted_conformal[upper] * divergence_shift
      value -= 2 * lapse * kappa1 * (evolved_gamma[upper] - contracted_conformal[upper])
      gamma_rhs_vector.append(value)
   theta_rhs = (
      sp.Rational(1, 2) * lapse * (
         sp.Symbol("evolution_geometry.ricci_scalar")
         - a_tilde_squared
         + sp.Rational(2, 3) * total_curvature**2
      )
      - lapse * (
         8 * sp.pi * sp.Symbol("matter.rho")
         + kappa1 * (2 + kappa2) * theta
      )
      + sum(
         shift[direction] * advection_theta[direction]
         for direction in range(DIMENSION)
      )
   )
   return StructKernel(
      name="z4c_geometric_rhs",
      return_type="Z4cGeometricRhs",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("const Z4cConformalMetric&", "metric"),
         ("const Z4cConnections&", "connections"),
         ("const Z4cEvolutionGeometry&", "evolution_geometry"),
         ("const Z4cMatter&", "matter"),
         ("amrex::Real", "kappa1"),
         ("amrex::Real", "kappa2"),
      ),
      outputs=(
         "chi",
         *_symmetric_outputs("gamma_tilde"),
         "k_hat",
         *_symmetric_outputs("a_tilde"),
         "gamma_tilde1",
         "gamma_tilde2",
         "gamma_tilde3",
         "theta",
      ),
      expressions=(
         chi_rhs,
         *_symmetric_expressions(gamma_rhs),
         k_hat_rhs,
         *_symmetric_expressions(a_rhs),
         *gamma_rhs_vector,
         theta_rhs,
      ),
   )


def z4c_harmonic_lapse_rhs_kernel() -> ScalarKernel:
   return ScalarKernel(
      name="z4c_harmonic_lapse_rhs",
      arguments=(("const Z4cState&", "state"), ("const Z4cSpatialDerivatives&", "derivatives")),
      expression=(
         -sp.Symbol("state.lapse") ** 2 * sp.Symbol("state.k_hat")
         + sum(
            sp.Symbol(f"state.shift{direction + 1}")
            * sp.Symbol(f"derivatives.advection_lapse[{direction}]")
            for direction in range(DIMENSION)
         )
      ),
   )


def z4c_one_plus_log_lapse_rhs_kernel() -> ScalarKernel:
   return ScalarKernel(
      name="z4c_one_plus_log_lapse_rhs",
      arguments=(("const Z4cState&", "state"), ("const Z4cSpatialDerivatives&", "derivatives")),
      expression=(
         -2 * sp.Symbol("state.lapse") * sp.Symbol("state.k_hat")
         + sum(
            sp.Symbol(f"state.shift{direction + 1}")
            * sp.Symbol(f"derivatives.advection_lapse[{direction}]")
            for direction in range(DIMENSION)
         )
      ),
   )


def z4c_trumpet_lapse_rhs_kernel() -> ScalarKernel:
   lapse = sp.Symbol("state.lapse")
   return ScalarKernel(
      name="z4c_trumpet_lapse_rhs",
      arguments=(("const Z4cState&", "state"), ("const Z4cSpatialDerivatives&", "derivatives")),
      expression=(
         -lapse * (1 - lapse) * sp.Symbol("state.k_hat")
         + sum(
            sp.Symbol(f"state.shift{direction + 1}")
            * sp.Symbol(f"derivatives.advection_lapse[{direction}]")
            for direction in range(DIMENSION)
         )
      ),
   )


def z4c_slow_start_lapse_rhs_kernel() -> ScalarKernel:
   chi = sp.Symbol("state.chi", positive=True)
   lapse = sp.Symbol("state.lapse")
   time = sp.Symbol("time", nonnegative=True)
   strength = sp.Symbol("strength")
   width = sp.Symbol("width", positive=True)
   one_plus_log_rhs = (
      -2 * lapse * sp.Symbol("state.k_hat")
      + sum(
         sp.Symbol(f"state.shift{direction + 1}")
         * sp.Symbol(f"derivatives.advection_lapse[{direction}]")
         for direction in range(DIMENSION)
      )
   )
   conformal_weight = sp.sqrt(chi)
   return ScalarKernel(
      name="z4c_slow_start_lapse_rhs",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("amrex::Real", "time"),
         ("amrex::Real", "strength"),
         ("amrex::Real", "width"),
      ),
      expression=(
         one_plus_log_rhs
         - conformal_weight
         * strength
         * sp.exp(-time**2 / (2 * width**2))
         * (lapse - conformal_weight)
      ),
   )


def z4c_gamma_driver_shift_rhs_kernel() -> StructKernel:
   shift_mu = sp.Symbol("shift_mu")
   shift_eta = sp.Symbol("shift_eta")
   lapse = sp.Symbol("state.lapse")
   expressions = []
   for component in range(DIMENSION):
      expressions.append(
         lapse**2 * shift_mu * sp.Symbol(f"state.gamma_tilde{component + 1}")
         - shift_eta * sp.Symbol(f"state.shift{component + 1}")
         + sum(
            sp.Symbol(f"state.shift{direction + 1}")
            * sp.Symbol(f"derivatives.advection_shift[{direction}][{component}]")
            for direction in range(DIMENSION)
         )
      )
   return StructKernel(
      name="z4c_gamma_driver_shift_rhs",
      return_type="Z4cShiftRhs",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("amrex::Real", "shift_mu"),
         ("amrex::Real", "shift_eta"),
      ),
      outputs=("shift1", "shift2", "shift3"),
      expressions=tuple(expressions),
   )


def z4c_geometry_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (
      z4c_conformal_metric_kernel(),
      z4c_spacetime_metric_kernel(),
      z4c_spacetime_metric_derivatives_kernel(),
      z4c_algebraic_projection_kernel(),
      z4c_connections_kernel(),
      z4c_diagnostic_geometry_kernel(),
      z4c_evolution_geometry_kernel(),
   )


def z4c_constraint_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (z4c_constraints_kernel(),)


def z4c_evolution_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (z4c_geometric_rhs_kernel(),)


def z4c_gauge_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (
      z4c_harmonic_lapse_rhs_kernel(),
      z4c_one_plus_log_lapse_rhs_kernel(),
      z4c_trumpet_lapse_rhs_kernel(),
      z4c_slow_start_lapse_rhs_kernel(),
      z4c_gamma_driver_shift_rhs_kernel(),
   )


def z4c_kernel_groups() -> dict[str, tuple[ScalarKernel | StructKernel, ...]]:
   return {
      "geometry": z4c_geometry_kernels(),
      "constraints": z4c_constraint_kernels(),
      "gauge": z4c_gauge_kernels(),
      "evolution": z4c_evolution_kernels(),
   }


def z4c_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return tuple(
      kernel
      for kernels in z4c_kernel_groups().values()
      for kernel in kernels
   )
