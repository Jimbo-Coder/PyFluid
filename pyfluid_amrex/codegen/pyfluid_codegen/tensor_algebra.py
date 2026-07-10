from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import sympy as sp


def _as_expr(value: sp.Expr | int | float) -> sp.Expr:
   return value if isinstance(value, sp.Expr) else sp.sympify(value)


def _unit_match(index: int, axis: int, dimension: int) -> int:
   if index < 0 or index >= dimension:
      raise ValueError(f"index {index} is outside [0, {dimension})")
   if axis < 0 or axis >= dimension:
      raise ValueError(f"axis {axis} is outside [0, {dimension})")
   return 1 if axis == index else 0


def _as_vector(symbols: tuple[sp.Expr, ...], indexes: tuple[int, int, int] | None = None) -> Vector3:
   # compatibility shim retained for localized convenience when symbol names are pre-grouped
   if indexes is None:
      indexes = (0, 1, 2)
   x, y, z = (symbols[i] for i in indexes)
   return Vector3(x, y, z)


@dataclass(frozen=True)
class Vector3:
   """Small immutable 3-component algebraic vector for codegen expressions."""

   x: sp.Expr
   y: sp.Expr
   z: sp.Expr

   @staticmethod
   def zero() -> Vector3:
      return Vector3(sp.Integer(0), sp.Integer(0), sp.Integer(0))

   @staticmethod
   def unit(index: int) -> Vector3:
      return Vector3(
         sp.Integer(_unit_match(index, 0, 3)),
         sp.Integer(_unit_match(index, 1, 3)),
         sp.Integer(_unit_match(index, 2, 3)),
      )

   def __iter__(self) -> Iterable[sp.Expr]:
      return iter((self.x, self.y, self.z))

   def __add__(self, other: Vector3) -> Vector3:
      return self.plus(other)

   def __sub__(self, other: Vector3) -> Vector3:
      return self.minus(other)

   def __neg__(self) -> Vector3:
      return Vector3(-self.x, -self.y, -self.z)

   def __mul__(self, scalar: sp.Expr | int | float) -> Vector3:
      return self.scaled(_as_expr(scalar))

   def __rmul__(self, scalar: sp.Expr | int | float) -> Vector3:
      return self * scalar

   def __truediv__(self, scalar: sp.Expr | int | float) -> Vector3:
      return self.divided_by(_as_expr(scalar))

   def with_component(self, index: int, value: sp.Expr | int | float) -> Vector3:
      return Vector3(
         _as_expr(value) if index == 0 else self.x,
         _as_expr(value) if index == 1 else self.y,
         _as_expr(value) if index == 2 else self.z,
      )

   def with_zero_component(self, index: int) -> Vector3:
      return self.with_component(index, sp.Integer(0))

   def as_tuple(self) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
      return (self.x, self.y, self.z)

   def component(self, index: int) -> sp.Expr:
      return self.as_tuple()[index]

   def scaled(self, scalar: sp.Expr | int | float) -> Vector3:
      scalar = _as_expr(scalar)
      return Vector3(scalar * self.x, scalar * self.y, scalar * self.z)

   def divided_by(self, scalar: sp.Expr | int | float) -> Vector3:
      scalar = _as_expr(scalar)
      return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

   def plus(self, other: Vector3) -> Vector3:
      return Vector3(
         self.x + other.x,
         self.y + other.y,
         self.z + other.z,
      )

   def minus(self, other: Vector3) -> Vector3:
      return Vector3(
         self.x - other.x,
         self.y - other.y,
         self.z - other.z,
      )

   def dot(self, other: Vector3) -> sp.Expr:
      return self.x * other.x + self.y * other.y + self.z * other.z

   def cross(self, other: Vector3) -> Vector3:
      return Vector3(
         self.y * other.z - self.z * other.y,
         self.z * other.x - self.x * other.z,
         self.x * other.y - self.y * other.x,
      )

   def outer(self, other: Vector3) -> Tensor3x3:
      return Tensor3x3(
         Vector3(self.x * other.x, self.x * other.y, self.x * other.z),
         Vector3(self.y * other.x, self.y * other.y, self.y * other.z),
         Vector3(self.z * other.x, self.z * other.y, self.z * other.z),
      )

   def norm_squared(self) -> sp.Expr:
      return self.dot(self)

   def scaled_add(self, other: Vector3, scalar: sp.Expr | int | float = 1) -> Vector3:
      scalar = _as_expr(scalar)
      return self + scalar * other


@dataclass(frozen=True)
class Vector4:
   """Small immutable 4-component algebraic vector for future relativistic states."""

   t: sp.Expr
   x: sp.Expr
   y: sp.Expr
   z: sp.Expr

   def component(self, index: int) -> sp.Expr:
      return (self.t, self.x, self.y, self.z)[index]

   @staticmethod
   def zero() -> Vector4:
      return Vector4(sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0))

   def scaled(self, scalar: sp.Expr | int | float) -> Vector4:
      scalar = _as_expr(scalar)
      return Vector4(scalar * self.t, scalar * self.x, scalar * self.y, scalar * self.z)

   def plus(self, other: Vector4) -> Vector4:
      return Vector4(
         self.t + other.t,
         self.x + other.x,
         self.y + other.y,
         self.z + other.z,
      )

   def minus(self, other: Vector4) -> Vector4:
      return Vector4(
         self.t - other.t,
         self.x - other.x,
         self.y - other.y,
         self.z - other.z,
      )

   def as_tuple(self) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
      return (self.t, self.x, self.y, self.z)

   def dot(self, other: Vector4) -> sp.Expr:
      return self.t * other.t + self.x * other.x + self.y * other.y + self.z * other.z


@dataclass(frozen=True)
class Tensor3x3:
   """Row-wise immutable 3x3 tensor in algebraic coordinates."""

   @staticmethod
   def identity(scale: sp.Expr | int | float = sp.Integer(1)) -> Tensor3x3:
      scale = _as_expr(scale)
      return Tensor3x3(
         Vector3(scale, sp.Integer(0), sp.Integer(0)),
         Vector3(sp.Integer(0), scale, sp.Integer(0)),
         Vector3(sp.Integer(0), sp.Integer(0), scale),
      )

   @classmethod
   def from_rows(cls, rows: tuple[tuple[sp.Expr, sp.Expr, sp.Expr], ...]) -> Tensor3x3:
      if len(rows) != 3:
         raise ValueError("expected 3 tensor rows")
      return cls(_as_vector(rows[0]), _as_vector(rows[1]), _as_vector(rows[2]))

   row1: Vector3
   row2: Vector3
   row3: Vector3

   def row(self, index: int) -> Vector3:
      return (self.row1, self.row2, self.row3)[index]

   def as_rows(self) -> tuple[tuple[sp.Expr, sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr, sp.Expr]]:
      return (self.row1.as_tuple(), self.row2.as_tuple(), self.row3.as_tuple())

   @staticmethod
   def zero() -> Tensor3x3:
      return Tensor3x3(
         Vector3.zero(),
         Vector3.zero(),
         Vector3.zero(),
      )

   def component(self, row: int, col: int) -> sp.Expr:
      return self.row(row).component(col)

   def trace(self) -> sp.Expr:
      return (
         self.row1.component(0) + self.row2.component(1) + self.row3.component(2)
      )

   def scaled(self, scalar: sp.Expr | int | float) -> Tensor3x3:
      scalar = _as_expr(scalar)
      return Tensor3x3(
         self.row1.scaled(scalar),
         self.row2.scaled(scalar),
         self.row3.scaled(scalar),
      )

   def contract_with(self, vector: Vector3) -> Vector3:
      return Vector3(
         self.row(0).dot(vector),
         self.row(1).dot(vector),
         self.row(2).dot(vector),
      )

   def transpose(self) -> Tensor3x3:
      return Tensor3x3(
         Vector3(self.row1.x, self.row2.x, self.row3.x),
         Vector3(self.row1.y, self.row2.y, self.row3.y),
         Vector3(self.row1.z, self.row2.z, self.row3.z),
      )

   def plus(self, other: Tensor3x3) -> Tensor3x3:
      return Tensor3x3(
         self.row1.plus(other.row1),
         self.row2.plus(other.row2),
         self.row3.plus(other.row3),
      )

   def minus(self, other: Tensor3x3) -> Tensor3x3:
      return Tensor3x3(
         self.row1.minus(other.row1),
         self.row2.minus(other.row2),
         self.row3.minus(other.row3),
      )


def orthogonal_velocity_gradient(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   velocities: Vector3,
   velocity_derivatives: Tensor3x3,
) -> Tensor3x3:
   """Return the orthogonal-coordinate covariant velocity gradient matrix."""
   h = scale_factors
   gradient: list[list[sp.Expr]] = [[sp.Integer(0) for _ in range(3)] for _ in range(3)]
   for row in range(3):
      for col in range(3):
         if row == col:
            component = velocity_derivatives.component(row, row) / h[row]
            for basis in range(3):
               if basis == row:
                  continue
               component += (
                  velocities.component(basis)
                  * sp.diff(h[row], coordinates[basis])
                  / (h[row] * h[basis])
               )
         else:
            component = h[col] / h[row] * (
               velocity_derivatives.component(col, row) / h[col]
               - velocities.component(col) * sp.diff(h[col], coordinates[row]) / (h[col] ** 2)
            )
         gradient[row][col] = sp.simplify(component)
   return Tensor3x3.from_rows(tuple(tuple(row) for row in gradient))


def orthogonal_velocity_divergence(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   velocities: Vector3,
   velocity_derivatives: Tensor3x3,
) -> sp.Expr:
   """Divergence in orthogonal coordinates using physical components."""
   volume_scale = sp.prod(scale_factors)
   h = scale_factors
   divergence = sp.Integer(0)
   for axis in range(3):
      transverse_area = sp.simplify(volume_scale / h[axis])
      divergence += transverse_area * velocity_derivatives.component(axis, axis)
      divergence += velocities.component(axis) * sp.diff(transverse_area, coordinates[axis])
   return sp.simplify(divergence / volume_scale)


def orthogonal_strain_rate_tensor(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   velocities: Vector3,
   velocity_derivatives: Tensor3x3,
) -> Tensor3x3:
   """Rate-of-strain tensor e_ij in orthogonal coordinates."""
   gradient = orthogonal_velocity_gradient(scale_factors, coordinates, velocities, velocity_derivatives)
   return gradient.plus(gradient.transpose()).scaled(sp.Rational(1, 2))


def orthogonal_newtonian_stress_tensor(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   velocities: Vector3,
   velocity_derivatives: Tensor3x3,
   viscosity: sp.Expr,
) -> Tensor3x3:
   """Newtonian viscous stress: tau_ij = 2 nu (e_ij - delta_ij div(v)/3)."""
   divergence = orthogonal_velocity_divergence(scale_factors, coordinates, velocities, velocity_derivatives)
   strain = orthogonal_strain_rate_tensor(scale_factors, coordinates, velocities, velocity_derivatives)
   return strain.minus(Tensor3x3.identity(sp.Rational(1, 3) * divergence)).scaled(2 * viscosity)


def orthogonal_connection_source_terms(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   tensor: Tensor3x3,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   """Covariant source terms from basis derivatives for a spatial tensor divergence."""
   sources: list[sp.Expr] = []
   for component in range(3):
      source = sp.Integer(0)
      h_component = scale_factors[component]
      q_component = coordinates[component]
      for normal in range(3):
         h_normal = scale_factors[normal]
         q_normal = coordinates[normal]
         source += tensor.component(normal, component) * sp.diff(h_component, q_normal) / (h_component * h_normal)
         source -= tensor.component(normal, normal) * sp.diff(h_normal, q_component) / (h_component * h_normal)
      sources.append(sp.trigsimp(sp.simplify(source)))
   return tuple(sources)


def replace_coordinates(
   values: tuple[sp.Expr, ...],
   source_coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   runtime_coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> tuple[sp.Expr, ...]:
   if len(source_coordinates) != len(runtime_coordinates):
      raise ValueError("source/runtime coordinate tuple length mismatch")
   substitutions = {
      source: runtime
      for source, runtime in zip(source_coordinates, runtime_coordinates, strict=True)
   }
   return tuple(sp.simplify(value.xreplace(substitutions)) for value in values)


def orthogonal_velocity_curvature_sources(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   density: sp.Expr,
   velocities: Vector3,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   """Curvature source terms for rho * v_i v_j in orthogonal coordinates."""
   momentum_flux = velocities.outer(velocities).scaled(density)
   return tuple(-expr for expr in orthogonal_connection_source_terms(scale_factors, coordinates, momentum_flux))


def orthogonal_mhd_stress_curvature_sources(
   scale_factors: tuple[sp.Expr, sp.Expr, sp.Expr],
   coordinates: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
   density: sp.Expr,
   velocities: Vector3,
   magnetic: Vector3,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
   """Curvature source terms for rho * v_i v_j - b_i b_j."""
   stress_tensor = velocities.outer(velocities).scaled(density).minus(magnetic.outer(magnetic))
   return tuple(
      -expr for expr in orthogonal_connection_source_terms(scale_factors, coordinates, stress_tensor)
   )


@dataclass(frozen=True)
class Tensor4x4:
   """Row-wise immutable 4x4 tensor in algebraic coordinates."""

   row1: Vector4
   row2: Vector4
   row3: Vector4
   row4: Vector4

   @staticmethod
   def zero() -> Tensor4x4:
      zero = Vector4(
         sp.Integer(0),
         sp.Integer(0),
         sp.Integer(0),
         sp.Integer(0),
      )
      return Tensor4x4(zero, zero, zero, zero)

   def row(self, index: int) -> Vector4:
      return (self.row1, self.row2, self.row3, self.row4)[index]

   def component(self, row: int, col: int) -> sp.Expr:
      return self.row(row).component(col)

   def contract_with(self, vector: Vector4) -> Vector4:
      return Vector4(
         self.row1.dot(vector),
         self.row2.dot(vector),
         self.row3.dot(vector),
         self.row4.dot(vector),
      )

   def plus(self, other: Tensor4x4) -> Tensor4x4:
      return Tensor4x4(
         Vector4(
            self.row1.t + other.row1.t,
            self.row1.x + other.row1.x,
            self.row1.y + other.row1.y,
            self.row1.z + other.row1.z,
         ),
         Vector4(
            self.row2.t + other.row2.t,
            self.row2.x + other.row2.x,
            self.row2.y + other.row2.y,
            self.row2.z + other.row2.z,
         ),
         Vector4(
            self.row3.t + other.row3.t,
            self.row3.x + other.row3.x,
            self.row3.y + other.row3.y,
            self.row3.z + other.row3.z,
         ),
         Vector4(
            self.row4.t + other.row4.t,
            self.row4.x + other.row4.x,
            self.row4.y + other.row4.y,
            self.row4.z + other.row4.z,
         ),
      )

   def minus(self, other: Tensor4x4) -> Tensor4x4:
      return Tensor4x4(
         Vector4(
            self.row1.t - other.row1.t,
            self.row1.x - other.row1.x,
            self.row1.y - other.row1.y,
            self.row1.z - other.row1.z,
         ),
         Vector4(
            self.row2.t - other.row2.t,
            self.row2.x - other.row2.x,
            self.row2.y - other.row2.y,
            self.row2.z - other.row2.z,
         ),
         Vector4(
            self.row3.t - other.row3.t,
            self.row3.x - other.row3.x,
            self.row3.y - other.row3.y,
            self.row3.z - other.row3.z,
         ),
         Vector4(
            self.row4.t - other.row4.t,
            self.row4.x - other.row4.x,
            self.row4.y - other.row4.y,
            self.row4.z - other.row4.z,
         ),
      )


@dataclass(frozen=True)
class HydroState3:
   """Small immutable 3D hydro state container for symbolic kernels."""

   rho: sp.Expr
   momenta: Vector3
   energy: sp.Expr

   @classmethod
   def from_symbols(
      cls, rho: sp.Symbol, mom1: sp.Symbol, mom2: sp.Symbol, mom3: sp.Symbol, energy: sp.Symbol
   ) -> HydroState3:
      return cls(rho=rho, momenta=_as_vector((mom1, mom2, mom3)), energy=energy)

   def velocity(self) -> Vector3:
      return self.momenta.divided_by(self.rho)

   def momentum_flux(self, normal_direction: int, normal_pressure: sp.Expr) -> Vector3:
      return self.momenta * self.velocity().component(normal_direction) + Vector3.unit(normal_direction) * normal_pressure

   def energy_flux(self, normal_direction: int, pressure: sp.Expr) -> sp.Expr:
      return (self.energy + pressure) * self.velocity().component(normal_direction)


@dataclass(frozen=True)
class MHDState3(HydroState3):
   """Small immutable 3D MHD state container for symbolic kernels."""

   magnetic: Vector3

   @classmethod
   def from_symbols(
      cls,
      rho: sp.Symbol,
      mom1: sp.Symbol,
      mom2: sp.Symbol,
      mom3: sp.Symbol,
      energy: sp.Symbol,
      b1: sp.Symbol,
      b2: sp.Symbol,
      b3: sp.Symbol,
   ) -> MHDState3:
      return cls(
         rho=rho,
         momenta=_as_vector((mom1, mom2, mom3)),
         energy=energy,
         magnetic=_as_vector((b1, b2, b3)),
      )

   def magnetic_energy_density(self) -> sp.Expr:
      return sp.Rational(1, 2) * self.magnetic.norm_squared()

   def total_pressure(self, thermal_pressure: sp.Expr) -> sp.Expr:
      return thermal_pressure + self.magnetic_energy_density()

   def mhd_momentum_flux(self, normal_direction: int, thermal_pressure: sp.Expr) -> Vector3:
      normal_b = self.magnetic.component(normal_direction)
      return (
         self.momenta * self.velocity().component(normal_direction)
         - self.magnetic * normal_b
         + Vector3.unit(normal_direction) * self.total_pressure(thermal_pressure)
      )

   def mhd_magnetic_flux(self, normal_direction: int) -> Vector3:
      normal_b = self.magnetic.component(normal_direction)
      return (self.magnetic * self.velocity().component(normal_direction) - self.velocity() * normal_b).with_zero_component(
         normal_direction
      )

   def mhd_energy_flux(self, normal_direction: int, thermal_pressure: sp.Expr) -> sp.Expr:
      normal_b = self.magnetic.component(normal_direction)
      vdotb = self.momenta.dot(self.magnetic) / self.rho
      return (self.energy + self.total_pressure(thermal_pressure)) * self.velocity().component(normal_direction) - normal_b * vdotb
