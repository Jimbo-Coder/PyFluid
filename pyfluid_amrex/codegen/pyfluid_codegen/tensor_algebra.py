from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from enum import Enum

import sympy as sp


def _as_expr(value: sp.Expr | int | float) -> sp.Expr:
   return value if isinstance(value, sp.Expr) else sp.sympify(value)


def _unit_match(index: int, axis: int, dimension: int) -> int:
   if index < 0 or index >= dimension:
      raise ValueError(f"index {index} is outside [0, {dimension})")
   if axis < 0 or axis >= dimension:
      raise ValueError(f"axis {axis} is outside [0, {dimension})")
   return 1 if axis == index else 0


def symmetric_tensor3_symbols(prefix: str) -> sp.Matrix:
   """Return a symmetric 3-tensor using canonical 11,12,13,22,23,33 names."""
   components = {
      (first, second): sp.Symbol(f"{prefix}{first + 1}{second + 1}")
      for first, second in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
   }
   return sp.Matrix(
      3,
      3,
      lambda row, column: components[tuple(sorted((row, column)))],
   )


@dataclass(frozen=True, init=False)
class Vector:
   """Immutable algebraic vector whose dimension is carried by its values."""

   components: tuple[sp.Expr, ...]

   def __init__(self, *components: sp.Expr | int | float) -> None:
      if not components:
         raise ValueError("a vector must have at least one component")
      object.__setattr__(self, "components", tuple(_as_expr(value) for value in components))

   @property
   def dimension(self) -> int:
      return len(self.components)

   @classmethod
   def zero(cls, dimension: int) -> Vector:
      return cls(*(sp.Integer(0) for _ in range(dimension)))

   @classmethod
   def unit(cls, index: int, dimension: int) -> Vector:
      return cls(*(sp.Integer(_unit_match(index, axis, dimension)) for axis in range(dimension)))

   def __iter__(self) -> Iterable[sp.Expr]:
      return iter(self.components)

   def __add__(self, other: Vector) -> Vector:
      return self.plus(other)

   def __sub__(self, other: Vector) -> Vector:
      return self.minus(other)

   def __neg__(self) -> Vector:
      return Vector(*(-value for value in self))

   def __mul__(self, scalar: sp.Expr | int | float) -> Vector:
      return self.scaled(scalar)

   def __rmul__(self, scalar: sp.Expr | int | float) -> Vector:
      return self * scalar

   def __truediv__(self, scalar: sp.Expr | int | float) -> Vector:
      return self.divided_by(scalar)

   def _require_same_dimension(self, other: Vector) -> None:
      if self.dimension != other.dimension:
         raise ValueError(f"vector dimensions differ: {self.dimension} != {other.dimension}")

   def with_component(self, index: int, value: sp.Expr | int | float) -> Vector:
      return Vector(*(
         _as_expr(value) if axis == index else component
         for axis, component in enumerate(self.components)
      ))

   def with_zero_component(self, index: int) -> Vector:
      return self.with_component(index, sp.Integer(0))

   def as_tuple(self) -> tuple[sp.Expr, ...]:
      return self.components

   def component(self, index: int) -> sp.Expr:
      return self.components[index]

   def scaled(self, scalar: sp.Expr | int | float) -> Vector:
      scalar = _as_expr(scalar)
      return Vector(*(scalar * value for value in self))

   def divided_by(self, scalar: sp.Expr | int | float) -> Vector:
      scalar = _as_expr(scalar)
      return Vector(*(value / scalar for value in self))

   def plus(self, other: Vector) -> Vector:
      self._require_same_dimension(other)
      return Vector(*(left + right for left, right in zip(self, other, strict=True)))

   def minus(self, other: Vector) -> Vector:
      self._require_same_dimension(other)
      return Vector(*(left - right for left, right in zip(self, other, strict=True)))

   def dot(self, other: Vector) -> sp.Expr:
      self._require_same_dimension(other)
      return sum(left * right for left, right in zip(self, other, strict=True))

   def cross(self, other: Vector) -> Vector:
      self._require_same_dimension(other)
      if self.dimension != 3:
         raise ValueError("the vector cross product is defined here only in three dimensions")
      a1, a2, a3 = self.components
      b1, b2, b3 = other.components
      return Vector(a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)

   def outer(self, other: Vector) -> Tensor:
      return Tensor.from_rows(tuple(tuple(left * right for right in other) for left in self))

   def norm_squared(self) -> sp.Expr:
      return self.dot(self)

   def scaled_add(self, other: Vector, scalar: sp.Expr | int | float = 1) -> Vector:
      return self + _as_expr(scalar) * other


@dataclass(frozen=True, init=False)
class Tensor:
   """Immutable rank-two algebraic tensor stored row-wise."""

   rows: tuple[Vector, ...]

   def __init__(self, *rows: Vector) -> None:
      if not rows:
         raise ValueError("a tensor must have at least one row")
      column_count = rows[0].dimension
      if any(row.dimension != column_count for row in rows):
         raise ValueError("tensor rows must have equal dimensions")
      object.__setattr__(self, "rows", tuple(rows))

   @property
   def shape(self) -> tuple[int, int]:
      return len(self.rows), self.rows[0].dimension

   @classmethod
   def identity(cls, dimension: int, scale: sp.Expr | int | float = 1) -> Tensor:
      scale = _as_expr(scale)
      return cls.from_rows(tuple(
         tuple(scale if row == col else sp.Integer(0) for col in range(dimension))
         for row in range(dimension)
      ))

   @classmethod
   def from_rows(cls, rows: tuple[tuple[sp.Expr | int | float, ...], ...]) -> Tensor:
      return cls(*(Vector(*row) for row in rows))

   @classmethod
   def zero(cls, rows: int, columns: int | None = None) -> Tensor:
      columns = rows if columns is None else columns
      return cls(*(Vector.zero(columns) for _ in range(rows)))

   def __add__(self, other: Tensor) -> Tensor:
      return self.plus(other)

   def __sub__(self, other: Tensor) -> Tensor:
      return self.minus(other)

   def __neg__(self) -> Tensor:
      return self.scaled(-1)

   def __mul__(self, scalar: sp.Expr | int | float) -> Tensor:
      return self.scaled(scalar)

   def __rmul__(self, scalar: sp.Expr | int | float) -> Tensor:
      return self * scalar

   def __truediv__(self, scalar: sp.Expr | int | float) -> Tensor:
      return self.scaled(1 / _as_expr(scalar))

   def _require_same_shape(self, other: Tensor) -> None:
      if self.shape != other.shape:
         raise ValueError(f"tensor shapes differ: {self.shape} != {other.shape}")

   def row(self, index: int) -> Vector:
      return self.rows[index]

   def column(self, index: int) -> Vector:
      return Vector(*(row.component(index) for row in self.rows))

   def as_rows(self) -> tuple[tuple[sp.Expr, ...], ...]:
      return tuple(row.as_tuple() for row in self.rows)

   def component(self, row: int, col: int) -> sp.Expr:
      return self.row(row).component(col)

   def trace(self) -> sp.Expr:
      row_count, column_count = self.shape
      if row_count != column_count:
         raise ValueError("trace requires a square tensor")
      return sum(self.component(axis, axis) for axis in range(row_count))

   def double_contract(self, other: Tensor) -> sp.Expr:
      self._require_same_shape(other)
      rows, columns = self.shape
      return sum(
         self.component(row, col) * other.component(row, col)
         for row in range(rows)
         for col in range(columns)
      )

   def scaled(self, scalar: sp.Expr | int | float) -> Tensor:
      scalar = _as_expr(scalar)
      return Tensor(*(row.scaled(scalar) for row in self.rows))

   def contract_with(self, vector: Vector) -> Vector:
      if self.shape[1] != vector.dimension:
         raise ValueError("tensor/vector dimensions are incompatible")
      return Vector(*(row.dot(vector) for row in self.rows))

   def transpose(self) -> Tensor:
      return Tensor(*(self.column(index) for index in range(self.shape[1])))

   def symmetric(self) -> Tensor:
      if self.shape[0] != self.shape[1]:
         raise ValueError("symmetric part requires a square tensor")
      return (self + self.transpose()) / 2

   def deviatoric(self) -> Tensor:
      dimension = self.shape[0]
      if self.shape[1] != dimension:
         raise ValueError("deviatoric part requires a square tensor")
      return self - Tensor.identity(dimension, self.trace() / dimension)

   def plus(self, other: Tensor) -> Tensor:
      self._require_same_shape(other)
      return Tensor(*(left + right for left, right in zip(self.rows, other.rows, strict=True)))

   def minus(self, other: Tensor) -> Tensor:
      self._require_same_shape(other)
      return Tensor(*(left - right for left, right in zip(self.rows, other.rows, strict=True)))


@dataclass(frozen=True)
class MetricTensor:
   """Covariant and contravariant forms of one nondegenerate metric."""

   covariant: Tensor
   contravariant: Tensor

   def __post_init__(self) -> None:
      covariant_shape = self.covariant.shape
      if covariant_shape[0] != covariant_shape[1]:
         raise ValueError("a metric must be square")
      if self.contravariant.shape != covariant_shape:
         raise ValueError("covariant and contravariant metric shapes differ")

   @property
   def dimension(self) -> int:
      return self.covariant.shape[0]

   def lower(self, vector: Vector) -> Vector:
      return self.covariant.contract_with(vector)

   def raise_covector(self, covector: Vector) -> Vector:
      return self.contravariant.contract_with(covector)

   def norm_squared(self, vector: Vector) -> sp.Expr:
      return vector.dot(self.lower(vector))

   def inverse_identity_residual(self) -> Tensor:
      dimension = self.dimension
      product = Tensor.from_rows(tuple(
         tuple(
            sum(
               self.covariant.component(row, contracted)
               * self.contravariant.component(contracted, column)
               for contracted in range(dimension)
            )
            for column in range(dimension)
         )
         for row in range(dimension)
      ))
      return product - Tensor.identity(dimension)

   def indexed_covariant(self) -> IndexedTensor:
      return IndexedTensor(
         self.covariant,
         (IndexPosition.COVARIANT, IndexPosition.COVARIANT),
      )

   def indexed_contravariant(self) -> IndexedTensor:
      return IndexedTensor(
         self.contravariant,
         (IndexPosition.CONTRAVARIANT, IndexPosition.CONTRAVARIANT),
      )


class IndexPosition(Enum):
   COVARIANT = "down"
   CONTRAVARIANT = "up"

   @property
   def opposite(self) -> IndexPosition:
      return (
         IndexPosition.CONTRAVARIANT
         if self is IndexPosition.COVARIANT
         else IndexPosition.COVARIANT
      )


@dataclass(frozen=True)
class IndexedVector:
   """A vector with an explicit covariant or contravariant index."""

   values: Vector
   position: IndexPosition

   @property
   def dimension(self) -> int:
      return self.values.dimension

   def component(self, index: int) -> sp.Expr:
      return self.values.component(index)

   def as_tuple(self) -> tuple[sp.Expr, ...]:
      return self.values.as_tuple()

   def __add__(self, other: IndexedVector) -> IndexedVector:
      if self.position is not other.position:
         raise ValueError("vector addition requires equal index positions")
      return IndexedVector(self.values + other.values, self.position)

   def __sub__(self, other: IndexedVector) -> IndexedVector:
      if self.position is not other.position:
         raise ValueError("vector subtraction requires equal index positions")
      return IndexedVector(self.values - other.values, self.position)

   def __neg__(self) -> IndexedVector:
      return IndexedVector(-self.values, self.position)

   def __mul__(self, scalar: sp.Expr | int | float) -> IndexedVector:
      return IndexedVector(self.values * scalar, self.position)

   def __rmul__(self, scalar: sp.Expr | int | float) -> IndexedVector:
      return self * scalar

   def __truediv__(self, scalar: sp.Expr | int | float) -> IndexedVector:
      return IndexedVector(self.values / scalar, self.position)

   def raise_index(self, metric: MetricTensor) -> IndexedVector:
      if self.position is not IndexPosition.COVARIANT:
         raise ValueError("only a covariant vector index can be raised")
      if self.dimension != metric.dimension:
         raise ValueError("vector and metric dimensions differ")
      return IndexedVector(
         metric.raise_covector(self.values), IndexPosition.CONTRAVARIANT
      )

   def lower_index(self, metric: MetricTensor) -> IndexedVector:
      if self.position is not IndexPosition.CONTRAVARIANT:
         raise ValueError("only a contravariant vector index can be lowered")
      if self.dimension != metric.dimension:
         raise ValueError("vector and metric dimensions differ")
      return IndexedVector(metric.lower(self.values), IndexPosition.COVARIANT)

   def contract(self, other: IndexedVector) -> sp.Expr:
      if self.dimension != other.dimension:
         raise ValueError("vector dimensions differ")
      if self.position is other.position:
         raise ValueError("vector contraction requires one up and one down index")
      return self.values.dot(other.values)

   def outer(self, other: IndexedVector) -> IndexedTensor:
      return IndexedTensor(
         self.values.outer(other.values), (self.position, other.position)
      )


@dataclass(frozen=True)
class IndexedTensor:
   """A rank-two tensor with one position recorded for each index."""

   values: Tensor
   positions: tuple[IndexPosition, IndexPosition]

   def __post_init__(self) -> None:
      if len(self.positions) != 2:
         raise ValueError("a rank-two tensor requires two index positions")

   @property
   def shape(self) -> tuple[int, int]:
      return self.values.shape

   def component(self, first: int, second: int) -> sp.Expr:
      return self.values.component(first, second)

   def as_rows(self) -> tuple[tuple[sp.Expr, ...], ...]:
      return self.values.as_rows()

   def __add__(self, other: IndexedTensor) -> IndexedTensor:
      if self.positions != other.positions:
         raise ValueError("tensor addition requires equal index positions")
      return IndexedTensor(self.values + other.values, self.positions)

   def __sub__(self, other: IndexedTensor) -> IndexedTensor:
      if self.positions != other.positions:
         raise ValueError("tensor subtraction requires equal index positions")
      return IndexedTensor(self.values - other.values, self.positions)

   def __neg__(self) -> IndexedTensor:
      return IndexedTensor(-self.values, self.positions)

   def __mul__(self, scalar: sp.Expr | int | float) -> IndexedTensor:
      return IndexedTensor(self.values * scalar, self.positions)

   def __rmul__(self, scalar: sp.Expr | int | float) -> IndexedTensor:
      return self * scalar

   def __truediv__(self, scalar: sp.Expr | int | float) -> IndexedTensor:
      return IndexedTensor(self.values / scalar, self.positions)

   def raise_index(self, metric: MetricTensor, index: int) -> IndexedTensor:
      return self._move_index(metric, index, IndexPosition.CONTRAVARIANT)

   def lower_index(self, metric: MetricTensor, index: int) -> IndexedTensor:
      return self._move_index(metric, index, IndexPosition.COVARIANT)

   def _move_index(
      self,
      metric: MetricTensor,
      index: int,
      target: IndexPosition,
   ) -> IndexedTensor:
      if index not in (0, 1):
         raise ValueError("rank-two tensor index must be 0 or 1")
      if self.positions[index] is target:
         raise ValueError("tensor index already has the requested position")
      if self.shape != (metric.dimension, metric.dimension):
         raise ValueError("tensor and metric dimensions differ")
      transform = (
         metric.contravariant
         if target is IndexPosition.CONTRAVARIANT
         else metric.covariant
      )
      dimension = metric.dimension
      if index == 0:
         moved = Tensor.from_rows(tuple(
            tuple(
               sum(
                  transform.component(row, contracted)
                  * self.values.component(contracted, column)
                  for contracted in range(dimension)
               )
               for column in range(dimension)
            )
            for row in range(dimension)
         ))
      else:
         moved = Tensor.from_rows(tuple(
            tuple(
               sum(
                  transform.component(column, contracted)
                  * self.values.component(row, contracted)
                  for contracted in range(dimension)
               )
               for column in range(dimension)
            )
            for row in range(dimension)
         ))
      positions = list(self.positions)
      positions[index] = target
      return IndexedTensor(moved, tuple(positions))

   def contract_vector(self, vector: IndexedVector, index: int) -> IndexedVector:
      if index not in (0, 1):
         raise ValueError("rank-two tensor index must be 0 or 1")
      if self.positions[index] is vector.position:
         raise ValueError("contraction requires opposite index positions")
      if self.shape[index] != vector.dimension:
         raise ValueError("tensor and vector dimensions differ")
      dimension = vector.dimension
      remaining = 1 - index
      if index == 0:
         values = Vector(*(
            sum(
               self.values.component(contracted, component)
               * vector.component(contracted)
               for contracted in range(dimension)
            )
            for component in range(self.shape[1])
         ))
      else:
         values = Vector(*(
            sum(
               self.values.component(component, contracted)
               * vector.component(contracted)
               for contracted in range(dimension)
            )
            for component in range(self.shape[0])
         ))
      return IndexedVector(values, self.positions[remaining])

   def trace(self) -> sp.Expr:
      if self.shape[0] != self.shape[1]:
         raise ValueError("trace requires a square tensor")
      if self.positions[0] is self.positions[1]:
         raise ValueError("an invariant trace requires one up and one down index")
      return self.values.trace()

   def double_contract(self, other: IndexedTensor) -> sp.Expr:
      if self.shape != other.shape:
         raise ValueError("tensor shapes differ")
      if any(
         left is right
         for left, right in zip(self.positions, other.positions, strict=True)
      ):
         raise ValueError(
            "full tensor contraction requires opposite positions for each index"
         )
      return self.values.double_contract(other.values)


class Vector3(IndexedVector):
   def __init__(
      self,
      *components: sp.Expr | int | float,
      position: IndexPosition,
   ) -> None:
      if len(components) != 3:
         raise ValueError("Vector3 requires exactly three components")
      super().__init__(Vector(*components), position)


class Vector4(IndexedVector):
   def __init__(
      self,
      *components: sp.Expr | int | float,
      position: IndexPosition,
   ) -> None:
      if len(components) != 4:
         raise ValueError("Vector4 requires exactly four components")
      super().__init__(Vector(*components), position)


class Tensor3(IndexedTensor):
   def __init__(
      self,
      rows: tuple[tuple[sp.Expr | int | float, ...], ...],
      positions: tuple[IndexPosition, IndexPosition],
   ) -> None:
      values = Tensor.from_rows(rows)
      if values.shape != (3, 3):
         raise ValueError("Tensor3 requires a 3x3 component array")
      super().__init__(values, positions)


class Tensor4(IndexedTensor):
   def __init__(
      self,
      rows: tuple[tuple[sp.Expr | int | float, ...], ...],
      positions: tuple[IndexPosition, IndexPosition],
   ) -> None:
      values = Tensor.from_rows(rows)
      if values.shape != (4, 4):
         raise ValueError("Tensor4 requires a 4x4 component array")
      super().__init__(values, positions)


@dataclass(frozen=True)
class ScalarField:
   value: sp.Expr
   coordinate_gradient: Vector
   coordinate_hessian: Tensor | None = None
   time_derivative: sp.Expr | None = None

   def dt(self) -> sp.Expr:
      if self.time_derivative is None:
         raise ValueError("this scalar field has no time derivative")
      return self.time_derivative


@dataclass(frozen=True)
class VectorField:
   value: Vector
   coordinate_gradient: Tensor
   time_derivative: Vector | None = None

   def dt(self) -> Vector:
      if self.time_derivative is None:
         raise ValueError("this vector field has no time derivative")
      return self.time_derivative


@dataclass(frozen=True)
class OrthogonalOperators:
   """Continuous vector/tensor operators in an orthonormal physical basis.

   ``coordinate_derivatives[i, j]`` means ``d(field_i)/d(q_j)``. The
   discretization that supplies those derivatives remains outside this
   symbolic algebra layer.
   """

   scale_factors: tuple[sp.Expr, ...]
   coordinates: tuple[sp.Symbol, ...]

   def __post_init__(self) -> None:
      if not self.scale_factors or len(self.scale_factors) != len(self.coordinates):
         raise ValueError("scale factors and coordinates must have the same nonzero dimension")

   @property
   def dimension(self) -> int:
      return len(self.coordinates)

   @property
   def volume_scale(self) -> sp.Expr:
      return sp.prod(self.scale_factors)

   def gradient(self, field: ScalarField | VectorField) -> Vector | Tensor:
      if isinstance(field, ScalarField):
         if field.coordinate_gradient.dimension != self.dimension:
            raise ValueError("scalar-field and coordinate dimensions differ")
         return Vector(*(
            field.coordinate_gradient.component(axis) / self.scale_factors[axis]
            for axis in range(self.dimension)
         ))

      vector = field.value
      coordinate_derivatives = field.coordinate_gradient
      if vector.dimension != self.dimension or coordinate_derivatives.shape != (
         self.dimension,
         self.dimension,
      ):
         raise ValueError("vector-field and coordinate dimensions differ")
      h = self.scale_factors
      gradient: list[list[sp.Expr]] = [
         [sp.Integer(0) for _ in range(self.dimension)]
         for _ in range(self.dimension)
      ]
      for derivative_direction in range(self.dimension):
         for component_direction in range(self.dimension):
            if derivative_direction == component_direction:
               value = coordinate_derivatives.component(component_direction, derivative_direction) / h[derivative_direction]
               for basis in range(self.dimension):
                  if basis == derivative_direction:
                     continue
                  value += (
                     vector.component(basis)
                     * sp.diff(h[derivative_direction], self.coordinates[basis])
                     / (h[derivative_direction] * h[basis])
                  )
            else:
               value = h[component_direction] / h[derivative_direction] * (
                  coordinate_derivatives.component(component_direction, derivative_direction) / h[component_direction]
                  - vector.component(component_direction)
                  * sp.diff(h[component_direction], self.coordinates[derivative_direction])
                  / (h[component_direction] ** 2)
               )
            gradient[derivative_direction][component_direction] = sp.simplify(value)
      return Tensor.from_rows(tuple(tuple(row) for row in gradient))

   def curl(self, field: VectorField) -> Vector:
      if self.dimension != 3 or field.value.dimension != 3:
         raise ValueError("curl is defined here only for three-dimensional spatial fields")
      vector = field.value
      coordinate_derivatives = field.coordinate_gradient
      h = self.scale_factors

      def scaled_derivative(component: int, derivative_direction: int) -> sp.Expr:
         return (
            h[component] * coordinate_derivatives.component(component, derivative_direction)
            + vector.component(component) * sp.diff(h[component], self.coordinates[derivative_direction])
         )

      return Vector(
         sp.simplify((scaled_derivative(2, 1) - scaled_derivative(1, 2)) / (h[1] * h[2])),
         sp.simplify((scaled_derivative(0, 2) - scaled_derivative(2, 0)) / (h[2] * h[0])),
         sp.simplify((scaled_derivative(1, 0) - scaled_derivative(0, 1)) / (h[0] * h[1])),
      )

   def divergence(self, field: VectorField) -> sp.Expr:
      vector = field.value
      coordinate_derivatives = field.coordinate_gradient
      divergence = sp.Integer(0)
      for axis in range(self.dimension):
         transverse_area = sp.simplify(self.volume_scale / self.scale_factors[axis])
         divergence += transverse_area * coordinate_derivatives.component(axis, axis)
         divergence += vector.component(axis) * sp.diff(transverse_area, self.coordinates[axis])
      return sp.simplify(divergence / self.volume_scale)

   def laplacian(self, field: ScalarField) -> sp.Expr:
      if field.coordinate_hessian is None:
         raise ValueError("the scalar Laplacian requires a coordinate Hessian")
      if field.coordinate_gradient.dimension != self.dimension or field.coordinate_hessian.shape != (
         self.dimension,
         self.dimension,
      ):
         raise ValueError("scalar-field and coordinate dimensions differ")
      coordinate_gradient = field.coordinate_gradient
      coordinate_hessian = field.coordinate_hessian
      laplacian = sp.Integer(0)
      for axis in range(self.dimension):
         coefficient = sp.simplify(self.volume_scale / self.scale_factors[axis] ** 2)
         laplacian += coefficient * coordinate_hessian.component(axis, axis)
         laplacian += coordinate_gradient.component(axis) * sp.diff(coefficient, self.coordinates[axis])
      return sp.simplify(laplacian / self.volume_scale)

   def strain_rate(self, field: VectorField) -> Tensor:
      gradient = self.gradient(field)
      if not isinstance(gradient, Tensor):
         raise TypeError("a vector gradient must be a tensor")
      return gradient.symmetric()

   def tensor_divergence_connection(self, tensor: Tensor) -> Vector:
      """Connection terms complementing the finite-volume tensor divergence."""
      sources: list[sp.Expr] = []
      for component in range(self.dimension):
         source = sp.Integer(0)
         h_component = self.scale_factors[component]
         q_component = self.coordinates[component]
         for normal in range(self.dimension):
            h_normal = self.scale_factors[normal]
            q_normal = self.coordinates[normal]
            source += tensor.component(normal, component) * sp.diff(h_component, q_normal) / (h_component * h_normal)
            source -= tensor.component(normal, normal) * sp.diff(h_normal, q_component) / (h_component * h_normal)
         sources.append(sp.trigsimp(sp.simplify(source)))
      return Vector(*sources)


def replace_coordinates(
   values: tuple[sp.Expr, ...],
   source_coordinates: tuple[sp.Symbol, ...],
   runtime_coordinates: tuple[sp.Symbol, ...],
) -> tuple[sp.Expr, ...]:
   if len(source_coordinates) != len(runtime_coordinates):
      raise ValueError("source/runtime coordinate tuple length mismatch")
   substitutions = {
      source: runtime
      for source, runtime in zip(source_coordinates, runtime_coordinates, strict=True)
   }
   return tuple(sp.simplify(value.xreplace(substitutions)) for value in values)
