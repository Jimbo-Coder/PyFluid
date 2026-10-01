from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property

import sympy as sp

from .coordinates import OrthogonalFrame
from .tensor_algebra import Tensor, Vector


@dataclass(frozen=True)
class ReferenceMetricGeometry:
   """Reference-metric tensor algebra for one orthogonal coordinate frame.

   Vector and tensor inputs carrying an ``orthonormal`` suffix use physical
   components in the local unit-vector basis.  Mixed tensors use the Valencia
   flux convention ``flux[component, normal]``, with a covariant first index
   and contravariant second index.  This convention is explicit because it is
   not interchangeable with the transposed Newtonian stress convention when
   the flux is nonsymmetric.

   The class owns continuum geometry only.  It does not choose a finite-volume
   quadrature, source centering, executable dimension, or runtime method.
   """

   frame: OrthogonalFrame

   def __post_init__(self) -> None:
      if len(self.frame.coordinates) != 3 or len(self.frame.scale_factors) != 3:
         raise ValueError("reference-metric geometry requires a three-dimensional frame")
      if any(scale == 0 for scale in self.frame.scale_factors):
         raise ValueError("reference-metric scale factors must be nonzero on the regular chart")

   @property
   def coordinates(self) -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
      return self.frame.coordinates

   @property
   def scale_factors(self) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
      return self.frame.scale_factors

   @cached_property
   def sqrt_determinant(self) -> sp.Expr:
      return sp.simplify(self.frame.volume_scale)

   @cached_property
   def covariant_metric(self) -> Tensor:
      return Tensor.from_rows(tuple(
         tuple(
            self.scale_factors[row] ** 2 if row == column else sp.Integer(0)
            for column in range(3)
         )
         for row in range(3)
      ))

   @cached_property
   def contravariant_metric(self) -> Tensor:
      return Tensor.from_rows(tuple(
         tuple(
            1 / self.scale_factors[row] ** 2 if row == column else sp.Integer(0)
            for column in range(3)
         )
         for row in range(3)
      ))

   @cache
   def christoffel(self, upper: int, lower_first: int, lower_second: int) -> sp.Expr:
      """Return the Levi-Civita connection of the reference metric."""
      for index in (upper, lower_first, lower_second):
         if index < 0 or index >= 3:
            raise ValueError(f"reference-metric index {index} is outside [0, 3)")
      metric = self.covariant_metric
      inverse = self.contravariant_metric
      value = sp.Rational(1, 2) * sum(
         inverse.component(upper, contracted)
         * (
            sp.diff(metric.component(contracted, lower_second), self.coordinates[lower_first])
            + sp.diff(metric.component(contracted, lower_first), self.coordinates[lower_second])
            - sp.diff(metric.component(lower_first, lower_second), self.coordinates[contracted])
         )
         for contracted in range(3)
      )
      return sp.trigsimp(sp.simplify(value))

   def orthonormal_vector_to_coordinate(
      self, vector: Vector, *, covariant: bool
   ) -> Vector:
      self._require_vector3(vector)
      return Vector(*(
         vector.component(axis) * self.scale_factors[axis]
         if covariant
         else vector.component(axis) / self.scale_factors[axis]
         for axis in range(3)
      ))

   def coordinate_vector_to_orthonormal(
      self, vector: Vector, *, covariant: bool
   ) -> Vector:
      self._require_vector3(vector)
      return Vector(*(
         vector.component(axis) / self.scale_factors[axis]
         if covariant
         else vector.component(axis) * self.scale_factors[axis]
         for axis in range(3)
      ))

   def orthonormal_mixed_to_coordinate(self, flux: Tensor) -> Tensor:
      """Convert ``flux[lower component, upper normal]`` to coordinate basis."""
      self._require_tensor3(flux)
      return Tensor.from_rows(tuple(
         tuple(
            flux.component(component, normal)
            * self.scale_factors[component]
            / self.scale_factors[normal]
            for normal in range(3)
         )
         for component in range(3)
      ))

   def coordinate_mixed_to_orthonormal(self, flux: Tensor) -> Tensor:
      """Convert ``flux[lower component, upper normal]`` to unit-vector basis."""
      self._require_tensor3(flux)
      return Tensor.from_rows(tuple(
         tuple(
            flux.component(component, normal)
            * self.scale_factors[normal]
            / self.scale_factors[component]
            for normal in range(3)
         )
         for component in range(3)
      ))

   def coordinate_vector_divergence(self, vector: Vector) -> sp.Expr:
      """Return ``hat D_i vector^i`` for contravariant coordinate components."""
      self._require_vector3(vector)
      result = sum(
         sp.diff(self.sqrt_determinant * vector.component(normal), coordinate)
         for normal, coordinate in enumerate(self.coordinates)
      ) / self.sqrt_determinant
      return sp.trigsimp(sp.simplify(result))

   def orthonormal_vector_divergence(self, vector: Vector) -> sp.Expr:
      coordinate = self.orthonormal_vector_to_coordinate(vector, covariant=False)
      return self.coordinate_vector_divergence(coordinate)

   def orthonormal_vector_transport_divergence(self, vector: Vector) -> sp.Expr:
      """Derivative part suitable for direction-wise orthonormal face fluxes."""
      self._require_vector3(vector)
      return sp.simplify(sum(
         sp.diff(vector.component(normal), self.coordinates[normal])
         / self.scale_factors[normal]
         for normal in range(3)
      ))

   def orthonormal_vector_geometry_divergence(self, vector: Vector) -> sp.Expr:
      """Analytic reference-geometry part not present in transport derivatives."""
      return sp.trigsimp(sp.simplify(
         self.orthonormal_vector_divergence(vector)
         - self.orthonormal_vector_transport_divergence(vector)
      ))

   def coordinate_mixed_divergence(self, flux: Tensor) -> Vector:
      """Return ``hat D_i flux_j^i`` in covariant coordinate components."""
      self._require_tensor3(flux)
      derivatives = tuple(
         Tensor.from_rows(tuple(
            tuple(
               sp.diff(flux.component(component, normal), coordinate)
               for normal in range(3)
            )
            for component in range(3)
         ))
         for coordinate in self.coordinates
      )
      return self.coordinate_mixed_divergence_from_derivatives(flux, derivatives)

   def coordinate_mixed_divergence_from_derivatives(
      self,
      flux: Tensor,
      coordinate_derivatives: tuple[Tensor, Tensor, Tensor],
   ) -> Vector:
      """Mixed divergence from supplied coordinate-basis component jets."""
      self._require_tensor3(flux)
      self._require_tensor_derivatives(coordinate_derivatives)
      values: list[sp.Expr] = []
      for component in range(3):
         value = sp.Integer(0)
         for normal in range(3):
            value += coordinate_derivatives[normal].component(component, normal)
            for contracted in range(3):
               value += (
                  self.christoffel(normal, normal, contracted)
                  * flux.component(component, contracted)
               )
               value -= (
                  self.christoffel(contracted, normal, component)
                  * flux.component(contracted, normal)
               )
         values.append(sp.trigsimp(sp.simplify(value)))
      return Vector(*values)

   def orthonormal_mixed_divergence(self, flux: Tensor) -> Vector:
      self._require_tensor3(flux)
      derivatives = tuple(
         Tensor.from_rows(tuple(
            tuple(
               sp.diff(flux.component(component, normal), coordinate)
               for normal in range(3)
            )
            for component in range(3)
         ))
         for coordinate in self.coordinates
      )
      return self.orthonormal_mixed_divergence_from_derivatives(
         flux, derivatives
      )

   def orthonormal_mixed_divergence_from_derivatives(
      self,
      flux: Tensor,
      coordinate_derivatives: tuple[Tensor, Tensor, Tensor],
   ) -> Vector:
      """Mixed divergence from jets of orthonormal flux components."""
      self._require_tensor3(flux)
      self._require_tensor_derivatives(coordinate_derivatives)
      coordinate_flux = self.orthonormal_mixed_to_coordinate(flux)
      coordinate_flux_derivatives = tuple(
         Tensor.from_rows(tuple(
            tuple(
               sp.diff(
                  self.scale_factors[component]
                  / self.scale_factors[normal],
                  self.coordinates[derivative],
               ) * flux.component(component, normal)
               + self.scale_factors[component]
               / self.scale_factors[normal]
               * coordinate_derivatives[derivative].component(component, normal)
               for normal in range(3)
            )
            for component in range(3)
         ))
         for derivative in range(3)
      )
      coordinate_divergence = self.coordinate_mixed_divergence_from_derivatives(
         coordinate_flux, coordinate_flux_derivatives
      )
      return self.coordinate_vector_to_orthonormal(
         coordinate_divergence, covariant=True
      )

   def orthonormal_mixed_transport_divergence(self, flux: Tensor) -> Vector:
      """Direction-wise derivative part for ``flux[component, normal]``."""
      self._require_tensor3(flux)
      return Vector(*(
         sp.simplify(sum(
            sp.diff(flux.component(component, normal), self.coordinates[normal])
            / self.scale_factors[normal]
            for normal in range(3)
         ))
         for component in range(3)
      ))

   def orthonormal_mixed_geometry_divergence(self, flux: Tensor) -> Vector:
      full = self.orthonormal_mixed_divergence(flux)
      transport = self.orthonormal_mixed_transport_divergence(flux)
      return Vector(*(
         sp.trigsimp(sp.simplify(full.component(component) - transport.component(component)))
         for component in range(3)
      ))

   def orthonormal_mixed_basis_divergence(self, flux: Tensor) -> Vector:
      """Connection correction after row-wise physical-volume FV divergence.

      A mapped finite-volume divergence treats each momentum component as a
      scalar label and already includes the contravariant flux-normal geometry.
      This algebraic remainder accounts only for rotation of the covariant
      momentum basis.  Callers may remove an isotropic pressure tensor before
      evaluating it to obtain an exactly pressure-balanced discretization.
      """
      self._require_tensor3(flux)
      mixed_geometry = self.orthonormal_mixed_geometry_divergence(flux)
      row_geometry = Vector(*(
         self.orthonormal_vector_geometry_divergence(flux.row(component))
         for component in range(3)
      ))
      return Vector(*(
         sp.trigsimp(sp.simplify(
            mixed_geometry.component(component)
            - row_geometry.component(component)
         ))
         for component in range(3)
      ))

   @staticmethod
   def _require_vector3(vector: Vector) -> None:
      if vector.dimension != 3:
         raise ValueError("reference-metric vectors must have three components")

   @staticmethod
   def _require_tensor3(tensor: Tensor) -> None:
      if tensor.shape != (3, 3):
         raise ValueError("reference-metric tensors must be 3x3")

   @classmethod
   def _require_tensor_derivatives(
      cls, coordinate_derivatives: tuple[Tensor, Tensor, Tensor]
   ) -> None:
      if len(coordinate_derivatives) != 3:
         raise ValueError("reference-metric tensor jets need three derivatives")
      for derivative in coordinate_derivatives:
         cls._require_tensor3(derivative)
