from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .eos_equations import AnalyticEquationOfState
from .tensor_algebra import (
   IndexPosition,
   MetricTensor,
   OrthogonalOperators,
   Tensor,
   Tensor3,
   Tensor4,
   Vector,
   Vector3,
   Vector4,
   VectorField,
)


ConservedValue = sp.Expr | Vector
FluxValue = Vector | Tensor


def _zero_like(value: ConservedValue) -> ConservedValue:
   if isinstance(value, Vector):
      return Vector.zero(value.dimension)
   return sp.Integer(0)


@dataclass(frozen=True)
class ConservationEquation:
   """One coordinate-independent equation d_t U + div(F) = S."""

   conserved: ConservedValue
   flux: FluxValue
   source: ConservedValue

   @classmethod
   def source_free(cls, conserved: ConservedValue, flux: FluxValue) -> ConservationEquation:
      return cls(conserved=conserved, flux=flux, source=_zero_like(conserved))

   def flux_along(self, normal_direction: int) -> ConservedValue:
      if isinstance(self.flux, Tensor):
         return self.flux.column(normal_direction)
      return self.flux.component(normal_direction)


@dataclass(frozen=True)
class ConservationLaw:
   equations: tuple[ConservationEquation, ...]

   def flux_along(self, normal_direction: int) -> tuple[ConservedValue, ...]:
      return tuple(equation.flux_along(normal_direction) for equation in self.equations)


@dataclass(frozen=True)
class HydroState:
   rho: sp.Expr
   momenta: Vector
   energy: sp.Expr

   @classmethod
   def from_symbols(
      cls,
      rho: sp.Symbol,
      mom1: sp.Symbol,
      mom2: sp.Symbol,
      mom3: sp.Symbol,
      energy: sp.Symbol,
   ) -> HydroState:
      return cls(rho=rho, momenta=Vector(mom1, mom2, mom3), energy=energy)

   def velocity(self) -> Vector:
      return self.momenta / self.rho


@dataclass(frozen=True)
class MHDState(HydroState):
   magnetic: Vector

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
   ) -> MHDState:
      return cls(
         rho=rho,
         momenta=Vector(mom1, mom2, mom3),
         energy=energy,
         magnetic=Vector(b1, b2, b3),
      )

   def magnetic_energy_density(self) -> sp.Expr:
      return sp.Rational(1, 2) * self.magnetic.norm_squared()


@dataclass(frozen=True)
class Hydrodynamics:
   eos: AnalyticEquationOfState | None = None

   @property
   def is_magnetized(self) -> bool:
      return False

   @property
   def is_barotropic(self) -> bool:
      return self.eos is not None and self.eos.is_barotropic

   def conservation_law(self, state: HydroState, pressure: sp.Expr) -> ConservationLaw:
      velocity = state.velocity()
      dimension = velocity.dimension
      return ConservationLaw(
         equations=(
            ConservationEquation.source_free(state.rho, state.momenta),
            ConservationEquation.source_free(
               state.momenta,
               state.momenta.outer(velocity) + Tensor.identity(dimension, pressure),
            ),
            ConservationEquation.source_free(
               state.energy,
               (state.energy + pressure) * velocity,
            ),
         )
      )


@dataclass(frozen=True)
class Magnetohydrodynamics(Hydrodynamics):
   @property
   def is_magnetized(self) -> bool:
      return True

   def conservation_law(self, state: HydroState, pressure: sp.Expr) -> ConservationLaw:
      if not isinstance(state, MHDState):
         raise TypeError("magnetohydrodynamics requires an MHD state")
      velocity = state.velocity()
      dimension = velocity.dimension
      total_pressure = pressure + state.magnetic_energy_density()
      return ConservationLaw(
         equations=(
            ConservationEquation.source_free(state.rho, state.momenta),
            ConservationEquation.source_free(
               state.momenta,
               state.momenta.outer(velocity)
               + Tensor.identity(dimension, total_pressure)
               - state.magnetic.outer(state.magnetic),
            ),
            ConservationEquation.source_free(
               state.energy,
               (state.energy + total_pressure) * velocity
               - velocity.dot(state.magnetic) * state.magnetic,
            ),
            ConservationEquation.source_free(
               state.magnetic,
               state.magnetic.outer(velocity) - velocity.outer(state.magnetic),
            ),
         )
      )


@dataclass(frozen=True)
class ValenciaPrimitive:
   density: sp.Expr
   specific_internal_energy: sp.Expr
   pressure: sp.Expr
   velocity: Vector


@dataclass(frozen=True)
class ValenciaMetric:
   lapse: sp.Expr
   shift: Vector
   spatial_metric: MetricTensor
   sqrt_spatial_determinant: sp.Expr

   def spacetime_metric(self) -> MetricTensor:
      shift_covariant = self.spatial_metric.lower(self.shift)
      shift_squared = self.shift.dot(shift_covariant)
      covariant_rows = (
         (
            -self.lapse**2 + shift_squared,
            *shift_covariant.as_tuple(),
         ),
         *tuple(
            (
               shift_covariant.component(row),
               *self.spatial_metric.covariant.row(row).as_tuple(),
            )
            for row in range(3)
         ),
      )
      inverse_lapse_squared = 1 / self.lapse**2
      contravariant_rows = (
         (
            -inverse_lapse_squared,
            *(component * inverse_lapse_squared for component in self.shift),
         ),
         *tuple(
            (
               self.shift.component(row) * inverse_lapse_squared,
               *tuple(
                  self.spatial_metric.contravariant.component(row, column)
                  - self.shift.component(row)
                  * self.shift.component(column)
                  * inverse_lapse_squared
                  for column in range(3)
               ),
            )
            for row in range(3)
         ),
      )
      covariant = Tensor4(
         covariant_rows,
         (IndexPosition.COVARIANT, IndexPosition.COVARIANT),
      )
      contravariant = Tensor4(
         contravariant_rows,
         (IndexPosition.CONTRAVARIANT, IndexPosition.CONTRAVARIANT),
      )
      return MetricTensor(covariant.values, contravariant.values)

   def four_velocity(self, velocity: Vector) -> Vector4:
      velocity_squared = self.spatial_metric.norm_squared(velocity)
      lorentz_factor = 1 / sp.sqrt(1 - velocity_squared)
      transport_velocity = self.lapse * velocity - self.shift
      return Vector4(
         lorentz_factor / self.lapse,
         *(lorentz_factor * component / self.lapse for component in transport_velocity),
         position=IndexPosition.CONTRAVARIANT,
      )


@dataclass(frozen=True)
class ValenciaMetricDerivatives:
   lapse_gradient: Vector3
   shift_gradients: tuple[Vector3, Vector3, Vector3]
   spatial_metric_gradients: tuple[Tensor3, Tensor3, Tensor3]
   extrinsic_curvature: Tensor3

   def __post_init__(self) -> None:
      if self.lapse_gradient.position is not IndexPosition.COVARIANT:
         raise ValueError("the lapse gradient must carry a covariant index")
      if any(
         gradient.position is not IndexPosition.COVARIANT
         for gradient in self.shift_gradients
      ):
         raise ValueError("shift component gradients must carry covariant indices")
      if any(
         gradient.positions
         != (IndexPosition.COVARIANT, IndexPosition.COVARIANT)
         for gradient in self.spatial_metric_gradients
      ):
         raise ValueError("spatial-metric derivatives must be covariant tensors")
      if self.extrinsic_curvature.positions != (
         IndexPosition.COVARIANT,
         IndexPosition.COVARIANT,
      ):
         raise ValueError("extrinsic curvature must be covariant")


@dataclass(frozen=True)
class ValenciaConserved:
   density: sp.Expr
   momentum: Vector
   energy: sp.Expr


@dataclass(frozen=True)
class ValenciaSource:
   momentum: Vector
   energy: sp.Expr


def valencia_geometric_source(
   conserved: ValenciaConserved | ValenciaMHDConserved,
   stress_energy: Tensor4,
   metric: ValenciaMetric,
   derivatives: ValenciaMetricDerivatives,
) -> ValenciaSource:
   """Return the Valencia momentum and tau sources for one 3+1 geometry."""
   if stress_energy.positions != (
      IndexPosition.CONTRAVARIANT,
      IndexPosition.CONTRAVARIANT,
   ):
      raise ValueError("Valencia sources require contravariant stress energy")
   inverse_sqrt_determinant = 1 / metric.sqrt_spatial_determinant
   momentum_covariant = Vector3(
      *(component * inverse_sqrt_determinant for component in conserved.momentum),
      position=IndexPosition.COVARIANT,
   )
   momentum_contravariant = momentum_covariant.raise_index(metric.spatial_metric)
   eulerian_energy = (
      conserved.energy + conserved.density
   ) * inverse_sqrt_determinant
   spatial_stress = Tensor3(
      tuple(
         tuple(
            stress_energy.component(first + 1, second + 1)
            + metric.shift.component(first) * stress_energy.component(0, second + 1)
            + metric.shift.component(second) * stress_energy.component(first + 1, 0)
            + metric.shift.component(first)
            * metric.shift.component(second)
            * stress_energy.component(0, 0)
            for second in range(3)
         )
         for first in range(3)
      ),
      (IndexPosition.CONTRAVARIANT, IndexPosition.CONTRAVARIANT),
   )
   momentum_source = Vector(*(
      metric.sqrt_spatial_determinant
      * (
         -eulerian_energy * derivatives.lapse_gradient.component(direction)
         + sum(
            momentum_covariant.component(component)
            * derivatives.shift_gradients[component].component(direction)
            for component in range(3)
         )
         + metric.lapse
         * sp.Rational(1, 2)
         * spatial_stress.double_contract(
            derivatives.spatial_metric_gradients[direction]
         )
      )
      for direction in range(3)
   ))
   energy_source = metric.sqrt_spatial_determinant * (
      metric.lapse
      * spatial_stress.double_contract(derivatives.extrinsic_curvature)
      - momentum_contravariant.contract(derivatives.lapse_gradient)
   )
   return ValenciaSource(momentum=momentum_source, energy=energy_source)


@dataclass(frozen=True)
class ValenciaMHDPrimitive:
   density: sp.Expr
   specific_internal_energy: sp.Expr
   pressure: sp.Expr
   velocity: Vector
   magnetic: Vector


@dataclass(frozen=True)
class ValenciaMHDConserved:
   density: sp.Expr
   momentum: Vector
   energy: sp.Expr
   magnetic: Vector


@dataclass(frozen=True)
class ValenciaHydrodynamics:
   """Valencia RHD in a 3+1 metric using Eulerian three-velocity."""

   def primitive_to_conserved(
      self,
      primitive: ValenciaPrimitive,
      metric: ValenciaMetric,
   ) -> ValenciaConserved:
      velocity_squared = metric.spatial_metric.norm_squared(primitive.velocity)
      lorentz_factor = 1 / sp.sqrt(1 - velocity_squared)
      specific_enthalpy = (
         1
         + primitive.specific_internal_energy
         + primitive.pressure / primitive.density
      )
      density = (
         metric.sqrt_spatial_determinant
         * primitive.density
         * lorentz_factor
      )
      momentum = (
         metric.sqrt_spatial_determinant
         * primitive.density
         * specific_enthalpy
         * lorentz_factor**2
         * metric.spatial_metric.lower(primitive.velocity)
      )
      energy = metric.sqrt_spatial_determinant * (
         primitive.density * specific_enthalpy * lorentz_factor**2
         - primitive.pressure
         - primitive.density * lorentz_factor
      )
      return ValenciaConserved(density=density, momentum=momentum, energy=energy)

   def stress_energy(
      self,
      primitive: ValenciaPrimitive,
      metric: ValenciaMetric,
   ) -> Tensor4:
      specific_enthalpy = (
         1
         + primitive.specific_internal_energy
         + primitive.pressure / primitive.density
      )
      four_velocity = metric.four_velocity(primitive.velocity)
      spacetime_metric = metric.spacetime_metric().indexed_contravariant()
      stress_energy = (
         primitive.density * specific_enthalpy * four_velocity.outer(four_velocity)
         + primitive.pressure * spacetime_metric
      )
      return Tensor4(stress_energy.as_rows(), stress_energy.positions)

   def conservation_law(
      self,
      primitive: ValenciaPrimitive,
      metric: ValenciaMetric,
   ) -> ConservationLaw:
      conserved = self.primitive_to_conserved(primitive, metric)
      transport_velocity = metric.lapse * primitive.velocity - metric.shift
      pressure_transport = (
         metric.lapse * metric.sqrt_spatial_determinant * primitive.pressure
      )
      dimension = primitive.velocity.dimension
      return ConservationLaw(
         equations=(
            ConservationEquation.source_free(
               conserved.density,
               conserved.density * transport_velocity,
            ),
            ConservationEquation.source_free(
               conserved.momentum,
               conserved.momentum.outer(transport_velocity)
               + Tensor.identity(dimension, pressure_transport),
            ),
            ConservationEquation.source_free(
               conserved.energy,
               conserved.energy * transport_velocity
               + pressure_transport * primitive.velocity,
            ),
         )
      )


@dataclass(frozen=True)
class ValenciaMagnetohydrodynamics:
   """Ideal Valencia RMHD using Eulerian velocity and magnetic field."""

   def primitive_to_conserved(
      self,
      primitive: ValenciaMHDPrimitive,
      metric: ValenciaMetric,
   ) -> ValenciaMHDConserved:
      spatial_metric = metric.spatial_metric
      velocity_covariant = spatial_metric.lower(primitive.velocity)
      magnetic_covariant = spatial_metric.lower(primitive.magnetic)
      velocity_squared = primitive.velocity.dot(velocity_covariant)
      lorentz_factor = 1 / sp.sqrt(1 - velocity_squared)
      magnetic_squared = primitive.magnetic.dot(magnetic_covariant)
      alpha_b0 = lorentz_factor * primitive.magnetic.dot(velocity_covariant)
      comoving_magnetic_covariant = (
         magnetic_covariant / lorentz_factor
         + alpha_b0 * velocity_covariant
      )
      comoving_magnetic_squared = (
         magnetic_squared + alpha_b0**2
      ) / lorentz_factor**2
      specific_enthalpy = (
         1
         + primitive.specific_internal_energy
         + primitive.pressure / primitive.density
      )
      density = (
         metric.sqrt_spatial_determinant
         * primitive.density
         * lorentz_factor
      )
      fluid_enthalpy = (
         metric.sqrt_spatial_determinant
         * primitive.density
         * specific_enthalpy
         * lorentz_factor**2
      )
      total_enthalpy = fluid_enthalpy + metric.sqrt_spatial_determinant * (
         magnetic_squared + alpha_b0**2
      )
      momentum = (
         total_enthalpy * velocity_covariant
         - metric.sqrt_spatial_determinant
         * alpha_b0
         * comoving_magnetic_covariant
      )
      total_pressure = primitive.pressure + comoving_magnetic_squared / 2
      energy = (
         fluid_enthalpy
         - density
         + metric.sqrt_spatial_determinant
         * (magnetic_squared - total_pressure)
      )
      magnetic = metric.sqrt_spatial_determinant * primitive.magnetic
      return ValenciaMHDConserved(
         density=density,
         momentum=momentum,
         energy=energy,
         magnetic=magnetic,
      )

   def comoving_magnetic_four_vector(
      self,
      primitive: ValenciaMHDPrimitive,
      metric: ValenciaMetric,
   ) -> Vector4:
      velocity_covariant = metric.spatial_metric.lower(primitive.velocity)
      velocity_squared = primitive.velocity.dot(velocity_covariant)
      lorentz_factor = 1 / sp.sqrt(1 - velocity_squared)
      alpha_b0 = lorentz_factor * primitive.magnetic.dot(velocity_covariant)
      transport_velocity = metric.lapse * primitive.velocity - metric.shift
      return Vector4(
         alpha_b0 / metric.lapse,
         *(
            primitive.magnetic.component(index) / lorentz_factor
            + alpha_b0 * transport_velocity.component(index) / metric.lapse
            for index in range(3)
         ),
         position=IndexPosition.CONTRAVARIANT,
      )

   def stress_energy(
      self,
      primitive: ValenciaMHDPrimitive,
      metric: ValenciaMetric,
   ) -> Tensor4:
      velocity_covariant = metric.spatial_metric.lower(primitive.velocity)
      velocity_squared = primitive.velocity.dot(velocity_covariant)
      lorentz_factor_squared = 1 / (1 - velocity_squared)
      magnetic_covariant = metric.spatial_metric.lower(primitive.magnetic)
      alpha_b0 = sp.sqrt(lorentz_factor_squared) * primitive.magnetic.dot(
         velocity_covariant
      )
      comoving_magnetic_squared = (
         primitive.magnetic.dot(magnetic_covariant) + alpha_b0**2
      ) / lorentz_factor_squared
      specific_enthalpy = (
         1
         + primitive.specific_internal_energy
         + primitive.pressure / primitive.density
      )
      four_velocity = metric.four_velocity(primitive.velocity)
      comoving_magnetic = self.comoving_magnetic_four_vector(primitive, metric)
      spacetime_metric = metric.spacetime_metric().indexed_contravariant()
      stress_energy = (
         (primitive.density * specific_enthalpy + comoving_magnetic_squared)
         * four_velocity.outer(four_velocity)
         + (primitive.pressure + comoving_magnetic_squared / 2)
         * spacetime_metric
         - comoving_magnetic.outer(comoving_magnetic)
      )
      return Tensor4(stress_energy.as_rows(), stress_energy.positions)

   def conservation_law(
      self,
      primitive: ValenciaMHDPrimitive,
      metric: ValenciaMetric,
   ) -> ConservationLaw:
      conserved = self.primitive_to_conserved(primitive, metric)
      spatial_metric = metric.spatial_metric
      velocity_covariant = spatial_metric.lower(primitive.velocity)
      magnetic_covariant = spatial_metric.lower(primitive.magnetic)
      velocity_squared = primitive.velocity.dot(velocity_covariant)
      lorentz_factor = 1 / sp.sqrt(1 - velocity_squared)
      alpha_b0 = lorentz_factor * primitive.magnetic.dot(velocity_covariant)
      comoving_magnetic_covariant = (
         magnetic_covariant / lorentz_factor
         + alpha_b0 * velocity_covariant
      )
      comoving_magnetic_squared = (
         primitive.magnetic.dot(magnetic_covariant) + alpha_b0**2
      ) / lorentz_factor**2
      total_pressure = primitive.pressure + comoving_magnetic_squared / 2
      transport_velocity = metric.lapse * primitive.velocity - metric.shift
      dimension = primitive.velocity.dimension
      momentum_transport = conserved.momentum.outer(transport_velocity)
      magnetic_stress = comoving_magnetic_covariant.outer(
         primitive.magnetic / lorentz_factor
      )
      momentum_flux = momentum_transport + metric.lapse * metric.sqrt_spatial_determinant * (
         Tensor.identity(dimension, total_pressure) - magnetic_stress
      )
      energy_flux = (
         conserved.energy * transport_velocity
         + metric.lapse
         * metric.sqrt_spatial_determinant
         * (
            total_pressure * primitive.velocity
            - alpha_b0 * primitive.magnetic / lorentz_factor
         )
      )
      magnetic_flux = (
         conserved.magnetic.outer(transport_velocity)
         - transport_velocity.outer(conserved.magnetic)
      )
      return ConservationLaw(
         equations=(
            ConservationEquation.source_free(
               conserved.density,
               conserved.density * transport_velocity,
            ),
            ConservationEquation.source_free(conserved.momentum, momentum_flux),
            ConservationEquation.source_free(conserved.energy, energy_flux),
            ConservationEquation.source_free(conserved.magnetic, magnetic_flux),
         )
      )


@dataclass(frozen=True)
class NewtonianViscosity:
   dynamic_viscosity: sp.Expr

   def stress(self, velocity: VectorField, operators: OrthogonalOperators) -> Tensor:
      """Return tau = 2 mu [sym(grad(v)) - I div(v)/d]."""
      strain = operators.strain_rate(velocity)
      divergence = operators.divergence(velocity)
      dimension = velocity.value.dimension
      return 2 * self.dynamic_viscosity * (
         strain - Tensor.identity(dimension, divergence / dimension)
      )

   def energy_flux(self, velocity: Vector, stress: Tensor) -> Vector:
      return stress.contract_with(velocity)


def orthogonal_velocity_curvature_sources(
   operators: OrthogonalOperators,
   density: sp.Expr,
   velocity: Vector,
) -> tuple[sp.Expr, ...]:
   momentum_flux = density * velocity.outer(velocity)
   return tuple(-value for value in operators.tensor_divergence_connection(momentum_flux))


def orthogonal_mhd_stress_curvature_sources(
   operators: OrthogonalOperators,
   density: sp.Expr,
   velocity: Vector,
   magnetic: Vector,
) -> tuple[sp.Expr, ...]:
   stress = density * velocity.outer(velocity) - magnetic.outer(magnetic)
   return tuple(-value for value in operators.tensor_divergence_connection(stress))
