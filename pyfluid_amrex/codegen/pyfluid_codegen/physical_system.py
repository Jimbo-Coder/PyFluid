from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .eos_equations import AnalyticEquationOfState
from .tensor_algebra import OrthogonalOperators, Tensor, Vector, VectorField


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
