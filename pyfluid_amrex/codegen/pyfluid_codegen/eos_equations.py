from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .kernel_equations import ScalarKernel


@dataclass(frozen=True)
class AnalyticEquationOfState:
   """Analytic Newtonian EOS defined by one pressure relation P(rho, u)."""

   name: str
   density: sp.Symbol
   internal_energy_density: sp.Symbol
   pressure_expression: sp.Expr
   parameters: tuple[sp.Symbol, ...]
   barotropic_energy_density: sp.Expr | None = None

   @property
   def is_barotropic(self) -> bool:
      return not self.pressure_expression.has(self.internal_energy_density)

   def pressure(self) -> sp.Expr:
      return sp.simplify(self.pressure_expression)

   def internal_energy_from_pressure(self, pressure: sp.Symbol) -> sp.Expr:
      if self.is_barotropic:
         if self.barotropic_energy_density is not None:
            return sp.simplify(self.barotropic_energy_density)
         specific_energy = sp.integrate(
            self.pressure_expression / self.density**2,
            self.density,
         )
         return sp.simplify(self.density * specific_energy)

      solutions = sp.solve(
         sp.Eq(pressure, self.pressure_expression),
         self.internal_energy_density,
         dict=False,
      )
      if len(solutions) != 1:
         raise ValueError(
            f"{self.name} must have one analytic internal-energy inverse; got {solutions}"
         )
      return sp.simplify(solutions[0])

   def sound_speed_squared(self) -> sp.Expr:
      """Return (dP/drho)_s for P(rho, u), with u the energy density."""
      pressure = self.pressure_expression
      if self.is_barotropic:
         return sp.simplify(sp.diff(pressure, self.density))
      return sp.simplify(
         sp.diff(pressure, self.density)
         + sp.diff(pressure, self.internal_energy_density)
         * (self.internal_energy_density + pressure)
         / self.density
      )

   def sound_speed_squared_from_pressure(self, pressure: sp.Symbol) -> sp.Expr:
      expression = self.sound_speed_squared()
      if not self.is_barotropic:
         expression = expression.subs(
            self.internal_energy_density,
            self.internal_energy_from_pressure(pressure),
         )
      return sp.simplify(expression)

   def kernels(self, pressure: sp.Symbol) -> tuple[ScalarKernel, ...]:
      if self.is_barotropic:
         operations = (
            ("cold_pressure", self.pressure()),
            ("cold_energy_density", self.internal_energy_from_pressure(pressure)),
            ("sound_speed_squared", self.sound_speed_squared_from_pressure(pressure)),
         )
      else:
         operations = (
            ("pressure_from_density_energy", self.pressure()),
            (
               "internal_energy_density_from_pressure",
               self.internal_energy_from_pressure(pressure),
            ),
            ("sound_speed_squared", self.sound_speed_squared_from_pressure(pressure)),
         )

      argument_order = (
         self.density,
         self.internal_energy_density,
         pressure,
         *self.parameters,
      )
      return tuple(
         ScalarKernel(
            name=f"{self.name}_{operation_name}",
            arguments=tuple(
               str(symbol) for symbol in argument_order if symbol in expression.free_symbols
            ),
            expression=expression,
         )
         for operation_name, expression in operations
      )


def analytic_equations_of_state() -> tuple[AnalyticEquationOfState, ...]:
   rho, internal_energy_density = sp.symbols("rho internal_energy_density", positive=True)
   gamma, polytropic_k, polytropic_gamma, thermal_gamma = sp.symbols(
      "gamma polytropic_k polytropic_gamma thermal_gamma",
      positive=True,
   )

   ideal_gas = AnalyticEquationOfState(
      name="ideal_gas",
      density=rho,
      internal_energy_density=internal_energy_density,
      pressure_expression=(gamma - 1) * internal_energy_density,
      parameters=(gamma,),
   )

   cold_pressure = polytropic_k * rho**polytropic_gamma
   cold_energy = cold_pressure / (polytropic_gamma - 1)
   polytrope = AnalyticEquationOfState(
      name="polytrope",
      density=rho,
      internal_energy_density=internal_energy_density,
      pressure_expression=cold_pressure,
      parameters=(polytropic_k, polytropic_gamma),
      barotropic_energy_density=cold_energy,
   )
   hybrid_polytrope = AnalyticEquationOfState(
      name="hybrid_polytrope",
      density=rho,
      internal_energy_density=internal_energy_density,
      pressure_expression=(
         cold_pressure
         + (thermal_gamma - 1) * (internal_energy_density - cold_energy)
      ),
      parameters=(polytropic_k, polytropic_gamma, thermal_gamma),
   )
   return ideal_gas, polytrope, hybrid_polytrope


def eos_kernels() -> tuple[ScalarKernel, ...]:
   pressure = sp.Symbol("pressure", positive=True)
   return tuple(
      kernel
      for equation_of_state in analytic_equations_of_state()
      for kernel in equation_of_state.kernels(pressure)
   )
