from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .tensor_algebra import HydroState3, MHDState3


@dataclass(frozen=True)
class KernelEquations:
   name: str
   arguments: tuple[str, ...]
   outputs: tuple[str, ...]
   expressions: tuple[sp.Expr, ...]


def hydro_flux_equations(direction: int) -> KernelEquations:
   rho, mom1, mom2, mom3, energy, pressure = sp.symbols(
      "state.rho state.mom1 state.mom2 state.mom3 state.energy pressure"
   )
   state = HydroState3.from_symbols(rho, mom1, mom2, mom3, energy)
   normal_velocity = state.velocity().component(direction)
   momentum_flux = state.momentum_flux(direction, pressure)
   energy_flux = state.energy_flux(direction, pressure)
   return KernelEquations(
      name=f"hydro_flux_orthonormal_dir{direction}",
      arguments=("state", "pressure"),
      outputs=("rho", "mom1", "mom2", "mom3", "energy"),
      expressions=(
         rho * normal_velocity,
         momentum_flux.x,
         momentum_flux.y,
         momentum_flux.z,
         energy_flux,
      ),
   )


def mhd_flux_equations(direction: int) -> KernelEquations:
   rho, mom1, mom2, mom3, energy, pressure, b1, b2, b3 = sp.symbols(
      "state.rho state.mom1 state.mom2 state.mom3 state.energy pressure state.b1 state.b2 state.b3"
   )
   state = MHDState3.from_symbols(rho, mom1, mom2, mom3, energy, b1, b2, b3)
   velocity = state.velocity()
   normal_velocity = velocity.component(direction)
   momentum_flux = state.mhd_momentum_flux(direction, pressure)
   energy_flux = state.mhd_energy_flux(direction, pressure)
   magnetic_flux = state.mhd_magnetic_flux(direction)
   return KernelEquations(
      name=f"mhd_flux_orthonormal_dir{direction}",
      arguments=("state", "pressure"),
      outputs=("rho", "mom1", "mom2", "mom3", "energy", "b1", "b2", "b3"),
      expressions=(
         rho * normal_velocity,
         momentum_flux.x,
         momentum_flux.y,
         momentum_flux.z,
         energy_flux,
         magnetic_flux.x,
         magnetic_flux.y,
         magnetic_flux.z,
      ),
   )
