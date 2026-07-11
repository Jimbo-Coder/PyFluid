from __future__ import annotations

import sympy as sp

from .kernel_equations import StructKernel
from .physical_system import HydroState, Hydrodynamics, MHDState, Magnetohydrodynamics


def hydro_flux_equations(direction: int) -> StructKernel:
   rho, mom1, mom2, mom3, energy, pressure = sp.symbols(
      "state.rho state.mom1 state.mom2 state.mom3 state.energy pressure"
   )
   state = HydroState.from_symbols(rho, mom1, mom2, mom3, energy)
   mass_flux, momentum_flux, energy_flux = Hydrodynamics().conservation_law(
      state, pressure
   ).flux_along(direction)
   return StructKernel(
      name=f"hydro_flux_orthonormal_dir{direction}",
      return_type="HydroState",
      arguments=(("const HydroState&", "state"), ("amrex::Real", "pressure")),
      outputs=("rho", "mom1", "mom2", "mom3", "energy"),
      expressions=(
         mass_flux,
         *momentum_flux,
         energy_flux,
      ),
   )


def mhd_flux_equations(direction: int) -> StructKernel:
   rho, mom1, mom2, mom3, energy, pressure, b1, b2, b3 = sp.symbols(
      "state.rho state.mom1 state.mom2 state.mom3 state.energy pressure state.b1 state.b2 state.b3"
   )
   state = MHDState.from_symbols(rho, mom1, mom2, mom3, energy, b1, b2, b3)
   mass_flux, momentum_flux, energy_flux, magnetic_flux = Magnetohydrodynamics().conservation_law(
      state, pressure
   ).flux_along(direction)
   return StructKernel(
      name=f"mhd_flux_orthonormal_dir{direction}",
      return_type="MHDState",
      arguments=(("const MHDState&", "state"), ("amrex::Real", "pressure")),
      outputs=("rho", "mom1", "mom2", "mom3", "energy", "b1", "b2", "b3"),
      expressions=(
         mass_flux,
         *momentum_flux,
         energy_flux,
         *magnetic_flux,
      ),
   )
