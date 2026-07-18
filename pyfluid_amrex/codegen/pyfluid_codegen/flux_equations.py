from __future__ import annotations

import sympy as sp

from .field_catalog import (
   FLUID_NEWTONIAN_FIELDS,
   MAGNETIC_FIELD,
   NEWTONIAN_HYDRO_COMPONENT_NAMES,
   NEWTONIAN_MHD_COMPONENT_NAMES,
   component_symbols,
)
from .kernel_equations import StructKernel
from .physical_system import HydroState, Hydrodynamics, MHDState, Magnetohydrodynamics


def hydro_flux_equations(direction: int) -> StructKernel:
   components = component_symbols(FLUID_NEWTONIAN_FIELDS, owner="state")
   rho = components["RHO"]
   mom1, mom2, mom3 = (components[f"MOM{axis}"] for axis in range(1, 4))
   energy = components["ENERGY"]
   pressure = sp.Symbol("pressure")
   state = HydroState.from_symbols(rho, mom1, mom2, mom3, energy)
   mass_flux, momentum_flux, energy_flux = Hydrodynamics().conservation_law(
      state, pressure
   ).flux_along(direction)
   return StructKernel(
      name=f"hydro_flux_orthonormal_dir{direction}",
      return_type="HydroState",
      arguments=(("const HydroState&", "state"), ("amrex::Real", "pressure")),
      outputs=NEWTONIAN_HYDRO_COMPONENT_NAMES,
      expressions=(
         mass_flux,
         *momentum_flux,
         energy_flux,
      ),
   )


def mhd_flux_equations(direction: int) -> StructKernel:
   components = component_symbols(
      (*FLUID_NEWTONIAN_FIELDS, MAGNETIC_FIELD), owner="state"
   )
   rho = components["RHO"]
   mom1, mom2, mom3 = (components[f"MOM{axis}"] for axis in range(1, 4))
   energy = components["ENERGY"]
   pressure = sp.Symbol("pressure")
   b1, b2, b3 = (components[f"B{axis}"] for axis in range(1, 4))
   state = MHDState.from_symbols(rho, mom1, mom2, mom3, energy, b1, b2, b3)
   mass_flux, momentum_flux, energy_flux, magnetic_flux = Magnetohydrodynamics().conservation_law(
      state, pressure
   ).flux_along(direction)
   return StructKernel(
      name=f"mhd_flux_orthonormal_dir{direction}",
      return_type="MHDState",
      arguments=(("const MHDState&", "state"), ("amrex::Real", "pressure")),
      outputs=NEWTONIAN_MHD_COMPONENT_NAMES,
      expressions=(
         mass_flux,
         *momentum_flux,
         energy_flux,
         *magnetic_flux,
      ),
   )
