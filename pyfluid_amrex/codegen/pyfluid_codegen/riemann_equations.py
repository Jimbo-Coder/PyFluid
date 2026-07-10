from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class StateKernel:
   name: str
   return_type: str
   arguments: tuple[tuple[str, str], ...]
   outputs: tuple[str, ...]
   expressions: tuple[sp.Expr, ...]


HYDRO_FIELDS = ("rho", "mom1", "mom2", "mom3", "energy")
MHD_FIELDS = HYDRO_FIELDS + ("b1", "b2", "b3")


def _state_symbols(prefix: str, fields: tuple[str, ...]) -> tuple[sp.Symbol, ...]:
   return tuple(sp.symbols(f"{prefix}.{field}") for field in fields)


def _llf_kernel(name: str, return_type: str, fields: tuple[str, ...]) -> StateKernel:
   flux_l = _state_symbols("flux_l", fields)
   flux_r = _state_symbols("flux_r", fields)
   state_l = _state_symbols("state_l", fields)
   state_r = _state_symbols("state_r", fields)
   wavespeed = sp.symbols("wavespeed")
   expressions = tuple(
      sp.Rational(1, 2) * (fl + fr) - sp.Rational(1, 2) * wavespeed * (ur - ul)
      for fl, fr, ul, ur in zip(flux_l, flux_r, state_l, state_r, strict=True)
   )
   return StateKernel(
      name=name,
      return_type=return_type,
      arguments=(
         (f"const {return_type}&", "flux_l"),
         (f"const {return_type}&", "flux_r"),
         (f"const {return_type}&", "state_l"),
         (f"const {return_type}&", "state_r"),
         ("amrex::Real", "wavespeed"),
      ),
      outputs=fields,
      expressions=expressions,
   )


def _hll_kernel(name: str, return_type: str, fields: tuple[str, ...]) -> StateKernel:
   flux_l = _state_symbols("flux_l", fields)
   flux_r = _state_symbols("flux_r", fields)
   state_l = _state_symbols("state_l", fields)
   state_r = _state_symbols("state_r", fields)
   s_left, s_right = sp.symbols("s_left s_right")
   expressions = tuple(
      (s_right * fl - s_left * fr + s_left * s_right * (ur - ul)) / (s_right - s_left)
      for fl, fr, ul, ur in zip(flux_l, flux_r, state_l, state_r, strict=True)
   )
   return StateKernel(
      name=name,
      return_type=return_type,
      arguments=(
         (f"const {return_type}&", "flux_l"),
         (f"const {return_type}&", "flux_r"),
         (f"const {return_type}&", "state_l"),
         (f"const {return_type}&", "state_r"),
         ("amrex::Real", "s_left"),
         ("amrex::Real", "s_right"),
      ),
      outputs=fields,
      expressions=expressions,
   )


def riemann_flux_kernels() -> tuple[StateKernel, ...]:
   return (
      _llf_kernel("hydro_llf_flux", "HydroState", HYDRO_FIELDS),
      _hll_kernel("hydro_hll_flux", "HydroState", HYDRO_FIELDS),
      _llf_kernel("mhd_llf_flux", "MHDState", MHD_FIELDS),
      _hll_kernel("mhd_hll_flux", "MHDState", MHD_FIELDS),
   )
