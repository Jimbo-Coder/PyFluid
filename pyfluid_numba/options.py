from dataclasses import dataclass

import numpy as np

from pyfluid.initial_data import INITIAL_CONDITIONS
from pyfluid.runtime import DEFAULT_NUMBA_THREADS
from pyfluid.state import U, NVAR_HYDRO, NVAR_MHD
from pyfluid_numba.eigensystem import (
   HYDRO_MODE_ENTROPY,
   HYDRO_MODE_SHEAR,
   HYDRO_MODE_SHEAR_Z,
   HYDRO_MODE_SOUND_MINUS,
   HYDRO_MODE_SOUND_PLUS,
   MHD_MODE_ALFVEN_MINUS,
   MHD_MODE_ALFVEN_PLUS,
   MHD_MODE_ENTROPY,
   MHD_MODE_FAST_MINUS,
   MHD_MODE_FAST_PLUS,
   MHD_MODE_SLOW_MINUS,
   MHD_MODE_SLOW_PLUS,
   hydro_mode_name,
   mhd_mode_name,
   validate_hydro_characteristic_mode,
   validate_mhd_characteristic_mode,
)

DEFAULT_NG = 2
DEFAULT_GAMMA = 5.0 / 3.0
DEFAULT_CFL = 0.9
DEFAULT_DIFFUSION_CFL = 0.9
DEFAULT_THREADS = DEFAULT_NUMBA_THREADS
DEFAULT_DIFFUSION_TOLERANCE = 1.0e-10
DEFAULT_DIFFUSION_MAX_STEPS = 200
DEFAULT_DIFFUSION_SOR_OMEGA = 0.7
SIN_WAVE_IC = -1
SUPPORTED_HYDRO_INITIAL_CONDITIONS = tuple(sorted(ici for ici, info in INITIAL_CONDITIONS.items() if not info.magnetized))
SUPPORTED_MHD_INITIAL_CONDITIONS = tuple(sorted(ici for ici, info in INITIAL_CONDITIONS.items() if info.magnetized))
BC_PERIODIC = 0
BC_OUTFLOW = 1
BC_REFLECTIVE = 2
RECON_PWC = 0
RECON_PLM = 1
RECON_WENOZ = 2
RECON_WENOZ_CHAR = 3
LIMITER_MINMOD = 1
LIMITER_MC = 4
RECON_VARS_PRIMITIVE = 0
RECON_VARS_CONSERVED = 1
AXIS_X = 0
AXIS_Y = 1
SIDE_LOWER = 0
SIDE_UPPER = 1
RIEMANN_LLF = 0
RIEMANN_HLL = 1
RIEMANN_HLLE = 2
RIEMANN_HLLC = 3
RIEMANN_HLLD = 4
TIME_EULER = 0
TIME_CORRECTOR = 1
TIME_MIDPOINT = 2
TIME_RK4 = 3
TIME_SSPRK3 = 4
DIFFUSION_NONE = 0
DIFFUSION_EXPLICIT = 1
DIFFUSION_IMPLICIT = 2
CT_OFF = 0
CT_FLUX_TOTH = 1
RHO = int(U.RHO)
MOMX = int(U.MOMX)
MOMY = int(U.MOMY)
MOMZ = int(U.MOMZ)
ENERGY = int(U.ENERGY)
BX = int(U.BX)
BY = int(U.BY)
BZ = int(U.BZ)
MAGNETIC_EPS = 1.0e-14
WENO_EPS = 1.0e-8


@dataclass(frozen=True)
class HydroRunConfig:
   nx: int
   ny: int
   steps: int
   threads: int = DEFAULT_THREADS
   ng: int = DEFAULT_NG
   gamma: float = DEFAULT_GAMMA
   cfl: float = DEFAULT_CFL
   bc: tuple = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
   recon: int = RECON_PLM
   recon_vars: int = RECON_VARS_PRIMITIVE
   limiter: int = LIMITER_MC
   riemann: int = RIEMANN_HLLC
   timestep: int = TIME_SSPRK3
   ic: int = SIN_WAVE_IC
   delrho_factor: float = 6.0


@dataclass(frozen=True)
class MHDRunConfig:
   nx: int
   ny: int
   steps: int
   threads: int = DEFAULT_THREADS
   ng: int = 3
   gamma: float = DEFAULT_GAMMA
   cfl: float = DEFAULT_CFL
   bc: tuple = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
   recon: int = RECON_WENOZ
   recon_vars: int = RECON_VARS_PRIMITIVE
   limiter: int = LIMITER_MC
   riemann: int = RIEMANN_LLF
   timestep: int = TIME_SSPRK3
   ic: int = 8
   delrho_factor: float = 6.0
   ct_method: int = CT_FLUX_TOTH
   diffusion_method: int = DIFFUSION_NONE
   viscosity: float = -1.0
   resistivity: float = 1.0e-4
   diffusion_cfl: float = DEFAULT_DIFFUSION_CFL
   diffusion_tolerance: float = DEFAULT_DIFFUSION_TOLERANCE
   diffusion_max_steps: int = DEFAULT_DIFFUSION_MAX_STEPS
   diffusion_sor_omega: float = DEFAULT_DIFFUSION_SOR_OMEGA


@dataclass(frozen=True)
class EntropyWaveConfig:
   nx: int
   ny: int
   final_time: float
   threads: int = DEFAULT_THREADS
   ng: int = DEFAULT_NG
   gamma: float = DEFAULT_GAMMA
   cfl: float = DEFAULT_CFL
   bc: tuple = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
   recon: int = RECON_PLM
   recon_vars: int = RECON_VARS_PRIMITIVE
   limiter: int = LIMITER_MC
   riemann: int = RIEMANN_HLLC
   timestep: int = TIME_SSPRK3
   rho0: float = 1.0
   pressure0: float = 1.0
   amplitude: float = 0.01
   vx: float = 0.2
   vy: float = -0.1
   kx: int = 1
   ky: int = 1


@dataclass(frozen=True)
class AcousticWaveConfig:
   nx: int
   ny: int
   final_time: float
   threads: int = DEFAULT_THREADS
   ng: int = DEFAULT_NG
   gamma: float = DEFAULT_GAMMA
   cfl: float = DEFAULT_CFL
   bc: tuple = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
   recon: int = RECON_PLM
   recon_vars: int = RECON_VARS_PRIMITIVE
   limiter: int = LIMITER_MC
   riemann: int = RIEMANN_HLLC
   timestep: int = TIME_SSPRK3
   rho0: float = 1.0
   pressure0: float = 1.0
   amplitude: float = 1.0e-4
   kx: int = 1
   ky: int = 1


@dataclass(frozen=True)
class HydroCharacteristicWaveConfig:
   nx: int
   ny: int
   final_time: float
   mode: int = HYDRO_MODE_SOUND_PLUS
   threads: int = DEFAULT_THREADS
   ng: int = DEFAULT_NG
   gamma: float = DEFAULT_GAMMA
   cfl: float = DEFAULT_CFL
   bc: tuple = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
   recon: int = RECON_PLM
   recon_vars: int = RECON_VARS_PRIMITIVE
   limiter: int = LIMITER_MC
   riemann: int = RIEMANN_HLLC
   timestep: int = TIME_SSPRK3
   rho0: float = 1.0
   pressure0: float = 1.0
   vx0: float = 0.2
   vy0: float = -0.1
   vz0: float = 0.0
   amplitude: float = 1.0e-6
   kx: int = 1
   ky: int = 1


@dataclass(frozen=True)
class MHDCharacteristicWaveConfig:
   nx: int
   ny: int
   final_time: float
   mode: int = MHD_MODE_FAST_PLUS
   threads: int = DEFAULT_THREADS
   ng: int = 3
   gamma: float = DEFAULT_GAMMA
   cfl: float = DEFAULT_CFL
   bc: tuple = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
   recon: int = RECON_WENOZ
   recon_vars: int = RECON_VARS_PRIMITIVE
   limiter: int = LIMITER_MC
   riemann: int = RIEMANN_HLLD
   timestep: int = TIME_SSPRK3
   ct_method: int = CT_FLUX_TOTH
   rho0: float = 1.0
   pressure0: float = 1.0
   vx0: float = 0.2
   vy0: float = -0.1
   vz0: float = 0.05
   bx0: float = 0.75
   by0: float = -0.35
   bz0: float = 0.5
   amplitude: float = 1.0e-7
   kx: int = 1
   ky: int = 1


@dataclass(frozen=True)
class CircularAlfvenWaveConfig:
   nx: int
   ny: int
   final_time: float
   threads: int = DEFAULT_THREADS
   ng: int = 3
   gamma: float = DEFAULT_GAMMA
   cfl: float = DEFAULT_CFL
   bc: tuple = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
   recon: int = RECON_WENOZ_CHAR
   recon_vars: int = RECON_VARS_PRIMITIVE
   limiter: int = LIMITER_MC
   riemann: int = RIEMANN_HLLD
   timestep: int = TIME_RK4
   ct_method: int = CT_FLUX_TOTH
   rho0: float = 1.0
   pressure0: float = 1.0
   bx0: float = 1.0
   amplitude: float = 0.1
   wave_number: int = 1
   direction: int = 1


@dataclass(frozen=True)
class ResistiveWaveConfig:
   nx: int
   ny: int
   final_time: float
   steps: int
   threads: int = DEFAULT_THREADS
   ng: int = DEFAULT_NG
   gamma: float = DEFAULT_GAMMA
   bc: tuple = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
   rho0: float = 1.0
   pressure0: float = 1.0
   amplitude: float = 0.1
   resistivity: float = 1.0e-2
   wave_number: int = 2
   diffusion_method: int = DIFFUSION_IMPLICIT
   diffusion_tolerance: float = DEFAULT_DIFFUSION_TOLERANCE
   diffusion_max_steps: int = DEFAULT_DIFFUSION_MAX_STEPS
   diffusion_sor_omega: float = DEFAULT_DIFFUSION_SOR_OMEGA


def riemann_name(riemann):
   if riemann == RIEMANN_LLF:
      return "llf"
   if riemann == RIEMANN_HLL:
      return "hll"
   if riemann == RIEMANN_HLLE:
      return "hlle"
   if riemann == RIEMANN_HLLC:
      return "hllc"
   if riemann == RIEMANN_HLLD:
      return "hlld"
   raise ValueError(f"unsupported Numba Riemann solver: {riemann}")


def recon_name(recon):
   if recon == RECON_PWC:
      return "pwc"
   if recon == RECON_PLM:
      return "pwl"
   if recon == RECON_WENOZ:
      return "wenoz"
   if recon == RECON_WENOZ_CHAR:
      return "wenoz-char"
   raise ValueError(f"unsupported Numba hydro reconstruction: {recon}")


def recon_vars_name(recon_vars):
   if recon_vars == RECON_VARS_PRIMITIVE:
      return "prim"
   if recon_vars == RECON_VARS_CONSERVED:
      return "cons"
   raise ValueError(f"unsupported Numba hydro reconstruction variable: {recon_vars}")


def limiter_name(limiter):
   if limiter == LIMITER_MINMOD:
      return "minmod"
   if limiter == LIMITER_MC:
      return "mc"
   raise ValueError(f"unsupported Numba hydro limiter: {limiter}")


def timestep_name(timestep):
   if timestep == TIME_EULER:
      return "euler"
   if timestep == TIME_CORRECTOR:
      return "corrector"
   if timestep == TIME_MIDPOINT:
      return "midpoint"
   if timestep == TIME_RK4:
      return "rk4"
   if timestep == TIME_SSPRK3:
      return "ssprk3"
   raise ValueError(f"unsupported Numba timestepper: {timestep}")


def timestep_rhs_stages(timestep):
   if timestep == TIME_EULER:
      return 1
   if timestep == TIME_CORRECTOR:
      return 2
   if timestep == TIME_MIDPOINT:
      return 2
   if timestep == TIME_RK4:
      return 4
   if timestep == TIME_SSPRK3:
      return 3
   raise ValueError(f"unsupported Numba timestepper: {timestep}")


def diffusion_method_name(diffusion_method):
   if diffusion_method == DIFFUSION_NONE:
      return "off"
   if diffusion_method == DIFFUSION_EXPLICIT:
      return "explicit"
   if diffusion_method == DIFFUSION_IMPLICIT:
      return "implicit"
   raise ValueError(f"unsupported Numba diffusion method: {diffusion_method}")


def ct_method_name(ct_method):
   if ct_method == CT_OFF:
      return "no CT"
   if ct_method == CT_FLUX_TOTH:
      return "flux-Toth"
   raise ValueError(f"unsupported Numba CT method: {ct_method}")


def hydro_initial_condition_name(ic):
   if ic == SIN_WAVE_IC:
      return "sin wave"
   return INITIAL_CONDITIONS[int(ic)].name


def bc_name(bc_value):
   if bc_value == BC_PERIODIC:
      return "periodic"
   if bc_value == BC_OUTFLOW:
      return "outflow"
   if bc_value == BC_REFLECTIVE:
      return "reflective"
   raise ValueError(f"unsupported Numba boundary condition: {bc_value}")


def bc_tuple_name(bc):
   if len(bc) != 4:
      raise ValueError("boundary condition tuple must have four entries")
   names = tuple(bc_name(value) for value in bc)
   if names[0] == names[1] == names[2] == names[3]:
      return names[0]
   if names[0] == names[1] and names[2] == names[3]:
      return f"x={names[0]}, y={names[2]}"
   return f"L/R/B/T={names[0]}/{names[1]}/{names[2]}/{names[3]}"


def validate_hydro_initial_condition(ic):
   ic = int(ic)
   if ic == SIN_WAVE_IC:
      return ic
   if ic not in INITIAL_CONDITIONS:
      raise ValueError(f"unsupported Numba hydro initial condition: {ic}")
   if INITIAL_CONDITIONS[ic].magnetized:
      raise ValueError(f"initial condition {ic} is magnetized and is not supported by the Numba hydro backend")
   return ic


def validate_mhd_initial_condition(ic):
   ic = int(ic)
   if ic not in INITIAL_CONDITIONS:
      raise ValueError(f"unsupported Numba MHD initial condition: {ic}")
   if not INITIAL_CONDITIONS[ic].magnetized:
      raise ValueError(f"initial condition {ic} is hydro-only and is not supported by the Numba MHD backend")
   return ic


def validate_bc(bc):
   if len(bc) != 4:
      raise ValueError("boundary condition tuple must have four entries")
   for value in bc:
      if value not in (BC_PERIODIC, BC_OUTFLOW, BC_REFLECTIVE):
         raise ValueError(f"unsupported Numba hydro boundary condition: {value}")
   return tuple(int(value) for value in bc)


def validate_common_numeric_options(config, mhd):
   recon_name(config.recon)
   limiter_name(config.limiter)
   timestep_name(config.timestep)
   recon_vars_name(config.recon_vars)
   diffusion_method = getattr(config, "diffusion_method", DIFFUSION_NONE)
   diffusion_method_name(diffusion_method)
   ct_method = getattr(config, "ct_method", CT_OFF)
   ct_method_name(ct_method)
   if config.recon in (RECON_WENOZ, RECON_WENOZ_CHAR) and config.recon_vars != RECON_VARS_PRIMITIVE:
      raise ValueError("Numba WENO-Z reconstructs primitive or characteristic primitive variables only")
   if config.recon in (RECON_WENOZ, RECON_WENOZ_CHAR) and config.ng < 3:
      raise ValueError("Numba WENO-Z requires at least 3 ghost cells")
   if diffusion_method != DIFFUSION_NONE and config.diffusion_cfl <= 0.0:
      raise ValueError("Numba diffusion requires diffusion_cfl > 0")
   if diffusion_method == DIFFUSION_IMPLICIT:
      if config.diffusion_tolerance <= 0.0:
         raise ValueError("Numba implicit diffusion requires diffusion_tolerance > 0")
      if config.diffusion_max_steps < 1:
         raise ValueError("Numba implicit diffusion requires diffusion_max_steps >= 1")
      if config.diffusion_sor_omega <= 0.0 or config.diffusion_sor_omega >= 2.0:
         raise ValueError("Numba implicit diffusion requires 0 < diffusion_sor_omega < 2")
   if mhd and config.riemann == RIEMANN_HLLC:
      raise ValueError("Numba MHD HLLC is hydro-only; use LLF, HLL, HLLE, or HLLD")
   if mhd and config.riemann not in (RIEMANN_LLF, RIEMANN_HLL, RIEMANN_HLLE, RIEMANN_HLLD):
      raise ValueError(f"unsupported Numba MHD Riemann solver: {config.riemann}")
   if (not mhd) and config.riemann not in (RIEMANN_LLF, RIEMANN_HLL, RIEMANN_HLLE, RIEMANN_HLLC):
      raise ValueError(f"unsupported Numba hydro Riemann solver: {config.riemann}")


def validate_circular_alfven_options(config):
   validate_common_numeric_options(config, mhd=True)
   if config.rho0 <= 0.0:
      raise ValueError("circular Alfven wave requires rho0 > 0")
   if config.pressure0 <= 0.0:
      raise ValueError("circular Alfven wave requires pressure0 > 0")
   if config.bx0 == 0.0:
      raise ValueError("circular Alfven wave requires nonzero bx0")
   if config.amplitude <= 0.0:
      raise ValueError("circular Alfven wave requires amplitude > 0")
   if int(config.wave_number) < 1:
      raise ValueError("circular Alfven wave requires wave_number >= 1")
   if int(config.direction) not in (-1, 1):
      raise ValueError("circular Alfven wave direction must be -1 or +1")


def resolve_diffusion_coefficients(config, dx):
   viscosity = float(config.viscosity)
   if viscosity < 0.0:
      viscosity = dx * np.sqrt(config.gamma)
   resistivity = float(config.resistivity)
   if resistivity < 0.0:
      resistivity = 0.0
   return viscosity, resistivity


def require_reached_final_time(time_value, final_time, steps, max_steps):
   if steps >= max_steps and time_value < final_time:
      raise RuntimeError(f"adaptive hydro evolution stopped at t={time_value} before final_time={final_time}")
