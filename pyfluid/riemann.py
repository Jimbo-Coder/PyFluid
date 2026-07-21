import functools as ft

import jax
import jax.numpy as jnp

from pyfluid.state import MAG_P, MOM_U, P, S, U, VEL_P


class RiemannBase:
   def __init__(self, parent):
      self.parent = parent
      self.fluxi = parent.fluxi

      if self.fluxi == 0:
         self.calc_riemann_fluxes = self.calc_LLF_fluxes
         self.name = "LLF"
      elif self.fluxi == 1:
         self.calc_riemann_fluxes = self.calc_HLL_fluxes
         self.name = "HLL"
      elif self.fluxi == 2:
         self.calc_riemann_fluxes = self.calc_HLLE_fluxes
         self.name = "HLLE"
      elif self.fluxi == 3:
         self.calc_riemann_fluxes = self.calc_HLLC_fluxes
         self.name = "HLLC"
      elif self.fluxi == 4:
         raise NotImplementedError("HLLD is listed but not implemented yet.")
      else:
         raise ValueError("Riemann solver not recognized.")

   def validate_model(self, physicsmodel):
      if physicsmodel.boolmag and self.fluxi == 3:
         raise ValueError("HLLC is hydro-only in this code; use LLF, HLL, or HLLE for MHD.")

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(0,))
   def calc_LLF_fluxes(physicsmodelf, vaxisi, uvecintL, uvecintR, pvecintL, pvecintR, cs2vecintL, cs2vecintR, fvecintL, fvecintR):
      alphaint = jnp.maximum(
         jnp.abs(pvecintL[vaxisi, ...]) + jnp.sqrt(cs2vecintL[S.FAST2, ...]),
         jnp.abs(pvecintR[vaxisi, ...]) + jnp.sqrt(cs2vecintR[S.FAST2, ...]),
      )
      fvecint = 0.5 * (fvecintL + fvecintR) - 0.5 * alphaint * (uvecintR - uvecintL)
      return fvecint, alphaint

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(0,))
   def calc_HLL_fluxes(physicsmodelf, vaxisi, uvecintL, uvecintR, pvecintL, pvecintR, cs2vecintL, cs2vecintR, fvecintL, fvecintR):
      SL = jnp.minimum(
         pvecintL[vaxisi, ...] - jnp.sqrt(cs2vecintL[S.FAST2, ...]),
         pvecintR[vaxisi, ...] - jnp.sqrt(cs2vecintR[S.FAST2, ...]),
      )
      SR = jnp.maximum(
         pvecintL[vaxisi, ...] + jnp.sqrt(cs2vecintL[S.FAST2, ...]),
         pvecintR[vaxisi, ...] + jnp.sqrt(cs2vecintR[S.FAST2, ...]),
      )
      UvecintHll = (SR * uvecintR - SL * uvecintL + fvecintL - fvecintR) / (SR - SL)
      FvecintHLL = fvecintR + SR * (UvecintHll - uvecintR)
      maskLHLL = SL >= 0
      maskRHLL = SR <= 0
      maskHLL = (SL < 0) & (SR > 0)
      fvecint = jnp.where(maskLHLL, fvecintL, jnp.where(maskRHLL, fvecintR, jnp.where(maskHLL, FvecintHLL, 0.0)))
      alphaint = jnp.maximum(jnp.abs(SL), jnp.abs(SR))
      return fvecint, alphaint

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(0, 1))
   def calc_HLLE_fluxes(physicsmodelf, vaxisi, uvecintL, uvecintR, pvecintL, pvecintR, cs2vecintL, cs2vecintR, fvecintL, fvecintR):
      SL, SR = RiemannBase.calc_SLSR_roeDE(physicsmodelf, vaxisi, uvecintL, uvecintR, pvecintL, pvecintR)
      UvecintHll = (SR * uvecintR - SL * uvecintL + fvecintL - fvecintR) / (SR - SL)
      FvecintHLL = fvecintR + SR * (UvecintHll - uvecintR)
      maskLHLL = SL >= 0
      maskRHLL = SR <= 0
      maskHLL = (SL < 0) & (SR > 0)
      fvecint = jnp.where(maskLHLL, fvecintL, jnp.where(maskRHLL, fvecintR, jnp.where(maskHLL, FvecintHLL, 0.0)))
      alphaint = jnp.maximum(jnp.abs(SL), jnp.abs(SR))
      return fvecint, alphaint

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(0, 1))
   def calc_HLLC_fluxes(physicsmodelf, vaxisi, uvecintL, uvecintR, pvecintL, pvecintR, cs2vecintL, cs2vecintR, fvecintL, fvecintR):
      SL, SR = RiemannBase.calc_SLSR_roeDE(physicsmodelf, vaxisi, uvecintL, uvecintR, pvecintL, pvecintR)
      normal_vel_i = vaxisi - int(P.VX)
      rhoL = pvecintL[P.RHO, ...]
      rhoR = pvecintR[P.RHO, ...]
      vnL = pvecintL[vaxisi, ...]
      vnR = pvecintR[vaxisi, ...]
      lfac = SL - vnL
      lfacr = lfac * rhoL
      lfacru = lfacr * vnL
      rfac = SR - vnR
      rfacr = rfac * rhoR
      rfacru = rfacr * vnR
      Sstar = (pvecintR[P.PRESS, ...] - pvecintL[P.PRESS, ...] + lfacru - rfacru) / (lfacr - rfacr)
      lstarfac = lfacr / (SL - Sstar)
      rstarfac = rfacr / (SR - Sstar)
      UstarL = jnp.zeros_like(uvecintL)
      UstarR = jnp.zeros_like(uvecintR)
      vstarL = pvecintL[VEL_P, ...].at[normal_vel_i, ...].set(Sstar)
      vstarR = pvecintR[VEL_P, ...].at[normal_vel_i, ...].set(Sstar)
      UstarL = UstarL.at[U.RHO, ...].set(lstarfac)
      UstarL = UstarL.at[MOM_U, ...].set(lstarfac[None, ...] * vstarL)
      UstarL = UstarL.at[U.ENERGY, ...].set(
         lstarfac * (uvecintL[U.ENERGY, ...] / rhoL + (Sstar - vnL) * (Sstar + pvecintL[P.PRESS, ...] / lfacr))
      )
      UstarR = UstarR.at[U.RHO, ...].set(rstarfac)
      UstarR = UstarR.at[MOM_U, ...].set(rstarfac[None, ...] * vstarR)
      UstarR = UstarR.at[U.ENERGY, ...].set(
         rstarfac * (uvecintR[U.ENERGY, ...] / rhoR + (Sstar - vnR) * (Sstar + pvecintR[P.PRESS, ...] / rfacr))
      )
      FstarL = fvecintL + SL * (UstarL - uvecintL)
      FstarR = fvecintR + SR * (UstarR - uvecintR)
      maskL = 0 <= SL
      maskR = 0 >= SR
      maskLstar = (SL <= 0) & (0 <= Sstar)
      maskRstar = (Sstar <= 0) & (0 <= SR)
      fvecint = jnp.where(maskL, fvecintL, jnp.where(maskR, fvecintR, jnp.where(maskLstar, FstarL, jnp.where(maskRstar, FstarR, 0.0))))
      alphaint = jnp.maximum(jnp.abs(SL), jnp.abs(SR))
      return fvecint, alphaint

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(0, 1))
   def calc_SLSR_roeDE(physicsmodelf, vaxisi, uvecintL, uvecintR, pvecintL, pvecintR):
      sqrhol = jnp.sqrt(uvecintL[U.RHO, ...])
      sqrhor = jnp.sqrt(uvecintR[U.RHO, ...])
      mageL = 0.5 * jnp.sqrt(jnp.square(pvecintL[MAG_P, ...]).sum(axis=0))
      mageR = 0.5 * jnp.sqrt(jnp.square(pvecintR[MAG_P, ...]).sum(axis=0))
      hL = (uvecintL[U.ENERGY, ...] + pvecintL[P.PRESS, ...] + mageL) / uvecintL[U.RHO, ...]
      hR = (uvecintR[U.ENERGY, ...] + pvecintR[P.PRESS, ...] + mageR) / uvecintR[U.RHO, ...]
      hroe = (sqrhol * hL + sqrhor * hR) / (sqrhol + sqrhor)
      uroe = (sqrhol * pvecintL[vaxisi, ...] + sqrhor * pvecintR[vaxisi, ...]) / (sqrhol + sqrhor)
      aroe = jnp.sqrt((physicsmodelf.gamma - 1) * (hroe - 0.5 * (uroe**2)))
      QL = jnp.sqrt(
         1
         + ((physicsmodelf.gamma + 1) / (2 * physicsmodelf.gamma))
         * jnp.maximum(0, pvecintR[P.PRESS, ...] / pvecintL[P.PRESS, ...] - 1)
      )
      QR = jnp.sqrt(
         1
         + ((physicsmodelf.gamma + 1) / (2 * physicsmodelf.gamma))
         * jnp.maximum(0, pvecintL[P.PRESS, ...] / pvecintR[P.PRESS, ...] - 1)
      )
      return uroe - aroe * QL, uroe + aroe * QR
