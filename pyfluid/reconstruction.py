import functools as ft
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pyfluid.state import U


@jax.jit
def calc_sloplim_diff(a, b):
   return (a + b) / 2


@jax.jit
def calc_sloplim_minmod(a, b):
   return jnp.where(
      jnp.sign(a) == jnp.sign(b),
      jnp.where(jnp.abs(a) < jnp.abs(b), a, b),
      0.0,
   )


@jax.jit
def calc_sloplim_superbee(a, b):
   return (
      jnp.maximum(
         0,
         jnp.minimum(2 * jnp.abs(a), jnp.abs(b)),
         jnp.minimum(jnp.abs(a), 2 * jnp.abs(b)),
      )
      * jnp.sign(a)
      * jnp.where(jnp.sign(a) == jnp.sign(b), 1.0, 0.0)
   )


@jax.jit
def calc_sloplim_vanleer(a, b):
   out = (2 * a * b) / (a + b + 1e-16) * jnp.where(jnp.sign(a) == jnp.sign(b), 1.0, 0.0)
   return jnp.where(out > 1e15, 0.0, out)


@jax.jit
def calc_sloplim_mc(a, b):
   return (
      jnp.minimum(0.5 * (a + b), 2 * jnp.abs(a), 2 * jnp.abs(b))
      * jnp.sign(a)
      * jnp.where(jnp.sign(a) == jnp.sign(b), 1.0, 0.0)
   )


class ReconBase:
   @dataclass(frozen=True)
   class ReconModel:
      LRBTreconV: callable
      LRBTrecon_UP: callable
      calc_slopelimit: callable
      wenoweights: tuple
      weno_eps: float

   def __init__(self, parent):
      self.parent = parent
      self.ng = parent.ng
      self.ngi = parent.ngi
      self.ngnet = self.ng - self.ngi
      self.nx = parent.nx
      self.limi = parent.limi
      self.reconi = parent.reconi
      self.fluxct = parent.fluxct
      self.intisl = parent.intisl
      self.isl = parent.imasksl1D

      norm = 10.0
      self.wenoweights = tuple(x / norm for x in (3.0, 6.0, 1.0))
      self.weno_eps = 1e-8
      self.ip1sl1d = slice(self.ngnet, -self.ngnet)

      self.outer0 = -self.ng if self.ng >= 1 else None
      self.outeri1 = -(self.ng - 1) if self.ng >= 2 else None
      self.outeri2 = -(self.ng - 2) if self.ng >= 3 else None
      self.outeri3 = -(self.ng - 3) if self.ng >= 4 else None

      self.intLsl = slice(self.ngi - 1, -(self.ngi + 1))
      self.intRsl = slice(self.ngi + 1, -(self.ngi - 1)) if self.ngi >= 2 else slice(self.ngi + 1, None)

      self.Lsl = slice(self.ng - 1, -self.ng)
      self.Lsl2 = slice(self.ng - 2, -(self.ng + 1))
      self.Rsl = slice(self.ng, -(self.ng - 1)) if self.ng >= 2 else slice(self.ng, None)

      self.Lsld = slice(self.ng - 1, -(self.ng - 1)) if self.ng >= 2 else slice(self.ng - 1, None)
      self.Lsld2 = slice(self.ng - 2, -self.ng) if self.ng >= 1 else slice(self.ng - 2, None)
      self.Rsld = slice(self.ng, -(self.ng - 2)) if self.ng >= 3 else slice(self.ng, None)

      if self.limi == 0:
         self.calc_slopelimit = calc_sloplim_diff
         self.limname = "Cen.Diff."
      elif self.limi == 1:
         self.calc_slopelimit = calc_sloplim_minmod
         self.limname = "MinMod"
      elif self.limi == 2:
         self.calc_slopelimit = calc_sloplim_superbee
         self.limname = "Superbee"
      elif self.limi == 3:
         self.calc_slopelimit = calc_sloplim_vanleer
         self.limname = "VanLeer"
      elif self.limi == 4:
         self.calc_slopelimit = calc_sloplim_mc
         self.limname = "MC"
      else:
         raise ValueError("Slope limiter not recognized.")

      if self.reconi == 0:
         self.LRBTreconV = self.LRBTreconV_pwconstant
         self.reconname = "PW.Constant"
         self.reconnames = self.reconname
         assert self.ngnet >= 1
      elif self.reconi == 1:
         self.LRBTreconV = self.LRBTreconV_pwlinear_slopelimited
         self.reconname = "PW.Linear"
         self.reconnames = self.reconname + " " + self.limname
         assert self.ngnet >= 2
      elif self.reconi == 2:
         self.LRBTreconV = self.LRBTreconV_wenoz
         self.reconname = "WENOZ"
         self.reconnames = self.reconname
         assert self.ngnet >= 3
      else:
         raise ValueError("Reconstruction method not recognized.")

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1, 2, 3))
   def LRBTreconV_pwconstant(vvecf, gridparamsf, physicsmodelf, reconmodelf=None):
      sl = physicsmodelf.voi
      ngn = gridparamsf.ngnet
      nvar = vvecf.shape[0]
      vvecintLf = jnp.zeros((nvar, gridparamsf.nxifs, gridparamsf.ny + 2 * gridparamsf.ngi))
      vvecintRf = jnp.zeros((nvar, gridparamsf.nxifs, gridparamsf.ny + 2 * gridparamsf.ngi))
      vvecintBf = jnp.zeros((nvar, gridparamsf.nx + 2 * gridparamsf.ngi, gridparamsf.nyifs))
      vvecintTf = jnp.zeros((nvar, gridparamsf.nx + 2 * gridparamsf.ngi, gridparamsf.nyifs))
      vvecintLf = vvecintLf.at[0:sl, :, :].set(vvecf[0:sl, gridparamsf.ng - 1:gridparamsf.outern0, ngn:-(ngn)])
      vvecintRf = vvecintRf.at[0:sl, :, :].set(vvecf[0:sl, gridparamsf.ng:gridparamsf.outern1, ngn:-(ngn)])
      vvecintBf = vvecintBf.at[0:sl, :, :].set(vvecf[0:sl, ngn:-(ngn), gridparamsf.ng - 1:gridparamsf.youtern0])
      vvecintTf = vvecintTf.at[0:sl, :, :].set(vvecf[0:sl, ngn:-(ngn), gridparamsf.ng:gridparamsf.youtern1])
      return vvecintLf, vvecintRf, vvecintBf, vvecintTf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1, 2, 3))
   def LRBTreconV_pwlinear_slopelimited(vvecf, gridparamsf, physicsmodelf, reconmodelf):
      sl = physicsmodelf.voi
      ngn = gridparamsf.ngnet
      ng = gridparamsf.ng
      dVvecL = (
         vvecf[0:sl, ng - 1:gridparamsf.outern1, ngn:-(ngn)]
         - vvecf[0:sl, ng - 2:gridparamsf.outern0, ngn:-(ngn)]
      )
      dVvecR = (
         vvecf[0:sl, ng:gridparamsf.outern2, ngn:-(ngn)]
         - vvecf[0:sl, ng - 1:gridparamsf.outern1, ngn:-(ngn)]
      )
      dVvecB = (
         vvecf[0:sl, ngn:-(ngn), ng - 1:gridparamsf.youtern1]
         - vvecf[0:sl, ngn:-(ngn), ng - 2:gridparamsf.youtern0]
      )
      dVvecT = (
         vvecf[0:sl, ngn:-(ngn), ng:gridparamsf.youtern2]
         - vvecf[0:sl, ngn:-(ngn), ng - 1:gridparamsf.youtern1]
      )
      dVveclimx = reconmodelf.calc_slopelimit(dVvecL, dVvecR)
      dVveclimy = reconmodelf.calc_slopelimit(dVvecB, dVvecT)
      nvar = vvecf.shape[0]
      vvecintLf = jnp.zeros((nvar, gridparamsf.nxifs, gridparamsf.ny + 2 * gridparamsf.ngi))
      vvecintRf = jnp.zeros((nvar, gridparamsf.nxifs, gridparamsf.ny + 2 * gridparamsf.ngi))
      vvecintBf = jnp.zeros((nvar, gridparamsf.nx + 2 * gridparamsf.ngi, gridparamsf.nyifs))
      vvecintTf = jnp.zeros((nvar, gridparamsf.nx + 2 * gridparamsf.ngi, gridparamsf.nyifs))

      vvecintLf = vvecintLf.at[0:sl, ...].set(
         vvecf[0:sl, ng - 1:gridparamsf.outern0, ngn:-(ngn)] + 0.5 * dVveclimx[:, 0:-1, :]
      )
      vvecintRf = vvecintRf.at[0:sl, ...].set(
         vvecf[0:sl, ng:gridparamsf.outern1, ngn:-(ngn)] - 0.5 * dVveclimx[:, 1:, :]
      )
      vvecintBf = vvecintBf.at[0:sl, ...].set(
         vvecf[0:sl, ngn:-(ngn), ng - 1:gridparamsf.youtern0] + 0.5 * dVveclimy[:, :, 0:-1]
      )
      vvecintTf = vvecintTf.at[0:sl, ...].set(
         vvecf[0:sl, ngn:-(ngn), ng:gridparamsf.youtern1] - 0.5 * dVveclimy[:, :, 1:]
      )
      return vvecintLf, vvecintRf, vvecintBf, vvecintTf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1, 2, 3))
   def LRBTreconV_wenoz(vvecf, gridparamsf, physicsmodelf, reconmodelf):
      sl = physicsmodelf.voi
      ngn = gridparamsf.ngnet
      ng = gridparamsf.ng
      gf_Ixmm = vvecf[0:sl, ng - 3:-(ng + 1), ngn:-(ngn)]
      gf_Ixm = vvecf[0:sl, ng - 2:gridparamsf.outern0, ngn:-(ngn)]
      gf_Ix = vvecf[0:sl, ng - 1:gridparamsf.outern1, ngn:-(ngn)]
      gf_Ixp = vvecf[0:sl, ng:gridparamsf.outern2, ngn:-(ngn)]
      gf_Ixpp = vvecf[0:sl, ng + 1:gridparamsf.outern3, ngn:-(ngn)]

      gf_Iymm = vvecf[0:sl, ngn:-(ngn), ng - 3:-(ng + 1)]
      gf_Iym = vvecf[0:sl, ngn:-(ngn), ng - 2:gridparamsf.youtern0]
      gf_Iy = vvecf[0:sl, ngn:-(ngn), ng - 1:gridparamsf.youtern1]
      gf_Iyp = vvecf[0:sl, ngn:-(ngn), ng:gridparamsf.youtern2]
      gf_Iypp = vvecf[0:sl, ngn:-(ngn), ng + 1:gridparamsf.youtern3]

      omegaZ0mx, omegaZ1mx, omegaZ2mx, omegaZ0px, omegaZ1px, omegaZ2px = ReconBase.calc_weno_omegas(
         reconmodelf, gf_Ixmm, gf_Ixm, gf_Ix, gf_Ixp, gf_Ixpp
      )
      omegaZ0my, omegaZ1my, omegaZ2my, omegaZ0py, omegaZ1py, omegaZ2py = ReconBase.calc_weno_omegas(
         reconmodelf, gf_Iymm, gf_Iym, gf_Iy, gf_Iyp, gf_Iypp
      )

      Rres = (
         (omegaZ2mx / 6.0) * (2.0 * gf_Ixpp - 7.0 * gf_Ixp + 11.0 * gf_Ix)
         + (omegaZ1mx / 6.0) * (-1.0 * gf_Ixp + 5.0 * gf_Ix + 2.0 * gf_Ixm)
         + (omegaZ0mx / 6.0) * (2.0 * gf_Ix + 5.0 * gf_Ixm - 1.0 * gf_Ixmm)
      )[:, 1:, :]
      Lres = (
         (omegaZ0px / 6.0) * (2.0 * gf_Ixmm - 7.0 * gf_Ixm + 11.0 * gf_Ix)
         + (omegaZ1px / 6.0) * (-1.0 * gf_Ixm + 5.0 * gf_Ix + 2.0 * gf_Ixp)
         + (omegaZ2px / 6.0) * (2.0 * gf_Ix + 5.0 * gf_Ixp - 1.0 * gf_Ixpp)
      )[:, :-1, :]

      Tres = (
         (omegaZ2my / 6.0) * (2.0 * gf_Iypp - 7.0 * gf_Iyp + 11.0 * gf_Iy)
         + (omegaZ1my / 6.0) * (-1.0 * gf_Iyp + 5.0 * gf_Iy + 2.0 * gf_Iym)
         + (omegaZ0my / 6.0) * (2.0 * gf_Iy + 5.0 * gf_Iym - 1.0 * gf_Iymm)
      )[:, :, 1:]
      Bres = (
         (omegaZ0py / 6.0) * (2.0 * gf_Iymm - 7.0 * gf_Iym + 11.0 * gf_Iy)
         + (omegaZ1py / 6.0) * (-1.0 * gf_Iym + 5.0 * gf_Iy + 2.0 * gf_Iyp)
         + (omegaZ2py / 6.0) * (2.0 * gf_Iy + 5.0 * gf_Iyp - 1.0 * gf_Iypp)
      )[:, :, :-1]

      nvar = vvecf.shape[0]
      vvecintLf = jnp.zeros((nvar, gridparamsf.nxifs, gridparamsf.ny + 2 * gridparamsf.ngi))
      vvecintRf = jnp.zeros((nvar, gridparamsf.nxifs, gridparamsf.ny + 2 * gridparamsf.ngi))
      vvecintBf = jnp.zeros((nvar, gridparamsf.nx + 2 * gridparamsf.ngi, gridparamsf.nyifs))
      vvecintTf = jnp.zeros((nvar, gridparamsf.nx + 2 * gridparamsf.ngi, gridparamsf.nyifs))

      vvecintLf = vvecintLf.at[0:sl, ...].set(Lres)
      vvecintRf = vvecintRf.at[0:sl, ...].set(Rres)
      vvecintBf = vvecintBf.at[0:sl, ...].set(Bres)
      vvecintTf = vvecintTf.at[0:sl, ...].set(Tres)
      return vvecintLf, vvecintRf, vvecintBf, vvecintTf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(0,))
   def calc_weno_omegas(reconmodelf, gf_Imm, gf_Im, gf_I, gf_Ip, gf_Ipp):
      betaZ0 = (13 / 12) * (gf_Imm - 2 * gf_Im + gf_I) ** 2 + (1 / 4) * (gf_Imm - 4 * gf_Im + 3 * gf_I) ** 2
      betaZ1 = (13 / 12) * (gf_Im - 2 * gf_I + gf_Ip) ** 2 + (1 / 4) * (gf_Im - gf_Ip) ** 2
      betaZ2 = (13 / 12) * (gf_I - 2 * gf_Ip + gf_Ipp) ** 2 + (1 / 4) * (3 * gf_I - 4 * gf_Ip + gf_Ipp) ** 2
      tau5 = jnp.abs(betaZ0 - betaZ2)
      aux_alphaZ0 = 1.0 + tau5 / (reconmodelf.weno_eps + betaZ0)
      aux_alphaZ1 = 6.0 + tau5 / (reconmodelf.weno_eps + betaZ1)
      aux_alphaZ2 = 3.0 + tau5 / (reconmodelf.weno_eps + betaZ2)
      alphaZ0m = aux_alphaZ0 * reconmodelf.wenoweights[0]
      alphaZ1m = aux_alphaZ1 * reconmodelf.wenoweights[1]
      alphaZ2m = aux_alphaZ2 * reconmodelf.wenoweights[2]
      alphaZ0p = aux_alphaZ0 * reconmodelf.wenoweights[2]
      alphaZ1p = aux_alphaZ1 * reconmodelf.wenoweights[1]
      alphaZ2p = aux_alphaZ2 * reconmodelf.wenoweights[0]
      sum_alphaZm = alphaZ0m + alphaZ1m + alphaZ2m
      sum_alphaZp = alphaZ0p + alphaZ1p + alphaZ2p
      omegaZ0m = alphaZ0m / sum_alphaZm
      omegaZ1m = alphaZ1m / sum_alphaZm
      omegaZ2m = alphaZ2m / sum_alphaZm
      omegaZ0p = alphaZ0p / sum_alphaZp
      omegaZ1p = alphaZ1p / sum_alphaZp
      omegaZ2p = alphaZ2p / sum_alphaZp
      return omegaZ0m, omegaZ1m, omegaZ2m, omegaZ0p, omegaZ1p, omegaZ2p

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,))
   def FluxCT_Correction(fvecintx, fvecinty, gridparamsf):
      intisl = slice(gridparamsf.ngi, -gridparamsf.ngi) if gridparamsf.ngi >= 1 else slice(0, None)
      intLsl = slice(gridparamsf.ngi - 1, -(gridparamsf.ngi + 1))
      intRsl = slice(gridparamsf.ngi + 1, -(gridparamsf.ngi - 1)) if gridparamsf.ngi >= 2 else slice(gridparamsf.ngi + 1, None)
      fx_by = (
         1
         / 8
         * (
            2 * fvecintx[U.BY, :, intisl]
            + fvecintx[U.BY, :, intRsl]
            + fvecintx[U.BY, :, intLsl]
            - (
               fvecinty[U.BX, 0:-1, 0:-1]
               + fvecinty[U.BX, 1:, 0:-1]
               + fvecinty[U.BX, 0:-1, 1:]
               + fvecinty[U.BX, 1:, 1:]
            )
         )
      )
      fy_bx = (
         1
         / 8
         * (
            2 * fvecinty[U.BX, intisl, :]
            + fvecinty[U.BX, intRsl, :]
            + fvecinty[U.BX, intLsl, :]
            - (
               fvecintx[U.BY, 0:-1, 0:-1]
               + fvecintx[U.BY, 1:, 0:-1]
               + fvecintx[U.BY, 0:-1, 1:]
               + fvecintx[U.BY, 1:, 1:]
            )
         )
      )
      fvecintx = fvecintx.at[U.BY, :, intisl].set(fx_by)
      fvecinty = fvecinty.at[U.BX, intisl, :].set(fy_bx)
      return fvecintx, fvecinty
