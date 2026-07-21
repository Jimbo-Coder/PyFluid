import functools as ft
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as sciopt

from pyfluid.state import MAG_P, MAG_U, MOM_U, P, S, U, VEL_P


cl = 1.0
cl2 = cl * cl


class PhysicsBase:
   @dataclass(frozen=True)
   class PhysicsModel:
      gamma: float
      kappa: float
      cs20: float
      boolmag: bool
      fluxct: bool
      voi: int
      diffparams: tuple
      diffloopparams: tuple
      booldiff: bool
      impadv: bool
      EOS_P: callable
      EOS_eps: callable
      EOS_cs2: callable
      calc_energy: callable
      calc_eps: callable

      con2prim: callable
      prim2con: callable
      calc_anfluxesX: callable
      calc_anfluxesY: callable
      calc_anfluxesX_diff: callable
      calc_anfluxesY_diff: callable
      calc_stress_grad_stdotv: callable

   def __init__(self, parent):
      self.parent = parent
      self.reli = parent.reli
      self.eosi = parent.eosi

      if self.eosi == 0:
         self.EOS_P = self.EOS_P_iso
         self.EOS_cs2 = self.EOS_cs2_iso
         self.EOSname = "Isotherm."
      if self.eosi == 1:
         self.EOS_P = self.EOS_P_ideal
         self.EOSname = "Ideal Gas"
      if self.eosi == 2:
         self.EOS_P = self.EOS_P_polytrope
         self.EOSname = "Polytrope"
      if self.eosi in [0, 1, 2]:
         self.EOS_eps = self.EOS_eps_isoidpoly
      if self.eosi in [1, 2]:
         self.EOS_cs2 = self.EOS_cs2_idpoly
      if self.eosi not in [0, 1, 2]:
         raise ValueError("Equation of State not recognized.")

      if self.reli == 0:
         self.calc_energy = self.calc_energy_NR
         self.calc_eps = self.calc_eps_NR
         self.con2prim = self.con2prim_NR
         self.prim2con = self.prim2con_NR
         self.calc_anfluxesX = self.calc_anfluxesX_NR
         self.calc_anfluxesY = self.calc_anfluxesY_NR
         self.calc_anfluxesX_diff = self.calc_anfluxesX_diff_NR
         self.calc_anfluxesY_diff = self.calc_anfluxesY_diff_NR
         self.calc_stress_grad_stdotv = self.calc_stress_grad_stdotv_NR
         self.relname = "Non-Rel."
      elif self.reli == 1:
         self.con2prim = self.con2prim_SR
         self.prim2con = self.prim2con_SR
         self.calc_anfluxesX = self.calc_anfluxesX_SR
         self.relname = "Spec.Rel."

   def init_diffpars(self):
      self.nu = self.parent.dx * np.sqrt(self.parent.cs20)
      self.mu = self.nu * self.parent.rhob
      self.lam = -(2 / 3) * self.mu
      self.eta = 1e-4
      self.diffparams = tuple([self.nu, self.mu, self.lam, self.eta])
      self.diffloopparams = tuple([self.parent.difftol, self.parent.diffstepmax])

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,))
   def EOS_cs2_iso(rhof, presf, physicsmodelf=None):
      return presf / rhof

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,))
   def EOS_cs2_idpoly(rhof, presf, physicsmodelf):
      return physicsmodelf.gamma * presf / rhof

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,))
   def EOS_P_iso(rho, emd, physicsmodelf):
      return physicsmodelf.cs20 * rho

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,))
   def EOS_P_ideal(rho, emd, physicsmodelf):
      return (physicsmodelf.gamma - 1.0) * rho * emd

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,))
   def EOS_P_polytrope(rhof, emd, physicsmodelf):
      return physicsmodelf.kappa * (rhof**physicsmodelf.gamma)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,))
   def EOS_eps_isoidpoly(Pf, physicsmodelf):
      return Pf / (physicsmodelf.gamma - 1.0)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,))
   def calc_speeds(pvecf, physicsmodelf):
      cs2out = jnp.zeros((4, pvecf.shape[1], pvecf.shape[2]))
      cs2out = cs2out.at[S.SOUND2, ...].set(
         physicsmodelf.EOS_cs2(pvecf[P.RHO, ...], pvecf[P.PRESS, ...], physicsmodelf)
      )
      if physicsmodelf.boolmag:
         cs2out = cs2out.at[S.ALFVEN_X2, ...].set(jnp.square(pvecf[P.BX, ...]) / pvecf[P.RHO, ...])
         cs2out = cs2out.at[S.ALFVEN2, ...].set((jnp.square(pvecf[MAG_P, ...])).sum(axis=0) / pvecf[P.RHO, ...])
         cs2out = cs2out.at[S.FAST2, ...].set(
            0.5
            * (
               cs2out[S.SOUND2, ...]
               + cs2out[S.ALFVEN2, ...]
               + jnp.sqrt(
                  (cs2out[S.SOUND2, ...] + cs2out[S.ALFVEN2, ...]) ** 2
                  - 4 * cs2out[S.SOUND2, ...] * cs2out[S.ALFVEN_X2, ...]
               )
            )
         )
      else:
         cs2out = cs2out.at[S.FAST2, ...].set(cs2out[S.SOUND2, ...])
      return cs2out

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,))
   def con2prim_NR(uvecf, physicsmodelf):
      rhoinv = 1.0 / uvecf[U.RHO, ...]
      pvecout = jnp.zeros_like(uvecf)
      pvecout = pvecout.at[P.RHO, ...].set(uvecf[U.RHO, ...])
      pvecout = pvecout.at[VEL_P, ...].set(uvecf[MOM_U, ...] * rhoinv)
      eps = PhysicsBase.calc_eps_NR(uvecf, physicsmodelf)
      pvecout = pvecout.at[P.PRESS, ...].set(
         physicsmodelf.EOS_P(uvecf[U.RHO, ...], eps * rhoinv, physicsmodelf)
      )
      if physicsmodelf.boolmag:
         pvecout = pvecout.at[MAG_P, ...].set(uvecf[MAG_U, ...])
      return pvecout

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,))
   def prim2con_NR(pvecf, physicsmodelf):
      uvecout = jnp.zeros_like(pvecf)
      uvecout = uvecout.at[U.RHO, ...].set(pvecf[P.RHO, ...])
      uvecout = uvecout.at[MOM_U, ...].set(pvecf[P.RHO, ...] * pvecf[VEL_P, ...])
      uvecout = uvecout.at[U.ENERGY, ...].set(physicsmodelf.calc_energy(pvecf, physicsmodelf))
      if physicsmodelf.boolmag:
         uvecout = uvecout.at[MAG_U, ...].set(pvecf[MAG_P, ...])
      return uvecout

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,))
   def calc_energy_NR(pvecf, physicsmodelf):
      energyout = physicsmodelf.EOS_eps(pvecf[P.PRESS, ...], physicsmodelf)
      energyout = energyout + 0.5 * pvecf[P.RHO, ...] * jnp.square(pvecf[VEL_P, ...]).sum(axis=0)
      if physicsmodelf.boolmag:
         energyout = energyout + 0.5 * jnp.square(pvecf[MAG_P, ...]).sum(axis=0)
      return energyout

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,))
   def calc_eps_NR(uvecf, physicsmodelf):
      eps = uvecf[U.ENERGY, ...] - 0.5 * jnp.square(uvecf[MOM_U, ...]).sum(axis=0) / uvecf[U.RHO, ...]
      if physicsmodelf.boolmag:
         eps = eps - 0.5 * (jnp.square(uvecf[MAG_U, ...]).sum(axis=0))
      return eps

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,))
   def calc_anfluxesX_NR(uvecf, pvecf, physicsmodelf):
      fluxvecf = jnp.zeros_like(uvecf)
      fluxvecf = fluxvecf.at[U.RHO, ...].set(uvecf[U.MOMX, ...])
      fluxvecf = fluxvecf.at[MOM_U, ...].set(uvecf[U.MOMX, None, ...] * pvecf[VEL_P, ...])
      fluxvecf = fluxvecf.at[U.MOMX, ...].add(pvecf[P.PRESS, ...])
      fluxvecf = fluxvecf.at[U.ENERGY, ...].set((uvecf[U.ENERGY, ...] + pvecf[P.PRESS, ...]) * pvecf[P.VX, ...])
      if physicsmodelf.boolmag:
         metot = 0.5 * jnp.square(pvecf[MAG_P, ...]).sum(axis=0)
         fluxvecf = fluxvecf.at[MOM_U, ...].add(-pvecf[P.BX, None, ...] * pvecf[MAG_P, ...])
         fluxvecf = fluxvecf.at[U.MOMX, ...].add(metot)
         fluxvecf = fluxvecf.at[U.ENERGY, ...].add(
            metot * pvecf[P.VX, ...] - pvecf[P.BX, ...] * jnp.sum(pvecf[MAG_P, ...] * pvecf[VEL_P, ...], axis=0)
         )
         fluxvecf = fluxvecf.at[U.BX, ...].set(0.0)
         fluxvecf = fluxvecf.at[U.BY:U.BZ + 1, ...].set(
            pvecf[P.BY:P.BZ + 1, ...] * pvecf[P.VX, ...] - pvecf[P.BX, None, ...] * pvecf[P.VY:P.VZ + 1, ...]
         )
      return fluxvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,))
   def calc_anfluxesY_NR(uvecf, pvecf, physicsmodelf):
      fluxvecf = jnp.zeros_like(uvecf)
      fluxvecf = fluxvecf.at[U.RHO, ...].set(uvecf[U.MOMY, ...])
      fluxvecf = fluxvecf.at[MOM_U, ...].set(uvecf[U.MOMY, None, ...] * pvecf[VEL_P, ...])
      fluxvecf = fluxvecf.at[U.MOMY, ...].add(pvecf[P.PRESS, ...])
      fluxvecf = fluxvecf.at[U.ENERGY, ...].set((uvecf[U.ENERGY, ...] + pvecf[P.PRESS, ...]) * pvecf[P.VY, ...])
      if physicsmodelf.boolmag:
         metot = 0.5 * jnp.square(pvecf[MAG_P, ...]).sum(axis=0)
         fluxvecf = fluxvecf.at[MOM_U, ...].add(-pvecf[P.BY, None, ...] * pvecf[MAG_P, ...])
         fluxvecf = fluxvecf.at[U.MOMY, ...].add(metot)
         fluxvecf = fluxvecf.at[U.ENERGY, ...].add(
            metot * pvecf[P.VY, ...] - pvecf[P.BY, ...] * jnp.sum(pvecf[MAG_P, ...] * pvecf[VEL_P, ...], axis=0)
         )
         fluxvecf = fluxvecf.at[U.BX:U.BZ + 1:2, ...].set(
            pvecf[P.BX:P.BZ + 1:2, ...] * pvecf[P.VY, ...] - pvecf[P.BY, None, ...] * pvecf[P.VX:P.VZ + 1:2, ...]
         )
         fluxvecf = fluxvecf.at[U.BY, ...].set(0.0)
      return fluxvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(4, 5))
   def calc_anfluxesX_diff_NR(uvecf, pvecf, dx_stressf, stressdotv, gridparamsf, physicsmodelf):
      ng = gridparamsf.ng
      fluxvecf = jnp.zeros_like(uvecf)
      fluxvecf = fluxvecf.at[MOM_U, ng:-ng, ng:-ng].set(-dx_stressf[:, 0, ng:-ng, ng:-ng])
      fluxvecf = fluxvecf.at[U.ENERGY, ng:-ng, ng:-ng].set(
         -jnp.gradient(stressdotv[0, ...], gridparamsf.dx, axis=0)[ng:-ng, ng:-ng]
      )
      return fluxvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(4, 5))
   def calc_anfluxesY_diff_NR(uvecf, pvecf, dy_stressf, stressdotv, gridparamsf, physicsmodelf):
      ng = gridparamsf.ng
      fluxvecf = jnp.zeros_like(uvecf)
      fluxvecf = fluxvecf.at[MOM_U, ng:-ng, ng:-ng].set(-dy_stressf[:, 1, ng:-ng, ng:-ng])
      fluxvecf = fluxvecf.at[U.ENERGY, ng:-ng, ng:-ng].set(
         -jnp.gradient(stressdotv[1, ...], gridparamsf.dy, axis=1)[ng:-ng, ng:-ng]
      )
      return fluxvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1, 2))
   def calc_stress_grad_stdotv_NR(pvecf, gridparamsf, physicsmodelf):
      gradvx, gradvy = jnp.gradient(pvecf[VEL_P, ...], gridparamsf.dx, gridparamsf.dy, axis=(1, 2))
      gradvz = jnp.zeros_like(gradvx)
      gradv = jnp.stack((gradvx, gradvy, gradvz), axis=0)
      divv = gradv.sum(axis=0)
      stressf = physicsmodelf.diffparams[1] * (
         gradv + jnp.swapaxes(gradv, 0, 1) - (2 / 3) * jnp.eye(3)[:, :, None, None] * divv
      )
      gradstressxf = jnp.gradient(stressf, gridparamsf.dx, axis=2)
      gradstressyf = jnp.gradient(stressf, gridparamsf.dy, axis=3)
      stressdotvf = jnp.sum(stressf * pvecf[VEL_P, None, ...], axis=0)
      return stressf, gradstressxf, gradstressyf, stressdotvf

   def calc_lorentz_v2(self, v2f):
      return 1 / np.sqrt(1 - (v2f / cl2))

   def calc_lorentz_vec(self, vvecf):
      return self.calc_lorentz_v2(np.square(vvecf).sum(axis=0))

   def calc_hf(self, pvecf):
      eps = self.EOS_eps(pvecf[4, :])
      hf = 1 + (eps + pvecf[4, :]) / pvecf[0, :]
      return hf

   def calc_b2invariant_prim(self, pvecf):
      return np.square(pvecf[5:8, :] / pvecf[8, :]).sum(axis=0) + (np.sum(pvecf[5:8, :] * pvecf[1:4, :], axis=0) ** 2) / cl2

   def calc_b2invariant_con(self, wf, zf, j):
      return self.b2[j] / (wf**2) + (self.sdotb[j] ** 2) * cl2 / (zf**2)

   def c2p_constraint2D_SR(self, uvecf, v2f, Pf, j):
      wt = self.calc_lorentz_v2(v2f)
      rhot = uvecf[0, j] / wt
      epst = self.EOS_eps(Pf) / rhot
      ht = 1 + epst + Pf / rhot
      Zt = uvecf[0, j] * ht * wt
      b2invt = self.calc_b2invariant_con(wt, Zt, j)
      zpb22 = (Zt + b2invt) ** 2
      f1 = (zpb22 * v2f) / cl2 - (
         self.s2[j] + ((self.sdotb[j]) ** 2) * (2 * Zt + b2invt + self.b2[j]) / zpb22
      )
      f2 = Zt - uvecf[0, j] * cl2 - (uvecf[4, j] + Pf + self.b2[j] / 2.0)
      return np.array([f1, f2])

   def c2p_constraint1D_SR(self, uvecf, WCF, j):
      v2w = (self.s2[j] + (self.sdotb[j] ** 2) * (2 * WCF + self.b2[j]) / (WCF**2)) / (WCF + self.b2[j]) ** 2
      Wf = self.calc_lorentz_v2(v2w)
      rhof = uvecf[0, j] / Wf
      hf = WCF / (uvecf[0, j] * Wf)
      if self.eosi in [0, 1]:
         epsf = (hf - 1) / self.gamma
      elif self.eosi == 2:
         epsf = hf - 1 - self.EOS_P(rhof, 0) / rhof
      presf = self.EOS_P(rhof, epsf)
      return WCF - presf - uvecf[0, j] * cl2 - uvecf[4, j]

   def con2prim_SR(self, uvecf, pvecout):
      pvecout[5:8, :] = uvecf[5:8, :]
      self.s2 = np.square(uvecf[1:4, :]).sum(axis=0)
      self.b2 = np.square(uvecf[5:8, :]).sum(axis=0)
      self.d2 = np.square(uvecf[0, :])
      self.sdotb = np.sum(uvecf[1:4, :] * uvecf[5:8, :], axis=0)
      Wguess = np.sqrt(1 + self.s2 / self.d2)
      vguess = np.sqrt(1 - 1 / (Wguess**2)) * cl
      Pguess = (self.gamma - 1.0) * (uvecf[4, :] - self.s2 / (2 * uvecf[0, :]) - self.b2 / 2.0)
      WCFguess = np.sqrt(self.s2 + self.d2 + self.b2 + (self.sdotb**2) / self.d2)
      f0 = lambda rvec, j: self.c2p_constraint1D_SR(uvecf, rvec[0], j)
      f1 = lambda rvec, j: self.c2p_constraint2D_SR(uvecf, rvec[0], rvec[1], j)
      for j in range(len(uvecf[0, :])):
         sol1d = sciopt.root(f0, (WCFguess[j]), args=(j), tol=1e-12)
         if sol1d.success:
            WCFres = sol1d.x[0]
            v2res = (self.s2[j] + (self.sdotb[j] ** 2) * (2 * WCFres + self.b2[j]) / (WCFres**2)) / (
               WCFres + self.b2[j]
            ) ** 2
            Wres = self.calc_lorentz_v2(v2res)
            rhores = uvecf[0, j] / Wres
            hf = WCFres / (uvecf[0, j] * Wres)
            if self.eosi in [0, 1]:
               epsf = (hf - 1) / self.gamma
            elif self.eosi == 2:
               epsf = hf - 1 - self.EOS_P(rhores, 0) / rhores
            Pres = self.EOS_P(uvecf[0, j], epsf)
         if not sol1d.success:
            sol2d = sciopt.root(f1, (vguess[j], Pguess[j]), args=(j))
            v2res = sol2d.x[0]
            Pres = sol2d.x[1]
            Wres = self.calc_lorentz_v2(v2res)
            rhores = uvecf[0, j] / Wres

         pvecout[0, j] = rhores
         pvecout[4, j] = Pres
         hfres = self.calc_hf(pvecout[:, j, np.newaxis])
         Zres = uvecf[0, j] * hfres * Wres
         b2invres = self.calc_b2invariant_con(Wres, Zres, j)

         denom = Zres + b2invres
         numerator = uvecf[1:4, j] + (self.sdotb[j] * uvecf[5:8, j]) / (denom * cl2)
         pvecout[1:4, j] = numerator / denom
         pvecout[8, j] = Wres

   def prim2con_SR(self, pvecf, uvecout):
      b2inv = self.calc_b2invariant_prim(pvecf)
      hf = self.calc_hf(pvecf)
      htot = hf + b2inv / pvecf[0, :]
      pW = pvecf[0, :] * pvecf[8, :]
      uvecout[0, :] = pW
      uvecout[1:4, :] = pW * pvecf[8, :] * htot * pvecf[1:4, :] / cl2 - (
         pvecf[5:8, :] * np.sum(pvecf[5:8, :] * pvecf[1:4, :], axis=0)
      ) / cl2
      uvecout[4, :] = pW * (pvecf[8, :] * htot - cl2) - pvecf[4, :] - b2inv / 2.0
      uvecout[5:8, :] = pvecf[5:8, :]

   def calc_anfluxes_SR(self, uvecf, pvecf, fluxvecf):
      metot = 0.5 * np.square(pvecf[5:8, :]).sum(axis=0)
      hf = self.calc_hf(pvecf)
      pW = pvecf[0, :] * pvecf[8, :]
      pW2 = pW * pvecf[8, :]
      phW2 = hf * pW2
      fluxvecf[0, :] = pW * pvecf[1, :]
      fluxvecf[1:4, :] = phW2 * pvecf[1:4, :] * pvecf[1, :] - pvecf[5, np.newaxis, :] * pvecf[5:8, :]
      fluxvecf[1, :] += pvecf[4, :] + metot
      fluxvecf[4, :] = (phW2 * pvecf[1, :] - pvecf[4, :]) * pvecf[1, :] - pvecf[5, :] * np.sum(
         pvecf[5:8, :] * pvecf[1:4, :], axis=0
      )
      fluxvecf[5, :] = 0.0
      fluxvecf[6:8, :] = pvecf[1, :] * pvecf[6:8, :] - pvecf[5, :] * pvecf[2:4, :]
