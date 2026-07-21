import functools as ft
from dataclasses import dataclass

import jax

from pyfluid.state import P


class BoundaryConditionBase:
   @dataclass(frozen=True)
   class BCModel:
      alltupF : tuple
      lhgtupF : tuple
      rhgtupF : tuple
      lhitupF : tuple
      rhitupF : tuple
      ng : int

      bcLfunc : callable
      bcRfunc : callable
      bcBfunc : callable
      bcTfunc : callable

   def __init__(self,parent):
      self.parent = parent
      self.ng = parent.ng
      self.lhgslF = parent.lhgslF
      self.lhislF = parent.lhislF
      self.rhgslF = parent.rhgslF
      self.rhislF = parent.rhislF
      self.bci = parent.bci

      assert(len(self.bci) == 4)

      BC_L_List = [self.periodic_L, self.outflow_L, self.reflective_L]
      BC_R_List = [self.periodic_R, self.outflow_R, self.reflective_R]
      BC_B_List = [self.periodic_B, self.outflow_B, self.reflective_B]
      BC_T_List = [self.periodic_T, self.outflow_T, self.reflective_T]
      self.bcLfunc = BC_L_List[self.bci[0]]
      self.bcRfunc = BC_R_List[self.bci[1]]
      self.bcBfunc = BC_B_List[self.bci[2]]
      self.bcTfunc = BC_T_List[self.bci[3]]
      if self.bci[0] not in [0,1,2] or self.bci[1] not in [0,1,2] or self.bci[2] not in [0,1,2] or self.bci[3] not in [0,1,2]:
         raise ValueError("Boundary condition not recognized. Use 0 = periodic, 1 = outflow, 2 = reflective")

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def apply_bcs(pvecf, BCmodelf):
      pvecf = BCmodelf.bcLfunc(pvecf, BCmodelf)
      pvecf = BCmodelf.bcRfunc(pvecf, BCmodelf)
      pvecf = BCmodelf.bcBfunc(pvecf, BCmodelf)
      pvecf = BCmodelf.bcTfunc(pvecf, BCmodelf)
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def periodic_L(pvecf, BCmodelf):
         lhgtup = BCmodelf.lhgtupF; rhitup = BCmodelf.rhitupF; alltup = BCmodelf.alltupF;
         pvecf = pvecf.at[:,lhgtup[0]:lhgtup[1], alltup[0]:alltup[1]].set(pvecf[:,rhitup[0]:rhitup[1], alltup[0]:alltup[1]])
         return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def periodic_R(pvecf, BCmodelf):
      rhgtup = BCmodelf.rhgtupF; lhitup = BCmodelf.lhitupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:,rhgtup[0]:rhgtup[1], alltup[0]:alltup[1]].set(pvecf[:,lhitup[0]:lhitup[1], alltup[0]:alltup[1]])
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def periodic_T(pvecf, BCmodelf):
      rhgtup = BCmodelf.rhgtupF; lhitup = BCmodelf.lhitupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:, alltup[0]:alltup[1], rhgtup[0]:rhgtup[1]].set(pvecf[:, alltup[0]:alltup[1], lhitup[0]:lhitup[1]])
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def periodic_B(pvecf, BCmodelf):
      lhgtup = BCmodelf.lhgtupF; rhitup = BCmodelf.rhitupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:, alltup[0]:alltup[1], lhgtup[0]:lhgtup[1]].set(pvecf[:, alltup[0]:alltup[1], rhitup[0]:rhitup[1]])
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def outflow_L(pvecf, BCmodelf):
      lhgtup = BCmodelf.lhgtupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:,lhgtup[0]:lhgtup[1], alltup[0]:alltup[1]].set(pvecf[:,BCmodelf.ng,:][:,None,:])
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def outflow_R(pvecf, BCmodelf):
      rhgtup = BCmodelf.rhgtupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:,rhgtup[0]:rhgtup[1], alltup[0]:alltup[1]].set(pvecf[:,-(BCmodelf.ng+1),:][:,None,:])
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def outflow_T(pvecf, BCmodelf):
      rhgtup = BCmodelf.rhgtupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:, alltup[0]:alltup[1], rhgtup[0]:rhgtup[1]].set(pvecf[:,:,-(BCmodelf.ng+1)][:,:,None])
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def outflow_B(pvecf, BCmodelf):
      lhgtup = BCmodelf.lhgtupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:, alltup[0]:alltup[1], lhgtup[0]:lhgtup[1]].set(pvecf[:,:,BCmodelf.ng][:,:,None])
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def reflective_L(pvecf, BCmodelf):
      lhgtup = BCmodelf.lhgtupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:,lhgtup[0]:lhgtup[1], alltup[0]:alltup[1]].set(pvecf[:,BCmodelf.ng,:][:,None,:])
      pvecf = pvecf.at[P.VX,lhgtup[0]:lhgtup[1], alltup[0]:alltup[1]].multiply(-1.0)
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def reflective_R(pvecf, BCmodelf):
      rhgtup = BCmodelf.rhgtupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:,rhgtup[0]:rhgtup[1], alltup[0]:alltup[1]].set(pvecf[:,-(BCmodelf.ng+1),:][:,None,:])
      pvecf = pvecf.at[P.VX,rhgtup[0]:rhgtup[1], alltup[0]:alltup[1]].multiply(-1.0)
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def reflective_T(pvecf, BCmodelf):
      rhgtup = BCmodelf.rhgtupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:, alltup[0]:alltup[1], rhgtup[0]:rhgtup[1]].set(pvecf[:,:,-(BCmodelf.ng+1)][:,:,None])
      pvecf = pvecf.at[P.VY, alltup[0]:alltup[1], rhgtup[0]:rhgtup[1]].multiply(-1.0)
      return pvecf

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1))
   def reflective_B(pvecf, BCmodelf):
      lhgtup = BCmodelf.lhgtupF; alltup = BCmodelf.alltupF;
      pvecf = pvecf.at[:, alltup[0]:alltup[1], lhgtup[0]:lhgtup[1]].set(pvecf[:,:,BCmodelf.ng][:,:,None])
      pvecf = pvecf.at[P.VY, alltup[0]:alltup[1], lhgtup[0]:lhgtup[1]].multiply(-1.0)
      return pvecf
