import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # "gpu" or "cpu"
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
import h5py
import pathlib as path
import numpy as np

from tqdm import tqdm
import time
import functools as ft
from pyfluid.boundaries import BoundaryConditionBase
from pyfluid.checkpointing import compare_parameter_dicts
from pyfluid.config import SimulationConfig, parse_config_args
from pyfluid.diagnostics import plot_field_snapshot, plot_scalar_timeseries
from pyfluid.initial_data import (
   BALSARA_INITIAL_CONDITIONS,
   PLOT_BALSARA_INITIAL_CONDITIONS,
   ic_has_magnetic_fields,
   initial_condition_name,
)
from pyfluid.physics import PhysicsBase
from pyfluid.reconstruction import ReconBase
from pyfluid.riemann import RiemannBase
from pyfluid.state import D, DIAGNOSTIC_NAMES, MAG_P, MAG_U, MOM_U, NVAR_HYDRO, NVAR_MHD, P, PVAR_HYDRO, PVAR_MHD, PVAR_RMHD, P_NAMES, U, U_NAMES, VEL_P

maindir = path.Path(__file__).resolve().parent
plotdir = maindir / "plots"
exactdir = maindir / "exact_balsara"
checkdir = maindir / "checkpoints"


def configure_jax(config):
   jax.config.update("jax_enable_x64", config.x64)


def describe_jax_devices():
   print(f'Using jax device {jax.devices()}')
   for platform in ("cpu", "gpu"):
      try:
         print(f"{platform}: {jax.devices(platform)}")
      except RuntimeError as err:
         print(f"{platform}: unavailable ({err})")

def calc_v4(p3f,selff):
   return (p3f - selff.presR)*np.sqrt((1 - selff.gammaex)/(selff.rhoR*(p3f + selff.gammaex*selff.presR)))

def calc_v3(p3f,selff):
   return ((selff.presL**selff.betaex) - (p3f**selff.betaex))*np.sqrt(((1 - selff.gammaex**2)*(selff.presL**(1/selff.gamma))/((selff.gammaex**2)*selff.rhoL)))

def calc_root_pres3(p3f,selff):
   return(calc_v3(p3f,selff) - calc_v4(p3f,selff))

class Fluid2DMHD:
   def __init__(self, config):
      #Initialize the base grid, ghost cells and interior.
      if not isinstance(config, SimulationConfig):
         raise TypeError("Fluid2DMHD expects a SimulationConfig")
      self.config = config
      self.pardict = config.to_checkpoint_params()
      self.float_tol = 1e-14 if config.x64 else 1e-5
      self.numpy_dtype = np.float64 if config.x64 else np.float32
      self.nsteps = int(config.nsteps)

      self.x0,self.x1 = config.x0,config.x1
      self.y0,self.y1 = config.y0,config.y1
      self.nx = int(config.nx); self.ny = int(config.ny);
      self.ici = config.ic; self.bci = np.array(config.bc, dtype=int); self.reconi = config.recon;
      self.fluxi = config.riemann; self.timestepi = config.timestep; self.reconvari = config.recon_vars;
      self.eosi = config.eos; self.reli = config.relativity; self.fluxct = config.fluxct;
      self.store_every = config.store_every;
      self.checkname = config.checkname; self.save_final_state = config.save_final_state; self.recover_state = config.recover_state;
      self.save_hdf5 = config.save_hdf5;
      self.verbose = config.verbose

      self.checkdirsim = checkdir / self.checkname;


      self.booldiff = config.diffusion
      self.delrhofac = config.delrho_factor
      self.ng = config.ng #number of ghost cells
      self.ngi = config.ngi #number of interior ghost cells
      self.limi = config.limiter
      self.cflr = config.cfl #CFL ratio
      self.Bx0 = config.bx0 #background magnetic field in x direction
      self.By0 = config.by0
      self.Bz0 = config.bz0
      self.mhd_enabled = bool(config.force_mhd or ic_has_magnetic_fields(self.ici) or self.Bx0 != 0.0 or self.By0 != 0.0 or self.Bz0 != 0.0)

      
      if self.timestepi == 0 and self.reconi >=1:
         print(f"Warning: Euler Time step with Linear or better reconstruction can be unstable.")
      assert(self.ng > self.ngi)
      self.outerd0 = - (self.ngi) if self.ngi >=1 else None
      self.intisl = slice(self.ngi, -self.ngi) if self.ngi >=1 else slice(0,None);
      if self.fluxct:
         assert(self.ngi >=1)

      self.build_grid()
   

      
      self.allocate_arrays()
      
      
      #Toggle recon in Prims or Cons
      assert(self.reconvari in [0,1])
      if self.reconvari == 0:
         self.reconvarname = "Prims."
         self.dVvecL = self.dPvecL; self.dVvecR = self.dPvecR; self.dVveclimx = self.dPveclimx;
         self.dVvecB = self.dPvecB; self.dVvecT = self.dPvecT; self.dVveclimy = self.dPveclimy;
         self.LRBT_recon_UP = self.LRBT_recon_PSU
      elif self.reconvari == 1:
         self.reconvarname = "Cons."
         self.dVvecL = self.dUvecL; self.dVvecR = self.dUvecR; self.dVveclimx = self.dUveclimx;
         self.dVvecB = self.dUvecB; self.dVvecT = self.dUvecT; self.dVveclimy = self.dUveclimy; 
         self.LRBT_recon_UP = self.LRBT_recon_USP
      self.reconbase = ReconBase(self)
      self.reconmodel = self.reconbase.ReconModel(self.reconbase.LRBTreconV, self.LRBT_recon_UP, self.reconbase.calc_slopelimit,\
                                                  self.reconbase.wenoweights, self.reconbase.weno_eps)
      self.riemannbase = RiemannBase(self)
      self.calc_riemann_fluxes = self.riemannbase.calc_riemann_fluxes
      self.riemanname = self.riemannbase.name
      self.impadv = False
      if self.timestepi == 0:
         self.step_in_time = self.euler_step
         self.timestepname = "Euler"
      elif self.timestepi == 1:
         self.step_in_time = self.corrector_step
         self.timestepname = "Corrector"
      elif self.timestepi == 2:
         self.step_in_time = self.midpoint_step
         self.timestepname = "Midpoint"
      elif self.timestepi == 3:
         self.step_in_time = self.rk4_step
         self.timestepname = "RK4"
      elif self.timestepi == 4:
         self.step_in_time = self.ssprk3_step
         self.timestepname = "SSPRK3"
      elif self.timestepi == 5:
         raise NotImplementedError("Implicit Euler/IMEX path is not wired into the current JAX run loop.")
      elif self.timestepi not in [0,1,2,3,4]:
         raise ValueError("Time step method not recognized.")

      #Physics Assumptions
      #self.gamma = 1.4 #adiabatic index
      self.gamma = self.numpy_dtype(5.0/3.0) #adiabatic index
      self.kappa = self.numpy_dtype(1.0) #entropy

      self.rhob = self.numpy_dtype(1.0) #background density
      self.presb = self.numpy_dtype(1.0);

      self.difftol = 1e-13
      self.advtol =1e-13 

      self.differr = 0.0; self.adverr = 0.0; self.totierr = 0.0;
      self.diffstep = 0; self.advstep = 0; self.advstep2 = 0;
      self.diffstepmax = 1e3; self.advstepmax = 1e3; self.advstepmax2 = 1e3;
      self.physicsbase = PhysicsBase(self)
      if self.eosi >=2:
         self.presb = self.numpy_dtype(self.kappa * (self.rhob)**self.gamma)
      if self.eosi == 0: # isothermal
         self.cs20 = self.numpy_dtype(self.presb/self.rhob)
      elif self.eosi in [1,2]: #ideal gas or polytrope
         self.cs20 = self.numpy_dtype(self.gamma*self.presb/self.rhob)

      self.physicsbase.init_diffpars() #initialize diffusion parameters after cs20 is known
      self.cs0 = self.numpy_dtype(np.sqrt(self.cs20)) #background sound speed
      #Diffusion stuff
      self.Uvect = np.copy(self.Uvec) #temporary conserved variable state vector
      self.Pvect = np.copy(self.Pvec) #temporary primitive variable state vector
     
      


      self.boundaryconditionsbase = BoundaryConditionBase(self)
      self.BCmodel = self.boundaryconditionsbase.BCModel(self.alltupF,self.lhgtupF,self.rhgtupF,self.lhitupF,self.rhitupF,self.ng, \
                                                         self.boundaryconditionsbase.bcLfunc, self.boundaryconditionsbase.bcRfunc,\
                                                         self.boundaryconditionsbase.bcBfunc, self.boundaryconditionsbase.bcTfunc)

      self.t = 0.0; self.dt = 0.0;self.step= 0;self.storestep =0; self.step0 = 0;
      #initialize conserved variables

      self.sim_number = 0
   
      self.apply_ics()

      if self.recover_state:
         self.load_checkpoint()
      
      #Save initial state
      self.Uvecs[0,:,:] = self.Uvec
      self.Pvecs[0,:,:] = self.Pvec
      self.ts[0] = self.t 


      #Main Update operations
      self.update_operations = [self.step_in_time, lambda: self.apply_bcsU(self.Uvec,self.Pvec)]

      self.combined_LU_operations = [lambda x: self.advective_nLU(x, self.dUvecadv)]

      # Optional diffusion update. The implicit-advection branch is guarded at
      # timestep selection and is not part of the current JAX run path.
      if self.booldiff:
         if (not self.impadv):
            self.update_operations.append(self.diff_update)
            self.update_operations.append(lambda: self.apply_bcsU(self.Uvec,self.Pvec))
         else:
            self.combined_LU_operations.append(lambda x: self.diffusion_nLU(x, self.dUvecim))
         self.calc_differr = self.calc_operr;
      else:
         self.calc_differr = lambda uvec1d,uvec2d,duvecd,dtd: 0.0;
      
      self.calc_adverr = self.calc_operr;


      
      #Info string
      self.info = self.physicsbase.EOSname + ' ' + self.icname + ' ' + self.reconvarname +' '+ self.reconbase.reconnames + ' ' + self.riemanname + ' ' + self.timestepname

      #Main Run
      self.start_time = time.time()
      self.run_and_store()
      self.end_time = time.time()
      self.run_time = self.end_time - self.start_time;

      #After Run is done
      self.compute_diagnostics(self.dx,self.Uvecs,self.Pvecs,self.diagnostics,self.imasksl)

      
      #Rebuild exact solution for all variables :)
      if self.ici==0:
         self.calc_exact_sod()
      if self.ici in PLOT_BALSARA_INITIAL_CONDITIONS:
         #self.plotbalsara = True;
         #self.calc_exact_balsara()
         self.plotbalsara = False;
      else:
         self.plotbalsara = False;

      if self.save_final_state:
         self.save_final_checkpoint()
      if self.save_hdf5:
         self.save_hdf5_outputs()

   def build_grid(self):
      self.nxtot = self.nx + 2*self.ng #total number of grid points along x
      self.ngnet = self.ng - self.ngi
      self.nytot = self.ny + 2*self.ng #total number of grid points along y
      self.ntot2d = (self.nxtot,self.nytot)

      self.Lx = self.x1-self.x0; self.Ly = self.y1-self.y0;
      self.dx = self.Lx/ (self.nx); self.inv_dx = 1.0/self.dx;
      self.dy = self.Ly/ (self.ny); self.inv_dy = 1.0/self.dy;
      self.xg1d = jnp.linspace(self.x0-(self.ng-0.5)*self.dx,self. x1+(self.ng-0.5)*self.dx,self.nxtot, endpoint=True) #symmetric grid with ghost cells
      self.yg1d = jnp.linspace(self.y0-(self.ng-0.5)*self.dy,self. y1+(self.ng-0.5)*self.dy,self.nytot, endpoint=True) #symmetric grid with ghost cells
      self.xg, self.yg = jnp.meshgrid(self.xg1d,self.yg1d,indexing='ij') #2D grid with ghost cells
      self.Lxcc = self.xg[-self.ng-1,0] - self.xg[self.ng,0] #cell centered domain length for smooth solutions
      self.Lycc = self.yg[0,-self.ng-1] - self.yg[0,self.ng]
      self.gtol = self.float_tol
      assert(jnp.isclose(self.xg1d[1]-self.xg1d[0], self.dx, rtol = self.gtol))
      assert(jnp.isclose(self.yg1d[1]-self.yg1d[0], self.dy, rtol = self.gtol))

      self.gmask = jnp.zeros(self.ntot2d,dtype=bool);
      self.gmask = self.gmask.at[:self.ng, :].set(True);
      self.gmask = self.gmask.at[-self.ng:, :].set(True);
      self.gmask = self.gmask.at[:, :self.ng].set(True);
      self.gmask = self.gmask.at[:, -self.ng:].set(True);
      self.imask = ~self.gmask
      self.gmask1d = jnp.zeros(self.nxtot,dtype=bool);
      self.gmask1d = self.gmask1d.at[:self.ng].set(True);
      self.gmask1d = self.gmask1d.at[-self.ng:].set(True);
      self.imask1d = ~self.gmask1d
      
      self.xint = self.xg[self.imask] #interior grid
      self.yint = self.yg[self.imask]
      self.nxifs = self.nx + 1 #number of interfaces
      self.nyifs = self.ny + 1 
      self.xgi1d = jnp.linspace(self.xg1d[self.ng]-(0.5)*self.dx,self.xg1d[-self.ng-1]+(0.5)*self.dx,self.nxifs,endpoint=True) #grid interfaces
      self.ygi1d = jnp.linspace(self.yg1d[self.ng]-(0.5)*self.dy,self.yg1d[-self.ng-1]+(0.5)*self.dy,self.nyifs,endpoint=True) #grid interfaces
      
      self.xgi, self.ygi = jnp.meshgrid(self.xgi1d,self.ygi1d,indexing='ij') #2D grid interfaces
   

      assert(jnp.isclose(self.xgi1d[1]-self.xgi1d[0], self.dx, rtol = self.gtol))
      assert(jnp.isclose(self.ygi1d[1]-self.ygi1d[0], self.dy, rtol = self.gtol))
       #Interior domain length
      assert(jnp.isclose((self.nx)*self.dx,self.Lx, rtol = self.gtol))
      assert(jnp.isclose((self.ny)*self.dy,self.Ly, rtol = self.gtol))
      resfac = 1;
      self.nxhr = resfac*self.nx #number of points for high res grid
      self.nyhr = resfac*self.ny #number of points for high res grid
      self.dxhr = self.Lx/ (self.nxhr)
      self.dyhr = self.Ly/ (self.nyhr)
      self.xghr1d = jnp.linspace(self.x0 + 0.5*self.dxhr,self.x1 - 0.5*self.dxhr,self.nxhr, endpoint=True) #high res grid
      self.yghr1d = jnp.linspace(self.y0 + 0.5*self.dyhr,self.y1 - 0.5*self.dyhr,self.nyhr, endpoint=True) #high res grid
      self.xghr, self.yghr = jnp.meshgrid(self.xghr1d,self.yghr1d,indexing='ij') #2D high res grid
      assert(jnp.isclose(self.xghr1d[1]-self.xghr1d[0], self.dxhr, rtol = self.gtol))
      assert(jnp.isclose(self.yghr1d[1]-self.yghr1d[0], self.dyhr, rtol = self.gtol))

      self.imasksl1D = slice(self.ng,-self.ng) #interior mask slice
      self.imasksl = (self.imasksl1D,self.imasksl1D) #2D interior mask slice

      self.lhgslF1D = slice(0,self.ng) #left side ghost cell slice (L2R) \
      self.allpslF = slice(0,None)
      self.lhgslF = (self.lhgslF1D,self.allpslF) #2D left side ghost cell slice (L2R)
      self.bhgslF = (self.allpslF,self.lhgslF1D) #2D bottom side ghost cell slice (B2T)

      self.lhislF1D = slice(self.ng,2*self.ng) #left side interior slice (L2R)\
      self.lhislF = (self.lhislF1D,self.allpslF) #2D left side interior slice (L2R)
      self.bhislF = (self.allpslF,self.lhislF1D) #2D bottom side interior slice (B2T)

      self.rhgslF1D = slice(-self.ng,None) #right side ghost cell slice (L2R)
      self.rhgslF = (self.rhgslF1D,self.allpslF) #2D right side ghost cell slice (L2R)
      self.thgslF = (self.allpslF,self.rhgslF1D) #2D top side ghost cell slice (B2T)

      self.rhislF1D = slice(-2*self.ng,-self.ng) #right side interior slice (L2R)
      self.rhislF = (self.rhislF1D,self.allpslF) #2D right side interior slice (L2R)
      self.thislF = (self.allpslF,self.rhislF1D) #2D top side interior slice (B2T)
 
      self.lhsmask = (self.xg <= 0) #left half domain
      self.lhsmask = self.lhsmask.at[self.gmask].set(False)
      self.rhsmask = (self.xg > 0 ) #right half domain
      self.rhsmask = self.rhsmask.at[self.gmask].set(False)
      self.bhsmask = (self.yg <= 0) #bottom half domain
      self.bhsmask = self.bhsmask.at[self.gmask].set(False)
      self.thsmask = (self.yg > 0 ) #top half domain
      self.thsmask = self.thsmask.at[self.gmask].set(False)

      self.lhgtupF = (0, self.ng)
      self.lhitupF = (self.ng, 2*self.ng)
      self.rhgtupF = (-self.ng, None)
      self.rhitupF = (-2*self.ng, -self.ng)
      self.alltupF = (0, self.nxtot)

      @dataclass(frozen=True)
      class GridParams:
         nx : int #number of interior grid points
         ny : int #number of interior grid points
         nxtot : int #total number of grid points
         nytot : int #total number of grid points
         nxifs : int #number of grid interfaces (including ghost)
         nyifs : int #number of grid interfaces (including ghost)
         dx : float #grid spacing
         dy : float #grid spacing
         cflr : float #CFL number
         inv_dx : float #1/dx
         inv_dy : float #1/dy
         ng : int #number of ghost cells
         ngi : int #number of interface ghost cells
         ngnet : int #net ghost cells for recon (i.e. ng - ngi)
         outerd0 : int #outer index for interface
         outer0 : int #outer index to give full reconstructed interior
         outer1 : int #outer index to give full reconstructed interior + 1
         outer2: int #outer index to give full reconstructed interior + 2
         outer3: int #outer index to give full reconstructed interior + 3
         outern0: int #outer index for regular
         outern1 : int #outer index for regular + 1
         outern2 : int #outer index for regular + 2
         outern3 : int #outer index for regular + 3
         youterd0 : int #outer y index for interface
         youter0 : int #outer y index to give full reconstructed interior
         youter1 : int #outer y index to give full reconstructed interior + 1
         youter2: int #outer y index to give full reconstructed interior + 2
         youter3: int #outer y index to give full reconstructed interior + 3
         youtern0: int #outer y index for regular
         youtern1 : int #outer y index for regular + 1
         youtern2 : int #outer y index for regular + 2
         youtern3 : int #outer y index for regular + 3
      self.gridparams = GridParams(self.nx, self.ny, self.nxtot, self.nytot, self.nxifs, self.nyifs, self.dx,self.dy,self.cflr, self.inv_dx,self.inv_dy, self.ng, self.ngi,self.ngnet, self.nxifs - self.ngi\
                                   , self.nxtot - self.ngnet, self.nxtot - self.ngnet +1, self.nxtot - self.ngnet +2, self.nxtot - self.ngnet +3\
                                    , self.nxtot - self.ng, self.nxtot - self.ng + 1, self.nxtot - self.ng +2, self.nxtot - self.ng +3\
                                    , self.nyifs - self.ngi, self.nytot - self.ngnet, self.nytot - self.ngnet +1, self.nytot - self.ngnet +2, self.nytot - self.ngnet +3\
                                    , self.nytot - self.ng, self.nytot - self.ng + 1, self.nytot - self.ng +2, self.nytot - self.ng +3)

   def allocate_arrays(self):

      # Hydro uses 5 variables; MHD adds Bx, By, Bz.
      self.nvar = NVAR_MHD if self.mhd_enabled else NVAR_HYDRO;
      self.nvarnames = list(U_NAMES[:self.nvar])
      self.stagintx = (self.nxifs, self.ny + 2*self.ngi) #x staggered grid
      self.staginty = (self.nx+ 2*self.ngi, self.nyifs ) #y stagger
     
      self.Uvecshape = (self.nvar, *self.ntot2d)
      self.Uvecintxshape = (self.nvar, *self.stagintx)
      self.Uvecintyshape = (self.nvar, *self.staginty)
      self.dUvecxshape = (self.nvar, self.nx+2, self.ny + 2*self.ngi)
      self.dUvecyshape = (self.nvar, self.nx + 2*self.ngi, self.ny + 2)

      self.Uvec = jnp.zeros(self.Uvecshape) #conserved variable state vector
      self.UvecintxL= jnp.zeros(self.Uvecintxshape) #interface conserved variable state vector
      self.UvecintxR= jnp.zeros(self.Uvecintxshape) #interface conserved variable state vector
      self.UvecintyB= jnp.zeros(self.Uvecintyshape) #interface conserved variable state vector
      self.UvecintyT= jnp.zeros(self.Uvecintyshape) #interface conserved variable
      self.Uvecintx= jnp.zeros(self.Uvecintxshape) #interface conserved variable state vector
      self.Uvecinty= jnp.zeros(self.Uvecintyshape) #interface conserved variable state vector


      self.Uvectest = jnp.zeros(self.Uvecshape) #test conserved variable state vector

      self.Uvech = jnp.zeros(self.Uvecshape) #half step conserved variable state vector
      self.Uvech2 = jnp.zeros(self.Uvecshape) 
      self.dUvec = jnp.zeros(self.Uvecshape) #change of conserved variables
      self.dUvecim = jnp.zeros(self.Uvecshape) #change of conserved variables for implicit step
      self.dUvecimold = jnp.zeros(self.Uvecshape) #old change of conserved variables for implicit step
      self.dUvecadv = jnp.zeros(self.Uvecshape) #change of conserved variables for advection step
      self.dUvecadvold = jnp.zeros(self.Uvecshape) #old change of conserved variables for advection step
      self.dUvech = jnp.zeros(self.Uvecshape) #change in conserved variables at half step
      self.dUveck1 = jnp.zeros(self.Uvecshape) #k1 for rk4
      self.dUveck2 = jnp.zeros(self.Uvecshape) #k2 for rk4
      self.dUveck3 = jnp.zeros(self.Uvecshape) #k3 for rk4
      self.dUveck4 = jnp.zeros(self.Uvecshape) #k4 for rk4
      self.Uveck1 = jnp.zeros(self.Uvecshape) #U + k1/2 for rk4
      self.Uveck2 = jnp.zeros(self.Uvecshape) #U + k2/2 for rk4
      self.Uveck3 = jnp.zeros(self.Uvecshape) #U + k3 for rk4

      self.dUvecL = jnp.zeros(self.dUvecxshape) #left slope of conserved variables (for conserved recon)
      self.dUvecR = jnp.zeros(self.dUvecxshape) #right slope of conserved variables (for conserved recon)
      self.dUvecB = jnp.zeros(self.dUvecyshape) #down slope of conserved variables (for conserved recon)
      self.dUvecT = jnp.zeros(self.dUvecyshape) #up slope of conserved variables (for conserved recon)
      self.dUveclimx = jnp.zeros(self.dUvecxshape) #slope limited slope of conserved variables (for conserved recon)
      self.dUveclimy = jnp.zeros(self.dUvecyshape) #slope limited slope of conserved variables (for conserved recon)



      self.Fvec = jnp.zeros(self.Uvecshape) #flux vector
      self.Fvecintx = jnp.zeros(self.Uvecintxshape) #interface flux vector
      self.FvecintxL = jnp.zeros(self.Uvecintxshape) #interface left flux vector
      self.FvecintxR = jnp.zeros(self.Uvecintxshape) #interface right flux vector

      self.Fvecinty = jnp.zeros(self.Uvecintyshape) #interface flux vector
      self.FvecintyB = jnp.zeros(self.Uvecintyshape) #interface bottom flux
      self.FvecintyT = jnp.zeros(self.Uvecintyshape) #interface top flux

      self.alphaintx = jnp.zeros(self.stagintx) #max wave speed at interfaces
      self.alphainty = jnp.zeros(self.staginty) #max wave speed at interfaces

      self.pvar = PVAR_MHD if self.mhd_enabled else PVAR_HYDRO; #primitive variables, optionally including Bx, By, Bz
      self.pvarnames = list(P_NAMES[:self.pvar])
      if self.reli == 1:
         self.pvar = PVAR_RMHD; #number of primitive variables (rho, vx, vy, vz, P, Bx, By, Bz, gamma(W lorentz))
      self.Pvecshape = (self.pvar, *self.ntot2d)
      self.Pvecintxshape = (self.pvar, *self.stagintx)
      self.Pvecintyshape = (self.pvar, *self.staginty)
      self.dPvecxshape = (self.pvar, self.nx+2, self.ny+ 2*self.ngi)
      self.dPvecyshape = (self.pvar, self.nx+ 2*self.ngi, self.ny + 2)

      self.Pvec = jnp.zeros(self.Pvecshape) #primitive variable state vector
      self.PvecintxL = jnp.zeros(self.Pvecintxshape) #interface primitive variable state vector
      self.PvecintxR = jnp.zeros(self.Pvecintxshape) #interface primitive variable state vector
      self.PvecintyB = jnp.zeros(self.Pvecintyshape) #interface primitive variable state vector
      self.PvecintyT = jnp.zeros(self.Pvecintyshape) #interface primitive variable
      self.Pvectest = jnp.zeros(self.Pvecshape) #test primitive variable state vector

      self.Pvech = jnp.zeros(self.Pvecshape) #half step primitive variable state vector
      self.Pvecadvh = jnp.zeros(self.Pvecshape) #primitive variable state vector for advection step
      self.Pvecdiffh = jnp.zeros(self.Pvecshape) #primitive variable state vector for diffusion step

      self.dPvecL = jnp.zeros(self.dPvecxshape) #left slope of primitive variables
      self.dPvecR = jnp.zeros(self.dPvecxshape) #right slope of primitive variables
      self.dPveclimx = jnp.zeros(self.dPvecxshape) #slope limited slope of primitive variables

      self.dPvecB = jnp.zeros(self.dPvecyshape) #down slope of primitive variables
      self.dPvecT = jnp.zeros(self.dPvecyshape) #up slope
      self.dPveclimy = jnp.zeros(self.dPvecyshape) #slope limited slope of primitive variables


      self.speeds = 4; #number of auxiliary variables, (cs2,cs2Ax,cs2A,cs2f)
      self.cs2vecshape = (self.speeds, *self.ntot2d)
      self.cs2vecintxshape = (self.speeds, *self.stagintx)
      self.cs2vecintyshape = (self.speeds, *self.staginty)

      self.cs2vec = jnp.zeros(self.cs2vecshape) #auxiliary variable state vector
      self.cs2vecintxL = jnp.zeros(self.cs2vecintxshape) #interface auxiliary variable state vector
      self.cs2vecintxR = jnp.zeros(self.cs2vecintxshape) #interface auxiliary variable state vector
      self.cs2vecintx = jnp.zeros(self.cs2vecintxshape)

      self.cs2vecintyB = jnp.zeros(self.cs2vecintyshape) #interface auxiliary variable state vector
      self.cs2vecintyT = jnp.zeros(self.cs2vecintyshape) #interface
      self.cs2vecinty = jnp.zeros(self.cs2vecintyshape)


      self.tensorshape = (3,3,*self.ntot2d); self.vectorshape = (3,*self.ntot2d)
      self.stress = jnp.zeros(self.tensorshape) #stress tensor t_xx, t_yy, t_zz, t_xy, t_xz, t_yz
      self.dx_stress = jnp.zeros(self.tensorshape) #x derivative of stress tensor
      self.dy_stress = jnp.zeros(self.tensorshape) #y derivative of stress tensor
      self.stressdotv = jnp.zeros(self.vectorshape) #stress tensor dotted into velocity vector
      self.Fvecdiffx = jnp.zeros(self.Uvecshape) #diffusion flux vector
      self.Fvecdiffy = jnp.zeros(self.Uvecshape) #diffusion flux vector
      self.rhoPv0 = jnp.zeros(6) #rho_L0, P_L0, vx_L0, rho_R0, P_R0, vx_R0
      self.rhoL,self.presL,self.vxL,self.rhoR,self.presR,self.vxR = 0.0,0.0,0.0,0.0,0.0,0.0

      self.nstores = int(np.floor(self.nsteps / self.store_every) + 1)

      self.Uvecs = np.zeros((self.nstores,*self.Uvecshape), dtype=self.numpy_dtype) #conserved variable state vector storage
      self.Pvecs = np.zeros((self.nstores, *self.Pvecshape), dtype=self.numpy_dtype) #primitive variable state vector storage

      self.differrs = np.zeros(self.nstores -1, dtype=self.numpy_dtype); self.adverrs = np.zeros(self.nstores-1, dtype=self.numpy_dtype); self.totierrs = np.zeros(self.nstores-1, dtype=self.numpy_dtype);
      self.diffsteps = np.zeros(self.nstores-1, dtype=self.numpy_dtype); self.advsteps = np.zeros(self.nstores-1, dtype=self.numpy_dtype); self.advsteps2 = np.zeros(self.nstores-1, dtype=self.numpy_dtype);

      self.ts = np.zeros(self.nstores, dtype=self.numpy_dtype) #time storage
      self.dts = np.zeros(self.nstores - 1, dtype=self.numpy_dtype) #dt storage

      #HLL 
      # self.SL = np.zeros(self.stagintx) #left wave speed at interfaces
      # self.SR = np.zeros(self.stagintx) #right wave speed at interfaces
      # self.SB = np.zeros(self.staginty) #bottom wave speed at interfaces
      # self.ST = np.zeros(self.staginty) #top wave speed at interfaces
      # self.FvecintHLLx = np.zeros(self.Uvecintxshape) #interface HLL flux vector
      # self.FvecintHLLy = np.zeros(self.Uvecintyshape) #interface HLL flux vector
      # self.maskLHLL = np.zeros(self.stagintx,dtype=bool) #mask for left HLL flux
      # self.maskRHLL = np.zeros(self.stagintx,dtype=bool) #mask for right HLL flux
      # self.maskHLLx = np.zeros(self.stagintx,dtype=bool) #mask for HLL flux
      # self.maskBHLL = np.zeros(self.staginty,dtype=bool) #mask for bottom HLL flux
      # self.maskTHLL = np.zeros(self.staginty,dtype=bool) #mask for top HLL flux
      # self.maskHLLy = np.zeros(self.staginty,dtype=bool) #mask for HLL flux

      #HLLE
      # self.uroex = np.zeros(self.stagintx)
      # self.aroex = np.zeros(self.stagintx)
      # self.QL    = np.zeros(self.stagintx)
      # self.QR    = np.zeros(self.stagintx)

      # self.uroey = np.zeros(self.staginty)
      # self.aroey = np.zeros(self.staginty)
      # self.QB    = np.zeros(self.staginty)
      # self.QT    = np.zeros(self.staginty)

      
      self.diagvar = len(DIAGNOSTIC_NAMES); #Diagnostic scalars total(mass, energy, internal energy, rhovx, rhovy, rhovz, rhov, gradxP, gradyP, lapP, magE, KE, divB)
      self.diagnostic_names = DIAGNOSTIC_NAMES
      self.diagshape = (self.nstores,self.diagvar)
      self.diagnostics = np.zeros(self.diagshape, dtype=self.numpy_dtype) #diagnostic scalars storage

      self.Uvecexacts = np.zeros((self.nstores,self.nvar,self.nxhr,self.nyhr), dtype=self.numpy_dtype)
      self.Pvecexacts = np.zeros((self.nstores,self.pvar,self.nxhr,self.nyhr), dtype=self.numpy_dtype)
      self.diagnostics_exact = np.zeros(self.diagshape, dtype=self.numpy_dtype) #diagnostic scalars storage for exact solution

      self.lhgtupF = (0, self.ng)
      self.lhitupF = (self.ng, 2*self.ng)
      self.rhgtupF = (-self.ng, None)
      self.rhitupF = (-2*self.ng, -self.ng)

   def run_and_store(self):
      if self.nsteps == 0:
         self.evolution_run_time = 0.0
         if self.verbose:
            print("Skipping main run because Nsteps = 0")
         return
      first_chunk = min(self.store_every, self.nsteps)
      final_chunk = self.nsteps % self.store_every
      if self.verbose:
         print("Compiling...")
      compile_result = self.evolve_chunk(
         self.Uvec,
         self.t,
         first_chunk,
         self.gridparams,
         self.BCmodel,
         self.physicsmodel,
         self.reconmodel,
         self.calc_riemann_fluxes,
         self.step_in_time,
         self.calc_differr,
      )
      jax.block_until_ready(compile_result)
      if final_chunk not in (0, first_chunk):
         compile_result = self.evolve_chunk(
            self.Uvec,
            self.t,
            final_chunk,
            self.gridparams,
            self.BCmodel,
            self.physicsmodel,
            self.reconmodel,
            self.calc_riemann_fluxes,
            self.step_in_time,
            self.calc_differr,
         )
         jax.block_until_ready(compile_result)
      self.start_time1 = time.time()
      if self.verbose:
         print("Compiled, Starting main run")

      nchunks = int(np.ceil(self.nsteps / self.store_every))
      for _ in tqdm(range(nchunks), disable=not self.verbose):
         chunk_steps = min(self.store_every, self.nsteps - self.step)
         self.Pvec, self.Uvec, self.dt, self.t, self.differr, self.diffstep = self.evolve_chunk(
            self.Uvec,
            self.t,
            chunk_steps,
            self.gridparams,
            self.BCmodel,
            self.physicsmodel,
            self.reconmodel,
            self.calc_riemann_fluxes,
            self.step_in_time,
            self.calc_differr,
         )
         self.step += chunk_steps
         if ((self.step-self.step0) % self.store_every == 0):
            self.storestep +=1
            self.save_state(self.storestep)
      if self.verbose:
         print("Main Run Done")
      jax.block_until_ready((self.Pvec, self.Uvec, self.dt))
      self.evolution_run_time = time.time() - self.start_time1

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
   def evolve_chunk(Uvec, t, nsteps, gridparamsf, BCmodelf, physicsmodelf, reconmodelf, calc_riemann_f, step_in_timef, calc_differrf):
      def body(carry, _):
         Uvec, t, _, _, _, _ = carry
         Pvec, Uvec, dt = step_in_timef(Uvec, gridparamsf, BCmodelf, physicsmodelf, reconmodelf, calc_riemann_f)
         if physicsmodelf.booldiff and (not physicsmodelf.impadv):
            differr, diffstep, Pvec, Uvec = Fluid2DMHD.diff_update(Uvec, dt, gridparamsf, BCmodelf, physicsmodelf, calc_differrf)
         else:
            differr = jnp.array(0.0)
            diffstep = jnp.array(0)
         return (Uvec, t + dt, Pvec, dt, differr, diffstep), None

      Pvec = physicsmodelf.con2prim(Uvec, physicsmodelf)
      carry = (Uvec, t, Pvec, jnp.array(0.0), jnp.array(0.0), jnp.array(0))
      Uvec, t, Pvec, dt, differr, diffstep = jax.lax.scan(body, carry, xs=None, length=nsteps)[0]
      return Pvec, Uvec, dt, t, differr, diffstep

   def implicit_euler_driver(self):
      self.advective_nLU(self.Uvec, self.dUvec)
      dtr = self.dt;dtl = 0.0
      self.implicit_euler_step(dtr)
      self.advstep2 =1
      boolphys = self.IsPhysical(self.Uvech,self.Pvech)
      #Timestep too small, already converged
      if ((self.adverr <= self.advtol) and (self.differr <= self.difftol)):
         print
         while((self.adverr <= self.advtol) and (self.differr <= self.difftol) and boolphys):
            dtr*=2
            self.implicit_euler_step(dtr)
            self.advstep2 +=1
            boolphys = self.IsPhysical(self.Uvech,self.Pvech)
         if not boolphys:
            dtr *=0.5;
         dtl = dtr*0.5
      elif ((self.adverr >= self.advtol) or (self.differr >= self.difftol) or not boolphys):
         #print("Timestep too big, inner NR can't converge")
      #Timestep too big, inner NR can't converge
         while((self.adverr >= self.advtol) or (self.differr >= self.difftol) or not boolphys):
            dtr *=0.5
            self.implicit_euler_step(dtr)
            self.advstep2 +=1
            boolphys = self.IsPhysical(self.Uvech,self.Pvech)
         dtl = dtr
         dtr = dtr*2
         
      #Now Bisection
      #print(f'Bisection between {dtl} and {dtr}')
      dti = 0.5*(dtr+dtl)
      self.implicit_euler_step(dti)
      while (((self.adverr > self.advtol) or (self.differr > self.difftol)) and (self.advstep2 < self.advstepmax2)):
         if self.adverr > self.advtol:
            dtr = dti
         elif self.adverr <= self.advtol:
            dtl = dti
         dti = 0.5*(dtr+dtl)
         self.implicit_euler_step(dti)
         self.advstep2 +=1
         #print(f'Bisection step {self.advstep2}, dt = {dti}, adverr = {self.adverr}, differr = {self.differr}')
         
      #print(dti)
      #print(self.Uvech[0,:])
      np.copyto(self.Uvec,self.Uvech)
      self.apply_bcsU(self.Uvec,self.Pvec)
      self.dt = dti #reset dt to converged value

   def implicit_euler_step(self,dti):
      np.copyto(self.Uvect, self.Uvec); np.copyto(self.Pvect, self.Pvec);
      self.combined_LU(self.Uvect, self.dUvec)
      self.implicit_euler_NR_it(dti)
      self.advstep = 1
      self.boolphysit = self.IsPhysical(self.Uvect,self.Pvect)
      while (((self.adverr > self.advtol) or (self.differr > self.difftol)) and (self.advstep < self.advstepmax) and self.boolphysit):
         self.implicit_euler_NR_it(dti)
         self.advstep += 1
         if (((self.adverr >= 50) or (self.differr >= 50)) and self.advstep > 50):
            break
         
      np.copyto(self.Uvech, self.Uvect)
      np.copyto(self.Pvech, self.Pvect)

   def implicit_euler_NR_it(self,dti):
      self.Uvect[:,self.imask] =self.Uvec[:,self.imask] + self.dUvec[:,self.imask]*dti;
      self.boolphysit = self.IsPhysical(self.Uvect,self.Pvect)
      if self.boolphysit:
         #self.con2prim(self.Uvect,self.Pvect);
         self.apply_bcsU(self.Uvect,self.Pvect)
         self.dUvecadvold[...] = self.dUvecadv; self.dUvecimold[...] = self.dUvecim;
         self.combined_LU(self.Uvect, self.dUvec)
         self.adverr = self.calc_adverr(self.Uvec,self.Uvec + self.dUvecadvold*dti,self.dUvecadv, dti)
         self.differr = self.calc_differr(self.Uvec,self.Uvec + self.dUvecimold*dti,self.dUvecim, dti)
         self.totierr = self.adverr + self.differr
         
      
      #print((self.Uvect[:,self.imask] - (self.Uvect[:,self.imask] + self.dUvec[:,self.imask]*dti)))
      #print(self.adverr)

   @staticmethod
   @ft.partial(jax.jit, static_argnums =(4,5))
   def calc_operr(uvec1,uvec2,duvec,dtf, gridparamsf,physicsmodelf):
      imask = slice(gridparamsf.ng,-gridparamsf.ng)
      return jnp.sum(jnp.abs((uvec2[0:physicsmodelf.voi,imask] - (uvec1[0:physicsmodelf.voi, imask] + duvec[0:physicsmodelf.voi, imask]*dtf))))\
                           /jnp.sum(jnp.abs(uvec2[0:physicsmodelf.voi, imask] ))

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,2,3,4,5)) 
   def euler_step(Uvec, gridparamsf, BCmodelf, physicsmodelf, reconmodelf, calc_riemann_f):
      dUvec, dt = Fluid2DMHD.advective_nLU(Uvec, gridparamsf, physicsmodelf, reconmodelf, calc_riemann_f)
      Uvecn = Uvec + dt*dUvec
      pvecn, Uvecn = Fluid2DMHD.apply_bcsU(Uvecn, BCmodelf, physicsmodelf) 
      return(pvecn, Uvecn, dt)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,2,3,4,5)) 
   def corrector_step(Uvec, gridparamsf, BCmodelf, physicsmodelf, reconmodelf, calc_riemann_f):
      dUvec, dtc = Fluid2DMHD.advective_nLU(Uvec,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uvech = Uvec + dtc*dUvec
      Pvech, Uvech = Fluid2DMHD.apply_bcsU(Uvech, BCmodelf, physicsmodelf)
      dUvech, dtgarbage = Fluid2DMHD.advective_nLU(Uvech,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uvecn = 0.5*(Uvec + Uvech + dtc*dUvech)
      Pvecn, Uvecn = Fluid2DMHD.apply_bcsU(Uvecn, BCmodelf, physicsmodelf)
      return(Pvecn, Uvecn, dtc)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,2,3,4,5)) 
   def midpoint_step(Uvec, gridparamsf, BCmodelf, physicsmodelf, reconmodelf, calc_riemann_f):
      dUvec, dtm = Fluid2DMHD.advective_nLU(Uvec,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uvecm = Uvec + (dtm/2)*dUvec
      Pvecm, Uvecm = Fluid2DMHD.apply_bcsU(Uvecm, BCmodelf, physicsmodelf)
      dUvecm, dtgarbage = Fluid2DMHD.advective_nLU(Uvecm,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uvecn = Uvec + dtm*dUvecm
      Pvecn, Uvecn = Fluid2DMHD.apply_bcsU(Uvecn, BCmodelf, physicsmodelf)
      return(Pvecn, Uvecn, dtm)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,2,3,4,5)) 
   def rk4_step(Uvec, gridparamsf, BCmodelf, physicsmodelf, reconmodelf, calc_riemann_f):
      dUveck1, dtrk4 = Fluid2DMHD.advective_nLU(Uvec,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uveck1 = Uvec + 0.5*dtrk4*dUveck1
      Pveck1, Uveck1 = Fluid2DMHD.apply_bcsU(Uveck1, BCmodelf, physicsmodelf)
      dUveck2, dtgarbage = Fluid2DMHD.advective_nLU(Uveck1,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uveck2 = Uvec + 0.5*dtrk4*dUveck2
      Pveck2, Uveck2 = Fluid2DMHD.apply_bcsU(Uveck2, BCmodelf, physicsmodelf)
      dUveck3, dtgarbage = Fluid2DMHD.advective_nLU(Uveck2,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uveck3 = Uvec + dtrk4*dUveck3
      Pveck3, Uveck3 = Fluid2DMHD.apply_bcsU(Uveck3, BCmodelf, physicsmodelf)
      dUveck4, dtgarbage = Fluid2DMHD.advective_nLU(Uveck3,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uvecn = Uvec + (dtrk4/6)*(dUveck1 + 2*dUveck2 + 2*dUveck3 + dUveck4)
      Pvecn, Uvecn = Fluid2DMHD.apply_bcsU(Uvecn, BCmodelf, physicsmodelf)
      return(Pvecn, Uvecn, dtrk4)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,2,3,4,5)) 
   def ssprk3_step(Uvec, gridparamsf, BCmodelf, physicsmodelf, reconmodelf, calc_riemann_f):
      dUveck1, dtrk3 = Fluid2DMHD.advective_nLU(Uvec,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uveck1 = Uvec + dtrk3*dUveck1
      Pveck1, Uveck1 = Fluid2DMHD.apply_bcsU(Uveck1, BCmodelf, physicsmodelf)
      dUveck2, dtgarbage = Fluid2DMHD.advective_nLU(Uveck1,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uveck2 = (3/4)*Uvec + (1/4)*(Uveck1 + dtrk3*dUveck2)
      Pveck2, Uveck2 = Fluid2DMHD.apply_bcsU(Uveck2, BCmodelf, physicsmodelf)
      dUveck3, dtgarbage = Fluid2DMHD.advective_nLU(Uveck2,gridparamsf,physicsmodelf, reconmodelf, calc_riemann_f)
      Uvecn = (1/3)*Uvec + (2/3)*(Uveck2 + dtrk3*dUveck3)
      Pvecn, Uvecn = Fluid2DMHD.apply_bcsU(Uvecn, BCmodelf, physicsmodelf)
      return(Pvecn, Uvecn, dtrk3)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(0,))
   def calc_dt(gridparamsf,alphafx, alphafy):
      return(gridparamsf.cflr / (jnp.max(alphafx/gridparamsf.dx) + jnp.max(alphafy/gridparamsf.dy)))

   def combined_LU(self,uvecf,duvecout):
      for operation in self.combined_LU_operations:
         operation(uvecf)
      duvecout[...] = self.dUvecadv + self.dUvecim

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,2))
   def diffusion_nLU(uvecf, gridparamsf,physicsmodelf):
      ng =gridparamsf.ng
      Pvecdiffh = physicsmodelf.con2prim(uvecf,physicsmodelf)
      stress, dx_stress, dy_stress, stressdotv = physicsmodelf.calc_stress_grad_stdotv(Pvecdiffh,gridparamsf,physicsmodelf)
      Fvecdiffx = physicsmodelf.calc_anfluxesX_diff(uvecf,Pvecdiffh, dx_stress, stressdotv,gridparamsf,physicsmodelf) #Actually full operator
      Fvecdiffy = physicsmodelf.calc_anfluxesY_diff(uvecf,Pvecdiffh, dy_stress, stressdotv,gridparamsf,physicsmodelf) #Actually full operator
      duvecout = jnp.zeros_like(uvecf)
      duvecout = duvecout.at[:,ng:-ng,ng:-ng].set(-(Fvecdiffx[:,ng:-ng,ng:-ng]) - Fvecdiffy[:,ng:-ng,ng:-ng])
      return(duvecout)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(1,2,3,4))
   def advective_nLU(uvecf, gridparamsf,physicsmodelf, reconmodelf,calc_riemann_f):
      Pvecadvh = physicsmodelf.con2prim(uvecf, physicsmodelf)
      #Interface Reconstruction on prim or cons
      PvecintxL,PvecintxR,PvecintyB,PvecintyT,\
         UvecintxL,UvecintxR,UvecintyB,UvecintyT,\
         cs2vecintxL,cs2vecintxR,cs2vecintyB,cs2vecintyT,\
         FvecintxL,FvecintxR, FvecintyB,FvecintyT = Fluid2DMHD.LRBT_recon_all(uvecf,Pvecadvh, gridparamsf, physicsmodelf, reconmodelf)
      #Caclulate Riemann Fluxes and dt from CFL
      Fvecintx, alphaintx= calc_riemann_f(physicsmodelf,1,UvecintxL,UvecintxR,PvecintxL,PvecintxR, cs2vecintxL,cs2vecintxR,\
                              FvecintxL,FvecintxR)
      Fvecinty, alphainty= calc_riemann_f(physicsmodelf,2,UvecintyB,UvecintyT,PvecintyB,PvecintyT, cs2vecintyB,cs2vecintyT,\
                              FvecintyB,FvecintyT)
      if physicsmodelf.fluxct:
         Fvecintx, Fvecinty = ReconBase.FluxCT_Correction(Fvecintx, Fvecinty, gridparamsf)
      #calculate dt
      dt = Fluid2DMHD.calc_dt(gridparamsf,alphaintx[:,gridparamsf.ngi:gridparamsf.nyifs-gridparamsf.ngi],\
                               alphainty[gridparamsf.ngi:gridparamsf.nxifs-gridparamsf.ngi,:])
      duvecout = jnp.zeros_like(uvecf)
      duvecout = duvecout.at[:, gridparamsf.ng:-gridparamsf.ng, gridparamsf.ng:-gridparamsf.ng].set(\
         (-gridparamsf.inv_dx)*(Fvecintx[:,1:,gridparamsf.ngi:-gridparamsf.ngi] - Fvecintx[:,0:-1,gridparamsf.ngi:-gridparamsf.ngi]) \
         +(-gridparamsf.inv_dy)*(Fvecinty[:, gridparamsf.ngi:-gridparamsf.ngi,1:] - Fvecinty[:,gridparamsf.ngi:-gridparamsf.ngi,0:-1]))
      return(duvecout, dt)

   @staticmethod
   @ft.partial(jax.jit, static_argnums = (1,2))
   def apply_bcsU(uvecf, BCmodelf, physicsmodelf):
      pvecf = physicsmodelf.con2prim(uvecf,physicsmodelf)
      pvecf, uvecf = Fluid2DMHD.sync_state(pvecf, BCmodelf, physicsmodelf)
      return(pvecf, uvecf)

   @staticmethod
   @ft.partial(jax.jit, static_argnums = (1,2))
   def sync_state(pvecf,BCmodelf, physicsmodelf):
      pvecf = BoundaryConditionBase.apply_bcs(pvecf, BCmodelf)
      uvecf = physicsmodelf.prim2con(pvecf, physicsmodelf)
      return(pvecf, uvecf)
   
   def apply_ics(self):
      self.icname = initial_condition_name(self.ici)
      self.delrho = self.delrhofac*self.rhob;
      self.delP = self.delrho*self.cs20;
      self.Pvec = self.Pvec.at[P.RHO,self.imask].add(self.rhob) #background density
      self.Pvec = self.Pvec.at[P.PRESS,self.imask].add(self.presb) #background pressure
      self.U0width = 0.25
      if self.ici in [0,1,2,3]:
         #Sod shock tube
         if self.ici == 0: #sod shock (L)
            imasksided = self.lhsmask
         elif self.ici == 1: #sod shock (R)
            imasksided = self.rhsmask
         elif self.ici == 2: #sod shock (B)
            imasksided = self.bhsmask
         elif self.ici == 3: #sod shock (T)
            imasksided = self.thsmask
         self.Pvec = self.Pvec.at[P.RHO,imasksided].add(self.delrho) #density perturbation
         self.Pvec = self.Pvec.at[P.PRESS,imasksided].add(self.delP) #pressure perturbation
         if self.mhd_enabled:
            self.Pvec = self.Pvec.at[P.BY,self.imask].add(-2*self.By0) #background By magnetic field
            self.Pvec = self.Pvec.at[P.BY,imasksided].add(self.By0) #By magnetic field
            self.Pvec = self.Pvec.at[P.BZ,imasksided].add(self.Bz0) #Bz magnetic field
      if self.ici == 0:
            self.rhoPv0 = [self.Pvec[P.RHO,self.ng,self.ng],self.Pvec[P.PRESS,self.ng,self.ng],self.Pvec[P.VX,self.ng,self.ng]/self.Pvec[P.RHO,self.ng,self.ng], \
                            self.Pvec[P.RHO,-self.ng-1,self.ng],self.Pvec[P.PRESS,-self.ng-1,self.ng],self.Pvec[P.VX,-self.ng-1,self.ng]/self.Pvec[P.RHO,-self.ng-1,self.ng]]
            self.rhoL,self.presL,self.vxL,self.rhoR,self.presR,self.vxR = self.rhoPv0

      if self.ici == 4:
         #square pulse
         sqmask = (jnp.abs(self.xg) < self.U0width) & self.imask
         self.Pvec = self.Pvec.at[P.RHO,sqmask].add(self.delrho) #density perturbation
         self.Pvec = self.Pvec.at[P.PRESS,sqmask].add(self.delP) #pressure perturbation
         self.Pvec = self.Pvec.at[P.VX,sqmask].add(self.cs0*(self.delrho/self.rhob)) #velocity perturbation
      if self.ici == 5:
         #gaussian pulse
         gfac = 1/(self.U0width*jnp.sqrt(2*jnp.pi)) * jnp.exp(-0.5*jnp.square(self.xint/self.U0width))
         self.Pvec = self.Pvec.at[P.RHO,self.imask].add(gfac*self.delrho) #density perturbation
         self.Pvec = self.Pvec.at[P.PRESS,self.imask].add(gfac*self.delP) #pressure perturbation
         self.Pvec = self.Pvec.at[P.VX,self.imask].add(gfac*(self.cs0*(self.delrho/self.rhob))) #velocity perturbation
      if self.ici == 6:
         #Linear wave
         Lmod = self.Lxcc + self.dx;
         self.kx = (2*jnp.pi/Lmod) #wave number
         self.Pvec = self.Pvec.at[P.RHO,self.imask].add(self.delrho*jnp.sin(self.kx*self.xint)) #density perturbation
         self.Pvec = self.Pvec.at[P.PRESS,self.imask].add(self.delP*jnp.sin(self.kx*self.xint)) #pressure perturbation
         self.Pvec = self.Pvec.at[P.VX,self.imask].add(0*(jnp.sqrt(self.cs20)*(self.delrho/self.rhob))*jnp.sin(self.kx*self.xint)) #velocity perturbation
      if self.ici == 7:
         self.Pvec = self.Pvec.at[P.RHO,self.imask].add(-self.rhob) #background density
         self.Pvec = self.Pvec.at[P.PRESS,self.imask].add(-self.presb) #background pressure
         y1 =-0.5; y2 = 0.5; a = 0.05; sig = 0.2; uflow = 1.0; Apert = 0.01;
         y1fac = (self.yg-y1)/a; y2fac = (self.yg-y2)/a;
         y1facsig = (self.yg-y1)/sig; y2facsig = (self.yg-y2)/sig;
         tanfac = (jnp.tanh(y1fac) - jnp.tanh(y2fac)); expfac = (jnp.exp(-y1facsig**2) + jnp.exp(-y2facsig**2));
         self.Pvec = self.Pvec.at[P.RHO,self.imask].add(self.rhob + (self.delrho/2)*tanfac[self.imask])
         self.Pvec = self.Pvec.at[P.VX,self.imask].add(uflow * (tanfac[self.imask]- 1))
         self.Pvec = self.Pvec.at[P.VY,self.imask].add(Apert*jnp.sin(2*jnp.pi*self.xg[self.imask]/self.Lxcc)*expfac[self.imask])
         self.Pvec = self.Pvec.at[P.PRESS,self.imask].add(10.0);
      if self.ici == 8:
         self.Pvec = self.Pvec.at[P.RHO,self.imask].add(-self.rhob) #background density
         self.Pvec = self.Pvec.at[P.PRESS,self.imask].add(-self.presb) #background pressure
         y1 =-0.5; y2 = 0.5; a = 0.05; sig = 0.2; uflow = 1.0; Apert = 0.01;
         y1fac = (self.yg-y1)/a; y2fac = (self.yg-y2)/a;
         y1facsig = (self.yg-y1)/sig; y2facsig = (self.yg-y2)/sig;
         tanfac = (jnp.tanh(y1fac) - jnp.tanh(y2fac)); expfac = (jnp.exp(-y1facsig**2) + jnp.exp(-y2facsig**2));
         self.Pvec = self.Pvec.at[P.RHO,self.imask].add(self.rhob + (self.delrho/2)*tanfac[self.imask])
         self.Pvec = self.Pvec.at[P.VX,self.imask].add(uflow * (tanfac[self.imask]- 1))
         self.Pvec = self.Pvec.at[P.VY,self.imask].add(Apert*jnp.sin(2*jnp.pi*self.xg[self.imask]/self.Lxcc)*expfac[self.imask])
         self.Pvec = self.Pvec.at[P.PRESS,self.imask].add(10.0)

         self.Pvec = self.Pvec.at[P.BX,self.imask].add( 1.5)#Bx magnetic field
         self.Pvec = self.Pvec.at[P.BY,self.imask].add( -1.5) #By magnetic field
         self.Pvec = self.Pvec.at[P.BZ,self.imask].add( 1.5)#Bz magnetic field
      if self.ici in BALSARA_INITIAL_CONDITIONS:
         #Balsara 
         self.Pvec = self.Pvec.at[P.RHO,self.imask].add(-self.rhob ) #background density
         self.Pvec = self.Pvec.at[P.PRESS,self.imask].add(-self.presb)  #background pressure
         if self.ici == 10: #1A
            self.Pvec = self.Pvec.at[P.RHO,self.lhsmask].add(1.0); self.Pvec = self.Pvec.at[P.RHO,self.rhsmask].add(0.125);
            self.Pvec = self.Pvec.at[P.PRESS,self.lhsmask].add(1.0); self.Pvec = self.Pvec.at[P.PRESS,self.rhsmask].add(0.1);
            self.Pvec = self.Pvec.at[P.BX,self.lhsmask].add(0.5); self.Pvec = self.Pvec.at[P.BX,self.rhsmask].add(0.5);
            self.Pvec = self.Pvec.at[P.BY,self.lhsmask].add(1.0); self.Pvec = self.Pvec.at[P.BY,self.rhsmask].add(-1.0);
         if self.ici == 11: #1B
            self.Pvec = self.Pvec.at[P.RHO,self.lhsmask].add(1.0); self.Pvec = self.Pvec.at[P.RHO,self.rhsmask].add(1.0);
            self.Pvec = self.Pvec.at[P.PRESS,self.lhsmask].add(30.0); self.Pvec = self.Pvec.at[P.PRESS,self.rhsmask].add(1.0);
            self.Pvec = self.Pvec.at[P.BX,self.lhsmask].add(5.0); self.Pvec = self.Pvec.at[P.BX,self.rhsmask].add(5.0);
            self.Pvec = self.Pvec.at[P.BY,self.lhsmask].add(6.0); self.Pvec = self.Pvec.at[P.BY,self.rhsmask].add(0.7);
            self.Pvec = self.Pvec.at[P.BZ,self.lhsmask].add(6.0); self.Pvec = self.Pvec.at[P.BZ,self.rhsmask].add(0.7);
         if self.ici == 12: #1C
            self.Pvec = self.Pvec.at[P.RHO,self.lhsmask].add(1.0); self.Pvec = self.Pvec.at[P.RHO,self.rhsmask].add(1.0);
            self.Pvec = self.Pvec.at[P.PRESS,self.lhsmask].add(1000.0); self.Pvec = self.Pvec.at[P.PRESS,self.rhsmask].add(0.1);
            self.Pvec = self.Pvec.at[P.BX,self.lhsmask].add(10.0); self.Pvec = self.Pvec.at[P.BX,self.rhsmask].add(10.0);
            self.Pvec = self.Pvec.at[P.BY,self.lhsmask].add(7.0); self.Pvec = self.Pvec.at[P.BY,self.rhsmask].add(0.7);
            self.Pvec = self.Pvec.at[P.BZ,self.lhsmask].add(7.0); self.Pvec = self.Pvec.at[P.BZ,self.rhsmask].add(0.7);
         if self.ici == 13: #1D
            self.Pvec = self.Pvec.at[P.RHO,self.lhsmask].add(1.0); self.Pvec = self.Pvec.at[P.RHO,self.rhsmask].add(1.0);
            self.Pvec = self.Pvec.at[P.PRESS,self.lhsmask].add(0.1); self.Pvec = self.Pvec.at[P.PRESS,self.rhsmask].add(0.1);
            self.Pvec = self.Pvec.at[P.VX,self.lhsmask].add(0.999); self.Pvec = self.Pvec.at[P.VX,self.rhsmask].add(-0.999);
            self.Pvec = self.Pvec.at[P.BX,self.lhsmask].add(10.0); self.Pvec = self.Pvec.at[P.BX,self.rhsmask].add(10.0);
            self.Pvec = self.Pvec.at[P.BY,self.lhsmask].add(7.0); self.Pvec = self.Pvec.at[P.BY,self.rhsmask].add(-7.0);
            self.Pvec = self.Pvec.at[P.BZ,self.lhsmask].add(7.0); self.Pvec = self.Pvec.at[P.BZ,self.rhsmask].add(-7.0);
         if self.ici == 14: #1E
            self.Pvec = self.Pvec.at[P.RHO,self.lhsmask].add(1.08); self.Pvec = self.Pvec.at[P.RHO,self.rhsmask].add(1.0);
            self.Pvec = self.Pvec.at[P.PRESS,self.lhsmask].add(0.95); self.Pvec = self.Pvec.at[P.PRESS,self.rhsmask].add(1.0);
            self.Pvec = self.Pvec.at[P.VX,self.lhsmask].add(0.4); self.Pvec = self.Pvec.at[P.VX,self.rhsmask].add(-0.45);
            self.Pvec = self.Pvec.at[P.VY,self.lhsmask].add(0.3); self.Pvec = self.Pvec.at[P.VY,self.rhsmask].add(-0.2);
            self.Pvec = self.Pvec.at[P.VZ,self.lhsmask].add(0.2); self.Pvec = self.Pvec.at[P.VZ,self.rhsmask].add(-0.2);
            self.Pvec = self.Pvec.at[P.BX,self.lhsmask].add(2.0); self.Pvec = self.Pvec.at[P.BX,self.rhsmask].add(2.0);
            self.Pvec = self.Pvec.at[P.BY,self.lhsmask].add(0.3); self.Pvec = self.Pvec.at[P.BY,self.rhsmask].add(-0.7);
            self.Pvec = self.Pvec.at[P.BZ,self.lhsmask].add(0.3); self.Pvec = self.Pvec.at[P.BZ,self.rhsmask].add(0.5);
      self.boolmag = self.mhd_enabled and bool(jnp.max(jnp.abs(self.Pvec[MAG_P,:])) != 0.0)
      self.voi = NVAR_MHD if self.boolmag else NVAR_HYDRO
      self.physicsmodel = self.physicsbase.PhysicsModel(self.gamma,self.kappa,self.cs20, self.boolmag,self.fluxct,self.voi,self.physicsbase.diffparams, \
                                                   self.physicsbase.diffloopparams,self.booldiff, self.impadv, \
                                                   self.physicsbase.EOS_P, self.physicsbase.EOS_eps,\
                                             self.physicsbase.EOS_cs2, self.physicsbase.calc_energy, self.physicsbase.calc_eps,\
                                                self.physicsbase.con2prim, self.physicsbase.prim2con,\
                                             self.physicsbase.calc_anfluxesX, self.physicsbase.calc_anfluxesY, \
                                                self.physicsbase.calc_anfluxesX_diff, self.physicsbase.calc_anfluxesY_diff,\
                                                   self.physicsbase.calc_stress_grad_stdotv)
      self.riemannbase.validate_model(self.physicsmodel)
      if self.eosi == 2:
         self.Pvec = self.Pvec.at[P.PRESS,self.imask].set(self.physicsbase.EOS_P(self.Pvec[P.RHO,self.imask], None)) #reset pressure for polytrope, nonlinear EOS can't sum P.
      assert(np.all(self.Pvec[P.RHO,self.imask] > 0.0)) #check for positive density
      assert(np.all(self.Pvec[P.PRESS,self.imask] > 0.0)) #check for positive pressure
      assert(np.allclose(self.physicsmodel.EOS_P(self.Pvec[P.RHO,self.imask], self.physicsmodel.EOS_eps(self.Pvec[P.PRESS,self.imask], self.physicsmodel)/self.Pvec[P.RHO,self.imask], self.physicsmodel), self.Pvec[P.PRESS,self.imask], atol=self.float_tol, rtol=self.float_tol)) #check EOS consistency

      if self.reli ==1:
         self.Pvec = self.Pvec.at[P.LORENTZ,:].set(self.physicsbase.calc_lorentz_vec(self.Pvec[VEL_P,:]))

      self.vrsl = np.r_[0:self.voi]
      

      self.Pvec, self.Uvec = self.sync_state(self.Pvec,self.BCmodel, self.physicsmodel) #BC's and aux variables
      
      self.test_c2pp2c_p2cc2p()

   def test_c2pp2c_p2cc2p(self):
      self.Pvectest = self.physicsmodel.con2prim(self.Uvec, self.physicsmodel)
      self.Uvectest = self.physicsmodel.prim2con(self.Pvectest, self.physicsmodel)
      assert(jnp.allclose(self.Uvectest, self.Uvec, atol=self.float_tol, rtol=self.float_tol))
      self.Uvectest = self.physicsmodel.prim2con(self.Pvec, self.physicsmodel)
      self.Pvectest = self.physicsmodel.con2prim(self.Uvectest, self.physicsmodel)
      assert(jnp.allclose(self.Pvectest, self.Pvec, atol=self.float_tol, rtol=self.float_tol))

   def calc_exact_sod(self):
      ttg, xtg = np.meshgrid(self.ts,self.xghr1d,indexing='ij')
      ttg[ttg ==0] = 1e-14 #Start the exact solution at t = 0+ 
      self.xie = xtg/ttg 
      self.csL = np.sqrt(self.gamma*self.presL/self.rhoL);self.csR = np.sqrt(self.gamma*self.presR/self.rhoR);
      self.gammaex = (self.gamma - 1.0)/(self.gamma + 1.0); self.betaex = (self.gamma - 1)/(2*self.gamma);
      self.pres3 = (self.presL + self.presR)/2 #initial guess
      self.pres3 = sciopt.brentq(calc_root_pres3,self.presR,self.presL,args=(self,),xtol=1e-12,full_output=False)

      self.rho3 = self.rhoL * (self.pres3/self.presL)**(1/self.gamma)

      self.vShock = self.vxR + self.csR*np.sqrt(((self.gamma - 1) + self.pres3*(self.gamma + 1))/(2*self.gamma))
      self.vContact = self.vxL + (2*self.csL/(self.gamma - 1))*(1 - ((self.pres3*self.presR)/self.presL)**((self.gamma - 1)/(2*self.gamma)))
      self.vFanright = self.vContact - np.sqrt(self.gamma*self.pres3/self.rho3)

      self.vx3 = self.vContact#self.vxR + (self.pres3 - self.presR) / np.sqrt((self.rhoR/2)*(self.gamma+1)*self.pres3 + (self.gamma-1)*self.presR)
      

      self.pres4 = self.pres3
      self.rho4 = self.rhoR*((self.pres4/self.presR + self.gammaex)/(1 + self.gammaex*self.pres4/self.presR))
      self.vx4 = self.vx3
      #Interface velocities are (-csL, vFanright, vContact, vShock)
      v1 = -self.csL; v2 = self.vFanright; v3 = self.vContact; v4 = self.vShock
      valid = v1 < v2 < v3 < v4
      if not valid:
         print("Exact solution velocities not valid")
         return
      r1f = np.where(self.xie < v1,1.0,0.0)
      r2f = np.where((self.xie < v2) & (self.xie >= v1),1.0,0.0)
      r3f = np.where((self.xie < v3) & (self.xie >= v2),1.0,0.0)
      r4f = np.where((self.xie < v4) & (self.xie >= v3),1.0,0.0)
      r5f = np.where(self.xie >= v4,1.0,0.0)
      region_coverage = np.sum(r1f + r2f + r3f + r4f + r5f)/((self.nxhr)*(self.nstores))
      assert(np.isclose(region_coverage, 1.0)) #check that all regions accounted for

      #Region 2 requires a bit of work
      u2x = (2/(self.gamma+1)) * (self.csL + self.xie)*r2f
      rho2x = self.rhoL*(1 - ((self.gamma - 1)/2)*(u2x/self.csL))**(2/(self.gamma-1))
      pres2x = self.presL* (rho2x/self.rhoL)**self.gamma
      
      self.Pvecexacts[:,P.RHO,:,:] = (r1f*self.rhoL + r2f*rho2x + r3f*self.rho3 + r4f*self.rho4 + r5f*self.rhoR     )[:,:,np.newaxis]
      self.Pvecexacts[:,P.VX,:,:] = (r1f*self.vxL + r2f*u2x + r3f*self.vx3 + r4f*self.vx4 + r5f*self.vxR           )[:,:,np.newaxis]
      self.Pvecexacts[:,P.PRESS,:,:] = (r1f*self.presL + r2f*pres2x + r3f*self.pres3 + r4f*self.pres4 + r5f*self.presR)[:,:,np.newaxis]
      

      for i in range(self.nstores):
         if self.reli == 1:
               self.Pvecexacts[i,P.LORENTZ,:] = self.physicsbase.calc_lorentz_vec(self.Pvecexacts[i,VEL_P,:])
         self.Uvecexacts[i,:,:] = self.physicsbase.prim2con(self.Pvecexacts[i,:,:],self.physicsmodel)
      self.compute_diagnostics(self.dxhr,self.Uvecexacts,self.Pvecexacts,self.diagnostics_exact)

   def calc_exact_balsara(self):
      if self.ici in [6,7,8,9]:
         self.tbalsara = 0.4;
      elif self.ici == 10:
         self.tbalsara = 0.55;
      self.Bxbalsara = np.array([0.5,5.0,10.0,10.0,2.0])
      Bx0 = self.Bxbalsara[self.ici-6]
      balexdata = np.loadtxt(exactdir / f"solution_b{self.ici-5}.dat")
      self.xbal = balexdata[:,0];
      self.rhoexbal = balexdata[:,1];
      self.presexbal = balexdata[:,2]; 
      self.vexbal = balexdata[:,3:6];
      self.Byzexbal = balexdata[:,6:8];
      lenex = len(self.rhoexbal)
      ptx = (np.where(np.abs(self.xbal - 1.0) < 1e-7)[0][0]) + 1
      frames = lenex//ptx;
      self.xbal = np.reshape(self.xbal, (frames,lenex//frames)); self.xbal = self.xbal[0,:]; self.dxbal = self.xbal[1]-self.xbal[0];
      self.rhoexbal = np.reshape(self.rhoexbal, (frames,lenex//frames))
      self.presexbal = np.reshape(self.presexbal, (frames,lenex//frames))
      self.vexbal = np.reshape(self.vexbal, (frames,lenex//frames,3))
      self.Byzexbal = np.reshape(self.Byzexbal, (frames,lenex//frames,2))
      self.Pvecexbal = np.stack((self.rhoexbal,self.vexbal[:,:,0],self.vexbal[:,:,1],self.vexbal[:,:,2],self.presexbal,Bx0*np.ones_like(self.rhoexbal),self.Byzexbal[:,:,0],self.Byzexbal[:,:,1]),axis=1).astype(self.numpy_dtype, copy=False)
      self.Uvecexbal = np.zeros((frames,self.nvar,ptx), dtype=self.numpy_dtype)
      self.diagnostics_exbal = np.zeros((frames,self.diagvar), dtype=self.numpy_dtype)
      for i in range(frames):
         self.physicsbase.prim2con(self.Pvecexbal[i,:,:],self.Uvecexbal[i,:,:])
      self.compute_diagnostics(self.dxbal,self.Uvecexbal,self.Pvecexbal,self.diagnostics_exbal)

   def load_checkpoint(self):
      checkfile = self.checkdirsim / 'UPvec_final.h5'
      if not checkfile.exists():
         print("Warning: No final checkpoint found, starting new simulation.")
         return

      with np.load(self.checkdirsim / 'params.npz', allow_pickle=True) as pardict0, np.load(self.checkdirsim / 'grid.npz', allow_pickle=True) as grid0:
         xg0,yg0 = grid0['xg'], grid0['yg'];
         assert(np.allclose(xg0,self.xg, atol=self.float_tol, rtol=self.float_tol))
         assert(np.allclose(yg0,self.yg, atol=self.float_tol, rtol=self.float_tol))
         matches, mismatches = compare_parameter_dicts(
            pardict0,
            self.pardict,
            tol=1e-8,
            exclude={'recover_state', 'save_final_state', 'Nsteps'},
         )
      if matches:
         print("Checkpoint parameters match current simulation parameters.")
      else:
         print("Warning: Checkpoint parameters do not match current simulation parameters.")
         for mismatch in mismatches[:8]:
            print(f"  {mismatch}")
         if len(mismatches) > 8:
            print(f"  ... {len(mismatches) - 8} additional mismatches")

      with h5py.File(checkfile, 'r') as hf:
         self.Uvec = jnp.array(hf['Uvec'])
         self.Pvec = jnp.array(hf['Pvec'])
         self.t         = hf.attrs['t']
         self.sim_number = hf.attrs['sim_number']
         self.step0      = int(hf.attrs["Nfinal"]); self.step = self.step0;
      print(f"Recovered state from checkpoint {checkfile.name}, Simulation number {self.sim_number}, at time t = {self.t}, step = {self.step}")
      self.sim_number += 1 #increment sim number for next save

   def save_final_checkpoint(self):
      self.checkdirsim.mkdir(parents=True, exist_ok=True)
      h5namecheck = self.checkdirsim /  f'UPvec_final.h5'
      with h5py.File(h5namecheck, 'w') as hf:
         hf.create_dataset('Uvec', data=self.Uvec)
         hf.create_dataset('Pvec', data=self.Pvec)
         hf.attrs['t'] = self.t
         hf.attrs["Nfinal"] = self.step
         hf.attrs["sim_number"] = self.sim_number

   def save_hdf5_outputs(self):
      self.checkdirsim.mkdir(parents=True, exist_ok=True)
      h5name = self.checkdirsim /  f'UPvec_{self.sim_number:07d}.h5'
      if self.sim_number == 0:
         pardictname = self.checkdirsim / 'params.npz'
         griddictname = self.checkdirsim / 'grid.npz'
         np.savez(griddictname, xg=self.xg, yg=self.yg);
         np.savez(pardictname, **self.pardict)
         griddicth5name = self.checkdirsim / 'grid.h5'
         with h5py.File(griddicth5name, 'w') as hf:
            hf.create_dataset('xg', data=self.xg)
            hf.create_dataset('yg', data=self.yg)
      nvarskip = np.array([]) #No conserved variables skipped
      pvarskip = np.array([P.RHO, P.BX, P.BY, P.BZ], dtype=int) #Always duplicates
      if (not self.boolmag):
         nvarskip = np.array([U.BX, U.BY, U.BZ], dtype=int);
      
      storedfiles = sorted(self.checkdirsim.glob('UPvec_0*.h5'))
      storestart = int(path.Path(storedfiles[-1]).stem.split('_')[1]) if len(storedfiles) > 0 else 0
      for k in range(self.nstores):
         h5name = self.checkdirsim /  f'UPvec_{storestart+k:07d}.h5'
         with h5py.File(h5name, 'w') as hf:
            step = k*self.store_every
            for j in range(self.nvar):
               if j in nvarskip:
                  continue
               hf.create_dataset(self.nvarnames[j], data=self.Uvecs[k, j, :])
            for j in range(self.pvar):
               if j in pvarskip:
                  continue
               hf.create_dataset(self.pvarnames[j], data=self.Pvecs[k, j, :])
            hf.attrs["realstep"] = step
            hf.attrs["save_step"] = k
            hf.attrs["time"] = self.ts[k]
      files = sorted(self.checkdirsim.glob("UPvec_0*.h5"))
      dotvispath = self.checkdirsim / "run.visit"
      dotvispath.write_text("\n".join(str(f.name) for f in files))

   @staticmethod
   @ft.partial(jax.jit, static_argnums = (2,3,4))
   def LRBT_recon_USP(uvecf,pvecf, gridparamsf, physicsmodelf, reconmodelf):
      uvecoutL,uvecoutR,uvecoutB,uvecoutT = reconmodelf.LRBTreconV(uvecf,gridparamsf,physicsmodelf,reconmodelf) #reconstruct conserved variables at interface
      pvecoutL = physicsmodelf.con2prim(uvecoutL,physicsmodelf)
      pvecoutR = physicsmodelf.con2prim(uvecoutR,physicsmodelf)
      pvecoutB = physicsmodelf.con2prim(uvecoutB,physicsmodelf)
      pvecoutT = physicsmodelf.con2prim(uvecoutT,physicsmodelf)
      return(pvecoutL, pvecoutR,pvecoutB,pvecoutT,uvecoutL, uvecoutR,uvecoutB,uvecoutT)

   @staticmethod
   @ft.partial(jax.jit, static_argnums = (2,3,4))
   def LRBT_recon_PSU(uvecf,pvecf, gridparamsf, physicsmodelf, reconmodelf):
      pvecoutL,pvecoutR,pvecoutB,pvecoutT = reconmodelf.LRBTreconV(pvecf,gridparamsf,physicsmodelf,reconmodelf) #reconstruct primitive variables at interface
      uvecoutL = physicsmodelf.prim2con(pvecoutL,physicsmodelf)
      uvecoutR = physicsmodelf.prim2con(pvecoutR,physicsmodelf)
      uvecoutB = physicsmodelf.prim2con(pvecoutB,physicsmodelf)
      uvecoutT = physicsmodelf.prim2con(pvecoutT,physicsmodelf)
      return(pvecoutL, pvecoutR,pvecoutB,pvecoutT,uvecoutL, uvecoutR,uvecoutB,uvecoutT)

   @staticmethod
   @ft.partial(jax.jit, static_argnums = (2,3,4))
   def LRBT_recon_all(uvecf,pvecf, gridparamsf, physicsmodelf, reconmodelf):
      pvecoutL,pvecoutR,pvecoutB,pvecoutT,uvecoutL,uvecoutR,uvecoutB,uvecoutT = reconmodelf.LRBTrecon_UP(uvecf,pvecf,gridparamsf,physicsmodelf,reconmodelf)
      cs2outL = PhysicsBase.calc_speeds(pvecoutL,physicsmodelf); cs2outR = PhysicsBase.calc_speeds(pvecoutR,physicsmodelf);
      cs2outB = PhysicsBase.calc_speeds(pvecoutB,physicsmodelf); cs2outT = PhysicsBase.calc_speeds(pvecoutT,physicsmodelf);
      fvecoutL = physicsmodelf.calc_anfluxesX(uvecoutL,pvecoutL,physicsmodelf); fvecoutR = physicsmodelf.calc_anfluxesX(uvecoutR,pvecoutR,physicsmodelf);
      fvecoutB = physicsmodelf.calc_anfluxesY(uvecoutB,pvecoutB,physicsmodelf); fvecoutT = physicsmodelf.calc_anfluxesY(uvecoutT,pvecoutT,physicsmodelf);
      return(pvecoutL,pvecoutR,pvecoutB,pvecoutT,uvecoutL,uvecoutR,uvecoutB,uvecoutT,\
                     cs2outL,cs2outR,cs2outB,cs2outT,fvecoutL,fvecoutR,fvecoutB,fvecoutT)


   @staticmethod
   @ft.partial(jax.jit, static_argnums=(3,4,5))
   def calc_diff_iteration(uvec1,duvec1,dtf, gridparamsf, physicsmodelf, operrf):
      uvec2 = dtf*duvec1 + uvec1
      dUvec =  Fluid2DMHD.diffusion_nLU(uvec2, gridparamsf, physicsmodelf)
      differr = operrf(uvec1,uvec2,dUvec, dtf, gridparamsf,physicsmodelf)
      return(differr, dUvec, uvec2)

   @staticmethod
   @ft.partial(jax.jit, static_argnums=(2,3,4,5))
   def diff_update(uvecf, dtf, gridparamsf,bcmodelf, physicsmodelf, operrf):
      def cond_fun(carry):
        differr, diffstep, uvecf, dUvec = carry
        return (differr > physicsmodelf.diffloopparams[0]) & (diffstep < physicsmodelf.diffloopparams[1])
      def body_fun(carry):
        differr, diffstep, uvec2, dUvec = carry
        differr, dUvec, uvec2 = Fluid2DMHD.calc_diff_iteration(uvecf, dUvec, dtf, gridparamsf, physicsmodelf, operrf)
        return (differr, diffstep + 1, uvec2, dUvec)
      differr = 1e9;
      carry = (differr, 0, uvecf, jnp.zeros_like(uvecf))
      carry = jax.lax.while_loop(cond_fun, body_fun, carry)
      differr, diffstep, uvec2, dUvec = carry
      
      pvec2, uvec2 = Fluid2DMHD.apply_bcsU(uvec2, bcmodelf, physicsmodelf)
      return(differr, diffstep, pvec2, uvec2)
   
   def compute_diagnostics(self,dxf,uvecsf,pvecsf,diagvecf, imask = (slice(0,None), slice(0,None))):
      #(mass, energy, internal energy, rhovx, rhovy, rhovz, rhov, gradxP, gradyP, lapP, magE, KE)
      self.dtavg = np.mean(self.dts) if len(self.dts) else 0.0
      diagvus = np.array([D.MASS,D.ENERGY,D.MOMX,D.MOMY,D.MOMZ]) #mass, energy, rhovx, rhovy, rhovz
      uvecus = np.array([U.RHO,U.ENERGY,U.MOMX,U.MOMY,U.MOMZ]) #density, energy, rhovx, rhovy, rhovz
      dxf2 = dxf*dxf
      for j in range(len(diagvus)):
         diagvecf[:,diagvus[j]] = np.sum(uvecsf[:,uvecus[j], *imask],axis=(1,2))*dxf2
         #diagvecf[:,diagvus[j]] -= (uvecf[:,uvecus[j],0] + uvecf[:,uvecus[j],-1])*dxf/2 #Subtract off last 2 volume, they extend outside
      diagvecf[:,D.INTERNAL_ENERGY] = np.sum((pvecsf[:,P.PRESS, *imask]/(self.gamma - 1))*dxf2,axis=(1,2)) #internal energy
      gradPx = np.gradient(pvecsf[:,P.PRESS, ...],dxf,axis=1); lapPx = np.gradient(gradPx,dxf,axis=1);
      gradPy = np.gradient(pvecsf[:,P.PRESS, ...],dxf,axis=2); lapPy = np.gradient(gradPy,dxf,axis=2);
      lapP = lapPx + lapPy
      diagvecf[:,D.MOM] = np.sum(uvecsf[:,MOM_U,*imask], axis = (1,2,3))*dxf2
      diagvecf[:,D.GRADPX_L2] = np.sum(np.square(gradPx[:,*imask])*dxf2,axis=(1,2)) #gradx P
      diagvecf[:,D.GRADPY_L2] = np.sum(np.square(gradPy[:,*imask])*dxf2,axis=(1,2)) #grady P
      diagvecf[:,D.LAPP_L2] = np.sum(np.square(lapP[:,*imask])*dxf2,axis=(1,2)) #lap P
      diagvecf[:,D.MAGNETIC_ENERGY] = 0.5*np.sum(np.square(pvecsf[:,MAG_P, *imask])*dxf2,axis=(1,2,3)) #magnetic energy
      diagvecf[:,D.KINETIC_ENERGY] = 0.5*np.sum(np.square(pvecsf[:,VEL_P, *imask])*pvecsf[:,P.RHO:P.RHO+1,*imask]*dxf2,axis=(1,2,3)) #kinetic energy
      if self.boolmag:
         #diagvecf[:,12] = (np.gradient(pvecsf[:,5, ...],dxf,axis=1).sum(axis=(1,2)) + np.gradient(pvecsf[:,6, ...],dxf,axis=2).sum(axis=(1,2)))*dxf2 #div B
         sl1 = slice(self.ng-1,-(self.ng+1))
         sl2d = (sl1,sl1)
         divb = ((pvecsf[:,P.BX,1:,1:] + pvecsf[:,P.BX, 1:, 0:-1] - pvecsf[:,P.BX,:-1,1:] - pvecsf[:,P.BX,:-1, :-1])/ (2*dxf) +\
              (pvecsf[:,P.BY,1:,1:] + pvecsf[:,P.BY,0:-1,1:] - pvecsf[:,P.BY,1:,:-1] - pvecsf[:,P.BY, :-1, :-1])/(2*dxf))
         divb_l2 = np.sqrt(np.mean(np.square(divb[:,*sl2d]), axis=(1,2)))
         b_rms = np.sqrt(np.mean(np.sum(np.square(pvecsf[:,MAG_P,*imask]), axis=1), axis=(1,2)))
         diagvecf[:,D.DIVB_L2] = np.where(b_rms > 0.0, divb_l2/(b_rms/dxf), 0.0)

   def save_state(self,itf):
      self.Uvecs[itf,:,:] = self.Uvec
      self.Pvecs[itf,:,:] = self.Pvec
      self.ts[itf] = self.t
      self.dts[itf-1] = self.dt
      if self.booldiff:
         self.differrs[itf-1] = self.differr
         self.diffsteps[itf-1] = self.diffstep
      if self.impadv:
         self.adverrs[itf-1] = self.adverr
         self.differrs[itf-1] = self.differr
         self.totierrs[itf-1] = self.totierr
         self.advsteps[itf-1] = self.advstep
         self.advsteps2[itf-1] = self.advstep2

   def plot_all1d(self,itf, sli, axsl):
      # Density, momx, y, z
      # energy, velx, vely, velz
      # pressure, (Bx,By,Bz)
      # 
      if axsl is None or axsl == "x":
         self.p1dslice = (slice(0,self.nxtot), self.ng + sli)
         self.p1dslicex = (slice(0,self.nxhr), int((self.nxhr-1)/2))
         axsl = "x"
         xp = self.xg1d; xphr = self.xghr1d
      elif axsl == "y":
         self.p1dslice = (self.ng + sli, slice(0,self.nytot))
         self.p1dslicex = (int((self.nyhr-1)/2), slice(0,self.nyhr))
         axsl = "y"
         xp = self.yg1d; xphr = self.yghr1d
      Uvecmaxs = np.max(self.Uvecs[:itf,:,*self.p1dslice],axis=(0,2))
      Uvecmins = np.min(self.Uvecs[:itf,:,*self.p1dslice],axis=(0,2))
      Pvecmaxs = np.max(self.Pvecs[:itf,:,*self.p1dslice],axis=(0,2))
      Pvecmins = np.min(self.Pvecs[:itf,:,*self.p1dslice],axis=(0,2))
      ybotscl = 0.999999; ytopscl = 1.01
      if   ((np.abs(self.ts[itf] - 0.4) < 2*self.dtavg or (np.abs(self.ts[itf] - 0.55) < 2*self.dtavg) and self.ici == 10) and self.plotbalsara):
         plotbalsarax = True
      else:
         plotbalsarax = False
      
      nplots = 12
      rows = 3
     
      fig,axs = plt.subplots(rows,4,figsize=(12,nplots))
      suptitle = f'{self.info}, Time = {self.ts[itf]:.3f}, it = {itf}'
      fig.suptitle(suptitle,fontsize=16)
      #Density
      axs[0,0].plot(xp,self.Uvecs[itf,0,*self.p1dslice],label='Num.')
      
      if self.ici == 0:
         axs[0,0].plot(xphr,self.Uvecexacts[itf,0,*self.p1dslicex],"--",label='Exact',color='black')
         axs[0,0].legend()
      if plotbalsarax:
         Uvecmaxs[0] = np.max([Uvecmaxs[0],np.max(self.Pvecexbal[-1,0,:])])
         axs[0,0].plot(self.xbal,self.Pvecexbal[-1,0,:],"--",label='Balsara Exact',color='grey')
         axs[0,0].legend()
      axs[0,0].set_title('Density' )
      axs[0,0].set_ylim([ybotscl*Uvecmins[0],ytopscl*Uvecmaxs[0]])
      axs[0,0].set_xlabel(f'{axsl}')
      axs[0,0].set_ylabel('rho')
      #Momentum x
      axs[0,1].plot(xp,self.Uvecs[itf,1,*self.p1dslice],label='Momentum x',color='orange')
      if self.ici == 0:
         axs[0,1].plot(xphr,self.Uvecexacts[itf,1,*self.p1dslicex],"--",label='Exact',color='black')
         axs[0,1].legend()
      if plotbalsarax:
         axs[0,1].plot(self.xbal,self.Uvecexbal[-1,1,*self.p1dslice],"--",label='Balsara Exact',color='grey')
         axs[0,1].legend()
      axs[0,1].set_title('Momentum x')
      axs[0,1].set_xlabel(f'{axsl}')
      axs[0,1].set_ylabel('rhovx')
      axs[0,1].set_ylim([ybotscl*Uvecmins[1],ytopscl*Uvecmaxs[1]])
      #Momentum y
      axs[0,2].plot(xp,self.Uvecs[itf,2,*self.p1dslice],label='Momentum y',color='green')
      axs[0,2].set_title('Momentum y')
      if plotbalsarax:
         axs[0,2].plot(self.xbal,self.Uvecexbal[-1,2,*self.p1dslice],"--",label='Balsara Exact',color='grey')
         axs[0,2].legend()
      axs[0,2].set_xlabel(f'{axsl}')
      axs[0,2].set_ylabel('rhovy')
      #Momentum z
      axs[0,3].plot(xp,self.Uvecs[itf,3,*self.p1dslice],label='Momentum z',color='red')
      if plotbalsarax:
         axs[0,3].plot(self.xbal,self.Uvecexbal[-1,3,*self.p1dslice],"--",label='Balsara Exact',color='grey')
         axs[0,3].legend()
      axs[0,3].set_title('Momentum z')
      axs[0,3].set_xlabel(f'{axsl}')
      axs[0,3].set_ylabel('rhovz')
      #Energy
      axs[1,0].plot(xp,self.Uvecs[itf,4,*self.p1dslice],label='Energy',color='purple')
      if self.ici == 0:
         axs[1,0].plot(xphr,self.Uvecexacts[itf,4,*self.p1dslicex],"--",label='Exact',color='black')
         axs[1,0].legend()
      if plotbalsarax:
         axs[1,0].plot(self.xbal,self.Uvecexbal[-1,4,*self.p1dslice],"--",label='Balsara Exact',color='grey')
         axs[1,0].legend()
      axs[1,0].set_title('Energy')
      axs[1,0].set_ylim([ybotscl*Uvecmins[4],ytopscl*Uvecmaxs[4]])
      axs[1,0].set_xlabel(f'{axsl}')
      axs[1,0].set_ylabel('E')   
 
        

      #Velocity x
      axs[1,1].plot(xp,self.Pvecs[itf,1,*self.p1dslice],label='Velocity x',color='magenta')
      if self.ici == 0:
         axs[1,1].plot(xphr,self.Pvecexacts[itf,1,*self.p1dslicex],"--",label='Exact',color='black')
         axs[1,1].legend()
      if plotbalsarax:
         axs[1,1].plot(self.xbal,self.Pvecexbal[-1,1,*self.p1dslice],"--",label='Balsara Exact',color='grey')
         axs[1,1].legend()
      axs[1,1].set_title('Velocity x')
      axs[1,1].set_ylim([ybotscl*Pvecmins[1],ytopscl*Pvecmaxs[1]])
      axs[1,1].set_xlabel(f'{axsl}')
      axs[1,1].set_ylabel('vx')
      #Velocity y
      axs[1,2].plot(xp,self.Pvecs[itf,2,*self.p1dslice],label='Velocity y',color='brown')
      axs[1,2].set_title('Velocity y')
      axs[1,2].set_ylim([ybotscl*Pvecmins[2],ytopscl*Pvecmaxs[2]])
      axs[1,2].set_xlabel(f'{axsl}')
      axs[1,2].set_ylabel('vy')
      #velocity z
      axs[1,3].plot(xp,self.Pvecs[itf,3,*self.p1dslice],label='Velocity z',color='cyan')
      axs[1,3].set_title('Velocity z')
      axs[1,3].set_ylim([ybotscl*Pvecmins[3],ytopscl*Pvecmaxs[3]])
      axs[1,3].set_xlabel(f'{axsl}')
      axs[1,3].set_ylabel('vz')



           #Pressure
      pgas = self.Pvecs[itf,4,*self.p1dslice];
      if self.boolmag:
         pmag = 0.5*np.square(self.Pvecs[itf,5:8,*self.p1dslice]).sum(axis=0); ptot = pgas + pmag;
         Pvecmins[4], Pvecmaxs[4] = np.min(pgas), np.max(ptot)
         axs[2,0].plot(xp,ptot,label='P_total',color='blue')
         axs[2,0].plot(xp,pmag,label='P_mag',color='red')
      axs[2,0].plot(xp,pgas,label='P_gas',color='cyan')
      if self.ici == 0:
         axs[2,0].plot(xphr,self.Pvecexacts[itf,4,*self.p1dslicex],"--",label='Exact',color='black')
      if plotbalsarax:
         axs[2,0].plot(self.xbal,self.Pvecexbal[-1,4,*self.p1dslice],"--",label='Balsara Exact',color='grey') 
      axs[2,0].set_title('Pressure')
      axs[2,0].set_ylim([ybotscl*Pvecmins[4],ytopscl*Pvecmaxs[4]])
      axs[2,0].set_xlabel(f'{axsl}')
      axs[2,0].set_ylabel('P')
      axs[2,0].legend()
      #lap P
      # axs[1,3].plot(self.xg,self.lapPs[itf,:],label='Pressure Laplacian',color='lime')
      # if self.ici == 0:
      #    axs[1,3].plot(self.xghr,self.Auxvecexacts[itf,5,:],"--",label='Exact',color='black')
      # axs[1,3].set_title('Pressure Laplacian d2P/dx2')
      # axs[1,3].set_xlabel('x')
      # axs[1,3].set_ylabel('d2P/dx2')
      #Kinetic Energy
      # axs[1,3].plot(self.xg,self.Auxvecs[itf,6,:],label='Kinetic Energy',color='grey')
      # if self.ici == 0:
      #    axs[1,3].plot(self.xghr,self.Auxvecexacts[itf,6,:],"--",label='Exact',color='black')
      #    axs[1,3].legend()
      # axs[1,3].set_title('Kinetic Energy')
      # axs[1,3].set_xlabel('x')
      # axs[1,3].set_ylabel('KE')
      if self.boolmag:
         #Bx
         axs[2,1].plot(xp,self.Pvecs[itf,5,*self.p1dslice],label='Bx',color='green')
         axs[2,1].set_title('Magnetic Field Bx')
         axs[2,1].set_xlabel(f'{axsl}')
         axs[2,1].set_ylabel('Bx')
         
         #By  
         axs[2,2].plot(xp,self.Pvecs[itf,6,*self.p1dslice],label='By',color='brown')
         if plotbalsarax:
            axs[2,2].plot(self.xbal,self.Pvecexbal[-1,6,*self.p1dslice],"--",label='Balsara Exact',color='grey')
            axs[2,2].legend()
         axs[2,2].set_title('Magnetic Field By')
         axs[2,2].set_xlabel(f'{axsl}')
         axs[2,2].set_ylabel('By')
         #Bz
         axs[2,3].plot(xp,self.Pvecs[itf,7,*self.p1dslice],label='Bz',color='pink')
         axs[2,3].set_title('Magnetic Field Bz')
         if plotbalsarax:
            axs[2,3].plot(self.xbal,self.Pvecexbal[-1,7,*self.p1dslice],"--",label='Balsara Exact',color='grey')
            axs[2,3].legend()
         axs[2,3].set_xlabel(f'{axsl}')
         axs[2,3].set_ylabel('Bz')
         #Magnetic Energy
         # axs[2,3].plot(self.xg,self.Auxvecs[itf,7,:],label='Magnetic Energy',color='black')
         # axs[2,3].set_title('Magnetic Energy')
         # axs[2,3].set_xlabel('x')
         # axs[2,3].set_ylabel('ME')
      plt.tight_layout()
      plt.show()

      if self.boolmag:
         divb = np.gradient(self.Pvecs[itf,5,...],self.dx,axis=0) + np.gradient(self.Pvecs[itf,6,...],self.dx,axis=1)
         plt.figure()
         plt.plot(self.ts, divb[*self.p1dslice], label='Total divB', color='purple')
         plt.title(f'div B vs {axsl} at time {self.ts[itf]:.3f}')
         plt.xlabel(f'{axsl}')
         plt.ylabel('div B')
         plt.show()

   def plot_all2d(self,itf):
      plots_config = [ (self.Uvecs, 'Density', 0), (self.Uvecs, 'Momentum x', 1), (self.Uvecs, 'Momentum y', 2), (self.Uvecs, 'Momentum z', 3),
                       (self.Uvecs, 'Energy', 4), (self.Pvecs, 'Velocity x', 1), (self.Pvecs, 'Velocity y', 2), (self.Pvecs, 'Velocity z', 3),
                       (self.Pvecs, 'Pressure', 4)]
      vertices = (self.x0,self.x1,self.y0,self.y1)
      if self.boolmag:
         plots_config += [(self.Pvecs, 'Magnetic Field Bx', 5), (self.Pvecs, 'Magnetic Field By', 6), (self.Pvecs, 'Magnetic Field Bz', 7)]
      nplots = len(plots_config)
      rows = int(np.ceil(nplots/4))
      fig,axs = plt.subplots(rows,4,figsize=(16,4*rows))
      suptitle = f'{self.info}, Time = {self.ts[itf]:.3f}, it = {itf}'
      fig.suptitle(suptitle,fontsize=16)
      ax = axs.flatten()
      for i,(vecs,title,vi) in enumerate(plots_config):
         pcm = ax[i].imshow(vecs[itf,vi, ...].T, extent=vertices, aspect='auto')
         fig.colorbar(pcm, ax=ax[i], label=title)
         ax[i].set_title(title)
         ax[i].set_xlabel('x')
         ax[i].set_ylabel('y')
      plt.tight_layout()
      plt.show()
      if self.boolmag:
         divb = (self.Pvecs[itf,5,1:,1:] + self.Pvecs[itf,5, 1:, 0:-1] - self.Pvecs[itf,5,:-1,1:] - self.Pvecs[itf,5,:-1, :-1])/ (2*self.dx) +\
              (self.Pvecs[itf,6,1:,1:] + self.Pvecs[itf,6,0:-1,1:] - self.Pvecs[itf,6,1:,:-1] - self.Pvecs[itf,6, :-1, :-1])/(2*self.dy)
         plt.figure()
         pcm = plt.imshow(divb.T, extent=vertices, aspect='auto')
         plt.colorbar(pcm, label='div B')
         plt.title(f'div B at time {self.ts[itf]:.3f}')
         plt.xlabel('x')
         plt.ylabel('y')
         plt.show()

   def plot_diagnostics(self):
      plot_scalar_timeseries(self, show=True)

   def hardreset_state(self):
      self.Uvec = np.zeros((self.nvar,self.nxtot));
      self.Pvec = np.zeros((self.pvar,self.nxtot));
      self.apply_ics()
      self.t = 0.0; self.dt = 0.0; self.step = 0;

   def IsPhysical(self,uvecf,pvecf):
      boolrhop = np.all(uvecf[U.RHO,:] >= 0)
      boolpressp = np.all(pvecf[P.PRESS,:] >= 0)
      boolEp = np.all(uvecf[U.ENERGY,:] >= 0)
      if boolrhop & boolpressp & boolEp:
         return True
      else:
         return False

   def print_state(self):
      print(f"Final Step = {self.step}, Final Time = {self.t:.4f}, Mean dt = {self.dtavg:.5e}")
      print(f'Avg. Cell Updates /s = {self.cell_updates_per_second():4e}')
      if hasattr(self, 'diagnostics'):
         mass0, massf = self.diagnostics[0,D.MASS], self.diagnostics[-1,D.MASS]
         energy0, energyf = self.diagnostics[0,D.ENERGY], self.diagnostics[-1,D.ENERGY]
         print(f"Mass = {massf:.8e}, dMass = {massf - mass0:.3e}")
         print(f"Energy = {energyf:.8e}, dEnergy = {energyf - energy0:.3e}")
         if self.boolmag:
            print(f"Magnetic Energy = {self.diagnostics[-1,D.MAGNETIC_ENERGY]:.8e}, divB L2 = {self.diagnostics[-1,D.DIVB_L2]:.3e}")
      if self.booldiff:
         print(f"Mean Diffusion Steps = {self.diffsteps.mean():.5e}, Mean Diffusion Error = {self.differrs.mean():.5e}")
      if self.impadv:
         print(f"Mean Advection Inner Steps = {self.advsteps.mean():.5e}, Mean Advection Inner Error = {self.adverrs.mean():.5e}")
         print(f"Mean Advection Outer Steps = {self.advsteps2.mean():.5e}")

   def cell_updates_per_second(self):
      runtime = getattr(self, "evolution_run_time", getattr(self, "run_time", 0.0))
      if runtime <= 0.0:
         return 0.0
      return self.nx*self.ny*self.nsteps/runtime

def run_simulation(config=None):
   config = config or SimulationConfig()
   if not isinstance(config, SimulationConfig):
      raise TypeError("run_simulation expects a SimulationConfig")
   configure_jax(config)
   if config.print_devices:
      describe_jax_devices()
   jax.clear_caches()
   state = Fluid2DMHD(config);
   if config.plot:
      plot_field_snapshot(state, show=True)
      state.plot_diagnostics()
   if config.print_summary:
      state.print_state()
   return state


def main():
   run_simulation(parse_config_args("Run the 2D JAX MHD solver."))


if __name__ == "__main__":
   main()

