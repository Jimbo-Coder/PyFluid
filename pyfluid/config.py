import argparse
from collections.abc import Callable
from dataclasses import dataclass, field, fields

import numpy as np

from pyfluid.initial_data import INITIAL_CONDITIONS


DEFAULT_CFL = 0.9

EOS_CHOICES = {
   0: "isothermal",
   1: "ideal gas",
   2: "polytrope",
}

BOUNDARY_CHOICES = {
   0: "periodic",
   1: "outflow",
   2: "reflective",
}

RECON_CHOICES = {
   0: "piecewise constant",
   1: "piecewise linear",
   2: "WENO-Z",
}

RECON_VAR_CHOICES = {
   0: "primitive variables",
   1: "conserved variables",
}

LIMITER_CHOICES = {
   0: "centered difference",
   1: "minmod",
   2: "superbee",
   3: "van Leer",
   4: "MC",
}

RIEMANN_CHOICES = {
   0: "LLF",
   1: "HLL",
   2: "HLLE",
   3: "HLLC hydro only",
   4: "HLLD not implemented",
}

TIMESTEP_CHOICES = {
   0: "Euler",
   1: "corrector",
   2: "midpoint",
   3: "RK4",
   4: "SSPRK3",
   5: "IMEX not implemented in JAX driver",
}

RELATIVITY_CHOICES = {
   0: "non-relativistic",
   1: "special relativistic experimental",
}

INITIAL_CONDITION_CHOICES = {
   ic: info.name + (" magnetic" if info.magnetized else " hydro")
   for ic, info in INITIAL_CONDITIONS.items()
}


@dataclass(frozen=True)
class Option:
   flag: str
   help: str
   value_type: Callable | None = None
   choices: dict = None
   action: str = None
   target_value: object = None


def option(flag, help, value_type=None, choices=None, action=None):
   if action == "store_true":
      target_value = True
   elif action == "store_false":
      target_value = False
   else:
      target_value = None
   return {
      "option": Option(
         flag=flag,
         help=help,
         value_type=value_type,
         choices=choices,
         action=action,
         target_value=target_value,
      )
   }


def switch_help(choices):
   return ", ".join(f"{key}={value}" for key, value in choices.items())


def parse_bc(text):
   if isinstance(text, np.ndarray):
      values = tuple(int(x) for x in text.tolist())
   elif isinstance(text, str):
      values = tuple(int(item) for item in text.split(","))
   else:
      values = tuple(int(x) for x in text)
   if len(values) != 4:
      raise argparse.ArgumentTypeError("bc must have four entries: left,right,bottom,top")
   return values


@dataclass(frozen=True)
class SimulationConfig:
   # Grid and domain.
   x0: float = field(default=-1.0, metadata=option("--x0", "Left x domain boundary.", float))
   x1: float = field(default=1.0, metadata=option("--x1", "Right x domain boundary.", float))
   y0: float = field(default=-1.0, metadata=option("--y0", "Bottom y domain boundary.", float))
   y1: float = field(default=1.0, metadata=option("--y1", "Top y domain boundary.", float))
   nx: int = field(default=16, metadata=option("--nx", "Interior cells in x.", int))
   ny: int = field(default=16, metadata=option("--ny", "Interior cells in y.", int))
   ng: int = field(default=4, metadata=option("--ng", "Ghost cells per side.", int))
   ngi: int = field(default=1, metadata=option("--ngi", "Interior ghost cells used by staggered/interface arrays.", int))

   # Physics, initial data, and numerics.
   ic: int = field(default=8, metadata=option("--ic", f"Initial condition: {switch_help(INITIAL_CONDITION_CHOICES)}.", int, INITIAL_CONDITION_CHOICES))
   bc: tuple = field(default=(0, 0, 0, 0), metadata=option("--bc", f"Boundary switches left,right,bottom,top. Each entry: {switch_help(BOUNDARY_CHOICES)}.", parse_bc))
   eos: int = field(default=1, metadata=option("--eos", f"Equation of state: {switch_help(EOS_CHOICES)}.", int, EOS_CHOICES))
   relativity: int = field(default=0, metadata=option("--relativity", f"Physics regime: {switch_help(RELATIVITY_CHOICES)}.", int, RELATIVITY_CHOICES))
   recon: int = field(default=2, metadata=option("--recon", f"Reconstruction: {switch_help(RECON_CHOICES)}.", int, RECON_CHOICES))
   recon_vars: int = field(default=0, metadata=option("--recon-vars", f"Reconstruct variables: {switch_help(RECON_VAR_CHOICES)}.", int, RECON_VAR_CHOICES))
   limiter: int = field(default=1, metadata=option("--limiter", f"Slope limiter for linear reconstruction: {switch_help(LIMITER_CHOICES)}.", int, LIMITER_CHOICES))
   riemann: int = field(default=0, metadata=option("--riemann", f"Riemann solver: {switch_help(RIEMANN_CHOICES)}.", int, RIEMANN_CHOICES))
   timestep: int = field(default=3, metadata=option("--timestep", f"Time integrator: {switch_help(TIMESTEP_CHOICES)}.", int, TIMESTEP_CHOICES))
   cfl: float = field(default=DEFAULT_CFL, metadata=option("--cfl", "CFL factor.", float))
   nsteps: int = field(default=64, metadata=option("--steps", "Number of evolution steps.", int))
   store_every: int = field(default=1, metadata=option("--store-every", "Store every N steps.", int))
   delrho_factor: float = field(default=6.0, metadata=option("--delrho-factor", "Density perturbation amplitude as a multiple of background density.", float))
   diffusion: bool = field(default=False, metadata=option("--diffusion", "Enable diffusion update.", action="store_true"))
   fluxct: bool = field(default=True, metadata=option("--no-fluxct", "Disable flux-constrained transport correction.", action="store_false"))

   # Magnetic background and storage control.
   bx0: float = field(default=0.0, metadata=option("--bx0", "Background Bx.", float))
   by0: float = field(default=0.0, metadata=option("--by0", "Background By.", float))
   bz0: float = field(default=0.0, metadata=option("--bz0", "Background Bz.", float))
   force_mhd: bool = field(default=False, metadata=option("--force-mhd", "Allocate MHD variables even for hydro ICs.", action="store_true"))
   x64: bool = field(default=True, metadata=option("--float32", "Use float32 JAX arithmetic instead of the default x64 mode.", action="store_false"))

   # Repo-local outputs and run display.
   checkname: str = field(default="mhd2d", metadata=option("--checkname", "Repo-local checkpoint/output subdirectory name.", str))
   save_final_state: bool = field(default=False, metadata=option("--save-final-state", "Write final checkpoint.", action="store_true"))
   recover_state: bool = field(default=False, metadata=option("--recover-state", "Recover from final checkpoint.", action="store_true"))
   save_hdf5: bool = field(default=False, metadata=option("--save-hdf5", "Write HDF5 output frames.", action="store_true"))
   verbose: bool = field(default=True, metadata=option("--quiet", "Suppress progress messages.", action="store_false"))
   print_devices: bool = field(default=True, metadata=option("--quiet-devices", "Do not print JAX device details.", action="store_false"))
   print_summary: bool = field(default=True, metadata=option("--quiet-summary", "Do not print final scalar summary.", action="store_false"))
   plot: bool = field(default=False, metadata=option("--plot", "Show field and scalar diagnostic plots after the run.", action="store_true"))

   def to_checkpoint_params(self):
      validate_config(self)
      return {
         "x0": self.x0,
         "x1": self.x1,
         "nx": self.nx,
         "y0": self.y0,
         "y1": self.y1,
         "ny": self.ny,
         "ng": self.ng,
         "ici": self.ic,
         "bci": np.array(self.bc, dtype=int),
         "cflr": self.cfl,
         "Nsteps": self.nsteps,
         "eosi": self.eos,
         "fluxi": self.riemann,
         "reconi": self.recon,
         "reconvari": self.recon_vars,
         "limi": self.limiter,
         "timestepi": self.timestep,
         "delrhofac": self.delrho_factor,
         "booldiff": self.diffusion,
         "reli": self.relativity,
         "ngi": self.ngi,
         "fluxct": self.fluxct,
         "checkname": self.checkname,
         "save_final_state": self.save_final_state,
         "recover_state": self.recover_state,
         "store_every": self.store_every,
         "save_hdf5": self.save_hdf5,
         "verbose": self.verbose,
         "Bx0": self.bx0,
         "By0": self.by0,
         "Bz0": self.bz0,
         "force_mhd": self.force_mhd,
         "x64": self.x64,
      }


def validate_switch(name, value, choices):
   if int(value) not in choices:
      raise ValueError(f"{name}={value} is not supported; use one of {switch_help(choices)}")


def validate_config(config):
   validate_switch("ic", config.ic, INITIAL_CONDITION_CHOICES)
   validate_switch("eos", config.eos, EOS_CHOICES)
   validate_switch("recon", config.recon, RECON_CHOICES)
   validate_switch("recon_vars", config.recon_vars, RECON_VAR_CHOICES)
   validate_switch("limiter", config.limiter, LIMITER_CHOICES)
   validate_switch("riemann", config.riemann, RIEMANN_CHOICES)
   validate_switch("timestep", config.timestep, TIMESTEP_CHOICES)
   validate_switch("relativity", config.relativity, RELATIVITY_CHOICES)
   for boundary in config.bc:
      validate_switch("bc", boundary, BOUNDARY_CHOICES)
   if config.ng <= config.ngi:
      raise ValueError("ng must be larger than ngi")
   if config.nx <= 0 or config.ny <= 0:
      raise ValueError("nx and ny must be positive")
   if config.x1 <= config.x0 or config.y1 <= config.y0:
      raise ValueError("domain upper bounds must be larger than lower bounds")
   if config.cfl <= 0.0:
      raise ValueError("cfl must be positive")
   if config.store_every <= 0:
      raise ValueError("store_every must be positive")
   if config.nsteps < 0:
      raise ValueError("nsteps must be non-negative")


def _option_for(config_field):
   return config_field.metadata.get("option")


def _help_with_default(option_spec, default):
   return f"{option_spec.help} Default: {default!r}."


def _flag_help(option_spec, field_name, default):
   if option_spec.target_value is None:
      return _help_with_default(option_spec, default)
   return f"{option_spec.help} Sets {field_name}={option_spec.target_value!r}. Default: {field_name}={default!r}."


def add_config_arguments(parser):
   defaults = SimulationConfig()
   for config_field in fields(SimulationConfig):
      option_spec = _option_for(config_field)
      if option_spec is None:
         continue

      default = getattr(defaults, config_field.name)
      help_text = _flag_help(option_spec, config_field.name, default)
      if option_spec.action is not None:
         parser.add_argument(option_spec.flag, dest=config_field.name, action=option_spec.action, default=default, help=help_text)
         continue

      kwargs = {
         "dest": config_field.name,
         "default": default,
         "help": help_text,
      }
      if option_spec.value_type is not None:
         kwargs["type"] = option_spec.value_type
      if option_spec.choices is not None:
         kwargs["choices"] = tuple(option_spec.choices)
      parser.add_argument(option_spec.flag, **kwargs)


def config_from_namespace(args):
   values = {}
   for config_field in fields(SimulationConfig):
      if _option_for(config_field) is not None:
         values[config_field.name] = getattr(args, config_field.name)
   config = SimulationConfig(**values)
   validate_config(config)
   return config


def parse_config_args(description):
   parser = argparse.ArgumentParser(description=description)
   add_config_arguments(parser)
   return config_from_namespace(parser.parse_args())
