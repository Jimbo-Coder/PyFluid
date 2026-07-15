"""Shared postprocessing helpers for ideal RHD and RMHD validation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from plotfile_reader import (
   latest_plotfile,
   parse_plot_header_1d,
   read_composite_state_1d,
)


def entropy_density_exact(
   cells: int,
   time: float,
   *,
   rho0: float = 1.0,
   amplitude: float = 0.1,
   velocity: float = 0.3,
) -> np.ndarray:
   """Return densitized rest mass for the periodic Minkowski entropy mode."""
   x = (np.arange(cells, dtype=float) + 0.5) / cells
   rho = rho0 * (1.0 + amplitude * np.sin(2.0 * math.pi * (x - velocity * time)))
   lorentz = 1.0 / math.sqrt(1.0 - velocity * velocity)
   return rho * lorentz


def entropy_density_error(
   run_dir: Path,
   target_cells: int | None = None,
   *,
   rho0: float = 1.0,
   amplitude: float = 0.1,
   velocity: float = 0.3,
) -> float:
   """Return the entropy-mode normalized mean absolute dens error."""
   plotfile = latest_plotfile(run_dir, require=True)
   names, time_value = parse_plot_header_1d(plotfile)
   state = read_composite_state_1d(plotfile, names)
   if target_cells is not None and target_cells != state.shape[1]:
      if target_cells % state.shape[1] != 0:
         raise ValueError(
            f"target mesh {target_cells} is not an integer multiple of hierarchy mesh "
            f"{state.shape[1]}"
         )
      state = np.repeat(state, target_cells // state.shape[1], axis=1)
   exact = entropy_density_exact(
      state.shape[1],
      time_value,
      rho0=rho0,
      amplitude=amplitude,
      velocity=velocity,
   )
   dens_scale = rho0 * amplitude / math.sqrt(1.0 - velocity * velocity)
   return float(np.mean(np.abs(state[names.index("dens")] - exact)) / dens_scale)
