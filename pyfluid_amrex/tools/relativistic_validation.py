"""Shared postprocessing helpers for ideal RHD and RMHD validation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from plotfile_reader import (
   latest_plotfile,
   parse_plot_header,
   read_composite_state,
)


def entropy_density_exact(
   cells: int | tuple[int, ...],
   time: float,
   *,
   rho0: float = 1.0,
   amplitude: float = 0.1,
   velocity: float = 0.3,
) -> np.ndarray:
   """Return the unit-box entropy mode, advected along the first direction.

   The multidimensional sin_wave initializer uses phase x+y(+z); the velocity
   remains along the first direction. An integer retains the 1D interface.
   """
   shape = (cells,) if isinstance(cells, int) else cells
   phase = np.zeros(shape, dtype=float)
   for axis, count in enumerate(shape):
      coordinate = (np.arange(count, dtype=float) + 0.5) / count
      broadcast_shape = [1] * len(shape)
      broadcast_shape[axis] = count
      phase += coordinate.reshape(broadcast_shape)
   rho = rho0 * (1.0 + amplitude * np.sin(2.0 * math.pi * (phase - velocity * time)))
   lorentz = 1.0 / math.sqrt(1.0 - velocity * velocity)
   return rho * lorentz


def entropy_density_error(
   run_dir: Path,
   target_cells: int | tuple[int, ...] | None = None,
   *,
   rho0: float = 1.0,
   amplitude: float = 0.1,
   velocity: float = 0.3,
) -> float:
   """Return the entropy-mode normalized mean absolute dens error."""
   plotfile = latest_plotfile(run_dir, require=True)
   names, _, time_value = parse_plot_header(plotfile)
   state = read_composite_state(plotfile, names)
   if target_cells is not None:
      target_shape = (target_cells,) if isinstance(target_cells, int) else target_cells
      if len(target_shape) != state.ndim - 1:
         raise ValueError("target mesh and hierarchy dimensions differ")
      for axis, count in enumerate(target_shape, start=1):
         if count % state.shape[axis] != 0:
            raise ValueError(
               f"target mesh {target_shape} is not an integer multiple of hierarchy "
               f"mesh {state.shape[1:]}"
            )
         state = np.repeat(state, count // state.shape[axis], axis=axis)
   exact = entropy_density_exact(
      state.shape[1:],
      time_value,
      rho0=rho0,
      amplitude=amplitude,
      velocity=velocity,
   )
   dens_scale = rho0 * amplitude / math.sqrt(1.0 - velocity * velocity)
   return float(np.mean(np.abs(state[names.index("dens")] - exact)) / dens_scale)


def shock_heating_solution(
    lorentz: float, *, pressure: float = 1.0e-6, gamma: float = 4.0 / 3.0,
) -> dict[str, float]:
    """Exact symmetric collision of ideal-gas streams with upstream rho=1.

    The downstream fluid is stationary. Solve mass, momentum and total-energy
    Rankine-Hugoniot conditions for the outward shock speed. Finite upstream
    pressure is retained; as p -> 0 this reduces to Meliani et al. (2007), Eq. 7,
    https://arxiv.org/abs/astro-ph/0701434. This is a postprocessing oracle,
    independent of the production Riemann solver and primitive recovery.
    """
    if lorentz <= 1.0 or pressure < 0.0 or not 1.0 < gamma <= 2.0:
        raise ValueError("require W > 1, p >= 0 and 1 < gamma <= 2")
    velocity = math.sqrt(1.0 - lorentz**-2)
    enthalpy = 1.0 + gamma * pressure / (gamma - 1.0)
    upstream_energy = enthalpy * lorentz**2 - pressure
    upstream_momentum = enthalpy * lorentz**2 * velocity
    linear = ((gamma - 1.0) * (lorentz - upstream_energy)
              + pressure + upstream_momentum * velocity)
    constant = (gamma - 1.0) * (lorentz * velocity - upstream_momentum)
    discriminant = math.sqrt(linear**2 - 4.0 * upstream_momentum * constant)
    speed = -2.0 * constant / (linear + discriminant)
    density = lorentz * (velocity + speed) / speed
    downstream_pressure = pressure + upstream_momentum * (velocity + speed)
    return {
        "velocity": velocity, "shock_speed": speed,
        "upstream_dens": lorentz, "upstream_mom1": upstream_momentum,
        "upstream_tau": upstream_energy - lorentz,
        "downstream_dens": density,
        "downstream_tau": downstream_pressure / (gamma - 1.0),
        "downstream_pressure": downstream_pressure,
    }


def shock_heating_cell_averages(
    cells: int, time: float, solution: dict[str, float],
) -> dict[str, np.ndarray]:
    """Integrate the three constant states over cells of the unit interval."""
    extent = solution["shock_speed"] * time
    if cells <= 0 or not 0.0 <= extent < 0.5:
        raise ValueError("positive mesh and shocks inside the domain required")
    edges = np.linspace(0.0, 1.0, cells + 1)
    left = np.clip((0.5 - extent - edges[:-1]) * cells, 0.0, 1.0)
    right = np.clip((edges[1:] - 0.5 - extent) * cells, 0.0, 1.0)
    middle = 1.0 - left - right
    return {
        "dens": (left + right) * solution["upstream_dens"]
                + middle * solution["downstream_dens"],
        "mom1": (left - right) * solution["upstream_mom1"],
        "tau": (left + right) * solution["upstream_tau"]
               + middle * solution["downstream_tau"],
    }


def circular_alfven_speed() -> float:
    """ECHO Eq. 85 for rho=p=B_normal=B_transverse=1 and gamma=4/3."""
    inertia = 7.0  # rho*h + B_normal^2 + B_transverse^2
    speed_squared = 2.0 / (inertia + math.sqrt(inertia**2 - 4.0))
    # Independent defining momentum balance, ECHO Eq. 84.
    if not math.isclose((inertia - speed_squared) * speed_squared, 1.0,
                        rel_tol=2e-15):
        raise ValueError("CP Alfven speed fails the nonlinear wave relation")
    return math.sqrt(speed_squared)


def circular_alfven_conserved(cells: int, time: float, *,
                             cell_average: bool = True) -> dict[str, np.ndarray]:
    """Exact flat-Valencia CP wave on [0,1], with ECHO's unit amplitudes.

    Constant magnetic/velocity magnitudes make dens, normal momentum and tau
    constant. The remaining conserved fields are single harmonics, so their
    finite-volume averages follow exactly by multiplying by sinc(pi*dx).
    This oracle does not call the evolution code's generated P2C kernels.
    """
    speed = circular_alfven_speed()
    lorentz = 1.0 / math.sqrt(1.0 - speed**2)
    phase = 2.0 * math.pi * ((np.arange(cells) + 0.5) / cells - speed * time)
    cosine, sine = np.cos(phase), np.sin(phase)
    z_value = 5.0 * lorentz**2  # rho*h*W^2
    comoving_b_squared = 2.0 / lorentz**2 + speed**2
    average_factor = np.sinc(1.0 / cells) if cell_average else 1.0
    return {
        "dens": np.full(cells, lorentz),
        "mom1": np.full(cells, speed),
        "mom2": -(z_value + 1.0) * speed * cosine * average_factor,
        "mom3": -(z_value + 1.0) * speed * sine * average_factor,
        "tau": np.full(cells, z_value + 1.0 - 0.5 * comoving_b_squared - lorentz),
        "b1": np.ones(cells),
        "b2": cosine * average_factor,
        "b3": sine * average_factor,
    }
