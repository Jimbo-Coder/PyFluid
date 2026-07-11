#!/usr/bin/env python3
"""Causal-domain metadata for localized AMReX validation problems."""

from __future__ import annotations

import math
from dataclasses import dataclass


OPEN_BOUNDARIES = frozenset(("outflow", "diode"))


@dataclass(frozen=True)
class CausalModel:
   center: tuple[float, float, float]
   initial_support_radius: float
   signal_speed: float
   boundary_distance: float

   @property
   def boundary_contact_time(self) -> float:
      if not math.isfinite(self.boundary_distance):
         return math.inf
      return max(0.0, self.boundary_distance - self.initial_support_radius) / self.signal_speed

   def radius(self, time_value: float, *, safety: float = 1.0) -> float:
      return self.initial_support_radius + safety * self.signal_speed * max(time_value, 0.0)


def _float(params: dict[str, str], key: str, default: float) -> float:
   try:
      value = float(params.get(key, default))
   except (TypeError, ValueError):
      return default
   return value if math.isfinite(value) else default


def _tokens(params: dict[str, str], key: str) -> tuple[str, ...]:
   return tuple(params.get(key, "").replace(",", " ").split())


def _active_axes(params: dict[str, str], ndim: int) -> tuple[str, ...]:
   axes = _tokens(params, "mesh.axes")
   if axes:
      return axes
   coordinate = params.get("mesh.coordinate", "cartesian").strip().lower()
   canonical = {
      "cartesian": ("x", "y", "z"),
      "cylindrical": ("r", "phi", "z"),
      "spherical": ("r", "theta", "phi"),
   }
   return canonical.get(coordinate, canonical["cartesian"])[:ndim]


def _boundary_name(params: dict[str, str], direction: int, high_side: bool) -> str:
   suffix = ("x", "y", "z")[direction] + ("hi" if high_side else "lo")
   return params.get(f"pyfluid.bc_{suffix}", params.get("pyfluid.bc", "periodic")).strip().lower()


def _angular_surface_distance(radius: float, delta: float) -> float:
   delta = abs(delta)
   if delta <= 0.5 * math.pi:
      return radius * math.sin(delta)
   return radius


def _coordinate_boundary_distance(
   coordinate: str,
   axis: str,
   center_by_axis: dict[str, float],
   boundary: float,
) -> float:
   delta = abs(center_by_axis[axis] - boundary)
   if coordinate == "cartesian" or axis in ("r", "z"):
      return delta
   if coordinate == "cylindrical" and axis == "phi":
      return _angular_surface_distance(center_by_axis["r"], delta)
   if coordinate == "spherical" and axis == "theta":
      return _angular_surface_distance(center_by_axis["r"], delta)
   if coordinate == "spherical" and axis == "phi":
      cylindrical_radius = center_by_axis["r"] * math.sin(center_by_axis["theta"])
      return _angular_surface_distance(abs(cylindrical_radius), delta)
   return math.inf


def _open_boundary_distance(
   params: dict[str, str],
   center: tuple[float, float, float],
) -> float:
   prob_lo = tuple(float(value) for value in _tokens(params, "geometry.prob_lo"))
   prob_hi = tuple(float(value) for value in _tokens(params, "geometry.prob_hi"))
   if not prob_lo or len(prob_lo) != len(prob_hi):
      return math.inf
   axes = _active_axes(params, len(prob_lo))
   if len(axes) != len(prob_lo):
      return math.inf
   periodic = tuple(int(value) for value in _tokens(params, "geometry.is_periodic"))
   coordinate = params.get("mesh.coordinate", "cartesian").strip().lower()
   center_by_axis = {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "r": 1.0,
      "theta": 0.5 * math.pi,
      "phi": 0.0,
   }
   center_by_axis.update({axis: center[direction] for direction, axis in enumerate(axes)})
   distances: list[float] = []
   for direction, axis in enumerate(axes):
      if direction < len(periodic) and periodic[direction] != 0:
         continue
      for high_side, boundary in ((False, prob_lo[direction]), (True, prob_hi[direction])):
         if _boundary_name(params, direction, high_side) in OPEN_BOUNDARIES:
            distances.append(_coordinate_boundary_distance(coordinate, axis, center_by_axis, boundary))
   return min(distances, default=math.inf)


def _signal_speed(params: dict[str, str]) -> float:
   rho = max(_float(params, "pyfluid.rho0", 1.0), 1.0e-300)
   pressure = max(
      _float(params, "pyfluid.pressure0", 1.0)
      + abs(_float(params, "pyfluid.amplitude", 0.0)),
      0.0,
   )
   gamma = _float(params, "pyfluid.gamma", 5.0 / 3.0)
   sound_speed2 = max(gamma * pressure / rho, 0.0)
   magnetic_speed2 = 0.0
   if params.get("pyfluid.physics", "hydro").strip().lower() == "mhd":
      magnetic_speed2 = sum(_float(params, f"pyfluid.b{component}_0", 0.0) ** 2 for component in (1, 2, 3)) / rho
   background_speed = math.sqrt(
      sum(_float(params, f"pyfluid.v{component}_0", 0.0) ** 2 for component in (1, 2, 3))
   )
   return background_speed + math.sqrt(sound_speed2 + magnetic_speed2)


def infer_localized_causal_model(
   params: dict[str, str],
   *,
   support_widths: float = 8.0,
) -> CausalModel | None:
   """Return a conservative causal model for a supported localized run.

   Gaussian pressure bumps use an effective numerical support of
   ``support_widths * sigma``. The Gaussian is not mathematically compact, so
   this is intentionally a numerical validation region rather than an exact
   continuum domain of dependence.
   """
   if params.get("mesh.patch_system", "single").strip().lower() != "single":
      return None
   if params.get("pyfluid.ic", "").strip().lower() != "pressure_bump":
      return None
   if params.get("pyfluid.source", "none").strip().lower() not in ("", "none", "off"):
      return None
   if params.get("pyfluid.gravity", "none").strip().lower() not in ("", "none", "off"):
      return None
   if params.get("pyfluid.diffusion", "none").strip().lower() not in ("", "none", "off"):
      return None
   if abs(_float(params, "pyfluid.viscosity", 0.0)) > 0.0 or abs(_float(params, "pyfluid.resistivity", 0.0)) > 0.0:
      return None
   solid_motion = params.get("pyfluid.solid_motion", "none").strip().lower()
   if solid_motion not in ("", "none", "off", "static"):
      return None
   eos = params.get("pyfluid.eos", "ideal_gas").strip().lower()
   if eos != "ideal_gas":
      return None

   center = tuple(_float(params, f"pyfluid.bump_{axis}0", 0.0) for axis in ("x", "y", "z"))
   sigma = _float(params, "pyfluid.bump_sigma", 0.0)
   speed = _signal_speed(params)
   if sigma <= 0.0 or speed <= 0.0:
      return None
   return CausalModel(
      center=center,
      initial_support_radius=support_widths * sigma,
      signal_speed=speed,
      boundary_distance=_open_boundary_distance(params, center),
   )


def rows_before_boundary_contact(
   rows: list[dict[str, str]],
   model: CausalModel,
) -> list[dict[str, str]]:
   contact_time = model.boundary_contact_time
   return [
      row for row in rows
      if float(row.get("time", "nan")) <= contact_time
   ]
