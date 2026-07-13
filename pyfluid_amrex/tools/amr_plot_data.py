"""Shared hierarchy-aware data views for PyFluid AMReX plotting.

The hierarchy is authoritative.  Native renderers consume the original tiles;
uniform views are optional presentation products with finest-level data taking
precedence over covered coarse data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from plotfile_reader import (
   level_chunks,
   level_count,
   parse_level_shapes,
   parse_plot_bounds,
   parse_plot_header,
)


@dataclass(frozen=True)
class SliceTile:
   level: int
   lo: tuple[int, int]
   hi: tuple[int, int]
   level_shape: tuple[int, int]
   values: np.ndarray


@dataclass(frozen=True)
class UniformSlice:
   values: np.ndarray
   boxes: tuple[tuple[int, float, float, float, float], ...]
   time: float
   ndim: int
   names: tuple[str, ...]
   xlim: tuple[float, float]
   ylim: tuple[float, float]


@dataclass(frozen=True)
class HierarchyComparison:
   component: str
   relative_l1: float
   relative_l2: float
   relative_linf: float


def component_timeline_anchor(component: str) -> str:
   """Return one stored component from the plotfile series needed by a view."""
   if component.startswith("abs_"):
      return component_timeline_anchor(component.removeprefix("abs_"))
   if component in {"dye", "velocity_mag", "vorticity3", "current3", "schlieren", "pressure"}:
      return "rho"
   if component in {"v1", "v2", "v3"}:
      return "rho"
   if component.endswith("_mag"):
      return f"{component.removesuffix('_mag')}1"
   return component


def canvas_shape(
   xlim: tuple[float, float],
   ylim: tuple[float, float],
   max_pixels: int,
) -> tuple[int, int]:
   width = max(xlim[1] - xlim[0], 1.0e-30)
   height = max(ylim[1] - ylim[0], 1.0e-30)
   if width >= height:
      return max(1, int(round(max_pixels * height / width))), max_pixels
   return max_pixels, max(1, int(round(max_pixels * width / height)))


def plot_dimensions(ndim: int, slice_axis: str) -> tuple[int, int]:
   if ndim == 2:
      return 0, 1
   axis = {"x": 0, "y": 1, "z": 2}[slice_axis]
   dimensions = [dim for dim in range(3) if dim != axis]
   return dimensions[0], dimensions[1]


def slice_level_index(
   level_shape: tuple[int, ...],
   ndim: int,
   slice_axis: str,
   slice_fraction: float,
) -> int:
   if ndim == 2:
      return 0
   axis = {"x": 0, "y": 1, "z": 2}[slice_axis]
   return min(max(int(slice_fraction * level_shape[axis]), 0), level_shape[axis] - 1)


def extract_plane(
   values: np.ndarray,
   lo: tuple[int, ...],
   hi: tuple[int, ...],
   ndim: int,
   slice_axis: str,
   slice_index: int,
) -> np.ndarray | None:
   if ndim == 2:
      return values
   axis = {"x": 0, "y": 1, "z": 2}[slice_axis]
   if slice_index < lo[axis] or slice_index > hi[axis]:
      return None
   return np.take(values, slice_index - lo[axis], axis=axis)


def component_values(
   data: np.ndarray,
   names: list[str] | tuple[str, ...],
   component: str,
   gamma: float,
   cell_size: tuple[float, ...] | None = None,
) -> np.ndarray:
   if component.startswith("abs_"):
      source_component = component.removeprefix("abs_")
      return np.abs(component_values(data, names, source_component, gamma, cell_size))
   if component in names:
      return data[names.index(component)]
   if component == "dye" and "rho" in names and "dye_mass" in names:
      return data[names.index("dye_mass")] / data[names.index("rho")]
   velocity_names = {"v1": "mom1", "v2": "mom2", "v3": "mom3"}
   if component in velocity_names and "rho" in names and velocity_names[component] in names:
      return data[names.index(velocity_names[component])] / data[names.index("rho")]
   if component == "velocity_mag" and all(name in names for name in ("rho", "mom1", "mom2", "mom3")):
      rho = data[names.index("rho")]
      return np.sqrt(sum(data[names.index(name)] ** 2 for name in ("mom1", "mom2", "mom3"))) / rho
   if component == "vorticity3" and all(name in names for name in ("rho", "mom1", "mom2")):
      if data.ndim < 3:
         raise ValueError("vorticity3 requires at least 2D data")
      rho = data[names.index("rho")]
      v1 = data[names.index("mom1")] / rho
      v2 = data[names.index("mom2")] / rho
      if v1.shape[0] < 2 or v1.shape[1] < 2:
         return np.zeros_like(v1)
      dx1 = 1.0 if cell_size is None else cell_size[0]
      dx2 = 1.0 if cell_size is None else cell_size[1]
      return np.gradient(v2, dx1, axis=0, edge_order=1) - np.gradient(v1, dx2, axis=1, edge_order=1)
   if component == "current3" and all(name in names for name in ("b1", "b2")):
      if data.ndim < 3:
         raise ValueError("current3 requires at least 2D data")
      b1 = data[names.index("b1")]
      b2 = data[names.index("b2")]
      if b1.shape[0] < 2 or b1.shape[1] < 2:
         return np.zeros_like(b1)
      dx1 = 1.0 if cell_size is None else cell_size[0]
      dx2 = 1.0 if cell_size is None else cell_size[1]
      return np.gradient(b2, dx1, axis=0, edge_order=1) - np.gradient(b1, dx2, axis=1, edge_order=1)
   if component == "schlieren" and "rho" in names:
      rho = data[names.index("rho")]
      if rho.ndim < 2 or rho.shape[0] < 2 or rho.shape[1] < 2:
         return np.zeros_like(rho)
      dx1 = 1.0 if cell_size is None else cell_size[0]
      dx2 = 1.0 if cell_size is None else cell_size[1]
      grad1 = np.gradient(rho, dx1, axis=0, edge_order=1)
      grad2 = np.gradient(rho, dx2, axis=1, edge_order=1)
      return np.sqrt(grad1 * grad1 + grad2 * grad2)
   if component == "pressure":
      required = ("rho", "mom1", "mom2", "mom3", "energy")
      if not all(name in names for name in required):
         raise ValueError(f"pressure requires {', '.join(required)}")
      rho = data[names.index("rho")]
      kinetic = 0.5 * sum(data[names.index(name)] ** 2 for name in ("mom1", "mom2", "mom3")) / rho
      magnetic = np.zeros_like(rho)
      if all(name in names for name in ("b1", "b2", "b3")):
         magnetic = 0.5 * sum(data[names.index(name)] ** 2 for name in ("b1", "b2", "b3"))
      return (data[names.index("energy")] - kinetic - magnetic) * (gamma - 1.0)
   if component.endswith("_mag"):
      prefix = component.removesuffix("_mag")
      vector_names = tuple(f"{prefix}{axis}" for axis in (1, 2, 3))
      if all(name in names for name in vector_names):
         return np.sqrt(sum(data[names.index(name)] ** 2 for name in vector_names))
   raise ValueError(f"{component} not found; available components: {', '.join(names)}")


def _crop_pixel_extent(
   lo: int,
   hi: int,
   level_cells: int,
   domain: tuple[float, float],
   crop: tuple[float, float],
   canvas_cells: int,
) -> tuple[int, int]:
   domain_width = domain[1] - domain[0]
   crop_width = crop[1] - crop[0]
   physical_lo = domain[0] + lo * domain_width / level_cells
   physical_hi = domain[0] + (hi + 1) * domain_width / level_cells
   pixel_lo = int(np.floor((physical_lo - crop[0]) * canvas_cells / crop_width))
   pixel_hi = int(np.ceil((physical_hi - crop[0]) * canvas_cells / crop_width))
   pixel_lo = min(max(pixel_lo, 0), canvas_cells - 1)
   pixel_hi = min(max(pixel_hi, pixel_lo + 1), canvas_cells)
   return pixel_lo, pixel_hi


def _resize_nearest(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
   out_y, out_x = shape
   if out_y <= 0 or out_x <= 0:
      return np.empty((0, 0), dtype=values.dtype)
   source_y = np.minimum((np.arange(out_y) * values.shape[0]) // out_y, values.shape[0] - 1)
   source_x = np.minimum((np.arange(out_x) * values.shape[1]) // out_x, values.shape[1] - 1)
   return values[np.ix_(source_y, source_x)]


def _resize_linear(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
   out_y, out_x = shape
   if values.shape == shape:
      return values.copy()
   source_x = np.arange(values.shape[1], dtype=float)
   target_x = (np.arange(out_x, dtype=float) + 0.5) * values.shape[1] / out_x - 0.5
   target_x = np.clip(target_x, 0.0, max(values.shape[1] - 1, 0))
   horizontal = np.empty((values.shape[0], out_x), dtype=float)
   for row in range(values.shape[0]):
      horizontal[row] = np.interp(target_x, source_x, values[row])
   source_y = np.arange(values.shape[0], dtype=float)
   target_y = (np.arange(out_y, dtype=float) + 0.5) * values.shape[0] / out_y - 0.5
   target_y = np.clip(target_y, 0.0, max(values.shape[0] - 1, 0))
   output = np.empty((out_y, out_x), dtype=float)
   for column in range(out_x):
      output[:, column] = np.interp(target_y, source_y, horizontal[:, column])
   return output


def _overlap_weights(source_cells: int, output_cells: int) -> np.ndarray:
   """Return piecewise-constant overlap weights from source to output cells."""
   source_edges = np.linspace(0.0, 1.0, source_cells + 1)
   output_edges = np.linspace(0.0, 1.0, output_cells + 1)
   weights = np.zeros((output_cells, source_cells), dtype=float)
   for output_index in range(output_cells):
      left = np.maximum(source_edges[:-1], output_edges[output_index])
      right = np.minimum(source_edges[1:], output_edges[output_index + 1])
      overlap = np.maximum(right - left, 0.0)
      weights[output_index] = overlap * output_cells
   return weights


def _resize_average(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
   """Conservatively remap cell averages to a requested pixel grid."""
   output_y, output_x = shape
   if values.shape == shape:
      return values.copy()
   weights_y = _overlap_weights(values.shape[0], output_y)
   weights_x = _overlap_weights(values.shape[1], output_x)
   return weights_y @ values @ weights_x.T


class HierarchySlice:
   """Lazy logical slice over an arbitrary-depth AMReX hierarchy."""

   def __init__(
      self,
      plotfile: Path,
      component: str,
      gamma: float,
      *,
      slice_axis: str = "z",
      slice_fraction: float = 0.5,
      xlim: tuple[float, float] | None = None,
      ylim: tuple[float, float] | None = None,
   ) -> None:
      self.plotfile = Path(plotfile)
      names, self.ndim, self.time = parse_plot_header(self.plotfile)
      if self.ndim < 2:
         raise ValueError("hierarchy slices require a 2D or 3D plotfile")
      if not 0.0 <= slice_fraction <= 1.0:
         raise ValueError("slice_fraction must be in [0, 1]")
      self.names = tuple(names)
      self.component = component
      self.gamma = gamma
      self.slice_axis = slice_axis
      self.slice_fraction = slice_fraction
      self.problo, self.probhi = parse_plot_bounds(self.plotfile)
      self.level_shapes = tuple(parse_level_shapes(self.plotfile))
      self.dim0, self.dim1 = plot_dimensions(self.ndim, slice_axis)
      self.xlim = xlim or (self.problo[self.dim0], self.probhi[self.dim0])
      self.ylim = ylim or (self.problo[self.dim1], self.probhi[self.dim1])

   @property
   def num_levels(self) -> int:
      return level_count(self.plotfile)

   def tiles(self) -> Iterator[SliceTile]:
      for level in range(self.num_levels):
         level_shape = self.level_shapes[level]
         slice_index = slice_level_index(level_shape, self.ndim, self.slice_axis, self.slice_fraction)
         index_lo = [0] * self.ndim
         index_hi = [level_shape[dim] - 1 for dim in range(self.ndim)]
         index_lo[self.dim0] = max(
            0,
            int(np.floor((self.xlim[0] - self.problo[self.dim0]) * level_shape[self.dim0]
                         / (self.probhi[self.dim0] - self.problo[self.dim0]))),
         )
         index_hi[self.dim0] = min(
            level_shape[self.dim0] - 1,
            int(np.ceil((self.xlim[1] - self.problo[self.dim0]) * level_shape[self.dim0]
                        / (self.probhi[self.dim0] - self.problo[self.dim0]))) - 1,
         )
         index_lo[self.dim1] = max(
            0,
            int(np.floor((self.ylim[0] - self.problo[self.dim1]) * level_shape[self.dim1]
                         / (self.probhi[self.dim1] - self.problo[self.dim1]))),
         )
         index_hi[self.dim1] = min(
            level_shape[self.dim1] - 1,
            int(np.ceil((self.ylim[1] - self.problo[self.dim1]) * level_shape[self.dim1]
                        / (self.probhi[self.dim1] - self.problo[self.dim1]))) - 1,
         )
         if self.ndim == 3:
            slice_dim = {"x": 0, "y": 1, "z": 2}[self.slice_axis]
            index_lo[slice_dim] = slice_index
            index_hi[slice_dim] = slice_index
         cell_size = tuple(
            (self.probhi[dim] - self.problo[dim]) / level_shape[dim]
            for dim in range(self.ndim)
         )
         for lo, hi, data in level_chunks(
            self.plotfile,
            level,
            self.ndim,
            names=list(self.names),
            index_bounds=(tuple(index_lo), tuple(index_hi)),
         ):
            values = component_values(data, self.names, self.component, self.gamma, cell_size)
            plane = extract_plane(values, lo, hi, self.ndim, self.slice_axis, slice_index)
            if plane is None:
               continue
            logical_lo = (lo[self.dim0], lo[self.dim1])
            logical_hi = (hi[self.dim0], hi[self.dim1])
            yield SliceTile(
               level,
               logical_lo,
               logical_hi,
               (level_shape[self.dim0], level_shape[self.dim1]),
               plane.T,
            )

   def uniform(self, max_pixels: int, *, method: str = "nearest") -> UniformSlice:
      if method not in {"nearest", "linear", "average"}:
         raise ValueError("uniform sampling method must be nearest, linear, or average")
      canvas_y, canvas_x = canvas_shape(self.xlim, self.ylim, max_pixels)
      image = np.full((canvas_y, canvas_x), np.nan)
      boxes: list[tuple[int, float, float, float, float]] = []
      resize = {
         "nearest": _resize_nearest,
         "linear": _resize_linear,
         "average": _resize_average,
      }[method]
      domain_x = (self.problo[self.dim0], self.probhi[self.dim0])
      domain_y = (self.problo[self.dim1], self.probhi[self.dim1])
      for tile in self.tiles():
         level_x, level_y = tile.level_shape
         frac_x0 = (self.xlim[0] - domain_x[0]) / (domain_x[1] - domain_x[0])
         frac_x1 = (self.xlim[1] - domain_x[0]) / (domain_x[1] - domain_x[0])
         frac_y0 = (self.ylim[0] - domain_y[0]) / (domain_y[1] - domain_y[0])
         frac_y1 = (self.ylim[1] - domain_y[0]) / (domain_y[1] - domain_y[0])
         index_x0 = max(tile.lo[0], int(np.floor(frac_x0 * level_x)))
         index_x1 = min(tile.hi[0] + 1, int(np.ceil(frac_x1 * level_x)))
         index_y0 = max(tile.lo[1], int(np.floor(frac_y0 * level_y)))
         index_y1 = min(tile.hi[1] + 1, int(np.ceil(frac_y1 * level_y)))
         if index_x0 >= index_x1 or index_y0 >= index_y1:
            continue
         local = tile.values[
            index_y0 - tile.lo[1] : index_y1 - tile.lo[1],
            index_x0 - tile.lo[0] : index_x1 - tile.lo[0],
         ]
         pixel_x0, pixel_x1 = _crop_pixel_extent(
            index_x0, index_x1 - 1, level_x, domain_x, self.xlim, canvas_x
         )
         pixel_y0, pixel_y1 = _crop_pixel_extent(
            index_y0, index_y1 - 1, level_y, domain_y, self.ylim, canvas_y
         )
         image[pixel_y0:pixel_y1, pixel_x0:pixel_x1] = resize(
            local, (pixel_y1 - pixel_y0, pixel_x1 - pixel_x0)
         )
         if tile.level > 0:
            xlo = domain_x[0] + tile.lo[0] * (domain_x[1] - domain_x[0]) / level_x
            xhi = domain_x[0] + (tile.hi[0] + 1) * (domain_x[1] - domain_x[0]) / level_x
            ylo = domain_y[0] + tile.lo[1] * (domain_y[1] - domain_y[0]) / level_y
            yhi = domain_y[0] + (tile.hi[1] + 1) * (domain_y[1] - domain_y[0]) / level_y
            boxes.append((tile.level, xlo, xhi, ylo, yhi))
      return UniformSlice(
         image,
         tuple(boxes),
         self.time,
         self.ndim,
         self.names,
         self.xlim,
         self.ylim,
      )


def compare_hierarchy_slices(
   reference_plotfile: Path,
   candidate_plotfile: Path,
   components: tuple[str, ...],
   *,
   gamma: float = 5.0 / 3.0,
   max_pixels: int = 512,
   method: str = "average",
) -> tuple[HierarchyComparison, ...]:
   """Compare two AMR hierarchies on one conservative physical-space view."""
   comparisons = []
   for component in components:
      reference = HierarchySlice(reference_plotfile, component, gamma).uniform(max_pixels, method=method)
      candidate = HierarchySlice(candidate_plotfile, component, gamma).uniform(max_pixels, method=method)
      if reference.values.shape != candidate.values.shape:
         raise ValueError(f"{component} hierarchy views have different shapes")
      if reference.xlim != candidate.xlim or reference.ylim != candidate.ylim:
         raise ValueError(f"{component} hierarchy views cover different physical regions")
      time_scale = max(1.0, abs(reference.time), abs(candidate.time))
      if abs(reference.time - candidate.time) > 64.0 * np.finfo(float).eps * time_scale:
         raise ValueError(
            f"{component} hierarchy times differ: {reference.time:.17g} vs {candidate.time:.17g}"
         )
      mask = np.isfinite(reference.values) & np.isfinite(candidate.values)
      if not np.any(mask):
         raise ValueError(f"{component} hierarchy views have no overlapping finite cells")
      difference = candidate.values[mask] - reference.values[mask]
      scale = max(float(np.sqrt(np.mean(reference.values[mask] ** 2))), 1.0e-300)
      comparisons.append(HierarchyComparison(
         component=component,
         relative_l1=float(np.mean(np.abs(difference))) / scale,
         relative_l2=float(np.sqrt(np.mean(difference * difference))) / scale,
         relative_linf=float(np.max(np.abs(difference))) / scale,
      ))
   return tuple(comparisons)
