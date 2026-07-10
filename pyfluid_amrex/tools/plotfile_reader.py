#!/usr/bin/env python3
"""Shared readers for the canonical PyFluid AMReX output layout.

State plotfiles live in ``stateGF/`` and auxiliary plotfiles live in
``auxGF/``. Each may use AMReX native plotfiles or HDF5 plotfiles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
   import h5py
except Exception:
   h5py = None


FAB_LINE = re.compile(r"FabOnDisk:\s+(\S+)\s+(\d+)")
FAB_HEADER = re.compile(r"\)\)\(\(([^)]*)\)\s+\(([^)]*)\)\s+\(([^)]*)\)\)\s+(\d+)\s*$")
FAB_HEADER_1D = re.compile(r"\)\)\(\((-?\d+)\)\s+\((-?\d+)\)\s+\((-?\d+)\)\)\s+(\d+)\s*$")

PLOTFILE_STEP_RE = re.compile(r"(\d+)(?:\.(?:h5|hdf5))?$")
CHECKPOINT_STEP_RE = re.compile(r"^chk(\d+)$")
PLOTFILE_ROOTS = frozenset(("stateGF", "auxGF"))


def plotfile_step(path: Path) -> int:
   match = PLOTFILE_STEP_RE.search(_ensure_path(path).name)
   if not match:
      raise ValueError(f"could not parse step from {path}")
   return int(match.group(1))


@dataclass(frozen=True)
class PlotfileSample:
   step: int
   time: float
   path: Path


def _ensure_path(plotfile):
   return plotfile if isinstance(plotfile, Path) else Path(plotfile)


def _plot_root(directory: Path, prefix: str) -> Path:
   if prefix not in PLOTFILE_ROOTS:
      supported = ", ".join(sorted(PLOTFILE_ROOTS))
      raise ValueError(f"unsupported plotfile prefix {prefix!r}; use {supported}")
   path = _ensure_path(directory)
   return path if path.name == prefix else path / prefix


def _resolve_plotfile(plotfile: Path):
   path = _ensure_path(plotfile)
   if path.is_dir() and (path / "Header").is_file():
      return path, "amrex", path
   if path.suffix in {".h5", ".hdf5"} and path.is_file():
      return path, "hdf5", path
   if path.suffix == "":
      for suffix in (".h5", ".hdf5"):
         hdf5_candidate = path.with_suffix(suffix)
         if hdf5_candidate.is_file():
            return hdf5_candidate, "hdf5", path
   raise FileNotFoundError(f"could not find AMReX plotfile at {plotfile}")


def discover_plotfiles(directory: Path, prefix: str = "stateGF") -> list[Path]:
   plot_dir = _plot_root(directory, prefix)
   candidates: list[Path] = []
   if not plot_dir.is_dir():
      return candidates
   for path in plot_dir.iterdir():
      if not path.name.startswith(prefix):
         continue
      try:
         plotfile_step(path)
      except ValueError:
         continue
      if path.is_dir() and (path / "Header").exists():
         candidates.append(path)
         continue
      if path.is_file() and path.suffix in {".h5", ".hdf5"}:
         candidates.append(path)
   return candidates


def sorted_plotfiles(directory: Path, prefix: str = "stateGF", *, require: bool = False) -> list[Path]:
   plotfiles = discover_plotfiles(_ensure_path(directory), prefix)
   plotfiles = sorted(plotfiles, key=plotfile_step)
   if not plotfiles and require:
      raise FileNotFoundError(f"no plotfiles found under {directory} with prefix '{prefix}'")
   return plotfiles


def latest_plotfile(directory: Path, prefix: str = "stateGF", *, require: bool = False) -> Path | None:
   candidates = sorted_plotfiles(directory, prefix, require=require)
   return candidates[-1] if candidates else None


def plotfile_at_step(
   directory: Path,
   step: int,
   prefix: str = "stateGF",
   *,
   require: bool = False,
) -> Path | None:
   for plotfile in sorted_plotfiles(directory, prefix):
      if plotfile_step(plotfile) == step:
         return plotfile
   if require:
      raise FileNotFoundError(f"no {prefix} plotfile at step {step} under {directory}")
   return None


def sorted_aux_plotfiles(directory: Path, *, require: bool = False) -> list[Path]:
   return sorted_plotfiles(directory, "auxGF", require=require)


def _checkpoint_dir(output_dir: Path) -> Path:
   return output_dir / "checkpoints"


def checkpoint_step(checkpoint_dir: Path) -> int:
   match = CHECKPOINT_STEP_RE.match(_ensure_path(checkpoint_dir).name)
   if not match:
      raise ValueError(f"invalid checkpoint directory name: {_ensure_path(checkpoint_dir).name}")
   return int(match.group(1))


def checkpoint_path(output_dir: Path, step: int) -> Path:
   return _checkpoint_dir(output_dir) / f"chk{step:05d}"


def sorted_checkpoints(output_dir: Path, *, require: bool = False) -> list[Path]:
   checkpoint_root = _checkpoint_dir(output_dir)
   if not checkpoint_root.exists():
      if require:
         raise FileNotFoundError(f"missing checkpoint directory: {checkpoint_root}")
      return []
   checkpoints = []
   for entry in checkpoint_root.iterdir():
      if not entry.is_dir():
         continue
      if not CHECKPOINT_STEP_RE.fullmatch(entry.name):
         continue
      checkpoints.append(entry)
   checkpoints = sorted(checkpoints, key=checkpoint_step)
   if not checkpoints and require:
      raise FileNotFoundError(f"no checkpoints found under {checkpoint_root}")
   return checkpoints


def latest_checkpoint(output_dir: Path) -> Path | None:
   checkpoints = sorted_checkpoints(output_dir, require=False)
   return checkpoints[-1] if checkpoints else None


def _select_timeline_stride(
   samples: list[PlotfileSample],
   *,
   frame_stride: int,
) -> list[PlotfileSample]:
   if frame_stride <= 1:
      return samples
   if not samples:
      return samples
   selected = samples[::frame_stride]
   if selected[-1] != samples[-1]:
      selected.append(samples[-1])
   return selected


def _select_timeline_target_frames(
   samples: list[PlotfileSample],
   *,
   target_frames: int,
) -> list[PlotfileSample]:
   if target_frames <= 0 or len(samples) <= target_frames:
      return samples
   if target_frames == 1:
      return [samples[0]]
   last = len(samples) - 1
   indices = sorted({round(i * last / (target_frames - 1)) for i in range(target_frames)})
   return [samples[index] for index in indices]


def plotfile_timeline(
   output_dir: Path,
   *,
   prefix: str = "stateGF",
   require: bool = False,
   frame_stride: int = 1,
   target_frames: int = 0,
   max_frames: int = 0
) -> list[PlotfileSample]:
   plotfiles = sorted_plotfiles(_ensure_path(output_dir), prefix=prefix, require=require)
   samples = [
      PlotfileSample(plotfile_step(path), parse_plot_header(path)[2], path)
      for path in plotfiles
   ]
   if frame_stride < 1:
      raise ValueError("frame_stride must be >= 1")
   if target_frames > 0 and max_frames > 0:
      raise ValueError("target_frames and max_frames cannot both be set")
   samples = _select_timeline_stride(samples, frame_stride=frame_stride)
   if target_frames > 0:
      samples = _select_timeline_target_frames(samples, target_frames=target_frames)
   if max_frames <= 0 or len(samples) <= max_frames:
      return samples
   return samples[-max_frames:]


def read_checkpoint_metadata(checkpoint_name: Path) -> dict[str, str]:
   return read_output_metadata(_ensure_path(checkpoint_name) / "Header.meta")


def read_scalars_metadata(run_dir: Path) -> dict[str, str]:
   return read_output_metadata(_ensure_path(run_dir) / "scalars.tsv")


def read_amr_levels_metadata(run_dir: Path) -> dict[str, str]:
   return read_output_metadata(_ensure_path(run_dir) / "amr_levels.tsv")


def read_run_output_metadata(run_dir: Path, filename: str) -> dict[str, str]:
   """Read the metadata header from a run output file.

   `filename` should be one of scalars.tsv, amr_levels.tsv, etc.
   """
   return read_output_metadata(_ensure_path(run_dir) / filename)


def _metadata_path(plotfile: Path, kind: str) -> Path:
   if kind == "amrex":
      return plotfile / "Header"
   return Path(f"{plotfile}.meta")


def _read_hdf5_metadata(plotfile: Path) -> dict[str, str]:
   if h5py is None:
      raise RuntimeError("h5py not installed; cannot read AMReX HDF5 metadata")
   metadata = {}
   with h5py.File(plotfile, "r") as f:
      for key, value in f.attrs.items():
         if isinstance(value, (bytes, np.bytes_)):
            metadata[key] = value.decode("utf-8")
         elif isinstance(value, np.ndarray):
            if value.shape == ():
               metadata[key] = _decode_text_or_bytes(value.item())
            else:
               metadata[key] = ",".join(_decode_text_or_bytes(item) for item in value.tolist())
         else:
            metadata[key] = str(value)
   return metadata


def _read_comment_metadata(path: Path) -> dict[str, str]:
   if not path.exists():
      return {}
   metadata: dict[str, str] = {}
   for raw in path.read_text(errors="ignore").splitlines():
      line = raw.strip()
      if not line.startswith("#"):
         continue
      key, sep, value = line[1:].partition("=")
      if not sep:
         continue
      metadata[key.strip()] = value.strip()
   return metadata


def read_output_metadata(path: Path) -> dict[str, str]:
   p = _ensure_path(path)
   if p.suffix in {".h5", ".hdf5"}:
      return _read_hdf5_metadata(p)
   return _read_comment_metadata(p)


def parse_int_tuple(text: str) -> tuple[int, ...]:
   return tuple(int(value) for value in text.split(","))


def _decode_text_or_bytes(value):
   if isinstance(value, (bytes, np.bytes_)):
      return value.decode("utf-8")
   return str(value)


def _read_text_header(plotfile: Path):
   header = (plotfile / "Header").read_text(errors="ignore").splitlines()
   nvar = int(header[1])
   names = header[2 : 2 + nvar]
   ndim = int(header[2 + nvar])
   time_value = float(header[3 + nvar])
   return header, names, ndim, time_value


def _read_hdf5_attrs(plotfile: Path):
   if h5py is None:
      raise RuntimeError("h5py not installed; cannot read AMReX HDF5 plotfiles")
   with h5py.File(plotfile, "r") as f:
      names = [_decode_text_or_bytes(f.attrs[f"component_{n}"]) for n in range(int(f.attrs["num_components"]))]
      ndim = int(f.attrs["dim"])
      time_value = float(f.attrs["time"])
      nlevels = int(f.attrs["num_levels"])
      prob_lo = tuple(float(v) for v in f["level_0"].attrs["prob_lo"])
      prob_hi = tuple(float(v) for v in f["level_0"].attrs["prob_hi"])
      return names, ndim, time_value, nlevels, prob_lo, prob_hi


def read_plotfile_metadata(plotfile: Path) -> dict[str, str]:
   resolved, kind, _ = _resolve_plotfile(plotfile)
   metadata = {}
   if kind == "hdf5":
      metadata.update(read_output_metadata(resolved))
   path = _metadata_path(resolved, kind)
   if path.exists() and path != resolved:
      metadata.update(read_output_metadata(path))
   if not metadata or path == resolved:
      return {}
   return metadata


def parse_plot_header(plotfile: Path) -> tuple[list[str], int, float]:
   resolved, kind, _ = _resolve_plotfile(plotfile)
   if kind == "amrex":
      _, names, ndim, time_value = _read_text_header(resolved)
      return names, ndim, time_value
   if kind == "hdf5":
      names, ndim, time_value, _, _, _ = _read_hdf5_attrs(resolved)
      return names, ndim, time_value
   raise ValueError(f"unknown plotfile kind for {resolved}")


def parse_plot_bounds(plotfile: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
   resolved, kind, _ = _resolve_plotfile(plotfile)
   if kind == "amrex":
      lines, _, ndim, _ = _read_text_header(resolved)
      nvar = len(lines[2 : 2 + int(lines[1])])
      problo = tuple(float(value) for value in lines[5 + nvar].split()[:ndim])
      probhi = tuple(float(value) for value in lines[6 + nvar].split()[:ndim])
      return problo, probhi
   names, ndim, _, _, problo, probhi = _read_hdf5_attrs(resolved)
   _ = names
   _ = ndim
   return problo, probhi


def parse_level_cell_sizes(plotfile: Path) -> list[tuple[float, ...]]:
   resolved, kind, _ = _resolve_plotfile(plotfile)
   if kind == "amrex":
      lines = (resolved / "Header").read_text(errors="ignore").splitlines()
      nvar = int(lines[1])
      ndim = int(lines[2 + nvar])
      max_level = int(lines[4 + nvar])
      first_dx_line = 10 + nvar
      return [
         tuple(float(value) for value in lines[first_dx_line + lev].split()[:ndim])
         for lev in range(max_level + 1)
      ]
   if h5py is None:
      raise RuntimeError("h5py not installed; cannot read AMReX HDF5 plotfiles")
   with h5py.File(resolved, "r") as f:
      nlevels = int(f.attrs["num_levels"])
      return [tuple(float(v) for v in f[f"level_{lev}"].attrs["Vec_dx"]) for lev in range(nlevels)]


def parse_level_shapes(plotfile: Path) -> list[tuple[int, ...]]:
   problo, probhi = parse_plot_bounds(plotfile)
   return [
      tuple(int(round((probhi[d] - problo[d]) / dx[d])) for d in range(len(dx)))
      for dx in parse_level_cell_sizes(plotfile)
   ]


def level_count(plotfile: Path) -> int:
   resolved, kind, _ = _resolve_plotfile(plotfile)
   if kind == "amrex":
      return len([path for path in resolved.glob("Level_*") if path.is_dir()])
   if h5py is None:
      raise RuntimeError("h5py not installed; cannot read AMReX HDF5 plotfiles")
   with h5py.File(resolved, "r") as f:
      return int(f.attrs["num_levels"])


def parse_fab_index(cell_header: Path) -> list[tuple[Path, int]]:
   path = _ensure_path(cell_header)
   if not path.is_file():
      raise ValueError(f"Cell_H path does not exist or is not a file: {cell_header}")
   entries: list[tuple[Path, int]] = []
   for line in path.read_text(errors="ignore").splitlines():
      match = FAB_LINE.match(line)
      if match:
         entries.append((path.parent / match.group(1), int(match.group(2))))
   return entries


def read_fab(path: Path, offset: int, ndim: int) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
   with path.open("rb") as f:
      f.seek(offset)
      header = f.readline().decode("ascii")
      match = FAB_HEADER.search(header)
      if not match:
         raise ValueError(f"could not parse FAB header in {path} at offset {offset}")
      lo = parse_int_tuple(match.group(1))
      hi = parse_int_tuple(match.group(2))
      _ = parse_int_tuple(match.group(3))
      ncomp = int(match.group(4))
      shape = tuple(hi[d] - lo[d] + 1 for d in range(ndim))
      ncell = int(np.prod(shape))
      data = np.fromfile(f, dtype="<f8", count=ncell * ncomp)
   if ndim == 2:
      return lo, hi, data.reshape((ncomp, shape[1], shape[0])).transpose(0, 2, 1)
   if ndim == 3:
      return lo, hi, data.reshape((ncomp, shape[2], shape[1], shape[0])).transpose(0, 3, 2, 1)
   return lo, hi, data.reshape((ncomp, shape[0]))


def _reshape_chunk(flat_data: np.ndarray, ndim: int, shape: tuple[int, ...]) -> np.ndarray:
   if ndim == 1:
      return flat_data
   if ndim == 2:
      return flat_data.reshape((flat_data.shape[0], shape[1], shape[0])).transpose(0, 2, 1)
   return flat_data.reshape((flat_data.shape[0], shape[2], shape[1], shape[0])).transpose(0, 3, 2, 1)


def _read_hdf5_level_chunks(plotfile: Path, level: int, ndim: int, ncomp: int):
   if h5py is None:
      raise RuntimeError("h5py not installed; cannot read AMReX HDF5 plotfiles")
   axis = "ijk"[:ndim]
   chunks = []
   with h5py.File(plotfile, "r") as f:
      level_group = f[f"level_{level}"]
      boxes = level_group["boxes"][()]
      offsets = level_group["data:offsets=0"][()]
      raw_data = level_group["data:datatype=0"][()]
   for ibox in range(len(boxes)):
      lo = tuple(int(boxes[ibox][f"lo_{axis[d]}"]) for d in range(ndim))
      hi = tuple(int(boxes[ibox][f"hi_{axis[d]}"]) for d in range(ndim))
      start = int(offsets[ibox])
      stop = int(offsets[ibox + 1])
      piece = raw_data[start:stop]
      piece_size = int(np.prod([hi[d] - lo[d] + 1 for d in range(ndim)]))
      expected = piece_size * ncomp
      if len(piece) != expected:
         raise ValueError(f"unexpected HDF5 chunk size for box {ibox}: got {len(piece)}, expected {expected}")
      shape = tuple(hi[d] - lo[d] + 1 for d in range(ndim))
      chunks.append((lo, hi, _reshape_chunk(piece.reshape((ncomp, piece_size)), ndim, shape)))
   return chunks


def level_chunks(
   plotfile: Path,
   level: int,
   ndim: int,
   names: list[str] | None = None,
):
   resolved, kind, _ = _resolve_plotfile(plotfile)
   if kind == "amrex":
      entries = parse_fab_index(resolved / f"Level_{level}" / "Cell_H")
      return [read_fab(path, offset, ndim) for path, offset in entries]
   if h5py is None:
      raise RuntimeError("h5py not installed; cannot read AMReX HDF5 plotfiles")
   if names is None:
      ncomp = int(_read_hdf5_attrs(resolved)[0].__len__())
   else:
      ncomp = len(names)
   return _read_hdf5_level_chunks(resolved, level, ndim, ncomp)


def level_index_boxes(plotfile: Path, level: int, ndim: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
   return [(lo, hi) for lo, hi, _data in level_chunks(plotfile, level, ndim)]


def read_level_state(plotfile: Path, names: list[str], ndim: int, level: int) -> np.ndarray:
   resolved, kind, _ = _resolve_plotfile(plotfile)
   if kind == "amrex":
      chunks = level_chunks(resolved, level, ndim)
      state = _allocate_state(chunks, ndim, names)
      for lo, hi, data in chunks:
         slices = tuple(slice(lo[d], hi[d] + 1) for d in range(ndim))
         state[(slice(None), *slices)] = data
      return state
   if h5py is None:
      raise RuntimeError("h5py not installed; cannot read AMReX HDF5 plotfiles")
   ncomp = len(names)
   with h5py.File(resolved, "r") as f:
      nlevels = int(f.attrs["num_levels"])
      if not (0 <= level < nlevels):
         raise ValueError(f"level {level} not in range [0, {nlevels - 1}]")
   chunks = level_chunks(resolved, level, ndim, names=names)
   state = _allocate_state(chunks, ndim, ncomp)
   for lo, hi, data in chunks:
      slices = tuple(slice(lo[d], hi[d] + 1) for d in range(ndim))
      state[(slice(None), *slices)] = data
   return state


def _allocate_state(
   chunks: list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]],
   ndim: int,
   ncomp: int | list[str],
) -> np.ndarray:
   shape = full_shape(chunks, ndim)
   ncomponents = len(ncomp) if isinstance(ncomp, list) else ncomp
   return np.full((ncomponents, *shape), np.nan, dtype=float)


def full_shape(chunks: list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]], ndim: int) -> tuple[int, ...]:
   return tuple(max(chunk[1][d] for chunk in chunks) + 1 for d in range(ndim))


def parse_plot_header_1d(plotfile: Path) -> tuple[list[str], float]:
   names, ndim, time_value = parse_plot_header(plotfile)
   if ndim != 1:
      raise ValueError(f"plotfile is {ndim}D, not 1D: {plotfile}")
   return names, time_value


def read_level0_state_1d(plotfile: Path, names: list[str]) -> np.ndarray:
   return read_level_state(plotfile, names, 1, 0)


def read_fab_1d(plotfile: Path, offset: int) -> tuple[int, int, np.ndarray]:
   path = _ensure_path(plotfile)
   if path.suffix in {".h5", ".hdf5"}:
      raise ValueError(f"hdf5 plotfile does not expose FabOnDisk index offsets: {path}")
   with path.open("rb") as f:
      f.seek(offset)
      header = f.readline().decode("ascii")
      match = FAB_HEADER_1D.search(header)
      if not match:
         raise ValueError(f"could not parse 1D FAB header in {path} at offset {offset}")
      lo = int(match.group(1))
      hi = int(match.group(2))
      ncomp = int(match.group(4))
      ncell = hi - lo + 1
      data = np.fromfile(f, dtype="<f8", count=ncell * ncomp)
   return lo, hi, data.reshape((ncomp, ncell))
