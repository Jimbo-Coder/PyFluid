#!/usr/bin/env python3
"""Postprocess KH AMReX plotfiles into spectra and scalar diagnostics.

This samples the finest visible AMR data onto a regular 2D canvas before taking
FFTs. The spectra are therefore diagnostic plots, not a conservative AMR
spectral estimator.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amrex_boxed_movie import composite_slice, plotfile_step  # noqa: E402
from plotfile_reader import plotfile_timeline  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOT_ROOT = ROOT / "plots/current/IC/kelvin_helmholtz_mhd"


def rel(path: Path) -> str:
   absolute = path if path.is_absolute() else ROOT / path
   return str(absolute.relative_to(ROOT))


def next_output_path(directory: Path, stem: str, suffix: str = ".png") -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   existing = sorted(directory.glob(f"{stem}_*.png"))
   indices = []
   for path in existing:
      raw = path.stem.removeprefix(f"{stem}_")
      if raw.isdigit():
         indices.append(int(raw))
   index = max(indices, default=0) + 1
   return directory / f"{stem}_{index:03d}{suffix}"


def sampled_field(
   plotfile: Path,
   component: str,
   args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
   image, _boxes, time_value, _ndim, _names, _xlim, _ylim = composite_slice(
      plotfile=plotfile,
      component=component,
      gamma=args.gamma,
      ref_ratio=args.ref_ratio,
      z_fraction=args.z_fraction,
      log10=False,
      subtract_constant=None,
      max_pixels=args.max_pixels,
      xlim=tuple(args.xlim),
      ylim=tuple(args.ylim),
   )
   mean = float(np.nanmean(image))
   return np.nan_to_num(image, nan=mean), time_value


def vorticity3(vx: np.ndarray, vy: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:
   dx = (xlim[1] - xlim[0]) / vx.shape[1]
   dy = (ylim[1] - ylim[0]) / vx.shape[0]
   _dvy_dy, dvy_dx = np.gradient(vy, dy, dx, edge_order=2)
   dvx_dy, _dvx_dx = np.gradient(vx, dy, dx, edge_order=2)
   return dvy_dx - dvx_dy


def radial_spectrum(fields: list[np.ndarray], xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
   ny, nx = fields[0].shape
   dx = (xlim[1] - xlim[0]) / nx
   dy = (ylim[1] - ylim[0]) / ny
   kx = np.fft.fftfreq(nx, d=dx)
   ky = np.fft.fftfreq(ny, d=dy)
   kkx, kky = np.meshgrid(kx, ky)
   kmag = np.sqrt(kkx * kkx + kky * kky)
   power = np.zeros_like(kmag)
   norm = float(nx * ny) ** 2
   for field in fields:
      centered = field - np.mean(field)
      fft = np.fft.fft2(centered)
      power += 0.5 * np.abs(fft) ** 2 / norm
   bin_index = np.floor(kmag).astype(int)
   max_bin = int(bin_index.max())
   spectrum = np.bincount(bin_index.ravel(), weights=power.ravel(), minlength=max_bin + 1)
   counts = np.bincount(bin_index.ravel(), minlength=max_bin + 1)
   k = np.arange(max_bin + 1, dtype=float)
   valid = (k > 0) & (counts > 0) & (spectrum > 0.0)
   return k[valid], spectrum[valid]


def guide_power_law(k: np.ndarray, spectrum: np.ndarray, exponent: float, anchor_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
   if k.size == 0:
      return k, spectrum
   anchor = max(0, min(k.size - 1, int(anchor_fraction * k.size)))
   amplitude = spectrum[anchor] * (k[anchor] ** (-exponent))
   return k, amplitude * k**exponent


def read_scalar_rows(path: Path | None) -> list[dict[str, float]]:
   if path is None or not path.exists():
      return []
   rows = []
   with path.open(newline="") as f:
      for row in csv.DictReader(f, delimiter="\t"):
         converted = {}
         for key, value in row.items():
            try:
               converted[key] = float(value)
            except ValueError:
               converted[key] = np.nan
         rows.append(converted)
   return rows


def write_summary(rows: list[dict[str, float]], output: Path) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   fields = ["step", "time", "kinetic_energy", "enstrophy", "rms_vorticity"]
   with output.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def plot_diagnostics(
   k_energy: np.ndarray,
   energy_spectrum: np.ndarray,
   k_enstrophy: np.ndarray,
   enstrophy_spectrum: np.ndarray,
   rows: list[dict[str, float]],
   scalar_rows: list[dict[str, float]],
   output: Path,
   title: str,
) -> None:
   fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
   axs[0, 0].loglog(k_energy, energy_spectrum, marker="o", linewidth=2.0, label="kinetic")
   if k_energy.size > 4:
      gk, guide = guide_power_law(k_energy, energy_spectrum, -5.0 / 3.0)
      axs[0, 0].loglog(gk, guide, linestyle=":", color="black", linewidth=2.0, label="k^-5/3 guide")
   axs[0, 0].set_title("Kinetic Energy Spectrum")
   axs[0, 0].set_xlabel("k")
   axs[0, 0].set_ylabel("E(k)")
   axs[0, 0].legend()

   axs[0, 1].loglog(k_enstrophy, enstrophy_spectrum, marker="o", linewidth=2.0, label="vorticity")
   if k_enstrophy.size > 4:
      gk, guide = guide_power_law(k_enstrophy, enstrophy_spectrum, -3.0)
      axs[0, 1].loglog(gk, guide, linestyle=":", color="black", linewidth=2.0, label="k^-3 guide")
   axs[0, 1].set_title("Enstrophy Spectrum")
   axs[0, 1].set_xlabel("k")
   axs[0, 1].set_ylabel("Omega(k)")
   axs[0, 1].legend()

   t = np.array([row["time"] for row in rows])
   enstrophy = np.array([row["enstrophy"] for row in rows])
   rms = np.array([row["rms_vorticity"] for row in rows])
   axs[1, 0].plot(t, enstrophy, marker="o", linewidth=2.0, label="enstrophy")
   axs[1, 0].plot(t, rms, marker="s", linewidth=2.0, label="rms vorticity")
   axs[1, 0].set_title("Vorticity Measures")
   axs[1, 0].set_xlabel("time")
   axs[1, 0].legend()

   if scalar_rows:
      st = np.array([row["time"] for row in scalar_rows])
      mass = np.maximum(np.array([abs(row.get("mass_rel_drift", np.nan)) for row in scalar_rows]), 1.0e-18)
      energy = np.maximum(np.array([abs(row.get("energy_rel_drift", np.nan)) for row in scalar_rows]), 1.0e-18)
      axs[1, 1].semilogy(st, mass, marker="o", linewidth=2.0, label="mass")
      axs[1, 1].semilogy(st, energy, marker="s", linewidth=2.0, label="energy")
      if "divb_l1" in scalar_rows[0]:
         divb = np.maximum(np.array([abs(row.get("divb_l1", np.nan)) for row in scalar_rows]), 1.0e-18)
         axs[1, 1].semilogy(st, divb, marker="^", linewidth=2.0, label="divb")
      axs[1, 1].set_title("Scalar Drift")
      axs[1, 1].set_xlabel("time")
      axs[1, 1].legend()
      mcells = [row.get("mcells_per_s", np.nan) for row in scalar_rows if row.get("mcells_per_s", 0.0) > 0.0]
      sim_rates = [row.get("sim_time_per_s", np.nan) for row in scalar_rows if row.get("sim_time_per_s", 0.0) > 0.0]
      if mcells or sim_rates:
         perf_text = []
         if mcells:
            perf_text.append(f"avg {np.nanmean(mcells):.3g} Mcells/s")
         if sim_rates:
            perf_text.append(f"avg {np.nanmean(sim_rates):.3g} sim time/s")
         axs[1, 1].text(
            0.02,
            0.04,
            "\n".join(perf_text),
            transform=axs[1, 1].transAxes,
            fontsize=9,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
         )
   else:
      kinetic = np.array([row["kinetic_energy"] for row in rows])
      axs[1, 1].plot(t, kinetic, marker="o", linewidth=2.0)
      axs[1, 1].set_title("Sampled Kinetic Energy")
      axs[1, 1].set_xlabel("time")

   for ax in axs.ravel():
      ax.grid(True, which="major", alpha=0.25)
   fig.suptitle(title)
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=170)
   plt.close(fig)


def analyze(args: argparse.Namespace) -> tuple[Path, Path]:
   run_dir = args.run_dir
   plot_dir = args.plot_dir or run_dir
   scalar_path = args.scalars or run_dir / "scalars.tsv"
   timeline = plotfile_timeline(plot_dir, require=True, target_frames=args.max_frames)
   plotfiles = [sample.path for sample in timeline]

   rows: list[dict[str, float]] = []
   final_vx = final_vy = final_vorticity = None
   final_time = 0.0
   for plotfile in plotfiles:
      vx, time_value = sampled_field(plotfile, "v1", args)
      vy, _ = sampled_field(plotfile, "v2", args)
      omega = vorticity3(vx, vy, tuple(args.xlim), tuple(args.ylim))
      rows.append({
         "step": float(plotfile_step(plotfile)),
         "time": time_value,
         "kinetic_energy": float(0.5 * np.mean(vx * vx + vy * vy)),
         "enstrophy": float(0.5 * np.mean(omega * omega)),
         "rms_vorticity": float(np.sqrt(np.mean(omega * omega))),
      })
      final_vx = vx
      final_vy = vy
      final_vorticity = omega
      final_time = time_value

   assert final_vx is not None and final_vy is not None and final_vorticity is not None
   k_energy, energy_spectrum = radial_spectrum([final_vx, final_vy], tuple(args.xlim), tuple(args.ylim))
   k_enstrophy, enstrophy_spectrum = radial_spectrum([final_vorticity], tuple(args.xlim), tuple(args.ylim))

   output_plot = args.output or next_output_path(args.plot_root, "kh_spectrum")
   output_tsv = args.summary or run_dir / "kh_spectrum_summary.tsv"
   title = (
      f"{args.title}: t={final_time:.4g}, sampled {final_vx.shape[1]}x{final_vx.shape[0]}, "
      f"{len(plotfiles)} frame(s)"
   )
   write_summary(rows, output_tsv)
   plot_diagnostics(
      k_energy,
      energy_spectrum,
      k_enstrophy,
      enstrophy_spectrum,
      rows,
      read_scalar_rows(scalar_path),
      output_plot,
      title,
   )
   return output_plot, output_tsv


def main() -> None:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--run-dir", required=True, type=Path)
   parser.add_argument("--plot-dir", type=Path)
   parser.add_argument("--scalars", type=Path)
   parser.add_argument("--output", type=Path)
   parser.add_argument("--summary", type=Path)
   parser.add_argument("--plot-root", type=Path, default=DEFAULT_PLOT_ROOT)
   parser.add_argument("--title", default="KH MHD 2D spectral diagnostics")
   parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
   parser.add_argument("--ref-ratio", type=int, default=2)
   parser.add_argument("--z-fraction", type=float, default=0.5)
   parser.add_argument("--max-pixels", type=int, default=1024)
   parser.add_argument("--max-frames", type=int, default=32)
   parser.add_argument("--xlim", nargs=2, type=float, default=(0.0, 1.0))
   parser.add_argument("--ylim", nargs=2, type=float, default=(0.0, 1.0))
   args = parser.parse_args()

   plot_path, summary_path = analyze(args)
   print(f"wrote {rel(plot_path)}")
   print(f"wrote {rel(summary_path)}")


if __name__ == "__main__":
   main()
