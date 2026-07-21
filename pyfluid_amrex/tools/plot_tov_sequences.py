#!/usr/bin/env python3
"""Plot one or more TOV sequence TSV files produced by the C++ solver."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser()
   parser.add_argument("sequences", nargs="+", type=Path)
   parser.add_argument("--output", type=Path, required=True)
   return parser.parse_args()


def sequence_metadata(path: Path) -> tuple[int, str]:
   with path.open(encoding="utf-8") as sequence_file:
      first_line = sequence_file.readline().strip()
   prefix = "# eos_label="
   if not first_line.startswith(prefix):
      return 0, path.stem.upper()
   label = first_line.removeprefix(prefix)
   if label.startswith("Gamma="):
      math_label = label.replace("Gamma", r"\Gamma").replace(", ", r",\ ")
      label = f"${math_label}$"
   return 1, label


def load_sequence(path: Path, skip_header: int) -> np.ndarray:
   data = np.genfromtxt(
      path, names=True, delimiter="\t", skip_header=skip_header
   )
   if data.size == 0:
      raise RuntimeError(f"empty TOV sequence: {path}")
   return np.atleast_1d(data)


def plot_branch(
   axis: plt.Axes,
   x: np.ndarray,
   y: np.ndarray,
   stable: np.ndarray,
   label: str,
) -> None:
   line, = axis.plot(x, np.where(stable, y, np.nan), label=label, linewidth=1.8)
   if np.any(~stable):
      axis.plot(
         x,
         np.where(~stable, y, np.nan),
         color=line.get_color(),
         linewidth=1.2,
         linestyle=":",
      )


def main() -> int:
   args = parse_args()
   figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), constrained_layout=True)
   for path in args.sequences:
      skip_header, label = sequence_metadata(path)
      data = load_sequence(path, skip_header)
      stable = data["stable_branch"].astype(bool)
      plot_branch(
         axes[0, 0],
         data["areal_radius"],
         data["gravitational_mass"],
         stable,
         label,
      )
      plot_branch(
         axes[0, 1],
         data["central_density"],
         data["gravitational_mass"],
         stable,
         label,
      )
      tidal = data["tidal_available"].astype(bool) & (data["tidal_deformability"] > 0.0)
      if np.any(tidal):
         plot_branch(
            axes[1, 0],
            data["gravitational_mass"][tidal],
            data["tidal_deformability"][tidal],
            stable[tidal],
            label,
         )
      plot_branch(
         axes[1, 1],
         data["gravitational_mass"],
         data["binding_energy"] / data["baryonic_mass"],
         stable,
         label,
      )

   axes[0, 0].set(xlabel=r"$R$", ylabel=r"$M$")
   axes[0, 1].set(xlabel=r"$\rho_c$", ylabel=r"$M$", xscale="log")
   axes[1, 0].set(xlabel=r"$M$", ylabel=r"$\Lambda$", yscale="log")
   axes[1, 1].set(xlabel=r"$M$", ylabel=r"$(M_b-M)/M_b$")
   for axis in axes.flat:
      axis.grid(alpha=0.2, linewidth=0.6)
      axis.legend(frameon=False, fontsize=8)
   args.output.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(args.output, dpi=220)
   plt.close(figure)
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
