#!/usr/bin/env python3
"""Plot oblique-shock scan errors from a validation TSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "runs" / "current" / "oblique_shock_sweep_summary_002.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "current" / "scans"


def read_rows(path: Path) -> list[dict[str, str]]:
   with path.open(newline="") as f:
      return list(csv.DictReader(f, delimiter="\t"))


def next_output_path(directory: Path, stem: str) -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   existing = sorted(directory.glob(f"{stem}_*.png"))
   if not existing:
      return directory / f"{stem}_001.png"
   last = existing[-1].stem.rsplit("_", 1)[-1]
   try:
      index = int(last) + 1
   except ValueError:
      index = 1
   return directory / f"{stem}_{index:03d}.png"


def numeric(row: dict[str, str], key: str) -> float:
   return float(row[key])


def plot_scan(rows: list[dict[str, str]], output: Path) -> None:
   resolution = sorted((row for row in rows if row["group"] == "resolution"), key=lambda row: numeric(row, "base_nx"))
   mach = sorted((row for row in rows if row["group"] == "mach"), key=lambda row: numeric(row, "mach"))
   if not resolution or not mach:
      raise ValueError("scan TSV must contain both resolution and mach groups")

   res_nx = np.array([numeric(row, "base_nx") for row in resolution])
   res_error = np.abs(np.array([numeric(row, "error_deg") for row in resolution]))
   mach_values = np.array([numeric(row, "mach") for row in mach])
   mach_error = np.abs(np.array([numeric(row, "error_deg") for row in mach]))

   fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
   fig.suptitle("Oblique shock AMR validation: wedge, theta=10 deg, gamma=1.4, PWC/HLLC/SSPRK3, CFL=0.9")

   ax = axes[0]
   ax.loglog(res_nx, res_error, marker="o", lw=2.4, label="absolute error")
   reference = res_error[0] * (res_nx[0] / res_nx)
   ax.loglog(res_nx, reference, "k:", lw=1.8, label="dx")
   for x, err, row in zip(res_nx, res_error, resolution):
      ax.annotate(f"{err:.2g} deg", (x, err), textcoords="offset points", xytext=(4, 5))
   ax.set_title("Mach 3 wedge angle error")
   ax.set_xlabel("base grid nx")
   ax.set_ylabel("|measured - analytic beta| [deg]")
   ax.grid(True, which="major", alpha=0.28)
   ax.legend()

   ax = axes[1]
   ax.semilogy(mach_values, mach_error, marker="s", lw=2.4, color="#ff7f0e")
   for x, err, row in zip(mach_values, mach_error, mach):
      ax.annotate(f"{err:.2g} deg", (x, err), textcoords="offset points", xytext=(4, 5))
   ax.set_title("Wedge shock angle error, base nx=192")
   ax.set_xlabel("Mach number")
   ax.set_ylabel("|measured - analytic beta| [deg]")
   ax.grid(True, which="major", alpha=0.28)

   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def main() -> None:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--output", type=Path)
   args = parser.parse_args()

   output = args.output
   if output is None:
      output = next_output_path(DEFAULT_OUTPUT_DIR, "oblique_shock_resolution_mach_scan")
   plot_scan(read_rows(args.input), output)
   print(f"wrote {output}")


if __name__ == "__main__":
   main()
