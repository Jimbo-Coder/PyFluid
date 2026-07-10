#!/usr/bin/env python3
"""Plot Newtonian gravity energy bookkeeping from a scalars.tsv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, float]]:
   rows: list[dict[str, float]] = []
   with path.open(newline="") as f:
      reader = csv.DictReader(f, delimiter="\t")
      required = {"time", "energy", "gravity_potential_energy", "gravity_total_energy", "gravity_total_energy_rel_drift"}
      missing = required.difference(reader.fieldnames or [])
      if missing:
         raise ValueError(f"{path} is missing gravity energy columns: {', '.join(sorted(missing))}")
      for row in reader:
         try:
            rows.append({key: float(value) for key, value in row.items() if value not in ("", None)})
         except ValueError:
            continue
   if not rows:
      raise ValueError(f"no scalar rows found in {path}")
   return rows


def positive(values: np.ndarray) -> np.ndarray:
   return np.maximum(np.abs(values), 1.0e-18)


def main() -> None:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--input", required=True, type=Path)
   parser.add_argument("--output", required=True, type=Path)
   parser.add_argument("--title", default="Gravity energy")
   args = parser.parse_args()

   rows = read_rows(args.input)
   t = np.asarray([row["time"] for row in rows])
   fluid = np.asarray([row["energy"] for row in rows])
   potential = np.asarray([row["gravity_potential_energy"] for row in rows])
   total = np.asarray([row["gravity_total_energy"] for row in rows])
   total0 = total[0]
   relative_drift = np.abs(total - total0) / max(abs(total0), 1.0e-300)

   args.output.parent.mkdir(parents=True, exist_ok=True)
   fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
   fig.suptitle(args.title)

   axs[0].plot(t, fluid, label="fluid", linewidth=2.2)
   axs[0].plot(t, potential, label="potential", linewidth=2.2)
   axs[0].plot(t, total, label="fluid + potential", color="black", linewidth=2.6)
   axs[0].axhline(total0, color="0.45", linestyle=":", linewidth=1.8, label="initial total")
   axs[0].set_ylabel("energy")
   axs[0].grid(True, which="major", alpha=0.25)
   axs[0].legend()

   axs[1].plot(t, fluid - fluid[0], label="fluid change", linewidth=2.2)
   axs[1].plot(t, potential - potential[0], label="potential change", linewidth=2.2)
   axs[1].plot(t, total - total0, label="total change", color="black", linewidth=2.6)
   axs[1].axhline(0.0, color="0.45", linestyle=":", linewidth=1.8)
   axs[1].set_ylabel("change")
   axs[1].grid(True, which="major", alpha=0.25)
   axs[1].legend()

   axs[2].semilogy(t, positive(total - total0), label="|total - initial|", linewidth=2.2)
   axs[2].semilogy(t, positive(relative_drift), label="relative drift", linewidth=2.2)
   axs[2].set_xlabel("time")
   axs[2].set_ylabel("total-energy error")
   axs[2].grid(True, which="major", alpha=0.25)
   axs[2].legend()

   fig.savefig(args.output, dpi=180)
   plt.close(fig)


if __name__ == "__main__":
   main()
