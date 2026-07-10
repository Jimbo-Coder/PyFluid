#!/usr/bin/env python3
"""Plot the advective CFL limiter location recorded in scalars.tsv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_params(path: Path) -> dict[str, str]:
   params: dict[str, str] = {}
   if not path.exists():
      return params
   with path.open() as f:
      for raw in f:
         line = raw.strip()
         if not line or line.startswith("#") or "=" not in line:
            continue
         key, value = line.split("=", 1)
         params[key.strip()] = value.strip()
   return params


def read_rows(path: Path) -> list[dict[str, float]]:
   rows: list[dict[str, float]] = []
   with path.open(newline="") as f:
      reader = csv.DictReader(f, delimiter="\t")
      for raw in reader:
         try:
            row = {key: float(value) for key, value in raw.items() if value not in (None, "")}
         except ValueError:
            continue
         if row.get("cfl_limiter_level", -1.0) >= 0.0:
            rows.append(row)
   if not rows:
      raise RuntimeError(f"{path} has no active cfl_limiter_* rows; enable pyfluid.cfl_limiter_diagnostics=1")
   return rows


def default_output(run_dir: Path) -> Path:
   return run_dir / f"{run_dir.name}_cfl_limiter.png"


def canonical_to_cartesian(coordinate: str, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   if coordinate == "cylindrical":
      return q1 * np.cos(q2), q1 * np.sin(q2), q3
   if coordinate == "spherical":
      sin_theta = np.sin(q2)
      return q1 * sin_theta * np.cos(q3), q1 * sin_theta * np.sin(q3), q1 * np.cos(q2)
   return q1, q2, q3


def main() -> None:
   parser = argparse.ArgumentParser()
   parser.add_argument("run_dir", type=Path, help="Run directory containing scalars.tsv.")
   parser.add_argument("--output", type=Path, help="Output PNG path. Defaults to a stable run-local plot.")
   parser.add_argument("--title", default="", help="Optional plot title.")
   args = parser.parse_args()

   run_dir = args.run_dir.resolve()
   rows = read_rows(run_dir / "scalars.tsv")
   params = read_params(run_dir / "run.in")
   coordinate = params.get("mesh.coordinate", "cartesian").strip().lower()
   output = args.output.resolve() if args.output else default_output(run_dir)
   output.parent.mkdir(parents=True, exist_ok=True)

   time = np.asarray([row["time"] for row in rows])
   level = np.asarray([row["cfl_limiter_level"] for row in rows])
   q1 = np.asarray([row["cfl_limiter_q1"] for row in rows])
   q2 = np.asarray([row["cfl_limiter_q2"] for row in rows])
   q3 = np.asarray([row["cfl_limiter_q3"] for row in rows])
   if "cfl_limiter_x" in rows[0]:
      x = np.asarray([row["cfl_limiter_x"] for row in rows])
      y = np.asarray([row["cfl_limiter_y"] for row in rows])
      z = np.asarray([row["cfl_limiter_z"] for row in rows])
   else:
      x, y, z = canonical_to_cartesian(coordinate, q1, q2, q3)
   r_cyl = np.sqrt(x * x + y * y)
   r_sph = np.sqrt(x * x + y * y + z * z)
   freq = np.asarray([row["cfl_limiter_frequency"] for row in rows])
   dt = np.asarray([row["dt"] for row in rows])

   fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
   title = args.title or f"{run_dir.name}: advective CFL limiter"
   fig.suptitle(title)

   scatter = axs[0].scatter(x, y, c=time, s=42, cmap="viridis", edgecolor="black", linewidth=0.4)
   axs[0].plot(x, y, color="0.35", linewidth=1.0, alpha=0.7)
   axs[0].scatter([x[0]], [y[0]], marker="s", s=64, color="tab:green", label="first")
   axs[0].scatter([x[-1]], [y[-1]], marker="*", s=100, color="tab:red", label="last")
   axs[0].set_aspect("equal", adjustable="box")
   axs[0].set_xlabel("x")
   axs[0].set_ylabel("y")
   axs[0].set_title("Cartesian Track")
   axs[0].legend()
   fig.colorbar(scatter, ax=axs[0], label="time")

   axs[1].plot(time, r_cyl, marker="o", label="cyl radius", linewidth=2.0)
   axs[1].plot(time, r_sph, marker="s", label="sph radius", linewidth=1.6, linestyle=":")
   if np.any(np.abs(z) > 0.0):
      axs[1].plot(time, z, marker="^", label="z", linewidth=1.4, linestyle="--")
   axs[1].set_xlabel("time")
   axs[1].set_ylabel("location")
   axs[1].set_title("Limiter Radius")
   axs[1].legend()

   axs[2].semilogy(time, np.maximum(freq, 1.0e-300), marker="o", label="signal frequency", linewidth=2.0)
   axs[2].semilogy(time, np.maximum(dt, 1.0e-300), marker="s", label="dt", linewidth=1.6, linestyle=":")
   ax_level = axs[2].twinx()
   ax_level.plot(time, level, marker="^", color="tab:green", label="AMR level", linewidth=1.3, linestyle="--")
   axs[2].set_xlabel("time")
   axs[2].set_ylabel("frequency / dt")
   ax_level.set_ylabel("AMR level")
   axs[2].set_title("Limiter Strength")
   lines, labels = axs[2].get_legend_handles_labels()
   lines2, labels2 = ax_level.get_legend_handles_labels()
   axs[2].legend(lines + lines2, labels + labels2, loc="best")

   fig.savefig(output, dpi=180)
   print(output)


if __name__ == "__main__":
   main()
