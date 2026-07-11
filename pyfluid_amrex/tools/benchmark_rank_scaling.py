#!/usr/bin/env python3
"""Rank-scaling benchmark for stable PyFluid AMReX paths.

This is a comparison benchmark, not a pass/fail regression gate. It keeps a
small fixed set of well-behaved 2D hydro cases and runs each on MPI ranks
1, 2, 3, and 4. The output records throughput, simulation-time throughput,
speedup, and parallel efficiency so large infrastructure changes can be
compared against a stable local table.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from validation_common import materialize_run_input, parameters_from_assignments, run_mpi
from validation_common import append_performance_history as append_common_performance_history


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "hydro_sine_2d.in"
DEFAULT_RUN_ROOT = ROOT / "runs" / "current" / "pyfluid_amrex_rank_scaling"
DEFAULT_OUTPUT = DEFAULT_RUN_ROOT / "results.tsv"
DEFAULT_PLOT_OUTPUT = ROOT / "plots" / "current" / "scans" / "pyfluid_amrex_rank_scaling_001.png"
DEFAULT_HISTORY = ROOT / "runs" / "current" / "performance_history.tsv"

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
LAYOUT_AVERAGE_RE = re.compile(r"layout_average=([0-9.eE+-]+)\s+layout_Mcells/s")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")


@dataclass(frozen=True)
class ScalingCase:
   name: str
   label: str
   overrides: tuple[str, ...]


CASES = (
   ScalingCase(
      name="hydro2d_512_pwc_llf_euler",
      label="PWC/LLF/Euler",
      overrides=(
         "amr.n_cell=512 512",
         "amr.max_level=0",
         "amr.max_grid_size=64",
         "pyfluid.max_step=256",
         "pyfluid.ng=1",
         "pyfluid.eos=ideal_gas",
         "pyfluid.recon=pwc",
         "pyfluid.riemann=llf",
         "pyfluid.timestep=euler",
      ),
   ),
   ScalingCase(
      name="hydro2d_512_pwl_hllc_ssprk3",
      label="PWL/HLLC/SSPRK3",
      overrides=(
         "amr.n_cell=512 512",
         "amr.max_level=0",
         "amr.max_grid_size=64",
         "pyfluid.max_step=128",
         "pyfluid.ng=2",
         "pyfluid.eos=ideal_gas",
         "pyfluid.recon=pwl",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=ssprk3",
      ),
   ),
)


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def parse_ranks(text: str) -> list[int]:
   ranks = [int(item.strip()) for item in text.split(",") if item.strip()]
   if not ranks:
      raise ValueError("at least one rank count is required")
   return ranks


def mean(values: list[float]) -> float:
   return sum(values) / len(values) if values else float("nan")


def run_once(
   case: ScalingCase,
   executable: Path,
   input_file: Path,
   run_root: Path,
   ranks: int,
   repeat: int,
) -> dict[str, object]:
   run_dir = run_root / case.name / f"rank{ranks}_rep{repeat}"
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      input_file,
      run_dir / "run.in",
      parameters_from_assignments(
         [
            f"pyfluid.output_dir={repo_relative(run_dir)}",
            *case.overrides,
            "pyfluid.scalar_int=-1",
            "pyfluid.plot_int=-1",
            "pyfluid.checkpoint_int=-1",
            "pyfluid.aux_gfs=none",
            "pyfluid.aux_gf_int=-1",
            "pyfluid.aux_gf_compute_int=-1",
            "pyfluid.profile=off",
         ]
      ),
   )
   completed = run_mpi(
      executable,
      [input_path],
      ranks=ranks,
   )
   average_match = AVERAGE_RE.search(completed.stdout)
   layout_average_match = LAYOUT_AVERAGE_RE.search(completed.stdout)
   sim_time_match = SIM_TIME_RE.search(completed.stdout)
   return {
      "case": case.name,
      "label": case.label,
      "ranks": ranks,
      "repeat": repeat,
      "mcells_s": float(average_match.group(1)) if average_match else float("nan"),
      "layout_mcells_s": float(layout_average_match.group(1)) if layout_average_match else float("nan"),
      "sim_time_per_s": float(sim_time_match.group(1)) if sim_time_match else float("nan"),
      "returncode": completed.returncode,
      "run_dir": repo_relative(run_dir),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def summarize(raw_rows: list[dict[str, object]], run_root: Path) -> list[dict[str, object]]:
   grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
   labels: dict[str, str] = {}
   for row in raw_rows:
      case = str(row["case"])
      ranks = int(row["ranks"])
      labels[case] = str(row["label"])
      grouped.setdefault((case, ranks), []).append(row)

   baseline: dict[str, dict[str, float]] = {}
   for (case, ranks), rows in grouped.items():
      if ranks == 1:
         baseline[case] = {
            "mcells_s": mean([float(row["mcells_s"]) for row in rows]),
            "sim_time_per_s": mean([float(row["sim_time_per_s"]) for row in rows]),
         }

   summary: list[dict[str, object]] = []
   for case, ranks in sorted(grouped):
      rows = grouped[(case, ranks)]
      mcells_s = mean([float(row["mcells_s"]) for row in rows])
      layout_mcells_s = mean([float(row["layout_mcells_s"]) for row in rows])
      sim_time_per_s = mean([float(row["sim_time_per_s"]) for row in rows])
      base = baseline.get(case, {})
      speedup = mcells_s / base.get("mcells_s", float("nan"))
      sim_speedup = sim_time_per_s / base.get("sim_time_per_s", float("nan"))
      status = "pass" if all(int(row["returncode"]) == 0 for row in rows) else "fail"
      summary.append({
         "case": case,
         "label": labels.get(case, case),
         "status": status,
         "ranks": ranks,
         "repeats": len(rows),
         "mcells_s": mcells_s,
         "layout_mcells_s": layout_mcells_s,
         "sim_time_per_s": sim_time_per_s,
         "speedup_vs_1rank": speedup,
         "efficiency_vs_1rank": speedup / ranks,
         "sim_time_speedup_vs_1rank": sim_speedup,
         "sim_time_efficiency_vs_1rank": sim_speedup / ranks,
         "run_dir": repo_relative(run_root / case / f"rank{ranks}_rep0"),
      })
   return summary


def write_tsv(rows: list[dict[str, object]], output: Path, keys: list[str]) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   with output.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)


def append_scaling_history(rows: list[dict[str, object]], history: Path) -> None:
   append_common_performance_history(history, [{"kind": "rank_scaling", **row} for row in rows])


def write_plot(rows: list[dict[str, object]], output: Path) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   output.parent.mkdir(parents=True, exist_ok=True)
   cases = sorted({str(row["case"]) for row in rows})
   fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
   for case in cases:
      case_rows = sorted((row for row in rows if row["case"] == case), key=lambda row: int(row["ranks"]))
      ranks = [int(row["ranks"]) for row in case_rows]
      label = str(case_rows[0]["label"])
      axes[0].plot(ranks, [float(row["mcells_s"]) for row in case_rows], marker="o", label=label)
      axes[1].plot(ranks, [float(row["sim_time_per_s"]) for row in case_rows], marker="o", label=label)
      axes[2].plot(ranks, [float(row["efficiency_vs_1rank"]) for row in case_rows], marker="o", label=label)

   axes[0].set_title("Throughput")
   axes[0].set_ylabel("Mcells/s")
   axes[1].set_title("Evolution Rate")
   axes[1].set_ylabel("simulation time / second")
   axes[2].set_title("MPI Efficiency")
   axes[2].set_ylabel("speedup / ranks")
   axes[2].set_ylim(0.0, 1.05)
   for ax in axes:
      ax.set_xlabel("MPI ranks")
      ax.set_xticks([1, 2, 3, 4])
      ax.grid(True, which="major", alpha=0.25)
      ax.legend()
   fig.suptitle(
      "AMReX 2D rank scaling: IC=sin_wave, 512x512, BC=periodic, CFL=0.9, EOS=ideal_gas, AMR off"
   )
   fig.savefig(output, dpi=150)
   plt.close(fig)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
   parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
   parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
   parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
   parser.add_argument("--ranks", default="1,2,3,4")
   parser.add_argument("--repeats", type=int, default=1)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   if args.repeats < 1:
      raise SystemExit("--repeats must be positive")
   ranks = parse_ranks(args.ranks)
   if args.force and args.run_root.exists():
      shutil.rmtree(args.run_root)

   raw_rows: list[dict[str, object]] = []
   for case in CASES:
      for rank_count in ranks:
         for repeat in range(args.repeats):
            row = run_once(case, args.executable, args.input, args.run_root, rank_count, repeat)
            raw_rows.append(row)
            print(
               f"{row['label']} ranks={rank_count} rep={repeat}: "
               f"{float(row['mcells_s']):.2f} Mcells/s, "
               f"sim_time/s={float(row['sim_time_per_s']):.4g}, "
               f"returncode={row['returncode']}"
            )

   summary = summarize(raw_rows, args.run_root)
   write_tsv(
      summary,
      args.output,
      [
         "case",
         "label",
         "status",
         "ranks",
         "repeats",
         "mcells_s",
         "layout_mcells_s",
         "sim_time_per_s",
         "speedup_vs_1rank",
         "efficiency_vs_1rank",
         "sim_time_speedup_vs_1rank",
         "sim_time_efficiency_vs_1rank",
         "run_dir",
      ],
   )
   write_tsv(
      raw_rows,
      args.output.with_name("raw_results.tsv"),
      [
         "case",
         "label",
         "ranks",
         "repeat",
         "mcells_s",
         "layout_mcells_s",
         "sim_time_per_s",
         "returncode",
         "run_dir",
         "tail",
      ],
   )
   write_plot(summary, args.plot_output)
   append_scaling_history(summary, args.history)

   print(f"wrote {repo_relative(args.output)}")
   print(f"wrote {repo_relative(args.plot_output)}")
   print(f"appended {repo_relative(args.history)}")
   failed = [row for row in summary if row["status"] != "pass"]
   return 0 if not failed else 1


if __name__ == "__main__":
   raise SystemExit(main())
