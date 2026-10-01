#!/usr/bin/env python3
"""Opt-in matched-time BH excision controls, separate from the short gate.

The domain pair has identical dx; the resolution ladder has identical extent.
Global exterior L2 norms on different domains are not comparable error norms.
Runtime exterior masks also change with dx: they omit two outer node layers
and require every axis-aligned +/-2-node stencil to lie outside the analytic
coordinate horizon. The opt-in fixed-region diagnostics instead use one
coordinate box, exclusion sphere and radial split at every resolution.
Completed evolution and scalar closure do not by themselves establish accuracy.
"""

from __future__ import annotations

import argparse
import math
import time

from validate_black_hole import INTERIOR_RECOVERY, RECOVERY
from validation_common import (
    OPEN_POLICY,
    ROOT,
    append_validation_history,
    check_external_conservation,
    materialize_run_input,
    read_tsv,
    repo_relative,
    run_amrex,
    scalar_validation_row,
    write_tsv,
)


RUN_ROOT = ROOT / "runs/current/black_hole_excision_convergence"
BASE_INPUT = ROOT / "pyfluid_amrex/inputs/kerr_schild_coupled_excision_3d.in"
LAYOUTS = {
    "extent4_n32": (4.0, 32),
    "extent4_n64": (4.0, 64),
    "extent6_n48": (6.0, 48),
    "extent6_n96": (6.0, 96),
    "extent4_n96": (4.0, 96),
}

# Physical coordinate lengths for this M=1 Kerr control, fixed before the scan.
# The coarsest h=0.25 stencil stays inside the R=4 physical box and outside
# the horizon throughout these regions. Reflection leaves the z>=0 half only.
FIXED_REGION_PARAMETERS = {
    "diagnostics.constraint_region_extent": "3.5",
    "diagnostics.constraint_exclusion_radius": "2.5",
    "diagnostics.constraint_split_radius": "3.0",
}


def fixed_region_measurements(rows: list[dict]) -> dict:
    """Validate and extract runtime-owned norms, without recomputing physics."""
    if not rows:
        raise ValueError("missing fixed-region scalar rows")
    result = {
        "fixed_region_geometry": "box_minus_sphere_split_at_radius",
        "fixed_region_domain": "intersection_with_physical_half_domain",
        "fixed_region_weighting": "coordinate_volume_nodal",
        **FIXED_REGION_PARAMETERS,
    }
    for row in rows:
        values = {}
        for region in ("region", "near", "far"):
            for column in (
                f"z4c_hamiltonian_{region}_l2",
                f"z4c_hamiltonian_{region}_linf",
                f"z4c_momentum_{region}_linf",
                f"z4c_{region}_volume",
            ):
                value = float(row[column])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid fixed-region diagnostic {column}={value}")
                values[column] = value
            if values[f"z4c_{region}_volume"] <= 0:
                raise ValueError(f"empty fixed diagnostic {region}")
            # These controls use a fixed, uniform hierarchy. The geometric
            # sampling measure must not drift as the spacetime evolves.
            if not math.isclose(
                values[f"z4c_{region}_volume"],
                float(rows[0][f"z4c_{region}_volume"]), rel_tol=2e-12,
            ):
                raise ValueError(f"fixed diagnostic {region} volume changed in time")
        volume = values["z4c_region_volume"]
        parts_volume = values["z4c_near_volume"] + values["z4c_far_volume"]
        if not math.isclose(volume, parts_volume, rel_tol=2e-12):
            raise ValueError("fixed-region volumes do not partition the whole region")
        squared_integral = values["z4c_hamiltonian_region_l2"]**2 * volume
        parts_integral = sum(
            values[f"z4c_hamiltonian_{region}_l2"]**2
            * values[f"z4c_{region}_volume"] for region in ("near", "far")
        )
        if not math.isclose(squared_integral, parts_integral, rel_tol=2e-12, abs_tol=1e-28):
            raise ValueError("fixed-region Hamiltonian integrals do not partition")
        for field in ("hamiltonian", "momentum"):
            if not math.isclose(
                values[f"z4c_{field}_region_linf"],
                max(values[f"z4c_{field}_near_linf"], values[f"z4c_{field}_far_linf"]),
                rel_tol=2e-12, abs_tol=1e-28,
            ):
                raise ValueError(f"fixed-region {field} maxima do not partition")
    for stage, row in (("initial", rows[0]), ("final", rows[-1])):
        for region in ("region", "near", "far"):
            for column in (
                f"z4c_hamiltonian_{region}_l2",
                f"z4c_hamiltonian_{region}_linf",
                f"z4c_momentum_{region}_linf",
                f"z4c_{region}_volume",
            ):
                result[f"{column}_{stage}"] = float(row[column])
    return result


def plot_fixed_regions(results: list[dict], output) -> None:
    """Plot runtime near/far norms; no constraint formula lives here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    passed = [result for result in results if result["status"] == "PASS"]
    spacings = sorted({float(result["dx"]) for result in passed}, reverse=True)
    extents = sorted({float(result["extent"]) for result in passed})
    for result in passed:
        rows = read_tsv(ROOT / result["run_dir"] / "scalars.tsv", require_complete=True)
        fixed_region_measurements(rows)
        label = f"box R={float(result['extent']):g}M, dx={float(result['dx']):g}M"
        color = f"C{spacings.index(float(result['dx']))}"
        style = ("-", "--", ":", "-.")[extents.index(float(result["extent"])) % 4]
        for axis, region in zip(axes, ("near", "far")):
            axis.semilogy(
                [float(row["time"]) for row in rows],
                [float(row[f"z4c_hamiltonian_{region}_l2"]) for row in rows],
                label=label, linewidth=1.7, color=color, linestyle=style,
            )
    axes[0].set_title("Near BH: Euclidean r in [2.5M, 3M)")
    axes[1].set_title("Outer region: Euclidean r >= 3M")
    for axis in axes:
        axis.set_xlabel("t / M")
        axis.set_ylabel("Hamiltonian constraint RMS")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Kerr spin 0.5, CFL 0.9: identical physical measurement regions\n"
        "Both regions inside |x|,|y|,|z| <= 3.5M, z >= 0; coordinate-volume nodal norms",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_control(name: str, *, final_time: float, ranks: int, timeout: int,
                run_root=RUN_ROOT, outer_boundary="sommerfeld", sommerfeld_power=3,
                fixed_regions=False) -> dict:
    extent, cells = LAYOUTS[name]
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        BASE_INPUT, run_dir / "input.in", {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "geometry.prob_lo": f"{-extent} {-extent} 0",
            "geometry.prob_hi": f"{extent} {extent} {extent}",
            "amr.n_cell": f"{cells} {cells} {cells // 2}",
            "amr.max_grid_size": "32",
            "spacetime.matter_coupling": "vacuum",
            "spacetime.outer_boundary": outer_boundary,
            "spacetime.sommerfeld_power": str(sommerfeld_power),
            "pyfluid.rho0": "1e-8",
            "pyfluid.pressure0": "1e-10",
            "pyfluid.bc": "diode",
            "pyfluid.final_time": str(final_time),
            "pyfluid.max_step": "10000",
            "horizon.find_int": "4",
            "pyfluid.scalar_diagnostics": "standard,horizon,constraints"
            + (",constraint_regions" if fixed_regions else ""),
            "pyfluid.plot_int": "-1",
            **(FIXED_REGION_PARAMETERS if fixed_regions else {}),
        },
    )
    started = time.perf_counter()
    completed = run_amrex(dim=3, input_path=input_path, ranks=ranks, timeout=timeout)
    (run_dir / "run.log").write_text(completed.stdout)
    result = {
        "case": name, "status": "FAIL", "extent": extent,
        "outer_boundary": outer_boundary,
        "sommerfeld_power": sommerfeld_power,
        "dx": 2 * extent / cells, "target_time": final_time,
        "constraint_region": "runtime_stencil_safe_bh_exterior",
        "constraint_region_dx_dependent": 1,
        "outer_constraint_margin": 4 * extent / cells,
        "horizon_exclusion_stencil_radius_cells": 2,
        "constraint_l2_weighting": "coordinate_volume",
        "seconds": time.perf_counter() - started,
        "run_dir": repo_relative(run_dir),
    }
    if completed.returncode:
        result["reason"] = f"solver exit {completed.returncode}; see run.log"
        return result
    scalar = scalar_validation_row(
        run_dir, case=name, input_file=run_dir / "run.in", checks=OPEN_POLICY,
        mhd=False, dim=3, ranks=ranks, kind="black_hole_excision_convergence",
    )
    append_validation_history(ROOT / "runs/current/validation_history.tsv", [scalar])
    rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
    horizons = read_tsv(run_dir / "apparent_horizon.tsv", require_complete=True)
    first, last = rows[0], rows[-1]
    if fixed_regions:
        try:
            result.update(fixed_region_measurements(rows))
        except (KeyError, ValueError) as error:
            result.update(status="FAIL", reason=str(error))
            return result
    initial_horizon, final_horizon = horizons[0], horizons[-1]
    exact_area = 8 * math.pi * (1 + math.sqrt(1 - 0.5**2))
    result.update({
        "status": scalar["status"], "reason": scalar["reason"],
        "time": float(last["time"]), "steps": int(last["step"]),
        "horizon_time": float(final_horizon["time"]),
        "area_initial_rel_error": float(initial_horizon["area"]) / exact_area - 1,
        "area_final_rel_error": float(final_horizon["area"]) / exact_area - 1,
        "area_rel_drift": float(final_horizon["area"]) / float(initial_horizon["area"]) - 1,
        "mass_initial": initial_horizon["coordinate_christodoulou_mass"],
        "mass_final": final_horizon["coordinate_christodoulou_mass"],
        "spin_initial": initial_horizon["coordinate_christodoulou_dimensionless_spin3"],
        "spin_final": final_horizon["coordinate_christodoulou_dimensionless_spin3"],
        "hamiltonian_exterior_l2_initial": first["z4c_hamiltonian_bh_exterior_l2"],
        "hamiltonian_exterior_l2_final": last["z4c_hamiltonian_bh_exterior_l2"],
        "hamiltonian_exterior_linf_initial": first["z4c_hamiltonian_bh_exterior_linf"],
        "hamiltonian_exterior_linf_final": last["z4c_hamiltonian_bh_exterior_linf"],
        "momentum_exterior_linf_final": last["z4c_momentum_bh_exterior_linf"],
        "boundary_mass_change": last["boundary_mass_change"],
        "external_mass_rel_error": scalar["external_mass_rel_error"],
    })
    recovery = RECOVERY.findall(completed.stdout)
    interior = INTERIOR_RECOVERY.findall(completed.stdout)
    counters = (*map(int, recovery[-1]), *map(int, interior[-1])) \
        if recovery and interior else (-1, -1, -1, -1)
    for key, value in zip(("c2p_fallbacks", "atmosphere_resets",
                           "interior_bounds", "interior_resets"), counters):
        result[key] = value
    if any(value != 0 for value in counters):
        result.update(status="FAIL", reason="missing or nonzero recovery/interior counters")
    ledger_failures = check_external_conservation(rows, energy=False)
    if any(float(row["external_mass_budget_valid"]) != 1.0 for row in rows):
        ledger_failures.append("missing evolving-geometry baryon ledger")
    if ledger_failures:
        result.update(status="FAIL", reason="; ".join(ledger_failures))
    if abs(result["time"] - final_time) > 1e-10:
        result.update(status="FAIL", reason="did not reach matched physical time")
    if abs(result["horizon_time"] - final_time) > 1e-10:
        result.update(status="FAIL", reason="final horizon is not at matched time")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=tuple(LAYOUTS), default=tuple(LAYOUTS))
    parser.add_argument("--final-time", type=float, default=1.2)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--label", default="")
    parser.add_argument("--outer-boundary",
                        choices=("extrapolate", "sommerfeld"),
                        default="sommerfeld")
    parser.add_argument("--sommerfeld-power", type=int, default=3,
                        help="non-wave RHS falloff; -1 selects the pure-wave control")
    parser.add_argument("--fixed-regions", action="store_true",
                        help="measure fixed near-BH and outer coordinate regions")
    parser.add_argument("--plot", action="store_true",
                        help="plot the fixed-region runtime norm histories")
    args = parser.parse_args()
    if args.plot and not args.fixed_regions:
        parser.error("--plot requires --fixed-regions")
    if args.final_time <= 0:
        parser.error("--final-time must be positive")
    if args.label and not all(char.isalnum() or char in "_-" for char in args.label):
        parser.error("--label must contain only letters, numbers, underscores, or hyphens")
    run_root = RUN_ROOT / args.label if args.label else RUN_ROOT
    if args.outer_boundary != "sommerfeld":
        run_root /= args.outer_boundary
    if args.fixed_regions:
        run_root /= "fixed_regions"
    results = []
    summary = run_root / ("summary_" + "_".join(args.cases) + ".tsv")
    summary_rows = [{"case": name, "status": "INCOMPLETE"} for name in args.cases]
    write_tsv(summary, summary_rows, ["case", "status"])
    for case_index, name in enumerate(args.cases):
        print(f"START {name}", flush=True)
        row = run_control(name, final_time=args.final_time, ranks=args.ranks,
                          timeout=args.timeout, run_root=run_root,
                          outer_boundary=args.outer_boundary,
                          sommerfeld_power=args.sommerfeld_power,
                          fixed_regions=args.fixed_regions)
        results.append(row)
        summary_rows[case_index] = row
        fields = list(dict.fromkeys(key for result in summary_rows for key in result))
        write_tsv(summary, summary_rows, fields)
        print(f"{row['status']} {name}: {row}", flush=True)
    print(f"summary: {repo_relative(summary)}")
    print("PASS means completion/scalar sanity, not demonstrated long-time convergence.")
    print("Constraint norms use dx-dependent runtime exterior masks, not a common fixed "
          "physical region: two outer node layers are omitted and each axis +/-2-node "
          "stencil must remain outside the analytic coordinate horizon. L2 uses "
          "coordinate-volume weighting; different-domain L2 values are not comparable.")
    if args.fixed_regions:
        print("Additional region/near/far norms use the same fixed box and radii in every "
              "case; their coordinate volumes and defining parameters are recorded.")
        if all(row["status"] == "PASS" for row in results):
            by_extent = {}
            for row in results:
                by_extent.setdefault(row["extent"], []).append(row)
            for extent, controls in by_extent.items():
                controls.sort(key=lambda row: row["dx"], reverse=True)
                for coarse, fine in zip(controls, controls[1:]):
                    if coarse["dx"] == fine["dx"]:
                        continue
                    for region in ("region", "near", "far"):
                        for stage in ("initial", "final"):
                            key = f"z4c_hamiltonian_{region}_l2_{stage}"
                            first_error, second_error = coarse[key], fine[key]
                            order = math.log(first_error / second_error) \
                                / math.log(coarse["dx"] / fine["dx"]) \
                                if first_error > 0 and second_error > 0 else float("nan")
                            print(f"R={extent:g} fixed {region} H_L2 {stage}: "
                                  f"{first_error:.8e} -> {second_error:.8e}; "
                                  f"observed order={order:.3f}")
            if args.plot:
                plot_path = ROOT / "plots/current/scans/black_hole_excision" \
                    / run_root.relative_to(RUN_ROOT) / "fixed_constraints.png"
                plot_fixed_regions(results, plot_path)
                print(f"plot: {repo_relative(plot_path)}")
    return int(any(row["status"] != "PASS" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
