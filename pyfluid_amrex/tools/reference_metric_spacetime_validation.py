"""Regular-chart evolution controls selected by validate_z4c.py.

Chart propagation and physical symmetry are separate controls. Neither implies
coordinate-singularity regularity, conservative AMR or magnetic evolution.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from plotfile_reader import (
    parse_level_cell_sizes, parse_plot_bounds, parse_plot_header, read_level_state,
)
from spacetime_validation import (
    GAUGE_WAVE_INPUT, gauge_wave_exact, latest_fluid_plotfile,
    latest_spacetime_plotfile, run_dimension_case,
)
from validation_common import OPEN_POLICY, ROOT, repo_relative


def validate_spherical_gauge_wave(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Propagate a Cartesian plane wave through a regular spherical chart.

    This deliberately opt-in three-grid control tests every conformal field.
    The fixed physical interior excludes open radial/polar boundaries, and
    the full azimuth is periodic. It does not test origin or pole regularity.
    """
    if args.evolution_formulation != "fccz4":
        raise ValueError("spherical_gauge_wave requires --evolution-formulation fccz4")
    results = []
    scalars = []
    previous = {}
    resolutions = (16, 32, 64)
    for cells in resolutions:
        run_dir, scalar = run_dimension_case(
            f"spherical_gauge_n{cells}", GAUGE_WAVE_INPUT, {
                "mesh.coordinate": "spherical",
                "mesh.axes": "r theta phi",
                "geometry.prob_lo": "2 0.7 0",
                "geometry.prob_hi": f"3 2.1 {2.0 * math.pi}",
                "geometry.is_periodic": "0 0 1",
                "geometry.coord_sys": "0",
                "amr.n_cell": f"{cells} {cells} {cells}",
                "amr.max_grid_size": str(min(cells, 32)),
                "spacetime.initial_data": "gauge_wave",
                "spacetime.lapse_gauge": "harmonic",
                "spacetime.shift_gauge": "frozen",
                "spacetime.amplitude": "0.01",
                "spacetime.wave_number": "0.1",
                "spacetime.ko_dissipation": "0",
                "pyfluid.timestep": "rk4",
                "pyfluid.cfl": "0.25",
                "pyfluid.final_time": "0.01",
                "pyfluid.max_step": "100000",
                "pyfluid.bc": "outflow",
                "pyfluid.bc_zlo": "periodic",
                "pyfluid.bc_zhi": "periodic",
                "pyfluid.plot_int": "1000000",
                "pyfluid.scalar_int": "1",
                "pyfluid.debug_check_finite": "1",
                "pyfluid.debug_check_physical": "1",
            }, args, dim=3, checks=OPEN_POLICY, mhd=False,
        )
        plotfile = latest_spacetime_plotfile(run_dir)
        if plotfile is None:
            raise RuntimeError(f"missing final spacetime output: {run_dir}")
        names, dim, time = parse_plot_header(plotfile)
        if abs(time - 0.01) > 1.0e-12:
            raise RuntimeError(f"spherical gauge wave snapshot has wrong time: {time}")
        state = read_level_state(plotfile, names, dim, 0)
        lower, upper = parse_plot_bounds(plotfile)
        spacing = parse_level_cell_sizes(plotfile)[0]
        coordinates = [
            lower[axis] + np.arange(state.shape[axis + 1]) * spacing[axis]
            for axis in range(3)
        ]
        radius, theta, phi = np.meshgrid(*coordinates, indexing="ij")
        normal = (
            np.sin(theta) * np.cos(phi),
            np.cos(theta) * np.cos(phi),
            -np.sin(phi),
        )
        exact = gauge_wave_exact(
            radius * normal[0], time, amplitude=0.01, wave_number=0.1,
            normal=normal,
        )
        mask = np.ones(radius.shape, dtype=bool)
        for coordinate, low, high in zip((radius, theta), lower[:2], upper[:2]):
            margin = 0.375 * (high - low)
            mask &= (coordinate >= low + margin) & (coordinate <= high - margin)
        for name, expected in exact.items():
            difference = state[names.index(name)][mask] - expected[mask]
            error = float(np.sqrt(np.mean(difference**2)))
            order = (
                math.log2(previous[name] / error)
                if name in previous and previous[name] > 0.0 and error > 0.0
                else ""
            )
            # RK4 bounds the full evolution order even with higher spatial
            # stencils. Do not infer an order from errors at the roundoff floor.
            expected_order = min(getattr(args, "derivative_order", 4), 4)
            roundoff_limited = error <= 1.0e-12
            passed = math.isfinite(error)
            if name.startswith("shift"):
                passed = passed and error == 0.0
            elif cells == resolutions[-1]:
                passed = passed and (roundoff_limited or (order != "" and order >= expected_order - 0.5))
            results.append({
                "kind": "convergence",
                "case": f"spherical_gauge_n{cells}",
                "status": "PASS" if passed else "FAIL",
                "metric": f"{name}_l2",
                "value": error,
                "order": order,
                "roundoff_limited": roundoff_limited,
                "run_dir": repo_relative(run_dir),
            })
            previous[name] = error
        scalars.append(scalar)
    return results, scalars


def validate_spherical_symmetry(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compare one coupled radial TOV-shell step in all spherical layouts."""
    if args.evolution_formulation != "fccz4":
        raise ValueError("spherical_symmetry requires --evolution-formulation fccz4")
    common = {
        "spacetime.evolution": "evolved",
        "spacetime.initial_data": "minkowski",
        "spacetime.matter_coupling": "coupled",
        "spacetime.lapse_gauge": "frozen",
        "spacetime.shift_gauge": "frozen",
        "spacetime.ko_dissipation": "0",
        "mesh.coordinate": "spherical",
        "pyfluid.ic": "tov",
        "pyfluid.eos": "hybrid_polytrope",
        "pyfluid.polytropic_k": "100",
        "pyfluid.polytropic_gamma": "2",
        "pyfluid.thermal_gamma": "2",
        "pyfluid.tov_central_density": "0.00128",
        "pyfluid.rho_floor": "1e-12",
        "pyfluid.pressure_floor": "1e-16",
        "pyfluid.recon": "pwl",
        "pyfluid.bc": "outflow",
        "pyfluid.ng": "3",
        "pyfluid.timestep": "rk4",
        "pyfluid.cfl": "0.25",
        "pyfluid.final_time": "0.02",
        "pyfluid.max_step": "100000",
        "pyfluid.plot_int": "1000000",
        "pyfluid.scalar_int": "1",
        "pyfluid.debug_check_finite": "1",
        "pyfluid.debug_check_physical": "1",
        "amr.max_grid_size": "32",
    }
    layouts = (
        ("r", "1", "4", "32", "0"),
        ("r theta", "1 0.7", "4 2.1", "32 16", "0 0"),
        ("r phi", "1 0", f"4 {2 * math.pi}", "32 16", "0 1"),
        ("r theta phi", "1 0.7 0", f"4 2.1 {2 * math.pi}", "32 16 16", "0 0 1"),
    )
    results = []
    scalars = []
    references = {}
    base_input = ROOT / "pyfluid_amrex/inputs/rhd_spherical_uniform_static_1d.in"
    for axes, lower, upper, cells, periodic in layouts:
        dim = len(axes.split())
        parameters = {
            **common,
            "mesh.axes": axes,
            "geometry.coord_sys": "2" if axes in ("r", "r theta") else "0",
            "geometry.prob_lo": lower,
            "geometry.prob_hi": upper,
            "geometry.is_periodic": periodic,
            "amr.n_cell": cells,
        }
        if "phi" in axes:
            boundary_axis = "y" if dim == 2 else "z"
            for side in ("lo", "hi"):
                parameters[f"pyfluid.bc_{boundary_axis}{side}"] = "periodic"
        name = "spherical_symmetry_" + axes.replace(" ", "_")
        run_dir, scalar = run_dimension_case(
            name, base_input, parameters, args, dim=dim, checks=OPEN_POLICY, mhd=False,
        )
        for group, find_plotfile in (
            ("spacetime", latest_spacetime_plotfile), ("fluid", latest_fluid_plotfile),
        ):
            plotfile = find_plotfile(run_dir)
            if plotfile is None:
                raise RuntimeError(f"missing {group} snapshot: {run_dir}")
            names, dimension, time = parse_plot_header(plotfile)
            if dimension != dim or abs(time - 0.02) > 1e-12:
                raise RuntimeError(f"wrong dimension/time in {plotfile}")
            state = read_level_state(plotfile, names, dim, 0)
            # Stored components use runtime order. Swap both tensor indices for
            # r-phi and restore canonical symmetric-index order (32 -> 23).
            if axes == "r phi":
                names = [
                    name.translate(str.maketrans("23", "32")).replace("32", "23")
                    for name in names
                ]
            if dim == 1:
                references[group] = dict(zip(names, state, strict=True))
            errors = []
            for field, values in zip(names, state, strict=True):
                reference = references[group][field]
                reference = reference.reshape(reference.shape + (1,) * (dim - 1))
                errors.append(float(np.max(np.abs(values - reference))))
            error = max(errors)
            results.append({
                "kind": "symmetry",
                "case": name,
                "status": "PASS" if math.isfinite(error) and error <= 1e-10 else "FAIL",
                "metric": f"{group}_projection_linf",
                "value": error,
                "run_dir": repo_relative(run_dir),
            })
        scalars.append(scalar)
    return results, scalars
