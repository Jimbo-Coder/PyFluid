#!/usr/bin/env python3
"""Render a simulation-derived BH accretion octant cutaway movie."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np

from amr_plot_data import HierarchySlice, UniformSlice
from plotfile_reader import plotfile_timeline


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        lines = (line for line in stream if not line.startswith("#"))
        return [
            row
            for row in csv.DictReader(lines, delimiter="\t")
            if row and all(value not in (None, "") for value in row.values())
        ]


def read_parameters(path: Path) -> dict[str, str]:
    parameters: dict[str, str] = {}
    if not path.exists():
        return parameters
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parameters[key.strip()] = value.strip()
    return parameters


def latest_row(
    rows: list[dict[str, str]], step: int
) -> dict[str, str] | None:
    candidates = [row for row in rows if int(row["step"]) <= step]
    return candidates[-1] if candidates else None


def horizon_surface(run_dir: Path, step: int) -> tuple[np.ndarray, ...] | None:
    candidates = []
    for path in (run_dir / "apparent_horizon").glob("ah*.tsv"):
        try:
            surface_step = int(path.stem.removeprefix("ah"))
        except ValueError:
            continue
        if surface_step <= step:
            candidates.append((surface_step, path))
    if not candidates:
        return None
    _surface_step, path = max(candidates)
    surface = np.genfromtxt(path, names=True, delimiter="\t")
    theta = np.unique(surface["theta"])
    phi = np.unique(surface["phi"])
    shape = (theta.size, phi.size)
    return tuple(surface[name].reshape(shape) for name in ("x", "y", "z"))


def exact_kerr_horizon(
    parameters: dict[str, str],
) -> tuple[np.ndarray, ...] | None:
    if parameters.get("spacetime.fixed_metric") != "kerr_schild":
        return None
    mass = float(parameters.get("spacetime.mass", "1.0"))
    spin = np.array([
        float(parameters.get(f"spacetime.dimensionless_spin{axis}", "0.0"))
        for axis in (1, 2, 3)
    ])
    if abs(spin[0]) > 1.0e-14 or abs(spin[1]) > 1.0e-14:
        return None
    dimensionless_spin = abs(float(spin[2]))
    if dimensionless_spin >= 1.0:
        return None
    spin_length = mass * dimensionless_spin
    horizon_radius = mass + np.sqrt(mass * mass - spin_length * spin_length)
    equatorial_radius = np.sqrt(
        horizon_radius * horizon_radius + spin_length * spin_length
    )
    theta = np.linspace(0.0, np.pi, 33)
    phi = np.linspace(0.0, 2.0 * np.pi, 65)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    center = np.array([
        float(parameters.get(f"spacetime.origin{axis}", "0.0"))
        for axis in (1, 2, 3)
    ])
    x = center[0] + equatorial_radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = center[1] + equatorial_radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = center[2] + horizon_radius * np.cos(theta_grid)
    return x, y, z


def slice_views(
    plotfile: Path,
    radial_extent: float,
    vertical_extent: float,
    max_pixels: int,
) -> tuple[UniformSlice, UniformSlice, UniformSlice]:
    common = {"gamma": 2.0, "component": "dens"}
    equatorial = HierarchySlice(
        plotfile,
        slice_axis="z",
        slice_fraction=0.0,
        xlim=(0.0, radial_extent),
        ylim=(0.0, radial_extent),
        **common,
    ).uniform(max_pixels, method="nearest")
    xz = HierarchySlice(
        plotfile,
        slice_axis="y",
        slice_fraction=0.5,
        xlim=(0.0, radial_extent),
        ylim=(0.0, vertical_extent),
        **common,
    ).uniform(max_pixels, method="nearest")
    yz = HierarchySlice(
        plotfile,
        slice_axis="x",
        slice_fraction=0.5,
        xlim=(0.0, radial_extent),
        ylim=(0.0, vertical_extent),
        **common,
    ).uniform(max_pixels, method="nearest")
    return equatorial, xz, yz


def facecolors(
    values: np.ndarray,
    reference_density: float,
    normalization: colors.Normalize,
    colormap: colors.Colormap,
) -> np.ndarray:
    logarithmic = np.log10(
        np.maximum(values, reference_density * 1.0e-7) / reference_density
    )
    rgba = colormap(normalization(logarithmic))
    opacity = np.clip((logarithmic + 4.1) / 2.0, 0.0, 1.0)
    rgba[..., 3] = np.sqrt(opacity)
    rgba[logarithmic < -4.05, 3] = 0.0
    rgba[~np.isfinite(values), 3] = 0.0
    return rgba


def coordinates(view: UniformSlice) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(view.xlim[0], view.xlim[1], view.values.shape[1])
    y = np.linspace(view.ylim[0], view.ylim[1], view.values.shape[0])
    return np.meshgrid(x, y)


def plot_boxes(
    axis,
    view: UniformSlice,
    plane: str,
    *,
    offset: float,
) -> None:
    if not view.boxes:
        return
    maximum_level = max(box[0] for box in view.boxes)
    for level, lo1, hi1, lo2, hi2 in view.boxes:
        if level != maximum_level:
            continue
        coordinate1 = np.array([lo1, hi1, hi1, lo1, lo1])
        coordinate2 = np.array([lo2, lo2, hi2, hi2, lo2])
        if plane == "xy":
            axis.plot(
                coordinate1, coordinate2, offset,
                color="#7fdbff", alpha=0.38, linewidth=0.45,
            )
        elif plane == "xz":
            axis.plot(
                coordinate1, offset, coordinate2,
                color="#7fdbff", alpha=0.38, linewidth=0.45,
            )
        else:
            axis.plot(
                offset, coordinate1, coordinate2,
                color="#7fdbff", alpha=0.38, linewidth=0.45,
            )


def normalized_mode_amplitudes(
    rows: list[dict[str, str]], step: int
) -> dict[int, float]:
    available_steps = sorted({
        int(row["step"])
        for row in rows
        if int(row["step"]) <= step and row["field"] == "rest_mass"
    })
    if not available_steps:
        return {}
    selected_step = available_steps[-1]
    step_rows = [
        row
        for row in rows
        if int(row["step"]) == selected_step and row["field"] == "rest_mass"
    ]
    modes = {int(row["m"]): row for row in step_rows}
    if 0 not in modes:
        return {}
    mass = abs(float(modes[0]["real"]))
    denominator = max(mass, 1.0e-300)
    return {
        mode: np.hypot(float(row["real"]), float(row["imag"]))
        / denominator
        for mode, row in modes.items()
        if mode > 0
    }


def mode_history(
    rows: list[dict[str, str]], requested_modes: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    rest_mass_rows = [
        row for row in rows if row.get("field") == "rest_mass"
    ]
    steps = sorted({int(row["step"]) for row in rest_mass_rows})
    times: list[float] = []
    amplitudes = {mode: [] for mode in requested_modes}
    retained_steps: list[int] = []
    for step in steps:
        step_rows = {
            int(row["m"]): row
            for row in rest_mass_rows
            if int(row["step"]) == step
        }
        if not all(mode in step_rows for mode in (0, *requested_modes)):
            continue
        mass = abs(float(step_rows[0]["real"]))
        if mass <= 0.0:
            continue
        retained_steps.append(step)
        times.append(float(step_rows[0]["time"]))
        for mode in requested_modes:
            row = step_rows[mode]
            amplitude = np.hypot(
                float(row["real"]), float(row["imag"])
            )
            amplitudes[mode].append(amplitude / mass)
    return (
        np.asarray(retained_steps),
        np.asarray(times),
        {mode: np.asarray(values) for mode, values in amplitudes.items()},
    )


def plot_diagnostics(
    output: Path,
    scalar_rows: list[dict[str, str]],
    mode_rows: list[dict[str, str]],
    characteristic_period: float | None,
) -> None:
    steps, times, amplitudes = mode_history(mode_rows, (1, 2))
    if times.size == 0:
        raise RuntimeError("matter-mode history is empty")
    c0_rows = {
        int(row["step"]): row
        for row in mode_rows
        if row.get("field") == "rest_mass" and int(row["m"]) == 0
    }
    masses = np.asarray([
        abs(float(c0_rows[step]["real"])) for step in steps
    ])

    time_scale = characteristic_period or 1.0
    normalized_times = times / time_scale
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    for mode, color in ((1, "#3366cc"), (2, "#dc3912")):
        axes[0].semilogy(
            normalized_times,
            np.maximum(amplitudes[mode], 1.0e-16),
            color=color,
            linewidth=1.8,
            label=rf"$|C_{mode}|/C_0$",
        )
    axes[0].set_ylabel("normalized matter mode")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        normalized_times,
        masses / masses[0] - 1.0,
        color="#109618",
        linewidth=1.8,
        label=rf"$C_0/C_0(0)-1$",
    )
    axes[1].set_xlabel(r"$t/P_c$" if characteristic_period else r"$t/M$")
    axes[1].set_ylabel("disk baryon-mass change")
    axes[1].grid(alpha=0.25)
    flux_rows = [
        row
        for row in scalar_rows
        if row.get("black_hole_horizon_baryon_accretion_rate")
        not in (None, "")
    ]
    if flux_rows:
        flux_axis = axes[1].twinx()
        flux_axis.semilogy(
            [float(row["time"]) / time_scale for row in flux_rows],
            [
                max(
                    abs(float(row["black_hole_horizon_baryon_accretion_rate"])),
                    1.0e-30,
                )
                for row in flux_rows
            ],
            color="#990099",
            alpha=0.72,
            linewidth=1.4,
            label=rf"$|dM_0/dt|_{{\rm AH}}$",
        )
        flux_axis.set_ylabel("absolute horizon baryon flux")
        handles, labels = axes[1].get_legend_handles_labels()
        flux_handles, flux_labels = flux_axis.get_legend_handles_labels()
        axes[1].legend(
            handles + flux_handles,
            labels + flux_labels,
            loc="best",
            frameon=False,
        )
    else:
        axes[1].legend(frameon=False)
    title = "Black-hole disk modes and accretion"
    if characteristic_period:
        title += rf" ($P_c={characteristic_period:.3f}\,M$)"
    figure.suptitle(title)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def frame_annotation(
    scalar_rows: list[dict[str, str]],
    horizon_rows: list[dict[str, str]],
    mode_rows: list[dict[str, str]],
    parameters: dict[str, str],
    step: int,
    time: float,
    characteristic_period: float | None,
) -> str:
    scalar = latest_row(scalar_rows, step)
    horizon = latest_row(horizon_rows, step)
    lines = [rf"$t/M={time:.1f}$"]
    if characteristic_period:
        lines.append(rf"$t/P_c={time / characteristic_period:.3f}$")
    lines.append(f"step {step}")
    if horizon is not None:
        lines.extend((
            rf"$M_{{\rm AH}}={float(horizon['kerr_horizon_mass']):.4f}$",
            rf"$a_{{\rm AH}}/M={float(horizon['coordinate_christodoulou_dimensionless_spin_magnitude']):.4f}$",
        ))
    elif parameters.get("spacetime.fixed_metric") == "kerr_schild":
        mass = float(parameters.get("spacetime.mass", "1.0"))
        spin = np.linalg.norm([
            float(parameters.get(f"spacetime.dimensionless_spin{axis}", "0.0"))
            for axis in (1, 2, 3)
        ])
        lines.extend((rf"$M_{{\rm BH}}={mass:.4f}$", rf"$a/M={spin:.4f}$"))
    if scalar is not None:
        lines.append(rf"$M_0={float(scalar['mass']):.4e}$")
        if scalar.get("black_hole_horizon_baryon_accretion_rate") not in (
            None,
            "",
        ):
            lines.append(
                rf"$dM_0/dt|_{{\rm AH}}="
                rf"{float(scalar['black_hole_horizon_baryon_accretion_rate']):.2e}$"
            )
    mode_amplitudes = normalized_mode_amplitudes(mode_rows, step)
    for mode in (1, 2):
        if mode in mode_amplitudes:
            lines.append(
                rf"$|C_{mode}|/C_0={mode_amplitudes[mode]:.3e}$"
            )
    return "\n".join(lines)


def render_frame(
    plotfile: Path,
    output: Path,
    run_dir: Path,
    reference_density: float,
    scalar_rows: list[dict[str, str]],
    horizon_rows: list[dict[str, str]],
    mode_rows: list[dict[str, str]],
    amr_rows: list[dict[str, str]],
    parameters: dict[str, str],
    characteristic_period: float | None,
    radial_extent: float,
    vertical_extent: float,
    max_pixels: int,
) -> None:
    equatorial, xz, yz = slice_views(
        plotfile, radial_extent, vertical_extent, max_pixels
    )
    step = int(plotfile.stem.removeprefix("stateGF"))
    time = equatorial.time
    normalization = colors.Normalize(vmin=-5.2, vmax=0.0, clip=True)
    colormap = matplotlib.colormaps["magma"]

    figure = plt.figure(figsize=(13.6, 9.0), facecolor="#05070c")
    axis = figure.add_subplot(111, projection="3d", facecolor="#05070c")
    axis.set_proj_type("persp", focal_length=0.86)

    x, y = coordinates(equatorial)
    axis.plot_surface(
        x,
        y,
        np.zeros_like(x),
        facecolors=facecolors(
            equatorial.values, reference_density, normalization, colormap
        ),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
    )
    x, z = coordinates(xz)
    axis.plot_surface(
        x,
        np.zeros_like(x),
        z,
        facecolors=facecolors(xz.values, reference_density, normalization, colormap),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
    )
    y, z = coordinates(yz)
    axis.plot_surface(
        np.zeros_like(y),
        y,
        z,
        facecolors=facecolors(yz.values, reference_density, normalization, colormap),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
    )

    horizon = horizon_surface(run_dir, step)
    horizon_kind = "live apparent horizon"
    if horizon is None:
        horizon = exact_kerr_horizon(parameters)
        horizon_kind = "exact Kerr horizon"
    if horizon is not None:
        horizon_x, horizon_y, horizon_z = horizon
        positive_octant = (
            (horizon_x >= 0.0) & (horizon_y >= 0.0) & (horizon_z >= 0.0)
        )
        horizon_x = np.where(positive_octant, horizon_x, np.nan)
        horizon_y = np.where(positive_octant, horizon_y, np.nan)
        horizon_z = np.where(positive_octant, horizon_z, np.nan)
        axis.plot_surface(
            horizon_x,
            horizon_y,
            horizon_z,
            color="#020308",
            edgecolor="#f5c451",
            linewidth=0.22,
            alpha=1.0,
            shade=True,
        )

    box_offset = 0.025
    plot_boxes(axis, equatorial, "xy", offset=box_offset)
    plot_boxes(axis, xz, "xz", offset=box_offset)
    plot_boxes(axis, yz, "yz", offset=box_offset)

    axis.set_xlim(0.0, radial_extent)
    axis.set_ylim(0.0, radial_extent)
    axis.set_zlim(0.0, vertical_extent)
    axis.set_box_aspect((radial_extent, radial_extent, 1.35 * vertical_extent))
    axis.view_init(elev=26.0, azim=43.0)
    axis.set_xlabel(r"$x/M$", color="white", labelpad=8)
    axis.set_ylabel(r"$y/M$", color="white", labelpad=8)
    axis.set_zlabel(r"$z/M$", color="white", labelpad=8)
    axis.tick_params(colors="#c7d0df", labelsize=8)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.02, 0.03, 0.06, 0.78))
        pane.set_edgecolor((0.18, 0.25, 0.36, 0.55))
    axis.grid(False)

    figure.suptitle(
        "Spinning black-hole accretion — positive-octant cutaway",
        color="white",
        fontsize=17,
        y=0.965,
    )
    figure.text(
        0.035,
        0.825,
        frame_annotation(
            scalar_rows,
            horizon_rows,
            mode_rows,
            parameters,
            step,
            time,
            characteristic_period,
        ),
        color="#e7edf7",
        fontsize=11,
        va="top",
        family="monospace",
        bbox={
            "facecolor": "#090d16",
            "edgecolor": "#33425b",
            "alpha": 0.88,
            "boxstyle": "round,pad=0.55",
        },
    )
    figure.text(
        0.035,
        0.16,
        "dynamical fluid • fixed Kerr spacetime\n"
        "equatorial reflection • full azimuth\n"
        f"{parameters.get('pyfluid.recon', 'unknown').upper()} + "
        f"{parameters.get('pyfluid.riemann', 'unknown').upper()} • "
        f"{parameters.get('pyfluid.timestep', 'unknown').upper()} • "
        f"CFL {parameters.get('pyfluid.cfl', 'unknown')}\n"
        + (
            rf"AMR $\Delta x_{{\rm min}}="
            rf"{float(latest_row(amr_rows, step)['finest_dx']):.3g}M$ • "
            rf"active L{latest_row(amr_rows, step)['active_finest_level']}"
            if latest_row(amr_rows, step) is not None
            else "AMR hierarchy"
        ),
        color="#aebbd0",
        fontsize=9.5,
        va="bottom",
    )
    figure.text(
        0.735,
        0.105,
        f"gold: {horizon_kind}\ncyan: finest AMR boxes",
        color="#c8d3e6",
        fontsize=9.5,
        ha="left",
    )
    if horizon is not None:
        horizon_x, horizon_y, horizon_z = horizon
        equator_index = int(np.argmin(np.mean(np.abs(horizon_z), axis=1)))
        horizon_zoom = figure.add_axes((0.745, 0.665, 0.18, 0.19))
        horizon_zoom.set_facecolor("#090d16")
        outline_x = np.append(
            horizon_x[equator_index], horizon_x[equator_index, 0]
        )
        outline_y = np.append(
            horizon_y[equator_index], horizon_y[equator_index, 0]
        )
        horizon_zoom.fill(
            outline_x,
            outline_y,
            facecolor="#010207",
            edgecolor="#f5c451",
            linewidth=1.7,
        )
        horizon_zoom.axhline(0.0, color="#51627e", linewidth=0.45)
        horizon_zoom.axvline(0.0, color="#51627e", linewidth=0.45)
        horizon_zoom.set_xlim(-2.35, 2.35)
        horizon_zoom.set_ylim(-2.35, 2.35)
        horizon_zoom.set_aspect("equal")
        horizon_zoom.tick_params(colors="#aebbd0", labelsize=6, length=2)
        horizon_zoom.set_title(
            f"{horizon_kind} — equatorial close-up",
            color="#e7edf7",
            fontsize=8,
            pad=4,
        )
        for spine in horizon_zoom.spines.values():
            spine.set_color("#33425b")
    colorbar_axis = figure.add_axes((0.295, 0.075, 0.38, 0.025))
    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=colormap),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(
        r"$\log_{10}(D/D_{\rm max,first\ frame})$",
        color="white",
        fontsize=10,
    )
    colorbar.ax.tick_params(colors="#c7d0df", labelsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def encode_movie(frames_dir: Path, output: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the MP4 movie")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--radial-extent", type=float, default=28.0)
    parser.add_argument("--vertical-extent", type=float, default=14.0)
    parser.add_argument("--max-pixels", type=int, default=240)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--characteristic-period",
        type=float,
        help=(
            "Initial orbital period at the disk density maximum, in M. "
            "Defaults to the run's orbital_normalization.tsv when present."
        ),
    )
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    characteristic_period = args.characteristic_period
    if characteristic_period is None:
        orbital_rows = read_tsv(run_dir / "orbital_normalization.tsv")
        if orbital_rows:
            characteristic_period = float(orbital_rows[0]["period"])
    if characteristic_period is not None and (
        not np.isfinite(characteristic_period) or characteristic_period <= 0.0
    ):
        parser.error("the characteristic period must be positive and finite")
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "bh_accretion_octant.mp4"
    )
    timeline = plotfile_timeline(
        run_dir,
        prefix="stateGF",
        require=True,
        target_frames=args.max_frames,
    )
    scalar_rows = read_tsv(run_dir / "scalars.tsv")
    horizon_rows = read_tsv(run_dir / "apparent_horizon.tsv")
    mode_rows = read_tsv(run_dir / "multipoles.tsv")
    amr_rows = read_tsv(run_dir / "amr_levels.tsv")
    parameters = read_parameters(run_dir / "run.in")

    first_views = slice_views(
        timeline[0].path,
        args.radial_extent,
        args.vertical_extent,
        args.max_pixels,
    )
    reference_density = max(
        float(np.nanmax(view.values)) for view in first_views
    )
    if not np.isfinite(reference_density) or reference_density <= 0.0:
        raise RuntimeError("first movie frame has no positive finite density")

    frames_dir = run_dir / "bh_accretion_octant_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    for frame_index, sample in enumerate(timeline):
        frame = frames_dir / f"frame_{frame_index:05d}.png"
        render_frame(
            sample.path,
            frame,
            run_dir,
            reference_density,
            scalar_rows,
            horizon_rows,
            mode_rows,
            amr_rows,
            parameters,
            characteristic_period,
            args.radial_extent,
            args.vertical_extent,
            args.max_pixels,
        )
        print(
            f"rendered {frame_index + 1}/{len(timeline)}: "
            f"step={sample.step} time={sample.time:.6g}"
        )

    encode_movie(frames_dir, output, args.fps)
    poster = output.with_name(f"{output.stem}_poster.png")
    diagnostics = output.with_name(f"{output.stem}_diagnostics.png")
    shutil.copy2(frames_dir / f"frame_{len(timeline) // 2:05d}.png", poster)
    plot_diagnostics(
        diagnostics,
        scalar_rows,
        mode_rows,
        characteristic_period,
    )
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
    print(f"wrote {output}")
    print(f"wrote {poster}")
    print(f"wrote {diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
