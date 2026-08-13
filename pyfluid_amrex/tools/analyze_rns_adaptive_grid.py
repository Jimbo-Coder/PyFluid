#!/usr/bin/env python3
"""Measure radial allocation and exterior angular content of an RNS ID."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from validation_common import ROOT, repo_relative, write_tsv


ADM_BASELINE = np.array(
    [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    + [0.0] * 6
    + [1.0]
    + [0.0] * 3,
    dtype=float,
)
ADM_COMPONENTS = (
    "gamma11", "gamma12", "gamma13", "gamma22", "gamma23", "gamma33",
    "k11", "k12", "k13", "k22", "k23", "k33",
    "lapse", "shift1", "shift2", "shift3",
)


def read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in (path / "Header.meta").read_text().splitlines():
        if not line.startswith("# ") or "=" not in line:
            continue
        key, value = line[2:].split("=", 1)
        metadata[key] = value
    return metadata


def read_dense_field(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        num_components = int(np.asarray(handle.attrs["num_components"])[0])
        group = handle["level_0"]
        boxes = np.asarray(group["boxes"])
        offsets = np.asarray(group["data:offsets=0"], dtype=np.int64)
        data = np.asarray(group["data:datatype=0"])
        upper = np.max(
            np.asarray([tuple(int(value) for value in box)[2:] for box in boxes]),
            axis=0,
        )
        field = np.full((num_components, upper[0] + 1, upper[1] + 1), np.nan)
        for box, begin, end in zip(boxes, offsets[:-1], offsets[1:]):
            bounds = tuple(int(value) for value in box)
            lower = bounds[:2]
            box_upper = bounds[2:]
            extent = (
                box_upper[0] - lower[0] + 1,
                box_upper[1] - lower[1] + 1,
            )
            block = data[begin:end].reshape(
                num_components, extent[1], extent[0]
            ).transpose(0, 2, 1)
            destination = field[
                :,
                lower[0] : box_upper[0] + 1,
                lower[1] : box_upper[1] + 1,
            ]
            overlap = np.isfinite(destination)
            if np.any(overlap) and not np.allclose(
                destination[overlap], block[overlap], rtol=2.0e-12, atol=2.0e-14
            ):
                disagreement = float(
                    np.max(np.abs(destination[overlap] - block[overlap]))
                )
                raise RuntimeError(
                    "overlapping physical-ID nodes disagree by "
                    f"{disagreement:.6e}"
                )
            destination[...] = block
        if not np.all(np.isfinite(field)):
            raise RuntimeError("physical-ID HDF5 boxes do not cover a dense level")
        return field


def angular_basis(
    theta: np.ndarray,
    modes: int,
    odd_axis: bool,
    odd_equator: bool,
) -> np.ndarray:
    if odd_axis and odd_equator:
        wave_numbers = 2 * np.arange(1, modes)
        return np.sin(np.outer(theta, wave_numbers))
    if odd_axis:
        wave_numbers = 2 * np.arange(modes) + 1
        return np.sin(np.outer(theta, wave_numbers))
    if odd_equator:
        wave_numbers = 2 * np.arange(modes) + 1
        return np.cos(np.outer(theta, wave_numbers))
    wave_numbers = 2 * np.arange(modes + 1)
    return np.cos(np.outer(theta, wave_numbers))


def angular_compression_error(
    perturbation: np.ndarray,
    theta: np.ndarray,
    exterior_begin: int,
    angular_cells: int,
) -> tuple[float, float, int, int]:
    full_angular_cells = theta.size - 1
    if angular_cells >= full_angular_cells:
        return 0.0, 0.0, perturbation.shape[0], -1
    sample_indices = np.rint(
        np.linspace(0, full_angular_cells, angular_cells + 1)
    ).astype(int)
    if np.unique(sample_indices).size != angular_cells + 1:
        raise RuntimeError("requested angular compression repeats source nodes")
    active_components = 0
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    worst_component = -1
    for component in range(perturbation.shape[0]):
        values = perturbation[component, exterior_begin:, :]
        amplitude = float(np.max(np.abs(values)))
        if amplitude <= 1.0e-13:
            continue
        active_components += 1
        boundary_tolerance = max(1.0e-12, 1.0e-8 * amplitude)
        odd_axis = float(np.max(np.abs(values[:, 0]))) <= boundary_tolerance
        odd_equator = (
            float(np.max(np.abs(values[:, -1]))) <= boundary_tolerance
        )
        coarse_basis = angular_basis(
            theta[sample_indices], angular_cells, odd_axis, odd_equator
        )
        full_basis = angular_basis(
            theta, angular_cells, odd_axis, odd_equator
        )
        for radial_values in values:
            coefficients, *_ = np.linalg.lstsq(
                coarse_basis,
                radial_values[sample_indices],
                rcond=None,
            )
            reconstructed = full_basis @ coefficients
            error = float(np.max(np.abs(reconstructed - radial_values)))
            if error > maximum_absolute_error:
                maximum_absolute_error = error
                worst_component = component
            maximum_relative_error = max(
                maximum_relative_error, error / amplitude
            )
    return (
        maximum_absolute_error,
        maximum_relative_error,
        active_components,
        worst_component,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--angular-cells", type=int, nargs="+", default=(4, 8, 16, 32)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    metadata = read_metadata(artifact)
    if metadata.get("mesh_coordinate") != "spherical" \
            or metadata.get("mesh_axes") != "r theta" \
            or metadata.get("source_ic") != "rotating_star":
        raise RuntimeError("analysis requires a spherical r-theta rotating-star ID")
    radial_faces = np.fromstring(metadata["coordinate_faces_0_0"], sep=",")
    if radial_faces.size < 3 or not np.all(np.diff(radial_faces) > 0.0):
        raise RuntimeError("artifact does not contain a valid explicit radial map")
    widths = np.diff(radial_faces)
    width_tolerance = 64.0 * np.finfo(float).eps * max(1.0, widths[0])
    changed = np.flatnonzero(np.abs(widths - widths[0]) > width_tolerance)
    exterior_begin = int(changed[0]) if changed.size else widths.size
    source_radius = radial_faces[exterior_begin]

    adm = read_dense_field(artifact / "adm_geometry.h5")
    if adm.shape[0] != ADM_BASELINE.size \
            or adm.shape[1] != radial_faces.size:
        raise RuntimeError("ADM field shape is inconsistent with radial metadata")
    theta = np.linspace(0.0, 0.5 * np.pi, adm.shape[2])
    perturbation = adm - ADM_BASELINE[:, None, None]

    equatorial_radius = float(metadata["equatorial_radius"])
    region_radii = [source_radius]
    for factor in (1.5, 2.0, 3.0):
        radius = factor * equatorial_radius
        if source_radius < radius < radial_faces[-1]:
            region_radii.append(radius)
    rows: list[dict[str, object]] = []
    for region_radius in region_radii:
        region_begin = int(np.searchsorted(radial_faces, region_radius))
        global_amplitude = float(
            np.max(np.abs(perturbation[:, region_begin:, :]))
        )
        for angular_cells in sorted(set(args.angular_cells)):
            if angular_cells < 2:
                raise RuntimeError("angular compression needs at least two cells")
            absolute, relative, active, worst_component = \
                angular_compression_error(
                    perturbation, theta, region_begin, angular_cells
                )
            rows.append(
                {
                    "kind": "initial_data",
                    "case": "rns_exterior_angular_content",
                    "status": "PASS",
                    "region_radius": region_radius,
                    "region_radius_over_equator": (
                        region_radius / equatorial_radius
                    ),
                    "angular_cells": angular_cells,
                    "source_angular_cells": theta.size - 1,
                    "radial_cells": widths.size,
                    "source_cells": exterior_begin,
                    "exterior_cells": widths.size - exterior_begin,
                    "source_radius": source_radius,
                    "outer_radius": radial_faces[-1],
                    "active_components": active,
                    "worst_component": (
                        ADM_COMPONENTS[worst_component]
                        if worst_component >= 0 else "none"
                    ),
                    "maximum_absolute_error": absolute,
                    "maximum_global_relative_error": (
                        absolute / max(global_amplitude, 1.0e-30)
                    ),
                    "maximum_component_relative_error": relative,
                    "artifact": repo_relative(artifact),
                }
            )

    output = args.output or artifact.parent / "adaptive_grid_analysis.tsv"
    write_tsv(output, rows, list(rows[0]))
    print(
        f"radial cells={widths.size} source={exterior_begin} "
        f"exterior={widths.size - exterior_begin} "
        f"source_radius={source_radius:.8g} outer_radius={radial_faces[-1]:.8g}"
    )
    previous_radius = None
    for row in rows:
        if row["region_radius"] != previous_radius:
            previous_radius = row["region_radius"]
            print(
                f"region r>={row['region_radius']:.8g} "
                f"({row['region_radius_over_equator']:.3f} R_eq)"
            )
        print(
            f"theta={row['angular_cells']:>3} "
            f"max_abs={row['maximum_absolute_error']:.6e} "
            f"global_rel={row['maximum_global_relative_error']:.6e} "
            f"worst={row['worst_component']} "
            f"max_component_rel={row['maximum_component_relative_error']:.6e}"
        )
    output_label = repo_relative(output) if output.resolve().is_relative_to(
        ROOT
    ) else str(output.resolve())
    print(f"wrote {output_label}")


if __name__ == "__main__":
    main()
