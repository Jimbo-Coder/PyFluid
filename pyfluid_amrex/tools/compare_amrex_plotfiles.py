#!/usr/bin/env python3
"""Compare two AMReX plotfiles on a fixed nearest-neighbor display grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amrex_boxed_movie import composite_slice, parse_limits, plotfile_step  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--component", default="rho")
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--ref-ratio", type=int, default=2)
    parser.add_argument("--z-fraction", type=float, default=0.5)
    parser.add_argument("--max-pixels", type=int, default=1024)
    parser.add_argument("--title", default="AMR comparison")
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    args = parser.parse_args()
    xlim = parse_limits(args.xlim, (0.0, 1.0))
    ylim = parse_limits(args.ylim, (0.0, 1.0))

    ref, _boxes, tref, _ndim, _names, xlim, ylim = composite_slice(
        args.reference, args.component, args.gamma, args.ref_ratio, args.z_fraction, False, None, args.max_pixels, xlim, ylim
    )
    cand, _boxes, tcand, _ndim, _names, xlim, ylim = composite_slice(
        args.candidate, args.component, args.gamma, args.ref_ratio, args.z_fraction, False, None, args.max_pixels, xlim, ylim
    )
    mask = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(mask):
        raise ValueError("no overlapping finite pixels to compare")
    diff = cand - ref
    scale = max(float(np.mean(np.abs(ref[mask]))), 1.0e-30)
    l1 = float(np.mean(np.abs(diff[mask]))) / scale
    l2 = float(np.sqrt(np.mean(diff[mask] ** 2))) / scale
    linf = float(np.max(np.abs(diff[mask]))) / scale

    if not args.skip_plot:
        if args.output is None:
            parser.error("--output is required unless --skip-plot is selected")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vmin = float(np.percentile(np.concatenate([ref[mask], cand[mask]]), 2.0))
        vmax = float(np.percentile(np.concatenate([ref[mask], cand[mask]]), 98.0))
        dmax = float(np.percentile(np.abs(diff[mask]), 99.0))
        dmax = max(dmax, float(np.max(np.abs(diff[mask]))) * 0.2, 1.0e-30)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
        panels = [
            (axes[0], ref, f"reference step={plotfile_step(args.reference)}, t={tref:.4f}", "viridis", vmin, vmax),
            (axes[1], cand, f"candidate step={plotfile_step(args.candidate)}, t={tcand:.4f}", "viridis", vmin, vmax),
            (axes[2], diff, f"candidate - reference\nrel L1={l1:.3e}, L2={l2:.3e}, Linf={linf:.3e}", "coolwarm", -dmax, dmax),
        ]
        for ax, image, subtitle, cmap, lo, hi in panels:
            im = ax.imshow(
                image,
                origin="lower",
                extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                interpolation="nearest",
                cmap=cmap,
                vmin=lo,
                vmax=hi,
            )
            ax.set_title(subtitle)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"{args.title}: {args.component}")
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
    print(f"{args.component} rel_l1={l1:.8e} rel_l2={l2:.8e} rel_linf={linf:.8e}")


if __name__ == "__main__":
    main()
