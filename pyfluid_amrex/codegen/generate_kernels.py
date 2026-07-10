from __future__ import annotations

import argparse
from pathlib import Path

from pyfluid_codegen.emit_cpp import (
   CsePolicy,
   write_flux_header,
   write_metric_header,
   write_operator_header,
   write_patch_map_header,
   write_riemann_header,
   write_vector_header,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLUX_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "ideal_mhd_flux_cartesian.hpp"
)
DEFAULT_OPERATOR_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "scalar_ops.hpp"
)
DEFAULT_RIEMANN_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "riemann_ops.hpp"
)
DEFAULT_VECTOR_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "vector_ops.hpp"
)
DEFAULT_METRIC_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "coordinate_metrics.hpp"
)
DEFAULT_PATCH_MAP_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "patch_maps.hpp"
)


def main() -> None:
   parser = argparse.ArgumentParser(description="Generate PyFluid AMReX C++ kernels.")
   parser.add_argument("--flux-header", type=Path, default=DEFAULT_FLUX_HEADER)
   parser.add_argument("--operator-header", type=Path, default=DEFAULT_OPERATOR_HEADER)
   parser.add_argument("--riemann-header", type=Path, default=DEFAULT_RIEMANN_HEADER)
   parser.add_argument("--vector-header", type=Path, default=DEFAULT_VECTOR_HEADER)
   parser.add_argument("--metric-header", type=Path, default=DEFAULT_METRIC_HEADER)
   parser.add_argument("--patch-map-header", type=Path, default=DEFAULT_PATCH_MAP_HEADER)
   parser.add_argument("--hard-drop-ops", type=int, default=0)
   parser.add_argument("--soft-drop-ops", type=int, default=3)
   args = parser.parse_args()
   policy = CsePolicy(hard_drop_ops=args.hard_drop_ops, soft_drop_ops=args.soft_drop_ops)
   write_flux_header(args.flux_header, policy)
   write_operator_header(args.operator_header, policy)
   write_riemann_header(args.riemann_header, policy)
   write_vector_header(args.vector_header, policy)
   write_metric_header(args.metric_header, policy)
   write_patch_map_header(args.patch_map_header, policy)
   print(args.flux_header)
   print(args.operator_header)
   print(args.riemann_header)
   print(args.vector_header)
   print(args.metric_header)
   print(args.patch_map_header)


if __name__ == "__main__":
   main()
