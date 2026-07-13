from __future__ import annotations

import argparse
from pathlib import Path

from pyfluid_codegen.emit_cpp import (
   CsePolicy,
   write_characteristic_header,
   write_eos_header,
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
DEFAULT_EOS_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "eos.hpp"
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
DEFAULT_CHARACTERISTIC_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "characteristic_transforms.hpp"
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
   parser.add_argument(
      "--family",
      action="append",
      choices=("characteristic", "eos", "flux", "operator", "riemann", "vector", "metric", "patch_map"),
      help="generate only this kernel family; repeat as needed (default: all)",
   )
   parser.add_argument("--flux-header", type=Path, default=DEFAULT_FLUX_HEADER)
   parser.add_argument("--eos-header", type=Path, default=DEFAULT_EOS_HEADER)
   parser.add_argument("--operator-header", type=Path, default=DEFAULT_OPERATOR_HEADER)
   parser.add_argument("--riemann-header", type=Path, default=DEFAULT_RIEMANN_HEADER)
   parser.add_argument("--characteristic-header", type=Path, default=DEFAULT_CHARACTERISTIC_HEADER)
   parser.add_argument("--vector-header", type=Path, default=DEFAULT_VECTOR_HEADER)
   parser.add_argument("--metric-header", type=Path, default=DEFAULT_METRIC_HEADER)
   parser.add_argument("--patch-map-header", type=Path, default=DEFAULT_PATCH_MAP_HEADER)
   parser.add_argument("--hard-drop-ops", type=int, default=0)
   parser.add_argument("--soft-drop-ops", type=int, default=3)
   args = parser.parse_args()
   policy = CsePolicy(hard_drop_ops=args.hard_drop_ops, soft_drop_ops=args.soft_drop_ops)
   generators = {
      "characteristic": (args.characteristic_header, write_characteristic_header),
      "eos": (args.eos_header, write_eos_header),
      "flux": (args.flux_header, write_flux_header),
      "operator": (args.operator_header, write_operator_header),
      "riemann": (args.riemann_header, write_riemann_header),
      "vector": (args.vector_header, write_vector_header),
      "metric": (args.metric_header, write_metric_header),
      "patch_map": (args.patch_map_header, write_patch_map_header),
   }
   families = tuple(args.family) if args.family else tuple(generators)
   for family in families:
      output_path, writer = generators[family]
      writer(output_path, policy)
      print(output_path)


if __name__ == "__main__":
   main()
