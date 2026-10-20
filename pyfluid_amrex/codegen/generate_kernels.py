from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_DIFFERENTIAL_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "differential_ops.hpp"
)
DEFAULT_DIAGNOSTIC_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "diagnostics.hpp"
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
DEFAULT_RELATIVISTIC_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "relativistic_fluid.hpp"
)
DEFAULT_Z4C_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "z4c.hpp"
)
DEFAULT_INITIAL_DATA_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "kernels_generated"
   / "initial_data.hpp"
)
DEFAULT_FIELD_IDS_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "generated"
   / "field_ids.hpp"
)
DEFAULT_FIELD_CATALOG_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "generated"
   / "field_catalog.hpp"
)
DEFAULT_DIAGNOSTIC_CATALOG_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "generated"
   / "diagnostic_catalog.hpp"
)
DEFAULT_METHOD_CATALOG_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "generated"
   / "method_catalog.hpp"
)
DEFAULT_AMR_TAG_CATALOG_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "generated"
   / "amr_tag_catalog.hpp"
)
DEFAULT_INITIAL_DATA_METHOD_CATALOG_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "generated"
   / "initial_data_method_catalog.hpp"
)
DEFAULT_PHYSICAL_ID_CATALOG_HEADER = (
   ROOT
   / "pyfluid_amrex"
   / "include"
   / "pyfluid_amrex"
   / "generated"
   / "physical_id_catalog.hpp"
)
DEFAULT_EQUATION_MARKDOWN = ROOT / "pyfluid_amrex" / "docs" / "equations.md"
DEFAULT_EQUATION_LATEX = ROOT / "pyfluid_amrex" / "docs" / "equations.tex"
DEFAULT_FIELD_CATALOG_JSON = ROOT / "pyfluid_amrex" / "generated" / "field_catalog.json"
DEFAULT_PHYSICAL_ID_CATALOG_JSON = (
   ROOT / "pyfluid_amrex" / "generated" / "physical_id_catalog.json"
)


def main() -> None:
   parser = argparse.ArgumentParser(description="Generate PyFluid AMReX C++ kernels.")
   parser.add_argument(
      "--family",
      action="append",
      choices=("catalog", "characteristic", "diagnostic", "differential", "eos", "flux", "initial_data", "operator", "relativistic", "riemann", "vector", "metric", "patch_map", "z4c", "spacetime", "fccz4"),
      help="generate only this kernel family; repeat as needed (default: all)",
   )
   parser.add_argument("--flux-header", type=Path, default=DEFAULT_FLUX_HEADER)
   parser.add_argument("--eos-header", type=Path, default=DEFAULT_EOS_HEADER)
   parser.add_argument("--operator-header", type=Path, default=DEFAULT_OPERATOR_HEADER)
   parser.add_argument("--differential-header", type=Path, default=DEFAULT_DIFFERENTIAL_HEADER)
   parser.add_argument("--diagnostic-header", type=Path, default=DEFAULT_DIAGNOSTIC_HEADER)
   parser.add_argument("--riemann-header", type=Path, default=DEFAULT_RIEMANN_HEADER)
   parser.add_argument("--characteristic-header", type=Path, default=DEFAULT_CHARACTERISTIC_HEADER)
   parser.add_argument("--vector-header", type=Path, default=DEFAULT_VECTOR_HEADER)
   parser.add_argument("--metric-header", type=Path, default=DEFAULT_METRIC_HEADER)
   parser.add_argument("--patch-map-header", type=Path, default=DEFAULT_PATCH_MAP_HEADER)
   parser.add_argument("--relativistic-header", type=Path, default=DEFAULT_RELATIVISTIC_HEADER)
   parser.add_argument("--z4c-header", type=Path, default=DEFAULT_Z4C_HEADER)
   parser.add_argument("--fccz4-header", type=Path, default=DEFAULT_Z4C_HEADER.with_name("fccz4.hpp"))
   parser.add_argument("--spacetime-header", type=Path,
                       default=DEFAULT_Z4C_HEADER.with_name("spacetime_geometry.hpp"))
   parser.add_argument("--initial-data-header", type=Path, default=DEFAULT_INITIAL_DATA_HEADER)
   parser.add_argument("--field-ids-header", type=Path, default=DEFAULT_FIELD_IDS_HEADER)
   parser.add_argument("--field-catalog-header", type=Path, default=DEFAULT_FIELD_CATALOG_HEADER)
   parser.add_argument("--diagnostic-catalog-header", type=Path, default=DEFAULT_DIAGNOSTIC_CATALOG_HEADER)
   parser.add_argument("--method-catalog-header", type=Path, default=DEFAULT_METHOD_CATALOG_HEADER)
   parser.add_argument(
      "--amr-tag-catalog-header",
      type=Path,
      default=DEFAULT_AMR_TAG_CATALOG_HEADER,
   )
   parser.add_argument(
      "--initial-data-method-catalog-header",
      type=Path,
      default=DEFAULT_INITIAL_DATA_METHOD_CATALOG_HEADER,
   )
   parser.add_argument(
      "--physical-id-catalog-header",
      type=Path,
      default=DEFAULT_PHYSICAL_ID_CATALOG_HEADER,
   )
   parser.add_argument("--field-catalog-json", type=Path, default=DEFAULT_FIELD_CATALOG_JSON)
   parser.add_argument(
      "--physical-id-catalog-json",
      type=Path,
      default=DEFAULT_PHYSICAL_ID_CATALOG_JSON,
   )
   parser.add_argument("--equation-markdown", type=Path, default=DEFAULT_EQUATION_MARKDOWN)
   parser.add_argument("--equation-latex", type=Path, default=DEFAULT_EQUATION_LATEX)
   parser.add_argument("--hard-drop-ops", type=int, default=0)
   parser.add_argument("--soft-drop-ops", type=int, default=3)
   parser.add_argument("--freshness-file", type=Path,
                       help="shared CMake family provenance; skip current outputs before symbolic imports")
   parser.add_argument("--dependency", type=Path, action="append", default=[])
   parser.add_argument("--expected-output", type=Path, action="append", default=[])
   parser.add_argument("--lock-timeout", type=float, default=300.0)
   args = parser.parse_args()
   kernel_family_names = (
      "characteristic",
      "differential",
      "diagnostic",
      "eos",
      "flux",
      "initial_data",
      "operator",
      "riemann",
      "vector",
      "metric",
      "patch_map",
      "relativistic",
      "z4c",
      "spacetime",
      "fccz4",
   )
   families = (
      tuple(args.family)
      if args.family
      else ("catalog", *kernel_family_names)
   )
   if args.freshness_file is None:
      generate_families(args, families)
      return
   from generation_cache import generate_if_needed

   def generate_staged(replacements):
      staged = argparse.Namespace(**vars(args))
      for name, value in vars(args).items():
         if isinstance(value, Path) and value.resolve() in replacements:
            setattr(staged, name, replacements[value.resolve()])
      generate_families(staged, families)

   changed = generate_if_needed(
      args.freshness_file,
      args.dependency,
      args.expected_output,
      {"families": list(families), "hard_drop_ops": args.hard_drop_ops,
       "soft_drop_ops": args.soft_drop_ops},
      generate_staged,
      lock_timeout=args.lock_timeout,
   )
   print(f"Codegen {'generated' if changed else 'current'}: {','.join(families)}")


def generate_families(args, families) -> None:
   generators = None
   policy = None
   for family in families:
      if family == "catalog":
         from pyfluid_codegen.emit_catalog import write_catalog

         write_catalog(
            args.field_ids_header,
            args.field_catalog_header,
            args.diagnostic_catalog_header,
            args.method_catalog_header,
            args.amr_tag_catalog_header,
            args.initial_data_method_catalog_header,
            args.physical_id_catalog_header,
            args.field_catalog_json,
            args.physical_id_catalog_json,
            args.equation_markdown,
            args.equation_latex,
         )
         print(args.field_ids_header)
         print(args.field_catalog_header)
         print(args.diagnostic_catalog_header)
         print(args.method_catalog_header)
         print(args.amr_tag_catalog_header)
         print(args.initial_data_method_catalog_header)
         print(args.physical_id_catalog_header)
         print(args.field_catalog_json)
         print(args.physical_id_catalog_json)
         print(args.equation_markdown)
         print(args.equation_latex)
         continue
      if generators is None:
         from pyfluid_codegen.emit_cpp import (
            CsePolicy,
            write_characteristic_header,
            write_differential_header,
            write_diagnostic_header,
            write_eos_header,
            write_flux_header,
            write_initial_data_header,
            write_metric_header,
            write_operator_header,
            write_patch_map_header,
            write_relativistic_header,
            write_riemann_header,
            write_vector_header,
            write_z4c_header,
            write_spacetime_header,
            write_fccz4_header,
         )

         policy = CsePolicy(
            hard_drop_ops=args.hard_drop_ops,
            soft_drop_ops=args.soft_drop_ops,
         )
         generators = {
            "characteristic": (
               args.characteristic_header,
               write_characteristic_header,
            ),
            "differential": (
               args.differential_header,
               write_differential_header,
            ),
            "diagnostic": (
               args.diagnostic_header,
               write_diagnostic_header,
            ),
            "eos": (args.eos_header, write_eos_header),
            "flux": (args.flux_header, write_flux_header),
            "initial_data": (
               args.initial_data_header,
               write_initial_data_header,
            ),
            "operator": (args.operator_header, write_operator_header),
            "riemann": (args.riemann_header, write_riemann_header),
            "vector": (args.vector_header, write_vector_header),
            "metric": (args.metric_header, write_metric_header),
            "patch_map": (
               args.patch_map_header,
               write_patch_map_header,
            ),
            "relativistic": (
               args.relativistic_header,
               write_relativistic_header,
            ),
            "z4c": (args.z4c_header, write_z4c_header),
            "spacetime": (args.spacetime_header, write_spacetime_header),
            "fccz4": (args.fccz4_header, write_fccz4_header),
         }
      output_path, writer = generators[family]
      writer(output_path, policy)
      print(output_path)


if __name__ == "__main__":
   main()
