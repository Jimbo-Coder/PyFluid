"""Small build-ownership regressions; no AMReX rebuild or simulation needed."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codegen"))
sys.path.insert(0, str(ROOT / "tools"))
from pyfluid_codegen import emit_catalog, emit_cpp
from pyfluid_codegen.vector_equations import VECTOR_KERNEL_GROUPS
from pyfluid_codegen.method_catalog import SCHEMA_PATH, _load_initial_data_owner
from benchmark_compile import compile_status
from build_dependencies import IncludeGraph, ownership_errors
from profile_compile import isolated_command, select_entry, trace_totals


class BuildContractTests(unittest.TestCase):
    def test_local_presets_pin_architecture_across_sandboxed_configures(self):
        presets = json.loads((ROOT / "CMakePresets.json").read_text())
        by_name = {item["name"]: item for item in presets["configurePresets"]}
        full = by_name["full-release"]
        self.assertEqual(full["generator"], "Ninja")
        self.assertEqual(full["cacheVariables"]["CMAKE_EXPORT_COMPILE_COMMANDS"], "ON")
        self.assertEqual(full["cacheVariables"]["CMAKE_OSX_ARCHITECTURES"], "arm64")
        self.assertIn("/ccache", full["cacheVariables"]["CMAKE_CXX_COMPILER_LAUNCHER"])
        for name, preset_name in (("local", "full-release"),
                                  ("local_dev_rmhd", "focused-rmhd-pwl")):
            with self.subTest(machine=name):
                with (ROOT / "machines" / f"{name}.toml").open("rb") as stream:
                    profile = tomllib.load(stream)
                self.assertEqual(profile["build"]["configure_preset"], preset_name)
                self.assertNotIn("cmake_args", profile["build"])
        focused = by_name["focused-rmhd-pwl"]
        self.assertEqual(focused["inherits"], "full-release")
        self.assertNotIn("generator", focused)  # Inherits the full build context.
        for name in ("full-release-openmp", "full-release-fp32"):
            self.assertEqual(by_name[name]["inherits"], "full-release")
            self.assertEqual(by_name[name]["generator"], "Unix Makefiles")
        self.assertEqual(by_name["experimental-metal"]["generator"], "Ninja")
        with (ROOT / "machines/local_openmp.toml").open("rb") as stream:
            openmp = tomllib.load(stream)
        self.assertIn("-DCMAKE_OSX_ARCHITECTURES=arm64", openmp["build"]["cmake_args"])

    def test_solver_projection_layouts_are_canonical_and_unambiguous(self):
        with SCHEMA_PATH.open("rb") as stream:
            schema = tomllib.load(stream)
        record = next(item for item in schema["initial_data_owner"]
                      if item["name"] == "cartesian_conformal_constraints")
        owner = _load_initial_data_owner(record, 0)
        self.assertEqual(owner.dimension, 3)
        self.assertIn("x z", owner.solver_axis_layouts)
        for invalid in ("x x", "z x", "x  z", " x", "r", "x y z x"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _load_initial_data_owner(
                    {**record, "solver_axis_layouts": ["x y z", invalid]}, 0
                )
        with self.assertRaises(ValueError):
            _load_initial_data_owner({**record, "solver_axis_layouts": ["x"]}, 0)

    def test_indirect_leaks_report_the_chain_and_cycles_terminate(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-includes-") as temporary:
            root = Path(temporary).resolve()
            files = {
                "include/pyfluid_amrex/cache_factory.hpp":
                    '#include "pyfluid_amrex/bridge.hpp"\n',
                "include/pyfluid_amrex/bridge.hpp":
                    '#include "pyfluid_amrex/cache_factory.hpp"\n'
                    '#include "pyfluid_amrex/kernels_generated/heavy.hpp"\n',
                "include/pyfluid_amrex/kernels_generated/heavy.hpp": "",
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            errors = ownership_errors(IncludeGraph(root))
            self.assertEqual(len(errors), 1)
            self.assertIn("cache_factory.hpp -> include/pyfluid_amrex/bridge.hpp ->", errors[0])

    def test_method_ownership_allows_shared_ppm_but_not_global_leaks(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-method-owners-") as temporary:
            root = Path(temporary).resolve()
            files = {
                "include/pyfluid_amrex/reconstruction/parabolic.hpp": "",
                "include/pyfluid_amrex/reconstruction/ppm4.hpp":
                    '#include "parabolic.hpp"\n',
                "include/pyfluid_amrex/bridge.hpp":
                    '#if SOME_FEATURE\n#include "reconstruction/ppm4.hpp"\n#endif\n',
                "src/flux_kernels/ppm4.cpp":
                    '#include <pyfluid_amrex/reconstruction/ppm4.hpp>\n',
                "src/relativity/rmhd_advanced_flux.cpp":
                    '#include "pyfluid_amrex/bridge.hpp"\n',
                "src/amr/unrelated.cpp":
                    '/* #include "pyfluid_amrex/bridge.hpp" */\n#include <vector>\n',
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            self.assertEqual(ownership_errors(IncludeGraph(root)), [])
            (root / "src/amr/unrelated.cpp").write_text(
                '#include "pyfluid_amrex/bridge.hpp"\n'
            )
            errors = ownership_errors(IncludeGraph(root))
            self.assertEqual(len(errors), 2)  # Both PPM4 and its shared helper.
            self.assertTrue(all("src/amr/unrelated.cpp ->" in error for error in errors))

    def test_host_field_metadata_cannot_reach_numerical_owners(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-field-metadata-") as temporary:
            root = Path(temporary).resolve()
            files = {
                "include/pyfluid_amrex/fields.hpp": "",
                "include/pyfluid_amrex/generated/field_ids.hpp": "",
                "include/pyfluid_amrex/config.hpp":
                    '#include "pyfluid_amrex/generated/field_ids.hpp"\n',
                "include/pyfluid_amrex/z4c_state_access.hpp":
                    '#include "pyfluid_amrex/generated/field_ids.hpp"\n',
                "include/pyfluid_amrex/numerical_values.hpp":
                    '#include "pyfluid_amrex/generated/field_ids.hpp"\n',
                "include/pyfluid_amrex/bridge.hpp":
                    '#include "pyfluid_amrex/fields.hpp"\n',
                "src/flux_kernels/pwc.cpp":
                    '#include "pyfluid_amrex/numerical_values.hpp"\n',
                "src/relativity/rmhd_correction_flux_llf.cpp":
                    '#include "pyfluid_amrex/z4c_state_access.hpp"\n',
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            self.assertEqual(ownership_errors(IncludeGraph(root)), [])
            for header, owner in (
                ("config.hpp", "include/pyfluid_amrex/config.hpp"),
                ("z4c_state_access.hpp", "src/relativity/rmhd_correction_flux_llf.cpp"),
                ("numerical_values.hpp", "src/flux_kernels/pwc.cpp"),
            ):
                with self.subTest(header=header):
                    path = root / "include/pyfluid_amrex" / header
                    original = path.read_text()
                    path.write_text('#include "pyfluid_amrex/bridge.hpp"\n')
                    errors = ownership_errors(IncludeGraph(root))
                    self.assertTrue(any(
                        "numerical contract reaches host field metadata" in error
                        and owner + " ->" in error and "bridge.hpp ->" in error
                        for error in errors
                    ))
                    path.write_text(original)

    def test_numerical_owners_cannot_import_host_config_through_diode(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-device-config-") as temporary:
            root = Path(temporary).resolve()
            files = {
                "include/pyfluid_amrex/config.hpp": "",
                "include/pyfluid_amrex/relativistic_diode.hpp": "",
                "include/pyfluid_amrex/relativistic_flux_limiter.hpp":
                    '#include "pyfluid_amrex/relativistic_diode.hpp"\n',
                "src/relativity/rmhd_correction_flux_llf.cpp":
                    '#include "pyfluid_amrex/relativistic_flux_limiter.hpp"\n',
                "src/flux_kernels/pwc.cpp":
                    '#include "pyfluid_amrex/relativistic_diode.hpp"\n',
                "src/relativity/rmhd_advanced_flux.cpp":
                    '#include "pyfluid_amrex/relativistic_diode.hpp"\n',
                "src/relativity/relativistic_flux_limiter.cpp":
                    '#include "pyfluid_amrex/relativistic_diode.hpp"\n',
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            self.assertEqual(ownership_errors(IncludeGraph(root)), [])
            (root / "include/pyfluid_amrex/relativistic_diode.hpp").write_text(
                '#include "pyfluid_amrex/config.hpp"\n'
            )
            errors = ownership_errors(IncludeGraph(root))
            for owner in ("src/flux_kernels/pwc.cpp",
                          "src/relativity/rmhd_advanced_flux.cpp",
                          "src/relativity/relativistic_flux_limiter.cpp",
                          "src/relativity/rmhd_correction_flux_llf.cpp"):
                with self.subTest(owner=owner):
                    self.assertTrue(any(
                        "numerical value contract reaches host configuration" in error
                        and owner + " ->" in error
                        and "relativistic_diode.hpp ->" in error
                        for error in errors
                    ))

    def test_passive_flux_does_not_import_fluid_implementation(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-passive-flux-") as temporary:
            root = Path(temporary).resolve()
            helper = root / "src/numerics/passive_flux.hpp"
            bridge = root / "src/numerics/bridge.hpp"
            riemann = root / "include/pyfluid_amrex/riemann.hpp"
            for path in (helper, bridge, riemann):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            helper.write_text('#include "numerics/bridge.hpp"\n')
            self.assertEqual(ownership_errors(IncludeGraph(root)), [])
            bridge.write_text('#include "pyfluid_amrex/riemann.hpp"\n')
            errors = ownership_errors(IncludeGraph(root))
            self.assertTrue(any(
                "scalar-only flux reaches fluid implementation" in error
                and "passive_flux.hpp -> src/numerics/bridge.hpp ->" in error
                for error in errors
            ))

    def test_excision_kernels_stay_with_pointwise_owners(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-excision-owners-") as temporary:
            root = Path(temporary).resolve()
            files = {
                "src/relativity/black_hole_excision_kernels.hpp": "",
                "src/relativity/black_hole_excision.cpp":
                    '#include "black_hole_excision_kernels.hpp"\n',
                "src/relativity/relativistic_flux_limiter.cpp":
                    '#include "black_hole_excision_kernels.hpp"\n',
                "src/amr/unrelated.cpp": "",
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            self.assertEqual(ownership_errors(IncludeGraph(root)), [])
            (root / "src/amr/unrelated.cpp").write_text(
                '#include "relativity/black_hole_excision_kernels.hpp"\n'
            )
            errors = ownership_errors(IncludeGraph(root))
            self.assertEqual(len(errors), 1)
            self.assertIn("src/amr/unrelated.cpp ->", errors[0])

    def test_rmhd_launch_interface_cannot_reimport_its_kernel(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-rmhd-owners-") as temporary:
            root = Path(temporary).resolve()
            files = {
                "src/relativity/rmhd_correction_flux.hpp": "",
                "src/relativity/rmhd_correction_flux_impl.hpp": "",
                "src/relativity/rmhd_correction_flux_llf.cpp":
                    '#include "rmhd_correction_flux_impl.hpp"\n',
                "src/relativity/relativistic_flux_limiter.cpp":
                    '#include "rmhd_correction_flux.hpp"\n',
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            self.assertEqual(ownership_errors(IncludeGraph(root)), [])
            (root / "src/relativity/rmhd_correction_flux.hpp").write_text(
                '#include "rmhd_correction_flux_impl.hpp"\n'
            )
            errors = ownership_errors(IncludeGraph(root))
            self.assertTrue(any("declaration-only interface" in error for error in errors))
            self.assertTrue(any("relativistic_flux_limiter.cpp ->" in error
                                and "rmhd_correction_flux_impl.hpp" in error
                                for error in errors))

    def test_budget_failure_is_distinct_from_compile_failure(self):
        self.assertEqual(compile_status(0, 50, None), "pass")
        self.assertEqual(compile_status(0, 20, 20), "pass")
        self.assertEqual(compile_status(0, 20.1, 20), "slow")
        self.assertEqual(compile_status(1, 2, 20), "fail")

    def test_vector_leaves_cannot_leak_through_eos_or_timestep_policy(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-vector-owners-") as temporary:
            root = Path(temporary).resolve()
            files = {
                "include/pyfluid_amrex/kernels_generated/vector/magnetic.hpp": "",
                "include/pyfluid_amrex/kernels_generated/vector/viscous.hpp": "",
                "include/pyfluid_amrex/kernels_generated/vector_ops.hpp":
                    '#include "vector/magnetic.hpp"\n#include "vector/viscous.hpp"\n',
                "include/pyfluid_amrex/eos/equation_of_state.hpp":
                    '#include "pyfluid_amrex/kernels_generated/vector/magnetic.hpp"\n',
                "src/flux_kernels/pwc.cpp":
                    '#include "pyfluid_amrex/eos/equation_of_state.hpp"\n',
                "src/relativity/signal_speed_factory.cpp":
                    '#include "pyfluid_amrex/eos/equation_of_state.hpp"\n',
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            self.assertEqual(ownership_errors(IncludeGraph(root)), [])
            (root / "include/pyfluid_amrex/eos/equation_of_state.hpp").write_text(
                '#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"\n'
            )
            errors = ownership_errors(IncludeGraph(root))
            self.assertEqual(len(errors), 4)  # EoS, timestep, two production umbrella users.
            self.assertTrue(any("EoS reaches unrelated" in error for error in errors))
            self.assertTrue(any("timestep policy reaches viscous RHS" in error for error in errors))

    def test_profile_retargets_outputs_without_changing_optimization(self):
        arguments = ["/opt/bin/mpicxx", "-O3", "-arch", "arm64", "-mcpu=native",
                     "-MD", "-MF", "working.d", "-MTworking.o", "-c", "source name.cpp",
                     "-o", "working.o"]
        command = isolated_command({"arguments": arguments}, Path("isolated.o"), Path("trace.json"))
        self.assertEqual(command, arguments[:5] + ["-c", "source name.cpp", "-o", "isolated.o",
                                                 "-ftime-trace=trace.json",
                                                 "-ftime-trace-granularity=10000"])
        self.assertEqual(arguments[-1], "working.o")
        for extra in ("@response", "-save-temps", "-Wp,-MD,working.d"):
            with self.assertRaises(ValueError):
                isolated_command({"arguments": arguments + [extra]}, Path("isolated.o"))
        cuda = ["/opt/cuda/bin/nvcc", "-O3", "-arch=sm_86", "-rdc=false",
                "--fdevice-time-trace=working.json", "-MD", "-MF", "working.d",
                "-c", "source.cpp", "-o", "working.o"]
        self.assertEqual(
            isolated_command({"arguments": cuda}, Path("isolated.o"), Path("trace.json")),
            cuda[:4] + ["-c", "source.cpp", "-o", "isolated.o",
                        "--fdevice-time-trace=trace.json"],
        )
        aliases = [cuda[0], "-O3", "-opt-info=inline", "--generate-dependencies-with-compile",
                   "--dependency-output=working.d", "--compile", "source.cpp",
                   "--output-file", "working.o"]
        self.assertEqual(
            isolated_command({"arguments": aliases}, Path("isolated.o")),
            aliases[:3] + ["--compile", "source.cpp", "-o", "isolated.o"],
        )
        for extra in ("--keep", "-keep-dir=working", "-optf=response", "--time=working.csv"):
            with self.assertRaises(ValueError):
                isolated_command({"arguments": cuda + [extra]}, Path("isolated.o"))

    def test_profile_selects_dimension_and_rejects_ambiguous_owners(self):
        entry = {"file": "src/a.cpp", "directory": "/project",
                 "command": "clang++ -DAMREX_SPACEDIM=3 -c src/a.cpp -o a.o"}
        self.assertIs(select_entry([entry], Path("/project/src/a.cpp"), 3), entry)
        for entries, dimension in (([entry], 2), ([entry, entry], 3)):
            with self.assertRaises(ValueError):
                select_entry(entries, Path("/project/src/a.cpp"), dimension)

    def test_trace_uses_inclusive_totals_without_double_counting(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-trace-") as temporary:
            trace = Path(temporary) / "trace.json"
            trace.write_text(json.dumps({"traceEvents": [
                {"name": "Frontend", "dur": 1_000_000},
                {"name": "Total Frontend", "dur": 1_000_000},
                {"name": "Total InstantiateFunction", "dur": 500_000},
                {"name": "Total Backend", "dur": 4_000_000},
            ]}))
            self.assertEqual(trace_totals(trace),
                             {"Frontend": 1, "InstantiateFunction": 0.5, "Backend": 4})
            trace.write_text(json.dumps({"otherData": {"isNVRTC": 0}, "traceEvents": [
                {"name": "process_name", "pid": 1, "args": {"name": "cicc"}},
                {"name": "frontend", "ph": "X", "pid": 1, "ts": 0, "dur": 1_000_000},
                {"name": "nested", "ph": "X", "pid": 1, "ts": 100_000, "dur": 500_000},
                {"name": "optimizer", "ph": "X", "pid": 1, "ts": 1_000_000, "dur": 2_000_000},
                {"name": "process_name", "pid": 2, "args": {"name": "ptxas"}},
                {"name": "total", "ph": "X", "pid": 2, "ts": 2_000_000, "dur": 4_000_000},
            ]}))
            self.assertEqual(trace_totals(trace),
                             {"cicc (process 1)": 3, "ptxas (process 2)": 4})


class VectorHeaderTests(unittest.TestCase):
    def test_formula_edit_changes_only_its_generated_leaf(self):
        before = emit_cpp.render_vector_headers()
        umbrella = emit_cpp.render_vector_header()
        kernels = list(VECTOR_KERNEL_GROUPS["viscous"]())
        kernels[0] = replace(kernels[0], expression=kernels[0].expression + 1)
        with patch.dict(VECTOR_KERNEL_GROUPS, viscous=lambda: tuple(kernels)):
            after = emit_cpp.render_vector_headers()
            self.assertEqual(umbrella, emit_cpp.render_vector_header())
        self.assertEqual([name for name in before if before[name] != after[name]],
                         ["viscous"])
        names = [kernel.name for factory in VECTOR_KERNEL_GROUPS.values()
                 for kernel in factory()]
        self.assertEqual(len(names), len(set(names)))

    def test_unchanged_vector_leaves_keep_their_timestamps(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-vector-headers-") as temporary:
            root = Path(temporary)
            header = root / "vector_ops.hpp"
            emit_cpp.write_vector_header(header)
            files = [header, *sorted((root / "vector").glob("*.hpp"))]
            before = [path.stat().st_mtime_ns for path in files]
            emit_cpp.write_vector_header(header)
            self.assertEqual(before, [path.stat().st_mtime_ns for path in files])
            self.assertEqual(len(files), len(VECTOR_KERNEL_GROUPS) + 1)


class DiagnosticHeaderTests(unittest.TestCase):
    def test_metadata_edits_do_not_change_value_interface(self):
        types_before = emit_catalog.render_diagnostic_types_header()
        catalog_before = emit_catalog.render_diagnostic_catalog_header()
        diagnostics = list(emit_catalog.RUNTIME_SCALAR_DIAGNOSTICS)
        diagnostics[0] = replace(diagnostics[0], label="Different display label")
        with patch.object(emit_catalog, "RUNTIME_SCALAR_DIAGNOSTICS", diagnostics):
            self.assertEqual(types_before, emit_catalog.render_diagnostic_types_header())
            self.assertNotEqual(catalog_before, emit_catalog.render_diagnostic_catalog_header())
        self.assertNotIn("std::string_view", types_before)
        self.assertNotIn("struct ScalarDiagnostics", catalog_before)

    def test_unchanged_generated_header_keeps_timestamp(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-catalog-") as temporary:
            header = Path(temporary) / "diagnostic_types.hpp"
            content = emit_catalog.render_diagnostic_types_header()
            emit_catalog._write_if_changed(header, content)
            before = header.stat().st_mtime_ns
            emit_catalog._write_if_changed(header, content)
            self.assertEqual(before, header.stat().st_mtime_ns)


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class SharedObjectTests(unittest.TestCase):
    def test_reuse_is_dimension_local_and_does_not_build_unrelated_code(self):
        self.check_shared_objects("Unix Makefiles")

    @unittest.skipUnless(shutil.which("ninja"), "Ninja is required")
    def test_ninja_reuses_objects_without_relinking_an_unchanged_build(self):
        self.check_shared_objects("Ninja")

    def check_shared_objects(self, generator):
        with tempfile.TemporaryDirectory(prefix="pyfluid-shared-objects-") as temporary:
            source = Path(temporary)
            build = source / "build"
            (source / "src").mkdir()
            (source / "src" / "shared.cpp").write_text(
                "int value() { return DIM + VALUE_OFFSET; }\n"
            )
            (source / "src" / "unused.cpp").write_text("#error Should not be built\n")
            (source / "main.cpp").write_text(
                '#include <iostream>\nint value();\n'
                'int main() { std::cout << value(); }\n'
            )
            module = ROOT / "cmake" / "PyFluidSharedObjects.cmake"
            (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.25)
project(shared_object_test LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
function(pyfluid_amrex_configure_target target dim)
   target_compile_definitions(${{target}} PRIVATE DIM=${{dim}})
   set_property(TARGET ${{target}} PROPERTY PYFLUID_AMREX_DIM ${{dim}})
endfunction()
foreach(dim 1 2)
   foreach(name first second)
      add_executable(${{name}}_${{dim}}d main.cpp src/shared.cpp)
      pyfluid_amrex_configure_target(${{name}}_${{dim}}d ${{dim}})
      target_compile_definitions(${{name}}_${{dim}}d PRIVATE VALUE_OFFSET=10)
   endforeach()
endforeach()
add_executable(unused EXCLUDE_FROM_ALL main.cpp src/unused.cpp)
pyfluid_amrex_configure_target(unused 3)
include("{module}")
pyfluid_amrex_share_implementation_objects()
""")

            def run(*command):
                return subprocess.run(
                    command, check=True, capture_output=True, text=True,
                ).stdout

            run("cmake", "-S", str(source), "-B", str(build), "-G", generator)
            run("cmake", "--build", str(build), "--parallel", "2")
            commands = json.loads((build / "compile_commands.json").read_text())
            shared = [entry for entry in commands if entry["file"].endswith("/shared.cpp")]
            self.assertEqual(len(shared), 2)
            self.assertTrue(all("pyfluid_amrex_shared_" in entry["command"] for entry in shared))
            binaries = []
            for dim in (1, 2):
                for name in ("first", "second"):
                    binary = build / f"{name}_{dim}d"
                    binaries.append(binary)
                    self.assertEqual(run(str(binary)), str(dim + 10))
            self.assertFalse((build / "unused").exists())
            timestamps = [binary.stat().st_mtime_ns for binary in binaries]
            run("cmake", "--build", str(build), "--parallel", "2")
            self.assertEqual(timestamps, [binary.stat().st_mtime_ns for binary in binaries])

            # A target with private instrumentation must not silently inherit
            # an object built with another target's compilation options.
            configuration = source / "CMakeLists.txt"
            configuration.write_text(configuration.read_text().replace(
                f'include("{module}")',
                'target_compile_definitions(second_1d PRIVATE PRIVATE_CHECK=1)\n'
                f'include("{module}")',
            ))
            run("cmake", "-S", str(source), "-B", str(build))
            run("cmake", "--build", str(build), "--parallel", "2")
            commands = json.loads((build / "compile_commands.json").read_text())
            shared = [entry for entry in commands if entry["file"].endswith("/shared.cpp")]
            self.assertEqual(len(shared), 3)
            self.assertEqual(sum("pyfluid_amrex_shared_" in e["command"] for e in shared), 1)


if __name__ == "__main__":
    unittest.main()
