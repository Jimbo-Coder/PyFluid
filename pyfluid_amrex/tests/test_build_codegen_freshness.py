"""Shared source generation across CMake profiles, without heavy symbolic cases."""

from __future__ import annotations

import copy
from importlib.metadata import PackageNotFoundError
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

AMREX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AMREX_ROOT / "codegen"))
from generation_cache import generate_if_needed, generation_lock, generator_environment


class GenerationCacheTests(unittest.TestCase):
    def test_family_isolation_and_unchanged_output_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = [root / "diagnostic.py", root / "evolution.py"]
            for path in dependencies:
                path.write_text("first")
            calls = [0, 0]

            def check(index):
                output = root / f"family{index}.hpp"

                def generate(replacements):
                    calls[index] += 1
                    replacements[output].parent.mkdir(parents=True, exist_ok=True)
                    replacements[output].write_text("same complete output")

                return generate_if_needed(root / f"family{index}.json", [dependencies[index]],
                                          [output], {}, generate)

            self.assertTrue(check(0))
            self.assertTrue(check(1))
            output_time = (root / "family0.hpp").stat().st_mtime_ns
            original_stat = dependencies[0].stat()
            dependencies[0].write_text("other")
            os.utime(dependencies[0], ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            self.assertTrue(check(0))
            self.assertFalse(check(1))
            self.assertEqual(calls, [2, 1])
            self.assertEqual((root / "family0.hpp").stat().st_mtime_ns, output_time)

    def test_failed_or_incomplete_generation_never_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependency = root / "formula.py"
            dependency.write_text("first")
            output = root / "output.hpp"
            manifest = root / "family.json"

            def complete(replacements):
                replacements[output].write_text("accepted")

            generate_if_needed(manifest, [dependency], [output], {}, complete)
            accepted_manifest = manifest.read_bytes()
            dependency.write_text("second")

            def interrupted(replacements):
                replacements[output].write_text("incomplete")
                raise RuntimeError("injected writer failure")

            with self.assertRaisesRegex(RuntimeError, "writer failure"):
                generate_if_needed(manifest, [dependency], [output], {}, interrupted)
            with self.assertRaisesRegex(RuntimeError, "output declaration mismatch"):
                generate_if_needed(manifest, [dependency], [output], {}, lambda _: None)
            self.assertEqual(output.read_text(), "accepted")
            self.assertEqual(manifest.read_bytes(), accepted_manifest)

    def test_input_change_during_generation_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependency = root / "formula.py"
            dependency.write_text("before")
            output = root / "output.hpp"

            def changed(replacements):
                replacements[output].write_text("stale")
                dependency.write_text("after")

            with self.assertRaisesRegex(RuntimeError, "dependencies changed"):
                generate_if_needed(root / "family.json", [dependency], [output], {}, changed)
            self.assertFalse(output.exists())
            self.assertFalse((root / "family.json").exists())

    def test_family_lock_has_a_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "family.lock"
            with generation_lock(path, 1.0):
                with self.assertRaises(TimeoutError):
                    with generation_lock(path, 0.05):
                        self.fail("a second owner acquired the family lock")


@unittest.skipUnless(shutil.which("cmake"), "CMake is required for profile integration")
class CMakeGenerationTests(unittest.TestCase):
    def test_two_profiles_share_real_generation_and_repair_outputs(self):
        self.check_two_profiles("Unix Makefiles")

    @unittest.skipUnless(shutil.which("ninja"), "Ninja is required for profile integration")
    def test_ninja_profiles_repair_outputs_without_rebuilding_current_consumers(self):
        self.check_two_profiles("Ninja")

    def check_two_profiles(self, generator):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "project/pyfluid_amrex"
            source.mkdir(parents=True)
            shutil.copytree(AMREX_ROOT / "codegen", source / "codegen",
                            ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(AMREX_ROOT / "schema", source / "schema")
            for name in ("pyproject.toml", "uv.lock"):
                shutil.copyfile(AMREX_ROOT.parent / name, source.parent / name)
            # The test uses its current interpreter and installed SymPy, avoiding
            # an unrelated uv environment installation in the temporary project.
            uv_shim = root / "uv-shim"
            uv_shim.write_text(
                f"#!{sys.executable}\nimport os,sys\n"
                "assert sys.argv[1:4] == ['run','--locked','python'], sys.argv\n"
                "os.execv(sys.executable,[sys.executable,*sys.argv[4:]])\n")
            uv_shim.chmod(0o755)
            cmake = (AMREX_ROOT / "CMakeLists.txt").read_text()
            start = cmake.index("   function(pyfluid_amrex_add_codegen_check family)")
            end = cmake.index("   endfunction()", start) + len("   endfunction()")
            helper = cmake[start:end]
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\nproject(CodegenFreshness LANGUAGES CXX)\n"
                "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n"
                f'set(PYFLUID_UV_EXECUTABLE "{uv_shim}")\n'
                'set(PYFLUID_AMREX_CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_kernels.py")\n'
                + helper + "\n"
                "pyfluid_amrex_add_codegen_check(eos\n"
                " OUTPUTS ${CMAKE_CURRENT_SOURCE_DIR}/generated_eos.hpp\n"
                " ARGUMENTS --family eos --eos-header ${CMAKE_CURRENT_SOURCE_DIR}/generated_eos.hpp\n"
                " DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/codegen/pyfluid_codegen/eos_equations.py)\n"
                # A real compiled consumer tests CMake/Ninja BYPRODUCT restat
                # without requiring AMReX to compile the generated EoS header.
                "add_executable(consumer main.cpp)\n"
                "add_dependencies(consumer pyfluid_amrex_codegen_eos)\n"
                "set_source_files_properties(main.cpp PROPERTIES\n"
                " OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generated_eos.hpp)\n")
            (source / "main.cpp").write_text("int main() { return 0; }\n")
            profiles = [root / "profile-a", root / "profile-b"]
            for profile in profiles:
                subprocess.run(["cmake", "-S", str(source), "-B", str(profile),
                                "-G", generator], capture_output=True, text=True,
                               timeout=30, check=True)

            def build(profile):
                return subprocess.run(
                    ["cmake", "--build", str(profile), "--target", "consumer"],
                    capture_output=True, text=True, timeout=30, check=True).stdout

            # Cold concurrent profiles must serialize and publish only once.
            commands = [["cmake", "--build", str(profile), "--target", "consumer"]
                        for profile in profiles]
            processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True) for command in commands]
            try:
                outputs = [process.communicate(timeout=30)[0] for process in processes]
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            self.assertEqual([process.returncode for process in processes], [0, 0], outputs)
            self.assertEqual(sum("Codegen generated: eos" in output for output in outputs), 1)
            self.assertEqual(sum("Codegen current: eos" in output for output in outputs), 1)
            consumer_artifacts = []
            for profile in profiles:
                entry, = json.loads((profile / "compile_commands.json").read_text())
                consumer_artifacts.extend((profile / entry["output"], profile / "consumer"))
            consumer_times = [path.stat().st_mtime_ns for path in consumer_artifacts]
            header = source / "generated_eos.hpp"
            content = header.read_bytes()
            timestamp = header.stat().st_mtime_ns
            manifest = source / ".cache/codegen/eos.json"
            first_manifest = manifest.read_bytes()
            symbolic_import_trap = source / "codegen/sympy.py"
            symbolic_import_trap.write_text("raise RuntimeError('fresh check imported SymPy')\n")
            self.assertIn("Codegen current: eos", build(profiles[1]))
            symbolic_import_trap.unlink()
            self.assertEqual(manifest.read_bytes(), first_manifest)
            self.assertEqual(header.stat().st_mtime_ns, timestamp)
            self.assertEqual([path.stat().st_mtime_ns for path in consumer_artifacts],
                             consumer_times)
            for filename in ("pyproject.toml", "uv.lock"):
                path = source.parent / filename
                path.write_text(path.read_text() + "\n# unrelated project dependency edit\n")
            self.assertIn("Codegen current: eos", build(profiles[0]))
            self.assertEqual(manifest.read_bytes(), first_manifest)
            self.assertEqual([path.stat().st_mtime_ns for path in consumer_artifacts],
                             consumer_times)

            # Edit a real symbolic dependency without relying on mtime changes.
            dependency = source / "codegen/pyfluid_codegen/eos_equations.py"
            previous = dependency.stat()
            dependency.write_text(dependency.read_text() + "\n# dependency edit\n")
            os.utime(dependency, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            self.assertIn("Codegen generated: eos", build(profiles[0]))
            self.assertNotEqual(manifest.read_bytes(), first_manifest)
            self.assertEqual(header.stat().st_mtime_ns, timestamp)
            self.assertEqual([path.stat().st_mtime_ns for path in consumer_artifacts],
                             consumer_times)

            previous = header.stat()
            header.write_bytes(content + b"\n// corrupt output\n")
            os.utime(header, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            self.assertIn("Codegen generated: eos", build(profiles[1]))
            self.assertEqual(header.read_bytes(), content)
            header.unlink()
            self.assertIn("Codegen generated: eos", build(profiles[0]))
            self.assertEqual(header.read_bytes(), content)
            self.assertEqual(len(json.loads(manifest.read_text())["outputs"]), 1)


class GeneratorEnvironmentTests(unittest.TestCase):
    def test_only_symbolic_runtime_packages_enter_fingerprint(self):
        packages = {"sympy": "1.14.0", "mpmath": "1.3.0", "certifi": "old"}
        with patch("generation_cache.package_version", side_effect=packages.__getitem__) as lookup:
            first = generator_environment()
            packages["certifi"] = "new"
            self.assertEqual(generator_environment(), first)
        self.assertEqual([call.args[0] for call in lookup.call_args_list],
                         ["sympy", "mpmath", "sympy", "mpmath"])
        self.assertEqual(first["python_implementation"], sys.implementation.name)
        self.assertEqual(first["python_version"], list(sys.version_info))
        self.assertEqual(first["implementation_version"], list(sys.implementation.version))

    def test_each_environment_change_invalidates_and_v1_is_not_migrated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependency = root / "formula.py"
            dependency.write_text("same")
            output = root / "output.hpp"
            manifest = root / "family.json"
            environment = generator_environment()
            calls = []

            def generate(replacements):
                calls.append(True)
                replacements[output].write_text("same output")

            def check(value):
                with patch("generation_cache.generator_environment", return_value=value):
                    return generate_if_needed(manifest, [dependency], [output], {}, generate)

            self.assertTrue(check(environment))
            self.assertFalse(check(environment))
            for field in ("sympy", "mpmath", "python_implementation",
                          "python_version", "implementation_version"):
                changed = copy.deepcopy(environment)
                changed[field] = "different"
                self.assertTrue(check(changed), field)
                self.assertFalse(check(changed), field)
                self.assertTrue(check(environment), field)
            old = json.loads(manifest.read_text())
            old["version"] = 1
            del old["environment"]
            manifest.write_text(json.dumps(old))
            before = len(calls)
            self.assertTrue(check(environment))
            self.assertEqual(len(calls), before + 1)
            self.assertEqual(json.loads(manifest.read_text())["version"], 2)

    def test_missing_metadata_and_mid_generation_changes_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependency = root / "formula.py"
            dependency.write_text("same")
            output = root / "output.hpp"
            manifest = root / "family.json"

            def generate(replacements):
                replacements[output].write_text("unpublished")

            with patch("generation_cache.package_version", side_effect=PackageNotFoundError("sympy")):
                with self.assertRaises(PackageNotFoundError):
                    generate_if_needed(manifest, [dependency], [output], {}, generate)
            self.assertFalse(output.exists())
            first = generator_environment()
            second = {**first, "sympy": "different"}
            with patch("generation_cache.generator_environment", side_effect=[first, second]):
                with self.assertRaisesRegex(RuntimeError, "environment changed"):
                    generate_if_needed(manifest, [dependency], [output], {}, generate)
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())


class CsePruningTests(unittest.TestCase):
    def test_guards_preserve_controlled_outputs_and_skip_unneeded_use_counts(self):
        import sympy as sp
        from pyfluid_codegen import emit_cpp

        x, y = sp.symbols("x y")
        first, large, small = sp.symbols("tmp0 tmp1 tmp2")
        large_expression = sp.sin(x) + sp.cos(y) + x * y + x**3
        replacements = [(first, x), (large, large_expression), (small, x + y)]
        outputs = [first + large + small, small]
        with patch.object(emit_cpp.sp, "cse", return_value=(replacements, outputs)):
            with patch.object(emit_cpp, "_symbol_count", wraps=emit_cpp._symbol_count) as count:
                actual = emit_cpp._optimize((x + y,), emit_cpp.CsePolicy())
            self.assertEqual(actual, (replacements[1:], [x + large + small, small]))
            self.assertEqual(count.call_count, 1)
            with patch.object(emit_cpp, "_symbol_count", wraps=emit_cpp._symbol_count) as count:
                actual = emit_cpp._optimize(
                    (x + y,), emit_cpp.CsePolicy(hard_drop_ops=10, soft_drop_ops=3))
            self.assertEqual(actual, ([], [2 * x + y + large_expression, x + y]))
            self.assertEqual(count.call_count, 0)

    def test_cheap_real_kernels_keep_existing_generated_bytes(self):
        from pyfluid_codegen import diagnostic_equations, emit_cpp

        self.assertEqual(emit_cpp.render_eos_header(),
                         (AMREX_ROOT / "include/pyfluid_amrex/kernels_generated/eos.hpp").read_text())
        diagnostics = (AMREX_ROOT / "include/pyfluid_amrex/kernels_generated/diagnostics.hpp").read_text()
        for builder in (diagnostic_equations.conserved_diagnostic_integrands_kernel,
                        diagnostic_equations.baryon_flux_integrands_kernel,
                        diagnostic_equations.matter_energy_flux_integrands_kernel):
            self.assertIn(emit_cpp._emit_struct_kernel(builder(), emit_cpp.CsePolicy()), diagnostics)


@unittest.skipUnless(shutil.which("uv"), "uv is required for locked invocation")
class LockedInvocationTests(unittest.TestCase):
    def test_locked_temp_project_never_rewrites_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            project = root / "pyproject.toml"
            project.write_text('[project]\nname="codegen-lock-test"\nversion="0.0.0"\n'
                               'requires-python=">=3.13"\ndependencies=[]\n')
            environment = dict(os.environ, UV_CACHE_DIR=str(root / "uv-cache"),
                               UV_PYTHON_DOWNLOADS="never")
            subprocess.run(["uv", "lock", "--offline", "--python", sys.executable],
                           cwd=root, env=environment, capture_output=True, text=True,
                           check=True, timeout=30)
            lock = root / "uv.lock"
            before = lock.read_bytes()
            command = ["uv", "run", "--locked", "--offline", "--python", sys.executable,
                       "python", "-c", "pass"]
            subprocess.run(command, cwd=root, env=environment, capture_output=True,
                           text=True, check=True, timeout=30)
            self.assertEqual(lock.read_bytes(), before)
            project.write_text(project.read_text().replace('dependencies=[]',
                                                           'dependencies=["mpmath>=1.3"]'))
            result = subprocess.run(command, cwd=root, env=environment,
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(lock.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
