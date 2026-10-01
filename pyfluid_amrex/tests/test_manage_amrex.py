"""Independent dependency updates, conservative retirement and CMake identity."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import manage_amrex as manager
import manage_metal


class DependencyActivationTests(unittest.TestCase):
    def test_metal_patch_chain_replays_without_overwriting_local_work(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-patch-chain-") as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            def git(*arguments):
                return subprocess.check_output(["git", *arguments], cwd=source)

            git("init", "-q")
            original = source / "input.txt"
            original.write_text("base\n")
            git("add", "input.txt")
            git("-c", "user.name=PyFluid test", "-c", "user.email=test@example.invalid",
                "commit", "-qm", "fixture")
            commit = git("rev-parse", "HEAD").decode().strip()
            patches = []
            for number, (before, after) in enumerate((("base", "middle"), ("middle", "final"))):
                patch_file = root / f"{number}.patch"
                patch_file.write_text(
                    "diff --git a/input.txt b/input.txt\n"
                    "--- a/input.txt\n+++ b/input.txt\n"
                    f"@@ -1 +1 @@\n-{before}\n+{after}\n"
                )
                patches.append(patch_file)
            original.write_text("final\n")
            expected = hashlib.sha256(manage_metal.source_diff(source)).hexdigest()
            original.write_text("base\n")
            manage_metal.prepare_source(source, "unused", commit, patches, expected, build=True)
            self.assertEqual(original.read_text(), "final\n")
            self.assertEqual(git("diff", "--cached"), b"")
            marker = source / "local-work.txt"
            marker.write_text("preserve")
            with self.assertRaisesRegex(RuntimeError, "reviewed patch result"):
                manage_metal.prepare_source(source, "unused", commit, patches, expected, build=True)
            self.assertEqual(original.read_text(), "final\n")
            self.assertEqual(marker.read_text(), "preserve")

    def test_install_contract_rejects_wrong_precision_or_backend(self):
        with tempfile.TemporaryDirectory(prefix="pyfluid-precision-") as temp:
            root = Path(temp)
            config = root / "lib/cmake/AMReX/AMReXConfig.cmake"
            config.parent.mkdir(parents=True)
            common = (
                "set(AMReX_MPI_FOUND ON)\nset(AMReX_OMP_FOUND OFF)\n"
                "set(AMReX_HDF5_FOUND ON)\nset(AMReX_LSOLVERS_FOUND ON)\n"
                "set(AMReX_SPACEDIM 1;2;3)\nset(AMReX_GPU_RDC OFF)\n"
            )
            for dimension in (1, 2, 3):
                (root / f"lib/libamrex_{dimension}d.a").touch()
            for variant in ("mpi", "mpi-fp32", "cuda", "cuda-fp32"):
                options = manager.VARIANTS[variant]
                cuda = "ON" if options.gpu_backend == "CUDA" else "OFF"
                config.write_text(common + f"set(AMReX_PRECISION {options.precision})\n"
                                  + f"set(AMReX_CUDA_FOUND {cuda})\n")
                self.assertEqual(manager.install_contract_errors(root, variant), [])
                if options.gpu_backend == "CUDA":
                    supported = config.read_text()
                    config.write_text(supported.replace(
                        "AMReX_GPU_RDC OFF", "AMReX_GPU_RDC ON",
                    ))
                    self.assertIn(
                        "AMReX_GPU_RDC is not OFF",
                        manager.install_contract_errors(root, variant),
                    )
                    config.write_text(supported)
                for other in ("mpi", "mpi-fp32", "cuda", "cuda-fp32"):
                    if other != variant:
                        self.assertTrue(manager.install_contract_errors(root, other))

    def test_each_variant_updates_without_building_pyfluid(self):
        pin = manager.AmrexPin("unused", "26.09", "1" * 40)
        for variant in manager.VARIANTS:
            with (
                self.subTest(variant=variant),
                patch.object(manager, "build_amrex", return_value=Path("candidate")),
                patch.object(manager, "activate_install"),
                patch.object(manager, "retire_obsolete") as retire,
                patch.object(manager, "run") as run,
            ):
                manager.install_variant(pin, pin, 3, variant, update_pin=False)
                run.assert_not_called()
                retire.assert_called_once_with(pin)

    def test_failed_pin_update_restores_link_without_building_pyfluid(self):
        pin = manager.AmrexPin("unused", "26.08", "0" * 40)
        target = manager.AmrexPin("unused", "26.09", "1" * 40)
        previous = manager.ActivationState("symlink", "previous-install")
        with (
            patch.object(manager, "build_amrex", return_value=Path("candidate")),
            patch.object(manager, "activate_install", return_value=previous),
            patch.object(manager, "restore_install") as restore,
            patch.object(manager, "write_pin", side_effect=RuntimeError("pin failed")),
            patch.object(manager, "retire_obsolete") as retire,
            patch.object(manager, "run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "pin failed"):
                manager.install_variant(pin, target, 3, "mpi", update_pin=True)
            restore.assert_called_once_with(previous, "mpi")
            run.assert_not_called()
            retire.assert_not_called()


class DependencyRetirementTests(unittest.TestCase):
    def test_retirement_preserves_current_active_dirty_and_unknown_trees(self):
        pin = manager.AmrexPin("unused", "26.09", "1" * 40)
        with tempfile.TemporaryDirectory(prefix="pyfluid-dependency-prune-") as temp:
            root = Path(temp)
            with (
                patch.object(manager, "DEPS_DIR", root),
                patch("builtins.print"),
                patch.object(
                    manager, "managed_source_dirty",
                    side_effect=lambda version: version == "26.06",
                ),
            ):
                for version in ("26.06", "26.07", "26.08", "26.09"):
                    manager.source_path(version).mkdir()
                    for variant in manager.VARIANTS:
                        manager.build_path(version, variant).mkdir()
                        include = manager.install_path(version, variant) / "include"
                        include.mkdir(parents=True)
                        (include / "AMReX_Version.H").write_text(
                            f'#define AMREX_GIT_VERSION "{version}"\n'
                        )
                unknown = root / "amrex-private-experiment"
                unknown.mkdir()
                source_link = root / "amrex-26.05"
                source_link.symlink_to(unknown, target_is_directory=True)
                manager.active_install_path("mpi").symlink_to(
                    manager.install_path("26.09", "mpi"), target_is_directory=True
                )
                manager.active_install_path("openmp").symlink_to(
                    manager.install_path("26.08", "openmp"), target_is_directory=True
                )
                before = set(root.iterdir())
                manager.retire_obsolete(pin, dry_run=True)
                self.assertEqual(set(root.iterdir()), before)
                manager.retire_obsolete(pin)
                self.assertTrue(manager.source_path("26.09").exists())
                self.assertTrue(manager.source_path("26.08").exists())
                self.assertTrue(manager.source_path("26.06").exists())
                self.assertFalse(manager.source_path("26.07").exists())
                self.assertFalse(manager.install_path("26.08", "mpi").exists())
                self.assertFalse(manager.build_path("26.08", "mpi").exists())
                self.assertTrue(manager.install_path("26.08", "openmp").exists())
                self.assertTrue(manager.build_path("26.08", "openmp").exists())
                self.assertTrue(unknown.exists())
                self.assertTrue(source_link.is_symlink())
                manager.active_install_path("openmp").unlink()
                manager.active_install_path("openmp").symlink_to(
                    manager.install_path("26.09", "openmp"), target_is_directory=True
                )
                manager.retire_obsolete(pin)
                self.assertFalse(manager.source_path("26.08").exists())
                self.assertFalse(manager.install_path("26.08", "openmp").exists())
                self.assertTrue(manager.source_path("26.06").exists())


@unittest.skipUnless(shutil.which("cmake"), "requires CMake")
class DependencyIdentityTests(unittest.TestCase):
    def test_cmake_switches_older_dependencies_without_cleaning(self):
        # Exercise the production CMake setting, not a second resolver.
        project = (manager.ROOT / "pyfluid_amrex/CMakeLists.txt").read_text()
        setting = re.search(
            r"^set\(CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS TRUE\).*?(?=find_package\(AMReX)",
            project,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(setting)
        self.assertLess(setting.start(), project.index("find_package(AMReX"))

        def command(*args):
            result = subprocess.run(
                args, check=True, text=True, capture_output=True
            )
            return result.stdout

        with tempfile.TemporaryDirectory(prefix="pyfluid-cmake-identity-") as temp:
            root = Path(temp)
            source = root / "source"
            build = root / "build"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.25)\n"
                "project(dependency_identity LANGUAGES CXX)\n"
                f"{setting.group().replace('AMReX_DIR', 'Probe_DIR')}\n"
                "find_package(Probe CONFIG REQUIRED)\n"
                "set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS\n"
                '  "${CMAKE_CURRENT_SOURCE_DIR}/dependency.txt")\n'
                "get_target_property(selected Probe::Probe INTERFACE_INCLUDE_DIRECTORIES)\n"
                'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/selected.txt" "${selected}")\n'
                "add_executable(probe main.cpp)\n"
                "target_link_libraries(probe PRIVATE Probe::Probe)\n"
                "add_custom_command(OUTPUT generated.txt\n"
                "  COMMAND ${CMAKE_COMMAND} -E copy\n"
                "    ${CMAKE_CURRENT_SOURCE_DIR}/payload.txt generated.txt\n"
                "  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/payload.txt)\n"
                "add_custom_target(generate DEPENDS generated.txt)\n"
                "add_dependencies(probe generate)\n"
            )
            (source / "main.cpp").write_text(
                '#include "version.hpp"\n#include <iostream>\n'
                "int main() { std::cout << PROBE_VERSION; }\n"
            )
            (source / "payload.txt").write_text("unchanged generated artifact\n")
            for version in (1, 2):
                package = root / f"version-{version}"
                package.mkdir()
                (package / "ProbeConfig.cmake").write_text(
                    "add_library(Probe::Probe INTERFACE IMPORTED)\n"
                    "set_target_properties(Probe::Probe PROPERTIES\n"
                    '  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}")\n'
                )
                header = package / "version.hpp"
                header.write_text(f"#define PROBE_VERSION {version}\n")
                # Both headers predate every object, reproducing the real bug.
                os.utime(header, (946684800, 946684800))

            active = root / "active"
            generated_mtime = None
            for version in (1, 2, 1):
                if active.is_symlink():
                    # macOS Make 3.81 compares whole-second timestamps. Let
                    # changed compiler flags become newer than the last object.
                    time.sleep(1.05)
                    active.unlink()
                active.symlink_to(root / f"version-{version}", target_is_directory=True)
                (source / "dependency.txt").write_text(str(version))
                if generated_mtime is None:
                    command(
                        "cmake", "-S", str(source), "-B", str(build),
                        "-G", "Unix Makefiles", f"-DProbe_DIR={active}",
                    )
                command("cmake", "--build", str(build), "--parallel", "2")
                selected = Path((build / "selected.txt").read_text())
                self.assertEqual(selected, active.resolve())
                self.assertEqual(
                    command(str(build / "probe")), str(version),
                    (build / "CMakeFiles/probe.dir/flags.make").read_text(),
                )
                artifact_mtime = (build / "generated.txt").stat().st_mtime_ns
                if generated_mtime is None:
                    generated_mtime = artifact_mtime
                self.assertEqual(artifact_mtime, generated_mtime)

            executable_mtime = (build / "probe").stat().st_mtime_ns
            command("cmake", "--build", str(build), "--parallel", "2")
            self.assertEqual((build / "probe").stat().st_mtime_ns, executable_mtime)


if __name__ == "__main__":
    unittest.main()
