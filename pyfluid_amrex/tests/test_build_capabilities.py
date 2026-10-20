"""Focused build ownership and preflight checks without compiling the solver."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_case
import pyfluid_machine
from machine_common import load_machine


class BuildCapabilityTests(unittest.TestCase):
    def test_failed_preflight_preserves_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "runs" / "case"
            output.mkdir(parents=True)
            scalars = output / "scalars.tsv"
            scalars.write_text("retained evidence\n")
            source = root / "input.in"
            source.write_text(
                'amr.n_cell = 8\npyfluid.run_name = "case"\n'
                'pyfluid.output_dir = "runs/case"\npyfluid.recon = wenoz\n'
            )
            capabilities = {"schema_version": 1, "dimension": 1}
            with (
                patch.object(run_case, "ROOT", root),
                patch.object(run_case, "amrex_executable", return_value=Path(sys.executable)),
                patch.object(run_case.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps(capabilities)
                )),
                patch.object(run_case, "run_mpi", return_value=SimpleNamespace(
                    returncode=1, stdout="requires PYFLUID_AMREX_ENABLE_RECON_WENOZ=ON"
                )) as preflight,
                patch.object(run_case, "run_amrex") as evolution,
                patch.object(sys, "argv", ["run_case.py", str(source)]),
            ):
                with self.assertRaisesRegex(ValueError, "RECON_WENOZ=ON") as error:
                    run_case.main()
                self.assertIn("pyfluid_machine.py", str(error.exception))
                self.assertIn("build --dim 1", str(error.exception))
                self.assertEqual(scalars.read_text(), "retained evidence\n")
                self.assertFalse((output / "build.json").exists())
                evolution.assert_not_called()
                self.assertEqual(preflight.call_args.args[1], ["--check-build", source])
                self.assertEqual(preflight.call_args.kwargs["ranks"], 1)

    def test_identity_belongs_to_the_selected_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "selected"
            binary.write_bytes(b"selected executable contents")
            capabilities = {"schema_version": 1, "dimension": 3, "backend": "NONE"}
            with (
                patch.object(run_case.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps(capabilities)
                )),
                patch.object(run_case, "run_mpi", return_value=SimpleNamespace(returncode=0)),
            ):
                identity = run_case.check_executable_build(binary, Path("input.in"), None, None)
            self.assertEqual(identity["capabilities"], capabilities)
            self.assertEqual(identity["executable"], str(binary.resolve()))
            self.assertEqual(identity["executable_sha256"], hashlib.sha256(binary.read_bytes()).hexdigest())

    def test_machine_configuration_uses_the_explicit_preset_and_directory(self):
        profile = load_machine("local_dev_rmhd")
        with patch.object(pyfluid_machine.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(pyfluid_machine.configure(profile), 0)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--preset") + 1], "focused-rmhd-pwl")
        self.assertEqual(command[command.index("-B") + 1], profile.expand(str(profile.build["build_dir"])))
        self.assertEqual(profile.build["dimensions"], [1])

    def test_old_or_renamed_executable_is_rejected_before_mpi(self):
        for returncode, report in ((2, ""), (0, '{"schema_version":1,"dimension":3}')):
            with (
                self.subTest(returncode=returncode),
                patch.object(run_case.subprocess, "run", return_value=SimpleNamespace(
                    returncode=returncode, stdout=report,
                )),
                patch.object(run_case, "run_mpi") as preflight,
            ):
                with self.assertRaisesRegex(ValueError, "configure"):
                    run_case.check_executable_build(
                        Path("selected"), Path("input.in"), None, None, expected_dimension=1,
                    )
                preflight.assert_not_called()

    @unittest.skipUnless(shutil.which("cmake"), "CMake is unavailable")
    def test_focused_configuration_prunes_actual_production_source_owners(self):
        presets = json.loads((ROOT / "CMakePresets.json").read_text())
        focused = next(preset for preset in presets["configurePresets"]
                       if preset["name"] == "focused-rmhd-pwl")
        with tempfile.TemporaryDirectory(prefix="pyfluid-capabilities-") as temporary:
            temporary = Path(temporary)
            package = temporary / "amrex"
            package.mkdir()
            (package / "AMReXConfig.cmake").write_text(
                "set(AMReX_MPI_FOUND TRUE)\nset(AMReX_MPI TRUE)\n"
                "set(AMReX_HDF5_FOUND TRUE)\nset(AMReX_HDF5 TRUE)\n"
                "set(AMReX_LSOLVERS_FOUND TRUE)\nset(AMReX_PRECISION DOUBLE)\n"
                "set(AMReX_GPU_BACKEND NONE)\n"
                "foreach(dim 1 2 3)\n"
                "add_library(AMReX::amrex_${dim}d INTERFACE IMPORTED)\n"
                "endforeach()\n"
            )

            def configure(name, features):
                build = temporary / name
                command = [
                    "cmake", "-S", str(ROOT), "-B", str(build),
                    f"-DAMReX_DIR={package}", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    "-DPYFLUID_AMREX_NATIVE=OFF", "-DPYFLUID_AMREX_HORIZON_OPENMP=OFF",
                    *(f"-D{key}={value}" for key, value in features.items()),
                ]
                completed = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                entries = json.loads((build / "compile_commands.json").read_text())
                return {Path(entry["file"]).name for entry in entries}, build

            full_sources, _ = configure("full", {})
            focused_sources, build = configure("focused", focused["cacheVariables"])
            for absent in ("pwc.cpp", "wenoz.cpp", "hydro_passive.cpp", "mhd_ct.cpp",
                           "rmhd_correction_flux_hllc.cpp"):
                self.assertIn(absent, full_sources)
                self.assertNotIn(absent, focused_sources)
            for retained in ("pwl.cpp", "relativistic_flux_limiter.cpp", "hydro.cpp",
                             "rmhd_correction_flux_llf.cpp", "rmhd_correction_flux_hll.cpp",
                             "rmhd_correction_flux_hlld.cpp"):
                self.assertIn(retained, focused_sources)
            header = (build / "generated/pyfluid_amrex/build_configuration.hpp").read_text()
            self.assertIn('{"PYFLUID_AMREX_ENABLE_RECON_WENOZ", false}', header)
            self.assertIn('{"PYFLUID_AMREX_ENABLE_RIEMANN_HLLD", true}', header)

            # Primary LLF/HLL switches must not remove internal correction
            # dependencies. Disabling MHD does remove all RMHD launch owners.
            fallback_sources, _ = configure("fallback", {
                **focused["cacheVariables"], "PYFLUID_AMREX_ENABLE_RIEMANN_LLF": "OFF",
                "PYFLUID_AMREX_ENABLE_RIEMANN_HLL": "OFF",
            })
            self.assertIn("rmhd_correction_flux_llf.cpp", fallback_sources)
            self.assertIn("rmhd_correction_flux_hll.cpp", fallback_sources)
            hydro_sources, _ = configure("hydro", {"PYFLUID_AMREX_ENABLE_MHD": "OFF"})
            self.assertFalse(any(name.startswith("rmhd_correction_flux") for name in hydro_sources))


if __name__ == "__main__":
    unittest.main()
