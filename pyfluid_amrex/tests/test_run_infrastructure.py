"""Input, launch failure, and data preservation contracts; no solver required."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_case
from plotfile_reader import read_fab, read_fab_1d
from input_parameters import read_key_values
from validation_common import (
    cartesian_divb_scale, check_scalar_rows, infer_scalar_policy, materialize_run_input,
    precision_scalar_policy, run_bounded_command, run_mpi, scalar_checks, scalar_validation_row,
)


class RunInfrastructureTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_command_deadline_stops_descendant_even_after_launcher_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            child_pid = Path(temporary) / "child.pid"
            # A compiler driver may exit on TERM while its worker ignores it.
            child = ("import os, signal, time; "
                     "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                     "print(os.getpid(), flush=True); time.sleep(30)")
            parent = (
                "import subprocess, sys, time; from pathlib import Path; "
                f"child = subprocess.Popen([sys.executable, '-c', {child!r}], "
                "stdout=subprocess.PIPE, text=True); "
                "Path(sys.argv[1]).write_text(child.stdout.readline()); time.sleep(30)"
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded_command(
                    [sys.executable, "-c", parent, str(child_pid)], timeout=1,
                )
            worker = int(child_pid.read_text())
            process_state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(worker)],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            self.assertTrue(not process_state or process_state.startswith("Z"), process_state)
        result = run_bounded_command([sys.executable, "-c", "raise SystemExit(7)"], timeout=2)
        self.assertEqual(result.returncode, 7)

    def test_precision_divergence_accepts_roundoff_but_rejects_defects(self):
        base = scalar_checks(mass=False, energy=False, momentum=False, performance=False)
        row = {"min_rho": "1", "min_pressure": "1", "divb_l1": "0"}
        for bits, dtype in ((24, np.float32), (53, np.float64)):
            for field in (0.125, 1.0, 8.0):
                for width in (1 / 32, 1 / 512):
                    with self.subTest(bits=bits, field=field, width=width):
                        policy = precision_scalar_policy(base, bits, divb_scale=field / width)
                        # Independent face-difference construction: one representable
                        # field increment, then a resolvable 256-increment defect.
                        increment = float(np.nextafter(dtype(field), dtype(np.inf)) - dtype(field))
                        roundoff = {**row, "divb_l1": str(increment / width)}
                        defect = {**row, "divb_l1": str(256 * increment / width)}
                        self.assertEqual(check_scalar_rows([roundoff], policy, mhd=True), [])
                        self.assertTrue(check_scalar_rows([defect], policy, mhd=True))
                        self.assertTrue(check_scalar_rows(
                            [{**roundoff, "min_pressure": "-1"}], policy, mhd=True))
                        self.assertIsNone(policy.mass_tol)
        for scale in (None, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                precision_scalar_policy(base, 24, divb_scale=scale)
        with self.assertRaises(ValueError):
            precision_scalar_policy(base, 11, divb_scale=1)
        # An explicitly exact check must not be replaced by a default tolerance.
        exact = scalar_checks(mass=False, energy=False, momentum=False,
                              performance=False, divb_tol=0)
        exact = precision_scalar_policy(exact, 24)
        self.assertTrue(check_scalar_rows([{**row, "divb_l1": "1e-16"}], exact, mhd=True))

    def test_divergence_scale_uses_active_widths_and_checks_precision_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "run.in").write_text(
                "amr.n_cell = 8 16\ngeometry.coord_sys = 0\n"
                "geometry.is_periodic = 1 1\n"
                "geometry.prob_lo = 0 0\ngeometry.prob_hi = 2 3\n")
            # Anisotropic refinement changes the smallest active width. The
            # inactive third direction and maximum-width summary must not win.
            (out / "amr_levels.tsv").write_text(
                "finest_dx\tfinest_dx1\tfinest_dx2\tfinest_dx3\n"
                "0.25\t0.25\t0.1875\t0\n0.25\t0.25\t0.046875\t0\n")
            scale = cartesian_divb_scale(out, field_scale=0.75)
            self.assertEqual(scale, 96)  # Volume 6, not a cell-count multiplier.
            (out / "scalars.tsv").write_text(
                "# real_mantissa_bits=24\nmin_rho\tmin_pressure\tdivb_l1\n1\t1\t1e-5\n")
            base = scalar_checks(mass=False, energy=False, momentum=False, performance=False)
            policy = precision_scalar_policy(base, 24, divb_scale=scale)
            with patch("validation_common.ROOT", out.resolve()):
                result = scalar_validation_row(out, checks=policy, mhd=True)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["divb_l1"], 1e-5)
            self.assertAlmostEqual(result["divb_l1_scaled"], 1e-5 / 96)
            (out / "scalars.tsv").write_text((out / "scalars.tsv").read_text().replace("=24", "=53"))
            with patch("validation_common.ROOT", out.resolve()):
                mismatch = scalar_validation_row(out, checks=policy, mhd=True)
            self.assertEqual(mismatch["status"], "FAIL")
            self.assertIn("precision policy expects", mismatch["reason"])
            (out / "run.in").write_text("amr.n_cell = 8 16\ngeometry.coord_sys = 2\n")
            with self.assertRaisesRegex(ValueError, "curvilinear"):
                cartesian_divb_scale(out, field_scale=0.75)

    def test_amrex_segfault_output_terminates_stalled_backtrace(self):
        with (
            patch("validation_common._mpi_prefix", return_value=[]),
            patch("validation_common.load_machine",
                  return_value=SimpleNamespace(environment={})),
        ):
            result = run_mpi(
                Path(sys.executable),
                ["-c", "import time; print('Segfault', flush=True); time.sleep(10)"],
                timeout=3,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.timed_out)
        self.assertIn("MPI process group terminated after fatal rank output", result.stdout)

    def test_native_fab_uses_serialized_precision_and_byte_order(self):
        layouts = {4: "32 8 23 0 1 9 0 127", 8: "64 11 52 0 1 12 0 1023"}
        expected = np.array([[1.25, -2.5], [0.125, 7.0]])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Cell_D_00000"
            for size, layout in layouts.items():
                for endian in ("<", ">"):
                    order = range(size, 0, -1) if endian == "<" else range(1, size + 1)
                    header = (f"FAB ((8, ({layout})),({size}, "
                              f"({' '.join(map(str, order))})))((0) (1) (0)) 2\n")
                    payload = expected.astype(f"{endian}f{size}").tobytes()
                    path.write_bytes(header.encode() + payload)
                    with self.subTest(size=size, endian=endian):
                        np.testing.assert_array_equal(read_fab(path, 0, 1)[2], expected)
                        np.testing.assert_array_equal(read_fab_1d(path, 0)[2], expected)
                        path.write_bytes(header.encode() + payload[:-1])
                        with self.assertRaisesRegex(ValueError, "truncated FAB"):
                            read_fab(path, 0, 1)

    def test_decoded_reporting_and_serialized_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.in"
            generated = Path(temporary) / "generated.in"
            source.write_text(
                'pyfluid.diffusion = "none" # comment\n'
                'pyfluid.output_dir = "runs/a # b=c"\n'
                'pyfluid.run_name = ""\n'
                'amr.n_cell = 8 \\\n 16\n'
                'label = """first "quote"\nsecond"""\n'
                'literal = "a\\b"\n'
            )
            decoded = read_key_values(source)
            self.assertEqual(infer_scalar_policy(decoded, {}, [])[0].name, "closed")
            self.assertEqual(decoded["pyfluid.output_dir"], "runs/a # b=c")
            self.assertEqual(decoded["pyfluid.run_name"], "")
            self.assertEqual(decoded["amr.n_cell"].split(), ["8", "16"])
            self.assertEqual(decoded["label"], 'first "quote"\nsecond')
            self.assertEqual(decoded["literal"], "a\\b")
            materialize_run_input(source, generated, {"amr.n_cell": "4 4"})
            self.assertEqual(read_key_values(generated), {**decoded, "amr.n_cell": "4 4"})
            source.write_text('label = "unfinished\n')
            with self.assertRaisesRegex(ValueError, "unterminated"):
                read_key_values(source)

    def test_cleanup_checks_all_dependencies_before_removing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "runs" / "case"
            output.mkdir(parents=True)
            checkpoint = output / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "Header").write_text("preserve")
            alias = root / "checkpoint_alias"
            alias.symlink_to(checkpoint, target_is_directory=True)
            outside = root / "outside"
            outside.mkdir()
            inside_alias = output / "external_input"
            inside_alias.symlink_to(outside, target_is_directory=True)
            with patch.object(run_case, "ROOT", root):
                for dependency in (checkpoint, alias, output, inside_alias):
                    with self.subTest(dependency=dependency):
                        with self.assertRaisesRegex(ValueError, "overlaps run dependency"):
                            run_case.safe_clean_run_dir(output, dependencies=(dependency,))
                        self.assertEqual((checkpoint / "Header").read_text(), "preserve")
                run_case.safe_clean_run_dir(output, dependencies=(root / "outside",))
                self.assertFalse(output.exists())

    def test_runner_refuses_nested_restart_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "runs" / "case"
            output.mkdir(parents=True)
            checkpoint = output / "checkpoint"
            checkpoint.mkdir()
            source = root / "input.in"
            source.write_text(
                'amr.n_cell = 8\npyfluid.run_name = "case"\n'
                'pyfluid.output_dir = "runs/case"\n'
                'pyfluid.restart_file = "runs/case/checkpoint"\n'
            )
            with (
                patch.object(run_case, "ROOT", root),
                patch.object(run_case, "amrex_executable", return_value=Path(sys.executable)),
                patch.object(run_case, "run_amrex") as launch,
                patch.object(sys, "argv", ["run_case.py", str(source)]),
            ):
                with self.assertRaisesRegex(ValueError, "overlaps run dependency"):
                    run_case.main()
                launch.assert_not_called()
                self.assertTrue(checkpoint.is_dir())


if __name__ == "__main__":
    unittest.main()
