"""Cheap emission/pruning contracts without generating a complete family."""

from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch

AMREX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AMREX_ROOT / "codegen"))

import sympy as sp

from pyfluid_codegen import emit_cpp
from pyfluid_codegen.kernel_equations import StructKernel


def old_optimize(expressions, policy):
    """Frozen pre-saturation algorithm, including exact occurrence counts."""
    replacements, reduced = sp.cse(
        expressions,
        symbols=sp.numbered_symbols("tmp"),
        optimizations="basic",
    )
    replacements = list(replacements)
    reduced = list(reduced)
    changed = True
    while changed:
        changed = False
        kept = []
        for index, (symbol, expression) in enumerate(replacements):
            op_count = int(sp.count_ops(expression))
            hard_drop = op_count <= policy.hard_drop_ops
            soft_drop = False
            if not hard_drop and op_count <= policy.soft_drop_ops:
                later = [rhs for _, rhs in replacements[index + 1:]] + reduced
                count = sum(
                    1 for value in later
                    for node in sp.preorder_traversal(value) if node == symbol
                )
                soft_drop = count <= 1
            if hard_drop or soft_drop:
                replacement = {symbol: expression}
                replacements = kept + [
                    (later_symbol, later_expression.xreplace(replacement))
                    for later_symbol, later_expression in replacements[index + 1:]
                ]
                reduced = [value.xreplace(replacement) for value in reduced]
                changed = True
                break
            kept.append((symbol, expression))
    return replacements, reduced


class SingleEmissionTests(unittest.TestCase):
    def test_relativistic_groups_emit_once_with_identical_wrappers_and_order(self):
        groups = {name: (name,) for name in
                  ("core", "flux", "sources", "c2p_rhd", "c2p_rmhd")}
        bodies = {name: [f"kernel_{name}; amrex::Math::abs(0);"] for name in groups}
        includes = ('#include "pyfluid_amrex/constants.hpp"',
                    '#include "pyfluid_amrex/relativistic_state.hpp"',
                    "#include <cmath>")
        expected = {
            name: emit_cpp._render_generated_group(
                "\n\n".join(bodies[name]),
                includes + (('#include "pyfluid_amrex/c2p_root.hpp"',)
                            if name == "c2p_rmhd" else ()),
            )
            for name in groups
        }
        expected["flux"] = emit_cpp._render_generated_group(
            "\n\n".join([*bodies["flux"], *emit_cpp._relativistic_flux_wrappers()]),
            includes,
        )
        policy = emit_cpp.CsePolicy()
        with patch.object(emit_cpp, "relativistic_kernel_groups", return_value=groups), \
                patch.object(emit_cpp, "_emit_kernel_group", side_effect=
                             lambda kernels, _: list(bodies[kernels[0]])) as emit:
            actual = emit_cpp.render_relativistic_headers(policy)
        self.assertEqual(list(actual), list(groups))
        self.assertEqual(actual, expected)
        self.assertEqual(emit.call_args_list,
                         [call(kernels, policy) for kernels in groups.values()])

    def test_z4c_groups_emit_once_with_identical_wrappers_and_order(self):
        groups = {name: (name,) for name in
                  ("geometry", "constraints", "gauge", "evolution")}
        bodies = {name: [f"kernel_{name};"] for name in groups}
        includes = ('#include "pyfluid_amrex/constants.hpp"',
                    '#include "pyfluid_amrex/z4c_state.hpp"',
                    "#include <AMReX_Math.H>", "#include <cmath>")
        expected = {
            name: emit_cpp._render_generated_group("\n\n".join(bodies[name]), includes)
            for name in groups
        }
        geometry = '#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"'
        gauge = '#include "pyfluid_amrex/kernels_generated/z4c/gauge.hpp"'
        expected["constraints"] = emit_cpp._render_generated_group(
            "\n\n".join([*bodies["constraints"], emit_cpp._z4c_constraint_wrappers()]),
            includes + (geometry,),
        )
        expected["evolution"] = emit_cpp._render_generated_group(
            "\n\n".join([*bodies["evolution"], emit_cpp._z4c_evolution_wrappers()]),
            includes + (geometry, gauge),
        )
        policy = emit_cpp.CsePolicy()
        with patch.object(emit_cpp, "z4c_kernel_groups", return_value=groups), \
                patch.object(emit_cpp, "_emit_kernel_group", side_effect=
                             lambda kernels, _: list(bodies[kernels[0]])) as emit:
            actual = emit_cpp.render_z4c_headers(policy)
        self.assertEqual(list(actual), list(groups))
        self.assertEqual(actual, expected)
        self.assertEqual(emit.call_args_list,
                         [call(kernels, policy) for kernels in groups.values()])


class BoundedUseCountTests(unittest.TestCase):
    def test_count_stops_at_second_occurrence_without_visiting_tail(self):
        temporary, other = sp.symbols("temporary other")
        visited = []

        def traversal(_):
            for node in (other, temporary, other, temporary):
                visited.append(node)
                yield node
            self.fail("visited an irrelevant tail after the second use")

        with patch.object(emit_cpp.sp, "preorder_traversal", side_effect=traversal) as walk:
            self.assertEqual(emit_cpp._symbol_count([other, other], temporary), 2)
        self.assertEqual(visited, [other, temporary, other, temporary])
        self.assertEqual(walk.call_count, 1)

    def test_zero_single_and_repeated_dag_occurrences_keep_predicate(self):
        temporary, other = sp.symbols("temporary other")
        expressions = ([other], [temporary], [temporary, temporary],
                       [sp.Add(temporary, temporary, evaluate=False)],
                       [temporary, other, temporary, temporary])
        for values in expressions:
            with self.subTest(values=values):
                exact = sum(1 for value in values
                            for node in sp.preorder_traversal(value)
                            if node == temporary)
                self.assertEqual(emit_cpp._symbol_count(values, temporary), min(exact, 2))

    def test_old_optimizer_order_and_emitted_bytes_across_policies(self):
        x, y, z = sp.symbols("x y z")
        common = x + y
        expressions = (
            (x, y),
            (common**2 + sp.sin(common), common * z + sp.cos(common)),
            ((x + y) * (x - y), (x + y)**3, sp.sin((x + y) * (x - y))),
            (sp.sqrt(x**2 + y**2) + z, (x**2 + y**2) / (z + 1)),
        )
        policies = (emit_cpp.CsePolicy(), emit_cpp.CsePolicy(0, 0),
                    emit_cpp.CsePolicy(0, 1), emit_cpp.CsePolicy(1, 3),
                    emit_cpp.CsePolicy(2, 0), emit_cpp.CsePolicy(10, 3))
        for values in expressions:
            kernel = StructKernel(
                name="probe", return_type="Probe",
                arguments=(("amrex::Real", "x"), ("amrex::Real", "y"),
                           ("amrex::Real", "z")),
                outputs=tuple(f"out{index}" for index in range(len(values))),
                expressions=values,
            )
            for policy in policies:
                with self.subTest(values=values, policy=policy):
                    self.assertEqual(emit_cpp._optimize(values, policy),
                                     old_optimize(values, policy))
                    actual = emit_cpp._emit_struct_kernel(kernel, policy)
                    with patch.object(emit_cpp, "_optimize", side_effect=old_optimize):
                        expected = emit_cpp._emit_struct_kernel(kernel, policy)
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
