#!/usr/bin/env python3
"""Conservative project-header impact and ownership checks, without a build.

Follow literal includes in all preprocessor branches. This is a source-level
upper bound, not the configured compiler dependency graph: system headers,
macro-expanded includes, generation dependencies and disabled CMake owners are
outside its scope. CMake remains the sole build/dependency executor.
"""

from __future__ import annotations

import argparse
from collections import deque
from functools import cache
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INCLUDE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)


class IncludeGraph:
    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        paths = {
            path.resolve()
            for directory in (self.root / "include", self.root / "src")
            for path in directory.rglob("*")
            if path.suffix in (".hpp", ".h", ".cpp")
        }
        self.edges: dict[Path, set[Path]] = {}
        for path in sorted(paths):
            # Ignore commented-out includes; ordinary C++ strings do not
            # contain a preprocessor directive at the beginning of a line.
            source = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.DOTALL)
            dependencies = set()
            for name in INCLUDE.findall(source):
                candidates = (path.parent / name, self.root / "include" / name,
                              self.root / "src" / name)
                for candidate in candidates:
                    resolved = candidate.resolve()
                    if resolved in paths:
                        dependencies.add(resolved)
                        break
            self.edges[path] = dependencies

    @cache
    def paths_from(self, source: Path) -> dict[Path, Path | None]:
        """Breadth-first predecessor map, including the starting file."""
        source = source.resolve()
        predecessors = {source: None}
        pending = deque([source])
        while pending:
            current = pending.popleft()
            for dependency in sorted(self.edges.get(current, ())):
                if dependency not in predecessors:
                    predecessors[dependency] = current
                    pending.append(dependency)
        return predecessors

    def consumers(self, header: Path) -> list[Path]:
        return [path for path in self.edges if path.suffix == ".cpp"
                and header.resolve() in self.paths_from(path)]

    def chain(self, source: Path, dependency: Path) -> str:
        predecessors = self.paths_from(source)
        chain = [dependency.resolve()]
        while predecessors[chain[-1]] is not None:
            chain.append(predecessors[chain[-1]])
        return " -> ".join(str(path.relative_to(self.root)) for path in reversed(chain))


def ownership_errors(graph: IncludeGraph) -> list[str]:
    """Protect declaration boundaries and additive reconstruction ownership."""
    public = graph.root / "include" / "pyfluid_amrex"
    sources = graph.root / "src"
    errors = []
    interfaces = [
        *public.glob("*factory.hpp"),
        public / "black_hole_excision.hpp",
        public / "config.hpp",
        public / "diagnostics_types.hpp",
        public / "method_codes.hpp",
        public / "relativistic_flux_limiter.hpp",
        public / "spacetime_excision.hpp",
        sources / "relativity" / "rmhd_correction_flux.hpp",
    ]
    heavy = {public / "eos" / "equation_of_state.hpp", public / "fixed_spacetime.hpp",
             public / "generated" / "diagnostic_catalog.hpp",
             public / "generated" / "time_integrator_catalog.hpp"}
    for interface in interfaces:
        for dependency in sorted(graph.paths_from(interface)):
            if dependency in heavy or (sources in dependency.parents and dependency != interface) \
                    or "kernels_generated" in dependency.parts:
                errors.append("declaration-only interface reaches implementation: "
                              + graph.chain(interface, dependency))

    # Field identity belongs in generated/field_ids.hpp. Host strings,
    # descriptors and output metadata must not invalidate numerical kernels.
    field_metadata = public / "fields.hpp"
    metadata_free = {
        public / "config.hpp",
        public / "z4c_state_access.hpp",
        public / "fccz4_state_access.hpp",
        sources / "relativity" / "relativistic_flux_limiter.cpp",
        sources / "relativity" / "rmhd_advanced_flux.cpp",
        *(sources / "flux_kernels").glob("*.cpp"),
        *(sources / "relativity").glob("rmhd_correction_flux*.cpp"),
    }
    for owner in sorted(metadata_free):
        if field_metadata in graph.paths_from(owner):
            errors.append("numerical contract reaches host field metadata: "
                          + graph.chain(owner, field_metadata))

    config = public / "config.hpp"
    config_free = {
        public / "relativistic_diode.hpp",
        public / "relativistic_flux_limiter.hpp",
        public / "spacetime_excision.hpp",
        sources / "relativity" / "relativistic_flux_limiter.cpp",
        sources / "relativity" / "rmhd_advanced_flux.cpp",
        *(sources / "flux_kernels").glob("*.cpp"),
        *(sources / "relativity").glob("rmhd_correction_flux*.cpp"),
    }
    for owner in sorted(config_free):
        if config in graph.paths_from(owner):
            errors.append("numerical value contract reaches host configuration: "
                          + graph.chain(owner, config))

    # Optional tracer work must not regain the fluid Riemann/EOS product.
    passive_flux = sources / "numerics" / "passive_flux.hpp"
    for dependency in (
        config, field_metadata, public / "riemann.hpp",
        public / "eos" / "equation_of_state.hpp",
        public / "eos" / "config.hpp",
        sources / "numerics" / "rhs.hpp",
    ):
        if dependency in graph.paths_from(passive_flux):
            errors.append("scalar-only flux reaches fluid implementation: "
                          + graph.chain(passive_flux, dependency))

    isolated = {
        sources / "relativity" / "black_hole_excision_kernels.hpp": {
            sources / "relativity" / "black_hole_excision.cpp",
            sources / "relativity" / "relativistic_flux_limiter.cpp",
            *(sources / "relativity" / f"rmhd_correction_flux_{method}.cpp"
              for method in ("llf", "hll", "hllc", "hlld")),
        },
        sources / "relativity" / "rmhd_correction_flux_impl.hpp": {
            sources / "relativity" / f"rmhd_correction_flux_{method}.cpp"
            for method in ("llf", "hll", "hllc", "hlld")
        },
        public / "kernels_generated" / "vector_ops.hpp": set(),
        public / "generated" / "diagnostic_catalog.hpp": {
            sources / "amr" / "amr_solver_diagnostics.cpp",
            sources / "amr" / "amr_solver_radiation.cpp",
            sources / "framework" / "multipoles.cpp",
        },
    }
    vector_leaves = public / "kernels_generated" / "vector"
    # The EoS needs magnetic energy, not CT, calculus, or viscous operators.
    # Check the transitive path so adding a bridge cannot bypass this rule.
    equation_of_state = public / "eos" / "equation_of_state.hpp"
    for dependency in sorted(graph.paths_from(equation_of_state)):
        if dependency.parent == vector_leaves and dependency.name != "magnetic.hpp":
            errors.append("EoS reaches unrelated vector implementation: "
                          + graph.chain(equation_of_state, dependency))
    signal_speed = sources / "relativity" / "signal_speed_factory.cpp"
    viscous = vector_leaves / "viscous.hpp"
    if viscous in graph.paths_from(signal_speed):
        errors.append("timestep policy reaches viscous RHS: "
                      + graph.chain(signal_speed, viscous))
    for header in (public / "reconstruction").glob("*.hpp"):
        if header.stem == "scalar":  # Shared scalar/limiter primitives.
            continue
        methods = [header.stem]
        if header.stem == "parabolic":  # Explicitly shared PPM machinery.
            methods += ["ppm4", "ppmx"]
        isolated[header] = {
            sources / "flux_kernels" / f"{method}{suffix}.cpp"
            for method in methods for suffix in ("", "_char")
        }
        # Advanced RMHD has one dedicated runtime-dispatch owner to avoid
        # cloning its large Riemann algorithms into every generic flux owner.
        isolated[header].add(sources / "relativity" / "rmhd_advanced_flux.cpp")
    for header, allowed in isolated.items():
        for consumer in graph.consumers(header):
            if consumer not in allowed:
                errors.append("implementation escaped its compile owner: "
                              + graph.chain(consumer, header))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, action="append", default=[],
                        help="report consumers; path relative to pyfluid_amrex")
    args = parser.parse_args()
    graph = IncludeGraph()
    for header in args.header:
        header = (ROOT / header).resolve()
        if header not in graph.edges:
            parser.error(f"not a project source/header: {header}")
        consumers = graph.consumers(header)
        print(f"{header.relative_to(ROOT)}: {len(consumers)} source owners "
              "(all dimensions/features combined; not object count)")
        for consumer in consumers:
            print(f"  {consumer.relative_to(ROOT)}")
    errors = ownership_errors(graph)
    for error in errors:
        print(f"FAIL build ownership: {error}")
    if not errors:
        print("PASS build ownership")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
