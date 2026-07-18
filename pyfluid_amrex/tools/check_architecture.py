#!/usr/bin/env python3
"""Enforce the AMReX source-ownership and generated-schema boundaries."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
INCLUDE_ROOT = ROOT / "include" / "pyfluid_amrex"
GENERATED_FIELD_IDS = INCLUDE_ROOT / "generated" / "field_ids.hpp"


def relative_paths(paths: list[Path]) -> str:
    return ", ".join(str(path.relative_to(ROOT)) for path in sorted(paths))


def main() -> None:
    errors: list[str] = []

    source_root_files = [path for path in SOURCE_ROOT.iterdir() if path.is_file()]
    if source_root_files:
        errors.append(
            "production source files must belong to an ownership directory: "
            + relative_paths(source_root_files)
        )

    implementation_headers = [
        *SOURCE_ROOT.rglob("*_impl.hpp"),
        *INCLUDE_ROOT.rglob("*_impl.hpp"),
    ]
    if implementation_headers:
        errors.append(
            "wrapper implementation headers are prohibited: "
            + relative_paths(implementation_headers)
        )

    misplaced_programs = [
        path
        for path in SOURCE_ROOT.rglob("*.cpp")
        if path.name.endswith(("_check.cpp", "_benchmark.cpp"))
    ]
    if misplaced_programs:
        errors.append(
            "checks and benchmarks must live outside production src/: "
            + relative_paths(misplaced_programs)
        )

    explicit_instantiation_roots = {
        SOURCE_ROOT / "flux_kernels",
        SOURCE_ROOT / "kernel_bundles",
    }
    wrapper_sources = [
        path
        for path in SOURCE_ROOT.rglob("*.cpp")
        if len(path.read_text().splitlines()) <= 20
        and not any(root in path.parents for root in explicit_instantiation_roots)
    ]
    if wrapper_sources:
        errors.append(
            "short production wrappers need a real ownership boundary or removal: "
            + relative_paths(wrapper_sources)
        )

    declarations = {
        "enum Conserved": [],
        "enum Z4cComponent": [],
        "enum class AuxGridField": [],
    }
    for path in INCLUDE_ROOT.rglob("*.hpp"):
        text = path.read_text()
        for declaration in declarations:
            if declaration in text:
                declarations[declaration].append(path)

    for declaration, owners in declarations.items():
        if owners != [GENERATED_FIELD_IDS]:
            errors.append(
                f"{declaration!r} must be owned only by generated/field_ids.hpp; "
                f"found: {relative_paths(owners) if owners else 'none'}"
            )

    required_catalog_outputs = (
        ROOT / "generated" / "field_catalog.json",
        ROOT / "docs" / "equations.md",
        ROOT / "docs" / "equations.tex",
        INCLUDE_ROOT / "generated" / "field_catalog.hpp",
        INCLUDE_ROOT / "generated" / "diagnostic_catalog.hpp",
    )
    missing = [path for path in required_catalog_outputs if not path.is_file()]
    if missing:
        errors.append("required generated catalog outputs are missing: " + relative_paths(missing))

    synthesized_field_names = []
    for root in (SOURCE_ROOT, INCLUDE_ROOT):
        for path in (*root.rglob("*.cpp"), *root.rglob("*.hpp")):
            if '"face_" +' in path.read_text():
                synthesized_field_names.append(path)
    if synthesized_field_names:
        errors.append(
            "grid-field names must come from the generated catalog, not string synthesis: "
            + relative_paths(synthesized_field_names)
        )

    if errors:
        raise SystemExit("\n".join(f"FAIL architecture: {error}" for error in errors))

    print("PASS amrex_architecture")


if __name__ == "__main__":
    main()
