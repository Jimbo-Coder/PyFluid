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

    private_implementation_leaks = [
        path
        for path in (
            INCLUDE_ROOT / "amr_solver_base.hpp",
            INCLUDE_ROOT / "amr_solver_internal.hpp",
            INCLUDE_ROOT / "solver_sections",
            INCLUDE_ROOT / "output.hpp",
            INCLUDE_ROOT / "derived_aux.hpp",
            INCLUDE_ROOT / "solid_conservation.hpp",
            INCLUDE_ROOT / "state_interpolation.hpp",
            INCLUDE_ROOT / "sources.hpp",
            INCLUDE_ROOT / "patch_wall_flux.hpp",
            INCLUDE_ROOT / "primitive_vector_transform.hpp",
            INCLUDE_ROOT / "rhs.hpp",
            INCLUDE_ROOT / "mapped_patch_interfaces.hpp",
            INCLUDE_ROOT / "ct_diagnostics.hpp",
            INCLUDE_ROOT / "rhd_system.hpp",
            INCLUDE_ROOT / "rmhd_system.hpp",
            INCLUDE_ROOT / "primitive_cache.hpp",
            INCLUDE_ROOT / "relativistic_atmosphere.hpp",
            INCLUDE_ROOT / "relativistic_sources.hpp",
        )
        if path.exists()
    ]
    if private_implementation_leaks:
        errors.append(
            "backend implementation headers must remain under their src/ owner: "
            + relative_paths(private_implementation_leaks)
        )

    public_config = INCLUDE_ROOT / "config.hpp"
    public_config_text = public_config.read_text()
    leaked_config_catalogs = [
        token
        for token in ("RiemannOptions", "InitialDataOptions", "NamedOptionCode")
        if token in public_config_text
    ]
    if leaked_config_catalogs:
        errors.append(
            "host-only option catalogs must remain private in src/framework/: "
            + ", ".join(leaked_config_catalogs)
        )
    if "inline " in public_config_text:
        errors.append(
            "public config.hpp must declare configuration contracts, not own "
            "host-side implementations"
        )
    leaked_config_implementations = [
        include
        for include in (
            '"pyfluid_amrex/eos/equation_of_state.hpp"',
            '"pyfluid_amrex/fixed_spacetime.hpp"',
        )
        if include in public_config_text
    ]
    if leaked_config_implementations:
        errors.append(
            "public config.hpp must depend on stable type schemas, not pointwise "
            "physics implementations: " + ", ".join(leaked_config_implementations)
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
        "struct EOSParams": [],
        "struct FixedSpacetimeParameters": [],
    }
    for path in INCLUDE_ROOT.rglob("*.hpp"):
        text = path.read_text()
        for declaration in declarations:
            if declaration in text:
                declarations[declaration].append(path)

    declaration_owners = {
        "enum Conserved": GENERATED_FIELD_IDS,
        "enum Z4cComponent": GENERATED_FIELD_IDS,
        "enum class AuxGridField": GENERATED_FIELD_IDS,
        "struct EOSParams": INCLUDE_ROOT / "eos" / "types.hpp",
        "struct FixedSpacetimeParameters": INCLUDE_ROOT / "fixed_spacetime_types.hpp",
    }
    for declaration, owners in declarations.items():
        expected_owner = declaration_owners[declaration]
        if owners != [expected_owner]:
            errors.append(
                f"{declaration!r} must be owned only by "
                f"{expected_owner.relative_to(INCLUDE_ROOT)}; "
                f"found: {relative_paths(owners) if owners else 'none'}"
            )

    required_catalog_outputs = (
        ROOT / "generated" / "field_catalog.json",
        ROOT / "docs" / "equations.md",
        ROOT / "docs" / "equations.tex",
        INCLUDE_ROOT / "generated" / "field_catalog.hpp",
        INCLUDE_ROOT / "generated" / "diagnostic_catalog.hpp",
        INCLUDE_ROOT / "generated" / "method_catalog.hpp",
    )
    missing = [path for path in required_catalog_outputs if not path.is_file()]
    if missing:
        errors.append("required generated catalog outputs are missing: " + relative_paths(missing))

    method_codes = (INCLUDE_ROOT / "method_codes.hpp").read_text()
    if "enum ReconstructionMethodCode" in method_codes:
        errors.append(
            "reconstruction identity must be generated from schema/methods.toml, "
            "not redeclared in method_codes.hpp"
        )
    config_options = (SOURCE_ROOT / "framework" / "config_options.hpp").read_text()
    if "ReconOptions = ReconstructionMethodCatalog" not in config_options:
        errors.append(
            "runtime reconstruction options must consume ReconstructionMethodCatalog"
        )

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
