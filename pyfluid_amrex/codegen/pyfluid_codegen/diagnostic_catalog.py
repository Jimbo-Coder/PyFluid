from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticEvaluation(str, Enum):
    RUNTIME_REDUCTION = "runtime_reduction"
    FIELD_OUTPUT = "field_output"
    POSTPROCESS = "postprocess"


class DiagnosticOperation(str, Enum):
    COORDINATE = "coordinate"
    VOLUME_INTEGRAL = "volume_integral"
    SURFACE_INTEGRAL = "surface_integral"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    LINF_NORM = "linf_norm"
    DERIVED = "derived"
    COUNTER = "counter"
    PERFORMANCE = "performance"
    FIELD = "field"


class DiagnosticStorage(str, Enum):
    REAL = "real"
    INTEGER = "integer"
    LONG = "long"


class DiagnosticValueKind(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"


@dataclass(frozen=True)
class Diagnostic:
    name: str
    label: str
    evaluation: DiagnosticEvaluation
    operation: DiagnosticOperation = DiagnosticOperation.DERIVED
    storage: DiagnosticStorage = DiagnosticStorage.REAL
    groups: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    default: str = "0.0"
    family: str = ""
    value_kind: DiagnosticValueKind = DiagnosticValueKind.SCALAR
    component_index: int = 0
    component_count: int = 1


def _scalar(
    name: str,
    label: str,
    operation: DiagnosticOperation,
    *groups: str,
    storage: DiagnosticStorage = DiagnosticStorage.REAL,
    dependencies: tuple[str, ...] = (),
    requirements: tuple[str, ...] = (),
    default: str | None = None,
) -> Diagnostic:
    if default is None:
        default = "0.0" if storage is DiagnosticStorage.REAL else "0"
    return Diagnostic(
        name=name,
        label=label,
        evaluation=DiagnosticEvaluation.RUNTIME_REDUCTION,
        operation=operation,
        storage=storage,
        groups=groups,
        dependencies=dependencies,
        requirements=requirements,
        default=default,
        family=name,
    )


def _vector(
    family: str,
    names: tuple[str, str, str],
    labels: tuple[str, str, str],
    operation: DiagnosticOperation,
    *groups: str,
    dependencies: tuple[str, ...] = (),
    requirements: tuple[str, ...] = (),
) -> tuple[Diagnostic, Diagnostic, Diagnostic]:
    return tuple(
        Diagnostic(
            name=name,
            label=label,
            evaluation=DiagnosticEvaluation.RUNTIME_REDUCTION,
            operation=operation,
            groups=groups,
            dependencies=dependencies,
            requirements=requirements,
            family=family,
            value_kind=DiagnosticValueKind.VECTOR,
            component_index=component,
            component_count=3,
        )
        for component, (name, label) in enumerate(zip(names, labels, strict=True))
    )


# Ordering here is the canonical scalars.tsv ordering.  The generated C++
# record, descriptors, selectors, and writer visitor all consume this tuple.
RUNTIME_SCALAR_DIAGNOSTICS = (
    _scalar("step", r"n", DiagnosticOperation.COORDINATE, "coordinates", storage=DiagnosticStorage.INTEGER),
    _scalar("time", r"t", DiagnosticOperation.COORDINATE, "coordinates"),
    _scalar("dt", r"\Delta t", DiagnosticOperation.COORDINATE, "coordinates"),
    _scalar("mass", r"M_0", DiagnosticOperation.VOLUME_INTEGRAL, "conservation", dependencies=("rho",)),
    *_vector(
        "conserved_momentum",
        ("mom1", "mom2", "mom3"),
        (r"P_1", r"P_2", r"P_3"),
        DiagnosticOperation.VOLUME_INTEGRAL,
        "conservation",
        dependencies=("momentum",),
    ),
    *_vector(
        "coordinate_angular_momentum",
        ("angular_mom1", "angular_mom2", "angular_mom3"),
        (r"J_1", r"J_2", r"J_3"),
        DiagnosticOperation.VOLUME_INTEGRAL,
        "conservation",
        dependencies=("momentum",),
    ),
    _scalar("energy", r"E", DiagnosticOperation.VOLUME_INTEGRAL, "conservation", dependencies=("energy",)),
    _scalar("gravity_potential_energy", r"E_{\rm grav}", DiagnosticOperation.VOLUME_INTEGRAL, "gravity", requirements=("gravity",)),
    _scalar("gravity_total_energy", r"E_{\rm total}", DiagnosticOperation.DERIVED, "gravity", requirements=("gravity",)),
    _scalar("gravity_total_energy_rel_drift", r"\Delta E_{\rm total}/E_{\rm total,0}", DiagnosticOperation.DERIVED, "gravity", requirements=("gravity",)),
    _scalar("passive_mass", r"M_{\rm passive}", DiagnosticOperation.VOLUME_INTEGRAL, "passive", requirements=("passive",)),
    _scalar("passive_mass_rel_drift", r"\Delta M_{\rm passive}/M_{\rm passive,0}", DiagnosticOperation.DERIVED, "passive", requirements=("passive",)),
    _scalar("wall_work", r"W_{\rm wall}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("wall_power", r"\dot W_{\rm wall}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("energy_minus_wall_work", r"\Delta E-W_{\rm wall}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("wall_work_rel_error", r"\epsilon_{W}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("external_budget_valid", r"b_{\rm ext}", DiagnosticOperation.COUNTER, "ledger", storage=DiagnosticStorage.INTEGER, default="1"),
    _scalar("boundary_mass_change", r"\Delta M_{\partial\Omega}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("boundary_energy_change", r"\Delta E_{\partial\Omega}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("source_mass_change", r"\Delta M_{S}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("source_energy_change", r"\Delta E_{S}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("external_mass_change", r"\Delta M_{\rm ext}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("external_energy_change", r"\Delta E_{\rm ext}", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("external_mass_residual", r"R_M", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("external_energy_residual", r"R_E", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("external_mass_rel_error", r"\epsilon_M", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("external_energy_rel_error", r"\epsilon_E", DiagnosticOperation.DERIVED, "ledger"),
    _scalar("mass_rel_drift", r"\Delta M/M_0", DiagnosticOperation.DERIVED, "conservation"),
    _scalar("energy_rel_drift", r"\Delta E/E_0", DiagnosticOperation.DERIVED, "conservation"),
    _scalar("min_rho", r"\min\rho", DiagnosticOperation.MINIMUM, "state_bounds", dependencies=("rho",)),
    _scalar("max_rho", r"\max\rho", DiagnosticOperation.MAXIMUM, "state_bounds", dependencies=("rho",)),
    _scalar("max_rho_rel_change", r"\Delta\rho_{\max}/\rho_{\max,0}", DiagnosticOperation.DERIVED, "state_bounds"),
    _scalar("min_pressure", r"\min P", DiagnosticOperation.MINIMUM, "state_bounds", dependencies=("pressure",)),
    _scalar("min_lapse", r"\min\alpha", DiagnosticOperation.MINIMUM, "state_bounds", requirements=("spacetime",)),
    _scalar("newtonian_magnetic_energy", r"E_B^{\rm NR}", DiagnosticOperation.VOLUME_INTEGRAL, "mhd", requirements=("newtonian_mhd",)),
    _scalar("rmhd_electromagnetic_energy", r"E_{\rm EM}", DiagnosticOperation.VOLUME_INTEGRAL, "rmhd", requirements=("rmhd",)),
    _scalar("rmhd_max_b_squared", r"\max b^2", DiagnosticOperation.MAXIMUM, "rmhd", requirements=("rmhd",)),
    _scalar("rmhd_max_enthalpy_magnetization", r"\max[b^2/(\rho h)]", DiagnosticOperation.MAXIMUM, "rmhd", requirements=("rmhd",)),
    _scalar("rmhd_min_plasma_beta", r"\min(2P/b^2)", DiagnosticOperation.MINIMUM, "rmhd", requirements=("rmhd",)),
    _scalar("max_lorentz_factor", r"\max W", DiagnosticOperation.MAXIMUM, "relativistic", requirements=("relativistic",)),
    _scalar("relativistic_atmosphere_volume_fraction", r"V_{\rm atm}/V", DiagnosticOperation.DERIVED, "relativistic", requirements=("relativistic",)),
    _scalar("divb_l1", r"\|\nabla_i B^i\|_1", DiagnosticOperation.L1_NORM, "mhd", requirements=("mhd",)),
    _scalar("divb_l2", r"\|\nabla_i B^i\|_2", DiagnosticOperation.L2_NORM, "mhd", requirements=("mhd",)),
    _scalar("divb_linf", r"\|\nabla_i B^i\|_\infty", DiagnosticOperation.LINF_NORM, "mhd", requirements=("mhd",)),
    _scalar("diffusion_residual", r"R_{\rm diff}", DiagnosticOperation.DERIVED, "diffusion", requirements=("diffusion",)),
    _scalar("diffusion_iterations", r"N_{\rm diff}", DiagnosticOperation.COUNTER, "diffusion", storage=DiagnosticStorage.INTEGER, requirements=("diffusion",)),
    _scalar("amr_total_cells", r"N_{\rm cells}", DiagnosticOperation.COUNTER, "amr", storage=DiagnosticStorage.LONG),
    _scalar("amr_active_finest_level", r"L_{\rm active}", DiagnosticOperation.COUNTER, "amr", storage=DiagnosticStorage.INTEGER),
    _scalar("amr_finest_headroom", r"L_{\max}-L_{\rm active}", DiagnosticOperation.COUNTER, "amr", storage=DiagnosticStorage.INTEGER),
    _scalar("amr_touches_max_level", r"b_{L_{\max}}", DiagnosticOperation.COUNTER, "amr", storage=DiagnosticStorage.INTEGER),
    _scalar("amr_total_budget_fraction", r"f_{\rm AMR}", DiagnosticOperation.DERIVED, "amr", default="-1.0"),
    _scalar("amr_peak_level_budget_fraction", r"f_{\rm AMR,peak}", DiagnosticOperation.DERIVED, "amr", default="-1.0"),
    _scalar("mcells_per_s", r"10^6\,\mathrm{cells}/\mathrm{s}", DiagnosticOperation.PERFORMANCE, "performance"),
    _scalar("layout_mcells_per_s", r"10^6\,\mathrm{layout\ cells}/\mathrm{s}", DiagnosticOperation.PERFORMANCE, "performance"),
    _scalar("sim_time_per_s", r"\Delta t_{\rm sim}/\Delta t_{\rm wall}", DiagnosticOperation.PERFORMANCE, "performance"),
    _scalar("cfl_limiter_frequency", r"\lambda_{\rm CFL}", DiagnosticOperation.MAXIMUM, "cfl"),
    _scalar("cfl_limiter_level", r"L_{\rm CFL}", DiagnosticOperation.DERIVED, "cfl", storage=DiagnosticStorage.INTEGER, default="-1"),
    _scalar("cfl_limiter_i", r"i_{\rm CFL}", DiagnosticOperation.DERIVED, "cfl", storage=DiagnosticStorage.INTEGER),
    _scalar("cfl_limiter_j", r"j_{\rm CFL}", DiagnosticOperation.DERIVED, "cfl", storage=DiagnosticStorage.INTEGER),
    _scalar("cfl_limiter_k", r"k_{\rm CFL}", DiagnosticOperation.DERIVED, "cfl", storage=DiagnosticStorage.INTEGER),
    _scalar("cfl_limiter_q1", r"q^1_{\rm CFL}", DiagnosticOperation.DERIVED, "cfl"),
    _scalar("cfl_limiter_q2", r"q^2_{\rm CFL}", DiagnosticOperation.DERIVED, "cfl"),
    _scalar("cfl_limiter_q3", r"q^3_{\rm CFL}", DiagnosticOperation.DERIVED, "cfl"),
    _scalar("z4c_hamiltonian_linf", r"\|\mathcal H\|_\infty", DiagnosticOperation.LINF_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_hamiltonian_l2", r"\|\mathcal H\|_2", DiagnosticOperation.L2_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_hamiltonian_interior_linf", r"\|\mathcal H\|_{\infty,\mathrm{interior}}", DiagnosticOperation.LINF_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_hamiltonian_interior_l2", r"\|\mathcal H\|_{2,\mathrm{interior}}", DiagnosticOperation.L2_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_hamiltonian_cf_linf", r"\|\mathcal H\|_{\infty,\mathrm{CF}}", DiagnosticOperation.LINF_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_hamiltonian_cf_l2", r"\|\mathcal H\|_{2,\mathrm{CF}}", DiagnosticOperation.L2_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_momentum_linf", r"\|\mathcal M_i\|_\infty", DiagnosticOperation.LINF_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_z_linf", r"\|Z_i\|_\infty", DiagnosticOperation.LINF_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_theta_linf", r"\|\Theta\|_\infty", DiagnosticOperation.LINF_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_log_conformal_determinant_linf", r"\|\ln\det\tilde\gamma\|_\infty", DiagnosticOperation.LINF_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("z4c_trace_a_tilde_linf", r"\|\mathrm{tr}\tilde A\|_\infty", DiagnosticOperation.LINF_NORM, "constraints", requirements=("spacetime",), dependencies=("z4c_constraints",)),
    _scalar("adm_mass_surface", r"M_{\rm ADM}^{S}", DiagnosticOperation.SURFACE_INTEGRAL, "global_gr", requirements=("spacetime",), dependencies=("z4c_physical_geometry",)),
    _scalar("adm_mass_volume", r"M_{\rm ADM}^{V}", DiagnosticOperation.VOLUME_INTEGRAL, "global_gr", requirements=("spacetime",), dependencies=("z4c_physical_geometry",)),
    _scalar("proper_mass_volume", r"M_{\rm proper}^{V}", DiagnosticOperation.VOLUME_INTEGRAL, "global_gr", requirements=("spacetime",), dependencies=("z4c_physical_geometry",)),
    _scalar("komar_mass_surface", r"M_{\rm K}^{S}", DiagnosticOperation.SURFACE_INTEGRAL, "global_gr", requirements=("spacetime",), dependencies=("z4c_physical_geometry",)),
    _scalar("komar_mass_volume", r"M_{\rm K}^{V}", DiagnosticOperation.VOLUME_INTEGRAL, "global_gr", requirements=("spacetime",), dependencies=("z4c_physical_geometry",)),
    _scalar("binding_energy_volume", r"E_{\rm bind}^{V}", DiagnosticOperation.DERIVED, "global_gr", requirements=("spacetime",), dependencies=("mass", "adm_mass_volume")),
    *_vector(
        "adm_momentum_surface",
        ("adm_momentum1_surface", "adm_momentum2_surface", "adm_momentum3_surface"),
        (r"P_{\rm ADM,1}^{S}", r"P_{\rm ADM,2}^{S}", r"P_{\rm ADM,3}^{S}"),
        DiagnosticOperation.SURFACE_INTEGRAL,
        "global_gr",
        requirements=("spacetime",),
        dependencies=("z4c_physical_geometry",),
    ),
    *_vector(
        "adm_momentum_volume",
        ("adm_momentum1_volume", "adm_momentum2_volume", "adm_momentum3_volume"),
        (r"P_{\rm ADM,1}^{V}", r"P_{\rm ADM,2}^{V}", r"P_{\rm ADM,3}^{V}"),
        DiagnosticOperation.VOLUME_INTEGRAL,
        "global_gr",
        requirements=("spacetime",),
        dependencies=("z4c_physical_geometry",),
    ),
    *_vector(
        "adm_angular_momentum_surface",
        ("adm_angular_momentum1_surface", "adm_angular_momentum2_surface", "adm_angular_momentum3_surface"),
        (r"J_{\rm ADM,1}^{S}", r"J_{\rm ADM,2}^{S}", r"J_{\rm ADM,3}^{S}"),
        DiagnosticOperation.SURFACE_INTEGRAL,
        "global_gr",
        requirements=("spacetime",),
        dependencies=("z4c_physical_geometry",),
    ),
    *_vector(
        "adm_angular_momentum_volume",
        ("adm_angular_momentum1_volume", "adm_angular_momentum2_volume", "adm_angular_momentum3_volume"),
        (r"J_{\rm ADM,1}^{V}", r"J_{\rm ADM,2}^{V}", r"J_{\rm ADM,3}^{V}"),
        DiagnosticOperation.VOLUME_INTEGRAL,
        "global_gr",
        requirements=("spacetime",),
        dependencies=("z4c_physical_geometry",),
    ),
    _scalar("extraction_radius", r"r_{\rm ext}", DiagnosticOperation.COORDINATE, "radiation", requirements=("spacetime",)),
    _scalar("matter_energy_flux", r"\dot E_{\rm matter}^{S}", DiagnosticOperation.SURFACE_INTEGRAL, "radiation", requirements=("spacetime",)),
    _scalar("psi4_l2", r"\|\Psi_4\|_{2,S}", DiagnosticOperation.L2_NORM, "radiation", requirements=("spacetime",)),
)


# "standard" preserves the historical compact scalars.tsv contract. Global GR
# surface/volume charges are opt-in until a surface is explicitly configured.
DIAGNOSTIC_GROUPS = {
    "standard": tuple(
        diagnostic.name
        for diagnostic in RUNTIME_SCALAR_DIAGNOSTICS
        if "global_gr" not in diagnostic.groups and "radiation" not in diagnostic.groups
    ),
    **{
        group: tuple(
            diagnostic.name
            for diagnostic in RUNTIME_SCALAR_DIAGNOSTICS
            if group in diagnostic.groups
        )
        for group in (
            "coordinates", "conservation", "gravity", "passive", "ledger",
            "state_bounds", "mhd", "rmhd", "relativistic", "diffusion", "amr", "performance", "cfl",
            "constraints", "global_gr",
            "radiation",
        )
    },
}


POSTPROCESS_DIAGNOSTICS = (
    Diagnostic("pressure", r"P", DiagnosticEvaluation.POSTPROCESS, DiagnosticOperation.FIELD, dependencies=("rho", "momentum", "energy")),
    Diagnostic("velocity_mag", r"|v|", DiagnosticEvaluation.POSTPROCESS, DiagnosticOperation.FIELD, dependencies=("rho", "momentum")),
    Diagnostic("schlieren", r"|\nabla\rho|", DiagnosticEvaluation.POSTPROCESS, DiagnosticOperation.FIELD, dependencies=("rho",)),
    Diagnostic("j_mag", r"|J|", DiagnosticEvaluation.POSTPROCESS, DiagnosticOperation.FIELD, dependencies=("j1", "j2", "j3")),
    Diagnostic("e_mag", r"|E|", DiagnosticEvaluation.POSTPROCESS, DiagnosticOperation.FIELD, dependencies=("e1", "e2", "e3")),
    Diagnostic("vorticity_mag", r"|\omega|", DiagnosticEvaluation.POSTPROCESS, DiagnosticOperation.FIELD, dependencies=("vorticity1", "vorticity2", "vorticity3")),
)


def validate_diagnostic_catalog() -> None:
    diagnostics = (*RUNTIME_SCALAR_DIAGNOSTICS, *POSTPROCESS_DIAGNOSTICS)
    names = [diagnostic.name for diagnostic in diagnostics]
    if len(names) != len(set(names)):
        raise ValueError("diagnostic names must be unique")
    family_shapes: dict[str, tuple[DiagnosticValueKind, int]] = {}
    family_components: dict[str, set[int]] = {}
    for diagnostic in diagnostics:
        family = diagnostic.family or diagnostic.name
        shape = (diagnostic.value_kind, diagnostic.component_count)
        if family in family_shapes and family_shapes[family] != shape:
            raise ValueError(f"diagnostic family {family} has inconsistent shape")
        family_shapes[family] = shape
        family_components.setdefault(family, set()).add(diagnostic.component_index)
    for family, (_, component_count) in family_shapes.items():
        expected = set(range(component_count))
        if family_components[family] != expected:
            raise ValueError(f"diagnostic family {family} has incomplete components")
    runtime_names = {diagnostic.name for diagnostic in RUNTIME_SCALAR_DIAGNOSTICS}
    for group, members in DIAGNOSTIC_GROUPS.items():
        unknown = set(members) - runtime_names
        if unknown:
            raise ValueError(f"diagnostic group {group} contains unknown members: {sorted(unknown)}")


validate_diagnostic_catalog()
