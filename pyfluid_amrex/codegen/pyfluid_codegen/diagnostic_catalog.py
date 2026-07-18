from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticEvaluation(str, Enum):
    RUNTIME_REDUCTION = "runtime_reduction"
    FIELD_OUTPUT = "field_output"
    POSTPROCESS = "postprocess"


@dataclass(frozen=True)
class Diagnostic:
    name: str
    label: str
    evaluation: DiagnosticEvaluation
    dependencies: tuple[str, ...] = ()


_SCALAR_NAMES = (
    "step", "time", "dt", "mass", "mom1", "mom2", "mom3",
    "angular_mom1", "angular_mom2", "angular_mom3", "energy",
    "gravity_potential_energy", "gravity_total_energy",
    "gravity_total_energy_rel_drift", "passive_mass", "passive_mass_rel_drift",
    "wall_work", "wall_power", "energy_minus_wall_work", "wall_work_rel_error",
    "external_budget_valid", "boundary_mass_change", "boundary_energy_change",
    "source_mass_change", "source_energy_change", "external_mass_change",
    "external_energy_change", "external_mass_residual", "external_energy_residual",
    "external_mass_rel_error", "external_energy_rel_error", "mass_rel_drift",
    "energy_rel_drift", "min_rho", "max_rho", "max_rho_rel_change",
    "min_pressure", "magnetic_energy", "divb_l1", "diffusion_residual",
    "diffusion_iterations", "amr_total_cells", "amr_active_finest_level",
    "amr_finest_headroom", "amr_touches_max_level", "amr_total_budget_fraction",
    "amr_peak_level_budget_fraction", "mcells_per_s", "layout_mcells_per_s",
    "sim_time_per_s", "cfl_limiter_frequency", "cfl_limiter_level",
    "cfl_limiter_i", "cfl_limiter_j", "cfl_limiter_k", "cfl_limiter_q1",
    "cfl_limiter_q2", "cfl_limiter_q3", "z4c_hamiltonian_linf",
    "z4c_hamiltonian_l2", "z4c_momentum_linf", "z4c_z_linf",
    "z4c_theta_linf", "z4c_log_conformal_determinant_linf",
    "z4c_trace_a_tilde_linf",
)

_SCALAR_LABELS = {
    "mass": r"M",
    "mom1": r"P_1",
    "mom2": r"P_2",
    "mom3": r"P_3",
    "energy": r"E",
    "magnetic_energy": r"E_B",
    "divb_l1": r"\|\nabla_i B^i\|_1",
    "mass_rel_drift": r"\Delta M/M_0",
    "energy_rel_drift": r"\Delta E/E_0",
    "mcells_per_s": r"10^6\,\mathrm{cells}/\mathrm{s}",
    "sim_time_per_s": r"\Delta t_{\rm sim}/\Delta t_{\rm wall}",
}


RUNTIME_SCALAR_DIAGNOSTICS = tuple(
    Diagnostic(
        name,
        _SCALAR_LABELS.get(name, rf"\mathrm{{{name.replace('_', r'\_')}}}"),
        DiagnosticEvaluation.RUNTIME_REDUCTION,
    )
    for name in _SCALAR_NAMES
)

POSTPROCESS_DIAGNOSTICS = (
    Diagnostic("pressure", r"P", DiagnosticEvaluation.POSTPROCESS, ("rho", "momentum", "energy")),
    Diagnostic("velocity_mag", r"|v|", DiagnosticEvaluation.POSTPROCESS, ("rho", "momentum")),
    Diagnostic("schlieren", r"|\nabla\rho|", DiagnosticEvaluation.POSTPROCESS, ("rho",)),
    Diagnostic("j_mag", r"|J|", DiagnosticEvaluation.POSTPROCESS, ("j1", "j2", "j3")),
    Diagnostic("e_mag", r"|E|", DiagnosticEvaluation.POSTPROCESS, ("e1", "e2", "e3")),
    Diagnostic(
        "vorticity_mag",
        r"|\omega|",
        DiagnosticEvaluation.POSTPROCESS,
        ("vorticity1", "vorticity2", "vorticity3"),
    ),
)


def validate_diagnostic_catalog() -> None:
    diagnostics = (*RUNTIME_SCALAR_DIAGNOSTICS, *POSTPROCESS_DIAGNOSTICS)
    names = [diagnostic.name for diagnostic in diagnostics]
    if len(names) != len(set(names)):
        raise ValueError("diagnostic names must be unique")


validate_diagnostic_catalog()
