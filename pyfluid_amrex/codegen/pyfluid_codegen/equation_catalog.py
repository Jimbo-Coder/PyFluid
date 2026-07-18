from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .eos_equations import analytic_equations_of_state
from .field_catalog import (
    FLUID_NEWTONIAN_FIELDS,
    MAGNETIC_FIELD,
    GridField,
)
from .physical_system import (
    ConservationLaw,
    HydroState,
    Hydrodynamics,
    MHDState,
    Magnetohydrodynamics,
)
from .tensor_algebra import Tensor, Vector


@dataclass(frozen=True)
class Equation:
    identifier: str
    title: str
    conserved_label: str
    flux: sp.Expr | Vector | Tensor
    source: sp.Expr | Vector


@dataclass(frozen=True)
class EquationSystem:
    identifier: str
    title: str
    description: str
    fields: tuple[GridField, ...]
    equations: tuple[Equation, ...]


def _equations(
    prefix: str,
    law: ConservationLaw,
    fields: tuple[GridField, ...],
) -> tuple[Equation, ...]:
    titles = ("Mass", "Momentum", "Energy", "Induction")
    if len(law.equations) != len(fields):
        raise ValueError(f"{prefix} equation and field counts differ")
    return tuple(
        Equation(
            identifier=f"{prefix}.{title.lower()}",
            title=title,
            conserved_label=field.label,
            flux=equation.flux,
            source=equation.source,
        )
        for title, equation, field in zip(
            titles[: len(law.equations)], law.equations, fields, strict=True
        )
    )


def fluid_equation_systems() -> tuple[EquationSystem, ...]:
    rho, mom1, mom2, mom3, energy, pressure = sp.symbols(
        r"rho m_1 m_2 m_3 E P"
    )
    hydro_state = HydroState.from_symbols(rho, mom1, mom2, mom3, energy)
    hydro_fields = FLUID_NEWTONIAN_FIELDS
    hydro_law = Hydrodynamics().conservation_law(hydro_state, pressure)

    b1, b2, b3 = sp.symbols(r"B_1 B_2 B_3")
    mhd_state = MHDState.from_symbols(
        rho, mom1, mom2, mom3, energy, b1, b2, b3
    )
    mhd_fields = (*FLUID_NEWTONIAN_FIELDS, MAGNETIC_FIELD)
    mhd_law = Magnetohydrodynamics().conservation_law(mhd_state, pressure)

    return (
        EquationSystem(
            "hd",
            "Newtonian Hydrodynamics",
            "Conservative ideal-fluid equations with an equation-of-state closure.",
            hydro_fields,
            _equations("hd", hydro_law, hydro_fields),
        ),
        EquationSystem(
            "mhd",
            "Newtonian Ideal Magnetohydrodynamics",
            "Conservative ideal-MHD equations with cell-centered magnetic state and optional constrained transport.",
            mhd_fields,
            _equations("mhd", mhd_law, mhd_fields),
        ),
    )


@dataclass(frozen=True)
class EquationOfStateRecord:
    identifier: str
    title: str
    pressure: sp.Expr
    energy_from_pressure: sp.Expr
    sound_speed_squared: sp.Expr
    is_barotropic: bool


def equation_of_state_records() -> tuple[EquationOfStateRecord, ...]:
    pressure = sp.Symbol("P", positive=True)
    return tuple(
        EquationOfStateRecord(
            identifier=f"eos.{eos.name}",
            title=eos.name.replace("_", " ").title(),
            pressure=eos.pressure(),
            energy_from_pressure=eos.internal_energy_from_pressure(pressure),
            sound_speed_squared=eos.sound_speed_squared_from_pressure(pressure),
            is_barotropic=eos.is_barotropic,
        )
        for eos in analytic_equations_of_state()
    )


def validate_equation_catalog() -> None:
    identifiers = [
        equation.identifier
        for system in fluid_equation_systems()
        for equation in system.equations
    ]
    identifiers.extend(record.identifier for record in equation_of_state_records())
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("equation identifiers must be unique")


validate_equation_catalog()
