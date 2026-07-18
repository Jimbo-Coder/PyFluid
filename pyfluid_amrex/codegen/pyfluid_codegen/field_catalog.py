from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp


class Centering(str, Enum):
    CELL = "Cell"
    VERTEX = "Vertex"
    FACE1 = "Face1"
    FACE2 = "Face2"
    FACE3 = "Face3"
    EDGE1 = "Edge1"
    EDGE2 = "Edge2"
    EDGE3 = "Edge3"


class ValueKind(str, Enum):
    SCALAR = "Scalar"
    VECTOR3 = "Vector3"
    VECTOR4 = "Vector4"
    SYMMETRIC_TENSOR3 = "SymmetricTensor3"
    TENSOR3 = "Tensor3"
    TENSOR4 = "Tensor4"


class Symmetry(str, Enum):
    SCALAR = "Scalar"
    VECTOR1 = "Vector1"
    VECTOR2 = "Vector2"
    VECTOR3 = "Vector3"
    TENSOR11 = "Tensor11"
    TENSOR12 = "Tensor12"
    TENSOR13 = "Tensor13"
    TENSOR22 = "Tensor22"
    TENSOR23 = "Tensor23"
    TENSOR33 = "Tensor33"


class FieldRole(str, Enum):
    EVOLVED_STATE = "EvolvedState"
    EVOLUTION_AUXILIARY = "EvolutionAuxiliary"
    CACHED_DERIVATIVE = "CachedDerivative"
    OUTPUT_ONLY_AUXILIARY = "OutputOnlyAuxiliary"
    SOLVER_SCRATCH = "SolverScratch"


class DiagnosticKind(str, Enum):
    STORED_STATE = "StoredState"
    PURE_AUXILIARY = "PureAuxiliary"
    MIXED_AUXILIARY = "MixedAuxiliary"


class EvolutionKind(str, Enum):
    MAIN_STATE = "MainState"
    AUXILIARY_STATE = "AuxiliaryState"
    NON_BACKREACTING_EXTRA = "NonBackreactingExtra"
    NONE = "None"


@dataclass(frozen=True)
class FieldComponent:
    name: str
    label: str
    cpp_id: str | None = None
    centering: Centering = Centering.CELL
    symmetry: Symmetry = Symmetry.SCALAR

    def cpp_symbol(self, owner: str) -> str:
        return f"{owner}.{self.name}"


@dataclass(frozen=True)
class GridField:
    name: str
    label: str
    value_kind: ValueKind
    components: tuple[FieldComponent, ...]
    role: FieldRole
    evolution_kind: EvolutionKind
    diagnostic_kind: DiagnosticKind = DiagnosticKind.STORED_STATE
    sector: str = "None"
    ghost_width: int = 0
    checkpoint: bool = True
    output: bool = True
    aux_id: str | None = None
    requires_mhd: bool = False
    requires_solid_geometry: bool = False
    requires_gravity: bool = False
    min_dimension: int = 1


def scalar_component(
    name: str,
    label: str,
    *,
    cpp_id: str | None = None,
    centering: Centering = Centering.CELL,
) -> FieldComponent:
    return FieldComponent(name, label, cpp_id, centering, Symmetry.SCALAR)


def vector_components(
    stem: str,
    label: str,
    *,
    cpp_ids: tuple[str, str, str] | None = None,
    centerings: tuple[Centering, Centering, Centering] = (
        Centering.CELL,
        Centering.CELL,
        Centering.CELL,
    ),
) -> tuple[FieldComponent, ...]:
    ids = cpp_ids or (None, None, None)
    return tuple(
        FieldComponent(
            f"{stem}{component}",
            rf"{label}_{{{component}}}",
            ids[component - 1],
            centerings[component - 1],
            Symmetry[f"VECTOR{component}"],
        )
        for component in range(1, 4)
    )


def symmetric_tensor3_components(
    stem: str,
    label: str,
    cpp_ids: tuple[str, str, str, str, str, str],
    *,
    centering: Centering,
) -> tuple[FieldComponent, ...]:
    indices = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    return tuple(
        FieldComponent(
            f"{stem}{first}{second}",
            rf"{label}_{{{first}{second}}}",
            cpp_id,
            centering,
            Symmetry[f"TENSOR{first}{second}"],
        )
        for (first, second), cpp_id in zip(indices, cpp_ids, strict=True)
    )


FLUID_NEWTONIAN_FIELDS = (
    GridField(
        "rho",
        r"\rho",
        ValueKind.SCALAR,
        (scalar_component("rho", r"\rho", cpp_id="RHO"),),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Fluid",
    ),
    GridField(
        "momentum",
        r"m_i",
        ValueKind.VECTOR3,
        vector_components(
            "mom", "m", cpp_ids=("MOM1", "MOM2", "MOM3")
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Fluid",
    ),
    GridField(
        "energy",
        r"E",
        ValueKind.SCALAR,
        (scalar_component("energy", r"E", cpp_id="ENERGY"),),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Fluid",
    ),
)


FLUID_RELATIVISTIC_FIELDS = (
    GridField(
        "dens",
        r"D",
        ValueKind.SCALAR,
        (scalar_component("dens", r"D", cpp_id="RHO"),),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Fluid",
    ),
    FLUID_NEWTONIAN_FIELDS[1],
    GridField(
        "tau",
        r"\tau",
        ValueKind.SCALAR,
        (scalar_component("tau", r"\tau", cpp_id="ENERGY"),),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Fluid",
    ),
)


MAGNETIC_FIELD = GridField(
    "magnetic_field",
    r"B^i",
    ValueKind.VECTOR3,
    vector_components("b", "B", cpp_ids=("B1", "B2", "B3")),
    FieldRole.EVOLVED_STATE,
    EvolutionKind.MAIN_STATE,
    sector="Fluid",
    requires_mhd=True,
)

CT_MAGNETIC_FIELD = GridField(
    "ct_magnetic_field",
    r"B^i",
    ValueKind.VECTOR3,
    vector_components(
        "face_b",
        "B",
        centerings=(Centering.FACE1, Centering.FACE2, Centering.FACE3),
    ),
    FieldRole.EVOLUTION_AUXILIARY,
    EvolutionKind.AUXILIARY_STATE,
    DiagnosticKind.STORED_STATE,
    sector="Fluid",
    output=False,
    requires_mhd=True,
)

PASSIVE_SCALAR_FIELD = GridField(
    "passive_scalar",
    r"\rho C",
    ValueKind.SCALAR,
    (scalar_component("dye_mass", r"\rho C"),),
    FieldRole.EVOLVED_STATE,
    EvolutionKind.NON_BACKREACTING_EXTRA,
    sector="Fluid",
)


def component_names(fields: tuple[GridField, ...]) -> tuple[str, ...]:
    return tuple(
        component.name for field in fields for component in field.components
    )


NEWTONIAN_HYDRO_COMPONENT_NAMES = component_names(FLUID_NEWTONIAN_FIELDS)
RELATIVISTIC_HYDRO_COMPONENT_NAMES = component_names(FLUID_RELATIVISTIC_FIELDS)
MAGNETIC_COMPONENT_NAMES = component_names((MAGNETIC_FIELD,))
NEWTONIAN_MHD_COMPONENT_NAMES = (
    *NEWTONIAN_HYDRO_COMPONENT_NAMES,
    *MAGNETIC_COMPONENT_NAMES,
)
RELATIVISTIC_MHD_COMPONENT_NAMES = (
    *RELATIVISTIC_HYDRO_COMPONENT_NAMES,
    *MAGNETIC_COMPONENT_NAMES,
)
MOMENTUM_COMPONENT_NAMES = component_names((FLUID_NEWTONIAN_FIELDS[1],))


Z4C_FIELDS = (
    GridField(
        "chi",
        r"\chi",
        ValueKind.SCALAR,
        (
            scalar_component(
                "chi", r"\chi", cpp_id="Z4C_CHI", centering=Centering.VERTEX
            ),
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Spacetime",
        ghost_width=3,
    ),
    GridField(
        "gamma_tilde",
        r"\tilde{\gamma}_{ij}",
        ValueKind.SYMMETRIC_TENSOR3,
        symmetric_tensor3_components(
            "gamma_tilde",
            r"\tilde{\gamma}",
            (
                "Z4C_GAMMA_TILDE11",
                "Z4C_GAMMA_TILDE12",
                "Z4C_GAMMA_TILDE13",
                "Z4C_GAMMA_TILDE22",
                "Z4C_GAMMA_TILDE23",
                "Z4C_GAMMA_TILDE33",
            ),
            centering=Centering.VERTEX,
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Spacetime",
        ghost_width=3,
    ),
    GridField(
        "k_hat",
        r"\hat{K}",
        ValueKind.SCALAR,
        (
            scalar_component(
                "k_hat", r"\hat{K}", cpp_id="Z4C_K_HAT", centering=Centering.VERTEX
            ),
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Spacetime",
        ghost_width=3,
    ),
    GridField(
        "a_tilde",
        r"\tilde{A}_{ij}",
        ValueKind.SYMMETRIC_TENSOR3,
        symmetric_tensor3_components(
            "a_tilde",
            r"\tilde{A}",
            (
                "Z4C_A_TILDE11",
                "Z4C_A_TILDE12",
                "Z4C_A_TILDE13",
                "Z4C_A_TILDE22",
                "Z4C_A_TILDE23",
                "Z4C_A_TILDE33",
            ),
            centering=Centering.VERTEX,
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Spacetime",
        ghost_width=3,
    ),
    GridField(
        "gamma_tilde_vector",
        r"\tilde{\Gamma}^{i}",
        ValueKind.VECTOR3,
        vector_components(
            "gamma_tilde",
            r"\tilde{\Gamma}",
            cpp_ids=(
                "Z4C_GAMMA_TILDE1",
                "Z4C_GAMMA_TILDE2",
                "Z4C_GAMMA_TILDE3",
            ),
            centerings=(Centering.VERTEX,) * 3,
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Spacetime",
        ghost_width=3,
    ),
    GridField(
        "theta",
        r"\Theta",
        ValueKind.SCALAR,
        (
            scalar_component(
                "theta", r"\Theta", cpp_id="Z4C_THETA", centering=Centering.VERTEX
            ),
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Spacetime",
        ghost_width=3,
    ),
    GridField(
        "lapse",
        r"\alpha",
        ValueKind.SCALAR,
        (
            scalar_component(
                "lapse", r"\alpha", cpp_id="Z4C_LAPSE", centering=Centering.VERTEX
            ),
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Spacetime",
        ghost_width=3,
    ),
    GridField(
        "shift",
        r"\beta^{i}",
        ValueKind.VECTOR3,
        vector_components(
            "shift",
            r"\beta",
            cpp_ids=("Z4C_SHIFT1", "Z4C_SHIFT2", "Z4C_SHIFT3"),
            centerings=(Centering.VERTEX,) * 3,
        ),
        FieldRole.EVOLVED_STATE,
        EvolutionKind.MAIN_STATE,
        sector="Spacetime",
        ghost_width=3,
    ),
)


def auxiliary_field(
    aux_id: str,
    name: str,
    label: str,
    value_kind: ValueKind,
    components: tuple[FieldComponent, ...],
    role: FieldRole,
    diagnostic_kind: DiagnosticKind,
    evolution_kind: EvolutionKind,
    *,
    requires_mhd: bool = False,
    requires_solid_geometry: bool = False,
    requires_gravity: bool = False,
    min_dimension: int = 1,
) -> GridField:
    return GridField(
        name,
        label,
        value_kind,
        components,
        role,
        evolution_kind,
        diagnostic_kind,
        aux_id=aux_id,
        requires_mhd=requires_mhd,
        requires_solid_geometry=requires_solid_geometry,
        requires_gravity=requires_gravity,
        min_dimension=min_dimension,
    )


AUXILIARY_FIELDS = (
    auxiliary_field(
        "CurrentDensity",
        "j",
        r"J^i",
        ValueKind.VECTOR3,
        vector_components(
            "j", "J", centerings=(Centering.EDGE1, Centering.EDGE2, Centering.EDGE3)
        ),
        FieldRole.CACHED_DERIVATIVE,
        DiagnosticKind.MIXED_AUXILIARY,
        EvolutionKind.AUXILIARY_STATE,
        requires_mhd=True,
        min_dimension=2,
    ),
    auxiliary_field(
        "ElectricField",
        "e",
        r"E^i",
        ValueKind.VECTOR3,
        vector_components(
            "e", "E", centerings=(Centering.EDGE1, Centering.EDGE2, Centering.EDGE3)
        ),
        FieldRole.CACHED_DERIVATIVE,
        DiagnosticKind.MIXED_AUXILIARY,
        EvolutionKind.AUXILIARY_STATE,
        requires_mhd=True,
        min_dimension=2,
    ),
    auxiliary_field(
        "DivB", "divb", r"\nabla_i B^i", ValueKind.SCALAR,
        (scalar_component("divb", r"\nabla_i B^i"),),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE, requires_mhd=True,
    ),
    auxiliary_field(
        "MagneticEnergy", "magnetic_energy", r"B^2/2", ValueKind.SCALAR,
        (scalar_component("magnetic_energy", r"B^2/2"),),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE, requires_mhd=True,
    ),
    auxiliary_field(
        "Vorticity", "vorticity", r"\omega^i", ValueKind.VECTOR3,
        vector_components("vorticity", r"\omega"),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE, min_dimension=2,
    ),
    auxiliary_field(
        "MachNumber", "mach", r"\mathcal{M}", ValueKind.SCALAR,
        (scalar_component("mach", r"\mathcal{M}"),),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE,
    ),
    auxiliary_field(
        "PlasmaBeta", "plasma_beta", r"\beta_{\rm plasma}", ValueKind.SCALAR,
        (scalar_component("plasma_beta", r"\beta_{\rm plasma}"),),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE, requires_mhd=True,
    ),
    auxiliary_field(
        "EntropyProxy", "entropy", r"K", ValueKind.SCALAR,
        (scalar_component("entropy", r"K"),),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE,
    ),
    auxiliary_field(
        "AmrTagMetric", "amr_tag_metric", r"\eta_{\rm AMR}", ValueKind.SCALAR,
        (scalar_component("amr_tag_metric", r"\eta_{\rm AMR}"),),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE,
    ),
    auxiliary_field(
        "SolidMask", "solid_mask", r"M_{\rm solid}", ValueKind.SCALAR,
        (scalar_component("solid_mask", r"M_{\rm solid}"),),
        FieldRole.EVOLUTION_AUXILIARY, DiagnosticKind.STORED_STATE,
        EvolutionKind.AUXILIARY_STATE, requires_solid_geometry=True,
    ),
    auxiliary_field(
        "SolidVolumeFraction", "solid_volume_fraction", r"\kappa_{\rm fluid}",
        ValueKind.SCALAR,
        (scalar_component("solid_volume_fraction", r"\kappa_{\rm fluid}"),),
        FieldRole.EVOLUTION_AUXILIARY, DiagnosticKind.STORED_STATE,
        EvolutionKind.AUXILIARY_STATE, requires_solid_geometry=True,
    ),
    *tuple(
        auxiliary_field(
            f"SolidAperture{direction}", f"solid_aperture{direction}",
            rf"\alpha_{{{direction}}}", ValueKind.SCALAR,
            (scalar_component(
                f"solid_aperture{direction}", rf"\alpha_{{{direction}}}",
                centering=Centering[f"FACE{direction}"],
            ),),
            FieldRole.EVOLUTION_AUXILIARY, DiagnosticKind.STORED_STATE,
            EvolutionKind.AUXILIARY_STATE, requires_solid_geometry=True,
            min_dimension=direction,
        )
        for direction in range(1, 4)
    ),
    auxiliary_field(
        "SolidSweptVolume", "solid_swept_volume", r"\Delta V_{\rm solid}",
        ValueKind.SCALAR,
        (scalar_component("solid_swept_volume", r"\Delta V_{\rm solid}"),),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE, requires_solid_geometry=True,
    ),
    auxiliary_field(
        "SolidPhi", "solid_phi", r"\phi_{\rm solid}", ValueKind.SCALAR,
        (scalar_component("solid_phi", r"\phi_{\rm solid}"),),
        FieldRole.OUTPUT_ONLY_AUXILIARY, DiagnosticKind.PURE_AUXILIARY,
        EvolutionKind.NONE, requires_solid_geometry=True,
    ),
    auxiliary_field(
        "GravityPotential", "phi", r"\Phi", ValueKind.SCALAR,
        (scalar_component("phi", r"\Phi"),),
        FieldRole.EVOLUTION_AUXILIARY, DiagnosticKind.STORED_STATE,
        EvolutionKind.AUXILIARY_STATE, requires_gravity=True,
    ),
    auxiliary_field(
        "GravityGradient", "grad_phi", r"\nabla_i\Phi", ValueKind.VECTOR3,
        vector_components("grad_phi", r"\nabla\Phi"),
        FieldRole.CACHED_DERIVATIVE, DiagnosticKind.STORED_STATE,
        EvolutionKind.AUXILIARY_STATE, requires_gravity=True,
    ),
)


def all_fields() -> tuple[GridField, ...]:
    return (
        *FLUID_NEWTONIAN_FIELDS,
        *FLUID_RELATIVISTIC_FIELDS,
        MAGNETIC_FIELD,
        CT_MAGNETIC_FIELD,
        PASSIVE_SCALAR_FIELD,
        *Z4C_FIELDS,
        *AUXILIARY_FIELDS,
    )


def component_symbols(
    fields: tuple[GridField, ...],
    *,
    owner: str,
) -> dict[str, sp.Symbol]:
    """Map canonical component IDs to symbols used by generated C++ kernels."""
    symbols: dict[str, sp.Symbol] = {}
    for field in fields:
        for component in field.components:
            if component.cpp_id is None:
                continue
            if component.cpp_id in symbols:
                raise ValueError(f"duplicate component ID {component.cpp_id}")
            symbols[component.cpp_id] = sp.Symbol(component.cpp_symbol(owner))
    return symbols


def validate_field_catalog() -> None:
    auxiliary_ids = [field.aux_id for field in AUXILIARY_FIELDS]
    if len(auxiliary_ids) != len(set(auxiliary_ids)):
        raise ValueError("auxiliary field identifiers must be unique")
    for family in (FLUID_NEWTONIAN_FIELDS, FLUID_RELATIVISTIC_FIELDS, Z4C_FIELDS):
        component_names = [component.name for field in family for component in field.components]
        if len(component_names) != len(set(component_names)):
            raise ValueError("component names must be unique within a field set")
        if any(component.cpp_id is None for field in family for component in field.components):
            raise ValueError("evolved field components require C++ identifiers")


validate_field_catalog()
