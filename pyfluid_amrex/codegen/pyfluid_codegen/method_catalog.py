from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "methods.toml"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RotationLaw:
    name: str
    cpp_id: str
    label: str
    description: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class DiskRotationLaw:
    name: str
    cpp_id: str
    label: str
    description: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class InitialDataPerturbation:
    name: str
    cpp_id: str
    label: str
    description: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class InitialDataMagneticSeed:
    name: str
    cpp_id: str
    label: str
    description: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class InitialDataMagneticEquilibrium:
    name: str
    cpp_id: str
    label: str
    description: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class ReconstructionMethod:
    name: str
    cpp_id: str
    label: str
    description: str
    stencil_radius: int
    characteristic: bool


@dataclass(frozen=True)
class AmrTagPolicy:
    name: str
    cpp_id: str
    label: str
    description: str
    default_threshold: float
    density: bool
    pressure: bool
    velocity: bool
    magnetic: bool
    spacetime: bool
    density_floor_multiple: float


@dataclass(frozen=True)
class InitialDataOwner:
    name: str
    cpp_id: str
    label: str
    description: str
    dimension: int
    coordinate: str
    axes: tuple[str, ...]
    central_objects: tuple[str, ...]
    spacetime_methods: tuple[str, ...]
    disk_kinds: tuple[str, ...]
    disk_gravity_modes: tuple[str, ...]
    grid_kinds: tuple[str, ...]
    relative_tilt: bool
    reflection_class: str


def _reject_unknown_keys(
    record: dict[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")


def _required_string(record: dict[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a nonempty string")
    return value


def _required_string_array(
    record: dict[str, Any], key: str, context: str
) -> tuple[str, ...]:
    values = record.get(key)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
    ):
        raise ValueError(f"{context}.{key} must be a nonempty string array")
    if len(values) != len(set(values)):
        raise ValueError(f"{context}.{key} contains duplicates")
    return tuple(values)


def _load_rotation_law(record: dict[str, Any], index: int) -> RotationLaw:
    context = f"rotation_law[{index}]"
    _reject_unknown_keys(
        record,
        {"name", "cpp_id", "label", "description", "parameters"},
        context,
    )
    parameters = record.get("parameters")
    if (
        not isinstance(parameters, list)
        or not parameters
        or not all(isinstance(parameter, str) and parameter for parameter in parameters)
    ):
        raise ValueError(f"{context}.parameters must be a nonempty string array")
    if len(parameters) != len(set(parameters)):
        raise ValueError(f"{context}.parameters contains duplicates")
    return RotationLaw(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        parameters=tuple(parameters),
    )


def _load_disk_rotation_law(
    record: dict[str, Any], index: int
) -> DiskRotationLaw:
    context = f"disk_rotation_law[{index}]"
    _reject_unknown_keys(
        record,
        {"name", "cpp_id", "label", "description", "parameters"},
        context,
    )
    parameters = record.get("parameters")
    if (
        not isinstance(parameters, list)
        or not parameters
        or not all(isinstance(parameter, str) and parameter for parameter in parameters)
    ):
        raise ValueError(f"{context}.parameters must be a nonempty string array")
    if len(parameters) != len(set(parameters)):
        raise ValueError(f"{context}.parameters contains duplicates")
    return DiskRotationLaw(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        parameters=tuple(parameters),
    )


def _load_initial_data_perturbation(
    record: dict[str, Any], index: int
) -> InitialDataPerturbation:
    context = f"initial_data_perturbation[{index}]"
    _reject_unknown_keys(
        record,
        {"name", "cpp_id", "label", "description", "parameters"},
        context,
    )
    parameters = record.get("parameters")
    if (
        not isinstance(parameters, list)
        or not all(isinstance(parameter, str) and parameter for parameter in parameters)
    ):
        raise ValueError(f"{context}.parameters must be a string array")
    if len(parameters) != len(set(parameters)):
        raise ValueError(f"{context}.parameters contains duplicates")
    return InitialDataPerturbation(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        parameters=tuple(parameters),
    )


def _load_initial_data_magnetic_seed(
    record: dict[str, Any], index: int
) -> InitialDataMagneticSeed:
    context = f"initial_data_magnetic_seed[{index}]"
    _reject_unknown_keys(
        record,
        {"name", "cpp_id", "label", "description", "parameters"},
        context,
    )
    parameters = record.get("parameters")
    if (
        not isinstance(parameters, list)
        or not all(isinstance(parameter, str) and parameter for parameter in parameters)
    ):
        raise ValueError(f"{context}.parameters must be a string array")
    if len(parameters) != len(set(parameters)):
        raise ValueError(f"{context}.parameters contains duplicates")
    return InitialDataMagneticSeed(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        parameters=tuple(parameters),
    )


def _load_initial_data_magnetic_equilibrium(
    record: dict[str, Any], index: int
) -> InitialDataMagneticEquilibrium:
    context = f"initial_data_magnetic_equilibrium[{index}]"
    _reject_unknown_keys(
        record,
        {"name", "cpp_id", "label", "description", "parameters"},
        context,
    )
    parameters = record.get("parameters")
    if (
        not isinstance(parameters, list)
        or not all(isinstance(parameter, str) and parameter for parameter in parameters)
    ):
        raise ValueError(f"{context}.parameters must be a string array")
    if len(parameters) != len(set(parameters)):
        raise ValueError(f"{context}.parameters contains duplicates")
    return InitialDataMagneticEquilibrium(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        parameters=tuple(parameters),
    )


def _load_reconstruction(
    record: dict[str, Any], index: int
) -> ReconstructionMethod:
    context = f"reconstruction[{index}]"
    _reject_unknown_keys(
        record,
        {
            "name",
            "cpp_id",
            "label",
            "description",
            "stencil_radius",
            "characteristic",
        },
        context,
    )
    stencil_radius = record.get("stencil_radius")
    if type(stencil_radius) is not int or stencil_radius < 0:
        raise ValueError(f"{context}.stencil_radius must be a nonnegative integer")
    characteristic = record.get("characteristic")
    if type(characteristic) is not bool:
        raise ValueError(f"{context}.characteristic must be a boolean")
    if characteristic and stencil_radius < 1:
        raise ValueError(
            f"{context} characteristic reconstruction needs a nonzero stencil"
        )
    return ReconstructionMethod(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        stencil_radius=stencil_radius,
        characteristic=characteristic,
    )


def _load_amr_tag_policy(
    record: dict[str, Any], index: int
) -> AmrTagPolicy:
    context = f"amr_tag_policy[{index}]"
    _reject_unknown_keys(
        record,
        {
            "name",
            "cpp_id",
            "label",
            "description",
            "default_threshold",
            "density",
            "pressure",
            "velocity",
            "magnetic",
            "spacetime",
            "density_floor_multiple",
        },
        context,
    )
    flags: dict[str, bool] = {}
    for key in ("density", "pressure", "velocity", "magnetic", "spacetime"):
        value = record.get(key)
        if type(value) is not bool:
            raise ValueError(f"{context}.{key} must be a boolean")
        flags[key] = value
    default_threshold = record.get("default_threshold")
    if (
        type(default_threshold) not in (int, float)
        or default_threshold <= 0.0
    ):
        raise ValueError(f"{context}.default_threshold must be positive")
    density_floor_multiple = record.get("density_floor_multiple")
    if (
        type(density_floor_multiple) not in (int, float)
        or density_floor_multiple < 0.0
    ):
        raise ValueError(
            f"{context}.density_floor_multiple must be a nonnegative number"
        )
    if not flags["density"] and density_floor_multiple != 0.0:
        raise ValueError(
            f"{context}.density_floor_multiple requires density = true"
        )
    if index > 0 and not any(flags.values()):
        raise ValueError(f"{context} must enable at least one indicator")
    return AmrTagPolicy(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        default_threshold=float(default_threshold),
        density_floor_multiple=float(density_floor_multiple),
        **flags,
    )


def _load_initial_data_owner(
    record: dict[str, Any], index: int
) -> InitialDataOwner:
    context = f"initial_data_owner[{index}]"
    _reject_unknown_keys(
        record,
        {
            "name",
            "cpp_id",
            "label",
            "description",
            "dimension",
            "coordinate",
            "axes",
            "central_objects",
            "spacetime_methods",
            "disk_kinds",
            "disk_gravity_modes",
            "grid_kinds",
            "relative_tilt",
            "reflection_class",
        },
        context,
    )
    dimension = record.get("dimension")
    if type(dimension) is not int or dimension not in (1, 2, 3):
        raise ValueError(f"{context}.dimension must be 1, 2, or 3")
    coordinate = _required_string(record, "coordinate", context)
    axes = _required_string_array(record, "axes", context)
    canonical_axes = {
        "cartesian": ("x", "y", "z"),
        "cylindrical": ("r", "phi", "z"),
        "spherical": ("r", "theta", "phi"),
    }
    if coordinate not in canonical_axes:
        raise ValueError(
            f"{context}.coordinate must be cartesian, cylindrical, or spherical"
        )
    if len(axes) != dimension:
        raise ValueError(f"{context}.axes must contain dimension entries")
    canonical = canonical_axes[coordinate]
    try:
        positions = tuple(canonical.index(axis) for axis in axes)
    except ValueError as error:
        raise ValueError(
            f"{context}.axes contains an axis outside the coordinate family"
        ) from error
    if tuple(sorted(positions)) != positions:
        raise ValueError(f"{context}.axes must preserve canonical order")
    relative_tilt = record.get("relative_tilt")
    if type(relative_tilt) is not bool:
        raise ValueError(f"{context}.relative_tilt must be a boolean")
    reflection_class = _required_string(record, "reflection_class", context)
    if reflection_class not in {
        "spherical",
        "axis_equatorial",
        "object_dependent",
    }:
        raise ValueError(f"{context}.reflection_class is unsupported")
    central_objects = _required_string_array(
        record, "central_objects", context
    )
    if not set(central_objects) <= {"tov", "rotating_star", "black_hole"}:
        raise ValueError(f"{context}.central_objects contains an unsupported value")
    spacetime_methods = _required_string_array(
        record, "spacetime_methods", context
    )
    if not set(spacetime_methods) <= {"tov", "ctt", "cttk", "xcts", "kerr_schild"}:
        raise ValueError(f"{context}.spacetime_methods contains an unsupported value")
    disk_kinds = _required_string_array(record, "disk_kinds", context)
    if not set(disk_kinds) <= {"none", "equilibrium_torus", "tilted_torus"}:
        raise ValueError(f"{context}.disk_kinds contains an unsupported value")
    disk_gravity_modes = _required_string_array(
        record, "disk_gravity_modes", context
    )
    if not set(disk_gravity_modes) <= {"none", "test_fluid", "coupled"}:
        raise ValueError(
            f"{context}.disk_gravity_modes contains an unsupported value"
        )
    grid_kinds = _required_string_array(record, "grid_kinds", context)
    if not set(grid_kinds) <= {
        "uniform",
        "mapped_radial",
        "uniform_shell",
        "mapped_radial_shell",
        "block_amr",
    }:
        raise ValueError(f"{context}.grid_kinds contains an unsupported value")
    if relative_tilt and "tilted_torus" not in disk_kinds:
        raise ValueError(
            f"{context}.relative_tilt requires tilted_torus disk support"
        )
    return InitialDataOwner(
        name=_required_string(record, "name", context),
        cpp_id=_required_string(record, "cpp_id", context),
        label=_required_string(record, "label", context),
        description=_required_string(record, "description", context),
        dimension=dimension,
        coordinate=coordinate,
        axes=axes,
        central_objects=central_objects,
        spacetime_methods=spacetime_methods,
        disk_kinds=disk_kinds,
        disk_gravity_modes=disk_gravity_modes,
        grid_kinds=grid_kinds,
        relative_tilt=relative_tilt,
        reflection_class=reflection_class,
    )


def _validate_unique_ids(records: tuple[Any, ...], context: str) -> None:
    names = [record.name for record in records]
    cpp_ids = [record.cpp_id for record in records]
    if len(names) != len(set(names)):
        raise ValueError(f"{context} names must be unique")
    if len(cpp_ids) != len(set(cpp_ids)):
        raise ValueError(f"{context} cpp_id values must be unique")


def _load_schema() -> tuple[
    tuple[ReconstructionMethod, ...],
    tuple[AmrTagPolicy, ...],
    tuple[RotationLaw, ...],
    tuple[DiskRotationLaw, ...],
    tuple[InitialDataPerturbation, ...],
    tuple[InitialDataMagneticSeed, ...],
    tuple[InitialDataMagneticEquilibrium, ...],
    tuple[InitialDataOwner, ...],
]:
    with SCHEMA_PATH.open("rb") as stream:
        schema = tomllib.load(stream)
    _reject_unknown_keys(
        schema,
        {
            "schema_version",
            "reconstruction",
            "amr_tag_policy",
            "rotation_law",
            "disk_rotation_law",
            "initial_data_perturbation",
            "initial_data_magnetic_seed",
            "initial_data_magnetic_equilibrium",
            "initial_data_owner",
        },
        "methods schema",
    )
    schema_version = schema.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported methods schema version {schema_version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    reconstruction_records = schema.get("reconstruction")
    if not isinstance(reconstruction_records, list) or not reconstruction_records:
        raise ValueError("reconstruction must be a nonempty array of tables")
    reconstruction_methods = tuple(
        _load_reconstruction(record, index)
        for index, record in enumerate(reconstruction_records)
        if isinstance(record, dict)
    )
    if len(reconstruction_methods) != len(reconstruction_records):
        raise ValueError("reconstruction entries must be tables")
    _validate_unique_ids(reconstruction_methods, "reconstruction")

    amr_tag_policy_records = schema.get("amr_tag_policy")
    if not isinstance(amr_tag_policy_records, list) or not amr_tag_policy_records:
        raise ValueError("amr_tag_policy must be a nonempty array of tables")
    amr_tag_policies = tuple(
        _load_amr_tag_policy(record, index)
        for index, record in enumerate(amr_tag_policy_records)
        if isinstance(record, dict)
    )
    if len(amr_tag_policies) != len(amr_tag_policy_records):
        raise ValueError("amr_tag_policy entries must be tables")
    _validate_unique_ids(amr_tag_policies, "amr_tag_policy")
    if amr_tag_policies[0].name != "none" or any(
        (
            amr_tag_policies[0].density,
            amr_tag_policies[0].pressure,
            amr_tag_policies[0].velocity,
            amr_tag_policies[0].magnetic,
            amr_tag_policies[0].spacetime,
        )
    ):
        raise ValueError("amr_tag_policy[0] must be the inactive none policy")

    rotation_records = schema.get("rotation_law")
    if not isinstance(rotation_records, list) or not rotation_records:
        raise ValueError("rotation_law must be a nonempty array of tables")
    laws = tuple(
        _load_rotation_law(record, index)
        for index, record in enumerate(rotation_records)
        if isinstance(record, dict)
    )
    if len(laws) != len(rotation_records):
        raise ValueError("rotation_law entries must be tables")
    _validate_unique_ids(laws, "rotation_law")

    disk_rotation_records = schema.get("disk_rotation_law")
    if not isinstance(disk_rotation_records, list) or not disk_rotation_records:
        raise ValueError("disk_rotation_law must be a nonempty array of tables")
    disk_laws = tuple(
        _load_disk_rotation_law(record, index)
        for index, record in enumerate(disk_rotation_records)
        if isinstance(record, dict)
    )
    if len(disk_laws) != len(disk_rotation_records):
        raise ValueError("disk_rotation_law entries must be tables")
    _validate_unique_ids(disk_laws, "disk_rotation_law")
    perturbation_records = schema.get("initial_data_perturbation")
    if not isinstance(perturbation_records, list) or not perturbation_records:
        raise ValueError(
            "initial_data_perturbation must be a nonempty array of tables"
        )
    perturbations = tuple(
        _load_initial_data_perturbation(record, index)
        for index, record in enumerate(perturbation_records)
        if isinstance(record, dict)
    )
    if len(perturbations) != len(perturbation_records):
        raise ValueError("initial_data_perturbation entries must be tables")
    _validate_unique_ids(perturbations, "initial_data_perturbation")
    magnetic_seed_records = schema.get("initial_data_magnetic_seed")
    if not isinstance(magnetic_seed_records, list) or not magnetic_seed_records:
        raise ValueError(
            "initial_data_magnetic_seed must be a nonempty array of tables"
        )
    magnetic_seeds = tuple(
        _load_initial_data_magnetic_seed(record, index)
        for index, record in enumerate(magnetic_seed_records)
        if isinstance(record, dict)
    )
    if len(magnetic_seeds) != len(magnetic_seed_records):
        raise ValueError("initial_data_magnetic_seed entries must be tables")
    _validate_unique_ids(magnetic_seeds, "initial_data_magnetic_seed")
    magnetic_equilibrium_records = schema.get(
        "initial_data_magnetic_equilibrium"
    )
    if (
        not isinstance(magnetic_equilibrium_records, list)
        or not magnetic_equilibrium_records
    ):
        raise ValueError(
            "initial_data_magnetic_equilibrium must be a nonempty array of tables"
        )
    magnetic_equilibria = tuple(
        _load_initial_data_magnetic_equilibrium(record, index)
        for index, record in enumerate(magnetic_equilibrium_records)
        if isinstance(record, dict)
    )
    if len(magnetic_equilibria) != len(magnetic_equilibrium_records):
        raise ValueError(
            "initial_data_magnetic_equilibrium entries must be tables"
        )
    _validate_unique_ids(
        magnetic_equilibria, "initial_data_magnetic_equilibrium"
    )
    owner_records = schema.get("initial_data_owner")
    if not isinstance(owner_records, list) or not owner_records:
        raise ValueError("initial_data_owner must be a nonempty array of tables")
    owners = tuple(
        _load_initial_data_owner(record, index)
        for index, record in enumerate(owner_records)
        if isinstance(record, dict)
    )
    if len(owners) != len(owner_records):
        raise ValueError("initial_data_owner entries must be tables")
    _validate_unique_ids(owners, "initial_data_owner")
    return (
        reconstruction_methods,
        amr_tag_policies,
        laws,
        disk_laws,
        perturbations,
        magnetic_seeds,
        magnetic_equilibria,
        owners,
    )


(
    RECONSTRUCTION_METHODS,
    AMR_TAG_POLICIES,
    ROTATION_LAWS,
    DISK_ROTATION_LAWS,
    INITIAL_DATA_PERTURBATIONS,
    INITIAL_DATA_MAGNETIC_SEEDS,
    INITIAL_DATA_MAGNETIC_EQUILIBRIA,
    INITIAL_DATA_OWNERS,
) = _load_schema()
