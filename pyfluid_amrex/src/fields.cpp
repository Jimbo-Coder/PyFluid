#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/state.hpp"

#include <AMReX.H>

#include <array>
#include <sstream>

namespace pyfluid_amrex {

namespace {

using FieldTable = std::array<AuxGridFieldInfo, 18>;
using GFC = GridFieldCentering;
using Role = FieldRole;
using DiagKind = DiagnosticFieldKind;
using EvoKind = EvolutionFieldKind;
using Sym = GridFieldSymmetryKind;

constexpr std::array<const char*, MaxGridFieldComponents> scalar_components(const char* name)
{
   return {name, "", ""};
}

constexpr std::array<const char*, MaxGridFieldComponents> vector_components(
   const char* name1,
   const char* name2,
   const char* name3
)
{
   return {name1, name2, name3};
}

constexpr std::array<GridFieldCentering, MaxGridFieldComponents> scalar_centering(GridFieldCentering centering)
{
   return {centering, centering, centering};
}

constexpr std::array<GridFieldCentering, MaxGridFieldComponents> vector_centering(
   GridFieldCentering centering1,
   GridFieldCentering centering2,
   GridFieldCentering centering3
)
{
   return {centering1, centering2, centering3};
}

constexpr AuxGridFieldInfo aux_field(
   AuxGridField field,
   const char* name,
   std::array<const char*, MaxGridFieldComponents> component_names,
   std::array<GridFieldCentering, MaxGridFieldComponents> centerings,
   FieldRole role,
   DiagnosticFieldKind diagnostic_kind,
   EvolutionFieldKind evolution_kind,
   int components,
   bool requires_mhd,
   bool requires_solid_geometry,
   bool requires_gravity,
   int min_dimension
)
{
   return {
      field,
      name,
      component_names,
      centerings,
      role,
      diagnostic_kind,
      evolution_kind,
      components,
      requires_mhd,
      requires_solid_geometry,
      requires_gravity,
      min_dimension,
   };
}

constexpr std::array<StateGridFieldInfo, NVAR_MHD> state_grid_fields = {{
   {"rho", RHO, GFC::Cell, Sym::Scalar, false},
   {"mom1", MOM1, GFC::Cell, Sym::Vector1, false},
   {"mom2", MOM2, GFC::Cell, Sym::Vector2, false},
   {"mom3", MOM3, GFC::Cell, Sym::Vector3, false},
   {"energy", ENERGY, GFC::Cell, Sym::Scalar, false},
   {"b1", B1, GFC::Cell, Sym::Vector1, true},
   {"b2", B2, GFC::Cell, Sym::Vector2, true},
   {"b3", B3, GFC::Cell, Sym::Vector3, true},
}};

constexpr std::array<StateGridFieldInfo, NVAR_MHD> relativistic_state_grid_fields = {{
   {"dens", RHO, GFC::Cell, Sym::Scalar, false},
   {"mom1", MOM1, GFC::Cell, Sym::Vector1, false},
   {"mom2", MOM2, GFC::Cell, Sym::Vector2, false},
   {"mom3", MOM3, GFC::Cell, Sym::Vector3, false},
   {"tau", ENERGY, GFC::Cell, Sym::Scalar, false},
   {"b1", B1, GFC::Cell, Sym::Vector1, true},
   {"b2", B2, GFC::Cell, Sym::Vector2, true},
   {"b3", B3, GFC::Cell, Sym::Vector3, true},
}};

constexpr FieldTable aux_grid_fields = {{
   aux_field(
      AuxGridField::CurrentDensity,
      "j",
      vector_components("j1", "j2", "j3"),
      vector_centering(GFC::Edge1, GFC::Edge2, GFC::Edge3),
      Role::CachedDerivative,
      DiagKind::MixedAuxiliary,
      EvoKind::AuxiliaryState,
      3,
      true,
      false,
      false,
      2
   ),
   aux_field(
      AuxGridField::ElectricField,
      "e",
      vector_components("e1", "e2", "e3"),
      vector_centering(GFC::Edge1, GFC::Edge2, GFC::Edge3),
      Role::CachedDerivative,
      DiagKind::MixedAuxiliary,
      EvoKind::AuxiliaryState,
      3,
      true,
      false,
      false,
      2
   ),
   aux_field(
      AuxGridField::DivB,
      "divb",
      scalar_components("divb"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      1,
      true,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::MagneticEnergy,
      "magnetic_energy",
      scalar_components("magnetic_energy"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      1,
      true,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::Vorticity,
      "vorticity",
      vector_components("vorticity1", "vorticity2", "vorticity3"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      3,
      false,
      false,
      false,
      2
   ),
   aux_field(
      AuxGridField::MachNumber,
      "mach",
      scalar_components("mach"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      1,
      false,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::PlasmaBeta,
      "plasma_beta",
      scalar_components("plasma_beta"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      1,
      true,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::EntropyProxy,
      "entropy",
      scalar_components("entropy"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      1,
      false,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::AmrTagMetric,
      "amr_tag_metric",
      scalar_components("amr_tag_metric"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      1,
      false,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidMask,
      "solid_mask",
      scalar_components("solid_mask"),
      scalar_centering(GFC::Cell),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      1,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidVolumeFraction,
      "solid_volume_fraction",
      scalar_components("solid_volume_fraction"),
      scalar_centering(GFC::Cell),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      1,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidAperture1,
      "solid_aperture1",
      scalar_components("solid_aperture1"),
      scalar_centering(GFC::Face1),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      1,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidAperture2,
      "solid_aperture2",
      scalar_components("solid_aperture2"),
      scalar_centering(GFC::Face2),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      1,
      false,
      true,
      false,
      2
   ),
   aux_field(
      AuxGridField::SolidAperture3,
      "solid_aperture3",
      scalar_components("solid_aperture3"),
      scalar_centering(GFC::Face3),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      1,
      false,
      true,
      false,
      3
   ),
   aux_field(
      AuxGridField::SolidSweptVolume,
      "solid_swept_volume",
      scalar_components("solid_swept_volume"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      1,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidPhi,
      "solid_phi",
      scalar_components("solid_phi"),
      scalar_centering(GFC::Cell),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      1,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::GravityPotential,
      "phi",
      scalar_components("phi"),
      scalar_centering(GFC::Cell),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      1,
      false,
      false,
      true,
      1
   ),
   aux_field(
      AuxGridField::GravityGradient,
      "grad_phi",
      vector_components("grad_phi1", "grad_phi2", "grad_phi3"),
      scalar_centering(GFC::Cell),
      Role::CachedDerivative,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      3,
      false,
      false,
      true,
      1
   ),
}};

} // namespace

AuxGridField aux_grid_field_from_name(std::string const& name)
{
   for (AuxGridFieldInfo const& info : aux_grid_fields) {
      if (name == info.name) {
         return info.field;
      }
   }
   return AuxGridField::Unknown;
}

AuxGridFieldInfo aux_grid_field_info(AuxGridField field)
{
   for (AuxGridFieldInfo const& info : aux_grid_fields) {
      if (field == info.field) {
         return info;
      }
   }
   return {};
}

StateGridFieldInfo state_grid_field_info(int component)
{
   for (StateGridFieldInfo const& info : state_grid_fields) {
      if (component == info.component) {
         return info;
      }
   }
   if (component >= NVAR_HYDRO) {
      return {"dye_mass", component, GFC::Cell, Sym::Scalar, false};
   }
   return {};
}

int state_grid_field_origin_parity(int coordinate, MeshAxes const& axes, int component)
{
   const StateGridFieldInfo info = state_grid_field_info(component);
   int vector_component = -1;
   if (info.symmetry == Sym::Vector1) {
      vector_component = 0;
   } else if (info.symmetry == Sym::Vector2) {
      vector_component = 1;
   } else if (info.symmetry == Sym::Vector3) {
      vector_component = 2;
   } else {
      return 1;
   }

   const int axis = mesh_vector_axis_for_component(coordinate, axes, vector_component);
   if (coordinate == MeshCylindrical) {
      return (axis == MeshAxisR || axis == MeshAxisPhi) ? -1 : 1;
   }
   if (coordinate == MeshSpherical) {
      return (axis == MeshAxisR || axis == MeshAxisPhi) ? -1 : 1;
   }
   return axis == mesh_axis(axes, 0) ? -1 : 1;
}

std::string supported_aux_grid_field_names()
{
   std::ostringstream out;
   for (std::size_t i = 0; i < aux_grid_fields.size(); ++i) {
      if (i != 0) {
         out << ", ";
      }
      out << aux_grid_fields[i].name;
   }
   return out.str();
}

amrex::Vector<std::string> state_grid_field_names(PhysicalSystem system, int passive_scalars)
{
   amrex::Vector<std::string> names;
   if (physical_system_is_relativistic(system)) {
      const bool magnetized = physical_system_has_magnetic_field(system);
      for (StateGridFieldInfo const& info : relativistic_state_grid_fields) {
         if (info.requires_mhd && !magnetized) {
            continue;
         }
         names.push_back(info.name);
      }
   } else {
      const bool magnetized = physical_system_has_magnetic_field(system);
      for (StateGridFieldInfo const& info : state_grid_fields) {
         if (info.requires_mhd && !magnetized) {
            continue;
         }
         names.push_back(info.name);
      }
   }
   for (int n = 0; n < passive_scalars; ++n) {
      names.push_back(n == 0 ? "dye_mass" : "dye_mass_" + std::to_string(n));
   }
   return names;
}

std::vector<std::string> aux_grid_field_output_names(AuxGridField field)
{
   const AuxGridFieldInfo info = aux_grid_field_info(field);
   if (info.field == AuxGridField::Unknown) {
      return {};
   }
   std::vector<std::string> names;
   for (int n = 0; n < info.components; ++n) {
      const char* name = info.component_names[n];
      if (name == nullptr || name[0] == '\0') {
         amrex::Abort("aux grid field registry has an empty component name for " + std::string(info.name));
      }
      names.emplace_back(name);
   }
   return names;
}

std::string checkpoint_state_field_name()
{
   return "state";
}

std::string checkpoint_face_b_field_name(int direction)
{
   const StateGridFieldInfo info = state_grid_field_info(B1 + direction);
   if (info.name == nullptr || info.name[0] == '\0') {
      amrex::Abort("invalid face magnetic checkpoint direction");
   }
   return "face_" + std::string(info.name);
}

int aux_grid_field_component_count(AuxGridField field)
{
   return aux_grid_field_info(field).components;
}

} // namespace pyfluid_amrex
