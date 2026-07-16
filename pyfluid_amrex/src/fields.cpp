#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/state.hpp"

#include <AMReX.H>
#include <AMReX_IntVect.H>

#include <array>
#include <sstream>
#include <utility>

namespace pyfluid_amrex {

namespace {

using FieldTable = std::array<AuxGridFieldInfo, 18>;
using GFC = GridFieldCentering;
using Role = FieldRole;
using DiagKind = DiagnosticFieldKind;
using EvoKind = EvolutionFieldKind;
using Sym = GridFieldSymmetryKind;

std::vector<GridFieldComponentInfo> scalar_components(
   std::string_view name,
   GridFieldCentering centering = GFC::Cell
)
{
   return {{std::string(name), centering, Sym::Scalar}};
}

std::vector<GridFieldComponentInfo> vector_components(
   std::string_view name1,
   std::string_view name2,
   std::string_view name3,
   GFC centering1 = GFC::Cell,
   GFC centering2 = GFC::Cell,
   GFC centering3 = GFC::Cell
)
{
   return {
      {std::string(name1), centering1, Sym::Vector1},
      {std::string(name2), centering2, Sym::Vector2},
      {std::string(name3), centering3, Sym::Vector3},
   };
}

AuxGridFieldInfo aux_field(
   AuxGridField field,
   std::string_view name,
   std::vector<GridFieldComponentInfo> components,
   FieldRole role,
   DiagnosticFieldKind diagnostic_kind,
   EvolutionFieldKind evolution_kind,
   bool requires_mhd,
   bool requires_solid_geometry,
   bool requires_gravity,
   int min_dimension
)
{
   return {
      field,
      std::string(name),
      std::move(components),
      role,
      diagnostic_kind,
      evolution_kind,
      requires_mhd,
      requires_solid_geometry,
      requires_gravity,
      min_dimension,
   };
}

GridFieldInfo state_field(
   std::string_view name,
   GridFieldValueKind value_kind,
   std::vector<GridFieldComponentInfo> components,
   EvolutionFieldKind evolution_kind = EvolutionFieldKind::MainState
)
{
   return {
      std::string(name),
      value_kind,
      std::move(components),
      FieldRole::EvolvedState,
      evolution_kind,
      0,
      true,
      true,
   };
}

const FieldTable aux_grid_fields = {{
   aux_field(
      AuxGridField::CurrentDensity,
      "j",
      vector_components("j1", "j2", "j3", GFC::Edge1, GFC::Edge2, GFC::Edge3),
      Role::CachedDerivative,
      DiagKind::MixedAuxiliary,
      EvoKind::AuxiliaryState,
      true,
      false,
      false,
      2
   ),
   aux_field(
      AuxGridField::ElectricField,
      "e",
      vector_components("e1", "e2", "e3", GFC::Edge1, GFC::Edge2, GFC::Edge3),
      Role::CachedDerivative,
      DiagKind::MixedAuxiliary,
      EvoKind::AuxiliaryState,
      true,
      false,
      false,
      2
   ),
   aux_field(
      AuxGridField::DivB,
      "divb",
      scalar_components("divb"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      true,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::MagneticEnergy,
      "magnetic_energy",
      scalar_components("magnetic_energy"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      true,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::Vorticity,
      "vorticity",
      vector_components("vorticity1", "vorticity2", "vorticity3"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      false,
      false,
      false,
      2
   ),
   aux_field(
      AuxGridField::MachNumber,
      "mach",
      scalar_components("mach"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      false,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::PlasmaBeta,
      "plasma_beta",
      scalar_components("plasma_beta"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      true,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::EntropyProxy,
      "entropy",
      scalar_components("entropy"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      false,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::AmrTagMetric,
      "amr_tag_metric",
      scalar_components("amr_tag_metric"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      false,
      false,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidMask,
      "solid_mask",
      scalar_components("solid_mask"),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidVolumeFraction,
      "solid_volume_fraction",
      scalar_components("solid_volume_fraction"),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidAperture1,
      "solid_aperture1",
      scalar_components("solid_aperture1", GFC::Face1),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidAperture2,
      "solid_aperture2",
      scalar_components("solid_aperture2", GFC::Face2),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      false,
      true,
      false,
      2
   ),
   aux_field(
      AuxGridField::SolidAperture3,
      "solid_aperture3",
      scalar_components("solid_aperture3", GFC::Face3),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      false,
      true,
      false,
      3
   ),
   aux_field(
      AuxGridField::SolidSweptVolume,
      "solid_swept_volume",
      scalar_components("solid_swept_volume"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::SolidPhi,
      "solid_phi",
      scalar_components("solid_phi"),
      Role::OutputOnlyAuxiliary,
      DiagKind::PureAuxiliary,
      EvoKind::None,
      false,
      true,
      false,
      1
   ),
   aux_field(
      AuxGridField::GravityPotential,
      "phi",
      scalar_components("phi"),
      Role::EvolutionAuxiliary,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
      false,
      false,
      true,
      1
   ),
   aux_field(
      AuxGridField::GravityGradient,
      "grad_phi",
      vector_components("grad_phi1", "grad_phi2", "grad_phi3"),
      Role::CachedDerivative,
      DiagKind::StoredState,
      EvoKind::AuxiliaryState,
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

std::vector<GridFieldCentering> const& grid_field_centerings()
{
   static const std::vector<GridFieldCentering> centerings = {
      GFC::Cell,
      GFC::Vertex,
      GFC::Face1,
      GFC::Face2,
      GFC::Face3,
      GFC::Edge1,
      GFC::Edge2,
      GFC::Edge3,
   };
   return centerings;
}

GridFieldCentering face_grid_field_centering(int direction)
{
   switch (direction) {
   case 0:
      return GFC::Face1;
   case 1:
      return GFC::Face2;
   case 2:
      return GFC::Face3;
   default:
      amrex::Abort("invalid face-centered grid-field direction");
      return GFC::Face1;
   }
}

GridFieldCentering edge_grid_field_centering(int component)
{
   switch (component) {
   case 0:
      return GFC::Edge1;
   case 1:
      return GFC::Edge2;
   case 2:
      return GFC::Edge3;
   default:
      amrex::Abort("invalid edge-centered grid-field component");
      return GFC::Edge1;
   }
}

amrex::IndexType grid_field_index_type(GridFieldCentering centering)
{
   amrex::IntVect type(AMREX_D_DECL(0, 0, 0));
   switch (centering) {
   case GFC::Cell:
      break;
   case GFC::Vertex:
      type = amrex::IntVect::TheNodeVector();
      break;
   case GFC::Face1:
      type = amrex::IntVect::TheDimensionVector(0);
      break;
   case GFC::Face2:
#if (AMREX_SPACEDIM >= 2)
      type = amrex::IntVect::TheDimensionVector(1);
      break;
#else
      amrex::Abort("face2 centering requires at least two dimensions");
#endif
   case GFC::Face3:
#if (AMREX_SPACEDIM == 3)
      type = amrex::IntVect::TheDimensionVector(2);
      break;
#else
      amrex::Abort("face3 centering requires three dimensions");
#endif
   case GFC::Edge1:
#if (AMREX_SPACEDIM == 2)
      type = amrex::IntVect(AMREX_D_DECL(0, 1, 0));
#elif (AMREX_SPACEDIM == 3)
      type = amrex::IntVect(AMREX_D_DECL(0, 1, 1));
#endif
      break;
   case GFC::Edge2:
#if (AMREX_SPACEDIM == 2)
      type = amrex::IntVect(AMREX_D_DECL(1, 0, 0));
#elif (AMREX_SPACEDIM == 3)
      type = amrex::IntVect(AMREX_D_DECL(1, 0, 1));
#endif
      break;
   case GFC::Edge3:
#if (AMREX_SPACEDIM == 2)
      type = amrex::IntVect(AMREX_D_DECL(1, 1, 0));
#elif (AMREX_SPACEDIM == 3)
      type = amrex::IntVect(AMREX_D_DECL(1, 1, 0));
#endif
      break;
   }
   return amrex::IndexType(type);
}

std::string grid_field_stem_suffix(GridFieldCentering centering)
{
   switch (centering) {
   case GFC::Cell:
      return "";
   case GFC::Vertex:
      return "_vertex";
   case GFC::Face1:
      return "_face1";
   case GFC::Face2:
      return "_face2";
   case GFC::Face3:
      return "_face3";
   case GFC::Edge1:
      return "_edge1";
   case GFC::Edge2:
      return "_edge2";
   case GFC::Edge3:
      return "_edge3";
   }
   amrex::Abort("unknown grid-field centering");
   return {};
}

StateGridFieldInfo state_grid_field_info(int component)
{
   const GridFieldSetInfo state = fluid_state_grid_fields(PhysicalSystem::MHD, 0);
   int flat_component = 0;
   for (GridFieldInfo const& field : state.fields) {
      for (GridFieldComponentInfo const& info : field.components) {
         if (component == flat_component) {
            return {
               info.name,
               component,
               info.centering,
               info.symmetry,
               component >= B1 && component <= B3,
            };
         }
         ++flat_component;
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
   return grid_field_component_names(fluid_state_grid_fields(system, passive_scalars));
}

GridFieldSetInfo fluid_state_grid_fields(PhysicalSystem system, int passive_scalars)
{
   const bool relativistic = physical_system_is_relativistic(system);
   const bool magnetized = physical_system_has_magnetic_field(system);
   GridFieldSetInfo set{
      "fluid_state",
      EvolutionSector::Fluid,
      {
         state_field(
            relativistic ? "dens" : "rho",
            GridFieldValueKind::Scalar,
            scalar_components(relativistic ? "dens" : "rho")
         ),
         state_field(
            "momentum",
            GridFieldValueKind::Vector3,
            vector_components("mom1", "mom2", "mom3")
         ),
         state_field(
            relativistic ? "tau" : "energy",
            GridFieldValueKind::Scalar,
            scalar_components(relativistic ? "tau" : "energy")
         ),
      },
   };
   if (magnetized) {
      set.fields.push_back(state_field(
         "magnetic_field",
         GridFieldValueKind::Vector3,
         vector_components("b1", "b2", "b3")
      ));
   }
   for (int n = 0; n < passive_scalars; ++n) {
      const std::string component_name = n == 0 ? "dye_mass" : "dye_mass_" + std::to_string(n);
      set.fields.push_back(state_field(
         component_name,
         GridFieldValueKind::Scalar,
         scalar_components(component_name),
         EvolutionFieldKind::NonBackreactingExtra
      ));
   }
   return set;
}

std::vector<std::string> aux_grid_field_output_names(AuxGridField field)
{
   const AuxGridFieldInfo info = aux_grid_field_info(field);
   if (info.field == AuxGridField::Unknown) {
      return {};
   }
   std::vector<std::string> names;
   for (GridFieldComponentInfo const& component : info.components) {
      if (component.name.empty()) {
         amrex::Abort("aux grid field registry has an empty component name for " + std::string(info.name));
      }
      names.emplace_back(component.name);
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
   if (info.name.empty()) {
      amrex::Abort("invalid face magnetic checkpoint direction");
   }
   return "face_" + info.name;
}

int aux_grid_field_component_count(AuxGridField field)
{
   return static_cast<int>(aux_grid_field_info(field).components.size());
}

int grid_field_component_count(GridFieldSetInfo const& set)
{
   int count = 0;
   for (GridFieldInfo const& field : set.fields) {
      count += static_cast<int>(field.components.size());
   }
   return count;
}

amrex::Vector<std::string> grid_field_component_names(GridFieldSetInfo const& set)
{
   amrex::Vector<std::string> names;
   names.reserve(grid_field_component_count(set));
   for (GridFieldInfo const& field : set.fields) {
      for (GridFieldComponentInfo const& component : field.components) {
         names.emplace_back(component.name);
      }
   }
   return names;
}

} // namespace pyfluid_amrex
