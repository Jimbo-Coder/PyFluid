#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/generated/field_catalog.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/state.hpp"

#include <AMReX.H>
#include <AMReX_IntVect.H>

#include <algorithm>
#include <sstream>

namespace pyfluid_amrex {

using GFC = GridFieldCentering;
using Sym = GridFieldSymmetryKind;

AuxGridField aux_grid_field_from_name(std::string const& name)
{
   for (AuxGridFieldInfo const& info : generated::auxiliary_grid_fields()) {
      if (name == info.name) {
         return info.field;
      }
   }
   return AuxGridField::Unknown;
}

AuxGridFieldInfo aux_grid_field_info(AuxGridField field)
{
   for (AuxGridFieldInfo const& info : generated::auxiliary_grid_fields()) {
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

StateGridFieldInfo state_grid_field_info(
   PhysicalSystem system,
   int passive_scalars,
   int component
)
{
   const GridFieldSetInfo state = fluid_state_grid_fields(system, passive_scalars);
   int flat_component = 0;
   for (GridFieldInfo const& field : state.fields) {
      for (GridFieldComponentInfo const& info : field.components) {
         if (component == flat_component) {
            return {
               info.name,
               info.label,
               component,
               info.centering,
               info.symmetry,
               physical_system_has_magnetic_field(system)
                  && component >= B1 && component <= B3,
            };
         }
         ++flat_component;
      }
   }
   return {};
}

int state_grid_field_origin_parity(
   int coordinate,
   MeshAxes const& axes,
   PhysicalSystem system,
   int passive_scalars,
   int component
)
{
   const StateGridFieldInfo info = state_grid_field_info(system, passive_scalars, component);
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
   auto const& fields = generated::auxiliary_grid_fields();
   std::ostringstream out;
   for (std::size_t i = 0; i < fields.size(); ++i) {
      if (i != 0) {
         out << ", ";
      }
      out << fields[i].name;
   }
   return out.str();
}

amrex::Vector<std::string> state_grid_field_names(PhysicalSystem system, int passive_scalars)
{
   return grid_field_component_names(fluid_state_grid_fields(system, passive_scalars));
}

GridFieldSetInfo fluid_state_grid_fields(PhysicalSystem system, int passive_scalars)
{
   return generated::fluid_state_grid_fields(system, passive_scalars);
}

GridFieldSetInfo physical_fluid_grid_fields(
   PhysicalSystem system,
   int passive_scalars
)
{
   return generated::physical_fluid_grid_fields(system, passive_scalars);
}

GridFieldSetInfo const& adm_initial_data_grid_fields()
{
   return generated::adm_initial_data_grid_fields();
}

GridFieldSetInfo const& z4c_state_grid_fields()
{
   return generated::z4c_state_grid_fields();
}

GridFieldInfo const& ct_magnetic_grid_field()
{
   return generated::ct_magnetic_grid_field();
}

amrex::Vector<std::string> z4c_state_grid_field_names()
{
   return grid_field_component_names(z4c_state_grid_fields());
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
         amrex::Abort(
            "aux grid field registry has an empty component name for " + info.name
         );
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
   GridFieldInfo const& field = ct_magnetic_grid_field();
   if (direction < 0 || direction >= static_cast<int>(field.components.size())) {
      amrex::Abort("invalid face magnetic checkpoint direction");
   }
   return field.components[direction].name;
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

int scalar_grid_field_component(
   GridFieldSetInfo const& set,
   std::string const& name
)
{
   int component = 0;
   for (GridFieldInfo const& field : set.fields) {
      const bool scalar = field.value_kind == GridFieldValueKind::Scalar
         && field.components.size() == 1;
      for (GridFieldComponentInfo const& field_component : field.components) {
         if (scalar && (name == field.name || name == field_component.name)) {
            return component;
         }
         ++component;
      }
   }
   return -1;
}

int grid_field_ghost_width(GridFieldSetInfo const& set)
{
   int ghost_width = 0;
   for (GridFieldInfo const& field : set.fields) {
      ghost_width = std::max(ghost_width, field.ghost_width);
   }
   return ghost_width;
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
