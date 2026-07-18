#pragma once

#include <AMReX_IndexType.H>
#include <AMReX_Vector.H>

#include <string>
#include <string_view>
#include <vector>

#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/generated/field_ids.hpp"

namespace pyfluid_amrex {

enum class GridFieldCentering {
   Cell,
   Vertex,
   Face1,
   Face2,
   Face3,
   Edge1,
   Edge2,
   Edge3,
};

enum class FieldRole {
   EvolvedState,
   EvolutionAuxiliary,
   CachedDerivative,
   OutputOnlyAuxiliary,
   SolverScratch,
};

enum class DiagnosticFieldKind {
   StoredState,
   PureAuxiliary,
   MixedAuxiliary,
};

enum class EvolutionFieldKind {
   MainState,
   AuxiliaryState,
   NonBackreactingExtra,
   None,
};

enum class GridFieldSymmetryKind {
   Scalar,
   Vector1,
   Vector2,
   Vector3,
   Tensor11,
   Tensor12,
   Tensor13,
   Tensor22,
   Tensor23,
   Tensor33,
};

enum class GridFieldValueKind {
   Scalar,
   Vector3,
   Vector4,
   SymmetricTensor3,
   Tensor3,
   Tensor4,
};

struct GridFieldComponentInfo {
   std::string name{};
   std::string label{};
   GridFieldCentering centering = GridFieldCentering::Cell;
   GridFieldSymmetryKind symmetry = GridFieldSymmetryKind::Scalar;
};

struct GridFieldInfo {
   std::string name{};
   std::string label{};
   GridFieldValueKind value_kind = GridFieldValueKind::Scalar;
   std::vector<GridFieldComponentInfo> components{};
   FieldRole role = FieldRole::EvolvedState;
   EvolutionFieldKind evolution_kind = EvolutionFieldKind::MainState;
   int ghost_width = 0;
   bool checkpoint = true;
   bool output = true;
};

struct GridFieldSetInfo {
   std::string name{};
   std::string label{};
   EvolutionSector sector = EvolutionSector::None;
   std::vector<GridFieldInfo> fields{};
};

struct AuxGridFieldInfo {
   AuxGridField field = AuxGridField::Unknown;
   std::string name{};
   std::string label{};
   std::vector<GridFieldComponentInfo> components{};
   FieldRole role = FieldRole::OutputOnlyAuxiliary;
   DiagnosticFieldKind diagnostic_kind = DiagnosticFieldKind::PureAuxiliary;
   EvolutionFieldKind evolution_kind = EvolutionFieldKind::None;
   bool requires_mhd = false;
   bool requires_solid_geometry = false;
   bool requires_gravity = false;
   int min_dimension = 1;
};

struct StateGridFieldInfo {
   std::string name{};
   std::string label{};
   int component = -1;
   GridFieldCentering centering = GridFieldCentering::Cell;
   GridFieldSymmetryKind symmetry = GridFieldSymmetryKind::Scalar;
   bool requires_mhd = false;
};

AuxGridField aux_grid_field_from_name(std::string const& name);
AuxGridFieldInfo aux_grid_field_info(AuxGridField field);
std::vector<GridFieldCentering> const& grid_field_centerings();
GridFieldCentering face_grid_field_centering(int direction);
GridFieldCentering edge_grid_field_centering(int component);
amrex::IndexType grid_field_index_type(GridFieldCentering centering);
std::string grid_field_stem_suffix(GridFieldCentering centering);
StateGridFieldInfo state_grid_field_info(
   PhysicalSystem system,
   int passive_scalars,
   int component
);
int state_grid_field_origin_parity(
   int coordinate,
   MeshAxes const& axes,
   PhysicalSystem system,
   int passive_scalars,
   int component
);
std::string supported_aux_grid_field_names();
amrex::Vector<std::string> state_grid_field_names(PhysicalSystem system, int passive_scalars);
GridFieldSetInfo fluid_state_grid_fields(PhysicalSystem system, int passive_scalars);
GridFieldInfo const& ct_magnetic_grid_field();
std::vector<std::string> aux_grid_field_output_names(AuxGridField field);
std::string checkpoint_state_field_name();
std::string checkpoint_face_b_field_name(int direction);
int aux_grid_field_component_count(AuxGridField field);
int grid_field_component_count(GridFieldSetInfo const& set);
amrex::Vector<std::string> grid_field_component_names(GridFieldSetInfo const& set);

} // namespace pyfluid_amrex
