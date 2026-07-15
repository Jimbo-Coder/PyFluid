#pragma once

#include <AMReX_Vector.H>

#include <array>
#include <string>
#include <vector>

#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace pyfluid_amrex {

static constexpr int MaxGridFieldComponents = 3;

enum class GridFieldCentering {
   Cell,
   Face1,
   Face2,
   Face3,
   Edge1,
   Edge2,
   Edge3,
};

enum class AuxGridField {
   CurrentDensity,
   ElectricField,
   DivB,
   MagneticEnergy,
   Vorticity,
   MachNumber,
   PlasmaBeta,
   EntropyProxy,
   AmrTagMetric,
   SolidMask,
   SolidVolumeFraction,
   SolidAperture1,
   SolidAperture2,
   SolidAperture3,
   SolidSweptVolume,
   SolidPhi,
   GravityPotential,
   GravityGradient,
   Unknown,
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
};

struct AuxGridFieldInfo {
   AuxGridField field = AuxGridField::Unknown;
   const char* name = "";
   std::array<const char*, MaxGridFieldComponents> component_names{};
   std::array<GridFieldCentering, MaxGridFieldComponents> centerings{
      GridFieldCentering::Cell,
      GridFieldCentering::Cell,
      GridFieldCentering::Cell,
   };
   FieldRole role = FieldRole::OutputOnlyAuxiliary;
   DiagnosticFieldKind diagnostic_kind = DiagnosticFieldKind::PureAuxiliary;
   EvolutionFieldKind evolution_kind = EvolutionFieldKind::None;
   int components = 1;
   bool requires_mhd = false;
   bool requires_solid_geometry = false;
   bool requires_gravity = false;
   int min_dimension = 1;
};

struct StateGridFieldInfo {
   const char* name = "";
   int component = -1;
   GridFieldCentering centering = GridFieldCentering::Cell;
   GridFieldSymmetryKind symmetry = GridFieldSymmetryKind::Scalar;
   bool requires_mhd = false;
};

AuxGridField aux_grid_field_from_name(std::string const& name);
AuxGridFieldInfo aux_grid_field_info(AuxGridField field);
StateGridFieldInfo state_grid_field_info(int component);
int state_grid_field_origin_parity(int coordinate, MeshAxes const& axes, int component);
std::string supported_aux_grid_field_names();
amrex::Vector<std::string> state_grid_field_names(PhysicalSystem system, int passive_scalars);
std::vector<std::string> aux_grid_field_output_names(AuxGridField field);
std::string checkpoint_state_field_name();
std::string checkpoint_face_b_field_name(int direction);
int aux_grid_field_component_count(AuxGridField field);

} // namespace pyfluid_amrex
