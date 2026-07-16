#include "pyfluid_amrex/spacetime_fields.hpp"

#include <utility>

namespace pyfluid_amrex {

namespace {

using Centering = GridFieldCentering;
using Symmetry = GridFieldSymmetryKind;
using Value = GridFieldValueKind;

GridFieldComponentInfo component(std::string_view name, Symmetry symmetry)
{
   return {std::string(name), Centering::Vertex, symmetry};
}

GridFieldInfo field(
   std::string_view name,
   GridFieldValueKind value_kind,
   std::vector<GridFieldComponentInfo> components
)
{
   return {
      std::string(name),
      value_kind,
      std::move(components),
      FieldRole::EvolvedState,
      EvolutionFieldKind::MainState,
      3,
      true,
      true,
   };
}

const GridFieldSetInfo z4c_fields = {
   "spacetime_state",
   EvolutionSector::Spacetime,
   {
      field("chi", Value::Scalar, {component("chi", Symmetry::Scalar)}),
      field(
         "gamma_tilde",
         Value::SymmetricTensor3,
         {
            component("gamma_tilde11", Symmetry::Tensor11),
            component("gamma_tilde12", Symmetry::Tensor12),
            component("gamma_tilde13", Symmetry::Tensor13),
            component("gamma_tilde22", Symmetry::Tensor22),
            component("gamma_tilde23", Symmetry::Tensor23),
            component("gamma_tilde33", Symmetry::Tensor33),
         }
      ),
      field("k_hat", Value::Scalar, {component("k_hat", Symmetry::Scalar)}),
      field(
         "a_tilde",
         Value::SymmetricTensor3,
         {
            component("a_tilde11", Symmetry::Tensor11),
            component("a_tilde12", Symmetry::Tensor12),
            component("a_tilde13", Symmetry::Tensor13),
            component("a_tilde22", Symmetry::Tensor22),
            component("a_tilde23", Symmetry::Tensor23),
            component("a_tilde33", Symmetry::Tensor33),
         }
      ),
      field(
         "gamma_tilde_vector",
         Value::Vector3,
         {
            component("gamma_tilde1", Symmetry::Vector1),
            component("gamma_tilde2", Symmetry::Vector2),
            component("gamma_tilde3", Symmetry::Vector3),
         }
      ),
      field("theta", Value::Scalar, {component("theta", Symmetry::Scalar)}),
      field("lapse", Value::Scalar, {component("lapse", Symmetry::Scalar)}),
      field(
         "shift",
         Value::Vector3,
         {
            component("shift1", Symmetry::Vector1),
            component("shift2", Symmetry::Vector2),
            component("shift3", Symmetry::Vector3),
         }
      ),
   },
};

static_assert(Z4C_NUM_COMPONENTS == 22);

} // namespace

GridFieldSetInfo const& z4c_state_grid_fields()
{
   return z4c_fields;
}

amrex::Vector<std::string> z4c_state_grid_field_names()
{
   return grid_field_component_names(z4c_fields);
}

} // namespace pyfluid_amrex
