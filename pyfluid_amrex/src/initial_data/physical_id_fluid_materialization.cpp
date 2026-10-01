#include "pyfluid_amrex/initial_data/physical_id.hpp"

#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace {

template <int EOSMethod>
void materialize_physical_initial_data_fluid_method(
   RunConfig const& cfg,
   PhysicalSystem system,
   amrex::Vector<amrex::Geometry> const& geometry,
   ConstPhysicalIdField const& primitive,
   ConstPhysicalIdField const& cell_spacetime,
   MutablePhysicalIdField const& evolution_fluid
)
{
   const EOSParams eos = eos_params(cfg);
   const bool relativistic = physical_system_is_relativistic(system);
   const bool magnetized = physical_system_has_magnetic_field(system);
   for (std::size_t level = 0; level < evolution_fluid.size(); ++level) {
      const bool stored_spacetime = !cell_spacetime.empty();
      const auto lower = geometry[level].ProbLoArray();
      const auto spacing = geometry[level].CellSizeArray();
      const FixedSpacetimeParameters fixed = fixed_spacetime_parameters(cfg);
      for (amrex::MFIter mfi(
              *evolution_fluid[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto physical = primitive[level]->const_array(mfi);
         const auto output = evolution_fluid[level]->array(mfi);
         const auto stored_metric = stored_spacetime
            ? cell_spacetime[level]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const int num_components = evolution_fluid[level]->nComp();
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (!relativistic) {
               if (magnetized) {
                  const MHDPrimitive value{
                     physical(i, j, k, RHO),
                     physical(i, j, k, MOM1),
                     physical(i, j, k, MOM2),
                     physical(i, j, k, MOM3),
                     physical(i, j, k, ENERGY),
                     physical(i, j, k, B1),
                     physical(i, j, k, B2),
                     physical(i, j, k, B3),
                  };
                  const MHDState conserved =
                     conserved_from_primitive<EOSMethod>(value, eos);
                  output(i, j, k, RHO) = conserved.rho;
                  output(i, j, k, MOM1) = conserved.mom1;
                  output(i, j, k, MOM2) = conserved.mom2;
                  output(i, j, k, MOM3) = conserved.mom3;
                  output(i, j, k, ENERGY) = conserved.energy;
                  output(i, j, k, B1) = conserved.b1;
                  output(i, j, k, B2) = conserved.b2;
                  output(i, j, k, B3) = conserved.b3;
               } else {
                  const HydroPrimitive value{
                     physical(i, j, k, RHO),
                     physical(i, j, k, MOM1),
                     physical(i, j, k, MOM2),
                     physical(i, j, k, MOM3),
                     physical(i, j, k, ENERGY),
                  };
                  const HydroState conserved =
                     conserved_from_primitive<EOSMethod>(value, eos);
                  output(i, j, k, RHO) = conserved.rho;
                  output(i, j, k, MOM1) = conserved.mom1;
                  output(i, j, k, MOM2) = conserved.mom2;
                  output(i, j, k, MOM3) = conserved.mom3;
                  output(i, j, k, ENERGY) = conserved.energy;
               }
            } else {
               const SpacetimeMetric metric = stored_spacetime
                  ? generated::z4c_spacetime_metric(
                       load_z4c_state(stored_metric, i, j, k)
                    )
                  : fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
                       fixed, lower, spacing, i, j, k, 0.0_rt
                    ).metric;
               const amrex::Real density = physical(i, j, k, RHO);
               const amrex::Real pressure = physical(i, j, k, ENERGY);
               const amrex::Real specific_internal_energy =
                  relativistic_specific_internal_energy_from_pressure<
                     EOSMethod
                  >(density, pressure, eos);
               if (magnetized) {
                  const RMHDPrimitive value{
                     density,
                     specific_internal_energy,
                     pressure,
                     physical(i, j, k, MOM1),
                     physical(i, j, k, MOM2),
                     physical(i, j, k, MOM3),
                     physical(i, j, k, B1),
                     physical(i, j, k, B2),
                     physical(i, j, k, B3),
                  };
                  store_rmhd_conserved(
                     output,
                     i,
                     j,
                     k,
                     generated::rmhd_primitive_to_conserved(value, metric)
                  );
               } else {
                  const RHDPrimitive value{
                     density,
                     specific_internal_energy,
                     pressure,
                     physical(i, j, k, MOM1),
                     physical(i, j, k, MOM2),
                     physical(i, j, k, MOM3),
                  };
                  store_rhd_conserved(
                     output,
                     i,
                     j,
                     k,
                     generated::rhd_primitive_to_conserved(value, metric)
                  );
               }
            }
            const int base_components = magnetized ? NVAR_MHD : NVAR_HYDRO;
            for (int component = base_components;
                 component < num_components; ++component) {
               output(i, j, k, component) = output(i, j, k, RHO)
                  * physical(i, j, k, component);
            }
         });
      }
   }
}

} // namespace

void materialize_physical_initial_data_fluid(
   RunConfig const& cfg,
   PhysicalSystem system,
   amrex::Vector<amrex::Geometry> const& geometry,
   ConstPhysicalIdField const& primitive,
   ConstPhysicalIdField const& cell_spacetime,
   MutablePhysicalIdField const& evolution_fluid
)
{
   if (geometry.size() != primitive.size()
       || geometry.size() != evolution_fluid.size()
       || (!cell_spacetime.empty()
           && geometry.size() != cell_spacetime.size())) {
      amrex::Abort("physical-ID fluid materialization hierarchy is inconsistent");
   }
   materialize_physical_initial_data_fluid_method<EOSRuntime>(
      cfg, system, geometry, primitive, cell_spacetime, evolution_fluid
   );
}

} // namespace pyfluid_amrex
