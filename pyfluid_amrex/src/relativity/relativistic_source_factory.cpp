#include "pyfluid_amrex/relativistic_source_factory.hpp"

#include "relativity/relativistic_sources.hpp"

namespace pyfluid_amrex {
namespace {

template <PhysicalSystem System, int EosMethod>
void launch_sources(RelativisticSourceLaunch const& launch)
{
   if (launch.cell_spacetime != nullptr) {
      add_stored_spacetime_sources<System, EosMethod>(
         launch.rhs,
         launch.primitive,
         *launch.cell_spacetime,
         launch.geometry,
         launch.eos,
         launch.excision_region,
         launch.mesh_coordinate,
         launch.mesh_axes
      );
      return;
   }
   add_fixed_spacetime_sources<System, EosMethod>(
      launch.rhs,
      launch.primitive,
      launch.geometry,
      launch.fixed_spacetime,
      launch.eos,
      launch.time,
      launch.mesh_coordinate,
      launch.mesh_axes
   );
}

template <PhysicalSystem System>
void launch_runtime_eos_sources(RelativisticSourceLaunch const& launch)
{
   launch_sources<System, EOSRuntime>(launch);
}

} // namespace

void add_relativistic_sources(RelativisticSourceLaunch const& launch)
{
   switch (launch.physical_system) {
   case PhysicalSystem::RHD:
      launch_runtime_eos_sources<PhysicalSystem::RHD>(launch);
      return;
   case PhysicalSystem::RMHD:
      launch_runtime_eos_sources<PhysicalSystem::RMHD>(launch);
      return;
   case PhysicalSystem::HD:
   case PhysicalSystem::MHD:
      return;
   }
   amrex::Abort("invalid physical system in relativistic-source factory");
}

} // namespace pyfluid_amrex
