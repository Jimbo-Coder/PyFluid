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
         launch.eos
      );
      return;
   }
   add_fixed_spacetime_sources<System, EosMethod>(
      launch.rhs,
      launch.primitive,
      launch.geometry,
      launch.fixed_spacetime,
      launch.eos,
      launch.time
   );
}

template <PhysicalSystem System>
void dispatch_eos(RelativisticSourceLaunch const& launch)
{
   if constexpr (System == PhysicalSystem::RMHD) {
      launch_sources<System, EOSIdealGas>(launch);
      return;
   }
   switch (launch.eos_method) {
   case EOSPolytrope:
      launch_sources<System, EOSPolytrope>(launch);
      return;
   case EOSHybridPolytrope:
      launch_sources<System, EOSHybridPolytrope>(launch);
      return;
   case EOSCompose:
      launch_sources<System, EOSCompose>(launch);
      return;
   case EOSIdealGas:
      launch_sources<System, EOSIdealGas>(launch);
      return;
   default:
      amrex::Abort("invalid EOS method in relativistic-source factory");
   }
}

} // namespace

void add_relativistic_sources(RelativisticSourceLaunch const& launch)
{
   switch (launch.physical_system) {
   case PhysicalSystem::RHD:
      dispatch_eos<PhysicalSystem::RHD>(launch);
      return;
   case PhysicalSystem::RMHD:
      dispatch_eos<PhysicalSystem::RMHD>(launch);
      return;
   case PhysicalSystem::HD:
   case PhysicalSystem::MHD:
      return;
   }
   amrex::Abort("invalid physical system in relativistic-source factory");
}

} // namespace pyfluid_amrex
