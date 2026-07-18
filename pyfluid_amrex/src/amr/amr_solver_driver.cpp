#include "pyfluid_amrex/amr_solver.hpp"

#include "amr/amr_solver_internal.hpp"

namespace pyfluid_amrex {

void AmrSolver::evolve()
{
   bool use_ct = false;
   EvolutionEosDispatch eos_dispatch = EvolutionEosDispatch::Configured;

   if (physics_system_ == PhysicalSystem::RMHD) {
#if PYFLUID_AMREX_ENABLE_MHD
      use_ct = ct_enabled(cfg_);
      eos_dispatch = EvolutionEosDispatch::IdealGas;
#else
      abort_disabled_build_feature("physics", "rmhd");
#endif
   } else if (physics_system_ == PhysicalSystem::RHD) {
      use_ct = false;
   } else if (is_mhd(cfg_)) {
#if PYFLUID_AMREX_ENABLE_MHD
      use_ct = ct_enabled(cfg_);
#else
      abort_disabled_build_feature("physics", "mhd");
#endif
   } else {
#if !PYFLUID_AMREX_ENABLE_HYDRO
      abort_disabled_build_feature("physics", "hydro");
#endif
   }

   if (use_ct) {
#if !PYFLUID_AMREX_ENABLE_CT
      abort_disabled_build_feature("CT", "on");
#endif
   }

   const EvolutionKernelBundleKey key{
      num_state_components(),
      use_ct,
      eos_dispatch,
   };
   const EvolutionKernelEntry entry = find_evolution_kernel_entry(key);
   if (entry == nullptr) {
      amrex::Abort(
         std::string("No compiled evolution kernel bundle matches num_components=")
         + std::to_string(key.num_components)
         + " ct=" + (key.ct_enabled ? "on" : "off")
         + " eos_dispatch="
         + (key.eos_dispatch == EvolutionEosDispatch::IdealGas
               ? "ideal_gas"
               : "configured")
      );
   }
   entry(*this);
}

void run_amr_solver(RunConfig const& cfg)
{
   AmrSolver solver(cfg);
   solver.evolve();
}

} // namespace pyfluid_amrex
