#pragma once

#include <utility>

#include "amr/amr_solver_base.hpp"
#include "pyfluid_amrex/checkpoint.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/diffusion.hpp"
#include "pyfluid_amrex/evolution_kernel_bundle.hpp"
#include "pyfluid_amrex/flux_factory.hpp"
#include "pyfluid_amrex/flux_divergence.hpp"
#include "pyfluid_amrex/flux_update.hpp"
#include "pyfluid_amrex/implicit_resistivity.hpp"
#include "numerics/patch_wall_flux.hpp"
#include "numerics/primitive_vector_transform.hpp"
#include "pyfluid_amrex/primitive_cache_factory.hpp"
#include "pyfluid_amrex/relativistic_source_factory.hpp"
#include "pyfluid_amrex/signal_speed_factory.hpp"
#include "pyfluid_amrex/solid_wall_power.hpp"
#include "numerics/sources.hpp"
#include "pyfluid_amrex/state_update.hpp"

namespace pyfluid_amrex {

class AmrSolver final : public AmrSolverBase {
public:
   explicit AmrSolver(RunConfig cfg) : AmrSolverBase(std::move(cfg)) {}

   void evolve();

   template <int NComp, bool CTEnabled>
   void evolve_configured_eos_bundle()
   {
      dispatch_evolve_eos<NComp, CTEnabled>();
   }

   template <int NComp, bool CTEnabled>
   void evolve_ideal_gas_bundle()
   {
      dispatch_evolve_timestep<NComp, CTEnabled, EOSIdealGas>();
   }

private:
   void repair_after_hierarchy_change(HydroConservedTotals const& target) override
   {
      repair_nonpositive_hydro_fluid_cells_all_levels();
      apply_hydro_valid_conserved_total_correction(target);
   }

   static DiffusionStats combine_diffusion_stats(DiffusionStats a, DiffusionStats b)
   {
      a.residual = amrex::max(a.residual, b.residual);
      a.iterations += b.iterations;
      return a;
   }

   // Physical-system evolution templates. Kernel-bundle translation units own
   // their unique instantiations; AMR infrastructure sources include the base.
#include "amr/solver_sections/evolve_core.inc.hpp"
#include "amr/solver_sections/ct_evolution.inc.hpp"
#include "amr/solver_sections/gravity_sources.inc.hpp"
#include "amr/solver_sections/solid_conservation.inc.hpp"
#include "amr/solver_sections/state_repair.inc.hpp"
#include "amr/solver_sections/timestep_control.inc.hpp"
#include "amr/solver_sections/rhs_assembly.inc.hpp"
#include "amr/solver_sections/rk_stage_algebra.inc.hpp"
#include "amr/solver_sections/time_integrators.inc.hpp"
#include "amr/solver_sections/barotropic_energy.inc.hpp"
#include "amr/solver_sections/subcycling.inc.hpp"
#include "amr/solver_sections/diffusion.inc.hpp"
};

} // namespace pyfluid_amrex
