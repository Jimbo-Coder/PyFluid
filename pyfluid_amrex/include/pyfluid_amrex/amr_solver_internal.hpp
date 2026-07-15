#pragma once

#include <utility>

#include "pyfluid_amrex/amr_solver_base.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/diffusion.hpp"
#include "pyfluid_amrex/implicit_resistivity.hpp"
#include "pyfluid_amrex/rhs.hpp"
#include "pyfluid_amrex/relativistic_sources.hpp"
#include "pyfluid_amrex/signal_speed.hpp"
#include "pyfluid_amrex/sources.hpp"

namespace pyfluid_amrex {

class AmrSolver final : public AmrSolverBase {
public:
   explicit AmrSolver(RunConfig cfg) : AmrSolverBase(std::move(cfg)) {}

   void evolve()
   {
      dispatch_evolve();
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

   // Physical-system evolution templates. Only the driver translation unit
   // includes these definitions; AMR infrastructure sources include the base.
#include "pyfluid_amrex/solver_sections/evolve_core.inc.hpp"
#include "pyfluid_amrex/solver_sections/ct_evolution.inc.hpp"
#include "pyfluid_amrex/solver_sections/gravity_sources.inc.hpp"
#include "pyfluid_amrex/solver_sections/solid_conservation.inc.hpp"
#include "pyfluid_amrex/solver_sections/state_repair.inc.hpp"
#include "pyfluid_amrex/solver_sections/timestep_control.inc.hpp"
#include "pyfluid_amrex/solver_sections/rhs_assembly.inc.hpp"
#include "pyfluid_amrex/solver_sections/rk_stage_algebra.inc.hpp"
#include "pyfluid_amrex/solver_sections/time_integrators.inc.hpp"
#include "pyfluid_amrex/solver_sections/barotropic_energy.inc.hpp"
#include "pyfluid_amrex/solver_sections/subcycling.inc.hpp"
#include "pyfluid_amrex/solver_sections/diffusion.inc.hpp"
};

} // namespace pyfluid_amrex
