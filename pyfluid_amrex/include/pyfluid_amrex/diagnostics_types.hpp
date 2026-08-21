#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <iosfwd>
#include <string>
#include <vector>

#include "pyfluid_amrex/generated/diagnostic_catalog.hpp"

namespace pyfluid_amrex {

struct RunConfig;

struct ConservedDiagnosticIntegrands {
   amrex::Real mass = 0.0;
   amrex::Real momentum1 = 0.0;
   amrex::Real momentum2 = 0.0;
   amrex::Real momentum3 = 0.0;
   amrex::Real energy = 0.0;
   amrex::Real angular_momentum1 = 0.0;
   amrex::Real angular_momentum2 = 0.0;
   amrex::Real angular_momentum3 = 0.0;
   amrex::Real magnetic_energy = 0.0;
   amrex::Real gravity_potential_energy = 0.0;
   amrex::Real passive_mass = 0.0;
   amrex::Real divb = 0.0;
};

struct AdmFluxIntegrands {
   amrex::Real mass1 = 0.0;
   amrex::Real mass2 = 0.0;
   amrex::Real mass3 = 0.0;
   amrex::Real komar_mass1 = 0.0;
   amrex::Real komar_mass2 = 0.0;
   amrex::Real komar_mass3 = 0.0;
   amrex::Real linear_momentum11 = 0.0;
   amrex::Real linear_momentum12 = 0.0;
   amrex::Real linear_momentum13 = 0.0;
   amrex::Real linear_momentum21 = 0.0;
   amrex::Real linear_momentum22 = 0.0;
   amrex::Real linear_momentum23 = 0.0;
   amrex::Real linear_momentum31 = 0.0;
   amrex::Real linear_momentum32 = 0.0;
   amrex::Real linear_momentum33 = 0.0;
   amrex::Real angular_momentum11 = 0.0;
   amrex::Real angular_momentum12 = 0.0;
   amrex::Real angular_momentum13 = 0.0;
   amrex::Real angular_momentum21 = 0.0;
   amrex::Real angular_momentum22 = 0.0;
   amrex::Real angular_momentum23 = 0.0;
   amrex::Real angular_momentum31 = 0.0;
   amrex::Real angular_momentum32 = 0.0;
   amrex::Real angular_momentum33 = 0.0;
};

struct AdmVolumeIntegrands {
   amrex::Real mass = 0.0;
   amrex::Real proper_mass = 0.0;
   amrex::Real komar_mass = 0.0;
   amrex::Real linear_momentum1 = 0.0;
   amrex::Real linear_momentum2 = 0.0;
   amrex::Real linear_momentum3 = 0.0;
   amrex::Real angular_momentum1 = 0.0;
   amrex::Real angular_momentum2 = 0.0;
   amrex::Real angular_momentum3 = 0.0;
};

struct WeylRadiationIntegrands {
   amrex::Real electric11 = 0.0;
   amrex::Real electric12 = 0.0;
   amrex::Real electric13 = 0.0;
   amrex::Real electric22 = 0.0;
   amrex::Real electric23 = 0.0;
   amrex::Real electric33 = 0.0;
   amrex::Real magnetic11 = 0.0;
   amrex::Real magnetic12 = 0.0;
   amrex::Real magnetic13 = 0.0;
   amrex::Real magnetic22 = 0.0;
   amrex::Real magnetic23 = 0.0;
   amrex::Real magnetic33 = 0.0;
};

struct Psi4Integrands {
   amrex::Real psi4_real = 0.0;
   amrex::Real psi4_imaginary = 0.0;
};

struct MatterEnergyFluxIntegrands {
   amrex::Real flux1 = 0.0;
   amrex::Real flux2 = 0.0;
   amrex::Real flux3 = 0.0;
};

struct SpacetimeConstraintDiagnostics {
   amrex::Real hamiltonian_linf = 0.0;
   amrex::Real hamiltonian_l2 = 0.0;
   amrex::Real hamiltonian_interior_linf = 0.0;
   amrex::Real hamiltonian_interior_l2 = 0.0;
   amrex::Real hamiltonian_cf_linf = 0.0;
   amrex::Real hamiltonian_cf_l2 = 0.0;
   amrex::Real hamiltonian_bh_exterior_linf = 0.0;
   amrex::Real hamiltonian_bh_exterior_l2 = 0.0;
   amrex::Real momentum_bh_exterior_linf = 0.0;
   amrex::Real momentum_linf = 0.0;
   amrex::Real z_linf = 0.0;
   amrex::Real theta_linf = 0.0;
   amrex::Real log_conformal_determinant_linf = 0.0;
   amrex::Real trace_a_tilde_linf = 0.0;
};

struct SpacetimeGlobalDiagnostics {
   amrex::Real adm_mass_surface = 0.0;
   amrex::Real adm_mass_volume = 0.0;
   amrex::Real proper_mass_volume = 0.0;
   amrex::Real komar_mass_surface = 0.0;
   amrex::Real komar_mass_volume = 0.0;
   amrex::GpuArray<amrex::Real, 3> adm_momentum_surface{};
   amrex::GpuArray<amrex::Real, 3> adm_momentum_volume{};
   amrex::GpuArray<amrex::Real, 3> adm_angular_momentum_surface{};
   amrex::GpuArray<amrex::Real, 3> adm_angular_momentum_volume{};
};

struct AmrLevelDiagnostics {
   amrex::Long total_cells = 0;
   int active_finest_level = 0;
   int max_level = 0;
   int finest_headroom = 0;
   int touches_max_level = 0;
   amrex::Real total_budget_fraction = -1.0;
   amrex::Real peak_level_budget_fraction = -1.0;
};

struct CflLimiterLocation {
   amrex::Real frequency = 0.0;
   int level = -1;
   int i = 0;
   int j = 0;
   int k = 0;
   amrex::Real q1 = 0.0;
   amrex::Real q2 = 0.0;
   amrex::Real q3 = 0.0;
};

struct ScalarDiagnosticSelection {
   std::array<bool, static_cast<std::size_t>(ScalarDiagnosticId::Count)> enabled{};

   bool contains(ScalarDiagnosticId id) const noexcept
   {
      return enabled[static_cast<std::size_t>(id)];
   }

   bool contains_any(ScalarDiagnosticGroup group) const noexcept
   {
      auto const& members = ScalarDiagnosticGroups[static_cast<std::size_t>(group)].members;
      for (std::size_t index = 0; index < enabled.size(); ++index) {
         if (enabled[index] && members[index]) {
            return true;
         }
      }
      return false;
   }

   void enable_group(ScalarDiagnosticGroup group) noexcept
   {
      auto const& members = ScalarDiagnosticGroups[static_cast<std::size_t>(group)].members;
      for (std::size_t index = 0; index < enabled.size(); ++index) {
         enabled[index] = enabled[index] || members[index];
      }
   }

   std::size_t size() const noexcept;
};

ScalarDiagnosticSelection scalar_diagnostic_selection(std::string const& selectors);
void apply_scalar_diagnostic_requirements(
   ScalarDiagnosticSelection& selection,
   RunConfig const& config
);

AmrLevelDiagnostics summarize_amr_levels(std::vector<amrex::Long> const& level_cells, int max_level,
                                         amrex::Long total_cell_budget, amrex::Long level_cell_budget);

void write_scalar_header(std::ofstream& out, RunConfig const& config, int step, amrex::Real time,
                         ScalarDiagnosticSelection const& selection);
void append_scalar_row(std::ofstream& out, ScalarDiagnostics const& diagnostics,
                       ScalarDiagnosticSelection const& selection);

void write_amr_level_header(std::ofstream& out, int max_level, RunConfig const& config, int step, amrex::Real time);
void append_amr_level_row(std::ofstream& out, int step, amrex::Real time, std::vector<amrex::Long> const& level_cells,
                          amrex::Vector<int> const& level_steps, int max_level, amrex::Long total_cell_budget,
                          amrex::Long level_cell_budget, amrex::Real finest_dx,
                          std::array<amrex::Real, AMREX_SPACEDIM> const& finest_dx_by_dir,
                          amrex::Real min_feature_length,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& min_feature_lengths,
                          amrex::Real min_refine_radius, amrex::Real solid_curvature_error,
                          amrex::Real solid_curvature_tolerance, std::vector<int> const& level_box_counts,
                          std::vector<amrex::Real> const& level_max_box_aspects,
                          std::array<std::vector<int>, AMREX_SPACEDIM> const& level_ref_ratios,
                          std::vector<amrex::Long> const& auto_tagged_cells,
                          std::array<std::vector<amrex::Long>, AMREX_SPACEDIM> const& auto_direction_tagged_cells,
                          std::array<std::vector<amrex::Real>, AMREX_SPACEDIM> const& auto_direction_strength,
                          std::vector<std::string> const& refinement_decisions);

} // namespace pyfluid_amrex
