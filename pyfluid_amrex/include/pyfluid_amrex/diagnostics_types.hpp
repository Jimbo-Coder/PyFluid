#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <iosfwd>
#include <string>
#include <vector>

namespace pyfluid_amrex {

struct RunConfig;

struct ScalarDiagnostics {
   int step = 0;
   amrex::Real time = 0.0;
   amrex::Real dt = 0.0;
   amrex::Real mass = 0.0;
   amrex::Real mom1 = 0.0;
   amrex::Real mom2 = 0.0;
   amrex::Real mom3 = 0.0;
   amrex::Real angular_mom1 = 0.0;
   amrex::Real angular_mom2 = 0.0;
   amrex::Real angular_mom3 = 0.0;
   amrex::Real energy = 0.0;
   amrex::Real gravity_potential_energy = 0.0;
   amrex::Real gravity_total_energy = 0.0;
   amrex::Real gravity_total_energy_rel_drift = 0.0;
   amrex::Real passive_mass = 0.0;
   amrex::Real passive_mass_rel_drift = 0.0;
   amrex::Real wall_work = 0.0;
   amrex::Real wall_power = 0.0;
   amrex::Real energy_minus_wall_work = 0.0;
   amrex::Real wall_work_rel_error = 0.0;
   int external_budget_valid = 1;
   amrex::Real boundary_mass_change = 0.0;
   amrex::Real boundary_energy_change = 0.0;
   amrex::Real source_mass_change = 0.0;
   amrex::Real source_energy_change = 0.0;
   amrex::Real external_mass_change = 0.0;
   amrex::Real external_energy_change = 0.0;
   amrex::Real external_mass_residual = 0.0;
   amrex::Real external_energy_residual = 0.0;
   amrex::Real external_mass_rel_error = 0.0;
   amrex::Real external_energy_rel_error = 0.0;
   amrex::Real mass_rel_drift = 0.0;
   amrex::Real energy_rel_drift = 0.0;
   amrex::Real min_rho = 0.0;
   amrex::Real max_rho = 0.0;
   amrex::Real max_rho_rel_change = 0.0;
   amrex::Real min_pressure = 0.0;
   amrex::Real magnetic_energy = 0.0;
   amrex::Real divb_l1 = 0.0;
   amrex::Real diffusion_residual = 0.0;
   int diffusion_iterations = 0;
   amrex::Long amr_total_cells = 0;
   int amr_active_finest_level = 0;
   int amr_finest_headroom = 0;
   int amr_touches_max_level = 0;
   amrex::Real amr_total_budget_fraction = -1.0;
   amrex::Real amr_peak_level_budget_fraction = -1.0;
   amrex::Real mcells_per_s = 0.0;
   amrex::Real layout_mcells_per_s = 0.0;
   amrex::Real sim_time_per_s = 0.0;
   amrex::Real cfl_limiter_frequency = 0.0;
   int cfl_limiter_level = -1;
   int cfl_limiter_i = 0;
   int cfl_limiter_j = 0;
   int cfl_limiter_k = 0;
   amrex::Real cfl_limiter_q1 = 0.0;
   amrex::Real cfl_limiter_q2 = 0.0;
   amrex::Real cfl_limiter_q3 = 0.0;
   amrex::Real z4c_hamiltonian_linf = 0.0;
   amrex::Real z4c_hamiltonian_l2 = 0.0;
   amrex::Real z4c_momentum_linf = 0.0;
   amrex::Real z4c_z_linf = 0.0;
   amrex::Real z4c_theta_linf = 0.0;
   amrex::Real z4c_log_conformal_determinant_linf = 0.0;
   amrex::Real z4c_trace_a_tilde_linf = 0.0;
};

struct SpacetimeConstraintDiagnostics {
   amrex::Real hamiltonian_linf = 0.0;
   amrex::Real hamiltonian_l2 = 0.0;
   amrex::Real momentum_linf = 0.0;
   amrex::Real z_linf = 0.0;
   amrex::Real theta_linf = 0.0;
   amrex::Real log_conformal_determinant_linf = 0.0;
   amrex::Real trace_a_tilde_linf = 0.0;
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

AmrLevelDiagnostics summarize_amr_levels(std::vector<amrex::Long> const& level_cells, int max_level,
                                         amrex::Long total_cell_budget, amrex::Long level_cell_budget);

void write_scalar_header(std::ofstream& out, RunConfig const& config, int step, amrex::Real time);
void append_scalar_row(std::ofstream& out, ScalarDiagnostics const& diagnostics);

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
