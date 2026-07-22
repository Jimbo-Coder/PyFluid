#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "pyfluid_amrex/initial_data/ctt.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos/presets.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/initial_data/rotating_star.hpp"
#include "pyfluid_amrex/initial_data/tov.hpp"
#include "pyfluid_amrex/kernels_generated/initial_data.hpp"

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real manufactured_solution(amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
#if (AMREX_SPACEDIM == 1)
   return 1.0 + pyfluid_amrex::generated::initial_data_poisson_solution_1d(x, y, z);
#elif (AMREX_SPACEDIM == 2)
   return 1.0 + pyfluid_amrex::generated::initial_data_poisson_solution_2d(x, y, z);
#else
   return 1.0 + pyfluid_amrex::generated::initial_data_poisson_solution_3d(x, y, z);
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real manufactured_rhs(amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
#if (AMREX_SPACEDIM == 1)
   return pyfluid_amrex::generated::initial_data_poisson_rhs_1d(x, y, z);
#elif (AMREX_SPACEDIM == 2)
   return pyfluid_amrex::generated::initial_data_poisson_rhs_2d(x, y, z);
#else
   return pyfluid_amrex::generated::initial_data_poisson_rhs_3d(x, y, z);
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real manufactured_robin_c(
   amrex::Real x,
   amrex::Real y,
   amrex::Real z,
   int direction,
   amrex::Real coefficient_b
) noexcept
{
   const amrex::Real sine_x = std::sin(pyfluid_amrex::constants::pi * x);
   const amrex::Real cosine_x = std::cos(pyfluid_amrex::constants::pi * x);
#if (AMREX_SPACEDIM >= 2)
   const amrex::Real sine_y = std::sin(pyfluid_amrex::constants::pi * y);
   const amrex::Real cosine_y = std::cos(pyfluid_amrex::constants::pi * y);
#else
   const amrex::Real sine_y = 1.0;
   const amrex::Real cosine_y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
   const amrex::Real sine_z = std::sin(pyfluid_amrex::constants::pi * z);
   const amrex::Real cosine_z = std::cos(pyfluid_amrex::constants::pi * z);
#else
   const amrex::Real sine_z = 1.0;
   const amrex::Real cosine_z = 0.0;
#endif
   const amrex::Real solution = 1.0 + sine_x * sine_y * sine_z;
   amrex::Real coordinate_derivative =
      pyfluid_amrex::constants::pi * cosine_x * sine_y * sine_z;
#if (AMREX_SPACEDIM >= 2)
   if (direction == 1) {
      coordinate_derivative =
         pyfluid_amrex::constants::pi * sine_x * cosine_y * sine_z;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (direction == 2) {
      coordinate_derivative =
         pyfluid_amrex::constants::pi * sine_x * sine_y * cosine_z;
   }
#endif
   return solution + coefficient_b * coordinate_derivative;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real manufactured_ctt_conformal_factor(
   amrex::Real x,
   amrex::Real y,
   amrex::Real z,
   amrex::Real amplitude
) noexcept
{
#if (AMREX_SPACEDIM == 1)
   return pyfluid_amrex::generated::ctt_manufactured_conformal_factor_1d(
      x, y, z, amplitude
   );
#elif (AMREX_SPACEDIM == 2)
   return pyfluid_amrex::generated::ctt_manufactured_conformal_factor_2d(
      x, y, z, amplitude
   );
#else
   return pyfluid_amrex::generated::ctt_manufactured_conformal_factor_3d(
      x, y, z, amplitude
   );
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real manufactured_ctt_energy_density(
   amrex::Real x,
   amrex::Real y,
   amrex::Real z,
   amrex::Real amplitude
) noexcept
{
#if (AMREX_SPACEDIM == 1)
   return pyfluid_amrex::generated::ctt_manufactured_energy_density_1d(
      x, y, z, amplitude
   );
#elif (AMREX_SPACEDIM == 2)
   return pyfluid_amrex::generated::ctt_manufactured_energy_density_2d(
      x, y, z, amplitude
   );
#else
   return pyfluid_amrex::generated::ctt_manufactured_energy_density_3d(
      x, y, z, amplitude
   );
#endif
}

amrex::Real validate_centered_power_law_robin(
   pyfluid_amrex::InitialDataEllipticHierarchy const& hierarchy,
   pyfluid_amrex::InitialDataEllipticCentering centering
)
{
   constexpr amrex::Real asymptotic_value = 1.25;
   constexpr amrex::Real amplitude = 0.2;
   constexpr amrex::Real falloff_power = 1.0;
   auto boundary = pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
      hierarchy, 1, 0.0, 0.0, 0.0, centering
   );
   pyfluid_amrex::set_centered_power_law_robin_boundary(
      hierarchy, boundary, {asymptotic_value}, falloff_power
   );

   amrex::Real max_identity_error = 0.0;
   for (int level = 0; level < boundary.num_levels(); ++level) {
      amrex::MultiFab error(
         boundary.coefficient_a(level).boxArray(),
         hierarchy.distribution[level],
         1,
         1
      );
      error.setVal(0.0);
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto probhi = hierarchy.geometry[level].ProbHiArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center;
      amrex::GpuArray<int, AMREX_SPACEDIM> periodic;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         center[direction] = 0.5 * (problo[direction] + probhi[direction]);
         periodic[direction] = hierarchy.geometry[level].isPeriodic(direction);
      }
      const bool vertex_centered =
         centering == pyfluid_amrex::InitialDataEllipticCentering::vertex;
      const amrex::Box coefficient_domain = vertex_centered
         ? amrex::surroundingNodes(hierarchy.geometry[level].Domain())
         : hierarchy.geometry[level].Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(coefficient_domain);
      const amrex::Dim3 domain_hi = amrex::ubound(coefficient_domain);
      const amrex::Real coordinate_offset = vertex_centered ? 0.0 : 0.5;
      for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.fabbox();
         const auto coefficient_a = boundary.coefficient_a(level).const_array(mfi);
         const auto coefficient_b = boundary.coefficient_b(level).const_array(mfi);
         const auto coefficient_c = boundary.coefficient_c(level).const_array(mfi);
         const auto identity_error = error.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int face_direction = -1;
            amrex::Real position_1 = problo[0]
               + (i + coordinate_offset) * dx[0] - center[0];
#if (AMREX_SPACEDIM >= 2)
            amrex::Real position_2 = problo[1]
               + (j + coordinate_offset) * dx[1] - center[1];
#else
            amrex::Real position_2 = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
            amrex::Real position_3 = problo[2]
               + (k + coordinate_offset) * dx[2] - center[2];
#else
            amrex::Real position_3 = 0.0;
#endif
            if (!periodic[0] && i < domain_lo.x) {
               face_direction = 0;
               position_1 = problo[0] - center[0];
            } else if (!periodic[0] && i > domain_hi.x) {
               face_direction = 0;
               position_1 = probhi[0] - center[0];
            }
#if (AMREX_SPACEDIM >= 2)
            else if (!periodic[1] && j < domain_lo.y) {
               face_direction = 1;
               position_2 = problo[1] - center[1];
            } else if (!periodic[1] && j > domain_hi.y) {
               face_direction = 1;
               position_2 = probhi[1] - center[1];
            }
#endif
#if (AMREX_SPACEDIM == 3)
            else if (!periodic[2] && k < domain_lo.z) {
               face_direction = 2;
               position_3 = problo[2] - center[2];
            } else if (!periodic[2] && k > domain_hi.z) {
               face_direction = 2;
               position_3 = probhi[2] - center[2];
            }
#endif
            if (face_direction >= 0) {
               const amrex::Real radius_squared =
                  position_1 * position_1
                  + position_2 * position_2
                  + position_3 * position_3;
               const amrex::Real radius = std::sqrt(radius_squared);
               amrex::Real face_position = position_1;
#if (AMREX_SPACEDIM >= 2)
               if (face_direction == 1) {
                  face_position = position_2;
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (face_direction == 2) {
                  face_position = position_3;
               }
#endif
               const amrex::Real field = asymptotic_value
                  + amplitude / std::pow(radius, falloff_power);
               const amrex::Real coordinate_derivative =
                  -falloff_power * amplitude * face_position
                  / std::pow(radius, falloff_power + 2.0);
               identity_error(i, j, k, 0) = amrex::Math::abs(
                  coefficient_a(i, j, k, 0) * field
                  + coefficient_b(i, j, k, 0) * coordinate_derivative
                  - coefficient_c(i, j, k, 0)
               );
            }
         });
      }
      max_identity_error = std::max(
         max_identity_error, error.norm0(0, 1, false)
      );
   }
   return max_identity_error;
}

bool validate_nodal_poisson(
   pyfluid_amrex::InitialDataEllipticHierarchy const& hierarchy,
   bool use_robin
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> solution_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhs_storage(num_levels);
   amrex::Vector<amrex::MultiFab*> solution(num_levels);
   amrex::Vector<amrex::MultiFab const*> rhs(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      const amrex::BoxArray nodal_grids = amrex::convert(
         hierarchy.grids[level], amrex::IntVect::TheNodeVector()
      );
      solution_storage[level] = std::make_unique<amrex::MultiFab>(
         nodal_grids, hierarchy.distribution[level], 1, 0
      );
      rhs_storage[level] = std::make_unique<amrex::MultiFab>(
         nodal_grids, hierarchy.distribution[level], 1, 0
      );
      solution_storage[level]->setVal(0.0);
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      for (amrex::MFIter mfi(*rhs_storage[level]); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto values = rhs_storage[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real x = problo[0] + i * dx[0];
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = problo[1] + j * dx[1];
#else
            const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = problo[2] + k * dx[2];
#else
            const amrex::Real z = 0.0;
#endif
            values(i, j, k, 0) = manufactured_rhs(x, y, z);
         });
      }
      solution[level] = solution_storage[level].get();
      rhs[level] = rhs_storage[level].get();
   }

   auto boundary = pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      1,
      1.0,
      0.0,
      use_robin ? 0.0 : 1.0,
      pyfluid_amrex::InitialDataEllipticCentering::vertex
   );
   if (use_robin) {
      constexpr amrex::Real robin_b_magnitude = 0.25;
      for (int level = 0; level < num_levels; ++level) {
         const amrex::Box nodal_domain = amrex::surroundingNodes(
            hierarchy.geometry[level].Domain()
         );
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto probhi = hierarchy.geometry[level].ProbHiArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(boundary.coefficient_c(level)); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.fabbox();
            const auto coefficient_b = boundary.coefficient_b(level).array(mfi);
            const auto coefficient_c = boundary.coefficient_c(level).array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               int direction = -1;
               amrex::Real coordinate_b = 0.0;
               amrex::Real x = problo[0] + i * dx[0];
#if (AMREX_SPACEDIM >= 2)
               amrex::Real y = problo[1] + j * dx[1];
#else
               amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               amrex::Real z = problo[2] + k * dx[2];
#else
               amrex::Real z = 0.0;
#endif
               if (i < nodal_domain.smallEnd(0)) {
                  direction = 0;
                  coordinate_b = -robin_b_magnitude;
                  x = problo[0];
               } else if (i > nodal_domain.bigEnd(0)) {
                  direction = 0;
                  coordinate_b = robin_b_magnitude;
                  x = probhi[0];
               }
#if (AMREX_SPACEDIM >= 2)
               else if (j < nodal_domain.smallEnd(1)) {
                  direction = 1;
                  coordinate_b = -robin_b_magnitude;
                  y = problo[1];
               } else if (j > nodal_domain.bigEnd(1)) {
                  direction = 1;
                  coordinate_b = robin_b_magnitude;
                  y = probhi[1];
               }
#endif
#if (AMREX_SPACEDIM == 3)
               else if (k < nodal_domain.smallEnd(2)) {
                  direction = 2;
                  coordinate_b = -robin_b_magnitude;
                  z = problo[2];
               } else if (k > nodal_domain.bigEnd(2)) {
                  direction = 2;
                  coordinate_b = robin_b_magnitude;
                  z = probhi[2];
               }
#endif
               coefficient_b(i, j, k, 0) = coordinate_b;
               coefficient_c(i, j, k, 0) = direction < 0
                  ? 0.0
                  : manufactured_robin_c(
                       x, y, z, direction, coordinate_b
                    );
            });
         }
      }
   }
   const pyfluid_amrex::InitialDataEllipticResult result =
      pyfluid_amrex::solve_initial_data_poisson(
         hierarchy, solution, rhs, boundary
      );

   amrex::Real max_error = 0.0;
   for (int level = 0; level < num_levels; ++level) {
      amrex::MultiFab error(
         solution_storage[level]->boxArray(),
         hierarchy.distribution[level],
         1,
         0
      );
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto numerical = solution_storage[level]->const_array(mfi);
         const auto difference = error.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real x = problo[0] + i * dx[0];
#if (AMREX_SPACEDIM >= 2)
            const amrex::Real y = problo[1] + j * dx[1];
#else
            const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real z = problo[2] + k * dx[2];
#else
            const amrex::Real z = 0.0;
#endif
            difference(i, j, k, 0) = amrex::Math::abs(
               numerical(i, j, k, 0) - manufactured_solution(x, y, z)
            );
         });
      }
      max_error = std::max(max_error, error.norm0(0));
   }
   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::cout << std::setprecision(16)
                << "initial_data_nodal_poisson_"
                << (use_robin ? "robin" : "dirichlet")
                << " residual=" << result.residual
                << " boundary_residual=" << result.boundary_residual
                << " boundary_iterations=" << result.boundary_iterations
                << " linf_error=" << max_error << '\n';
   }
   return max_error < 5.0e-3;
}

bool validate_tov_solution()
{
   pyfluid_amrex::EOSParams eos;
   eos.polytrope_segments = 1;
   eos.segment_k[0] = 100.0;
   eos.segment_gamma[0] = 2.0;
   const pyfluid_amrex::TovSolution reference(
      eos,
      {
         1.28e-3,
         1.0e-13,
         1.0e-11,
         100000,
      }
   );
   const pyfluid_amrex::TovSolution relaxed(
      eos,
      {
         1.28e-3,
         1.0e-10,
         1.0e-8,
         100000,
      }
   );
   const pyfluid_amrex::TovObservables& values = reference.observables();
   const pyfluid_amrex::TovObservables& relaxed_values = relaxed.observables();
   const amrex::Real mass_tolerance_error = std::abs(
      relaxed_values.gravitational_mass - values.gravitational_mass
   ) / values.gravitational_mass;
   const amrex::Real radius_tolerance_error = std::abs(
      relaxed_values.areal_radius - values.areal_radius
   ) / values.areal_radius;
   const amrex::Real komar_error = std::abs(
      values.komar_mass - values.gravitational_mass
   ) / values.gravitational_mass;
   bool monotone_profile = !reference.profile().empty();
   for (std::size_t point = 1; point < reference.profile().size(); ++point) {
      pyfluid_amrex::TovProfilePoint const& previous = reference.profile()[point - 1];
      pyfluid_amrex::TovProfilePoint const& current = reference.profile()[point];
      monotone_profile = monotone_profile
         && current.areal_radius > previous.areal_radius
         && current.isotropic_radius > previous.isotropic_radius
         && current.enclosed_gravitational_mass >= previous.enclosed_gravitational_mass
         && current.enclosed_baryonic_mass >= previous.enclosed_baryonic_mass;
   }
   pyfluid_amrex::TovProfilePoint const& surface = reference.profile().back();
   const bool physical =
      values.gravitational_mass > 0.0
      && values.proper_mass > values.baryonic_mass
      && values.baryonic_mass > values.gravitational_mass
      && values.binding_energy > 0.0
      && values.compactness > 0.0
      && values.compactness < 0.5
      && values.surface_redshift > 0.0
      && values.tidal_observables_available
      && values.love_number_k2 > 0.0
      && values.tidal_deformability > 0.0
      && surface.log_specific_enthalpy == 0.0
      && surface.rho == 0.0
      && surface.pressure == 0.0;
   const std::vector<pyfluid_amrex::TovSolution> sequence =
      pyfluid_amrex::solve_tov_sequence(eos, {8.0e-4, 1.0e-3, 1.28e-3});
   const bool sequence_increases = sequence.size() == 3
      && sequence[0].observables().gravitational_mass
         < sequence[1].observables().gravitational_mass
      && sequence[1].observables().gravitational_mass
         < sequence[2].observables().gravitational_mass;
   bool named_eos_sequences_are_physical = true;
   for (char const* preset_name : {"sly", "apr4", "h4", "ms1"}) {
      const pyfluid_amrex::EOSParams preset_eos =
         pyfluid_amrex::eos::make_read_piecewise_polytrope(
            pyfluid_amrex::eos::piecewise_polytrope_preset(preset_name),
            2.0,
            1.0e-10,
            1.0e-14
         );
      const std::vector<pyfluid_amrex::TovSolution> preset_sequence =
         pyfluid_amrex::solve_tov_sequence(
            preset_eos, {8.0e-4, 1.5e-3, 3.0e-3}
         );
      for (pyfluid_amrex::TovSolution const& solution : preset_sequence) {
         pyfluid_amrex::TovObservables const& observable =
            solution.observables();
         named_eos_sequences_are_physical = named_eos_sequences_are_physical
            && std::isfinite(observable.gravitational_mass)
            && std::isfinite(observable.areal_radius)
            && std::isfinite(observable.tidal_deformability)
            && observable.gravitational_mass > 0.0
            && observable.areal_radius > 2.0 * observable.gravitational_mass
            && observable.baryonic_mass > observable.gravitational_mass
            && observable.tidal_deformability > 0.0
            && std::abs(observable.komar_mass - observable.gravitational_mass)
               < 1.0e-7 * observable.gravitational_mass;
      }
   }
   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::cout << std::setprecision(16)
                << "tov_solution M=" << values.gravitational_mass
                << " Mb=" << values.baryonic_mass
                << " Mp=" << values.proper_mass
                << " Mkomar=" << values.komar_mass
                << " R=" << values.areal_radius
                << " k2=" << values.love_number_k2
                << " Lambda=" << values.tidal_deformability
                << " tolerance_errors=(" << mass_tolerance_error
                << ", " << radius_tolerance_error << ")"
                << " komar_error=" << komar_error << '\n';
   }
   return physical
      && monotone_profile
      && sequence_increases
      && named_eos_sequences_are_physical
      && mass_tolerance_error < 1.0e-7
      && radius_tolerance_error < 1.0e-7
      && komar_error < 1.0e-8;
}

bool validate_uniform_rotation_first_integral()
{
   pyfluid_amrex::RotatingStarParameters parameters;
   parameters.active = true;
   parameters.angular_velocity = 0.02;
   parameters.rotation_axis = {0.0, 0.0, 1.0};
   parameters.rotation_center = {0.5, -0.5, 0.25};
   parameters.injection_energy = 0.84;
   pyfluid_amrex::SpacetimeMetric metric;
   metric.lapse = 0.8;
   metric.shift1 = 0.01;
   metric.shift2 = -0.02;
   metric.shift3 = 0.005;
   metric.gamma11 = 1.2;
   metric.gamma12 = 0.03;
   metric.gamma13 = -0.01;
   metric.gamma22 = 1.1;
   metric.gamma23 = 0.02;
   metric.gamma33 = 0.9;
   const pyfluid_amrex::PatchCoordinate physical{2.5, 0.5, -0.75};
   const pyfluid_amrex::RotatingStarPoint point =
      pyfluid_amrex::rotating_star_point(parameters, physical, metric);
   const amrex::Real expected_v1 = (-0.02 + metric.shift1) / metric.lapse;
   const amrex::Real expected_v2 = (0.04 + metric.shift2) / metric.lapse;
   const amrex::Real expected_v3 = metric.shift3 / metric.lapse;
   const amrex::Real expected_velocity_squared =
      pyfluid_amrex::spatial_vector_norm_squared(
         expected_v1, expected_v2, expected_v3, metric
      );
   const amrex::Real expected_log_enthalpy = std::log(
      parameters.injection_energy / metric.lapse
   ) - 0.5 * std::log(1.0 - expected_velocity_squared);
   const amrex::Real error = std::max({
      std::abs(point.v1 - expected_v1),
      std::abs(point.v2 - expected_v2),
      std::abs(point.v3 - expected_v3),
      std::abs(point.velocity_squared - expected_velocity_squared),
      std::abs(point.log_specific_enthalpy - expected_log_enthalpy),
   });
   bool catalog_valid = false;
   for (pyfluid_amrex::RotationLawInfo const& law
        : pyfluid_amrex::RotationLawCatalog) {
      catalog_valid = catalog_valid
         || (law.id == pyfluid_amrex::RotationLaw::Uniform
             && law.name == "uniform");
   }
   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::cout << std::setprecision(16)
                << "uniform_rotation_first_integral error=" << error
                << " timelike=" << point.timelike
                << " catalog_valid=" << catalog_valid << '\n';
   }
   return point.timelike && catalog_valid && error < 1.0e-14;
}

bool validate_differential_rotation_first_integrals()
{
   pyfluid_amrex::RotatingStarParameters j_parameters;
   j_parameters.active = true;
   j_parameters.rotation_law = pyfluid_amrex::RotationLaw::JConstant;
   j_parameters.angular_velocity = 0.04;
   j_parameters.differential_rotation_length = 3.0;
   j_parameters.rotation_axis = {0.0, 0.0, 1.0};
   j_parameters.injection_energy = 0.9;
   pyfluid_amrex::RotatingStarParameters q_parameters = j_parameters;
   q_parameters.rotation_law = pyfluid_amrex::RotationLaw::QPower;
   q_parameters.rotation_power_q = 1.0;

   pyfluid_amrex::SpacetimeMetric metric;
   metric.lapse = 1.0;
   metric.gamma11 = 1.0;
   metric.gamma22 = 1.0;
   metric.gamma33 = 1.0;
   const pyfluid_amrex::PatchCoordinate physical{1.0, 0.0, 0.0};
   const pyfluid_amrex::RotatingStarPoint j_point =
      pyfluid_amrex::rotating_star_point(j_parameters, physical, metric);
   const pyfluid_amrex::RotatingStarPoint q_point =
      pyfluid_amrex::rotating_star_point(q_parameters, physical, metric);

   const amrex::Real omega = j_point.v2;
   const amrex::Real metric_j = omega / (1.0 - omega * omega);
   const amrex::Real law_j =
      pyfluid_amrex::generated::j_constant_specific_angular_momentum(
         omega,
         j_parameters.angular_velocity,
         j_parameters.differential_rotation_length
      );
   const amrex::Real j_integral =
      pyfluid_amrex::generated::j_constant_rotation_integral(
         omega,
         j_parameters.angular_velocity,
         j_parameters.differential_rotation_length
      );
   const amrex::Real q_integral =
      pyfluid_amrex::generated::q_power_rotation_integral(
         omega,
         q_parameters.angular_velocity,
         q_parameters.differential_rotation_length,
         q_parameters.rotation_power_q
      );
   const amrex::Real expected_log_enthalpy =
      std::log(j_parameters.injection_energy)
      - 0.5 * std::log(1.0 - omega * omega) - j_integral;
   const amrex::Real q_two_integral =
      pyfluid_amrex::generated::q_power_two_rotation_integral(
         0.8 * q_parameters.angular_velocity,
         q_parameters.angular_velocity,
         q_parameters.differential_rotation_length
      );
   pyfluid_amrex::RotatingStarParameters negative_j_parameters = j_parameters;
   negative_j_parameters.angular_velocity = -j_parameters.angular_velocity;
   pyfluid_amrex::RotatingStarParameters negative_q_parameters = q_parameters;
   negative_q_parameters.angular_velocity = -q_parameters.angular_velocity;
   const pyfluid_amrex::RotatingStarPoint negative_j_point =
      pyfluid_amrex::rotating_star_point(
         negative_j_parameters, physical, metric
      );
   const pyfluid_amrex::RotatingStarPoint negative_q_point =
      pyfluid_amrex::rotating_star_point(
         negative_q_parameters, physical, metric
      );
   pyfluid_amrex::RotatingStarParameters fractional_q_parameters =
      negative_q_parameters;
   fractional_q_parameters.rotation_power_q = 1.5;
   const pyfluid_amrex::RotatingStarPoint fractional_q_point =
      pyfluid_amrex::rotating_star_point(
         fractional_q_parameters, physical, metric
      );
   const amrex::Real fractional_omega = fractional_q_point.v2;
   const amrex::Real fractional_metric_j = fractional_omega
      / (1.0 - fractional_omega * fractional_omega);
   const amrex::Real fractional_law_j =
      pyfluid_amrex::generated::q_power_specific_angular_momentum(
         fractional_omega,
         fractional_q_parameters.angular_velocity,
         fractional_q_parameters.differential_rotation_length,
         fractional_q_parameters.rotation_power_q
      );
   const amrex::Real identity_error = std::max({
      std::abs(j_point.log_specific_enthalpy - q_point.log_specific_enthalpy),
      std::abs(j_point.v1 - q_point.v1),
      std::abs(j_point.v2 - q_point.v2),
      std::abs(j_point.v3 - q_point.v3),
      std::abs(j_integral - q_integral),
      std::abs(
         negative_j_point.log_specific_enthalpy
         - negative_q_point.log_specific_enthalpy
      ),
      std::abs(negative_j_point.v2 - negative_q_point.v2),
   });
   const amrex::Real residual = std::abs(metric_j - law_j);
   const amrex::Real fractional_residual = std::abs(
      fractional_metric_j - fractional_law_j
   );
   const amrex::Real enthalpy_error = std::abs(
      j_point.log_specific_enthalpy - expected_log_enthalpy
   );
   bool has_uniform = false;
   bool has_j_constant = false;
   bool has_q_power = false;
   for (pyfluid_amrex::RotationLawInfo const& law
        : pyfluid_amrex::RotationLawCatalog) {
      has_uniform = has_uniform
         || law.id == pyfluid_amrex::RotationLaw::Uniform;
      has_j_constant = has_j_constant
         || law.id == pyfluid_amrex::RotationLaw::JConstant;
      has_q_power = has_q_power
         || law.id == pyfluid_amrex::RotationLaw::QPower;
   }
   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::cout << std::setprecision(16)
                << "differential_rotation_first_integrals identity_error="
                << identity_error
                << " residual=" << residual
                << " fractional_negative_residual=" << fractional_residual
                << " enthalpy_error=" << enthalpy_error
                << " q_two_integral=" << q_two_integral
                << " catalog_valid="
                << (has_uniform && has_j_constant && has_q_power) << '\n';
   }
   return j_point.timelike && q_point.timelike
      && negative_j_point.timelike && negative_q_point.timelike
      && fractional_q_point.timelike
      && has_uniform && has_j_constant && has_q_power
      && std::isfinite(q_two_integral)
      && identity_error < 1.0e-13
      && residual < 1.0e-13
      && fractional_residual < 1.0e-13
      && enthalpy_error < 1.0e-14;
}

#if (AMREX_SPACEDIM == 3)
bool validate_tov_ctt()
{
   constexpr int coarse_cells = 32;
   const amrex::IntVect coarse_lo(0, 0, 0);
   const amrex::IntVect coarse_hi(
      coarse_cells - 1, coarse_cells - 1, coarse_cells - 1
   );
   const amrex::Box coarse_domain(coarse_lo, coarse_hi);
   const amrex::IntVect refinement_ratio(2, 2, 2);
   const amrex::Box fine_domain = amrex::refine(coarse_domain, refinement_ratio);
   const amrex::Box fine_patch(
      amrex::IntVect(coarse_cells / 2, coarse_cells / 2, coarse_cells / 2),
      amrex::IntVect(
         3 * coarse_cells / 2 - 1,
         3 * coarse_cells / 2 - 1,
         3 * coarse_cells / 2 - 1
      )
   );
   const amrex::RealBox physical_domain(
      {-16.0, -16.0, -16.0},
      {16.0, 16.0, 16.0}
   );
   const amrex::Array<int, AMREX_SPACEDIM> periodic{0, 0, 0};

   pyfluid_amrex::InitialDataEllipticHierarchy hierarchy;
   hierarchy.geometry.emplace_back(
      coarse_domain, physical_domain, amrex::CoordSys::cartesian, periodic
   );
   hierarchy.geometry.emplace_back(
      fine_domain, physical_domain, amrex::CoordSys::cartesian, periodic
   );
   hierarchy.grids.emplace_back(coarse_domain);
   hierarchy.grids.emplace_back(fine_patch);
   hierarchy.grids[0].maxSize(16);
   hierarchy.grids[1].maxSize(16);
   hierarchy.distribution.emplace_back(hierarchy.grids[0]);
   hierarchy.distribution.emplace_back(hierarchy.grids[1]);
   hierarchy.refinement_ratio.push_back(refinement_ratio);

   pyfluid_amrex::RunConfig config;
   config.ic = "tov";
   config.polytropic_k = 100.0;
   config.polytropic_gamma = 2.0;
   config.thermal_gamma = 2.0;
   config.rho_floor = 1.0e-10;
   config.tov_central_density = 1.28e-3;
   config.tov_absolute_tolerance = 1.0e-12;
   config.tov_relative_tolerance = 1.0e-10;
   config.tov_max_steps = 100000;
   const pyfluid_amrex::TovInitialData tov(config);
   const pyfluid_amrex::TovInitialDataView tov_view = tov.view();

   amrex::Vector<std::unique_ptr<amrex::MultiFab>> conformal_storage(2);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> energy_storage(2);
   amrex::Vector<amrex::MultiFab*> conformal_factor(2);
   amrex::Vector<amrex::MultiFab const*> energy_density(2);
   for (int level = 0; level < 2; ++level) {
      conformal_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 1
      );
      energy_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 0
      );
      conformal_storage[level]->setVal(1.0);
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      for (amrex::MFIter mfi(*energy_storage[level]); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto energy = energy_storage[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
            const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
            const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
            const pyfluid_amrex::TovSample point = tov_view.sample(
               std::sqrt(x * x + y * y + z * z)
            );
            energy(i, j, k, 0) = point.rho * (
               1.0 + point.specific_internal_energy
            );
         });
      }
      conformal_factor[level] = conformal_storage[level].get();
      energy_density[level] = energy_storage[level].get();
   }

   auto boundary = pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
      hierarchy, 1, 0.0, 0.0, 0.0
   );
   pyfluid_amrex::set_centered_power_law_robin_boundary(
      hierarchy, boundary, {0.0}, 1.0
   );

   pyfluid_amrex::CttHamiltonianOptions options;
   options.relative_update_tolerance = 1.0e-10;
   options.relaxation = 0.5;
   const pyfluid_amrex::CttHamiltonianResult result =
      pyfluid_amrex::solve_ctt_time_symmetric_hamiltonian(
         hierarchy, conformal_factor, energy_density, boundary, options
      );

   amrex::Real max_error = 0.0;
   for (int level = 0; level < 2; ++level) {
      amrex::MultiFab error(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 0
      );
      const auto level_problo = hierarchy.geometry[level].ProbLoArray();
      const auto level_dx = hierarchy.geometry[level].CellSizeArray();
      for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto numerical = conformal_storage[level]->const_array(mfi);
         const auto difference = error.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real x = level_problo[0] + (i + 0.5) * level_dx[0];
            const amrex::Real y = level_problo[1] + (j + 0.5) * level_dx[1];
            const amrex::Real z = level_problo[2] + (k + 0.5) * level_dx[2];
            const pyfluid_amrex::TovSample point = tov_view.sample(
               std::sqrt(x * x + y * y + z * z)
            );
            difference(i, j, k, 0) = amrex::Math::abs(
               numerical(i, j, k, 0) - point.conformal_factor
            );
         });
      }
      max_error = std::max(max_error, error.norm0(0));
   }

   amrex::Vector<std::unique_ptr<amrex::MultiFab>> xcts_conformal_storage(2);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> xcts_lapse_storage(2);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> xcts_shift_storage(2);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> xcts_extrinsic_storage(2);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> xcts_energy_storage(2);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> xcts_momentum_storage(2);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> xcts_stress_storage(2);
   amrex::Vector<amrex::MultiFab*> xcts_conformal(2);
   amrex::Vector<amrex::MultiFab*> xcts_lapse(2);
   amrex::Vector<amrex::MultiFab*> xcts_shift(2);
   amrex::Vector<amrex::MultiFab*> xcts_extrinsic(2);
   amrex::Vector<amrex::MultiFab const*> xcts_energy(2);
   amrex::Vector<amrex::MultiFab const*> xcts_momentum(2);
   amrex::Vector<amrex::MultiFab const*> xcts_stress(2);
   for (int level = 0; level < 2; ++level) {
      const amrex::BoxArray vertex_grids = amrex::convert(
         hierarchy.grids[level], amrex::IntVect::TheNodeVector()
      );
      xcts_conformal_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 1
      );
      xcts_lapse_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 1
      );
      xcts_shift_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 1
      );
      xcts_extrinsic_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids,
         hierarchy.distribution[level],
         pyfluid_amrex::CONFORMAL_A_NUM_COMPONENTS,
         1
      );
      xcts_energy_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      xcts_momentum_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 0
      );
      xcts_stress_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      xcts_conformal_storage[level]->setVal(1.0);
      xcts_lapse_storage[level]->setVal(1.0);
      xcts_shift_storage[level]->setVal(0.0);
      xcts_extrinsic_storage[level]->setVal(0.0);
      xcts_momentum_storage[level]->setVal(0.0);
      const auto level_problo = hierarchy.geometry[level].ProbLoArray();
      const auto level_dx = hierarchy.geometry[level].CellSizeArray();
      for (amrex::MFIter mfi(*xcts_energy_storage[level]); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto energy = xcts_energy_storage[level]->array(mfi);
         const auto stress = xcts_stress_storage[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const amrex::Real x = level_problo[0] + i * level_dx[0];
            const amrex::Real y = level_problo[1] + j * level_dx[1];
            const amrex::Real z = level_problo[2] + k * level_dx[2];
            const pyfluid_amrex::TovSample point = tov_view.sample(
               std::sqrt(x * x + y * y + z * z)
            );
            energy(i, j, k, 0) = point.rho * (
               1.0 + point.specific_internal_energy
            );
            stress(i, j, k, 0) = 3.0 * point.pressure;
         });
      }
      xcts_conformal[level] = xcts_conformal_storage[level].get();
      xcts_lapse[level] = xcts_lapse_storage[level].get();
      xcts_shift[level] = xcts_shift_storage[level].get();
      xcts_extrinsic[level] = xcts_extrinsic_storage[level].get();
      xcts_energy[level] = xcts_energy_storage[level].get();
      xcts_momentum[level] = xcts_momentum_storage[level].get();
      xcts_stress[level] = xcts_stress_storage[level].get();
   }
   auto xcts_conformal_boundary =
      pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
         hierarchy,
         1,
         1.0,
         0.0,
         0.0,
         pyfluid_amrex::InitialDataEllipticCentering::vertex
      );
   pyfluid_amrex::set_centered_power_law_robin_boundary(
      hierarchy, xcts_conformal_boundary, {0.0}, 1.0
   );
   auto xcts_lapse_boundary =
      pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
         hierarchy,
         1,
         1.0,
         0.0,
         0.0,
         pyfluid_amrex::InitialDataEllipticCentering::vertex
      );
   pyfluid_amrex::set_centered_power_law_robin_boundary(
      hierarchy, xcts_lapse_boundary, {0.0}, 1.0
   );
   auto xcts_shift_boundary =
      pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
         hierarchy,
         3,
         1.0,
         0.0,
         0.0,
         pyfluid_amrex::InitialDataEllipticCentering::vertex
      );
   pyfluid_amrex::XctsOptions xcts_options;
   xcts_options.max_coupled_iterations = 100;
   xcts_options.relative_update_tolerance = 1.0e-10;
   xcts_options.relaxation = 0.5;
   const pyfluid_amrex::XctsResult xcts_result =
      pyfluid_amrex::solve_xcts_constraints(
         hierarchy,
         xcts_conformal,
         xcts_lapse,
         xcts_shift,
         xcts_extrinsic,
         xcts_energy,
         xcts_momentum,
         xcts_stress,
         xcts_conformal_boundary,
         xcts_lapse_boundary,
         xcts_shift_boundary,
         xcts_options
      );
   amrex::Real xcts_conformal_error = 0.0;
   amrex::Real xcts_lapse_error = 0.0;
   amrex::Real xcts_shift_error = 0.0;
   for (int level = 0; level < 2; ++level) {
      amrex::MultiFab conformal_error(
         xcts_conformal_storage[level]->boxArray(),
         hierarchy.distribution[level],
         1,
         0
      );
      amrex::MultiFab lapse_error(
         xcts_lapse_storage[level]->boxArray(),
         hierarchy.distribution[level],
         1,
         0
      );
      const auto level_problo = hierarchy.geometry[level].ProbLoArray();
      const auto level_dx = hierarchy.geometry[level].CellSizeArray();
      for (amrex::MFIter mfi(conformal_error); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto numerical_conformal =
            xcts_conformal_storage[level]->const_array(mfi);
         const auto numerical_lapse = xcts_lapse_storage[level]->const_array(mfi);
         const auto conformal_difference = conformal_error.array(mfi);
         const auto lapse_difference = lapse_error.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const amrex::Real x = level_problo[0] + i * level_dx[0];
            const amrex::Real y = level_problo[1] + j * level_dx[1];
            const amrex::Real z = level_problo[2] + k * level_dx[2];
            const pyfluid_amrex::TovSample point = tov_view.sample(
               std::sqrt(x * x + y * y + z * z)
            );
            conformal_difference(i, j, k, 0) = amrex::Math::abs(
               numerical_conformal(i, j, k, 0) - point.conformal_factor
            );
            lapse_difference(i, j, k, 0) = amrex::Math::abs(
               numerical_lapse(i, j, k, 0)
               - point.lapse * point.conformal_factor
            );
         });
      }
      xcts_conformal_error = std::max(
         xcts_conformal_error, conformal_error.norm0(0)
      );
      xcts_lapse_error = std::max(xcts_lapse_error, lapse_error.norm0(0));
      for (int component = 0; component < 3; ++component) {
         xcts_shift_error = std::max(
            xcts_shift_error,
            xcts_shift_storage[level]->norm0(component, 0, false)
         );
      }
   }
   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::cout << std::setprecision(16)
                << "ctt_tov converged=" << result.converged
                << " iterations=" << result.nonlinear_iterations
                << " relative_update=" << result.relative_update
                << " linear_residual=" << result.linear_residual
                << " linf_error=" << max_error << '\n';
      std::cout << "xcts_tov converged=" << xcts_result.converged
                << " iterations=" << xcts_result.coupled_iterations
                << " relative_update=" << xcts_result.relative_update
                << " hamiltonian_residual=" << xcts_result.hamiltonian_residual
                << " lapse_residual=" << xcts_result.lapse_residual
                << " momentum_residual=" << xcts_result.momentum_residual
                << " conformal_linf_error=" << xcts_conformal_error
                << " lapse_conformal_linf_error=" << xcts_lapse_error
                << " shift_linf_error=" << xcts_shift_error << '\n';
   }
   return result.converged
      && max_error < 5.0e-2
      && xcts_result.converged
      && xcts_conformal_error < 5.0e-2
      && xcts_lapse_error < 5.0e-2
      && xcts_shift_error < 1.0e-10;
}
#endif

} // namespace

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   int return_code = 0;
   {
      if (!validate_tov_solution()) {
         return_code = 1;
      }
      if (!validate_uniform_rotation_first_integral()) {
         return_code = 1;
      }
      if (!validate_differential_rotation_first_integrals()) {
         return_code = 1;
      }
      constexpr int coarse_cells = 32;
      const amrex::IntVect coarse_lo(AMREX_D_DECL(0, 0, 0));
      const amrex::IntVect coarse_hi(AMREX_D_DECL(
         coarse_cells - 1,
         coarse_cells - 1,
         coarse_cells - 1
      ));
      const amrex::Box coarse_domain(coarse_lo, coarse_hi);
      const amrex::IntVect refinement_ratio(AMREX_D_DECL(2, 2, 2));
      const amrex::Box fine_domain = amrex::refine(coarse_domain, refinement_ratio);
      const amrex::IntVect fine_lo(AMREX_D_DECL(
         coarse_cells / 2,
         coarse_cells / 2,
         coarse_cells / 2
      ));
      const amrex::IntVect fine_hi(AMREX_D_DECL(
         3 * coarse_cells / 2 - 1,
         3 * coarse_cells / 2 - 1,
         3 * coarse_cells / 2 - 1
      ));
      const amrex::Box fine_patch(fine_lo, fine_hi);
      const amrex::RealBox physical_domain(
         AMREX_D_DECL(0.0, 0.0, 0.0),
         AMREX_D_DECL(1.0, 1.0, 1.0)
      );
      const amrex::Array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(0, 0, 0)};

      pyfluid_amrex::InitialDataEllipticHierarchy hierarchy;
      hierarchy.geometry.emplace_back(
         coarse_domain, physical_domain, amrex::CoordSys::cartesian, periodic
      );
      hierarchy.geometry.emplace_back(
         fine_domain, physical_domain, amrex::CoordSys::cartesian, periodic
      );
      hierarchy.grids.emplace_back(coarse_domain);
      hierarchy.grids.emplace_back(fine_patch);
      hierarchy.grids[0].maxSize(16);
      hierarchy.grids[1].maxSize(16);
      hierarchy.distribution.emplace_back(hierarchy.grids[0]);
      hierarchy.distribution.emplace_back(hierarchy.grids[1]);
      hierarchy.refinement_ratio.push_back(refinement_ratio);

      const amrex::Real power_law_robin_cell_error =
         validate_centered_power_law_robin(
            hierarchy, pyfluid_amrex::InitialDataEllipticCentering::cell
         );
      const amrex::Real power_law_robin_vertex_error =
         validate_centered_power_law_robin(
            hierarchy, pyfluid_amrex::InitialDataEllipticCentering::vertex
         );
      if (amrex::ParallelDescriptor::IOProcessor()) {
         std::cout << std::setprecision(16)
                   << "centered_power_law_robin cell_identity_error="
                   << power_law_robin_cell_error
                   << " vertex_identity_error="
                   << power_law_robin_vertex_error << '\n';
      }
      if (!(power_law_robin_cell_error < 1.0e-13)
          || !(power_law_robin_vertex_error < 1.0e-13)) {
         return_code = 1;
      }
      pyfluid_amrex::InitialDataEllipticHierarchy nodal_hierarchy = hierarchy;
#if (AMREX_SPACEDIM == 1)
      // AMReX 25.11 does not implement multilevel reflux for MLNodeLaplacian
      // in 1D. The physical TOV provider remains a native 1D ODE solve.
      nodal_hierarchy.geometry.resize(1);
      nodal_hierarchy.grids.resize(1);
      nodal_hierarchy.distribution.resize(1);
      nodal_hierarchy.refinement_ratio.clear();
#endif
      if (!validate_nodal_poisson(nodal_hierarchy, false)
          || !validate_nodal_poisson(nodal_hierarchy, true)) {
         return_code = 1;
      }

      amrex::Vector<std::unique_ptr<amrex::MultiFab>> solution_storage(2);
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhs_storage(2);
      amrex::Vector<amrex::MultiFab*> solution(2);
      amrex::Vector<amrex::MultiFab const*> rhs(2);
      for (int level = 0; level < 2; ++level) {
         solution_storage[level] = std::make_unique<amrex::MultiFab>(
            hierarchy.grids[level], hierarchy.distribution[level], 1, 1
         );
         rhs_storage[level] = std::make_unique<amrex::MultiFab>(
            hierarchy.grids[level], hierarchy.distribution[level], 1, 0
         );
         solution_storage[level]->setVal(0.0);
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(*rhs_storage[level]); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto values = rhs_storage[level]->array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
#else
               const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
#else
               const amrex::Real z = 0.0;
#endif
               values(i, j, k, 0) = manufactured_rhs(x, y, z);
            });
         }
         solution[level] = solution_storage[level].get();
         rhs[level] = rhs_storage[level].get();
      }

      constexpr amrex::Real robin_b_magnitude = 0.25;
      auto boundary = pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
         hierarchy, 1, 1.0, 0.0, 0.0
      );
      for (int level = 0; level < 2; ++level) {
         const amrex::Box domain = hierarchy.geometry[level].Domain();
         const amrex::Dim3 domain_lo = amrex::lbound(domain);
         const amrex::Dim3 domain_hi = amrex::ubound(domain);
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto probhi = hierarchy.geometry[level].ProbHiArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(boundary.coefficient_c(level)); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.fabbox();
            const auto coefficient_b = boundary.coefficient_b(level).array(mfi);
            const auto coefficient_c = boundary.coefficient_c(level).array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               int direction = -1;
               amrex::Real coordinate_b = 0.0;
               amrex::Real x = problo[0] + (i + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               amrex::Real y = problo[1] + (j + 0.5) * dx[1];
#else
               amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               amrex::Real z = problo[2] + (k + 0.5) * dx[2];
#else
               amrex::Real z = 0.0;
#endif
               if (i < domain_lo.x) {
                  direction = 0;
                  coordinate_b = -robin_b_magnitude;
                  x = problo[0];
               } else if (i > domain_hi.x) {
                  direction = 0;
                  coordinate_b = robin_b_magnitude;
                  x = probhi[0];
               }
#if (AMREX_SPACEDIM >= 2)
               else if (j < domain_lo.y) {
                  direction = 1;
                  coordinate_b = -robin_b_magnitude;
                  y = problo[1];
               } else if (j > domain_hi.y) {
                  direction = 1;
                  coordinate_b = robin_b_magnitude;
                  y = probhi[1];
               }
#endif
#if (AMREX_SPACEDIM == 3)
               else if (k < domain_lo.z) {
                  direction = 2;
                  coordinate_b = -robin_b_magnitude;
                  z = problo[2];
               } else if (k > domain_hi.z) {
                  direction = 2;
                  coordinate_b = robin_b_magnitude;
                  z = probhi[2];
               }
#endif
               coefficient_b(i, j, k, 0) = coordinate_b;
               coefficient_c(i, j, k, 0) = direction < 0
                  ? 0.0
                  : manufactured_robin_c(x, y, z, direction, coordinate_b);
            });
         }
      }
      const pyfluid_amrex::InitialDataEllipticResult result =
         pyfluid_amrex::solve_initial_data_poisson(
            hierarchy, solution, rhs, boundary
         );

      amrex::Real max_error = 0.0;
      for (int level = 0; level < 2; ++level) {
         amrex::MultiFab error(
            hierarchy.grids[level], hierarchy.distribution[level], 1, 0
         );
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto numerical = solution_storage[level]->const_array(mfi);
            const auto difference = error.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
#else
               const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
#else
               const amrex::Real z = 0.0;
#endif
               difference(i, j, k, 0) = amrex::Math::abs(
                  numerical(i, j, k, 0) - manufactured_solution(x, y, z)
               );
            });
         }
         max_error = std::max(max_error, error.norm0(0));
      }

      if (amrex::ParallelDescriptor::IOProcessor()) {
         std::cout << std::setprecision(16)
                   << "initial_data_poisson residual=" << result.residual
                   << " linf_error=" << max_error << '\n';
      }
      if (!(max_error < 5.0e-3)) {
         return_code = 1;
      }

      constexpr amrex::Real ctt_amplitude = 0.01;
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> conformal_storage(2);
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> energy_storage(2);
      amrex::Vector<amrex::MultiFab*> conformal_factor(2);
      amrex::Vector<amrex::MultiFab const*> energy_density(2);
      auto ctt_boundary =
         pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
            hierarchy, 1, 1.0, 0.0, 0.0
         );
      for (int level = 0; level < 2; ++level) {
         conformal_storage[level] = std::make_unique<amrex::MultiFab>(
            hierarchy.grids[level], hierarchy.distribution[level], 1, 1
         );
         energy_storage[level] = std::make_unique<amrex::MultiFab>(
            hierarchy.grids[level], hierarchy.distribution[level], 1, 0
         );
         conformal_storage[level]->setVal(1.0);
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(*energy_storage[level]); mfi.isValid(); ++mfi) {
            const amrex::Box valid_box = mfi.validbox();
            const auto energy = energy_storage[level]->array(mfi);
            amrex::ParallelFor(valid_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
#else
               const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
#else
               const amrex::Real z = 0.0;
#endif
               energy(i, j, k, 0) = manufactured_ctt_energy_density(
                  x, y, z, ctt_amplitude
               );
            });
         }
         conformal_factor[level] = conformal_storage[level].get();
         energy_density[level] = energy_storage[level].get();
      }
      pyfluid_amrex::CttHamiltonianOptions ctt_options;
      ctt_options.relative_update_tolerance = 1.0e-11;
      ctt_options.relaxation = 0.25;
      const pyfluid_amrex::CttHamiltonianResult ctt_result =
         pyfluid_amrex::solve_ctt_time_symmetric_hamiltonian(
            hierarchy,
            conformal_factor,
            energy_density,
            ctt_boundary,
            ctt_options
         );

      amrex::Real ctt_max_error = 0.0;
      for (int level = 0; level < 2; ++level) {
         amrex::MultiFab error(
            hierarchy.grids[level], hierarchy.distribution[level], 1, 0
         );
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto numerical = conformal_storage[level]->const_array(mfi);
            const auto difference = error.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
               const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
#else
               const amrex::Real y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
#else
               const amrex::Real z = 0.0;
#endif
               difference(i, j, k, 0) = amrex::Math::abs(
                  numerical(i, j, k, 0)
                  - manufactured_ctt_conformal_factor(x, y, z, ctt_amplitude)
               );
            });
         }
         ctt_max_error = std::max(ctt_max_error, error.norm0(0));
      }
      const amrex::Real ctt_solution_min = conformal_storage[0]->min(0);
      const amrex::Real ctt_solution_max = conformal_storage[0]->max(0);
      if (amrex::ParallelDescriptor::IOProcessor()) {
         std::cout << std::setprecision(16)
                   << "ctt_hamiltonian converged=" << ctt_result.converged
                   << " iterations=" << ctt_result.nonlinear_iterations
                   << " relative_update=" << ctt_result.relative_update
                   << " linear_residual=" << ctt_result.linear_residual
                   << " solution_min=" << ctt_solution_min
                   << " solution_max=" << ctt_solution_max
                   << " linf_error=" << ctt_max_error << '\n';
      }
      if (!ctt_result.converged || !(ctt_max_error < 5.0e-3)) {
         return_code = 1;
      }
#if (AMREX_SPACEDIM == 3)
      constexpr amrex::Real momentum_amplitude = 0.01;
      for (const auto centering : std::array{
              pyfluid_amrex::InitialDataEllipticCentering::cell,
              pyfluid_amrex::InitialDataEllipticCentering::vertex,
           }) {
         const bool vertex_centered =
            centering == pyfluid_amrex::InitialDataEllipticCentering::vertex;
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> vector_storage(2);
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> momentum_source_storage(2);
         amrex::Vector<amrex::MultiFab*> vector_potential(2);
         amrex::Vector<amrex::MultiFab const*> momentum_source(2);
         for (int level = 0; level < 2; ++level) {
            const amrex::BoxArray field_grids = vertex_centered
               ? amrex::convert(
                    hierarchy.grids[level], amrex::IntVect::TheNodeVector()
                 )
               : hierarchy.grids[level];
            vector_storage[level] = std::make_unique<amrex::MultiFab>(
               field_grids, hierarchy.distribution[level], 3, 1
            );
            momentum_source_storage[level] = std::make_unique<amrex::MultiFab>(
               field_grids, hierarchy.distribution[level], 3, 0
            );
            vector_storage[level]->setVal(0.0);
            const auto problo = hierarchy.geometry[level].ProbLoArray();
            const auto dx = hierarchy.geometry[level].CellSizeArray();
            const amrex::Real offset = vertex_centered ? 0.0 : 0.5;
            for (amrex::MFIter mfi(*momentum_source_storage[level]); mfi.isValid(); ++mfi) {
               const amrex::Box box = mfi.validbox();
               const auto source_values = momentum_source_storage[level]->array(mfi);
               amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const amrex::Real x = problo[0] + (i + offset) * dx[0];
                  const amrex::Real y = problo[1] + (j + offset) * dx[1];
                  const amrex::Real z = problo[2] + (k + offset) * dx[2];
                  source_values(i, j, k, 0) =
                     pyfluid_amrex::generated::ctt_manufactured_momentum_source_1(
                        x, y, z, momentum_amplitude
                     );
                  source_values(i, j, k, 1) =
                     pyfluid_amrex::generated::ctt_manufactured_momentum_source_2(
                        x, y, z, momentum_amplitude
                     );
                  source_values(i, j, k, 2) =
                     pyfluid_amrex::generated::ctt_manufactured_momentum_source_3(
                        x, y, z, momentum_amplitude
                     );
               });
            }
            vector_potential[level] = vector_storage[level].get();
            momentum_source[level] = momentum_source_storage[level].get();
         }
         auto momentum_boundary =
            pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
               hierarchy, 3, 1.0, 0.0, 0.0, centering
            );
         const pyfluid_amrex::InitialDataEllipticResult momentum_result =
            pyfluid_amrex::solve_initial_data_conformal_vector_laplacian(
               hierarchy,
               vector_potential,
               momentum_source,
               momentum_boundary
            );
         amrex::Real momentum_max_error = 0.0;
         for (int level = 0; level < 2; ++level) {
            amrex::MultiFab error(
               vector_storage[level]->boxArray(),
               hierarchy.distribution[level],
               3,
               0
            );
            const auto problo = hierarchy.geometry[level].ProbLoArray();
            const auto dx = hierarchy.geometry[level].CellSizeArray();
            const amrex::Real offset = vertex_centered ? 0.0 : 0.5;
            for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
               const amrex::Box box = mfi.validbox();
               const auto numerical = vector_storage[level]->const_array(mfi);
               const auto difference = error.array(mfi);
               amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const amrex::Real x = problo[0] + (i + offset) * dx[0];
                  const amrex::Real y = problo[1] + (j + offset) * dx[1];
                  const amrex::Real z = problo[2] + (k + offset) * dx[2];
                  difference(i, j, k, 0) = amrex::Math::abs(
                     numerical(i, j, k, 0)
                     - pyfluid_amrex::generated::ctt_manufactured_vector_potential_1(
                        x, y, z, momentum_amplitude
                     )
                  );
                  difference(i, j, k, 1) = amrex::Math::abs(
                     numerical(i, j, k, 1)
                     - pyfluid_amrex::generated::ctt_manufactured_vector_potential_2(
                        x, y, z, momentum_amplitude
                     )
                  );
                  difference(i, j, k, 2) = amrex::Math::abs(
                     numerical(i, j, k, 2)
                     - pyfluid_amrex::generated::ctt_manufactured_vector_potential_3(
                        x, y, z, momentum_amplitude
                     )
                  );
               });
            }
            for (int component = 0; component < 3; ++component) {
               momentum_max_error = std::max(
                  momentum_max_error, error.norm0(component)
               );
            }
         }
         if (amrex::ParallelDescriptor::IOProcessor()) {
            std::cout << std::setprecision(16)
                      << "ctt_momentum centering="
                      << (vertex_centered ? "vertex" : "cell")
                      << " residual=" << momentum_result.residual
                      << " linf_error=" << momentum_max_error << '\n';
         }
         if (!(momentum_max_error < 5.0e-3)) {
            return_code = 1;
         }
      }

      constexpr amrex::Real coupled_conformal_amplitude = 0.01;
      constexpr amrex::Real coupled_vector_amplitude = 0.005;
      for (const auto coupled_centering : {
              pyfluid_amrex::InitialDataEllipticCentering::cell,
              pyfluid_amrex::InitialDataEllipticCentering::vertex
           }) {
         const bool coupled_vertex_centered =
            coupled_centering == pyfluid_amrex::InitialDataEllipticCentering::vertex;
         amrex::Vector<amrex::BoxArray> coupled_grids(2);
         for (int level = 0; level < 2; ++level) {
            coupled_grids[level] = coupled_vertex_centered
               ? amrex::convert(
                    hierarchy.grids[level], amrex::IntVect::TheNodeVector()
                 )
               : hierarchy.grids[level];
         }
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> coupled_conformal_storage(2);
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> coupled_vector_storage(2);
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> coupled_extrinsic_storage(2);
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> coupled_energy_storage(2);
         amrex::Vector<std::unique_ptr<amrex::MultiFab>> coupled_momentum_storage(2);
         amrex::Vector<amrex::MultiFab*> coupled_conformal(2);
         amrex::Vector<amrex::MultiFab*> coupled_vector(2);
         amrex::Vector<amrex::MultiFab*> coupled_extrinsic(2);
         amrex::Vector<amrex::MultiFab const*> coupled_energy(2);
         amrex::Vector<amrex::MultiFab const*> coupled_momentum(2);
         for (int level = 0; level < 2; ++level) {
            coupled_conformal_storage[level] = std::make_unique<amrex::MultiFab>(
               coupled_grids[level], hierarchy.distribution[level], 1, 1
            );
            coupled_vector_storage[level] = std::make_unique<amrex::MultiFab>(
               coupled_grids[level], hierarchy.distribution[level], 3, 1
            );
            coupled_extrinsic_storage[level] = std::make_unique<amrex::MultiFab>(
               coupled_grids[level],
               hierarchy.distribution[level],
               pyfluid_amrex::CONFORMAL_A_NUM_COMPONENTS,
               0
            );
            coupled_energy_storage[level] = std::make_unique<amrex::MultiFab>(
               coupled_grids[level], hierarchy.distribution[level], 1, 0
            );
            coupled_momentum_storage[level] = std::make_unique<amrex::MultiFab>(
               coupled_grids[level], hierarchy.distribution[level], 3, 0
            );
            coupled_conformal_storage[level]->setVal(1.0);
            coupled_vector_storage[level]->setVal(0.0);
            coupled_extrinsic_storage[level]->setVal(0.0);
            const auto problo = hierarchy.geometry[level].ProbLoArray();
            const auto dx = hierarchy.geometry[level].CellSizeArray();
            const amrex::Real coordinate_offset = coupled_vertex_centered ? 0.0 : 0.5;
            for (amrex::MFIter mfi(*coupled_energy_storage[level]); mfi.isValid(); ++mfi) {
               const amrex::Box box = mfi.validbox();
               const auto energy = coupled_energy_storage[level]->array(mfi);
               const auto momentum = coupled_momentum_storage[level]->array(mfi);
               amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const amrex::Real x = problo[0] + (i + coordinate_offset) * dx[0];
                  const amrex::Real y = problo[1] + (j + coordinate_offset) * dx[1];
                  const amrex::Real z = problo[2] + (k + coordinate_offset) * dx[2];
                  energy(i, j, k, 0) =
                     pyfluid_amrex::generated::ctt_manufactured_coupled_energy_density(
                        x,
                        y,
                        z,
                        coupled_conformal_amplitude,
                        coupled_vector_amplitude
                     );
                  momentum(i, j, k, 0) =
                     pyfluid_amrex::generated::ctt_manufactured_coupled_momentum_density_1(
                        x,
                        y,
                        z,
                        coupled_conformal_amplitude,
                        coupled_vector_amplitude
                     );
                  momentum(i, j, k, 1) =
                     pyfluid_amrex::generated::ctt_manufactured_coupled_momentum_density_2(
                        x,
                        y,
                        z,
                        coupled_conformal_amplitude,
                        coupled_vector_amplitude
                     );
                  momentum(i, j, k, 2) =
                     pyfluid_amrex::generated::ctt_manufactured_coupled_momentum_density_3(
                        x,
                        y,
                        z,
                        coupled_conformal_amplitude,
                        coupled_vector_amplitude
                     );
               });
            }
            coupled_conformal[level] = coupled_conformal_storage[level].get();
            coupled_vector[level] = coupled_vector_storage[level].get();
            coupled_extrinsic[level] = coupled_extrinsic_storage[level].get();
            coupled_energy[level] = coupled_energy_storage[level].get();
            coupled_momentum[level] = coupled_momentum_storage[level].get();
         }
         pyfluid_amrex::CttOptions coupled_options;
         coupled_options.max_coupled_iterations = 80;
         coupled_options.relative_update_tolerance = 1.0e-10;
         coupled_options.relaxation = 0.5;
         auto coupled_momentum_boundary =
            pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
               hierarchy, 3, 1.0, 0.0, 0.0, coupled_centering
            );
         auto coupled_conformal_boundary =
            pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
               hierarchy, 1, 1.0, 0.0, 0.0, coupled_centering
            );
         const pyfluid_amrex::CttResult coupled_result =
            pyfluid_amrex::solve_ctt_constraints(
               hierarchy,
               coupled_conformal,
               coupled_vector,
               coupled_extrinsic,
               coupled_energy,
               coupled_momentum,
               coupled_conformal_boundary,
               coupled_momentum_boundary,
               coupled_options
            );
         amrex::Real coupled_conformal_error = 0.0;
         amrex::Real coupled_vector_error = 0.0;
         for (int level = 0; level < 2; ++level) {
            amrex::MultiFab conformal_error(
               coupled_grids[level], hierarchy.distribution[level], 1, 0
            );
            amrex::MultiFab vector_error(
               coupled_grids[level], hierarchy.distribution[level], 3, 0
            );
            const auto problo = hierarchy.geometry[level].ProbLoArray();
            const auto dx = hierarchy.geometry[level].CellSizeArray();
            const amrex::Real coordinate_offset = coupled_vertex_centered ? 0.0 : 0.5;
            for (amrex::MFIter mfi(conformal_error); mfi.isValid(); ++mfi) {
               const amrex::Box box = mfi.validbox();
               const auto psi = coupled_conformal_storage[level]->const_array(mfi);
               const auto vector = coupled_vector_storage[level]->const_array(mfi);
               const auto psi_error = conformal_error.array(mfi);
               const auto w_error = vector_error.array(mfi);
               amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const amrex::Real x = problo[0] + (i + coordinate_offset) * dx[0];
                  const amrex::Real y = problo[1] + (j + coordinate_offset) * dx[1];
                  const amrex::Real z = problo[2] + (k + coordinate_offset) * dx[2];
                  psi_error(i, j, k, 0) = amrex::Math::abs(
                     psi(i, j, k, 0)
                     - pyfluid_amrex::generated::ctt_manufactured_conformal_factor_3d(
                        x, y, z, coupled_conformal_amplitude
                     )
                  );
                  w_error(i, j, k, 0) = amrex::Math::abs(
                     vector(i, j, k, 0)
                     - pyfluid_amrex::generated::ctt_manufactured_vector_potential_1(
                        x, y, z, coupled_vector_amplitude
                     )
                  );
                  w_error(i, j, k, 1) = amrex::Math::abs(vector(i, j, k, 1));
                  w_error(i, j, k, 2) = amrex::Math::abs(vector(i, j, k, 2));
               });
            }
            coupled_conformal_error = std::max(
               coupled_conformal_error, conformal_error.norm0(0)
            );
            for (int component = 0; component < 3; ++component) {
               coupled_vector_error = std::max(
                  coupled_vector_error, vector_error.norm0(component)
               );
            }
         }
         if (amrex::ParallelDescriptor::IOProcessor()) {
            std::cout << std::setprecision(16)
                      << "ctt_coupled centering="
                      << (coupled_vertex_centered ? "vertex" : "cell")
                      << " converged=" << coupled_result.converged
                      << " iterations=" << coupled_result.coupled_iterations
                      << " relative_update=" << coupled_result.relative_update
                      << " hamiltonian_residual="
                      << coupled_result.hamiltonian_residual
                      << " momentum_residual=" << coupled_result.momentum_residual
                      << " conformal_linf_error=" << coupled_conformal_error
                      << " vector_linf_error=" << coupled_vector_error << '\n';
         }
         if (!coupled_result.converged
             || !(coupled_conformal_error < 5.0e-3)
             || !(coupled_vector_error < 5.0e-3)) {
            return_code = 1;
         }
      }

      constexpr amrex::Real cttk_curvature_background = 1.0;
      constexpr amrex::Real cttk_curvature_amplitude = 0.01;
      constexpr amrex::Real cttk_vector_amplitude = 0.002;
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> cttk_conformal_storage(2);
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> cttk_curvature_storage(2);
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> cttk_vector_storage(2);
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> cttk_extrinsic_storage(2);
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> cttk_energy_storage(2);
      amrex::Vector<std::unique_ptr<amrex::MultiFab>> cttk_momentum_storage(2);
      amrex::Vector<amrex::MultiFab const*> cttk_conformal(2);
      amrex::Vector<amrex::MultiFab*> cttk_curvature(2);
      amrex::Vector<amrex::MultiFab*> cttk_vector(2);
      amrex::Vector<amrex::MultiFab*> cttk_extrinsic(2);
      amrex::Vector<amrex::MultiFab const*> cttk_energy(2);
      amrex::Vector<amrex::MultiFab const*> cttk_momentum(2);
      for (int level = 0; level < 2; ++level) {
         const amrex::BoxArray vertex_grids = amrex::convert(
            hierarchy.grids[level], amrex::IntVect::TheNodeVector()
         );
         cttk_conformal_storage[level] = std::make_unique<amrex::MultiFab>(
            vertex_grids, hierarchy.distribution[level], 1, 1
         );
         cttk_curvature_storage[level] = std::make_unique<amrex::MultiFab>(
            vertex_grids, hierarchy.distribution[level], 1, 1
         );
         cttk_vector_storage[level] = std::make_unique<amrex::MultiFab>(
            vertex_grids, hierarchy.distribution[level], 3, 1
         );
         cttk_extrinsic_storage[level] = std::make_unique<amrex::MultiFab>(
            vertex_grids,
            hierarchy.distribution[level],
            pyfluid_amrex::CONFORMAL_A_NUM_COMPONENTS,
            1
         );
         cttk_energy_storage[level] = std::make_unique<amrex::MultiFab>(
            vertex_grids, hierarchy.distribution[level], 1, 0
         );
         cttk_momentum_storage[level] = std::make_unique<amrex::MultiFab>(
            vertex_grids, hierarchy.distribution[level], 3, 0
         );
         cttk_conformal_storage[level]->setVal(1.0);
         cttk_curvature_storage[level]->setVal(0.0);
         cttk_vector_storage[level]->setVal(0.0);
         cttk_extrinsic_storage[level]->setVal(0.0);
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(*cttk_energy_storage[level]); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto energy = cttk_energy_storage[level]->array(mfi);
            const auto momentum = cttk_momentum_storage[level]->array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real x = problo[0] + i * dx[0];
               const amrex::Real y = problo[1] + j * dx[1];
               const amrex::Real z = problo[2] + k * dx[2];
               energy(i, j, k, 0) =
                  pyfluid_amrex::generated::cttk_manufactured_energy_density(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude,
                     cttk_vector_amplitude
                  );
               momentum(i, j, k, 0) =
                  pyfluid_amrex::generated::cttk_manufactured_momentum_density_1(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude,
                     cttk_vector_amplitude
                  );
               momentum(i, j, k, 1) =
                  pyfluid_amrex::generated::cttk_manufactured_momentum_density_2(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude,
                     cttk_vector_amplitude
                  );
               momentum(i, j, k, 2) =
                  pyfluid_amrex::generated::cttk_manufactured_momentum_density_3(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude,
                     cttk_vector_amplitude
                  );
            });
         }
         cttk_conformal[level] = cttk_conformal_storage[level].get();
         cttk_curvature[level] = cttk_curvature_storage[level].get();
         cttk_vector[level] = cttk_vector_storage[level].get();
         cttk_extrinsic[level] = cttk_extrinsic_storage[level].get();
         cttk_energy[level] = cttk_energy_storage[level].get();
         cttk_momentum[level] = cttk_momentum_storage[level].get();
      }
      auto cttk_vector_boundary =
         pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
            hierarchy,
            3,
            1.0,
            0.0,
            0.0,
            pyfluid_amrex::InitialDataEllipticCentering::vertex
         );
      pyfluid_amrex::CttkOptions cttk_options;
      cttk_options.max_coupled_iterations = 80;
      cttk_options.relative_update_tolerance = 1.0e-9;
      cttk_options.relaxation = 0.5;
      cttk_options.mean_curvature_branch =
         pyfluid_amrex::CttkMeanCurvatureBranch::positive;
      const pyfluid_amrex::CttkResult cttk_result =
         pyfluid_amrex::solve_cttk_constraints(
            hierarchy,
            cttk_conformal,
            cttk_curvature,
            cttk_vector,
            cttk_extrinsic,
            cttk_energy,
            cttk_momentum,
            cttk_vector_boundary,
            cttk_options
         );
      amrex::Real cttk_curvature_error = 0.0;
      amrex::Real cttk_vector_error = 0.0;
      for (int level = 0; level < 2; ++level) {
         const amrex::BoxArray vertex_grids = amrex::convert(
            hierarchy.grids[level], amrex::IntVect::TheNodeVector()
         );
         amrex::MultiFab curvature_error(
            vertex_grids, hierarchy.distribution[level], 1, 0
         );
         amrex::MultiFab vector_error(
            vertex_grids, hierarchy.distribution[level], 3, 0
         );
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(curvature_error); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto curvature = cttk_curvature_storage[level]->const_array(mfi);
            const auto vector = cttk_vector_storage[level]->const_array(mfi);
            const auto curvature_difference = curvature_error.array(mfi);
            const auto vector_difference = vector_error.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real x = problo[0] + i * dx[0];
               const amrex::Real y = problo[1] + j * dx[1];
               const amrex::Real z = problo[2] + k * dx[2];
               curvature_difference(i, j, k, 0) = amrex::Math::abs(
                  curvature(i, j, k, 0)
                  - pyfluid_amrex::generated::cttk_manufactured_mean_curvature(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude
                  )
               );
               vector_difference(i, j, k, 0) = amrex::Math::abs(
                  vector(i, j, k, 0)
                  - pyfluid_amrex::generated::cttk_manufactured_vector_potential_1(
                     x, y, z, cttk_vector_amplitude
                  )
               );
               vector_difference(i, j, k, 1) = amrex::Math::abs(
                  vector(i, j, k, 1)
                  - pyfluid_amrex::generated::cttk_manufactured_vector_potential_2(
                     x, y, z, cttk_vector_amplitude
                  )
               );
               vector_difference(i, j, k, 2) = amrex::Math::abs(
                  vector(i, j, k, 2)
                  - pyfluid_amrex::generated::cttk_manufactured_vector_potential_3(
                     x, y, z, cttk_vector_amplitude
                  )
               );
            });
         }
         cttk_curvature_error = std::max(
            cttk_curvature_error, curvature_error.norm0(0)
         );
         for (int component = 0; component < 3; ++component) {
            cttk_vector_error = std::max(
               cttk_vector_error, vector_error.norm0(component)
            );
         }
      }
      if (amrex::ParallelDescriptor::IOProcessor()) {
         std::cout << std::setprecision(16)
                   << "cttk_coupled converged=" << cttk_result.converged
                   << " iterations=" << cttk_result.coupled_iterations
                   << " relative_update=" << cttk_result.relative_update
                   << " momentum_residual=" << cttk_result.momentum_residual
                   << " min_k_squared="
                   << cttk_result.minimum_mean_curvature_squared
                   << " curvature_linf_error=" << cttk_curvature_error
                   << " vector_linf_error=" << cttk_vector_error << '\n';
      }
      if (!cttk_result.converged
          || !(cttk_curvature_error < 5.0e-3)
          || !(cttk_vector_error < 5.0e-3)) {
         return_code = 1;
      }

      for (int level = 0; level < 2; ++level) {
         cttk_conformal_storage[level]->setVal(1.0);
         cttk_curvature_storage[level]->setVal(0.0);
         cttk_vector_storage[level]->setVal(0.0);
         cttk_extrinsic_storage[level]->setVal(0.0);
         const auto problo = hierarchy.geometry[level].ProbLoArray();
         const auto dx = hierarchy.geometry[level].CellSizeArray();
         for (amrex::MFIter mfi(*cttk_energy_storage[level]); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto energy = cttk_energy_storage[level]->array(mfi);
            const auto momentum = cttk_momentum_storage[level]->array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real x = problo[0] + i * dx[0];
               const amrex::Real y = problo[1] + j * dx[1];
               const amrex::Real z = problo[2] + k * dx[2];
               energy(i, j, k, 0) =
                  pyfluid_amrex::generated::cttk_manufactured_energy_density(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude,
                     0.0
                  );
               momentum(i, j, k, 0) =
                  pyfluid_amrex::generated::cttk_manufactured_momentum_density_1(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude,
                     0.0
                  );
               momentum(i, j, k, 1) =
                  pyfluid_amrex::generated::cttk_manufactured_momentum_density_2(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude,
                     0.0
                  );
               momentum(i, j, k, 2) =
                  pyfluid_amrex::generated::cttk_manufactured_momentum_density_3(
                     x,
                     y,
                     z,
                     cttk_curvature_background,
                     cttk_curvature_amplitude,
                     0.0
                  );
            });
         }
      }
      auto cttk_conformal_boundary =
         pyfluid_amrex::InitialDataEllipticBoundaryConditions::constant(
            hierarchy,
            1,
            1.0,
            0.0,
            0.0,
            pyfluid_amrex::InitialDataEllipticCentering::vertex
         );
      const pyfluid_amrex::CttkResult cttk_hybrid_result =
         pyfluid_amrex::solve_cttk_hybrid_constraints(
            hierarchy,
            amrex::GetVecOfPtrs(cttk_conformal_storage),
            cttk_curvature,
            cttk_vector,
            cttk_extrinsic,
            cttk_energy,
            cttk_momentum,
            cttk_conformal_boundary,
            cttk_vector_boundary,
            cttk_options
         );
      amrex::Real cttk_hybrid_error = 0.0;
      for (int level = 0; level < 2; ++level) {
         amrex::MultiFab conformal_error(
            cttk_conformal_storage[level]->boxArray(),
            hierarchy.distribution[level],
            1,
            0
         );
         amrex::MultiFab::Copy(
            conformal_error, *cttk_conformal_storage[level], 0, 0, 1, 0
         );
         conformal_error.plus(-1.0, 0, 1, 0);
         cttk_hybrid_error = std::max(
            cttk_hybrid_error, conformal_error.norm0(0)
         );
         for (int component = 0; component < 3; ++component) {
            cttk_hybrid_error = std::max(
               cttk_hybrid_error,
               cttk_vector_storage[level]->norm0(component, 0, false)
            );
         }
      }
      if (amrex::ParallelDescriptor::IOProcessor()) {
         std::cout << std::setprecision(16)
                   << "cttk_hybrid converged=" << cttk_hybrid_result.converged
                   << " iterations=" << cttk_hybrid_result.coupled_iterations
                   << " relative_update=" << cttk_hybrid_result.relative_update
                   << " hamiltonian_residual="
                   << cttk_hybrid_result.hamiltonian_residual
                   << " momentum_residual="
                   << cttk_hybrid_result.momentum_residual
                   << " linf_error=" << cttk_hybrid_error << '\n';
      }
      if (!cttk_hybrid_result.converged || !(cttk_hybrid_error < 5.0e-3)) {
         return_code = 1;
      }
      if (!validate_tov_ctt()) {
         return_code = 1;
      }
#endif
   }
   amrex::Finalize();
   return return_code;
}
