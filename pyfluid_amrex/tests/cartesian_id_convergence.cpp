#include "cartesian_id_convergence.hpp"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/initial_data/ctt.hpp"

namespace {

using namespace amrex::literals;

using namespace pyfluid_amrex;

struct ManufacturedPoint {
   amrex::Real conformal = 1.0_rt;
   amrex::Real lapse_conformal = 1.0_rt;
   amrex::GpuArray<amrex::Real, 3> vector{};
   amrex::GpuArray<amrex::Real, 9> extrinsic{};
   amrex::Real energy = 0.0_rt;
   amrex::GpuArray<amrex::Real, 3> momentum{};
   amrex::Real stress = 0.0_rt;
};

// Independently invert the continuum constraints for prescribed smooth fields.
// This is an operator test, not necessarily matter obeying an EoS or an energy
// condition. The same three-dimensional equations are projected by zeroing
// inactive derivatives, never by changing the physical tensor rank.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ManufacturedPoint manufactured_point(
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
   amrex::GpuArray<int, 3> const& logical_to_physical,
   bool xcts
) noexcept
{
   constexpr amrex::Real conformal_amplitude = 0.01_rt;
   const amrex::Real lapse_amplitude = xcts ? -0.008_rt : 0.0_rt;
   const amrex::Real logical_amplitude[3]{0.001_rt, -0.0015_rt, 0.002_rt};
   amrex::Real sine[3]{1.0_rt, 1.0_rt, 1.0_rt};
   amrex::Real cosine[3]{};
   amrex::Real profile = 1.0_rt;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      sine[direction] = std::sin(constants::pi * position[direction]);
      cosine[direction] = std::cos(constants::pi * position[direction]);
      profile *= sine[direction];
   }
   const amrex::Real pi_squared = constants::pi * constants::pi;
   const amrex::Real laplacian = -AMREX_SPACEDIM * pi_squared * profile;
   amrex::Real gradient[3]{};
   amrex::Real hessian[3][3]{};
   amrex::Real amplitude[3]{};
   for (int component = 0; component < 3; ++component) {
      amplitude[logical_to_physical[component]] = logical_amplitude[component];
   }
   for (int first = 0; first < AMREX_SPACEDIM; ++first) {
      const int physical_first = logical_to_physical[first];
      amrex::Real derivative = constants::pi * cosine[first];
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (direction != first) {
            derivative *= sine[direction];
         }
      }
      gradient[physical_first] = derivative;
      for (int second = 0; second < AMREX_SPACEDIM; ++second) {
         amrex::Real derivative2 = -pi_squared * profile;
         if (first != second) {
            derivative2 = pi_squared * cosine[first] * cosine[second];
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
               if (direction != first && direction != second) {
                  derivative2 *= sine[direction];
               }
            }
         }
         hessian[physical_first][logical_to_physical[second]] = derivative2;
      }
   }

   ManufacturedPoint point;
   point.conformal += conformal_amplitude * profile;
   point.lapse_conformal += lapse_amplitude * profile;
   const amrex::Real psi = point.conformal;
   const amrex::Real lapse_psi = point.lapse_conformal;
   const amrex::Real psi4 = std::pow(psi, 4);
   const amrex::Real psi7 = std::pow(psi, 7);
   const amrex::Real scale = xcts ? psi7 / (2.0_rt * lapse_psi) : 1.0_rt;
   amrex::Real divergence = 0.0_rt;
   for (int component = 0; component < 3; ++component) {
      point.vector[component] = amplitude[component] * profile;
      divergence += amplitude[component] * gradient[component];
   }
   amrex::Real squared = 0.0_rt;
   for (int first = 0; first < 3; ++first) {
      amrex::Real grad_div = 0.0_rt;
      amrex::Real weighted_longitudinal = 0.0_rt;
      for (int second = 0; second < 3; ++second) {
         const amrex::Real longitudinal =
            amplitude[first] * gradient[second]
            + amplitude[second] * gradient[first]
            - (first == second ? (2.0_rt / 3.0_rt) * divergence : 0.0_rt);
         const amrex::Real value = scale * longitudinal;
         point.extrinsic[3 * first + second] = value;
         squared += value * value;
         grad_div += amplitude[second] * hessian[first][second];
         weighted_longitudinal += longitudinal * gradient[second]
            * (7.0_rt * conformal_amplitude / psi - lapse_amplitude / lapse_psi);
      }
      const amrex::Real vector_laplacian =
         amplitude[first] * laplacian + grad_div / 3.0_rt;
      // CTT takes contravariant S^i; XCTS takes covariant Eulerian S_i.
      point.momentum[first] = xcts
         ? psi * (vector_laplacian + weighted_longitudinal)
            / (16.0_rt * constants::pi * lapse_psi)
         : vector_laplacian / (8.0_rt * constants::pi * std::pow(psi, 10));
   }
   point.energy = (-conformal_amplitude * laplacian - squared / (8.0_rt * psi7))
      / (2.0_rt * constants::pi * std::pow(psi, 5));
   point.stress = (
      lapse_amplitude * laplacian / lapse_psi
      - 0.875_rt * squared / (psi4 * psi4)
   ) / (4.0_rt * constants::pi * psi4) - 0.5_rt * point.energy;
   return point;
}

struct SolveErrors {
   // psi, alpha*psi, physical vector, physical trace-free tensor.
   std::array<amrex::Real, 4> linf{};
   amrex::Real trace = 0.0_rt;
   bool converged = false;
};

SolveErrors solve_manufactured(
   int cells,
   InitialDataEllipticCentering centering,
   bool xcts,
   amrex::GpuArray<int, 3> const& logical_to_physical
)
{
   const amrex::Box domain(
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(cells - 1, cells - 1, cells - 1))
   );
   const amrex::RealBox physical_domain(
      AMREX_D_DECL(0.0_rt, 0.0_rt, 0.0_rt), AMREX_D_DECL(1.0_rt, 1.0_rt, 1.0_rt)
   );
   InitialDataEllipticHierarchy hierarchy;
   hierarchy.geometry.emplace_back(
      domain, physical_domain, amrex::CoordSys::cartesian,
      amrex::Array<int, AMREX_SPACEDIM>{AMREX_D_DECL(0, 0, 0)}
   );
   hierarchy.grids.emplace_back(domain);
   hierarchy.grids[0].maxSize(cells / 2);
   hierarchy.distribution.emplace_back(hierarchy.grids[0]);
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      hierarchy.cartesian_physical_axes[direction] = logical_to_physical[direction];
   }
   const bool nodal = centering == InitialDataEllipticCentering::vertex;
   const amrex::BoxArray grids = nodal
      ? amrex::convert(hierarchy.grids[0], amrex::IntVect::TheNodeVector())
      : hierarchy.grids[0];
   const auto& distribution = hierarchy.distribution[0];
   amrex::MultiFab conformal(grids, distribution, 1, 1);
   amrex::MultiFab lapse_conformal(grids, distribution, 1, 1);
   amrex::MultiFab vector(grids, distribution, 3, 1);
   amrex::MultiFab extrinsic(grids, distribution, CONFORMAL_A_NUM_COMPONENTS, 0);
   amrex::MultiFab energy(grids, distribution, 1, 0);
   amrex::MultiFab momentum(grids, distribution, 3, 0);
   amrex::MultiFab stress(grids, distribution, 1, 0);
   conformal.setVal(1.0_rt);
   lapse_conformal.setVal(1.0_rt);
   vector.setVal(0.0_rt);
   extrinsic.setVal(0.0_rt);
   const amrex::Real offset = nodal ? 0.0_rt : 0.5_rt;
   const amrex::Real spacing = 1.0_rt / cells;
   for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
      const auto energy_values = energy.array(mfi);
      const auto momentum_values = momentum.array(mfi);
      const auto stress_values = stress.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const auto point = manufactured_point(
            {AMREX_D_DECL((i + offset) * spacing, (j + offset) * spacing,
                          (k + offset) * spacing)},
            logical_to_physical, xcts
         );
         energy_values(i, j, k) = point.energy;
         stress_values(i, j, k) = point.stress;
         for (int component = 0; component < 3; ++component) {
            momentum_values(i, j, k, component) = point.momentum[component];
         }
      });
   }
   const auto scalar_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy, 1, 1.0_rt, 0.0_rt, 0.0_rt, centering
   );
   const auto vector_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy, 3, 1.0_rt, 0.0_rt, 0.0_rt, centering
   );
   SolveErrors errors;
   int iterations = 0;
   if (xcts) {
      XctsOptions options;
      options.relative_update_tolerance = 1.0e-11_rt;
      const auto result = solve_xcts_constraints(
         hierarchy, {&conformal}, {&lapse_conformal}, {&vector}, {&extrinsic},
         {&energy}, {&momentum}, {&stress},
         scalar_boundary, scalar_boundary, vector_boundary, options
      );
      errors.converged = result.converged;
      iterations = result.coupled_iterations;
   } else {
      CttOptions options;
      options.relative_update_tolerance = 1.0e-11_rt;
      const auto result = solve_ctt_constraints(
         hierarchy, {&conformal}, {&vector}, {&extrinsic}, {&energy}, {&momentum},
         scalar_boundary, vector_boundary, options
      );
      errors.converged = result.converged;
      iterations = result.coupled_iterations;
   }
   amrex::MultiFab difference(grids, distribution, 5, 0);
   for (amrex::MFIter mfi(difference); mfi.isValid(); ++mfi) {
      const auto psi = conformal.const_array(mfi);
      const auto lapse_psi = lapse_conformal.const_array(mfi);
      const auto vector_values = vector.const_array(mfi);
      const auto tensor_values = extrinsic.const_array(mfi);
      const auto delta = difference.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const auto point = manufactured_point(
            {AMREX_D_DECL((i + offset) * spacing, (j + offset) * spacing,
                          (k + offset) * spacing)},
            logical_to_physical, xcts
         );
         delta(i, j, k, 0) = std::abs(psi(i, j, k) - point.conformal);
         delta(i, j, k, 1) = std::abs(lapse_psi(i, j, k) - point.lapse_conformal);
         amrex::Real vector_error = 0.0_rt;
         for (int component = 0; component < 3; ++component) {
            vector_error = amrex::max(vector_error, std::abs(
               vector_values(i, j, k, component) - point.vector[component]
            ));
         }
         delta(i, j, k, 2) = vector_error;
         const int components[6]{CONFORMAL_A11, CONFORMAL_A12, CONFORMAL_A13,
                                 CONFORMAL_A22, CONFORMAL_A23, CONFORMAL_A33};
         const int matrix_entries[6]{0, 1, 2, 4, 5, 8};
         amrex::Real tensor_error = 0.0_rt;
         for (int entry = 0; entry < 6; ++entry) {
            tensor_error = amrex::max(tensor_error, std::abs(
               tensor_values(i, j, k, components[entry])
               - point.extrinsic[matrix_entries[entry]]
            ));
         }
         delta(i, j, k, 3) = tensor_error;
         delta(i, j, k, 4) = std::abs(
            tensor_values(i, j, k, CONFORMAL_A11)
            + tensor_values(i, j, k, CONFORMAL_A22)
            + tensor_values(i, j, k, CONFORMAL_A33)
         );
      });
   }
   for (int field = 0; field < 4; ++field) {
      errors.linf[field] = difference.norm0(field);
   }
   errors.trace = difference.norm0(4);
   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::cout << std::setprecision(16)
                << "cartesian_matter dimension=" << AMREX_SPACEDIM
                << " method=" << (xcts ? "xcts" : "ctt")
                << " centering=" << (nodal ? "vertex" : "cell")
                << " cells=" << cells << " axes=";
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         std::cout << "xyz"[logical_to_physical[direction]];
      }
      std::cout << " converged=" << errors.converged
                << " iterations=" << iterations
                << " psi_error=" << errors.linf[0]
                << " lapse_error=" << errors.linf[1]
                << " vector_error=" << errors.linf[2]
                << " tensor_error=" << errors.linf[3]
                << " trace_error=" << errors.trace << std::endl;
   }
   return errors;
}

} // namespace

bool validate_cartesian_id_matter_convergence()
{
   bool valid = true;
   for (const bool xcts : {false, true}) {
      for (const auto centering : {InitialDataEllipticCentering::cell,
                                  InitialDataEllipticCentering::vertex}) {
         const auto coarse = solve_manufactured(16, centering, xcts, {0, 1, 2});
         const auto fine = solve_manufactured(32, centering, xcts, {0, 1, 2});
         bool case_valid = coarse.converged && fine.converged;
         amrex::Real minimum_order = 100.0_rt;
         for (int field = 0; field < 4; ++field) {
            if (field == 1 && !xcts) {
               continue;
            }
            const amrex::Real order = std::log2(coarse.linf[field] / fine.linf[field]);
            minimum_order = std::min(minimum_order, order);
            case_valid = case_valid && std::isfinite(order) && order > 1.7_rt;
         }
         case_valid = case_valid && fine.trace < 1.0e-12_rt
            && *std::max_element(fine.linf.begin(), fine.linf.end()) < 5.0e-4_rt;
         amrex::Real axis_error = 0.0_rt;
#if (AMREX_SPACEDIM < 3)
#if (AMREX_SPACEDIM == 1)
         const std::array<amrex::GpuArray<int, 3>, 2> alternate_axes{{
            {1, 2, 0}, {2, 0, 1}
         }};
#else
         const std::array<amrex::GpuArray<int, 3>, 2> alternate_axes{{
            {1, 2, 0}, {0, 2, 1}
         }};
#endif
         for (const auto axes : alternate_axes) {
            const auto rotated = solve_manufactured(32, centering, xcts, axes);
            case_valid = case_valid && rotated.converged && rotated.trace < 1.0e-12_rt;
            for (int field = 0; field < 4; ++field) {
               axis_error = std::max(axis_error,
                  std::abs(rotated.linf[field] - fine.linf[field]));
            }
         }
         case_valid = case_valid && axis_error < 1.0e-10_rt;
#endif
         valid = valid && case_valid;
         if (amrex::ParallelDescriptor::IOProcessor()) {
            std::cout << "cartesian_matter_convergence dimension=" << AMREX_SPACEDIM
                      << " method=" << (xcts ? "xcts" : "ctt")
                      << " centering="
                      << (centering == InitialDataEllipticCentering::vertex
                          ? "vertex" : "cell")
                      << " minimum_order=" << minimum_order
                      << " axis_error=" << axis_error
                      << " valid=" << case_valid << std::endl;
         }
      }
   }
   return valid;
}
