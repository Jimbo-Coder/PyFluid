#include "pyfluid_amrex/initial_data/elliptic.hpp"

#include <AMReX.H>
#include <AMReX_BCUtil.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_FabArrayUtility.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "pyfluid_amrex/constants.hpp"
#include "initial_data_elliptic_internal.hpp"

namespace pyfluid_amrex {

using detail::configure_mlmg;
using detail::elliptic_result;
using detail::lower_boundary_conditions;
using detail::LoweredBoundaryConditions;
using detail::resolved_boundary_type;
using detail::set_boundary_conditions;
using detail::validate_elliptic_solve;

namespace {

enum class AxisymmetricSphericalMode {
   scalar,
   toroidal_vector,
};

InitialDataEllipticResult solve_axisymmetric_spherical_separable(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options,
   AxisymmetricSphericalMode mode
);

InitialDataEllipticResult solve_initial_data_spherical_mode(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options,
   amrex::Real angular_spin_squared
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy, solution, rhs, boundary, options, angular_spin_squared
   );
   amrex::Abort(
      "the full spherical elliptic operator requires three dimensions"
   );
   return {};
#else
   if (!(angular_spin_squared >= 0.0)
       || !std::isfinite(angular_spin_squared)) {
      amrex::Abort("invalid full spherical angular spin weight");
   }
   const int num_levels = validate_elliptic_solve(
      hierarchy, solution, rhs, boundary, options, 1
   );
   if (boundary.centering() != InitialDataEllipticCentering::cell) {
      amrex::Abort(
         "the full spherical elliptic operator requires cell-centered fields"
      );
   }
   auto near = [](amrex::Real first, amrex::Real second) {
      const amrex::Real scale = std::max({
         amrex::Real(1.0), std::abs(first), std::abs(second)
      });
      return std::abs(first - second)
         <= 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
   };
   const amrex::Real half_pi = 0.5 * constants::pi;
   for (int level = 0; level < num_levels; ++level) {
      const amrex::Geometry& geometry = hierarchy.geometry[level];
      const auto problo = geometry.ProbLoArray();
      const auto probhi = geometry.ProbHiArray();
      const amrex::Vector<amrex::Real> radial_faces =
         hierarchy.coordinate_faces(level, 0);
      const bool theta_domain =
         (near(problo[1], 0.0)
          && (near(probhi[1], half_pi) || near(probhi[1], constants::pi)))
         || (near(problo[1], half_pi) && near(probhi[1], constants::pi));
      if (geometry.isPeriodic(0) || geometry.isPeriodic(1)
          || !geometry.isPeriodic(2)
          || !near(radial_faces.front(), problo[0])
          || !near(radial_faces.back(), probhi[0])
          || !(radial_faces.front() >= 0.0)
          || !(radial_faces.back() > radial_faces.front())
          || !theta_domain
          || !near(probhi[2] - problo[2], 2.0 * constants::pi)) {
         amrex::Abort(
            "full spherical elliptic coordinates require physical ordered "
            "(r,theta,phi), nonperiodic r/theta, a full periodic 2pi phi "
            "interval, and theta in [0,pi/2], [pi/2,pi], or [0,pi]"
         );
      }
      const amrex::Vector<amrex::Real> theta_faces =
         hierarchy.coordinate_faces(level, 1);
      const auto dx = geometry.CellSizeArray();
      for (int face = 0; face < static_cast<int>(theta_faces.size()); ++face) {
         if (!near(theta_faces[face], problo[1] + face * dx[1])) {
            amrex::Abort(
               "the full spherical elliptic operator currently requires uniform theta"
            );
         }
      }
   }

   const bool radial_origin = near(
      hierarchy.coordinate_faces(0, 0).front(), 0.0
   );
   const auto first_problo = hierarchy.geometry.front().ProbLoArray();
   const auto first_probhi = hierarchy.geometry.front().ProbHiArray();
   const bool toroidal = angular_spin_squared > 0.0;
   const InitialDataEllipticBoundaryType radial_inner_type =
      resolved_boundary_type(boundary, 0, 0, false);
   const InitialDataEllipticBoundaryType radial_outer_type =
      resolved_boundary_type(boundary, 0, 0, true);
   const bool valid_inner = radial_origin
      ? radial_inner_type == (
           toroidal
              ? InitialDataEllipticBoundaryType::dirichlet
              : InitialDataEllipticBoundaryType::neumann
        )
      : radial_inner_type == InitialDataEllipticBoundaryType::dirichlet
         || radial_inner_type == InitialDataEllipticBoundaryType::neumann
         || radial_inner_type == InitialDataEllipticBoundaryType::robin;
   const InitialDataEllipticBoundaryType lower_theta_type =
      toroidal && near(first_problo[1], 0.0)
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   const InitialDataEllipticBoundaryType upper_theta_type =
      toroidal && near(first_probhi[1], constants::pi)
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   if (!valid_inner
       || (radial_outer_type != InitialDataEllipticBoundaryType::dirichlet
           && radial_outer_type != InitialDataEllipticBoundaryType::robin)
       || resolved_boundary_type(boundary, 0, 1, false)
          != lower_theta_type
       || resolved_boundary_type(boundary, 0, 1, true)
          != upper_theta_type
       || resolved_boundary_type(boundary, 0, 2, false)
          != InitialDataEllipticBoundaryType::periodic
       || resolved_boundary_type(boundary, 0, 2, true)
          != InitialDataEllipticBoundaryType::periodic) {
      amrex::Abort(
         "invalid origin/shell, axis/equator, radial outer, or phi boundary "
         "for the full spherical scalar operator"
      );
   }

   auto logical_boundary = boundary.component(0);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> weighted_rhs_storage(
      num_levels
   );
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> a_coefficient(num_levels);
   amrex::Vector<amrex::MultiFab const*> weighted_rhs(num_levels);
   amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>
      b_coefficient(num_levels);

   for (int level = 0; level < num_levels; ++level) {
      const amrex::Geometry& geometry = hierarchy.geometry[level];
      const amrex::Box domain = geometry.Domain();
      const int radial_begin = domain.smallEnd(0);
      const int angular_begin = domain.smallEnd(1);
      const auto problo = geometry.ProbLoArray();
      const auto dx = geometry.CellSizeArray();
      const InitialDataCoordinateLine radial = hierarchy.coordinate_line(
         level, 0
      );
      const int radial_cells = static_cast<int>(radial.centers.size());
      std::vector<amrex::Real> radial_cell_jacobian(radial_cells);
      std::vector<amrex::Real> radial_face_jacobian(radial_cells + 1);
      for (int cell = 0; cell < radial_cells; ++cell) {
         radial_cell_jacobian[cell] = radial.widths[cell] / dx[0];
      }
      radial_face_jacobian.front() = radial_cell_jacobian.front();
      radial_face_jacobian.back() = radial_cell_jacobian.back();
      for (int face = 1; face < radial_cells; ++face) {
         radial_face_jacobian[face] = (
            radial.centers[face] - radial.centers[face - 1]
         ) / dx[0];
      }
      for (amrex::Real jacobian : radial_cell_jacobian) {
         if (!(jacobian > 0.0) || !std::isfinite(jacobian)) {
            amrex::Abort("invalid mapped spherical radial cell Jacobian");
         }
      }
      for (amrex::Real jacobian : radial_face_jacobian) {
         if (!(jacobian > 0.0) || !std::isfinite(jacobian)) {
            amrex::Abort("invalid mapped spherical radial face Jacobian");
         }
      }

      amrex::Gpu::DeviceVector<amrex::Real> device_radial_faces(
         radial.faces.size()
      );
      amrex::Gpu::DeviceVector<amrex::Real> device_radial_centers(
         radial.centers.size()
      );
      amrex::Gpu::DeviceVector<amrex::Real> device_cell_jacobian(
         radial_cell_jacobian.size()
      );
      amrex::Gpu::DeviceVector<amrex::Real> device_face_jacobian(
         radial_face_jacobian.size()
      );
      amrex::Gpu::copy(
         amrex::Gpu::hostToDevice,
         radial.faces.begin(), radial.faces.end(),
         device_radial_faces.begin()
      );
      amrex::Gpu::copy(
         amrex::Gpu::hostToDevice,
         radial.centers.begin(), radial.centers.end(),
         device_radial_centers.begin()
      );
      amrex::Gpu::copy(
         amrex::Gpu::hostToDevice,
         radial_cell_jacobian.begin(), radial_cell_jacobian.end(),
         device_cell_jacobian.begin()
      );
      amrex::Gpu::copy(
         amrex::Gpu::hostToDevice,
         radial_face_jacobian.begin(), radial_face_jacobian.end(),
         device_face_jacobian.begin()
      );
      const amrex::Real* radial_face = device_radial_faces.data();
      const amrex::Real* radial_center = device_radial_centers.data();
      const amrex::Real* cell_jacobian = device_cell_jacobian.data();
      const amrex::Real* face_jacobian = device_face_jacobian.data();

      weighted_rhs_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 0
      );
      a_coefficient[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 0
      );
      weighted_rhs[level] = weighted_rhs_storage[level].get();
      for (amrex::MFIter mfi(*weighted_rhs_storage[level]);
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto physical_rhs = rhs[level]->const_array(mfi);
         const auto weighted = weighted_rhs_storage[level]->array(mfi);
         const auto zeroth_order = a_coefficient[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const int radial_index = i - radial_begin;
            const amrex::Real radius = radial_center[radial_index];
            const amrex::Real theta = problo[1]
               + (j - angular_begin + 0.5) * dx[1];
            weighted(i, j, k, 0) = physical_rhs(i, j, k, 0)
               * cell_jacobian[radial_index] * radius * radius
               * std::sin(theta);
            zeroth_order(i, j, k, 0) = cell_jacobian[radial_index]
               / std::sin(theta);
         });
      }

      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         b_coefficient[level][direction].define(
            amrex::convert(
               hierarchy.grids[level],
               amrex::IntVect::TheDimensionVector(direction)
            ),
            hierarchy.distribution[level],
            1,
            0
         );
      }
      for (amrex::MFIter mfi(b_coefficient[level][0]);
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto coefficient = b_coefficient[level][0].array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const int radial_index = i - radial_begin;
            const amrex::Real theta = problo[1]
               + (j - angular_begin + 0.5) * dx[1];
            const amrex::Real radius = radial_face[radial_index];
            coefficient(i, j, k, 0) = radius * radius * std::sin(theta)
               / face_jacobian[radial_index];
         });
      }
      for (amrex::MFIter mfi(b_coefficient[level][1]);
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto coefficient = b_coefficient[level][1].array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const int radial_index = i - radial_begin;
            const amrex::Real theta_face = problo[1]
               + (j - angular_begin) * dx[1];
            coefficient(i, j, k, 0) = cell_jacobian[radial_index]
               * std::sin(theta_face);
         });
      }
      for (amrex::MFIter mfi(b_coefficient[level][2]);
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto coefficient = b_coefficient[level][2].array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const int radial_index = i - radial_begin;
            const amrex::Real theta = problo[1]
               + (j - angular_begin + 0.5) * dx[1];
            coefficient(i, j, k, 0) = cell_jacobian[radial_index]
               / std::sin(theta);
         });
      }

      const amrex::Dim3 domain_lo = amrex::lbound(domain);
      const amrex::Dim3 domain_hi = amrex::ubound(domain);
      const amrex::Real inverse_inner_jacobian =
         1.0 / radial_face_jacobian.front();
      const amrex::Real inverse_outer_jacobian =
         1.0 / radial_face_jacobian.back();
      for (amrex::MFIter mfi(logical_boundary.coefficient_b(level));
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.fabbox();
         const auto coefficient_b =
            logical_boundary.coefficient_b(level).array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            if (i < domain_lo.x) {
               coefficient_b(i, j, k, 0) *= inverse_inner_jacobian;
            } else if (i > domain_hi.x) {
               coefficient_b(i, j, k, 0) *= inverse_outer_jacobian;
            }
         });
      }
      amrex::Gpu::streamSynchronize();
   }

   amrex::LPInfo info;
   // A one-cell periodic phi slab already has identically cancelling
   // azimuthal fluxes. Keep it in the ordinary operator instead of AMReX's
   // hidden-direction path: polar spherical grids need semicoarsening, and
   // AMReX intentionally forbids combining those two LPInfo modes.
   info.setSemicoarsening(true).setMaxSemicoarseningLevel(8);
   amrex::MLABecLaplacian poisson(
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      info
   );
   poisson.setMaxOrder(options.boundary_interpolation_order);
   poisson.setScalars(-angular_spin_squared, -1.0);
   const LoweredBoundaryConditions lowered = lower_boundary_conditions(
      hierarchy, logical_boundary
   );
   for (int level = 0; level < num_levels; ++level) {
      poisson.setACoeffs(level, *a_coefficient[level]);
      poisson.setBCoeffs(
         level, amrex::GetArrOfConstPtrs(b_coefficient[level])
      );
   }
   set_boundary_conditions(poisson, logical_boundary, lowered);

   amrex::MLMG mlmg(poisson);
   configure_mlmg(mlmg, options);
   return elliptic_result(mlmg.solve(
      solution,
      weighted_rhs,
      options.relative_tolerance,
      options.absolute_tolerance
   ));
#endif
}

} // namespace

InitialDataEllipticResult solve_initial_data_spherical_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   return solve_initial_data_spherical_mode(
      hierarchy, solution, rhs, boundary, options, 0.0
   );
}

InitialDataEllipticResult solve_initial_data_spherical_axisymmetric_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   return solve_axisymmetric_spherical_separable(
      hierarchy,
      solution,
      rhs,
      boundary,
      options,
      AxisymmetricSphericalMode::scalar
   );
}

InitialDataEllipticResult
solve_initial_data_spherical_toroidal_vector_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_solution,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   return solve_initial_data_spherical_mode(
      hierarchy,
      azimuthal_solution,
      azimuthal_source,
      boundary,
      options,
      1.0
   );
}

InitialDataEllipticResult
solve_initial_data_spherical_axisymmetric_toroidal_vector_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_solution,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   return solve_axisymmetric_spherical_separable(
      hierarchy,
      azimuthal_solution,
      azimuthal_source,
      boundary,
      options,
      AxisymmetricSphericalMode::toroidal_vector
   );
}

#if (AMREX_SPACEDIM >= 2)
namespace {

struct SymmetricTridiagonalEigensystem {
   std::vector<amrex::Real> eigenvalues;
   // Mode-major eigenvectors: vectors[mode * size + row].
   std::vector<amrex::Real> vectors;
};

struct AxisymmetricAngularDecomposition {
   std::vector<amrex::Real> weights;
   std::vector<amrex::Real> rhs_projection_vectors;
   std::vector<amrex::Real> solution_reconstruction_vectors;
   SymmetricTridiagonalEigensystem eigensystem;
};

SymmetricTridiagonalEigensystem symmetric_tridiagonal_eigensystem(
   std::vector<amrex::Real> diagonal,
   std::vector<amrex::Real> off_diagonal
)
{
   // Implicit QL iteration keeps the self-adjoint angular solve independent
   // of an optional platform LAPACK while retaining an orthonormal basis.
   const int size = static_cast<int>(diagonal.size());
   if (size < 1 || static_cast<int>(off_diagonal.size()) != size) {
      amrex::Abort("invalid symmetric tridiagonal eigensystem");
   }
   std::vector<amrex::Real> vectors(
      static_cast<std::size_t>(size) * size, 0.0
   );
   for (int index = 0; index < size; ++index) {
      vectors[static_cast<std::size_t>(index) * size + index] = 1.0;
   }

   off_diagonal[size - 1] = 0.0;
   amrex::Real accumulated_shift = 0.0;
   amrex::Real running_scale = 0.0;
   const amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
   for (int lower = 0; lower < size; ++lower) {
      running_scale = std::max(
         running_scale,
         std::abs(diagonal[lower]) + std::abs(off_diagonal[lower])
      );
      int upper = lower;
      while (upper < size
             && std::abs(off_diagonal[upper]) > epsilon * running_scale) {
         ++upper;
      }
      if (upper > lower) {
         int iterations = 0;
         do {
            if (++iterations > 64) {
               amrex::Abort(
                  "symmetric tridiagonal eigensystem did not converge"
               );
            }
            amrex::Real pivot = diagonal[lower];
            amrex::Real ratio = (
               diagonal[lower + 1] - pivot
            ) / (2.0 * off_diagonal[lower]);
            amrex::Real radius = std::hypot(ratio, 1.0);
            if (ratio < 0.0) {
               radius = -radius;
            }
            diagonal[lower] = off_diagonal[lower] / (ratio + radius);
            diagonal[lower + 1] = off_diagonal[lower] * (ratio + radius);
            const amrex::Real next_diagonal = diagonal[lower + 1];
            const amrex::Real shift = pivot - diagonal[lower];
            for (int index = lower + 2; index < size; ++index) {
               diagonal[index] -= shift;
            }
            accumulated_shift += shift;

            pivot = diagonal[upper];
            amrex::Real cosine = 1.0;
            amrex::Real previous_cosine = cosine;
            amrex::Real previous_previous_cosine = cosine;
            amrex::Real sine = 0.0;
            amrex::Real previous_sine = 0.0;
            const amrex::Real next_off_diagonal = off_diagonal[lower + 1];
            for (int index = upper - 1; index >= lower; --index) {
               previous_previous_cosine = previous_cosine;
               previous_cosine = cosine;
               previous_sine = sine;
               const amrex::Real product = cosine * off_diagonal[index];
               const amrex::Real diagonal_product = cosine * pivot;
               radius = std::hypot(pivot, off_diagonal[index]);
               off_diagonal[index + 1] = sine * radius;
               sine = off_diagonal[index] / radius;
               cosine = pivot / radius;
               pivot = cosine * diagonal[index] - sine * product;
               diagonal[index + 1] = diagonal_product + sine * (
                  cosine * product + sine * diagonal[index]
               );
               for (int row = 0; row < size; ++row) {
                  const std::size_t upper_offset =
                     static_cast<std::size_t>(row) * size + index + 1;
                  const std::size_t lower_offset = upper_offset - 1;
                  const amrex::Real upper_value = vectors[upper_offset];
                  vectors[upper_offset] = sine * vectors[lower_offset]
                     + cosine * upper_value;
                  vectors[lower_offset] = cosine * vectors[lower_offset]
                     - sine * upper_value;
               }
            }
            pivot = -sine * previous_sine * previous_previous_cosine
               * next_off_diagonal * off_diagonal[lower] / next_diagonal;
            off_diagonal[lower] = sine * pivot;
            diagonal[lower] = cosine * pivot;
         } while (
            std::abs(off_diagonal[lower]) > epsilon * running_scale
         );
      }
      diagonal[lower] += accumulated_shift;
      off_diagonal[lower] = 0.0;
   }

   for (int first = 0; first < size - 1; ++first) {
      int smallest = first;
      for (int second = first + 1; second < size; ++second) {
         if (diagonal[second] < diagonal[smallest]) {
            smallest = second;
         }
      }
      if (smallest != first) {
         std::swap(diagonal[first], diagonal[smallest]);
         for (int row = 0; row < size; ++row) {
            std::swap(
               vectors[static_cast<std::size_t>(row) * size + first],
               vectors[static_cast<std::size_t>(row) * size + smallest]
            );
         }
      }
   }
   std::vector<amrex::Real> mode_vectors(vectors.size());
   for (int mode = 0; mode < size; ++mode) {
      for (int row = 0; row < size; ++row) {
         mode_vectors[static_cast<std::size_t>(mode) * size + row] =
            vectors[static_cast<std::size_t>(row) * size + mode];
      }
   }
   return {std::move(diagonal), std::move(mode_vectors)};
}

std::shared_ptr<AxisymmetricAngularDecomposition const>
axisymmetric_angular_decomposition(
   int angular_cells,
   amrex::Real angular_lower,
   amrex::Real angular_spacing,
   AxisymmetricSphericalMode mode
)
{
   struct CacheEntry {
      int angular_cells = -1;
      amrex::Real angular_lower = 0.0;
      amrex::Real angular_spacing = 0.0;
      std::shared_ptr<AxisymmetricAngularDecomposition const> decomposition;
   };
   static std::array<CacheEntry, 2> cache;
   static std::mutex cache_mutex;
   const int cache_index = mode == AxisymmetricSphericalMode::scalar ? 0 : 1;
   std::lock_guard<std::mutex> guard(cache_mutex);
   CacheEntry& entry = cache[cache_index];
   if (entry.decomposition && entry.angular_cells == angular_cells
       && entry.angular_lower == angular_lower
       && entry.angular_spacing == angular_spacing) {
      return entry.decomposition;
   }

   auto decomposition = std::make_shared<AxisymmetricAngularDecomposition>();
   decomposition->weights.resize(angular_cells, 0.0);
   std::vector<amrex::Real> angular_diagonal(angular_cells, 0.0);
   std::vector<amrex::Real> angular_off_diagonal(angular_cells, 0.0);
   const amrex::Real angular_scale = 1.0 / (
      angular_spacing * angular_spacing
   );
   const amrex::Real angular_spin_squared =
      mode == AxisymmetricSphericalMode::toroidal_vector ? 1.0 : 0.0;
   for (int angular_index = 0;
        angular_index < angular_cells;
        ++angular_index) {
      const amrex::Real theta = angular_lower
         + (angular_index + 0.5) * angular_spacing;
      decomposition->weights[angular_index] = std::sqrt(std::sin(theta));
      if (!(decomposition->weights[angular_index] > 0.0)) {
         amrex::Abort("invalid spherical angular cell weight");
      }
      angular_diagonal[angular_index] -= angular_spin_squared / (
         decomposition->weights[angular_index]
         * decomposition->weights[angular_index]
         * decomposition->weights[angular_index]
         * decomposition->weights[angular_index]
      );
   }
   for (int angular_index = 0;
        angular_index + 1 < angular_cells;
        ++angular_index) {
      const amrex::Real face_theta = angular_lower
         + (angular_index + 1) * angular_spacing;
      const amrex::Real face_coefficient =
         std::sin(face_theta) * angular_scale;
      angular_off_diagonal[angular_index] = face_coefficient / (
         decomposition->weights[angular_index]
         * decomposition->weights[angular_index + 1]
      );
      angular_diagonal[angular_index] -= face_coefficient / (
         decomposition->weights[angular_index]
         * decomposition->weights[angular_index]
      );
      angular_diagonal[angular_index + 1] -= face_coefficient / (
         decomposition->weights[angular_index + 1]
         * decomposition->weights[angular_index + 1]
      );
   }
   decomposition->eigensystem = symmetric_tridiagonal_eigensystem(
      std::move(angular_diagonal), std::move(angular_off_diagonal)
   );
   const std::size_t vector_values = static_cast<std::size_t>(angular_cells)
      * angular_cells;
   decomposition->rhs_projection_vectors.resize(vector_values);
   decomposition->solution_reconstruction_vectors.resize(vector_values);
   for (int mode_index = 0; mode_index < angular_cells; ++mode_index) {
      for (int angular_index = 0;
           angular_index < angular_cells;
           ++angular_index) {
         const std::size_t offset =
            static_cast<std::size_t>(mode_index) * angular_cells
            + angular_index;
         const amrex::Real eigenvector =
            decomposition->eigensystem.vectors[offset];
         decomposition->rhs_projection_vectors[offset] = eigenvector
            * decomposition->weights[angular_index];
         decomposition->solution_reconstruction_vectors[offset] = eigenvector
            / decomposition->weights[angular_index];
      }
   }
   entry = {
      angular_cells,
      angular_lower,
      angular_spacing,
      decomposition,
   };
   return decomposition;
}

amrex::MultiFab host_copy(amrex::MultiFab const& source)
{
   amrex::MultiFab result(
      source.boxArray(), source.DistributionMap(), source.nComp(),
      source.nGrowVect(),
      amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
   );
   amrex::dtoh_memcpy(result, source);
   amrex::Gpu::streamSynchronize();
   return result;
}

void copy_host_to_device(
   amrex::MultiFab& destination,
   amrex::MultiFab const& source
)
{
   amrex::htod_memcpy(destination, source);
   amrex::Gpu::streamSynchronize();
}

} // namespace
#endif

namespace {

InitialDataEllipticResult solve_axisymmetric_spherical_separable(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options,
   AxisymmetricSphericalMode mode
)
{
#if (AMREX_SPACEDIM < 2)
   amrex::ignore_unused(hierarchy, solution, rhs, boundary, options, mode);
   amrex::Abort(
      "the axisymmetric spherical elliptic operator requires r-theta storage"
   );
   return {};
#else
   const int num_levels = validate_elliptic_solve(
      hierarchy, solution, rhs, boundary, options, 1
   );
   if (boundary.centering() != InitialDataEllipticCentering::cell) {
      amrex::Abort(
         "the axisymmetric spherical elliptic operator requires cell-centered fields"
      );
   }

   auto near = [](amrex::Real first, amrex::Real second) {
      const amrex::Real scale = std::max({
         amrex::Real(1.0), std::abs(first), std::abs(second)
      });
      return std::abs(first - second)
         <= 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
   };
   if (num_levels != 1) {
      amrex::Abort(
         "the tensor-product spherical elliptic solver requires one uniform level"
      );
   }
   amrex::Geometry const& geometry = hierarchy.geometry[0];
   const auto problo = geometry.ProbLoArray();
   const auto probhi = geometry.ProbHiArray();
   const auto dx = geometry.CellSizeArray();
   const amrex::Box domain = geometry.Domain();
   const amrex::Vector<amrex::Real> radial_faces =
      hierarchy.coordinate_faces(0, 0);
   const amrex::Vector<amrex::Real> angular_faces =
      hierarchy.coordinate_faces(0, 1);
   const amrex::Real half_pi = 0.5 * constants::pi;
   const bool lower_axis = near(problo[1], 0.0);
   const bool upper_axis = near(probhi[1], constants::pi);
   const bool standard_theta_domain =
      (lower_axis
       && (near(probhi[1], half_pi) || near(probhi[1], constants::pi)))
      || (near(problo[1], half_pi) && near(probhi[1], constants::pi));
   const bool radial_origin = near(radial_faces.front(), 0.0);
   if ((geometry.Coord() != amrex::CoordSys::cartesian
        && geometry.Coord() != amrex::CoordSys::SPHERICAL)
       || geometry.isPeriodic(0) || geometry.isPeriodic(1)
       || !near(radial_faces.front(), problo[0])
       || !near(radial_faces.back(), probhi[0])
       || !(problo[0] >= 0.0 && probhi[0] > problo[0])
       || !standard_theta_domain) {
      amrex::Abort(
         "axisymmetric spherical elliptic coordinates require a nonperiodic "
         "physical (r,theta) grid with 0 <= r_lo < r_hi and theta in "
         "[0,pi/2], [pi/2,pi], or [0,pi]"
      );
   }
#if (AMREX_SPACEDIM == 3)
   if (!geometry.isPeriodic(2)
       || !near(probhi[2] - problo[2], 2.0 * constants::pi)
       || resolved_boundary_type(boundary, 0, 2, false)
          != InitialDataEllipticBoundaryType::periodic
       || resolved_boundary_type(boundary, 0, 2, true)
          != InitialDataEllipticBoundaryType::periodic) {
      amrex::Abort(
         "full-storage axisymmetric spherical solves require one periodic "
         "2pi phi interval"
      );
   }
#endif
   const bool toroidal =
      mode == AxisymmetricSphericalMode::toroidal_vector;
   const InitialDataEllipticBoundaryType origin_type = toroidal
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   const InitialDataEllipticBoundaryType lower_angular_type =
      toroidal && lower_axis
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   const InitialDataEllipticBoundaryType upper_angular_type =
      toroidal && upper_axis
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   const InitialDataEllipticBoundaryType radial_inner_type =
      resolved_boundary_type(boundary, 0, 0, false);
   const bool valid_shell_inner_type =
      radial_inner_type == InitialDataEllipticBoundaryType::dirichlet
      || radial_inner_type == InitialDataEllipticBoundaryType::neumann
      || radial_inner_type == InitialDataEllipticBoundaryType::robin;
   if ((radial_origin && radial_inner_type != origin_type)
       || (!radial_origin && !valid_shell_inner_type)
       || resolved_boundary_type(boundary, 0, 1, false)
          != lower_angular_type
       || resolved_boundary_type(boundary, 0, 1, true)
          != upper_angular_type) {
      amrex::Abort(toroidal
         ? "the spherical toroidal solve requires a fixed shell inner face "
           "or odd origin, odd axis, and even equator regularity"
         : "the spherical scalar solve requires a fixed shell inner face "
           "or even origin, axis, and equator regularity");
   }
   const InitialDataEllipticBoundaryType radial_outer_type =
      resolved_boundary_type(boundary, 0, 0, true);
   if (radial_outer_type != InitialDataEllipticBoundaryType::dirichlet
       && radial_outer_type != InitialDataEllipticBoundaryType::robin) {
      amrex::Abort(
         "the spherical scalar solve requires Dirichlet or Robin radial closure"
      );
   }

   const int radial_cells = domain.length(0);
   const int angular_cells = domain.length(1);
   const int radial_begin = domain.smallEnd(0);
   const int angular_begin = domain.smallEnd(1);
   const int radial_end = domain.bigEnd(0);
   for (int face = 0; face <= angular_cells; ++face) {
      if (!near(angular_faces[face], problo[1] + face * dx[1])) {
         amrex::Abort(
            "the separable spherical solver currently requires uniform theta"
         );
      }
   }
   std::vector<amrex::Real> radial_centers(radial_cells);
   std::vector<amrex::Real> radial_widths(radial_cells);
   for (int radial_index = 0;
        radial_index < radial_cells;
        ++radial_index) {
      radial_centers[radial_index] = 0.5 * (
         radial_faces[radial_index] + radial_faces[radial_index + 1]
      );
      radial_widths[radial_index] =
         radial_faces[radial_index + 1] - radial_faces[radial_index];
   }
#if (AMREX_SPACEDIM == 3)
   const int plane_index = domain.smallEnd(2);
#else
   constexpr int plane_index = 0;
#endif
   const std::size_t total_cells =
      static_cast<std::size_t>(radial_cells) * angular_cells;

#ifdef AMREX_USE_GPU
   amrex::MultiFab host_rhs_storage = host_copy(*rhs[0]);
   amrex::MultiFab host_coefficient_a_storage = host_copy(
      boundary.coefficient_a(0)
   );
   amrex::MultiFab host_coefficient_b_storage = host_copy(
      boundary.coefficient_b(0)
   );
   amrex::MultiFab host_coefficient_c_storage = host_copy(
      boundary.coefficient_c(0)
   );
   amrex::MultiFab const& host_rhs = host_rhs_storage;
   amrex::MultiFab const& host_coefficient_a = host_coefficient_a_storage;
   amrex::MultiFab const& host_coefficient_b = host_coefficient_b_storage;
   amrex::MultiFab const& host_coefficient_c = host_coefficient_c_storage;
#else
   amrex::MultiFab const& host_rhs = *rhs[0];
   amrex::MultiFab const& host_coefficient_a = boundary.coefficient_a(0);
   amrex::MultiFab const& host_coefficient_b = boundary.coefficient_b(0);
   amrex::MultiFab const& host_coefficient_c = boundary.coefficient_c(0);
#endif
   std::vector<amrex::Real> weighted_rhs(total_cells, 0.0);
   std::vector<amrex::Real> inner_a(angular_cells, 0.0);
   std::vector<amrex::Real> inner_b(angular_cells, 0.0);
   std::vector<amrex::Real> inner_c(angular_cells, 0.0);
   std::vector<int> inner_counts(angular_cells, 0);
   std::vector<amrex::Real> outer_a(angular_cells, 0.0);
   std::vector<amrex::Real> outer_b(angular_cells, 0.0);
   std::vector<amrex::Real> outer_c(angular_cells, 0.0);
   std::vector<int> outer_counts(angular_cells, 0);
   for (amrex::MFIter mfi(host_rhs); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.validbox();
#if (AMREX_SPACEDIM == 3)
      if (plane_index < box.smallEnd(2) || plane_index > box.bigEnd(2)) {
         continue;
      }
#endif
      const auto source = host_rhs.const_array(mfi);
      for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
         const int angular_index = j - angular_begin;
         for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
            const int radial_index = i - radial_begin;
            const amrex::Real radius = radial_centers[radial_index];
            weighted_rhs[
               static_cast<std::size_t>(radial_index) * angular_cells
               + angular_index
            ] = radius * radius * source(i, j, plane_index, 0);
         }
      }
   }
   for (amrex::MFIter mfi(host_coefficient_a); mfi.isValid(); ++mfi) {
      const amrex::Box valid_box = mfi.validbox();
#if (AMREX_SPACEDIM == 3)
      if (plane_index < valid_box.smallEnd(2)
          || plane_index > valid_box.bigEnd(2)) {
         continue;
      }
#endif
      const auto coefficient_a = host_coefficient_a.const_array(mfi);
      const auto coefficient_b = host_coefficient_b.const_array(mfi);
      const auto coefficient_c = host_coefficient_c.const_array(mfi);
      if (!radial_origin && valid_box.smallEnd(0) == radial_begin) {
         const int ghost_i = radial_begin - 1;
         for (int j = valid_box.smallEnd(1); j <= valid_box.bigEnd(1); ++j) {
            const int angular_index = j - angular_begin;
            inner_a[angular_index] = coefficient_a(
               ghost_i, j, plane_index, 0
            );
            inner_b[angular_index] = coefficient_b(
               ghost_i, j, plane_index, 0
            );
            inner_c[angular_index] = coefficient_c(
               ghost_i, j, plane_index, 0
            );
            inner_counts[angular_index] = 1;
         }
      }
      if (valid_box.bigEnd(0) == radial_end) {
         const int ghost_i = radial_end + 1;
         for (int j = valid_box.smallEnd(1); j <= valid_box.bigEnd(1); ++j) {
            const int angular_index = j - angular_begin;
            outer_a[angular_index] = coefficient_a(
               ghost_i, j, plane_index, 0
            );
            outer_b[angular_index] = coefficient_b(
               ghost_i, j, plane_index, 0
            );
            outer_c[angular_index] = coefficient_c(
               ghost_i, j, plane_index, 0
            );
            outer_counts[angular_index] = 1;
         }
      }
   }
   const auto communicator = amrex::ParallelContext::CommunicatorSub();
   amrex::ParallelAllReduce::Sum(
      weighted_rhs.data(), static_cast<int>(weighted_rhs.size()), communicator
   );
#if (AMREX_SPACEDIM == 3)
   amrex::Real azimuthal_rhs_difference = 0.0;
   amrex::Real azimuthal_rhs_scale = 0.0;
   for (amrex::MFIter mfi(host_rhs); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.validbox();
      const auto source = host_rhs.const_array(mfi);
      for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
         for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            const int angular_index = j - angular_begin;
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
               const int radial_index = i - radial_begin;
               const amrex::Real radius = radial_centers[radial_index];
               const amrex::Real weighted_source =
                  radius * radius * source(i, j, k, 0);
               const amrex::Real reference = weighted_rhs[
                  static_cast<std::size_t>(radial_index) * angular_cells
                  + angular_index
               ];
               azimuthal_rhs_difference = std::max(
                  azimuthal_rhs_difference,
                  std::abs(weighted_source - reference)
               );
               azimuthal_rhs_scale = std::max(
                  azimuthal_rhs_scale,
                  std::max(std::abs(weighted_source), std::abs(reference))
               );
            }
         }
      }
   }
   amrex::ParallelAllReduce::Max(azimuthal_rhs_difference, communicator);
   amrex::ParallelAllReduce::Max(azimuthal_rhs_scale, communicator);
   const amrex::Real azimuthal_rhs_tolerance =
      1.0e-14 + 1.0e-12 * azimuthal_rhs_scale;
   if (azimuthal_rhs_difference > azimuthal_rhs_tolerance) {
      amrex::Abort(
         "full-storage axisymmetric spherical solve received a "
         "phi-dependent right-hand side"
      );
   }
#endif
   amrex::ParallelAllReduce::Sum(
      inner_a.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      inner_b.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      inner_c.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      inner_counts.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      outer_a.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      outer_b.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      outer_c.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      outer_counts.data(), angular_cells, communicator
   );

   std::vector<amrex::Real> inner_gamma(angular_cells, 0.0);
   amrex::Real inner_delta = 0.0;
   if (!radial_origin) {
      for (int angular_index = 0;
           angular_index < angular_cells;
           ++angular_index) {
         if (inner_counts[angular_index] != 1) {
            amrex::Abort(
               "inner radial boundary data do not have one unique owner"
            );
         }
         const amrex::Real a = inner_a[angular_index];
         const amrex::Real b = radial_inner_type
               == InitialDataEllipticBoundaryType::dirichlet
            ? 0.0
            : inner_b[angular_index];
         const amrex::Real inner_width = radial_widths.front();
         const amrex::Real denominator = 0.5 * a - b / inner_width;
         if (denominator == 0.0 || !std::isfinite(denominator)) {
            amrex::Abort("singular inner radial elliptic boundary coefficients");
         }
         const amrex::Real delta = -(
            0.5 * a + b / inner_width
         ) / denominator;
         if (angular_index == 0) {
            inner_delta = delta;
         } else if (!near(delta, inner_delta)) {
            amrex::Abort(
               "the tensor-product spherical solve requires an angularly "
               "uniform inner radial boundary operator"
            );
         }
         inner_gamma[angular_index] = inner_c[angular_index] / denominator;
      }
   }

   std::vector<amrex::Real> outer_gamma(angular_cells);
   amrex::Real outer_delta = 0.0;
   for (int angular_index = 0;
        angular_index < angular_cells;
        ++angular_index) {
      if (outer_counts[angular_index] != 1) {
         amrex::Abort("radial boundary data do not have one unique owner");
      }
      const amrex::Real a = outer_a[angular_index];
      const amrex::Real b = radial_outer_type
            == InitialDataEllipticBoundaryType::dirichlet
         ? 0.0
         : outer_b[angular_index];
      const amrex::Real outer_width = radial_widths.back();
      const amrex::Real denominator = 0.5 * a + b / outer_width;
      if (denominator == 0.0 || !std::isfinite(denominator)) {
         amrex::Abort("singular radial elliptic boundary coefficients");
      }
      const amrex::Real delta =
         (b / outer_width - 0.5 * a) / denominator;
      if (angular_index == 0) {
         outer_delta = delta;
      } else if (!near(delta, outer_delta)) {
         amrex::Abort(
            "the tensor-product spherical solve requires an angularly uniform "
            "radial boundary operator"
         );
      }
      outer_gamma[angular_index] = outer_c[angular_index] / denominator;
   }

   std::vector<amrex::Real> radial_lower(radial_cells, 0.0);
   std::vector<amrex::Real> radial_diagonal(radial_cells, 0.0);
   std::vector<amrex::Real> radial_upper(radial_cells, 0.0);
   for (int radial_index = 0; radial_index < radial_cells; ++radial_index) {
      if (radial_index > 0) {
         const amrex::Real face_radius = radial_faces[radial_index];
         const amrex::Real center_distance =
            radial_centers[radial_index]
            - radial_centers[radial_index - 1];
         radial_lower[radial_index] = face_radius * face_radius / (
            radial_widths[radial_index] * center_distance
         );
      }
      if (radial_index + 1 < radial_cells) {
         const amrex::Real face_radius = radial_faces[radial_index + 1];
         const amrex::Real center_distance =
            radial_centers[radial_index + 1]
            - radial_centers[radial_index];
         radial_upper[radial_index] = face_radius * face_radius / (
            radial_widths[radial_index] * center_distance
         );
      }
      radial_diagonal[radial_index] = -(
         radial_lower[radial_index] + radial_upper[radial_index]
      );
   }
   if (!radial_origin) {
      const amrex::Real inner_face_coefficient =
         radial_faces.front() * radial_faces.front() / (
            radial_widths.front() * radial_widths.front()
         );
      radial_diagonal.front() += inner_face_coefficient * (
         inner_delta - 1.0
      );
      for (int angular_index = 0;
           angular_index < angular_cells;
           ++angular_index) {
         weighted_rhs[angular_index] -=
            inner_face_coefficient * inner_gamma[angular_index];
      }
   }
   const amrex::Real outer_face_coefficient =
      radial_faces.back() * radial_faces.back() / (
         radial_widths.back() * radial_widths.back()
      );
   radial_diagonal.back() += outer_face_coefficient * (
      outer_delta - 1.0
   );
   for (int angular_index = 0;
        angular_index < angular_cells;
        ++angular_index) {
      weighted_rhs[
         static_cast<std::size_t>(radial_cells - 1) * angular_cells
         + angular_index
      ] -= outer_face_coefficient * outer_gamma[angular_index];
   }

   const auto angular_decomposition = axisymmetric_angular_decomposition(
      angular_cells, problo[1], dx[1], mode
   );
   const amrex::Real angular_scale = 1.0 / (dx[1] * dx[1]);
   const amrex::Real angular_spin_squared =
      mode == AxisymmetricSphericalMode::toroidal_vector ? 1.0 : 0.0;
   const auto& angular_eigensystem = angular_decomposition->eigensystem;
   const auto& rhs_projection_vectors =
      angular_decomposition->rhs_projection_vectors;
   const auto& solution_reconstruction_vectors =
      angular_decomposition->solution_reconstruction_vectors;
   std::vector<amrex::Real> angular_lower(angular_cells, 0.0);
   std::vector<amrex::Real> angular_upper(angular_cells, 0.0);
   std::vector<amrex::Real> angular_spin_coefficient(
      angular_cells, 0.0
   );
   for (int angular_index = 0;
        angular_index < angular_cells;
        ++angular_index) {
      const amrex::Real theta = problo[1]
         + (angular_index + 0.5) * dx[1];
      const amrex::Real sin_theta = std::sin(theta);
      if (angular_index > 0) {
         const amrex::Real face_theta = problo[1]
            + angular_index * dx[1];
         angular_lower[angular_index] =
            std::sin(face_theta) / sin_theta * angular_scale;
      }
      if (angular_index + 1 < angular_cells) {
         const amrex::Real face_theta = problo[1]
            + (angular_index + 1) * dx[1];
         angular_upper[angular_index] =
            std::sin(face_theta) / sin_theta * angular_scale;
      }
      angular_spin_coefficient[angular_index] =
         angular_spin_squared / (sin_theta * sin_theta);
   }

   const int rank = amrex::ParallelContext::MyProcSub();
   const int ranks = amrex::ParallelContext::NProcsSub();
   std::vector<amrex::Real> mode_solution(
      static_cast<std::size_t>(angular_cells) * radial_cells, 0.0
   );
   std::vector<amrex::Real> forward_upper(radial_cells, 0.0);
   std::vector<amrex::Real> forward_rhs(radial_cells, 0.0);
   for (int mode = rank; mode < angular_cells; mode += ranks) {
      const amrex::Real* projection_vector =
         rhs_projection_vectors.data()
         + static_cast<std::size_t>(mode) * angular_cells;
      amrex::Real* radial_mode = mode_solution.data()
         + static_cast<std::size_t>(mode) * radial_cells;
      for (int radial_index = 0;
           radial_index < radial_cells;
           ++radial_index) {
         amrex::Real projected_rhs = 0.0;
         const amrex::Real* radial_rhs = weighted_rhs.data()
            + static_cast<std::size_t>(radial_index) * angular_cells;
         for (int angular_index = 0;
              angular_index < angular_cells;
              ++angular_index) {
            projected_rhs += projection_vector[angular_index]
               * radial_rhs[angular_index];
         }
         radial_mode[radial_index] = projected_rhs;
      }

      amrex::Real denominator = radial_diagonal[0]
         + angular_eigensystem.eigenvalues[mode];
      if (denominator == 0.0 || !std::isfinite(denominator)) {
         amrex::Abort("singular spherical radial mode");
      }
      forward_upper[0] = radial_upper[0] / denominator;
      forward_rhs[0] = radial_mode[0] / denominator;
      for (int radial_index = 1;
           radial_index < radial_cells;
           ++radial_index) {
         denominator = radial_diagonal[radial_index]
            + angular_eigensystem.eigenvalues[mode]
            - radial_lower[radial_index]
               * forward_upper[radial_index - 1];
         if (denominator == 0.0 || !std::isfinite(denominator)) {
            amrex::Abort("singular spherical radial elimination");
         }
         forward_upper[radial_index] = radial_index + 1 < radial_cells
            ? radial_upper[radial_index] / denominator
            : 0.0;
         forward_rhs[radial_index] = (
            radial_mode[radial_index]
            - radial_lower[radial_index] * forward_rhs[radial_index - 1]
         ) / denominator;
      }
      radial_mode[radial_cells - 1] = forward_rhs[radial_cells - 1];
      for (int radial_index = radial_cells - 2;
           radial_index >= 0;
           --radial_index) {
         radial_mode[radial_index] = forward_rhs[radial_index]
            - forward_upper[radial_index]
               * radial_mode[radial_index + 1];
      }
   }

   std::vector<amrex::Real> solution_values(total_cells, 0.0);
   for (int mode = rank; mode < angular_cells; mode += ranks) {
      const amrex::Real* reconstruction_vector =
         solution_reconstruction_vectors.data()
         + static_cast<std::size_t>(mode) * angular_cells;
      const amrex::Real* radial_mode = mode_solution.data()
         + static_cast<std::size_t>(mode) * radial_cells;
      for (int radial_index = 0;
           radial_index < radial_cells;
           ++radial_index) {
         amrex::Real* radial_solution = solution_values.data()
            + static_cast<std::size_t>(radial_index) * angular_cells;
         for (int angular_index = 0;
              angular_index < angular_cells;
              ++angular_index) {
            radial_solution[angular_index] +=
               reconstruction_vector[angular_index]
               * radial_mode[radial_index];
         }
      }
   }
   amrex::ParallelAllReduce::Sum(
      solution_values.data(), static_cast<int>(solution_values.size()),
      communicator
   );

   amrex::Real residual = 0.0;
   for (int radial_index = 0;
        radial_index < radial_cells;
        ++radial_index) {
      for (int angular_index = 0;
           angular_index < angular_cells;
           ++angular_index) {
         const std::size_t offset =
            static_cast<std::size_t>(radial_index) * angular_cells
            + angular_index;
         amrex::Real applied = radial_diagonal[radial_index]
            * solution_values[offset];
         if (radial_index > 0) {
            applied += radial_lower[radial_index]
               * solution_values[offset - angular_cells];
         }
         if (radial_index + 1 < radial_cells) {
            applied += radial_upper[radial_index]
               * solution_values[offset + angular_cells];
         }
         if (angular_index > 0) {
            applied += angular_lower[angular_index] * (
               solution_values[offset - 1] - solution_values[offset]
            );
         }
         if (angular_index + 1 < angular_cells) {
            applied += angular_upper[angular_index] * (
               solution_values[offset + 1] - solution_values[offset]
            );
         }
         applied -= angular_spin_coefficient[angular_index]
            * solution_values[offset];
         residual = std::max(
            residual, std::abs(applied - weighted_rhs[offset])
         );
      }
   }

#ifdef AMREX_USE_GPU
   amrex::MultiFab host_solution_storage = host_copy(*solution[0]);
   amrex::MultiFab& host_solution = host_solution_storage;
#else
   amrex::MultiFab& host_solution = *solution[0];
#endif
   for (amrex::MFIter mfi(host_solution); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.validbox();
      const auto destination = host_solution.array(mfi);
#if (AMREX_SPACEDIM == 3)
      for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
#else
      constexpr int k = plane_index;
#endif
      for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
         const int angular_index = j - angular_begin;
         for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
            const int radial_index = i - radial_begin;
            destination(i, j, k, 0) = solution_values[
               static_cast<std::size_t>(radial_index) * angular_cells
               + angular_index
            ];
         }
      }
#if (AMREX_SPACEDIM == 3)
      }
#endif
   }
#ifdef AMREX_USE_GPU
   copy_host_to_device(*solution[0], host_solution);
#endif
   return elliptic_result(residual);
#endif
}

} // namespace

InitialDataEllipticResult solve_initial_data_axisymmetric_spherical_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   return solve_axisymmetric_spherical_separable(
      hierarchy,
      solution,
      rhs,
      boundary,
      options,
      AxisymmetricSphericalMode::scalar
   );
}

InitialDataEllipticResult
solve_initial_data_axisymmetric_spherical_toroidal_vector_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_solution,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   return solve_axisymmetric_spherical_separable(
      hierarchy,
      azimuthal_solution,
      azimuthal_source,
      boundary,
      options,
      AxisymmetricSphericalMode::toroidal_vector
   );
}


} // namespace pyfluid_amrex
