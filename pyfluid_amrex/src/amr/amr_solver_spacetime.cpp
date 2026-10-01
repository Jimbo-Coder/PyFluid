#include "pyfluid_amrex/spacetime_derivatives.hpp"
#include "amr/amr_solver_base.hpp"
#include "pyfluid_amrex/fccz4_runtime.hpp"
#include "pyfluid_amrex/fccz4_constraint_access.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"
#include "pyfluid_amrex/initial_data.hpp"

#include <AMReX_Math.H>
#include <AMReX_MultiFabUtil.H>

#include <cmath>
#include <type_traits>

#include "pyfluid_amrex/eulerian_matter_access.hpp"
#include "pyfluid_amrex/hierarchy_interpolation.hpp"
#include "pyfluid_amrex/initial_data/materialization.hpp"
#include "pyfluid_amrex/initial_data/magnetic.hpp"
#include "pyfluid_amrex/initial_data/kerr_schild.hpp"
#include "pyfluid_amrex/initial_data/physical_id.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/kernels_generated/diagnostics.hpp"
#include "pyfluid_amrex/nodal_fill.hpp"
#include "pyfluid_amrex/nodal_interpolater.hpp"
#include "pyfluid_amrex/surface_geometry.hpp"
#include "pyfluid_amrex/z4c_constraint_pointwise.hpp"
#include "pyfluid_amrex/spacetime_constraint_region.hpp"
#include "pyfluid_amrex/z4c_excision_derivatives.hpp"
#include "relativity/z4c_boundary.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

bool spacetime_boundary_reflects(
   RunConfig const& config,
   int direction,
   bool high_side
)
{
   const BoundaryKind kind = side_boundary_kind(
      config, direction, high_side
   );
   return kind == BoundaryKind::Reflecting || kind == BoundaryKind::Symmetry;
}

void fill_reflected_nodal_validity_ghosts(
   amrex::iMultiFab& validity,
   amrex::Geometry const& geometry,
   RunConfig const& config,
   int ghost_cells
)
{
   const amrex::Box nodal_domain = amrex::convert(
      geometry.Domain(), amrex::IntVect::TheNodeVector()
   );
   const auto periodic = geometry.isPeriodicArray();
   const int lo[3]{
      nodal_domain.smallEnd(0),
#if (AMREX_SPACEDIM >= 2)
      nodal_domain.smallEnd(1),
#else
      0,
#endif
#if (AMREX_SPACEDIM == 3)
      nodal_domain.smallEnd(2),
#else
      0,
#endif
   };
   const int hi[3]{
      nodal_domain.bigEnd(0),
#if (AMREX_SPACEDIM >= 2)
      nodal_domain.bigEnd(1),
#else
      0,
#endif
#if (AMREX_SPACEDIM == 3)
      nodal_domain.bigEnd(2),
#else
      0,
#endif
   };
   const bool reflect_lo[3]{
      spacetime_boundary_reflects(config, 0, false),
#if (AMREX_SPACEDIM >= 2)
      spacetime_boundary_reflects(config, 1, false),
#else
      false,
#endif
#if (AMREX_SPACEDIM == 3)
      spacetime_boundary_reflects(config, 2, false),
#else
      false,
#endif
   };
   const bool reflect_hi[3]{
      spacetime_boundary_reflects(config, 0, true),
#if (AMREX_SPACEDIM >= 2)
      spacetime_boundary_reflects(config, 1, true),
#else
      false,
#endif
#if (AMREX_SPACEDIM == 3)
      spacetime_boundary_reflects(config, 2, true),
#else
      false,
#endif
   };
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(
           validity, amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      amrex::Box grown = mfi.growntilebox(ghost_cells);
      grown &= validity[mfi].box();
      const amrex::BoxList ghost_boxes = amrex::boxDiff(
         grown, nodal_domain
      );
      const auto values = validity.array(mfi);
      for (amrex::Box const& ghost_box : ghost_boxes) {
         amrex::ParallelFor(
            ghost_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const int point[3]{i, j, k};
               int source[3]{i, j, k};
               bool touches_reflection = false;
               bool supported = true;
               for (int direction = 0; direction < AMREX_SPACEDIM;
                    ++direction) {
                  if (point[direction] < lo[direction]) {
                     if (periodic[direction]) {
                        continue;
                     } else if (reflect_lo[direction]) {
                        source[direction] =
                           2 * lo[direction] - point[direction];
                        touches_reflection = true;
                     } else {
                        supported = false;
                     }
                  } else if (point[direction] > hi[direction]) {
                     if (periodic[direction]) {
                        continue;
                     } else if (reflect_hi[direction]) {
                        source[direction] =
                           2 * hi[direction] - point[direction];
                        touches_reflection = true;
                     } else {
                        supported = false;
                     }
                  }
               }
               if (supported && touches_reflection) {
                  values(i, j, k, 0) = values(
                     source[0], source[1], source[2], 0
                  );
               }
            }
         );
      }
   }
}

void enforce_reflection_odd_nodal_plane_values(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   amrex::Vector<amrex::BCRec> const& boundary_records
)
{
   const amrex::Box nodal_domain = amrex::convert(
      geometry.Domain(), amrex::IntVect::TheNodeVector()
   );
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      for (int high_side = 0; high_side < 2; ++high_side) {
         amrex::GpuArray<int, Z4C_NUM_COMPONENTS> odd_component{};
         bool has_odd_component = false;
         for (int component = 0; component < Z4C_NUM_COMPONENTS;
              ++component) {
            const int boundary_kind = high_side != 0
               ? boundary_records[component].hi(direction)
               : boundary_records[component].lo(direction);
            odd_component[component] =
               boundary_kind == amrex::BCType::reflect_odd ? 1 : 0;
            has_odd_component = has_odd_component
               || odd_component[component] != 0;
         }
         if (!has_odd_component) {
            continue;
         }
         const int plane_index = high_side != 0
            ? nodal_domain.bigEnd(direction)
            : nodal_domain.smallEnd(direction);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         for (amrex::MFIter mfi(
                 state, amrex::TilingIfNotGPU()
              ); mfi.isValid(); ++mfi) {
            amrex::Box face = mfi.tilebox();
            if (plane_index < face.smallEnd(direction)
                || plane_index > face.bigEnd(direction)) {
               continue;
            }
            face.setSmall(direction, plane_index);
            face.setBig(direction, plane_index);
            const auto values = state.array(mfi);
            amrex::ParallelFor(
               face,
               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  for (int component = 0;
                       component < Z4C_NUM_COMPONENTS;
                       ++component) {
                     if (odd_component[component] != 0) {
                        values(i, j, k, component) = 0.0_rt;
                     }
                  }
               }
            );
         }
      }
   }
}

// Keep formulation selection on the host and accuracy dispatch inside the
// shared derivative gather. A named launch owner also avoids CUDA's ban on
// extended device lambdas nested in generic host lambdas.
template <bool reference_metric_formulation>
void fill_constraint_norms(
   amrex::Box const& box,
   amrex::Array4<amrex::Real const> input,
   amrex::Array4<amrex::Real const> matter,
   amrex::Array4<amrex::Real> output,
   amrex::Array4<int const> valid_points,
   amrex::Array4<int const> same_level_points,
   amrex::Array4<int const> owned_points,
   ExcisionGridView const& excision,
   bool matter_coupled,
   int stencil_radius,
   int derivative_order,
   amrex::GpuArray<amrex::Real, 3> const& inv_spacing,
   ReferenceChartGeometry const& reference_chart,
   bool include_regions,
   SpacetimeConstraintRegion const& region,
   bool has_black_hole,
   FixedSpacetimeParameters const& black_hole,
   amrex::Real horizon_radius_squared
)
{
   amrex::ParallelFor(box, [input, matter, output, valid_points, same_level_points, owned_points,
       excision, matter_coupled, stencil_radius, derivative_order, inv_spacing,
       reference_chart, include_regions, region, has_black_hole, black_hole,
       horizon_radius_squared] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if (valid_points(i, j, k, 0) == 0
          || owned_points(i, j, k, 0) == 0
          || (excision.region.enabled() && !excision(i, j, k, 0))) {
         return;
      }
      const Z4cMatter point_matter = matter_coupled
         ? load_eulerian_matter(matter, i, j, k)
         : vacuum_z4c_matter();
      bool same_level_stencil = true;
      for (int offset = 1; offset <= stencil_radius; ++offset) {
         same_level_stencil = same_level_stencil
            && same_level_points(i - offset, j, k, 0) != 0
            && same_level_points(i + offset, j, k, 0) != 0;
#if (AMREX_SPACEDIM >= 2)
         same_level_stencil = same_level_stencil
            && same_level_points(i, j - offset, k, 0) != 0
            && same_level_points(i, j + offset, k, 0) != 0;
#endif
#if (AMREX_SPACEDIM == 3)
         same_level_stencil = same_level_stencil
            && same_level_points(i, j, k - offset, 0) != 0
            && same_level_points(i, j, k + offset, 0) != 0;
#endif
      }
      const Z4cState local_state = load_z4c_state(input, i, j, k);
      const auto derivatives = gather_spacetime_derivatives<Z4cDerivativePurpose::Constraints>(
         derivative_order, input, i, j, k, inv_spacing, local_state, excision);
      Z4cConstraints c;
      if constexpr (reference_metric_formulation) {
         c = evaluate_fccz4_constraint_layout(input, i, j, k, derivatives,
            reference_chart.at(i, j, k), point_matter);
      } else {
         c = generated::evaluate_z4c_constraints(local_state, derivatives, point_matter);
      }
      output(i, j, k, 0) = amrex::Math::abs(c.hamiltonian);
      output(i, j, k, 1) = c.momentum1 * c.momentum1
         + c.momentum2 * c.momentum2 + c.momentum3 * c.momentum3;
      output(i, j, k, 2) = c.z_tilde1 * c.z_tilde1
         + c.z_tilde2 * c.z_tilde2 + c.z_tilde3 * c.z_tilde3;
      output(i, j, k, 3) = amrex::Math::abs(c.theta);
      output(i, j, k, 4) = amrex::Math::abs(
         c.log_conformal_determinant
      );
      output(i, j, k, 5) = amrex::Math::abs(c.trace_a_tilde);
      output(i, j, k, 6) = 1.0_rt;
      const int value_component = same_level_stencil ? 7 : 9;
      const int weight_component = same_level_stencil ? 8 : 10;
      output(i, j, k, value_component) = amrex::Math::abs(c.hamiltonian);
      output(i, j, k, weight_component) = 1.0_rt;
      if (include_regions) {
         const auto chart = reference_chart.at(i, j, k);
         const auto position = mesh_coordinate_triplet_to_cartesian(chart.coordinate, chart.position);
         const int partition = region.classify(position);
         if (partition >= 0) {
            // Reuse c before the legacy, dx-dependent horizon mask.
            // Each sample belongs to the complete region and exactly
            // one near/far partition. Reference-volume weighting is
            // applied after the unweighted maximum reduction.
            for (int entry = 0; entry < 2; ++entry) {
               const int region_index = entry == 0 ? 0 : partition + 1;
               const int first_component = 14 + 3 * region_index;
               output(i, j, k, first_component) = output(i, j, k, 0);
               output(i, j, k, first_component + 1) = output(i, j, k, 1);
               output(i, j, k, first_component + 2) = 1.0_rt;
            }
         }
      }
      if (has_black_hole) {
         bool exterior_stencil = true;
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            for (int offset = -stencil_radius; offset <= stencil_radius; ++offset) {
               int point_i = i;
               int point_j = j;
               int point_k = k;
               if (direction == 0) {
                  point_i += offset;
               }
#if (AMREX_SPACEDIM >= 2)
               if (direction == 1) {
                  point_j += offset;
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (direction == 2) {
                  point_k += offset;
               }
#endif
               const auto chart = reference_chart.at(point_i, point_j, point_k);
               const auto physical = mesh_coordinate_triplet_to_cartesian(chart.coordinate, chart.position);
               exterior_stencil = exterior_stencil
                  && black_hole_radial_coordinate_squared(
                       black_hole, physical[0], physical[1], physical[2]
                     ) > horizon_radius_squared;
            }
         }
         if (exterior_stencil) {
            output(i, j, k, 11) = amrex::Math::abs(c.hamiltonian);
            output(i, j, k, 12) = 1.0_rt;
            output(i, j, k, 13) = output(i, j, k, 1);
         }
      }
   });
}

} // namespace

void AmrSolverBase::fill_spacetime_level_boundaries(
   LevelData& data,
   int lev,
   amrex::Real time,
   bool synchronous_levels,
   amrex::Real fine_step_start,
   int rk_stage
)
{
   if (!spacetime_state_.active() || !data.at(lev)) {
      return;
   }
   SpacetimeFillPatchPhysBC fine_bc(
      Geom(lev), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   if (lev == 0) {
      amrex::Vector<amrex::MultiFab*> source{data[0].get()};
      amrex::Vector<amrex::Real> source_time{time};
      amrex::FillPatchSingleLevel(
         *data[0],
         data[0]->nGrowVect(),
         time,
         source,
         source_time,
         0,
         0,
         spacetime_state_.num_components(),
         Geom(0),
         fine_bc,
         0
      );
      enforce_reflection_odd_nodal_plane_values(
         *data[0], Geom(0), spacetime_fillpatch_bcs_
      );
      return;
   }

   amrex::Vector<amrex::MultiFab*> coarse_data;
   amrex::Vector<amrex::Real> coarse_time;
   if (!spacetime_state_.evolves()) {
      coarse_data.push_back(data[lev - 1].get());
      coarse_time.push_back(time);
   } else if (synchronous_levels) {
      coarse_data.push_back(data[lev - 1].get());
      coarse_time.push_back(time);
   } else if (
      rk_stage > 0
      && spacetime_state_.rk4_stage_data_ready(lev - 1, level_dts_[lev - 1])
   ) {
      spacetime_state_.fill_rk4_dense_state(
         lev - 1,
         fine_step_start,
         rk_stage,
         level_old_times_[lev - 1],
         level_times_[lev - 1],
         level_dts_[lev - 1],
         level_dts_[lev]
      );
      coarse_data.push_back(
         spacetime_state_.rk4_coarse_stage_state()[lev - 1].get()
      );
      coarse_time.push_back(time);
   } else if (coarse_time_interval_contains(lev - 1, time)) {
      coarse_data.push_back(spacetime_state_.old_state()[lev - 1].get());
      coarse_data.push_back(spacetime_state_.state()[lev - 1].get());
      coarse_time.push_back(level_old_times_[lev - 1]);
      coarse_time.push_back(level_times_[lev - 1]);
      time = clamp_to_coarse_time_interval(lev - 1, time);
   } else {
      coarse_data.push_back(data[lev - 1].get());
      coarse_time.push_back(time);
   }
   amrex::Vector<amrex::MultiFab*> fine_data{data[lev].get()};
   amrex::Vector<amrex::Real> fine_time{time};
   SpacetimeFillPatchPhysBC coarse_bc(
      Geom(lev - 1), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   fill_nodal_two_levels(
      *data[lev],
      data[lev]->nGrowVect(),
      time,
      coarse_data,
      coarse_time,
      fine_data,
      fine_time,
      Geom(lev - 1),
      Geom(lev),
      coarse_bc,
      fine_bc,
      refRatio(lev - 1),
      spacetime_fillpatch_bcs_,
      spacetime_state_.num_components(),
      &spacetime_interpolater_
   );
   enforce_reflection_odd_nodal_plane_values(
      *data[lev], Geom(lev), spacetime_fillpatch_bcs_
   );
}

void AmrSolverBase::fill_all_spacetime_boundaries(LevelData& data, amrex::Real time)
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      fill_spacetime_level_boundaries(data, lev, time);
   }
}

void AmrSolverBase::initialize_spacetime_level(int lev, amrex::Real time)
{
   if (!spacetime_state_.active()) {
      return;
   }
   if (cfg_.spacetime_formulation == "fccz4") {
      validate_fccz4_domain(Geom(lev), cfg_);
   }
   initialize_z4c_state(
      *spacetime_state_.state()[lev], Geom(lev), cfg_, time,
      initial_data_provider_->view()
   );
   if (cfg_.spacetime_formulation == "fccz4") {
      rotate_fccz4_initial_state(*spacetime_state_.state()[lev], Geom(lev), cfg_);
   }
   spacetime_coupling_.invalidate_cell_spacetime(lev);
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev, time);
   complete_spacetime_connections(*spacetime_state_.state()[lev], Geom(lev), cfg_);
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev, time);
}

bool AmrSolverBase::materialize_constraint_solved_initial_data(
   bool seed_from_loaded_id,
   bool reseed_static_tov_spacetime,
   bool preserve_conformal_baryon_density
)
{
   initial_data_rotating_star_summary_.reset();
   initial_data_black_hole_throat_area_ = -1.0_rt;
   initial_data_black_hole_irreducible_mass_ = -1.0_rt;
   initial_data_black_hole_throat_equatorial_circumference_ = -1.0_rt;
   initial_data_black_hole_throat_polar_circumference_ = -1.0_rt;
   initial_data_black_hole_throat_kerr_mass_ = -1.0_rt;
   initial_data_black_hole_throat_kerr_spin_ = -1.0_rt;
   initial_data_black_hole_throat_angular_momentum_ = 0.0_rt;
   initial_data_black_hole_throat_christodoulou_mass_ = -1.0_rt;
   initial_data_black_hole_throat_christodoulou_spin_ = 0.0_rt;
   initial_data_equilibrium_torus_summary_.reset();
   const bool uses_constraint_solve = cfg_.spacetime_initial_data == "ctt"
      || cfg_.spacetime_initial_data == "cttk"
      || cfg_.spacetime_initial_data == "xcts";
#if (AMREX_SPACEDIM == 1)
   if (cfg_.initial_data_radial_grid == "adaptive" && cfg_.ic == "tov"
       && !seed_from_loaded_id) {
      const RelativisticInitialDataView provider = initial_data_provider_->view();
      if (finestLevel() != 0 || !provider.tov.active()) {
         amrex::Abort(
            "adaptive TOV ID requires one level and a native TOV provider"
         );
      }
      InitialDataEllipticHierarchy mapped_hierarchy;
      mapped_hierarchy.geometry.push_back(Geom(0));
      mapped_hierarchy.grids.push_back(boxArray(0));
      mapped_hierarchy.distribution.push_back(DistributionMap(0));
      mapped_hierarchy.physical_coordinate_faces.resize(1);
      mapped_hierarchy.physical_coordinate_faces[0][0] =
         stellar_adaptive_radial_faces(
            provider.tov.surface_radius,
            Geom(0).ProbHi(0),
            Geom(0).Domain().length(0),
            cfg_.initial_data_adaptive_source_radius_factor,
            cfg_.initial_data_adaptive_exterior_fraction
         );
      initial_data_coordinate_faces_ =
         mapped_hierarchy.physical_coordinate_faces;
      initialize_relativistic_primitive_cache_from_initial_data(
         *primitive_[0], mapped_hierarchy, 0, cfg_, provider
      );
      initialize_mapped_z4c_state_from_provider(
         *spacetime_state_.state()[0],
         mapped_hierarchy,
         0,
         cfg_,
         0.0_rt,
         provider
      );
      const InitialDataCoordinateLine radial = mapped_hierarchy.coordinate_line(
         0, 0
      );
      amrex::Print()
         << "adaptive TOV ID radial_cells=" << radial.centers.size()
         << " surface=" << provider.tov.surface_radius
         << " inner_dr=" << radial.widths.front()
         << " outer_dr=" << radial.widths.back() << '\n';
      finalize_constraint_solved_initial_data_state();
      return true;
   }
#endif
   if (!uses_constraint_solve
       || !initial_data_provider_->view().fields.spacetime_from_id) {
      return false;
   }
   if (!spacetime_state_.active()) {
      amrex::Abort(
         "constraint-solved initial data requires stored fixed or evolved spacetime"
      );
   }

   InitialDataEllipticHierarchy hierarchy;
   if (cfg_.mesh_coordinate == "cartesian") {
      const MeshAxes axes = mesh_axes_layout(MeshCartesian, cfg_.mesh_axes);
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         hierarchy.cartesian_physical_axes[direction] =
            mesh_canonical_axis_index(
               MeshCartesian, mesh_axis(axes, direction)
            );
      }
   }
   amrex::Vector<amrex::MultiFab*> primitive;
   amrex::Vector<amrex::MultiFab*> spacetime;
   amrex::Vector<amrex::MultiFab*> cell_spacetime;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      hierarchy.geometry.push_back(Geom(lev));
      hierarchy.grids.push_back(boxArray(lev));
      hierarchy.distribution.push_back(DistributionMap(lev));
      if (seed_from_loaded_id) {
         fill_level_primitive_cache(lev, 0.0_rt);
      } else if (initial_data_provider_->view().has_fluid()
                 || cfg_.ic == "black_hole") {
         initialize_relativistic_primitive_cache_from_initial_data(
            *primitive_[lev], Geom(lev), cfg_, initial_data_provider_->view()
         );
      } else {
         fill_level_primitive_cache(lev, 0.0_rt);
      }
      primitive.push_back(primitive_[lev].get());
      spacetime.push_back(spacetime_state_.state()[lev].get());
      cell_spacetime.push_back(spacetime_coupling_.cell_spacetime()[lev].get());
      if (lev < finestLevel()) {
         hierarchy.refinement_ratio.push_back(refRatio(lev));
      }
   }
   hierarchy.cell_interpolater = selected_state_interpolater();
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      const BoundaryKind lower = side_boundary_kind(cfg_, direction, false);
      const BoundaryKind upper = side_boundary_kind(cfg_, direction, true);
      hierarchy.reflection_lo[direction] =
         lower == BoundaryKind::Symmetry || lower == BoundaryKind::Reflecting
         || lower == BoundaryKind::Origin;
      hierarchy.reflection_hi[direction] =
         upper == BoundaryKind::Symmetry || upper == BoundaryKind::Reflecting
         || upper == BoundaryKind::Origin;
   }
#if (AMREX_SPACEDIM >= 2)
   if (cfg_.mesh_coordinate == "spherical"
       && (cfg_.initial_data_radial_grid == "adaptive"
           || !initial_data_coordinate_faces_.empty())) {
      const RelativisticInitialDataView provider = initial_data_provider_->view();
      const bool black_hole_xcts = cfg_.ic == "black_hole"
         && cfg_.spacetime_initial_data == "xcts";
      if (finestLevel() != 0
          || (!provider.tov.active() && !black_hole_xcts)) {
         amrex::Abort(
            "adaptive spherical ID requires one level and a native TOV or "
            "black-hole XCTS source"
         );
      }
      if (seed_from_loaded_id) {
         if (initial_data_coordinate_faces_.empty()) {
            amrex::Abort(
               "adaptive stellar ID continuation requires mapped coordinate "
               "metadata in the saved artifact"
            );
         }
         hierarchy.physical_coordinate_faces = initial_data_coordinate_faces_;
         if (reseed_static_tov_spacetime) {
            if (cfg_.ic != "rotating_star"
                || cfg_.initial_data_angular_velocity != 0.0_rt
                || cfg_.initial_data_disk != "none"
                || !provider.tov.active()) {
               amrex::Abort(
                  "analytic static-TOV spacetime reseeding requires a "
                  "zero-rotation disk-free native TOV source"
               );
            }
            // A perturbation changes the live primitive, not the equilibrium
            // metric.  Start its constraint polish from the analytic mapped
            // TOV branch.  Reconstructing cell scalars through the generic
            // nodal handoff introduces a large axis-closure defect on this
            // reduced chart and can select a different nonlinear branch.
            initialize_mapped_z4c_state_from_provider(
               *spacetime_state_.state()[0],
               hierarchy,
               0,
               cfg_,
               0.0_rt,
               provider
            );
         }
      } else {
         hierarchy.physical_coordinate_faces.resize(1);
         hierarchy.physical_coordinate_faces[0][0] = black_hole_xcts
            ? stretched_radial_shell_faces(
                 0.5_rt * cfg_.spacetime_mass,
                 hierarchy.geometry[0].ProbHi(0),
                 hierarchy.geometry[0].Domain().length(0),
                 cfg_.initial_data_adaptive_shell_width_ratio
              )
            : stellar_adaptive_radial_faces(
                 provider.tov.surface_radius,
                 hierarchy.geometry[0].ProbHi(0),
                 hierarchy.geometry[0].Domain().length(0),
                 cfg_.initial_data_adaptive_source_radius_factor,
                 cfg_.initial_data_adaptive_exterior_fraction
              );
         initial_data_coordinate_faces_ = hierarchy.physical_coordinate_faces;
         initialize_relativistic_primitive_cache_from_initial_data(
            *primitive_[0], hierarchy, 0, cfg_, provider
         );
         if (provider.tov.active()) {
            // The mapped fluid and metric seed must represent the same
            // physical radii.  Sampling only the fluid on the adaptive map
            // leaves the analytic TOV spacetime on the uniform logical grid
            // and can converge to a low-mass nonlinear branch despite a tiny
            // algebraic residual.
            initialize_mapped_z4c_state_from_provider(
               *spacetime_state_.state()[0],
               hierarchy,
               0,
               cfg_,
               0.0_rt,
               provider
            );
         }
      }
      if (amrex::ParallelDescriptor::IOProcessor()) {
         const InitialDataCoordinateLine radial = hierarchy.coordinate_line(
            0, 0
         );
         amrex::Print()
            << "adaptive " << (black_hole_xcts ? "black-hole" : "stellar")
            << " ID radial_cells=" << radial.centers.size()
            << " inner_face=" << radial.faces.front()
            << " inner_dr=" << radial.widths.front()
            << " outer_dr=" << radial.widths.back() << '\n';
      }
   }
#endif
   ConformalConstraintFormulation formulation =
      ConformalConstraintFormulation::ctt;
   if (cfg_.spacetime_initial_data == "cttk") {
      formulation = ConformalConstraintFormulation::cttk;
   } else if (cfg_.spacetime_initial_data == "xcts") {
      formulation = ConformalConstraintFormulation::xcts;
   }
   const bool seed_conformal_from_spacetime = seed_from_loaded_id
      || (formulation == ConformalConstraintFormulation::xcts
          && initial_data_provider_->view().tov.active()
          && initial_data_provider_->view().fields.spacetime_from_id);
   ConformalContinuationObserver continuation_observer;
   const auto with_global_rotating_diagnostics = [this](
      RotatingStarUpdateSummary summary
   ) {
#if (AMREX_SPACEDIM == 3)
      if (cfg_.ic == "rotating_star"
          && cfg_.initial_data_angular_velocity != 0.0_rt
          && mesh_coordinate_kind() == MeshCartesian) {
         ScalarDiagnosticSelection selection = scalar_diagnostic_selection(
            "mass,adm_mass_volume,adm_angular_momentum3_volume"
         );
         apply_scalar_diagnostic_requirements(selection, cfg_);
         const ScalarDiagnostics diagnostics = compute_current_scalars(
            0,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            DiffusionStats{},
            eos_method_,
            selection
         );
         summary.adm_mass = diagnostics.adm_mass_volume;
         summary.baryonic_mass = diagnostics.mass;
         summary.angular_momentum =
            diagnostics.adm_angular_momentum3_volume;
         summary.dimensionless_spin = summary.angular_momentum / (
            summary.adm_mass * summary.adm_mass
         );
         if (!(summary.adm_mass > 0.0_rt)
             || !(summary.baryonic_mass > 0.0_rt)
             || !std::isfinite(summary.angular_momentum)
             || !std::isfinite(summary.dimensionless_spin)) {
            amrex::Abort("Cartesian rotating-star global diagnostic is invalid");
         }
      }
#endif
      return summary;
   };
   if (physical_initial_data_sequence_output_enabled()) {
      initialize_physical_initial_data_sequence_output();
      continuation_observer = [this, with_global_rotating_diagnostics](
         int continuation_step,
         int continuation_steps,
         amrex::Real stellar_angular_velocity,
         amrex::Real disk_angular_velocity,
         ConformalBlackHoleSummary const& black_hole,
         RotatingStarUpdateSummary const& rotating_star,
         EquilibriumTorusUpdateSummary const& equilibrium_torus
      ) {
         finalize_constraint_solved_initial_data_state();
         run_evolution_preflight(0.0_rt);
         const RotatingStarUpdateSummary persisted =
            with_global_rotating_diagnostics(rotating_star);
         write_physical_initial_data_sequence_model(
            continuation_step,
            continuation_steps,
            stellar_angular_velocity,
            disk_angular_velocity,
            black_hole,
            persisted,
            equilibrium_torus
         );
      };
   }
   RotatingStarParameters rotating_parameters =
      rotating_star_parameters(cfg_, initial_data_provider_->view());
   EquilibriumTorusParameters equilibrium_torus =
      equilibrium_torus_parameters(cfg_, initial_data_provider_->view());
   if (seed_from_loaded_id && loaded_physical_id_metadata_) {
      rotating_parameters.continuation_source_angular_velocity =
         loaded_physical_id_metadata_->angular_velocity;
      equilibrium_torus.continuation_source_angular_velocity =
         loaded_physical_id_metadata_->disk_angular_velocity;
      if (loaded_physical_id_metadata_->injection_energy > 0.0_rt) {
         rotating_parameters.injection_energy =
            loaded_physical_id_metadata_->injection_energy;
      }
   }
   PressurePoloidalMagneticSeedWorkspace magnetic_workspace;
   PressurePoloidalMagneticSeedSummary magnetic_summary;
   int magnetic_seed_updates = 0;
   ConformalMatterObserver matter_observer;
   PoloidalGradShafranovWorkspace magnetic_equilibrium_workspace;
   PoloidalGradShafranovSummary magnetic_equilibrium_summary;
   int magnetic_equilibrium_updates = 0;
   ConformalStellarMatterUpdater stellar_matter_updater;
   if (cfg_.initial_data_magnetic_seed == "pressure_poloidal") {
#if (AMREX_SPACEDIM != 3)
      amrex::Abort("pressure-poloidal magnetic seed requires 3D");
#else
      if (finestLevel() != 0 || maxLevel() != 0) {
         amrex::Abort(
            "pressure-poloidal magnetic seed currently requires a single "
            "uniform level; AMR needs one edge-potential prolongation owner"
         );
      }
      const PressurePoloidalMagneticSeedParameters magnetic_parameters{
         cfg_.initial_data_magnetic_seed_pressure_cutoff,
         cfg_.initial_data_magnetic_seed_exponent,
         cfg_.initial_data_magnetic_seed_maximum_comoving_field,
         {{
            cfg_.initial_data_rotation_axis1,
            cfg_.initial_data_rotation_axis2,
            cfg_.initial_data_rotation_axis3,
         }},
         {{
            cfg_.initial_data_rotation_center1,
            cfg_.initial_data_rotation_center2,
            cfg_.initial_data_rotation_center3,
         }},
      };
      matter_observer = [&, magnetic_parameters]() {
         magnetic_summary = apply_pressure_poloidal_magnetic_seed(
            face_b_[0],
            *primitive_[0],
            *spacetime_coupling_.cell_spacetime()[0],
            Geom(0),
            cfg_,
            eos_params(cfg_),
            magnetic_parameters,
            magnetic_workspace
         );
         ++magnetic_seed_updates;
      };
#endif
   }
   if (cfg_.initial_data_magnetic_equilibrium
          == "poloidal_grad_shafranov"
       && cfg_.initial_data_magnetic_equilibrium_poloidal_magnetization
          != 0.0_rt) {
#if (AMREX_SPACEDIM != 3)
      amrex::Abort("poloidal Grad-Shafranov equilibrium requires 3D");
#else
      if (finestLevel() != 0 || maxLevel() != 0) {
         amrex::Abort(
            "poloidal Grad-Shafranov equilibrium currently requires one "
            "uniform level"
         );
      }
      const PoloidalGradShafranovParameters magnetic_parameters{
         cfg_.initial_data_magnetic_equilibrium_poloidal_magnetization,
         cfg_.initial_data_magnetic_equilibrium_nonlinear_poloidal,
         cfg_.initial_data_magnetic_equilibrium_relaxation,
         {{
            cfg_.initial_data_rotation_axis1,
            cfg_.initial_data_rotation_axis2,
            cfg_.initial_data_rotation_axis3,
         }},
         {{
            cfg_.initial_data_rotation_center1,
            cfg_.initial_data_rotation_center2,
            cfg_.initial_data_rotation_center3,
         }},
      };
      stellar_matter_updater = [&, magnetic_parameters](
         RotatingStarParameters const& continued_rotating_star,
         bool update_fluid
      ) {
         magnetic_equilibrium_summary =
            update_poloidal_grad_shafranov_equilibrium(
               hierarchy,
               face_b_[0],
               *primitive_[0],
               *spacetime_coupling_.cell_spacetime()[0],
               cfg_,
               eos_method_,
               eos_params(cfg_),
               continued_rotating_star,
               magnetic_parameters,
               magnetic_equilibrium_workspace,
               update_fluid
            );
         ++magnetic_equilibrium_updates;
         if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
               << "poloidal Grad-Shafranov sweep="
               << magnetic_equilibrium_updates
               << " update_fluid=" << update_fluid
               << " relative_potential_update="
               << magnetic_equilibrium_summary.relative_potential_update
               << " equation_relative_residual="
               << magnetic_equilibrium_summary.equation_relative_residual
               << " bernoulli_residual="
               << magnetic_equilibrium_summary.bernoulli_residual
               << " maximum_comoving_field="
               << magnetic_equilibrium_summary.maximum_comoving_field
               << " divb_linf="
               << magnetic_equilibrium_summary.divergence_linf << '\n';
         }
         return ConformalStellarMatterUpdateResult{
            magnetic_equilibrium_summary.rotating_star,
            std::max(
               magnetic_equilibrium_summary.relative_potential_update,
               magnetic_equilibrium_summary.bernoulli_residual
            ),
         };
      };
#endif
   }
   ConformalMaterializationResult result = materialize_conformal_initial_data(
      hierarchy,
      primitive,
      spacetime,
      cell_spacetime,
      physics_system_,
      eos_method_,
      eos_params(cfg_),
      initial_data_provider_->view().fields,
      mesh_coordinate_kind(),
      formulation,
      {
         cfg_.ic == "black_hole",
         cfg_.spacetime_mass,
         cfg_.spacetime_dimensionless_spin3,
      },
      rotating_parameters,
      equilibrium_torus,
      seed_conformal_from_spacetime,
      continuation_observer,
      matter_observer,
      stellar_matter_updater,
      preserve_conformal_baryon_density
   );
   if (magnetic_seed_updates > 0) {
      amrex::Print()
         << "pressure-poloidal magnetic ID: updates=" << magnetic_seed_updates
         << " maximum_pressure=" << magnetic_summary.maximum_pressure
         << " normalization=" << magnetic_summary.normalization
         << " maximum_comoving_field="
         << magnetic_summary.maximum_comoving_field
         << " divb_linf=" << magnetic_summary.divergence_linf << '\n';
   }
   if (magnetic_equilibrium_updates > 0) {
      amrex::Print()
         << "poloidal Grad-Shafranov magnetic ID: updates="
         << magnetic_equilibrium_updates
         << " relative_potential_update="
         << magnetic_equilibrium_summary.relative_potential_update
         << " linear_residual="
         << magnetic_equilibrium_summary.linear_residual
         << " equation_relative_residual="
         << magnetic_equilibrium_summary.equation_relative_residual
         << " bernoulli_residual="
         << magnetic_equilibrium_summary.bernoulli_residual
         << " maximum_comoving_field="
         << magnetic_equilibrium_summary.maximum_comoving_field
         << " divb_linf="
         << magnetic_equilibrium_summary.divergence_linf << '\n';
   }
   finalize_constraint_solved_initial_data_state();
   if (magnetic_seed_updates > 0 || magnetic_equilibrium_updates > 0) {
      sync_rmhd_primitive_b_from_densitized_faces(
         *primitive_[0],
         face_b_[0],
         *spacetime_coupling_.cell_spacetime()[0]
      );
      // The first finalization establishes the authoritative cell-centered
      // metric from the completed nodal solve.  Rebuild the Valencia cache
      // once more after converting its densitized CT field to physical B^i.
      finalize_constraint_solved_initial_data_state();
   }
   result.rotating_star = with_global_rotating_diagnostics(
      result.rotating_star
   );
   if (cfg_.ic == "rotating_star") {
      initial_data_rotating_star_summary_ =
         std::make_unique<RotatingStarUpdateSummary>(result.rotating_star);
   }
   if (result.black_hole.throat_area > 0.0_rt) {
      initial_data_black_hole_throat_area_ = result.black_hole.throat_area;
      initial_data_black_hole_irreducible_mass_ =
         result.black_hole.irreducible_mass;
      initial_data_black_hole_throat_equatorial_circumference_ =
         result.black_hole.equatorial_circumference;
      initial_data_black_hole_throat_polar_circumference_ =
         result.black_hole.polar_circumference;
      initial_data_black_hole_throat_kerr_mass_ =
         result.black_hole.kerr_horizon_mass;
      initial_data_black_hole_throat_kerr_spin_ =
         result.black_hole.kerr_dimensionless_spin;
      initial_data_black_hole_throat_angular_momentum_ =
         result.black_hole.angular_momentum;
      initial_data_black_hole_throat_christodoulou_mass_ =
         result.black_hole.christodoulou_mass;
      initial_data_black_hole_throat_christodoulou_spin_ =
         result.black_hole.christodoulou_dimensionless_spin;
   }
   if (equilibrium_torus.active) {
      initial_data_equilibrium_torus_summary_ =
         std::make_unique<EquilibriumTorusUpdateSummary>(
            result.equilibrium_torus
         );
   }
   if (amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print()
         << cfg_.spacetime_initial_data
         << " initial data: matter_iterations=" << result.matter_iterations
         << " relative_matter_update=" << result.relative_matter_update
         << " coupled_iterations=" << result.constraints.coupled_iterations
         << " hamiltonian_residual=" << result.constraints.hamiltonian_residual
         << " lapse_residual=" << result.constraints.lapse_residual
         << " momentum_residual=" << result.constraints.momentum_residual << '\n';
      if (cfg_.ic == "rotating_star"
          && cfg_.initial_data_angular_velocity != 0.0_rt) {
         if (!(result.rotating_star.equatorial_radius > 0.0_rt)
             || !(result.rotating_star.polar_radius > 0.0_rt)) {
            amrex::Abort("rotating-star matter support has no resolved surface extent");
         }
         amrex::Print()
            << "rotating star: omega=" << cfg_.initial_data_angular_velocity
            << " maximum_density=" << result.rotating_star.maximum_density
            << " maximum_velocity=" << result.rotating_star.maximum_velocity
            << " injection_energy=" << result.rotating_star.injection_energy
            << " boundary_density_excess="
            << result.rotating_star.boundary_density_excess
            << " equatorial_radius=" << result.rotating_star.equatorial_radius
            << " polar_radius=" << result.rotating_star.polar_radius
            << " matter_outer_radius="
            << result.rotating_star.matter_outer_radius
            << " coordinate_axis_ratio="
            << result.rotating_star.equatorial_radius
               / result.rotating_star.polar_radius
            << " mass_shedding_indicator="
            << result.rotating_star.mass_shedding_indicator
            << " adm_mass=" << result.rotating_star.adm_mass
            << " baryonic_mass=" << result.rotating_star.baryonic_mass
            << " angular_momentum=" << result.rotating_star.angular_momentum
            << " dimensionless_spin="
            << result.rotating_star.dimensionless_spin
            << " equatorial_log_enthalpy_gradient="
            << result.rotating_star.equatorial_log_enthalpy_gradient
            << " polar_log_enthalpy_gradient="
            << result.rotating_star.polar_log_enthalpy_gradient
            << '\n';
      }
      if (result.black_hole.throat_area > 0.0_rt) {
         amrex::Print()
            << "black-hole throat: area=" << result.black_hole.throat_area
            << " irreducible_mass="
            << result.black_hole.irreducible_mass
            << " equatorial_circumference="
            << result.black_hole.equatorial_circumference
            << " polar_circumference="
            << result.black_hole.polar_circumference
            << " kerr_horizon_mass="
            << result.black_hole.kerr_horizon_mass
            << " kerr_dimensionless_spin="
            << result.black_hole.kerr_dimensionless_spin
            << " angular_momentum="
            << result.black_hole.angular_momentum
            << " christodoulou_mass="
            << result.black_hole.christodoulou_mass
            << " christodoulou_dimensionless_spin="
            << result.black_hole.christodoulou_dimensionless_spin << '\n';
      }
      if (equilibrium_torus.active) {
         amrex::Print()
            << "equilibrium torus: gravity="
            << cfg_.initial_data_disk_gravity
            << " rotation_law=" << cfg_.initial_data_disk_rotation_law
            << " specific_angular_momentum="
            << equilibrium_torus.specific_angular_momentum
            << " angular_velocity="
            << equilibrium_torus.angular_velocity
            << " differential_rotation_length="
            << equilibrium_torus.differential_rotation_length
            << " rotation_power_q="
            << equilibrium_torus.rotation_power_q
            << " requested_inner_radius=" << equilibrium_torus.inner_radius
            << " resolved_inner_radius="
            << result.equilibrium_torus.inner_radius
            << " outer_radius=" << result.equilibrium_torus.outer_radius
            << " density_maximum_radius="
            << result.equilibrium_torus.density_maximum_radius
            << " maximum_density="
            << result.equilibrium_torus.maximum_density
            << " maximum_velocity="
            << result.equilibrium_torus.maximum_velocity
            << " maximum_log_specific_enthalpy="
            << result.equilibrium_torus.maximum_log_specific_enthalpy
            << " baryonic_mass="
            << result.equilibrium_torus.baryonic_mass
            << " angular_momentum="
            << result.equilibrium_torus.angular_momentum << '\n';
      }
   }
   return true;
}

void AmrSolverBase::finalize_constraint_solved_initial_data_state()
{
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      spacetime_coupling_.invalidate_cell_spacetime(lev);
   }
   fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0_rt);
   if (initial_data_coordinate_faces_.empty()) {
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         complete_spacetime_connections(
            *spacetime_state_.state()[lev], Geom(lev), cfg_
         );
      }
   }
   fill_all_spacetime_boundaries(spacetime_state_.state(), 0.0_rt);
   prepare_spacetime_coupling_range(
      spacetime_state_.state(), 0, finestLevel() + 1, 0.0_rt, -1.0_rt, 0
   );
   amrex::Vector<amrex::Geometry> geometry;
   ConstPhysicalIdField primitive;
   ConstPhysicalIdField cell_spacetime;
   MutablePhysicalIdField fluid;
   geometry.reserve(finestLevel() + 1);
   primitive.reserve(finestLevel() + 1);
   cell_spacetime.reserve(finestLevel() + 1);
   fluid.reserve(finestLevel() + 1);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      geometry.push_back(Geom(lev));
      primitive.push_back(primitive_[lev].get());
      cell_spacetime.push_back(
         spacetime_coupling_.cell_spacetime()[lev].get()
      );
      fluid.push_back(fluid_state_.state()[lev].get());
   }
   materialize_physical_initial_data_fluid(
      cfg_, physics_system_, geometry, primitive, cell_spacetime, fluid
   );
   apply_stored_spacetime_fluid_boundaries_range(
      fluid_state_.state(), 0, finestLevel() + 1, 0.0_rt
   );
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      copy_level_state_to_old(lev);
   }
}

void AmrSolverBase::fill_new_spacetime_level_from_coarse(int lev, amrex::Real time)
{
   if (!spacetime_state_.active()) {
      return;
   }
   if (!spacetime_state_.evolves() || lev == 0 || time == 0.0_rt) {
      initialize_spacetime_level(lev, time);
      return;
   }
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev - 1, time);
   SpacetimeFillPatchPhysBC coarse_bc(
      Geom(lev - 1), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   SpacetimeFillPatchPhysBC fine_bc(
      Geom(lev), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   amrex::InterpFromCoarseLevel(
      *spacetime_state_.state()[lev],
      spacetime_state_.state()[lev]->nGrowVect(),
      time,
      *spacetime_state_.state()[lev - 1],
      0,
      0,
      spacetime_state_.num_components(),
      Geom(lev - 1),
      Geom(lev),
      coarse_bc,
      0,
      fine_bc,
      0,
      refRatio(lev - 1),
      static_cast<amrex::MFInterpolater*>(&spacetime_interpolater_),
      spacetime_fillpatch_bcs_,
      0
   );
   enforce_reflection_odd_nodal_plane_values(
      *spacetime_state_.state()[lev], Geom(lev), spacetime_fillpatch_bcs_
   );
}

void AmrSolverBase::fill_remade_spacetime_level(
   int lev,
   amrex::Real time,
   amrex::MultiFab& old_state
)
{
   if (!spacetime_state_.active()) {
      return;
   }
   if (!spacetime_state_.evolves()) {
      initialize_spacetime_level(lev, time);
      return;
   }
   if (lev == 0) {
      spacetime_state_.state()[0]->ParallelCopy(
         old_state,
         0,
         0,
         spacetime_state_.num_components(),
         0,
         0,
         Geom(0).periodicity()
      );
      fill_spacetime_level_boundaries(spacetime_state_.state(), 0, time);
      return;
   }
   fill_spacetime_level_boundaries(spacetime_state_.state(), lev - 1, time);
   amrex::Vector<amrex::MultiFab*> coarse_data{
      spacetime_state_.state()[lev - 1].get()
   };
   amrex::Vector<amrex::Real> coarse_time{time};
   amrex::Vector<amrex::MultiFab*> fine_data{&old_state};
   amrex::Vector<amrex::Real> fine_time{time};
   SpacetimeFillPatchPhysBC coarse_bc(
      Geom(lev - 1), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   SpacetimeFillPatchPhysBC fine_bc(
      Geom(lev), spacetime_fillpatch_bcs_, spacetime_fillpatch_func_
   );
   fill_nodal_two_levels(
      *spacetime_state_.state()[lev],
      spacetime_state_.state()[lev]->nGrowVect(),
      time,
      coarse_data,
      coarse_time,
      fine_data,
      fine_time,
      Geom(lev - 1),
      Geom(lev),
      coarse_bc,
      fine_bc,
      refRatio(lev - 1),
      spacetime_fillpatch_bcs_,
      spacetime_state_.num_components(),
      &spacetime_interpolater_
   );
   enforce_reflection_odd_nodal_plane_values(
      *spacetime_state_.state()[lev], Geom(lev), spacetime_fillpatch_bcs_
   );
}

void AmrSolverBase::average_down_spacetime_level(int fine_lev)
{
   if (spacetime_state_.evolves()) {
      spacetime_state_.average_down_level(fine_lev, refRatio(fine_lev - 1));
      spacetime_coupling_.invalidate_cell_spacetime(fine_lev - 1);
   }
}

void AmrSolverBase::average_down_spacetime()
{
   for (int fine_lev = finestLevel(); fine_lev > 0; --fine_lev) {
      average_down_spacetime_level(fine_lev);
   }
}

SpacetimeConstraintDiagnostics
AmrSolverBase::compute_spacetime_constraint_diagnostics(amrex::Real time, bool include_regions)
{
   SpacetimeConstraintDiagnostics diagnostics;
   if (!spacetime_state_.active()) {
      return diagnostics;
   }

   fill_all_spacetime_boundaries(spacetime_state_.state(), time);
   amrex::Real hamiltonian_squared_integral = 0.0_rt;
   amrex::Real composite_volume = 0.0_rt;
   amrex::Real interior_hamiltonian_squared_integral = 0.0_rt;
   amrex::Real interior_volume = 0.0_rt;
   amrex::Real cf_hamiltonian_squared_integral = 0.0_rt;
   amrex::Real cf_volume = 0.0_rt;
   amrex::Real bh_exterior_hamiltonian_squared_integral = 0.0_rt;
   amrex::Real bh_exterior_volume = 0.0_rt;
   const SpacetimeConstraintRegion region{
      {{cfg_.spacetime_origin1, cfg_.spacetime_origin2, cfg_.spacetime_origin3}},
      cfg_.diagnostic_constraint_region_extent,
      cfg_.diagnostic_constraint_exclusion_radius,
      cfg_.diagnostic_constraint_split_radius};
   std::array<amrex::Real, 3> regional_hamiltonian_squared_integral{};
   const FixedSpacetimeParameters black_hole =
      fixed_spacetime_parameters(cfg_);
   const bool has_black_hole =
      black_hole.metric == FixedSpacetimeMetric::KerrSchild
      || black_hole.metric == FixedSpacetimeMetric::StaticTrumpet;
   const amrex::Real horizon_radius = has_black_hole
      ? black_hole_coordinate_horizon_radius(black_hole) : 0.0_rt;
   const amrex::Real horizon_radius_squared =
      horizon_radius * horizon_radius;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      amrex::MultiFab const& state = *spacetime_state_.state()[lev];
      amrex::iMultiFab valid(state.boxArray(), state.DistributionMap(), 1, 0);
      if (lev < finestLevel()) {
         const amrex::BoxArray fine_nodes = amrex::convert(
            boxArray(lev + 1), amrex::IntVect::TheNodeVector()
         );
         valid = amrex::makeFineMask(
            state,
            fine_nodes,
            refRatio(lev),
            Geom(lev).periodicity(),
            1,
            0
         );
      } else {
         valid.setVal(1);
      }
      const std::unique_ptr<amrex::iMultiFab> owner = state.OwnerMask(
         Geom(lev).periodicity()
      );
      const int derivative_order = cfg_.spacetime_derivative_order;
      const int stencil_radius = centered_stencil_radius(derivative_order);
      amrex::iMultiFab stencil_valid(
         state.boxArray(), state.DistributionMap(), 1, stencil_radius
      );
      stencil_valid.setVal(0);
      stencil_valid.ParallelCopy(valid, 0, 0, 1, 0, 0);
      stencil_valid.FillBoundary(Geom(lev).periodicity());
      fill_reflected_nodal_validity_ghosts(
         stencil_valid, Geom(lev), cfg_, stencil_radius
      );
      amrex::MultiFab norms(
         state.boxArray(), state.DistributionMap(), include_regions ? 23 : 14, 0
      );
      norms.setVal(0.0_rt);
      const auto excision = excision_grid_view(spacetime_excision_region(cfg_), Geom(lev), state.ixType());
      const amrex::Box nodal_domain = amrex::convert(
         Geom(lev).Domain(), amrex::IntVect::TheNodeVector()
      );
      const auto spacing = Geom(lev).CellSizeArray();
      const auto lower = Geom(lev).ProbLoArray();
      const amrex::GpuArray<int, 3> domain_lower{{
         nodal_domain.smallEnd(0),
#if (AMREX_SPACEDIM >= 2)
         nodal_domain.smallEnd(1),
#else
         0,
#endif
#if (AMREX_SPACEDIM == 3)
         nodal_domain.smallEnd(2),
#else
         0,
#endif
      }};
      const bool use_fccz4 = cfg_.spacetime_formulation == "fccz4";
      const ReferenceChartGeometry reference_chart(Geom(lev), cfg_);
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
         1.0_rt / spacing[0],
#if (AMREX_SPACEDIM >= 2)
         1.0_rt / spacing[1],
#else
         1.0_rt,
#endif
#if (AMREX_SPACEDIM == 3)
         1.0_rt / spacing[2],
#else
         1.0_rt,
#endif
      }};
      // Each launch owns one formulation's equations. Keep accuracy-order
      // dispatch in the derivative operator, not around the equation kernel.
      const auto compute_norms = [&](auto formulation_tag) {
         constexpr bool reference_metric_formulation = decltype(formulation_tag)::value;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box box = mfi.tilebox();
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
               if (!Geom(lev).isPeriodic(direction)
                   && !spacetime_boundary_reflects(cfg_, direction, false)
                   && box.smallEnd(direction) == nodal_domain.smallEnd(direction)) {
                  box.setSmall(direction, box.smallEnd(direction) + stencil_radius);
               }
               if (!Geom(lev).isPeriodic(direction)
                   && !spacetime_boundary_reflects(cfg_, direction, true)
                   && box.bigEnd(direction) == nodal_domain.bigEnd(direction)) {
                  box.setBig(direction, box.bigEnd(direction) - stencil_radius);
               }
            }
            const auto input = state.const_array(mfi);
            const auto output = norms.array(mfi);
            const auto valid_points = valid.const_array(mfi);
            const auto same_level_points = stencil_valid.const_array(mfi);
            const auto owned_points = owner->const_array(mfi);
            const bool matter_coupled = spacetime_includes_matter(cfg_);
            const auto matter = spacetime_coupling_.vertex_matter()[lev]->const_array(mfi);
            fill_constraint_norms<reference_metric_formulation>(
               box, input, matter, output, valid_points, same_level_points,
               owned_points, excision, matter_coupled, stencil_radius,
               derivative_order, inv_spacing, reference_chart, include_regions,
               region, has_black_hole, black_hole, horizon_radius_squared
            );
         }
      };
      if (use_fccz4) {
         compute_norms(std::true_type{});
      } else {
         compute_norms(std::false_type{});
      }
      amrex::Vector<int> maximum_components{
         0, 7, 9, 11, 13, 1, 2, 3, 4, 5,
      };
      if (include_regions) {
         for (int region_index = 0; region_index < 3; ++region_index) {
            maximum_components.push_back(14 + 3 * region_index);
            maximum_components.push_back(15 + 3 * region_index);
         }
      }
      const amrex::Vector<amrex::Real> level_maxima = norms.norm0(
         maximum_components, 0, false
      );
      diagnostics.hamiltonian_linf = amrex::max(
         diagnostics.hamiltonian_linf, level_maxima[0]
      );
      const amrex::Real level_volume = cartesian_cell_volume(Geom(lev));
      if (reference_chart.coordinate != MeshCartesian) {
         // Maxima above use unweighted point values. L2 norms below use
         // the same reference-volume measure as the cell/face budgets,
         // with dual nodal cells clipped at the physical domain boundary.
         for (amrex::MFIter mfi(norms, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto values = norms.array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real volume_ratio = reference_chart.nodal_volume(i, j, k) / level_volume;
               const amrex::Real norm_weight = std::sqrt(volume_ratio);
               for (int component : {0, 7, 9, 11}) values(i, j, k, component) *= norm_weight;
               for (int component : {6, 8, 10, 12}) values(i, j, k, component) *= volume_ratio;
               if (include_regions) {
                  for (int region_index = 0; region_index < 3; ++region_index) {
                     values(i, j, k, 14 + 3 * region_index) *= norm_weight;
                     values(i, j, k, 16 + 3 * region_index) *= volume_ratio;
                  }
               }
            });
         }
      }
      const amrex::Real level_hamiltonian_norm = norms.norm2(0);
      hamiltonian_squared_integral += level_volume
         * level_hamiltonian_norm * level_hamiltonian_norm;
      composite_volume += level_volume * norms.sum(6, true);
      diagnostics.hamiltonian_interior_linf = amrex::max(
         diagnostics.hamiltonian_interior_linf, level_maxima[1]
      );
      const amrex::Real level_interior_norm = norms.norm2(7);
      interior_hamiltonian_squared_integral += level_volume
         * level_interior_norm * level_interior_norm;
      interior_volume += level_volume * norms.sum(8, true);
      diagnostics.hamiltonian_cf_linf = amrex::max(
         diagnostics.hamiltonian_cf_linf, level_maxima[2]
      );
      const amrex::Real level_cf_norm = norms.norm2(9);
      cf_hamiltonian_squared_integral += level_volume
         * level_cf_norm * level_cf_norm;
      cf_volume += level_volume * norms.sum(10, true);
      diagnostics.hamiltonian_bh_exterior_linf = amrex::max(
         diagnostics.hamiltonian_bh_exterior_linf,
         level_maxima[3]
      );
      const amrex::Real level_bh_exterior_norm = norms.norm2(11);
      bh_exterior_hamiltonian_squared_integral += level_volume
         * level_bh_exterior_norm * level_bh_exterior_norm;
      bh_exterior_volume += level_volume * norms.sum(12, true);
      diagnostics.momentum_bh_exterior_linf = amrex::max(
         diagnostics.momentum_bh_exterior_linf,
         std::sqrt(level_maxima[4])
      );
      diagnostics.momentum_linf = amrex::max(
         diagnostics.momentum_linf, std::sqrt(level_maxima[5])
      );
      diagnostics.z_linf = amrex::max(
         diagnostics.z_linf, std::sqrt(level_maxima[6])
      );
      diagnostics.theta_linf = amrex::max(
         diagnostics.theta_linf, level_maxima[7]
      );
      diagnostics.log_conformal_determinant_linf = amrex::max(
         diagnostics.log_conformal_determinant_linf,
         level_maxima[8]
      );
      diagnostics.trace_a_tilde_linf = amrex::max(
         diagnostics.trace_a_tilde_linf, level_maxima[9]
      );
      if (include_regions) {
         for (int region_index = 0; region_index < 3; ++region_index) {
            const int first_component = 14 + 3 * region_index;
            diagnostics.regional_hamiltonian_linf[region_index] = amrex::max(
               diagnostics.regional_hamiltonian_linf[region_index],
               level_maxima[10 + 2 * region_index]);
            diagnostics.regional_momentum_linf[region_index] = amrex::max(
               diagnostics.regional_momentum_linf[region_index],
               std::sqrt(level_maxima[11 + 2 * region_index]));
            const amrex::Real level_norm = norms.norm2(first_component);
            regional_hamiltonian_squared_integral[region_index] += level_volume * level_norm * level_norm;
            diagnostics.regional_volume[region_index] += level_volume * norms.sum(first_component + 2, true);
         }
      }
   }
   std::array<amrex::Real, 4> diagnostic_volumes{
      composite_volume,
      interior_volume,
      cf_volume,
      bh_exterior_volume,
   };
   amrex::ParallelDescriptor::ReduceRealSum(
      diagnostic_volumes.data(), static_cast<int>(diagnostic_volumes.size())
   );
   composite_volume = diagnostic_volumes[0];
   interior_volume = diagnostic_volumes[1];
   cf_volume = diagnostic_volumes[2];
   bh_exterior_volume = diagnostic_volumes[3];
   if (composite_volume > 0.0_rt) {
      diagnostics.hamiltonian_l2 = std::sqrt(
         hamiltonian_squared_integral / composite_volume
      );
   }
   if (interior_volume > 0.0_rt) {
      diagnostics.hamiltonian_interior_l2 = std::sqrt(
         interior_hamiltonian_squared_integral / interior_volume
      );
   }
   if (cf_volume > 0.0_rt) {
      diagnostics.hamiltonian_cf_l2 = std::sqrt(
         cf_hamiltonian_squared_integral / cf_volume
      );
   }
   if (bh_exterior_volume > 0.0_rt) {
      diagnostics.hamiltonian_bh_exterior_l2 = std::sqrt(
         bh_exterior_hamiltonian_squared_integral / bh_exterior_volume
      );
   }
   if (include_regions) {
      amrex::ParallelDescriptor::ReduceRealSum(diagnostics.regional_volume.data(), 3);
      for (int region_index = 0; region_index < 3; ++region_index) {
         if (!(diagnostics.regional_volume[region_index] > 0.0_rt)) {
            amrex::Abort("fixed constraint region or near/far partition contains no owned samples; "
               "adjust the physical region or resolution");
         }
         diagnostics.regional_hamiltonian_l2[region_index] = std::sqrt(
            regional_hamiltonian_squared_integral[region_index] / diagnostics.regional_volume[region_index]);
      }
   }
   return diagnostics;
}

SpacetimeGlobalDiagnostics
AmrSolverBase::compute_spacetime_global_diagnostics(
   amrex::Real time,
   bool surface_requested,
   bool volume_requested
)
{
   SpacetimeGlobalDiagnostics diagnostics;
   if (!spacetime_state_.active()) {
      return diagnostics;
   }
#if (AMREX_SPACEDIM != 3)
   amrex::Abort("global_gr diagnostics currently require a three-dimensional spacetime hierarchy");
#else
   if (mesh_coordinate_kind() != MeshCartesian || mixed_patch_system_enabled()) {
      amrex::Abort("global_gr diagnostics currently require a single Cartesian patch");
   }
   for (int direction = 0; direction < 3; ++direction) {
      if (Geom(0).isPeriodic(direction)) {
         amrex::Abort("global_gr diagnostics require nonperiodic asymptotic boundaries");
      }
   }

   fill_all_spacetime_boundaries(spacetime_state_.state(), time);
   enum IntegralComponent : int {
      AdmMassVolume = 0,
      AdmMomentum1Volume,
      AdmMomentum2Volume,
      AdmMomentum3Volume,
      AdmAngularMomentum1Volume,
      AdmAngularMomentum2Volume,
      AdmAngularMomentum3Volume,
      ProperMassVolume,
      KomarMassVolume,
      Count,
   };
   std::array<amrex::Real, IntegralComponent::Count> volume_integrals{};
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      amrex::MultiFab const& state = *spacetime_state_.state()[lev];
      if (!volume_requested) {
         continue;
      }
      amrex::iMultiFab valid(state.boxArray(), state.DistributionMap(), 1, 0);
      if (lev < finestLevel()) {
         const amrex::BoxArray fine_nodes = amrex::convert(
            boxArray(lev + 1), amrex::IntVect::TheNodeVector()
         );
         valid = amrex::makeFineMask(
            state, fine_nodes, refRatio(lev), Geom(lev).periodicity(), 1, 0
         );
      } else {
         valid.setVal(1);
      }
      const std::unique_ptr<amrex::iMultiFab> owner = state.OwnerMask(
         Geom(lev).periodicity()
      );
      amrex::MultiFab integrals(
         state.boxArray(), state.DistributionMap(), IntegralComponent::Count, 0
      );
      integrals.setVal(0.0_rt);
      const amrex::Box nodal_domain = amrex::convert(
         Geom(lev).Domain(), amrex::IntVect::TheNodeVector()
      );
      const auto spacing = Geom(lev).CellSizeArray();
      const auto lower = Geom(lev).ProbLoArray();
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
         1.0_rt / spacing[0], 1.0_rt / spacing[1], 1.0_rt / spacing[2]
      }};
      const amrex::GpuArray<int, 3> domain_lower{{
         nodal_domain.smallEnd(0),
         nodal_domain.smallEnd(1),
         nodal_domain.smallEnd(2),
      }};
      const amrex::GpuArray<int, 3> domain_upper{{
         nodal_domain.bigEnd(0),
         nodal_domain.bigEnd(1),
         nodal_domain.bigEnd(2),
      }};
      const amrex::Real cell_volume = spacing[0] * spacing[1] * spacing[2];
      const bool matter_coupled = spacetime_includes_matter(cfg_);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto input = state.const_array(mfi);
         const auto output = integrals.array(mfi);
         const auto valid_points = valid.const_array(mfi);
         const auto owned_points = owner->const_array(mfi);
         const auto matter = spacetime_coupling_.vertex_matter()[lev]->const_array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (valid_points(i, j, k, 0) == 0 || owned_points(i, j, k, 0) == 0) {
               return;
            }

            const int indices[3] = {i, j, k};
            const Z4cSpatialDerivatives derivatives =
               gather_z4c_geometry_derivatives<4, 3>(
                  input,
                  i,
                  j,
                  k,
                  inv_spacing,
                  domain_lower,
                  domain_upper
               );
            const Z4cState point_state = load_z4c_state(input, i, j, k);
            const Vector3 position{
               lower[0] + static_cast<amrex::Real>(i - nodal_domain.smallEnd(0)) * spacing[0],
               lower[1] + static_cast<amrex::Real>(j - nodal_domain.smallEnd(1)) * spacing[1],
               lower[2] + static_cast<amrex::Real>(k - nodal_domain.smallEnd(2)) * spacing[2],
            };
            const Z4cMatter point_matter = matter_coupled
               ? load_eulerian_matter(matter, i, j, k)
               : vacuum_z4c_matter();
            const AdmVolumeIntegrands point_volume =
               generated::adm_constraint_volume_integrands(
                  point_state, derivatives, point_matter, position
               );
            amrex::Real volume_weight = cell_volume;
            for (int direction = 0; direction < 3; ++direction) {
               if (indices[direction] == nodal_domain.smallEnd(direction)
                   || indices[direction] == nodal_domain.bigEnd(direction)) {
                  volume_weight *= 0.5_rt;
               }
            }
            output(i, j, k, IntegralComponent::AdmMassVolume) =
               volume_weight * point_volume.mass;
            output(i, j, k, IntegralComponent::AdmMomentum1Volume) =
               volume_weight * point_volume.linear_momentum1;
            output(i, j, k, IntegralComponent::AdmMomentum2Volume) =
               volume_weight * point_volume.linear_momentum2;
            output(i, j, k, IntegralComponent::AdmMomentum3Volume) =
               volume_weight * point_volume.linear_momentum3;
            output(i, j, k, IntegralComponent::AdmAngularMomentum1Volume) =
               volume_weight * point_volume.angular_momentum1;
            output(i, j, k, IntegralComponent::AdmAngularMomentum2Volume) =
               volume_weight * point_volume.angular_momentum2;
            output(i, j, k, IntegralComponent::AdmAngularMomentum3Volume) =
               volume_weight * point_volume.angular_momentum3;
            output(i, j, k, IntegralComponent::ProperMassVolume) =
               volume_weight * point_volume.proper_mass;
            output(i, j, k, IntegralComponent::KomarMassVolume) =
               volume_weight * point_volume.komar_mass;
         });
      }
      for (int component = 0; component < IntegralComponent::Count; ++component) {
         volume_integrals[component] += integrals.sum(component, true);
      }
   }
   if (volume_requested) {
      amrex::ParallelDescriptor::ReduceRealSum(
         volume_integrals.data(), static_cast<int>(volume_integrals.size())
      );
      diagnostics.adm_mass_volume = volume_integrals[IntegralComponent::AdmMassVolume];
      diagnostics.adm_momentum_volume = {
         volume_integrals[IntegralComponent::AdmMomentum1Volume],
         volume_integrals[IntegralComponent::AdmMomentum2Volume],
         volume_integrals[IntegralComponent::AdmMomentum3Volume],
      };
      diagnostics.adm_angular_momentum_volume = {
         volume_integrals[IntegralComponent::AdmAngularMomentum1Volume],
         volume_integrals[IntegralComponent::AdmAngularMomentum2Volume],
         volume_integrals[IntegralComponent::AdmAngularMomentum3Volume],
      };
      diagnostics.proper_mass_volume = volume_integrals[IntegralComponent::ProperMassVolume];
      diagnostics.komar_mass_volume = volume_integrals[IntegralComponent::KomarMassVolume];
      const amrex::Real volume_factor = symmetry_volume_factor();
      diagnostics.adm_mass_volume *= volume_factor;
      diagnostics.proper_mass_volume *= volume_factor;
      diagnostics.komar_mass_volume *= volume_factor;
      for (int axis = 0; axis < 3; ++axis) {
         diagnostics.adm_momentum_volume[axis] *= symmetry_momentum_factor(
            axis, volume_factor
         );
         diagnostics.adm_angular_momentum_volume[axis] *=
            symmetry_angular_momentum_factor(axis, volume_factor);
      }
   }
   if (surface_requested) {
      if (symmetry_volume_factor() != 1.0_rt) {
         amrex::Abort(
            "spherical GR surface diagnostics do not yet reconstruct omitted symmetry sectors"
         );
      }
      enum SurfaceFluxComponent : int {
         AdmMassFlux1 = 0,
         AdmMassFlux2,
         AdmMassFlux3,
         KomarMassFlux1,
         KomarMassFlux2,
         KomarMassFlux3,
         AdmMomentumFlux11,
         AdmMomentumFlux12,
         AdmMomentumFlux13,
         AdmMomentumFlux21,
         AdmMomentumFlux22,
         AdmMomentumFlux23,
         AdmMomentumFlux31,
         AdmMomentumFlux32,
         AdmMomentumFlux33,
         AdmAngularMomentumFlux11,
         AdmAngularMomentumFlux12,
         AdmAngularMomentumFlux13,
         AdmAngularMomentumFlux21,
         AdmAngularMomentumFlux22,
         AdmAngularMomentumFlux23,
         AdmAngularMomentumFlux31,
         AdmAngularMomentumFlux32,
         AdmAngularMomentumFlux33,
         SurfaceFluxCount,
      };
      LevelData surface_flux_hierarchy(finestLevel() + 1);
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         amrex::MultiFab const& state = *spacetime_state_.state()[lev];
         auto flux = std::make_unique<amrex::MultiFab>(
            state.boxArray(), state.DistributionMap(), SurfaceFluxCount, 1
         );
         flux->setVal(0.0_rt);
         const amrex::Box nodal_domain = amrex::convert(
            Geom(lev).Domain(), amrex::IntVect::TheNodeVector()
         );
         const auto spacing = Geom(lev).CellSizeArray();
         const auto lower = Geom(lev).ProbLoArray();
         const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
            1.0_rt / spacing[0], 1.0_rt / spacing[1], 1.0_rt / spacing[2]
         }};
         const amrex::GpuArray<int, 3> domain_lower{{
            nodal_domain.smallEnd(0),
            nodal_domain.smallEnd(1),
            nodal_domain.smallEnd(2),
         }};
         const amrex::GpuArray<int, 3> domain_upper{{
            nodal_domain.bigEnd(0),
            nodal_domain.bigEnd(1),
            nodal_domain.bigEnd(2),
         }};
         for (amrex::MFIter mfi(state, false); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto input = state.const_array(mfi);
            const auto output = flux->array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const Z4cSpatialDerivatives derivatives =
                  gather_z4c_geometry_derivatives<4, 3>(
                     input, i, j, k, inv_spacing, domain_lower, domain_upper
                  );
               const Z4cState point_state = load_z4c_state(input, i, j, k);
               const Vector3 position{
                  lower[0] + static_cast<amrex::Real>(i - nodal_domain.smallEnd(0)) * spacing[0],
                  lower[1] + static_cast<amrex::Real>(j - nodal_domain.smallEnd(1)) * spacing[1],
                  lower[2] + static_cast<amrex::Real>(k - nodal_domain.smallEnd(2)) * spacing[2],
               };
               const AdmFluxIntegrands point_flux = generated::adm_flux_integrands(
                  point_state, derivatives, position
               );
               output(i, j, k, AdmMassFlux1) = point_flux.mass1;
               output(i, j, k, AdmMassFlux2) = point_flux.mass2;
               output(i, j, k, AdmMassFlux3) = point_flux.mass3;
               output(i, j, k, KomarMassFlux1) = point_flux.komar_mass1;
               output(i, j, k, KomarMassFlux2) = point_flux.komar_mass2;
               output(i, j, k, KomarMassFlux3) = point_flux.komar_mass3;
               output(i, j, k, AdmMomentumFlux11) = point_flux.linear_momentum11;
               output(i, j, k, AdmMomentumFlux12) = point_flux.linear_momentum12;
               output(i, j, k, AdmMomentumFlux13) = point_flux.linear_momentum13;
               output(i, j, k, AdmMomentumFlux21) = point_flux.linear_momentum21;
               output(i, j, k, AdmMomentumFlux22) = point_flux.linear_momentum22;
               output(i, j, k, AdmMomentumFlux23) = point_flux.linear_momentum23;
               output(i, j, k, AdmMomentumFlux31) = point_flux.linear_momentum31;
               output(i, j, k, AdmMomentumFlux32) = point_flux.linear_momentum32;
               output(i, j, k, AdmMomentumFlux33) = point_flux.linear_momentum33;
               output(i, j, k, AdmAngularMomentumFlux11) = point_flux.angular_momentum11;
               output(i, j, k, AdmAngularMomentumFlux12) = point_flux.angular_momentum12;
               output(i, j, k, AdmAngularMomentumFlux13) = point_flux.angular_momentum13;
               output(i, j, k, AdmAngularMomentumFlux21) = point_flux.angular_momentum21;
               output(i, j, k, AdmAngularMomentumFlux22) = point_flux.angular_momentum22;
               output(i, j, k, AdmAngularMomentumFlux23) = point_flux.angular_momentum23;
               output(i, j, k, AdmAngularMomentumFlux31) = point_flux.angular_momentum31;
               output(i, j, k, AdmAngularMomentumFlux32) = point_flux.angular_momentum32;
               output(i, j, k, AdmAngularMomentumFlux33) = point_flux.angular_momentum33;
            });
         }
         surface_flux_hierarchy[lev] = std::move(flux);
      }
      amrex::Vector<amrex::MultiFab*> surface_flux_fields;
      surface_flux_fields.reserve(surface_flux_hierarchy.size());
      for (auto const& flux : surface_flux_hierarchy) {
         surface_flux_fields.push_back(flux.get());
      }
      amrex::FillBoundary(surface_flux_fields, Geom(0).periodicity());

      const ParametricSurface surface = diagnostic_extraction_surface();
      const HierarchyInterpolationPlan plan =
         diagnostic_surface_interpolation_plan(surface);
      const std::vector<amrex::Real> flux = plan.interpolate(
         surface_flux_hierarchy, 0, SurfaceFluxCount
      );
      const std::vector<amrex::Real> integrals = surface.integrate_normal_fluxes(
         flux, SurfaceFluxCount / 3
      );
      diagnostics.adm_mass_surface = integrals[0];
      diagnostics.komar_mass_surface = integrals[1];
      for (int axis = 0; axis < 3; ++axis) {
         diagnostics.adm_momentum_surface[axis] = integrals[2 + axis];
         diagnostics.adm_angular_momentum_surface[axis] = integrals[5 + axis];
      }
   }
#endif
   return diagnostics;
}

} // namespace pyfluid_amrex
