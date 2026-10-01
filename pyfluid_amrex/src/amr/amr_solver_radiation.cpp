#include "pyfluid_amrex/metric_reductions.hpp"
#include "amr/amr_solver_base.hpp"

#include <AMReX_BCUtil.H>
#include <AMReX_MultiFab.H>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/apparent_horizon_tracker.hpp"
#include "pyfluid_amrex/azimuthal_geometry.hpp"
#include "pyfluid_amrex/generated/diagnostic_catalog.hpp"
#include "pyfluid_amrex/hierarchy_interpolation.hpp"
#include "pyfluid_amrex/kernels_generated/diagnostics.hpp"
#include "pyfluid_amrex/kernels_generated/z4c.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/radiation.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/eulerian_matter_access.hpp"
#include "pyfluid_amrex/surface_geometry.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

std::string supported_azimuthal_mode_fields(
   PhysicalSystem system,
   int passive_scalars,
   bool spacetime_active
)
{
   std::ostringstream names;
   names << "rho, pressure";
   const auto append = [&](GridFieldSetInfo const& field_set) {
      for (GridFieldInfo const& field : field_set.fields) {
         if (field.value_kind == GridFieldValueKind::Scalar
             && field.components.size() == 1
             && field.name != "rho") {
            names << ", " << field.name;
         }
      }
   };
   append(fluid_state_grid_fields(system, passive_scalars));
   if (spacetime_active) {
      append(z4c_state_grid_fields());
   }
   return names.str();
}

} // namespace

ParametricSurface AmrSolverBase::diagnostic_extraction_surface() const
{
   const amrex::GpuArray<amrex::Real, 3> center{{
      cfg_.diagnostic_surface_center1,
      cfg_.diagnostic_surface_center2,
      cfg_.diagnostic_surface_center3,
   }};
   amrex::Real radius = cfg_.diagnostic_surface_radius;
   if (radius < 0.0_rt) {
      const auto lower = Geom(0).ProbLoArray();
      const auto upper = Geom(0).ProbHiArray();
      radius = 0.8_rt * amrex::min(
         amrex::min(center[0] - lower[0], upper[0] - center[0]),
         amrex::min(
            amrex::min(center[1] - lower[1], upper[1] - center[1]),
            amrex::min(center[2] - lower[2], upper[2] - center[2])
         )
      );
   }
   return ParametricSurface::sphere(
      center, radius, cfg_.diagnostic_surface_polar_points
   );
}

ParametricSurface AmrSolverBase::diagnostic_apparent_horizon_surface() const
{
   ApparentHorizonResult const* horizon = apparent_horizon_tracker_->result();
   if (horizon == nullptr) {
      amrex::Abort(
         "apparent-horizon surface diagnostics require a successful horizon search"
      );
   }
   ApparentHorizonSurface const& source = horizon->surface;
   const auto& source_center = source.center();
   return ParametricSurface::radial_graph(
      {{source_center[0], source_center[1], source_center[2]}},
      source.num_theta(),
      source.num_phi(),
      source.radius()
   );
}

HierarchyInterpolationPlan AmrSolverBase::diagnostic_point_interpolation_plan(
   std::vector<amrex::GpuArray<amrex::Real, 3>> positions,
   amrex::IndexType centering,
   std::array<int, 3> const& reflection_lo,
   std::array<int, 3> const& reflection_hi
) const
{
   std::vector<amrex::Geometry> geometry;
   std::vector<amrex::BoxArray> grids;
   std::vector<amrex::DistributionMapping> distribution;
   geometry.reserve(finestLevel() + 1);
   grids.reserve(finestLevel() + 1);
   distribution.reserve(finestLevel() + 1);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      geometry.push_back(Geom(lev));
      grids.push_back(boxArray(lev));
      distribution.push_back(DistributionMap(lev));
   }
   return HierarchyInterpolationPlan(
      std::move(positions),
      mesh_coordinate_kind(),
      active_mesh_axes(),
      geometry,
      grids,
      distribution,
      centering,
      reflection_lo,
      reflection_hi
   );
}

HierarchyInterpolationPlan AmrSolverBase::diagnostic_surface_interpolation_plan(
   ParametricSurface const& surface,
   amrex::IndexType centering
) const
{
   std::vector<amrex::GpuArray<amrex::Real, 3>> positions;
   positions.reserve(surface.points().size());
   for (SurfaceQuadraturePoint const& point : surface.points()) {
      positions.push_back(point.position);
   }
   return diagnostic_point_interpolation_plan(
      std::move(positions), centering
   );
}

amrex::Real AmrSolverBase::compute_baryon_surface_accretion_rate(
   ParametricSurface const& surface,
   amrex::Real time
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::Abort("baryon surface flux currently requires the 3D executable");
   return 0.0_rt;
#else
   if (!physical_system_is_relativistic(physics_system_)
       || !spacetime_state_.active()) {
      amrex::Abort(
         "baryon surface flux requires relativistic matter and stored spacetime"
      );
   }
   if (mesh_coordinate_kind() != MeshCartesian
       || mixed_patch_system_enabled()) {
      amrex::Abort(
         "baryon surface flux currently requires a single Cartesian patch"
      );
   }

   prepare_spacetime_coupling_range(
      spacetime_state_.state(), 0, finestLevel() + 1, time, -1.0_rt, 0
   );
   LevelData flux_hierarchy(finestLevel() + 1);
   std::array<int, 3> reflection_lo{};
   std::array<int, 3> reflection_hi{};
   for (int direction = 0; direction < 3; ++direction) {
      reflection_lo[direction] = reflective_boundary(
         side_boundary_kind(cfg_, direction, false)
      ) ? 1 : 0;
      reflection_hi[direction] = reflective_boundary(
         side_boundary_kind(cfg_, direction, true)
      ) ? 1 : 0;
   }
   amrex::Vector<amrex::BCRec> flux_boundaries(3);
   for (int component = 0; component < 3; ++component) {
      for (int direction = 0; direction < 3; ++direction) {
         const int lower_kind = Geom(0).isPeriodic(direction)
            ? amrex::BCType::int_dir
            : (reflection_lo[direction] != 0
                  ? (component == direction
                        ? amrex::BCType::reflect_odd
                        : amrex::BCType::reflect_even)
                  : amrex::BCType::foextrap);
         const int upper_kind = Geom(0).isPeriodic(direction)
            ? amrex::BCType::int_dir
            : (reflection_hi[direction] != 0
                  ? (component == direction
                        ? amrex::BCType::reflect_odd
                        : amrex::BCType::reflect_even)
                  : amrex::BCType::foextrap);
         flux_boundaries[component].setLo(direction, lower_kind);
         flux_boundaries[component].setHi(direction, upper_kind);
      }
   }
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      fill_level_primitive_cache(lev, time);
      amrex::MultiFab const& conserved = *fluid_state_.state()[lev];
      auto flux = std::make_unique<amrex::MultiFab>(
         conserved.boxArray(), conserved.DistributionMap(), 3, 1
      );
      flux->setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(
              conserved, amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto state = conserved.const_array(mfi);
         const auto primitive = primitive_[lev]->const_array(mfi);
         const auto spacetime =
            spacetime_coupling_.cell_spacetime()[lev]->const_array(mfi);
         const auto output = flux->array(mfi);
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const SpacetimeMetric metric = generated::z4c_spacetime_metric(
                  load_z4c_state(spacetime, i, j, k)
               );
               const BaryonFluxIntegrands point_flux =
                  generated::baryon_flux_integrands(
                     state(i, j, k, RHO),
                     primitive(i, j, k, MOM1),
                     primitive(i, j, k, MOM2),
                     primitive(i, j, k, MOM3),
                     metric
                  );
               output(i, j, k, 0) = point_flux.flux1;
               output(i, j, k, 1) = point_flux.flux2;
               output(i, j, k, 2) = point_flux.flux3;
            }
         );
      }
      flux_hierarchy[lev] = std::move(flux);
   }
   amrex::Vector<amrex::MultiFab*> flux_fields;
   flux_fields.reserve(flux_hierarchy.size());
   for (auto const& flux : flux_hierarchy) {
      flux_fields.push_back(flux.get());
   }
   amrex::FillBoundary(flux_fields, Geom(0).periodicity());
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      amrex::FillDomainBoundary(
         *flux_hierarchy[lev], Geom(lev), flux_boundaries
      );
   }

   std::vector<amrex::GpuArray<amrex::Real, 3>> positions;
   std::vector<int> point_reflections;
   positions.reserve(surface.points().size());
   point_reflections.reserve(surface.points().size());
   const auto hierarchy_lower = Geom(0).ProbLoArray();
   const auto hierarchy_upper = Geom(0).ProbHiArray();
   for (SurfaceQuadraturePoint const& point : surface.points()) {
      auto position = point.position;
      int reflection_mask = 0;
      for (int direction = 0; direction < 3; ++direction) {
         if (position[direction] < hierarchy_lower[direction]
             && reflection_lo[direction] != 0) {
            position[direction] = 2.0_rt * hierarchy_lower[direction]
               - position[direction];
            reflection_mask ^= 1 << direction;
         } else if (
            position[direction] > hierarchy_upper[direction]
            && reflection_hi[direction] != 0
         ) {
            position[direction] = 2.0_rt * hierarchy_upper[direction]
               - position[direction];
            reflection_mask ^= 1 << direction;
         }
      }
      positions.push_back(position);
      point_reflections.push_back(reflection_mask);
   }
   const HierarchyInterpolationPlan plan = diagnostic_point_interpolation_plan(
      std::move(positions),
      amrex::IndexType::TheCellType(),
      reflection_lo,
      reflection_hi
   );
   std::vector<amrex::Real> sampled_flux = plan.interpolate(
      flux_hierarchy, 0, 3
   );
   for (std::size_t point = 0; point < point_reflections.size(); ++point) {
      for (int direction = 0; direction < 3; ++direction) {
         if ((point_reflections[point] & (1 << direction)) != 0) {
            sampled_flux[3 * point + direction] *= -1.0_rt;
         }
      }
   }
   // ParametricSurface owns the outward coordinate directed area. The
   // conventional accretion rate is positive for inward baryon flow.
   return -surface.integrate_normal_fluxes(sampled_flux, 1)[0];
#endif
}

RadiationDiagnostics AmrSolverBase::compute_spacetime_radiation_diagnostics(
   amrex::Real time
)
{
   RadiationDiagnostics diagnostics;
   if (cfg_.diagnostic_multipole_l_max == 0) {
      return diagnostics;
   }
   if (!spacetime_state_.active()) {
      amrex::Abort("radiation diagnostics require a stored fixed or evolved spacetime");
   }
#if (AMREX_SPACEDIM != 3)
   amrex::Abort("radiation diagnostics currently require a three-dimensional hierarchy");
   return diagnostics;
#else
   if (mesh_coordinate_kind() != MeshCartesian || mixed_patch_system_enabled()) {
      amrex::Abort("radiation diagnostics currently require a single Cartesian patch");
   }
   for (int direction = 0; direction < 3; ++direction) {
      if (Geom(0).isPeriodic(direction)) {
         amrex::Abort("radiation diagnostics require nonperiodic extraction boundaries");
      }
   }

   fill_all_spacetime_boundaries(spacetime_state_.state(), time);
   prepare_spacetime_coupling_range(
      spacetime_state_.state(), 0, finestLevel() + 1, time, -1.0_rt, 0
   );
   if (spacetime_includes_matter(cfg_)) {
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         fill_level_primitive_cache(lev, time);
      }
   }
   fill_spacetime_matter_range(0, finestLevel() + 1);

   enum RadiationComponent : int {
      Electric11 = 0,
      Electric12,
      Electric13,
      Electric22,
      Electric23,
      Electric33,
      Magnetic11,
      Magnetic12,
      Magnetic13,
      Magnetic22,
      Magnetic23,
      Magnetic33,
      Gamma11,
      Gamma12,
      Gamma13,
      Gamma22,
      Gamma23,
      Gamma33,
      InvGamma11,
      InvGamma12,
      InvGamma13,
      InvGamma22,
      InvGamma23,
      InvGamma33,
      SqrtDetGamma,
      MatterFlux1,
      MatterFlux2,
      MatterFlux3,
      Count,
   };
   LevelData field_hierarchy(finestLevel() + 1);
   const bool matter_coupled = spacetime_includes_matter(cfg_);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      amrex::MultiFab const& state = *spacetime_state_.state()[lev];
      auto field = std::make_unique<amrex::MultiFab>(
         state.boxArray(), state.DistributionMap(), RadiationComponent::Count, 1
      );
      field->setVal(0.0_rt);
      const amrex::Box nodal_domain = amrex::convert(
         Geom(lev).Domain(), amrex::IntVect::TheNodeVector()
      );
      const auto spacing = Geom(lev).CellSizeArray();
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
         1.0_rt / spacing[0], 1.0_rt / spacing[1], 1.0_rt / spacing[2]
      }};
      const amrex::GpuArray<int, 3> domain_lower{{
         nodal_domain.smallEnd(0), nodal_domain.smallEnd(1), nodal_domain.smallEnd(2)
      }};
      const amrex::GpuArray<int, 3> domain_upper{{
         nodal_domain.bigEnd(0), nodal_domain.bigEnd(1), nodal_domain.bigEnd(2)
      }};
      for (amrex::MFIter mfi(state, false); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.validbox();
         const auto input = state.const_array(mfi);
         const auto matter = spacetime_coupling_.vertex_matter()[lev]->const_array(mfi);
         const auto output = field->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const Z4cSpatialDerivatives derivatives =
               gather_z4c_geometry_derivatives<4, 3>(
                  input, i, j, k, inv_spacing, domain_lower, domain_upper
               );
            const Z4cState point_state = load_z4c_state(input, i, j, k);
            const Z4cMatter point_matter = matter_coupled
               ? load_eulerian_matter(matter, i, j, k)
               : vacuum_z4c_matter();
            const Z4cConformalMetric point_metric =
               generated::z4c_conformal_metric(point_state);
            const Z4cConnections point_connections = generated::z4c_connections(
               point_state, derivatives, point_metric
            );
            const Z4cDiagnosticGeometry point_geometry =
               generated::z4c_diagnostic_geometry(
                  point_state, derivatives, point_metric, point_connections
               );
            const WeylRadiationIntegrands radiation =
               generated::weyl_radiation_integrands(
                  point_state,
                  derivatives,
                  point_metric,
                  point_connections,
                  point_geometry,
                  point_matter
               );
            const MatterEnergyFluxIntegrands matter_flux =
               generated::matter_energy_flux_integrands(
                  point_state, point_metric, point_matter
               );
            output(i, j, k, RadiationComponent::Electric11) = radiation.electric11;
            output(i, j, k, RadiationComponent::Electric12) = radiation.electric12;
            output(i, j, k, RadiationComponent::Electric13) = radiation.electric13;
            output(i, j, k, RadiationComponent::Electric22) = radiation.electric22;
            output(i, j, k, RadiationComponent::Electric23) = radiation.electric23;
            output(i, j, k, RadiationComponent::Electric33) = radiation.electric33;
            output(i, j, k, RadiationComponent::Magnetic11) = radiation.magnetic11;
            output(i, j, k, RadiationComponent::Magnetic12) = radiation.magnetic12;
            output(i, j, k, RadiationComponent::Magnetic13) = radiation.magnetic13;
            output(i, j, k, RadiationComponent::Magnetic22) = radiation.magnetic22;
            output(i, j, k, RadiationComponent::Magnetic23) = radiation.magnetic23;
            output(i, j, k, RadiationComponent::Magnetic33) = radiation.magnetic33;
            output(i, j, k, RadiationComponent::Gamma11) = point_metric.gamma11;
            output(i, j, k, RadiationComponent::Gamma12) = point_metric.gamma12;
            output(i, j, k, RadiationComponent::Gamma13) = point_metric.gamma13;
            output(i, j, k, RadiationComponent::Gamma22) = point_metric.gamma22;
            output(i, j, k, RadiationComponent::Gamma23) = point_metric.gamma23;
            output(i, j, k, RadiationComponent::Gamma33) = point_metric.gamma33;
            output(i, j, k, RadiationComponent::InvGamma11) = point_metric.inv_gamma11;
            output(i, j, k, RadiationComponent::InvGamma12) = point_metric.inv_gamma12;
            output(i, j, k, RadiationComponent::InvGamma13) = point_metric.inv_gamma13;
            output(i, j, k, RadiationComponent::InvGamma22) = point_metric.inv_gamma22;
            output(i, j, k, RadiationComponent::InvGamma23) = point_metric.inv_gamma23;
            output(i, j, k, RadiationComponent::InvGamma33) = point_metric.inv_gamma33;
            output(i, j, k, RadiationComponent::SqrtDetGamma) =
               point_metric.sqrt_det_gamma;
            output(i, j, k, RadiationComponent::MatterFlux1) = matter_flux.flux1;
            output(i, j, k, RadiationComponent::MatterFlux2) = matter_flux.flux2;
            output(i, j, k, RadiationComponent::MatterFlux3) = matter_flux.flux3;
         });
      }
      field_hierarchy[lev] = std::move(field);
   }
   amrex::Vector<amrex::MultiFab*> radiation_fields;
   radiation_fields.reserve(field_hierarchy.size());
   for (auto const& field : field_hierarchy) {
      radiation_fields.push_back(field.get());
   }
   amrex::FillBoundary(radiation_fields, Geom(0).periodicity());

   const ParametricSurface surface = diagnostic_extraction_surface();
   const HierarchyInterpolationPlan plan =
      diagnostic_surface_interpolation_plan(surface);
   const std::vector<amrex::Real> samples = plan.interpolate(
      field_hierarchy, 0, RadiationComponent::Count
   );
   std::vector<amrex::Real> psi4_real(surface.points().size());
   std::vector<amrex::Real> psi4_imaginary(surface.points().size());
   std::vector<amrex::Real> matter_flux(surface.points().size() * 3);
   const auto& surface_center = surface.center();
   for (std::size_t point = 0; point < surface.points().size(); ++point) {
      const std::size_t offset = point * RadiationComponent::Count;
      const WeylRadiationIntegrands radiation{
         samples[offset + RadiationComponent::Electric11],
         samples[offset + RadiationComponent::Electric12],
         samples[offset + RadiationComponent::Electric13],
         samples[offset + RadiationComponent::Electric22],
         samples[offset + RadiationComponent::Electric23],
         samples[offset + RadiationComponent::Electric33],
         samples[offset + RadiationComponent::Magnetic11],
         samples[offset + RadiationComponent::Magnetic12],
         samples[offset + RadiationComponent::Magnetic13],
         samples[offset + RadiationComponent::Magnetic22],
         samples[offset + RadiationComponent::Magnetic23],
         samples[offset + RadiationComponent::Magnetic33],
      };
      Z4cConformalMetric metric;
      metric.gamma11 = samples[offset + RadiationComponent::Gamma11];
      metric.gamma12 = samples[offset + RadiationComponent::Gamma12];
      metric.gamma13 = samples[offset + RadiationComponent::Gamma13];
      metric.gamma22 = samples[offset + RadiationComponent::Gamma22];
      metric.gamma23 = samples[offset + RadiationComponent::Gamma23];
      metric.gamma33 = samples[offset + RadiationComponent::Gamma33];
      metric.inv_gamma11 = samples[offset + RadiationComponent::InvGamma11];
      metric.inv_gamma12 = samples[offset + RadiationComponent::InvGamma12];
      metric.inv_gamma13 = samples[offset + RadiationComponent::InvGamma13];
      metric.inv_gamma22 = samples[offset + RadiationComponent::InvGamma22];
      metric.inv_gamma23 = samples[offset + RadiationComponent::InvGamma23];
      metric.inv_gamma33 = samples[offset + RadiationComponent::InvGamma33];
      metric.sqrt_det_gamma = samples[offset + RadiationComponent::SqrtDetGamma];
      const SurfaceQuadraturePoint& surface_point = surface.points()[point];
      const Vector3 relative_position{
         surface_point.position[0] - surface_center[0],
         surface_point.position[1] - surface_center[1],
         surface_point.position[2] - surface_center[2],
      };
      const Psi4Integrands psi4 = generated::psi4_from_weyl(
         metric, radiation, relative_position
      );
      psi4_real[point] = psi4.psi4_real;
      psi4_imaginary[point] = psi4.psi4_imaginary;
      for (int component = 0; component < 3; ++component) {
         matter_flux[point * 3 + component] =
            samples[offset + RadiationComponent::MatterFlux1 + component];
      }
   }
   diagnostics.radius = surface.radius();
   diagnostics.matter_energy_flux = surface.integrate_normal_fluxes(matter_flux, 1)[0];
   diagnostics.psi4_l2 = surface_complex_l2(surface, psi4_real, psi4_imaginary);
   diagnostics.psi4_modes = decompose_spin_weighted_field(
      surface,
      psi4_real,
      psi4_imaginary,
      "psi4",
      -2,
      2,
      cfg_.diagnostic_multipole_l_max
   );
   return diagnostics;
#endif
}

MultipoleDiagnostics AmrSolverBase::compute_azimuthal_surface_modes(
   amrex::Real time
)
{
   MultipoleDiagnostics diagnostics;
   const std::vector<std::string> field_names = azimuthal_mode_field_names(cfg_);
   if (field_names.empty()) {
      return diagnostics;
   }
#if (AMREX_SPACEDIM != 3)
   amrex::Abort("azimuthal surface modes currently require a three-dimensional hierarchy");
   return {};
#else
   if (mixed_patch_system_enabled()) {
      amrex::Abort("azimuthal surface modes currently require a single mapped patch");
   }

   fill_all_level_boundaries<false>(fluid_state_.state(), time);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      fill_level_primitive_cache(lev, time);
   }
   if (spacetime_state_.active()) {
      fill_all_spacetime_boundaries(spacetime_state_.state(), time);
   }

   const ParametricSurface surface = diagnostic_extraction_surface();
   const HierarchyInterpolationPlan cell_plan =
      diagnostic_surface_interpolation_plan(
         surface, amrex::IndexType::TheCellType()
      );
   std::optional<HierarchyInterpolationPlan> vertex_plan;
   diagnostics.radius = surface.radius();
   const GridFieldSetInfo fluid_fields = fluid_state_grid_fields(
      physics_system_, cfg_.passive_scalars
   );
   for (std::string const& field_name : field_names) {
      std::vector<amrex::Real> samples;
      if (field_name == "rho") {
         samples = cell_plan.interpolate(primitive_, RHO, 1);
      } else if (field_name == "pressure") {
         samples = cell_plan.interpolate(primitive_, ENERGY, 1);
      } else if (const int component = scalar_grid_field_component(
                    fluid_fields, field_name
                 ); component >= 0) {
         samples = cell_plan.interpolate(fluid_state_.state(), component, 1);
      } else if (spacetime_state_.active()) {
         const int component = scalar_grid_field_component(
            z4c_state_grid_fields(), field_name
         );
         if (component < 0) {
            amrex::Abort(
               "unsupported diagnostics.azimuthal_mode_fields entry '" + field_name
               + "'; supported scalar fields: "
               + supported_azimuthal_mode_fields(
                  physics_system_, cfg_.passive_scalars, true
               )
            );
         }
         if (!vertex_plan.has_value()) {
            vertex_plan.emplace(
               diagnostic_surface_interpolation_plan(
                  surface, amrex::IndexType::TheNodeType()
               )
            );
         }
         samples = vertex_plan->interpolate(
            spacetime_state_.state(), component, 1
         );
      } else {
         amrex::Abort(
            "unsupported diagnostics.azimuthal_mode_fields entry '" + field_name
            + "'; supported scalar fields: "
            + supported_azimuthal_mode_fields(
               physics_system_, cfg_.passive_scalars, false
            )
         );
      }
      auto field_modes = decompose_azimuthal_field(
         surface,
         samples,
         field_name,
         cfg_.diagnostic_azimuthal_mode_m_max
      );
      diagnostics.modes.insert(
         diagnostics.modes.end(),
         std::make_move_iterator(field_modes.begin()),
         std::make_move_iterator(field_modes.end())
      );
   }
   return diagnostics;
#endif
}

MatterModeDiagnostics AmrSolverBase::compute_matter_volume_modes(
   amrex::Real time
)
{
   MatterModeDiagnostics diagnostics;
   if (cfg_.diagnostic_matter_mode_m_max < 0) {
      return diagnostics;
   }
#if (AMREX_SPACEDIM != 3)
   amrex::Abort(
      "matter azimuthal volume modes currently require a three-dimensional "
      "hierarchy"
   );
   return {};
#else
   if (mixed_patch_system_enabled()) {
      amrex::Abort(
         "matter azimuthal volume modes currently require a single mapped "
         "patch"
      );
   }
   const amrex::GpuArray<amrex::Real, 3> origin{{
      cfg_.diagnostic_matter_mode_origin1,
      cfg_.diagnostic_matter_mode_origin2,
      cfg_.diagnostic_matter_mode_origin3,
   }};
   const amrex::GpuArray<amrex::Real, 3> configured_axis{{
      cfg_.diagnostic_matter_mode_axis1,
      cfg_.diagnostic_matter_mode_axis2,
      cfg_.diagnostic_matter_mode_axis3,
   }};
   const AzimuthalFrame azimuthal_frame = make_azimuthal_frame(
      origin, configured_axis
   );

   int symmetry_direction = -1;
   for (int direction = 0; direction < 3; ++direction) {
      if (!symmetry_axis_active(direction)) {
         continue;
      }
      if (symmetry_direction >= 0) {
         amrex::Abort(
            "matter azimuthal modes support only equatorial reflection; "
            "multiple reflection planes would constrain the requested modes"
         );
      }
      symmetry_direction = direction;
   }
   amrex::Real symmetry_plane = 0.0_rt;
   amrex::Real symmetry_factor = 1.0_rt;
   if (symmetry_direction >= 0) {
      const bool low_symmetry = side_boundary_kind(
         cfg_, symmetry_direction, false
      ) == BoundaryKind::Symmetry;
      const bool high_symmetry = side_boundary_kind(
         cfg_, symmetry_direction, true
      ) == BoundaryKind::Symmetry;
      if (low_symmetry == high_symmetry) {
         amrex::Abort(
            "matter azimuthal modes require exactly one equatorial symmetry "
            "boundary on the reflected axis"
         );
      }
      symmetry_plane = low_symmetry
         ? Geom(0).ProbLo(symmetry_direction)
         : Geom(0).ProbHi(symmetry_direction);
      const amrex::Real tolerance = 256.0_rt
         * std::numeric_limits<amrex::Real>::epsilon()
         * std::max<amrex::Real>(1.0_rt, std::abs(symmetry_plane));
      if (std::abs(
             std::abs(azimuthal_frame.axis[symmetry_direction]) - 1.0_rt
          ) > tolerance
          || std::abs(origin[symmetry_direction] - symmetry_plane)
             > tolerance) {
         amrex::Abort(
            "matter azimuthal modes can reconstruct only an equatorial "
            "reflection plane normal to the configured mode axis and through "
            "the configured origin"
         );
      }
      symmetry_factor = 2.0_rt;
   }

   const int mode_count = cfg_.diagnostic_matter_mode_m_max + 1;
   constexpr ComplexModeDiagnosticInfo mode_info = ComplexModeDiagnosticCatalog[
      static_cast<std::size_t>(
         ComplexModeDiagnosticId::MatterAzimuthalVolume
      )
   ];
   const int barycenter_begin = 2 * mode_count;
   const int reduction_components = barycenter_begin + 3;
   std::vector<amrex::Real> reduction_totals(
      static_cast<std::size_t>(reduction_components), 0.0_rt
   );
   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      fill_level_primitive_cache(lev, time);
      amrex::iMultiFab mask = valid_cell_mask(lev);
      amrex::MultiFab reduction(
         fluid_state_.state()[lev]->boxArray(),
         fluid_state_.state()[lev]->DistributionMap(),
         reduction_components,
         0
      );
      reduction.setVal(0.0_rt);
      const auto problo = Geom(lev).ProbLoArray();
      const auto dx = Geom(lev).CellSizeArray();
      const EOSParams eos = eos_params(cfg_);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(reduction, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto state = fluid_state_.state()[lev]->const_array(mfi);
         const auto primitive = primitive_[lev]->const_array(mfi);
         const auto valid = mask.const_array(mfi);
         const auto output = reduction.array(mfi);
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if (valid(i, j, k, 0) == 0
                   || primitive_is_atmosphere(
                      primitive(i, j, k, RHO),
                      primitive(i, j, k, ENERGY),
                      eos
                   )) {
                  return;
               }
               const auto q = mesh_coordinate_triplet_at_location(
                  coordinate,
                  axes,
                  problo,
                  dx,
                  i,
                  j,
                  k,
                  false,
                  false,
                  false
               );
               const auto physical = mesh_coordinate_triplet_to_cartesian(
                  coordinate, q
               );
               const AzimuthalCoordinate azimuthal = azimuthal_coordinate(
                  azimuthal_frame, physical
               );
               const amrex::Real matter_weight = state(i, j, k, RHO)
                  * mesh_cell_volume_for_reduction(
                     coordinate, axes, problo, dx, i, j, k
                  );
               for (int m = 0; m < mode_count; ++m) {
                  output(i, j, k, 2 * m) = matter_weight
                     * std::cos(
                        static_cast<amrex::Real>(m) * azimuthal.phi
                     );
                  output(i, j, k, 2 * m + 1) = matter_weight
                     * std::sin(
                        static_cast<amrex::Real>(m) * azimuthal.phi
                     );
               }
               output(i, j, k, barycenter_begin) =
                  matter_weight * physical[0];
               output(i, j, k, barycenter_begin + 1) =
                  matter_weight * physical[1];
               output(i, j, k, barycenter_begin + 2) =
                  matter_weight * physical[2];
            }
         );
      }
      for (int component = 0; component < reduction_components; ++component) {
         reduction_totals[static_cast<std::size_t>(component)] +=
            reduction.sum(component, true);
      }
   }
   amrex::ParallelDescriptor::ReduceRealSum(
      reduction_totals.data(), reduction_components
   );
   for (int m = 0; m < mode_count; ++m) {
      diagnostics.modes.push_back({
         "rest_mass",
         std::string(mode_info.basis),
         0,
         std::nullopt,
         m,
         {
            symmetry_factor * reduction_totals[static_cast<std::size_t>(2 * m)],
            symmetry_factor * reduction_totals[static_cast<std::size_t>(2 * m + 1)],
         },
         std::nullopt,
      });
   }
   for (int direction = 0; direction < 3; ++direction) {
      diagnostics.barycenter[direction] = symmetry_factor
         * reduction_totals[static_cast<std::size_t>(barycenter_begin + direction)];
   }
   diagnostics.mass = diagnostics.modes.front().value.real();
   if (diagnostics.mass > 0.0_rt) {
      for (int direction = 0; direction < 3; ++direction) {
         diagnostics.barycenter[direction] /= diagnostics.mass;
      }
      if (symmetry_direction >= 0) {
         diagnostics.barycenter[symmetry_direction] = symmetry_plane;
      }
   }
   return diagnostics;
#endif
}

} // namespace pyfluid_amrex
