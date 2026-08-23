#include "pyfluid_amrex/initial_data/magnetic.hpp"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_GpuLaunch.H>

#include <array>
#include <cmath>
#include <limits>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/cold_barotrope.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {
namespace {

void sync_rmhd_primitive_b_from_densitized_faces_impl(
   amrex::MultiFab& primitive,
   FaceMagneticField const& face_magnetic,
   amrex::MultiFab const& cell_spacetime
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(primitive, face_magnetic, cell_spacetime);
   amrex::Abort("densitized RMHD face-field synchronization requires 3D");
#else
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto output = primitive.array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto magnetic1 = face_magnetic.b[0]->const_array(mfi);
      const auto magnetic2 = face_magnetic.b[1]->const_array(mfi);
      const auto magnetic3 = face_magnetic.b[2]->const_array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(spacetime, i, j, k)
         );
         const amrex::Real inverse_volume = 1.0 / metric.sqrt_det_gamma;
         output(i, j, k, B1) = 0.5 * inverse_volume * (
            magnetic1(i, j, k, 0) + magnetic1(i + 1, j, k, 0)
         );
         output(i, j, k, B2) = 0.5 * inverse_volume * (
            magnetic2(i, j, k, 0) + magnetic2(i, j + 1, k, 0)
         );
         output(i, j, k, B3) = 0.5 * inverse_volume * (
            magnetic3(i, j, k, 0) + magnetic3(i, j, k + 1, 0)
         );
      });
   }
   }
#endif
}

} // namespace

void sync_rmhd_primitive_b_from_densitized_faces(
   amrex::MultiFab& primitive,
   FaceMagneticField const& face_magnetic,
   amrex::MultiFab const& cell_spacetime
)
{
   sync_rmhd_primitive_b_from_densitized_faces_impl(
      primitive, face_magnetic, cell_spacetime
   );
}

PressurePoloidalMagneticSeedSummary apply_pressure_poloidal_magnetic_seed(
   FaceMagneticField& face_magnetic,
   amrex::MultiFab& primitive,
   amrex::MultiFab const& cell_spacetime,
   amrex::Geometry const& geometry,
   RunConfig const& config,
   EOSParams const& eos,
   PressurePoloidalMagneticSeedParameters const& parameters,
   PressurePoloidalMagneticSeedWorkspace& workspace
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      face_magnetic,
      primitive,
      cell_spacetime,
      geometry,
      config,
      eos,
      parameters,
      workspace
   );
   amrex::Abort("pressure-poloidal magnetic seed requires 3D");
   return {};
#else
   if (primitive.nComp() < NVAR_MHD
       || primitive.boxArray() != cell_spacetime.boxArray()) {
      amrex::Abort("pressure-poloidal magnetic seed received incompatible fields");
   }
   const amrex::Real axis_norm = std::sqrt(
      parameters.axis[0] * parameters.axis[0]
      + parameters.axis[1] * parameters.axis[1]
      + parameters.axis[2] * parameters.axis[2]
   );
   if (!(axis_norm > 0.0) || !std::isfinite(axis_norm)) {
      amrex::Abort("pressure-poloidal magnetic seed axis is invalid");
   }
   const std::array<amrex::Real, 3> axis{{
      parameters.axis[0] / axis_norm,
      parameters.axis[1] / axis_norm,
      parameters.axis[2] / axis_norm,
   }};

   if (!workspace.pressure) {
      workspace.pressure = std::make_unique<amrex::MultiFab>(
         primitive.boxArray(), primitive.DistributionMap(), 1, 1
      );
      workspace.comoving_field_squared = std::make_unique<amrex::MultiFab>(
         primitive.boxArray(), primitive.DistributionMap(), 1, 0
      );
      workspace.divergence = std::make_unique<amrex::MultiFab>(
         primitive.boxArray(), primitive.DistributionMap(), 1, 0
      );
      allocate_edge_electric_field(
         workspace.vector_potential,
         primitive.boxArray(),
         primitive.DistributionMap()
      );
   }

   workspace.pressure->setVal(0.0);
   amrex::MultiFab::Copy(
      *workspace.pressure, primitive, ENERGY, 0, 1, 0
   );
   workspace.pressure->FillBoundary(geometry.periodicity());
   amrex::Real maximum_pressure = workspace.pressure->norm0(0, 0, true);
   amrex::ParallelDescriptor::ReduceRealMax(maximum_pressure);
   if (!(maximum_pressure > 0.0) || !std::isfinite(maximum_pressure)) {
      amrex::Abort("pressure-poloidal magnetic seed has no positive pressure support");
   }

   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const amrex::Real inverse_maximum_pressure = 1.0 / maximum_pressure;
   const amrex::Real cutoff = parameters.pressure_cutoff_fraction;
   const amrex::Real exponent = parameters.exponent;
   const amrex::Real axis1 = axis[0];
   const amrex::Real axis2 = axis[1];
   const amrex::Real axis3 = axis[2];
   const amrex::Real center1 = parameters.center[0];
   const amrex::Real center2 = parameters.center[1];
   const amrex::Real center3 = parameters.center[2];
   set_edge_field_value(workspace.vector_potential, 0.0);
   for (int component = 0; component < 3; ++component) {
      amrex::MultiFab& edge = *workspace.vector_potential.e[component];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(edge, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto pressure = workspace.pressure->const_array(mfi);
         const auto potential = edge.array(mfi);
         const int vector_component = component;
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            amrex::Real edge_pressure = 0.0;
            if (vector_component == 0) {
               edge_pressure = 0.25 * (
                  pressure(i, j - 1, k - 1, 0)
                  + pressure(i, j, k - 1, 0)
                  + pressure(i, j - 1, k, 0)
                  + pressure(i, j, k, 0)
               );
            } else if (vector_component == 1) {
               edge_pressure = 0.25 * (
                  pressure(i - 1, j, k - 1, 0)
                  + pressure(i, j, k - 1, 0)
                  + pressure(i - 1, j, k, 0)
                  + pressure(i, j, k, 0)
               );
            } else {
               edge_pressure = 0.25 * (
                  pressure(i - 1, j - 1, k, 0)
                  + pressure(i, j - 1, k, 0)
                  + pressure(i - 1, j, k, 0)
                  + pressure(i, j, k, 0)
               );
            }
            const amrex::Real x = lower[0]
               + (static_cast<amrex::Real>(i)
                  + (vector_component == 0 ? 0.5 : 0.0)) * spacing[0]
               - center1;
            const amrex::Real y = lower[1]
               + (static_cast<amrex::Real>(j)
                  + (vector_component == 1 ? 0.5 : 0.0)) * spacing[1]
               - center2;
            const amrex::Real z = lower[2]
               + (static_cast<amrex::Real>(k)
                  + (vector_component == 2 ? 0.5 : 0.0)) * spacing[2]
               - center3;
            const amrex::Real tangent1 = axis2 * z - axis3 * y;
            const amrex::Real tangent2 = axis3 * x - axis1 * z;
            const amrex::Real tangent3 = axis1 * y - axis2 * x;
            const amrex::Real tangent = vector_component == 0
               ? tangent1 : (vector_component == 1 ? tangent2 : tangent3);
            const amrex::Real support = amrex::max(
               edge_pressure * inverse_maximum_pressure - cutoff,
               amrex::Real(0.0)
            );
            potential(i, j, k, 0) = tangent * std::pow(support, exponent);
         });
      }
      }
   }
   enforce_edge_electric_boundary(
      workspace.vector_potential, geometry, config
   );
   for (int direction = 0; direction < 3; ++direction) {
      face_magnetic.b[direction]->setVal(0.0);
   }
   // The CT update is B <- B - dt curl(E); dt=-1 applies B=curl(A).
   apply_edge_electric_curl(
      face_magnetic, workspace.vector_potential, geometry, -1.0, config
   );
   apply_face_b_boundary(face_magnetic, geometry, config);
   // The CT face owner is the Valencia magnetic density sqrt(gamma) B^i.
   // Primitive stress-energy needs the physical Eulerian B^i.
   sync_rmhd_primitive_b_from_densitized_faces(
      primitive, face_magnetic, cell_spacetime
   );

   workspace.comoving_field_squared->setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(
           *workspace.comoving_field_squared, amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto output = workspace.comoving_field_squared->array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(spacetime, i, j, k)
         );
         const RMHDPrimitive point = rmhd_primitive_at<EOSRuntime>(
            cached, i, j, k, eos
         );
         output(i, j, k, 0) = generated::rmhd_comoving_magnetic_squared(
            point, metric
         );
      });
   }
   }
   amrex::Real maximum_comoving_squared =
      workspace.comoving_field_squared->norm0(0, 0, true);
   amrex::ParallelDescriptor::ReduceRealMax(maximum_comoving_squared);
   if (!(maximum_comoving_squared > 0.0)
       || !std::isfinite(maximum_comoving_squared)) {
      amrex::Abort("pressure-poloidal magnetic seed produced no finite field");
   }
   const amrex::Real unit_maximum = std::sqrt(maximum_comoving_squared);
   const amrex::Real normalization =
      parameters.maximum_comoving_field / unit_maximum;
   for (int direction = 0; direction < 3; ++direction) {
      face_magnetic.b[direction]->mult(
         normalization, 0, 1, face_magnetic.b[direction]->nGrow()
      );
   }
   sync_rmhd_primitive_b_from_densitized_faces(
      primitive, face_magnetic, cell_spacetime
   );

   fill_face_divergence(
      *workspace.divergence, face_magnetic, geometry, false
   );
   amrex::Real divergence_linf = workspace.divergence->norm0(0, 0, true);
   amrex::ParallelDescriptor::ReduceRealMax(divergence_linf);
   return {
      maximum_pressure,
      unit_maximum,
      normalization,
      parameters.maximum_comoving_field,
      divergence_linf,
   };
#endif
}

namespace {

amrex::Vector<GridFieldSymmetryKind> magnetic_vector_potential_symmetries()
{
   // A is a polar vector.  A reflection is a symmetry of an oriented
   // poloidal field only when its plane is normal to the magnetic axis; the
   // configuration validator rejects transverse reflection planes.
   return {
      GridFieldSymmetryKind::Vector1,
      GridFieldSymmetryKind::Vector2,
      GridFieldSymmetryKind::Vector3,
   };
}

void allocate_grad_shafranov_workspace(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& primitive,
   PoloidalGradShafranovWorkspace& workspace
)
{
   if (workspace.vector_potential[0]) {
      return;
   }
   for (int component = 0; component < 3; ++component) {
      workspace.vector_potential[component] =
         std::make_unique<amrex::MultiFab>(
            primitive.boxArray(), primitive.DistributionMap(), 1, 1
         );
      workspace.candidate[component] = std::make_unique<amrex::MultiFab>(
         primitive.boxArray(), primitive.DistributionMap(), 1, 1
      );
      workspace.rhs[component] = std::make_unique<amrex::MultiFab>(
         primitive.boxArray(), primitive.DistributionMap(), 1, 0
      );
      workspace.vector_potential[component]->setVal(0.0);
      workspace.candidate[component]->setVal(0.0);
      workspace.rhs[component]->setVal(0.0);
   }
   workspace.flux_function = std::make_unique<amrex::MultiFab>(
      primitive.boxArray(), primitive.DistributionMap(), 1, 1
   );
   workspace.metric_potential = std::make_unique<amrex::MultiFab>(
      primitive.boxArray(), primitive.DistributionMap(), 2, 1
   );
   workspace.equation_residual = std::make_unique<amrex::MultiFab>(
      primitive.boxArray(), primitive.DistributionMap(), 3, 0
   );
   workspace.bernoulli_residual = std::make_unique<amrex::MultiFab>(
      primitive.boxArray(), primitive.DistributionMap(), 1, 0
   );
   workspace.comoving_field_squared = std::make_unique<amrex::MultiFab>(
      primitive.boxArray(), primitive.DistributionMap(), 1, 0
   );
   workspace.divergence = std::make_unique<amrex::MultiFab>(
      primitive.boxArray(), primitive.DistributionMap(), 1, 0
   );
   workspace.flux_function->setVal(0.0);
   workspace.metric_potential->setVal(0.0);
   allocate_edge_electric_field(
      workspace.edge_vector_potential,
      hierarchy.grids[0],
      hierarchy.distribution[0]
   );
}

void fill_grad_shafranov_geometry_and_flux(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& cell_spacetime,
   PoloidalGradShafranovParameters const& parameters,
   PoloidalGradShafranovWorkspace& workspace
)
{
   const amrex::Geometry& geometry = hierarchy.geometry[0];
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const amrex::Box domain = geometry.Domain();
   const int ilo = domain.smallEnd(0);
   const int jlo = domain.smallEnd(1);
   const int klo = domain.smallEnd(2);
   const amrex::Real axis1 = parameters.axis[0];
   const amrex::Real axis2 = parameters.axis[1];
   const amrex::Real axis3 = parameters.axis[2];
   const amrex::Real center1 = parameters.center[0];
   const amrex::Real center2 = parameters.center[1];
   const amrex::Real center3 = parameters.center[2];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(
           *workspace.flux_function, amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto potential1 = workspace.vector_potential[0]->const_array(mfi);
      const auto potential2 = workspace.vector_potential[1]->const_array(mfi);
      const auto potential3 = workspace.vector_potential[2]->const_array(mfi);
      const auto flux = workspace.flux_function->array(mfi);
      const auto metric_output = workspace.metric_potential->array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const amrex::Real x = lower[0]
            + (static_cast<amrex::Real>(i - ilo) + 0.5) * spacing[0]
            - center1;
         const amrex::Real y = lower[1]
            + (static_cast<amrex::Real>(j - jlo) + 0.5) * spacing[1]
            - center2;
         const amrex::Real z = lower[2]
            + (static_cast<amrex::Real>(k - klo) + 0.5) * spacing[2]
            - center3;
         const amrex::Real tangent1 = axis2 * z - axis3 * y;
         const amrex::Real tangent2 = axis3 * x - axis1 * z;
         const amrex::Real tangent3 = axis1 * y - axis2 * x;
         flux(i, j, k, 0) =
            potential1(i, j, k, 0) * tangent1
            + potential2(i, j, k, 0) * tangent2
            + potential3(i, j, k, 0) * tangent3;
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(spacetime, i, j, k)
         );
         const amrex::Real log_volume = std::log(metric.sqrt_det_gamma);
         metric_output(i, j, k, 0) = std::log(metric.lapse)
            - log_volume / 3.0;
         metric_output(i, j, k, 1) = std::exp(
            (4.0 / 3.0) * log_volume
         );
      });
   }
   }

   for (int component = 0; component < 3; ++component) {
      amrex::Vector<amrex::MultiFab*> field{
         workspace.vector_potential[component].get()
      };
      fill_initial_data_ghosts(
         hierarchy,
         field,
         initial_data_boundary_records(
            hierarchy,
            {magnetic_vector_potential_symmetries()[component]}
         )
      );
   }
   amrex::Vector<amrex::MultiFab*> flux_field{
      workspace.flux_function.get()
   };
   fill_initial_data_ghosts(
      hierarchy,
      flux_field,
      initial_data_boundary_records(
         hierarchy, {GridFieldSymmetryKind::Scalar}
      )
   );
   amrex::Vector<amrex::MultiFab*> metric_field{
      workspace.metric_potential.get()
   };
   fill_initial_data_ghosts(
      hierarchy,
      metric_field,
      initial_data_boundary_records(
         hierarchy,
         {GridFieldSymmetryKind::Scalar, GridFieldSymmetryKind::Scalar}
      )
   );
}

void fill_grad_shafranov_rhs(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   RotatingStarParameters const& rotating_star,
   PoloidalGradShafranovParameters const& parameters,
   PoloidalGradShafranovWorkspace& workspace
)
{
   const amrex::Geometry& geometry = hierarchy.geometry[0];
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const amrex::Box domain = geometry.Domain();
   const int ilo = domain.smallEnd(0);
   const int jlo = domain.smallEnd(1);
   const int klo = domain.smallEnd(2);
   const amrex::Real inverse_two_dx1 = 0.5 / spacing[0];
   const amrex::Real inverse_two_dx2 = 0.5 / spacing[1];
   const amrex::Real inverse_two_dx3 = 0.5 / spacing[2];
   const amrex::Real axis1 = parameters.axis[0];
   const amrex::Real axis2 = parameters.axis[1];
   const amrex::Real axis3 = parameters.axis[2];
   const amrex::Real center1 = parameters.center[0];
   const amrex::Real center2 = parameters.center[1];
   const amrex::Real center3 = parameters.center[2];
   const amrex::Real magnetic_k = parameters.poloidal_magnetization;
   const amrex::Real magnetic_xi = parameters.nonlinear_poloidal;
   const InitialAtmosphereParameters atmosphere = rotating_star.atmosphere;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto state = primitive.const_array(mfi);
      const auto flux = workspace.flux_function->const_array(mfi);
      const auto metric = workspace.metric_potential->const_array(mfi);
      const auto rhs1 = workspace.rhs[0]->array(mfi);
      const auto rhs2 = workspace.rhs[1]->array(mfi);
      const auto rhs3 = workspace.rhs[2]->array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const amrex::Real x = lower[0]
            + (static_cast<amrex::Real>(i - ilo) + 0.5) * spacing[0]
            - center1;
         const amrex::Real y = lower[1]
            + (static_cast<amrex::Real>(j - jlo) + 0.5) * spacing[1]
            - center2;
         const amrex::Real z = lower[2]
            + (static_cast<amrex::Real>(k - klo) + 0.5) * spacing[2]
            - center3;
         const amrex::Real tangent1 = axis2 * z - axis3 * y;
         const amrex::Real tangent2 = axis3 * x - axis1 * z;
         const amrex::Real tangent3 = axis1 * y - axis2 * x;
         const amrex::Real cylindrical_radius_squared =
            tangent1 * tangent1 + tangent2 * tangent2
            + tangent3 * tangent3;
         amrex::Real source1 = 0.0;
         amrex::Real source2 = 0.0;
         amrex::Real source3 = 0.0;
         if (cylindrical_radius_squared
             > 64.0 * std::numeric_limits<amrex::Real>::epsilon()) {
            const amrex::Real cylindrical_radius =
               std::sqrt(cylindrical_radius_squared);
            const amrex::Real flux_value = flux(i, j, k, 0);
            const amrex::Real dflux1 = (
               flux(i + 1, j, k, 0) - flux(i - 1, j, k, 0)
            ) * inverse_two_dx1;
            const amrex::Real dflux2 = (
               flux(i, j + 1, k, 0) - flux(i, j - 1, k, 0)
            ) * inverse_two_dx2;
            const amrex::Real dflux3 = (
               flux(i, j, k + 1, 0) - flux(i, j, k - 1, 0)
            ) * inverse_two_dx3;
            const amrex::Real dq1 = (
               metric(i + 1, j, k, 0) - metric(i - 1, j, k, 0)
            ) * inverse_two_dx1;
            const amrex::Real dq2 = (
               metric(i, j + 1, k, 0) - metric(i, j - 1, k, 0)
            ) * inverse_two_dx2;
            const amrex::Real dq3 = (
               metric(i, j, k + 1, 0) - metric(i, j, k - 1, 0)
            ) * inverse_two_dx3;
            const amrex::Real gradient_coupling =
               dflux1 * dq1 + dflux2 * dq2 + dflux3 * dq3;
            const PatchCoordinate physical{x + center1, y + center2, z + center3};
            const InitialDataSamplePoint sample{physical, physical};
            const EOSThermodynamicState atmosphere_state =
               initial_atmosphere_state<EOSRuntime>(atmosphere, sample);
            amrex::Real rho_h = 0.0;
            if (initial_matter_exceeds_atmosphere(
                   state(i, j, k, RHO), atmosphere_state.rho
                )) {
               const ColdBarotropeState matter = cold_barotrope_from_density(
                  state(i, j, k, RHO), eos
               );
               rho_h = matter.rho * matter.specific_enthalpy;
            }
            const amrex::Real magnetization_derivative = magnetic_k * (
               1.0 + magnetic_xi * flux_value
            );
            const amrex::Real amplitude = -(
               gradient_coupling / cylindrical_radius
               + metric(i, j, k, 1) * cylindrical_radius * rho_h
                  * magnetization_derivative
            );
            const amrex::Real inverse_radius = 1.0 / cylindrical_radius;
            source1 = amplitude * tangent1 * inverse_radius;
            source2 = amplitude * tangent2 * inverse_radius;
            source3 = amplitude * tangent3 * inverse_radius;
         }
         rhs1(i, j, k, 0) = source1;
         rhs2(i, j, k, 0) = source2;
         rhs3(i, j, k, 0) = source3;
      });
   }
   }
}

void fill_edge_vector_potential(
   InitialDataEllipticHierarchy const& hierarchy,
   RunConfig const& config,
   PoloidalGradShafranovWorkspace& workspace
)
{
   set_edge_field_value(workspace.edge_vector_potential, 0.0);
   for (int component = 0; component < 3; ++component) {
      amrex::MultiFab& edge = *workspace.edge_vector_potential.e[component];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(edge, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto cell = workspace.vector_potential[component]->const_array(mfi);
         const auto output = edge.array(mfi);
         const int vector_component = component;
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            if (vector_component == 0) {
               output(i, j, k, 0) = 0.25 * (
                  cell(i, j - 1, k - 1, 0) + cell(i, j, k - 1, 0)
                  + cell(i, j - 1, k, 0) + cell(i, j, k, 0)
               );
            } else if (vector_component == 1) {
               output(i, j, k, 0) = 0.25 * (
                  cell(i - 1, j, k - 1, 0) + cell(i, j, k - 1, 0)
                  + cell(i - 1, j, k, 0) + cell(i, j, k, 0)
               );
            } else {
               output(i, j, k, 0) = 0.25 * (
                  cell(i - 1, j - 1, k, 0) + cell(i, j - 1, k, 0)
                  + cell(i - 1, j, k, 0) + cell(i, j, k, 0)
               );
            }
         });
      }
      }
   }
   enforce_edge_electric_boundary(
      workspace.edge_vector_potential, hierarchy.geometry[0], config
   );
}

} // namespace

PoloidalGradShafranovSummary update_poloidal_grad_shafranov_equilibrium(
   InitialDataEllipticHierarchy const& hierarchy,
   FaceMagneticField& face_magnetic,
   amrex::MultiFab& primitive,
   amrex::MultiFab const& cell_spacetime,
   RunConfig const& config,
   int eos_method,
   EOSParams const& eos,
   RotatingStarParameters const& rotating_star,
   PoloidalGradShafranovParameters const& parameters,
   PoloidalGradShafranovWorkspace& workspace,
   bool update_fluid
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy, face_magnetic, primitive, cell_spacetime, config,
      eos_method, eos, rotating_star, parameters, workspace, update_fluid
   );
   amrex::Abort("poloidal Grad-Shafranov equilibrium requires 3D");
   return {};
#else
   if (hierarchy.geometry.size() != 1 || primitive.nComp() < NVAR_MHD
       || primitive.boxArray() != hierarchy.grids[0]
       || cell_spacetime.boxArray() != hierarchy.grids[0]) {
      amrex::Abort(
         "poloidal Grad-Shafranov equilibrium requires one compatible level"
      );
   }
   const amrex::Real axis_norm = std::sqrt(
      parameters.axis[0] * parameters.axis[0]
      + parameters.axis[1] * parameters.axis[1]
      + parameters.axis[2] * parameters.axis[2]
   );
   if (!(axis_norm > 0.0) || !std::isfinite(axis_norm)
       || !cold_barotrope_supported(eos)) {
      amrex::Abort("invalid poloidal Grad-Shafranov equilibrium parameters");
   }
   PoloidalGradShafranovParameters normalized = parameters;
   for (int component = 0; component < 3; ++component) {
      normalized.axis[component] /= axis_norm;
   }
   allocate_grad_shafranov_workspace(hierarchy, primitive, workspace);
   fill_grad_shafranov_geometry_and_flux(
      hierarchy, cell_spacetime, normalized, workspace
   );
   fill_grad_shafranov_rhs(
      hierarchy, primitive, eos, rotating_star, normalized, workspace
   );

   auto vector_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy, 3, 1.0, 0.0, 0.0
   );
   set_initial_data_reflection_boundaries(
      hierarchy, vector_boundary, magnetic_vector_potential_symmetries()
   );
   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center{{
      normalized.center[0], normalized.center[1], normalized.center[2]
   }};
   set_centered_power_law_robin_boundary(
      hierarchy,
      vector_boundary,
      amrex::Vector<amrex::Real>{0.0, 0.0, 0.0},
      2.0,
      &center
   );
   InitialDataEllipticOptions linear_options;
   linear_options.boundary_interpolation_order = 4;
   linear_options.max_iterations = 300;
   linear_options.max_fmg_iterations = 1;
   linear_options.relative_tolerance = 1.0e-12;

   PoloidalGradShafranovSummary summary;
   amrex::Real update_norm = 0.0;
   amrex::Real potential_norm = 0.0;
   for (int component = 0; component < 3; ++component) {
      amrex::MultiFab::Copy(
         *workspace.candidate[component],
         *workspace.vector_potential[component],
         0,
         0,
         1,
         1
      );
      amrex::Vector<amrex::MultiFab*> candidate{
         workspace.candidate[component].get()
      };
      amrex::Vector<amrex::MultiFab const*> rhs{
         workspace.rhs[component].get()
      };
      const InitialDataEllipticResult linear = solve_initial_data_poisson(
         hierarchy,
         candidate,
         rhs,
         vector_boundary.component(component),
         linear_options
      );
      summary.linear_residual = std::max(
         summary.linear_residual, linear.residual
      );
      amrex::MultiFab::Subtract(
         *workspace.candidate[component],
         *workspace.vector_potential[component],
         0,
         0,
         1,
         1
      );
      workspace.candidate[component]->mult(
         normalized.relaxation, 0, 1, 1
      );
      update_norm = std::max(
         update_norm,
         workspace.candidate[component]->norm0(0, 0, false)
      );
      amrex::MultiFab::Add(
         *workspace.vector_potential[component],
         *workspace.candidate[component],
         0,
         0,
         1,
         1
      );
      potential_norm = std::max(
         potential_norm,
         workspace.vector_potential[component]->norm0(0, 0, false)
      );
   }
   summary.relative_potential_update = update_norm / std::max(
      potential_norm, amrex::Real(1.0e-300)
   );

   fill_grad_shafranov_geometry_and_flux(
      hierarchy, cell_spacetime, normalized, workspace
   );
   fill_edge_vector_potential(hierarchy, config, workspace);
   for (int direction = 0; direction < 3; ++direction) {
      face_magnetic.b[direction]->setVal(0.0);
   }
   // curl(A) is the densitized Valencia field sqrt(gamma) B^i.
   apply_edge_electric_curl(
      face_magnetic,
      workspace.edge_vector_potential,
      hierarchy.geometry[0],
      -1.0,
      config
   );
   apply_face_b_boundary(face_magnetic, hierarchy.geometry[0], config);
   sync_rmhd_primitive_b_from_densitized_faces(
      primitive, face_magnetic, cell_spacetime
   );

   if (update_fluid) {
      const PoloidalMagneticBernoulli magnetic_bernoulli{
         {workspace.flux_function.get()},
         normalized.poloidal_magnetization,
         normalized.nonlinear_poloidal,
      };
      summary.rotating_star = update_rotating_star_primitives(
         hierarchy,
         {&primitive},
         {&cell_spacetime},
         eos_method,
         eos,
         rotating_star,
         &magnetic_bernoulli
      );
   }

   // Certify the final, post-Euler matter state.  The Euler update changes
   // primitive thermodynamics only; the post-solve flux function and current
   // metric were already refreshed immediately above and remain authoritative.
   fill_grad_shafranov_rhs(
      hierarchy, primitive, eos, rotating_star, normalized, workspace
   );
   workspace.equation_residual->setVal(0.0);
   workspace.bernoulli_residual->setVal(0.0);
   workspace.comoving_field_squared->setVal(0.0);
   const auto spacing = hierarchy.geometry[0].CellSizeArray();
   const amrex::Real inverse_dx1_squared = 1.0 / (spacing[0] * spacing[0]);
   const amrex::Real inverse_dx2_squared = 1.0 / (spacing[1] * spacing[1]);
   const amrex::Real inverse_dx3_squared = 1.0 / (spacing[2] * spacing[2]);
   const auto lower = hierarchy.geometry[0].ProbLoArray();
   const amrex::Box domain = hierarchy.geometry[0].Domain();
   const int ilo = domain.smallEnd(0);
   const int ihi = domain.bigEnd(0);
   const int jlo = domain.smallEnd(1);
   const int jhi = domain.bigEnd(1);
   const int klo = domain.smallEnd(2);
   const int khi = domain.bigEnd(2);
   RotatingStarParameters diagnostic_rotation = rotating_star;
   if (update_fluid) {
      diagnostic_rotation.injection_energy =
         summary.rotating_star.injection_energy;
   }
   const amrex::Real magnetic_k = normalized.poloidal_magnetization;
   const amrex::Real magnetic_xi = normalized.nonlinear_poloidal;
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto state = primitive.const_array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto flux = workspace.flux_function->const_array(mfi);
      const auto residual = workspace.equation_residual->array(mfi);
      const auto bernoulli = workspace.bernoulli_residual->array(mfi);
      const auto comoving = workspace.comoving_field_squared->array(mfi);
      const auto potential1 = workspace.vector_potential[0]->const_array(mfi);
      const auto potential2 = workspace.vector_potential[1]->const_array(mfi);
      const auto potential3 = workspace.vector_potential[2]->const_array(mfi);
      const auto rhs1 = workspace.rhs[0]->const_array(mfi);
      const auto rhs2 = workspace.rhs[1]->const_array(mfi);
      const auto rhs3 = workspace.rhs[2]->const_array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const amrex::Array4<amrex::Real const> potentials[3]{
            potential1, potential2, potential3
         };
         const amrex::Array4<amrex::Real const> sources[3]{rhs1, rhs2, rhs3};
         const bool interior = i > ilo && i < ihi && j > jlo && j < jhi
            && k > klo && k < khi;
         for (int component = 0; component < 3; ++component) {
            const auto field = potentials[component];
            const amrex::Real laplacian =
               (field(i + 1, j, k, 0) - 2.0 * field(i, j, k, 0)
                  + field(i - 1, j, k, 0)) * inverse_dx1_squared
               + (field(i, j + 1, k, 0) - 2.0 * field(i, j, k, 0)
                  + field(i, j - 1, k, 0)) * inverse_dx2_squared
               + (field(i, j, k + 1, 0) - 2.0 * field(i, j, k, 0)
                  + field(i, j, k - 1, 0)) * inverse_dx3_squared;
            residual(i, j, k, component) = interior
               ? laplacian - sources[component](i, j, k, 0) : 0.0;
         }
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(spacetime, i, j, k)
         );
         const RMHDPrimitive point = rmhd_primitive_at<EOSRuntime>(
            state, i, j, k, eos
         );
         comoving(i, j, k, 0) =
            generated::rmhd_comoving_magnetic_squared(point, metric);
         if (update_fluid) {
            const PatchCoordinate physical{
               lower[0] + (static_cast<amrex::Real>(i - ilo) + 0.5)
                  * spacing[0],
               lower[1] + (static_cast<amrex::Real>(j - jlo) + 0.5)
                  * spacing[1],
               lower[2] + (static_cast<amrex::Real>(k - klo) + 0.5)
                  * spacing[2],
            };
            const InitialDataSamplePoint sample{physical, physical};
            const EOSThermodynamicState atmosphere =
               initial_atmosphere_state<EOSRuntime>(
                  diagnostic_rotation.atmosphere, sample
               );
            if (initial_matter_exceeds_atmosphere(
                   point.rho, atmosphere.rho
                )) {
               const RotatingStarPoint target = rotating_star_point(
                  diagnostic_rotation, physical, metric
               );
               const amrex::Real flux_value = flux(i, j, k, 0);
               const amrex::Real target_log_enthalpy =
                  target.log_specific_enthalpy
                  + magnetic_k * (
                     flux_value + 0.5 * magnetic_xi
                        * flux_value * flux_value
                  );
               bernoulli(i, j, k, 0) = std::abs(
                  cold_log_specific_enthalpy(point.rho, eos)
                  - target_log_enthalpy
               );
            }
         }
      });
   }
   amrex::Real rhs_norm = 0.0;
   amrex::Real residual_norm = 0.0;
   for (int component = 0; component < 3; ++component) {
      rhs_norm = std::max(
         rhs_norm, workspace.rhs[component]->norm0(0, 0, false)
      );
      residual_norm = std::max(
         residual_norm,
         workspace.equation_residual->norm0(component, 0, false)
      );
   }
   summary.equation_relative_residual = residual_norm / std::max(
      rhs_norm, amrex::Real(1.0e-300)
   );
   summary.bernoulli_residual =
      workspace.bernoulli_residual->norm0(0, 0, false);
   const amrex::Real maximum_comoving_squared =
      workspace.comoving_field_squared->norm0(0, 0, false);
   summary.maximum_comoving_field = std::sqrt(std::max(
      maximum_comoving_squared, amrex::Real(0.0)
   ));
   fill_face_divergence(
      *workspace.divergence, face_magnetic, hierarchy.geometry[0], false
   );
   summary.divergence_linf = workspace.divergence->norm0(0, 0, false);
   if (!std::isfinite(summary.relative_potential_update)
       || !std::isfinite(summary.equation_relative_residual)
       || !std::isfinite(summary.maximum_comoving_field)
       || !std::isfinite(summary.divergence_linf)) {
      amrex::Abort("poloidal Grad-Shafranov equilibrium is nonfinite");
   }
   return summary;
#endif
}

} // namespace pyfluid_amrex
