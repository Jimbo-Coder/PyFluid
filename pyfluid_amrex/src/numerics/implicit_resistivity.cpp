#include "pyfluid_amrex/implicit_resistivity.hpp"

#if (AMREX_SPACEDIM >= 2)

#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLCurlCurl.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/ct_fields.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {
namespace {

using ImplicitEdgeField = amrex::Array<amrex::MultiFab, 3>;

amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> periodic_linear_operator_bc()
{
   amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc{};
   bc.fill(amrex::LinOpBCType::Periodic);
   return bc;
}

DiffusionStats solve_implicit_resistive_edge_emf(
   EdgeElectricField& edge_e,
   FaceMagneticField const& face_b,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   const amrex::BoxArray cell_boxes = amrex::convert(face_b.b[0]->boxArray(), amrex::IntVect::TheCellVector());
   const amrex::DistributionMapping& distribution = face_b.b[0]->DistributionMap();
   ImplicitEdgeField solution;
   ImplicitEdgeField rhs;

   fill_ct_resistive_emf_from_face_b(edge_e, face_b, geom, cfg.resistivity, cfg);
   enforce_edge_electric_boundary(edge_e, geom, cfg);
   amrex::Vector<amrex::MultiFab*> solution_components;
   solution_components.reserve(3);
   for (int component = 0; component < 3; ++component) {
      const amrex::BoxArray edge_boxes = amrex::convert(
         cell_boxes,
         grid_field_index_type(edge_grid_field_centering(component))
      );
      solution[component].define(edge_boxes, distribution, 1, 1);
      rhs[component].define(edge_boxes, distribution, 1, 0);
      amrex::MultiFab::Copy(rhs[component], *edge_e.e[component], 0, 0, 1, 0);
      amrex::MultiFab::Copy(solution[component], rhs[component], 0, 0, 1, 0);
      solution_components.push_back(&solution[component]);
   }
   amrex::FillBoundary(solution_components, geom.periodicity());

   amrex::LPInfo info;
   amrex::MLCurlCurl curl_curl({geom}, {cell_boxes}, {distribution}, info, 0);
   const auto bc = periodic_linear_operator_bc();
   curl_curl.setDomainBC(bc, bc);
   curl_curl.setLevelBC(0, nullptr);
   // E = eta curl(B*) - eta dt curl(curl(E)). The right-hand side is
   // the same generated resistive EMF used by the explicit CT update.
   curl_curl.setScalars(cfg.resistivity * dt, 1.0);
   curl_curl.prepareRHS({&rhs});

   amrex::MLMGT<ImplicitEdgeField> mlmg(curl_curl);
   mlmg.setMaxIter(cfg.diffusion_max_iterations);
   mlmg.setVerbose(0);
   mlmg.setBottomVerbose(0);
   const amrex::Real residual = mlmg.solve({&solution}, {&rhs}, cfg.diffusion_tolerance, 0.0);

   for (int component = 0; component < 3; ++component) {
      amrex::MultiFab::Copy(*edge_e.e[component], solution[component], 0, 0, 1, 0);
   }
   enforce_edge_electric_boundary(edge_e, geom, cfg);
   return {residual, mlmg.getNumIters()};
}

#if (AMREX_SPACEDIM == 2)
DiffusionStats solve_implicit_inactive_magnetic_component(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   const amrex::BoxArray& boxes = state.boxArray();
   const amrex::DistributionMapping& distribution = state.DistributionMap();
   amrex::MultiFab solution(boxes, distribution, 1, 1);
   amrex::MultiFab rhs(boxes, distribution, 1, 0);
   amrex::MultiFab::Copy(solution, state, B3, 0, 1, 0);
   amrex::MultiFab::Copy(rhs, state, B3, 0, 1, 0);
   solution.FillBoundary(geom.periodicity());

   amrex::MultiFab a_coefficient(boxes, distribution, 1, 0);
   a_coefficient.setVal(1.0);
   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> b_coefficient;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      b_coefficient[dir].define(
         amrex::convert(boxes, amrex::IntVect::TheDimensionVector(dir)),
         distribution,
         1,
         0
      );
      b_coefficient[dir].setVal(1.0);
   }

   amrex::LPInfo info;
   amrex::MLABecLaplacian operator_abec({geom}, {boxes}, {distribution}, info);
   const auto bc = periodic_linear_operator_bc();
   operator_abec.setDomainBC(bc, bc);
   operator_abec.setLevelBC(0, nullptr);
   operator_abec.setScalars(1.0, dt * cfg.resistivity);
   operator_abec.setACoeffs(0, a_coefficient);
   operator_abec.setBCoeffs(0, amrex::GetArrOfConstPtrs(b_coefficient));

   amrex::MLMG mlmg(operator_abec);
   mlmg.setMaxIter(cfg.diffusion_max_iterations);
   mlmg.setMaxFmgIter(0);
   mlmg.setVerbose(0);
   mlmg.setBottomVerbose(0);
   const amrex::Real residual = mlmg.solve({&solution}, {&rhs}, cfg.diffusion_tolerance, 0.0);
   amrex::MultiFab::Copy(state, solution, 0, B3, 1, 0);
   return {residual, mlmg.getNumIters()};
}
#endif

} // namespace

DiffusionStats advance_ct_resistive_magnetic_field_implicit(
   FaceMagneticField& face_b,
   EdgeElectricField& edge_e,
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   amrex::Real dt,
   RunConfig const& cfg
)
{
   DiffusionStats stats = solve_implicit_resistive_edge_emf(edge_e, face_b, geom, dt, cfg);
   require_implicit_diffusion_converged(stats, cfg, "edge-centered implicit resistive CT");
   apply_edge_electric_curl(face_b, edge_e, geom, dt, cfg);
#if (AMREX_SPACEDIM == 2)
   const DiffusionStats inactive = solve_implicit_inactive_magnetic_component(state, geom, dt, cfg);
   require_implicit_diffusion_converged(inactive, cfg, "cell-centered inactive magnetic diffusion");
   stats.residual = amrex::max(stats.residual, inactive.residual);
   stats.iterations += inactive.iterations;
#else
   amrex::ignore_unused(state);
#endif
   return stats;
}

} // namespace pyfluid_amrex

#endif
