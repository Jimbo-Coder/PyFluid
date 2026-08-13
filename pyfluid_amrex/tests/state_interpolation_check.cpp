#include <AMReX.H>
#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_OpenMP.H>
#include <AMReX_RealBox.H>

#include "amr/state_interpolation.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "relativity/relativistic_atmosphere.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

using namespace pyfluid_amrex;

constexpr int NumComponents = 8;
constexpr int CoarseCenter = 686;
constexpr int FineLeft = 1372;
constexpr int FineRight = 1373;

using State = std::array<amrex::Real, NumComponents>;

struct MinkowskiMetricAccessor {
   AMREX_GPU_HOST_DEVICE
   SpacetimeMetric operator()(int, int, int) const noexcept
   {
      return minkowski_spacetime_metric();
   }
};

// Captured from the accepted L3 state immediately before the Balsara-3
// step-24 L3-to-L4 regrid. AMReX's coupled conservative interpolation of
// this physical coarse stencil preserves the parent average but creates one
// nonphysical child.
constexpr std::array<State, 3> CoarseStencil{{
   {{
      2.1931306385515779,
      22.495734802485632,
      -26.609584540900791,
      -26.609544467491101,
      72.601985874927166,
      10.0,
      3.4113931078726103,
      3.4113872685502082,
   }},
   {{
      1.3699592236632692,
      6.8011337171385131,
      -14.773294585508712,
      -14.773284706240204,
      57.37705792368817,
      10.0,
      2.1928094578507338,
      2.1928102038848603,
   }},
   {{
      0.98281245229913061,
      2.6424267903033103,
      -8.2115307564322428,
      -8.2115403976466528,
      53.399264193074906,
      10.0,
      1.6013592363971536,
      1.6013598254512778,
   }},
}};

constexpr std::array<State, 2> RawChildren{{
   {{
      1.495316352019509,
      8.8574142134971776,
      -16.67884902231825,
      -16.67883399392236,
      59.365954788994799,
      10.0,
      2.3802813565076075,
      2.3802814367310154,
   }},
   {{
      1.2446020953070294,
      4.7448532207798486,
      -12.867740148699172,
      -12.867735418558047,
      55.388161058381542,
      10.0,
      2.00533755919386,
      2.0053389710387051,
   }},
}};

bool close_enough(
   amrex::Real actual,
   amrex::Real expected,
   amrex::Real tolerance = 8.0e-14
)
{
   const amrex::Real scale = std::max(
      1.0,
      std::max(std::abs(actual), std::abs(expected))
   );
   return std::abs(actual - expected) <= tolerance * scale;
}

void require(bool condition, std::string const& message)
{
   if (!condition) {
      throw std::runtime_error(message);
   }
}

void require_close(
   std::string const& label,
   amrex::Real actual,
   amrex::Real expected,
   amrex::Real tolerance = 8.0e-14
)
{
   if (!close_enough(actual, expected, tolerance)) {
      throw std::runtime_error(
         label + ": got " + std::to_string(actual)
         + ", expected " + std::to_string(expected)
      );
   }
}

amrex::Box cell_box(int lo, int hi)
{
   return {
      amrex::IntVect(AMREX_D_DECL(lo, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(hi, 0, 0)),
   };
}

amrex::Geometry make_geometry(int cells)
{
   const amrex::Box domain = cell_box(0, cells - 1);
   const amrex::RealBox physical_domain(
      AMREX_D_DECL(-1.0, 0.0, 0.0),
      AMREX_D_DECL(1.0, 1.0, 1.0)
   );
   const amrex::Array<int, AMREX_SPACEDIM> periodic{
      AMREX_D_DECL(0, 0, 0)
   };
   return amrex::Geometry(
      domain,
      physical_domain,
      amrex::CoordSys::cartesian,
      periodic
   );
}

amrex::Vector<amrex::BCRec> boundary_conditions()
{
   amrex::Vector<amrex::BCRec> records(NumComponents);
   for (amrex::BCRec& record : records) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         record.setLo(direction, amrex::BCType::foextrap);
         record.setHi(direction, amrex::BCType::foextrap);
      }
   }
   return records;
}

void fill_coarse(
   amrex::FArrayBox& coarse,
   amrex::Real stencil_scale = 1.0
)
{
   State const& parent = CoarseStencil[1];
   for (int offset = -1; offset <= 1; ++offset) {
      for (int component = 0; component < NumComponents; ++component) {
         coarse(
            amrex::IntVect(AMREX_D_DECL(CoarseCenter + offset, 0, 0)),
            component
         ) = parent[component] + stencil_scale * (
            CoarseStencil[offset + 1][component] - parent[component]
         );
      }
   }
}

State read_state(amrex::FArrayBox const& fab, int i)
{
   State state{};
   const amrex::IntVect cell(AMREX_D_DECL(i, 0, 0));
   for (int component = 0; component < NumComponents; ++component) {
      state[component] = fab(cell, component);
   }
   return state;
}

RMHDConserved conserved(State const& state)
{
   return {
      state[RHO],
      state[MOM1],
      state[MOM2],
      state[MOM3],
      state[ENERGY],
      state[B1],
      state[B2],
      state[B3],
   };
}

void interpolate(
   amrex::Interpolater& interpolater,
   amrex::FArrayBox const& coarse,
   amrex::FArrayBox& fine,
   amrex::Geometry const& coarse_geometry,
   amrex::Geometry const& fine_geometry,
   amrex::Box const& fine_region = cell_box(FineLeft, FineRight)
)
{
   interpolater.interp(
      coarse,
      0,
      fine,
      0,
      NumComponents,
      fine_region,
      amrex::IntVect(AMREX_D_DECL(2, 1, 1)),
      coarse_geometry,
      fine_geometry,
      boundary_conditions(),
      RHO,
      0,
      amrex::RunOn::Host
   );
}

MeshBasisStateInterpolater make_protected_interpolater(EOSParams const& eos);

void check_split_sibling_interpolation_matches_complete_family()
{
   const amrex::Geometry coarse_geometry = make_geometry(1024);
   const amrex::Geometry fine_geometry = make_geometry(2048);
   const amrex::Box coarse_box = cell_box(CoarseCenter - 1, CoarseCenter + 1);
   amrex::FArrayBox coarse(coarse_box, NumComponents);
   amrex::FArrayBox complete(cell_box(FineLeft, FineRight), NumComponents);
   amrex::FArrayBox split_left(cell_box(FineLeft, FineLeft), NumComponents);
   amrex::FArrayBox split_right(cell_box(FineRight, FineRight), NumComponents);
   fill_coarse(coarse);

   const EOSParams eos{};
   MeshBasisStateInterpolater interpolater = make_protected_interpolater(eos);
   interpolate(interpolater, coarse, complete, coarse_geometry, fine_geometry);
   interpolate(
      interpolater,
      coarse,
      split_left,
      coarse_geometry,
      fine_geometry,
      cell_box(FineLeft, FineLeft)
   );
   interpolate(
      interpolater,
      coarse,
      split_right,
      coarse_geometry,
      fine_geometry,
      cell_box(FineRight, FineRight)
   );

   const State complete_left = read_state(complete, FineLeft);
   const State complete_right = read_state(complete, FineRight);
   const State left = read_state(split_left, FineLeft);
   const State right = read_state(split_right, FineRight);
   for (int component = 0; component < NumComponents; ++component) {
      require(
         left[component] == complete_left[component],
         "left child depends on interpolation-region decomposition"
      );
      require(
         right[component] == complete_right[component],
         "right child depends on interpolation-region decomposition"
      );
      require_close(
         "split protected child average",
         0.5 * (left[component] + right[component]),
         CoarseStencil[1][component]
      );
   }
}

MeshBasisStateInterpolater make_protected_interpolater(EOSParams const& eos)
{
   MeshAxes axes{};
   axes.axis[0] = MeshAxisX;
   MeshBasisStateInterpolater interpolater;
   interpolater.configure(
      &amrex::lincc_interp,
      MeshCartesian,
      axes,
      false,
      PhysicalSystem::RMHD,
      EOSIdealGas,
      eos,
      FixedSpacetimeParameters{},
      true
   );
   return interpolater;
}

void check_captured_stencil()
{
   const amrex::Geometry coarse_geometry = make_geometry(1024);
   const amrex::Geometry fine_geometry = make_geometry(2048);
   const amrex::Box coarse_box = cell_box(CoarseCenter - 1, CoarseCenter + 1);
   const amrex::Box fine_box = cell_box(FineLeft, FineRight);
   amrex::FArrayBox coarse(coarse_box, NumComponents);
   amrex::FArrayBox raw_fine(fine_box, NumComponents);
   amrex::FArrayBox protected_fine(fine_box, NumComponents);
   fill_coarse(coarse);

   interpolate(
      amrex::lincc_interp,
      coarse,
      raw_fine,
      coarse_geometry,
      fine_geometry
   );
   for (int child = 0; child < 2; ++child) {
      const State raw = read_state(raw_fine, FineLeft + child);
      for (int component = 0; component < NumComponents; ++component) {
         require_close(
            "captured raw child",
            raw[component],
            RawChildren[child][component]
         );
      }
   }

   const EOSParams eos{};
   const SpacetimeMetric metric = minkowski_spacetime_metric();
   const RMHDRecoveryResult coarse_parent = recover_srmhd(
      conserved(CoarseStencil[1]), metric, eos.gamma
   );
   const RMHDRecoveryResult raw_left = recover_srmhd(
      conserved(read_state(raw_fine, FineLeft)), metric, eos.gamma
   );
   const RMHDRecoveryResult raw_right = recover_srmhd(
      conserved(read_state(raw_fine, FineRight)), metric, eos.gamma
   );
   require(coarse_parent.succeeded(), "captured coarse parent should be admissible");
   require(raw_left.succeeded(), "captured left child should be admissible");
   require(!raw_right.succeeded(), "captured right child should be nonphysical");

   MeshBasisStateInterpolater protected_interpolater =
      make_protected_interpolater(eos);
   interpolate(
      protected_interpolater,
      coarse,
      protected_fine,
      coarse_geometry,
      fine_geometry
   );

   const State parent = CoarseStencil[1];
   const State raw_left_state = read_state(raw_fine, FineLeft);
   const State raw_right_state = read_state(raw_fine, FineRight);
   for (int component = 0; component < NumComponents; ++component) {
      require_close(
         "raw child average",
         0.5 * (raw_left_state[component] + raw_right_state[component]),
         parent[component]
      );
   }
   amrex::Real common_theta = -1.0;
   for (int child = 0; child < 2; ++child) {
      const State raw = read_state(raw_fine, FineLeft + child);
      const State limited = read_state(protected_fine, FineLeft + child);
      const RMHDRecoveryResult recovered = recover_srmhd(
         conserved(limited), metric, eos.gamma
      );
      require(recovered.succeeded(), "protected child failed C2P");
      require(
         recovered.primitive.rho >= eos.rho_floor
            && recovered.primitive.pressure >= eos.pressure_floor,
         "protected child violated the configured floors"
      );

      for (int component = 0; component < NumComponents; ++component) {
         const amrex::Real raw_deviation = raw[component] - parent[component];
         if (raw_deviation == 0.0) {
            require_close(
               "zero-deviation component",
               limited[component],
               parent[component]
            );
            continue;
         }
         const amrex::Real theta = (
            limited[component] - parent[component]
         ) / raw_deviation;
         if (common_theta < 0.0) {
            common_theta = theta;
         } else {
            require_close("shared child theta", theta, common_theta, 2.0e-12);
         }
      }
   }
   require(
      common_theta >= 0.60 && common_theta <= 0.631,
      "protected interpolation did not retain the near-maximal admissible slope"
   );

   const State limited_left = read_state(protected_fine, FineLeft);
   const State limited_right = read_state(protected_fine, FineRight);
   for (int component = 0; component < NumComponents; ++component) {
      require_close(
         "protected child average",
         0.5 * (limited_left[component] + limited_right[component]),
         parent[component]
      );
   }
}

void check_admissible_interpolation_is_unchanged()
{
   const amrex::Geometry coarse_geometry = make_geometry(1024);
   const amrex::Geometry fine_geometry = make_geometry(2048);
   const amrex::Box coarse_box = cell_box(CoarseCenter - 1, CoarseCenter + 1);
   const amrex::Box fine_box = cell_box(FineLeft, FineRight);
   amrex::FArrayBox coarse(coarse_box, NumComponents);
   amrex::FArrayBox raw_fine(fine_box, NumComponents);
   amrex::FArrayBox protected_fine(fine_box, NumComponents);
   fill_coarse(coarse, 0.1);

   interpolate(
      amrex::lincc_interp,
      coarse,
      raw_fine,
      coarse_geometry,
      fine_geometry
   );
   const EOSParams eos{};
   MeshBasisStateInterpolater protected_interpolater =
      make_protected_interpolater(eos);
   interpolate(
      protected_interpolater,
      coarse,
      protected_fine,
      coarse_geometry,
      fine_geometry
   );

   const SpacetimeMetric metric = minkowski_spacetime_metric();
   for (int child = 0; child < 2; ++child) {
      const State raw = read_state(raw_fine, FineLeft + child);
      const State limited = read_state(protected_fine, FineLeft + child);
      require(
         recover_srmhd(conserved(raw), metric, eos.gamma).succeeded(),
         "smooth-control child should be admissible"
      );
      for (int component = 0; component < NumComponents; ++component) {
         require(
            limited[component] == raw[component],
            "admissible interpolation was modified"
         );
      }
   }
}

void check_post_magnetic_sync_atmosphere_backstop_preserves_b()
{
   const amrex::Box box = cell_box(0, 0);
   amrex::FArrayBox state(box, NumComponents);
   const auto values = state.array();
   values(0, 0, 0, RHO) = 1.0;
   values(0, 0, 0, MOM1) = 0.0;
   values(0, 0, 0, MOM2) = 0.0;
   values(0, 0, 0, MOM3) = 0.0;
   values(0, 0, 0, ENERGY) = 0.15;
   values(0, 0, 0, B1) = 0.6;
   values(0, 0, 0, B2) = 0.0;
   values(0, 0, 0, B3) = 0.0;

   const EOSParams eos{};
   const SpacetimeMetric metric = minkowski_spacetime_metric();
   require(
      !recover_srmhd(load_rmhd_conserved(values, 0, 0, 0), metric, eos.gamma)
          .succeeded(),
      "post-sync fixture should be non-admissible before the backstop"
   );
   const EOSThermodynamicState atmosphere = eos_project_to_floors<EOSIdealGas>(
      eos.rho_floor, eos.pressure_floor, eos
   );
   RelativisticRecoveryCellEvents events;
   const RMHDRecoveryResult recovered =
      recover_relativistic_primitive_with_atmosphere<
         PhysicalSystem::RMHD, EOSIdealGas
      >(values, 0, 0, 0, eos, metric, atmosphere, &events);
   require(recovered.succeeded(), "post-sync atmosphere backstop did not recover");
   require(events.atmosphere_reset_count == 1, "post-sync reset was not counted");
   require(values(0, 0, 0, B1) == 0.6, "post-sync reset changed canonical B1");
   require(values(0, 0, 0, B2) == 0.0, "post-sync reset changed canonical B2");
   require(values(0, 0, 0, B3) == 0.0, "post-sync reset changed canonical B3");
}

void check_tiled_ghost_projection_has_unique_valid_cell_owner()
{
   const amrex::Box valid_box = cell_box(0, 7);
   const amrex::BoxArray boxes(valid_box);
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, NumComponents, 1);
   state.setVal(0.0);
   state.setVal(1.0, RHO, 1, 1);
   state.setVal(0.15, ENERGY, 1, 1);

   const amrex::IntVect invalid_cell(AMREX_D_DECL(4, 0, 0));
   for (amrex::MFIter mfi(state, false); mfi.isValid(); ++mfi) {
      if (mfi.validbox().contains(invalid_cell)) {
         state.array(mfi)(4, 0, 0, B1) = 0.6;
      }
   }

   const EOSParams eos{};
   const EOSThermodynamicState atmosphere = eos_project_to_floors<EOSIdealGas>(
      eos.rho_floor, eos.pressure_floor, eos
   );
   amrex::Long atmosphere_resets = 0;

   const amrex::IntVect previous_tile_size =
      amrex::FabArrayBase::mfiter_tile_size;
   const int previous_thread_count = amrex::OpenMP::get_max_threads();
   amrex::FabArrayBase::mfiter_tile_size = amrex::IntVect(4);
   amrex::OpenMP::set_num_threads(1);
   project_relativistic_atmosphere<
      PhysicalSystem::RMHD, EOSIdealGas
   >(
      state,
      eos,
      atmosphere,
      [](amrex::MFIter const&) { return MinkowskiMetricAccessor{}; },
      1,
      nullptr,
      &atmosphere_resets
   );
   amrex::OpenMP::set_num_threads(previous_thread_count);
   amrex::FabArrayBase::mfiter_tile_size = previous_tile_size;

   require(
      atmosphere_resets == 1,
      "tiled ghost projection did not count the invalid valid cell exactly once"
   );
   for (amrex::MFIter mfi(state, false); mfi.isValid(); ++mfi) {
      if (!mfi.validbox().contains(invalid_cell)) {
         continue;
      }
      const auto values = state.const_array(mfi);
      const RMHDRecoveryResult recovered = recover_srmhd(
         load_rmhd_conserved(values, 4, 0, 0),
         minkowski_spacetime_metric(),
         eos.gamma
      );
      require(recovered.succeeded(), "tiled ghost projection did not recover the cell");
      require(values(4, 0, 0, B1) == 0.6, "tiled ghost projection changed B1");
   }
}

} // namespace

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   int status = 0;
   try {
      check_captured_stencil();
      check_admissible_interpolation_is_unchanged();
      check_split_sibling_interpolation_matches_complete_family();
      check_post_magnetic_sync_atmosphere_backstop_preserves_b();
      check_tiled_ghost_projection_has_unique_valid_cell_owner();
      amrex::Print()
         << "PASS relativistic conservative state interpolation check\n";
   } catch (std::exception const& error) {
      amrex::Print() << "FAIL " << error.what() << '\n';
      status = 1;
   }
   amrex::Finalize();
   return status;
}
