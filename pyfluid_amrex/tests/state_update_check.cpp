#include <AMReX.H>
#include <AMReX_CArena.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <limits>

#include "pyfluid_amrex/generated/time_integrator_catalog.hpp"
#include "pyfluid_amrex/state_update.hpp"
#include "pyfluid_amrex/diagnostic_reductions.hpp"

namespace {

bool check_box_launch_ownership()
{
   // Box-loop launches, used by AMReX flux registers, must partition work
   // just as pointwise ParallelFor does, including a partial final box.
   const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(258));
   int host_ranges = 0;
   for (const auto& work_box : amrex::Gpu::Range(domain)) {
      if (work_box != domain) return false;
      ++host_ranges;
   }
   if (host_ranges != 1) return false;
   amrex::BoxArray boxes(domain);
   boxes.maxSize(128);
   amrex::DistributionMapping mapping(boxes);
   amrex::iMultiFab visits(boxes, mapping, 1, 0);
   visits.setVal(0);
   for (amrex::MFIter mfi(visits); mfi.isValid(); ++mfi) {
      const auto counts = visits.array(mfi);
      AMREX_LAUNCH_HOST_DEVICE_LAMBDA(mfi.validbox(), work_box, {
         const auto lower = amrex::lbound(work_box);
         const auto upper = amrex::ubound(work_box);
         for (int i = lower.x; i <= upper.x; ++i) {
            amrex::HostDevice::Atomic::Add(&counts(i, 0, 0), 1);
         }
      });
   }
   const int minimum = visits.min(0);
   const int maximum = visits.max(0);
   const bool passed = minimum == 1 && maximum == 1;
   amrex::Print() << "box launch ownership: min=" << minimum << " max=" << maximum
                  << " passed=" << passed << '\n';
   return passed;
}

bool check_diagnostic_sum()
{
   using namespace amrex::literals;
   constexpr int cells = 8192;
   // Identical binary32 data has an exact FP64 sum, while a linear FP32
   // accumulator loses residuals over this many additions.
   constexpr amrex::Real value = static_cast<amrex::Real>(1.0820445F);
   amrex::BoxArray boxes(amrex::Box(amrex::IntVect(0), amrex::IntVect(cells - 1)));
   boxes.maxSize(2048);
   amrex::DistributionMapping mapping(boxes);
   amrex::MultiFab field(boxes, mapping, 1, 0);
   field.setVal(value);
   const amrex::Real expected = cells * value;
   const amrex::Real total = pyfluid_amrex::diagnostic_component_sum(field, 0);
   const bool passed = std::abs(total - expected)
      <= 2.0_rt * std::numeric_limits<amrex::Real>::epsilon() * expected;
   amrex::Print() << "diagnostic sum: total=" << total << " expected=" << expected
                  << " passed=" << passed << '\n';
   return passed;
}

bool check_masked_cell_maximum()
{
   using namespace amrex::literals;
   const amrex::Box domain(amrex::IntVect(-32), amrex::IntVect(226));
   amrex::BoxArray boxes(domain);
   boxes.maxSize(64);
   amrex::DistributionMapping mapping(boxes);
   amrex::MultiFab field(boxes, mapping, 1, 1);
   amrex::iMultiFab mask(boxes, mapping, 1, 1);
   field.setVal(1000.0_rt);  // Ghosts must never win the valid-cell maximum.
   mask.setVal(1);
   constexpr amrex::Real infinity = std::numeric_limits<amrex::Real>::infinity();
   for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
      const auto values = field.array(mfi);
      const auto valid = mask.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         values(i, j, k) = i == -2 || i == 150 ? 7.0_rt : 2.0_rt;
         if (i == 100) {
            values(i, j, k) = infinity;
            valid(i, j, k) = 0;  // Covered/masked cells must not participate.
         }
      });
   }
   const auto local = pyfluid_amrex::diagnostic_masked_maximum(field, mask, domain, 0);
   amrex::Real maximum = local.value;
   amrex::ParallelDescriptor::ReduceRealMax(maximum);
   amrex::Long index = local.value == maximum
      ? local.cell_index : std::numeric_limits<amrex::Long>::max();
   amrex::ParallelDescriptor::ReduceLongMin(index);
   bool passed = maximum == 7.0_rt && index == 30;
   mask.setVal(0);
   const auto empty = pyfluid_amrex::diagnostic_masked_maximum(field, mask, domain, 0);
   passed = passed && empty.cell_index == -1 && empty.value == -infinity;
   amrex::Print() << "masked cell maximum: value=" << maximum << " index=" << index
                  << " passed=" << passed << '\n';
   return passed;
}

bool check_state_storage_and_update()
{
   using namespace amrex::literals;
   constexpr int cells = 256;
   amrex::BoxArray boxes(amrex::Box(amrex::IntVect(0), amrex::IntVect(cells - 1)));
   boxes.maxSize(64);
   amrex::DistributionMapping mapping(boxes);

   // Array4/BoxIndexer metadata uses The_Arena. Put the actual fields in a
   // different allocation so indirect resource access cannot pass accidentally
   // because an explicitly bound metadata buffer also contains the field.
   amrex::ArenaInfo arena_info;
#ifdef AMREX_USE_GPU
   arena_info.SetDeviceMemory();
#endif
   amrex::CArena field_arena(131072, arena_info);
   const auto info = amrex::MFInfo().SetArena(&field_arena);
   amrex::MultiFab state(boxes, mapping, 2, 0, info);
   amrex::iMultiFab mask(boxes, mapping, 1, 0, info);
   state.setVal(3.0_rt, 0, 1);
   state.setVal(7.0_rt, 1, 1);
   mask.setVal(1);

   using Reducer = amrex::Reducer<
      amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpMin, amrex::ReduceOpMax>,
      amrex::TypeList<amrex::Long, amrex::Real, amrex::Real>>;
   const auto values = state.const_arrays();
   const auto masks = mask.const_arrays();
   const amrex::Real invalid = std::numeric_limits<amrex::Real>::max();
   Reducer reducer;
   reducer.eval(state, amrex::IntVect(0),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> Reducer::Result_t {
         if (masks[box_no](i, j, k, 0) == 0) {
            return {0, invalid, -invalid};
         }
         return {1, values[box_no](i, j, k, 0), values[box_no](i, j, k, 1)};
      });
   auto result = reducer.getResult();
   auto count = amrex::get<0>(result);
   auto minimum = amrex::get<1>(result);
   auto maximum = amrex::get<2>(result);
   amrex::ParallelDescriptor::ReduceLongSum(count);
   amrex::ParallelDescriptor::ReduceRealMin(minimum);
   amrex::ParallelDescriptor::ReduceRealMax(maximum);
   const bool passed = count == cells && minimum == 3.0_rt && maximum == 7.0_rt;
   amrex::Print() << "indirect storage: count=" << count << " min=" << minimum
                  << " max=" << maximum << " passed=" << passed << '\n';
   if (!passed) return false;

   // A zero-RHS RK stage must preserve every representable constant state,
   // including values for which separately weighted copies round differently.
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const auto data = state.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         for (int q = 0; q < 2; ++q) {
            data(i, j, k, q) = 0.9_rt + (i + q * cells) * 0.00002_rt;
         }
      });
   }
   amrex::MultiFab output(boxes, mapping, 2, 0, info);
   amrex::MultiFab rhs(boxes, mapping, 2, 0, info);
   rhs.setVal(0.0_rt);
   const auto check_stages = [&](auto const& stages) {
      for (auto const& stage : stages) {
         pyfluid_amrex::affine_combination_plus_scaled_components<2>(
            output, state, state, rhs, stage.input_weight, 1.0_rt
         );
         amrex::MultiFab::Subtract(output, state, 0, 0, 2, 0);
         if (output.norm0(0) != 0.0_rt || output.norm0(1) != 0.0_rt) {
            amrex::Print() << "RK blend changed a constant state with zero RHS\n";
            return false;
         }
      }
      return true;
   };
   using namespace pyfluid_amrex;
   return check_stages(time_integrator_stages<TimeEuler>())
      && check_stages(time_integrator_stages<TimeCorrector>())
      && check_stages(time_integrator_stages<TimeMidpoint>())
      && check_stages(time_integrator_stages<TimeSSPRK3>())
      && check_stages(time_integrator_stages<TimeRK4>());
}

} // namespace

int main(int argc, char** argv)
{
   amrex::Initialize(argc, argv);
   const bool passed = check_state_storage_and_update() && check_diagnostic_sum()
      && check_box_launch_ownership() && check_masked_cell_maximum();
   amrex::Print() << "state storage/update: passed=" << passed << '\n';
   amrex::Finalize();
   return passed ? 0 : 1;
}
