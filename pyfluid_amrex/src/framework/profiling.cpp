#include "pyfluid_amrex/profiling.hpp"

#include <AMReX_ParallelDescriptor.H>

#include <fstream>
#include <iomanip>
#include <utility>

namespace pyfluid_amrex {

const char* profile_stage_name(ProfileStage stage)
{
   switch (stage) {
   case ProfileStage::BoundaryFill: return "boundary_fill";
   case ProfileStage::Timestep: return "timestep";
   case ProfileStage::PrimitiveCache: return "primitive_cache";
   case ProfileStage::Flux: return "flux";
   case ProfileStage::Divergence: return "divergence";
   case ProfileStage::Sources: return "sources";
   case ProfileStage::FluxRegister: return "flux_register";
   case ProfileStage::Update: return "update";
   case ProfileStage::CT: return "ct";
   case ProfileStage::Diffusion: return "diffusion";
   case ProfileStage::AverageDown: return "average_down";
   case ProfileStage::Regrid: return "regrid";
   case ProfileStage::Diagnostics: return "diagnostics";
   case ProfileStage::Output: return "output";
   case ProfileStage::Count: return "count";
   }
   return "unknown";
}

const char* profile_detail_stage_name(ProfileDetailStage stage)
{
   switch (stage) {
   case ProfileDetailStage::FluxX: return "flux_x";
   case ProfileDetailStage::FluxY: return "flux_y";
   case ProfileDetailStage::FluxZ: return "flux_z";
   case ProfileDetailStage::Count: return "count";
   }
   return "unknown";
}

void ProfileCounters::configure(std::string profile, std::string detail)
{
   profile_ = std::move(profile);
   detail_ = std::move(detail);
}

bool ProfileCounters::enabled() const
{
   return profile_ != "off";
}

bool ProfileCounters::flux_detail_enabled() const
{
   return enabled() && detail_ == "flux";
}

WallTime ProfileCounters::begin() const
{
   return enabled() ? amrex::ParallelDescriptor::second() : 0.0;
}

void ProfileCounters::end(ProfileStage stage, WallTime start)
{
   if (!enabled()) {
      return;
   }
   const int index = static_cast<int>(stage);
   seconds_[index] += amrex::ParallelDescriptor::second() - start;
   calls_[index] += 1;
}

void ProfileCounters::detail_end(ProfileDetailStage stage, WallTime start)
{
   if (!flux_detail_enabled()) {
      return;
   }
   const int index = static_cast<int>(stage);
   detail_seconds_[index] += amrex::ParallelDescriptor::second() - start;
   detail_calls_[index] += 1;
}

void ProfileCounters::write(std::filesystem::path const& output_dir, WallTime elapsed_wall) const
{
   if (!enabled()) {
      return;
   }
   std::array<WallTime, ProfileStageCount> max_seconds = seconds_;
   for (WallTime& seconds : max_seconds) {
      amrex::ParallelDescriptor::ReduceRealMax(seconds, amrex::ParallelDescriptor::IOProcessorNumber());
   }
   if (!amrex::ParallelDescriptor::IOProcessor()) {
      return;
   }
   std::ofstream out(output_dir / "profile.tsv", std::ios::out | std::ios::trunc);
   out << "stage\trank_max_seconds\tfraction_of_run_wall\tcalls\n";
   for (int index = 0; index < ProfileStageCount; ++index) {
      const WallTime fraction = elapsed_wall > 0.0 ? max_seconds[index] / elapsed_wall : 0.0;
      out << profile_stage_name(static_cast<ProfileStage>(index)) << '\t'
          << std::setprecision(17) << max_seconds[index] << '\t'
          << fraction << '\t'
          << calls_[index] << '\n';
   }
}

void ProfileCounters::write_detail(std::filesystem::path const& output_dir, WallTime elapsed_wall) const
{
   if (!flux_detail_enabled()) {
      return;
   }
   std::array<WallTime, ProfileDetailStageCount> max_seconds = detail_seconds_;
   for (WallTime& seconds : max_seconds) {
      amrex::ParallelDescriptor::ReduceRealMax(seconds, amrex::ParallelDescriptor::IOProcessorNumber());
   }
   if (!amrex::ParallelDescriptor::IOProcessor()) {
      return;
   }
   std::ofstream out(output_dir / "profile_detail.tsv", std::ios::out | std::ios::trunc);
   out << "stage\trank_max_seconds\tfraction_of_run_wall\tcalls\n";
   for (int index = 0; index < ProfileDetailStageCount; ++index) {
#if (AMREX_SPACEDIM < 3)
      if (static_cast<ProfileDetailStage>(index) == ProfileDetailStage::FluxZ) {
         continue;
      }
#endif
#if (AMREX_SPACEDIM < 2)
      if (static_cast<ProfileDetailStage>(index) == ProfileDetailStage::FluxY) {
         continue;
      }
#endif
      const WallTime fraction = elapsed_wall > 0.0 ? max_seconds[index] / elapsed_wall : 0.0;
      out << profile_detail_stage_name(static_cast<ProfileDetailStage>(index)) << '\t'
          << std::setprecision(17) << max_seconds[index] << '\t'
          << fraction << '\t'
          << detail_calls_[index] << '\n';
   }
}

} // namespace pyfluid_amrex
