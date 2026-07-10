#pragma once

#include <AMReX_REAL.H>

#include <array>
#include <filesystem>
#include <string>

namespace pyfluid_amrex {

enum class ProfileStage : int {
   BoundaryFill = 0,
   Timestep,
   PrimitiveCache,
   Flux,
   Divergence,
   Sources,
   FluxRegister,
   Update,
   CT,
   Diffusion,
   AverageDown,
   Regrid,
   Diagnostics,
   Output,
   Count
};

inline constexpr int ProfileStageCount = static_cast<int>(ProfileStage::Count);

enum class ProfileDetailStage : int {
   FluxX = 0,
   FluxY,
   FluxZ,
   Count
};

inline constexpr int ProfileDetailStageCount = static_cast<int>(ProfileDetailStage::Count);

const char* profile_stage_name(ProfileStage stage);
const char* profile_detail_stage_name(ProfileDetailStage stage);

class ProfileCounters {
public:
   void configure(std::string profile, std::string detail);

   bool enabled() const;

   bool flux_detail_enabled() const;

   amrex::Real begin() const;

   void end(ProfileStage stage, amrex::Real start);

   void detail_end(ProfileDetailStage stage, amrex::Real start);

   void write(std::filesystem::path const& output_dir, amrex::Real elapsed_wall) const;

   void write_detail(std::filesystem::path const& output_dir, amrex::Real elapsed_wall) const;

private:
   std::string profile_ = "off";
   std::string detail_ = "off";
   std::array<amrex::Real, ProfileStageCount> seconds_{};
   std::array<long long, ProfileStageCount> calls_{};
   std::array<amrex::Real, ProfileDetailStageCount> detail_seconds_{};
   std::array<long long, ProfileDetailStageCount> detail_calls_{};
};

} // namespace pyfluid_amrex
