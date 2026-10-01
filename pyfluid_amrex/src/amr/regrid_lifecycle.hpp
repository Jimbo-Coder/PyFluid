#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace pyfluid_amrex {

struct RegridClock {
   std::int64_t finest_steps = 0;
   std::int64_t last_base_regrid = 0;
};

struct RegridStatistics {
   std::int64_t attempts = 0;
   std::int64_t noops = 0;
   std::int64_t modified_levels = 0;
};

// Bookkeeping only: the driver owns synchronization and AMReX owns topology.
// Level callbacks report changes during one attempt. Cadence counts accepted
// finest-level steps; descendant attempts do not consume the base cadence.
class RegridLifecycle {
public:
   void begin_attempt();
   void mark_level_modified(int level);
   std::optional<int> finish_attempt();
   RegridStatistics const& statistics() const noexcept;

   void advance_finest_step(int interval);
   bool base_regrid_due(int interval) const noexcept;
   bool descendants_regrid_due() const noexcept;
   void complete_base_regrid() noexcept;
   void clear_descendant_request() noexcept;
   RegridClock clock() const noexcept;
   void restore_clock(RegridClock clock, int interval) noexcept;

private:
   RegridClock clock_;
   RegridStatistics statistics_;
   bool descendants_due_ = false;
   bool tracking_changes_ = false;
   int first_modified_level_ = std::numeric_limits<int>::max();
   int last_modified_level_ = -1;
};

} // namespace pyfluid_amrex
