#include "regrid_lifecycle.hpp"

#include <algorithm>
#include <cassert>

namespace pyfluid_amrex {

void RegridLifecycle::begin_attempt()
{
   assert(!tracking_changes_);
   tracking_changes_ = true;
   first_modified_level_ = std::numeric_limits<int>::max();
   last_modified_level_ = -1;
   ++statistics_.attempts;
}

void RegridLifecycle::mark_level_modified(int level)
{
   // Initial allocation and restart also use AMReX level callbacks.
   if (!tracking_changes_) {
      return;
   }
   first_modified_level_ = std::min(first_modified_level_, level);
   last_modified_level_ = std::max(last_modified_level_, level);
}

std::optional<int> RegridLifecycle::finish_attempt()
{
   assert(tracking_changes_);
   tracking_changes_ = false;
   if (last_modified_level_ < first_modified_level_) {
      ++statistics_.noops;
      return std::nullopt;
   }
   statistics_.modified_levels += last_modified_level_ - first_modified_level_ + 1;
   return first_modified_level_;
}

RegridStatistics const& RegridLifecycle::statistics() const noexcept
{
   return statistics_;
}

void RegridLifecycle::advance_finest_step(int interval)
{
   assert(interval > 0);
   ++clock_.finest_steps;
   descendants_due_ = clock_.finest_steps % interval == 0;
}

bool RegridLifecycle::base_regrid_due(int interval) const noexcept
{
   return clock_.finest_steps - clock_.last_base_regrid >= interval;
}

bool RegridLifecycle::descendants_regrid_due() const noexcept
{
   return descendants_due_;
}

void RegridLifecycle::complete_base_regrid() noexcept
{
   clock_.last_base_regrid = clock_.finest_steps;
}

void RegridLifecycle::clear_descendant_request() noexcept
{
   descendants_due_ = false;
}

RegridClock RegridLifecycle::clock() const noexcept
{
   return clock_;
}

void RegridLifecycle::restore_clock(RegridClock clock, int interval) noexcept
{
   clock_ = clock;
   // Older checkpoints omit the last whole-hierarchy regrid tick.
   if (clock_.last_base_regrid < 0) {
      clock_.last_base_regrid = clock_.finest_steps
         - clock_.finest_steps % std::max(interval, 1);
   }
   descendants_due_ = false;
}

} // namespace pyfluid_amrex
