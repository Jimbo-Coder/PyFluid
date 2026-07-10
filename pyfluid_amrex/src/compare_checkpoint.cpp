#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "pyfluid_amrex/checkpoint.hpp"
#include "pyfluid_amrex/fields.hpp"

namespace {

struct FieldDiff {
   amrex::Real max_abs = 0.0;
   amrex::Real l1 = 0.0;
};

void print_usage(char const* exe)
{
   amrex::Print()
      << "usage: " << exe << " checkpoint_a checkpoint_b [tolerance] [--fields-only]\n"
      << "Compares PyFluid AMReX checkpoint headers, state MultiFabs, and any face_b MultiFabs.\n"
      << "--fields-only ignores step/dt/time metadata differences in the exit status.\n";
}

bool nearly_equal(amrex::Real a, amrex::Real b, amrex::Real tol)
{
   return std::abs(a - b) <= tol;
}

amrex::Real max_vector_difference(amrex::Vector<amrex::Real> const& a, amrex::Vector<amrex::Real> const& b)
{
   amrex::Real diff = 0.0;
   const int n = std::min(a.size(), b.size());
   for (int i = 0; i < n; ++i) {
      diff = std::max(diff, std::abs(a[i] - b[i]));
   }
   return diff;
}

FieldDiff compare_multifab(
   std::string const& checkpoint_a,
   std::string const& checkpoint_b,
   int level,
   std::string const& field_name,
   amrex::BoxArray const& boxes,
   int num_components
)
{
   const amrex::DistributionMapping distribution_a(boxes);
   const amrex::DistributionMapping distribution_b(boxes);
   amrex::MultiFab a(boxes, distribution_a, num_components, 0);
   amrex::MultiFab b(boxes, distribution_b, num_components, 0);
   pyfluid_amrex::read_checkpoint_field(a, checkpoint_a, level, field_name);
   pyfluid_amrex::read_checkpoint_field(b, checkpoint_b, level, field_name);

   amrex::MultiFab diff;
   diff.define(a.boxArray(), b.DistributionMap(), a.nComp(), 0);
#ifdef AMREX_USE_MPI
   amrex::MultiFab tmp(a.boxArray(), a.DistributionMap(), a.nComp(), 0);
   amrex::MultiFab::Copy(tmp, a, 0, 0, a.nComp(), 0);
   diff.Redistribute(tmp, 0, 0, a.nComp(), amrex::IntVect(0));
#else
   amrex::MultiFab::Copy(diff, a, 0, 0, a.nComp(), 0);
#endif
   amrex::MultiFab::Subtract(diff, b, 0, 0, a.nComp(), 0);

   amrex::Vector<int> comps(a.nComp());
   std::iota(comps.begin(), comps.end(), 0);
   const amrex::Vector<amrex::Real> max_by_comp = diff.norm0(comps, 0, false);
   const amrex::Vector<amrex::Real> l1_by_comp = diff.norm1(comps, 0, false);

   FieldDiff summary;
   for (amrex::Real value : max_by_comp) {
      summary.max_abs = std::max(summary.max_abs, value);
   }
   for (amrex::Real value : l1_by_comp) {
      summary.l1 += value;
   }
   return summary;
}

} // namespace

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);

   int return_code = 0;
   {
      if (argc < 3 || argc > 5) {
         print_usage(argv[0]);
         amrex::Finalize();
         return 1;
      }

      const std::string checkpoint_a = argv[1];
      const std::string checkpoint_b = argv[2];
      amrex::Real tolerance = 0.0;
      bool fields_only = false;
      for (int argi = 3; argi < argc; ++argi) {
         const std::string arg = argv[argi];
         if (arg == "--fields-only") {
            fields_only = true;
         } else {
            tolerance = std::stod(arg);
         }
      }

      const pyfluid_amrex::CheckpointHeader header_a =
         pyfluid_amrex::read_checkpoint_header(checkpoint_a);
      const pyfluid_amrex::CheckpointHeader header_b =
         pyfluid_amrex::read_checkpoint_header(checkpoint_b);

      bool metadata_ok = true;
      if (header_a.finest_level != header_b.finest_level) {
         metadata_ok = false;
         amrex::Print() << "finest_level differs: "
                        << header_a.finest_level << " vs " << header_b.finest_level << "\n";
      }
      if (header_a.state_components != header_b.state_components) {
         metadata_ok = false;
         amrex::Print() << "state component count differs: "
                        << header_a.state_components << " vs " << header_b.state_components << "\n";
      }

      const int finest = std::min(header_a.finest_level, header_b.finest_level);
      if (header_a.level_steps.size() != header_b.level_steps.size() ||
          header_a.dt.size() != header_b.dt.size() ||
          header_a.time.size() != header_b.time.size()) {
         metadata_ok = false;
         amrex::Print() << "checkpoint metadata vector sizes differ\n";
      }

      amrex::Real max_dt_diff = max_vector_difference(header_a.dt, header_b.dt);
      amrex::Real max_time_diff = max_vector_difference(header_a.time, header_b.time);
      amrex::Real reference_mass_diff = 0.0;
      amrex::Real reference_energy_diff = 0.0;
      const bool has_reference_a =
         std::isfinite(header_a.reference_mass) && std::isfinite(header_a.reference_energy);
      const bool has_reference_b =
         std::isfinite(header_b.reference_mass) && std::isfinite(header_b.reference_energy);
      if (has_reference_a != has_reference_b) {
         metadata_ok = false;
         amrex::Print() << "checkpoint scalar references exist in only one checkpoint\n";
      } else if (has_reference_a) {
         reference_mass_diff = std::abs(header_a.reference_mass - header_b.reference_mass);
         reference_energy_diff = std::abs(header_a.reference_energy - header_b.reference_energy);
      }
      if (!fields_only &&
          (!nearly_equal(max_dt_diff, 0.0, tolerance) ||
           !nearly_equal(max_time_diff, 0.0, tolerance) ||
           !nearly_equal(reference_mass_diff, 0.0, tolerance) ||
           !nearly_equal(reference_energy_diff, 0.0, tolerance))) {
         metadata_ok = false;
      }

      for (int lev = 0; lev <= finest; ++lev) {
         if (!fields_only && header_a.level_steps[lev] != header_b.level_steps[lev]) {
            metadata_ok = false;
            amrex::Print() << "level " << lev << " step differs: "
                           << header_a.level_steps[lev] << " vs " << header_b.level_steps[lev] << "\n";
         }
         if (!(header_a.box_arrays[lev] == header_b.box_arrays[lev])) {
            metadata_ok = false;
            amrex::Print() << "level " << lev << " BoxArray differs\n";
         }
      }

      FieldDiff total;
      for (int lev = 0; lev <= finest; ++lev) {
         const FieldDiff state = compare_multifab(
            checkpoint_a,
            checkpoint_b,
            lev,
            pyfluid_amrex::checkpoint_state_field_name(),
            header_a.box_arrays[lev],
            header_a.state_components
         );
         total.max_abs = std::max(total.max_abs, state.max_abs);
         total.l1 += state.l1;
         amrex::Print() << "level " << lev << " state max_abs=" << state.max_abs
                        << " l1=" << state.l1 << "\n";

         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const std::string field = pyfluid_amrex::checkpoint_face_b_field_name(dir);
            const bool exists_a = pyfluid_amrex::checkpoint_field_exists(checkpoint_a, field);
            const bool exists_b = pyfluid_amrex::checkpoint_field_exists(checkpoint_b, field);
            if (exists_a != exists_b) {
               metadata_ok = false;
               amrex::Print() << "level " << lev << " " << field
                              << " exists in only one checkpoint\n";
               continue;
            }
            if (!exists_a) {
               continue;
            }

            const amrex::BoxArray face_boxes = amrex::convert(
               header_a.box_arrays[lev],
               amrex::IntVect::TheDimensionVector(dir)
            );
            const FieldDiff face = compare_multifab(
               checkpoint_a,
               checkpoint_b,
               lev,
               field,
               face_boxes,
               1
            );
            total.max_abs = std::max(total.max_abs, face.max_abs);
            total.l1 += face.l1;
            amrex::Print() << "level " << lev << " " << field
                           << " max_abs=" << face.max_abs << " l1=" << face.l1 << "\n";
         }
      }

      amrex::Print() << "checkpoint max_dt_diff=" << max_dt_diff
                     << " max_time_diff=" << max_time_diff
                     << " reference_mass_diff=" << reference_mass_diff
                     << " reference_energy_diff=" << reference_energy_diff
                     << " max_field_abs=" << total.max_abs
                     << " total_l1=" << total.l1
                     << " tolerance=" << tolerance
                     << " fields_only=" << (fields_only ? "true" : "false") << "\n";

      if (!metadata_ok || total.max_abs > tolerance) {
         return_code = 2;
      }
   }

   amrex::Finalize();
   return return_code;
}
