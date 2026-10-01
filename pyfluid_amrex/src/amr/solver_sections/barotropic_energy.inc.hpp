// Included inside pyfluid_amrex::AmrSolver private section.
// Barotropic EOS energy reset helpers.

public:
   template <int NComp, int EOSMethod>
   void reset_barotropic_energy_level(int lev)
   {
      if constexpr (EOSMethod == EOSPolytrope || EOSMethod == EOSRuntime) {
         if (eos_method_ != EOSPolytrope) {
            return;
         }
         if (physical_system_is_relativistic(physics_system_)) {
            return;
         }
         const EOSParams eos = eos_params(cfg_);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
         {
         for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto s = fluid_state_.state()[lev]->array(mfi);
            amrex::ParallelFor(bx, [s, eos] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               if constexpr (has_mhd_components<NComp>()) {
                  MHDState cell = load_mhd_state(s, i, j, k);
                  reset_barotropic_mhd_energy<EOSMethod>(cell, eos);
                  s(i, j, k, ENERGY) = cell.energy;
               } else {
                  HydroState cell = load_hydro_state(s, i, j, k);
                  reset_barotropic_hydro_energy<EOSMethod>(cell, eos);
                  s(i, j, k, ENERGY) = cell.energy;
               }
            });
         }
         }
      }
   }

private:

   template <int NComp, int EOSMethod>
   void reset_barotropic_energy_all_levels()
   {
      if constexpr (EOSMethod == EOSPolytrope || EOSMethod == EOSRuntime) {
         if (eos_method_ != EOSPolytrope) {
            return;
         }
         for (int lev = 0; lev <= finestLevel(); ++lev) {
            reset_barotropic_energy_level<NComp, EOSMethod>(lev);
         }
      }
   }
