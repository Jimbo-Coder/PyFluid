// Included inside pyfluid_amrex::AmrSolverBase private section.
// AMReX owns the callback shape; amr_solver_metrics.cpp owns the implementation.

   void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;
