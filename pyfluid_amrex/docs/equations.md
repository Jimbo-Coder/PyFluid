# PyFluid Equation and Field Catalog

This file is generated from the executable symbolic catalog. Do not edit it directly.

## Evolution Systems

### Newtonian Hydrodynamics

Conservative ideal-fluid equations with an equation-of-state closure.

**hd.mass — Mass**

$$\partial_t \rho + \nabla\!\cdot\!\left(\begin{pmatrix}m_{1} \\ m_{2} \\ m_{3}\end{pmatrix}\right) = 0.$$

**hd.momentum — Momentum**

$$\partial_t m_i + \nabla\!\cdot\!\left(\begin{pmatrix}P + \frac{m_{1}^{2}}{\rho} & \frac{m_{1} m_{2}}{\rho} & \frac{m_{1} m_{3}}{\rho} \\ \frac{m_{1} m_{2}}{\rho} & P + \frac{m_{2}^{2}}{\rho} & \frac{m_{2} m_{3}}{\rho} \\ \frac{m_{1} m_{3}}{\rho} & \frac{m_{2} m_{3}}{\rho} & P + \frac{m_{3}^{2}}{\rho}\end{pmatrix}\right) = \begin{pmatrix}0 \\ 0 \\ 0\end{pmatrix}.$$

**hd.energy — Energy**

$$\partial_t E + \nabla\!\cdot\!\left(\begin{pmatrix}\frac{m_{1} \left(E + P\right)}{\rho} \\ \frac{m_{2} \left(E + P\right)}{\rho} \\ \frac{m_{3} \left(E + P\right)}{\rho}\end{pmatrix}\right) = 0.$$

### Newtonian Ideal Magnetohydrodynamics

Conservative ideal-MHD equations with cell-centered magnetic state and optional constrained transport.

**mhd.mass — Mass**

$$\partial_t \rho + \nabla\!\cdot\!\left(\begin{pmatrix}m_{1} \\ m_{2} \\ m_{3}\end{pmatrix}\right) = 0.$$

**mhd.momentum — Momentum**

$$\partial_t m_i + \nabla\!\cdot\!\left(\begin{pmatrix}- \frac{B_{1}^{2}}{2} + \frac{B_{2}^{2}}{2} + \frac{B_{3}^{2}}{2} + P + \frac{m_{1}^{2}}{\rho} & - B_{1} B_{2} + \frac{m_{1} m_{2}}{\rho} & - B_{1} B_{3} + \frac{m_{1} m_{3}}{\rho} \\ - B_{1} B_{2} + \frac{m_{1} m_{2}}{\rho} & \frac{B_{1}^{2}}{2} - \frac{B_{2}^{2}}{2} + \frac{B_{3}^{2}}{2} + P + \frac{m_{2}^{2}}{\rho} & - B_{2} B_{3} + \frac{m_{2} m_{3}}{\rho} \\ - B_{1} B_{3} + \frac{m_{1} m_{3}}{\rho} & - B_{2} B_{3} + \frac{m_{2} m_{3}}{\rho} & \frac{B_{1}^{2}}{2} + \frac{B_{2}^{2}}{2} - \frac{B_{3}^{2}}{2} + P + \frac{m_{3}^{2}}{\rho}\end{pmatrix}\right) = \begin{pmatrix}0 \\ 0 \\ 0\end{pmatrix}.$$

**mhd.energy — Energy**

$$\partial_t E + \nabla\!\cdot\!\left(\begin{pmatrix}- B_{1} \left(\frac{B_{1} m_{1}}{\rho} + \frac{B_{2} m_{2}}{\rho} + \frac{B_{3} m_{3}}{\rho}\right) + \frac{m_{1} \left(\frac{B_{1}^{2}}{2} + \frac{B_{2}^{2}}{2} + \frac{B_{3}^{2}}{2} + E + P\right)}{\rho} \\ - B_{2} \left(\frac{B_{1} m_{1}}{\rho} + \frac{B_{2} m_{2}}{\rho} + \frac{B_{3} m_{3}}{\rho}\right) + \frac{m_{2} \left(\frac{B_{1}^{2}}{2} + \frac{B_{2}^{2}}{2} + \frac{B_{3}^{2}}{2} + E + P\right)}{\rho} \\ - B_{3} \left(\frac{B_{1} m_{1}}{\rho} + \frac{B_{2} m_{2}}{\rho} + \frac{B_{3} m_{3}}{\rho}\right) + \frac{m_{3} \left(\frac{B_{1}^{2}}{2} + \frac{B_{2}^{2}}{2} + \frac{B_{3}^{2}}{2} + E + P\right)}{\rho}\end{pmatrix}\right) = 0.$$

**mhd.induction — Induction**

$$\partial_t B^i + \nabla\!\cdot\!\left(\begin{pmatrix}0 & \frac{B_{1} m_{2}}{\rho} - \frac{B_{2} m_{1}}{\rho} & \frac{B_{1} m_{3}}{\rho} - \frac{B_{3} m_{1}}{\rho} \\ - \frac{B_{1} m_{2}}{\rho} + \frac{B_{2} m_{1}}{\rho} & 0 & \frac{B_{2} m_{3}}{\rho} - \frac{B_{3} m_{2}}{\rho} \\ - \frac{B_{1} m_{3}}{\rho} + \frac{B_{3} m_{1}}{\rho} & - \frac{B_{2} m_{3}}{\rho} + \frac{B_{3} m_{2}}{\rho} & 0\end{pmatrix}\right) = \begin{pmatrix}0 \\ 0 \\ 0\end{pmatrix}.$$

## Equations of State

### Ideal Gas

`eos.ideal_gas` (energy closing)

$$P = internal_{energy density} \left(\gamma - 1\right).$$

$$u(P,\rho) = \frac{P}{\gamma - 1}.$$

$$c_s^2 = \frac{P \gamma}{\rho}.$$

### Polytrope

`eos.polytrope` (barotropic)

$$P = polytropic_{k} \rho^{polytropic_{\gamma}}.$$

$$u(P,\rho) = \frac{cold_{specific energy offset} \rho \left(polytropic_{\gamma} - 1\right) + polytropic_{k} \rho^{polytropic_{\gamma}}}{polytropic_{\gamma} - 1}.$$

$$c_s^2 = polytropic_{\gamma} polytropic_{k} \rho^{polytropic_{\gamma} - 1}.$$

### Hybrid Polytrope

`eos.hybrid_polytrope` (energy closing)

$$P = \frac{polytropic_{k} \rho^{polytropic_{\gamma}} \left(polytropic_{\gamma} - 1\right) - \left(thermal_{\gamma} - 1\right) \left(polytropic_{k} \rho^{polytropic_{\gamma}} + \left(polytropic_{\gamma} - 1\right) \left(cold_{specific energy offset} \rho - internal_{energy density}\right)\right)}{polytropic_{\gamma} - 1}.$$

$$u(P,\rho) = \frac{P polytropic_{\gamma} - P + cold_{specific energy offset} polytropic_{\gamma} \rho thermal_{\gamma} - cold_{specific energy offset} polytropic_{\gamma} \rho - cold_{specific energy offset} \rho thermal_{\gamma} + cold_{specific energy offset} \rho - polytropic_{\gamma} polytropic_{k} \rho^{polytropic_{\gamma}} + polytropic_{k} \rho^{polytropic_{\gamma}} thermal_{\gamma}}{polytropic_{\gamma} thermal_{\gamma} - polytropic_{\gamma} - thermal_{\gamma} + 1}.$$

$$c_s^2 = \frac{P thermal_{\gamma} + polytropic_{\gamma} polytropic_{k} \rho^{polytropic_{\gamma}} - polytropic_{k} \rho^{polytropic_{\gamma}} thermal_{\gamma}}{\rho}.$$

## Diagnostic Integrands

### Total rest mass

`rest-mass`

$$M_0 = \int_\Omega D\,d^3x.$$

### ADM mass surface integral

`adm-mass-surface`

$$M_{\rm ADM}^{S}=\frac{1}{16\pi}\oint_{\partial\Omega}\sqrt{\gamma}\,\gamma^{ij}\gamma^{kl}(\partial_j\gamma_{ik}-\partial_k\gamma_{ij})\,dS_l.$$

### Constraint-assisted ADM mass volume integral

`adm-mass-volume`

$$M_{\rm ADM}^{V}=\int_\Omega\sqrt{\gamma}\left[\rho+\frac{1}{16\pi}\left(K_{ij}K^{ij}-K^2-\mathcal G\right)\right]d^3x,\qquad\mathcal G=\gamma^{ij}\left(\Gamma^k{}_{i\ell}\Gamma^\ell{}_{jk}-\Gamma^k{}_{ij}\Gamma^\ell{}_{k\ell}\right).$$

### ADM linear momentum surface integral

`adm-linear-momentum-surface`

$$P_i^{S}=\frac{1}{8\pi}\oint_{\partial\Omega}\sqrt{\gamma}(K^l{}_i-\delta^l{}_iK)\,dS_l.$$

### Constraint-assisted ADM linear momentum volume integral

`adm-linear-momentum-volume`

$$P_a^{V}=\int_\Omega\sqrt{\gamma}\left[S_i\xi_a^i+\frac{1}{16\pi}(K^{ij}-\gamma^{ij}K)\mathcal L_{\xi_a}\gamma_{ij}\right]d^3x,\qquad \xi_a^i=\delta_a^i.$$

### ADM angular momentum surface integral

`adm-angular-momentum-surface`

$$J_i^{S}=\frac{1}{8\pi}\oint_{\partial\Omega}\sqrt{\gamma}(K^l{}_k-\delta^l{}_kK)\phi_i^k\,dS_l,\qquad\phi_i^k=\epsilon_{ijk}x^j.$$

### Constraint-assisted ADM angular momentum volume integral

`adm-angular-momentum-volume`

$$J_a^{V}=\int_\Omega\sqrt{\gamma}\left[S_i\phi_a^i+\frac{1}{16\pi}(K^{ij}-\gamma^{ij}K)\mathcal L_{\phi_a}\gamma_{ij}\right]d^3x,\qquad \phi_a^i=\epsilon_{aji}x^j.$$

### Proper matter-energy volume integral

`proper-mass-volume`

$$M_{\rm proper}^{V}=\int_\Omega\sqrt{\gamma}\,E\,d^3x.$$

### Komar mass surface integral

`komar-mass-surface`

$$M_{\rm K}^{S}=\frac{1}{4\pi}\oint_{\partial\Omega}\left(D^i\alpha-K^i{}_j\beta^j\right)dS_i.$$

### Stationary Komar mass volume integral

`komar-mass-volume`

$$M_{\rm K}^{V}=\int_\Omega\sqrt{\gamma}\left[\alpha(E+S)-2\beta^iS_i\right]d^3x,\qquadS=\gamma^{ij}S_{ij}.$$

### Hamiltonian constraint norms

`hamiltonian-norms`

$$\|\mathcal H\|_2=\sqrt{V^{-1}\int_\Omega\mathcal H^2dV},\qquad\|\mathcal H\|_\infty=\max_\Omega|\mathcal H|.$$

### Relativistic magnetofluid invariants

`rmhd-runtime-invariants`

$$b^2=b^\mu b_\mu,\qquad \sigma_h=\frac{b^2}{\rho h},\qquad\beta_{\rm plasma}=\frac{2P}{b^2},\qquad W=(1-v_iv^i)^{-1/2}.$$

### Eulerian electromagnetic energy

`rmhd-electromagnetic-energy`

$$E_{\rm EM}=\int_\Omega\sqrt{\gamma}\,n_\mu n_\nu\left(T_{\rm RMHD}^{\mu\nu}-T_{\rm RHD}^{\mu\nu}\right)d^3x.$$

### Electric and magnetic Weyl radiation

`weyl-radiation`

$$E_{ij}=\left[R_{ij}+KK_{ij}-K_{ik}K^k{}_j-4\pi\left(S_{ij}+\gamma_{ij}(E-S)\right)\right]^{\rm TF},\qquadB_{ij}=\left[\epsilon_i{}^{kl}D_kK_{lj}\right]^{\rm STF},\qquad\Psi_4=(E_{ij}-iB_{ij})\bar m^i\bar m^j.$$

### Finite-radius matter energy flux

`matter-energy-flux`

$$\mathcal F_{\rm matter}=\oint \alpha\sqrt{\gamma}\,J^i_{(t)}dS_i,\qquadJ^i_{(t)}=-E\beta^i+\alpha S^i+\frac{\beta^i}{\alpha}S_j\beta^j-S^i{}_j\beta^j.$$

## Runtime Diagnostic Groups

| Group | Diagnostics |
|---|---|
| `standard` | `step`, `time`, `dt`, `mass`, `mom1`, `mom2`, `mom3`, `angular_mom1`, `angular_mom2`, `angular_mom3`, `energy`, `gravity_potential_energy`, `gravity_total_energy`, `gravity_total_energy_rel_drift`, `passive_mass`, `passive_mass_rel_drift`, `wall_work`, `wall_power`, `energy_minus_wall_work`, `wall_work_rel_error`, `external_budget_valid`, `boundary_mass_change`, `boundary_energy_change`, `source_mass_change`, `source_energy_change`, `external_mass_change`, `external_energy_change`, `external_mass_residual`, `external_energy_residual`, `external_mass_rel_error`, `external_energy_rel_error`, `mass_rel_drift`, `energy_rel_drift`, `min_rho`, `max_rho`, `max_rho_rel_change`, `min_pressure`, `min_lapse`, `newtonian_magnetic_energy`, `rmhd_electromagnetic_energy`, `rmhd_max_b_squared`, `rmhd_max_enthalpy_magnetization`, `rmhd_min_plasma_beta`, `max_lorentz_factor`, `relativistic_atmosphere_volume_fraction`, `divb_l1`, `divb_l2`, `divb_linf`, `diffusion_residual`, `diffusion_iterations`, `amr_total_cells`, `amr_active_finest_level`, `amr_finest_headroom`, `amr_touches_max_level`, `amr_total_budget_fraction`, `amr_peak_level_budget_fraction`, `mcells_per_s`, `layout_mcells_per_s`, `sim_time_per_s`, `cfl_limiter_frequency`, `cfl_limiter_level`, `cfl_limiter_i`, `cfl_limiter_j`, `cfl_limiter_k`, `cfl_limiter_q1`, `cfl_limiter_q2`, `cfl_limiter_q3`, `z4c_hamiltonian_linf`, `z4c_hamiltonian_l2`, `z4c_hamiltonian_interior_linf`, `z4c_hamiltonian_interior_l2`, `z4c_hamiltonian_cf_linf`, `z4c_hamiltonian_cf_l2`, `z4c_hamiltonian_bh_exterior_linf`, `z4c_hamiltonian_bh_exterior_l2`, `z4c_momentum_bh_exterior_linf`, `z4c_momentum_linf`, `z4c_z_linf`, `z4c_theta_linf`, `z4c_log_conformal_determinant_linf`, `z4c_trace_a_tilde_linf`, `black_hole_horizon_radius` |
| `coordinates` | `step`, `time`, `dt` |
| `conservation` | `mass`, `mom1`, `mom2`, `mom3`, `angular_mom1`, `angular_mom2`, `angular_mom3`, `energy`, `mass_rel_drift`, `energy_rel_drift` |
| `gravity` | `gravity_potential_energy`, `gravity_total_energy`, `gravity_total_energy_rel_drift` |
| `passive` | `passive_mass`, `passive_mass_rel_drift` |
| `ledger` | `wall_work`, `wall_power`, `energy_minus_wall_work`, `wall_work_rel_error`, `external_budget_valid`, `boundary_mass_change`, `boundary_energy_change`, `source_mass_change`, `source_energy_change`, `external_mass_change`, `external_energy_change`, `external_mass_residual`, `external_energy_residual`, `external_mass_rel_error`, `external_energy_rel_error` |
| `state_bounds` | `min_rho`, `max_rho`, `max_rho_rel_change`, `min_pressure`, `min_lapse` |
| `mhd` | `newtonian_magnetic_energy`, `divb_l1`, `divb_l2`, `divb_linf` |
| `rmhd` | `rmhd_electromagnetic_energy`, `rmhd_max_b_squared`, `rmhd_max_enthalpy_magnetization`, `rmhd_min_plasma_beta` |
| `relativistic` | `max_lorentz_factor`, `relativistic_atmosphere_volume_fraction` |
| `diffusion` | `diffusion_residual`, `diffusion_iterations` |
| `amr` | `amr_total_cells`, `amr_active_finest_level`, `amr_finest_headroom`, `amr_touches_max_level`, `amr_total_budget_fraction`, `amr_peak_level_budget_fraction` |
| `performance` | `mcells_per_s`, `layout_mcells_per_s`, `sim_time_per_s` |
| `cfl` | `cfl_limiter_frequency`, `cfl_limiter_level`, `cfl_limiter_i`, `cfl_limiter_j`, `cfl_limiter_k`, `cfl_limiter_q1`, `cfl_limiter_q2`, `cfl_limiter_q3` |
| `constraints` | `z4c_hamiltonian_linf`, `z4c_hamiltonian_l2`, `z4c_hamiltonian_interior_linf`, `z4c_hamiltonian_interior_l2`, `z4c_hamiltonian_cf_linf`, `z4c_hamiltonian_cf_l2`, `z4c_hamiltonian_bh_exterior_linf`, `z4c_hamiltonian_bh_exterior_l2`, `z4c_momentum_bh_exterior_linf`, `z4c_momentum_linf`, `z4c_z_linf`, `z4c_theta_linf`, `z4c_log_conformal_determinant_linf`, `z4c_trace_a_tilde_linf` |
| `black_hole` | `z4c_hamiltonian_bh_exterior_linf`, `z4c_hamiltonian_bh_exterior_l2`, `z4c_momentum_bh_exterior_linf`, `black_hole_horizon_radius` |
| `global_gr` | `adm_mass_surface`, `adm_mass_volume`, `proper_mass_volume`, `komar_mass_surface`, `komar_mass_volume`, `binding_energy_volume`, `adm_momentum1_surface`, `adm_momentum2_surface`, `adm_momentum3_surface`, `adm_momentum1_volume`, `adm_momentum2_volume`, `adm_momentum3_volume`, `adm_angular_momentum1_surface`, `adm_angular_momentum2_surface`, `adm_angular_momentum3_surface`, `adm_angular_momentum1_volume`, `adm_angular_momentum2_volume`, `adm_angular_momentum3_volume` |
| `radiation` | `extraction_radius`, `matter_energy_flux`, `psi4_l2` |

## Grid Fields

| Field | Label | Components | Centering | Role | Sector |
|---|---|---|---|---|---|
| `rho` | $\rho$ | `rho` | `Cell` | `EvolvedState` | `Fluid` |
| `momentum` | $m_i$ | `mom1, mom2, mom3` | `Cell, Cell, Cell` | `EvolvedState` | `Fluid` |
| `energy` | $E$ | `energy` | `Cell` | `EvolvedState` | `Fluid` |
| `dens` | $D$ | `dens` | `Cell` | `EvolvedState` | `Fluid` |
| `tau` | $\tau$ | `tau` | `Cell` | `EvolvedState` | `Fluid` |
| `magnetic_field` | $B^i$ | `b1, b2, b3` | `Cell, Cell, Cell` | `EvolvedState` | `Fluid` |
| `ct_magnetic_field` | $B^i$ | `face_b1, face_b2, face_b3` | `Face1, Face2, Face3` | `EvolutionAuxiliary` | `Fluid` |
| `passive_scalar` | $\rho C$ | `dye_mass` | `Cell` | `EvolvedState` | `Fluid` |
| `rest_mass_density` | $\rho$ | `rho` | `Cell` | `InitialDataState` | `Fluid` |
| `velocity` | $v^i$ | `v1, v2, v3` | `Cell, Cell, Cell` | `InitialDataState` | `Fluid` |
| `pressure` | $p$ | `pressure` | `Cell` | `InitialDataState` | `Fluid` |
| `physical_magnetic_field` | $B^i$ | `b1, b2, b3` | `Cell, Cell, Cell` | `InitialDataState` | `Fluid` |
| `passive_scalar_fraction` | $C$ | `dye_fraction` | `Cell` | `InitialDataState` | `Fluid` |
| `spatial_metric` | $\gamma_{ij}$ | `gamma11, gamma12, gamma13, gamma22, gamma23, gamma33` | `Vertex, Vertex, Vertex, Vertex, Vertex, Vertex` | `InitialDataState` | `Spacetime` |
| `extrinsic_curvature` | $K_{ij}$ | `k11, k12, k13, k22, k23, k33` | `Vertex, Vertex, Vertex, Vertex, Vertex, Vertex` | `InitialDataState` | `Spacetime` |
| `adm_lapse` | $\alpha$ | `lapse` | `Vertex` | `InitialDataState` | `Spacetime` |
| `adm_shift` | $\beta^i$ | `shift1, shift2, shift3` | `Vertex, Vertex, Vertex` | `InitialDataState` | `Spacetime` |
| `chi` | $\chi$ | `chi` | `Vertex` | `EvolvedState` | `Spacetime` |
| `gamma_tilde` | $\tilde{\gamma}_{ij}$ | `gamma_tilde11, gamma_tilde12, gamma_tilde13, gamma_tilde22, gamma_tilde23, gamma_tilde33` | `Vertex, Vertex, Vertex, Vertex, Vertex, Vertex` | `EvolvedState` | `Spacetime` |
| `k_hat` | $\hat{K}$ | `k_hat` | `Vertex` | `EvolvedState` | `Spacetime` |
| `a_tilde` | $\tilde{A}_{ij}$ | `a_tilde11, a_tilde12, a_tilde13, a_tilde22, a_tilde23, a_tilde33` | `Vertex, Vertex, Vertex, Vertex, Vertex, Vertex` | `EvolvedState` | `Spacetime` |
| `gamma_tilde_vector` | $\tilde{\Gamma}^{i}$ | `gamma_tilde1, gamma_tilde2, gamma_tilde3` | `Vertex, Vertex, Vertex` | `EvolvedState` | `Spacetime` |
| `theta` | $\Theta$ | `theta` | `Vertex` | `EvolvedState` | `Spacetime` |
| `lapse` | $\alpha$ | `lapse` | `Vertex` | `EvolvedState` | `Spacetime` |
| `shift` | $\beta^{i}$ | `shift1, shift2, shift3` | `Vertex, Vertex, Vertex` | `EvolvedState` | `Spacetime` |
| `j` | $J^i$ | `j1, j2, j3` | `Edge1, Edge2, Edge3` | `CachedDerivative` | `None` |
| `e` | $E^i$ | `e1, e2, e3` | `Edge1, Edge2, Edge3` | `CachedDerivative` | `None` |
| `divb` | $\nabla_i B^i$ | `divb` | `Cell` | `OutputOnlyAuxiliary` | `None` |
| `magnetic_energy` | $B^2/2$ | `magnetic_energy` | `Cell` | `OutputOnlyAuxiliary` | `None` |
| `vorticity` | $\omega^i$ | `vorticity1, vorticity2, vorticity3` | `Cell, Cell, Cell` | `OutputOnlyAuxiliary` | `None` |
| `mach` | $\mathcal{M}$ | `mach` | `Cell` | `OutputOnlyAuxiliary` | `None` |
| `plasma_beta` | $\beta_{\rm plasma}$ | `plasma_beta` | `Cell` | `OutputOnlyAuxiliary` | `None` |
| `thermal_entropy_proxy` | $K_{\rm th}$ | `thermal_entropy_proxy` | `Cell` | `OutputOnlyAuxiliary` | `None` |
| `amr_tag_metric` | $\eta_{\rm AMR}$ | `amr_tag_metric` | `Cell` | `OutputOnlyAuxiliary` | `None` |
| `solid_mask` | $M_{\rm solid}$ | `solid_mask` | `Cell` | `EvolutionAuxiliary` | `None` |
| `solid_volume_fraction` | $\kappa_{\rm fluid}$ | `solid_volume_fraction` | `Cell` | `EvolutionAuxiliary` | `None` |
| `solid_aperture1` | $\alpha_{1}$ | `solid_aperture1` | `Face1` | `EvolutionAuxiliary` | `None` |
| `solid_aperture2` | $\alpha_{2}$ | `solid_aperture2` | `Face2` | `EvolutionAuxiliary` | `None` |
| `solid_aperture3` | $\alpha_{3}$ | `solid_aperture3` | `Face3` | `EvolutionAuxiliary` | `None` |
| `solid_swept_volume` | $\Delta V_{\rm solid}$ | `solid_swept_volume` | `Cell` | `OutputOnlyAuxiliary` | `None` |
| `solid_phi` | $\phi_{\rm solid}$ | `solid_phi` | `Cell` | `OutputOnlyAuxiliary` | `None` |
| `phi` | $\Phi$ | `phi` | `Cell` | `EvolutionAuxiliary` | `None` |
| `grad_phi` | $\nabla_i\Phi$ | `grad_phi1, grad_phi2, grad_phi3` | `Cell, Cell, Cell` | `CachedDerivative` | `None` |
