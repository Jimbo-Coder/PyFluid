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
| `entropy` | $K$ | `entropy` | `Cell` | `OutputOnlyAuxiliary` | `None` |
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
