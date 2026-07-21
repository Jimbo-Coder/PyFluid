from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from pyfluid.state import D, P, VEL_P


@dataclass(frozen=True)
class HealthCriteria:
   min_rho: float = 0.0
   min_pressure: float = 0.0
   max_mass_rel_drift: float = 1e-10
   max_energy_rel_drift: float = 1e-10
   max_momentum_drift: float = 1e-10
   max_divb: float = None
   max_primitive_amplification: float = 100.0
   no_conservation_check: bool = False

   def failure_reasons(self, summary):
      return health_failure_reasons(
         summary,
         min_rho=self.min_rho,
         min_pressure=self.min_pressure,
         max_mass_rel_drift=None if self.no_conservation_check else self.max_mass_rel_drift,
         max_energy_rel_drift=None if self.no_conservation_check else self.max_energy_rel_drift,
         max_momentum_drift=None if self.no_conservation_check else self.max_momentum_drift,
         max_divb=self.max_divb,
         max_primitive_amplification=self.max_primitive_amplification,
      )


def add_health_arguments(parser):
   defaults = HealthCriteria()
   parser.add_argument("--min-rho", type=float, default=defaults.min_rho, help=f"Required lower bound for stored density. Default: {defaults.min_rho!r}.")
   parser.add_argument("--min-pressure", type=float, default=defaults.min_pressure, help=f"Required lower bound for stored pressure. Default: {defaults.min_pressure!r}.")
   parser.add_argument("--max-mass-rel-drift", type=float, default=defaults.max_mass_rel_drift, help=f"Maximum allowed stored relative mass drift. Default: {defaults.max_mass_rel_drift!r}.")
   parser.add_argument("--max-energy-rel-drift", type=float, default=defaults.max_energy_rel_drift, help=f"Maximum allowed stored relative energy drift. Default: {defaults.max_energy_rel_drift!r}.")
   parser.add_argument("--max-momentum-drift", type=float, default=defaults.max_momentum_drift, help=f"Maximum allowed stored absolute total-momentum drift. Default: {defaults.max_momentum_drift!r}.")
   parser.add_argument("--max-divb", type=float, default=defaults.max_divb, help=f"Maximum allowed normalized divB. Default: {defaults.max_divb!r}.")
   parser.add_argument("--max-primitive-amplification", type=float, default=defaults.max_primitive_amplification, help=f"Maximum allowed max(abs(primitive))/initial max(abs(primitive)). Default: {defaults.max_primitive_amplification!r}.")
   parser.add_argument("--no-conservation-check", action="store_true", default=defaults.no_conservation_check, help=f"Disable mass, energy, and total-momentum drift checks for open-boundary cases. Default: {defaults.no_conservation_check!r}.")


def health_criteria_from_args(args):
   return HealthCriteria(
      min_rho=args.min_rho,
      min_pressure=args.min_pressure,
      max_mass_rel_drift=args.max_mass_rel_drift,
      max_energy_rel_drift=args.max_energy_rel_drift,
      max_momentum_drift=args.max_momentum_drift,
      max_divb=args.max_divb,
      max_primitive_amplification=args.max_primitive_amplification,
      no_conservation_check=args.no_conservation_check,
   )


def abs_relative_drift(series):
   series = np.asarray(series)
   denom = max(abs(float(series[0])), 1.0)
   return np.abs(series - series[0]) / denom


def momentum_drift_norm(diagnostics):
   diagnostics = np.asarray(diagnostics)
   mom = np.stack(
      (
         diagnostics[:, D.MOMX],
         diagnostics[:, D.MOMY],
         diagnostics[:, D.MOMZ],
      ),
      axis=1,
   )
   return np.linalg.norm(mom - mom[0], axis=1)


def stored_min(state, component):
   return np.min(np.asarray(state.Pvecs[:, component, ...]), axis=(1, 2))


def final_scalar_summary(state):
   health = scalar_health_summary(state)
   return {
      "time": health["time"],
      "dt_mean": health["dt_mean"],
      "run_time": health["run_time"],
      "evolution_run_time": health["evolution_run_time"],
      "cell_updates": health["cell_updates"],
      "cell_updates_per_second": health["cell_updates_per_second"],
      "million_cell_updates_per_second": health["million_cell_updates_per_second"],
      "mass": health["mass"],
      "mass_rel_drift": health["mass_rel_drift_final"],
      "energy": health["energy"],
      "energy_rel_drift": health["energy_rel_drift_final"],
      "momentum_drift": health["momentum_drift_final"],
      "min_rho": health["min_rho"],
      "min_pressure": health["min_pressure"],
      "divb": health["divb_final"],
   }


def scalar_health_summary(state):
   diagnostics = np.asarray(state.diagnostics)
   pvecs = np.asarray(state.Pvecs)
   run_time = float(getattr(state, "run_time", 0.0))
   evolution_run_time = float(getattr(state, "evolution_run_time", run_time))
   cell_updates = float(state.nx * state.ny * state.nsteps)
   if hasattr(state, "cell_updates_per_second"):
      updates_per_second = float(state.cell_updates_per_second())
   elif evolution_run_time > 0.0:
      updates_per_second = cell_updates / evolution_run_time
   else:
      updates_per_second = 0.0
   primitive_scale = max(float(np.max(np.abs(pvecs[0]))), 1.0)
   velocity_scale = max(float(np.max(np.abs(pvecs[0, VEL_P, ...]))), 1.0)
   mass_rel = abs_relative_drift(diagnostics[:, D.MASS])
   energy_rel = abs_relative_drift(diagnostics[:, D.ENERGY])
   momentum = momentum_drift_norm(diagnostics)
   return {
      "time": float(state.t),
      "dt_mean": float(state.dtavg),
      "run_time": run_time,
      "evolution_run_time": evolution_run_time,
      "cell_updates": cell_updates,
      "cell_updates_per_second": updates_per_second,
      "million_cell_updates_per_second": updates_per_second / 1.0e6,
      "mass": float(diagnostics[-1, D.MASS]),
      "mass_rel_drift_final": float(mass_rel[-1]),
      "mass_rel_drift_max": float(np.max(mass_rel)),
      "energy": float(diagnostics[-1, D.ENERGY]),
      "energy_rel_drift_final": float(energy_rel[-1]),
      "energy_rel_drift_max": float(np.max(energy_rel)),
      "momentum_drift_final": float(momentum[-1]),
      "momentum_drift_max": float(np.max(momentum)),
      "min_rho": float(np.min(pvecs[:, P.RHO, ...])),
      "min_pressure": float(np.min(pvecs[:, P.PRESS, ...])),
      "divb_final": float(diagnostics[-1, D.DIVB_L2]) if state.boolmag else 0.0,
      "divb_max": float(np.max(diagnostics[:, D.DIVB_L2])) if state.boolmag else 0.0,
      "primitive_amplification": float(np.max(np.abs(pvecs)) / primitive_scale),
      "velocity_amplification": float(np.max(np.abs(pvecs[:, VEL_P, ...])) / velocity_scale),
      "finite": bool(np.all(np.isfinite(pvecs)) and np.all(np.isfinite(np.asarray(state.Uvecs))) and np.all(np.isfinite(diagnostics))),
   }


def health_failure_reasons(
   summary,
   min_rho=0.0,
   min_pressure=0.0,
   max_mass_rel_drift=None,
   max_energy_rel_drift=None,
   max_momentum_drift=None,
   max_divb=None,
   max_primitive_amplification=None,
):
   reasons = []
   if not summary["finite"]:
      reasons.append("non-finite state or diagnostics")
   if summary["min_rho"] <= min_rho:
      reasons.append(f"min_rho={summary['min_rho']:.3e} <= {min_rho:.3e}")
   if summary["min_pressure"] <= min_pressure:
      reasons.append(f"min_pressure={summary['min_pressure']:.3e} <= {min_pressure:.3e}")
   if max_mass_rel_drift is not None and summary["mass_rel_drift_max"] > max_mass_rel_drift:
      reasons.append(f"mass_rel_drift={summary['mass_rel_drift_max']:.3e} > {max_mass_rel_drift:.3e}")
   if max_energy_rel_drift is not None and summary["energy_rel_drift_max"] > max_energy_rel_drift:
      reasons.append(f"energy_rel_drift={summary['energy_rel_drift_max']:.3e} > {max_energy_rel_drift:.3e}")
   if max_momentum_drift is not None and summary["momentum_drift_max"] > max_momentum_drift:
      reasons.append(f"momentum_drift={summary['momentum_drift_max']:.3e} > {max_momentum_drift:.3e}")
   if max_divb is not None and summary["divb_max"] > max_divb:
      reasons.append(f"divb={summary['divb_max']:.3e} > {max_divb:.3e}")
   if max_primitive_amplification is not None and summary["primitive_amplification"] > max_primitive_amplification:
      reasons.append(f"primitive_amplification={summary['primitive_amplification']:.3e} > {max_primitive_amplification:.3e}")
   return reasons


def write_scalar_summary_csv(state, outpath):
   summary = final_scalar_summary(state)
   keys = tuple(summary)
   lines = [",".join(keys), ",".join(f"{summary[key]:.16e}" for key in keys)]
   outpath.write_text("\n".join(lines) + "\n")


def plot_scalar_timeseries(state, outpath=None, show=False, title=None):
   ts = np.asarray(state.ts)
   diagnostics = np.asarray(state.diagnostics)
   fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
   axs = axs.ravel()

   axs[0].semilogy(ts, np.maximum(abs_relative_drift(diagnostics[:, D.MASS]), 1e-30), label="mass")
   axs[0].semilogy(ts, np.maximum(abs_relative_drift(diagnostics[:, D.ENERGY]), 1e-30), label="energy")
   axs[0].set_title("Conserved Scalars")
   axs[0].set_ylabel("relative drift")
   axs[0].legend()

   axs[1].semilogy(ts, np.maximum(momentum_drift_norm(diagnostics), 1e-30), label="|dP|")
   axs[1].set_title("Total Momentum Drift")
   axs[1].set_ylabel("absolute drift")

   axs[2].plot(ts, stored_min(state, P.RHO), label="min rho")
   axs[2].plot(ts, stored_min(state, P.PRESS), label="min P")
   axs[2].set_title("Primitive Floors")
   axs[2].legend()

   axs[3].plot(ts, diagnostics[:, D.INTERNAL_ENERGY], label="internal")
   axs[3].plot(ts, diagnostics[:, D.KINETIC_ENERGY], label="kinetic")
   if state.boolmag:
      axs[3].plot(ts, diagnostics[:, D.MAGNETIC_ENERGY], label="magnetic")
   axs[3].plot(ts, diagnostics[:, D.ENERGY], label="total", color="black", linewidth=1.3)
   axs[3].set_title("Energy Components")
   axs[3].legend()

   if len(state.dts):
      axs[4].plot(ts[1:], np.asarray(state.dts), label="dt")
   axs[4].set_title("Timestep")
   axs[4].set_ylabel("dt")

   if state.boolmag:
      axs[5].semilogy(ts, np.maximum(diagnostics[:, D.DIVB_L2], 1e-30), label="divB")
      axs[5].set_title("Magnetic Constraint")
      axs[5].set_ylabel("normalized divB")
   else:
      axs[5].plot(ts, diagnostics[:, D.GRADPX_L2], label="gradPx L2")
      axs[5].plot(ts, diagnostics[:, D.GRADPY_L2], label="gradPy L2")
      axs[5].set_title("Pressure Smoothness")
      axs[5].legend()

   for ax in axs:
      ax.set_xlabel("time")
      ax.grid(True, alpha=0.25)

   fig.suptitle(title or f"{state.info} scalar diagnostics")
   if outpath is not None:
      fig.savefig(outpath, dpi=160)
   if show:
      plt.show()
   else:
      plt.close(fig)
   return fig


def plot_field_snapshot(state, outpath=None, it=-1, show=False):
   fields = [(state.Pvecs, P.RHO, "rho"), (state.Pvecs, P.PRESS, "pressure")]
   if state.boolmag:
      fields.extend([(state.Pvecs, P.BX, "Bx"), (state.Pvecs, P.BY, "By")])
   else:
      fields.extend([(state.Pvecs, P.VX, "vx"), (state.Pvecs, P.VY, "vy")])

   fig, axs = plt.subplots(2, 2, figsize=(9.5, 8), constrained_layout=True)
   vertices = (state.x0, state.x1, state.y0, state.y1)
   for ax, (vecs, component, label) in zip(axs.ravel(), fields):
      data = np.asarray(vecs[it, component, ...])
      pcm = ax.imshow(data.T, origin="lower", extent=vertices, aspect="equal")
      fig.colorbar(pcm, ax=ax, shrink=0.85)
      ax.set_title(label)
      ax.set_xlabel("x")
      ax.set_ylabel("y")
   fig.suptitle(f"{state.info} fields at t={state.ts[it]:.6g}")
   if outpath is not None:
      fig.savefig(outpath, dpi=160)
   if show:
      plt.show()
   else:
      plt.close(fig)
   return fig
