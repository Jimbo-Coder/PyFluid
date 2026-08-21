# Bundled CompOSE cold equations of state

These directories contain small one-parameter, cold beta-equilibrium tables
distributed by CompOSE. PyFluid reads `eos.nb` and `eos.thermo` directly,
preserves the tabulated total energy density, and uses a table-specific baryon
mass normalization that matches the first table point to a vacuum polytrope
with `h -> 1`. Above the retained causal prefix it uses a documented
constant-sound-speed continuation. Evolution adds thermal pressure through the
configured `pyfluid.thermal_gamma`; this is not finite-temperature CompOSE
support.

Each `table.toml` is the required machine-readable provenance and physics
contract. The runtime identity is a path-independent fingerprint of the
normalized retained table and continuation parameters. `eos.mr` and the
provider README are retained for independent stellar validation.

The datasets are redistributed under Creative Commons Attribution 4.0. Cite
CompOSE, the Zenodo dataset DOI, and the model papers listed in the provider
README when using a table. See <https://creativecommons.org/licenses/by/4.0/>.

