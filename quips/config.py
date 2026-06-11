"""YAML configuration loading for detector/simulation parameters and isotopes."""

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import constants


def _deep_merge(base, override):
    """Recursively merge dict `override` into dict `base` (returns a new dict)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass
class EmissionLine:
    """A secondary particle emission line (Auger e-, x ray, gamma, conversion e-)."""

    energy_kev: float
    intensity: float  # mean number emitted per decay (may exceed 1)


@dataclass
class DecayBranch:
    fraction: float  # fraction of decays through this branch
    level_kev: float = 0.0  # daughter excitation energy (reduces neutrino energy)
    gammas: list = field(default_factory=list)  # de-excitation gammas
    conversion_electrons: list = field(default_factory=list)


@dataclass
class Isotope:
    name: str
    decay: str  # 'EC' or 'beta-'
    Z: int  # atomic number of the parent
    A: int  # mass number
    q_kev: float  # total decay energy (Q_EC or beta endpoint to ground state)
    half_life_days: float
    branches: list
    augers: list = field(default_factory=list)  # atomic relaxation, all branches
    xrays: list = field(default_factory=list)

    @property
    def max_neutrino_energy_kev(self):
        return max(self.q_kev - b.level_kev for b in self.branches)


def _lines(entries):
    return [EmissionLine(e["energy_keV"], e["intensity"]) for e in (entries or [])]


def load_isotopes(path):
    """Load all isotopes from the isotope YAML file. Returns {name: Isotope}."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    isotopes = {}
    for name, d in raw.items():
        branches = [
            DecayBranch(
                fraction=b["fraction"],
                level_kev=b.get("level_keV", 0.0),
                gammas=_lines(b.get("gammas")),
                conversion_electrons=_lines(b.get("conversion_electrons")),
            )
            for b in d["branches"]
        ]
        total = sum(b.fraction for b in branches)
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"{name}: branch fractions sum to {total}, expected 1")
        decay = d["decay"]
        if decay not in ("EC", "beta-"):
            raise ValueError(f"{name}: unknown decay type {decay!r}")
        isotopes[name] = Isotope(
            name=name,
            decay=decay,
            Z=d["Z"],
            A=d["A"],
            q_kev=d["Q_keV"],
            half_life_days=d["half_life_days"],
            branches=branches,
            augers=_lines(d.get("augers")),
            xrays=_lines(d.get("xrays")),
        )
    return isotopes


@dataclass
class DetectorConfig:
    """Detector + simulation configuration, after isotope overrides are applied."""

    raw: dict
    base_dir: Path  # directory used to resolve relative paths in the config

    # --- sphere / trap -------------------------------------------------
    @property
    def sphere_diameter_nm(self):
        return float(self.raw["sphere"]["diameter_nm"])

    @property
    def material(self):
        return self.raw["sphere"]["material"]

    @property
    def loading_fraction(self):
        return float(self.raw["sphere"]["loading_fraction"])

    @property
    def density_g_cm3(self):
        return float(self.raw["materials"][self.material]["density_g_cm3"])

    @property
    def stopping_table_path(self):
        return self.base_dir / self.raw["materials"][self.material]["stopping_table"]

    @property
    def trap_frequency_hz(self):
        return float(self.raw["trap"]["frequency_hz"])

    @property
    def sphere_mass_g(self):
        return constants.sphere_mass_grams(self.sphere_diameter_nm, self.density_g_cm3)

    @property
    def sql_momentum_kev(self):
        return constants.sql_momentum_kev(self.sphere_mass_g, self.trap_frequency_hz)

    @property
    def momentum_noise_kev(self):
        """Per-axis Gaussian momentum noise [keV/c]: SQL degraded by collection
        efficiency, sigma_i = dp_SQL / sqrt(eta_i)."""
        eta = np.asarray(self.raw["readout"]["collection_efficiency"], dtype=float)
        return self.sql_momentum_kev / np.sqrt(eta)

    # --- secondary detection -------------------------------------------
    @property
    def trigger_efficiency(self):
        return float(self.raw["secondaries"]["trigger_efficiency"])

    @property
    def angular_resolution_rad(self):
        return float(self.raw["secondaries"]["angular_resolution_rad"])

    @property
    def energy_resolution_coeff(self):
        return float(self.raw["secondaries"]["energy_resolution_coeff"])

    @property
    def min_secondary_energy_kev(self):
        return float(self.raw["secondaries"].get("min_energy_keV", 0.0))

    # --- exposure -------------------------------------------------------
    @property
    def n_spheres(self):
        return float(self.raw["exposure"]["n_spheres"])

    @property
    def live_time_days(self):
        return float(self.raw["exposure"]["live_time_days"])

    # --- simulation / statistics ----------------------------------------
    @property
    def n_mc_events(self):
        return int(self.raw["simulation"]["n_mc_events"])

    @property
    def random_seed(self):
        return int(self.raw["simulation"]["random_seed"])

    @property
    def m4_grid(self):
        return self.raw["simulation"]["m4_grid"]

    @property
    def binning(self):
        return self.raw["binning"]

    @property
    def statistics(self):
        return self.raw["statistics"]

    @property
    def plotting(self):
        return self.raw.get("plotting", {})

    # --- derived experiment-level quantities -----------------------------
    def n_isotope_atoms(self, isotope):
        """Number of isotope atoms loaded in one sphere."""
        return self.loading_fraction * self.sphere_mass_g / (isotope.A * constants.AMU_GRAMS)

    def total_decays(self, isotope):
        """Expected number of decays in the full exposure.

        Spheres are replaced once per half-life, so the decay rate stays at
        approximately its initial value throughout the live time (paper Sec. III C).
        """
        rate_per_day = self.n_isotope_atoms(isotope) * np.log(2.0) / isotope.half_life_days
        return rate_per_day * self.live_time_days * self.n_spheres


def load_detector_config(path, isotope_name=None):
    """Load the detector YAML, applying any per-isotope overrides."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    overrides = raw.get("isotope_overrides") or {}
    if isotope_name is not None and isotope_name in overrides:
        raw = _deep_merge(raw, overrides[isotope_name])
    return DetectorConfig(raw=raw, base_dir=path.resolve().parent.parent)
