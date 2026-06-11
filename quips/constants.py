"""Physical constants and unit conversions.

Internal unit system: energies and momenta in keV (c = 1), lengths in nm,
times in days, masses of macroscopic objects in grams.
"""

import numpy as np

# Particle / atomic constants
M_ELECTRON_KEV = 510.99895  # electron rest mass [keV]
AMU_KEV = 931494.10242  # atomic mass unit [keV]
AMU_GRAMS = 1.66053907e-24  # atomic mass unit [g]
ALPHA_FS = 1.0 / 137.035999  # fine-structure constant

# SI helpers
HBAR_SI = 1.054571817e-34  # [J s]
KEV_PER_C_SI = 5.344286e-25  # 1 keV/c in [kg m/s]
GRAMS_PER_FG = 1e-15

SECONDS_PER_DAY = 86400.0


def sphere_mass_grams(diameter_nm, density_g_cm3):
    """Mass of a solid sphere [g] given diameter [nm] and density [g/cm^3]."""
    radius_cm = 0.5 * diameter_nm * 1e-7
    volume_cm3 = (4.0 / 3.0) * np.pi * radius_cm**3
    return density_g_cm3 * volume_cm3


def sql_momentum_kev(sphere_mass_g, trap_frequency_hz):
    """Standard-quantum-limit momentum resolution sqrt(hbar*m*omega) [keV/c].

    Evaluates to ~15 keV/c for a 1 fg sphere in a 100 kHz trap (paper Eq. 5).
    """
    omega = 2.0 * np.pi * trap_frequency_hz
    dp_si = np.sqrt(HBAR_SI * sphere_mass_g * 1e-3 * omega)  # [kg m/s]
    return dp_si / KEV_PER_C_SI


def electron_momentum_kev(kinetic_kev):
    """Relativistic momentum [keV/c] of an electron with kinetic energy [keV]."""
    return np.sqrt(kinetic_kev * (kinetic_kev + 2.0 * M_ELECTRON_KEV))
