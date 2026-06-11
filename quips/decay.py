"""Weak-decay kinematics: EC two-body momenta, allowed beta spectra with a
massive neutrino, spectrum sampling, and phase-space ratios for the sterile
branching fraction BR4 = |Ue4|^2 * rho(m4)/rho(0).
"""

import numpy as np

from .constants import ALPHA_FS, M_ELECTRON_KEV


def neutrino_momentum_kev(e_nu_kev, m4_kev):
    """|p_nu| for total neutrino energy e_nu and mass m4 (0 if forbidden)."""
    e = np.asarray(e_nu_kev, dtype=float)
    return np.sqrt(np.maximum(e**2 - m4_kev**2, 0.0))


def fermi_function(z_daughter, kinetic_kev):
    """Nonrelativistic Fermi function F = 2*pi*eta / (1 - exp(-2*pi*eta)) for
    beta- decay (attractive Coulomb correction), eta = Z*alpha*E/p."""
    t = np.asarray(kinetic_kev, dtype=float)
    energy = t + M_ELECTRON_KEV
    momentum = np.sqrt(np.maximum(t * (t + 2.0 * M_ELECTRON_KEV), 1e-12))
    eta = z_daughter * ALPHA_FS * energy / momentum
    x = 2.0 * np.pi * eta
    return x / (1.0 - np.exp(-x))


def beta_spectrum(te_kev, endpoint_kev, z_daughter, m4_kev=0.0):
    """Unnormalized allowed beta- spectrum dN/dTe including neutrino mass m4.

    dN/dTe ~ F(Z, Te) * p_e * E_e * E_nu * sqrt(E_nu^2 - m4^2), Te in
    [0, endpoint - m4]; zero outside.
    """
    te = np.asarray(te_kev, dtype=float)
    e_nu = endpoint_kev - te
    p_nu = neutrino_momentum_kev(e_nu, m4_kev)
    e_e = te + M_ELECTRON_KEV
    p_e = np.sqrt(np.maximum(te * (te + 2.0 * M_ELECTRON_KEV), 0.0))
    spec = fermi_function(z_daughter, te) * p_e * e_e * e_nu * p_nu
    return np.where((te > 0.0) & (e_nu > m4_kev), spec, 0.0)


def _te_grid(endpoint_kev, m4_kev, n=4000):
    return np.linspace(0.0, max(endpoint_kev - m4_kev, 1e-9), n)


def beta_spectrum_integral(endpoint_kev, z_daughter, m4_kev=0.0):
    """Integral of the allowed beta spectrum (phase-space weight)."""
    if endpoint_kev <= m4_kev:
        return 0.0
    te = _te_grid(endpoint_kev, m4_kev)
    return np.trapezoid(beta_spectrum(te, endpoint_kev, z_daughter, m4_kev), te)


def sample_beta_kinetic_energy(rng, n, endpoint_kev, z_daughter, m4_kev=0.0):
    """Inverse-CDF sampling of the beta kinetic energy [keV]."""
    te = _te_grid(endpoint_kev, m4_kev)
    pdf = beta_spectrum(te, endpoint_kev, z_daughter, m4_kev)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(te))])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, te)


def branch_phase_space_ratio(isotope, branch, m4_kev):
    """rho_b(m4)/rho_b(0) for a single decay branch."""
    e_nu = isotope.q_kev - branch.level_kev
    if e_nu <= m4_kev:
        return 0.0
    if isotope.decay == "EC":
        # two-body: rate ~ p_nu * E_nu with E_nu fixed by energy conservation
        return neutrino_momentum_kev(e_nu, m4_kev) / e_nu
    z_d = isotope.Z + 1
    return beta_spectrum_integral(e_nu, z_d, m4_kev) / beta_spectrum_integral(e_nu, z_d, 0.0)


def phase_space_ratio(isotope, m4_kev):
    """rho(m4)/rho(0): relative total decay rate to a state of mass m4 vs
    massless, weighting each branch by its (massless) branching fraction.
    Multiply by |Ue4|^2 for the sterile branching fraction."""
    return sum(
        b.fraction * branch_phase_space_ratio(isotope, b, m4_kev) for b in isotope.branches
    )
