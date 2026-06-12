"""Binned probability templates for the reconstructed observables.

EC decays: 1D PDF of |p_nu|.  Beta decays: 2D PDF of (|p_nu|, Te,meas),
following paper Sec. III C.  Templates are normalized over detectable events;
the detectable fraction times the trigger efficiency gives the per-decay
detection efficiency used to convert yields to |Ue4|^2.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Template:
    probs: np.ndarray  # flattened bin probabilities (sums to 1)
    efficiency: float  # detected events per decay (incl. trigger efficiency)


def make_binning(detector, isotope):
    """Bin edges for |p_nu| (and Te for beta decays).

    'auto' chooses a bin width of about half the per-axis momentum noise
    (capped so the full range has at most ~300 bins).
    """
    cfg = detector.binning
    sigma_p = float(np.min(detector.momentum_noise_kev[detector.measured_axes]))
    p_max = isotope.max_neutrino_energy_kev + 8.0 * sigma_p
    max_bins = 300 if isotope.decay == "EC" else 100  # keep 2D templates tractable

    def edges(setting, lo, hi, resolution):
        if setting == "auto":
            width = max(resolution / 2.0, (hi - lo) / max_bins)
        else:
            width = float(setting)
        return np.arange(lo, hi + width, width)

    p_edges = edges(cfg["p_nu_bin_keV"], 0.0, p_max, sigma_p)
    if isotope.decay == "EC":
        return (p_edges,)
    sigma_te = detector.energy_resolution_coeff * np.sqrt(1000.0 * isotope.q_kev)
    te_max = isotope.q_kev + 8.0 * sigma_te
    te_edges = edges(cfg["te_bin_keV"], 0.0, te_max, sigma_te)
    return (p_edges, te_edges)


def make_template(sample, bin_edges, trigger_efficiency):
    """Histogram a detectable EventSample into a normalized Template."""
    sel = sample.detectable
    if sample.te_rec_kev is not None and len(bin_edges) == 2:
        counts, _ = np.histogramdd(
            np.column_stack([sample.p_nu_rec_kev[sel], sample.te_rec_kev[sel]]),
            bins=bin_edges,
        )
    else:
        counts, _ = np.histogram(sample.p_nu_rec_kev[sel], bins=bin_edges[0])
    counts = counts.ravel().astype(float)
    in_range = counts.sum() / max(sel.sum(), 1)
    if counts.sum() == 0.0:
        return Template(probs=counts, efficiency=0.0)
    return Template(
        probs=counts / counts.sum(),
        efficiency=sample.detectable_fraction * in_range * trigger_efficiency,
    )
