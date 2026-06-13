"""Plotting: reconstructed-spectrum diagnostics (paper Figs. 2-3 style) and
|Ue4|^2 vs m4 sensitivity curves with existing-limit overlays (Figs. 4-5 style).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KEV_PER_GEV = 1e6


def load_limit_curve(data_dir, name):
    """Load data/<name>_data.csv: columns log10(m4/GeV), log10(|Ue4|^2).
    Returns (m4_kev, ue4sq) sorted by mass."""
    arr = np.loadtxt(Path(data_dir) / f"{name}_data.csv", delimiter=",")
    order = np.argsort(arr[:, 0])
    return 10.0 ** arr[order, 0] * KEV_PER_GEV, 10.0 ** arr[order, 1]


def limits_envelope(data_dir, names, m4_grid_kev):
    """Pointwise minimum of the existing limits over a mass grid (NaN where
    no experiment constrains)."""
    envelope = np.full_like(m4_grid_kev, np.inf, dtype=float)
    log_grid = np.log10(m4_grid_kev)
    for name in names:
        m4, ue4 = load_limit_curve(data_dir, name)
        interp = np.interp(
            log_grid, np.log10(m4), np.log10(ue4), left=np.inf, right=np.inf
        )
        envelope = np.minimum(envelope, 10.0**interp)
    return np.where(np.isfinite(envelope), envelope, np.nan)


def seesaw_ue4sq(m4_kev, m_nu_ev):
    """Type-I seesaw expectation |Ue4|^2 = m_nu / m4 (Bolton et al.,
    arXiv:1912.03058): the mixing needed to generate a light-neutrino mass
    m_nu from a sterile state of mass m4."""
    return 1e-3 * m_nu_ev / np.asarray(m4_kev, dtype=float)


def plot_sensitivity(results, data_dir, limit_names, output_path, title=None,
                     seesaw_m_nu_ev=None):
    """Sensitivity curves |Ue4|^2 vs m4.

    results: {isotope_name: (m4_kev, ue4sq)}; one panel per decay type is the
    caller's choice -- this draws everything passed in on a single axis.
    seesaw_m_nu_ev: light-neutrino mass [eV] for the seesaw target line, or a
    [low, high] pair to draw a band.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    m4_lo = min(np.nanmin(m[0]) for m in results.values())
    m4_hi = max(np.nanmax(m[0]) for m in results.values())
    grid = np.geomspace(0.5 * m4_lo, 2.0 * m4_hi, 600)
    if limit_names:
        env = limits_envelope(data_dir, limit_names, grid)
        ax.fill_between(grid, env, 1.0, where=np.isfinite(env), color="0.8", lw=0)
        ax.plot(grid, env, color="0.5", lw=1, label="existing limits")
    for name, (m4, ue4) in results.items():
        good = np.isfinite(ue4)
        ax.plot(m4[good], ue4[good], lw=2, label=name)
    bottom = max(min(np.nanmin(u[1]) for u in results.values()) * 0.1, 1e-12)
    if seesaw_m_nu_ev is not None:
        m_nu = np.atleast_1d(seesaw_m_nu_ev)
        if len(m_nu) == 2:
            ax.fill_between(grid, seesaw_ue4sq(grid, m_nu.min()),
                            seesaw_ue4sq(grid, m_nu.max()), color="C8", alpha=0.4,
                            lw=0, label=rf"seesaw, $m_\nu$={m_nu.min():g}-{m_nu.max():g} eV")
            bottom = min(bottom, 0.5 * seesaw_ue4sq(m4_hi, m_nu.min()))
        else:
            ax.plot(grid, seesaw_ue4sq(grid, m_nu[0]), "k--", lw=1,
                    label=rf"seesaw, $m_\nu$={m_nu[0]:g} eV")
            bottom = min(bottom, 0.5 * seesaw_ue4sq(m4_hi, m_nu[0]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_4$ [keV]")
    ax.set_ylabel(r"$|U_{e4}|^2$ (95% CL)")
    ax.set_xlim(0.5 * m4_lo, 2.0 * m4_hi)
    ax.set_ylim(bottom=bottom, top=1.0)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ec_spectrum(
    light_sample, heavy_sample, bin_edges, n_light, n_heavy, isotope_name, m4_kev,
    ue4sq, output_path, rng=None,
):
    """Fig. 2-style reconstructed |p_nu| spectrum: a toy realization of the
    light-neutrino spectrum with the sterile contribution overlaid."""
    edges = bin_edges[0]
    centers = 0.5 * (edges[1:] + edges[:-1])
    h_light, _ = np.histogram(light_sample.p_nu_rec_kev[light_sample.detectable], bins=edges)
    h_heavy, _ = np.histogram(heavy_sample.p_nu_rec_kev[heavy_sample.detectable], bins=edges)
    mu_light = n_light * h_light / max(h_light.sum(), 1)
    mu_heavy = n_heavy * h_heavy / max(h_heavy.sum(), 1)
    counts = (np.random.default_rng() if rng is None else rng).poisson(mu_light + mu_heavy)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    nonzero = counts > 0
    ax.errorbar(centers[nonzero], counts[nonzero], yerr=np.sqrt(counts[nonzero]),
                fmt="k.", ms=3, lw=1, label="simulated data")
    ax.plot(centers, mu_light, "C0", label=r"light $\nu$")
    ax.plot(centers, mu_heavy, "C3",
            label=rf"$m_4$={m4_kev:.0f} keV, $|U_{{e4}}|^2$={ue4sq:g}")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.1)
    ax.set_xlabel(r"reconstructed $|p_\nu|$ [keV/c]")
    ax.set_ylabel(f"counts per {edges[1] - edges[0]:.3g} keV/c")
    ax.set_title(isotope_name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_beta_2d(samples_by_m4, bin_edges, isotope_name, output_path):
    """Fig. 3-style |p_nu| vs Te: light-neutrino density plus 2-sigma contours
    for each m4 in samples_by_m4 ({m4_kev: EventSample}, key 0.0 = light)."""
    p_edges, te_edges = bin_edges
    fig, ax = plt.subplots(figsize=(6, 4.5))
    light = samples_by_m4[0.0]
    sel = light.detectable
    h, _, _ = np.histogram2d(light.te_rec_kev[sel], light.p_nu_rec_kev[sel],
                             bins=(te_edges, p_edges))
    ax.pcolormesh(te_edges, p_edges, h.T, cmap="Greys", rasterized=True)
    for i, (m4, sample) in enumerate(sorted(samples_by_m4.items())):
        sel = sample.detectable
        h, _, _ = np.histogram2d(sample.te_rec_kev[sel], sample.p_nu_rec_kev[sel],
                                 bins=(te_edges, p_edges))
        h_smooth = h / max(h.sum(), 1)
        # 2-sigma contour: smallest density level containing 95% of events
        flat = np.sort(h_smooth.ravel())[::-1]
        level = flat[np.searchsorted(np.cumsum(flat), 0.95)]
        tc = 0.5 * (te_edges[1:] + te_edges[:-1])
        pc = 0.5 * (p_edges[1:] + p_edges[:-1])
        label = r"light $\nu$ ($2\sigma$)" if m4 == 0.0 else rf"$m_4$={m4:.0f} keV"
        ax.contour(tc, pc, h_smooth.T, levels=[level], colors=[f"C{i}"])
        ax.plot([], [], color=f"C{i}", label=label)  # legend proxy for the contour
    ax.set_xlabel(r"$T_e$ [keV]")
    ax.set_ylabel(r"reconstructed $|p_\nu|$ [keV/c]")
    ax.set_title(isotope_name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
