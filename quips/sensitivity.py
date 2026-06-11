"""Statistical sensitivity: extended binned likelihood fits of a light-neutrino
template plus a sterile template, profile-likelihood 95% CL upper limits, with
either toy Monte Carlo experiments (median limit) or the Asimov dataset.
"""

import numpy as np
from scipy import optimize, special, stats


def binned_nll(n_obs, mu):
    """Extended binned Poisson negative log likelihood (constants dropped)."""
    mu = np.maximum(mu, 1e-300)
    return float(np.sum(mu) - np.sum(special.xlogy(n_obs, mu)))


def _profile_light_yield(n_obs, f_light, f_sig, s, n_guess):
    """Maximize the likelihood over the light yield at fixed signal yield s.

    Solves sum_b f_light_b * (1 - n_b/mu_b) = 0, which is monotonic in the
    light yield.
    """

    def grad(nl):
        mu = nl * f_light + s * f_sig
        return float(np.sum(f_light * (1.0 - np.where(mu > 0, n_obs / np.maximum(mu, 1e-300), 0.0))))

    lo, hi = 1e-9, max(2.0 * n_guess, 10.0)
    while grad(hi) < 0.0:
        hi *= 4.0
    if grad(lo) >= 0.0:
        return lo
    return optimize.brentq(grad, lo, hi, xtol=1e-6, rtol=1e-10)


def profile_nll(n_obs, f_light, f_sig, s, n_guess):
    nl = _profile_light_yield(n_obs, f_light, f_sig, s, n_guess)
    return binned_nll(n_obs, nl * f_light + s * f_sig)


def upper_limit_yield(n_obs, f_light, f_sig, cl=0.95):
    """Profile-likelihood upper limit on the signal yield s >= 0.

    Uses the one-sided asymptotic criterion 2*[NLL(s) - NLL(s_hat)] = q_cl
    with q_cl = (Phi^-1(cl))^2 (2.706 at 95% CL).
    """
    q_cl = stats.norm.ppf(cl) ** 2
    n_tot = float(np.sum(n_obs))

    def pnll(s):
        return profile_nll(n_obs, f_light, f_sig, s, n_tot)

    # best fit with s >= 0
    s_scale = max(np.sqrt(n_tot), 10.0)
    res = optimize.minimize_scalar(pnll, bounds=(0.0, 50.0 * s_scale), method="bounded")
    s_hat, nll_hat = max(res.x, 0.0), res.fun
    if pnll(0.0) <= nll_hat:  # data prefer no signal
        s_hat, nll_hat = 0.0, pnll(0.0)

    def q_minus_crit(s):
        return 2.0 * (pnll(s) - nll_hat) - q_cl

    hi = s_hat + s_scale
    while q_minus_crit(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            return np.inf
    return optimize.brentq(q_minus_crit, s_hat, hi, xtol=1e-3, rtol=1e-6)


def median_toy_upper_limit(rng, n_light_expected, f_light, f_sig, n_toys, cl=0.95):
    """Median upper limit over toy experiments drawn from the light-only model."""
    limits = np.empty(n_toys)
    for i in range(n_toys):
        n_obs = rng.poisson(n_light_expected * f_light)
        limits[i] = upper_limit_yield(n_obs, f_light, f_sig, cl)
    return float(np.median(limits))


def asimov_upper_limit(n_light_expected, f_light, f_sig, cl=0.95):
    """Upper limit on the Asimov (expectation-valued) light-only dataset,
    the asymptotic median expected limit."""
    return upper_limit_yield(n_light_expected * f_light, f_light, f_sig, cl)
