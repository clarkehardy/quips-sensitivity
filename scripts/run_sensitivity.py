#!/usr/bin/env python
"""Run the full sensitivity scan: for each isotope, simulate the light-neutrino
and sterile-neutrino reconstructed spectra over a grid of m4, compute the
median 95% CL upper limit on |Ue4|^2, and write CSVs plus sensitivity figures.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quips import decay, plotting, sensitivity
from quips.config import load_detector_config, load_isotopes
from quips.pdfs import make_binning, make_template
from quips.simulate import DecaySimulator
from quips.stopping import StoppingModel


def m4_grid_kev(cfg, isotope):
    g = cfg.m4_grid
    hi = min(float(g["max_keV"]), 0.995 * isotope.max_neutrino_energy_kev)
    lo = min(float(g["min_keV"]), 0.5 * hi)
    n = int(g["n_points"])
    if g.get("log_spacing", True):
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def scan_isotope(name, detector_path, isotopes, verbose=True):
    cfg = load_detector_config(detector_path, name)
    iso = isotopes[name]
    rng = np.random.default_rng(cfg.random_seed)
    stopping = StoppingModel(cfg.stopping_table_path, cfg.density_g_cm3)
    sim = DecaySimulator(cfg, iso, stopping, rng)

    n_mc = cfg.n_mc_events
    bins = make_binning(cfg, iso)
    light = sim.run(0.0, n_mc)
    t_light = make_template(light, bins, cfg.trigger_efficiency)
    n_dec = cfg.total_decays(iso)
    n_light_exp = n_dec * t_light.efficiency

    stats_cfg = cfg.statistics
    method = stats_cfg["method"]
    cl = float(stats_cfg["confidence_level"])

    if verbose:
        print(
            f"[{name}] {iso.decay}, Q={iso.q_kev:g} keV | sphere {cfg.sphere_diameter_nm:g} nm "
            f"{cfg.material} ({cfg.sphere_mass_g * 1e15:.3g} fg), dp_SQL={cfg.sql_momentum_kev:.3g} keV/c | "
            f"{n_dec:.3g} decays, {n_light_exp:.3g} detected | method={method}"
        )

    grid = m4_grid_kev(cfg, iso)
    limits = np.full(len(grid), np.nan)
    for j, m4 in enumerate(grid):
        rho = decay.phase_space_ratio(iso, m4)
        if rho <= 1e-12:
            continue
        heavy = sim.run(m4, n_mc)
        t_sig = make_template(heavy, bins, cfg.trigger_efficiency)
        if t_sig.efficiency <= 0.0:
            continue
        keep = (t_light.probs > 0) | (t_sig.probs > 0)
        f_l, f_s = t_light.probs[keep], t_sig.probs[keep]
        if method == "toys":
            s_ul = sensitivity.median_toy_upper_limit(
                rng, n_light_exp, f_l, f_s, int(stats_cfg["n_toys"]), cl
            )
        else:
            s_ul = sensitivity.asimov_upper_limit(n_light_exp, f_l, f_s, cl)
        limits[j] = s_ul / (n_dec * rho * t_sig.efficiency)
        if verbose:
            print(f"  m4 = {m4:8.1f} keV  ->  |Ue4|^2 < {limits[j]:.3g}")
    return cfg, grid, limits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", default="config/detector.yaml")
    ap.add_argument("--isotopes", default="config/isotopes.yaml")
    ap.add_argument("--isotope", nargs="+", default=None,
                    help="isotope names to run (default: all in the isotope file)")
    ap.add_argument("--output", "-o", default="results")
    ap.add_argument("--data-dir", default="data",
                    help="directory with existing-limit CSVs")
    args = ap.parse_args()

    isotopes = load_isotopes(args.isotopes)
    names = args.isotope or list(isotopes)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results, last_cfg = {}, None
    for name in names:
        t0 = time.time()
        last_cfg, grid, limits = scan_isotope(name, args.detector, isotopes)
        results[name] = (grid, limits)
        np.savetxt(
            out / f"sensitivity_{name}.csv",
            np.column_stack([grid, limits]),
            delimiter=",",
            header="m4_keV,Ue4sq_95CL",
            comments="",
        )
        print(f"[{name}] done in {time.time() - t0:.1f} s -> {out}/sensitivity_{name}.csv")

    limit_names = last_cfg.plotting.get("existing_limits", []) if last_cfg else []
    for decay_type, suffix in (("EC", "EC"), ("beta-", "beta")):
        subset = {n: r for n, r in results.items() if isotopes[n].decay == decay_type}
        if subset:
            path = out / f"sensitivity_{suffix}.png"
            label = "EC" if decay_type == "EC" else "$\\beta^-$"
            plotting.plot_sensitivity(
                subset, args.data_dir, limit_names, path, title=f"{label} isotopes",
                seesaw_m_nu_ev=last_cfg.plotting.get("seesaw_m_nu_eV"),
            )
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
