#!/usr/bin/env python
"""Diagnostic spectrum plots (paper Figs. 2-3 style): the reconstructed
neutrino momentum spectrum for an EC isotope with an injected sterile signal,
or the 2D (|p_nu|, Te) distribution with 2-sigma contours for a beta isotope.

Examples:
  python scripts/plot_spectra.py --isotope Ar37 --m4 750 --ue4sq 2e-4
  python scripts/plot_spectra.py --isotope P32 --m4 100 340 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quips import decay, plotting
from quips.config import load_detector_config, load_isotopes
from quips.pdfs import make_binning
from quips.simulate import DecaySimulator
from quips.stopping import StoppingModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", default="config/detector.yaml")
    ap.add_argument("--isotopes", default="config/isotopes.yaml")
    ap.add_argument("--isotope", required=True)
    ap.add_argument("--m4", nargs="+", type=float, required=True, help="m4 [keV]")
    ap.add_argument("--ue4sq", type=float, default=2e-4,
                    help="injected |Ue4|^2 for the EC spectrum overlay")
    ap.add_argument("--output", "-o", default="results")
    args = ap.parse_args()

    isotopes = load_isotopes(args.isotopes)
    iso = isotopes[args.isotope]
    cfg = load_detector_config(args.detector, args.isotope)
    rng = np.random.default_rng(cfg.random_seed)
    stopping = StoppingModel(cfg.stopping_table_path, cfg.density_g_cm3)
    sim = DecaySimulator(cfg, iso, stopping, rng)
    bins = make_binning(cfg, iso)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    n_mc = cfg.n_mc_events
    light = sim.run(0.0, n_mc)
    n_dec = cfg.total_decays(iso)

    if iso.decay == "EC":
        m4 = args.m4[0]
        heavy = sim.run(m4, n_mc)
        eff = cfg.trigger_efficiency
        n_light = n_dec * light.detectable_fraction * eff
        n_heavy = (n_dec * args.ue4sq * decay.phase_space_ratio(iso, m4)
                   * heavy.detectable_fraction * eff)
        path = out / f"spectrum_{args.isotope}.png"
        plotting.plot_ec_spectrum(light, heavy, bins, n_light, n_heavy,
                                  args.isotope, m4, args.ue4sq, path, rng=rng)
    else:
        samples = {0.0: light}
        for m4 in args.m4:
            samples[m4] = sim.run(m4, n_mc)
        path = out / f"spectrum2d_{args.isotope}.png"
        plotting.plot_beta_2d(samples, bins, args.isotope, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
