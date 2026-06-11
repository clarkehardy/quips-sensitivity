# quips-sensitivity

Sensitivity projections for sterile-neutrino searches with optically levitated
nanospheres, following the Monte Carlo described in Carney, Leach & Moore,
[*Searches for Massive Neutrinos with Mechanical Quantum Sensors*, PRX Quantum
**4**, 010315 (2023)](https://doi.org/10.1103/PRXQuantum.4.010315)
(`papers/Carney_PRX-Quantum_2023.pdf`).

Radioisotopes embedded in a levitated nanosphere undergo EC or β⁻ decay; the
sphere's center-of-mass momentum kick, measured near the standard quantum
limit, together with any detected secondaries reconstructs the neutrino
momentum event by event. A heavy sterile state `m4` appears as a displaced
population in reconstructed `|p_nu|` (EC) or in the (`|p_nu|`, `Te`) plane
(β⁻); a binned profile-likelihood fit yields the median 95% CL upper limit on
`|Ue4|^2` as a function of `m4`.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# full sensitivity scan, all isotopes in config/isotopes.yaml
python scripts/run_sensitivity.py

# specific isotopes, custom output directory
python scripts/run_sensitivity.py --isotope Ar37 P32 -o results

# diagnostic spectra (paper Figs. 2-3)
python scripts/plot_spectra.py --isotope Ar37 --m4 750 --ue4sq 2e-4
python scripts/plot_spectra.py --isotope P32 --m4 100 340 500 800
```

Outputs in `results/`: per-isotope `sensitivity_<name>.csv`
(`m4_keV, Ue4sq_95CL`) and `sensitivity_EC.png` / `sensitivity_beta.png`
with the existing laboratory limits (CSVs in `data/`, format
`log10(m4/GeV), log10(|Ue4|^2)`) drawn as a gray envelope.

## Configuration

All experimental parameters live in `config/detector.yaml`: sphere size,
material and isotope loading, trap frequency, per-axis information collection
efficiencies (SQL readout noise `sigma_i = sqrt(hbar*m*omega/eta_i)`),
secondary-particle trigger efficiency / angular and energy resolution /
detection threshold, exposure, the `m4` grid, binning, and the statistical
method (`asimov` for the fast asymptotic median limit, `toys` for the paper's
toy-MC median). `isotope_overrides` applies per-isotope settings (e.g. the
25 nm polystyrene sphere for ³H).

Isotope nuclear data live in `config/isotopes.yaml` (see the schema comment
at the top). To add an isotope, add an entry with its decay type, Q value,
half-life, branches, and Auger/x-ray lines — no code changes needed.

## Package layout

| module | role |
|---|---|
| `quips/constants.py` | units and physical constants (keV, keV/c, nm, days) |
| `quips/config.py` | YAML loading, derived quantities (sphere mass, SQL noise, decay counts) |
| `quips/stopping.py` | CSDA electron transport with tabulated stopping power (`data/stopping/`) |
| `quips/decay.py` | EC/β kinematics, allowed β spectrum, phase-space ratios |
| `quips/simulate.py` | vectorized event MC: emission, propagation, smearing, `p_nu` reconstruction |
| `quips/pdfs.py` | binned 1D/2D templates and detection efficiencies |
| `quips/sensitivity.py` | extended binned NLL, profile-likelihood upper limits, toys/Asimov |
| `quips/plotting.py` | spectrum and sensitivity figures |

## Validation against the paper

- 100 nm SiO₂ sphere: mass 1.05 fg, `dp_SQL` = 15.6 keV/c at 100 kHz (Eq. 5).
- Live time for 10⁴ detected events: Ar37 9.0 (paper: 9), Be7 20 (24),
  V49 124 (119), Ge68 127 (147), P32 2.6 (2.8), Y90 1.4 (1.7) sphere-days
  (Tables I-II).
- Reconstructed spectra reproduce Figs. 2-3; sensitivity curves reproduce the
  reach of Figs. 4-5 (minima at `|Ue4|^2` ~ 1-10 × 10⁻⁵ for the 100 nm
  isotopes at `m4` ≈ 0.3-1 MeV with a 1 sphere-month exposure).
