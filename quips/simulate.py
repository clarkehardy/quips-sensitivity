"""Event-level Monte Carlo of decays inside the nanosphere.

For each simulated decay: the neutrino (and, for beta decay, the electron)
kinematics are drawn, atomic/nuclear secondaries are emitted, electrons are
propagated to the sphere surface with CSDA energy loss, the sphere momentum
kick is computed from momentum conservation over all escaping particles, and
the measurement process (SQL readout noise, secondary angular/energy
resolution) is applied to form the reconstructed neutrino momentum
p_nu = -(dp_sphere + sum p_secondary)  (paper Sec. III).
"""

from dataclasses import dataclass

import numpy as np

from . import decay
from .constants import electron_momentum_kev


@dataclass
class EventSample:
    """Result of simulating n decays to a neutrino of mass m4."""

    m4_kev: float
    p_nu_rec_kev: np.ndarray  # reconstructed |p_nu| per event
    te_rec_kev: np.ndarray  # measured beta kinetic energy (NaN for EC)
    detectable: np.ndarray  # bool: >=1 escaping secondary (incl. beta for beta decay)
    detectable_fraction: float


def _isotropic_directions(rng, n):
    cos_t = rng.uniform(-1.0, 1.0, n)
    sin_t = np.sqrt(1.0 - cos_t**2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])


def _smear_directions(rng, dirs, sigma_rad):
    """Gaussian angular smearing of unit vectors by sigma_rad per transverse axis."""
    n = len(dirs)
    # build an orthonormal basis transverse to each direction
    ref = np.where(np.abs(dirs[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    t1 = np.cross(dirs, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(dirs, t1)
    smeared = (
        dirs
        + rng.normal(0.0, sigma_rad, (n, 1)) * t1
        + rng.normal(0.0, sigma_rad, (n, 1)) * t2
    )
    return smeared / np.linalg.norm(smeared, axis=1, keepdims=True)


def _path_to_surface_nm(positions, directions, radius_nm):
    """Straight-line distance from interior points to the sphere surface."""
    b = np.einsum("ij,ij->i", positions, directions)
    r2 = np.einsum("ij,ij->i", positions, positions)
    return -b + np.sqrt(np.maximum(radius_nm**2 - r2 + b**2, 0.0))


def _emission_counts(rng, intensity, n):
    """Number of particles emitted per event for a line with mean intensity
    (deterministic floor plus Bernoulli remainder)."""
    base = int(np.floor(intensity))
    return base + (rng.uniform(size=n) < (intensity - base))


class DecaySimulator:
    def __init__(self, detector, isotope, stopping, rng):
        self.det = detector
        self.iso = isotope
        self.stopping = stopping
        self.rng = rng
        self.radius_nm = detector.sphere_diameter_nm / 2.0

    def _smeared_energy(self, energy_kev):
        """Apply sigma_E/E = coeff * sqrt(1 MeV / E) and clip at zero."""
        sigma = self.det.energy_resolution_coeff * np.sqrt(1000.0 * np.maximum(energy_kev, 1e-9))
        return np.maximum(energy_kev + self.rng.normal(0.0, 1.0, len(energy_kev)) * sigma, 0.0)

    def _emit_photons(self, mask, energy_kev, state):
        """Photons escape freely with their full energy."""
        n_sel = int(mask.sum())
        if n_sel == 0:
            return
        dirs = _isotropic_directions(self.rng, n_sel)
        state["p_escape"][mask] += energy_kev * dirs
        if energy_kev < self.det.min_secondary_energy_kev:
            return  # escapes but is below the detection threshold
        meas_dirs = _smear_directions(self.rng, dirs, self.det.angular_resolution_rad)
        e_meas = self._smeared_energy(np.full(n_sel, energy_kev))
        state["p_sec_rec"][mask] += e_meas[:, None] * meas_dirs
        state["n_detected"][mask] += 1

    def _emit_electrons(self, mask, energy_kev, state, record_te=False):
        """Electrons propagate with CSDA loss; only escapees contribute.

        energy_kev: scalar (line) or per-selected-event array (beta spectrum).
        """
        n_sel = int(mask.sum())
        if n_sel == 0:
            return
        energies = np.broadcast_to(np.asarray(energy_kev, dtype=float), (n_sel,))
        dirs = _isotropic_directions(self.rng, n_sel)
        path = _path_to_surface_nm(state["positions"][mask], dirs, self.radius_nm)
        e_exit = self.stopping.exit_energy_kev(energies, path)
        escaped = e_exit > 0.0
        p_exit = np.where(escaped, electron_momentum_kev(e_exit), 0.0)
        state["p_escape"][mask] += p_exit[:, None] * dirs
        detected = e_exit > max(self.det.min_secondary_energy_kev, 0.0)
        meas_dirs = _smear_directions(self.rng, dirs, self.det.angular_resolution_rad)
        e_meas = np.where(detected, self._smeared_energy(e_exit), 0.0)
        state["p_sec_rec"][mask] += np.where(
            detected, electron_momentum_kev(e_meas), 0.0
        )[:, None] * meas_dirs
        state["n_detected"][mask] += detected
        if record_te:
            te = np.full(len(mask), np.nan)
            te[mask] = np.where(detected, e_meas, np.nan)
            state["te_rec"] = np.where(np.isnan(state["te_rec"]), te, state["te_rec"])
            beta_detected = np.zeros(len(mask), dtype=bool)
            beta_detected[mask] = detected
            state["beta_detected"] |= beta_detected

    def run(self, m4_kev, n_events):
        det, iso, rng = self.det, self.iso, self.rng

        # --- choose decay branches, weighted by their rate at mass m4 -----
        weights = np.array(
            [b.fraction * decay.branch_phase_space_ratio(iso, b, m4_kev) for b in iso.branches]
        )
        if weights.sum() <= 0.0:
            raise ValueError(f"m4 = {m4_kev} keV is kinematically forbidden for {iso.name}")
        branch_idx = rng.choice(len(iso.branches), size=n_events, p=weights / weights.sum())

        # --- decay positions: uniform in sphere volume --------------------
        positions = (
            self.radius_nm
            * np.cbrt(rng.uniform(size=n_events))[:, None]
            * _isotropic_directions(rng, n_events)
        )

        state = {
            "positions": positions,
            "p_escape": np.zeros((n_events, 3)),  # true momentum leaving the sphere
            "p_sec_rec": np.zeros((n_events, 3)),  # reconstructed secondary momentum
            "n_detected": np.zeros(n_events, dtype=int),
            "te_rec": np.full(n_events, np.nan),
            "beta_detected": np.zeros(n_events, dtype=bool),
        }

        # --- primary neutrino (and beta electron) -------------------------
        e_nu = np.empty(n_events)
        for i, branch in enumerate(iso.branches):
            mask = branch_idx == i
            if not mask.any():
                continue
            if iso.decay == "EC":
                e_nu[mask] = iso.q_kev - branch.level_kev
            else:
                endpoint = iso.q_kev - branch.level_kev
                te = decay.sample_beta_kinetic_energy(
                    rng, int(mask.sum()), endpoint, iso.Z + 1, m4_kev
                )
                e_nu[mask] = endpoint - te
                self._emit_electrons(mask, te, state, record_te=True)
        p_nu = decay.neutrino_momentum_kev(e_nu, m4_kev)
        state["p_escape"] += p_nu[:, None] * _isotropic_directions(rng, n_events)

        # --- atomic relaxation (all branches) and de-excitation -----------
        for line in iso.augers:
            counts = _emission_counts(rng, line.intensity, n_events)
            for copy in range(counts.max()):
                self._emit_electrons(counts > copy, line.energy_kev, state)
        for line in iso.xrays:
            counts = _emission_counts(rng, line.intensity, n_events)
            for copy in range(counts.max()):
                self._emit_photons(counts > copy, line.energy_kev, state)
        for i, branch in enumerate(iso.branches):
            in_branch = branch_idx == i
            for line in branch.gammas:
                emitted = in_branch & (rng.uniform(size=n_events) < line.intensity)
                self._emit_photons(emitted, line.energy_kev, state)
            for line in branch.conversion_electrons:
                emitted = in_branch & (rng.uniform(size=n_events) < line.intensity)
                self._emit_electrons(emitted, line.energy_kev, state)

        # --- sphere kick, readout noise, neutrino reconstruction ----------
        dp_sphere = -state["p_escape"]
        noise = rng.normal(0.0, 1.0, (n_events, 3)) * det.momentum_noise_kev
        p_nu_rec_vec = -(dp_sphere + noise + state["p_sec_rec"])
        # use only the measured components (single-axis analyses fit the
        # projected spectrum |p_z| rather than the full 3D |p_nu|)
        p_nu_rec = np.linalg.norm(p_nu_rec_vec[:, det.measured_axes], axis=1)

        if iso.decay == "EC":
            detectable = state["n_detected"] >= 1
        else:
            # the 2D (|p_nu|, Te) analysis requires the beta to be measured
            detectable = state["beta_detected"]

        return EventSample(
            m4_kev=m4_kev,
            p_nu_rec_kev=p_nu_rec,
            te_rec_kev=state["te_rec"],
            detectable=detectable,
            detectable_fraction=float(detectable.mean()),
        )
