"""Electron energy loss in the nanosphere via the continuous-slowing-down
approximation (CSDA), using a tabulated electronic stopping power.

Electrons are assumed to travel in straight lines from the decay location to
the sphere surface; an electron escapes if its CSDA range exceeds the path
length, exiting with the energy at which its residual range equals zero.
"""

import numpy as np


class StoppingModel:
    def __init__(self, table_path, density_g_cm3):
        energies, powers = [], []
        with open(table_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                e, s = line.split(",")
                energies.append(float(e))
                powers.append(float(s))
        self.energy_kev = np.array(energies)  # [keV]
        # stopping power dE/dx [keV/nm]: MeV cm^2/g * g/cm^3 = MeV/cm = 1e-4 keV/nm
        self.dedx_kev_nm = np.array(powers) * density_g_cm3 * 1e-4
        self._log_e = np.log(self.energy_kev)
        self._log_dedx = np.log(self.dedx_kev_nm)
        self._build_range_table()

    def _build_range_table(self):
        """CSDA range R(E) = int_0^E dE'/S(E') on a fine log grid [nm]."""
        e_grid = np.geomspace(self.energy_kev[0], self.energy_kev[-1], 2000)
        dedx = self.dedx(e_grid)
        # integrate 1/S dE cumulatively (trapezoid); range below the table
        # minimum (10 eV) is negligible and treated as zero
        inv = 1.0 / dedx
        dr = 0.5 * (inv[1:] + inv[:-1]) * np.diff(e_grid)
        self._range_e = e_grid
        self._range_nm = np.concatenate([[0.0], np.cumsum(dr)])

    def dedx(self, energy_kev):
        """Interpolated stopping power [keV/nm] (log-log)."""
        e = np.clip(energy_kev, self.energy_kev[0], self.energy_kev[-1])
        return np.exp(np.interp(np.log(e), self._log_e, self._log_dedx))

    def csda_range_nm(self, energy_kev):
        e = np.clip(energy_kev, self.energy_kev[0], self.energy_kev[-1])
        return np.interp(e, self._range_e, self._range_nm)

    def energy_from_range_nm(self, range_nm):
        """Inverse of csda_range_nm."""
        return np.interp(range_nm, self._range_nm, self._range_e)

    def exit_energy_kev(self, energy_kev, path_nm):
        """Kinetic energy [keV] after traversing path_nm; 0 if stopped."""
        energy_kev = np.asarray(energy_kev, dtype=float)
        residual = self.csda_range_nm(energy_kev) - np.asarray(path_nm, dtype=float)
        exit_e = np.where(residual > 0.0, self.energy_from_range_nm(np.maximum(residual, 0.0)), 0.0)
        # electrons below the table minimum effectively stop immediately
        return np.where(exit_e > self.energy_kev[0], exit_e, 0.0)
