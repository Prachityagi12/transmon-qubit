"""
architectures/bare_resonator.py

Simulates a linear superconducting resonator (harmonic oscillator).
Used for initial characterization, VNA sweeps, and calibrating the
relationship between input power (dBm) and internal photon number.
"""

import numpy as np
import qutip as qt

class BareResonator:
    """
    A linear resonator modeled as a quantum harmonic oscillator.
    
    Parameters
    ----------
    f_res_ghz : float
        Resonance frequency in GHz.
    Q_int : float
        Internal quality factor (dielectric/material losses).
    Q_ext : float
        External quality factor (coupling to the transmission line).
    geometry : str
        'side_coupled' (hanger/notch) or 'transmission_coupled'.
    N : int
        Size of the Hilbert space (truncation level).
    """
    def __init__(self, f_res_ghz=6.0, Q_int=1e6, Q_ext=5e3, 
                 geometry="side_coupled", N=10):
        if geometry not in ["side_coupled", "transmission_coupled"]:
            raise ValueError("Geometry must be 'side_coupled' or 'transmission_coupled'")

        self.geometry = geometry
        self.wr = f_res_ghz * 2 * np.pi
        
        # Calculate decay rates (kappa = omega / Q)
        self.kappa_int = self.wr / Q_int
        self.kappa_ext = self.wr / Q_ext
        self.kappa_tot = self.kappa_int + self.kappa_ext
        
        # Define operators for the harmonic oscillator
        self.N = N
        self.a = qt.destroy(self.N)
        self.adag = self.a.dag()
        
        # --- CALIBRATION: Power (dBm) to Drive Amplitude (Hz) ---
        # We need to map real-world units (dBm) to the Hamiltonian drive strength (epsilon).
        # We define a reference point: -100 dBm creates exactly 0.1 photons on resonance.
        # This provides a consistent scaling factor ('attenuation') for all experiments.
        p_ref_dbm = -100.0
        n_ref_photons = 0.1
        p_ref_watts = 10**(p_ref_dbm/10.0) / 1000.0
        
        # Derived from steady-state solution: n = |epsilon / (kappa/2)|^2
        epsilon_ref = self.kappa_tot * np.sqrt(n_ref_photons) / 2.0 
        self.attenuation = epsilon_ref / np.sqrt(p_ref_watts)

    def response(self, drive_freq_ghz, power_dbm=-100):
        """
        Calculates the steady-state transmission (S21) and photon number.

        Parameters
        ----------
        drive_freq_ghz : float
            Frequency of the VNA probe tone.
        power_dbm : float
            Power of the probe tone in dBm.

        Returns
        -------
        s21 : complex
            Complex transmission coefficient (S21).
        n_avg : float
            Average steady-state photon number <a^dag a>.
        """
        wd = drive_freq_ghz * 2 * np.pi
        
        # Detuning (Rotating Frame of the Drive)
        # delta = w_resonator - w_drive
        delta = self.wr - wd
        
        # Convert Power to Drive Amp using the calibrated attenuation factor
        p_watts = 10**(power_dbm/10.0) / 1000.0
        drive_amp = np.sqrt(p_watts) * self.attenuation

        # Hamiltonian in the Rotating Frame
        # H = delta * a^dag * a  +  epsilon * (a + a^dag)
        H = (delta * self.adag * self.a) + drive_amp * (self.a + self.adag)
        
        # Collapse operators (Energy relaxation via kappa_tot)
        c_ops = [np.sqrt(self.kappa_tot) * self.a]
        
        # Solve Master Equation for Steady State
        rho_ss = qt.steadystate(H, c_ops)
        a_exp = qt.expect(self.a, rho_ss)
        
        # --- INPUT-OUTPUT THEORY ---
        # Relates the internal field <a> to the transmitted signal S21.
        # Leakage ~ kappa_ext * <a>
        leakage_signal = 1j * 0.5 * (self.kappa_ext * a_exp / drive_amp)
        
        if self.geometry == "side_coupled":
            # Hanger measurement (S21 = 1 - transmission through coupler)
            s21 = 1.0 - leakage_signal
        elif self.geometry == "transmission_coupled":
            # Inline measurement
            s21 = -leakage_signal
            
        # Calc Photons
        n_avg = qt.expect(self.adag * self.a, rho_ss)
            
        return s21, n_avg