"""
instruments/vna.py

Virtual Vector Network Analyzer (VNA).
Simulates the measurement equipment used to characterize microwave resonators.
Handles frequency sweeping, power control, noise simulation, and averaging.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs): return iterable

class VirtualVNA:
    """
    A simulator for a Vector Network Analyzer.
    
    Mimics real-world VNA behaviors including:
    - Finite bandwidth noise floor (-160 dBm/Hz).
    - Electrical delay (cable phase roll).
    - Averaging (SNR improvement).
    - Complex S-parameter measurement (S21).
    """
    def __init__(self, start_ghz=1.0, stop_ghz=12.0, points=1001):
        self.freqs = np.linspace(start_ghz, stop_ghz, int(points))
        
        # Storage for measurement results
        self.s21_data = np.zeros_like(self.freqs, dtype=np.complex128)
        self.photon_data = np.zeros_like(self.freqs, dtype=np.float64) # "God Mode" data (simulation only)
        
        # VNA Settings
        self.power_dbm = -40.0   # Output power
        self.ifbw_hz = 1000.0    # Intermediate Frequency Bandwidth (Noise filter)
        self.averages = 100      # Number of sweeps to average
        self.elec_delay_ns = 0.0 # Virtual cable length compensation
        
        # Physics Constants for Noise
        self.noise_floor_per_hz_dbm = -160.0 # Thermal noise limit at room/cryo temp mix
        self.system_impedance = 50.0         # Standard microwave impedance

    def set_sweep(self, start_ghz, stop_ghz, points):
        """Configure the frequency axis."""
        self.freqs = np.linspace(start_ghz, stop_ghz, int(points))

    def set_power(self, dbm): self.power_dbm = dbm
    def set_ifbw(self, hz): self.ifbw_hz = hz
    def set_averages(self, n): self.averages = int(n)
    def set_electrical_delay(self, ns): self.elec_delay_ns = ns

    def scan(self, dut):
        """
        Performs a frequency sweep on the Device Under Test (DUT).
        
        The Scan Process:
        1. Calculate drive voltage from power (dBm).
        2. Query DUT for ideal physics response.
        3. Apply 'Cable Delay' phase rotation.
        4. Add Gaussian noise based on IFBW.
        5. Average multiple traces to simulate VNA integration.
        """
        print(f"--- [VNA Scan Started] ---")
        print(f"  Range: {self.freqs[0]:.2f} - {self.freqs[-1]:.2f} GHz ({len(self.freqs)} pts)")
        print(f"  Power: {self.power_dbm} dBm | IFBW: {self.ifbw_hz} Hz")
        
        # Convert Power (dBm) -> Voltage (V)
        p_drive_watts = 10**(self.power_dbm/10.0)/1000.0
        v_drive = np.sqrt(p_drive_watts * self.system_impedance)
        
        # Calculate Noise Voltage Sigma
        # Noise Power = Floor + 10*log10(Bandwidth)
        p_noise_dbm = self.noise_floor_per_hz_dbm + 10*np.log10(self.ifbw_hz)
        p_noise_watts = 10**(p_noise_dbm/10.0)/1000.0
        v_noise_sigma = np.sqrt(p_noise_watts * self.system_impedance)
        
        # --- Step 1: Ideal Physics Query ---
        ideal_s21 = []
        ideal_n = [] # Store ideal photon numbers
        iterator = tqdm(self.freqs, desc=f"Sweeping {self.power_dbm} dBm", unit="pts")
        
        for f in iterator:
            # DUT.response returns (S21_complex, Photon_Number_float)
            resp, n = dut.response(readout_freq_ghz=f, readout_dbm=self.power_dbm)
            ideal_s21.append(resp)
            ideal_n.append(n)
            
        ideal_s21 = np.array(ideal_s21)
        self.photon_data = np.array(ideal_n) # Stored for plotting/debugging
        
        # --- Step 2: Cable Delay Simulation ---
        # Real cables add a phase slope: phi = -omega * delay
        if self.elec_delay_ns != 0:
            omega = 2 * np.pi * self.freqs * 1e9 
            tau = self.elec_delay_ns * 1e-9      
            phase_factor = np.exp(-1j * omega * tau)
            ideal_s21 *= phase_factor

        # --- Step 3: Measurement Loop (Noise & Averaging) ---
        acc_s21 = np.zeros_like(ideal_s21, dtype=np.complex128)

        for i in range(self.averages):
            # Generate random thermal noise (I and Q channels independent)
            noise_i = np.random.normal(0, v_noise_sigma, len(self.freqs))
            noise_q = np.random.normal(0, v_noise_sigma, len(self.freqs))
            complex_noise = noise_i + 1j * noise_q
            
            # Simulated Measured Voltage = Signal + Noise
            measured_voltage = (v_drive * ideal_s21) + complex_noise
            
            # VNA reads ratio: V_measured / V_source
            acc_s21 += measured_voltage / v_drive

        # Average the accumulation buffer
        self.s21_data = acc_s21 / self.averages
        print("--- [Scan Complete] ---\n")
        return self.freqs, self.s21_data

    def plot(self, auto_compensate=False, filename=None):
        """
        Visualizes the measurement data in a standard 4-panel dashboard.
        
        Panels:
        1. Magnitude (dB) - The dip/peak.
        2. Phase (Radians) - The phase shift across resonance.
        3. IQ Plane (Smith Chart style) - The resonance circle.
        4. Photon Number - Internal cavity population (Simulation exclusive).
        
        Parameters
        ----------
        auto_compensate : bool
            If True, removes the linear phase slope (electrical delay) 
            from the Phase and IQ plots for clearer viewing.
        """
        if self.s21_data is None: 
            print("No data to plot. Run .scan(dut) first.")
            return

        s21 = self.s21_data
        s21_mag = 20 * np.log10(np.abs(s21) + 1e-20)
        s21_phase = np.angle(s21)
        
        slope_title = ""
        # Auto-remove cable delay for visualization purposes
        if auto_compensate:
            s21_phase = np.unwrap(s21_phase)
            slope = np.polyfit(self.freqs, s21_phase, 1)
            s21_phase -= np.polyval(slope, self.freqs)
            phase_corr = np.exp(-1j * np.polyval(slope, self.freqs))
            s21_corrected = s21 * phase_corr
            slope_title = " (Delay Removed)"
        else:
            s21_corrected = s21

        # --- PLOTTING (2x2 Grid) ---
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"VNA Scan: P={self.power_dbm} dBm | Avg={self.averages}", fontsize=14)

        # 1. Magnitude
        plt.subplot(2, 2, 1)
        plt.plot(self.freqs, s21_mag, 'b')
        plt.title("Magnitude (dB)")
        plt.ylabel("dB"); plt.xlabel("Frequency (GHz)")
        plt.grid(True, alpha=0.3)

        # 2. Phase
        plt.subplot(2, 2, 2)
        plt.plot(self.freqs, s21_phase, 'r')
        plt.title(f"Phase{slope_title}")
        plt.ylabel("Radians"); plt.xlabel("Frequency (GHz)")
        plt.grid(True, alpha=0.3)

        # 3. IQ Circle (Polar Plot)
        plt.subplot(2, 2, 3)
        plt.plot(s21_corrected.real, s21_corrected.imag, 'g-', alpha=0.5, marker='o', markersize=3)
        
        # Highlight specific points (Resonance, Start)
        res_idx = np.argmin(np.abs(s21)) 
        plt.plot(s21_corrected.real[res_idx], s21_corrected.imag[res_idx], 'ro', label="Res")
        plt.plot(s21_corrected.real[0], s21_corrected.imag[0], 'kx', label="Start")
        
        plt.title(f"IQ Circle{slope_title}")
        plt.xlabel("I"); plt.ylabel("Q")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
        # 4. Photon Number (Internal Physics)
        plt.subplot(2, 2, 4)
        plt.plot(self.freqs, self.photon_data, 'orange', lw=2)
        plt.title("Photon Number")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel(r"$\langle n \rangle = \langle a^\dagger a \rangle$")
        plt.grid(True, alpha=0.3)
        
        # Label Max Photons
        max_n = np.max(self.photon_data)
        plt.text(0.5, 0.9, f"Max n = {max_n:.3f}", transform=plt.gca().transAxes, 
                 ha='center', bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        if filename:
            if not os.path.exists("images"): os.makedirs("images")
            plt.savefig(f"images/{filename}")
            print(f"Saved plot to images/{filename}")

        plt.show()