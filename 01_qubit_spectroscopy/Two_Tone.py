import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
try: from tqdm import tqdm
except ImportError: tqdm = lambda x, **k: x

from architectures.qubit_resonator import QubitResonator

# VirtualTwoTone Class
class VirtualTwoTone:
    def __init__(self):
        self.readout_freq = 6.000125
        self.readout_power = -100   # safe Readout power
        self.drive_power = -120     # Low power for natural linewidth
        self.start = 5.18           # Zoomed Start
        self.stop = 5.22            # Zoomed Stop
        self.points = 401
        self.freqs = None
        self.mags = None

    def scan(self, dut):
        self.freqs = np.linspace(self.start, self.stop, self.points)
        self.mags = []
        for f_drive in tqdm(self.freqs):
            s21, _ = dut.response(
                readout_freq_ghz=self.readout_freq,
                readout_dbm=self.readout_power,
                qubit_drive_freq_ghz=f_drive,
                qubit_drive_dbm=self.drive_power
            )
            self.mags.append(np.abs(s21))
        self.mags = np.array(self.mags)

# Lorentzian Fit Function
def lorentzian(f, A, f0, gamma, offset):
    return A * (gamma**2 / ((f - f0)**2 + gamma**2)) + offset

# Execution

# DUT Setup 
dut = QubitResonator(wc_ghz=6.0, wq_ghz=5.2, g_mhz=10, t1_ns=1000, t2_ns=1000)

spec = VirtualTwoTone()

spec.start = 5.18
spec.stop = 5.22
spec.points = 401
spec.drive_power = -120

spec.scan(dut=dut)

# Fitting Logic
freqs = spec.freqs
mags = spec.mags

# Initial Guess (for a Narrow peak gamma is kept small)
p0 = [max(mags)-min(mags), freqs[np.argmax(mags)], 0.001, min(mags)]

try:
    popt, _ = curve_fit(lorentzian, freqs, mags, p0=p0)
    A_fit, f0_fit, gamma_fit, offset_fit = popt
    
    fwhm = 2 * abs(gamma_fit)
    t2_star = 1 / (np.pi * fwhm) # Result in ns (since freq is in GHz)

    print("\n" + "="*40)
    print(f"RESULTS AT DRIVE POWER: {spec.drive_power} dBm")
    print("-" * 40)
    print(f"Qubit Freq (f01):  {f0_fit:.6f} GHz")
    print(f"Linewidth (FWHM): {fwhm*1000:.3f} MHz")
    print(f"T2* Calculation:  {t2_star:.2f} ns")
    print("="*40)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mags, 'b.', label=f"Data ({spec.drive_power} dBm)", alpha=0.6)
    plt.plot(freqs, lorentzian(freqs, *popt), 'r-', label="Lorentzian Fit")
    plt.title(f"Qubit Spectroscopy: $T_2^*$ = {t2_star:.1f} ns")
    plt.xlabel("Drive Frequency (GHz)")
    plt.ylabel("|S21|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

except Exception as e:
    print(f"Fitting Fail ho gayi: {e}")
    #if fit fail
    plt.plot(freqs, mags)
    plt.show()