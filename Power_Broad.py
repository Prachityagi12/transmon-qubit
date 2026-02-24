# for power broadening and to see the results  qubit freq linewidth and T2*
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
try: from tqdm import tqdm
except ImportError: tqdm = lambda x, **k: x

from architectures.qubit_resonator import QubitResonator

# Lorentzian Fit Function
def lorentzian(f, A, f0, gamma, offset):
    return A * (gamma**2 / ((f - f0)**2 + gamma**2)) + offset

# VirtualTwoTone Class
class VirtualTwoTone:
    def __init__(self):
        self.readout_freq = 6.0125
        self.readout_power = -100
        self.drive_power = -100
        self.start = 4.5
        self.stop = 5.5
        self.points = 401
        self.freqs = None
        self.mags = None

    def set_readout(self, freq_ghz, power_dbm):
        self.readout_freq = freq_ghz
        self.readout_power = power_dbm

    def set_drive_sweep(self, start_ghz, stop_ghz, points):
        self.start = start_ghz
        self.stop = stop_ghz
        self.points = points

    def set_drive_power(self, power_dbm):
        self.drive_power = power_dbm

    def scan(self, dut):
        self.freqs = np.linspace(self.start, self.stop, self.points)
        self.mags = []
        for f_drive in self.freqs:
            s21, _ = dut.response(
                readout_freq_ghz=self.readout_freq,
                readout_dbm=self.readout_power,
                qubit_drive_freq_ghz=f_drive,
                qubit_drive_dbm=self.drive_power
            )
            self.mags.append(np.abs(s21))
        self.mags = np.array(self.mags)

# Execution 
dut = QubitResonator(wc_ghz=6.0, wq_ghz=5.2, g_mhz=100, t1_ns=1000, t2_ns=1000)
spec = VirtualTwoTone()

readout_freq = 6.0125 # 
spec.set_readout(freq_ghz=readout_freq, power_dbm=-100)
spec.set_drive_sweep(start_ghz=5.0, stop_ghz=5.5, points=401)

drive_powers = range(-130, -60, 5)
powers_list = []
linewidths_list = []

# Terminal Header
print("\n" + "="*70)
print(f"{'Power (dBm)':<12} | {'Freq (GHz)':<12} | {'FWHM (MHz)':<12}")
print("-" * 70)

plt.figure(figsize=(12, 7))

# MAIN LOOP: 
for p in tqdm(drive_powers, desc="Power Sweep Running"):
    spec.set_drive_power(power_dbm=p)
    spec.scan(dut=dut)
    
    # Spectroscopy Plot
    plt.plot(spec.freqs, spec.mags, linewidth=1.5, label=f"Drive {p} dBm")
    
    # Fitting & Extraction
    y_data = spec.mags
    x_data = spec.freqs
    p0 = [np.max(y_data)-np.min(y_data), x_data[np.argmax(y_data)], 0.001, np.min(y_data)]
    
    try:
        popt, _ = curve_fit(lorentzian, x_data, y_data, p0=p0)
        f0_fit = popt[1]
        fwhm_ghz = 2 * abs(popt[2])
        fwhm_mhz = fwhm_ghz * 1000
        t2_star_ns = 1 / (np.pi * fwhm_ghz) #
        
        powers_list.append(p)
        linewidths_list.append(fwhm_mhz)
        
        # to show the result after every power
        print(f"{p:<12} | {f0_fit:<12.6f} | {fwhm_mhz:<12.3f}")
        
    except:
        print(f"{p:<12} | Fit Failed")

print("="*70)

# Plots 
plt.title(f"Power Broadening Spectroscopy")
plt.xlabel("Qubit Drive Frequency (GHz)")
plt.ylabel("|S21| Magnitude")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# PLOT for Broadening
plt.figure(figsize=(10, 5))
plt.plot(powers_list, linewidths_list, 'ro-', linewidth=2)
plt.title("Step-4: Linewidth vs Drive Power")
plt.xlabel("Drive Power (dBm)")
plt.ylabel("Linewidth (MHz)")
plt.grid(True)
plt.show()

