import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from architectures.qubit_resonator import QubitResonator

# Lorentzian Model (Peak form)
def lorentzian(f, A, f0, gamma, offset):
    return A * (gamma**2 / ((f - f0)**2 + gamma**2)) + offset

# Device Setup
dut = QubitResonator(
    wc_ghz=6.0,
    wq_ghz=5.2,
    g_mhz=10,
    t1_ns=1000,
    t2_ns=1000
)

class VirtualTwoTone:
    def __init__(self):
        self.readout_freq = 6.000125
        self.readout_power = -100
        self.drive_power = -130
        self.start = 5.19
        self.stop = 5.21
        self.points = 401

    def set_drive_power(self, power_dbm):
        self.drive_power = power_dbm

    def scan(self, dut):
        self.freqs = np.linspace(self.start, self.stop, self.points)
        self.mags = [
            np.abs(dut.response(
                self.readout_freq,
                self.readout_power,
                f,
                self.drive_power
            )[0])
            for f in self.freqs
        ]

spec = VirtualTwoTone()

# Extract Peak Height and Linewidth
def extract_parameters(dut, spec, power):
    spec.set_drive_power(power)
    spec.scan(dut)
    popt, _ = curve_fit(
        lorentzian,
        spec.freqs,
        spec.mags,
        p0=[0.1, 5.2, 0.001, 0.99]
    )
    peak_height = popt[0]
    linewidth_mhz = 2 * abs(popt[2]) * 1000  # FWHM in MHz

    return peak_height, linewidth_mhz

# Binary Search for Optimal Power (Linewidth Based)
def find_optimal_power(dut, spec, low_p, high_p, threshold_factor=1.2, tolerance=0.5):

    # Natural linewidth at lowest power
    _, gamma0 = extract_parameters(dut, spec, low_p)
    gamma_limit = gamma0 * threshold_factor
    print(f"Natural Linewidth: {gamma0:.4f} MHz")
    print(f"Allowed Limit: {gamma_limit:.4f} MHz\n")

    optimal_p = low_p

    while (high_p - low_p) > tolerance:
        mid = (low_p + high_p) / 2
        _, gamma_mid = extract_parameters(dut, spec, mid)

        if gamma_mid < gamma_limit:
            optimal_p = mid
            low_p = mid
        else:
            high_p = mid

    return optimal_p, gamma0

# Full Sweep + Plotting

def run_analysis(dut, spec):
    sweep_powers = np.arange(-130, -60, 5)
    powers = []
    heights = []
    linewidths = []

    print(f"{'Power(dBm)':<12} | {'Peak Height':<15} | {'Linewidth (MHz)':<15}")
    print("-" * 50)

    for p in sweep_powers:
        h, lw = extract_parameters(dut, spec, p)
        powers.append(p)
        heights.append(h)
        linewidths.append(lw)

        print(f"{p:<12.2f} | {h:<15.6f} | {lw:<15.4f}")

    # Binary search
    optimal_power, baseline_gamma = find_optimal_power(dut, spec, -130, -60)
    #PLOTS 
    # Plot 1: Peak Height vs Power
    plt.figure(figsize=(7,5))
    plt.plot(powers, heights, 'o-')
    plt.axvline(optimal_power, linestyle='--')
    plt.title("Peak Height vs Drive Power")
    plt.xlabel("Drive Power (dBm)")
    plt.ylabel("Peak Height (A)")
    plt.grid(True)
    plt.show()

    # Plot 2: Linewidth vs Power
    plt.figure(figsize=(7,5))
    plt.plot(powers, linewidths, 's-')
    plt.axvline(optimal_power, linestyle='--')
    plt.axhline(baseline_gamma, linestyle=':')
    plt.title("Linewidth (FWHM) vs Drive Power")
    plt.xlabel("Drive Power (dBm)")
    plt.ylabel("FWHM (MHz)")
    plt.grid(True)
    plt.show()

    print("\n-----------------------------------")
    print(f"Optimal Drive Power = {optimal_power:.2f} dBm")
    print("-----------------------------------")

    return optimal_power

# 6. Run Everything
if __name__ == "__main__":
    optimal_power = run_analysis(dut, spec)
