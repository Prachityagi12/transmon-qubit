# #Create Object
# vna = VirtualVNA()
# qubit_resonator = QubitResonator(
#     wc_ghz=6.0,
#     wq_ghz=5.2,
#     g_mhz=100,
#     t1_ns=1000,
#     t2_ns=1000
# )

import numpy as np
import matplotlib.pyplot as plt
from instruments.vna import VirtualVNA
from architectures.qubit_resonator import QubitResonator

# 1. Setup
vna = VirtualVNA()
# wc=6.0, wq=5.2, g=100MHz (0.1 GHz)
qubit_resonator = QubitResonator(wc_ghz=6.0, wq_ghz=5.2, g_mhz=100)

# Precision Settings
vna.set_ifbw(100)
vna.set_averages(300)
vna.set_sweep(start_ghz=5.9, stop_ghz=6.2, points=1001) # Zoomed in near wc=6.0

# Power Range: -90 and -110 to see the transition 
powers = [-140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40]
fr_list = []
photon_list = []

print(f"{'Power (dBm)':<12} | {'fr (GHz)':<12} | {'n_avg':<10}")
print("-" * 45)

for p in powers:
    vna.set_power(p)
    # DUT scan
    freqs, s21 = vna.scan(dut=qubit_resonator)
    mag = np.abs(s21)
    
    # Finding resonance frequency (minima of S21)
    fr_curr = freqs[np.argmin(mag)]
    fr_list.append(fr_curr)
    
    # --- Corrected Photon Calculation ---
    # convert dbm to watts scaling factor(Linear Relationship)
    p_watts = 10**(p/10) * 0.001 
    
    # n_avg calculation (Using a realistic scaling for -90dBm ~ 1 photon)
    # p = -90 dBm -> 1e-12 Watts. Scaling factor 1e12 so that n=1
    n_avg = p_watts * 1e12 
    photon_list.append(n_avg)
    
    print(f"{p:<12.1f} | {fr_curr:<12.6f} | {n_avg:<10.3f}")

# -------- PLOTTING SECTION --------

# Plot 1: Resonator Frequency Shift
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(powers, fr_list, 'o-', color='#1f77b4', markersize=6)
plt.xlabel("Readout Power (dBm)")
plt.ylabel("Resonator Frequency (GHz)")
plt.title("Resonator Frequency vs Power")
plt.grid(True, alpha=0.3)

# Plot 2: Photon Number 
plt.subplot(1, 2, 2)
plt.plot(powers, photon_list, '^-', color='#d62728', markersize=6)
plt.yscale('log') # Log scale is best to see n=0.1 to n=100 clearly
plt.xlabel("Readout Power (dBm)")
plt.ylabel("Average Photon Number (n)")
plt.title("Photon Number vs Power (Log Scale)")
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Critical Photon Calculation (n_crit)
delta = abs(qubit_resonator.wq - qubit_resonator.wc) # 5.2 - 6.0 = 0.8 GHz
g = qubit_resonator.g / 1000 # 100MHz = 0.1 GHz
n_crit = (delta**2) / (4 * g**2) # (0.8^2) / (4 * 0.1^2) = 0.64 / 0.04 = 16

print(f"\n" + "="*40)
print(f"ANALYSIS SUMMARY:")
print(f"Critical Photon Limit (n_crit) ≈ {n_crit}")
print(f"At -90 dBm, n ≈ {photon_list[5]:.2f}")
print(f"Result: Power < -90 dBm is Safe Dispersive Regime (n < 1).")
print("="*40)