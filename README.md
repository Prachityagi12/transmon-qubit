# Transmon Qubit & Resonator Spectroscopy (Circuit QED)

## Overview
This project simulates a transmon qubit coupled to a readout resonator in the circuit QED framework. The goal is to study resonator spectroscopy and extract key physical parameters such as resonance frequency, linewidth, and quality factor.
The simulations also explore the dispersive regime, where the qubit-resonator interaction leads to measurable frequency shifts and photon-number splitting.
This project extends beyond resonator spectroscopy to full qubit characterization, including linewidth analysis, power-dependent effects, and photon-number-resolved behavior.

---

## What This Project Covers
- Simulation of resonator response under varying drive conditions  
- Linear and polar representation of resonator spectroscopy data  
- Analysis of dispersive regime behavior  

- Frequency-domain qubit spectroscopy using dispersive readout  
- Linewidth analysis and extraction of coherence time \( T_2^* \)  
- Power-dependent effects including power broadening and dressed-state formation  

- Observation of photon-number splitting in the strong dispersive regime  
- Extraction of key physical parameters from simulated data  

## Key Insights
- Clear resonance peaks observed in resonator spectroscopy with accurate extraction of frequency, linewidth, and quality factor  
- Frequency shift in dispersive regime consistent with theoretical expectations  
- Photon-number splitting observed, confirming strong dispersive coupling  

- Qubit transition frequency accurately extracted from low-power spectroscopy  
- Linewidth analysis used to estimate coherence time \( T_2^* \)  
- Power-dependent broadening observed, showing transition from linear response to saturation  

- Emergence of dressed-state splitting at intermediate drive powers  
- Optimal drive regime identified to balance signal strength and minimal decoherence  

- Polar plots provide improved visualization of amplitude and phase compared to linear plots

---

## Repository Structure
    
### Resonator Spectroscopy
This folder contains simulation and analysis of resonator spectroscopy in circuit QED systems. The data, plots, and code included here correspond to studying the resonator response and extracting physical parameters.

### Folders
- **Linear_Plot**:  
  Contains Jupyter notebooks for linear representation of resonator spectroscopy. Includes code and plots showing readout frequency vs. signal response in linear scale.
- **Polar_Plot**:  
  Contains Jupyter notebooks for polar representation of resonator spectroscopy. Visualizes amplitude and phase in polar coordinates.
- **Adaptive_Plot**:  
  Contains code and results for adaptive plotting techniques used to better visualize spectroscopy data.
- **resonator_spect_dispersive_regime**:  
   Analysis of dispersive regime and parameter extraction
  
### Files
- **photon_no_plot.png**: Photon-number splitting plot generated from simulation data.  
- **readout_freq_vs_power_plot.png**: Readout frequency vs. readout power plot (Dispersive regime).

## Analysis
- Simulated resonator spectroscopy and fitting to extract:
  - Resonance frequency
  - Linewidth
  - Quality factor
- Plots in linear and polar representations
- Dispersive regime analysis and photon-number splitting investigation

---

## Qubit Spectroscopy

This section focuses on frequency-domain characterization of the transmon qubit using dispersive readout.

### Methodology
- System operated in the **dispersive regime** (large detuning)  
- Readout power optimized (~ -100 dBm) for linear response  
- Qubit spectroscopy performed by sweeping drive frequency

---

### Low-Power Spectroscopy
- Single Lorentzian peak observed  
- Qubit transition frequency extracted  
- Linewidth used to estimate coherence time \( T_2^* \)  

---

### Photon-Number Splitting
- Multiple peaks appear at higher readout power (~ -90 dBm)  
- Frequency shift of ~ \( 2\chi \) per photon  
- Confirms strong dispersive regime  

---

### Power Broadening
- Linewidth increases with qubit drive power  
- Shows transition from sharp to broadened peaks  

---

### Dressed-State Formation
- Peak splitting observed at intermediate powers  
- Indicates formation of dressed states  
- At high power, peaks merge due to strong broadening  

---

### Optimal Drive Power
- Linewidth constant in low-power regime  
- Increases beyond threshold due to power broadening  
- Optimal ≈ **-120 dBm**  

---

### Key Outcome
- Accurate extraction of qubit frequency and linewidth  
- Identification of optimal operating regimes  
- Observation of photon-number splitting and dressed-state physics  

## Tools & Technologies
- Python  
- NumPy / SciPy  
- Matplotlib  
- Jupyter Notebook

---

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

---

## Note
- This work is inspired by experimental circuit QED setups and aims to bridge theoretical simulation with realistic superconducting qubit behavior.
- All simulations were performed using Python scripts and Jupyter notebooks.
- Each folder contains corresponding `.ipynb` files with code and resulting plots.
