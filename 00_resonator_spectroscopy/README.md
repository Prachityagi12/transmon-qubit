# Resonator Spectroscopy (Circuit QED)

## Overview
This folder contains simulations and analysis of resonator spectroscopy in a circuit QED system. The main objective is to study the response of a superconducting resonator under different drive conditions and extract key physical parameters such as resonance frequency, linewidth, and quality factor.

The analysis also explores the dispersive regime, where the presence of a qubit leads to measurable shifts in the resonator frequency.

---

## What This Folder Contains

### 1. Simulation Codes
- `vna.py`: Simulates Vector Network Analyzer response (S21 transmission) of the resonator.
- `bare_resonator.py`: Models the resonator without qubit coupling to extract baseline behavior.
- `dispersive_regime.py` (if present): Simulates qubit–resonator interaction in the dispersive regime.

---

### 2. Analysis Notebooks
- `Linear_Plot.ipynb`: Linear representation of resonator response and peak analysis.
- `Polar_Plot.ipynb`: Polar representation showing amplitude and phase (resonance circle).
- `Adaptive_Test_Code.ipynb`: Adaptive methods for improved visualization and analysis.

---

### 3. Plots and Results
Generated outputs include:
- Resonance peak vs frequency plots
- Linewidth and quality factor extraction
- Polar resonance circle visualization
- Photon-number splitting (in dispersive regime)

---

## Key Concepts Studied
- Resonance frequency extraction  
- Linewidth and quality factor estimation  
- VNA-based transmission spectroscopy  
- Bare resonator baseline analysis  
- Dispersive frequency shift due to qubit coupling  
- Amplitude and phase response in polar form  

---

## Physical Insight
In the dispersive regime, the qubit shifts the resonator frequency without energy exchange. This allows indirect measurement of qubit state via resonator response.

Baseline behavior refers to the intrinsic response of the resonator in the absence of qubit coupling or external perturbations.

---

## Tools Used
- Python  
- NumPy / SciPy  
- Matplotlib  
- Jupyter Notebook  
- QuTip

---

## Summary
This folder focuses on resonator-based measurement techniques used in circuit QED systems and forms the basis for qubit state readout and further quantum characterization.
