# Transmon Qubit & Resonator Spectroscopy (Circuit QED)

## Project Structure

- 00_resonator_spectroscopy → Bare resonator analysis  
- 01_dispersive_regime → Qubit–resonator interaction  
- 02_qubit_spectroscopy → Qubit characterization  

---

## Overview
This project simulates a transmon qubit coupled to a readout resonator in the circuit QED framework. The goal is to study resonator spectroscopy and extract key physical parameters such as resonance frequency, linewidth, and quality factor.

The simulations further explore the dispersive regime, where qubit–resonator interaction leads to measurable frequency shifts and photon-number splitting. The project extends to full qubit characterization, including linewidth analysis, power-dependent effects, and photon-number-resolved behavior.

---

## What This Project Covers
This project bridges theoretical modeling with realistic superconducting qubit measurement techniques.

- Simulation of resonator response under varying drive conditions  
- Linear and polar representation of resonator spectroscopy data  
- Analysis of dispersive regime behavior
- Observation of photon-number splitting in the strong dispersive regime   

- Frequency-domain qubit spectroscopy using dispersive readout  
- Linewidth analysis and extraction of coherence time \( T_2^* \)  
- Power-dependent effects including power broadening and dressed-state formation  

- Extraction of key physical parameters from simulated data  

---

## Key Insights
- Clear resonance peaks observed with accurate extraction of frequency, linewidth, and quality factor  
- Frequency shift in the dispersive regime consistent with theoretical expectations  
- Photon-number splitting observed, confirming strong dispersive coupling  

- Qubit transition frequency accurately extracted from low-power spectroscopy  
- Linewidth analysis used to estimate coherence time \( T_2^* \)  
- Power-dependent broadening observed, showing transition from linear response to saturation  

- Emergence of dressed-state splitting at intermediate drive powers  
- Optimal drive regime identified to balance signal strength and minimal decoherence  

- Polar plots provide improved visualization of amplitude and phase compared to linear plots  

---

## Physical Workflow

This project follows the actual experimental workflow used in circuit QED systems:

1. **Resonator Spectroscopy** → Characterize bare resonator properties  
2. **Dispersive Regime** → Study qubit–resonator interaction  
3. **Qubit Spectroscopy** → Extract qubit parameters and coherence properties  

This structured approach mirrors real superconducting qubit measurement pipelines.

---

## Repository Structure Details

### 00_resonator_spectroscopy
Study of the bare resonator response, including:
- Resonance frequency extraction  
- Linewidth and quality factor estimation  
- Linear and polar visualization of spectroscopy data  

---

### 01_dispersive_regime
Analysis of qubit–resonator interaction in the dispersive limit:
- Qubit-state-dependent frequency shift  
- Photon number vs readout power  
- Photon-number splitting in strong dispersive regime  

**Key Outcome**
- Observation of photon-number splitting and dispersive readout physics  

---

### 02_qubit_spectroscopy
Frequency-domain characterization of the qubit:
- Two-tone spectroscopy for transition frequency extraction  
- Linewidth analysis and coherence time \( T_2^* \) estimation  
- Power broadening and dressed-state formation  
- Optimal drive power identification  

**Key Outcomes**
- Accurate extraction of qubit frequency and linewidth  
- Identification of optimal operating regimes  

---

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

## Note
- This work is inspired by experimental circuit QED setups and aims to bridge theoretical simulation with realistic superconducting qubit behavior.
- All simulations were performed using Python scripts and Jupyter notebooks.
- The `00_resonator_spectroscopy` folder contains Jupyter notebooks (`.ipynb`) with code and resulting plots for analysis and visualization.
- The `01_dispersive_regime` and `02_qubit_spectroscopy` folders primarily contain Python scripts (`.py`) for simulation and data generation.
