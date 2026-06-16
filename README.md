# Transmon Qubit & Resonator Spectroscopy (Circuit QED)

## Overview

This project simulates a superconducting transmon qubit coupled to a readout resonator within the circuit QED framework.

The primary objective is to investigate resonator and qubit spectroscopy and extract key physical parameters, including:

* Resonator resonance frequency
* Resonator linewidth
* Resonator quality factor
* Optimal readout power
* Qubit transition frequency
* Optimal drive power
* Transmon anharmonicity

The project further explores qubit–resonator interactions in the dispersive regime, photon-number-resolved behavior, power-dependent effects, and coherence characterization. The simulations reproduce several important phenomena observed in superconducting qubit experiments, including dispersive frequency shifts, photon-number splitting, power broadening, dressed-state formation, and anharmonicity extraction.

By combining resonator spectroscopy, dispersive analysis, qubit spectroscopy, anharmonicity measurements, and time-domain characterization, this repository aims to bridge theoretical modeling with realistic experimental techniques used in superconducting quantum computing platforms.

The simulations are based on numerical solutions of open quantum system dynamics using the Lindblad master equation and QuTiP.

---

## Project Structure

```text
transmon-qubit/
│
├── 00_resonator_spectroscopy/
├── 01_dispersive_regime/
├── 02_qubit_spectroscopy/
├── 03_Anharmonicity/
├── 04_Time_Domain_Characterization/
│
├── Plots/
├── Thesis/
├── Presentation/
└── README.md
```

---

## What This Project Covers

### Resonator Spectroscopy

* Bare resonator characterization
* Resonance frequency extraction
* Linewidth estimation
* Quality factor calculation
* Linear and polar visualization of spectroscopy data

### Dispersive Regime Analysis

* Qubit–resonator interaction
* Dispersive frequency shifts
* Photon number estimation
* Photon-number splitting
* Strong dispersive coupling effects

### Qubit Spectroscopy

* Two-tone spectroscopy
* Qubit transition frequency extraction
* Linewidth analysis
* Estimation of coherence time (T₂*)
* Power broadening analysis
* Dressed-state formation

### Anharmonicity Extraction

* Continuous frequency sweep spectroscopy
* Adaptive spectroscopy techniques
* Identification of higher transmon transitions
* Extraction of transmon anharmonicity
* Comparison of spectroscopy strategies

### Time-Domain Characterization

* Rabi oscillations
* Ramsey interference
* Energy relaxation measurements (T₁)
* Coherence measurements (T₂)
* Qubit control and calibration techniques

---

## Key Insights

* Accurate extraction of resonator frequency, linewidth, and quality factor
* Observation of dispersive shifts consistent with circuit QED theory
* Clear photon-number splitting in the strong dispersive regime
* Accurate determination of qubit transition frequencies
* Estimation of coherence times from linewidth measurements
* Observation of power broadening and saturation effects
* Identification of optimal spectroscopy power regimes
* Extraction of transmon anharmonicity using multiple spectroscopy methods
* Improved visualization through linear and polar representations

---

## Physical Workflow

The repository follows a realistic superconducting qubit measurement pipeline:

```text
Resonator Spectroscopy
          ↓
Dispersive Regime Analysis
          ↓
Qubit Spectroscopy
          ↓
Anharmonicity Extraction
          ↓
Time-Domain Characterization
```

This mirrors the workflow commonly used in experimental circuit QED systems.

---

## Repository Details

### 00_resonator_spectroscopy

Study of the bare resonator response:

* Resonance frequency extraction
* Linewidth estimation
* Quality factor calculation
* Linear and polar spectroscopy plots

### 01_dispersive_regime

Analysis of qubit–resonator interaction:

* Dispersive frequency shifts
* Photon number estimation
* Photon-number splitting

### 02_qubit_spectroscopy

Frequency-domain characterization of the qubit:

* Two-tone spectroscopy
* Linewidth analysis
* Coherence estimation
* Power broadening studies

### 03_Anharmonicity

Extraction of transmon anharmonicity:

* Brute-force spectroscopy sweeps
* Adaptive spectroscopy sweeps
* Comparison of extraction methods
* Anharmonicity figures and analysis notebooks

### 04_Time_Domain_Characterization

Time-domain measurements and coherence studies:

* Rabi oscillations
* Ramsey experiments
* Relaxation and dephasing analysis

### Plots

Collection of simulation and analysis figures generated throughout the project.

### Thesis

Complete M.Tech thesis:

**Simulation and Spectroscopic Analysis of a Superconducting Transmon Qubit Coupled to a Microwave Resonator**

### Presentation

Final project presentation summarizing methodology, simulations, and results.

---

## Tools & Technologies

* Python
* NumPy
* SciPy
* Matplotlib
* Jupyter Notebook
* QuTiP (Quantum Toolbox in Python)

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/Prachityagi12/transmon-qubit.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks or Python scripts contained in the corresponding folders.

---

## Academic Context

This work was carried out as part of an M.Tech project on superconducting quantum circuits and circuit QED.

The project aims to bridge theoretical simulation with realistic experimental techniques used in transmon-based quantum computing platforms.

---

## References

This work is inspired by spectroscopic measurements performed in superconducting circuit QED systems.

Primary reference:

B. Suri, *Spectroscopic Measurements of Transmon–Resonator cQED Devices*, Ph.D. Thesis, 2015.


