## Qubit Spectroscopy (Circuit QED)

---

## Overview

This folder contains simulation and analysis of frequency-domain spectroscopy of a superconducting transmon qubit in a circuit QED system.

The main objective is to characterize the qubit by extracting its transition frequency, linewidth, and coherence time (T₂*) using microwave drive-based spectroscopy techniques.

---

## What This Folder Contains
### 1.Two-Tone Spectroscopy
- Measurement of qubit transition frequency using a two-tone method
- Identification of resonance peak corresponding to |g⟩ → |e⟩ transition
## Result:
Two Tone
The peak position gives the qubit transition frequency, while the linewidth provides information about decoherence.

### 2. Power Broadening Analysis
- Study of linewidth variation with increasing drive power
- Observation of transition from sharp Lorentzian peak to broadened response
- Demonstrates decoherence effects under strong driving
## Result (High Power):
Power Broadening High

## Result (g = 10, Q = 2000):
Increasing drive power leads to linewidth broadening, indicating saturation and reduced coherence.

### 3. Optimal Drive Power 
- Determination of optimal operating power for spectroscopy
- Trade-off between signal strength and minimal spectral distortion
- Selection of regime for accurate qubit characterization
## Insight:
Low power → sharp peaks (accurate but weak signal)
High power → broad peaks (strong signal but distorted)
Optimal region balances signal-to-noise ratio and minimal broadening.

### 4. Linewidth vs Drive Power
- Systematic study of spectral linewidth as a function of drive amplitude
- Identification of linear and nonlinear response regimes
## Results:
Linewidth remains constant at low power and increases at high power due to power broadening.

### 5. Coherence Time Extraction (T₂*)
- Extraction of coherence time from spectral linewidth
## Result:
Because Coherence time is inversely related to linewidth:
At Narrow peak → high T₂*
At Broad peak → low T₂*

---

## Key Concepts Studied
- Two-tone spectroscopy
- Qubit transition frequency extraction
- Linewidth analysis
- Power broadening effects
- Optimal drive regime selection
- Coherence time (T₂*) estimation

---

## Physical Insight
Qubit spectroscopy relies on applying a microwave drive to induce transitions between energy levels of the transmon qubit. The resulting spectral response reveals the qubit’s transition frequency and coherence properties.
At higher drive powers, nonlinear effects such as power broadening distort the spectral lineshape, requiring careful selection of optimal operating conditions.

---

## Summary
This folder demonstrates how frequency-domain spectroscopy can be used to extract key qubit parameters and analyze both linear and nonlinear regimes, providing a complete characterization of a superconducting qubit.
































