\# Qubit Spectroscopy (Circuit QED)



\## Overview

This folder contains simulation and analysis of frequency-domain spectroscopy of a superconducting transmon qubit in a circuit QED system.



The main objective is to characterize the qubit by extracting its transition frequency, linewidth, and coherence time (T₂\*) using microwave drive-based spectroscopy techniques.



\---



\## What This Folder Contains



\### 1. Two-Tone Spectroscopy

\- Measurement of qubit transition frequency using a two-tone method  

\- Identification of resonance peak corresponding to |g⟩ → |e⟩ transition  



\---



\### 2. Power Broadening Analysis

\- Study of linewidth variation with increasing drive power  

\- Observation of transition from sharp Lorentzian peak to broadened response  

\- Demonstrates decoherence effects under strong driving  



\---



\### 3. Optimal Drive Power

\- Determination of optimal operating power for spectroscopy  

\- Trade-off between signal strength and minimal spectral distortion  

\- Selection of regime for accurate qubit characterization  



\---



\### 4. Linewidth vs Drive Power

\- Systematic study of spectral linewidth as a function of drive amplitude  

\- Identification of linear and nonlinear response regimes  



\---



\### 5. Coherence Time Extraction (T₂\*)

\- Extraction of coherence time from spectral linewidth  

\- Relation between linewidth and dephasing time  



\---



\## Key Concepts Studied

\- Two-tone spectroscopy  

\- Qubit transition frequency extraction  

\- Linewidth analysis  

\- Power broadening effects  

\- Optimal drive regime selection  

\- Coherence time (T₂\*) estimation  



\---



\## Physical Insight

Qubit spectroscopy relies on applying a microwave drive to induce transitions between energy levels of the transmon qubit. The resulting spectral response reveals the qubit’s transition frequency and coherence properties.



At higher drive powers, nonlinear effects such as power broadening distort the spectral lineshape, requiring careful selection of optimal operating conditions.



\---



\## Tools Used

\- Python  

\- NumPy / SciPy  

\- Matplotlib  

\- Jupyter Notebook  



\---



\## Summary

This folder focuses on the frequency-domain characterization of a superconducting qubit using spectroscopy techniques. It captures both linear-response behavior and power-dependent nonlinear effects, enabling accurate extraction of qubit parameters.

