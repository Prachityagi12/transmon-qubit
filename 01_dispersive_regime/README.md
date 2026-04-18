
# Dispersive Regime (Circuit QED)



## Overview
This folder contains simulations and analysis of the dispersive regime in a circuit QED system, where a superconducting qubit is coupled to a resonator but operates in the off-resonant limit
In this regime, there is no direct energy exchange between the qubit and the resonator. Instead, the interaction leads to a qubit-state-dependent shift in the resonator frequency, enabling indirect qubit readout.

---

## What This Folder Contains

### 1. Dispersive Regime Simulation

- Modeling of qubit–resonator interaction in the dispersive limit  
- Calculation of frequency shift as a function of system parameters  

---

### 2. Qubit–Resonator Interaction
- Study of how qubit state modifies resonator response  

- Basis for dispersive readout in superconducting qubits  

---

## Plots and Results

### Resonator Frequency vs Readout Power

[Resonator Frequency](plots/Resonator\_fre\_vs\_readout\_pow.png)
Shows how the resonator frequency shifts with increasing readout power due to nonlinear effects.

---

### Photon Number vs Readout Power

!\[Photon Number](plots/Photon\_number\_vs\_readout\_pow.png)
Demonstrates the increase in intracavity photon number as a function of applied drive power.

---

### Photon Number Splitting (-90 dBm)

!\[Photon Splitting -90dBm](plots/Photon\_num\_splitting(-90dbm).png)

---

### Photon Number Splitting (-100 dBm)

!\[Photon Splitting -100dBm](plots/Photon\_number\_splitting(-100dbm).png)
At different readout powers, the resonator spectrum splits into multiple peaks corresponding to different photon number states.

---

## Key Concepts Studied
- Dispersive coupling  
- Qubit-state-dependent frequency shift  
- Photon number in resonator  
- Readout power dependence  
- Photon-number splitting  

---

## Physical Insight
In the dispersive regime, the interaction Hamiltonian leads to a shift in the resonator frequency proportional to the qubit state.
This enables measurement of the qubit without directly exciting it, forming the basis of quantum non-demolition (QND) readout.
At higher photon numbers, nonlinear effects become significant, leading to observable photon-number splitting in the spectrum.

---

## Summary
This folder explores the dispersive interaction between a qubit and resonator, highlighting how measurement in circuit QED systems is performed indirectly through frequency shifts and photon statistics.

