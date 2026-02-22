 # Baseline Model — Schimicek Spindle  Detector

This folder contains a **rule-based baseline model** for sleep spindle detection, inspired by the classical method of **Schimicek et al.**  
It is used as a **non-learning reference** for comparison with deep learning and graph-based models.

---

# Overview

- Operates on **single EEG channels**
- Fully **deterministic** (no training required)
- Based on classical EEG **signal-processing rules**
- Provides a **lower-bound benchmark** for model comparison

---

Method Summary

The detector follows four main steps:

1. **Band-pass filtering**
   - Spindle: 11.5–16 Hz  
   - Alpha: 5–12 Hz  
   - Muscle: 30–40 Hz  

2. **Candidate spindle detection**
   - Peak-to-peak amplitude threshold (µV)
   - Minimum duration constraint  

3. **Artifact rejection** (5-second epochs)
   - Alpha artifact: RMS ratio
   - Muscle artifact: RMS threshold  

4. **Final detection**
   - Artifact-free spindle mask
   - Event-level spindle extraction  

---

##  Files



schimicek_baseline/
├── schimicek_spindle.py # Core detection logic
├── all_data.py # Execution & evaluation
├── config.yaml # Default parameters
├── sweep.yaml # W&B parameter sweeps
├── run_command.txt
└── README.md


---




# Reference

Schimicek et al.
Automatic sleep spindle detection using EEG signal processing techniques.
