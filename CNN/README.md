# CNN Model — Window-Level Spindle Detector

This folder contains a deep learning baseline model for sleep spindle detection based on a 2D Convolutional Neural Network (CNN).  
It operates on fixed-length EEG windows and serves as a learning-based baseline for comparison with graph-based and time-point-level models.

---

# Overview

- Operates on multi-channel EEG windows  
- Supervised learning (trained with labeled spindle data)  
- Produces one probability per window  
- Uses F1-optimized thresholding for final decisions  
- Provides a strong baseline between rule-based and graph models  

---

# Method Summary

The CNN pipeline follows these steps.

## EEG Preprocessing

- Band-pass filtering (configurable)  
- Re-referencing (e.g. CAR / average)  
- Channel selection  

## Windowing

- Fixed-length windows ( 2seconds)  
- Overlapping windows (1 step size)  
- Labels derived from spindle overlap ratio  

## CNN Inference

- 2D CNN over (channels × time) representation  
- Convolution + pooling blocks  
- Fully connected classifier  
- Sigmoid output → probability per window  

## Threshold Optimization

- Threshold selected on validation set  
- Maximizes F1 score  
- Frozen threshold applied to test set  

---



