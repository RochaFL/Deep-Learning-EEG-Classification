# Deep-Learning-EEG-Classification

# Deep Learning Pipeline for EEG Signal Classification (BCI)

This repository contains the complete implementation of my Bachelor's Thesis in Computer Science, focusing on a **hybrid Deep Learning and classical Machine Learning pipeline** for 4-class Motor Imagery (MI) classification using EEG signals from the [BCI Competition IV 2a Dataset](https://www.google.com/search?q=http://www.bbci.de/competition/iv/desc_2a.pdf).

##  Overview

The primary goal of this work was to develop a robust, open-source software pipeline capable of performing reliable Motor Imagery classification. The project addresses critical challenges in biological signal processing, including low Signal-to-Noise Ratio (SNR), significant inter-subject variability, and the prevention of data leakage.

##  Key Technical Features

### 1. Robust Preprocessing (Anti-Data Leakage)

Strict adherence to scientific methodology, specifically targeting **data leakage**:

* **EOG Channel Removal:** Explicit exclusion of ocular artifacts to ensure the models learn brain activity rather than eye movements.
* **Advanced Filtering:** FIR band-pass filtering (8-30 Hz) targeting the Mu and Beta rhythms.
* **Spatial Filtering:** Application of Common Average Reference (CAR) to reduce background noise.
* **Baseline Correction & Cropping:** Window-based epoching with precise temporal cropping (1.0s to 4.0s) to focus on the MI-related signal.

### 2. Hybrid Architecture (Ensemble)

The pipeline compares and fuses two distinct approaches to maximize accuracy:

* **Classical Baseline:** Spatial feature extraction using **Common Spatial Patterns (CSP)** combined with **Linear Discriminant Analysis (LDA)**.
* **Deep Learning:** A compact, regularized implementation of the **EEGNet** architecture, utilizing 2D, Depthwise, and Separable convolutions to extract spatial and temporal features efficiently.
* **Ensemble Fusion:** An ensemble approach that averages probability outputs from both models, yielding a final global mean accuracy of **63.43%** (vs 25% chance level).

### 3. Training & Validation

* **Subject-Specific Strategy:** Implementation of individual training regimens for each subject to account for brain variability.
* **Statistical Rigor:** 5-Fold Stratified Cross-Validation for every subject.
* **Training Stability:** Custom callbacks including `ReduceLROnPlateau` and `EarlyStopping` with weight restoration to prevent overfitting and ensure optimal performance.

##  Performance Summary

| Model | Mean Accuracy (%) |
| --- | --- |
| **EEGNet (Deep Learning)** | 46.41% |
| **CSP + LDA (Classical)** | 62.85% |
| **Hybrid (Ensemble)** | **63.43%** |

*Note: Peak performance reached **82.28%** for individual subjects.*

##  Technical Stack

* **Languages:** Python 3.10
* **Signal Processing:** `MNE-Python`
* **Deep Learning:** `TensorFlow / Keras`
* **Machine Learning:** `Scikit-learn`
* **Math & Stats:** `NumPy`

##  Repository Structure

* `pipeline_completo.py`: Main script for processing, training, and evaluation.
* `resultados_pipeline_fix.txt`: Detailed transcription of results, confusion matrices, and classification reports.
* `FABIO_LOPES_ROCHA_V03.pdf`: Full Bachelor's Thesis monography for academic reference.
* `dados/`: Directory for the BCI Competition IV 2a `.gdf` files (expected).

##  How to Run

1. Install dependencies:
`pip install mne scikit-learn tensorflow numpy`
2. Place your BCI Competition IV 2a dataset files (.gdf) in a folder named `dados/`.
3. Execute the pipeline:
`python pipeline_completo.py`

---

*This repository is part of a Bachelor's Thesis in Computer Science at UVA (2025).*
