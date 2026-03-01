# UHF Partial Discharge Detection: Mathematical Frameworks & Implementation Roadmap

  

This document outlines the recent scholarly articles concerning UHF Partial Discharge (PD) detection, featuring functional hyperlinks to source materials. For each article, **Implementation Notes** and a **Mathematical Framework** are provided, detailing how the distinct methodologies and algorithms can be directly integrated into the `SDFDP` analytical application (leveraging existing modules like `preprocessing.py`, `blind_algorithms.py`, `descriptors.py`, and the Dash GUI).

  

---

  

## Part I: Recent Scholarly Articles (2024–2026)

  

### 1. [Research on GIS Partial Discharge Detection Based on UHF Sensors](https://www.researchgate.net/publication/400853962_Research_on_GIS_Partial_Discharge_Detection_Based_on_UHF_Sensors)

**Date:** February 19, 2026

**Summary:** Evaluates change laws of intermittent discharge in metal defects using combined pulse current and UHF method validation.

**Implementation Notes for App:**

We can implement a multi-sensor cross-correlation algorithm in `preprocessing.py` or `hardware_bridge.py` that synchronizes a pulse current reference signal with the UHF captured pulse.

**Mathematical Framework:**

The cross-correlation function $R_{xy}(\tau)$ determines the time-delay between the high-frequency UHF event $x(t)$ and the low-frequency pulse current $y(t)$:

$$ R_{xy}(\tau) = \int_{-\infty}^{\infty} x(t) y(t+\tau) dt $$

By identifying the peak of $R_{xy}(\tau)$, the app can auto-align and validate the PD emission source against grid-frequency pulse current.

  

### 2. [Classification and Recognition of UHF-PD Signals Based on Adaptive Hybrid Attention Fusion Network](https://www.mdpi.com/2076-3417/16/3/1479)

**Date:** February 02, 2026

**Summary:** Proposes a dual-flow network structure (ResNet + Swin Transformer) for local and global feature extraction (99.58% accuracy).

**Implementation Notes for App:**

While full deep learning training might exist outside `SDFDP`, we can integrate an inference API in `file_analysis.py` taking pre-processed Phase-Resolved Partial Discharge (PRPD) matrices as input. The self-attention weighting mechanism can also be mathematically abstracted as a severity feature.

**Mathematical Framework:**

The Swin Transformer's core operation relies on multi-head self-attention applied over windowed signal patches:

$$ Attention(Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V $$

Where $Q$ (Query), $K$ (Key), and $V$ (Value) are derived from the 2D spectrogram $X(\tau, \omega)$ of the UHF signal.

  

### 3. [Research on UAV Ultra-High Frequency Partial Discharge Detection](https://www.researchgate.net/publication/399377109_Research_on_UAV_Ultra-High_Frequency_Partial_Discharge_Detection)

**Date:** January 04, 2026

**Summary:** Uses peak envelope detection for online monitoring with high sensitivity and interference suppression.

**Implementation Notes for App:**

We can add an envelope detection pipeline in `preprocessing.py` and visualize it via `gui/time_series.py` using a Hilbert Transform to reduce the immense UHF sampling rate down to a manageable boundary curve.

**Mathematical Framework:**

The analytic signal $x_a(t)$ is derived using the Hilbert transform $\mathcal{H}\{x(t)\}$:

$$ x_a(t) = x(t) + j \mathcal{H}\{x(t)\} = x(t) + j \left( \frac{1}{\pi} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} d\tau \right) $$

The peak envelope $E(t)$ is the magnitude of the analytical signal:

$$ E(t) = \sqrt{x(t)^2 + \mathcal{H}\{x(t)\}^2} $$

  

### 4. [Advanced Signal Processing Methods for Partial Discharge Analysis: A Comprehensive Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12694323/)

**Date:** December 01, 2025

**Summary:** Compares Wavelet Transforms (WT), Hilbert-Huang Transforms (HHT), and AI methods for non-stationary noise.

**Implementation Notes for App:**

Using the `PyWavelets` package installed in requirements, we can implement Continuous Wavelet Transform (CWT) to generate scalograms for non-stationary PD characterization in `descriptors.py` and `gui/plot_utils.py`.

**Mathematical Framework:**

The CWT decomposes the signal $x(t)$ into time-scale space using a mother wavelet $\psi(t)$:

$$ CWT_x(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^* \left( \frac{t - b}{a} \right) dt $$

Where $a$ represents the scale (inverse of frequency) and $b$ denotes the time translation.

  

### 5. [Research on PD Signal Recognition and Localization Based on Multi-Scale Feature Fusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC12633952/)

**Date:** November 21, 2025

**Summary:** Synergistic integration of adaptive processing achieving 96.2% classification accuracy.

**Implementation Notes for App:**

We can implement a multi-scale descriptor mechanism in `descriptors.py`. Instead of using deep learning directly, we compute statistical descriptors (Variance, Skewness, Kurtosis) across different frequency bands (scales) and fuse them into a singular feature vector.

**Mathematical Framework:**

For multiple frequency scales $j = 1 \dots J$, we calculate band-specific energy limits $E_j$:

$$ E_j = \sum_{n} |X_j[n]|^2 $$

The fused multi-scale feature vector $F$ is a concatenated weighting:

$$ F_{fusion} = [w_1 E_1, w_2 E_2, \dots, w_J E_J] \quad \text{where } \sum w_j = 1 $$

  

### 6. [Research on a Degradation Identification Method for GIS UHF Sensors](https://www.mdpi.com/1424-8220/25/22/6860)

**Date:** November 10, 2025

**Summary:** Employs a sensitivity identification procedure to account for UHF sensor physical degradation over time.

**Implementation Notes for App:**

`threshold_config.py` and `severity.py` can be updated to include a "Sensor Age/Calibration Factor". This offsets false negatives caused by natural sensor sensitivity decay over operational years.

**Mathematical Framework:**

Adjusting the raw measured voltage $V_{raw}(t)$ using an exponential sensor degradation factor $\lambda$:

$$ V_{calibrated}(t) = V_{raw}(t) \times e^{\lambda (T_{current} - T_{install})} $$

Thresholds $\Gamma_{alert}$ geometrically tighten based on the same degradation constant to maintain dynamic reliability.

  

### 7. [UHF Signal Processing and Pattern Recognition of PD in GIS Using Chromatic Methodology](https://www.researchgate.net/publication/312542880_UHF_Signal_Processing_and_Pattern_Recognition_of_Partial_Discharge_in_Gas-Insulated_Switchgear_Using_Chromatic_Methodology)

**Date:** October 16, 2025

**Summary:** Uses "chromatic methodology" (borrowed from color science) mapping UHF into Hue, Lightness, and Saturation (HLS).

**Implementation Notes for App:**

We can implement a novel visualization tab in `app.py` taking signal spectral energy bands (Low, Mid, High) and mapping them to Red, Green, and Blue, then converting them to an HLS parameter space for `plot_utils.py`.

**Mathematical Framework:**

Let $E_L, E_M, E_H$ be energy integrals from low, medium, and high frequency sub-bands of the UHF spectrum. They represent color tristimulus values. The nominal Lightness $L$ and nominal Hue $H$ mapping angle $\theta$ is:

$$ L = \frac{E_L + E_M + E_H}{3} $$

$$ \theta = \cos^{-1} \left( \frac{ \frac{1}{2} ( (E_R - E_G) + (E_R - E_B) ) }{ \sqrt{ (E_R - E_G)^2 + (E_R - E_B)(E_G - E_B) } } \right) $$

  

### 8. [Design and Evaluation of an Integrated Ultra‐High Frequency and Optical Sensor](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/smt2.70006)

**Date:** April 16, 2025

**Summary:** Joint analysis of UHF and optical signals to improve PD detection reliability for defects like floating objects.

**Implementation Notes for App:**

Add supplementary channels in `hardware_bridge.py` allowing a boolean or analog optical sensor feed. A Bayesian fusion algorithm can reside in `validation.py` to confirm faults.

**Mathematical Framework:**

Using Bayes' rule to compute the posterior probability of a valid PD event $P(PD|U, O)$ given the joint UHF event $U$ and Optical event $O$:

$$ P(PD | U, O) = \frac{ P(U, O | PD) P(PD) }{ P(U, O | PD) P(PD) + P(U, O | \neg PD) P(\neg PD) } $$

  

### 9. [Research on State Diagnosis Methods of UHF PD Sensors Based on Improved ViT](https://www.mdpi.com/2076-3417/14/23/11214)

**Date:** December 02, 2024

**Summary:** Diagnoses sensor health states via a Vision Transformer (ViT) equipped with a sliding window applied to 2D representations.

**Implementation Notes for App:**

Convert the 1D UHF signals into 2D time-frequency grids using Short-Time Fourier Transform (STFT) inside `preprocessing.py`. This serves as the precise structured input required if the user decides to run external ViT models.

**Mathematical Framework:**

The STFT of the discrete signal sequence $x[n]$ with a sliding window $w[n]$ over an overlap parameter $m$:

$$ X_m(\omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j\omega n} $$

This matrix $X_m(\omega)$ is exactly what feeds the image patching algorithm of a ViT.

  

### 10. [Classification of Partial Discharge Sources in UHF Range Using Envelope Detection](https://www.mdpi.com/2079-9292/13/12/2399)

**Date:** June 19, 2024

**Summary:** Democratizes monitoring through low-cost printed monopole antennas and RC envelope detection bypassing high-speed sampling.

**Implementation Notes for App:**

Since analog envelope detection handles the peak extraction via hardware, the software in `time_series.py` simply treats the input as a highly smoothed pulse train. We can simulate this hardware behavior algorithmically using a recursive moving average filter.

**Mathematical Framework:**

Simulation of the RC envelope detector on the raw signal $x[t]$ using a single-pole IIR low-pass filter with coefficient $\alpha$:

$$ y[n] = \alpha y[n-1] + (1 - \alpha) |x[n]| $$

Where $\alpha = e^{-\Delta t / RC}$.

  

---

  

## Part II: Research Conducted in Panama (UTP)

  

### 11. [Detection of Partial Discharge Sources Using UHF Sensors and Independent Component Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC5712867/)

**Summary:** Resolves the overlapping signal problem using an Independent Component Analysis (ICA) algorithm.

**Implementation Notes for App:**

We already have `blind_algorithms.py` in the workspace. We can utilize `scipy` or `sklearn.decomposition.FastICA` here to segregate multiple concurrent discharge phenomena directly into cleanly separated source vectors.

**Mathematical Framework:**

Assuming we have $n$ UHF sensor observations $\vec{x}(t)$ resulting from a linear mixture $A$ of $m$ independent PD sources $\vec{s}(t)$:

$$ \vec{x}(t) = A \vec{s}(t) $$

ICA algorithms seek a de-mixing matrix $W$ to recover the estimated independent components $\vec{y}(t)$:

$$ \vec{y}(t) = W \vec{x}(t) \approx \vec{s}(t) $$

The objective is to maximize the statistical non-Gaussianity (negentropy) of $W \vec{x}$.

  

### 12. [Variation in the Spectral Content of UHF PD Signals Due to the Presence of Obstacles](https://ieeexplore.ieee.org/document/9600947/)

**Summary:** Characterizes how physical obstacles (e.g., transformer geometry) modify the high-frequency spectral content of propagating PD signals.

**Implementation Notes for App:**

We can implement an Environmental Spectrum Correction block inside `validation.py`. By characterizing geometric attenuation and reflection, we apply inverse filters to equalize the received signal before evaluating severity.

**Mathematical Framework:**

The transfer function of the environment $H_{env}(f)$ models the path attenuation $\alpha(f)$ and phase shift $\phi(f)$ caused by physical obstacles:

$$ H_{env}(f) = e^{-\alpha(f) d} \cdot e^{-j \phi(f)} $$

The received distorted signal spectrum $Y(f) = X(f) H_{env}(f)$. The application can reconstruct the assumed original emission characteristics via an inverse spectral equalization:

$$ \hat{X}(f) = Y(f) \cdot H_{env}^{-1}(f) $$