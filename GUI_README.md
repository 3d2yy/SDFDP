# 🔌 UHF Partial Discharge Detection System - Graphical Interface

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.14+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Professional platform for real-time monitoring and offline analysis of partial discharge signals**

</div>

---

## 🚀 Main Features

### 📡 **Live Capture**
- **Real Hardware**: Compatible with NI PXIe-5185 (12.5 GS/s, 3 GHz BW, 8-bit)
- **Simulation Mode**: Synthetic generation for no-hardware testing
- **Real-Time Monitoring**: Continuous plotting of signals and descriptors
- **Automatic Classification**: Traffic-light severity states (Green/Yellow/Orange/Red)

### 📂 **File Analysis**
- **Multiple Formats**: CSV, HDF5 (.h5), MATLAB (.mat)
- **Full Visualizations**: Signal, spectrum, descriptors, radar chart
- **Advanced Processing**: Filtering, normalization, envelope extraction
- **Severity Evaluation**: Automatic classification with detailed outputs

### ⚙️ **Signal Generator**
- **Custom Parameters**: State, amplitude, frequency, noise
- **Noise Types**: Gaussian, Pink, Brown, Uniform
- **Multi-Format Export**: CSV, HDF5, MAT with metadata
- **Immediate Analysis**: Statistics, spectrum, histograms

### 🎯 **Threshold Configuration**
- **Custom Thresholds**: Adjust classification boundaries
- **Descriptor Weights**: Control relative importance
- **Interactive Tests**: Generate and classify in real time
- **Full Validation**: Confusion matrix and accuracy metrics

### 📚 **Integrated Documentation**
- Step-by-step user guidance
- Technical specifications
- Best practices

---

## 📦 Installation

### 1. Clone or download the repository

```bash
cd /workspaces/V2DP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install NI hardware support

If you plan to use National Instruments hardware:

```bash
pip install nidaqmx
```

---

## 🎯 Quick Use

### Start the application

```bash
python app.py
```

The interface is available at: **http://localhost:8050**

### Recommended workflow

1. **📚 Documentation**: Understand system behavior
2. **🎯 Threshold Configuration**: Adjust parameters as needed
3. **⚙️ Generator**: Create synthetic test signals
4. **📂 File Analysis**: Analyze existing recordings
5. **📡 Live Capture**: Move to real-time monitoring

---

## 🔧 Configuration

### NI PXIe-5185 Hardware

To use real hardware, in **Live Capture**:

1. Select "NI PXIe-5185 Hardware"
2. Configure:
   - **Device**: Device name (for example `PXI1Slot2`)
   - **Channel**: Analog channel number (for example `0`)
   - **Sampling Rate**: In GS/s (for example `12.5`)
3. Start capture

### Simulation Mode

For no-hardware testing:

1. Select "Simulation Mode"
2. Choose state:
   - 🟢 Green (Normal)
   - 🟡 Yellow (Caution)
   - 🟠 Orange (Alert)
   - 🔴 Red (Critical)
3. Tune noise level
4. Start capture

---

## 📊 Computed Descriptors

The operational path computes nine descriptors:

| # | Descriptor | Description |
|---|------------|-------------|
| 1 | **Total Energy** | Sum of squared signal amplitudes |
| 2 | **RMS** | Root mean square value |
| 3 | **Kurtosis** | Tail/peakedness indicator |
| 4 | **Skewness** | Distribution asymmetry |
| 5 | **Crest Factor** | Peak-to-RMS ratio |
| 6 | **Peak Count** | Number of significant peaks |
| 7 | **Spectral Entropy** | Spectral disorder |
| 8 | **Spectral Stability** | Inter-window spectral consistency |
| 9 | **Zero-Crossing Rate** | Sign-change frequency |

---

## 🎨 Project Structure

```
V2DP/
├── app.py                      # Main Dash application
├── gui/
│   ├── __init__.py
│   ├── live_capture.py         # Real-time capture tab
│   ├── file_analysis.py        # File analysis tab
│   ├── signal_generator.py     # Signal generator tab
│   ├── threshold_config.py     # Threshold configuration tab
│   └── documentation.py        # In-app docs tab
├── main.py                     # Backend processing layer
├── preprocessing.py            # Signal preprocessing + MC optimization
├── descriptors.py              # Δt extraction + legacy descriptors
├── severity.py                 # Severity scoring and traffic-light mapping
├── blind_algorithms.py         # Δt tracking algorithms
├── validation.py               # Complexity and validation metrics
└── requirements.txt            # Dependencies
```

---

## 🔬 Technical Specifications

### Acquisition System

| Component | Specification |
|------------|----------------|
| **System** | NI PXIe-1071 |
| **Controller** | NI PXIe-8135 (Embedded) |
| **Digitizer** | NI PXIe-5185 |
| **Bandwidth** | 3 GHz |
| **Sampling Rate** | 12.5 GS/s |
| **Resolution** | 8 bits |

### Signal Processing

- **Filtering**: Band-pass (1% - 40% of fs)
- **Normalization**: Adaptive
- **Envelope**: Hilbert transform
- **Denoising**: Wavelets

---

## 📖 Usage Examples

### Example 1: Analyze a CSV file

```python
# In the "File Analysis" tab:
# 1. Upload a CSV signal file
# 2. Set fs = 10000 Hz
# 3. Set data column = "signal"
# 4. Click "Analyze Signal"
# 5. Review classification and descriptors
```

### Example 2: Generate a synthetic dataset

```python
# In the "Signal Generator" tab:
# 1. State = "Orange"
# 2. Duration = 5000 samples
# 3. Discharges = 30
# 4. Amplitude = 4.0
# 5. Click "Generate Signal"
# 6. Export as HDF5 with metadata
```

### Example 3: Calibrate thresholds

```python
# In the "Threshold Configuration" tab:
# 1. Set Green→Yellow = 0.3
# 2. Set Yellow→Orange = 0.6
# 3. Set Orange→Red = 0.8
# 4. Click "Run Full Test"
# 5. Review confusion matrix and accuracy
```

---

## 🐛 Troubleshooting

### Error: "nidaqmx is not installed"

```bash
pip install nidaqmx
```

### Error: "h5py not found"

```bash
pip install h5py
```

### Application does not start

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### NI hardware is not detected

1. Verify NI-DAQmx driver installation
2. Confirm the device name in NI MAX
3. Use the exact device name in settings

---

## 🤝 Contributing

Contributions are welcome. Please open an issue before large structural changes.

---

## 📄 License

See LICENSE / license.md.

---

## 🙏 Acknowledgements

Built with:
- **Dash & Plotly** for interactive visualization
- **NumPy & SciPy** for scientific processing
- **NI-DAQmx** for instrumentation integration
- **Bootstrap** for responsive UI design

---

<div align="center">

**🔌 UHF Partial Discharge Detection System**

*Professional real-time monitoring for high-voltage assets*

</div>
