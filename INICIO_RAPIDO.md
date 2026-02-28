# 🚀 Quick Start Guide - UHF-PD System

## ✅ Current Status

The project is installed and ready to run.

---

## 🎯 Start the Application

### Option 1: Simple start
```bash
python app.py
```

### Option 2: Start script (recommended)
```bash
python start_gui.py
```

### Option 3: Custom options
```bash
python start_gui.py --port 8080
python start_gui.py --debug
python start_gui.py --port 8080 --debug
```

---

## 🌐 Open the UI

After startup, open:

**http://localhost:8050**

For remote servers:

**http://[SERVER_IP]:8050**

---

## 📋 Available Tabs

### 📡 Live Capture
- Real-time monitoring
- NI PXIe-5185 hardware mode
- Simulation mode

### 📂 File Analysis
- Offline analysis of saved data
- Supports CSV, HDF5, and MATLAB files

### ⚙️ Signal Generator
- Generate synthetic datasets
- Export in multiple formats

### 🎯 Threshold Configuration
- Calibrate severity thresholds
- Run validation tests

### 📚 Documentation
- In-app reference and technical notes

---

## 🧪 Quick Validation Steps

### 1) Verify installation
```bash
python test_system.py
```

### 2) First simulation test
1. Run `python app.py`
2. Open **📡 Live Capture**
3. Select **Simulation Mode**
4. Select **🟢 Green**
5. Click **Start Capture**
6. Confirm live plots update continuously

### 3) Generator test
1. Open **⚙️ Signal Generator**
2. Select **🔴 Red**
3. Set duration, number of discharges, and amplitude
4. Click **Generate Signal**
5. Review spectrum/statistics and optionally export

### 4) Threshold test
1. Open **🎯 Threshold Configuration**
2. Keep default thresholds (or edit)
3. Run full validation
4. Review confusion matrix and metrics

---

## 🔧 Real Hardware Setup (NI PXIe-5185)

1. Install NI-DAQmx driver from National Instruments.
2. Install Python bindings:
```bash
pip install nidaqmx
```
3. Check device name in NI MAX (for example `PXI1Slot2`).

In **📡 Live Capture**:
- Select hardware mode
- Set device/channel/sample rate
- Start capture

If capture fails:
- check cabling
- verify NI MAX device name
- validate permissions
- test simulation mode first

---

## 📊 File Input Formats

### CSV
Expected layout example:
```csv
time,signal
0.0000,0.0123
0.0001,0.0245
```

### HDF5
Provide dataset name containing the signal.

### MATLAB
Provide variable name containing the signal.

---

## 🎓 Result Interpretation

| State | Meaning | Typical Action |
|-------|---------|----------------|
| 🟢 Green | Normal | Routine monitoring |
| 🟡 Yellow | Caution | Increase monitoring frequency |
| 🟠 Orange | Alert | Plan maintenance |
| 🔴 Red | Critical | Immediate intervention |

---

## 🆘 Troubleshooting

**App does not start**
```bash
pip install -r requirements.txt
python test_system.py
```

**Plots do not refresh**
- hard refresh browser
- clear cache
- inspect browser console

**File upload fails**
- verify format (CSV/H5/MAT)
- verify signal field/column name

---

## 📌 Recommended Next Steps

1. Explore simulation mode
2. Analyze existing recordings
3. Generate synthetic data for benchmarking
4. Calibrate thresholds to your equipment
5. Move to real-hardware acquisition
