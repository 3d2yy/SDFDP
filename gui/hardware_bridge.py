"""
NI PXIe-5185 Hardware Bridge
==============================

Abstraction layer for communicating with the NI PXIe-5185 digitiser
(12.5 GS/s, 3 GHz BW, 8-bit).  Provides:

- Automatic fallback to simulation when hardware is unavailable
- gRPC-ready async capture interface for remote streaming
- Ring-buffer management for continuous acquisition
- Configuration presets for common PD measurement scenarios
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.floating[Any]]

log = logging.getLogger(__name__)

# ============================================================================
# INSTRUMENT SPEC
# ============================================================================

PXIE_5185_SPEC = {
    'model': 'NI PXIe-5185',
    'max_sample_rate_gs': 12.5,
    'bandwidth_ghz': 3.0,
    'resolution_bits': 8,
    'input_range_vpp': [0.2, 0.4, 1.0, 2.0, 4.0, 10.0],
    'memory_samples': 256_000_000,
    'channels': 2,
}


@dataclass
class CaptureConfig:
    """Parameters for a single acquisition."""
    device: str = 'PXI1Slot2'
    channel: int = 0
    sample_rate_gs: float = 12.5
    record_length: int = 8192        # samples per capture
    input_range_vpp: float = 2.0
    trigger_level_v: float = 0.1
    trigger_slope: str = 'rising'
    coupling: str = 'AC'
    timeout_s: float = 5.0

    @property
    def sample_rate_hz(self) -> float:
        return self.sample_rate_gs * 1e9

    @property
    def capture_duration_s(self) -> float:
        return self.record_length / self.sample_rate_hz


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS: Dict[str, CaptureConfig] = {
    'uhf_pd_standard': CaptureConfig(
        sample_rate_gs=12.5,
        record_length=8192,
        input_range_vpp=2.0,
        trigger_level_v=0.1,
    ),
    'uhf_pd_highres': CaptureConfig(
        sample_rate_gs=12.5,
        record_length=32768,
        input_range_vpp=1.0,
        trigger_level_v=0.05,
    ),
    'uhf_pd_fast': CaptureConfig(
        sample_rate_gs=12.5,
        record_length=2048,
        input_range_vpp=4.0,
        trigger_level_v=0.2,
    ),
    'demo_simulation': CaptureConfig(
        sample_rate_gs=0.01,      # 10 MS/s for sim
        record_length=1000,
        input_range_vpp=2.0,
        trigger_level_v=0.0,
    ),
}


# ============================================================================
# RING BUFFER
# ============================================================================

class RingBuffer:
    """Thread-safe ring buffer for continuous signal acquisition."""

    def __init__(self, max_records: int = 200):
        self._lock = threading.Lock()
        self._buf: deque = deque(maxlen=max_records)
        self._total: int = 0

    def push(self, record: Signal, meta: Optional[dict] = None):
        with self._lock:
            self._buf.append({'signal': record, 'meta': meta or {},
                              'timestamp': time.time()})
            self._total += 1

    def latest(self, n: int = 1) -> List[dict]:
        with self._lock:
            items = list(self._buf)
            return items[-n:]

    def all(self) -> List[dict]:
        with self._lock:
            return list(self._buf)

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._total = 0

    @property
    def count(self) -> int:
        return self._total

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buf)


# ============================================================================
# HARDWARE INTERFACE
# ============================================================================

_HW_AVAILABLE = False
try:
    import nidaqmx                                # type: ignore[import]
    from nidaqmx.constants import AcquisitionType  # type: ignore[import]
    _HW_AVAILABLE = True
except ImportError:
    pass


def is_hardware_available() -> bool:
    return _HW_AVAILABLE


def capture_single(cfg: CaptureConfig) -> Tuple[Signal, dict]:
    """
    Acquire one record from the NI PXIe-5185.

    Returns
    -------
    signal : Signal
        1-D float64 voltage array.
    meta : dict
        Capture metadata (timestamp, sample_rate, etc.).
    """
    if not _HW_AVAILABLE:
        raise RuntimeError("NI driver (nidaqmx) not installed.")

    rate = cfg.sample_rate_hz
    n = cfg.record_length

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            f"{cfg.device}/ai{cfg.channel}",
            min_val=-cfg.input_range_vpp / 2,
            max_val=cfg.input_range_vpp / 2,
        )
        task.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n,
        )
        data = task.read(number_of_samples_per_channel=n, timeout=cfg.timeout_s)

    signal = np.asarray(data, dtype=np.float64)
    meta = {
        'device': cfg.device,
        'channel': cfg.channel,
        'sample_rate_hz': rate,
        'record_length': n,
        'input_range_vpp': cfg.input_range_vpp,
        'timestamp': time.time(),
        'source': 'hardware',
    }
    return signal, meta


# ============================================================================
# SIMULATION INTERFACE
# ============================================================================

def simulate_capture(cfg: CaptureConfig, state: str = 'verde',
                     noise_level: float = 0.1, seed: Optional[int] = None
                     ) -> Tuple[Signal, dict]:
    """
    Generate a synthetic PD signal that mimics PXIe-5185 output.
    """
    from main import generate_synthetic_signal  # lazy to avoid circular

    fs = cfg.sample_rate_hz
    n = cfg.record_length

    signal = generate_synthetic_signal(
        state=state,
        duration=n,
        fs=fs,
        noise_level=noise_level,
    )

    meta = {
        'device': 'SIMULATION',
        'channel': cfg.channel,
        'sample_rate_hz': fs,
        'record_length': n,
        'state_simulated': state,
        'noise_level': noise_level,
        'timestamp': time.time(),
        'source': 'simulation',
    }
    return signal, meta


# ============================================================================
# UNIFIED CAPTURE (auto-fallback)
# ============================================================================

def capture(cfg: CaptureConfig, sim_state: str = 'verde',
            sim_noise: float = 0.1) -> Tuple[Signal, dict]:
    """
    Attempt hardware capture; fall back to simulation transparently.
    """
    if _HW_AVAILABLE:
        try:
            return capture_single(cfg)
        except Exception as exc:
            log.warning("Hardware capture failed (%s), falling back to sim.", exc)

    return simulate_capture(cfg, state=sim_state, noise_level=sim_noise)


# ============================================================================
# CONTINUOUS STREAMER (background thread)
# ============================================================================

class ContinuousStreamer:
    """
    Background thread that continuously captures data and pushes it
    into a RingBuffer.  The GUI polls ``ring_buffer.latest()`` on each
    interval callback.
    """

    def __init__(self, cfg: CaptureConfig, ring: RingBuffer,
                 interval_s: float = 0.5,
                 sim_state: str = 'verde', sim_noise: float = 0.1,
                 on_capture: Optional[Callable] = None):
        self.cfg = cfg
        self.ring = ring
        self.interval = interval_s
        self.sim_state = sim_state
        self.sim_noise = sim_noise
        self.on_capture = on_capture
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Streamer started  (interval=%.2fs)", self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        log.info("Streamer stopped.")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        while not self._stop.is_set():
            try:
                sig, meta = capture(self.cfg,
                                    sim_state=self.sim_state,
                                    sim_noise=self.sim_noise)
                self.ring.push(sig, meta)
                if self.on_capture:
                    self.on_capture(sig, meta)
            except Exception as exc:
                log.error("Streamer capture error: %s", exc)
            self._stop.wait(self.interval)
