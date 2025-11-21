"""Signal preprocessing: baseline, smoothing, noise estimation, peak detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import medfilt, savgol_filter, find_peaks


@dataclass
class PreprocessConfig:
    """Parameters controlling smoothing, baseline, and peak spacing."""
    baseline_window: int = 51
    smoothing_window: int = 11
    smoothing_poly: int = 3
    k_noise: float = 5.0  # min height = k_noise * sigma
    min_distance_pts: Optional[int] = None  # peak separation in points


def ensure_odd(n: int) -> int:
    """Return n if odd, else n+1."""
    return n if n % 2 == 1 else n + 1


def rolling_median_baseline(y: np.ndarray, window: int) -> np.ndarray:
    """Rolling median baseline estimate."""
    window = max(3, ensure_odd(window))
    return medfilt(y, kernel_size=window)


def estimate_noise_sigma(y: np.ndarray) -> float:
    """Robust noise estimate via MAD."""
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    return 1.4826 * mad if mad > 0 else max(np.std(y), 1e-12)


def smooth_signal(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Light Savitzky-Golay smoothing with guardrails."""
    n = len(y)
    if n < poly + 2 or n < 5:
        return y
    window = ensure_odd(max(poly + 2, min(window, n - (1 - n % 2))))
    if window > n:
        window = ensure_odd(n - (1 - n % 2))
    return savgol_filter(y, window_length=window, polyorder=poly)


def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    config: PreprocessConfig,
    min_height: float,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Find candidate peaks on a preprocessed signal."""
    smooth_y = smooth_signal(y, config.smoothing_window, config.smoothing_poly)
    distance = config.min_distance_pts or max(1, len(x) // 200)
    peaks, props = find_peaks(
        smooth_y,
        height=min_height,
        prominence=min_height,
        distance=distance,
    )
    return peaks, props, smooth_y
