"""High-level spectrum analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.signal import find_peaks

from .fitting import PeakFit, fit_peak
from .preprocessing import (
    PreprocessConfig,
    detect_peaks,
    estimate_noise_sigma,
    rolling_median_baseline,
)


@dataclass
class AnalysisConfig:
    top_fraction: float = 0.2
    k_noise: float = 5.0
    baseline_window: int = 51
    smoothing_window: int = 11
    smoothing_poly: int = 3
    peak_window: float = 30.0  # window width in x units around each peak
    min_distance_pts: int | None = None
    enable_shoulders: bool = True
    shoulder_height_ratio: float = 0.5  # vs main threshold
    shoulder_prominence_ratio: float = 0.3
    shoulder_distance: float | None = None  # in x units
    dedup_distance_factor: float = 0.5  # scaling for dedup distance to allow close shoulders


@dataclass
class AnalysisResult:
    x: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray
    noise_sigma: float
    peaks: List[PeakFit]

    def reconstructed(self, x_grid: np.ndarray | None = None) -> np.ndarray:
        x_eval = x_grid if x_grid is not None else self.x
        total = np.zeros_like(x_eval, dtype=float)
        for peak in self.peaks:
            total += peak.evaluate(x_eval)
        if x_grid is None:
            return total + self.baseline
        baseline_interp = np.interp(x_eval, self.x, self.baseline)
        return total + baseline_interp


def _deduplicate_peaks(peaks: List[PeakFit], min_distance: float) -> List[PeakFit]:
    if not peaks:
        return []
    peaks_sorted = sorted(peaks, key=lambda p: p.center)
    kept: List[PeakFit] = [peaks_sorted[0]]
    for pk in peaks_sorted[1:]:
        prev = kept[-1]
        if abs(pk.center - prev.center) < min_distance:
            # keep the one with better (lower) AIC
            kept[-1] = pk if pk.aic < prev.aic else prev
        else:
            kept.append(pk)
    return kept


def detect_shoulders(
    x: np.ndarray,
    smooth_y: np.ndarray,
    main_idxs: np.ndarray,
    noise_sigma: float,
    config: AnalysisConfig,
) -> np.ndarray:
    """Find shoulder candidates near main peaks using a relaxed threshold."""
    if not len(main_idxs):
        return np.array([], dtype=int)

    min_height = max(noise_sigma * config.k_noise * config.shoulder_height_ratio, noise_sigma * 0.5)
    prominence = noise_sigma * config.k_noise * config.shoulder_prominence_ratio
    distance_pts = max(1, int((config.min_distance_pts or max(1, len(x) // 200)) * 0.5))

    candidates, _ = find_peaks(
        smooth_y,
        height=min_height,
        prominence=prominence if prominence > 0 else None,
        distance=distance_pts,
    )

    main_pos = x[main_idxs]
    shoulder_dist = config.shoulder_distance if config.shoulder_distance is not None else config.peak_window * 0.6

    shoulder_idxs: list[int] = []
    for idx in candidates:
        pos = x[idx]
        if np.min(np.abs(pos - main_pos)) <= shoulder_dist and idx not in main_idxs:
            shoulder_idxs.append(int(idx))
    return np.array(shoulder_idxs, dtype=int)


def analyze_spectrum(
    x: Sequence[float],
    y: Sequence[float],
    config: AnalysisConfig,
) -> AnalysisResult:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    baseline = (
        rolling_median_baseline(y_arr, config.baseline_window)
        if config.baseline_window and config.baseline_window > 2
        else np.zeros_like(y_arr)
    )
    residual = y_arr - baseline

    noise_sigma = estimate_noise_sigma(residual)
    preprocess_cfg = PreprocessConfig(
        baseline_window=config.baseline_window,
        smoothing_window=config.smoothing_window,
        smoothing_poly=config.smoothing_poly,
        k_noise=config.k_noise,
        min_distance_pts=config.min_distance_pts,
    )
    min_height = config.k_noise * noise_sigma
    peak_idxs, _, smooth_y = detect_peaks(x_arr, residual, preprocess_cfg, min_height=min_height)

    seeds = peak_idxs
    if config.enable_shoulders:
        shoulder_idxs = detect_shoulders(x_arr, smooth_y, peak_idxs, noise_sigma, config)
        if len(shoulder_idxs):
            seeds = np.unique(np.concatenate([peak_idxs, shoulder_idxs]))

    fits: List[PeakFit] = []
    for idx in seeds:
        center = float(x_arr[idx])
        fit = fit_peak(x_arr, residual, center, config.peak_window, config.top_fraction)
        if fit and fit.amplitude >= min_height * 0.8:  # allow slight margin
            fits.append(fit)

    # deduplicate overlapping centers roughly within half the peak window
    min_dist_val = config.min_distance_pts if config.min_distance_pts else max(1, len(x_arr) // 200)
    if len(x_arr) > 1:
        x_step = np.median(np.diff(np.sort(x_arr)))
        min_dist_x = max(x_step, min_dist_val * x_step * config.dedup_distance_factor)
    else:
        min_dist_x = config.peak_window * 0.2
    fits = _deduplicate_peaks(fits, min_distance=min_dist_x)

    return AnalysisResult(x_arr, y_arr, baseline, residual, noise_sigma, fits)
