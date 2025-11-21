"""Peak fitting (top-fraction Lorentzian/Voigt)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz


# Model functions ------------------------------------------------------------

def lorentzian(x: np.ndarray, x0: float, gamma: float, amplitude: float) -> np.ndarray:
    gamma = np.abs(gamma)
    return amplitude * (0.5 * gamma) ** 2 / ((x - x0) ** 2 + (0.5 * gamma) ** 2)


def voigt_profile(x: np.ndarray, x0: float, sigma: float, gamma: float, amplitude: float) -> np.ndarray:
    sigma = max(np.abs(sigma), 1e-6)
    gamma = max(np.abs(gamma), 1e-6)
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


MODEL_FUNCS: Dict[str, Callable[..., np.ndarray]] = {
    "lorentzian": lorentzian,
    "voigt": voigt_profile,
}


@dataclass
class PeakFit:
    center: float
    amplitude: float
    fwhm: float
    model: str
    aic: float
    rss: float
    n_points: int
    params: Dict[str, float]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        func = MODEL_FUNCS[self.model]
        if self.model == "lorentzian":
            return func(x, self.params["x0"], self.params["gamma"], self.params["amplitude"])
        return func(
            x,
            self.params["x0"],
            self.params["sigma"],
            self.params["gamma"],
            self.params["amplitude"],
        )


def _fwhm_from_params(model: str, params: Dict[str, float]) -> float:
    if model == "lorentzian":
        return float(np.abs(params["gamma"]))
    # Voigt approximation (Olivero-Longbothum)
    sigma = max(np.abs(params["sigma"]), 1e-12)
    gamma = max(np.abs(params["gamma"]), 1e-12)
    return 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma) ** 2 + (2.0 * np.sqrt(2 * np.log(2)) * sigma) ** 2)


def _aic(n: int, rss: float, k: int) -> float:
    if rss <= 0:
        rss = 1e-12
    return n * np.log(rss / n) + 2 * k


def select_top_fraction(x: np.ndarray, y: np.ndarray, fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    fraction = np.clip(fraction, 0.01, 0.9)
    y_max = float(np.max(y))
    threshold = y_max * (1 - fraction)
    mask = y >= threshold
    if mask.sum() < 5:
        return x, y  # fall back to full window
    return x[mask], y[mask]


def _fit_model(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    center_guess: float,
    height_guess: float,
    width_guess: float,
) -> Tuple[Optional[PeakFit], np.ndarray]:
    func = MODEL_FUNCS[model]
    try:
        if model == "lorentzian":
            p0 = [center_guess, max(width_guess, 1e-3), max(height_guess, 1e-6)]
            bounds = ([center_guess - width_guess * 2, 1e-6, 0], [center_guess + width_guess * 2, np.inf, np.inf])
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
            params = {"x0": popt[0], "gamma": popt[1], "amplitude": popt[2]}
            y_fit = func(x, *popt)
            rss = float(np.sum((y - y_fit) ** 2))
            aic = _aic(len(x), rss, k=3)
            fwhm = _fwhm_from_params(model, params)
            return PeakFit(params["x0"], params["amplitude"], fwhm, model, aic, rss, len(x), params), y_fit
        else:
            p0 = [center_guess, width_guess / 3.0, max(width_guess, 1e-3), max(height_guess, 1e-6)]
            bounds = (
                [center_guess - width_guess * 2, 1e-6, 1e-6, 0],
                [center_guess + width_guess * 2, np.inf, np.inf, np.inf],
            )
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=12000)
            params = {"x0": popt[0], "sigma": popt[1], "gamma": popt[2], "amplitude": popt[3]}
            y_fit = func(x, *popt)
            rss = float(np.sum((y - y_fit) ** 2))
            aic = _aic(len(x), rss, k=4)
            fwhm = _fwhm_from_params(model, params)
            return PeakFit(params["x0"], params["amplitude"], fwhm, model, aic, rss, len(x), params), y_fit
    except Exception:
        return None, np.array([])


def fit_peak(
    x: np.ndarray,
    y: np.ndarray,
    center_guess: float,
    window: float,
    top_fraction: float,
) -> Optional[PeakFit]:
    """Fit a single peak around a seed center using top fraction of data."""
    if len(x) < 5:
        return None
    if window <= 0:
        window = (np.max(x) - np.min(x)) * 0.05

    half_w = window / 2
    mask = np.abs(x - center_guess) <= half_w
    if mask.sum() < 5:
        # fallback: pick nearest points
        idx = np.argsort(np.abs(x - center_guess))[:7]
        mask = np.zeros_like(x, dtype=bool)
        mask[idx] = True
    x_win = x[mask]
    y_win = y[mask]
    if len(x_win) < 5:
        return None

    x_top, y_top = select_top_fraction(x_win, y_win, top_fraction)
    height_guess = float(np.max(y_top))
    width_guess = max(np.median(np.diff(np.sort(x_win))), (x_win.max() - x_win.min()) / 10, 1e-3)

    results: List[PeakFit] = []
    for model in ("lorentzian", "voigt"):
        fit_result, _ = _fit_model(model, x_top, y_top, center_guess, height_guess, width_guess)
        if fit_result:
            results.append(fit_result)

    if not results:
        return None

    # pick model with lowest AIC
    results.sort(key=lambda r: r.aic)
    return results[0]
