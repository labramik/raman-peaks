"""Plotting helpers."""
from __future__ import annotations

import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .analysis import AnalysisResult
from .fitting import PeakFit


def build_peak_table(peaks: List[PeakFit]) -> List[dict]:
    table = []
    for p in peaks:
        row = {
            "center": p.center,
            "amplitude": p.amplitude,
            "fwhm": p.fwhm,
            "model": p.model,
            "aic": p.aic,
            "rss": p.rss,
        }
        table.append(row)
    return table


def plot_analysis(result: AnalysisResult, show_baseline: bool = True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.x, result.y, label="spectrum", color="C0", lw=1.2)

    if show_baseline and result.baseline is not None:
        ax.plot(result.x, result.baseline, label="baseline", color="C2", lw=1.0, ls="--")

    if result.peaks:
        y_fit = result.reconstructed()
        ax.plot(result.x, y_fit, label="fit", color="C3", lw=1.0, alpha=0.8)
        centers = [p.center for p in result.peaks]
        base_at_centers = np.interp(centers, result.x, result.baseline)
        heights = [b + p.amplitude for b, p in zip(base_at_centers, result.peaks)]
        ax.scatter(centers, heights, color="C1", s=30, zorder=5, label="fitted peaks")
        for p, h in zip(result.peaks, heights):
            ax.annotate(f"{p.center:.1f}", xy=(p.center, h), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Raman peak fitting (top-fraction)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def save_plot_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    return buf.getvalue()
