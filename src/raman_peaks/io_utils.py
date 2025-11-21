"""Spectrum loading utilities."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np


def _detect_delimiter(lines: Iterable[str]) -> str:
    """Guess delimiter from a few lines."""
    candidates = ["\t", ",", " "]
    counts = {d: 0 for d in candidates}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        for d in candidates:
            counts[d] += stripped.count(d)
    # pick delimiter with most splits; fall back to whitespace
    delim = max(counts, key=counts.get)
    return delim if counts[delim] > 0 else " "


def load_spectrum_txt(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load a two-column Raman spectrum (shift, intensity) from txt.

    The function tries tab, comma, and space as delimiters and assumes
    two numeric columns with no header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    preview: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            preview.append(line)

    delim = _detect_delimiter(preview)

    try:
        data = np.loadtxt(path, delimiter=delim, ndmin=2)
    except Exception:
        # try whitespace as final fallback
        data = np.loadtxt(path, ndmin=2)

    if data.shape[1] < 2:
        raise ValueError("Expected at least two numeric columns (shift, intensity)")

    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)
    return x, y


def load_spectrum_from_string(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse spectrum from a text blob (two columns)."""
    lines = text.splitlines()
    preview = lines[:20]
    delim = _detect_delimiter(preview)
    buffer = io.StringIO(text)
    try:
        data = np.loadtxt(buffer, delimiter=delim, ndmin=2)
    except Exception:
        buffer.seek(0)
        data = np.loadtxt(buffer, ndmin=2)

    if data.shape[1] < 2:
        raise ValueError("Expected at least two numeric columns (shift, intensity)")

    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)
    return x, y
