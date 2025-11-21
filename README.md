# Raman peak fitter (top-fraction)

Desktop-ready toolkit to detect Raman peaks from a two-column TXT export (shift, intensity), fit only the top fraction of each peak (Lorentzian vs Voigt best pick), and export tables/plots. Includes CLI and a lightweight Streamlit UI. Supports Python 3.9+.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CLI

```bash
python -m raman_peaks.cli input.txt \
  --top-fraction 0.2 \
  --k-noise 5 \
  --baseline-window 51 \
  --smoothing-window 11 \
  --peak-window 30 \
  --shoulder-height-ratio 0.5 \
  --plot output.png \
  --csv peaks.csv
```

### Streamlit UI

```bash
streamlit run src/raman_peaks/streamlit_app.py
```

Upload a TXT/CSV with two numeric columns. Use the sliders to set:
- **Top fraction**: percentage of peak height to include in the fit.
- **Sensitivity (k x noise)**: minimum peak height relative to robust noise estimate.
- **Baseline window**: rolling median size; disable if your baseline is already flat.
- **Peak window**: width around each candidate used for fitting.

Outputs: PNG plot of spectrum/baseline/fits with labeled peak centers and a CSV of peak parameters (center, amplitude, FWHM, model, AIC).

## How it works

1) Load spectrum (delimiter auto-detected) and optional rolling-median baseline removal.
2) Robust noise estimate via MAD; sensitivity sets min peak height = k x noise.
3) Light Savitzky-Golay smoothing + peak seeding via prominence, plus optional shoulder detection using a relaxed threshold near main peaks.
4) For each seed, take a configurable window and keep only the top fraction of points; fit Lorentzian and Voigt with nonlinear least squares; choose the lower AIC.
5) Deduplicate nearby centers (looser threshold to allow shoulders) and report center/height/FWHM/model/fit score. The reconstructed fit is baseline + sum of peak curves.

## Notes
- Assumes Raman shift (cm^-1) then intensity, no header. If parsing fails, open the file to verify two numeric columns.
- Defaults: top fraction 0.2, sensitivity k=5, baseline window 51 points, smoothing window 11 points, peak window 30 cm^-1.
- Shoulder detection is on by default with height ratio 0.5 vs main threshold; disable with `--no-shoulders` in the CLI.
- Command-line and UI share the same analysis core; adjust parameters in either place to tune sensitivity or focus on peak tops.
