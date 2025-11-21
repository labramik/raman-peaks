"""Streamlit interface for Raman peak fitting."""
from __future__ import annotations

import csv
import io

import streamlit as st

from .analysis import AnalysisConfig, analyze_spectrum
from .io_utils import load_spectrum_from_string
from .plotting import build_peak_table, plot_analysis, save_plot_bytes


def _peaks_to_csv(peaks) -> bytes:
    """Serialize peaks to CSV bytes buffer."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["center", "amplitude", "fwhm", "model", "aic", "rss"],
    )
    writer.writeheader()
    for row in build_peak_table(peaks):
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def main():
    """Streamlit entry point."""
    # pylint: disable=too-many-locals
    st.set_page_config(page_title="Raman Peak Fitter", layout="wide")
    st.title("Raman peak fitter (top-fraction)")
    st.write(
        "Upload a two-column TXT spectrum (shift, intensity). "
        "The app will detect and fit peaks using only the top fraction of each peak."
    )

    with st.sidebar:
        st.header("Parameters")
        top_fraction = st.slider(
            "Top fraction of peak height",
            min_value=0.05,
            max_value=0.6,
            value=0.2,
            step=0.05,
        )
        k_noise = st.slider(
            "Sensitivity (k x noise)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
        )
        smoothing_window = st.slider(
            "Smoothing window (points)",
            min_value=5,
            max_value=51,
            value=11,
            step=2,
        )
        baseline_on = st.checkbox("Apply baseline (rolling median)", value=True)
        baseline_window = st.slider(
            "Baseline window (points)",
            min_value=11,
            max_value=201,
            value=51,
            step=2,
        )
        peak_window = st.slider(
            "Peak fit window (cm^-1)",
            min_value=5.0,
            max_value=80.0,
            value=30.0,
            step=1.0,
        )
        shoulders_on = st.checkbox("Detect shoulders (relaxed threshold)", value=True)
        shoulder_height_ratio = st.slider(
            "Shoulder height ratio vs main threshold",
            min_value=0.2,
            max_value=0.8,
            value=0.5,
            step=0.05,
        )

    upload = st.file_uploader("Spectrum TXT", type=["txt", "dat", "csv"])

    if upload:
        try:
            content = upload.getvalue().decode("utf-8", errors="ignore")
            x, y = load_spectrum_from_string(content)
        except (UnicodeDecodeError, ValueError) as exc:
            st.error(f"Failed to read spectrum: {exc}")
            return

        config = AnalysisConfig(
            top_fraction=top_fraction,
            k_noise=k_noise,
            baseline_window=baseline_window if baseline_on else 0,
            smoothing_window=smoothing_window,
            peak_window=peak_window,
            enable_shoulders=shoulders_on,
            shoulder_height_ratio=shoulder_height_ratio,
        )

        result = analyze_spectrum(x, y, config)

        st.info(
            f"Detected {len(result.peaks)} peaks | noise sigma ~ {result.noise_sigma:.3g} "
            f"| top fraction {top_fraction:.2f}"
        )

        col_plot, col_table = st.columns([2, 1])
        with col_plot:
            fig = plot_analysis(result, show_baseline=baseline_on)
            st.pyplot(fig)
            png_bytes = save_plot_bytes(fig)
            st.download_button(
                "Download plot (PNG)",
                data=png_bytes,
                file_name="raman_peaks.png",
                mime="image/png",
            )
        with col_table:
            if result.peaks:
                st.subheader("Fitted peaks")
                st.table(build_peak_table(result.peaks))
                st.download_button(
                    "Download peaks (CSV)",
                    data=_peaks_to_csv(result.peaks),
                    file_name="peaks.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No peaks detected under current settings.")
    else:
        st.info("Upload a text file to start.")


if __name__ == "__main__":
    main()
