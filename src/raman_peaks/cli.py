"""Command-line interface for Raman peak fitting."""
import argparse
import csv
from pathlib import Path
from typing import List, Optional

from .analysis import AnalysisConfig, analyze_spectrum
from .io_utils import load_spectrum_txt
from .plotting import build_peak_table, plot_analysis


def write_csv(peaks, path: Path):
    """Write fitted peaks to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["center", "amplitude", "fwhm", "model", "aic", "rss"],
        )
        writer.writeheader()
        for row in build_peak_table(peaks):
            writer.writerow(row)


def main(argv: Optional[List[str]] = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Fit Raman peaks using top-fraction fitting.")
    parser.add_argument("input", help="Input TXT with two numeric columns (shift, intensity)")
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.2,
        help="Top fraction of peak height to fit (0-1)",
    )
    parser.add_argument(
        "--k-noise",
        type=float,
        default=5.0,
        help="Minimum peak height as k x noise",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=51,
        help="Rolling median baseline window (points). 0 disables.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=11,
        help="Savitzky-Golay smoothing window (points)",
    )
    parser.add_argument(
        "--smoothing-poly",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order",
    )
    parser.add_argument(
        "--peak-window",
        type=float,
        default=30.0,
        help="Window width around each peak (in x units)",
    )
    parser.add_argument(
        "--min-distance-pts",
        type=int,
        default=None,
        help="Minimum separation between peaks (in points)",
    )
    parser.add_argument("--no-shoulders", action="store_true", help="Disable shoulder detection")
    parser.add_argument(
        "--shoulder-height-ratio",
        type=float,
        default=0.5,
        help="Shoulder detection height ratio vs main threshold",
    )
    parser.add_argument(
        "--shoulder-distance",
        type=float,
        default=None,
        help="Shoulder search distance in x units (default: 0.6 * peak window)",
    )
    parser.add_argument("--plot", type=Path, help="Save plot to PNG")
    parser.add_argument("--csv", type=Path, help="Save peaks table to CSV")

    args = parser.parse_args(argv)

    x, y = load_spectrum_txt(args.input)
    config = AnalysisConfig(
        top_fraction=args.top_fraction,
        k_noise=args.k_noise,
        baseline_window=args.baseline_window,
        smoothing_window=args.smoothing_window,
        smoothing_poly=args.smoothing_poly,
        peak_window=args.peak_window,
        min_distance_pts=args.min_distance_pts,
        enable_shoulders=not args.no_shoulders,
        shoulder_height_ratio=args.shoulder_height_ratio,
        shoulder_distance=args.shoulder_distance,
    )
    result = analyze_spectrum(x, y, config)

    print(f"Detected {len(result.peaks)} peaks | noise sigma ~ {result.noise_sigma:.3g}")
    for p in result.peaks:
        print(
            f"{p.center:10.3f}  amp={p.amplitude:10.3f}  "
            f"fwhm={p.fwhm:8.3f}  model={p.model:9s}  AIC={p.aic:8.2f}"
        )

    if args.csv:
        write_csv(result.peaks, args.csv)
        print(f"Saved peaks to {args.csv}")

    if args.plot:
        fig = plot_analysis(result)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
