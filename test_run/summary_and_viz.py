"""Generate summary CSV and visualizations from a shallow simulation HDF5.

Usage:
    python summary_and_viz.py                     # uses output/sim_fra_shallow.h5
    python summary_and_viz.py CAN                 # uses output/sim_can_shallow.h5
    python summary_and_viz.py FRA path/to/file.h5

The script writes a CSV with the summary table and PNG plots to the chosen
output directory (default: `output/`).
"""
from pathlib import Path
import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_COUNTRY = "FRA"


def load_shallow(h5_path: Path, country: str):
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    store = pd.HDFStore(h5_path, mode="r")
    try:
        keys = [k.strip("/") for k in store.keys()]
        if country not in keys:
            raise KeyError(f"Country key '{country}' not found in HDF5. Available: {keys}")

        summary = store[country]
        industries = None
        ind_key = f"{country}_industries"
        if ind_key in keys:
            industries = store[ind_key]
    finally:
        store.close()

    return summary, industries


def save_csv(df: pd.DataFrame, out_dir: Path, country: str):
    out_path = out_dir / f"summary_{country.lower()}.csv"
    df.to_csv(out_path)
    return out_path


def plot_timeseries(df: pd.DataFrame, out_dir: Path, country: str):
    metrics = [
        "Production",
        "Sales",
        "Wages",
        "Profits",
        "Taxes Paid on Production",
        "Household Consumption",
        "Government Consumption",
        "Imports",
        "Exports",
        "Gross Output",
        "Unemployment Rate",
        "CPI",
        "PPI",
    ]

    found = [m for m in metrics if m in df.columns]
    if not found:
        print("No standard metrics found to plot.")
        return None

    n = len(found)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 3.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes[n:]:
        ax.remove()

    for i, metric in enumerate(found):
        ax = axes[i]
        ser = df[metric]
        try:
            ser.plot(ax=ax, marker="o")
        except Exception:
            ax.plot(ser.values)
        ax.set_title(metric)
        ax.grid(True)

    plt.tight_layout()
    out_path = out_dir / f"summary_{country.lower()}_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_industries(industries_df: pd.DataFrame, out_dir: Path, country: str, top_n: int = 20):
    if industries_df is None:
        return None

    counts = industries_df["Industry"].value_counts()
    top = counts.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top[::-1].plot(kind="barh", ax=ax)
    ax.set_title(f"Top {len(top)} Industries by Firm Count — {country}")
    ax.set_xlabel("Number of firms")
    plt.tight_layout()
    out_path = out_dir / f"summary_{country.lower()}_industries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize shallow simulation outputs and plot metrics.")
    parser.add_argument("country", nargs="?", default=DEFAULT_COUNTRY, help="Country code (default: FRA)")
    parser.add_argument("h5_file", nargs="?", default=None, help="Path to shallow HDF5 file")
    parser.add_argument("--out", default="output", help="Output directory for CSV/plots (default: output)")
    args = parser.parse_args(argv)

    country = args.country.upper()
    if args.h5_file:
        h5_path = Path(args.h5_file)
    else:
        h5_path = Path(__file__).resolve().parent.parent / "output" / f"sim_{country.lower()}_shallow.h5"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary_df, industries_df = load_shallow(h5_path, country)
    except Exception as e:
        print(f"Error loading HDF5: {e}")
        return 2

    print(f"Loaded summary for {country} — shape: {summary_df.shape}")

    csv_path = save_csv(summary_df, out_dir, country)
    print(f"Saved CSV: {csv_path}")

    ts_path = plot_timeseries(summary_df, out_dir, country)
    if ts_path:
        print(f"Saved timeseries plot: {ts_path}")

    ind_path = plot_industries(industries_df, out_dir, country)
    if ind_path:
        print(f"Saved industries plot: {ind_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
