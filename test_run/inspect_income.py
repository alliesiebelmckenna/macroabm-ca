"""Inspect income distribution at initial state from a simulation HDF5 output.

Usage:
    uv run python inspect_income.py              # defaults to FRA
    uv run python inspect_income.py CAN          # Canada
    uv run python inspect_income.py FRA output/custom.h5  # custom file
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def gini_coefficient(x):
    """Compute Gini coefficient from an array of values."""
    x = np.sort(x[x > 0])
    n = len(x)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


REPO_ROOT = Path(__file__).resolve().parent.parent

# Default scale factor — each synthetic agent represents this many real people/households.
# Set by the `scale` parameter in `default_data_configuration()` (default: 10,000).
SCALE = 10_000

# --- Parse command line ---
COUNTRY = sys.argv[1] if len(sys.argv) > 1 else "FRA"
custom_h5 = sys.argv[2] if len(sys.argv) > 2 else None

if custom_h5:
    H5_PATH = Path(custom_h5)
else:
    H5_PATH = REPO_ROOT / f"output/sim_{COUNTRY.lower()}.h5"

if not H5_PATH.exists():
    alt = REPO_ROOT / "output/sim_fra.h5"
    print(f"ERROR: {H5_PATH} not found.")
    if alt.exists():
        print(f"  (Found {alt} — try running with 'FRA' or pass a custom path as 2nd arg)")
    sys.exit(1)


with h5py.File(H5_PATH, "r") as f:
    group = f[COUNTRY]
    print(f"=== Groups in {COUNTRY} ===")
    for key in group.keys():
        print(f"  {key}")

    ind = group["individuals"]
    hh = group["households"]

    # ---- INDIVIDUAL INCOME (timestep 0 = initial) ----
    ind_income = ind["income"][0]
    ind_emp_income = ind["employee_income"][0]

    print(f"\n===== INDIVIDUAL INCOME (initial state) =====")
    print(f"  NOTE: Each synthetic individual agent represents ~{SCALE:,} real people.")
    print(f"  All income values are aggregated at the synthetic-agent level, not per-capita.")
    print(f"  N = {len(ind_income):,} individuals")
    print(f"  Mean:   {ind_income.mean():,.6f}")
    print(f"  Median: {np.median(ind_income):,.6f}")
    print(f"  Std:    {ind_income.std():,.6f}")
    print(f"  Min:    {ind_income.min():,.6f}")
    print(f"  Max:    {ind_income.max():,.6f}")
    print(f"  Gini:   {gini_coefficient(ind_income):.4f}")
    print(f"\n  Deciles:")
    for p, v in zip(np.arange(0, 101, 10), np.percentile(ind_income, np.arange(0, 101, 10))):
        print(f"    {p:3d}%: {v:,.6f}")
    print(f"\n  Income composition:")
    print(f"    Employed (income>0): {(ind_income > 0).sum()} ({(ind_income > 0).mean()*100:.1f}%)")
    zero = (ind_income == 0).sum()
    print(f"    Zero income:         {zero} ({zero/len(ind_income)*100:.1f}%)")
    print(f"    Mean employee income (employed): {ind_emp_income[ind_emp_income > 0].mean():,.6f}")

    # ---- HOUSEHOLD INCOME (timestep 0 = initial) ----
    hh_income = hh["income"][0]

    print(f"\n===== HOUSEHOLD INCOME (initial state) =====")
    print(f"  NOTE: Each synthetic household represents ~{SCALE:,} real households.")
    print(f"  N = {len(hh_income):,} households")
    print(f"  Mean:   {hh_income.mean():,.6f}")
    print(f"  Median: {np.median(hh_income):,.6f}")
    print(f"  Std:    {hh_income.std():,.6f}")
    print(f"  Min:    {hh_income.min():,.6f}")
    print(f"  Max:    {hh_income.max():,.6f}")
    print(f"  Gini:   {gini_coefficient(hh_income):.4f}")
    print(f"\n  Deciles:")
    for p, v in zip(np.arange(0, 101, 10), np.percentile(hh_income, np.arange(0, 101, 10))):
        print(f"    {p:3d}%: {v:,.6f}")


# ---- Lorenz curve helper ----
def lorenz_curve(x):
    """Return cumulative population share and cumulative income share."""
    x = np.sort(x[x > 0])
    if len(x) == 0:
        return np.array([0, 1]), np.array([0, 1])
    cum_pop = np.linspace(0, 1, len(x))
    cum_inc = np.cumsum(x) / np.sum(x)
    return cum_pop, cum_inc


# ---- Plots ----
fig, axes = plt.subplots(2, 3, figsize=(21, 12))

# === ROW 1: INDIVIDUALS ===

# Column 1: Individual income histogram (log scale)
ax = axes[0, 0]
pos_income = ind_income[ind_income > 0]
log_bins = np.logspace(np.log10(pos_income.min()), np.log10(pos_income.max()), 50)
ax.hist(pos_income, bins=log_bins, edgecolor="white", alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel(f"Income — log scale (1 agent ≈ {SCALE:,} people)")
ax.set_ylabel("Number of Individuals")
ax.set_title(f"Individual Income — {COUNTRY} (initial state)")
ax.axvline(np.median(ind_income), color="red", linestyle="--", label=f"Median: {np.median(ind_income):.4f}")
ax.axvline(ind_income.mean(), color="green", linestyle="--", label=f"Mean: {ind_income.mean():.4f}")
ax.legend()

# Column 2: Individual decile bar chart
ax = axes[0, 1]
ind_dec = np.percentile(ind_income, np.arange(0, 101, 10))
dec_labels = [f"{d}–{d+10}" for d in range(0, 100, 10)]
dec_means_ind = []
for lo, hi in zip(ind_dec[:-1], ind_dec[1:]):
    mask = (ind_income >= lo) & (ind_income < hi)
    if hi == ind_dec[-1]:
        mask = (ind_income >= lo) & (ind_income <= hi)
    dec_means_ind.append(ind_income[mask].mean() if mask.any() else 0)
ax.bar(dec_labels, dec_means_ind, edgecolor="white", alpha=0.7)
ax.set_xlabel("Income Decile")
ax.set_ylabel("Mean Income")
ax.set_title(f"Individual Mean Income by Decile — {COUNTRY}")
ax.tick_params(axis="x", rotation=45)

# Column 3: Individual Lorenz curve
ax = axes[0, 2]
pop_i, inc_i = lorenz_curve(ind_income)
ax.plot(pop_i, inc_i, linewidth=2, label=f"Gini = {gini_coefficient(ind_income):.4f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect equality")
ax.fill_between(pop_i, pop_i, inc_i, alpha=0.1)
ax.set_xlabel("Cumulative population share")
ax.set_ylabel("Cumulative income share")
ax.set_title(f"Individual Lorenz Curve — {COUNTRY}")
ax.legend(loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# === ROW 2: HOUSEHOLDS ===

# Column 1: Household income histogram (log scale)
ax = axes[1, 0]
pos_hh = hh_income[hh_income > 0]
log_bins_h = np.logspace(np.log10(pos_hh.min()), np.log10(pos_hh.max()), 50)
ax.hist(pos_hh, bins=log_bins_h, edgecolor="white", alpha=0.7, color="orange")
ax.set_xscale("log")
ax.set_xlabel(f"Income — log scale (1 agent ≈ {SCALE:,} households)")
ax.set_ylabel("Number of Households")
ax.set_title(f"Household Income — {COUNTRY} (initial state)")
ax.axvline(np.median(hh_income), color="red", linestyle="--", label=f"Median: {np.median(hh_income):,.4f}")
ax.axvline(hh_income.mean(), color="green", linestyle="--", label=f"Mean: {hh_income.mean():,.4f}")
ax.legend()

# Column 2: Household decile bar chart
ax = axes[1, 1]
hh_dec = np.percentile(hh_income, np.arange(0, 101, 10))
dec_means_hh = []
for lo, hi in zip(hh_dec[:-1], hh_dec[1:]):
    mask = (hh_income >= lo) & (hh_income < hi)
    if hi == hh_dec[-1]:
        mask = (hh_income >= lo) & (hh_income <= hi)
    dec_means_hh.append(hh_income[mask].mean() if mask.any() else 0)
ax.bar(dec_labels, dec_means_hh, edgecolor="white", alpha=0.7, color="orange")
ax.set_xlabel("Income Decile")
ax.set_ylabel("Mean Income")
ax.set_title(f"Household Mean Income by Decile — {COUNTRY}")
ax.tick_params(axis="x", rotation=45)

# Column 3: Household Lorenz curve
ax = axes[1, 2]
pop_h, inc_h = lorenz_curve(hh_income)
ax.plot(pop_h, inc_h, linewidth=2, color="orange", label=f"Gini = {gini_coefficient(hh_income):.4f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect equality")
ax.fill_between(pop_h, pop_h, inc_h, alpha=0.1, color="orange")
ax.set_xlabel("Cumulative population share")
ax.set_ylabel("Cumulative income share")
ax.set_title(f"Household Lorenz Curve — {COUNTRY}")
ax.legend(loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plot_path = REPO_ROOT / "output" / f"income_distribution_{COUNTRY.lower()}.png"
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {plot_path}")
plt.show()