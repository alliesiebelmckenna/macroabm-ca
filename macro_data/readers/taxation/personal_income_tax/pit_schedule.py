"""Progressive Personal Income Tax (PIT) schedule computation.

This module provides two layers:

1. **Standalone functions** ``compute_progressive_tax`` /
   ``compute_progressive_tax_quick`` — low-level vectorized tax
   computation from arbitrary threshold/rate arrays.

2. **PITSchedule class** — reads a local CSV of per-year bracket
   definitions and returns the published brackets for a requested year
   (statutory lookup).  It does not project past the published years —
   a simulation year beyond the last one in the CSV is an error (see
   ``CentralGovernment.set_pit_for_year``); the CSV itself is expected to
   carry an explicit row for every year the schedule is known to hold,
   including years where indexation is legislatively frozen.

CSV format (consolidated ``rates_thresholds.csv``)
--------------------------------------------------
The CSV must contain the following columns (case-insensitive)::

    tax_year  int     Taxation year the row applies to (e.g. 2014).
    geo       str     Jurisdiction key (e.g. "BC"); rows are filtered to the
                      requested jurisdiction on read.
    lower     float   Nominal lower income boundary for that year.
    rate      float   Marginal tax rate applied within this bracket (0–1).
    index     int     Flag: 1 = the bound is statutorily indexed; 0 = held
                      nominal.  Recorded for reference; not used to compute
                      anything (there is no forward projection).

On read, the consolidated columns are mapped to the internal field names
(``lower_bound`` / ``marginal_rate`` / ``indexing``).  Brackets are ordered by
``lower`` ascending; the row with the smallest ``lower`` (0) is the first
bracket.  Each bracket spans [lower_k, lower_{k+1}] (or [lower_k, ∞) for the
highest bracket).  No explicit bracket-ordinal column is required — the
ordinal is implied by the sorted lower bounds (a new tax-year block restarts
at ``lower = 0``).

Non-refundable tax credits (including the basic personal amount) are
supplied separately via the companion ``non_refundable_tax_credits.csv`` (see
:class:`TaxCreditSchedule`), not in this bracket CSV.

Example::

    tax_year,geo,lower,rate,index
    2014,BC,0.0,0.0506,1
    2014,BC,37606.0,0.077,1
    ...

Lookup-only
-----------
``get_brackets(tax_year=T)`` returns the actual published rows for *T* —
bounds and rates — when *T* is one of the schedule's years, and raises
otherwise.  Quick-add values are recomputed from each year's published
bounds so they remain self-consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from macro_data.readers.taxation.personal_income_tax.tax_credit_schedule import (
    TaxCreditSchedule,
)

# ── required CSV columns (consolidated ``rates_thresholds.csv`` format) ──
_REQUIRED_COLS = {
    "tax_year",  # int — taxation year the row applies to
    "geo",       # str — jurisdiction key (e.g. "BC"); rows are geo-filtered
    "lower",     # float — nominal lower income boundary
    "rate",      # float — marginal rate for this bracket
    "index",     # bool (0/1) — whether the bound is indexed in projection
}

# Boundary mapping from the consolidated CSV columns to the internal field
# names used throughout the pipeline (applied unconditionally on read).
_BRACKET_COLUMN_MAP = {
    "lower": "lower_bound",
    "rate": "marginal_rate",
    "index": "indexing",
}

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Low-level vectorized computation
# =====================================================================

def compute_progressive_tax(
    incomes: np.ndarray,
    thresholds: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """Compute progressive tax using marginal rates on income slices.

    Income in each bracket [lower, upper] is taxed at the bracket's
    marginal rate.  Income exactly equal to a boundary is assigned
    to the **lower** bracket.  The last threshold should be
    ``np.inf`` to capture all remaining income.

    Args:
        incomes: Shape (n,) — taxable income per individual.
        thresholds: Shape (k,) — bracket *upper* bounds.  Must be
            strictly increasing; last entry conventionally ``np.inf``.
        rates: Shape (k,) — marginal tax rate for each bracket [0, 1].

    Returns:
        Shape (n,) — tax owed per individual.
    """
    _validate_brackets(thresholds, rates)

    tax = np.zeros_like(incomes, dtype=float)
    lower = 0.0
    for threshold, rate in zip(thresholds, rates):
        amount_in_bracket = np.clip(incomes, lower, threshold) - lower
        tax += rate * np.maximum(amount_in_bracket, 0.0)
        lower = threshold
    return tax


def compute_progressive_tax_quick(
    incomes: np.ndarray,
    lower_bounds: np.ndarray,
    marginal_rates: np.ndarray,
    quick_adds: np.ndarray,
) -> np.ndarray:
    """Compute progressive tax using pre-computed cumulative quick-add values.

    For each income *x*, find the highest bracket *b* where
    ``x >= lower_bounds[b]``, then::

        tax = quick_adds[b] + marginal_rates[b] * (x - lower_bounds[b])

    Args:
        incomes: Shape (n,) — taxable income per individual.
        lower_bounds: Shape (k,) — lower income boundary of each bracket.
        marginal_rates: Shape (k,) — marginal rate for each bracket.
        quick_adds: Shape (k,) — cumulative tax from all brackets
            below the current one.

    Returns:
        Shape (n,) — tax owed per individual.
    """
    if not (len(lower_bounds) == len(marginal_rates) == len(quick_adds)):
        raise ValueError(
            "lower_bounds, marginal_rates, and quick_adds must have the same length"
        )

    bracket_idx = np.searchsorted(lower_bounds, incomes, side="right") - 1
    bracket_idx = np.clip(bracket_idx, 0, len(lower_bounds) - 1)

    tax = quick_adds[bracket_idx] + marginal_rates[bracket_idx] * (
        incomes - lower_bounds[bracket_idx]
    )
    return np.maximum(tax, 0.0)


# =====================================================================
# 2. PITSchedule — CSV-backed multi-year schedule (statutory lookup)
# =====================================================================

class PITSchedule:
    """Progressive PIT schedule backed by a per-year bracket CSV.

    Typical usage::

        schedule = PITSchedule.from_csv("rates_thresholds.csv")

        # Get the published brackets for a listed year (statutory lookup)
        thresholds, rates, *_ = schedule.get_brackets(tax_year=2016)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tax_credits: Optional["TaxCreditSchedule"] = None,
    ) -> None:
        self._df = df.copy()
        # Minimum year, not the first row's — a valid but unsorted CSV must
        # not silently shift the base year (it feeds the "before base year"
        # validation in get_brackets).
        self._base_year: int = int(self._df["tax_year"].min())
        self._tax_credits: Optional["TaxCreditSchedule"] = tax_credits

    # ── factories ───────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        jurisdiction: str = "bc",
    ) -> "PITSchedule":
        """Load bracket definitions from a CSV file.

        Args:
            path: Path to the CSV file.
            jurisdiction: Jurisdiction key used when the CSV is geo-keyed.

        Returns:
            A configured ``PITSchedule`` instance.
        """
        df = pd.read_csv(Path(path))

        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)

        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)} "
                f"(consolidated rates_thresholds format). "
                f"Found columns: {sorted(df.columns)}"
            )

        geo = jurisdiction.upper()
        df = df[df["geo"].astype(str).str.upper() == geo].copy()
        if df.empty:
            raise ValueError(
                f"CSV {path} does not contain any rows for geo {geo}"
            )

        # Boundary mapping: consolidated column names -> internal field names.
        df = df.rename(columns=_BRACKET_COLUMN_MAP)

        df = df.dropna(subset=["tax_year", "lower_bound", "marginal_rate", "indexing"])

        df["tax_year"] = df["tax_year"].astype(int)
        for col in ("lower_bound", "marginal_rate"):
            df[col] = df[col].astype(float)
        df["indexing"] = df["indexing"].astype(bool)

        return cls(df)

    @classmethod
    def from_name(
        cls,
        filename: str,
        schedule_dir: Path,
        jurisdiction: str = "bc",
    ) -> "PITSchedule":
        """Load a schedule by filename from *schedule_dir*.

        Args:
            filename: Bracket CSV filename (``"rates_thresholds.csv"``).
            schedule_dir: Directory holding the schedule CSVs — typically
                ``raw_data_path / "taxation" / "personal_income_tax"``.
            jurisdiction: Jurisdiction key used when the CSV is geo-keyed.
        """
        schedule_dir = Path(schedule_dir)
        path = schedule_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Schedule file not found: {path}\n"
                f"Available: {sorted([p.name for p in schedule_dir.glob('*.csv')])}"
            )
        schedule = cls.from_csv(path, jurisdiction=jurisdiction)
        schedule._load_companion_tax_credits(path, jurisdiction=jurisdiction)
        return schedule

    # ── companion tax-credit file auto-discovery ────────────────────

    def _load_companion_tax_credits(
        self,
        bracket_path: Path,
        jurisdiction: str = "bc",
    ) -> None:
        """Load the companion ``non_refundable_tax_credits.csv`` when present.

        The credit file is optional — when not found the model applies no tax
        credits (none configured).
        """
        candidate = bracket_path.parent / "non_refundable_tax_credits.csv"
        if not candidate.exists():
            logger.debug("No companion tax-credit file found (%s)", candidate)
            return

        logger.info("Loading companion tax-credit file: %s", candidate.name)
        self._tax_credits = TaxCreditSchedule.from_csv(
            candidate, jurisdiction=jurisdiction
        )

    # ── properties ──────────────────────────────────────────────────

    @property
    def base_year(self) -> int:
        """The base tax year from the CSV (all bounds are nominal for this year)."""
        return self._base_year

    @property
    def available_years(self) -> np.ndarray:
        """Sorted unique tax years present in the schedule."""
        return np.sort(self._df["tax_year"].unique())

    @property
    def tax_credits(self) -> Optional["TaxCreditSchedule"]:
        """Tax credit schedule (multi-component), or ``None`` if not loaded.

        When a companion ``*_tax_credit_*.csv`` file was found alongside
        the bracket CSV, it is parsed into a ``TaxCreditSchedule``.
        Otherwise this is ``None`` and no tax credits are applied.
        """
        return self._tax_credits

    # ── public methods ──────────────────────────────────────────────

    def get_brackets(
        self,
        tax_year: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the published bracket arrays for *tax_year* (statutory lookup).

        Lookup-only: a year present in the schedule returns its OWN published
        rows — actual bounds and rates — and ``quick_add`` values are recomputed
        from them.  A year not in the schedule raises; there is no forward
        projection — the CSV must carry an explicit row for every year the
        schedule is known to hold (including legislatively frozen years).

        Args:
            tax_year: The tax year to retrieve.

        Returns:
            Tuple ``(thresholds, rates, lower_bounds, quick_adds)``.

        Raises:
            ValueError: If *tax_year* is before the base year, or is not one of
                the published years.
        """
        # Bracket order is implied by ascending lower_bound (the former
        # explicit ``step`` column); the first bracket starts at 0.
        years_present = {int(y) for y in self._df["tax_year"].unique()}

        if tax_year in years_present:
            # ── Statutory lookup ──────────────────────────────────────────
            # A year present in the table returns its OWN published rows —
            # actual bounds AND rates — so per-year rate changes, freezes, and
            # re-basings come through exactly, with no compounding/double-count.
            # (Filter to this year's rows only; never sort across all years.)
            year_df = self._df[self._df["tax_year"] == tax_year].sort_values(
                "lower_bound"
            )
            lower_bounds = year_df["lower_bound"].values.astype(float).copy()
            marginal_rates = year_df["marginal_rate"].values.astype(float).copy()
        elif tax_year < self.base_year:
            raise ValueError(
                f"tax_year {tax_year} is before base year {self.base_year}"
            )
        else:
            # ── Out of table (year not published) ──────────────────────────
            # The reader is lookup-only: it returns a year's actual published
            # rows or nothing.  There is no forward projection — a schedule
            # meant to cover a later year (including a legislated freeze) must
            # carry an explicit row for it.
            raise ValueError(
                f"tax_year {tax_year} is not in the published schedule "
                f"(available years: {sorted(years_present)}). The reader is "
                f"lookup-only and does not project past the published years; "
                f"add an explicit row for {tax_year} to the schedule CSV."
            )

        quick_adds = _recompute_quick_add(lower_bounds, marginal_rates)
        thresholds = np.append(lower_bounds[1:].copy(), np.inf)

        return thresholds, marginal_rates, lower_bounds, quick_adds

    def compute_tax(
        self,
        incomes: np.ndarray,
        tax_year: int,
    ) -> np.ndarray:
        """Compute progressive tax for a given year (convenience wrapper)."""
        thresholds, rates, _, _ = self.get_brackets(tax_year)
        return compute_progressive_tax(incomes, thresholds, rates)


# =====================================================================
# Internal helpers
# =====================================================================

def _validate_brackets(thresholds: np.ndarray, rates: np.ndarray) -> None:
    """Check threshold / rate invariants."""
    if len(thresholds) != len(rates):
        raise ValueError(
            f"thresholds and rates must have the same length, "
            f"got {len(thresholds)} and {len(rates)}"
        )
    if len(thresholds) > 1 and not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly increasing")
    if np.any(rates < 0) or np.any(rates > 1):
        raise ValueError("rates must be in [0, 1]")


def _recompute_quick_add(
    lower_bounds: np.ndarray,
    marginal_rates: np.ndarray,
) -> np.ndarray:
    """Recompute quick-add values from lower_bounds and marginal_rates."""
    quick = np.zeros(len(lower_bounds), dtype=float)
    for i in range(1, len(lower_bounds)):
        quick[i] = quick[i - 1] + marginal_rates[i - 1] * (
            lower_bounds[i] - lower_bounds[i - 1]
        )
    return quick

