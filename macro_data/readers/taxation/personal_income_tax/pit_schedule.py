"""Module for producing a personal income tax (PIT) schedule.

This module reads and processes year-specific PIT parameters like tax rates
and tax bracket thresholds to compute income tax amounts in a progressive
taxation system. A tax is "progressive" if an individual's income in higher
tax brackets (i.e., exceeding certain thresholds) is taxed at a higher
marginal rate than their income in lower brackets. The ``PITSchedule`` class
produced by this module is used in ``TaxationReader`` which is then used to
initialize the Central Government agent in the Macromodel package of
MacroABM-CA.

Bracket parameters are read from a consolidated ``rates_thresholds.csv`` keyed
by tax year and jurisdiction. Lookups are statutory only: a requested year
must be published in the CSV, since the schedule is never projected past the
years it records. Non-refundable tax credits are supplied separately by
``TaxCreditSchedule``.
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

# Required columns in the consolidated rates_thresholds.csv.
_REQUIRED_COLS = {
    "tax_year",  # taxation year the row applies to
    "geo",       # jurisdiction key (e.g. "BC")
    "lower",     # nominal lower income boundary
    "rate",      # marginal rate for this bracket
    "index",     # whether the bound is statutorily indexed
}

logger = logging.getLogger(__name__)


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
    lowers: np.ndarray,
    rates: np.ndarray,
    quick_adds: np.ndarray,
) -> np.ndarray:
    """Compute progressive tax using pre-computed cumulative quick-add values.

    For each income *x*, find the highest bracket *b* where
    ``x >= lowers[b]``, then::

        tax = quick_adds[b] + rates[b] * (x - lowers[b])

    Args:
        incomes: Shape (n,) — taxable income per individual.
        lowers: Shape (k,) — lower income boundary of each bracket.
        rates: Shape (k,) — marginal rate for each bracket.
        quick_adds: Shape (k,) — cumulative tax from all brackets
            below the current one.

    Returns:
        Shape (n,) — tax owed per individual.
    """
    if not (len(lowers) == len(rates) == len(quick_adds)):
        raise ValueError(
            "lowers, rates, and quick_adds must have the same length"
        )

    bracket_idx = np.searchsorted(lowers, incomes, side="right") - 1
    bracket_idx = np.clip(bracket_idx, 0, len(lowers) - 1)

    tax = quick_adds[bracket_idx] + rates[bracket_idx] * (
        incomes - lowers[bracket_idx]
    )
    return np.maximum(tax, 0.0)


class PITSchedule:
    """Progressive PIT schedule backed by a per-year bracket CSV."""

    def __init__(
        self,
        df: pd.DataFrame,
        tax_credits: Optional["TaxCreditSchedule"] = None,
    ) -> None:
        self._df = df.copy()
        # Minimum year, not the first row's, so an unsorted CSV does not shift
        # the base year used by get_brackets.
        self._base_year: int = int(self._df["tax_year"].min())
        self._tax_credits: Optional["TaxCreditSchedule"] = tax_credits

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        jurisdiction: str,
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

        df = df.dropna(subset=["tax_year", "lower", "rate", "index"])

        df["tax_year"] = df["tax_year"].astype(int)
        for col in ("lower", "rate"):
            df[col] = df[col].astype(float)
        df["index"] = df["index"].astype(bool)

        return cls(df)

    @classmethod
    def from_name(
        cls,
        filename: str,
        schedule_dir: Path,
        jurisdiction: str,
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
        # Companion credit file, by convention, alongside the bracket file.
        schedule.load_tax_credits(
            path.parent / "non_refundable_tax_credits.csv", jurisdiction=jurisdiction
        )
        return schedule

    def load_tax_credits(
        self,
        credits_path: Optional[Path],
        jurisdiction: str,
    ) -> None:
        """Attach *jurisdiction*'s non-refundable credits from *credits_path*.

        Both the file and any single jurisdiction's presence within it are
        optional: a jurisdiction that publishes brackets but no non-refundable
        credits is a normal state, not an error. Either way the model applies no
        tax credits for it.
        """
        if credits_path is None or not Path(credits_path).exists():
            logger.debug("No tax-credit file supplied (%s)", credits_path)
            self._tax_credits = None
            return

        credits_path = Path(credits_path)
        logger.info("Loading tax-credit file: %s", credits_path.name)
        try:
            self._tax_credits = TaxCreditSchedule.from_csv(
                credits_path, jurisdiction=jurisdiction
            )
        except ValueError:
            # The file exists but carries no rows for this jurisdiction.
            logger.debug(
                "Tax-credit file %s has no rows for %s; no credits applied.",
                credits_path.name,
                jurisdiction,
            )
            self._tax_credits = None

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

    def get_brackets(
        self,
        tax_year: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the published bracket arrays for *tax_year* (statutory lookup).

        A year present in the schedule returns its own published bounds and
        rates; ``quick_add`` values are recomputed from them. A year not in the
        schedule raises, since the reader does not project past published years.

        Args:
            tax_year: The tax year to retrieve.

        Returns:
            Tuple ``(thresholds, rates, lowers, quick_adds)``.

        Raises:
            ValueError: If *tax_year* is before the base year, or is not one of
                the published years.
        """
        # Bracket order is implied by ascending lower; the first starts at 0.
        years_present = {int(y) for y in self._df["tax_year"].unique()}

        if tax_year in years_present:
            # Filter to this year's rows only; never sort across all years.
            year_df = self._df[self._df["tax_year"] == tax_year].sort_values("lower")
            lowers = year_df["lower"].values.astype(float).copy()
            rates = year_df["rate"].values.astype(float).copy()
        elif tax_year < self.base_year:
            raise ValueError(
                f"tax_year {tax_year} is before base year {self.base_year}"
            )
        else:
            raise ValueError(
                f"tax_year {tax_year} is not in the published schedule "
                f"(available years: {sorted(years_present)}). The reader is "
                f"lookup-only and does not project past the published years; "
                f"add an explicit row for {tax_year} to the schedule CSV."
            )

        quick_adds = _recompute_quick_add(lowers, rates)
        thresholds = np.append(lowers[1:].copy(), np.inf)

        return thresholds, rates, lowers, quick_adds

    def compute_tax(
        self,
        incomes: np.ndarray,
        tax_year: int,
    ) -> np.ndarray:
        """Compute progressive tax for a given year (convenience wrapper)."""
        thresholds, rates, _, _ = self.get_brackets(tax_year)
        return compute_progressive_tax(incomes, thresholds, rates)


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
    lowers: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """Recompute quick-add values from lowers and rates."""
    quick = np.zeros(len(lowers), dtype=float)
    for i in range(1, len(lowers)):
        quick[i] = quick[i - 1] + rates[i - 1] * (
            lowers[i] - lowers[i - 1]
        )
    return quick

