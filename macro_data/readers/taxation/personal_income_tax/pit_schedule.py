"""
Module for producing a Canadian personal income tax (PIT) schedule.

This module produces the ``PITSchedule`` class, which reads, processes and stores year- and jurisdiction-specific personal income tax (PIT) parameters like tax rates and tax bracket thresholds that are used to compute income tax amounts payable by individuals in a progressive tax system.

A tax is "progressive" if an individual's income in higher tax brackets (i.e., exceeding certain thresholds) is taxed at a higher marginal rate than their income in lower brackets, such that their average tax rate is less than the marginal tax rate corresponding to their income. "Jurisdictions" here refer to whether taxes are provincial, territorial, or federal.

If supplied, ``PITSchedule`` also incorporates year- and jurisdiction-specific non-refundable tax credit (NRTC) parameters like credit amounts, clawback rates and eligibility criteria that can be used to reduce an individual's income tax payable. More information on NRTC functionality can be found in the documentation for ``TaxCreditComponent`` and ``NRTCSchedule``.

``PITSchedule`` is used by ``TaxationReader`` which is then used to initialize the Central Government agent in the ``macromodel`` package of MacroABM-CA.

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.taxation.personal_income_tax.pit_schedule import PITSchedule

    # To pull historical B.C. PIT parameters
    # Note: to pull federal PIT parameters, use "CA"
    jurisdiction = "BC"

    bc_pit_schedule = PITSchedule.from_csv("path/to/pit/parameters", jurisdiction=jurisdiction)

    # View DataFrame containing PIT parameters
    bc_pit_schedule._df

    # To load NRTC parameters
    bc_pit_schedule.load_non_refundable_tax_credits("path/to/pit/credits", jurisdiction=jurisdiction)

    # View dictionary containing NRTC parameters
    bc_pit_schedule._non_refundable_tax_credits.credits
    ```
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from macro_data.readers.taxation.personal_income_tax.nrtc_schedule import (
    NRTCSchedule,
)

# For this functionality to work as expected, the user must supply a CSV file named "rates_thresholds.csv" that contains the following columns (case-sensitive):
_REQUIRED_COLS = {
    "year",  # tax parameter year
    "jurisdiction",  # jurisdiction key (e.g. "BC")
    "lower",  # nominal lower income threshold
    "rate",  # corresponding marginal tax rate
    "index",  # schema check only: required for shape, no longer read
}

logger = logging.getLogger(__name__)


class PITSchedule:
    """
    Personal income tax (PIT) schedule under a progressive tax system using year- and jurisdiction-specific PIT parameters supplied by the user as raw data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        non_refundable_tax_credits: Optional["NRTCSchedule"] = None,
    ) -> None:
        self._df = df.copy()
        # First year of simulation run
        self._start_year: int = int(self._df["year"].min())
        self._non_refundable_tax_credits: Optional["NRTCSchedule"] = non_refundable_tax_credits

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        jurisdiction: str,
    ) -> "PITSchedule":
        """
        Load jurisdiction-specific PIT parameters.

        Args:
            path: Name of CSV file containing PIT parameters (``"rates_thresholds.csv"``).
            jurisdiction: Denotes whether PIT parameters listed are provincial/territorial or federal (e.g., "BC", "CA" for federal, etc.)

        Returns:
            Configured ``PITSchedule`` instance
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

        juris = jurisdiction.upper()
        df = df[df["jurisdiction"].astype(str).str.upper() == juris].copy()
        if df.empty:
            raise ValueError(f"CSV {path} does not contain any data for jurisdiction {juris}")

        df = df.dropna(subset=["year", "lower", "rate", "index"])

        df["year"] = df["year"].astype(int)
        for col in ("lower", "rate"):
            df[col] = df[col].astype(float)

        return cls(df)

    def load_non_refundable_tax_credits(
        self,
        nrtc_path: Optional[Path],
        jurisdiction: str,
    ) -> None:
        """If available, add year- and jurisdiction-specific tax credits to ``PITSchedule``."""

        # If path does not exist
        if nrtc_path is None or not Path(nrtc_path).exists():
            logger.debug("No tax credit parameter file supplied (%s)", nrtc_path)
            self._non_refundable_tax_credits = None
            return

        nrtc_path = Path(nrtc_path)
        logger.info("Loading tax credit parameter file: %s", nrtc_path.name)
        try:
            self._non_refundable_tax_credits = NRTCSchedule.from_csv(nrtc_path, jurisdiction=jurisdiction)
        except ValueError:
            # Tax credit parameter file exists but has no data for this jurisdiction.
            logger.debug(
                "Tax credit parameter file %s has no data for %s. No tax credits have been applied.",
                nrtc_path.name,
                jurisdiction,
            )
            self._non_refundable_tax_credits = None

    @property
    def start_year(self) -> int:
        """Simulation start year."""
        return self._start_year

    @property
    def available_years(self) -> np.ndarray:
        """Sorted unique years in PIT parameters."""
        return np.sort(self._df["year"].unique())

    @property
    def non_refundable_tax_credits(self) -> Optional["NRTCSchedule"]:
        """
        Non-refundable tax credit parameters or ``None`` if not loaded.

        When there is a ``*non_refundable_tax_credits*.csv`` file located in the same directory as PIT parameters, tax credit parameters found within that file are passed into the ``NRTCSchedule`` class to be used by ``PITSchedule``. If no file is found, no tax credits are added to ``PITSchedule``.
        """
        return self._non_refundable_tax_credits

    def get_brackets(
        self,
        year: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a tuple containing year-specific PIT parameters (upper and lower income thresholds and corresponding marginal tax rates)

        Args:
            year: Year for which to pull PIT parameters

        Returns:
            Tuple ``(lowers, uppers, rates)``.

        Raises:
            ValueError: If user-specified year is out of bounds.
        """
        years_present = {int(y) for y in self._df["year"].unique()}

        # Pull PIT parameters if data are available (i.e., if user-specified year within bounds)
        if year in years_present:
            # Filter to this year's rows only; never sort across all years.
            year_df = self._df[self._df["year"] == year].sort_values("lower")
            lowers = year_df["lower"].values.astype(float).copy()
            rates = year_df["rate"].values.astype(float).copy()
        else:
            raise ValueError(
                f"No PIT data available for year {year} "
                f"(available years: {sorted(years_present)}). The reader is "
                f"lookup-only and does not project beyond the published years; "
                f"add an explicit row for {year} to the schedule CSV."
            )

        # Array of upper thresholds = lower thresholds (minus lowest lower bound i.e., 0) and np.inf (i.e., upper bound of highest tax bracket)
        uppers = np.append(lowers[1:].copy(), np.inf)

        return lowers, uppers, rates


def compute_personal_income_tax(
    incomes: np.ndarray,
    uppers: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """
    Compute personal income tax (PIT) payable using a progressive tax schedule.

    Each portion of an individual's income in different tax brackets (each defined by a lower and upper threshold) is taxed at corresponding marginal tax rates that increase for income exceeding pre-defined thresholds.

    Args:
        incomes: array of individual-level incomes
        uppers: array of tax bracket upper thresholds (strictly increasing, ending in ``np.inf``)
        rates: array of marginal tax rates corresponding to each tax bracket [0.0, 1.0].

    Returns:
        array of individual-level income tax payable amounts
    """
    # Fail fast on malformed brackets: equal lengths, increasing uppers, rates in [0, 1].
    _validate_brackets(uppers, rates)

    tax = np.zeros_like(incomes, dtype=float)
    lower = 0.0
    for upper, rate in zip(uppers, rates):
        amount_in_bracket = np.clip(incomes, lower, upper) - lower
        tax += rate * np.maximum(amount_in_bracket, 0.0)
        lower = upper
    return tax


def _validate_brackets(uppers: np.ndarray, rates: np.ndarray) -> None:
    """
    Ensure that:
    1. Each tax bracket has a corresponding marginal tax rate,
    2. Upper thresholds of tax brackets are strictly increasing, and
    3. Tax rates are in [0.0, 1.0]
    """
    if len(uppers) != len(rates):
        raise ValueError(f"uppers and rates must have the same length, got {len(uppers)} and {len(rates)}")
    # NaN passes every comparison below, so it would validate and then poison the population.
    if np.isnan(np.asarray(uppers, dtype=float)).any() or np.isnan(np.asarray(rates, dtype=float)).any():
        raise ValueError("bracket thresholds and rates must not be NaN")
    if len(uppers) > 1 and not np.all(np.diff(uppers) > 0):
        raise ValueError("uppers must be strictly increasing")
    if np.any(rates < 0) or np.any(rates > 1):
        raise ValueError("rates must be in [0, 1]")
