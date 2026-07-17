"""
Module for producing a Canadian personal income tax (PIT) schedule.

This module produces the ``PITSchedule`` class, which reads, processes and stores year- and jurisdiction-specific personal income tax (PIT) parameters like tax rates and tax bracket thresholds that are used to compute income tax amounts payable by individuals in a progressive tax system. 

A tax is "progressive" if an individual's income in higher tax brackets (i.e., exceeding certain thresholds) is taxed at a higher marginal rate than their income in lower brackets, such that their average tax rate is less than the marginal tax rate corresponding to their income. "Jurisdictions" here refer to whether taxes are provincial, territorial, or federal.

If supplied, ``PITSchedule`` also incorporates year- and jurisdiction-specific non-refundable tax credit (NRTC) parameters like credit amounts, clawback rates and eligibility criteria that can be used to reduce an individual's income tax payable. More information on NRTC functionality can be found in the documentation for ``TaxCreditComponent`` and ``TaxCreditSchedule``.

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
    bc_pit_schedule.load_tax_credits("path/to/pit/credits", jurisdiction=jurisdiction)

    # View dictionary containing NRTC parameters
    bc_pit_schedule._tax_credits.credits
    ```
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

# For this functionality to work as expected, the user must supply a CSV file named "rates_thresholds.csv" that contains the following columns (case-sensitive):
_REQUIRED_COLS = {
    "tax_year",  # tax parameter year
    "geo",       # jurisdiction key (e.g. "BC")
    "lower",     # nominal lower income threshold
    "rate",      # corresponding marginal tax rate
    "index",     # whether tax brackets grow with inflation over time (1) or stay the same (0)
}

logger = logging.getLogger(__name__)

class PITSchedule:
    """
    Personal income tax (PIT) schedule under a progressive tax system using year- and jurisdiction-specific PIT parameters supplied by the user as raw data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tax_credits: Optional["TaxCreditSchedule"] = None,
    ) -> None:
        self._df = df.copy()
        # First year of simulation run
        self._base_year: int = int(self._df["tax_year"].min())
        self._tax_credits: Optional["TaxCreditSchedule"] = tax_credits

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

        geo = jurisdiction.upper()
        df = df[df["geo"].astype(str).str.upper() == geo].copy()
        if df.empty:
            raise ValueError(
                f"CSV {path} does not contain any data for jurisdiction {geo}"
            )

        df = df.dropna(subset=["tax_year", "lower", "rate", "index"])

        df["tax_year"] = df["tax_year"].astype(int)
        for col in ("lower", "rate"):
            df[col] = df[col].astype(float)
        df["index"] = df["index"].astype(bool)

        return cls(df)

    def load_tax_credits(
        self,
        credits_path: Optional[Path],
        jurisdiction: str,
    ) -> None:
        """If available, add year- and jurisdiction-specific tax credits to ``PITSchedule``."""

        # If path does not exist
        if credits_path is None or not Path(credits_path).exists():
            logger.debug("No tax credit parameter file supplied (%s)", credits_path)
            self._tax_credits = None
            return

        credits_path = Path(credits_path)
        logger.info("Loading tax credit parameter file: %s", credits_path.name)
        try:
            self._tax_credits = TaxCreditSchedule.from_csv(
                credits_path, jurisdiction=jurisdiction
            )
        except ValueError:
            # Tax credit parameter file exists but has no data for this jurisdiction.
            logger.debug(
                "Tax credit parameter file %s has no data for %s. No tax credits have been applied.",
                credits_path.name,
                jurisdiction,
            )
            self._tax_credits = None

    @property
    def base_year(self) -> int:
        """Simulation start year."""
        return self._base_year

    @property
    def available_years(self) -> np.ndarray:
        """Sorted unique years in PIT parameters."""
        return np.sort(self._df["tax_year"].unique())

    @property
    def tax_credits(self) -> Optional["TaxCreditSchedule"]:
        """
        Tax credit parameters or ``None`` if not loaded.

        When there is a ``*_tax_credit*.csv`` file located in the same directory as PIT parameters, tax credit parameters found within that file are passed into the ``TaxCreditSchedule`` class to be used by ``PITSchedule``. If no file is found, no tax credits are added to ``PITSchedule``.
        """
        return self._tax_credits

    def get_brackets(
        self,
        tax_year: int,
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
        years_present = {int(y) for y in self._df["tax_year"].unique()}

        # Pull PIT parameters if data are available (i.e., if user-specified year within bounds) 
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

        # Array of upper thresholds = lower thresholds (minus lowest lower bound i.e., 0) and np.inf (i.e., upper bound of highest tax bracket)
        thresholds = np.append(lowers[1:].copy(), np.inf)

        return lowers, thresholds, rates

def compute_progressive_tax(
    incomes: np.ndarray,
    thresholds: np.ndarray,
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
    _validate_brackets(thresholds, rates)

    tax = np.zeros_like(incomes, dtype=float)
    lower = 0.0
    for threshold, rate in zip(thresholds, rates):
        amount_in_bracket = np.clip(incomes, lower, threshold) - lower
        tax += rate * np.maximum(amount_in_bracket, 0.0)
        lower = threshold
    return tax

    def compute_tax(
        self,
        incomes: np.ndarray,
        tax_year: int,
    ) -> np.ndarray:
        """Compute personal income tax for a given year (convenience wrapper)."""
        thresholds, rates, _, _ = self.get_brackets(tax_year)
        return compute_progressive_tax(incomes, thresholds, rates)

def _validate_brackets(thresholds: np.ndarray, rates: np.ndarray) -> None:
    """
    Ensure that:
    1. Each tax bracket has a corresponding marginal tax rate,
    2. Upper thresholds of tax brackets are strictly increasing, and
    3. Tax rates are in [0.0, 1.0]
    """
    if len(thresholds) != len(rates):
        raise ValueError(
            f"thresholds and rates must have the same length, "
            f"got {len(thresholds)} and {len(rates)}"
        )
    if len(thresholds) > 1 and not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly increasing")
    if np.any(rates < 0) or np.any(rates > 1):
        raise ValueError("rates must be in [0, 1]")

