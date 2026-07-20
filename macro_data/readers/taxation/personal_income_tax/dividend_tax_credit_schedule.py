"""Module for producing a dividend tax credit schedule.

This module reads the Canadian dividend gross-up and dividend tax credit (DTC)
rates from a consolidated ``dividend_tax_credit_schedule.csv`` keyed by tax year,
jurisdiction, and dividend type (``eligible`` / ``non_eligible``), and supplies
the rates that apply in a requested year. The gross-up is federally set and
currently uniform across jurisdictions, while the DTC rate is
jurisdiction-specific; both are carried per row under the ``jurisdiction`` column, so a
jurisdiction that diverges (e.g. Quebec) can hold its own gross-up without any
schema change. Lookups are statutory only: a requested year must be published
in the CSV, since the schedule is never projected past the years it records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Required CSV columns in the consolidated dividend_tax_credit_schedule.csv.
_DTC_REQUIRED_COLS = {
    "year",              # taxation year the row applies to
    "jurisdiction",          # jurisdiction key (e.g. "BC", "CA")
    "dividend_type",         # "eligible" / "non_eligible"
    "gross_up_rate",         # taxable = (1 + rate) x cash (federal, uniform)
    "dtc_pct_of_grossed_up", # DTC as a fraction of grossed-up
}

_DIVIDEND_TYPES = ("eligible", "non_eligible")


@dataclass(frozen=True)
class DividendRates:
    """Gross-up and DTC rates for one dividend type in one tax year.

    Attributes:
        dividend_type: ``"eligible"`` or ``"non_eligible"``.
        gross_up_rate: Gross-up rate; taxable dividend = ``(1 + rate) x cash``.
        dtc_pct_of_grossed_up: DTC as a fraction of the grossed-up dividend (the
            rate the model applies).
        dtc_pct_of_actual: DTC as a fraction of the actual cash dividend, when
            recorded. ``None`` when absent from the CSV.
    """

    dividend_type: str
    gross_up_rate: float
    dtc_pct_of_grossed_up: float
    dtc_pct_of_actual: Optional[float] = None


class DividendTaxCreditSchedule:
    """Per-year dividend gross-up and DTC rates for one jurisdiction."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        jurisdiction: str,
    ) -> "DividendTaxCreditSchedule":
        """Load the dividend-rate schedule from a CSV file.

        Args:
            path: Path to the CSV (``dividend_tax_credit_schedule.csv``).
            jurisdiction: Jurisdiction key used to filter the jurisdiction rows.

        Returns:
            A configured ``DividendTaxCreditSchedule``.

        Raises:
            ValueError: If required columns are missing, or the CSV contains no
                rows for *jurisdiction*.
        """
        df = pd.read_csv(Path(path))

        # Normalise column names (lower-case, spaces to underscores).
        df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})

        missing = _DTC_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Dividend-rate CSV is missing required columns: {sorted(missing)}. "
                f"Found: {sorted(df.columns)}"
            )

        juris = jurisdiction.upper()
        df = df[df["jurisdiction"].astype(str).str.upper() == juris].copy()
        if df.empty:
            raise ValueError(
                f"Dividend-rate CSV {path} does not contain any rows for jurisdiction {juris}"
            )

        df["dividend_type"] = df["dividend_type"].astype(str).str.strip().str.lower()
        df["year"] = df["year"].astype(int)
        for col in ("gross_up_rate", "dtc_pct_of_grossed_up"):
            df[col] = df[col].astype(float)
        if "dtc_pct_of_actual" in df.columns:
            df["dtc_pct_of_actual"] = pd.to_numeric(
                df["dtc_pct_of_actual"], errors="coerce"
            )

        return cls(df)

    @classmethod
    def from_name(
        cls,
        filename: str,
        schedule_dir: Path,
        jurisdiction: str,
    ) -> "DividendTaxCreditSchedule":
        """Load the schedule by filename from *schedule_dir*.

        Args:
            filename: CSV filename (``"dividend_tax_credit_schedule.csv"``).
            schedule_dir: Directory holding the schedule CSVs — typically
                ``raw_data_path / "taxation" / "personal_income_tax"``.
            jurisdiction: Jurisdiction key used to filter the jurisdiction rows.

        Returns:
            A configured ``DividendTaxCreditSchedule``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        directory = Path(schedule_dir)
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Dividend-rate file not found: {path}\n"
                f"Available: {sorted(p.name for p in directory.glob('*.csv'))}"
            )
        return cls.from_csv(path, jurisdiction=jurisdiction)

    def get_rates(self, year: int, dividend_type: str) -> DividendRates:
        """Return the published rates for *dividend_type* in *year* (statutory lookup).

        A year present in the schedule returns its own published row. A year not
        in the schedule raises, since the reader does not project past published
        years.

        Args:
            year: The tax year to look up.
            dividend_type: ``"eligible"`` or ``"non_eligible"``.

        Returns:
            The matching :class:`DividendRates`.

        Raises:
            ValueError: If no row, or more than one row, applies.
        """
        dtype = str(dividend_type).strip().lower()
        sub = self._df[self._df["dividend_type"] == dtype]
        if sub.empty:
            raise ValueError(
                f"No rows for dividend_type '{dtype}'. "
                f"Available: {sorted(self._df['dividend_type'].unique())}"
            )

        matches = sub[sub["year"] == year]
        if matches.empty:
            raise ValueError(
                f"No '{dtype}' dividend rate row for tax year {year} "
                f"(available years: {sorted(int(y) for y in sub['year'].unique())}). "
                f"The reader is lookup-only and does not project past the published "
                f"years; add an explicit row for {year} to the schedule CSV."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Duplicate '{dtype}' dividend rate rows for tax year {year}: "
                f"{len(matches)} rows match (each year must have one row per type)."
            )

        row = matches.iloc[0]
        actual = row.get("dtc_pct_of_actual")
        return DividendRates(
            dividend_type=dtype,
            gross_up_rate=float(row["gross_up_rate"]),
            dtc_pct_of_grossed_up=float(row["dtc_pct_of_grossed_up"]),
            dtc_pct_of_actual=(
                None if actual is None or pd.isna(actual) else float(actual)
            ),
        )

    def get_year_rates(self, year: int) -> dict[str, DividendRates]:
        """Return the rates for every dividend type in *year*.

        Args:
            year: The tax year to look up.

        Returns:
            Dict mapping each dividend type (``"eligible"`` /
            ``"non_eligible"``) to its :class:`DividendRates`.

        Raises:
            ValueError: If any dividend type has no applicable row.
        """
        return {
            dtype: self.get_rates(year, dtype) for dtype in _DIVIDEND_TYPES
        }
