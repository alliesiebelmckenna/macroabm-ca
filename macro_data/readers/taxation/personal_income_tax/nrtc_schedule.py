"""Module for producing a non-refundable tax credit (NRTC) schedule.

This module reads per-credit definitions (base amounts, eligibility rules,
phaseout thresholds) from a consolidated ``non_refundable_tax_credits.csv`` keyed by
tax year and jurisdiction, and supplies the published credit components for a
requested year. The ``NRTCSchedule`` class is the companion to
``PITSchedule``: an individual's eligible credit bases are summed and valued at
the bottom marginal rate, then subtracted from gross tax.

Only the credits the runtime can express are registered here — the Personal
Amount is universal, the Age Amount is age-gated, and the Spousal and Equivalent
To Spouse amounts are household-tested. Any other credit the CSV publishes is
unregistered: it is marked unmapped on load and dropped by the configuration
builder, so it is never granted. Lookups are statutory only: a requested year
must be published in the CSV, since the schedule is never projected past the
years it records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Required columns in the consolidated non_refundable_tax_credits.csv.
_TC_REQUIRED_COLS = {
    "year",  # taxation year the row applies to
    "jurisdiction",  # jurisdiction key (e.g. "BC")
    "credit",  # credit name, e.g. "Personal Amount", "Age Amount"
    "index",  # schema check only: required for shape, no longer read
}


# Maps each credit to a dict of eligibility rules; an individual is
# eligible when all rules in the dict are satisfied. Expand as credits activate.
#
# Only credits the runtime can express are registered. A credit published in the
# schedule but absent here is unmapped: the builder drops it and the runtime
# credit pool contributes zero, so an unexpressible credit is never granted
# universally. Register a credit here only together with its runtime branch.
_ELIGIBILITY_RULES: dict[str, dict[str, object]] = {
    "Personal Amount": {},  # universal
    "Age Amount": {"age_min": 65},
    "Spousal Amount": {"in_couple_household": True},  # married / common-law
    "Equivalent To Spouse Amount": {"is_single_parent": True},  # single parent / caregiver
}


@dataclass
class TaxCreditComponent:
    """A single non-refundable tax credit defined for a base tax year.

    Attributes:
        credit: Human-readable credit name (e.g. ``"Age Amount"``).
        amount: Base dollar amount in the base tax year.
        eligibility: Dict of eligibility rules (e.g. ``{"age_min": 65}``).
            Empty dict means universal.
        clawback: Income at which the phaseout begins.  Whose income depends
            on the credit: own income for the Age Amount, the spouse's for the
            Spousal Amount.  None means no clawback.
        top: Income at which the credit is fully eliminated.  Where a credit
            publishes no clawback, this carries the exemption implicitly as
            ``top - amount``.  None means no cap.
    """

    credit: str
    amount: float
    eligibility: dict[str, object] = field(default_factory=dict)
    clawback: Optional[float] = None
    top: Optional[float] = None


class NRTCSchedule:
    """Collection of published non-refundable tax credits, looked up by year."""

    def __init__(
        self,
        credits: list[TaxCreditComponent],
        start_year: int,
        credits_by_year: Optional[dict[int, list[TaxCreditComponent]]] = None,
    ) -> None:
        self._credits = list(credits)
        self._start_year = start_year
        # Per-year groups for statutory lookup; default = single-year (start only).
        self._by_year: dict[int, list[TaxCreditComponent]] = (
            {int(y): list(cs) for y, cs in credits_by_year.items()}
            if credits_by_year is not None
            else {start_year: list(credits)}
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        jurisdiction: str,
    ) -> "NRTCSchedule":
        """Load tax credit definitions from a CSV file.

        Args:
            path: Path to the CSV (``non_refundable_tax_credits.csv``).
            jurisdiction: Jurisdiction key used when the CSV is jurisdiction-keyed.

        Returns:
            Configured ``NRTCSchedule``.
        """
        import pandas as pd

        df = pd.read_csv(Path(path))

        # Normalise column names
        col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=col_map)

        # Validate the consolidated columns exist
        missing = _TC_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Tax-credit CSV is missing required columns: {sorted(missing)} "
                f"(consolidated non_refundable_tax_credits format). "
                f"Found: {sorted(df.columns)}"
            )

        juris = jurisdiction.upper()
        df = df[df["jurisdiction"].astype(str).str.upper() == juris].copy()
        if df.empty:
            raise ValueError(f"Tax-credit CSV {path} does not contain any rows for jurisdiction {juris}")

        # Minimum year, not the first row's, so an unsorted CSV does not shift
        # the base credit set.
        start_year = int(df["year"].min())

        # Build a component per row, grouped by year for statutory lookup.
        credits_by_year: dict[int, list[TaxCreditComponent]] = {}
        for _, row in df.iterrows():
            credit = str(row["credit"]).strip()

            # Parse amount — empty means no predetermined amount
            raw_amount = row.get("amount")
            if pd.isna(raw_amount) or str(raw_amount).strip() == "":
                # Credits like CPP/EI have no predetermined amount; skip
                continue

            amount = float(str(raw_amount).replace(",", ""))
            # Keep $0 credits: they carry the eligibility wiring with no
            # revenue impact by default.

            # Parse optional clawback fields (spousal / dependent income tests).
            clawback: Optional[float] = None
            raw_cs = row.get("clawback")
            if raw_cs is not None and not (isinstance(raw_cs, float) and pd.isna(raw_cs)) and str(raw_cs).strip() != "":
                clawback = float(str(raw_cs).replace(",", ""))

            top: Optional[float] = None
            raw_cc = row.get("top")
            if raw_cc is not None and not (isinstance(raw_cc, float) and pd.isna(raw_cc)) and str(raw_cc).strip() != "":
                top = float(str(raw_cc).replace(",", ""))

            # Look up eligibility rules
            eligibility = _ELIGIBILITY_RULES.get(credit)
            if eligibility is None:
                # Unknown credit: mark it non-expressible so the config builder
                # skips it rather than granting it to everyone.
                eligibility = {"_deferred_unmapped": True}

            credits_by_year.setdefault(int(row["year"]), []).append(
                TaxCreditComponent(
                    credit=credit,
                    amount=amount,
                    eligibility=eligibility,
                    clawback=clawback,
                    top=top,
                )
            )

        base_credits = credits_by_year.get(start_year, [])
        return cls(
            credits=base_credits,
            start_year=start_year,
            credits_by_year=credits_by_year,
        )

    @classmethod
    def from_name(
        cls,
        filename: str,
        schedule_dir: Path,
        jurisdiction: str,
    ) -> "NRTCSchedule":
        """Load by filename from *schedule_dir*.

        Args:
            filename: CSV filename (``"non_refundable_tax_credits.csv"``).
            schedule_dir: Directory holding the schedule CSVs — typically
                ``raw_data_path / "taxation" / "personal_income_tax"``.
            jurisdiction: Jurisdiction key used when the CSV is jurisdiction-keyed.

        Returns:
            Configured ``NRTCSchedule``.
        """
        schedule_dir = Path(schedule_dir)
        path = schedule_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Tax-credit file not found: {path}\nAvailable: {sorted([p.name for p in schedule_dir.glob('*.csv')])}"
            )
        return cls.from_csv(path, jurisdiction=jurisdiction)

    @property
    def start_year(self) -> int:
        """First published year (all amounts are nominal for this year)."""
        return self._start_year

    @property
    def credits(self) -> list[TaxCreditComponent]:
        """All credit components for the start year (nominal amounts)."""
        return list(self._credits)

    def get_credits(
        self,
        year: int,
    ) -> list[TaxCreditComponent]:
        """Return the published credit components for *year* (statutory lookup).

        A year present in the schedule returns its own published amounts,
        clawbacks, and rates. A year not in the schedule raises, since the
        reader does not project past published years.

        Args:
            year: Target tax year.

        Returns:
            List of ``TaxCreditComponent`` for *year*.

        Raises:
            ValueError: If *year* is not one of the published years.
        """
        if year in self._by_year:
            return list(self._by_year[year])

        raise ValueError(
            f"No non-refundable tax credit data available for year {year} "
            f"(available years: {sorted(self._by_year)}). The reader is "
            f"lookup-only and does not project past the published years; add "
            f"an explicit row for {year} to the schedule CSV."
        )
