"""Module for producing a non-refundable tax credit (NRTC) schedule.

This module reads per-credit definitions (base amounts, eligibility rules,
index flags) from a consolidated ``non_refundable_tax_credits.csv`` keyed by
tax year and jurisdiction, and supplies the published credit components for a
requested year. The ``NRTCSchedule`` class is the companion to
``PITSchedule``: an individual's eligible credit bases are summed and valued at
the bottom marginal rate, then subtracted from gross tax.

Each credit is mapped internally to an eligibility rule — the Personal
Amount is universal, the Age Amount is age-gated, the Spousal and Equivalent To
Spouse amounts are household-tested; other known BC credits are recorded but
deferred until the model carries their eligibility signal. Lookups are
statutory only: a requested year must be published in the CSV, since the
schedule is never projected past the years it records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# Required columns in the consolidated non_refundable_tax_credits.csv.
_TC_REQUIRED_COLS = {
    "year",  # taxation year the row applies to
    "jurisdiction",  # jurisdiction key (e.g. "BC")
    "credit",    # credit name, e.g. "Personal Amount", "Age Amount"
    "index",     # whether the amount is statutorily indexed
}


# Maps each credit to a dict of eligibility rules; an individual is
# eligible when all rules in the dict are satisfied. Expand as credits activate.
_ELIGIBILITY_RULES: dict[str, dict[str, object]] = {
    # Active: eligibility the runtime credit pool can express today.
    "Personal Amount":          {},                                     # universal
    "Age Amount":               {"age_min": 65},
    "Spousal Amount":           {"in_couple_household": True},          # married / common-law
    "Equivalent To Spouse Amount": {"is_single_parent": True},         # single parent / caregiver
    # Deferred: known BC credits whose eligibility signal the model does not yet
    # carry, so the builder skips them. Trailing notes flag nuances to resolve
    # before activating one.
    "Pension Income Amount":    {"has_eligible_pension_income": True},  # lesser of $1000 or actual eligible pension income; NOT age-based (CPP may start 60-70, also covers non-CPP pension)
    "B.C. Caregiver Amount":        {"is_caregiver": True},            # caring for a dependant with impairment; clawed back on the dependant's income
    "Disability Amount":            {"has_disability": True},          # DTC-eligible individual
    "Disability Amount (Child)":    {"has_disability_dependant": True},# supplement for a dependant under 18 with a disability
    "Adoption Amount":              {"has_adoption_expense": True},    # event-based: amount column is the MAX eligible expense, not a flat base
    "Volunteer Firefighter Amount": {"is_volunteer_first_responder": True},  # 200+ volunteer hours
    "Medical Expense Amount":       {"has_medical_expense": True},     # formula: expenses minus lesser(3% net income, cap); amount column blank
    "BC Tax Reduction Credit":      {"is_income_tested_reduction": True},  # direct $ reduction, NOT base x rate; reduced by 3.56% of net income over threshold
}


@dataclass
class TaxCreditComponent:
    """A single non-refundable tax credit defined for a base tax year.

    Attributes:
        credit: Human-readable credit name (e.g. ``"Age Amount"``).
        amount: Base dollar amount in the base tax year.
        index: Whether the amount is statutorily indexed.  Recorded for
            reference only; not consumed anywhere (there is no forward
            projection).
        eligibility: Dict of eligibility rules (e.g. ``{"age_min": 65}``).
            Empty dict means universal.
        clawback: Income of spouse/dependent at which clawback
            begins.  None means no clawback.
        top: Income of spouse/dependent at which the credit
            is fully eliminated.  None means no cap.
    """

    credit: str
    amount: float
    index: bool = True
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
            raise ValueError(
                f"Tax-credit CSV {path} does not contain any rows for jurisdiction {juris}"
            )

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

            index = bool(int(row["index"])) if not pd.isna(row["index"]) else True

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
                    index=index,
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
                f"Tax-credit file not found: {path}\n"
                f"Available: {sorted([p.name for p in schedule_dir.glob('*.csv')])}"
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
