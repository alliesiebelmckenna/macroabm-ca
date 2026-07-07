"""Tax credit schedule — per-individual non-refundable tax credit definitions.

This module provides the ``TaxCreditSchedule`` class, which reads a CSV
containing per-credit-kind definitions (base amounts, eligibility rules,
indexing flags) and supplies per-individual credit eligibility at
computation time.

CSV format (consolidated ``non_refundable_tax_credits.csv``)
------------------------------------------------------------
::

    tax_year,geo,credit,amount,top,rate,clawback,clawback_rate,index
    2014,BC,Personal Amount,9869.0,,0.0506,,,1
    2014,BC,Age Amount,4426.0,62450.0,0.0506,32943.0,0.15,1
    ...

Columns (mapped on read to the internal field names in parentheses):
    - ``tax_year`` (int): Base year for nominal amounts.
    - ``geo`` (str): Jurisdiction key (e.g. "BC"); rows are filtered to the
      requested jurisdiction on read.
    - ``credit`` (str, → ``credit_kind``): Human-readable credit name.
    - ``amount`` (float, optional, → ``credit_amount``): Base dollar amount.
    - ``top`` (float, optional, → ``cap``): Income cap / upper phaseout bound.
    - ``rate`` (float): Credit rate, recorded in the CSV for reference; not
      parsed at runtime — the model values the credit base at the bottom
      marginal rate (``pit_rates[0]``).
    - ``clawback`` (float, optional, → ``clawback_start``): Income where
      phaseout begins.
    - ``clawback_rate`` (float, optional): Phaseout rate, recorded for
      reference; not parsed (the runtime applies no spousal clawback —
      the clawback data is saved but deliberately inert).
    - ``index`` (int, 0/1, → ``indexing``): Whether the amount is statutorily
      indexed. Recorded for reference; not used to compute anything (there is
      no forward projection).

Eligibility mapping
--------------------
The credit *kind* string is mapped to eligibility rules internally:

    ==================== ===============================================
    ``credit_kind``       Eligibility rule
    ==================== ===============================================
    ``Personal Amount``   Universal (every individual).
    ``Age Amount``        Age ≥ 65.
    ``Spousal Amount``    Couple household, income-tested against the spouse.
    ``Equivalent To…``    Single-parent household.
    ``CPP Amount``        *(deferred — requires contribution data)*
    ``EI Amount``         *(deferred — requires contribution data)*
    ``Pension Income…``   *(deferred — requires pension income data)*
    ==================== ===============================================

Lookup-only
-----------
``get_credits(tax_year=T)`` returns the actual published components for *T*
(amounts, clawbacks, rates) when *T* is one of the schedule's years, and raises
otherwise.  The reader does not compound or project past the published years —
there is no forward projection anywhere in the pipeline, so a schedule meant to
cover a later year (including a legislated freeze) must carry an explicit row
for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── required CSV columns (consolidated ``non_refundable_tax_credits.csv``) ──
_TC_REQUIRED_COLS = {
    "tax_year",  # int — taxation year the row applies to
    "geo",       # str — jurisdiction key (e.g. "BC"); rows are geo-filtered
    "credit",    # str — credit kind, e.g. "Personal Amount", "Age Amount"
    "index",     # int (0/1) — whether indexed in projection
}

# Boundary mapping from the consolidated CSV columns to the internal field
# names used throughout the pipeline (applied unconditionally on read).
_CREDIT_COLUMN_MAP = {
    "credit": "credit_kind",
    "amount": "credit_amount",
    "top": "cap",
    "clawback": "clawback_start",
    "index": "indexing",
}


# ── eligibility mapping: credit_kind → eligibility rules ──────────────
# Each entry is a dict of rules to check.  An individual is eligible
# iff *all* rules in the dict are satisfied.  Expand as new credits
# are activated.

_ELIGIBILITY_RULES: dict[str, dict[str, object]] = {
    # ── Active: eligibility the runtime credit pool can express today. ──
    "Personal Amount":          {},                                     # universal
    "Age Amount":               {"age_min": 65},
    "Spousal Amount":           {"in_couple_household": True},          # married / common-law
    "Equivalent To Spouse Amount": {"is_single_parent": True},         # single parent / caregiver
    # ── Deferred: known BC credits whose eligibility signal the model does not
    #    yet carry.  Each key is deliberately one the config builder cannot
    #    express (see _EXPRESSIBLE_ELIGIBILITY_KEYS in central_government_builder),
    #    so the builder SKIPS the credit rather than granting it to everyone.
    #    To activate one: add the per-individual attribute, give it a runtime
    #    branch in pit_pools._credit_amount, and move its key into the
    #    expressible set.  See nuances flagged on each before activating.
    "Pension Income Amount":    {"has_eligible_pension_income": True},  # lesser of $1000 or actual eligible pension income; NOT age-based (CPP may start 60-70, also covers non-CPP pension)
    "B.C. Caregiver Amount":        {"is_caregiver": True},            # caring for a dependant with impairment; clawed back on the dependant's income
    "Disability Amount":            {"has_disability": True},          # DTC-eligible individual
    "Disability Amount (Child)":    {"has_disability_dependant": True},# supplement for a dependant under 18 with a disability
    "Adoption Amount":              {"has_adoption_expense": True},    # event-based: amount column is the MAX eligible expense, not a flat base
    "Volunteer Firefighter Amount": {"is_volunteer_first_responder": True},  # 200+ volunteer hours
    "Medical Expense Amount":       {"has_medical_expense": True},     # formula: expenses minus lesser(3% net income, cap); amount column blank
    "BC Tax Reduction Credit":      {"is_income_tested_reduction": True},  # direct $ reduction, NOT base x rate; reduced by 3.56% of net income over threshold
}


# ══════════════════════════════════════════════════════════════════════
# TaxCreditComponent
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TaxCreditComponent:
    """A single non-refundable tax credit defined for a base tax year.

    Attributes:
        kind: Human-readable credit name (e.g. ``"Age Amount"``).
        amount: Base dollar amount in the base tax year.
        indexing: Whether the amount is statutorily indexed.  Recorded for
            reference only; not consumed anywhere (there is no forward
            projection).
        eligibility: Dict of eligibility rules (e.g. ``{"age_min": 65}``).
            Empty dict means universal.
        clawback_start: Income of spouse/dependent at which clawback
            begins.  None means no clawback.
        clawback_cap: Income of spouse/dependent at which the credit
            is fully eliminated.  None means no cap.
    """

    kind: str
    amount: float
    indexing: bool = True
    eligibility: dict[str, object] = field(default_factory=dict)
    clawback_start: Optional[float] = None
    clawback_cap: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════
# TaxCreditSchedule
# ══════════════════════════════════════════════════════════════════════

class TaxCreditSchedule:
    """Collection of published tax credits, looked up by year.

    Typical usage::

        schedule = TaxCreditSchedule.from_csv("non_refundable_tax_credits.csv")
        credits = schedule.get_credits(tax_year=2017)
        # → list[TaxCreditComponent] published for 2017

        # At tax time, sum eligible credit bases per individual:
        for ind_age, ind_income in zip(ages, incomes):
            eligible_bases = [
                c.amount for c in credits if _is_eligible(c, ind_age, ind_income)
            ]
            credit = sum(eligible_bases) * bottom_bracket_rate
    """

    # ── internal ──────────────────────────────────────────────────────

    def __init__(
        self,
        credits: list[TaxCreditComponent],
        base_year: int,
        credits_by_year: Optional[dict[int, list[TaxCreditComponent]]] = None,
    ) -> None:
        self._credits = list(credits)
        self._base_year = base_year
        # Per-year groups for statutory lookup; default = single-year (base only).
        self._by_year: dict[int, list[TaxCreditComponent]] = (
            {int(y): list(cs) for y, cs in credits_by_year.items()}
            if credits_by_year is not None
            else {base_year: list(credits)}
        )

    # ── factories ────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        jurisdiction: str = "bc",
    ) -> "TaxCreditSchedule":
        """Load tax credit definitions from a CSV file.

        Args:
            path: Path to the CSV (``non_refundable_tax_credits.csv``).
            jurisdiction: Jurisdiction key used when the CSV is geo-keyed.

        Returns:
            Configured ``TaxCreditSchedule``.
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

        geo = jurisdiction.upper()
        df = df[df["geo"].astype(str).str.upper() == geo].copy()
        if df.empty:
            raise ValueError(
                f"Tax-credit CSV {path} does not contain any rows for geo {geo}"
            )

        # Boundary mapping: consolidated column names -> internal field names.
        df = df.rename(columns=_CREDIT_COLUMN_MAP)

        # Minimum year, not the first row's — a valid but unsorted CSV must
        # not silently shift the base year (it selects the base credit set).
        base_year = int(df["tax_year"].min())

        # ── build TaxCreditComponent for each row, grouped by tax_year so a
        #    multi-year schedule supports statutory lookup (not a mixed list) ──
        credits_by_year: dict[int, list[TaxCreditComponent]] = {}
        for _, row in df.iterrows():
            kind = str(row["credit_kind"]).strip()

            # Parse amount — empty means no predetermined amount
            raw_amount = row.get("credit_amount")
            if pd.isna(raw_amount) or str(raw_amount).strip() == "":
                # Credits like CPP/EI have no predetermined amount; skip
                continue

            amount = float(str(raw_amount).replace(",", ""))
            # Keep credits even when $0 — they prove the eligibility
            # infrastructure and open policy discussions.  The zero
            # amount means they have no revenue impact by default.

            indexing = bool(int(row["indexing"])) if not pd.isna(row["indexing"]) else True

            # Parse optional clawback fields (spousal / dependent income tests).
            _clawback_start: Optional[float] = None
            raw_cs = row.get("clawback_start")
            if raw_cs is not None and not (isinstance(raw_cs, float) and pd.isna(raw_cs)) and str(raw_cs).strip() != "":
                _clawback_start = float(str(raw_cs).replace(",", ""))

            _clawback_cap: Optional[float] = None
            raw_cc = row.get("cap")
            if raw_cc is not None and not (isinstance(raw_cc, float) and pd.isna(raw_cc)) and str(raw_cc).strip() != "":
                _clawback_cap = float(str(raw_cc).replace(",", ""))

            # Look up eligibility rules
            eligibility = _ELIGIBILITY_RULES.get(kind)
            if eligibility is None:
                # Unknown credit kind → DEFER, do not apply universally.  Assign a
                # deliberately non-expressible marker so the config builder skips
                # the credit instead of granting it to every individual at full
                # amount.  Register the kind in _ELIGIBILITY_RULES (above) to make
                # its deferral explicit, or wire a runtime branch to activate it.
                eligibility = {"_deferred_unmapped": True}

            credits_by_year.setdefault(int(row["tax_year"]), []).append(
                TaxCreditComponent(
                    kind=kind,
                    amount=amount,
                    indexing=indexing,
                    eligibility=eligibility,
                    clawback_start=_clawback_start,
                    clawback_cap=_clawback_cap,
                )
            )

        base_credits = credits_by_year.get(base_year, [])
        return cls(
            credits=base_credits,
            base_year=base_year,
            credits_by_year=credits_by_year,
        )

    @classmethod
    def from_name(
        cls,
        filename: str,
        schedule_dir: Path,
        jurisdiction: str = "bc",
    ) -> "TaxCreditSchedule":
        """Load by filename from *schedule_dir*.

        Args:
            filename: CSV filename (``"non_refundable_tax_credits.csv"``).
            schedule_dir: Directory holding the schedule CSVs — typically
                ``raw_data_path / "taxation" / "personal_income_tax"``.
            jurisdiction: Jurisdiction key used when the CSV is geo-keyed.

        Returns:
            Configured ``TaxCreditSchedule``.
        """
        schedule_dir = Path(schedule_dir)
        path = schedule_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Tax-credit file not found: {path}\n"
                f"Available: {sorted([p.name for p in schedule_dir.glob('*.csv')])}"
            )
        return cls.from_csv(path, jurisdiction=jurisdiction)

    # ── properties ───────────────────────────────────────────────────

    @property
    def base_year(self) -> int:
        """Base tax year (all amounts are nominal for this year)."""
        return self._base_year

    @property
    def credits(self) -> list[TaxCreditComponent]:
        """All credit components for the base year (nominal amounts)."""
        return list(self._credits)

    # ── public methods ───────────────────────────────────────────────

    def get_credits(
        self,
        tax_year: int,
    ) -> list[TaxCreditComponent]:
        """Return the published credit components for *tax_year* (statutory lookup).

        Lookup-only: a year present in the schedule returns its OWN published
        components — actual amounts, clawbacks, and rates.  A year not in the
        schedule raises; there is no forward projection — a schedule meant to
        cover a later year (including a legislated freeze) must carry an
        explicit row for it.

        Args:
            tax_year: Target tax year.

        Returns:
            List of ``TaxCreditComponent`` for *tax_year*.

        Raises:
            ValueError: If *tax_year* is not one of the published years.
        """
        if tax_year in self._by_year:
            return list(self._by_year[tax_year])

        raise ValueError(
            f"tax_year {tax_year} is not in the published credit schedule "
            f"(available years: {sorted(self._by_year)}). The reader is "
            f"lookup-only and does not project past the published years; add "
            f"an explicit row for {tax_year} to the schedule CSV."
        )


# ══════════════════════════════════════════════════════════════════════
# Public helper: per-individual eligibility
# ══════════════════════════════════════════════════════════════════════

def is_eligible_for_credit(
    credit: TaxCreditComponent,
    age: Optional[float] = None,
    gender: Optional[int] = None,
    employee_income: Optional[float] = None,
    in_couple_household: Optional[bool] = None,
    is_single_parent: Optional[bool] = None,
) -> bool:
    """Check whether an individual qualifies for a given tax credit.

    All rules in ``credit.eligibility`` must be satisfied.
    Empty eligibility dict → universal credit (always ``True``).

    Args:
        credit: Credit component to check.
        age: Individual's age (required if eligibility has ``age_min``).
        gender: Individual's gender (required if eligibility has ``gender``).
        employee_income: Individual's income (required for future clawback).
        in_couple_household: Whether the individual lives in a couple
            household (required for ``in_couple_household``).
        is_single_parent: Whether the individual is a single parent
            (required for ``is_single_parent``).

    Returns:
        ``True`` if the individual qualifies.
    """
    rules = credit.eligibility
    if not rules:
        return True

    if "age_min" in rules and (age is None or age < rules["age_min"]):
        return False

    if "in_couple_household" in rules and not in_couple_household:
        return False

    if "is_single_parent" in rules and not is_single_parent:
        return False

    return True
