"""Tax credit schedule — per-individual non-refundable tax credit definitions.

This module provides the ``TaxCreditSchedule`` class, which reads a CSV
containing per-credit-kind definitions (base amounts, eligibility rules,
indexing flags) and supplies per-individual credit eligibility at
computation time.

CSV format (tax credit definitions)
------------------------------------
::

    tax_year,credit_kind,credit_amount,clawback_start,cap,rate,clawback_rate,indexing
    2014,Personal Amount,9869,,,,0.0506,,1
    2014,Age Amount,4426,32943,"62,450",0.0506,0.15,1
    ...

Columns:
    - ``tax_year`` (int): Base year for nominal amounts.
    - ``credit_kind`` (str): Human-readable credit name.
    - ``credit_amount`` (float, optional): Base dollar amount.
    - ``clawback_start`` (float, optional): Income where phaseout begins.
    - ``cap`` (float, optional): Income cap / upper phaseout bound.
    - ``rate`` (float): Credit rate (stored but not used at runtime —
      the model always applies ``pit_rates[0]``).
    - ``clawback_rate`` (float, optional): Phaseout rate.
    - ``indexing`` (int, 0/1): Whether CPI-indexed in later years.

Eligibility mapping
--------------------
The credit *kind* string is mapped to eligibility rules internally:

    ==================== ===============================================
    ``credit_kind``       Eligibility rule
    ==================== ===============================================
    ``Personal Amount``   Universal (every individual).
    ``Age Amount``        Age ≥ 65.
    ``Spousal Amount``    *(deferred — requires spousal income lookup)*
    ``Equivalent To…``    *(deferred — requires single-parent detection)*
    ``CPP Amount``        *(deferred — requires contribution data)*
    ``EI Amount``         *(deferred — requires contribution data)*
    ``Pension Income…``   *(deferred — requires pension income data)*
    ``Child Amount…``     *(deferred — requires child agents)*
    ==================== ===============================================

CPI inflation
-------------
When ``get_credits(tax_year=T)`` is called with *T* beyond the base
year, indexed credits are compound-inflated using the provided CPI map
(the same map used for bracket indexation).  Non-indexed credits
(e.g., ``Pension Income Amount``) remain at their nominal base-year
values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── required CSV columns ──────────────────────────────────────────────
_TC_REQUIRED_COLS = {
    "tax_year",          # int — base year
    "credit_kind",       # str — e.g. "Personal Amount", "Age Amount"
    "indexing",          # int (0/1) — whether CPI-indexed
}


# ── eligibility mapping: credit_kind → eligibility rules ──────────────
# Each entry is a dict of rules to check.  An individual is eligible
# iff *all* rules in the dict are satisfied.  Expand as new credits
# are activated.

_ELIGIBILITY_RULES: dict[str, dict[str, object]] = {
    "Personal Amount":          {},                                     # universal
    "Age Amount":               {"age_min": 65},
    "Pension Income Amount":    {"age_min": 65},                       # age proxy for pension receipt
    "Spousal Amount":           {"in_couple_household": True},          # married / common-law
    "Equivalent To Spouse Amount": {"is_single_parent": True},         # single parent / caregiver
    "Child Amount Under 18":    {"num_children_under_18_min": 1},      # parent
    "Child Amount Under 6":     {"num_children_under_6_min": 1},       # parent
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
        indexing: Whether this credit is CPI-indexed in later years.
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
    """Collection of tax credits for a base year, with CPI indexing.

    Typical usage::

        schedule = TaxCreditSchedule.from_csv("BC_tax_credit_2014.csv")
        credits = schedule.get_credits(tax_year=2017, cpi_map={2014: 0.01, ...})
        # → list[TaxCreditComponent] with CPI-inflated amounts

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
        cpi_map: Optional[dict[int, float]] = None,
    ) -> None:
        self._credits = list(credits)
        self._base_year = base_year
        self._cpi_map: dict[int, float] = dict(cpi_map) if cpi_map else {}

    # ── factories ────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        cpi_map: Optional[dict[int, float]] = None,
    ) -> "TaxCreditSchedule":
        """Load tax credit definitions from a CSV file.

        Args:
            path: Path to the CSV (e.g. ``BC_tax_credit_2014.csv``).
            cpi_map: Optional ``{year: inflation_rate}`` for CPI indexing.

        Returns:
            Configured ``TaxCreditSchedule``.
        """
        import pandas as pd

        df = pd.read_csv(Path(path))

        # Normalise column names
        col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=col_map)

        # Validate required columns exist
        missing = _TC_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Tax-credit CSV is missing required columns: {sorted(missing)}. "
                f"Found: {sorted(df.columns)}"
            )

        base_year = int(df["tax_year"].iloc[0])

        # ── build TaxCreditComponent for each row ──
        credits: list[TaxCreditComponent] = []
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
                # Unknown credit kind → universal fallback (deferred credits
                # loaded from CSV but not yet mapped get universal treatment;
                # they'll become meaningful once the eligibility map is
                # expanded).
                eligibility = {}

            credits.append(TaxCreditComponent(
                kind=kind,
                amount=amount,
                indexing=indexing,
                eligibility=eligibility,
                clawback_start=_clawback_start,
                clawback_cap=_clawback_cap,
            ))

        return cls(credits=credits, base_year=base_year, cpi_map=cpi_map)

    @classmethod
    def from_name(
        cls,
        filename: str,
        cpi_map: Optional[dict[int, float]] = None,
        schedule_dir: Optional[Path] = None,
    ) -> "TaxCreditSchedule":
        """Load by filename from the schedule directory.

        Args:
            filename: CSV filename (e.g. ``"BC_tax_credit_2014.csv"``).
            cpi_map: Optional CPI inflation map.
            schedule_dir: Override the default ``spoof_data/freda/`` directory.

        Returns:
            Configured ``TaxCreditSchedule``.
        """
        if schedule_dir is None:
            schedule_dir = (
                Path(__file__).resolve().parent.parent.parent.parent.parent
                / "spoof_data" / "freda"
            )
        path = schedule_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Tax-credit file not found: {path}\n"
                f"Available: {sorted([p.name for p in schedule_dir.glob('*.csv')])}"
            )
        return cls.from_csv(path, cpi_map=cpi_map)

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
        cpi_map: Optional[dict[int, float]] = None,
    ) -> list[TaxCreditComponent]:
        """Return credits with CPI-inflated amounts for *tax_year*.

        Indexed credits are compound-inflated from the base year.
        Non-indexed credits keep their nominal amount.

        Args:
            tax_year: Target tax year.
            cpi_map: CPI map (overrides the one stored at construction).

        Returns:
            List of ``TaxCreditComponent`` with amounts in *tax_year* dollars.
        """
        cmap = dict(cpi_map) if cpi_map else self._cpi_map

        if tax_year <= self._base_year or not cmap:
            return self._credits

        # Compute compound inflation factor
        factor = 1.0
        for y in range(self._base_year, tax_year):
            rate = cmap.get(y)
            if rate is not None:
                factor *= 1.0 + rate

        return [
            TaxCreditComponent(
                kind=c.kind,
                amount=c.amount * factor if c.indexing else c.amount,
                indexing=c.indexing,
                eligibility=c.eligibility,
                clawback_start=c.clawback_start * factor if (c.indexing and c.clawback_start is not None) else c.clawback_start,
                clawback_cap=c.clawback_cap * factor if (c.indexing and c.clawback_cap is not None) else c.clawback_cap,
            )
            for c in self._credits
        ]


# ══════════════════════════════════════════════════════════════════════
# Public helper: per-individual eligibility
# ══════════════════════════════════════════════════════════════════════

def is_eligible_for_credit(
    credit: TaxCreditComponent,
    age: Optional[float] = None,
    gender: Optional[int] = None,
    employee_income: Optional[float] = None,
    num_children_under_18: Optional[int] = None,
    num_children_under_6: Optional[int] = None,
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
        num_children_under_18: Number of children under 18 in the individual's
            household (required for ``num_children_under_18_min``).
        num_children_under_6: Number of children under 6 in the individual's
            household (required for ``num_children_under_6_min``).
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

    if "num_children_under_18_min" in rules:
        if num_children_under_18 is None or num_children_under_18 < rules["num_children_under_18_min"]:
            return False

    if "num_children_under_6_min" in rules:
        if num_children_under_6 is None or num_children_under_6 < rules["num_children_under_6_min"]:
            return False

    if "in_couple_household" in rules and not in_couple_household:
        return False

    if "is_single_parent" in rules and not is_single_parent:
        return False

    return True
