"""Refundable tax credit schedule, looked up by taxation year.

The companion to ``nrtc_schedule`` and deliberately close to it in shape, but
the two differ in three ways that matter and are easy to conflate:

* A refundable credit is PAID even when no tax is owed, so it is government
  EXPENDITURE at its full amount rather than revenue foregone. Nothing here
  floors at zero.
* Rows sharing a ``credit_name`` are components of ONE instrument and are
  summed before a single taper applies. ``credit`` is the eligibility class and
  the lookup key; ``amount_basis`` is multiplicity only.
* ``rate`` means something different than it does in the non-refundable file.
  There it is the statutory valuation rate; here it is a fraction-of-amount
  multiplier, ``1.0`` throughout, and the model does not read it.

Lookups are statutory only: a requested year must be published, since the
schedule is never projected past the years it records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Required columns in the consolidated refundable_tax_credits.csv.
_RTC_REQUIRED_COLS = {
    "year",  # taxation year the row applies to
    "jurisdiction",  # jurisdiction key (e.g. "BC")
    "credit_name",  # the INSTRUMENT; components sharing it are summed
    "credit",  # the eligibility class, and the reader's single lookup key
    "delivery",  # "settlement" or "instalments" -- see below
    "amount_basis",  # "once" or "per_dependant"; multiplicity only
}

# NOT interchangeable: the two legs refer to different years.
DELIVERY_SETTLEMENT = "settlement"
DELIVERY_INSTALMENTS = "instalments"
_DELIVERIES = frozenset({DELIVERY_SETTLEMENT, DELIVERY_INSTALMENTS})

# Eligibility per credit; an unmapped credit is dropped and contributes zero.
_ELIGIBILITY_RULES: dict[str, dict[str, object]] = {
    "Eligible Individual Amount": {"age_min": 19},
    "Spousal Amount": {"in_couple_household": True},
    "Equivalent To Spouse Amount": {"is_single_parent": True},
    "Dependant Amount": {},  # multiplicity comes from amount_basis
    # Social housing is excluded by model convention, not by statute; tenure -1 marks it.
    "Renter's Amount": {"is_renter": True},
}


@dataclass(frozen=True)
class RefundableCreditComponent:
    """One published row: a component of a refundable credit for one year.

    Attributes:
        credit_name: The instrument. Components sharing it are summed before
            the taper, so the elimination point belongs to the household rather
            than to any single row.
        credit: Eligibility class, and the single key the eligibility table uses.
        delivery: ``settlement`` or ``instalments``.
        amount: Credit value in per-person statutory dollars.
        amount_basis: ``once`` or ``per_dependant``.
        clawback: Income at which the taper begins.
        clawback_rate: Fraction of income above ``clawback`` that reduces it.
        eligibility: Rules an individual must satisfy, or ``None`` if unmapped.
    """

    credit_name: str
    credit: str
    delivery: str
    amount: float
    amount_basis: str
    clawback: Optional[float]
    clawback_rate: Optional[float]
    eligibility: Optional[dict[str, object]]


class RefundableSchedule:
    """Published refundable credits, looked up by year."""

    def __init__(
        self,
        credits_by_year: dict[int, list[RefundableCreditComponent]],
    ) -> None:
        self._by_year = {int(y): list(cs) for y, cs in credits_by_year.items()}

    @property
    def years(self) -> list[int]:
        """Every published year, ascending."""
        return sorted(self._by_year)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        jurisdiction: str,
    ) -> "RefundableSchedule":
        """Load refundable credit definitions from a CSV file.

        Args:
            path: Path to ``refundable_tax_credits.csv``.
            jurisdiction: Jurisdiction key; the CSV is jurisdiction-keyed.

        Returns:
            Configured ``RefundableSchedule``.

        Raises:
            ValueError: If required columns are missing, the jurisdiction has no
                rows, or a row carries an unknown ``delivery``.
        """
        import pandas as pd

        df = pd.read_csv(Path(path))
        df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})

        missing = _RTC_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Refundable tax-credit CSV is missing required columns: {sorted(missing)}. Found: {sorted(df.columns)}"
            )

        juris = jurisdiction.upper()
        df = df[df["jurisdiction"].astype(str).str.upper() == juris].copy()
        if df.empty:
            raise ValueError(f"Refundable tax-credit CSV {path} has no rows for jurisdiction {juris}")

        # Fail closed on an unknown delivery: a row read as a settlement would pay a year early.
        unknown = set(df["delivery"].astype(str).str.strip()) - _DELIVERIES
        if unknown:
            raise ValueError(
                f"Refundable tax-credit CSV {path} has unknown delivery values "
                f"{sorted(unknown)}; expected one of {sorted(_DELIVERIES)}."
            )

        by_year: dict[int, list[RefundableCreditComponent]] = {}
        for _, row in df.iterrows():
            credit = str(row["credit"]).strip()
            by_year.setdefault(int(row["year"]), []).append(
                RefundableCreditComponent(
                    credit_name=str(row["credit_name"]).strip(),
                    credit=credit,
                    delivery=str(row["delivery"]).strip(),
                    amount=float(row["amount"]),
                    amount_basis=str(row["amount_basis"]).strip(),
                    clawback=_opt_float(row.get("clawback")),
                    clawback_rate=_opt_float(row.get("clawback_rate")),
                    eligibility=_ELIGIBILITY_RULES.get(credit),
                )
            )
        return cls(by_year)

    def get_credits(self, year: int) -> list[RefundableCreditComponent]:
        """Return the published components for *year* (statutory lookup).

        Args:
            year: Target tax year.

        Returns:
            List of ``RefundableCreditComponent`` for *year*.

        Raises:
            ValueError: If *year* is not one of the published years.
        """
        if year in self._by_year:
            return list(self._by_year[year])

        raise ValueError(
            f"No refundable tax credit data available for year {year} "
            f"(available years: {self.years}). The reader is lookup-only and "
            f"does not project past the published years; add an explicit row "
            f"for {year} to the schedule CSV."
        )


def _opt_float(value: object) -> Optional[float]:
    """Parse an optional numeric cell, treating blank as absent."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return float(text)
