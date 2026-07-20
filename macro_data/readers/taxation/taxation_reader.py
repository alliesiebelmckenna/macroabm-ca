"""Bundled personal-income-tax schedules read from a taxation data directory.

``TaxationReader`` is the data-layer handle for a jurisdiction's personal
income tax schedules. It is built by ``DataReaders.from_raw_data`` when a
``taxation`` tree is present under the raw-data root, and consumed by
``build_central_government_configuration``. It loads the progressive bracket
schedule (with its companion non-refundable tax-credit schedule, auto-discovered
by ``PITSchedule``) and, when present, the dividend gross-up / DTC rate schedule.
The consumer selects a tax year when reading brackets, credits, and rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from macro_data.readers.taxation.personal_income_tax.dividend_tax_credit_schedule import (
    DividendTaxCreditSchedule,
)
from macro_data.readers.taxation.personal_income_tax.pit_schedule import PITSchedule


# Default schedule filenames; jurisdictions live in the jurisdiction column. They are
# defaults, not requirements: point SchedulePaths at any files with this schema.
RATES_THRESHOLDS_FILENAME = "rates_thresholds.csv"
TAX_CREDITS_FILENAME = "non_refundable_tax_credits.csv"
DIVIDEND_FILENAME = "dividend_tax_credit_schedule.csv"


@dataclass(frozen=True)
class SchedulePaths:
    """Which files the taxation schedules are read from.

    The reader is not tied to particular filenames — only to the consolidated,
    jurisdiction-keyed schema. Pointing it at a different set of files is how a run
    chooses its tax data: a bracket file covering only BC activates progressive
    PIT for BC alone, and one covering every province activates them all. The
    jurisdictions are whatever the ``jurisdiction`` column contains.

    Attributes:
        rates: Bracket schedule (required — it defines the covered jurisdictions).
        credits: Non-refundable credit schedule, or ``None`` to apply no credits.
        dividend: Dividend gross-up / DTC schedule, or ``None`` for no dividend
            path.
    """

    rates: Path
    credits: Optional[Path] = None
    dividend: Optional[Path] = None

    @classmethod
    def in_dir(
        cls,
        schedule_dir: Path,
        *,
        rates: str = RATES_THRESHOLDS_FILENAME,
        credits: str = TAX_CREDITS_FILENAME,
        dividend: str = DIVIDEND_FILENAME,
    ) -> "SchedulePaths":
        """Resolve the schedule files inside *schedule_dir* by filename.

        The filenames default to the canonical set; override them to read an
        alternative data source (e.g. a comprehensive all-province file kept
        under a different name). A credit or dividend file that does not exist is
        simply absent — that component is skipped, not an error.
        """
        schedule_dir = Path(schedule_dir)
        credits_path = schedule_dir / credits
        dividend_path = schedule_dir / dividend
        return cls(
            rates=schedule_dir / rates,
            credits=credits_path if credits_path.exists() else None,
            dividend=dividend_path if dividend_path.exists() else None,
        )


@dataclass
class TaxationReader:
    """Loaded personal-income-tax schedules for one jurisdiction.

    Attributes:
        pit_schedule: Progressive bracket schedule (statutory lookup only),
            carrying the companion tax-credit schedule via
            ``pit_schedule.non_refundable_tax_credits``.
        dividend_schedule: Dividend gross-up / DTC rate schedule, or ``None``
            when no dividend schedule is present.
        jurisdiction: The taxing-authority key these schedules belong to (e.g.
            ``"bc"``), so a consumer can attach them to the matching government
            agent when the model gains multiple governments.
    """

    pit_schedule: PITSchedule
    dividend_schedule: Optional[DividendTaxCreditSchedule]
    jurisdiction: str

    @classmethod
    def from_paths(
        cls,
        paths: SchedulePaths,
        *,
        jurisdiction: str,
    ) -> "TaxationReader":
        """Load *jurisdiction*'s schedules from the files named by *paths*.

        Args:
            paths: The schedule files to read.
            jurisdiction: Jurisdiction key — selects the jurisdiction rows read from each
                consolidated file.

        Returns:
            A ``TaxationReader`` with the bracket schedule (and its credits when
            the jurisdiction appears in the credit file) and the dividend
            schedule when the jurisdiction appears in the dividend file.
        """
        pit_schedule = PITSchedule.from_csv(paths.rates, jurisdiction=jurisdiction)
        pit_schedule.load_non_refundable_tax_credits(paths.credits, jurisdiction=jurisdiction)

        dividend_schedule: Optional[DividendTaxCreditSchedule] = None
        if paths.dividend is not None:
            try:
                dividend_schedule = DividendTaxCreditSchedule.from_csv(
                    paths.dividend, jurisdiction=jurisdiction
                )
            except ValueError:
                # The file carries no rows for this jurisdiction: it taxes
                # dividends at the ordinary rates, with no gross-up / DTC path.
                # A normal state for a jurisdiction whose dividend rates are not
                # yet sourced, not an error.
                dividend_schedule = None

        return cls(
            pit_schedule=pit_schedule,
            dividend_schedule=dividend_schedule,
            jurisdiction=jurisdiction,
        )

    @classmethod
    def from_dir(
        cls,
        schedule_dir: Path,
        *,
        jurisdiction: str,
    ) -> "TaxationReader":
        """Load *jurisdiction*'s schedules from the canonical files in *schedule_dir*."""
        return cls.from_paths(
            SchedulePaths.in_dir(schedule_dir), jurisdiction=jurisdiction
        )
