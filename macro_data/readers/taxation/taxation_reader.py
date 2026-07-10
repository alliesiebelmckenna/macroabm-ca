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


# Consolidated schedule filenames; jurisdictions live in the geo column.
_RATES_THRESHOLDS_FILENAME = "rates_thresholds.csv"
_DIVIDEND_FILENAME = "dividend_tax_credit_schedule.csv"


@dataclass
class TaxationReader:
    """Loaded personal-income-tax schedules for one jurisdiction.

    Attributes:
        pit_schedule: Progressive bracket schedule (statutory lookup only),
            carrying the companion tax-credit schedule via
            ``pit_schedule.tax_credits``.
        dividend_schedule: Dividend gross-up / DTC rate schedule, or ``None``
            when no dividend schedule is present.
        jurisdiction: The taxing-authority key these schedules belong to (e.g.
            ``"bc"``), so a consumer can attach them to the matching government
            agent when the model gains multiple governments.
    """

    pit_schedule: PITSchedule
    dividend_schedule: Optional[DividendTaxCreditSchedule]
    jurisdiction: str = "bc"

    @classmethod
    def from_dir(
        cls,
        schedule_dir: Path,
        *,
        jurisdiction: str = "bc",
    ) -> "TaxationReader":
        """Load the schedules for *jurisdiction* from *schedule_dir*.

        Args:
            schedule_dir: Directory holding the consolidated schedule CSVs —
                typically ``raw_data_path / "taxation" / "personal_income_tax"``.
            jurisdiction: Jurisdiction key — selects the geo rows read from the
                consolidated bracket, credit, and dividend files.

        Returns:
            A ``TaxationReader`` with the bracket schedule (and its companion
            credits) loaded, and the dividend schedule loaded when present.
        """
        schedule_dir = Path(schedule_dir)

        # Geo-filtered bracket schedule; the companion credit CSV is
        # auto-discovered by PITSchedule.
        pit_schedule = PITSchedule.from_name(
            _RATES_THRESHOLDS_FILENAME,
            schedule_dir=schedule_dir,
            jurisdiction=jurisdiction,
        )

        try:
            dividend_schedule: Optional[DividendTaxCreditSchedule] = (
                DividendTaxCreditSchedule.from_name(
                    _DIVIDEND_FILENAME,
                    schedule_dir=schedule_dir,
                    jurisdiction=jurisdiction,
                )
            )
        except FileNotFoundError:
            dividend_schedule = None

        return cls(
            pit_schedule=pit_schedule,
            dividend_schedule=dividend_schedule,
            jurisdiction=jurisdiction,
        )
