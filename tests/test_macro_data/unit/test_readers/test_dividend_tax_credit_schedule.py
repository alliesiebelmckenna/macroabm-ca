"""Unit tests for the dividend gross-up / DTC rate schedule reader."""

import tempfile
from pathlib import Path

import pytest

from macro_data.readers.taxation.personal_income_tax.dividend_tax_credit_schedule import (
    DividendTaxCreditSchedule,
)

# Committed fallback copy of the BC PIT schedules
# (repo-root/spoof_data/freda/personal_income_tax).
#   parents[0]=test_readers [1]=unit [2]=test_macro_data [3]=tests [4]=repo root
BC_SCHEDULE_DIR = Path(__file__).resolve().parents[4] / "spoof_data" / "freda" / "personal_income_tax"

_FILENAME = "dividend_tax_credit_schedule.csv"


# from_name — the packaged BC schedule (jurisdiction-keyed, per-year rows)


class TestPackagedSchedule:
    def _load(self):
        return DividendTaxCreditSchedule.from_name(_FILENAME, schedule_dir=BC_SCHEDULE_DIR, jurisdiction="bc")

    def test_2014_eligible_rates(self):
        rates = self._load().get_rates(2014, "eligible")
        assert rates.gross_up_rate == pytest.approx(0.38)
        assert rates.dtc_pct_of_grossed_up == pytest.approx(0.10)
        assert rates.dtc_pct_of_actual == pytest.approx(0.138)

    def test_year_not_in_schedule_raises(self):
        """The schedule starts in 2014; an earlier year has no row."""
        with pytest.raises(ValueError, match="tax year 2013"):
            self._load().get_rates(2013, "eligible")

    def test_federal_ca_rows_jurisdiction_filter(self):
        """The packaged file also carries federal (CA) rows, selected by jurisdiction."""
        ca = DividendTaxCreditSchedule.from_name(_FILENAME, schedule_dir=BC_SCHEDULE_DIR, jurisdiction="ca")
        elig = ca.get_rates(2014, "eligible")
        assert elig.gross_up_rate == pytest.approx(0.38)
        assert elig.dtc_pct_of_grossed_up == pytest.approx(0.150198)
        non = ca.get_rates(2019, "non_eligible")
        assert non.gross_up_rate == pytest.approx(0.15)
        assert non.dtc_pct_of_grossed_up == pytest.approx(0.090301)


# from_csv — synthetic CSVs


def _write_csv(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        return f.name


_HEADER = "year,jurisdiction,dividend_type,gross_up_rate,dtc_pct_of_grossed_up,dtc_pct_of_actual"


class TestFromCsv:
    def test_missing_required_column_raises(self):
        csv = "dividend_type,gross_up_rate\neligible,0.38\n"
        p = _write_csv(csv)
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                DividendTaxCreditSchedule.from_csv(p, jurisdiction="bc")
        finally:
            Path(p).unlink(missing_ok=True)
