"""Unit tests for the dividend gross-up / DTC rate schedule reader."""

import tempfile
from pathlib import Path

import pytest

from macro_data.readers.taxation.personal_income_tax.dividend_tax_credit_schedule import (
    DividendRates,
    DividendTaxCreditSchedule,
)

# Committed fallback copy of the BC PIT schedules
# (repo-root/spoof_data/freda/personal_income_tax).
#   parents[0]=test_readers [1]=unit [2]=test_macro_data [3]=tests [4]=repo root
BC_SCHEDULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "spoof_data" / "freda" / "personal_income_tax"
)

_FILENAME = "dividend_tax_credit_schedule.csv"


# from_name — the packaged BC schedule (geo-keyed, per-year rows)


class TestPackagedSchedule:
    def _load(self):
        return DividendTaxCreditSchedule.from_name(
            _FILENAME, schedule_dir=BC_SCHEDULE_DIR, jurisdiction="bc"
        )

    def test_2014_eligible_rates(self):
        rates = self._load().get_rates(2014, "eligible")
        assert rates.gross_up_rate == pytest.approx(0.38)
        assert rates.dtc_pct_of_grossed_up == pytest.approx(0.10)
        assert rates.dtc_pct_of_actual == pytest.approx(0.138)

    def test_2014_non_eligible_rates(self):
        rates = self._load().get_rates(2014, "non_eligible")
        assert rates.gross_up_rate == pytest.approx(0.18)
        assert rates.dtc_pct_of_grossed_up == pytest.approx(0.0259)
        assert rates.dtc_pct_of_actual == pytest.approx(0.0306)

    def test_get_year_rates_returns_both_types(self):
        year = self._load().get_year_rates(2014)
        assert set(year) == {"eligible", "non_eligible"}
        assert year["eligible"].dtc_pct_of_grossed_up == pytest.approx(0.10)
        assert year["non_eligible"].gross_up_rate == pytest.approx(0.18)

    def test_frozen_later_year(self):
        """A post-2019 year carries the frozen 2019 values."""
        schedule = self._load()
        assert schedule.get_rates(2020, "eligible").dtc_pct_of_grossed_up == pytest.approx(0.12)
        assert schedule.get_rates(2020, "non_eligible").gross_up_rate == pytest.approx(0.15)

    def test_interior_year(self):
        """The 2016 eligible row still uses the pre-2019 rate."""
        rates = self._load().get_rates(2016, "eligible")
        assert rates.gross_up_rate == pytest.approx(0.38)
        assert rates.dtc_pct_of_grossed_up == pytest.approx(0.10)

    def test_year_not_in_schedule_raises(self):
        """The schedule starts in 2014; an earlier year has no row."""
        with pytest.raises(ValueError, match="tax year 2013"):
            self._load().get_rates(2013, "eligible")

    def test_federal_ca_rows_geo_filter(self):
        """The packaged file also carries federal (CA) rows, selected by geo."""
        ca = DividendTaxCreditSchedule.from_name(
            _FILENAME, schedule_dir=BC_SCHEDULE_DIR, jurisdiction="ca"
        )
        elig = ca.get_rates(2014, "eligible")
        assert elig.gross_up_rate == pytest.approx(0.38)
        assert elig.dtc_pct_of_grossed_up == pytest.approx(0.150198)
        non = ca.get_rates(2019, "non_eligible")
        assert non.gross_up_rate == pytest.approx(0.15)
        assert non.dtc_pct_of_grossed_up == pytest.approx(0.090301)

    def test_unknown_dividend_type_raises(self):
        with pytest.raises(ValueError, match="No rows for dividend_type"):
            self._load().get_rates(2014, "preferred")

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Dividend-rate file not found"):
            DividendTaxCreditSchedule.from_name(
                "does_not_exist.csv", schedule_dir=BC_SCHEDULE_DIR
            )


# from_csv — synthetic CSVs


def _write_csv(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        return f.name


_HEADER = "tax_year,geo,dividend_type,gross_up_rate,dtc_pct_of_grossed_up,dtc_pct_of_actual"


class TestFromCsv:
    def test_missing_required_column_raises(self):
        csv = "dividend_type,gross_up_rate\neligible,0.38\n"
        p = _write_csv(csv)
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                DividendTaxCreditSchedule.from_csv(p)
        finally:
            Path(p).unlink(missing_ok=True)

    def test_geo_filter_selects_jurisdiction(self):
        csv = (
            _HEADER + "\n"
            "2014,BC,eligible,0.38,0.10,0.138\n"
            "2014,ON,eligible,0.38,0.10,0.138\n"
        )
        p = _write_csv(csv)
        try:
            bc = DividendTaxCreditSchedule.from_csv(p, jurisdiction="bc")
            assert bc.get_rates(2014, "eligible").gross_up_rate == pytest.approx(0.38)
            # ON rows are excluded from the BC-filtered schedule.
            on = DividendTaxCreditSchedule.from_csv(p, jurisdiction="on")
            assert on.get_rates(2014, "eligible").gross_up_rate == pytest.approx(0.38)
        finally:
            Path(p).unlink(missing_ok=True)

    def test_absent_jurisdiction_raises(self):
        csv = _HEADER + "\n2014,BC,eligible,0.38,0.10,0.138\n"
        p = _write_csv(csv)
        try:
            with pytest.raises(ValueError, match="does not contain any rows for geo AB"):
                DividendTaxCreditSchedule.from_csv(p, jurisdiction="ab")
        finally:
            Path(p).unlink(missing_ok=True)

    def test_duplicate_year_rows_raise(self):
        csv = (
            _HEADER + "\n"
            "2014,BC,eligible,0.38,0.10,0.138\n"
            "2014,BC,eligible,0.38,0.11,0.152\n"  # duplicate year+type
        )
        p = _write_csv(csv)
        try:
            schedule = DividendTaxCreditSchedule.from_csv(p)
            with pytest.raises(ValueError, match="Duplicate"):
                schedule.get_rates(2014, "eligible")
        finally:
            Path(p).unlink(missing_ok=True)

    def test_actual_rate_optional(self):
        """A CSV without dtc_pct_of_actual yields None for that field."""
        csv = (
            "tax_year,geo,dividend_type,gross_up_rate,dtc_pct_of_grossed_up\n"
            "2014,BC,eligible,0.38,0.10\n"
        )
        p = _write_csv(csv)
        try:
            schedule = DividendTaxCreditSchedule.from_csv(p)
            rates = schedule.get_rates(2014, "eligible")
            assert rates == DividendRates("eligible", 0.38, 0.10, None)
        finally:
            Path(p).unlink(missing_ok=True)
