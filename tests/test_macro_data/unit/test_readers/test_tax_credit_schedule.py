"""Unit tests for TaxCreditSchedule statutory lookup over a multi-year file.

A multi-year credit schedule must return ONLY the requested year's actual
components (statutory lookup), not a mix of every year's rows compounded from
the base year.
"""

from pathlib import Path

import pytest

from macro_data.readers.taxation.personal_income_tax.tax_credit_schedule import (
    TaxCreditSchedule,
)


def _multiyear_csv(tmp_path) -> Path:
    # Distinct per-year Personal Amount so lookup is distinguishable from a
    # compounded/mixed result. 2016 also carries a different valuation `rate`.
    p = tmp_path / "credits_multi.csv"
    p.write_text(
        "tax_year,geo,credit,amount,top,rate,clawback,clawback_rate,index\n"
        "2014,BC,Personal Amount,9000,,0.0506,,,1\n"
        "2015,BC,Personal Amount,9500,,0.0506,,,1\n"
        "2016,BC,Personal Amount,12000,,0.0560,,,1\n"
    )
    return p


class TestCreditStatutoryLookup:
    @staticmethod
    def _personal(sched, year):
        return [
            c for c in sched.get_credits(tax_year=year) if c.kind == "Personal Amount"
        ]

    def test_lookup_2015(self, tmp_path):
        sched = TaxCreditSchedule.from_csv(_multiyear_csv(tmp_path))
        comps = self._personal(sched, 2015)
        assert len(comps) == 1  # only 2015's row, not all three years
        assert comps[0].amount == pytest.approx(9500)

    def test_lookup_2016(self, tmp_path):
        sched = TaxCreditSchedule.from_csv(_multiyear_csv(tmp_path))
        comps = self._personal(sched, 2016)
        assert len(comps) == 1
        assert comps[0].amount == pytest.approx(12000)

    def test_base_year_2014_not_polluted(self, tmp_path):
        sched = TaxCreditSchedule.from_csv(_multiyear_csv(tmp_path))
        comps = self._personal(sched, 2014)
        assert len(comps) == 1
        assert comps[0].amount == pytest.approx(9000)

    def test_base_year_is_minimum_year_even_when_unsorted(self, tmp_path):
        """A valid but unsorted CSV (later year block first) must not shift
        the base year or the base credit set."""
        path = tmp_path / "unsorted_credits.csv"
        path.write_text(
            "tax_year,geo,credit,amount,index\n"
            "2016,BC,Personal Amount,12000,1\n"
            "2014,BC,Personal Amount,9000,1\n"
        )
        sched = TaxCreditSchedule.from_csv(path)
        assert sched.base_year == 2014
        assert sched.credits[0].amount == pytest.approx(9000)

    def test_consolidated_geo_format_loads_bc_rows(self):
        """The contributor's consolidated credit file loads BC rows cleanly."""
        sched = TaxCreditSchedule.from_name(
            "non_refundable_tax_credits.csv",
            schedule_dir=Path(__file__).resolve().parents[4]
            / "spoof_data"
            / "freda"
            / "personal_income_tax",
            jurisdiction="bc",
        )
        comps = self._personal(sched, 2014)
        assert len(comps) == 1
        assert comps[0].amount == pytest.approx(9869)
