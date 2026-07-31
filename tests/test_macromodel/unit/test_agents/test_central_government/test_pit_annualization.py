"""Annualized assessment of the progressive PIT, and the year-end filing.

Income arrives per period but the brackets and credits are annual, so the pools
are scaled to a yearly rate before assessment and the tax is apportioned back.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.central_government.central_government import CentralGovernment

UPPERS = np.array([50000.0, np.inf])
RATES = np.array([0.10, 0.20])
F = 4.0


def _gov(reconcile: bool = False) -> SimpleNamespace:
    """A stand-in carrying only the states the PIT core reads."""
    return SimpleNamespace(
        states={
            "pit_uppers": UPPERS,
            "pit_rates": RATES,
            "pit_year_end_reconciliation": reconcile,
        }
    )


def _period_tax(gov, annual_pool, out=None):
    return CentralGovernment.compute_pit(
        gov, annual_pool, None, None, steps_per_year=F, out_tax_per_ind=out
    )


class TestApportionment:
    def test_period_owes_its_share_of_the_annual_liability(self):
        pool = np.array([10000.0, 30000.0]) * F
        annual = CentralGovernment.compute_pit(_gov(), pool, None, None)
        assert _period_tax(_gov(), pool) == pytest.approx(annual / F)


class TestYearEndFiling:
    """Steps 1..F are a calendar year; the filing at F+1 settles it."""

    def _year(self, quarterly, reconcile=True):
        gov = _gov(reconcile)
        booked = []
        for income in quarterly:
            pool = income * F
            out: list = []
            tax = _period_tax(gov, pool, out)
            tax += CentralGovernment._reconcile_tax_year(
                gov, pool, out[0], None, None, steps_per_year=F
            )
            booked.append(tax)
        return np.array(booked), gov

    def test_uneven_income_is_refunded_at_the_filing(self):
        earning = np.array([40000.0, 40000.0])
        idle = np.zeros(2)
        booked, _ = self._year([earning, earning, idle, idle, idle])

        true_annual = CentralGovernment.compute_pit(
            _gov(), earning * 2, None, None
        )
        assert booked[4] < 0.0
        assert booked[:5].sum() == pytest.approx(true_annual)
