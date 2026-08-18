"""Annualized assessment of the progressive PIT, and the year-end filing.

Income arrives per period but the brackets and credits are annual, so the pools
are scaled to a yearly rate before assessment and the tax is apportioned back.
"""

from types import SimpleNamespace

import warnings

import numpy as np
import pytest

from macro_data.readers.taxation import TaxationDataWarning
from macromodel.agents.central_government.central_government import (
    FILING_DEPENDENT_FLAGS,
    CentralGovernment,
)

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


def _advance(gov):
    """Advance the government's shared period counter.

    The runtime does this once per period in ``compute_taxes``. These tests
    drive ``_reconcile_tax_year`` directly, so they own the advance -- which is
    the point of the counter being government-owned rather than PIT-private.
    """
    gov.states["tax_step"] = int(gov.states.get("tax_step", 0)) + 1


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
            _advance(gov)
            pool = income * F
            out: list = []
            tax = _period_tax(gov, pool, out)
            # The settlement is per individual; this test books the period's
            # revenue, which is its sum.
            tax += float(
                np.sum(
                    CentralGovernment._reconcile_tax_year(
                        gov, pool, out[0], None, None, steps_per_year=F
                    )
                )
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


class TestSettlementIsPerIndividual:
    """The filing returns a settlement PER INDIVIDUAL, not a total.

    The shape is load-bearing rather than cosmetic: the refundable credit has to
    reach the individuals who earned it, and a total cannot be attributed back
    without a second allocation pass. Every assertion below fails if the return
    collapses to a scalar or is broadcast from one.
    """

    def _settle(self, quarterly, gov=None):
        """Run a year of *quarterly* pools and return the filing settlement."""
        gov = gov or _gov(reconcile=True)
        settlement = None
        for income in quarterly:
            _advance(gov)
            pool = income * F
            out: list = []
            _period_tax(gov, pool, out)
            settlement = CentralGovernment._reconcile_tax_year(
                gov, pool, out[0], None, None, steps_per_year=F
            )
        return settlement

    def test_entries_differ_when_the_individuals_do(self):
        # One earns only in Q1, one earns evenly. Annualizing Q1 income over the
        # year over-withholds the first and not the second, so their settlements
        # must differ -- a broadcast scalar would make them equal.
        front_loaded = np.array([80000.0, 20000.0])
        idle = np.array([0.0, 20000.0])
        settlement = self._settle([front_loaded, idle, idle, idle, idle])

        assert settlement[0] != settlement[1]
        # The front-loaded individual over-withheld, so is refunded.
        assert settlement[0] < 0.0

class TestSharedPeriodCounter:
    """``tax_step`` belongs to the government, not to the PIT.

    A second tax with a year boundary reads the same count. Each property below
    is one way the counter could go back to being PIT-private without anything
    else noticing.
    """

    def test_reconcile_reads_the_counter_and_never_writes_it(self):
        # Writing it here is what made it PIT-private: the write sat below an
        # early return, so the count froze whenever PIT's reconciliation was off
        # and a differently-gated tax would have stalled with it.
        gov = _gov(reconcile=True)
        gov.states["tax_step"] = 7
        CentralGovernment._reconcile_tax_year(
            gov, np.zeros(2), np.zeros(2), None, None, steps_per_year=F
        )
        assert gov.states["tax_step"] == 7

class TestCreditsDeferredToTheFiling:
    """``pit_credits_at_filing`` withholds GROSS and credits once, at the filing."""

    BASE = np.array([12000.0, 12000.0])

    def _period(self, defer):
        gov = _gov()
        gov.states["pit_credits_at_filing"] = defer
        pool = np.array([40000.0, 40000.0]) * F
        return _period_tax(gov, pool, None), gov

    def test_withholding_is_gross_when_credits_are_deferred(self):
        gov = _gov()
        gov.states["pit_credits_at_filing"] = True
        pool = np.array([40000.0, 40000.0]) * F
        with_credit = CentralGovernment.compute_pit(
            gov, pool, self.BASE, None, steps_per_year=F
        )
        no_credit = CentralGovernment.compute_pit(
            gov, pool, None, None, steps_per_year=F
        )
        # Deferred: supplying a credit arm must not change what is withheld.
        assert with_credit == pytest.approx(no_credit)

class TestDeferredWorkFallsBackRatherThanFailing:
    """An off-condition DEGRADES; it does not raise.

    Tax functionality is on by default and turns off two ways -- absent or
    incomplete data, or the user switching it off -- and both are fallbacks.
    An earlier build refused instead, which turned an off-switch into a crash.
    """

    def _gov_with(self, **flags):
        gov = _gov()
        gov.states.update(flags)
        return gov

    def test_disables_deferral_and_warns_when_no_filing_runs(self):
        gov = self._gov_with(
            pit_credits_at_filing=True,
            pit_investment_at_year_end=True,
            pit_year_end_reconciliation=False,
        )
        with pytest.warns(TaxationDataWarning, match="Falling back"):
            CentralGovernment._fall_back_if_deferred_work_cannot_land(gov, True)
        assert gov.states["pit_credits_at_filing"] is False
        assert gov.states["pit_investment_at_year_end"] is False

class TestEffectiveRateFollowsTheAssessedPool:
    """``states["Income Tax"]`` is a SIDE EFFECT of the assessment, not a switch.

    No ``pit_wage_basis_effective_rate`` flag is built. The rate is
    ``total_annual_tax / total_taxable_base`` over whichever pool ``compute_pit``
    is handed, so narrowing the withheld pool makes it wage-based with nothing
    else to set -- and a flag could not make it blended again without a second
    assessment on a base the period no longer computes.
    """

    WAGES = np.array([30000.0, 30000.0])
    INVESTMENT = np.array([20000.0, 20000.0])

    def _rate(self, pool):
        gov = _gov()
        _period_tax(gov, pool * F)
        return gov.states["Income Tax"]

    def test_narrowing_the_pool_lowers_the_rate_to_a_wage_basis(self):
        blended = self._rate(self.WAGES + self.INVESTMENT)
        wage_only = self._rate(self.WAGES)
        # Progressive brackets: dropping the top slice of the base lowers the
        # average rate. Equality here would mean the pool never reached the rate.
        assert wage_only < blended

