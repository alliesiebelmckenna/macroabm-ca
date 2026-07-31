"""End-to-end wiring for the annualized PIT and the year-end filing.

Unit tests cover the arithmetic; these drive the real objects through the real
construction path, so a seam that comes loose fails here rather than passing
quietly.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from macro_data.readers.taxation import TaxationReader
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import (
    CentralGovernmentConfiguration,
    CountryConfiguration,
    ExchangeRatesConfiguration,
)
from macromodel.country import Country
from macromodel.exchange_rates import ExchangeRates
from macromodel.sim_calendar import steps_per_year

_COMMITTED_PIT_DIR = (
    Path(__file__).resolve().parents[4]
    / "spoof_data" / "freda" / "personal_income_tax"
)


def _build_country(datawrapper, **cg_kwargs):
    """A real country with the progressive schedule active."""
    base = datawrapper.synthetic_countries["FRA"]
    synthetic_country = dataclasses.replace(
        base, taxation=TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction="bc")
    )
    country_configuration = CountryConfiguration(
        central_government=CentralGovernmentConfiguration(
            activate_progressive_pit=True, **cg_kwargs
        ),
    )
    exchange_rates = ExchangeRates.from_data(
        exchange_rates_data=datawrapper.exchange_rates,
        exchange_rate_config=ExchangeRatesConfiguration(),
        initial_year=2014,
        country_names=["FRA"],
    )
    emission_factors = np.array(
        [
            datawrapper.emission_factors["coal"],
            datawrapper.emission_factors["gas"],
            datawrapper.emission_factors["oil"],
        ]
    )
    return Country.from_pickled_country(
        synthetic_country=synthetic_country,
        country_configuration=country_configuration,
        exchange_rates=exchange_rates,
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        industries=datawrapper.industries,
        initial_year=datawrapper.configuration.year,
        t_max=12,
        running_multiple_countries=False,
        emission_factors_usd=emission_factors,
    )


def _charge(cg, taxable, credit_base=None, direct_credits=None):
    """Drive a period through the real tax entry point and return the revenue."""
    n = len(taxable)
    cg.compute_taxes(
        current_ind_employee_income=np.zeros(n),
        current_total_rent_paid=0.0,
        current_income_financial_assets=np.zeros(n),
        current_ind_activity=np.full(n, ActivityStatus.EMPLOYED),
        current_ind_realised_cons=np.zeros(n),
        current_bank_profits=np.zeros(1),
        current_firm_production=np.zeros(1),
        current_firm_price=np.ones(1),
        current_firm_profits=np.zeros(1),
        current_firm_industries=np.zeros(1, dtype=int),
        current_household_new_real_wealth=np.zeros(n),
        taxes_less_subsidies_rates=np.zeros(1),
        taxable_income_per_ind=taxable,
        credit_base_per_ind=(
            np.zeros_like(taxable) if credit_base is None else credit_base
        ),
        direct_credits_per_ind=direct_credits,
    )
    return cg.ts.get_aggregate("taxes_income")[-1]


class TestFilingAcrossRealPeriods:
    """State has to survive between calls, which a single-call test cannot show."""

    def test_no_settlement_reaches_revenue_when_the_filing_is_off(self, datawrapper):
        cg = _build_country(
            datawrapper, pit_year_end_reconciliation=False
        ).central_government
        f = int(steps_per_year())
        earning = np.full(2, float(cg.states["pit_uppers"][0])) * steps_per_year()
        idle = np.zeros(2)

        _charge(cg, earning)
        _charge(cg, earning)
        for _ in range(f - 2):
            _charge(cg, idle)

        assert _charge(cg, idle) == pytest.approx(0.0)


class TestSettlementIsRecorded:
    """The settlement is netted into taxes_income, so it needs its own series to
    be visible at all."""

    def test_a_refund_is_recorded_and_other_periods_are_zero(self, datawrapper):
        cg = _build_country(
            datawrapper, pit_year_end_reconciliation=True
        ).central_government
        f = int(steps_per_year())
        earning = np.full(2, float(cg.states["pit_uppers"][0])) * steps_per_year()
        idle = np.zeros(2)

        # Uneven income over-withholds, so the filing refunds (Jensen's direction).
        _charge(cg, earning)
        _charge(cg, earning)
        for _ in range(f - 2):
            _charge(cg, idle)
        during_year = float(cg.ts.current("pit_year_end_settlement")[0])

        _charge(cg, idle)  # first period of the new year: the filing
        at_filing = float(cg.ts.current("pit_year_end_settlement")[0])

        assert during_year == pytest.approx(0.0)
        assert at_filing < 0.0


class TestFilingUsesTheSettledYearsSchedule:
    """The schedule advances before the filing runs, so the year it settles has
    to be looked up rather than read from the live states."""

    def test_a_closed_year_is_settled_at_its_own_rates(self, datawrapper):
        cg = _build_country(
            datawrapper, pit_year_end_reconciliation=True
        ).central_government
        table = cg.states.get("pit_schedule_by_year")
        assert table, "the real government must carry a per-year schedule table"

        f = int(steps_per_year())
        cg.set_pit_for_year(2014)
        # The stamp is what lets the filing find the year it settles; without it
        # the lookup silently falls back to the live states.
        assert cg.states["pit_calendar_year"] == 2014
        settled_rates = np.array(cg.states["pit_rates"], dtype=float)

        # Force a rise in the following year so the assertion can discriminate;
        # without a difference the test would pass either way. Copied rather
        # than mutated in place: a gap year resolves to the prior year's
        # fragment, and editing that would corrupt the year under test.
        following = dict(cg._pit_schedule_for_year(2015))
        following["pit_rates"] = settled_rates + 0.10
        table[2015] = following

        level = np.full(2, 40000.0)
        for _ in range(f):
            _charge(cg, level)

        cg.set_pit_for_year(2015)
        assert not np.allclose(cg.states["pit_rates"], settled_rates)
        assert np.allclose(cg._pit_schedule_for_year(2014)["pit_rates"], settled_rates)

        # Idle, so the period books only the settlement. Level income was
        # withheld exactly, leaving nothing to settle; assessed at the following
        # year's higher rates the closed year would owe, and the settlement is
        # taken at the filing, so it would appear here.
        _charge(cg, np.zeros(2))
        assert float(cg.ts.current("pit_year_end_settlement")[0]) == pytest.approx(0.0)
