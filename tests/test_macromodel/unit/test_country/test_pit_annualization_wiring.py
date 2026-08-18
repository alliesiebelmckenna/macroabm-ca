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

_COMMITTED_PIT_DIR = Path(__file__).resolve().parents[4] / "spoof_data" / "freda" / "personal_income_tax"


def _build_country(datawrapper, **cg_kwargs):
    """A real country with the progressive schedule active."""
    base = datawrapper.synthetic_countries["FRA"]
    synthetic_country = dataclasses.replace(
        base, taxation=TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction="bc")
    )
    country_configuration = CountryConfiguration(
        central_government=CentralGovernmentConfiguration(activate_progressive_pit=True, **cg_kwargs),
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
        nrtc_base_per_ind=(np.zeros_like(taxable) if credit_base is None else credit_base),
        nrtc_direct_per_ind=direct_credits,
    )
    return cg.ts.get_aggregate("taxes_income")[-1]


class TestFilingAcrossRealPeriods:
    """State has to survive between calls, which a single-call test cannot show."""

    def test_no_settlement_reaches_revenue_when_the_filing_is_off(self, datawrapper):
        # Reconciliation off is only a coherent configuration with nothing
        # deferred to the filing. Leaving the deferring flags at their defaults
        # here is what the fail-closed validator refuses, and rightly: the
        # credits would be granted nowhere and investment income never taxed.
        cg = _build_country(
            datawrapper,
            pit_year_end_reconciliation=False,
            pit_credits_at_filing=False,
            pit_investment_at_year_end=False,
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
        cg = _build_country(datawrapper, pit_year_end_reconciliation=True).central_government
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
        cg = _build_country(datawrapper, pit_year_end_reconciliation=True).central_government
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


class TestRefundableCreditReachesHouseholds:
    """The wiring, not the arithmetic: does the credit actually get PAID?

    The aggregation helper is unit-tested elsewhere. This covers the step the
    unit tests cannot: that the aggregated credit is added to the household
    transfer series at all. Removing it there leaves every unit test green while
    the credit is computed correctly and reaches nobody -- the least visible
    failure in this feature.
    """

    def test_the_model_actually_passes_a_refundable_callable(self):
        """The filing must RECEIVE `annual_rtc`, or nothing is ever computed.

        Found live: the callable was defined nowhere and the filing was called
        without it, so the whole refundable mechanism was unreachable from a
        running model while every unit test passed. This asserts the wire
        exists, which is the cheapest thing that would have caught it.
        """
        import inspect

        src = (
            inspect.getsource(Country.update_realised_metrics) + inspect.getsource(Country.compute_taxes_and_deficit)
            if hasattr(Country, "compute_taxes_and_deficit")
            else inspect.getsource(Country)
        )
        assert "annual_rtc=annual_rtc" in src, (
            "compute_taxes is called without annual_rtc, so the refundable credit is never valued"
        )

    def test_the_context_carries_tenure(self):
        """Without it the renter's credit grants nothing, silently.

        Also found live: `PitContext` gained a tenure field and the model never
        populated it, so the credit was reachable but always zero.
        """
        import inspect

        src = inspect.getsource(Country)
        assert "households_tenure=" in src, (
            "the PIT context is built without tenure, so the renter's credit has no signal and grants nothing"
        )

    def test_the_refundable_credits_reach_agent_states_scaled(self, datawrapper):
        """The whole chain on a real object: config -> scaled -> agent state.

        Four wires in this feature were missing, and each broke it the same
        silent way: the credit computed correctly and paid nobody. Two were
        exactly this -- the list never reached agent states, so the filing
        callable read None. Asserting the SCALED value covers both the seeding
        and the units in one assertion.
        """
        country = _build_country(datawrapper)
        rows = country.central_government.states.get("pit_refundable_tax_credits")

        assert rows, (
            "the refundable credits never reach agent states, so the filing "
            "callable reads None and the credit is never valued"
        )
        individual = next(r for r in rows if r["credit"] == "Eligible Individual Amount")
        # Per-person dollars are a few hundred; agent dollars are that times the
        # scale, so anything unscaled fails here.
        assert individual["amount"] > 1000.0, (
            f"amount {individual['amount']} looks like per-person dollars -- the "
            f"scaling seam did not reach the refundable list"
        )
        # Dimensionless, so it must NOT have moved.
        assert individual["clawback_rate"] == pytest.approx(0.02)

    def test_the_year_advance_swaps_the_refundable_credits(self, datawrapper):
        """A year-varying parameter must move with the year, not freeze.

        The amounts change every published year. Omitted from the swap they hold
        the construction year's values for the whole run -- 2014 amounts paid in
        2030, with nothing indicating it.
        """
        cg = _build_country(datawrapper).central_government
        if cg.states.get("pit_schedule_by_year") is None:
            pytest.skip("no per-year schedule table on this build")

        seen = {}
        for year in (2017, 2023):
            cg.set_pit_for_year(year)
            rows = cg.states.get("pit_refundable_tax_credits")
            assert rows, f"the refundable credits vanished at {year}"
            seen[year] = next(r["amount"] for r in rows if r["credit"] == "Eligible Individual Amount")
        assert seen[2017] != seen[2023], (
            f"the amount did not move between years ({seen}), so it is frozen at whichever year the model was built for"
        )
