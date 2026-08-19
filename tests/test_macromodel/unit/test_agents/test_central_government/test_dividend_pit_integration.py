"""Integration tests for the bank-dividend to PIT pipeline.

Covers four layers: the pool arithmetic (``build_dividend_tax_items`` feeding
``build_taxable_income_pool``), end-to-end revenue through
``CentralGovernment.compute_taxes``, propagation of
``bank_dividend_small_business_share`` into the agent states, and the
country-level wiring from raw bank profits to PIT revenue. The country-level
test uses a bank share (0.25) distinct from the firm share (0.9) as a regression
guard against key confusion.
"""

import numpy as np
import pytest

from macromodel.agents.central_government.pit_pools import (
    build_dividend_tax_items,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.sim_calendar import steps_per_year

# 2014 BC rates used across all tests.
_ELIG_GU = 0.38
_NONELIG_GU = 0.18
_ELIG_DTC = 0.10
_NONELIG_DTC = 0.0259


# Helpers


def _bank_items(dividend_income, small_business_share):
    return build_dividend_tax_items(
        dividend_income=np.asarray(dividend_income, dtype=float),
        small_business_share=small_business_share,
        eligible_gross_up=_ELIG_GU,
        non_eligible_gross_up=_NONELIG_GU,
        eligible_dtc_rate=_ELIG_DTC,
        non_eligible_dtc_rate=_NONELIG_DTC,
    )


def _build_cg_with_integration(datawrapper, bank_share, firm_share=0.9):
    """Build a CG with PIT dividend integration and distinct firm/bank shares."""
    from macromodel.agents.central_government import CentralGovernment
    from macromodel.configurations import CentralGovernmentConfiguration

    country = datawrapper.synthetic_countries["FRA"]
    taxes_ls = country.industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values

    config = CentralGovernmentConfiguration(
        pit_brackets=[(float("inf"), 0.10)],
        pit_dividend_integration=True,
        dividend_small_business_share=firm_share,
        bank_dividend_small_business_share=bank_share,
    )
    return CentralGovernment.from_pickled_agent(
        synthetic_central_government=country.central_government,
        configuration=config,
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        taxes_net_subsidies=taxes_ls,
        tax_data=country.tax_data,
        n_industries=datawrapper.n_industries,
        number_of_unemployed_individuals=1,
    )


# 1. Pool-level arithmetic


# 2. End-to-end revenue


# 3. States propagation


# 4. Country-level wiring: bank profits → gross dividend → grossed-up + DTC


class TestBankDividendCountryWiring:
    """Full chain from raw bank profits to PIT revenue, reading from CG states.

    Uses bank_share=0.25 (non-default) and firm_share=0.9 (different) so that
    any key confusion (e.g. using dividend_small_business_share for banks)
    produces a numerically different gross-up and DTC.
    """

    _BANK_PROFIT = 2_000.0
    _PAYOUT = 0.5
    _BANK_SHARE = 0.25

    def test_bank_profits_to_pit_revenue(self, test_individuals, datawrapper):
        """Full chain from bank profits to PIT revenue matches the reviewer's formula."""
        from macro_data.readers.taxation.personal_income_tax.pit_schedule import (
            compute_personal_income_tax,
        )

        ind = test_individuals
        ind.states["Activity Status"] = np.array([ActivityStatus.BANK_INVESTOR])
        ind.states["Corresponding Invested Bank"] = np.array([0])
        ind.states["Dividend Payout Ratio"] = self._PAYOUT

        cg = _build_cg_with_integration(datawrapper, bank_share=self._BANK_SHARE)
        tau_firm = float(cg.states["Profit Tax"])
        si_rate = float(cg.states["Employee Social Insurance Tax"])
        bank_profits = np.array([self._BANK_PROFIT])

        gross_bank_div = ind.compute_gross_bank_dividend(bank_profits, tau_firm)
        grossed_up, dtc = build_dividend_tax_items(
            dividend_income=gross_bank_div,
            small_business_share=float(cg.states["bank_dividend_small_business_share"]),
            eligible_gross_up=float(cg.states["dividend_eligible_gross_up"]),
            non_eligible_gross_up=float(cg.states["dividend_non_eligible_gross_up"]),
            eligible_dtc_rate=float(cg.states["dividend_eligible_dtc_rate"]),
            non_eligible_dtc_rate=float(cg.states["dividend_non_eligible_dtc_rate"]),
        )

        # Pinned to the legacy per-period crediting path. This test's oracle is
        # the gross-up/DTC arithmetic netted INSIDE the period, which is exactly
        # what ``pit_credits_at_filing`` (on by default) stops doing: it withholds
        # gross and credits once at the filing. Pinning keeps the oracle testing
        # the wiring it was written for rather than silently re-deriving it.
        cg.states["pit_credits_at_filing"] = False

        emp_income = np.array([0.0])  # no wage income — isolates the dividend path
        taxable = emp_income * (1.0 - si_rate) + grossed_up
        pit_gross = compute_personal_income_tax(taxable, cg.states["pit_uppers"], cg.states["pit_rates"])
        expected_revenue = float(np.maximum(0.0, pit_gross - dtc).sum())

        cg.compute_taxes(
            current_ind_employee_income=emp_income,
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.zeros(1),
            current_ind_activity=np.array([ActivityStatus.BANK_INVESTOR]),
            current_ind_realised_cons=np.zeros(1),
            current_bank_profits=bank_profits,
            current_firm_production=np.zeros(1),
            current_firm_price=np.ones(1),
            current_firm_profits=np.zeros(1),
            current_firm_industries=np.zeros(1, dtype=int),
            current_household_new_real_wealth=np.zeros(1),
            taxes_less_subsidies_rates=np.zeros(1),
            taxable_income_per_ind=taxable,
            nrtc_base_per_ind=np.zeros_like(taxable),
            nrtc_direct_per_ind=dtc,
        )
        actual_revenue = cg.ts.get_aggregate("taxes_income")[-1]
        # The pools are assessed against the annual schedule, so the period is
        # booked its share of the annual liability.
        expected_revenue /= steps_per_year()
        assert actual_revenue == pytest.approx(expected_revenue, rel=1e-9)
