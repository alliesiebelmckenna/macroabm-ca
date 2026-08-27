"""The progressive path withholds once, and the money goes where it is withheld from.

Two income-tax withholdings used to coexist: the wage setter's flat effective rate, netted out
of pay, and the PIT pool's own, booked as revenue. The flat one is a wage-setting quantity with
no counterpart in the tax system, so it is gone on the progressive path. What the pool withholds
now leaves the household's pay, which it never did before.

The flat path is unchanged: there the wage setter's deduction IS the tax.
"""

import numpy as np
import pytest

from macromodel.agents.central_government.func.pit_pools import (
    PitContext,
    build_taxable_income_pool,
    build_withheld_income_pool,
)
from macromodel.agents.firms.func.wage_setter import WorkEffortFirmWageSetter
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations.firms_configuration import WageSetter

EMPLOYER_COST = 100_000.0
WAGE_SETTER = WorkEffortFirmWageSetter(**WageSetter().parameters)


def _set_progressive(country, active: bool) -> None:
    states = country.central_government.states
    if active:
        states["pit_uppers"] = np.array([1.0e12])
        states["pit_rates"] = np.array([0.15])
    else:
        states.pop("pit_uppers", None)
        states.pop("pit_rates", None)
    assert country.central_government.progressive_pit_active is active


def _take_home(income_tax: float, employee_si: float, employer_si: float) -> float:
    """What the wage setter pays out, driven through the wage setter itself."""
    return WAGE_SETTER.set_employee_income(
        corresponding_firm=np.array([0]),
        current_individual_labour_inputs=np.array([1.0]),
        current_individual_stating_new_job=np.array([False]),
        current_employee_income=np.array([0.0]),
        current_individual_offered_wage=np.array([0.0]),
        current_target_production=np.array([1.0]),
        current_limiting_intermediate_inputs=np.array([1.0]),
        current_limiting_capital_inputs=np.array([1.0]),
        labour_inputs_from_employees=np.array([1.0]),
        industry_labour_productivity_by_firm=np.array([1.0]),
        initial_wage_per_capita=np.array([EMPLOYER_COST]),
        current_wage_per_capita=np.array([EMPLOYER_COST]),
        current_labour_productivity_factor=np.array([1.0]),
        prev_labour_productivity_factor=np.array([1.0]),
        current_wage_tightness_markup=np.array([0.0]),
        estimated_ppi_inflation=0.0,
        income_taxes=income_tax,
        employee_social_insurance_tax=employee_si,
        employer_social_insurance_tax=employer_si,
    )[0]


class TestTheWageSetterStopsWithholdingIncomeTax:
    def test_the_rate_is_zero_on_the_progressive_path(self, test_country):
        _set_progressive(test_country, True)
        assert test_country._wage_income_tax_rate() == 0.0

    def test_the_flat_path_keeps_it(self, test_country):
        _set_progressive(test_country, False)
        expected = test_country.central_government.states["Income Tax"]
        assert test_country._wage_income_tax_rate() == expected

    def test_pay_is_then_the_gross_wage_less_employee_social_insurance_only(self, test_country):
        _set_progressive(test_country, True)
        employee_si = test_country.central_government.states["Employee Social Insurance Tax"]
        employer_si = test_country.central_government.states["Employer Social Insurance Tax"]

        pay = _take_home(test_country._wage_income_tax_rate(), employee_si, employer_si)

        gross = EMPLOYER_COST / (1 + employer_si)
        assert pay == pytest.approx(gross * (1 - employee_si))


class TestThePoolNoLongerDeductsSocialInsuranceTwice:
    def test_the_withheld_pool_is_the_pay_it_is_given(self):
        pay = np.array([1_000.0])
        pool = build_withheld_income_pool(PitContext(employee_income=pay, employee_si_rate=0.05))
        assert pool == pytest.approx(pay)

    def test_the_taxable_pool_is_too_when_it_is_the_only_stream(self):
        pay = np.array([1_000.0])
        pool = build_taxable_income_pool(PitContext(employee_income=pay, employee_si_rate=0.05))
        assert pool == pytest.approx(pay)


class TestTheWithholdingReachesHouseholds:
    def _withhold(self, country, per_individual) -> None:
        country.central_government.states["pit_withheld_per_ind"] = per_individual

    def test_it_is_a_debit_aggregated_to_the_right_household(self, test_country):
        corr = np.asarray(test_country.individuals.states["Corresponding Household ID"]).astype(int)
        withheld = np.zeros(len(corr))
        withheld[0] = 300.0
        self._withhold(test_country, withheld)

        delivered = test_country._pit_withheld_per_household()

        assert delivered[corr[0]] == pytest.approx(-300.0)
        assert delivered.sum() == pytest.approx(-300.0)

    def test_nothing_is_withheld_on_the_flat_path(self, test_country):
        self._withhold(test_country, 0.0)
        assert test_country._pit_withheld_per_household().sum() == pytest.approx(0.0)

    def test_both_income_legs_carry_it_because_pay_is_known_when_planning(self, test_country):
        households = test_country.households
        n_hh = int(households.ts.current("n_households"))
        debit = np.full(n_hh, -25.0)

        before_expected = households.compute_expected_income().sum()
        before_realised = households.compute_income().sum()
        households.ts.income_pit_withheld.append(debit)

        assert households.compute_expected_income().sum() - before_expected == pytest.approx(debit.sum())
        assert households.compute_income().sum() - before_realised == pytest.approx(debit.sum())


class TestTheCircuitCloses:
    def test_employer_cost_equals_pay_plus_everything_the_government_books(self, test_central_government):
        # C - N == employer SI + employee SI + PIT withheld, with nothing left over for nobody.
        government = test_central_government
        employee_si = government.states["Employee Social Insurance Tax"]
        employer_si = government.states["Employer Social Insurance Tax"]
        government.states["pit_uppers"] = np.array([1.0e12])
        government.states["pit_rates"] = np.array([0.15])

        pay = _take_home(0.0, employee_si, employer_si)
        pool = build_withheld_income_pool(
            PitContext(employee_income=np.array([pay]), employee_si_rate=employee_si)
        )

        government.compute_taxes(
            current_ind_employee_income=np.array([pay]),
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.zeros(1),
            current_ind_activity=np.array([ActivityStatus.EMPLOYED]),
            current_ind_realised_cons=np.zeros(1),
            current_bank_profits=np.zeros(1),
            current_firm_production=np.zeros(1),
            current_firm_price=np.zeros(1),
            current_firm_profits=np.zeros(1),
            current_firm_industries=np.zeros(1, dtype=int),
            current_household_new_real_wealth=np.zeros(1),
            taxes_less_subsidies_rates=np.zeros(1),
            current_total_exports=0.0,
            taxable_income_per_ind=pool,
            withheld_income_per_ind=pool,
            nrtc_base_per_ind=np.zeros(1),
            nrtc_direct_per_ind=np.zeros(1),
        )

        withheld = float(np.sum(government.states["pit_withheld_per_ind"]))
        booked = (
            government.ts.current("taxes_employer_si")[0]
            + government.ts.current("taxes_employee_si")[0]
            + government.ts.current("taxes_income")[0]
        )
        household_receives = pay - withheld

        assert EMPLOYER_COST - household_receives == pytest.approx(booked)
