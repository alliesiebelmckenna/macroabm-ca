"""Unit tests for the base the payroll levies are applied to.

The wage setter returns take-home pay, so applying income tax and the two social-insurance
levies to that series charged them on income from which they had already been deducted.
"""

import numpy as np
import pytest

from macromodel.agents.firms.func.wage_setter import WorkEffortFirmWageSetter
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations.firms_configuration import WageSetter

EMPLOYER_COST = 100_000.0

# Built from the shipped configuration so the test tracks their defaults rather than copies.
WAGE_SETTER = WorkEffortFirmWageSetter(**WageSetter().parameters)


def _net_take_home(employer_cost: float, income_tax: float, employee_si: float, employer_si: float) -> float:
    """Take-home pay for one employee, from the wage setter itself rather than a restatement."""
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
        initial_wage_per_capita=np.array([employer_cost]),
        current_wage_per_capita=np.array([employer_cost]),
        current_labour_productivity_factor=np.array([1.0]),
        prev_labour_productivity_factor=np.array([1.0]),
        current_wage_tightness_markup=np.array([0.0]),
        estimated_ppi_inflation=0.0,
        income_taxes=income_tax,
        employee_social_insurance_tax=employee_si,
        employer_social_insurance_tax=employer_si,
    )[0]


def _levy_on_one_employee(government, net_take_home: float) -> None:
    """Run the tax computation with one employed individual and every other base zeroed."""
    government.compute_taxes(
        current_ind_employee_income=np.array([net_take_home]),
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
    )


def _gross_wage(employer_si: float) -> float:
    return EMPLOYER_COST / (1 + employer_si)


class TestPayrollLevyBase:
    def test_wage_setter_returns_the_wage_net_of_both_levies(self):
        # The relation the levy site relies on: take-home is the gross wage less employee
        # social insurance and then income tax, the gross wage being employer cost less
        # employer social insurance.
        income_tax, employee_si, employer_si = 0.20, 0.05, 0.10

        net = _net_take_home(EMPLOYER_COST, income_tax, employee_si, employer_si)

        expected = _gross_wage(employer_si) * (1 - employee_si) * (1 - income_tax)
        assert net == pytest.approx(expected)

    def test_income_tax_collected_equals_what_the_wage_setter_deducted(self, test_central_government):
        government = test_central_government
        income_tax = government.states["Income Tax"]
        employee_si = government.states["Employee Social Insurance Tax"]
        employer_si = government.states["Employer Social Insurance Tax"]
        net = _net_take_home(EMPLOYER_COST, income_tax, employee_si, employer_si)

        _levy_on_one_employee(government, net)

        deducted_by_the_wage_setter = income_tax * (1 - employee_si) * _gross_wage(employer_si)
        assert government.ts.current("taxes_income")[0] == pytest.approx(deducted_by_the_wage_setter)

    def test_employee_social_insurance_sits_on_the_gross_wage(self, test_central_government):
        government = test_central_government
        employee_si = government.states["Employee Social Insurance Tax"]
        employer_si = government.states["Employer Social Insurance Tax"]
        net = _net_take_home(EMPLOYER_COST, government.states["Income Tax"], employee_si, employer_si)

        _levy_on_one_employee(government, net)

        expected = employee_si * _gross_wage(employer_si)
        assert government.ts.current("taxes_employee_si")[0] == pytest.approx(expected)
