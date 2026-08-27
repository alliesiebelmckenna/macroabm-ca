"""The payroll levy base on the progressive path.

The gross-up at the levy site is path-independent for the two social-insurance levies: they
are computed outside the progressive/flat branch, so they sit on the gross wage on both paths.
Only the income-tax component differs, coming from the PIT pool when a schedule is configured.
Upstream cannot cover this, because the progressive path does not exist there.
"""

import numpy as np
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus

EMPLOYER_COST = 100_000.0


def _set_progressive(government, active: bool) -> None:
    if active:
        government.states["pit_uppers"] = np.array([1.0e12])
        government.states["pit_rates"] = np.array([0.15])
    else:
        government.states.pop("pit_uppers", None)
        government.states.pop("pit_rates", None)
    assert government.progressive_pit_active is active


def _levy(government, net_take_home: float) -> None:
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
        taxable_income_per_ind=np.zeros(1),
        withheld_income_per_ind=np.zeros(1),
        nrtc_base_per_ind=np.zeros(1),
        nrtc_direct_per_ind=np.zeros(1),
    )


def _rates(government) -> tuple[float, float, float]:
    return (
        government.states["Income Tax"],
        government.states["Employee Social Insurance Tax"],
        government.states["Employer Social Insurance Tax"],
    )


class TestPayrollLevyBaseOnTheProgressivePath:
    def test_social_insurance_sits_on_the_gross_wage_when_progressive(self, test_central_government):
        government = test_central_government
        income_tax, employee_si, employer_si = _rates(government)
        gross = EMPLOYER_COST / (1 + employer_si)
        net = gross * (1 - employee_si) * (1 - income_tax)
        _set_progressive(government, True)

        _levy(government, net)

        assert government.ts.current("taxes_employee_si")[0] == pytest.approx(employee_si * gross)
        assert government.ts.current("taxes_employer_si")[0] == pytest.approx(employer_si * gross)

    def test_the_social_insurance_base_does_not_depend_on_the_path(self, test_central_government):
        government = test_central_government
        income_tax, employee_si, employer_si = _rates(government)
        net = (EMPLOYER_COST / (1 + employer_si)) * (1 - employee_si) * (1 - income_tax)

        _set_progressive(government, False)
        _levy(government, net)
        flat = (
            government.ts.current("taxes_employee_si")[0],
            government.ts.current("taxes_employer_si")[0],
        )

        _set_progressive(government, True)
        _levy(government, net)
        progressive = (
            government.ts.current("taxes_employee_si")[0],
            government.ts.current("taxes_employer_si")[0],
        )

        assert progressive == pytest.approx(flat)

    def test_income_tax_comes_from_the_pool_when_progressive(self, test_central_government):
        # The levy-site gross-up must not reach taxes_income on this path: the pool supplies it,
        # and with an empty pool the only income-tax revenue is zero.
        government = test_central_government
        income_tax, employee_si, employer_si = _rates(government)
        net = (EMPLOYER_COST / (1 + employer_si)) * (1 - employee_si) * (1 - income_tax)
        _set_progressive(government, True)

        _levy(government, net)

        assert government.ts.current("taxes_income")[0] == pytest.approx(0.0)
