import numpy as np
import pytest

from macro_data.readers.taxation.personal_income_tax.pit_schedule import compute_progressive_tax
from macromodel.agents.individuals.individual_properties import ActivityStatus


class TestCentralGovernment:
    def test__create(self, test_central_government):
        assert test_central_government.country_name == "FRA"

    def test__central_government_states(self, test_central_government):
        assert test_central_government is not None
        for state in [
            "Value-added Tax",
            "Export Tax",
            "Employer Social Insurance Tax",
            "Employee Social Insurance Tax",
            "Profit Tax",
            "Income Tax",
            "Taxes Less Subsidies Rates",
        ]:
            assert state in test_central_government.states.keys()

    def test__central_government_ts(self, test_central_government):
        for ts_key in [
            "unemployment_benefits_by_individual",
            "total_other_benefits",
        ]:
            assert ts_key in test_central_government.ts.get_keys()

    def test__distribute_unemployment_benefits_to_individuals(self, test_central_government):
        benefits = test_central_government.ts.current("unemployment_benefits_by_individual")
        assert np.allclose(
            test_central_government.distribute_unemployment_benefits_to_individuals(
                current_individual_activity_status=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
            ),
            np.array([0.0, benefits[0]]),
        )


class TestCentralGovernmentPIT:
    """Progressive PIT: state storage, tax computation, and effective-rate update."""


    def test_flat_config_has_no_pit_states(self, test_central_government):
        """Without pit_brackets, pit_thresholds/rates are absent."""
        assert "pit_thresholds" not in test_central_government.states
        assert "pit_rates" not in test_central_government.states


    def test_compute_taxes_effective_rate_update(self, test_central_government_pit):
        """After compute_taxes, the effective Income Tax rate is
        consistent with the progressive schedule."""
        cg = test_central_government_pit

        emp_income = np.array([50000.0, 50000.0])
        activity = np.array([ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED])

        cg.compute_taxes(
            current_ind_employee_income=emp_income,
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.zeros(2),
            current_ind_activity=activity,
            current_ind_realised_cons=np.zeros(2),
            current_bank_profits=np.zeros(1),
            current_firm_production=np.zeros(1),
            current_firm_price=np.ones(1),
            current_firm_profits=np.zeros(1),
            current_firm_industries=np.zeros(1, dtype=int),
            current_household_new_real_wealth=np.zeros(1),
            taxes_less_subsidies_rates=np.zeros(1),
            current_total_exports=0.0,
        )

        # Recompute the expected effective rate from the tax paid
        taxable = emp_income * (1 - cg.states["Employee Social Insurance Tax"])
        pit = compute_progressive_tax(
            taxable,
            cg.states["pit_thresholds"],
            cg.states["pit_rates"],
        )
        expected_rate = float(pit.sum() / taxable.sum())

        assert cg.states["Income Tax"] == pytest.approx(expected_rate, rel=1e-10), (
            f"Effective rate {cg.states['Income Tax']} != expected {expected_rate}"
        )


    # pit_tax_credits (multi-component)




    def test_credits_applied_when_only_taxable_pool_supplied(
        self, test_central_government_pit_full,
    ):
        """Supplying taxable_income_per_ind but omitting credit_base_per_ind
        must still apply configured credits (the missing pool is built),
        not leak gross PIT through."""
        cg = test_central_government_pit_full

        emp_income = np.array([50000.0, 50000.0])
        activity = np.array([ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED])
        taxable = emp_income * (1 - cg.states["Employee Social Insurance Tax"])

        cg.compute_taxes(
            current_ind_employee_income=emp_income,
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.zeros(2),
            current_ind_activity=activity,
            current_ind_realised_cons=np.zeros(2),
            current_bank_profits=np.zeros(1),
            current_firm_production=np.zeros(1),
            current_firm_price=np.ones(1),
            current_firm_profits=np.zeros(1),
            current_firm_industries=np.zeros(1, dtype=int),
            current_household_new_real_wealth=np.zeros(1),
            taxes_less_subsidies_rates=np.zeros(1),
            current_total_exports=0.0,
            # Pool A provided, Pool B intentionally omitted.
            taxable_income_per_ind=taxable,
        )

        tax = cg.ts.get_aggregate("taxes_income")[-1]
        pit_raw = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"],
        ).sum()
        # Credits must have been applied → net tax strictly below gross PIT.
        assert tax < pit_raw
        expected_reduction = 9869.0 * float(cg.states["pit_rates"][0]) * 2
        assert pit_raw - tax == pytest.approx(expected_reduction, rel=1e-10)

    def test_tax_credits_floor_at_zero(
        self, test_central_government_pit_full,
    ):
        """Tax credit is non-refundable: tax floored at 0."""
        cg = test_central_government_pit_full

        emp_income = np.array([5000.0])
        activity = np.array([ActivityStatus.EMPLOYED])

        cg.compute_taxes(
            current_ind_employee_income=emp_income,
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.zeros(1),
            current_ind_activity=activity,
            current_ind_realised_cons=np.zeros(1),
            current_bank_profits=np.zeros(1),
            current_firm_production=np.zeros(1),
            current_firm_price=np.ones(1),
            current_firm_profits=np.zeros(1),
            current_firm_industries=np.zeros(1, dtype=int),
            current_household_new_real_wealth=np.zeros(1),
            taxes_less_subsidies_rates=np.zeros(1),
            current_total_exports=0.0,
        )

        last_tax = cg.ts.get_aggregate("taxes_income")[-1]
        assert last_tax == pytest.approx(0.0, abs=1e-6)

    def test_compute_pit_deductions_lower_bracket_base(self, test_central_government_pit):
        """Taxable-income deductions reduce the base *before* the brackets,
        so compute_pit taxes (income - deduction)."""
        cg = test_central_government_pit
        taxable = np.array([40000.0])

        pit_no_deduction = cg.compute_pit(taxable.copy())

        cg.states["pit_taxable_income_deductions"] = 5000.0
        pit_with_deduction = cg.compute_pit(taxable.copy())

        assert pit_with_deduction < pit_no_deduction
        expected = compute_progressive_tax(
            np.array([35000.0]),
            cg.states["pit_thresholds"],
            cg.states["pit_rates"],
        ).sum()
        assert pit_with_deduction == pytest.approx(expected)







    # Pre-calibration: effective rate from employee income




