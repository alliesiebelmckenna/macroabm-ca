import numpy as np
import pytest

from macro_data.readers.taxation.personal_income_tax.pit_schedule import compute_personal_income_tax
from macromodel.agents.central_government.pit_pools import (
    PitContext,
    build_credit_base_pool,
    build_taxable_income_pool,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus


def _pools_for(cg, employee_income):
    """Assemble the two PIT pools the way the country step does.

    ``compute_taxes`` requires both; only the processing phase holds the
    household context they depend on.
    """
    ctx = PitContext(
        employee_income=employee_income,
        employee_si_rate=float(cg.states["Employee Social Insurance Tax"]),
    )
    taxable = build_taxable_income_pool(ctx)
    credits = build_credit_base_pool(
        cg.states.get("pit_non_refundable_tax_credits"), taxable, ctx
    )
    return taxable, credits


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
        """Without pit_brackets, pit_uppers/rates are absent."""
        assert "pit_uppers" not in test_central_government.states
        assert "pit_rates" not in test_central_government.states



    def test_compute_taxes_effective_rate_update(self, test_central_government_pit):
        """After compute_taxes, the effective Income Tax rate is
        consistent with the progressive schedule."""
        cg = test_central_government_pit

        emp_income = np.array([50000.0, 50000.0])
        activity = np.array([ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED])
        taxable, credits = _pools_for(cg, emp_income)

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
            taxable_income_per_ind=taxable,
            credit_base_per_ind=credits,
        )

        # Recompute the expected effective rate from the tax paid
        taxable = emp_income * (1 - cg.states["Employee Social Insurance Tax"])
        pit = compute_personal_income_tax(
            taxable,
            cg.states["pit_uppers"],
            cg.states["pit_rates"],
        )
        expected_rate = float(pit.sum() / taxable.sum())

        assert cg.states["Income Tax"] == pytest.approx(expected_rate, rel=1e-10), (
            f"Effective rate {cg.states['Income Tax']} != expected {expected_rate}"
        )


    # pit_non_refundable_tax_credits (multi-component)




    def test_missing_pool_raises_rather_than_assembling_one(
        self, test_central_government_pit_full,
    ):
        """A pool the caller failed to supply is an error, not something to
        assemble here: only the processing phase holds the household context
        and the dividend items, so a silently self-built pool could omit a
        dividend's credit while taxing its grossed-up amount."""
        cg = test_central_government_pit_full

        emp_income = np.array([50000.0, 50000.0])
        taxable, _ = _pools_for(cg, emp_income)

        with pytest.raises(ValueError, match="requires both pools"):
            cg.compute_taxes(
                current_ind_employee_income=emp_income,
                current_total_rent_paid=0.0,
                current_income_financial_assets=np.zeros(2),
                current_ind_activity=np.array(
                    [ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED]
                ),
                current_ind_realised_cons=np.zeros(2),
                current_bank_profits=np.zeros(1),
                current_firm_production=np.zeros(1),
                current_firm_price=np.ones(1),
                current_firm_profits=np.zeros(1),
                current_firm_industries=np.zeros(1, dtype=int),
                current_household_new_real_wealth=np.zeros(1),
                taxes_less_subsidies_rates=np.zeros(1),
                current_total_exports=0.0,
                # Pool A supplied, Pool B omitted.
                taxable_income_per_ind=taxable,
            )

    def test_tax_credits_floor_at_zero(
        self, test_central_government_pit_full,
    ):
        """Tax credit is non-refundable: tax floored at 0."""
        cg = test_central_government_pit_full

        emp_income = np.array([5000.0])
        activity = np.array([ActivityStatus.EMPLOYED])
        taxable, credits = _pools_for(cg, emp_income)

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
            taxable_income_per_ind=taxable,
            credit_base_per_ind=credits,
        )

        last_tax = cg.ts.get_aggregate("taxes_income")[-1]
        assert last_tax == pytest.approx(0.0, abs=1e-6)








    # Pre-calibration: effective rate from employee income




