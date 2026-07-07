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

    def test_pit_thresholds_and_rates_stored(self, test_central_government_pit):
        """PIT brackets are stored as states when pit_brackets is set."""
        cg = test_central_government_pit
        assert "pit_thresholds" in cg.states
        assert "pit_rates" in cg.states
        assert len(cg.states["pit_thresholds"]) == 6
        assert len(cg.states["pit_rates"]) == 6
        # Last threshold is inf
        assert np.isinf(cg.states["pit_thresholds"][-1])

    def test_flat_config_has_no_pit_states(self, test_central_government):
        """Without pit_brackets, pit_thresholds/rates are absent."""
        assert "pit_thresholds" not in test_central_government.states
        assert "pit_rates" not in test_central_government.states

    def test_compute_taxes_progressive_branch(self, test_central_government_pit):
        """Progressive PIT path: tax revenue > 0 and effective rate updated."""
        cg = test_central_government_pit

        # Two employed individuals: low earner (30k) and high earner (100k)
        emp_income = np.array([30000.0, 100000.0])
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

        # Income tax revenue should be positive
        last_tax = cg.ts.get_aggregate("taxes_income")[-1]
        assert last_tax > 0, f"Expected positive tax revenue, got {last_tax}"

        # Effective rate should be between the lowest and highest bracket rates
        rate = cg.states["Income Tax"]
        assert 0.05 < rate < 0.17, (
            f"Effective rate {rate:.4f} should be between 5% and 17%"
        )

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

    def test_progressive_vs_flat_tax_ordering(self, test_central_government_pit):
        """Progressive schedule taxes low earners less, high earners more
        than an equivalent flat rate."""
        cg = test_central_government_pit

        low_income = np.array([20000.0])
        high_income = np.array([200000.0])
        thresholds = cg.states["pit_thresholds"]
        rates = cg.states["pit_rates"]

        # The flat effective rate from compute_taxes (approximate)
        # We just check that the progressive tax is lower for low and
        # higher for high compared to the top marginal rate.
        low_tax = compute_progressive_tax(low_income, thresholds, rates)
        high_tax = compute_progressive_tax(high_income, thresholds, rates)

        low_effective = low_tax / low_income
        high_effective = high_tax / high_income

        # Progressive: low earner effective rate < high earner effective rate
        assert low_effective[0] < high_effective[0], (
            f"Low earner rate {low_effective[0]:.4f} should be < "
            f"high earner rate {high_effective[0]:.4f}"
        )
        # High earner effective rate should be > lowest bracket rate
        assert high_effective[0] > rates[0]
        # Low earner effective rate should be < highest bracket rate
        assert low_effective[0] < rates[-1]

    # ── pit_tax_credits (multi-component) ───────

    def test_pit_tax_credits_stored(self, test_central_government_pit_full):
        """pit_tax_credits is stored in states when configured."""
        cg = test_central_government_pit_full
        assert "pit_tax_credits" in cg.states
        assert len(cg.states["pit_tax_credits"]) == 1
        assert cg.states["pit_tax_credits"][0]["kind"] == "Personal Amount"
        assert cg.states["pit_tax_credits"][0]["amount"] == 9869.0

    def test_pit_tax_credits_not_stored_without_config(self, test_central_government_pit):
        """Without pit_tax_credits in config, state key is absent."""
        assert "pit_tax_credits" not in test_central_government_pit.states

    def test_tax_credits_reduce_tax_revenue(
        self, test_central_government_pit_full,
    ):
        """Non-refundable tax credits reduce tax revenue vs raw PIT."""
        cg = test_central_government_pit_full

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

        tax_with_credits = cg.ts.get_aggregate("taxes_income")[-1]

        taxable = emp_income * (1 - cg.states["Employee Social Insurance Tax"])
        pit_raw = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"],
        )
        assert tax_with_credits < pit_raw.sum(), (
            f"Tax with credits ({tax_with_credits:.2f}) should be "
            f"less than raw PIT ({pit_raw.sum():.2f})"
        )

        expected_credit = 9869.0 * 0.0506 * 2
        actual_reduction = pit_raw.sum() - tax_with_credits
        assert actual_reduction == pytest.approx(expected_credit, rel=1e-10)

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

    def test_compute_pit_credit_floored_at_zero(self, test_central_government_pit):
        """compute_pit never returns negative tax: a credit larger than the
        gross tax is clamped, not refunded."""
        cg = test_central_government_pit
        taxable = np.array([20000.0])
        gross = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"],
        ).sum()

        # Credit base whose value (x bottom rate) exceeds the gross tax.
        huge_credit_base = np.array([gross / float(cg.states["pit_rates"][0]) + 1000.0])
        total = cg.compute_pit(taxable, huge_credit_base)
        assert total == pytest.approx(0.0)

    def test_compute_pit_subtracts_direct_credits_2b(self, test_central_government_pit):
        """2b direct credits (the dividend tax credit) are subtracted from
        gross tax in addition to the 2a base credits."""
        cg = test_central_government_pit
        taxable = np.array([40000.0])
        gross = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"],
        ).sum()

        credit_base = np.array([1000.0])    # 2a base
        direct_credit = np.array([250.0])   # 2b dividend tax credit
        net = cg.compute_pit(taxable, credit_base, direct_credit)

        expected = gross - 1000.0 * float(cg.states["pit_rates"][0]) - 250.0
        assert net == pytest.approx(expected)

    def test_compute_pit_direct_credit_floored_with_base(self, test_central_government_pit):
        """2a + 2b are floored together — a large direct credit can't refund."""
        cg = test_central_government_pit
        taxable = np.array([20000.0])
        gross = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"],
        ).sum()
        net = cg.compute_pit(taxable, np.zeros(1), np.array([gross + 5000.0]))
        assert net == pytest.approx(0.0)

    def test_compute_pit_direct_credit_alone(self, test_central_government_pit):
        """A 2b direct credit applies even when there is no 2a base."""
        cg = test_central_government_pit
        taxable = np.array([40000.0])
        gross = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"],
        ).sum()
        net = cg.compute_pit(taxable, None, np.array([300.0]))
        assert net == pytest.approx(gross - 300.0)

    def test_dividend_integration_raises_pit_via_compute_taxes(self, test_central_government_pit):
        """Routing a grossed-up dividend + DTC through compute_taxes taxes the
        grossed-up amount (net of the credit), raising PIT revenue."""
        cg = test_central_government_pit
        emp = np.array([50000.0])
        base_kwargs = dict(
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.zeros(1),
            current_ind_activity=np.array([ActivityStatus.EMPLOYED]),
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
        cg.compute_taxes(current_ind_employee_income=emp, **base_kwargs)
        tax_no_div = cg.ts.get_aggregate("taxes_income")[-1]

        cg.compute_taxes(
            current_ind_employee_income=emp,
            grossed_up_dividend_per_ind=np.array([120.0]),  # grossed-up D=100, s=0.9
            direct_credits_per_ind=np.array([4.13058]),     # the matching DTC
            **base_kwargs,
        )
        tax_with_div = cg.ts.get_aggregate("taxes_income")[-1]

        # Exact: the grossed-up 120 stacks onto the wage taxable base, then the
        # DTC is subtracted.  (This fixture has no 2a credits or deductions.)
        si = float(cg.states["Employee Social Insurance Tax"])
        taxable = emp * (1 - si) + 120.0
        gross = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"],
        ).sum()
        assert tax_with_div == pytest.approx(gross - 4.13058)
        assert tax_with_div > tax_no_div

    def test_no_credits_noop(self, test_central_government_pit):
        """Without pit_tax_credits, gross tax = net tax (no credits)."""
        cg = test_central_government_pit

        emp_income = np.array([50000.0])
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

        tax = cg.ts.get_aggregate("taxes_income")[-1]
        taxable = emp_income * (1 - cg.states["Employee Social Insurance Tax"])
        expected = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"],
        ).sum()
        assert tax == pytest.approx(expected)

    # ── Pre-calibration: effective rate from employee income ─────

    def test_pre_calibration_effective_rate_in_country_construction(
        self, datawrapper, test_individuals,
    ):
        """When country.py constructs a Country with pit_brackets,
        the Income Tax effective rate is pre-calibrated at t=0."""
        from macromodel.agents.central_government import CentralGovernment
        from macromodel.configurations import CentralGovernmentConfiguration

        country_data = datawrapper.synthetic_countries["FRA"]

        pit_config = CentralGovernmentConfiguration(
            pit_brackets=[
                (37606, 0.0506),
                (75213, 0.077),
                (86354, 0.105),
                (104858, 0.1229),
                (150000, 0.147),
                (float("inf"), 0.168),
            ],
        )

        n_unemployed = np.sum(
            test_individuals.states["Activity Status"] == ActivityStatus.UNEMPLOYED
        )

        cg = CentralGovernment.from_pickled_agent(
            synthetic_central_government=country_data.central_government,
            configuration=pit_config,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            taxes_net_subsidies=country_data.industry_data["industry_vectors"][
                "Taxes Less Subsidies Rates"
            ].values,
            tax_data=country_data.tax_data,
            n_industries=datawrapper.n_industries,
            number_of_unemployed_individuals=n_unemployed,
        )

        # Effective rate must have been set — not the raw OECD average
        assert "Income Tax" in cg.states
        rate = cg.states["Income Tax"]
        assert 0.05 < rate < 0.17, (
            f"Pre-calibrated effective rate {rate:.4f} should be between 5% and 17%"
        )

    def test_pre_calibration_with_credits(
        self, test_central_government_pit_full,
    ):
        """With tax credits, the pre-calibrated effective rate is lower
        than brackets-only (credit reduces net tax but not taxable base)."""
        cg = test_central_government_pit_full

        rate = cg.states["Income Tax"]
        assert rate < 0.10, (
            f"Effective rate with credits ({rate:.4f}) should be < 10%"
        )
        assert rate > 0.0


class TestSetPitForYear:
    """``set_pit_for_year`` performs a statutory lookup against the per-year
    schedule table — swapping in each year's actual published brackets, rates,
    credits, and deductions.  It carries a marginal *rate* change (which a pure
    indexation of the base year cannot); a year before the table's first entry
    holds the first schedule; a gap year within the table's range holds the
    most recent published year; and a year beyond the table's last entry
    raises — there is no forward projection, so the caller must stop rather
    than silently run on an unfunded schedule.
    """

    @staticmethod
    def _credit(amount):
        return {
            "kind": "Personal Amount",
            "amount": amount,
            "indexing": True,
            "age_min": None,
            "clawback_start": None,
            "clawback_cap": None,
        }

    def _seed_table(self, cg):
        """Install a two-year table on the agent: 2014 and 2016 (a 2015 gap).

        2016 changes the *bottom marginal rate* (0.05 → 0.056), the threshold,
        the credit amount, and the deduction — so a correct lookup must replace
        all four, not just inflate thresholds.
        """
        table = {
            2014: {
                "pit_thresholds": np.array([100.0, 200.0, np.inf]),
                "pit_rates": np.array([0.05, 0.10, 0.15]),
                "pit_tax_credits": [self._credit(1000.0)],
                "pit_taxable_income_deductions": 50.0,
            },
            2016: {
                "pit_thresholds": np.array([110.0, 210.0, np.inf]),
                "pit_rates": np.array([0.056, 0.10, 0.15]),
                "pit_tax_credits": [self._credit(1100.0)],
                "pit_taxable_income_deductions": 55.0,
            },
        }
        cg.states["pit_schedule_by_year"] = table
        return table

    def test_selects_exact_year(self, test_central_government_pit_full):
        cg = test_central_government_pit_full
        self._seed_table(cg)

        cg.set_pit_for_year(2014)

        np.testing.assert_array_equal(
            cg.states["pit_thresholds"], np.array([100.0, 200.0, np.inf])
        )
        assert cg.states["pit_rates"][0] == 0.05
        assert cg.states["pit_tax_credits"][0]["amount"] == 1000.0
        assert cg.states["pit_taxable_income_deductions"] == 50.0

    def test_carries_marginal_rate_change(self, test_central_government_pit_full):
        """A statutory rate change (not expressible by CPI compounding) is
        carried by the lookup."""
        cg = test_central_government_pit_full
        self._seed_table(cg)

        cg.set_pit_for_year(2016)

        assert cg.states["pit_rates"][0] == pytest.approx(0.056)
        np.testing.assert_array_equal(
            cg.states["pit_thresholds"], np.array([110.0, 210.0, np.inf])
        )
        assert cg.states["pit_tax_credits"][0]["amount"] == 1100.0
        assert cg.states["pit_taxable_income_deductions"] == 55.0

    def test_gap_year_holds_most_recent(self, test_central_government_pit_full):
        """A year between table entries (2015) holds the most recent year at or
        before it (2014), not the next one."""
        cg = test_central_government_pit_full
        self._seed_table(cg)

        cg.set_pit_for_year(2015)

        assert cg.states["pit_rates"][0] == 0.05
        np.testing.assert_array_equal(
            cg.states["pit_thresholds"], np.array([100.0, 200.0, np.inf])
        )

    def test_beyond_table_raises(self, test_central_government_pit_full):
        """A year past the table's last entry raises rather than holding the
        last known schedule flat — the model does not project past the
        published data."""
        cg = test_central_government_pit_full
        self._seed_table(cg)
        before = cg.states["pit_thresholds"].copy()

        with pytest.raises(ValueError, match="exceeds the last available"):
            cg.set_pit_for_year(2050)

        # The agent's live state is untouched by the rejected lookup.
        np.testing.assert_array_equal(cg.states["pit_thresholds"], before)

    def test_before_table_holds_first_year(self, test_central_government_pit_full):
        """A year before the table's first entry holds the first schedule."""
        cg = test_central_government_pit_full
        self._seed_table(cg)

        cg.set_pit_for_year(2000)

        assert cg.states["pit_rates"][0] == 0.05
        np.testing.assert_array_equal(
            cg.states["pit_thresholds"], np.array([100.0, 200.0, np.inf])
        )

    def test_noop_without_table(self, test_central_government_pit):
        """No schedule table (single-year / frozen) ⇒ schedule untouched."""
        cg = test_central_government_pit
        assert "pit_schedule_by_year" not in cg.states
        original = cg.states["pit_thresholds"].copy()

        cg.set_pit_for_year(2025)

        np.testing.assert_array_equal(cg.states["pit_thresholds"], original)

    def test_noop_on_flat_government(self, test_central_government):
        """Flat-tax government: set_pit_for_year is a no-op (no PIT states)."""
        cg = test_central_government
        cg.set_pit_for_year(2025)
        assert "pit_thresholds" not in cg.states
