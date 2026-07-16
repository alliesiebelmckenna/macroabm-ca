import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus


class TestIndividuals:
    def test__individuals_states(self, test_individuals):
        assert test_individuals is not None
        for state in [
            "Gender",
            "Age",
            "Education",
            "Activity Status",
            "Employment Industry",
            "Income",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Corresponding Household ID",
            "Corresponding Firm ID",
        ]:
            assert state in test_individuals.states.keys()

    def test__individuals_ts(self, test_individuals):
        for ts_key in [
            "n_individuals",
            "employee_income",
            "income_from_unemployment_benefits",
            "income",
            "labour_inputs",
            "reservation_wages",
        ]:
            assert ts_key in test_individuals.ts.get_keys()

    def test_compute_gross_firm_dividend(self, test_individuals):
        """D_i = payout × (1 − tau_firm) × max(0, profits) for firm investors,
        zero otherwise; a loss-making firm pays no dividend."""
        ind = test_individuals
        ind.states["Activity Status"] = np.array(
            [
                ActivityStatus.EMPLOYED,
                ActivityStatus.FIRM_INVESTOR,
                ActivityStatus.FIRM_INVESTOR,
            ]
        )
        ind.states["Corresponding Invested Firm"] = np.array([-1, 0, 1])
        ind.states["Dividend Payout Ratio"] = 0.5

        firm_profits = np.array([1000.0, -200.0])  # firm 1 is loss-making
        dividend = ind.compute_gross_firm_dividend(firm_profits=firm_profits, tau_firm=0.20)

        # ind0 employed -> 0; ind1 invests firm0 -> 0.5×0.8×1000 = 400;
        # ind2 invests the loss-making firm1 -> max(0, -200) = 0.
        np.testing.assert_allclose(dividend, [0.0, 400.0, 0.0])

    def test_compute_gross_firm_dividend_has_no_income_tax_haircut(self, test_individuals):
        """Unlike compute_income, D_i carries NO (1 − income_taxes) haircut —
        it is the real pre-personal-tax dividend the household receives."""
        ind = test_individuals
        ind.states["Activity Status"] = np.array([ActivityStatus.FIRM_INVESTOR])
        ind.states["Corresponding Invested Firm"] = np.array([0])
        ind.states["Dividend Payout Ratio"] = 1.0
        dividend = ind.compute_gross_firm_dividend(firm_profits=np.array([500.0]), tau_firm=0.10)
        # 1.0 × (1 − 0.10) × 500 = 450 — no income-tax factor applied.
        np.testing.assert_allclose(dividend, [450.0])

    def test_compute_gross_bank_dividend(self, test_individuals):
        """D_i = payout × (1 − tau_firm) × max(0, profits) for bank investors,
        zero otherwise; a loss-making bank pays no dividend."""
        ind = test_individuals
        ind.states["Activity Status"] = np.array(
            [
                ActivityStatus.EMPLOYED,
                ActivityStatus.BANK_INVESTOR,
                ActivityStatus.BANK_INVESTOR,
            ]
        )
        ind.states["Corresponding Invested Bank"] = np.array([-1, 0, 1])
        ind.states["Dividend Payout Ratio"] = 0.5

        bank_profits = np.array([1000.0, -200.0])  # bank 1 is loss-making
        dividend = ind.compute_gross_bank_dividend(bank_profits=bank_profits, tau_firm=0.20)

        # ind0 employed -> 0; ind1 invests bank0 -> 0.5×0.8×1000 = 400;
        # ind2 invests the loss-making bank1 -> max(0, -200) = 0.
        np.testing.assert_allclose(dividend, [0.0, 400.0, 0.0])

    def test_compute_gross_bank_dividend_has_no_income_tax_haircut(self, test_individuals):
        """Unlike compute_income, D_i carries NO (1 − income_taxes) haircut —
        it is the real pre-personal-tax dividend the household receives."""
        ind = test_individuals
        ind.states["Activity Status"] = np.array([ActivityStatus.BANK_INVESTOR])
        ind.states["Corresponding Invested Bank"] = np.array([0])
        ind.states["Dividend Payout Ratio"] = 1.0
        dividend = ind.compute_gross_bank_dividend(bank_profits=np.array([500.0]), tau_firm=0.10)
        # 1.0 × (1 − 0.10) × 500 = 450 — no income-tax factor applied.
        np.testing.assert_allclose(dividend, [450.0])
