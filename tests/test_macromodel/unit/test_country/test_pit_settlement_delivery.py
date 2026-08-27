"""The year-end filing settlement has to reach the households it settles for.

The settlement was computed per individual, booked into `taxes_income`, and reported on its
own series -- and then reached nobody. A household the filing said had underpaid kept its
money, and one owed a refund never received it, so the revenue line was funded by no one.

It joins the REALISED income only: a household does not know its filing outcome when it plans
consumption, so the settlement lands on saving and wealth rather than on a plan already made.
"""

import numpy as np
import pytest


def _corr(country) -> np.ndarray:
    return np.asarray(country.individuals.states["Corresponding Household ID"]).astype(int)


def _set_settlement(country, per_individual) -> None:
    country.central_government.states["pit_settlement_per_ind"] = per_individual


class TestPitSettlementDelivery:
    def test_a_household_that_underpaid_loses_income(self, test_country):
        corr = _corr(test_country)
        settlement = np.zeros(len(corr))
        settlement[0] = 250.0  # positive: this individual owes more at the filing
        _set_settlement(test_country, settlement)

        delivered = test_country._pit_settlement_per_household()

        assert delivered[corr[0]] == pytest.approx(-250.0)
        assert delivered.sum() == pytest.approx(-250.0)

    def test_a_household_owed_a_refund_gains_income(self, test_country):
        corr = _corr(test_country)
        settlement = np.zeros(len(corr))
        settlement[0] = -75.0  # negative: overpaid, refunded at the filing
        _set_settlement(test_country, settlement)

        delivered = test_country._pit_settlement_per_household()

        assert delivered[corr[0]] == pytest.approx(75.0)

    def test_individuals_are_aggregated_to_their_own_household(self, test_country):
        corr = _corr(test_country)
        settlement = np.arange(len(corr), dtype=float)
        _set_settlement(test_country, settlement)

        delivered = test_country._pit_settlement_per_household()

        expected = -np.bincount(corr, weights=settlement, minlength=len(delivered))[: len(delivered)]
        assert delivered == pytest.approx(expected)

    def test_what_the_government_books_is_what_households_are_debited(self, test_country):
        # The identity the delivery exists to make true.
        corr = _corr(test_country)
        rng = np.random.default_rng(0)
        settlement = rng.normal(scale=100.0, size=len(corr))
        _set_settlement(test_country, settlement)

        booked_as_revenue = float(np.sum(settlement))
        taken_from_households = -float(test_country._pit_settlement_per_household().sum())

        assert taken_from_households == pytest.approx(booked_as_revenue)

    def test_no_filing_this_period_delivers_nothing(self, test_country):
        _set_settlement(test_country, 0.0)
        assert test_country._pit_settlement_per_household().sum() == pytest.approx(0.0)

        test_country.central_government.states.pop("pit_settlement_per_ind", None)
        assert test_country._pit_settlement_per_household().sum() == pytest.approx(0.0)

    def test_realised_income_carries_the_settlement_and_expected_income_does_not(self, test_country):
        households = test_country.households
        n_hh = int(households.ts.current("n_households"))
        debit = np.full(n_hh, -40.0)

        before_realised = households.compute_income().sum()
        before_expected = households.compute_expected_income().sum()
        households.ts.income_pit_settlement.append(debit)
        after_realised = households.compute_income().sum()
        after_expected = households.compute_expected_income().sum()

        assert after_realised - before_realised == pytest.approx(debit.sum())
        assert after_expected == pytest.approx(before_expected)
