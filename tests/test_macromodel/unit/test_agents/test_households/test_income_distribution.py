"""Tests for the household→individual income distribution helpers.

``distribute_rental_income_to_individuals`` and
``distribute_financial_income_to_individuals`` feed the progressive-PIT taxable
pool (Pool A).  These tests pin the two invariants the pool relies on:

* **Adults only** — recipients are the household members aged >= 18 (matching
  ``pit_pools._household_context``); children receive nothing.
* **Conservation** — the distributed total equals the household total, for the
  adult path and for both conserving fallbacks (no age data / no adult member).

The methods only touch ``self.ts.current("n_households")``,
``self._adult_members`` and (rental) ``self.compute_gross_rental_income``, so a
duck-typed stub exercises them without the full agent fixture.
"""

import types

import numpy as np
import numpy.testing as npt

from macromodel.agents.households.households import Households


def _stub(n_households: int, gross_rental: np.ndarray | None = None):
    stub = types.SimpleNamespace()
    stub.ts = types.SimpleNamespace(
        current=lambda key: {"n_households": n_households}[key]
    )
    stub._adult_members = Households._adult_members
    if gross_rental is not None:
        stub.compute_gross_rental_income = lambda housing_data: gross_rental
    return stub


class TestDistributeFinancialIncome:
    def test_adults_only_and_conserved(self):
        """2 adults + 2 children, $1,000: $500 per adult, nothing to children,
        total conserved (the H2 bug distributed $2,000 here)."""
        ages = np.array([40.0, 38.0, 10.0, 8.0])
        corr = np.array([0, 0, 0, 0])
        out = Households.distribute_financial_income_to_individuals(
            _stub(1),
            household_financial_income=np.array([1000.0]),
            corr_households=corr,
            n_individuals=4,
            individuals_age=ages,
        )
        npt.assert_allclose(out, [500.0, 500.0, 0.0, 0.0])
        assert out.sum() == 1000.0

    def test_no_age_data_falls_back_to_all_members_conserved(self):
        corr = np.array([0, 0, 0, 0])
        out = Households.distribute_financial_income_to_individuals(
            _stub(1),
            household_financial_income=np.array([1000.0]),
            corr_households=corr,
            n_individuals=4,
            individuals_age=None,
        )
        npt.assert_allclose(out, [250.0] * 4)
        assert out.sum() == 1000.0

    def test_no_adult_household_falls_back_conserved(self):
        ages = np.array([16.0, 12.0])
        corr = np.array([0, 0])
        out = Households.distribute_financial_income_to_individuals(
            _stub(1),
            household_financial_income=np.array([300.0]),
            corr_households=corr,
            n_individuals=2,
            individuals_age=ages,
        )
        npt.assert_allclose(out, [150.0, 150.0])

    def test_multiple_households_conserved(self):
        # hh0: 1 adult + 1 child; hh1: 2 adults; hh2: no income.
        ages = np.array([30.0, 5.0, 45.0, 44.0, 70.0])
        corr = np.array([0, 0, 1, 1, 2])
        hh_income = np.array([400.0, 600.0, 0.0])
        out = Households.distribute_financial_income_to_individuals(
            _stub(3),
            household_financial_income=hh_income,
            corr_households=corr,
            n_individuals=5,
            individuals_age=ages,
        )
        npt.assert_allclose(out, [400.0, 0.0, 300.0, 300.0, 0.0])
        assert out.sum() == hh_income.sum()


class TestDistributeRentalIncome:
    def test_children_excluded_split_among_adults(self):
        """2 adults (distinct earnings) + 2 children, split 0.7: the higher
        earner gets 70%, the other adult 30%, children nothing (the M1 bug
        gave each child a remainder share)."""
        ages = np.array([40.0, 38.0, 10.0, 8.0])
        wages = np.array([50000.0, 30000.0, 0.0, 0.0])
        corr = np.array([0, 0, 0, 0])
        out = Households.distribute_rental_income_to_individuals(
            _stub(1, gross_rental=np.array([1000.0])),
            housing_data=None,
            corr_households=corr,
            individual_employee_income=wages,
            couple_rental_income_split=0.7,
            individuals_age=ages,
        )
        npt.assert_allclose(out, [700.0, 300.0, 0.0, 0.0])
        assert out.sum() == 1000.0

    def test_single_adult_selected_by_age_not_position(self):
        """The sole adult receives 100% even when a child precedes them in the
        individuals array (the pre-fix code paid whoever sat at index 0)."""
        ages = np.array([9.0, 35.0])  # child first, adult second
        wages = np.array([0.0, 20000.0])
        corr = np.array([0, 0])
        out = Households.distribute_rental_income_to_individuals(
            _stub(1, gross_rental=np.array([1000.0])),
            housing_data=None,
            corr_households=corr,
            individual_employee_income=wages,
            couple_rental_income_split=0.7,
            individuals_age=ages,
        )
        npt.assert_allclose(out, [0.0, 1000.0])

    def test_three_adults_remainder_split_equally(self):
        ages = np.array([50.0, 48.0, 22.0])
        wages = np.array([80000.0, 40000.0, 20000.0])
        corr = np.array([0, 0, 0])
        out = Households.distribute_rental_income_to_individuals(
            _stub(1, gross_rental=np.array([1000.0])),
            housing_data=None,
            corr_households=corr,
            individual_employee_income=wages,
            couple_rental_income_split=0.7,
            individuals_age=ages,
        )
        npt.assert_allclose(out, [700.0, 150.0, 150.0])
        assert out.sum() == 1000.0

    def test_no_age_data_falls_back_to_all_members(self):
        wages = np.array([50000.0, 30000.0, 0.0])
        corr = np.array([0, 0, 0])
        out = Households.distribute_rental_income_to_individuals(
            _stub(1, gross_rental=np.array([1000.0])),
            housing_data=None,
            corr_households=corr,
            individual_employee_income=wages,
            couple_rental_income_split=0.7,
            individuals_age=None,
        )
        # Highest earner 70%; the two remaining members split 30%.
        npt.assert_allclose(out, [700.0, 150.0, 150.0])
        assert out.sum() == 1000.0
