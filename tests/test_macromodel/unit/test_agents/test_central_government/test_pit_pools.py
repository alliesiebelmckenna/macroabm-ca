"""Unit tests for the PIT pool builders (the processing phase).

These exercise ``pit_pools`` in isolation: the pure functions that assemble
Pool A (taxable income) and Pool B (credit base) before the government applies
tax policy.
"""

import numpy as np
import pytest

from macromodel.agents.central_government.pit_pools import (
    PitContext,
    build_credit_base_pool,
    build_dividend_tax_items,
    build_taxable_income_pool,
)

# 2014 BC firm-dividend integration constants (s = 0.90 provisional split).
_S = 0.90
_ELIG_GROSSUP = 0.38
_NONELIG_GROSSUP = 0.18
_ELIG_DTC = 0.10
_NONELIG_DTC = 0.0259


class TestTaxableIncomePool:
    def test_streams_stack_additively(self):
        """Each income stream contributes to the same pool (so the
        progressive brackets later apply once to the combined total)."""
        ctx = PitContext(
            employee_income=np.array([100.0, 200.0]),
            employee_si_rate=0.0,
            rental_income=np.array([10.0, 20.0]),
            financial_income=np.array([1.0, 2.0]),
        )
        pool = build_taxable_income_pool(ctx)
        np.testing.assert_allclose(pool, [111.0, 222.0])


class TestTargetedCreditsRequireContext:
    """Targeted credits must return zero when their required context is
    missing — never silently become a universal credit."""

    def _bare_ctx(self):
        # Only income; no age, household, or child context.
        return PitContext(
            employee_income=np.array([40000.0, 40000.0]),
            employee_si_rate=0.0,
        )

    def test_age_amount_without_age_context_is_zero(self):
        ctx = self._bare_ctx()
        taxable = build_taxable_income_pool(ctx)
        credit_defs = [
            {
                "credit": "Age Amount",
                "amount": 4426.0,
                "age_min": 65,
                "clawback": 32943.0,
                "top": 62450.0,
            }
        ]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_array_equal(base, [0.0, 0.0])


class TestCreditBasePool:
    def test_universal_credit_applies_to_all(self):
        taxable = np.array([100.0, 200.0])
        ctx = PitContext(employee_income=taxable, employee_si_rate=0.0)
        credit_defs = [{"credit": "Personal Amount", "amount": 9869.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_allclose(base, [9869.0, 9869.0])

    def test_equivalent_to_spouse_amount_single_parent_only(self):
        """The eligible-dependant credit applies only to single parents.

        ind0/ind1 are a parent and their minor child; ind2 is a lone adult on
        the same income, so the credit turns on household type rather than on
        income or age alone.
        """
        from macromodel.agents.households.household_properties import HouseholdType

        taxable = np.array([30000.0, 0.0, 30000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_age=np.array([45, 10, 30]),
            individuals_corr_households=np.array([0, 0, 1]),
            households_type=np.array(
                [
                    HouseholdType.SINGLE_PARENT_WITH_CHILDREN,  # hh0 = single parent
                    HouseholdType.ONE_ADULT_YOUNGER_THAN_64,  # hh1 = single adult
                ],
                dtype=object,
            ),
        )
        credit_defs = [{"credit": "Equivalent To Spouse Amount", "amount": 12000.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        # Only the single parent (ind0) is eligible.
        np.testing.assert_allclose(base, [12000.0, 0.0, 0.0])

    def test_age_amount_only_for_eligible_age_with_clawback(self):
        """Age Amount applies only to age >= age_min, even when a clawback
        is configured. Two filers with identical income but different ages
        must get different credits (under-age gets nothing)."""
        taxable = np.array([40000.0, 40000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_age=np.array([40, 70]),
        )
        credit_defs = [
            {
                "credit": "Age Amount",
                "amount": 4426.0,
                "age_min": 65,
                "clawback": 32943.0,
                "top": 62450.0,
            }
        ]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        expected_70 = 4426.0 - (40000.0 - 32943.0) * 4426.0 / (62450.0 - 32943.0)
        np.testing.assert_allclose(base, [0.0, expected_70], rtol=1e-6)


class TestSpousalAmountGrouping:
    """Spousal Amount exercises the per-household spouse-income grouping.

    Covers the case where two adults of one household are *not* adjacent in
    the individual array, which the sort-based grouping must still pair."""

    def _ctx(self, corr, employee_income):
        from macromodel.agents.households.household_properties import HouseholdType

        return PitContext(
            employee_income=np.asarray(employee_income, dtype=float),
            employee_si_rate=0.0,
            individuals_corr_households=np.asarray(corr),
            households_type=np.array(
                [
                    HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,  # household 0 = couple
                    HouseholdType.ONE_ADULT_YOUNGER_THAN_64,  # household 1 = single
                ],
                dtype=object,
            ),
        )

    def test_spousal_pairs_adults_ignoring_children(self):
        """A couple-with-children household has 4 individuals (2 adults + 2
        children) in the array, but only the two adults are paired — the
        children must not block the pairing or receive the credit."""
        from macromodel.agents.households.household_properties import HouseholdType

        # ind0, ind1 = adults; ind2, ind3 = children; all in household 0.
        ctx = PitContext(
            employee_income=np.array([50000.0, 5000.0, 0.0, 0.0]),
            employee_si_rate=0.0,
            individuals_age=np.array([40, 38, 10, 8]),
            individuals_corr_households=np.array([0, 0, 0, 0]),
            households_type=np.array([HouseholdType.TWO_ADULTS_WITH_TWO_CHILDREN], dtype=object),
        )
        taxable = build_taxable_income_pool(ctx)  # [50000, 5000, 0, 0]

        credit_defs = [{"credit": "Spousal Amount", "amount": 12000.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)

        # Adults are paired: ind0 spouse=5000 → 12000-5000=7000;
        # ind1 spouse=50000 → 0. Children get no credit.
        np.testing.assert_allclose(base, [7000.0, 0.0, 0.0, 0.0])


class TestDividendTaxItems:
    """The firm-dividend gross-up + dividend tax credit (tax-only items)."""

    def test_grossup_and_dtc_match_hand_calc(self):
        # D = 100, s = 0.9 → eligible 10, non-eligible 90.
        #   grossed eligible     = 10 × 1.38 = 13.8
        #   grossed non-eligible = 90 × 1.18 = 106.2
        #   grossed-up dividend  = 120.0  (= 1.20 × D)
        #   DTC = 0.10×13.8 + 0.0259×106.2 = 1.38 + 2.75058 = 4.13058
        grossed, dtc = build_dividend_tax_items(
            dividend_income=np.array([100.0]),
            small_business_share=_S,
            eligible_gross_up=_ELIG_GROSSUP,
            non_eligible_gross_up=_NONELIG_GROSSUP,
            eligible_dtc_rate=_ELIG_DTC,
            non_eligible_dtc_rate=_NONELIG_DTC,
        )
        np.testing.assert_allclose(grossed, [120.0])
        np.testing.assert_allclose(dtc, [4.13058])


class TestUnmappedCreditFailClosed:
    """A credit kind with no runtime branch must contribute zero — never fall
    through to a universal grant (defence in depth behind the builder's
    expressibility filter)."""

    def _ctx(self):
        return PitContext(
            employee_income=np.array([100.0, 200.0]),
            employee_si_rate=0.0,
            individuals_age=np.array([70.0, 40.0]),
        )

    def test_unknown_kind_contributes_zero(self):
        credit_defs = [{"credit": "Disability Amount", "amount": 8000.0}]
        pool = build_credit_base_pool(credit_defs, np.array([50000.0, 20000.0]), self._ctx())
        np.testing.assert_array_equal(pool, [0.0, 0.0])


# Published BC 2014 amounts, from non_refundable_tax_credits.csv.
_SPOUSAL_2014 = {
    "credit": "Spousal Amount",
    "amount": 8450.0,
    "clawback": 845.0,
    "top": 9295.0,
}
_ETS_2014 = {
    "credit": "Equivalent To Spouse Amount",
    "amount": 8450.0,
    "top": 9295.0,
}


def _household_ctx(incomes, ages, household_type):
    """One household holding every listed individual."""
    incomes = np.asarray(incomes, dtype=float)
    return PitContext(
        employee_income=incomes,
        employee_si_rate=0.0,
        individuals_age=np.asarray(ages),
        individuals_corr_households=np.zeros(len(incomes), dtype=int),
        households_type=np.array([household_type], dtype=object),
    )


class TestSpousalAmountHonoursItsSchedule:
    """The Spousal Amount tapers against the spouse's income above the
    published exemption, not from the spouse's first dollar."""

    def test_spouse_income_below_the_exemption_leaves_the_credit_whole(self):
        from macromodel.agents.households.household_properties import HouseholdType

        # Spouse earns 500, inside the published 845 exemption.
        ctx = _household_ctx(
            [40000.0, 500.0],
            [45, 43],
            HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,
        )
        taxable = build_taxable_income_pool(ctx)
        base = build_credit_base_pool([_SPOUSAL_2014], taxable, ctx)
        # The claimant keeps the full base; only the spouse's income above
        # 845 reduces it, and there is none.
        assert base[0] == pytest.approx(8450.0)


class TestCoupleWithAnAdditionalAdult:
    """A couple-typed household is paired on its type, not on containing
    exactly two adults — a resident adult child must not void the credit."""

    def test_third_adult_does_not_void_the_spousal_amount(self):
        from macromodel.agents.households.household_properties import HouseholdType

        # Two spouses plus an 18-year-old still at home.
        ctx = _household_ctx(
            [0.0, 60000.0, 0.0],
            [45, 43, 18],
            HouseholdType.TWO_ADULTS_WITH_ONE_CHILD,
        )
        taxable = build_taxable_income_pool(ctx)
        base = build_credit_base_pool([_SPOUSAL_2014], taxable, ctx)
        # The earner claims against a spouse with no income.
        assert base.sum() > 0.0


class TestEquivalentToSpouseEligibility:
    """One claim per single-parent household supporting a minor child.

    The exception for a dependant aged 18 or over with an infirmity is not
    expressed: the model carries no infirmity signal, so a household whose
    children have all reached 18 is treated as ineligible.
    """

    def test_parent_of_a_minor_claims_once(self):
        from macromodel.agents.households.household_properties import HouseholdType

        ctx = _household_ctx(
            [40000.0, 0.0],
            [45, 10],
            HouseholdType.SINGLE_PARENT_WITH_CHILDREN,
        )
        taxable = build_taxable_income_pool(ctx)
        base = build_credit_base_pool([_ETS_2014], taxable, ctx)
        np.testing.assert_allclose(base, [8450.0, 0.0])

    def test_dependant_income_above_the_exemption_reduces_the_claim(self):
        from macromodel.agents.households.household_properties import HouseholdType

        # BC 2014 publishes ETS as amount 8450 / top 9295, i.e. an implied 845
        # exemption. A minor earning 2000 reduces the parent's claim by 1155.
        ctx = _household_ctx(
            [40000.0, 2000.0],
            [45, 16],
            HouseholdType.SINGLE_PARENT_WITH_CHILDREN,
        )
        taxable = build_taxable_income_pool(ctx)
        base = build_credit_base_pool([_ETS_2014], taxable, ctx)
        np.testing.assert_allclose(base, [8450.0 - (2000.0 - 845.0), 0.0])
