"""Unit tests for the PIT pool builders (the processing phase).

These exercise :mod:`macromodel.agents.central_government.pit_pools` in
isolation — the pure functions that assemble Pool A (taxable income) and
Pool B (credit base) before the government applies tax policy.  They also
demonstrate the extensibility goal: a new income stream is just another
term in Pool A; a new credit is just another contributor to Pool B.
"""

import numpy as np

from macromodel.agents.central_government.pit_pools import (
    PitContext,
    build_credit_base_pool,
    build_taxable_income_pool,
)


class TestTaxableIncomePool:
    def test_employee_income_net_of_si(self):
        ctx = PitContext(
            employee_income=np.array([100.0, 200.0]),
            employee_si_rate=0.10,
        )
        pool = build_taxable_income_pool(ctx)
        np.testing.assert_allclose(pool, [90.0, 180.0])

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

    def test_missing_streams_are_skipped(self):
        ctx = PitContext(
            employee_income=np.array([50.0]),
            employee_si_rate=0.0,
            rental_income=None,
            financial_income=np.array([5.0]),
        )
        pool = build_taxable_income_pool(ctx)
        np.testing.assert_allclose(pool, [55.0])


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
        credit_defs = [{
            "kind": "Age Amount", "amount": 4426.0, "age_min": 65,
            "clawback_start": 32943.0, "clawback_cap": 62450.0,
        }]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_array_equal(base, [0.0, 0.0])

    def test_spousal_amount_without_household_context_is_zero(self):
        ctx = self._bare_ctx()
        taxable = build_taxable_income_pool(ctx)
        credit_defs = [{"kind": "Spousal Amount", "amount": 12000.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_array_equal(base, [0.0, 0.0])

    def test_equivalent_to_spouse_without_household_context_is_zero(self):
        ctx = self._bare_ctx()
        taxable = build_taxable_income_pool(ctx)
        credit_defs = [{"kind": "Equivalent To Spouse Amount", "amount": 12000.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_array_equal(base, [0.0, 0.0])

    def test_child_credits_without_child_context_are_zero(self):
        ctx = self._bare_ctx()
        taxable = build_taxable_income_pool(ctx)
        for kind in ("Child Amount Under 18", "Child Amount Under 6"):
            base = build_credit_base_pool(
                [{"kind": kind, "amount": 100.0}], taxable, ctx
            )
            np.testing.assert_array_equal(base, [0.0, 0.0])

    def test_universal_credit_still_applies_without_context(self):
        """A genuinely universal credit (Personal Amount) is unaffected."""
        ctx = self._bare_ctx()
        taxable = build_taxable_income_pool(ctx)
        base = build_credit_base_pool(
            [{"kind": "Personal Amount", "amount": 9869.0}], taxable, ctx
        )
        np.testing.assert_allclose(base, [9869.0, 9869.0])


class TestCreditBasePool:
    def test_no_credits_returns_zeros(self):
        taxable = np.array([100.0, 200.0])
        ctx = PitContext(employee_income=taxable, employee_si_rate=0.0)
        base = build_credit_base_pool(None, taxable, ctx)
        np.testing.assert_array_equal(base, [0.0, 0.0])

    def test_universal_credit_applies_to_all(self):
        taxable = np.array([100.0, 200.0])
        ctx = PitContext(employee_income=taxable, employee_si_rate=0.0)
        credit_defs = [{"kind": "Personal Amount", "amount": 9869.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_allclose(base, [9869.0, 9869.0])

    def test_age_credit_gated_by_age(self):
        taxable = np.array([30000.0, 30000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_age=np.array([40, 70]),
        )
        credit_defs = [{"kind": "Age Amount", "amount": 5000.0, "age_min": 65}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        # Only the 70-year-old is eligible (income below clawback range).
        np.testing.assert_allclose(base, [0.0, 5000.0])

    def test_multiple_credits_sum(self):
        taxable = np.array([30000.0, 30000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_age=np.array([70, 40]),
        )
        credit_defs = [
            {"kind": "Personal Amount", "amount": 1000.0},
            {"kind": "Age Amount", "amount": 500.0, "age_min": 65},
        ]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        # Person 0: personal + age; person 1: personal only.
        np.testing.assert_allclose(base, [1500.0, 1000.0])

    def test_child_credit_scales_per_child(self):
        taxable = np.array([40000.0, 40000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            children_under_18_per_ind=np.array([0, 3]),
        )
        credit_defs = [{"kind": "Child Amount Under 18", "amount": 100.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_allclose(base, [0.0, 300.0])

    def test_child_amount_under_6_scales_per_child(self):
        taxable = np.array([40000.0, 40000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            children_under_6_per_ind=np.array([0, 2]),
        )
        credit_defs = [{"kind": "Child Amount Under 6", "amount": 200.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_allclose(base, [0.0, 400.0])

    def test_equivalent_to_spouse_amount_single_parent_only(self):
        """The eligible-dependant credit applies only to single parents."""
        from macromodel.agents.households.household_properties import HouseholdType

        taxable = np.array([30000.0, 30000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_corr_households=np.array([0, 1]),
            households_type=np.array(
                [
                    HouseholdType.SINGLE_PARENT_WITH_CHILDREN,  # hh0 = single parent
                    HouseholdType.ONE_ADULT_YOUNGER_THAN_64,    # hh1 = single adult
                ],
                dtype=object,
            ),
            households_n_adults=np.array([1, 1]),
        )
        credit_defs = [{"kind": "Equivalent To Spouse Amount", "amount": 12000.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        # Only the single parent (ind0) is eligible.
        np.testing.assert_allclose(base, [12000.0, 0.0])

    def test_age_amount_phaseout(self):
        """Age Amount decreases once own income exceeds clawback_start."""
        taxable = np.array([40000.0])  # above the 32943 start
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_age=np.array([70]),
        )
        credit_defs = [{
            "kind": "Age Amount", "amount": 4426.0,
            "age_min": 65, "clawback_start": 32943.0, "clawback_cap": 62450.0,
        }]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        # 4426 - (40000 - 32943) * 4426/(62450 - 32943)
        expected = 4426.0 - (40000.0 - 32943.0) * 4426.0 / (62450.0 - 32943.0)
        np.testing.assert_allclose(base, [expected], rtol=1e-6)

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
        credit_defs = [{
            "kind": "Age Amount", "amount": 4426.0,
            "age_min": 65, "clawback_start": 32943.0, "clawback_cap": 62450.0,
        }]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        expected_70 = 4426.0 - (40000.0 - 32943.0) * 4426.0 / (62450.0 - 32943.0)
        np.testing.assert_allclose(base, [0.0, expected_70], rtol=1e-6)

    def test_age_amount_fully_clawed_back_above_cap(self):
        """Above clawback_cap the Age Amount is fully eliminated."""
        taxable = np.array([70000.0])  # above the 62450 cap
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_age=np.array([70]),
        )
        credit_defs = [{
            "kind": "Age Amount", "amount": 4426.0,
            "age_min": 65, "clawback_start": 32943.0, "clawback_cap": 62450.0,
        }]
        base = build_credit_base_pool(credit_defs, taxable, ctx)
        np.testing.assert_allclose(base, [0.0])


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
                    HouseholdType.ONE_ADULT_YOUNGER_THAN_64,   # household 1 = single
                ],
                dtype=object,
            ),
            households_n_adults=np.array([2, 1]),
        )

    def test_spousal_credit_with_noncontiguous_couple(self):
        # ind0 & ind2 form couple household 0; ind1 is single household 1.
        ctx = self._ctx(corr=[0, 1, 0], employee_income=[50000, 30000, 10000])
        taxable = build_taxable_income_pool(ctx)  # [50000, 30000, 10000]

        credit_defs = [{"kind": "Spousal Amount", "amount": 12000.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)

        # ind0: max(0, 12000 - spouse(=10000)) = 2000
        # ind2: max(0, 12000 - spouse(=50000)) = 0
        # ind1: single → spouse income inf → 0
        np.testing.assert_allclose(base, [2000.0, 0.0, 0.0])

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
            households_type=np.array(
                [HouseholdType.TWO_ADULTS_WITH_TWO_CHILDREN], dtype=object
            ),
            households_n_adults=np.array([2]),
        )
        taxable = build_taxable_income_pool(ctx)  # [50000, 5000, 0, 0]

        credit_defs = [{"kind": "Spousal Amount", "amount": 12000.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)

        # Adults are paired: ind0 spouse=5000 → 12000-5000=7000;
        # ind1 spouse=50000 → 0. Children get no credit.
        np.testing.assert_allclose(base, [7000.0, 0.0, 0.0, 0.0])

    def test_spousal_amount_clawback(self):
        """Spousal Amount falls to zero once the spouse's income exceeds
        the base amount, and is partial when below it."""
        # Couple household 0 with a low earner (ind0) and high earner (ind1).
        ctx = self._ctx(corr=[0, 0], employee_income=[5000, 60000])
        taxable = build_taxable_income_pool(ctx)  # [5000, 60000]

        credit_defs = [{"kind": "Spousal Amount", "amount": 12000.0}]
        base = build_credit_base_pool(credit_defs, taxable, ctx)

        # ind0's spouse earns 60000 (> 12000 base) -> credit clawed to 0.
        # ind1's spouse earns  5000 (< 12000 base) -> 12000 - 5000 = 7000.
        np.testing.assert_allclose(base, [0.0, 7000.0])

    def test_spouse_income_matches_bruteforce(self):
        """Vectorised grouping equals the original per-household scan in the
        no-ages fallback (all members treated as adults).  Child-aware
        pairing is covered by test_spousal_pairs_adults_ignoring_children."""
        from macromodel.agents.central_government.pit_pools import _household_context

        rng = np.random.default_rng(0)
        # 10 individuals scattered across households 0..3 (only 2-adult
        # households should be paired).  No ages → all members count as
        # adults, so household types here are all childless couples.
        corr = np.array([0, 3, 0, 1, 2, 3, 1, 2, 1, 0])
        taxable = rng.uniform(1000, 90000, size=len(corr))

        from macromodel.agents.households.household_properties import HouseholdType

        hh_type = np.array(
            [
                HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,   # hh0 (3 members → not paired)
                HouseholdType.THREE_OR_MORE_ADULTS,         # hh1 (non-couple)
                HouseholdType.TWO_ADULTS_ONE_AT_LEAST_65,   # hh2 (2 members → paired)
                HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,   # hh3 (2 members → paired)
            ],
            dtype=object,
        )
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_corr_households=corr,
            households_type=hh_type,
            households_n_adults=np.array([3, 3, 2, 2]),
        )

        spouse = _household_context(len(corr), taxable, ctx).spouse_income

        # Brute-force reference (the original O(H*N) scan).
        couple_types = {
            HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,
            HouseholdType.TWO_ADULTS_ONE_AT_LEAST_65,
            HouseholdType.TWO_ADULTS_WITH_ONE_CHILD,
            HouseholdType.TWO_ADULTS_WITH_TWO_CHILDREN,
            HouseholdType.TWO_ADULTS_WITH_AT_LEAST_THREE_CHILDREN,
        }
        expected = np.full(len(corr), np.inf)
        for hh_id in np.unique(corr):
            members = np.where(corr == hh_id)[0]
            if len(members) == 2 and hh_type[hh_id] in couple_types:
                expected[members[0]] = taxable[members[1]]
                expected[members[1]] = taxable[members[0]]

        np.testing.assert_array_equal(spouse, expected)


class TestPoolsFeedPolicyOnce:
    """Document the progressivity contract: streams must pool *before* the
    brackets, never be taxed separately."""

    def test_pooling_differs_from_separate_taxation(self):
        from macro_data.readers.taxation.personal_income_tax.pit_schedule import (
            compute_progressive_tax,
        )

        thresholds = np.array([1000.0, np.inf])
        rates = np.array([0.10, 0.30])

        employee = np.array([800.0])
        rental = np.array([800.0])
        ctx = PitContext(
            employee_income=employee,
            employee_si_rate=0.0,
            rental_income=rental,
        )
        pooled = build_taxable_income_pool(ctx)  # [1600]

        tax_pooled = compute_progressive_tax(pooled, thresholds, rates).sum()
        tax_separate = (
            compute_progressive_tax(employee, thresholds, rates).sum()
            + compute_progressive_tax(rental, thresholds, rates).sum()
        )
        # Pooling stacks the second stream into the higher bracket, so it
        # is taxed more than taxing each stream from the bottom bracket.
        assert tax_pooled > tax_separate
