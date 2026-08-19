"""Refundable tax credits: valuation at the filing, and the two delivery paths.

A refundable credit differs from the non-refundable ones already in the model in
three ways, and each is asserted here rather than assumed: it is paid whether or
not tax is owed, it is a HOUSEHOLD entitlement paid to one claimant, and its two
delivery paths refer to different years.
"""

import numpy as np
import pytest

from macromodel.agents.central_government.pit_pools import (
    PitContext,
    build_refundable_credit_pools,
)
from macromodel.agents.households.household_properties import HouseholdType

# Two households. hh0: two adults and a child, RENTING. hh1: a single parent and
# two children, OWNING. Between them they exercise every eligibility branch.
AGES = np.array([40.0, 38.0, 8.0, 35.0, 10.0, 6.0])
CORR = np.array([0, 0, 0, 1, 1, 1])
HH_TYPE = np.array(
    [
        HouseholdType.TWO_ADULTS_WITH_ONE_CHILD,
        HouseholdType.SINGLE_PARENT_WITH_CHILDREN,
    ],
    dtype=object,
)
RENTS, OWNS = 3, 1  # HFCS tenure codes
SOCIAL_HOUSING = -1

CLIMATE = "Climate Action Tax Credit"
RENTER = "Renter's Tax Credit"


def _ctx(tenure=(RENTS, OWNS), ages=AGES, corr=CORR, hh_type=HH_TYPE):
    return PitContext(
        employee_income=np.zeros(len(ages)),
        employee_si_rate=0.0,
        individuals_age=ages,
        individuals_corr_households=corr,
        households_type=hh_type,
        households_tenure=np.array(tenure),
    )


def _defs(**over):
    """The 2023 published components, overridable per test."""
    base = [
        dict(
            credit_name=CLIMATE,
            credit="Eligible Individual Amount",
            delivery="instalments",
            amount=504.0,
            eligibility_age_min=19,
            clawback=41071.0,
            clawback_rate=0.02,
        ),
        dict(
            credit_name=CLIMATE,
            credit="Spousal Amount",
            delivery="instalments",
            amount=252.0,
            clawback=57288.0,
            clawback_rate=0.02,
        ),
        dict(
            credit_name=CLIMATE,
            credit="Equivalent To Spouse Amount",
            delivery="instalments",
            amount=252.0,
            clawback=57288.0,
            clawback_rate=0.02,
        ),
        dict(
            credit_name=CLIMATE,
            credit="Dependant Amount",
            delivery="instalments",
            amount=126.0,
            clawback=57288.0,
            clawback_rate=0.02,
        ),
        dict(
            credit_name=RENTER,
            credit="Renter's Amount",
            delivery="settlement",
            amount=400.0,
            clawback=60000.0,
            clawback_rate=0.02,
        ),
    ]
    for d in base:
        d.update(over.get(d["credit"], {}))
    return base


POOR = np.zeros(6)  # below every threshold, so nothing tapers


class TestOneClaimantPerHousehold:
    """The credit is a FAMILY entitlement paid to one person, not a per-head one."""

    def test_a_couple_draws_the_individual_amount_once_not_twice(self):
        # The failure this guards: granting per adult doubles a couple's credit,
        # and granting the spousal amount per member pays the child too.
        _settle, inst = build_refundable_credit_pools(_defs(), POOR, _ctx())
        assert inst[0] == pytest.approx(504.0 + 252.0 + 126.0)
        assert inst[1] == 0.0  # the spouse
        assert inst[2] == 0.0  # the child


class TestDependantCount:
    def test_the_para_d_exclusion_drops_the_child_claimed_under_c(self):
        # A single parent with two children draws the equivalent-to-spouse
        # amount for one, so only the OTHER counts as a dependant. Without the
        # exclusion the household is overpaid by one dependant amount.
        _settle, inst = build_refundable_credit_pools(_defs(), POOR, _ctx())
        assert inst[3] == pytest.approx(504.0 + 252.0 + 126.0)


class TestRenterEligibility:
    def test_only_a_renting_household_draws_the_renters_credit(self):
        settle, _inst = build_refundable_credit_pools(_defs(), POOR, _ctx())
        assert settle[0] == 400.0  # renting
        assert settle[3] == 0.0  # owning
        assert settle.sum() == 400.0


class TestTaperAppliesToTheInstrumentNotTheComponent:
    """Components sharing a `credit_name` are summed FIRST, then tapered once."""

    def test_a_family_is_tapered_on_the_family_threshold_not_the_single_one(self):
        """The threshold belongs to the HOUSEHOLD, not to a component row.

        The climate rows publish two -- 41,071 on the individual row, 57,288 on
        the family rows. Taking any one row's figure for the whole instrument
        would taper a family from 41,071 and strip most of its credit: at a
        household income of 50,000 that is 178.58 instead of the full 882.

        The income is stated as a HOUSEHOLD total split across its members,
        because the means test is on household income -- giving every member
        50,000 would test hh0 on 150,000 and taper it away.
        """
        income = np.array([25000.0, 25000.0, 0.0, 50000.0, 0.0, 0.0])
        _settle, inst = build_refundable_credit_pools(_defs(), income, _ctx())
        assert inst[0] == pytest.approx(882.0)  # household 50,000, under 57,288


class TestDeliverySplit:
    """The two paths refer to DIFFERENT years and must never be merged."""

    def test_each_credit_lands_in_its_own_leg(self):
        settle, inst = build_refundable_credit_pools(_defs(), POOR, _ctx())
        assert settle.sum() == 400.0  # renter's, with the settlement
        assert inst.sum() == 882.0 * 2  # climate, over the following year


class TestScalingToAgentUnits:
    """The amounts are per-person statutory dollars; agents are not people."""

    def test_currency_fields_scale_and_rates_do_not(self):
        from macromodel.configurations.central_government_configuration import (
            CentralGovernmentConfiguration,
            RefundableCreditDef,
        )
        from macromodel.country.country import _scale_pit_policy

        d = RefundableCreditDef(
            credit_name=CLIMATE,
            credit="Dependant Amount",
            delivery="instalments",
            amount=126.0,
            amount_basis="per_dependant",
            clawback=57288.0,
            clawback_rate=0.02,
        )
        scaled = _scale_pit_policy(
            CentralGovernmentConfiguration(pit_refundable_tax_credits=[d]), 1000
        ).pit_refundable_tax_credits[0]

        assert scaled.amount == pytest.approx(126_000.0)
        assert scaled.clawback == pytest.approx(57_288_000.0)
        # Dimensionless: scaling a rate would change the taper itself.
        assert scaled.clawback_rate == pytest.approx(0.02)


class TestEligibilityHasThreeLimbs:
    """19 or older, OR a spouse, OR a parent residing with their child.

    Gating on age alone excludes an under-19 parent or spouse, who is eligible.
    Widening it cannot double-pay: only one person receives the credit on
    behalf of a family, which the single-claimant design already guarantees.
    """

    def test_an_under_19_parent_can_claim(self):
        # Limb 3. An 18-year-old with a younger child in the household is a
        # parent, so the family is eligible even though nobody is 19.
        ages = np.array([18.0, 2.0])
        ctx = _ctx(
            tenure=(RENTS,),
            ages=ages,
            corr=np.array([0, 0]),
            hh_type=np.array([HouseholdType.SINGLE_PARENT_WITH_CHILDREN], dtype=object),
        )
        _settle, inst = build_refundable_credit_pools(_defs(), np.zeros(2), ctx)
        assert inst[0] > 0.0, "an under-19 parent was excluded"
        assert inst[1] == 0.0, "the child must not be a claimant"
