"""Contract tests binding the published credit schedule to the runtime.

The credit pool is driven by a schedule the data layer publishes. Two contracts
hold that pairing together, and neither was previously asserted anywhere:

* every phaseout column the schedule populates must reach a runtime branch and
  change the credit base — a parameter no branch reads is a silent policy loss;
* every credit the configuration builder lets through must have a runtime
  branch, since the builder's allow-list and ``_credit_amount``'s branches are
  coupled only by the credit-name string.

Fixtures read the committed schedules rather than invented amounts, per the
testing guideline ("use actual data values in sample data, not synthetic ones").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macro_data.readers.taxation import TaxationReader
from macro_data.readers.taxation.personal_income_tax.nrtc_schedule import NRTCSchedule
from macro_data.readers.taxation.personal_income_tax.pit_schedule import (
    compute_personal_income_tax,
)
from macromodel.agents.central_government import pit_pools
from macromodel.agents.central_government.central_government import (
    pit_credit_defs_to_state_dicts,
)
from macromodel.agents.central_government.pit_pools import (
    PitContext,
    build_credit_base_pool,
    build_taxable_income_pool,
)
from macromodel.agents.households.household_properties import HouseholdType
from macromodel.configurations import CentralGovernmentConfiguration
from macromodel.configurations.tax_parameters.central_government_builder import (
    _credit_component_to_def,
    activate_taxation,
)

# Committed schedules.
#   parents[0]=test_central_government [1]=test_agents [2]=unit
#   [3]=test_macromodel [4]=tests [5]=repo root
_COMMITTED_PIT_DIR = Path(__file__).resolve().parents[5] / "spoof_data" / "freda" / "personal_income_tax"
_CREDITS_CSV = _COMMITTED_PIT_DIR / "non_refundable_tax_credits.csv"

_YEAR = 2014
_JURISDICTION = "bc"


@pytest.fixture(scope="module")
def published_credits():
    """The credit components BC publishes for the base year."""
    schedule = NRTCSchedule.from_csv(_CREDITS_CSV, jurisdiction=_JURISDICTION)
    return schedule.get_credits(_YEAR)


def _state_dict(component) -> dict:
    """The runtime credit dict for *component*, as the agent would build it."""
    definition = _credit_component_to_def(component)
    if definition is None:
        return {}
    return {
        "credit": definition.credit,
        "amount": definition.amount,
        "age_min": definition.eligibility_age_min,
        "clawback": definition.clawback,
        "top": definition.top,
    }


def _is_derived(column: str, published: dict) -> bool:
    """Whether *column* merely restates the credit's other published columns.

    A dollar-for-dollar taper starting at ``clawback`` exhausts the credit at
    ``amount + clawback``, so a ``top`` equal to that sum adds no policy the
    other two do not already carry. A ``top`` that differs — or one published
    without a ``clawback``, as the eligible-dependant credit is — does.
    """
    if column != "top":
        return False
    clawback = published.get("clawback")
    if clawback is None:
        return False
    return published["top"] == pytest.approx(published["amount"] + clawback)


def _eligible_ctx() -> PitContext:
    """A population in which every published credit is actually claimable.

    One fixture cannot do this: a household is either a couple or a single
    parent, so a credit that is merely ineligible would look indistinguishable
    from one whose parameters are ignored. Two households cover both shapes,
    with incomes placed inside each credit's taper so a consumed threshold
    necessarily moves the base.

    ind0/ind1 — senior couple (Age Amount, Spousal Amount);
    ind2/ind3 — single parent with an earning minor (Equivalent To Spouse).
    """
    return PitContext(
        employee_income=np.array([40000.0, 5000.0, 40000.0, 2000.0]),
        employee_si_rate=0.0,
        individuals_age=np.array([70, 68, 45, 16]),
        individuals_corr_households=np.array([0, 0, 1, 1]),
        households_type=np.array(
            [
                HouseholdType.TWO_ADULTS_ONE_AT_LEAST_65,
                HouseholdType.SINGLE_PARENT_WITH_CHILDREN,
            ],
            dtype=object,
        ),
    )


class TestPublishedParametersReachTheRuntime:
    """A populated phaseout column must change the credit base.

    Perturbing the parameter and observing an unchanged base means no runtime
    branch consumes it: the published policy is silently discarded.
    """

    @pytest.mark.parametrize("column", ["clawback", "top"])
    def test_phaseout_column_changes_the_credit_base(self, published_credits, column):
        """Every credit publishing *column* must respond to it."""
        ctx = _eligible_ctx()
        taxable = build_taxable_income_pool(ctx)

        checked = []
        ignored = []
        ineligible = []
        for component in published_credits:
            published = _state_dict(component)
            if not published or published.get(column) is None:
                continue
            if _is_derived(column, published):
                # A column restating others carries no independent policy:
                # honouring the columns it derives from honours it too.
                continue

            perturbed = dict(published)
            # Move the threshold in the direction that keeps the schedule valid:
            # a clawback down, a top up, so top always stays above clawback.
            # Perturbing either one inward would produce a malformed phaseout,
            # which the Age Amount rejects outright, and a raise says nothing
            # about whether the parameter is read.
            factor = 0.5 if column == "clawback" else 1.5
            perturbed[column] = float(published[column]) * factor

            base_published = build_credit_base_pool([published], taxable, ctx)
            base_perturbed = build_credit_base_pool([perturbed], taxable, ctx)

            if not base_published.any():
                # Granted to nobody here, so an unchanged base would say
                # nothing about whether the threshold is read.
                ineligible.append(component.credit)
                continue

            checked.append(component.credit)
            if np.array_equal(base_published, base_perturbed):
                ignored.append(component.credit)

        assert not ineligible, (
            f"the fixture grants these credits to nobody, so their {column!r} "
            f"cannot be exercised — widen _eligible_ctx: {ineligible}"
        )
        assert checked, f"no published credit carries a {column!r} column"
        assert not ignored, (
            f"credits publish a {column!r} threshold that no runtime branch "
            f"consumes, so the published policy is silently discarded: {ignored}"
        )


class TestBuilderAllowListMatchesRuntimeBranches:
    """Credits the builder admits must have a runtime branch.

    The builder's expressibility filter and ``_credit_amount``'s branches are
    coupled only by the credit-name string. A credit that passes the filter
    with no matching branch reaches the fail-closed fallback and contributes
    zero — correct as a safety net, but silent as a configuration error.
    """

    def test_admitted_credits_all_have_a_runtime_branch(self, published_credits):
        admitted = [c for c in published_credits if _credit_component_to_def(c)]
        assert admitted, "builder admitted no credits from the published schedule"

        ctx = _eligible_ctx()
        taxable = build_taxable_income_pool(ctx)

        # The fail-closed fallback records the credits it could not dispatch.
        # This module-level set is warn-once and never cleared, so bracket the
        # assertion with clears: entering, so an earlier test cannot seed a
        # false positive; leaving, so this test cannot mask a later one.
        pit_pools._UNMAPPED_KINDS_WARNED.clear()
        try:
            for component in admitted:
                build_credit_base_pool([_state_dict(component)], taxable, ctx)
            unbranched = sorted(pit_pools._UNMAPPED_KINDS_WARNED)
        finally:
            pit_pools._UNMAPPED_KINDS_WARNED.clear()

        assert not unbranched, (
            "credits pass the builder's expressibility filter but have no "
            f"runtime branch, so they silently contribute zero: {unbranched}"
        )


class TestPublishedScheduleReachesTheTaxComputation:
    """The published CSV must survive every hop to the tax it produces.

    Reader, configuration builder, the agent's state dicts, the pools and the
    bracket walk are each covered in isolation elsewhere; nothing previously
    drove one published amount through all of them. The oracle is the published
    figure, computed by hand from the schedule, not from the code.
    """

    def test_published_personal_amount_relieves_tax_at_the_bottom_rate(self):
        reader = TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction=_JURISDICTION)
        config = activate_taxation(
            base_config=CentralGovernmentConfiguration(activate_progressive_pit=True),
            taxation_reader=reader,
            year=_YEAR,
        )

        # A lone filer: no age and no household context, so the Personal Amount
        # is the only credit for which they qualify.
        ctx = PitContext(employee_income=np.array([40000.0]), employee_si_rate=0.0)
        taxable = build_taxable_income_pool(ctx)
        credit_defs = pit_credit_defs_to_state_dicts(config.pit_non_refundable_tax_credits)
        credit_base = build_credit_base_pool(credit_defs, taxable, ctx)

        # BC 2014 published Personal Amount, valued at the published bottom rate.
        assert credit_base[0] == pytest.approx(9869.0)

        uppers = [upper for upper, _ in config.pit_brackets]
        rates = [rate for _, rate in config.pit_brackets]
        gross = compute_personal_income_tax(taxable, np.array(uppers), np.array(rates))
        net = np.maximum(0.0, gross - credit_base * rates[0])

        # The filer's tax must exceed the credit, or the floor would mask it.
        assert gross[0] > 9869.0 * 0.0506
        assert (gross - net)[0] == pytest.approx(9869.0 * 0.0506)
