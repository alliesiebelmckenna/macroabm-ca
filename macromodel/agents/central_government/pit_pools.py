"""Assembly of the two Personal Income Tax (PIT) pools.

This is the processing phase of PIT: it turns raw per-agent state into the two
per-individual arrays the central government's tax core consumes — Pool A, the
taxable income per individual (``build_taxable_income_pool``), and Pool B, the
non-refundable tax-credit base per individual (``build_credit_base_pool``). The
``CentralGovernment`` agent then applies fixed policy to these pools.

Because the agent only ever sees the two finished pools, extending the model
with a new income stream or tax credit means editing only this module. A new
income stream is a field on ``PitContext`` and a line in
``build_taxable_income_pool``; a new credit is a branch in
``_credit_amount``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Credits granted to everyone; the fail-closed dispatch in _credit_amount
# gives any unmapped credit zero rather than granting it to everyone.
_UNIVERSAL_CREDIT_KINDS = frozenset({"Personal Amount"})

# Unmapped credits already warned about (warn once, not once per step).
_UNMAPPED_KINDS_WARNED: set[str] = set()


@dataclass
class PitContext:
    """Per-individual inputs needed to assemble the PIT pools.

    Income-stream fields are per individual. Household / demographic fields drive
    tax-credit eligibility and may be ``None`` when the data is unavailable (e.g.
    employee-only pre-calibration).

    Units invariant: every monetary field is in agent-level dollars. The policy
    dollars the pools are compared against (brackets, credit amounts, clawback
    bounds, deductions) are converted to the same units at construction by
    ``country._scale_pit_policy``, so a new income stream added here must already
    be in agent dollars.
    """

    # income streams (per individual)
    employee_income: np.ndarray
    employee_si_rate: float
    rental_income: np.ndarray | None = None
    financial_income: np.ndarray | None = None
    # Grossed-up firm dividend: a tax fiction for taxable income and the DTC
    # (the dividend received is unchanged). None when integration is off.
    grossed_up_dividend: np.ndarray | None = None

    # household / demographic context (tax-credit eligibility)
    individuals_age: np.ndarray | None = None
    individuals_corr_households: np.ndarray | None = None
    households_type: np.ndarray | None = None
    households_n_adults: np.ndarray | None = None


def build_taxable_income_pool(ctx: PitContext) -> np.ndarray:
    """Pool A: total taxable income per individual.

    Each income stream contributes its taxable amount (after any stream-specific
    adjustment such as the employee social-insurance offset or an inclusion
    rate). The pooled total later flows through the progressive brackets exactly
    once. To add a new income stream, add one line here (and a field on
    ``PitContext``).

    Args:
        ctx: Per-individual income and context.

    Returns:
        Taxable income per individual.
    """
    # Employee wages are taxed net of the employee social-insurance levy.
    pool = ctx.employee_income * (1.0 - ctx.employee_si_rate)

    if ctx.rental_income is not None:
        pool = pool + ctx.rental_income
    if ctx.financial_income is not None:
        pool = pool + ctx.financial_income
    if ctx.grossed_up_dividend is not None:
        # Grossed-up dividend (CRA line 12000): notional, so the household
        # receives only the un-grossed cash.
        pool = pool + ctx.grossed_up_dividend

    # pool = pool + ctx.pension_income            # ← example: new stream
    # pool = pool + ctx.capital_gains * 0.5       # ← example: inclusion rate

    return pool


def build_dividend_tax_items(
    dividend_income: np.ndarray,
    small_business_share: float,
    eligible_gross_up: float,
    non_eligible_gross_up: float,
    eligible_dtc_rate: float,
    non_eligible_dtc_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn the actual per-individual dividend into its two tax-only items.

    Implements the Canadian gross-up + dividend tax credit for firm dividends
    with a provisional uniform split: a share ``s`` of each dividend is treated
    as other-than-eligible (small-business-rate income) and ``1 - s`` as eligible
    (general-rate income). Each portion is grossed up at its own rate, and the
    dividend tax credit is that grossed-up amount valued at its own DTC rate.
    Neither changes the actual dividend received — they feed only the
    taxable-income pool and the credit pool.

    Args:
        dividend_income: Actual dividend received per individual (``D_i``).
        small_business_share: ``s`` — other-than-eligible share (0..1).
        eligible_gross_up: Gross-up rate for eligible dividends (0.38 → ×1.38).
        non_eligible_gross_up: Gross-up rate for other-than-eligible (0.18 → ×1.18).
        eligible_dtc_rate: DTC rate on grossed-up eligible dividends.
        non_eligible_dtc_rate: DTC rate on grossed-up other-than-eligible.

    Returns:
        ``(grossed_up_dividend, dividend_tax_credit)`` per individual.
    """
    eligible = (1.0 - small_business_share) * dividend_income
    non_eligible = small_business_share * dividend_income

    grossed_eligible = eligible * (1.0 + eligible_gross_up)
    grossed_non_eligible = non_eligible * (1.0 + non_eligible_gross_up)

    grossed_up_dividend = grossed_eligible + grossed_non_eligible
    dividend_tax_credit = (
        eligible_dtc_rate * grossed_eligible
        + non_eligible_dtc_rate * grossed_non_eligible
    )
    return grossed_up_dividend, dividend_tax_credit


def build_credit_base_pool(
    credit_defs: list[dict] | None,
    taxable_income_per_ind: np.ndarray,
    ctx: PitContext,
) -> np.ndarray:
    """Pool B: summed non-refundable tax-credit base per individual.

    ``credit_defs`` are the credit definitions owned by the government agent
    (``states["pit_non_refundable_tax_credits"]``), each a dict with ``credit``, ``amount`` and
    optional eligibility keys. The agent later values this base at the bottom
    marginal rate and subtracts it from gross tax, floored at zero. To add a new
    credit, add a branch in ``_credit_amount``.

    Args:
        credit_defs: Credit definitions, or ``None`` / empty for no credits.
        taxable_income_per_ind: Pool A — used by income-tested credits
            (Age Amount clawback, Spousal Amount).
        ctx: Per-individual demographic / household context.

    Returns:
        Summed credit base per individual (zeros when no credits apply).
    """
    n_ind = len(taxable_income_per_ind)
    if not credit_defs:
        return np.zeros(n_ind)

    household = _household_context(n_ind, taxable_income_per_ind, ctx)

    base = np.zeros(n_ind)
    for tc in credit_defs:
        base += _credit_amount(tc, taxable_income_per_ind, ctx, household)
    return base


@dataclass
class _HouseholdContext:
    """Derived per-individual household relationships for credit tests."""

    in_couple: np.ndarray | None
    is_single_parent: np.ndarray | None
    spouse_income: np.ndarray | None  # the OTHER adult's taxable base; inf if none


def _household_context(
    n_ind: int,
    taxable_income_per_ind: np.ndarray,
    ctx: PitContext,
) -> _HouseholdContext:
    """Build couple / single-parent flags and spouse-income per individual.

    Spouse income is the other adult's taxable base in a two-adult couple
    household, and ``inf`` for everyone else (so an income-tested credit
    clamps to zero where there is no spouse).
    """
    corr = ctx.individuals_corr_households
    hh_type = ctx.households_type
    hh_n_adults = ctx.households_n_adults

    if corr is None or hh_type is None or hh_n_adults is None:
        return _HouseholdContext(None, None, None)

    from macromodel.agents.households.household_properties import HouseholdType

    couple_types = {
        HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,
        HouseholdType.TWO_ADULTS_ONE_AT_LEAST_65,
        HouseholdType.TWO_ADULTS_WITH_ONE_CHILD,
        HouseholdType.TWO_ADULTS_WITH_TWO_CHILDREN,
        HouseholdType.TWO_ADULTS_WITH_AT_LEAST_THREE_CHILDREN,
    }
    single_parent_types = {HouseholdType.SINGLE_PARENT_WITH_CHILDREN}

    hh_of_ind = np.asarray(corr).astype(int)
    hh_type_of_ind = np.array(
        [hh_type[h] if h < len(hh_type) else None for h in hh_of_ind]
    )

    in_couple = np.array([t in couple_types for t in hh_type_of_ind])
    is_single_parent = np.array([t in single_parent_types for t in hh_type_of_ind])

    # Spouse income is the other adult's taxable base in a two-adult couple, inf
    # elsewhere. Pairing is over adults (age >= 18) only, since the individual
    # array includes children.
    spouse_income = np.full(n_ind, np.inf)

    ages = ctx.individuals_age
    if ages is not None:
        adult_idx = np.where(np.asarray(ages) >= 18)[0]
    else:
        # Without ages we cannot single out adults; fall back to all members
        # (correct for childless couples, the only 2-member couple case).
        adult_idx = np.arange(n_ind)

    adult_hh = hh_of_ind[adult_idx]
    order = np.argsort(adult_hh, kind="stable")
    sorted_idx = adult_idx[order]
    _, group_start, group_counts = np.unique(
        adult_hh[order], return_index=True, return_counts=True
    )

    # Couple households with exactly two adults.
    pair_groups = group_counts == 2
    first = sorted_idx[group_start[pair_groups]]
    second = sorted_idx[group_start[pair_groups] + 1]

    # Keep only couple-type households.  in_couple already encodes both the
    # type membership and the id bounds check, and both adults share it.
    is_couple_pair = in_couple[first]
    first = first[is_couple_pair]
    second = second[is_couple_pair]

    spouse_income[first] = taxable_income_per_ind[second]
    spouse_income[second] = taxable_income_per_ind[first]

    return _HouseholdContext(in_couple, is_single_parent, spouse_income)


def _credit_amount(
    tc: dict,
    taxable_income_per_ind: np.ndarray,
    ctx: PitContext,
    household: _HouseholdContext,
) -> np.ndarray:
    """Per-individual base for a single tax-credit component.

    Add a branch here to support a new credit. The returned array is the
    credit base (dollar amount), not the tax reduction — the agent values the
    summed base at the bottom marginal rate. A targeted credit whose required
    context (age, household relationships) is missing returns zero and must never
    fall through to a universal amount.

    Dispatch is fail-closed: only credits with a dedicated branch, an ``age_min``
    gate, or membership in ``_UNIVERSAL_CREDIT_KINDS`` contribute; any other credit
    contributes zero with a one-time warning. A genuinely universal new credit is
    activated by adding it to the allow-list, not by falling through.
    """
    credit = tc["credit"]
    amount = tc["amount"]
    age_min = tc.get("age_min")
    n_ind = len(taxable_income_per_ind)
    ages = ctx.individuals_age
    zeros = np.zeros(n_ind)

    # Age Amount: age-gated, with an optional own-income phaseout.
    if credit == "Age Amount":
        if age_min is None or ages is None:
            return zeros
        eligible = ages >= age_min
        cs = tc.get("clawback")
        cc = tc.get("top")
        if cs is not None and cc is not None and cc > cs:
            clawback_rate = amount / (cc - cs)
            excess = np.maximum(0.0, taxable_income_per_ind - cs)
            return np.where(
                eligible, np.maximum(0.0, amount - clawback_rate * excess), 0.0
            )
        return np.where(eligible, amount, 0.0)

    # Spousal Amount: max(0, base - spouse_income); spouse_income is inf for
    # non-couples so they clamp to zero.
    if credit == "Spousal Amount":
        if household.in_couple is None or household.spouse_income is None:
            return zeros
        return np.maximum(0.0, amount - household.spouse_income)

    # Equivalent To Spouse Amount: single parents only. The dependant has no
    # income in the model, so the full base applies where eligible.
    if credit == "Equivalent To Spouse Amount":
        if household.is_single_parent is None:
            return zeros
        return np.where(household.is_single_parent, amount, 0.0)

    # Other age-gated credits (age_min set, not Age Amount).
    if age_min is not None:
        if ages is None:
            return zeros
        return np.where(ages >= age_min, amount, 0.0)

    # Universal credits (explicit allow-list, e.g. Personal Amount).
    if credit in _UNIVERSAL_CREDIT_KINDS:
        return np.full(n_ind, float(amount))

    # Unknown credit: fail closed, contributing zero rather than granting the
    # amount to everyone.
    if credit not in _UNMAPPED_KINDS_WARNED:
        _UNMAPPED_KINDS_WARNED.add(credit)
        logger.warning(
            "Tax credit '%s' has no runtime branch in _credit_amount; "
            "contributing zero (fail-closed). Add a branch (or, for a "
            "genuinely universal credit, add it to "
            "_UNIVERSAL_CREDIT_KINDS) to activate it.",
            credit,
        )
    return zeros
