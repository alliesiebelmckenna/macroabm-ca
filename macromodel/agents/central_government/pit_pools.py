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
from dataclasses import dataclass, replace

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
    bounds) are converted to the same units at construction by
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


# Raw income streams the annualization scales; the grossed-up dividend is
# annualized at its source so its credit scales with it exactly once.
PIT_INCOME_STREAMS = frozenset({"employee_income", "rental_income", "financial_income"})

# Every stream ``build_taxable_income_pool`` sums. Deliberately NOT the same set
# as the one above: membership here means the pooled divide-by-factor applies to
# the stream, so it must be scaled up somewhere or it is understated.
_POOLED_STREAMS = (
    "employee_income",
    "rental_income",
    "financial_income",
    "grossed_up_dividend",
)


def annualize_pit_context(
    ctx: PitContext,
    factor: float,
    streams: frozenset[str] = PIT_INCOME_STREAMS,
) -> PitContext:
    """Scale per-step income to a yearly rate for assessment.

    The brackets and credit amounts are annual, so income is scaled up before it
    is assessed and ``compute_pit`` divides the resulting tax back down by the
    same factor. Rates and demographic fields are untouched.

    Args:
        ctx: Per-individual income and context.
        factor: Steps per year; 1 returns ``ctx`` unchanged.
        streams: Income-stream fields to scale.

    Returns:
        A new context with the named streams scaled.
    """
    if factor == 1.0:
        return ctx
    scaled = {
        name: getattr(ctx, name) * factor
        for name in streams
        if getattr(ctx, name) is not None
    }
    return replace(ctx, **scaled)


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

    # Social transfers are deliberately outside the taxable pool; they are not
    # a missing stream.

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


def build_withheld_income_pool(ctx: PitContext) -> np.ndarray:
    """The pool the PERIOD withholds against: employment income only.

    The split narrows what is withheld, not what is assessed. The year's
    liability is still taken on the full base -- ``build_taxable_income_pool``
    -- at the filing; only the per-period withholding stops reaching income
    that no one is paid on a quarterly schedule.

    Employment is net of the employee social-insurance levy, exactly as in the
    full pool, so the two agree on the one stream they share.

    Args:
        ctx: The annualized PIT context.

    Returns:
        Withheld-against income per individual.
    """
    return ctx.employee_income * (1.0 - ctx.employee_si_rate)


def assert_pooled_streams_are_scaled(ctx: PitContext, streams=PIT_INCOME_STREAMS) -> None:
    """Every stream in the taxable pool must be scaled by F somewhere.

    W1, and it guards a trap rather than a style: ``annualize_pit_context``
    scales PER STREAM while the scale-down at ``central_government.py`` divides
    the WHOLE pooled array by the same factor. A stream that sits in the pool
    but in no scaling site is therefore divided and never multiplied --
    understated by a factor of F, silently.

    The invariant is *pool membership implies a scaling site*, NOT *membership
    of ``PIT_INCOME_STREAMS``*: the grossed-up dividend is outside the frozenset
    and correct, because it is scaled at its source.

    Args:
        ctx: The PIT context whose populated streams are checked.
        streams: Names ``annualize_pit_context`` scales.

    Raises:
        ValueError: If a populated pooled stream is scaled nowhere.
    """
    scaled_elsewhere = {"employee_income", "grossed_up_dividend"}
    unscaled = [
        name
        for name in _POOLED_STREAMS
        if getattr(ctx, name, None) is not None
        and name not in streams
        and name not in scaled_elsewhere
    ]
    if unscaled:
        raise ValueError(
            f"{', '.join(unscaled)} are summed into the taxable pool but are "
            f"scaled by no annualization site, so the pooled divide-by-factor "
            f"understates them {int(1)}-for-F. Either add them to the scaling "
            f"set or drop them from build_taxable_income_pool -- doing only one "
            f"is the defect."
        )


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
    spouse_income: np.ndarray | None  # the other spouse's taxable base; inf if none


def _household_context(
    n_ind: int,
    taxable_income_per_ind: np.ndarray,
    ctx: PitContext,
) -> _HouseholdContext:
    """Build couple / single-parent flags and spouse-income per individual.

    Spouse income is the other spouse's taxable base in a couple household, and
    ``inf`` for everyone else (so an income-tested credit clamps to zero where
    there is no spouse). The two eldest adults are taken as the spouses, so a
    resident adult child neither blocks the pairing nor joins it; without ages
    only unambiguous two-member couples are paired.
    """
    corr = ctx.individuals_corr_households
    hh_type = ctx.households_type

    if corr is None or hh_type is None:
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
        # Bounded on both sides: a negative sentinel (an unassigned household)
        # is a legal numpy index and would otherwise borrow the last household.
        [hh_type[h] if 0 <= h < len(hh_type) else None for h in hh_of_ind]
    )

    in_couple = np.array([t in couple_types for t in hh_type_of_ind])
    is_single_parent = np.array([t in single_parent_types for t in hh_type_of_ind])

    # Spouse income is the other spouse's taxable base in a couple household,
    # inf elsewhere. Pairing is over adults (age >= 18) only, since the
    # individual array includes children.
    spouse_income = np.full(n_ind, np.inf)

    ages = ctx.individuals_age
    if ages is not None:
        adult_idx = np.where(np.asarray(ages) >= 18)[0]
    else:
        # Without ages we cannot single out adults; fall back to all members
        # (correct for childless couples, the only 2-member couple case).
        adult_idx = np.arange(n_ind)

    adult_hh = hh_of_ind[adult_idx]
    if ages is not None:
        # Eldest first within each household, so the two spouses sort ahead of
        # any resident adult child. Age is the available proxy: the model
        # records no spousal link.
        order = np.lexsort((-np.asarray(ages)[adult_idx], adult_hh))
        # A third adult is a resident adult child. Tax is assessed per person,
        # so their presence must not disturb the couple's own credits.
        allow_extra_adults = True
    else:
        # Without ages, adults cannot be told from children, so an extra member
        # is indistinguishable from a spouse; pair only unambiguous couples.
        order = np.argsort(adult_hh, kind="stable")
        allow_extra_adults = False

    sorted_idx = adult_idx[order]
    _, group_start, group_counts = np.unique(
        adult_hh[order], return_index=True, return_counts=True
    )

    pair_groups = group_counts >= 2 if allow_extra_adults else group_counts == 2
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


def _published_exemption(tc: dict, amount: float) -> float:
    """Income a spouse or dependant may earn before the credit starts tapering.

    The Spousal Amount and the eligible-dependant credit share this mechanism
    but publish it differently: one carries an explicit ``clawback``, the other
    implies it as ``top - amount``. Deriving it here means neither branch loses
    the threshold when its jurisdiction publishes only the other column. The
    two credits keep their own rows, so they may still differ in value.
    """
    clawback = tc.get("clawback")
    if clawback is not None:
        return float(clawback)
    top = tc.get("top")
    if top is not None:
        return max(0.0, float(top) - amount)
    return 0.0


def _sole_claimant_credit(
    amount: float,
    exemption: float,
    ages: np.ndarray,
    hh_of_ind: np.ndarray,
    is_single_parent: np.ndarray,
    taxable_income_per_ind: np.ndarray,
    n_ind: int,
) -> np.ndarray:
    """Grant *amount* once per qualifying household, to its eldest adult.

    A household qualifies when it is single-parent typed and contains at least
    one individual under 18. The eldest adult stands in for the supporting
    parent: the model records no parent-child link, so age is the available
    proxy. Households with no minor return nothing.

    The claim is reduced by the dependant's income above *exemption*. A filer
    may claim for one dependant, so the lowest-income minor is used — the choice
    that yields the largest credit, and the one a filer would make.
    """
    adult = ages >= 18
    dependants = np.where(is_single_parent & ~adult)[0]
    base = np.zeros(n_ind)
    if dependants.size == 0:
        return base

    qualifying_hh = np.unique(hh_of_ind[dependants])
    claimants = np.where(
        adult & is_single_parent & np.isin(hh_of_ind, qualifying_hh)
    )[0]
    if claimants.size == 0:
        return base

    # Group by household, eldest first, then keep each group's first member.
    ordered = claimants[np.lexsort((-ages[claimants], hh_of_ind[claimants]))]
    _, first_in_group = np.unique(hh_of_ind[ordered], return_index=True)
    claimant_idx = ordered[first_in_group]

    # Likewise per household, poorest dependant first.
    dep_ordered = dependants[
        np.lexsort((taxable_income_per_ind[dependants], hh_of_ind[dependants]))
    ]
    dep_hh, dep_first = np.unique(hh_of_ind[dep_ordered], return_index=True)
    dep_income = taxable_income_per_ind[dep_ordered[dep_first]]

    # Every claimant's household holds a dependant, so the lookup always hits.
    matched = dep_income[np.searchsorted(dep_hh, hh_of_ind[claimant_idx])]
    base[claimant_idx] = np.maximum(
        0.0, amount - np.maximum(0.0, matched - exemption)
    )
    return base


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
        if cs is None and cc is None:
            # No phaseout published: a genuinely unphased age credit.
            return np.where(eligible, amount, 0.0)
        if cs is None or cc is None or cc <= cs:
            # A half-published phaseout is a data error, not an unphased credit.
            # Falling back to the full amount here would over-credit every
            # eligible filer above the threshold, silently and by the whole
            # taper, so it fails loudly instead.
            raise ValueError(
                f"Age Amount publishes an incomplete phaseout (clawback={cs}, "
                f"top={cc}). Both are required, and top must exceed clawback."
            )
        clawback_rate = amount / (cc - cs)
        excess = np.maximum(0.0, taxable_income_per_ind - cs)
        return np.where(
            eligible, np.maximum(0.0, amount - clawback_rate * excess), 0.0
        )

    # Spousal Amount: the base less the spouse's income above the published
    # exemption. spouse_income is inf for non-couples so they clamp to zero.
    if credit == "Spousal Amount":
        if household.in_couple is None or household.spouse_income is None:
            return zeros
        exemption = _published_exemption(tc, amount)
        excess = np.maximum(0.0, household.spouse_income - exemption)
        return np.maximum(0.0, amount - excess)

    # Equivalent To Spouse Amount: one claim per single-parent household
    # supporting a minor child, taken by the parent. The exception for a dependant
    # aged 18 or over with an infirmity is not expressed, because the
    # model carries no infirmity signal; a household whose children have all
    # reached 18 is therefore treated as ineligible rather than granted it.
    if credit == "Equivalent To Spouse Amount":
        corr = ctx.individuals_corr_households
        if household.is_single_parent is None or ages is None or corr is None:
            return zeros
        exemption = _published_exemption(tc, amount)
        return _sole_claimant_credit(
            amount,
            exemption,
            np.asarray(ages),
            np.asarray(corr).astype(int),
            household.is_single_parent,
            taxable_income_per_ind,
            n_ind,
        )

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
