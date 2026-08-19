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

# Universal credits; anything unmapped falls closed to zero in _credit_amount.
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
    # None when dividend integration is off.
    grossed_up_dividend: np.ndarray | None = None

    # household / demographic context (tax-credit eligibility)
    individuals_age: np.ndarray | None = None
    individuals_corr_households: np.ndarray | None = None
    households_type: np.ndarray | None = None
    # Housing tenure per household, HFCS codes.
    households_tenure: np.ndarray | None = None


# Streams the annualization scales; the dividend is scaled at its source.
PIT_INCOME_STREAMS = frozenset({"employee_income", "rental_income", "financial_income"})

_QUALIFIED_DEPENDANT_AGE = 19
_ADULT_AGE = 19
# HFCS tenure: 3 rented/sublet. Social housing is -1 and deliberately excluded.
_RENTING_TENURE_CODES = (3,)

# Streams the pool sums; each must be scaled somewhere or it is understated.
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
    scaled = {name: getattr(ctx, name) * factor for name in streams if getattr(ctx, name) is not None}
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
    pool = ctx.employee_income * (1.0 - ctx.employee_si_rate)

    if ctx.rental_income is not None:
        pool = pool + ctx.rental_income
    if ctx.financial_income is not None:
        pool = pool + ctx.financial_income
    if ctx.grossed_up_dividend is not None:
        pool = pool + ctx.grossed_up_dividend


    # Social transfers are deliberately outside the taxable pool.

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
    dividend_tax_credit = eligible_dtc_rate * grossed_eligible + non_eligible_dtc_rate * grossed_non_eligible
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

    It guards a trap rather than a style: ``annualize_pit_context``
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
        if getattr(ctx, name, None) is not None and name not in streams and name not in scaled_elsewhere
    ]
    if unscaled:
        raise ValueError(
            f"{', '.join(unscaled)} are summed into the taxable pool but are "
            f"scaled by no annualization site, so the pooled divide-by-factor "
            f"understates them {int(1)}-for-F. Either add them to the scaling "
            f"set or drop them from build_taxable_income_pool -- doing only one "
            f"is the defect."
        )


def build_refundable_credit_pools(
    credit_defs,
    annual_income_per_ind: np.ndarray,
    ctx: PitContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Value the refundable credits on a YEAR's income, split by delivery.

    Called once, at the filing, never per period: the taper needs the settled
    year's income, which does not exist until then. This is why the refundable
    credit is a filing-time callable rather than an arm alongside the
    non-refundable ones -- those are per-period inputs, and there is nothing to
    compute per period here.

    Components sharing a ``credit_name`` are SUMMED FIRST and tapered ONCE. The
    elimination point therefore belongs to the household, not to any row, which
    is why the schedule leaves ``top`` blank on the climate rows: tapering each
    component independently would over-reduce the credit.

    Nothing is floored against tax. A refundable credit is paid whether or not
    tax is owed -- that is what makes it refundable, and what makes it
    expenditure rather than revenue foregone.

    Args:
        credit_defs: Refundable credit definitions for the settled year, or
            ``None`` for no refundable credit.
        annual_income_per_ind: The settled year's income per individual, which
            the taper is applied to.
        ctx: The PIT context, for eligibility.

    Returns:
        ``(settlement_per_ind, instalments_per_ind)`` -- both per-individual
        arrays in agent dollars, one per delivery path.
    """
    n_ind = len(annual_income_per_ind)
    zeros = np.zeros(n_ind)
    if not credit_defs:
        return zeros, zeros.copy()

    household = _household_context(n_ind, annual_income_per_ind, ctx)

    # Means-tested on HOUSEHOLD income, mapped back per individual.
    means_income = _household_income_per_ind(annual_income_per_ind, ctx, n_ind)
    # Sum each instrument's components before tapering; the threshold is the household's.
    by_instrument: dict[str, dict] = {}
    for d in credit_defs:
        key = str(d.get("credit_name", d.get("credit", "")))
        slot = by_instrument.setdefault(
            key,
            {
                "gross": np.zeros(n_ind),
                "delivery": str(d.get("delivery", "settlement")),
                "clawback": np.zeros(n_ind),
                "clawback_rate": d.get("clawback_rate"),
            },
        )
        drawn = _refundable_component(d, annual_income_per_ind, ctx, household, n_ind)
        slot["gross"] = slot["gross"] + drawn
        row_clawback = d.get("clawback")
        if row_clawback is not None:
            # Whichever threshold this household's own components reach.
            slot["clawback"] = np.where(
                drawn > 0.0,
                np.maximum(slot["clawback"], float(row_clawback)),
                slot["clawback"],
            )

    settlement = np.zeros(n_ind)
    instalments = np.zeros(n_ind)
    for slot in by_instrument.values():
        net = _tapered(slot["gross"], means_income, slot["clawback"], slot["clawback_rate"])
        if slot["delivery"] == "instalments":
            instalments = instalments + net
        else:
            settlement = settlement + net
    return settlement, instalments


def _household_income_per_ind(annual_income_per_ind: np.ndarray, ctx: PitContext, n_ind: int) -> np.ndarray:
    """Each individual's HOUSEHOLD income, for the means test.

    Falls back to individual income when the household mapping is absent, which
    keeps a context without it usable rather than silently testing on zero.
    """
    corr = ctx.individuals_corr_households
    if corr is None:
        return annual_income_per_ind
    hh = np.asarray(corr).astype(int)
    totals = np.bincount(hh, weights=annual_income_per_ind, minlength=int(hh.max()) + 1)
    return totals[hh]


def _tapered(
    gross: np.ndarray,
    income: np.ndarray,
    clawback,
    clawback_rate,
) -> np.ndarray:
    """Reduce a summed instrument by its taper, floored at zero.

    Floored at zero because a taper cannot turn a credit into a charge; this is
    NOT the non-refundable floor against tax, which does not apply here.
    """
    if clawback is None or clawback_rate is None:
        return np.maximum(0.0, gross)
    # Per individual: the threshold depends on which components the household draws.
    excess = np.maximum(0.0, income - np.asarray(clawback, dtype=float))
    return np.maximum(0.0, gross - float(clawback_rate) * excess)


def _refundable_component(
    tc: dict,
    income: np.ndarray,
    ctx: PitContext,
    household: "_HouseholdContext",
    n_ind: int,
) -> np.ndarray:
    """One component's UNTAPERED value, placed on the household's claimant.

    Every component is a HOUSEHOLD entitlement paid to ONE person. A family
    receives the individual amount once, a spouse amount once, and one amount
    per child -- not the individual amount per adult, nor the spouse amount to
    everyone in a couple. Granting per member overpays a couple twofold and
    pays children who have no income to receive it against.

    The claimant is the household's eldest adult, matching
    ``_sole_claimant_credit``, so a household with no adult draws nothing.

    Fail-closed: a credit with no branch here contributes zero rather than
    being granted universally.
    """
    zeros = np.zeros(n_ind)
    credit = str(tc.get("credit", ""))
    amount = float(tc.get("amount") or 0.0)
    ages = ctx.individuals_age
    corr = ctx.individuals_corr_households
    if ages is None or corr is None:
        return zeros

    ages = np.asarray(ages, dtype=float)
    hh = np.asarray(corr).astype(int)
    claimant = _eldest_adult_index(ages, hh, n_ind, household)
    n_hh = int(hh.max()) + 1 if len(hh) else 0

    # Per-HOUSEHOLD entitlement, then placed on the claimant.
    per_hh = np.zeros(n_hh)

    if credit == "Eligible Individual Amount":
        age_min = tc.get("eligibility_age_min")
        if age_min is None:
            return zeros
        # The claimant is a qualifying adult by construction.
        for _i, h in claimant.items():
            per_hh[h] = amount

    elif credit == "Spousal Amount":
        if household.in_couple is None:
            return zeros
        for _i, h in claimant.items():
            if household.in_couple[_i]:
                per_hh[h] = amount

    elif credit == "Equivalent To Spouse Amount":
        if household.is_single_parent is None:
            return zeros
        minors_in = np.bincount(hh[ages < _QUALIFIED_DEPENDANT_AGE], minlength=n_hh)
        for _i, h in claimant.items():
            if household.is_single_parent[_i] and minors_in[h] > 0:
                per_hh[h] = amount

    elif credit == "Dependant Amount":
        per_hh = amount * _dependant_counts(ages, hh, household, n_hh)

    elif credit == "Renter's Amount":
        tenure = ctx.households_tenure
        if tenure is None:
            return zeros
        renting = np.isin(np.asarray(tenure), _RENTING_TENURE_CODES)
        for _i, h in claimant.items():
            if h < len(renting) and renting[h]:
                per_hh[h] = amount

    else:
        return zeros

    out = np.zeros(n_ind)
    for i, h in claimant.items():
        out[i] = per_hh[h]
    return out


def _dependant_counts(
    ages: np.ndarray,
    hh: np.ndarray,
    household: "_HouseholdContext",
    n_hh: int,
) -> np.ndarray:
    """Qualifying dependants per household, after the paragraph-(d) exclusion.

    Under 19, per s.8.1(2)'s import of the federal definition -- NOT the
    under-18 boundary the non-refundable equivalent-to-spouse credit uses. The
    two come from different provisions, so the divergence is correct.

    Paragraph (d) counts dependants "other than a qualified dependant in
    respect of whom an amount is included under paragraph (c)", so the child
    who drew the equivalent-to-spouse amount is not counted again. Without it a
    single parent with two children is overpaid by one dependant amount.
    """
    counts = np.bincount(hh[ages < _QUALIFIED_DEPENDANT_AGE], minlength=n_hh).astype(float)
    if household.is_single_parent is not None:
        sp_hh = np.unique(hh[np.asarray(household.is_single_parent)])
        if len(sp_hh):
            counts[sp_hh] = np.maximum(0.0, counts[sp_hh] - 1.0)
    return counts


def _eldest_adult_index(
    ages: np.ndarray,
    hh_of_ind: np.ndarray,
    n_ind: int,
    household: "_HouseholdContext | None" = None,
) -> dict[int, int]:
    """Map each household's claimant to that household.

    Eligibility has three limbs: 19 or older, OR having a spouse or common-law
    partner, OR being a parent residing with their child — so gating on age alone
    would wrongly exclude an under-19 parent or spouse. Only one person may claim
    on behalf of a family, so the limbs widen who may claim without ever paying a
    household twice; the claimant is its eldest qualifying member. The parent limb
    is inferred, the model carrying no parent linkage: a member with a younger
    minor in the same household is treated as that child's parent. Measured empty
    on the current population, so the limbs beyond age change no figure today.

    Returns:
        ``{individual index: household index}``, one entry per household that
        has a qualifying claimant.
    """
    # Youngest minor per household, so "has a younger minor" is one lookup.
    n_hh = int(hh_of_ind.max()) + 1 if len(hh_of_ind) else 0
    youngest_minor = np.full(n_hh, np.inf)
    for i in range(n_ind):
        if ages[i] < _QUALIFIED_DEPENDANT_AGE:
            h = int(hh_of_ind[i])
            youngest_minor[h] = min(youngest_minor[h], ages[i])

    in_couple = household.in_couple if household is not None else None

    best: dict[int, tuple[float, int]] = {}
    for i in range(n_ind):
        h = int(hh_of_ind[i])
        qualifies = (
            ages[i] >= _ADULT_AGE  # limb 1: 19+
            or (in_couple is not None and bool(in_couple[i]))  # limb 2: spouse
            or ages[i] > youngest_minor[h]  # limb 3: parent
        )
        if not qualifies:
            continue
        if h not in best or ages[i] > best[h][0]:
            best[h] = (ages[i], i)
    return {idx: h for h, (_age, idx) in best.items()}


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
        # Bounded both sides: a negative sentinel is a legal numpy index.
        [hh_type[h] if 0 <= h < len(hh_type) else None for h in hh_of_ind]
    )

    in_couple = np.array([t in couple_types for t in hh_type_of_ind])
    is_single_parent = np.array([t in single_parent_types for t in hh_type_of_ind])

    # The other spouse's taxable base in a couple, inf elsewhere; pairing is adults-only.
    spouse_income = np.full(n_ind, np.inf)

    ages = ctx.individuals_age
    if ages is not None:
        adult_idx = np.where(np.asarray(ages) >= 18)[0]
    else:
        # Without ages, fall back to all members.
        adult_idx = np.arange(n_ind)

    adult_hh = hh_of_ind[adult_idx]
    if ages is not None:
        # Eldest first, so spouses sort ahead of a resident adult child.
        order = np.lexsort((-np.asarray(ages)[adult_idx], adult_hh))
        # A third adult is a resident adult child.
        allow_extra_adults = True
    else:
        # Without ages an extra member is indistinguishable from a spouse.
        order = np.argsort(adult_hh, kind="stable")
        allow_extra_adults = False

    sorted_idx = adult_idx[order]
    _, group_start, group_counts = np.unique(adult_hh[order], return_index=True, return_counts=True)

    pair_groups = group_counts >= 2 if allow_extra_adults else group_counts == 2
    first = sorted_idx[group_start[pair_groups]]
    second = sorted_idx[group_start[pair_groups] + 1]

    # Couple-type households only; in_couple encodes both type and bounds.
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
    claimants = np.where(adult & is_single_parent & np.isin(hh_of_ind, qualifying_hh))[0]
    if claimants.size == 0:
        return base

    # Group by household, eldest first, then keep each group's first member.
    ordered = claimants[np.lexsort((-ages[claimants], hh_of_ind[claimants]))]
    _, first_in_group = np.unique(hh_of_ind[ordered], return_index=True)
    claimant_idx = ordered[first_in_group]

    # Likewise per household, poorest dependant first.
    dep_ordered = dependants[np.lexsort((taxable_income_per_ind[dependants], hh_of_ind[dependants]))]
    dep_hh, dep_first = np.unique(hh_of_ind[dep_ordered], return_index=True)
    dep_income = taxable_income_per_ind[dep_ordered[dep_first]]

    # Every claimant's household holds a dependant, so the lookup always hits.
    matched = dep_income[np.searchsorted(dep_hh, hh_of_ind[claimant_idx])]
    base[claimant_idx] = np.maximum(0.0, amount - np.maximum(0.0, matched - exemption))
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
            # A half-published phaseout is a data error; fail loudly rather than over-credit.
            raise ValueError(
                f"Age Amount publishes an incomplete phaseout (clawback={cs}, "
                f"top={cc}). Both are required, and top must exceed clawback."
            )
        clawback_rate = amount / (cc - cs)
        excess = np.maximum(0.0, taxable_income_per_ind - cs)
        return np.where(eligible, np.maximum(0.0, amount - clawback_rate * excess), 0.0)

    # spouse_income is inf for non-couples, so they clamp to zero.
    if credit == "Spousal Amount":
        if household.in_couple is None or household.spouse_income is None:
            return zeros
        exemption = _published_exemption(tc, amount)
        excess = np.maximum(0.0, household.spouse_income - exemption)
        return np.maximum(0.0, amount - excess)

    # One claim per single-parent household; the infirmity exception is not modelled.
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

    # Fail closed: contribute zero rather than grant the amount to everyone.
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
