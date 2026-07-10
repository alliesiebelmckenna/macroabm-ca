from functools import lru_cache
from typing import Literal, Optional, get_args

from pydantic import BaseModel, Field

# ── Unit declarations for tax-policy fields ──────────────────────────────
# Statutory tax parameters are published in per-person dollars, while agent
# incomes are in agent-level dollars (each synthetic agent represents ``scale``
# people).  The conversion is applied once, model-side, by reflection over the
# unit declared in each field's metadata (``json_schema_extra={"unit": ...}``)
# — see ``country._scale_pit_policy`` — instead of a hand-maintained field
# list that a new field could silently miss:
#   "currency"      — a dollar amount; multiplied by the population scale.
#   "dimensionless" — a rate / share / ratio; never scaled.
#   "years"         — a calendar year or an age; never scaled.
# ``monetary_field_names`` enforces the declaration fail-closed: a numeric
# field with no unit raises, so adding a tax-policy field forces a units
# decision at definition time.  Structured fields (``pit_brackets``,
# ``pit_tax_credits``) are exempt here and handled explicitly by the scaler.
CURRENCY = {"unit": "currency"}
DIMENSIONLESS = {"unit": "dimensionless"}
YEARS = {"unit": "years"}

_ALLOWED_UNITS = frozenset({"currency", "dimensionless", "years"})


def _is_numeric_annotation(annotation) -> bool:
    """Whether a field annotation is int/float (incl. ``Optional`` of either).

    ``bool`` is deliberately excluded (flags carry no unit).  Structured
    annotations (lists, tuples, nested models) are not numeric scalars and are
    exempt from the unit requirement.
    """
    if annotation in (int, float):
        return True
    return any(arg in (int, float) for arg in get_args(annotation))


@lru_cache(maxsize=None)
def monetary_field_names(model_cls: type[BaseModel]) -> frozenset[str]:
    """Names of *model_cls* fields declared ``unit="currency"``.

    Fail-closed: raises ``TypeError`` for a numeric field that declares no
    unit, or declares one outside ``_ALLOWED_UNITS`` — a new tax-policy field
    cannot silently skip agent scaling (the failure mode this convention
    exists to prevent).  A numeric field added upstream to a shared model will
    trigger the same error, forcing a conscious unit classification at merge
    time.

    Args:
        model_cls: A pydantic model class carrying unit declarations.

    Returns:
        The frozen set of field names whose values are per-person dollars.
    """
    monetary: list[str] = []
    for name, field_info in model_cls.model_fields.items():
        extra = field_info.json_schema_extra
        unit = extra.get("unit") if isinstance(extra, dict) else None
        if unit == "currency":
            monetary.append(name)
        elif unit is None:
            if _is_numeric_annotation(field_info.annotation):
                raise TypeError(
                    f"Numeric field '{model_cls.__name__}.{name}' declares no "
                    f"unit. Tax-policy fields must declare "
                    f"json_schema_extra={{'unit': ...}} (one of "
                    f"{sorted(_ALLOWED_UNITS)}) so agent scaling cannot "
                    f"silently miss a dollar amount."
                )
        elif unit not in _ALLOWED_UNITS:
            raise TypeError(
                f"Field '{model_cls.__name__}.{name}' declares unknown unit "
                f"'{unit}'. Allowed: {sorted(_ALLOWED_UNITS)}."
            )
    return frozenset(monetary)


class TaxCreditDef(BaseModel):
    """A single non-refundable tax credit component with eligibility rules.

    Mirrors the ``TaxCreditComponent`` dataclass from the data layer.  Dollar
    amounts are per-person; every numeric field must declare its unit (see the
    unit-declaration block above) so ``country._scale_pit_policy`` converts
    all currency fields — including ones added later — to agent units.
    """

    kind: str = Field(description="Human-readable credit name (e.g. 'Age Amount').")
    amount: float = Field(
        default=0.0, ge=0.0, json_schema_extra=CURRENCY,
        description="Base dollar amount.",
    )
    indexing: bool = Field(default=True, description="Whether CPI-indexed.")
    eligibility_age_min: Optional[int] = Field(
        default=None, json_schema_extra=YEARS,
        description="Minimum age (e.g. 65 for Age Amount).",
    )
    clawback_start: Optional[float] = Field(
        default=None, ge=0.0, json_schema_extra=CURRENCY,
        description="Income at which phaseout begins (own income for Age Amount, spouse for Spousal).",
    )
    clawback_cap: Optional[float] = Field(
        default=None, ge=0.0, json_schema_extra=CURRENCY,
        description="Income at which credit is fully eliminated.",
    )


class SocialBenefits(BaseModel):
    name: Literal["ConstantSocialBenefitsSetter", "DefaultSocialBenefitsSetter", "GrowthSocialBenefitsSetter"] = (
        "GrowthSocialBenefitsSetter"
    )
    path_name: str = "social_benefits"
    parameters: dict = {}


class SocialHousing(BaseModel):
    name: Literal["DefaultSocialHousing"] = "DefaultSocialHousing"
    path_name: str = "social_housing"
    parameters: dict = {"rent_as_fraction_of_unemployment_rate": 0.25}


class CentralGovernmentFunctions(BaseModel):
    social_benefits: SocialBenefits = SocialBenefits()
    social_housing: SocialHousing = SocialHousing()


class CentralGovernmentConfiguration(BaseModel):
    functions: CentralGovernmentFunctions = CentralGovernmentFunctions()

    # Opt-in switch for this government's progressive PIT.  When True AND the
    # country carries taxation data (``SyntheticCountry.taxation``), the run
    # builds the progressive schedule (brackets, credits, dividend rates) onto
    # this config from that data; when False (default) the flat ``Income Tax``
    # scalar is used (upstream parity).  It is a *per-government* flag, so when
    # the model gains multiple government agents each opts in independently and
    # is matched to its own jurisdiction's schedules.
    activate_progressive_pit: bool = Field(
        default=False,
        description="Opt in to progressive PIT for this government, built from "
        "the country's taxation data. False keeps the flat Income Tax rate.",
    )

    # Progressive Personal Income Tax schedule.
    # Each tuple is (bracket_upper_bound, marginal_rate).
    # The last bound should be float("inf") for the top bracket.
    # When None (default), the flat ``Income Tax`` scalar is used for
    # both behavioural decisions and government revenue (backward
    # compatible).  When set, revenue is computed progressively on
    # employee income while wage-setting and after-tax income
    # calculations continue to use the scalar ``Income Tax`` effective
    # rate (which is updated each period to actual / taxable base).
    pit_brackets: Optional[list[tuple[float, float]]] = Field(
        default=None,
        description="Progressive PIT brackets as (upper_bound, marginal_rate). "
        "None means use the flat Income Tax rate.",
    )

    # Multi-component non-refundable tax credits with per-individual
    # eligibility rules.  Each component is a ``TaxCreditDef`` with its
    # own base amount, indexing flag, and eligibility conditions
    # (e.g. age ≥ 65 for Age Amount).
    #
    # At computation time, an individual's eligible credit bases are
    # summed and multiplied by the bottom bracket marginal rate.  The
    # resulting credit is subtracted from gross tax, floored at 0.
    #
    # When None (default), no post-bracket credits are applied.
    pit_tax_credits: Optional[list[TaxCreditDef]] = Field(
        default=None,
        description="List of non-refundable tax credits with eligibility rules. "
        "None means no credits applied.",
    )

    # Per-individual deduction(s) subtracted from the combined taxable
    # base (employee + rental + financial income) *before* the
    # progressive bracket calculation.  Unlike non-refundable tax
    # (a non-refundable credit), these deductions lower the bracket a
    # filer falls into and are therefore more powerful.
    #
    # Currently a single flat amount per individual.  Extensible to a
    # list of named deductions (e.g. age, employment, pension) when
    # individual-level attributes are needed.
    pit_taxable_income_deductions: Optional[float] = Field(
        default=None,
        ge=0.0,
        json_schema_extra=CURRENCY,
        description="Flat per-individual deduction from taxable income before brackets.",
    )


    # Fraction of a couple household's rental income assigned to the
    # higher-earning adult when distributing household-level rental
    # income to individuals for progressive PIT.  The lower earner
    # receives (1 - split).  Applies only to couple households
    # (Type 2 = couple, Type 4 = couple with children).
    # Non-couple households split rental income equally among adults.
    # Default 0.5 = equal 50/50 split.
    couple_rental_income_split: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        json_schema_extra=DIMENSIONLESS,
        description="Share of couple rental income to higher earner (0.5 = 50/50).",
    )

    # ── Dividend integration (Canadian gross-up + dividend tax credit) ──
    # When False (default), dividends keep the legacy at-source flat treatment
    # in income.py and never enter the PIT schedule (upstream parity).  When
    # True, both firm-investor and bank-investor dividends are grossed up and
    # added to taxable income (pool A), and the dividend tax credit is added as
    # a direct credit (the "2b" term) subtracted from gross PIT alongside the
    # base credits.
    #
    # The grossed-up amount is a tax fiction used only for the income-tax and
    # credit math; the actual dividend received by the household is unchanged.
    #
    # The field defaults below are the 2014 BC values, kept so a bare config is
    # self-consistent.  In a real run the gross-up and DTC rates are sourced from
    # the schedule CSV bc_dividend_tax_credit_schedule.csv in the taxation
    # directory (raw_data_path/"taxation", spoof_data/freda fallback; read by
    # DividendTaxCreditSchedule) and applied by
    # build_central_government_configuration; they are not YAML scalars.
    pit_dividend_integration: bool = Field(
        default=False,
        description="Enable Canadian dividend gross-up + dividend tax credit for firm and bank dividends.",
    )
    dividend_small_business_share: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        json_schema_extra=DIMENSIONLESS,
        description="Share s of firm dividends treated as other-than-eligible (small-business "
        "rate income); (1 - s) is eligible. Provisional uniform split per firm.",
    )
    bank_dividend_small_business_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        json_schema_extra=DIMENSIONLESS,
        description="Share s of bank dividends treated as other-than-eligible. Banks are taxed "
        "at the general corporate rate so all their dividends are eligible (s = 0).",
    )
    dividend_eligible_gross_up: float = Field(
        default=0.38,
        ge=0.0,
        json_schema_extra=DIMENSIONLESS,
        description="Gross-up rate for eligible dividends (0.38 => taxable = 1.38 x cash).",
    )
    dividend_non_eligible_gross_up: float = Field(
        default=0.18,
        ge=0.0,
        json_schema_extra=DIMENSIONLESS,
        description="Gross-up rate for other-than-eligible dividends (2014: 0.18 => 1.18 x cash).",
    )
    dividend_eligible_dtc_rate: float = Field(
        default=0.10,
        ge=0.0,
        json_schema_extra=DIMENSIONLESS,
        description="BC dividend tax credit on the grossed-up eligible dividend (2014: 0.10).",
    )
    dividend_non_eligible_dtc_rate: float = Field(
        default=0.0259,
        ge=0.0,
        json_schema_extra=DIMENSIONLESS,
        description="BC dividend tax credit on the grossed-up other-than-eligible dividend (2014: 0.0259).",
    )

