from functools import lru_cache
from typing import Literal, Optional, get_args

from pydantic import BaseModel, Field

# Unit declarations for tax-policy fields, read by ``country._scale_pit_policy``
# to scale per-person statutory dollars to agent units. Only "currency" is
# scaled; "dimensionless" and "years" are not. ``monetary_field_names`` enforces
# this fail-closed, so a new numeric field must declare its unit.
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

    Fail-closed: raises ``TypeError`` for a numeric field with no unit, or one
    outside ``_ALLOWED_UNITS``, so a field cannot silently skip agent scaling.
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

    Mirrors the ``TaxCreditComponent`` dataclass from the data layer. Dollar
    amounts are per-person; every numeric field declares its unit so
    ``country._scale_pit_policy`` can convert currency fields to agent units.
    """

    credit: str = Field(description="Human-readable credit name (e.g. 'Age Amount').")
    amount: float = Field(
        default=0.0, ge=0.0, json_schema_extra=CURRENCY,
        description="Base dollar amount.",
    )
    index: bool = Field(default=True, description="Whether statutorily indexed.")
    eligibility_age_min: Optional[int] = Field(
        default=None, json_schema_extra=YEARS,
        description="Minimum age (e.g. 65 for Age Amount).",
    )
    clawback: Optional[float] = Field(
        default=None, ge=0.0, json_schema_extra=CURRENCY,
        description="Income at which phaseout begins (own income for Age Amount, spouse for Spousal).",
    )
    top: Optional[float] = Field(
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

    # Per-government flag: each government opts in independently.
    activate_progressive_pit: bool = Field(
        default=False,
        description="Opt in to progressive PIT for this government, built from "
        "the country's taxation data. False keeps the flat Income Tax rate.",
    )

    # When set, revenue is progressive but wage-setting and after-tax income keep
    # using the scalar Income Tax rate (refreshed each period to actual / base).
    pit_brackets: Optional[list[tuple[float, float]]] = Field(
        default=None,
        description="Progressive PIT brackets as (upper_bound, rate). "
        "None means use the flat Income Tax rate.",
    )

    pit_non_refundable_tax_credits: Optional[list[TaxCreditDef]] = Field(
        default=None,
        description="List of non-refundable tax credits with eligibility rules. "
        "None means no credits applied.",
    )

    pit_taxable_income_deductions: Optional[float] = Field(
        default=None,
        ge=0.0,
        json_schema_extra=CURRENCY,
        description="Flat per-individual deduction from taxable income before brackets.",
    )

    couple_rental_income_split: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        json_schema_extra=DIMENSIONLESS,
        description="Share of couple rental income to higher earner (0.5 = 50/50).",
    )

    # Dividend integration (off for parity). The defaults below are 2014 BC
    # values; a real run sources the rates from the dividend schedule CSV via
    # build_central_government_configuration, not the YAML.
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

