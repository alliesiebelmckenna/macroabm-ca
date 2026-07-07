from typing import Literal, Optional

from pydantic import BaseModel, Field


class TaxCreditDef(BaseModel):
    """A single non-refundable tax credit component with eligibility rules.

    Mirrors the ``TaxCreditComponent`` dataclass from the data layer.
    """

    kind: str = Field(description="Human-readable credit name (e.g. 'Age Amount').")
    amount: float = Field(default=0.0, ge=0.0, description="Base dollar amount.")
    indexing: bool = Field(default=True, description="Whether CPI-indexed.")
    eligibility_age_min: Optional[int] = Field(
        default=None, description="Minimum age (e.g. 65 for Age Amount)."
    )
    clawback_start: Optional[float] = Field(
        default=None, ge=0.0,
        description="Income at which phaseout begins (own income for Age Amount, spouse for Spousal).",
    )
    clawback_cap: Optional[float] = Field(
        default=None, ge=0.0,
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
        description="Share s of firm dividends treated as other-than-eligible (small-business "
        "rate income); (1 - s) is eligible. Provisional uniform split per firm.",
    )
    bank_dividend_small_business_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share s of bank dividends treated as other-than-eligible. Banks are taxed "
        "at the general corporate rate so all their dividends are eligible (s = 0).",
    )
    dividend_eligible_gross_up: float = Field(
        default=0.38,
        ge=0.0,
        description="Gross-up rate for eligible dividends (0.38 => taxable = 1.38 x cash).",
    )
    dividend_non_eligible_gross_up: float = Field(
        default=0.18,
        ge=0.0,
        description="Gross-up rate for other-than-eligible dividends (2014: 0.18 => 1.18 x cash).",
    )
    dividend_eligible_dtc_rate: float = Field(
        default=0.10,
        ge=0.0,
        description="BC dividend tax credit on the grossed-up eligible dividend (2014: 0.10).",
    )
    dividend_non_eligible_dtc_rate: float = Field(
        default=0.0259,
        ge=0.0,
        description="BC dividend tax credit on the grossed-up other-than-eligible dividend (2014: 0.0259).",
    )

