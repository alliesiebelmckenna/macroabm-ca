"""Tests for ``country._scale_pit_policy`` — agent-unit scaling of PIT policy dollars.

Statutory tax parameters are published in per-person dollars while agent incomes
are agent-level dollars (each synthetic agent represents ``scale`` people).
These tests pin the single scaling seam:

* brackets, credit currency fields (amount, clawback bounds), and the
  taxable-income deduction are multiplied by ``scale``; rates, ages, years, and
  dimensionless scalars are untouched;
* the conversion is reflection-driven off the ``unit`` field declarations, so a
  currency field added later is scaled automatically and an undeclared numeric
  field raises (fail-closed) instead of silently skipping the conversion;
* the credit pool is homogeneous of degree one in ``scale`` — the behavioural
  guarantee that clawbacks operate in agent units;
* the per-year schedule table applies the identical conversion to every year.
"""

import numpy as np
import pytest
from pydantic import Field

from macro_data.readers.taxation import TaxationReader
from macromodel.agents.central_government.central_government import (
    pit_credit_defs_to_state_dicts,
)
from macromodel.agents.central_government.pit_pools import (
    PitContext,
    build_credit_base_pool,
)
from macromodel.configurations import CentralGovernmentConfiguration, TaxCreditDef
from macromodel.configurations.central_government_configuration import (
    monetary_field_names,
)
from macromodel.country.country import (
    _build_pit_schedule_by_year,
    _scale_pit_policy,
    _scaled_tax_credit,
)

SCALE = 1000


def _pit_config(**overrides) -> CentralGovernmentConfiguration:
    base = dict(
        activate_progressive_pit=True,
        pit_brackets=[(37606.0, 0.0506), (75213.0, 0.0770), (float("inf"), 0.1050)],
        pit_tax_credits=[
            TaxCreditDef(kind="Personal Amount", amount=9869.0),
            TaxCreditDef(
                kind="Age Amount",
                amount=4426.0,
                eligibility_age_min=65,
                clawback_start=32943.0,
                clawback_cap=62450.0,
            ),
        ],
        pit_taxable_income_deductions=500.0,
    )
    base.update(overrides)
    return CentralGovernmentConfiguration(**base)


class TestScalePitPolicy:
    def test_bracket_thresholds_scaled_rates_untouched(self):
        scaled = _scale_pit_policy(_pit_config(), SCALE)
        thresholds = [t for t, _ in scaled.pit_brackets]
        rates = [r for _, r in scaled.pit_brackets]
        assert thresholds[:2] == [37606.0 * SCALE, 75213.0 * SCALE]
        assert np.isinf(thresholds[2])
        assert rates == [0.0506, 0.0770, 0.1050]

    def test_credit_currency_fields_scaled_age_untouched(self):
        scaled = _scale_pit_policy(_pit_config(), SCALE)
        personal, age = scaled.pit_tax_credits
        assert personal.amount == 9869.0 * SCALE
        assert age.amount == 4426.0 * SCALE
        assert age.clawback_start == 32943.0 * SCALE
        assert age.clawback_cap == 62450.0 * SCALE
        # Non-currency fields pass through unchanged.
        assert age.eligibility_age_min == 65
        assert personal.kind == "Personal Amount"
        assert personal.indexing is True

    def test_deduction_scaled(self):
        scaled = _scale_pit_policy(_pit_config(), SCALE)
        assert scaled.pit_taxable_income_deductions == 500.0 * SCALE

    def test_dimensionless_scalars_untouched(self):
        config = _pit_config(
            couple_rental_income_split=0.7,
            dividend_eligible_gross_up=0.38,
            dividend_small_business_share=0.9,
        )
        scaled = _scale_pit_policy(config, SCALE)
        assert scaled.couple_rental_income_split == 0.7
        assert scaled.dividend_eligible_gross_up == 0.38
        assert scaled.dividend_small_business_share == 0.9

    def test_scale_one_is_identity(self):
        config = _pit_config()
        assert _scale_pit_policy(config, 1) is config

    def test_caller_config_never_mutated(self):
        config = _pit_config()
        _scale_pit_policy(config, SCALE)
        assert config.pit_brackets[0][0] == 37606.0
        assert config.pit_tax_credits[0].amount == 9869.0
        assert config.pit_tax_credits[1].clawback_start == 32943.0
        assert config.pit_taxable_income_deductions == 500.0

    def test_flat_config_passes_through(self):
        flat = CentralGovernmentConfiguration()
        scaled = _scale_pit_policy(flat, SCALE)
        assert scaled.pit_brackets is None
        assert scaled.pit_tax_credits is None
        assert scaled.pit_taxable_income_deductions is None


class TestUnitDeclarationFailClosed:
    def test_undeclared_numeric_credit_field_raises(self):
        class BadCredit(TaxCreditDef):
            mystery_amount: float = 0.0

        with pytest.raises(TypeError, match="unit"):
            monetary_field_names(BadCredit)

    def test_undeclared_numeric_config_field_raises(self):
        class BadConfig(CentralGovernmentConfiguration):
            new_tax_scalar: float = 0.0

        with pytest.raises(TypeError, match="unit"):
            monetary_field_names(BadConfig)

    def test_unknown_unit_raises(self):
        class OddCredit(TaxCreditDef):
            weird: float = Field(default=1.0, json_schema_extra={"unit": "furlongs"})

        with pytest.raises(TypeError, match="furlongs"):
            monetary_field_names(OddCredit)

    def test_new_declared_currency_field_scales_automatically(self):
        """A future currency field needs only its declaration to be scaled."""

        class ExtendedCredit(TaxCreditDef):
            supplement: float = Field(
                default=250.0, json_schema_extra={"unit": "currency"}
            )

        scaled = _scaled_tax_credit(
            ExtendedCredit(kind="Future Credit", amount=100.0), SCALE
        )
        assert scaled.supplement == 250.0 * SCALE
        assert scaled.amount == 100.0 * SCALE

    def test_current_models_pass_the_declaration_check(self):
        """Every numeric field on the live models is unit-declared."""
        assert "amount" in monetary_field_names(TaxCreditDef)
        assert "pit_taxable_income_deductions" in monetary_field_names(
            CentralGovernmentConfiguration
        )


class TestScalingHomogeneity:
    def test_credit_pool_homogeneous_in_scale(self):
        """credit_pool(scale·income, scaled policy) == scale · credit_pool(income, policy).

        The first individual is a senior with income inside the Age Amount
        clawback band — under the pre-fix behaviour (unscaled policy against
        agent-scale income) the clawback would zero the credit entirely, so
        this pins that clawbacks operate in agent units.
        """
        config = _pit_config()
        scaled = _scale_pit_policy(config, SCALE)

        defs_per_person = pit_credit_defs_to_state_dicts(config.pit_tax_credits)
        defs_agent = pit_credit_defs_to_state_dicts(scaled.pit_tax_credits)

        incomes = np.array([40000.0, 20000.0])  # senior in-band, non-senior
        ages = np.array([70.0, 40.0])
        ctx = PitContext(
            employee_income=incomes,
            employee_si_rate=0.0,
            individuals_age=ages,
        )

        pool_per_person = build_credit_base_pool(defs_per_person, incomes, ctx)
        pool_agent = build_credit_base_pool(defs_agent, incomes * SCALE, ctx)

        # The senior's Age Amount is partially clawed back, not zeroed.
        assert pool_per_person[0] > 9869.0  # Personal + a surviving Age slice
        np.testing.assert_allclose(pool_agent, pool_per_person * SCALE)


# ── Per-year schedule table: identical conversion for every year ──────────

_PIT_HISTORICAL = """tax_year,geo,lower,rate,index
2014,BC,0,0.0506,1
2014,BC,37606,0.0770,1
2016,BC,0,0.0600,1
2016,BC,40000,0.0770,1
"""

_TAX_CREDITS = """tax_year,geo,credit,amount,top,rate,clawback,clawback_rate,index
2014,BC,Personal Amount,9869,,,,,1
2014,BC,Age Amount,4426,62450,,32943,,1
2016,BC,Personal Amount,10027,,,,,1
2016,BC,Age Amount,4497,63455,,33473,,1
"""


@pytest.fixture(name="credit_bearing_reader")
def _credit_bearing_reader(tmp_path):
    (tmp_path / "rates_thresholds.csv").write_text(_PIT_HISTORICAL)
    (tmp_path / "non_refundable_tax_credits.csv").write_text(_TAX_CREDITS)
    return TaxationReader.from_dir(tmp_path)


class TestScheduleTableScaling:
    def test_credits_scaled_for_every_published_year(self, credit_bearing_reader):
        config = CentralGovernmentConfiguration(activate_progressive_pit=True)
        unscaled = _build_pit_schedule_by_year(config, credit_bearing_reader, scale=1)
        scaled = _build_pit_schedule_by_year(
            config, credit_bearing_reader, scale=SCALE
        )

        for year in (2014, 2016):
            for base, converted in zip(
                unscaled[year]["pit_tax_credits"], scaled[year]["pit_tax_credits"]
            ):
                assert converted["amount"] == base["amount"] * SCALE
                for bound in ("clawback_start", "clawback_cap"):
                    if base[bound] is not None:
                        assert converted[bound] == base[bound] * SCALE
                # Non-currency fields identical.
                assert converted["kind"] == base["kind"]
                assert converted["age_min"] == base["age_min"]
                assert converted["indexing"] == base["indexing"]

    def test_projected_years_inherit_agent_units(self, credit_bearing_reader):
        """Projection compounds inflation on top of already-scaled values, so
        projected years stay homogeneous in ``scale`` too."""
        config = CentralGovernmentConfiguration(activate_progressive_pit=True)
        unscaled = _build_pit_schedule_by_year(config, credit_bearing_reader, scale=1)
        scaled = _build_pit_schedule_by_year(
            config, credit_bearing_reader, scale=SCALE
        )

        projected_years = sorted(set(scaled) - {2014, 2016})
        assert projected_years, "projection must extend past the published years"
        year = projected_years[0]
        for base, converted in zip(
            unscaled[year]["pit_tax_credits"], scaled[year]["pit_tax_credits"]
        ):
            assert converted["amount"] == pytest.approx(base["amount"] * SCALE)
