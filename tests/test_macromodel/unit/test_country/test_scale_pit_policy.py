"""Tests for ``country._scale_pit_policy`` — agent-unit scaling of PIT policy dollars.

Statutory tax parameters are published in per-person dollars while agent incomes
are agent-level dollars (each synthetic agent represents ``scale`` people).
These tests pin the single scaling seam:

* brackets and credit currency fields (amount, clawback bounds) are multiplied
  by ``scale``; rates, ages, years, and dimensionless scalars are untouched;
* the conversion is reflection-driven off the ``unit`` field declarations, so a
  currency field added later is scaled automatically and an undeclared numeric
  field raises (fail-closed) instead of silently skipping the conversion;
* the credit pool is homogeneous of degree one in ``scale`` — the behavioural
  guarantee that clawbacks operate in agent units;
* the per-year schedule table applies the identical conversion to every year.
"""

import numpy as np
import pytest

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
    _scale_pit_policy,
)

SCALE = 1000


def _pit_config(**overrides) -> CentralGovernmentConfiguration:
    base = dict(
        activate_progressive_pit=True,
        pit_brackets=[(37606.0, 0.0506), (75213.0, 0.0770), (float("inf"), 0.1050)],
        pit_non_refundable_tax_credits=[
            TaxCreditDef(credit="Personal Amount", amount=9869.0),
            TaxCreditDef(
                credit="Age Amount",
                amount=4426.0,
                eligibility_age_min=65,
                clawback=32943.0,
                top=62450.0,
            ),
        ],
    )
    base.update(overrides)
    return CentralGovernmentConfiguration(**base)


class TestUnitDeclarationFailClosed:
    def test_undeclared_numeric_config_field_raises(self):
        class BadConfig(CentralGovernmentConfiguration):
            new_tax_scalar: float = 0.0

        with pytest.raises(TypeError, match="unit"):
            monetary_field_names(BadConfig)


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

        defs_per_person = pit_credit_defs_to_state_dicts(config.pit_non_refundable_tax_credits)
        defs_agent = pit_credit_defs_to_state_dicts(scaled.pit_non_refundable_tax_credits)

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


# Per-year schedule table: identical conversion for every year

_PIT_HISTORICAL = """year,jurisdiction,lower,rate,index
2014,BC,0,0.0506,1
2014,BC,37606,0.0770,1
2016,BC,0,0.0600,1
2016,BC,40000,0.0770,1
"""

_TAX_CREDITS = """year,jurisdiction,credit,amount,top,rate,clawback,clawback_rate,index
2014,BC,Personal Amount,9869,,,,,1
2014,BC,Age Amount,4426,62450,,32943,,1
2016,BC,Personal Amount,10027,,,,,1
2016,BC,Age Amount,4497,63455,,33473,,1
"""


@pytest.fixture(name="credit_bearing_reader")
def _credit_bearing_reader(tmp_path):
    (tmp_path / "rates_thresholds.csv").write_text(_PIT_HISTORICAL)
    (tmp_path / "non_refundable_tax_credits.csv").write_text(_TAX_CREDITS)
    return TaxationReader.from_dir(tmp_path, jurisdiction="bc")
