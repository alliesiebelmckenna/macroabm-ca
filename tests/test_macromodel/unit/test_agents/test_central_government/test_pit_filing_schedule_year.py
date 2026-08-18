"""The year-end filing assesses a tax year under that year's own schedule.

The schedule pre-hook advances the live states to the new calendar year before
the filing runs, so a filing that read those states would settle the closing
year at the incoming year's rates.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.central_government.central_government import CentralGovernment

UPPERS = np.array([50000.0, np.inf])
SETTLED_RATES = np.array([0.10, 0.20])
RAISED_RATES = np.array([0.30, 0.40])
F = 4.0


def _gov() -> SimpleNamespace:
    """A stand-in whose schedule rises between the settled year and the next."""
    gov = SimpleNamespace(
        states={
            "pit_uppers": UPPERS,
            "pit_rates": SETTLED_RATES,
            "pit_year_end_reconciliation": True,
            "pit_calendar_year": 2014,
            "pit_schedule_by_year": {
                2014: {"pit_uppers": UPPERS, "pit_rates": SETTLED_RATES},
                2015: {
                    "pit_uppers": UPPERS,
                    "pit_rates": RAISED_RATES,
                    "pit_non_refundable_tax_credits": [{"credit": "new", "amount": 1.0}],
                },
            },
        }
    )
    gov._pit_schedule_for_year = lambda year: CentralGovernment._pit_schedule_for_year(
        gov, year
    )
    return gov


def _advance(gov):
    """Advance the government's shared period counter.

    The runtime does this once per period in ``compute_taxes``. These tests
    drive ``_reconcile_tax_year`` directly, so they own the advance -- which is
    the point of the counter being government-owned rather than PIT-private.
    """
    gov.states["tax_step"] = int(gov.states.get("tax_step", 0)) + 1


def _step(gov, income_per_ind):
    """One period: assess, then settle any year that closed.

    Returns the settlement alone, which is the observable the filing tests
    need; it is zero at every period that is not a filing.
    """
    pool = income_per_ind * F
    out: list = []
    CentralGovernment.compute_pit(
        gov, pool, None, None, steps_per_year=F, out_tax_per_ind=out
    )
    # Per-individual settlement; these cases assert on the settled TOTAL.
    return float(
        np.sum(
            CentralGovernment._reconcile_tax_year(
                gov, pool, out[0], None, None, steps_per_year=F
            )
        )
    )


class TestFilingUsesTheSettledYearsSchedule:
    def test_a_rate_rise_does_not_reopen_the_closed_year(self):
        income = np.array([10000.0, 10000.0])
        gov = _gov()
        for _ in range(4):
            _step(gov, income)

        # The pre-hook advances to the new year before the filing period runs.
        gov.states["pit_calendar_year"] = 2015
        gov.states["pit_rates"] = RAISED_RATES
        settlement = _step(gov, income)

        # Income was level, so the year was withheld exactly: nothing settles.
        # Assessed at 2015's rates the closed year would owe, and the settlement
        # is taken at the filing itself, so it would show up here.
        assert settlement == pytest.approx(0.0)

    def test_the_settled_years_rate_values_the_credits(self):
        gov = _gov()
        seen: list = []

        def annual_credit_base(income_per_ind, year_credit_defs=None):
            seen.append(year_credit_defs)
            return np.zeros_like(income_per_ind)

        income = np.array([10000.0, 10000.0])
        for _ in range(4):
            _advance(gov)
            pool = income * F
            out: list = []
            CentralGovernment.compute_pit(
                gov, pool, None, None, steps_per_year=F, out_tax_per_ind=out
            )
            CentralGovernment._reconcile_tax_year(
                gov,
                pool,
                out[0],
                None,
                None,
                steps_per_year=F,
                annual_credit_base=annual_credit_base,
            )
        gov.states["pit_calendar_year"] = 2015
        gov.states["pit_rates"] = RAISED_RATES
        _advance(gov)
        pool = income * F
        out = []
        CentralGovernment.compute_pit(
            gov, pool, None, None, steps_per_year=F, out_tax_per_ind=out
        )
        CentralGovernment._reconcile_tax_year(
            gov,
            pool,
            out[0],
            None,
            None,
            steps_per_year=F,
            annual_credit_base=annual_credit_base,
        )

        # The filing asked for the settled year's credits, not the new year's.
        # 2014 publishes none, so the fragment carries no credit definitions.
        assert seen and seen[-1] is None
