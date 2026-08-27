"""Every stream in the taxable pool must be scaled by the annualization.

The scale-up is PER STREAM and the scale-down divides the WHOLE pooled array by
the same factor, so a stream that sits in the pool but in no scaling site is
divided and never multiplied. Nothing raises today; the tax is simply a quarter
of what it should be for that stream.
"""

import numpy as np
import pytest

from macromodel.agents.central_government.func.pit_pools import (
    PitContext,
    assert_pooled_streams_are_scaled,
    build_withheld_income_pool,
)

N = 3


def _ctx(**streams):
    base = dict(
        employee_income=np.full(N, 40000.0),
        employee_si_rate=0.05,
    )
    base.update(streams)
    return PitContext(**base)


class TestPooledStreamsAreScaled:
    def test_refuses_a_pooled_stream_dropped_from_the_scaling_set(self):
        # Exactly the change section 4.3 recommends. Dropping the stream from the
        # scaling set WITHOUT dropping it from the pool understates it fourfold.
        ctx = _ctx(rental_income=np.full(N, 5000.0))
        narrowed = frozenset({"employee_income", "financial_income"})
        with pytest.raises(ValueError, match="rental_income"):
            assert_pooled_streams_are_scaled(ctx, streams=narrowed)


class TestWithheldPool:
    def test_is_employment_only(self):
        # Rental and financial income are assessed at the filing, never withheld against. The
        # employment figure is the pay as given: the wage setter has already withheld the
        # social-insurance levy, so the pool does not deduct it a second time.
        ctx = _ctx(rental_income=np.full(N, 5000.0), financial_income=np.full(N, 1000.0))
        withheld = build_withheld_income_pool(ctx)
        assert withheld == pytest.approx(np.full(N, 40000.0))
