"""The refundable credit reaches households by eligibility, not by allocation.

The credit rides the transfer machinery's WIRING -- that is how government money
reaches a household -- but never its ALLOCATION, which distributes a fixed
budget by a regression's fitted share. Routed through the allocation the
aggregate would still look right while the money went to the wrong households.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.country.country import Country


def _country(paid_per_ind, corr, n_households):
    """A stand-in carrying only what the aggregation reads."""
    return SimpleNamespace(
        households=SimpleNamespace(ts=SimpleNamespace(current=lambda _k: n_households)),
        individuals=SimpleNamespace(states={"Corresponding Household ID": np.asarray(corr)}),
        central_government=SimpleNamespace(states={"pit_rtc_paid_per_ind": paid_per_ind}),
    )


class TestAggregationToHouseholds:
    def test_each_households_credit_is_the_sum_of_its_members(self):
        # The credit is computed per individual because eligibility is
        # individual; households are what the transfer machinery pays.
        c = _country(np.array([882.0, 0.0, 0.0, 400.0]), [0, 0, 0, 1], 2)
        out = Country._rtc_per_household(c)
        assert out == pytest.approx([882.0, 400.0])
