"""Tests for the rental withholding gate.

Canadian rental income has no tax withheld at the point of transaction; it is assessed at the
year-end filing, where the PIT pool already taxes it. The pre-existing flat haircut was never
suppressed when the progressive path was added, so rent was taxed twice. The flat path must be
left exactly as it was, where the haircut is the tax.
"""

import numpy as np
import pandas as pd
import pytest

RENT = 1_000.0
N_RENTED = 2
GROSS_RENT = RENT * N_RENTED


def _rented_out_properties() -> pd.DataFrame:
    """Two properties let to someone other than their owner, plus one owner-occupied."""
    return pd.DataFrame(
        {
            "Corresponding Owner Household ID": np.array([0, 1, 0], dtype=int),
            "Corresponding Inhabitant Household ID": np.array([2, 3, 0], dtype=int),
            "Is Owner-Occupied": np.array([0, 0, 1], dtype=int),
            "Rent": np.array([RENT, RENT, RENT]),
        }
    )


def _set_progressive(country, active: bool) -> None:
    """Toggle the one predicate the whole gate reads."""
    states = country.central_government.states
    if active:
        states["pit_uppers"] = np.array([[1.0e9]])
        states["pit_rates"] = np.array([[0.15]])
    else:
        states.pop("pit_uppers", None)
        states.pop("pit_rates", None)
    assert country.central_government.progressive_pit_active is active


class TestRentalWithholding:
    def test_progressive_path_pays_rent_gross(self, test_country):
        _set_progressive(test_country, True)

        rate = test_country._rental_withholding_rate()
        received = test_country.households.compute_rental_income(
            housing_data=_rented_out_properties(),
            income_taxes=rate,
        )

        assert rate == 0.0
        assert received.sum() == pytest.approx(GROSS_RENT)

    def test_flat_path_still_withholds(self, test_country):
        # The path the experiment never exercised: here the haircut IS the tax and must stay.
        _set_progressive(test_country, False)
        income_tax = test_country.central_government.states["Income Tax"]

        rate = test_country._rental_withholding_rate()
        received = test_country.households.compute_rental_income(
            housing_data=_rented_out_properties(),
            income_taxes=rate,
        )

        assert rate == income_tax
        assert received.sum() == pytest.approx(GROSS_RENT * (1 - income_tax))

    def test_progressive_path_does_not_gross_up_rent_received(self, test_country):
        _set_progressive(test_country, True)

        assert test_country._rent_received_gross_up() == 0.0

    def test_flat_path_still_grosses_up_rent_received(self, test_country):
        _set_progressive(test_country, False)

        booked = test_country.central_government.ts.current("taxes_rental_income")[0]
        assert test_country._rent_received_gross_up() == booked
