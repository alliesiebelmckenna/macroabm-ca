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
        states["pit_uppers"] = np.array([1.0e9])
        states["pit_rates"] = np.array([0.15])
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


class _Captured(Exception):
    """Raised by a spy once it has recorded the argument, to stop the method there."""


def _property_frame() -> pd.DataFrame:
    """The property frame in the shape ``country.py`` consumes it from the housing market.

    ``HousingMarket.from_data`` — what the shared fixture uses — leaves ``states`` a bare
    frame, while the pickled constructor the model runs on nests it under ``properties``.
    The call sites read the nested form, so that is the form to hand them.
    """
    frame = pd.DataFrame(
        {
            "House ID": np.array([0, 1, 2], dtype=int),
            "Value": np.array([100_000.0, 100_000.0, 100_000.0]),
            "Rent": np.array([RENT, RENT, RENT]),
            "Corresponding Owner Household ID": np.array([0, 1, 0], dtype=int),
            "Corresponding Inhabitant Household ID": np.array([2, 3, 0], dtype=int),
            "Is Owner-Occupied": np.array([0, 0, 1], dtype=int),
        }
    )
    frame.rename_axis("Properties", inplace=True)
    frame["Sale Price"] = frame["Value"]
    frame["Newly on the Rental Market"] = False
    frame["Up for Rent"] = False
    frame["Temporarily for Sale"] = False
    return frame


def _withholding_reaching(country, monkeypatch, method_name: str) -> float:
    """Run ``method_name`` and return the rate its call site actually passed on."""
    monkeypatch.setattr(
        country.housing_market,
        "states",
        {"properties": _property_frame(), "current_sales": pd.DataFrame()},
    )
    seen = {}

    def spy(housing_data, income_taxes):
        seen["income_taxes"] = income_taxes
        raise _Captured

    monkeypatch.setattr(country.households, "compute_rental_income", spy)
    with pytest.raises(_Captured):
        getattr(country, method_name)()
    return seen["income_taxes"]


class TestRentalWithholdingIsWiredIn:
    """The call sites, not the helpers.

    The tests above hand a rate to ``compute_rental_income`` themselves, so they hold
    whether or not ``country.py`` consults the gate at all: reverting both call sites to
    ``states["Income Tax"]`` leaves every one of them green. These drive the real methods
    and read back what the call site passed.
    """

    @pytest.mark.parametrize("method_name", ["update_planning_metrics", "update_realised_metrics"])
    def test_progressive_path_withholds_nothing_at_either_call_site(
        self, test_country, monkeypatch, method_name
    ):
        _set_progressive(test_country, True)

        assert _withholding_reaching(test_country, monkeypatch, method_name) == 0.0

    @pytest.mark.parametrize("method_name", ["update_planning_metrics", "update_realised_metrics"])
    def test_flat_path_still_withholds_at_either_call_site(self, test_country, monkeypatch, method_name):
        _set_progressive(test_country, False)
        income_tax = test_country.central_government.states["Income Tax"]

        assert _withholding_reaching(test_country, monkeypatch, method_name) == income_tax
