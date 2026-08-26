"""Unit tests for the housing market constructors.

The property columns were previously overwritten with the scalar -1, because
get_histogram.fillna(-1) binds -1 to the array argument and returns it unchanged.
"""

from typing import Any

import numpy as np
import pandas as pd

from macromodel.configurations import HousingMarketConfiguration
from macromodel.markets.housing_market.housing_market import HousingMarket

# Row 2 carries no inhabitant, which is the case the fill exists for. Rows 0, 3 and 5
# are owner-occupied, rows 1 and 4 are let to someone other than the owner.
PROPERTY_DATA = pd.DataFrame(
    {
        "House ID": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "Corresponding Inhabitant Household ID": [10.0, 11.0, np.nan, 12.0, 13.0, 11.0],
        "Corresponding Owner Household ID": [10.0, 20.0, 21.0, 12.0, 22.0, 11.0],
        "Is Owner-Occupied": [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        "Value": [3.0e5, 4.0e5, 2.5e5, 5.0e5, 3.5e5, 4.5e5],
        "Rent": [1.2e3, 1.5e3, 1.0e3, 1.8e3, 1.4e3, 1.6e3],
    }
)

FROM_DATA_CONFIG: dict[str, Any] = {
    "functions": {
        "clearing": {
            "name": {"value": "NoHousingMarketClearer"},
            "parameters": {"random_assignment_shock_variance": {"value": 0.0}},
        },
        "value": {
            "name": {"value": "DefaultPropertyValueSetter"},
            "parameters": {"random_fluctuation_std": {"value": 0.0}},
        },
    }
}


class _PickledMarket:
    """Stand-in for SyntheticHousingMarket, carrying the one attribute the constructor reads."""

    def __init__(self, housing_market_data: pd.DataFrame):
        self.housing_market_data = housing_market_data


def _pickled_market() -> HousingMarket:
    return HousingMarket.from_pickled_market(
        synthetic_housing_market=_PickledMarket(PROPERTY_DATA.copy()),
        housing_market_configuration=HousingMarketConfiguration(),
        scale=1,
        country_name="TEST",
    )


def _count(market: HousingMarket, series_name: str) -> int:
    """Read an occupancy count, which the time series holds as a one-element sequence."""
    return int(np.ravel(market.ts.current(series_name))[0])


class TestHousingMarketConstruction:
    def test_from_pickled_market_keeps_per_property_values(self):
        properties = _pickled_market().states["properties"]

        assert properties["House ID"].tolist() == [100, 101, 102, 103, 104, 105]
        assert properties["Corresponding Inhabitant Household ID"].tolist() == [10, 11, -1, 12, 13, 11]
        assert properties["Is Owner-Occupied"].tolist() == [1, 0, 0, 1, 0, 1]
        assert properties["Corresponding Owner Household ID"].tolist() == [10, 20, 21, 12, 22, 11]

    def test_occupancy_series_partition_the_stock(self):
        market = _pickled_market()

        rented = _count(market, "total_number_of_houses_rented")
        owner_occupied = _count(market, "total_number_of_houses_owner_occupied")
        unoccupied = _count(market, "total_number_of_houses_unoccupied")

        assert (rented, owner_occupied, unoccupied) == (2, 3, 1)
        assert rented + owner_occupied + unoccupied == len(PROPERTY_DATA)

    def test_rented_out_selection_reaches_properties(self):
        # The selection compute_rental_income applies to decide which properties earn rent.
        properties = _pickled_market().states["properties"]

        rented_out = properties.loc[
            np.logical_and(
                properties["Is Owner-Occupied"] == 0,
                properties["Corresponding Inhabitant Household ID"] != -1,
            )
        ]

        assert rented_out.index.tolist() == [1, 4]

    def test_from_data_fills_every_coerced_column(self):
        data = PROPERTY_DATA.copy()
        data.loc[2, "Corresponding Owner Household ID"] = np.nan

        states = HousingMarket.from_data(
            country_name="TEST",
            scale=1,
            data=data,
            config=FROM_DATA_CONFIG,
        ).states

        assert states["House ID"].tolist() == [100, 101, 102, 103, 104, 105]
        assert states["Corresponding Inhabitant Household ID"].tolist() == [10, 11, -1, 12, 13, 11]
        assert states["Is Owner-Occupied"].tolist() == [1, 0, 0, 1, 0, 1]
        assert states["Corresponding Owner Household ID"].tolist() == [10, 20, -1, 12, 22, 11]
