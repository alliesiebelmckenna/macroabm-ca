"""Unit tests for rental-vacancy detection in prepare_housing_market_clearing.

Vacancy was tested with np.isnan against a column that is int64 and uses -1 for vacant,
so no property was ever marked available to rent, and the previous step was compared by
value against integer indices.
"""

import numpy as np
import pandas as pd

VACANT = -1


def _properties(inhabitant_ids: list[int]) -> pd.DataFrame:
    """A properties frame carrying the columns the clearing preparation reads."""
    n = len(inhabitant_ids)
    return pd.DataFrame(
        {
            "Corresponding Inhabitant Household ID": np.array(inhabitant_ids, dtype=int),
            "Corresponding Owner Household ID": np.arange(n, dtype=int),
            "Is Owner-Occupied": np.zeros(n, dtype=int),
            "Value": np.full(n, 3.0e5),
            "Rent": np.full(n, 1.2e3),
            "Sale Price": np.full(n, 3.0e5),
            "Temporarily for Sale": np.zeros(n, dtype=bool),
            "Up for Rent": None,
            "Newly on the Rental Market": False,
        }
    )


def _prepare(households, housing_data: pd.DataFrame) -> None:
    households.prepare_housing_market_clearing(
        housing_data=housing_data,
        observed_fraction_value_price=np.array([0.0, 1.0]),
        observed_fraction_rent_value=np.array([0.0, 0.004]),
        expected_hpi_growth=0.0,
        assumed_mortgage_maturity=100,
        rental_income_taxes=0.0,
    )


class TestRentalVacancyDetection:
    def test_vacant_properties_are_offered_for_rent(self, test_households):
        housing_data = _properties([VACANT, 1, VACANT, 2])

        _prepare(test_households, housing_data)

        assert housing_data["Up for Rent"].tolist() == [True, False, True, False]

    def test_a_still_vacant_property_is_not_newly_on_the_market_twice(self, test_households):
        housing_data = _properties([VACANT, 1, VACANT, 2])

        _prepare(test_households, housing_data)
        first_pass = housing_data["Newly on the Rental Market"].tolist()
        _prepare(test_households, housing_data)
        second_pass = housing_data["Newly on the Rental Market"].tolist()

        assert first_pass == [True, False, True, False]
        assert second_pass == [False, False, False, False]
        assert housing_data["Up for Rent"].tolist() == [True, False, True, False]

    def test_a_newly_vacated_property_joins_the_market(self, test_households):
        housing_data = _properties([VACANT, 1, VACANT, 2])

        _prepare(test_households, housing_data)
        housing_data.loc[1, "Corresponding Inhabitant Household ID"] = VACANT
        _prepare(test_households, housing_data)

        assert housing_data["Up for Rent"].tolist() == [True, True, True, False]
        assert housing_data["Newly on the Rental Market"].tolist() == [False, True, False, False]
