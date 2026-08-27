"""Seam test: the GDP rent-received term is assembled through the gross-up gate.

``country.py`` builds the income leg's rent term as ``total_real_rent_rec`` plus
``_rent_received_gross_up()``. The unit tests around that helper pin what it returns; they
say nothing about whether the assembly still consults it, so a call site that went back to
adding ``taxes_rental_income`` unconditionally would leave them green.

The arm exercised here is the FLAT one -- the committed CAN fixture carries no PIT
schedule, so ``progressive_pit_active`` is False and the withholding still fires. On this
path the tax line must still be added back, because there the haircut is the tax and the
received figure is net.
"""

import pytest

from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation


def _can_simulation(datawrapper) -> Simulation:
    industries = datawrapper.synthetic_countries["CAN"].firms.firm_data["Industry"].unique()
    configuration = SimulationConfiguration(
        country_configurations={"CAN": CountryConfiguration.n_industry_default(n_industries=len(industries))}
    )
    configuration.seed = 0
    return Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)


def test_flat_path_gdp_rent_received_is_grossed_up(can_disagg_datawrapper):
    simulation = _can_simulation(can_disagg_datawrapper)
    country = simulation.countries["CAN"]
    seen = {}
    compute_gdp = country.economy.compute_gdp

    def spy(*args, **kwargs):
        # Read the two components at the call site, so no later append can shift them.
        seen["rent_received"] = kwargs["rent_received"]
        seen["received"] = country.economy.ts.current("total_real_rent_rec")[0]
        seen["tax"] = country.central_government.ts.current("taxes_rental_income")[0]
        return compute_gdp(*args, **kwargs)

    country.economy.compute_gdp = spy
    simulation.iterate()

    assert country.central_government.progressive_pit_active is False
    # Without a positive tax line the assertion below would hold whether or not the
    # assembly consults the gate, so the arm has to be shown non-degenerate first.
    assert seen["tax"] > 0
    assert seen["rent_received"] == pytest.approx(seen["received"] + seen["tax"])

