import dataclasses
import types
from pathlib import Path

import numpy as np
import pytest

from macro_data.readers.taxation import TaxationReader
from macromodel.configurations import (
    CentralGovernmentConfiguration,
    CountryConfiguration,
    ExchangeRatesConfiguration,
)
from macromodel.country import Country
from macromodel.exchange_rates import ExchangeRates
from macromodel.utils.prehooks import create_pit_schedule_update_hook

# Committed BC schedules (test fixtures).
#   parents[0]=test_country [1]=unit [2]=test_macromodel [3]=tests [4]=repo root
_COMMITTED_PIT_DIR = (
    Path(__file__).resolve().parents[4]
    / "spoof_data" / "freda" / "personal_income_tax"
)


class TestCountry:
    def test__init(self, datawrapper):
        synthetic_country = datawrapper.synthetic_countries["FRA"]
        country_configuration = CountryConfiguration()

        exchange_rates_config = ExchangeRatesConfiguration()
        exchange_rates_df = datawrapper.exchange_rates
        initial_year = 2014
        country_names = ["FRA"]

        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=exchange_rates_df,
            exchange_rate_config=exchange_rates_config,
            initial_year=initial_year,
            country_names=country_names,
        )

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )

        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        assert country is not None

    def test__country(self, test_country):
        assert test_country is not None

    def test_pit_bracket_scaling_does_not_mutate_config(self, datawrapper):
        """Building a Country must not scale the caller's pit_brackets in
        place: repeated construction from the same config stays stable."""
        synthetic_country = datawrapper.synthetic_countries["FRA"]
        scale = synthetic_country.scale
        assert scale > 1, "test data must exercise the bracket-scaling path"

        country_configuration = CountryConfiguration(
            central_government=CentralGovernmentConfiguration(
                pit_brackets=[(50000.0, 0.10), (float("inf"), 0.20)],
            ),
        )
        original = list(country_configuration.central_government.pit_brackets)

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )

        def build():
            exchange_rates = ExchangeRates.from_data(
                exchange_rates_data=datawrapper.exchange_rates,
                exchange_rate_config=ExchangeRatesConfiguration(),
                initial_year=2014,
                country_names=["FRA"],
            )
            return Country.from_pickled_country(
                synthetic_country=synthetic_country,
                country_configuration=country_configuration,
                exchange_rates=exchange_rates,
                country_name="FRA",
                all_country_names=["FRA", "ROW"],
                industries=datawrapper.industries,
                initial_year=datawrapper.configuration.year,
                t_max=12,
                running_multiple_countries=False,
                emission_factors_usd=emission_factors,
            )

        build()
        country = build()  # second construction from the same config object

        # Caller's config is untouched after repeated construction.
        assert list(country_configuration.central_government.pit_brackets) == original
        # Scale was applied exactly once (not compounded across builds).
        assert country.central_government.states["pit_uppers"][0] == 50000.0 * scale

    def test_taxation_data_activates_progressive_pit_end_to_end(self, datawrapper):
        """End-to-end: a country carrying taxation data, whose government opts in,
        gets the progressive BC schedule on its central-government agent. This
        exercises the full wired stream: SyntheticCountry.taxation ->
        from_pickled_country -> activate_taxation -> build -> from_pickled_agent."""
        base = datawrapper.synthetic_countries["FRA"]
        scale = base.scale
        # Attach BC taxation data to a *copy* of the synthetic country (leave the
        # shared fixture untouched).
        synthetic_country = dataclasses.replace(
            base, taxation=TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction="bc")
        )

        country_configuration = CountryConfiguration(
            central_government=CentralGovernmentConfiguration(
                activate_progressive_pit=True
            ),
        )

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )
        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=datawrapper.exchange_rates,
            exchange_rate_config=ExchangeRatesConfiguration(),
            initial_year=2014,
            country_names=["FRA"],
        )
        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        # Progressive PIT is active: the agent carries the BC schedule, with the
        # first bracket upper scaled to agent units (37,606 per individual).
        states = country.central_government.states
        assert "pit_uppers" in states
        assert states["pit_uppers"][0] == 37606.0 * scale
        assert states["pit_rates"][0] == 0.0506
        # Dividend integration switched on with the schedule present.
        assert states["pit_dividend_integration"] is True

    def test_no_optin_keeps_flat_even_with_taxation_data(self, datawrapper):
        """Taxation data present but the government did not opt in ⇒ no
        progressive PIT (flat parity preserved)."""
        base = datawrapper.synthetic_countries["FRA"]
        synthetic_country = dataclasses.replace(
            base, taxation=TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction="bc")
        )
        country_configuration = CountryConfiguration()  # activate_progressive_pit=False

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )
        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=datawrapper.exchange_rates,
            exchange_rate_config=ExchangeRatesConfiguration(),
            initial_year=2014,
            country_names=["FRA"],
        )
        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )
        assert "pit_uppers" not in country.central_government.states

    def test_pit_schedule_update_advances_brackets_end_to_end(self, datawrapper, tmp_path):
        """Integration of the whole indexing chain on a real ``Country``.

        Injects a *multi-year* taxation schedule (a 2016 bottom-rate change), so
        ``Country.from_pickled_country`` builds the per-year schedule table onto
        the real central-government agent.  The real ``pit_schedule_update`` pre-hook is
        then driven across calendar years and the agent's live ``pit_uppers``
        / ``pit_rates`` are checked to advance (published) and to reject a year
        beyond the last published one — exercising builder → agent states →
        prehook → ``set_pit_for_year`` together, not in isolation.
        """
        # Two published years; 2016 changes the bottom rate.
        (tmp_path / "rates_thresholds.csv").write_text(
            "year,jurisdiction,lower,rate,index\n"
            "2014,BC,0,0.0506,1\n2014,BC,37606,0.0770,1\n2014,BC,75213,0.1050,1\n"
            "2016,BC,0,0.0600,1\n2016,BC,40000,0.0770,1\n2016,BC,80000,0.1050,1\n"
        )
        reader = TaxationReader.from_dir(tmp_path, jurisdiction="bc")

        base = datawrapper.synthetic_countries["FRA"]
        scale = base.scale
        synthetic_country = dataclasses.replace(base, taxation=reader)
        country_configuration = CountryConfiguration(
            central_government=CentralGovernmentConfiguration(
                activate_progressive_pit=True
            ),
        )

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )
        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=datawrapper.exchange_rates,
            exchange_rate_config=ExchangeRatesConfiguration(),
            initial_year=2014,
            country_names=["FRA"],
        )
        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        cg = country.central_government
        # The construction built the per-year table onto the real agent.
        assert "pit_schedule_by_year" in cg.states

        # Drive the REAL pre-hook (not a stub) over a minimal simulation holder;
        # the country and agent are the real constructed objects.
        hook = create_pit_schedule_update_hook()
        simulation = types.SimpleNamespace(countries={"FRA": country})

        # 2014 (construction year): the published base brackets, scaled.
        hook(simulation, 2014, 1)
        uppers_2014 = cg.states["pit_uppers"].copy()
        assert uppers_2014[0] == pytest.approx(37606.0 * scale)
        assert cg.states["pit_rates"][0] == pytest.approx(0.0506)

        # 2016: brackets and the bottom marginal rate advance to the published
        # 2016 values (a rate change CPI compounding could not express).
        hook(simulation, 2016, 1)
        uppers_2016 = cg.states["pit_uppers"].copy()
        assert uppers_2016[0] == pytest.approx(40000.0 * scale)
        assert cg.states["pit_rates"][0] == pytest.approx(0.0600)

        # A year beyond the last published one (2016) is not projectable — the
        # schedule is a statutory lookup only, so the hook must raise rather
        # than silently hold the 2016 values flat.
        with pytest.raises(ValueError, match="exceeds the last available"):
            hook(simulation, 2017, 1)
        # The agent's live state is untouched by the rejected lookup.
        np.testing.assert_array_equal(cg.states["pit_uppers"], uppers_2016)
        assert cg.states["pit_rates"][0] == pytest.approx(0.0600)
