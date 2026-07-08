"""Tests for ``_build_pit_schedule_by_year`` — the per-year PIT schedule table.

The table is assembled in ``country.py`` from a multi-year taxation schedule and
stashed on the central-government agent; the ``pit_indexing`` pre-hook then reads
it to advance brackets/credits as the calendar year progresses.  These tests
drive the builder with a small two-year reader (a rate change between years) and
pin: opt-in / data-presence gating, one entry per published year, the per-year
statutory values, and the agent-unit scaling applied identically to every year.
There is no forward projection — a year past the last published one is the
caller's error (see ``test_country.py`` for the ``set_pit_for_year`` overflow
behaviour).
"""

import numpy as np
import pytest

from macro_data.readers.taxation import TaxationReader
from macromodel.configurations import CentralGovernmentConfiguration
from macromodel.country.country import _build_pit_schedule_by_year

# A two-year bracket schedule: 2016 changes the bottom marginal rate (0.0506 →
# 0.0600), which a CPI inflation of the base year could not express.
_PIT_HISTORICAL = """tax_year,geo,lower,rate,index
2014,BC,0,0.0506,1
2014,BC,37606,0.0770,1
2014,BC,75213,0.1050,0
2016,BC,0,0.0600,1
2016,BC,40000,0.0770,1
2016,BC,80000,0.1050,0
"""


@pytest.fixture(name="multi_year_reader")
def _multi_year_reader(tmp_path):
    (tmp_path / "rates_thresholds.csv").write_text(_PIT_HISTORICAL)
    return TaxationReader.from_dir(tmp_path)


class TestBuildPitScheduleByYear:
    def test_none_reader_returns_none(self):
        config = CentralGovernmentConfiguration(activate_progressive_pit=True)
        assert _build_pit_schedule_by_year(config, None, scale=1) is None

    def test_not_opted_in_returns_none(self, multi_year_reader):
        # Reader present, but the government has not opted into progressive PIT.
        config = CentralGovernmentConfiguration(activate_progressive_pit=False)
        assert _build_pit_schedule_by_year(config, multi_year_reader, scale=1) is None

    def test_published_years_present(self, multi_year_reader):
        config = CentralGovernmentConfiguration(activate_progressive_pit=True)
        table = _build_pit_schedule_by_year(config, multi_year_reader, scale=1)
        assert table is not None
        assert set(table) == {2014, 2016}

    def test_carries_per_year_marginal_rate(self, multi_year_reader):
        """Each published year's entry holds that year's *actual* bottom rate,
        not a CPI inflation of the base year."""
        config = CentralGovernmentConfiguration(activate_progressive_pit=True)
        table = _build_pit_schedule_by_year(config, multi_year_reader, scale=1)
        assert table[2014]["pit_rates"][0] == pytest.approx(0.0506)
        assert table[2016]["pit_rates"][0] == pytest.approx(0.0600)

    def test_thresholds_scaled_to_agent_units(self, multi_year_reader):
        """Thresholds are multiplied by ``scale`` for every year, identically to
        the single construction-year path."""
        scale = 1000
        config = CentralGovernmentConfiguration(activate_progressive_pit=True)
        unscaled = _build_pit_schedule_by_year(config, multi_year_reader, scale=1)
        scaled = _build_pit_schedule_by_year(config, multi_year_reader, scale=scale)

        for year in (2014, 2016):
            # Finite thresholds scale linearly; the open-top bracket stays inf.
            finite = np.isfinite(unscaled[year]["pit_thresholds"])
            np.testing.assert_allclose(
                scaled[year]["pit_thresholds"][finite],
                unscaled[year]["pit_thresholds"][finite] * scale,
            )
            assert np.isinf(scaled[year]["pit_thresholds"][-1])
            # Rates are unit-free — unchanged by scaling.
            np.testing.assert_array_equal(
                scaled[year]["pit_rates"], unscaled[year]["pit_rates"]
            )
