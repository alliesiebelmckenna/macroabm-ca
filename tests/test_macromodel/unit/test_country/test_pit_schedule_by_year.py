"""Tests for ``_build_pit_schedule_by_year`` — the per-year PIT schedule table.

The table is assembled in ``country.py`` from a multi-year taxation schedule and
stashed on the central-government agent; the ``pit_schedule_update`` pre-hook then reads
it to advance brackets/credits as the calendar year progresses.  These tests
drive the builder with a small two-year reader (a rate change between years) and
pin: opt-in / data-presence gating, one entry per published year, the per-year
statutory values, and the agent-unit scaling applied identically to every year.
There is no forward projection — a year past the last published one is the
caller's error (see ``test_country.py`` for the ``set_pit_for_year`` overflow
behaviour).
"""

from pathlib import Path

import numpy as np
import pytest

from macro_data.readers.taxation import TaxationReader
from macromodel.configurations import CentralGovernmentConfiguration
from macromodel.configurations.tax_parameters.central_government_builder import (
    activate_taxation,
)
from macromodel.country.country import _build_pit_schedule_by_year

# A two-year bracket schedule: 2016 changes the bottom marginal rate (0.0506 →
# 0.0600), which a CPI inflation of the base year could not express.
_PIT_HISTORICAL = """year,jurisdiction,lower,rate,index
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
    return TaxationReader.from_dir(tmp_path, jurisdiction="bc")


class TestBuildPitScheduleByYear:

    def test_not_opted_in_returns_none(self, multi_year_reader):
        # Reader present, but the government has not opted into progressive PIT.
        config = CentralGovernmentConfiguration(activate_progressive_pit=False)
        assert _build_pit_schedule_by_year(config, multi_year_reader, scale=1) is None


    def test_carries_per_year_marginal_rate(self, multi_year_reader):
        """Each published year's entry holds that year's *actual* bottom rate,
        not a CPI inflation of the base year."""
        config = CentralGovernmentConfiguration(activate_progressive_pit=True)
        table = _build_pit_schedule_by_year(config, multi_year_reader, scale=1)
        assert table[2014]["pit_rates"][0] == pytest.approx(0.0506)
        assert table[2016]["pit_rates"][0] == pytest.approx(0.0600)


# Committed schedules.
#   parents[0]=test_country [1]=unit [2]=test_macromodel [3]=tests [4]=repo root
_COMMITTED_PIT_DIR = (
    Path(__file__).resolve().parents[4] / "spoof_data" / "freda" / "personal_income_tax"
)

# Config fields the fragment stores under different key names.
_FRAGMENT_KEYS_FOR = {"pit_brackets": {"pit_uppers", "pit_rates"}}

# Non-tax configuration carried for the agent's own construction.
_NOT_SCHEDULE_DRIVEN = {"functions"}


def _differs(left, right) -> bool:
    """Value inequality that tolerates un-comparable configuration objects."""
    try:
        return bool(left != right)
    except Exception:  # pragma: no cover - defensive
        return False


class TestFragmentCoversEveryYearVaryingParameter:
    """Whatever the schedule varies by year, the fragment must carry.

    ``set_pit_for_year`` advances a government by swapping one fragment into
    its states, so a parameter the schedule publishes per year but the fragment
    omits is frozen at its construction-year value for the whole run — silently,
    and diverging further the longer the simulation runs.
    """

    def test_no_year_varying_parameter_is_omitted_from_the_fragment(self):
        reader = TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction="bc")
        config = CentralGovernmentConfiguration(activate_progressive_pit=True)

        # Two published years far enough apart that the schedules differ.
        early = activate_taxation(base_config=config, taxation_reader=reader, year=2014)
        late = activate_taxation(base_config=config, taxation_reader=reader, year=2019)

        varying = {
            name
            for name in type(early).model_fields
            if name not in _NOT_SCHEDULE_DRIVEN
            and _differs(getattr(early, name), getattr(late, name))
        }
        assert varying, "the published schedule does not vary between 2014 and 2019"

        table = _build_pit_schedule_by_year(config, reader, scale=1)
        fragment_keys = set(table[2014])

        omitted = sorted(
            name
            for name in varying
            if not (_FRAGMENT_KEYS_FOR.get(name, {name}) & fragment_keys)
        )
        assert not omitted, (
            "these parameters vary by published year but are absent from the "
            "per-year fragment, so they freeze at the construction year: "
            f"{omitted}"
        )

