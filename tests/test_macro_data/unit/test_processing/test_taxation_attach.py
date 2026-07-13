"""Tests for the per-country slice of the taxation data layer.

The taxation store carries every taxing authority in the data; each country is
built with its OWN schedules, mirroring the other country-keyed readers. Three
properties matter:

* a country is taxed by its own jurisdiction's schedule and never by another's
  (BC's brackets must not tax Ontario);
* a country the data does not cover falls back to the flat rate, silently and
  without raising — Quebec's state before its schedules are sourced;
* partial coverage degrades per-component: a jurisdiction with brackets but no
  dividend rows still runs progressive PIT, without that component.

The attach has regressed silently once before — a country built with
``taxation=None`` runs the flat rate regardless of ``activate_progressive_pit``,
with no error anywhere — so the routing is pinned rather than trusted.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from macro_data.processing.synthetic_country import taxation_for_country
from macro_data.readers.taxation import TaxationStore

# Committed schedules (BC + CA federal).
#   parents[0]=test_processing [1]=unit [2]=test_macro_data [3]=tests [4]=repo root
_COMMITTED_PIT_DIR = (
    Path(__file__).resolve().parents[4] / "spoof_data" / "freda" / "personal_income_tax"
)


@pytest.fixture(scope="module")
def store() -> TaxationStore:
    """The committed schedules: BC (provincial) and CA (federal)."""
    return TaxationStore.from_dir(_COMMITTED_PIT_DIR)


def test_country_receives_its_own_schedule(store):
    """BC is taxed by BC's schedules."""
    reader = taxation_for_country("CAN_BC", store)
    assert reader is not None
    assert reader.jurisdiction == "bc"


@pytest.mark.parametrize("code", ["CAN_ON", "CAN_QC"])
def test_uncovered_countries_fall_back_to_flat(code, store):
    """A country the data omits gets None - it must never inherit BC's brackets."""
    assert taxation_for_country(code, store) is None


def test_absent_taxation_data_gives_none():
    """With no taxation store at all, even a covered country stays flat."""
    assert taxation_for_country("CAN_BC", None) is None


def test_each_country_gets_a_distinct_schedule(tmp_path):
    """With several jurisdictions present, each country slices its own - not the first.

    Guards the regression the old design was prone to: one reader loaded at import
    time and handed to everybody.
    """
    pit_dir = tmp_path / "personal_income_tax"
    pit_dir.mkdir(parents=True)
    for csv in _COMMITTED_PIT_DIR.glob("*.csv"):
        shutil.copy(csv, pit_dir)

    # Give Alberta a distinctive single-bracket schedule alongside BC's.
    rates_path = pit_dir / "rates_thresholds.csv"
    rates = pd.read_csv(rates_path)
    ab = pd.DataFrame(
        [
            {"tax_year": y, "geo": "AB", "lower": 0.0, "rate": 0.10, "index": 0}
            for y in sorted(rates["tax_year"].unique())
        ]
    )
    pd.concat([rates, ab], ignore_index=True).to_csv(rates_path, index=False)

    store = TaxationStore.from_dir(pit_dir)
    bc = taxation_for_country("CAN_BC", store)
    alberta = taxation_for_country("CAN_AB", store)
    assert bc is not None and alberta is not None
    assert bc.jurisdiction == "bc" and alberta.jurisdiction == "ab"

    _th_bc, rates_bc, _lo_bc, _q_bc = bc.pit_schedule.get_brackets(2014)
    _th_ab, rates_ab, _lo_ab, _q_ab = alberta.pit_schedule.get_brackets(2014)

    # Alberta's flat 10% must not be BC's progressive ladder.
    assert list(rates_ab) == [0.10]
    assert len(rates_bc) > 1


def test_the_data_file_decides_who_is_taxed_progressively(tmp_path):
    """The SAME code, pointed at two different bracket files, taxes different provinces.

    The reader is not tied to a filename, only to the geo-keyed schema: a bracket
    file covering BC alone activates progressive PIT for BC and leaves the rest
    flat; one covering every province activates them all. Nothing but the data
    changes.
    """
    pit_dir = tmp_path / "personal_income_tax"
    pit_dir.mkdir(parents=True)
    for csv in _COMMITTED_PIT_DIR.glob("*.csv"):
        shutil.copy(csv, pit_dir)

    rates = pd.read_csv(pit_dir / "rates_thresholds.csv")
    extra = pd.DataFrame(
        [
            {"tax_year": y, "geo": geo, "lower": 0.0, "rate": rate, "index": 0}
            for geo, rate in (("ON", 0.0505), ("QC", 0.15))
            for y in sorted(rates["tax_year"].unique())
        ]
    )
    pd.concat([rates, extra], ignore_index=True).to_csv(
        pit_dir / "rates_thresholds_all_provinces.csv", index=False
    )

    narrow = TaxationStore.from_dir(pit_dir)
    assert taxation_for_country("CAN_BC", narrow) is not None
    assert taxation_for_country("CAN_ON", narrow) is None, "ON has no schedule -> flat"

    wide = TaxationStore.from_dir(pit_dir, rates="rates_thresholds_all_provinces.csv")
    assert taxation_for_country("CAN_BC", wide) is not None
    on = taxation_for_country("CAN_ON", wide)
    assert on is not None, "ON now has a schedule -> progressive"
    assert on.jurisdiction == "on"


def test_missing_dividend_rows_degrade_rather_than_raise(tmp_path):
    """A jurisdiction with brackets but no dividend rows still runs progressive PIT.

    The dividend file covers BC and CA only. A newly added province must not crash
    the build for lack of dividend rates - it runs without the gross-up / DTC path
    until they are sourced.
    """
    pit_dir = tmp_path / "personal_income_tax"
    pit_dir.mkdir(parents=True)
    for csv in _COMMITTED_PIT_DIR.glob("*.csv"):
        shutil.copy(csv, pit_dir)

    rates_path = pit_dir / "rates_thresholds.csv"
    rates = pd.read_csv(rates_path)
    sk = pd.DataFrame(
        [
            {"tax_year": y, "geo": "SK", "lower": 0.0, "rate": 0.11, "index": 0}
            for y in sorted(rates["tax_year"].unique())
        ]
    )
    pd.concat([rates, sk], ignore_index=True).to_csv(rates_path, index=False)

    reader = taxation_for_country("CAN_SK", TaxationStore.from_dir(pit_dir))

    assert reader is not None, "SK has brackets, so it must be covered"
    assert reader.pit_schedule is not None
    assert reader.dividend_schedule is None, "no SK dividend rows -> degrade, not raise"
