"""Tests for the taxation reader and its DataReaders-level resolution.

The taxation schedules are wired into the data pipeline like the optional
energy-sector readers: ``DataReaders.from_raw_data`` builds a ``TaxationReader``
from ``raw_data/taxation/personal_income_tax`` when present, and ``None``
otherwise.  Unlike the silent energy readers, a ``taxation/`` directory that
exists but lacks its schedules is surfaced via ``TaxationDataWarning``.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from macro_data.readers.default_readers import DataPaths, _load_taxation_reader
from macro_data.readers.taxation import TaxationDataWarning, TaxationReader, TaxationStore

# Committed schedules.
#   parents[0]=test_readers [1]=unit [2]=test_macro_data [3]=tests [4]=repo root
_COMMITTED_PIT_DIR = Path(__file__).resolve().parents[4] / "spoof_data" / "freda" / "personal_income_tax"


def _make_taxation_tree(root: Path, *, with_schedules: bool) -> Path:
    """Create a ``taxation/`` tree under *root*.

    When *with_schedules* is true the committed CSVs are copied into
    ``taxation/personal_income_tax``; otherwise the ``taxation/`` directory
    exists but its ``personal_income_tax`` subdirectory does not (the
    misconfiguration case).
    """
    taxation = root / "taxation"
    if with_schedules:
        pit = taxation / "personal_income_tax"
        pit.mkdir(parents=True)
        for csv in _COMMITTED_PIT_DIR.glob("*.csv"):
            shutil.copy(csv, pit)
    else:
        taxation.mkdir(parents=True)
    return taxation


class TestTaxationReader:
    def test_from_dir_loads_schedules(self):
        reader = TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction="bc")
        assert reader.pit_schedule.start_year == 2014
        assert reader.pit_schedule.non_refundable_tax_credits is not None
        assert reader.dividend_schedule is not None

    def test_from_dir_prefers_consolidated_jurisdiction_format(self):
        reader = TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction="bc")
        lower_bounds, _, rates = reader.pit_schedule.get_brackets(2015)
        assert np.allclose(lower_bounds, [0, 37869, 75740, 86958, 105592, 151050])
        assert np.allclose(rates, [0.0506, 0.077, 0.105, 0.1229, 0.147, 0.168])
        credits = reader.pit_schedule.non_refundable_tax_credits.get_credits(2014)
        assert any(c.credit == "Personal Amount" and c.amount == pytest.approx(9869) for c in credits)

    def test_dividend_schedule_optional(self, tmp_path):
        # Copy the two consolidated files, omit the dividend CSV.
        for name in (
            "rates_thresholds.csv",
            "non_refundable_tax_credits.csv",
        ):
            shutil.copy(_COMMITTED_PIT_DIR / name, tmp_path)
        reader = TaxationReader.from_dir(tmp_path, jurisdiction="bc")
        assert reader.dividend_schedule is None


class TestLoadTaxationReader:
    def test_none_path_is_silent_none(self, recwarn):
        assert _load_taxation_reader(None) is None
        assert not [w for w in recwarn.list if issubclass(w.category, TaxationDataWarning)]

    def test_absent_taxation_dir_is_silent_none(self, tmp_path, recwarn):
        # A taxation root that does not exist -> taxation simply not in use.
        assert _load_taxation_reader(tmp_path / "taxation") is None
        assert not [w for w in recwarn.list if issubclass(w.category, TaxationDataWarning)]

    def test_taxation_dir_without_schedules_warns(self, tmp_path):
        taxation = _make_taxation_tree(tmp_path, with_schedules=False)
        with pytest.warns(TaxationDataWarning, match="personal-income-tax schedules"):
            assert _load_taxation_reader(taxation) is None

    def test_populated_taxation_dir_loads_store(self, tmp_path):
        # The seam yields the STORE (every jurisdiction in the data); a country's
        # own schedules are the per-country slice taken at construction time.
        taxation = _make_taxation_tree(tmp_path, with_schedules=True)
        store = _load_taxation_reader(taxation)
        assert isinstance(store, TaxationStore)
        assert "bc" in store.jurisdictions

        reader = store.for_country("CAN_BC")
        assert isinstance(reader, TaxationReader)
        assert reader.pit_schedule.start_year == 2014
        assert reader.dividend_schedule is not None


class TestDataPathsTaxationWiring:
    """Cover the ``from_raw_data`` taxation seam without running the full,
    heavy ``DataReaders.from_raw_data``.

    ``from_raw_data`` resolves the taxation reader as
    ``_load_taxation_reader(datapaths.taxation_path)`` (default_readers.py), and
    ``datapaths`` comes from ``DataPaths.default_paths``.  These tests pin both
    halves of that composition: that ``default_paths`` points
    ``taxation_path`` at ``raw_data_path / "taxation"``, and that feeding that
    exact path through the loader yields a populated ``TaxationReader``.
    """

    def test_default_paths_wires_taxation_path(self, tmp_path):
        datapaths = DataPaths.default_paths(tmp_path, icio_years=[])
        assert datapaths.taxation_path == tmp_path / "taxation"

    def test_default_paths_taxation_path_loads_store(self, tmp_path):
        # Lay a populated taxation tree under a raw-data root, then drive the
        # exact path resolution ``from_raw_data`` performs.
        _make_taxation_tree(tmp_path, with_schedules=True)
        datapaths = DataPaths.default_paths(tmp_path, icio_years=[])
        store = _load_taxation_reader(datapaths.taxation_path)
        assert isinstance(store, TaxationStore)

        reader = store.for_country("CAN_BC")
        assert isinstance(reader, TaxationReader)
        assert reader.pit_schedule.start_year == 2014
        assert reader.dividend_schedule is not None

    def test_default_paths_no_taxation_tree_is_none(self, tmp_path):
        # A raw-data root with no taxation tree -> the seam yields no reader
        # (progressive PIT stays inactive), matching flat-rate parity.
        datapaths = DataPaths.default_paths(tmp_path, icio_years=[])
        assert _load_taxation_reader(datapaths.taxation_path) is None


_ON_RATES_THRESHOLDS = """year,jurisdiction,lower,rate,index
2014,ON,0.0,0.0505,1
2014,ON,43906.0,0.0915,1
2014,BC,0.0,0.0506,1
2014,BC,37606.0,0.077,1
"""


class TestGeoFilteredBrackets:
    """Bracket rows are jurisdiction-filtered — another jurisdiction's rows in the same
    consolidated CSV must never be read silently."""

    def test_brackets_filtered_by_jurisdiction(self, tmp_path):
        (tmp_path / "rates_thresholds.csv").write_text(_ON_RATES_THRESHOLDS)

        reader = TaxationReader.from_dir(tmp_path, jurisdiction="on")
        assert reader.jurisdiction == "on"
        # ON's two brackets only, never a mix with the BC rows in the same file.
        lower_bounds, _, rates = reader.pit_schedule.get_brackets(2014)
        assert np.allclose(lower_bounds, [0.0, 43906.0])
        assert np.allclose(rates, [0.0505, 0.0915])
