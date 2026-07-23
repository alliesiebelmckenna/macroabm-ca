"""Unit tests for the Progressive Personal Income Tax (PIT) schedule.

Covers:
- Pure-function tax computation (``compute_personal_income_tax``)
- Bracket validation (``_validate_brackets``)
- ``PITSchedule`` class: CSV loading, CPI indexing, error handling
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from macro_data.readers.taxation.personal_income_tax.pit_schedule import (
    PITSchedule,
    compute_personal_income_tax,
    _validate_brackets,
)

# Committed fallback copy of the BC PIT schedules
# (repo-root/spoof_data/freda/personal_income_tax).
#   parents[0]=test_readers [1]=unit [2]=test_macro_data [3]=tests [4]=repo root
BC_SCHEDULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "spoof_data" / "freda" / "personal_income_tax"
)


# Fixtures


@pytest.fixture(scope="module")
def sample_csv_path() -> Path:
    """Write a minimal 2-bracket PIT CSV to a temp file."""
    csv_content = (
        "year,jurisdiction,lower,rate,index\n"
        "2020,BC,0,0.10,1\n"
        "2020,BC,50000,0.25,1\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        f.write(csv_content)
        return Path(f.name)


# 1. compute_personal_income_tax — pure-function tests


class TestComputeProgressiveTax:
    """Standalone progressive tax computation."""


    def test_two_brackets_marginal_slicing(self):
        """Income spans two brackets — each slice taxed at its own rate."""
        incomes = np.array([30.0, 80.0, 200.0])
        uppers = np.array([50.0, np.inf])
        rates = np.array([0.10, 0.25])
        tax = compute_personal_income_tax(incomes, uppers, rates)
        # 30  → 30×0.10 = 3
        # 80  → 50×0.10 + 30×0.25 = 5 + 7.5 = 12.5
        # 200 → 50×0.10 + 150×0.25 = 5 + 37.5 = 42.5
        assert np.allclose(tax, [3.0, 12.5, 42.5])







# 2. _validate_brackets — input validation


class TestValidateBrackets:
    """Input validation for bracket arrays."""


    def test_non_increasing_uppers_raises(self):
        """Non-strictly-increasing uppers raise."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_brackets(
                np.array([100.0, 50.0]), np.array([0.1, 0.2])
            )





# 3. PITSchedule — class-level tests


    def test_nan_is_rejected_in_thresholds_and_rates(self):
        """NaN passes every range check, so it needs its own.

        ``nan < 0`` and ``nan > 1`` are both False, so a NaN rate would
        validate and then produce NaN tax for the whole population, which
        propagates into revenue, deficit and accumulating debt.
        """
        nan = float("nan")
        with pytest.raises(ValueError, match="must not be NaN"):
            _validate_brackets(np.array([1000.0, np.inf]), np.array([nan, 0.2]))
        # A single bracket skips the strictly-increasing check entirely, so the
        # incidental protection NaN uppers used to get does not reach here.
        with pytest.raises(ValueError, match="must not be NaN"):
            _validate_brackets(np.array([nan]), np.array([0.1]))


class TestPITSchedule:
    """Integration tests for the PITSchedule class."""

    def test_from_csv_loads_bc_2014(self):
        """The consolidated file loads BC 2014 with 6 brackets."""
        schedule = PITSchedule.from_csv(
            BC_SCHEDULE_DIR / "rates_thresholds.csv", jurisdiction="bc"
        )
        assert schedule.start_year == 2014
        lower_bounds, uppers, rates = schedule.get_brackets(
            year=2014
        )
        assert len(uppers) == 6
        assert len(rates) == 6
        assert len(lower_bounds) == 6
        # First bracket starts at 0
        assert lower_bounds[0] == 0.0
        # Last upper is inf
        assert np.isinf(uppers[-1])
        # Rates match expected BC 2014 values
        assert np.allclose(
            rates, [0.0506, 0.077, 0.105, 0.1229, 0.147, 0.168]
        )




    def test_get_brackets_out_of_table_raises(self):
        """A year past the published schedule raises (the reader is lookup-only;
        there is no forward projection anywhere in the pipeline)."""
        schedule = PITSchedule.from_csv(
            BC_SCHEDULE_DIR / "rates_thresholds.csv",
            jurisdiction="bc",
        )
        # The consolidated fixture publishes 2014-2030, so a far-future year is
        # out of table and raises.
        with pytest.raises(ValueError, match="No PIT data available"):
            schedule.get_brackets(year=2099)

    def test_from_csv_missing_columns_raises(self):
        """CSV missing required columns raises ValueError."""
        csv = "year,lower\n2020,0\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv)
            p = f.name

        try:
            with pytest.raises(
                ValueError, match="missing required columns"
            ):
                PITSchedule.from_csv(p, jurisdiction="bc")
        finally:
            Path(p).unlink(missing_ok=True)





# 6. PITSchedule — statutory lookup over a MULTI-YEAR schedule


class TestStatutoryLookup:
    """A multi-year schedule must return the ACTUAL published rows for a year
    present in the table (statutory lookup) — NOT values compounded from the
    base year. A year NOT in the table is out of range and raises (the reader is
    lookup-only)."""

    @staticmethod
    def _multiyear_csv(tmp_path) -> Path:
        # Distinct per-year values so lookup is distinguishable from compounding.
        # 2015 changes a threshold (40000->41000); 2016 ALSO changes the bottom
        # rate (0.05->0.06) and top rate (0.10->0.11) — and compounding NEVER
        # changes rates, so a correct rate can only come from lookup.
        p = tmp_path / "multi.csv"
        p.write_text(
            "year,jurisdiction,lower,rate,index\n"
            "2014,BC,0,0.05,1\n"
            "2014,BC,40000,0.10,1\n"
            "2015,BC,0,0.05,1\n"
            "2015,BC,41000,0.10,1\n"
            "2016,BC,0,0.06,1\n"
            "2016,BC,42000,0.11,1\n"
        )
        return p


    def test_lookup_captures_statutory_rate_change(self, tmp_path):
        """2016's bottom/top rates (0.06/0.11) differ from 2014's (0.05/0.10);
        compounding can't produce them — only a per-year lookup can."""
        sched = PITSchedule.from_csv(self._multiyear_csv(tmp_path), jurisdiction="bc")
        lower_bounds, _, rates = sched.get_brackets(year=2016)
        assert np.allclose(lower_bounds, [0.0, 42000.0])
        assert np.allclose(rates, [0.06, 0.11])


