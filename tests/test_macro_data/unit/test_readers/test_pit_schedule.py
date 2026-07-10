"""Unit tests for the Progressive Personal Income Tax (PIT) schedule.

Covers:
- Pure-function tax computation (``compute_progressive_tax``,
  ``compute_progressive_tax_quick``)
- Bracket validation (``_validate_brackets``)
- ``PITSchedule`` class: CSV loading, CPI indexing, error handling
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from macro_data.readers.taxation.personal_income_tax.pit_schedule import (
    PITSchedule,
    compute_progressive_tax,
    compute_progressive_tax_quick,
    _recompute_quick_add,
    _validate_brackets,
)

# Committed fallback copy of the BC PIT schedules
# (repo-root/spoof_data/freda/personal_income_tax).
#   parents[0]=test_readers [1]=unit [2]=test_macro_data [3]=tests [4]=repo root
BC_SCHEDULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "spoof_data" / "freda" / "personal_income_tax"
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def bc_2014_rates() -> np.ndarray:
    """Real BC 2014 marginal rates (6 brackets)."""
    return np.array([0.0506, 0.077, 0.105, 0.1229, 0.147, 0.168])


@pytest.fixture(scope="module")
def bc_2014_lower_bounds() -> np.ndarray:
    """Real BC 2014 lower bounds (6 brackets)."""
    return np.array([0, 37606, 75213, 86354, 104858, 150000], dtype=float)


@pytest.fixture(scope="module")
def bc_2014_thresholds(bc_2014_lower_bounds) -> np.ndarray:
    """Real BC 2014 upper-bound thresholds (6 brackets, last = inf)."""
    return np.append(bc_2014_lower_bounds[1:].copy(), np.inf)


@pytest.fixture(scope="module")
def bc_2014_quick_adds(bc_2014_lower_bounds, bc_2014_rates) -> np.ndarray:
    """Pre-computed quick-add values for BC 2014 brackets."""
    return _recompute_quick_add(bc_2014_lower_bounds, bc_2014_rates)


@pytest.fixture(scope="module")
def sample_csv_path() -> Path:
    """Write a minimal 2-bracket PIT CSV to a temp file."""
    csv_content = (
        "tax_year,geo,lower,rate,index\n"
        "2020,BC,0,0.10,1\n"
        "2020,BC,50000,0.25,1\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        f.write(csv_content)
        return Path(f.name)


# ═══════════════════════════════════════════════════════════════════════
# 1. compute_progressive_tax — pure-function tests
# ═══════════════════════════════════════════════════════════════════════


class TestComputeProgressiveTax:
    """Standalone progressive tax computation."""

    def test_single_bracket_flat_tax(self):
        """One infinite bracket → flat percentage on all income."""
        incomes = np.array([100.0, 200.0, 0.0])
        thresholds = np.array([np.inf])
        rates = np.array([0.15])
        tax = compute_progressive_tax(incomes, thresholds, rates)
        assert np.allclose(tax, [15.0, 30.0, 0.0])

    def test_two_brackets_marginal_slicing(self):
        """Income spans two brackets — each slice taxed at its own rate."""
        incomes = np.array([30.0, 80.0, 200.0])
        thresholds = np.array([50.0, np.inf])
        rates = np.array([0.10, 0.25])
        tax = compute_progressive_tax(incomes, thresholds, rates)
        # 30  → 30×0.10 = 3
        # 80  → 50×0.10 + 30×0.25 = 5 + 7.5 = 12.5
        # 200 → 50×0.10 + 150×0.25 = 5 + 37.5 = 42.5
        assert np.allclose(tax, [3.0, 12.5, 42.5])

    def test_boundary_goes_to_lower_bracket(self):
        """Income exactly at a threshold belongs to the lower bracket."""
        incomes = np.array([50.0])
        thresholds = np.array([50.0, np.inf])
        rates = np.array([0.10, 0.25])
        tax = compute_progressive_tax(incomes, thresholds, rates)
        assert np.allclose(tax, [5.0])  # 50×0.10, not 50×0.25

    def test_zero_income(self):
        """Zero income → zero tax."""
        incomes = np.array([0.0, 0.0])
        thresholds = np.array([50.0, np.inf])
        rates = np.array([0.10, 0.25])
        tax = compute_progressive_tax(incomes, thresholds, rates)
        assert np.allclose(tax, [0.0, 0.0])

    def test_vectorized_against_loop(self):
        """Vectorized path matches element-wise loop for random incomes."""
        rng = np.random.default_rng(42)
        incomes = rng.uniform(0, 500_000, size=1000)
        thresholds = np.array([37606, 75213, 86354, 104858, 150000, np.inf])
        rates = np.array([0.0506, 0.077, 0.105, 0.1229, 0.147, 0.168])
        tax_vec = compute_progressive_tax(incomes, thresholds, rates)

        tax_loop = np.zeros_like(incomes)
        for i, inc in enumerate(incomes):
            t = 0.0
            lo = 0.0
            for th, r in zip(thresholds, rates):
                amt = max(0.0, min(inc, th) - lo)
                t += r * amt
                lo = th
            tax_loop[i] = t
        assert np.allclose(tax_vec, tax_loop)

    def test_very_high_income_no_overflow(self):
        """Stress-test: extremely high income doesn't overflow."""
        incomes = np.array([1e12, 1e15])
        thresholds = np.array([37606, 75213, np.inf])
        rates = np.array([0.05, 0.10, 0.15])
        tax = compute_progressive_tax(incomes, thresholds, rates)
        assert np.all(np.isfinite(tax))
        assert np.all(tax > 0)

    def test_random_income_bc_brackets(
        self, bc_2014_thresholds, bc_2014_rates
    ):
        """Random incomes against real BC 2014 brackets — sanity check."""
        rng = np.random.default_rng(123)
        incomes = rng.uniform(0, 300_000, size=500)
        tax = compute_progressive_tax(incomes, bc_2014_thresholds, bc_2014_rates)
        # All taxes should be strictly less than income (rates < 1)
        assert np.all(tax < incomes)
        # Rates are monotonic, so lower incomes should pay <= higher incomes
        assert np.all(np.diff(tax[np.argsort(incomes)]) >= -1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 2. compute_progressive_tax_quick — fast-path tests
# ═══════════════════════════════════════════════════════════════════════


class TestComputeProgressiveTaxQuick:
    """Optimised quick-add path must match the slow path exactly."""

    def test_matches_slow_path_bc_brackets(
        self,
        bc_2014_thresholds,
        bc_2014_rates,
        bc_2014_lower_bounds,
        bc_2014_quick_adds,
    ):
        """BC 2014 brackets: slow and fast agree on 1 000 random incomes."""
        rng = np.random.default_rng(42)
        incomes = rng.uniform(0, 500_000, size=1000)
        slow = compute_progressive_tax(incomes, bc_2014_thresholds, bc_2014_rates)
        fast = compute_progressive_tax_quick(
            incomes, bc_2014_lower_bounds, bc_2014_rates, bc_2014_quick_adds
        )
        assert np.allclose(slow, fast, atol=1e-10)

    def test_matches_slow_path_simple(self):
        """Two-bracket case: slow and fast match."""
        incomes = np.array([0.0, 30.0, 80.0, 200.0])
        thresholds = np.array([50.0, np.inf])
        rates = np.array([0.10, 0.25])
        lower_bounds = np.array([0.0, 50.0])
        quick_adds = _recompute_quick_add(lower_bounds, rates)

        slow = compute_progressive_tax(incomes, thresholds, rates)
        fast = compute_progressive_tax_quick(
            incomes, lower_bounds, rates, quick_adds
        )
        assert np.allclose(slow, fast)

    def test_negative_incomes_yield_zero_tax(
        self,
        bc_2014_lower_bounds,
        bc_2014_rates,
        bc_2014_quick_adds,
    ):
        """Negative incomes (if any) should yield zero tax."""
        incomes = np.array([-100.0, -1.0, 0.0])
        fast = compute_progressive_tax_quick(
            incomes, bc_2014_lower_bounds, bc_2014_rates, bc_2014_quick_adds
        )
        assert np.allclose(fast, [0.0, 0.0, 0.0])

    def test_length_mismatch_raises(self):
        """Different-length arrays raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            compute_progressive_tax_quick(
                np.array([100.0]),
                np.array([0.0, 50.0]),
                np.array([0.1]),
                np.array([0.0, 5.0]),
            )


# ═══════════════════════════════════════════════════════════════════════
# 3. _validate_brackets — input validation
# ═══════════════════════════════════════════════════════════════════════


class TestValidateBrackets:
    """Input validation for bracket arrays."""

    def test_valid_brackets_pass(self):
        """Well-formed brackets pass validation silently."""
        _validate_brackets(
            np.array([50.0, np.inf]), np.array([0.1, 0.2])
        )

    def test_non_increasing_thresholds_raises(self):
        """Non-strictly-increasing thresholds raise."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_brackets(
                np.array([100.0, 50.0]), np.array([0.1, 0.2])
            )

    def test_duplicate_thresholds_raises(self):
        """Duplicate thresholds raise."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_brackets(
                np.array([50.0, 50.0, np.inf]), np.array([0.1, 0.2, 0.3])
            )

    def test_rate_outside_range_raises(self):
        """Rates outside [0, 1] raise."""
        with pytest.raises(ValueError, match="rates must be in"):
            _validate_brackets(
                np.array([50.0, np.inf]), np.array([0.1, 1.5])
            )

    def test_mismatched_lengths_raises(self):
        """Different-length threshold/rate arrays raise."""
        with pytest.raises(ValueError, match="same length"):
            _validate_brackets(
                np.array([50.0, np.inf]), np.array([0.1, 0.2, 0.3])
            )


# ═══════════════════════════════════════════════════════════════════════
# 4. _recompute_quick_add — helper
# ═══════════════════════════════════════════════════════════════════════


class TestRecomputeQuickAdd:
    def test_first_bracket_quick_add_zero(self):
        """First bracket always has quick_add = 0."""
        quick = _recompute_quick_add(
            np.array([0.0, 50.0, np.inf]), np.array([0.1, 0.2, 0.3])
        )
        assert quick[0] == 0.0

    def test_known_values(self):
        """Two-bracket case: known quick-add values."""
        lower = np.array([0.0, 50.0])
        rates = np.array([0.10, 0.25])
        quick = _recompute_quick_add(lower, rates)
        # quick[0] = 0
        # quick[1] = 0 + 0.10 * (50 - 0) = 5
        assert np.allclose(quick, [0.0, 5.0])

    def test_three_brackets(self):
        """Three-bracket cumulative quick-add."""
        lower = np.array([0.0, 100.0, 200.0])
        rates = np.array([0.10, 0.20, 0.30])
        quick = _recompute_quick_add(lower, rates)
        # quick[0] = 0
        # quick[1] = 0 + 0.10 * 100 = 10
        # quick[2] = 10 + 0.20 * 100 = 30
        assert np.allclose(quick, [0.0, 10.0, 30.0])


# ═══════════════════════════════════════════════════════════════════════
# 5. PITSchedule — class-level tests
# ═══════════════════════════════════════════════════════════════════════


class TestPITSchedule:
    """Integration tests for the PITSchedule class."""

    def test_from_name_loads_bc_2014(self):
        """The consolidated file loads BC 2014 with 6 brackets."""
        schedule = PITSchedule.from_name("rates_thresholds.csv", schedule_dir=BC_SCHEDULE_DIR)
        assert schedule.base_year == 2014
        thresholds, rates, lower_bounds, quick_adds = schedule.get_brackets(
            tax_year=2014
        )
        assert len(thresholds) == 6
        assert len(rates) == 6
        assert len(lower_bounds) == 6
        assert len(quick_adds) == 6
        # First bracket starts at 0
        assert lower_bounds[0] == 0.0
        # Last threshold is inf
        assert np.isinf(thresholds[-1])
        # Rates match expected BC 2014 values
        assert np.allclose(
            rates, [0.0506, 0.077, 0.105, 0.1229, 0.147, 0.168]
        )

    def test_from_name_loads_consolidated_geo_format(self):
        """The contributor's geo-keyed consolidated file loads for BC too."""
        schedule = PITSchedule.from_name(
            "rates_thresholds.csv",
            schedule_dir=BC_SCHEDULE_DIR,
            jurisdiction="bc",
        )
        assert schedule.base_year == 2014
        thresholds, rates, lower_bounds, _ = schedule.get_brackets(tax_year=2015)
        assert len(thresholds) == 6
        assert np.allclose(lower_bounds, [0, 37869, 75740, 86958, 105592, 151050])
        assert np.allclose(rates, [0.0506, 0.077, 0.105, 0.1229, 0.147, 0.168])

    def test_get_brackets_base_year_no_inflation(self):
        """Base-year brackets equal nominal CSV values."""
        schedule = PITSchedule.from_name("rates_thresholds.csv", schedule_dir=BC_SCHEDULE_DIR)
        _, _, lower_bounds, _ = schedule.get_brackets(tax_year=2014)
        # Known nominal BC 2014 lower bounds from the consolidated file
        expected = [0, 37606, 75213, 86354, 104858, 150000]
        assert np.allclose(lower_bounds, expected)

    def test_get_brackets_before_base_year_raises(self):
        """Requesting a year before the base year raises."""
        schedule = PITSchedule.from_name("rates_thresholds.csv", schedule_dir=BC_SCHEDULE_DIR)
        with pytest.raises(
            ValueError, match="before base year"
        ):
            schedule.get_brackets(tax_year=2013)

    def test_get_brackets_out_of_table_raises(self):
        """A year past the published schedule raises (the reader is lookup-only;
        there is no forward projection anywhere in the pipeline)."""
        schedule = PITSchedule.from_name(
            "rates_thresholds.csv",
            schedule_dir=BC_SCHEDULE_DIR,
        )
        # The consolidated fixture publishes 2014-2030, so a far-future year is
        # out of table and raises.
        with pytest.raises(ValueError, match="not in the published schedule"):
            schedule.get_brackets(tax_year=2099)

    def test_from_csv_missing_columns_raises(self):
        """CSV missing required columns raises ValueError."""
        csv = "tax_year,lower\n2020,0\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv)
            p = f.name

        try:
            with pytest.raises(
                ValueError, match="missing required columns"
            ):
                PITSchedule.from_csv(p)
        finally:
            Path(p).unlink(missing_ok=True)

    def test_from_name_file_not_found_raises(self):
        """Non-existent schedule name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Schedule file not found"):
            PITSchedule.from_name("NONEXISTENT.csv", schedule_dir=BC_SCHEDULE_DIR)

    def test_compute_tax_convenience(self):
        """PITSchedule.compute_tax wrapper matches manual call."""
        schedule = PITSchedule.from_name("rates_thresholds.csv", schedule_dir=BC_SCHEDULE_DIR)
        incomes = np.array([30000.0, 80000.0, 200000.0])

        via_method = schedule.compute_tax(incomes, tax_year=2014)
        thresholds, rates, _, _ = schedule.get_brackets(tax_year=2014)
        via_direct = compute_progressive_tax(incomes, thresholds, rates)

        assert np.allclose(via_method, via_direct)

    def test_available_years(self):
        """available_years returns sorted unique years from CSV."""
        schedule = PITSchedule.from_name("rates_thresholds.csv", schedule_dir=BC_SCHEDULE_DIR)
        years = schedule.available_years
        assert len(years) > 0
        assert years[0] == 2014
        assert np.all(np.diff(years) >= 0)  # sorted


# ═══════════════════════════════════════════════════════════════════════
# 6. PITSchedule — statutory lookup over a MULTI-YEAR schedule
# ═══════════════════════════════════════════════════════════════════════


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
            "tax_year,geo,lower,rate,index\n"
            "2014,BC,0,0.05,1\n"
            "2014,BC,40000,0.10,1\n"
            "2015,BC,0,0.05,1\n"
            "2015,BC,41000,0.10,1\n"
            "2016,BC,0,0.06,1\n"
            "2016,BC,42000,0.11,1\n"
        )
        return p

    def test_present_year_returns_actual_rows(self, tmp_path):
        """get_brackets for a listed year returns that year's own bounds/rates."""
        sched = PITSchedule.from_csv(self._multiyear_csv(tmp_path))
        _, rates, lower_bounds, _ = sched.get_brackets(tax_year=2015)
        assert np.allclose(lower_bounds, [0.0, 41000.0])
        assert np.allclose(rates, [0.05, 0.10])

    def test_lookup_captures_statutory_rate_change(self, tmp_path):
        """2016's bottom/top rates (0.06/0.11) differ from 2014's (0.05/0.10);
        compounding can't produce them — only a per-year lookup can."""
        sched = PITSchedule.from_csv(self._multiyear_csv(tmp_path))
        _, rates, lower_bounds, _ = sched.get_brackets(tax_year=2016)
        assert np.allclose(lower_bounds, [0.0, 42000.0])
        assert np.allclose(rates, [0.06, 0.11])

    def test_base_year_lookup_not_polluted_by_later_years(self, tmp_path):
        """Base-year lookup returns ONLY the base year's two brackets, not a
        mix of all years' rows (the bug in the all-rows sort)."""
        sched = PITSchedule.from_csv(self._multiyear_csv(tmp_path))
        _, rates, lower_bounds, _ = sched.get_brackets(tax_year=2014)
        assert len(lower_bounds) == 2
        assert np.allclose(lower_bounds, [0.0, 40000.0])
        assert np.allclose(rates, [0.05, 0.10])

    def test_base_year_is_minimum_year_even_when_unsorted(self, tmp_path):
        """A valid but unsorted CSV (later year block first) must not shift
        the base year — it is the minimum year, not the first row's."""
        path = tmp_path / "unsorted.csv"
        path.write_text(
            "tax_year,geo,lower,rate,index\n"
            "2016,BC,0,0.06,1\n"
            "2016,BC,42000,0.11,1\n"
            "2014,BC,0,0.05,1\n"
            "2014,BC,40000,0.10,1\n"
        )
        sched = PITSchedule.from_csv(path)
        assert sched.base_year == 2014
        # The pre-base-year validation keys off the true base year.
        _, rates, _, _ = sched.get_brackets(tax_year=2014)
        assert np.allclose(rates, [0.05, 0.10])
