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
        "tax_year,step,lower_bound,marginal_rate,indexing\n"
        "2020,1,0,0.10,1\n"
        "2020,2,50000,0.25,1\n"
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
        """Built-in BC_PIT_2014.csv loads with 6 brackets."""
        schedule = PITSchedule.from_name("BC_PIT_2014.csv")
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

    def test_from_name_with_cpi_map(self):
        """Explicit CPI map overrides any cached/default."""
        schedule = PITSchedule.from_name(
            "BC_PIT_2014.csv", cpi_map={2014: 0.10, 2015: 0.20}
        )
        assert schedule.cpi_map[2014] == 0.10
        assert schedule.cpi_map[2015] == 0.20

    def test_get_brackets_base_year_no_inflation(self):
        """Base-year brackets equal nominal CSV values."""
        schedule = PITSchedule.from_name("BC_PIT_2014.csv")
        _, _, lower_bounds, _ = schedule.get_brackets(tax_year=2014)
        # Known nominal lower bounds from BC_PIT_2014.csv
        expected = [0, 37606, 75213, 86354, 104858, 150000]
        assert np.allclose(lower_bounds, expected)

    def test_get_brackets_before_base_year_raises(self):
        """Requesting a year before the base year raises."""
        schedule = PITSchedule.from_name("BC_PIT_2014.csv")
        with pytest.raises(
            ValueError, match="before base year"
        ):
            schedule.get_brackets(tax_year=2013)

    def test_cpi_indexing_inflates_indexed_bounds(self):
        """CPI-inflated year: indexed bounds compound, non-indexed stay."""
        schedule = PITSchedule.from_name(
            "BC_PIT_2014.csv",
            cpi_map={2014: 0.01, 2015: 0.02, 2016: 0.03},
        )
        _, _, lbs_base, _ = schedule.get_brackets(tax_year=2014)
        _, _, lbs_2017, _ = schedule.get_brackets(tax_year=2017)

        # All BC brackets are indexed, so:
        # factor = (1.01) * (1.02) * (1.03) ≈ 1.061106
        factor = 1.01 * 1.02 * 1.03

        # First bracket lower_bound=0 stays 0
        assert lbs_2017[0] == 0.0
        # Second bracket: 37606 × factor
        assert lbs_2017[1] == pytest.approx(lbs_base[1] * factor)
        # All indexed bounds should be scaled
        for i in range(1, len(lbs_base)):
            assert lbs_2017[i] == pytest.approx(lbs_base[i] * factor)

    def test_missing_cpi_year_raises(self):
        """CPI gap between base_year and requested year raises."""
        schedule = PITSchedule.from_name(
            "BC_PIT_2014.csv",
            cpi_map={2014: 0.01},  # only 2014, not 2015
        )
        with pytest.raises(
            ValueError, match="Missing CPI inflation"
        ):
            schedule.get_brackets(tax_year=2016)

    def test_cpi_indexing_respects_indexing_flag(self):
        """Only brackets with indexing=1 are inflated."""
        # Write a temp CSV with mixed indexing flags
        csv = (
            "tax_year,step,lower_bound,marginal_rate,indexing\n"
            "2020,1,0,0.10,1\n"       # indexed
            "2020,2,50000,0.20,0\n"    # NOT indexed
            "2020,3,100000,0.30,1\n"   # indexed
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv)
            p = f.name

        try:
            schedule = PITSchedule.from_csv(
                p, cpi_map={2020: 0.10, 2021: 0.10}
            )
            _, _, lbs_base, _ = schedule.get_brackets(tax_year=2020)
            _, _, lbs_2022, _ = schedule.get_brackets(tax_year=2022)
            factor = 1.10 * 1.10  # 1.21

            # Bracket 0 (indexed, lower_bound=0): stays 0
            assert lbs_2022[0] == 0.0
            # Bracket 1 (NOT indexed): stays nominal
            assert lbs_2022[1] == pytest.approx(lbs_base[1])
            # Bracket 2 (indexed): inflated
            assert lbs_2022[2] == pytest.approx(lbs_base[2] * factor)
        finally:
            Path(p).unlink(missing_ok=True)

    def test_from_csv_missing_columns_raises(self):
        """CSV missing required columns raises ValueError."""
        csv = "tax_year,step,lower_bound\n2020,1,0\n"
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
            PITSchedule.from_name("NONEXISTENT.csv")

    def test_compute_tax_convenience(self):
        """PITSchedule.compute_tax wrapper matches manual call."""
        schedule = PITSchedule.from_name("BC_PIT_2014.csv")
        incomes = np.array([30000.0, 80000.0, 200000.0])

        via_method = schedule.compute_tax(incomes, tax_year=2014)
        thresholds, rates, _, _ = schedule.get_brackets(tax_year=2014)
        via_direct = compute_progressive_tax(incomes, thresholds, rates)

        assert np.allclose(via_method, via_direct)

    def test_available_years(self):
        """available_years returns sorted unique years from CSV."""
        schedule = PITSchedule.from_name("BC_PIT_2014.csv")
        years = schedule.available_years
        assert len(years) > 0
        assert years[0] == 2014
        assert np.all(np.diff(years) >= 0)  # sorted

    def test_from_csv_with_embedded_inflation(self):
        """CSV with embedded `inflation` column auto-extracts cpi_map."""
        csv = (
            "tax_year,step,lower_bound,marginal_rate,indexing,inflation\n"
            "2022,1,0,0.05,1,0.030\n"
            "2022,2,40000,0.15,1,0.030\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv)
            p = f.name

        try:
            schedule = PITSchedule.from_csv(p)
            assert schedule.cpi_map[2022] == pytest.approx(0.030)
        finally:
            Path(p).unlink(missing_ok=True)

    def test_get_brackets_returns_compound_thresholds(self):
        """Compound-inflated thresholds are computed from inflated bounds."""
        schedule = PITSchedule.from_name(
            "BC_PIT_2014.csv",
            cpi_map={2014: 0.01, 2015: 0.01},
        )
        thresholds, rates, lower_bounds, quick_adds = schedule.get_brackets(
            tax_year=2016
        )
        # thresholds = lower_bounds[1:] + [inf]
        assert np.isinf(thresholds[-1])
        for i in range(len(thresholds) - 1):
            assert thresholds[i] == pytest.approx(lower_bounds[i + 1])
        # Quick-adds should be consistent with recomputed values
        expected_quick = _recompute_quick_add(lower_bounds, rates)
        assert np.allclose(quick_adds, expected_quick)
