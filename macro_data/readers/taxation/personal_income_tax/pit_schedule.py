"""Progressive Personal Income Tax (PIT) schedule computation.

This module provides three layers:

1. **Standalone functions** ``compute_progressive_tax`` /
   ``compute_progressive_tax_quick`` — low-level vectorized tax
   computation from arbitrary threshold/rate arrays.

2. **fetch_bc_cpi_inflation()** — retrieves BC all-items CPI inflation
   from Statistics Canada table 18-10-0005-01, computes year-over-year
   percentage changes, and caches them as a local CSV.

3. **PITSchedule class** — reads a local CSV containing bracket
   definitions for a *base tax year*, plus a separate CPI inflation
   map (from StatCan or a cached CSV).  When brackets are requested for
   a year beyond the base, lower bounds are compound-inflated using all
   intermediate annual CPI rates — only for brackets whose ``indexing``
   flag is 1.

CSV format (bracket definitions)
--------------------------------
The CSV must contain the following columns (case-insensitive)::

    tax_year      int     Base taxation year (e.g. 2014).
    step          int     Bracket ordinal (1, 2, 3, ...).
    lower_bound   float   Nominal lower income boundary for the base year.
    marginal_rate float   Tax rate applied to income within this bracket (0–1).
    indexing      int     Flag: 1 = inflate this boundary in later years;
                          0 = keep nominal.

An optional seventh column ``inflation`` may be included to supply the
annual CPI inflation rate directly in the CSV.  All rows for the same
``tax_year`` must share the same value.  When this column is present,
``from_csv`` automatically extracts it and no StatCan fetch is needed.
When absent, call :meth:`from_csv_with_cpi` to pull inflation from the
cached ``bc_cpi_inflation.csv`` or StatCan.

An optional eighth column ``basic_deduction`` supplies the basic
personal amount (non-refundable tax credit base) for the tax year.
All rows for the same ``tax_year`` must share the same value.  The
credit applied is ``basic_deduction × lowest_marginal_rate``, subtracted
after the progressive calculation.  When CPI-indexing is active, this
value is compound-inflated alongside the indexed lower bounds.

A row with ``step = k`` defines bracket *k* spanning
[lower_bound_k, lower_bound_{k+1}] (or [lower_bound_k, ∞) for the
highest step).

Example::

    tax_year,step,lower_bound,marginal_rate,indexing
    2014,1,0,0.0506,1
    2014,2,37606,0.077,1
    ...

Or with explicit inflation (no StatCan fetch needed)::

    tax_year,step,lower_bound,marginal_rate,indexing,inflation
    2014,1,0,0.0506,1,0.010
    2014,2,37606,0.077,1,0.010
    ...

CPI inflation map
-----------------
Inflation data is stored separately (not in the bracket CSV).  It is a
simple CSV with columns ``year`` and ``inflation``, where ``inflation`` is
the BC all-items CPI year-over-year change as a decimal (e.g. 0.018
for 1.8 %).  This file lives at ``spoof_data/freda/bc_cpi_inflation.csv``
and can be refreshed at any time via ``fetch_bc_cpi_inflation()``.

Compound inflation
------------------
When ``get_brackets(tax_year=T)`` is called and *T* is later than the
CSV's base year, each indexed lower bound is multiplied by::

    ∏_{y = base_year}^{T-1} (1 + inflation_map[y])

Non-indexed boundaries stay at their nominal base-year values.

Quick-add values are recomputed from the inflated bounds so they remain
self-consistent.

StatCan integration
-------------------
``fetch_bc_cpi_inflation()`` queries Statistics Canada table
**18-10-0005-01** (Consumer Price Index, monthly, not seasonally
adjusted) for:

* **Geography** = British Columbia
* **Products and product groups** = All-items
* **Reference period** = 2013-01 onward

It computes the December-over-December (annual average) inflation rate
for each calendar year and writes the result to the local cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from macro_data.readers.taxation.personal_income_tax.tax_credit_schedule import (
    TaxCreditSchedule,
)

# ── required CSV columns for bracket definitions ────────────────────
_REQUIRED_COLS = {
    "tax_year",       # int — base year these brackets apply to
    "step",           # int — bracket ordinal
    "lower_bound",    # float — nominal lower bound (base year)
    "marginal_rate",  # float — rate for this bracket
    "indexing",       # bool (0/1) — whether this bound is CPI-indexed
}

# ── paths ────────────────────────────────────────────────────────────
_PIT_SCHEDULE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "spoof_data" / "freda"
_CPI_CACHE_FILE = _PIT_SCHEDULE_DIR / "bc_cpi_inflation.csv"

# ── StatCan table details ────────────────────────────────────────────
_STATCAN_CPI_TABLE = "18-10-0005-01"
_STATCAN_CPI_START = "2013-01"

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Low-level vectorized computation
# =====================================================================

def compute_progressive_tax(
    incomes: np.ndarray,
    thresholds: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """Compute progressive tax using marginal rates on income slices.

    Income in each bracket [lower, upper] is taxed at the bracket's
    marginal rate.  Income exactly equal to a boundary is assigned
    to the **lower** bracket.  The last threshold should be
    ``np.inf`` to capture all remaining income.

    Args:
        incomes: Shape (n,) — taxable income per individual.
        thresholds: Shape (k,) — bracket *upper* bounds.  Must be
            strictly increasing; last entry conventionally ``np.inf``.
        rates: Shape (k,) — marginal tax rate for each bracket [0, 1].

    Returns:
        Shape (n,) — tax owed per individual.
    """
    _validate_brackets(thresholds, rates)

    tax = np.zeros_like(incomes, dtype=float)
    lower = 0.0
    for threshold, rate in zip(thresholds, rates):
        amount_in_bracket = np.clip(incomes, lower, threshold) - lower
        tax += rate * np.maximum(amount_in_bracket, 0.0)
        lower = threshold
    return tax


def compute_progressive_tax_quick(
    incomes: np.ndarray,
    lower_bounds: np.ndarray,
    marginal_rates: np.ndarray,
    quick_adds: np.ndarray,
) -> np.ndarray:
    """Compute progressive tax using pre-computed cumulative quick-add values.

    For each income *x*, find the highest bracket *b* where
    ``x >= lower_bounds[b]``, then::

        tax = quick_adds[b] + marginal_rates[b] * (x - lower_bounds[b])

    Args:
        incomes: Shape (n,) — taxable income per individual.
        lower_bounds: Shape (k,) — lower income boundary of each bracket.
        marginal_rates: Shape (k,) — marginal rate for each bracket.
        quick_adds: Shape (k,) — cumulative tax from all brackets
            below the current one.

    Returns:
        Shape (n,) — tax owed per individual.
    """
    if not (len(lower_bounds) == len(marginal_rates) == len(quick_adds)):
        raise ValueError(
            "lower_bounds, marginal_rates, and quick_adds must have the same length"
        )

    bracket_idx = np.searchsorted(lower_bounds, incomes, side="right") - 1
    bracket_idx = np.clip(bracket_idx, 0, len(lower_bounds) - 1)

    tax = quick_adds[bracket_idx] + marginal_rates[bracket_idx] * (
        incomes - lower_bounds[bracket_idx]
    )
    return np.maximum(tax, 0.0)


# =====================================================================
# 2. StatCan CPI inflation retrieval
# =====================================================================

def fetch_bc_cpi_inflation(
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> dict[int, float]:
    """Return BC all-items annual CPI inflation as ``{year: rate}``.

    On first call (or when *force_refresh* is ``True``) the function
    downloads table **18-10-0005-01** from Statistics Canada, filters
    for British Columbia / All-items, computes the December-over-
    December year-over-year change, and caches the result as a CSV.

    Subsequent calls read the cached file unless *force_refresh* is set.

    Args:
        cache_path: Where to read/write the cached CPI CSV.
            Defaults to ``spoof_data/freda/bc_cpi_inflation.csv``.
        force_refresh: If ``True``, always re-fetch from StatCan.

    Returns:
        Dict mapping calendar year (int) to BC all-items CPI inflation
        rate as a decimal (e.g. 0.018 for 1.8 %).

    Raises:
        ImportError: If ``stats_can`` is not installed.
        RuntimeError: If the StatCan table cannot be retrieved or
            filtered.
    """
    if cache_path is None:
        cache_path = _CPI_CACHE_FILE

    if not force_refresh and cache_path.exists():
        df = pd.read_csv(cache_path)
        return dict(zip(df["year"].astype(int), df["inflation"].astype(float)))

    try:
        from stats_can.sc import zip_table_to_dataframe
    except ImportError as exc:
        raise ImportError(
            "stats-can is required for live CPI fetching. "
            "Install it with: pip install stats-can"
        ) from exc

    import tempfile
    import zipfile

    # Download the zip to a temp directory so we can read the CSV
    # with proper encoding (stats-can v3 may mishandle BOM-quoted
    # column headers, producing NaT for REF_DATE).
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        logger.info("Fetching StatCan table %s ...", _STATCAN_CPI_TABLE)
        zip_table_to_dataframe(_STATCAN_CPI_TABLE, path=tmp_path)

        # Find the downloaded zip and read the CSV with utf-8-sig
        zips = list(tmp_path.glob("*.zip"))
        if not zips:
            raise RuntimeError(
                "StatCan download did not produce a zip file in %s" % tmpdir
            )
        zip_path = zips[0]
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv") and "MetaData" not in n]
            if not csv_names:
                raise RuntimeError("No data CSV found in zip: %s" % zf.namelist())
            raw = pd.read_csv(zf.open(csv_names[0]), encoding="utf-8-sig")

    # Strip any remaining quotes from column names (StatsCan CSVs
    # sometimes quote headers like \"REF_DATE\").
    raw.columns = raw.columns.str.strip('"')

    # We need: GEO = "British Columbia", Products = "All-items".
    geo_col = _find_column(raw, ["geo", "geography", "geographic", "geog"])
    product_col = _find_column(raw, [
        "products and product groups",
        "products",
        "product",
        "product groups",
        "products_and_product_groups",
    ])
    ref_col = _find_column(raw, ["ref_date", "refdate", "reference_period", "reference period"])
    value_col = _find_column(raw, ["value", "values", "val"])

    geo_mask = raw[geo_col].str.strip().str.lower() == "british columbia"
    product_mask = raw[product_col].str.strip().str.lower() == "all-items"

    bc_all = raw[geo_mask & product_mask].copy()

    if bc_all.empty:
        raise RuntimeError(
            "Could not find BC/All-items rows in table %s. "
            "Check that the column names match: %s" % (_STATCAN_CPI_TABLE, list(raw.columns))
        )

    bc_all[value_col] = pd.to_numeric(bc_all[value_col], errors="coerce")
    bc_all = bc_all.dropna(subset=[value_col])

    # --- Parse REF_DATE into year & optional month -----------------
    # StatsCan CSVs may have REF_DATE as:
    #   * plain years  (e.g. 1979)        → int / float columns
    #   * YYYY-MM      (e.g. "2023-01")   → object columns
    #   * YYYY-MM-DD   (e.g. "2023-01-01")
    ref_series = bc_all[ref_col]
    if pd.api.types.is_numeric_dtype(ref_series):
        # Already years — use directly.
        bc_all["year"] = ref_series.astype(int)
        # Annual data: one row per year → no month filtering needed.
        annual = bc_all.sort_values("year")
    else:
        ref_dt = pd.to_datetime(ref_series, errors="coerce")
        bc_all["year"] = ref_dt.dt.year
        # Try December-first; fall back to annual mean.
        dec_rows = bc_all[ref_dt.dt.month == 12]
        if len(dec_rows) > 0:
            annual = dec_rows.sort_values("year")
        else:
            annual = bc_all.groupby("year")[value_col].mean().reset_index()

    # Compute year-over-year inflation as pct_change of the CPI level.
    annual = annual.set_index("year")
    annual["inflation"] = annual[value_col].pct_change()

    result = annual["inflation"].dropna().to_dict()
    result = {int(k): float(v) for k, v in result.items()}

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"year": sorted(result), "inflation": [result[y] for y in sorted(result)]}
    ).to_csv(cache_path, index=False)
    logger.info("CPI inflation cached to %s", cache_path)

    return result


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first column name in *candidates* that exists (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    raise KeyError(
        f"None of {candidates} found in DataFrame columns: {list(df.columns)}"
    )


# =====================================================================
# 3. PITSchedule — CSV + CPI-backed multi-year schedule
# =====================================================================

class PITSchedule:
    """Progressive PIT schedule backed by a base-year CSV + CPI inflation map.

    Typical usage::

        # Load brackets + fetch live CPI from StatCan
        schedule = PITSchedule.from_csv_with_cpi("BC_PIT_2014.csv")

        # Or load brackets with a pre-computed CPI dict
        cpi = {2014: 0.01, 2015: 0.011, 2016: 0.018}
        schedule = PITSchedule.from_csv("BC_PIT_2014.csv", cpi_map=cpi)

        # Get brackets for 2016 — compound-inflated from 2014
        thresholds, rates = schedule.get_brackets(tax_year=2016)

        # Get nominal base-year brackets (no inflation)
        thresholds, rates = schedule.get_brackets(tax_year=2014)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cpi_map: Optional[dict[int, float]] = None,
        basic_deduction: Optional[float] = None,
        tax_credits: Optional["TaxCreditSchedule"] = None,
    ) -> None:
        self._df = df.copy()
        self._base_year: int = int(self._df["tax_year"].iloc[0])
        self._cpi_map: dict[int, float] = dict(cpi_map) if cpi_map else {}
        self._basic_deduction: Optional[float] = basic_deduction
        self._tax_credits: Optional["TaxCreditSchedule"] = tax_credits

    # ── factories ───────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        cpi_map: Optional[dict[int, float]] = None,
    ) -> "PITSchedule":
        """Load bracket definitions from a CSV file.

        Args:
            path: Path to the CSV file.
            cpi_map: Optional ``{year: inflation_rate}`` dict.  When
                provided, ``get_brackets`` can compound-inflate bounds
                for years beyond the base year.

        Returns:
            A configured ``PITSchedule`` instance.
        """
        df = pd.read_csv(Path(path))

        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)

        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found columns: {sorted(df.columns)}"
            )

        for col in ("tax_year", "step"):
            df[col] = df[col].astype(int)
        for col in ("lower_bound", "marginal_rate"):
            df[col] = df[col].astype(float)
        df["indexing"] = df["indexing"].astype(bool)

        # Optional inflation column: extract per-year CPI rates.
        inflation_col = None
        for c in df.columns:
            if c.lower() == "inflation":
                inflation_col = c
                break

        if inflation_col is not None and cpi_map is None:
            df[inflation_col] = df[inflation_col].astype(float)
            infl = df[["tax_year", inflation_col]].drop_duplicates()
            if infl["tax_year"].duplicated().any():
                raise ValueError(
                    "Inconsistent inflation values: multiple rows for the "
                    "same tax_year have different inflation values."
                )
            cpi_map = dict(zip(infl["tax_year"].astype(int), infl[inflation_col]))

        # Optional basic_deduction column: non-refundable tax credit base.
        basic_deduction: Optional[float] = None
        for c in df.columns:
            if c.lower() == "basic_deduction":
                raw_vals = df[c].astype(str).str.replace(",", "").astype(float)
                unique_vals = raw_vals.drop_duplicates()
                if len(unique_vals) > 1:
                    raise ValueError(
                        "Inconsistent basic_deduction values: multiple rows "
                        "for the same tax_year have different values."
                    )
                basic_deduction = float(unique_vals.iloc[0])
                break

        return cls(df, cpi_map=cpi_map, basic_deduction=basic_deduction)

    @classmethod
    def from_name(
        cls,
        filename: str,
        cpi_map: Optional[dict[int, float]] = None,
    ) -> "PITSchedule":
        """Load a schedule by filename from ``spoof_data/freda/``."""
        path = _PIT_SCHEDULE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Schedule file not found: {path}\n"
                f"Available: {sorted([p.name for p in _PIT_SCHEDULE_DIR.glob('*.csv')])}"
            )
        schedule = cls.from_csv(path, cpi_map=cpi_map)
        schedule._load_companion_tax_credits(path)
        return schedule

    @classmethod
    def from_name_with_cpi(
        cls,
        filename: str,
        force_refresh: bool = False,
    ) -> "PITSchedule":
        """Load a schedule by filename from ``spoof_data/freda/`` with CPI.

        Resolution order:
        1. If the CSV has an ``inflation`` column → use it (no fetch).
        2. Else if ``bc_cpi_inflation.csv`` exists → load from cache.
        3. Else → fetch live from StatCan table 18-10-0005-01.

        Args:
            filename: CSV filename (e.g. ``"BC_PIT_2014.csv"``).
            force_refresh: If ``True``, re-download CPI from StatCan
                (ignored when the CSV has an ``inflation`` column).

        Returns:
            A ``PITSchedule`` with CPI inflation data.
        """
        path = _PIT_SCHEDULE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Schedule file not found: {path}\n"
                f"Available: {sorted([p.name for p in _PIT_SCHEDULE_DIR.glob('*.csv')])}"
            )
        schedule = cls.from_csv_with_cpi(path, force_refresh=force_refresh)
        schedule._load_companion_tax_credits(path)
        return schedule

    @classmethod
    def from_csv_with_cpi(
        cls,
        path: str | Path,
        force_refresh: bool = False,
    ) -> "PITSchedule":
        """Load brackets from CSV with CPI inflation from cache or StatCan.

        Resolution order:
        1. If the CSV has an ``inflation`` column → use it (no fetch).
        2. Else if ``bc_cpi_inflation.csv`` exists → load from cache.
        3. Else → fetch live from StatCan table 18-10-0005-01.

        Pass *force_refresh=True* to skip the cache and always fetch
        from StatCan (unless the CSV itself carries an ``inflation``
        column, which always takes precedence).

        Args:
            path: Path to the bracket CSV.
            force_refresh: If ``True``, re-download CPI from StatCan
                (ignored when the CSV has an ``inflation`` column).

        Returns:
            A ``PITSchedule`` with CPI inflation data.
        """
        p = Path(path)

        # Peek at the CSV header to see if inflation is baked in.
        header = pd.read_csv(p, nrows=0)
        has_inflation = any(c.lower() == "inflation" for c in header.columns)

        if has_inflation:
            logger.info("CSV has an inflation column — using embedded values.")
            return cls.from_csv(p)

        cpi = fetch_bc_cpi_inflation(force_refresh=force_refresh)
        return cls.from_csv(p, cpi_map=cpi)

    # ── companion tax-credit file auto-discovery ────────────────────

    def _load_companion_tax_credits(self, bracket_path: Path) -> None:
        """Try to load a companion ``*_tax_credit_*.csv`` alongside the bracket CSV.

        Convention: ``BC_PIT_2014.csv`` → looks for ``BC_tax_credit_2014.csv``
        in the same directory.  The credit file is optional — when not found
        the model applies no tax credits (none configured).
        """
        stem = bracket_path.stem          # e.g. "BC_PIT_2014"
        parent_dir = bracket_path.parent

        # Derive companion name: replace "PIT" with "tax_credit"
        candidate_stem = stem.replace("PIT", "tax_credit")
        candidate = parent_dir / f"{candidate_stem}.csv"

        if not candidate.exists():
            # Also try without the suffix pattern
            logger.debug("No companion tax-credit file found (%s)", candidate)
            return

        logger.info("Loading companion tax-credit file: %s", candidate.name)
        self._tax_credits = TaxCreditSchedule.from_csv(
            candidate, cpi_map=dict(self._cpi_map)
        )

    # ── properties ──────────────────────────────────────────────────

    @property
    def base_year(self) -> int:
        """The base tax year from the CSV (all bounds are nominal for this year)."""
        return self._base_year

    @property
    def available_years(self) -> np.ndarray:
        """Sorted unique tax years present in the schedule."""
        return np.sort(self._df["tax_year"].unique())

    @property
    def cpi_map(self) -> dict[int, float]:
        """Annual BC CPI inflation rates ``{year: rate}``."""
        return dict(self._cpi_map)

    @property
    def basic_deduction(self) -> Optional[float]:
        """Basic personal amount (non-refundable credit base) for the base year.

        Returns ``None`` when the CSV did not include a ``basic_deduction``
        column.
        """
        return self._basic_deduction

    @property
    def tax_credits(self) -> Optional["TaxCreditSchedule"]:
        """Tax credit schedule (multi-component), or ``None`` if not loaded.

        When a companion ``*_tax_credit_*.csv`` file was found alongside
        the bracket CSV, it is parsed into a ``TaxCreditSchedule``.
        Otherwise this is ``None`` and no tax credits are applied.
        """
        return self._tax_credits

    def get_basic_deduction(self, tax_year: int) -> Optional[float]:
        """Return the CPI-inflated basic deduction for *tax_year*.

        When *tax_year* is later than :attr:`base_year`, the value is
        compound-inflated using the same factor applied to indexed
        lower bounds.  Returns ``None`` when no basic deduction is
        configured.

        Args:
            tax_year: The tax year to retrieve.

        Returns:
            Inflated basic deduction, or ``None``.

        Raises:
            ValueError: If CPI data for an intermediate year is missing.
        """
        if self._basic_deduction is None:
            return None
        if tax_year <= self.base_year:
            return self._basic_deduction

        factor = 1.0
        for y in range(self.base_year, tax_year):
            rate = self._cpi_map.get(y)
            if rate is None:
                raise ValueError(
                    f"Missing CPI inflation for year {y} "
                    f"(needed to inflate basic_deduction {self.base_year} → {tax_year}). "
                    f"Available years: {sorted(self._cpi_map)}"
                )
            factor *= 1.0 + rate
        return self._basic_deduction * factor

    # ── public methods ──────────────────────────────────────────────

    def get_brackets(
        self,
        tax_year: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return bracket arrays for *tax_year*, with compound CPI indexing.

        When *tax_year* is later than :attr:`base_year`, each indexed
        lower bound is multiplied by the product of
        ``(1 + inflation_y)`` for every intermediate year.  Non-indexed bounds stay at their
        nominal base-year values.  ``quick_add`` values are recomputed.

        Args:
            tax_year: The tax year to retrieve.

        Returns:
            Tuple ``(thresholds, rates, lower_bounds, quick_adds)``.

        Raises:
            ValueError: If *tax_year* is before the base year or CPI
                data for an intermediate year is missing.
        """
        base_df = self._df.sort_values("step")

        lower_bounds = base_df["lower_bound"].values.copy()
        marginal_rates = base_df["marginal_rate"].values.copy()
        indexing_flags = base_df["indexing"].values

        # Compound inflation from base_year through tax_year-1.
        if tax_year > self.base_year and indexing_flags.any():
            factor = 1.0
            for y in range(self.base_year, tax_year):
                rate = self._cpi_map.get(y)
                if rate is None:
                    raise ValueError(
                        f"Missing CPI inflation for year {y} "
                        f"(needed to inflate {self.base_year} → {tax_year}). "
                        f"Available years: {sorted(self._cpi_map)}"
                    )
                factor *= 1.0 + rate

            scale = np.where(indexing_flags, factor, 1.0)
            lower_bounds = lower_bounds * scale
        elif tax_year < self.base_year:
            raise ValueError(
                f"tax_year {tax_year} is before base year {self.base_year}"
            )

        quick_adds = _recompute_quick_add(lower_bounds, marginal_rates)
        thresholds = np.append(lower_bounds[1:].copy(), np.inf)

        return thresholds, marginal_rates, lower_bounds, quick_adds

    def compute_tax(
        self,
        incomes: np.ndarray,
        tax_year: int,
    ) -> np.ndarray:
        """Compute progressive tax for a given year (convenience wrapper)."""
        thresholds, rates, _, _ = self.get_brackets(tax_year)
        return compute_progressive_tax(incomes, thresholds, rates)


# =====================================================================
# Internal helpers
# =====================================================================

def _validate_brackets(thresholds: np.ndarray, rates: np.ndarray) -> None:
    """Check threshold / rate invariants."""
    if len(thresholds) != len(rates):
        raise ValueError(
            f"thresholds and rates must have the same length, "
            f"got {len(thresholds)} and {len(rates)}"
        )
    if len(thresholds) > 1 and not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly increasing")
    if np.any(rates < 0) or np.any(rates > 1):
        raise ValueError("rates must be in [0, 1]")


def _recompute_quick_add(
    lower_bounds: np.ndarray,
    marginal_rates: np.ndarray,
) -> np.ndarray:
    """Recompute quick-add values from lower_bounds and marginal_rates."""
    quick = np.zeros(len(lower_bounds), dtype=float)
    for i in range(1, len(lower_bounds)):
        quick[i] = quick[i - 1] + marginal_rates[i - 1] * (
            lower_bounds[i] - lower_bounds[i - 1]
        )
    return quick

