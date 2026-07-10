"""Reader for the Canada/BC-specific tax scalar parameters.

This module loads the scalar tax parameters recorded in
``tax_parameters.yaml`` and applies them as an override onto a
:class:`~macromodel.configurations.central_government_configuration.CentralGovernmentConfiguration`.

It mirrors the pattern used by ``read_country_conf`` in
``macro_data.configuration_utils``: read a YAML block and apply it via
``model_copy(update=...)``.  The configuration class remains the schema and
validation layer; this reader only supplies jurisdiction- and year-specific
values from an external, easily editable file.

Scope is *scalars only*.  Progressive bracket schedules, tax-credit amount
schedules, and dividend gross-up / DTC rate schedules are deliberately
excluded -- they live in their own CSV files in the taxation directory
(``raw_data_path / "taxation"``, with ``spoof_data/freda`` as the committed
fallback) and are read by ``PITSchedule`` / ``TaxCreditSchedule`` /
``DividendTaxCreditSchedule``.  To enforce that boundary,
:func:`read_tax_parameters` rejects any schedule field appearing in the YAML.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from macromodel.configurations.central_government_configuration import (
    CentralGovernmentConfiguration,
)

logger = logging.getLogger(__name__)

_TAX_PARAMS_PATH = Path(__file__).parent / "tax_parameters.yaml"

# (yaml path, jurisdiction, fallback year) combinations already warned about.
# The per-year schedule table requests every published year and each miss
# falls back to the same block — one warning per fallback block, not one per
# requested year.
_FALLBACK_WARNED: set[tuple[str, str, int]] = set()

# Scalar fields on CentralGovernmentConfiguration that this file may override.
_ALLOWED_FIELDS = frozenset(
    {
        "pit_dividend_integration",
        "dividend_small_business_share",
        "bank_dividend_small_business_share",
        "couple_rental_income_split",
        "pit_taxable_income_deductions",
    }
)

# Schedule fields that must NOT appear here -- they are sourced from CSVs in the
# taxation directory (raw_data_path/"taxation", spoof_data/freda fallback): PIT
# brackets and credit amounts via PITSchedule / TaxCreditSchedule; dividend
# gross-up / DTC rates via DividendTaxCreditSchedule.
_SCHEDULE_FIELDS = frozenset(
    {
        "pit_brackets",
        "pit_tax_credits",
        "dividend_eligible_gross_up",
        "dividend_non_eligible_gross_up",
        "dividend_eligible_dtc_rate",
        "dividend_non_eligible_dtc_rate",
    }
)


def read_tax_parameters(
    jurisdiction: str = "bc",
    year: int = 2014,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the scalar tax-parameter overrides for *jurisdiction* / *year*.

    Args:
        jurisdiction: Top-level key in the YAML (e.g. ``"bc"``).
        year: Tax year key nested under the jurisdiction.
        path: Optional override for the YAML file location.  Defaults to the
            packaged ``tax_parameters.yaml``.

    Returns:
        A dict mapping ``CentralGovernmentConfiguration`` field names to values,
        suitable for ``model_copy(update=...)``.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If *jurisdiction* is absent, or if *year* precedes every
            available year for the jurisdiction (no prior block to fall back
            to).  A *year* later than the available blocks falls back to the
            latest prior year's scalars, with a warning.
        ValueError: If a schedule field (e.g. ``pit_brackets``) appears in the
            block, or an unrecognised field name is present.
    """
    yaml_path = Path(path) if path is not None else _TAX_PARAMS_PATH
    if not yaml_path.exists():
        raise FileNotFoundError(f"Tax parameter file not found: {yaml_path}")

    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file) or {}

    if jurisdiction not in data:
        raise KeyError(
            f"Jurisdiction '{jurisdiction}' not found in {yaml_path.name}. "
            f"Available: {sorted(data)}"
        )
    by_year = data[jurisdiction] or {}
    if year not in by_year:
        # Fall back to the latest available year not exceeding the request.
        # The packaged 2014 block is a demonstration default; a real run would
        # supply the requested year's scalars.  When they are absent, reuse the
        # latest prior year's assumptions rather than failing -- these are
        # modelling assumptions (small-business share, rental split, the
        # integration switch), not inflation-indexed figures, so carrying them
        # forward is sound.  (Schedules -- brackets, credit amounts, dividend
        # rates -- are already year-aware via their own per-year CSV rows, so
        # only these scalars need a fallback.)  A year preceding every available
        # block stays an error.
        prior_years = [y for y in by_year if y <= year]
        if not prior_years:
            raise KeyError(
                f"Year {year} precedes all available years for jurisdiction "
                f"'{jurisdiction}' in {yaml_path.name}. Available: {sorted(by_year)}"
            )
        fallback_year = max(prior_years)
        warn_key = (str(yaml_path), jurisdiction, fallback_year)
        if warn_key not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(warn_key)
            logger.warning(
                "Tax-parameter scalars for %s %d not found in %s; falling back "
                "to the latest available year %d (logged once per fallback "
                "block; later requests reuse it silently).",
                jurisdiction,
                year,
                yaml_path.name,
                fallback_year,
            )
        year = fallback_year

    overrides = by_year[year] or {}

    schedule_keys = _SCHEDULE_FIELDS.intersection(overrides)
    if schedule_keys:
        raise ValueError(
            f"Schedule field(s) {sorted(schedule_keys)} found in {yaml_path.name}. "
            "Bracket, credit, and dividend-rate schedules belong in their CSV "
            "files in the taxation directory (raw_data_path/'taxation', "
            "spoof_data/freda fallback), not in the scalar parameter file."
        )

    unknown_keys = set(overrides) - _ALLOWED_FIELDS
    if unknown_keys:
        raise ValueError(
            f"Unrecognised tax-parameter field(s) {sorted(unknown_keys)} in "
            f"{yaml_path.name}. Allowed: {sorted(_ALLOWED_FIELDS)}"
        )

    return dict(overrides)


def apply_tax_parameters(
    configuration: CentralGovernmentConfiguration,
    jurisdiction: str = "bc",
    year: int = 2014,
    path: str | Path | None = None,
) -> CentralGovernmentConfiguration:
    """Return a copy of *configuration* with the scalar tax parameters applied.

    The schedule fields (``pit_brackets``, ``pit_tax_credits``) on
    *configuration* are left untouched -- only the scalar fields listed in the
    YAML block are overridden.

    Args:
        configuration: The base central-government configuration.
        jurisdiction: Top-level key in the YAML (e.g. ``"bc"``).
        year: Tax year key nested under the jurisdiction.
        path: Optional override for the YAML file location.

    Returns:
        A new ``CentralGovernmentConfiguration`` with the overrides applied.
    """
    overrides = read_tax_parameters(jurisdiction=jurisdiction, year=year, path=path)
    return configuration.model_copy(update=overrides)
