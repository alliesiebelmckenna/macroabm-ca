"""Reader for the Canada/BC-specific scalar tax parameters.

This module loads the scalar tax parameters recorded in ``tax_parameters.yaml``
and applies them as an override onto a ``CentralGovernmentConfiguration``,
mirroring the ``read_country_conf`` pattern (read a YAML block, apply it via
``model_copy(update=...)``).

Scope is scalars only. Progressive bracket, tax-credit amount, and dividend
rate schedules live in their own CSV files and are read by ``PITSchedule`` /
``NRTCSchedule`` / ``DividendTaxCreditSchedule``. To enforce that boundary,
``read_tax_parameters`` rejects any schedule field appearing in the YAML.
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

# Scalars every jurisdiction starts from; its own block overrides field by field.
_DEFAULT_KEY = "default"

# Already-warned fallbacks, so each block warns once rather than once per year.
_FALLBACK_WARNED: set[tuple[str, str, int]] = set()

# Scalar fields on CentralGovernmentConfiguration that this file may override.
_ALLOWED_FIELDS = frozenset(
    {
        "pit_dividend_integration",
        "dividend_small_business_share",
        "bank_dividend_small_business_share",
        "couple_rental_income_split",
    }
)

# Schedule fields must NOT appear here; they come from the taxation CSVs.
_SCHEDULE_FIELDS = frozenset(
    {
        "pit_brackets",
        "pit_non_refundable_tax_credits",
        "dividend_eligible_gross_up",
        "dividend_non_eligible_gross_up",
        "dividend_eligible_dtc_rate",
        "dividend_non_eligible_dtc_rate",
    }
)


def _select_year(
    by_year: dict[int, Any],
    year: int,
    jurisdiction: str,
    yaml_path: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    """Return *jurisdiction*'s scalars for *year*, falling back to the latest prior year.

    These are modelling assumptions, not indexed figures, so carrying the most
    recent block forward is sound. An empty result means the block has nothing to
    say for this year — the caller decides whether that is an error.
    """
    if not by_year:
        return {}
    if year in by_year:
        return dict(by_year[year] or {})

    prior_years = [y for y in by_year if y <= year]
    if not prior_years:
        if required:
            raise KeyError(
                f"Year {year} precedes all available years for jurisdiction "
                f"'{jurisdiction}' in {yaml_path.name}. Available: {sorted(by_year)}"
            )
        return {}

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
    return dict(by_year[fallback_year] or {})


def read_tax_parameters(
    jurisdiction: str,
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

    # A bare `on:` key is a YAML boolean, so Ontario would parse as True and be silently unreachable.
    non_string_keys = [k for k in data if not isinstance(k, str)]
    if non_string_keys:
        raise ValueError(
            f"Non-string jurisdiction key(s) {non_string_keys} in {yaml_path.name}. "
            "A bare `on`, `no`, `yes` or `off` is parsed as a YAML boolean; quote "
            'the key (e.g. `"on":`) so it stays a jurisdiction.'
        )

    if jurisdiction not in data and _DEFAULT_KEY not in data:
        raise KeyError(
            f"Jurisdiction '{jurisdiction}' not found in {yaml_path.name}, and no "
            f"'{_DEFAULT_KEY}' block is present to fall back to. "
            f"Available: {sorted(data)}"
        )

    # Defaults fill any jurisdiction not listed, which lets a new province run untuned.
    base = _select_year(data.get(_DEFAULT_KEY) or {}, year, _DEFAULT_KEY, yaml_path, required=False)
    own = _select_year(data.get(jurisdiction) or {}, year, jurisdiction, yaml_path, required=False)

    if not base and not own:
        raise KeyError(
            f"Year {year} precedes all available years for jurisdiction "
            f"'{jurisdiction}' and for '{_DEFAULT_KEY}' in {yaml_path.name}."
        )

    overrides = {**base, **own}

    schedule_keys = _SCHEDULE_FIELDS.intersection(overrides)
    if schedule_keys:
        raise ValueError(
            f"Schedule field(s) {sorted(schedule_keys)} found in {yaml_path.name}. "
            "Bracket, credit, and dividend-rate schedules belong in their CSV "
            "files in the taxation directory under the configured data root, "
            "not in the scalar parameter file."
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
    jurisdiction: str,
    year: int = 2014,
    path: str | Path | None = None,
) -> CentralGovernmentConfiguration:
    """Return a copy of *configuration* with the scalar tax parameters applied.

    The schedule fields (``pit_brackets``, ``pit_non_refundable_tax_credits``) on
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
