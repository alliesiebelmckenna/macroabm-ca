"""Single seam for assembling a jurisdiction's central-government tax config.

This module is the one place that builds a fully-populated
``CentralGovernmentConfiguration`` for a real run, combining two sources of tax
inputs: the schedules carried by a loaded ``TaxationReader`` (progressive
brackets, companion tax-credit amounts, dividend gross-up / DTC rates) and the
scalar assumptions read from ``tax_parameters.yaml``. The result is consumed
unchanged by the existing ``Country.from_pickled_country`` flow.

Progressive PIT is opt-in: it changes government revenue relative to the
upstream flat rate, so when no ``TaxationReader`` is supplied the base (flat)
configuration is returned unchanged. Within that path, dividend integration is
switched on automatically when a dividend schedule is present, overriding the
``pit_dividend_integration`` switch in the YAML (which governs only when no
schedule is found).

Monetary fields (brackets, credit amounts and clawback bounds, deductions) are
returned in per-individual dollars; conversion to agent units happens later in
``country._scale_pit_policy``, so the builder must not pre-scale. Credits whose
eligibility the runtime credit pool cannot yet evaluate are skipped and logged
rather than applied to everyone, so the allow-list below must stay in step with
the ``credit`` branches in ``pit_pools._credit_amount``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from macromodel.configurations.central_government_configuration import (
    CentralGovernmentConfiguration,
    TaxCreditDef,
)

from .tax_parameters_reader import apply_tax_parameters

if TYPE_CHECKING:
    from macro_data.readers.taxation import TaxationReader
    from macro_data.readers.taxation.personal_income_tax.nrtc_schedule import (
        TaxCreditComponent,
    )

logger = logging.getLogger(__name__)

# Eligibility keys the runtime credit pool can act on; a credit using any other
# key is skipped. These map to the ``credit`` branches in ``_credit_amount``.
_EXPRESSIBLE_ELIGIBILITY_KEYS = frozenset(
    {"age_min", "in_couple_household", "is_single_parent"}
)


def _credit_component_to_def(component: TaxCreditComponent) -> Optional[TaxCreditDef]:
    """Map a data-layer ``TaxCreditComponent`` to a config-layer ``TaxCreditDef``.

    Returns ``None`` (and logs) when the component's eligibility uses rules the
    runtime credit pool cannot yet act on, so the caller can skip it instead of
    applying it universally.
    """
    extra_keys = set(component.eligibility) - _EXPRESSIBLE_ELIGIBILITY_KEYS
    if extra_keys:
        logger.warning(
            "Skipping tax credit '%s': eligibility rule(s) %s are not yet "
            "applied by the runtime credit pool (supported: universal, age, "
            "couple, single-parent). The credit is omitted to avoid applying it "
            "universally.",
            component.credit,
            sorted(extra_keys),
        )
        return None

    return TaxCreditDef(
        credit=component.credit,
        amount=component.amount,
        index=component.index,
        eligibility_age_min=component.eligibility.get("age_min"),
        clawback=component.clawback,
        top=component.top,
    )


def activate_taxation(
    base_config: CentralGovernmentConfiguration,
    taxation_reader: Optional["TaxationReader"],
    year: int,
    params_path: str | Path | None = None,
) -> CentralGovernmentConfiguration:
    """Layer a government's progressive PIT schedules onto its config, if opted in.

    The macromodel-side consumption seam for ``SyntheticCountry.taxation``: given
    one government's *base_config* and the reader for its taxing authority, it
    returns the progressive config when the government has opted in
    (``base_config.activate_progressive_pit``) and a reader is available, and the
    unchanged *base_config* (flat parity) otherwise. It is per-government and
    jurisdiction-keyed, reading the jurisdiction from
    ``taxation_reader.jurisdiction``, so a future model with several governments
    calls it once per government.

    Args:
        base_config: The government's configuration (its non-tax fields are
            preserved when schedules are layered on).
        taxation_reader: The taxation schedules for this government's authority,
            or ``None`` when the country carries no taxation data.
        year: Tax year for which to compute brackets / credits / dividend rates.
        params_path: Optional override for the scalar YAML file location.

    Returns:
        The progressive config when opted in with data present, else *base_config*.
    """
    if not base_config.activate_progressive_pit or taxation_reader is None:
        return base_config
    return build_central_government_configuration(
        taxation_reader,
        jurisdiction=taxation_reader.jurisdiction,
        year=year,
        params_path=params_path,
        base_config=base_config,
    )


def build_central_government_configuration(
    taxation_reader: "TaxationReader",
    jurisdiction: str,
    year: int = 2014,
    params_path: str | Path | None = None,
    base_config: Optional[CentralGovernmentConfiguration] = None,
) -> CentralGovernmentConfiguration:
    """Assemble a central-government configuration from loaded schedules + YAML scalars.

    The taxation schedules arrive as a :class:`~macro_data.readers.taxation.TaxationReader`
    built by ``DataReaders.from_raw_data`` (mirroring the energy-sector readers).
    When no reader is supplied (taxation data absent), progressive PIT is *not*
    activated and the base (flat) configuration is returned unchanged, preserving
    upstream parity.

    Args:
        taxation_reader: Loaded personal-income-tax schedules, or ``None`` when no
            taxation data is present.  ``None`` ⇒ the base config is returned
            unchanged (flat tax, no progressive PIT).
        jurisdiction: Jurisdiction key for the scalar-block lookup in
            ``tax_parameters.yaml``.
        year: Tax year for which to compute the (CPI-indexed) brackets and
            credit amounts, select the dividend rates, and key the scalar lookup.
        params_path: Optional override for the scalar YAML file location.
        base_config: Optional base configuration whose non-tax fields (functions,
            social benefits) are preserved.  Defaults to a fresh
            ``CentralGovernmentConfiguration()``.

    Returns:
        A ``CentralGovernmentConfiguration``: when a reader is supplied, with
        ``pit_brackets``, ``pit_non_refundable_tax_credits`` and the dividend gross-up / DTC
        rates populated from the schedules and the scalar fields overridden from
        the YAML; otherwise the unmodified base (flat) configuration.
    """
    base = base_config if base_config is not None else CentralGovernmentConfiguration()

    # No taxation data: progressive PIT is not activated, so return the base
    # (flat) configuration unchanged for upstream parity.
    if taxation_reader is None:
        return base

    # Brackets, plus the companion credit schedule carried by the reader.
    schedule = taxation_reader.pit_schedule
    _lowers, uppers, rates = schedule.get_brackets(year=year)
    pit_brackets = [(float(u), float(r)) for u, r in zip(uppers, rates)]

    # Map the companion tax credits, skipping the not-yet-expressible ones.
    pit_non_refundable_tax_credits: Optional[list[TaxCreditDef]] = None
    if schedule.non_refundable_tax_credits is not None:
        components = schedule.non_refundable_tax_credits.get_credits(year=year)
        mapped = [
            d for d in (_credit_component_to_def(c) for c in components) if d is not None
        ]
        pit_non_refundable_tax_credits = mapped or None

    # Dividend gross-up / DTC rates: their presence on the reader is the
    # activation signal; when absent, integration stays off.
    dividend_updates: dict = {}
    dividend_schedule_present = taxation_reader.dividend_schedule is not None
    if dividend_schedule_present:
        dividend_rates = taxation_reader.dividend_schedule.get_year_rates(
            year=year
        )
        eligible = dividend_rates["eligible"]
        non_eligible = dividend_rates["non_eligible"]
        dividend_updates = {
            "dividend_eligible_gross_up": eligible.gross_up_rate,
            "dividend_non_eligible_gross_up": non_eligible.gross_up_rate,
            "dividend_eligible_dtc_rate": eligible.dtc_pct_of_grossed_up,
            "dividend_non_eligible_dtc_rate": non_eligible.dtc_pct_of_grossed_up,
        }

    # Apply schedules, then the scalar overrides from the YAML.
    config = base.model_copy(
        update={
            "pit_brackets": pit_brackets,
            "pit_non_refundable_tax_credits": pit_non_refundable_tax_credits,
            **dividend_updates,
        }
    )
    config = apply_tax_parameters(
        config, jurisdiction=jurisdiction, year=year, path=params_path
    )
    # Applied after the YAML scalars so schedule presence wins over the YAML
    # switch (which governs only when no schedule is present).
    if dividend_schedule_present:
        config = config.model_copy(update={"pit_dividend_integration": True})
    return config
