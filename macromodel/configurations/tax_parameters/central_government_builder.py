"""Single seam for assembling a jurisdiction's central-government tax config.

This module is the one place that builds a fully-populated
:class:`~macromodel.configurations.central_government_configuration.CentralGovernmentConfiguration`
for a real run.  It combines the two sources of tax inputs the project adds:

1. **Schedules** (multi-row, year-ranged tables) supplied as a loaded
   :class:`~macro_data.readers.taxation.TaxationReader` — the progressive bracket
   schedule, its companion non-refundable tax-credit amounts, and the dividend
   gross-up / DTC rates.  The reader is built by ``DataReaders.from_raw_data``
   from ``raw_data_path / "taxation" / "personal_income_tax"`` (mirroring the
   energy-sector readers); the builder consumes it rather than resolving paths
   itself.
2. **Scalars** (single-value assumptions) read from ``tax_parameters.yaml`` via
   :func:`apply_tax_parameters` — the dividend-integration switch, the
   small-business share, and the couple rental-income split.

The result is a configuration object the existing ``Country.from_pickled_country``
flow consumes unchanged: it reads ``pit_brackets`` (and scales the thresholds to
agent units), ``pit_tax_credits``, and the scalar fields.

Opt-in by design
----------------
Supplying a ``TaxationReader`` activates the progressive PIT schedule, which
changes government revenue relative to the upstream flat-rate behaviour.  It is
therefore *opt-in*: when no reader is supplied (taxation data absent) the base
(flat) configuration is returned unchanged, and a scenario must pass a reader to
switch BC taxation on.

Dividend integration activates on schedule presence
---------------------------------------------------
Within this opt-in path, dividend integration (the Canadian gross-up + dividend
tax credit) is switched on automatically when a dividend tax credit schedule
CSV is present: the builder loads its rates and sets
``pit_dividend_integration`` to ``True``.  When the schedule is absent the
builder leaves integration off and firm dividends keep the legacy treatment.
Schedule presence overrides the ``pit_dividend_integration`` switch in
``tax_parameters.yaml``; that YAML value governs only when no schedule is found.

Monetary units
--------------
``pit_brackets``, credit amounts and clawback bounds, and
``pit_taxable_income_deductions`` are returned in *per-individual* dollar
units.  The conversion to agent-level units (each synthetic agent represents
``scale`` people) is applied later, model-side, by
``country._scale_pit_policy`` — driven by the ``unit`` declarations on the
configuration fields — and the builder must not pre-scale.

Tax-credit coverage
-------------------
The runtime credit pool (``pit_pools._credit_amount``) dispatches by credit
``kind`` and already implements universal, age-based, couple (Spousal Amount,
income-tested) and single-parent (Equivalent To Spouse Amount) credits, deriving
the household context per individual.  This builder therefore carries those
kinds.  Credits whose eligibility the runtime does not yet evaluate are
*skipped* and logged rather than silently applied to everyone.  The allow-list
below must stay in step with the ``kind`` branches in ``_credit_amount``: a key
is admitted here only once the runtime can apply the credit it gates.
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
    from macro_data.readers.taxation.personal_income_tax.tax_credit_schedule import (
        TaxCreditComponent,
    )

logger = logging.getLogger(__name__)

# Eligibility keys the runtime credit pool can act on.  A credit whose
# eligibility dict contains any other key is not yet applicable at runtime and is
# skipped.  These map to the ``kind`` branches in ``pit_pools._credit_amount``:
#   age_min            -> Age Amount (and other age-gated credits)
#   in_couple_household -> Spousal Amount (income-tested against the spouse)
#   is_single_parent   -> Equivalent To Spouse Amount
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
            component.kind,
            sorted(extra_keys),
        )
        return None

    return TaxCreditDef(
        kind=component.kind,
        amount=component.amount,
        indexing=component.indexing,
        eligibility_age_min=component.eligibility.get("age_min"),
        clawback_start=component.clawback_start,
        clawback_cap=component.clawback_cap,
    )


def activate_taxation(
    base_config: CentralGovernmentConfiguration,
    taxation_reader: Optional["TaxationReader"],
    tax_year: int,
    params_path: str | Path | None = None,
) -> CentralGovernmentConfiguration:
    """Layer a government's progressive PIT schedules onto its config, if opted in.

    This is the macromodel-side consumption seam for ``SyntheticCountry.taxation``:
    given one government's *base_config* and the taxation reader for its taxing
    authority, it returns the progressive config when the government has opted in
    (``base_config.activate_progressive_pit``) and a reader is available, and the
    unchanged *base_config* (flat parity) otherwise.

    It is deliberately **per-government and jurisdiction-keyed** — it acts on a
    single config and reads the jurisdiction from ``taxation_reader.jurisdiction``
    rather than assuming a particular government.  A future model with multiple
    government agents calls this once per government, each with its own config and
    its jurisdiction's reader; nothing here presumes a single central government.

    Args:
        base_config: The government's configuration (its non-tax fields are
            preserved when schedules are layered on).
        taxation_reader: The taxation schedules for this government's authority,
            or ``None`` when the country carries no taxation data.
        tax_year: Tax year for which to compute brackets / credits / dividend rates.
        params_path: Optional override for the scalar YAML file location.

    Returns:
        The progressive config when opted in with data present, else *base_config*.
    """
    if not base_config.activate_progressive_pit or taxation_reader is None:
        return base_config
    return build_central_government_configuration(
        taxation_reader,
        jurisdiction=taxation_reader.jurisdiction,
        tax_year=tax_year,
        params_path=params_path,
        base_config=base_config,
    )


def build_central_government_configuration(
    taxation_reader: Optional["TaxationReader"] = None,
    jurisdiction: str = "bc",
    tax_year: int = 2014,
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
        tax_year: Tax year for which to compute the (CPI-indexed) brackets and
            credit amounts, select the dividend rates, and key the scalar lookup.
        params_path: Optional override for the scalar YAML file location.
        base_config: Optional base configuration whose non-tax fields (functions,
            social benefits) are preserved.  Defaults to a fresh
            ``CentralGovernmentConfiguration()``.

    Returns:
        A ``CentralGovernmentConfiguration``: when a reader is supplied, with
        ``pit_brackets``, ``pit_tax_credits`` and the dividend gross-up / DTC
        rates populated from the schedules and the scalar fields overridden from
        the YAML; otherwise the unmodified base (flat) configuration.
    """
    base = base_config if base_config is not None else CentralGovernmentConfiguration()

    # No taxation data ⇒ progressive PIT is not activated; return the base (flat)
    # configuration unchanged.  This keeps the default path at upstream parity.
    if taxation_reader is None:
        return base

    # ── Brackets (and the companion credit schedule carried by the reader) ──
    schedule = taxation_reader.pit_schedule
    thresholds, rates, _, _ = schedule.get_brackets(tax_year=tax_year)
    pit_brackets = [(float(t), float(r)) for t, r in zip(thresholds, rates)]

    # ── Map the companion tax credits, skipping the not-yet-expressible ones ──
    pit_tax_credits: Optional[list[TaxCreditDef]] = None
    if schedule.tax_credits is not None:
        components = schedule.tax_credits.get_credits(tax_year=tax_year)
        mapped = [
            d for d in (_credit_component_to_def(c) for c in components) if d is not None
        ]
        pit_tax_credits = mapped or None

    # ── Dividend gross-up / DTC rates: present on the reader ⇒ integration on ──
    # The dividend schedule's presence is the activation signal; when it is absent
    # integration stays off and firm dividends keep the legacy treatment.
    dividend_updates: dict = {}
    dividend_schedule_present = taxation_reader.dividend_schedule is not None
    if dividend_schedule_present:
        dividend_rates = taxation_reader.dividend_schedule.get_year_rates(
            tax_year=tax_year
        )
        eligible = dividend_rates["eligible"]
        non_eligible = dividend_rates["non_eligible"]
        dividend_updates = {
            "dividend_eligible_gross_up": eligible.gross_up_rate,
            "dividend_non_eligible_gross_up": non_eligible.gross_up_rate,
            "dividend_eligible_dtc_rate": eligible.dtc_rate_of_grossed_up,
            "dividend_non_eligible_dtc_rate": non_eligible.dtc_rate_of_grossed_up,
        }

    # ── Apply schedules, then the scalar overrides from the YAML ──
    config = base.model_copy(
        update={
            "pit_brackets": pit_brackets,
            "pit_tax_credits": pit_tax_credits,
            **dividend_updates,
        }
    )
    config = apply_tax_parameters(
        config, jurisdiction=jurisdiction, year=tax_year, path=params_path
    )
    # The dividend schedule's presence activates integration.  This is applied
    # AFTER the YAML scalars so schedule-presence wins over the YAML switch (the
    # YAML value governs only when no schedule is present).
    if dividend_schedule_present:
        config = config.model_copy(update={"pit_dividend_integration": True})
    return config
