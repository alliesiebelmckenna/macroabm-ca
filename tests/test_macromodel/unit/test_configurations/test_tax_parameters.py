"""Tests for the BC/Canada tax scalar-parameter reader and config builder."""

import functools
import logging
import math
from pathlib import Path

import pytest

from macro_data.readers.taxation import TaxationReader
from macromodel.configurations import (
    CentralGovernmentConfiguration,
    activate_taxation,
    apply_tax_parameters,
    build_central_government_configuration,
    read_tax_parameters,
)
from macromodel.configurations.tax_parameters.tax_parameters_reader import (
    _ALLOWED_FIELDS,
)

# Committed schedules used as explicit test fixtures (the builder now consumes a
# loaded TaxationReader rather than resolving paths itself).
#   parents[0]=test_configurations [1]=unit [2]=test_macromodel [3]=tests [4]=repo root
_COMMITTED_PIT_DIR = (
    Path(__file__).resolve().parents[4]
    / "spoof_data" / "freda" / "personal_income_tax"
)


@functools.lru_cache(maxsize=1)
def _committed_reader() -> TaxationReader:
    """Load the committed BC schedules once (shared, read-only across tests)."""
    return TaxationReader.from_dir(_COMMITTED_PIT_DIR)


def _build(jurisdiction: str = "bc", tax_year: int = 2014, **kwargs):
    """Build a BC config from the committed schedules (the common test path)."""
    return build_central_government_configuration(
        _committed_reader(), jurisdiction, tax_year, **kwargs
    )


class TestReadTaxParameters:
    def test_returns_only_scalar_fields(self):
        overrides = read_tax_parameters("bc", 2014)
        assert set(overrides).issubset(_ALLOWED_FIELDS)
        # Schedule fields must never appear in the scalar file.
        assert "pit_brackets" not in overrides
        assert "pit_tax_credits" not in overrides

    def test_values_match_packaged_yaml(self):
        overrides = read_tax_parameters("bc", 2014)
        assert overrides["dividend_small_business_share"] == 0.90
        assert overrides["couple_rental_income_split"] == 0.5
        assert overrides["pit_dividend_integration"] is False

    def test_unknown_jurisdiction_raises(self):
        with pytest.raises(KeyError):
            read_tax_parameters("atlantis", 2014)

    def test_year_before_all_available_raises(self):
        # 1900 precedes every available block (2014+), so there is no prior
        # year to fall back to -> KeyError.
        with pytest.raises(KeyError):
            read_tax_parameters("bc", 1900)

    def test_absent_later_year_falls_back_to_latest_prior(self):
        # 2015 is not packaged (only 2014); its scalar assumptions fall back to
        # the latest prior year (2014) rather than raising.
        assert read_tax_parameters("bc", 2015) == read_tax_parameters("bc", 2014)

    def test_schedule_field_in_file_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("bc:\n  2014:\n    pit_brackets: [[1.0, 0.1]]\n")
        with pytest.raises(ValueError, match="Schedule field"):
            read_tax_parameters("bc", 2014, path=bad)

    def test_dividend_rate_field_in_file_is_rejected(self, tmp_path):
        """Dividend gross-up / DTC rates are schedule-sourced; the YAML rejects them."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("bc:\n  2014:\n    dividend_eligible_gross_up: 0.38\n")
        with pytest.raises(ValueError, match="Schedule field"):
            read_tax_parameters("bc", 2014, path=bad)

    def test_unknown_field_in_file_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("bc:\n  2014:\n    made_up_knob: 1.0\n")
        with pytest.raises(ValueError, match="Unrecognised"):
            read_tax_parameters("bc", 2014, path=bad)


class TestApplyTaxParameters:
    def test_overrides_scalars_preserves_schedules(self):
        base = CentralGovernmentConfiguration(
            pit_brackets=[(50000.0, 0.1), (math.inf, 0.2)],
            couple_rental_income_split=0.9,
        )
        applied = apply_tax_parameters(base, "bc", 2014)
        # Scalar overridden from the YAML ...
        assert applied.couple_rental_income_split == 0.5
        # ... while the schedule the caller set is left untouched.
        assert applied.pit_brackets == [(50000.0, 0.1), (math.inf, 0.2)]

    def test_parity_with_defaults(self):
        """The packaged scalars mirror the config defaults exactly, so applying
        them is a true no-op."""
        base = CentralGovernmentConfiguration()
        applied = apply_tax_parameters(base, "bc", 2014)
        assert applied.model_dump() == base.model_dump()


class TestBuildCentralGovernmentConfiguration:
    def test_brackets_from_schedule(self):
        config = _build("bc", 2014)
        assert config.pit_brackets[0] == (37606.0, 0.0506)
        # Top bracket is open-ended.
        assert math.isinf(config.pit_brackets[-1][0])
        assert config.pit_brackets[-1][1] == 0.168

    def test_only_runtime_applicable_credits_are_carried(self):
        config = _build("bc", 2014)
        kinds = {c.credit for c in config.pit_tax_credits}
        # Universal, age-based, and the household-composition credits the runtime
        # dispatches by kind (Spousal, Equivalent-To-Spouse) are carried.
        assert "Personal Amount" in kinds
        assert "Age Amount" in kinds
        assert "Spousal Amount" in kinds
        assert "Equivalent To Spouse Amount" in kinds
        # Pension Income Amount is NOT age-based (CPP may start 60-70 and the
        # credit covers non-CPP pension income); without pension-income data it
        # is deferred rather than proxied by age, so it must not be carried.
        assert "Pension Income Amount" not in kinds

    def test_household_credits_apply_through_build_to_run(self):
        """End-to-end: the Spousal and Equivalent-To-Spouse credits carried by
        the builder actually reduce tax through the runtime credit pool.  Age no
        longer gates them; the runtime dispatches by kind and tests household
        composition.  The contribution is isolated by differencing the same pool
        with and without the two credits."""
        import numpy as np

        from macromodel.agents.central_government.pit_pools import (
            PitContext,
            build_credit_base_pool,
        )
        from macromodel.agents.households.household_properties import HouseholdType

        config = _build("bc", 2014)
        spousal_amt = next(
            c.amount for c in config.pit_tax_credits if c.credit == "Spousal Amount"
        )
        equiv_amt = next(
            c.amount
            for c in config.pit_tax_credits
            if c.credit == "Equivalent To Spouse Amount"
        )

        # Convert the built credits into the runtime states-dict form (mirrors
        # CentralGovernment.from_synthetic).
        def to_defs(credits):
            return [
                {
                    "credit": t.credit,
                    "amount": t.amount,
                    "index": t.index,
                    "age_min": t.eligibility_age_min,
                    "clawback": t.clawback,
                    "top": t.top,
                }
                for t in credits
            ]

        # Individuals 0,1 are a couple (household 0); individual 2 is a single
        # parent (household 1).  Individual 1 has zero income, so individual 0
        # receives the full Spousal Amount.
        taxable = np.array([60000.0, 0.0, 40000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_age=np.array([40, 40, 40]),
            individuals_corr_households=np.array([0, 0, 1]),
            households_type=np.array(
                [
                    HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,
                    HouseholdType.SINGLE_PARENT_WITH_CHILDREN,
                ]
            ),
            households_n_adults=np.array([2, 1]),
        )

        all_defs = to_defs(config.pit_tax_credits)
        wo_defs = [
            d
            for d in all_defs
            if d["credit"] not in {"Spousal Amount", "Equivalent To Spouse Amount"}
        ]
        delta = build_credit_base_pool(all_defs, taxable, ctx) - build_credit_base_pool(
            wo_defs, taxable, ctx
        )

        # Individual 0: spouse (individual 1) has zero income -> full Spousal.
        assert delta[0] == pytest.approx(spousal_amt)
        # Individual 1: spouse (individual 0) earns 60k > amount -> Spousal zero.
        assert delta[1] == pytest.approx(0.0)
        # Individual 2: single parent -> Equivalent-To-Spouse amount.
        assert delta[2] == pytest.approx(equiv_amt)

    def test_age_credit_carries_age_and_clawback(self):
        config = _build("bc", 2014)
        age = next(c for c in config.pit_tax_credits if c.credit == "Age Amount")
        assert age.eligibility_age_min == 65
        assert age.clawback == 32943.0
        assert age.top == 62450.0

    def test_scalars_applied(self):
        config = _build("bc", 2014)
        assert config.dividend_small_business_share == 0.90
        assert config.couple_rental_income_split == 0.5

    def test_dividend_rates_from_schedule(self):
        """The gross-up / DTC rates come from the dividend schedule CSV."""
        config = _build("bc", 2014)
        assert config.dividend_eligible_gross_up == pytest.approx(0.38)
        assert config.dividend_non_eligible_gross_up == pytest.approx(0.18)
        assert config.dividend_eligible_dtc_rate == pytest.approx(0.10)
        assert config.dividend_non_eligible_dtc_rate == pytest.approx(0.0259)

    def test_dividend_integration_activates_when_schedule_present(self):
        """Presence of the dividend schedule switches integration on
        automatically, overriding the (False) YAML fallback switch."""
        config = _build("bc", 2014)
        assert config.pit_dividend_integration is True

    def test_dividend_integration_off_when_schedule_absent(self):
        """A reader with no dividend schedule leaves integration off (legacy
        treatment) instead of failing; the bracket schedule still builds."""
        reader = TaxationReader(
            pit_schedule=_committed_reader().pit_schedule, dividend_schedule=None
        )
        config = build_central_government_configuration(reader, "bc", 2014)
        assert config.pit_dividend_integration is False
        # The rest of the configuration is unaffected.
        assert config.pit_brackets[0] == (37606.0, 0.0506)

    def test_brackets_not_pre_scaled(self):
        """Country.from_pickled_country applies the agent-scale; the builder must
        return per-individual units, not pre-scaled ones."""
        config = _build("bc", 2014)
        # 37,606 is the per-individual first threshold, not 37,606 * scale.
        assert config.pit_brackets[0][0] == 37606.0

    def test_does_not_mutate_default_path(self):
        """Building the BC config must not change a fresh default config."""
        default = CentralGovernmentConfiguration()
        _build("bc", 2014)
        assert default.pit_brackets is None

    def test_dividend_schedule_per_year_lookup(self):
        """The dividend schedule returns a later year's OWN published rates,
        distinct from 2014's (0.10 DTC / 0.18 non-eligible gross-up). 2019 is a
        real per-year lookup, not a projection."""
        rates = _committed_reader().dividend_schedule.get_year_rates(tax_year=2019)
        assert rates["eligible"].dtc_pct_of_grossed_up == pytest.approx(0.12)
        assert rates["non_eligible"].gross_up_rate == pytest.approx(0.15)

    def test_build_out_of_table_bracket_year_raises(self):
        """The builder is lookup-only on brackets: a year the bracket schedule
        does not publish raises rather than projecting — there is no forward
        projection anywhere in the pipeline, and in production the builder is
        only ever called with published years.  The consolidated fixture
        publishes 2014-2030, so 2099 is out of table."""
        with pytest.raises(ValueError, match="not in the published schedule"):
            _build("bc", 2099)


class TestDeferredCreditSafety:
    """Credits the runtime cannot yet express must be DEFERRED — skipped by the
    builder — never applied universally.  Covers both explicitly-registered
    deferred kinds and unknown kinds (the unmapped fallback).  Without this, the
    expanded historical CSV would grant e.g. the Disability Amount to every
    individual at full value."""

    @staticmethod
    def _load_one(kind: str, tmp_path: Path, amount: str = "1000"):
        from macro_data.readers.taxation.personal_income_tax.tax_credit_schedule import (
            TaxCreditSchedule,
        )
        csv = tmp_path / "tc.csv"
        csv.write_text(
            "tax_year,geo,credit,amount,top,rate,clawback,clawback_rate,index\n"
            f"2014,BC,{kind},{amount},,0.0506,,,1\n"
        )
        return TaxCreditSchedule.from_csv(csv).credits[0]

    def test_unknown_kind_is_deferred_not_universal(self, tmp_path):
        """An unmapped credit kind must NOT get the empty (universal) eligibility
        dict; it gets a non-expressible deferred marker instead."""
        c = self._load_one("Totally Made Up Credit", tmp_path)
        assert c.eligibility != {}  # the old behaviour — would be universal
        assert c.eligibility == {"_deferred_unmapped": True}

    def test_unknown_kind_dropped_by_builder(self, tmp_path):
        from macromodel.configurations.tax_parameters.central_government_builder import (
            _credit_component_to_def,
        )
        c = self._load_one("Totally Made Up Credit", tmp_path)
        assert _credit_component_to_def(c) is None  # not carried to the runtime

    @pytest.mark.parametrize(
        "kind",
        [
            "B.C. Caregiver Amount",
            "Disability Amount",
            "Disability Amount (Child)",
            "Adoption Amount",
            "Volunteer Firefighter Amount",
            "Medical Expense Amount",
            "BC Tax Reduction Credit",
        ],
    )
    def test_registered_deferred_credit_dropped_by_builder(self, kind, tmp_path):
        """Every currently-live credit the model cannot yet compute is registered
        deferred and skipped by the builder, so loading the full historical CSV
        never grants them universally."""
        from macromodel.configurations.tax_parameters.central_government_builder import (
            _credit_component_to_def,
        )
        assert _credit_component_to_def(self._load_one(kind, tmp_path)) is None

    def test_active_credits_still_carried(self, tmp_path):
        """The four implemented credits are NOT swept up by the deferral — each
        still maps to a runtime def."""
        from macromodel.configurations.tax_parameters.central_government_builder import (
            _credit_component_to_def,
        )
        for kind in (
            "Personal Amount",
            "Age Amount",
            "Spousal Amount",
            "Equivalent To Spouse Amount",
        ):
            assert _credit_component_to_def(self._load_one(kind, tmp_path)) is not None


class TestNoTaxationData:
    """With no taxation reader supplied (taxation data absent), the builder
    returns the base (flat) configuration unchanged — progressive PIT is not
    activated, preserving upstream parity."""

    def test_none_reader_returns_flat_base(self):
        config = build_central_government_configuration(None, "bc", 2014)
        assert config.pit_brackets is None
        assert config.pit_tax_credits is None
        assert config.pit_dividend_integration is False

    def test_none_reader_preserves_supplied_base(self):
        base = CentralGovernmentConfiguration(couple_rental_income_split=0.123)
        config = build_central_government_configuration(
            None, "bc", 2014, base_config=base
        )
        # Returned unchanged (no schedules, no scalar overrides applied).
        assert config is base
        assert config.couple_rental_income_split == 0.123


class TestActivateTaxation:
    """The per-government, jurisdiction-keyed consumption seam: opt in + a reader
    present ⇒ progressive config; otherwise the base config is returned unchanged
    (flat parity).  Jurisdiction comes from the reader, never hardcoded."""

    def test_opted_in_with_reader_builds_progressive(self):
        base = CentralGovernmentConfiguration(activate_progressive_pit=True)
        config = activate_taxation(base, _committed_reader(), tax_year=2014)
        assert config is not base  # a new, progressive config
        assert config.pit_brackets[0] == (37606.0, 0.0506)
        assert config.pit_dividend_integration is True

    def test_not_opted_in_returns_base_unchanged(self):
        """Reader present but the government did not opt in ⇒ flat parity."""
        base = CentralGovernmentConfiguration(activate_progressive_pit=False)
        config = activate_taxation(base, _committed_reader(), tax_year=2014)
        assert config is base
        assert config.pit_brackets is None

    def test_opted_in_without_reader_returns_base(self):
        """Opted in but the country carries no taxation data ⇒ flat parity."""
        base = CentralGovernmentConfiguration(activate_progressive_pit=True)
        config = activate_taxation(base, None, tax_year=2014)
        assert config is base
        assert config.pit_brackets is None

    def test_jurisdiction_taken_from_reader_not_hardcoded(self):
        """The jurisdiction used for the YAML scalar lookup comes from the
        reader, so a non-bc reader would not silently use bc."""
        base = CentralGovernmentConfiguration(activate_progressive_pit=True)
        reader = _committed_reader()
        assert reader.jurisdiction == "bc"
        # A reader with an unknown jurisdiction raises on the scalar lookup,
        # proving the jurisdiction is read from the reader (not hardcoded "bc").
        mismatched = TaxationReader(
            pit_schedule=reader.pit_schedule,
            dividend_schedule=reader.dividend_schedule,
            jurisdiction="atlantis",
        )
        with pytest.raises(KeyError):
            activate_taxation(base, mismatched, tax_year=2014)


class TestFallbackWarnOnce:
    def test_fallback_warns_once_per_block(self, tmp_path, caplog):
        """The per-year schedule table requests every published year past the
        last scalar block; the fallback warning fires once per (file,
        jurisdiction, block), not once per requested year."""
        yaml_file = tmp_path / "params.yaml"
        yaml_file.write_text(
            "bc:\n  2014:\n    couple_rental_income_split: 0.5\n"
        )
        with caplog.at_level(
            logging.WARNING,
            logger="macromodel.configurations.tax_parameters.tax_parameters_reader",
        ):
            first = read_tax_parameters("bc", 2015, path=yaml_file)
            second = read_tax_parameters("bc", 2016, path=yaml_file)

        # Both requests still resolve to the fallback block.
        assert first == second == {"couple_rental_income_split": 0.5}
        fallback_records = [
            r for r in caplog.records if "falling back" in r.getMessage()
        ]
        assert len(fallback_records) == 1
