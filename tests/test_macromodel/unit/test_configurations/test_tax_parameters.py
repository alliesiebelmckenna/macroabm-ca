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
    return TaxationReader.from_dir(_COMMITTED_PIT_DIR, jurisdiction="bc")


def _build(jurisdiction: str = "bc", year: int = 2014, **kwargs):
    """Build a BC config from the committed schedules (the common test path)."""
    return build_central_government_configuration(
        _committed_reader(), jurisdiction, year, **kwargs
    )


class TestReadTaxParameters:




    def test_absent_later_year_falls_back_to_latest_prior(self):
        # 2015 is not packaged (only 2014); its scalar assumptions fall back to
        # the latest prior year (2014) rather than raising.
        assert read_tax_parameters("bc", 2015) == read_tax_parameters("bc", 2014)

    def test_schedule_field_in_file_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("bc:\n  2014:\n    pit_brackets: [[1.0, 0.1]]\n")
        with pytest.raises(ValueError, match="Schedule field"):
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



class TestBuildCentralGovernmentConfiguration:


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
            c.amount for c in config.pit_non_refundable_tax_credits if c.credit == "Spousal Amount"
        )
        equiv_amt = next(
            c.amount
            for c in config.pit_non_refundable_tax_credits
            if c.credit == "Equivalent To Spouse Amount"
        )

        # Convert the built credits into the runtime states-dict form using the
        # production helper itself, so this test cannot drift from the shape the
        # agent actually builds.
        from macromodel.agents.central_government.central_government import (
            pit_credit_defs_to_state_dicts as to_defs,
        )

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
        )

        all_defs = to_defs(config.pit_non_refundable_tax_credits)
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









    def test_build_out_of_table_bracket_year_raises(self):
        """The builder is lookup-only on brackets: a year the bracket schedule
        does not publish raises rather than projecting — there is no forward
        projection anywhere in the pipeline, and in production the builder is
        only ever called with published years.  The consolidated fixture
        publishes 2014-2030, so 2099 is out of table."""
        with pytest.raises(ValueError, match="No PIT data available"):
            _build("bc", 2099)


class TestDeferredCreditSafety:
    """Credits the runtime cannot express must be skipped by the builder, never
    applied universally.  Since the dormant kinds were unregistered from
    ``_ELIGIBILITY_RULES``, every one of them reaches the builder as an *unmapped*
    credit, so that single fallback is now the only thing standing between the
    published schedule and a universal grant.  The parametrised case below pins
    it against the real credit names the CSV still publishes — without it, the
    historical CSV would grant e.g. the Disability Amount to every individual at
    full value."""

    @staticmethod
    def _load_one(kind: str, tmp_path: Path, amount: str = "1000"):
        from macro_data.readers.taxation.personal_income_tax.nrtc_schedule import (
            NRTCSchedule,
        )
        csv = tmp_path / "tc.csv"
        csv.write_text(
            "year,jurisdiction,credit,amount,top,rate,clawback,clawback_rate,index\n"
            f"2014,BC,{kind},{amount},,0.0506,,,1\n"
        )
        return NRTCSchedule.from_csv(csv, jurisdiction="bc").credits[0]


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
    def test_published_dormant_credit_dropped_by_builder(self, kind, tmp_path):
        """Each credit the CSV still publishes but the runtime cannot compute is
        dropped by the builder, so loading the full historical CSV never grants
        them universally.  These names are read from the published schedule, not
        invented, so the case fails if one is ever re-registered without a
        matching runtime branch."""
        from macromodel.configurations.tax_parameters.central_government_builder import (
            _credit_component_to_def,
        )
        assert _credit_component_to_def(self._load_one(kind, tmp_path)) is None





class TestActivateTaxation:
    """The per-government, jurisdiction-keyed consumption seam: opt in + a reader
    present ⇒ progressive config; otherwise the base config is returned unchanged
    (flat parity).  Jurisdiction comes from the reader, never hardcoded."""

    def test_opted_in_with_reader_builds_progressive(self):
        base = CentralGovernmentConfiguration(activate_progressive_pit=True)
        config = activate_taxation(base, _committed_reader(), year=2014)
        assert config is not base  # a new, progressive config
        assert config.pit_brackets[0] == (37606.0, 0.0506)
        assert config.pit_dividend_integration is True

    def test_not_opted_in_returns_base_unchanged(self):
        """Reader present but the government did not opt in ⇒ flat parity."""
        base = CentralGovernmentConfiguration(activate_progressive_pit=False)
        config = activate_taxation(base, _committed_reader(), year=2014)
        assert config is base
        assert config.pit_brackets is None




