"""Central Government agent implementation for macroeconomic modeling.

This module implements the central government agent, which manages:
- Tax collection and administration
- Social benefits distribution
- Fiscal policy implementation
- Government debt management

The central government plays a crucial role in:
- Revenue generation through various tax instruments
- Social welfare through benefits and transfers
- Economic stabilization through fiscal policy
- Public finance management
"""

import warnings
from typing import Any

import h5py
import numpy as np

from macro_data import SyntheticCentralGovernment
from macro_data.processing import TaxData
from macro_data.readers.taxation import TaxationDataWarning
from macro_data.readers.taxation.personal_income_tax.pit_schedule import compute_personal_income_tax
from macromodel.agents.agent import Agent
from macromodel.agents.central_government.central_government_ts import (
    create_central_government_timeseries,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import CentralGovernmentConfiguration
from macromodel.sim_calendar import steps_per_year
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions

# Scalars that vary by published year; each must ride the per-year fragment or it freezes.
PIT_PER_YEAR_SCALARS = (
    "pit_dividend_integration",
    "dividend_small_business_share",
    "bank_dividend_small_business_share",
    "dividend_eligible_gross_up",
    "dividend_non_eligible_gross_up",
    "dividend_eligible_dtc_rate",
    "dividend_non_eligible_dtc_rate",
    "couple_rental_income_split",
)


def pit_credit_defs_to_state_dicts(pit_non_refundable_tax_credits) -> list[dict]:
    """Convert configuration ``TaxCreditDef`` objects to the runtime credit dicts.

    Shared by ``from_pickled_agent`` and the per-year schedule table so both
    produce the identical dict shape the runtime credit pool consumes.
    """
    return [
        {
            "credit": t.credit,
            "amount": t.amount,
            "age_min": t.eligibility_age_min,
            "clawback": t.clawback,
            "top": t.top,
        }
        for t in pit_non_refundable_tax_credits
    ]


def pit_refundable_defs_to_state_dicts(pit_refundable_tax_credits) -> list[dict]:
    """Convert configuration ``RefundableCreditDef`` objects to runtime dicts.

    The refundable counterpart of ``pit_credit_defs_to_state_dicts``, and it
    carries two keys that one has no use for: ``credit_name``, which groups the
    components the taper applies to jointly, and ``delivery``, which decides
    which year the money reaches the household in.
    """
    return [
        {
            "credit_name": t.credit_name,
            "credit": t.credit,
            "delivery": t.delivery,
            "amount": t.amount,
            "amount_basis": t.amount_basis,
            "eligibility_age_min": t.eligibility_age_min,
            "clawback": t.clawback,
            "clawback_rate": t.clawback_rate,
        }
        for t in pit_refundable_tax_credits
    ]


# Flags that defer work to the year-end filing; the check derives its invalid set from this.
_RTC_INSTALMENTS = 4

FILING_DEPENDENT_FLAGS = (
    "pit_credits_at_filing",
    "pit_investment_at_year_end",
)


class CentralGovernment(Agent):
    """Central Government agent responsible for fiscal policy and social benefits.

    This class implements government fiscal operations including:
    - Tax collection (VAT, income, corporate, etc.)
    - Social benefit distribution (unemployment, other transfers)
    - Public finance management (revenue, deficit, debt)

    The agent manages multiple tax instruments:
    - Value-added Tax (VAT)
    - Income Tax
    - Corporate Tax
    - Social Insurance Contributions
    - Export and Capital Formation Taxes

    Attributes:
        functions (dict[str, Any]): Mapping of function names to implementations
        states (dict[str, float | np.ndarray]): Current state variables including
            tax rates and benefit models
        ts (TimeSeries): Time series data for government variables
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        """Initialize the Central Government agent.

        Args:
            country_name (str): Name of the country this government represents
            all_country_names (list[str]): List of all countries in the model
            n_industries (int): Number of industries in the economy
            functions (dict[str, Any]): Function implementations for government operations
            ts (TimeSeries): Time series data for tracking variables
            states (dict[str, float | np.ndarray]): State variables including tax rates
        """
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            0,
            0,
            ts,
            states,
        )
        self.functions = functions

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_central_government: SyntheticCentralGovernment,
        configuration: CentralGovernmentConfiguration,
        n_industries: int,
        country_name: str,
        all_country_names: list[str],
        tax_data: TaxData,
        number_of_unemployed_individuals: int,
        taxes_net_subsidies: np.ndarray,
    ):
        """Create a Central Government instance from pickled data.

        Initializes the government with:
        - Tax rates from historical data
        - Benefit models from synthetic data
        - Configuration parameters
        - Country-specific settings

        Args:
            synthetic_central_government (SyntheticCentralGovernment): Synthetic data
            configuration (CentralGovernmentConfiguration): Configuration parameters
            n_industries (int): Number of industries
            country_name (str): Country name
            all_country_names (list[str]): All country names
            tax_data (TaxData): Historical tax rate data
            number_of_unemployed_individuals (int): Count of unemployed
            taxes_net_subsidies (np.ndarray): Net tax rates by sector

        Returns:
            CentralGovernment: Initialized government agent
        """
        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.central_government")

        states = {
            "Value-added Tax": tax_data.value_added_tax,
            "Export Tax": tax_data.export_tax,
            "Employer Social Insurance Tax": tax_data.employer_social_insurance_tax,
            "Employee Social Insurance Tax": tax_data.employee_social_insurance_tax,
            "Profit Tax": tax_data.profit_tax,
            "Income Tax": tax_data.income_tax,
            "Capital Formation Tax": tax_data.capital_formation_tax,
            "Taxes Less Subsidies Rates": taxes_net_subsidies,
            "unemployment_benefits_model": synthetic_central_government.unemployment_benefits_model,
            "other_benefits_model": synthetic_central_government.other_benefits_model,
        }

        # Progressive PIT schedule; absent pit_brackets means flat Income Tax.
        if configuration.pit_brackets is not None:
            brackets = np.array(configuration.pit_brackets, dtype=float)
            states["pit_uppers"] = brackets[:, 0]
            states["pit_rates"] = brackets[:, 1]
            if configuration.pit_refundable_tax_credits is not None:
                states["pit_refundable_tax_credits"] = pit_refundable_defs_to_state_dicts(
                    configuration.pit_refundable_tax_credits
                )
            if configuration.pit_non_refundable_tax_credits is not None:
                states["pit_non_refundable_tax_credits"] = pit_credit_defs_to_state_dicts(
                    configuration.pit_non_refundable_tax_credits
                )

        states["couple_rental_income_split"] = configuration.couple_rental_income_split
        states["pit_year_end_reconciliation"] = configuration.pit_year_end_reconciliation
        states["pit_credits_at_filing"] = configuration.pit_credits_at_filing
        states["pit_investment_at_year_end"] = configuration.pit_investment_at_year_end

        # Dividend integration params (flag defaults False for parity).
        states["pit_dividend_integration"] = configuration.pit_dividend_integration
        states["dividend_small_business_share"] = configuration.dividend_small_business_share
        states["bank_dividend_small_business_share"] = configuration.bank_dividend_small_business_share
        states["dividend_eligible_gross_up"] = configuration.dividend_eligible_gross_up
        states["dividend_non_eligible_gross_up"] = configuration.dividend_non_eligible_gross_up
        states["dividend_eligible_dtc_rate"] = configuration.dividend_eligible_dtc_rate
        states["dividend_non_eligible_dtc_rate"] = configuration.dividend_non_eligible_dtc_rate

        data = (synthetic_central_government.central_gov_data.astype(float)).rename_axis("Central Government ID")

        ts = create_central_government_timeseries(
            data=data,
            number_of_unemployed_individuals=number_of_unemployed_individuals,
        )

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
            states,
        )

    def reset(self, configuration: CentralGovernmentConfiguration):
        """Reset the government agent to initial state.

        Resets all state variables and updates function implementations
        based on the provided configuration.

        Args:
            configuration (CentralGovernmentConfiguration): New configuration
                parameters for the reset state
        """
        self.gen_reset()
        update_functions(
            model=configuration.functions, loc="macromodel.agents.central_government", functions=self.functions
        )

    def update_benefits(
        self,
        historic_ppi_inflation: list[np.ndarray],
        exogenous_ppi_inflation: np.ndarray,
        current_estimated_ppi_inflation: float,
        current_unemployment_rate: float,
        current_estimated_growth: float,
    ) -> None:
        """Update social benefit levels based on economic conditions.

        Adjusts both unemployment benefits and other social transfers
        considering:
        - Historical and expected inflation
        - Current unemployment rate
        - Economic growth estimates

        Args:
            historic_ppi_inflation (list[np.ndarray]): Past inflation rates
            exogenous_ppi_inflation (np.ndarray): External inflation factors
            current_estimated_ppi_inflation (float): Current inflation estimate
            current_unemployment_rate (float): Current unemployment rate
            current_estimated_growth (float): Estimated economic growth
        """
        all_ppi_inflation = np.concatenate(
            (
                exogenous_ppi_inflation,
                np.array(historic_ppi_inflation).flatten(),
                [current_estimated_ppi_inflation],
            )
        )

        # Unemployment benefits
        self.ts.unemployment_benefits_by_individual.append(
            [
                self.functions["social_benefits"].compute_unemployment_benefits(
                    prev_unemployment_benefits=self.ts.current("unemployment_benefits_by_individual")[0],
                    historic_ppi_inflation=all_ppi_inflation,
                    current_estimated_growth=current_estimated_growth,
                    current_unemployment_rate=current_unemployment_rate,
                    model=self.states["unemployment_benefits_model"],
                )
            ]
        )

        # Regular social transfers to households
        self.ts.total_other_benefits.append(
            [
                self.functions["social_benefits"].compute_regular_transfer_to_households(
                    prev_regular_transfer_to_households=self.ts.current("total_other_benefits")[0],
                    historic_ppi_inflation=all_ppi_inflation,
                    current_estimated_growth=current_estimated_growth,
                    current_unemployment_rate=current_unemployment_rate,
                    model=self.states["other_benefits_model"],
                )
            ]
        )

    def distribute_unemployment_benefits_to_individuals(
        self,
        current_individual_activity_status: np.ndarray,
    ) -> np.ndarray:
        """Distribute unemployment benefits to eligible individuals.

        Allocates unemployment benefits to individuals based on their
        current activity status (employed vs. unemployed).

        Args:
            current_individual_activity_status (np.ndarray): Activity status
                for each individual

        Returns:
            np.ndarray: Unemployment benefits by individual (zero for employed)
        """
        unemployment_benefits = np.zeros(current_individual_activity_status.shape)
        unemployment_benefits[current_individual_activity_status == ActivityStatus.UNEMPLOYED] = self.ts.current(
            "unemployment_benefits_by_individual"
        )[0]
        return unemployment_benefits.astype(float)

    @property
    def progressive_pit_active(self) -> bool:
        """Whether a progressive schedule is configured; the flat path applies when it is not."""
        return self.states.get("pit_uppers") is not None and self.states.get("pit_rates") is not None

    def _fall_back_if_deferred_work_cannot_land(self, progressive_active: bool) -> None:
        """Disable deferral when no filing executes, instead of failing the run.

        Tax functionality ships ON and turns off two ways: absent or incomplete
        taxation data, or configuration switching it off. Both are FALLBACKS --
        they degrade to the legacy path with a warning, they do not raise. A
        guard that refused here would turn an off-switch into a crash.

        With no schedule the addon is inactive and these flags mean nothing: the
        flat branch taxes investment income directly and grants no credits, so
        nothing is stranded. The live case is a schedule present with
        reconciliation off -- deferral would then drop the credits and leave
        investment income untaxed, so the deferral is switched off and the
        period reverts to crediting and withholding as it did before.

        Warns once by construction: the flags are cleared, so a later period
        finds nothing stranded ([[warn-once-rule]]).

        Args:
            progressive_active: Whether a progressive schedule is configured.
        """
        if not progressive_active:
            return
        if self.states.get("pit_year_end_reconciliation", False):
            return

        stranded = [name for name in FILING_DEPENDENT_FLAGS if self.states.get(name, False)]
        if not stranded:
            return

        for name in stranded:
            self.states[name] = False
        warnings.warn(
            f"{', '.join(stranded)} defer work to the year-end filing, but "
            f"pit_year_end_reconciliation is off so no filing runs. Falling back "
            f"to the per-period behaviour for this run: credits are applied each "
            f"period and the full base is withheld against. Switch reconciliation "
            f"on to use the deferred path.",
            TaxationDataWarning,
            stacklevel=2,
        )

    def compute_taxes(
        self,
        current_ind_employee_income: np.ndarray,
        current_total_rent_paid: float,
        current_income_financial_assets: np.ndarray,
        current_ind_activity: np.ndarray | None = None,
        current_ind_realised_cons: np.ndarray | None = None,
        current_bank_profits: np.ndarray | None = None,
        current_firm_production: np.ndarray | None = None,
        current_firm_price: np.ndarray | None = None,
        current_firm_profits: np.ndarray | None = None,
        current_firm_industries: np.ndarray | None = None,
        current_household_new_real_wealth: np.ndarray | None = None,
        taxes_less_subsidies_rates: np.ndarray | None = None,
        current_total_exports: float = 0.0,
        # New parameters are appended after the upstream ones, so positional callers still bind.
        taxable_income_per_ind: np.ndarray | None = None,
        withheld_income_per_ind: np.ndarray | None = None,
        nrtc_base_per_ind: np.ndarray | None = None,
        nrtc_direct_per_ind: np.ndarray | None = None,
        annual_credit_base=None,
        annual_rtc=None,
    ) -> None:
        """Calculate all tax revenues for the current period.

        Computes revenues from multiple tax sources:
        - Production and VAT
        - Income and corporate taxes
        - Social insurance contributions
        - Capital formation and export taxes

        Progressive PIT consumes the pre-assembled taxable-income and credit-base
        pools; both are required, since only the processing phase holds the
        household context and the dividend items they depend on.

        Args:
            current_ind_employee_income (np.ndarray): Employee incomes per individual
            current_total_rent_paid (float): Total rent paid by renters (scalar)
            current_income_financial_assets (np.ndarray): Financial income per household
            current_ind_activity (np.ndarray): Individual activity status
            current_ind_realised_cons (np.ndarray): Consumption levels
            current_bank_profits (np.ndarray): Bank profits
            current_firm_production (np.ndarray): Firm production
            current_firm_price (np.ndarray): Product prices
            current_firm_profits (np.ndarray): Firm profits
            current_firm_industries (np.ndarray): Industry classifications
            current_household_new_real_wealth (np.ndarray): New wealth
            taxes_less_subsidies_rates (np.ndarray): Net tax rates
            current_total_exports (float): Total exports
            withheld_income_per_ind (np.ndarray): The narrower pool the PERIOD
                withholds against, employment only. None withholds against the
                full base, which is the legacy behaviour.
            taxable_income_per_ind (np.ndarray): Pool A, the taxable income per
                individual, assembled by the processing phase. Required.
            nrtc_base_per_ind (np.ndarray): Pool B, the non-refundable credit
                base per individual, assembled by the processing phase. Required.
        """
        # Taxes on production
        self.ts.taxes_production.append(
            [np.sum(taxes_less_subsidies_rates[current_firm_industries] * current_firm_production * current_firm_price)]
        )

        # Value-added taxes
        self.ts.taxes_vat.append([self.states["Value-added Tax"] * np.sum(current_ind_realised_cons)])

        # Taxes on capital formation
        self.ts.taxes_cf.append(
            [self.states["Capital Formation Tax"] * np.sum(np.maximum(0.0, current_household_new_real_wealth))]
        )

        # Corporate income taxes
        self.ts.taxes_corporate_income.append(
            [
                self.states["Profit Tax"]
                * (np.sum(np.maximum(current_firm_profits, 0)) + np.sum(np.maximum(current_bank_profits, 0)))
            ]
        )

        # Taxes on exports
        self.ts.taxes_exports.append([self.states["Export Tax"] * current_total_exports])

        # Total wages of employed individuals
        tot_wages_employed_ind = np.sum([current_ind_employee_income[current_ind_activity == ActivityStatus.EMPLOYED]])

        # Advanced once per period so the gated taxes share one year boundary; pre-calibration does not advance it.
        self.states["tax_step"] = int(self.states.get("tax_step", 0)) + 1

        # Personal income tax: progressive when a schedule is configured, else flat.
        pit_uppers = self.states.get("pit_uppers")
        pit_rates = self.states.get("pit_rates")
        settlement = 0.0  # only a filing on the progressive path makes this non-zero
        # Refundable credits are expenditure at full value, so they never net into taxes_income.
        rtc_settlement = 0.0
        rtc_instalment = 0.0
        # Filled only by a filing.
        credit_granted: list = []

        self._fall_back_if_deferred_work_cannot_land(self.progressive_pit_active)

        if pit_uppers is not None and pit_rates is not None:
            # The processing phase alone holds the household context, so a missing pool raises.
            if taxable_income_per_ind is None or nrtc_base_per_ind is None:
                raise ValueError(
                    "Progressive PIT requires both pools. Assemble them with "
                    "pit_pools.build_taxable_income_pool and "
                    "pit_pools.build_credit_base_pool at the call site."
                )

            # The pools arrive annualized; the same factor apportions the tax back.
            factor = steps_per_year()
            tax_per_ind: list = []
            # Only the withholding narrows; the year's liability is still taken on the full base.
            withheld_pool = taxable_income_per_ind
            if self.states.get("pit_investment_at_year_end", False) and withheld_income_per_ind is not None:
                withheld_pool = withheld_income_per_ind
            total_income_tax = self.compute_pit(
                withheld_pool,
                nrtc_base_per_ind,
                nrtc_direct_per_ind,
                steps_per_year=factor,
                out_tax_per_ind=tax_per_ind,
            )
            settlement, rtc_settlement, rtc_instalment = self._reconcile_tax_year(
                taxable_income_per_ind,
                tax_per_ind[0],
                nrtc_base_per_ind,
                nrtc_direct_per_ind,
                steps_per_year=factor,
                annual_credit_base=annual_credit_base,
                annual_rtc=annual_rtc,
                out_credit_granted=credit_granted,
            )
            # Revenue is a scalar line, so the per-individual settlement collapses here.
            total_income_tax += float(np.sum(settlement))
        else:
            # Flat tax (backward-compatible path).
            total_income_tax = (
                self.states["Income Tax"] * (1 - self.states["Employee Social Insurance Tax"]) * tot_wages_employed_ind
                + self.states["Income Tax"] * current_total_rent_paid
                + self.states["Income Tax"] * current_income_financial_assets.sum()
            )

        self.ts.taxes_income.append([total_income_tax])

        # Recorded separately: the settlement is invisible inside taxes_income.
        self.ts.pit_year_end_settlement.append([float(np.sum(settlement))])

        # Separate series: the settlement leg and the instalment leg refer to different years.
        self.ts.pit_rtc_settlement.append([float(np.sum(rtc_settlement))])
        self.ts.pit_rtc_instalments.append([float(np.sum(rtc_instalment))])

        # Revenue foregone to the non-refundable credits: reported, never booked.
        self.ts.pit_non_refundable_credits_granted.append([float(sum(credit_granted))])

        # Per individual, because the payment circuit needs who is owed, not just how much.
        self.states["pit_rtc_paid_per_ind"] = (
            np.asarray(rtc_settlement) + np.asarray(rtc_instalment) if not np.isscalar(rtc_settlement) else 0.0
        )

        # Reporting figure (feeds GDP rent_received); already inside taxes_income.
        self.ts.taxes_rental_income.append([self.states["Income Tax"] * current_total_rent_paid])

        # Taxes on employer social insurance
        self.ts.taxes_employer_si.append([self.states["Employer Social Insurance Tax"] * tot_wages_employed_ind])

        # Taxes on employee social insurance
        self.ts.taxes_employee_si.append([self.states["Employee Social Insurance Tax"] * tot_wages_employed_ind])

    def compute_pit(
        self,
        taxable_income_per_ind: np.ndarray,
        nrtc_base_per_ind: np.ndarray | None = None,
        nrtc_direct_per_ind: np.ndarray | None = None,
        steps_per_year: float = 1.0,
        out_tax_per_ind: list | None = None,
    ) -> float:
        """Apply fixed PIT policy (brackets, credits) to the assembled pools.

        The government's tax core; it references no income streams or specific
        credits, so extending either in ``pit_pools`` leaves it untouched. As a side
        effect it updates the scalar ``states["Income Tax"]`` effective rate to
        the schedule-implied average, which keeps wage-setting aligned with the
        schedule (an accepted, bounded ripple when dividend integration is on).

        Args:
            taxable_income_per_ind: Pool A — taxable income per individual.
            nrtc_base_per_ind: Pool B — summed non-refundable credit base per
                individual (``None`` or zeros when no credits apply).
            nrtc_direct_per_ind: Direct dollar credits per individual (the
                dividend tax credit); ``None`` when not applicable.
            steps_per_year: Steps in a year when the pools were annualized; the
                assessed annual tax is divided by it to give the period's revenue.
            out_tax_per_ind: Optional list receiving the per-individual period
                tax, which the year-end reconciliation accumulates.

        Returns:
            float: Personal income tax revenue for the period.
        """
        pit_uppers = self.states["pit_uppers"]
        pit_rates = self.states["pit_rates"]

        pit_per_individual = compute_personal_income_tax(taxable_income_per_ind, pit_uppers, pit_rates)

        # Floored at zero; skipped when the credits are deferred to the filing.
        if not self.states.get("pit_credits_at_filing", False):
            total_credit = np.zeros_like(pit_per_individual, dtype=float)
            if nrtc_base_per_ind is not None:
                total_credit = total_credit + nrtc_base_per_ind * float(pit_rates[0])
            if nrtc_direct_per_ind is not None:
                total_credit = total_credit + nrtc_direct_per_ind
            pit_per_individual = np.maximum(0.0, pit_per_individual - total_credit)

        # Assessed against annual policy, so this is the annual liability.
        total_annual_tax = float(pit_per_individual.sum())

        total_taxable_base = float(taxable_income_per_ind.sum())
        # A non-finite total would leave a plausible rate beside NaN revenue, so it raises.
        if not (np.isfinite(total_annual_tax) and np.isfinite(total_taxable_base)):
            raise ValueError(
                f"PIT produced a non-finite result (tax={total_annual_tax}, "
                f"base={total_taxable_base}); check the income pools for NaN."
            )
        # Both terms are annual, so the rate is frequency-invariant.
        if total_taxable_base > 0:
            self.states["Income Tax"] = total_annual_tax / total_taxable_base

        # The period collects its share of the annual liability.
        pit_per_individual = pit_per_individual / steps_per_year
        if out_tax_per_ind is not None:
            out_tax_per_ind.append(pit_per_individual)

        return total_annual_tax / steps_per_year

    def _reconcile_tax_year(
        self,
        annualized_income_per_ind: np.ndarray,
        tax_per_ind: np.ndarray,
        nrtc_base_per_ind: np.ndarray | None,
        nrtc_direct_per_ind: np.ndarray | None,
        steps_per_year: float = 1.0,
        annual_credit_base=None,
        annual_rtc=None,
        out_credit_granted: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Settle the previous tax year and carry this period into the next.

        At the first period of a new year the liability on the year's actual
        income is compared with what was withheld, and the difference settles
        through the income-tax line at that filing: an over-payment is refunded
        and an under-payment collected, both at the same moment, as a filing has
        one outcome. Returns the settlement PER INDIVIDUAL, negative for a refund;
        callers that need the period's revenue sum it themselves. The per-individual
        shape is what lets the refundable credit reach the individuals who earned it
        without a second allocation pass. Off unless
        ``states["pit_year_end_reconciliation"]`` is set.

        Both halves are taken on the year's actual income: ``annual_credit_base``
        re-values the credits there, since a credit that tapers with income does
        not average back to its annual figure. Without it the accumulated average
        stands in, and the settlement is approximate for those credits.

        ``out_credit_granted`` optionally receives one float per filing: the
        revenue the non-refundable credits removed from the settled year's bill.
        It is an out-parameter rather than a fourth return value so that the
        callers already unpacking three keep working.
        """
        if not self.states.get("pit_year_end_reconciliation", False):
            zero = np.zeros(len(annualized_income_per_ind))
            return zero, zero.copy(), zero.copy()

        factor = steps_per_year
        n_ind = len(annualized_income_per_ind)

        settlement = np.zeros(n_ind)
        rtc_settlement = np.zeros(n_ind)

        ytd_income = self.states.get("pit_ytd_income")
        if ytd_income is None or len(ytd_income) != n_ind:
            # First period, or the population resized: the per-individual totals restart.
            ytd_income = np.zeros(n_ind)
            ytd_tax = np.zeros(n_ind)
            ytd_credit = np.zeros(n_ind)
            ytd_direct_credit = np.zeros(n_ind)
            # The shared counter is NOT reset: a resize is PIT-private and must not move another tax's calendar.
        else:
            ytd_tax = self.states["pit_ytd_tax"]
            ytd_credit = self.states["pit_ytd_credit"]
            ytd_direct_credit = self.states["pit_ytd_direct_credit"]

        # Advanced by ``compute_taxes`` before this runs; read, never written.
        step = int(self.states.get("tax_step", 0))

        # Step 1 is the first period of a year; settle before this period is added.
        if step > factor and (step - 1) % factor == 0:
            # Assess under the schedule in force FOR the settled year, not the live one.
            uppers = self.states["pit_uppers"]
            rates = self.states["pit_rates"]
            year_credits = None
            year_rtc = None
            calendar_year = self.states.get("pit_calendar_year")
            if calendar_year is not None:
                settled = self._pit_schedule_for_year(int(calendar_year) - 1)
                if settled is not None:
                    uppers = settled["pit_uppers"]
                    rates = settled["pit_rates"]
                    year_credits = settled.get("pit_non_refundable_tax_credits")
                    year_rtc = settled.get("pit_refundable_tax_credits")

            annual_tax = compute_personal_income_tax(ytd_income, uppers, rates)
            # Re-value the credits on the income the tax was assessed on.
            annual_credit = ytd_credit
            if annual_credit_base is not None:
                annual_credit = ytd_direct_credit + annual_credit_base(ytd_income, year_credits) * float(rates[0])
            # Per individual: the floor must apply per person, or a negative liability offsets a positive one.
            liability = np.maximum(0.0, annual_tax - annual_credit)
            # Revenue foregone, not entitlement granted: the floor discards any excess credit.
            if out_credit_granted is not None:
                out_credit_granted.append(float(np.sum(annual_tax - liability)))
            withheld = ytd_tax
            # A filing has one outcome: a refund reduces the period's revenue, an amount owing adds to it.
            settlement += liability - withheld

            # Valued here on the settled year's income; NRTC first, then RTC, or the RTC books twice.
            if annual_rtc is not None:
                rtc_now, rtc_later = annual_rtc(ytd_income, year_rtc)
                rtc_settlement += rtc_now
                # The entitlement is fixed at the filing and drawn over four periods starting here.
                self.states["pit_rtc_instalment_amount"] = rtc_later / _RTC_INSTALMENTS
                self.states["pit_rtc_instalments_left"] = _RTC_INSTALMENTS

            ytd_income = np.zeros(n_ind)
            ytd_tax = np.zeros(n_ind)
            ytd_credit = np.zeros(n_ind)
            ytd_direct_credit = np.zeros(n_ind)

        period_credit = np.zeros(n_ind)
        if nrtc_base_per_ind is not None:
            period_credit = period_credit + nrtc_base_per_ind * float(self.states["pit_rates"][0])
        period_direct = np.zeros(n_ind)
        if nrtc_direct_per_ind is not None:
            period_direct = period_direct + nrtc_direct_per_ind

        # The year's income is the sum of the unscaled periods.
        self.states["pit_ytd_income"] = ytd_income + annualized_income_per_ind / factor
        self.states["pit_ytd_tax"] = ytd_tax + tax_per_ind
        self.states["pit_ytd_credit"] = ytd_credit + (period_credit + period_direct) / factor
        self.states["pit_ytd_direct_credit"] = ytd_direct_credit + period_direct / factor

        # Carried in states: the periods paying it out have no access to the settled year's income.
        rtc_instalment = np.zeros(n_ind)
        left = int(self.states.get("pit_rtc_instalments_left", 0))
        if left > 0:
            per_period = np.asarray(self.states.get("pit_rtc_instalment_amount", rtc_instalment))
            if len(per_period) == n_ind:
                rtc_instalment = per_period.copy()
            self.states["pit_rtc_instalments_left"] = left - 1

        return settlement, rtc_settlement, rtc_instalment

    def compute_taxes_on_products(self) -> float:
        """Calculate total taxes on products and production.

        Aggregates various product-related taxes:
        - Production taxes
        - Value-added tax (VAT)
        - Capital formation tax
        - Export taxes

        Returns:
            float: Total tax revenue from products and production
        """
        return (
            self.ts.current("taxes_production")[0]
            + self.ts.current("taxes_vat")[0]
            + self.ts.current("taxes_cf")[0]
            + self.ts.current("taxes_exports")[0]
        )

    def _pit_schedule_for_year(self, year: int) -> dict | None:
        """Return the published PIT schedule in force for *year*.

        A statutory lookup with no forward projection: a year before the first
        published year uses the first, a gap year holds at the most recent prior
        year, and a year beyond the last raises. Shared by the schedule pre-hook
        and the year-end filing so the two cannot resolve a year differently.

        Args:
            year: The calendar year to resolve.

        Returns:
            dict | None: The schedule fragment, or ``None`` when the government
            carries no schedule table.

        Raises:
            ValueError: If ``year`` exceeds the last published year.
        """
        table = self.states.get("pit_schedule_by_year")
        if not table:
            return None

        years = sorted(table)
        if year in table:
            return table[year]
        if year < years[0]:
            return table[years[0]]
        if year > years[-1]:
            raise ValueError(
                f"Simulation year {year} exceeds the last available PIT "
                f"schedule year {years[-1]}. The schedule is a statutory "
                f"lookup with no forward projection; extend the taxation "
                f"schedule CSVs to cover {year} before running a "
                f"simulation this far."
            )
        # Gap year: hold at the most recent published year at or before it.
        return table[max(y for y in years if y <= year)]

    def set_pit_for_year(self, year: int) -> None:
        """Swap in the PIT schedule for *year* from ``states["pit_schedule_by_year"]``.

        Resolves the year through ``_pit_schedule_for_year``. No-op when no
        schedule table is present (flat and single-year governments stay frozen
        at construction).

        Args:
            year: The simulation's current calendar year.

        Raises:
            ValueError: If ``year`` exceeds the last published year.
        """
        fragment = self._pit_schedule_for_year(year)
        if fragment is None:
            return

        # The year the live states describe, so the filing can find the schedule it settles.
        self.states["pit_calendar_year"] = year
        self.states["pit_uppers"] = fragment["pit_uppers"]
        self.states["pit_rates"] = fragment["pit_rates"]
        # Clear on absence so a field omitted this year drops any stale value.
        if "pit_non_refundable_tax_credits" in fragment:
            self.states["pit_non_refundable_tax_credits"] = fragment["pit_non_refundable_tax_credits"]
        else:
            self.states.pop("pit_non_refundable_tax_credits", None)
        if "pit_refundable_tax_credits" in fragment:
            self.states["pit_refundable_tax_credits"] = fragment["pit_refundable_tax_credits"]
        else:
            self.states.pop("pit_refundable_tax_credits", None)
        # Assigned but never popped: a missing year must hold the previous value.
        for name in PIT_PER_YEAR_SCALARS:
            if name in fragment:
                self.states[name] = fragment[name]

    def compute_revenue(
        self,
        household_rent_paid_to_government: float,
    ) -> float:
        """Calculate total government revenue.

        Aggregates all revenue sources:
        - All tax revenues
        - Social insurance contributions
        - Rental income from public housing

        Args:
            household_rent_paid_to_government (float): Rent from public housing

        Returns:
            float: Total government revenue
        """
        self.ts.total_rent_received.append([household_rent_paid_to_government])
        return (
            self.ts.current("taxes_production")[0]
            + self.ts.current("taxes_vat")[0]
            + self.ts.current("taxes_cf")[0]
            + self.ts.current("taxes_corporate_income")[0]
            + self.ts.current("taxes_exports")[0]
            + self.ts.current("taxes_income")[0]
            + self.ts.current("taxes_employee_si")[0]
            + self.ts.current("taxes_employer_si")[0]
            + household_rent_paid_to_government
        )

    def compute_deficit(
        self,
        current_ind_activity: np.ndarray,
        current_household_social_transfers: np.ndarray,
        current_government_nominal_amount_spent: np.ndarray,
        government_interest_rates: float,
    ) -> np.ndarray:
        """Calculate the government deficit.

        Computes deficit as the difference between:
        Expenditures:
        - Unemployment benefits
        - Social transfers
        - Government spending
        - Interest payments
        And:
        - Total revenue

        Args:
            current_ind_activity (np.ndarray): Individual activity status
            current_household_social_transfers (np.ndarray): Social transfers
            current_government_nominal_amount_spent (np.ndarray): Spending
            government_interest_rates (float): Interest rate on debt

        Returns:
            np.ndarray: Government deficit (positive = deficit)
        """
        total_unemployment_benefits = (
            np.sum(current_ind_activity == ActivityStatus.UNEMPLOYED)
            * self.ts.current("unemployment_benefits_by_individual")[0]
        )
        total_household_social_transfers = np.sum(current_household_social_transfers)
        all_benefits = total_unemployment_benefits + total_household_social_transfers
        interest_payments = government_interest_rates * self.ts.current("debt")[0]
        return np.array(
            [
                all_benefits
                + np.sum(current_government_nominal_amount_spent)
                + interest_payments
                - self.ts.current("revenue")[0]
            ]
        )

    def compute_debt(self) -> np.ndarray:
        """Update government debt level.

        Calculates new debt level by adding current deficit
        to existing debt stock.

        Returns:
            np.ndarray: Updated government debt level
        """
        return np.array([self.ts.current("debt")[0] + self.ts.current("deficit")[0]])

    def save_to_h5(self, group: h5py.Group):
        """Save government data to HDF5 format.

        Stores all time series data in the specified HDF5 group.

        Args:
            group (h5py.Group): HDF5 group to save data in
        """
        self.ts.write_to_h5("central_government", group)

    def total_taxes(self):
        """Calculate total tax revenue on products.

        Returns:
            float: Aggregate tax revenue from all product-related taxes
        """
        return self.ts.get_aggregate("taxes_on_products")
