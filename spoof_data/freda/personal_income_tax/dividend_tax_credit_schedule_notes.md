# dividend_tax_credit_schedule.csv — provenance

Consolidated, geo-keyed, per-year dividend gross-up and dividend tax credit
(DTC) rates. Columns: `tax_year, geo, dividend_type, gross_up_rate,
dtc_pct_of_grossed_up, dtc_pct_of_actual`. The gross-up is a federal mechanism
(uniform across jurisdictions in the current data); the DTC rate is
jurisdiction-specific. `dtc_pct_of_actual` is a reference figure equal to
`dtc_pct_of_grossed_up x (1 + gross_up_rate)`; the model uses
`dtc_pct_of_grossed_up`.

Years 2014–2030. The last legislated values (2019) are frozen forward to 2030
(no announced change), mirroring the bracket/credit schedule freeze.

## BC — verified
Migrated from the previous `bc_dividend_tax_credit_schedule.csv` and confirmed
against the taxtips.ca BC dividend tax credit page (retrieved 2026-07-09):
eligible DTC 10.0% (2014–2018) → 12.0% (2019+, BC Sept 2017 budget);
non-eligible DTC 2.59% (2014–15), 2.47% (2016), 2.18% (2017), 2.07% (2018),
1.96% (2019+).

## CA (federal) — provisional
Sourced from taxtips.ca eligible / non-eligible dividend tax credit rate pages
(retrieved 2026-07-09). **Provisional**: per the source-priority rule the
federal figures should be confirmed against canada.ca, which blocks automated
fetching (cite-only). Eligible DTC 15.0198% of grossed-up (constant 2012+, 38%
gross-up); non-eligible DTC 11.0169% (2014–15), 10.5217% (2016–17), 10.0313%
(2018), 9.0301% (2019+), tracking the federal non-eligible gross-up of
18%/17%/16%/15%.

## Scope
CA (federal) and BC only. Other provinces and Quebec are deferred
(see complete-tax-schedule-all-jurisdictions). Quebec is expected to share the
federal gross-up but carries its own DTC; the per-row `gross_up_rate` column
lets a divergent jurisdiction hold its own value without a schema change.
