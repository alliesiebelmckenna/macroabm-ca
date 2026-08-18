"""The refundable credit schedule reader: lookup only, and fail-closed."""

from pathlib import Path

from macro_data.readers.taxation.personal_income_tax.rtc_schedule import (
    RefundableSchedule,
)

#   parents[0]=test_readers [1]=unit [2]=test_macro_data [3]=tests [4]=repo root
_CSV = (
    Path(__file__).resolve().parents[4] / "spoof_data" / "freda" / "personal_income_tax" / "refundable_tax_credits.csv"
)


class TestPublishedLookup:
    def test_a_published_year_returns_its_own_components(self):
        s = RefundableSchedule.from_csv(_CSV, jurisdiction="BC")
        by_credit = {c.credit: c for c in s.get_credits(2023)}
        # The published 2023 figures, which the schedule reproduces exactly.
        assert by_credit["Eligible Individual Amount"].amount == 504.0
        assert by_credit["Spousal Amount"].amount == 252.0
        assert by_credit["Dependant Amount"].amount == 126.0
        assert by_credit["Renter's Amount"].amount == 400.0
