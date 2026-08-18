"""Tests for the ``pit_schedule_update`` pre-hook.

The hook walks every country's central government and, where a per-year PIT
schedule table is present, calls ``set_pit_for_year`` for the current year.
Governments without a table are skipped, so the hook self-gates and can be
registered unconditionally.  These tests use lightweight stubs so the gating and
dispatch logic is exercised without constructing a full ``Simulation``.
"""

from macromodel.utils.prehooks import create_pit_schedule_update_hook


class _StubGovernment:
    def __init__(self, *, has_table):
        self.states = {}
        if has_table:
            self.states["pit_schedule_by_year"] = {2014: object()}
        self.set_for_year_calls = []

    def set_pit_for_year(self, year):
        self.set_for_year_calls.append(year)


class _StubCountry:
    def __init__(self, government):
        self.central_government = government


class _StubSimulation:
    def __init__(self, countries):
        self.countries = countries


def test_only_tabled_governments_are_advanced():
    with_table = _StubGovernment(has_table=True)
    without_table = _StubGovernment(has_table=False)
    sim = _StubSimulation(
        {
            "CAN": _StubCountry(with_table),
            "USA": _StubCountry(without_table),
        }
    )

    create_pit_schedule_update_hook()(sim, 2030, 7)

    assert with_table.set_for_year_calls == [2030]
    assert without_table.set_for_year_calls == []
