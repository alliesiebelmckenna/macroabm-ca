"""Personal-income-tax schedule-update pre-hook.

This module provides a factory for a pre-hook that advances each government's
progressive PIT schedule to the current calendar year, swapping in that year's
published brackets and credits. The per-year schedule table is assembled in
``country._build_pit_schedule_by_year`` and stashed on the government agent as
``states["pit_schedule_by_year"]``; each timestep the hook calls
``CentralGovernment.set_pit_for_year`` to swap in the brackets, rates, credits,
and deductions published for the current year. The lookup is a pure, idempotent
assignment, so running it every timestep is cheap.

The hook self-gates on data presence: a government with no schedule table (flat
governments, or progressive ones built from a single-year schedule) is skipped
and simply holds flat (upstream parity), so the hook can be registered
unconditionally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    # Annotation-only import: kept out of the runtime path to avoid an import
    # cycle with ``macromodel.simulation``, which imports this module back.
    from macromodel.simulation import Simulation


def create_pit_schedule_update_hook() -> Callable[[Simulation, int, int], None]:
    """Create a pre-hook that advances progressive PIT schedules by calendar year.

    Returns:
        Callable: A pre-hook with signature ``(simulation, year, month) -> None``
        that, for every country whose central government carries a per-year PIT
        schedule table, selects the schedule for ``year``. Governments without a
        table are left untouched.
    """

    def pit_schedule_update_hook(simulation: Simulation, year: int, month: int) -> None:
        """Pre-hook that sets each government's PIT schedule for the current year."""
        for country in simulation.countries.values():
            central_government = getattr(country, "central_government", None)
            if central_government is None:
                continue
            if "pit_schedule_by_year" in central_government.states:
                central_government.set_pit_for_year(year)

    return pit_schedule_update_hook
