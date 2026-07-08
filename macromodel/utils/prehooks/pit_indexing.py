"""Personal-income-tax indexing pre-hook.

This module provides a factory for a pre-hook that advances each government's
progressive PIT schedule as the simulation's calendar year changes, mirroring
real-world annual bracket / credit indexation.

The per-year schedule table is assembled at construction in
``country.py`` (``_build_pit_schedule_by_year``) and stashed on the government
agent as ``states["pit_schedule_by_year"]``.  Each timestep this hook calls
:meth:`CentralGovernment.set_pit_for_year` so the agent swaps in the brackets,
rates, credits, and deductions published for the current year.  The lookup is a
pure, idempotent assignment, so running it every (quarterly) timestep is cheap
and safe.

Gating is by data presence: a government with no schedule table — flat-tax
governments, and progressive governments built from a single-year schedule —
is skipped, so the schedule simply holds flat (upstream parity).  This is why
the hook can be registered unconditionally; it self-gates per government.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    # Annotation-only import: kept out of the runtime path so importing this hook
    # does not import ``macromodel.simulation`` (which imports this module back,
    # forming a cycle).  Postponed annotations keep the type references as strings.
    from macromodel.simulation import Simulation


def create_pit_indexing_hook() -> Callable[[Simulation, int, int], None]:
    """Create a pre-hook that advances progressive PIT schedules by calendar year.

    Returns:
        Callable: A pre-hook with signature ``(simulation, year, month) -> None``
        that, for every country whose central government carries a per-year PIT
        schedule table, selects the schedule for ``year``.  Governments without a
        table are left untouched.

    Example:
        >>> hook = create_pit_indexing_hook()
        >>> simulation.prehooks.append(hook)
        >>> simulation.run()
    """

    def pit_indexing_hook(simulation: Simulation, year: int, month: int) -> None:
        """Pre-hook that sets each government's PIT schedule for the current year."""
        for country in simulation.countries.values():
            central_government = getattr(country, "central_government", None)
            if central_government is None:
                continue
            if "pit_schedule_by_year" in central_government.states:
                central_government.set_pit_for_year(year)

    return pit_indexing_hook
