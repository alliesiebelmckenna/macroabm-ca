"""Steps-per-year seam for the annual-basis probe.

Several behavioural equations convert a per-step quantity to an annual one with
a hardcoded factor of four, which is correct only when a step is three months.
This module holds that factor in one place so it can be derived from the
configured ``time_unit`` instead.

The default is ``4.0``, so any code path that never calls ``set_steps_per_year``
behaves exactly as before.

This is a probe-grade seam: a module-level value set once at simulation
construction. A merge-grade implementation would thread the value through the
configuration objects to the agents that need it rather than holding it
globally, since a global cannot serve two simulations at different time units
in one process.
"""

from __future__ import annotations

_STEPS_PER_YEAR: float = 4.0


def set_steps_per_year(time_unit: int) -> None:
    """Set the steps-per-year factor from the step length in months.

    Args:
        time_unit (int): Step length in months, 1-12.
    """
    global _STEPS_PER_YEAR
    if not 1 <= time_unit <= 12:
        raise ValueError(f"time_unit must be in 1..12, got {time_unit}")
    _STEPS_PER_YEAR = 12.0 / float(time_unit)


def steps_per_year() -> float:
    """float: Number of simulation steps in one calendar year."""
    return _STEPS_PER_YEAR
