"""Every committed tax schedule must cover the model's run horizon.

The readers are pure statutory lookup: a year they do not publish RAISES rather
than projecting or holding flat.  The run horizon is therefore the MINIMUM
terminal year across every active schedule -- one short file caps the whole run,
however complete the others are -- and because the tax features ship enabled by
default, a short schedule surfaces as an exception rather than a narrower run.

This is the case that motivated the check: the refundable-credit schedule ran to
2023 while every sibling ran to 2030, and it survived a design sweep and an
independent review because its provenance note asserted the opposite.  Nothing
compared the schedules against each other.

The directory is GLOBBED rather than listed, so a schedule added later is covered
without anyone remembering to extend this test.
"""

from pathlib import Path

import pandas as pd
import pytest

# The run horizon every schedule must reach.  A schedule stopping short is
# incomplete, not merely narrow.
HORIZON = 2030

#   parents[0]=test_readers [1]=unit [2]=test_macro_data [3]=tests [4]=repo root
_COMMITTED_PIT_DIR = Path(__file__).resolve().parents[4] / "spoof_data" / "freda" / "personal_income_tax"

# Known coverage gaps, asserted EXACT: a new gap fails, and so does a closed one.
_DEFERRED_GAPS: dict[tuple[str, str], int] = {
    ("rates_thresholds", "CA"): 2026,  # federal statute itself ends 2026
}


def _schedule_files() -> list[Path]:
    """Every committed schedule CSV, discovered rather than enumerated."""
    return sorted(_COMMITTED_PIT_DIR.glob("*.csv"))


def _last_year_by_jurisdiction(path: Path) -> "pd.Series":
    frame = pd.read_csv(path)
    assert "year" in frame.columns, f"{path.name} has no `year` column"
    if "jurisdiction" not in frame.columns:
        return pd.Series({"-": int(frame.year.max())})
    return frame.groupby("jurisdiction").year.max().astype(int)


def test_schedule_directory_is_populated() -> None:
    """Guard the guard: an empty glob would make every check below vacuous.

    Without this, a moved or renamed fixture directory turns the horizon test
    into a test that cannot fail -- it would collect zero cases and report green.
    """
    assert _COMMITTED_PIT_DIR.is_dir(), f"missing fixture dir: {_COMMITTED_PIT_DIR}"
    assert _schedule_files(), f"no schedule CSVs found under {_COMMITTED_PIT_DIR}"


@pytest.mark.parametrize("path", _schedule_files(), ids=lambda p: p.stem)
def test_schedule_reaches_horizon(path: Path) -> None:
    """The schedule publishes rows through ``HORIZON`` for every jurisdiction.

    Checked PER JURISDICTION, not per file: a file whose maximum year reaches the
    horizon can still strand one jurisdiction short, and the run horizon is the
    minimum across active jurisdictions rather than the file's maximum.
    """
    short = {
        geo: last
        for geo, last in _last_year_by_jurisdiction(path).items()
        if last < HORIZON and (path.stem, geo) not in _DEFERRED_GAPS
    }
    assert not short, (
        f"{path.name} stops short of the {HORIZON} horizon for {short}. The reader "
        f"raises past its last published year, so this caps the whole run. Extend "
        f"the GENERATOR's year bound -- these files are generated, and a hand-added "
        f"row is reverted by the next run."
    )


@pytest.mark.parametrize("path", _schedule_files(), ids=lambda p: p.stem)
def test_schedule_has_no_year_gaps(path: Path) -> None:
    """Every year from the schedule's start to the horizon is present.

    Reaching the horizon is not sufficient: a schedule running 2014, 2015 and
    2030 satisfies the check above and still raises for 2016.  A hole is the same
    defect as a short tail, arriving mid-run instead of at the end.
    """
    frame = pd.read_csv(path)
    groups = frame.groupby("jurisdiction") if "jurisdiction" in frame.columns else [("-", frame)]

    holes: dict[str, list[int]] = {}
    for geo, rows in groups:
        years = set(rows.year.astype(int))
        # A deferred jurisdiction is checked only up to the year it reaches, so a
        # hole INSIDE its published range is still caught.
        last = _DEFERRED_GAPS.get((path.stem, str(geo)), HORIZON)
        if missing := sorted(set(range(min(years), last + 1)) - years):
            holes[str(geo)] = missing

    assert not holes, f"{path.name} is missing years: {holes}"


@pytest.mark.parametrize("path", _schedule_files(), ids=lambda p: p.stem)
def test_schedule_does_not_run_past_the_horizon(path: Path) -> None:
    """HORIZON is a ceiling as well as a floor: no rows after it.

    Rows past 2030 are not extra coverage. Nothing is published there, so they
    could only be projections of an unbounded future -- and the reader is meant
    to RAISE past its last published year rather than serve them. Keeping the
    schedules bounded is what stops a run ever reaching that raise.
    """
    frame = pd.read_csv(path)
    beyond = sorted(set(frame.year.astype(int)) - set(range(0, HORIZON + 1)))
    assert not beyond, (
        f"{path.name} publishes rows past the {HORIZON} horizon: {beyond}. The "
        f"schedules are bounded at {HORIZON} deliberately -- past it the reader "
        f"raises, and that is the intended behaviour."
    )


def test_deferred_gaps_are_exactly_as_recorded() -> None:
    """The waiver list matches reality -- so it cannot silently rot.

    Fails in BOTH directions on purpose.  An unrecorded gap means something
    regressed; a recorded gap that no longer exists means the waiver is now
    standing over healthy data and must be deleted.
    """
    actual = {
        (path.stem, str(geo)): int(last)
        for path in _schedule_files()
        for geo, last in _last_year_by_jurisdiction(path).items()
        if last < HORIZON
    }
    assert actual == _DEFERRED_GAPS, (
        f"the recorded deferrals no longer match the data.\n"
        f"  recorded: {_DEFERRED_GAPS}\n"
        f"  actual:   {actual}\n"
        f"An EXTRA actual entry is a new gap. A MISSING one means that schedule was "
        f"extended -- delete its line from _DEFERRED_GAPS."
    )
