"""The taxation data layer's country-keyed reader.

``TaxationStore`` is what ``DataReaders.taxation`` holds: the schedules for
*every* taxing authority in the data, not one jurisdiction's. It follows the
same shape as the other country-keyed readers in this package — the reader
carries all countries and the caller selects one at construction time by passing
the country in (compare ``OecdEconomicData.read_long_term_interest_rates``,
``EurostatReader.dividend_payout_ratio``) — so a regional run builds every
province from one reader, as every other data layer already does.

The per-country slice is a ``TaxationReader``: one jurisdiction's schedules, the
object that crosses the pickle boundary on ``SyntheticCountry.taxation`` and is
consumed by the macromodel's central-government builder.

Jurisdictions are discovered from the bracket file's ``jurisdiction`` column rather than
hardcoded, so extending the data to new provinces or to the federal authority
needs no code change here. A country the data does not cover yields ``None`` —
it falls back to the flat Income Tax rate, which is the correct treatment for a
jurisdiction whose schedules have not been sourced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers.taxation.taxation_reader import SchedulePaths, TaxationReader


def jurisdiction_of(country: "Country | Region | str") -> str:
    """The taxing-authority key for *country* (``CAN_BC`` -> ``bc``, ``CA`` -> ``ca``).

    A region's own jurisdiction is its region-code suffix: BC's provincial
    schedules govern ``CAN_BC``. (A province is ALSO subject to its parent
    country's federal schedule; that second authority is not threaded here — see
    the government-layers design.)
    """
    return str(country).split("_")[-1].lower()


@dataclass
class TaxationStore:
    """Every taxing authority's personal-income-tax schedules.

    Attributes:
        paths: The schedule files this store reads.
        jurisdictions: Authority keys the data covers, from the bracket file's
            ``jurisdiction`` column (lowercased). This set — not any filename — is what
            decides which governments can run a progressive PIT.
    """

    paths: SchedulePaths
    jurisdictions: frozenset[str]
    _cache: dict[str, TaxationReader] = field(default_factory=dict, repr=False)

    @classmethod
    def from_paths(cls, paths: SchedulePaths) -> "TaxationStore":
        """Discover the jurisdictions covered by the bracket file in *paths*.

        Raises:
            FileNotFoundError: If the bracket schedule is absent. The caller
                (``_load_taxation_reader``) turns this into a warning and
                disables taxation.
        """
        if not Path(paths.rates).exists():
            raise FileNotFoundError(f"Schedule file not found: {paths.rates}")

        jurisdiction_col = pd.read_csv(paths.rates, usecols=["jurisdiction"])["jurisdiction"]
        jurisdictions = frozenset(str(g).strip().lower() for g in jurisdiction_col.dropna().unique())

        return cls(paths=paths, jurisdictions=jurisdictions)

    @classmethod
    def from_dir(cls, schedule_dir: Path, **filenames: str) -> "TaxationStore":
        """Discover the covered jurisdictions among the schedule files in *schedule_dir*.

        Args:
            schedule_dir: Directory holding the consolidated schedule CSVs —
                typically ``raw_data_path / "taxation" / "personal_income_tax"``.
            **filenames: Optional ``rates`` / ``credits`` / ``dividend`` filename
                overrides, to read an alternative data source in the same schema.
        """
        return cls.from_paths(SchedulePaths.in_dir(Path(schedule_dir), **filenames))

    def for_country(self, country: "Country | Region | str") -> Optional[TaxationReader]:
        """This country's own schedules, or ``None`` when the data does not cover it.

        ``None`` is a normal result, not an error: an uncovered jurisdiction runs
        the flat Income Tax rate. Readers are cached, so a 10-province build
        parses each jurisdiction once.
        """
        juris = jurisdiction_of(country)
        if juris not in self.jurisdictions:
            return None
        if juris not in self._cache:
            self._cache[juris] = TaxationReader.from_paths(self.paths, jurisdiction=juris)
        return self._cache[juris]
