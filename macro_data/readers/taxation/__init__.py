class TaxationDataWarning(Warning):
    """Raised when a taxation tree is present but its PIT schedules are missing.

    A ``taxation/`` directory that exists but lacks its schedules is treated as a
    misconfiguration and surfaced; the wholesale absence of a ``taxation/``
    directory stays silent (taxation simply not in use).
    """

    pass


from macro_data.readers.taxation.taxation_reader import (  # noqa: E402
    DIVIDEND_FILENAME,
    RATES_THRESHOLDS_FILENAME,
    TAX_CREDITS_FILENAME,
    SchedulePaths,
    TaxationReader,
)
from macro_data.readers.taxation.taxation_store import (  # noqa: E402
    TaxationStore,
    jurisdiction_of,
)

__all__ = [
    "DIVIDEND_FILENAME",
    "RATES_THRESHOLDS_FILENAME",
    "TAX_CREDITS_FILENAME",
    "SchedulePaths",
    "TaxationDataWarning",
    "TaxationReader",
    "TaxationStore",
    "jurisdiction_of",
]
