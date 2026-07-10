class TaxationDataWarning(Warning):
    """Raised when a taxation tree is present but its PIT schedules are missing.

    A ``taxation/`` directory that exists but lacks its schedules is treated as a
    misconfiguration and surfaced; the wholesale absence of a ``taxation/``
    directory stays silent (taxation simply not in use).
    """

    pass


from macro_data.readers.taxation.taxation_reader import TaxationReader  # noqa: E402

__all__ = ["TaxationDataWarning", "TaxationReader"]
