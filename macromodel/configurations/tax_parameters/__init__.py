"""Canada/BC-specific scalar tax parameters and their reader.

The ``tax_parameters.yaml`` file records the scalar tax parameters this project
adds on top of the upstream model, keyed by jurisdiction and tax year; the
reader applies a block as an override onto ``CentralGovernmentConfiguration``.
Schedule data (brackets, credit amounts) is excluded — it lives in the taxation
CSVs.
"""

from .central_government_builder import (
    activate_taxation,
    build_central_government_configuration,
)
from .tax_parameters_reader import apply_tax_parameters, read_tax_parameters

__all__ = [
    "activate_taxation",
    "apply_tax_parameters",
    "build_central_government_configuration",
    "read_tax_parameters",
]
