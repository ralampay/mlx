"""Dataset partitions supplied to native training workflows."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingDatasets:
    train: Any
    validation: Any
    # An optional source understood by the injected sample-generation command.
    sample_source: Any = None
