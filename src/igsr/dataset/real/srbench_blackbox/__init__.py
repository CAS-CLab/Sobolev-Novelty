"""Offline controlled loader for the SRBench/PMLB black-box tasks."""

from .loader import (
    SPLIT_PROTOCOL,
    load_srbench_blackbox_dataset,
)

__all__ = ["SPLIT_PROTOCOL", "load_srbench_blackbox_dataset"]
