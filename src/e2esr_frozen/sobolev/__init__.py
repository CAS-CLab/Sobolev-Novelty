"""Minimal Sobolev Novelty evaluator used by the E2ESR data filter."""

from .cache import TermEvaluationCache
from .config import SobolevConfig
from .evaluator import SobolevEvaluator
from .geometry_cache import CandidateGeometryCache
from .geometry_state import CandidateGeometryKey, GeometryState
from .types import EvaluationResult, FailureType, TermSpec

__all__ = [
    "EvaluationResult",
    "FailureType",
    "CandidateGeometryCache",
    "CandidateGeometryKey",
    "GeometryState",
    "SobolevConfig",
    "SobolevEvaluator",
    "TermEvaluationCache",
    "TermSpec",
]
