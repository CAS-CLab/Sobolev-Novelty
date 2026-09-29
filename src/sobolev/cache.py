"""Candidate-independent raw term value/gradient cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class CacheKey:
    canonical_term: str
    dataset_identity: str
    geometry_subset_identity: str
    input_normalization_identity: str
    derivative_order: int
    operator_configuration: str


@dataclass
class RawTermEvaluation:
    values: np.ndarray
    gradients: np.ndarray
    derivatives: tuple[sp.Expr, ...]
    symbolic_differentiation_time: float = 0.0
    value_evaluation_time: float = 0.0
    gradient_evaluation_time: float = 0.0

    @property
    def memory_bytes(self) -> int:
        return int(self.values.nbytes + self.gradients.nbytes)


class TermEvaluationCache:
    """Cache exact derivatives separately from geometry-specific raw arrays."""

    def __init__(
        self,
        enabled: bool = True,
        max_entries: int | None = None,
        max_memory_bytes: int | None = None,
        derivative_max_entries: int | None = None,
    ):
        for name, value in (
            ("max_entries", max_entries),
            ("max_memory_bytes", max_memory_bytes),
            ("derivative_max_entries", derivative_max_entries),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None")
        self.enabled = enabled
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.derivative_max_entries = derivative_max_entries
        self._raw: OrderedDict[CacheKey, RawTermEvaluation] = OrderedDict()
        self._derivatives: OrderedDict[
            tuple[str, tuple[str, ...]], tuple[sp.Expr, ...]
        ] = OrderedDict()
        self._memory_bytes = 0
        self.hits = 0
        self.misses = 0
        self.symbolic_hits = 0
        self.symbolic_misses = 0
        self.evictions = 0
        self.symbolic_evictions = 0

    def get_raw(self, key: CacheKey) -> RawTermEvaluation | None:
        if self.enabled and key in self._raw:
            self.hits += 1
            value = self._raw.pop(key)
            self._raw[key] = value
            return value
        self.misses += 1
        return None

    def put_raw(self, key: CacheKey, value: RawTermEvaluation) -> None:
        if self.enabled:
            previous = self._raw.pop(key, None)
            if previous is not None:
                self._memory_bytes -= previous.memory_bytes
            self._raw[key] = value
            self._memory_bytes += value.memory_bytes
            while self._raw and (
                (self.max_entries is not None and len(self._raw) > self.max_entries)
                or (
                    self.max_memory_bytes is not None
                    and self._memory_bytes > self.max_memory_bytes
                )
            ):
                _, evicted = self._raw.popitem(last=False)
                self._memory_bytes -= evicted.memory_bytes
                self.evictions += 1

    def get_derivatives(
        self,
        canonical_term: str,
        symbols: Sequence[sp.Symbol],
    ) -> tuple[sp.Expr, ...] | None:
        key = (canonical_term, tuple(str(symbol) for symbol in symbols))
        if self.enabled and key in self._derivatives:
            self.symbolic_hits += 1
            value = self._derivatives.pop(key)
            self._derivatives[key] = value
            return value
        self.symbolic_misses += 1
        return None

    def put_derivatives(
        self,
        canonical_term: str,
        symbols: Sequence[sp.Symbol],
        derivatives: tuple[sp.Expr, ...],
    ) -> None:
        if self.enabled:
            key = (canonical_term, tuple(str(symbol) for symbol in symbols))
            self._derivatives.pop(key, None)
            self._derivatives[key] = derivatives
            while (
                self.derivative_max_entries is not None
                and len(self._derivatives) > self.derivative_max_entries
            ):
                self._derivatives.popitem(last=False)
                self.symbolic_evictions += 1

    @property
    def memory_bytes(self) -> int:
        return self._memory_bytes

    @property
    def entry_count(self) -> int:
        return len(self._raw)

    @property
    def derivative_entry_count(self) -> int:
        return len(self._derivatives)

    def clear(self) -> None:
        self._raw.clear()
        self._derivatives.clear()
        self._memory_bytes = 0
        self.hits = self.misses = self.symbolic_hits = self.symbolic_misses = 0
        self.evictions = self.symbolic_evictions = 0
