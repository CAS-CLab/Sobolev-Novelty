"""Bounded candidate geometry, decomposition, and typed failure caches."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .geometry_state import CandidateGeometryKey, GeometryState
from .types import FailureType, TermSpec


@dataclass(frozen=True)
class DecompositionRecord:
    """Cached fitted-expression parse/decomposition or its typed failure."""

    expression: sp.Expr | None = None
    symbols: tuple[sp.Symbol, ...] = ()
    terms: tuple[TermSpec, ...] = ()
    success: bool = True
    failure_type: FailureType = FailureType.NONE
    failure_message: str = ""


class CandidateGeometryCache:
    """LRU cache with explicit memory and entry bounds.

    Geometry entries retain duplicate basis terms because the key uses a sorted
    tuple, not a set. Decomposition failures are cached separately because a
    geometry key does not exist when parsing fails.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_entries: int = 20_000,
        max_memory_bytes: int = 512 * 1024 * 1024,
        decomposition_max_entries: int = 50_000,
    ) -> None:
        self.enabled = enabled
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.decomposition_max_entries = decomposition_max_entries
        self._geometry: OrderedDict[CandidateGeometryKey, GeometryState] = OrderedDict()
        self._geometry_by_digest: dict[str, CandidateGeometryKey] = {}
        self._decomposition: OrderedDict[tuple[str, tuple[str, ...], str], DecompositionRecord] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.failure_hits = 0
        self.decomposition_hits = 0
        self.decomposition_misses = 0
        self.evictions = 0
        self.incremental_lookups = 0
        self.incremental_hits = 0
        self.memory_bytes = 0

    def get(self, key: CandidateGeometryKey) -> GeometryState | None:
        if self.enabled and key in self._geometry:
            state = self._geometry.pop(key)
            self._geometry[key] = state
            self.hits += 1
            if not state.success:
                self.failure_hits += 1
            return state
        self.misses += 1
        return None

    def put(self, state: GeometryState) -> None:
        if not self.enabled:
            return
        old = self._geometry.pop(state.key, None)
        if old is not None:
            self.memory_bytes -= old.memory_bytes
            self._geometry_by_digest.pop(old.key.digest, None)
        self._geometry[state.key] = state
        self._geometry_by_digest[state.key.digest] = state.key
        self.memory_bytes += state.memory_bytes
        self._evict_geometry()

    def get_decomposition(
        self,
        fitted_expression: str,
        feature_names: Sequence[str],
        operator_configuration: str,
    ) -> DecompositionRecord | None:
        key = (fitted_expression, tuple(feature_names), operator_configuration)
        if self.enabled and key in self._decomposition:
            record = self._decomposition.pop(key)
            self._decomposition[key] = record
            self.decomposition_hits += 1
            if not record.success:
                self.failure_hits += 1
            return record
        self.decomposition_misses += 1
        return None

    def put_decomposition(
        self,
        fitted_expression: str,
        feature_names: Sequence[str],
        operator_configuration: str,
        record: DecompositionRecord,
    ) -> None:
        if not self.enabled:
            return
        key = (fitted_expression, tuple(feature_names), operator_configuration)
        self._decomposition.pop(key, None)
        self._decomposition[key] = record
        while len(self._decomposition) > self.decomposition_max_entries:
            self._decomposition.popitem(last=False)

    @property
    def entry_count(self) -> int:
        return len(self._geometry)

    @property
    def decomposition_entry_count(self) -> int:
        return len(self._decomposition)

    def get_by_digest(self, digest: str) -> GeometryState | None:
        """Return a parent state for delta reuse without counting a key hit."""

        self.incremental_lookups += 1
        key = self._geometry_by_digest.get(digest)
        if not self.enabled or key is None or key not in self._geometry:
            return None
        state = self._geometry.pop(key)
        self._geometry[key] = state
        self.incremental_hits += 1
        return state

    def clear(self) -> None:
        self._geometry.clear()
        self._geometry_by_digest.clear()
        self._decomposition.clear()
        self.hits = self.misses = self.failure_hits = 0
        self.decomposition_hits = self.decomposition_misses = 0
        self.evictions = 0
        self.incremental_lookups = self.incremental_hits = 0
        self.memory_bytes = 0

    def _evict_geometry(self) -> None:
        while self._geometry and (
            len(self._geometry) > self.max_entries
            or self.memory_bytes > self.max_memory_bytes
        ):
            _, state = self._geometry.popitem(last=False)
            self._geometry_by_digest.pop(state.key.digest, None)
            self.memory_bytes -= state.memory_bytes
            self.evictions += 1
