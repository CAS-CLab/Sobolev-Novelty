"""Propagate additive-term provenance from an original nd2py GP AST."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Sequence

import sympy as sp

from ...nd2py import nd2py as nd
from ...sobolev.decomposition import (
    ProtectedInverse,
    decompose_expand_mul,
    parse_expression,
)
from ...sobolev.types import TermSpec


NodePath = tuple[int, ...]


def path_text(path: NodePath) -> str:
    return "root" if not path else "root/" + "/".join(map(str, path))


@dataclass(frozen=True)
class TermProvenance:
    term_id: int
    canonical_term_expression: str
    source_ast_node_paths: tuple[NodePath, ...]
    source_operator_paths: tuple[NodePath, ...]
    source_subtree_ids: tuple[str, ...]
    coefficient_placeholder: bool = False
    source_term_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "canonical_term_expression": self.canonical_term_expression,
            "source_ast_node_paths": [path_text(value) for value in self.source_ast_node_paths],
            "source_operator_paths": [path_text(value) for value in self.source_operator_paths],
            "source_subtree_ids": list(self.source_subtree_ids),
            "coefficient_placeholder": self.coefficient_placeholder,
            "source_term_index": self.source_term_index,
        }


@dataclass(frozen=True)
class ProvenanceResult:
    success: bool
    raw_expression: str
    terms: tuple[TermProvenance, ...] = ()
    node_to_term_ids: tuple[tuple[NodePath, tuple[int, ...]], ...] = ()
    raw_source_term_count: int = 0
    failure_reason: str | None = None

    def associated_terms(self, path: NodePath) -> tuple[int, ...]:
        for candidate_path, term_ids in self.node_to_term_ids:
            if candidate_path == path:
                return term_ids
        return ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "raw_expression": self.raw_expression,
            "terms": [value.as_dict() for value in self.terms],
            "node_to_term_ids": {
                path_text(path): list(term_ids) for path, term_ids in self.node_to_term_ids
            },
            "raw_source_term_count": self.raw_source_term_count,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class _ExpandedSourceTerm:
    expression: sp.Expr
    node_paths: frozenset[NodePath]
    operator_paths: frozenset[NodePath]

    def add_path(self, path: NodePath, *, operator: bool) -> "_ExpandedSourceTerm":
        return _ExpandedSourceTerm(
            self.expression,
            self.node_paths | {path},
            self.operator_paths | ({path} if operator else set()),
        )


class TermProvenanceBuilder:
    """Mirror the project's structural distributor while retaining AST paths."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(str(value) for value in feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        _, self.symbols = parse_expression(self.feature_names[0], self.feature_names)

    def build(
        self,
        tree: nd.Symbol,
        fitted_terms: Sequence[TermSpec],
    ) -> ProvenanceResult:
        raw_expression = tree.to_str(number_format=".17g")
        try:
            expression, _ = parse_expression(raw_expression, self.feature_names)
            expected = tuple(decompose_expand_mul(expression, self.symbols))
            propagated = list(self._propagate(tree, ()))
            propagated.sort(key=lambda value: sp.default_sort_key(value.expression))
            actual_specs = tuple(self._single_spec(value.expression) for value in propagated)
            if tuple(value.canonical for value in actual_specs) != tuple(
                value.canonical for value in expected
            ):
                raise ValueError("propagated canonical term order differs from decompose_expand_mul")

            available: dict[str, list[int]] = {}
            for index, spec in enumerate(actual_specs):
                available.setdefault(spec.canonical, []).append(index)
            mapped: list[TermProvenance] = []
            for term_id, term in enumerate(fitted_terms):
                if not term.basis.free_symbols:
                    mapped.append(
                        TermProvenance(
                            term_id=term_id,
                            canonical_term_expression=term.canonical,
                            source_ast_node_paths=(),
                            source_operator_paths=(),
                            source_subtree_ids=(),
                            coefficient_placeholder=True,
                        )
                    )
                    continue
                matches = available.get(term.canonical, [])
                if not matches:
                    raise ValueError(
                        f"fitted term {term_id} has no unused raw provenance occurrence: "
                        f"{term.canonical}"
                    )
                source_index = matches.pop(0)
                source = propagated[source_index]
                node_paths = tuple(sorted(source.node_paths, key=_path_sort_key))
                operator_paths = tuple(sorted(source.operator_paths, key=_path_sort_key))
                mapped.append(
                    TermProvenance(
                        term_id=term_id,
                        canonical_term_expression=term.canonical,
                        source_ast_node_paths=node_paths,
                        source_operator_paths=operator_paths,
                        source_subtree_ids=tuple(path_text(path) for path in node_paths),
                        source_term_index=source_index,
                    )
                )

            node_map: dict[NodePath, list[int]] = {}
            for term in mapped:
                if term.coefficient_placeholder:
                    continue
                for path in term.source_ast_node_paths:
                    node_map.setdefault(path, []).append(term.term_id)
            node_to_terms = tuple(
                (path, tuple(sorted(set(term_ids))))
                for path, term_ids in sorted(node_map.items(), key=lambda item: _path_sort_key(item[0]))
            )
            return ProvenanceResult(
                success=True,
                raw_expression=raw_expression,
                terms=tuple(mapped),
                node_to_term_ids=node_to_terms,
                raw_source_term_count=len(propagated),
            )
        except Exception as error:
            return ProvenanceResult(
                success=False,
                raw_expression=raw_expression,
                failure_reason=f"{type(error).__name__}: {error}",
            )

    def _propagate(
        self, node: nd.Symbol, path: NodePath
    ) -> tuple[_ExpandedSourceTerm, ...]:
        if isinstance(node, nd.Add):
            values = self._propagate(node.operands[0], path + (0,)) + self._propagate(
                node.operands[1], path + (1,)
            )
            return tuple(value.add_path(path, operator=True) for value in values)
        if isinstance(node, nd.Sub):
            left = self._propagate(node.operands[0], path + (0,))
            right = tuple(
                _ExpandedSourceTerm(
                    sp.Mul(sp.Integer(-1), value.expression),
                    value.node_paths,
                    value.operator_paths,
                )
                for value in self._propagate(node.operands[1], path + (1,))
            )
            return tuple(value.add_path(path, operator=True) for value in left + right)
        if isinstance(node, nd.Mul):
            left = self._propagate(node.operands[0], path + (0,))
            right = self._propagate(node.operands[1], path + (1,))
            return tuple(
                _ExpandedSourceTerm(
                    sp.Mul(a.expression, b.expression),
                    a.node_paths | b.node_paths | {path},
                    a.operator_paths | b.operator_paths | {path},
                )
                for a, b in product(left, right)
            )
        if isinstance(node, nd.Div):
            numerator = self._propagate(node.operands[0], path + (0,))
            denominator_node = node.operands[1]
            denominator_expression, _ = parse_expression(
                denominator_node.to_str(number_format=".17g"), self.feature_names
            )
            denominator_paths = frozenset(
                child_path for child_path, _ in _walk(denominator_node, path + (1,))
            )
            denominator_operators = frozenset(
                child_path
                for child_path, child in _walk(denominator_node, path + (1,))
                if child.n_operands > 0
            )
            return tuple(
                _ExpandedSourceTerm(
                    sp.Mul(value.expression, ProtectedInverse(denominator_expression)),
                    value.node_paths | denominator_paths | {path},
                    value.operator_paths | denominator_operators | {path},
                )
                for value in numerator
            )
        if isinstance(node, nd.Neg):
            return tuple(
                _ExpandedSourceTerm(
                    sp.Mul(sp.Integer(-1), value.expression),
                    value.node_paths | {path},
                    value.operator_paths | {path},
                )
                for value in self._propagate(node.operands[0], path + (0,))
            )

        expression, _ = parse_expression(
            node.to_str(number_format=".17g"), self.feature_names
        )
        walked = tuple(_walk(node, path))
        return (
            _ExpandedSourceTerm(
                expression,
                frozenset(value_path for value_path, _ in walked),
                frozenset(
                    value_path for value_path, value in walked if value.n_operands > 0
                ),
            ),
        )

    def _single_spec(self, expression: sp.Expr) -> TermSpec:
        specs = decompose_expand_mul(expression, self.symbols)
        if len(specs) != 1:
            raise ValueError(
                f"propagated source unexpectedly decomposed into {len(specs)} terms"
            )
        return specs[0]


def _walk(node: nd.Symbol, path: NodePath) -> Iterable[tuple[NodePath, nd.Symbol]]:
    yield path, node
    for index, operand in enumerate(node.operands):
        yield from _walk(operand, path + (index,))


def _path_sort_key(path: NodePath) -> tuple[int, NodePath]:
    return (len(path), path)
