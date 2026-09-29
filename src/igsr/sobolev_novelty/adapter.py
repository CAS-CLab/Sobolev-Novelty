"""Adapter from fitted IGSR additive terms to EIC Sobolev novelty.

The mathematical implementation is intentionally *not* duplicated here.  A
successful evaluation delegates parsing/evaluation to EIC's
``SobolevEvaluator``/``SobolevConfig`` and delegates scoring to EIC's
``sobolev_penalty``.  This module only performs syntax normalization, semantic
cross-checks against NumPy, and JSON-friendly result adaptation.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class AdapterFailureType(str, Enum):
    """Typed, fail-closed adapter outcomes."""

    NONE = "none"
    DEPENDENCY_LOAD_FAILURE = "dependency_load_failure"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_SYNTAX = "unsupported_syntax"
    PARSE_FAILURE = "parse_failure"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    NO_ACTIVE_TERMS = "no_active_terms"
    EVALUATOR_FAILURE = "evaluator_failure"
    PENALTY_FAILURE = "penalty_failure"


@dataclass(frozen=True)
class EICComponents:
    """The exact EIC components used by the adapter.

    This small dependency-injection boundary also makes it possible to embed
    EIC as an installed package rather than modifying ``sys.path``.
    """

    SobolevEvaluator: type
    SobolevConfig: type
    sobolev_penalty: Callable[[Sequence[float], float], float]
    parse_expression: Callable[[object, Sequence[str]], tuple[Any, tuple[Any, ...]]]
    evaluate_sympy: Callable[[Any, Sequence[Any], np.ndarray, float], np.ndarray]
    select_geometry_indices: Callable[[int, str, int, int | None], np.ndarray]
    array_identity: Callable[..., str]
    source_root: str


@dataclass
class NoveltyDiagnostics:
    """JSON-friendly diagnostic for one fitted, single-output IGSR model."""

    success: bool = False
    penalty: float = 1.0
    term_novelties: list[float | None] = field(default_factory=list)
    normalized_terms: list[str] = field(default_factory=list)
    active_term_indices: list[int] = field(default_factory=list)
    zero_coefficient_indices: list[int] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    intercept: float | None = None
    intercept_included_as_term: bool = False
    intercept_novelty: float | None = None
    geometry_key: str | None = None
    candidate_geometry_key: str | None = None
    parent_geometry_key: str | None = None
    geometry_indices: list[int] = field(default_factory=list)
    dataset_identity: str = ""
    threshold: float | None = None
    failure_type: AdapterFailureType = AdapterFailureType.NONE
    failure_message: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)
    eic_result: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-serializable mapping."""

        output = asdict(self)
        output["failure_type"] = self.failure_type.value
        return _json_safe(output)


@dataclass(frozen=True)
class FixedGeometry:
    """One immutable training geometry shared by a complete IGSR search."""

    dataset_identity: str
    indices: np.ndarray


_NUMPY_FUNCTIONS: dict[str, str] = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "abs",
    "power": "pow",
    "square": "pow2",
}
_NUMPY_CONSTANTS: dict[str, str] = {"pi": "pi", "e": "E"}
_BARE_FUNCTIONS = frozenset(_NUMPY_FUNCTIONS)
_RESERVED_NAMES = frozenset({"np", "numpy", *_BARE_FUNCTIONS, "pi", "E"})
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Attribute,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


def load_eic_components(eic_src_root: str | Path) -> EICComponents:
    """Load and verify EIC's canonical Sobolev implementation.

    ``eic_src_root`` must point at the directory containing the ``sobolev``
    package (normally ``EIC/src``).  If a different top-level ``sobolev``
    package is already imported, loading fails instead of silently mixing two
    implementations.
    """

    root = Path(eic_src_root).expanduser().resolve()
    package_root = root / "sobolev"
    required = (
        package_root / "__init__.py",
        package_root / "config.py",
        package_root / "evaluator.py",
        package_root / "scoring.py",
        package_root / "decomposition.py",
        package_root / "signature.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ImportError(
            f"EIC Sobolev source is incomplete under {root}: missing={missing}"
        )

    loaded = sys.modules.get("sobolev")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None or not _is_relative_to(
            Path(loaded_file).resolve(), package_root
        ):
            raise ImportError(
                "A different 'sobolev' package is already imported: "
                f"{loaded_file!r}; expected under {package_root}"
            )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    config_module = importlib.import_module("sobolev.config")
    evaluator_module = importlib.import_module("sobolev.evaluator")
    scoring_module = importlib.import_module("sobolev.scoring")
    decomposition_module = importlib.import_module("sobolev.decomposition")
    signature_module = importlib.import_module("sobolev.signature")
    for module in (
        config_module,
        evaluator_module,
        scoring_module,
        decomposition_module,
        signature_module,
    ):
        _verify_module_source(module, package_root)
    return EICComponents(
        SobolevEvaluator=evaluator_module.SobolevEvaluator,
        SobolevConfig=config_module.SobolevConfig,
        sobolev_penalty=scoring_module.sobolev_penalty,
        parse_expression=decomposition_module.parse_expression,
        evaluate_sympy=signature_module.evaluate_sympy,
        select_geometry_indices=signature_module.select_geometry_indices,
        array_identity=signature_module.array_identity,
        source_root=str(root),
    )


class SobolevNoveltyAdapter:
    """Evaluate IGSR's surviving additive terms with EIC Sobolev novelty."""

    def __init__(
        self,
        *,
        eic_src_root: str | Path | None = None,
        components: EICComponents | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        zero_coefficient_tolerance: float = 0.0,
        semantic_rtol: float = 1e-10,
        semantic_atol: float = 1e-12,
    ) -> None:
        if (eic_src_root is None) == (components is None):
            raise ValueError("Provide exactly one of eic_src_root or components")
        if zero_coefficient_tolerance < 0:
            raise ValueError("zero_coefficient_tolerance must be non-negative")
        if semantic_rtol < 0 or semantic_atol < 0:
            raise ValueError("semantic tolerances must be non-negative")
        self.zero_coefficient_tolerance = float(zero_coefficient_tolerance)
        self.semantic_rtol = float(semantic_rtol)
        self.semantic_atol = float(semantic_atol)
        self._components: EICComponents | None = None
        self._dependency_error = ""
        try:
            self._components = components or load_eic_components(eic_src_root)  # type: ignore[arg-type]
            config = self._components.SobolevConfig(**dict(config_overrides or {}))
            self._evaluator = self._components.SobolevEvaluator(config)
        except Exception as error:  # dependency failures are reported by evaluate()
            self._evaluator = None
            self._dependency_error = f"{type(error).__name__}: {error}"

    def select_fixed_geometry(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray | None = None,
        task_identity: str,
        run_seed: int,
        geometry_seed: int = 20260731,
        sample_size: int | None = 64,
    ) -> FixedGeometry:
        """Bind a search geometry to task, seed, and exact training contents.

        Selection and content hashing are delegated to EIC's canonical
        ``select_geometry_indices`` and ``array_identity`` helpers.  The caller
        invokes this once per run and reuses the returned indices for every
        candidate, which prevents candidate-dependent resampling.
        """

        if self._components is None:
            raise RuntimeError(
                self._dependency_error or "EIC components are unavailable"
            )
        points = np.asarray(X, dtype=float)
        if points.ndim != 2 or points.shape[0] < 1:
            raise ValueError(f"X must be a non-empty matrix, got shape {points.shape}")
        if not isinstance(task_identity, str) or not task_identity.strip():
            raise ValueError("task_identity must be a non-empty string")
        if sample_size is not None and int(sample_size) < 1:
            raise ValueError("sample_size must be positive or None")

        targets = None if y is None else np.asarray(y, dtype=float)
        if targets is not None and (
            targets.ndim not in {1, 2} or targets.shape[0] != points.shape[0]
        ):
            raise ValueError(
                f"y must have the same number of rows as X, got X={points.shape}, y={targets.shape}"
            )
        content_identity = self._components.array_identity(
            points, *(() if targets is None else (targets,))
        )
        dataset_identity = (
            f"igsr-task={task_identity}|run-seed={int(run_seed)}|"
            f"train-content-sha256={content_identity}"
        )
        indices = self._components.select_geometry_indices(
            int(points.shape[0]),
            dataset_identity,
            int(geometry_seed),
            None if sample_size is None else int(sample_size),
        )
        indices = np.asarray(indices, dtype=np.int64)
        indices.setflags(write=False)
        return FixedGeometry(dataset_identity=dataset_identity, indices=indices)

    def evaluate(
        self,
        *,
        terms: Sequence[str],
        coefficients: Sequence[float],
        intercept: float | None,
        feature_names: Sequence[str],
        X: np.ndarray,
        geometry_indices: Sequence[int] | np.ndarray,
        dataset_identity: str,
        parent_geometry_key: str | None = None,
    ) -> NoveltyDiagnostics:
        """Return Sobolev diagnostics without changing IGSR search state.

        Zero-coefficient terms are excluded from EIC evaluation and represented
        by ``None`` in ``term_novelties``.  A non-zero fitted intercept is
        included as the constant additive basis term ``1`` and its separate
        novelty is reported in ``intercept_novelty``.  This matches EIC's
        decomposition of the complete fitted expression.
        """

        started = time.perf_counter()
        diagnostics = NoveltyDiagnostics(
            term_novelties=[None] * len(terms),
            parent_geometry_key=parent_geometry_key,
            dataset_identity=str(dataset_identity),
        )
        if self._components is None or self._evaluator is None:
            return self._fail(
                diagnostics,
                AdapterFailureType.DEPENDENCY_LOAD_FAILURE,
                self._dependency_error or "EIC components are unavailable",
                started,
            )
        validation_started = time.perf_counter()
        try:
            points, names, indices, weights, scalar_intercept = _validate_inputs(
                terms,
                coefficients,
                intercept,
                feature_names,
                X,
                geometry_indices,
                dataset_identity,
            )
        except (TypeError, ValueError) as error:
            return self._fail(
                diagnostics,
                AdapterFailureType.INVALID_INPUT,
                f"{type(error).__name__}: {error}",
                started,
            )
        diagnostics.coefficients = weights.tolist()
        diagnostics.intercept = scalar_intercept
        diagnostics.geometry_indices = indices.tolist()
        diagnostics.geometry_key = _geometry_key(
            points, names, indices, str(dataset_identity)
        )
        diagnostics.runtime["input_validation_seconds"] = (
            time.perf_counter() - validation_started
        )

        active = np.flatnonzero(np.abs(weights) > self.zero_coefficient_tolerance)
        zeros = np.flatnonzero(np.abs(weights) <= self.zero_coefficient_tolerance)
        diagnostics.active_term_indices = active.astype(int).tolist()
        diagnostics.zero_coefficient_indices = zeros.astype(int).tolist()
        include_intercept = (
            scalar_intercept is not None
            and abs(scalar_intercept) > self.zero_coefficient_tolerance
        )
        diagnostics.intercept_included_as_term = include_intercept
        if active.size == 0 and not include_intercept:
            return self._fail(
                diagnostics,
                AdapterFailureType.NO_ACTIVE_TERMS,
                "All surviving terms and the fitted intercept have zero coefficient; "
                "no active structural term remains",
                started,
            )

        normalization_started = time.perf_counter()
        # Remain position-aligned with the fitted coefficient vector, but skip
        # parsing zero-weight terms: they are absent from the fitted expression
        # and unsupported inactive syntax must not fail an otherwise valid
        # candidate diagnostic.
        normalized_terms: list[str] = [str(term) for term in terms]
        parsed_terms: list[Any | None] = [None] * len(terms)
        symbols: tuple[Any, ...] | None = None
        try:
            for term_index in active:
                term = terms[int(term_index)]
                normalized = _normalize_term(term, names)
                parsed, parsed_symbols = self._components.parse_expression(
                    normalized, names
                )
                if symbols is None:
                    symbols = parsed_symbols
                elif tuple(parsed_symbols) != tuple(symbols):
                    raise ValueError("EIC returned inconsistent symbols across terms")
                normalized_terms[int(term_index)] = normalized
                parsed_terms[int(term_index)] = parsed
            if symbols is None:
                # A nonzero intercept may be the only active basis.
                _, symbols = self._components.parse_expression("1", names)
        except UnsupportedTermSyntax as error:
            diagnostics.normalized_terms = normalized_terms
            return self._fail(
                diagnostics,
                AdapterFailureType.UNSUPPORTED_SYNTAX,
                str(error),
                started,
            )
        except Exception as error:
            diagnostics.normalized_terms = normalized_terms
            return self._fail(
                diagnostics,
                AdapterFailureType.PARSE_FAILURE,
                f"{type(error).__name__}: {error}",
                started,
            )
        diagnostics.normalized_terms = normalized_terms
        diagnostics.runtime["normalization_and_parse_seconds"] = (
            time.perf_counter() - normalization_started
        )
        assert symbols is not None

        semantic_started = time.perf_counter()
        geometry_points = points[indices]
        local_values = {
            name: geometry_points[:, column] for column, name in enumerate(names)
        }
        try:
            for term_index in active:
                raw_values = _evaluate_numpy_term(terms[int(term_index)], local_values)
                eic_values = self._components.evaluate_sympy(
                    parsed_terms[int(term_index)],
                    symbols,
                    geometry_points,
                    float(self._evaluator.config.protected_epsilon),
                )
                _assert_same_semantics(
                    raw_values,
                    eic_values,
                    term_index=int(term_index),
                    raw_term=terms[int(term_index)],
                    normalized_term=normalized_terms[int(term_index)],
                    geometry_indices=indices,
                    rtol=self.semantic_rtol,
                    atol=self.semantic_atol,
                )
        except UnsupportedTermSyntax as error:
            return self._fail(
                diagnostics,
                AdapterFailureType.UNSUPPORTED_SYNTAX,
                str(error),
                started,
            )
        except SemanticMismatch as error:
            diagnostics.runtime["semantic_validation_seconds"] = (
                time.perf_counter() - semantic_started
            )
            return self._fail(
                diagnostics,
                AdapterFailureType.SEMANTIC_MISMATCH,
                str(error),
                started,
            )
        except Exception as error:
            diagnostics.runtime["semantic_validation_seconds"] = (
                time.perf_counter() - semantic_started
            )
            return self._fail(
                diagnostics,
                AdapterFailureType.SEMANTIC_MISMATCH,
                f"Semantic validation failed: {type(error).__name__}: {error}",
                started,
            )
        diagnostics.runtime["semantic_validation_seconds"] = (
            time.perf_counter() - semantic_started
        )

        evaluator_started = time.perf_counter()
        evaluator_terms = [
            (float(weights[index]), parsed_terms[index])
            for index in diagnostics.active_term_indices
        ]
        raw_expression_parts = [
            f"({weights[index]:.17g})*({normalized_terms[index]})"
            for index in diagnostics.active_term_indices
        ]
        if include_intercept:
            try:
                constant_basis, constant_symbols = self._components.parse_expression(
                    "1", names
                )
            except Exception as error:
                return self._fail(
                    diagnostics,
                    AdapterFailureType.PARSE_FAILURE,
                    f"Could not construct intercept basis: {type(error).__name__}: {error}",
                    started,
                )
            if tuple(constant_symbols) != tuple(symbols):
                return self._fail(
                    diagnostics,
                    AdapterFailureType.PARSE_FAILURE,
                    "EIC returned inconsistent symbols for the intercept basis",
                    started,
                )
            evaluator_terms.append((float(scalar_intercept), constant_basis))
            raw_expression_parts.append(f"({scalar_intercept:.17g})*(1)")
        raw_expression = " + ".join(raw_expression_parts)
        try:
            result = self._evaluator.evaluate_terms(
                evaluator_terms,
                symbols,
                points,
                dataset_identity=str(dataset_identity),
                geometry_indices=indices,
                raw_expression=raw_expression,
                parent_geometry_key=parent_geometry_key,
            )
        except Exception as error:
            diagnostics.runtime["evaluator_seconds"] = (
                time.perf_counter() - evaluator_started
            )
            return self._fail(
                diagnostics,
                AdapterFailureType.EVALUATOR_FAILURE,
                f"SobolevEvaluator raised {type(error).__name__}: {error}",
                started,
            )
        diagnostics.runtime["evaluator_seconds"] = (
            time.perf_counter() - evaluator_started
        )
        diagnostics.eic_result = result.as_dict()
        diagnostics.candidate_geometry_key = result.candidate_geometry_key
        diagnostics.threshold = float(result.threshold)
        if not result.success:
            return self._fail(
                diagnostics,
                AdapterFailureType.EVALUATOR_FAILURE,
                f"EIC {result.failure_type.value}: {result.failure_message}",
                started,
            )
        active_novelties = result.term_novelties[: len(diagnostics.active_term_indices)]
        for source_index, novelty in zip(
            diagnostics.active_term_indices, active_novelties, strict=True
        ):
            diagnostics.term_novelties[source_index] = float(novelty)
        if include_intercept:
            diagnostics.intercept_novelty = float(result.term_novelties[-1])

        penalty_started = time.perf_counter()
        try:
            diagnostics.penalty = float(
                self._components.sobolev_penalty(
                    result.term_novelties, result.threshold
                )
            )
        except Exception as error:
            diagnostics.runtime["penalty_seconds"] = (
                time.perf_counter() - penalty_started
            )
            return self._fail(
                diagnostics,
                AdapterFailureType.PENALTY_FAILURE,
                f"sobolev_penalty raised {type(error).__name__}: {error}",
                started,
            )
        diagnostics.runtime["penalty_seconds"] = time.perf_counter() - penalty_started
        diagnostics.success = True
        diagnostics.failure_type = AdapterFailureType.NONE
        diagnostics.runtime["total_seconds"] = time.perf_counter() - started
        return diagnostics

    @staticmethod
    def _fail(
        diagnostics: NoveltyDiagnostics,
        failure_type: AdapterFailureType,
        message: str,
        started: float,
    ) -> NoveltyDiagnostics:
        diagnostics.success = False
        diagnostics.penalty = 1.0
        diagnostics.failure_type = failure_type
        diagnostics.failure_message = message
        diagnostics.runtime["total_seconds"] = time.perf_counter() - started
        return diagnostics


class UnsupportedTermSyntax(ValueError):
    """The NumPy expression is outside the deliberately small term grammar."""


class SemanticMismatch(ValueError):
    """NumPy/IGSR and EIC do not have the same values on fixed geometry."""


def _validate_inputs(
    terms: Sequence[str],
    coefficients: Sequence[float],
    intercept: float | None,
    feature_names: Sequence[str],
    X: np.ndarray,
    geometry_indices: Sequence[int] | np.ndarray,
    dataset_identity: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, float | None]:
    if not terms or len(terms) != len(coefficients):
        raise ValueError(
            "terms and coefficients must be non-empty and have equal length"
        )
    if any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValueError("every term must be a non-empty string")
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    invalid_names = [
        name for name in names if not name.isidentifier() or name in _RESERVED_NAMES
    ]
    if invalid_names:
        raise ValueError(
            f"feature names are not safe expression identifiers: {invalid_names}"
        )
    points = np.asarray(X, dtype=float)
    if points.ndim != 2 or points.shape[1] != len(names) or points.shape[0] == 0:
        raise ValueError(f"X shape {points.shape} does not match {len(names)} features")
    if not np.all(np.isfinite(points)):
        raise ValueError("X must contain only finite values")
    weights = np.asarray(coefficients, dtype=float)
    if (
        weights.ndim != 1
        or weights.shape != (len(terms),)
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError("coefficients must be a finite one-dimensional vector")
    scalar_intercept = None if intercept is None else float(intercept)
    if scalar_intercept is not None and not math.isfinite(scalar_intercept):
        raise ValueError("intercept must be finite or None")
    indices = np.asarray(geometry_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.any(indices < 0)
        or np.any(indices >= len(points))
        or np.unique(indices).size != indices.size
    ):
        raise ValueError(
            "geometry_indices must be a non-empty, unique, in-range vector"
        )
    if not isinstance(dataset_identity, str) or not dataset_identity:
        raise ValueError("dataset_identity must be a non-empty string")
    return points, names, indices, weights, scalar_intercept


def _normalize_term(term: str, feature_names: Sequence[str]) -> str:
    if len(term) > 4096:
        raise UnsupportedTermSyntax("term exceeds the 4096-character safety limit")
    try:
        tree = ast.parse(term, mode="eval")
    except SyntaxError as error:
        raise UnsupportedTermSyntax(
            f"invalid Python expression {term!r}: {error}"
        ) from error
    nodes = list(ast.walk(tree))
    if len(nodes) > 512:
        raise UnsupportedTermSyntax("term exceeds the 512-node AST safety limit")
    allowed_names = (
        set(feature_names) | set(_BARE_FUNCTIONS) | {"np", "numpy", "pi", "E"}
    )
    for node in nodes:
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise UnsupportedTermSyntax(
                f"unsupported AST node {type(node).__name__} in {term!r}"
            )
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise UnsupportedTermSyntax(
                    f"only real numeric literals are allowed in {term!r}"
                )
        elif isinstance(node, ast.Name) and node.id not in allowed_names:
            raise UnsupportedTermSyntax(f"unknown name {node.id!r} in {term!r}")
        elif isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id not in {
                "np",
                "numpy",
            }:
                raise UnsupportedTermSyntax(
                    f"only one-level np/numpy attributes are allowed in {term!r}"
                )
            if node.attr not in _NUMPY_FUNCTIONS and node.attr not in _NUMPY_CONSTANTS:
                raise UnsupportedTermSyntax(
                    f"unsupported NumPy attribute {node.attr!r} in {term!r}"
                )
        elif isinstance(node, ast.Call):
            if node.keywords:
                raise UnsupportedTermSyntax(
                    f"keyword arguments are not allowed in {term!r}"
                )
            if isinstance(node.func, ast.Name):
                if node.func.id not in _BARE_FUNCTIONS:
                    raise UnsupportedTermSyntax(
                        f"unsupported function {node.func.id!r} in {term!r}"
                    )
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            else:
                raise UnsupportedTermSyntax(f"unsupported call target in {term!r}")
            expected_arity = 2 if function_name == "power" else 1
            if len(node.args) != expected_arity:
                raise UnsupportedTermSyntax(
                    f"function {function_name!r} expects {expected_arity} argument(s) in {term!r}"
                )
    normalized = _NumpyToEIC().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.unparse(normalized.body)


class _NumpyToEIC(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if node.attr in _NUMPY_CONSTANTS:
            return ast.copy_location(
                ast.Name(_NUMPY_CONSTANTS[node.attr], ast.Load()), node
            )
        return ast.copy_location(
            ast.Name(_NUMPY_FUNCTIONS[node.attr], ast.Load()), node
        )

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _NUMPY_FUNCTIONS:
            return ast.copy_location(
                ast.Name(_NUMPY_FUNCTIONS[node.id], ast.Load()), node
            )
        return node


def _evaluate_numpy_term(
    term: str, local_values: Mapping[str, np.ndarray]
) -> np.ndarray:
    # Re-validate here so this function remains safe if called independently of
    # normalization.  The globals contain no builtins and only explicit ufuncs.
    _normalize_term(term, tuple(local_values))
    functions = {name: getattr(np, name) for name in _BARE_FUNCTIONS if name != "abs"}
    functions["abs"] = np.abs
    globals_dict: dict[str, Any] = {
        "__builtins__": {},
        "np": np,
        "numpy": np,
        "pi": np.pi,
        "E": np.e,
        **functions,
    }
    tree = ast.parse(term, mode="eval")
    with np.errstate(all="ignore"):
        value = eval(
            compile(tree, "<igsr-sobolev-term>", "eval"),
            globals_dict,
            dict(local_values),
        )
    array = np.asarray(value)
    if np.iscomplexobj(array):
        finite_imaginary = np.abs(np.imag(array))[np.isfinite(np.imag(array))]
        if finite_imaginary.size and float(np.max(finite_imaginary)) > 1e-10:
            array = np.full(len(next(iter(local_values.values()))), np.nan)
        else:
            array = np.real(array)
    array = np.asarray(array, dtype=float)
    n_points = len(next(iter(local_values.values())))
    if array.ndim == 0 or array.size == 1:
        return np.full(n_points, float(array.ravel()[0] if array.ndim else array))
    array = array.ravel()
    if array.shape != (n_points,):
        raise ValueError(
            f"NumPy term produced shape {array.shape}, expected {(n_points,)}"
        )
    return array


def _assert_same_semantics(
    numpy_values: np.ndarray,
    eic_values: np.ndarray,
    *,
    term_index: int,
    raw_term: str,
    normalized_term: str,
    geometry_indices: np.ndarray,
    rtol: float,
    atol: float,
) -> None:
    numpy_values = np.asarray(numpy_values, dtype=float)
    eic_values = np.asarray(eic_values, dtype=float)
    if numpy_values.shape != eic_values.shape:
        raise SemanticMismatch(
            f"term[{term_index}] shape mismatch: NumPy={numpy_values.shape}, EIC={eic_values.shape}"
        )
    numpy_finite = np.isfinite(numpy_values)
    eic_finite = np.isfinite(eic_values)
    mask_difference = np.flatnonzero(numpy_finite != eic_finite)
    if mask_difference.size:
        local_index = int(mask_difference[0])
        raise SemanticMismatch(
            f"term[{term_index}] finite-mask mismatch at data row {int(geometry_indices[local_index])}: "
            f"raw={raw_term!r}, normalized={normalized_term!r}, "
            f"NumPy={numpy_values[local_index]!r}, EIC={eic_values[local_index]!r}"
        )
    finite = numpy_finite & eic_finite
    close = np.isclose(numpy_values[finite], eic_values[finite], rtol=rtol, atol=atol)
    if not np.all(close):
        finite_positions = np.flatnonzero(finite)
        local_index = int(finite_positions[int(np.flatnonzero(~close)[0])])
        raise SemanticMismatch(
            f"term[{term_index}] value mismatch at data row {int(geometry_indices[local_index])}: "
            f"raw={raw_term!r}, normalized={normalized_term!r}, "
            f"NumPy={numpy_values[local_index]!r}, EIC={eic_values[local_index]!r}"
        )


def _geometry_key(
    points: np.ndarray,
    feature_names: Sequence[str],
    indices: np.ndarray,
    dataset_identity: str,
) -> str:
    geometry = np.ascontiguousarray(points[indices], dtype="<f8")
    header = json.dumps(
        {
            "version": "igsr-sobolev-geometry-v1",
            "dataset_identity": dataset_identity,
            "feature_names": list(feature_names),
            "shape": list(geometry.shape),
            "geometry_indices": indices.astype(int).tolist(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(geometry.tobytes(order="C"))
    return digest.hexdigest()


def _verify_module_source(module: ModuleType, package_root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None or not _is_relative_to(
        Path(module_file).resolve(), package_root
    ):
        raise ImportError(
            f"EIC module {module.__name__} came from unexpected path {module_file!r}"
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
