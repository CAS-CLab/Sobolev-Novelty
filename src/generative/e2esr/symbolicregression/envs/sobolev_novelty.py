"""Sobolev-Novelty rejection filter for E2ESR expression trees.

The online path preserves E2ESR's expression tree, performs a structural
``expand_mul`` decomposition, and propagates values and first derivatives in
one vectorised pass.  :func:`evaluate_reference` provides the symbolic/SVD
reference evaluator used for tests and numerical audits.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np

from src.e2esr_frozen.sobolev.config import TAU_THEORY, SobolevConfig
from src.e2esr_frozen.sobolev.evaluator import SobolevEvaluator
from src.e2esr_frozen.sobolev.novelty import compute_novelties, reference_novelties
from src.e2esr_frozen.sobolev.signature import stable_norm, stable_rms


SUPPORTED_BINARY = frozenset({"add", "sub", "mul", "div", "pow", "max", "min"})
SUPPORTED_UNARY = frozenset(
    {
        "abs",
        "inv",
        "sqrt",
        "log",
        "exp",
        "sin",
        "arcsin",
        "cos",
        "arccos",
        "tan",
        "arctan",
        "pow2",
        "pow3",
        "id",
    }
)
NUMERIC_CONSTANTS = {
    "e": math.e,
    "pi": math.pi,
    "euler_gamma": 0.5772156649015329,
}


class StructuralSNError(ValueError):
    """A failure depending only on expression structure, safe to cache."""


@dataclass(frozen=True)
class SobolevFilterConfig:
    """Frozen online-filter settings written to logs and checkpoints."""

    threshold: float = TAU_THEORY
    geometry_rows: int = 200
    min_valid_rows: int = 32
    value_weight: float = 1.0
    gradient_weight: float = 1.0
    geometry_seed: int = 20260806
    input_scale_tolerance: float = 1e-12
    output_scale_tolerance: float = 1e-12
    zero_signature_tolerance: float = 1e-15
    rcond: float = 1e-10
    gram_condition_threshold: float = 1e6
    max_expanded_terms: int = 4096

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if self.geometry_rows < 1:
            raise ValueError("geometry_rows must be positive")
        if self.min_valid_rows < 1 or self.min_valid_rows > self.geometry_rows:
            raise ValueError("min_valid_rows must be in [1, geometry_rows]")
        if self.value_weight <= 0 or self.gradient_weight < 0:
            raise ValueError("value_weight must be positive and gradient_weight non-negative")
        if self.input_scale_tolerance <= 0 or self.output_scale_tolerance <= 0:
            raise ValueError("scale tolerances must be positive")
        if self.zero_signature_tolerance <= 0 or self.rcond <= 0:
            raise ValueError("zero-signature tolerance and rcond must be positive")
        if self.gram_condition_threshold <= 1:
            raise ValueError("gram_condition_threshold must exceed one")
        if self.max_expanded_terms < 1:
            raise ValueError("max_expanded_terms must be positive")


@dataclass(frozen=True)
class TermFactor:
    node: Any
    inverse: bool = False


@dataclass(frozen=True)
class TermPlan:
    coefficient: float
    factors: tuple[TermFactor, ...]

    @property
    def canonical(self) -> str:
        if not self.factors:
            return "1"
        return "*".join(
            ("inv(" + factor.node.prefix() + ")") if factor.inverse else factor.node.prefix()
            for factor in self.factors
        )


@dataclass
class SobolevFilterResult:
    success: bool = False
    accepted: bool = False
    failure_type: str = "none"
    failure_message: str = ""
    term_count: int = 0
    terms: list[str] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    novelties: list[float] = field(default_factory=list)
    min_novelty: float | None = None
    mean_novelty: float | None = None
    valid_rows: int = 0
    geometry_indices: list[int] = field(default_factory=list)
    input_means: list[float] = field(default_factory=list)
    input_scales: list[float] = field(default_factory=list)
    near_zero_scale_dims: list[int] = field(default_factory=list)
    output_scale: float | None = None
    signature_shape: tuple[int, int] = (0, 0)
    term_norms: list[float] = field(default_factory=list)
    rank: int = 0
    condition_number: float | None = None
    minimum_singular_value: float | None = None
    algorithm_used: str = "not_run"
    fallback_reason: str | None = None
    gram_reference_max_abs_error: float | None = None
    elapsed_seconds: float = 0.0
    decomposition_seconds: float = 0.0
    propagation_seconds: float = 0.0
    signature_seconds: float = 0.0
    linear_algebra_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuralFailureCache:
    """Small bounded cache for structure-only failures.

    Domain, finite-mask, scale, and linear-algebra failures depend on the
    sampled geometry and are intentionally never cached here.
    """

    def __init__(self, max_entries: int = 50_000) -> None:
        self.max_entries = int(max_entries)
        self._values: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[str, str] | None:
        value = self._values.get(key)
        if value is None:
            self.misses += 1
            return None
        self._values.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, failure_type: str, message: str) -> None:
        self._values[key] = (failure_type, message)
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)


def deterministic_geometry_indices(
    n_rows: int,
    formula_identity: str,
    seed: int,
    geometry_rows: int,
) -> np.ndarray:
    """Return a reproducible subset; no per-call random draw is performed."""

    if n_rows < 1:
        raise ValueError("input geometry is empty")
    if geometry_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    payload = f"e2esr-sn|{seed}|{geometry_rows}|{n_rows}|{formula_identity}".encode()
    stable_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(stable_seed)
    return np.sort(rng.choice(n_rows, size=geometry_rows, replace=False)).astype(np.int64)


class FastSobolevNovelty:
    """Vectorised AST value/gradient evaluator plus conditioned Gram path."""

    def __init__(
        self,
        config: SobolevFilterConfig | None = None,
        failure_cache: StructuralFailureCache | None = None,
    ) -> None:
        self.config = config or SobolevFilterConfig()
        self.failure_cache = failure_cache or StructuralFailureCache()

    def evaluate(
        self,
        tree: Any,
        X: np.ndarray,
        *,
        geometry_indices: np.ndarray | None = None,
        audit_gram_against_reference: bool = False,
    ) -> SobolevFilterResult:
        start = time.perf_counter()
        result = SobolevFilterResult()
        formula_identity: str | None = None
        structural_cache_key: str | None = None
        try:
            points = np.asarray(X, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] < 1:
                return _fail(result, "invalid_input", f"expected non-empty 2-D X, got {points.shape}", start)
            if not np.all(np.isfinite(points)):
                return _fail(result, "invalid_input", "X contains NaN or Inf", start)

            formula_identity = tree.prefix()
            # A prefix alone is not a sufficient structural identity: variable
            # validity depends on input dimension, constant subexpressions can
            # depend on the generator's abs-domain convention, and an expansion
            # cap failure depends on the configured cap.  Including all three
            # prevents a cached failure from changing a later evaluation.
            use_abs = bool(getattr(getattr(tree, "params", None), "use_abs", False))
            structural_cache_key = (
                f"dimension={points.shape[1]}|use_abs={int(use_abs)}|"
                f"max_expanded_terms={self.config.max_expanded_terms}|{formula_identity}"
            )
            cached_failure = self.failure_cache.get(structural_cache_key)
            if cached_failure is not None:
                failure_type, message = cached_failure
                return _fail(result, failure_type, message + " [structural failure cache]", start)

            input_means = np.mean(points, axis=0)
            input_scales = np.std(points, axis=0, ddof=0)
            result.input_means = input_means.tolist()
            result.input_scales = input_scales.tolist()
            near_zero = np.flatnonzero(input_scales <= self.config.input_scale_tolerance)
            result.near_zero_scale_dims = near_zero.astype(int).tolist()
            if near_zero.size:
                return _fail(
                    result,
                    "constant_input_scale",
                    f"near-zero input standard deviation in dimensions {result.near_zero_scale_dims}",
                    start,
                )

            decomposition_start = time.perf_counter()
            try:
                terms = _decompose_expand_mul(tree, points.shape[1], self.config.max_expanded_terms)
            except StructuralSNError as error:
                message = f"{type(error).__name__}: {error}"
                self.failure_cache.put(structural_cache_key, "decomposition_failure", message)
                return _fail(result, "decomposition_failure", message, start)
            result.decomposition_seconds = time.perf_counter() - decomposition_start
            result.term_count = len(terms)
            result.terms = [term.canonical for term in terms]
            result.coefficients = [float(term.coefficient) for term in terms]

            if geometry_indices is None:
                geometry_indices = deterministic_geometry_indices(
                    len(points), formula_identity, self.config.geometry_seed, self.config.geometry_rows
                )
            indices = np.asarray(geometry_indices, dtype=np.int64)
            if (
                indices.ndim != 1
                or indices.size == 0
                or np.any(indices < 0)
                or np.any(indices >= len(points))
                or np.unique(indices).size != indices.size
            ):
                return _fail(result, "invalid_input", "invalid geometry indices", start)
            geometry = points[indices]
            result.geometry_indices = indices.astype(int).tolist()

            propagation_start = time.perf_counter()
            cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            full_values, _ = _evaluate_node(tree, geometry, cache)
            raw_values: list[np.ndarray] = []
            raw_gradients: list[np.ndarray] = []
            for term in terms:
                values, gradients = _evaluate_term(term, geometry, cache)
                raw_values.append(values)
                raw_gradients.append(gradients)
            result.propagation_seconds = time.perf_counter() - propagation_start
            values = np.column_stack(raw_values)
            gradients = np.stack(raw_gradients, axis=2)  # rows, dimensions, terms

            signature_start = time.perf_counter()
            valid = np.all(np.isfinite(geometry), axis=1)
            valid &= np.isfinite(full_values)
            valid &= np.all(np.isfinite(values), axis=1)
            valid &= np.all(np.isfinite(gradients), axis=(1, 2))
            result.valid_rows = int(valid.sum())
            if result.valid_rows < self.config.min_valid_rows:
                evaluated_arrays = (full_values, values, gradients)
                if any(np.any(np.isinf(array)) for array in evaluated_arrays):
                    failure_type = "numerical_overflow"
                elif np.any(~np.isfinite(full_values)) or np.any(~np.isfinite(values)):
                    failure_type = "domain_failure"
                elif np.any(~np.isfinite(gradients)):
                    failure_type = "gradient_failure"
                else:
                    failure_type = "insufficient_valid_samples"
                return _fail(
                    result,
                    failure_type,
                    f"only {result.valid_rows}/{len(geometry)} shared finite rows; need {self.config.min_valid_rows}",
                    start,
                )
            valid_values = values[valid]
            valid_gradients = gradients[valid]
            output_scale = stable_rms(full_values[valid])
            result.output_scale = output_scale
            if not np.isfinite(output_scale) or output_scale < self.config.output_scale_tolerance:
                return _fail(
                    result,
                    "zero_output_scale",
                    f"candidate RMS {output_scale!r} is below {self.config.output_scale_tolerance}",
                    start,
                )
            n_valid = result.valid_rows
            dimension = geometry.shape[1]
            blocks = [
                math.sqrt(self.config.value_weight / n_valid) * valid_values / output_scale
            ]
            if self.config.gradient_weight > 0:
                gradients_z = valid_gradients * input_scales[None, :, None]
                factor = math.sqrt(self.config.gradient_weight / (n_valid * dimension)) / output_scale
                blocks.extend(factor * gradients_z[:, index, :] for index in range(dimension))
            signatures = np.vstack(blocks)
            if not np.all(np.isfinite(signatures)):
                return _fail(result, "numerical_overflow", "non-finite normalized signature", start)
            term_norms = np.asarray(
                [stable_norm(signatures[:, index]) for index in range(signatures.shape[1])]
            )
            zero_terms = np.flatnonzero(term_norms <= self.config.zero_signature_tolerance)
            result.term_norms = term_norms.tolist()
            result.signature_shape = tuple(int(value) for value in signatures.shape)
            result.signature_seconds = time.perf_counter() - signature_start
            if zero_terms.size:
                return _fail(
                    result,
                    "zero_signature",
                    f"zero Sobolev signature for term indices {(zero_terms + 1).tolist()}",
                    start,
                )

            linear_start = time.perf_counter()
            computation = compute_novelties(
                signatures,
                term_norms,
                self.config.rcond,
                True,
                self.config.gram_condition_threshold,
            )
            result.linear_algebra_seconds = time.perf_counter() - linear_start
            result.novelties = computation.novelties.tolist()
            result.min_novelty = float(np.min(computation.novelties))
            result.mean_novelty = float(np.mean(computation.novelties))
            result.rank = computation.rank
            result.condition_number = computation.condition_number
            result.minimum_singular_value = (
                float(computation.singular_values[-1]) if computation.singular_values.size else None
            )
            result.algorithm_used = computation.algorithm_used
            result.fallback_reason = computation.fallback_reason
            if audit_gram_against_reference and computation.algorithm_used == "gram_cholesky":
                normalized = signatures / term_norms[None, :]
                trusted, _ = reference_novelties(normalized, self.config.rcond)
                result.gram_reference_max_abs_error = float(
                    np.max(np.abs(trusted - computation.novelties))
                )
            result.success = True
            # The strict inequality is part of the frozen experimental protocol.
            result.accepted = result.term_count == 1 or bool(result.min_novelty > self.config.threshold)
            result.elapsed_seconds = time.perf_counter() - start
            return result
        except StructuralSNError as error:
            message = f"{type(error).__name__}: {error}"
            if structural_cache_key is not None:
                self.failure_cache.put(structural_cache_key, "gradient_failure", message)
            return _fail(result, "gradient_failure", message, start)
        except (FloatingPointError, OverflowError) as error:
            return _fail(result, "numerical_overflow", f"{type(error).__name__}: {error}", start)
        except np.linalg.LinAlgError as error:
            return _fail(result, "linear_algebra_failure", f"{type(error).__name__}: {error}", start)
        except Exception as error:  # candidate-local failures must never kill a producer
            return _fail(result, "evaluation_failure", f"{type(error).__name__}: {error}", start)


def evaluate_reference(
    tree: Any,
    X: np.ndarray,
    config: SobolevFilterConfig | None = None,
    *,
    geometry_indices: np.ndarray | None = None,
) -> Any:
    """Correctness-first SymPy differentiation and LOO-``lstsq`` evaluator."""

    settings = config or SobolevFilterConfig()
    points = np.asarray(X, dtype=np.float64)
    names = tuple(f"x_{index}" for index in range(points.shape[1]))
    if geometry_indices is None:
        geometry_indices = deterministic_geometry_indices(
            len(points), tree.prefix(), settings.geometry_seed, settings.geometry_rows
        )
    reference_config = SobolevConfig(
        lambda_value=settings.value_weight,
        lambda_gradient=settings.gradient_weight,
        rcond=settings.rcond,
        min_valid_samples=settings.min_valid_rows,
        threshold=settings.threshold,
        output_scale_tolerance=settings.output_scale_tolerance,
        input_scale_tolerance=settings.input_scale_tolerance,
        zero_signature_tolerance=settings.zero_signature_tolerance,
        fast_gram=False,
        cache_enabled=False,
        geometry_sample_size=None,
    )
    expression = node_to_expression(tree)
    return SobolevEvaluator(reference_config).evaluate(
        expression,
        expression,
        points,
        names,
        dataset_identity="e2esr-reference",
        geometry_indices=np.asarray(geometry_indices, dtype=np.int64),
    )


def node_to_expression(node: Any) -> str:
    """Translate an E2ESR node to parser-safe text without simplification."""

    value = str(node.value)
    children = tuple(node.children)
    if not children:
        if value in NUMERIC_CONSTANTS:
            return repr(NUMERIC_CONSTANTS[value])
        if value == "rand":
            return "0.0"
        return value
    rendered = [node_to_expression(child) for child in children]
    if value == "add":
        return f"({rendered[0]} + {rendered[1]})"
    if value == "sub":
        return f"({rendered[0]} - {rendered[1]})"
    if value == "mul":
        return f"({rendered[0]} * {rendered[1]})"
    if value == "div":
        return f"({rendered[0]} / {rendered[1]})"
    if value == "pow":
        return f"({rendered[0]} ** {rendered[1]})"
    if value == "pow2":
        return f"({rendered[0]} ** 2)"
    if value == "pow3":
        return f"({rendered[0]} ** 3)"
    if value == "inv":
        return f"(1 / {rendered[0]})"
    if value == "id":
        return rendered[0]
    use_abs = bool(getattr(getattr(node, "params", None), "use_abs", False))
    if value == "log" and use_abs:
        return f"logabs({rendered[0]})"
    if value == "sqrt" and use_abs:
        return f"sqrtabs({rendered[0]})"
    if value in {"max", "min"}:
        return f"{value}({rendered[0]}, {rendered[1]})"
    return f"{value}({rendered[0]})"


def _fail(
    result: SobolevFilterResult,
    failure_type: str,
    message: str,
    start: float,
) -> SobolevFilterResult:
    result.success = False
    result.accepted = False
    result.failure_type = failure_type
    result.failure_message = message
    result.elapsed_seconds = time.perf_counter() - start
    return result


def _is_variable_free(node: Any, cache: dict[int, bool]) -> bool:
    key = id(node)
    if key in cache:
        return cache[key]
    value = str(node.value)
    if not node.children:
        answer = not value.startswith("x_") and value != "rand"
    else:
        answer = all(_is_variable_free(child, cache) for child in node.children)
    cache[key] = answer
    return answer


def _constant_value(node: Any, dimension: int) -> float:
    values, gradients = _evaluate_node(node, np.zeros((1, dimension), dtype=float), {})
    value = float(values[0])
    if not np.isfinite(value) or not np.all(np.isfinite(gradients)) or np.any(gradients != 0):
        raise StructuralSNError(f"invalid symbol-free coefficient {node.prefix()}")
    return value


def _atomic_term(node: Any, dimension: int, variable_cache: dict[int, bool]) -> TermPlan:
    if _is_variable_free(node, variable_cache):
        return TermPlan(_constant_value(node, dimension), ())
    return TermPlan(1.0, (TermFactor(node),))


def _decompose_expand_mul(node: Any, dimension: int, max_terms: int) -> list[TermPlan]:
    variable_cache: dict[int, bool] = {}

    def recurse(current: Any) -> list[TermPlan]:
        value = str(current.value)
        children = tuple(current.children)
        if value == "add" and len(children) == 2:
            output = recurse(children[0]) + recurse(children[1])
        elif value == "sub" and len(children) == 2:
            output = recurse(children[0]) + [
                TermPlan(-term.coefficient, term.factors) for term in recurse(children[1])
            ]
        elif value == "mul" and len(children) == 2:
            left = recurse(children[0])
            right = recurse(children[1])
            output = [
                TermPlan(a.coefficient * b.coefficient, a.factors + b.factors)
                for a in left
                for b in right
            ]
        elif value == "div" and len(children) == 2:
            numerator = recurse(children[0])
            denominator = children[1]
            if _is_variable_free(denominator, variable_cache):
                divisor = _constant_value(denominator, dimension)
                if divisor == 0:
                    raise StructuralSNError("division by a symbol-free zero denominator")
                output = [TermPlan(term.coefficient / divisor, term.factors) for term in numerator]
            else:
                output = [
                    TermPlan(term.coefficient, term.factors + (TermFactor(denominator, inverse=True),))
                    for term in numerator
                ]
        else:
            output = [_atomic_term(current, dimension, variable_cache)]
        if len(output) > max_terms:
            raise StructuralSNError(
                f"expand_mul produced {len(output)} terms, above cap {max_terms}"
            )
        return output

    terms = recurse(node)
    if not terms:
        raise StructuralSNError("expression has no additive terms")
    return terms


def _evaluate_term(
    term: TermPlan,
    X: np.ndarray,
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    rows, dimension = X.shape
    values = np.ones(rows, dtype=np.float64)
    gradients = np.zeros((rows, dimension), dtype=np.float64)
    for factor in term.factors:
        factor_values, factor_gradients = _evaluate_node(factor.node, X, cache)
        if factor.inverse:
            with np.errstate(all="ignore"):
                inverse_values = 1.0 / factor_values
                inverse_gradients = -factor_gradients / np.square(factor_values)[:, None]
            zero = factor_values == 0
            inverse_values[zero] = np.nan
            inverse_gradients[zero] = np.nan
            factor_values, factor_gradients = inverse_values, inverse_gradients
        previous = values
        with np.errstate(all="ignore"):
            gradients = gradients * factor_values[:, None] + previous[:, None] * factor_gradients
            values = previous * factor_values
    return values, gradients


def _evaluate_node(
    node: Any,
    X: np.ndarray,
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key = id(node)
    cached = cache.get(key)
    if cached is not None:
        return cached
    rows, dimension = X.shape
    value = str(node.value)
    children = tuple(node.children)
    if not children:
        gradients = np.zeros((rows, dimension), dtype=np.float64)
        if value.startswith("x_"):
            try:
                index = int(value.split("_", 1)[1])
            except ValueError as error:
                raise StructuralSNError(f"invalid variable leaf {value}") from error
            if index < 0 or index >= dimension:
                raise StructuralSNError(f"variable {value} outside input dimension {dimension}")
            values = X[:, index]
            gradients[:, index] = 1.0
        elif value == "rand":
            values = np.zeros(rows, dtype=np.float64)
        elif value in NUMERIC_CONSTANTS:
            values = np.full(rows, NUMERIC_CONSTANTS[value], dtype=np.float64)
        elif value == "CONSTANT":
            raise StructuralSNError("uninstantiated CONSTANT leaf")
        else:
            try:
                values = np.full(rows, float(value), dtype=np.float64)
            except ValueError as error:
                raise StructuralSNError(f"unsupported leaf {value}") from error
        answer = (values, gradients)
        cache[key] = answer
        return answer

    if len(children) not in {1, 2}:
        raise StructuralSNError(f"operator {value} has unsupported arity {len(children)}")
    if len(children) == 2 and value not in SUPPORTED_BINARY:
        raise StructuralSNError(f"unsupported binary operator {value}")
    if len(children) == 1 and value not in SUPPORTED_UNARY:
        raise StructuralSNError(f"unsupported unary operator {value}")

    a, grad_a = _evaluate_node(children[0], X, cache)
    with np.errstate(all="ignore"):
        if len(children) == 2:
            b, grad_b = _evaluate_node(children[1], X, cache)
            if value == "add":
                values, gradients = a + b, grad_a + grad_b
            elif value == "sub":
                values, gradients = a - b, grad_a - grad_b
            elif value == "mul":
                values = a * b
                gradients = grad_a * b[:, None] + a[:, None] * grad_b
            elif value == "div":
                values = a / b
                gradients = (grad_a * b[:, None] - a[:, None] * grad_b) / np.square(b)[:, None]
                zero = b == 0
                values[zero] = np.nan
                gradients[zero] = np.nan
            elif value == "pow":
                values = np.power(a, b)
                if np.all(grad_b == 0):
                    gradients = b[:, None] * np.power(a, b - 1)[:, None] * grad_a
                else:
                    gradients = values[:, None] * (
                        grad_b * np.log(a)[:, None] + b[:, None] * grad_a / a[:, None]
                    )
            elif value in {"max", "min"}:
                choose_a = a >= b if value == "max" else a <= b
                tie = a == b
                values = np.maximum(a, b) if value == "max" else np.minimum(a, b)
                gradients = np.where(choose_a[:, None], grad_a, grad_b)
                gradients[tie] = np.nan
            else:  # guarded by SUPPORTED_BINARY
                raise StructuralSNError(f"unsupported binary operator {value}")
        else:
            use_abs = bool(getattr(getattr(node, "params", None), "use_abs", False))
            if value == "id":
                values, gradients = a, grad_a
            elif value == "abs":
                values = np.abs(a)
                gradients = np.sign(a)[:, None] * grad_a
                gradients[a == 0] = np.nan
            elif value == "inv":
                values = 1.0 / a
                gradients = -grad_a / np.square(a)[:, None]
                zero = a == 0
                values[zero] = np.nan
                gradients[zero] = np.nan
            elif value == "sqrt":
                argument = np.abs(a) if use_abs else a
                values = np.sqrt(argument)
                multiplier = np.sign(a) if use_abs else np.ones_like(a)
                gradients = multiplier[:, None] * grad_a / (2 * values[:, None])
            elif value == "log":
                argument = np.abs(a) if use_abs else a
                values = np.log(argument)
                gradients = grad_a / a[:, None]
            elif value == "exp":
                values = np.exp(a)
                gradients = values[:, None] * grad_a
            elif value == "sin":
                values = np.sin(a)
                gradients = np.cos(a)[:, None] * grad_a
            elif value == "cos":
                values = np.cos(a)
                gradients = -np.sin(a)[:, None] * grad_a
            elif value == "tan":
                values = np.tan(a)
                gradients = grad_a / np.square(np.cos(a))[:, None]
            elif value == "arcsin":
                values = np.arcsin(a)
                gradients = grad_a / np.sqrt(1 - np.square(a))[:, None]
            elif value == "arccos":
                values = np.arccos(a)
                gradients = -grad_a / np.sqrt(1 - np.square(a))[:, None]
            elif value == "arctan":
                values = np.arctan(a)
                gradients = grad_a / (1 + np.square(a))[:, None]
            elif value == "pow2":
                values = np.square(a)
                gradients = 2 * a[:, None] * grad_a
            elif value == "pow3":
                values = np.power(a, 3)
                gradients = 3 * np.square(a)[:, None] * grad_a
            else:  # guarded by SUPPORTED_UNARY
                raise StructuralSNError(f"unsupported unary operator {value}")
    values = np.asarray(values, dtype=np.float64).reshape(rows)
    gradients = np.asarray(gradients, dtype=np.float64).reshape(rows, dimension)
    answer = (values, gradients)
    cache[key] = answer
    return answer


__all__ = [
    "FastSobolevNovelty",
    "SobolevFilterConfig",
    "SobolevFilterResult",
    "StructuralFailureCache",
    "deterministic_geometry_indices",
    "evaluate_reference",
    "node_to_expression",
]
