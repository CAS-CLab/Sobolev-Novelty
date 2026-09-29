"""Additive linear phenotype fitting for controlled population GP."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import sympy as sp

from ...nd2py import nd2py as nd
from ...sobolev.decomposition import (
    ProtectedDivision,
    decompose_expand_mul,
    parse_expression,
    to_project_expression_string,
)
from ...sobolev.signature import evaluate_sympy
from ...sobolev.types import TermSpec


class AdditiveTermLimitError(ValueError):
    """A mathematically valid GP tree exceeds the frozen expansion budget."""


@dataclass
class AdditiveFitResult:
    raw_expression: str
    success: bool = False
    fitted_tree: nd.Symbol | None = None
    fitted_expression: sp.Expr | None = None
    symbols: tuple[sp.Symbol, ...] = ()
    terms: tuple[TermSpec, ...] = ()
    coefficients: tuple[float, ...] = ()
    mse: float = float("inf")
    r2: float = float("-inf")
    design_mse: float = float("inf")
    design_r2: float = float("-inf")
    score_semantics: str = "design_matrix"
    nonfinite_export_predictions: int = 0
    complexity: int = 0
    base_reward: float = 0.0
    source_term_count: int = 0
    nonfinite_design_values: int = 0
    rank: int = 0
    singular_values: tuple[float, ...] = ()
    failure_type: str = ""
    failure_message: str = ""
    elapsed_seconds: float = 0.0

    @property
    def fitted_expression_text(self) -> str:
        return "" if self.fitted_tree is None else self.fitted_tree.to_str(number_format=".17g")

    @property
    def canonical_fitted_expression(self) -> str:
        return "" if self.fitted_expression is None else sp.srepr(self.fitted_expression)

    def clone(self) -> "AdditiveFitResult":
        value = AdditiveFitResult(**{**self.__dict__})
        value.fitted_tree = None if self.fitted_tree is None else self.fitted_tree.copy()
        return value


class AdditiveLinearEvaluator:
    """Fit one coefficient per exact ``expand_mul`` basis plus an intercept."""

    def __init__(
        self,
        feature_names: Sequence[str],
        dataset_identity: str,
        *,
        eta: float = 0.999,
        coefficient_decimals: int = 6,
        max_genotype_len: int = 30,
        max_additive_terms: int = 128,
        cache_max_entries: int = 50_000,
        protected_epsilon: float = 1e-6,
        score_semantics: str = "design_matrix",
    ) -> None:
        self.feature_names = tuple(str(name) for name in feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        if not 0 < eta <= 1:
            raise ValueError("eta must lie in (0, 1]")
        if coefficient_decimals < 0 or max_genotype_len < 1:
            raise ValueError("invalid coefficient/max-length configuration")
        if max_additive_terms < 1 or cache_max_entries < 1:
            raise ValueError("term and cache limits must be positive")
        if score_semantics not in {"design_matrix", "export_aligned"}:
            raise ValueError(
                "score_semantics must be design_matrix or export_aligned"
            )
        self.dataset_identity = str(dataset_identity)
        self.eta = float(eta)
        self.coefficient_decimals = int(coefficient_decimals)
        self.max_genotype_len = int(max_genotype_len)
        self.max_additive_terms = int(max_additive_terms)
        self.cache_max_entries = int(cache_max_entries)
        self.protected_epsilon = float(protected_epsilon)
        self.score_semantics = str(score_semantics)
        self._cache: OrderedDict[str, AdditiveFitResult] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.failure_counts: dict[str, int] = {}

    def evaluate(
        self,
        genotype: nd.Symbol,
        X: np.ndarray,
        y: np.ndarray,
    ) -> AdditiveFitResult:
        started = time.perf_counter()
        raw_expression = genotype.to_str(number_format=".17g")
        cache_key = self._cache_key(raw_expression)
        cached = self._cache.pop(cache_key, None)
        if cached is not None:
            self.hits += 1
            self._cache[cache_key] = cached
            value = cached.clone()
            value.elapsed_seconds = time.perf_counter() - started
            return value
        self.misses += 1
        if len(genotype) > self.max_genotype_len:
            return self._remember_failure(
                cache_key,
                AdditiveFitResult(
                    raw_expression=raw_expression,
                    failure_type="genotype_too_long",
                    failure_message=(
                        f"genotype length {len(genotype)} exceeds {self.max_genotype_len}"
                    ),
                ),
                started,
            )
        try:
            expression, symbols = parse_expression(raw_expression, self.feature_names)
            estimated = expanded_term_count(expression, self.max_additive_terms)
            if estimated > self.max_additive_terms:
                raise AdditiveTermLimitError(
                    f"expand_mul term count exceeds {self.max_additive_terms}"
                )
            decomposed = tuple(decompose_expand_mul(expression, symbols))
            if len(decomposed) > self.max_additive_terms:
                raise AdditiveTermLimitError(
                    f"expand_mul produced {len(decomposed)} terms; "
                    f"limit={self.max_additive_terms}"
                )
            # Symbol-free pieces are represented by the single fitted intercept.
            bases = tuple(term.basis for term in decomposed if term.basis.free_symbols)
            result = self._fit_bases(
                raw_expression,
                symbols,
                bases,
                np.asarray(X, dtype=float),
                np.asarray(y, dtype=float),
                include_intercept=True,
            )
            result.source_term_count = len(decomposed)
        except Exception as error:
            result = AdditiveFitResult(
                raw_expression=raw_expression,
                failure_type=(
                    "additive_term_limit"
                    if isinstance(error, AdditiveTermLimitError)
                    else "additive_fit_failure"
                ),
                failure_message=f"{type(error).__name__}: {error}",
            )
        if result.success:
            return self._remember(cache_key, result, started)
        return self._remember_failure(cache_key, result, started)

    def refit_terms(
        self,
        terms: Sequence[TermSpec],
        X: np.ndarray,
        y: np.ndarray,
        *,
        raw_expression: str,
    ) -> AdditiveFitResult:
        """Refit retained fitted terms without silently adding a new intercept."""

        started = time.perf_counter()
        try:
            symbols = tuple(sp.Symbol(name, real=True) for name in self.feature_names)
            bases = tuple(term.basis for term in terms)
            if not bases:
                raise ValueError("pruning retained no additive terms")
            result = self._fit_bases(
                raw_expression,
                symbols,
                bases,
                np.asarray(X, dtype=float),
                np.asarray(y, dtype=float),
                include_intercept=False,
            )
            result.source_term_count = len(bases)
        except Exception as error:
            result = AdditiveFitResult(
                raw_expression=raw_expression,
                failure_type="pruning_refit_failure",
                failure_message=f"{type(error).__name__}: {error}",
            )
        result.elapsed_seconds = time.perf_counter() - started
        return result

    def _fit_bases(
        self,
        raw_expression: str,
        symbols: tuple[sp.Symbol, ...],
        bases: tuple[sp.Expr, ...],
        X: np.ndarray,
        y: np.ndarray,
        *,
        include_intercept: bool,
    ) -> AdditiveFitResult:
        if X.ndim != 2 or X.shape[1] != len(symbols) or X.shape[0] == 0:
            raise ValueError(f"X shape {X.shape} does not match symbols {symbols}")
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != len(X) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X/y must be aligned and finite")
        y_variance = float(np.var(y))
        if not np.isfinite(y_variance) or y_variance <= 1e-15:
            raise ValueError("target variance is zero or non-finite")
        columns: list[np.ndarray] = []
        basis_specs: list[sp.Expr] = []
        if include_intercept:
            columns.append(np.ones(len(y), dtype=float))
            basis_specs.append(sp.Integer(1))
        for basis in bases:
            columns.append(
                evaluate_sympy(
                    basis,
                    symbols,
                    X,
                    protected_epsilon=self.protected_epsilon,
                )
            )
            basis_specs.append(basis)
        if not columns:
            raise ValueError("design matrix has no retained columns")
        design = np.column_stack(columns)
        nonfinite = int(np.size(design) - np.count_nonzero(np.isfinite(design)))
        design[~np.isfinite(design)] = 0.0
        coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
        coefficients = np.round(coefficients, self.coefficient_decimals)
        design_prediction = design @ coefficients
        design_mse = float(np.mean(np.square(design_prediction - y)))
        design_r2 = float(1.0 - design_mse / y_variance)
        fitted_terms: list[TermSpec] = []
        components: list[sp.Expr] = []
        for coefficient, basis in zip(coefficients, basis_specs, strict=True):
            coefficient = float(coefficient)
            if coefficient == 0.0:
                continue
            fitted_terms.append(TermSpec(coefficient, basis, sp.srepr(basis)))
            components.append(
                sp.Mul(sp.Float(str(coefficient)), basis, evaluate=False)
            )
        if not components:
            fitted_expression: sp.Expr = sp.Float("0.0")
            fitted_tree = nd.Number(0.0)
        else:
            fitted_expression = sp.Add(*components, evaluate=False)
            fitted_tree = nd.parse(to_project_expression_string(fitted_expression))
        nonfinite_export_predictions = 0
        if self.score_semantics == "export_aligned":
            values = {
                name: X[:, index]
                for index, name in enumerate(self.feature_names)
            }
            with np.errstate(all="ignore"):
                prediction = np.asarray(
                    fitted_tree.eval(values, use_eps=self.protected_epsilon),
                    dtype=float,
                )
            if prediction.ndim == 0:
                prediction = np.full(len(y), float(prediction))
            prediction = prediction.reshape(-1)
            if len(prediction) != len(y):
                raise ValueError(
                    "exported fitted expression returned the wrong row count"
                )
            nonfinite_export_predictions = int(
                np.count_nonzero(~np.isfinite(prediction))
            )
            prediction[~np.isfinite(prediction)] = 0.0
            mse = float(np.mean(np.square(prediction - y)))
            r2 = float(1.0 - mse / y_variance)
        else:
            mse = design_mse
            r2 = design_r2
        complexity = int(len(fitted_tree))
        denominator = 2.0 - r2
        base_reward = (
            float(self.eta**complexity / denominator)
            if np.isfinite(denominator) and denominator > 0
            else 0.0
        )
        if not np.isfinite(base_reward):
            base_reward = 0.0
        return AdditiveFitResult(
            raw_expression=raw_expression,
            success=True,
            fitted_tree=fitted_tree,
            fitted_expression=fitted_expression,
            symbols=symbols,
            terms=tuple(fitted_terms),
            coefficients=tuple(float(value) for value in coefficients),
            mse=mse,
            r2=r2,
            design_mse=design_mse,
            design_r2=design_r2,
            score_semantics=self.score_semantics,
            nonfinite_export_predictions=nonfinite_export_predictions,
            complexity=complexity,
            base_reward=base_reward,
            nonfinite_design_values=nonfinite,
            rank=int(rank),
            singular_values=tuple(float(value) for value in singular),
        )

    def _cache_key(self, raw_expression: str) -> str:
        payload = (
            f"{self.dataset_identity}|{self.feature_names}|{self.eta:.17g}|"
            f"{self.coefficient_decimals}|{self.max_additive_terms}|{raw_expression}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _remember(
        self,
        key: str,
        result: AdditiveFitResult,
        started: float,
    ) -> AdditiveFitResult:
        result.elapsed_seconds = time.perf_counter() - started
        self._cache[key] = result.clone()
        while len(self._cache) > self.cache_max_entries:
            self._cache.popitem(last=False)
            self.evictions += 1
        return result

    def _remember_failure(
        self,
        key: str,
        result: AdditiveFitResult,
        started: float,
    ) -> AdditiveFitResult:
        name = result.failure_type or "unknown"
        self.failure_counts[name] = self.failure_counts.get(name, 0) + 1
        return self._remember(key, result, started)

    @property
    def entry_count(self) -> int:
        return len(self._cache)


def expanded_term_count(expression: sp.Expr, limit: int) -> int:
    """Count ``decompose_expand_mul`` outputs, stopping above ``limit``."""

    if isinstance(expression, sp.Add):
        total = 0
        for argument in expression.args:
            total += expanded_term_count(argument, limit)
            if total > limit:
                return limit + 1
        return total
    if isinstance(expression, ProtectedDivision):
        return expanded_term_count(expression.args[0], limit)
    if isinstance(expression, sp.Mul):
        total = 1
        for argument in expression.args:
            total *= expanded_term_count(argument, limit)
            if total > limit:
                return limit + 1
        return total
    return 1
