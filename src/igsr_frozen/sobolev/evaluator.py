"""Unified, MCTS-independent Sobolev novelty evaluator."""

from __future__ import annotations

import hashlib
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sympy as sp

from .cache import CacheKey, RawTermEvaluation, TermEvaluationCache
from .config import SobolevConfig
from .decomposition import decompose_expand_mul, expression_to_string, parse_expression
from .geometry_cache import CandidateGeometryCache, DecompositionRecord
from .geometry_state import CandidateGeometryKey, GeometryState
from .novelty import compute_novelties
from .signature import (
    SignatureError,
    array_identity,
    construct_signatures,
    evaluate_sympy,
    select_geometry_indices,
    stable_norm,
    stable_rms,
)
from .types import DetailedRuntime, EvaluationResult, FailureType, TermSpec


# SymPy can surface malformed or pathologically large candidate expressions
# through implementation-level exceptions in addition to its public
# ``SympifyError``/``ValueError`` paths.  These are candidate-local failures:
# MCTS must record and penalize the candidate rather than abort the search.
# Keep the list explicit so genuine evaluator programming errors still escape.
_SYMBOLIC_CANDIDATE_EXCEPTIONS = (
    SyntaxError,
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
    MemoryError,
    OverflowError,
    RecursionError,
    ZeroDivisionError,
)


class SobolevEvaluator:
    """Evaluate fitted candidates with exact term derivatives and shared geometry."""

    def __init__(
        self,
        config: SobolevConfig | None = None,
        cache: TermEvaluationCache | None = None,
        geometry_cache: CandidateGeometryCache | None = None,
    ) -> None:
        self.config = config or SobolevConfig()
        self.cache = cache or TermEvaluationCache(enabled=self.config.cache_enabled)
        self.geometry_cache = geometry_cache or CandidateGeometryCache(
            enabled=self.config.candidate_geometry_cache,
            max_entries=self.config.geometry_cache_max_entries,
            max_memory_bytes=self.config.geometry_cache_max_memory_bytes,
            decomposition_max_entries=self.config.decomposition_cache_max_entries,
        )

    def evaluate(
        self,
        raw_expression: object,
        fitted_expression: object,
        X: np.ndarray | pd.DataFrame | Mapping[str, np.ndarray],
        feature_names: Sequence[str] | None = None,
        dataset_identity: str = "dataset",
        geometry_indices: np.ndarray | None = None,
        parent_geometry_key: str | None = None,
    ) -> EvaluationResult:
        """Parse, decompose, evaluate, and score a complete fitted candidate."""

        total_start = time.perf_counter()
        runtime = DetailedRuntime()
        raw_text = expression_to_string(raw_expression)
        fitted_text = expression_to_string(fitted_expression)
        base = EvaluationResult(
            raw_expression=raw_text,
            fitted_expression=fitted_text,
            threshold=self.config.threshold,
            detailed_runtime=runtime,
        )
        try:
            points, names = _coerce_inputs(X, feature_names)
        except (TypeError, ValueError, KeyError) as error:
            return self._failure(base, FailureType.INVALID_INPUT, f"{type(error).__name__}: {error}", total_start)
        dataset_identity = _content_bound_dataset_identity(
            dataset_identity, points, names,
        )
        if not np.all(np.isfinite(points)):
            return self._failure(base, FailureType.INVALID_INPUT, "Training inputs contain non-finite values", total_start)
        input_means = np.mean(points, axis=0)
        input_scales = np.std(points, axis=0, ddof=0)
        invalid_scale = np.flatnonzero(input_scales <= self.config.input_scale_tolerance)
        if invalid_scale.size:
            names_bad = [names[int(index)] for index in invalid_scale]
            base.input_means = input_means.tolist()
            base.input_scales = input_scales.tolist()
            return self._failure(
                base,
                FailureType.CONSTANT_INPUT_SCALE,
                f"Near-constant training input scale for {names_bad}",
                total_start,
            )
        operator_configuration = self._operator_configuration(names)
        decomposition_record = None
        if self.config.candidate_geometry_cache:
            cache_start = time.perf_counter()
            decomposition_record = self.geometry_cache.get_decomposition(
                fitted_text, names, operator_configuration,
            )
            runtime.cache_lookup += time.perf_counter() - cache_start
        if decomposition_record is not None:
            if not decomposition_record.success:
                base.failure_cache_hit = True
                return self._failure(
                    base,
                    decomposition_record.failure_type,
                    decomposition_record.failure_message,
                    total_start,
                )
            expression = decomposition_record.expression
            symbols = decomposition_record.symbols
            terms = list(decomposition_record.terms)
            assert expression is not None
        else:
            try:
                parse_start = time.perf_counter()
                expression, symbols = parse_expression(fitted_expression, names)
                runtime.parsing = time.perf_counter() - parse_start
            except _SYMBOLIC_CANDIDATE_EXCEPTIONS as error:
                message = f"{type(error).__name__}: {error}"
                if self.config.candidate_geometry_cache:
                    self.geometry_cache.put_decomposition(
                        fitted_text,
                        names,
                        operator_configuration,
                        DecompositionRecord(
                            success=False,
                            failure_type=FailureType.PARSE_FAILURE,
                            failure_message=message,
                        ),
                    )
                return self._failure(base, FailureType.PARSE_FAILURE, message, total_start)
            try:
                decomposition_start = time.perf_counter()
                terms = decompose_expand_mul(expression, symbols)
                runtime.decomposition = time.perf_counter() - decomposition_start
            except _SYMBOLIC_CANDIDATE_EXCEPTIONS as error:
                message = f"{type(error).__name__}: {error}"
                if self.config.candidate_geometry_cache:
                    self.geometry_cache.put_decomposition(
                        fitted_text,
                        names,
                        operator_configuration,
                        DecompositionRecord(
                            success=False,
                            failure_type=FailureType.DECOMPOSITION_FAILURE,
                            failure_message=message,
                        ),
                    )
                return self._failure(
                    base,
                    FailureType.DECOMPOSITION_FAILURE,
                    message,
                    total_start,
                )
            if self.config.candidate_geometry_cache:
                self.geometry_cache.put_decomposition(
                    fitted_text,
                    names,
                    operator_configuration,
                    DecompositionRecord(
                        expression=expression,
                        symbols=tuple(symbols),
                        terms=tuple(terms),
                    ),
                )
        return self._evaluate_parsed_terms(
            base=base,
            expression=expression,
            symbols=symbols,
            terms=terms,
            points=points,
            input_means=input_means,
            input_scales=input_scales,
            dataset_identity=dataset_identity,
            geometry_indices=geometry_indices,
            total_start=total_start,
            parent_geometry_key=parent_geometry_key,
        )

    def evaluate_terms(
        self,
        terms: Sequence[tuple[float, sp.Expr]],
        symbols: Sequence[sp.Symbol],
        X: np.ndarray,
        dataset_identity: str = "explicit_terms",
        geometry_indices: np.ndarray | None = None,
        raw_expression: str = "explicit_terms",
        parent_geometry_key: str | None = None,
    ) -> EvaluationResult:
        """Evaluate explicitly supplied terms; useful for audits and unit tests."""

        total_start = time.perf_counter()
        runtime = DetailedRuntime()
        term_specs = [TermSpec(float(coefficient), basis, sp.srepr(basis)) for coefficient, basis in terms]
        expression = sp.Add(
            *(sp.Mul(sp.Float(term.coefficient), term.basis) for term in term_specs),
            evaluate=False,
        )
        base = EvaluationResult(
            raw_expression=raw_expression,
            fitted_expression=str(expression),
            threshold=self.config.threshold,
            detailed_runtime=runtime,
        )
        points = np.asarray(X, dtype=float)
        if points.ndim != 2 or points.shape[1] != len(symbols):
            return self._failure(base, FailureType.INVALID_INPUT, "Explicit X/symbol shape mismatch", total_start)
        if not np.all(np.isfinite(points)):
            return self._failure(base, FailureType.INVALID_INPUT, "Training inputs contain non-finite values", total_start)
        dataset_identity = _content_bound_dataset_identity(
            dataset_identity, points, tuple(str(symbol) for symbol in symbols),
        )
        input_means = np.mean(points, axis=0)
        input_scales = np.std(points, axis=0, ddof=0)
        invalid_scale = np.flatnonzero(input_scales <= self.config.input_scale_tolerance)
        if invalid_scale.size:
            base.input_means = input_means.tolist()
            base.input_scales = input_scales.tolist()
            return self._failure(base, FailureType.CONSTANT_INPUT_SCALE, "Near-constant input", total_start)
        return self._evaluate_parsed_terms(
            base,
            expression,
            tuple(symbols),
            term_specs,
            points,
            input_means,
            input_scales,
            dataset_identity,
            geometry_indices,
            total_start,
            parent_geometry_key,
        )

    def evaluate_preparsed(
        self,
        raw_expression: object,
        fitted_expression: object,
        expression: sp.Expr,
        symbols: Sequence[sp.Symbol],
        terms: Sequence[TermSpec],
        X: np.ndarray | pd.DataFrame | Mapping[str, np.ndarray],
        dataset_identity: str = "preparsed",
        geometry_indices: np.ndarray | None = None,
        parent_geometry_key: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a caller-owned immutable AST/decomposition without parsing.

        MCTS can use this entry point when a child action already carries the
        fitted SymPy AST and additive term IDs. The exact same downstream
        shared-mask/signature/novelty implementation is used.
        """

        total_start = time.perf_counter()
        runtime = DetailedRuntime()
        base = EvaluationResult(
            raw_expression=expression_to_string(raw_expression),
            fitted_expression=expression_to_string(fitted_expression),
            threshold=self.config.threshold,
            detailed_runtime=runtime,
        )
        names = tuple(str(symbol) for symbol in symbols)
        try:
            points, input_names = _coerce_inputs(X, names)
        except (TypeError, ValueError, KeyError) as error:
            return self._failure(
                base,
                FailureType.INVALID_INPUT,
                f"{type(error).__name__}: {error}",
                total_start,
            )
        if input_names != names or not np.all(np.isfinite(points)):
            return self._failure(
                base, FailureType.INVALID_INPUT, "Preparsed symbols/input mismatch", total_start,
            )
        dataset_identity = _content_bound_dataset_identity(
            dataset_identity, points, names,
        )
        input_means = np.mean(points, axis=0)
        input_scales = np.std(points, axis=0, ddof=0)
        if np.any(input_scales <= self.config.input_scale_tolerance):
            base.input_means = input_means.tolist()
            base.input_scales = input_scales.tolist()
            return self._failure(
                base, FailureType.CONSTANT_INPUT_SCALE, "Near-constant input", total_start,
            )
        return self._evaluate_parsed_terms(
            base,
            expression,
            tuple(symbols),
            tuple(terms),
            points,
            input_means,
            input_scales,
            dataset_identity,
            geometry_indices,
            total_start,
            parent_geometry_key,
        )

    def _evaluate_parsed_terms(
        self,
        base: EvaluationResult,
        expression: sp.Expr,
        symbols: Sequence[sp.Symbol],
        terms: Sequence[TermSpec],
        points: np.ndarray,
        input_means: np.ndarray,
        input_scales: np.ndarray,
        dataset_identity: str,
        geometry_indices: np.ndarray | None,
        total_start: float,
        parent_geometry_key: str | None = None,
    ) -> EvaluationResult:
        if self.config.output_scale_free_internal:
            return self._evaluate_parsed_terms_scale_free(
                base,
                expression,
                symbols,
                terms,
                points,
                input_means,
                input_scales,
                dataset_identity,
                geometry_indices,
                total_start,
                parent_geometry_key,
            )
        runtime = base.detailed_runtime
        base.terms = [term.display for term in terms]
        base.coefficients = [term.coefficient for term in terms]
        base.input_means = input_means.tolist()
        base.input_scales = input_scales.tolist()
        if geometry_indices is None:
            geometry_indices = select_geometry_indices(
                len(points),
                dataset_identity,
                self.config.geometry_seed,
                self.config.geometry_sample_size,
            )
        geometry_indices = np.asarray(geometry_indices, dtype=np.int64)
        if (
            geometry_indices.ndim != 1
            or geometry_indices.size == 0
            or np.any(geometry_indices < 0)
            or np.any(geometry_indices >= len(points))
            or np.unique(geometry_indices).size != geometry_indices.size
        ):
            return self._failure(base, FailureType.INVALID_INPUT, "Invalid geometry indices", total_start)
        geometry_points = points[geometry_indices]
        base.geometry_indices = geometry_indices.tolist()
        geometry_identity = array_identity(geometry_indices)
        normalization_identity = array_identity(input_means, input_scales)
        hits_before, misses_before = self.cache.hits, self.cache.misses
        raw_values: list[np.ndarray] = []
        raw_gradients: list[np.ndarray] = []
        for term in terms:
            cache_start = time.perf_counter()
            key = CacheKey(
                canonical_term=term.canonical,
                dataset_identity=dataset_identity,
                geometry_subset_identity=geometry_identity,
                input_normalization_identity=normalization_identity,
                derivative_order=1,
                operator_configuration=f"protected_eps={self.config.protected_epsilon:.17g}",
            )
            cached = self.cache.get_raw(key)
            runtime.cache_lookup += time.perf_counter() - cache_start
            cache_miss = cached is None
            if cached is None:
                try:
                    cached = self._evaluate_raw_term(term, symbols, geometry_points)
                except Exception as error:
                    failure = (
                        FailureType.DIFFERENTIATION_FAILURE
                        if "differentiat" in str(error).lower()
                        else FailureType.EVALUATION_FAILURE
                    )
                    base.cache_hits = self.cache.hits - hits_before
                    base.cache_misses = self.cache.misses - misses_before
                    return self._failure(
                        base,
                        failure,
                        f"term={term.display}; {type(error).__name__}: {error}",
                        total_start,
                    )
                self.cache.put_raw(key, cached)
            if cache_miss:
                runtime.symbolic_differentiation += cached.symbolic_differentiation_time
                runtime.term_value_evaluation += cached.value_evaluation_time
                runtime.gradient_evaluation += cached.gradient_evaluation_time
            raw_values.append(cached.values)
            raw_gradients.append(cached.gradients)
        base.cache_hits = self.cache.hits - hits_before
        base.cache_misses = self.cache.misses - misses_before
        base.cache_memory_bytes = self.cache.memory_bytes
        values = np.column_stack(raw_values)
        gradients = np.stack(raw_gradients, axis=2)
        try:
            full_start = time.perf_counter()
            full_values = evaluate_sympy(
                expression,
                symbols,
                geometry_points,
                self.config.protected_epsilon,
            )
            runtime.term_value_evaluation += time.perf_counter() - full_start
            bundle = construct_signatures(
                values=values,
                gradients=gradients,
                full_values=full_values,
                geometry_points=geometry_points,
                input_means=input_means,
                input_scales=input_scales,
                lambda_value=self.config.lambda_value,
                lambda_gradient=self.config.lambda_gradient,
                min_valid_samples=self.config.min_valid_samples,
                output_scale_tolerance=self.config.output_scale_tolerance,
                zero_signature_tolerance=self.config.zero_signature_tolerance,
            )
            runtime.shared_mask += bundle.shared_mask_seconds
            runtime.signature_construction += bundle.signature_construction_seconds
        except SignatureError as error:
            base.valid_mask_size = error.valid_mask_size
            base.term_norms = getattr(error, "term_norms", np.asarray([])).tolist()
            base.signature_shape = tuple(getattr(error, "signature_shape", (0, len(terms))))
            base.output_scale = getattr(error, "output_scale", None)
            return self._failure(base, FailureType(error.failure_type), str(error), total_start)
        except Exception as error:
            return self._failure(
                base,
                FailureType.EVALUATION_FAILURE,
                f"{type(error).__name__}: {error}",
                total_start,
            )
        base.valid_mask_size = int(bundle.valid_mask.sum())
        base.valid_mask_identity = array_identity(bundle.valid_mask.astype(np.uint8))
        base.output_scale = bundle.output_scale
        base.signature_shape = tuple(bundle.signatures.shape)
        base.term_norms = bundle.term_norms.tolist()
        try:
            linear_start = time.perf_counter()
            novelty = compute_novelties(
                signatures=bundle.signatures,
                term_norms=bundle.term_norms,
                rcond=self.config.rcond,
                fast_gram=self.config.fast_gram,
                condition_threshold=self.config.gram_condition_threshold,
            )
            runtime.novelty_linear_algebra = time.perf_counter() - linear_start
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
            return self._failure(
                base,
                FailureType.LINEAR_ALGEBRA_FAILURE,
                f"{type(error).__name__}: {error}",
                total_start,
            )
        values_novelty = novelty.novelties
        base.term_novelties = values_novelty.tolist()
        base.min_novelty = float(np.min(values_novelty))
        base.mean_novelty = float(np.mean(values_novelty))
        base.low_novelty_count = int(np.sum(values_novelty < self.config.threshold))
        base.low_novelty_ratio = base.low_novelty_count / len(values_novelty)
        base.singular_values = novelty.singular_values.tolist()
        base.rank = novelty.rank
        base.condition_number = novelty.condition_number
        base.algorithm_used = novelty.algorithm_used
        base.fallback_reason = novelty.fallback_reason
        base.term_diagnostics = []
        for index, (term, norm, score, diagnostic) in enumerate(
            zip(terms, bundle.term_norms, values_novelty, novelty.term_diagnostics, strict=True)
        ):
            base.term_diagnostics.append(
                {
                    "term_index": index + 1,
                    "coefficient": term.coefficient,
                    "basis_term": term.display,
                    "signature_norm": float(norm),
                    "novelty": float(score),
                    "deletion_impact": abs(term.coefficient) * float(score) * float(norm),
                    **diagnostic,
                }
            )
        base.success = True
        base.failure_type = FailureType.NONE
        runtime.total_candidate_evaluation = time.perf_counter() - total_start
        return base

    def _evaluate_parsed_terms_scale_free(
        self,
        base: EvaluationResult,
        expression: sp.Expr,
        symbols: Sequence[sp.Symbol],
        terms: Sequence[TermSpec],
        points: np.ndarray,
        input_means: np.ndarray,
        input_scales: np.ndarray,
        dataset_identity: str,
        geometry_indices: np.ndarray | None,
        total_start: float,
        parent_geometry_key: str | None,
    ) -> EvaluationResult:
        """Evaluate coefficient-free geometry and restore public scale semantics."""

        runtime = base.detailed_runtime
        base.terms = [term.display for term in terms]
        base.coefficients = [term.coefficient for term in terms]
        base.input_means = input_means.tolist()
        base.input_scales = input_scales.tolist()
        base.output_scale_free_internal = True
        base.parent_geometry_key = parent_geometry_key
        if geometry_indices is None:
            geometry_indices = select_geometry_indices(
                len(points),
                dataset_identity,
                self.config.geometry_seed,
                self.config.geometry_sample_size,
            )
        geometry_indices = np.asarray(geometry_indices, dtype=np.int64)
        if (
            geometry_indices.ndim != 1
            or geometry_indices.size == 0
            or np.any(geometry_indices < 0)
            or np.any(geometry_indices >= len(points))
            or np.unique(geometry_indices).size != geometry_indices.size
        ):
            return self._failure(base, FailureType.INVALID_INPUT, "Invalid geometry indices", total_start)
        geometry_points = points[geometry_indices]
        base.geometry_indices = geometry_indices.tolist()
        geometry_identity = array_identity(geometry_indices)
        normalization_identity = array_identity(input_means, input_scales)
        canonical_order = sorted(
            range(len(terms)), key=lambda index: (terms[index].canonical, index)
        )
        canonical_terms = [terms[index] for index in canonical_order]
        operator_configuration = self._operator_configuration(tuple(str(symbol) for symbol in symbols))
        key = CandidateGeometryKey(
            canonical_terms=tuple(term.canonical for term in canonical_terms),
            dataset_identity=dataset_identity,
            geometry_subset_identity=geometry_identity,
            input_normalization_identity=normalization_identity,
            derivative_order=1,
            lambda_value=self.config.lambda_value,
            lambda_gradient=self.config.lambda_gradient,
            operator_configuration=operator_configuration,
        )
        base.candidate_geometry_key = key.digest
        state = None
        if self.config.candidate_geometry_cache:
            lookup_start = time.perf_counter()
            state = self.geometry_cache.get(key)
            runtime.cache_lookup += time.perf_counter() - lookup_start
        if state is not None:
            base.geometry_cache_hit = True
            base.geometry_reuse_mode = "candidate_geometry_cache"
            if not state.success:
                base.failure_cache_hit = True
        else:
            base.geometry_reuse_mode = "scale_free_full"
            raw = None
            if self.config.parent_child_incremental and parent_geometry_key:
                raw = self._collect_incremental_terms(
                    parent_geometry_key,
                    key,
                    canonical_terms,
                    symbols,
                    geometry_points,
                    dataset_identity,
                    geometry_identity,
                    normalization_identity,
                    runtime,
                    base,
                    total_start,
                )
            if raw is None:
                raw = self._collect_raw_terms(
                    canonical_terms,
                    symbols,
                    geometry_points,
                    dataset_identity,
                    geometry_identity,
                    normalization_identity,
                    runtime,
                    base,
                    total_start,
                )
            if isinstance(raw, EvaluationResult):
                return raw
            values, gradients = raw
            state = self._build_geometry_state(
                key,
                values,
                gradients,
                geometry_points,
                input_scales,
                runtime,
            )
            if self.config.candidate_geometry_cache:
                self.geometry_cache.put(state)
        coefficients = np.asarray(
            [terms[index].coefficient for index in canonical_order], dtype=float,
        )
        with np.errstate(all="ignore"):
            # The original fitted expression remains the source of truth for
            # its finite mask and RMS. Algebraically reconstructed term sums
            # can erase protected-domain failures such as 0*asin(2).
            full_start = time.perf_counter()
            full_values = evaluate_sympy(
                expression,
                symbols,
                geometry_points,
                self.config.protected_epsilon,
            )
            runtime.term_value_evaluation += time.perf_counter() - full_start
        candidate_mask = state.shared_valid_mask & np.isfinite(full_values)
        if not np.array_equal(candidate_mask, state.shared_valid_mask):
            base.geometry_fallback_reason = "coefficient_dependent_finite_mask"
            base.geometry_reuse_mode = "coefficient_mask_rebuild"
            state = self._build_geometry_state(
                key,
                state.values,
                state.gradients,
                geometry_points,
                input_scales,
                runtime,
                additional_mask=np.isfinite(full_values),
            )
            candidate_mask = state.shared_valid_mask
        base.valid_mask_size = int(candidate_mask.sum())
        base.valid_mask_identity = array_identity(candidate_mask.astype(np.uint8))
        if (
            not state.success
            and state.failure_type == FailureType.INSUFFICIENT_VALID_SAMPLES
        ):
            return self._failure(
                base, state.failure_type, state.failure_message, total_start,
            )
        output_scale = stable_rms(full_values[candidate_mask])
        base.output_scale = output_scale
        if not np.isfinite(output_scale) or output_scale < self.config.output_scale_tolerance:
            return self._failure(
                base,
                FailureType.ZERO_OUTPUT_SCALE,
                f"Candidate RMS output scale {output_scale!r} is below {self.config.output_scale_tolerance}",
                total_start,
            )
        if not state.success:
            return self._failure(
                base, state.failure_type, state.failure_message, total_start,
            )
        scaled_norms = state.term_norms / output_scale
        zero_terms = np.flatnonzero(scaled_norms <= self.config.zero_signature_tolerance)
        if zero_terms.size:
            base.term_norms = self._canonical_to_original(scaled_norms, canonical_order).tolist()
            base.signature_shape = tuple(state.signatures.shape)
            human_indices = ",".join(str(int(index + 1)) for index in zero_terms)
            return self._failure(
                base,
                FailureType.ZERO_SIGNATURE,
                f"Zero Sobolev signature for canonical term index/indices {human_indices}",
                total_start,
            )
        novelties = self._canonical_to_original(state.term_novelties, canonical_order)
        public_norms = self._canonical_to_original(scaled_norms, canonical_order)
        base.signature_shape = tuple(state.signatures.shape)
        legacy_tie_fallback = self._has_unstable_pruning_tie(
            np.asarray(base.coefficients, dtype=float),
            novelties,
            public_norms,
            self.config.threshold,
        )
        legacy_condition_fallback = (
            state.algorithm_used != "gram_cholesky"
            or not np.isfinite(state.condition_number)
            or state.condition_number > 1e4
        )
        legacy_numeric_fallback = legacy_tie_fallback or legacy_condition_fallback
        if legacy_numeric_fallback:
            original_values = self._canonical_columns_to_original(state.values, canonical_order)
            original_gradients = self._canonical_columns_to_original(state.gradients, canonical_order)
            try:
                bundle = construct_signatures(
                    values=original_values,
                    gradients=original_gradients,
                    full_values=full_values,
                    geometry_points=geometry_points,
                    input_means=input_means,
                    input_scales=input_scales,
                    lambda_value=self.config.lambda_value,
                    lambda_gradient=self.config.lambda_gradient,
                    min_valid_samples=self.config.min_valid_samples,
                    output_scale_tolerance=self.config.output_scale_tolerance,
                    zero_signature_tolerance=self.config.zero_signature_tolerance,
                )
                linear_start = time.perf_counter()
                exact_novelty = compute_novelties(
                    bundle.signatures,
                    bundle.term_norms,
                    self.config.rcond,
                    self.config.fast_gram,
                    self.config.gram_condition_threshold,
                )
                runtime.novelty_linear_algebra += time.perf_counter() - linear_start
            except SignatureError as error:
                base.valid_mask_size = error.valid_mask_size
                return self._failure(
                    base, FailureType(error.failure_type), str(error), total_start,
                )
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
                return self._failure(
                    base,
                    FailureType.LINEAR_ALGEBRA_FAILURE,
                    f"{type(error).__name__}: {error}",
                    total_start,
                )
            novelties = exact_novelty.novelties
            public_norms = bundle.term_norms
            output_scale = bundle.output_scale
            base.output_scale = output_scale
            base.valid_mask_size = int(bundle.valid_mask.sum())
            base.valid_mask_identity = array_identity(bundle.valid_mask.astype(np.uint8))
            base.signature_shape = tuple(bundle.signatures.shape)
            base.singular_values = exact_novelty.singular_values.tolist()
            base.rank = exact_novelty.rank
            base.condition_number = exact_novelty.condition_number
            base.algorithm_used = exact_novelty.algorithm_used
            base.fallback_reason = exact_novelty.fallback_reason
            base.geometry_reuse_mode = "cached_raw_legacy_numeric_fallback"
            base.geometry_fallback_reason = (
                "near_tied_pruning_impacts"
                if legacy_tie_fallback
                else "scale_sensitive_conditioning"
            )
        base.term_norms = public_norms.tolist()
        base.term_novelties = novelties.tolist()
        base.min_novelty = float(np.min(novelties))
        base.mean_novelty = float(np.mean(novelties))
        base.low_novelty_count = int(np.sum(novelties < self.config.threshold))
        base.low_novelty_ratio = base.low_novelty_count / len(novelties)
        if not legacy_numeric_fallback:
            base.singular_values = state.singular_values.tolist()
            base.rank = state.rank
            base.condition_number = state.condition_number
            base.algorithm_used = state.algorithm_used
            base.fallback_reason = state.fallback_reason
        base.term_diagnostics = []
        for index, term in enumerate(terms):
            base.term_diagnostics.append(
                {
                    "term_index": index + 1,
                    "coefficient": term.coefficient,
                    "basis_term": term.display,
                    "signature_norm": float(public_norms[index]),
                    "novelty": float(novelties[index]),
                    "deletion_impact": (
                        abs(term.coefficient)
                        * float(novelties[index])
                        * float(public_norms[index])
                    ),
                }
            )
        base.success = True
        base.failure_type = FailureType.NONE
        runtime.total_candidate_evaluation = time.perf_counter() - total_start
        return base

    def _collect_incremental_terms(
        self,
        parent_geometry_key: str,
        child_key: CandidateGeometryKey,
        child_terms: Sequence[TermSpec],
        symbols: Sequence[sp.Symbol],
        geometry_points: np.ndarray,
        dataset_identity: str,
        geometry_identity: str,
        normalization_identity: str,
        runtime: DetailedRuntime,
        base: EvaluationResult,
        total_start: float,
    ) -> tuple[np.ndarray, np.ndarray] | EvaluationResult | None:
        """Reuse unchanged parent columns and evaluate only added occurrences."""

        parent = self.geometry_cache.get_by_digest(parent_geometry_key)
        if parent is None:
            base.incremental_fallback_reason = "parent_geometry_not_cached"
            return None
        parent_context = (
            parent.key.dataset_identity,
            parent.key.geometry_subset_identity,
            parent.key.input_normalization_identity,
            parent.key.derivative_order,
            parent.key.lambda_value,
            parent.key.lambda_gradient,
            parent.key.operator_configuration,
        )
        child_context = (
            child_key.dataset_identity,
            child_key.geometry_subset_identity,
            child_key.input_normalization_identity,
            child_key.derivative_order,
            child_key.lambda_value,
            child_key.lambda_gradient,
            child_key.operator_configuration,
        )
        if parent_context != child_context:
            base.incremental_fallback_reason = "geometry_context_changed"
            return None
        parent_occurrences: dict[str, list[int]] = {}
        for index, canonical in enumerate(parent.key.canonical_terms):
            parent_occurrences.setdefault(canonical, []).append(index)
        matches: dict[int, int] = {}
        added_indices: list[int] = []
        for child_index, term in enumerate(child_terms):
            available = parent_occurrences.get(term.canonical, [])
            if available:
                matches[child_index] = available.pop(0)
            else:
                added_indices.append(child_index)
        removed_count = sum(len(indices) for indices in parent_occurrences.values())
        changed_count = max(len(added_indices), removed_count)
        base.reused_term_count = len(matches)
        base.added_term_count = len(added_indices)
        base.removed_term_count = removed_count
        base.changed_term_count = changed_count
        if changed_count > self.config.incremental_max_changed_terms:
            base.incremental_fallback_reason = (
                f"too_many_changed_terms:{changed_count}>"
                f"{self.config.incremental_max_changed_terms}"
            )
            return None
        if parent.values.shape[0] != len(geometry_points):
            base.incremental_fallback_reason = "parent_array_shape_changed"
            return None
        added_terms = [child_terms[index] for index in added_indices]
        if added_terms:
            added = self._collect_raw_terms(
                added_terms,
                symbols,
                geometry_points,
                dataset_identity,
                geometry_identity,
                normalization_identity,
                runtime,
                base,
                total_start,
            )
            if isinstance(added, EvaluationResult):
                return added
            added_values, added_gradients = added
        else:
            added_values = np.empty((len(geometry_points), 0), dtype=float)
            added_gradients = np.empty(
                (len(symbols), len(geometry_points), 0), dtype=float,
            )
        values = np.empty((len(geometry_points), len(child_terms)), dtype=float)
        gradients = np.empty(
            (len(symbols), len(geometry_points), len(child_terms)), dtype=float,
        )
        for child_index, parent_index in matches.items():
            values[:, child_index] = parent.values[:, parent_index]
            gradients[:, :, child_index] = parent.gradients[:, :, parent_index]
        for added_position, child_index in enumerate(added_indices):
            values[:, child_index] = added_values[:, added_position]
            gradients[:, :, child_index] = added_gradients[:, :, added_position]
        base.incremental_hit = True
        base.geometry_reuse_mode = "parent_child_incremental"
        return values, gradients

    def _collect_raw_terms(
        self,
        terms: Sequence[TermSpec],
        symbols: Sequence[sp.Symbol],
        geometry_points: np.ndarray,
        dataset_identity: str,
        geometry_identity: str,
        normalization_identity: str,
        runtime: DetailedRuntime,
        base: EvaluationResult,
        total_start: float,
    ) -> tuple[np.ndarray, np.ndarray] | EvaluationResult:
        hits_before, misses_before = self.cache.hits, self.cache.misses
        raw_values: list[np.ndarray] = []
        raw_gradients: list[np.ndarray] = []
        for term in terms:
            cache_start = time.perf_counter()
            key = CacheKey(
                canonical_term=term.canonical,
                dataset_identity=dataset_identity,
                geometry_subset_identity=geometry_identity,
                input_normalization_identity=normalization_identity,
                derivative_order=1,
                operator_configuration=f"protected_eps={self.config.protected_epsilon:.17g}",
            )
            cached = self.cache.get_raw(key)
            runtime.cache_lookup += time.perf_counter() - cache_start
            cache_miss = cached is None
            if cached is None:
                try:
                    cached = self._evaluate_raw_term(term, symbols, geometry_points)
                except Exception as error:
                    failure = (
                        FailureType.DIFFERENTIATION_FAILURE
                        if "differentiat" in str(error).lower()
                        else FailureType.EVALUATION_FAILURE
                    )
                    base.cache_hits = self.cache.hits - hits_before
                    base.cache_misses = self.cache.misses - misses_before
                    return self._failure(
                        base,
                        failure,
                        f"term={term.display}; {type(error).__name__}: {error}",
                        total_start,
                    )
                self.cache.put_raw(key, cached)
            if cache_miss:
                runtime.symbolic_differentiation += cached.symbolic_differentiation_time
                runtime.term_value_evaluation += cached.value_evaluation_time
                runtime.gradient_evaluation += cached.gradient_evaluation_time
            raw_values.append(cached.values)
            raw_gradients.append(cached.gradients)
        base.cache_hits = self.cache.hits - hits_before
        base.cache_misses = self.cache.misses - misses_before
        base.cache_memory_bytes = self.cache.memory_bytes
        return np.column_stack(raw_values), np.stack(raw_gradients, axis=2)

    def _build_geometry_state(
        self,
        key: CandidateGeometryKey,
        values: np.ndarray,
        gradients: np.ndarray,
        geometry_points: np.ndarray,
        input_scales: np.ndarray,
        runtime: DetailedRuntime,
        additional_mask: np.ndarray | None = None,
    ) -> GeometryState:
        mask_start = time.perf_counter()
        per_term_finite = np.isfinite(values)
        if gradients.shape[0]:
            per_term_finite &= np.all(np.isfinite(gradients), axis=0)
        shared_mask = np.all(np.isfinite(geometry_points), axis=1)
        shared_mask &= np.all(per_term_finite, axis=1)
        if additional_mask is not None:
            shared_mask &= np.asarray(additional_mask, dtype=bool)
        runtime.shared_mask += time.perf_counter() - mask_start
        n_valid = int(shared_mask.sum())
        empty = np.empty((0, values.shape[1]), dtype=float)
        if n_valid < self.config.min_valid_samples:
            return GeometryState(
                key=key,
                values=values,
                gradients=gradients,
                per_term_finite_masks=per_term_finite,
                shared_valid_mask=shared_mask,
                signatures=empty,
                term_norms=np.asarray([], dtype=float),
                term_novelties=np.asarray([], dtype=float),
                singular_values=np.asarray([], dtype=float),
                rank=0,
                condition_number=float("inf"),
                algorithm_used="not_run",
                fallback_reason=None,
                term_diagnostics=[],
                success=False,
                failure_type=FailureType.INSUFFICIENT_VALID_SAMPLES,
                failure_message=(
                    f"Only {n_valid}/{len(shared_mask)} candidate-wide finite samples; "
                    f"need {self.config.min_valid_samples}"
                ),
            )
        signature_start = time.perf_counter()
        valid_values = values[shared_mask]
        valid_gradients = gradients[:, shared_mask, :]
        dimension = geometry_points.shape[1]
        blocks = [
            np.sqrt(self.config.lambda_value / n_valid) * valid_values
        ]
        if self.config.lambda_gradient > 0 and dimension:
            gradients_z = valid_gradients * input_scales[:, None, None]
            factor = np.sqrt(self.config.lambda_gradient / (n_valid * dimension))
            blocks.extend(factor * gradients_z[index] for index in range(dimension))
        signatures = np.vstack(blocks)
        runtime.signature_construction += time.perf_counter() - signature_start
        term_norms = np.asarray(
            [stable_norm(signatures[:, index]) for index in range(signatures.shape[1])]
        )
        if np.any(~np.isfinite(signatures)):
            failure_type = FailureType.NUMERICAL_OVERFLOW
            failure_message = "Non-finite output-scale-free Sobolev signature"
        elif np.any(term_norms == 0):
            failure_type = FailureType.ZERO_SIGNATURE
            failure_message = "Exactly zero output-scale-free Sobolev signature"
        else:
            failure_type = FailureType.NONE
            failure_message = ""
        if failure_type is not FailureType.NONE:
            return GeometryState(
                key=key,
                values=values,
                gradients=gradients,
                per_term_finite_masks=per_term_finite,
                shared_valid_mask=shared_mask,
                signatures=signatures,
                term_norms=term_norms,
                term_novelties=np.asarray([], dtype=float),
                singular_values=np.asarray([], dtype=float),
                rank=0,
                condition_number=float("inf"),
                algorithm_used="not_run",
                fallback_reason=None,
                term_diagnostics=[],
                success=False,
                failure_type=failure_type,
                failure_message=failure_message,
            )
        linear_start = time.perf_counter()
        try:
            novelty = compute_novelties(
                signatures,
                term_norms,
                self.config.rcond,
                self.config.fast_gram,
                self.config.gram_condition_threshold,
            )
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
            runtime.novelty_linear_algebra += time.perf_counter() - linear_start
            return GeometryState(
                key=key,
                values=values,
                gradients=gradients,
                per_term_finite_masks=per_term_finite,
                shared_valid_mask=shared_mask,
                signatures=signatures,
                term_norms=term_norms,
                term_novelties=np.asarray([], dtype=float),
                singular_values=np.asarray([], dtype=float),
                rank=0,
                condition_number=float("inf"),
                algorithm_used="not_run",
                fallback_reason=None,
                term_diagnostics=[],
                success=False,
                failure_type=FailureType.LINEAR_ALGEBRA_FAILURE,
                failure_message=f"{type(error).__name__}: {error}",
            )
        runtime.novelty_linear_algebra += time.perf_counter() - linear_start
        return GeometryState(
            key=key,
            values=values,
            gradients=gradients,
            per_term_finite_masks=per_term_finite,
            shared_valid_mask=shared_mask,
            signatures=signatures,
            term_norms=term_norms,
            term_novelties=novelty.novelties,
            singular_values=novelty.singular_values,
            rank=novelty.rank,
            condition_number=novelty.condition_number,
            algorithm_used=novelty.algorithm_used,
            fallback_reason=novelty.fallback_reason,
            term_diagnostics=novelty.term_diagnostics,
        )

    @staticmethod
    def _canonical_to_original(values: np.ndarray, canonical_order: Sequence[int]) -> np.ndarray:
        output = np.empty_like(values)
        for canonical_index, original_index in enumerate(canonical_order):
            output[original_index] = values[canonical_index]
        return output

    @staticmethod
    def _canonical_columns_to_original(
        values: np.ndarray, canonical_order: Sequence[int],
    ) -> np.ndarray:
        output = np.empty_like(values)
        axis = values.ndim - 1
        for canonical_index, original_index in enumerate(canonical_order):
            source = [slice(None)] * values.ndim
            target = [slice(None)] * values.ndim
            source[axis] = canonical_index
            target[axis] = original_index
            output[tuple(target)] = values[tuple(source)]
        return output

    @staticmethod
    def _has_unstable_pruning_tie(
        coefficients: np.ndarray,
        novelties: np.ndarray,
        term_norms: np.ndarray,
        threshold: float,
    ) -> bool:
        eligible = np.flatnonzero(novelties < threshold)
        if eligible.size < 2:
            return False
        impacts = np.abs(coefficients) * novelties * term_norms
        ordered = sorted(float(impacts[index]) for index in eligible)
        return bool(np.isclose(ordered[0], ordered[1], rtol=1e-10, atol=1e-12))

    def _operator_configuration(self, feature_names: Sequence[str]) -> str:
        return (
            f"features={tuple(feature_names)!r};"
            f"protected_eps={self.config.protected_epsilon:.17g};"
            f"rcond={self.config.rcond:.17g};"
            f"fast_gram={self.config.fast_gram};"
            f"condition={self.config.gram_condition_threshold:.17g};"
            f"min_valid={self.config.min_valid_samples};"
            f"zero_tol={self.config.zero_signature_tolerance:.17g}"
        )

    def _evaluate_raw_term(
        self,
        term: TermSpec,
        symbols: Sequence[sp.Symbol],
        geometry_points: np.ndarray,
    ) -> RawTermEvaluation:
        differentiation_time = 0.0
        derivatives = self.cache.get_derivatives(term.canonical, symbols)
        if derivatives is None:
            start = time.perf_counter()
            try:
                derivatives = tuple(sp.diff(term.basis, symbol) for symbol in symbols)
            except Exception as error:
                raise ValueError(f"symbolic differentiation failed: {error}") from error
            differentiation_time = time.perf_counter() - start
            self.cache.put_derivatives(term.canonical, symbols, derivatives)
        value_start = time.perf_counter()
        values = evaluate_sympy(
            term.basis,
            symbols,
            geometry_points,
            self.config.protected_epsilon,
        )
        value_time = time.perf_counter() - value_start
        gradient_start = time.perf_counter()
        gradients = np.vstack(
            [
                evaluate_sympy(
                    derivative,
                    symbols,
                    geometry_points,
                    self.config.protected_epsilon,
                )
                for derivative in derivatives
            ]
        )
        gradient_time = time.perf_counter() - gradient_start
        return RawTermEvaluation(
            values=values,
            gradients=gradients,
            derivatives=derivatives,
            symbolic_differentiation_time=differentiation_time,
            value_evaluation_time=value_time,
            gradient_evaluation_time=gradient_time,
        )

    @staticmethod
    def _failure(
        result: EvaluationResult,
        failure_type: FailureType,
        message: str,
        total_start: float,
    ) -> EvaluationResult:
        result.success = False
        result.failure_type = failure_type
        result.failure_message = message
        result.detailed_runtime.total_candidate_evaluation = time.perf_counter() - total_start
        return result


def _coerce_inputs(
    X: np.ndarray | pd.DataFrame | Mapping[str, np.ndarray],
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    requested_names = (
        tuple(str(name) for name in feature_names)
        if feature_names is not None else None
    )
    if requested_names is not None and len(set(requested_names)) != len(requested_names):
        raise ValueError(f"Feature names must be unique: {requested_names}")
    if isinstance(X, pd.DataFrame):
        available = tuple(str(column) for column in X.columns)
        names = requested_names or available
        if set(names) != set(available) or len(names) != len(available):
            raise ValueError(
                f"DataFrame columns {available} do not match requested features {names}"
            )
        points = X.loc[:, list(names)].to_numpy(dtype=float)
    elif isinstance(X, Mapping):
        string_to_key = {str(name): name for name in X.keys()}
        available = tuple(string_to_key)
        names = requested_names or available
        if set(names) != set(available) or len(names) != len(available):
            raise ValueError(
                f"Mapping keys {available} do not match requested features {names}"
            )
        columns = [np.asarray(X[string_to_key[name]], dtype=float) for name in names]
        if not columns or any(column.ndim != 1 for column in columns):
            raise ValueError("Input mapping must contain one-dimensional columns")
        if len({len(column) for column in columns}) != 1:
            raise ValueError("Input mapping columns have inconsistent lengths")
        points = np.column_stack(columns)
    else:
        points = np.asarray(X, dtype=float)
        if points.ndim != 2:
            raise ValueError(f"Expected 2D X, got {points.shape}")
        names = requested_names or tuple(
            f"x_{index + 1}" for index in range(points.shape[1])
        )
    if points.ndim != 2 or points.shape[1] != len(names) or points.shape[0] == 0:
        raise ValueError(f"Input shape {points.shape} is inconsistent with features {names}")
    return points, names


def _content_bound_dataset_identity(
    caller_identity: str,
    points: np.ndarray,
    feature_names: Sequence[str],
) -> str:
    """Bind cache identity to ordered feature names and actual input bytes.

    Callers still provide a human/audit label, but an accidentally reused label
    can no longer make geometry or raw-term arrays from another dataset hit.
    """

    digest = hashlib.sha256()
    digest.update(str(caller_identity).encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        "\0".join(str(name) for name in feature_names).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(array_identity(np.asarray(points, dtype=float)).encode("ascii"))
    return f"{caller_identity}|content_sha256={digest.hexdigest()}"
