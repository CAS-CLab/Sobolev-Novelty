"""Correctness-first tests for the independent Sobolev evaluator."""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

import src.sobolev.evaluator as evaluator_module
from src.sobolev import (
    CandidateGeometryCache,
    FailureType,
    SobolevConfig,
    SobolevEvaluator,
    TermEvaluationCache,
)
from src.sobolev.decomposition import (
    DeferredSqrt,
    ProtectedInverse,
    _evaluate_numeric_coefficient,
    decompose_expand_mul,
    parse_expression,
    to_project_expression_string,
)
from src.sobolev.novelty import (
    gram_novelties,
    normalize_signatures,
    reference_novelties,
)
from src.sobolev.signature import evaluate_sympy, select_geometry_indices


def config(**kwargs: object) -> SobolevConfig:
    values: dict[str, object] = {"min_valid_samples": 16}
    values.update(kwargs)
    return SobolevConfig(**values)


def test_expand_mul_is_structural_and_preserves_repeated_terms() -> None:
    expression, symbols = parse_expression("x*(4-x-y/(1+x))", ("x", "y"))
    terms = decompose_expand_mul(expression, symbols)
    reconstructed = sum(term.coefficient * term.basis for term in terms)
    assert len(terms) == 3
    points = np.column_stack([np.linspace(0.1, 2, 100), np.linspace(-2, 1, 100)])
    assert np.allclose(
        evaluate_sympy(reconstructed, symbols, points),
        evaluate_sympy(expression, symbols, points),
        atol=1e-12,
        rtol=1e-12,
    )
    repeated, repeated_symbols = parse_expression("x+x", ("x",))
    assert len(decompose_expand_mul(repeated, repeated_symbols)) == 2


def test_symbol_free_protected_factors_are_numeric_coefficients() -> None:
    expression, symbols = parse_expression("2*x/tanh(2)**0.25", ("x",))
    terms = decompose_expand_mul(expression, symbols)
    assert len(terms) == 1
    assert terms[0].basis == symbols[0]
    expected = 2.0 / np.tanh(2.0) ** 0.25
    assert np.isclose(terms[0].coefficient, expected, atol=1e-14, rtol=1e-14)
    result = SobolevEvaluator(config()).evaluate(
        "2*x/tanh(2)**0.25",
        "2*x/tanh(2)**0.25",
        np.linspace(-1, 1, 64)[:, None],
        ("x",),
    )
    assert result.success


def test_deep_symbol_free_coefficient_is_evaluated_without_recursion() -> None:
    expression: sp.Expr = sp.Float(2)
    for _ in range(3000):
        expression = ProtectedInverse(expression)
    assert _evaluate_numeric_coefficient(expression) == complex(2.0)


def test_deep_sqrt_over_power_defers_only_pathological_sympy_simplification() -> None:
    deep = "sqrt(x2 / 3.1415926535897931) + x3"
    for _ in range(11):
        deep = f"sqrt({deep})"
    raw = f"-x1 + arccos(x2) + arccos(x2) - sqrt(sqrt(sqrt(({deep}) ** 2)))"
    expression, symbols = parse_expression(raw, ("x1", "x2", "x3"))
    assert expression.has(DeferredSqrt)
    assert "deferred_sqrt" not in to_project_expression_string(expression)
    points = np.array([[0.1, 0.4, 0.6], [0.2, 0.8, 0.3]])
    values = evaluate_sympy(expression, symbols, points)
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))

    ordinary, _ = parse_expression("sqrt(x1**2)", ("x1",))
    assert not ordinary.has(DeferredSqrt)


def test_deep_constant_subtrees_embedded_in_variable_expressions_finish() -> None:
    nonfinite = (
        "sqrt(sqrt(sqrt(sqrt(sqrt(log(sin(1)))) ** 3) ** 3)) * x3 * x4"
    )
    expression, symbols = parse_expression(nonfinite, ("x1", "x2", "x3", "x4"))
    with pytest.raises(ValueError, match="Non-finite numeric coefficient"):
        decompose_expand_mul(expression, symbols)

    finite = (
        "sqrt(sqrt(sqrt(cos(1) / 30.369656907932235)) ** 2) "
        "* x2 / 30.369656907932235"
    )
    expression, symbols = parse_expression(finite, ("x1", "x2"))
    terms = decompose_expand_mul(expression, symbols)
    assert len(terms) == 1
    assert np.isclose(terms[0].coefficient, 0.012025671831360182)
    assert terms[0].basis == symbols[1]

    overflow = (
        "exp(exp(exp(exp(3.1415926535897931))) - x1) "
        "- (cos(x2 + arcsin(tanh(exp(3.1415926535897931))) - x1) "
        "- sin(x2) / sin(1))"
    )
    expression, symbols = parse_expression(overflow, ("x1", "x2"))
    terms = decompose_expand_mul(expression, symbols)
    assert len(terms) == 3
    assert all(np.isfinite(term.coefficient) for term in terms)


def test_sympy_internal_parse_failure_is_structured_and_cached(monkeypatch) -> None:
    calls = 0

    def fail_parse(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AttributeError("LazyExceptionMessage has no attribute startswith")

    monkeypatch.setattr(evaluator_module, "parse_expression", fail_parse)
    evaluator = SobolevEvaluator(config(candidate_geometry_cache=True))
    points = np.linspace(-1, 1, 64)[:, None]
    first = evaluator.evaluate("x", "x", points, ("x",), "parse-failure")
    second = evaluator.evaluate("x", "x", points, ("x",), "parse-failure")
    assert not first.success and not second.success
    assert first.failure_type == second.failure_type == FailureType.PARSE_FAILURE
    assert "AttributeError" in first.failure_message
    assert not first.failure_cache_hit and second.failure_cache_hit
    assert calls == 1


@pytest.mark.parametrize(
    "raised",
    [MemoryError("pathological symbolic coefficient"), KeyError("ComplexInfinity")],
)
def test_sympy_internal_decomposition_failure_is_structured(monkeypatch, raised) -> None:
    def fail_decomposition(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr(evaluator_module, "decompose_expand_mul", fail_decomposition)
    result = SobolevEvaluator(config()).evaluate(
        "x", "x", np.linspace(-1, 1, 64)[:, None], ("x",), "decomposition-failure"
    )
    assert not result.success
    assert result.failure_type == FailureType.DECOMPOSITION_FAILURE
    assert type(raised).__name__ in result.failure_message


def test_exact_algebraic_redundancy() -> None:
    x = sp.Symbol("x", real=True)
    points = np.linspace(-math.pi, math.pi, 256)[:, None]
    result = SobolevEvaluator(config()).evaluate_terms(
        [(1.0, sp.sin(x) ** 2), (1.0, sp.Integer(1)), (1.0, sp.cos(2 * x))],
        (x,),
        points,
    )
    assert result.success
    assert result.rank == 2
    assert result.algorithm_used == "loo_lstsq"
    assert max(result.term_novelties) < 1e-10


def test_local_geometric_redundancy_changes_with_domain() -> None:
    x = sp.Symbol("x", real=True)
    evaluator = SobolevEvaluator(config(fast_gram=False))
    small = evaluator.evaluate_terms(
        [(1.0, sp.sin(x)), (1.0, x)], (x,), np.linspace(-0.05, 0.05, 256)[:, None], "small"
    )
    large = evaluator.evaluate_terms(
        [(1.0, sp.sin(x)), (1.0, x)], (x,), np.linspace(-4, 4, 256)[:, None], "large"
    )
    assert small.success and large.success
    assert small.mean_novelty is not None and large.mean_novelty is not None
    assert small.mean_novelty < 0.01
    assert large.mean_novelty > small.mean_novelty + 0.5


def test_gradient_block_detects_value_close_derivative_difference() -> None:
    x = sp.Symbol("x", real=True)
    points = np.linspace(-1, 1, 1000)[:, None]
    terms = [(1.0, x**2), (1.0, x**2 + 0.01 * sp.sin(100 * x))]
    value = SobolevEvaluator(config(lambda_gradient=0, fast_gram=False)).evaluate_terms(terms, (x,), points)
    sobolev = SobolevEvaluator(config(lambda_gradient=1, fast_gram=False)).evaluate_terms(terms, (x,), points)
    assert value.success and sobolev.success
    assert max(sobolev.term_novelties) > max(value.term_novelties) * 20


def test_single_term_and_exact_repeat_conventions() -> None:
    points = np.linspace(-1, 1, 128)[:, None]
    single = SobolevEvaluator(config()).evaluate("x", "x", points, ("x",))
    repeated = SobolevEvaluator(config()).evaluate("x+x", "x+x", points, ("x",))
    assert single.success and single.term_novelties == [1.0]
    assert repeated.success and repeated.rank == 1
    assert repeated.fallback_reason == "rank_deficient:1<2"
    assert max(repeated.term_novelties) < 1e-12


def test_basis_scaling_invariance() -> None:
    x = sp.Symbol("x", real=True)
    points = np.linspace(-2, 2, 256)[:, None]
    evaluator = SobolevEvaluator(config(fast_gram=False))
    original = evaluator.evaluate_terms([(1.0, x), (1.0, x**2)], (x,), points, "scale1")
    scaled = evaluator.evaluate_terms([(1.0, 37 * x), (1.0, x**2)], (x,), points, "scale2")
    assert original.success and scaled.success
    assert np.allclose(original.term_novelties, scaled.term_novelties, atol=1e-12, rtol=1e-12)


def test_constants_zero_terms_and_zero_output_are_explicit() -> None:
    x = sp.Symbol("x", real=True)
    points = np.linspace(-1, 1, 128)[:, None]
    evaluator = SobolevEvaluator(config())
    constant = evaluator.evaluate_terms([(2.0, sp.Integer(1))], (x,), points, "constant")
    zero_term = evaluator.evaluate_terms([(1.0, x), (1.0, sp.Integer(0))], (x,), points, "zero_term")
    zero_output = evaluator.evaluate_terms([(0.0, x)], (x,), points, "zero_output")
    assert constant.success and constant.term_novelties == [1.0]
    assert not zero_term.success and zero_term.failure_type == FailureType.ZERO_SIGNATURE
    assert not zero_output.success and zero_output.failure_type == FailureType.ZERO_OUTPUT_SCALE


def test_invalid_domains_share_one_mask_and_fail_loudly() -> None:
    x = sp.Symbol("x", real=True)
    points = np.linspace(-2, 2, 64)[:, None]
    strict = SobolevEvaluator(config(min_valid_samples=48))
    result = strict.evaluate_terms([(1.0, sp.sqrt(x)), (1.0, sp.log(x))], (x,), points, "domain")
    assert not result.success
    assert result.failure_type == FailureType.INSUFFICIENT_VALID_SAMPLES
    assert result.valid_mask_size < 48
    overflow_points = np.linspace(700, 800, 64)[:, None]
    overflow = strict.evaluate_terms([(1.0, sp.exp(x)), (1.0, x)], (x,), overflow_points, "overflow")
    assert not overflow.success
    assert overflow.failure_type == FailureType.INSUFFICIENT_VALID_SAMPLES


def test_protected_division_and_cached_derivatives_are_exact() -> None:
    points = np.linspace(-1, 1, 129)[:, None]
    cache = TermEvaluationCache(enabled=True)
    evaluator = SobolevEvaluator(config(), cache)
    result = evaluator.evaluate("1/x+x", "1/x+x", points, ("x",), "protected_division")
    assert result.success
    assert result.valid_mask_size == len(points)
    derivative_sets = list(cache._derivatives.values())
    assert derivative_sets
    # Symbolic expressions, not finite differences, are retained in the cache.
    assert all(isinstance(derivative, sp.Expr) for group in derivative_sets for derivative in group)


def test_multivariate_input_scale_is_applied_to_gradients() -> None:
    x, y = sp.symbols("x y", real=True)
    rng = np.random.default_rng(4)
    points = np.column_stack([rng.normal(size=300), 100 * rng.normal(size=300)])
    result = SobolevEvaluator(config(fast_gram=False)).evaluate_terms(
        [(1.0, x), (1.0, y / 100)], (x, y), points, "multiscale"
    )
    assert result.success
    assert np.allclose(result.input_scales, np.std(points, axis=0), rtol=1e-12, atol=1e-12)
    # Once d/dz_j = sigma_j*d/dx_j is used, these equally distributed terms have equal norms.
    assert np.isclose(result.term_norms[0], result.term_norms[1], rtol=0.08)


def test_gram_matches_reference_and_rank_deficiency_falls_back() -> None:
    rng = np.random.default_rng(12)
    signatures = rng.normal(size=(500, 8))
    norms = np.linalg.norm(signatures, axis=0)
    normalized = normalize_signatures(signatures, norms)
    reference, _ = reference_novelties(normalized)
    gram = gram_novelties(normalized)
    assert np.max(np.abs(reference - gram)) <= 1e-8
    assert np.max(np.abs(reference - gram) / np.maximum(np.abs(reference), 1e-15)) <= 1e-6
    x = sp.Symbol("x", real=True)
    duplicate = SobolevEvaluator(config()).evaluate_terms(
        [(1.0, x), (1.0, x)], (x,), np.linspace(-1, 1, 128)[:, None], "duplicate"
    )
    assert duplicate.success and duplicate.algorithm_used == "loo_lstsq"
    assert duplicate.condition_number == math.inf


def test_cache_on_off_is_numerically_identical_and_cache_hits() -> None:
    points = np.linspace(-2, 2, 256)[:, None]
    enabled_cache = TermEvaluationCache(enabled=True)
    enabled = SobolevEvaluator(config(cache_enabled=True), enabled_cache)
    first = enabled.evaluate("x+sin(x)", "x+sin(x)", points, ("x",), "cache")
    second = enabled.evaluate("x+sin(x)", "x+sin(x)", points, ("x",), "cache")
    disabled = SobolevEvaluator(config(cache_enabled=False), TermEvaluationCache(enabled=False)).evaluate(
        "x+sin(x)", "x+sin(x)", points, ("x",), "cache"
    )
    assert first.success and second.success and disabled.success
    assert second.cache_hits == len(second.terms)
    assert np.array_equal(np.asarray(first.term_novelties), np.asarray(second.term_novelties))
    assert np.allclose(first.term_novelties, disabled.term_novelties, atol=1e-14, rtol=1e-14)


def test_scale_free_geometry_cache_matches_reference_and_ignores_coefficients() -> None:
    points = np.linspace(-2, 2, 512)[:, None]
    reference = SobolevEvaluator(config(fast_gram=False))
    geometry_cache = CandidateGeometryCache(enabled=True)
    optimized = SobolevEvaluator(
        config(
            fast_gram=False,
            candidate_geometry_cache=True,
            output_scale_free_internal=True,
        ),
        geometry_cache=geometry_cache,
    )
    expressions = [
        "2*x + 3*sin(x) + 0.2*x**2",
        "11*x + 7*sin(x) - 4*x**2",
    ]
    old_results = [reference.evaluate(value, value, points, ("x",), "candidate-cache") for value in expressions]
    new_results = [optimized.evaluate(value, value, points, ("x",), "candidate-cache") for value in expressions]
    assert all(result.success for result in [*old_results, *new_results])
    for old, new in zip(old_results, new_results, strict=True):
        assert np.allclose(old.term_novelties, new.term_novelties, atol=1e-12, rtol=1e-12)
        assert np.allclose(old.term_norms, new.term_norms, atol=1e-12, rtol=1e-12)
        assert old.rank == new.rank and old.algorithm_used == new.algorithm_used
    assert not new_results[0].geometry_cache_hit
    assert new_results[1].geometry_cache_hit
    assert new_results[0].candidate_geometry_key == new_results[1].candidate_geometry_key
    assert geometry_cache.hits == 1


def test_geometry_key_is_a_multiset_and_failure_cache_is_typed() -> None:
    x = sp.Symbol("x", real=True)
    points = np.linspace(-2, 2, 128)[:, None]
    evaluator = SobolevEvaluator(
        config(
            min_valid_samples=100,
            candidate_geometry_cache=True,
            output_scale_free_internal=True,
        )
    )
    first = evaluator.evaluate_terms(
        [(1.0, sp.sqrt(x)), (1.0, sp.log(x))],
        (x,),
        points,
        "geometry-failure",
    )
    second = evaluator.evaluate_terms(
        [(8.0, sp.log(x)), (-3.0, sp.sqrt(x))],
        (x,),
        points,
        "geometry-failure",
    )
    assert first.failure_type == second.failure_type == FailureType.INSUFFICIENT_VALID_SAMPLES
    assert not first.failure_cache_hit and second.failure_cache_hit
    assert first.candidate_geometry_key == second.candidate_geometry_key
    duplicate = evaluator.evaluate_terms(
        [(1.0, x), (1.0, x)], (x,), points, "geometry-duplicate"
    )
    single = evaluator.evaluate_terms([(2.0, x)], (x,), points, "geometry-duplicate")
    assert duplicate.candidate_geometry_key != single.candidate_geometry_key


def test_scale_free_path_preserves_zero_output_and_pruning_order() -> None:
    x = sp.Symbol("x", real=True)
    points = np.linspace(-1, 1, 256)[:, None]
    old = SobolevEvaluator(config(fast_gram=False))
    new = SobolevEvaluator(
        config(
            fast_gram=False,
            candidate_geometry_cache=True,
            output_scale_free_internal=True,
        )
    )
    zero_old = old.evaluate("x-x", "x-x", points, ("x",), "zero-scale")
    zero_new = new.evaluate("x-x", "x-x", points, ("x",), "zero-scale")
    assert zero_old.failure_type == zero_new.failure_type == FailureType.ZERO_OUTPUT_SCALE
    expression = "0.5*x + 4*sin(x) + 0.1*x**2"
    result_old = old.evaluate(expression, expression, points, ("x",), "prune-scale")
    result_new = new.evaluate(expression, expression, points, ("x",), "prune-scale")
    assert result_old.success and result_new.success
    impact_old = np.abs(result_old.coefficients) * result_old.term_novelties * result_old.term_norms
    impact_new = np.abs(result_new.coefficients) * result_new.term_novelties * result_new.term_norms
    eligible_old = np.flatnonzero(np.asarray(result_old.term_novelties) < result_old.threshold)
    eligible_new = np.flatnonzero(np.asarray(result_new.term_novelties) < result_new.threshold)
    assert np.array_equal(eligible_old, eligible_new)
    if eligible_old.size:
        assert eligible_old[np.argmin(impact_old[eligible_old])] == eligible_new[np.argmin(impact_new[eligible_new])]


def test_parent_child_incremental_reuses_only_unchanged_occurrences() -> None:
    rng = np.random.default_rng(20260803)
    points = rng.uniform(0.2, 2.0, size=(256, 2))
    names = ("x", "y")
    reference = SobolevEvaluator(config(fast_gram=False))
    incremental = SobolevEvaluator(
        config(
            fast_gram=False,
            candidate_geometry_cache=True,
            output_scale_free_internal=True,
            parent_child_incremental=True,
        )
    )
    parent_expression = "x + sin(y) + x*y"
    child_expression = "x + sin(y) + x*y + y**2"
    parent = incremental.evaluate(
        parent_expression, parent_expression, points, names, "incremental-add"
    )
    child = incremental.evaluate(
        child_expression,
        child_expression,
        points,
        names,
        "incremental-add",
        parent_geometry_key=parent.candidate_geometry_key,
    )
    expected = reference.evaluate(
        child_expression, child_expression, points, names, "incremental-add"
    )
    assert parent.success and child.success and expected.success
    assert child.incremental_hit
    assert child.reused_term_count == 3
    assert child.added_term_count == 1
    assert child.removed_term_count == 0
    assert child.changed_term_count == 1
    assert np.allclose(child.term_novelties, expected.term_novelties, atol=1e-10, rtol=1e-10)


def test_parent_child_deletion_rebuilds_expanded_shared_mask() -> None:
    x = np.linspace(-1, 1, 128)
    points = np.column_stack([x, np.linspace(0.2, 2.0, 128)])
    names = ("x", "y")
    incremental = SobolevEvaluator(
        config(
            min_valid_samples=32,
            fast_gram=False,
            candidate_geometry_cache=True,
            output_scale_free_internal=True,
            parent_child_incremental=True,
        )
    )
    parent = incremental.evaluate(
        "sqrt(x)+y", "sqrt(x)+y", points, names, "incremental-mask"
    )
    child = incremental.evaluate(
        "y",
        "y",
        points,
        names,
        "incremental-mask",
        parent_geometry_key=parent.candidate_geometry_key,
    )
    reference = SobolevEvaluator(config(min_valid_samples=32, fast_gram=False)).evaluate(
        "y", "y", points, names, "incremental-mask"
    )
    assert parent.success and child.success and reference.success
    assert child.incremental_hit
    assert child.reused_term_count == 1
    assert child.added_term_count == 0 and child.removed_term_count == 1
    assert child.valid_mask_size == reference.valid_mask_size == len(points)
    assert child.term_novelties == reference.term_novelties == [1.0]


def test_parent_child_incremental_has_explicit_context_fallback() -> None:
    points = np.linspace(-1, 1, 128)[:, None]
    evaluator = SobolevEvaluator(
        config(
            candidate_geometry_cache=True,
            output_scale_free_internal=True,
            parent_child_incremental=True,
        )
    )
    parent = evaluator.evaluate("x+sin(x)", "x+sin(x)", points, ("x",), "parent-data")
    child = evaluator.evaluate(
        "x+cos(x)",
        "x+cos(x)",
        points,
        ("x",),
        "different-data",
        parent_geometry_key=parent.candidate_geometry_key,
    )
    assert parent.success and child.success
    assert not child.incremental_hit
    assert child.incremental_fallback_reason == "geometry_context_changed"


def test_preparsed_incremental_api_bypasses_parse_without_changing_result() -> None:
    points = np.linspace(-1, 1, 128)[:, None]
    evaluator = SobolevEvaluator(
        config(
            candidate_geometry_cache=True,
            output_scale_free_internal=True,
            parent_child_incremental=True,
        )
    )
    parent_text = "x+sin(x)"
    parent = evaluator.evaluate(parent_text, parent_text, points, ("x",), "preparsed")
    expression, symbols = parse_expression("x+sin(x)+x**2", ("x",))
    terms = decompose_expand_mul(expression, symbols)
    child = evaluator.evaluate_preparsed(
        "preparsed-child",
        expression,
        expression,
        symbols,
        terms,
        points,
        "preparsed",
        parent_geometry_key=parent.candidate_geometry_key,
    )
    reference = SobolevEvaluator(config()).evaluate(
        expression, expression, points, ("x",), "preparsed"
    )
    assert child.success and reference.success and child.incremental_hit
    assert child.detailed_runtime.parsing == 0
    assert child.detailed_runtime.decomposition == 0
    assert np.allclose(child.term_novelties, reference.term_novelties, atol=1e-10, rtol=1e-10)


def test_geometry_subset_is_fixed_and_full_is_distinguished() -> None:
    indices_a = select_geometry_indices(1000, "dataset", 42, 256)
    indices_b = select_geometry_indices(1000, "dataset", 42, 256)
    indices_c = select_geometry_indices(1000, "dataset", 43, 256)
    full = select_geometry_indices(1000, "dataset", 42, None)
    assert np.array_equal(indices_a, indices_b)
    assert not np.array_equal(indices_a, indices_c)
    assert len(indices_a) == 256 and len(full) == 1000
    points = np.linspace(-3, 3, 1000)[:, None]
    subset_result = SobolevEvaluator(config(geometry_sample_size=256)).evaluate(
        "x+sin(x)", "x+sin(x)", points, ("x",), "dataset"
    )
    full_result = SobolevEvaluator(config(geometry_sample_size=None)).evaluate(
        "x+sin(x)", "x+sin(x)", points, ("x",), "dataset"
    )
    assert subset_result.success and full_result.success
    assert len(subset_result.geometry_indices) == 256
    assert len(full_result.geometry_indices) == 1000
    assert max(abs(a - b) for a, b in zip(subset_result.term_novelties, full_result.term_novelties)) < 0.02


def test_against_direct_final_report_lstsq_definition() -> None:
    x = sp.Symbol("x", real=True)
    points = np.linspace(-1.5, 2.0, 300)[:, None]
    result = SobolevEvaluator(config(fast_gram=False)).evaluate_terms(
        [(2.0, x), (-0.5, x**2), (1.0, sp.sin(x))], (x,), points, "reference"
    )
    assert result.success
    values = np.column_stack([points[:, 0], points[:, 0] ** 2, np.sin(points[:, 0])])
    gradients = np.column_stack([np.ones(len(points)), 2 * points[:, 0], np.cos(points[:, 0])])
    scale = np.std(points[:, 0])
    full = values @ np.array([2.0, -0.5, 1.0])
    sy = np.sqrt(np.mean(full**2))
    signature = np.vstack([values / np.sqrt(len(points)) / sy, gradients * scale / np.sqrt(len(points)) / sy])
    direct = []
    for index in range(3):
        vector = signature[:, index]
        others = np.delete(signature, index, axis=1)
        coefficient, *_ = np.linalg.lstsq(others, vector, rcond=1e-10)
        direct.append(np.linalg.norm(vector - others @ coefficient) / np.linalg.norm(vector))
    assert np.allclose(result.term_novelties, direct, atol=1e-12, rtol=1e-12)
