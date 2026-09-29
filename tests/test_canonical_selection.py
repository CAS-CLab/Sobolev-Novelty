from __future__ import annotations

import numpy as np

from sn_dsrrans.canonical_selection import (
    canonical_terms,
    choose_candidate,
    conditional_novelties,
    marginal_sse_reductions,
    sobolev_signatures,
    tensor_design,
    tensor_sobolev_signatures,
)
from sn_dsrrans.data import DSRRANSData


def test_canonical_dictionary_contains_each_channel_monomial_once() -> None:
    terms = canonical_terms(2)
    assert len(terms) == 18
    for channel in range(3):
        assert {
            term.exponent for term in terms if term.channel == channel
        } == {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}


def test_base_marginal_reduction_selects_exact_predictor() -> None:
    rng = np.random.default_rng(17)
    exact = rng.normal(size=200)
    nuisance = rng.normal(size=200)
    design = np.column_stack((exact, nuisance, exact + nuisance))
    reductions, residual, coefficients = marginal_sse_reductions(
        design, exact, [], [0, 1, 2]
    )
    assert coefficients.size == 0
    np.testing.assert_allclose(residual, exact)
    selected, details = choose_candidate(
        [0, 1, 2],
        reductions,
        np.ones(3),
        method="base",
        epsilon=0.05,
    )
    assert selected == 0
    assert details["changed_from_base"] == 0


def test_sn_epsilon_only_changes_base_equivalent_choice() -> None:
    candidates = [4, 8, 12]
    reductions = np.asarray([10.0, 9.7, 5.0])
    novelties = np.asarray([0.1, 0.9, 1.0])
    selected, details = choose_candidate(
        candidates,
        reductions,
        novelties,
        method="sn_epsilon",
        epsilon=0.05,
    )
    assert selected == 8
    assert details["eligible_count"] == 2
    assert details["changed_from_base"] == 1
    assert details["chosen_reduction"] >= 0.95 * details["base_best_reduction"]


def test_tau_guard_only_intervenes_when_base_term_is_below_tau() -> None:
    candidates = [4, 8, 12]
    reductions = np.asarray([10.0, 9.8, 9.7])
    selected, details = choose_candidate(
        candidates,
        reductions,
        np.asarray([0.1, 0.4, 0.9]),
        method="sn_tau_guard",
        epsilon=0.05,
        tau=0.316,
    )
    assert selected == 8
    assert details["changed_from_base"] == 1
    selected, details = choose_candidate(
        candidates,
        reductions,
        np.asarray([0.4, 0.9, 1.0]),
        method="sn_tau_guard",
        epsilon=0.05,
        tau=0.316,
    )
    assert selected == 4
    assert details["changed_from_base"] == 0


def test_conditional_novelty_is_channel_local() -> None:
    points = np.random.default_rng(21).uniform(-1.0, 1.0, size=(64, 2))
    terms = canonical_terms(1)
    signatures = sobolev_signatures(points, terms)
    channel_zero_constant = next(
        index
        for index, term in enumerate(terms)
        if term.channel == 0 and term.exponent == (0, 0)
    )
    channel_one_constant = next(
        index
        for index, term in enumerate(terms)
        if term.channel == 1 and term.exponent == (0, 0)
    )
    channel_zero_x1 = next(
        index
        for index, term in enumerate(terms)
        if term.channel == 0 and term.exponent == (1, 0)
    )
    novelties = conditional_novelties(
        signatures,
        terms,
        [channel_zero_constant],
        [channel_one_constant, channel_zero_x1],
    )
    assert novelties[0] == 1.0
    assert 0.9 < novelties[1] <= 1.0


def test_tensor_value_signature_matches_normalized_output_design() -> None:
    rng = np.random.default_rng(29)
    n = 12
    data = DSRRANSData(
        raw_invariants=rng.normal(size=(n, 2)),
        invariants=rng.uniform(-0.8, 0.8, size=(n, 2)),
        tensor_bases=rng.normal(size=(n, 3, 3, 3)),
        target_tensor=np.zeros((n, 3, 3)),
    )
    terms = canonical_terms(2)
    signatures = tensor_sobolev_signatures(
        data, terms, lambda_gradient=0.0
    )
    design = tensor_design(data, terms)
    expected = design / np.linalg.norm(design, axis=0)
    np.testing.assert_allclose(signatures, expected, atol=1e-12)


def test_tensor_geometry_can_compare_terms_across_channels() -> None:
    n = 10
    tensors = np.zeros((n, 3, 3, 3))
    tensors[:, 0, 0, 0] = 1.0
    tensors[:, 1, 0, 0] = 1.0
    data = DSRRANSData(
        raw_invariants=np.zeros((n, 2)),
        invariants=np.column_stack((np.linspace(-0.5, 0.5, n), np.ones(n))),
        tensor_bases=tensors,
        target_tensor=np.zeros((n, 3, 3)),
    )
    terms = canonical_terms(0)
    signatures = tensor_sobolev_signatures(data, terms)
    local = conditional_novelties(signatures, terms, [0], [1])
    global_novelty = conditional_novelties(
        signatures, terms, [0], [1], channel_local=False
    )
    assert local[0] == 1.0
    assert global_novelty[0] < 1e-8
