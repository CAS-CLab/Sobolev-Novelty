"""Unit tests for independent Sobolev prune-and-refit orchestration."""

from __future__ import annotations

import numpy as np
import src.nd2py.nd2py as nd

from src.sobolev import (
    EvaluationResult,
    PruningConfig,
    RefitResult,
    deletion_impacts,
    prune_and_refit,
)
from src.sobolev.decomposition import parse_expression, to_project_expression_string
from src.heuristic.mcts.mcts import MCTS, Node


def analysis(
    terms: list[str],
    coefficients: list[float],
    novelties: list[float],
    norms: list[float] | None = None,
    *,
    rank: int | None = None,
) -> EvaluationResult:
    values = norms or [1.0] * len(terms)
    return EvaluationResult(
        raw_expression="raw",
        fitted_expression=" + ".join(terms),
        terms=terms,
        coefficients=coefficients,
        term_novelties=novelties,
        term_norms=values,
        rank=len(terms) if rank is None else rank,
        success=True,
    )


def fit(expression: str, reward: float, *, r2: float = 0.9, complexity: int = 10) -> RefitResult:
    return RefitResult(
        expression=expression,
        r2=r2,
        complexity=complexity,
        eic=0.0,
        base_reward=reward,
        success=True,
    )


def test_exact_duplicate_is_deleted_and_rank_deficiency_is_allowed() -> None:
    initial_analysis = analysis(["x", "x"], [0.5, 0.5], [0.0, 0.0], rank=1)
    initial_fit = fit("0.5*x + 0.5*x", 0.8)

    def refit_callback(_fit: RefitResult, _analysis: EvaluationResult, index: int) -> RefitResult:
        assert index == 0
        return fit("1.0*x", 0.8, complexity=2)

    result = prune_and_refit(
        initial_fit,
        initial_analysis,
        PruningConfig(threshold=0.3, max_prunes=3),
        refit_callback,
        lambda _: analysis(["x"], [1.0], [1.0]),
    )
    assert result.success
    assert result.accepted_prunes == 1 and result.rejected_prunes == 0
    assert result.final_expression == "1.0*x"
    assert result.termination_reason == "single_term_remaining"
    assert result.steps[0].removed_term == "x"


def test_deletion_impact_uses_coefficient_novelty_and_norm() -> None:
    current = analysis(["large", "small", "high_novelty"], [10.0, 0.2, 1e-8], [0.1, 0.2, 0.9], [2, 3, 4])
    assert np.allclose(deletion_impacts(current), [2.0, 0.12, 3.6e-08], atol=0, rtol=1e-15)
    removed: list[int] = []

    def refit_callback(_fit: RefitResult, _analysis: EvaluationResult, index: int) -> RefitResult:
        removed.append(index)
        return fit("large + high_novelty", 0.9)

    result = prune_and_refit(
        fit("all", 0.9),
        current,
        PruningConfig(threshold=0.3, max_prunes=1),
        refit_callback,
        lambda _: analysis(["large", "high_novelty"], [10.0, 1e-8], [0.5, 0.9]),
    )
    assert removed == [1]
    assert result.steps[0].removed_term == "small"
    # The tiny-coefficient high-novelty term has smaller D but is not eligible.
    assert result.steps[0].removed_term != "high_novelty"


def test_excluded_low_novelty_term_is_not_selected_for_repair() -> None:
    current = analysis(
        ["intercept", "duplicate", "independent"],
        [0.01, 0.2, 1.0],
        [0.0, 0.1, 0.9],
        [1.0, 1.0, 1.0],
    )
    removed: list[int] = []

    def refit_callback(
        _fit: RefitResult, _analysis: EvaluationResult, index: int
    ) -> RefitResult:
        removed.append(index)
        return fit("intercept + independent", 0.9)

    result = prune_and_refit(
        fit("all", 0.9),
        current,
        PruningConfig(
            threshold=0.3,
            max_prunes=1,
            excluded_term_indices=(0,),
        ),
        refit_callback,
        lambda _: analysis(
            ["intercept", "independent"], [0.01, 1.0], [0.5, 0.9]
        ),
    )
    assert removed == [1]
    assert result.steps[0].removed_term == "duplicate"


def test_reward_decrease_rejects_without_changing_final_expression() -> None:
    result = prune_and_refit(
        fit("x + x", 0.8),
        analysis(["x", "x"], [1, 1], [0, 0]),
        PruningConfig(threshold=0.3, max_prunes=2),
        lambda *_: fit("x", 0.799),
        lambda _: analysis(["x"], [1], [1]),
    )
    assert result.success
    assert result.accepted_prunes == 0 and result.rejected_prunes == 1
    assert result.final_expression == "x + x"
    assert result.termination_reason == "base_reward_decrease"
    assert not result.steps[0].accepted


def test_acceptance_tolerance_and_post_accept_reevaluation() -> None:
    calls = {"reevaluate": 0}

    def reevaluate(_: RefitResult) -> EvaluationResult:
        calls["reevaluate"] += 1
        return analysis(["x"], [1], [1])

    result = prune_and_refit(
        fit("x+x", 0.8),
        analysis(["x", "x"], [1, 1], [0, 0]),
        PruningConfig(threshold=0.3, max_prunes=2, acceptance_tolerance=0.001),
        lambda *_: fit("x", 0.7995),
        reevaluate,
    )
    assert result.accepted_prunes == 1
    assert calls["reevaluate"] == 1
    assert result.final_analysis.term_novelties == [1]
    assert result.steps[0].evaluator_success_after is True


def test_multiple_prunes_refit_each_time_and_terminate() -> None:
    queued = [
        analysis(["b", "c"], [1, 1], [0.1, 0.2]),
        analysis(["c"], [1], [1.0]),
    ]
    expressions = iter(["b+c", "c"])
    callback_indices: list[int] = []

    def refit_callback(current: RefitResult, _analysis: EvaluationResult, index: int) -> RefitResult:
        callback_indices.append(index)
        return fit(next(expressions), float(current.base_reward), complexity=int(current.complexity) - 2)

    result = prune_and_refit(
        fit("a+b+c", 0.75, complexity=9),
        analysis(["a", "b", "c"], [0.1, 1, 1], [0.01, 0.1, 0.2]),
        PruningConfig(threshold=0.3, max_prunes=5),
        refit_callback,
        lambda _: queued.pop(0),
    )
    assert result.accepted_prunes == 2
    assert callback_indices == [0, 0]
    assert result.final_expression == "c"
    assert result.termination_reason == "single_term_remaining"
    assert [step.expression_before for step in result.steps] == ["a+b+c", "b+c"]


def test_single_term_and_max_prunes_zero_never_call_refit() -> None:
    calls = 0

    def forbidden(*_: object) -> RefitResult:
        nonlocal calls
        calls += 1
        raise AssertionError("refit must not be called")

    single = prune_and_refit(
        fit("x", 0.8),
        analysis(["x"], [1], [0.0]),
        PruningConfig(threshold=0.3, max_prunes=4),
        forbidden,
        lambda _: analysis(["x"], [1], [1]),
    )
    disabled = prune_and_refit(
        fit("x+x", 0.8),
        analysis(["x", "x"], [1, 1], [0, 0]),
        PruningConfig(threshold=0.3, max_prunes=0),
        forbidden,
        lambda _: analysis(["x"], [1], [1]),
    )
    assert calls == 0
    assert single.termination_reason == "single_term_remaining"
    assert disabled.termination_reason == "max_prunes_zero"


def test_refit_failure_and_raw_to_final_mapping_are_explicit() -> None:
    failed = RefitResult(success=False, failure_type="lstsq_failure", failure_message="rank exploded")
    result = prune_and_refit(
        fit("x+x", 0.8),
        analysis(["x", "x"], [1, 1], [0, 0]),
        PruningConfig(threshold=0.3, max_prunes=2),
        lambda *_: failed,
        lambda _: analysis(["x"], [1], [1]),
    )
    assert not result.success
    assert result.initial_expression == result.final_expression == "x+x"
    assert result.termination_reason == "refit_failure"
    assert result.steps[0].decision == "refit_failure"
    serialized = result.as_dict()
    assert serialized["steps"][0]["removed_term"] == "x"


def mcts_model(**overrides: object) -> MCTS:
    values: dict[str, object] = {
        "binary": [nd.Add],
        "unary": [nd.Sin],
        "leaf": [],
        "child_num": 4,
        "n_playout": 2,
        "d_playout": 2,
        "max_len": 8,
        "n_iter": 2,
        "sample_num": 64,
        "random_state": 987,
        "eta": 0.999,
        "structural_metric": "none",
        "sobolev_min_valid_samples": 16,
    }
    values.update(overrides)
    return MCTS(**values)


def mcts_trace(model: MCTS, x: np.ndarray) -> tuple[list[tuple[object, ...]], str]:
    X = {"x": x}
    y = x + np.sin(x)
    model.fit(X, y, early_stop=lambda *_: False)
    records = [
        (row["iter"], row.get("reward"), row.get("r2"), row.get("complexity"), row.get("eqtree"))
        for row in model.records
    ]
    return records, str(model.eqtree)


def test_pruning_off_and_max_prunes_zero_strictly_preserve_baseline() -> None:
    x = np.linspace(-2, 2, 64)
    base = mcts_model(sobolev_pruning=False, sobolev_max_prunes=0)
    max_zero = mcts_model(sobolev_pruning=True, sobolev_max_prunes=0)
    assert mcts_trace(base, x) == mcts_trace(max_zero, x)
    assert base.sobolev_evaluator is None and max_zero.sobolev_evaluator is None
    assert max_zero.pruning_candidates_attempted == 0


def test_mcts_duplicate_refit_updates_final_metrics_and_trace(tmp_path) -> None:
    x = np.linspace(-2, 2, 128)
    X = {"x": x}
    model = mcts_model(
        sobolev_pruning=True,
        sobolev_max_prunes=3,
        sobolev_pruning_trace_path=tmp_path / "pruning.jsonl",
        sobolev_detailed_logging=True,
        sobolev_candidate_log_path=tmp_path / "candidates.jsonl",
    )
    model._initialize_sobolev(X)
    model.pareto_front = []
    model.current_iter = 1
    node = Node([nd.Variable("x"), nd.Variable("x")])
    model.set_reward(node, X, 2 * x)
    assert str(node.fitted_expression_before_pruning) == "x + x"
    assert str(node.phi) == "2 * x"
    assert node.pruned_expression == "2 * x"
    assert node.r2 == 1.0
    assert node.base_reward == node.base_reward_before_pruning
    assert node.pruning_result.accepted_prunes == 1
    trace = (tmp_path / "pruning.jsonl").read_text()
    candidate = (tmp_path / "candidates.jsonl").read_text()
    assert '"initial_expression": "x + x"' in trace
    assert '"final_expression": "2 * x"' in trace
    assert '"pruned_expression": "2 * x"' in candidate


def test_mcts_pruning_geometry_reuse_matches_full_reevaluation() -> None:
    x = np.linspace(-2, 2, 128)
    X = {"x": x}

    def run(**kwargs: object) -> Node:
        model = mcts_model(
            sobolev_pruning=True,
            sobolev_max_prunes=3,
            **kwargs,
        )
        model._initialize_sobolev(X)
        model.pareto_front = []
        model.current_iter = 1
        node = Node([nd.Variable("x"), nd.Variable("x")])
        model.set_reward(node, X, 2 * x)
        return node

    full = run(sobolev_evaluator_mode="full")
    reused = run(
        sobolev_evaluator_mode="incremental",
        sobolev_pruning_geometry_reuse=True,
    )
    assert str(full.phi) == str(reused.phi) == "2 * x"
    assert full.r2 == reused.r2 == 1.0
    assert full.base_reward == reused.base_reward
    assert full.pruning_result.accepted_prunes == reused.pruning_result.accepted_prunes == 1
    assert [step.removed_index for step in full.pruning_result.steps] == [
        step.removed_index for step in reused.pruning_result.steps
    ]
    reused_step = reused.pruning_result.steps[0]
    assert reused_step.geometry_reused is True
    assert reused_step.geometry_key_before
    assert reused_step.geometry_key_after
    assert reused_step.geometry_reuse_mode in {
        "parent_child_incremental",
        "candidate_geometry_cache",
        "cached_raw_legacy_numeric_fallback",
    }


def test_sn_elite_prunes_only_configured_reranked_top_k() -> None:
    x = np.linspace(-2, 2, 128)
    X = {"x": x}
    model = mcts_model(
        structural_metric="sobolev",
        sobolev_alpha=0.0,
        sobolev_pruning=True,
        sobolev_max_prunes=2,
        sobolev_selection_mode="elite",
        shortlist_mode="fixed",
        shortlist_size=2,
        prune_elite_k=1,
        sobolev_evaluator_mode="incremental",
        sobolev_pruning_geometry_reuse=True,
        eic_random_state=987,
    )
    model._initialize_sobolev(X)
    model.pareto_front = []
    model.current_iter = 1
    duplicate = Node([nd.Variable("x"), nd.Variable("x")])
    weak = Node([nd.Sin(nd.Variable("x"))])
    model.set_reward(duplicate, X, 2 * x, defer_sobolev=True)
    model.set_reward(weak, X, 2 * x, defer_sobolev=True)
    selected = model._select_elite_candidate([duplicate, weak], X, 2 * x)
    assert model.sobolev_shortlist_evaluation_count == 2
    assert model.sobolev_elite_prune_count == 1
    assert model.pruning_candidates_attempted == 1
    assert sum(node.pruning_result is not None for node in [duplicate, weak]) == 1
    assert selected.structural_rank == 1


def test_sn_prune_and_sn_full_have_distinct_reward_semantics() -> None:
    x = np.linspace(-0.05, 0.05, 128)
    X = {"x": x}
    y = x + np.sin(x)

    def evaluate(model: MCTS) -> Node:
        model._initialize_sobolev(X)
        model.pareto_front = []
        node = Node([nd.Variable("x"), nd.Sin(nd.Variable("x"))])
        model.set_reward(node, X, y)
        return node

    prune_only = evaluate(mcts_model(
        structural_metric="none",
        sobolev_pruning=True,
        sobolev_max_prunes=2,
        eta=1.0,
    ))
    full = evaluate(mcts_model(
        structural_metric="sobolev",
        sobolev_alpha=0.03,
        sobolev_pruning=True,
        sobolev_max_prunes=2,
        eta=1.0,
    ))
    assert prune_only.pruning_result.termination_reason == "base_reward_decrease"
    assert full.pruning_result.termination_reason == "base_reward_decrease"
    assert prune_only.reward == prune_only.base_reward == 1.0
    assert full.sobolev_penalty > 0.99
    assert full.reward < full.base_reward
    assert str(prune_only.phi) == str(full.phi) == "x + sin(x)"


def test_project_expression_printer_uses_nd2py_inverse_trig_names() -> None:
    expression, _ = parse_expression(
        "arcsin(x) + arccos(x) + arctan(x)",
        ("x",),
    )
    text = to_project_expression_string(expression)

    assert "arcsin(" in text
    assert "arccos(" in text
    assert "arctan(" in text
    assert "asin(" not in text
    assert "acos(" not in text
    assert "atan(" not in text
    assert len(nd.parse(text)) > 0
