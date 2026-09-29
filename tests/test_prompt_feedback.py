from __future__ import annotations

from igsr.sobolev_novelty.prompt_feedback import (
    POLICY_NAME,
    PROMPT_MARKER,
    build_prompt_feedback_payload,
)


def test_empty_root_never_injects_feedback() -> None:
    payload = build_prompt_feedback_payload(
        target_node_id="node_0_0",
        parent_node_id="node_0",
        parent_terms=[],
        diagnostics=None,
    )
    assert payload["policy"] == POLICY_NAME
    assert payload["injected"] is False
    assert payload["reason"] == "empty_root_or_parent"
    assert payload["feedback_text"] == ""
    assert payload["held_out_rows_consulted"] is False


def test_feedback_is_deterministic_bounded_and_strictly_candidate_internal() -> None:
    diagnostics = {
        "success": True,
        "threshold": 10 ** -0.5,
        "candidate_geometry_key": "geometry-parent",
        "term_novelties": [0.30, 0.02, None, 0.90, 0.20, 0.80],
    }
    kwargs = {
        "target_node_id": "node_0_3_1",
        "parent_node_id": "node_0_3",
        "parent_terms": ["x1", "x2", "x3", "x4", "x5", "x6"],
        "diagnostics": diagnostics,
        "low_term_limit": 2,
        "anchor_term_limit": 1,
    }
    first = build_prompt_feedback_payload(**kwargs)
    second = build_prompt_feedback_payload(**kwargs)
    assert first == second
    assert first["injected"] is True
    assert [row["term"] for row in first["low_terms"]] == ["x2", "x5"]
    assert [row["term"] for row in first["high_novelty_anchors"]] == ["x4"]
    assert PROMPT_MARKER in first["feedback_text"]
    assert "do not delete" in first["feedback_text"]
    assert len(first["feedback_sha256"]) == 64
    assert first["held_out_rows_consulted"] is False


def test_failed_or_high_novelty_parent_fails_closed_without_prompt_change() -> None:
    failed = build_prompt_feedback_payload(
        target_node_id="node_0_1_0",
        parent_node_id="node_0_1",
        parent_terms=["x1"],
        diagnostics={"success": False, "failure_type": "evaluation_failure"},
    )
    high = build_prompt_feedback_payload(
        target_node_id="node_0_1_0",
        parent_node_id="node_0_1",
        parent_terms=["x1", "x2"],
        diagnostics={
            "success": True,
            "threshold": 10 ** -0.5,
            "candidate_geometry_key": "g",
            "term_novelties": [0.8, 0.9],
        },
    )
    assert failed["injected"] is False
    assert failed["reason"] == "parent_diagnostic_unavailable"
    assert high["injected"] is False
    assert high["reason"] == "no_low_novelty_parent_term"
    assert failed["feedback_text"] == high["feedback_text"] == ""
