"""Stage03 train-only Sobolev feedback for the IGSR proposal agent.

This module is deliberately an overlay instead of a modification to
``igsr.method.igsr``.  The completed Stage01/Stage02 experiments therefore
retain byte-for-byte verifiable live source files.  A Stage03-only entry point
installs the overlay in its own child process.

The intervention is intentionally narrow:

* the root prompts remain identical to Base because the empty root has no
  diagnostic;
* later SN proposal prompts receive a short, deterministic summary of the
  parent model's candidate-internal Sobolev novelty;
* no term is deleted, no reward is changed, and MCTS selection is untouched;
* all diagnostics and refits use search-training rows only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from contextvars import ContextVar
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


POLICY_NAME = "train_only_parent_term_sobolev_prompt_feedback_v1"
PROMPT_MARKER = "SOBOLEV STRUCTURE FEEDBACK (search-training rows only)"
DEFAULT_LOW_TERM_LIMIT = 3
DEFAULT_ANCHOR_TERM_LIMIT = 2


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_novelty_rows(
    terms: Sequence[str], diagnostics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw = diagnostics.get("term_novelties")
    if not isinstance(raw, list) or len(raw) != len(terms):
        return []
    rows: list[dict[str, Any]] = []
    for index, (term, value) in enumerate(zip(terms, raw)):
        if value is None:
            continue
        try:
            novelty = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(novelty):
            continue
        rows.append(
            {
                "term_index": index,
                "term": str(term),
                "novelty": novelty,
            }
        )
    return rows


def build_prompt_feedback_payload(
    *,
    target_node_id: Optional[str],
    parent_node_id: Optional[str],
    parent_terms: Sequence[str],
    diagnostics: Optional[Mapping[str, Any]],
    low_term_limit: int = DEFAULT_LOW_TERM_LIMIT,
    anchor_term_limit: int = DEFAULT_ANCHOR_TERM_LIMIT,
) -> dict[str, Any]:
    """Create the deterministic structured ledger and exact prompt text.

    Only the fields represented in ``source_diagnostic_used`` influence the
    text.  Runtime/cache timings from the EIC adapter are intentionally absent.
    """

    if low_term_limit < 1:
        raise ValueError("low_term_limit must be positive")
    if anchor_term_limit < 0:
        raise ValueError("anchor_term_limit must be non-negative")

    terms = [str(term) for term in parent_terms]
    base: dict[str, Any] = {
        "schema_version": 1,
        "policy": POLICY_NAME,
        "target_node_id": None if target_node_id is None else str(target_node_id),
        "source_parent_node_id": (
            None if parent_node_id is None else str(parent_node_id)
        ),
        "parent_terms": terms,
        "parent_terms_sha256": _sha256_json(terms),
        "low_term_limit": int(low_term_limit),
        "anchor_term_limit": int(anchor_term_limit),
        "held_out_rows_consulted": False,
    }

    success = isinstance(diagnostics, Mapping) and diagnostics.get("success") is True
    if not terms:
        reason = "empty_root_or_parent"
    elif not success:
        reason = "parent_diagnostic_unavailable"
    else:
        reason = "no_low_novelty_parent_term"

    threshold: Optional[float] = None
    rows: list[dict[str, Any]] = []
    source_used: dict[str, Any] = {
        "success": bool(success),
        "candidate_geometry_key": None,
        "threshold": None,
        "term_novelties": None,
    }
    if success:
        try:
            threshold = float(diagnostics["threshold"])
        except (KeyError, TypeError, ValueError):
            success = False
            reason = "parent_diagnostic_unavailable"
        if threshold is not None and not math.isfinite(threshold):
            success = False
            reason = "parent_diagnostic_unavailable"
        if success:
            rows = _finite_novelty_rows(terms, diagnostics)
            if not rows:
                success = False
                reason = "parent_diagnostic_unavailable"

    if success:
        assert threshold is not None
        source_used = {
            "success": True,
            "candidate_geometry_key": diagnostics.get("candidate_geometry_key"),
            "threshold": threshold,
            "term_novelties": [
                None if value is None else float(value)
                for value in diagnostics["term_novelties"]
            ],
        }
        low = sorted(
            (row for row in rows if row["novelty"] <= threshold),
            key=lambda row: (row["novelty"], row["term_index"]),
        )[:low_term_limit]
        anchors = sorted(
            (row for row in rows if row["novelty"] > threshold),
            key=lambda row: (-row["novelty"], row["term_index"]),
        )[:anchor_term_limit]
    else:
        low = []
        anchors = []

    injected = bool(low)
    if injected:
        reason = "low_novelty_parent_terms_reported"
        low_text = "; ".join(
            f"`{row['term']}` (nu={row['novelty']:.3f})" for row in low
        )
        anchor_text = ""
        if anchors:
            anchor_text = (
                "\n- High-novelty current anchors: "
                + "; ".join(
                    f"`{row['term']}` (nu={row['novelty']:.3f})"
                    for row in anchors
                )
                + "."
            )
        feedback_text = (
            f"{PROMPT_MARKER}:\n"
            f"- Low candidate-internal novelty at tau={threshold:.3f}: {low_text}."
            f"{anchor_text}\n"
            "Use this only as proposal-diversity guidance: propose structurally "
            "different terms whose value-and-gradient behavior may add a direction "
            "outside the current span. Avoid simple rescalings or algebraic variants "
            "of the low-novelty terms. Keep predictive relevance and the executable "
            "grammar primary; do not delete a current term merely because its "
            "novelty is low."
        )
    else:
        feedback_text = ""

    payload = {
        **base,
        "injected": injected,
        "reason": reason,
        "threshold": threshold,
        "low_terms": low,
        "high_novelty_anchors": anchors,
        "source_diagnostic_used": source_used,
        "source_diagnostic_used_sha256": _sha256_json(source_used),
        "feedback_text": feedback_text,
    }
    payload["feedback_sha256"] = _sha256_json(payload)
    return payload


_ACTIVE_FEEDBACK: ContextVar[Optional[Mapping[str, Any]]] = ContextVar(
    "igsr_stage03_active_sobolev_feedback", default=None
)
_DIAGNOSTIC_CACHE: Dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
_INSTALLED = False


def _parent_node_id(target_node_id: Optional[str]) -> Optional[str]:
    if target_node_id is None:
        return None
    value = str(target_node_id)
    return value.rsplit("_", 1)[0] if "_" in value else None


def _enabled(cfg: Any) -> bool:
    return bool(getattr(cfg.experiment, "sobolev_prompt_feedback_enabled", False))


def _limits(cfg: Any) -> tuple[int, int]:
    return (
        int(
            getattr(
                cfg.experiment,
                "sobolev_prompt_feedback_low_term_limit",
                DEFAULT_LOW_TERM_LIMIT,
            )
        ),
        int(
            getattr(
                cfg.experiment,
                "sobolev_prompt_feedback_anchor_term_limit",
                DEFAULT_ANCHOR_TERM_LIMIT,
            )
        ),
    )


def _train_only_parent_diagnostics(
    core: Any,
    cfg: Any,
    context: Any,
    prepared_dataset: Any,
    terms: Sequence[str],
) -> dict[str, Any]:
    key = (str(context.dataset_identity), tuple(str(term) for term in terms))
    cached = _DIAGNOSTIC_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        matrix = core._strict_term_matrix(
            list(terms), prepared_dataset.data, split_name="feedback_train"
        )
        reg_type, reg_kwargs = core.get_optimization_method(cfg)
        target = np.column_stack(list(prepared_dataset.y.values()))
        fitted = reg_type(**reg_kwargs).fit(matrix, target)
        diagnostics = core._evaluate_sobolev_candidate(
            context,
            terms=list(terms),
            fitted_regressor=fitted,
            n_outputs=len(prepared_dataset.y),
            parent_geometry_key=None,
        )
    except Exception as error:  # fail closed: Base proposal proceeds unchanged
        diagnostics = {
            "success": False,
            "failure_type": "prompt_feedback_train_refit_failure",
            "failure_message": f"{type(error).__name__}: {error}",
        }
    _DIAGNOSTIC_CACHE[key] = diagnostics
    return diagnostics


def install_prompt_feedback_overlay() -> None:
    """Install the Stage03 process-local overlay exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return
    core = importlib.import_module("igsr.method.igsr")
    utils = importlib.import_module("igsr.method.igsr_utils")

    original_prompt = core.prompt_generate_terms
    original_prompt_simple = core.prompt_generate_terms_simple
    original_context = core._llm_cache_context
    original_propose = core.propose_and_prune_once
    original_igsr = core.igsr
    original_to_dict = utils.IterState.to_dict

    def append_feedback(prompt: str) -> str:
        active = _ACTIVE_FEEDBACK.get()
        if not isinstance(active, Mapping) or active.get("injected") is not True:
            return prompt
        text = str(active.get("feedback_text") or "")
        return prompt if not text else f"{prompt}\n\n{text}\n"

    def prompt_generate_terms(*args: Any, **kwargs: Any) -> str:
        return append_feedback(original_prompt(*args, **kwargs))

    def prompt_generate_terms_simple(*args: Any, **kwargs: Any) -> str:
        return append_feedback(original_prompt_simple(*args, **kwargs))

    def llm_cache_context(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_context(*args, **kwargs)
        active = _ACTIVE_FEEDBACK.get()
        if (
            kwargs.get("stage") == "propose"
            and isinstance(active, Mapping)
            and active.get("injected") is True
        ):
            payload["sobolev_prompt_feedback_sha256"] = active[
                "feedback_sha256"
            ]
            payload["sobolev_prompt_feedback_policy"] = POLICY_NAME
        return payload

    def propose_and_prune_once(*args: Any, **kwargs: Any) -> Any:
        cfg = kwargs.get("cfg")
        if cfg is None and len(args) > 4:
            cfg = args[4]
        if cfg is None or not _enabled(cfg):
            result = original_propose(*args, **kwargs)
            setattr(result, "sobolev_prompt_feedback", None)
            return result

        terms = list(kwargs.get("current_terms") or [])
        context = kwargs.get("sobolev_context")
        prepared_dataset = kwargs.get("prepared_dataset")
        target_node_id = kwargs.get("node_id")
        low_limit, anchor_limit = _limits(cfg)
        diagnostics: Optional[Mapping[str, Any]] = None
        if terms and context is not None and prepared_dataset is not None:
            diagnostics = _train_only_parent_diagnostics(
                core, cfg, context, prepared_dataset, terms
            )
        ledger = build_prompt_feedback_payload(
            target_node_id=target_node_id,
            parent_node_id=_parent_node_id(target_node_id),
            parent_terms=terms,
            diagnostics=diagnostics,
            low_term_limit=low_limit,
            anchor_term_limit=anchor_limit,
        )
        token = _ACTIVE_FEEDBACK.set(ledger)
        try:
            result = original_propose(*args, **kwargs)
        finally:
            _ACTIVE_FEEDBACK.reset(token)
        setattr(result, "sobolev_prompt_feedback", ledger)
        logger = kwargs.get("exp_logger")
        if logger is not None:
            logger.info(
                "sobolev_prompt_feedback_finished",
                seed=kwargs.get("seed"),
                node_id=target_node_id,
                feedback=ledger,
            )
        return result

    def iter_state_to_dict(self: Any) -> dict[str, Any]:
        payload = original_to_dict(self)
        payload["sobolev_prompt_feedback"] = getattr(
            self, "sobolev_prompt_feedback", None
        )
        return payload

    def stage03_igsr(cfg: Any, seed: int) -> dict[str, Any]:
        result = original_igsr(cfg, seed)
        if not _enabled(cfg):
            return result
        history = result.get("history_validation") or []
        ledgers = [
            row.get("metadata", {}).get("sobolev_prompt_feedback")
            for row in history
        ]
        if any(not isinstance(row, Mapping) for row in ledgers):
            raise RuntimeError("Stage03 result is missing a prompt-feedback ledger")
        injected = [row for row in ledgers if row.get("injected") is True]
        reasons: dict[str, int] = {}
        for row in ledgers:
            reason = str(row.get("reason") or "missing_reason")
            reasons[reason] = reasons.get(reason, 0) + 1
        summary = {
            "schema_version": 1,
            "policy": POLICY_NAME,
            "feedback_nodes": len(ledgers),
            "feedback_injected_nodes": len(injected),
            "feedback_not_injected_nodes": len(ledgers) - len(injected),
            "feedback_reasons": dict(sorted(reasons.items())),
            "unique_feedback_sha256_count": len(
                {str(row["feedback_sha256"]) for row in injected}
            ),
            "held_out_rows_consulted": False,
        }
        status = dict(result.get("sobolev_run_status") or {})
        status.update(summary)
        status["mode"] = "prompt_feedback"
        result["core_sobolev_mode"] = result.get("sobolev_mode")
        result["sobolev_mode"] = "prompt_feedback"
        result["method_profile"] = "sn_igsr"
        result["sobolev_prompt_feedback_summary"] = summary
        result["sobolev_run_status"] = status
        return result

    core.prompt_generate_terms = prompt_generate_terms
    core.prompt_generate_terms_simple = prompt_generate_terms_simple
    core._llm_cache_context = llm_cache_context
    core.propose_and_prune_once = propose_and_prune_once
    core.igsr = stage03_igsr
    utils.IterState.to_dict = iter_state_to_dict
    _INSTALLED = True


__all__ = [
    "DEFAULT_ANCHOR_TERM_LIMIT",
    "DEFAULT_LOW_TERM_LIMIT",
    "POLICY_NAME",
    "PROMPT_MARKER",
    "build_prompt_feedback_payload",
    "install_prompt_feedback_overlay",
]
