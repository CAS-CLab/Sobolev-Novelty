import json
import hashlib
import os
import re
import time
import traceback
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from omegaconf import OmegaConf

from igsr.agent.agent import AgentResponse, LiteLLMAgent
from igsr.agent.completion_cache import CompletionCacheConflictError, deterministic_request_seed
from igsr.compute_profiler import ComputeProfiler
from igsr.config_typing import MainConfig
from igsr.cpu_mutex import cpu_mutex
from igsr.dataset import DataBundle
from igsr.dataset.util import get_dataset
from igsr.formula_tracker import FormulaTrackerMultiMetric
from igsr.method.igsr_utils import (
    DesignMatrix,
    IterState,
    OptimizationMethod,
    PreparedDataset,
    _get_intercept,
    add_to_history,
    allowed_numpy_function_names,
    check_early_stop,
    check_token_budget_and_maybe_break,
    check_wallclock_budget_and_maybe_break,
    compute_raw_feature_importance,
    evaluate,
    get_iter_or_node_id,
    get_optimization_method,
    get_total_iters_for_print,
    ols_influence,
    postprocessor_generate_terms,
    postprocessor_pruning,
    prepare_dataset,
    pretty_equations,
)
from igsr.retry_decor import retryable
from igsr.sobolev_novelty import (
    AdapterFailureType,
    CrossfitStability,
    NoveltyDiagnostics,
    SobolevNoveltyAdapter,
    rank_pareto_siblings,
    rank_prune_candidates,
    rank_siblings,
    repeated_crossfit_stability,
    select_pareto_sibling_for_expansion,
    select_sibling_for_expansion,
)
from igsr.tree_search.problem import Problem
from igsr.tree_search.session import SearchSession
from igsr.logging_setup import get_logger, safe_cfg_yaml


LLM_CACHE_PROTOCOL = "igsr-paired-completion-v2"
PARETO_CROSSFIT_POLICY = (
    "strict_train_crossfit_guarded_sobolev_pareto_sibling_replacement_v1"
)


@dataclass(frozen=True)
class _SobolevDiagnosticContext:
    """Seed-local, immutable context shared by every node in one search."""

    adapter: SobolevNoveltyAdapter
    feature_names: Tuple[str, ...]
    X_train: np.ndarray
    geometry_indices: np.ndarray
    dataset_identity: str
    feature_policy: str = "all"
    excluded_feature_names: Tuple[str, ...] = ()
    initialization_failure: Optional[str] = None


@dataclass
class _PruneRefitOutcome:
    terms: List[str]
    regressor: Optional[OptimizationMethod]
    mse_validation: Any
    mse_validation_total: float
    metrics_validation: Dict[str, float]
    equation: str
    diagnostics: Dict[str, Any]
    audit: Dict[str, Any]


def _optional_cfg_value(section: Any, name: str, default: Any = None) -> Any:
    """Read an optional dynamic OmegaConf/dataclass field."""

    try:
        return getattr(section, name, default)
    except (AttributeError, KeyError):
        return default


def _llm_completion_kwargs(
    cfg: MainConfig,
    api_key: str,
    *,
    cache_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build LiteLLM kwargs, including optional local OpenAI-compatible routing."""

    if cfg.llm.kind == "azure_ai":
        kwargs: Dict[str, Any] = {"api_base": cfg.llm.endpoint, "api_key": api_key}
    elif cfg.llm.kind == "azure":
        kwargs = {
            "api_base": cfg.llm.endpoint,
            "api_version": cfg.llm.api_version,
            "api_key": api_key,
        }
    elif cfg.llm.kind == "openai":
        kwargs = {"api_key": api_key}
        endpoint = _optional_cfg_value(cfg.llm, "endpoint")
        if endpoint:
            kwargs["api_base"] = endpoint
    else:
        raise ValueError(f"Invalid LLM kind: {cfg.llm.kind}")

    for field_name in ("temperature", "request_timeout", "max_tokens"):
        value = _optional_cfg_value(cfg.llm, field_name)
        if value is not None:
            kwargs[field_name] = value
    extra_body = _optional_cfg_value(cfg.llm, "extra_body")
    if extra_body is not None:
        if OmegaConf.is_config(extra_body):
            extra_body = OmegaConf.to_container(extra_body, resolve=True)
        if not isinstance(extra_body, Mapping):
            raise ValueError("llm.extra_body must be a mapping or null")
        kwargs["extra_body"] = dict(extra_body)
    if bool(_optional_cfg_value(cfg.llm, "deterministic_request_seeds", False)):
        if cache_context is None:
            raise ValueError("cache_context is required for deterministic local request seeds")
        # Step zero is resolved here; LiteLLMAgent re-derives the same identity
        # with an explicit agent_step for every self-correction turn.
        kwargs["seed"] = deterministic_request_seed(cache_context)
    return kwargs


def _llm_cache_context(
    cfg: MainConfig,
    data_bundle: DataBundle,
    *,
    seed: int,
    stage: str,
    iter_num: Optional[int],
    node_id: Optional[str],
    sibling_slot: Optional[int],
    retry_ordinal: int = 0,
) -> Dict[str, Any]:
    """Build provider-private identity for one logical IGSR LLM request."""

    if not isinstance(retry_ordinal, int) or isinstance(retry_ordinal, bool) or retry_ordinal < 0:
        raise ValueError("retry_ordinal must be a non-negative integer")

    settings = data_bundle.data_settings if isinstance(data_bundle.data_settings, Mapping) else {}
    split_settings = settings.get("split", {})
    if not isinstance(split_settings, Mapping):
        split_settings = {}
    task_identity = (
        settings.get("instance_id")
        or settings.get("problem_id")
        or f"{cfg.dataset.name}:{data_bundle.name}"
    )
    # Provider/cache randomness is keyed by the search-only identity.  The
    # complete split identity still travels in the final result for provenance,
    # but held-out hashes must never perturb an LLM request or pairing tape.
    split_identity = (
        split_settings.get("search_split_identity_sha256")
        or split_settings.get("split_identity_sha256")
        or {
            "dataset_name": str(cfg.dataset.name),
            "dataset_seed": int(seed),
        }
    )
    model_revision = _optional_cfg_value(cfg.llm, "model_revision") or _optional_cfg_value(
        cfg.llm, "model_version"
    )
    return {
        "protocol": LLM_CACHE_PROTOCOL,
        "request_seed_policy": "sha256_context_retry_plus_agent_step_v1",
        "task_identity": str(task_identity),
        "split_identity": split_identity,
        "seed": int(seed),
        "stage": str(stage),
        "iteration": int(iter_num) if iter_num is not None else None,
        "node_id": str(node_id) if node_id is not None else None,
        "sibling_slot": int(sibling_slot) if sibling_slot is not None else None,
        "retry_ordinal": retry_ordinal,
        "model_revision": str(model_revision) if model_revision is not None else None,
    }


def _require_agent_parsed_output(result: AgentResponse, *, stage: str, request_id: str) -> Any:
    """Return validated postprocessor output or raise an actionable stage error."""

    if not result.success:
        detail = result.error or result.message or "unknown agent failure"
        raise RuntimeError(f"{stage} agent failed for {request_id}: {detail}")
    postprocessor_output = result.postprocessor_output
    if postprocessor_output is None:
        raise RuntimeError(f"{stage} agent returned no postprocessor output for {request_id}")
    if not postprocessor_output.success:
        detail = postprocessor_output.feedback or "postprocessor rejected the response"
        raise RuntimeError(f"{stage} postprocessor failed for {request_id}: {detail}")
    if postprocessor_output.parsed_output is None:
        raise RuntimeError(f"{stage} postprocessor returned no parsed output for {request_id}")
    return postprocessor_output.parsed_output


def _sobolev_mode(cfg: MainConfig) -> str:
    mode = str(_optional_cfg_value(cfg.experiment, "sobolev_mode", "off")).strip().lower()
    if mode not in {
        "off",
        "diagnostic",
        "sibling_rerank",
        "prune_refit",
        "pareto_crossfit",
    }:
        raise ValueError(
            f"Unknown experiment.sobolev_mode={mode!r}; supported values are "
            "'off', 'diagnostic', 'sibling_rerank', 'prune_refit', and "
            "'pareto_crossfit'"
        )
    return mode


def _mapping_from_dynamic_cfg(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"experiment.{field_name} must be a mapping or null")
    return dict(value)


def _initialize_sobolev_diagnostics(
    cfg: MainConfig,
    *,
    seed: int,
    data_bundle: DataBundle,
    prepared_dataset: PreparedDataset,
) -> Optional[_SobolevDiagnosticContext]:
    """Create one adapter and one fixed train geometry for a complete seed run."""

    if _sobolev_mode(cfg) == "off":
        return None

    eic_src_root = _optional_cfg_value(cfg.experiment, "sobolev_eic_src_root")
    if not isinstance(eic_src_root, str) or not eic_src_root.strip():
        raise ValueError(
            "experiment.sobolev_eic_src_root must be explicitly set when "
            "experiment.sobolev_mode enables Sobolev diagnostics or reranking"
        )
    geometry_sample_size = _optional_cfg_value(cfg.experiment, "geometry_sample_size", 64)
    if geometry_sample_size is not None:
        geometry_sample_size = int(geometry_sample_size)
        if geometry_sample_size < 1:
            raise ValueError("experiment.geometry_sample_size must be positive or null")
    geometry_seed = int(_optional_cfg_value(cfg.experiment, "geometry_seed", 20260731))
    override_field = "sobolev_eic_config_overrides"
    config_overrides = _mapping_from_dynamic_cfg(
        _optional_cfg_value(cfg.experiment, override_field, {}), override_field
    )
    # The run-level fields are authoritative because geometry is selected once
    # here rather than independently inside candidate evaluation.
    config_overrides["geometry_sample_size"] = geometry_sample_size
    config_overrides["geometry_seed"] = geometry_seed
    adapter = SobolevNoveltyAdapter(
        eic_src_root=eic_src_root,
        config_overrides=config_overrides,
    )

    all_feature_names = tuple(prepared_dataset.data)
    all_X_train = np.column_stack(
        [np.asarray(prepared_dataset.data[name], dtype=float) for name in all_feature_names]
    )
    feature_policy = str(
        _optional_cfg_value(cfg.experiment, "sobolev_feature_policy", "all")
    ).strip().lower()
    if feature_policy == "all":
        selected_indices = tuple(range(len(all_feature_names)))
    elif feature_policy == "nonconstant_train_v1":
        input_tolerance = float(
            _optional_cfg_value(
                cfg.experiment,
                "constant_input_scale_tolerance",
                config_overrides.get("input_scale_tolerance", 1.0e-12),
            )
        )
        if not np.isfinite(input_tolerance) or input_tolerance <= 0.0:
            raise ValueError("experiment.constant_input_scale_tolerance must be positive")

        def stable_std(values: np.ndarray) -> float:
            values = np.asarray(values, dtype=float)
            peak = float(np.max(np.abs(values)))
            if peak == 0.0:
                return 0.0
            return peak * float(np.std(values / peak, ddof=0))

        selected_indices = tuple(
            index
            for index in range(all_X_train.shape[1])
            if stable_std(all_X_train[:, index]) > input_tolerance
        )
    else:
        raise ValueError(
            f"Unknown experiment.sobolev_feature_policy={feature_policy!r}"
        )
    if selected_indices:
        feature_names = tuple(all_feature_names[index] for index in selected_indices)
        X_train = all_X_train[:, selected_indices]
    else:
        # Retain a well-shaped context so candidate diagnostics fail closed with
        # an explicit run-initialization reason below.
        feature_names = all_feature_names
        X_train = all_X_train
    excluded_feature_names = tuple(
        name for name in all_feature_names if name not in set(feature_names)
    )
    y_train = np.column_stack(
        [np.asarray(prepared_dataset.y[name], dtype=float) for name in prepared_dataset.y]
    )
    task_identity = json.dumps(
        {
            "bundle": str(data_bundle.name),
            "dataset_kind": str(_optional_cfg_value(cfg.dataset, "kind", "")),
            "dataset_name": str(_optional_cfg_value(cfg.dataset, "name", "")),
            "problem": str(_optional_cfg_value(cfg.dataset, "problem", "")),
            "targets": list(prepared_dataset.y),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        if not selected_indices:
            raise ValueError("No nonconstant training input remains for Sobolev geometry")
        fixed_geometry = adapter.select_fixed_geometry(
            X=X_train,
            y=y_train,
            task_identity=task_identity,
            run_seed=seed,
            geometry_seed=geometry_seed,
            sample_size=geometry_sample_size,
        )
        geometry_indices = fixed_geometry.indices
        dataset_identity = fixed_geometry.dataset_identity
        initialization_failure = None
    except Exception as error:
        # Dependency/geometry failures remain observational.  Every candidate
        # receives a max-penalty diagnostic, while Base search proceeds intact.
        geometry_indices = np.empty(0, dtype=np.int64)
        dataset_identity = f"igsr-task={task_identity}|run-seed={int(seed)}|geometry-unavailable"
        initialization_failure = f"{type(error).__name__}: {error}"

    X_train.setflags(write=False)
    return _SobolevDiagnosticContext(
        adapter=adapter,
        feature_names=feature_names,
        X_train=X_train,
        geometry_indices=geometry_indices,
        dataset_identity=dataset_identity,
        feature_policy=feature_policy,
        excluded_feature_names=excluded_feature_names,
        initialization_failure=initialization_failure,
    )


def _failed_sobolev_diagnostics(
    context: _SobolevDiagnosticContext,
    failure_type: AdapterFailureType,
    message: str,
    parent_geometry_key: Optional[str],
) -> Dict[str, Any]:
    return NoveltyDiagnostics(
        success=False,
        penalty=1.0,
        geometry_indices=context.geometry_indices.astype(int).tolist(),
        dataset_identity=context.dataset_identity,
        parent_geometry_key=parent_geometry_key,
        failure_type=failure_type,
        failure_message=message,
    ).as_dict()


def _evaluate_sobolev_candidate(
    context: _SobolevDiagnosticContext,
    *,
    terms: List[str],
    fitted_regressor: Optional[OptimizationMethod],
    n_outputs: int,
    parent_geometry_key: Optional[str],
) -> Dict[str, Any]:
    """Evaluate diagnostics fail-closed; never mutate or score search state."""

    if context.initialization_failure is not None:
        return _failed_sobolev_diagnostics(
            context,
            AdapterFailureType.DEPENDENCY_LOAD_FAILURE,
            f"Sobolev run initialization failed: {context.initialization_failure}",
            parent_geometry_key,
        )
    if n_outputs != 1:
        return _failed_sobolev_diagnostics(
            context,
            AdapterFailureType.INVALID_INPUT,
            f"Diagnostic-only Sobolev adapter currently supports one output; got {n_outputs}",
            parent_geometry_key,
        )
    if not terms or fitted_regressor is None:
        return _failed_sobolev_diagnostics(
            context,
            AdapterFailureType.NO_ACTIVE_TERMS,
            "No surviving fitted terms are available for Sobolev diagnostics",
            parent_geometry_key,
        )

    coefficients = np.asarray(fitted_regressor.coef_, dtype=float)
    if coefficients.ndim == 2 and coefficients.shape[0] == 1:
        coefficients = coefficients[0]
    coefficients = coefficients.ravel()
    intercept_values = _get_intercept(fitted_regressor)
    intercept = None if intercept_values is None else float(np.asarray(intercept_values).ravel()[0])
    try:
        return context.adapter.evaluate(
            terms=terms,
            coefficients=coefficients,
            intercept=intercept,
            feature_names=context.feature_names,
            X=context.X_train,
            geometry_indices=context.geometry_indices,
            dataset_identity=context.dataset_identity,
            parent_geometry_key=parent_geometry_key,
        ).as_dict()
    except Exception as error:
        return _failed_sobolev_diagnostics(
            context,
            AdapterFailureType.EVALUATOR_FAILURE,
            f"Sobolev diagnostic raised {type(error).__name__}: {error}",
            parent_geometry_key,
        )


def _candidate_sobolev_geometry_key(state: IterState) -> Optional[str]:
    diagnostics = state.sobolev_diagnostics
    if not isinstance(diagnostics, Mapping):
        return None
    value = diagnostics.get("candidate_geometry_key")
    return str(value) if value else None


def _pareto_crossfit_random_state(
    cfg: MainConfig,
    context: _SobolevDiagnosticContext,
) -> int:
    """Derive one task/seed-local fold seed without touching search RNG state."""

    payload = json.dumps(
        {
            "policy": PARETO_CROSSFIT_POLICY,
            "base_seed": int(
                _optional_cfg_value(cfg.experiment, "sobolev_crossfit_seed", 20260807)
            ),
            "dataset_identity": context.dataset_identity,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def _failed_crossfit_stability(
    *,
    cfg: MainConfig,
    random_state: int,
    n_rows: int,
    n_features: int,
    error: Exception,
) -> CrossfitStability:
    return CrossfitStability(
        success=False,
        n_rows=int(n_rows),
        n_features=int(n_features),
        n_splits=int(_optional_cfg_value(cfg.experiment, "sobolev_crossfit_splits", 3)),
        n_repeats=int(_optional_cfg_value(cfg.experiment, "sobolev_crossfit_repeats", 3)),
        random_state=int(random_state),
        ridge_alpha=float(
            _optional_cfg_value(cfg.experiment, "sobolev_crossfit_ridge_alpha", 1.0e-8)
        ),
        fit_intercept=bool(
            _optional_cfg_value(cfg.experiment, "sobolev_crossfit_fit_intercept", True)
        ),
        ridge_solver=str(
            _optional_cfg_value(cfg.experiment, "sobolev_crossfit_ridge_solver", "svd")
        ),
        nmse_values=(),
        nmse_mean=None,
        nmse_median=None,
        nmse_worst=None,
        nmse_std=None,
        held_out_rows_consulted=False,
        failure_type=type(error).__name__,
        failure_message=str(error),
    )


def _evaluate_train_crossfit_candidate(
    cfg: MainConfig,
    prepared_dataset: PreparedDataset,
    terms: List[str],
    *,
    random_state: int,
) -> CrossfitStability:
    """Evaluate a candidate on search-train rows only, failing closed."""

    n_rows = len(next(iter(prepared_dataset.data.values())))
    try:
        design = DesignMatrix.from_terms(terms, prepared_dataset.data)
        if design.errors or design.term_names != terms:
            detail = "; ".join(design.errors) or "the evaluated term list changed"
            raise ValueError(
                "Every cross-fit term must evaluate on search train; "
                f"refusing a partial design matrix ({detail})"
            )
        if not np.isfinite(design.phi).all():
            raise ValueError("Cross-fit design matrix contains NaN or infinity")
        targets = np.column_stack(list(prepared_dataset.y.values()))
        return repeated_crossfit_stability(
            design.phi,
            targets,
            n_splits=int(
                _optional_cfg_value(cfg.experiment, "sobolev_crossfit_splits", 3)
            ),
            n_repeats=int(
                _optional_cfg_value(cfg.experiment, "sobolev_crossfit_repeats", 3)
            ),
            random_state=random_state,
            ridge_alpha=float(
                _optional_cfg_value(
                    cfg.experiment, "sobolev_crossfit_ridge_alpha", 1.0e-8
                )
            ),
            fit_intercept=bool(
                _optional_cfg_value(
                    cfg.experiment, "sobolev_crossfit_fit_intercept", True
                )
            ),
            ridge_solver=str(
                _optional_cfg_value(
                    cfg.experiment, "sobolev_crossfit_ridge_solver", "svd"
                )
            ),
        )
    except Exception as error:
        return _failed_crossfit_stability(
            cfg=cfg,
            random_state=random_state,
            n_rows=n_rows,
            n_features=len(terms),
            error=error,
        )


def _summarize_sobolev_run(
    *,
    mode: str,
    context: Optional[_SobolevDiagnosticContext],
    validation_history: Iterable[Mapping[str, Any]],
    expansion_decisions: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build a fail-closed, JSON-friendly health summary for one seed run.

    A missing or malformed candidate diagnostic is counted as a failure.  This
    keeps the search-level max-penalty fallback separate from the experiment
    validity decision: a formal scheduler can reject systematic Sobolev failure
    without requiring every individual candidate to be supported.
    """

    enabled = context is not None
    diagnostic_success = 0
    diagnostic_failure = 0
    failure_types: Dict[str, int] = {}

    if enabled:
        for history_entry in validation_history:
            metadata = history_entry.get("metadata")
            diagnostic = metadata.get("sobolev_diagnostics") if isinstance(metadata, Mapping) else None
            if isinstance(diagnostic, Mapping) and diagnostic.get("success") is True:
                diagnostic_success += 1
                continue

            diagnostic_failure += 1
            raw_failure_type = (
                diagnostic.get("failure_type", "unspecified_failure")
                if isinstance(diagnostic, Mapping)
                else "missing_diagnostic"
            )
            if isinstance(raw_failure_type, AdapterFailureType):
                failure_type = raw_failure_type.value
            else:
                failure_type = str(raw_failure_type or "unspecified_failure")
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

    decisions = list(expansion_decisions)
    novelty_applied_count = sum(
        1
        for decision in decisions
        if isinstance(decision, Mapping) and decision.get("novelty_applied") is True
    )
    pruning_attempts = 0
    accepted_prunes = 0
    pruning_nodes = 0
    if enabled:
        for history_entry in validation_history:
            metadata = history_entry.get("metadata")
            pruning = metadata.get("sobolev_pruning") if isinstance(metadata, Mapping) else None
            if not isinstance(pruning, Mapping):
                continue
            pruning_nodes += 1
            trials = pruning.get("trials")
            if isinstance(trials, list):
                pruning_attempts += len(trials)
                accepted_prunes += sum(
                    isinstance(trial, Mapping) and trial.get("accepted") is True
                    for trial in trials
                )
    summary = {
        "mode": mode,
        "enabled": enabled,
        "initialization_attempted": enabled,
        "initialization_success": (
            None if context is None else context.initialization_failure is None
        ),
        "initialization_failure": (
            None if context is None else context.initialization_failure
        ),
        "dataset_identity": None if context is None else context.dataset_identity,
        "geometry_sample_count": 0 if context is None else int(len(context.geometry_indices)),
        "diagnostics_total": diagnostic_success + diagnostic_failure,
        "diagnostics_success": diagnostic_success,
        "diagnostics_failure": diagnostic_failure,
        "diagnostic_failure_types": dict(sorted(failure_types.items())),
        "expansion_decisions_total": len(decisions),
        "novelty_applied_count": novelty_applied_count,
    }
    if context is not None:
        summary.update(
            {
                "feature_policy": context.feature_policy,
                "feature_count": len(context.feature_names),
                "excluded_feature_count": len(context.excluded_feature_names),
                "feature_names_sha256": hashlib.sha256(
                    "\0".join(context.feature_names).encode("utf-8")
                ).hexdigest(),
                "pruning_nodes": pruning_nodes,
                "pruning_attempts": pruning_attempts,
                "accepted_prunes": accepted_prunes,
            }
        )
    return summary

# ================================================================================
# Prompt building
# ================================================================================


def _format_history(history: List[Dict[str, Any]]) -> str:
    """Pretty-print history for prompt injection."""
    if not history:
        return "- (no prior rounds) -"
    return "\n".join(
        f"Round {h['round']}:  KEEP={h['keep']}  |  DROP={h['drop']}  |  "
        f"MSE before pruning={h['MSE_before (total)']:.6f} (per-output={[float(m) for m in list(h['MSE_before (per-output)'])]}) |  "
        f"MSE after pruning={h['MSE_after (total)']:.6f} (per-output={[float(m) for m in list(h['MSE_after (per-output)'])]})"
        for h in history
    )


def _format_feature_importance(
    feature_importances: List[Tuple[str, float]],
    total_features: int,
) -> str:
    """Format feature importance as a markdown table for prompt injection."""
    n_shown = len(feature_importances)
    header = f"Top {n_shown} of {total_features} raw input features by importance (standardized Ridge |coefficient|, averaged across targets):"
    rows = "\n".join(
        f"| {rank} | {name} | {imp:.4f} |"
        for rank, (name, imp) in enumerate(feature_importances, 1)
    )
    return (
        f"{header}\n"
        f"| Rank | Feature | Importance |\n"
        f"|------|---------|------------|\n"
        f"{rows}\n"
        f"Higher importance = more predictive (linear effects). Nonlinear interactions may also matter."
    )


def _parse_seq_feature_name(feature_name: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (side, idx) for sequence one-hot features like seq_neg20__A or seq_20__T."""
    neg_match = re.fullmatch(r"seq_neg(\d+)__.+", feature_name)
    if neg_match:
        return "neg", int(neg_match.group(1))
    pos_match = re.fullmatch(r"seq_(\d+)__.+", feature_name)
    if pos_match:
        return "pos", int(pos_match.group(1))
    return None, None


def _iter_preview_feature_lines(
    data: Dict[str, np.ndarray],
    preview_N_elements: int,
    collapse_seq_feats: bool,
) -> Iterable[str]:
    """
    Yield feature preview lines, collapsing middle sequence positions to a single "...".
    """
    if not collapse_seq_feats:
        for feat_name, feat_val in data.items():
            yield f"{feat_name}[:{preview_N_elements}]:\n{feat_val[:preview_N_elements]}"
        return

    seq_indices: Dict[str, set[int]] = {"neg": set(), "pos": set()}
    for feat_name in data.keys():
        side, idx = _parse_seq_feature_name(feat_name)
        if side is not None and idx is not None:
            seq_indices[side].add(idx)

    keep_indices: Dict[str, set[int]] = {}
    for side, indices in seq_indices.items():
        if len(indices) <= 2:
            keep_indices[side] = set(indices)
        else:
            keep_indices[side] = {min(indices), max(indices)}

    inserted_gap_for_side: Dict[str, bool] = {"neg": False, "pos": False}
    for feat_name, feat_val in data.items():
        side, idx = _parse_seq_feature_name(feat_name)
        if side is not None and idx is not None:
            if idx not in keep_indices[side]:
                if not inserted_gap_for_side[side]:
                    yield "..."
                    inserted_gap_for_side[side] = True
                continue
        yield f"{feat_name}[:{preview_N_elements}]:\n{feat_val[:preview_N_elements]}"


def _atomic_term_prompt_guidance(cfg: MainConfig, *, bullet: str = "*") -> str:
    """Describe the shared one-line/one-additive-basis contract when enabled."""

    if not bool(_optional_cfg_value(cfg.experiment, "require_expand_mul_atomic_terms", False)):
        return ""
    return (
        f"\n{bullet} Each line MUST remain exactly one additive basis after "
        "multiplication is distributed over addition (`expand_mul`). Write outer "
        "sums and products of sums as separate lines: `x1 + x2` and "
        "`x1 * (x2 + 1)` are invalid single lines. Additions protected inside a "
        "function, a denominator, or a power remain one basis, so "
        "`np.log(x1 + 1)`, `1 / (x1 + 1)`, and `(x1 + 1) ** 2` are valid."
    )


def prompt_generate_terms(
    cfg: MainConfig,
    data_bundle: DataBundle,
    data: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    history: List[Dict[str, Any]],
    current_terms: List[str],
    terms_sentinel: str,
    current_equation: Optional[str] = None,
    feature_importance_text: Optional[str] = None,
) -> str:
    preview_N_elements = getattr(cfg.experiment, "preview_N_elements", 50)
    collapse_seq_feats = getattr(cfg.experiment, "collapse_seq_feats", False)
    snippet_parts = ["Input data:\n"]

    for line in _iter_preview_feature_lines(data, preview_N_elements, collapse_seq_feats):
        snippet_parts.append(line)
    snippet_parts.append("\nTarget variables:\n")
    for y_name, y_val in y.items():
        snippet_parts.append(f"{y_name}[:{preview_N_elements}]:\n{y_val[:preview_N_elements]}")
    snippet = "\n".join(snippet_parts)

    if cfg.experiment.history_enabled:
        hist_txt = _format_history(history)
    else:
        hist_txt = "History is disabled."

    focus_on_seq_feats = getattr(cfg.experiment, "focus_on_seq_feats", False)
    focus_on_seq_feats_text = """

========================================================
**NOTE FOR THIS SPECIFIC EXPERIMENT**
See if you can find "complex DNA sequence features" to further improve the predictions.
However, balance this with predictive accuracy - if simpler signal features increase accuracy, do not lose them.
========================================================
"""

    current_equation = current_equation if current_equation is not None else "N/A"
    allowed_functions = ", ".join(allowed_numpy_function_names(cfg))
    atomic_term_guidance = _atomic_term_prompt_guidance(cfg)
    prompt = f"""You are an automated assistant for proposing linear terms for the equations in a symbolic regression pipeline.

Your proposed terms will be:
1. Concatenated with the current candidate terms.
2. Sent to a LLM term pruner agent that will use various computed signals to decide which terms to keep and which to drop.

# Instructions:
Given the information below, propose candidate terms using only this executable grammar:
* Operands may be exact input feature names shown under Input data or real numeric constants. Target variable names are outputs, not input features, and MUST NOT appear in any term.
* The only arithmetic operators are `+`, `-`, `*`, `/`, and `**`.
* The only allowed functions are: {allowed_functions}.
* Every function call MUST use the exact `np.` prefix, for example `np.sin(x1)`. Bare calls such as `sin(x1)` and other spellings such as `numpy.sin(x1)` are invalid.
* Return at least one nonempty term. No other names, functions, operators, or Python syntax are allowed.
{atomic_term_guidance}
Make use of the dataset and problem description to propose relevant terms.
Make sure to use the learnings from the history of previous rounds.

Return these between triple backticks and one term on each line.
The first backticks must be prepended with {terms_sentinel}

Example output:

{terms_sentinel}
```
x1
x2**2
np.sin(x3)
```

NOTES:
* Propose around {cfg.experiment.terms_per_round} terms, generally not too many unless this is the first round.
* If this *is* the first round, propose around {cfg.experiment.first_round_n_candidates} terms.
* You MUST propose at least one term between the TERMS backticks.

DATASET AND PROBLEM DESCRIPTION
------------------------------
{data_bundle.data_dictionary}
{"" if feature_importance_text is None else chr(10) + "RAW FEATURE IMPORTANCE" + chr(10) + "------------------------------" + chr(10) + feature_importance_text + chr(10)}
CURRENT TERMS:
------------------------------
{current_terms}

# Current equation:
{current_equation}

HISTORY OF KEEP/DROP DECISIONS
------------------------------
{hist_txt}

=========
The input data and target variable(s) preview:
{snippet}
"""
    if focus_on_seq_feats:
        prompt += focus_on_seq_feats_text
    if cfg.experiment.print_prompts:
        print(f"prompt_generate_terms:\n==============\n{prompt}\n==============\n")
    return prompt


def prompt_generate_terms_simple(
    cfg: MainConfig,
    data_bundle: DataBundle,
    data: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    history: List[Dict[str, Any]],
    current_terms: List[str],
    terms_sentinel: str,
    current_equation: Optional[str] = None,
    feature_importance_text: Optional[str] = None,
) -> str:
    """
    A simplified version of the generate-terms prompt.
    Keeps history but removes verbose previews and extra notes.
    """
    if cfg.experiment.history_enabled:
        hist_txt = _format_history(history)
    else:
        hist_txt = "History is disabled."
    current_equation = current_equation if current_equation is not None else "N/A"
    allowed_functions = ", ".join(allowed_numpy_function_names(cfg))
    atomic_term_guidance = _atomic_term_prompt_guidance(cfg, bullet="-")
    prompt = f"""Propose candidate numpy expressions (terms) for linear symbolic regression.

Return your proposal between triple backticks, one term per line, and prepend the block with {terms_sentinel}.

Guidelines:
- Propose around {cfg.experiment.terms_per_round} terms (if first round: ~{cfg.experiment.first_round_n_candidates}).
- Use only exact input feature names and real numeric constants; target variable names MUST NOT appear in a term.
- The only arithmetic operators are `+`, `-`, `*`, `/`, and `**`.
- The only allowed functions are: {allowed_functions}.
- Every function call MUST use the exact `np.` prefix. Valid: `np.sin(x1)`. Invalid: `sin(x1)` or `numpy.sin(x1)`.
- No other names, functions, operators, or Python syntax are allowed.
{atomic_term_guidance}
- Use the dataset/problem description and the history to stay relevant.
- Return at least one nonempty term.

Example:
{terms_sentinel}
```
x1
x2**2
np.sin(x3)
```

DATASET AND PROBLEM DESCRIPTION
------------------------------
{data_bundle.data_dictionary}
{"" if feature_importance_text is None else chr(10) + "RAW FEATURE IMPORTANCE" + chr(10) + "------------------------------" + chr(10) + feature_importance_text + chr(10)}
CURRENT TERMS
------------------------------
{current_terms}

Current equation:
{current_equation}

HISTORY
------------------------------
{hist_txt}
"""
    if cfg.experiment.print_prompts:
        print(f"prompt_generate_terms_simple:\n==============\n{prompt}\n==============\n")
    return prompt


def prompt_pruning(
    cfg: MainConfig,
    data_bundle: DataBundle,
    dfs: Dict[str, pd.DataFrame],
    mse: list[float],
    mse_total: float,
    history: List[Dict[str, Any]],
    current_terms: List[str],
    decision_sentinel: str,
    current_equation: str,
    feature_importance_text: Optional[str] = None,
) -> str:
    tbls = ""
    for y_name, df in dfs.items():
        tbls += f"{y_name}:\n" + df.to_markdown(index=False) + "\n\n"

    if cfg.experiment.history_enabled:
        hist_txt = _format_history(history)
    else:
        hist_txt = "History is disabled."

    if cfg.experiment.influence_feedback:
        if getattr(cfg.experiment, "refit_aware", False):
            influence_definition = (
                "**Influence definition (refit-aware)**\n\n"
                "Δ_k^{refit} is the increase in validation mean-squared-error (MSE) when the k-th term is "
                "removed and the model is refit:\n\n"
                "    Δ_k^{refit} = MSE_val( w^{(-k)} ) - MSE_val( w ).\n\n"
                "Here w^{(-k)} denotes the least-squares (or ridge) solution after dropping term k.\n"
            )
        else:
            influence_definition = (
                "**Influence definition (no refit)**\n\n"
                "Δ_k is the increase in mean-squared-error (MSE) if the k-th weight is deleted while all other "
                "weights stay fixed. For OLS this is\n\n"
                "    Δ_k = (w_k² / n) · ∑ φ_k(x_i)²  (always ≥ 0).\n"
            )
        # Add note that it isn't strictly "always ≥ 0" as values for validation rather than train are used.
        influence_definition += "\n\n* Note that the influence values are computed on the validation (rather than training) set, and thus may not always be ≥ 0."

        input_you_receive = f"""* A table (or dictionary/json representation) where each row has:

| field     | meaning                                                                   |
|-----------|---------------------------------------------------------------------------|
| term      | Name of the symbolic basis function φ_k(x) (e.g. "x1", "sin(x2)", ...).   |
| weight    | Fitted scalar coefficient w_k obtained by ordinary least squares (OLS).   |
| influence | Influence score for term k (see definition below).                        |

* The validation set MSE.

{influence_definition}

Hence:

* If influence is large ⇒ the term is important (its removal hurts the loss a lot).  
* If influence ≈ 0 ⇒ the term is useless (its removal makes no noticeable difference).
"""
    else:
        input_you_receive = """* A table (or dictionary/json representation) where each row has:

| field     | meaning                                                                   |
|-----------|---------------------------------------------------------------------------|
| term      | Name of the symbolic basis function φ_k(x) (e.g. "x1", "sin(x2)", ...).   |
| weight    | Fitted scalar coefficient w_k obtained by ordinary least squares (OLS).   |

* The validation set MSE.
"""

    if cfg.experiment.influence_feedback:
        example_input = """y_1:
| term   | weight | influence |
| ------ | ------ | --------- |
| x1     | 3.00   | 12.21     |
| x2     | -1.96  | 5.14      |
| x1**2  | 0.53   | 0.91      |
| sin    | 0.93   | 0.52      |
| cos    | -0.05  | 0.0009    |

y_2:
| term   | weight | influence |
| ------ | ------ | --------- |
| x1     | 3.00   | 12.21     |
| x2     | -1.96  | 5.14      |

MSE (per-output): [0.217, 0.145]
MSE overall: 0.181
"""
    else:
        example_input = """y_1:
| term   | weight |
| ------ | ------ |
| x1     | 3.00   |
| x2     | -1.96  |
| x1**2  | 0.53   |
| sin    | 0.93   |
| cos    | -0.05  |

y_2:
| term   | weight |
| ------ | ------ |
| x1     | 3.00   |
| x2     | -1.96  |

MSE (per-output): [0.217, 0.145]
MSE overall: 0.181
"""

    extra_notes = ""
    if cfg.experiment.influence_feedback:
        extra_notes += """\n* (!) You should also consider BOTH the weights and the influence of the terms."""
    if cfg.experiment.keep_n_terms is not None:
        extra_notes += (
            f"""\n* (!) You MUST keep {cfg.experiment.keep_n_terms} terms at most, to keep the model interpretable."""
        )

    prompt = f"""You are an equation-pruning assistant for symbolic regression.

========================================================
INPUT YOU RECEIVE
========================================================

{input_you_receive}

========================================================
YOUR TASK
========================================================
1. **Inspect every row**.
2. **Decide “keep” or “drop”** for each term using the rule:

* Use the heuristic: "Δₖ ≈ 0 → drop", "large Δₖ → keep" and your own judgement.

3. **Return** a python dictionary after "{decision_sentinel}" with exactly the two keys

{decision_sentinel}
```
{{
  "keep":  ["term_a", "term_b", ...],
  "drop":  ["term_c", "term_d", ...]
}}
````

Place each term name in either **keep** or **drop** — never both, never neither.

**IMPORTANT:**
* Make use of the dataset and problem description to make the best decision.
* Make sure to use the learnings from the history of previous rounds.
* (!) You must consider the generalization beyond the validation set and make decisions accordingly.{extra_notes}

────────────────────────────────────────────────────────
CONVENTIONS & NOTES
────────────────────────────────────────────────────────

# Important notes
* Treat terms independently; no need to refit or update weights.
* Note that everything was evaluated on the validation set to avoid overfitting.
* If there are multiple outputs (targets), you will see multiple tables, one for each target.
* Use all the information available, but keep in mind that you must only return one keep/drop decision even if there are multiple outputs.
* Keep only the most important terms for each output.

# Output format
* Feel free to comment briefly (≤ 30 chars) about each decision, but keep the python dictionary in the right format.
* The dictionary MUST be provided between triple backticks, otherwise it cannot be parsed.
* It must be prepended with "{decision_sentinel}", otherwise it cannot be parsed.

────────────────────────────────────────────────────────
EXAMPLE
────────────────────────────────────────────────────────
# INPUT TABLE(S):

{example_input}

# Output:

{decision_sentinel}
```
{{
  "keep": ["x1", "x2", "x1**2", "sin"],
  "drop": ["cos"]
}}
```

That's it — perform the keep/drop decision based on the information provided.


========================================================
DATASET AND PROBLEM DESCRIPTION
========================================================
{data_bundle.data_dictionary}
{"" if feature_importance_text is None else chr(10) + "========================================================" + chr(10) + "RAW FEATURE IMPORTANCE" + chr(10) + "========================================================" + chr(10) + feature_importance_text + chr(10)}

========================================================
CURRENT TERMS
========================================================
{current_terms}

# Current equation:
{current_equation}

========================================================
HISTORY OF KEEP/DROP DECISIONS
========================================================
{hist_txt}


========================================================
INPUT YOU RECEIVE
========================================================

INPUT TABLE(S):

{tbls}

MSE (per-output): {[float(m) for m in list(mse)]}
MSE overall: {mse_total:.6f}
"""
    if cfg.experiment.print_prompts:
        print(f"prompt_pruning:\n==============\n{prompt}\n==============\n")
    return prompt


def prompt_pruning_simple(
    cfg: MainConfig,
    data_bundle: DataBundle,
    dfs: Dict[str, pd.DataFrame],
    mse: list[float],
    mse_total: float,
    history: List[Dict[str, Any]],
    current_terms: List[str],
    decision_sentinel: str,
    current_equation: str,
    feature_importance_text: Optional[str] = None,
) -> str:
    """
    A simplified version of the pruning prompt.
    Keeps history, presents compact instructions and the essential tables/metrics.
    """
    tbls = ""
    for y_name, df in dfs.items():
        tbls += f"{y_name}:\n" + df.to_markdown(index=False) + "\n\n"
    if cfg.experiment.history_enabled:
        hist_txt = _format_history(history)
    else:
        hist_txt = "History is disabled."
    extra_notes = ""
    if cfg.experiment.influence_feedback:
        extra_notes += "Consider both weights and influence if provided. "
    if cfg.experiment.keep_n_terms is not None:
        extra_notes += f"You MUST keep at most {cfg.experiment.keep_n_terms} terms. "
    fi_section = ""
    if feature_importance_text is not None:
        fi_section = f"""
RAW FEATURE IMPORTANCE
------------------------------
{feature_importance_text}
"""
    prompt = f"""Decide which terms to KEEP or DROP for the linear symbolic model.

What you have:
- Table(s) per target with columns: term, weight{", influence" if cfg.experiment.influence_feedback else ""}.
- Validation MSE (per-output and overall).
- Dataset/problem context and the history of previous rounds.

Instructions:
- Treat each term independently; use weights{"+influence" if cfg.experiment.influence_feedback else ""}, MSE, history and context.
- Favor generalization beyond validation.
- {extra_notes.strip()}

Return a python dict after "{decision_sentinel}" exactly in this format:
{decision_sentinel}
```
{{
  "keep": ["term_a", "term_b"],
  "drop": ["term_c", "term_d"]
}}
```

CURRENT TERMS
------------------------------
{current_terms}

Current equation:
{current_equation}
{fi_section}
HISTORY
------------------------------
{hist_txt}

INPUT TABLE(S)
------------------------------
{tbls}

MSE (per-output): {[float(m) for m in list(mse)]}
MSE overall: {mse_total:.6f}
"""
    if cfg.experiment.print_prompts:
        print(f"prompt_pruning_simple:\n==============\n{prompt}\n==============\n")
    return prompt


# ================================================================================
# Main components
# ================================================================================


def _append_new_terms_to_json(
    cfg: MainConfig,
    seed: int,
    iter_or_node_id: str,
    new_terms: List[str],
    exp_logger: structlog.stdlib.BoundLogger,
) -> None:
    """
    Append the new_terms for a given iter_or_node_id into a per-seed JSON file
    located next to the configured log_file. If log_file is not set, write in CWD.
    This function is a no-op if cfg.experiment.save_new_terms is False.
    """
    if not getattr(cfg.experiment, "save_new_terms", False):
        return
    try:
        base_log = getattr(cfg.logging, "log_file", None)
        dir_path = os.path.dirname(base_log) if base_log else "."
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"new_terms.seed_{seed}.json")

        # Load existing
        data: Dict[str, Any]
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception:
                # If the file is corrupted or empty, start fresh
                data = {"seed": seed, "entries": []}
        else:
            data = {"seed": seed, "entries": []}

        # Ensure structure
        if not isinstance(data, dict):
            data = {"seed": seed, "entries": []}
        if "entries" not in data or not isinstance(data["entries"], list):
            data["entries"] = []
        if "seed" not in data:
            data["seed"] = seed

        # Append new entry
        data["entries"].append(
            {
                "id": str(iter_or_node_id),
                "new_terms": list(new_terms),
            }
        )

        # Write back
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        exp_logger.info(
            "new_terms_saved",
            seed=seed,
            iter_or_node_id=iter_or_node_id,
            file_path=file_path,
            num_terms=len(new_terms),
        )
    except Exception as e:
        exp_logger.error("new_terms_save_error", seed=seed, iter_or_node_id=iter_or_node_id, error=e)


def train_and_evaluate(
    cfg: MainConfig,
    terms: List[str],
    data: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    data_eval: Dict[str, np.ndarray],
    y_eval: Dict[str, np.ndarray],
) -> Tuple[
    OptimizationMethod,
    np.ndarray,  # per-output MSE
    float,  # total MSE
    Dict[str, float],  # All metrics per-output
]:
    Φ_final = _strict_term_matrix(terms, data, split_name="train")
    reg, reg_kwargs = get_optimization_method(cfg)
    reg = reg(**reg_kwargs).fit(Φ_final, np.column_stack(list(y.values())))
    mse_eval, mse_eval_total, all_metrics_eval = evaluate(terms, data_eval, y_eval, reg)
    return reg, mse_eval, mse_eval_total, all_metrics_eval


def _strict_term_matrix(
    terms: List[str],
    data: Dict[str, np.ndarray],
    *,
    split_name: str,
) -> np.ndarray:
    """Build a matrix without silently dropping a fitted/exported term."""

    design_matrix = DesignMatrix.from_terms(terms, data)
    if design_matrix.errors or design_matrix.term_names != terms:
        errors = "; ".join(design_matrix.errors) or "the evaluated term list changed"
        raise ValueError(
            f"Every fitted term must evaluate on {split_name}; refusing a partial design matrix ({errors})"
        )
    if not np.isfinite(design_matrix.phi).all():
        raise ValueError(f"The fitted expression produced NaN or infinity on {split_name}")
    return design_matrix.phi


def _fit_xgb_predictor(
    terms: List[str],
    prepared_dataset: PreparedDataset,
) -> Any:
    """Fit the existing frozen XGBoost predictor on train only."""
    try:
        import xgboost as xgb  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "use_xgb_predictor=True requires the 'xgboost' (and scikit-learn) packages to be installed."
        ) from e

    Φ_train = _strict_term_matrix(terms, prepared_dataset.data, split_name="train")
    y_train = np.column_stack(list(prepared_dataset.y.values()))

    # Create a default XGBRegressor; wrap for multi-target if needed
    base_reg = xgb.XGBRegressor()
    if y_train.ndim == 2 and y_train.shape[1] > 1:
        reg = MultiOutputRegressor(base_reg)
    else:
        reg = base_reg

    with cpu_mutex():
        reg = reg.fit(Φ_train, y_train)
    return reg


def _train_xgb_and_eval_validation(
    terms: List[str],
    prepared_dataset: PreparedDataset,
) -> Dict[str, float]:
    """Fit XGBoost on train and evaluate validation only during search."""

    if not terms:
        return _failed_report_metrics()
    reg = _fit_xgb_predictor(terms, prepared_dataset)
    _, _, metrics_val = evaluate(
        terms,
        prepared_dataset.data_validation,
        prepared_dataset.y_validation,
        reg,
    )
    return metrics_val


def _failed_report_metrics() -> Dict[str, float]:
    """Finite fail-closed metrics for an invalid selected report expression.

    ``NMSE=1e12`` is the frozen maximum penalty.  ``R²=1-NMSE`` preserves
    the usual identity used by the benchmark while keeping JSON/CSV output
    finite and aggregation-safe.
    """

    return {
        "mse": 1.0e12,
        "rmse": 1.0e6,
        "r2": 1.0 - 1.0e12,
        "nrmse": 1.0e6,
        "nmse": 1.0e12,
        "accuracy_tol": 0.0,
        "accuracy_tol_max": 0.0,
    }


def _fit_selected_predictor(
    cfg: MainConfig,
    terms: List[str],
    prepared_dataset: PreparedDataset,
) -> Any:
    """Refit the validation-selected term set on train with the frozen optimizer."""

    if cfg.experiment.use_xgb_predictor:
        return _fit_xgb_predictor(terms, prepared_dataset)
    Φ_train = _strict_term_matrix(terms, prepared_dataset.data, split_name="train")
    reg_cls, reg_kwargs = get_optimization_method(cfg)
    return reg_cls(**reg_kwargs).fit(
        Φ_train,
        np.column_stack(list(prepared_dataset.y.values())),
    )


def _evaluate_selected_report_split(
    *,
    terms: List[str],
    data: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    reg: Any,
    split_name: str,
    exp_logger: structlog.stdlib.BoundLogger,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Evaluate one held-out split without changing or aborting the search result.

    The policy is deliberately fail-closed: if any selected term is undefined,
    non-finite, shape-invalid, or otherwise unscorable on this split, the whole
    split receives the finite frozen worst metrics.  Terms are never dropped and a
    different node is never selected as a fallback.
    """

    try:
        _, _, metrics = evaluate(terms, data, y, reg)
        core_metrics = (
            "mse",
            "rmse",
            "r2",
            "nrmse",
            "nmse",
            "accuracy_tol",
            "accuracy_tol_max",
        )
        if any(not np.isfinite(metrics[name]) for name in core_metrics):
            raise ValueError("report evaluation returned a non-finite core metric")
    except Exception as exc:
        status = {
            "status": "failed_closed",
            "split": split_name,
            "policy": "selected_expression_all_terms_or_finite_max_penalty_v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exp_logger.warning("selected_report_evaluation_failed_closed", **status)
        return _failed_report_metrics(), status
    return metrics, {
        "status": "ok",
        "split": split_name,
        "policy": "selected_expression_all_terms_or_finite_max_penalty_v1",
    }


def _refit_and_evaluate_selected_for_reporting(
    cfg: MainConfig,
    selected_validation: Mapping[str, Any],
    prepared_dataset: PreparedDataset,
    exp_logger: structlog.stdlib.BoundLogger,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]], Dict[str, Any]]:
    """After validation selection, refit once and evaluate report splits only."""

    metadata = selected_validation["metadata"]
    terms = list(metadata.get("terms_after", []))
    if not terms:
        status = {
            "policy": "selected_expression_all_terms_or_finite_max_penalty_v1",
            "refit": {"status": "no_terms"},
            "test": {"status": "failed_closed", "reason": "no_terms"},
        }
        if prepared_dataset.data_ood_test is not None:
            status["ood"] = {"status": "failed_closed", "reason": "no_terms"}
        return (
            _failed_report_metrics(),
            _failed_report_metrics() if prepared_dataset.data_ood_test is not None else None,
            status,
        )

    reg = _fit_selected_predictor(cfg, terms, prepared_dataset)
    test_metrics, test_status = _evaluate_selected_report_split(
        terms=terms,
        data=prepared_dataset.data_test,
        y=prepared_dataset.y_test,
        reg=reg,
        split_name="test",
        exp_logger=exp_logger,
    )
    if prepared_dataset.data_ood_test is not None:
        ood_metrics, ood_status = _evaluate_selected_report_split(
            terms=terms,
            data=prepared_dataset.data_ood_test,
            y=prepared_dataset.y_ood_test,
            reg=reg,
            split_name="ood_test",
            exp_logger=exp_logger,
        )
    else:
        ood_metrics = None
        ood_status = None
    return test_metrics, ood_metrics, {
        "policy": "selected_expression_all_terms_or_finite_max_penalty_v1",
        "refit": {"status": "ok", "optimizer": type(reg).__name__},
        "test": test_status,
        "ood": ood_status,
    }


def _with_report_status_fields(
    metrics: Mapping[str, float],
    status: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach disclosure fields without changing the numeric metric keys."""

    failed = status.get("status") != "ok"
    return {
        **dict(metrics),
        "evaluation_status": str(status.get("status", "unknown")),
        "failure_policy_applied": bool(failed),
        "failure_type": (
            str(status.get("error_type") or status.get("reason") or "unknown")
            if failed
            else None
        ),
        "failure_policy": "selected_expression_all_terms_or_finite_max_penalty_v1",
    }


def _validation_guarded_sobolev_prune(
    *,
    cfg: MainConfig,
    context: _SobolevDiagnosticContext,
    terms: List[str],
    regressor: Optional[OptimizationMethod],
    mse_validation: Any,
    mse_validation_total: float,
    metrics_validation: Dict[str, float],
    equation: str,
    prepared_dataset: PreparedDataset,
    parent_geometry_key: Optional[str],
) -> _PruneRefitOutcome:
    """Delete low-novelty terms only after a fresh validation-safe refit.

    Ranking uses the coefficient-aware independent contribution from the
    Sobolev report.  Acceptance observes search train/validation only; held-out
    report rows are structurally unavailable to this function.
    """

    diagnostics = _evaluate_sobolev_candidate(
        context,
        terms=terms,
        fitted_regressor=regressor,
        n_outputs=len(prepared_dataset.y),
        parent_geometry_key=parent_geometry_key,
    )
    max_prunes = int(_optional_cfg_value(cfg.experiment, "sobolev_max_prunes", 1))
    max_trials = int(_optional_cfg_value(cfg.experiment, "sobolev_prune_max_trials", 1))
    acceptance_tolerance = float(
        _optional_cfg_value(
            cfg.experiment,
            "sobolev_prune_acceptance_nmse_tolerance",
            0.0,
        )
    )
    if max_prunes < 0 or max_trials < 1:
        raise ValueError("Sobolev prune counts must satisfy max_prunes>=0 and max_trials>=1")
    if not np.isfinite(acceptance_tolerance) or acceptance_tolerance < 0.0:
        raise ValueError("Sobolev prune acceptance NMSE tolerance must be finite and non-negative")

    audit: Dict[str, Any] = {
        "schema_version": 1,
        "policy": "coefficient_aware_low_novelty_delete_refit_validation_nmse_v1",
        "ranking": "ascending_abs_coefficient_times_novelty_times_signature_norm",
        "acceptance_metric": "search_validation_nmse",
        "acceptance_tolerance": acceptance_tolerance,
        "max_prunes": max_prunes,
        "max_trials_per_step": max_trials,
        "held_out_rows_consulted": False,
        "initial_terms": list(terms),
        "initial_nmse": float(metrics_validation.get("nmse", float("inf"))),
        "initial_diagnostic_success": diagnostics.get("success") is True,
        "initial_candidate_geometry_key": diagnostics.get("candidate_geometry_key"),
        "trials": [],
        "accepted_prunes": 0,
        "termination_reason": "",
    }
    current_terms = list(terms)
    current_regressor = regressor
    current_mse = mse_validation
    current_mse_total = float(mse_validation_total)
    current_metrics = dict(metrics_validation)
    current_equation = str(equation)
    current_diagnostics = diagnostics

    if max_prunes == 0:
        audit["termination_reason"] = "max_prunes_zero"
    elif len(current_terms) <= 1:
        audit["termination_reason"] = "single_term_remaining"
    elif current_diagnostics.get("success") is not True:
        audit["termination_reason"] = "initial_diagnostic_failure"
    else:
        for prune_step in range(1, max_prunes + 1):
            ranked = rank_prune_candidates(current_terms, current_diagnostics)
            if not ranked:
                audit["termination_reason"] = "no_low_novelty_terms"
                break
            accepted_this_step = False
            for rank, candidate in enumerate(ranked[:max_trials], start=1):
                if len(current_terms) <= 1:
                    break
                trial_terms = [
                    term for index, term in enumerate(current_terms)
                    if index != candidate.index
                ]
                trial: Dict[str, Any] = {
                    "step": prune_step,
                    "rank": rank,
                    "candidate": candidate.as_dict(),
                    "terms_before": list(current_terms),
                    "terms_after": list(trial_terms),
                    "nmse_before": float(current_metrics["nmse"]),
                    "nmse_after": None,
                    "accepted": False,
                    "decision": "",
                }
                try:
                    (
                        trial_regressor,
                        trial_mse,
                        trial_mse_total,
                        trial_metrics,
                    ) = train_and_evaluate(
                        cfg,
                        trial_terms,
                        prepared_dataset.data,
                        prepared_dataset.y,
                        prepared_dataset.data_validation,
                        prepared_dataset.y_validation,
                    )
                    trial_nmse = float(trial_metrics["nmse"])
                    trial["nmse_after"] = trial_nmse
                    trial["nmse_delta"] = trial_nmse - float(current_metrics["nmse"])
                    if trial_nmse <= float(current_metrics["nmse"]) + acceptance_tolerance:
                        trial["accepted"] = True
                        trial["decision"] = "accepted_validation_guard"
                        accepted_this_step = True
                        audit["accepted_prunes"] += 1
                        previous_geometry_key = current_diagnostics.get(
                            "candidate_geometry_key"
                        )
                        current_terms = trial_terms
                        current_regressor = trial_regressor
                        current_mse = trial_mse
                        current_mse_total = float(trial_mse_total)
                        current_metrics = dict(trial_metrics)
                        current_equation = pretty_equations(
                            w=trial_regressor.coef_.T,
                            terms=current_terms,
                            y=prepared_dataset.y,
                            intercept=_get_intercept(trial_regressor),
                        )
                        current_diagnostics = _evaluate_sobolev_candidate(
                            context,
                            terms=current_terms,
                            fitted_regressor=current_regressor,
                            n_outputs=len(prepared_dataset.y),
                            parent_geometry_key=(
                                str(previous_geometry_key)
                                if previous_geometry_key
                                else None
                            ),
                        )
                        trial["post_accept_diagnostic_success"] = (
                            current_diagnostics.get("success") is True
                        )
                        trial["post_accept_candidate_geometry_key"] = (
                            current_diagnostics.get("candidate_geometry_key")
                        )
                    else:
                        trial["decision"] = "rejected_validation_nmse_increase"
                except Exception as error:
                    trial["decision"] = "rejected_refit_failure"
                    trial["error_type"] = type(error).__name__
                    trial["error"] = str(error)
                audit["trials"].append(trial)
                if accepted_this_step:
                    break
            if not accepted_this_step:
                audit["termination_reason"] = "all_ranked_trials_rejected"
                break
            if current_diagnostics.get("success") is not True:
                audit["termination_reason"] = "post_accept_diagnostic_failure"
                break
        else:
            audit["termination_reason"] = "max_prunes_reached"

    if not audit["termination_reason"]:
        audit["termination_reason"] = "max_prunes_reached"
    audit["final_terms"] = list(current_terms)
    audit["final_nmse"] = float(current_metrics.get("nmse", float("inf")))
    audit["final_diagnostic_success"] = current_diagnostics.get("success") is True
    audit["final_candidate_geometry_key"] = current_diagnostics.get(
        "candidate_geometry_key"
    )
    return _PruneRefitOutcome(
        terms=current_terms,
        regressor=current_regressor,
        mse_validation=current_mse,
        mse_validation_total=current_mse_total,
        metrics_validation=current_metrics,
        equation=current_equation,
        diagnostics=current_diagnostics,
        audit=audit,
    )


def propose_and_prune_once(
    current_terms: List[str],
    history: List[Dict[str, Any]],
    prepared_dataset: PreparedDataset,
    # ---
    cfg: MainConfig,
    data_bundle: DataBundle,
    exp_logger: structlog.stdlib.BoundLogger,
    seed: int,
    # --
    azure_model_key: str,
    iter_num: Optional[int] = None,
    node_id: Optional[str] = None,
    # ---
    compute_profiler: ComputeProfiler = None,
    feature_importance_text: Optional[str] = None,
    sobolev_context: Optional[_SobolevDiagnosticContext] = None,
    parent_sobolev_geometry_key: Optional[str] = None,
    sibling_slot: Optional[int] = None,
) -> IterState:
    iter_or_node_id = get_iter_or_node_id(cfg, iter_num, node_id)
    total_iters_str = get_total_iters_for_print(cfg)
    stage_name = "propose"

    TERMS_SENTINEL = "TERMS:"
    agent_generate_terms_sentinels = [TERMS_SENTINEL]
    task_generate_terms = (
        prompt_generate_terms_simple if cfg.experiment.simplified_prompts else prompt_generate_terms
    )(
        cfg,
        data_bundle,
        data=prepared_dataset.data,
        y=prepared_dataset.y,
        history=history,
        current_terms=current_terms,
        terms_sentinel=TERMS_SENTINEL,
        current_equation=history[-1]["equation_after"] if history else None,
        feature_importance_text=feature_importance_text,
    )

    def on_retry(exc: Exception, tries_left: int):
        exp_logger.info("retry_attempt", seed=seed, error=exc, tries_left=tries_left)

    allowed_functions = ", ".join(allowed_numpy_function_names(cfg))
    atomic_term_guidance = _atomic_term_prompt_guidance(cfg)
    correct_format_generate_terms = dedent(f"""
        Reminder of correct output format:

        {TERMS_SENTINEL}
        ```
        term1
        ...
        termN
        ```

        Submit at least one nonempty term. Each term may contain only exact input
        feature names, real numeric constants, and the operators +, -, *, /, **.
        Target variable names MUST NOT appear in a term.

        The only allowed functions are: {allowed_functions}. Every
        function call MUST use the exact `np.` prefix. Valid: np.sin(P),
        np.exp(t), np.log(P + 1). Invalid: sin(P), exp(t), log(P + 1),
        numpy.sin(P). No other names, functions, operators, or Python syntax
        are allowed.
        {atomic_term_guidance}
        """)

    retry_ordinal = -1

    @retryable(attempts=cfg.experiment.retry_attempts, delay=0, on_retry=on_retry)
    def _():
        nonlocal retry_ordinal
        retry_ordinal += 1
        cache_context = _llm_cache_context(
            cfg,
            data_bundle,
            seed=seed,
            stage=stage_name,
            iter_num=iter_num,
            node_id=node_id,
            sibling_slot=sibling_slot,
            retry_ordinal=retry_ordinal,
        )
        completion_kwargs = _llm_completion_kwargs(
            cfg,
            azure_model_key,
            cache_context=cache_context,
        )
        agent_generate_terms = LiteLLMAgent(
            task_description=task_generate_terms,
            system_prompt="",  # Only use the task description.
            logger=exp_logger,
            max_steps=cfg.experiment.max_agent_steps,
            model=f"{cfg.llm.kind}/{cfg.llm.deployment}",
            completion_kwargs=completion_kwargs,
            task_complete_sentinels=agent_generate_terms_sentinels,
            postprocessor=postprocessor_generate_terms,
            postprocessor_kwargs={
                "iter_num": iter_num,
                "node_id": node_id,
                "cfg": cfg,
                "data_bundle": data_bundle,
                "seed": seed,
                "terms_sentinel": TERMS_SENTINEL,
                "stage_name": stage_name,
                "current_terms": current_terms,
                # ---
                "prepared_dataset": prepared_dataset,  # To check for invalid values.
                # ---
                "correct_format_generate_terms": correct_format_generate_terms,
            },
            reminder=correct_format_generate_terms,
            cache_context=cache_context,
        )

        try:
            time_start = time.time()

            result = agent_generate_terms.run()

            time_end = time.time()
            compute_profiler.accumulate("wall_clock_llm_calls", time_end - time_start)
            for usage_entry in agent_generate_terms.usage_history:
                if "completion_tokens" in usage_entry:
                    compute_profiler.accumulate("output_tokens", usage_entry["completion_tokens"])
                if "prompt_tokens" in usage_entry:
                    compute_profiler.accumulate("input_tokens", usage_entry["prompt_tokens"])
            compute_profiler.accumulate("number_llm_calls", agent_generate_terms.num_llm_calls)

            new_terms = _require_agent_parsed_output(
                result,
                stage=stage_name,
                request_id=str(iter_or_node_id),
            )
            exp_logger.info(
                f"{stage_name}_agent_run_finished",
                seed=seed,
                stage_name=stage_name,
                iter_num=iter_num,
                node_id=node_id,
                terms=new_terms,
                conversation_history=agent_generate_terms.conversation_history,
                usage_history=agent_generate_terms.usage_history,
                completion_cache_accesses=agent_generate_terms.completion_cache.access_log,
            )
            print(
                f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - "
                f"Stage {stage_name} - New terms:\n{new_terms}"
            )
            return new_terms
        except Exception as e:
            if isinstance(e, CompletionCacheConflictError):
                raise RuntimeError(
                    f"{stage_name} completion cache conflict for {iter_or_node_id}; "
                    "record mode is fail-closed and will not overwrite a frozen response"
                ) from e
            exp_logger.error(
                f"{stage_name}_agent_run_error",
                seed=seed,
                stage_name=stage_name,
                iter_num=iter_or_node_id,
                node_id=node_id,
                error=e,
            )
            traceback.print_exc()
            print(f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - " f"Stage {stage_name} - Error: {e}")
            exp_logger.info(
                f"{stage_name}_agent_run_error",
                seed=seed,
                stage_name=stage_name,
                iter_num=iter_num,
                node_id=node_id,
                error=e,
            )
            raise

    new_terms = _()

    # Optionally save new terms for this iteration/node.
    _append_new_terms_to_json(cfg, seed, iter_or_node_id, new_terms, exp_logger)

    # Concatenate terms, unique & order-stable:
    candidate_terms = list(dict.fromkeys(current_terms + new_terms))
    print(f"Candidate terms:\n{candidate_terms}")
    # OLS + influence:
    dm = DesignMatrix.from_terms(candidate_terms, prepared_dataset.data)
    dm_validation = DesignMatrix.from_terms(candidate_terms, prepared_dataset.data_validation)
    influence_compute_results = ols_influence(
        cfg,
        dm.phi,
        np.column_stack(list(prepared_dataset.y.values())),
        dm_validation.phi,
        np.column_stack(list(prepared_dataset.y_validation.values())),
    )
    mse_before_validation, mse_before_validation_total, all_metrics_before_validation = evaluate(
        dm.term_names, prepared_dataset.data_validation, prepared_dataset.y_validation, influence_compute_results.reg
    )
    print(f"New candidate terms: {new_terms}")
    print(f"MSE before pruning = {mse_before_validation_total:.6f} -- per-output: {list(mse_before_validation)}")
    equation_before = pretty_equations(
        w=influence_compute_results.weights, terms=dm.term_names, y=prepared_dataset.y,
        intercept=influence_compute_results.intercept,
    )
    print(equation_before)

    # Construct the dataframes for preview.
    dfs = dict()
    for i, y_name in enumerate(prepared_dataset.y.keys()):
        if cfg.experiment.influence_feedback:
            df = pd.DataFrame(
                {
                    "term": dm.term_names,
                    "weight": influence_compute_results.weights[:, i],
                    "influence": influence_compute_results.delta_validation[:, i],
                }  # Validation!
            )
        else:
            df = pd.DataFrame(
                {
                    "term": dm.term_names,
                    "weight": influence_compute_results.weights[:, i],
                }  # Validation!
            )
        dfs[y_name] = df
    print(f"dfs:\n{dfs}")

    stage_name = "prune"
    PRUNING_SENTINEL = "DECISION:"
    agent_pruning_sentinels = [PRUNING_SENTINEL]

    # Algorithmic pruning option: keep top-K by influence, drop the rest.
    if cfg.experiment.algorithmic_pruning is True:
        delta_val = influence_compute_results.delta_validation
        if delta_val.ndim != 2:
            raise ValueError(f"delta_val.ndim {delta_val.ndim} must be 2 in algorithmic pruning")
        # Take worst across dimensions - other alternatives like mean are possible.
        aggregated_influence = np.max(delta_val, axis=1)

        n_keep_cfg = cfg.experiment.keep_n_terms
        n_keep = max(0, min(int(n_keep_cfg), len(dm.term_names)))

        sorted_idx = np.argsort(aggregated_influence)[::-1]
        selected_idx = set(sorted_idx[:n_keep].tolist())
        keep = [t for i, t in enumerate(dm.term_names) if i in selected_idx]
        drop = [t for i, t in enumerate(dm.term_names) if i not in selected_idx]

        exp_logger.info(
            f"{stage_name}_algorithmic_decision",
            seed=seed,
            stage_name=stage_name,
            iter_num=iter_num,
            node_id=node_id,
            n_keep=n_keep,
            keep=keep,
            drop=drop,
        )
        print(
            f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - "
            f"Stage {stage_name} (algorithmic) - Keep: {keep}, Drop: {drop}"
        )
    else:
        task_prompt_pruning = (prompt_pruning_simple if cfg.experiment.simplified_prompts else prompt_pruning)(
            cfg,
            data_bundle,
            dfs=dfs,
            mse=list(influence_compute_results.mse_validation),
            mse_total=influence_compute_results.mse_validation_total,
            history=history,
            current_terms=dm.term_names,
            decision_sentinel=PRUNING_SENTINEL,
            current_equation=equation_before,
            feature_importance_text=feature_importance_text,
        )

        correct_format_pruning = dedent(f"""
            Reminder of correct output format:

            {PRUNING_SENTINEL}
            ```
            {{
                "keep": [term1, term2, ...],
                "drop": [term1, term2, ...],
            }}
            ```
            """)
        retry_ordinal = -1

        @retryable(attempts=cfg.experiment.retry_attempts, delay=0, on_retry=on_retry)
        def _():
            nonlocal retry_ordinal
            retry_ordinal += 1
            cache_context = _llm_cache_context(
                cfg,
                data_bundle,
                seed=seed,
                stage=stage_name,
                iter_num=iter_num,
                node_id=node_id,
                sibling_slot=sibling_slot,
                retry_ordinal=retry_ordinal,
            )
            completion_kwargs = _llm_completion_kwargs(
                cfg,
                azure_model_key,
                cache_context=cache_context,
            )
            agent_pruning = LiteLLMAgent(
                task_description=task_prompt_pruning,
                system_prompt="",  # Only use the task description.
                logger=exp_logger,
                max_steps=cfg.experiment.max_agent_steps,
                model=f"{cfg.llm.kind}/{cfg.llm.deployment}",
                completion_kwargs=completion_kwargs,
                task_complete_sentinels=agent_pruning_sentinels,
                postprocessor=postprocessor_pruning,
                postprocessor_kwargs={
                    "iter_num": iter_num,
                    "node_id": node_id,
                    "cfg": cfg,
                    "data_bundle": data_bundle,
                    "seed": seed,
                    "pruning_sentinel": PRUNING_SENTINEL,
                    "stage_name": stage_name,
                    "current_terms": dm.term_names,
                    # ---
                    # "data": prepared_dataset.data,  # To check for invalid values.
                    # ---
                    "correct_format_pruning": correct_format_pruning,
                },
                reminder=correct_format_pruning,
                cache_context=cache_context,
            )
            try:
                time_start = time.time()

                result = agent_pruning.run()

                time_end = time.time()
                compute_profiler.accumulate("wall_clock_llm_calls", time_end - time_start)
                for usage_entry in agent_pruning.usage_history:
                    if "completion_tokens" in usage_entry:
                        compute_profiler.accumulate("output_tokens", usage_entry["completion_tokens"])
                    if "prompt_tokens" in usage_entry:
                        compute_profiler.accumulate("input_tokens", usage_entry["prompt_tokens"])
                compute_profiler.accumulate("number_llm_calls", agent_pruning.num_llm_calls)

                keep, drop = _require_agent_parsed_output(
                    result,
                    stage=stage_name,
                    request_id=str(iter_or_node_id),
                )
                exp_logger.info(
                    f"{stage_name}_agent_run_finished",
                    seed=seed,
                    stage_name=stage_name,
                    iter_num=iter_num,
                    keep=keep,
                    drop=drop,
                    conversation_history=agent_pruning.conversation_history,
                    usage_history=agent_pruning.usage_history,
                    completion_cache_accesses=agent_pruning.completion_cache.access_log,
                )
                print(
                    f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - "
                    f"Stage {stage_name} - Keep: {keep}, Drop: {drop}"
                )
                return keep, drop
            except Exception as e:
                if isinstance(e, CompletionCacheConflictError):
                    raise RuntimeError(
                        f"{stage_name} completion cache conflict for {iter_or_node_id}; "
                        "record mode is fail-closed and will not overwrite a frozen response"
                    ) from e
                exp_logger.error(
                    f"{stage_name}_agent_run_error",
                    seed=seed,
                    stage_name=stage_name,
                    iter_num=iter_num,
                    error=e,
                )
                traceback.print_exc()
                print(
                    f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - " f"Stage {stage_name} - Error: {e}"
                )
                raise

        keep, drop = _()

    # Concatenate terms, unique & order-stable:
    print(f"\nkeep: {keep}")
    print(f"drop: {drop}")
    surviving_terms = [t for t in candidate_terms if t in keep]
    print(f"Surviving terms:\n{surviving_terms}")

    reg_pruned: Optional[OptimizationMethod] = None
    if len(surviving_terms) == 0:
        mse_after_validation_total = float("inf")
        all_metrics_after_validation = {
            "mse": float("inf"),
            "rmse": float("inf"),
            "r2": float("-inf"),
            "nrmse": float("inf"),
            "nmse": float("inf"),
            "accuracy_tol": 0,
            "accuracy_tol_max": 0,
        }
        mse_after_validation = [float("inf")] * len(prepared_dataset.y.keys())
        equation_after = "N/A"
        print("No surviving terms.")
    else:
        # Refit on the surviving terms:
        (
            reg_pruned,
            mse_after_validation,
            mse_after_validation_total,
            all_metrics_after_validation,
        ) = train_and_evaluate(
            cfg,
            surviving_terms,
            prepared_dataset.data,
            prepared_dataset.y,
            prepared_dataset.data_validation,
            prepared_dataset.y_validation,
        )
        print(
            f"MSE AFTER PRUNING = {mse_after_validation_total:.6f} -- "
            f"per-output: {[float(m) for m in list(mse_after_validation)]}\n"
        )
        equation_after = pretty_equations(
            w=reg_pruned.coef_.T, terms=surviving_terms, y=prepared_dataset.y,
            intercept=_get_intercept(reg_pruned),
        )
    print(equation_after)

    sobolev_diagnostics = None
    sobolev_pruning = None
    if sobolev_context is not None:
        if _sobolev_mode(cfg) == "prune_refit":
            outcome = _validation_guarded_sobolev_prune(
                cfg=cfg,
                context=sobolev_context,
                terms=surviving_terms,
                regressor=reg_pruned,
                mse_validation=mse_after_validation,
                mse_validation_total=mse_after_validation_total,
                metrics_validation=all_metrics_after_validation,
                equation=equation_after,
                prepared_dataset=prepared_dataset,
                parent_geometry_key=parent_sobolev_geometry_key,
            )
            surviving_terms = outcome.terms
            reg_pruned = outcome.regressor
            mse_after_validation = outcome.mse_validation
            mse_after_validation_total = outcome.mse_validation_total
            all_metrics_after_validation = outcome.metrics_validation
            equation_after = outcome.equation
            sobolev_diagnostics = outcome.diagnostics
            sobolev_pruning = outcome.audit
            keep = list(surviving_terms)
            drop = [term for term in candidate_terms if term not in set(surviving_terms)]
            exp_logger.info(
                "sobolev_prune_refit_finished",
                seed=seed,
                iter_num=iter_num,
                node_id=node_id,
                audit=sobolev_pruning,
            )
            print(
                f"Sobolev prune/refit: accepted={sobolev_pruning['accepted_prunes']} "
                f"final_terms={surviving_terms} final_nmse={all_metrics_after_validation['nmse']:.6g}"
            )
        else:
            sobolev_diagnostics = _evaluate_sobolev_candidate(
                sobolev_context,
                terms=surviving_terms,
                fitted_regressor=reg_pruned,
                n_outputs=len(prepared_dataset.y),
                parent_geometry_key=parent_sobolev_geometry_key,
            )

    iter_result = IterState(
        iter_num=iter_num,
        node_id=node_id,
        terms_before=candidate_terms,
        terms_after=surviving_terms,
        keep=keep,
        drop=drop,
        # ---
        mse_before=influence_compute_results.mse_validation,
        mse_before_total=influence_compute_results.mse_validation_total,
        mse_after=mse_after_validation,
        mse_after_total=mse_after_validation_total,
        # ---
        metrics_before=all_metrics_before_validation,
        metrics_after=all_metrics_after_validation,
        # Held-out report splits are deliberately absent from node state.
        # They are evaluated once, after validation fixes the selected node.
        metrics_test_before={},
        metrics_test_after={},
        # ---
        metrics_ood_test_before=None,
        metrics_ood_test_after=None,
        # ---
        history=[],  # Set below.
        equation_before=equation_before,
        equation_after=equation_after,
        sobolev_diagnostics=sobolev_diagnostics,
        sobolev_pruning=sobolev_pruning,
    )
    history = add_to_history(
        history=history,
        iter_id=get_iter_or_node_id(cfg, iter_num, node_id),
        iter_result=iter_result,  # Doesn't store iter_result itself, just takes some info from it.
        exp_logger=exp_logger,
        seed=seed,
    )
    iter_result.history = history  # Set the history after adding to it.
    return iter_result


# For ablation only.
def drop_columns_randomly(
    df: pd.DataFrame,
    n_keep: int,
    target_columns: List[str],
    random_seed: int,
    always_keep_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Randomly drop columns from a dataframe, keeping n_keep non-target columns plus all target columns.
    Target columns are never dropped. If always_keep_columns is provided, those columns are guaranteed
    to be among the kept columns (counting toward n_keep).
    """
    np.random.seed(random_seed)

    all_columns = list(df.columns)

    missing_targets = set(target_columns) - set(all_columns)
    if missing_targets:
        raise ValueError(f"Target columns not found in dataframe: {missing_targets}")

    if n_keep < 0:
        raise ValueError(f"n_keep ({n_keep}) cannot be negative")

    droppable_columns = [col for col in all_columns if col not in target_columns]

    if n_keep > len(droppable_columns):
        raise ValueError(
            f"n_keep ({n_keep}) cannot be greater than available non-target columns ({len(droppable_columns)})"
        )

    # Handle always_keep_columns
    if always_keep_columns:
        # Filter to only those that actually exist as droppable columns
        forced_keep = [col for col in always_keep_columns if col in droppable_columns]
        missing_ak = [col for col in always_keep_columns if col not in droppable_columns and col not in target_columns]
        if missing_ak:
            print(f"WARNING: keep_n_columns_always_keep columns not found in data (ignoring): {missing_ak}")
        if len(forced_keep) > n_keep:
            print(
                f"WARNING: always_keep_columns ({len(forced_keep)}) > n_keep ({n_keep}), "
                f"keeping all forced columns ({len(forced_keep)} total)."
            )
            n_keep = len(forced_keep)
    else:
        forced_keep = []

    if n_keep >= len(droppable_columns):
        columns_to_keep = all_columns
    else:
        # Random pool excludes forced-keep columns
        random_pool = [col for col in droppable_columns if col not in forced_keep]
        n_random = n_keep - len(forced_keep)
        randomly_selected = np.random.choice(random_pool, size=n_random, replace=False).tolist() if n_random > 0 else []
        columns_to_keep = target_columns + forced_keep + randomly_selected

    return df[columns_to_keep]


def _select_result_by_validation(
    validation_tracker: FormulaTrackerMultiMetric,
    test_tracker: FormulaTrackerMultiMetric,
    ood_tracker: Optional[FormulaTrackerMultiMetric] = None,
) -> Dict[str, Any]:
    """Select exactly one iteration by validation MSE and align reports to it.

    Test and OOD metrics are intentionally never consulted during selection.
    They are looked up only after the validation-selected ``iter_id`` is fixed.
    """

    selected_validation = validation_tracker.best_result(metric="mse")
    selected_iter_id = selected_validation["iter_id"]

    def metrics_for_iter_id(tracker: FormulaTrackerMultiMetric, split_name: str) -> Dict[str, float]:
        matching_entries = [
            entry for entry in tracker.get_history(metric=None) if entry["iter_id"] == selected_iter_id
        ]
        if not matching_entries:
            raise ValueError(
                f"No {split_name} metrics found for validation-selected iteration {selected_iter_id!r}"
            )
        # Iteration/node ids are unique in the current IGSR runners. Using the
        # most recent entry is defensive if a caller updates an id twice.
        return dict(matching_entries[-1]["metrics"])

    selected_ood_metrics = metrics_for_iter_id(ood_tracker, "OOD") if ood_tracker is not None else None
    return {
        "iter_id": selected_iter_id,
        "metadata": dict(selected_validation["metadata"]),
        "validation_mse": selected_validation["metric_value"],
        "test_metrics": metrics_for_iter_id(test_tracker, "test"),
        "ood_metrics": selected_ood_metrics,
    }


def igsr(cfg: MainConfig, seed: int) -> Dict[str, Any]:
    """
    Run the experiment with a specific seed.

    Args:
        cfg (MainConfig): The Hydra configuration
        seed (int): The random seed for this run

    Returns:
        Dict[str, Any]: A dictionary containing:
            {
                "seed": int,
                "best_formula": str,
                "best_metric": float,  # validation MSE used for selection
                "selected_test_metrics": Dict[str, float],
                "selected_ood_metrics": Optional[Dict[str, float]],
            }
    """
    # import pysr  #?
    # Profiling:
    compute_profiler = ComputeProfiler(
        {"wall_clock_total", "wall_clock_llm_calls", "input_tokens", "output_tokens", "number_llm_calls"}
    )
    start_overall = time.time()

    print(cfg)
    sobolev_mode = _sobolev_mode(cfg)
    sobolev_expansion_decisions: List[Dict[str, Any]] = []

    # Create a seed-specific logger
    log_file = cfg.logging.log_file.replace(".jsonl", f".seed_{seed}.jsonl") if cfg.logging.log_file else None
    exp_logger = get_logger(f"{cfg.logging.logger_name}.seed_{seed}", log_file_path=log_file, mode=cfg.logging.mode)
    exp_logger.info("run_start", seed=seed, config=safe_cfg_yaml(cfg))

    # Azure / LLM config:
    azure_model_key = cfg.llm.api_key

    # Data:
    try:
        # First try to use the centralized dataset loading function
        cfg.dataset.seed = seed  # NOTE: Set the dataset seed as per the experiment!
        data_bundle = get_dataset(cfg.dataset)
    except NotImplementedError:
        raise NotImplementedError(f"Could not load dataset {cfg.dataset}")

    # Drop columns ablation ------------------------------------------------------------
    if cfg.experiment.keep_n_columns is not None:
        actually_keep_n_columns = min(
            cfg.experiment.keep_n_columns, data_bundle.dataset_train.shape[1] - len(data_bundle.target_columns)
        )
        always_keep = getattr(cfg.experiment, "keep_n_columns_always_keep", None)
        print(f"Actually keeping {actually_keep_n_columns} columns.")
        if always_keep:
            print(f"Always keeping columns: {always_keep}")
        data_bundle.dataset_train = drop_columns_randomly(
            data_bundle.dataset_train, actually_keep_n_columns, data_bundle.target_columns, seed, always_keep
        )
        data_bundle.dataset_validation = drop_columns_randomly(
            data_bundle.dataset_validation, actually_keep_n_columns, data_bundle.target_columns, seed, always_keep
        )
        data_bundle.dataset_test = drop_columns_randomly(
            data_bundle.dataset_test, actually_keep_n_columns, data_bundle.target_columns, seed, always_keep
        )
        remaining_columns = [f"{c}" for c in list(data_bundle.dataset_train.columns) if c not in data_bundle.target_columns]
        print(f"Remaining columns: {remaining_columns}")
        data_bundle.data_dictionary += f"""
# Important note:
In this run some data columns were dropped and will not be available.
ONLY THE FOLLOWING {len(remaining_columns)} COLUMNS (and the target columns) ARE AVAILABLE:
{remaining_columns}
"""
    # -----------------------------------------------------------------------------------

    # Prepare the dataset:
    prepared_dataset = prepare_dataset(data_bundle, exp_logger)
    sobolev_context = _initialize_sobolev_diagnostics(
        cfg,
        seed=seed,
        data_bundle=data_bundle,
        prepared_dataset=prepared_dataset,
    )
    exp_logger.info(
        "sobolev_diagnostic_initialized",
        seed=seed,
        mode=sobolev_mode,
        enabled=sobolev_context is not None,
        dataset_identity=(sobolev_context.dataset_identity if sobolev_context is not None else None),
        geometry_indices=(
            sobolev_context.geometry_indices.astype(int).tolist()
            if sobolev_context is not None
            else None
        ),
        feature_policy=(sobolev_context.feature_policy if sobolev_context is not None else None),
        feature_names=(list(sobolev_context.feature_names) if sobolev_context is not None else None),
        excluded_feature_names=(
            list(sobolev_context.excluded_feature_names)
            if sobolev_context is not None
            else None
        ),
        initialization_failure=(
            sobolev_context.initialization_failure if sobolev_context is not None else None
        ),
    )

    # Compute raw feature importance (if enabled):
    if getattr(cfg.experiment, "show_feature_importance", False):
        max_fi = getattr(cfg.experiment, "feature_importance_max_features", 20)
        feature_importances = compute_raw_feature_importance(prepared_dataset, max_features=max_fi)
        feature_importance_text = _format_feature_importance(
            feature_importances, total_features=len(prepared_dataset.data)
        )
        print(f"Feature importance (top {len(feature_importances)}):\n{feature_importance_text}")
    else:
        feature_importance_text = None

    # Initializations
    formula_tracker_val_set = FormulaTrackerMultiMetric(
        metric_directions={
            "mse": "min",
            "rmse": "min",
            "r2": "max",
            "nrmse": "min",
            "nmse": "min",
            "accuracy_tol": "max",
            "accuracy_tol_max": "max",
        }
    )
    formula_tracker_test_set = FormulaTrackerMultiMetric(
        metric_directions={
            "mse": "min",
            "rmse": "min",
            "r2": "max",
            "nrmse": "min",
            "nmse": "min",
            "accuracy_tol": "max",
            "accuracy_tol_max": "max",
        }
    )
    if data_bundle.dataset_ood_test is not None:
        formula_tracker_ood_test_set = FormulaTrackerMultiMetric(
            metric_directions={
                "mse": "min",
                "rmse": "min",
                "r2": "max",
                "nrmse": "min",
                "nmse": "min",
                "accuracy_tol": "max",
                "accuracy_tol_max": "max",
            }
        )
    else:
        formula_tracker_ood_test_set = None

    try:
        # ================================================================================
        # Non-tree flow
        # ================================================================================

        if not cfg.experiment.tree:
            patience = cfg.experiment.early_stop_patience

            iter_state = IterState.get_empty_state(iter_num=0)
            for iter_num in range(cfg.experiment.n_iters):
                print(f"Iteration {iter_num} of {cfg.experiment.n_iters}")

                iter_state = propose_and_prune_once(
                    current_terms=iter_state.terms_after,
                    history=iter_state.history,
                    prepared_dataset=prepared_dataset,
                    # ---
                    cfg=cfg,
                    data_bundle=data_bundle,
                    exp_logger=exp_logger,
                    seed=seed,
                    iter_num=iter_num,
                    node_id=None,
                    # --
                    azure_model_key=azure_model_key,
                    # ---
                    compute_profiler=compute_profiler,
                    feature_importance_text=feature_importance_text,
                    sobolev_context=sobolev_context,
                    parent_sobolev_geometry_key=_candidate_sobolev_geometry_key(iter_state),
                    sibling_slot=None,
                )

                iter_id = get_iter_or_node_id(cfg, iter_num, iter_state.node_id)
                if cfg.experiment.use_xgb_predictor:
                    # XGB validation metrics may govern validation selection,
                    # but held-out report splits remain untouched during search.
                    metrics_val_xgb = _train_xgb_and_eval_validation(
                        iter_state.terms_after, prepared_dataset
                    )
                    formula_tracker_val_set.update(iter_id, metrics_val_xgb, metadata=iter_state.to_dict())
                else:
                    formula_tracker_val_set.update(iter_id, iter_state.metrics_after, metadata=iter_state.to_dict())

                exp_logger.info(
                    "igsr_iter_finished",
                    seed=seed,
                    iter_num=iter_state.iter_num,
                    node_id=iter_state.node_id,
                    iter_result=iter_state.to_dict(),
                )

                best_result = formula_tracker_val_set.best_result(metric="mse")["metric_value"]

                if cfg.experiment.early_stop_enabled:
                    patience = check_early_stop(
                        metric_name="mse",
                        direction="min",
                        best_result=best_result,
                        iter_result=iter_state,
                        patience_left=patience,
                        patience=cfg.experiment.early_stop_patience,
                    )
                    if patience <= 0:
                        exp_logger.info("igsr_early_stop", seed=seed, iter_num=iter_num)
                        break

                if check_token_budget_and_maybe_break(
                    cfg=cfg, round=iter_num, compute_profiler=compute_profiler, seed=seed, logger=exp_logger
                ):
                    break
                if check_wallclock_budget_and_maybe_break(
                    cfg=cfg, round=iter_num, start_time=start_overall, seed=seed, logger=exp_logger
                ):
                    break

        # ================================================================================
        # Tree flow
        # ================================================================================

        if cfg.experiment.tree:
            sibling_rankings_by_node_id: Dict[str, Dict[str, Any]] = {}
            pareto_crossfit_random_state = (
                _pareto_crossfit_random_state(cfg, sobolev_context)
                if sobolev_mode == "pareto_crossfit" and sobolev_context is not None
                else None
            )

            class SRProblem(Problem[IterState, float]):
                def successors(self, s: IterState) -> Iterable[IterState]:
                    print(f"====== Generating successors of {s.node_id} ======")
                    outputs: List[IterState] = []
                    for i in range(cfg.experiment.n_successors):
                        output = propose_and_prune_once(
                            current_terms=s.terms_after,
                            history=s.history,
                            prepared_dataset=prepared_dataset,
                            # ---
                            cfg=cfg,
                            data_bundle=data_bundle,
                            exp_logger=exp_logger,
                            seed=seed,
                            iter_num=None,
                            node_id=f"{s.node_id}_{i}",
                            # --
                            azure_model_key=azure_model_key,
                            # --
                            compute_profiler=compute_profiler,
                            feature_importance_text=feature_importance_text,
                            sobolev_context=sobolev_context,
                            parent_sobolev_geometry_key=_candidate_sobolev_geometry_key(s),
                            sibling_slot=i,
                        )
                        outputs.append(output)

                    if sobolev_mode == "sibling_rerank":
                        relative_tolerance = float(
                            _optional_cfg_value(
                                cfg.experiment,
                                "sobolev_quality_relative_mse_tolerance",
                                0.01,
                            )
                        )
                        absolute_tolerance = float(
                            _optional_cfg_value(
                                cfg.experiment,
                                "sobolev_quality_absolute_mse_tolerance",
                                0.0,
                            )
                        )
                        max_failure_penalty = float(
                            _optional_cfg_value(
                                cfg.experiment,
                                "sobolev_max_failure_penalty",
                                1.0,
                            )
                        )
                        diagnostics_by_node_id = {
                            str(output.node_id): output.sobolev_diagnostics or {}
                            for output in outputs
                        }
                        ranking = rank_siblings(
                            outputs,
                            diagnostics_by_node_id,
                            relative_mse_tolerance=relative_tolerance,
                            absolute_mse_tolerance=absolute_tolerance,
                            max_failure_penalty=max_failure_penalty,
                        )
                        for record in ranking:
                            payload = {
                                **record.as_dict(),
                                "parent_node_id": str(s.node_id),
                                "relative_mse_tolerance": relative_tolerance,
                                "absolute_mse_tolerance": absolute_tolerance,
                                "max_failure_penalty": max_failure_penalty,
                            }
                            sibling_rankings_by_node_id[record.node_id] = payload
                        for output in outputs:
                            output.sobolev_sibling_ranking = dict(
                                sibling_rankings_by_node_id[str(output.node_id)]
                            )
                        exp_logger.info(
                            "sobolev_sibling_batch_ranked",
                            seed=seed,
                            parent_node_id=str(s.node_id),
                            ranking=[record.as_dict() for record in ranking],
                            relative_mse_tolerance=relative_tolerance,
                            absolute_mse_tolerance=absolute_tolerance,
                            max_failure_penalty=max_failure_penalty,
                        )
                    elif sobolev_mode == "pareto_crossfit":
                        if pareto_crossfit_random_state is None:
                            raise RuntimeError(
                                "pareto_crossfit requires an initialized Sobolev context"
                            )
                        relative_tolerance = float(
                            _optional_cfg_value(
                                cfg.experiment,
                                "sobolev_quality_relative_mse_tolerance",
                                0.01,
                            )
                        )
                        absolute_tolerance = float(
                            _optional_cfg_value(
                                cfg.experiment,
                                "sobolev_quality_absolute_mse_tolerance",
                                0.0,
                            )
                        )
                        max_failure_penalty = float(
                            _optional_cfg_value(
                                cfg.experiment,
                                "sobolev_max_failure_penalty",
                                1.0,
                            )
                        )
                        diagnostics_by_node_id = {
                            str(output.node_id): output.sobolev_diagnostics or {}
                            for output in outputs
                        }
                        crossfit_results = {
                            str(output.node_id): _evaluate_train_crossfit_candidate(
                                cfg,
                                prepared_dataset,
                                output.terms_after,
                                random_state=pareto_crossfit_random_state,
                            )
                            for output in outputs
                        }
                        ranking = rank_pareto_siblings(
                            outputs,
                            diagnostics_by_node_id,
                            {
                                node_id: crossfit.as_dict()
                                for node_id, crossfit in crossfit_results.items()
                            },
                            relative_validation_nmse_tolerance=relative_tolerance,
                            absolute_validation_nmse_tolerance=absolute_tolerance,
                            max_failure_penalty=max_failure_penalty,
                        )
                        for record in ranking:
                            output = next(
                                candidate
                                for candidate in outputs
                                if str(candidate.node_id) == record.node_id
                            )
                            payload = {
                                **record.as_dict(),
                                "policy": PARETO_CROSSFIT_POLICY,
                                "parent_node_id": str(s.node_id),
                                "terms_after": list(output.terms_after),
                                "held_out_rows_consulted": False,
                                "relative_validation_nmse_tolerance": relative_tolerance,
                                "absolute_validation_nmse_tolerance": absolute_tolerance,
                                "max_failure_penalty": max_failure_penalty,
                                "crossfit": crossfit_results[record.node_id].as_dict(),
                            }
                            sibling_rankings_by_node_id[record.node_id] = payload
                        for output in outputs:
                            output.sobolev_sibling_ranking = dict(
                                sibling_rankings_by_node_id[str(output.node_id)]
                            )
                        exp_logger.info(
                            "sobolev_pareto_crossfit_sibling_batch_ranked",
                            seed=seed,
                            parent_node_id=str(s.node_id),
                            ranking=[
                                sibling_rankings_by_node_id[record.node_id]
                                for record in ranking
                            ],
                            crossfit_random_state=pareto_crossfit_random_state,
                        )

                    for output in outputs:
                        node_id = get_iter_or_node_id(cfg, None, output.node_id)
                        if cfg.experiment.use_xgb_predictor:
                            metrics_val_xgb = _train_xgb_and_eval_validation(
                                output.terms_after, prepared_dataset
                            )
                            formula_tracker_val_set.update(node_id, metrics_val_xgb, metadata=output.to_dict())
                        else:
                            formula_tracker_val_set.update(node_id, output.metrics_after, metadata=output.to_dict())
                        yield output

                def select_untried_successor(self, candidates, rng):
                    if sobolev_mode not in {"sibling_rerank", "pareto_crossfit"}:
                        return rng.choice(candidates)
                    if sobolev_mode == "sibling_rerank":
                        selected, decision = select_sibling_for_expansion(
                            candidates,
                            sibling_rankings_by_node_id,
                            rng,
                        )
                    else:
                        selected, decision = select_pareto_sibling_for_expansion(
                            candidates,
                            sibling_rankings_by_node_id,
                            rng,
                            relative_crossfit_mean_tolerance=float(
                                _optional_cfg_value(
                                    cfg.experiment,
                                    "sobolev_crossfit_mean_relative_tolerance",
                                    0.0,
                                )
                            ),
                            absolute_crossfit_mean_tolerance=float(
                                _optional_cfg_value(
                                    cfg.experiment,
                                    "sobolev_crossfit_mean_absolute_tolerance",
                                    0.0,
                                )
                            ),
                            relative_crossfit_worst_tolerance=float(
                                _optional_cfg_value(
                                    cfg.experiment,
                                    "sobolev_crossfit_worst_relative_tolerance",
                                    0.0,
                                )
                            ),
                            absolute_crossfit_worst_tolerance=float(
                                _optional_cfg_value(
                                    cfg.experiment,
                                    "sobolev_crossfit_worst_absolute_tolerance",
                                    0.0,
                                )
                            ),
                            minimum_penalty_gain=float(
                                _optional_cfg_value(
                                    cfg.experiment,
                                    "sobolev_minimum_penalty_gain",
                                    0.0,
                                )
                            ),
                        )
                    if selected.sobolev_sibling_ranking is not None:
                        selected.sobolev_sibling_ranking["expansion_decision"] = decision.as_dict()
                    exp_logger.info(
                        "sobolev_sibling_expansion_selected",
                        seed=seed,
                        decision=decision.as_dict(),
                    )
                    sobolev_expansion_decisions.append(decision.as_dict())
                    return selected

                # lower = better
                def evaluate(self, s: IterState) -> float:
                    reward_metric = str(
                        _optional_cfg_value(cfg.experiment, "tree_reward_metric", "mse")
                    ).strip().lower()
                    if reward_metric == "mse":
                        return float(s.mse_after_total)
                    if reward_metric == "nmse":
                        value = s.metrics_after.get("nmse")
                        return float("inf") if value is None else float(value)
                    raise ValueError(
                        f"Unknown experiment.tree_reward_metric={reward_metric!r}"
                    )

                def is_terminal(self, s: IterState) -> bool:
                    return False

                # pretty labels for the visualiser
                def label(self, s: IterState) -> str:
                    return (
                        f"{s.node_id}" f"\nmse={s.mse_before:.3g}"
                        # f"\nr2={s.r2:.8f}"
                        # f"\nnrmse={s.nrmse:.8f}"
                    )

                def key(self, s: IterState) -> Hashable:
                    return str(s.node_id)

            problem = SRProblem()
            sess = SearchSession(
                problem,
                visualiser=None,
                early_stopping_enabled=cfg.experiment.early_stop_enabled,
                early_stopping_patience=cfg.experiment.early_stop_patience,
            )
            start = IterState.get_empty_state(node_id="node_0")
            best_val, best_state = sess.mcts(
                start,
                depth_limit=cfg.experiment.depth_limit,
                total_budget=cfg.experiment.total_budget,
                seed=seed,
                stop_on_terminal=True,
                # As successor is expensive. =========================================
                rollout_is_just_node_reward=cfg.experiment.rollout_is_just_node_reward,
                rollout_depth=cfg.experiment.rollout_depth,
                cache_successors=True,
                always_exit_on_terminal=True,
                # ====================================================================
                c=cfg.experiment.c,  # with default sqrt(2) value explores too much, lower it.
                # --- For token/wallclock early stopping:
                extras_dict={
                    "cfg": cfg,
                    "compute_profiler": compute_profiler,
                    "start_time": start_overall,
                    "seed": seed,
                    "logger": exp_logger,
                },
                # ---
            )

            exp_logger.info(
                "igsr_iter_finished",
                seed=seed,
                iter_num=best_state.iter_num,
                node_id=best_state.node_id,
                iter_result=best_state.to_dict(),
            )

    except Exception as e:
        len_history = len(formula_tracker_val_set.get_history(metric=None))
        if len_history >= cfg.experiment.allow_error_history_length:
            exp_logger.error("error_allow_result", len_history=len_history, seed=seed, error=e)
        else:
            exp_logger.error("error_raise_result", len_history=len_history, seed=seed, error=e)
            raise e

    elapsed_overall = time.time() - start_overall
    compute_profiler.accumulate("wall_clock_total", elapsed_overall)
    print(f">>> Compute profiler: {compute_profiler.as_dict()}")

    # Validation fixes exactly one node before either held-out report split is
    # touched.  Refit that frozen term set on train with the same configured
    # optimizer, then evaluate ID-test/OOD once.  This ordering prevents report
    # labels, metrics, and even report-domain feature validity from influencing
    # generation, pruning, MCTS reward, sibling ranking, or selection.
    selected_validation = formula_tracker_val_set.best_result(metric="mse")
    selected_test_metrics, selected_ood_metrics, report_status = (
        _refit_and_evaluate_selected_for_reporting(
            cfg,
            selected_validation,
            prepared_dataset,
            exp_logger,
        )
    )
    selected_iter_id = selected_validation["iter_id"]
    report_metadata = dict(selected_validation["metadata"])
    report_metadata["selected_report_evaluation"] = report_status
    formula_tracker_test_set.update(
        selected_iter_id,
        selected_test_metrics,
        metadata=report_metadata,
    )
    if formula_tracker_ood_test_set is not None:
        formula_tracker_ood_test_set.update(
            selected_iter_id,
            selected_ood_metrics,
            metadata=report_metadata,
        )
    selected_result = _select_result_by_validation(
        formula_tracker_val_set,
        formula_tracker_test_set,
        formula_tracker_ood_test_set,
    )
    selected_result["test_metrics"] = _with_report_status_fields(
        selected_result["test_metrics"],
        report_status["test"],
    )
    if selected_result["ood_metrics"] is not None:
        selected_result["ood_metrics"] = _with_report_status_fields(
            selected_result["ood_metrics"],
            report_status["ood"],
        )

    data_settings = data_bundle.data_settings if isinstance(data_bundle.data_settings, Mapping) else {}
    split_settings = data_settings.get("split", {})
    if not isinstance(split_settings, Mapping):
        split_settings = {}
    history_validation = formula_tracker_val_set.get_history(metric=None)
    history_test = formula_tracker_test_set.get_history(metric=None)
    history_ood_test = (
        formula_tracker_ood_test_set.get_history(metric=None) if formula_tracker_ood_test_set is not None else None
    )
    sobolev_run_status = _summarize_sobolev_run(
        mode=sobolev_mode,
        context=sobolev_context,
        validation_history=history_validation,
        expansion_decisions=sobolev_expansion_decisions,
    )
    exp_logger.info("sobolev_run_status", seed=seed, status=sobolev_run_status)
    return {
        "seed": seed,
        "method_profile": (
            "sn_igsr"
            if sobolev_mode in {"sibling_rerank", "prune_refit", "pareto_crossfit"}
            else "igsr"
        ),
        "sobolev_mode": sobolev_mode,
        "task_identity": str(
            data_settings.get("instance_id")
            or data_settings.get("problem_id")
        ),
        "split_identity_sha256": split_settings.get("split_identity_sha256"),
        "search_split_identity_sha256": split_settings.get(
            "search_split_identity_sha256"
        ),
        "model_revision": _optional_cfg_value(cfg.llm, "model_revision")
        or _optional_cfg_value(cfg.llm, "model_version"),
        "best_formula": selected_result["metadata"]["equation_after"],
        # ``best_metric`` remains the selection metric and is therefore validation MSE.
        "best_metric": selected_result["validation_mse"],
        "selected_test_metrics": selected_result["test_metrics"],
        "selected_ood_metrics": selected_result["ood_metrics"],
        "history_validation": history_validation,
        "history_test": history_test,
        "history_ood_test": history_ood_test,
        "compute_profiler": compute_profiler.as_dict(),
        "sobolev_expansion_decisions": sobolev_expansion_decisions,
        "sobolev_run_status": sobolev_run_status,
    }
