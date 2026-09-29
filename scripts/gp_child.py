#!/usr/bin/env python3
"""Run one Base-GP or SN-GP plug-in job to an exact generation target."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.heuristic.gp.gp import (
    GP,
    PROFILE_BASIS_FOREST_BASE,
    PROFILE_BASIS_FOREST_SN_CROSSOVER,
    PROFILE_BASIS_FOREST_SN_RESIDUAL_INFUSION,
    PROFILE_BASIS_FOREST_SN_SAFE_MUTATION,
    PROFILE_BASIS_FOREST_SN_VERIFIED_COMPRESSION,
    PROFILE_BASIS_FOREST_SN_CONSTRUCT_COMPRESS,
    PROFILE_BASIS_FOREST_SN_RESIDUAL_SHADOW,
    PROFILE_BASIS_FOREST_SN_REPAIR_COVERAGE,
    PROFILE_BASIS_FOREST_SN_COMPRESS_SHADOW,
    PROFILE_BASIS_FOREST_SN_MUTATION_RECOMBINATION,
    PROFILE_BASIS_FOREST_SN_MUTATION_SHADOW,
    PROFILE_BASIS_FOREST_SN_COMPRESS_MUTATION_SHADOW,
    PROFILE_BASIS_FOREST_SN_COMPRESS_MULTISOURCE_SHADOW,
    PROFILE_BASE,
    PROFILE_BASE_EXPORT_ALIGNED,
    PROFILE_SN_ARCHIVE_EXPORT,
    PROFILE_SN_ARCHIVE_ANCHOR_EXPORT,
    PROFILE_SN_ARCHIVE_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_DIVERSE_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_DUAL_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_DUAL_POOL_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_REFINED_DUAL_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_FLOATING_DUAL_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_BALANCED_SCREEN_DUAL_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_BALANCED_FLOATING_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_INNOVATION_SCREEN_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_PAIR_LOOKAHEAD_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_PRODUCT_LIFT_BEAM_EXPORT,
    PROFILE_SN_ARCHIVE_PRODUCT_ACCURACY_EXPORT,
    PROFILE_SN_ARCHIVE_CONDITIONAL_PRODUCT_ACCURACY_EXPORT,
    PROFILE_SN_ARCHIVE_ITERATED_CONDITIONAL_PRODUCT_ACCURACY_EXPORT,
    PROFILE_SN_ARCHIVE_CONDITIONAL_UNARY_ACCURACY_EXPORT,
    PROFILE_SN_ARCHIVE_CONDITIONAL_RATIONAL_ACCURACY_EXPORT,
    PROFILE_SN_ARCHIVE_CONDITIONAL_AFFINE_UNARY_ACCURACY_EXPORT,
    PROFILE_SN_ARCHIVE_CONDITIONAL_RADIAL_ACCURACY_EXPORT,
    PROFILE_SN_ARCHIVE_FEATURE_RADIAL_ACCURACY_EXPORT,
    PROFILE_SN_DIRECT_FEATURE_RADIAL_ACCURACY_EXPORT,
    PROFILE_SN_DIRECT_FEATURE_AFFINE_UNARY_ACCURACY_EXPORT,
    PROFILE_SN_DIRECT_FEATURE_PRODUCT_ACCURACY_EXPORT,
    PROFILE_SN_DIRECT_PHASE_ACCURACY_EXPORT,
    PROFILE_SN_DIRECT_PHASE_INTERACTION_ACCURACY_EXPORT,
    PROFILE_SN_DIRECT_RADICAL_PHASE_ACCURACY_EXPORT,
    PROFILE_SN_SHARED_PHASE_RATIONAL_ACCURACY_EXPORT,
    PROFILE_SN_DAMPED_EXPONENTIAL_ACCURACY_EXPORT,
    PROFILE_SN_AFFINE_GAUSSIAN_ACCURACY_EXPORT,
    PROFILE_SN_SINC_SQUARED_ACCURACY_EXPORT,
    PROFILE_SN_SHARED_UNARY_POLYNOMIAL_ACCURACY_EXPORT,
    PROFILE_SN_RECIPROCAL_TRIG_ACCURACY_EXPORT,
    PROFILE_SN_CROSS_UNARY_AFFINE_ACCURACY_EXPORT,
    PROFILE_SN_SHARED_DENOMINATOR_ACCURACY_EXPORT,
    PROFILE_SN_RELATIVISTIC_RATIONAL_ACCURACY_EXPORT,
    PROFILE_SN_COSINE_LAW_RADIAL_ACCURACY_EXPORT,
    PROFILE_SN_RECIPROCAL_SINE_SQUARE_ACCURACY_EXPORT,
    PROFILE_SN_INVERSE_COSINE_RADIAL_ACCURACY_EXPORT,
    PROFILE_SN_MULTIAXIS_INVERSE_SQUARE_ACCURACY_EXPORT,
    PROFILE_SN_INTERFERENCE_SINE_RATIO_ACCURACY_EXPORT,
    PROFILE_SN_SPARSE_RADICAL_ACCURACY_EXPORT,
    PROFILE_SN_SHARED_RATIO_TRIG_POLYNOMIAL_ACCURACY_EXPORT,
    DEEP_ACCURACY_PARAMETER_NAMES,
    DEEP_ACCURACY_PROFILE_NAMES,
    PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
    PROFILE_SN_BASIS_EXCHANGE,
    PROFILE_SN_BASIS_INFUSION,
    PROFILE_SN_BASIS_ARCHIVE,
    PROFILE_SN_PARTIAL_RESIDUAL_ARCHIVE,
    PROFILE_SN_SCREENED_BASIS_PURSUIT,
    PROFILE_SN_SCREENED_PURSUIT_SHADOW,
    PROFILE_SN_PARETO_PURSUIT_SHADOW,
    PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
    PROFILE_SN_BIDIRECTIONAL_BASIS_REPLACEMENT,
    PROFILE_SN_CONDITIONAL_BASIS_REPLACEMENT,
    PROFILE_SN_ORTHOGONAL_BASIS_CROSSOVER,
    PROFILE_SN_DUAL_PATH_SHADOW_SLOT,
    PROFILE_SN_RESIDUAL_BASIS_INFUSION,
    PROFILE_SN_CHILD_GEOMETRY,
    PROFILE_SN_STABLE,
    PROFILE_SN_REPAIR_COVERAGE,
    PROFILE_SN_REPAIR_SHADOW,
    PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
    PROFILE_SN_VERIFIED_REPAIR,
    PROFILE_SN_V2,
)
from src.nd2py import nd2py as nd


class HardGuard(BaseException):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal worker for the two final GP profiles")
    for name in ("config", "input", "initial-population", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("task-index", "seed", "target-generation"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", choices=("base_gp_export_aligned", "sn_gp_inverse_trig_mobius_accuracy_export"), required=True)
    return parser.parse_args()


def write_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_expression(
    expression: str, X: np.ndarray, y: np.ndarray, names: tuple[str, ...]
) -> dict:
    tree = nd.parse(expression)
    values = {name: X[:, index] for index, name in enumerate(names)}
    with np.errstate(all="ignore"):
        prediction = np.asarray(tree.eval(values, use_eps=1e-6), dtype=float)
    if prediction.ndim == 0:
        prediction = np.full(len(y), float(prediction))
    prediction = prediction.reshape(-1)
    nonfinite = int(np.count_nonzero(~np.isfinite(prediction)))
    prediction[~np.isfinite(prediction)] = 0.0
    residual = prediction - y
    mse = float(np.mean(np.square(residual)))
    return {
        "exported_expression": expression,
        "complexity": int(len(tree)),
        "exported_search_r2": float(1.0 - mse / np.var(y)),
        "exported_search_rmse": float(np.sqrt(mse)),
        "evaluation_rows": int(len(y)),
        "nonfinite_prediction_count_before_zero_fill": nonfinite,
        "refit_performed": False,
        "metric_scope": "same materialized search rows; not test R2",
    }


def main() -> int:
    args = parse_args()
    if args.target_generation < 0:
        raise ValueError("target generation must be non-negative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    algorithm = config["algorithm"]
    budget = config["budget"]
    equal = config.get("equal_generation", config.get("fixed_generation"))
    if equal is None:
        raise KeyError("config requires equal_generation or fixed_generation")
    profile_config = config["profiles"][args.profile]
    time_limit = float(equal["time_limit_seconds"])
    hard_time_limit = float(equal["hard_time_limit_seconds"])
    args.output.mkdir(parents=True, exist_ok=False)

    with np.load(args.input, allow_pickle=False) as bundle:
        X = np.ascontiguousarray(bundle["X"], dtype=float)
        y = np.ascontiguousarray(bundle["y"], dtype=float)
        row_ids = np.ascontiguousarray(bundle["row_ids"], dtype=int)
        feature_names = tuple(str(value) for value in bundle["feature_names"].tolist())
        input_key = (
            int(bundle["task_index"]),
            str(bundle["dataset"].item()),
            int(bundle["seed"]),
        )
    if input_key != (args.task_index, args.dataset, args.seed):
        raise RuntimeError("Run specification and paired input differ")
    initial = json.loads(args.initial_population.read_text(encoding="utf-8"))
    if (
        int(initial["task_index"]),
        str(initial["dataset"]),
        int(initial["seed"]),
    ) != input_key:
        raise RuntimeError("Initial population and paired input differ")
    expressions = tuple(str(value) for value in initial["expressions"])
    if len(expressions) != int(algorithm["population_size"]):
        raise RuntimeError("Initial population has the wrong size")
    write_json(
        args.output / "run.json",
        {
            "task_index": args.task_index,
            "dataset": args.dataset,
            "seed": args.seed,
            "profile": args.profile,
            "target_generation": args.target_generation,
            "input": str(args.input.resolve()),
            "initial_population": str(args.initial_population.resolve()),
            "selected_rows": int(len(y)),
            "physical_row_ids": row_ids.tolist(),
            "time_limit_seconds": time_limit,
            "hard_time_limit_seconds": hard_time_limit,
        },
    )

    variables = [nd.Variable(name, nettype="scalar") for name in feature_names]
    deep_accuracy_parameters = {
        name: profile_config[name]
        for name in DEEP_ACCURACY_PARAMETER_NAMES
        if name in profile_config
    }
    model = GP(
        variables,
        profile=args.profile,
        population_size=int(algorithm["population_size"]),
        elitism_k=int(algorithm["elitism_k"]),
        tournament_size=int(algorithm["tournament_size"]),
        p_crossover=float(algorithm["p_crossover"]),
        p_subtree_mutation=float(algorithm["p_subtree_mutation"]),
        p_hoist_mutation=float(algorithm["p_hoist_mutation"]),
        p_point_mutation=float(algorithm["p_point_mutation"]),
        p_point_replace=float(algorithm["p_point_replace"]),
        depth_range=tuple(int(value) for value in algorithm["depth_range"]),
        full_prob=float(algorithm["full_prob"]),
        fixed_constants=tuple(float(value) for value in algorithm["fixed_constants"]),
        random_state=args.seed,
        n_iter=args.target_generation,
        time_limit=time_limit,
        hard_time_limit=hard_time_limit,
        max_len=int(algorithm["max_len"]),
        max_additive_terms=int(algorithm["max_additive_terms"]),
        eta=float(algorithm["eta"]),
        ratio=float(budget["ratio"]),
        sobolev_alpha=float(profile_config.get("sobolev_alpha", 0.0)),
        sobolev_tau=float(profile_config.get("tau", 1.0 / np.sqrt(10.0))),
        sobolev_lambda_value=float(profile_config.get("lambda_value", 1.0)),
        sobolev_lambda_gradient=float(profile_config.get("lambda_gradient", 1.0)),
        geometry_sample_size=int(profile_config.get("geometry_sample_size", 64)),
        shortlist_size=int(profile_config.get("shortlist_size", 64)),
        sobolev_failure_policy=str(profile_config.get("failure_policy", "max_penalty")),
        sobolev_pruning=bool(profile_config.get("sobolev_pruning", False)),
        sobolev_max_prunes=int(profile_config.get("sobolev_max_prunes", 0)),
        prune_elite_k=int(profile_config.get("prune_elite_k", 0)),
        sobolev_acceptance_tolerance=float(
            profile_config.get("acceptance_tolerance", 0.0)
        ),
        sn_compare_epsilon_abs=float(
            profile_config.get("sn_compare_epsilon_abs", 1e-6)
        ),
        sn_compare_epsilon_rel=float(
            profile_config.get("sn_compare_epsilon_rel", 1e-4)
        ),
        sn_mutation_delta=float(profile_config.get("sn_mutation_delta", 0.05)),
        sn_mutation_gamma=float(profile_config.get("sn_mutation_gamma", 2.0)),
        sn_mutation_impact_beta=float(
            profile_config.get("sn_mutation_impact_beta", 0.0)
        ),
        sn_mutation_impact_epsilon=float(
            profile_config.get("sn_mutation_impact_epsilon", 1e-6)
        ),
        sn_mutation_max_normalized_impact=float(
            profile_config.get("sn_mutation_max_normalized_impact", 1.0)
        ),
        sn_base_anchor_enabled=profile_config.get("sn_base_anchor_enabled"),
        sn_base_anchor_elite_slots=int(
            profile_config.get("sn_base_anchor_elite_slots", 1)
        ),
        sn_export_mode=profile_config.get("sn_export_mode"),
        sn_tau=float(
            profile_config.get("sn_tau", profile_config.get("tau", 1.0 / np.sqrt(10.0)))
        ),
        sn_geometry_sample_size=int(
            profile_config.get(
                "sn_geometry_sample_size",
                profile_config.get("geometry_sample_size", 64),
            )
        ),
        sn_cache_enabled=bool(profile_config.get("sn_cache_enabled", True)),
        sn_provenance_enabled=bool(profile_config.get("sn_provenance_enabled", True)),
        sn_trace_every=int(profile_config.get("sn_trace_every", 10000)),
        sn_repair_shortlist_size=int(
            profile_config.get("sn_repair_shortlist_size", 16)
        ),
        sn_repair_max_accepted_per_generation=int(
            profile_config.get("sn_repair_max_accepted_per_generation", 1)
        ),
        sn_repair_shadow_enabled=profile_config.get("sn_repair_shadow_enabled"),
        sn_archive_export_enabled=profile_config.get("sn_archive_export_enabled"),
        sn_archive_export_max_steps=int(
            profile_config.get("sn_archive_export_max_steps", 8)
        ),
        sn_archive_anchor_export_enabled=profile_config.get(
            "sn_archive_anchor_export_enabled"
        ),
        sn_archive_anchor_export_max_steps=int(
            profile_config.get("sn_archive_anchor_export_max_steps", 4)
        ),
        sn_archive_beam_export_enabled=profile_config.get(
            "sn_archive_beam_export_enabled"
        ),
        sn_archive_beam_export_max_steps=int(
            profile_config.get("sn_archive_beam_export_max_steps", 8)
        ),
        sn_archive_beam_width=int(profile_config.get("sn_archive_beam_width", 4)),
        sn_archive_beam_shortlist_size=int(
            profile_config.get("sn_archive_beam_shortlist_size", 4)
        ),
        sn_archive_beam_value_shortlist_size=int(
            profile_config.get("sn_archive_beam_value_shortlist_size", 0)
        ),
        sn_archive_beam_innovation_shortlist_size=int(
            profile_config.get("sn_archive_beam_innovation_shortlist_size", 0)
        ),
        sn_archive_beam_pair_first_shortlist_size=int(
            profile_config.get("sn_archive_beam_pair_first_shortlist_size", 0)
        ),
        sn_archive_beam_pair_second_shortlist_size=int(
            profile_config.get("sn_archive_beam_pair_second_shortlist_size", 0)
        ),
        sn_archive_interaction_joint_shortlist_size=int(
            profile_config.get("sn_archive_interaction_joint_shortlist_size", 0)
        ),
        sn_archive_interaction_value_shortlist_size=int(
            profile_config.get("sn_archive_interaction_value_shortlist_size", 0)
        ),
        sn_archive_unary_transforms=tuple(
            profile_config.get("sn_archive_unary_transforms", ["sin", "cos", "tanh"])
        ),
        sn_direct_phase_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_direct_phase_scales",
                [0.5, 1.0, 2.0, float(np.pi), 2.0 * float(np.pi)],
            )
        ),
        sn_direct_phase_include_squares=bool(
            profile_config.get("sn_direct_phase_include_squares", True)
        ),
        sn_phase_interaction_batch_size=int(
            profile_config.get("sn_phase_interaction_batch_size", 512)
        ),
        sn_radical_scales=tuple(
            float(value)
            for value in profile_config.get("sn_radical_scales", [0.5, 1.0, 2.0])
        ),
        sn_radical_max_dimension=int(profile_config.get("sn_radical_max_dimension", 6)),
        sn_radical_phase_anchor_limit=int(
            profile_config.get("sn_radical_phase_anchor_limit", 4)
        ),
        sn_radical_amplitude_joint_shortlist_size=int(
            profile_config.get("sn_radical_amplitude_joint_shortlist_size", 4)
        ),
        sn_radical_amplitude_value_shortlist_size=int(
            profile_config.get("sn_radical_amplitude_value_shortlist_size", 4)
        ),
        sn_shared_phase_mobius_shifts=tuple(
            float(value)
            for value in profile_config.get(
                "sn_shared_phase_mobius_shifts",
                [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
            )
        ),
        sn_shared_phase_max_dimension=int(
            profile_config.get("sn_shared_phase_max_dimension", 6)
        ),
        sn_shared_phase_max_numerator_degree=int(
            profile_config.get("sn_shared_phase_max_numerator_degree", 4)
        ),
        sn_shared_phase_max_denominator_degree=int(
            profile_config.get("sn_shared_phase_max_denominator_degree", 2)
        ),
        sn_shared_phase_backbone_value_pool_size=int(
            profile_config.get("sn_shared_phase_backbone_value_pool_size", 64)
        ),
        sn_shared_phase_backbone_shortlist_size=int(
            profile_config.get("sn_shared_phase_backbone_shortlist_size", 16)
        ),
        sn_shared_phase_modulated_shortlist_size=int(
            profile_config.get("sn_shared_phase_modulated_shortlist_size", 32)
        ),
        sn_shared_phase_complement_pool_size=int(
            profile_config.get("sn_shared_phase_complement_pool_size", 8)
        ),
        sn_shared_phase_complement_shortlist_size=int(
            profile_config.get("sn_shared_phase_complement_shortlist_size", 4)
        ),
        sn_shared_phase_proposal_limit=int(
            profile_config.get("sn_shared_phase_proposal_limit", 32)
        ),
        sn_exponential_scales=tuple(
            float(value)
            for value in profile_config.get("sn_exponential_scales", [0.5, 1.0, 2.0])
        ),
        sn_exponential_max_dimension=int(
            profile_config.get("sn_exponential_max_dimension", 6)
        ),
        sn_exponential_max_numerator_degree=int(
            profile_config.get("sn_exponential_max_numerator_degree", 3)
        ),
        sn_exponential_max_denominator_degree=int(
            profile_config.get("sn_exponential_max_denominator_degree", 2)
        ),
        sn_exponential_value_pool_size=int(
            profile_config.get("sn_exponential_value_pool_size", 64)
        ),
        sn_exponential_shortlist_size=int(
            profile_config.get("sn_exponential_shortlist_size", 16)
        ),
        sn_exponential_composite_value_pool_size=int(
            profile_config.get("sn_exponential_composite_value_pool_size", 64)
        ),
        sn_exponential_composite_shortlist_size=int(
            profile_config.get("sn_exponential_composite_shortlist_size", 16)
        ),
        sn_exponential_proposal_limit=int(
            profile_config.get("sn_exponential_proposal_limit", 32)
        ),
        sn_affine_gaussian_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_affine_gaussian_scales", [0.5, 1.0, 2.0]
            )
        ),
        sn_affine_gaussian_max_dimension=int(
            profile_config.get("sn_affine_gaussian_max_dimension", 6)
        ),
        sn_affine_gaussian_value_pool_size=int(
            profile_config.get("sn_affine_gaussian_value_pool_size", 64)
        ),
        sn_affine_gaussian_shortlist_size=int(
            profile_config.get("sn_affine_gaussian_shortlist_size", 16)
        ),
        sn_affine_gaussian_composite_value_pool_size=int(
            profile_config.get("sn_affine_gaussian_composite_value_pool_size", 64)
        ),
        sn_affine_gaussian_composite_shortlist_size=int(
            profile_config.get("sn_affine_gaussian_composite_shortlist_size", 16)
        ),
        sn_affine_gaussian_proposal_limit=int(
            profile_config.get("sn_affine_gaussian_proposal_limit", 32)
        ),
        sn_sinc_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_sinc_scales",
                [0.5, 1.0, 2.0, float(np.pi), float(2.0 * np.pi)],
            )
        ),
        sn_sinc_max_dimension=int(profile_config.get("sn_sinc_max_dimension", 6)),
        sn_sinc_max_numerator_degree=int(
            profile_config.get("sn_sinc_max_numerator_degree", 3)
        ),
        sn_sinc_max_denominator_degree=int(
            profile_config.get("sn_sinc_max_denominator_degree", 2)
        ),
        sn_sinc_shape_value_pool_size=int(
            profile_config.get("sn_sinc_shape_value_pool_size", 64)
        ),
        sn_sinc_shape_shortlist_size=int(
            profile_config.get("sn_sinc_shape_shortlist_size", 16)
        ),
        sn_sinc_composite_value_pool_size=int(
            profile_config.get("sn_sinc_composite_value_pool_size", 64)
        ),
        sn_sinc_composite_shortlist_size=int(
            profile_config.get("sn_sinc_composite_shortlist_size", 16)
        ),
        sn_sinc_proposal_limit=int(profile_config.get("sn_sinc_proposal_limit", 32)),
        sn_shared_unary_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_shared_unary_scales",
                [0.5, 1.0, 2.0, float(np.pi), float(2.0 * np.pi)],
            )
        ),
        sn_shared_unary_transforms=tuple(
            str(value)
            for value in profile_config.get(
                "sn_shared_unary_transforms", ["sin", "cos", "tanh"]
            )
        ),
        sn_shared_unary_max_dimension=int(
            profile_config.get("sn_shared_unary_max_dimension", 6)
        ),
        sn_shared_unary_max_numerator_degree=int(
            profile_config.get("sn_shared_unary_max_numerator_degree", 2)
        ),
        sn_shared_unary_max_denominator_degree=int(
            profile_config.get("sn_shared_unary_max_denominator_degree", 2)
        ),
        sn_shared_unary_phase_value_pool_size=int(
            profile_config.get("sn_shared_unary_phase_value_pool_size", 64)
        ),
        sn_shared_unary_phase_shortlist_size=int(
            profile_config.get("sn_shared_unary_phase_shortlist_size", 16)
        ),
        sn_shared_unary_amplitude_value_pool_size=int(
            profile_config.get("sn_shared_unary_amplitude_value_pool_size", 32)
        ),
        sn_shared_unary_amplitude_shortlist_size=int(
            profile_config.get("sn_shared_unary_amplitude_shortlist_size", 8)
        ),
        sn_shared_unary_proposal_limit=int(
            profile_config.get("sn_shared_unary_proposal_limit", 32)
        ),
        sn_reciprocal_trig_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_reciprocal_trig_scales",
                [0.5, 1.0, 2.0, float(np.pi), float(2.0 * np.pi)],
            )
        ),
        sn_reciprocal_trig_transforms=tuple(
            str(value)
            for value in profile_config.get(
                "sn_reciprocal_trig_transforms", ["sin", "cos", "tanh"]
            )
        ),
        sn_reciprocal_trig_max_dimension=int(
            profile_config.get("sn_reciprocal_trig_max_dimension", 6)
        ),
        sn_reciprocal_trig_value_pool_size=int(
            profile_config.get("sn_reciprocal_trig_value_pool_size", 64)
        ),
        sn_reciprocal_trig_shortlist_size=int(
            profile_config.get("sn_reciprocal_trig_shortlist_size", 16)
        ),
        sn_reciprocal_trig_proposal_limit=int(
            profile_config.get("sn_reciprocal_trig_proposal_limit", 32)
        ),
        sn_cross_unary_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_cross_unary_scales",
                [0.5, 1.0, 2.0, float(np.pi), float(2.0 * np.pi)],
            )
        ),
        sn_cross_unary_transforms=tuple(
            str(value)
            for value in profile_config.get(
                "sn_cross_unary_transforms", ["sin", "cos", "tanh"]
            )
        ),
        sn_cross_unary_inner_powers=tuple(
            int(value)
            for value in profile_config.get("sn_cross_unary_inner_powers", [1, 2])
        ),
        sn_cross_unary_max_dimension=int(
            profile_config.get("sn_cross_unary_max_dimension", 6)
        ),
        sn_cross_unary_max_numerator_degree=int(
            profile_config.get("sn_cross_unary_max_numerator_degree", 2)
        ),
        sn_cross_unary_max_denominator_degree=int(
            profile_config.get("sn_cross_unary_max_denominator_degree", 1)
        ),
        sn_cross_unary_value_pool_size=int(
            profile_config.get("sn_cross_unary_value_pool_size", 64)
        ),
        sn_cross_unary_shortlist_size=int(
            profile_config.get("sn_cross_unary_shortlist_size", 16)
        ),
        sn_cross_unary_proposal_limit=int(
            profile_config.get("sn_cross_unary_proposal_limit", 32)
        ),
        sn_shared_denominator_signs=tuple(
            int(value)
            for value in profile_config.get("sn_shared_denominator_signs", [1, -1])
        ),
        sn_shared_denominator_numerator_signs=tuple(
            int(value)
            for value in profile_config.get(
                "sn_shared_denominator_numerator_signs", [1]
            )
        ),
        sn_shared_denominator_max_dimension=int(
            profile_config.get("sn_shared_denominator_max_dimension", 6)
        ),
        sn_shared_denominator_value_pool_size=int(
            profile_config.get("sn_shared_denominator_value_pool_size", 64)
        ),
        sn_shared_denominator_shortlist_size=int(
            profile_config.get("sn_shared_denominator_shortlist_size", 16)
        ),
        sn_shared_denominator_proposal_limit=int(
            profile_config.get("sn_shared_denominator_proposal_limit", 32)
        ),
        sn_relativistic_rational_signs=tuple(
            int(value)
            for value in profile_config.get("sn_relativistic_rational_signs", [1, -1])
        ),
        sn_relativistic_rational_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_relativistic_rational_scales", [0.5, 1.0, 2.0]
            )
        ),
        sn_relativistic_rational_max_dimension=int(
            profile_config.get("sn_relativistic_rational_max_dimension", 6)
        ),
        sn_relativistic_rational_value_pool_size=int(
            profile_config.get("sn_relativistic_rational_value_pool_size", 64)
        ),
        sn_relativistic_rational_shortlist_size=int(
            profile_config.get("sn_relativistic_rational_shortlist_size", 16)
        ),
        sn_relativistic_rational_proposal_limit=int(
            profile_config.get("sn_relativistic_rational_proposal_limit", 32)
        ),
        sn_cosine_law_phase_signs=tuple(
            int(value)
            for value in profile_config.get("sn_cosine_law_phase_signs", [1, -1])
        ),
        sn_cosine_law_radial_signs=tuple(
            int(value)
            for value in profile_config.get("sn_cosine_law_radial_signs", [1, -1])
        ),
        sn_cosine_law_phase_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_cosine_law_phase_scales", [0.5, 1.0, 2.0]
            )
        ),
        sn_cosine_law_max_dimension=int(
            profile_config.get("sn_cosine_law_max_dimension", 6)
        ),
        sn_cosine_law_value_pool_size=int(
            profile_config.get("sn_cosine_law_value_pool_size", 64)
        ),
        sn_cosine_law_shortlist_size=int(
            profile_config.get("sn_cosine_law_shortlist_size", 16)
        ),
        sn_cosine_law_proposal_limit=int(
            profile_config.get("sn_cosine_law_proposal_limit", 32)
        ),
        sn_reciprocal_sine_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_reciprocal_sine_scales",
                [0.5, 1.0, 2.0, float(np.pi), float(2.0 * np.pi)],
            )
        ),
        sn_reciprocal_sine_max_dimension=int(
            profile_config.get("sn_reciprocal_sine_max_dimension", 8)
        ),
        sn_reciprocal_sine_max_numerator_degree=int(
            profile_config.get("sn_reciprocal_sine_max_numerator_degree", 5)
        ),
        sn_reciprocal_sine_value_pool_size=int(
            profile_config.get("sn_reciprocal_sine_value_pool_size", 64)
        ),
        sn_reciprocal_sine_shortlist_size=int(
            profile_config.get("sn_reciprocal_sine_shortlist_size", 16)
        ),
        sn_reciprocal_sine_proposal_limit=int(
            profile_config.get("sn_reciprocal_sine_proposal_limit", 32)
        ),
        sn_inverse_cosine_phase_signs=tuple(
            int(value)
            for value in profile_config.get("sn_inverse_cosine_phase_signs", [1, -1])
        ),
        sn_inverse_cosine_radial_signs=tuple(
            int(value)
            for value in profile_config.get("sn_inverse_cosine_radial_signs", [1, -1])
        ),
        sn_inverse_cosine_phase_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_inverse_cosine_phase_scales", [0.5, 1.0, 2.0]
            )
        ),
        sn_inverse_cosine_max_dimension=int(
            profile_config.get("sn_inverse_cosine_max_dimension", 6)
        ),
        sn_inverse_cosine_value_pool_size=int(
            profile_config.get("sn_inverse_cosine_value_pool_size", 64)
        ),
        sn_inverse_cosine_shortlist_size=int(
            profile_config.get("sn_inverse_cosine_shortlist_size", 16)
        ),
        sn_inverse_cosine_proposal_limit=int(
            profile_config.get("sn_inverse_cosine_proposal_limit", 32)
        ),
        sn_multiaxis_pair_signs=tuple(
            int(value)
            for value in profile_config.get("sn_multiaxis_pair_signs", [1, -1])
        ),
        sn_multiaxis_radial_term_counts=tuple(
            int(value)
            for value in profile_config.get("sn_multiaxis_radial_term_counts", [2, 3])
        ),
        sn_multiaxis_max_dimension=int(
            profile_config.get("sn_multiaxis_max_dimension", 10)
        ),
        sn_multiaxis_max_numerator_degree=int(
            profile_config.get("sn_multiaxis_max_numerator_degree", 3)
        ),
        sn_multiaxis_value_pool_size=int(
            profile_config.get("sn_multiaxis_value_pool_size", 64)
        ),
        sn_multiaxis_shortlist_size=int(
            profile_config.get("sn_multiaxis_shortlist_size", 16)
        ),
        sn_multiaxis_proposal_limit=int(
            profile_config.get("sn_multiaxis_proposal_limit", 32)
        ),
        sn_interference_sine_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_interference_sine_scales",
                [0.5, 1.0, 2.0, float(np.pi), float(2.0 * np.pi)],
            )
        ),
        sn_interference_sine_max_dimension=int(
            profile_config.get("sn_interference_sine_max_dimension", 8)
        ),
        sn_interference_sine_value_pool_size=int(
            profile_config.get("sn_interference_sine_value_pool_size", 64)
        ),
        sn_interference_sine_shortlist_size=int(
            profile_config.get("sn_interference_sine_shortlist_size", 16)
        ),
        sn_interference_sine_proposal_limit=int(
            profile_config.get("sn_interference_sine_proposal_limit", 32)
        ),
        sn_sparse_radical_offsets=tuple(
            float(value)
            for value in profile_config.get("sn_sparse_radical_offsets", [1.0])
        ),
        sn_sparse_radical_scales=tuple(
            float(value)
            for value in profile_config.get("sn_sparse_radical_scales", [0.5, 1.0, 2.0])
        ),
        sn_sparse_radical_max_dimension=int(
            profile_config.get("sn_sparse_radical_max_dimension", 8)
        ),
        sn_sparse_radical_max_abs_exponent=int(
            profile_config.get("sn_sparse_radical_max_abs_exponent", 4)
        ),
        sn_sparse_radical_max_numerator_degree=int(
            profile_config.get("sn_sparse_radical_max_numerator_degree", 5)
        ),
        sn_sparse_radical_max_denominator_degree=int(
            profile_config.get("sn_sparse_radical_max_denominator_degree", 9)
        ),
        sn_sparse_radical_integer_candidate_limit=int(
            profile_config.get("sn_sparse_radical_integer_candidate_limit", 1024)
        ),
        sn_sparse_radical_value_pool_size=int(
            profile_config.get("sn_sparse_radical_value_pool_size", 64)
        ),
        sn_sparse_radical_shortlist_size=int(
            profile_config.get("sn_sparse_radical_shortlist_size", 16)
        ),
        sn_sparse_radical_proposal_limit=int(
            profile_config.get("sn_sparse_radical_proposal_limit", 32)
        ),
        sn_shared_ratio_trig_phase_scales=tuple(
            float(value)
            for value in profile_config.get(
                "sn_shared_ratio_trig_phase_scales",
                [0.5, 1.0, 2.0, float(np.pi), float(2.0 * np.pi)],
            )
        ),
        sn_shared_ratio_trig_required_dimension=int(
            profile_config.get("sn_shared_ratio_trig_required_dimension", 7)
        ),
        sn_shared_ratio_trig_value_pool_size=int(
            profile_config.get("sn_shared_ratio_trig_value_pool_size", 64)
        ),
        sn_shared_ratio_trig_shortlist_size=int(
            profile_config.get("sn_shared_ratio_trig_shortlist_size", 16)
        ),
        sn_shared_ratio_trig_proposal_limit=int(
            profile_config.get("sn_shared_ratio_trig_proposal_limit", 32)
        ),
        sn_deep_accuracy_parameters=deep_accuracy_parameters,
        sn_archive_beam_diversity_slots=int(
            profile_config.get("sn_archive_beam_diversity_slots", 0)
        ),
        sn_archive_beam_refine_max_steps=int(
            profile_config.get("sn_archive_beam_refine_max_steps", 4)
        ),
        sn_coverage_shortlist_size=int(
            profile_config.get("sn_coverage_shortlist_size", 32)
        ),
        sn_coverage_elite_slots=int(profile_config.get("sn_coverage_elite_slots", 1)),
        sn_coverage_max_base_reward_gap=float(
            profile_config.get("sn_coverage_max_base_reward_gap", 0.001)
        ),
        sn_coverage_crossover_rate=float(
            profile_config.get("sn_coverage_crossover_rate", 0.05)
        ),
        sn_child_geometry_enabled=profile_config.get("sn_child_geometry_enabled"),
        sn_basis_exchange_enabled=profile_config.get("sn_basis_exchange_enabled"),
        sn_basis_infusion_enabled=profile_config.get("sn_basis_infusion_enabled"),
        sn_residual_basis_infusion_enabled=profile_config.get(
            "sn_residual_basis_infusion_enabled"
        ),
        sn_basis_archive_enabled=profile_config.get("sn_basis_archive_enabled"),
        sn_partial_residual_archive_enabled=profile_config.get(
            "sn_partial_residual_archive_enabled"
        ),
        sn_basis_pursuit_enabled=profile_config.get("sn_basis_pursuit_enabled"),
        sn_basis_pursuit_shortlist_size=int(
            profile_config.get("sn_basis_pursuit_shortlist_size", 4)
        ),
        sn_pursuit_stagnation_patience=int(
            profile_config.get("sn_pursuit_stagnation_patience", 5)
        ),
        sn_pursuit_stagnation_epsilon_abs=float(
            profile_config.get("sn_pursuit_stagnation_epsilon_abs", 1e-6)
        ),
        sn_pursuit_stagnation_epsilon_rel=float(
            profile_config.get("sn_pursuit_stagnation_epsilon_rel", 1e-4)
        ),
        sn_pursuit_preserve_fallback_anchor=bool(
            profile_config.get("sn_pursuit_preserve_fallback_anchor", False)
        ),
        sn_pursuit_select_by_coverage=bool(
            profile_config.get("sn_pursuit_select_by_coverage", False)
        ),
        sn_basis_replacement_enabled=profile_config.get("sn_basis_replacement_enabled"),
        sn_basis_replacement_removal_shortlist_size=int(
            profile_config.get("sn_basis_replacement_removal_shortlist_size", 2)
        ),
        sn_conditional_basis_replacement_enabled=profile_config.get(
            "sn_conditional_basis_replacement_enabled"
        ),
        sn_orthogonal_basis_crossover_enabled=profile_config.get(
            "sn_orthogonal_basis_crossover_enabled"
        ),
        sn_orthogonal_basis_max_steps=int(
            profile_config.get("sn_orthogonal_basis_max_steps", 8)
        ),
        sn_shadow_slot_enabled=profile_config.get("sn_shadow_slot_enabled"),
        sn_shadow_max_slots=int(profile_config.get("sn_shadow_max_slots", 1)),
        sn_shadow_minimum_coverage_gain=float(
            profile_config.get("sn_shadow_minimum_coverage_gain", 0.0)
        ),
        sn_shadow_allow_represented_amplification=bool(
            profile_config.get("sn_shadow_allow_represented_amplification", False)
        ),
        sn_basis_forest_crossover_rate=float(
            profile_config.get("sn_basis_forest_crossover_rate", 0.10)
        ),
        sn_basis_forest_mutation_recombination_enabled=profile_config.get(
            "sn_basis_forest_mutation_recombination_enabled"
        ),
        sn_basis_forest_mutation_shadow_enabled=profile_config.get(
            "sn_basis_forest_mutation_shadow_enabled"
        ),
        sn_basis_archive_capacity=int(
            profile_config.get("sn_basis_archive_capacity", 64)
        ),
        sn_basis_archive_source_candidates=int(
            profile_config.get("sn_basis_archive_source_candidates", 8)
        ),
        sn_basis_quality_archive_enabled=profile_config.get(
            "sn_basis_quality_archive_enabled"
        ),
        sn_basis_quality_archive_capacity=int(
            profile_config.get("sn_basis_quality_archive_capacity", 64)
        ),
        term_cache_max_entries=int(algorithm["term_cache_max_entries"]),
        term_cache_max_memory_bytes=int(algorithm["term_cache_max_memory_bytes"]),
        derivative_cache_max_entries=int(algorithm["derivative_cache_max_entries"]),
        geometry_cache_max_entries=int(algorithm["geometry_cache_max_entries"]),
        geometry_cache_max_memory_bytes=int(
            algorithm["geometry_cache_max_memory_bytes"]
        ),
        base_cache_max_entries=int(algorithm["base_cache_max_entries"]),
        dataset_identity=f"{args.dataset}|seed={args.seed}",
        initial_population_expressions=expressions,
        output_dir=(
            args.output
            if args.profile
            in {
                PROFILE_SN_V2,
                PROFILE_SN_STABLE,
                PROFILE_SN_REPAIR_COVERAGE,
                PROFILE_SN_REPAIR_SHADOW,
                PROFILE_SN_ARCHIVE_EXPORT,
                PROFILE_SN_ARCHIVE_ANCHOR_EXPORT,
                PROFILE_SN_ARCHIVE_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_DIVERSE_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_DUAL_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_DUAL_POOL_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_REFINED_DUAL_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_FLOATING_DUAL_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_BALANCED_SCREEN_DUAL_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_BALANCED_FLOATING_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_INNOVATION_SCREEN_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_PAIR_LOOKAHEAD_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_PRODUCT_LIFT_BEAM_EXPORT,
                PROFILE_SN_ARCHIVE_PRODUCT_ACCURACY_EXPORT,
                PROFILE_SN_ARCHIVE_CONDITIONAL_PRODUCT_ACCURACY_EXPORT,
                PROFILE_SN_ARCHIVE_ITERATED_CONDITIONAL_PRODUCT_ACCURACY_EXPORT,
                PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
                PROFILE_SN_BASIS_EXCHANGE,
                PROFILE_SN_BASIS_INFUSION,
                PROFILE_SN_BASIS_ARCHIVE,
                PROFILE_SN_PARTIAL_RESIDUAL_ARCHIVE,
                PROFILE_SN_SCREENED_BASIS_PURSUIT,
                PROFILE_SN_SCREENED_PURSUIT_SHADOW,
                PROFILE_SN_PARETO_PURSUIT_SHADOW,
                PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
                PROFILE_SN_BIDIRECTIONAL_BASIS_REPLACEMENT,
                PROFILE_SN_CONDITIONAL_BASIS_REPLACEMENT,
                PROFILE_SN_ORTHOGONAL_BASIS_CROSSOVER,
                PROFILE_SN_DUAL_PATH_SHADOW_SLOT,
                PROFILE_SN_RESIDUAL_BASIS_INFUSION,
                PROFILE_SN_CHILD_GEOMETRY,
                PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                PROFILE_SN_VERIFIED_REPAIR,
                PROFILE_BASIS_FOREST_BASE,
                PROFILE_BASIS_FOREST_SN_CROSSOVER,
                PROFILE_BASIS_FOREST_SN_RESIDUAL_INFUSION,
                PROFILE_BASIS_FOREST_SN_SAFE_MUTATION,
                PROFILE_BASIS_FOREST_SN_VERIFIED_COMPRESSION,
                PROFILE_BASIS_FOREST_SN_CONSTRUCT_COMPRESS,
                PROFILE_BASIS_FOREST_SN_RESIDUAL_SHADOW,
                PROFILE_BASIS_FOREST_SN_REPAIR_COVERAGE,
                PROFILE_BASIS_FOREST_SN_COMPRESS_SHADOW,
                PROFILE_BASIS_FOREST_SN_MUTATION_RECOMBINATION,
                PROFILE_BASIS_FOREST_SN_MUTATION_SHADOW,
                PROFILE_BASIS_FOREST_SN_COMPRESS_MUTATION_SHADOW,
                PROFILE_BASIS_FOREST_SN_COMPRESS_MULTISOURCE_SHADOW,
            }
            else None
        ),
        detailed_logging=bool(profile_config.get("detailed_logging", False)),
        record_integrity_metadata=False,
        base_score_semantics=profile_config.get("base_score_semantics"),
    )

    previous_alarm = signal.getsignal(signal.SIGALRM)

    def hard_guard(_signum, _frame):
        raise HardGuard(f"Hard guard {hard_time_limit}s reached")

    signal.signal(signal.SIGALRM, hard_guard)
    signal.setitimer(signal.ITIMER_REAL, hard_time_limit)
    failure = None
    returncode = 0
    try:
        model.fit(X, y)
    except HardGuard as error:
        model.status = "hard_time_limit"
        failure = str(error)
        returncode = 2
    except BaseException as error:
        model.status = "error"
        failure = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        returncode = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)

    result = model.result_document()
    result.update(
        {
            "protocol": f"{config['protocol_version']}_result",
            "task_index": args.task_index,
            "dataset": args.dataset,
            "selected_rows": int(len(y)),
            "physical_row_ids": row_ids.tolist(),
            "target_generation": args.target_generation,
            "failure": failure,
            "finished_unix": time.time(),
        }
    )
    write_json(args.output / "result.json", result)
    selected = result.get("exported_candidate")
    if (
        returncode == 0
        and result.get("status") == "iter_limit"
        and int(result.get("completed_generations", -1)) == args.target_generation + 1
        and isinstance(selected, dict)
    ):
        evaluation = evaluate_expression(
            str(selected["exported_expression"]), X, y, feature_names
        )
        evaluation.update(
            {
                "task_index": args.task_index,
                "dataset": args.dataset,
                "seed": args.seed,
                "profile": args.profile,
                "target_generation": args.target_generation,
            }
        )
        if args.profile == PROFILE_SN_STABLE:
            sn_candidate = result["best_by_sn_comparator"]
            sn_evaluation = evaluate_expression(
                str(sn_candidate["exported_expression"]),
                X,
                y,
                feature_names,
            )
            evaluation.update(
                {
                    "base_anchor_expression": evaluation["exported_expression"],
                    "base_anchor_exported_search_r2": evaluation["exported_search_r2"],
                    "sn_structure_expression": sn_evaluation["exported_expression"],
                    "sn_structure_complexity": sn_evaluation["complexity"],
                    "sn_structure_exported_search_r2": sn_evaluation[
                        "exported_search_r2"
                    ],
                    "sn_structure_internal_r2": sn_candidate.get("search_internal_r2"),
                    "sn_structure_base_reward": sn_candidate.get("base_reward"),
                    "anchor_minus_sn_structure_r2": (
                        evaluation["exported_search_r2"]
                        - sn_evaluation["exported_search_r2"]
                    ),
                    "anchor_recovered_negative_sn": bool(
                        sn_evaluation["exported_search_r2"] < 0
                        and evaluation["exported_search_r2"] >= 0
                    ),
                }
            )
        write_json(args.output / "evaluation.json", evaluation)
    else:
        returncode = returncode or 2
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
