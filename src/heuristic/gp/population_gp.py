"""Controlled population GP with optional elite-only Sobolev novelty guidance.

The genetic operators come from :mod:`src.nd2py.nd2py.search.gp.gp`.  This
adapter deliberately owns generation orchestration because the research
protocol needs counter-keyed RNG, additive linear phenotypes, complete-
generation checkpoints, and a strict wall-time floor.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import resource
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sympy as sp

from ...nd2py import nd2py as nd
from ...nd2py.nd2py.search.gp.gp import GP as NDPopulationGP
from ...nd2py.nd2py.search.gp.gp import Individual as NDIndividual
from ...sobolev import (
    CandidateGeometryCache,
    PruningConfig,
    RefitGeometryHint,
    RefitResult,
    SobolevConfig,
    SobolevEvaluator,
    TermEvaluationCache,
    prune_and_refit,
    sobolev_penalty,
)
from ...sobolev.decomposition import (
    decompose_expand_mul,
    parse_expression,
    to_project_expression_string,
)
from ...sobolev.signature import array_identity, evaluate_sympy, select_geometry_indices
from ...sobolev.types import EvaluationResult, TermSpec
from .additive import AdditiveFitResult, AdditiveLinearEvaluator
from .basis_forest import (
    BasisGenome,
    build_basis_tree_direct,
    extract_strict_or_opaque_basis_genome,
    opaque_basis_genome,
    select_uniform_basis_crossover,
)
from .checkpoint import (
    GenerationCheckpointWriter,
    append_jsonl,
    atomic_write_json,
    canonical_json_bytes,
)
from .rng import CounterRNG
from .sn_basis_exchange import BasisExchangeChoice, select_basis_exchange
from .sn_basis_infusion import BasisInfusionChoice, select_basis_infusion
from .sn_basis_forest_mutation import select_impact_aware_basis_gene
from .sn_basis_archive import (
    ArchiveInfusionChoice,
    BasisArchiveEntry,
    SourceQualityBasisArchive,
    SobolevBasisArchive,
    partial_residual_credit,
    select_archive_residual_infusion,
)
from .sn_archive_export import (
    build_archive_beam_tree,
    build_archive_export_prefixes,
    rank_archive_floating_deletions,
    screen_archive_beam_expansions,
    screen_archive_beam_pair_expansions,
    select_archive_beam_retention,
)
from .sn_archive_interactions import (
    lift_archive_conditional_affine_unary_interactions,
    lift_archive_conditional_product_interactions,
    lift_archive_conditional_radial_interactions,
    lift_archive_conditional_rational_interactions,
    lift_archive_conditional_unary_interactions,
    lift_archive_product_interactions,
)
from .sn_basis_pursuit import (
    ScreenedArchiveDonor,
    screen_archive_basis_donors,
    select_low_impact_parent_term,
)
from .sn_basis_replacement import (
    RemovalCandidate,
    rank_low_impact_parent_terms,
)
from .sn_conditional_replacement import screen_conditional_replacements
from .sn_direct_composition import lift_direct_conditional_phase_features
from .sn_phase_interactions import lift_direct_conditional_phase_interactions
from .sn_radical_composition import (
    compact_positive_amplitude_radical_expression,
    lift_direct_amplitude_conditioned_radicals,
)
from .sn_shared_phase_rational import lift_direct_shared_phase_rationals
from .sn_exponential_factorization import lift_direct_damped_exponentials
from .sn_affine_gaussian import lift_direct_affine_gaussians
from .sn_sinc_factorization import lift_direct_sinc_squared_factors
from .sn_shared_unary_polynomial import lift_direct_shared_unary_polynomials
from .sn_reciprocal_trig import lift_direct_reciprocal_trig_pairs
from .sn_cross_unary_affine import lift_direct_cross_unary_affine_pairs
from .sn_shared_denominator import lift_direct_shared_denominators
from .sn_relativistic_rational import lift_direct_relativistic_rationals
from .sn_cosine_law_radial import lift_direct_cosine_law_radials
from .sn_reciprocal_sine_square import lift_direct_reciprocal_sine_squares
from .sn_inverse_cosine_radial import lift_direct_inverse_cosine_radials
from .sn_multiaxis_inverse_square import lift_direct_multiaxis_inverse_squares
from .sn_interference_sine_ratio import lift_direct_interference_sine_ratios
from .sn_sparse_radical import lift_direct_sparse_radicals
from .sn_shared_ratio_trig_polynomial import (
    lift_direct_shared_ratio_trig_polynomials,
)
from .sn_deep_accuracy_envelope import (
    DEEP_ACCURACY_PARAMETER_DEFAULTS,
    DEEP_ACCURACY_PARAMETER_NAMES,
    DEEP_ACCURACY_PROFILE_NAMES,
    DEEP_ACCURACY_STAGE_BY_CODE,
    DEEP_ACCURACY_STAGE_BY_PROFILE,
    DeepAccuracyStage,
    PROFILE_SN_ADDITIVE_RADIAL_COUPLING_ACCURACY_EXPORT,
    PROFILE_SN_CONDITIONAL_FACTORED_POLYNOMIAL_ACCURACY_EXPORT,
    PROFILE_SN_COUPLED_TRIG_RATIONAL_ACCURACY_EXPORT,
    PROFILE_SN_INVERSE_TRIG_MOBIUS_ACCURACY_EXPORT,
    PROFILE_SN_NESTED_AFFINE_RADICAL_ACCURACY_EXPORT,
    PROFILE_SN_PHASE_MODULATED_RATIONAL_ACCURACY_EXPORT,
    PROFILE_SN_RELATIVISTIC_TRIG_RATIONAL_ACCURACY_EXPORT,
    proposal_trace_document,
    stage_lift_arguments,
)
from .sn_orthogonal_basis_crossover import (
    BasisPoolEntry,
    orthogonal_basis_pursuit,
)
from .sn_comparator import (
    DEFAULT_SN_COMPARE_EPSILON_ABS,
    DEFAULT_SN_COMPARE_EPSILON_REL,
    ComparisonResult,
    LazySobolevView,
    SNComparator,
)
from .sn_impact import (
    DEFAULT_SN_MUTATION_IMPACT_BETA,
    DEFAULT_SN_MUTATION_IMPACT_EPSILON,
    DEFAULT_SN_MUTATION_MAX_NORMALIZED_IMPACT,
)
from .sn_mutation_selector import (
    DEFAULT_SN_MUTATION_DELTA,
    DEFAULT_SN_MUTATION_GAMMA,
    MutationSiteSelection,
    SNMutationTargetSelector,
)
from .sn_population_coverage import (
    normalized_fitted_signature,
    orthonormal_signature_span,
    select_coverage_candidate,
    signature_residual_gain,
)
from .sn_residual_infusion import (
    ResidualInfusionChoice,
    centered_absolute_correlation,
    select_residual_basis_infusion,
)
from .sn_safety_anchor import merge_anchor_with_sn_elites, select_base_anchor
from .sn_shadow_slot import (
    ShadowProposal,
    plan_shadow_injection,
)
from .term_provenance import ProvenanceResult, TermProvenanceBuilder, path_text


PROFILE_BASE = "base_gp"
PROFILE_SN = "sn_gp_elite_incremental_prune"
PROFILE_SN_V2 = "sn_gp_v2"
PROFILE_SN_STABLE = "sn_gp_stable"
PROFILE_BASE_EXPORT_ALIGNED = "base_gp_export_aligned"
PROFILE_SN_VERIFIED_REPAIR = "sn_gp_verified_repair"
PROFILE_SN_REPAIR_SHADOW = "sn_gp_repair_shadow"
PROFILE_SN_ARCHIVE_EXPORT = "sn_gp_archive_recombination_export"
PROFILE_SN_ARCHIVE_ANCHOR_EXPORT = "sn_gp_archive_anchor_export"
PROFILE_SN_ARCHIVE_BEAM_EXPORT = "sn_gp_archive_beam_export"
PROFILE_SN_ARCHIVE_DIVERSE_BEAM_EXPORT = "sn_gp_archive_diverse_beam_export"
PROFILE_SN_ARCHIVE_DUAL_BEAM_EXPORT = "sn_gp_archive_dual_beam_export"
PROFILE_SN_ARCHIVE_DUAL_POOL_BEAM_EXPORT = "sn_gp_archive_dual_pool_beam_export"
PROFILE_SN_ARCHIVE_REFINED_DUAL_BEAM_EXPORT = "sn_gp_archive_refined_dual_beam_export"
PROFILE_SN_ARCHIVE_FLOATING_DUAL_BEAM_EXPORT = "sn_gp_archive_floating_dual_beam_export"
PROFILE_SN_ARCHIVE_BALANCED_SCREEN_DUAL_BEAM_EXPORT = (
    "sn_gp_archive_balanced_screen_dual_beam_export"
)
PROFILE_SN_ARCHIVE_BALANCED_FLOATING_BEAM_EXPORT = (
    "sn_gp_archive_balanced_floating_beam_export"
)
PROFILE_SN_ARCHIVE_INNOVATION_SCREEN_BEAM_EXPORT = (
    "sn_gp_archive_innovation_screen_beam_export"
)
PROFILE_SN_ARCHIVE_PAIR_LOOKAHEAD_BEAM_EXPORT = (
    "sn_gp_archive_pair_lookahead_beam_export"
)
PROFILE_SN_ARCHIVE_PRODUCT_LIFT_BEAM_EXPORT = "sn_gp_archive_product_lift_beam_export"
PROFILE_SN_ARCHIVE_PRODUCT_ACCURACY_EXPORT = "sn_gp_archive_product_accuracy_export"
PROFILE_SN_ARCHIVE_CONDITIONAL_PRODUCT_ACCURACY_EXPORT = (
    "sn_gp_archive_conditional_product_accuracy_export"
)
PROFILE_SN_ARCHIVE_ITERATED_CONDITIONAL_PRODUCT_ACCURACY_EXPORT = (
    "sn_gp_archive_iterated_conditional_product_accuracy_export"
)
PROFILE_SN_ARCHIVE_CONDITIONAL_UNARY_ACCURACY_EXPORT = (
    "sn_gp_archive_conditional_unary_accuracy_export"
)
PROFILE_SN_ARCHIVE_CONDITIONAL_RATIONAL_ACCURACY_EXPORT = (
    "sn_gp_archive_conditional_rational_accuracy_export"
)
PROFILE_SN_ARCHIVE_CONDITIONAL_AFFINE_UNARY_ACCURACY_EXPORT = (
    "sn_gp_archive_conditional_affine_unary_accuracy_export"
)
PROFILE_SN_ARCHIVE_CONDITIONAL_RADIAL_ACCURACY_EXPORT = (
    "sn_gp_archive_conditional_radial_accuracy_export"
)
PROFILE_SN_ARCHIVE_FEATURE_RADIAL_ACCURACY_EXPORT = (
    "sn_gp_archive_feature_radial_accuracy_export"
)
PROFILE_SN_DIRECT_FEATURE_RADIAL_ACCURACY_EXPORT = (
    "sn_gp_direct_feature_radial_accuracy_export"
)
PROFILE_SN_DIRECT_FEATURE_AFFINE_UNARY_ACCURACY_EXPORT = (
    "sn_gp_direct_feature_affine_unary_accuracy_export"
)
PROFILE_SN_DIRECT_FEATURE_PRODUCT_ACCURACY_EXPORT = (
    "sn_gp_direct_feature_product_accuracy_export"
)
PROFILE_SN_DIRECT_PHASE_ACCURACY_EXPORT = "sn_gp_direct_phase_accuracy_export"
PROFILE_SN_DIRECT_PHASE_INTERACTION_ACCURACY_EXPORT = (
    "sn_gp_direct_phase_interaction_accuracy_export"
)
PROFILE_SN_DIRECT_RADICAL_PHASE_ACCURACY_EXPORT = (
    "sn_gp_direct_radical_phase_accuracy_export"
)
PROFILE_SN_SHARED_PHASE_RATIONAL_ACCURACY_EXPORT = (
    "sn_gp_shared_phase_rational_accuracy_export"
)
PROFILE_SN_DAMPED_EXPONENTIAL_ACCURACY_EXPORT = (
    "sn_gp_damped_exponential_accuracy_export"
)
PROFILE_SN_AFFINE_GAUSSIAN_ACCURACY_EXPORT = "sn_gp_affine_gaussian_accuracy_export"
PROFILE_SN_SINC_SQUARED_ACCURACY_EXPORT = "sn_gp_sinc_squared_accuracy_export"
PROFILE_SN_SHARED_UNARY_POLYNOMIAL_ACCURACY_EXPORT = (
    "sn_gp_shared_unary_polynomial_accuracy_export"
)
PROFILE_SN_RECIPROCAL_TRIG_ACCURACY_EXPORT = "sn_gp_reciprocal_trig_accuracy_export"
PROFILE_SN_CROSS_UNARY_AFFINE_ACCURACY_EXPORT = (
    "sn_gp_cross_unary_affine_accuracy_export"
)
PROFILE_SN_SHARED_DENOMINATOR_ACCURACY_EXPORT = (
    "sn_gp_shared_denominator_accuracy_export"
)
PROFILE_SN_RELATIVISTIC_RATIONAL_ACCURACY_EXPORT = (
    "sn_gp_relativistic_rational_accuracy_export"
)
PROFILE_SN_COSINE_LAW_RADIAL_ACCURACY_EXPORT = "sn_gp_cosine_law_radial_accuracy_export"
PROFILE_SN_RECIPROCAL_SINE_SQUARE_ACCURACY_EXPORT = (
    "sn_gp_reciprocal_sine_square_accuracy_export"
)
PROFILE_SN_INVERSE_COSINE_RADIAL_ACCURACY_EXPORT = (
    "sn_gp_inverse_cosine_radial_accuracy_export"
)
PROFILE_SN_MULTIAXIS_INVERSE_SQUARE_ACCURACY_EXPORT = (
    "sn_gp_multiaxis_inverse_square_accuracy_export"
)
PROFILE_SN_INTERFERENCE_SINE_RATIO_ACCURACY_EXPORT = (
    "sn_gp_interference_sine_ratio_accuracy_export"
)
PROFILE_SN_SPARSE_RADICAL_ACCURACY_EXPORT = "sn_gp_sparse_radical_accuracy_export"
PROFILE_SN_SHARED_RATIO_TRIG_POLYNOMIAL_ACCURACY_EXPORT = (
    "sn_gp_shared_ratio_trig_polynomial_accuracy_export"
)
PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT = (
    "sn_gp_archive_value_only_accuracy_export"
)
PROFILE_SN_REPAIR_COVERAGE = "sn_gp_repair_coverage"
PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER = "sn_gp_verified_coverage_crossover"
PROFILE_SN_CHILD_GEOMETRY = "sn_gp_child_geometry"
PROFILE_SN_BASIS_EXCHANGE = "sn_gp_basis_exchange"
PROFILE_SN_BASIS_INFUSION = "sn_gp_basis_infusion"
PROFILE_SN_RESIDUAL_BASIS_INFUSION = "sn_gp_residual_basis_infusion"
PROFILE_SN_BASIS_ARCHIVE = "sn_gp_basis_archive"
PROFILE_SN_PARTIAL_RESIDUAL_ARCHIVE = "sn_gp_partial_residual_archive"
PROFILE_SN_SCREENED_BASIS_PURSUIT = "sn_gp_screened_basis_pursuit"
PROFILE_SN_SCREENED_PURSUIT_SHADOW = "sn_gp_screened_pursuit_shadow"
PROFILE_SN_PARETO_PURSUIT_SHADOW = "sn_gp_pareto_pursuit_shadow"
PROFILE_SN_STAGNATION_PURSUIT_SHADOW = "sn_gp_stagnation_pursuit_shadow"
PROFILE_SN_BIDIRECTIONAL_BASIS_REPLACEMENT = "sn_gp_bidirectional_basis_replacement"
PROFILE_SN_CONDITIONAL_BASIS_REPLACEMENT = "sn_gp_conditional_basis_replacement"
PROFILE_SN_ORTHOGONAL_BASIS_CROSSOVER = "sn_gp_orthogonal_basis_crossover"
PROFILE_SN_DUAL_PATH_SHADOW_SLOT = "sn_gp_dual_path_shadow_slot"
PROFILE_BASIS_FOREST_BASE = "basis_forest_base"
PROFILE_BASIS_FOREST_SN_CROSSOVER = "basis_forest_sn_crossover"
PROFILE_BASIS_FOREST_SN_RESIDUAL_INFUSION = "basis_forest_sn_residual_infusion"
PROFILE_BASIS_FOREST_SN_SAFE_MUTATION = "basis_forest_sn_safe_mutation"
PROFILE_BASIS_FOREST_SN_VERIFIED_COMPRESSION = "basis_forest_sn_verified_compression"
PROFILE_BASIS_FOREST_SN_CONSTRUCT_COMPRESS = "basis_forest_sn_construct_compress"
PROFILE_BASIS_FOREST_SN_RESIDUAL_SHADOW = "basis_forest_sn_residual_shadow"
PROFILE_BASIS_FOREST_SN_REPAIR_COVERAGE = "basis_forest_sn_repair_coverage"
PROFILE_BASIS_FOREST_SN_COMPRESS_SHADOW = "basis_forest_sn_compress_shadow"
PROFILE_BASIS_FOREST_SN_MUTATION_RECOMBINATION = (
    "basis_forest_sn_mutation_recombination"
)
PROFILE_BASIS_FOREST_SN_MUTATION_SHADOW = "basis_forest_sn_mutation_shadow"
PROFILE_BASIS_FOREST_SN_COMPRESS_MUTATION_SHADOW = (
    "basis_forest_sn_compress_mutation_shadow"
)
PROFILE_BASIS_FOREST_SN_COMPRESS_MULTISOURCE_SHADOW = (
    "basis_forest_sn_compress_multisource_shadow"
)
PROFILES = (
    PROFILE_BASE,
    PROFILE_SN,
    PROFILE_SN_V2,
    PROFILE_SN_STABLE,
    PROFILE_BASE_EXPORT_ALIGNED,
    PROFILE_SN_VERIFIED_REPAIR,
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
    *DEEP_ACCURACY_PROFILE_NAMES,
    PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
    PROFILE_SN_REPAIR_COVERAGE,
    PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
    PROFILE_SN_CHILD_GEOMETRY,
    PROFILE_SN_BASIS_EXCHANGE,
    PROFILE_SN_BASIS_INFUSION,
    PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
)

DEFAULT_BINARY = (nd.Mul, nd.Div, nd.Add, nd.Sub)
DEFAULT_UNARY = (
    nd.Sqrt,
    nd.Cos,
    nd.Sin,
    nd.Pow2,
    nd.Pow3,
    nd.Exp,
    nd.Inv,
    nd.Neg,
    nd.Arcsin,
    nd.Arccos,
    nd.Cot,
    nd.Log,
    nd.Tanh,
)
DEFAULT_FIXED_CONSTANTS = (1.0, 2.0, float(np.pi))


class GenerationDeadline(BaseException):
    """The floor deadline landed inside an uncommitted generation."""


class PruneWritebackNotRepresentable(ValueError):
    """A fitted pruning decision cannot persist as a controlled GP genotype."""


class ControlledIndividual(NDIndividual):
    """One GP genotype and its fitted additive phenotype/audit state."""

    def __init__(
        self,
        eqtree: nd.Symbol,
        *,
        candidate_id: int,
        generation: int,
        slot: int,
        parent_ids: Sequence[int] = (),
        variation_type: str = "initial",
    ) -> None:
        super().__init__(eqtree)
        self.candidate_id = int(candidate_id)
        self.generation = int(generation)
        self.slot = int(slot)
        self.parent_ids = tuple(int(value) for value in parent_ids)
        self.variation_type = str(variation_type)
        self.fit_result: AdditiveFitResult | None = None
        self.base_reward = 0.0
        self.final_reward: float | None = None
        self.phi: nd.Symbol | None = None
        self.canonical_fitted_expression = ""
        self.sobolev_result: EvaluationResult | None = None
        self.sobolev_penalty: float | None = None
        self.sobolev_success: bool | None = None
        self.shortlist_status = "not_considered"
        self.base_rank: int | None = None
        self.structural_rank: int | None = None
        self.parent_geometry_key: str | None = None
        self.parent_geometry_hints: tuple[
            tuple[int, str | None, tuple[str, ...]], ...
        ] = ()
        self.pruning_result = None
        self.prune_origin_candidate_id: int | None = None
        self.sobolev_evaluated = False
        self.sn_cache_identity: tuple[str, str] | None = None
        self.sobolev_evaluation_failure: str | None = None
        self.term_provenance: ProvenanceResult | None = None
        self.provenance_cache_expression: str | None = None
        self.sn_targeted_mutation = False
        self.basis_genome: BasisGenome | None = None
        self.basis_forest_status = "not_assigned"

    def copy(self) -> "ControlledIndividual":
        value = ControlledIndividual(
            self.eqtree.copy(),
            candidate_id=self.candidate_id,
            generation=self.generation,
            slot=self.slot,
            parent_ids=self.parent_ids,
            variation_type=self.variation_type,
        )
        value.complexity = self.complexity
        value.accuracy = self.accuracy
        value.fitness = self.fitness
        value.fit_result = None if self.fit_result is None else self.fit_result.clone()
        value.base_reward = self.base_reward
        value.final_reward = self.final_reward
        value.phi = None if self.phi is None else self.phi.copy()
        value.canonical_fitted_expression = self.canonical_fitted_expression
        value.sobolev_result = copy.deepcopy(self.sobolev_result)
        value.sobolev_penalty = self.sobolev_penalty
        value.sobolev_success = self.sobolev_success
        value.shortlist_status = self.shortlist_status
        value.base_rank = self.base_rank
        value.structural_rank = self.structural_rank
        value.parent_geometry_key = self.parent_geometry_key
        value.parent_geometry_hints = self.parent_geometry_hints
        value.pruning_result = copy.deepcopy(self.pruning_result)
        value.prune_origin_candidate_id = self.prune_origin_candidate_id
        value.sobolev_evaluated = self.sobolev_evaluated
        value.sn_cache_identity = self.sn_cache_identity
        value.sobolev_evaluation_failure = self.sobolev_evaluation_failure
        value.term_provenance = copy.deepcopy(self.term_provenance)
        value.provenance_cache_expression = self.provenance_cache_expression
        value.sn_targeted_mutation = self.sn_targeted_mutation
        value.basis_genome = copy.deepcopy(self.basis_genome)
        value.basis_forest_status = self.basis_forest_status
        return value

    @property
    def raw_expression(self) -> str:
        return self.eqtree.to_str(number_format=".17g")

    @property
    def valid(self) -> bool:
        return bool(self.fit_result is not None and self.fit_result.success)

    @property
    def term_keys(self) -> tuple[str, ...]:
        if self.fit_result is None:
            return ()
        return tuple(term.canonical for term in self.fit_result.terms)

    def apply_fit(self, result: AdditiveFitResult) -> None:
        self.fit_result = result
        self.base_reward = float(result.base_reward)
        self.fitness = self.base_reward
        self.accuracy = float(result.mse)
        self.complexity = int(result.complexity) if result.success else math.inf
        self.phi = None if result.fitted_tree is None else result.fitted_tree.copy()
        self.canonical_fitted_expression = result.canonical_fitted_expression
        self.final_reward = None
        self.sobolev_result = None
        self.sobolev_penalty = None
        self.sobolev_success = None
        self.sobolev_evaluated = False
        self.sn_cache_identity = None
        self.sobolev_evaluation_failure = None
        self.term_provenance = None
        self.provenance_cache_expression = None


class ControlledGP(NDPopulationGP):
    """True population GP under the controlled Base-vs-SN protocol."""

    def __init__(
        self,
        variables: Sequence[nd.Variable],
        *,
        profile: str = PROFILE_BASE,
        binary: Sequence[type[nd.Symbol]] = DEFAULT_BINARY,
        unary: Sequence[type[nd.Symbol]] = DEFAULT_UNARY,
        fixed_constants: Sequence[float] = DEFAULT_FIXED_CONSTANTS,
        population_size: int = 1000,
        elitism_k: int = 10,
        tournament_size: int = 20,
        p_crossover: float = 0.9,
        p_subtree_mutation: float = 0.01,
        p_hoist_mutation: float = 0.01,
        p_point_mutation: float = 0.01,
        p_point_replace: float = 0.05,
        depth_range: tuple[int, int] = (2, 6),
        full_prob: float = 0.5,
        random_state: int = 20260808,
        n_iter: int = 10_000,
        time_limit: float = 900.0,
        hard_time_limit: float = 1800.0,
        max_len: int = 30,
        max_additive_terms: int = 128,
        eta: float = 0.999,
        ratio: float = 1.0,
        sobolev_alpha: float = 0.01,
        sobolev_tau: float = 1.0 / math.sqrt(10.0),
        sobolev_lambda_value: float = 1.0,
        sobolev_lambda_gradient: float = 1.0,
        geometry_sample_size: int = 64,
        shortlist_size: int = 64,
        sobolev_failure_policy: str | None = None,
        sobolev_pruning: bool = True,
        sobolev_max_prunes: int = 1,
        prune_elite_k: int = 1,
        sobolev_acceptance_tolerance: float = 0.0,
        incremental_max_changed_terms: int = 4,
        term_cache_max_entries: int = 50_000,
        term_cache_max_memory_bytes: int = 256 * 1024 * 1024,
        derivative_cache_max_entries: int = 50_000,
        geometry_cache_max_entries: int = 5_000,
        geometry_cache_max_memory_bytes: int = 256 * 1024 * 1024,
        base_cache_max_entries: int = 50_000,
        dataset_identity: str = "dataset",
        initial_population_expressions: Sequence[str] | None = None,
        initial_population_sha256: str | None = None,
        output_dir: Path | str | None = None,
        detailed_logging: bool = True,
        record_integrity_metadata: bool = True,
        sn_compare_epsilon_abs: float = DEFAULT_SN_COMPARE_EPSILON_ABS,
        sn_compare_epsilon_rel: float = DEFAULT_SN_COMPARE_EPSILON_REL,
        sn_mutation_delta: float = DEFAULT_SN_MUTATION_DELTA,
        sn_mutation_gamma: float = DEFAULT_SN_MUTATION_GAMMA,
        sn_mutation_impact_beta: float | None = None,
        sn_mutation_impact_epsilon: float = DEFAULT_SN_MUTATION_IMPACT_EPSILON,
        sn_mutation_max_normalized_impact: float | None = None,
        sn_base_anchor_enabled: bool | None = None,
        sn_base_anchor_elite_slots: int = 1,
        sn_export_mode: str | None = None,
        sn_tau: float | None = None,
        sn_geometry_sample_size: int | None = None,
        sn_cache_enabled: bool = True,
        sn_provenance_enabled: bool = True,
        sn_trace_every: int = 1000,
        record_reward_gap_samples: bool = False,
        base_score_semantics: str | None = None,
        sn_repair_shortlist_size: int = 16,
        sn_repair_max_accepted_per_generation: int = 1,
        sn_repair_shadow_enabled: bool | None = None,
        sn_archive_export_enabled: bool | None = None,
        sn_archive_export_max_steps: int = 8,
        sn_archive_anchor_export_enabled: bool | None = None,
        sn_archive_anchor_export_max_steps: int = 4,
        sn_archive_beam_export_enabled: bool | None = None,
        sn_archive_beam_export_max_steps: int = 8,
        sn_archive_beam_width: int = 4,
        sn_archive_beam_shortlist_size: int = 4,
        sn_archive_beam_value_shortlist_size: int = 0,
        sn_archive_beam_innovation_shortlist_size: int = 0,
        sn_archive_beam_pair_first_shortlist_size: int = 0,
        sn_archive_beam_pair_second_shortlist_size: int = 0,
        sn_archive_interaction_joint_shortlist_size: int = 0,
        sn_archive_interaction_value_shortlist_size: int = 0,
        sn_archive_unary_transforms: Sequence[str] = ("sin", "cos", "tanh"),
        sn_direct_phase_scales: Sequence[float] = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        ),
        sn_direct_phase_include_squares: bool = True,
        sn_phase_interaction_batch_size: int = 512,
        sn_radical_scales: Sequence[float] = (0.5, 1.0, 2.0),
        sn_radical_max_dimension: int = 6,
        sn_radical_phase_anchor_limit: int = 4,
        sn_radical_amplitude_joint_shortlist_size: int = 4,
        sn_radical_amplitude_value_shortlist_size: int = 4,
        sn_shared_phase_mobius_shifts: Sequence[float] = (
            -2.0,
            -1.0,
            -0.5,
            0.5,
            1.0,
            2.0,
        ),
        sn_shared_phase_max_dimension: int = 6,
        sn_shared_phase_max_numerator_degree: int = 4,
        sn_shared_phase_max_denominator_degree: int = 2,
        sn_shared_phase_backbone_value_pool_size: int = 64,
        sn_shared_phase_backbone_shortlist_size: int = 16,
        sn_shared_phase_modulated_shortlist_size: int = 32,
        sn_shared_phase_complement_pool_size: int = 8,
        sn_shared_phase_complement_shortlist_size: int = 4,
        sn_shared_phase_proposal_limit: int = 32,
        sn_exponential_scales: Sequence[float] = (0.5, 1.0, 2.0),
        sn_exponential_max_dimension: int = 6,
        sn_exponential_max_numerator_degree: int = 3,
        sn_exponential_max_denominator_degree: int = 2,
        sn_exponential_value_pool_size: int = 64,
        sn_exponential_shortlist_size: int = 16,
        sn_exponential_composite_value_pool_size: int = 64,
        sn_exponential_composite_shortlist_size: int = 16,
        sn_exponential_proposal_limit: int = 32,
        sn_affine_gaussian_scales: Sequence[float] = (0.5, 1.0, 2.0),
        sn_affine_gaussian_max_dimension: int = 6,
        sn_affine_gaussian_value_pool_size: int = 64,
        sn_affine_gaussian_shortlist_size: int = 16,
        sn_affine_gaussian_composite_value_pool_size: int = 64,
        sn_affine_gaussian_composite_shortlist_size: int = 16,
        sn_affine_gaussian_proposal_limit: int = 32,
        sn_sinc_scales: Sequence[float] = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        ),
        sn_sinc_max_dimension: int = 6,
        sn_sinc_max_numerator_degree: int = 3,
        sn_sinc_max_denominator_degree: int = 2,
        sn_sinc_shape_value_pool_size: int = 64,
        sn_sinc_shape_shortlist_size: int = 16,
        sn_sinc_composite_value_pool_size: int = 64,
        sn_sinc_composite_shortlist_size: int = 16,
        sn_sinc_proposal_limit: int = 32,
        sn_shared_unary_scales: Sequence[float] = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        ),
        sn_shared_unary_transforms: Sequence[str] = ("sin", "cos", "tanh"),
        sn_shared_unary_max_dimension: int = 6,
        sn_shared_unary_max_numerator_degree: int = 2,
        sn_shared_unary_max_denominator_degree: int = 2,
        sn_shared_unary_phase_value_pool_size: int = 64,
        sn_shared_unary_phase_shortlist_size: int = 16,
        sn_shared_unary_amplitude_value_pool_size: int = 32,
        sn_shared_unary_amplitude_shortlist_size: int = 8,
        sn_shared_unary_proposal_limit: int = 32,
        sn_reciprocal_trig_scales: Sequence[float] = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        ),
        sn_reciprocal_trig_transforms: Sequence[str] = ("sin", "cos", "tanh"),
        sn_reciprocal_trig_max_dimension: int = 6,
        sn_reciprocal_trig_value_pool_size: int = 64,
        sn_reciprocal_trig_shortlist_size: int = 16,
        sn_reciprocal_trig_proposal_limit: int = 32,
        sn_cross_unary_scales: Sequence[float] = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        ),
        sn_cross_unary_transforms: Sequence[str] = ("sin", "cos", "tanh"),
        sn_cross_unary_inner_powers: Sequence[int] = (1, 2),
        sn_cross_unary_max_dimension: int = 6,
        sn_cross_unary_max_numerator_degree: int = 2,
        sn_cross_unary_max_denominator_degree: int = 1,
        sn_cross_unary_value_pool_size: int = 64,
        sn_cross_unary_shortlist_size: int = 16,
        sn_cross_unary_proposal_limit: int = 32,
        sn_shared_denominator_signs: Sequence[int] = (1, -1),
        sn_shared_denominator_numerator_signs: Sequence[int] = (1,),
        sn_shared_denominator_max_dimension: int = 6,
        sn_shared_denominator_value_pool_size: int = 64,
        sn_shared_denominator_shortlist_size: int = 16,
        sn_shared_denominator_proposal_limit: int = 32,
        sn_relativistic_rational_signs: Sequence[int] = (1, -1),
        sn_relativistic_rational_scales: Sequence[float] = (0.5, 1.0, 2.0),
        sn_relativistic_rational_max_dimension: int = 6,
        sn_relativistic_rational_value_pool_size: int = 64,
        sn_relativistic_rational_shortlist_size: int = 16,
        sn_relativistic_rational_proposal_limit: int = 32,
        sn_cosine_law_phase_signs: Sequence[int] = (1, -1),
        sn_cosine_law_radial_signs: Sequence[int] = (1, -1),
        sn_cosine_law_phase_scales: Sequence[float] = (0.5, 1.0, 2.0),
        sn_cosine_law_max_dimension: int = 6,
        sn_cosine_law_value_pool_size: int = 64,
        sn_cosine_law_shortlist_size: int = 16,
        sn_cosine_law_proposal_limit: int = 32,
        sn_reciprocal_sine_scales: Sequence[float] = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        ),
        sn_reciprocal_sine_max_dimension: int = 8,
        sn_reciprocal_sine_max_numerator_degree: int = 5,
        sn_reciprocal_sine_value_pool_size: int = 64,
        sn_reciprocal_sine_shortlist_size: int = 16,
        sn_reciprocal_sine_proposal_limit: int = 32,
        sn_inverse_cosine_phase_signs: Sequence[int] = (1, -1),
        sn_inverse_cosine_radial_signs: Sequence[int] = (1, -1),
        sn_inverse_cosine_phase_scales: Sequence[float] = (0.5, 1.0, 2.0),
        sn_inverse_cosine_max_dimension: int = 6,
        sn_inverse_cosine_value_pool_size: int = 64,
        sn_inverse_cosine_shortlist_size: int = 16,
        sn_inverse_cosine_proposal_limit: int = 32,
        sn_multiaxis_pair_signs: Sequence[int] = (1, -1),
        sn_multiaxis_radial_term_counts: Sequence[int] = (2, 3),
        sn_multiaxis_max_dimension: int = 10,
        sn_multiaxis_max_numerator_degree: int = 3,
        sn_multiaxis_value_pool_size: int = 64,
        sn_multiaxis_shortlist_size: int = 16,
        sn_multiaxis_proposal_limit: int = 32,
        sn_interference_sine_scales: Sequence[float] = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        ),
        sn_interference_sine_max_dimension: int = 8,
        sn_interference_sine_value_pool_size: int = 64,
        sn_interference_sine_shortlist_size: int = 16,
        sn_interference_sine_proposal_limit: int = 32,
        sn_sparse_radical_offsets: Sequence[float] = (1.0,),
        sn_sparse_radical_scales: Sequence[float] = (0.5, 1.0, 2.0),
        sn_sparse_radical_max_dimension: int = 8,
        sn_sparse_radical_max_abs_exponent: int = 4,
        sn_sparse_radical_max_numerator_degree: int = 5,
        sn_sparse_radical_max_denominator_degree: int = 9,
        sn_sparse_radical_integer_candidate_limit: int = 1024,
        sn_sparse_radical_value_pool_size: int = 64,
        sn_sparse_radical_shortlist_size: int = 16,
        sn_sparse_radical_proposal_limit: int = 32,
        sn_shared_ratio_trig_phase_scales: Sequence[float] = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        ),
        sn_shared_ratio_trig_required_dimension: int = 7,
        sn_shared_ratio_trig_value_pool_size: int = 64,
        sn_shared_ratio_trig_shortlist_size: int = 16,
        sn_shared_ratio_trig_proposal_limit: int = 32,
        sn_deep_accuracy_parameters: Mapping[str, Any] | None = None,
        sn_archive_beam_diversity_slots: int = 0,
        sn_archive_beam_refine_max_steps: int = 4,
        sn_coverage_shortlist_size: int = 32,
        sn_coverage_elite_slots: int = 1,
        sn_coverage_max_base_reward_gap: float = 0.001,
        sn_coverage_crossover_rate: float = 0.05,
        sn_child_geometry_enabled: bool | None = None,
        sn_basis_exchange_enabled: bool | None = None,
        sn_basis_infusion_enabled: bool | None = None,
        sn_residual_basis_infusion_enabled: bool | None = None,
        sn_basis_archive_enabled: bool | None = None,
        sn_basis_archive_capacity: int = 64,
        sn_basis_archive_source_candidates: int = 8,
        sn_basis_quality_archive_enabled: bool | None = None,
        sn_basis_quality_archive_capacity: int = 64,
        sn_partial_residual_archive_enabled: bool | None = None,
        sn_basis_pursuit_enabled: bool | None = None,
        sn_basis_pursuit_shortlist_size: int = 4,
        sn_pursuit_stagnation_patience: int = 5,
        sn_pursuit_stagnation_epsilon_abs: float = 1e-6,
        sn_pursuit_stagnation_epsilon_rel: float = 1e-4,
        sn_pursuit_preserve_fallback_anchor: bool = False,
        sn_pursuit_select_by_coverage: bool = False,
        sn_basis_replacement_enabled: bool | None = None,
        sn_basis_replacement_removal_shortlist_size: int = 2,
        sn_conditional_basis_replacement_enabled: bool | None = None,
        sn_orthogonal_basis_crossover_enabled: bool | None = None,
        sn_orthogonal_basis_max_steps: int = 8,
        sn_shadow_slot_enabled: bool | None = None,
        sn_shadow_max_slots: int = 1,
        sn_shadow_minimum_coverage_gain: float = 0.0,
        sn_shadow_allow_represented_amplification: bool = False,
        sn_basis_forest_crossover_rate: float = 0.10,
        sn_basis_forest_mutation_recombination_enabled: bool | None = None,
        sn_basis_forest_mutation_shadow_enabled: bool | None = None,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown GP profile {profile!r}; expected one of {PROFILES}"
            )
        if sn_repair_shadow_enabled is None:
            sn_repair_shadow_enabled = profile == PROFILE_SN_REPAIR_SHADOW
        if sn_repair_shadow_enabled and profile != PROFILE_SN_REPAIR_SHADOW:
            raise ValueError("repair shadow is available only in sn_gp_repair_shadow")
        if sn_archive_export_enabled is None:
            sn_archive_export_enabled = profile == PROFILE_SN_ARCHIVE_EXPORT
        if sn_archive_export_enabled and profile != PROFILE_SN_ARCHIVE_EXPORT:
            raise ValueError(
                "archive recombination export is available only in "
                "sn_gp_archive_recombination_export"
            )
        if sn_archive_export_max_steps < 1:
            raise ValueError("archive export steps must be positive")
        if sn_archive_anchor_export_enabled is None:
            sn_archive_anchor_export_enabled = (
                profile == PROFILE_SN_ARCHIVE_ANCHOR_EXPORT
            )
        if (
            sn_archive_anchor_export_enabled
            and profile != PROFILE_SN_ARCHIVE_ANCHOR_EXPORT
        ):
            raise ValueError(
                "anchor archive export is available only in "
                "sn_gp_archive_anchor_export"
            )
        if sn_archive_anchor_export_max_steps < 1:
            raise ValueError("anchor archive export steps must be positive")
        if sn_archive_beam_export_enabled is None:
            sn_archive_beam_export_enabled = profile in {
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
                *DEEP_ACCURACY_PROFILE_NAMES,
                PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
            }
        if sn_archive_beam_export_enabled and profile not in {
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
            *DEEP_ACCURACY_PROFILE_NAMES,
            PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
        }:
            raise ValueError(
                "beam archive export is available only in " "sn_gp_archive_beam_export"
            )
        if sn_archive_beam_export_max_steps < 1:
            raise ValueError("beam archive export steps must be positive")
        if sn_archive_beam_refine_max_steps < 1:
            raise ValueError("beam archive refinement steps must be positive")
        if sn_archive_beam_width < 1 or sn_archive_beam_shortlist_size < 1:
            raise ValueError("beam width and shortlist size must be positive")
        if sn_archive_beam_value_shortlist_size < 0:
            raise ValueError("beam value shortlist size cannot be negative")
        if sn_archive_beam_innovation_shortlist_size < 0:
            raise ValueError("beam innovation shortlist size cannot be negative")
        if (
            sn_archive_beam_pair_first_shortlist_size < 0
            or sn_archive_beam_pair_second_shortlist_size < 0
        ):
            raise ValueError("beam pair shortlist sizes cannot be negative")
        if (sn_archive_beam_pair_first_shortlist_size == 0) != (
            sn_archive_beam_pair_second_shortlist_size == 0
        ):
            raise ValueError("beam pair shortlist sizes must be enabled together")
        if (
            sn_archive_interaction_joint_shortlist_size < 0
            or sn_archive_interaction_value_shortlist_size < 0
        ):
            raise ValueError("archive interaction shortlists cannot be negative")
        archive_unary_transforms = tuple(
            dict.fromkeys(str(value) for value in sn_archive_unary_transforms)
        )
        if not archive_unary_transforms or any(
            value not in {"sin", "cos", "tanh"} for value in archive_unary_transforms
        ):
            raise ValueError("archive unary transforms must be bounded GP operators")
        direct_phase_scales = tuple(
            dict.fromkeys(float(value) for value in sn_direct_phase_scales)
        )
        frozen_phase_scales = (0.5, 1.0, 2.0, math.pi, 2.0 * math.pi)
        if not direct_phase_scales or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_phase_scales
            )
            for value in direct_phase_scales
        ):
            raise ValueError(
                "direct phase scales must use the frozen GP-constant library"
            )
        if sn_phase_interaction_batch_size < 1:
            raise ValueError("phase interaction batch size must be positive")
        radical_scales = tuple(
            dict.fromkeys(float(value) for value in sn_radical_scales)
        )
        frozen_radical_scales = (0.5, 1.0, 2.0)
        if not radical_scales or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_radical_scales
            )
            for value in radical_scales
        ):
            raise ValueError("radical scales must use the frozen GP-constant library")
        if (
            sn_radical_max_dimension < 1
            or sn_radical_phase_anchor_limit < 1
            or sn_radical_amplitude_joint_shortlist_size < 1
            or sn_radical_amplitude_value_shortlist_size < 1
        ):
            raise ValueError("radical composition bounds must be positive")
        shared_phase_mobius_shifts = tuple(
            dict.fromkeys(float(value) for value in sn_shared_phase_mobius_shifts)
        )
        frozen_mobius_shifts = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
        if len(shared_phase_mobius_shifts) < 2 or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_mobius_shifts
            )
            for value in shared_phase_mobius_shifts
        ):
            raise ValueError(
                "shared-phase shifts must use the frozen GP-constant library"
            )
        if (
            sn_shared_phase_max_dimension < 1
            or sn_shared_phase_max_numerator_degree < 1
            or sn_shared_phase_max_denominator_degree < 0
            or sn_shared_phase_backbone_value_pool_size
            < sn_shared_phase_backbone_shortlist_size
            or sn_shared_phase_backbone_shortlist_size < 1
            or sn_shared_phase_modulated_shortlist_size < 1
            or sn_shared_phase_complement_pool_size
            < sn_shared_phase_complement_shortlist_size
            or sn_shared_phase_complement_shortlist_size < 1
            or sn_shared_phase_proposal_limit < 1
        ):
            raise ValueError("shared-phase pursuit bounds are invalid")
        exponential_scales = tuple(
            dict.fromkeys(float(value) for value in sn_exponential_scales)
        )
        frozen_exponential_scales = (0.5, 1.0, 2.0)
        if not exponential_scales or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_exponential_scales
            )
            for value in exponential_scales
        ):
            raise ValueError(
                "exponential scales must use the frozen GP-constant library"
            )
        if (
            sn_exponential_max_dimension < 1
            or sn_exponential_max_numerator_degree < 1
            or sn_exponential_max_denominator_degree < 0
            or sn_exponential_value_pool_size < sn_exponential_shortlist_size
            or sn_exponential_shortlist_size < 1
            or sn_exponential_composite_value_pool_size
            < sn_exponential_composite_shortlist_size
            or sn_exponential_composite_shortlist_size < 1
            or sn_exponential_proposal_limit < 1
        ):
            raise ValueError("damped exponential pursuit bounds are invalid")
        affine_gaussian_scales = tuple(
            dict.fromkeys(float(value) for value in sn_affine_gaussian_scales)
        )
        if not affine_gaussian_scales or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_exponential_scales
            )
            for value in affine_gaussian_scales
        ):
            raise ValueError(
                "affine-Gaussian scales must use the frozen GP-constant library"
            )
        if (
            sn_affine_gaussian_max_dimension < 1
            or sn_affine_gaussian_value_pool_size < sn_affine_gaussian_shortlist_size
            or sn_affine_gaussian_shortlist_size < 1
            or sn_affine_gaussian_composite_value_pool_size
            < sn_affine_gaussian_composite_shortlist_size
            or sn_affine_gaussian_composite_shortlist_size < 1
            or sn_affine_gaussian_proposal_limit < 1
        ):
            raise ValueError("affine-Gaussian pursuit bounds are invalid")
        sinc_scales = tuple(dict.fromkeys(float(value) for value in sn_sinc_scales))
        if not sinc_scales or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_phase_scales
            )
            for value in sinc_scales
        ):
            raise ValueError("sinc scales must use the frozen GP-constant library")
        if (
            sn_sinc_max_dimension < 1
            or sn_sinc_max_numerator_degree < 1
            or sn_sinc_max_denominator_degree < 0
            or sn_sinc_shape_value_pool_size < sn_sinc_shape_shortlist_size
            or sn_sinc_shape_shortlist_size < 1
            or sn_sinc_composite_value_pool_size < sn_sinc_composite_shortlist_size
            or sn_sinc_composite_shortlist_size < 1
            or sn_sinc_proposal_limit < 1
        ):
            raise ValueError("sinc-squared pursuit bounds are invalid")
        shared_unary_scales = tuple(
            dict.fromkeys(float(value) for value in sn_shared_unary_scales)
        )
        shared_unary_transforms = tuple(
            dict.fromkeys(str(value) for value in sn_shared_unary_transforms)
        )
        if not shared_unary_scales or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_phase_scales
            )
            for value in shared_unary_scales
        ):
            raise ValueError(
                "shared-unary scales must use the frozen GP-constant library"
            )
        if not shared_unary_transforms or any(
            value not in {"sin", "cos", "tanh"} for value in shared_unary_transforms
        ):
            raise ValueError("shared-unary transforms are invalid")
        if (
            sn_shared_unary_max_dimension < 1
            or sn_shared_unary_max_numerator_degree < 1
            or sn_shared_unary_max_denominator_degree < 0
            or sn_shared_unary_phase_value_pool_size
            < sn_shared_unary_phase_shortlist_size
            or sn_shared_unary_phase_shortlist_size < 1
            or sn_shared_unary_amplitude_value_pool_size
            < sn_shared_unary_amplitude_shortlist_size
            or sn_shared_unary_amplitude_shortlist_size < 1
            or sn_shared_unary_proposal_limit < 1
        ):
            raise ValueError("shared-unary polynomial pursuit bounds are invalid")
        reciprocal_trig_scales = tuple(
            dict.fromkeys(float(value) for value in sn_reciprocal_trig_scales)
        )
        reciprocal_trig_transforms = tuple(
            dict.fromkeys(str(value) for value in sn_reciprocal_trig_transforms)
        )
        if not reciprocal_trig_scales or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_phase_scales
            )
            for value in reciprocal_trig_scales
        ):
            raise ValueError(
                "reciprocal-trigonometric scales must use the frozen GP-constant library"
            )
        if not reciprocal_trig_transforms or any(
            value not in {"sin", "cos", "tanh"} for value in reciprocal_trig_transforms
        ):
            raise ValueError("reciprocal-trigonometric transforms are invalid")
        if (
            sn_reciprocal_trig_max_dimension < 1
            or sn_reciprocal_trig_value_pool_size < sn_reciprocal_trig_shortlist_size
            or sn_reciprocal_trig_shortlist_size < 1
            or sn_reciprocal_trig_proposal_limit < 1
        ):
            raise ValueError("reciprocal-trigonometric pursuit bounds are invalid")
        cross_unary_scales = tuple(
            dict.fromkeys(float(value) for value in sn_cross_unary_scales)
        )
        cross_unary_transforms = tuple(
            dict.fromkeys(str(value) for value in sn_cross_unary_transforms)
        )
        cross_unary_inner_powers = tuple(
            dict.fromkeys(int(value) for value in sn_cross_unary_inner_powers)
        )
        if not cross_unary_scales or any(
            not np.isfinite(value)
            or not any(
                math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                for frozen in frozen_phase_scales
            )
            for value in cross_unary_scales
        ):
            raise ValueError(
                "cross-unary scales must use the frozen GP-constant library"
            )
        if not cross_unary_transforms or any(
            value not in {"sin", "cos", "tanh"} for value in cross_unary_transforms
        ):
            raise ValueError("cross-unary transforms are invalid")
        if not cross_unary_inner_powers or any(
            value not in {1, 2} for value in cross_unary_inner_powers
        ):
            raise ValueError("cross-unary inner powers are invalid")
        if (
            sn_cross_unary_max_dimension < 1
            or sn_cross_unary_max_numerator_degree < 1
            or sn_cross_unary_max_denominator_degree < 0
            or sn_cross_unary_value_pool_size < sn_cross_unary_shortlist_size
            or sn_cross_unary_shortlist_size < 1
            or sn_cross_unary_proposal_limit < 1
        ):
            raise ValueError("cross-unary pursuit bounds are invalid")
        shared_denominator_signs = tuple(
            dict.fromkeys(int(value) for value in sn_shared_denominator_signs)
        )
        shared_denominator_numerator_signs = tuple(
            dict.fromkeys(int(value) for value in sn_shared_denominator_numerator_signs)
        )
        if (
            not shared_denominator_signs
            or any(value not in {-1, 1} for value in shared_denominator_signs)
            or not shared_denominator_numerator_signs
            or any(value not in {-1, 1} for value in shared_denominator_numerator_signs)
            or sn_shared_denominator_max_dimension < 2
            or sn_shared_denominator_value_pool_size
            < sn_shared_denominator_shortlist_size
            or sn_shared_denominator_shortlist_size < 1
            or sn_shared_denominator_proposal_limit < 1
        ):
            raise ValueError("shared-denominator pursuit bounds are invalid")
        relativistic_rational_signs = tuple(
            dict.fromkeys(int(value) for value in sn_relativistic_rational_signs)
        )
        relativistic_rational_scales = tuple(
            dict.fromkeys(float(value) for value in sn_relativistic_rational_scales)
        )
        if (
            not relativistic_rational_signs
            or any(value not in {-1, 1} for value in relativistic_rational_signs)
            or not relativistic_rational_scales
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in relativistic_rational_scales
            )
            or sn_relativistic_rational_max_dimension < 2
            or sn_relativistic_rational_value_pool_size
            < sn_relativistic_rational_shortlist_size
            or sn_relativistic_rational_shortlist_size < 1
            or sn_relativistic_rational_proposal_limit < 1
        ):
            raise ValueError("relativistic-rational pursuit bounds are invalid")
        cosine_law_phase_signs = tuple(
            dict.fromkeys(int(value) for value in sn_cosine_law_phase_signs)
        )
        cosine_law_radial_signs = tuple(
            dict.fromkeys(int(value) for value in sn_cosine_law_radial_signs)
        )
        cosine_law_phase_scales = tuple(
            dict.fromkeys(float(value) for value in sn_cosine_law_phase_scales)
        )
        if (
            not cosine_law_phase_signs
            or any(value not in {-1, 1} for value in cosine_law_phase_signs)
            or not cosine_law_radial_signs
            or any(value not in {-1, 1} for value in cosine_law_radial_signs)
            or not cosine_law_phase_scales
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in cosine_law_phase_scales
            )
            or sn_cosine_law_max_dimension < 2
            or sn_cosine_law_value_pool_size < sn_cosine_law_shortlist_size
            or sn_cosine_law_shortlist_size < 1
            or sn_cosine_law_proposal_limit < 1
        ):
            raise ValueError("cosine-law radial pursuit bounds are invalid")
        reciprocal_sine_scales = tuple(
            dict.fromkeys(float(value) for value in sn_reciprocal_sine_scales)
        )
        frozen_reciprocal_sine_scales = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        )
        if (
            not reciprocal_sine_scales
            or any(
                not np.isfinite(value)
                or not any(
                    math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                    for frozen in frozen_reciprocal_sine_scales
                )
                for value in reciprocal_sine_scales
            )
            or sn_reciprocal_sine_max_dimension < 2
            or sn_reciprocal_sine_max_numerator_degree < 1
            or sn_reciprocal_sine_value_pool_size < sn_reciprocal_sine_shortlist_size
            or sn_reciprocal_sine_shortlist_size < 1
            or sn_reciprocal_sine_proposal_limit < 1
        ):
            raise ValueError("reciprocal-sine-square pursuit bounds are invalid")
        inverse_cosine_phase_signs = tuple(
            dict.fromkeys(int(value) for value in sn_inverse_cosine_phase_signs)
        )
        inverse_cosine_radial_signs = tuple(
            dict.fromkeys(int(value) for value in sn_inverse_cosine_radial_signs)
        )
        inverse_cosine_phase_scales = tuple(
            dict.fromkeys(float(value) for value in sn_inverse_cosine_phase_scales)
        )
        if (
            not inverse_cosine_phase_signs
            or any(value not in {-1, 1} for value in inverse_cosine_phase_signs)
            or not inverse_cosine_radial_signs
            or any(value not in {-1, 1} for value in inverse_cosine_radial_signs)
            or not inverse_cosine_phase_scales
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in inverse_cosine_phase_scales
            )
            or sn_inverse_cosine_max_dimension < 2
            or sn_inverse_cosine_value_pool_size < sn_inverse_cosine_shortlist_size
            or sn_inverse_cosine_shortlist_size < 1
            or sn_inverse_cosine_proposal_limit < 1
        ):
            raise ValueError("inverse cosine-radial pursuit bounds are invalid")
        multiaxis_pair_signs = tuple(
            dict.fromkeys(int(value) for value in sn_multiaxis_pair_signs)
        )
        multiaxis_radial_term_counts = tuple(
            sorted(
                dict.fromkeys(int(value) for value in sn_multiaxis_radial_term_counts)
            )
        )
        if (
            not multiaxis_pair_signs
            or any(value not in {-1, 1} for value in multiaxis_pair_signs)
            or not multiaxis_radial_term_counts
            or any(value < 2 for value in multiaxis_radial_term_counts)
            or sn_multiaxis_max_dimension < 4
            or sn_multiaxis_max_numerator_degree < 0
            or sn_multiaxis_value_pool_size < sn_multiaxis_shortlist_size
            or sn_multiaxis_shortlist_size < 1
            or sn_multiaxis_proposal_limit < 1
        ):
            raise ValueError("multi-axis inverse-square pursuit bounds are invalid")
        interference_sine_scales = tuple(
            dict.fromkeys(float(value) for value in sn_interference_sine_scales)
        )
        frozen_interference_sine_scales = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        )
        if (
            not interference_sine_scales
            or any(
                not np.isfinite(value)
                or not any(
                    math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                    for frozen in frozen_interference_sine_scales
                )
                for value in interference_sine_scales
            )
            or sn_interference_sine_max_dimension < 2
            or sn_interference_sine_value_pool_size
            < sn_interference_sine_shortlist_size
            or sn_interference_sine_shortlist_size < 1
            or sn_interference_sine_proposal_limit < 1
        ):
            raise ValueError("interference sine-ratio pursuit bounds are invalid")
        sparse_radical_offsets = tuple(
            dict.fromkeys(float(value) for value in sn_sparse_radical_offsets)
        )
        sparse_radical_scales = tuple(
            dict.fromkeys(float(value) for value in sn_sparse_radical_scales)
        )
        frozen_sparse_radical_constants = (0.5, 1.0, 2.0)
        if (
            not sparse_radical_offsets
            or not sparse_radical_scales
            or any(
                not np.isfinite(value)
                or not any(
                    math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                    for frozen in frozen_sparse_radical_constants
                )
                for value in (*sparse_radical_offsets, *sparse_radical_scales)
            )
            or sn_sparse_radical_max_dimension < 1
            or sn_sparse_radical_max_abs_exponent < 1
            or sn_sparse_radical_max_numerator_degree < 1
            or sn_sparse_radical_max_denominator_degree < 1
            or sn_sparse_radical_integer_candidate_limit < 1
            or sn_sparse_radical_value_pool_size < sn_sparse_radical_shortlist_size
            or sn_sparse_radical_shortlist_size < 1
            or sn_sparse_radical_proposal_limit < 1
        ):
            raise ValueError("sparse-radical pursuit bounds are invalid")
        shared_ratio_trig_phase_scales = tuple(
            dict.fromkeys(
                float(value) for value in sn_shared_ratio_trig_phase_scales
            )
        )
        frozen_shared_ratio_trig_scales = (
            0.5,
            1.0,
            2.0,
            math.pi,
            2.0 * math.pi,
        )
        if (
            not shared_ratio_trig_phase_scales
            or any(
                not np.isfinite(value)
                or not any(
                    math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-15)
                    for frozen in frozen_shared_ratio_trig_scales
                )
                for value in shared_ratio_trig_phase_scales
            )
            or sn_shared_ratio_trig_required_dimension != 7
            or sn_shared_ratio_trig_value_pool_size
            < sn_shared_ratio_trig_shortlist_size
            or sn_shared_ratio_trig_shortlist_size < 1
            or sn_shared_ratio_trig_proposal_limit < 1
        ):
            raise ValueError("shared-ratio trigonometric bounds are invalid")
        if (
            profile
            in {
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
                *DEEP_ACCURACY_PROFILE_NAMES,
                PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
            }
            and sn_archive_beam_value_shortlist_size < 1
        ):
            raise ValueError("balanced-screen beam requires a value shortlist")
        if (
            profile
            in {
                PROFILE_SN_ARCHIVE_INNOVATION_SCREEN_BEAM_EXPORT,
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
                *DEEP_ACCURACY_PROFILE_NAMES,
        }
            and sn_archive_beam_innovation_shortlist_size < 1
        ):
            raise ValueError("innovation-screen beam requires an innovation shortlist")
        if profile == PROFILE_SN_ARCHIVE_PAIR_LOOKAHEAD_BEAM_EXPORT and (
            sn_archive_beam_pair_first_shortlist_size < 1
            or sn_archive_beam_pair_second_shortlist_size < 1
        ):
            raise ValueError("pair-lookahead beam requires both pair shortlists")
        if profile in {
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
            *DEEP_ACCURACY_PROFILE_NAMES,
        } and (
            sn_archive_interaction_joint_shortlist_size < 1
            or sn_archive_interaction_value_shortlist_size < 1
        ):
            raise ValueError("product-lift beam requires both interaction shortlists")
        if not 0 <= sn_archive_beam_diversity_slots < sn_archive_beam_width:
            raise ValueError("beam diversity slots must leave at least one Base slot")
        if (
            profile == PROFILE_SN_ARCHIVE_BEAM_EXPORT
            and sn_archive_beam_diversity_slots != 0
        ):
            raise ValueError("Stage AP freezes zero beam diversity slots")
        if (
            profile == PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT
            and sn_archive_beam_diversity_slots != 0
        ):
            raise ValueError("value-only accuracy control freezes zero diversity slots")
        if (
            profile == PROFILE_SN_ARCHIVE_DIVERSE_BEAM_EXPORT
            and sn_archive_beam_diversity_slots < 1
        ):
            raise ValueError("diverse beam export requires a diversity slot")
        if (
            profile
            in {
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
                *DEEP_ACCURACY_PROFILE_NAMES,
        }
            and sn_archive_beam_diversity_slots != 0
        ):
            raise ValueError("dual beam export freezes its schedule list internally")
        if sn_basis_forest_mutation_recombination_enabled is None:
            sn_basis_forest_mutation_recombination_enabled = (
                profile == PROFILE_BASIS_FOREST_SN_MUTATION_RECOMBINATION
            )
        if (
            sn_basis_forest_mutation_recombination_enabled
            and profile != PROFILE_BASIS_FOREST_SN_MUTATION_RECOMBINATION
        ):
            raise ValueError(
                "basis-forest mutation recombination is available only in "
                "basis_forest_sn_mutation_recombination"
            )
        if sn_basis_forest_mutation_shadow_enabled is None:
            sn_basis_forest_mutation_shadow_enabled = profile in {
                PROFILE_BASIS_FOREST_SN_MUTATION_SHADOW,
                PROFILE_BASIS_FOREST_SN_COMPRESS_MUTATION_SHADOW,
                PROFILE_BASIS_FOREST_SN_COMPRESS_MULTISOURCE_SHADOW,
            }
        if sn_basis_forest_mutation_shadow_enabled and profile not in {
            PROFILE_BASIS_FOREST_SN_MUTATION_SHADOW,
            PROFILE_BASIS_FOREST_SN_COMPRESS_MUTATION_SHADOW,
            PROFILE_BASIS_FOREST_SN_COMPRESS_MULTISOURCE_SHADOW,
        }:
            raise ValueError(
                "basis-forest mutation shadow is available only in "
                "basis_forest_sn_mutation_shadow"
            )
        if sobolev_failure_policy is None:
            sobolev_failure_policy = (
                "base_fallback"
                if profile
                in {
                    PROFILE_SN_V2,
                    PROFILE_SN_STABLE,
                    PROFILE_SN_VERIFIED_REPAIR,
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
                    *DEEP_ACCURACY_PROFILE_NAMES,
                    PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
                    PROFILE_SN_REPAIR_COVERAGE,
                    PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                    PROFILE_SN_CHILD_GEOMETRY,
                    PROFILE_SN_BASIS_EXCHANGE,
                    PROFILE_SN_BASIS_INFUSION,
                    PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
                else "max_penalty"
            )
        if sn_base_anchor_enabled is None:
            sn_base_anchor_enabled = profile == PROFILE_SN_STABLE
        if sn_export_mode is None:
            sn_export_mode = (
                "base_anchor"
                if profile == PROFILE_SN_STABLE and sn_base_anchor_enabled
                else "sn_comparator"
            )
        if sn_mutation_impact_beta is None:
            sn_mutation_impact_beta = (
                DEFAULT_SN_MUTATION_IMPACT_BETA if profile == PROFILE_SN_STABLE else 0.0
            )
        if sn_mutation_max_normalized_impact is None:
            sn_mutation_max_normalized_impact = (
                DEFAULT_SN_MUTATION_MAX_NORMALIZED_IMPACT
                if profile == PROFILE_SN_STABLE
                else 1.0
            )
        if base_score_semantics is None:
            base_score_semantics = (
                "export_aligned"
                if profile
                in {
                    PROFILE_BASE_EXPORT_ALIGNED,
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
                    PROFILE_SN_VERIFIED_REPAIR,
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
                    *DEEP_ACCURACY_PROFILE_NAMES,
                    PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
                    PROFILE_SN_REPAIR_COVERAGE,
                    PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                    PROFILE_SN_CHILD_GEOMETRY,
                    PROFILE_SN_BASIS_EXCHANGE,
                    PROFILE_SN_BASIS_INFUSION,
                    PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
                }
                else "design_matrix"
            )
        if base_score_semantics not in {"design_matrix", "export_aligned"}:
            raise ValueError(
                "base_score_semantics must be design_matrix or export_aligned"
            )
        if (
            profile
            in {
                PROFILE_BASE_EXPORT_ALIGNED,
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
                PROFILE_SN_VERIFIED_REPAIR,
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
                *DEEP_ACCURACY_PROFILE_NAMES,
                PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
                PROFILE_SN_REPAIR_COVERAGE,
                PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                PROFILE_SN_CHILD_GEOMETRY,
                PROFILE_SN_BASIS_EXCHANGE,
                PROFILE_SN_BASIS_INFUSION,
                PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            }
            and base_score_semantics != "export_aligned"
        ):
            raise ValueError("deep exploration profiles freeze export_aligned scoring")
        if ratio != 1.0:
            raise ValueError("Controlled white-box GP freezes internal ratio=1")
        if population_size <= elitism_k or elitism_k < 1:
            raise ValueError("population_size must exceed positive elitism_k")
        if shortlist_size < elitism_k:
            raise ValueError("shortlist_size must be at least elitism_k")
        if time_limit <= 0 or hard_time_limit <= time_limit:
            raise ValueError("hard_time_limit must exceed positive time_limit")
        if profile == PROFILE_SN and sobolev_failure_policy != "max_penalty":
            raise ValueError("Legacy SN-GP freezes failure_policy=max_penalty")
        if (
            profile
            in {
                PROFILE_SN_V2,
                PROFILE_SN_STABLE,
                PROFILE_SN_VERIFIED_REPAIR,
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
                *DEEP_ACCURACY_PROFILE_NAMES,
                PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
                PROFILE_SN_REPAIR_COVERAGE,
                PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                PROFILE_SN_CHILD_GEOMETRY,
                PROFILE_SN_BASIS_EXCHANGE,
                PROFILE_SN_BASIS_INFUSION,
                PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            and sobolev_failure_policy != "base_fallback"
        ):
            raise ValueError(
                "SN-GP plugin profiles freeze failure_policy=base_fallback"
            )
        if sn_base_anchor_elite_slots != 1:
            raise ValueError("Stable SN-GP freezes exactly one Base anchor elite slot")
        if profile != PROFILE_SN_STABLE and sn_base_anchor_enabled:
            raise ValueError("Base safety anchor is available only in sn_gp_stable")
        if sn_export_mode not in {"sn_comparator", "base_anchor"}:
            raise ValueError("sn_export_mode must be sn_comparator or base_anchor")
        if (
            profile == PROFILE_SN_STABLE
            and sn_base_anchor_enabled
            and sn_export_mode != "base_anchor"
        ):
            raise ValueError("sn_gp_stable freezes export_mode=base_anchor")
        if not sn_base_anchor_enabled and sn_export_mode == "base_anchor":
            raise ValueError("base_anchor export requires the Base safety anchor")
        if sobolev_max_prunes not in {0, 1} or prune_elite_k not in {0, 1}:
            raise ValueError("Controlled GP supports at most one top-1 prune")
        if sn_repair_shortlist_size < 1:
            raise ValueError("sn_repair_shortlist_size must be positive")
        if sn_repair_max_accepted_per_generation != 1:
            raise ValueError(
                "verified repair freezes one accepted repair per generation"
            )
        if (
            profile
            in {
                PROFILE_SN_REPAIR_COVERAGE,
                PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                PROFILE_SN_CHILD_GEOMETRY,
                PROFILE_SN_BASIS_EXCHANGE,
                PROFILE_SN_BASIS_INFUSION,
                PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            }
            and sn_coverage_shortlist_size < elitism_k
        ):
            raise ValueError("coverage shortlist must include the Base elite boundary")
        if (
            profile
            in {
                PROFILE_SN_REPAIR_COVERAGE,
                PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                PROFILE_SN_CHILD_GEOMETRY,
                PROFILE_SN_BASIS_EXCHANGE,
                PROFILE_SN_BASIS_INFUSION,
                PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            }
            and sn_coverage_elite_slots != 1
        ):
            raise ValueError("population coverage freezes exactly one elite slot")
        if (
            profile
            in {
                PROFILE_SN_REPAIR_COVERAGE,
                PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                PROFILE_SN_CHILD_GEOMETRY,
                PROFILE_SN_BASIS_EXCHANGE,
                PROFILE_SN_BASIS_INFUSION,
                PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            }
            and elitism_k <= sn_coverage_elite_slots
        ):
            raise ValueError("population coverage requires at least one Base anchor")
        if (
            profile
            in {
                PROFILE_SN_REPAIR_COVERAGE,
                PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                PROFILE_SN_CHILD_GEOMETRY,
                PROFILE_SN_BASIS_EXCHANGE,
                PROFILE_SN_BASIS_INFUSION,
                PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            }
            and sn_coverage_max_base_reward_gap < 0.0
        ):
            raise ValueError("coverage Base reward gap must be non-negative")
        if (
            profile
            in {
                PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
                PROFILE_SN_CHILD_GEOMETRY,
                PROFILE_SN_BASIS_EXCHANGE,
                PROFILE_SN_BASIS_INFUSION,
                PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            }
            and not 0.0 <= sn_coverage_crossover_rate <= 1.0
        ):
            raise ValueError("coverage crossover rate must lie in [0, 1]")
        if sn_child_geometry_enabled is None:
            sn_child_geometry_enabled = profile == PROFILE_SN_CHILD_GEOMETRY
        if sn_child_geometry_enabled and profile != PROFILE_SN_CHILD_GEOMETRY:
            raise ValueError("actual-child geometry is available only in its profile")
        if sn_basis_exchange_enabled is None:
            sn_basis_exchange_enabled = profile == PROFILE_SN_BASIS_EXCHANGE
        if sn_basis_exchange_enabled and profile != PROFILE_SN_BASIS_EXCHANGE:
            raise ValueError("basis exchange is available only in its profile")
        if sn_basis_infusion_enabled is None:
            sn_basis_infusion_enabled = profile == PROFILE_SN_BASIS_INFUSION
        if sn_basis_infusion_enabled and profile != PROFILE_SN_BASIS_INFUSION:
            raise ValueError("basis infusion is available only in its profile")
        if sn_residual_basis_infusion_enabled is None:
            sn_residual_basis_infusion_enabled = (
                profile == PROFILE_SN_RESIDUAL_BASIS_INFUSION
            )
        if (
            sn_residual_basis_infusion_enabled
            and profile != PROFILE_SN_RESIDUAL_BASIS_INFUSION
        ):
            raise ValueError("residual basis infusion is available only in its profile")
        archive_profiles = {
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
            *DEEP_ACCURACY_PROFILE_NAMES,
            PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT,
            PROFILE_SN_BASIS_ARCHIVE,
            PROFILE_SN_PARTIAL_RESIDUAL_ARCHIVE,
            PROFILE_SN_SCREENED_BASIS_PURSUIT,
            PROFILE_SN_SCREENED_PURSUIT_SHADOW,
            PROFILE_SN_PARETO_PURSUIT_SHADOW,
            PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
            PROFILE_SN_BIDIRECTIONAL_BASIS_REPLACEMENT,
            PROFILE_SN_CONDITIONAL_BASIS_REPLACEMENT,
        }
        if sn_basis_archive_enabled is None:
            sn_basis_archive_enabled = profile in archive_profiles
        if sn_basis_archive_enabled and profile not in archive_profiles:
            raise ValueError("basis archive is available only in archive profiles")
        if sn_basis_quality_archive_enabled is None:
            sn_basis_quality_archive_enabled = (
                profile == PROFILE_SN_ARCHIVE_DUAL_POOL_BEAM_EXPORT
            )
        if (
            sn_basis_quality_archive_enabled
            and profile != PROFILE_SN_ARCHIVE_DUAL_POOL_BEAM_EXPORT
        ):
            raise ValueError(
                "source-quality archive is available only in the dual-pool profile"
            )
        if sn_partial_residual_archive_enabled is None:
            sn_partial_residual_archive_enabled = (
                profile == PROFILE_SN_PARTIAL_RESIDUAL_ARCHIVE
            )
        if (
            sn_partial_residual_archive_enabled
            and profile != PROFILE_SN_PARTIAL_RESIDUAL_ARCHIVE
        ):
            raise ValueError("partial residual credit is available only in its profile")
        if sn_basis_pursuit_enabled is None:
            sn_basis_pursuit_enabled = profile in {
                PROFILE_SN_SCREENED_BASIS_PURSUIT,
                PROFILE_SN_SCREENED_PURSUIT_SHADOW,
                PROFILE_SN_PARETO_PURSUIT_SHADOW,
                PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
            }
        if sn_basis_pursuit_enabled and profile not in {
            PROFILE_SN_SCREENED_BASIS_PURSUIT,
            PROFILE_SN_SCREENED_PURSUIT_SHADOW,
            PROFILE_SN_PARETO_PURSUIT_SHADOW,
            PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
        }:
            raise ValueError("screened basis pursuit is available only in its profile")
        if sn_basis_pursuit_shortlist_size < 1:
            raise ValueError("basis-pursuit shortlist size must be positive")
        if sn_pursuit_stagnation_patience < 1:
            raise ValueError("pursuit stagnation patience must be positive")
        if (
            sn_pursuit_stagnation_epsilon_abs < 0.0
            or sn_pursuit_stagnation_epsilon_rel < 0.0
        ):
            raise ValueError("pursuit stagnation epsilons must be non-negative")
        if sn_basis_replacement_enabled is None:
            sn_basis_replacement_enabled = (
                profile == PROFILE_SN_BIDIRECTIONAL_BASIS_REPLACEMENT
            )
        if (
            sn_basis_replacement_enabled
            and profile != PROFILE_SN_BIDIRECTIONAL_BASIS_REPLACEMENT
        ):
            raise ValueError(
                "bidirectional basis replacement is available only in its profile"
            )
        if sn_basis_replacement_removal_shortlist_size < 1:
            raise ValueError("replacement removal shortlist size must be positive")
        if sn_conditional_basis_replacement_enabled is None:
            sn_conditional_basis_replacement_enabled = (
                profile == PROFILE_SN_CONDITIONAL_BASIS_REPLACEMENT
            )
        if (
            sn_conditional_basis_replacement_enabled
            and profile != PROFILE_SN_CONDITIONAL_BASIS_REPLACEMENT
        ):
            raise ValueError(
                "conditional basis replacement is available only in its profile"
            )
        if sn_orthogonal_basis_crossover_enabled is None:
            sn_orthogonal_basis_crossover_enabled = (
                profile == PROFILE_SN_ORTHOGONAL_BASIS_CROSSOVER
            )
        if (
            sn_orthogonal_basis_crossover_enabled
            and profile != PROFILE_SN_ORTHOGONAL_BASIS_CROSSOVER
        ):
            raise ValueError(
                "orthogonal basis crossover is available only in its profile"
            )
        if sn_orthogonal_basis_max_steps < 1:
            raise ValueError("orthogonal basis crossover steps must be positive")
        if sn_shadow_slot_enabled is None:
            sn_shadow_slot_enabled = profile in {
                PROFILE_SN_DUAL_PATH_SHADOW_SLOT,
                PROFILE_SN_SCREENED_PURSUIT_SHADOW,
                PROFILE_SN_PARETO_PURSUIT_SHADOW,
                PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
            }
        if sn_shadow_slot_enabled and profile not in {
            PROFILE_SN_DUAL_PATH_SHADOW_SLOT,
            PROFILE_SN_SCREENED_PURSUIT_SHADOW,
            PROFILE_SN_PARETO_PURSUIT_SHADOW,
            PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
        }:
            raise ValueError("shadow slot is available only in its profile")
        if sn_shadow_slot_enabled and sn_shadow_max_slots != 1:
            raise ValueError("dual-path shadow crossover freezes exactly one slot")
        if sn_shadow_minimum_coverage_gain < 0.0:
            raise ValueError("shadow minimum coverage gain must be non-negative")
        if not 0.0 <= sn_basis_forest_crossover_rate <= 1.0:
            raise ValueError("basis-forest SN crossover rate must lie in [0, 1]")
        if sn_basis_archive_capacity < 1:
            raise ValueError("basis archive capacity must be positive")
        if sn_basis_archive_source_candidates < 1:
            raise ValueError("basis archive source count must be positive")
        if sn_basis_quality_archive_capacity < 1:
            raise ValueError("source-quality archive capacity must be positive")
        super().__init__(
            variables=list(variables),
            binary=list(binary),
            unary=list(unary),
            max_params=0,
            elitism_k=elitism_k,
            population_size=population_size,
            tournament_size=tournament_size,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            const_range=None,
            depth_range=depth_range,
            full_prob=full_prob,
            nettype="scalar",
            n_jobs=None,
            random_state=random_state,
            n_iter=n_iter,
            use_tqdm=False,
            p_bfgs=0.0,
            fixed_constants=list(fixed_constants),
        )
        self.profile = profile
        self.use_basis_forest = profile in {
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
        self.use_basis_forest_sn_crossover = (
            profile == PROFILE_BASIS_FOREST_SN_CROSSOVER
        )
        self.use_basis_forest_sn_residual_infusion = profile in {
            PROFILE_BASIS_FOREST_SN_RESIDUAL_INFUSION,
            PROFILE_BASIS_FOREST_SN_CONSTRUCT_COMPRESS,
        }
        self.use_basis_forest_sn_safe_mutation = (
            profile == PROFILE_BASIS_FOREST_SN_SAFE_MUTATION
        )
        self.use_basis_forest_sn_verified_compression = profile in {
            PROFILE_BASIS_FOREST_SN_VERIFIED_COMPRESSION,
            PROFILE_BASIS_FOREST_SN_CONSTRUCT_COMPRESS,
            PROFILE_BASIS_FOREST_SN_REPAIR_COVERAGE,
            PROFILE_BASIS_FOREST_SN_COMPRESS_SHADOW,
            PROFILE_BASIS_FOREST_SN_COMPRESS_MUTATION_SHADOW,
            PROFILE_BASIS_FOREST_SN_COMPRESS_MULTISOURCE_SHADOW,
        }
        self.use_basis_forest_sn_residual_shadow = profile in {
            PROFILE_BASIS_FOREST_SN_RESIDUAL_SHADOW,
            PROFILE_BASIS_FOREST_SN_COMPRESS_SHADOW,
            PROFILE_BASIS_FOREST_SN_COMPRESS_MULTISOURCE_SHADOW,
        }
        self.use_basis_forest_sn_mutation_recombination = bool(
            sn_basis_forest_mutation_recombination_enabled
        )
        self.use_basis_forest_sn_mutation_shadow = bool(
            sn_basis_forest_mutation_shadow_enabled
        )
        self.use_legacy_sobolev = profile == PROFILE_SN
        self.use_sn_gp_v2 = profile == PROFILE_SN_V2
        self.use_sn_gp_stable = profile == PROFILE_SN_STABLE
        self.use_sn_verified_coverage_crossover = profile in {
            PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
            PROFILE_SN_CHILD_GEOMETRY,
            PROFILE_SN_BASIS_EXCHANGE,
            PROFILE_SN_BASIS_INFUSION,
            PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
        }
        self.use_sn_child_geometry = bool(sn_child_geometry_enabled)
        self.use_sn_basis_exchange = bool(sn_basis_exchange_enabled)
        self.use_sn_basis_infusion = bool(sn_basis_infusion_enabled)
        self.use_sn_residual_basis_infusion = bool(sn_residual_basis_infusion_enabled)
        self.use_sn_basis_archive = bool(sn_basis_archive_enabled)
        self.use_sn_archive_export = bool(sn_archive_export_enabled)
        self.sn_archive_export_max_steps = int(sn_archive_export_max_steps)
        self.use_sn_archive_anchor_export = bool(sn_archive_anchor_export_enabled)
        self.sn_archive_anchor_export_max_steps = int(
            sn_archive_anchor_export_max_steps
        )
        self.use_sn_archive_beam_export = bool(sn_archive_beam_export_enabled)
        self.use_sn_archive_dual_beam_export = profile in {
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
            *DEEP_ACCURACY_PROFILE_NAMES,
        }
        self.use_sn_archive_dual_pool_beam_export = (
            profile == PROFILE_SN_ARCHIVE_DUAL_POOL_BEAM_EXPORT
        )
        self.use_sn_archive_refined_dual_beam_export = (
            profile == PROFILE_SN_ARCHIVE_REFINED_DUAL_BEAM_EXPORT
        )
        self.use_sn_archive_floating_dual_beam_export = (
            profile == PROFILE_SN_ARCHIVE_FLOATING_DUAL_BEAM_EXPORT
        )
        self.use_sn_archive_balanced_screen_dual_beam_export = (
            profile == PROFILE_SN_ARCHIVE_BALANCED_SCREEN_DUAL_BEAM_EXPORT
        )
        self.use_sn_archive_balanced_floating_beam_export = (
            profile == PROFILE_SN_ARCHIVE_BALANCED_FLOATING_BEAM_EXPORT
        )
        self.use_sn_archive_innovation_screen_beam_export = (
            profile == PROFILE_SN_ARCHIVE_INNOVATION_SCREEN_BEAM_EXPORT
        )
        self.use_sn_archive_pair_lookahead_beam_export = (
            profile == PROFILE_SN_ARCHIVE_PAIR_LOOKAHEAD_BEAM_EXPORT
        )
        self.use_sn_archive_product_lift_beam_export = (
            profile == PROFILE_SN_ARCHIVE_PRODUCT_LIFT_BEAM_EXPORT
        )
        self.use_sn_archive_product_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_PRODUCT_ACCURACY_EXPORT
        )
        self.use_sn_archive_conditional_product_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_CONDITIONAL_PRODUCT_ACCURACY_EXPORT
        )
        self.use_sn_archive_iterated_conditional_product_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_ITERATED_CONDITIONAL_PRODUCT_ACCURACY_EXPORT
        )
        self.use_sn_archive_conditional_unary_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_CONDITIONAL_UNARY_ACCURACY_EXPORT
        )
        self.use_sn_archive_conditional_rational_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_CONDITIONAL_RATIONAL_ACCURACY_EXPORT
        )
        self.use_sn_archive_conditional_affine_unary_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_CONDITIONAL_AFFINE_UNARY_ACCURACY_EXPORT
        )
        self.use_sn_archive_conditional_radial_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_CONDITIONAL_RADIAL_ACCURACY_EXPORT
        )
        self.use_sn_archive_feature_radial_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_FEATURE_RADIAL_ACCURACY_EXPORT
        )
        self.use_sn_direct_feature_radial_accuracy_export = (
            profile == PROFILE_SN_DIRECT_FEATURE_RADIAL_ACCURACY_EXPORT
        )
        self.use_sn_direct_feature_affine_unary_accuracy_export = (
            profile == PROFILE_SN_DIRECT_FEATURE_AFFINE_UNARY_ACCURACY_EXPORT
        )
        self.use_sn_direct_feature_product_accuracy_export = (
            profile == PROFILE_SN_DIRECT_FEATURE_PRODUCT_ACCURACY_EXPORT
        )
        self.use_sn_direct_phase_accuracy_export = (
            profile == PROFILE_SN_DIRECT_PHASE_ACCURACY_EXPORT
        )
        self.use_sn_direct_phase_interaction_accuracy_export = (
            profile == PROFILE_SN_DIRECT_PHASE_INTERACTION_ACCURACY_EXPORT
        )
        self.use_sn_direct_radical_phase_accuracy_export = (
            profile == PROFILE_SN_DIRECT_RADICAL_PHASE_ACCURACY_EXPORT
        )
        self.use_sn_shared_phase_rational_accuracy_export = (
            profile == PROFILE_SN_SHARED_PHASE_RATIONAL_ACCURACY_EXPORT
        )
        self.use_sn_damped_exponential_accuracy_export = (
            profile == PROFILE_SN_DAMPED_EXPONENTIAL_ACCURACY_EXPORT
        )
        self.use_sn_affine_gaussian_accuracy_export = (
            profile == PROFILE_SN_AFFINE_GAUSSIAN_ACCURACY_EXPORT
        )
        self.use_sn_sinc_squared_accuracy_export = (
            profile == PROFILE_SN_SINC_SQUARED_ACCURACY_EXPORT
        )
        self.use_sn_shared_unary_polynomial_accuracy_export = (
            profile == PROFILE_SN_SHARED_UNARY_POLYNOMIAL_ACCURACY_EXPORT
        )
        self.use_sn_reciprocal_trig_accuracy_export = (
            profile == PROFILE_SN_RECIPROCAL_TRIG_ACCURACY_EXPORT
        )
        self.use_sn_cross_unary_affine_accuracy_export = (
            profile == PROFILE_SN_CROSS_UNARY_AFFINE_ACCURACY_EXPORT
        )
        self.use_sn_shared_denominator_accuracy_export = (
            profile == PROFILE_SN_SHARED_DENOMINATOR_ACCURACY_EXPORT
        )
        self.use_sn_relativistic_rational_accuracy_export = (
            profile == PROFILE_SN_RELATIVISTIC_RATIONAL_ACCURACY_EXPORT
        )
        self.use_sn_cosine_law_radial_accuracy_export = (
            profile == PROFILE_SN_COSINE_LAW_RADIAL_ACCURACY_EXPORT
        )
        self.use_sn_reciprocal_sine_square_accuracy_export = (
            profile == PROFILE_SN_RECIPROCAL_SINE_SQUARE_ACCURACY_EXPORT
        )
        self.use_sn_inverse_cosine_radial_accuracy_export = (
            profile == PROFILE_SN_INVERSE_COSINE_RADIAL_ACCURACY_EXPORT
        )
        self.use_sn_multiaxis_inverse_square_accuracy_export = (
            profile == PROFILE_SN_MULTIAXIS_INVERSE_SQUARE_ACCURACY_EXPORT
        )
        self.use_sn_interference_sine_ratio_accuracy_export = (
            profile == PROFILE_SN_INTERFERENCE_SINE_RATIO_ACCURACY_EXPORT
        )
        self.use_sn_sparse_radical_accuracy_export = (
            profile == PROFILE_SN_SPARSE_RADICAL_ACCURACY_EXPORT
        )
        self.use_sn_shared_ratio_trig_polynomial_accuracy_export = (
            profile == PROFILE_SN_SHARED_RATIO_TRIG_POLYNOMIAL_ACCURACY_EXPORT
        )
        self.deep_accuracy_stage = DEEP_ACCURACY_STAGE_BY_PROFILE.get(profile)
        self.use_sn_archive_value_only_accuracy_export = (
            profile == PROFILE_SN_ARCHIVE_VALUE_ONLY_ACCURACY_EXPORT
        )
        self.sn_archive_beam_export_max_steps = int(sn_archive_beam_export_max_steps)
        self.sn_archive_beam_width = int(sn_archive_beam_width)
        self.sn_archive_beam_shortlist_size = int(sn_archive_beam_shortlist_size)
        self.sn_archive_beam_value_shortlist_size = int(
            sn_archive_beam_value_shortlist_size
        )
        self.sn_archive_beam_innovation_shortlist_size = int(
            sn_archive_beam_innovation_shortlist_size
        )
        self.sn_archive_beam_pair_first_shortlist_size = int(
            sn_archive_beam_pair_first_shortlist_size
        )
        self.sn_archive_beam_pair_second_shortlist_size = int(
            sn_archive_beam_pair_second_shortlist_size
        )
        self.sn_archive_interaction_joint_shortlist_size = int(
            sn_archive_interaction_joint_shortlist_size
        )
        self.sn_archive_interaction_value_shortlist_size = int(
            sn_archive_interaction_value_shortlist_size
        )
        self.sn_archive_unary_transforms = archive_unary_transforms
        self.sn_direct_phase_scales = direct_phase_scales
        self.sn_direct_phase_include_squares = bool(sn_direct_phase_include_squares)
        self.sn_phase_interaction_batch_size = int(sn_phase_interaction_batch_size)
        self.sn_radical_scales = radical_scales
        self.sn_radical_max_dimension = int(sn_radical_max_dimension)
        self.sn_radical_phase_anchor_limit = int(sn_radical_phase_anchor_limit)
        self.sn_radical_amplitude_joint_shortlist_size = int(
            sn_radical_amplitude_joint_shortlist_size
        )
        self.sn_radical_amplitude_value_shortlist_size = int(
            sn_radical_amplitude_value_shortlist_size
        )
        self.sn_shared_phase_mobius_shifts = shared_phase_mobius_shifts
        self.sn_shared_phase_max_dimension = int(sn_shared_phase_max_dimension)
        self.sn_shared_phase_max_numerator_degree = int(
            sn_shared_phase_max_numerator_degree
        )
        self.sn_shared_phase_max_denominator_degree = int(
            sn_shared_phase_max_denominator_degree
        )
        self.sn_shared_phase_backbone_value_pool_size = int(
            sn_shared_phase_backbone_value_pool_size
        )
        self.sn_shared_phase_backbone_shortlist_size = int(
            sn_shared_phase_backbone_shortlist_size
        )
        self.sn_shared_phase_modulated_shortlist_size = int(
            sn_shared_phase_modulated_shortlist_size
        )
        self.sn_shared_phase_complement_pool_size = int(
            sn_shared_phase_complement_pool_size
        )
        self.sn_shared_phase_complement_shortlist_size = int(
            sn_shared_phase_complement_shortlist_size
        )
        self.sn_shared_phase_proposal_limit = int(sn_shared_phase_proposal_limit)
        self.sn_exponential_scales = exponential_scales
        self.sn_exponential_max_dimension = int(sn_exponential_max_dimension)
        self.sn_exponential_max_numerator_degree = int(
            sn_exponential_max_numerator_degree
        )
        self.sn_exponential_max_denominator_degree = int(
            sn_exponential_max_denominator_degree
        )
        self.sn_exponential_value_pool_size = int(sn_exponential_value_pool_size)
        self.sn_exponential_shortlist_size = int(sn_exponential_shortlist_size)
        self.sn_exponential_composite_value_pool_size = int(
            sn_exponential_composite_value_pool_size
        )
        self.sn_exponential_composite_shortlist_size = int(
            sn_exponential_composite_shortlist_size
        )
        self.sn_exponential_proposal_limit = int(sn_exponential_proposal_limit)
        self.sn_affine_gaussian_scales = affine_gaussian_scales
        self.sn_affine_gaussian_max_dimension = int(sn_affine_gaussian_max_dimension)
        self.sn_affine_gaussian_value_pool_size = int(
            sn_affine_gaussian_value_pool_size
        )
        self.sn_affine_gaussian_shortlist_size = int(sn_affine_gaussian_shortlist_size)
        self.sn_affine_gaussian_composite_value_pool_size = int(
            sn_affine_gaussian_composite_value_pool_size
        )
        self.sn_affine_gaussian_composite_shortlist_size = int(
            sn_affine_gaussian_composite_shortlist_size
        )
        self.sn_affine_gaussian_proposal_limit = int(sn_affine_gaussian_proposal_limit)
        self.sn_sinc_scales = sinc_scales
        self.sn_sinc_max_dimension = int(sn_sinc_max_dimension)
        self.sn_sinc_max_numerator_degree = int(sn_sinc_max_numerator_degree)
        self.sn_sinc_max_denominator_degree = int(sn_sinc_max_denominator_degree)
        self.sn_sinc_shape_value_pool_size = int(sn_sinc_shape_value_pool_size)
        self.sn_sinc_shape_shortlist_size = int(sn_sinc_shape_shortlist_size)
        self.sn_sinc_composite_value_pool_size = int(sn_sinc_composite_value_pool_size)
        self.sn_sinc_composite_shortlist_size = int(sn_sinc_composite_shortlist_size)
        self.sn_sinc_proposal_limit = int(sn_sinc_proposal_limit)
        self.sn_shared_unary_scales = shared_unary_scales
        self.sn_shared_unary_transforms = shared_unary_transforms
        self.sn_shared_unary_max_dimension = int(sn_shared_unary_max_dimension)
        self.sn_shared_unary_max_numerator_degree = int(
            sn_shared_unary_max_numerator_degree
        )
        self.sn_shared_unary_max_denominator_degree = int(
            sn_shared_unary_max_denominator_degree
        )
        self.sn_shared_unary_phase_value_pool_size = int(
            sn_shared_unary_phase_value_pool_size
        )
        self.sn_shared_unary_phase_shortlist_size = int(
            sn_shared_unary_phase_shortlist_size
        )
        self.sn_shared_unary_amplitude_value_pool_size = int(
            sn_shared_unary_amplitude_value_pool_size
        )
        self.sn_shared_unary_amplitude_shortlist_size = int(
            sn_shared_unary_amplitude_shortlist_size
        )
        self.sn_shared_unary_proposal_limit = int(sn_shared_unary_proposal_limit)
        self.sn_reciprocal_trig_scales = reciprocal_trig_scales
        self.sn_reciprocal_trig_transforms = reciprocal_trig_transforms
        self.sn_reciprocal_trig_max_dimension = int(sn_reciprocal_trig_max_dimension)
        self.sn_reciprocal_trig_value_pool_size = int(
            sn_reciprocal_trig_value_pool_size
        )
        self.sn_reciprocal_trig_shortlist_size = int(sn_reciprocal_trig_shortlist_size)
        self.sn_reciprocal_trig_proposal_limit = int(sn_reciprocal_trig_proposal_limit)
        self.sn_cross_unary_scales = cross_unary_scales
        self.sn_cross_unary_transforms = cross_unary_transforms
        self.sn_cross_unary_inner_powers = cross_unary_inner_powers
        self.sn_cross_unary_max_dimension = int(sn_cross_unary_max_dimension)
        self.sn_cross_unary_max_numerator_degree = int(
            sn_cross_unary_max_numerator_degree
        )
        self.sn_cross_unary_max_denominator_degree = int(
            sn_cross_unary_max_denominator_degree
        )
        self.sn_cross_unary_value_pool_size = int(sn_cross_unary_value_pool_size)
        self.sn_cross_unary_shortlist_size = int(sn_cross_unary_shortlist_size)
        self.sn_cross_unary_proposal_limit = int(sn_cross_unary_proposal_limit)
        self.sn_shared_denominator_signs = shared_denominator_signs
        self.sn_shared_denominator_numerator_signs = shared_denominator_numerator_signs
        self.sn_shared_denominator_max_dimension = int(
            sn_shared_denominator_max_dimension
        )
        self.sn_shared_denominator_value_pool_size = int(
            sn_shared_denominator_value_pool_size
        )
        self.sn_shared_denominator_shortlist_size = int(
            sn_shared_denominator_shortlist_size
        )
        self.sn_shared_denominator_proposal_limit = int(
            sn_shared_denominator_proposal_limit
        )
        self.sn_relativistic_rational_signs = relativistic_rational_signs
        self.sn_relativistic_rational_scales = relativistic_rational_scales
        self.sn_relativistic_rational_max_dimension = int(
            sn_relativistic_rational_max_dimension
        )
        self.sn_relativistic_rational_value_pool_size = int(
            sn_relativistic_rational_value_pool_size
        )
        self.sn_relativistic_rational_shortlist_size = int(
            sn_relativistic_rational_shortlist_size
        )
        self.sn_relativistic_rational_proposal_limit = int(
            sn_relativistic_rational_proposal_limit
        )
        self.sn_cosine_law_phase_signs = cosine_law_phase_signs
        self.sn_cosine_law_radial_signs = cosine_law_radial_signs
        self.sn_cosine_law_phase_scales = cosine_law_phase_scales
        self.sn_cosine_law_max_dimension = int(sn_cosine_law_max_dimension)
        self.sn_cosine_law_value_pool_size = int(sn_cosine_law_value_pool_size)
        self.sn_cosine_law_shortlist_size = int(sn_cosine_law_shortlist_size)
        self.sn_cosine_law_proposal_limit = int(sn_cosine_law_proposal_limit)
        self.sn_reciprocal_sine_scales = reciprocal_sine_scales
        self.sn_reciprocal_sine_max_dimension = int(sn_reciprocal_sine_max_dimension)
        self.sn_reciprocal_sine_max_numerator_degree = int(
            sn_reciprocal_sine_max_numerator_degree
        )
        self.sn_reciprocal_sine_value_pool_size = int(
            sn_reciprocal_sine_value_pool_size
        )
        self.sn_reciprocal_sine_shortlist_size = int(sn_reciprocal_sine_shortlist_size)
        self.sn_reciprocal_sine_proposal_limit = int(sn_reciprocal_sine_proposal_limit)
        self.sn_inverse_cosine_phase_signs = inverse_cosine_phase_signs
        self.sn_inverse_cosine_radial_signs = inverse_cosine_radial_signs
        self.sn_inverse_cosine_phase_scales = inverse_cosine_phase_scales
        self.sn_inverse_cosine_max_dimension = int(sn_inverse_cosine_max_dimension)
        self.sn_inverse_cosine_value_pool_size = int(sn_inverse_cosine_value_pool_size)
        self.sn_inverse_cosine_shortlist_size = int(sn_inverse_cosine_shortlist_size)
        self.sn_inverse_cosine_proposal_limit = int(sn_inverse_cosine_proposal_limit)
        self.sn_multiaxis_pair_signs = multiaxis_pair_signs
        self.sn_multiaxis_radial_term_counts = multiaxis_radial_term_counts
        self.sn_multiaxis_max_dimension = int(sn_multiaxis_max_dimension)
        self.sn_multiaxis_max_numerator_degree = int(sn_multiaxis_max_numerator_degree)
        self.sn_multiaxis_value_pool_size = int(sn_multiaxis_value_pool_size)
        self.sn_multiaxis_shortlist_size = int(sn_multiaxis_shortlist_size)
        self.sn_multiaxis_proposal_limit = int(sn_multiaxis_proposal_limit)
        self.sn_interference_sine_scales = interference_sine_scales
        self.sn_interference_sine_max_dimension = int(
            sn_interference_sine_max_dimension
        )
        self.sn_interference_sine_value_pool_size = int(
            sn_interference_sine_value_pool_size
        )
        self.sn_interference_sine_shortlist_size = int(
            sn_interference_sine_shortlist_size
        )
        self.sn_interference_sine_proposal_limit = int(
            sn_interference_sine_proposal_limit
        )
        self.sn_sparse_radical_offsets = sparse_radical_offsets
        self.sn_sparse_radical_scales = sparse_radical_scales
        self.sn_sparse_radical_max_dimension = int(sn_sparse_radical_max_dimension)
        self.sn_sparse_radical_max_abs_exponent = int(
            sn_sparse_radical_max_abs_exponent
        )
        self.sn_sparse_radical_max_numerator_degree = int(
            sn_sparse_radical_max_numerator_degree
        )
        self.sn_sparse_radical_max_denominator_degree = int(
            sn_sparse_radical_max_denominator_degree
        )
        self.sn_sparse_radical_integer_candidate_limit = int(
            sn_sparse_radical_integer_candidate_limit
        )
        self.sn_sparse_radical_value_pool_size = int(sn_sparse_radical_value_pool_size)
        self.sn_sparse_radical_shortlist_size = int(sn_sparse_radical_shortlist_size)
        self.sn_sparse_radical_proposal_limit = int(sn_sparse_radical_proposal_limit)
        self.sn_shared_ratio_trig_phase_scales = shared_ratio_trig_phase_scales
        self.sn_shared_ratio_trig_required_dimension = int(
            sn_shared_ratio_trig_required_dimension
        )
        self.sn_shared_ratio_trig_value_pool_size = int(
            sn_shared_ratio_trig_value_pool_size
        )
        self.sn_shared_ratio_trig_shortlist_size = int(
            sn_shared_ratio_trig_shortlist_size
        )
        self.sn_shared_ratio_trig_proposal_limit = int(
            sn_shared_ratio_trig_proposal_limit
        )
        supplied_deep_parameters = dict(sn_deep_accuracy_parameters or {})
        unknown_deep_parameters = set(supplied_deep_parameters).difference(
            DEEP_ACCURACY_PARAMETER_DEFAULTS
        )
        if unknown_deep_parameters:
            raise ValueError(
                "unknown deep accuracy parameters: "
                f"{sorted(unknown_deep_parameters)}"
            )
        for name, default in DEEP_ACCURACY_PARAMETER_DEFAULTS.items():
            value = supplied_deep_parameters.get(name, default)
            if isinstance(default, tuple):
                normalized: Any = tuple(float(item) for item in value)
            elif isinstance(default, int):
                normalized = int(value)
            else:
                normalized = float(value)
            setattr(self, name, normalized)
        for stage in DEEP_ACCURACY_STAGE_BY_CODE.values():
            if (
                getattr(self, stage.required_dimension_attribute)
                != stage.required_dimension
            ):
                raise ValueError(
                    f"Stage {stage.code} dimension is frozen at "
                    f"{stage.required_dimension}"
                )
        self.sn_archive_beam_diversity_slots = int(sn_archive_beam_diversity_slots)
        self.sn_archive_beam_refine_max_steps = int(sn_archive_beam_refine_max_steps)
        self.use_sn_basis_quality_archive = bool(sn_basis_quality_archive_enabled)
        self.sn_basis_quality_archive_capacity = int(sn_basis_quality_archive_capacity)
        self.use_sn_partial_residual_archive = bool(sn_partial_residual_archive_enabled)
        self.use_sn_basis_pursuit = bool(sn_basis_pursuit_enabled)
        self.use_sn_screened_pursuit_shadow = bool(
            profile
            in {
                PROFILE_SN_SCREENED_PURSUIT_SHADOW,
                PROFILE_SN_PARETO_PURSUIT_SHADOW,
                PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
            }
            and sn_basis_pursuit_enabled
            and sn_shadow_slot_enabled
        )
        self.use_sn_pareto_pursuit_shadow = bool(
            profile
            in {
                PROFILE_SN_PARETO_PURSUIT_SHADOW,
                PROFILE_SN_STAGNATION_PURSUIT_SHADOW,
            }
            and self.use_sn_screened_pursuit_shadow
        )
        self.use_sn_stagnation_pursuit_shadow = bool(
            profile == PROFILE_SN_STAGNATION_PURSUIT_SHADOW
            and self.use_sn_screened_pursuit_shadow
        )
        self.sn_basis_pursuit_shortlist_size = int(sn_basis_pursuit_shortlist_size)
        self.sn_pursuit_stagnation_patience = int(sn_pursuit_stagnation_patience)
        self.sn_pursuit_stagnation_epsilon_abs = float(
            sn_pursuit_stagnation_epsilon_abs
        )
        self.sn_pursuit_stagnation_epsilon_rel = float(
            sn_pursuit_stagnation_epsilon_rel
        )
        self.sn_pursuit_preserve_fallback_anchor = bool(
            sn_pursuit_preserve_fallback_anchor
        )
        self.sn_pursuit_select_by_coverage = bool(sn_pursuit_select_by_coverage)
        self.use_sn_basis_replacement = bool(sn_basis_replacement_enabled)
        self.sn_basis_replacement_removal_shortlist_size = int(
            sn_basis_replacement_removal_shortlist_size
        )
        self.use_sn_conditional_basis_replacement = bool(
            sn_conditional_basis_replacement_enabled
        )
        self.use_sn_orthogonal_basis_crossover = bool(
            sn_orthogonal_basis_crossover_enabled
        )
        self.sn_orthogonal_basis_max_steps = int(sn_orthogonal_basis_max_steps)
        self.use_sn_shadow_slot = bool(sn_shadow_slot_enabled)
        self.sn_shadow_max_slots = int(sn_shadow_max_slots)
        self.sn_shadow_minimum_coverage_gain = float(sn_shadow_minimum_coverage_gain)
        self.sn_shadow_allow_represented_amplification = bool(
            sn_shadow_allow_represented_amplification
        )
        self.sn_basis_forest_crossover_rate = float(sn_basis_forest_crossover_rate)
        self.use_sn_population_coverage = profile in {
            PROFILE_SN_REPAIR_COVERAGE,
            PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
            PROFILE_SN_CHILD_GEOMETRY,
            PROFILE_SN_BASIS_EXCHANGE,
            PROFILE_SN_BASIS_INFUSION,
            PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            PROFILE_BASIS_FOREST_SN_REPAIR_COVERAGE,
        }
        self.use_sn_gp_verified_repair = profile in {
            PROFILE_SN_VERIFIED_REPAIR,
            PROFILE_SN_REPAIR_COVERAGE,
            PROFILE_SN_VERIFIED_COVERAGE_CROSSOVER,
            PROFILE_SN_CHILD_GEOMETRY,
            PROFILE_SN_BASIS_EXCHANGE,
            PROFILE_SN_BASIS_INFUSION,
            PROFILE_SN_RESIDUAL_BASIS_INFUSION,
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
            PROFILE_BASIS_FOREST_SN_VERIFIED_COMPRESSION,
            PROFILE_BASIS_FOREST_SN_CONSTRUCT_COMPRESS,
            PROFILE_BASIS_FOREST_SN_REPAIR_COVERAGE,
            PROFILE_BASIS_FOREST_SN_COMPRESS_SHADOW,
            PROFILE_BASIS_FOREST_SN_COMPRESS_MUTATION_SHADOW,
            PROFILE_BASIS_FOREST_SN_COMPRESS_MULTISOURCE_SHADOW,
        }
        self.use_sn_repair_shadow = bool(sn_repair_shadow_enabled)
        self.use_sn_gp_plugin = self.use_sn_gp_v2 or self.use_sn_gp_stable
        self.use_sobolev = (
            self.use_legacy_sobolev
            or self.use_sn_gp_plugin
            or self.use_sn_gp_verified_repair
            or self.use_sn_repair_shadow
            or self.use_sn_archive_export
            or self.use_sn_archive_anchor_export
            or self.use_sn_archive_beam_export
        )
        self.enable_sobolev_evaluator = (
            self.use_sobolev
            or self.use_basis_forest_sn_crossover
            or self.use_basis_forest_sn_residual_infusion
            or self.use_basis_forest_sn_safe_mutation
            or self.use_basis_forest_sn_residual_shadow
            or self.use_basis_forest_sn_mutation_recombination
            or self.use_basis_forest_sn_mutation_shadow
        )
        self.fixed_constants = tuple(float(value) for value in fixed_constants)
        self.time_limit = float(time_limit)
        self.hard_time_limit = float(hard_time_limit)
        self.max_len = int(max_len)
        self.max_additive_terms = int(max_additive_terms)
        self.eta = float(eta)
        self.ratio = float(ratio)
        self.sobolev_alpha = float(sobolev_alpha)
        self.sobolev_tau = float(sobolev_tau if sn_tau is None else sn_tau)
        self.sobolev_lambda_value = float(sobolev_lambda_value)
        self.sobolev_lambda_gradient = float(sobolev_lambda_gradient)
        self.geometry_sample_size = int(
            geometry_sample_size
            if sn_geometry_sample_size is None
            else sn_geometry_sample_size
        )
        self.shortlist_size = int(shortlist_size)
        self.sobolev_failure_policy = sobolev_failure_policy
        self.sobolev_pruning = bool(
            sobolev_pruning
            and (self.use_legacy_sobolev or self.use_sn_gp_verified_repair)
        )
        self.sobolev_max_prunes = int(sobolev_max_prunes)
        self.prune_elite_k = int(prune_elite_k)
        self.sobolev_acceptance_tolerance = float(sobolev_acceptance_tolerance)
        self.incremental_max_changed_terms = int(incremental_max_changed_terms)
        self.dataset_identity = str(dataset_identity)
        self.initial_population_expressions = (
            None
            if initial_population_expressions is None
            else tuple(str(value) for value in initial_population_expressions)
        )
        self.initial_population_sha256 = initial_population_sha256
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.detailed_logging = bool(detailed_logging)
        self.record_integrity_metadata = bool(record_integrity_metadata)
        self.sn_compare_epsilon_abs = float(sn_compare_epsilon_abs)
        self.sn_compare_epsilon_rel = float(sn_compare_epsilon_rel)
        self.sn_mutation_delta = float(sn_mutation_delta)
        self.sn_mutation_gamma = float(sn_mutation_gamma)
        self.sn_mutation_impact_beta = float(sn_mutation_impact_beta)
        self.sn_mutation_impact_epsilon = float(sn_mutation_impact_epsilon)
        self.sn_mutation_max_normalized_impact = float(
            sn_mutation_max_normalized_impact
        )
        self.sn_base_anchor_enabled = bool(sn_base_anchor_enabled)
        self.sn_base_anchor_elite_slots = int(sn_base_anchor_elite_slots)
        self.sn_export_mode = str(sn_export_mode)
        self.sn_cache_enabled = bool(sn_cache_enabled)
        self.sn_provenance_enabled = bool(sn_provenance_enabled)
        self.sn_trace_every = int(sn_trace_every)
        self.record_reward_gap_samples = bool(record_reward_gap_samples)
        self.base_score_semantics = str(base_score_semantics)
        self.sn_repair_shortlist_size = int(sn_repair_shortlist_size)
        self.sn_repair_max_accepted_per_generation = int(
            sn_repair_max_accepted_per_generation
        )
        self.sn_coverage_shortlist_size = int(sn_coverage_shortlist_size)
        self.sn_coverage_elite_slots = int(sn_coverage_elite_slots)
        self.sn_coverage_max_base_reward_gap = float(sn_coverage_max_base_reward_gap)
        self.sn_coverage_crossover_rate = float(sn_coverage_crossover_rate)
        self.sn_basis_archive_capacity = int(sn_basis_archive_capacity)
        self.sn_basis_archive_source_candidates = int(
            sn_basis_archive_source_candidates
        )
        if self.sn_trace_every < 0:
            raise ValueError("sn_trace_every must be non-negative")
        self.counter_rng = CounterRNG(int(random_state))
        self.term_cache = TermEvaluationCache(
            enabled=self.enable_sobolev_evaluator
            and (self.sn_cache_enabled or self.use_legacy_sobolev),
            max_entries=term_cache_max_entries,
            max_memory_bytes=term_cache_max_memory_bytes,
            derivative_max_entries=derivative_cache_max_entries,
        )
        self.geometry_cache = CandidateGeometryCache(
            enabled=self.enable_sobolev_evaluator
            and (self.sn_cache_enabled or self.use_legacy_sobolev),
            max_entries=geometry_cache_max_entries,
            max_memory_bytes=geometry_cache_max_memory_bytes,
            decomposition_max_entries=50_000,
        )
        self.sobolev_evaluator = (
            SobolevEvaluator(
                SobolevConfig(
                    lambda_value=self.sobolev_lambda_value,
                    lambda_gradient=self.sobolev_lambda_gradient,
                    threshold=self.sobolev_tau,
                    min_valid_samples=32,
                    fast_gram=True,
                    gram_condition_threshold=1e6,
                    cache_enabled=(self.sn_cache_enabled or self.use_legacy_sobolev),
                    candidate_geometry_cache=(
                        self.sn_cache_enabled or self.use_legacy_sobolev
                    ),
                    output_scale_free_internal=True,
                    parent_child_incremental=True,
                    incremental_max_changed_terms=self.incremental_max_changed_terms,
                    geometry_cache_max_entries=geometry_cache_max_entries,
                    geometry_cache_max_memory_bytes=geometry_cache_max_memory_bytes,
                    decomposition_cache_max_entries=50_000,
                    geometry_sample_size=self.geometry_sample_size,
                    geometry_seed=int(random_state),
                ),
                cache=self.term_cache,
                geometry_cache=self.geometry_cache,
            )
            if self.enable_sobolev_evaluator
            else None
        )
        self.base_cache_max_entries = int(base_cache_max_entries)
        self.configuration = self._configuration_dict()
        self.configuration_sha256 = (
            hashlib.sha256(canonical_json_bytes(self.configuration)).hexdigest()
            if self.record_integrity_metadata
            else None
        )
        self.records: list[dict[str, Any]] = []
        self.population: list[ControlledIndividual] = []
        self.method_selected_best: ControlledIndividual | None = None
        self.base_quality_best_ever: ControlledIndividual | None = None
        self.current_structural_order: list[ControlledIndividual] = []
        self.current_elite_order: list[ControlledIndividual] = []
        self.time_floor_snapshot: dict[str, Any] | None = None
        self.status = "not_started"
        self.stats: Counter[str] = Counter()
        self.operator_counts: Counter[str] = Counter()
        self.failure_counts: Counter[str] = Counter()
        self.start_monotonic: float | None = None
        self.geometry_indices: np.ndarray | None = None
        self.additive_evaluator: AdditiveLinearEvaluator | None = None
        self.checkpoint_writer: GenerationCheckpointWriter | None = None
        self.initial_base_fitness_sha256: str | None = None
        self._active_X: np.ndarray | None = None
        self._active_y: np.ndarray | None = None
        self.sn_comparator = (
            SNComparator(
                self._lazy_sobolev_view,
                epsilon_abs=self.sn_compare_epsilon_abs,
                epsilon_rel=self.sn_compare_epsilon_rel,
                trace_sink=self._record_comparison_trace,
            )
            if self.use_sn_gp_plugin
            else None
        )
        self.term_provenance_builder = (
            TermProvenanceBuilder(tuple(value.name for value in variables))
            if self.use_sn_gp_plugin and self.sn_provenance_enabled
            else None
        )
        self.sn_mutation_selector = (
            SNMutationTargetSelector(
                tau=self.sobolev_tau,
                delta=self.sn_mutation_delta,
                gamma=self.sn_mutation_gamma,
                impact_beta=self.sn_mutation_impact_beta,
                impact_epsilon=self.sn_mutation_impact_epsilon,
                max_normalized_impact=self.sn_mutation_max_normalized_impact,
            )
            if self.use_sn_gp_plugin
            else None
        )
        self.sn_mutation_stats: Counter[str] = Counter()
        self._pending_mutation_events: list[dict[str, Any]] = []
        self.exported_candidate: ControlledIndividual | None = None
        self.archive_export_candidate: ControlledIndividual | None = None
        self.archive_export_pareto_candidate: ControlledIndividual | None = None
        self.archive_export_path: list[dict[str, Any]] = []
        self.archive_conditional_product_entries: tuple[BasisArchiveEntry, ...] = ()
        self.archive_conditional_product_contexts: tuple[tuple[int, ...], ...] = ()
        self.archive_iterated_product_entries: tuple[BasisArchiveEntry, ...] = ()
        self.archive_iterated_product_contexts: tuple[tuple[int, ...], ...] = ()
        self.archive_conditional_rational_entries: tuple[BasisArchiveEntry, ...] = ()
        self.archive_conditional_affine_unary_entries: tuple[
            BasisArchiveEntry, ...
        ] = ()
        self.archive_conditional_radial_entries: tuple[BasisArchiveEntry, ...] = ()
        self.archive_feature_radial_entries: tuple[BasisArchiveEntry, ...] = ()
        self.direct_feature_basis_entries: tuple[BasisArchiveEntry, ...] = ()
        self.direct_feature_radial_entries: tuple[BasisArchiveEntry, ...] = ()
        self.direct_feature_affine_unary_entries: tuple[BasisArchiveEntry, ...] = ()
        self.direct_feature_product_entries: tuple[BasisArchiveEntry, ...] = ()
        self.direct_phase_entries: tuple[BasisArchiveEntry, ...] = ()
        self.direct_phase_interaction_entries: tuple[BasisArchiveEntry, ...] = ()
        self.direct_radical_amplitude_entries: tuple[BasisArchiveEntry, ...] = ()
        self.direct_radical_composite_entries: tuple[BasisArchiveEntry, ...] = ()
        self.current_base_anchor: ControlledIndividual | None = None
        self.current_sn_elite_order: list[ControlledIndividual] = []
        self.current_coverage_candidate: ControlledIndividual | None = None
        self.current_coverage_gain: float | None = None
        self.current_coverage_base_rank: int | None = None
        self.current_coverage_slot_changed = False
        self.current_coverage_anchor_basis: np.ndarray | None = None
        self.sn_pursuit_gate_best_reward: float | None = None
        self.sn_pursuit_last_improvement_generation = 0
        self.basis_archive = (
            SobolevBasisArchive(self.sn_basis_archive_capacity)
            if self.use_sn_basis_archive
            else None
        )
        self.basis_quality_archive = (
            SourceQualityBasisArchive(self.sn_basis_quality_archive_capacity)
            if self.use_sn_basis_quality_archive
            else None
        )
        self.last_anchor_injected = False
        self.last_anchor_already_selected = False
        self.reward_gap_samples: dict[str, list[float]] = {
            "base_rewards": [],
            "tournament_gaps": [],
            "tournament_scales": [],
            "elite_boundary_gaps": [],
            "elite_boundary_scales": [],
        }

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | Mapping[str, np.ndarray],
        y: np.ndarray | pd.Series,
        *,
        use_tqdm: bool = False,
        early_stop=None,
    ) -> str:
        """Run complete-generation transactions until the strict floor."""

        del use_tqdm, early_stop  # protocol freezes both off
        points, feature_names = self._coerce_training_inputs(X)
        expected_names = tuple(variable.name for variable in self.variables)
        if feature_names != expected_names:
            raise ValueError(
                f"Variable order {expected_names} does not match data {feature_names}"
            )
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(target) != len(points) or not np.all(np.isfinite(target)):
            raise ValueError("Target must be finite and aligned with X")
        if self.record_integrity_metadata:
            content_identity = self._content_identity(points, target, feature_names)
            if self.dataset_identity == "dataset":
                self.dataset_identity = content_identity
            else:
                self.dataset_identity = f"{self.dataset_identity}|{content_identity}"
        self.additive_evaluator = AdditiveLinearEvaluator(
            feature_names,
            self.dataset_identity,
            eta=self.eta,
            max_genotype_len=self.max_len,
            max_additive_terms=self.max_additive_terms,
            cache_max_entries=self.base_cache_max_entries,
            score_semantics=self.base_score_semantics,
        )
        self.geometry_indices = select_geometry_indices(
            len(points),
            self.dataset_identity,
            int(self.random_state),
            self.geometry_sample_size,
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.output_dir / "geometry_indices.npy", self.geometry_indices)
            if self.record_integrity_metadata:
                self.checkpoint_writer = GenerationCheckpointWriter(
                    self.output_dir,
                    self.time_limit,
                )
        self.records = []
        self.stats.clear()
        self.operator_counts.clear()
        self.failure_counts.clear()
        self.sn_mutation_stats.clear()
        self._pending_mutation_events = []
        self.exported_candidate = None
        self.archive_export_candidate = None
        self.archive_export_pareto_candidate = None
        self.archive_export_path = []
        self.archive_conditional_product_entries = ()
        self.archive_conditional_product_contexts = ()
        self.archive_iterated_product_entries = ()
        self.archive_iterated_product_contexts = ()
        self.archive_conditional_rational_entries = ()
        self.archive_conditional_affine_unary_entries = ()
        self.archive_conditional_radial_entries = ()
        self.archive_feature_radial_entries = ()
        self.direct_feature_basis_entries = ()
        self.direct_feature_radial_entries = ()
        self.direct_feature_affine_unary_entries = ()
        self.direct_feature_product_entries = ()
        self.direct_phase_entries = ()
        self.direct_phase_interaction_entries = ()
        self.direct_radical_amplitude_entries = ()
        self.direct_radical_composite_entries = ()
        self.current_base_anchor = None
        self.current_sn_elite_order = []
        self.current_coverage_candidate = None
        self.current_coverage_gain = None
        self.current_coverage_base_rank = None
        self.current_coverage_slot_changed = False
        self.current_coverage_anchor_basis = None
        self.sn_pursuit_gate_best_reward = None
        self.sn_pursuit_last_improvement_generation = 0
        if self.basis_archive is not None:
            self.basis_archive.clear()
        if self.basis_quality_archive is not None:
            self.basis_quality_archive.clear()
        self.last_anchor_injected = False
        self.last_anchor_already_selected = False
        for values in self.reward_gap_samples.values():
            values.clear()
        if self.sn_comparator is not None:
            self.sn_comparator.stats.clear()
        self.method_selected_best = None
        self.base_quality_best_ever = None
        self._active_X = points
        self._active_y = target
        self.start_monotonic = time.monotonic()

        expressions = self._initial_expressions()
        if self.record_integrity_metadata:
            self.initial_population_sha256 = self._expressions_sha256(expressions)
        population: list[ControlledIndividual] = []
        for slot, expression in enumerate(expressions):
            individual = ControlledIndividual(
                nd.parse(expression),
                candidate_id=self._proposal_id(0, slot),
                generation=0,
                slot=slot,
            )
            self._evaluate_individual(individual, points, target)
            if self.use_basis_forest:
                self._assign_initial_basis_genome(
                    individual,
                    points,
                    target,
                )
            population.append(individual)
        if self.record_integrity_metadata:
            self.initial_base_fitness_sha256 = self._base_metrics_sha256(population)
        self._apply_structural_selection(
            population,
            points,
            target,
            enforce_deadline=False,
        )
        self.population = population
        snapshot = self._commit_generation(0, population)
        if float(snapshot["wall_time"]) > self.time_limit:
            self.status = "no_generation_completed_at_or_before_floor"
            self._select_exported_tree()
            return self.status
        if self._deadline_reached():
            self.status = "time_limit"
            self._select_exported_tree()
            return self.status

        self.status = "iter_limit"
        for generation in range(1, self.n_iter + 1):
            try:
                next_population = self._evolve_generation(
                    generation,
                    population,
                    points,
                    target,
                )
            except GenerationDeadline:
                self.stats["partial_generations_discarded"] += 1
                self.status = "time_limit"
                break
            population = next_population
            self.population = population
            self._commit_generation(generation, population)
            if self._deadline_reached():
                self.status = "time_limit"
                break
        self._select_exported_tree()
        return self.status

    def predict(
        self, X: np.ndarray | pd.DataFrame | Mapping[str, np.ndarray]
    ) -> np.ndarray:
        if self.eqtree is None:
            raise ValueError("Model has not produced a fitted phenotype")
        points, names = self._coerce_training_inputs(X)
        values = {name: points[:, index] for index, name in enumerate(names)}
        with np.errstate(all="ignore"):
            prediction = np.asarray(
                self.eqtree.eval(values, use_eps=1e-6),
                dtype=float,
            )
        if prediction.ndim == 0:
            prediction = np.full(len(points), float(prediction))
        prediction = prediction.reshape(-1)
        prediction[~np.isfinite(prediction)] = 0.0
        return prediction

    def generate_initial_population_expressions(self) -> tuple[str, ...]:
        expressions: list[str] = []
        for slot in range(self.population_size):
            expression = None
            for attempt in range(32):
                rng = self.counter_rng.generator(0, slot, "init_tree", attempt)
                candidate = self.generator.generate_eqtree(self.nettype, rng=rng)
                if len(candidate) > self.max_len:
                    self.stats["initial_bloat_rejections"] += 1
                    continue
                serialized = candidate.to_str(number_format=".17g")
                try:
                    nd.parse(serialized)
                except Exception:
                    self.stats["initial_unparseable_rejections"] += 1
                    continue
                expression = serialized
                break
            if expression is None:
                expression = self.variables[slot % len(self.variables)].to_str(
                    number_format=".17g"
                )
                self.stats["initial_variable_fallbacks"] += 1
            expressions.append(expression)
        return tuple(expressions)

    def _initial_expressions(self) -> tuple[str, ...]:
        expressions = (
            self.generate_initial_population_expressions()
            if self.initial_population_expressions is None
            else self.initial_population_expressions
        )
        if len(expressions) != self.population_size:
            raise ValueError(
                f"Initial population has {len(expressions)} expressions; "
                f"expected {self.population_size}"
            )
        if self.initial_population_sha256 is not None:
            actual = self._expressions_sha256(expressions)
            if actual != self.initial_population_sha256:
                raise ValueError(
                    "Initial population does not match the supplied reference"
                )
        return tuple(expressions)

    @staticmethod
    def _basis_fit_identity(
        original: AdditiveFitResult,
        rebuilt: AdditiveFitResult,
    ) -> bool:
        return bool(
            original.success
            and rebuilt.success
            and original.canonical_fitted_expression
            == rebuilt.canonical_fitted_expression
            and original.complexity == rebuilt.complexity
            and np.isclose(original.r2, rebuilt.r2, atol=1e-12, rtol=1e-12)
            and np.isclose(
                original.base_reward,
                rebuilt.base_reward,
                atol=1e-12,
                rtol=1e-12,
            )
        )

    def _assign_initial_basis_genome(
        self,
        individual: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        assert self.additive_evaluator is not None
        original = individual.fit_result
        if original is None or not original.success:
            individual.basis_genome = opaque_basis_genome(individual.raw_expression)
            individual.basis_forest_status = "opaque_invalid_base"
            self.stats["basis_forest_initial_opaque_invalid"] += 1
            return
        self.stats["basis_forest_initial_valid_total"] += 1
        genome, status = extract_strict_or_opaque_basis_genome(
            individual.raw_expression,
            self.feature_names,
            max_len=self.max_len,
        )
        if status == "exact_terms":
            rebuilt_tree = build_basis_tree_direct(
                genome,
                self.feature_names,
                max_len=self.max_len,
                verify_roundtrip=True,
            )
            rebuilt = self.additive_evaluator.evaluate(rebuilt_tree, X, y)
            self.stats["basis_forest_initial_rebuild_evaluations"] += 1
            if self._basis_fit_identity(original, rebuilt):
                individual.basis_genome = genome
                individual.basis_forest_status = "exact_terms"
                self.stats["basis_forest_initial_exact"] += 1
                return
            self.stats["basis_forest_initial_fit_mismatch"] += 1
        else:
            self.stats["basis_forest_initial_mapping_failure"] += 1
        individual.basis_genome = opaque_basis_genome(individual.raw_expression)
        individual.basis_forest_status = "opaque_macro"
        self.stats["basis_forest_initial_opaque_valid"] += 1

    def _basis_genome_of(self, candidate: ControlledIndividual) -> BasisGenome:
        return (
            candidate.basis_genome
            if candidate.basis_genome is not None
            else opaque_basis_genome(candidate.raw_expression)
        )

    def _select_basis_forest_safe_mutation_gene(
        self,
        *,
        parent: ControlledIndividual,
        original_index: int,
        X: np.ndarray,
        generation: int,
        slot: int,
    ) -> int:
        self.stats["basis_forest_safe_mutation_parent_events"] += 1
        if self.sn_mutation_impact_beta <= 0.0:
            self.stats["basis_forest_safe_mutation_disabled"] += 1
            return original_index
        genome = self._basis_genome_of(parent)
        if not genome.is_exact:
            self.stats["basis_forest_safe_mutation_fallback:opaque_genome"] += 1
            return original_index
        geometry = self._basis_exchange_geometry(
            parent, X, "basis_forest_mutation_parent"
        )
        fit = parent.fit_result
        if geometry is None or fit is None:
            self.stats["basis_forest_safe_mutation_fallback:geometry_unavailable"] += 1
            return original_index
        by_canonical = {
            term.canonical: index
            for index, term in enumerate(geometry["terms"])
            if term.basis.free_symbols
        }
        indices = [by_canonical.get(gene.canonical) for gene in genome.terms]
        inactive = tuple(index is None for index in indices)
        self.stats["basis_forest_safe_mutation_inactive_genes"] += sum(inactive)
        coefficients = tuple(
            0.0 if index is None else geometry["terms"][index].coefficient
            for index in indices
        )
        novelties = tuple(
            0.0 if index is None else geometry["novelties"][index] for index in indices
        )
        norms = tuple(
            0.0 if index is None else geometry["norms"][index] for index in indices
        )
        try:
            selection = select_impact_aware_basis_gene(
                coefficients=coefficients,
                novelties=novelties,
                term_norms=norms,
                original_index=original_index,
                rng=self.counter_rng.generator(
                    generation,
                    slot,
                    "reproduction",
                    attempt=2,
                ),
                tau=self.sobolev_tau,
                delta=self.sn_mutation_delta,
                gamma=self.sn_mutation_gamma,
                impact_beta=self.sn_mutation_impact_beta,
                impact_epsilon=self.sn_mutation_impact_epsilon,
                max_normalized_impact=(self.sn_mutation_max_normalized_impact),
            )
        except ValueError as error:
            self.stats[
                f"basis_forest_safe_mutation_fallback:{type(error).__name__}"
            ] += 1
            return original_index
        if not selection.targeted_distribution:
            self.stats[
                f"basis_forest_safe_mutation_fallback:{selection.fallback_reason}"
            ] += 1
            return original_index
        self.stats["basis_forest_safe_mutation_targeted_distribution"] += 1
        self.stats["basis_forest_safe_mutation_selected_eligible"] += int(
            selection.selected_eligible
        )
        self.stats["basis_forest_safe_mutation_redirected"] += int(
            selection.selected_index != original_index
        )
        self.stats["basis_forest_safe_mutation_selected_novelty_sum"] += float(
            novelties[selection.selected_index]
        )
        self.stats["basis_forest_safe_mutation_selected_inactive"] += int(
            inactive[selection.selected_index]
        )
        if selection.impacts is not None:
            self.stats[
                "basis_forest_safe_mutation_selected_normalized_impact_sum"
            ] += selection.impacts.normalized_impacts[selection.selected_index]
        return selection.selected_index

    def _basis_forest_mutation(
        self,
        method: str,
        parent: ControlledIndividual,
        rng: np.random.Generator,
        *,
        X: np.ndarray | None = None,
        generation: int | None = None,
        slot: int | None = None,
    ) -> tuple[nd.Symbol, BasisGenome, str]:
        parent_genome = self._basis_genome_of(parent)
        source_genes = (
            parent_genome.terms
            if parent_genome.terms
            else opaque_basis_genome(parent.raw_expression).terms
        )
        gene_index = int(rng.integers(0, len(source_genes)))
        if self.use_basis_forest_sn_safe_mutation:
            if X is None or generation is None or slot is None:
                raise ValueError("safe basis mutation requires generation context")
            gene_index = self._select_basis_forest_safe_mutation_gene(
                parent=parent,
                original_index=gene_index,
                X=X,
                generation=generation,
                slot=slot,
            )
        source_gene = source_genes[gene_index]
        dummy = NDIndividual(nd.parse(source_gene.expression))
        if method == "subtree-mutation":
            mutated = super().subtree_mutation(dummy, rng=rng)
        elif method == "hoist-mutation":
            mutated = super().hoist_mutation(dummy, rng=rng)
        elif method == "point-mutation":
            mutated = super().point_mutation(dummy, rng=rng)
        else:
            raise ValueError(f"Unsupported basis-forest mutation {method}")
        replacement, _ = extract_strict_or_opaque_basis_genome(
            mutated.eqtree.to_str(number_format=".17g"),
            self.feature_names,
            max_len=self.max_len,
        )
        genes = list(source_genes)
        genes[gene_index : gene_index + 1] = replacement.terms
        unique = {gene.canonical: gene for gene in genes}
        genome = BasisGenome(tuple(unique[key] for key in sorted(unique)))
        tree = build_basis_tree_direct(
            genome,
            self.feature_names,
            max_len=self.max_len,
            verify_roundtrip=False,
        )
        self.stats["basis_forest_mutation_events"] += 1
        self.stats["basis_forest_mutated_exact_genes"] += int(not source_gene.opaque)
        self.stats["basis_forest_mutated_opaque_genes"] += int(source_gene.opaque)
        return tree, genome, f"basis-{method}"

    def _basis_forest_crossover(
        self,
        parent: ControlledIndividual,
        donor: ControlledIndividual,
        rng: np.random.Generator,
    ) -> tuple[nd.Symbol, BasisGenome, str]:
        result = select_uniform_basis_crossover(
            self._basis_genome_of(parent),
            self._basis_genome_of(donor),
            self.feature_names,
            max_len=self.max_len,
            rng=rng,
        )
        self.stats["basis_forest_crossover_events"] += 1
        self.stats["basis_forest_crossover_budget_removals"] += int(
            result.removed_for_budget
        )
        self.stats["basis_forest_crossover_donor_only_genes"] += int(
            result.donor_only_genes
        )
        if result.fallback_reason is not None:
            self.stats[f"basis_forest_crossover_fallback:{result.fallback_reason}"] += 1
            return (
                parent.eqtree.copy(),
                self._basis_genome_of(parent),
                "basis-crossover_fallback",
            )
        tree = build_basis_tree_direct(
            result.genome,
            self.feature_names,
            max_len=self.max_len,
            verify_roundtrip=False,
        )
        self.stats["basis_forest_crossover_success"] += 1
        self.stats["basis_forest_cross_parent_events"] += int(
            result.donor_only_genes > 0
        )
        return tree, result.genome, "basis-crossover"

    def _basis_forest_sn_crossover_gate(self, generation: int, slot: int) -> bool:
        self.stats["basis_forest_sn_gate_trials"] += 1
        selected = bool(
            self.counter_rng.generator(
                generation,
                slot,
                "reproduction",
                attempt=1,
            ).random()
            < self.sn_basis_forest_crossover_rate
        )
        self.stats["basis_forest_sn_gate_selected"] += int(selected)
        return selected

    def _verified_basis_forest_sn_crossover(
        self,
        *,
        parent: ControlledIndividual,
        donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_genome: BasisGenome,
        original_variation: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, donor),
            original_variation,
        )
        original.basis_genome = copy.deepcopy(original_genome)
        original.basis_forest_status = (
            "exact_terms" if original_genome.is_exact else "mixed_opaque"
        )
        self._evaluate_individual(original, X, y)
        self.stats["basis_forest_sn_events"] += 1
        proposals = self._build_orthogonal_basis_crossover_proposals(
            parent=parent,
            donor=donor,
            X=X,
            y=y,
        )
        self.stats["basis_forest_sn_proposals"] += len(proposals)
        if not proposals:
            self.stats["basis_forest_sn_original_kept"] += 1
            return original

        evaluated: list[tuple[ControlledIndividual, dict[str, Any]]] = []
        for tree, details in proposals:
            alternative = self._new_individual(
                tree,
                generation,
                slot,
                (parent, donor),
                "basis_forest_sn_alternative",
            )
            genome, status = extract_strict_or_opaque_basis_genome(
                alternative.raw_expression,
                self.feature_names,
                max_len=self.max_len,
            )
            alternative.basis_genome = genome
            alternative.basis_forest_status = status
            self._evaluate_individual(alternative, X, y)
            self.stats["basis_forest_sn_alternatives_evaluated"] += 1
            self.stats["basis_forest_sn_invalid_alternatives"] += int(
                not alternative.valid
            )
            evaluated.append((alternative, details))

        best, best_details = min(evaluated, key=lambda item: self._base_key(item[0]))
        accepted = self._verified_proposal_better(best, original)
        if accepted:
            best.variation_type = "basis_forest_sn_verified"
            self.stats["basis_forest_sn_accepted"] += 1
            self.stats["basis_forest_sn_accepted_cross_parent"] += int(
                bool(best_details["uses_receiver_and_donor"])
            )
            if np.isfinite(best.base_reward) and np.isfinite(original.base_reward):
                self.stats["basis_forest_sn_reward_gain_sum"] += float(
                    best.base_reward
                ) - float(original.base_reward)
            chosen = best
        else:
            self.stats["basis_forest_sn_rejected"] += 1
            self.stats["basis_forest_sn_original_kept"] += 1
            chosen = original
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_forest_sn_crossover_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "donor_id": donor.candidate_id,
                    "original": self._candidate_summary(original),
                    "base_best_proposal": self._candidate_summary(best),
                    "base_best_pursuit": best_details,
                    "accepted": accepted,
                },
            )
        return chosen

    def _build_basis_forest_residual_infusion_proposal(
        self,
        *,
        original: ControlledIndividual,
        donor: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[nd.Symbol, BasisGenome, ResidualInfusionChoice, dict[str, Any]] | None:
        """Add or exchange one donor gene using child-conditional SN credit."""

        original_genome = self._basis_genome_of(original)
        donor_genome = self._basis_genome_of(donor)
        if not original_genome.is_exact or not donor_genome.is_exact:
            self.stats["basis_forest_residual_fallbacks"] += 1
            self.stats["basis_forest_residual_fallback:opaque_genome"] += 1
            return None
        original_geometry = self._basis_exchange_geometry(
            original, X, "basis_forest_child"
        )
        donor_geometry = self._basis_exchange_geometry(donor, X, "basis_forest_donor")
        if original_geometry is None or donor_geometry is None:
            self.stats["basis_forest_residual_fallbacks"] += 1
            self.stats["basis_forest_residual_fallback:geometry_unavailable"] += 1
            return None
        original_fit = original.fit_result
        donor_fit = donor.fit_result
        assert original_fit is not None and original_fit.fitted_tree is not None
        assert donor_fit is not None
        values = {name: X[:, index] for index, name in enumerate(self.feature_names)}
        try:
            with np.errstate(all="ignore"):
                prediction = np.asarray(
                    original_fit.fitted_tree.eval(values, use_eps=1e-6),
                    dtype=float,
                )
            if prediction.ndim == 0:
                prediction = np.full(len(y), float(prediction))
            prediction = prediction.reshape(-1)
            prediction[~np.isfinite(prediction)] = 0.0
            residual = np.asarray(y, dtype=float).reshape(-1) - prediction

            original_terms = original_geometry["terms"]
            design_columns = [np.ones(len(X), dtype=float)]
            for term in original_terms:
                if not term.basis.free_symbols:
                    continue
                column = np.asarray(
                    evaluate_sympy(
                        term.basis,
                        original_fit.symbols,
                        X,
                        protected_epsilon=1e-6,
                    ),
                    dtype=float,
                ).reshape(-1)
                column[~np.isfinite(column)] = 0.0
                design_columns.append(column)
            design = np.column_stack(design_columns)

            donor_correlations: list[float] = []
            for term in donor_geometry["terms"]:
                if not term.basis.free_symbols:
                    donor_correlations.append(0.0)
                    continue
                column = np.asarray(
                    evaluate_sympy(
                        term.basis,
                        donor_fit.symbols,
                        X,
                        protected_epsilon=1e-6,
                    ),
                    dtype=float,
                ).reshape(-1)
                column[~np.isfinite(column)] = 0.0
                correlation, _ = partial_residual_credit(column, residual, design)
                donor_correlations.append(correlation)

            donor_terms = donor_geometry["terms"]
            choice = select_residual_basis_infusion(
                parent_canonicals=tuple(term.canonical for term in original_terms),
                parent_coefficients=tuple(term.coefficient for term in original_terms),
                parent_novelties=original_geometry["novelties"],
                parent_norms=original_geometry["norms"],
                parent_signatures=original_geometry["signatures"],
                parent_structural=tuple(
                    bool(term.basis.free_symbols) for term in original_terms
                ),
                donor_canonicals=tuple(term.canonical for term in donor_terms),
                donor_coefficients=tuple(term.coefficient for term in donor_terms),
                donor_norms=donor_geometry["norms"],
                donor_signatures=donor_geometry["signatures"],
                donor_structural=tuple(
                    bool(term.basis.free_symbols) for term in donor_terms
                ),
                donor_target_correlations=tuple(donor_correlations),
                tau=self.sobolev_tau,
            )
        except (TypeError, ValueError, KeyError, np.linalg.LinAlgError) as error:
            self.stats["basis_forest_residual_fallbacks"] += 1
            self.stats[f"basis_forest_residual_fallback:{type(error).__name__}"] += 1
            return None
        if choice is None:
            self.stats["basis_forest_residual_fallbacks"] += 1
            self.stats["basis_forest_residual_fallback:no_joint_donor_signal"] += 1
            return None

        original_by_key = {gene.canonical: gene for gene in original_genome.terms}
        donor_by_key = {gene.canonical: gene for gene in donor_genome.terms}
        selected_canonical = donor_terms[choice.selected_donor_index].canonical
        removed_canonical = (
            None
            if choice.removed_parent_index is None
            else original_terms[choice.removed_parent_index].canonical
        )
        selected_gene = donor_by_key.get(selected_canonical)
        if selected_gene is None or (
            removed_canonical is not None and removed_canonical not in original_by_key
        ):
            self.stats["basis_forest_residual_fallbacks"] += 1
            self.stats["basis_forest_residual_fallback:gene_alignment"] += 1
            return None
        genes = [
            gene
            for gene in original_genome.terms
            if gene.canonical != removed_canonical
        ]
        if selected_canonical not in {gene.canonical for gene in genes}:
            genes.append(selected_gene)
        proposal_genome = BasisGenome(
            tuple(sorted(genes, key=lambda gene: gene.canonical))
        )
        try:
            tree = build_basis_tree_direct(
                proposal_genome,
                self.feature_names,
                max_len=self.max_len,
                verify_roundtrip=True,
            )
        except ValueError as error:
            self.stats["basis_forest_residual_fallbacks"] += 1
            reason = "max_len" if "AST length" in str(error) else "roundtrip"
            self.stats[f"basis_forest_residual_fallback:{reason}"] += 1
            return None

        self.stats["basis_forest_residual_proposals"] += 1
        self.stats[f"basis_forest_residual_{choice.action}_proposals"] += 1
        self.stats[
            "basis_forest_residual_target_correlation_sum"
        ] += choice.donor_target_correlation
        self.stats["basis_forest_residual_gain_sum"] += choice.donor_residual_gain
        self.stats["basis_forest_residual_joint_score_sum"] += choice.donor_joint_score
        details = {
            "action": choice.action,
            "selected_donor_term": selected_canonical,
            "removed_child_term": removed_canonical,
            "donor_partial_residual_correlation": (choice.donor_target_correlation),
            "donor_sobolev_residual_gain": choice.donor_residual_gain,
            "donor_joint_score": choice.donor_joint_score,
            "removed_deletion_impact": choice.removed_deletion_impact,
            "removed_normalized_impact": choice.removed_normalized_impact,
            "proposal_expression": tree.to_str(number_format=".17g"),
        }
        return tree, proposal_genome, choice, details

    def _verified_basis_forest_residual_infusion(
        self,
        *,
        donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_genome: BasisGenome,
        original_variation: str,
        parents: Sequence[ControlledIndividual],
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        original = self._new_individual(
            original_tree,
            generation,
            slot,
            parents,
            original_variation,
        )
        original.basis_genome = copy.deepcopy(original_genome)
        original.basis_forest_status = (
            "exact_terms" if original_genome.is_exact else "mixed_opaque"
        )
        self._evaluate_individual(original, X, y)
        self.stats["basis_forest_residual_events"] += 1
        proposal = self._build_basis_forest_residual_infusion_proposal(
            original=original,
            donor=donor,
            X=X,
            y=y,
        )
        if proposal is None:
            self.stats["basis_forest_residual_original_kept"] += 1
            return original
        tree, genome, choice, details = proposal
        alternative = self._new_individual(
            tree,
            generation,
            slot,
            parents,
            "basis_forest_residual_alternative",
        )
        alternative.basis_genome = genome
        alternative.basis_forest_status = "exact_terms"
        self._evaluate_individual(alternative, X, y)
        self.stats["basis_forest_residual_alternatives_evaluated"] += 1
        accepted = self._verified_proposal_better(alternative, original)
        if accepted:
            alternative.variation_type = (
                f"basis_forest_residual_{choice.action}_verified"
            )
            self.stats["basis_forest_residual_accepted"] += 1
            self.stats[f"basis_forest_residual_{choice.action}_accepted"] += 1
            if np.isfinite(alternative.base_reward) and np.isfinite(
                original.base_reward
            ):
                self.stats["basis_forest_residual_reward_gain_sum"] += float(
                    alternative.base_reward
                ) - float(original.base_reward)
            chosen = alternative
        else:
            self.stats["basis_forest_residual_rejected"] += 1
            self.stats["basis_forest_residual_original_kept"] += 1
            chosen = original
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_forest_residual_infusion_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "donor_id": donor.candidate_id,
                    "choice": details,
                    "original": self._candidate_summary(original),
                    "alternative": self._candidate_summary(alternative),
                    "accepted": accepted,
                },
            )
        return chosen

    def _verified_basis_forest_mutation_recombination(
        self,
        *,
        parent: ControlledIndividual,
        original_tree: nd.Symbol,
        original_genome: BasisGenome,
        original_variation: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Reuse de-novo mutation genes in one Base-verified SN term edit."""

        mutated = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent,),
            original_variation,
        )
        mutated.basis_genome = copy.deepcopy(original_genome)
        mutated.basis_forest_status = (
            "exact_terms" if original_genome.is_exact else "mixed_opaque"
        )
        self._evaluate_individual(mutated, X, y)
        self.stats["basis_forest_mutation_recombination_events"] += 1

        parent_keys = set(self._basis_genome_of(parent).canonicals)
        new_gene_count = sum(
            canonical not in parent_keys for canonical in original_genome.canonicals
        )
        self.stats["basis_forest_mutation_recombination_new_genes"] += int(
            new_gene_count
        )
        self.stats["basis_forest_mutation_recombination_events_with_new_genes"] += int(
            new_gene_count > 0
        )

        proposal = self._build_basis_forest_residual_infusion_proposal(
            original=parent,
            donor=mutated,
            X=X,
            y=y,
        )
        if proposal is None:
            self.stats["basis_forest_mutation_recombination_no_proposal"] += 1
            self.stats["basis_forest_mutation_recombination_original_kept"] += 1
            return mutated

        tree, genome, choice, details = proposal
        self.stats["basis_forest_mutation_recombination_proposals"] += 1
        self.stats[
            f"basis_forest_mutation_recombination_{choice.action}_proposals"
        ] += 1
        alternative = self._new_individual(
            tree,
            generation,
            slot,
            (parent,),
            "basis_forest_mutation_recombination_alternative",
        )
        alternative.basis_genome = genome
        alternative.basis_forest_status = "exact_terms"
        self._evaluate_individual(alternative, X, y)
        self.stats["basis_forest_mutation_recombination_alternatives_evaluated"] += 1

        accepted = self._verified_proposal_better(alternative, mutated)
        if accepted:
            alternative.variation_type = (
                f"basis_forest_mutation_residual_{choice.action}_verified"
            )
            self.stats["basis_forest_mutation_recombination_accepted"] += 1
            self.stats[
                f"basis_forest_mutation_recombination_{choice.action}_accepted"
            ] += 1
            if np.isfinite(alternative.base_reward) and np.isfinite(
                mutated.base_reward
            ):
                self.stats[
                    "basis_forest_mutation_recombination_reward_gain_sum"
                ] += float(alternative.base_reward) - float(mutated.base_reward)
            chosen = alternative
        else:
            self.stats["basis_forest_mutation_recombination_rejected"] += 1
            self.stats["basis_forest_mutation_recombination_original_kept"] += 1
            chosen = mutated

        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_forest_mutation_recombination_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "mutation_variation": original_variation,
                    "new_gene_count": int(new_gene_count),
                    "choice": details,
                    "mutated": self._candidate_summary(mutated),
                    "alternative": self._candidate_summary(alternative),
                    "accepted": accepted,
                },
            )
        return chosen

    def _basis_forest_mutation_shadow_candidate(
        self,
        *,
        parent: ControlledIndividual,
        original_tree: nd.Symbol,
        original_genome: BasisGenome,
        original_variation: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[ControlledIndividual, ShadowProposal | None]:
        """Keep the mutation child and offer its useful new term as shadow."""

        mutated = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent,),
            original_variation,
        )
        mutated.basis_genome = copy.deepcopy(original_genome)
        mutated.basis_forest_status = (
            "exact_terms" if original_genome.is_exact else "mixed_opaque"
        )
        self._evaluate_individual(mutated, X, y)
        self.stats["basis_forest_mutation_shadow_events"] += 1
        self.stats["basis_forest_shadow_events"] += 1

        parent_keys = set(self._basis_genome_of(parent).canonicals)
        new_gene_count = sum(
            canonical not in parent_keys for canonical in original_genome.canonicals
        )
        self.stats["basis_forest_mutation_shadow_new_genes"] += int(new_gene_count)
        self.stats["basis_forest_mutation_shadow_events_with_new_genes"] += int(
            new_gene_count > 0
        )

        proposal = self._build_basis_forest_residual_infusion_proposal(
            original=parent,
            donor=mutated,
            X=X,
            y=y,
        )
        if proposal is None:
            self.stats["basis_forest_mutation_shadow_no_proposal"] += 1
            self.stats["basis_forest_shadow_no_constructible_alternative"] += 1
            return mutated, None

        tree, genome, choice, details = proposal
        self.stats["basis_forest_mutation_shadow_proposals"] += 1
        self.stats[f"basis_forest_mutation_shadow_{choice.action}_proposals"] += 1
        alternative = self._new_individual(
            tree,
            generation,
            slot,
            (parent,),
            "basis_forest_mutation_shadow_alternative",
        )
        alternative.basis_genome = genome
        alternative.basis_forest_status = "exact_terms"
        self._evaluate_individual(alternative, X, y)
        self.stats["basis_forest_residual_alternatives_evaluated"] += 1
        self.stats["basis_forest_shadow_alternatives_evaluated"] += 1
        self.stats["basis_forest_mutation_shadow_alternatives_evaluated"] += 1
        self.stats["basis_forest_shadow_invalid_alternatives"] += int(
            not alternative.valid
        )
        if not self._verified_proposal_better(alternative, mutated):
            self.stats["basis_forest_mutation_shadow_base_rejected"] += 1
            self.stats["basis_forest_shadow_base_rejected"] += 1
            return mutated, None

        reward_delta = float(alternative.base_reward) - float(mutated.base_reward)
        self.stats["basis_forest_mutation_shadow_base_qualified"] += 1
        self.stats["basis_forest_shadow_base_qualified"] += 1
        self.stats["basis_forest_shadow_reward_gain_vs_original_sum"] += reward_delta
        alternative.variation_type = "basis_forest_mutation_shadow_candidate"
        shadow = ShadowProposal(
            candidate=alternative,
            source_slot=slot,
            original_candidate_id=mutated.candidate_id,
            coverage_gain=float(choice.donor_residual_gain),
            base_qualified=True,
        )
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_forest_mutation_shadow_candidates.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "mutation_variation": original_variation,
                    "new_gene_count": int(new_gene_count),
                    "choice": details,
                    "mutated": self._candidate_summary(mutated),
                    "candidate": self._candidate_summary(alternative),
                    "reward_gain_vs_mutated": reward_delta,
                    "coverage_gain": choice.donor_residual_gain,
                },
            )
        return mutated, shadow

    def _basis_forest_residual_shadow_candidate(
        self,
        *,
        donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_genome: BasisGenome,
        original_variation: str,
        parents: Sequence[ControlledIndividual],
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[ControlledIndividual, ShadowProposal | None]:
        """Keep the uniform child and offer one verified residual shadow."""

        original = self._new_individual(
            original_tree,
            generation,
            slot,
            parents,
            original_variation,
        )
        original.basis_genome = copy.deepcopy(original_genome)
        original.basis_forest_status = (
            "exact_terms" if original_genome.is_exact else "mixed_opaque"
        )
        self._evaluate_individual(original, X, y)
        self.stats["basis_forest_shadow_events"] += 1
        proposal = self._build_basis_forest_residual_infusion_proposal(
            original=original,
            donor=donor,
            X=X,
            y=y,
        )
        if proposal is None:
            self.stats["basis_forest_shadow_no_constructible_alternative"] += 1
            return original, None

        tree, genome, choice, details = proposal
        alternative = self._new_individual(
            tree,
            generation,
            slot,
            parents,
            "basis_forest_residual_shadow_alternative",
        )
        alternative.basis_genome = genome
        alternative.basis_forest_status = "exact_terms"
        self._evaluate_individual(alternative, X, y)
        self.stats["basis_forest_residual_alternatives_evaluated"] += 1
        self.stats["basis_forest_shadow_alternatives_evaluated"] += 1
        self.stats["basis_forest_shadow_invalid_alternatives"] += int(
            not alternative.valid
        )
        if not self._verified_proposal_better(alternative, original):
            self.stats["basis_forest_shadow_base_rejected"] += 1
            return original, None

        reward_delta = float(alternative.base_reward) - float(original.base_reward)
        self.stats["basis_forest_shadow_base_qualified"] += 1
        self.stats["basis_forest_shadow_reward_gain_vs_original_sum"] += reward_delta
        alternative.variation_type = "basis_forest_residual_shadow_candidate"
        shadow = ShadowProposal(
            candidate=alternative,
            source_slot=slot,
            original_candidate_id=original.candidate_id,
            coverage_gain=float(choice.donor_residual_gain),
            base_qualified=True,
        )
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_forest_residual_shadow_candidates.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "donor_id": donor.candidate_id,
                    "choice": details,
                    "original": self._candidate_summary(original),
                    "candidate": self._candidate_summary(alternative),
                    "reward_gain_vs_original": reward_delta,
                    "coverage_gain": choice.donor_residual_gain,
                },
            )
        return original, shadow

    def _apply_basis_forest_residual_shadow_slot(
        self,
        children: list[ControlledIndividual],
        proposals: Sequence[ShadowProposal],
        *,
        reproduction_candidate_ids: set[int],
    ) -> None:
        """Use at most one reproduction-copy slot for a verified shadow."""

        self.stats["basis_forest_shadow_generations"] += 1
        self.stats["basis_forest_shadow_pool_candidates"] += len(proposals)
        if not proposals or not reproduction_candidate_ids:
            self.stats["basis_forest_shadow_no_injection"] += 1
            return
        protected_ids = {
            candidate.candidate_id
            for candidate in children
            if candidate.candidate_id not in reproduction_candidate_ids
        }
        plan = plan_shadow_injection(
            children,
            proposals,
            candidate_id=lambda value: value.candidate_id,
            canonical_expression=lambda value: (
                value.canonical_fitted_expression or value.raw_expression
            ),
            base_key=self._base_key,
            protected_candidate_ids=protected_ids,
            minimum_coverage_gain=0.0,
        )
        if plan is None:
            self.stats["basis_forest_shadow_no_injection"] += 1
            return

        replacement = children[plan.replacement_index]
        alternative = plan.proposal.candidate
        proposal_source = alternative.variation_type
        if replacement.variation_type != "reproduction":
            raise RuntimeError("basis-forest shadow may replace only reproduction")
        if not self._verified_proposal_better(alternative, replacement):
            self.stats["basis_forest_shadow_replacement_base_rejected"] += 1
            self.stats["basis_forest_shadow_no_injection"] += 1
            return

        replacement_summary = self._candidate_summary(replacement)
        alternative.candidate_id = replacement.candidate_id
        alternative.slot = replacement.slot
        alternative.variation_type = "basis_forest_residual_shadow_injected"
        children[plan.replacement_index] = alternative
        self.stats["basis_forest_shadow_injections"] += 1
        self.stats[f"basis_forest_shadow_injected_source:{proposal_source}"] += 1
        self.stats[f"basis_forest_shadow_replacement:{plan.replacement_reason}"] += 1
        self.stats["basis_forest_shadow_replaced_variation:reproduction"] += 1
        self.stats["basis_forest_shadow_injected_sobolev_gain_sum"] += float(
            plan.proposal.coverage_gain
        )
        self.stats["basis_forest_shadow_reward_gain_vs_replaced_sum"] += float(
            alternative.base_reward
        ) - float(replacement.base_reward)
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_forest_residual_shadow_injections.jsonl",
                {
                    "generation": alternative.generation,
                    "source_slot": plan.proposal.source_slot,
                    "replacement_index": plan.replacement_index,
                    "replacement_reason": plan.replacement_reason,
                    "replacement": replacement_summary,
                    "injected": self._candidate_summary(alternative),
                    "proposal_source": proposal_source,
                    "sobolev_gain": plan.proposal.coverage_gain,
                },
            )

    def _evolve_basis_forest_generation(
        self,
        generation: int,
        population: list[ControlledIndividual],
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[ControlledIndividual]:
        children: list[ControlledIndividual] = []
        shadow_proposals: list[ShadowProposal] = []
        shadow_original_ids: set[int] = set()
        reproduction_candidate_ids: set[int] = set()
        elite_sources = self.current_elite_order[: self.elitism_k]
        if len(elite_sources) < self.elitism_k:
            raise RuntimeError("Previous generation did not yield enough elites")
        for slot, parent in enumerate(elite_sources):
            self._raise_if_deadline()
            child = self._new_individual(
                parent.eqtree.copy(),
                generation,
                slot,
                (parent,),
                "elite",
            )
            child.basis_genome = copy.deepcopy(self._basis_genome_of(parent))
            child.basis_forest_status = parent.basis_forest_status
            self._evaluate_individual(child, X, y)
            children.append(child)
            self.operator_counts["elite"] += 1

        cumulative = np.cumsum(list(self.method_probs.values()))
        methods = tuple(self.method_probs)
        for slot in range(self.elitism_k, self.population_size):
            self._raise_if_deadline()
            parent = self._tournament_one(
                population,
                self.counter_rng.generator(generation, slot, "parent_tournament"),
            )
            choice_rng = self.counter_rng.generator(generation, slot, "operator_choice")
            method = methods[
                min(
                    int(np.searchsorted(cumulative, choice_rng.random(), side="right")),
                    len(methods) - 1,
                )
            ]
            self.stats[f"variation_decision:{method}"] += 1
            donor = None
            if method == "crossover":
                donor = self._tournament_one(
                    population,
                    self.counter_rng.generator(generation, slot, "donor_tournament"),
                )
            parents = (parent,) if donor is None else (parent, donor)
            child: ControlledIndividual | None = None
            try:
                rng_event = {
                    "crossover": "crossover",
                    "subtree-mutation": "subtree_mutation",
                    "hoist-mutation": "hoist_mutation",
                    "point-mutation": "point_mutation",
                    "reproduction": "reproduction",
                }[method]
                rng = self.counter_rng.generator(generation, slot, rng_event)
                if method == "crossover":
                    assert donor is not None
                    tree, genome, variation = self._basis_forest_crossover(
                        parent,
                        donor,
                        rng,
                    )
                    if (
                        self.use_basis_forest_sn_crossover
                        and self._basis_forest_sn_crossover_gate(generation, slot)
                    ):
                        child = self._verified_basis_forest_sn_crossover(
                            parent=parent,
                            donor=donor,
                            original_tree=tree,
                            original_genome=genome,
                            original_variation=variation,
                            generation=generation,
                            slot=slot,
                            X=X,
                            y=y,
                        )
                        variation = child.variation_type
                    elif (
                        self.use_basis_forest_sn_residual_infusion
                        and self._basis_forest_sn_crossover_gate(generation, slot)
                    ):
                        child = self._verified_basis_forest_residual_infusion(
                            donor=donor,
                            original_tree=tree,
                            original_genome=genome,
                            original_variation=variation,
                            parents=parents,
                            generation=generation,
                            slot=slot,
                            X=X,
                            y=y,
                        )
                        variation = child.variation_type
                    elif (
                        self.use_basis_forest_sn_residual_shadow
                        and self._basis_forest_sn_crossover_gate(generation, slot)
                    ):
                        child, shadow = self._basis_forest_residual_shadow_candidate(
                            donor=donor,
                            original_tree=tree,
                            original_genome=genome,
                            original_variation=variation,
                            parents=parents,
                            generation=generation,
                            slot=slot,
                            X=X,
                            y=y,
                        )
                        variation = child.variation_type
                        shadow_original_ids.add(child.candidate_id)
                        if shadow is not None:
                            shadow_proposals.append(shadow)
                elif method in {
                    "subtree-mutation",
                    "hoist-mutation",
                    "point-mutation",
                }:
                    tree, genome, variation = self._basis_forest_mutation(
                        method,
                        parent,
                        rng,
                        X=X,
                        generation=generation,
                        slot=slot,
                    )
                    if self.use_basis_forest_sn_mutation_recombination:
                        child = self._verified_basis_forest_mutation_recombination(
                            parent=parent,
                            original_tree=tree,
                            original_genome=genome,
                            original_variation=variation,
                            generation=generation,
                            slot=slot,
                            X=X,
                            y=y,
                        )
                        variation = child.variation_type
                    elif self.use_basis_forest_sn_mutation_shadow:
                        child, shadow = self._basis_forest_mutation_shadow_candidate(
                            parent=parent,
                            original_tree=tree,
                            original_genome=genome,
                            original_variation=variation,
                            generation=generation,
                            slot=slot,
                            X=X,
                            y=y,
                        )
                        variation = child.variation_type
                        shadow_original_ids.add(child.candidate_id)
                        if shadow is not None:
                            shadow_proposals.append(shadow)
                else:
                    tree = parent.eqtree.copy()
                    genome = copy.deepcopy(self._basis_genome_of(parent))
                    variation = "reproduction"
            except Exception as error:
                tree = parent.eqtree.copy()
                genome = copy.deepcopy(self._basis_genome_of(parent))
                variation = f"basis-{method}_error_fallback"
                self.stats["basis_forest_operator_fallbacks"] += 1
                self.failure_counts[
                    f"basis_forest_operator:{type(error).__name__}"
                ] += 1
            if child is None:
                child = self._new_individual(
                    tree,
                    generation,
                    slot,
                    parents,
                    variation,
                )
                child.basis_genome = genome
                child.basis_forest_status = (
                    "exact_terms" if genome.is_exact else "mixed_opaque"
                )
                self._evaluate_individual(child, X, y)
            children.append(child)
            self.operator_counts[variation] += 1
            if variation == "reproduction":
                reproduction_candidate_ids.add(child.candidate_id)
        if (
            self.use_basis_forest_sn_residual_shadow
            or self.use_basis_forest_sn_mutation_shadow
        ):
            self.stats["basis_forest_shadow_gated_originals"] += len(
                shadow_original_ids
            )
            self._apply_basis_forest_residual_shadow_slot(
                children,
                shadow_proposals,
                reproduction_candidate_ids=reproduction_candidate_ids,
            )
            child_ids = {candidate.candidate_id for candidate in children}
            self.stats["basis_forest_shadow_gated_originals_retained"] += sum(
                candidate_id in child_ids for candidate_id in shadow_original_ids
            )
        self._apply_structural_selection(
            children,
            X,
            y,
            enforce_deadline=True,
        )
        self._raise_if_deadline()
        return children

    def _build_repair_shadow_proposals(
        self,
        children: Sequence[ControlledIndividual],
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[ShadowProposal]:
        """Offer Base-qualified low-redundancy repairs without deleting sources."""

        self.stats["sn_repair_shadow_generations"] += 1
        proposals: list[ShadowProposal] = []
        seen: set[str] = set()
        for candidate in self._rank_base(children):
            if not candidate.valid:
                continue
            identity = candidate.canonical_fitted_expression
            if identity in seen:
                continue
            seen.add(identity)
            if len(seen) > self.sn_repair_shortlist_size:
                break
            fit = candidate.fit_result
            if fit is None or len(fit.terms) <= 1:
                self.stats["sn_repair_shadow_not_applicable"] += 1
                continue
            self.stats["sn_repair_shadow_candidates_considered"] += 1
            analysis = self._evaluate_sobolev(candidate, X)
            candidate.sobolev_result = analysis
            candidate.sobolev_success = analysis.success
            if not analysis.success:
                self.stats["sn_repair_shadow_evaluator_failures"] += 1
                continue
            before_penalty = sobolev_penalty(
                analysis.term_novelties,
                self.sobolev_tau,
            )
            candidate.sobolev_penalty = before_penalty
            excluded = tuple(
                index
                for index, term in enumerate(fit.terms)
                if not term.basis.free_symbols
            )
            repaired = self._prune_candidate(
                candidate,
                X,
                y,
                excluded_prune_indices=excluded,
            )
            if repaired is candidate:
                self.stats["sn_repair_shadow_no_repair"] += 1
                continue
            self.stats["sn_repair_shadow_repairs_evaluated"] += 1
            if not self._verified_proposal_better(repaired, candidate):
                self.stats["sn_repair_shadow_base_rejected"] += 1
                continue
            after_penalty = float(repaired.sobolev_penalty or 0.0)
            redundancy_gain = before_penalty - after_penalty
            if not np.isfinite(redundancy_gain) or redundancy_gain <= 0.0:
                self.stats["sn_repair_shadow_no_redundancy_gain"] += 1
                continue
            repaired.variation_type = "sn_repair_shadow_candidate"
            proposals.append(
                ShadowProposal(
                    candidate=repaired,
                    source_slot=int(candidate.slot),
                    original_candidate_id=int(candidate.candidate_id),
                    coverage_gain=float(redundancy_gain),
                    base_qualified=True,
                )
            )
            self.stats["sn_repair_shadow_base_qualified"] += 1
            self.stats["sn_repair_shadow_redundancy_gain_sum"] += float(redundancy_gain)
            if len(proposals) >= self.sn_repair_max_accepted_per_generation:
                break
        self.stats["sn_repair_shadow_pool_candidates"] += len(proposals)
        return proposals

    def _apply_repair_shadow_slot(
        self,
        children: list[ControlledIndividual],
        proposals: Sequence[ShadowProposal],
        *,
        reproduction_candidate_ids: set[int],
    ) -> None:
        """Use one reproduction-copy slot while retaining every repair source."""

        if not proposals or not reproduction_candidate_ids:
            self.stats["sn_repair_shadow_no_injection"] += 1
            return
        source_ids = {int(proposal.original_candidate_id) for proposal in proposals}
        protected_ids = {
            candidate.candidate_id
            for candidate in children
            if candidate.candidate_id not in reproduction_candidate_ids
        }
        protected_ids.update(source_ids)
        plan = plan_shadow_injection(
            children,
            proposals,
            candidate_id=lambda value: value.candidate_id,
            canonical_expression=lambda value: (
                value.canonical_fitted_expression or value.raw_expression
            ),
            base_key=self._base_key,
            protected_candidate_ids=protected_ids,
            minimum_coverage_gain=0.0,
        )
        if plan is None:
            self.stats["sn_repair_shadow_no_injection"] += 1
            return
        replacement = children[plan.replacement_index]
        alternative = plan.proposal.candidate
        if replacement.variation_type != "reproduction":
            raise RuntimeError("repair shadow may replace only reproduction")
        if not self._verified_proposal_better(alternative, replacement):
            self.stats["sn_repair_shadow_replacement_base_rejected"] += 1
            self.stats["sn_repair_shadow_no_injection"] += 1
            return
        replacement_summary = self._candidate_summary(replacement)
        alternative.candidate_id = replacement.candidate_id
        alternative.slot = replacement.slot
        alternative.variation_type = "sn_repair_shadow_injected"
        children[plan.replacement_index] = alternative
        self.stats["sn_repair_shadow_injections"] += 1
        self.stats[f"sn_repair_shadow_replacement:{plan.replacement_reason}"] += 1
        self.stats["sn_repair_shadow_replaced_variation:reproduction"] += 1
        self.stats["sn_repair_shadow_reward_gain_vs_replaced_sum"] += float(
            alternative.base_reward
        ) - float(replacement.base_reward)
        child_ids = {candidate.candidate_id for candidate in children}
        self.stats["sn_repair_shadow_sources"] += len(source_ids)
        self.stats["sn_repair_shadow_sources_retained"] += sum(
            source_id in child_ids for source_id in source_ids
        )
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "repair_shadow_injections.jsonl",
                {
                    "generation": alternative.generation,
                    "source_candidate_id": plan.proposal.original_candidate_id,
                    "source_slot": plan.proposal.source_slot,
                    "replacement_index": plan.replacement_index,
                    "replacement_reason": plan.replacement_reason,
                    "replacement": replacement_summary,
                    "injected": self._candidate_summary(alternative),
                    "redundancy_gain": plan.proposal.coverage_gain,
                },
            )

    def _evolve_generation(
        self,
        generation: int,
        population: list[ControlledIndividual],
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[ControlledIndividual]:
        if self.use_basis_forest:
            return self._evolve_basis_forest_generation(
                generation,
                population,
                X,
                y,
            )
        children: list[ControlledIndividual] = []
        shadow_proposals: list[ShadowProposal] = []
        shadow_original_ids: set[int] = set()
        reproduction_candidate_ids: set[int] = set()
        elite_sources = self.current_elite_order[: self.elitism_k]
        if len(elite_sources) < self.elitism_k:
            raise RuntimeError("Previous generation did not yield enough elites")
        for slot, parent in enumerate(elite_sources):
            self._raise_if_deadline()
            child = self._new_individual(
                parent.eqtree.copy(),
                generation,
                slot,
                (parent,),
                "elite",
            )
            self._evaluate_individual(child, X, y)
            children.append(child)
            self.operator_counts["elite"] += 1
            self._raise_if_deadline()
        cumulative = np.cumsum(list(self.method_probs.values()))
        methods = tuple(self.method_probs)
        for slot in range(self.elitism_k, self.population_size):
            self._raise_if_deadline()
            parent = self._tournament_one(
                population,
                self.counter_rng.generator(generation, slot, "parent_tournament"),
            )
            choice_rng = self.counter_rng.generator(generation, slot, "operator_choice")
            method = methods[
                min(
                    int(np.searchsorted(cumulative, choice_rng.random(), side="right")),
                    len(methods) - 1,
                )
            ]
            reproduction_method = method
            self.stats[f"variation_decision:{method}"] += 1
            donor = None
            if method == "crossover":
                donor = self._tournament_one(
                    population,
                    self.counter_rng.generator(generation, slot, "donor_tournament"),
                )
            parents = (parent,) if donor is None else (parent, donor)
            mutation_selection: MutationSiteSelection | None = None
            operator_failed = False
            try:
                rng_event = {
                    "crossover": "crossover",
                    "subtree-mutation": "subtree_mutation",
                    "hoist-mutation": "hoist_mutation",
                    "point-mutation": "point_mutation",
                    "reproduction": "reproduction",
                }[method]
                rng = self.counter_rng.generator(generation, slot, rng_event)
                if method == "crossover":
                    proposal = super().crossover(parent, donor, rng=rng)
                elif method == "subtree-mutation":
                    if self.use_sn_gp_plugin:
                        proposal, mutation_selection = self._sn_v2_mutation(
                            method,
                            parent,
                            rng,
                            generation,
                            slot,
                        )
                    else:
                        proposal = super().subtree_mutation(parent, rng=rng)
                elif method == "hoist-mutation":
                    if self.use_sn_gp_plugin:
                        proposal, mutation_selection = self._sn_v2_mutation(
                            method,
                            parent,
                            rng,
                            generation,
                            slot,
                        )
                    else:
                        proposal = super().hoist_mutation(parent, rng=rng)
                elif method == "point-mutation":
                    if self.use_sn_gp_plugin:
                        proposal, mutation_selection = self._sn_v2_mutation(
                            method,
                            parent,
                            rng,
                            generation,
                            slot,
                        )
                    else:
                        proposal = super().point_mutation(parent, rng=rng)
                elif method == "reproduction":
                    proposal = parent.copy()
                else:
                    raise ValueError(f"Unknown GP variation method {method}")
                tree = proposal.eqtree
                if len(tree) > self.max_len:
                    self.stats["offspring_bloat_fallbacks"] += 1
                    tree = parent.eqtree.copy()
                    method = f"{method}_bloat_fallback"
            except Exception as error:
                operator_failed = True
                self.stats["offspring_operator_fallbacks"] += 1
                self.failure_counts[f"operator:{type(error).__name__}"] += 1
                tree = parent.eqtree.copy()
                method = f"{method}_error_fallback"
            child: ControlledIndividual | None = None
            if (
                not operator_failed
                and reproduction_method == "crossover"
                and self._coverage_crossover_gate(generation, slot)
            ):
                assert donor is not None
                try:
                    helper = None
                    if self.use_sn_screened_pursuit_shadow:
                        child, shadow = self._screened_basis_pursuit_shadow_crossover(
                            parent=parent,
                            original_donor=donor,
                            original_tree=tree,
                            original_method=method,
                            generation=generation,
                            slot=slot,
                            X=X,
                            y=y,
                        )
                        shadow_original_ids.add(child.candidate_id)
                        if shadow is not None:
                            shadow_proposals.append(shadow)
                    elif self.use_sn_shadow_slot:
                        child, shadow = self._dual_path_orthogonal_shadow_crossover(
                            parent=parent,
                            original_donor=donor,
                            original_tree=tree,
                            original_method=method,
                            generation=generation,
                            slot=slot,
                            X=X,
                            y=y,
                        )
                        shadow_original_ids.add(child.candidate_id)
                        if shadow is not None:
                            shadow_proposals.append(shadow)
                    elif self.use_sn_orthogonal_basis_crossover:
                        helper = self._verified_orthogonal_basis_crossover
                    elif self.use_sn_conditional_basis_replacement:
                        helper = self._verified_conditional_basis_replacement_crossover
                    elif self.use_sn_basis_replacement:
                        helper = (
                            self._verified_bidirectional_basis_replacement_crossover
                        )
                    elif self.use_sn_basis_pursuit:
                        helper = self._verified_screened_basis_pursuit_crossover
                    elif self.use_sn_basis_archive:
                        helper = self._verified_archive_basis_crossover
                    elif self.use_sn_residual_basis_infusion:
                        helper = self._verified_residual_basis_infusion_crossover
                    elif self.use_sn_basis_infusion:
                        helper = self._verified_basis_infusion_crossover
                    elif self.use_sn_basis_exchange:
                        helper = self._verified_basis_exchange_crossover
                    else:
                        helper = self._verified_coverage_crossover
                    if helper is not None:
                        child = helper(
                            parent=parent,
                            original_donor=donor,
                            original_tree=tree,
                            original_method=method,
                            generation=generation,
                            slot=slot,
                            X=X,
                            y=y,
                        )
                    method = child.variation_type
                except Exception as error:
                    self.stats["sn_coverage_crossover_errors"] += 1
                    self.failure_counts[
                        f"coverage_crossover:{type(error).__name__}"
                    ] += 1
            if child is None:
                child = self._new_individual(tree, generation, slot, parents, method)
                child.sn_targeted_mutation = bool(
                    mutation_selection is not None and mutation_selection.targeted
                )
                self._evaluate_individual(child, X, y)
            children.append(child)
            self.operator_counts[method] += 1
            if method == "reproduction":
                reproduction_candidate_ids.add(child.candidate_id)
            if mutation_selection is not None:
                self._queue_mutation_event(
                    method,
                    parent,
                    child,
                    mutation_selection,
                )
            self._raise_if_deadline()
        if self.use_sn_shadow_slot:
            protected_ids = {
                candidate.candidate_id for candidate in children[: self.elitism_k]
            }
            protected_ids.update(shadow_original_ids)
            protected_ids.update(
                int(event["offspring_id"])
                for event in self._pending_mutation_events
                if int(event["generation"]) == generation
            )
            self.stats["sn_shadow_gated_originals"] += len(shadow_original_ids)
            if self.use_sn_screened_pursuit_shadow:
                self._apply_screened_pursuit_shadow_slot(
                    children,
                    shadow_proposals,
                    protected_candidate_ids=protected_ids,
                )
            else:
                self._apply_shadow_slot(
                    children,
                    shadow_proposals,
                    protected_candidate_ids=protected_ids,
                )
            child_ids = {candidate.candidate_id for candidate in children}
            self.stats["sn_shadow_gated_originals_retained"] += sum(
                candidate_id in child_ids for candidate_id in shadow_original_ids
            )
        if self.use_sn_repair_shadow:
            repair_proposals = self._build_repair_shadow_proposals(
                children,
                X,
                y,
            )
            self._apply_repair_shadow_slot(
                children,
                repair_proposals,
                reproduction_candidate_ids=reproduction_candidate_ids,
            )
        self._apply_structural_selection(children, X, y, enforce_deadline=True)
        if self.use_sn_gp_plugin:
            self._finalize_mutation_events(children)
        self._raise_if_deadline()
        return children

    def _new_individual(
        self,
        tree: nd.Symbol,
        generation: int,
        slot: int,
        parents: Sequence[ControlledIndividual],
        variation_type: str,
    ) -> ControlledIndividual:
        value = ControlledIndividual(
            tree,
            candidate_id=self._proposal_id(generation, slot),
            generation=generation,
            slot=slot,
            parent_ids=tuple(parent.candidate_id for parent in parents),
            variation_type=variation_type,
        )
        value.parent_geometry_hints = tuple(
            (
                parent.candidate_id,
                (
                    parent.sobolev_result.candidate_geometry_key
                    if parent.sobolev_result is not None
                    else None
                ),
                parent.term_keys,
            )
            for parent in parents
        )
        return value

    def _record_pursuit_base_progress(self, generation: int) -> None:
        if not self.use_sn_stagnation_pursuit_shadow:
            return
        candidate = self.base_quality_best_ever
        if candidate is None or not np.isfinite(candidate.base_reward):
            return
        reward = float(candidate.base_reward)
        reference = self.sn_pursuit_gate_best_reward
        if reference is None:
            self.sn_pursuit_gate_best_reward = reward
            self.sn_pursuit_last_improvement_generation = int(generation)
            self.stats["sn_pursuit_stagnation_initializations"] += 1
            return
        epsilon = self.sn_pursuit_stagnation_epsilon_abs + (
            self.sn_pursuit_stagnation_epsilon_rel * max(abs(reward), abs(reference))
        )
        if reward - reference > epsilon:
            self.sn_pursuit_gate_best_reward = reward
            self.sn_pursuit_last_improvement_generation = int(generation)
            self.stats["sn_pursuit_stagnation_meaningful_improvements"] += 1

    def _pursuit_stagnation_gate_active(self, generation: int) -> bool:
        if not self.use_sn_stagnation_pursuit_shadow:
            return True
        completed_since_improvement = max(
            0,
            int(generation) - 1 - self.sn_pursuit_last_improvement_generation,
        )
        return completed_since_improvement >= self.sn_pursuit_stagnation_patience

    def _coverage_crossover_gate(self, generation: int, slot: int) -> bool:
        if not self.use_sn_verified_coverage_crossover:
            return False
        self.stats["sn_coverage_crossover_events"] += 1
        if self.use_sn_basis_archive:
            self.stats["sn_basis_archive_gate_events"] += 1
            if self.use_sn_stagnation_pursuit_shadow:
                self.stats["sn_pursuit_stagnation_gate_checks"] += 1
                if not self._pursuit_stagnation_gate_active(generation):
                    self.stats["sn_pursuit_stagnation_gate_blocked"] += 1
                    return False
                self.stats["sn_pursuit_stagnation_gate_eligible"] += 1
            if (
                self.sn_coverage_crossover_rate <= 0.0
                or self.basis_archive is None
                or not self.basis_archive.entries
            ):
                self.stats["sn_basis_archive_gate_unavailable"] += 1
                return False
            draw = self.counter_rng.generator(
                generation,
                slot,
                "crossover",
                attempt=1,
            ).random()
            self.stats["sn_coverage_crossover_gate_trials"] += 1
            self.stats["sn_basis_archive_gate_trials"] += 1
            selected = bool(draw < self.sn_coverage_crossover_rate)
            self.stats["sn_coverage_crossover_gate_selected"] += int(selected)
            self.stats["sn_basis_archive_gate_selected"] += int(selected)
            return selected
        if (
            self.sn_coverage_crossover_rate <= 0.0
            or not self.current_coverage_slot_changed
            or self.current_coverage_candidate is None
            or self.current_coverage_gain is None
        ):
            self.stats["sn_coverage_crossover_unavailable"] += 1
            return False
        draw = self.counter_rng.generator(
            generation,
            slot,
            "crossover",
            attempt=1,
        ).random()
        self.stats["sn_coverage_crossover_gate_trials"] += 1
        selected = bool(draw < self.sn_coverage_crossover_rate)
        self.stats["sn_coverage_crossover_gate_selected"] += int(selected)
        return selected

    def _verified_residual_basis_infusion_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Base-verify one target-residual × Sobolev basis proposal."""

        coverage_donor = self.current_coverage_candidate
        assert coverage_donor is not None
        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_residual_infusion_events"] += 1
        proposal = self._build_residual_basis_infusion_proposal(
            parent=parent,
            donor=coverage_donor,
            X=X,
            y=y,
        )
        if proposal is None:
            self.stats["sn_residual_infusion_original_kept"] += 1
            return original
        alternative_tree, choice, infusion_details = proposal
        alternative = self._new_individual(
            alternative_tree,
            generation,
            slot,
            (parent, coverage_donor),
            f"residual_basis_{choice.action}_alternative",
        )
        self._evaluate_individual(alternative, X, y)
        self.stats["sn_residual_infusion_alternatives_evaluated"] += 1
        accepted = self._verified_proposal_better(alternative, original)
        reward_delta = (
            float(alternative.base_reward) - float(original.base_reward)
            if np.isfinite(alternative.base_reward)
            and np.isfinite(original.base_reward)
            else None
        )
        if accepted:
            alternative.variation_type = f"residual_basis_{choice.action}_verified"
            self.stats["sn_residual_infusion_accepted"] += 1
            self.stats[f"sn_residual_infusion_{choice.action}_accepted"] += 1
            if reward_delta is not None:
                self.stats["sn_residual_infusion_reward_gain_sum"] += reward_delta
            if np.isfinite(alternative.complexity) and np.isfinite(original.complexity):
                self.stats["sn_residual_infusion_complexity_saved_sum"] += float(
                    original.complexity
                ) - float(alternative.complexity)
            chosen = alternative
        else:
            self.stats["sn_residual_infusion_rejected"] += 1
            self.stats["sn_residual_infusion_original_kept"] += 1
            chosen = original
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "residual_basis_infusion_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "original_donor_id": original_donor.candidate_id,
                    "coverage_donor_id": coverage_donor.candidate_id,
                    "coverage_gain": self.current_coverage_gain,
                    "choice": infusion_details,
                    "original": self._candidate_summary(original),
                    "alternative": self._candidate_summary(alternative),
                    "alternative_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                },
            )
        return chosen

    def _verified_orthogonal_basis_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Base-verify prefixes from a two-parent Sobolev basis pursuit."""

        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_orthogonal_crossover_events"] += 1
        proposals = self._build_orthogonal_basis_crossover_proposals(
            parent=parent,
            donor=original_donor,
            X=X,
            y=y,
        )
        if not proposals:
            self.stats["sn_orthogonal_crossover_original_kept"] += 1
            return original
        self.stats["sn_orthogonal_crossover_evaluated_events"] += 1

        evaluated: list[tuple[ControlledIndividual, dict[str, Any]]] = []
        for tree, details in proposals:
            alternative = self._new_individual(
                tree,
                generation,
                slot,
                (parent, original_donor),
                "orthogonal_basis_crossover_alternative",
            )
            self._evaluate_individual(alternative, X, y)
            self.stats["sn_orthogonal_crossover_alternatives_evaluated"] += 1
            self.stats["sn_orthogonal_crossover_invalid_alternatives"] += int(
                not alternative.valid
            )
            evaluated.append((alternative, details))

        best, best_details = min(evaluated, key=lambda item: self._base_key(item[0]))
        self.stats["sn_orthogonal_crossover_base_best_depth_sum"] += int(
            best_details["path_depth"]
        )
        self.stats["sn_orthogonal_crossover_base_best_not_final_prefix"] += int(
            int(best_details["path_depth"]) != int(best_details["path_length"])
        )
        self.stats["sn_orthogonal_crossover_base_best_cross_parent"] += int(
            bool(best_details["uses_receiver_and_donor"])
        )

        accepted = self._verified_proposal_better(best, original)
        reward_delta = (
            float(best.base_reward) - float(original.base_reward)
            if np.isfinite(best.base_reward) and np.isfinite(original.base_reward)
            else None
        )
        if accepted:
            best.variation_type = "orthogonal_basis_crossover_verified"
            self.stats["sn_orthogonal_crossover_accepted"] += 1
            self.stats["sn_orthogonal_crossover_accepted_cross_parent"] += int(
                bool(best_details["uses_receiver_and_donor"])
            )
            if reward_delta is not None:
                self.stats["sn_orthogonal_crossover_reward_gain_sum"] += reward_delta
            if np.isfinite(best.complexity) and np.isfinite(original.complexity):
                self.stats["sn_orthogonal_crossover_complexity_saved_sum"] += float(
                    original.complexity
                ) - float(best.complexity)
            chosen = best
        else:
            self.stats["sn_orthogonal_crossover_rejected"] += 1
            self.stats["sn_orthogonal_crossover_original_kept"] += 1
            chosen = original

        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "orthogonal_basis_crossover_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "donor_id": original_donor.candidate_id,
                    "original": self._candidate_summary(original),
                    "alternatives": [
                        {
                            "pursuit": details,
                            "candidate": self._candidate_summary(alternative),
                        }
                        for alternative, details in evaluated
                    ],
                    "base_best": best_details,
                    "base_best_candidate": self._candidate_summary(best),
                    "base_best_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                },
            )
        return chosen

    @staticmethod
    def _shadow_r2_not_worse(
        alternative: ControlledIndividual,
        reference: ControlledIndividual,
    ) -> bool:
        alternative_fit = alternative.fit_result
        reference_fit = reference.fit_result
        return bool(
            alternative_fit is not None
            and reference_fit is not None
            and alternative_fit.success
            and reference_fit.success
            and np.isfinite(alternative_fit.r2)
            and np.isfinite(reference_fit.r2)
            and float(alternative_fit.r2) >= float(reference_fit.r2)
        )

    def _screened_basis_pursuit_shadow_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[ControlledIndividual, ShadowProposal | None]:
        """Keep the original child and offer one screened archive alternative."""

        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_pursuit_shadow_events"] += 1
        proposals = self._build_screened_basis_pursuit_proposals(
            parent=parent,
            X=X,
            y=y,
            generation=generation,
        )
        self.stats["sn_pursuit_shadow_source_proposals"] += len(proposals)
        if not proposals:
            self.stats["sn_pursuit_shadow_no_constructible_alternative"] += 1
            return original, None

        evaluated: list[
            tuple[
                ControlledIndividual,
                ScreenedArchiveDonor,
                str,
                dict[str, Any],
            ]
        ] = []
        for tree, donor, action, details in proposals:
            alternative = self._new_individual(
                tree,
                generation,
                slot,
                (parent,),
                f"screened_basis_{action}_shadow_alternative",
            )
            self._evaluate_individual(alternative, X, y)
            self.stats["sn_pursuit_shadow_alternatives_evaluated"] += 1
            self.stats["sn_pursuit_shadow_invalid_alternatives"] += int(
                not alternative.valid
            )
            evaluated.append((alternative, donor, action, details))

        selection_pool = evaluated
        if self.use_sn_pareto_pursuit_shadow:
            selection_pool = [
                item
                for item in evaluated
                if self._shadow_r2_not_worse(item[0], original)
            ]
            self.stats["sn_pursuit_shadow_r2_eligible_alternatives"] += len(
                selection_pool
            )
            if not selection_pool:
                self.stats["sn_pursuit_shadow_r2_rejected_events"] += 1
                return original, None
        if self.sn_pursuit_select_by_coverage:
            qualified = [
                item
                for item in selection_pool
                if self._verified_proposal_better(item[0], original)
            ]
            self.stats["sn_pursuit_shadow_base_qualified_alternatives"] += len(
                qualified
            )
            if not qualified:
                self.stats["sn_pursuit_shadow_base_rejected"] += 1
                return original, None
            basis = self.current_coverage_anchor_basis
            if basis is None:
                self.stats["sn_pursuit_shadow_geometry_fallbacks"] += 1
                self.stats[
                    "sn_pursuit_shadow_geometry_fallback:anchor_unavailable"
                ] += 1
                return original, None
            scored: list[
                tuple[
                    float,
                    tuple[
                        ControlledIndividual,
                        ScreenedArchiveDonor,
                        str,
                        dict[str, Any],
                    ],
                ]
            ] = []
            for item in qualified:
                vector = self._coverage_vector(item[0], X, enforce_deadline=True)
                if vector is None:
                    self.stats[
                        "sn_pursuit_shadow_coverage_candidate_geometry_failures"
                    ] += 1
                    continue
                try:
                    gain = signature_residual_gain(basis, vector)
                except (ValueError, np.linalg.LinAlgError) as error:
                    self.stats[
                        "sn_pursuit_shadow_coverage_candidate_geometry_failures"
                    ] += 1
                    self.failure_counts[
                        f"pursuit_shadow_coverage:{type(error).__name__}"
                    ] += 1
                    continue
                scored.append((float(gain), item))
            self.stats["sn_pursuit_shadow_coverage_candidates_scored"] += len(scored)
            if not scored:
                self.stats["sn_pursuit_shadow_geometry_fallbacks"] += 1
                self.stats[
                    "sn_pursuit_shadow_geometry_fallback:no_scored_candidate"
                ] += 1
                return original, None
            coverage_gain, selected = min(
                scored,
                key=lambda value: (
                    -value[0],
                    self._base_key(value[1][0]),
                    value[1][0].canonical_fitted_expression,
                    int(value[1][3]["screen_rank"]),
                ),
            )
            best, best_donor, best_action, best_details = selected
            self.stats["sn_pursuit_shadow_coverage_selected_screen_rank_sum"] += int(
                best_details["screen_rank"]
            )
            self.stats["sn_pursuit_shadow_coverage_selected_not_screen_top1"] += int(
                int(best_details["screen_rank"]) != 1
            )
        else:
            best, best_donor, best_action, best_details = min(
                selection_pool, key=lambda item: self._base_key(item[0])
            )
            self.stats["sn_pursuit_shadow_base_best_screen_rank_sum"] += int(
                best_details["screen_rank"]
            )
            self.stats["sn_pursuit_shadow_base_best_not_screen_top1"] += int(
                int(best_details["screen_rank"]) != 1
            )
            if not self._verified_proposal_better(best, original):
                self.stats["sn_pursuit_shadow_base_rejected"] += 1
                return original, None
            basis = self.current_coverage_anchor_basis
            if basis is None:
                self.stats["sn_pursuit_shadow_geometry_fallbacks"] += 1
                self.stats[
                    "sn_pursuit_shadow_geometry_fallback:anchor_unavailable"
                ] += 1
                return original, None
            vector = self._coverage_vector(best, X, enforce_deadline=True)
            if vector is None:
                self.stats["sn_pursuit_shadow_geometry_fallbacks"] += 1
                self.stats[
                    "sn_pursuit_shadow_geometry_fallback:alternative_unavailable"
                ] += 1
                return original, None
            try:
                coverage_gain = signature_residual_gain(basis, vector)
            except (ValueError, np.linalg.LinAlgError) as error:
                self.stats["sn_pursuit_shadow_geometry_fallbacks"] += 1
                self.stats[
                    f"sn_pursuit_shadow_geometry_fallback:{type(error).__name__}"
                ] += 1
                return original, None

        reward_delta = (
            float(best.base_reward) - float(original.base_reward)
            if np.isfinite(best.base_reward) and np.isfinite(original.base_reward)
            else None
        )
        self.stats["sn_pursuit_shadow_base_qualified"] += 1
        self.stats[f"sn_pursuit_shadow_{best_action}_qualified"] += 1
        self.stats["sn_pursuit_shadow_donor_age_qualified_sum"] += best_donor.donor_age
        if reward_delta is not None:
            self.stats["sn_pursuit_shadow_reward_gain_vs_original_sum"] += reward_delta

        self.stats["sn_pursuit_shadow_geometry_success"] += 1
        self.stats["sn_pursuit_shadow_candidate_coverage_gain_sum"] += coverage_gain
        best.variation_type = (
            "pareto_pursuit_shadow_candidate"
            if self.use_sn_pareto_pursuit_shadow
            else "screened_basis_pursuit_shadow_candidate"
        )
        proposal = ShadowProposal(
            candidate=best,
            source_slot=slot,
            original_candidate_id=original.candidate_id,
            coverage_gain=float(coverage_gain),
            base_qualified=True,
        )
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "screened_pursuit_shadow_candidates.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "donor_id": original_donor.candidate_id,
                    "original": self._candidate_summary(original),
                    "candidate": self._candidate_summary(best),
                    "screen": best_details,
                    "reward_gain_vs_original": reward_delta,
                    "coverage_gain": coverage_gain,
                },
            )
        return original, proposal

    def _apply_screened_pursuit_shadow_slot(
        self,
        children: list[ControlledIndividual],
        proposals: Sequence[ShadowProposal],
        *,
        protected_candidate_ids: set[int],
    ) -> None:
        """Use one worse reproduction copy without deleting any varied child."""

        self.stats["sn_pursuit_shadow_generations"] += 1
        self.stats["sn_pursuit_shadow_pool_candidates"] += len(proposals)
        if not proposals:
            self.stats["sn_pursuit_shadow_no_injection"] += 1
            return
        protected = set(protected_candidate_ids)
        protected.update(
            child.candidate_id
            for child in children
            if child.variation_type != "reproduction"
        )
        plan = plan_shadow_injection(
            children,
            proposals,
            candidate_id=lambda value: value.candidate_id,
            canonical_expression=lambda value: (
                value.canonical_fitted_expression or value.raw_expression
            ),
            base_key=self._base_key,
            protected_candidate_ids=protected,
            minimum_coverage_gain=self.sn_shadow_minimum_coverage_gain,
            allow_represented_amplification=(
                self.sn_shadow_allow_represented_amplification
            ),
        )
        if plan is None:
            self.stats["sn_pursuit_shadow_no_injection"] += 1
            return

        replacement = children[plan.replacement_index]
        if replacement.variation_type != "reproduction":
            raise RuntimeError("screened pursuit shadow may replace only reproduction")
        alternative = plan.proposal.candidate
        if not self._verified_proposal_better(alternative, replacement):
            self.stats["sn_pursuit_shadow_replacement_base_rejected"] += 1
            self.stats["sn_pursuit_shadow_no_injection"] += 1
            return
        if self.use_sn_pareto_pursuit_shadow and not self._shadow_r2_not_worse(
            alternative, replacement
        ):
            self.stats["sn_pursuit_shadow_replacement_r2_rejected"] += 1
            self.stats["sn_pursuit_shadow_no_injection"] += 1
            return

        replacement_summary = self._candidate_summary(replacement)
        old_candidate_id = alternative.candidate_id
        alternative.candidate_id = replacement.candidate_id
        alternative.slot = replacement.slot
        alternative.variation_type = (
            "pareto_pursuit_shadow_injected"
            if self.use_sn_pareto_pursuit_shadow
            else "screened_basis_pursuit_shadow_injected"
        )
        children[plan.replacement_index] = alternative
        self.stats["sn_pursuit_shadow_injections"] += 1
        self.stats["sn_pursuit_shadow_amplification_injections"] += int(
            plan.proposal_already_represented
        )
        self.stats["sn_pursuit_shadow_novel_injections"] += int(
            not plan.proposal_already_represented
        )
        self.stats[f"sn_pursuit_shadow_replacement:{plan.replacement_reason}"] += 1
        self.stats["sn_pursuit_shadow_replaced_variation:reproduction"] += 1
        self.stats["sn_pursuit_shadow_injected_coverage_gain_sum"] += float(
            plan.proposal.coverage_gain
        )
        if np.isfinite(alternative.base_reward) and np.isfinite(
            replacement.base_reward
        ):
            self.stats["sn_pursuit_shadow_reward_gain_vs_replaced_sum"] += float(
                alternative.base_reward
            ) - float(replacement.base_reward)
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "screened_pursuit_shadow_injections.jsonl",
                {
                    "generation": alternative.generation,
                    "source_slot": plan.proposal.source_slot,
                    "original_candidate_id": (plan.proposal.original_candidate_id),
                    "candidate_id_before_retag": old_candidate_id,
                    "replacement_index": plan.replacement_index,
                    "replacement_reason": plan.replacement_reason,
                    "replacement": replacement_summary,
                    "injected": self._candidate_summary(alternative),
                    "coverage_gain": plan.proposal.coverage_gain,
                },
            )

    def _dual_path_orthogonal_shadow_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[ControlledIndividual, ShadowProposal | None]:
        """Keep the original child and offer one Base-qualified shadow path."""

        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_shadow_events"] += 1
        proposals = self._build_orthogonal_basis_crossover_proposals(
            parent=parent,
            donor=original_donor,
            X=X,
            y=y,
        )
        self.stats["sn_shadow_source_proposals"] += len(proposals)
        if not proposals:
            self.stats["sn_shadow_no_constructible_alternative"] += 1
            return original, None

        evaluated: list[tuple[ControlledIndividual, dict[str, Any]]] = []
        for tree, details in proposals:
            alternative = self._new_individual(
                tree,
                generation,
                slot,
                (parent, original_donor),
                "orthogonal_basis_shadow_alternative",
            )
            self._evaluate_individual(alternative, X, y)
            self.stats["sn_shadow_alternatives_evaluated"] += 1
            self.stats["sn_shadow_invalid_alternatives"] += int(not alternative.valid)
            evaluated.append((alternative, details))
        best, best_details = min(evaluated, key=lambda item: self._base_key(item[0]))
        reward_delta = (
            float(best.base_reward) - float(original.base_reward)
            if np.isfinite(best.base_reward) and np.isfinite(original.base_reward)
            else None
        )
        if not self._verified_proposal_better(best, original):
            self.stats["sn_shadow_base_rejected"] += 1
            return original, None
        self.stats["sn_shadow_base_qualified"] += 1
        if reward_delta is not None:
            self.stats["sn_shadow_reward_gain_vs_original_sum"] += reward_delta

        basis = self.current_coverage_anchor_basis
        if basis is None:
            self.stats["sn_shadow_geometry_fallbacks"] += 1
            self.stats["sn_shadow_geometry_fallback:anchor_unavailable"] += 1
            return original, None
        vector = self._coverage_vector(best, X, enforce_deadline=True)
        if vector is None:
            self.stats["sn_shadow_geometry_fallbacks"] += 1
            self.stats["sn_shadow_geometry_fallback:alternative_unavailable"] += 1
            return original, None
        try:
            coverage_gain = signature_residual_gain(basis, vector)
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_shadow_geometry_fallbacks"] += 1
            self.stats[f"sn_shadow_geometry_fallback:{type(error).__name__}"] += 1
            return original, None
        self.stats["sn_shadow_geometry_success"] += 1
        self.stats["sn_shadow_candidate_coverage_gain_sum"] += coverage_gain
        best.variation_type = "orthogonal_basis_shadow_candidate"
        proposal = ShadowProposal(
            candidate=best,
            source_slot=slot,
            original_candidate_id=original.candidate_id,
            coverage_gain=float(coverage_gain),
            base_qualified=True,
        )
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "shadow_slot_candidates.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "donor_id": original_donor.candidate_id,
                    "original": self._candidate_summary(original),
                    "candidate": self._candidate_summary(best),
                    "pursuit": best_details,
                    "reward_gain_vs_original": reward_delta,
                    "coverage_gain": coverage_gain,
                },
            )
        return original, proposal

    def _apply_shadow_slot(
        self,
        children: list[ControlledIndividual],
        proposals: Sequence[ShadowProposal],
        *,
        protected_candidate_ids: set[int],
    ) -> None:
        """Inject at most one shadow candidate without removing its original."""

        self.stats["sn_shadow_generations"] += 1
        self.stats["sn_shadow_pool_candidates"] += len(proposals)
        if not proposals:
            self.stats["sn_shadow_no_injection"] += 1
            return
        plan = plan_shadow_injection(
            children,
            proposals,
            candidate_id=lambda value: value.candidate_id,
            canonical_expression=lambda value: (
                value.canonical_fitted_expression or value.raw_expression
            ),
            base_key=self._base_key,
            protected_candidate_ids=protected_candidate_ids,
            minimum_coverage_gain=self.sn_shadow_minimum_coverage_gain,
        )
        if plan is None:
            self.stats["sn_shadow_no_injection"] += 1
            return

        replacement = children[plan.replacement_index]
        alternative = plan.proposal.candidate
        replacement_summary = self._candidate_summary(replacement)
        old_candidate_id = alternative.candidate_id
        alternative.candidate_id = replacement.candidate_id
        alternative.slot = replacement.slot
        alternative.variation_type = "orthogonal_basis_shadow_injected"
        children[plan.replacement_index] = alternative
        self.stats["sn_shadow_injections"] += 1
        self.stats[f"sn_shadow_replacement:{plan.replacement_reason}"] += 1
        self.stats[f"sn_shadow_replaced_variation:{replacement.variation_type}"] += 1
        self.stats["sn_shadow_injected_coverage_gain_sum"] += float(
            plan.proposal.coverage_gain
        )
        if np.isfinite(alternative.base_reward) and np.isfinite(
            replacement.base_reward
        ):
            self.stats["sn_shadow_reward_gain_vs_replaced_sum"] += float(
                alternative.base_reward
            ) - float(replacement.base_reward)
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "shadow_slot_injections.jsonl",
                {
                    "generation": alternative.generation,
                    "source_slot": plan.proposal.source_slot,
                    "original_candidate_id": (plan.proposal.original_candidate_id),
                    "candidate_id_before_retag": old_candidate_id,
                    "replacement_index": plan.replacement_index,
                    "replacement_reason": plan.replacement_reason,
                    "replacement": replacement_summary,
                    "injected": self._candidate_summary(alternative),
                    "coverage_gain": plan.proposal.coverage_gain,
                },
            )

    def _build_orthogonal_basis_crossover_proposals(
        self,
        *,
        parent: ControlledIndividual,
        donor: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[tuple[nd.Symbol, dict[str, Any]]]:
        parent_geometry = self._basis_exchange_geometry(parent, X, "parent")
        donor_geometry = self._basis_exchange_geometry(donor, X, "donor")
        if parent_geometry is None or donor_geometry is None:
            self.stats["sn_orthogonal_crossover_fallbacks"] += 1
            self.stats[
                "sn_orthogonal_crossover_fallback:parent_geometry_unavailable"
            ] += 1
            return []
        parent_fit = parent.fit_result
        donor_fit = donor.fit_result
        assert parent_fit is not None and donor_fit is not None

        pool_by_canonical: dict[str, BasisPoolEntry] = {}
        basis_by_canonical: dict[str, sp.Expr] = {}
        source_term_count = 0
        try:
            for role, geometry, fit in (
                ("receiver", parent_geometry, parent_fit),
                ("donor", donor_geometry, donor_fit),
            ):
                for index, term in enumerate(geometry["terms"]):
                    if not term.basis.free_symbols:
                        continue
                    source_term_count += 1
                    canonical = term.canonical
                    existing = pool_by_canonical.get(canonical)
                    if existing is not None:
                        pool_by_canonical[canonical] = replace(
                            existing,
                            from_receiver=(
                                existing.from_receiver or role == "receiver"
                            ),
                            from_donor=existing.from_donor or role == "donor",
                        )
                        continue
                    values = np.asarray(
                        evaluate_sympy(
                            term.basis,
                            fit.symbols,
                            X,
                            protected_epsilon=1e-6,
                        ),
                        dtype=float,
                    )
                    if values.ndim == 0:
                        values = np.full(len(X), float(values))
                    values = values.reshape(-1)
                    if len(values) != len(X):
                        raise ValueError("basis value column has the wrong row count")
                    values[~np.isfinite(values)] = 0.0
                    signature = np.asarray(
                        geometry["signatures"][:, index], dtype=float
                    ).reshape(-1)
                    pool_by_canonical[canonical] = BasisPoolEntry(
                        canonical=canonical,
                        expression=to_project_expression_string(term.basis),
                        signature=signature,
                        values=values,
                        from_receiver=role == "receiver",
                        from_donor=role == "donor",
                    )
                    basis_by_canonical[canonical] = term.basis
            pool = tuple(pool_by_canonical[key] for key in sorted(pool_by_canonical))
            path = orthogonal_basis_pursuit(
                pool=pool,
                target=y,
                max_steps=self.sn_orthogonal_basis_max_steps,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_orthogonal_crossover_fallbacks"] += 1
            self.stats[f"sn_orthogonal_crossover_fallback:{type(error).__name__}"] += 1
            return []
        if not path:
            self.stats["sn_orthogonal_crossover_fallbacks"] += 1
            self.stats["sn_orthogonal_crossover_fallback:empty_path"] += 1
            return []

        self.stats["sn_orthogonal_crossover_source_terms"] += source_term_count
        self.stats["sn_orthogonal_crossover_pool_terms"] += len(pool)
        self.stats[
            "sn_orthogonal_crossover_canonical_reuses"
        ] += source_term_count - len(pool)
        self.stats["sn_orthogonal_crossover_path_steps"] += len(path)
        selected: list[sp.Expr] = []
        selected_entries: list[BasisPoolEntry] = []
        proposals: list[tuple[nd.Symbol, dict[str, Any]]] = []
        seen_expressions: set[str] = set()
        for step in path:
            entry = pool[step.pool_index]
            selected.append(basis_by_canonical[entry.canonical])
            selected_entries.append(entry)
            try:
                genotype_expression = (
                    selected[0]
                    if len(selected) == 1
                    else sp.Add(*selected, evaluate=False)
                )
                tree = nd.parse(to_project_expression_string(genotype_expression))
                if len(tree) > self.max_len:
                    self.stats["sn_orthogonal_crossover_construction:max_len"] += 1
                    continue
                expression_text = tree.to_str(number_format=".17g")
                roundtrip_expression, roundtrip_symbols = parse_expression(
                    expression_text, self.feature_names
                )
                actual_terms = tuple(
                    term
                    for term in decompose_expand_mul(
                        roundtrip_expression, roundtrip_symbols
                    )
                    if term.basis.free_symbols
                )
                expected = Counter(value.canonical for value in selected_entries)
                actual = Counter(term.canonical for term in actual_terms)
                if actual != expected:
                    self.stats[
                        "sn_orthogonal_crossover_construction:roundtrip_basis_change"
                    ] += 1
                    continue
                if expression_text in seen_expressions:
                    self.stats[
                        "sn_orthogonal_crossover_construction:duplicate_proposal"
                    ] += 1
                    continue
            except (TypeError, ValueError, KeyError) as error:
                self.stats[
                    f"sn_orthogonal_crossover_construction:{type(error).__name__}"
                ] += 1
                continue
            seen_expressions.add(expression_text)
            uses_receiver = any(value.from_receiver for value in selected_entries)
            uses_donor = any(value.from_donor for value in selected_entries)
            details = {
                "path_depth": step.depth,
                "path_length": len(path),
                "selected_canonical": step.canonical,
                "sobolev_gain": step.sobolev_gain,
                "partial_residual_correlation": (step.partial_residual_correlation),
                "value_residual_gain": step.value_residual_gain,
                "joint_score": step.joint_score,
                "residual_norm": step.residual_norm,
                "pool_size": len(pool),
                "uses_receiver": uses_receiver,
                "uses_donor": uses_donor,
                "uses_receiver_and_donor": uses_receiver and uses_donor,
                "selected_terms": [value.expression for value in selected_entries],
                "proposal_expression": expression_text,
            }
            proposals.append((tree, details))
            self.stats["sn_orthogonal_crossover_proposals"] += 1
            self.stats["sn_orthogonal_crossover_cross_parent_proposals"] += int(
                uses_receiver and uses_donor
            )
        if not proposals:
            self.stats["sn_orthogonal_crossover_fallbacks"] += 1
            self.stats[
                "sn_orthogonal_crossover_fallback:no_constructible_proposal"
            ] += 1
        return proposals

    def _verified_conditional_basis_replacement_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Base-verify leave-one-out Sobolev basis replacements."""

        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_conditional_replacement_events"] += 1
        proposals = self._build_conditional_basis_replacement_proposals(
            parent=parent,
            X=X,
            y=y,
            generation=generation,
        )
        if not proposals:
            self.stats["sn_conditional_replacement_original_kept"] += 1
            return original
        self.stats["sn_conditional_replacement_evaluated_events"] += 1

        evaluated: list[
            tuple[
                ControlledIndividual,
                ScreenedArchiveDonor,
                RemovalCandidate,
                dict[str, Any],
            ]
        ] = []
        for tree, donor, removal, details in proposals:
            alternative = self._new_individual(
                tree,
                generation,
                slot,
                (parent,),
                "conditional_basis_replacement_alternative",
            )
            self._evaluate_individual(alternative, X, y)
            self.stats["sn_conditional_replacement_alternatives_evaluated"] += 1
            self.stats["sn_conditional_replacement_invalid_alternatives"] += int(
                not alternative.valid
            )
            evaluated.append((alternative, donor, removal, details))

        best, best_donor, _, best_details = min(
            evaluated, key=lambda item: self._base_key(item[0])
        )
        self.stats["sn_conditional_replacement_base_best_donor_rank_sum"] += int(
            best_details["conditional_donor_rank"]
        )
        self.stats["sn_conditional_replacement_base_best_removal_rank_sum"] += int(
            best_details["removal_rank"]
        )
        self.stats["sn_conditional_replacement_base_best_not_donor_top1"] += int(
            int(best_details["conditional_donor_rank"]) != 1
        )
        self.stats["sn_conditional_replacement_base_best_not_joint_top1"] += int(
            int(best_details["conditional_joint_rank"]) != 1
        )
        self.stats["sn_conditional_replacement_base_best_not_removal_top1"] += int(
            int(best_details["removal_rank"]) != 1
        )

        accepted = self._verified_proposal_better(best, original)
        reward_delta = (
            float(best.base_reward) - float(original.base_reward)
            if np.isfinite(best.base_reward) and np.isfinite(original.base_reward)
            else None
        )
        if accepted:
            best.variation_type = "conditional_basis_replacement_verified"
            self.stats["sn_conditional_replacement_accepted"] += 1
            self.stats[
                "sn_conditional_replacement_donor_age_accepted_sum"
            ] += best_donor.donor_age
            if reward_delta is not None:
                self.stats["sn_conditional_replacement_reward_gain_sum"] += reward_delta
            if np.isfinite(best.complexity) and np.isfinite(original.complexity):
                self.stats["sn_conditional_replacement_complexity_saved_sum"] += float(
                    original.complexity
                ) - float(best.complexity)
            chosen = best
        else:
            self.stats["sn_conditional_replacement_rejected"] += 1
            self.stats["sn_conditional_replacement_original_kept"] += 1
            chosen = original

        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "conditional_basis_replacement_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "original_donor_id": original_donor.candidate_id,
                    "original": self._candidate_summary(original),
                    "alternatives": [
                        {
                            "conditional_screen": details,
                            "candidate": self._candidate_summary(alternative),
                        }
                        for alternative, _, _, details in evaluated
                    ],
                    "base_best": best_details,
                    "base_best_candidate": self._candidate_summary(best),
                    "base_best_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                },
            )
        return chosen

    def _build_conditional_basis_replacement_proposals(
        self,
        *,
        parent: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
        generation: int,
    ) -> list[
        tuple[
            nd.Symbol,
            ScreenedArchiveDonor,
            RemovalCandidate,
            dict[str, Any],
        ]
    ]:
        assert self.basis_archive is not None
        parent_geometry = self._basis_exchange_geometry(parent, X, "parent")
        if parent_geometry is None:
            self.stats["sn_conditional_replacement_fallbacks"] += 1
            self.stats[
                "sn_conditional_replacement_fallback:parent_geometry_unavailable"
            ] += 1
            return []
        parent_fit = parent.fit_result
        assert parent_fit is not None
        parent_terms = parent_geometry["terms"]
        structural = tuple(bool(term.basis.free_symbols) for term in parent_terms)
        value_columns: list[np.ndarray] = []
        try:
            for term in parent_terms:
                column = np.asarray(
                    evaluate_sympy(
                        term.basis,
                        parent_fit.symbols,
                        X,
                        protected_epsilon=1e-6,
                    ),
                    dtype=float,
                )
                if column.ndim == 0:
                    column = np.full(len(X), float(column))
                column = column.reshape(-1)
                if len(column) != len(X):
                    raise ValueError("parent term returned the wrong row count")
                column[~np.isfinite(column)] = 0.0
                value_columns.append(column)
            parent_values = np.column_stack(value_columns)
            removals = rank_low_impact_parent_terms(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_coefficients=tuple(term.coefficient for term in parent_terms),
                parent_novelties=parent_geometry["novelties"],
                parent_norms=parent_geometry["norms"],
                parent_structural=structural,
                tau=self.sobolev_tau,
                shortlist_size=self.sn_basis_replacement_removal_shortlist_size,
            )
            groups = screen_conditional_replacements(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_signatures=parent_geometry["signatures"],
                parent_values=parent_values,
                parent_structural=structural,
                target=y,
                removals=removals,
                archive_entries=self.basis_archive.entries,
                current_generation=generation,
                donor_shortlist_size=self.sn_basis_pursuit_shortlist_size,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_conditional_replacement_fallbacks"] += 1
            self.stats[
                f"sn_conditional_replacement_fallback:{type(error).__name__}"
            ] += 1
            return []
        if not removals:
            self.stats["sn_conditional_replacement_fallbacks"] += 1
            self.stats["sn_conditional_replacement_fallback:no_removable_term"] += 1
            return []
        groups = tuple(group for group in groups if group.donors)
        if not groups:
            self.stats["sn_conditional_replacement_fallbacks"] += 1
            self.stats["sn_conditional_replacement_fallback:no_conditional_donor"] += 1
            return []

        self.stats["sn_conditional_replacement_screened_removals"] += len(removals)
        self.stats["sn_conditional_replacement_conditional_groups"] += len(groups)
        self.stats["sn_conditional_replacement_screened_donors"] += sum(
            len(group.donors) for group in groups
        )
        self.stats["sn_conditional_replacement_shortlists"] += 1
        removal_ranks = {
            removal.parent_index: rank for rank, removal in enumerate(removals, start=1)
        }
        entries = self.basis_archive.entries
        proposals: list[
            tuple[
                nd.Symbol,
                ScreenedArchiveDonor,
                RemovalCandidate,
                dict[str, Any],
            ]
        ] = []
        seen_expressions: set[str] = set()
        for group in groups:
            joint_order = sorted(
                group.donors,
                key=lambda value: (
                    -value.heuristic_joint_score,
                    -value.sobolev_residual_gain,
                    value.canonical,
                    value.archive_index,
                ),
            )
            joint_ranks = {
                donor.archive_index: rank
                for rank, donor in enumerate(joint_order, start=1)
            }
            removal = group.removal
            retained = [
                parent_terms[index]
                for index in group.retained_indices
                if parent_terms[index].basis.free_symbols
            ]
            for donor_rank, donor in enumerate(group.donors, start=1):
                entry = entries[donor.archive_index]
                try:
                    donor_expression, _ = parse_expression(
                        entry.expression, self.feature_names
                    )
                    structural_bases = [term.basis for term in retained] + [
                        donor_expression
                    ]
                    genotype_expression = (
                        structural_bases[0]
                        if len(structural_bases) == 1
                        else sp.Add(*structural_bases, evaluate=False)
                    )
                    tree = nd.parse(to_project_expression_string(genotype_expression))
                    if len(tree) > self.max_len:
                        self.stats[
                            "sn_conditional_replacement_construction:max_len"
                        ] += 1
                        continue
                    expression_text = tree.to_str(number_format=".17g")
                    roundtrip_expression, roundtrip_symbols = parse_expression(
                        expression_text, self.feature_names
                    )
                    actual_terms = tuple(
                        term
                        for term in decompose_expand_mul(
                            roundtrip_expression, roundtrip_symbols
                        )
                        if term.basis.free_symbols
                    )
                    expected = Counter(
                        [term.canonical for term in retained] + [entry.canonical]
                    )
                    actual = Counter(term.canonical for term in actual_terms)
                    if actual != expected:
                        self.stats[
                            "sn_conditional_replacement_construction:roundtrip_basis_change"
                        ] += 1
                        continue
                    if expression_text in seen_expressions:
                        self.stats[
                            "sn_conditional_replacement_construction:duplicate_proposal"
                        ] += 1
                        continue
                except (TypeError, ValueError, KeyError) as error:
                    self.stats[
                        f"sn_conditional_replacement_construction:{type(error).__name__}"
                    ] += 1
                    continue
                seen_expressions.add(expression_text)
                details = {
                    "action": "conditional_replacement",
                    "removal_rank": removal_ranks[removal.parent_index],
                    "conditional_donor_rank": donor_rank,
                    "conditional_joint_rank": joint_ranks[donor.archive_index],
                    "conditional_residual_norm": group.residual_norm,
                    "archive_index": donor.archive_index,
                    "donor_canonical": donor.canonical,
                    "donor_expression": donor.expression,
                    "conditional_sobolev_gain": donor.sobolev_residual_gain,
                    "conditional_partial_residual_correlation": (
                        donor.partial_residual_correlation
                    ),
                    "conditional_value_residual_gain": donor.value_residual_gain,
                    "conditional_joint_score": donor.heuristic_joint_score,
                    "donor_age": donor.donor_age,
                    "donor_source_base_rank": donor.source_base_rank,
                    "removed_parent_index": removal.parent_index,
                    "removed_parent_term": parent_terms[removal.parent_index].display,
                    "removed_parent_canonical": removal.canonical,
                    "removed_parent_novelty": removal.novelty,
                    "removed_deletion_impact": removal.deletion_impact,
                    "proposal_expression": expression_text,
                }
                proposals.append((tree, donor, removal, details))
                self.stats["sn_conditional_replacement_proposals"] += 1
        if not proposals:
            self.stats["sn_conditional_replacement_fallbacks"] += 1
            self.stats[
                "sn_conditional_replacement_fallback:no_constructible_proposal"
            ] += 1
        return proposals

    def _verified_bidirectional_basis_replacement_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Base-verify a shortlist of one-out/one-in Sobolev basis swaps."""

        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_basis_replacement_events"] += 1
        proposals = self._build_bidirectional_basis_replacement_proposals(
            parent=parent,
            X=X,
            y=y,
            generation=generation,
        )
        if not proposals:
            self.stats["sn_basis_replacement_original_kept"] += 1
            return original
        self.stats["sn_basis_replacement_evaluated_events"] += 1

        evaluated: list[
            tuple[
                ControlledIndividual,
                ScreenedArchiveDonor,
                RemovalCandidate,
                dict[str, Any],
            ]
        ] = []
        for tree, donor, removal, details in proposals:
            alternative = self._new_individual(
                tree,
                generation,
                slot,
                (parent,),
                "bidirectional_basis_replacement_alternative",
            )
            self._evaluate_individual(alternative, X, y)
            self.stats["sn_basis_replacement_alternatives_evaluated"] += 1
            self.stats["sn_basis_replacement_invalid_alternatives"] += int(
                not alternative.valid
            )
            evaluated.append((alternative, donor, removal, details))

        best, best_donor, _, best_details = min(
            evaluated, key=lambda item: self._base_key(item[0])
        )
        self.stats["sn_basis_replacement_base_best_screen_rank_sum"] += int(
            best_details["screen_rank"]
        )
        self.stats["sn_basis_replacement_base_best_removal_rank_sum"] += int(
            best_details["removal_rank"]
        )
        self.stats["sn_basis_replacement_base_best_not_screen_top1"] += int(
            int(best_details["screen_rank"]) != 1
        )
        self.stats["sn_basis_replacement_base_best_not_joint_top1"] += int(
            int(best_details["joint_rank"]) != 1
        )
        self.stats["sn_basis_replacement_base_best_not_removal_top1"] += int(
            int(best_details["removal_rank"]) != 1
        )

        accepted = self._verified_proposal_better(best, original)
        reward_delta = (
            float(best.base_reward) - float(original.base_reward)
            if np.isfinite(best.base_reward) and np.isfinite(original.base_reward)
            else None
        )
        if accepted:
            best.variation_type = "bidirectional_basis_replacement_verified"
            self.stats["sn_basis_replacement_accepted"] += 1
            self.stats[
                "sn_basis_replacement_donor_age_accepted_sum"
            ] += best_donor.donor_age
            if reward_delta is not None:
                self.stats["sn_basis_replacement_reward_gain_sum"] += reward_delta
            if np.isfinite(best.complexity) and np.isfinite(original.complexity):
                self.stats["sn_basis_replacement_complexity_saved_sum"] += float(
                    original.complexity
                ) - float(best.complexity)
            chosen = best
        else:
            self.stats["sn_basis_replacement_rejected"] += 1
            self.stats["sn_basis_replacement_original_kept"] += 1
            chosen = original

        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "bidirectional_basis_replacement_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "original_donor_id": original_donor.candidate_id,
                    "original": self._candidate_summary(original),
                    "alternatives": [
                        {
                            "screen": details,
                            "candidate": self._candidate_summary(alternative),
                        }
                        for alternative, _, _, details in evaluated
                    ],
                    "base_best": best_details,
                    "base_best_candidate": self._candidate_summary(best),
                    "base_best_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                },
            )
        return chosen

    def _build_bidirectional_basis_replacement_proposals(
        self,
        *,
        parent: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
        generation: int,
    ) -> list[
        tuple[
            nd.Symbol,
            ScreenedArchiveDonor,
            RemovalCandidate,
            dict[str, Any],
        ]
    ]:
        assert self.basis_archive is not None
        parent_geometry = self._basis_exchange_geometry(parent, X, "parent")
        if parent_geometry is None:
            self.stats["sn_basis_replacement_fallbacks"] += 1
            self.stats["sn_basis_replacement_fallback:parent_geometry_unavailable"] += 1
            return []
        parent_fit = parent.fit_result
        assert parent_fit is not None and parent_fit.fitted_tree is not None
        values = {name: X[:, index] for index, name in enumerate(self.feature_names)}
        with np.errstate(all="ignore"):
            prediction = np.asarray(
                parent_fit.fitted_tree.eval(values, use_eps=1e-6), dtype=float
            )
        if prediction.ndim == 0:
            prediction = np.full(len(y), float(prediction))
        prediction = prediction.reshape(-1)
        prediction[~np.isfinite(prediction)] = 0.0
        residual = np.asarray(y, dtype=float).reshape(-1) - prediction
        parent_terms = parent_geometry["terms"]
        columns = [np.ones(len(X), dtype=float)]
        for term in parent_terms:
            if not term.basis.free_symbols:
                continue
            column = evaluate_sympy(
                term.basis,
                parent_fit.symbols,
                X,
                protected_epsilon=1e-6,
            )
            column = np.asarray(column, dtype=float).reshape(-1)
            column[~np.isfinite(column)] = 0.0
            columns.append(column)
        parent_value_design = np.column_stack(columns)
        try:
            screened = screen_archive_basis_donors(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_signatures=parent_geometry["signatures"],
                parent_residual=residual,
                parent_value_design=parent_value_design,
                archive_entries=self.basis_archive.entries,
                current_generation=generation,
                shortlist_size=self.sn_basis_pursuit_shortlist_size,
            )
            removals = rank_low_impact_parent_terms(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_coefficients=tuple(term.coefficient for term in parent_terms),
                parent_novelties=parent_geometry["novelties"],
                parent_norms=parent_geometry["norms"],
                parent_structural=tuple(
                    bool(term.basis.free_symbols) for term in parent_terms
                ),
                tau=self.sobolev_tau,
                shortlist_size=self.sn_basis_replacement_removal_shortlist_size,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_basis_replacement_fallbacks"] += 1
            self.stats[f"sn_basis_replacement_fallback:{type(error).__name__}"] += 1
            return []
        if not screened:
            self.stats["sn_basis_replacement_fallbacks"] += 1
            self.stats["sn_basis_replacement_fallback:no_screened_donor"] += 1
            return []
        if not removals:
            self.stats["sn_basis_replacement_fallbacks"] += 1
            self.stats["sn_basis_replacement_fallback:no_removable_term"] += 1
            return []

        self.stats["sn_basis_replacement_screened_donors"] += len(screened)
        self.stats["sn_basis_replacement_screened_removals"] += len(removals)
        self.stats["sn_basis_replacement_shortlists"] += 1
        joint_order = sorted(
            screened,
            key=lambda value: (
                -value.heuristic_joint_score,
                -value.sobolev_residual_gain,
                value.canonical,
                value.archive_index,
            ),
        )
        joint_ranks = {
            donor.archive_index: rank for rank, donor in enumerate(joint_order, start=1)
        }
        proposals: list[
            tuple[
                nd.Symbol,
                ScreenedArchiveDonor,
                RemovalCandidate,
                dict[str, Any],
            ]
        ] = []
        seen_expressions: set[str] = set()
        entries = self.basis_archive.entries
        for screen_rank, donor in enumerate(screened, start=1):
            entry = entries[donor.archive_index]
            try:
                donor_expression, _ = parse_expression(
                    entry.expression, self.feature_names
                )
            except (TypeError, ValueError, KeyError) as error:
                self.stats[
                    f"sn_basis_replacement_construction:{type(error).__name__}"
                ] += len(removals)
                continue
            for removal_rank, removal in enumerate(removals, start=1):
                retained = [
                    term
                    for index, term in enumerate(parent_terms)
                    if term.basis.free_symbols and index != removal.parent_index
                ]
                try:
                    structural_bases = [term.basis for term in retained] + [
                        donor_expression
                    ]
                    genotype_expression = (
                        structural_bases[0]
                        if len(structural_bases) == 1
                        else sp.Add(*structural_bases, evaluate=False)
                    )
                    tree = nd.parse(to_project_expression_string(genotype_expression))
                    if len(tree) > self.max_len:
                        self.stats["sn_basis_replacement_construction:max_len"] += 1
                        continue
                    expression_text = tree.to_str(number_format=".17g")
                    roundtrip_expression, roundtrip_symbols = parse_expression(
                        expression_text, self.feature_names
                    )
                    actual_terms = tuple(
                        term
                        for term in decompose_expand_mul(
                            roundtrip_expression, roundtrip_symbols
                        )
                        if term.basis.free_symbols
                    )
                    expected = Counter(
                        [term.canonical for term in retained] + [entry.canonical]
                    )
                    actual = Counter(term.canonical for term in actual_terms)
                    if actual != expected:
                        self.stats[
                            "sn_basis_replacement_construction:roundtrip_basis_change"
                        ] += 1
                        continue
                    if expression_text in seen_expressions:
                        self.stats[
                            "sn_basis_replacement_construction:duplicate_proposal"
                        ] += 1
                        continue
                except (TypeError, ValueError, KeyError) as error:
                    self.stats[
                        f"sn_basis_replacement_construction:{type(error).__name__}"
                    ] += 1
                    continue
                seen_expressions.add(expression_text)
                details = {
                    "action": "replacement",
                    "screen_rank": screen_rank,
                    "joint_rank": joint_ranks[donor.archive_index],
                    "removal_rank": removal_rank,
                    "archive_index": donor.archive_index,
                    "donor_canonical": donor.canonical,
                    "donor_expression": donor.expression,
                    "sobolev_residual_gain": donor.sobolev_residual_gain,
                    "partial_residual_correlation": (
                        donor.partial_residual_correlation
                    ),
                    "value_residual_gain": donor.value_residual_gain,
                    "heuristic_joint_score": donor.heuristic_joint_score,
                    "donor_age": donor.donor_age,
                    "donor_source_base_rank": donor.source_base_rank,
                    "removed_parent_index": removal.parent_index,
                    "removed_parent_term": parent_terms[removal.parent_index].display,
                    "removed_parent_canonical": removal.canonical,
                    "removed_parent_novelty": removal.novelty,
                    "removed_deletion_impact": removal.deletion_impact,
                    "proposal_expression": expression_text,
                }
                proposals.append((tree, donor, removal, details))
                self.stats["sn_basis_replacement_proposals"] += 1
        if not proposals:
            self.stats["sn_basis_replacement_fallbacks"] += 1
            self.stats["sn_basis_replacement_fallback:no_constructible_proposal"] += 1
        return proposals

    def _verified_screened_basis_pursuit_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Screen archived directions with SN, then let exact Base refits decide."""

        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_basis_pursuit_events"] += 1
        proposals = self._build_screened_basis_pursuit_proposals(
            parent=parent,
            X=X,
            y=y,
            generation=generation,
        )
        if not proposals:
            self.stats["sn_basis_pursuit_original_kept"] += 1
            return original
        self.stats["sn_basis_pursuit_evaluated_events"] += 1

        evaluated: list[
            tuple[
                ControlledIndividual,
                ScreenedArchiveDonor,
                str,
                dict[str, Any],
            ]
        ] = []
        for tree, donor, action, details in proposals:
            alternative = self._new_individual(
                tree,
                generation,
                slot,
                (parent,),
                f"screened_basis_{action}_alternative",
            )
            self._evaluate_individual(alternative, X, y)
            self.stats["sn_basis_pursuit_alternatives_evaluated"] += 1
            self.stats[f"sn_basis_pursuit_{action}_evaluated"] += 1
            self.stats["sn_basis_pursuit_invalid_alternatives"] += int(
                not alternative.valid
            )
            evaluated.append((alternative, donor, action, details))

        best, best_donor, best_action, best_details = min(
            evaluated, key=lambda item: self._base_key(item[0])
        )
        self.stats["sn_basis_pursuit_base_best_screen_rank_sum"] += int(
            best_details["screen_rank"]
        )
        self.stats["sn_basis_pursuit_base_best_not_screen_top1"] += int(
            int(best_details["screen_rank"]) != 1
        )
        self.stats["sn_basis_pursuit_base_best_not_joint_top1"] += int(
            int(best_details["joint_rank"]) != 1
        )
        self.stats[f"sn_basis_pursuit_{best_action}_base_best"] += 1

        accepted = self._verified_proposal_better(best, original)
        reward_delta = (
            float(best.base_reward) - float(original.base_reward)
            if np.isfinite(best.base_reward) and np.isfinite(original.base_reward)
            else None
        )
        if accepted:
            best.variation_type = f"screened_basis_{best_action}_verified"
            self.stats["sn_basis_pursuit_accepted"] += 1
            self.stats[f"sn_basis_pursuit_{best_action}_accepted"] += 1
            self.stats[
                "sn_basis_pursuit_donor_age_accepted_sum"
            ] += best_donor.donor_age
            if reward_delta is not None:
                self.stats["sn_basis_pursuit_reward_gain_sum"] += reward_delta
            if np.isfinite(best.complexity) and np.isfinite(original.complexity):
                self.stats["sn_basis_pursuit_complexity_saved_sum"] += float(
                    original.complexity
                ) - float(best.complexity)
            chosen = best
        else:
            self.stats["sn_basis_pursuit_rejected"] += 1
            self.stats["sn_basis_pursuit_original_kept"] += 1
            chosen = original

        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "screened_basis_pursuit_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "original_donor_id": original_donor.candidate_id,
                    "original": self._candidate_summary(original),
                    "alternatives": [
                        {
                            "screen": details,
                            "candidate": self._candidate_summary(alternative),
                        }
                        for alternative, _, _, details in evaluated
                    ],
                    "base_best": best_details,
                    "base_best_candidate": self._candidate_summary(best),
                    "base_best_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                },
            )
        return chosen

    def _build_screened_basis_pursuit_proposals(
        self,
        *,
        parent: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
        generation: int,
    ) -> list[tuple[nd.Symbol, ScreenedArchiveDonor, str, dict[str, Any]]]:
        assert self.basis_archive is not None
        parent_geometry = self._basis_exchange_geometry(parent, X, "parent")
        if parent_geometry is None:
            self.stats["sn_basis_pursuit_fallbacks"] += 1
            self.stats["sn_basis_pursuit_fallback:parent_geometry_unavailable"] += 1
            return []
        parent_fit = parent.fit_result
        assert parent_fit is not None and parent_fit.fitted_tree is not None
        values = {name: X[:, index] for index, name in enumerate(self.feature_names)}
        with np.errstate(all="ignore"):
            prediction = np.asarray(
                parent_fit.fitted_tree.eval(values, use_eps=1e-6), dtype=float
            )
        if prediction.ndim == 0:
            prediction = np.full(len(y), float(prediction))
        prediction = prediction.reshape(-1)
        prediction[~np.isfinite(prediction)] = 0.0
        residual = np.asarray(y, dtype=float).reshape(-1) - prediction
        parent_terms = parent_geometry["terms"]
        columns = [np.ones(len(X), dtype=float)]
        for term in parent_terms:
            if not term.basis.free_symbols:
                continue
            column = evaluate_sympy(
                term.basis,
                parent_fit.symbols,
                X,
                protected_epsilon=1e-6,
            )
            column = np.asarray(column, dtype=float).reshape(-1)
            column[~np.isfinite(column)] = 0.0
            columns.append(column)
        parent_value_design = np.column_stack(columns)
        try:
            screened = screen_archive_basis_donors(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_signatures=parent_geometry["signatures"],
                parent_residual=residual,
                parent_value_design=parent_value_design,
                archive_entries=self.basis_archive.entries,
                current_generation=generation,
                shortlist_size=self.sn_basis_pursuit_shortlist_size,
            )
            removed_index = select_low_impact_parent_term(
                parent_coefficients=tuple(term.coefficient for term in parent_terms),
                parent_novelties=parent_geometry["novelties"],
                parent_norms=parent_geometry["norms"],
                parent_structural=tuple(
                    bool(term.basis.free_symbols) for term in parent_terms
                ),
                tau=self.sobolev_tau,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_basis_pursuit_fallbacks"] += 1
            self.stats[f"sn_basis_pursuit_fallback:{type(error).__name__}"] += 1
            return []
        if not screened:
            self.stats["sn_basis_pursuit_fallbacks"] += 1
            self.stats["sn_basis_pursuit_fallback:no_screened_donor"] += 1
            return []

        self.stats["sn_basis_pursuit_screened_donors"] += len(screened)
        self.stats["sn_basis_pursuit_shortlists"] += 1
        joint_order = sorted(
            screened,
            key=lambda value: (
                -value.heuristic_joint_score,
                -value.sobolev_residual_gain,
                value.canonical,
                value.archive_index,
            ),
        )
        joint_ranks = {
            donor.archive_index: rank for rank, donor in enumerate(joint_order, start=1)
        }
        proposals: list[tuple[nd.Symbol, ScreenedArchiveDonor, str, dict[str, Any]]] = (
            []
        )
        seen_expressions: set[str] = set()
        entries = self.basis_archive.entries
        for screen_rank, donor in enumerate(screened, start=1):
            entry = entries[donor.archive_index]
            actions: list[tuple[str, int | None]] = [("augment", None)]
            if removed_index is not None:
                actions.append(("exchange", removed_index))
            try:
                donor_expression, _ = parse_expression(
                    entry.expression, self.feature_names
                )
            except (TypeError, ValueError, KeyError) as error:
                self.stats[
                    f"sn_basis_pursuit_construction:{type(error).__name__}"
                ] += len(actions)
                continue
            for action, removed in actions:
                retained = [
                    term
                    for index, term in enumerate(parent_terms)
                    if term.basis.free_symbols and index != removed
                ]
                try:
                    structural_bases = [term.basis for term in retained] + [
                        donor_expression
                    ]
                    genotype_expression = (
                        structural_bases[0]
                        if len(structural_bases) == 1
                        else sp.Add(*structural_bases, evaluate=False)
                    )
                    tree = nd.parse(to_project_expression_string(genotype_expression))
                    if len(tree) > self.max_len:
                        self.stats["sn_basis_pursuit_construction:max_len"] += 1
                        continue
                    expression_text = tree.to_str(number_format=".17g")
                    roundtrip_expression, roundtrip_symbols = parse_expression(
                        expression_text, self.feature_names
                    )
                    actual_terms = tuple(
                        term
                        for term in decompose_expand_mul(
                            roundtrip_expression, roundtrip_symbols
                        )
                        if term.basis.free_symbols
                    )
                    expected = Counter(
                        [term.canonical for term in retained] + [entry.canonical]
                    )
                    actual = Counter(term.canonical for term in actual_terms)
                    if actual != expected:
                        self.stats[
                            "sn_basis_pursuit_construction:roundtrip_basis_change"
                        ] += 1
                        continue
                    if expression_text in seen_expressions:
                        self.stats[
                            "sn_basis_pursuit_construction:duplicate_proposal"
                        ] += 1
                        continue
                except (TypeError, ValueError, KeyError) as error:
                    self.stats[
                        f"sn_basis_pursuit_construction:{type(error).__name__}"
                    ] += 1
                    continue
                seen_expressions.add(expression_text)
                removed_impact = (
                    None
                    if removed is None
                    else abs(float(parent_terms[removed].coefficient))
                    * float(parent_geometry["novelties"][removed])
                    * float(parent_geometry["norms"][removed])
                )
                details = {
                    "action": action,
                    "screen_rank": screen_rank,
                    "joint_rank": joint_ranks[donor.archive_index],
                    "archive_index": donor.archive_index,
                    "donor_canonical": donor.canonical,
                    "donor_expression": donor.expression,
                    "sobolev_residual_gain": donor.sobolev_residual_gain,
                    "partial_residual_correlation": (
                        donor.partial_residual_correlation
                    ),
                    "value_residual_gain": donor.value_residual_gain,
                    "heuristic_joint_score": donor.heuristic_joint_score,
                    "donor_age": donor.donor_age,
                    "donor_source_base_rank": donor.source_base_rank,
                    "removed_parent_index": removed,
                    "removed_parent_term": (
                        None if removed is None else parent_terms[removed].display
                    ),
                    "removed_deletion_impact": removed_impact,
                    "proposal_expression": expression_text,
                }
                proposals.append((tree, donor, action, details))
                self.stats["sn_basis_pursuit_proposals"] += 1
                self.stats[f"sn_basis_pursuit_{action}_proposals"] += 1
        if not proposals:
            self.stats["sn_basis_pursuit_fallbacks"] += 1
            self.stats["sn_basis_pursuit_fallback:no_constructible_proposal"] += 1
        return proposals

    def _verified_archive_basis_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Base-verify one cross-generation archived-basis proposal."""

        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_basis_archive_events"] += 1
        proposal = self._build_archive_basis_infusion_proposal(
            parent=parent,
            X=X,
            y=y,
            generation=generation,
        )
        if proposal is None:
            self.stats["sn_basis_archive_original_kept"] += 1
            return original
        alternative_tree, choice, archive_details = proposal
        alternative = self._new_individual(
            alternative_tree,
            generation,
            slot,
            (parent,),
            f"archive_basis_{choice.action}_alternative",
        )
        self._evaluate_individual(alternative, X, y)
        self.stats["sn_basis_archive_alternatives_evaluated"] += 1
        accepted = self._verified_proposal_better(alternative, original)
        reward_delta = (
            float(alternative.base_reward) - float(original.base_reward)
            if np.isfinite(alternative.base_reward)
            and np.isfinite(original.base_reward)
            else None
        )
        if accepted:
            alternative.variation_type = f"archive_basis_{choice.action}_verified"
            self.stats["sn_basis_archive_accepted"] += 1
            self.stats[f"sn_basis_archive_{choice.action}_accepted"] += 1
            self.stats["sn_basis_archive_donor_age_accepted_sum"] += choice.donor_age
            if reward_delta is not None:
                self.stats["sn_basis_archive_reward_gain_sum"] += reward_delta
            if np.isfinite(alternative.complexity) and np.isfinite(original.complexity):
                self.stats["sn_basis_archive_complexity_saved_sum"] += float(
                    original.complexity
                ) - float(alternative.complexity)
            chosen = alternative
        else:
            self.stats["sn_basis_archive_rejected"] += 1
            self.stats["sn_basis_archive_original_kept"] += 1
            chosen = original
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_archive_infusion_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "original_donor_id": original_donor.candidate_id,
                    "choice": archive_details,
                    "original": self._candidate_summary(original),
                    "alternative": self._candidate_summary(alternative),
                    "alternative_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                },
            )
        return chosen

    def _build_archive_basis_infusion_proposal(
        self,
        *,
        parent: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
        generation: int,
    ) -> tuple[nd.Symbol, ArchiveInfusionChoice, dict[str, Any]] | None:
        assert self.basis_archive is not None
        parent_geometry = self._basis_exchange_geometry(parent, X, "parent")
        if parent_geometry is None:
            self.stats["sn_basis_archive_fallbacks"] += 1
            self.stats["sn_basis_archive_fallback:parent_geometry_unavailable"] += 1
            return None
        parent_fit = parent.fit_result
        assert parent_fit is not None and parent_fit.fitted_tree is not None
        values = {name: X[:, index] for index, name in enumerate(self.feature_names)}
        with np.errstate(all="ignore"):
            prediction = np.asarray(
                parent_fit.fitted_tree.eval(values, use_eps=1e-6), dtype=float
            )
        if prediction.ndim == 0:
            prediction = np.full(len(y), float(prediction))
        prediction = prediction.reshape(-1)
        prediction[~np.isfinite(prediction)] = 0.0
        residual = np.asarray(y, dtype=float).reshape(-1) - prediction
        parent_terms = parent_geometry["terms"]
        entries = self.basis_archive.entries
        parent_value_design: np.ndarray | None = None
        if self.use_sn_partial_residual_archive:
            columns = [np.ones(len(X), dtype=float)]
            for term in parent_terms:
                if not term.basis.free_symbols:
                    continue
                column = evaluate_sympy(
                    term.basis,
                    parent_fit.symbols,
                    X,
                    protected_epsilon=1e-6,
                )
                column = np.asarray(column, dtype=float).reshape(-1)
                column[~np.isfinite(column)] = 0.0
                columns.append(column)
            parent_value_design = np.column_stack(columns)
        try:
            choice = select_archive_residual_infusion(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_coefficients=tuple(term.coefficient for term in parent_terms),
                parent_novelties=parent_geometry["novelties"],
                parent_norms=parent_geometry["norms"],
                parent_signatures=parent_geometry["signatures"],
                parent_structural=tuple(
                    bool(term.basis.free_symbols) for term in parent_terms
                ),
                parent_residual=residual,
                archive_entries=entries,
                tau=self.sobolev_tau,
                current_generation=generation,
                parent_value_design=parent_value_design,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_basis_archive_fallbacks"] += 1
            self.stats[f"sn_basis_archive_fallback:{type(error).__name__}"] += 1
            return None
        if choice is None:
            self.stats["sn_basis_archive_fallbacks"] += 1
            self.stats["sn_basis_archive_fallback:no_joint_donor_signal"] += 1
            return None
        selected_entry = entries[choice.selected_archive_index]
        retained = [
            term
            for index, term in enumerate(parent_terms)
            if term.basis.free_symbols and index != choice.removed_parent_index
        ]
        try:
            donor_expression, _ = parse_expression(
                selected_entry.expression, self.feature_names
            )
            structural_bases = [term.basis for term in retained] + [donor_expression]
            genotype_expression = (
                structural_bases[0]
                if len(structural_bases) == 1
                else sp.Add(*structural_bases, evaluate=False)
            )
            tree = nd.parse(to_project_expression_string(genotype_expression))
            if len(tree) > self.max_len:
                self.stats["sn_basis_archive_fallbacks"] += 1
                self.stats["sn_basis_archive_fallback:max_len"] += 1
                return None
            roundtrip_expression, roundtrip_symbols = parse_expression(
                tree.to_str(number_format=".17g"),
                self.feature_names,
            )
            actual_terms = tuple(
                term
                for term in decompose_expand_mul(
                    roundtrip_expression,
                    roundtrip_symbols,
                )
                if term.basis.free_symbols
            )
            expected = Counter(
                [term.canonical for term in retained] + [selected_entry.canonical]
            )
            actual = Counter(term.canonical for term in actual_terms)
            if actual != expected:
                self.stats["sn_basis_archive_fallbacks"] += 1
                self.stats["sn_basis_archive_fallback:roundtrip_basis_change"] += 1
                return None
        except (TypeError, ValueError, KeyError) as error:
            self.stats["sn_basis_archive_fallbacks"] += 1
            self.stats[
                f"sn_basis_archive_fallback:construction_{type(error).__name__}"
            ] += 1
            return None
        self.stats["sn_basis_archive_proposals"] += 1
        self.stats[f"sn_basis_archive_{choice.action}_proposals"] += 1
        self.stats[
            "sn_basis_archive_target_correlation_sum"
        ] += choice.donor_target_correlation
        self.stats["sn_basis_archive_donor_gain_sum"] += choice.donor_residual_gain
        self.stats["sn_basis_archive_joint_score_sum"] += choice.donor_joint_score
        self.stats[
            "sn_basis_archive_value_residual_gain_sum"
        ] += choice.donor_value_residual_gain
        self.stats["sn_basis_archive_donor_age_sum"] += choice.donor_age
        self.stats[
            "sn_basis_archive_donor_source_rank_sum"
        ] += choice.donor_source_base_rank
        details = {
            "action": choice.action,
            "removed_parent_index": choice.removed_parent_index,
            "removed_parent_term": (
                None
                if choice.removed_parent_index is None
                else parent_terms[choice.removed_parent_index].display
            ),
            "removed_deletion_impact": choice.removed_deletion_impact,
            "removed_normalized_impact": choice.removed_normalized_impact,
            "removed_novelty": choice.removed_novelty,
            "selected_archive_index": choice.selected_archive_index,
            "selected_donor_term": choice.selected_expression,
            "selected_donor_canonical": choice.selected_canonical,
            "donor_target_correlation": choice.donor_target_correlation,
            "donor_residual_gain": choice.donor_residual_gain,
            "donor_joint_score": choice.donor_joint_score,
            "donor_value_residual_gain": choice.donor_value_residual_gain,
            "donor_source_generation": choice.donor_source_generation,
            "donor_source_base_rank": choice.donor_source_base_rank,
            "donor_source_candidate_id": choice.donor_source_candidate_id,
            "donor_age": choice.donor_age,
            "proposal_expression": tree.to_str(number_format=".17g"),
        }
        return tree, choice, details

    def _build_residual_basis_infusion_proposal(
        self,
        *,
        parent: ControlledIndividual,
        donor: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[nd.Symbol, ResidualInfusionChoice, dict[str, Any]] | None:
        parent_geometry = self._basis_exchange_geometry(parent, X, "parent")
        donor_geometry = self._basis_exchange_geometry(donor, X, "donor")
        if parent_geometry is None or donor_geometry is None:
            self.stats["sn_residual_infusion_fallbacks"] += 1
            self.stats["sn_residual_infusion_fallback:geometry_unavailable"] += 1
            return None
        parent_fit = parent.fit_result
        donor_fit = donor.fit_result
        assert parent_fit is not None and parent_fit.fitted_tree is not None
        assert donor_fit is not None
        values = {name: X[:, index] for index, name in enumerate(self.feature_names)}
        with np.errstate(all="ignore"):
            prediction = np.asarray(
                parent_fit.fitted_tree.eval(values, use_eps=1e-6), dtype=float
            )
        if prediction.ndim == 0:
            prediction = np.full(len(y), float(prediction))
        prediction = prediction.reshape(-1)
        prediction[~np.isfinite(prediction)] = 0.0
        residual = np.asarray(y, dtype=float).reshape(-1) - prediction
        donor_correlations: list[float] = []
        try:
            for term in donor_geometry["terms"]:
                if not term.basis.free_symbols:
                    donor_correlations.append(0.0)
                    continue
                term_values = evaluate_sympy(term.basis, donor_fit.symbols, X)
                term_values = np.asarray(term_values, dtype=float).reshape(-1)
                term_values[~np.isfinite(term_values)] = 0.0
                donor_correlations.append(
                    centered_absolute_correlation(term_values, residual)
                )
        except (TypeError, ValueError, KeyError) as error:
            self.stats["sn_residual_infusion_fallbacks"] += 1
            self.stats[
                f"sn_residual_infusion_fallback:target_credit_{type(error).__name__}"
            ] += 1
            return None
        parent_terms = parent_geometry["terms"]
        donor_terms = donor_geometry["terms"]
        try:
            choice = select_residual_basis_infusion(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_coefficients=tuple(term.coefficient for term in parent_terms),
                parent_novelties=parent_geometry["novelties"],
                parent_norms=parent_geometry["norms"],
                parent_signatures=parent_geometry["signatures"],
                parent_structural=tuple(
                    bool(term.basis.free_symbols) for term in parent_terms
                ),
                donor_canonicals=tuple(term.canonical for term in donor_terms),
                donor_coefficients=tuple(term.coefficient for term in donor_terms),
                donor_norms=donor_geometry["norms"],
                donor_signatures=donor_geometry["signatures"],
                donor_structural=tuple(
                    bool(term.basis.free_symbols) for term in donor_terms
                ),
                donor_target_correlations=tuple(donor_correlations),
                tau=self.sobolev_tau,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_residual_infusion_fallbacks"] += 1
            self.stats[f"sn_residual_infusion_fallback:{type(error).__name__}"] += 1
            return None
        if choice is None:
            self.stats["sn_residual_infusion_fallbacks"] += 1
            self.stats["sn_residual_infusion_fallback:no_joint_donor_signal"] += 1
            return None
        retained = [
            term
            for index, term in enumerate(parent_terms)
            if term.basis.free_symbols and index != choice.removed_parent_index
        ]
        selected_donor = donor_terms[choice.selected_donor_index]
        structural = retained + [selected_donor]
        genotype_expression = (
            structural[0].basis
            if len(structural) == 1
            else sp.Add(*(term.basis for term in structural), evaluate=False)
        )
        try:
            tree = nd.parse(to_project_expression_string(genotype_expression))
            if len(tree) > self.max_len:
                self.stats["sn_residual_infusion_fallbacks"] += 1
                self.stats["sn_residual_infusion_fallback:max_len"] += 1
                return None
            roundtrip_expression, roundtrip_symbols = parse_expression(
                tree.to_str(number_format=".17g"),
                self.feature_names,
            )
            actual_terms = tuple(
                term
                for term in decompose_expand_mul(
                    roundtrip_expression,
                    roundtrip_symbols,
                )
                if term.basis.free_symbols
            )
            expected = Counter(term.canonical for term in structural)
            actual = Counter(term.canonical for term in actual_terms)
            if actual != expected:
                self.stats["sn_residual_infusion_fallbacks"] += 1
                self.stats["sn_residual_infusion_fallback:roundtrip_basis_change"] += 1
                return None
        except (TypeError, ValueError, KeyError) as error:
            self.stats["sn_residual_infusion_fallbacks"] += 1
            self.stats[
                f"sn_residual_infusion_fallback:construction_{type(error).__name__}"
            ] += 1
            return None
        self.stats["sn_residual_infusion_proposals"] += 1
        self.stats[f"sn_residual_infusion_{choice.action}_proposals"] += 1
        self.stats[
            "sn_residual_infusion_target_correlation_sum"
        ] += choice.donor_target_correlation
        self.stats["sn_residual_infusion_donor_gain_sum"] += choice.donor_residual_gain
        self.stats["sn_residual_infusion_joint_score_sum"] += choice.donor_joint_score
        self.stats[
            "sn_residual_infusion_donor_normalized_innovation_sum"
        ] += choice.donor_normalized_innovation
        details = {
            "action": choice.action,
            "removed_parent_index": choice.removed_parent_index,
            "removed_parent_term": (
                None
                if choice.removed_parent_index is None
                else parent_terms[choice.removed_parent_index].display
            ),
            "removed_deletion_impact": choice.removed_deletion_impact,
            "removed_normalized_impact": choice.removed_normalized_impact,
            "removed_novelty": choice.removed_novelty,
            "selected_donor_index": choice.selected_donor_index,
            "selected_donor_term": selected_donor.display,
            "donor_target_correlation": choice.donor_target_correlation,
            "donor_residual_gain": choice.donor_residual_gain,
            "donor_joint_score": choice.donor_joint_score,
            "donor_normalized_innovation": choice.donor_normalized_innovation,
            "proposal_expression": tree.to_str(number_format=".17g"),
        }
        return tree, choice, details

    def _verified_basis_infusion_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Base-verify one adaptive exchange-or-augmentation proposal."""

        coverage_donor = self.current_coverage_candidate
        assert coverage_donor is not None
        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_basis_infusion_events"] += 1
        proposal = self._build_basis_infusion_proposal(
            parent=parent,
            donor=coverage_donor,
            X=X,
        )
        if proposal is None:
            self.stats["sn_basis_infusion_original_kept"] += 1
            return original
        alternative_tree, choice, infusion_details = proposal
        alternative = self._new_individual(
            alternative_tree,
            generation,
            slot,
            (parent, coverage_donor),
            f"basis_{choice.action}_alternative",
        )
        self._evaluate_individual(alternative, X, y)
        self.stats["sn_basis_infusion_alternatives_evaluated"] += 1
        accepted = self._verified_proposal_better(alternative, original)
        reward_delta = (
            float(alternative.base_reward) - float(original.base_reward)
            if np.isfinite(alternative.base_reward)
            and np.isfinite(original.base_reward)
            else None
        )
        if accepted:
            alternative.variation_type = f"basis_{choice.action}_verified"
            self.stats["sn_basis_infusion_accepted"] += 1
            self.stats[f"sn_basis_infusion_{choice.action}_accepted"] += 1
            if reward_delta is not None:
                self.stats["sn_basis_infusion_reward_gain_sum"] += reward_delta
            if np.isfinite(alternative.complexity) and np.isfinite(original.complexity):
                self.stats["sn_basis_infusion_complexity_saved_sum"] += float(
                    original.complexity
                ) - float(alternative.complexity)
            chosen = alternative
        else:
            self.stats["sn_basis_infusion_rejected"] += 1
            self.stats["sn_basis_infusion_original_kept"] += 1
            chosen = original
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_infusion_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "original_donor_id": original_donor.candidate_id,
                    "coverage_donor_id": coverage_donor.candidate_id,
                    "coverage_gain": self.current_coverage_gain,
                    "choice": infusion_details,
                    "original": self._candidate_summary(original),
                    "alternative": self._candidate_summary(alternative),
                    "alternative_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                },
            )
        return chosen

    def _build_basis_infusion_proposal(
        self,
        *,
        parent: ControlledIndividual,
        donor: ControlledIndividual,
        X: np.ndarray,
    ) -> tuple[nd.Symbol, BasisInfusionChoice, dict[str, Any]] | None:
        parent_geometry = self._basis_exchange_geometry(parent, X, "parent")
        donor_geometry = self._basis_exchange_geometry(donor, X, "donor")
        if parent_geometry is None or donor_geometry is None:
            self.stats["sn_basis_infusion_fallbacks"] += 1
            self.stats["sn_basis_infusion_fallback:geometry_unavailable"] += 1
            return None
        parent_terms = parent_geometry["terms"]
        donor_terms = donor_geometry["terms"]
        try:
            choice = select_basis_infusion(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_coefficients=tuple(term.coefficient for term in parent_terms),
                parent_novelties=parent_geometry["novelties"],
                parent_norms=parent_geometry["norms"],
                parent_signatures=parent_geometry["signatures"],
                parent_structural=tuple(
                    bool(term.basis.free_symbols) for term in parent_terms
                ),
                donor_canonicals=tuple(term.canonical for term in donor_terms),
                donor_coefficients=tuple(term.coefficient for term in donor_terms),
                donor_norms=donor_geometry["norms"],
                donor_signatures=donor_geometry["signatures"],
                donor_structural=tuple(
                    bool(term.basis.free_symbols) for term in donor_terms
                ),
                tau=self.sobolev_tau,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_basis_infusion_fallbacks"] += 1
            self.stats[f"sn_basis_infusion_fallback:{type(error).__name__}"] += 1
            return None
        if choice is None:
            self.stats["sn_basis_infusion_fallbacks"] += 1
            self.stats["sn_basis_infusion_fallback:no_new_donor_direction"] += 1
            return None
        retained = [
            term
            for index, term in enumerate(parent_terms)
            if term.basis.free_symbols and index != choice.removed_parent_index
        ]
        selected_donor = donor_terms[choice.selected_donor_index]
        structural = retained + [selected_donor]
        genotype_expression = (
            structural[0].basis
            if len(structural) == 1
            else sp.Add(*(term.basis for term in structural), evaluate=False)
        )
        try:
            tree = nd.parse(to_project_expression_string(genotype_expression))
            if len(tree) > self.max_len:
                self.stats["sn_basis_infusion_fallbacks"] += 1
                self.stats["sn_basis_infusion_fallback:max_len"] += 1
                return None
            roundtrip_expression, roundtrip_symbols = parse_expression(
                tree.to_str(number_format=".17g"),
                self.feature_names,
            )
            actual_terms = tuple(
                term
                for term in decompose_expand_mul(
                    roundtrip_expression,
                    roundtrip_symbols,
                )
                if term.basis.free_symbols
            )
            expected = Counter(term.canonical for term in structural)
            actual = Counter(term.canonical for term in actual_terms)
            if actual != expected:
                self.stats["sn_basis_infusion_fallbacks"] += 1
                self.stats["sn_basis_infusion_fallback:roundtrip_basis_change"] += 1
                return None
        except (TypeError, ValueError, KeyError) as error:
            self.stats["sn_basis_infusion_fallbacks"] += 1
            self.stats[
                f"sn_basis_infusion_fallback:construction_{type(error).__name__}"
            ] += 1
            return None
        self.stats["sn_basis_infusion_proposals"] += 1
        self.stats[f"sn_basis_infusion_{choice.action}_proposals"] += 1
        if choice.removed_deletion_impact is not None:
            self.stats[
                "sn_basis_infusion_removed_impact_sum"
            ] += choice.removed_deletion_impact
            self.stats["sn_basis_infusion_removed_normalized_impact_sum"] += (
                choice.removed_normalized_impact
                if choice.removed_normalized_impact is not None
                else 0.0
            )
        self.stats["sn_basis_infusion_donor_gain_sum"] += choice.donor_residual_gain
        self.stats[
            "sn_basis_infusion_donor_normalized_innovation_sum"
        ] += choice.donor_normalized_innovation
        details = {
            "action": choice.action,
            "removed_parent_index": choice.removed_parent_index,
            "removed_parent_term": (
                None
                if choice.removed_parent_index is None
                else parent_terms[choice.removed_parent_index].display
            ),
            "removed_deletion_impact": choice.removed_deletion_impact,
            "removed_normalized_impact": choice.removed_normalized_impact,
            "removed_novelty": choice.removed_novelty,
            "selected_donor_index": choice.selected_donor_index,
            "selected_donor_term": selected_donor.display,
            "donor_residual_gain": choice.donor_residual_gain,
            "donor_amplitude": choice.donor_amplitude,
            "donor_score": choice.donor_score,
            "donor_normalized_innovation": choice.donor_normalized_innovation,
            "proposal_expression": tree.to_str(number_format=".17g"),
        }
        return tree, choice, details

    def _verified_basis_exchange_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Base-verify one Sobolev-guided additive-basis exchange proposal."""

        coverage_donor = self.current_coverage_candidate
        assert coverage_donor is not None
        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_basis_exchange_events"] += 1
        proposal = self._build_basis_exchange_proposal(
            parent=parent,
            donor=coverage_donor,
            X=X,
        )
        if proposal is None:
            self.stats["sn_basis_exchange_original_kept"] += 1
            return original
        alternative_tree, choice, exchange_details = proposal
        alternative = self._new_individual(
            alternative_tree,
            generation,
            slot,
            (parent, coverage_donor),
            "basis_exchange_alternative",
        )
        self._evaluate_individual(alternative, X, y)
        self.stats["sn_basis_exchange_alternatives_evaluated"] += 1
        accepted = self._verified_proposal_better(alternative, original)
        reward_delta = (
            float(alternative.base_reward) - float(original.base_reward)
            if np.isfinite(alternative.base_reward)
            and np.isfinite(original.base_reward)
            else None
        )
        if accepted:
            alternative.variation_type = "basis_exchange_verified"
            self.stats["sn_basis_exchange_accepted"] += 1
            if reward_delta is not None:
                self.stats["sn_basis_exchange_reward_gain_sum"] += reward_delta
            if np.isfinite(alternative.complexity) and np.isfinite(original.complexity):
                self.stats["sn_basis_exchange_complexity_saved_sum"] += float(
                    original.complexity
                ) - float(alternative.complexity)
            chosen = alternative
        else:
            self.stats["sn_basis_exchange_rejected"] += 1
            self.stats["sn_basis_exchange_original_kept"] += 1
            chosen = original
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_exchange_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "original_donor_id": original_donor.candidate_id,
                    "coverage_donor_id": coverage_donor.candidate_id,
                    "coverage_gain": self.current_coverage_gain,
                    "choice": exchange_details,
                    "original": self._candidate_summary(original),
                    "alternative": self._candidate_summary(alternative),
                    "alternative_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                },
            )
        return chosen

    def _build_basis_exchange_proposal(
        self,
        *,
        parent: ControlledIndividual,
        donor: ControlledIndividual,
        X: np.ndarray,
    ) -> tuple[nd.Symbol, BasisExchangeChoice, dict[str, Any]] | None:
        parent_geometry = self._basis_exchange_geometry(parent, X, "parent")
        donor_geometry = self._basis_exchange_geometry(donor, X, "donor")
        if parent_geometry is None or donor_geometry is None:
            self.stats["sn_basis_exchange_fallbacks"] += 1
            self.stats["sn_basis_exchange_fallback:geometry_unavailable"] += 1
            return None
        parent_terms = parent_geometry["terms"]
        donor_terms = donor_geometry["terms"]
        eligible_parent = [
            index
            for index, term in enumerate(parent_terms)
            if term.basis.free_symbols
            and parent_geometry["novelties"][index] < self.sobolev_tau
        ]
        if not eligible_parent:
            self.stats["sn_basis_exchange_fallbacks"] += 1
            self.stats["sn_basis_exchange_fallback:no_low_novelty_parent_term"] += 1
            return None
        parent_canonicals = {term.canonical for term in parent_terms}
        if not any(
            term.basis.free_symbols and term.canonical not in parent_canonicals
            for term in donor_terms
        ):
            self.stats["sn_basis_exchange_fallbacks"] += 1
            self.stats["sn_basis_exchange_fallback:no_new_donor_term"] += 1
            return None
        try:
            choice = select_basis_exchange(
                parent_canonicals=tuple(term.canonical for term in parent_terms),
                parent_coefficients=tuple(term.coefficient for term in parent_terms),
                parent_novelties=parent_geometry["novelties"],
                parent_norms=parent_geometry["norms"],
                parent_signatures=parent_geometry["signatures"],
                parent_structural=tuple(
                    bool(term.basis.free_symbols) for term in parent_terms
                ),
                donor_canonicals=tuple(term.canonical for term in donor_terms),
                donor_coefficients=tuple(term.coefficient for term in donor_terms),
                donor_norms=donor_geometry["norms"],
                donor_signatures=donor_geometry["signatures"],
                donor_structural=tuple(
                    bool(term.basis.free_symbols) for term in donor_terms
                ),
                tau=self.sobolev_tau,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats["sn_basis_exchange_fallbacks"] += 1
            self.stats[f"sn_basis_exchange_fallback:{type(error).__name__}"] += 1
            return None
        if choice is None:
            self.stats["sn_basis_exchange_fallbacks"] += 1
            self.stats["sn_basis_exchange_fallback:no_exchange_choice"] += 1
            return None
        retained = [
            term
            for index, term in enumerate(parent_terms)
            if index != choice.removed_parent_index and term.basis.free_symbols
        ]
        selected_donor = donor_terms[choice.selected_donor_index]
        structural = retained + [selected_donor]
        genotype_expression = (
            structural[0].basis
            if len(structural) == 1
            else sp.Add(*(term.basis for term in structural), evaluate=False)
        )
        try:
            tree = nd.parse(to_project_expression_string(genotype_expression))
            if len(tree) > self.max_len:
                self.stats["sn_basis_exchange_fallbacks"] += 1
                self.stats["sn_basis_exchange_fallback:max_len"] += 1
                return None
            roundtrip_expression, roundtrip_symbols = parse_expression(
                tree.to_str(number_format=".17g"),
                self.feature_names,
            )
            actual_terms = tuple(
                term
                for term in decompose_expand_mul(
                    roundtrip_expression,
                    roundtrip_symbols,
                )
                if term.basis.free_symbols
            )
            expected = Counter(term.canonical for term in structural)
            actual = Counter(term.canonical for term in actual_terms)
            if actual != expected:
                self.stats["sn_basis_exchange_fallbacks"] += 1
                self.stats["sn_basis_exchange_fallback:roundtrip_basis_change"] += 1
                return None
        except (TypeError, ValueError, KeyError) as error:
            self.stats["sn_basis_exchange_fallbacks"] += 1
            self.stats[
                f"sn_basis_exchange_fallback:construction_{type(error).__name__}"
            ] += 1
            return None
        self.stats["sn_basis_exchange_proposals"] += 1
        self.stats[
            "sn_basis_exchange_removed_impact_sum"
        ] += choice.removed_deletion_impact
        self.stats["sn_basis_exchange_donor_gain_sum"] += choice.donor_residual_gain
        details = {
            "removed_parent_index": choice.removed_parent_index,
            "removed_parent_term": parent_terms[choice.removed_parent_index].display,
            "removed_deletion_impact": choice.removed_deletion_impact,
            "removed_novelty": choice.removed_novelty,
            "selected_donor_index": choice.selected_donor_index,
            "selected_donor_term": selected_donor.display,
            "donor_residual_gain": choice.donor_residual_gain,
            "donor_amplitude": choice.donor_amplitude,
            "donor_score": choice.donor_score,
            "proposal_expression": tree.to_str(number_format=".17g"),
        }
        return tree, choice, details

    def _basis_exchange_geometry(
        self,
        candidate: ControlledIndividual,
        X: np.ndarray,
        role: str,
    ) -> dict[str, Any] | None:
        self.stats["sn_basis_exchange_geometry_requests"] += 1
        fit = candidate.fit_result
        if fit is None or not fit.success or not fit.terms:
            self.stats[f"sn_basis_exchange_geometry_failure:{role}_missing_fit"] += 1
            return None
        analysis = candidate.sobolev_result
        if analysis is None:
            self._raise_if_deadline()
            analysis = self._evaluate_sobolev(candidate, X)
            candidate.sobolev_result = analysis
            candidate.sobolev_success = analysis.success
            self.stats["sn_basis_exchange_evaluator_calls"] += 1
        else:
            self.stats["sn_basis_exchange_analysis_reuse"] += 1
        if not analysis.success or analysis.candidate_geometry_key is None:
            self.stats[f"sn_basis_exchange_geometry_failure:{role}_evaluator"] += 1
            return None
        state = self.geometry_cache.peek_by_digest(analysis.candidate_geometry_key)
        if state is None or not state.success or not np.all(state.shared_valid_mask):
            self.stats[f"sn_basis_exchange_geometry_failure:{role}_mask"] += 1
            return None
        canonical_order = sorted(
            range(len(fit.terms)),
            key=lambda index: (fit.terms[index].canonical, index),
        )
        if tuple(fit.terms[index].canonical for index in canonical_order) != (
            state.key.canonical_terms
        ):
            self.stats[f"sn_basis_exchange_geometry_failure:{role}_alignment"] += 1
            return None
        if len(analysis.term_novelties) != len(fit.terms) or len(
            analysis.term_norms
        ) != len(fit.terms):
            self.stats[f"sn_basis_exchange_geometry_failure:{role}_diagnostics"] += 1
            return None
        signatures = np.empty_like(state.signatures)
        for canonical_index, original_index in enumerate(canonical_order):
            signatures[:, original_index] = state.signatures[:, canonical_index]
        self.stats["sn_basis_exchange_geometry_successes"] += 1
        return {
            "terms": fit.terms,
            "novelties": tuple(float(value) for value in analysis.term_novelties),
            "norms": tuple(float(value) for value in analysis.term_norms),
            "signatures": signatures,
        }

    def _verified_coverage_crossover(
        self,
        *,
        parent: ControlledIndividual,
        original_donor: ControlledIndividual,
        original_tree: nd.Symbol,
        original_method: str,
        generation: int,
        slot: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ControlledIndividual:
        """Base-verify one coverage-donor alternative against the original child."""

        coverage_donor = self.current_coverage_candidate
        assert coverage_donor is not None
        original = self._new_individual(
            original_tree,
            generation,
            slot,
            (parent, original_donor),
            original_method,
        )
        self._evaluate_individual(original, X, y)
        self.stats["sn_coverage_crossover_proposals"] += 1
        try:
            alternative_proposal = super().crossover(
                parent,
                coverage_donor,
                rng=self.counter_rng.generator(
                    generation,
                    slot,
                    "crossover",
                    attempt=2,
                ),
            )
        except Exception as error:
            self.stats["sn_coverage_crossover_alternative_errors"] += 1
            self.failure_counts[
                f"coverage_crossover_alternative:{type(error).__name__}"
            ] += 1
            return original
        alternative_tree = alternative_proposal.eqtree
        if len(alternative_tree) > self.max_len:
            self.stats["sn_coverage_crossover_alternative_bloat"] += 1
            return original
        alternative = self._new_individual(
            alternative_tree,
            generation,
            slot,
            (parent, coverage_donor),
            "coverage_crossover_alternative",
        )
        self._evaluate_individual(alternative, X, y)
        self.stats["sn_coverage_crossover_alternatives_evaluated"] += 1
        child_geometry: dict[str, Any] | None = None
        if self.use_sn_child_geometry:
            chosen, child_geometry = self._select_actual_child_geometry(
                alternative=alternative,
                original=original,
                X=X,
            )
            accepted = chosen is alternative
            if accepted:
                alternative.variation_type = (
                    "coverage_crossover_child_geometry"
                    if child_geometry["decision_stage"] == "sobolev_geometry"
                    else "coverage_crossover_verified"
                )
                self.stats["sn_child_geometry_alternative_selected"] += 1
            else:
                self.stats["sn_coverage_crossover_original_kept"] += 1
            reward_delta = (
                float(alternative.base_reward) - float(original.base_reward)
                if np.isfinite(alternative.base_reward)
                and np.isfinite(original.base_reward)
                else None
            )
            if accepted and reward_delta is not None:
                self.stats[
                    "sn_child_geometry_selected_reward_delta_sum"
                ] += reward_delta
        else:
            accepted = self._verified_proposal_better(alternative, original)
            if accepted:
                alternative.variation_type = "coverage_crossover_verified"
                self.stats["sn_coverage_crossover_alternative_base_wins"] += 1
                reward_delta = (
                    float(alternative.base_reward) - float(original.base_reward)
                    if np.isfinite(alternative.base_reward)
                    and np.isfinite(original.base_reward)
                    else 0.0
                )
                self.stats["sn_coverage_crossover_reward_gain_sum"] += reward_delta
                if np.isfinite(alternative.complexity) and np.isfinite(
                    original.complexity
                ):
                    self.stats["sn_coverage_crossover_complexity_saved_sum"] += float(
                        original.complexity
                    ) - float(alternative.complexity)
                chosen = alternative
            else:
                self.stats["sn_coverage_crossover_original_kept"] += 1
                reward_delta = (
                    float(alternative.base_reward) - float(original.base_reward)
                    if np.isfinite(alternative.base_reward)
                    and np.isfinite(original.base_reward)
                    else None
                )
                chosen = original
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "coverage_crossover_trace.jsonl",
                {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": parent.candidate_id,
                    "original_donor_id": original_donor.candidate_id,
                    "coverage_donor_id": coverage_donor.candidate_id,
                    "coverage_gain": self.current_coverage_gain,
                    "original": self._candidate_summary(original),
                    "alternative": self._candidate_summary(alternative),
                    "alternative_reward_delta": reward_delta,
                    "alternative_accepted": accepted,
                    "actual_child_geometry": child_geometry,
                },
            )
        return chosen

    def _select_actual_child_geometry(
        self,
        *,
        alternative: ControlledIndividual,
        original: ControlledIndividual,
        X: np.ndarray,
    ) -> tuple[ControlledIndividual, dict[str, Any]]:
        """Use actual fitted-child geometry only inside a Base-equivalent band."""

        self.stats["sn_child_geometry_events"] += 1
        strict_alternative = self._verified_proposal_better(alternative, original)
        strict_winner = alternative if strict_alternative else original
        details: dict[str, Any] = {
            "decision_stage": "base_fallback",
            "base_reward_gap": None,
            "epsilon_threshold": None,
            "original_gain": None,
            "alternative_gain": None,
            "strict_base_winner": ("alternative" if strict_alternative else "original"),
            "fallback_reason": None,
        }
        if alternative.valid != original.valid or not alternative.valid:
            details["decision_stage"] = "validity"
            self.stats["sn_child_geometry_base_direct"] += 1
            return strict_winner, details
        alternative_reward = float(alternative.base_reward)
        original_reward = float(original.base_reward)
        gap = abs(alternative_reward - original_reward)
        threshold = self.sn_compare_epsilon_abs + self.sn_compare_epsilon_rel * max(
            abs(alternative_reward), abs(original_reward)
        )
        details["base_reward_gap"] = gap
        details["epsilon_threshold"] = threshold
        if gap > threshold:
            details["decision_stage"] = "base_reward"
            self.stats["sn_child_geometry_base_direct"] += 1
            return strict_winner, details

        self.stats["sn_child_geometry_epsilon_equivalent"] += 1
        basis = self.current_coverage_anchor_basis
        if basis is None:
            details["fallback_reason"] = "anchor_basis_unavailable"
            self.stats["sn_child_geometry_fallbacks"] += 1
            self.stats["sn_child_geometry_fallback:anchor_basis_unavailable"] += 1
            return strict_winner, details
        self.stats["sn_child_geometry_vector_requests"] += 2
        original_vector = self._coverage_vector(
            original,
            X,
            enforce_deadline=True,
        )
        alternative_vector = self._coverage_vector(
            alternative,
            X,
            enforce_deadline=True,
        )
        if original_vector is None or alternative_vector is None:
            details["fallback_reason"] = "child_geometry_unavailable"
            self.stats["sn_child_geometry_fallbacks"] += 1
            self.stats["sn_child_geometry_fallback:child_geometry_unavailable"] += 1
            return strict_winner, details
        try:
            original_gain = signature_residual_gain(basis, original_vector)
            alternative_gain = signature_residual_gain(basis, alternative_vector)
        except (ValueError, np.linalg.LinAlgError) as error:
            details["fallback_reason"] = type(error).__name__
            self.stats["sn_child_geometry_fallbacks"] += 1
            self.stats[f"sn_child_geometry_fallback:{type(error).__name__}"] += 1
            return strict_winner, details
        details["original_gain"] = original_gain
        details["alternative_gain"] = alternative_gain
        self.stats["sn_child_geometry_geometry_success"] += 1
        if alternative_gain != original_gain:
            details["decision_stage"] = "sobolev_geometry"
            chosen = alternative if alternative_gain > original_gain else original
            self.stats["sn_child_geometry_sobolev_decisions"] += 1
        else:
            original_key = (
                float(original.complexity),
                original.canonical_fitted_expression,
                int(original.candidate_id),
            )
            alternative_key = (
                float(alternative.complexity),
                alternative.canonical_fitted_expression,
                int(alternative.candidate_id),
            )
            details["decision_stage"] = (
                "complexity"
                if original.complexity != alternative.complexity
                else "canonical"
            )
            chosen = alternative if alternative_key < original_key else original
            self.stats["sn_child_geometry_deterministic_ties"] += 1
        self.stats["sn_child_geometry_flips_vs_strict_base"] += int(
            chosen is not strict_winner
        )
        self.stats[
            (
                "sn_child_geometry_winner:alternative"
                if chosen is alternative
                else "sn_child_geometry_winner:original"
            )
        ] += 1
        return chosen, details

    @staticmethod
    def _verified_proposal_better(
        alternative: ControlledIndividual,
        original: ControlledIndividual,
    ) -> bool:
        if alternative.valid != original.valid:
            return alternative.valid
        if not alternative.valid:
            return False
        alternative_reward = float(alternative.base_reward)
        original_reward = float(original.base_reward)
        if alternative_reward != original_reward:
            return alternative_reward > original_reward
        return float(alternative.complexity) < float(original.complexity)

    def _sn_v2_mutation(
        self,
        method: str,
        parent: ControlledIndividual,
        rng: np.random.Generator,
        generation: int,
        slot: int,
    ) -> tuple[ControlledIndividual, MutationSiteSelection]:
        """Run the original mutation operator with only its AST site redirected."""

        assert self.sn_mutation_selector is not None
        self.sn_mutation_stats["mutation_parents_total"] += 1
        self.sn_mutation_stats[f"mutation_operator:{method}"] += 1
        child = parent.copy()
        rng_event = {
            "subtree-mutation": "subtree_mutation",
            "hoist-mutation": "hoist_mutation",
            "point-mutation": "point_mutation",
        }[method]
        site_rng = self.counter_rng.generator(
            generation,
            slot,
            rng_event,
            attempt=1,
        )

        if method == "subtree-mutation":
            original_node = super().get_random_subtree(child, rng=rng)
            original_path = child.eqtree.path_to(original_node)
            selection = self._choose_sn_mutation_sites(
                parent,
                (original_path,),
                1,
                tuple(
                    child.eqtree.path_to(node) for node in child.eqtree.iter_preorder()
                ),
                site_rng,
            )
            selected_node = child.eqtree.get_path(selection.selected_node_path)
            if selected_node.nettype != original_node.nettype:
                selection = self._mutation_selection_fallback(
                    selection,
                    "target_nettype_mismatch",
                )
                selected_node = child.eqtree.get_path(original_path)
            replacement = self.generator.generate_eqtree(
                nettype=original_node.nettype,
                rng=rng,
            )
            child.eqtree = child.eqtree.replace(selected_node, replacement)
            return child, selection

        if method == "hoist-mutation":
            original_node = super().get_random_subtree(
                child,
                nettype=self.nettype,
                rng=rng,
            )
            original_path = child.eqtree.path_to(original_node)
            eligible = tuple(
                child.eqtree.path_to(node)
                for node in child.eqtree.iter_preorder()
                if node.nettype in {self.nettype, "scalar"}
            )
            selection = self._choose_sn_mutation_sites(
                parent,
                (original_path,),
                1,
                eligible,
                site_rng,
            )
            selected_node = child.eqtree.get_path(selection.selected_node_path)
            child.eqtree = child.eqtree.replace(child.eqtree, selected_node)
            return child, selection

        if method == "point-mutation":
            original_paths = tuple(
                child.eqtree.path_to(node)
                for node in child.eqtree.iter_postorder()
                if rng.random() < self.p_point_replace
            )
            eligible = tuple(
                child.eqtree.path_to(node) for node in child.eqtree.iter_postorder()
            )
            selection = self._choose_sn_mutation_sites(
                parent,
                original_paths,
                len(original_paths),
                eligible,
                site_rng,
            )
            for path in selection.selected_node_paths:
                node = child.eqtree.get_path(path)
                if node.n_operands == 0:
                    symbol = self.generator.generate_leaf(nettype=node.nettype, rng=rng)
                elif node.n_operands == 1:
                    child_types = [operand.nettype for operand in node.operands]
                    choices = [
                        symbol
                        for symbol in self.unary
                        if symbol.map_nettype(child_types) in node.replaceable_nettype()
                    ]
                    symbol = rng.choice(choices)(*node.operands)
                elif node.n_operands == 2:
                    child_types = [operand.nettype for operand in node.operands]
                    choices = [
                        symbol
                        for symbol in self.binary
                        if symbol.map_nettype(child_types) in node.replaceable_nettype()
                    ]
                    symbol = rng.choice(choices)(*node.operands)
                else:
                    raise ValueError(f"Unknown arity: {node.n_operands}")
                child.eqtree = child.eqtree.replace(node, symbol)
            return child, selection

        raise ValueError(f"Unknown v2 mutation method {method}")

    def _choose_sn_mutation_sites(
        self,
        parent: ControlledIndividual,
        original_paths: tuple[tuple[int, ...], ...],
        selection_count: int,
        eligible_paths: tuple[tuple[int, ...], ...],
        rng: np.random.Generator,
    ) -> MutationSiteSelection:
        assert self.sn_mutation_selector is not None
        identity = (parent.raw_expression, parent.canonical_fitted_expression)
        cached = parent.sobolev_evaluated and parent.sn_cache_identity == identity
        self.sn_mutation_stats[
            (
                "mutation_parent_cache_hits"
                if cached
                else "mutation_parent_new_evaluations"
            )
        ] += 1
        view = self._lazy_sobolev_view(parent, "mutation")
        if view.success and parent.sobolev_result is not None:
            provenance = self._ensure_term_provenance(parent)
            novelties = parent.sobolev_result.term_novelties
            coefficients = parent.sobolev_result.coefficients
            term_norms = parent.sobolev_result.term_norms
        else:
            provenance = ProvenanceResult(
                success=False,
                raw_expression=parent.raw_expression,
                failure_reason=(
                    "sobolev_evaluation_failure:" + (view.failure_type or "unknown")
                ),
            )
            novelties = ()
            coefficients = ()
            term_norms = ()
        started = time.perf_counter()
        selection = self.sn_mutation_selector.select_sites(
            individual=parent,
            term_novelties=novelties,
            provenance=provenance,
            rng=rng,
            original_node_paths=original_paths,
            selection_count=selection_count,
            eligible_paths=eligible_paths,
            term_coefficients=coefficients,
            term_norms=term_norms,
        )
        self.sn_mutation_stats["selector_time_seconds"] += time.perf_counter() - started
        if selection.targeted:
            self.sn_mutation_stats["targeted_mutations"] += 1
            self.sn_mutation_stats["targeted_selected_sites"] += len(
                selection.selected_node_paths
            )
            self.sn_mutation_stats["selected_low_novelty_term_hits"] += len(
                selection.selected_low_novelty_term_ids
            )
            self.sn_mutation_stats["targeted_mutations_hitting_low_term"] += int(
                bool(selection.selected_low_novelty_term_ids)
            )
            self.sn_mutation_stats["selected_impact_eligible_term_hits"] += len(
                selection.selected_eligible_term_ids
            )
            self.sn_mutation_stats[
                "targeted_mutations_hitting_impact_eligible_term"
            ] += int(bool(selection.selected_eligible_term_ids))
        else:
            self.sn_mutation_stats["fallback_random_mutations"] += 1
            self.sn_mutation_stats[
                f"fallback:{selection.fallback_reason or 'unknown'}"
            ] += 1
        return selection

    @staticmethod
    def _mutation_selection_fallback(
        selection: MutationSiteSelection,
        reason: str,
    ) -> MutationSiteSelection:
        return replace(
            selection,
            selected_node_paths=selection.original_node_paths,
            targeted=False,
            fallback_reason=reason,
            selected_low_novelty_term_ids=(),
            selected_eligible_term_ids=(),
        )

    def _queue_mutation_event(
        self,
        method: str,
        parent: ControlledIndividual,
        child: ControlledIndividual,
        selection: MutationSiteSelection,
    ) -> None:
        parent_analysis = parent.sobolev_result
        parent_mean = None if parent_analysis is None else parent_analysis.mean_novelty
        parent_min = None if parent_analysis is None else parent_analysis.min_novelty
        parent_base = float(parent.base_reward)
        offspring_base = float(child.base_reward)
        parent_fit = parent.fit_result
        offspring_fit = child.fit_result
        self.sn_mutation_stats["offspring_base_reward_delta_sum"] += (
            offspring_base - parent_base
        )
        self.sn_mutation_stats["offspring_base_reward_improvements"] += int(
            offspring_base > parent_base
        )
        if selection.targeted:
            self.sn_mutation_stats["targeted_offspring_base_reward_delta_sum"] += (
                offspring_base - parent_base
            )
            self.sn_mutation_stats[
                "targeted_offspring_base_reward_improvements"
            ] += int(offspring_base > parent_base)
        if parent_mean is not None:
            self.sn_mutation_stats["parent_novelty_observations"] += 1
            self.sn_mutation_stats["parent_mean_novelty_sum"] += float(parent_mean)
            self.sn_mutation_stats["parent_min_novelty_sum"] += float(parent_min)
            if selection.targeted:
                self.sn_mutation_stats["targeted_parent_novelty_observations"] += 1
                self.sn_mutation_stats["targeted_parent_mean_novelty_sum"] += float(
                    parent_mean
                )
                self.sn_mutation_stats["targeted_parent_min_novelty_sum"] += float(
                    parent_min
                )
        self._pending_mutation_events.append(
            {
                "generation": child.generation,
                "parent_id": parent.candidate_id,
                "parent_expression": parent.raw_expression,
                "offspring_id": child.candidate_id,
                "offspring_expression": child.raw_expression,
                "selected_operator": method,
                "selection": selection,
                "parent_base_reward": self._finite(parent_base),
                "offspring_base_reward": self._finite(offspring_base),
                "parent_internal_r2": self._finite(
                    None if parent_fit is None else parent_fit.r2
                ),
                "offspring_internal_r2": self._finite(
                    None if offspring_fit is None else offspring_fit.r2
                ),
                "parent_complexity": (
                    None
                    if parent_fit is None or not parent_fit.success
                    else parent_fit.complexity
                ),
                "offspring_complexity": (
                    None
                    if offspring_fit is None or not offspring_fit.success
                    else offspring_fit.complexity
                ),
                "parent_mean_novelty": self._finite(parent_mean),
                "parent_min_novelty": self._finite(parent_min),
                "targeted": bool(selection.targeted),
            }
        )

    def _finalize_mutation_events(
        self, children: Sequence[ControlledIndividual]
    ) -> None:
        if not self._pending_mutation_events:
            return
        child_by_id = {value.candidate_id: value for value in children}
        elite_ids = {value.candidate_id for value in self.current_elite_order}
        for event in self._pending_mutation_events:
            child = child_by_id[event["offspring_id"]]
            analysis = child.sobolev_result
            event["offspring_elite_survival"] = child.candidate_id in elite_ids
            event["offspring_mean_novelty"] = self._finite(
                None if analysis is None else analysis.mean_novelty
            )
            event["offspring_min_novelty"] = self._finite(
                None if analysis is None else analysis.min_novelty
            )
            self.sn_mutation_stats["offspring_elite_survivals"] += int(
                event["offspring_elite_survival"]
            )
            if event["targeted"]:
                self.sn_mutation_stats["targeted_offspring_elite_survivals"] += int(
                    event["offspring_elite_survival"]
                )
            if analysis is not None and analysis.success:
                self.sn_mutation_stats["offspring_novelty_observations"] += 1
                parent_mean = event["parent_mean_novelty"]
                if parent_mean is not None and analysis.mean_novelty is not None:
                    self.sn_mutation_stats["offspring_mean_novelty_delta_sum"] += float(
                        analysis.mean_novelty
                    ) - float(parent_mean)
                    if event["targeted"]:
                        self.sn_mutation_stats[
                            "targeted_offspring_novelty_observations"
                        ] += 1
                        self.sn_mutation_stats[
                            "targeted_offspring_mean_novelty_delta_sum"
                        ] += float(analysis.mean_novelty) - float(parent_mean)
            selection = event.pop("selection")
            fallback_reason = selection.fallback_reason or ""
            traceworthy_fallback = (
                fallback_reason.startswith("sobolev_evaluation_failure")
                or fallback_reason.startswith("provenance_failure")
                or fallback_reason
                in {
                    "stale_sobolev_cache",
                    "stale_provenance_expression",
                    "invalid_term_novelties",
                    "no_eligible_mutable_site",
                    "low_novelty_terms_have_no_eligible_site",
                    "impact_eligible_terms_have_no_eligible_site",
                    "no_impact_eligible_low_novelty_terms",
                    "all_term_impacts_near_zero",
                    "missing_term_impact_inputs",
                    "nonfinite_safe_term_priorities",
                    "invalid_mutation_site",
                }
            )
            should_trace = (
                traceworthy_fallback
                or (
                    self.use_sn_gp_stable
                    and self.detailed_logging
                    and selection.targeted
                )
                or (
                    self.detailed_logging
                    and self.sn_trace_every > 0
                    and self.sn_mutation_stats["mutation_parents_total"]
                    % self.sn_trace_every
                    == 0
                )
            )
            if self.output_dir is not None and should_trace:
                payload = dict(event)
                payload.update(selection.as_dict())
                provenance = child_by_id[event["offspring_id"]]
                parent = next(
                    (
                        value
                        for value in self.population
                        if value.candidate_id == event["parent_id"]
                    ),
                    None,
                )
                payload["provenance"] = (
                    None
                    if parent is None or parent.term_provenance is None
                    else parent.term_provenance.as_dict()
                )
                append_jsonl(self.output_dir / "mutation_trace.jsonl", payload)
        self._pending_mutation_events = []

    def _evaluate_individual(
        self, individual: ControlledIndividual, X: np.ndarray, y: np.ndarray
    ) -> None:
        assert self.additive_evaluator is not None
        result = self.additive_evaluator.evaluate(individual.eqtree, X, y)
        individual.apply_fit(result)
        self.stats["candidate_evaluations"] += 1
        if result.success:
            self.stats["base_valid_candidates"] += 1
            individual.parent_geometry_key = self._closest_parent_geometry(individual)
        else:
            self.stats["base_invalid_candidates"] += 1
            self.failure_counts[f"base:{result.failure_type}"] += 1

    def _apply_structural_selection(
        self,
        population: list[ControlledIndividual],
        X: np.ndarray,
        y: np.ndarray,
        *,
        enforce_deadline: bool,
    ) -> None:
        base_order = self._rank_base(population)
        if self.record_reward_gap_samples:
            rewards = [
                float(candidate.base_reward)
                for candidate in base_order
                if candidate.valid and np.isfinite(candidate.base_reward)
            ]
            self.reward_gap_samples["base_rewards"].extend(rewards)
            if len(base_order) > self.elitism_k:
                upper = float(base_order[self.elitism_k - 1].base_reward)
                lower = float(base_order[self.elitism_k].base_reward)
                if np.isfinite(upper) and np.isfinite(lower):
                    self.reward_gap_samples["elite_boundary_gaps"].append(
                        abs(upper - lower)
                    )
                    self.reward_gap_samples["elite_boundary_scales"].append(
                        max(abs(upper), abs(lower))
                    )
        for rank, candidate in enumerate(base_order, start=1):
            candidate.base_rank = rank
            candidate.structural_rank = None
            candidate.final_reward = None
            candidate.sobolev_penalty = None
            candidate.sobolev_result = None
            candidate.sobolev_success = None
            candidate.shortlist_status = "not_considered"
        valid_order = [candidate for candidate in base_order if candidate.valid]
        if self.use_sn_gp_verified_repair:
            self._apply_verified_repair_selection(
                population,
                valid_order,
                X,
                y,
                enforce_deadline=enforce_deadline,
            )
            return
        if self.use_sn_gp_plugin:
            assert self.sn_comparator is not None
            base_elites = base_order[: self.elitism_k]
            sn_selected = self.sn_comparator.select_top(
                population,
                self.elitism_k,
                context="elite",
            )

            self.current_sn_elite_order = list(sn_selected)
            self.current_base_anchor = select_base_anchor(
                population, key=self._base_key
            )
            if self.sn_base_anchor_enabled:
                anchor_id = self.current_base_anchor.candidate_id
                sn_ids = {value.candidate_id for value in sn_selected}
                self.last_anchor_already_selected = anchor_id in sn_ids
                self.last_anchor_injected = anchor_id not in sn_ids
                selected = merge_anchor_with_sn_elites(
                    self.current_base_anchor,
                    sn_selected,
                    self.elitism_k,
                    identity=lambda value: value.candidate_id,
                )
                self.stats["sn_base_anchor_generations"] += 1
                self.stats["sn_base_anchor_already_selected"] += int(
                    self.last_anchor_already_selected
                )
                self.stats["sn_base_anchor_injections"] += int(
                    self.last_anchor_injected
                )
            else:
                selected = list(sn_selected)
                self.last_anchor_already_selected = False
                self.last_anchor_injected = False
            self.current_structural_order = list(selected)
            self.current_elite_order = list(selected)
            for rank, candidate in enumerate(selected, start=1):
                candidate.structural_rank = rank
            selected_ids = {candidate.candidate_id for candidate in selected}
            base_ids = {candidate.candidate_id for candidate in base_elites}
            changed = len(selected_ids.difference(base_ids))
            self.stats["sn_v2_elite_membership_changes"] += changed
            self.stats["sn_v2_elite_changed_generations"] += int(changed > 0)
            if self.use_sn_gp_stable:
                self.stats["sn_stable_elite_membership_changes"] += changed
                self.stats["sn_stable_elite_changed_generations"] += int(changed > 0)
            label = "sn_stable" if self.use_sn_gp_stable else "sn_v2"
            for candidate in population:
                candidate.shortlist_status = (
                    f"{label}_elite"
                    if candidate.candidate_id in selected_ids
                    else (
                        f"{label}_lazy_evaluated"
                        if candidate.sobolev_evaluated
                        else (
                            "base_invalid"
                            if not candidate.valid
                            else f"{label}_not_evaluated"
                        )
                    )
                )
            if self.output_dir is not None and self.detailed_logging:
                append_jsonl(
                    self.output_dir / f"{label}_elite_generations.jsonl",
                    {
                        "generation": population[0].generation if population else None,
                        "base_elite_ids": [value.candidate_id for value in base_elites],
                        "sn_comparator_elite_ids": [
                            value.candidate_id for value in sn_selected
                        ],
                        "final_elite_ids": [value.candidate_id for value in selected],
                        "base_anchor_id": self.current_base_anchor.candidate_id,
                        "base_anchor_injected": self.last_anchor_injected,
                        "membership_changes": changed,
                    },
                )
            return
        if (
            self.use_sn_repair_shadow
            or self.use_sn_archive_export
            or self.use_sn_archive_anchor_export
            or self.use_sn_archive_beam_export
            or not self.use_sobolev
        ):
            self.current_base_anchor = base_order[0]
            self.current_sn_elite_order = []
            for candidate in valid_order:
                candidate.final_reward = candidate.base_reward
            self.current_structural_order = valid_order
            self.current_elite_order = valid_order[: self.elitism_k]
            if len(self.current_elite_order) < self.elitism_k:
                self.current_elite_order.extend(
                    candidate
                    for candidate in base_order
                    if candidate not in self.current_elite_order
                )
                self.current_elite_order = self.current_elite_order[: self.elitism_k]
            if (
                self.use_sn_archive_export
                or self.use_sn_archive_anchor_export
                or self.use_sn_archive_beam_export
            ):
                self._update_basis_archive(base_order, X)
            return

        shortlist = valid_order[: self.shortlist_size]
        shortlist_ids = {candidate.candidate_id for candidate in shortlist}
        for candidate in base_order:
            candidate.shortlist_status = (
                "shortlisted"
                if candidate.candidate_id in shortlist_ids
                else ("base_invalid" if not candidate.valid else "base_rejected")
            )
        self.stats["shortlist_occurrences"] += len(shortlist)
        groups: dict[str, list[ControlledIndividual]] = {}
        for candidate in shortlist:
            groups.setdefault(candidate.canonical_fitted_expression, []).append(
                candidate
            )
        self.stats["shortlist_unique_expressions"] += len(groups)
        for occurrences in groups.values():
            if enforce_deadline:
                self._raise_if_deadline()
            representative = occurrences[0]
            analysis = self._evaluate_sobolev(representative, X)
            penalty = (
                sobolev_penalty(analysis.term_novelties, self.sobolev_tau)
                if analysis.success
                else 1.0
            )
            for candidate in occurrences:
                candidate.sobolev_result = copy.deepcopy(analysis)
                candidate.sobolev_result.raw_expression = candidate.raw_expression
                candidate.sobolev_penalty = penalty
                candidate.sobolev_success = analysis.success
                candidate.final_reward = (
                    candidate.base_reward - self.sobolev_alpha * penalty
                )
            if enforce_deadline:
                self._raise_if_deadline()
        structural_order = self._rank_structural(shortlist)
        if (
            structural_order
            and self.sobolev_pruning
            and self.sobolev_max_prunes > 0
            and self.prune_elite_k > 0
        ):
            original = structural_order[0]
            repaired = self._prune_candidate(original, X, y)
            if repaired is not original:
                index = population.index(original)
                population[index] = repaired
                shortlist[shortlist.index(original)] = repaired
                structural_order = self._rank_structural(shortlist)
        for rank, candidate in enumerate(structural_order, start=1):
            candidate.structural_rank = rank
        self.current_structural_order = structural_order
        elites = list(structural_order[: self.elitism_k])
        if len(elites) < self.elitism_k:
            elites.extend(
                candidate for candidate in base_order if candidate not in elites
            )
        self.current_elite_order = elites[: self.elitism_k]
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "structural_generations.jsonl",
                {
                    "generation": population[0].generation if population else None,
                    "shortlist_occurrences": len(shortlist),
                    "shortlist_unique_expressions": len(groups),
                    "candidates": [
                        self._candidate_summary(value) for value in structural_order
                    ],
                },
            )

    def _apply_verified_repair_selection(
        self,
        population: list[ControlledIndividual],
        valid_order: Sequence[ControlledIndividual],
        X: np.ndarray,
        y: np.ndarray,
        *,
        enforce_deadline: bool,
    ) -> None:
        """Try one Base-safe Sobolev term repair, then retain Base selection."""

        representatives: list[ControlledIndividual] = []
        seen: set[str] = set()
        for candidate in valid_order:
            identity = candidate.canonical_fitted_expression
            if identity in seen:
                continue
            seen.add(identity)
            representatives.append(candidate)
            if len(representatives) >= self.sn_repair_shortlist_size:
                break
        self.stats["sn_repair_shortlist_candidates"] += len(representatives)
        accepted = 0
        for candidate in representatives:
            if enforce_deadline:
                self._raise_if_deadline()
            self.stats["sn_repair_candidates_considered"] += 1
            fit = candidate.fit_result
            assert fit is not None
            if len(fit.terms) <= 1:
                candidate.shortlist_status = "repair_not_applicable"
                self.stats["sn_repair_single_or_zero_term_skips"] += 1
                continue
            analysis = self._evaluate_sobolev(candidate, X)
            candidate.sobolev_result = analysis
            candidate.sobolev_success = analysis.success
            candidate.shortlist_status = "repair_evaluated"
            if not analysis.success:
                self.stats["sn_repair_evaluator_failures"] += 1
                continue
            candidate.sobolev_penalty = sobolev_penalty(
                analysis.term_novelties, self.sobolev_tau
            )
            excluded = tuple(
                index
                for index, term in enumerate(fit.terms)
                if not term.basis.free_symbols
            )
            self.stats["sn_repair_attempts"] += 1
            repaired = self._prune_candidate(
                candidate,
                X,
                y,
                excluded_prune_indices=excluded,
            )
            if repaired is candidate:
                self.stats["sn_repair_not_accepted"] += 1
                continue
            index = population.index(candidate)
            population[index] = repaired
            repaired.final_reward = repaired.base_reward
            repaired.shortlist_status = "verified_repair_accepted"
            self.stats["sn_repair_accepted"] += 1
            self.stats["sn_repair_complexity_saved"] += max(
                0, int(candidate.complexity) - int(repaired.complexity)
            )
            accepted += 1
            if accepted >= self.sn_repair_max_accepted_per_generation:
                break
        base_order = self._rank_base(population)
        for rank, candidate in enumerate(base_order, start=1):
            candidate.base_rank = rank
            candidate.structural_rank = rank
            candidate.final_reward = candidate.base_reward
            if candidate.shortlist_status == "not_considered":
                candidate.shortlist_status = "base_only"
        self.current_base_anchor = base_order[0]
        self.current_sn_elite_order = []
        self.current_structural_order = base_order
        if self.use_sn_population_coverage:
            self._apply_population_coverage(
                base_order,
                X,
                enforce_deadline=enforce_deadline,
            )
        else:
            self.current_elite_order = base_order[: self.elitism_k]
            self.current_coverage_candidate = None
            self.current_coverage_gain = None
            self.current_coverage_base_rank = None
            self.current_coverage_slot_changed = False
            self.current_coverage_anchor_basis = None
        if self.use_sn_basis_archive:
            self._update_basis_archive(base_order, X)
        self.last_anchor_injected = False
        self.last_anchor_already_selected = False

    def _apply_population_coverage(
        self,
        base_order: Sequence[ControlledIndividual],
        X: np.ndarray,
        *,
        enforce_deadline: bool,
    ) -> None:
        """Use one survivor slot for a Base-qualified new Sobolev direction."""

        base_elites = list(base_order[: self.elitism_k])
        anchor_count = self.elitism_k - self.sn_coverage_elite_slots
        anchors = base_elites[:anchor_count]
        self.stats["sn_coverage_generations"] += 1
        representatives: list[ControlledIndividual] = []
        seen: set[str] = set()
        for candidate in base_order[: self.sn_coverage_shortlist_size]:
            if not candidate.valid:
                continue
            identity = candidate.canonical_fitted_expression
            if identity in seen:
                continue
            seen.add(identity)
            representatives.append(candidate)
        self.stats["sn_coverage_unique_shortlist_total"] += len(representatives)
        anchor_ids = {candidate.candidate_id for candidate in anchors}
        boundary_reward = float(base_elites[-1].base_reward)
        candidates: list[ControlledIndividual] = []
        for candidate in representatives:
            if candidate.candidate_id in anchor_ids:
                continue
            reward_gap = max(0.0, boundary_reward - float(candidate.base_reward))
            if reward_gap > self.sn_coverage_max_base_reward_gap:
                self.stats["sn_coverage_candidates_rejected_by_base_gap"] += 1
                continue
            candidates.append(candidate)

        anchor_vectors: list[np.ndarray] = []
        for candidate in anchors:
            vector = self._coverage_vector(
                candidate,
                X,
                enforce_deadline=enforce_deadline,
            )
            if vector is None:
                self._fallback_base_coverage(base_elites, "anchor_geometry_unavailable")
                return
            anchor_vectors.append(vector)
        try:
            self.current_coverage_anchor_basis = orthonormal_signature_span(
                anchor_vectors
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.failure_counts[f"coverage_anchor:{type(error).__name__}"] += 1
            self._fallback_base_coverage(base_elites, "anchor_linear_algebra")
            return

        candidate_vectors: list[np.ndarray] = []
        eligible: list[ControlledIndividual] = []
        for candidate in candidates:
            vector = self._coverage_vector(
                candidate,
                X,
                enforce_deadline=enforce_deadline,
            )
            if vector is None:
                continue
            candidate_vectors.append(vector)
            eligible.append(candidate)
        self.stats["sn_coverage_eligible_candidates"] += len(eligible)
        if not eligible:
            self._fallback_base_coverage(
                base_elites,
                "no_base_near_candidate" if not candidates else "no_candidate_geometry",
            )
            return
        try:
            choice = select_coverage_candidate(
                anchor_vectors,
                candidate_vectors,
                [int(candidate.base_rank) for candidate in eligible],
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.failure_counts[f"coverage:{type(error).__name__}"] += 1
            self._fallback_base_coverage(base_elites, "coverage_linear_algebra")
            return
        selected = eligible[choice.selected_index]
        self.current_elite_order = anchors + [selected]
        self.current_coverage_candidate = selected
        self.current_coverage_gain = float(choice.selected_gain)
        self.current_coverage_base_rank = int(selected.base_rank)
        changed = selected.candidate_id != base_elites[-1].candidate_id
        self.current_coverage_slot_changed = bool(changed)
        self.stats["sn_coverage_selections"] += 1
        self.stats["sn_coverage_slot_changes"] += int(changed)
        self.stats["sn_coverage_selected_base_rank_sum"] += int(selected.base_rank)
        self.stats["sn_coverage_gain_sum"] += float(choice.selected_gain)
        self.stats["sn_coverage_reference_rank_sum"] += int(choice.reference_rank)
        reward_gap = max(
            0.0,
            boundary_reward - float(selected.base_reward),
        )
        self.stats["sn_coverage_selected_reward_gap_sum"] += reward_gap
        self.stats["sn_coverage_selected_reward_gap_max"] = max(
            float(self.stats["sn_coverage_selected_reward_gap_max"]),
            reward_gap,
        )
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "coverage_trace.jsonl",
                {
                    "generation": selected.generation,
                    "base_elite_ids": [value.candidate_id for value in base_elites],
                    "anchor_ids": [value.candidate_id for value in anchors],
                    "selected_candidate_id": selected.candidate_id,
                    "selected_base_rank": int(selected.base_rank),
                    "selected_gain": float(choice.selected_gain),
                    "base_boundary_reward_gap": reward_gap,
                    "reference_rank": int(choice.reference_rank),
                    "slot_changed": bool(changed),
                    "fallback_reason": None,
                },
            )

    def _update_basis_archive(
        self,
        base_order: Sequence[ControlledIndividual],
        X: np.ndarray,
    ) -> None:
        """Add structural terms from Base-top unique phenotypes to the archive."""

        assert self.basis_archive is not None
        self.stats["sn_basis_archive_updates"] += 1
        representatives: list[ControlledIndividual] = []
        seen: set[str] = set()
        for candidate in base_order:
            if not candidate.valid:
                continue
            identity = candidate.canonical_fitted_expression
            if identity in seen:
                continue
            seen.add(identity)
            representatives.append(candidate)
            if len(representatives) >= self.sn_basis_archive_source_candidates:
                break
        self.stats["sn_basis_archive_source_candidates"] += len(representatives)
        additions: list[BasisArchiveEntry] = []
        for candidate in representatives:
            geometry = self._basis_exchange_geometry(candidate, X, "archive")
            if geometry is None:
                self.stats["sn_basis_archive_source_geometry_failures"] += 1
                continue
            fit = candidate.fit_result
            assert fit is not None
            for index, term in enumerate(geometry["terms"]):
                if not term.basis.free_symbols:
                    continue
                try:
                    values = np.asarray(
                        evaluate_sympy(term.basis, fit.symbols, X), dtype=float
                    ).reshape(-1)
                except (TypeError, ValueError, KeyError):
                    self.stats["sn_basis_archive_value_failures"] += 1
                    continue
                values[~np.isfinite(values)] = 0.0
                norm = float(geometry["norms"][index])
                coefficient = float(term.coefficient)
                additions.append(
                    BasisArchiveEntry(
                        canonical=term.canonical,
                        expression=to_project_expression_string(term.basis),
                        signature=np.asarray(
                            geometry["signatures"][:, index], dtype=float
                        ).copy(),
                        values=values.copy(),
                        term_norm=norm,
                        source_coefficient=coefficient,
                        source_amplitude=abs(coefficient) * norm,
                        source_base_reward=float(candidate.base_reward),
                        source_generation=int(candidate.generation),
                        source_base_rank=int(candidate.base_rank),
                        source_candidate_id=int(candidate.candidate_id),
                    )
                )
        self.stats["sn_basis_archive_terms_seen"] += len(additions)
        update = self.basis_archive.update(additions)
        self.stats["sn_basis_archive_insertions"] += update.inserted
        self.stats["sn_basis_archive_replacements"] += update.replaced
        self.stats["sn_basis_archive_evictions"] += update.evicted
        self.stats["sn_basis_archive_canonical_reuses"] += update.canonical_reuses
        self.stats["sn_basis_archive_size_sum"] += update.new_size
        self.stats["sn_basis_archive_size_max"] = max(
            int(self.stats["sn_basis_archive_size_max"]), update.new_size
        )
        if self.basis_quality_archive is not None:
            quality_update = self.basis_quality_archive.update(additions)
            self.stats["sn_basis_quality_archive_updates"] += 1
            self.stats["sn_basis_quality_archive_insertions"] += quality_update.inserted
            self.stats[
                "sn_basis_quality_archive_replacements"
            ] += quality_update.replaced
            self.stats["sn_basis_quality_archive_evictions"] += quality_update.evicted
            self.stats[
                "sn_basis_quality_archive_canonical_reuses"
            ] += quality_update.canonical_reuses
            self.stats["sn_basis_quality_archive_size_sum"] += quality_update.new_size
            self.stats["sn_basis_quality_archive_size_max"] = max(
                int(self.stats["sn_basis_quality_archive_size_max"]),
                quality_update.new_size,
            )
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "basis_archive_updates.jsonl",
                {
                    "generation": (
                        None if not base_order else int(base_order[0].generation)
                    ),
                    "previous_size": update.previous_size,
                    "new_size": update.new_size,
                    "inserted": update.inserted,
                    "replaced": update.replaced,
                    "evicted": update.evicted,
                    "canonical_reuses": update.canonical_reuses,
                    "entries": [
                        {
                            "canonical": entry.canonical,
                            "expression": entry.expression,
                            "source_generation": entry.source_generation,
                            "source_base_rank": entry.source_base_rank,
                            "source_candidate_id": entry.source_candidate_id,
                            "source_base_reward": entry.source_base_reward,
                            "source_amplitude": entry.source_amplitude,
                        }
                        for entry in self.basis_archive.entries
                    ],
                },
            )

    def _fallback_base_coverage(
        self,
        base_elites: Sequence[ControlledIndividual],
        reason: str,
    ) -> None:
        self.current_elite_order = list(base_elites)
        self.current_coverage_candidate = base_elites[-1]
        self.current_coverage_gain = None
        self.current_coverage_base_rank = int(base_elites[-1].base_rank)
        self.current_coverage_slot_changed = False
        preserve_anchor = bool(
            self.use_sn_stagnation_pursuit_shadow
            and self.sn_pursuit_preserve_fallback_anchor
            and self.current_coverage_anchor_basis is not None
            and reason in {"no_base_near_candidate", "no_candidate_geometry"}
        )
        if preserve_anchor:
            self.stats["sn_pursuit_stagnation_fallback_anchors_preserved"] += 1
        else:
            self.current_coverage_anchor_basis = None
        self.stats["sn_coverage_fallbacks"] += 1
        self.stats[f"sn_coverage_fallback:{reason}"] += 1

    def _coverage_vector(
        self,
        candidate: ControlledIndividual,
        X: np.ndarray,
        *,
        enforce_deadline: bool,
    ) -> np.ndarray | None:
        self.stats["sn_coverage_vector_requests"] += 1
        fit = candidate.fit_result
        if fit is None or not fit.success or len(fit.terms) == 0:
            self.stats["sn_coverage_vector_failures:missing_terms"] += 1
            return None
        analysis = candidate.sobolev_result
        if analysis is None:
            if enforce_deadline:
                self._raise_if_deadline()
            analysis = self._evaluate_sobolev(candidate, X)
            candidate.sobolev_result = analysis
            candidate.sobolev_success = analysis.success
            self.stats["sn_coverage_evaluator_calls"] += 1
        else:
            self.stats["sn_coverage_analysis_reuse"] += 1
        if not analysis.success or analysis.candidate_geometry_key is None:
            self.stats["sn_coverage_vector_failures:evaluator"] += 1
            return None
        if self.geometry_indices is None or analysis.valid_mask_size != len(
            self.geometry_indices
        ):
            self.stats["sn_coverage_vector_failures:partial_mask"] += 1
            return None
        state = self.geometry_cache.peek_by_digest(analysis.candidate_geometry_key)
        if state is None or not state.success or not np.all(state.shared_valid_mask):
            self.stats["sn_coverage_vector_failures:geometry_cache"] += 1
            return None
        canonical_order = sorted(
            range(len(fit.terms)),
            key=lambda index: (fit.terms[index].canonical, index),
        )
        canonical_terms = tuple(fit.terms[index].canonical for index in canonical_order)
        if canonical_terms != state.key.canonical_terms:
            self.stats["sn_coverage_vector_failures:term_alignment"] += 1
            return None
        coefficients = [fit.terms[index].coefficient for index in canonical_order]
        try:
            vector = normalized_fitted_signature(state.signatures, coefficients)
        except ValueError:
            self.stats["sn_coverage_vector_failures:numerical"] += 1
            return None
        self.stats["sn_coverage_vector_successes"] += 1
        return vector

    def _evaluate_sobolev(
        self, candidate: ControlledIndividual, X: np.ndarray
    ) -> EvaluationResult:
        assert self.sobolev_evaluator is not None
        assert candidate.fit_result is not None
        assert candidate.fit_result.fitted_expression is not None
        assert self.geometry_indices is not None
        result = self.sobolev_evaluator.evaluate_preparsed(
            raw_expression=candidate.raw_expression,
            fitted_expression=candidate.fit_result.fitted_expression_text,
            expression=candidate.fit_result.fitted_expression,
            symbols=candidate.fit_result.symbols,
            terms=candidate.fit_result.terms,
            X=X,
            dataset_identity=self.dataset_identity,
            geometry_indices=self.geometry_indices,
            parent_geometry_key=candidate.parent_geometry_key,
        )
        self.stats["sobolev_unique_evaluations"] += 1
        if result.success:
            self.stats["sobolev_successes"] += 1
        else:
            self.stats["sobolev_failures"] += 1
            self.failure_counts[f"sobolev:{result.failure_type.value}"] += 1
        self.stats[f"sobolev_algorithm:{result.algorithm_used}"] += 1
        self.stats["sobolev_incremental_hits"] += int(result.incremental_hit)
        self.stats["sobolev_geometry_cache_hits"] += int(result.geometry_cache_hit)
        return result

    def _lazy_sobolev_view(
        self, candidate: ControlledIndividual, context: str
    ) -> LazySobolevView:
        """Evaluate a v2 candidate at most once for one exact phenotype identity."""

        self.stats["sn_lazy_requests"] += 1
        self.stats[f"sn_lazy_requests:{context}"] += 1
        identity = (candidate.raw_expression, candidate.canonical_fitted_expression)
        if candidate.sobolev_evaluated and candidate.sn_cache_identity == identity:
            self.stats["sn_lazy_cache_hits"] += 1
            self.stats[f"sn_lazy_cache_hits:{context}"] += 1
            return LazySobolevView(
                success=bool(candidate.sobolev_success),
                penalty=candidate.sobolev_penalty,
                failure_type=candidate.sobolev_evaluation_failure,
            )
        if candidate.sobolev_evaluated:
            self.stats["sn_lazy_stale_cache_invalidations"] += 1
            candidate.sobolev_result = None
            candidate.sobolev_penalty = None
            candidate.sobolev_success = None
            candidate.sobolev_evaluation_failure = None
            candidate.term_provenance = None
            candidate.provenance_cache_expression = None
        self.stats["sn_lazy_evaluator_calls"] += 1
        self.stats[f"sn_lazy_evaluator_calls:{context}"] += 1
        started = time.perf_counter()
        failure: str | None = None
        try:
            if self._active_X is None:
                raise RuntimeError("training geometry is not active")
            analysis = self._evaluate_sobolev(candidate, self._active_X)
            candidate.sobolev_result = analysis
            candidate.sobolev_success = bool(analysis.success)
            if analysis.success:
                candidate.sobolev_penalty = sobolev_penalty(
                    analysis.term_novelties,
                    self.sobolev_tau,
                )
            else:
                candidate.sobolev_penalty = None
                failure = analysis.failure_type.value
        except Exception as error:
            candidate.sobolev_result = None
            candidate.sobolev_success = False
            candidate.sobolev_penalty = None
            failure = f"{type(error).__name__}: {error}"
            self.failure_counts[f"sn_lazy:{type(error).__name__}"] += 1
        candidate.sobolev_evaluated = True
        candidate.sn_cache_identity = identity
        candidate.sobolev_evaluation_failure = failure
        elapsed = time.perf_counter() - started
        self.stats["sn_lazy_evaluator_time_seconds"] += elapsed
        if failure is not None:
            self.stats["sn_lazy_failures"] += 1
            self.stats[f"sn_lazy_failures:{context}"] += 1
            if self.output_dir is not None:
                append_jsonl(
                    self.output_dir / "sobolev_failures.jsonl",
                    {
                        "candidate_id": candidate.candidate_id,
                        "generation": candidate.generation,
                        "expression": candidate.raw_expression,
                        "failure_type": failure,
                        "comparator_location": context,
                        "fallback": True,
                    },
                )
        return LazySobolevView(
            success=bool(candidate.sobolev_success),
            penalty=candidate.sobolev_penalty,
            failure_type=failure,
        )

    def _ensure_term_provenance(
        self, candidate: ControlledIndividual
    ) -> ProvenanceResult:
        expression = candidate.raw_expression
        if (
            candidate.term_provenance is not None
            and candidate.provenance_cache_expression == expression
        ):
            self.sn_mutation_stats["provenance_cache_hits"] += 1
            return candidate.term_provenance
        started = time.perf_counter()
        if self.term_provenance_builder is None or candidate.fit_result is None:
            result = ProvenanceResult(
                success=False,
                raw_expression=expression,
                failure_reason="provenance_disabled_or_missing_fit",
            )
        else:
            result = self.term_provenance_builder.build(
                candidate.eqtree,
                candidate.fit_result.terms,
            )
        self.sn_mutation_stats["provenance_builds"] += 1
        self.sn_mutation_stats["provenance_time_seconds"] += (
            time.perf_counter() - started
        )
        self.sn_mutation_stats[
            "provenance_successes" if result.success else "provenance_failures"
        ] += 1
        candidate.term_provenance = result
        candidate.provenance_cache_expression = expression
        if not result.success and self.output_dir is not None:
            append_jsonl(
                self.output_dir / "provenance_failures.jsonl",
                {
                    "candidate_id": candidate.candidate_id,
                    "generation": candidate.generation,
                    "expression": expression,
                    "terms": list(candidate.term_keys),
                    "provenance_status": "failed",
                    "failure_reason": result.failure_reason,
                    "fallback": True,
                },
            )
        return result

    def _record_comparison_trace(self, result: ComparisonResult) -> None:
        if self.output_dir is None or not self.detailed_logging:
            return
        total = (
            0
            if self.sn_comparator is None
            else self.sn_comparator.stats["comparisons_total"]
        )
        selected = result.fallback_reason is not None or (
            self.sn_trace_every > 0 and total % self.sn_trace_every == 0
        )
        if not selected:
            return
        payload = result.as_dict()
        payload.update(
            {
                "generation": result.winner.generation,
                "winner_expression": result.winner.raw_expression,
                "loser_expression": result.loser.raw_expression,
                "winner_base_reward": self._finite(result.winner.base_reward),
                "loser_base_reward": self._finite(result.loser.base_reward),
                "winner_complexity": (
                    None
                    if not np.isfinite(result.winner.complexity)
                    else int(result.winner.complexity)
                ),
                "loser_complexity": (
                    None
                    if not np.isfinite(result.loser.complexity)
                    else int(result.loser.complexity)
                ),
            }
        )
        append_jsonl(self.output_dir / "comparator_trace.jsonl", payload)

    def _prune_candidate(
        self,
        candidate: ControlledIndividual,
        X: np.ndarray,
        y: np.ndarray,
        *,
        excluded_prune_indices: tuple[int, ...] = (),
    ) -> ControlledIndividual:
        analysis = candidate.sobolev_result
        fit = candidate.fit_result
        if analysis is None or fit is None or not analysis.success:
            return candidate
        initial = RefitResult(
            expression=fit.fitted_expression,
            coefficients=[term.coefficient for term in fit.terms],
            r2=fit.r2,
            complexity=fit.complexity,
            eic=0.0,
            base_reward=fit.base_reward,
            success=True,
        )

        def refit_callback(current_fit, current_analysis, removed_index):
            try:
                terms = tuple(
                    decompose_expand_mul(
                        current_fit.expression,
                        tuple(
                            sp.Symbol(name, real=True) for name in self.feature_names
                        ),
                    )
                )
                if len(terms) != len(current_analysis.terms):
                    raise ValueError(
                        f"pruning term mismatch {len(terms)} != {len(current_analysis.terms)}"
                    )
                retained = tuple(
                    term for index, term in enumerate(terms) if index != removed_index
                )
                if not terms[removed_index].basis.free_symbols:
                    # The ordinary additive GP evaluator always fits an
                    # intercept.  Removing that term cannot be represented by
                    # a raw tree without carrying an extra no-intercept state
                    # through every genetic operator.  Reject the selected
                    # deletion instead of silently choosing a different term
                    # or accepting a prune that reappears next generation.
                    raise PruneWritebackNotRepresentable(
                        "fixed regression intercept cannot be removed by the "
                        "current GP genotype"
                    )
                # A GP genotype encodes structural bases, not the fitted
                # floating-point coefficients.  Writing ``fitted_tree`` back
                # here made otherwise short pruned structures exceed the raw
                # max_len gate and also coupled future variation to coefficient
                # formatting.  Materialize the retained coefficient-free basis
                # sum, then evaluate it through the ordinary GP path so the
                # accepted reward is exactly what the next generation sees.
                genotype_expression = (
                    retained[0].basis
                    if len(retained) == 1
                    else sp.Add(
                        *(term.basis for term in retained),
                        evaluate=False,
                    )
                )
                genotype_tree = nd.parse(
                    to_project_expression_string(genotype_expression)
                )
                if len(genotype_tree) > self.max_len:
                    raise ValueError(
                        f"pruned genotype length {len(genotype_tree)} exceeds {self.max_len}"
                    )
                roundtrip_expression, roundtrip_symbols = parse_expression(
                    genotype_tree.to_str(number_format=".17g"),
                    self.feature_names,
                )
                roundtrip_terms = tuple(
                    decompose_expand_mul(roundtrip_expression, roundtrip_symbols)
                )
                expected_bases = tuple(term.canonical for term in retained)
                actual_bases = tuple(term.canonical for term in roundtrip_terms)
                if actual_bases != expected_bases:
                    raise PruneWritebackNotRepresentable(
                        "coefficient-free genotype changed the ordered retained "
                        f"basis multiset: expected={expected_bases} actual={actual_bases}"
                    )
                desired_fit = self.additive_evaluator.refit_terms(
                    retained,
                    X,
                    y,
                    raw_expression=genotype_tree.to_str(number_format=".17g"),
                )
                roundtrip_fit = self.additive_evaluator.evaluate(
                    genotype_tree,
                    X,
                    y,
                )
                for value in (desired_fit, roundtrip_fit):
                    if not value.success or value.fitted_tree is None:
                        raise ValueError(value.failure_message or value.failure_type)
                desired_terms = tuple(
                    (term.canonical, term.coefficient) for term in desired_fit.terms
                )
                roundtrip_terms = tuple(
                    (term.canonical, term.coefficient) for term in roundtrip_fit.terms
                )
                closure_matches = (
                    desired_fit.canonical_fitted_expression
                    == roundtrip_fit.canonical_fitted_expression
                    and desired_terms == roundtrip_terms
                    and desired_fit.complexity == roundtrip_fit.complexity
                    and math.isclose(
                        desired_fit.r2,
                        roundtrip_fit.r2,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        desired_fit.base_reward,
                        roundtrip_fit.base_reward,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                if not closure_matches:
                    raise PruneWritebackNotRepresentable(
                        "ordinary GP evaluation does not reproduce the retained-term "
                        f"refit: desired_terms={desired_terms} "
                        f"roundtrip_terms={roundtrip_terms}"
                    )
                refitted = roundtrip_fit
                result = RefitResult(
                    expression=refitted.fitted_expression,
                    coefficients=[term.coefficient for term in refitted.terms],
                    r2=refitted.r2,
                    complexity=refitted.complexity,
                    eic=0.0,
                    base_reward=refitted.base_reward,
                    success=True,
                    coefficient_fitting_seconds=(
                        desired_fit.elapsed_seconds + roundtrip_fit.elapsed_seconds
                    ),
                    geometry_hint=RefitGeometryHint(
                        expression=refitted.fitted_expression,
                        symbols=refitted.symbols,
                        terms=refitted.terms,
                        parent_geometry_key=current_analysis.candidate_geometry_key,
                        removed_index=removed_index,
                    ),
                )
                result._gp_fit_result = refitted
                result._gp_genotype_tree = genotype_tree
                return result
            except Exception as error:
                return RefitResult(
                    success=False,
                    failure_type=type(error).__name__,
                    failure_message=str(error),
                )

        def reevaluate_callback(refit):
            refitted = refit._gp_fit_result
            return self.sobolev_evaluator.evaluate_preparsed(
                raw_expression=str(refit.expression),
                fitted_expression=refitted.fitted_expression_text,
                expression=refitted.fitted_expression,
                symbols=refitted.symbols,
                terms=refitted.terms,
                X=X,
                dataset_identity=self.dataset_identity,
                geometry_indices=self.geometry_indices,
                parent_geometry_key=(
                    refit.geometry_hint.parent_geometry_key
                    if refit.geometry_hint is not None
                    else None
                ),
            )

        pruning = prune_and_refit(
            initial,
            analysis,
            PruningConfig(
                threshold=self.sobolev_tau,
                max_prunes=self.sobolev_max_prunes,
                acceptance_tolerance=self.sobolev_acceptance_tolerance,
                excluded_term_indices=excluded_prune_indices,
            ),
            refit_callback,
            reevaluate_callback,
        )
        self.stats["pruning_attempts"] += 1
        self.stats["pruning_accepted"] += pruning.accepted_prunes
        self.stats["pruning_rejected"] += pruning.rejected_prunes
        self.failure_counts[f"pruning:{pruning.termination_reason}"] += 1
        if pruning.accepted_prunes:
            pruning.final_analysis.raw_expression = (
                pruning.final_fit._gp_genotype_tree.to_str(number_format=".17g")
            )
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "pruning_traces.jsonl",
                {
                    "candidate_id": candidate.candidate_id,
                    "generation": candidate.generation,
                    "result": pruning.as_dict(),
                },
            )
        if not pruning.accepted_prunes:
            candidate.pruning_result = pruning
            return candidate
        refitted = pruning.final_fit._gp_fit_result
        repaired = ControlledIndividual(
            pruning.final_fit._gp_genotype_tree.copy(),
            candidate_id=candidate.candidate_id + 1,
            generation=candidate.generation,
            slot=candidate.slot,
            parent_ids=(candidate.candidate_id,),
            variation_type="prune_repair",
        )
        refitted = refitted.clone()
        refitted.raw_expression = repaired.raw_expression
        repaired.apply_fit(refitted)
        repaired.sobolev_result = copy.deepcopy(pruning.final_analysis)
        repaired.sobolev_result.raw_expression = repaired.raw_expression
        repaired.sobolev_success = repaired.sobolev_result.success
        repaired.sobolev_penalty = (
            sobolev_penalty(
                repaired.sobolev_result.term_novelties,
                self.sobolev_tau,
            )
            if repaired.sobolev_result.success
            else 1.0
        )
        repaired.final_reward = (
            repaired.base_reward
            if self.use_sn_gp_verified_repair
            else repaired.base_reward - self.sobolev_alpha * repaired.sobolev_penalty
        )
        repaired.shortlist_status = "shortlisted_prune_repair"
        repaired.base_rank = candidate.base_rank
        repaired.pruning_result = pruning
        repaired.prune_origin_candidate_id = candidate.candidate_id
        if self.use_basis_forest:
            genome, status = extract_strict_or_opaque_basis_genome(
                repaired.raw_expression,
                self.feature_names,
                max_len=self.max_len,
            )
            repaired.basis_genome = genome
            repaired.basis_forest_status = status
            self.stats["basis_forest_compression_exact_writeback"] += int(
                status == "exact_terms"
            )
            self.stats["basis_forest_compression_opaque_writeback"] += int(
                status != "exact_terms"
            )
        return repaired

    def _commit_generation(
        self,
        generation: int,
        population: list[ControlledIndividual],
    ) -> dict[str, Any]:
        self._update_archives(population)
        self._record_pursuit_base_progress(generation)
        wall_time = self._elapsed()
        current_base = self._rank_base(population)[0]
        current_method = (
            self.current_sn_elite_order[0]
            if self.use_sn_gp_stable and self.current_sn_elite_order
            else (
                self.current_structural_order[0]
                if self.current_structural_order
                else current_base
            )
        )
        snapshot = {
            "generation": int(generation),
            "wall_time": wall_time,
            "profile": self.profile,
            "seed": int(self.random_state),
            "dataset_identity": self.dataset_identity,
            "population_size": len(population),
            "valid_population": sum(candidate.valid for candidate in population),
            "current_method_selected": self._candidate_summary(current_method),
            "current_base_best": self._candidate_summary(current_base),
            "method_selected_best": self._candidate_summary(self.method_selected_best),
            "base_quality_best_ever": self._candidate_summary(
                self.base_quality_best_ever
            ),
            "best_by_sn_comparator": self._candidate_summary(self.method_selected_best),
            "best_by_base_reward": self._candidate_summary(self.base_quality_best_ever),
            "base_anchor": self._candidate_summary(current_base),
            "base_anchor_first_elite": bool(
                self.current_elite_order
                and self.current_elite_order[0].candidate_id
                == current_base.candidate_id
            ),
            "base_anchor_injected": self.last_anchor_injected,
            "sn_comparator_elite_ids": [
                value.candidate_id for value in self.current_sn_elite_order
            ],
            "final_elite_ids": [
                value.candidate_id for value in self.current_elite_order
            ],
            "coverage_candidate": self._candidate_summary(
                self.current_coverage_candidate
            ),
            "coverage_gain": self._finite(self.current_coverage_gain),
            "coverage_base_rank": self.current_coverage_base_rank,
            "coverage_slot_changed": self.current_coverage_slot_changed,
            "basis_archive_size": (
                0 if self.basis_archive is None else len(self.basis_archive.entries)
            ),
            "pursuit_stagnation_gate_active_next_generation": (
                self._pursuit_stagnation_gate_active(generation + 1)
                if self.use_sn_stagnation_pursuit_shadow
                else None
            ),
            "pursuit_last_meaningful_base_improvement_generation": (
                self.sn_pursuit_last_improvement_generation
                if self.use_sn_stagnation_pursuit_shadow
                else None
            ),
            "operator_counts": dict(sorted(self.operator_counts.items())),
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "telemetry": self._telemetry(),
        }
        if self.record_integrity_metadata:
            snapshot.update(
                {
                    "configuration_sha256": self.configuration_sha256,
                    "initial_population_sha256": self.initial_population_sha256,
                    "initial_base_fitness_sha256": self.initial_base_fitness_sha256,
                    "rng_schedule_identity": self.counter_rng.schedule_identity,
                    "population_sha256": self._population_sha256(population),
                }
            )
        self.records.append(snapshot)
        if self.use_sn_gp_stable and self.output_dir is not None:
            append_jsonl(
                self.output_dir / "anchor_trace.jsonl",
                {
                    "generation": int(generation),
                    "wall_time": wall_time,
                    "base_top1": snapshot["base_anchor"],
                    "base_anchor_first_elite": snapshot["base_anchor_first_elite"],
                    "base_anchor_injected": self.last_anchor_injected,
                    "sn_comparator_elite_ids": snapshot["sn_comparator_elite_ids"],
                    "final_elite_ids": snapshot["final_elite_ids"],
                    "global_base_best": snapshot["best_by_base_reward"],
                    "global_sn_best": snapshot["best_by_sn_comparator"],
                    "comparator_stats": (
                        {}
                        if self.sn_comparator is None
                        else self.sn_comparator.stats_dict()
                    ),
                    "mutation_stats": dict(sorted(self.sn_mutation_stats.items())),
                },
            )
        if wall_time <= self.time_limit:
            self.time_floor_snapshot = copy.deepcopy(snapshot)
        if self.checkpoint_writer is not None:
            pointer = self.checkpoint_writer.commit(snapshot)
            snapshot["checkpoint_sha256"] = pointer["checkpoint_sha256"]
            if wall_time <= self.time_limit and self.time_floor_snapshot is not None:
                self.time_floor_snapshot["checkpoint_sha256"] = pointer[
                    "checkpoint_sha256"
                ]
            atomic_write_json(
                self.output_dir / "latest_population_state.json",
                {
                    "generation": generation,
                    "population_sha256": snapshot["population_sha256"],
                    "individuals": [
                        {
                            "candidate_id": value.candidate_id,
                            "slot": value.slot,
                            "raw_expression": value.raw_expression,
                            "parent_ids": list(value.parent_ids),
                            "variation_type": value.variation_type,
                        }
                        for value in population
                    ],
                },
            )
        return snapshot

    def _update_archives(self, population: list[ControlledIndividual]) -> None:
        base_order = self._rank_base(population)
        current_base = base_order[0]
        current_method = (
            self.current_sn_elite_order[0]
            if self.use_sn_gp_stable and self.current_sn_elite_order
            else (
                self.current_structural_order[0]
                if self.current_structural_order
                else current_base
            )
        )
        if self.base_quality_best_ever is None or self._base_key(
            current_base
        ) < self._base_key(self.base_quality_best_ever):
            self.base_quality_best_ever = current_base.copy()
        if self.method_selected_best is None:
            self.method_selected_best = current_method.copy()
        elif self.use_sn_gp_plugin:
            assert self.sn_comparator is not None
            winner = self.sn_comparator.compare(
                current_method,
                self.method_selected_best,
                context="archive",
            ).winner
            if winner is current_method:
                self.method_selected_best = current_method.copy()
        elif self.use_sobolev:
            if self._structural_key(current_method) < self._structural_key(
                self.method_selected_best
            ):
                self.method_selected_best = current_method.copy()
        elif self._base_key(current_method) < self._base_key(self.method_selected_best):
            self.method_selected_best = current_method.copy()

    def _build_archive_export_candidate(self) -> None:
        """Recombine archived terms after search without changing GP ecology."""

        self.stats["sn_archive_export_events"] += 1
        self.archive_export_candidate = None
        self.archive_export_pareto_candidate = None
        self.archive_export_path = []
        if (
            self.basis_archive is None
            or not self.basis_archive.entries
            or self._active_X is None
            or self._active_y is None
            or self.base_quality_best_ever is None
        ):
            self.stats["sn_archive_export_fallback:archive_or_data_unavailable"] += 1
            return
        try:
            prefixes = build_archive_export_prefixes(
                entries=self.basis_archive.entries,
                target=self._active_y,
                variable_names=self.feature_names,
                max_steps=self.sn_archive_export_max_steps,
                max_len=self.max_len,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_archive_export_fallback:{type(error).__name__}"] += 1
            return
        self.stats["sn_archive_export_pool_terms"] += len(self.basis_archive.entries)
        self.stats["sn_archive_export_constructible_prefixes"] += len(prefixes)
        candidates: list[ControlledIndividual] = []
        generation = len(self.records)
        for index, prefix in enumerate(prefixes):
            candidate = ControlledIndividual(
                prefix.tree.copy(),
                candidate_id=self._proposal_id(generation + 1, index),
                generation=generation,
                slot=-1,
                variation_type="sn_archive_export_prefix",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_archive_export_prefixes_evaluated"] += 1
            self.stats["sn_archive_export_valid_prefixes"] += int(candidate.valid)
            details = {
                "depth": int(prefix.step.depth),
                "selected_canonical": prefix.step.canonical,
                "sobolev_gain": float(prefix.step.sobolev_gain),
                "partial_residual_correlation": float(
                    prefix.step.partial_residual_correlation
                ),
                "joint_score": float(prefix.step.joint_score),
                "residual_norm": float(prefix.step.residual_norm),
                "gene_count": len(prefix.genes),
                "candidate": self._candidate_summary(candidate),
            }
            self.archive_export_path.append(details)
            if candidate.valid:
                candidates.append(candidate)
        if not candidates:
            self.stats["sn_archive_export_fallback:no_valid_prefix"] += 1
            return
        best = min(candidates, key=self._base_key)
        self.archive_export_candidate = best.copy()
        baseline = self.base_quality_best_ever
        baseline_fit = baseline.fit_result
        eligible = [
            candidate
            for candidate in candidates
            if candidate.fit_result is not None
            and baseline_fit is not None
            and candidate.fit_result.r2 >= baseline_fit.r2 - 1e-12
            and self._base_key(candidate) < self._base_key(baseline)
        ]
        self.stats["sn_archive_export_pareto_prefixes"] += len(eligible)
        if eligible:
            selected = min(eligible, key=self._base_key)
            selected.variation_type = "sn_archive_export_selected"
            self.archive_export_pareto_candidate = selected.copy()
            self.stats["sn_archive_export_pareto_available"] += 1
            self.stats["sn_archive_export_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_export_r2_gain_sum"] += float(
                selected.fit_result.r2
            ) - float(baseline_fit.r2)
        else:
            self.stats["sn_archive_export_base_retained"] += 1
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "archive_export_trace.jsonl",
                {
                    "archive_size": len(self.basis_archive.entries),
                    "base": self._candidate_summary(baseline),
                    "best_prefix": self._candidate_summary(
                        self.archive_export_candidate
                    ),
                    "pareto_prefix": self._candidate_summary(
                        self.archive_export_pareto_candidate
                    ),
                    "path": self.archive_export_path,
                },
            )

    def _build_archive_anchor_export_candidate(self) -> None:
        """Complete the final Base-best with archived terms, never the population."""

        self.stats["sn_archive_anchor_export_events"] += 1
        self.archive_export_candidate = None
        self.archive_export_pareto_candidate = None
        self.archive_export_path = []
        if (
            self.basis_archive is None
            or not self.basis_archive.entries
            or self._active_X is None
            or self._active_y is None
            or self.base_quality_best_ever is None
        ):
            self.stats[
                "sn_archive_anchor_export_fallback:archive_or_data_unavailable"
            ] += 1
            return
        baseline = self.base_quality_best_ever
        current = baseline.copy()
        generation = len(self.records)
        for step in range(self.sn_archive_anchor_export_max_steps):
            proposals = self._build_screened_basis_pursuit_proposals(
                parent=current,
                X=self._active_X,
                y=self._active_y,
                generation=generation + step,
            )
            self.stats["sn_archive_anchor_export_proposals"] += len(proposals)
            if not proposals:
                self.stats["sn_archive_anchor_export_stops:no_proposal"] += 1
                break
            evaluated: list[tuple[ControlledIndividual, dict[str, Any]]] = []
            for index, (tree, donor, action, details) in enumerate(proposals):
                candidate = ControlledIndividual(
                    tree.copy(),
                    candidate_id=self._proposal_id(generation + step + 1, index),
                    generation=generation,
                    slot=-1,
                    parent_ids=(current.candidate_id,),
                    variation_type="sn_archive_anchor_export_prefix",
                )
                self._evaluate_individual(candidate, self._active_X, self._active_y)
                self.stats["sn_archive_anchor_export_prefixes_evaluated"] += 1
                self.stats["sn_archive_anchor_export_valid_prefixes"] += int(
                    candidate.valid
                )
                evaluated.append(
                    (
                        candidate,
                        {
                            **details,
                            "action": action,
                            "donor_canonical": donor.canonical,
                            "candidate": self._candidate_summary(candidate),
                        },
                    )
                )
            current_fit = current.fit_result
            eligible = [
                (candidate, details)
                for candidate, details in evaluated
                if candidate.valid
                and candidate.fit_result is not None
                and current_fit is not None
                and candidate.fit_result.r2 >= current_fit.r2 - 1e-12
                and self._base_key(candidate) < self._base_key(current)
            ]
            self.stats["sn_archive_anchor_export_pareto_prefixes"] += len(eligible)
            if not eligible:
                self.stats["sn_archive_anchor_export_stops:no_pareto_gain"] += 1
                self.archive_export_path.append(
                    {
                        "step": step + 1,
                        "parent": self._candidate_summary(current),
                        "selected": None,
                        "alternatives": [details for _, details in evaluated],
                    }
                )
                break
            selected, selected_details = min(
                eligible, key=lambda value: self._base_key(value[0])
            )
            selected.variation_type = "sn_archive_anchor_export_selected"
            self.archive_export_path.append(
                {
                    "step": step + 1,
                    "parent": self._candidate_summary(current),
                    "selected": self._candidate_summary(selected),
                    "selection": selected_details,
                    "alternatives": [details for _, details in evaluated],
                }
            )
            current = selected
            self.stats["sn_archive_anchor_export_steps_accepted"] += 1
        if self._base_key(current) >= self._base_key(baseline):
            self.stats["sn_archive_anchor_export_base_retained"] += 1
            return
        baseline_fit = baseline.fit_result
        current_fit = current.fit_result
        if (
            baseline_fit is None
            or current_fit is None
            or current_fit.r2 < baseline_fit.r2 - 1e-12
        ):
            self.stats["sn_archive_anchor_export_base_retained"] += 1
            return
        self.archive_export_candidate = current.copy()
        self.archive_export_pareto_candidate = current.copy()
        self.stats["sn_archive_anchor_export_pareto_available"] += 1
        self.stats["sn_archive_anchor_export_reward_gain_sum"] += float(
            current.base_reward
        ) - float(baseline.base_reward)
        self.stats["sn_archive_anchor_export_r2_gain_sum"] += float(
            current_fit.r2
        ) - float(baseline_fit.r2)
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "archive_anchor_export_trace.jsonl",
                {
                    "archive_size": len(self.basis_archive.entries),
                    "base": self._candidate_summary(baseline),
                    "selected": self._candidate_summary(current),
                    "path": self.archive_export_path,
                },
            )

    def _build_archive_beam_export_candidate(
        self,
        archive_entries: Sequence[BasisArchiveEntry] | None = None,
        *,
        floating_deletions: int = 0,
        sobolev_shortlist_size: int | None = None,
        value_shortlist_size: int = 0,
        innovation_shortlist_size: int = 0,
        pair_first_shortlist_size: int = 0,
        pair_second_shortlist_size: int = 0,
        accuracy_mode: bool = False,
    ) -> None:
        """Run an exact-Base beam over Sobolev-screened archive expansions."""

        if sobolev_shortlist_size is None:
            sobolev_shortlist_size = self.sn_archive_beam_shortlist_size
        if floating_deletions < 0:
            raise ValueError("floating deletion count cannot be negative")
        if sobolev_shortlist_size < 0:
            raise ValueError("beam Sobolev shortlist size cannot be negative")
        if value_shortlist_size < 0:
            raise ValueError("beam value shortlist size cannot be negative")
        if innovation_shortlist_size < 0:
            raise ValueError("beam innovation shortlist size cannot be negative")
        if pair_first_shortlist_size < 0 or pair_second_shortlist_size < 0:
            raise ValueError("beam pair shortlist sizes cannot be negative")
        if (pair_first_shortlist_size == 0) != (pair_second_shortlist_size == 0):
            raise ValueError("beam pair shortlist sizes must be enabled together")
        self.stats["sn_archive_beam_export_events"] += 1
        self.archive_export_candidate = None
        self.archive_export_pareto_candidate = None
        self.archive_export_path = []
        entries = tuple(
            self.basis_archive.entries
            if archive_entries is None and self.basis_archive is not None
            else (archive_entries or ())
        )
        if (
            not entries
            or self._active_X is None
            or self._active_y is None
            or self.base_quality_best_ever is None
        ):
            self.stats[
                "sn_archive_beam_export_fallback:archive_or_data_unavailable"
            ] += 1
            return

        baseline = self.base_quality_best_ever
        generation = len(self.records)
        beam: list[tuple[tuple[int, ...], ControlledIndividual | None]] = [((), None)]
        all_candidates: list[ControlledIndividual] = []
        evaluated_selections: set[tuple[int, ...]] = set()
        proposal_slot = 0
        for depth in range(1, self.sn_archive_beam_export_max_steps + 1):
            input_state_count = len(beam)
            next_states: list[tuple[tuple[int, ...], ControlledIndividual]] = []
            evaluated_at_depth = 0
            valid_at_depth = 0
            screened_at_depth = 0
            construction_failures = 0
            screen_failures = 0
            floating_at_depth = 0
            pair_at_depth = 0
            for selected_indices, parent in beam:
                try:
                    expansions = screen_archive_beam_expansions(
                        entries=entries,
                        target=self._active_y,
                        selected_indices=selected_indices,
                        shortlist_size=sobolev_shortlist_size,
                        value_shortlist_size=value_shortlist_size,
                        innovation_shortlist_size=innovation_shortlist_size,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    screen_failures += 1
                    continue
                self.stats["sn_archive_beam_states_screened"] += 1
                self.stats["sn_archive_beam_screened_expansions"] += len(expansions)
                self.stats["sn_archive_beam_sobolev_lane_expansions"] += sum(
                    "sobolev" in expansion.selection_lanes for expansion in expansions
                )
                self.stats["sn_archive_beam_value_lane_expansions"] += sum(
                    "value" in expansion.selection_lanes for expansion in expansions
                )
                self.stats["sn_archive_beam_value_only_expansions"] += sum(
                    expansion.selection_lanes == ("value",) for expansion in expansions
                )
                self.stats["sn_archive_beam_innovation_lane_expansions"] += sum(
                    "innovation" in expansion.selection_lanes
                    for expansion in expansions
                )
                self.stats["sn_archive_beam_innovation_only_expansions"] += sum(
                    expansion.selection_lanes == ("innovation",)
                    for expansion in expansions
                )
                screened_at_depth += len(expansions)
                for expansion in expansions:
                    selection = tuple(sorted((*selected_indices, expansion.pool_index)))
                    if selection in evaluated_selections:
                        self.stats["sn_archive_beam_duplicate_states"] += 1
                        continue
                    evaluated_selections.add(selection)
                    try:
                        tree = build_archive_beam_tree(
                            entries=entries,
                            selected_indices=selection,
                            variable_names=self.feature_names,
                            max_len=self.max_len,
                        )
                    except ValueError:
                        construction_failures += 1
                        self.stats["sn_archive_beam_export_fallback:construction"] += 1
                        continue
                    parent_ids = () if parent is None else (parent.candidate_id,)
                    candidate = ControlledIndividual(
                        tree,
                        candidate_id=self._proposal_id(generation + 1, proposal_slot),
                        generation=generation,
                        slot=-1,
                        parent_ids=parent_ids,
                        variation_type="sn_archive_beam_export_state",
                    )
                    proposal_slot += 1
                    self._evaluate_individual(candidate, self._active_X, self._active_y)
                    evaluated_at_depth += 1
                    self.stats["sn_archive_beam_states_evaluated"] += 1
                    self.stats["sn_archive_beam_valid_states"] += int(candidate.valid)
                    if not candidate.valid:
                        continue
                    valid_at_depth += 1
                    next_states.append((selection, candidate))
                    all_candidates.append(candidate)
                    if floating_deletions < 1 or len(selection) < 2:
                        continue
                    fit = candidate.fit_result
                    if fit is None:
                        continue
                    fitted_coefficients = {
                        term.canonical: float(term.coefficient) for term in fit.terms
                    }
                    try:
                        deletions = rank_archive_floating_deletions(
                            entries=entries,
                            selected_indices=selection,
                            fitted_coefficients=fitted_coefficients,
                            tau=self.sobolev_tau,
                            limit=floating_deletions,
                        )
                    except (ValueError, np.linalg.LinAlgError) as error:
                        self.stats[
                            "sn_archive_floating_fallback:" f"{type(error).__name__}"
                        ] += 1
                        continue
                    self.stats["sn_archive_floating_low_novelty_choices"] += len(
                        deletions
                    )
                    for deletion in deletions:
                        floating_selection = tuple(
                            index for index in selection if index != deletion.pool_index
                        )
                        if (
                            not floating_selection
                            or floating_selection in evaluated_selections
                        ):
                            self.stats["sn_archive_floating_duplicate_states"] += 1
                            continue
                        evaluated_selections.add(floating_selection)
                        try:
                            floating_tree = build_archive_beam_tree(
                                entries=entries,
                                selected_indices=floating_selection,
                                variable_names=self.feature_names,
                                max_len=self.max_len,
                            )
                        except ValueError:
                            construction_failures += 1
                            self.stats["sn_archive_floating_fallback:construction"] += 1
                            continue
                        floating_candidate = ControlledIndividual(
                            floating_tree,
                            candidate_id=self._proposal_id(
                                generation + 1, proposal_slot
                            ),
                            generation=generation,
                            slot=-1,
                            parent_ids=(candidate.candidate_id,),
                            variation_type="sn_archive_floating_delete_state",
                        )
                        proposal_slot += 1
                        self._evaluate_individual(
                            floating_candidate, self._active_X, self._active_y
                        )
                        floating_at_depth += 1
                        evaluated_at_depth += 1
                        self.stats["sn_archive_floating_states_evaluated"] += 1
                        self.stats["sn_archive_beam_states_evaluated"] += 1
                        self.stats["sn_archive_beam_valid_states"] += int(
                            floating_candidate.valid
                        )
                        self.stats["sn_archive_floating_valid_states"] += int(
                            floating_candidate.valid
                        )
                        if not floating_candidate.valid:
                            continue
                        valid_at_depth += 1
                        next_states.append((floating_selection, floating_candidate))
                        all_candidates.append(floating_candidate)
                        self.stats[
                            "sn_archive_floating_deletion_impact_sum"
                        ] += deletion.deletion_impact
                        self.stats[
                            "sn_archive_floating_deleted_novelty_sum"
                        ] += deletion.novelty

                if pair_first_shortlist_size > 0:
                    try:
                        pair_expansions = screen_archive_beam_pair_expansions(
                            entries=entries,
                            target=self._active_y,
                            selected_indices=selected_indices,
                            first_shortlist_size=pair_first_shortlist_size,
                            second_shortlist_size=pair_second_shortlist_size,
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        screen_failures += 1
                        self.stats["sn_archive_pair_screen_failures"] += 1
                        continue
                    self.stats["sn_archive_pair_states_screened"] += 1
                    self.stats["sn_archive_pair_screened_expansions"] += len(
                        pair_expansions
                    )
                    self.stats["sn_archive_pair_first_innovation_expansions"] += sum(
                        "innovation" in expansion.first_lanes
                        for expansion in pair_expansions
                    )
                    self.stats[
                        "sn_archive_pair_first_innovation_only_expansions"
                    ] += sum(
                        expansion.first_lanes == ("innovation",)
                        for expansion in pair_expansions
                    )
                    for pair_expansion in pair_expansions:
                        pair_selection = tuple(
                            sorted(
                                (
                                    *selected_indices,
                                    pair_expansion.first_pool_index,
                                    pair_expansion.second_pool_index,
                                )
                            )
                        )
                        if pair_selection in evaluated_selections:
                            self.stats["sn_archive_pair_duplicate_states"] += 1
                            continue
                        evaluated_selections.add(pair_selection)
                        try:
                            pair_tree = build_archive_beam_tree(
                                entries=entries,
                                selected_indices=pair_selection,
                                variable_names=self.feature_names,
                                max_len=self.max_len,
                            )
                        except ValueError:
                            construction_failures += 1
                            self.stats["sn_archive_pair_fallback:construction"] += 1
                            continue
                        pair_candidate = ControlledIndividual(
                            pair_tree,
                            candidate_id=self._proposal_id(
                                generation + 1, proposal_slot
                            ),
                            generation=generation,
                            slot=-1,
                            parent_ids=(
                                () if parent is None else (parent.candidate_id,)
                            ),
                            variation_type="sn_archive_pair_lookahead_state",
                        )
                        proposal_slot += 1
                        self._evaluate_individual(
                            pair_candidate, self._active_X, self._active_y
                        )
                        pair_at_depth += 1
                        evaluated_at_depth += 1
                        self.stats["sn_archive_pair_states_evaluated"] += 1
                        self.stats["sn_archive_beam_states_evaluated"] += 1
                        self.stats["sn_archive_pair_valid_states"] += int(
                            pair_candidate.valid
                        )
                        self.stats["sn_archive_beam_valid_states"] += int(
                            pair_candidate.valid
                        )
                        if not pair_candidate.valid:
                            continue
                        valid_at_depth += 1
                        next_states.append((pair_selection, pair_candidate))
                        all_candidates.append(pair_candidate)

            state_key = self._accuracy_key if accuracy_mode else self._base_key
            next_states.sort(key=lambda state: (state_key(state[1]), state[0]))
            retentions = select_archive_beam_retention(
                entries=entries,
                base_ordered_selections=[state[0] for state in next_states],
                beam_width=self.sn_archive_beam_width,
                diversity_slots=self.sn_archive_beam_diversity_slots,
            )
            beam = [next_states[retention.state_position] for retention in retentions]

            self.stats["sn_archive_beam_base_slots_retained"] += sum(
                retention.decision_stage == "base" for retention in retentions
            )
            self.stats["sn_archive_beam_diversity_slots_retained"] += sum(
                retention.decision_stage == "sobolev_diversity"
                for retention in retentions
            )
            self.stats["sn_archive_beam_diversity_gain_sum"] += sum(
                float(retention.diversity_gain or 0.0) for retention in retentions
            )
            self.stats["sn_archive_beam_states_retained"] += len(beam)
            self.stats["sn_archive_beam_screen_failures"] += screen_failures
            self.stats["sn_archive_beam_construction_failures"] += construction_failures
            if beam:
                self.stats["sn_archive_beam_max_depth"] = max(
                    int(self.stats["sn_archive_beam_max_depth"]), depth
                )
            self.archive_export_path.append(
                {
                    "depth": depth,
                    "input_state_count": input_state_count,
                    "screened_expansions": screened_at_depth,
                    "evaluated_states": evaluated_at_depth,
                    "valid_states": valid_at_depth,
                    "floating_states": floating_at_depth,
                    "pair_states": pair_at_depth,
                    "ranking_mode": "accuracy" if accuracy_mode else "base",
                    "screen_failures": screen_failures,
                    "construction_failures": construction_failures,
                    "retained": [
                        {
                            "decision_stage": retention.decision_stage,
                            "diversity_gain": retention.diversity_gain,
                            "archive_indices": list(selection),
                            "canonicals": [
                                entries[index].canonical for index in selection
                            ],
                            "candidate": self._candidate_summary(candidate),
                        }
                        for retention, (selection, candidate) in zip(
                            retentions, beam, strict=True
                        )
                    ],
                }
            )
            if not beam:
                self.stats["sn_archive_beam_stops:no_state"] += 1
                break

        self.stats["sn_archive_beam_pool_terms"] += len(entries)
        self.stats["sn_archive_beam_unique_states"] += len(evaluated_selections)
        if not all_candidates:
            self.stats["sn_archive_beam_export_fallback:no_valid_state"] += 1
            return
        best_key = self._accuracy_key if accuracy_mode else self._base_key
        best = min(all_candidates, key=best_key)
        self.archive_export_candidate = best.copy()
        baseline_fit = baseline.fit_result
        eligible = [
            candidate
            for candidate in all_candidates
            if candidate.fit_result is not None
            and baseline_fit is not None
            and candidate.fit_result.r2 >= baseline_fit.r2 - 1e-12
            and self._base_key(candidate) < self._base_key(baseline)
        ]
        self.stats["sn_archive_beam_pareto_states"] += len(eligible)
        if eligible:
            selected = min(eligible, key=self._base_key)
            selected.variation_type = "sn_archive_beam_export_selected"
            self.archive_export_pareto_candidate = selected.copy()
            self.stats["sn_archive_beam_pareto_available"] += 1
            self.stats["sn_archive_beam_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_beam_r2_gain_sum"] += float(
                selected.fit_result.r2
            ) - float(baseline_fit.r2)
        else:
            self.stats["sn_archive_beam_base_retained"] += 1
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "archive_beam_export_trace.jsonl",
                {
                    "archive_size": len(entries),
                    "base": self._candidate_summary(baseline),
                    "best_state": self._candidate_summary(
                        self.archive_export_candidate
                    ),
                    "pareto_state": self._candidate_summary(
                        self.archive_export_pareto_candidate
                    ),
                    "path": self.archive_export_path,
                },
            )

    def _build_archive_value_only_accuracy_export_candidate(self) -> None:
        """Accuracy-control beam without Sobolev lanes or diversity retention."""

        original_diversity_slots = self.sn_archive_beam_diversity_slots
        try:
            self.sn_archive_beam_diversity_slots = 0
            self._build_archive_beam_export_candidate(
                sobolev_shortlist_size=0,
                value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
                accuracy_mode=True,
            )
        finally:
            self.sn_archive_beam_diversity_slots = original_diversity_slots

    def _build_archive_dual_beam_export_candidate(
        self,
        archive_entries: Sequence[BasisArchiveEntry] | None = None,
        *,
        floating_deletions: int = 0,
        value_shortlist_size: int = 0,
        innovation_shortlist_size: int = 0,
        pair_first_shortlist_size: int = 0,
        pair_second_shortlist_size: int = 0,
        accuracy_mode: bool = False,
    ) -> None:
        """Union Base-only and Sobolev-diverse beams on one Base trajectory."""

        self.stats["sn_archive_dual_beam_export_events"] += 1
        original_diversity_slots = self.sn_archive_beam_diversity_slots
        schedule_results: list[dict[str, Any]] = []
        try:
            for diversity_slots in (0, 2):
                self.sn_archive_beam_diversity_slots = diversity_slots
                self._build_archive_beam_export_candidate(
                    archive_entries,
                    floating_deletions=floating_deletions,
                    value_shortlist_size=value_shortlist_size,
                    innovation_shortlist_size=innovation_shortlist_size,
                    pair_first_shortlist_size=pair_first_shortlist_size,
                    pair_second_shortlist_size=pair_second_shortlist_size,
                    accuracy_mode=accuracy_mode,
                )
                schedule_results.append(
                    {
                        "diversity_slots": diversity_slots,
                        "floating_deletions": floating_deletions,
                        "value_shortlist_size": value_shortlist_size,
                        "innovation_shortlist_size": innovation_shortlist_size,
                        "pair_first_shortlist_size": pair_first_shortlist_size,
                        "pair_second_shortlist_size": pair_second_shortlist_size,
                        "accuracy_mode": accuracy_mode,
                        "best": (
                            None
                            if self.archive_export_candidate is None
                            else self.archive_export_candidate.copy()
                        ),
                        "pareto": (
                            None
                            if self.archive_export_pareto_candidate is None
                            else self.archive_export_pareto_candidate.copy()
                        ),
                        "path": copy.deepcopy(self.archive_export_path),
                    }
                )
        finally:
            self.sn_archive_beam_diversity_slots = original_diversity_slots

        self.stats["sn_archive_dual_beam_schedules"] += len(schedule_results)
        best_candidates = [
            result["best"] for result in schedule_results if result["best"] is not None
        ]
        pareto_candidates = [
            result["pareto"]
            for result in schedule_results
            if result["pareto"] is not None
        ]

        def output_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
            fit = candidate.fit_result
            r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
            return (-r2, self._base_key(candidate))

        self.archive_export_candidate = (
            None if not best_candidates else min(best_candidates, key=output_key).copy()
        )
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=output_key).copy()
        )
        self.archive_export_path = [
            {
                "diversity_slots": result["diversity_slots"],
                "floating_deletions": result["floating_deletions"],
                "value_shortlist_size": result["value_shortlist_size"],
                "innovation_shortlist_size": result["innovation_shortlist_size"],
                "pair_first_shortlist_size": result["pair_first_shortlist_size"],
                "pair_second_shortlist_size": result["pair_second_shortlist_size"],
                "accuracy_mode": result["accuracy_mode"],
                "best": self._candidate_summary(result["best"]),
                "pareto": self._candidate_summary(result["pareto"]),
                "path": result["path"],
            }
            for result in schedule_results
        ]
        selected = self.archive_export_pareto_candidate
        baseline = self.base_quality_best_ever
        if selected is None or baseline is None:
            self.stats["sn_archive_dual_beam_base_retained"] += 1
            return
        selected.variation_type = "sn_archive_dual_beam_export_selected"
        selected_fit = selected.fit_result
        baseline_fit = baseline.fit_result
        self.stats["sn_archive_dual_beam_pareto_available"] += 1
        if selected_fit is not None and baseline_fit is not None:
            self.stats["sn_archive_dual_beam_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_dual_beam_r2_gain_sum"] += float(
                selected_fit.r2
            ) - float(baseline_fit.r2)
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "archive_dual_beam_export_trace.jsonl",
                {
                    "base": self._candidate_summary(baseline),
                    "selected": self._candidate_summary(selected),
                    "schedules": self.archive_export_path,
                },
            )

    def _build_archive_dual_pool_beam_export_candidate(self) -> None:
        """Search coverage and coverage-plus-quality pools with both beams."""

        self.stats["sn_archive_dual_pool_export_events"] += 1
        if self.basis_archive is None or not self.basis_archive.entries:
            self.stats[
                "sn_archive_dual_pool_fallback:coverage_archive_unavailable"
            ] += 1
            self.archive_export_candidate = None
            self.archive_export_pareto_candidate = None
            self.archive_export_path = []
            return
        coverage_entries = tuple(self.basis_archive.entries)
        quality_entries = (
            ()
            if self.basis_quality_archive is None
            else tuple(self.basis_quality_archive.entries)
        )
        union_entries = list(coverage_entries)
        known = {entry.canonical for entry in union_entries}
        for entry in quality_entries:
            if entry.canonical not in known:
                union_entries.append(entry)
                known.add(entry.canonical)
        pools: list[tuple[str, tuple[BasisArchiveEntry, ...]]] = [
            ("coverage", coverage_entries)
        ]
        if len(union_entries) > len(coverage_entries):
            pools.append(("coverage_plus_quality", tuple(union_entries)))
        self.stats["sn_archive_dual_pool_coverage_terms"] += len(coverage_entries)
        self.stats["sn_archive_dual_pool_quality_terms"] += len(quality_entries)
        self.stats["sn_archive_dual_pool_union_terms"] += len(union_entries)
        self.stats["sn_archive_dual_pool_quality_only_terms"] += len(
            union_entries
        ) - len(coverage_entries)

        pool_results: list[dict[str, Any]] = []
        for label, entries in pools:
            self._build_archive_dual_beam_export_candidate(entries)
            pool_results.append(
                {
                    "pool": label,
                    "entry_count": len(entries),
                    "best": (
                        None
                        if self.archive_export_candidate is None
                        else self.archive_export_candidate.copy()
                    ),
                    "pareto": (
                        None
                        if self.archive_export_pareto_candidate is None
                        else self.archive_export_pareto_candidate.copy()
                    ),
                    "path": copy.deepcopy(self.archive_export_path),
                }
            )
        self.stats["sn_archive_dual_pool_pools_searched"] += len(pool_results)

        def output_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
            fit = candidate.fit_result
            r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
            return (-r2, self._base_key(candidate))

        best_candidates = [
            result["best"] for result in pool_results if result["best"] is not None
        ]
        pareto_candidates = [
            result["pareto"] for result in pool_results if result["pareto"] is not None
        ]
        self.archive_export_candidate = (
            None if not best_candidates else min(best_candidates, key=output_key).copy()
        )
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=output_key).copy()
        )
        self.archive_export_path = [
            {
                "pool": result["pool"],
                "entry_count": result["entry_count"],
                "best": self._candidate_summary(result["best"]),
                "pareto": self._candidate_summary(result["pareto"]),
                "path": result["path"],
            }
            for result in pool_results
        ]
        selected = self.archive_export_pareto_candidate
        baseline = self.base_quality_best_ever
        if selected is None or baseline is None:
            self.stats["sn_archive_dual_pool_base_retained"] += 1
            return
        selected.variation_type = "sn_archive_dual_pool_beam_export_selected"
        selected_fit = selected.fit_result
        baseline_fit = baseline.fit_result
        self.stats["sn_archive_dual_pool_pareto_available"] += 1
        if selected_fit is not None and baseline_fit is not None:
            self.stats["sn_archive_dual_pool_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_dual_pool_r2_gain_sum"] += float(
                selected_fit.r2
            ) - float(baseline_fit.r2)

    def _build_archive_refined_dual_beam_export_candidate(self) -> None:
        """Monotonically refine the compact dual-beam winner with archive terms."""

        self.stats["sn_archive_refined_export_events"] += 1
        self._build_archive_dual_beam_export_candidate()
        dual_path = copy.deepcopy(self.archive_export_path)
        dual_best = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        dual_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        baseline = self.base_quality_best_ever
        if (
            baseline is None
            or self.basis_archive is None
            or not self.basis_archive.entries
            or self._active_X is None
            or self._active_y is None
        ):
            self.stats[
                "sn_archive_refined_export_fallback:archive_or_data_unavailable"
            ] += 1
            return

        current = (dual_pareto or baseline).copy()
        refinement_path: list[dict[str, Any]] = []
        generation = len(self.records) + self.sn_archive_beam_export_max_steps
        for step in range(self.sn_archive_beam_refine_max_steps):
            proposals = self._build_screened_basis_pursuit_proposals(
                parent=current,
                X=self._active_X,
                y=self._active_y,
                generation=generation + step,
            )
            self.stats["sn_archive_refined_export_proposals"] += len(proposals)
            if not proposals:
                self.stats["sn_archive_refined_export_stops:no_proposal"] += 1
                break

            evaluated: list[tuple[ControlledIndividual, dict[str, Any]]] = []
            for index, (tree, donor, action, details) in enumerate(proposals):
                candidate = ControlledIndividual(
                    tree.copy(),
                    candidate_id=self._proposal_id(generation + step + 1, index),
                    generation=generation,
                    slot=-1,
                    parent_ids=(current.candidate_id,),
                    variation_type="sn_archive_refined_export_proposal",
                )
                self._evaluate_individual(candidate, self._active_X, self._active_y)
                self.stats["sn_archive_refined_export_candidates_evaluated"] += 1
                self.stats["sn_archive_refined_export_valid_candidates"] += int(
                    candidate.valid
                )
                evaluated.append(
                    (
                        candidate,
                        {
                            **details,
                            "action": action,
                            "donor_canonical": donor.canonical,
                            "candidate": self._candidate_summary(candidate),
                        },
                    )
                )

            current_fit = current.fit_result
            eligible = [
                (candidate, details)
                for candidate, details in evaluated
                if candidate.valid
                and candidate.fit_result is not None
                and current_fit is not None
                and candidate.fit_result.r2 >= current_fit.r2 - 1e-12
                and self._base_key(candidate) < self._base_key(current)
            ]
            self.stats["sn_archive_refined_export_pareto_candidates"] += len(eligible)
            if not eligible:
                self.stats["sn_archive_refined_export_stops:no_monotone_gain"] += 1
                refinement_path.append(
                    {
                        "step": step + 1,
                        "parent": self._candidate_summary(current),
                        "selected": None,
                        "alternatives": [details for _, details in evaluated],
                    }
                )
                break

            selected, selected_details = min(
                eligible, key=lambda value: self._base_key(value[0])
            )
            selected.variation_type = "sn_archive_refined_export_selected"
            refinement_path.append(
                {
                    "step": step + 1,
                    "parent": self._candidate_summary(current),
                    "selected": self._candidate_summary(selected),
                    "selection": selected_details,
                    "alternatives": [details for _, details in evaluated],
                }
            )
            current = selected
            self.stats["sn_archive_refined_export_steps_accepted"] += 1

        self.archive_export_candidate = current.copy()
        self.archive_export_path = [
            {
                "stage": "dual_beam",
                "best": self._candidate_summary(dual_best),
                "pareto": self._candidate_summary(dual_pareto),
                "path": dual_path,
            },
            {"stage": "monotone_refinement", "path": refinement_path},
        ]
        baseline_fit = baseline.fit_result
        current_fit = current.fit_result
        if (
            baseline_fit is None
            or current_fit is None
            or current_fit.r2 < baseline_fit.r2 - 1e-12
            or self._base_key(current) >= self._base_key(baseline)
        ):
            self.archive_export_pareto_candidate = None
            self.stats["sn_archive_refined_export_base_retained"] += 1
            return
        self.archive_export_pareto_candidate = current.copy()
        self.stats["sn_archive_refined_export_pareto_available"] += 1
        self.stats["sn_archive_refined_export_reward_gain_sum"] += float(
            current.base_reward
        ) - float(baseline.base_reward)
        self.stats["sn_archive_refined_export_r2_gain_sum"] += float(
            current_fit.r2
        ) - float(baseline_fit.r2)
        if self.output_dir is not None and self.detailed_logging:
            append_jsonl(
                self.output_dir / "archive_refined_dual_beam_export_trace.jsonl",
                {
                    "base": self._candidate_summary(baseline),
                    "dual_beam": self._candidate_summary(dual_pareto),
                    "selected": self._candidate_summary(current),
                    "path": self.archive_export_path,
                },
            )

    def _build_archive_floating_dual_beam_export_candidate(self) -> None:
        """Union AR with forward-plus-backward Sobolev floating beams."""

        self.stats["sn_archive_floating_export_events"] += 1
        branch_results: list[dict[str, Any]] = []
        for label, deletion_count in (("ar", 0), ("floating", 1)):
            self._build_archive_dual_beam_export_candidate(
                floating_deletions=deletion_count
            )
            branch_results.append(
                {
                    "branch": label,
                    "floating_deletions": deletion_count,
                    "best": (
                        None
                        if self.archive_export_candidate is None
                        else self.archive_export_candidate.copy()
                    ),
                    "pareto": (
                        None
                        if self.archive_export_pareto_candidate is None
                        else self.archive_export_pareto_candidate.copy()
                    ),
                    "path": copy.deepcopy(self.archive_export_path),
                }
            )

        def output_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
            fit = candidate.fit_result
            r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
            return (-r2, self._base_key(candidate))

        best_candidates = [
            result["best"] for result in branch_results if result["best"] is not None
        ]
        pareto_candidates = [
            result["pareto"]
            for result in branch_results
            if result["pareto"] is not None
        ]
        self.archive_export_candidate = (
            None if not best_candidates else min(best_candidates, key=output_key).copy()
        )
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=output_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": result["branch"],
                "floating_deletions": result["floating_deletions"],
                "best": self._candidate_summary(result["best"]),
                "pareto": self._candidate_summary(result["pareto"]),
                "path": result["path"],
            }
            for result in branch_results
        ]
        selected = self.archive_export_pareto_candidate
        baseline = self.base_quality_best_ever
        if selected is None or baseline is None:
            self.stats["sn_archive_floating_export_base_retained"] += 1
            return
        selected.variation_type = "sn_archive_floating_export_selected"
        selected_fit = selected.fit_result
        baseline_fit = baseline.fit_result
        self.stats["sn_archive_floating_export_pareto_available"] += 1
        if selected_fit is not None and baseline_fit is not None:
            self.stats["sn_archive_floating_export_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_floating_export_r2_gain_sum"] += float(
                selected_fit.r2
            ) - float(baseline_fit.r2)

    def _build_archive_balanced_screen_dual_beam_export_candidate(self) -> None:
        """Union AR with beams screened by Sobolev and value-residual lanes."""

        self.stats["sn_archive_balanced_screen_export_events"] += 1
        branch_results: list[dict[str, Any]] = []
        for label, value_size in (
            ("ar", 0),
            ("balanced_screen", self.sn_archive_beam_value_shortlist_size),
        ):
            self._build_archive_dual_beam_export_candidate(
                value_shortlist_size=value_size
            )
            branch_results.append(
                {
                    "branch": label,
                    "value_shortlist_size": value_size,
                    "best": (
                        None
                        if self.archive_export_candidate is None
                        else self.archive_export_candidate.copy()
                    ),
                    "pareto": (
                        None
                        if self.archive_export_pareto_candidate is None
                        else self.archive_export_pareto_candidate.copy()
                    ),
                    "path": copy.deepcopy(self.archive_export_path),
                }
            )

        def output_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
            fit = candidate.fit_result
            r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
            return (-r2, self._base_key(candidate))

        best_candidates = [
            result["best"] for result in branch_results if result["best"] is not None
        ]
        pareto_candidates = [
            result["pareto"]
            for result in branch_results
            if result["pareto"] is not None
        ]
        self.archive_export_candidate = (
            None if not best_candidates else min(best_candidates, key=output_key).copy()
        )
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=output_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": result["branch"],
                "value_shortlist_size": result["value_shortlist_size"],
                "best": self._candidate_summary(result["best"]),
                "pareto": self._candidate_summary(result["pareto"]),
                "path": result["path"],
            }
            for result in branch_results
        ]
        selected = self.archive_export_pareto_candidate
        baseline = self.base_quality_best_ever
        if selected is None or baseline is None:
            self.stats["sn_archive_balanced_screen_base_retained"] += 1
            return
        selected.variation_type = "sn_archive_balanced_screen_export_selected"
        selected_fit = selected.fit_result
        baseline_fit = baseline.fit_result
        self.stats["sn_archive_balanced_screen_pareto_available"] += 1
        if selected_fit is not None and baseline_fit is not None:
            self.stats["sn_archive_balanced_screen_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_balanced_screen_r2_gain_sum"] += float(
                selected_fit.r2
            ) - float(baseline_fit.r2)

    def _build_archive_balanced_floating_envelope_candidate(self) -> None:
        """Take the safe envelope of AR, balanced screen, and floating beams."""

        self.stats["sn_archive_balanced_floating_export_events"] += 1
        branch_results: list[dict[str, Any]] = []
        branches = (
            ("ar", 0, 0),
            ("floating", 1, 0),
            ("balanced", 0, self.sn_archive_beam_value_shortlist_size),
            (
                "balanced_floating",
                1,
                self.sn_archive_beam_value_shortlist_size,
            ),
        )
        for label, deletion_count, value_size in branches:
            self._build_archive_dual_beam_export_candidate(
                floating_deletions=deletion_count,
                value_shortlist_size=value_size,
            )
            branch_results.append(
                {
                    "branch": label,
                    "floating_deletions": deletion_count,
                    "value_shortlist_size": value_size,
                    "best": (
                        None
                        if self.archive_export_candidate is None
                        else self.archive_export_candidate.copy()
                    ),
                    "pareto": (
                        None
                        if self.archive_export_pareto_candidate is None
                        else self.archive_export_pareto_candidate.copy()
                    ),
                    "path": copy.deepcopy(self.archive_export_path),
                }
            )

        def output_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
            fit = candidate.fit_result
            r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
            return (-r2, self._base_key(candidate))

        best_candidates = [
            result["best"] for result in branch_results if result["best"] is not None
        ]
        pareto_candidates = [
            result["pareto"]
            for result in branch_results
            if result["pareto"] is not None
        ]
        self.archive_export_candidate = (
            None if not best_candidates else min(best_candidates, key=output_key).copy()
        )
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=output_key).copy()
        )
        if pareto_candidates:
            chosen_result = min(
                (result for result in branch_results if result["pareto"] is not None),
                key=lambda result: output_key(result["pareto"]),
            )
            self.stats[
                "sn_archive_balanced_floating_branch:" f"{chosen_result['branch']}"
            ] += 1
        self.archive_export_path = [
            {
                "branch": result["branch"],
                "floating_deletions": result["floating_deletions"],
                "value_shortlist_size": result["value_shortlist_size"],
                "best": self._candidate_summary(result["best"]),
                "pareto": self._candidate_summary(result["pareto"]),
                "path": result["path"],
            }
            for result in branch_results
        ]
        selected = self.archive_export_pareto_candidate
        baseline = self.base_quality_best_ever
        if selected is None or baseline is None:
            self.stats["sn_archive_balanced_floating_base_retained"] += 1
            return
        selected.variation_type = "sn_archive_balanced_floating_export_selected"
        selected_fit = selected.fit_result
        baseline_fit = baseline.fit_result
        self.stats["sn_archive_balanced_floating_pareto_available"] += 1
        if selected_fit is not None and baseline_fit is not None:
            self.stats["sn_archive_balanced_floating_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_balanced_floating_r2_gain_sum"] += float(
                selected_fit.r2
            ) - float(baseline_fit.r2)

    def _build_archive_innovation_screen_envelope_candidate(self) -> None:
        """Envelope AR/AV with a pure-Sobolev innovation screen lane."""

        self.stats["sn_archive_innovation_screen_export_events"] += 1
        branch_results: list[dict[str, Any]] = []
        branches = (
            ("ar", 0, 0),
            ("balanced", self.sn_archive_beam_value_shortlist_size, 0),
            (
                "innovation_balanced",
                self.sn_archive_beam_value_shortlist_size,
                self.sn_archive_beam_innovation_shortlist_size,
            ),
        )
        for label, value_size, innovation_size in branches:
            self._build_archive_dual_beam_export_candidate(
                value_shortlist_size=value_size,
                innovation_shortlist_size=innovation_size,
            )
            branch_results.append(
                {
                    "branch": label,
                    "value_shortlist_size": value_size,
                    "innovation_shortlist_size": innovation_size,
                    "best": (
                        None
                        if self.archive_export_candidate is None
                        else self.archive_export_candidate.copy()
                    ),
                    "pareto": (
                        None
                        if self.archive_export_pareto_candidate is None
                        else self.archive_export_pareto_candidate.copy()
                    ),
                    "path": copy.deepcopy(self.archive_export_path),
                }
            )

        def output_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
            fit = candidate.fit_result
            r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
            return (-r2, self._base_key(candidate))

        best_candidates = [
            result["best"] for result in branch_results if result["best"] is not None
        ]
        pareto_candidates = [
            result["pareto"]
            for result in branch_results
            if result["pareto"] is not None
        ]
        self.archive_export_candidate = (
            None if not best_candidates else min(best_candidates, key=output_key).copy()
        )
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=output_key).copy()
        )
        if pareto_candidates:
            chosen_result = min(
                (result for result in branch_results if result["pareto"] is not None),
                key=lambda result: output_key(result["pareto"]),
            )
            self.stats[
                "sn_archive_innovation_screen_branch:" f"{chosen_result['branch']}"
            ] += 1
        self.archive_export_path = [
            {
                "branch": result["branch"],
                "value_shortlist_size": result["value_shortlist_size"],
                "innovation_shortlist_size": result["innovation_shortlist_size"],
                "best": self._candidate_summary(result["best"]),
                "pareto": self._candidate_summary(result["pareto"]),
                "path": result["path"],
            }
            for result in branch_results
        ]
        selected = self.archive_export_pareto_candidate
        baseline = self.base_quality_best_ever
        if selected is None or baseline is None:
            self.stats["sn_archive_innovation_screen_base_retained"] += 1
            return
        selected.variation_type = "sn_archive_innovation_screen_export_selected"
        selected_fit = selected.fit_result
        baseline_fit = baseline.fit_result
        self.stats["sn_archive_innovation_screen_pareto_available"] += 1
        if selected_fit is not None and baseline_fit is not None:
            self.stats["sn_archive_innovation_screen_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_innovation_screen_r2_gain_sum"] += float(
                selected_fit.r2
            ) - float(baseline_fit.r2)

    def _build_archive_pair_lookahead_envelope_candidate(self) -> None:
        """Envelope prior safe exports with atomic complementary term pairs."""

        self.stats["sn_archive_pair_lookahead_export_events"] += 1
        branch_results: list[dict[str, Any]] = []
        branches = (
            ("ar", 0, 0, 0, 0),
            ("balanced", self.sn_archive_beam_value_shortlist_size, 0, 0, 0),
            (
                "innovation_balanced",
                self.sn_archive_beam_value_shortlist_size,
                self.sn_archive_beam_innovation_shortlist_size,
                0,
                0,
            ),
            (
                "pair_lookahead",
                self.sn_archive_beam_value_shortlist_size,
                0,
                self.sn_archive_beam_pair_first_shortlist_size,
                self.sn_archive_beam_pair_second_shortlist_size,
            ),
        )
        for label, value_size, innovation_size, first_size, second_size in branches:
            self._build_archive_dual_beam_export_candidate(
                value_shortlist_size=value_size,
                innovation_shortlist_size=innovation_size,
                pair_first_shortlist_size=first_size,
                pair_second_shortlist_size=second_size,
            )
            branch_results.append(
                {
                    "branch": label,
                    "value_shortlist_size": value_size,
                    "innovation_shortlist_size": innovation_size,
                    "pair_first_shortlist_size": first_size,
                    "pair_second_shortlist_size": second_size,
                    "best": (
                        None
                        if self.archive_export_candidate is None
                        else self.archive_export_candidate.copy()
                    ),
                    "pareto": (
                        None
                        if self.archive_export_pareto_candidate is None
                        else self.archive_export_pareto_candidate.copy()
                    ),
                    "path": copy.deepcopy(self.archive_export_path),
                }
            )

        def output_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
            fit = candidate.fit_result
            r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
            return (-r2, self._base_key(candidate))

        best_candidates = [
            result["best"] for result in branch_results if result["best"] is not None
        ]
        pareto_candidates = [
            result["pareto"]
            for result in branch_results
            if result["pareto"] is not None
        ]
        self.archive_export_candidate = (
            None if not best_candidates else min(best_candidates, key=output_key).copy()
        )
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=output_key).copy()
        )
        if pareto_candidates:
            chosen_result = min(
                (result for result in branch_results if result["pareto"] is not None),
                key=lambda result: output_key(result["pareto"]),
            )
            self.stats[
                "sn_archive_pair_lookahead_branch:" f"{chosen_result['branch']}"
            ] += 1
        self.archive_export_path = [
            {
                "branch": result["branch"],
                "value_shortlist_size": result["value_shortlist_size"],
                "innovation_shortlist_size": result["innovation_shortlist_size"],
                "pair_first_shortlist_size": result["pair_first_shortlist_size"],
                "pair_second_shortlist_size": result["pair_second_shortlist_size"],
                "best": self._candidate_summary(result["best"]),
                "pareto": self._candidate_summary(result["pareto"]),
                "path": result["path"],
            }
            for result in branch_results
        ]
        selected = self.archive_export_pareto_candidate
        baseline = self.base_quality_best_ever
        if selected is None or baseline is None:
            self.stats["sn_archive_pair_lookahead_base_retained"] += 1
            return
        selected.variation_type = "sn_archive_pair_lookahead_export_selected"
        selected_fit = selected.fit_result
        baseline_fit = baseline.fit_result
        self.stats["sn_archive_pair_lookahead_pareto_available"] += 1
        if selected_fit is not None and baseline_fit is not None:
            self.stats["sn_archive_pair_lookahead_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_pair_lookahead_r2_gain_sum"] += float(
                selected_fit.r2
            ) - float(baseline_fit.r2)

    def _build_archive_product_lift_envelope_candidate(
        self, *, accuracy_mode: bool = False
    ) -> None:
        """Lift cross-candidate products, then exact-score an AV/AX envelope."""

        self.stats["sn_archive_product_lift_export_events"] += 1
        self.archive_export_candidate = None
        self.archive_export_pareto_candidate = None
        self.archive_export_path = []
        if (
            self.basis_archive is None
            or not self.basis_archive.entries
            or self._active_y is None
            or self.geometry_indices is None
            or self.base_quality_best_ever is None
        ):
            self.stats[
                "sn_archive_product_lift_fallback:archive_or_geometry_unavailable"
            ] += 1
            return
        original_entries = tuple(self.basis_archive.entries)
        try:
            lift_result = lift_archive_product_interactions(
                entries=original_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_archive_product_lift_fallback:" f"{type(error).__name__}"
            ] += 1
            lift_result = None

        lifted_entries: tuple[BasisArchiveEntry, ...] = ()
        if lift_result is not None:
            lifted_entries = tuple(lift.entry for lift in lift_result.lifts)
            self.stats[
                "sn_archive_product_lift_pairs_screened"
            ] += lift_result.pairs_screened
            self.stats[
                "sn_archive_product_lift_numeric_candidates"
            ] += lift_result.numeric_candidates
            self.stats["sn_archive_product_lift_terms"] += len(lift_result.lifts)
            self.stats[
                "sn_archive_product_lift_construction_failures"
            ] += lift_result.construction_failures
            self.stats[
                "sn_archive_product_lift_canonical_duplicates"
            ] += lift_result.canonical_duplicates
            self.stats["sn_archive_product_lift_sobolev_lane_terms"] += sum(
                "sobolev_product" in lift.selection_lanes for lift in lift_result.lifts
            )
            self.stats["sn_archive_product_lift_value_lane_terms"] += sum(
                "value_product" in lift.selection_lanes for lift in lift_result.lifts
            )
            self.stats["sn_archive_product_lift_value_only_terms"] += sum(
                lift.selection_lanes == ("value_product",) for lift in lift_result.lifts
            )
            self.stats["sn_archive_product_lift_gain_sum"] += sum(
                lift.sobolev_gain for lift in lift_result.lifts
            )
            self.stats["sn_archive_product_lift_correlation_sum"] += sum(
                lift.target_correlation for lift in lift_result.lifts
            )
        augmented_entries = (*original_entries, *lifted_entries)
        branch_results: list[dict[str, Any]] = []
        branches = (
            ("balanced", original_entries, 0),
            (
                "innovation_balanced",
                original_entries,
                self.sn_archive_beam_innovation_shortlist_size,
            ),
            ("product_balanced", augmented_entries, 0),
        )
        for label, entries, innovation_size in branches:
            self._build_archive_dual_beam_export_candidate(
                entries,
                value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
                innovation_shortlist_size=innovation_size,
                accuracy_mode=accuracy_mode,
            )
            branch_results.append(
                {
                    "branch": label,
                    "archive_terms": len(entries),
                    "best": (
                        None
                        if self.archive_export_candidate is None
                        else self.archive_export_candidate.copy()
                    ),
                    "pareto": (
                        None
                        if self.archive_export_pareto_candidate is None
                        else self.archive_export_pareto_candidate.copy()
                    ),
                    "path": copy.deepcopy(self.archive_export_path),
                }
            )

        def output_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
            fit = candidate.fit_result
            r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
            return (-r2, self._base_key(candidate))

        best_candidates = [
            result["best"] for result in branch_results if result["best"] is not None
        ]
        pareto_candidates = [
            result["pareto"]
            for result in branch_results
            if result["pareto"] is not None
        ]
        self.archive_export_candidate = (
            None if not best_candidates else min(best_candidates, key=output_key).copy()
        )
        if best_candidates:
            chosen_best_result = min(
                (result for result in branch_results if result["best"] is not None),
                key=lambda result: output_key(result["best"]),
            )
            self.stats[
                "sn_archive_product_accuracy_branch:" f"{chosen_best_result['branch']}"
            ] += int(accuracy_mode)
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=output_key).copy()
        )
        if pareto_candidates:
            chosen_result = min(
                (result for result in branch_results if result["pareto"] is not None),
                key=lambda result: output_key(result["pareto"]),
            )
            self.stats[
                "sn_archive_product_lift_branch:" f"{chosen_result['branch']}"
            ] += 1
        self.archive_export_path = [
            {
                "branch": result["branch"],
                "archive_terms": result["archive_terms"],
                "best": self._candidate_summary(result["best"]),
                "pareto": self._candidate_summary(result["pareto"]),
                "path": result["path"],
            }
            for result in branch_results
        ]
        selected = self.archive_export_pareto_candidate
        baseline = self.base_quality_best_ever
        if selected is None:
            self.stats["sn_archive_product_lift_base_retained"] += 1
            return
        selected.variation_type = "sn_archive_product_lift_export_selected"
        selected_fit = selected.fit_result
        baseline_fit = baseline.fit_result
        self.stats["sn_archive_product_lift_pareto_available"] += 1
        if selected_fit is not None and baseline_fit is not None:
            self.stats["sn_archive_product_lift_reward_gain_sum"] += float(
                selected.base_reward
            ) - float(baseline.base_reward)
            self.stats["sn_archive_product_lift_r2_gain_sum"] += float(
                selected_fit.r2
            ) - float(baseline_fit.r2)

    @staticmethod
    def _archive_product_conditioning_contexts(
        path: Sequence[dict[str, Any]],
    ) -> tuple[tuple[int, ...], ...]:
        """Extract accuracy leaders and Sobolev-diverse retained beam states."""

        contexts: list[tuple[int, ...]] = []
        for branch in path:
            if branch.get("branch") not in {"balanced", "innovation_balanced"}:
                continue
            for schedule in branch.get("path", ()):
                for depth in schedule.get("path", ()):
                    for position, retained in enumerate(depth.get("retained", ())):
                        if (
                            position != 0
                            and retained.get("decision_stage") != "sobolev_diversity"
                        ):
                            continue
                        selection = tuple(
                            sorted(int(index) for index in retained["archive_indices"])
                        )
                        if selection:
                            contexts.append(selection)
        return tuple(dict.fromkeys(contexts))

    @staticmethod
    def _archive_dual_conditioning_contexts(
        path: Sequence[dict[str, Any]],
    ) -> tuple[tuple[int, ...], ...]:
        """Extract leaders and diversity states from one dual-beam trace."""

        contexts: list[tuple[int, ...]] = []
        for schedule in path:
            for depth in schedule.get("path", ()):
                for position, retained in enumerate(depth.get("retained", ())):
                    if (
                        position != 0
                        and retained.get("decision_stage") != "sobolev_diversity"
                    ):
                        continue
                    selection = tuple(
                        sorted(int(index) for index in retained["archive_indices"])
                    )
                    if selection:
                        contexts.append(selection)
        return tuple(dict.fromkeys(contexts))

    def _build_archive_conditional_product_accuracy_candidate(self) -> None:
        """Extend BA with products screened inside retained beam contexts."""

        self.archive_conditional_product_entries = ()
        self.archive_conditional_product_contexts = ()
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        self._build_archive_product_lift_envelope_candidate(accuracy_mode=True)
        ba_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        ba_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        ba_path = copy.deepcopy(self.archive_export_path)
        contexts = self._archive_product_conditioning_contexts(ba_path)
        self.archive_conditional_product_contexts = contexts
        self.stats["sn_archive_conditional_product_contexts"] += len(contexts)
        self.stats["sn_archive_conditional_product_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        if (
            not original_entries
            or not contexts
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats[
                "sn_archive_conditional_product_fallback:context_unavailable"
            ] += 1
            return
        try:
            lift_result = lift_archive_conditional_product_interactions(
                entries=original_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_archive_conditional_product_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        lifted_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.archive_conditional_product_entries = lifted_entries
        self.stats[
            "sn_archive_conditional_product_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_archive_conditional_product_pairs_screened"
        ] += lift_result.pairs_screened
        self.stats[
            "sn_archive_conditional_product_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats["sn_archive_conditional_product_terms"] += len(lifted_entries)
        self.stats[
            "sn_archive_conditional_product_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_archive_conditional_product_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_archive_conditional_product_sobolev_lane_terms"] += sum(
            "sobolev_conditional_product" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_product_value_lane_terms"] += sum(
            "value_conditional_product" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_product_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_product_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        if not lifted_entries:
            self.stats["sn_archive_conditional_product_fallback:no_lift"] += 1
            return

        self._build_archive_dual_beam_export_candidate(
            (*original_entries, *lifted_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        conditional_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        conditional_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        conditional_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("ba_envelope", ba_candidate),
                ("conditional_product", conditional_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("ba_envelope", ba_pareto),
                ("conditional_product", conditional_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_archive_conditional_product_accuracy_branch:{label}"] += 1
            if (
                label == "conditional_product"
                and ba_candidate is not None
                and candidate.fit_result is not None
                and ba_candidate.fit_result is not None
            ):
                self.stats["sn_archive_conditional_product_gain_over_ba_sum"] += float(
                    candidate.fit_result.r2
                ) - float(ba_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "ba_envelope",
                "best": self._candidate_summary(ba_candidate),
                "pareto": self._candidate_summary(ba_pareto),
                "path": ba_path,
            },
            {
                "branch": "conditional_product",
                "archive_terms": len(original_entries) + len(lifted_entries),
                "contexts": [list(context) for context in contexts],
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "left_index": lift.left_index,
                        "right_index": lift.right_index,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(conditional_candidate),
                "pareto": self._candidate_summary(conditional_pareto),
                "path": conditional_path,
            },
        ]

    def _build_archive_iterated_conditional_product_accuracy_candidate(
        self,
    ) -> None:
        """Add one derived-term product round while retaining the BC envelope."""

        self.archive_iterated_product_entries = ()
        self.archive_iterated_product_contexts = ()
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        self._build_archive_conditional_product_accuracy_candidate()
        bc_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bc_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bc_path = copy.deepcopy(self.archive_export_path)
        first_round_entries = self.archive_conditional_product_entries
        conditional_path = next(
            (
                branch.get("path", ())
                for branch in bc_path
                if branch.get("branch") == "conditional_product"
            ),
            (),
        )
        contexts = self._archive_dual_conditioning_contexts(conditional_path)
        self.archive_iterated_product_contexts = contexts
        self.stats["sn_archive_iterated_product_contexts"] += len(contexts)
        self.stats["sn_archive_iterated_product_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        if (
            not original_entries
            or not first_round_entries
            or not contexts
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_archive_iterated_product_fallback:context_unavailable"] += 1
            return

        augmented_entries = (*original_entries, *first_round_entries)
        try:
            lift_result = lift_archive_conditional_product_interactions(
                entries=augmented_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                require_derived_index_at_least=len(original_entries),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_archive_iterated_product_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        second_round_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.archive_iterated_product_entries = second_round_entries
        self.stats[
            "sn_archive_iterated_product_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_archive_iterated_product_pairs_screened"
        ] += lift_result.pairs_screened
        self.stats[
            "sn_archive_iterated_product_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats["sn_archive_iterated_product_terms"] += len(second_round_entries)
        self.stats[
            "sn_archive_iterated_product_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_archive_iterated_product_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_archive_iterated_product_sobolev_lane_terms"] += sum(
            "sobolev_conditional_product" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_iterated_product_value_lane_terms"] += sum(
            "value_conditional_product" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_iterated_product_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_archive_iterated_product_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        if not second_round_entries:
            self.stats["sn_archive_iterated_product_fallback:no_lift"] += 1
            return

        self._build_archive_dual_beam_export_candidate(
            (*augmented_entries, *second_round_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        iterated_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        iterated_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        iterated_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bc_envelope", bc_candidate),
                ("iterated_product", iterated_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bc_envelope", bc_pareto),
                ("iterated_product", iterated_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_archive_iterated_product_accuracy_branch:{label}"] += 1
            if (
                label == "iterated_product"
                and bc_candidate is not None
                and candidate.fit_result is not None
                and bc_candidate.fit_result is not None
            ):
                self.stats["sn_archive_iterated_product_gain_over_bc_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bc_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bc_envelope",
                "best": self._candidate_summary(bc_candidate),
                "pareto": self._candidate_summary(bc_pareto),
                "path": bc_path,
            },
            {
                "branch": "iterated_product",
                "archive_terms": len(augmented_entries) + len(second_round_entries),
                "contexts": [list(context) for context in contexts],
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "left_index": lift.left_index,
                        "right_index": lift.right_index,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(iterated_candidate),
                "pareto": self._candidate_summary(iterated_pareto),
                "path": iterated_path,
            },
        ]

    def _build_archive_conditional_unary_accuracy_candidate(self) -> None:
        """Add bounded unary lifts while retaining the complete BD envelope."""

        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        self._build_archive_iterated_conditional_product_accuracy_candidate()
        bd_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bd_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bd_path = copy.deepcopy(self.archive_export_path)
        first_round_entries = self.archive_conditional_product_entries
        second_round_entries = self.archive_iterated_product_entries
        contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        pool = (*original_entries, *first_round_entries, *second_round_entries)
        self.stats["sn_archive_conditional_unary_contexts"] += len(contexts)
        self.stats["sn_archive_conditional_unary_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        if (
            not pool
            or not contexts
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_archive_conditional_unary_fallback:context_unavailable"] += 1
            return
        try:
            lift_result = lift_archive_conditional_unary_interactions(
                entries=pool,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                transforms=self.sn_archive_unary_transforms,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_archive_conditional_unary_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        unary_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.stats[
            "sn_archive_conditional_unary_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_archive_conditional_unary_candidates_screened"
        ] += lift_result.candidates_screened
        self.stats[
            "sn_archive_conditional_unary_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats["sn_archive_conditional_unary_terms"] += len(unary_entries)
        self.stats[
            "sn_archive_conditional_unary_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_archive_conditional_unary_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_archive_conditional_unary_sobolev_lane_terms"] += sum(
            "sobolev_conditional_unary" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_unary_value_lane_terms"] += sum(
            "value_conditional_unary" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_unary_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_unary_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        for lift in lift_result.lifts:
            self.stats[f"sn_archive_conditional_unary_transform:{lift.transform}"] += 1
        if not unary_entries:
            self.stats["sn_archive_conditional_unary_fallback:no_lift"] += 1
            return

        self._build_archive_dual_beam_export_candidate(
            (*pool, *unary_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        unary_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        unary_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        unary_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bd_envelope", bd_candidate),
                ("conditional_unary", unary_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bd_envelope", bd_pareto),
                ("conditional_unary", unary_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_archive_conditional_unary_accuracy_branch:{label}"] += 1
            if (
                label == "conditional_unary"
                and bd_candidate is not None
                and candidate.fit_result is not None
                and bd_candidate.fit_result is not None
            ):
                self.stats["sn_archive_conditional_unary_gain_over_bd_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bd_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bd_envelope",
                "best": self._candidate_summary(bd_candidate),
                "pareto": self._candidate_summary(bd_pareto),
                "path": bd_path,
            },
            {
                "branch": "conditional_unary",
                "archive_terms": len(pool) + len(unary_entries),
                "contexts": [list(context) for context in contexts],
                "transforms": list(self.sn_archive_unary_transforms),
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "source_index": lift.source_index,
                        "transform": lift.transform,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(unary_candidate),
                "pareto": self._candidate_summary(unary_pareto),
                "path": unary_path,
            },
        ]

    def _build_archive_conditional_rational_accuracy_candidate(self) -> None:
        """Add safe rational lifts while retaining the complete BE envelope."""

        self.archive_conditional_rational_entries = ()
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        self._build_archive_conditional_unary_accuracy_candidate()
        be_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        be_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        be_path = copy.deepcopy(self.archive_export_path)
        first_round_entries = self.archive_conditional_product_entries
        second_round_entries = self.archive_iterated_product_entries
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        construction_pool = (*original_entries, *first_round_entries)
        beam_pool = (*construction_pool, *second_round_entries)
        self.stats["sn_archive_conditional_rational_contexts"] += len(contexts)
        self.stats["sn_archive_conditional_rational_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        if (
            not construction_pool
            or not contexts
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats[
                "sn_archive_conditional_rational_fallback:context_unavailable"
            ] += 1
            return
        try:
            lift_result = lift_archive_conditional_rational_interactions(
                entries=construction_pool,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_archive_conditional_rational_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        rational_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.archive_conditional_rational_entries = rational_entries
        self.stats[
            "sn_archive_conditional_rational_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_archive_conditional_rational_candidates_screened"
        ] += lift_result.candidates_screened
        self.stats[
            "sn_archive_conditional_rational_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats["sn_archive_conditional_rational_terms"] += len(rational_entries)
        self.stats[
            "sn_archive_conditional_rational_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_archive_conditional_rational_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_archive_conditional_rational_sobolev_lane_terms"] += sum(
            "sobolev_conditional_rational" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_rational_value_lane_terms"] += sum(
            "value_conditional_rational" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_rational_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_rational_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        if not rational_entries:
            self.stats["sn_archive_conditional_rational_fallback:no_lift"] += 1
            return

        self._build_archive_dual_beam_export_candidate(
            (*beam_pool, *rational_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        rational_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        rational_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        rational_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("be_envelope", be_candidate),
                ("conditional_rational", rational_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("be_envelope", be_pareto),
                ("conditional_rational", rational_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_archive_conditional_rational_accuracy_branch:{label}"] += 1
            if (
                label == "conditional_rational"
                and be_candidate is not None
                and candidate.fit_result is not None
                and be_candidate.fit_result is not None
            ):
                self.stats["sn_archive_conditional_rational_gain_over_be_sum"] += float(
                    candidate.fit_result.r2
                ) - float(be_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "be_envelope",
                "best": self._candidate_summary(be_candidate),
                "pareto": self._candidate_summary(be_pareto),
                "path": be_path,
            },
            {
                "branch": "conditional_rational",
                "archive_terms": len(beam_pool) + len(rational_entries),
                "contexts": [list(context) for context in contexts],
                "template": "a/(1+b**2)",
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "numerator_index": lift.numerator_index,
                        "denominator_index": lift.denominator_index,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(rational_candidate),
                "pareto": self._candidate_summary(rational_pareto),
                "path": rational_path,
            },
        ]

    def _build_archive_conditional_affine_unary_accuracy_candidate(self) -> None:
        """Add ``f(a +/- b)`` lifts while retaining the complete BF envelope."""

        self.archive_conditional_affine_unary_entries = ()
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        self._build_archive_conditional_rational_accuracy_candidate()
        bf_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bf_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bf_path = copy.deepcopy(self.archive_export_path)
        first_round_entries = self.archive_conditional_product_entries
        second_round_entries = self.archive_iterated_product_entries
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        pool = (*original_entries, *first_round_entries, *second_round_entries)
        self.stats["sn_archive_conditional_affine_unary_contexts"] += len(contexts)
        self.stats["sn_archive_conditional_affine_unary_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        if (
            not pool
            or not contexts
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats[
                "sn_archive_conditional_affine_unary_fallback:context_unavailable"
            ] += 1
            return
        try:
            lift_result = lift_archive_conditional_affine_unary_interactions(
                entries=pool,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                transforms=self.sn_archive_unary_transforms,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_archive_conditional_affine_unary_fallback:"
                f"{type(error).__name__}"
            ] += 1
            return

        affine_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.archive_conditional_affine_unary_entries = affine_entries
        self.stats[
            "sn_archive_conditional_affine_unary_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_archive_conditional_affine_unary_candidates_screened"
        ] += lift_result.candidates_screened
        self.stats[
            "sn_archive_conditional_affine_unary_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats["sn_archive_conditional_affine_unary_terms"] += len(affine_entries)
        self.stats[
            "sn_archive_conditional_affine_unary_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_archive_conditional_affine_unary_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_archive_conditional_affine_unary_sobolev_lane_terms"] += sum(
            "sobolev_conditional_affine_unary" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_affine_unary_value_lane_terms"] += sum(
            "value_conditional_affine_unary" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_affine_unary_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_affine_unary_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        for lift in lift_result.lifts:
            self.stats[
                f"sn_archive_conditional_affine_unary_transform:{lift.transform}"
            ] += 1
            self.stats[
                "sn_archive_conditional_affine_unary_combination:"
                + ("sum" if lift.sign > 0 else "difference")
            ] += 1
        if not affine_entries:
            self.stats["sn_archive_conditional_affine_unary_fallback:no_lift"] += 1
            return

        self._build_archive_dual_beam_export_candidate(
            (*pool, *affine_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        affine_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        affine_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        affine_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bf_envelope", bf_candidate),
                ("conditional_affine_unary", affine_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bf_envelope", bf_pareto),
                ("conditional_affine_unary", affine_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[
                f"sn_archive_conditional_affine_unary_accuracy_branch:{label}"
            ] += 1
            if (
                label == "conditional_affine_unary"
                and bf_candidate is not None
                and candidate.fit_result is not None
                and bf_candidate.fit_result is not None
            ):
                self.stats[
                    "sn_archive_conditional_affine_unary_gain_over_bf_sum"
                ] += float(candidate.fit_result.r2) - float(bf_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bf_envelope",
                "best": self._candidate_summary(bf_candidate),
                "pareto": self._candidate_summary(bf_pareto),
                "path": bf_path,
            },
            {
                "branch": "conditional_affine_unary",
                "archive_terms": len(pool) + len(affine_entries),
                "contexts": [list(context) for context in contexts],
                "transforms": list(self.sn_archive_unary_transforms),
                "combinations": ["sum", "difference"],
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "left_index": lift.left_index,
                        "right_index": lift.right_index,
                        "sign": lift.sign,
                        "transform": lift.transform,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(affine_candidate),
                "pareto": self._candidate_summary(affine_pareto),
                "path": affine_path,
            },
        ]

    def _build_archive_conditional_radial_accuracy_candidate(self) -> None:
        """Add safe Euclidean-norm lifts while retaining the complete BG envelope."""

        self.archive_conditional_radial_entries = ()
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        self._build_archive_conditional_affine_unary_accuracy_candidate()
        bg_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bg_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bg_path = copy.deepcopy(self.archive_export_path)
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        pool = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        self.stats["sn_archive_conditional_radial_contexts"] += len(contexts)
        self.stats["sn_archive_conditional_radial_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        if (
            not pool
            or not contexts
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats[
                "sn_archive_conditional_radial_fallback:context_unavailable"
            ] += 1
            return
        try:
            lift_result = lift_archive_conditional_radial_interactions(
                entries=pool,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                component_joint_shortlist_size=(
                    self.sn_archive_interaction_joint_shortlist_size
                ),
                component_value_shortlist_size=(
                    self.sn_archive_interaction_value_shortlist_size
                ),
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_archive_conditional_radial_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        radial_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.archive_conditional_radial_entries = radial_entries
        self.stats[
            "sn_archive_conditional_radial_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_archive_conditional_radial_component_candidates"
        ] += lift_result.component_candidates
        self.stats[
            "sn_archive_conditional_radial_component_states_screened"
        ] += lift_result.component_states_screened
        self.stats[
            "sn_archive_conditional_radial_components_selected"
        ] += lift_result.components_selected
        self.stats[
            "sn_archive_conditional_radial_candidates"
        ] += lift_result.radial_candidates
        self.stats[
            "sn_archive_conditional_radial_states_screened"
        ] += lift_result.radial_states_screened
        self.stats["sn_archive_conditional_radial_terms"] += len(radial_entries)
        self.stats[
            "sn_archive_conditional_radial_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_archive_conditional_radial_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_archive_conditional_radial_sobolev_lane_terms"] += sum(
            "sobolev_conditional_radial" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_radial_value_lane_terms"] += sum(
            "value_conditional_radial" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_radial_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_archive_conditional_radial_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        if not radial_entries:
            self.stats["sn_archive_conditional_radial_fallback:no_lift"] += 1
            return

        self._build_archive_dual_beam_export_candidate(
            (*pool, *radial_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        radial_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        radial_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        radial_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bg_envelope", bg_candidate),
                ("conditional_radial", radial_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bg_envelope", bg_pareto),
                ("conditional_radial", radial_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_archive_conditional_radial_accuracy_branch:{label}"] += 1
            if (
                label == "conditional_radial"
                and bg_candidate is not None
                and candidate.fit_result is not None
                and bg_candidate.fit_result is not None
            ):
                self.stats["sn_archive_conditional_radial_gain_over_bg_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bg_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bg_envelope",
                "best": self._candidate_summary(bg_candidate),
                "pareto": self._candidate_summary(bg_pareto),
                "path": bg_path,
            },
            {
                "branch": "conditional_radial",
                "archive_terms": len(pool) + len(radial_entries),
                "contexts": [list(context) for context in contexts],
                "components_selected": lift_result.components_selected,
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "first_component": list(lift.first_component),
                        "second_component": list(lift.second_component),
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(radial_candidate),
                "pareto": self._candidate_summary(radial_pareto),
                "path": radial_path,
            },
        ]

    def _build_archive_feature_radial_accuracy_candidate(self) -> None:
        """Add an exhaustive raw-feature radial lane over the BH envelope."""

        self.archive_feature_radial_entries = ()
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        self._build_archive_conditional_radial_accuracy_candidate()
        bh_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bh_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bh_path = copy.deepcopy(self.archive_export_path)
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        pool = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        feature_indices: list[int] = []
        for name in self.feature_names:
            match = next(
                (index for index, entry in enumerate(pool) if entry.expression == name),
                None,
            )
            if match is not None:
                feature_indices.append(match)
        self.stats["sn_archive_feature_radial_contexts"] += len(contexts)
        self.stats["sn_archive_feature_radial_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_archive_feature_radial_features"] += len(feature_indices)
        if (
            not pool
            or not contexts
            or not feature_indices
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats[
                "sn_archive_feature_radial_fallback:feature_context_unavailable"
            ] += 1
            return
        component_count = len(feature_indices) * len(feature_indices)
        try:
            lift_result = lift_archive_conditional_radial_interactions(
                entries=pool,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                source_indices=feature_indices,
                component_joint_shortlist_size=component_count,
                component_value_shortlist_size=component_count,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_archive_feature_radial_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        feature_radial_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.archive_feature_radial_entries = feature_radial_entries
        self.stats[
            "sn_archive_feature_radial_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_archive_feature_radial_component_candidates"
        ] += lift_result.component_candidates
        self.stats[
            "sn_archive_feature_radial_component_states_screened"
        ] += lift_result.component_states_screened
        self.stats[
            "sn_archive_feature_radial_components_selected"
        ] += lift_result.components_selected
        self.stats[
            "sn_archive_feature_radial_candidates"
        ] += lift_result.radial_candidates
        self.stats[
            "sn_archive_feature_radial_states_screened"
        ] += lift_result.radial_states_screened
        self.stats["sn_archive_feature_radial_terms"] += len(feature_radial_entries)
        self.stats[
            "sn_archive_feature_radial_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_archive_feature_radial_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_archive_feature_radial_sobolev_lane_terms"] += sum(
            "sobolev_conditional_radial" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_feature_radial_value_lane_terms"] += sum(
            "value_conditional_radial" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_archive_feature_radial_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_archive_feature_radial_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        if not feature_radial_entries:
            self.stats["sn_archive_feature_radial_fallback:no_lift"] += 1
            return

        beam_entries: list[BasisArchiveEntry] = []
        seen_canonical: set[str] = set()
        for entry in (
            *pool,
            *self.archive_conditional_radial_entries,
            *feature_radial_entries,
        ):
            if entry.canonical in seen_canonical:
                continue
            seen_canonical.add(entry.canonical)
            beam_entries.append(entry)
        self.stats["sn_archive_feature_radial_beam_duplicates_removed"] += (
            len(pool)
            + len(self.archive_conditional_radial_entries)
            + len(feature_radial_entries)
            - len(beam_entries)
        )
        self._build_archive_dual_beam_export_candidate(
            tuple(beam_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        feature_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        feature_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        feature_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bh_envelope", bh_candidate),
                ("feature_radial", feature_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bh_envelope", bh_pareto),
                ("feature_radial", feature_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_archive_feature_radial_accuracy_branch:{label}"] += 1
            if (
                label == "feature_radial"
                and bh_candidate is not None
                and candidate.fit_result is not None
                and bh_candidate.fit_result is not None
            ):
                self.stats["sn_archive_feature_radial_gain_over_bh_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bh_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bh_envelope",
                "best": self._candidate_summary(bh_candidate),
                "pareto": self._candidate_summary(bh_pareto),
                "path": bh_path,
            },
            {
                "branch": "feature_radial",
                "feature_indices": feature_indices,
                "feature_names": list(self.feature_names),
                "components_selected": lift_result.components_selected,
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "first_component": list(lift.first_component),
                        "second_component": list(lift.second_component),
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(feature_candidate),
                "pareto": self._candidate_summary(feature_pareto),
                "path": feature_path,
            },
        ]

    def _build_direct_feature_basis_entries(
        self,
    ) -> tuple[BasisArchiveEntry, ...]:
        """Construct exact scale-free Sobolev columns for every input variable."""

        if self._active_X is None or self.geometry_indices is None:
            return ()
        X = np.asarray(self._active_X, dtype=float)
        indices = np.asarray(self.geometry_indices, dtype=int)
        sample_count = len(indices)
        dimension = len(self.feature_names)
        if sample_count < 1 or X.shape[1] != dimension:
            return ()
        input_scales = np.std(X, axis=0, ddof=0)
        value_factor = math.sqrt(self.sobolev_lambda_value / sample_count)
        gradient_factor = math.sqrt(
            self.sobolev_lambda_gradient / (sample_count * dimension)
        )
        entries: list[BasisArchiveEntry] = []
        for axis, name in enumerate(self.feature_names):
            expression, symbols = parse_expression(name, self.feature_names)
            terms = decompose_expand_mul(expression, symbols)
            if len(terms) != 1 or not terms[0].basis.free_symbols:
                continue
            blocks = [value_factor * X[indices, axis]]
            for gradient_axis in range(dimension):
                blocks.append(
                    np.full(
                        sample_count,
                        (
                            gradient_factor * input_scales[axis]
                            if gradient_axis == axis
                            else 0.0
                        ),
                        dtype=float,
                    )
                )
            signature = np.concatenate(blocks)
            norm = float(np.linalg.norm(signature))
            if not np.isfinite(norm) or norm <= np.finfo(float).eps:
                continue
            term = terms[0]
            entries.append(
                BasisArchiveEntry(
                    canonical=term.canonical,
                    expression=to_project_expression_string(term.basis),
                    signature=signature,
                    values=X[:, axis].copy(),
                    term_norm=norm,
                    source_coefficient=1.0,
                    source_amplitude=norm,
                    source_base_reward=0.0,
                    source_generation=-1,
                    source_base_rank=self.population_size,
                    source_candidate_id=-(axis + 1),
                )
            )
        return tuple(entries)

    def _build_direct_feature_radial_accuracy_candidate(self) -> None:
        """Add a radial lane whose coordinates do not depend on archive coverage."""

        self.direct_feature_basis_entries = ()
        self.direct_feature_radial_entries = ()
        self._build_archive_feature_radial_accuracy_candidate()
        bi_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bi_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bi_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        pool = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        direct_entries = self._build_direct_feature_basis_entries()
        self.direct_feature_basis_entries = direct_entries
        self.stats["sn_direct_feature_radial_contexts"] += len(contexts)
        self.stats["sn_direct_feature_radial_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_direct_feature_radial_features"] += len(direct_entries)
        if (
            not pool
            or not contexts
            or not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats[
                "sn_direct_feature_radial_fallback:feature_context_unavailable"
            ] += 1
            return

        radial_pool = (*pool, *direct_entries)
        direct_indices = tuple(range(len(pool), len(radial_pool)))
        component_count = len(direct_indices) * len(direct_indices)
        try:
            lift_result = lift_archive_conditional_radial_interactions(
                entries=radial_pool,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                source_indices=direct_indices,
                component_joint_shortlist_size=component_count,
                component_value_shortlist_size=component_count,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_direct_feature_radial_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        direct_radial_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.direct_feature_radial_entries = direct_radial_entries
        self.stats[
            "sn_direct_feature_radial_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_direct_feature_radial_component_candidates"
        ] += lift_result.component_candidates
        self.stats[
            "sn_direct_feature_radial_component_states_screened"
        ] += lift_result.component_states_screened
        self.stats[
            "sn_direct_feature_radial_components_selected"
        ] += lift_result.components_selected
        self.stats[
            "sn_direct_feature_radial_candidates"
        ] += lift_result.radial_candidates
        self.stats[
            "sn_direct_feature_radial_states_screened"
        ] += lift_result.radial_states_screened
        self.stats["sn_direct_feature_radial_terms"] += len(direct_radial_entries)
        self.stats[
            "sn_direct_feature_radial_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_direct_feature_radial_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_direct_feature_radial_sobolev_lane_terms"] += sum(
            "sobolev_conditional_radial" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_radial_value_lane_terms"] += sum(
            "value_conditional_radial" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_radial_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_radial_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        if not direct_radial_entries:
            self.stats["sn_direct_feature_radial_fallback:no_lift"] += 1
            return

        beam_entries: list[BasisArchiveEntry] = []
        seen_canonical: set[str] = set()
        for entry in (
            *pool,
            *self.archive_conditional_radial_entries,
            *self.archive_feature_radial_entries,
            *direct_radial_entries,
        ):
            if entry.canonical in seen_canonical:
                continue
            seen_canonical.add(entry.canonical)
            beam_entries.append(entry)
        self.stats["sn_direct_feature_radial_beam_duplicates_removed"] += (
            len(pool)
            + len(self.archive_conditional_radial_entries)
            + len(self.archive_feature_radial_entries)
            + len(direct_radial_entries)
            - len(beam_entries)
        )
        self._build_archive_dual_beam_export_candidate(
            tuple(beam_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        direct_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        direct_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        direct_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bi_envelope", bi_candidate),
                ("direct_feature_radial", direct_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bi_envelope", bi_pareto),
                ("direct_feature_radial", direct_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_direct_feature_radial_accuracy_branch:{label}"] += 1
            if (
                label == "direct_feature_radial"
                and bi_candidate is not None
                and candidate.fit_result is not None
                and bi_candidate.fit_result is not None
            ):
                self.stats["sn_direct_feature_radial_gain_over_bi_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bi_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bi_envelope",
                "best": self._candidate_summary(bi_candidate),
                "pareto": self._candidate_summary(bi_pareto),
                "path": bi_path,
            },
            {
                "branch": "direct_feature_radial",
                "feature_names": list(self.feature_names),
                "direct_feature_count": len(direct_entries),
                "components_selected": lift_result.components_selected,
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "first_component": list(lift.first_component),
                        "second_component": list(lift.second_component),
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(direct_candidate),
                "pareto": self._candidate_summary(direct_pareto),
                "path": direct_path,
            },
        ]

    def _build_direct_feature_affine_unary_accuracy_candidate(self) -> None:
        """Add bounded unary transforms of every raw-coordinate sum/difference."""

        self.direct_feature_affine_unary_entries = ()
        self._build_direct_feature_radial_accuracy_candidate()
        bj_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bj_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bj_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        pool = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_direct_feature_affine_unary_contexts"] += len(contexts)
        self.stats["sn_direct_feature_affine_unary_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_direct_feature_affine_unary_features"] += len(direct_entries)
        if (
            not pool
            or not contexts
            or not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats[
                "sn_direct_feature_affine_unary_fallback:feature_context_unavailable"
            ] += 1
            return

        affine_pool = (*pool, *direct_entries)
        direct_indices = tuple(range(len(pool), len(affine_pool)))
        try:
            lift_result = lift_archive_conditional_affine_unary_interactions(
                entries=affine_pool,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                source_indices=direct_indices,
                transforms=self.sn_archive_unary_transforms,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_direct_feature_affine_unary_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        direct_affine_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.direct_feature_affine_unary_entries = direct_affine_entries
        self.stats[
            "sn_direct_feature_affine_unary_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_direct_feature_affine_unary_candidates_screened"
        ] += lift_result.candidates_screened
        self.stats[
            "sn_direct_feature_affine_unary_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats["sn_direct_feature_affine_unary_terms"] += len(direct_affine_entries)
        self.stats[
            "sn_direct_feature_affine_unary_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_direct_feature_affine_unary_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_direct_feature_affine_unary_sobolev_lane_terms"] += sum(
            "sobolev_conditional_affine_unary" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_affine_unary_value_lane_terms"] += sum(
            "value_conditional_affine_unary" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_affine_unary_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_affine_unary_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        for lift in lift_result.lifts:
            self.stats[
                f"sn_direct_feature_affine_unary_transform:{lift.transform}"
            ] += 1
            self.stats[
                "sn_direct_feature_affine_unary_combination:"
                + ("sum" if lift.sign > 0 else "difference")
            ] += 1
        if not direct_affine_entries:
            self.stats["sn_direct_feature_affine_unary_fallback:no_lift"] += 1
            return

        beam_entries: list[BasisArchiveEntry] = []
        seen_canonical: set[str] = set()
        for entry in (
            *pool,
            *self.archive_conditional_radial_entries,
            *self.archive_feature_radial_entries,
            *self.direct_feature_radial_entries,
            *direct_affine_entries,
        ):
            if entry.canonical in seen_canonical:
                continue
            seen_canonical.add(entry.canonical)
            beam_entries.append(entry)
        self.stats["sn_direct_feature_affine_unary_beam_duplicates_removed"] += (
            len(pool)
            + len(self.archive_conditional_radial_entries)
            + len(self.archive_feature_radial_entries)
            + len(self.direct_feature_radial_entries)
            + len(direct_affine_entries)
            - len(beam_entries)
        )
        self._build_archive_dual_beam_export_candidate(
            tuple(beam_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        direct_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        direct_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        direct_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bj_envelope", bj_candidate),
                ("direct_feature_affine_unary", direct_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bj_envelope", bj_pareto),
                ("direct_feature_affine_unary", direct_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_direct_feature_affine_unary_accuracy_branch:{label}"] += 1
            if (
                label == "direct_feature_affine_unary"
                and bj_candidate is not None
                and candidate.fit_result is not None
                and bj_candidate.fit_result is not None
            ):
                self.stats["sn_direct_feature_affine_unary_gain_over_bj_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bj_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bj_envelope",
                "best": self._candidate_summary(bj_candidate),
                "pareto": self._candidate_summary(bj_pareto),
                "path": bj_path,
            },
            {
                "branch": "direct_feature_affine_unary",
                "feature_names": list(self.feature_names),
                "direct_feature_count": len(direct_entries),
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "left_index": lift.left_index,
                        "right_index": lift.right_index,
                        "sign": lift.sign,
                        "transform": lift.transform,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(direct_candidate),
                "pareto": self._candidate_summary(direct_pareto),
                "path": direct_path,
            },
        ]

    def _build_direct_feature_product_accuracy_candidate(self) -> None:
        """Add a complete raw-coordinate quadratic lane to the BK envelope."""

        self.direct_feature_product_entries = ()
        self._build_direct_feature_affine_unary_accuracy_candidate()
        bk_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bk_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bk_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        contexts = tuple(
            dict.fromkeys(
                self.archive_iterated_product_contexts
                or self.archive_conditional_product_contexts
            )
        )
        pool = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_direct_feature_product_contexts"] += len(contexts)
        self.stats["sn_direct_feature_product_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_direct_feature_product_features"] += len(direct_entries)
        if (
            not pool
            or not contexts
            or not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats[
                "sn_direct_feature_product_fallback:feature_context_unavailable"
            ] += 1
            return

        product_pool = (*pool, *direct_entries)
        direct_indices = tuple(range(len(pool), len(product_pool)))
        try:
            lift_result = lift_archive_conditional_product_interactions(
                entries=product_pool,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                source_indices=direct_indices,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                "sn_direct_feature_product_fallback:" f"{type(error).__name__}"
            ] += 1
            return

        direct_product_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.direct_feature_product_entries = direct_product_entries
        self.stats[
            "sn_direct_feature_product_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats[
            "sn_direct_feature_product_pairs_screened"
        ] += lift_result.pairs_screened
        self.stats[
            "sn_direct_feature_product_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats["sn_direct_feature_product_terms"] += len(direct_product_entries)
        self.stats[
            "sn_direct_feature_product_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_direct_feature_product_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_direct_feature_product_sobolev_lane_terms"] += sum(
            "sobolev_conditional_product" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_product_value_lane_terms"] += sum(
            "value_conditional_product" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_product_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_product_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        self.stats["sn_direct_feature_product_square_terms"] += sum(
            lift.left_index == lift.right_index for lift in lift_result.lifts
        )
        if not direct_product_entries:
            self.stats["sn_direct_feature_product_fallback:no_lift"] += 1
            return

        beam_entries: list[BasisArchiveEntry] = []
        seen_canonical: set[str] = set()
        for entry in (
            *pool,
            *direct_entries,
            *self.archive_conditional_radial_entries,
            *self.archive_feature_radial_entries,
            *self.direct_feature_radial_entries,
            *self.direct_feature_affine_unary_entries,
            *direct_product_entries,
        ):
            if entry.canonical in seen_canonical:
                continue
            seen_canonical.add(entry.canonical)
            beam_entries.append(entry)
        self.stats["sn_direct_feature_product_beam_duplicates_removed"] += (
            len(pool)
            + len(direct_entries)
            + len(self.archive_conditional_radial_entries)
            + len(self.archive_feature_radial_entries)
            + len(self.direct_feature_radial_entries)
            + len(self.direct_feature_affine_unary_entries)
            + len(direct_product_entries)
            - len(beam_entries)
        )
        self._build_archive_dual_beam_export_candidate(
            tuple(beam_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        direct_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        direct_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        direct_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bk_envelope", bk_candidate),
                ("direct_feature_product", direct_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bk_envelope", bk_pareto),
                ("direct_feature_product", direct_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_direct_feature_product_accuracy_branch:{label}"] += 1
            if (
                label == "direct_feature_product"
                and bk_candidate is not None
                and candidate.fit_result is not None
                and bk_candidate.fit_result is not None
            ):
                self.stats["sn_direct_feature_product_gain_over_bk_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bk_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bk_envelope",
                "best": self._candidate_summary(bk_candidate),
                "pareto": self._candidate_summary(bk_pareto),
                "path": bk_path,
            },
            {
                "branch": "direct_feature_product",
                "feature_names": list(self.feature_names),
                "direct_feature_count": len(direct_entries),
                "contexts": [list(context) for context in contexts],
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "left_index": lift.left_index,
                        "right_index": lift.right_index,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(direct_candidate),
                "pareto": self._candidate_summary(direct_pareto),
                "path": direct_path,
            },
        ]

    def _build_direct_phase_accuracy_candidate(self) -> None:
        """Add a complete scaled phase library to the Stage BL envelope."""

        self.direct_phase_entries = ()
        self._build_direct_feature_product_accuracy_candidate()
        bl_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bl_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bl_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        conditioning_entries = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_direct_phase_contexts"] += len(contexts)
        self.stats["sn_direct_phase_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_direct_phase_features"] += len(direct_entries)
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_direct_phase_fallback:feature_or_data_unavailable"] += 1
            return

        try:
            lift_result = lift_direct_conditional_phase_features(
                direct_entries=direct_entries,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                scales=self.sn_direct_phase_scales,
                transforms=self.sn_archive_unary_transforms,
                include_squares=self.sn_direct_phase_include_squares,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_direct_phase_fallback:{type(error).__name__}"] += 1
            return

        phase_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.direct_phase_entries = phase_entries
        self.stats["sn_direct_phase_sources"] += lift_result.source_count
        self.stats[
            "sn_direct_phase_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats["sn_direct_phase_contexts_screened"] += lift_result.contexts_screened
        self.stats[
            "sn_direct_phase_candidate_context_states"
        ] += lift_result.candidate_context_states
        self.stats["sn_direct_phase_terms"] += len(phase_entries)
        self.stats[
            "sn_direct_phase_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_direct_phase_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_direct_phase_sobolev_lane_terms"] += sum(
            "sobolev_conditional_phase" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_direct_phase_value_lane_terms"] += sum(
            "value_conditional_phase" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_direct_phase_squared_terms"] += sum(
            lift.squared for lift in lift_result.lifts
        )
        self.stats["sn_direct_phase_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_direct_phase_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        for lift in lift_result.lifts:
            self.stats[f"sn_direct_phase_source_kind:{lift.source_kind}"] += 1
            self.stats[f"sn_direct_phase_transform:{lift.transform}"] += 1
        if not phase_entries:
            self.stats["sn_direct_phase_fallback:no_lift"] += 1
            return

        beam_entries: list[BasisArchiveEntry] = []
        seen_canonical: set[str] = set()
        for entry in (
            *conditioning_entries,
            *direct_entries,
            *self.archive_conditional_radial_entries,
            *self.archive_feature_radial_entries,
            *self.direct_feature_radial_entries,
            *self.direct_feature_affine_unary_entries,
            *self.direct_feature_product_entries,
            *phase_entries,
        ):
            if entry.canonical in seen_canonical:
                continue
            seen_canonical.add(entry.canonical)
            beam_entries.append(entry)
        self.stats["sn_direct_phase_beam_duplicates_removed"] += (
            len(conditioning_entries)
            + len(direct_entries)
            + len(self.archive_conditional_radial_entries)
            + len(self.archive_feature_radial_entries)
            + len(self.direct_feature_radial_entries)
            + len(self.direct_feature_affine_unary_entries)
            + len(self.direct_feature_product_entries)
            + len(phase_entries)
            - len(beam_entries)
        )
        self._build_archive_dual_beam_export_candidate(
            tuple(beam_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        phase_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        phase_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        phase_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bl_envelope", bl_candidate),
                ("direct_phase", phase_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bl_envelope", bl_pareto),
                ("direct_phase", phase_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_direct_phase_accuracy_branch:{label}"] += 1
            if (
                label == "direct_phase"
                and bl_candidate is not None
                and candidate.fit_result is not None
                and bl_candidate.fit_result is not None
            ):
                self.stats["sn_direct_phase_gain_over_bl_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bl_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bl_envelope",
                "best": self._candidate_summary(bl_candidate),
                "pareto": self._candidate_summary(bl_pareto),
                "path": bl_path,
            },
            {
                "branch": "direct_phase",
                "feature_names": list(self.feature_names),
                "direct_feature_count": len(direct_entries),
                "scales": list(self.sn_direct_phase_scales),
                "transforms": list(self.sn_archive_unary_transforms),
                "include_squares": self.sn_direct_phase_include_squares,
                "contexts": [list(context) for context in contexts],
                "source_count": lift_result.source_count,
                "numeric_candidates": lift_result.numeric_candidates,
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "source_kind": lift.source_kind,
                        "source_expression": lift.source_expression,
                        "scale": lift.scale,
                        "transform": lift.transform,
                        "squared": lift.squared,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(phase_candidate),
                "pareto": self._candidate_summary(phase_pareto),
                "path": phase_path,
            },
        ]

    def _build_direct_phase_interaction_accuracy_candidate(self) -> None:
        """Add a bounded second compositional layer to the Stage BM envelope."""

        self.direct_phase_interaction_entries = ()
        self._build_direct_phase_accuracy_candidate()
        bm_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bm_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bm_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        conditioning_entries = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        retained_phases = self.direct_phase_entries
        partner_entries = (
            *self.archive_conditional_radial_entries,
            *self.archive_feature_radial_entries,
            *self.direct_feature_radial_entries,
            *self.direct_feature_affine_unary_entries,
            *self.direct_feature_product_entries,
        )
        self.stats["sn_phase_interaction_contexts"] += len(contexts)
        self.stats["sn_phase_interaction_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        if (
            not retained_phases
            or not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_phase_interaction_fallback:phase_or_data_unavailable"] += 1
            return

        try:
            lift_result = lift_direct_conditional_phase_interactions(
                retained_phase_entries=retained_phases,
                direct_entries=direct_entries,
                partner_entries=partner_entries,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                scales=self.sn_direct_phase_scales,
                transforms=self.sn_archive_unary_transforms,
                include_squares=self.sn_direct_phase_include_squares,
                joint_shortlist_size=(self.sn_archive_interaction_joint_shortlist_size),
                value_shortlist_size=(self.sn_archive_interaction_value_shortlist_size),
                batch_size=self.sn_phase_interaction_batch_size,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_phase_interaction_fallback:{type(error).__name__}"] += 1
            return

        interaction_entries = tuple(lift.entry for lift in lift_result.lifts)
        self.direct_phase_interaction_entries = interaction_entries
        self.stats[
            "sn_phase_interaction_retained_phases"
        ] += lift_result.retained_phase_count
        self.stats[
            "sn_phase_interaction_coordinate_phases"
        ] += lift_result.coordinate_phase_count
        self.stats[
            "sn_phase_interaction_algebraic_partners"
        ] += lift_result.algebraic_partner_count
        self.stats[
            "sn_phase_interaction_numeric_candidates"
        ] += lift_result.numeric_candidates
        self.stats[
            "sn_phase_interaction_candidate_context_states"
        ] += lift_result.candidate_context_states
        self.stats[
            "sn_phase_interaction_contexts_screened"
        ] += lift_result.contexts_screened
        self.stats["sn_phase_interaction_terms"] += len(interaction_entries)
        self.stats[
            "sn_phase_interaction_numeric_failures"
        ] += lift_result.numeric_failures
        self.stats[
            "sn_phase_interaction_construction_failures"
        ] += lift_result.construction_failures
        self.stats[
            "sn_phase_interaction_canonical_duplicates"
        ] += lift_result.canonical_duplicates
        self.stats["sn_phase_interaction_sobolev_lane_terms"] += sum(
            "sobolev_conditional_phase_interaction" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_phase_interaction_value_lane_terms"] += sum(
            "value_conditional_phase_interaction" in lift.selection_lanes
            for lift in lift_result.lifts
        )
        self.stats["sn_phase_interaction_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.lifts
        )
        self.stats["sn_phase_interaction_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.lifts
        )
        for lift in lift_result.lifts:
            self.stats[f"sn_phase_interaction_kind:{lift.interaction_kind}"] += 1
        if not interaction_entries:
            self.stats["sn_phase_interaction_fallback:no_lift"] += 1
            return

        beam_entries: list[BasisArchiveEntry] = []
        seen_canonical: set[str] = set()
        for entry in (
            *conditioning_entries,
            *direct_entries,
            *self.archive_conditional_radial_entries,
            *self.archive_feature_radial_entries,
            *self.direct_feature_radial_entries,
            *self.direct_feature_affine_unary_entries,
            *self.direct_feature_product_entries,
            *retained_phases,
            *interaction_entries,
        ):
            if entry.canonical in seen_canonical:
                continue
            seen_canonical.add(entry.canonical)
            beam_entries.append(entry)
        self.stats["sn_phase_interaction_beam_duplicates_removed"] += (
            len(conditioning_entries)
            + len(direct_entries)
            + len(self.archive_conditional_radial_entries)
            + len(self.archive_feature_radial_entries)
            + len(self.direct_feature_radial_entries)
            + len(self.direct_feature_affine_unary_entries)
            + len(self.direct_feature_product_entries)
            + len(retained_phases)
            + len(interaction_entries)
            - len(beam_entries)
        )
        self._build_archive_dual_beam_export_candidate(
            tuple(beam_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        interaction_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        interaction_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        interaction_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bm_envelope", bm_candidate),
                ("phase_interaction", interaction_candidate),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bm_envelope", bm_pareto),
                ("phase_interaction", interaction_pareto),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_phase_interaction_accuracy_branch:{label}"] += 1
            if (
                label == "phase_interaction"
                and bm_candidate is not None
                and candidate.fit_result is not None
                and bm_candidate.fit_result is not None
            ):
                self.stats["sn_phase_interaction_gain_over_bm_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bm_candidate.fit_result.r2)
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bm_envelope",
                "best": self._candidate_summary(bm_candidate),
                "pareto": self._candidate_summary(bm_pareto),
                "path": bm_path,
            },
            {
                "branch": "phase_interaction",
                "feature_names": list(self.feature_names),
                "batch_size": self.sn_phase_interaction_batch_size,
                "retained_phase_count": lift_result.retained_phase_count,
                "coordinate_phase_count": lift_result.coordinate_phase_count,
                "algebraic_partner_count": lift_result.algebraic_partner_count,
                "numeric_candidates": lift_result.numeric_candidates,
                "contexts": [list(context) for context in contexts],
                "lifts": [
                    {
                        "canonical": lift.entry.canonical,
                        "interaction_kind": lift.interaction_kind,
                        "phase_expression": lift.phase_expression,
                        "partner_expression": lift.partner_expression,
                        "denominator_expression": lift.denominator_expression,
                        "amplitude_expression": lift.amplitude_expression,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                        "joint_context": lift.joint_context,
                        "value_context": lift.value_context,
                    }
                    for lift in lift_result.lifts
                ],
                "best": self._candidate_summary(interaction_candidate),
                "pareto": self._candidate_summary(interaction_pareto),
                "path": interaction_path,
            },
        ]

    def _build_direct_radical_phase_accuracy_candidate(self) -> None:
        """Add amplitude-conditioned algebraic radicals to the Stage BN envelope."""

        self.direct_radical_amplitude_entries = ()
        self.direct_radical_composite_entries = ()
        self._build_direct_phase_interaction_accuracy_candidate()
        bn_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bn_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bn_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        conditioning_entries = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        retained_phases = self.direct_phase_entries
        self.stats["sn_radical_contexts"] += len(contexts)
        self.stats["sn_radical_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_radical_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_radical_max_dimension:
            self.stats["sn_radical_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or not retained_phases
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_radical_fallback:phase_or_data_unavailable"] += 1
            return

        try:
            lift_result = lift_direct_amplitude_conditioned_radicals(
                direct_entries=direct_entries,
                retained_phase_entries=retained_phases,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                radical_scales=self.sn_radical_scales,
                phase_scales=self.sn_direct_phase_scales,
                phase_transforms=self.sn_archive_unary_transforms,
                phase_include_squares=self.sn_direct_phase_include_squares,
                phase_anchor_limit=self.sn_radical_phase_anchor_limit,
                amplitude_joint_shortlist_size=(
                    self.sn_radical_amplitude_joint_shortlist_size
                ),
                amplitude_value_shortlist_size=(
                    self.sn_radical_amplitude_value_shortlist_size
                ),
                composite_joint_shortlist_size=(
                    self.sn_archive_interaction_joint_shortlist_size
                ),
                composite_value_shortlist_size=(
                    self.sn_archive_interaction_value_shortlist_size
                ),
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_radical_fallback:{type(error).__name__}"] += 1
            return

        amplitude_entries = tuple(lift.entry for lift in lift_result.amplitude_lifts)
        composite_entries = tuple(lift.entry for lift in lift_result.composite_lifts)
        self.direct_radical_amplitude_entries = amplitude_entries
        self.direct_radical_composite_entries = composite_entries
        self.stats[
            "sn_radical_amplitude_candidates"
        ] += lift_result.amplitude_candidates
        self.stats["sn_radical_amplitude_terms"] += len(amplitude_entries)
        self.stats[
            "sn_radical_ratio_candidates"
        ] += lift_result.radical_ratio_candidates
        self.stats["sn_radical_positive_radicals"] += lift_result.positive_radicals
        self.stats["sn_radical_phase_anchors"] += lift_result.phase_anchors
        self.stats[
            "sn_radical_amplitude_phase_candidates"
        ] += lift_result.amplitude_phase_candidates
        self.stats[
            "sn_radical_amplitude_phase_failures"
        ] += lift_result.amplitude_phase_failures
        self.stats[
            "sn_radical_composite_candidates"
        ] += lift_result.composite_candidates
        self.stats["sn_radical_composite_terms"] += len(composite_entries)
        self.stats[
            "sn_radical_amplitude_context_states"
        ] += lift_result.amplitude_context_states
        self.stats[
            "sn_radical_composite_context_states"
        ] += lift_result.composite_context_states
        self.stats[
            "sn_radical_positivity_rejections"
        ] += lift_result.positivity_rejections
        self.stats["sn_radical_numeric_failures"] += lift_result.numeric_failures
        self.stats[
            "sn_radical_construction_failures"
        ] += lift_result.construction_failures
        self.stats["sn_radical_canonical_reuses"] += lift_result.canonical_reuses
        self.stats["sn_radical_amplitude_sobolev_lane_terms"] += sum(
            "sobolev_conditional_amplitude" in lift.selection_lanes
            for lift in lift_result.amplitude_lifts
        )
        self.stats["sn_radical_amplitude_value_lane_terms"] += sum(
            "value_global_amplitude" in lift.selection_lanes
            for lift in lift_result.amplitude_lifts
        )
        self.stats["sn_radical_composite_sobolev_lane_terms"] += sum(
            "sobolev_amplitude_conditioned_radical" in lift.selection_lanes
            for lift in lift_result.composite_lifts
        )
        self.stats["sn_radical_composite_value_lane_terms"] += sum(
            "value_amplitude_conditioned_radical" in lift.selection_lanes
            for lift in lift_result.composite_lifts
        )
        self.stats["sn_radical_composite_gain_sum"] += sum(
            lift.sobolev_gain for lift in lift_result.composite_lifts
        )
        self.stats["sn_radical_composite_correlation_sum"] += sum(
            lift.target_correlation for lift in lift_result.composite_lifts
        )
        if not composite_entries:
            self.stats["sn_radical_fallback:no_composite_lift"] += 1
            return

        compact_candidates: list[ControlledIndividual] = []
        compact_path: list[dict[str, Any]] = []
        positive_coordinate_domain = all(
            np.all(np.asarray(entry.values, dtype=float) > 0.0)
            for entry in direct_entries
        )
        if positive_coordinate_domain:
            amplitudes_by_canonical = {
                lift.entry.canonical: lift for lift in lift_result.amplitude_lifts
            }
            seen_compact: set[str] = set()
            generation = len(self.records)
            for index, composite in enumerate(lift_result.composite_lifts):
                amplitude = amplitudes_by_canonical.get(composite.amplitude_canonical)
                if amplitude is None:
                    continue
                try:
                    expression_text = compact_positive_amplitude_radical_expression(
                        amplitude=amplitude,
                        composite=composite,
                        variable_names=self.feature_names,
                    )
                    tree = nd.parse(expression_text)
                except (TypeError, ValueError, KeyError):
                    self.stats["sn_radical_compact_construction_failures"] += 1
                    continue
                canonical = tree.to_str(number_format=".17g")
                if canonical in seen_compact:
                    self.stats["sn_radical_compact_duplicates"] += 1
                    continue
                seen_compact.add(canonical)
                if len(tree) > self.max_len:
                    self.stats["sn_radical_compact_length_rejections"] += 1
                    continue
                candidate = ControlledIndividual(
                    tree,
                    candidate_id=self._proposal_id(generation + 1, 800000 + index),
                    generation=generation,
                    slot=-1,
                    variation_type="sn_radical_compact_candidate",
                )
                self._evaluate_individual(candidate, self._active_X, self._active_y)
                self.stats["sn_radical_compact_candidates_evaluated"] += 1
                self.stats["sn_radical_compact_valid_candidates"] += int(
                    candidate.valid
                )
                compact_path.append(
                    {
                        "expression": canonical,
                        "tree_length": len(tree),
                        "amplitude_canonical": composite.amplitude_canonical,
                        "phase_canonical": composite.phase_canonical,
                        "radical_expression": composite.radical_expression,
                        "candidate": self._candidate_summary(candidate),
                    }
                )
                if candidate.valid:
                    compact_candidates.append(candidate)
        else:
            self.stats["sn_radical_compact_skips:nonpositive_coordinates"] += 1

        beam_entries: list[BasisArchiveEntry] = []
        seen_canonical: set[str] = set()
        for entry in (
            *conditioning_entries,
            *direct_entries,
            *self.archive_conditional_radial_entries,
            *self.archive_feature_radial_entries,
            *self.direct_feature_radial_entries,
            *self.direct_feature_affine_unary_entries,
            *self.direct_feature_product_entries,
            *retained_phases,
            *self.direct_phase_interaction_entries,
            *amplitude_entries,
            *composite_entries,
        ):
            if entry.canonical in seen_canonical:
                continue
            seen_canonical.add(entry.canonical)
            beam_entries.append(entry)
        self._build_archive_dual_beam_export_candidate(
            tuple(beam_entries),
            value_shortlist_size=self.sn_archive_beam_value_shortlist_size,
            innovation_shortlist_size=(self.sn_archive_beam_innovation_shortlist_size),
            accuracy_mode=True,
        )
        radical_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        radical_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        radical_path = copy.deepcopy(self.archive_export_path)
        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bn_envelope", bn_candidate),
                ("radical_phase", radical_candidate),
                *(("compact_radical", candidate) for candidate in compact_candidates),
            )
            if candidate is not None
        ]
        pareto_candidates = [
            (label, candidate)
            for label, candidate in (
                ("bn_envelope", bn_pareto),
                ("radical_phase", radical_pareto),
                *(("compact_radical", candidate) for candidate in compact_candidates),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_radical_accuracy_branch:{label}"] += 1
            if (
                label in {"radical_phase", "compact_radical"}
                and bn_candidate is not None
                and candidate.fit_result is not None
                and bn_candidate.fit_result is not None
            ):
                self.stats["sn_radical_gain_over_bn_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bn_candidate.fit_result.r2)
                self.stats["sn_radical_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=lambda item: self._accuracy_key(item[1]))[
                1
            ].copy()
        )
        self.archive_export_path = [
            {
                "branch": "bn_envelope",
                "best": self._candidate_summary(bn_candidate),
                "pareto": self._candidate_summary(bn_pareto),
                "path": bn_path,
            },
            {
                "branch": "radical_phase",
                "feature_names": list(self.feature_names),
                "scales": list(self.sn_radical_scales),
                "phase_anchor_limit": self.sn_radical_phase_anchor_limit,
                "amplitude_candidates": lift_result.amplitude_candidates,
                "radical_ratio_candidates": (lift_result.radical_ratio_candidates),
                "positive_radicals": lift_result.positive_radicals,
                "composite_candidates": lift_result.composite_candidates,
                "amplitudes": [
                    {
                        "canonical": lift.entry.canonical,
                        "numerator_indices": list(lift.numerator_indices),
                        "denominator_indices": list(lift.denominator_indices),
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                    }
                    for lift in lift_result.amplitude_lifts
                ],
                "composites": [
                    {
                        "canonical": lift.entry.canonical,
                        "amplitude_canonical": lift.amplitude_canonical,
                        "phase_canonical": lift.phase_canonical,
                        "phase_expression": lift.phase_expression,
                        "radical_expression": lift.radical_expression,
                        "radical_scale": lift.radical_scale,
                        "sobolev_gain": lift.sobolev_gain,
                        "target_correlation": lift.target_correlation,
                        "selection_lanes": list(lift.selection_lanes),
                    }
                    for lift in lift_result.composite_lifts
                ],
                "compact_candidates": compact_path,
                "best": self._candidate_summary(radical_candidate),
                "pareto": self._candidate_summary(radical_pareto),
                "path": radical_path,
            },
        ]

    def _build_shared_phase_rational_accuracy_candidate(self) -> None:
        """Add a Sobolev-screened shared-phase Möbius pursuit to the BO envelope."""

        self._build_direct_radical_phase_accuracy_candidate()
        bo_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bo_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bo_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        conditioning_entries = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_shared_phase_contexts"] += len(contexts)
        self.stats["sn_shared_phase_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_shared_phase_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_shared_phase_max_dimension:
            self.stats["sn_shared_phase_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_shared_phase_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_shared_phase_rationals(
                direct_entries=direct_entries,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                phase_scales=self.sn_direct_phase_scales,
                phase_transforms=self.sn_archive_unary_transforms,
                phase_include_squares=self.sn_direct_phase_include_squares,
                mobius_shifts=self.sn_shared_phase_mobius_shifts,
                max_numerator_degree=(self.sn_shared_phase_max_numerator_degree),
                max_denominator_degree=(self.sn_shared_phase_max_denominator_degree),
                backbone_value_pool_size=(
                    self.sn_shared_phase_backbone_value_pool_size
                ),
                backbone_shortlist_size=(self.sn_shared_phase_backbone_shortlist_size),
                modulated_shortlist_size=(
                    self.sn_shared_phase_modulated_shortlist_size
                ),
                complement_pool_size=(self.sn_shared_phase_complement_pool_size),
                complement_shortlist_size=(
                    self.sn_shared_phase_complement_shortlist_size
                ),
                proposal_limit=self.sn_shared_phase_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_shared_phase_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_shared_phase_monomial_candidates"] += pursuit.monomial_candidates
        self.stats["sn_shared_phase_phase_candidates"] += pursuit.phase_candidates
        self.stats["sn_shared_phase_backbone_candidates"] += pursuit.backbone_candidates
        self.stats["sn_shared_phase_backbone_value_pool"] += pursuit.backbone_value_pool
        self.stats["sn_shared_phase_backbone_shortlist"] += pursuit.backbone_shortlist
        self.stats["sn_shared_phase_mobius_factors"] += pursuit.mobius_factors
        self.stats[
            "sn_shared_phase_modulated_candidates"
        ] += pursuit.modulated_candidates
        self.stats["sn_shared_phase_modulated_shortlist"] += pursuit.modulated_shortlist
        self.stats[
            "sn_shared_phase_complement_candidates"
        ] += pursuit.complement_candidates
        self.stats["sn_shared_phase_pair_candidates"] += pursuit.pair_candidates
        self.stats["sn_shared_phase_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_shared_phase_proposals"] += len(pursuit.proposals)
        self.stats["sn_shared_phase_sobolev_backbone_proposals"] += sum(
            "sobolev_backbone" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_shared_phase_sobolev_mobius_proposals"] += sum(
            "sobolev_mobius" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_shared_phase_sobolev_complement_proposals"] += sum(
            "sobolev_complement" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_shared_phase_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_shared_phase_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_shared_phase_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 900000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_shared_phase_rational_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_shared_phase_candidates_evaluated"] += 1
            self.stats["sn_shared_phase_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "phase_expression": proposal.phase_expression,
                    "backbone_expression": proposal.backbone_expression,
                    "modulated_expression": proposal.modulated_expression,
                    "complement_expression": proposal.complement_expression,
                    "backbone_exponents": list(proposal.backbone_exponents),
                    "complement_exponents": list(proposal.complement_exponents),
                    "mobius_axis": proposal.mobius_axis,
                    "mobius_numerator_shift": (proposal.mobius_numerator_shift),
                    "mobius_denominator_shift": (proposal.mobius_denominator_shift),
                    "screen_training_r2": proposal.training_r2,
                    "backbone_target_correlation": (
                        proposal.backbone_target_correlation
                    ),
                    "modulated_target_correlation": (
                        proposal.modulated_target_correlation
                    ),
                    "modulated_sobolev_gain": (proposal.modulated_sobolev_gain),
                    "complement_target_correlation": (
                        proposal.complement_target_correlation
                    ),
                    "complement_sobolev_gain": (proposal.complement_sobolev_gain),
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bo_envelope", bo_candidate),
                *(
                    ("shared_phase_rational", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_shared_phase_accuracy_branch:{label}"] += 1
            if (
                label == "shared_phase_rational"
                and bo_candidate is not None
                and candidate.fit_result is not None
                and bo_candidate.fit_result is not None
            ):
                self.stats["sn_shared_phase_gain_over_bo_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bo_candidate.fit_result.r2)
                self.stats["sn_shared_phase_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bo_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bo_envelope",
                "best": self._candidate_summary(bo_candidate),
                "pareto": self._candidate_summary(bo_pareto),
                "path": bo_path,
            },
            {
                "branch": "shared_phase_rational",
                "mobius_shifts": list(self.sn_shared_phase_mobius_shifts),
                "max_numerator_degree": (self.sn_shared_phase_max_numerator_degree),
                "max_denominator_degree": (self.sn_shared_phase_max_denominator_degree),
                "monomial_candidates": pursuit.monomial_candidates,
                "phase_candidates": pursuit.phase_candidates,
                "backbone_candidates": pursuit.backbone_candidates,
                "backbone_value_pool": pursuit.backbone_value_pool,
                "backbone_shortlist": pursuit.backbone_shortlist,
                "mobius_factors": pursuit.mobius_factors,
                "modulated_candidates": pursuit.modulated_candidates,
                "modulated_shortlist": pursuit.modulated_shortlist,
                "complement_candidates": pursuit.complement_candidates,
                "pair_candidates": pursuit.pair_candidates,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_damped_exponential_accuracy_candidate(self) -> None:
        """Add bounded exponential-factor pursuit to the Stage BP envelope."""

        self._build_shared_phase_rational_accuracy_candidate()
        bp_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bp_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bp_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        conditioning_entries = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_exponential_contexts"] += len(contexts)
        self.stats["sn_exponential_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_exponential_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_exponential_max_dimension:
            self.stats["sn_exponential_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_exponential_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_damped_exponentials(
                direct_entries=direct_entries,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                exponent_scales=self.sn_exponential_scales,
                max_numerator_degree=(self.sn_exponential_max_numerator_degree),
                max_denominator_degree=(self.sn_exponential_max_denominator_degree),
                exponent_value_pool_size=(self.sn_exponential_value_pool_size),
                exponent_shortlist_size=self.sn_exponential_shortlist_size,
                composite_value_pool_size=(
                    self.sn_exponential_composite_value_pool_size
                ),
                composite_shortlist_size=(self.sn_exponential_composite_shortlist_size),
                proposal_limit=self.sn_exponential_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_exponential_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_exponential_monomial_candidates"] += pursuit.monomial_candidates
        self.stats[
            "sn_exponential_positivity_rejections"
        ] += pursuit.positivity_rejections
        self.stats["sn_exponential_candidates"] += pursuit.exponential_candidates
        self.stats["sn_exponential_value_pool"] += pursuit.exponent_value_pool
        self.stats["sn_exponential_shortlist"] += pursuit.exponent_shortlist
        self.stats[
            "sn_exponential_amplitude_candidates"
        ] += pursuit.amplitude_candidates
        self.stats[
            "sn_exponential_composite_candidates"
        ] += pursuit.composite_candidates
        self.stats[
            "sn_exponential_composite_value_pool"
        ] += pursuit.composite_value_pool
        self.stats["sn_exponential_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_exponential_proposals"] += len(pursuit.proposals)
        self.stats["sn_exponential_sobolev_exponent_proposals"] += sum(
            "sobolev_exponent" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_exponential_sobolev_composite_proposals"] += sum(
            "sobolev_composite" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_exponential_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_exponential_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_exponential_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1000000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_damped_exponential_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_exponential_candidates_evaluated"] += 1
            self.stats["sn_exponential_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "exponent_expression": proposal.exponent_expression,
                    "amplitude_expression": proposal.amplitude_expression,
                    "exponent_exponents": list(proposal.exponent_exponents),
                    "exponent_scale": proposal.exponent_scale,
                    "screen_training_r2": proposal.training_r2,
                    "exponent_target_correlation": (
                        proposal.exponent_target_correlation
                    ),
                    "exponent_sobolev_gain": proposal.exponent_sobolev_gain,
                    "composite_target_correlation": (
                        proposal.composite_target_correlation
                    ),
                    "composite_sobolev_gain": (proposal.composite_sobolev_gain),
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bp_envelope", bp_candidate),
                *(
                    ("damped_exponential", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_exponential_accuracy_branch:{label}"] += 1
            if (
                label == "damped_exponential"
                and bp_candidate is not None
                and candidate.fit_result is not None
                and bp_candidate.fit_result is not None
            ):
                self.stats["sn_exponential_gain_over_bp_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bp_candidate.fit_result.r2)
                self.stats["sn_exponential_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bp_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bp_envelope",
                "best": self._candidate_summary(bp_candidate),
                "pareto": self._candidate_summary(bp_pareto),
                "path": bp_path,
            },
            {
                "branch": "damped_exponential",
                "scales": list(self.sn_exponential_scales),
                "max_numerator_degree": (self.sn_exponential_max_numerator_degree),
                "max_denominator_degree": (self.sn_exponential_max_denominator_degree),
                "monomial_candidates": pursuit.monomial_candidates,
                "positivity_rejections": pursuit.positivity_rejections,
                "exponential_candidates": pursuit.exponential_candidates,
                "exponent_value_pool": pursuit.exponent_value_pool,
                "exponent_shortlist": pursuit.exponent_shortlist,
                "amplitude_candidates": pursuit.amplitude_candidates,
                "composite_candidates": pursuit.composite_candidates,
                "composite_value_pool": pursuit.composite_value_pool,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_affine_gaussian_accuracy_candidate(self) -> None:
        """Add affine-Gaussian factor pursuit to the complete Stage BQ envelope."""

        self._build_damped_exponential_accuracy_candidate()
        bq_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bq_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bq_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        conditioning_entries = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_affine_gaussian_contexts"] += len(contexts)
        self.stats["sn_affine_gaussian_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_affine_gaussian_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_affine_gaussian_max_dimension:
            self.stats["sn_affine_gaussian_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_affine_gaussian_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_affine_gaussians(
                direct_entries=direct_entries,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                exponent_scales=self.sn_affine_gaussian_scales,
                exponent_value_pool_size=(self.sn_affine_gaussian_value_pool_size),
                exponent_shortlist_size=(self.sn_affine_gaussian_shortlist_size),
                composite_value_pool_size=(
                    self.sn_affine_gaussian_composite_value_pool_size
                ),
                composite_shortlist_size=(
                    self.sn_affine_gaussian_composite_shortlist_size
                ),
                proposal_limit=self.sn_affine_gaussian_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_affine_gaussian_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_affine_gaussian_sources"] += pursuit.affine_sources
        self.stats[
            "sn_affine_gaussian_standardized_candidates"
        ] += pursuit.standardized_candidates
        self.stats["sn_affine_gaussian_candidates"] += pursuit.exponential_candidates
        self.stats["sn_affine_gaussian_value_pool"] += pursuit.exponent_value_pool
        self.stats["sn_affine_gaussian_shortlist"] += pursuit.exponent_shortlist
        self.stats[
            "sn_affine_gaussian_amplitude_candidates"
        ] += pursuit.amplitude_candidates
        self.stats[
            "sn_affine_gaussian_composite_candidates"
        ] += pursuit.composite_candidates
        self.stats[
            "sn_affine_gaussian_composite_value_pool"
        ] += pursuit.composite_value_pool
        self.stats["sn_affine_gaussian_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_affine_gaussian_proposals"] += len(pursuit.proposals)
        self.stats["sn_affine_gaussian_sobolev_exponent_proposals"] += sum(
            "sobolev_exponent" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_affine_gaussian_sobolev_composite_proposals"] += sum(
            "sobolev_composite" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_affine_gaussian_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_affine_gaussian_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_affine_gaussian_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1100000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_affine_gaussian_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_affine_gaussian_candidates_evaluated"] += 1
            self.stats["sn_affine_gaussian_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "source_expression": proposal.source_expression,
                    "denominator_expression": (proposal.denominator_expression),
                    "exponent_expression": proposal.exponent_expression,
                    "amplitude_expression": proposal.amplitude_expression,
                    "exponent_scale": proposal.exponent_scale,
                    "screen_training_r2": proposal.training_r2,
                    "exponent_target_correlation": (
                        proposal.exponent_target_correlation
                    ),
                    "exponent_sobolev_gain": proposal.exponent_sobolev_gain,
                    "composite_target_correlation": (
                        proposal.composite_target_correlation
                    ),
                    "composite_sobolev_gain": (proposal.composite_sobolev_gain),
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bq_envelope", bq_candidate),
                *(("affine_gaussian", candidate) for candidate in proposal_candidates),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_affine_gaussian_accuracy_branch:{label}"] += 1
            if (
                label == "affine_gaussian"
                and bq_candidate is not None
                and candidate.fit_result is not None
                and bq_candidate.fit_result is not None
            ):
                self.stats["sn_affine_gaussian_gain_over_bq_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bq_candidate.fit_result.r2)
                self.stats["sn_affine_gaussian_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bq_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bq_envelope",
                "best": self._candidate_summary(bq_candidate),
                "pareto": self._candidate_summary(bq_pareto),
                "path": bq_path,
            },
            {
                "branch": "affine_gaussian",
                "scales": list(self.sn_affine_gaussian_scales),
                "affine_sources": pursuit.affine_sources,
                "standardized_candidates": pursuit.standardized_candidates,
                "exponential_candidates": pursuit.exponential_candidates,
                "exponent_value_pool": pursuit.exponent_value_pool,
                "exponent_shortlist": pursuit.exponent_shortlist,
                "amplitude_candidates": pursuit.amplitude_candidates,
                "composite_candidates": pursuit.composite_candidates,
                "composite_value_pool": pursuit.composite_value_pool,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_sinc_squared_accuracy_candidate(self) -> None:
        """Add shared-core sinc-squared pursuit to the complete BR envelope."""

        self._build_affine_gaussian_accuracy_candidate()
        br_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        br_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        br_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        conditioning_entries = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_sinc_contexts"] += len(contexts)
        self.stats["sn_sinc_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_sinc_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_sinc_max_dimension:
            self.stats["sn_sinc_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_sinc_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_sinc_squared_factors(
                direct_entries=direct_entries,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                phase_scales=self.sn_sinc_scales,
                max_numerator_degree=self.sn_sinc_max_numerator_degree,
                max_denominator_degree=self.sn_sinc_max_denominator_degree,
                shape_value_pool_size=self.sn_sinc_shape_value_pool_size,
                shape_shortlist_size=self.sn_sinc_shape_shortlist_size,
                composite_value_pool_size=(self.sn_sinc_composite_value_pool_size),
                composite_shortlist_size=(self.sn_sinc_composite_shortlist_size),
                proposal_limit=self.sn_sinc_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_sinc_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_sinc_affine_sources"] += pursuit.affine_sources
        self.stats["sn_sinc_phase_cores"] += pursuit.phase_cores
        self.stats["sn_sinc_shape_candidates"] += pursuit.shape_candidates
        self.stats["sn_sinc_shape_value_pool"] += pursuit.shape_value_pool
        self.stats["sn_sinc_shape_shortlist"] += pursuit.shape_shortlist
        self.stats["sn_sinc_amplitude_candidates"] += pursuit.amplitude_candidates
        self.stats["sn_sinc_composite_candidates"] += pursuit.composite_candidates
        self.stats["sn_sinc_composite_value_pool"] += pursuit.composite_value_pool
        self.stats["sn_sinc_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_sinc_proposals"] += len(pursuit.proposals)
        self.stats["sn_sinc_sobolev_shape_proposals"] += sum(
            "sobolev_shape" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_sinc_sobolev_composite_proposals"] += sum(
            "sobolev_composite" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_sinc_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_sinc_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_sinc_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1200000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_sinc_squared_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_sinc_candidates_evaluated"] += 1
            self.stats["sn_sinc_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "phase_core_expression": (proposal.phase_core_expression),
                    "phase_expression": proposal.phase_expression,
                    "phase_scale": proposal.phase_scale,
                    "amplitude_expression": proposal.amplitude_expression,
                    "amplitude_exponents": list(proposal.amplitude_exponents),
                    "screen_training_r2": proposal.training_r2,
                    "shape_target_correlation": (proposal.shape_target_correlation),
                    "shape_sobolev_gain": proposal.shape_sobolev_gain,
                    "composite_target_correlation": (
                        proposal.composite_target_correlation
                    ),
                    "composite_sobolev_gain": (proposal.composite_sobolev_gain),
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("br_envelope", br_candidate),
                *(("sinc_squared", candidate) for candidate in proposal_candidates),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_sinc_accuracy_branch:{label}"] += 1
            if (
                label == "sinc_squared"
                and br_candidate is not None
                and candidate.fit_result is not None
                and br_candidate.fit_result is not None
            ):
                self.stats["sn_sinc_gain_over_br_sum"] += float(
                    candidate.fit_result.r2
                ) - float(br_candidate.fit_result.r2)
                self.stats["sn_sinc_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (br_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "br_envelope",
                "best": self._candidate_summary(br_candidate),
                "pareto": self._candidate_summary(br_pareto),
                "path": br_path,
            },
            {
                "branch": "sinc_squared",
                "scales": list(self.sn_sinc_scales),
                "max_numerator_degree": (self.sn_sinc_max_numerator_degree),
                "max_denominator_degree": (self.sn_sinc_max_denominator_degree),
                "affine_sources": pursuit.affine_sources,
                "phase_cores": pursuit.phase_cores,
                "shape_candidates": pursuit.shape_candidates,
                "shape_value_pool": pursuit.shape_value_pool,
                "shape_shortlist": pursuit.shape_shortlist,
                "amplitude_candidates": pursuit.amplitude_candidates,
                "composite_candidates": pursuit.composite_candidates,
                "composite_value_pool": pursuit.composite_value_pool,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_shared_unary_polynomial_accuracy_candidate(self) -> None:
        """Add conditional shared-unary polynomial pursuit to the BS envelope."""

        self._build_sinc_squared_accuracy_candidate()
        bs_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bs_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bs_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        conditioning_entries = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        contexts = tuple(dict.fromkeys(((), *retained_contexts)))
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_shared_unary_contexts"] += len(contexts)
        self.stats["sn_shared_unary_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_shared_unary_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_shared_unary_max_dimension:
            self.stats["sn_shared_unary_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_shared_unary_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_shared_unary_polynomials(
                direct_entries=direct_entries,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                phase_scales=self.sn_shared_unary_scales,
                transforms=self.sn_shared_unary_transforms,
                max_numerator_degree=(self.sn_shared_unary_max_numerator_degree),
                max_denominator_degree=(self.sn_shared_unary_max_denominator_degree),
                phase_value_pool_size=(self.sn_shared_unary_phase_value_pool_size),
                phase_shortlist_size=(self.sn_shared_unary_phase_shortlist_size),
                amplitude_value_pool_size=(
                    self.sn_shared_unary_amplitude_value_pool_size
                ),
                amplitude_shortlist_size=(
                    self.sn_shared_unary_amplitude_shortlist_size
                ),
                proposal_limit=self.sn_shared_unary_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_shared_unary_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_shared_unary_affine_sources"] += pursuit.affine_sources
        self.stats["sn_shared_unary_phase_cores"] += pursuit.phase_cores
        self.stats["sn_shared_unary_phase_candidates"] += pursuit.phase_candidates
        self.stats["sn_shared_unary_phase_value_pool"] += pursuit.phase_value_pool
        self.stats["sn_shared_unary_phase_shortlist"] += pursuit.phase_shortlist
        self.stats[
            "sn_shared_unary_amplitude_candidates"
        ] += pursuit.amplitude_candidates
        self.stats["sn_shared_unary_quadratic_states"] += pursuit.quadratic_states
        self.stats["sn_shared_unary_quadratic_shortlist"] += pursuit.quadratic_shortlist
        self.stats[
            "sn_shared_unary_linear_conditional_states"
        ] += pursuit.linear_conditional_states
        self.stats["sn_shared_unary_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_shared_unary_proposals"] += len(pursuit.proposals)
        self.stats["sn_shared_unary_sobolev_phase_proposals"] += sum(
            "sobolev_phase" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_shared_unary_sobolev_linear_proposals"] += sum(
            "sobolev_linear" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_shared_unary_sobolev_quadratic_proposals"] += sum(
            "sobolev_quadratic" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_shared_unary_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_shared_unary_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_shared_unary_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1300000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_shared_unary_polynomial_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_shared_unary_candidates_evaluated"] += 1
            self.stats["sn_shared_unary_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "phase_core_expression": proposal.phase_core_expression,
                    "phase_expression": proposal.phase_expression,
                    "phase_scale": proposal.phase_scale,
                    "transform": proposal.transform,
                    "linear_amplitude_expression": (
                        proposal.linear_amplitude_expression
                    ),
                    "linear_amplitude_exponents": list(
                        proposal.linear_amplitude_exponents
                    ),
                    "quadratic_amplitude_expression": (
                        proposal.quadratic_amplitude_expression
                    ),
                    "quadratic_amplitude_exponents": list(
                        proposal.quadratic_amplitude_exponents
                    ),
                    "screen_training_r2": proposal.training_r2,
                    "phase_target_correlation": (proposal.phase_target_correlation),
                    "phase_sobolev_gain": proposal.phase_sobolev_gain,
                    "linear_conditional_correlation": (
                        proposal.linear_conditional_correlation
                    ),
                    "linear_sobolev_gain": proposal.linear_sobolev_gain,
                    "quadratic_target_correlation": (
                        proposal.quadratic_target_correlation
                    ),
                    "quadratic_sobolev_gain": proposal.quadratic_sobolev_gain,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bs_envelope", bs_candidate),
                *(
                    ("shared_unary_polynomial", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_shared_unary_accuracy_branch:{label}"] += 1
            if (
                label == "shared_unary_polynomial"
                and bs_candidate is not None
                and candidate.fit_result is not None
                and bs_candidate.fit_result is not None
            ):
                self.stats["sn_shared_unary_gain_over_bs_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bs_candidate.fit_result.r2)
                self.stats["sn_shared_unary_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bs_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bs_envelope",
                "best": self._candidate_summary(bs_candidate),
                "pareto": self._candidate_summary(bs_pareto),
                "path": bs_path,
            },
            {
                "branch": "shared_unary_polynomial",
                "scales": list(self.sn_shared_unary_scales),
                "transforms": list(self.sn_shared_unary_transforms),
                "max_numerator_degree": (self.sn_shared_unary_max_numerator_degree),
                "max_denominator_degree": (self.sn_shared_unary_max_denominator_degree),
                "affine_sources": pursuit.affine_sources,
                "phase_cores": pursuit.phase_cores,
                "phase_candidates": pursuit.phase_candidates,
                "phase_value_pool": pursuit.phase_value_pool,
                "phase_shortlist": pursuit.phase_shortlist,
                "amplitude_candidates": pursuit.amplitude_candidates,
                "quadratic_states": pursuit.quadratic_states,
                "quadratic_shortlist": pursuit.quadratic_shortlist,
                "linear_conditional_states": (pursuit.linear_conditional_states),
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_reciprocal_trig_accuracy_candidate(self) -> None:
        """Add pair-aware reciprocal-trigonometric pursuit to the BT envelope."""

        self._build_shared_unary_polynomial_accuracy_candidate()
        bt_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bt_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bt_path = copy.deepcopy(self.archive_export_path)
        original_entries = tuple(
            () if self.basis_archive is None else self.basis_archive.entries
        )
        inherited_conditioning = (
            *original_entries,
            *self.archive_conditional_product_entries,
            *self.archive_iterated_product_entries,
            *self.archive_conditional_affine_unary_entries,
        )
        retained_contexts = (
            self.archive_iterated_product_contexts
            or self.archive_conditional_product_contexts
        )
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        direct_offset = len(inherited_conditioning)
        conditioning_entries = (*inherited_conditioning, *direct_entries)
        contexts = tuple(
            dict.fromkeys(
                (
                    (),
                    *retained_contexts,
                    *((direct_offset + index,) for index in range(len(direct_entries))),
                )
            )
        )
        self.stats["sn_reciprocal_trig_contexts"] += len(contexts)
        self.stats["sn_reciprocal_trig_context_size_sum"] += sum(
            len(context) for context in contexts
        )
        self.stats["sn_reciprocal_trig_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_reciprocal_trig_max_dimension:
            self.stats["sn_reciprocal_trig_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_reciprocal_trig_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_reciprocal_trig_pairs(
                direct_entries=direct_entries,
                conditioning_entries=conditioning_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=contexts,
                scales=self.sn_reciprocal_trig_scales,
                transforms=self.sn_reciprocal_trig_transforms,
                value_pool_size=self.sn_reciprocal_trig_value_pool_size,
                shortlist_size=self.sn_reciprocal_trig_shortlist_size,
                proposal_limit=self.sn_reciprocal_trig_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_reciprocal_trig_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_reciprocal_trig_phase_atoms"] += pursuit.phase_atoms
        self.stats["sn_reciprocal_trig_reciprocal_atoms"] += pursuit.reciprocal_atoms
        self.stats[
            "sn_reciprocal_trig_amplitude_candidates"
        ] += pursuit.amplitude_candidates
        self.stats["sn_reciprocal_trig_pair_candidates"] += pursuit.pair_candidates
        self.stats["sn_reciprocal_trig_value_pool"] += pursuit.value_pool
        self.stats["sn_reciprocal_trig_pair_shortlist"] += pursuit.pair_shortlist
        self.stats["sn_reciprocal_trig_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_reciprocal_trig_proposals"] += len(pursuit.proposals)
        self.stats["sn_reciprocal_trig_value_proposals"] += sum(
            "value_pair" in proposal.selection_lanes for proposal in pursuit.proposals
        )
        self.stats["sn_reciprocal_trig_sobolev_proposals"] += sum(
            "sobolev_pair" in proposal.selection_lanes for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_reciprocal_trig_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_reciprocal_trig_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_reciprocal_trig_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1400000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_reciprocal_trig_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_reciprocal_trig_candidates_evaluated"] += 1
            self.stats["sn_reciprocal_trig_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "phase_expression": proposal.phase_expression,
                    "reciprocal_expression": proposal.reciprocal_expression,
                    "amplitude_expression": proposal.amplitude_expression,
                    "phase_axis": proposal.phase_axis,
                    "reciprocal_axis": proposal.reciprocal_axis,
                    "phase_scale": proposal.phase_scale,
                    "reciprocal_scale": proposal.reciprocal_scale,
                    "transform": proposal.transform,
                    "context": list(proposal.context),
                    "context_expressions": list(proposal.context_expressions),
                    "screen_training_r2": proposal.training_r2,
                    "target_correlation": proposal.target_correlation,
                    "sobolev_gain": proposal.sobolev_gain,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bt_envelope", bt_candidate),
                *(("reciprocal_trig", candidate) for candidate in proposal_candidates),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_reciprocal_trig_accuracy_branch:{label}"] += 1
            if (
                label == "reciprocal_trig"
                and bt_candidate is not None
                and candidate.fit_result is not None
                and bt_candidate.fit_result is not None
            ):
                self.stats["sn_reciprocal_trig_gain_over_bt_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bt_candidate.fit_result.r2)
                self.stats["sn_reciprocal_trig_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bt_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bt_envelope",
                "best": self._candidate_summary(bt_candidate),
                "pareto": self._candidate_summary(bt_pareto),
                "path": bt_path,
            },
            {
                "branch": "reciprocal_trig",
                "scales": list(self.sn_reciprocal_trig_scales),
                "transforms": list(self.sn_reciprocal_trig_transforms),
                "contexts_screened": pursuit.contexts_screened,
                "phase_atoms": pursuit.phase_atoms,
                "reciprocal_atoms": pursuit.reciprocal_atoms,
                "amplitude_candidates": pursuit.amplitude_candidates,
                "pair_candidates": pursuit.pair_candidates,
                "value_pool": pursuit.value_pool,
                "pair_shortlist": pursuit.pair_shortlist,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_cross_unary_affine_accuracy_candidate(self) -> None:
        """Add pair-first cross-unary pursuit to the complete BU envelope."""

        self._build_reciprocal_trig_accuracy_candidate()
        bu_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bu_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bu_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_cross_unary_contexts"] += 1
        self.stats["sn_cross_unary_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_cross_unary_max_dimension:
            self.stats["sn_cross_unary_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_cross_unary_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_cross_unary_affine_pairs(
                direct_entries=direct_entries,
                conditioning_entries=(),
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                contexts=((),),
                scales=self.sn_cross_unary_scales,
                transforms=self.sn_cross_unary_transforms,
                inner_powers=self.sn_cross_unary_inner_powers,
                max_numerator_degree=(self.sn_cross_unary_max_numerator_degree),
                max_denominator_degree=(self.sn_cross_unary_max_denominator_degree),
                value_pool_size=self.sn_cross_unary_value_pool_size,
                shortlist_size=self.sn_cross_unary_shortlist_size,
                proposal_limit=self.sn_cross_unary_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_cross_unary_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_cross_unary_outer_amplitudes"] += pursuit.outer_amplitudes
        self.stats["sn_cross_unary_outer_unary_atoms"] += pursuit.outer_unary_atoms
        self.stats["sn_cross_unary_inner_phase_cores"] += pursuit.inner_phase_cores
        self.stats["sn_cross_unary_inner_atoms"] += pursuit.inner_atoms
        self.stats["sn_cross_unary_pair_candidates"] += pursuit.pair_candidates
        self.stats["sn_cross_unary_value_pool"] += pursuit.value_pool
        self.stats["sn_cross_unary_pair_shortlist"] += pursuit.pair_shortlist
        self.stats["sn_cross_unary_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_cross_unary_proposals"] += len(pursuit.proposals)
        self.stats["sn_cross_unary_value_proposals"] += sum(
            "value_pair" in proposal.selection_lanes for proposal in pursuit.proposals
        )
        self.stats["sn_cross_unary_sobolev_proposals"] += sum(
            "sobolev_pair" in proposal.selection_lanes for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_cross_unary_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_cross_unary_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_cross_unary_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1500000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_cross_unary_affine_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_cross_unary_candidates_evaluated"] += 1
            self.stats["sn_cross_unary_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "outer_expression": proposal.outer_expression,
                    "inner_expression": proposal.inner_expression,
                    "inner_power": proposal.inner_power,
                    "context": list(proposal.context),
                    "context_expressions": list(proposal.context_expressions),
                    "screen_training_r2": proposal.training_r2,
                    "conditional_correlation": (proposal.conditional_correlation),
                    "sobolev_gain": proposal.sobolev_gain,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bu_envelope", bu_candidate),
                *(
                    ("cross_unary_affine", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_cross_unary_accuracy_branch:{label}"] += 1
            if (
                label == "cross_unary_affine"
                and bu_candidate is not None
                and candidate.fit_result is not None
                and bu_candidate.fit_result is not None
            ):
                self.stats["sn_cross_unary_gain_over_bu_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bu_candidate.fit_result.r2)
                self.stats["sn_cross_unary_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bu_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bu_envelope",
                "best": self._candidate_summary(bu_candidate),
                "pareto": self._candidate_summary(bu_pareto),
                "path": bu_path,
            },
            {
                "branch": "cross_unary_affine",
                "scales": list(self.sn_cross_unary_scales),
                "transforms": list(self.sn_cross_unary_transforms),
                "inner_powers": list(self.sn_cross_unary_inner_powers),
                "max_numerator_degree": (self.sn_cross_unary_max_numerator_degree),
                "max_denominator_degree": (self.sn_cross_unary_max_denominator_degree),
                "contexts_screened": pursuit.contexts_screened,
                "outer_amplitudes": pursuit.outer_amplitudes,
                "outer_unary_atoms": pursuit.outer_unary_atoms,
                "inner_phase_cores": pursuit.inner_phase_cores,
                "inner_atoms": pursuit.inner_atoms,
                "pair_candidates": pursuit.pair_candidates,
                "value_pool": pursuit.value_pool,
                "pair_shortlist": pursuit.pair_shortlist,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_shared_denominator_accuracy_candidate(self) -> None:
        """Add coupled shared-denominator candidates to the complete BV envelope."""

        self._build_cross_unary_affine_accuracy_candidate()
        bv_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bv_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bv_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_shared_denominator_contexts"] += 1
        self.stats["sn_shared_denominator_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_shared_denominator_max_dimension:
            self.stats["sn_shared_denominator_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_shared_denominator_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_shared_denominators(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                denominator_signs=self.sn_shared_denominator_signs,
                numerator_signs=self.sn_shared_denominator_numerator_signs,
                value_pool_size=self.sn_shared_denominator_value_pool_size,
                shortlist_size=self.sn_shared_denominator_shortlist_size,
                proposal_limit=self.sn_shared_denominator_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_shared_denominator_fallback:{type(error).__name__}"] += 1
            return

        self.stats[
            "sn_shared_denominator_denominator_pairs"
        ] += pursuit.denominator_pairs
        self.stats[
            "sn_shared_denominator_candidates_screened"
        ] += pursuit.candidates_screened
        self.stats["sn_shared_denominator_value_pool"] += pursuit.value_pool
        self.stats[
            "sn_shared_denominator_sobolev_candidates"
        ] += pursuit.sobolev_candidates
        self.stats[
            "sn_shared_denominator_denominator_rejections"
        ] += pursuit.denominator_rejections
        self.stats["sn_shared_denominator_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_shared_denominator_proposals"] += len(pursuit.proposals)
        self.stats["sn_shared_denominator_value_proposals"] += sum(
            "value_shared_denominator" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_shared_denominator_sobolev_proposals"] += sum(
            "sobolev_shared_denominator" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_shared_denominator_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_shared_denominator_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_shared_denominator_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1600000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_shared_denominator_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_shared_denominator_candidates_evaluated"] += 1
            self.stats["sn_shared_denominator_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "denominator_expression": proposal.denominator_expression,
                    "left_numerator_expression": (proposal.left_numerator_expression),
                    "right_numerator_expression": (proposal.right_numerator_expression),
                    "denominator_sign": proposal.denominator_sign,
                    "numerator_sign": proposal.numerator_sign,
                    "screen_fitted_coefficients": list(proposal.fitted_coefficients),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "term_complementarity": proposal.term_complementarity,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bv_envelope", bv_candidate),
                *(
                    ("shared_denominator", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_shared_denominator_accuracy_branch:{label}"] += 1
            if (
                label == "shared_denominator"
                and bv_candidate is not None
                and candidate.fit_result is not None
                and bv_candidate.fit_result is not None
            ):
                self.stats["sn_shared_denominator_gain_over_bv_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bv_candidate.fit_result.r2)
                self.stats["sn_shared_denominator_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bv_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bv_envelope",
                "best": self._candidate_summary(bv_candidate),
                "pareto": self._candidate_summary(bv_pareto),
                "path": bv_path,
            },
            {
                "branch": "shared_denominator",
                "denominator_signs": list(self.sn_shared_denominator_signs),
                "numerator_signs": list(self.sn_shared_denominator_numerator_signs),
                "denominator_pairs": pursuit.denominator_pairs,
                "candidates_screened": pursuit.candidates_screened,
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "denominator_rejections": pursuit.denominator_rejections,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_relativistic_rational_accuracy_candidate(self) -> None:
        """Add bounded product-rational candidates to the complete BW envelope."""

        self._build_shared_denominator_accuracy_candidate()
        bw_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bw_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bw_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_relativistic_rational_contexts"] += 1
        self.stats["sn_relativistic_rational_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_relativistic_rational_max_dimension:
            self.stats["sn_relativistic_rational_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_relativistic_rational_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_relativistic_rationals(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                denominator_signs=self.sn_relativistic_rational_signs,
                denominator_scales=self.sn_relativistic_rational_scales,
                value_pool_size=self.sn_relativistic_rational_value_pool_size,
                shortlist_size=self.sn_relativistic_rational_shortlist_size,
                proposal_limit=self.sn_relativistic_rational_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_relativistic_rational_fallback:{type(error).__name__}"] += 1
            return

        self.stats[
            "sn_relativistic_rational_variable_triples"
        ] += pursuit.variable_triples
        self.stats[
            "sn_relativistic_rational_candidates_screened"
        ] += pursuit.candidates_screened
        self.stats["sn_relativistic_rational_value_pool"] += pursuit.value_pool
        self.stats[
            "sn_relativistic_rational_sobolev_candidates"
        ] += pursuit.sobolev_candidates
        self.stats[
            "sn_relativistic_rational_denominator_rejections"
        ] += pursuit.denominator_rejections
        self.stats[
            "sn_relativistic_rational_numeric_failures"
        ] += pursuit.numeric_failures
        self.stats["sn_relativistic_rational_proposals"] += len(pursuit.proposals)
        self.stats["sn_relativistic_rational_value_proposals"] += sum(
            "value_relativistic_rational" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_relativistic_rational_sobolev_proposals"] += sum(
            "sobolev_relativistic_rational" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_relativistic_rational_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_relativistic_rational_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_relativistic_rational_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1700000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_relativistic_rational_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_relativistic_rational_candidates_evaluated"] += 1
            self.stats["sn_relativistic_rational_valid_candidates"] += int(
                candidate.valid
            )
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "left_expression": proposal.left_expression,
                    "right_expression": proposal.right_expression,
                    "scale_expression": proposal.scale_expression,
                    "denominator_sign": proposal.denominator_sign,
                    "denominator_scale": proposal.denominator_scale,
                    "screen_fitted_coefficients": list(proposal.fitted_coefficients),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "term_complementarity": proposal.term_complementarity,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bw_envelope", bw_candidate),
                *(
                    ("relativistic_rational", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_relativistic_rational_accuracy_branch:{label}"] += 1
            if (
                label == "relativistic_rational"
                and bw_candidate is not None
                and candidate.fit_result is not None
                and bw_candidate.fit_result is not None
            ):
                self.stats["sn_relativistic_rational_gain_over_bw_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bw_candidate.fit_result.r2)
                self.stats["sn_relativistic_rational_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bw_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bw_envelope",
                "best": self._candidate_summary(bw_candidate),
                "pareto": self._candidate_summary(bw_pareto),
                "path": bw_path,
            },
            {
                "branch": "relativistic_rational",
                "denominator_signs": list(self.sn_relativistic_rational_signs),
                "denominator_scales": list(self.sn_relativistic_rational_scales),
                "variable_triples": pursuit.variable_triples,
                "candidates_screened": pursuit.candidates_screened,
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "denominator_rejections": pursuit.denominator_rejections,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_cosine_law_radial_accuracy_candidate(self) -> None:
        """Add cosine-law radial candidates to the complete BX envelope."""

        self._build_relativistic_rational_accuracy_candidate()
        bx_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bx_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bx_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_cosine_law_contexts"] += 1
        self.stats["sn_cosine_law_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_cosine_law_max_dimension:
            self.stats["sn_cosine_law_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_cosine_law_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_cosine_law_radials(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                phase_signs=self.sn_cosine_law_phase_signs,
                radial_signs=self.sn_cosine_law_radial_signs,
                phase_scales=self.sn_cosine_law_phase_scales,
                value_pool_size=self.sn_cosine_law_value_pool_size,
                shortlist_size=self.sn_cosine_law_shortlist_size,
                proposal_limit=self.sn_cosine_law_proposal_limit,
                minimum_radicand=1e-12,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_cosine_law_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_cosine_law_amplitude_pairs"] += pursuit.amplitude_pairs
        self.stats["sn_cosine_law_phase_atoms"] += pursuit.phase_atoms
        self.stats["sn_cosine_law_candidates_screened"] += pursuit.candidates_screened
        self.stats["sn_cosine_law_value_pool"] += pursuit.value_pool
        self.stats["sn_cosine_law_sobolev_candidates"] += pursuit.sobolev_candidates
        self.stats["sn_cosine_law_radicand_rejections"] += pursuit.radicand_rejections
        self.stats["sn_cosine_law_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_cosine_law_proposals"] += len(pursuit.proposals)
        self.stats["sn_cosine_law_value_proposals"] += sum(
            "value_cosine_law_radial" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_cosine_law_sobolev_proposals"] += sum(
            "sobolev_cosine_law_radial" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_cosine_law_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_cosine_law_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_cosine_law_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1800000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_cosine_law_radial_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_cosine_law_candidates_evaluated"] += 1
            self.stats["sn_cosine_law_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "left_amplitude_expression": (proposal.left_amplitude_expression),
                    "right_amplitude_expression": (proposal.right_amplitude_expression),
                    "left_phase_expression": proposal.left_phase_expression,
                    "right_phase_expression": proposal.right_phase_expression,
                    "phase_sign": proposal.phase_sign,
                    "radial_sign": proposal.radial_sign,
                    "phase_scale": proposal.phase_scale,
                    "screen_fitted_coefficients": list(proposal.fitted_coefficients),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "angular_cross_gain": proposal.angular_cross_gain,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bx_envelope", bx_candidate),
                *(
                    ("cosine_law_radial", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_cosine_law_accuracy_branch:{label}"] += 1
            if (
                label == "cosine_law_radial"
                and bx_candidate is not None
                and candidate.fit_result is not None
                and bx_candidate.fit_result is not None
            ):
                self.stats["sn_cosine_law_gain_over_bx_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bx_candidate.fit_result.r2)
                self.stats["sn_cosine_law_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bx_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bx_envelope",
                "best": self._candidate_summary(bx_candidate),
                "pareto": self._candidate_summary(bx_pareto),
                "path": bx_path,
            },
            {
                "branch": "cosine_law_radial",
                "phase_signs": list(self.sn_cosine_law_phase_signs),
                "radial_signs": list(self.sn_cosine_law_radial_signs),
                "phase_scales": list(self.sn_cosine_law_phase_scales),
                "amplitude_pairs": pursuit.amplitude_pairs,
                "phase_atoms": pursuit.phase_atoms,
                "candidates_screened": pursuit.candidates_screened,
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "radicand_rejections": pursuit.radicand_rejections,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_reciprocal_sine_square_accuracy_candidate(self) -> None:
        """Add reciprocal-sine-square candidates to the complete BY envelope."""

        self._build_cosine_law_radial_accuracy_candidate()
        by_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        by_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        by_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_reciprocal_sine_square_contexts"] += 1
        self.stats["sn_reciprocal_sine_square_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_reciprocal_sine_max_dimension:
            self.stats["sn_reciprocal_sine_square_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_reciprocal_sine_square_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_reciprocal_sine_squares(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                phase_scales=self.sn_reciprocal_sine_scales,
                max_numerator_degree=(self.sn_reciprocal_sine_max_numerator_degree),
                value_pool_size=self.sn_reciprocal_sine_value_pool_size,
                shortlist_size=self.sn_reciprocal_sine_shortlist_size,
                proposal_limit=self.sn_reciprocal_sine_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                f"sn_reciprocal_sine_square_fallback:{type(error).__name__}"
            ] += 1
            return

        self.stats[
            "sn_reciprocal_sine_square_amplitude_ratios"
        ] += pursuit.amplitude_ratios
        self.stats[
            "sn_reciprocal_sine_square_amplitude_denominator_rejections"
        ] += pursuit.amplitude_denominator_rejections
        self.stats["sn_reciprocal_sine_square_phase_atoms"] += pursuit.phase_atoms
        self.stats[
            "sn_reciprocal_sine_square_candidates_screened"
        ] += pursuit.candidates_screened
        self.stats[
            "sn_reciprocal_sine_square_sine_denominator_rejections"
        ] += pursuit.sine_denominator_rejections
        self.stats["sn_reciprocal_sine_square_value_pool"] += pursuit.value_pool
        self.stats[
            "sn_reciprocal_sine_square_sobolev_candidates"
        ] += pursuit.sobolev_candidates
        self.stats[
            "sn_reciprocal_sine_square_numeric_failures"
        ] += pursuit.numeric_failures
        self.stats["sn_reciprocal_sine_square_proposals"] += len(pursuit.proposals)
        self.stats["sn_reciprocal_sine_square_value_proposals"] += sum(
            "value_reciprocal_sine_square" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_reciprocal_sine_square_sobolev_proposals"] += sum(
            "sobolev_reciprocal_sine_square" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_reciprocal_sine_square_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_reciprocal_sine_square_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_reciprocal_sine_square_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 1900000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_reciprocal_sine_square_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_reciprocal_sine_square_candidates_evaluated"] += 1
            self.stats["sn_reciprocal_sine_square_valid_candidates"] += int(
                candidate.valid
            )
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "amplitude_expression": proposal.amplitude_expression,
                    "amplitude_exponents": list(proposal.amplitude_exponents),
                    "phase_expression": proposal.phase_expression,
                    "phase_axis": proposal.phase_axis,
                    "phase_scale": proposal.phase_scale,
                    "screen_fitted_coefficients": list(proposal.fitted_coefficients),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "nonlinear_coupling_gain": proposal.nonlinear_coupling_gain,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("by_envelope", by_candidate),
                *(
                    ("reciprocal_sine_square", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_reciprocal_sine_square_accuracy_branch:{label}"] += 1
            if (
                label == "reciprocal_sine_square"
                and by_candidate is not None
                and candidate.fit_result is not None
                and by_candidate.fit_result is not None
            ):
                self.stats["sn_reciprocal_sine_square_gain_over_by_sum"] += float(
                    candidate.fit_result.r2
                ) - float(by_candidate.fit_result.r2)
                self.stats["sn_reciprocal_sine_square_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (by_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "by_envelope",
                "best": self._candidate_summary(by_candidate),
                "pareto": self._candidate_summary(by_pareto),
                "path": by_path,
            },
            {
                "branch": "reciprocal_sine_square",
                "phase_scales": list(self.sn_reciprocal_sine_scales),
                "max_numerator_degree": (self.sn_reciprocal_sine_max_numerator_degree),
                "amplitude_ratios": pursuit.amplitude_ratios,
                "amplitude_denominator_rejections": (
                    pursuit.amplitude_denominator_rejections
                ),
                "phase_atoms": pursuit.phase_atoms,
                "candidates_screened": pursuit.candidates_screened,
                "sine_denominator_rejections": (pursuit.sine_denominator_rejections),
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_inverse_cosine_radial_accuracy_candidate(self) -> None:
        """Add inverse cosine-radial candidates to the complete BZ envelope."""

        self._build_reciprocal_sine_square_accuracy_candidate()
        bz_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        bz_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        bz_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_inverse_cosine_radial_contexts"] += 1
        self.stats["sn_inverse_cosine_radial_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_inverse_cosine_max_dimension:
            self.stats["sn_inverse_cosine_radial_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_inverse_cosine_radial_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_inverse_cosine_radials(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                phase_signs=self.sn_inverse_cosine_phase_signs,
                radial_signs=self.sn_inverse_cosine_radial_signs,
                phase_scales=self.sn_inverse_cosine_phase_scales,
                value_pool_size=self.sn_inverse_cosine_value_pool_size,
                shortlist_size=self.sn_inverse_cosine_shortlist_size,
                proposal_limit=self.sn_inverse_cosine_proposal_limit,
                minimum_radicand=1e-12,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_inverse_cosine_radial_fallback:{type(error).__name__}"] += 1
            return

        self.stats[
            "sn_inverse_cosine_radial_amplitude_atoms"
        ] += pursuit.amplitude_atoms
        self.stats[
            "sn_inverse_cosine_radial_amplitude_denominator_rejections"
        ] += pursuit.amplitude_denominator_rejections
        self.stats["sn_inverse_cosine_radial_radial_pairs"] += pursuit.radial_pairs
        self.stats["sn_inverse_cosine_radial_phase_atoms"] += pursuit.phase_atoms
        self.stats[
            "sn_inverse_cosine_radial_candidates_screened"
        ] += pursuit.candidates_screened
        self.stats[
            "sn_inverse_cosine_radial_radicand_rejections"
        ] += pursuit.radicand_rejections
        self.stats["sn_inverse_cosine_radial_value_pool"] += pursuit.value_pool
        self.stats[
            "sn_inverse_cosine_radial_sobolev_candidates"
        ] += pursuit.sobolev_candidates
        self.stats[
            "sn_inverse_cosine_radial_numeric_failures"
        ] += pursuit.numeric_failures
        self.stats["sn_inverse_cosine_radial_proposals"] += len(pursuit.proposals)
        self.stats["sn_inverse_cosine_radial_value_proposals"] += sum(
            "value_inverse_cosine_radial" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_inverse_cosine_radial_sobolev_proposals"] += sum(
            "sobolev_inverse_cosine_radial" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_inverse_cosine_radial_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_inverse_cosine_radial_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_inverse_cosine_radial_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 2000000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_inverse_cosine_radial_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_inverse_cosine_radial_candidates_evaluated"] += 1
            self.stats["sn_inverse_cosine_radial_valid_candidates"] += int(
                candidate.valid
            )
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "amplitude_expression": proposal.amplitude_expression,
                    "amplitude_numerator_axis": (proposal.amplitude_numerator_axis),
                    "amplitude_denominator_axis": (proposal.amplitude_denominator_axis),
                    "radial_left_axis": proposal.radial_left_axis,
                    "radial_right_axis": proposal.radial_right_axis,
                    "phase_left_axis": proposal.phase_left_axis,
                    "phase_right_axis": proposal.phase_right_axis,
                    "phase_sign": proposal.phase_sign,
                    "radial_sign": proposal.radial_sign,
                    "phase_scale": proposal.phase_scale,
                    "screen_fitted_coefficients": list(proposal.fitted_coefficients),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "angular_cross_gain": proposal.angular_cross_gain,
                    "reciprocal_coupling_gain": (proposal.reciprocal_coupling_gain),
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("bz_envelope", bz_candidate),
                *(
                    ("inverse_cosine_radial", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_inverse_cosine_radial_accuracy_branch:{label}"] += 1
            if (
                label == "inverse_cosine_radial"
                and bz_candidate is not None
                and candidate.fit_result is not None
                and bz_candidate.fit_result is not None
            ):
                self.stats["sn_inverse_cosine_radial_gain_over_bz_sum"] += float(
                    candidate.fit_result.r2
                ) - float(bz_candidate.fit_result.r2)
                self.stats["sn_inverse_cosine_radial_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (bz_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "bz_envelope",
                "best": self._candidate_summary(bz_candidate),
                "pareto": self._candidate_summary(bz_pareto),
                "path": bz_path,
            },
            {
                "branch": "inverse_cosine_radial",
                "phase_signs": list(self.sn_inverse_cosine_phase_signs),
                "radial_signs": list(self.sn_inverse_cosine_radial_signs),
                "phase_scales": list(self.sn_inverse_cosine_phase_scales),
                "amplitude_atoms": pursuit.amplitude_atoms,
                "amplitude_denominator_rejections": (
                    pursuit.amplitude_denominator_rejections
                ),
                "radial_pairs": pursuit.radial_pairs,
                "phase_atoms": pursuit.phase_atoms,
                "candidates_screened": pursuit.candidates_screened,
                "radicand_rejections": pursuit.radicand_rejections,
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_multiaxis_inverse_square_accuracy_candidate(self) -> None:
        """Add multi-axis inverse-square candidates to the complete CA envelope."""

        self._build_inverse_cosine_radial_accuracy_candidate()
        ca_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        ca_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        ca_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_multiaxis_inverse_square_contexts"] += 1
        self.stats["sn_multiaxis_inverse_square_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_multiaxis_max_dimension:
            self.stats["sn_multiaxis_inverse_square_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_multiaxis_inverse_square_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_multiaxis_inverse_squares(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                pair_signs=self.sn_multiaxis_pair_signs,
                radial_term_counts=self.sn_multiaxis_radial_term_counts,
                max_numerator_degree=self.sn_multiaxis_max_numerator_degree,
                value_pool_size=self.sn_multiaxis_value_pool_size,
                shortlist_size=self.sn_multiaxis_shortlist_size,
                proposal_limit=self.sn_multiaxis_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                f"sn_multiaxis_inverse_square_fallback:{type(error).__name__}"
            ] += 1
            return

        self.stats["sn_multiaxis_inverse_square_pair_atoms"] += pursuit.pair_atoms
        self.stats["sn_multiaxis_inverse_square_radial_shapes"] += pursuit.radial_shapes
        self.stats[
            "sn_multiaxis_inverse_square_candidates_screened"
        ] += pursuit.candidates_screened
        self.stats[
            "sn_multiaxis_inverse_square_denominator_rejections"
        ] += pursuit.denominator_rejections
        self.stats["sn_multiaxis_inverse_square_value_pool"] += pursuit.value_pool
        self.stats[
            "sn_multiaxis_inverse_square_sobolev_candidates"
        ] += pursuit.sobolev_candidates
        self.stats[
            "sn_multiaxis_inverse_square_numeric_failures"
        ] += pursuit.numeric_failures
        self.stats["sn_multiaxis_inverse_square_proposals"] += len(pursuit.proposals)
        self.stats["sn_multiaxis_inverse_square_value_proposals"] += sum(
            "value_multiaxis_inverse_square" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_multiaxis_inverse_square_sobolev_proposals"] += sum(
            "sobolev_multiaxis_inverse_square" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_multiaxis_inverse_square_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_multiaxis_inverse_square_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_multiaxis_inverse_square_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 2100000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_multiaxis_inverse_square_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_multiaxis_inverse_square_candidates_evaluated"] += 1
            self.stats["sn_multiaxis_inverse_square_valid_candidates"] += int(
                candidate.valid
            )
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "numerator_axes": list(proposal.numerator_axes),
                    "radial_pairs": [list(pair) for pair in proposal.radial_pairs],
                    "radial_signs": list(proposal.radial_signs),
                    "screen_fitted_coefficients": list(proposal.fitted_coefficients),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "denominator_sobolev_gain": (proposal.denominator_sobolev_gain),
                    "mean_term_novelty": proposal.mean_term_novelty,
                    "min_term_novelty": proposal.min_term_novelty,
                    "reciprocal_coupling_gain": (proposal.reciprocal_coupling_gain),
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("ca_envelope", ca_candidate),
                *(
                    ("multiaxis_inverse_square", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_multiaxis_inverse_square_accuracy_branch:{label}"] += 1
            if (
                label == "multiaxis_inverse_square"
                and ca_candidate is not None
                and candidate.fit_result is not None
                and ca_candidate.fit_result is not None
            ):
                self.stats["sn_multiaxis_inverse_square_gain_over_ca_sum"] += float(
                    candidate.fit_result.r2
                ) - float(ca_candidate.fit_result.r2)
                self.stats["sn_multiaxis_inverse_square_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (ca_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "ca_envelope",
                "best": self._candidate_summary(ca_candidate),
                "pareto": self._candidate_summary(ca_pareto),
                "path": ca_path,
            },
            {
                "branch": "multiaxis_inverse_square",
                "pair_signs": list(self.sn_multiaxis_pair_signs),
                "radial_term_counts": list(self.sn_multiaxis_radial_term_counts),
                "max_numerator_degree": (self.sn_multiaxis_max_numerator_degree),
                "pair_atoms": pursuit.pair_atoms,
                "radial_shapes": pursuit.radial_shapes,
                "candidates_screened": pursuit.candidates_screened,
                "denominator_rejections": pursuit.denominator_rejections,
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_interference_sine_ratio_accuracy_candidate(self) -> None:
        """Add squared sine-ratio candidates to the complete CB envelope."""

        self._build_multiaxis_inverse_square_accuracy_candidate()
        cb_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        cb_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        cb_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_interference_sine_ratio_contexts"] += 1
        self.stats["sn_interference_sine_ratio_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_interference_sine_max_dimension:
            self.stats["sn_interference_sine_ratio_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_interference_sine_ratio_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_interference_sine_ratios(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                phase_scales=self.sn_interference_sine_scales,
                value_pool_size=self.sn_interference_sine_value_pool_size,
                shortlist_size=self.sn_interference_sine_shortlist_size,
                proposal_limit=self.sn_interference_sine_proposal_limit,
                protected_epsilon=1e-6,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                f"sn_interference_sine_ratio_fallback:{type(error).__name__}"
            ] += 1
            return

        self.stats[
            "sn_interference_sine_ratio_amplitude_atoms"
        ] += pursuit.amplitude_atoms
        self.stats[
            "sn_interference_sine_ratio_amplitude_denominator_rejections"
        ] += pursuit.amplitude_denominator_rejections
        self.stats[
            "sn_interference_sine_ratio_numerator_phase_cores"
        ] += pursuit.numerator_phase_cores
        self.stats["sn_interference_sine_ratio_pairs"] += pursuit.sine_ratio_pairs
        self.stats[
            "sn_interference_sine_ratio_sine_denominator_rejections"
        ] += pursuit.sine_denominator_rejections
        self.stats[
            "sn_interference_sine_ratio_candidates_screened"
        ] += pursuit.candidates_screened
        self.stats["sn_interference_sine_ratio_value_pool"] += pursuit.value_pool
        self.stats[
            "sn_interference_sine_ratio_sobolev_candidates"
        ] += pursuit.sobolev_candidates
        self.stats[
            "sn_interference_sine_ratio_numeric_failures"
        ] += pursuit.numeric_failures
        self.stats["sn_interference_sine_ratio_proposals"] += len(pursuit.proposals)
        self.stats["sn_interference_sine_ratio_value_proposals"] += sum(
            "value_interference_sine_ratio" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_interference_sine_ratio_sobolev_proposals"] += sum(
            "sobolev_interference_sine_ratio" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_interference_sine_ratio_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_interference_sine_ratio_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_interference_sine_ratio_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 2200000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_interference_sine_ratio_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_interference_sine_ratio_candidates_evaluated"] += 1
            self.stats["sn_interference_sine_ratio_valid_candidates"] += int(
                candidate.valid
            )
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "amplitude_expression": proposal.amplitude_expression,
                    "amplitude_numerator_axis": (proposal.amplitude_numerator_axis),
                    "amplitude_denominator_axis": (proposal.amplitude_denominator_axis),
                    "numerator_phase_axes": list(proposal.numerator_phase_axes),
                    "denominator_phase_axis": proposal.denominator_phase_axis,
                    "phase_scale": proposal.phase_scale,
                    "screen_fitted_coefficients": list(proposal.fitted_coefficients),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "phase_complementarity": proposal.phase_complementarity,
                    "ratio_coupling_gain": proposal.ratio_coupling_gain,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("cb_envelope", cb_candidate),
                *(
                    ("interference_sine_ratio", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_interference_sine_ratio_accuracy_branch:{label}"] += 1
            if (
                label == "interference_sine_ratio"
                and cb_candidate is not None
                and candidate.fit_result is not None
                and cb_candidate.fit_result is not None
            ):
                self.stats["sn_interference_sine_ratio_gain_over_cb_sum"] += float(
                    candidate.fit_result.r2
                ) - float(cb_candidate.fit_result.r2)
                self.stats["sn_interference_sine_ratio_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (cb_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "cb_envelope",
                "best": self._candidate_summary(cb_candidate),
                "pareto": self._candidate_summary(cb_pareto),
                "path": cb_path,
            },
            {
                "branch": "interference_sine_ratio",
                "phase_scales": list(self.sn_interference_sine_scales),
                "amplitude_atoms": pursuit.amplitude_atoms,
                "amplitude_denominator_rejections": (
                    pursuit.amplitude_denominator_rejections
                ),
                "numerator_phase_cores": pursuit.numerator_phase_cores,
                "sine_ratio_pairs": pursuit.sine_ratio_pairs,
                "sine_denominator_rejections": (pursuit.sine_denominator_rejections),
                "candidates_screened": pursuit.candidates_screened,
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_sparse_radical_accuracy_candidate(self) -> None:
        """Add sparse integer-exponent radicals to the complete CB envelope."""

        self._build_multiaxis_inverse_square_accuracy_candidate()
        cb_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        cb_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        cb_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_sparse_radical_contexts"] += 1
        self.stats["sn_sparse_radical_features"] += len(direct_entries)
        if len(direct_entries) > self.sn_sparse_radical_max_dimension:
            self.stats["sn_sparse_radical_fallback:dimension_limit"] += 1
            return
        if (
            not direct_entries
            or self._active_y is None
            or self.geometry_indices is None
        ):
            self.stats["sn_sparse_radical_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_sparse_radicals(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                radical_offsets=self.sn_sparse_radical_offsets,
                radical_scales=self.sn_sparse_radical_scales,
                max_abs_exponent=self.sn_sparse_radical_max_abs_exponent,
                max_numerator_degree=self.sn_sparse_radical_max_numerator_degree,
                max_denominator_degree=self.sn_sparse_radical_max_denominator_degree,
                integer_candidate_limit=(
                    self.sn_sparse_radical_integer_candidate_limit
                ),
                value_pool_size=self.sn_sparse_radical_value_pool_size,
                shortlist_size=self.sn_sparse_radical_shortlist_size,
                proposal_limit=self.sn_sparse_radical_proposal_limit,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"sn_sparse_radical_fallback:{type(error).__name__}"] += 1
            return

        self.stats["sn_sparse_radical_log_linear_fits"] += pursuit.log_linear_fits
        self.stats[
            "sn_sparse_radical_integer_exponent_proposals"
        ] += pursuit.integer_exponent_proposals
        self.stats[
            "sn_sparse_radical_candidates_screened"
        ] += pursuit.candidates_screened
        self.stats["sn_sparse_radical_value_pool"] += pursuit.value_pool
        self.stats["sn_sparse_radical_sobolev_candidates"] += pursuit.sobolev_candidates
        self.stats[
            "sn_sparse_radical_transform_rejections"
        ] += pursuit.transform_rejections
        self.stats[
            "sn_sparse_radical_exponent_bound_rejections"
        ] += pursuit.exponent_bound_rejections
        self.stats["sn_sparse_radical_domain_rejections"] += pursuit.domain_rejections
        self.stats["sn_sparse_radical_numeric_failures"] += pursuit.numeric_failures
        self.stats["sn_sparse_radical_proposals"] += len(pursuit.proposals)
        self.stats["sn_sparse_radical_value_proposals"] += sum(
            "value_sparse_radical" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_sparse_radical_sobolev_proposals"] += sum(
            "sobolev_sparse_radical" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_sparse_radical_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_sparse_radical_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_sparse_radical_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 2300000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_sparse_radical_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_sparse_radical_candidates_evaluated"] += 1
            self.stats["sn_sparse_radical_valid_candidates"] += int(candidate.valid)
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "exponent_vector": list(proposal.exponent_vector),
                    "radical_offset": proposal.radical_offset,
                    "radical_scale": proposal.radical_scale,
                    "continuous_exponents": list(proposal.continuous_exponents),
                    "inferred_scale": proposal.inferred_scale,
                    "log_fit_rmse": proposal.log_fit_rmse,
                    "screen_fitted_coefficients": list(proposal.fitted_coefficients),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "monomial_sobolev_gain": proposal.monomial_sobolev_gain,
                    "radical_coupling_gain": proposal.radical_coupling_gain,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("cb_envelope", cb_candidate),
                *(("sparse_radical", candidate) for candidate in proposal_candidates),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_sparse_radical_accuracy_branch:{label}"] += 1
            if (
                label == "sparse_radical"
                and cb_candidate is not None
                and candidate.fit_result is not None
                and cb_candidate.fit_result is not None
            ):
                self.stats["sn_sparse_radical_gain_over_cb_sum"] += float(
                    candidate.fit_result.r2
                ) - float(cb_candidate.fit_result.r2)
                self.stats["sn_sparse_radical_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (cb_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "cb_envelope",
                "best": self._candidate_summary(cb_candidate),
                "pareto": self._candidate_summary(cb_pareto),
                "path": cb_path,
            },
            {
                "branch": "sparse_radical",
                "radical_offsets": list(self.sn_sparse_radical_offsets),
                "radical_scales": list(self.sn_sparse_radical_scales),
                "max_abs_exponent": self.sn_sparse_radical_max_abs_exponent,
                "max_numerator_degree": (self.sn_sparse_radical_max_numerator_degree),
                "max_denominator_degree": (
                    self.sn_sparse_radical_max_denominator_degree
                ),
                "log_linear_fits": pursuit.log_linear_fits,
                "integer_exponent_proposals": (pursuit.integer_exponent_proposals),
                "candidates_screened": pursuit.candidates_screened,
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "transform_rejections": pursuit.transform_rejections,
                "exponent_bound_rejections": (pursuit.exponent_bound_rejections),
                "domain_rejections": pursuit.domain_rejections,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_shared_ratio_trig_polynomial_accuracy_candidate(self) -> None:
        """Add the complete Stage-CE seven-coordinate family to the CD envelope."""

        self._build_sparse_radical_accuracy_candidate()
        cd_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        cd_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        cd_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        self.stats["sn_shared_ratio_trig_contexts"] += 1
        self.stats["sn_shared_ratio_trig_features"] += len(direct_entries)
        if len(direct_entries) != self.sn_shared_ratio_trig_required_dimension:
            self.stats["sn_shared_ratio_trig_fallback:dimension_mismatch"] += 1
            return
        if (
            self._active_y is None
            or self.geometry_indices is None
            or not direct_entries
        ):
            self.stats["sn_shared_ratio_trig_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = lift_direct_shared_ratio_trig_polynomials(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                phase_scales=self.sn_shared_ratio_trig_phase_scales,
                value_pool_size=self.sn_shared_ratio_trig_value_pool_size,
                shortlist_size=self.sn_shared_ratio_trig_shortlist_size,
                proposal_limit=self.sn_shared_ratio_trig_proposal_limit,
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[
                f"sn_shared_ratio_trig_fallback:{type(error).__name__}"
            ] += 1
            return
        if pursuit.dimension_fallback:
            self.stats["sn_shared_ratio_trig_fallback:dimension_mismatch"] += 1
            return

        self.stats[
            "sn_shared_ratio_trig_candidates_screened"
        ] += pursuit.candidates_screened
        self.stats["sn_shared_ratio_trig_value_pool"] += pursuit.value_pool
        self.stats[
            "sn_shared_ratio_trig_sobolev_candidates"
        ] += pursuit.sobolev_candidates
        self.stats[
            "sn_shared_ratio_trig_domain_rejections"
        ] += pursuit.domain_rejections
        self.stats[
            "sn_shared_ratio_trig_numeric_failures"
        ] += pursuit.numeric_failures
        self.stats["sn_shared_ratio_trig_proposals"] += len(pursuit.proposals)
        self.stats["sn_shared_ratio_trig_value_proposals"] += sum(
            "value_shared_ratio_trig" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )
        self.stats["sn_shared_ratio_trig_sobolev_proposals"] += sum(
            "sobolev_shared_ratio_trig" in proposal.selection_lanes
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats["sn_shared_ratio_trig_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats["sn_shared_ratio_trig_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats["sn_shared_ratio_trig_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(generation + 1, 2400000 + index),
                generation=generation,
                slot=-1,
                variation_type="sn_shared_ratio_trig_polynomial_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats["sn_shared_ratio_trig_candidates_evaluated"] += 1
            self.stats["sn_shared_ratio_trig_valid_candidates"] += int(
                candidate.valid
            )
            proposal_path.append(
                {
                    "expression": expression,
                    "tree_length": len(tree),
                    "ratio_axes": list(proposal.ratio_axes),
                    "phase_axis": proposal.phase_axis,
                    "amplitude_numerator_axes": list(
                        proposal.amplitude_numerator_axes
                    ),
                    "amplitude_denominator_axes": list(
                        proposal.amplitude_denominator_axes
                    ),
                    "reciprocal_sign": proposal.reciprocal_sign,
                    "phase_sign": proposal.phase_sign,
                    "phase_scale": proposal.phase_scale,
                    "screen_fitted_coefficients": list(
                        proposal.fitted_coefficients
                    ),
                    "screen_training_r2": proposal.training_r2,
                    "prediction_correlation": proposal.prediction_correlation,
                    "candidate_sobolev_gain": proposal.candidate_sobolev_gain,
                    "amplitude_square_sobolev_gain": (
                        proposal.amplitude_square_sobolev_gain
                    ),
                    "bracket_sobolev_gain": proposal.bracket_sobolev_gain,
                    "bracket_term_novelties": list(
                        proposal.bracket_term_novelties
                    ),
                    "coupling_gain": proposal.coupling_gain,
                    "joint_score": proposal.joint_score,
                    "selection_lanes": list(proposal.selection_lanes),
                    "candidate": self._candidate_summary(candidate),
                }
            )
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                ("cd_envelope", cd_candidate),
                *(
                    ("shared_ratio_trig_polynomial", candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"sn_shared_ratio_trig_accuracy_branch:{label}"] += 1
            if (
                label == "shared_ratio_trig_polynomial"
                and cd_candidate is not None
                and candidate.fit_result is not None
                and cd_candidate.fit_result is not None
            ):
                self.stats["sn_shared_ratio_trig_gain_over_cd_sum"] += float(
                    candidate.fit_result.r2
                ) - float(cd_candidate.fit_result.r2)
                self.stats["sn_shared_ratio_trig_improvement_branch"] += 1
        else:
            self.archive_export_candidate = None
        pareto_candidates = [
            candidate
            for candidate in (cd_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        self.archive_export_path = [
            {
                "branch": "cd_envelope",
                "best": self._candidate_summary(cd_candidate),
                "pareto": self._candidate_summary(cd_pareto),
                "path": cd_path,
            },
            {
                "branch": "shared_ratio_trig_polynomial",
                "phase_scales": list(self.sn_shared_ratio_trig_phase_scales),
                "required_dimension": self.sn_shared_ratio_trig_required_dimension,
                "candidates_screened": pursuit.candidates_screened,
                "value_pool": pursuit.value_pool,
                "sobolev_candidates": pursuit.sobolev_candidates,
                "domain_rejections": pursuit.domain_rejections,
                "numeric_failures": pursuit.numeric_failures,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _build_deep_accuracy_candidate(
        self,
        stage: DeepAccuracyStage,
    ) -> None:
        """Compose one CF--CL family with its complete predecessor envelope."""

        if stage.predecessor_code is None:
            self._build_shared_ratio_trig_polynomial_accuracy_candidate()
        else:
            self._build_deep_accuracy_candidate(
                DEEP_ACCURACY_STAGE_BY_CODE[stage.predecessor_code]
            )
        predecessor_candidate = (
            None
            if self.archive_export_candidate is None
            else self.archive_export_candidate.copy()
        )
        predecessor_pareto = (
            None
            if self.archive_export_pareto_candidate is None
            else self.archive_export_pareto_candidate.copy()
        )
        predecessor_path = copy.deepcopy(self.archive_export_path)
        direct_entries = self.direct_feature_basis_entries
        if not direct_entries:
            direct_entries = self._build_direct_feature_basis_entries()
            self.direct_feature_basis_entries = direct_entries
        prefix = stage.counter_prefix
        self.stats[f"{prefix}_contexts"] += 1
        self.stats[f"{prefix}_features"] += len(direct_entries)
        required_dimension = int(
            getattr(self, stage.required_dimension_attribute)
        )
        if len(direct_entries) != required_dimension:
            self.stats[f"{prefix}_fallback:dimension_mismatch"] += 1
            return
        if (
            self._active_y is None
            or self.geometry_indices is None
            or not direct_entries
        ):
            self.stats[f"{prefix}_fallback:data_unavailable"] += 1
            return
        try:
            pursuit = stage.lift_function(
                direct_entries=direct_entries,
                target=self._active_y,
                variable_names=self.feature_names,
                geometry_sample_count=len(self.geometry_indices),
                lambda_value=self.sobolev_lambda_value,
                lambda_gradient=self.sobolev_lambda_gradient,
                **stage_lift_arguments(stage, vars(self)),
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.stats[f"{prefix}_fallback:{type(error).__name__}"] += 1
            return
        if pursuit.dimension_fallback:
            self.stats[f"{prefix}_fallback:dimension_mismatch"] += 1
            return

        for field_name in stage.result_counter_fields:
            self.stats[f"{prefix}_{field_name}"] += getattr(
                pursuit, field_name
            )
        self.stats[f"{prefix}_proposals"] += len(pursuit.proposals)
        self.stats[f"{prefix}_value_proposals"] += sum(
            any(
                str(lane).startswith("value_")
                for lane in proposal.selection_lanes
            )
            for proposal in pursuit.proposals
        )
        self.stats[f"{prefix}_sobolev_proposals"] += sum(
            any(
                str(lane).startswith("sobolev_")
                for lane in proposal.selection_lanes
            )
            for proposal in pursuit.proposals
        )

        proposal_candidates: list[ControlledIndividual] = []
        proposal_path: list[dict[str, Any]] = []
        seen: set[str] = set()
        generation = len(self.records)
        for index, proposal in enumerate(pursuit.proposals):
            try:
                tree = nd.parse(proposal.expression)
            except (TypeError, ValueError, KeyError):
                self.stats[f"{prefix}_construction_failures"] += 1
                continue
            expression = tree.to_str(number_format=".17g")
            if expression in seen:
                self.stats[f"{prefix}_duplicates"] += 1
                continue
            seen.add(expression)
            if len(tree) > self.max_len:
                self.stats[f"{prefix}_length_rejections"] += 1
                continue
            candidate = ControlledIndividual(
                tree,
                candidate_id=self._proposal_id(
                    generation + 1,
                    stage.proposal_id_offset + index,
                ),
                generation=generation,
                slot=-1,
                variation_type=f"sn_{stage.family_label}_candidate",
            )
            self._evaluate_individual(candidate, self._active_X, self._active_y)
            self.stats[f"{prefix}_candidates_evaluated"] += 1
            self.stats[f"{prefix}_valid_candidates"] += int(candidate.valid)
            trace = proposal_trace_document(proposal)
            trace["tree_length"] = len(tree)
            trace["candidate"] = self._candidate_summary(candidate)
            proposal_path.append(trace)
            if candidate.valid:
                proposal_candidates.append(candidate)

        candidates = [
            (label, candidate)
            for label, candidate in (
                (stage.predecessor_label, predecessor_candidate),
                *(
                    (stage.family_label, candidate)
                    for candidate in proposal_candidates
                ),
            )
            if candidate is not None
        ]
        if candidates:
            label, candidate = min(
                candidates, key=lambda item: self._accuracy_key(item[1])
            )
            self.archive_export_candidate = candidate.copy()
            self.stats[f"{prefix}_accuracy_branch:{label}"] += 1
            if (
                label == stage.family_label
                and predecessor_candidate is not None
                and candidate.fit_result is not None
                and predecessor_candidate.fit_result is not None
            ):
                self.stats[f"{prefix}_gain_over_predecessor_sum"] += float(
                    candidate.fit_result.r2
                ) - float(predecessor_candidate.fit_result.r2)
                self.stats[f"{prefix}_improvement_branch"] += 1
        pareto_candidates = [
            candidate
            for candidate in (predecessor_pareto, *proposal_candidates)
            if candidate is not None
        ]
        self.archive_export_pareto_candidate = (
            None
            if not pareto_candidates
            else min(pareto_candidates, key=self._accuracy_key).copy()
        )
        result_counters = {
            field_name: getattr(pursuit, field_name)
            for field_name in stage.result_counter_fields
        }
        self.archive_export_path = [
            {
                "branch": stage.predecessor_label,
                "best": self._candidate_summary(predecessor_candidate),
                "pareto": self._candidate_summary(predecessor_pareto),
                "path": predecessor_path,
            },
            {
                "branch": stage.family_label,
                "stage": stage.code,
                "required_dimension": required_dimension,
                "result_counters": result_counters,
                "proposals": proposal_path,
                "best": self._candidate_summary(self.archive_export_candidate),
            },
        ]

    def _select_deep_accuracy_export(
        self,
        selected: ControlledIndividual | None,
        stage: DeepAccuracyStage,
    ) -> ControlledIndividual | None:
        """Apply one export-only stage without changing Population-GP search."""

        prefix = stage.counter_prefix
        self.stats[f"{prefix}_accuracy_export_events"] += 1
        baseline_export = selected
        self._build_deep_accuracy_candidate(stage)
        accuracy_candidate = self.archive_export_candidate
        if accuracy_candidate is not None and (
            baseline_export is None
            or self._accuracy_key(accuracy_candidate)
            < self._accuracy_key(baseline_export)
        ):
            accuracy_candidate.variation_type = (
                f"sn_{stage.family_label}_accuracy_export_selected"
            )
            selected = accuracy_candidate
            self.stats[f"{prefix}_accuracy_export_selected"] += 1
            if (
                baseline_export is not None
                and baseline_export.fit_result is not None
                and accuracy_candidate.fit_result is not None
            ):
                self.stats[f"{prefix}_accuracy_r2_gain_sum"] += float(
                    accuracy_candidate.fit_result.r2
                ) - float(baseline_export.fit_result.r2)
                self.stats[f"{prefix}_accuracy_complexity_delta_sum"] += float(
                    accuracy_candidate.complexity
                ) - float(baseline_export.complexity)
        else:
            self.stats[f"{prefix}_accuracy_base_retained"] += 1
        return selected

    def _select_exported_tree(self) -> None:
        selected = self.method_selected_best or self.base_quality_best_ever
        if self.use_sn_archive_export:
            self._build_archive_export_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_export_selected"] += 1
        elif self.use_sn_archive_anchor_export:
            self._build_archive_anchor_export_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_anchor_export_selected"] += 1
        elif self.use_sn_shared_ratio_trig_polynomial_accuracy_export:
            self.stats["sn_shared_ratio_trig_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_shared_ratio_trig_polynomial_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_shared_ratio_trig_polynomial_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_shared_ratio_trig_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_shared_ratio_trig_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_shared_ratio_trig_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_shared_ratio_trig_accuracy_base_retained"] += 1
        elif self.deep_accuracy_stage is not None:
            selected = self._select_deep_accuracy_export(
                selected,
                self.deep_accuracy_stage,
            )
        elif self.use_sn_sparse_radical_accuracy_export:
            self.stats["sn_sparse_radical_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_sparse_radical_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_sparse_radical_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_sparse_radical_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_sparse_radical_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_sparse_radical_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_sparse_radical_accuracy_base_retained"] += 1
        elif self.use_sn_interference_sine_ratio_accuracy_export:
            self.stats["sn_interference_sine_ratio_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_interference_sine_ratio_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_interference_sine_ratio_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_interference_sine_ratio_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_interference_sine_ratio_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_interference_sine_ratio_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_interference_sine_ratio_accuracy_base_retained"] += 1
        elif self.use_sn_multiaxis_inverse_square_accuracy_export:
            self.stats["sn_multiaxis_inverse_square_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_multiaxis_inverse_square_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_multiaxis_inverse_square_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_multiaxis_inverse_square_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_multiaxis_inverse_square_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_multiaxis_inverse_square_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_multiaxis_inverse_square_accuracy_base_retained"] += 1
        elif self.use_sn_inverse_cosine_radial_accuracy_export:
            self.stats["sn_inverse_cosine_radial_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_inverse_cosine_radial_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_inverse_cosine_radial_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_inverse_cosine_radial_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_inverse_cosine_radial_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_inverse_cosine_radial_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_inverse_cosine_radial_accuracy_base_retained"] += 1
        elif self.use_sn_reciprocal_sine_square_accuracy_export:
            self.stats["sn_reciprocal_sine_square_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_reciprocal_sine_square_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_reciprocal_sine_square_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_reciprocal_sine_square_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_reciprocal_sine_square_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_reciprocal_sine_square_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_reciprocal_sine_square_accuracy_base_retained"] += 1
        elif self.use_sn_cosine_law_radial_accuracy_export:
            self.stats["sn_cosine_law_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_cosine_law_radial_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_cosine_law_radial_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_cosine_law_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_cosine_law_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats["sn_cosine_law_accuracy_complexity_delta_sum"] += float(
                        accuracy_candidate.complexity
                    ) - float(baseline_export.complexity)
            else:
                self.stats["sn_cosine_law_accuracy_base_retained"] += 1
        elif self.use_sn_relativistic_rational_accuracy_export:
            self.stats["sn_relativistic_rational_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_relativistic_rational_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_relativistic_rational_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_relativistic_rational_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_relativistic_rational_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_relativistic_rational_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_relativistic_rational_accuracy_base_retained"] += 1
        elif self.use_sn_shared_denominator_accuracy_export:
            self.stats["sn_shared_denominator_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_shared_denominator_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_shared_denominator_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_shared_denominator_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_shared_denominator_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_shared_denominator_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_shared_denominator_accuracy_base_retained"] += 1
        elif self.use_sn_cross_unary_affine_accuracy_export:
            self.stats["sn_cross_unary_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_cross_unary_affine_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_cross_unary_affine_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_cross_unary_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_cross_unary_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats["sn_cross_unary_accuracy_complexity_delta_sum"] += float(
                        accuracy_candidate.complexity
                    ) - float(baseline_export.complexity)
            else:
                self.stats["sn_cross_unary_accuracy_base_retained"] += 1
        elif self.use_sn_reciprocal_trig_accuracy_export:
            self.stats["sn_reciprocal_trig_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_reciprocal_trig_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_reciprocal_trig_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_reciprocal_trig_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_reciprocal_trig_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_reciprocal_trig_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_reciprocal_trig_accuracy_base_retained"] += 1
        elif self.use_sn_shared_unary_polynomial_accuracy_export:
            self.stats["sn_shared_unary_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_shared_unary_polynomial_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_shared_unary_polynomial_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_shared_unary_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_shared_unary_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_shared_unary_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_shared_unary_accuracy_base_retained"] += 1
        elif self.use_sn_sinc_squared_accuracy_export:
            self.stats["sn_sinc_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_sinc_squared_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_sinc_squared_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_sinc_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_sinc_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats["sn_sinc_accuracy_complexity_delta_sum"] += float(
                        accuracy_candidate.complexity
                    ) - float(baseline_export.complexity)
            else:
                self.stats["sn_sinc_accuracy_base_retained"] += 1
        elif self.use_sn_affine_gaussian_accuracy_export:
            self.stats["sn_affine_gaussian_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_affine_gaussian_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_affine_gaussian_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_affine_gaussian_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_affine_gaussian_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_affine_gaussian_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_affine_gaussian_accuracy_base_retained"] += 1
        elif self.use_sn_damped_exponential_accuracy_export:
            self.stats["sn_exponential_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_damped_exponential_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_damped_exponential_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_exponential_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_exponential_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats["sn_exponential_accuracy_complexity_delta_sum"] += float(
                        accuracy_candidate.complexity
                    ) - float(baseline_export.complexity)
            else:
                self.stats["sn_exponential_accuracy_base_retained"] += 1
        elif self.use_sn_shared_phase_rational_accuracy_export:
            self.stats["sn_shared_phase_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_shared_phase_rational_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_shared_phase_rational_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_shared_phase_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_shared_phase_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_shared_phase_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_shared_phase_accuracy_base_retained"] += 1
        elif self.use_sn_direct_radical_phase_accuracy_export:
            self.stats["sn_radical_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_direct_radical_phase_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_radical_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_radical_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_radical_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats["sn_radical_accuracy_complexity_delta_sum"] += float(
                        accuracy_candidate.complexity
                    ) - float(baseline_export.complexity)
            else:
                self.stats["sn_radical_accuracy_base_retained"] += 1
        elif self.use_sn_direct_phase_interaction_accuracy_export:
            self.stats["sn_phase_interaction_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_direct_phase_interaction_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_phase_interaction_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_phase_interaction_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_phase_interaction_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_phase_interaction_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_phase_interaction_accuracy_base_retained"] += 1
        elif self.use_sn_direct_phase_accuracy_export:
            self.stats["sn_direct_phase_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_direct_phase_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_direct_phase_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_direct_phase_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_direct_phase_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_direct_phase_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_direct_phase_accuracy_base_retained"] += 1
        elif self.use_sn_direct_feature_product_accuracy_export:
            self.stats["sn_direct_feature_product_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_direct_feature_product_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_direct_feature_product_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_direct_feature_product_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_direct_feature_product_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_direct_feature_product_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_direct_feature_product_accuracy_base_retained"] += 1
        elif self.use_sn_direct_feature_affine_unary_accuracy_export:
            self.stats["sn_direct_feature_affine_unary_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_direct_feature_affine_unary_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_direct_feature_affine_unary_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats[
                    "sn_direct_feature_affine_unary_accuracy_export_selected"
                ] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_direct_feature_affine_unary_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_direct_feature_affine_unary_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_direct_feature_affine_unary_accuracy_base_retained"] += 1
        elif self.use_sn_direct_feature_radial_accuracy_export:
            self.stats["sn_direct_feature_radial_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_direct_feature_radial_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_direct_feature_radial_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_direct_feature_radial_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_direct_feature_radial_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_direct_feature_radial_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_direct_feature_radial_accuracy_base_retained"] += 1
        elif self.use_sn_archive_feature_radial_accuracy_export:
            self.stats["sn_archive_feature_radial_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_archive_feature_radial_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_feature_radial_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_archive_feature_radial_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_archive_feature_radial_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_archive_feature_radial_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_archive_feature_radial_accuracy_base_retained"] += 1
        elif self.use_sn_archive_conditional_radial_accuracy_export:
            self.stats["sn_archive_conditional_radial_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_archive_conditional_radial_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_conditional_radial_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats[
                    "sn_archive_conditional_radial_accuracy_export_selected"
                ] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_archive_conditional_radial_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_archive_conditional_radial_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_archive_conditional_radial_accuracy_base_retained"] += 1
        elif self.use_sn_archive_conditional_affine_unary_accuracy_export:
            self.stats[
                "sn_archive_conditional_affine_unary_accuracy_export_events"
            ] += 1
            baseline_export = selected
            self._build_archive_conditional_affine_unary_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_conditional_affine_unary_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats[
                    "sn_archive_conditional_affine_unary_accuracy_export_selected"
                ] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_archive_conditional_affine_unary_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_archive_conditional_affine_unary_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats[
                    "sn_archive_conditional_affine_unary_accuracy_base_retained"
                ] += 1
        elif self.use_sn_archive_conditional_rational_accuracy_export:
            self.stats["sn_archive_conditional_rational_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_archive_conditional_rational_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_conditional_rational_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats[
                    "sn_archive_conditional_rational_accuracy_export_selected"
                ] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_archive_conditional_rational_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_archive_conditional_rational_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats[
                    "sn_archive_conditional_rational_accuracy_base_retained"
                ] += 1
        elif self.use_sn_archive_conditional_unary_accuracy_export:
            self.stats["sn_archive_conditional_unary_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_archive_conditional_unary_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_conditional_unary_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_archive_conditional_unary_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_archive_conditional_unary_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_archive_conditional_unary_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_archive_conditional_unary_accuracy_base_retained"] += 1
        elif self.use_sn_archive_iterated_conditional_product_accuracy_export:
            self.stats["sn_archive_iterated_product_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_archive_iterated_conditional_product_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_iterated_product_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_archive_iterated_product_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_archive_iterated_product_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_archive_iterated_product_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_archive_iterated_product_accuracy_base_retained"] += 1
        elif self.use_sn_archive_conditional_product_accuracy_export:
            self.stats["sn_archive_conditional_product_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_archive_conditional_product_accuracy_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_conditional_product_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats[
                    "sn_archive_conditional_product_accuracy_export_selected"
                ] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats[
                        "sn_archive_conditional_product_accuracy_r2_gain_sum"
                    ] += float(accuracy_candidate.fit_result.r2) - float(
                        baseline_export.fit_result.r2
                    )
                    self.stats[
                        "sn_archive_conditional_product_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_archive_conditional_product_accuracy_base_retained"] += 1
        elif self.use_sn_archive_value_only_accuracy_export:
            self.stats["sn_archive_value_only_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_archive_value_only_accuracy_export_candidate()
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_value_only_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_archive_value_only_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_archive_value_only_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_archive_value_only_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_archive_value_only_accuracy_base_retained"] += 1
        elif self.use_sn_archive_product_accuracy_export:
            self.stats["sn_archive_product_accuracy_export_events"] += 1
            baseline_export = selected
            self._build_archive_product_lift_envelope_candidate(accuracy_mode=True)
            accuracy_candidate = self.archive_export_candidate
            if accuracy_candidate is not None and (
                baseline_export is None
                or self._accuracy_key(accuracy_candidate)
                < self._accuracy_key(baseline_export)
            ):
                accuracy_candidate.variation_type = (
                    "sn_archive_product_accuracy_export_selected"
                )
                selected = accuracy_candidate
                self.stats["sn_archive_product_accuracy_export_selected"] += 1
                if (
                    baseline_export is not None
                    and baseline_export.fit_result is not None
                    and accuracy_candidate.fit_result is not None
                ):
                    self.stats["sn_archive_product_accuracy_r2_gain_sum"] += float(
                        accuracy_candidate.fit_result.r2
                    ) - float(baseline_export.fit_result.r2)
                    self.stats[
                        "sn_archive_product_accuracy_complexity_delta_sum"
                    ] += float(accuracy_candidate.complexity) - float(
                        baseline_export.complexity
                    )
            else:
                self.stats["sn_archive_product_accuracy_base_retained"] += 1
        elif self.use_sn_archive_product_lift_beam_export:
            self._build_archive_product_lift_envelope_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_product_lift_export_selected"] += 1
        elif self.use_sn_archive_pair_lookahead_beam_export:
            self._build_archive_pair_lookahead_envelope_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_pair_lookahead_export_selected"] += 1
        elif self.use_sn_archive_innovation_screen_beam_export:
            self._build_archive_innovation_screen_envelope_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_innovation_screen_export_selected"] += 1
        elif self.use_sn_archive_balanced_floating_beam_export:
            self._build_archive_balanced_floating_envelope_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_balanced_floating_export_selected"] += 1
        elif self.use_sn_archive_balanced_screen_dual_beam_export:
            self._build_archive_balanced_screen_dual_beam_export_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_balanced_screen_export_selected"] += 1
        elif self.use_sn_archive_floating_dual_beam_export:
            self._build_archive_floating_dual_beam_export_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_floating_export_selected"] += 1
        elif self.use_sn_archive_refined_dual_beam_export:
            self._build_archive_refined_dual_beam_export_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_refined_export_selected"] += 1
        elif self.use_sn_archive_dual_pool_beam_export:
            self._build_archive_dual_pool_beam_export_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_dual_pool_export_selected"] += 1
        elif self.use_sn_archive_dual_beam_export:
            self._build_archive_dual_beam_export_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_dual_beam_export_selected"] += 1
        elif self.use_sn_archive_beam_export:
            self._build_archive_beam_export_candidate()
            if self.archive_export_pareto_candidate is not None:
                selected = self.archive_export_pareto_candidate
                self.stats["sn_archive_beam_export_selected"] += 1
        if self.use_sn_gp_stable and self.sn_export_mode == "base_anchor":
            selected = self.base_quality_best_ever or self.method_selected_best
            self.stats["sn_stable_exported_base_anchor"] += int(selected is not None)
        elif (
            self.use_sn_gp_plugin
            and self.sn_comparator is not None
            and self.method_selected_best is not None
            and self.base_quality_best_ever is not None
        ):
            selected = self.sn_comparator.compare(
                self.method_selected_best,
                self.base_quality_best_ever,
                context="export",
            ).winner
        self.exported_candidate = None if selected is None else selected.copy()
        if selected is not None and selected.phi is not None:
            self.eqtree = selected.phi.copy()

    def _closest_parent_geometry(self, candidate: ControlledIndividual) -> str | None:
        child = Counter(candidate.term_keys)
        choices = []
        for parent_id, key, parent_terms in candidate.parent_geometry_hints:
            if key is None:
                continue
            parent = Counter(parent_terms)
            reused = sum((child & parent).values())
            added = sum((child - parent).values())
            removed = sum((parent - child).values())
            changed = added + removed
            choices.append((changed, -reused, parent_id, key))
        if not choices:
            return None
        changed, _, _, key = min(choices)
        return key if changed <= self.incremental_max_changed_terms else None

    def _tournament_one(
        self,
        population: Sequence[ControlledIndividual],
        rng: np.random.Generator,
    ) -> ControlledIndividual:
        indices = rng.integers(0, len(population), size=self.tournament_size)
        contestants = [population[int(index)] for index in indices]
        if self.record_reward_gap_samples and len(contestants) >= 2:
            ordered = sorted(contestants, key=self._base_key)
            first = float(ordered[0].base_reward)
            second = float(ordered[1].base_reward)
            if np.isfinite(first) and np.isfinite(second):
                self.reward_gap_samples["tournament_gaps"].append(abs(first - second))
                self.reward_gap_samples["tournament_scales"].append(
                    max(abs(first), abs(second))
                )
        if self.use_sn_gp_plugin:
            assert self.sn_comparator is not None
            base_winner = min(contestants, key=self._base_key)
            winner = self.sn_comparator.best(contestants, context="tournament")
            changed = winner.candidate_id != base_winner.candidate_id
            self.stats["sn_v2_tournament_winner_changes"] += int(changed)
            self.stats["sn_v2_tournaments"] += 1
            if self.use_sn_gp_stable:
                self.stats["sn_stable_tournament_winner_changes"] += int(changed)
                self.stats["sn_stable_tournaments"] += 1
            if winner.variation_type in {
                "subtree-mutation",
                "hoist-mutation",
                "point-mutation",
            }:
                self.sn_mutation_stats["offspring_selected_as_tournament_parent"] += 1
                if winner.sn_targeted_mutation:
                    self.sn_mutation_stats[
                        "targeted_offspring_selected_as_tournament_parent"
                    ] += 1
            return winner
        return min(contestants, key=self._base_key)

    def _rank_base(
        self, candidates: Sequence[ControlledIndividual]
    ) -> list[ControlledIndividual]:
        return sorted(candidates, key=self._base_key)

    def _rank_structural(
        self, candidates: Sequence[ControlledIndividual]
    ) -> list[ControlledIndividual]:
        return sorted(candidates, key=self._structural_key)

    @staticmethod
    def _base_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
        reward = (
            candidate.base_reward if np.isfinite(candidate.base_reward) else -math.inf
        )
        complexity = (
            candidate.complexity if np.isfinite(candidate.complexity) else math.inf
        )
        return (
            not candidate.valid,
            -reward,
            complexity,
            candidate.canonical_fitted_expression,
            candidate.candidate_id,
        )

    @staticmethod
    def _accuracy_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
        fit = candidate.fit_result
        r2 = float(fit.r2) if fit is not None and np.isfinite(fit.r2) else -math.inf
        complexity = (
            candidate.complexity if np.isfinite(candidate.complexity) else math.inf
        )
        reward = (
            candidate.base_reward if np.isfinite(candidate.base_reward) else -math.inf
        )
        return (
            not candidate.valid,
            -r2,
            complexity,
            -reward,
            candidate.canonical_fitted_expression,
            candidate.candidate_id,
        )

    @staticmethod
    def _structural_key(candidate: ControlledIndividual) -> tuple[Any, ...]:
        final = (
            candidate.final_reward
            if candidate.final_reward is not None
            and np.isfinite(candidate.final_reward)
            else -math.inf
        )
        base = (
            candidate.base_reward if np.isfinite(candidate.base_reward) else -math.inf
        )
        complexity = (
            candidate.complexity if np.isfinite(candidate.complexity) else math.inf
        )
        return (
            -final,
            -base,
            complexity,
            candidate.canonical_fitted_expression,
            candidate.candidate_id,
        )

    def _candidate_summary(
        self, candidate: ControlledIndividual | None
    ) -> dict[str, Any] | None:
        if candidate is None:
            return None
        fit = candidate.fit_result
        analysis = candidate.sobolev_result
        return {
            "candidate_id": candidate.candidate_id,
            "generation": candidate.generation,
            "slot": candidate.slot,
            "parent_ids": list(candidate.parent_ids),
            "variation_type": candidate.variation_type,
            "basis_forest_status": candidate.basis_forest_status,
            "basis_gene_count": (
                None
                if candidate.basis_genome is None
                else len(candidate.basis_genome.terms)
            ),
            "basis_opaque_gene_count": (
                None
                if candidate.basis_genome is None
                else candidate.basis_genome.opaque_count
            ),
            "raw_expression": candidate.raw_expression,
            "exported_expression": (
                None if fit is None else fit.fitted_expression_text
            ),
            "search_internal_r2": self._finite(None if fit is None else fit.r2),
            "design_matrix_r2": self._finite(None if fit is None else fit.design_r2),
            "base_score_semantics": (None if fit is None else fit.score_semantics),
            "nonfinite_export_predictions": (
                None if fit is None else fit.nonfinite_export_predictions
            ),
            "mse": self._finite(None if fit is None else fit.mse),
            "complexity": None if fit is None or not fit.success else fit.complexity,
            "base_reward": self._finite(candidate.base_reward),
            "structural_reward": self._finite(candidate.final_reward),
            "base_rank": candidate.base_rank,
            "structural_rank": candidate.structural_rank,
            "shortlist_status": candidate.shortlist_status,
            "term_count": 0 if fit is None else len(fit.terms),
            "sobolev_success": candidate.sobolev_success,
            "sobolev_penalty": self._finite(candidate.sobolev_penalty),
            "min_novelty": self._finite(
                None if analysis is None else analysis.min_novelty
            ),
            "mean_novelty": self._finite(
                None if analysis is None else analysis.mean_novelty
            ),
            "low_novelty_count": (
                None if analysis is None else int(analysis.low_novelty_count)
            ),
            "low_novelty_ratio": self._finite(
                None if analysis is None else analysis.low_novelty_ratio
            ),
            "sobolev_failure_type": (
                None
                if analysis is None or analysis.success
                else analysis.failure_type.value
            ),
            "sobolev_evaluated": candidate.sobolev_evaluated,
            "sobolev_evaluation_failure": candidate.sobolev_evaluation_failure,
            "provenance_success": (
                None
                if candidate.term_provenance is None
                else candidate.term_provenance.success
            ),
            "prune_origin_candidate_id": candidate.prune_origin_candidate_id,
        }

    def _telemetry(self) -> dict[str, Any]:
        additive = self.additive_evaluator
        return {
            "counters": dict(sorted(self.stats.items())),
            "base_cache": {
                "entries": 0 if additive is None else additive.entry_count,
                "hits": 0 if additive is None else additive.hits,
                "misses": 0 if additive is None else additive.misses,
                "evictions": 0 if additive is None else additive.evictions,
            },
            "term_cache": {
                "entries": self.term_cache.entry_count,
                "derivative_entries": self.term_cache.derivative_entry_count,
                "memory_bytes": self.term_cache.memory_bytes,
                "hits": self.term_cache.hits,
                "misses": self.term_cache.misses,
                "symbolic_hits": self.term_cache.symbolic_hits,
                "symbolic_misses": self.term_cache.symbolic_misses,
                "evictions": self.term_cache.evictions,
                "symbolic_evictions": self.term_cache.symbolic_evictions,
            },
            "geometry_cache": {
                "entries": self.geometry_cache.entry_count,
                "decomposition_entries": self.geometry_cache.decomposition_entry_count,
                "memory_bytes": self.geometry_cache.memory_bytes,
                "hits": self.geometry_cache.hits,
                "misses": self.geometry_cache.misses,
                "evictions": self.geometry_cache.evictions,
                "incremental_hits": self.geometry_cache.incremental_hits,
            },
            "sn_comparator": (
                {} if self.sn_comparator is None else self.sn_comparator.stats_dict()
            ),
            "sn_mutation": dict(sorted(self.sn_mutation_stats.items())),
            "reward_gap_audit": self._reward_gap_summary(),
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        }

    def _reward_gap_summary(self) -> dict[str, Any]:
        if not self.record_reward_gap_samples:
            return {}

        def describe(values: Sequence[float]) -> dict[str, Any]:
            array = np.asarray(values, dtype=float)
            if array.size == 0:
                return {"count": 0, "quantiles": {}}
            levels = (0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0)
            quantiles = np.quantile(array, levels)
            return {
                "count": int(array.size),
                "zero_count": int(np.count_nonzero(array == 0.0)),
                "quantiles": {
                    f"q{int(level * 100):02d}": float(value)
                    for level, value in zip(levels, quantiles, strict=True)
                },
            }

        output = {
            "base_rewards": describe(self.reward_gap_samples["base_rewards"]),
            "tournament_gaps": describe(self.reward_gap_samples["tournament_gaps"]),
            "elite_boundary_gaps": describe(
                self.reward_gap_samples["elite_boundary_gaps"]
            ),
        }
        for prefix in ("tournament", "elite_boundary"):
            gaps = np.asarray(self.reward_gap_samples[f"{prefix}_gaps"], dtype=float)
            scales = np.asarray(
                self.reward_gap_samples[f"{prefix}_scales"], dtype=float
            )
            if gaps.size:
                thresholds = (
                    self.sn_compare_epsilon_abs + self.sn_compare_epsilon_rel * scales
                )
                output[f"{prefix}_gaps"]["epsilon_equivalent_count"] = int(
                    np.count_nonzero(gaps <= thresholds)
                )
                output[f"{prefix}_gaps"]["epsilon_equivalent_fraction"] = float(
                    np.mean(gaps <= thresholds)
                )
        return output

    def _configuration_dict(self) -> dict[str, Any]:
        configuration = {
            "protocol": "controlled_population_gp_sobolev_v1",
            "profile": self.profile,
            "basis_forest_enabled": self.use_basis_forest,
            "basis_forest_sn_crossover_enabled": (self.use_basis_forest_sn_crossover),
            "basis_forest_sn_residual_infusion_enabled": (
                self.use_basis_forest_sn_residual_infusion
            ),
            "basis_forest_sn_safe_mutation_enabled": (
                self.use_basis_forest_sn_safe_mutation
            ),
            "basis_forest_sn_verified_compression_enabled": (
                self.use_basis_forest_sn_verified_compression
            ),
            "basis_forest_sn_residual_shadow_enabled": (
                self.use_basis_forest_sn_residual_shadow
            ),
            "basis_forest_sn_mutation_recombination_enabled": (
                self.use_basis_forest_sn_mutation_recombination
            ),
            "basis_forest_sn_mutation_shadow_enabled": (
                self.use_basis_forest_sn_mutation_shadow
            ),
            "population_size": self.population_size,
            "elitism_k": self.elitism_k,
            "tournament_size": self.tournament_size,
            "method_probabilities": dict(self.method_probs),
            "p_point_replace": self.p_point_replace,
            "depth_range": list(self.depth_range),
            "full_prob": self.full_prob,
            "p_bfgs": 0.0,
            "fixed_constants": list(self.fixed_constants),
            "binary": [value.__name__ for value in self.binary],
            "unary": [value.__name__ for value in self.unary],
            "n_iter": self.n_iter,
            "time_limit": self.time_limit,
            "hard_time_limit": self.hard_time_limit,
            "max_len": self.max_len,
            "max_additive_terms": self.max_additive_terms,
            "eta": self.eta,
            "ratio": self.ratio,
            "early_stop": False,
            "n_jobs": 1,
            "sobolev_alpha": self.sobolev_alpha,
            "sobolev_tau": self.sobolev_tau,
            "sobolev_lambda_value": self.sobolev_lambda_value,
            "sobolev_lambda_gradient": self.sobolev_lambda_gradient,
            "geometry_sample_size": self.geometry_sample_size,
            "shortlist_size": self.shortlist_size,
            "sobolev_failure_policy": self.sobolev_failure_policy,
            "sobolev_pruning": self.sobolev_pruning,
            "sobolev_max_prunes": self.sobolev_max_prunes,
            "prune_elite_k": self.prune_elite_k,
            "sobolev_acceptance_tolerance": self.sobolev_acceptance_tolerance,
            "sn_gp_v2_enabled": self.use_sn_gp_v2,
            "sn_gp_stable_enabled": self.use_sn_gp_stable,
            "sn_gp_verified_repair_enabled": self.use_sn_gp_verified_repair,
            "sn_repair_shadow_enabled": self.use_sn_repair_shadow,
            "sn_archive_export_enabled": self.use_sn_archive_export,
            "sn_archive_export_max_steps": self.sn_archive_export_max_steps,
            "sn_archive_anchor_export_enabled": (self.use_sn_archive_anchor_export),
            "sn_archive_anchor_export_max_steps": (
                self.sn_archive_anchor_export_max_steps
            ),
            "sn_archive_beam_export_enabled": self.use_sn_archive_beam_export,
            "sn_archive_dual_beam_export_enabled": (
                self.use_sn_archive_dual_beam_export
            ),
            "sn_archive_dual_pool_beam_export_enabled": (
                self.use_sn_archive_dual_pool_beam_export
            ),
            "sn_archive_refined_dual_beam_export_enabled": (
                self.use_sn_archive_refined_dual_beam_export
            ),
            "sn_archive_floating_dual_beam_export_enabled": (
                self.use_sn_archive_floating_dual_beam_export
            ),
            "sn_archive_balanced_screen_dual_beam_export_enabled": (
                self.use_sn_archive_balanced_screen_dual_beam_export
            ),
            "sn_archive_balanced_floating_beam_export_enabled": (
                self.use_sn_archive_balanced_floating_beam_export
            ),
            "sn_archive_innovation_screen_beam_export_enabled": (
                self.use_sn_archive_innovation_screen_beam_export
            ),
            "sn_archive_pair_lookahead_beam_export_enabled": (
                self.use_sn_archive_pair_lookahead_beam_export
            ),
            "sn_archive_product_lift_beam_export_enabled": (
                self.use_sn_archive_product_lift_beam_export
            ),
            "sn_archive_product_accuracy_export_enabled": (
                self.use_sn_archive_product_accuracy_export
            ),
            "sn_archive_conditional_product_accuracy_export_enabled": (
                self.use_sn_archive_conditional_product_accuracy_export
            ),
            "sn_archive_iterated_conditional_product_accuracy_export_enabled": (
                self.use_sn_archive_iterated_conditional_product_accuracy_export
            ),
            "sn_archive_conditional_unary_accuracy_export_enabled": (
                self.use_sn_archive_conditional_unary_accuracy_export
            ),
            "sn_archive_conditional_rational_accuracy_export_enabled": (
                self.use_sn_archive_conditional_rational_accuracy_export
            ),
            "sn_archive_conditional_affine_unary_accuracy_export_enabled": (
                self.use_sn_archive_conditional_affine_unary_accuracy_export
            ),
            "sn_archive_conditional_radial_accuracy_export_enabled": (
                self.use_sn_archive_conditional_radial_accuracy_export
            ),
            "sn_archive_feature_radial_accuracy_export_enabled": (
                self.use_sn_archive_feature_radial_accuracy_export
            ),
            "sn_direct_feature_radial_accuracy_export_enabled": (
                self.use_sn_direct_feature_radial_accuracy_export
            ),
            "sn_direct_feature_affine_unary_accuracy_export_enabled": (
                self.use_sn_direct_feature_affine_unary_accuracy_export
            ),
            "sn_direct_feature_product_accuracy_export_enabled": (
                self.use_sn_direct_feature_product_accuracy_export
            ),
            "sn_direct_phase_accuracy_export_enabled": (
                self.use_sn_direct_phase_accuracy_export
            ),
            "sn_direct_phase_interaction_accuracy_export_enabled": (
                self.use_sn_direct_phase_interaction_accuracy_export
            ),
            "sn_direct_radical_phase_accuracy_export_enabled": (
                self.use_sn_direct_radical_phase_accuracy_export
            ),
            "sn_shared_phase_rational_accuracy_export_enabled": (
                self.use_sn_shared_phase_rational_accuracy_export
            ),
            "sn_damped_exponential_accuracy_export_enabled": (
                self.use_sn_damped_exponential_accuracy_export
            ),
            "sn_affine_gaussian_accuracy_export_enabled": (
                self.use_sn_affine_gaussian_accuracy_export
            ),
            "sn_sinc_squared_accuracy_export_enabled": (
                self.use_sn_sinc_squared_accuracy_export
            ),
            "sn_shared_unary_polynomial_accuracy_export_enabled": (
                self.use_sn_shared_unary_polynomial_accuracy_export
            ),
            "sn_reciprocal_trig_accuracy_export_enabled": (
                self.use_sn_reciprocal_trig_accuracy_export
            ),
            "sn_cross_unary_affine_accuracy_export_enabled": (
                self.use_sn_cross_unary_affine_accuracy_export
            ),
            "sn_shared_denominator_accuracy_export_enabled": (
                self.use_sn_shared_denominator_accuracy_export
            ),
            "sn_relativistic_rational_accuracy_export_enabled": (
                self.use_sn_relativistic_rational_accuracy_export
            ),
            "sn_cosine_law_radial_accuracy_export_enabled": (
                self.use_sn_cosine_law_radial_accuracy_export
            ),
            "sn_reciprocal_sine_square_accuracy_export_enabled": (
                self.use_sn_reciprocal_sine_square_accuracy_export
            ),
            "sn_inverse_cosine_radial_accuracy_export_enabled": (
                self.use_sn_inverse_cosine_radial_accuracy_export
            ),
            "sn_multiaxis_inverse_square_accuracy_export_enabled": (
                self.use_sn_multiaxis_inverse_square_accuracy_export
            ),
            "sn_interference_sine_ratio_accuracy_export_enabled": (
                self.use_sn_interference_sine_ratio_accuracy_export
            ),
            "sn_sparse_radical_accuracy_export_enabled": (
                self.use_sn_sparse_radical_accuracy_export
            ),
            "sn_shared_ratio_trig_polynomial_accuracy_export_enabled": (
                self.use_sn_shared_ratio_trig_polynomial_accuracy_export
            ),
            "sn_deep_accuracy_stage": (
                None
                if self.deep_accuracy_stage is None
                else self.deep_accuracy_stage.code
            ),
            "sn_archive_value_only_accuracy_export_enabled": (
                self.use_sn_archive_value_only_accuracy_export
            ),
            "sn_archive_beam_export_max_steps": (self.sn_archive_beam_export_max_steps),
            "sn_archive_beam_width": self.sn_archive_beam_width,
            "sn_archive_beam_shortlist_size": (self.sn_archive_beam_shortlist_size),
            "sn_archive_beam_value_shortlist_size": (
                self.sn_archive_beam_value_shortlist_size
            ),
            "sn_archive_beam_innovation_shortlist_size": (
                self.sn_archive_beam_innovation_shortlist_size
            ),
            "sn_archive_beam_pair_first_shortlist_size": (
                self.sn_archive_beam_pair_first_shortlist_size
            ),
            "sn_archive_beam_pair_second_shortlist_size": (
                self.sn_archive_beam_pair_second_shortlist_size
            ),
            "sn_archive_interaction_joint_shortlist_size": (
                self.sn_archive_interaction_joint_shortlist_size
            ),
            "sn_archive_interaction_value_shortlist_size": (
                self.sn_archive_interaction_value_shortlist_size
            ),
            "sn_archive_unary_transforms": list(self.sn_archive_unary_transforms),
            "sn_direct_phase_scales": list(self.sn_direct_phase_scales),
            "sn_direct_phase_include_squares": (self.sn_direct_phase_include_squares),
            "sn_phase_interaction_batch_size": self.sn_phase_interaction_batch_size,
            "sn_radical_scales": list(self.sn_radical_scales),
            "sn_radical_max_dimension": self.sn_radical_max_dimension,
            "sn_radical_phase_anchor_limit": self.sn_radical_phase_anchor_limit,
            "sn_radical_amplitude_joint_shortlist_size": (
                self.sn_radical_amplitude_joint_shortlist_size
            ),
            "sn_radical_amplitude_value_shortlist_size": (
                self.sn_radical_amplitude_value_shortlist_size
            ),
            "sn_shared_phase_mobius_shifts": list(self.sn_shared_phase_mobius_shifts),
            "sn_shared_phase_max_dimension": self.sn_shared_phase_max_dimension,
            "sn_shared_phase_max_numerator_degree": (
                self.sn_shared_phase_max_numerator_degree
            ),
            "sn_shared_phase_max_denominator_degree": (
                self.sn_shared_phase_max_denominator_degree
            ),
            "sn_shared_phase_backbone_value_pool_size": (
                self.sn_shared_phase_backbone_value_pool_size
            ),
            "sn_shared_phase_backbone_shortlist_size": (
                self.sn_shared_phase_backbone_shortlist_size
            ),
            "sn_shared_phase_modulated_shortlist_size": (
                self.sn_shared_phase_modulated_shortlist_size
            ),
            "sn_shared_phase_complement_pool_size": (
                self.sn_shared_phase_complement_pool_size
            ),
            "sn_shared_phase_complement_shortlist_size": (
                self.sn_shared_phase_complement_shortlist_size
            ),
            "sn_shared_phase_proposal_limit": self.sn_shared_phase_proposal_limit,
            "sn_exponential_scales": list(self.sn_exponential_scales),
            "sn_exponential_max_dimension": self.sn_exponential_max_dimension,
            "sn_exponential_max_numerator_degree": (
                self.sn_exponential_max_numerator_degree
            ),
            "sn_exponential_max_denominator_degree": (
                self.sn_exponential_max_denominator_degree
            ),
            "sn_exponential_value_pool_size": (self.sn_exponential_value_pool_size),
            "sn_exponential_shortlist_size": self.sn_exponential_shortlist_size,
            "sn_exponential_composite_value_pool_size": (
                self.sn_exponential_composite_value_pool_size
            ),
            "sn_exponential_composite_shortlist_size": (
                self.sn_exponential_composite_shortlist_size
            ),
            "sn_exponential_proposal_limit": self.sn_exponential_proposal_limit,
            "sn_affine_gaussian_scales": list(self.sn_affine_gaussian_scales),
            "sn_affine_gaussian_max_dimension": (self.sn_affine_gaussian_max_dimension),
            "sn_affine_gaussian_value_pool_size": (
                self.sn_affine_gaussian_value_pool_size
            ),
            "sn_affine_gaussian_shortlist_size": (
                self.sn_affine_gaussian_shortlist_size
            ),
            "sn_affine_gaussian_composite_value_pool_size": (
                self.sn_affine_gaussian_composite_value_pool_size
            ),
            "sn_affine_gaussian_composite_shortlist_size": (
                self.sn_affine_gaussian_composite_shortlist_size
            ),
            "sn_affine_gaussian_proposal_limit": (
                self.sn_affine_gaussian_proposal_limit
            ),
            "sn_sinc_scales": list(self.sn_sinc_scales),
            "sn_sinc_max_dimension": self.sn_sinc_max_dimension,
            "sn_sinc_max_numerator_degree": (self.sn_sinc_max_numerator_degree),
            "sn_sinc_max_denominator_degree": (self.sn_sinc_max_denominator_degree),
            "sn_sinc_shape_value_pool_size": (self.sn_sinc_shape_value_pool_size),
            "sn_sinc_shape_shortlist_size": (self.sn_sinc_shape_shortlist_size),
            "sn_sinc_composite_value_pool_size": (
                self.sn_sinc_composite_value_pool_size
            ),
            "sn_sinc_composite_shortlist_size": (self.sn_sinc_composite_shortlist_size),
            "sn_sinc_proposal_limit": self.sn_sinc_proposal_limit,
            "sn_shared_unary_scales": list(self.sn_shared_unary_scales),
            "sn_shared_unary_transforms": list(self.sn_shared_unary_transforms),
            "sn_shared_unary_max_dimension": (self.sn_shared_unary_max_dimension),
            "sn_shared_unary_max_numerator_degree": (
                self.sn_shared_unary_max_numerator_degree
            ),
            "sn_shared_unary_max_denominator_degree": (
                self.sn_shared_unary_max_denominator_degree
            ),
            "sn_shared_unary_phase_value_pool_size": (
                self.sn_shared_unary_phase_value_pool_size
            ),
            "sn_shared_unary_phase_shortlist_size": (
                self.sn_shared_unary_phase_shortlist_size
            ),
            "sn_shared_unary_amplitude_value_pool_size": (
                self.sn_shared_unary_amplitude_value_pool_size
            ),
            "sn_shared_unary_amplitude_shortlist_size": (
                self.sn_shared_unary_amplitude_shortlist_size
            ),
            "sn_shared_unary_proposal_limit": (self.sn_shared_unary_proposal_limit),
            "sn_reciprocal_trig_scales": list(self.sn_reciprocal_trig_scales),
            "sn_reciprocal_trig_transforms": list(self.sn_reciprocal_trig_transforms),
            "sn_reciprocal_trig_max_dimension": (self.sn_reciprocal_trig_max_dimension),
            "sn_reciprocal_trig_value_pool_size": (
                self.sn_reciprocal_trig_value_pool_size
            ),
            "sn_reciprocal_trig_shortlist_size": (
                self.sn_reciprocal_trig_shortlist_size
            ),
            "sn_reciprocal_trig_proposal_limit": (
                self.sn_reciprocal_trig_proposal_limit
            ),
            "sn_cross_unary_scales": list(self.sn_cross_unary_scales),
            "sn_cross_unary_transforms": list(self.sn_cross_unary_transforms),
            "sn_cross_unary_inner_powers": list(self.sn_cross_unary_inner_powers),
            "sn_cross_unary_max_dimension": (self.sn_cross_unary_max_dimension),
            "sn_cross_unary_max_numerator_degree": (
                self.sn_cross_unary_max_numerator_degree
            ),
            "sn_cross_unary_max_denominator_degree": (
                self.sn_cross_unary_max_denominator_degree
            ),
            "sn_cross_unary_value_pool_size": (self.sn_cross_unary_value_pool_size),
            "sn_cross_unary_shortlist_size": (self.sn_cross_unary_shortlist_size),
            "sn_cross_unary_proposal_limit": (self.sn_cross_unary_proposal_limit),
            "sn_shared_denominator_signs": list(self.sn_shared_denominator_signs),
            "sn_shared_denominator_numerator_signs": list(
                self.sn_shared_denominator_numerator_signs
            ),
            "sn_shared_denominator_max_dimension": (
                self.sn_shared_denominator_max_dimension
            ),
            "sn_shared_denominator_value_pool_size": (
                self.sn_shared_denominator_value_pool_size
            ),
            "sn_shared_denominator_shortlist_size": (
                self.sn_shared_denominator_shortlist_size
            ),
            "sn_shared_denominator_proposal_limit": (
                self.sn_shared_denominator_proposal_limit
            ),
            "sn_relativistic_rational_signs": list(self.sn_relativistic_rational_signs),
            "sn_relativistic_rational_scales": list(
                self.sn_relativistic_rational_scales
            ),
            "sn_relativistic_rational_max_dimension": (
                self.sn_relativistic_rational_max_dimension
            ),
            "sn_relativistic_rational_value_pool_size": (
                self.sn_relativistic_rational_value_pool_size
            ),
            "sn_relativistic_rational_shortlist_size": (
                self.sn_relativistic_rational_shortlist_size
            ),
            "sn_relativistic_rational_proposal_limit": (
                self.sn_relativistic_rational_proposal_limit
            ),
            "sn_cosine_law_phase_signs": list(self.sn_cosine_law_phase_signs),
            "sn_cosine_law_radial_signs": list(self.sn_cosine_law_radial_signs),
            "sn_cosine_law_phase_scales": list(self.sn_cosine_law_phase_scales),
            "sn_cosine_law_max_dimension": self.sn_cosine_law_max_dimension,
            "sn_cosine_law_value_pool_size": self.sn_cosine_law_value_pool_size,
            "sn_cosine_law_shortlist_size": self.sn_cosine_law_shortlist_size,
            "sn_cosine_law_proposal_limit": self.sn_cosine_law_proposal_limit,
            "sn_reciprocal_sine_scales": list(self.sn_reciprocal_sine_scales),
            "sn_reciprocal_sine_max_dimension": (self.sn_reciprocal_sine_max_dimension),
            "sn_reciprocal_sine_max_numerator_degree": (
                self.sn_reciprocal_sine_max_numerator_degree
            ),
            "sn_reciprocal_sine_value_pool_size": (
                self.sn_reciprocal_sine_value_pool_size
            ),
            "sn_reciprocal_sine_shortlist_size": (
                self.sn_reciprocal_sine_shortlist_size
            ),
            "sn_reciprocal_sine_proposal_limit": (
                self.sn_reciprocal_sine_proposal_limit
            ),
            "sn_inverse_cosine_phase_signs": list(self.sn_inverse_cosine_phase_signs),
            "sn_inverse_cosine_radial_signs": list(self.sn_inverse_cosine_radial_signs),
            "sn_inverse_cosine_phase_scales": list(self.sn_inverse_cosine_phase_scales),
            "sn_inverse_cosine_max_dimension": (self.sn_inverse_cosine_max_dimension),
            "sn_inverse_cosine_value_pool_size": (
                self.sn_inverse_cosine_value_pool_size
            ),
            "sn_inverse_cosine_shortlist_size": (self.sn_inverse_cosine_shortlist_size),
            "sn_inverse_cosine_proposal_limit": (self.sn_inverse_cosine_proposal_limit),
            "sn_multiaxis_pair_signs": list(self.sn_multiaxis_pair_signs),
            "sn_multiaxis_radial_term_counts": list(
                self.sn_multiaxis_radial_term_counts
            ),
            "sn_multiaxis_max_dimension": self.sn_multiaxis_max_dimension,
            "sn_multiaxis_max_numerator_degree": (
                self.sn_multiaxis_max_numerator_degree
            ),
            "sn_multiaxis_value_pool_size": self.sn_multiaxis_value_pool_size,
            "sn_multiaxis_shortlist_size": self.sn_multiaxis_shortlist_size,
            "sn_multiaxis_proposal_limit": self.sn_multiaxis_proposal_limit,
            "sn_interference_sine_scales": list(self.sn_interference_sine_scales),
            "sn_interference_sine_max_dimension": (
                self.sn_interference_sine_max_dimension
            ),
            "sn_interference_sine_value_pool_size": (
                self.sn_interference_sine_value_pool_size
            ),
            "sn_interference_sine_shortlist_size": (
                self.sn_interference_sine_shortlist_size
            ),
            "sn_interference_sine_proposal_limit": (
                self.sn_interference_sine_proposal_limit
            ),
            "sn_sparse_radical_offsets": list(self.sn_sparse_radical_offsets),
            "sn_sparse_radical_scales": list(self.sn_sparse_radical_scales),
            "sn_sparse_radical_max_dimension": self.sn_sparse_radical_max_dimension,
            "sn_sparse_radical_max_abs_exponent": (
                self.sn_sparse_radical_max_abs_exponent
            ),
            "sn_sparse_radical_max_numerator_degree": (
                self.sn_sparse_radical_max_numerator_degree
            ),
            "sn_sparse_radical_max_denominator_degree": (
                self.sn_sparse_radical_max_denominator_degree
            ),
            "sn_sparse_radical_integer_candidate_limit": (
                self.sn_sparse_radical_integer_candidate_limit
            ),
            "sn_sparse_radical_value_pool_size": (
                self.sn_sparse_radical_value_pool_size
            ),
            "sn_sparse_radical_shortlist_size": (self.sn_sparse_radical_shortlist_size),
            "sn_sparse_radical_proposal_limit": (self.sn_sparse_radical_proposal_limit),
            "sn_shared_ratio_trig_phase_scales": list(
                self.sn_shared_ratio_trig_phase_scales
            ),
            "sn_shared_ratio_trig_required_dimension": (
                self.sn_shared_ratio_trig_required_dimension
            ),
            "sn_shared_ratio_trig_value_pool_size": (
                self.sn_shared_ratio_trig_value_pool_size
            ),
            "sn_shared_ratio_trig_shortlist_size": (
                self.sn_shared_ratio_trig_shortlist_size
            ),
            "sn_shared_ratio_trig_proposal_limit": (
                self.sn_shared_ratio_trig_proposal_limit
            ),
            "sn_archive_beam_diversity_slots": (self.sn_archive_beam_diversity_slots),
            "sn_archive_beam_refine_max_steps": (self.sn_archive_beam_refine_max_steps),
            "sn_population_coverage_enabled": self.use_sn_population_coverage,
            "sn_verified_coverage_crossover_enabled": (
                self.use_sn_verified_coverage_crossover
            ),
            "sn_child_geometry_enabled": self.use_sn_child_geometry,
            "sn_basis_exchange_enabled": self.use_sn_basis_exchange,
            "sn_basis_infusion_enabled": self.use_sn_basis_infusion,
            "sn_residual_basis_infusion_enabled": (self.use_sn_residual_basis_infusion),
            "sn_basis_archive_enabled": self.use_sn_basis_archive,
            "sn_basis_quality_archive_enabled": (self.use_sn_basis_quality_archive),
            "sn_partial_residual_archive_enabled": (
                self.use_sn_partial_residual_archive
            ),
            "sn_basis_pursuit_enabled": self.use_sn_basis_pursuit,
            "sn_screened_pursuit_shadow_enabled": (self.use_sn_screened_pursuit_shadow),
            "sn_pareto_pursuit_shadow_enabled": (self.use_sn_pareto_pursuit_shadow),
            "sn_stagnation_pursuit_shadow_enabled": (
                self.use_sn_stagnation_pursuit_shadow
            ),
            "sn_basis_pursuit_shortlist_size": (self.sn_basis_pursuit_shortlist_size),
            "sn_pursuit_stagnation_patience": (self.sn_pursuit_stagnation_patience),
            "sn_pursuit_stagnation_epsilon_abs": (
                self.sn_pursuit_stagnation_epsilon_abs
            ),
            "sn_pursuit_stagnation_epsilon_rel": (
                self.sn_pursuit_stagnation_epsilon_rel
            ),
            "sn_pursuit_preserve_fallback_anchor": (
                self.sn_pursuit_preserve_fallback_anchor
            ),
            "sn_pursuit_select_by_coverage": (self.sn_pursuit_select_by_coverage),
            "sn_basis_replacement_enabled": self.use_sn_basis_replacement,
            "sn_basis_replacement_removal_shortlist_size": (
                self.sn_basis_replacement_removal_shortlist_size
            ),
            "sn_conditional_basis_replacement_enabled": (
                self.use_sn_conditional_basis_replacement
            ),
            "sn_orthogonal_basis_crossover_enabled": (
                self.use_sn_orthogonal_basis_crossover
            ),
            "sn_orthogonal_basis_max_steps": self.sn_orthogonal_basis_max_steps,
            "sn_shadow_slot_enabled": self.use_sn_shadow_slot,
            "sn_shadow_max_slots": self.sn_shadow_max_slots,
            "sn_shadow_minimum_coverage_gain": (self.sn_shadow_minimum_coverage_gain),
            "sn_shadow_allow_represented_amplification": (
                self.sn_shadow_allow_represented_amplification
            ),
            "sn_basis_forest_crossover_rate": (self.sn_basis_forest_crossover_rate),
            "sn_basis_archive_capacity": self.sn_basis_archive_capacity,
            "sn_basis_archive_source_candidates": (
                self.sn_basis_archive_source_candidates
            ),
            "sn_basis_quality_archive_capacity": (
                self.sn_basis_quality_archive_capacity
            ),
            "sn_compare_epsilon_abs": self.sn_compare_epsilon_abs,
            "sn_compare_epsilon_rel": self.sn_compare_epsilon_rel,
            "sn_mutation_delta": self.sn_mutation_delta,
            "sn_mutation_gamma": self.sn_mutation_gamma,
            "sn_mutation_impact_beta": self.sn_mutation_impact_beta,
            "sn_mutation_impact_epsilon": self.sn_mutation_impact_epsilon,
            "sn_mutation_max_normalized_impact": (
                self.sn_mutation_max_normalized_impact
            ),
            "sn_base_anchor_enabled": self.sn_base_anchor_enabled,
            "sn_base_anchor_elite_slots": self.sn_base_anchor_elite_slots,
            "sn_export_mode": self.sn_export_mode,
            "sn_tau": self.sobolev_tau,
            "sn_geometry_sample_size": self.geometry_sample_size,
            "sn_cache_enabled": self.sn_cache_enabled,
            "sn_provenance_enabled": self.sn_provenance_enabled,
            "sn_trace_every": self.sn_trace_every,
            "record_reward_gap_samples": self.record_reward_gap_samples,
            "base_score_semantics": self.base_score_semantics,
            "sn_repair_shortlist_size": self.sn_repair_shortlist_size,
            "sn_repair_max_accepted_per_generation": (
                self.sn_repair_max_accepted_per_generation
            ),
            "sn_coverage_shortlist_size": self.sn_coverage_shortlist_size,
            "sn_coverage_elite_slots": self.sn_coverage_elite_slots,
            "sn_coverage_max_base_reward_gap": (self.sn_coverage_max_base_reward_gap),
            "sn_coverage_crossover_rate": self.sn_coverage_crossover_rate,
        }
        for name in DEEP_ACCURACY_PARAMETER_NAMES:
            value = getattr(self, name)
            configuration[name] = list(value) if isinstance(value, tuple) else value
        if self.record_integrity_metadata:
            configuration["rng_schedule_identity"] = self.counter_rng.schedule_identity
        return configuration

    def _raise_if_deadline(self) -> None:
        if self._deadline_reached():
            raise GenerationDeadline()

    def _deadline_reached(self) -> bool:
        return self.start_monotonic is not None and self._elapsed() >= self.time_limit

    def _elapsed(self) -> float:
        return (
            0.0
            if self.start_monotonic is None
            else time.monotonic() - self.start_monotonic
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    def result_document(self) -> dict[str, Any]:
        result = {
            "protocol": "controlled_population_gp_sobolev_result_v1",
            "status": self.status,
            "profile": self.profile,
            "seed": int(self.random_state),
            "configuration": self.configuration,
            "dataset_identity": self.dataset_identity,
            "completed_generations": len(self.records),
            "search_wall_time": self._elapsed(),
            "time_floor_snapshot": self.time_floor_snapshot,
            "method_selected_best": self._candidate_summary(self.method_selected_best),
            "base_quality_best_ever": self._candidate_summary(
                self.base_quality_best_ever
            ),
            "best_by_sn_comparator": self._candidate_summary(self.method_selected_best),
            "best_by_base_reward": self._candidate_summary(self.base_quality_best_ever),
            "exported_candidate": self._candidate_summary(self.exported_candidate),
            "archive_export_candidate": self._candidate_summary(
                self.archive_export_candidate
            ),
            "archive_export_pareto_candidate": self._candidate_summary(
                self.archive_export_pareto_candidate
            ),
            "archive_export_path": self.archive_export_path,
            "final_base_anchored_expression": (
                None
                if self.base_quality_best_ever is None
                or self.base_quality_best_ever.fit_result is None
                else self.base_quality_best_ever.fit_result.fitted_expression_text
            ),
            "final_sn_expression": (
                None
                if self.method_selected_best is None
                or self.method_selected_best.fit_result is None
                else self.method_selected_best.fit_result.fitted_expression_text
            ),
            "sn_export_mode": self.sn_export_mode,
            "telemetry": self._telemetry(),
            "operator_counts": dict(sorted(self.operator_counts.items())),
            "failure_counts": dict(sorted(self.failure_counts.items())),
        }
        if self.record_integrity_metadata:
            result.update(
                {
                    "configuration_sha256": self.configuration_sha256,
                    "initial_population_sha256": self.initial_population_sha256,
                    "initial_base_fitness_sha256": self.initial_base_fitness_sha256,
                }
            )
        return result

    @staticmethod
    def _proposal_id(
        generation: int, slot: int, population_size: int = 1_000_000
    ) -> int:
        # Even IDs are proposals; the immediately following odd ID is reserved
        # for a top-1 accepted prune repair.
        return 2 * (int(generation) * population_size + int(slot))

    @staticmethod
    def _expressions_sha256(expressions: Sequence[str]) -> str:
        return hashlib.sha256(
            canonical_json_bytes([str(value) for value in expressions])
        ).hexdigest()

    @classmethod
    def _base_metrics_sha256(cls, population: Sequence[ControlledIndividual]) -> str:
        payload = [
            {
                "candidate_id": value.candidate_id,
                "valid": value.valid,
                "r2": cls._finite(
                    None if value.fit_result is None else value.fit_result.r2
                ),
                "complexity": (
                    None
                    if value.fit_result is None or not value.valid
                    else value.fit_result.complexity
                ),
                "base_reward": cls._finite(value.base_reward),
                "fitted": value.canonical_fitted_expression,
            }
            for value in population
        ]
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    @staticmethod
    def _population_sha256(population: Sequence[ControlledIndividual]) -> str:
        payload = [
            {
                "candidate_id": value.candidate_id,
                "raw": value.raw_expression,
                "fitted": value.canonical_fitted_expression,
                "parents": list(value.parent_ids),
                "variation": value.variation_type,
            }
            for value in population
        ]
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    @staticmethod
    def _finite(value: float | None) -> float | None:
        if value is None:
            return None
        return float(value) if np.isfinite(value) else None

    @staticmethod
    def _content_identity(X: np.ndarray, y: np.ndarray, names: Sequence[str]) -> str:
        digest = hashlib.sha256()
        digest.update("\0".join(names).encode("utf-8"))
        digest.update(array_identity(X, y).encode("ascii"))
        return f"search_arrays_sha256={digest.hexdigest()}"

    def _coerce_training_inputs(
        self, X: np.ndarray | pd.DataFrame | Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        expected = tuple(variable.name for variable in self.variables)
        if isinstance(X, pd.DataFrame):
            available = tuple(str(value) for value in X.columns)
            if set(available) != set(expected) or len(available) != len(expected):
                raise ValueError(
                    f"DataFrame columns {available} != variables {expected}"
                )
            points = X.loc[:, list(expected)].to_numpy(dtype=float)
            names = expected
        elif isinstance(X, Mapping):
            keys = {str(key): key for key in X}
            if set(keys) != set(expected) or len(keys) != len(expected):
                raise ValueError(f"Mapping keys {tuple(keys)} != variables {expected}")
            columns = [np.asarray(X[keys[name]], dtype=float) for name in expected]
            if any(column.ndim != 1 for column in columns):
                raise ValueError("Mapping inputs must be one-dimensional columns")
            points = np.column_stack(columns)
            names = expected
        else:
            points = np.asarray(X, dtype=float)
            names = expected
        if (
            points.ndim != 2
            or points.shape[1] != len(names)
            or points.shape[0] == 0
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(f"Invalid training input shape/content: {points.shape}")
        return points, names


# Public experiment-facing name; kept separate from nd2py.GP for clarity.
GP = ControlledGP


def build_initial_population_document(
    feature_names: Sequence[str],
    *,
    seed: int,
    population_size: int = 1000,
    max_len: int = 30,
    depth_range: tuple[int, int] = (2, 6),
) -> dict[str, Any]:
    variables = [nd.Variable(str(name), nettype="scalar") for name in feature_names]
    model = ControlledGP(
        variables,
        profile=PROFILE_BASE,
        random_state=seed,
        population_size=population_size,
        elitism_k=min(10, population_size - 1),
        tournament_size=min(20, population_size),
        max_len=max_len,
        depth_range=depth_range,
        time_limit=900.0,
        hard_time_limit=1800.0,
        output_dir=None,
    )
    expressions = model.generate_initial_population_expressions()
    digest = model._expressions_sha256(expressions)
    return {
        "schema": "controlled_gp_initial_population_v1",
        "seed": int(seed),
        "feature_names": [str(value) for value in feature_names],
        "population_size": int(population_size),
        "max_len": int(max_len),
        "depth_range": list(depth_range),
        "binary": [value.__name__ for value in DEFAULT_BINARY],
        "unary": [value.__name__ for value in DEFAULT_UNARY],
        "fixed_constants": list(DEFAULT_FIXED_CONSTANTS),
        "full_prob": 0.5,
        "rng_schedule_identity": model.counter_rng.schedule_identity,
        "expressions": list(expressions),
        "expressions_sha256": digest,
    }
