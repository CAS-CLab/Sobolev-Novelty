"""Immutable registry for the CF--CL deep accuracy-envelope stages.

The registry keeps the append-only exploration profiles and their telemetry
names in one place.  Candidate formation remains in the stage-specific
modules; :mod:`population_gp` only uses this metadata to compose each new
family with the complete predecessor envelope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Mapping

from .sn_additive_radial_coupling import lift_direct_additive_radial_couplings
from .sn_conditional_factored_polynomial import (
    lift_direct_conditional_factored_polynomials,
)
from .sn_coupled_trig_rational import lift_direct_coupled_trig_rationals
from .sn_inverse_trig_mobius import lift_direct_inverse_trig_mobius
from .sn_nested_affine_radical import lift_direct_nested_affine_radicals
from .sn_phase_modulated_rational import lift_direct_phase_modulated_rationals
from .sn_relativistic_trig_rational import (
    lift_direct_relativistic_trig_rationals,
)


PROFILE_SN_CONDITIONAL_FACTORED_POLYNOMIAL_ACCURACY_EXPORT = (
    "sn_gp_conditional_factored_polynomial_accuracy_export"
)
PROFILE_SN_COUPLED_TRIG_RATIONAL_ACCURACY_EXPORT = (
    "sn_gp_coupled_trig_rational_accuracy_export"
)
PROFILE_SN_RELATIVISTIC_TRIG_RATIONAL_ACCURACY_EXPORT = (
    "sn_gp_relativistic_trig_rational_accuracy_export"
)
PROFILE_SN_PHASE_MODULATED_RATIONAL_ACCURACY_EXPORT = (
    "sn_gp_phase_modulated_rational_accuracy_export"
)
PROFILE_SN_NESTED_AFFINE_RADICAL_ACCURACY_EXPORT = (
    "sn_gp_nested_affine_radical_accuracy_export"
)
PROFILE_SN_ADDITIVE_RADIAL_COUPLING_ACCURACY_EXPORT = (
    "sn_gp_additive_radial_coupling_accuracy_export"
)
PROFILE_SN_INVERSE_TRIG_MOBIUS_ACCURACY_EXPORT = (
    "sn_gp_inverse_trig_mobius_accuracy_export"
)


@dataclass(frozen=True)
class DeepAccuracyStage:
    """One frozen strategy step in the cumulative accuracy envelope."""

    code: str
    profile: str
    counter_prefix: str
    family_label: str
    predecessor_label: str
    predecessor_code: str | None
    lift_function: Callable[..., Any]
    required_dimension: int
    required_dimension_attribute: str
    proposal_id_offset: int
    result_counter_fields: tuple[str, ...]
    lift_argument_attributes: tuple[tuple[str, str], ...]
    fixed_lift_arguments: tuple[tuple[str, Any], ...]


DEEP_ACCURACY_STAGES = (
    DeepAccuracyStage(
        code="CF",
        profile=PROFILE_SN_CONDITIONAL_FACTORED_POLYNOMIAL_ACCURACY_EXPORT,
        counter_prefix="sn_conditional_factored",
        family_label="conditional_factored_polynomial",
        predecessor_label="ce_envelope",
        predecessor_code=None,
        lift_function=lift_direct_conditional_factored_polynomials,
        required_dimension=6,
        required_dimension_attribute="sn_conditional_factored_required_dimension",
        proposal_id_offset=2_410_000,
        result_counter_fields=(
            "monomial_candidates",
            "first_layer_candidates",
            "first_layer_value_pool",
            "residual_candidates_screened",
            "complete_candidates",
            "sobolev_candidates",
            "numeric_failures",
        ),
        lift_argument_attributes=(
            ("exponent_min", "sn_conditional_factored_exponent_min"),
            ("exponent_max", "sn_conditional_factored_exponent_max"),
            (
                "max_numerator_degree",
                "sn_conditional_factored_max_numerator_degree",
            ),
            (
                "max_denominator_degree",
                "sn_conditional_factored_max_denominator_degree",
            ),
            ("affine_scales", "sn_conditional_factored_affine_scales"),
            (
                "first_layer_value_pool_size",
                "sn_conditional_factored_first_layer_value_pool_size",
            ),
            (
                "residual_shortlist_size",
                "sn_conditional_factored_residual_per_context",
            ),
            (
                "complete_shortlist_size",
                "sn_conditional_factored_shortlist_size",
            ),
            ("proposal_limit", "sn_conditional_factored_proposal_limit"),
            (
                "protected_epsilon",
                "sn_conditional_factored_protected_epsilon",
            ),
        ),
        fixed_lift_arguments=(("affine_signs", (-1, 1)),),
    ),
    DeepAccuracyStage(
        code="CG",
        profile=PROFILE_SN_COUPLED_TRIG_RATIONAL_ACCURACY_EXPORT,
        counter_prefix="sn_coupled_trig",
        family_label="coupled_trig_rational",
        predecessor_label="cf_envelope",
        predecessor_code="CF",
        lift_function=lift_direct_coupled_trig_rationals,
        required_dimension=4,
        required_dimension_attribute="sn_coupled_trig_required_dimension",
        proposal_id_offset=2_420_000,
        result_counter_fields=(
            "candidates_screened",
            "value_pool",
            "sobolev_candidates",
            "denominator_rejections",
            "numeric_failures",
        ),
        lift_argument_attributes=(
            ("numerator_scales", "sn_coupled_trig_numerator_scales"),
            ("denominator_scales", "sn_coupled_trig_denominator_scales"),
            ("phase_scales", "sn_coupled_trig_phase_scales"),
            ("value_pool_size", "sn_coupled_trig_value_pool_size"),
            ("shortlist_size", "sn_coupled_trig_shortlist_size"),
            ("proposal_limit", "sn_coupled_trig_proposal_limit"),
            ("protected_epsilon", "sn_coupled_trig_protected_epsilon"),
        ),
        fixed_lift_arguments=(),
    ),
    DeepAccuracyStage(
        code="CH",
        profile=PROFILE_SN_RELATIVISTIC_TRIG_RATIONAL_ACCURACY_EXPORT,
        counter_prefix="sn_relativistic_trig",
        family_label="relativistic_trig_rational",
        predecessor_label="cg_envelope",
        predecessor_code="CG",
        lift_function=lift_direct_relativistic_trig_rationals,
        required_dimension=4,
        required_dimension_attribute="sn_relativistic_trig_required_dimension",
        proposal_id_offset=2_430_000,
        result_counter_fields=(
            "candidates_screened",
            "defined_candidates",
            "value_pool",
            "sobolev_candidates",
            "ratio_rejections",
            "amplitude_rejections",
            "radical_rejections",
            "denominator_rejections",
            "numeric_failures",
        ),
        lift_argument_attributes=(
            ("radical_scales", "sn_relativistic_trig_radical_scales"),
            (
                "denominator_scales",
                "sn_relativistic_trig_denominator_scales",
            ),
            ("phase_scales", "sn_relativistic_trig_phase_scales"),
            ("value_pool_size", "sn_relativistic_trig_value_pool_size"),
            ("shortlist_size", "sn_relativistic_trig_shortlist_size"),
            ("proposal_limit", "sn_relativistic_trig_proposal_limit"),
            (
                "protected_epsilon",
                "sn_relativistic_trig_protected_epsilon",
            ),
        ),
        fixed_lift_arguments=(),
    ),
    DeepAccuracyStage(
        code="CI",
        profile=PROFILE_SN_PHASE_MODULATED_RATIONAL_ACCURACY_EXPORT,
        counter_prefix="sn_phase_modulated",
        family_label="phase_modulated_rational",
        predecessor_label="ch_envelope",
        predecessor_code="CH",
        lift_function=lift_direct_phase_modulated_rationals,
        required_dimension=4,
        required_dimension_attribute="sn_phase_modulated_required_dimension",
        proposal_id_offset=2_440_000,
        result_counter_fields=(
            "monomial_candidates",
            "candidates_screened",
            "defined_candidates",
            "value_pool",
            "sobolev_candidates",
            "monomial_rejections",
            "amplitude_rejections",
            "denominator_rejections",
            "numeric_failures",
        ),
        lift_argument_attributes=(
            ("outer_scales", "sn_phase_modulated_outer_scales"),
            ("inner_scales", "sn_phase_modulated_inner_scales"),
            ("phase_scales", "sn_phase_modulated_phase_scales"),
            ("value_pool_size", "sn_phase_modulated_value_pool_size"),
            ("shortlist_size", "sn_phase_modulated_shortlist_size"),
            ("proposal_limit", "sn_phase_modulated_proposal_limit"),
            ("chunk_size", "sn_phase_modulated_chunk_size"),
            (
                "protected_epsilon",
                "sn_phase_modulated_protected_epsilon",
            ),
        ),
        fixed_lift_arguments=(),
    ),
    DeepAccuracyStage(
        code="CJ",
        profile=PROFILE_SN_NESTED_AFFINE_RADICAL_ACCURACY_EXPORT,
        counter_prefix="sn_nested_affine_radical",
        family_label="nested_affine_radical",
        predecessor_label="ci_envelope",
        predecessor_code="CI",
        lift_function=lift_direct_nested_affine_radicals,
        required_dimension=5,
        required_dimension_attribute="sn_nested_affine_radical_required_dimension",
        proposal_id_offset=2_450_000,
        result_counter_fields=(
            "candidates_screened",
            "defined_candidates",
            "value_pool",
            "sobolev_candidates",
            "denominator_rejections",
            "radical_rejections",
            "numeric_failures",
        ),
        lift_argument_attributes=(
            ("outer_scales", "sn_nested_affine_radical_outer_scales"),
            ("inner_scales", "sn_nested_affine_radical_inner_scales"),
            ("value_pool_size", "sn_nested_affine_radical_value_pool_size"),
            ("shortlist_size", "sn_nested_affine_radical_shortlist_size"),
            ("proposal_limit", "sn_nested_affine_radical_proposal_limit"),
            (
                "protected_epsilon",
                "sn_nested_affine_radical_protected_epsilon",
            ),
        ),
        fixed_lift_arguments=(),
    ),
    DeepAccuracyStage(
        code="CK",
        profile=PROFILE_SN_ADDITIVE_RADIAL_COUPLING_ACCURACY_EXPORT,
        counter_prefix="sn_additive_radial",
        family_label="additive_radial_coupling",
        predecessor_label="cj_envelope",
        predecessor_code="CJ",
        lift_function=lift_direct_additive_radial_couplings,
        required_dimension=6,
        required_dimension_attribute="sn_additive_radial_required_dimension",
        proposal_id_offset=2_460_000,
        result_counter_fields=(
            "candidates_screened",
            "defined_candidates",
            "value_pool",
            "sobolev_candidates",
            "radical_rejections",
            "numeric_failures",
        ),
        lift_argument_attributes=(
            ("value_pool_size", "sn_additive_radial_value_pool_size"),
            ("shortlist_size", "sn_additive_radial_shortlist_size"),
            ("proposal_limit", "sn_additive_radial_proposal_limit"),
            ("radicand_epsilon", "sn_additive_radial_radicand_epsilon"),
        ),
        fixed_lift_arguments=(),
    ),
    DeepAccuracyStage(
        code="CL",
        profile=PROFILE_SN_INVERSE_TRIG_MOBIUS_ACCURACY_EXPORT,
        counter_prefix="sn_inverse_trig_mobius",
        family_label="inverse_trig_mobius",
        predecessor_label="ck_envelope",
        predecessor_code="CK",
        lift_function=lift_direct_inverse_trig_mobius,
        required_dimension=3,
        required_dimension_attribute="sn_inverse_trig_mobius_required_dimension",
        proposal_id_offset=2_470_000,
        result_counter_fields=(
            "candidates_screened",
            "defined_candidates",
            "value_pool",
            "sobolev_candidates",
            "ratio_rejections",
            "denominator_rejections",
            "domain_rejections",
            "numeric_failures",
        ),
        lift_argument_attributes=(
            ("phase_scales", "sn_inverse_trig_mobius_phase_scales"),
            ("value_pool_size", "sn_inverse_trig_mobius_value_pool_size"),
            ("shortlist_size", "sn_inverse_trig_mobius_shortlist_size"),
            ("proposal_limit", "sn_inverse_trig_mobius_proposal_limit"),
            (
                "protected_epsilon",
                "sn_inverse_trig_mobius_protected_epsilon",
            ),
            (
                "domain_tolerance",
                "sn_inverse_trig_mobius_domain_tolerance",
            ),
        ),
        fixed_lift_arguments=(),
    ),
)

DEEP_ACCURACY_PROFILE_NAMES = tuple(stage.profile for stage in DEEP_ACCURACY_STAGES)
DEEP_ACCURACY_STAGE_BY_PROFILE = {
    stage.profile: stage for stage in DEEP_ACCURACY_STAGES
}
DEEP_ACCURACY_STAGE_BY_CODE = {stage.code: stage for stage in DEEP_ACCURACY_STAGES}
DEEP_ACCURACY_PARAMETER_NAMES = tuple(
    dict.fromkeys(
        name
        for stage in DEEP_ACCURACY_STAGES
        for name in (
            stage.required_dimension_attribute,
            *(attribute for _, attribute in stage.lift_argument_attributes),
        )
    )
)
DEEP_ACCURACY_PARAMETER_DEFAULTS: dict[str, Any] = {
    "sn_conditional_factored_required_dimension": 6,
    "sn_conditional_factored_exponent_min": -2,
    "sn_conditional_factored_exponent_max": 4,
    "sn_conditional_factored_max_numerator_degree": 5,
    "sn_conditional_factored_max_denominator_degree": 3,
    "sn_conditional_factored_affine_scales": (0.5, 1.0, 2.0),
    "sn_conditional_factored_first_layer_value_pool_size": 64,
    "sn_conditional_factored_residual_per_context": 8,
    "sn_conditional_factored_shortlist_size": 16,
    "sn_conditional_factored_proposal_limit": 32,
    "sn_conditional_factored_protected_epsilon": 1e-6,
    "sn_coupled_trig_required_dimension": 4,
    "sn_coupled_trig_numerator_scales": (0.5, 1.0, 2.0),
    "sn_coupled_trig_denominator_scales": (0.5, 1.0, 2.0),
    "sn_coupled_trig_phase_scales": (
        0.5,
        1.0,
        2.0,
        3.141592653589793,
        6.283185307179586,
    ),
    "sn_coupled_trig_value_pool_size": 64,
    "sn_coupled_trig_shortlist_size": 16,
    "sn_coupled_trig_proposal_limit": 32,
    "sn_coupled_trig_protected_epsilon": 1e-6,
    "sn_relativistic_trig_required_dimension": 4,
    "sn_relativistic_trig_radical_scales": (0.5, 1.0, 2.0),
    "sn_relativistic_trig_denominator_scales": (0.5, 1.0, 2.0),
    "sn_relativistic_trig_phase_scales": (
        0.5,
        1.0,
        2.0,
        3.141592653589793,
        6.283185307179586,
    ),
    "sn_relativistic_trig_value_pool_size": 64,
    "sn_relativistic_trig_shortlist_size": 16,
    "sn_relativistic_trig_proposal_limit": 32,
    "sn_relativistic_trig_protected_epsilon": 1e-6,
    "sn_phase_modulated_required_dimension": 4,
    "sn_phase_modulated_outer_scales": (0.5, 1.0, 2.0),
    "sn_phase_modulated_inner_scales": (0.5, 1.0, 2.0),
    "sn_phase_modulated_phase_scales": (
        0.5,
        1.0,
        2.0,
        3.141592653589793,
        6.283185307179586,
    ),
    "sn_phase_modulated_value_pool_size": 64,
    "sn_phase_modulated_shortlist_size": 16,
    "sn_phase_modulated_proposal_limit": 32,
    "sn_phase_modulated_chunk_size": 8,
    "sn_phase_modulated_protected_epsilon": 1e-6,
    "sn_nested_affine_radical_required_dimension": 5,
    "sn_nested_affine_radical_outer_scales": (0.5, 1.0, 2.0),
    "sn_nested_affine_radical_inner_scales": (0.5, 1.0, 2.0),
    "sn_nested_affine_radical_value_pool_size": 64,
    "sn_nested_affine_radical_shortlist_size": 16,
    "sn_nested_affine_radical_proposal_limit": 32,
    "sn_nested_affine_radical_protected_epsilon": 1e-6,
    "sn_additive_radial_required_dimension": 6,
    "sn_additive_radial_value_pool_size": 64,
    "sn_additive_radial_shortlist_size": 16,
    "sn_additive_radial_proposal_limit": 32,
    "sn_additive_radial_radicand_epsilon": 1e-12,
    "sn_inverse_trig_mobius_required_dimension": 3,
    "sn_inverse_trig_mobius_phase_scales": (
        0.5,
        1.0,
        2.0,
        3.141592653589793,
        6.283185307179586,
    ),
    "sn_inverse_trig_mobius_value_pool_size": 64,
    "sn_inverse_trig_mobius_shortlist_size": 16,
    "sn_inverse_trig_mobius_proposal_limit": 32,
    "sn_inverse_trig_mobius_protected_epsilon": 1e-6,
    "sn_inverse_trig_mobius_domain_tolerance": 1e-12,
}


def proposal_trace_document(proposal: Any) -> dict[str, Any]:
    """Return the scalar/tuple proposal metadata used in compact export traces."""

    if not is_dataclass(proposal):
        raise TypeError("deep accuracy proposal must be a dataclass instance")
    return asdict(proposal)


def stage_lift_arguments(
    stage: DeepAccuracyStage,
    parameter_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Map frozen profile attributes to one stage constructor's arguments."""

    arguments = dict(stage.fixed_lift_arguments)
    arguments.update(
        {
            argument: parameter_values[attribute]
            for argument, attribute in stage.lift_argument_attributes
        }
    )
    return arguments
