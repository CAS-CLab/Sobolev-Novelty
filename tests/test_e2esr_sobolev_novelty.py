from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sympytorch", reason="Install the E2ESR environment to run these tests")

from src.generative.e2esr.parsers import get_parser
from src.generative.e2esr.symbolicregression.envs.environment import SPECIAL_WORDS
from src.generative.e2esr.symbolicregression.envs.generators import Node, RandomFunctions
from src.generative.e2esr.symbolicregression.envs.sobolev_novelty import (
    FastSobolevNovelty,
    SobolevFilterConfig,
    evaluate_reference,
)


PARAMS = SimpleNamespace(use_abs=False)


def node(value: object, *children: Node) -> Node:
    return Node(str(value), PARAMS, list(children))


def x(index: int = 0) -> Node:
    return node(f"x_{index}")


def test_single_term_is_accepted_with_novelty_one() -> None:
    result = FastSobolevNovelty().evaluate(
        node("sin", x()), np.linspace(-1.0, 1.0, 128)[:, None]
    )
    assert result.success and result.accepted
    assert result.term_count == 1
    assert result.novelties == [1.0]


def test_proportional_terms_are_rejected() -> None:
    result = FastSobolevNovelty().evaluate(
        node("add", x(), node("mul", node(2), x())),
        np.linspace(-2.0, 2.0, 128)[:, None],
    )
    assert result.success and not result.accepted
    assert max(result.novelties) < 1e-10


def test_reference_and_fast_paths_agree() -> None:
    expression = node("add", node("sin", x()), node("pow2", x()))
    points = np.random.default_rng(7).normal(size=(160, 1))
    config = SobolevFilterConfig(geometry_rows=128)
    fast = FastSobolevNovelty(config).evaluate(expression, points)
    reference = evaluate_reference(expression, points, config)
    assert fast.success and reference.success
    np.testing.assert_allclose(
        sorted(fast.novelties), sorted(reference.term_novelties), atol=2e-8, rtol=2e-7
    )


def test_geometry_seed_is_deterministic_and_scales_are_standardized() -> None:
    expression = node("add", x(), node("pow2", x()))
    evaluator = FastSobolevNovelty(SobolevFilterConfig(geometry_rows=64))
    base_points = np.linspace(-2.0, 2.0, 128)[:, None]
    geometry_indices = np.arange(64)
    first = evaluator.evaluate(expression, base_points, geometry_indices=geometry_indices)
    second = evaluator.evaluate(expression, base_points, geometry_indices=geometry_indices)
    scaled_expression = node(
        "add",
        node("div", x(), node(100)),
        node("pow2", node("div", x(), node(100))),
    )
    scaled = evaluator.evaluate(
        scaled_expression, 100.0 * base_points, geometry_indices=geometry_indices
    )
    assert first.success and second.success and scaled.success
    assert first.geometry_indices == second.geometry_indices
    np.testing.assert_array_equal(first.novelties, second.novelties)
    np.testing.assert_allclose(
        sorted(first.novelties), sorted(scaled.novelties), atol=1e-10, rtol=1e-10
    )


def test_invalid_geometry_is_rejected_explicitly() -> None:
    result = FastSobolevNovelty().evaluate(
        node("sqrt", x()), -np.linspace(0.1, 2.0, 64)[:, None]
    )
    assert not result.success
    assert result.failure_type in {"domain_failure", "insufficient_valid_samples"}


def test_parser_selects_sobolev_filter() -> None:
    params = get_parser().parse_args([])
    assert params.pretrain_filter == "sobolev_novelty"
    assert math.isclose(params.sn_threshold, 1.0 / math.sqrt(10.0))
    assert params.sn_geometry_rows == 200


def test_random_generator_remains_the_e2esr_proposal_source() -> None:
    params = get_parser().parse_args([])
    params.max_input_dimension = 2
    params.max_binary_ops_per_dim = 1
    params.max_binary_ops_offset = 2
    params.max_unary_ops = 2
    generator = RandomFunctions(params, SPECIAL_WORDS)
    tree, dimension, *_ = generator.generate_multi_dimensional_tree(
        np.random.RandomState(20260806)
    )
    assert tree is not None
    assert dimension >= 1


def test_e2esr_and_search_evaluators_are_isolated() -> None:
    from src.generative.e2esr.symbolicregression.envs import sobolev_novelty as plugin
    from src.e2esr_frozen.sobolev.evaluator import SobolevEvaluator as FrozenEvaluator
    from src.sobolev.evaluator import SobolevEvaluator as SearchEvaluator
    from src.sobolev import prune_and_refit, ShortlistConfig

    assert plugin.SobolevEvaluator is FrozenEvaluator
    assert FrozenEvaluator is not SearchEvaluator
    assert plugin.compute_novelties.__module__.startswith("src.e2esr_frozen.")
    assert plugin.stable_norm.__module__.startswith("src.e2esr_frozen.")
    assert callable(prune_and_refit)
    assert ShortlistConfig is not None


def test_launcher_defaults_and_argument_passthrough(tmp_path) -> None:
    from pathlib import Path
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_e2esr.sh"), "--cpu", "true", "--max_epoch", "1"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": "/bin/echo"},
        check=True,
        text=True,
        capture_output=True,
    )
    assert "-m src.generative.e2esr.train" in result.stdout
    assert "--pretrain_filter sobolev_novelty" in result.stdout
    assert "--sn_geometry_seed 20260806" in result.stdout
    assert result.stdout.rstrip().endswith("--cpu true --max_epoch 1")
