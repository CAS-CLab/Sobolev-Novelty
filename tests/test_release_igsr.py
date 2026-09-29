"""Offline integration: mocked provider, real fitting/search/tape and SN adapter."""

import sys
from pathlib import Path
import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("litellm")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_root_replay_and_logical_token_accounting(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import litellm
    from run_igsr import config, independent_evaluate
    from igsr.sobolev_novelty.prompt_feedback import install_prompt_feedback_overlay

    install_prompt_feedback_overlay()
    from igsr.method.igsr import igsr

    # A local synthetic table exercises the real SRBench loader without
    # redistributing benchmark data or contacting a model service.
    data_root = tmp_path / "pmlb"
    table_dir = data_root / "datasets/1027_ESL"
    table_dir.mkdir(parents=True)
    X = np.random.default_rng(123).normal(size=(160, 3))
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["target"] = 1 + X @ np.array([2.0, -0.5, 0.25])
    frame.to_csv(table_dir / "1027_ESL.tsv.gz", sep="\t", index=False)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return litellm.ModelResponse(
            id="offline-fixture",
            model=kwargs["model"],
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "TERMS:\n```\nx1\nx2\nx3\n```",
                    },
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setenv("IGSR_LLM_CACHE_MODE", "read_write")
    monkeypatch.setenv("IGSR_LLM_CACHE_DIR", str(tmp_path / "tape"))
    results = []
    for method in ["base", "sn"]:
        out = tmp_path / method
        out.mkdir()
        cfg = config(
            "blackbox", "1027_ESL", 20260925, method, "http://offline.invalid/v1", out
        )
        cfg.dataset.srbench_blackbox_root = str(data_root)
        cfg.experiment.total_budget = 1
        cfg.experiment.n_successors = 1
        cfg.experiment.break_above_N_tokens = 100
        cfg.experiment.print_prompts = False
        result = igsr(cfg, 20260925)
        results.append(result)
        external = independent_evaluate(cfg, result)
        assert external["status"] == "ok"
        assert len(result["history_test"]) == 1
        assert external["terms"] == result["history_test"][0]["metadata"]["terms_after"]
    assert len(calls) == 1, "SN root must replay Base, not call the provider"
    assert results[0]["best_formula"] == results[1]["best_formula"]
    for result in results:
        assert result["compute_profiler"]["input_tokens"] == 100
        assert result["compute_profiler"]["output_tokens"] == 20


def test_frozen_engine_loads_from_release():
    from igsr.sobolev_novelty.adapter import load_eic_components

    components = load_eic_components(ROOT / "src/igsr_frozen")
    assert (
        Path(components.source_root).resolve() == (ROOT / "src/igsr_frozen").resolve()
    )


def test_all_loaded_igsr_modules_are_local():
    import igsr.method.igsr

    for name, module in list(sys.modules.items()):
        if name.startswith("igsr.") and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(ROOT), (
                name,
                module.__file__,
            )


@pytest.mark.parametrize("seed", [42, 1234, 12345, 666, 777, 1337, 8675309, 9001, 20260925])
def test_srbench_pair_config_uses_seed_without_manifest(seed, tmp_path):
    from run_igsr import config

    base = config("blackbox", "1027_ESL", seed, "base", "http://offline.invalid/v1", tmp_path)
    sn = config("blackbox", "1027_ESL", seed, "sn", "http://offline.invalid/v1", tmp_path)
    assert base.dataset == sn.dataset
    assert base.dataset.seed == seed
    for key in ("payload", "split_identity", "search_split_identity"):
        assert base.dataset.get(f"srbench_blackbox_expected_{key}_sha256") is None


def test_unsupported_scope_rejected(tmp_path):
    from run_igsr import config

    with pytest.raises(ValueError, match="only SRBench blackbox"):
        config("unsupported", "1027_ESL", 42, "base", "http://offline.invalid/v1", tmp_path)
