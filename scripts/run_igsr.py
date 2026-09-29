"""Paired IGSR / SN-IGSR with local-model endpoints and seeded splits."""

import argparse
import os
import subprocess
import sys
from common import ROOT, load, write, new_output


def config(scope, task, seed, method, endpoint, out):
    from omegaconf import OmegaConf

    if scope != "blackbox":
        raise ValueError("IGSR supports only SRBench blackbox in this release")
    dataset = dict(
        task="regression",
        kind="real",
        name="srbench_blackbox_local_controlled",
        env_name="srbench_blackbox_local_controlled",
        seed=seed,
        srbench_blackbox_root=str(ROOT / "data/pmlb"),
        srbench_blackbox_task_list=str(ROOT / "data/srbench/blackbox.txt"),
        srbench_blackbox_dataset_id=task,
    )
    experiment = load(ROOT / f"configs/igsr_{method}.json")
    experiment["sobolev_eic_src_root"] = str(ROOT / "src/igsr_frozen")
    llm = dict(
        kind="openai",
        model_type="llama-3.1",
        model_version="local-8b-bf16",
        deployment="llama-3.1-8b-instruct",
        model_revision="700154e2ef0a98b5fe191b95df48cec1db0aad57a77d8ae42df4bc2c2720b086",
        endpoint=endpoint,
        api_version=None,
        api_key=os.environ.get("IGSR_API_KEY", "igsr-local-vllm"),
        temperature=1.0,
        request_timeout=600,
        max_tokens=2048,
        extra_body=dict(stop_token_ids=[128009], include_stop_str_in_output=False),
        deterministic_request_seeds=True,
    )
    return OmegaConf.create(
        dict(
            experiment=experiment,
            dataset=dataset,
            llm=llm,
            logging=dict(
                logger_name="igsr.review", mode="w", log_file=str(out / "search.jsonl")
            ),
        )
    )


def independent_evaluate(cfg, result):
    """Fresh fit of one validation-selected expression, after the child exits."""
    import numpy as np
    import structlog
    from sklearn.linear_model import Ridge
    from igsr.dataset.util import get_dataset
    from igsr.method.igsr_utils import DesignMatrix, prepare_dataset
    from igsr.util import compute_regression_metrics

    records = result["history_test"]
    if len(records) != 1:
        raise ValueError("Expected exactly one validation-selected model")
    terms = records[0]["metadata"]["terms_after"]
    prepared = prepare_dataset(
        get_dataset(cfg.dataset), structlog.get_logger("review.evaluate")
    )
    try:
        if not terms:
            raise ValueError("Empty selected expression")

        def matrix(data):
            m = DesignMatrix.from_terms(terms, dict(data))
            if m.errors or m.term_names != terms or not np.isfinite(m.phi).all():
                raise ValueError("Invalid complete expression")
            return m.phi

        y = np.column_stack(list(prepared.y.values()))
        yt = np.column_stack(list(prepared.y_test.values()))
        model = Ridge(alpha=1e-8, fit_intercept=True, solver="svd").fit(
            matrix(prepared.data), y
        )
        pred = model.predict(matrix(prepared.data_test))
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite prediction")
        metric = compute_regression_metrics(yt, pred)
        metric = {
            k: float(metric[k])
            for k in ["r2", "nmse", "accuracy_tol", "accuracy_tol_max"]
        }
        if not all(np.isfinite(v) for v in metric.values()):
            raise ValueError("Nonfinite metrics")
        status = "ok"
    except Exception as error:
        metric = dict(r2=1 - 1e12, nmse=1e12, accuracy_tol=0.0, accuracy_tol_max=0.0)
        status = f"failed_closed: {type(error).__name__}: {error}"
    return dict(
        metrics=metric,
        complexity=len(terms),
        terms=terms,
        status=status,
        scope="independent_test_after_child_exit",
        selection="validation_only",
    )


def json_safe(x):
    import math
    import numpy as np

    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main():
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["base", "sn", "pair"], default="pair")
    p.add_argument("--scope", choices=["blackbox"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--endpoint", default="http://127.0.0.1:8010/v1")
    p.add_argument("--output", required=True)
    p.add_argument("--tape", help="Existing paired tape; required for standalone SN")
    p.add_argument(
        "--check-data",
        action="store_true",
        help="Load data and check the seeded split without LLM requests",
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.check_data:
        from igsr.dataset.util import get_dataset

        cfg = config(a.scope, a.task, a.seed, "sn", a.endpoint, Path(a.output))
        b = get_dataset(cfg.dataset)
        print(
            dict(
                train=len(b.dataset_train),
                validation=len(b.dataset_validation),
                test=len(b.dataset_test),
                equation=b.equation,
                split_identity=b.data_settings["split"]["split_identity_sha256"],
            )
        )
        return
    if a.child:
        from igsr.sobolev_novelty.prompt_feedback import install_prompt_feedback_overlay

        install_prompt_feedback_overlay()
        from igsr.method.igsr import igsr

        out = Path(a.output)
        cfg = config(a.scope, a.task, a.seed, a.method, a.endpoint, out)
        write(out / "search_result.json", json_safe(igsr(cfg, a.seed)))
        return
    if a.method == "sn" and not a.tape:
        raise ValueError("Use --method pair or provide Base --tape")
    out = new_output(a.output)
    tape = Path(a.tape).resolve() if a.tape else out / "completion_tape"
    methods = ["base", "sn"] if a.method == "pair" else [a.method]
    env = dict(
        os.environ,
        IGSR_LLM_CACHE_DIR=str(tape),
        IGSR_LLM_CACHE_MODE="read_write",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for method in methods:
        child = out / method
        child.mkdir()
        cmd = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--child",
            "--method",
            method,
            "--scope",
            a.scope,
            "--task",
            a.task,
            "--seed",
            str(a.seed),
            "--endpoint",
            a.endpoint,
            "--output",
            str(child),
        ]
        subprocess.run(cmd, check=True, env=env, cwd=ROOT)
        result = load(child / "search_result.json")
        cfg = config(a.scope, a.task, a.seed, method, a.endpoint, child)
        report = independent_evaluate(cfg, result)
        report.update(
            task=a.task,
            seed=a.seed,
            method=method,
            compute_profiler=result["compute_profiler"],
            sobolev_run_status=result["sobolev_run_status"],
        )
        write(child / "result.json", report)


if __name__ == "__main__":
    main()
