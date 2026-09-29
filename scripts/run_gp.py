"""Final Population GP / SN-GP, with canonical Base-reward export."""

import argparse
import copy
import subprocess
import sys
from common import ROOT, load, write, new_output, srbench_rows, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["base", "sn"], required=True)
    p.add_argument("--scope", choices=["whitebox", "blackbox"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="20 individuals, 1 generation; not paper results",
    )
    a = p.parse_args()
    import numpy as np
    from src.heuristic.gp.gp import build_initial_population_document
    from src.heuristic.gp.canonical_export import candidate_pool, candidate_key

    out = new_output(a.output)
    data = srbench_rows(a.scope, a.task, a.seed, method="gp")
    cfg = copy.deepcopy(load(ROOT / "configs/gp.json"))
    if a.smoke:
        cfg["algorithm"].update(population_size=20, elitism_k=2, tournament_size=5)
        cfg["fixed_generation"].update(
            target_generation=1, time_limit_seconds=120, hard_time_limit_seconds=180
        )
    config = out / "config.json"
    write(config, cfg)
    np.savez_compressed(
        out / "search.npz",
        X=data["X"],
        y=data["y"],
        row_ids=data["rows"],
        feature_names=data["names"],
        task_index=data["task_index"],
        dataset=a.task,
        seed=a.seed,
    )
    initial = build_initial_population_document(
        data["names"],
        seed=a.seed,
        population_size=cfg["algorithm"]["population_size"],
    )
    initial.update(task_index=data["task_index"], dataset=a.task)
    write(out / "population.json", initial)
    profile = (
        "base_gp_export_aligned"
        if a.method == "base"
        else "sn_gp_inverse_trig_mobius_accuracy_export"
    )
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/gp_child.py"),
        "--config",
        str(config),
        "--input",
        str(out / "search.npz"),
        "--initial-population",
        str(out / "population.json"),
        "--task-index",
        str(data["task_index"]),
        "--dataset",
        a.task,
        "--seed",
        str(a.seed),
        "--profile",
        profile,
        "--target-generation",
        str(cfg["fixed_generation"]["target_generation"]),
        "--output",
        str(out / "search"),
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=ROOT,
        timeout=cfg["fixed_generation"]["hard_time_limit_seconds"] + 30,
    )
    result = load(out / "search/result.json")
    if (
        result["status"] != "iter_limit"
        or result["completed_generations"]
        < cfg["fixed_generation"]["target_generation"] + 1
    ):
        raise RuntimeError("Incomplete generation budget; not a valid formal result")
    pool = candidate_pool(result)
    if not pool:
        raise RuntimeError("No valid canonical export candidates")
    chosen = min(pool, key=candidate_key)
    expression = chosen["exported_expression"]
    write(
        out / "result.json",
        dict(
            method=a.method,
            profile=("sn_gp_base_reward_export" if a.method == "sn" else profile),
            task=a.task,
            seed=a.seed,
            smoke=a.smoke,
            exported_expression=expression,
            selected_candidate=chosen,
            search_row_ids=data["rows"],
            evaluation_row_ids=data["evaluation_rows"],
            evaluation_scope="held_out",
            metrics=metrics(
                expression, data["Xe"], data["ye"], data["names"], protected=True
            ),
        ),
    )


if __name__ == "__main__":
    main()
