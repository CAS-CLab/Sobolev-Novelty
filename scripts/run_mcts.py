"""Frozen Base / SN-MCTS, strict completed-iteration floor export."""

import argparse
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
        "--smoke", action="store_true", help="2 iterations; not a 900 s paper run"
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.child:
        from pathlib import Path

        if Path(a.output).exists():
            raise FileExistsError(a.output)
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--child",
            ],
            check=True,
            timeout=180 if a.smoke else 1800,
        )
        return
    import numpy as np
    from src.nd2py import nd2py as nd
    from src.nd2py.nd2py.utils import seed_all, init_logger
    from src.heuristic.mcts.mcts import MCTS, simplify

    out = new_output(a.output)
    init_logger("src", a.task, str(out / "search.log"))
    d = srbench_rows(a.scope, a.task, a.seed, method="mcts")
    c = load(ROOT / "configs/mcts.json")
    b = c["search_budget"]
    profile = next(
        p
        for p in c["profiles"]
        if p["name"] == ("base" if a.method == "base" else "sn_elite_incremental_prune")
    )
    allowed = set(__import__("inspect").signature(MCTS).parameters)
    kwargs = {k: v for k, v in profile.items() if k in allowed}
    if kwargs.get("geometry_sample_size") == "full":
        kwargs["geometry_sample_size"] = None
    kwargs.update({k: v for k, v in b.items() if k in allowed})
    kwargs.update(
        sample_num=200,
        max_var=10,
        random_state=a.seed,
        ratio=1.0,
        normalize_X=False,
        normalize_y=False,
        remove_abnormal=False,
        keep_vars=False,
        n_iter=2 if a.smoke else 10000,
        child_num=3 if a.smoke else 50,
        n_playout=2 if a.smoke else 100,
        d_playout=2 if a.smoke else 10,
        time_limit=120 if a.smoke else 900,
        time_floor_snapshot_seconds=120 if a.smoke else 900,
        save_path=str(out / "records.json"),
        log_per_sec=10,
        eic_random_state=a.seed,
        sobolev_detailed_logging=False,
        sobolev_dataset_identity=a.task,
        sobolev_run_summary_path=str(out / "mechanism_summary.json"),
        binary=[nd.Mul, nd.Div, nd.Add, nd.Sub],
        unary=[
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
        ],
        leaf=[nd.Number(1), nd.Number(2), nd.Number(np.pi)],
    )
    model = MCTS(**kwargs)
    seed_all(a.seed)
    status = model.fit(
        dict(zip(d["names"], d["X"].T)),
        d["y"],
        use_tqdm=False,
        early_stop=lambda *_: False,
    )
    floor = model.time_floor_snapshot
    if not floor:
        raise RuntimeError("No complete iteration at or before the floor")
    if floor["wall_time"] > kwargs["time_floor_snapshot_seconds"]:
        raise RuntimeError("Late floor")
    expression = str(simplify(nd.parse(floor["best_expression"])))
    write(
        out / "result.json",
        dict(
            method=a.method,
            profile=profile["name"],
            task=a.task,
            seed=a.seed,
            smoke=a.smoke,
            status=status,
            floor=floor,
            exported_expression=expression,
            search_row_ids=d["rows"],
            evaluation_row_ids=d["evaluation_rows"],
            evaluation_scope="held_out",
            metrics=metrics(expression, d["Xe"], d["ye"], d["names"]),
        ),
    )


if __name__ == "__main__":
    main()
