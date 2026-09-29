"""Portable paths, immutable run directories and seeded SRBench splits."""

from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def srbench_rows(scope, task, seed, *, method):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    tasks = (ROOT / f"data/srbench/{scope}.txt").read_text().split()
    index = tasks.index(task) + 1
    payload = ROOT / f"data/pmlb/datasets/{task}/{task}.tsv.gz"
    frame = pd.read_csv(payload, sep="\t")
    feature_names = [str(c) for c in frame.columns if c != "target"]
    eligible = frame.dropna()
    if len(eligible) < 4:
        raise ValueError("Evaluation requires at least four complete rows")
    # Generate the partition before search subsampling; no saved row manifest
    # is needed. Base/SN and GP/MCTS share the same task/seed partition.
    search_pool, test = train_test_split(
        eligible,
        test_size=max(2, int(np.ceil(0.25 * len(eligible)))),
        random_state=seed,
    )
    train_rows = (
        search_pool.sample(n=200, random_state=seed)
        if len(search_pool) > 200
        else search_pool
    ).index.tolist()
    evaluation_rows = test.index.tolist()
    if scope == "blackbox":
        # GP's final blackbox protocol explicitly truncates features. MCTS's
        # max_var limits its state width, not the input/derivative dimension.
        if method == "gp":
            feature_names = feature_names[:10]
    assert not set(train_rows) & set(evaluation_rows)
    names = (
        tuple(feature_names)
        if scope == "whitebox" and method == "mcts"
        else tuple(f"x{k+1}" for k in range(len(feature_names)))
    )

    def arrays(rows):
        selected = frame.loc[rows]
        return (
            np.ascontiguousarray(selected[feature_names], dtype=float),
            np.ascontiguousarray(selected.target, dtype=float),
        )

    X, y = arrays(train_rows)
    Xe, ye = arrays(evaluation_rows)
    if not all(np.isfinite(a).all() for a in (X, y, Xe, ye)):
        raise ValueError("Nonfinite data")
    return dict(
        X=X,
        y=y,
        Xe=Xe,
        ye=ye,
        names=names,
        rows=train_rows,
        evaluation_rows=evaluation_rows,
        task_index=index,
    )


def metrics(expression, X, y, names, *, protected=False):
    import numpy as np
    from src.nd2py import nd2py as nd

    tree = nd.parse(expression)
    inputs = dict(zip(names, X.T))
    with np.errstate(all="ignore"):
        prediction = np.asarray(
            tree.eval(inputs, **({"use_eps": 1e-6} if protected else {})), dtype=float
        )
    if prediction.ndim == 0:
        prediction = np.full(len(y), float(prediction))
    prediction = prediction.reshape(-1)
    if len(prediction) != len(y):
        raise ValueError("Prediction shape mismatch")
    bad = ~np.isfinite(prediction)
    prediction[bad] = 0
    nmse = float(np.mean((prediction - y) ** 2) / np.var(y))
    if not np.isfinite(nmse):
        raise ValueError("Nonfinite metric")
    return dict(
        r2=1 - nmse,
        nmse=nmse,
        complexity=len(tree),
        nonfinite_predictions_zero_filled=int(bad.sum()),
        refit=False,
    )
