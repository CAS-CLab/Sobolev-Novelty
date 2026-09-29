"""Portable row materialization and selection-only-on-training invariants."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import common
from common import srbench_rows, new_output
from src.heuristic.gp.canonical_export import candidate_pool, candidate_key


def require_table(task):
    if not (ROOT / f"data/pmlb/datasets/{task}/{task}.tsv.gz").is_file():
        pytest.skip("External SRBench dataset not bundled; see README.md")


def test_paired_blackbox_search_and_evaluation_rows_are_identical():
    require_table("1027_ESL")
    base = srbench_rows("blackbox", "1027_ESL", 20260808, method="mcts")
    sn = srbench_rows("blackbox", "1027_ESL", 20260808, method="gp")
    assert not set(base["rows"]) & set(base["evaluation_rows"])
    for key in ("X", "y", "Xe", "ye"):
        np.testing.assert_array_equal(base[key], sn[key])
    assert len(base["rows"]) == 200


@pytest.fixture
def whitebox_payload(tmp_path, monkeypatch):
    task = "synthetic_whitebox"
    tasks = tmp_path / "data/srbench/whitebox.txt"
    tasks.parent.mkdir(parents=True)
    tasks.write_text(task + "\n")
    payload = tmp_path / f"data/pmlb/datasets/{task}/{task}.tsv.gz"
    payload.parent.mkdir(parents=True)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    return task, payload


@pytest.mark.parametrize(
    "complete_rows,search_count,test_count",
    [(4, 2, 2), (7, 5, 2), (200, 150, 50), (401, 200, 101), (1000, 200, 250)],
)
def test_whitebox_has_disjoint_paired_test_rows(
    whitebox_payload, complete_rows, search_count, test_count
):
    task, payload = whitebox_payload
    x = np.arange(complete_rows + 2, dtype=float)
    frame = pd.DataFrame({"x1": x, "target": x**2})
    frame.loc[1, "x1"] = np.nan
    frame.loc[complete_rows, "target"] = np.nan
    frame.to_csv(payload, sep="\t", index=False)
    seed = 20260808
    gp = srbench_rows("whitebox", task, seed, method="gp")
    mcts = srbench_rows("whitebox", task, seed, method="mcts")
    replay = srbench_rows("whitebox", task, seed, method="gp")

    search_ids = set(gp["rows"])
    test_ids = set(gp["evaluation_rows"])
    eligible_ids = set(frame.dropna().index)
    assert len(gp["rows"]) == len(search_ids) == search_count
    assert len(gp["evaluation_rows"]) == len(test_ids) == test_count
    assert search_ids.isdisjoint(test_ids)
    assert search_ids | test_ids <= eligible_ids
    for paired in (mcts, replay):
        assert gp["rows"] == paired["rows"]
        assert gp["evaluation_rows"] == paired["evaluation_rows"]
        for key in ("X", "y", "Xe", "ye"):
            np.testing.assert_array_equal(gp[key], paired[key])
    np.testing.assert_array_equal(gp["Xe"][:, 0], frame.loc[gp["evaluation_rows"], "x1"])
    np.testing.assert_array_equal(gp["ye"], frame.loc[gp["evaluation_rows"], "target"])


def test_whitebox_test_values_do_not_affect_search_data(whitebox_payload):
    task, payload = whitebox_payload
    x = np.arange(400, dtype=float)
    frame = pd.DataFrame({"x1": x, "target": x**2})
    frame.to_csv(payload, sep="\t", index=False)
    original = srbench_rows("whitebox", task, 20260808, method="gp")
    other_seed = srbench_rows("whitebox", task, 20260809, method="gp")
    assert set(original["evaluation_rows"]) != set(other_seed["evaluation_rows"])

    frame.loc[original["evaluation_rows"], ["x1", "target"]] += 1e6
    frame.to_csv(payload, sep="\t", index=False)
    changed = srbench_rows("whitebox", task, 20260808, method="gp")
    assert original["rows"] == changed["rows"]
    assert original["evaluation_rows"] == changed["evaluation_rows"]
    for key in ("X", "y"):
        np.testing.assert_array_equal(original[key], changed[key])
    for key in ("Xe", "ye"):
        np.testing.assert_allclose(changed[key], original[key] + 1e6)


def test_whitebox_requires_enough_rows_for_train_and_test(whitebox_payload):
    task, payload = whitebox_payload
    pd.DataFrame({"x1": [0.0, 1.0, 2.0], "target": [0.0, 1.0, 4.0]}).to_csv(
        payload, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="at least four complete rows"):
        srbench_rows("whitebox", task, 20260808, method="gp")


def test_mcts_max_var_is_not_input_truncation():
    require_table("505_tecator")
    m = srbench_rows("blackbox", "505_tecator", 20260808, method="mcts")
    g = srbench_rows("blackbox", "505_tecator", 20260808, method="gp")
    assert m["X"].shape[1] > 10
    assert g["X"].shape[1] == 10
    np.testing.assert_array_equal(m["X"][:, :10], g["X"])
    assert m["rows"] == g["rows"]


def test_blackbox_generates_paired_split_for_new_seed(tmp_path, monkeypatch):
    task = "synthetic_blackbox"
    task_list = tmp_path / "data/srbench/blackbox.txt"
    task_list.parent.mkdir(parents=True)
    task_list.write_text(task + "\n")
    payload = tmp_path / f"data/pmlb/datasets/{task}/{task}.tsv.gz"
    payload.parent.mkdir(parents=True)
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(500, 12)), columns=[f"f{i}" for i in range(12)])
    frame["target"] = frame.f0 + frame.f1**2
    frame.to_csv(payload, sep="\t", index=False)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    mcts = srbench_rows("blackbox", task, 20260925, method="mcts")
    gp = srbench_rows("blackbox", task, 20260925, method="gp")
    repeat = srbench_rows("blackbox", task, 20260925, method="gp")
    assert len(gp["rows"]) == 200
    assert len(gp["evaluation_rows"]) == 125
    assert set(gp["rows"]).isdisjoint(gp["evaluation_rows"])
    assert gp["rows"] == mcts["rows"] == repeat["rows"]
    assert gp["evaluation_rows"] == mcts["evaluation_rows"] == repeat["evaluation_rows"]
    np.testing.assert_array_equal(gp["X"], mcts["X"][:, :10])
    np.testing.assert_array_equal(gp["Xe"], mcts["Xe"][:, :10])
    frame.loc[gp["evaluation_rows"], "target"] += 100
    frame.to_csv(payload, sep="\t", index=False)
    changed = srbench_rows("blackbox", task, 20260925, method="gp")
    np.testing.assert_array_equal(gp["y"], changed["y"])
    np.testing.assert_allclose(gp["ye"] + 100, changed["ye"])


def test_gp_generates_full_paired_population_for_new_seed():
    from src.heuristic.gp.gp import build_initial_population_document

    base = build_initial_population_document(("x1", "x2"), seed=20260925, population_size=1000)
    sn = build_initial_population_document(("x1", "x2"), seed=20260925, population_size=1000)
    assert base == sn
    assert len(base["expressions"]) == 1000


def test_canonical_export_does_not_use_test_score():
    def item(name, reward, test):
        return dict(
            exported_expression=name,
            base_reward=reward,
            complexity=3,
            search_internal_r2=0.9,
            test_r2=test,
            candidate_id=1,
        )

    result = dict(archive_export_path=[item("x1", 0.8, -100), item("x2", 0.7, 1)])
    assert min(candidate_pool(result), key=candidate_key)["exported_expression"] == "x1"


def test_output_refuses_overwrite(tmp_path):
    import pytest

    path = new_output(tmp_path / "run")
    with pytest.raises(FileExistsError):
        new_output(path)


def test_payloads_have_no_symlinks():
    assert not any(p.is_symlink() for p in (ROOT / "data").rglob("*"))
