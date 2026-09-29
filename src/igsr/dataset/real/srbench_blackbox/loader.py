"""Load one local PMLB SRBench black-box regression task without networking.

The protocol has two deliberately separate levels:

1. A seed-specific 75/25 outer split isolates the independent test rows.
2. At most 4,000 rows from the remaining side form a bounded IGSR search
   pool, split 60/40 into train and validation.

The bounded pool allows at most 2,400/1,600 train/validation rows while
retaining every outer-test row.
No ground-truth equation or semantic feature name is exposed to the model:
physical columns are mapped by ordinal position to ``x1 ... xD``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from igsr.config_typing import DatasetConfig
from igsr.dataset.data_bundle import DataBundle


SPLIT_PROTOCOL = "srbench_blackbox_outer75_25_searchcap4000_inner60_40_pcg64_v1"
DEFAULT_OUTER_TEST_FRACTION = 0.25
DEFAULT_SEARCH_POOL_MAX_ROWS = 4000
DEFAULT_INNER_TRAIN_FRACTION = 0.60
TARGET_COLUMN = "target"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_arrays(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(
            json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii")
        )
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _read_task_list(path: Path) -> list[str]:
    tasks = path.read_text(encoding="utf-8").split()
    if not tasks:
        raise ValueError(f"SRBench black-box task list is empty: {path}")
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"SRBench black-box task list contains duplicates: {path}")
    return tasks


def _derived_pcg64_seed(dataset_id: str, seed: int) -> int:
    payload = f"{SPLIT_PROTOCOL}\0{dataset_id}\0{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big", signed=False)


def _split_indices(
    n_rows: int,
    dataset_id: str,
    seed: int,
    *,
    outer_test_fraction: float,
    search_pool_max_rows: int,
    inner_train_fraction: float,
) -> dict[str, np.ndarray]:
    if n_rows < 4:
        raise ValueError(
            f"SRBench black-box split requires at least four rows, got {n_rows}"
        )
    if not math.isfinite(outer_test_fraction) or not 0.0 < outer_test_fraction < 1.0:
        raise ValueError("outer test fraction must lie strictly between zero and one")
    if not isinstance(search_pool_max_rows, int) or search_pool_max_rows < 2:
        raise ValueError("search pool maximum must be an integer of at least two")
    if not math.isfinite(inner_train_fraction) or not 0.0 < inner_train_fraction < 1.0:
        raise ValueError("inner train fraction must lie strictly between zero and one")

    permutation = (
        np.random.Generator(np.random.PCG64(_derived_pcg64_seed(dataset_id, seed)))
        .permutation(n_rows)
        .astype(np.int64, copy=False)
    )
    n_test = int(math.ceil(outer_test_fraction * n_rows))
    n_outer_search = n_rows - n_test
    if n_outer_search < 2:
        raise ValueError("outer split leaves fewer than two search-side rows")
    n_search_pool = min(search_pool_max_rows, n_outer_search)
    n_train = int(round(inner_train_fraction * n_search_pool))
    n_train = min(max(n_train, 1), n_search_pool - 1)
    n_validation = n_search_pool - n_train
    if n_validation < 1:
        raise ValueError("inner split leaves no validation row")

    test = permutation[:n_test]
    outer_search = permutation[n_test:]
    search_pool = outer_search[:n_search_pool]
    return {
        "train": search_pool[:n_train],
        "validation": search_pool[n_train:],
        "test": test,
        "unused_outer_search": outer_search[n_search_pool:],
    }


def _to_frame(
    features: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        np.column_stack((target[indices], features[indices])),
        columns=["y", *feature_names],
        copy=True,
    )


def load_srbench_blackbox_dataset(cfg: DatasetConfig) -> DataBundle:
    """Load one frozen SRBench black-box task with a controlled split."""

    root_value = getattr(cfg, "srbench_blackbox_root", None)
    list_value = getattr(cfg, "srbench_blackbox_task_list", None)
    dataset_value = getattr(cfg, "srbench_blackbox_dataset_id", None)
    seed_value = getattr(cfg, "seed", None)
    if not root_value or not list_value or not dataset_value:
        raise ValueError(
            "srbench_blackbox_root, srbench_blackbox_task_list and "
            "srbench_blackbox_dataset_id are required"
        )
    if seed_value is None:
        raise ValueError("An explicit dataset seed is required for the black-box split")

    root = Path(str(root_value)).expanduser().resolve(strict=True)
    task_list_path = Path(str(list_value)).expanduser().resolve(strict=True)
    dataset_id = str(dataset_value)
    seed = int(seed_value)
    task_list_sha = _sha256_file(task_list_path)
    expected_list_sha = getattr(cfg, "srbench_blackbox_expected_task_list_sha256", None)
    if expected_list_sha and str(expected_list_sha) != task_list_sha:
        raise ValueError(
            f"Expected task-list SHA-256 {expected_list_sha!r}, found {task_list_sha!r}"
        )
    tasks = _read_task_list(task_list_path)
    if dataset_id not in tasks:
        raise ValueError(f"Dataset {dataset_id!r} is absent from the frozen task list")

    payload = root / "datasets" / dataset_id / f"{dataset_id}.tsv.gz"
    if not payload.is_file():
        raise FileNotFoundError(f"PMLB payload is missing: {payload}")
    payload_sha = _sha256_file(payload)
    expected_payload_sha = getattr(
        cfg, "srbench_blackbox_expected_payload_sha256", None
    )
    if expected_payload_sha and str(expected_payload_sha) != payload_sha:
        raise ValueError(
            f"Expected payload SHA-256 {expected_payload_sha!r}, found {payload_sha!r}"
        )

    table = pd.read_csv(payload, sep="\t", compression="gzip")
    if TARGET_COLUMN not in table.columns:
        raise ValueError(f"PMLB payload has no {TARGET_COLUMN!r} column: {payload}")
    if list(table.columns).count(TARGET_COLUMN) != 1 or len(table.columns) < 2:
        raise ValueError(f"PMLB payload schema is invalid: {payload}")
    original_feature_names = [
        str(name) for name in table.columns if name != TARGET_COLUMN
    ]
    try:
        features = table[original_feature_names].to_numpy(dtype=np.float64, copy=True)
        target = table[TARGET_COLUMN].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PMLB payload is not wholly numeric: {payload}") from exc
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError(
            f"PMLB arrays are not aligned: {features.shape=} {target.shape=}"
        )
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError(f"PMLB payload contains NaN or infinity: {payload}")
    target_variance = float(np.var(target, ddof=0))
    if not math.isfinite(target_variance) or target_variance <= 0.0:
        raise ValueError(f"PMLB target variance must be finite and positive: {payload}")

    outer_test_fraction = float(
        getattr(
            cfg, "srbench_blackbox_outer_test_fraction", DEFAULT_OUTER_TEST_FRACTION
        )
    )
    search_pool_max_rows = int(
        getattr(
            cfg, "srbench_blackbox_search_pool_max_rows", DEFAULT_SEARCH_POOL_MAX_ROWS
        )
    )
    inner_train_fraction = float(
        getattr(
            cfg,
            "srbench_blackbox_train_fraction_within_search",
            DEFAULT_INNER_TRAIN_FRACTION,
        )
    )
    frozen_values = (
        outer_test_fraction,
        search_pool_max_rows,
        inner_train_fraction,
    )
    expected_values = (
        DEFAULT_OUTER_TEST_FRACTION,
        DEFAULT_SEARCH_POOL_MAX_ROWS,
        DEFAULT_INNER_TRAIN_FRACTION,
    )
    if frozen_values != expected_values:
        raise ValueError(
            f"{SPLIT_PROTOCOL} requires {expected_values}, received {frozen_values}"
        )
    indices = _split_indices(
        len(table),
        dataset_id,
        seed,
        outer_test_fraction=outer_test_fraction,
        search_pool_max_rows=search_pool_max_rows,
        inner_train_fraction=inner_train_fraction,
    )

    feature_names = [f"x{index}" for index in range(1, features.shape[1] + 1)]
    feature_mapping = [
        {
            "ordinal": index,
            "source_name": source,
            "search_name": feature_names[index - 1],
        }
        for index, source in enumerate(original_feature_names, 1)
    ]
    feature_mapping_sha = _sha256_json(feature_mapping)
    active_indices = {
        name: values
        for name, values in indices.items()
        if name != "unused_outer_search"
    }
    index_hashes = {name: _sha256_arrays([values]) for name, values in indices.items()}
    row_hashes = {
        name: _sha256_arrays([features[values], target[values]])
        for name, values in active_indices.items()
    }
    source_identity = {
        "dataset_id": dataset_id,
        "task_list_sha256": task_list_sha,
        "payload_sha256": payload_sha,
        "feature_mapping_sha256": feature_mapping_sha,
        "n_rows": int(len(table)),
        "n_features": int(features.shape[1]),
    }
    search_identity_payload = {
        "dataset_id": dataset_id,
        "protocol": SPLIT_PROTOCOL,
        "seed": seed,
        "parameters": {
            "outer_test_fraction": outer_test_fraction,
            "search_pool_max_rows": search_pool_max_rows,
            "inner_train_fraction": inner_train_fraction,
        },
        "index_sha256": {
            "train": index_hashes["train"],
            "validation": index_hashes["validation"],
        },
        "row_sha256": {
            "train": row_hashes["train"],
            "validation": row_hashes["validation"],
        },
        "feature_mapping_sha256": feature_mapping_sha,
    }
    search_split_identity = _sha256_json(search_identity_payload)
    split_identity = _sha256_json(
        {
            "source": source_identity,
            "search": search_identity_payload,
            "test_index_sha256": index_hashes["test"],
            "test_row_sha256": row_hashes["test"],
            "unused_outer_search_index_sha256": index_hashes["unused_outer_search"],
        }
    )
    expected_full_identity = getattr(
        cfg,
        "srbench_blackbox_expected_split_identity_sha256",
        None,
    )
    if expected_full_identity and str(expected_full_identity) != split_identity:
        raise ValueError(
            f"Expected split identity {expected_full_identity!r}, found {split_identity!r}"
        )
    expected_search_identity = getattr(
        cfg,
        "srbench_blackbox_expected_search_split_identity_sha256",
        None,
    )
    if (
        expected_search_identity
        and str(expected_search_identity) != search_split_identity
    ):
        raise ValueError(
            "Expected search split identity "
            f"{expected_search_identity!r}, found {search_split_identity!r}"
        )

    counts = {name: int(len(values)) for name, values in indices.items()}
    split_provenance: Mapping[str, Any] = {
        "protocol": SPLIT_PROTOCOL,
        "paper_compatible": False,
        "protocol_warning": (
            "Controlled IGSR-vs-SN split: 25% independent test, bounded 4,000-row "
            "search pool, then 60/40 train/validation; not a full SRBench budget replication."
        ),
        "seed": seed,
        "parameters": search_identity_payload["parameters"],
        "counts": counts,
        "index_sha256": index_hashes,
        "row_sha256": row_hashes,
        "split_identity_sha256": split_identity,
        "search_split_identity_sha256": search_split_identity,
    }
    data_dictionary = (
        "Anonymous real-world SRBench black-box regression task. "
        f"There are {features.shape[1]} numeric inputs named x1 through x{features.shape[1]} "
        "and one numeric target y. Original feature names, dataset semantics, and any "
        "ground-truth expression are intentionally unavailable. Infer useful symbolic "
        "terms only from the supplied training values and search history."
    )
    return DataBundle(
        name=str(getattr(cfg, "name", "srbench_blackbox_local_controlled")),
        data_settings={
            "adapter": "srbench_blackbox_pmlb_controlled_v1",
            "instance_id": f"srbench_blackbox::{dataset_id}",
            "problem_id": dataset_id,
            "source": {
                "root": str(root),
                "task_list": str(task_list_path),
                **source_identity,
            },
            "split": split_provenance,
            "feature_mapping": feature_mapping,
            "ground_truth": {
                "raw_expression": None,
                "usage": "not_available_blackbox",
            },
            "input_vars": feature_names,
            "output_vars": ["y"],
        },
        dataset_train=_to_frame(features, target, indices["train"], feature_names),
        dataset_validation=_to_frame(
            features,
            target,
            indices["validation"],
            feature_names,
        ),
        dataset_test=_to_frame(features, target, indices["test"], feature_names),
        dataset_ood_test=None,
        target_columns=["y"],
        equation=None,
        operations_set=None,
        data_dictionary=data_dictionary,
    )
