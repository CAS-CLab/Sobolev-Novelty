#!/usr/bin/env python3
"""Explore Base and Sobolev-guided exact backward pruning for DSRRANS."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sn_dsrrans.backward_selection import (  # noqa: E402
    QRBackwardWorkspace,
    choose_deletion,
    leave_one_out_novelties,
)
from sn_dsrrans.canonical_selection import (  # noqa: E402
    CanonicalTerm,
    canonical_terms,
    flattened_row_indices,
    sobolev_signatures,
    tensor_sobolev_signatures,
    tensor_design,
)
from sn_dsrrans.data import DSRRANSData  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=str(
            PROJECT_ROOT / "data/dsrrans"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-degree", type=int, default=7)
    parser.add_argument("--min-terms", type=int, default=10)
    parser.add_argument(
        "--initial-trace",
        default=None,
        help="Optional forward-selection trace whose first N terms seed pruning",
    )
    parser.add_argument("--initial-terms", type=int, default=None)
    parser.add_argument("--fit-fraction", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--geometry-samples", type=int, default=9600)
    parser.add_argument(
        "--geometry-mode",
        choices=("coefficient", "tensor_sobolev", "tensor_value"),
        default="coefficient",
    )
    parser.add_argument("--geometry-gradient-weight", type=float, default=1.0)
    parser.add_argument("--geometry-cross-channel", action="store_true")
    parser.add_argument("--epsilon-r2", type=float, default=1e-5)
    parser.add_argument("--methods", default="base,sn_epsilon")
    parser.add_argument("--rcond", type=float, default=1e-10)
    parser.add_argument(
        "--snapshot-terms", default="100,80,60,40,30,20,10"
    )
    return parser.parse_args()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    mse = float(np.mean(np.square(error)))
    return {
        "mse": mse,
        "inv_nrmse": float(1.0 / (1.0 + np.sqrt(mse / np.var(target)))),
        "energy_r2": float(
            1.0 - np.sum(np.square(error)) / np.sum(np.square(target))
        ),
    }


def fit_and_measure(
    design: np.ndarray,
    target: np.ndarray,
    selected: list[int],
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    selected_design = design[:, selected]
    fit_design = selected_design[fit_indices]
    scales = np.linalg.norm(fit_design, axis=0)
    scales = np.where(scales > 1e-15, scales, 1.0)
    scaled, _, rank, _ = np.linalg.lstsq(
        fit_design / scales, target[fit_indices], rcond=rcond
    )
    coefficients = scaled / scales
    prediction = selected_design @ coefficients
    full_scales = np.linalg.norm(selected_design, axis=0)
    full_scales = np.where(full_scales > 1e-15, full_scales, 1.0)
    scaled_full, _, full_rank, _ = np.linalg.lstsq(
        selected_design / full_scales, target, rcond=rcond
    )
    full_coefficients = scaled_full / full_scales
    full_prediction = selected_design @ full_coefficients
    result: dict[str, float | int] = {
        "rank": int(rank),
        "full_refit_rank": int(full_rank),
    }
    for prefix, indices in (
        ("fit", fit_indices),
        ("validation", validation_indices),
        ("full", np.arange(len(target))),
    ):
        if len(indices):
            measured = metrics(target[indices], prediction[indices])
            for name, value in measured.items():
                result[f"{prefix}_{name}"] = value
        else:
            for name in ("mse", "inv_nrmse", "energy_r2"):
                result[f"{prefix}_{name}"] = float("nan")
    for name, value in metrics(target, full_prediction).items():
        result[f"full_refit_{name}"] = value
    return coefficients, result


def expression_bundle(
    terms: tuple[CanonicalTerm, ...],
    selected: list[int],
    coefficients: np.ndarray,
) -> list[str]:
    x1, x2 = sp.symbols("x1 x2")
    result = []
    for channel in range(3):
        pieces = []
        for index, coefficient in zip(selected, coefficients, strict=True):
            term = terms[index]
            if term.channel == channel and abs(coefficient) > 1e-12:
                pieces.append(
                    sp.Float(float(coefficient))
                    * x1**term.exponent_x1
                    * x2**term.exponent_x2
                )
        result.append(str(sp.Add(*pieces)))
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = DSRRANSData.load(args.dataset_dir)
    source_terms = canonical_terms(args.max_degree)
    source_indices = list(range(len(source_terms)))
    if args.initial_trace:
        with Path(args.initial_trace).open(newline="", encoding="utf-8") as handle:
            forward_rows = list(csv.DictReader(handle))
        initial_count = args.initial_terms or len(forward_rows)
        source_indices = [
            int(row["selected_index"]) for row in forward_rows[:initial_count]
        ]
    terms = tuple(source_terms[index] for index in source_indices)
    design = tensor_design(data, terms)
    target = data.target_components.reshape(-1)

    permutation = np.random.default_rng(args.split_seed).permutation(
        len(data.invariants)
    )
    fit_count = int(round(args.fit_fraction * len(permutation)))
    fit_order = permutation[:fit_count]
    fit_rows = np.sort(fit_order)
    validation_rows = np.sort(permutation[fit_count:])
    fit_indices = flattened_row_indices(fit_rows)
    validation_indices = flattened_row_indices(validation_rows)
    geometry_rows = np.sort(
        fit_order[: min(args.geometry_samples, len(fit_order))]
    )
    if args.geometry_mode == "coefficient":
        signatures = sobolev_signatures(data.invariants[geometry_rows], terms)
    else:
        signatures = tensor_sobolev_signatures(
            data,
            terms,
            rows=geometry_rows,
            lambda_gradient=(
                0.0
                if args.geometry_mode == "tensor_value"
                else args.geometry_gradient_weight
            ),
        )
    _, signatures = np.linalg.qr(signatures, mode="reduced")
    workspace = QRBackwardWorkspace(
        design[fit_indices], target[fit_indices], rcond=args.rcond
    )
    snapshot_counts = {
        int(value) for value in args.snapshot_terms.split(",") if value.strip()
    }
    snapshot_counts.add(len(terms))
    config = {
        **vars(args),
        "dictionary_terms": len(terms),
        "source_dictionary_terms": len(source_terms),
        "source_indices": source_indices,
        "fit_rows": len(fit_rows),
        "validation_rows": len(validation_rows),
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_summaries = []
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        selected = list(range(len(terms)))
        trace: list[dict[str, object]] = []
        snapshots = []
        changed_count = 0

        def capture_snapshot() -> None:
            coefficients, measured = fit_and_measure(
                design,
                target,
                selected,
                fit_indices,
                validation_indices,
                rcond=args.rcond,
            )
            snapshots.append(
                {
                    "term_count": len(selected),
                    "active_term_count": int(
                        np.count_nonzero(np.abs(coefficients) > 1e-12)
                    ),
                    **measured,
                    "expressions": expression_bundle(
                        terms, selected, coefficients
                    ),
                    "selected_indices": [source_indices[index] for index in selected],
                    "coefficients": list(map(float, coefficients)),
                }
            )

        if len(selected) in snapshot_counts:
            capture_snapshot()
        while len(selected) > args.min_terms:
            increases, current_fit = workspace.deletion_increases(selected)
            novelties = leave_one_out_novelties(
                signatures,
                terms,
                selected,
                rcond=args.rcond,
                channel_local=not args.geometry_cross_channel,
            )
            deleted, decision = choose_deletion(
                selected,
                increases,
                novelties,
                method=method,
                epsilon_r2=args.epsilon_r2,
                target_energy=workspace.target_energy,
            )
            deleted_position = selected.index(deleted)
            deleted_coefficient = float(
                current_fit.coefficients[deleted_position]
            )
            selected.remove(deleted)
            changed_count += int(decision["changed_from_base"])
            next_fit = workspace.fit(selected)
            fit_mse = next_fit.sse / len(fit_indices)
            fit_inv_nrmse = 1.0 / (
                1.0 + np.sqrt(fit_mse / np.var(target[fit_indices]))
            )
            fit_energy_r2 = 1.0 - next_fit.sse / workspace.target_energy
            term = terms[deleted]
            row: dict[str, object] = {
                "method": method,
                "term_count": len(selected),
                "deleted_index": deleted,
                "deleted_source_index": source_indices[deleted],
                "deleted_channel": term.channel,
                "deleted_exponent_x1": term.exponent_x1,
                "deleted_exponent_x2": term.exponent_x2,
                "deleted_label": term.label,
                "deleted_coefficient": deleted_coefficient,
                **decision,
                "cumulative_changes": changed_count,
                "fit_sse": next_fit.sse,
                "fit_rank": next_fit.rank,
                "fit_inv_nrmse": fit_inv_nrmse,
                "fit_energy_r2": fit_energy_r2,
            }
            trace.append(row)
            if len(selected) in snapshot_counts:
                capture_snapshot()
            print(
                f"{method:10s} terms={len(selected):3d} "
                f"drop=G{term.channel + 1}:{term.label:12s} "
                f"nov={decision['chosen_novelty']:.5f} "
                f"R2={fit_energy_r2:.9f} changes={changed_count}",
                flush=True,
            )
        write_csv(output / f"{method}_trace.csv", trace)
        summary = {
            "method": method,
            "epsilon_r2": args.epsilon_r2,
            "changed_from_base_steps": changed_count,
            "snapshots": snapshots,
        }
        all_summaries.append(summary)
        (output / f"{method}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
