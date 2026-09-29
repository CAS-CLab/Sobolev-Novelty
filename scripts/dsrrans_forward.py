#!/usr/bin/env python3
"""Compare Base and Sobolev-guided direct term selection for DSRRANS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sn_dsrrans.canonical_selection import (  # noqa: E402
    CanonicalTerm,
    canonical_terms,
    choose_candidate,
    conditional_novelties,
    flattened_row_indices,
    marginal_sse_reductions,
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
            PROJECT_ROOT
            / "data/dsrrans"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-degree", type=int, default=12)
    parser.add_argument("--max-terms", type=int, default=60)
    parser.add_argument(
        "--initial-trace",
        default=None,
        help="Optional selection trace whose first N terms seed continuation",
    )
    parser.add_argument("--initial-terms", type=int, default=None)
    parser.add_argument("--fit-fraction", type=float, default=0.75)
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--inner-fit-fraction", type=float, default=1.0)
    parser.add_argument("--inner-split-seed", type=int, default=271828)
    parser.add_argument("--geometry-samples", type=int, default=64)
    parser.add_argument(
        "--geometry-mode",
        choices=("coefficient", "tensor_sobolev", "tensor_value"),
        default="coefficient",
    )
    parser.add_argument("--geometry-gradient-weight", type=float, default=1.0)
    parser.add_argument("--geometry-cross-channel", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0 / math.sqrt(10.0))
    parser.add_argument(
        "--epsilon-schedule",
        default=None,
        help="Comma-separated step:value pairs, for example 1:0.05,41:0.15",
    )
    parser.add_argument("--structural-mix", type=float, default=0.25)
    parser.add_argument("--probe-fraction", type=float, default=0.20)
    parser.add_argument("--sn-start-step", type=int, default=1)
    parser.add_argument(
        "--methods", default="base,sn_epsilon,sn_rank"
    )
    parser.add_argument("--rcond", type=float, default=1e-10)
    parser.add_argument("--skip-degree-curve", action="store_true")
    return parser.parse_args()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    mse = float(np.mean(np.square(error)))
    variance = float(np.var(target))
    energy = float(np.sum(np.square(target)))
    return {
        "mse": mse,
        "inv_nrmse": float(1.0 / (1.0 + np.sqrt(mse / variance))),
        "energy_r2": float(1.0 - np.sum(np.square(error)) / energy),
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
    fit_scales = np.linalg.norm(fit_design, axis=0)
    fit_scales = np.where(fit_scales > 1e-15, fit_scales, 1.0)
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(
        fit_design / fit_scales, target[fit_indices], rcond=rcond
    )
    coefficients = scaled_coefficients / fit_scales
    prediction = selected_design @ coefficients
    full_scales = np.linalg.norm(selected_design, axis=0)
    full_scales = np.where(full_scales > 1e-15, full_scales, 1.0)
    scaled_full_coefficients, _, full_rank, _ = np.linalg.lstsq(
        selected_design / full_scales, target, rcond=rcond
    )
    full_coefficients = scaled_full_coefficients / full_scales
    full_refit_prediction = selected_design @ full_coefficients
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
        else:
            measured = {
                "mse": float("nan"),
                "inv_nrmse": float("nan"),
                "energy_r2": float("nan"),
            }
        for name, value in measured.items():
            result[f"{prefix}_{name}"] = value
    for name, value in metrics(target, full_refit_prediction).items():
        result[f"full_refit_{name}"] = value
    return coefficients, result


def expression_bundle(
    terms: tuple[CanonicalTerm, ...],
    selected: list[int],
    coefficients: np.ndarray,
) -> list[str]:
    x1, x2 = sp.symbols("x1 x2")
    expressions = []
    for channel in range(3):
        pieces = []
        for index, coefficient in zip(selected, coefficients, strict=True):
            term = terms[index]
            if term.channel != channel or abs(coefficient) <= 1e-12:
                continue
            pieces.append(
                sp.Float(float(coefficient))
                * x1**term.exponent_x1
                * x2**term.exponent_x2
            )
        expressions.append(str(sp.Add(*pieces)))
    return expressions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = DSRRANSData.load(args.dataset_dir)
    terms = canonical_terms(args.max_degree)
    design = tensor_design(data, terms)
    target = data.target_components.reshape(-1)

    permutation = np.random.default_rng(args.split_seed).permutation(
        len(data.invariants)
    )
    fit_count = int(round(args.fit_fraction * len(permutation)))
    fit_rows = np.sort(permutation[:fit_count])
    validation_rows = np.sort(permutation[fit_count:])
    fit_indices = flattened_row_indices(fit_rows)
    validation_indices = flattened_row_indices(validation_rows)
    if not 0.0 < args.inner_fit_fraction <= 1.0:
        raise ValueError("inner_fit_fraction must lie in (0, 1]")
    if args.inner_fit_fraction < 1.0:
        inner_permutation = np.random.default_rng(
            args.inner_split_seed
        ).permutation(fit_rows)
        inner_fit_count = int(
            round(args.inner_fit_fraction * len(inner_permutation))
        )
        selection_fit_order = inner_permutation[:inner_fit_count]
        selection_fit_rows = np.sort(selection_fit_order)
        selection_validation_rows = np.sort(inner_permutation[inner_fit_count:])
    else:
        selection_fit_order = permutation[:fit_count]
        selection_fit_rows = fit_rows
        selection_validation_rows = validation_rows
    selection_fit_indices = flattened_row_indices(selection_fit_rows)
    selection_validation_indices = flattened_row_indices(
        selection_validation_rows
    )
    geometry_rows = np.sort(
        selection_fit_order[: min(args.geometry_samples, len(selection_fit_order))]
    )
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    signatures: np.ndarray | None = None
    if any(method != "base" for method in methods):
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

    config = vars(args).copy()
    initial_selected: list[int] = []
    if args.initial_trace:
        with Path(args.initial_trace).open(newline="", encoding="utf-8") as handle:
            initial_rows = list(csv.DictReader(handle))
        initial_count = args.initial_terms or len(initial_rows)
        initial_selected = [
            int(row["selected_index"]) for row in initial_rows[:initial_count]
        ]
    config.update(
        {
            "fit_rows": int(len(fit_rows)),
            "validation_rows": int(len(validation_rows)),
            "selection_fit_rows": int(len(selection_fit_rows)),
            "selection_validation_rows": int(len(selection_validation_rows)),
            "dictionary_terms": int(len(terms)),
            "initial_selected": initial_selected,
        }
    )
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    epsilon_schedule = [(1, args.epsilon)]
    if args.epsilon_schedule:
        epsilon_schedule = [
            (int(step), float(value))
            for item in args.epsilon_schedule.split(",")
            for step, value in [item.split(":", 1)]
        ]
        epsilon_schedule.sort()
        if epsilon_schedule[0][0] != 1:
            raise ValueError("epsilon_schedule must start at step 1")

    if not args.skip_degree_curve:
        degree_rows = []
        for degree in range(args.max_degree + 1):
            selected = [
                index
                for index, term in enumerate(terms)
                if sum(term.exponent) <= degree
            ]
            _, measured = fit_and_measure(
                design,
                target,
                selected,
                fit_indices,
                validation_indices,
                rcond=args.rcond,
            )
            row: dict[str, object] = {
                "degree": degree,
                "term_count": len(selected),
                **measured,
            }
            degree_rows.append(row)
            print(
                f"degree={degree:2d} terms={len(selected):3d} "
                f"fit={measured['fit_inv_nrmse']:.9f} "
                f"validation={measured['validation_inv_nrmse']:.9f}",
                flush=True,
            )
        write_csv(output / "degree_curve.csv", degree_rows)

    method_summaries = []
    for method in methods:
        selected: list[int] = list(initial_selected)
        trace: list[dict[str, object]] = []
        changed_count = 0
        for step in range(
            len(selected) + 1, min(args.max_terms, len(terms)) + 1
        ):
            current_epsilon = next(
                value
                for start, value in reversed(epsilon_schedule)
                if step >= start
            )
            candidates = [index for index in range(len(terms)) if index not in selected]
            reductions, _, _ = marginal_sse_reductions(
                design[selection_fit_indices],
                target[selection_fit_indices],
                selected,
                candidates,
                rcond=args.rcond,
            )
            active_method = (
                "base" if step < args.sn_start_step else method
            )
            if active_method == "base":
                novelties = np.ones(len(candidates), dtype=float)
            else:
                if signatures is None:
                    raise RuntimeError("Sobolev signatures were not initialized")
                novelties = conditional_novelties(
                    signatures,
                    terms,
                    selected,
                    candidates,
                    rcond=args.rcond,
                    channel_local=not args.geometry_cross_channel,
                )
            chosen, decision = choose_candidate(
                candidates,
                reductions,
                novelties,
                method=active_method,
                epsilon=current_epsilon,
                tau=args.tau,
                structural_mix=args.structural_mix,
                probe_fraction=args.probe_fraction,
            )
            selected.append(chosen)
            changed_count += int(decision["changed_from_base"])
            coefficients, measured = fit_and_measure(
                design,
                target,
                selected,
                selection_fit_indices,
                selection_validation_indices,
                rcond=args.rcond,
            )
            term = terms[chosen]
            trace.append(
                {
                    "method": method,
                    "selection_method": active_method,
                    "selection_epsilon": current_epsilon,
                    "step": step,
                    "selected_index": chosen,
                    "selected_channel": term.channel,
                    "selected_exponent_x1": term.exponent_x1,
                    "selected_exponent_x2": term.exponent_x2,
                    "selected_label": term.label,
                    **decision,
                    "cumulative_changes": changed_count,
                    **measured,
                }
            )
            print(
                f"{method:10s} step={step:2d} term=G{term.channel + 1}:{term.label:12s} "
                f"nov={decision['chosen_novelty']:.5f} "
                f"validation={measured['validation_inv_nrmse']:.9f} "
                f"changes={changed_count}",
                flush=True,
            )
        write_csv(output / f"{method}_trace.csv", trace)
        if len(selection_validation_indices):
            best = max(
                trace, key=lambda row: float(row["validation_inv_nrmse"])
            )
        else:
            best = trace[-1]
        best_count = int(best["step"])
        best_selected = list(initial_selected) + [
            int(row["selected_index"])
            for row in trace
            if int(row["step"]) <= best_count
        ]
        best_coefficients, outer_measured = fit_and_measure(
            design,
            target,
            best_selected,
            fit_indices,
            validation_indices,
            rcond=args.rcond,
        )
        summary = {
            "method": method,
            "best_validation_step": best_count,
            "changed_from_base_steps": changed_count,
            "selection_fit_inv_nrmse": float(best["fit_inv_nrmse"]),
            "selection_validation_inv_nrmse": float(
                best["validation_inv_nrmse"]
            ),
            **outer_measured,
            "expressions": expression_bundle(
                terms, best_selected, best_coefficients
            ),
            "selected_terms": [
                {
                    "channel": terms[index].channel,
                    "exponent": list(terms[index].exponent),
                    "coefficient": float(coefficient),
                }
                for index, coefficient in zip(
                    best_selected, best_coefficients, strict=True
                )
            ],
        }
        method_summaries.append(summary)
        (output / f"{method}_best.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (output / "summary.json").write_text(
        json.dumps(method_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
