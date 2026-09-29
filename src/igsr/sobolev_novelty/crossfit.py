"""Deterministic search-train-only repeated cross-fit stability summaries."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold

from igsr.util import compute_regression_metrics


@dataclass(frozen=True)
class CrossfitStability:
    """JSON-safe evidence used to guard a Sobolev sibling replacement."""

    success: bool
    n_rows: int
    n_features: int
    n_splits: int
    n_repeats: int
    random_state: int
    ridge_alpha: float
    fit_intercept: bool
    ridge_solver: str
    nmse_values: tuple[float, ...]
    nmse_mean: float | None
    nmse_median: float | None
    nmse_worst: float | None
    nmse_std: float | None
    held_out_rows_consulted: bool = False
    failure_type: str | None = None
    failure_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["nmse_values"] = list(self.nmse_values)
        return payload


def repeated_crossfit_stability(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    n_splits: int = 3,
    n_repeats: int = 3,
    random_state: int = 20260807,
    ridge_alpha: float = 1.0e-8,
    fit_intercept: bool = True,
    ridge_solver: str = "svd",
) -> CrossfitStability:
    """Return repeated pooled-OOF NMSE, failing closed on candidate errors.

    Each repeat predicts every search-train row exactly once.  NMSE is computed
    on that complete OOF vector, so tiny individual folds do not redefine the
    target variance.  No validation/test array is accepted by the function.
    """

    try:
        phi = np.asarray(design, dtype=float)
        y = np.asarray(targets, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if phi.ndim != 2 or y.ndim != 2 or phi.shape[0] != y.shape[0]:
            raise ValueError(
                f"design/targets must be aligned 2-D matrices, got {phi.shape} and {y.shape}"
            )
        if phi.shape[0] < 2 or phi.shape[1] < 1 or y.shape[1] < 1:
            raise ValueError(
                "cross-fit requires at least two rows, one term, and one target"
            )
        if not np.isfinite(phi).all() or not np.isfinite(y).all():
            raise ValueError("cross-fit inputs must be finite")
        if not isinstance(n_splits, int) or isinstance(n_splits, bool):
            raise ValueError("n_splits must be an integer")
        if not isinstance(n_repeats, int) or isinstance(n_repeats, bool):
            raise ValueError("n_repeats must be an integer")
        if n_splits < 2 or n_splits > phi.shape[0] or n_repeats < 1:
            raise ValueError(
                "cross-fit counts must satisfy 2<=n_splits<=n_rows and n_repeats>=1"
            )
        if not math.isfinite(float(ridge_alpha)) or float(ridge_alpha) < 0.0:
            raise ValueError("ridge_alpha must be finite and non-negative")

        splitter = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=int(random_state),
        )
        values: list[float] = []
        prediction = np.empty_like(y, dtype=float)
        assigned = np.zeros(phi.shape[0], dtype=np.int64)
        fold_in_repeat = 0
        for train_indices, holdout_indices in splitter.split(phi):
            model = Ridge(
                alpha=float(ridge_alpha),
                fit_intercept=bool(fit_intercept),
                solver=str(ridge_solver),
            ).fit(phi[train_indices], y[train_indices])
            fold_prediction = np.asarray(
                model.predict(phi[holdout_indices]), dtype=float
            )
            if fold_prediction.ndim == 1:
                fold_prediction = fold_prediction[:, None]
            if fold_prediction.shape != y[holdout_indices].shape:
                raise ValueError(
                    "cross-fit prediction shape changed: "
                    f"expected {y[holdout_indices].shape}, got {fold_prediction.shape}"
                )
            if not np.isfinite(fold_prediction).all():
                raise ValueError("cross-fit prediction is non-finite")
            prediction[holdout_indices] = fold_prediction
            assigned[holdout_indices] += 1
            fold_in_repeat += 1
            if fold_in_repeat == n_splits:
                if not np.all(assigned == 1):
                    raise RuntimeError(
                        "one cross-fit repeat did not predict every row exactly once"
                    )
                value = float(compute_regression_metrics(y, prediction)["nmse"])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"cross-fit NMSE is invalid: {value!r}")
                values.append(value)
                assigned.fill(0)
                fold_in_repeat = 0

        if fold_in_repeat != 0 or len(values) != n_repeats:
            raise RuntimeError("cross-fit splitter produced an incomplete repeat set")
        mean_value = fmean(values)
        return CrossfitStability(
            success=True,
            n_rows=int(phi.shape[0]),
            n_features=int(phi.shape[1]),
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=int(random_state),
            ridge_alpha=float(ridge_alpha),
            fit_intercept=bool(fit_intercept),
            ridge_solver=str(ridge_solver),
            nmse_values=tuple(values),
            nmse_mean=mean_value,
            nmse_median=median(values),
            nmse_worst=max(values),
            nmse_std=float(np.std(np.asarray(values, dtype=float), ddof=0)),
        )
    except Exception as error:
        rows = (
            int(design.shape[0])
            if isinstance(design, np.ndarray) and design.ndim
            else 0
        )
        features = (
            int(design.shape[1])
            if isinstance(design, np.ndarray) and design.ndim == 2
            else 0
        )
        return CrossfitStability(
            success=False,
            n_rows=rows,
            n_features=features,
            n_splits=int(n_splits) if isinstance(n_splits, int) else 0,
            n_repeats=int(n_repeats) if isinstance(n_repeats, int) else 0,
            random_state=int(random_state),
            ridge_alpha=float(ridge_alpha),
            fit_intercept=bool(fit_intercept),
            ridge_solver=str(ridge_solver),
            nmse_values=(),
            nmse_mean=None,
            nmse_median=None,
            nmse_worst=None,
            nmse_std=None,
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
