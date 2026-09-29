"""Miscellaneous helpers: regression metrics, seed generation, etc."""

import math
import random
from typing import List

import numpy as np
from sklearn.metrics import r2_score


def get_n_seeds(n_seeds: int) -> List[int]:
    """
    Get a list of integers to use as random seeds.

    Args:
        n_seeds (int): The number of seeds to get.

    Returns:
        List[int]: A list of integers to use as random seeds.
    """
    first_10 = [
        42,
        1234,
        12345,
        666,
        777,
        1337,
        8675309,
        9001,
        1984,
        31415,
    ]
    if n_seeds <= len(first_10):
        return first_10[:n_seeds]
    else:
        random.seed(42)
        extra_seeds = random.sample(range(10000), n_seeds - len(first_10))
        return first_10 + extra_seeds


def _to_py_float(x):
    """
    Convert `x` to a built-in Python float unless it is NaN.

    Works for numpy scalars, numpy.float64, python floats, etc.
    """
    try:
        if math.isnan(x):  # Covers float('nan') and np.nan
            return np.nan
    except TypeError:
        pass  # x is not a float-like (rare here)
    return float(x)


# ============================================================
# Regression metrics
# ============================================================


def nrmse(y_true, y_pred, *, norm="std"):
    """
    Normalised Root-Mean-Squared-Error (NRMSE)

    Args:
        y_true (array-like, shape (n_samples,)): Ground-truth targets.
        y_pred (array-like, shape (n_samples,)): Predicted targets.
        norm (str, default="std"):
            - "min-max":  RMSE / (y_max - y_min)               <-- SRBench choice
            - "mean":     RMSE / |y_mean|                      <-- common in physics
            - "std":      RMSE / y_std                         <-- scale-invariant

    Returns:
        float: The chosen normalised RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    if norm == "min-max":
        denom = y_true.max() - y_true.min()
    elif norm == "mean":
        denom = np.abs(y_true.mean())
    elif norm == "std":
        denom = y_true.std(ddof=0)
    else:
        raise ValueError("norm must be 'min-max', 'mean', or 'std'")

    # Avoid division by zero (e.g. constant ground-truth)
    return np.inf if denom == 0 else rmse / denom


def nmse(y_true, y_pred):
    """
    Calculate the Normalized Mean Squared Error (NMSE).

    Input shape expected: (n_samples, n_outputs)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        NMSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"y_true.shape {y_true.shape} != y_pred.shape {y_pred.shape} in nmse"

    var = np.var(y_true, axis=0)
    nmse = np.mean((y_true - y_pred) ** 2) / var

    return np.mean(nmse)


def accuracy_tol(y_true, y_pred, tau=0.1, eps=1e-12):
    """
    Calculate the fraction of predictions within relative tolerance tau.

    Input shape expected: (n_samples, n_outputs)

    Args:
        y_true: True values
        y_pred: Predicted values
        tau: Relative tolerance threshold (default 0.1)
        eps: Small constant for numerical stability in denominator (default 1e-12)

    Returns:
        Fraction of predictions within tolerance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"y_true.shape {y_true.shape} != y_pred.shape {y_pred.shape} in accuracy_tol"

    relative_error = np.abs((y_pred - y_true) / (y_true + eps))
    within_tolerance = relative_error < tau

    return np.sum(within_tolerance) / np.prod(y_true.shape)


def accuracy_tol_max(y_true, y_pred, tau=0.1, remove_fraction_worst=0.05, eps=1e-12):
    """
    Return 1 if, for every output dimension, the (possibly trimmed) maximum
    relative error across samples is < tau. Otherwise return 0.

    Input shape expected: (n_samples, n_outputs)

    Args:
        y_true: True values, shape (n_samples, n_outputs)
        y_pred: Predicted values, same shape as y_true
        tau: Relative tolerance threshold
        remove_fraction_worst: Optional float in [0, 1). If provided, for each
            output dimension we remove this fraction of samples with the highest
            relative error *before* computing the maximum.
        eps: Small constant for numerical stability in denominator

    Returns:
        1.0 if all outputs satisfy the tolerance condition, else 0.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert (
        y_true.shape == y_pred.shape
    ), f"y_true.shape {y_true.shape} != y_pred.shape {y_pred.shape} in accuracy_tol_max"

    # Elementwise relative error
    relative_error = np.abs((y_pred - y_true) / (y_true + eps))

    # Optionally drop a fraction of the worst errors per output dimension
    if remove_fraction_worst is not None:
        if not (0.0 <= remove_fraction_worst < 1.0):
            raise ValueError("remove_fraction_worst must be in [0, 1).")
        n_samples = relative_error.shape[0]
        k = int(np.floor(remove_fraction_worst * n_samples))  # number to drop
        if k > 0:
            # Sort ascending along samples for each output dim
            sorted_err = np.sort(relative_error, axis=0)
            # Drop the k largest (i.e., keep first n_samples - k)
            trimmed = sorted_err[: n_samples - k, :]
        else:
            trimmed = relative_error
    else:
        trimmed = relative_error

    # Max relative error per output dimension
    max_err_per_dim = np.max(trimmed, axis=0)

    # If any output has max error >= tau, fail (0); else pass (1)
    return float(np.all(max_err_per_dim < tau))


def compute_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics for a given true and predicted values.

    Input shape expected: (n_samples, n_outputs).

    Metrics computed:
        - MSE per-output
        - MSE
        - RMSE
        - R2
        - NRMSE
        - NMSE
        - Accuracy tol

    Args:
        y_true: True values (array-like, shape (n_samples, n_outputs))
        y_pred: Predicted values (array-like, shape (n_samples, n_outputs))

    Returns:
        Dict[str, float]: A dictionary of regression metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    assert (
        y_true.shape == y_pred.shape
    ), f"y_true.shape {y_true.shape} != y_pred.shape {y_pred.shape} in compute_regression_metrics"
    assert y_true.ndim in [1, 2], f"y_true.ndim {y_true.ndim} must be 1 or 2 in compute_regression_metrics"
    assert y_pred.ndim in [1, 2], f"y_pred.ndim {y_pred.ndim} must be 1 or 2 in compute_regression_metrics"

    # Check if predictions are finite (excludes nans also).
    is_finite = np.isfinite(y_pred).all()

    # MSE:
    mse_per_output = np.mean((y_true - y_pred) ** 2, axis=0) if is_finite else np.full(y_true.shape[1], np.inf)
    mse_eval_total = np.mean((y_true - y_pred) ** 2) if is_finite else np.inf

    # RMSE:
    rmse_eval_total = np.sqrt(mse_eval_total) if is_finite else np.inf

    # R2:
    # Keep the mathematical R² value.  Scikit-learn otherwise replaces the
    # undefined constant-target cases with 0/1, which is a convenience for
    # model selection but would silently change the requested raw report.
    r2_eval_total = (
        r2_score(y_true, y_pred, force_finite=False) if is_finite else -np.inf
    )

    # NRMSE:
    nrmse_eval_total = nrmse(y_true, y_pred) if is_finite else np.inf

    # NMSE:
    nmse_eval_total = nmse(y_true, y_pred) if is_finite else np.inf

    # Accuracy tol:
    accuracy_tol_eval_total = accuracy_tol(y_true, y_pred) if is_finite else 0

    # Accuracy tol max:
    accuracy_tol_max_eval_total = accuracy_tol_max(y_true, y_pred) if is_finite else 0

    return {
        "mse_per_output": mse_per_output,
        "mse": mse_eval_total,
        "rmse": rmse_eval_total,
        "r2": r2_eval_total,
        "nrmse": nrmse_eval_total,
        "nmse": nmse_eval_total,
        "accuracy_tol": accuracy_tol_eval_total,
        "accuracy_tol_max": accuracy_tol_max_eval_total,
    }
