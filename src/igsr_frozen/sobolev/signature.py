"""Exact symbolic term evaluation and shared empirical Sobolev signatures."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sympy as sp

from .decomposition import (
    ProtectedDivision,
    ProtectedInverse,
    ProtectedLog,
    ProtectedLogAbs,
)


@dataclass
class SignatureBundle:
    signatures: np.ndarray
    valid_mask: np.ndarray
    output_scale: float
    input_means: np.ndarray
    input_scales: np.ndarray
    term_norms: np.ndarray
    shared_mask_seconds: float
    signature_construction_seconds: float


class SignatureError(ValueError):
    """An expected, typed signature construction failure."""

    def __init__(self, failure_type: str, message: str, valid_mask_size: int = 0):
        super().__init__(message)
        self.failure_type = failure_type
        self.valid_mask_size = valid_mask_size


def evaluate_sympy(
    expression: sp.Expr,
    symbols: Sequence[sp.Symbol],
    points: np.ndarray,
    protected_epsilon: float = 1e-6,
) -> np.ndarray:
    """Evaluate one symbolic expression, broadcasting scalar results."""

    def protected_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        denominator_array = np.asarray(denominator)
        return np.asarray(numerator) / (
            denominator_array + protected_epsilon * (denominator_array == 0)
        )

    def protected_log(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        return np.log(array + protected_epsilon * (array == 0))

    def protected_logabs(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        return np.log(np.abs(array + protected_epsilon * (array == 0)))

    modules = [
        {
            "ProtectedDivision": protected_divide,
            "ProtectedInverse": lambda value: protected_divide(1.0, value),
            "ProtectedLog": protected_log,
            "ProtectedLogAbs": protected_logabs,
            "cot": lambda value: 1.0 / np.tan(value),
            "sec": lambda value: 1.0 / np.cos(value),
            "csc": lambda value: 1.0 / np.sin(value),
        },
        "numpy",
    ]
    function = sp.lambdify(symbols, expression, modules=modules, cse=True)
    with np.errstate(all="ignore"):
        value = function(*[points[:, index] for index in range(points.shape[1])])
    array = np.asarray(value)
    if np.iscomplexobj(array):
        finite_imaginary = np.abs(np.imag(array))[np.isfinite(np.imag(array))]
        if finite_imaginary.size and float(np.max(finite_imaginary)) > 1e-10:
            return np.full(points.shape[0], np.nan)
        array = np.real(array)
    array = np.asarray(array, dtype=float)
    if array.ndim == 0 or array.size == 1:
        scalar = float(array.ravel()[0] if array.ndim else array)
        return np.full(points.shape[0], scalar)
    array = array.ravel()
    if array.shape != (points.shape[0],):
        raise ValueError(f"Expression {expression} produced {array.shape}, expected {(points.shape[0],)}")
    return array


def construct_signatures(
    values: np.ndarray,
    gradients: np.ndarray,
    full_values: np.ndarray,
    geometry_points: np.ndarray,
    input_means: np.ndarray,
    input_scales: np.ndarray,
    lambda_value: float,
    lambda_gradient: float,
    min_valid_samples: int,
    output_scale_tolerance: float,
    zero_signature_tolerance: float,
) -> SignatureBundle:
    """Apply one candidate-wide mask and build value-plus-gradient columns."""

    n_samples, n_terms = values.shape
    dimension = geometry_points.shape[1]
    if gradients.shape != (dimension, n_samples, n_terms):
        raise SignatureError(
            "evaluation_failure",
            f"Gradient shape {gradients.shape} does not match {(dimension, n_samples, n_terms)}",
        )
    mask_start = time.perf_counter()
    valid_mask = np.all(np.isfinite(geometry_points), axis=1)
    valid_mask &= np.isfinite(full_values)
    valid_mask &= np.all(np.isfinite(values), axis=1)
    if dimension:
        valid_mask &= np.all(np.isfinite(gradients), axis=(0, 2))
    n_valid = int(valid_mask.sum())
    shared_mask_seconds = time.perf_counter() - mask_start
    if n_valid < min_valid_samples:
        raise SignatureError(
            "insufficient_valid_samples",
            f"Only {n_valid}/{n_samples} candidate-wide finite samples; need {min_valid_samples}",
            n_valid,
        )
    construction_start = time.perf_counter()
    valid_values = values[valid_mask]
    valid_gradients = gradients[:, valid_mask, :]
    valid_full = full_values[valid_mask]
    output_scale = stable_rms(valid_full)
    if not np.isfinite(output_scale) or output_scale < output_scale_tolerance:
        raise SignatureError(
            "zero_output_scale",
            f"Candidate RMS output scale {output_scale!r} is below {output_scale_tolerance}",
            n_valid,
        )
    gradients_z = valid_gradients * input_scales[:, None, None]
    blocks = [math.sqrt(lambda_value / n_valid) * valid_values / output_scale]
    if lambda_gradient > 0 and dimension:
        factor = math.sqrt(lambda_gradient / (n_valid * dimension)) / output_scale
        blocks.extend(factor * gradients_z[index] for index in range(dimension))
    signatures = np.vstack(blocks)
    if not np.all(np.isfinite(signatures)):
        raise SignatureError(
            "numerical_overflow",
            "Non-finite normalized Sobolev signature",
            n_valid,
        )
    term_norms = np.asarray([stable_norm(signatures[:, index]) for index in range(n_terms)])
    zero_terms = np.flatnonzero(term_norms <= zero_signature_tolerance)
    if zero_terms.size:
        human_indices = ",".join(str(int(index + 1)) for index in zero_terms)
        error = SignatureError(
            "zero_signature",
            f"Zero Sobolev signature for term index/indices {human_indices}",
            n_valid,
        )
        error.term_norms = term_norms  # type: ignore[attr-defined]
        error.signature_shape = signatures.shape  # type: ignore[attr-defined]
        error.output_scale = output_scale  # type: ignore[attr-defined]
        raise error
    return SignatureBundle(
        signatures=signatures,
        valid_mask=valid_mask,
        output_scale=output_scale,
        input_means=input_means,
        input_scales=input_scales,
        term_norms=term_norms,
        shared_mask_seconds=shared_mask_seconds,
        signature_construction_seconds=time.perf_counter() - construction_start,
    )


def stable_norm(vector: np.ndarray) -> float:
    maximum = float(np.max(np.abs(vector))) if vector.size else 0.0
    if not np.isfinite(maximum):
        return math.inf
    if maximum == 0:
        return 0.0
    return maximum * float(np.linalg.norm(vector / maximum))


def stable_rms(vector: np.ndarray) -> float:
    maximum = float(np.max(np.abs(vector))) if vector.size else 0.0
    if not np.isfinite(maximum):
        return math.inf
    if maximum == 0:
        return 0.0
    return maximum * float(np.sqrt(np.mean(np.square(vector / maximum))))


def select_geometry_indices(
    n_samples: int,
    dataset_identity: str,
    seed: int,
    sample_size: int | None,
) -> np.ndarray:
    """Select one deterministic, sorted training subset for an entire search."""

    if n_samples < 1:
        raise ValueError("Training input is empty")
    if sample_size is None or sample_size >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    payload = f"{dataset_identity}|{seed}|{sample_size}|{n_samples}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(stable_seed)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False)).astype(np.int64)


def array_identity(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()
