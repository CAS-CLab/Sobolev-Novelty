"""Faithful loader and scorer for the public DSRRANS training arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCORED_COMPONENTS = np.asarray((0, 1, 4, 8), dtype=np.int64)


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    nrmse: float
    inv_nrmse: float
    energy_r2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "nrmse": self.nrmse,
            "inv_nrmse": self.inv_nrmse,
            "energy_r2": self.energy_r2,
        }


@dataclass(frozen=True)
class DSRRANSData:
    """The exact arrays consumed by the public DSRRANS regression task."""

    raw_invariants: np.ndarray
    invariants: np.ndarray
    tensor_bases: np.ndarray
    target_tensor: np.ndarray

    @classmethod
    def load(cls, dataset_dir: str | Path) -> "DSRRANSData":
        root = Path(dataset_dir)
        raw = np.load(root / "case_0p8_Lambda.npy")
        tensors = np.load(root / "case_0p8_Tensors.npy")
        target = np.load(root / "case_0p8_nonLinearRShat.npy")
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError(f"unexpected Lambda shape: {raw.shape}")
        if tensors.shape != (len(raw), 10, 3, 3):
            raise ValueError(f"unexpected Tensors shape: {tensors.shape}")
        if target.shape != (len(raw), 3, 3):
            raise ValueError(f"unexpected target shape: {target.shape}")
        arrays = (raw, tensors, target)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("DSRRANS arrays contain non-finite values")
        # This is algebraically identical to the public implementation's
        # (1-exp(-I))/(1+exp(-I)), but numerically stable for large |I|.
        scaled = np.tanh(raw[:, :2] / 2.0)
        return cls(
            raw_invariants=np.asarray(raw, dtype=float),
            invariants=np.asarray(scaled, dtype=float),
            tensor_bases=np.asarray(tensors[:, :3], dtype=float),
            target_tensor=np.asarray(target, dtype=float),
        )

    @property
    def target_components(self) -> np.ndarray:
        return self.target_tensor.reshape(len(self.target_tensor), 9)[
            :, SCORED_COMPONENTS
        ]

    @property
    def basis_components(self) -> np.ndarray:
        return self.tensor_bases.reshape(len(self.tensor_bases), 3, 9)[
            :, :, SCORED_COMPONENTS
        ]

    def predict(self, coefficients: np.ndarray) -> np.ndarray:
        values = np.asarray(coefficients, dtype=float)
        if values.shape != (len(self.invariants), 3):
            raise ValueError(
                f"coefficient functions must have shape {(len(self.invariants), 3)}, "
                f"got {values.shape}"
            )
        return np.einsum(
            "nm,nmc->nc", values, self.basis_components, optimize=True
        )

    def metrics(self, prediction: np.ndarray) -> RegressionMetrics:
        predicted = np.asarray(prediction, dtype=float)
        target = self.target_components
        if predicted.shape != target.shape:
            raise ValueError(f"prediction shape {predicted.shape} != {target.shape}")
        mse = float(np.mean(np.square(target - predicted)))
        variance = float(np.var(target))
        nrmse = float(np.sqrt(mse / variance))
        target_energy = float(np.sum(np.square(target)))
        energy_r2 = float(1.0 - np.sum(np.square(target - predicted)) / target_energy)
        return RegressionMetrics(
            mse=mse,
            nrmse=nrmse,
            inv_nrmse=float(1.0 / (1.0 + nrmse)),
            energy_r2=energy_r2,
        )

    def pointwise_oracle_metrics(self) -> RegressionMetrics:
        """Upper bound when every row may choose three unrelated coefficients."""

        target = self.target_components
        bases = self.basis_components
        prediction = np.empty_like(target)
        for row in range(len(target)):
            coefficients, *_ = np.linalg.lstsq(
                bases[row].T, target[row], rcond=None
            )
            prediction[row] = coefficients @ bases[row]
        return self.metrics(prediction)


def published_model_1(invariants: np.ndarray) -> np.ndarray:
    """The three coefficient expressions shipped in turbSymbolicExpression.H."""

    values = np.asarray(invariants, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("invariants must have shape (n, 2)")
    i1, i2 = values[:, 0], values[:, 1]
    g1 = (
        0.1893473982370307 * i1
        + 0.222874993672973 * i2
        + 0.1176297370550852
    )
    g2 = (
        0.17177936132896954
        * i1
        * (
            i1
            - 0.30167164472699687 * np.square(i2) * (i1 + 2.0 * i2)
        )
        - 0.23331840158994696
    )
    g3 = i2 * (
        -2.0 * i1
        - i2
        * (
            2.5143546943288464 * i1 * np.square(i2)
            + 3.5143546943288464 * i2
            + 0.011051768568515552
        )
        + 2.979758260315827
    )
    return np.column_stack((g1, g2, g3))
