import ast
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import structlog
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.utils.validation import check_array

from igsr.agent.agent import PostprocessorOutput
from igsr.compute_profiler import ComputeProfiler
from igsr.config_typing import MainConfig
from igsr.dataset import DataBundle
from igsr.parser import extract_code_blocks
from igsr.util import compute_regression_metrics

OptimizationMethod = Union[LinearRegression, Ridge, Lasso]

_ALLOWED_NUMPY_FUNCTIONS = {
    # Trigonometric / hyperbolic functions.
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    # Element-wise elementary functions useful in symbolic-regression terms.
    "exp": np.exp,
    "expm1": np.expm1,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "abs": np.abs,
    "absolute": np.absolute,
    "sign": np.sign,
    "square": np.square,
    "power": np.power,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}

# Exact overlap between the executable IGSR grammar and the EIC adapter's
# differentiable grammar.  Opt-in experiments use this for *both* arms so the
# root prompts and completion tape stay strictly paired.
_SOBOLEV_COMMON_NUMPY_FUNCTIONS = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "sqrt",
        "abs",
        "square",
        "power",
    }
)

# ``eval`` inserts Python builtins when the key is absent, so keep it explicitly
# empty even though the AST validator below already rejects builtin names/calls.
SAFE_GLOBALS = {"__builtins__": {}, "np": np, "numpy": np}

SAFE_GLOBALS_AIF = {
    **SAFE_GLOBALS,
    # AI-Feynman expressions may use unqualified elementary function names.
    **_ALLOWED_NUMPY_FUNCTIONS,
    "pow": np.power,
    "pi": np.pi,
}

_ALLOWED_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)


def _validate_safe_expression(node: ast.AST, local_names: set[str], use_aif: bool) -> None:
    """Validate one expression node against IGSR's executable term grammar.

    The validator deliberately handles every permitted node recursively instead
    of walking the tree and trying to blacklist dangerous constructs. This keeps
    attributes and calls safe even when Python or NumPy gains new APIs.
    """

    if isinstance(node, ast.Expression):
        _validate_safe_expression(node.body, local_names, use_aif)
        return

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Only real numeric constants are allowed, got {node.value!r}")
        return

    if isinstance(node, ast.Name):
        if node.id in local_names or (use_aif and node.id == "pi"):
            return
        raise ValueError(f"Unknown or reserved name: {node.id!r}")

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
            raise ValueError(f"Operator {type(node.op).__name__!r} is not allowed")
        _validate_safe_expression(node.left, local_names, use_aif)
        _validate_safe_expression(node.right, local_names, use_aif)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
            raise ValueError(f"Unary operator {type(node.op).__name__!r} is not allowed")
        _validate_safe_expression(node.operand, local_names, use_aif)
        return

    if isinstance(node, ast.Attribute):
        # Attribute access is otherwise forbidden. ``np.pi`` is the only safe
        # non-call attribute in the term language.
        if not (
            isinstance(node.value, ast.Name)
            and node.value.id in {"np", "numpy"}
            and node.attr == "pi"
        ):
            raise ValueError("Only np.pi/numpy.pi attribute access is allowed")
        return

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in term functions")

        if isinstance(node.func, ast.Attribute):
            if not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy"}
                and node.func.attr in _ALLOWED_NUMPY_FUNCTIONS
            ):
                raise ValueError("Only explicitly whitelisted np/numpy functions may be called")
        elif isinstance(node.func, ast.Name):
            allowed_aif_names = set(_ALLOWED_NUMPY_FUNCTIONS) | {"pow"}
            if not use_aif or node.func.id not in allowed_aif_names:
                raise ValueError("Only explicitly whitelisted mathematical functions may be called")
        else:
            raise ValueError("Indirect or nested callable expressions are not allowed")

        for arg in node.args:
            _validate_safe_expression(arg, local_names, use_aif)
        return

    raise ValueError(f"Expression syntax {type(node).__name__!r} is not allowed")


# ================================================================================
# Design Matrix, Optimization, Influence
# ================================================================================


@dataclass
class DesignMatrix:
    phi: np.ndarray
    term_names: List[str]
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_terms(cls, terms: List[str], data: Dict[str, np.ndarray]) -> "DesignMatrix":
        if not data:
            raise ValueError("DesignMatrix requires at least one input feature")
        row_counts = {len(np.asarray(values)) for values in data.values()}
        if len(row_counts) != 1:
            raise ValueError(f"Input features have inconsistent row counts: {sorted(row_counts)}")
        n_rows = row_counts.pop()
        cols = []
        ok_terms = []
        errors = []
        for t in terms:
            try:
                values = np.asarray(safe_eval(t, data))
                if np.iscomplexobj(values):
                    raise ValueError("complex-valued terms are not supported")
                if values.ndim == 0 or values.size == 1:
                    values = np.full(n_rows, float(values.reshape(-1)[0] if values.ndim else values))
                else:
                    values = np.squeeze(values)
                    if values.shape != (n_rows,):
                        raise ValueError(
                            f"term produced shape {values.shape}, expected a scalar or {(n_rows,)}"
                        )
                    values = np.asarray(values, dtype=float)
                cols.append(values)
                ok_terms.append(t)
            except Exception as e:
                print(f"[term error] {t!r}: {e}")
                errors.append(f"[term error] {t!r}: {e}")
        phi = np.column_stack(cols) if cols else np.empty((n_rows, 0), dtype=float)
        return cls(phi, ok_terms, errors)


def allowed_numpy_function_names(cfg: MainConfig) -> tuple[str, ...]:
    """Return the ordered function vocabulary for an experiment profile."""

    profile = str(getattr(cfg.experiment, "term_grammar_profile", "full")).strip().lower()
    if profile == "full":
        allowed = set(_ALLOWED_NUMPY_FUNCTIONS)
    elif profile == "sobolev_common_v1":
        allowed = set(_SOBOLEV_COMMON_NUMPY_FUNCTIONS)
    else:
        raise ValueError(f"Unknown experiment.term_grammar_profile={profile!r}")
    return tuple(name for name in _ALLOWED_NUMPY_FUNCTIONS if name in allowed)


def _stable_rms(values: np.ndarray) -> float:
    """Compute a float64 RMS without squaring the original magnitude."""

    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0 or not np.isfinite(array).all():
        return float("nan")
    peak = float(np.max(np.abs(array)))
    if peak == 0.0:
        return 0.0
    with np.errstate(all="ignore"):
        return peak * float(np.sqrt(np.mean((array / peak) ** 2)))


def _stable_std(values: np.ndarray) -> float:
    """Compute a population standard deviation without avoidable overflow."""

    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0 or not np.isfinite(array).all():
        return float("nan")
    peak = float(np.max(np.abs(array)))
    if peak == 0.0:
        return 0.0
    with np.errstate(all="ignore"):
        return peak * float(np.std(array / peak, ddof=0))


def _candidate_policy_errors(
    terms: List[str],
    data: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    design_matrix: DesignMatrix,
    cfg: MainConfig,
    *,
    split_name: str,
) -> List[str]:
    """Apply opt-in, search-only grammar/domain/numerical policy guards."""

    errors: List[str] = []
    allowed_functions = set(allowed_numpy_function_names(cfg))
    profile = str(getattr(cfg.experiment, "term_grammar_profile", "full")).strip().lower()
    reject_constant = bool(getattr(cfg.experiment, "reject_constant_input_terms", False))
    tolerance = float(getattr(cfg.experiment, "constant_input_scale_tolerance", 1.0e-12))
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("experiment.constant_input_scale_tolerance must be finite and positive")

    constant_names = {
        name for name, values in data.items() if _stable_std(values) <= tolerance
    } if reject_constant else set()

    def exposes_additive_sum(node: ast.AST) -> bool:
        """Whether ``expand_mul`` would split this proposed basis line."""

        if isinstance(node, ast.Expression):
            return exposes_additive_sum(node.body)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return True
            if isinstance(node.op, ast.Mult):
                return exposes_additive_sum(node.left) or exposes_additive_sum(node.right)
            if isinstance(node.op, ast.Div):
                # expand_mul distributes a sum in the numerator, not inside a denominator.
                return exposes_additive_sum(node.left)
            if isinstance(node.op, ast.Pow):
                # The source-of-truth policy does not expand powers.
                return False
        if isinstance(node, ast.UnaryOp):
            return exposes_additive_sum(node.operand)
        # Function arguments are not expanded into outer additive terms.
        return False

    for term in terms:
        try:
            tree = ast.parse(term, mode="eval")
        except (SyntaxError, ValueError):
            continue  # DesignMatrix/safe_eval owns the syntax error message.
        if profile != "full":
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute):
                    function_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    function_name = node.func.id
                else:
                    continue
                if function_name not in allowed_functions:
                    errors.append(
                        f"Term {term} uses function {function_name!r}, outside "
                        f"term_grammar_profile={profile!r}."
                    )
        if bool(getattr(cfg.experiment, "require_expand_mul_atomic_terms", False)) and exposes_additive_sum(tree):
            errors.append(
                f"Term {term} contains an outer additive sum that expand_mul would split; "
                "submit each additive term on a separate line."
            )
        if constant_names:
            references = {
                node.id for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id in constant_names
            }
            if references:
                errors.append(
                    f"Term {term} references constant {split_name} input(s) "
                    f"{sorted(references)}; the shared Sobolev-compatible policy rejects it."
                )

    raw_ratio = getattr(cfg.experiment, "max_design_column_scale_ratio", None)
    if raw_ratio is not None and design_matrix.phi.shape[1]:
        max_ratio = float(raw_ratio)
        if not np.isfinite(max_ratio) or max_ratio <= 1.0:
            raise ValueError("experiment.max_design_column_scale_ratio must exceed one or be null")
        target_matrix = np.column_stack(list(y.values()))
        target_scale = _stable_rms(target_matrix)
        column_scales = [_stable_rms(design_matrix.phi[:, index]) for index in range(design_matrix.phi.shape[1])]
        reference_scales = [scale for scale in [target_scale, *column_scales] if np.isfinite(scale) and scale > 0.0]
        if reference_scales:
            minimum = min(reference_scales)
            policy_terms = set(terms)
            for term, scale in zip(design_matrix.term_names, column_scales, strict=True):
                if (
                    term in policy_terms
                    and np.isfinite(scale)
                    and scale > max_ratio * minimum
                ):
                    errors.append(
                        f"Term {term} has RMS scale {scale:.6g} on {split_name}, "
                        f"exceeding the shared design/target scale ratio limit {max_ratio:.6g}."
                    )
    return errors


def _search_matrix_numerical_errors(
    design_matrix: DesignMatrix,
    y: Dict[str, np.ndarray],
    *,
    split_name: str,
) -> List[str]:
    """Reject finite columns that cannot enter the frozen float64 Ridge path.

    ``check_array`` only rejects NaN/Inf values.  A column can nevertheless be
    finite while its Gram diagonal overflows (for example ``P**t`` around
    ``1e162``).  Ridge then fails while forming ``X.T @ X``.  Search candidate
    validation therefore mirrors the quadratic quantities used downstream,
    without clipping, rescaling, or consulting held-out report rows.
    """

    errors: List[str] = []
    phi = np.asarray(design_matrix.phi, dtype=float)
    y_matrix = np.column_stack(list(y.values())).astype(float, copy=False)

    if not np.isfinite(phi).all():
        errors.append(f"{split_name} design matrix contains NaN or infinity values.")
        return errors
    if not np.isfinite(y_matrix).all():
        errors.append(f"{split_name} targets contain NaN or infinity values.")
        return errors

    with np.errstate(over="ignore", invalid="ignore"):
        gram = phi.T @ phi
        feature_target_cross = phi.T @ y_matrix

    if gram.size:
        bad_diagonal = np.flatnonzero(~np.isfinite(np.diag(gram)))
        for index in bad_diagonal:
            errors.append(
                f"Term {design_matrix.term_names[int(index)]} has a non-finite "
                f"float64 squared norm on {split_name}; it cannot enter Ridge."
            )
        if not np.isfinite(gram).all() and not bad_diagonal.size:
            errors.append(
                f"The float64 feature Gram matrix is non-finite on {split_name}; "
                "the proposed term combination cannot enter Ridge."
            )

    if not np.isfinite(feature_target_cross).all():
        bad_rows = np.flatnonzero(~np.isfinite(feature_target_cross).all(axis=1))
        bad_terms = [design_matrix.term_names[int(index)] for index in bad_rows]
        errors.append(
            f"Terms {bad_terms} produce a non-finite float64 feature-target "
            f"cross-product on {split_name}; they cannot enter Ridge."
        )
    return errors


class NoOptimization:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data: np.ndarray, y: np.ndarray):
        _, n_terms = data.shape
        n_outputs = y.shape[1]
        self.coef_ = np.ones((n_outputs, n_terms))
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        return data @ self.coef_.T


def get_optimization_method(cfg: MainConfig) -> Tuple[OptimizationMethod, Dict[str, Any]]:
    requested_intercept = getattr(cfg.experiment, "fit_intercept", None)
    ridge_solver = getattr(cfg.experiment, "ridge_solver", None)

    def ridge_kwargs(*, alpha: Optional[float] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "fit_intercept": False if requested_intercept is None else bool(requested_intercept)
        }
        if alpha is not None:
            kwargs["alpha"] = float(alpha)
        if ridge_solver is not None:
            kwargs["solver"] = str(ridge_solver)
        return kwargs

    if cfg.experiment.optimization_method == "linear":
        if cfg.experiment.fallback_ridge_small_alpha_for_stability:
            return Ridge, ridge_kwargs(alpha=1e-8)
        return LinearRegression, (
            {} if requested_intercept is None else {"fit_intercept": bool(requested_intercept)}
        )
    elif cfg.experiment.optimization_method == "ridge":
        return Ridge, ridge_kwargs()
    elif cfg.experiment.optimization_method == "lasso":
        return Lasso, {
            "fit_intercept": False if requested_intercept is None else bool(requested_intercept)
        }
    elif cfg.experiment.optimization_method == "no_optimization":
        return NoOptimization, {"fit_intercept": False}
    else:
        raise ValueError(f"Unknown optimization: {cfg.experiment.optimization_method}")


def _get_intercept(reg) -> Optional[np.ndarray]:
    """Extract intercept from a fitted model, or None if not fitted / all zeros."""
    intercept = getattr(reg, "intercept_", None)
    if intercept is None:
        return None
    intercept = np.atleast_1d(np.asarray(intercept, dtype=float))
    if not np.any(intercept != 0):
        return None
    return intercept


@dataclass
class OLSResult:
    # Core
    weights: np.ndarray  # (n_terms, n_outputs)
    y_pred: np.ndarray  # (n_samples, n_outputs)
    mse: np.ndarray  # per-output  (n_outputs,)
    mse_total: float  # scalar – mean over all outputs

    delta: np.ndarray  # (n_terms, n_outputs)

    reg: OptimizationMethod
    intercept: Optional[np.ndarray]  # (n_outputs,) or None when fit_intercept=False

    # Validation (optional)
    y_pred_validation: Optional[np.ndarray]  # (n_val, n_outputs) or None
    mse_validation: Optional[np.ndarray]  # per-output (n_outputs,) or None
    mse_validation_total: Optional[float]  # scalar or None
    delta_validation: Optional[np.ndarray]  # (n_terms, n_outputs) or None


def _as_2d(y: np.ndarray) -> np.ndarray:
    """
    Ensure `y` is` (n_samples, n_outputs)` even when `n_outputs == 1`.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    return y


def ols_influence(
    cfg: MainConfig,
    Φ: np.ndarray,
    y: np.ndarray,
    Φ_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> OLSResult:
    """
    OLS fit + leave-one-weight-zero influence Δ_k for multiple outputs.

    * Φ        ... (n_samples, n_terms)
    * y        ... (n_samples,)  or  (n_samples, n_outputs)
    * *_val    ... same for validation set (optional)

    Returns per-output vectors for MSE and Δ, and a weight matrix of shape (n_terms, n_outputs).
    """
    y2 = _as_2d(np.asarray(y))  # (n, m)
    reg, reg_kwargs = get_optimization_method(cfg)
    reg = reg(**reg_kwargs).fit(Φ, y2)
    w = reg.coef_.T  # (p, m)
    w = _as_2d(w)

    y_hat = reg.predict(Φ)  # (n, m)
    y_hat = _as_2d(y_hat)
    res = y2 - y_hat
    mse = np.mean(res**2, axis=0)  # (m,)
    mse_total = np.mean(res**2)  # scalar

    p, m = w.shape

    # Flags
    refit_aware = cfg.experiment.refit_aware
    efficient = cfg.experiment.refit_aware_efficient
    opt_method = cfg.experiment.optimization_method
    if cfg.experiment.fallback_ridge_small_alpha_for_stability:
        opt_method = "ridge"

    if not refit_aware:
        # Non-refit (leave-one-weight-zero)
        delta = np.empty((p, m))
        for k in range(p):
            y_hat_minus_k = y_hat - Φ[:, [k]] * w[k]
            delta[k] = np.mean((y2 - y_hat_minus_k) ** 2, axis=0) - mse

        # Validation:
        if Φ_val is not None and y_val is not None:
            yv2 = _as_2d(np.asarray(y_val))
            y_hat_val = reg.predict(Φ_val)
            y_hat_val = _as_2d(y_hat_val)
            res_val = yv2 - y_hat_val
            mse_val = np.mean(res_val**2, axis=0)
            mse_val_total = np.mean(res_val**2)

            delta_val = np.empty_like(delta)
            for k in range(p):
                y_hat_val_minus_k = y_hat_val - Φ_val[:, [k]] * w[k]
                delta_val[k] = np.mean((yv2 - y_hat_val_minus_k) ** 2, axis=0) - mse_val
        else:
            y_hat_val = mse_val = mse_val_total = delta_val = None
    else:
        # Refit-aware: compute w^{(-k)} and evaluate deltas

        # Baseline validation stats if available
        if Φ_val is not None and y_val is not None:
            yv2 = _as_2d(np.asarray(y_val))
            y_hat_val = reg.predict(Φ_val)
            res_val = yv2 - y_hat_val
            mse_val = np.mean(res_val**2, axis=0)
            mse_val_total = np.mean(res_val**2)
        else:
            y_hat_val = mse_val = mse_val_total = None

        if efficient and (opt_method in ("linear", "ridge")):
            # NOTE: Efficient implementation option, if available, default false.

            # Efficient mode (default):
            # uses partitioned-inverse identities to compute w^{(-j)} without full refits;
            # - for OLS, train ΔMSE uses a closed form;
            # - for Ridge, it constructs w^{(-j)} via A_inv and evaluates train/val via predictions.

            # Build (X^T X [+ λ I])^{-1}
            A = Φ.T @ Φ
            if opt_method == "ridge":
                ridge_lambda = float(getattr(reg, "alpha", 0.0))
                if ridge_lambda != 0.0:
                    A = A + ridge_lambda * np.eye(p, dtype=A.dtype)
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A)

            delta = np.empty((p, m))
            delta_val = None if Φ_val is None or y_val is None else np.empty((p, m))

            # For OLS, an efficient closed-form exists for train ΔMSE
            if opt_method == "linear":
                n = Φ.shape[0]
                diag = np.diag(A_inv)
                # Avoid division by zero
                diag = np.where(diag == 0, np.finfo(diag.dtype).eps, diag)
                for j in range(p):
                    delta[j] = (w[j] ** 2) / diag[j] / n
            else:
                # Ridge: compute via predictions after weight update
                for j in range(p):
                    mask = np.ones(p, dtype=bool)
                    mask[j] = False
                    alpha_j = A_inv[j, j]
                    beta = A_inv[mask, j]
                    w_minus_j = w[mask, :] - (beta[:, None] / alpha_j) * w[j, :][None, :]
                    y_hat_train_minus_j = Φ[:, mask] @ w_minus_j
                    delta[j] = np.mean((y2 - y_hat_train_minus_j) ** 2, axis=0) - mse

            # Validation deltas via predictions
            if Φ_val is not None and y_val is not None:
                for j in range(p):
                    mask = np.ones(p, dtype=bool)
                    mask[j] = False
                    alpha_j = A_inv[j, j]
                    beta = A_inv[mask, j]
                    w_minus_j = w[mask, :] - (beta[:, None] / alpha_j) * w[j, :][None, :]
                    y_hat_val_minus_j = Φ_val[:, mask] @ w_minus_j
                    delta_val[j] = np.mean((yv2 - y_hat_val_minus_j) ** 2, axis=0) - mse_val
        else:
            # Full refit loop (or unsupported method like lasso)
            delta = np.empty((p, m))
            delta_val = None if Φ_val is None or y_val is None else np.empty((p, m))

            for j in range(p):
                mask = np.ones(p, dtype=bool)
                mask[j] = False

                opt_cls, reg_kwargs = get_optimization_method(cfg)
                # Try to preserve alpha for ridge/lasso if available
                if opt_method in ("ridge", "lasso"):
                    reg_kwargs["alpha"] = getattr(reg, "alpha", 1.0)
                    reg_refit = opt_cls(**reg_kwargs)
                else:
                    reg_refit = opt_cls(**reg_kwargs)

                reg_refit = reg_refit.fit(Φ[:, mask], y2)
                w_minus_j = reg_refit.coef_.T

                # Train delta
                y_hat_train_minus_j = Φ[:, mask] @ w_minus_j
                delta[j] = np.mean((y2 - y_hat_train_minus_j) ** 2, axis=0) - mse

                # Validation delta
                if Φ_val is not None and y_val is not None:
                    y_hat_val_minus_j = reg_refit.predict(Φ_val[:, mask])
                    if mse_val is None:
                        # compute baseline if missing
                        y_hat_val = reg.predict(Φ_val)
                        res_val = _as_2d(np.asarray(y_val)) - y_hat_val
                        mse_val = np.mean(res_val**2, axis=0)
                        mse_val_total = np.mean(res_val**2)
                    delta_val[j] = np.mean((_as_2d(np.asarray(y_val)) - y_hat_val_minus_j) ** 2, axis=0) - mse_val

    # Check for the case where all w's are ~zero (tolerance = 1e-8) and raise an error.
    if cfg.experiment.error_on_all_weights_zero:
        EPSILON = 1e-8
        if np.abs(w).max() < EPSILON:
            raise ValueError(f"All weights are ~zero (tolerance = {EPSILON}): {w}")

    return OLSResult(
        weights=w,
        y_pred=y_hat,
        mse=mse,
        mse_total=mse_total,
        delta=delta,
        reg=reg,
        intercept=_get_intercept(reg),
        y_pred_validation=y_hat_val,
        mse_validation=mse_val,
        mse_validation_total=mse_val_total,
        delta_validation=delta_val,
    )


def safe_eval(expr: str, local_ctx: Dict[str, np.ndarray], use_aif: bool = False) -> np.ndarray:
    """
    Evaluate `expr` using only numpy + the variables in `local_ctx`.
    Raises ValueError on any unsafe code pattern.

    Args:
        expr (str): The expression to evaluate.
        local_ctx (Dict[str, np.ndarray]): The local context to evaluate the expression in.

    Returns:
        np.ndarray: The evaluated expression.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid term expression: {exc}") from exc

    reserved_names = {"np", "numpy", "__builtins__"}
    if use_aif:
        reserved_names |= set(_ALLOWED_NUMPY_FUNCTIONS) | {"pow", "pi"}
    safe_locals = {name: value for name, value in local_ctx.items() if name not in reserved_names}

    try:
        _validate_safe_expression(tree, set(safe_locals), use_aif)
        return eval(compile(tree, "<expr>", "eval"), SAFE_GLOBALS_AIF if use_aif else SAFE_GLOBALS, safe_locals)
    except Exception as exc:
        raise ValueError(f"Error evaluating term {expr!r}: {exc}") from exc


def evaluate(
    terms: List[str],
    data_eval: Dict[str, np.ndarray],
    y_eval: Dict[str, np.ndarray],
    reg: OptimizationMethod,
) -> Tuple[
    np.ndarray,  # Main metric per-output
    float,  # Main metric total
    Dict[str, float],  # All metrics per-output
]:
    design_matrix = DesignMatrix.from_terms(terms, data_eval)
    if design_matrix.errors or design_matrix.term_names != terms:
        errors = "; ".join(design_matrix.errors) or "the evaluated term list changed"
        raise ValueError(
            "Report/search evaluation is fail-closed: every fitted term must "
            f"evaluate on the requested split ({errors})"
        )
    if not np.isfinite(design_matrix.phi).all():
        raise ValueError(
            "Report/search evaluation is fail-closed: the fitted expression "
            "produced NaN or infinity on the requested split"
        )
    Φ_final_eval = design_matrix.phi
    pred = reg.predict(Φ_final_eval)
    y_eval_np = np.column_stack(list(y_eval.values()))
    if not np.isfinite(pred).all():
        raise ValueError("The fitted expression produced non-finite predictions")
    if not np.isfinite(y_eval_np).all():
        raise ValueError("The evaluation targets contain non-finite values")
    eval_metrics = compute_regression_metrics(y_eval_np, pred)
    scalar_metric_names = (
        "mse",
        "rmse",
        "r2",
        "nrmse",
        "nmse",
        "accuracy_tol",
        "accuracy_tol_max",
    )
    if any(not np.isfinite(eval_metrics[name]) for name in scalar_metric_names):
        raise ValueError("Regression metric computation produced non-finite values")
    return (
        eval_metrics["mse_per_output"],
        eval_metrics["mse"],
        {
            "mse": eval_metrics["mse"],
            "rmse": eval_metrics["rmse"],
            "r2": eval_metrics["r2"],
            "nrmse": eval_metrics["nrmse"],
            "nmse": eval_metrics["nmse"],
            "accuracy_tol": eval_metrics["accuracy_tol"],
            "accuracy_tol_max": eval_metrics["accuracy_tol_max"],
        },
    )


# ================================================================================
# Reusable IterState
# ================================================================================


@dataclass
class IterState:
    terms_before: List[str]
    terms_after: List[str]
    keep: List[str]
    drop: List[str]
    mse_before: List[float]  # per-output
    mse_before_total: float  # total
    mse_after: List[float]  # per-output
    mse_after_total: float  # total
    metrics_before: Dict[str, float]  # per-output
    metrics_after: Dict[str, float]  # per-output
    metrics_test_before: Dict[str, float]  # per-output
    metrics_test_after: Dict[str, float]  # per-output
    history: List[Dict[str, Any]]
    equation_before: str
    equation_after: str
    iter_num: Optional[str] = None
    node_id: Optional[str] = None
    history_id: Optional[str] = None
    metrics_ood_test_before: Optional[Dict[str, float]] = None
    metrics_ood_test_after: Optional[Dict[str, float]] = None
    # Optional, observational-only metadata.  Search/reward/pruning never read
    # this field in diagnostic mode.
    sobolev_diagnostics: Optional[Dict[str, Any]] = None
    # Populated only for a full sibling batch in ``sibling_rerank`` or
    # ``pareto_crossfit`` mode.  It records the frozen train-only selection
    # evidence used by SN-IGSR.
    # Test/OOD metrics are never inputs to this record.
    sobolev_sibling_ranking: Optional[Dict[str, Any]] = None
    # Validation-guarded term compression audit for ``prune_refit`` mode.
    # It contains search train/validation information only.
    sobolev_pruning: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iter_num": self.iter_num,
            "node_id": self.node_id,
            "terms_before": self.terms_before,
            "terms_after": self.terms_after,
            "keep": self.keep,
            "drop": self.drop,
            "mse_before": self.mse_before,
            "mse_before_total": self.mse_before_total,
            "mse_after": self.mse_after,
            "mse_after_total": self.mse_after_total,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "metrics_test_before": self.metrics_test_before,
            "metrics_test_after": self.metrics_test_after,
            "metrics_ood_test_before": self.metrics_ood_test_before,
            "metrics_ood_test_after": self.metrics_ood_test_after,
            "history": self.history,
            "equation_before": self.equation_before,
            "equation_after": self.equation_after,
            "sobolev_diagnostics": self.sobolev_diagnostics,
            "sobolev_sibling_ranking": self.sobolev_sibling_ranking,
            "sobolev_pruning": self.sobolev_pruning,
        }

    @staticmethod
    def get_empty_state(node_id: Optional[str] = None, iter_num: Optional[str] = None) -> "IterState":
        if node_id is None and iter_num is None:
            raise ValueError("Either node_id or iter_num must be provided.")
        return IterState(
            terms_before=[],
            terms_after=[],
            keep=[],
            drop=[],
            mse_before=[],
            mse_before_total=float("inf"),
            mse_after=[],
            mse_after_total=float("inf"),
            metrics_before={},
            metrics_after={},
            metrics_test_before={},
            metrics_test_after={},
            metrics_ood_test_before=None,
            metrics_ood_test_after=None,
            history=[],
            equation_before="",
            equation_after="",
            sobolev_diagnostics=None,
            sobolev_sibling_ranking=None,
            sobolev_pruning=None,
            node_id=node_id,
            iter_num=iter_num,
        )


# ================================================================================
# Prepared Dataset
# ================================================================================


@dataclass
class PreparedDataset:
    data: Dict[str, np.ndarray]
    y: Dict[str, np.ndarray]
    data_validation: Dict[str, np.ndarray]
    y_validation: Dict[str, np.ndarray]
    data_test: Dict[str, np.ndarray]
    y_test: Dict[str, np.ndarray]
    y_ood_test: Optional[Dict[str, np.ndarray]] = None
    data_ood_test: Optional[Dict[str, np.ndarray]] = None


def prepare_dataset(
    data_bundle: DataBundle,
    exp_logger: structlog.stdlib.BoundLogger,
) -> PreparedDataset:
    """
    Prepare the dataset into the `data` Dict[str, np.ndarray] and `y` Dict[str, np.ndarray] format for each split.
    The str keys are the feature/target names, and the values are the numpy arrays of the feature/target values for each split.

    Args:
        data_bundle (DataBundle): The dataset to prepare.
        exp_logger (structlog.stdlib.BoundLogger): The logger for the experiment.

    Returns:
        PreparedDataset: A dataclass containing the prepared dataset.
    """
    target_columns = data_bundle.target_columns

    targets_train = data_bundle.dataset_train[target_columns].to_numpy()
    targets_validation = data_bundle.dataset_validation[target_columns].to_numpy()
    targets_test = data_bundle.dataset_test[target_columns].to_numpy()
    if data_bundle.dataset_ood_test is not None:
        targets_ood_test = data_bundle.dataset_ood_test[target_columns].to_numpy()
    else:
        targets_ood_test = None

    features_names = [c for c in data_bundle.dataset_train.columns if c not in target_columns]
    features_train = data_bundle.dataset_train[features_names].to_numpy()
    features_validation = data_bundle.dataset_validation[features_names].to_numpy()
    features_test = data_bundle.dataset_test[features_names].to_numpy()
    if data_bundle.dataset_ood_test is not None:
        features_ood_test = data_bundle.dataset_ood_test[features_names].to_numpy()
    else:
        features_ood_test = None

    data = {name: features_train[:, i] for i, name in enumerate(features_names)}
    data_validation = {name: features_validation[:, i] for i, name in enumerate(features_names)}
    data_test = {name: features_test[:, i] for i, name in enumerate(features_names)}
    if data_bundle.dataset_ood_test is not None:
        data_ood_test = {name: features_ood_test[:, i] for i, name in enumerate(features_names)}
    else:
        data_ood_test = None

    y = {name: targets_train[:, i] for i, name in enumerate(target_columns)}
    y_validation = {name: targets_validation[:, i] for i, name in enumerate(target_columns)}
    y_test = {name: targets_test[:, i] for i, name in enumerate(target_columns)}
    if data_bundle.dataset_ood_test is not None:
        y_ood_test = {name: targets_ood_test[:, i] for i, name in enumerate(target_columns)}
    else:
        y_ood_test = None

    return PreparedDataset(
        data=data,
        y=y,
        data_validation=data_validation,
        y_validation=y_validation,
        data_test=data_test,
        y_test=y_test,
        y_ood_test=y_ood_test,
        data_ood_test=data_ood_test,
    )


def compute_raw_feature_importance(
    prepared_dataset: PreparedDataset,
    max_features: Optional[int] = 20,
) -> List[Tuple[str, float]]:
    """
    Compute per-raw-feature importance via standardized Ridge regression.

    Steps:
    1. Standardize training features to zero mean, unit variance.
    2. Fit Ridge(alpha=1.0) on standardized features -> targets.
    3. Importance = mean(|coef_|) across targets (standardized => comparable).

    Returns a list of (feature_name, importance) tuples, sorted descending,
    truncated to max_features (or all if max_features is None).
    """
    feature_names = list(prepared_dataset.data.keys())
    X = np.column_stack([prepared_dataset.data[f] for f in feature_names])
    Y = np.column_stack(list(prepared_dataset.y.values()))

    # Standardize features (handle zero-variance gracefully)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # zero-variance features get importance 0 naturally
    X_std = (X - mean) / std

    reg = Ridge(alpha=1.0, fit_intercept=False).fit(X_std, Y)
    coefs = np.atleast_2d(reg.coef_)  # (n_targets, n_features)
    importance = np.mean(np.abs(coefs), axis=0)  # (n_features,)

    ranked = sorted(zip(feature_names, importance.tolist()), key=lambda x: -x[1])
    if max_features is not None:
        ranked = ranked[:max_features]
    return ranked


# ================================================================================
# History and Tracking
# ================================================================================


def add_to_history(
    history: List[Dict[str, Any]],
    iter_id: Any,
    iter_result: IterState,
    exp_logger: structlog.stdlib.BoundLogger,
    seed: int,
) -> List[Dict[str, Any]]:
    item = [
        {
            "round": iter_id,
            "keep": iter_result.keep,
            "drop": iter_result.drop,
            "MSE_before (per-output)": iter_result.mse_before,
            "MSE_before (total)": iter_result.mse_before_total,
            "MSE_after (per-output)": iter_result.mse_after,
            "MSE_after (total)": iter_result.mse_after_total,
            "equation_before": iter_result.equation_before,
            "equation_after": iter_result.equation_after,
        }
    ]
    exp_logger.info(
        "adding_history_item",
        seed=seed,
        iter_num=iter_id,
        history=item,
    )
    return history + item


def check_token_budget_and_maybe_break(
    cfg: MainConfig,
    round: Any,
    compute_profiler: ComputeProfiler,
    seed: int,
    logger: structlog.stdlib.BoundLogger,
    offset: int = 0,
) -> bool:
    """Return True and log if token budget exceeded; else False."""
    if not bool(cfg.experiment.break_on_tokens) or cfg.experiment.break_above_N_tokens is None:
        return False
    prof = compute_profiler.as_dict()
    tokens_tot = prof.get("input_tokens", 0.0) + prof.get("output_tokens", 0.0) - offset
    print(">>> Elapsed tokens total:", tokens_tot)
    if tokens_tot > float(cfg.experiment.break_above_N_tokens):
        logger.info(
            "token_budget_stop",
            seed=seed,
            round=round,
            offset=offset,
            tokens_total=float(tokens_tot),
            token_budget=float(cfg.experiment.break_above_N_tokens),
        )
        return True
    return False


def check_wallclock_budget_and_maybe_break(
    cfg: MainConfig, round: Any, start_time: float, seed: int, logger: structlog.stdlib.BoundLogger
) -> bool:
    """Return True and log if wall clock budget exceeded; else False."""
    if not bool(cfg.experiment.break_on_wallclock) or cfg.experiment.break_above_wallclock is None:
        return False
    wall_clock_total = time.time() - start_time
    print(">>> Elapsed wall clock total:", wall_clock_total)
    if wall_clock_total > float(cfg.experiment.break_above_wallclock):
        logger.info(
            "wallclock_budget_stop",
            seed=seed,
            round=round,
            wall_clock_total=float(wall_clock_total),
            wallclock_budget=float(cfg.experiment.break_above_wallclock),
        )
        return True
    return False


def check_early_stop(
    metric_name: str,
    direction: Literal["min", "max"],
    best_result: float,
    iter_result: IterState,
    patience_left: int,
    patience: int,
) -> int:
    if direction == "min":
        if best_result < iter_result.metrics_after[metric_name]:
            patience_left -= 1
        else:
            patience_left = patience
    elif direction == "max":
        if best_result > iter_result.metrics_after[metric_name]:
            patience_left -= 1
        else:
            patience_left = patience
    else:
        raise ValueError(f"Invalid direction: {direction}")
    return patience_left


def get_iter_or_node_id(cfg: MainConfig, iter_num: Optional[int] = None, node_id: Optional[str] = None) -> str:
    if cfg.experiment.tree:
        if node_id is None:
            raise ValueError("Node ID is required when tree is enabled.")
        return node_id
    else:
        if iter_num is None:
            raise ValueError("Iteration number is required when tree is disabled.")
        return str(iter_num)


def get_total_iters_for_print(cfg: MainConfig) -> int:
    if cfg.experiment.tree:
        if hasattr(cfg.experiment, "total_budget") and cfg.experiment.total_budget is not None:
            return cfg.experiment.total_budget
        else:
            raise ValueError("No total budget provided.")
    else:
        if hasattr(cfg.experiment, "n_iters") and cfg.experiment.n_iters is not None:
            return cfg.experiment.n_iters
        else:
            raise ValueError("No total budget or number of iterations provided.")


# ================================================================================
# Equation pretty-printing
# ================================================================================


def pretty_equation(
    w: np.ndarray,
    terms: List[str],
    *,
    y_var: str = "y",
    precision: int = 4,
    intercept: Optional[float] = None,
    skip_zeros: bool = True,
    zero_tol: float = 1e-10,
    omit_one: bool = True,
) -> str:
    """
    Return a human-readable equation string, e.g.
    ```
    y1 = 81.144 x1 - 0.8023 x2 + 0.04001 x1 * x2**2
    ```

    Args:
        w (1-D array-like of floats): Coefficients (same order as `terms`).
        terms (list[str]): Term names (strings you used in the design matrix).
        y_var (str, default "y"): Name of the dependent variable.
        precision (int, default 4): Significant digits for coefficients.
        intercept (float | None): Intercept term (if your model had one).  If None, no intercept shown.
        skip_zeros (bool, default True): Skip coefficients with |w| < zero_tol.
        zero_tol (float, default 1e-10): Threshold for “effectively zero”.
        omit_one (bool, default True): Print “− x1” instead of “− 1.000 x1” when |w|≈1.

    Returns:
        str: A human-readable equation string.
    """

    w = np.asarray(w).ravel()
    if len(w) != len(terms):
        raise ValueError("w and terms must have same length")

    parts = []

    # Optional intercept (subject to same skip_zeros logic as coefficients):
    if intercept is not None and not (skip_zeros and abs(intercept) < zero_tol):
        parts.append(f"{intercept:.{precision}g}")

    for coeff, term in zip(w, terms):
        if skip_zeros and abs(coeff) < zero_tol:
            continue

        sign = " + " if coeff >= 0 else " − "
        mag = abs(coeff)

        if omit_one and np.isclose(mag, 1.0, atol=10**-precision):
            coef_str = ""  # omit the '1'
        else:
            coef_str = f"{mag:.{precision}g} "

        parts.append(f"{sign}{coef_str}{term}")

    # If everything was skipped, equation is zero:
    rhs = "".join(parts) if parts else "0"
    # Replace leading " + " / " − " if intercept is None:
    rhs = rhs.lstrip(" +").replace(" − ", " − ", 1)

    return f"{y_var} = {rhs}"


def pretty_equations(
    w: np.ndarray,
    terms: List[str],
    y: Dict[str, np.ndarray],
    precision: int = 4,
    intercept: Optional[Union[float, np.ndarray]] = None,
    skip_zeros: bool = True,
    zero_tol: float = 1e-10,
    omit_one: bool = True,
) -> str:
    """
    Return a human-readable equation**s** string, e.g.
    ```
    y1 = 1.5 + 81.144 x1 - 0.8023 x2 + 0.04001 x1 * x2**2
    y2 = -0.8 + 0.001 x1 + 0.002 x2
    ```

    See `pretty_equation` for more details.

    Args:
        intercept: None (omit), a scalar (same for all targets), or
                   an array of shape (n_targets,) for per-target intercepts.
    """
    w = _as_2d(w)
    n_targets = len(y)
    # Normalize intercept to a list of per-target values (or Nones).
    if intercept is None:
        intercepts = [None] * n_targets
    elif np.ndim(intercept) == 0:
        intercepts = [float(intercept)] * n_targets
    else:
        intercept_arr = np.asarray(intercept).ravel()
        intercepts = [float(intercept_arr[i]) for i in range(n_targets)]
    output = ""
    for i, (y_name, y_val) in enumerate(y.items()):
        output += (
            pretty_equation(
                w[:, i],
                terms,
                y_var=y_name,
                precision=precision,
                intercept=intercepts[i],
                skip_zeros=skip_zeros,
                zero_tol=zero_tol,
                omit_one=omit_one,
            )
        ) + "\n"
    return output


# ================================================================================
# Postprocessors for Agents (Output Validation / Parsing)
# ================================================================================


def postprocessor_generate_terms(agent, *args, **kwargs) -> PostprocessorOutput:
    data_bundle: DataBundle = kwargs["data_bundle"]  # noqa: F841
    cfg: MainConfig = kwargs["cfg"]

    iter_num: int = kwargs.get("iter_num", None)
    node_id: int = kwargs.get("node_id", None)
    iter_num = get_iter_or_node_id(cfg, iter_num, node_id)
    total_iters = get_total_iters_for_print(cfg)

    seed: int = kwargs["seed"]
    stage_name: str = kwargs.get("stage_name", "N/A")

    terms_sentinel = kwargs["terms_sentinel"]

    prepared_dataset: PreparedDataset = kwargs["prepared_dataset"]
    data = prepared_dataset.data
    data_validation = prepared_dataset.data_validation

    correct_format_generate_terms = kwargs["correct_format_generate_terms"]

    last_response = agent.conversation_history[-1]["content"]

    print(
        f"Seed {seed} - Iteration {iter_num}/{total_iters} - Stage {stage_name} - " f"Last response:\n{last_response}"
    )
    if terms_sentinel not in last_response:
        return PostprocessorOutput(
            success=False,
            inform_agent=True,
            feedback=f"{terms_sentinel} not found in last response.\n{correct_format_generate_terms}",
        )

    # Extract the terms block:
    terms_blocks = extract_code_blocks(last_response, sentinel=f"{terms_sentinel}\n", return_info_string=False)
    print(f"Seed {seed} - Iteration {iter_num}/{total_iters} - Stage {stage_name} - " f"Terms blocks:\n{terms_blocks}")
    if len(terms_blocks) != 1:
        return PostprocessorOutput(
            success=False,
            inform_agent=True,
            feedback=f"Output error. Expected exactly one terms list block, got {len(terms_blocks)}.\n{correct_format_generate_terms}",
        )
    terms = terms_blocks[0]

    terms = [ln.strip() for ln in terms.splitlines() if ln.strip()]
    print(f"Terms:\n{terms}")

    # Validate the full matrix that will enter OLS.  ``current_terms`` matters:
    # accepting each new term in isolation is insufficient when the combined
    # float64 Gram/cross-product is not executable.
    current_terms = list(kwargs.get("current_terms", []))
    candidate_terms = list(dict.fromkeys(current_terms + terms))
    current_term_set = set(current_terms)
    policy_terms = [
        term for term in dict.fromkeys(terms)
        if term not in current_term_set
    ]

    # Make sure the terms can be evaluated.
    try:
        # Create design matrix and check for invalid values.
        dm = DesignMatrix.from_terms(candidate_terms, data)
        dm_val = DesignMatrix.from_terms(candidate_terms, data_validation)

        if not candidate_terms:
            dm.errors.append("At least one nonempty term is required.")

        if not dm.errors and dm.term_names == candidate_terms:
            dm.errors.extend(
                _search_matrix_numerical_errors(dm, prepared_dataset.y, split_name="train")
            )
            dm.errors.extend(
                _candidate_policy_errors(
                    policy_terms,
                    data,
                    prepared_dataset.y,
                    dm,
                    cfg,
                    split_name="train",
                )
            )
        if not dm_val.errors and dm_val.term_names == candidate_terms:
            dm_val.errors.extend(
                _search_matrix_numerical_errors(
                    dm_val,
                    prepared_dataset.y_validation,
                    split_name="validation",
                )
            )
            dm_val.errors.extend(
                _candidate_policy_errors(
                    policy_terms,
                    data_validation,
                    prepared_dataset.y_validation,
                    dm_val,
                    cfg,
                    split_name="validation",
                )
            )

        if cfg.experiment.additional_validation_of_terms:
            # For each, check for NaN and infinity values.
            # Candidate generation is intentionally restricted to train and
            # validation.  Held-out report features must not act as a domain
            # gate for node generation or pruning.
            for dm_obj in (dm, dm_val):
                for i, term in enumerate(dm_obj.term_names):
                    if np.isnan(dm_obj.phi[:, i]).any():
                        dm_obj.errors.append(f"Term {term} contains NaN values.")
                    if np.isinf(dm_obj.phi[:, i]).any():
                        dm_obj.errors.append(f"Term {term} contains infinity values.")
                # Check for extreme values.
                # 1. Apply standard scaler to .phi and identify columns that contain nans.
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler().fit(dm_obj.phi)
                temp = scaler.transform(dm_obj.phi)
                # 2. Identify columns that contain nans.
                for j in range(temp.shape[1]):
                    if np.isnan(temp[:, j]).any():
                        dm_obj.errors.append(
                            f"Term {dm_obj.term_names[j]} has features with extremely large magnitude values. "
                            "This will lead to numerical instability and is not allowed."
                        )

        if dm.errors or dm_val.errors:
            errors = "\n".join(dm.errors + dm_val.errors)
            return PostprocessorOutput(
                success=False,
                inform_agent=True,
                feedback=(
                    f"Output error. Failed to parse or evaluate terms:\n{errors}\n"
                    f"{correct_format_generate_terms}"
                ),
            )
    except Exception as e:
        return PostprocessorOutput(
            success=False,
            inform_agent=True,
            feedback=(
                f"Output error. Failed to parse or evaluate terms:\n{e}\n"
                f"{correct_format_generate_terms}"
            ),
        )

    # Check for invalid values.
    for i, term in enumerate(candidate_terms):
        try:
            check_array(dm.phi[:, [i]])
            check_array(dm_val.phi[:, [i]])
        except Exception as e:
            return PostprocessorOutput(
                success=False,
                inform_agent=True,
                feedback=(
                    f"Output error. Evaluated terms contain invalid values for term: {term}\n"
                    f"Error:\n{e}\n{correct_format_generate_terms}"
                ),
            )

    return PostprocessorOutput(
        success=True,
        inform_agent=False,
        feedback=None,
        parsed_output=terms,
    )


def postprocessor_pruning(agent, *args, **kwargs) -> PostprocessorOutput:
    data_bundle: DataBundle = kwargs["data_bundle"]  # noqa: F841
    cfg: MainConfig = kwargs["cfg"]

    iter_num: int = kwargs.get("iter_num", None)
    node_id: int = kwargs.get("node_id", None)
    iter_num = get_iter_or_node_id(cfg, iter_num, node_id)
    total_iters = get_total_iters_for_print(cfg)

    seed: int = kwargs["seed"]
    stage_name: str = kwargs.get("stage_name", "N/A")

    # data: Dict[str, np.ndarray] = kwargs["data"]

    pruning_sentinel = kwargs["pruning_sentinel"]
    # dfs: Dict[str, pd.DataFrame] = kwargs["dfs"]
    # mse: List[float] = kwargs["mse"]
    # mse_total: float = kwargs["mse_total"]
    # history: List[Dict[str, Any]] = kwargs["history"]
    current_terms: List[str] = kwargs["current_terms"]

    correct_format_pruning = kwargs["correct_format_pruning"]

    last_response = agent.conversation_history[-1]["content"]

    print(
        f"Seed {seed} - Iteration {iter_num}/{total_iters} - Stage {stage_name} - " f"Last response:\n{last_response}"
    )
    if pruning_sentinel not in last_response:
        return PostprocessorOutput(
            success=False,
            inform_agent=True,
            feedback=f"{pruning_sentinel} not found in last response.\n{correct_format_pruning}",
        )

    # Extract the decision block:
    decision_blocks = extract_code_blocks(last_response, sentinel=f"{pruning_sentinel}\n", return_info_string=False)
    print(
        f"Seed {seed} - Iteration {iter_num}/{total_iters} - Stage {stage_name} - "
        f"Decision blocks:\n{decision_blocks}"
    )
    if len(decision_blocks) != 1:
        return PostprocessorOutput(
            success=False,
            inform_agent=True,
            feedback=f"Output error. Expected exactly one decision block, got {len(decision_blocks)}.\n{correct_format_pruning}",
        )
    decision = decision_blocks[0]

    # Parse the decision:
    try:
        decision = ast.literal_eval(decision)
    except Exception as e:
        return PostprocessorOutput(
            success=False,
            inform_agent=True,
            feedback=f"Output error. Failed to parse decision as a python literal:\n{e}\n{correct_format_pruning}",
        )
    if "keep" not in decision or "drop" not in decision:
        return PostprocessorOutput(
            success=False,
            inform_agent=True,
            feedback=f"Output error. Decision must contain 'keep' and 'drop' keys.\n{correct_format_pruning}",
        )
    keep = decision["keep"]
    drop = decision["drop"]

    # Check for invalid values.
    invalid_terms = []
    for term in keep + drop:
        if term not in current_terms:
            invalid_terms.append(term)
    if invalid_terms:
        return PostprocessorOutput(
            success=False,
            inform_agent=True,
            feedback=f"Output error. Terms not found in the current set of terms:\n{invalid_terms}",
        )

    return PostprocessorOutput(success=True, inform_agent=False, feedback=None, parsed_output=(keep, drop))
