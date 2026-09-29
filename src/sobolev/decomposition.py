"""Deterministic SymPy parsing and structure-preserving ``expand_mul`` terms."""

from __future__ import annotations

import ast
from itertools import product
from typing import Mapping, Sequence

import numpy as np
import sympy as sp
from sympy.printing.str import StrPrinter
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .types import TermSpec


PROTECTED_EPSILON = sp.Float("1e-6")
DEEP_CONSTANT_NODE_LIMIT = 32
DEEP_CONSTANT_DEPTH_LIMIT = 4
DEFERRED_SQRT_POWER_BASE_DEPTH = 12


class ProjectExpressionPrinter(StrPrinter):
    """Serialize protected SymPy expressions back to nd2py-compatible text."""

    def _print_asin(self, expression: sp.Expr) -> str:
        return f"arcsin({self._print(expression.args[0])})"

    def _print_acos(self, expression: sp.Expr) -> str:
        return f"arccos({self._print(expression.args[0])})"

    def _print_atan(self, expression: sp.Expr) -> str:
        return f"arctan({self._print(expression.args[0])})"

    def _print_ProtectedDivision(self, expression: sp.Expr) -> str:
        numerator, denominator = expression.args
        return f"(({self._print(numerator)}) / ({self._print(denominator)}))"

    def _print_ProtectedInverse(self, expression: sp.Expr) -> str:
        return f"(1 / ({self._print(expression.args[0])}))"

    def _print_ProtectedLog(self, expression: sp.Expr) -> str:
        return f"log({self._print(expression.args[0])})"

    def _print_ProtectedLogAbs(self, expression: sp.Expr) -> str:
        return f"logabs({self._print(expression.args[0])})"

    def _print_Abs(self, expression: sp.Expr) -> str:
        return f"abs({self._print(expression.args[0])})"

    def _print_DeferredSqrt(self, expression: sp.Expr) -> str:
        return f"sqrt({self._print(expression.args[0])})"


def to_project_expression_string(expression: sp.Expr) -> str:
    """Return text accepted by nd2py while retaining protected operations."""

    return ProjectExpressionPrinter().doprint(expression)


class ProtectedDivision(sp.Function):
    """nd2py/EIC division, guarded only when the denominator is exactly zero."""

    nargs = 2

    def fdiff(self, argindex: int = 1) -> sp.Expr:
        numerator, denominator = self.args
        if argindex == 1:
            return sp.Piecewise((0, sp.Eq(denominator, 0)), (1 / denominator, True))
        if argindex == 2:
            return sp.Piecewise((0, sp.Eq(denominator, 0)), (-numerator / denominator**2, True))
        raise sp.ArgumentIndexError(self, argindex)


class ProtectedInverse(sp.Function):
    """Guarded reciprocal used after distributing a protected numerator."""

    nargs = 1

    def fdiff(self, argindex: int = 1) -> sp.Expr:
        if argindex != 1:
            raise sp.ArgumentIndexError(self, argindex)
        denominator = self.args[0]
        return sp.Piecewise((0, sp.Eq(denominator, 0)), (-1 / denominator**2, True))


class ProtectedLog(sp.Function):
    """nd2py/EIC log with an exact-zero guard and exact symbolic derivative."""

    nargs = 1

    def fdiff(self, argindex: int = 1) -> sp.Expr:
        if argindex != 1:
            raise sp.ArgumentIndexError(self, argindex)
        argument = self.args[0]
        return sp.Piecewise((0, sp.Eq(argument, 0)), (1 / argument, True))


class ProtectedLogAbs(sp.Function):
    """Guarded ``log(abs(x))`` used by nd2py's LogAbs operator."""

    nargs = 1

    def fdiff(self, argindex: int = 1) -> sp.Expr:
        if argindex != 1:
            raise sp.ArgumentIndexError(self, argindex)
        argument = self.args[0]
        return sp.Piecewise((0, sp.Eq(argument, 0)), (1 / argument, True))


class DeferredSqrt(sp.Function):
    """Square root whose construction defers SymPy's eager power simplifier."""

    nargs = 1

    def fdiff(self, argindex: int = 1) -> sp.Expr:
        if argindex != 1:
            raise sp.ArgumentIndexError(self, argindex)
        return 1 / (2 * DeferredSqrt(self.args[0]))


class _DivisionTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="pdiv", ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )
        return node


class _DeepConstantFolder(ast.NodeTransformer):
    """Evaluate only deep feature-free AST pieces before SymPy construction."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = frozenset(str(value) for value in feature_names)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        return self._fold(self.generic_visit(node))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        return self._fold(self.generic_visit(node))

    def visit_Call(self, node: ast.Call) -> ast.AST:
        return self._fold(self.generic_visit(node))

    def _fold(self, node: ast.AST) -> ast.AST:
        if any(
            isinstance(value, ast.Name) and value.id in self.feature_names
            for value in ast.walk(node)
        ):
            return node
        if (
            sum(1 for _ in ast.walk(node)) <= DEEP_CONSTANT_NODE_LIMIT
            and _python_ast_depth(node) <= DEEP_CONSTANT_DEPTH_LIMIT
        ):
            return node
        try:
            value = _evaluate_python_numeric_ast(node)
        except (TypeError, ValueError, KeyError):
            return node
        return ast.copy_location(ast.Constant(value=_real_ast_literal(value)), node)


class _DeepSqrtPowerTransformer(ast.NodeTransformer):
    """Defer only the confirmed pathological ``sqrt(deep_tree ** p)`` case."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.BinOp)
            and isinstance(node.args[0].op, ast.Pow)
            and _python_ast_depth(node.args[0].left) > DEFERRED_SQRT_POWER_BASE_DEPTH
        ):
            node.func.id = "deferred_sqrt"
        return node


def _python_ast_depth(node: ast.AST) -> int:
    children = tuple(ast.iter_child_nodes(node))
    return 1 + max((_python_ast_depth(child) for child in children), default=0)


def _real_ast_literal(value: complex | float) -> float:
    numeric = complex(value)
    if not np.isfinite(numeric.imag) or abs(numeric.imag) > 1e-12:
        return float("nan")
    return float(numeric.real)


def _evaluate_python_numeric_ast(node: ast.AST) -> complex | float:
    epsilon = float(PROTECTED_EPSILON)

    def evaluate(current: ast.AST) -> complex | float:
        if isinstance(current, ast.Constant) and isinstance(
            current.value, (int, float)
        ):
            return float(current.value)
        if isinstance(current, ast.Name):
            constants = {"pi": np.pi, "E": np.e, "e": np.e}
            if current.id not in constants:
                raise ValueError(f"Unsupported constant name {current.id}")
            return float(constants[current.id])
        if isinstance(current, ast.UnaryOp):
            operand = evaluate(current.operand)
            if isinstance(current.op, ast.UAdd):
                return operand
            if isinstance(current.op, ast.USub):
                return -operand
            raise ValueError("Unsupported constant unary operator")
        if isinstance(current, ast.BinOp):
            left = evaluate(current.left)
            right = evaluate(current.right)
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.Mult):
                return left * right
            if isinstance(current.op, ast.Div):
                return left / (right + epsilon * (right == 0))
            if isinstance(current.op, ast.Pow):
                return np.power(left, right)
            raise ValueError("Unsupported constant binary operator")
        if not (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Name)
            and not current.keywords
        ):
            raise ValueError("Unsupported constant AST node")
        arguments = tuple(evaluate(argument) for argument in current.args)
        name = current.func.id
        unary = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "arcsin": np.arcsin,
            "acos": np.arccos,
            "arccos": np.arccos,
            "atan": np.arctan,
            "arctan": np.arctan,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "Abs": np.abs,
        }
        if name in unary and len(arguments) == 1:
            return unary[name](arguments[0])
        if name in {"log", "ln"} and len(arguments) == 1:
            argument = arguments[0]
            return np.log(argument + epsilon * (argument == 0))
        if name == "logabs" and len(arguments) == 1:
            argument = arguments[0]
            return np.log(abs(argument + epsilon * (argument == 0)))
        if name == "sqrtabs" and len(arguments) == 1:
            return np.sqrt(abs(arguments[0]))
        if name in {"inv", "pinv"} and len(arguments) == 1:
            denominator = arguments[0]
            return 1.0 / (denominator + epsilon * (denominator == 0))
        if name == "pdiv" and len(arguments) == 2:
            return arguments[0] / (
                arguments[1] + epsilon * (arguments[1] == 0)
            )
        if name == "pow" and len(arguments) == 2:
            return np.power(arguments[0], arguments[1])
        if name == "pow2" and len(arguments) == 1:
            return np.power(arguments[0], 2)
        if name == "pow3" and len(arguments) == 1:
            return np.power(arguments[0], 3)
        if name == "cot" and len(arguments) == 1:
            return np.divide(1.0, np.tan(arguments[0]))
        if name == "sec" and len(arguments) == 1:
            return np.divide(1.0, np.cos(arguments[0]))
        if name == "csc" and len(arguments) == 1:
            return np.divide(1.0, np.sin(arguments[0]))
        if name == "sech" and len(arguments) == 1:
            return np.divide(1.0, np.cosh(arguments[0]))
        if name == "csch" and len(arguments) == 1:
            return np.divide(1.0, np.sinh(arguments[0]))
        if name == "max" and arguments:
            return max(arguments)
        if name == "min" and arguments:
            return min(arguments)
        raise ValueError(f"Unsupported constant function {name}")

    with np.errstate(all="ignore"):
        return evaluate(node)


def expression_to_string(expression: object) -> str:
    """Convert a string, SymPy object, or nd2py-like object without importing MCTS."""

    if isinstance(expression, str):
        return expression
    if isinstance(expression, sp.Expr):
        return str(expression)
    to_str = getattr(expression, "to_str", None)
    if callable(to_str):
        return str(to_str())
    return str(expression)


def parse_expression(
    expression: object,
    feature_names: Sequence[str],
) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    """Parse an expression with the project's protected operator semantics."""

    clean_names = tuple(str(name) for name in feature_names)
    if not clean_names or len(set(clean_names)) != len(clean_names):
        raise ValueError(f"Feature names must be non-empty and unique: {clean_names}")
    symbols = tuple(sp.Symbol(name, real=True) for name in clean_names)
    if isinstance(expression, sp.Expr):
        parsed = expression
    else:
        text = expression_to_string(expression).strip().replace("^", "**").replace("−", "-")
        if not text:
            raise ValueError("Expression is empty")
        tree = _DivisionTransformer().visit(ast.parse(text, mode="eval"))
        aliases = set(clean_names)
        if symbols:
            aliases.update(("x", "X"))
        for index in range(len(symbols)):
            aliases.update(
                (f"x_{index}", f"X_{index}", f"x{index + 1}", f"X{index + 1}")
            )
        tree = _DeepConstantFolder(tuple(sorted(aliases))).visit(tree)
        tree = _DeepSqrtPowerTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        normalized = ast.unparse(tree)
        local = _function_locals()
        local.update(dict(zip(clean_names, symbols, strict=True)))
        if symbols:
            local.setdefault("x", symbols[0])
            local.setdefault("X", symbols[0])
        for index, symbol in enumerate(symbols):
            local.setdefault(f"x_{index}", symbol)
            local.setdefault(f"X_{index}", symbol)
            local.setdefault(f"x{index + 1}", symbol)
            local.setdefault(f"X{index + 1}", symbol)
        parsed = parse_expr(
            normalized,
            local_dict=local,
            transformations=standard_transformations
            + (convert_xor, function_exponentiation, implicit_multiplication_application),
            evaluate=False,
        )
    if not isinstance(parsed, sp.Expr):
        raise TypeError(f"Parser returned {type(parsed).__name__}, not a SymPy expression")
    unexpected = parsed.free_symbols.difference(symbols)
    if unexpected:
        raise ValueError(f"Unknown variables {sorted(map(str, unexpected))}; features={list(clean_names)}")
    return parsed, symbols


def decompose_expand_mul(expression: sp.Expr, symbols: Sequence[sp.Symbol]) -> list[TermSpec]:
    """Distribute multiplication over addition without expanding powers/functions.

    Unlike a global simplifier, the structural distributor deliberately keeps
    repeated additive terms separate. This makes ``x+x`` auditable as two
    exactly repeated terms while retaining the ``expand_mul`` semantics used by
    the final Sobolev report.
    """

    raw_terms = _distributed_terms(expression)
    raw_terms.sort(key=sp.default_sort_key)
    output: list[TermSpec] = []
    for term in raw_terms:
        coefficient_expr, basis = term.as_independent(*symbols, as_Add=False)
        if coefficient_expr.free_symbols:
            coefficient_expr, basis = sp.Integer(1), term
        coefficient_complex = _evaluate_numeric_coefficient(coefficient_expr)
        if abs(coefficient_complex.imag) > 1e-12:
            raise ValueError(f"Complex coefficient {coefficient_expr} in term {term}")
        coefficient = float(coefficient_complex.real)
        output.append(TermSpec(coefficient, basis, sp.srepr(basis)))
    if not output:
        raise ValueError("Expression has no additive terms")
    return output


def _evaluate_numeric_coefficient(expression: sp.Expr) -> complex:
    """Evaluate a symbol-free coefficient with the protected operator rules.

    SymPy deliberately leaves custom protected functions such as
    ``ProtectedInverse(tanh(2))`` unevaluated.  Such factors are nevertheless
    numeric coefficients under the report's decomposition definition.  Keep
    the historical lambdified path for ordinary GP constants, and walk only a
    pathologically large constant subtree iteratively.  This preserves the
    frozen Base trajectory while avoiding recursive SymPy substitutions on the
    exceptional trees that previously exhausted the hard guard.
    """

    if not _numeric_expression_requires_iterative(
        expression,
        node_limit=32,
        depth_limit=5,
    ):
        return _evaluate_numeric_coefficient_legacy(expression)

    return _evaluate_numeric_coefficient_iterative(expression)


def _evaluate_numeric_coefficient_legacy(expression: sp.Expr) -> complex:
    epsilon = float(PROTECTED_EPSILON)

    def protected_divide(numerator: complex, denominator: complex) -> complex:
        return numerator / (denominator + epsilon * (denominator == 0))

    function = sp.lambdify(
        (),
        expression,
        modules=[
            {
                "ProtectedDivision": protected_divide,
                "ProtectedInverse": lambda value: protected_divide(1.0, value),
                "ProtectedLog": lambda value: sp.log(value + epsilon * (value == 0)),
                "ProtectedLogAbs": lambda value: sp.log(abs(value + epsilon * (value == 0))),
                "DeferredSqrt": np.sqrt,
            },
            "numpy",
        ],
        cse=True,
    )
    value = complex(function())
    if not (sp.Float(value.real).is_finite and sp.Float(value.imag).is_finite):
        raise ValueError(f"Non-finite numeric coefficient {expression}")
    return value


def _numeric_expression_requires_iterative(
    expression: sp.Expr,
    *,
    node_limit: int,
    depth_limit: int,
) -> bool:
    seen: set[int] = set()
    stack = [(expression, 1)]
    depth_exceeded = False
    while stack:
        node, depth = stack.pop()
        node_key = id(node)
        if node_key in seen:
            continue
        seen.add(node_key)
        if len(seen) > node_limit:
            return True
        depth_exceeded = depth_exceeded or depth > depth_limit
        stack.extend((argument, depth + 1) for argument in node.args)
    return depth_exceeded


def _evaluate_numeric_coefficient_iterative(expression: sp.Expr) -> complex:
    """Evaluate an unusually deep or large symbol-free tree without recursion."""

    epsilon = float(PROTECTED_EPSILON)

    def protected_divide(numerator: complex, denominator: complex) -> complex:
        return numerator / (denominator + epsilon * (denominator == 0))

    values: dict[int, complex | float] = {}
    stack: list[tuple[sp.Expr, bool]] = [(expression, False)]
    with np.errstate(all="ignore"):
        while stack:
            node, ready = stack.pop()
            node_key = id(node)
            if node_key in values:
                continue
            arguments = node.args
            if not arguments:
                values[node_key] = _numeric_atom(node)
                continue
            if not ready:
                stack.append((node, True))
                stack.extend(
                    (argument, False)
                    for argument in reversed(arguments)
                    if id(argument) not in values
                )
                continue
            numeric_arguments = tuple(values[id(argument)] for argument in arguments)
            function = node.func
            if function is sp.Add:
                values[node_key] = sum(numeric_arguments, 0.0)
            elif function is sp.Mul:
                value: complex | float = 1.0
                for argument in numeric_arguments:
                    value *= argument
                values[node_key] = value
            elif function is sp.Pow:
                values[node_key] = np.power(*numeric_arguments)
            elif function is ProtectedDivision:
                values[node_key] = protected_divide(*numeric_arguments)
            elif function is ProtectedInverse:
                values[node_key] = protected_divide(1.0, numeric_arguments[0])
            elif function is ProtectedLog:
                argument = numeric_arguments[0]
                values[node_key] = np.log(argument + epsilon * (argument == 0))
            elif function is ProtectedLogAbs:
                argument = numeric_arguments[0]
                values[node_key] = np.log(abs(argument + epsilon * (argument == 0)))
            elif function is sp.Abs:
                values[node_key] = abs(numeric_arguments[0])
            elif function is DeferredSqrt:
                values[node_key] = np.sqrt(numeric_arguments[0])
            elif function is sp.Max:
                values[node_key] = max(numeric_arguments)
            elif function is sp.Min:
                values[node_key] = min(numeric_arguments)
            else:
                values[node_key] = _numeric_function(function, numeric_arguments)
    value = complex(values[id(expression)])
    if not (sp.Float(value.real).is_finite and sp.Float(value.imag).is_finite):
        raise ValueError(f"Non-finite numeric coefficient {expression}")
    return value


def _numeric_atom(expression: sp.Expr) -> complex | float:
    if expression is sp.I:
        return 1j
    if isinstance(expression, (sp.Integer, sp.Rational, sp.Float, sp.NumberSymbol)):
        return float(expression)
    raise ValueError(f"Non-numeric coefficient atom {expression}")


def _numeric_function(
    function: object,
    arguments: tuple[complex | float, ...],
) -> complex | float:
    unary = {
        sp.sin: np.sin,
        sp.cos: np.cos,
        sp.tan: np.tan,
        sp.asin: np.arcsin,
        sp.acos: np.arccos,
        sp.atan: np.arctan,
        sp.sinh: np.sinh,
        sp.cosh: np.cosh,
        sp.tanh: np.tanh,
        sp.exp: np.exp,
        sp.log: np.log,
    }
    evaluator = unary.get(function)
    if evaluator is not None:
        return evaluator(arguments[0])
    if function is sp.cot:
        return np.divide(1.0, np.tan(arguments[0]))
    if function is sp.sec:
        return np.divide(1.0, np.cos(arguments[0]))
    if function is sp.csc:
        return np.divide(1.0, np.sin(arguments[0]))
    raise ValueError(f"Unsupported numeric coefficient function {function}")


def _distributed_terms(expression: sp.Expr) -> list[sp.Expr]:
    if isinstance(expression, sp.Add):
        terms: list[sp.Expr] = []
        for argument in expression.args:
            terms.extend(_distributed_terms(argument))
        return terms
    if isinstance(expression, ProtectedDivision):
        numerator, denominator = expression.args
        return [sp.Mul(term, ProtectedInverse(denominator)) for term in _distributed_terms(numerator)]
    if isinstance(expression, sp.Mul):
        factor_terms = [_distributed_terms(argument) for argument in expression.args]
        return [sp.Mul(*combination) for combination in product(*factor_terms)]
    # Powers and every function call are barriers by construction.
    return [expression]


def _function_locals() -> dict[str, object]:
    return {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "cot": sp.cot,
        "sec": sp.sec,
        "csc": sp.csc,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "sech": sp.sech,
        "csch": sp.csch,
        "exp": sp.exp,
        "log": ProtectedLog,
        "ln": ProtectedLog,
        "logabs": ProtectedLogAbs,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "sqrt": sp.sqrt,
        "deferred_sqrt": DeferredSqrt,
        "sqrtabs": lambda value: sp.sqrt(sp.Abs(value)),
        "inv": ProtectedInverse,
        "pdiv": ProtectedDivision,
        "pinv": ProtectedInverse,
        "pow": sp.Pow,
        "pow2": lambda value: sp.Pow(value, 2),
        "pow3": lambda value: sp.Pow(value, 3),
        "max": sp.Max,
        "min": sp.Min,
        "pi": sp.pi,
        "E": sp.E,
    }
