"""Deterministic SymPy parsing and structure-preserving ``expand_mul`` terms."""

from __future__ import annotations

import ast
from itertools import product
from typing import Mapping, Sequence

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
    numeric coefficients under the report's decomposition definition.  A
    zero-argument lambdified evaluation handles them without simplifying or
    changing the candidate's symbolic structure.
    """

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
            },
            "numpy",
        ],
        cse=True,
    )
    value = complex(function())
    if not (sp.Float(value.real).is_finite and sp.Float(value.imag).is_finite):
        raise ValueError(f"Non-finite numeric coefficient {expression}")
    return value


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
