"""Representation primitives for a native additive-basis GP backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sympy as sp

from ...nd2py import nd2py as nd
from ...sobolev.decomposition import (
    decompose_expand_mul,
    parse_expression,
    to_project_expression_string,
)


@dataclass(frozen=True)
class BasisGene:
    canonical: str
    expression: str
    opaque: bool = False


@dataclass(frozen=True)
class BasisGenome:
    terms: tuple[BasisGene, ...]

    @property
    def canonicals(self) -> tuple[str, ...]:
        return tuple(term.canonical for term in self.terms)

    @property
    def opaque_count(self) -> int:
        return sum(term.opaque for term in self.terms)

    @property
    def is_exact(self) -> bool:
        return self.opaque_count == 0


@dataclass(frozen=True)
class BasisCrossoverResult:
    genome: BasisGenome
    receiver_genes: int
    donor_genes: int
    shared_genes: int
    donor_only_genes: int
    removed_for_budget: int
    fallback_reason: str | None = None


def extract_basis_genome(
    expression: str,
    variable_names: Sequence[str],
) -> BasisGenome:
    """Expand one expression into unique structural additive basis genes."""

    symbolic, symbols = parse_expression(expression, variable_names)
    by_canonical: dict[str, BasisGene] = {}
    for term in decompose_expand_mul(symbolic, symbols):
        if not term.basis.free_symbols:
            continue
        by_canonical.setdefault(
            term.canonical,
            BasisGene(
                canonical=term.canonical,
                expression=to_project_expression_string(term.basis),
            ),
        )
    return BasisGenome(
        tuple(by_canonical[key] for key in sorted(by_canonical))
    )


def merge_basis_genomes(
    receiver: BasisGenome,
    donor: BasisGenome,
) -> tuple[tuple[BasisGene, bool, bool], ...]:
    """Return a canonical union with receiver/donor source membership."""

    receiver_by_key = {term.canonical: term for term in receiver.terms}
    donor_by_key = {term.canonical: term for term in donor.terms}
    keys = sorted(set(receiver_by_key).union(donor_by_key))
    return tuple(
        (
            (
                receiver_by_key[key]
                if key in receiver_by_key
                else donor_by_key[key]
            ),
            key in receiver_by_key,
            key in donor_by_key,
        )
        for key in keys
    )


def opaque_basis_genome(expression: str) -> BasisGenome:
    text = str(expression)
    return BasisGenome(
        (BasisGene(f"OPAQUE|{text}", text, opaque=True),)
    )


def extract_strict_or_opaque_basis_genome(
    expression: str,
    variable_names: Sequence[str],
    *,
    max_len: int,
) -> tuple[BasisGenome, str]:
    """Return exact additive genes when they rebuild, otherwise one macro-gene."""

    try:
        genome = extract_basis_genome(expression, variable_names)
        build_basis_tree_direct(
            genome,
            variable_names,
            max_len=max_len,
            verify_roundtrip=True,
        )
    except Exception:
        return opaque_basis_genome(expression), "opaque_macro"
    return genome, "exact_terms"


def build_basis_tree(
    genome: BasisGenome,
    variable_names: Sequence[str],
    *,
    max_len: int,
    verify_roundtrip: bool = True,
) -> nd.Symbol:
    """Build one additive AST and verify that its structural genes round-trip."""

    if not genome.terms:
        if max_len < 1:
            raise ValueError("basis genome exceeds the configured AST length")
        return nd.Number(0.0)
    symbolic_terms: list[sp.Expr] = []
    expected = tuple(sorted(genome.canonicals))
    for gene in genome.terms:
        symbolic, symbols = parse_expression(gene.expression, variable_names)
        decomposed = tuple(
            term
            for term in decompose_expand_mul(symbolic, symbols)
            if term.basis.free_symbols
        )
        if verify_roundtrip and (
            len(decomposed) != 1 or decomposed[0].canonical != gene.canonical
        ):
            raise ValueError("basis gene expression is not a single matching term")
        symbolic_terms.append(symbolic)
    combined = (
        symbolic_terms[0]
        if len(symbolic_terms) == 1
        else sp.Add(*symbolic_terms, evaluate=False)
    )
    tree = nd.parse(to_project_expression_string(combined))
    if len(tree) > max_len:
        raise ValueError("basis genome exceeds the configured AST length")
    if verify_roundtrip:
        actual = extract_basis_genome(
            tree.to_str(number_format=".17g"), variable_names
        )
        if tuple(sorted(actual.canonicals)) != expected:
            raise ValueError("additive AST round-trip changed the basis genome")
    return tree


def build_basis_tree_direct(
    genome: BasisGenome,
    variable_names: Sequence[str],
    *,
    max_len: int,
    verify_roundtrip: bool = True,
) -> nd.Symbol:
    """Build a left-associated additive AST without SymPy recombination."""

    if not genome.terms:
        if max_len < 1:
            raise ValueError("basis genome exceeds the configured AST length")
        return nd.Number(0.0)
    trees = [nd.parse(gene.expression) for gene in genome.terms]
    tree = trees[0]
    for term_tree in trees[1:]:
        tree = nd.Add(tree, term_tree)
    if len(tree) > max_len:
        raise ValueError("basis genome exceeds the configured AST length")
    if verify_roundtrip:
        if not genome.is_exact:
            raise ValueError("opaque basis genes cannot use exact round-trip verification")
        actual = extract_basis_genome(
            tree.to_str(number_format=".17g"), variable_names
        )
        if tuple(sorted(actual.canonicals)) != tuple(sorted(genome.canonicals)):
            raise ValueError("additive AST round-trip changed the basis genome")
    return tree


def select_uniform_basis_crossover(
    receiver: BasisGenome,
    donor: BasisGenome,
    variable_names: Sequence[str],
    *,
    max_len: int,
    rng: np.random.Generator,
) -> BasisCrossoverResult:
    """Uniform set crossover with one receiver/donor direction when possible."""

    pool = merge_basis_genomes(receiver, donor)
    if not pool:
        return BasisCrossoverResult(
            receiver,
            0,
            0,
            0,
            0,
            0,
            "empty_parent_pool",
        )
    priorities = {gene.canonical: float(rng.random()) for gene, _, _ in pool}
    selected = {
        gene.canonical
        for gene, _, _ in pool
        if priorities[gene.canonical] < 0.5
    }
    receiver_keys = {gene.canonical for gene in receiver.terms}
    donor_keys = {gene.canonical for gene in donor.terms}
    donor_only = donor_keys.difference(receiver_keys)

    if receiver_keys and not selected.intersection(receiver_keys):
        selected.add(max(receiver_keys, key=lambda key: (priorities[key], key)))
    if donor_only and not selected.intersection(donor_only):
        selected.add(max(donor_only, key=lambda key: (priorities[key], key)))
    if not selected:
        selected.add(max(priorities, key=lambda key: (priorities[key], key)))

    by_key = {gene.canonical: gene for gene, _, _ in pool}
    removed = 0
    while True:
        genes = tuple(by_key[key] for key in sorted(selected))
        genome = BasisGenome(genes)
        try:
            build_basis_tree_direct(
                genome,
                variable_names,
                max_len=max_len,
                verify_roundtrip=False,
            )
            break
        except ValueError as error:
            if "AST length" not in str(error):
                return BasisCrossoverResult(
                    receiver, 0, 0, 0, 0, removed, type(error).__name__
                )
        removable = [
            key
            for key in selected
            if not (
                key in receiver_keys
                and len(selected.intersection(receiver_keys)) == 1
            )
            and not (
                key in donor_only
                and len(selected.intersection(donor_only)) == 1
            )
        ]
        if not removable:
            return BasisCrossoverResult(
                receiver, 0, 0, 0, 0, removed, "max_len"
            )
        selected.remove(min(removable, key=lambda key: (priorities[key], key)))
        removed += 1

    return BasisCrossoverResult(
        genome=genome,
        receiver_genes=len(selected.intersection(receiver_keys)),
        donor_genes=len(selected.intersection(donor_keys)),
        shared_genes=len(selected.intersection(receiver_keys, donor_keys)),
        donor_only_genes=len(selected.intersection(donor_only)),
        removed_for_budget=removed,
    )
