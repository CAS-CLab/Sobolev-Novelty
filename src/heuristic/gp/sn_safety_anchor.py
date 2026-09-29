"""Small deterministic helpers for the stable SN-GP Base safety anchor."""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar


CandidateT = TypeVar("CandidateT")


def select_base_anchor(
    population: Sequence[CandidateT],
    *,
    key: Callable[[CandidateT], object],
) -> CandidateT:
    if not population:
        raise ValueError("cannot select a Base anchor from an empty population")
    return min(population, key=key)


def merge_anchor_with_sn_elites(
    base_anchor: CandidateT,
    sn_elites: Sequence[CandidateT],
    elite_count: int,
    *,
    identity: Callable[[CandidateT], object],
) -> list[CandidateT]:
    """Place one Base anchor first, de-duplicate, and keep elite count fixed."""

    if elite_count < 1:
        raise ValueError("elite_count must be positive")
    selected = [base_anchor]
    seen = {identity(base_anchor)}
    for candidate in sn_elites:
        candidate_identity = identity(candidate)
        if candidate_identity in seen:
            continue
        selected.append(candidate)
        seen.add(candidate_identity)
        if len(selected) == elite_count:
            break
    if len(selected) != elite_count:
        raise ValueError(
            f"only {len(selected)} unique elites available; expected {elite_count}"
        )
    return selected
