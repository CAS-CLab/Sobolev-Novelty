"""Leave-one-out geometry for conditional Sobolev basis replacement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .sn_basis_archive import BasisArchiveEntry
from .sn_basis_pursuit import ScreenedArchiveDonor, screen_archive_basis_donors
from .sn_basis_replacement import RemovalCandidate


@dataclass(frozen=True)
class ConditionalReplacementShortlist:
    removal: RemovalCandidate
    retained_indices: tuple[int, ...]
    residual_norm: float
    donors: tuple[ScreenedArchiveDonor, ...]


def screen_conditional_replacements(
    *,
    parent_canonicals: Sequence[str],
    parent_signatures: np.ndarray,
    parent_values: np.ndarray,
    parent_structural: Sequence[bool],
    target: Sequence[float],
    removals: Sequence[RemovalCandidate],
    archive_entries: Sequence[BasisArchiveEntry],
    current_generation: int,
    donor_shortlist_size: int,
    coefficient_decimals: int = 6,
) -> tuple[ConditionalReplacementShortlist, ...]:
    """Screen archive donors in each proposed removal's retained context."""

    count = len(parent_canonicals)
    signatures = np.asarray(parent_signatures, dtype=float)
    values = np.asarray(parent_values, dtype=float)
    structural = np.asarray(parent_structural, dtype=bool)
    y = np.asarray(target, dtype=float).reshape(-1)
    if (
        count == 0
        or signatures.ndim != 2
        or signatures.shape[1] != count
        or values.shape != (y.size, count)
        or structural.shape != (count,)
        or donor_shortlist_size < 1
        or coefficient_decimals < 0
        or not all(
            np.all(np.isfinite(item))
            for item in (signatures, values, y)
        )
    ):
        raise ValueError("conditional replacement inputs are not aligned and finite")

    output: list[ConditionalReplacementShortlist] = []
    parent_identities = {str(value) for value in parent_canonicals}
    usable_indices = tuple(
        index
        for index, entry in enumerate(archive_entries)
        if entry.canonical not in parent_identities
    )
    usable_archive = tuple(archive_entries[index] for index in usable_indices)
    for removal in removals:
        removed = int(removal.parent_index)
        if removed < 0 or removed >= count or not structural[removed]:
            raise ValueError("conditional replacement removal is not structural")
        retained = tuple(index for index in range(count) if index != removed)
        retained_structural = tuple(
            index for index in retained if structural[index]
        )
        design_columns = [np.ones(y.size, dtype=float)]
        design_columns.extend(values[:, index] for index in retained_structural)
        design = np.column_stack(design_columns)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        coefficients = np.round(coefficients, coefficient_decimals)
        residual = y - design @ coefficients
        screened = screen_archive_basis_donors(
            parent_canonicals=tuple(parent_canonicals[index] for index in retained),
            parent_signatures=signatures[:, retained],
            parent_residual=residual,
            parent_value_design=design,
            archive_entries=usable_archive,
            current_generation=current_generation,
            shortlist_size=donor_shortlist_size,
        )
        donors = tuple(
            replace(donor, archive_index=usable_indices[donor.archive_index])
            for donor in screened
        )
        output.append(
            ConditionalReplacementShortlist(
                removal=removal,
                retained_indices=retained,
                residual_norm=float(np.linalg.norm(residual)),
                donors=donors,
            )
        )
    return tuple(output)
