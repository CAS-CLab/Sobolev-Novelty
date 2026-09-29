from __future__ import annotations

import numpy as np

from sn_dsrrans.backward_selection import (
    QRBackwardWorkspace,
    choose_deletion,
    leave_one_out_novelties,
)
from sn_dsrrans.canonical_selection import CanonicalTerm


def test_qr_deletion_increases_match_explicit_refits() -> None:
    rng = np.random.default_rng(7)
    design = rng.normal(size=(80, 7))
    target = rng.normal(size=80)
    workspace = QRBackwardWorkspace(design, target)
    selected = list(range(design.shape[1]))
    increases, fit = workspace.deletion_increases(selected)
    explicit = []
    for deleted in selected:
        remaining = [index for index in selected if index != deleted]
        explicit.append(workspace.fit(remaining).sse - fit.sse)
    np.testing.assert_allclose(increases, explicit, atol=1e-10, rtol=1e-9)


def test_qr_deletion_detects_exactly_redundant_columns() -> None:
    rng = np.random.default_rng(11)
    first = rng.normal(size=50)
    second = rng.normal(size=50)
    design = np.column_stack((first, second, first + second))
    workspace = QRBackwardWorkspace(design, rng.normal(size=50))
    increases, _ = workspace.deletion_increases([0, 1, 2])
    np.testing.assert_allclose(increases, 0.0, atol=1e-10)


def test_leave_one_out_novelty_is_channel_local() -> None:
    signatures = np.column_stack(
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
    )
    terms = (
        CanonicalTerm(0, 0, 0),
        CanonicalTerm(0, 1, 0),
        CanonicalTerm(1, 0, 0),
    )
    novelties = leave_one_out_novelties(signatures, terms, [0, 1, 2])
    np.testing.assert_allclose(novelties, [0.0, 0.0, 1.0], atol=1e-12)


def test_sn_deletion_obeys_base_energy_r2_band() -> None:
    selected = [4, 7, 9]
    increases = np.array([1.0, 1.4, 3.0])
    novelties = np.array([0.8, 0.1, 0.0])
    chosen, decision = choose_deletion(
        selected,
        increases,
        novelties,
        method="sn_epsilon",
        epsilon_r2=0.05,
        target_energy=10.0,
    )
    assert chosen == 7
    assert decision["eligible_count"] == 2
    assert decision["changed_from_base"] == 1

