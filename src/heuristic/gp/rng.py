"""Counter-keyed random streams for paired Base-GP and SN-GP runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


RNG_SCHEMA_VERSION = "gp_counter_rng_v1"
RNG_EVENTS = frozenset(
    {
        "init_tree",
        "parent_tournament",
        "operator_choice",
        "donor_tournament",
        "crossover",
        "subtree_mutation",
        "hoist_mutation",
        "point_mutation",
        "reproduction",
    }
)


@dataclass(frozen=True)
class CounterRNG:
    """Derive an independent Philox stream for each genetic event.

    A structural method may change the parents selected in later generations,
    but it cannot shift unrelated random draws merely by evaluating Sobolev
    novelty or by traversing a tree with a different number of nodes.
    """

    run_seed: int
    namespace: str = "controlled_gp"

    def generator(
        self,
        generation: int,
        offspring_slot: int,
        event: str,
        attempt: int = 0,
    ) -> np.random.Generator:
        if generation < 0 or offspring_slot < 0 or attempt < 0:
            raise ValueError("generation, offspring_slot and attempt must be non-negative")
        if event not in RNG_EVENTS:
            raise ValueError(f"Unknown GP RNG event: {event}")
        payload = json.dumps(
            {
                "schema": RNG_SCHEMA_VERSION,
                "namespace": self.namespace,
                "run_seed": int(self.run_seed),
                "generation": int(generation),
                "offspring_slot": int(offspring_slot),
                "event": event,
                "attempt": int(attempt),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        entropy = hashlib.blake2b(
            payload,
            digest_size=16,
            person=b"EIC-GP-RNG-v1",
        ).digest()
        words = np.frombuffer(entropy, dtype="<u4").astype(np.uint64)
        seed_sequence = np.random.SeedSequence(words.tolist())
        return np.random.Generator(np.random.Philox(seed_sequence))

    @property
    def schedule_identity(self) -> str:
        payload = (
            f"{RNG_SCHEMA_VERSION}|{self.namespace}|{int(self.run_seed)}|"
            + ",".join(sorted(RNG_EVENTS))
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

