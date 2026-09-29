"""
Search implementation (Monte-Carlo Tree Search).
"""

import math
import random
from dataclasses import dataclass, field
from itertools import count
from math import log, sqrt
from typing import TYPE_CHECKING, Any, Dict, Generic, Literal, Optional

try:
    import networkx as nx
except ImportError:
    # networkx is optional: only used for tree-graph tracking / the visualiser (see [benchmarks]).
    nx = None

from igsr.logging_setup import get_logger
from igsr.method.igsr_utils import check_token_budget_and_maybe_break, check_wallclock_budget_and_maybe_break

from .problem import Problem, R, S

if TYPE_CHECKING:
    from .visualizer import Visualizer


_networkx_warned = False


class _NullGraph:
    """No-op stand-in for ``nx.DiGraph`` when networkx is not installed (search is unaffected)."""

    def add_node(self, *args, **kwargs) -> None:
        pass

    def add_edge(self, *args, **kwargs) -> None:
        pass


def _warn_networkx_missing() -> None:
    """Log once that networkx is needed for tree-graph tracking / visualisation."""
    global _networkx_warned
    if not _networkx_warned:
        _networkx_warned = True
        get_logger("igsr.tree_search.session").warning(
            "networkx_not_installed",
            message="networkx is not installed, so tree-graph tracking / visualisation is disabled. "
            "Install it (`pip install networkx`, or `pip install -e .[benchmarks]`) to enable it.",
        )


def check_early_stop(
    direction: Literal["min", "max"],
    best_result: float,
    result: float,
    patience_left: int,
    patience_max: int,
) -> int:
    if direction == "min":
        if best_result < result:
            patience_left -= 1
        else:
            patience_left = patience_max
    elif direction == "max":
        if best_result > result:
            patience_left -= 1
        else:
            patience_left = patience_max
    else:
        raise ValueError(f"Invalid direction: {direction}")
    return patience_left


class SearchSession(Generic[S, R]):
    """Monte-Carlo Tree Search session manager.

    Runs MCTS (see :meth:`mcts`) on a given :class:`~igsr.tree_search.problem.Problem`,
    with optional visualization support.

    Attributes:
        problem (Problem[S, R]): The problem instance to search on.
        visualiser (Optional[Visualizer]): Optional visualizer to show search progress.
        best_val (float): The best evaluation value found so far.
        best_state (Optional[S]): The state with the best evaluation value.
        iteration (int): Current iteration counter.
        graph (nx.DiGraph): NetworkX DiGraph storing the search tree.
    """

    def __init__(
        self,
        problem: Problem[S, R],
        visualiser: Optional["Visualizer"] = None,
        early_stopping_enabled: bool = False,  # MCTS only
        early_stopping_patience: int = 10,  # MCTS only
    ):
        self.problem = problem
        self.visualiser = visualiser
        self.reset()
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_patience = early_stopping_patience

    def reset(self, seed: int = 42):
        """Reset the search session to its initial state.

        This clears all search progress and prepares for a new search.

        Args:
            seed (int): Random seed for reproducibility.
        """
        # — private per-session state —–––––––––––––––––––––––––––––––––––––––
        self._rng: random.Random = random.Random(seed)
        self._id_counter: count = count()
        if nx is not None:
            self.graph = nx.DiGraph()
        else:
            if self.visualiser is not None:
                _warn_networkx_missing()
            self.graph = _NullGraph()

        self.best_val: float = float("inf")
        self.best_state: Optional[S] = None
        self.iteration: int = 0

        if self.visualiser:
            self.visualiser.reset()

    # =====================================================================
    #  Monte-Carlo Tree Search  (budget = #iterations)
    # =====================================================================
    @dataclass
    class _MCNode(Generic[S, R]):
        """Node representation for Monte Carlo Tree Search.

        Attributes:
            state (S): The state represented by this node.
            parent (Optional[int]): ID of the parent node, or None if root.
            children (list[int]): List of child node IDs.
            n (int): Number of visits to this node.
            w (float): Accumulated reward from this node.
        """

        state: S
        parent: Optional[int]
        children: list[int] = field(default_factory=list)
        n: int = 0  # visits
        w: float = 0.0  # accumulated reward
        succ: Optional[list[S]] = None  # <-- cache of successors

    def _succ(self, node: _MCNode[S, R]) -> list[S]:
        """Return (and lazily cache) node successors."""
        if node.succ is None:
            # expensive call done only once per node
            node.succ = list(self.problem.successors(node.state))
        return node.succ

    def _enact_token_or_wallclock_early_stop(self, extras_dict: Dict[str, Any]) -> bool:
        # Get all the needed items from extra_dict (no defaulting):
        cfg = extras_dict["cfg"]
        compute_profiler = extras_dict["compute_profiler"]
        start_time = extras_dict["start_time"]
        seed = extras_dict["seed"]
        logger = extras_dict["logger"]

        if check_token_budget_and_maybe_break(
            cfg=cfg, round=self.iteration, compute_profiler=compute_profiler, seed=seed, logger=logger
        ):
            return True
        if check_wallclock_budget_and_maybe_break(
            cfg=cfg, round=self.iteration, start_time=start_time, seed=seed, logger=logger
        ):
            return True

        return False

    def mcts(
        self,
        start: S,
        *,
        depth_limit: int,
        total_budget: int,
        rollout_depth: int = 10,
        c: float = math.sqrt(2),
        stop_on_terminal: bool = True,
        seed: int = 42,
        rollout_is_just_node_reward: bool = False,
        cache_successors: bool = True,
        always_exit_on_terminal: bool = False,
        # Any other extras:
        extras_dict: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, Optional[S]]:
        """Perform Monte Carlo Tree Search from a starting state.

        Args:
            start (S): The initial state to start search from.
            depth_limit (int): Maximum depth to explore in the search tree.
            total_budget (int): Maximum number of iterations (simulations) to run.
            rollout_depth (int): Maximum depth for the rollout phase.
            c (float): Exploration constant in UCT formula.
            stop_on_terminal (bool): If True, stop search when a terminal state is found.
            seed (int): Random seed for reproducibility.
            rollout_is_just_node_reward (bool): If True, the rollout phase is just a single node reward.
            cache_successors (bool): If True, cache successors for each node.
            always_exit_on_terminal (bool): If True, always exit on terminal state. EXPERIMENTAL.
            extras_dict (Optional[Dict[str, Any]]): Any other extras.

        Returns:
            tuple: A pair containing (best evaluation value, best state found).
        """
        # ── session-level init ─────────────────────────────────────────
        self.reset(seed=seed)

        if extras_dict is None:
            extras_dict = {}

        nodes: dict[int, SearchSession._MCNode] = {}
        root = next(self._id_counter)
        nodes[root] = self._MCNode(start, None)
        self.graph.add_node(root, state=start)

        if self.visualiser:
            self.visualiser.on_new_node(
                root,
                None,
                self.problem.label,
                start,  # state
                self.best_state,  # current best state
                self.best_val,
            )  # current best value

        self.best_val = self.problem.evaluate(start)
        self.best_state = start
        patience_left = self.early_stopping_patience

        # ── main loop (budget = iterations) ────────────────────────────
        for self.iteration in range(total_budget):
            # === 1. Selection =========================================
            path: list[int] = []
            node_id = root
            while True:
                path.append(node_id)
                node = nodes[node_id]

                # --- terminal check -----------------------------------
                if self.problem.is_terminal(node.state):
                    val = self.problem.evaluate(node.state)
                    if val < self.best_val:  # update best before exit
                        self.best_val, self.best_state = val, node.state
                    if stop_on_terminal:
                        return self.best_val, self.best_state
                    break  # else treat as leaf

                if self._enact_token_or_wallclock_early_stop(extras_dict):
                    return self.best_val, self.best_state

                if len(path) > depth_limit:
                    break

                # --- fully-expanded?  use key() for logical equality --
                if cache_successors:
                    succ_keys = {self.problem.key(s) for s in self._succ(node)}
                else:
                    succ_keys = {self.problem.key(s) for s in self.problem.successors(node.state)}
                child_keys = {self.problem.key(nodes[c].state) for c in node.children}

                if len(child_keys) < len(succ_keys):  # there is still room
                    break

                # --- UCT choice ---------------------------------------
                log_N = log(node.n)
                node_id = max(
                    node.children, key=lambda cid: (nodes[cid].w / nodes[cid].n) + c * sqrt(log_N / nodes[cid].n)
                )

            node = nodes[node_id]

            # === 2. Expansion ========================================
            if not self.problem.is_terminal(node.state) and len(path) <= depth_limit:
                tried_keys = {self.problem.key(nodes[c].state) for c in node.children}
                if cache_successors:
                    untried = [s for s in self._succ(node) if self.problem.key(s) not in tried_keys]
                else:
                    untried = [s for s in self.problem.successors(node.state) if self.problem.key(s) not in tried_keys]

                if untried:
                    # Domains may provide a paired, deterministic expansion
                    # selector.  Base IGSR has no selector and keeps the exact
                    # historical random-choice behaviour.
                    selector = getattr(self.problem, "select_untried_successor", None)
                    if callable(selector):
                        s_new = selector(untried, self._rng)
                        if s_new not in untried:
                            raise ValueError(
                                "Problem.select_untried_successor returned a state "
                                "outside the supplied untried successors"
                            )
                    else:
                        s_new = self._rng.choice(untried)
                    cid = next(self._id_counter)
                    nodes[cid] = self._MCNode(s_new, node_id)
                    node.children.append(cid)

                    self.graph.add_node(cid, state=s_new)
                    self.graph.add_edge(node_id, cid)

                    node_id = cid
                    path.append(cid)
                    node = nodes[cid]

                    val = self.problem.evaluate(s_new)
                    if val < self.best_val:
                        self.best_val, self.best_state = val, s_new

                    # == early stopping (patience + token / wall-clock budget) ==
                    if self.early_stopping_enabled:
                        patience_left = check_early_stop(
                            direction="min",
                            best_result=self.best_val,
                            result=val,
                            patience_left=patience_left,
                            patience_max=self.early_stopping_patience,
                        )
                        if patience_left <= 0:
                            return self.best_val, self.best_state

                    if self._enact_token_or_wallclock_early_stop(extras_dict):
                        return self.best_val, self.best_state
                    # === end of early stopping ===============================

                    if self.visualiser:
                        self.visualiser.on_new_node(
                            cid, node.parent, self.problem.label, s_new, self.best_state, self.best_val
                        )
            # === early exit on terminal (experimental) ================
            else:
                if always_exit_on_terminal and stop_on_terminal and self.problem.is_terminal(node.state):
                    if val < self.best_val:
                        self.best_val, self.best_state = val, node.state
                    return self.best_val, self.best_state

            if self._enact_token_or_wallclock_early_stop(extras_dict):
                return self.best_val, self.best_state
            # === end of early exit on terminal ========================

            # === 3. Roll-out =========================================
            if not rollout_is_just_node_reward:
                s_roll, d = node.state, 0
                while (not self.problem.is_terminal(s_roll)) and d < rollout_depth:
                    if cache_successors:
                        succ = self._succ(node)
                    else:
                        succ = list(self.problem.successors(s_roll))
                    if not succ:
                        break
                    s_roll = self._rng.choice(succ)
                    d += 1
                r = self.problem.reward(s_roll)
                # === early exit on terminal (experimental) ================
                if always_exit_on_terminal and stop_on_terminal and self.problem.is_terminal(s_roll):
                    if val < self.best_val:
                        self.best_val, self.best_state = val, node.state
                    return self.best_val, self.best_state

                if self._enact_token_or_wallclock_early_stop(extras_dict):
                    return self.best_val, self.best_state
                # === end of early exit on terminal ========================
            else:
                r = self.problem.reward(node.state)

            # === 4. Back-propagation =================================
            for nid in path:
                n = nodes[nid]
                n.n += 1
                n.w += r

        return self.best_val, self.best_state
