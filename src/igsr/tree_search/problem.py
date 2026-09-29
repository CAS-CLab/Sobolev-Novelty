"""
Minimal domain interface for tree search.
"""

from typing import Generic, Hashable, Iterable, Protocol, TypeVar

S = TypeVar("S")  # State.
R = TypeVar("R", bound=float)  # Reward  (float is enough for most cases).


class Problem(Protocol, Generic[S, R]):
    """
    Minimal domain interface for tree search.
    """

    # Mandatory ————————————————————————————————————————————————————————————
    def successors(self, s: S) -> Iterable[S]:
        """Generate successors of a state.

        Args:
            s: State.

        Returns:
            Iterable of successors.
        """
        ...

    def evaluate(self, s: S) -> float:
        """Evaluate a state. Lower is better.

        Args:
            s: State.

        Returns:
            Evaluation of the state.
        """
        ...

    def reward(self, s: S) -> R:
        """MCTS roll-out payoff.

        Args:
            s: State.

        Returns:
            Reward for the state.
        """
        return -self.evaluate(s)

    # Optional ————————————————————————————————————————————————————————————
    def is_terminal(self, s: S) -> bool:
        """Stop expanding?

        Args:
            s: State.

        Returns:
            True if the state is terminal, False otherwise.
        """
        return False

    # Helpers (no impact on search) ————————————————————————————————————————
    def key(self, s: S) -> Hashable:
        """Canonical hash (key) for comparing states. If your states are not hashable, you need to override this!

        Args:
            s: State.

        Returns:
            Canonical hash (key) for the state.
        """
        return s

    def label(self, s: S) -> str:
        """Label for pretty-print.

        Args:
            s: State.

        Returns:
            Label for the state.
        """
        return str(s)
