"""
Matplotlib / NetworkX plugin for the new Visualizer protocol
Depends on: matplotlib, networkx, IPython.display
"""

import time
from typing import Any, Callable, Dict, Generic, Optional, Protocol, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from IPython import display

from .problem import R, S


# ──────────────────────────────────────────────────────────────────────────
#  Visualiser plug-in API
# ──────────────────────────────────────────────────────────────────────────
class Visualizer(Protocol):
    """Protocol defining the interface for tree search visualizers.

    This protocol defines methods that tree search algorithms can use to
    visualize the search process.
    """

    def reset(self):
        """Reset the visualizer state.

        Clears any existing visualization data to prepare for a new search.
        """
        ...

    def on_new_node(
        self,
        node_id: int,
        parent_id: Optional[int],
        label_generator: Callable[[S], str],
        state: S,
        best_state: Optional[S],
        best_val: float,
    ):
        """Handle the addition of a new node to the search tree.

        Args:
            node_id (int): Unique identifier for the new node.
            parent_id (Optional[int]): Identifier of the parent node, or None if root.
            label_generator (Callable[[S], str]): Function to generate a text label for a state.
            state (S): The state represented by this node.
            best_state (Optional[S]): The current best state found in the search.
            best_val (float): The evaluation value of the best state.
        """
        ...


class MatplotlibVisualizer(Generic[S, R]):
    """
    Drop-in Visualizer that reproduces the live tree view you had:
      - blue nodes  = ordinary states
      - red node    = current best state
      - title       = best fitness / cost so far
    """

    def __init__(
        self,
        pause: float = 0.5,
        prog: str = "dot",
        nx_draw_kwargs: Optional[Dict[str, Any]] = None,
        nx_figsize: Tuple[float, float] = (10, 10),
    ):
        """
        Args:
            pause (float): real-time delay between frames (seconds)
            prog (str): Graphviz program used for layout (fallback → spring layout)
            nx_draw_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for nx.draw
            nx_figsize (Tuple[float, float]): Size of the figure for nx.draw
        """
        self.pause = pause
        self.prog = prog
        self.G = nx.DiGraph()
        self._pos = None  # cached node positions
        self.nx_figsize = nx_figsize

        # Set default kwargs for nx.draw.
        # Disallow node_color, labels, pos.
        disallowed_kwargs = {"node_color", "labels", "pos"}
        if set(nx_draw_kwargs.keys()) & disallowed_kwargs:
            raise ValueError(f"Disallowed kwargs for nx.draw (these are set internally): {disallowed_kwargs}")
        nx_draw_kwargs_ = dict(
            node_size=800,
            arrows=False,
            font_size=1,
        )
        nx_draw_kwargs_.update(nx_draw_kwargs)
        self.nx_draw_kwargs = nx_draw_kwargs_

        plt.ion()  # interactive mode

    # ------------------------------------------------------------------ #
    # Protocol hooks
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the visualizer state.

        Clears the graph, node positions, and display output to prepare for a new visualization.
        """
        self.G.clear()
        self._pos = None
        display.clear_output(wait=True)

    def on_new_node(
        self,
        node_id: int,
        parent_id: Optional[int],
        label_generator: Callable[[S], str],
        state: S,
        best_state: Optional[S],
        best_val: float,
    ) -> None:
        """Display a visualization update when a new node is added to the search tree.

        Args:
            node_id (int): Unique identifier for the new node.
            parent_id (Optional[int]): Identifier of the parent node, or None if root.
            label_generator (Callable[[S], str]): Function to generate a text label for a state.
            state (S): The state represented by this node.
            best_state (Optional[S]): The current best state found in the search.
            best_val (float): The evaluation value of the best state.
        """
        # -- keep a private copy of the tree --------------------------------
        self.G.add_node(node_id, state=state)
        if parent_id is not None:
            self.G.add_edge(parent_id, node_id)

        # (re-)compute layout only when the graph changes size
        if self._pos is None or len(self._pos) != self.G.number_of_nodes():
            self._pos = self._layout()

        # -- draw -----------------------------------------------------------
        labels = {n: label_generator(self.G.nodes[n]["state"]) for n in self.G.nodes}
        try:
            colours = ["#ff6666" if self.G.nodes[n]["state"] == best_state else "#99ccff" for n in self.G.nodes]
        except Exception:
            # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            try:
                colours = ["#ff6666" if self.G.nodes[n]["state"][0] == best_state else "#99ccff" for n in self.G.nodes]
            except Exception:
                # Fall back on no highlighting if we can't compare states.
                colours = ["#99ccff" for n in self.G.nodes]

        plt.clf()
        display.clear_output(wait=True)

        try:
            import IPython.display
        except ImportError:
            pass
        else:
            IPython.display.clear_output(wait=True)

        # Generate a large figure and pass its ax to nx.draw.
        fig = plt.figure(figsize=self.nx_figsize)
        ax = fig.add_subplot(111)
        nx.draw(self.G, pos=self._pos, labels=labels, node_color=colours, ax=ax, **self.nx_draw_kwargs)
        plt.title(f"Best value so far = {best_val}")
        plt.pause(0.001)
        time.sleep(self.pause)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _layout(self):
        """Compute the layout for the search tree visualization.

        Returns:
            dict: Mapping of node IDs to (x, y) positions.

        Note:
            Tries to use graphviz layout if available, with spring layout as fallback.
        """
        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            return graphviz_layout(self.G, prog=self.prog)
        except Exception:
            return nx.spring_layout(self.G)
