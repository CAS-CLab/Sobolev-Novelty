from typing import Any, Dict, List, Literal, Optional, Tuple, Union


class FormulaTracker:
    """Utility class for tracking (formula, metric) pairs during the symbolic regression loop.

    Allows to get:
    - History of the pairs (in order of appearance)
    - Top-K pairs (best metric first)
    """

    def __init__(self, metric_direction: Literal["min", "max"] = "min"):
        """Initialize the tracker.

        Args:
            metric_direction (Literal["min", "max"]): Whether to minimize or maximize the metric.
                i.e. "min" = lower is better, "max" = higher is better.
        """
        self._formula_to_metric: Dict[str, float] = dict()  # NOTE: Stores best metric for each formula
        self._full_history: List[Tuple[str, float]] = []
        self._metric_direction: Literal["min", "max"] = metric_direction

    def update(self, formula: str, metric: float) -> None:
        """Register a (formula, metric) observation.

        If the same formula has been seen before we keep the *best* metric encountered so far
        (for the purposes of top K / best result).

        Args:
            formula (str): The formula to register.
            metric (float): The metric to register.
        """
        # NOTE: Some formulas may be returned with an explicit leading ``y =`` part. To make sure string comparisons
        # behave as expected we strip it once here so that logically-identical formulae are treated as the same key
        # independent of that syntactic variant.
        key = formula.strip()
        if key.startswith("y ="):
            key = key[3:].strip()

        self._full_history.append((key, metric))

        current_best = self._formula_to_metric.get(key)
        if self._metric_direction == "min":
            if current_best is None or metric < current_best:
                self._formula_to_metric[key] = metric
        elif self._metric_direction == "max":
            if current_best is None or metric > current_best:
                self._formula_to_metric[key] = metric
        else:
            raise ValueError(f"Invalid metric direction: {self._metric_direction}")

    def top_k_results(self, k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-*k* (formula, metric) pairs ordered by ascending metric.

        Args:
            k (int): The number of top results to return.

        Returns:
            List[Tuple[str, float]]: The top-*k* (formula, metric) pairs ordered by ascending metric.
        """
        if self._metric_direction == "min":
            return sorted(self._formula_to_metric.items(), key=lambda item: item[1])[:k]
        elif self._metric_direction == "max":
            return sorted(self._formula_to_metric.items(), key=lambda item: item[1], reverse=True)[:k]
        else:
            raise ValueError(f"Invalid metric direction: {self._metric_direction}")

    def best_result(self) -> Tuple[str, float]:
        """Return the (formula, metric) pair with the best metric so far.

        Returns:
            Tuple[str, float]: The (formula, metric) pair with the best metric so far.
        """
        if not self._formula_to_metric:
            raise ValueError("No formulas have been tracked yet.")
        top_entries = self.top_k_results(1)
        return top_entries[0]

    def get_history(self) -> List[Tuple[str, float]]:
        """Return the history of (formula, metric) in order of appearance.

        Returns:
            List[Tuple[str, float]]: The history of (formula, metric) in order of appearance.
        """
        return self._full_history


class FormulaTrackerMultiMetric:
    """Track **multiple** metrics for each *iteration* identifier.

    This class is a generalisation of the original :class:`FormulaTracker` but it
    now assumes that observations are keyed by an **iteration id** (string) and
    can optionally carry an arbitrary *metadata* dictionary alongside the
    metrics.

    The public API (`update`, `top_k_results`, `best_result`, `get_history`)
    remains unchanged in name, however the argument/return signatures have been
    adapted as follows (per user request):

    * ``update(iter_id, metrics, *, metadata=None)``
    * ``top_k_results(metric, k=5) -> List[Dict[str, Any]]``
    * ``best_result(metric) -> Dict[str, Any]``
    * ``get_history(metric=None) -> List[Dict[str, Any]]``

    Where the result dictionaries always contain **three** keys::

        {
            "iter_id": "...",          # str
            "metric_value": 0.123,      # float (for the *requested* metric)
            "metadata": {...}           # dict (possibly empty)
        }

    The optimisation direction ("min" / "max") is specified separately for each
    metric name at construction time and behaves exactly like in the original
    implementation.
    """

    # ---------------------------------------------------------------------
    # Construction / validation
    # ---------------------------------------------------------------------

    def __init__(self, metric_directions: Dict[str, Literal["min", "max"]]):
        # --- Validate directions ------------------------------------------
        invalid = [m for m, d in metric_directions.items() if d not in {"min", "max"}]
        if invalid:
            raise ValueError(f"Invalid optimisation direction for metric(s): {invalid}. Use 'min' or 'max'.")

        # Public configuration ---------------------------------------------------
        self._metric_directions: Dict[str, Literal["min", "max"]] = dict(metric_directions)

        # Internal state ---------------------------------------------------------
        # *Best* metric values seen so far for each iter_id -> metric name -> value
        self._iter_to_best_metrics: Dict[str, Dict[str, float]] = {}

        # Latest metadata for each iter_id
        self._iter_to_metadata: Dict[str, Dict[str, Any]] = {}

        # Full chronological history.  Each entry is a *copy* so that external
        # mutation does not affect the stored state.
        #   [{"iter_id": str, "metrics": {...}, "metadata": {...}}, ...]
        self._full_history: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def update(self, iter_id: str, metrics: Dict[str, float], *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new observation for **all** metrics associated with *iter_id*.

        Parameters
        ----------
        iter_id
            A unique string identifier for the iteration (or any other logical
            key you wish to track).
        metrics
            Mapping from metric name → value for this observation.  **All**
            metrics declared at construction *must* be provided.
        metadata
            Optional arbitrary information (JSON-serialisable) associated with
            this iteration. Stored verbatim.
        """

        # --- Validate metrics --------------------------------------------------
        missing = [m for m in self._metric_directions if m not in metrics]
        if missing:
            raise ValueError(
                f"Missing metric(s) in update: {missing}. All metrics {list(self._metric_directions)} must be provided."
            )

        extra = [m for m in metrics if m not in self._metric_directions]
        if extra:
            raise ValueError(
                f"Unknown metric(s) encountered in update: {extra}. Expected only {list(self._metric_directions)}."
            )

        # --- Store in history ---------------------------------------------------
        hist_entry = {
            "iter_id": iter_id,
            "metrics": dict(metrics),  # copy to avoid external mutation
            "metadata": dict(metadata) if metadata is not None else {},
        }
        self._full_history.append(hist_entry)

        # --- Update best-so-far records ----------------------------------------
        best_for_iter = self._iter_to_best_metrics.get(iter_id, {})
        updated_best = dict(best_for_iter)  # shallow copy

        for m_name, m_value in metrics.items():
            direction = self._metric_directions[m_name]
            current_best = best_for_iter.get(m_name)

            if current_best is None:
                updated_best[m_name] = m_value
            else:
                if (direction == "min" and m_value < current_best) or (direction == "max" and m_value > current_best):
                    updated_best[m_name] = m_value

        self._iter_to_best_metrics[iter_id] = updated_best

        # Latest metadata wins (simple policy)
        if metadata is not None:
            self._iter_to_metadata[iter_id] = dict(metadata)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _build_result_dict(self, iter_id: str, metric_value: float) -> Dict[str, Any]:
        """Helper to construct the standard result dictionary structure."""

        return {
            "iter_id": iter_id,
            "metric_value": metric_value,
            "metadata": self._iter_to_metadata.get(iter_id, {}),
        }

    def top_k_results(self, metric: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return the *top-k* results for **metric** as a list of dictionaries.

        The list is ordered according to the optimisation direction specified
        for the metric.
        """

        if metric not in self._metric_directions:
            raise ValueError(f"Unknown metric '{metric}'. Known metrics: {list(self._metric_directions)}")

        direction = self._metric_directions[metric]

        # Gather available values (skip iterations that don't have the metric yet)
        values: List[Tuple[str, float]] = [
            (iter_id, m_dict[metric]) for iter_id, m_dict in self._iter_to_best_metrics.items() if metric in m_dict
        ]

        reverse = direction == "max"  # descending when we *maximise*
        sorted_values = sorted(values, key=lambda item: item[1], reverse=reverse)[:k]

        return [self._build_result_dict(iter_id, value) for iter_id, value in sorted_values]

    def best_result(self, metric: str) -> Dict[str, Any]:
        """Return the single *best* result dictionary for **metric**."""

        top_entries = self.top_k_results(metric=metric, k=1)
        if not top_entries:
            raise ValueError("No iterations have been tracked yet.")
        return top_entries[0]

    def get_history(self, metric: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the full observation history.

        If *metric* is ``None`` the raw history entries are returned as stored
        during :py:meth:`update`.  When *metric* is provided the method extracts
        the value for that metric for each entry and returns a simplified list
        where every element has exactly the *three* required keys.
        """

        if metric is None:
            # Return *copies* to avoid external mutation affecting internal state
            return [dict(entry) for entry in self._full_history]

        if metric not in self._metric_directions:
            raise ValueError(f"Unknown metric '{metric}'. Known metrics: {list(self._metric_directions)}")

        processed: List[Dict[str, Any]] = []
        for entry in self._full_history:
            value = entry["metrics"][metric]
            processed.append(
                {
                    "iter_id": entry["iter_id"],
                    "metric_value": value,
                    "metadata": dict(entry["metadata"]),
                }
            )
        return processed


def build_history_section_top_K(
    top_k: Union[List[Tuple[str, float]], Tuple[str, float]],
    metric_name: Optional[str] = None,
    history_title: str = "## History",
    history_description: str = "Here are the top candidate formulas discovered so far (best metric first):",
    no_history_message: str = "No formulas have been tracked yet.",
) -> str:
    """Return a textual `## History` section to be appended to the LLM task.

    The section lists the best `k` distinct formulas (best metric first).

    Args:
        top_k (Union[List[Tuple[str, float]], Tuple[str, float]]):
            The top-*k* (formula, metric) pairs ordered by best metric first.
            If a single pair is provided, it will be converted to a list.
        metric_name (Optional[str]):
            The name of the metric to display in the history section.
            If not provided, use "Metric" as default.
        history_title (str):
            The title of the history section.
            If not provided, use "## History" as default.
        history_description (str):
            The description of the history section.
            If not provided, use "Here are the top candidate formulas discovered so far (best metric first):" as default.
        no_history_message (str):
            The message to display if no formulas have been tracked yet.
            If not provided, use "No formulas have been tracked yet." as default.

    Returns:
        str: The textual `## History` section to be appended to the LLM task.
    """

    if isinstance(top_k, tuple):
        top_k = [top_k]

    if metric_name is None:
        metric_name = "Metric"

    if not top_k:
        return no_history_message

    lines = [history_title, history_description, ""]
    for idx, (formula, metric) in enumerate(top_k, start=1):
        lines.append(f"{idx}. {metric_name} = {metric:.6g} | y = {formula}")
    lines.append("")  # Trailing newline for neatness.
    return "\n".join(lines)


def build_history_section_in_order(
    history: List[Tuple[str, float]],
    n: Optional[int] = None,
    metric_name: Optional[str] = None,
    history_title: str = "## History",
    history_description: str = "Here are the formulas in order of appearance:",
    no_history_message: str = "No formulas have been tracked yet.",
) -> str:
    """Return a textual `## History` section to be appended to the LLM task.

    The section lists the formulas in order of appearance, optionally limiting to the last `n` entries.

    Args:
        history (List[Tuple[str, float]]):
            The history of (formula, metric) pairs in order of appearance.
        n (Optional[int]):
            The number of most recent entries to display.
            If None, all history entries are shown.
            If not provided, defaults to None (show all).
        metric_name (Optional[str]):
            The name of the metric to display in the history section.
            If not provided, use "Metric" as default.
        history_title (str):
            The title of the history section.
            If not provided, use "## History" as default.
        history_description (str):
            The description of the history section.
            If not provided, use "Here are the formulas in order of appearance:" as default.
        no_history_message (str):
            The message to display if no formulas have been tracked yet.
            If not provided, use "No formulas have been tracked yet." as default.

    Returns:
        str: The textual `## History` section to be appended to the LLM task.
    """
    if metric_name is None:
        metric_name = "Metric"

    if not history:
        return no_history_message

    # Get the entries from the history
    if n is None or n >= len(history):
        recent_entries = history
        # Use the original description since we're showing all entries
        description = history_description
    elif n <= 0:
        recent_entries = []
        description = history_description
    else:
        recent_entries = history[-n:]
        # Add clarification about showing only most recent entries
        description = f"Here are the {n} most recent formulas in order of appearance:"

    lines = [history_title, description, ""]
    for idx, (formula, metric) in enumerate(recent_entries, start=1):
        lines.append(f"{idx}. {metric_name} = {metric:.6g} | y = {formula}")
    lines.append("")  # Trailing newline for neatness.
    return "\n".join(lines)
