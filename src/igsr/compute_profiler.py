from typing import Dict, Iterable


class ComputeProfiler:
    """
    Simple accumulator for a fixed set of metric names.

    Example
    -------
    >>> profiler = ComputeProfiler({"wall_clock_optim", "num_steps"})
    >>> profiler.accumulate("wall_clock_optim", 0.10)
    >>> profiler.accumulate("wall_clock_optim", 1.23)
    >>> profiler.accumulate("num_steps", 5)
    >>> print(profiler.as_dict())
    {'wall_clock_optim': 1.33, 'num_steps': 5.0}
    """

    def __init__(self, metric_names: Iterable[str]) -> None:
        # Initialise every allowed metric at zero (float).
        self._metrics: Dict[str, float] = {name: 0.0 for name in metric_names}

    def accumulate(self, metric: str, value: float) -> None:
        """
        Add *value* to the current total for *metric*.

        Raises
        ------
        KeyError
            If *metric* was not defined at construction time.
        """
        if metric not in self._metrics:
            raise KeyError(f"Metric '{metric}' is not registered. " f"Allowed metrics: {list(self._metrics.keys())}")
        self._metrics[metric] += float(value)

    # ---- Helper / convenience methods -------------------------------------

    def as_dict(self) -> Dict[str, float]:
        """Return a copy of the internal metric dictionary."""
        return dict(self._metrics)

    def __repr__(self) -> str:
        # Useful for quick interactive inspection.
        return f"{self.__class__.__name__}({self._metrics})"
