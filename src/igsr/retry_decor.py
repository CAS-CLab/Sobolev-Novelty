import time
from contextlib import AbstractContextManager
from typing import Any, Callable, Optional, Tuple, Type


class retry(AbstractContextManager):
    """
    Re-usable context manager that retries a callable up to *attempts* times.

    Args:
        attempts (int) - maximum tries (≥ 1)
        delay (float) - initial pause between retries (seconds)
        backoff (float) - multiplier applied to *delay* after every failure
        exceptions (tuple) - exception types that trigger a retry
        on_retry (func) - optional hook (exc, remaining) → None

    Example:
    >>> with retry(attempts=5, delay=0.5, backoff=2) as r:
    ...     result = r(flaky_network_call, url)
    """

    def __init__(
        self,
        attempts: int = 3,
        delay: float = 0.0,
        backoff: float = 1.0,
        exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        on_retry: Optional[Callable[[BaseException, int], Any]] = None,
    ):
        if attempts < 1:
            raise ValueError("attempts must be ≥ 1")
        self.attempts = attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
        self.on_retry = on_retry or (lambda exc, remaining: None)

    # -- context-manager protocol ------------------------------------------------
    def __enter__(self):
        def _run(func: Callable, *args, **kwargs):
            tries_left, wait = self.attempts, self.delay
            while True:
                try:
                    return func(*args, **kwargs)
                except self.exceptions as exc:
                    tries_left -= 1
                    self.on_retry(exc, tries_left)  # optional side-effects
                    if tries_left == 0:
                        raise  # bubble up final failure
                    if wait:
                        time.sleep(wait)
                        wait *= self.backoff

        return _run

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: D401
        # The context wrapper never suppresses errors raised *outside* `_run`.
        return False


# Convenience decorator:
def retryable(**retry_kwargs):
    """Decorator variant using the same semantics as the context manager."""

    def _decorator(fn):
        def _wrapper(*args, **kwargs):
            with retry(**retry_kwargs) as r:
                return r(fn, *args, **kwargs)

        return _wrapper

    return _decorator
