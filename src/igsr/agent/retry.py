"""Retry handling for API calls with configurable logging and backoff."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, TypeVar

from litellm.exceptions import RateLimitError
from openai import OpenAIError
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Type variable for the decorated function:
F = TypeVar("F", bound=Callable[..., Any])


class Logger(Protocol):
    """Protocol defining the logging interface we need."""

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message with structured data.

        Args:
            message: The log message.
            **kwargs: Additional structured data to log.
        """
        ...


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Args:
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.
        min_wait (float, optional): Minimum wait time in seconds. Defaults to 1.
        max_wait (float, optional): Maximum wait time in seconds. Defaults to 60.
        logger (Optional[Logger], optional): Logger instance for retry events. Defaults to None.
    """

    max_retries: int = 5
    min_wait: float = 1
    max_wait: float = 60
    logger: Optional[Logger] = None


class RetryHandler:
    """Handles retrying of operations with configurable backoff and logging.

    Args:
        config (RetryConfig): Configuration for retry behavior.
    """

    def __init__(self, config: RetryConfig) -> None:
        """Initialize the retry handler."""
        self.config = config

    def _should_retry(self, error: Exception) -> bool:
        """Determine if the error should trigger a retry.

        Args:
            error: The exception that was raised.

        Returns:
            bool: True if should retry, False otherwise.
        """
        if isinstance(error, RateLimitError):
            return True

        if isinstance(error, OpenAIError):
            # Retry on server errors (5xx) and specific client errors:
            if hasattr(error, "status_code"):
                return error.status_code >= 500 or error.status_code in {408, 429}  # type: ignore

        if isinstance(error, RateLimitError):
            return True

        return False

    def _before_sleep(self, retry_state: RetryCallState) -> None:
        """Called before each sleep between retries.

        Args:
            retry_state: Current state of the retry operation.
        """
        if retry_state.outcome is None:
            raise ValueError("Retry state outcome is None")
        if retry_state.next_action is None:
            raise ValueError("Retry state next_action is None")
        if retry_state.fn is None:
            raise ValueError("Retry state fn is None")

        error = retry_state.outcome.exception()

        if self.config.logger is not None:
            # Use structured logging if logger is available:
            self.config.logger.warning(
                "retrying_api_call",
                attempt=retry_state.attempt_number,
                next_attempt_in=retry_state.next_action.sleep,
                error=str(error),
            )
        else:
            # Fallback to standard logging if no structured logger:
            logging.warning(
                "Retrying %s in %.1f seconds as it raised %s: %s",
                f"{retry_state.fn.__qualname__}",
                retry_state.next_action.sleep,
                error.__class__.__name__,
                str(error),
            )

    def with_retries(self) -> Callable[[F], F]:
        """Create a retry decorator for functions that should be retried on failure.

        Returns:
            A decorator that adds retry functionality to the decorated function.

        Example:
            ```python
            handler = RetryHandler(RetryConfig())

            @handler.with_retries()
            def my_function():
                # Function implementation.
                pass
            ```
        """

        def decorator(func: F) -> F:
            retry_decorator = retry(
                retry=retry_if_exception_type(OpenAIError),
                stop=stop_after_attempt(self.config.max_retries),
                wait=wait_exponential(min=self.config.min_wait, max=self.config.max_wait),
                before_sleep=self._before_sleep,
                retry_error_callback=lambda retry_state: retry_state.outcome.result(),  # type: ignore
            )
            return retry_decorator(func)

        return decorator
