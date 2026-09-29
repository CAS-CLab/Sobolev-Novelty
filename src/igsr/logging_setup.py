"""Structured logging setup for igsr.

All loggers live under the ``"igsr"`` namespace: those created via ``get_logger`` and the
Hydra ``logging.logger_name`` (``igsr.experiments.discovery.*``) are children of it, and
``configure_logging`` attaches the console/file handlers to that parent.

Public API: ``configure_logging``, ``get_logger``, ``log_session_id``, ``log_file``.
"""

import copy
import logging
import os
import sys
from datetime import datetime
from typing import Any, List, Optional, Union

import structlog
from omegaconf import OmegaConf
from structlog.types import Processor

# Logger namespace the file/console handlers are attached to (Hydra configs set
# logger_name to "igsr.experiments.discovery.*", i.e. children of this).
_LOGGER_NAMESPACE = "igsr"

_JSON_FMT = structlog.stdlib.ProcessorFormatter(
    processor=structlog.processors.JSONRenderer(),
)
_CONSOLE_FMT = structlog.stdlib.ProcessorFormatter(
    processor=structlog.dev.ConsoleRenderer(colors=True, sort_keys=True),
)


def _get_log_level(level: Union[str, int]) -> int:
    """Convert string log level to numeric value if needed."""
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper())


def configure_logging(
    log_level: Union[str, int] = "INFO",
    timestamp_key: str = "timestamp",
    log_file_path: Optional[str] = None,
) -> None:
    """Configure global structured-logging settings.

    Args:
        log_level (Union[str, int]): Logging level. Defaults to "INFO".
        timestamp_key (str): Key to use for timestamp in log output. Defaults to "timestamp".
        log_file_path (Optional[str]): Path to JSON log file. Logs will be written as JSON to this file
            if provided. Defaults to None.

    Example:
        >>> configure_logging("INFO")
        >>> configure_logging("DEBUG")  # Pretty printing for console, JSON for file
        >>> configure_logging(logging.INFO, log_file_path="/var/log/app.json")
        >>> configure_logging(20)  # INFO level
    """
    # Convert string level to numeric if needed:
    numeric_level = _get_log_level(log_level)

    # First set our logger to the desired level:
    namespace_logger = logging.getLogger(_LOGGER_NAMESPACE)
    namespace_logger.setLevel(numeric_level)

    # Remove any existing handlers
    namespace_logger.handlers = []

    # Define shared processors for both outputs
    shared_processors: List[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", key=timestamp_key),
        structlog.processors.StackInfoRenderer(),
    ]

    # Configure console output with pretty printing
    console_handler = logging.StreamHandler(sys.stdout)
    namespace_logger.addHandler(console_handler)

    # Set up file handler with JSON output if needed
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        namespace_logger.addHandler(file_handler)

    # Configure structlog to handle the rendering
    structlog.configure(
        processors=shared_processors
        + [
            # This processor checks which handler is active and formats accordingly
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Apply formatters to handlers
    console_handler.setFormatter(_CONSOLE_FMT)
    if log_file_path:
        file_handler.setFormatter(_JSON_FMT)

    # Set root logger to WARNING to suppress third-party logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)


def _attach_file_handler(logger: logging.Logger, path: str, mode: str = "a") -> None:
    """
    Attach a JSON-rendering FileHandler to `logger` that writes to `path`.

    If a handler for that exact path already exists we don't add a second one.
    """
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == path:
            return  # Already attached.

    fh = logging.FileHandler(path, mode=mode)
    fh.setFormatter(_JSON_FMT)
    logger.addHandler(fh)


def get_logger(
    name: str,
    log_file_path: Optional[str] = None,
    mode: str = "a",
    propagate: bool = True,
    **initial_context: Any,
) -> structlog.stdlib.BoundLogger:
    """Get a logger with the given name and initial context.

    Args:
        name (str): Logger name (e.g., "igsr.experiments.discovery.igsr").
        log_file_path (Optional[str]): Path to JSON log file. Logs will be written as JSON to this file
            if provided. Defaults to None.
        mode (str): Mode to open the log file in. Relevant only when `log_file_path` is provided.
            Defaults to "a".
        propagate (bool): Whether to propagate messages to the root logger. Relevant only when
            `log_file_path` is provided. Defaults to True.
        **initial_context: Additional context to bind to the logger. Defaults to an empty dictionary.

    Returns:
        structlog.stdlib.BoundLogger: A structured logger instance.

    Example:
        >>> logger = get_logger("igsr.experiments.discovery.icl", seed=0)
        >>> logger.info("run_start")
    """
    std_logger = logging.getLogger(name)

    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        _attach_file_handler(std_logger, log_file_path, mode=mode)
        std_logger.propagate = propagate

    return structlog.wrap_logger(std_logger).bind(**initial_context)


def safe_cfg_yaml(cfg: Any) -> str:
    """Render an OmegaConf config to YAML (interpolations resolved) with the LLM API key redacted.

    Use this instead of ``OmegaConf.to_yaml(cfg, resolve=True)`` anywhere the resolved config is
    logged or printed. ``cfg.llm.api_key`` resolves from the environment (``${oc.env:...}``), so a
    raw resolved dump would leak the secret into stdout / JSONL logs. The key is replaced with a
    placeholder on a copy *before* resolving, so the secret is never materialised.
    """
    c = copy.deepcopy(cfg)
    try:
        OmegaConf.set_struct(c, False)
        if "llm" in c and "api_key" in c.llm:
            c.llm.api_key = "***REDACTED***"
    except Exception:
        # Redaction is best-effort; never let logging-safety crash the run.
        pass
    return OmegaConf.to_yaml(c, resolve=True)


# Default configuration with console and file output.
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_session_id = f"igsr_session_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
log_file = os.path.join(log_dir, f"{log_session_id}.jsonl")

configure_logging(log_level="DEBUG", log_file_path=log_file)
