"""Content-addressed cache for LiteLLM chat completions.

The cache is deliberately small and provider agnostic.  It records the exact
model, messages, and non-secret completion arguments that identify a request,
then stores a JSON representation of the response.  Callers decide how a
stored response is reconstructed.

The four modes are:

``off``
    Always make a real request and never access the cache.
``record``
    Always make a real request, then atomically freeze it.  Existing,
    inconsistent content is never overwritten.
``replay``
    Read only.  A missing or damaged entry fails closed without making a
    request.
``read_write``
    Replay a hit; otherwise make one real request and freeze it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeVar


CACHE_SCHEMA_VERSION = "igsr-litellm-completion-v2"
CACHE_MODES = frozenset({"off", "record", "replay", "read_write"})
CACHE_DIR_ENV = "IGSR_LLM_CACHE_DIR"
CACHE_MODE_ENV = "IGSR_LLM_CACHE_MODE"
DETERMINISTIC_REQUEST_SEED_FIELDS = (
    "task_identity",
    "split_identity",
    "seed",
    "stage",
    "iteration",
    "node_id",
    "sibling_slot",
    "retry_ordinal",
    "agent_step",
)


class CompletionCacheError(RuntimeError):
    """Base class for completion-cache failures."""


class CompletionCacheConfigurationError(CompletionCacheError):
    """Raised when cache environment settings are invalid."""


class CompletionCacheMissError(CompletionCacheError):
    """Raised on a replay cache miss."""


class CompletionCacheCorruptionError(CompletionCacheError):
    """Raised when an existing entry cannot be trusted."""


class CompletionCacheConflictError(CompletionCacheError):
    """Raised when a request hash is already frozen with another response."""


_T = TypeVar("_T")

# Do not use a broad substring check for ``token`` or ``key``: parameters such
# as ``max_tokens`` and ``logit_bias`` are generation inputs and must remain in
# the request hash.  These are credential-bearing field names used by LiteLLM
# and common OpenAI-compatible providers.
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "api_secret",
        "api_token",
        "access_key",
        "access_token",
        "auth_token",
        "authorization",
        "azure_ad_token",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "client_secret",
        "credentials",
        "key",
        "password",
        "proxy_authorization",
        "secret",
        "token",
        "vertex_credentials",
    }
)
_SCHEMA_PATH_KEYS = frozenset({"function", "functions", "json_schema", "parameters", "properties", "tools"})


def _normalise_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _is_secret_key(key: str) -> bool:
    normalised = _normalise_key(key)
    return (
        normalised in _SECRET_KEYS
        or normalised.endswith("_api_key")
        or normalised.endswith("_api_secret")
        or normalised.endswith("_access_token")
        or normalised.endswith("_auth_token")
        or normalised.endswith("_client_secret")
        or normalised.endswith("_password")
        or normalised.endswith("_credentials")
    )


def _should_strip_key(path: tuple, key: str) -> bool:
    # A tool/JSON-schema property named ``password`` or ``api_key`` is prompt
    # material, not a provider credential, and therefore must affect the hash.
    # Values under those schema branches are safe to retain for reproducibility.
    if any(_normalise_key(str(part)) in _SCHEMA_PATH_KEYS for part in path):
        return False
    return _is_secret_key(key)


def _json_value(value: Any, *, strip_secrets: bool, _path: tuple = ()) -> Any:
    """Convert a request/response value to deterministic JSON data.

    Cache-enabled execution fails closed for unsupported Python objects instead
    of hashing their potentially unstable ``repr``.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # ``allow_nan=False`` in ``_canonical_json`` rejects non-finite values.
        return value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, (str, int, float, bool)):
                raise CompletionCacheConfigurationError(
                    f"Unsupported non-scalar mapping key in completion cache: {type(raw_key).__name__}"
                )
            key = str(raw_key)
            if strip_secrets and _should_strip_key(_path, key):
                continue
            if key in result:
                raise CompletionCacheConfigurationError(f"JSON key collision in completion cache: {key!r}")
            result[key] = _json_value(raw_value, strip_secrets=strip_secrets, _path=_path + (key,))
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, strip_secrets=strip_secrets, _path=_path) for item in value]

    # LiteLLM/OpenAI responses are Pydantic-like objects.  Request values are
    # normally JSON-native, but supporting ``model_dump`` also covers typed
    # response members without relying on unstable string representations.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        return _json_value(dumped, strip_secrets=strip_secrets, _path=_path)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict(), strip_secrets=strip_secrets, _path=_path)

    raise CompletionCacheConfigurationError(
        f"Unsupported {type(value).__name__} value in completion cache; use JSON-compatible completion arguments"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CompletionCacheConfigurationError(f"Cannot create stable completion-cache JSON: {exc}") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_request_seed(cache_context: Mapping[str, Any]) -> int:
    """Derive the reproducible provider seed for one agent conversation step."""

    identity = {
        field: cache_context.get(field, 0 if field in {"retry_ordinal", "agent_step"} else None)
        for field in DETERMINISTIC_REQUEST_SEED_FIELDS
    }
    for field in ("retry_ordinal", "agent_step"):
        value = identity[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CompletionCacheConfigurationError(f"cache_context.{field} must be a non-negative integer")
    return int.from_bytes(hashlib.sha256(_canonical_json(identity).encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


class CompletionCache:
    """A filesystem-backed, content-addressed completion cache."""

    def __init__(self, mode: str = "off", cache_dir: Optional[Path] = None):
        normalised_mode = mode.strip().lower()
        if normalised_mode not in CACHE_MODES:
            choices = ", ".join(sorted(CACHE_MODES))
            raise CompletionCacheConfigurationError(
                f"Invalid {CACHE_MODE_ENV}={mode!r}; expected one of: {choices}"
            )
        if normalised_mode != "off" and cache_dir is None:
            raise CompletionCacheConfigurationError(
                f"{CACHE_DIR_ENV} is required when cache mode is {normalised_mode!r}"
            )
        self.mode = normalised_mode
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        # Ordered, provider-private audit trail for the owning agent.  Callers
        # may persist this after a stage to prove that paired arms consumed the
        # same request hashes in the same order.
        self.access_log: List[Dict[str, Any]] = []

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None) -> "CompletionCache":
        """Build cache configuration from ``IGSR_LLM_CACHE_*`` variables."""

        source = os.environ if environ is None else environ
        mode = source.get(CACHE_MODE_ENV, "off") or "off"
        raw_dir = source.get(CACHE_DIR_ENV)
        cache_dir = Path(raw_dir) if raw_dir else None
        return cls(mode=mode, cache_dir=cache_dir)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def sanitized_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        completion_kwargs: Mapping[str, Any],
        cache_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the stable, non-secret request identity stored on disk."""

        # Messages are deliberately not secret-filtered: the complete prompt is
        # part of a reproducible generation request.  Only credential-bearing
        # completion argument fields are removed.
        request = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model": model,
            "messages": _json_value(messages, strip_secrets=False),
            "completion_kwargs": _json_value(completion_kwargs, strip_secrets=True),
            # Experiment provenance used only for cache identity.  It is never
            # forwarded to LiteLLM/the provider.
            "cache_context": _json_value(cache_context or {}, strip_secrets=False),
        }
        # Validate finite floats and JSON serialisability now, before an API call.
        _canonical_json(request)
        return request

    def request_hash(
        self,
        model: str,
        messages: List[Dict[str, str]],
        completion_kwargs: Mapping[str, Any],
        cache_context: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Calculate the SHA-256 key for a complete sanitized request."""

        return _sha256_json(self.sanitized_request(model, messages, completion_kwargs, cache_context))

    def entry_path(self, request_hash: str) -> Path:
        """Map a request hash to its sharded JSON path."""

        if self.cache_dir is None:
            raise CompletionCacheConfigurationError("Completion cache directory is not configured")
        if len(request_hash) != 64 or any(char not in "0123456789abcdef" for char in request_hash):
            raise CompletionCacheConfigurationError(f"Invalid completion-cache request hash: {request_hash!r}")
        return self.cache_dir / request_hash[:2] / request_hash[2:4] / f"{request_hash}.json"

    def _read_entry(self, path: Path, expected_request: Dict[str, Any], expected_hash: str) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                entry = json.load(handle)
        except FileNotFoundError:
            raise CompletionCacheMissError(f"Completion cache miss for request {expected_hash}") from None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CompletionCacheCorruptionError(f"Cannot read completion cache entry {path}: {exc}") from exc

        if not isinstance(entry, dict):
            raise CompletionCacheCorruptionError(f"Completion cache entry is not an object: {path}")

        integrity = entry.get("entry_integrity_sha256")
        unsigned_entry = {key: value for key, value in entry.items() if key != "entry_integrity_sha256"}
        if not isinstance(integrity, str) or _sha256_json(unsigned_entry) != integrity:
            raise CompletionCacheCorruptionError(f"Completion cache integrity check failed: {path}")
        if entry.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise CompletionCacheCorruptionError(f"Unsupported completion cache schema in {path}")
        if entry.get("request_hash") != expected_hash:
            raise CompletionCacheCorruptionError(f"Completion cache request hash mismatch: {path}")
        if entry.get("request") != expected_request or _sha256_json(entry.get("request")) != expected_hash:
            raise CompletionCacheCorruptionError(f"Completion cache request identity mismatch: {path}")
        if not isinstance(entry.get("response"), dict):
            raise CompletionCacheCorruptionError(f"Completion cache response is not an object: {path}")
        return entry

    def load(
        self,
        model: str,
        messages: List[Dict[str, str]],
        completion_kwargs: Mapping[str, Any],
        cache_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load and validate a cached response as plain JSON data."""

        request = self.sanitized_request(model, messages, completion_kwargs, cache_context)
        request_hash = _sha256_json(request)
        entry = self._read_entry(self.entry_path(request_hash), request, request_hash)
        return entry["response"]

    def store(
        self,
        model: str,
        messages: List[Dict[str, str]],
        completion_kwargs: Mapping[str, Any],
        response: Any,
        cache_context: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Atomically freeze a response and return its request hash.

        A hard-link publish step provides create-if-absent semantics: a writer
        can never replace an existing file.  Concurrent identical writers are
        accepted, while differing responses raise a conflict.
        """

        request = self.sanitized_request(model, messages, completion_kwargs, cache_context)
        request_hash = _sha256_json(request)
        path = self.entry_path(request_hash)
        response_data = _json_value(response, strip_secrets=True)
        if not isinstance(response_data, dict):
            raise CompletionCacheConfigurationError("LiteLLM completion response must serialize to a JSON object")

        unsigned_entry = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "request_hash": request_hash,
            "request": request,
            "response": response_data,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        entry = dict(unsigned_entry)
        entry["entry_integrity_sha256"] = _sha256_json(unsigned_entry)
        encoded = (_canonical_json(entry) + "\n").encode("utf-8")

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{request_hash}.", suffix=".tmp", dir=str(path.parent))
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Atomic create-if-absent.  Unlike os.replace(), this cannot
                # overwrite a cache entry produced by another process.
                os.link(str(temporary_path), str(path))
            except FileExistsError:
                existing = self._read_entry(path, request, request_hash)
                if existing["response"] != response_data:
                    raise CompletionCacheConflictError(
                        f"Completion cache request {request_hash} is already frozen with a different response"
                    ) from None
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        return request_hash

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        completion_kwargs: Mapping[str, Any],
        make_request: Callable[[], _T],
        restore_response: Callable[[Dict[str, Any]], _T],
        cache_context: Optional[Mapping[str, Any]] = None,
    ) -> _T:
        """Execute a request according to the configured cache mode."""

        request_hash = self.request_hash(model, messages, completion_kwargs, cache_context)

        def record_access(source: str) -> None:
            self.access_log.append(
                {
                    "sequence": len(self.access_log),
                    "request_hash": request_hash,
                    "cache_mode": self.mode,
                    "source": source,
                }
            )

        if self.mode == "off":
            response = make_request()
            record_access("provider_off")
            return response
        if self.mode == "replay":
            response = restore_response(self.load(model, messages, completion_kwargs, cache_context))
            record_access("cache_hit")
            return response
        if self.mode == "read_write":
            try:
                response = restore_response(self.load(model, messages, completion_kwargs, cache_context))
                record_access("cache_hit")
                return response
            except CompletionCacheMissError:
                pass

        # ``record`` intentionally makes a real request.  It is used to freeze
        # the first arm of a paired experiment, not to silently consume a hit.
        response = make_request()
        stored_hash = self.store(model, messages, completion_kwargs, response, cache_context)
        if stored_hash != request_hash:  # defensive: both derive from the same sanitized request
            raise CompletionCacheCorruptionError(
                f"Completion cache hash changed while storing: {request_hash} != {stored_hash}"
            )
        record_access("provider_record" if self.mode == "record" else "provider_read_write_miss")
        return response
