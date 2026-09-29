"""Tests for deterministic LiteLLM completion recording and replay."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from igsr.agent.completion_cache import (
    CACHE_DIR_ENV,
    CACHE_MODE_ENV,
    CompletionCache,
    CompletionCacheConflictError,
    CompletionCacheCorruptionError,
    CompletionCacheMissError,
)


MESSAGES: List[Dict[str, str]] = [
    {"role": "system", "content": "Return one symbolic term."},
    {"role": "user", "content": "Variables: x0, x1"},
]
RESPONSE: Dict[str, Any] = {
    "id": "chatcmpl-fixed",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "x0 + x1"},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
}


def test_record_then_replay_same_request_makes_zero_replay_calls(tmp_path: Path) -> None:
    """A replay hit reconstructs the frozen response without an API call."""

    calls = []
    record = CompletionCache(mode="record", cache_dir=tmp_path)

    def first_request() -> Dict[str, Any]:
        calls.append("record")
        return RESPONSE

    assert record.complete(
        "openai/test-model", MESSAGES, {"temperature": 0.5}, first_request, lambda data: data
    ) == RESPONSE
    assert calls == ["record"]

    replay = CompletionCache(mode="replay", cache_dir=tmp_path)

    def forbidden_request() -> Dict[str, Any]:
        calls.append("replay-api")
        raise AssertionError("replay must not make an API call")

    restored = replay.complete(
        "openai/test-model", MESSAGES, {"temperature": 0.5}, forbidden_request, lambda data: data
    )
    assert restored == RESPONSE
    assert calls == ["record"]
    assert record.access_log == [
        {
            "sequence": 0,
            "request_hash": record.request_hash(
                "openai/test-model", MESSAGES, {"temperature": 0.5}
            ),
            "cache_mode": "record",
            "source": "provider_record",
        }
    ]
    assert replay.access_log == [
        {
            "sequence": 0,
            "request_hash": replay.request_hash(
                "openai/test-model", MESSAGES, {"temperature": 0.5}
            ),
            "cache_mode": "replay",
            "source": "cache_hit",
        }
    ]


def test_prompt_temperature_model_and_api_base_change_key(tmp_path: Path) -> None:
    """Every non-secret generation request field participates in the key."""

    cache = CompletionCache(mode="record", cache_dir=tmp_path)
    base_kwargs = {"temperature": 0.5, "api_base": "http://127.0.0.1:11434/v1", "max_tokens": 64}
    base = cache.request_hash("openai/test-model", MESSAGES, base_kwargs)
    changed_prompt = cache.request_hash(
        "openai/test-model",
        MESSAGES + [{"role": "user", "content": "Try another term"}],
        base_kwargs,
    )
    changed_temperature = cache.request_hash(
        "openai/test-model", MESSAGES, {**base_kwargs, "temperature": 0.6}
    )
    changed_model = cache.request_hash("openai/other-model", MESSAGES, base_kwargs)
    changed_api_base = cache.request_hash(
        "openai/test-model", MESSAGES, {**base_kwargs, "api_base": "http://localhost:8000/v1"}
    )

    assert len({base, changed_prompt, changed_temperature, changed_model, changed_api_base}) == 5


def test_secret_named_tool_schema_fields_still_affect_key(tmp_path: Path) -> None:
    """Schema property names are generation inputs, not provider credentials."""

    cache = CompletionCache(mode="record", cache_dir=tmp_path)
    first_tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"api_key": {"type": "string", "description": "first schema description"}},
                },
            },
        }
    ]
    second_tools = [
        {
            **first_tools[0],
            "function": {
                **first_tools[0]["function"],
                "parameters": {
                    "type": "object",
                    "properties": {"api_key": {"type": "string", "description": "second schema description"}},
                },
            },
        }
    ]
    first = cache.request_hash("openai/test-model", MESSAGES, {"tools": first_tools})
    second = cache.request_hash("openai/test-model", MESSAGES, {"tools": second_tools})
    assert first != second


def test_cache_context_distinguishes_sibling_task_seed_and_model_revision(tmp_path: Path) -> None:
    """Provider-private experiment identity separates stochastic sampling slots."""

    cache = CompletionCache(mode="record", cache_dir=tmp_path)
    base_context = {
        "protocol": "igsr-paired-completion-v1",
        "task_identity": "BPG10",
        "split_identity": "split-sha-1",
        "seed": 20260805,
        "stage": "propose",
        "node_id": "node_0_0",
        "sibling_slot": 0,
        "model_revision": "revision-a",
    }
    variants = [
        base_context,
        {**base_context, "sibling_slot": 1},
        {**base_context, "task_identity": "CRK11"},
        {**base_context, "seed": 20260806},
        {**base_context, "model_revision": "revision-b"},
    ]
    hashes = {
        cache.request_hash(
            "openai/test-model",
            MESSAGES,
            {"temperature": 0.5},
            cache_context=context,
        )
        for context in variants
    }
    assert len(hashes) == len(variants)


def test_replay_miss_fails_closed_without_request(tmp_path: Path) -> None:
    """A replay miss raises and never invokes the supplied request callable."""

    called = False

    def forbidden_request() -> Dict[str, Any]:
        nonlocal called
        called = True
        return RESPONSE

    cache = CompletionCache(mode="replay", cache_dir=tmp_path)
    with pytest.raises(CompletionCacheMissError, match="cache miss"):
        cache.complete(
            "openai/test-model", MESSAGES, {"temperature": 0.5}, forbidden_request, lambda data: data
        )
    assert called is False
    assert cache.access_log == []


def test_credentials_are_neither_stored_nor_hashed(tmp_path: Path) -> None:
    """Credential changes do not change identity and secret values never reach disk."""

    cache = CompletionCache(mode="record", cache_dir=tmp_path)
    first_kwargs = {
        "api_key": "first-super-secret-api-key",
        "api_base": "http://127.0.0.1:11434/v1",
        "temperature": 0.25,
        "max_tokens": 77,
        "extra_headers": {"Authorization": "Bearer first-secret-token", "X-Experiment": "paired-v1"},
    }
    second_kwargs = {
        **first_kwargs,
        "api_key": "second-super-secret-api-key",
        "extra_headers": {"Authorization": "Bearer second-secret-token", "X-Experiment": "paired-v1"},
    }
    assert cache.request_hash("openai/test-model", MESSAGES, first_kwargs) == cache.request_hash(
        "openai/test-model", MESSAGES, second_kwargs
    )

    cache.store("openai/test-model", MESSAGES, first_kwargs, RESPONSE)
    entry_text = next(tmp_path.rglob("*.json")).read_text(encoding="utf-8")
    assert "first-super-secret-api-key" not in entry_text
    assert "first-secret-token" not in entry_text
    assert "api_key" not in entry_text
    assert "Authorization" not in entry_text
    assert "http://127.0.0.1:11434/v1" in entry_text
    assert '"max_tokens":77' in entry_text
    assert "paired-v1" in entry_text

    # A different credential can replay the same generation request.
    replay = CompletionCache(mode="replay", cache_dir=tmp_path)
    assert replay.load("openai/test-model", MESSAGES, second_kwargs) == RESPONSE


def test_valid_json_tampering_is_detected_as_corruption(tmp_path: Path) -> None:
    """The entry checksum catches corruption even when JSON still parses."""

    cache = CompletionCache(mode="record", cache_dir=tmp_path)
    request_hash = cache.store("openai/test-model", MESSAGES, {"temperature": 0.5}, RESPONSE)
    entry_path = cache.entry_path(request_hash)
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["response"]["choices"][0]["message"]["content"] = "tampered"
    entry_path.write_text(json.dumps(entry), encoding="utf-8")

    replay = CompletionCache(mode="replay", cache_dir=tmp_path)
    with pytest.raises(CompletionCacheCorruptionError, match="integrity check failed"):
        replay.load("openai/test-model", MESSAGES, {"temperature": 0.5})


def test_existing_different_response_is_never_overwritten(tmp_path: Path) -> None:
    """Record mode preserves the first response for a content-addressed key."""

    cache = CompletionCache(mode="record", cache_dir=tmp_path)
    request_hash = cache.store("openai/test-model", MESSAGES, {"temperature": 0.5}, RESPONSE)
    original_bytes = cache.entry_path(request_hash).read_bytes()
    different = {
        **RESPONSE,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "x0 * x1"},
            }
        ],
    }
    with pytest.raises(CompletionCacheConflictError, match="different response"):
        cache.store("openai/test-model", MESSAGES, {"temperature": 0.5}, different)
    assert cache.entry_path(request_hash).read_bytes() == original_bytes


def test_litellm_agent_round_trip_preserves_choices_and_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LiteLLM replay returns the object interface consumed by BaseOpenAIAgent."""

    litellm = pytest.importorskip("litellm")
    from igsr.agent.agent import LiteLLMAgent  # pylint: disable=C0415

    real_calls = []

    def fake_completion(**kwargs: Any) -> Any:
        real_calls.append(kwargs)
        return litellm.ModelResponse(
            id="chatcmpl-fixed",
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "x0 + x1"},
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(CACHE_MODE_ENV, "record")
    cache_context = {
        "protocol": "test-v1",
        "task_identity": "task-1",
        "split_identity": "split-1",
        "seed": 7,
        "stage": "propose",
        "node_id": "node_0_0",
        "sibling_slot": 0,
        "model_revision": "revision-1",
    }
    recorder = LiteLLMAgent(
        task_description="test",
        model="openai/test-model",
        cache_context=cache_context,
    )
    recorder._create_chat_completion(MESSAGES, {"temperature": 0.5})  # pylint: disable=W0212
    assert len(real_calls) == 1
    assert "cache_context" not in real_calls[0]

    monkeypatch.setenv(CACHE_MODE_ENV, "replay")
    replayer = LiteLLMAgent(
        task_description="test",
        model="openai/test-model",
        cache_context=cache_context,
    )
    replayed = replayer._create_chat_completion(MESSAGES, {"temperature": 0.5})  # pylint: disable=W0212
    assert len(real_calls) == 1
    assert replayed.choices[0].message.content == "x0 + x1"
    assert replayed.usage.to_dict()["total_tokens"] == 14
