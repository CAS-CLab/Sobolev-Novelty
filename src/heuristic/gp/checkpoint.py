"""Crash-safe, content-addressed generation checkpoints for controlled GP."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def append_jsonl(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


class GenerationCheckpointWriter:
    """Commit only complete generations; never mutate checkpoint content."""

    def __init__(self, output_dir: Path, floor_seconds: float) -> None:
        self.output_dir = Path(output_dir)
        self.floor_seconds = float(floor_seconds)
        if self.floor_seconds <= 0:
            raise ValueError("floor_seconds must be positive")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def commit(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        generation = int(snapshot["generation"])
        wall_time = float(snapshot["wall_time"])
        payload = dict(snapshot)
        payload["complete_generation"] = True
        payload["checkpoint_schema"] = "controlled_gp_generation_v1"
        digest = sha256_json(payload)
        checkpoint = self.checkpoint_dir / f"{digest}.json"
        data = canonical_json_bytes(payload)
        try:
            with checkpoint.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.checkpoint_dir)
        except FileExistsError:
            if checkpoint.read_bytes() != data:
                raise RuntimeError(f"Checkpoint hash collision at {checkpoint}")
        pointer = {
            "checkpoint_schema": "controlled_gp_pointer_v1",
            "checkpoint_sha256": digest,
            "checkpoint_path": str(checkpoint.resolve()),
            "generation": generation,
            "wall_time": wall_time,
        }
        atomic_write_json(self.output_dir / "latest_generation_checkpoint.json", pointer)
        if wall_time <= self.floor_seconds:
            floor_pointer = dict(pointer)
            floor_pointer["target_seconds"] = self.floor_seconds
            atomic_write_json(self.output_dir / "latest_floor_checkpoint.json", floor_pointer)
        append_jsonl(self.output_dir / "generation_records.jsonl", payload)
        return pointer


def load_checkpoint_pointer(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = json.loads(Path(path).read_text(encoding="utf-8"))
    checkpoint = Path(pointer["checkpoint_path"])
    data = checkpoint.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != pointer["checkpoint_sha256"]:
        raise RuntimeError(
            f"Checkpoint digest mismatch: expected={pointer['checkpoint_sha256']} actual={actual}"
        )
    return pointer, json.loads(data)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

