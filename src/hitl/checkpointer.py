"""A LangGraph checkpointer backed by PersistKV.

Every `put` and `put_writes` commits as a single PersistKV WriteBatch: one WAL record, one
fsync. After a crash a checkpoint is either fully present (metadata, channel blobs and the
`latest` pointer) or fully absent.

Key layout, with `T` and `N` being fixed-size hashes of the thread id and namespace, since
PersistKV keys are limited to 128 bytes:

    t/T                  thread id
    n/T/N                checkpoint namespace
    l/T/N                id of the newest checkpoint
    c/T/N/<ckpt id>      checkpoint, metadata, parent id
    b/T/N/<hash>         channel value, addressed by (channel, version)
    w/T/N/<ckpt id>/...  pending writes of a checkpoint
    x<any key above>#i   chunk i of a value too large for one PersistKV value

Checkpoint ids sort chronologically, so `c/` keys are in time order.

The Python binding has no reverse scan, so "newest checkpoint" reads the `l/` pointer. It is
updated in the same batch as the checkpoint, so the two cannot disagree after a crash.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import ormsgpack
import persistkv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from hitl.crash import crashpoint

# Leave room in each PersistKV value for the one-byte framing tag.
CHUNK_SIZE = persistkv.MAX_VALUE_SIZE - 1
_INLINE = b"\x00"
_CHUNKED = b"\x01"
_THREAD_PREFIXES = (b"c/", b"b/", b"w/", b"l/", b"n/")
_LOCK_STRIPES = 64


def _hash(data: str | bytes, size: int = 12) -> bytes:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.blake2b(raw, digest_size=size).hexdigest().encode("ascii")


def _pack(value: Any) -> bytes:
    return ormsgpack.packb(value)


def _unpack(data: bytes) -> Any:
    return ormsgpack.unpackb(data)


class PersistKVSaver(BaseCheckpointSaver[str]):
    """Durable LangGraph checkpointer. Takes an open `persistkv.DB`, so the caller can commit
    other records (such as a ledger or audit log) to the same database.

    A database may only be open in one process at a time, because PersistKV has no file lock.
    """

    def __init__(self, db: persistkv.DB, *, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self.db = db
        # Serializes read-then-write sequences per thread without holding a global lock across
        # an fsync, so that concurrent threads can still share fsyncs through group commit.
        self._locks = [threading.Lock() for _ in range(_LOCK_STRIPES)]

    # ------------------------------------------------------------------ keys and framing

    def _lock(self, thread_id: str) -> threading.Lock:
        return self._locks[int(_hash(thread_id, 4), 16) % _LOCK_STRIPES]

    @staticmethod
    def _scope(thread_id: str, checkpoint_ns: str) -> bytes:
        return _hash(thread_id) + b"/" + _hash(checkpoint_ns, 8)

    @staticmethod
    def _chunk_key(key: bytes, index: int) -> bytes:
        return b"x" + key + b"#%05d" % index

    def _put_value(self, batch: persistkv.WriteBatch, key: bytes, payload: bytes) -> None:
        if len(payload) <= CHUNK_SIZE:
            batch.put(key, _INLINE + payload)
            return
        count = 0
        for start in range(0, len(payload), CHUNK_SIZE):
            batch.put(self._chunk_key(key, count), payload[start : start + CHUNK_SIZE])
            count += 1
        batch.put(key, _CHUNKED + _pack(count))

    def _decode_value(self, key: bytes, raw: bytes | None) -> bytes | None:
        if raw is None:
            return None
        if raw[:1] == _INLINE:
            return raw[1:]
        parts = []
        for index in range(_unpack(raw[1:])):
            part = self.db.get(self._chunk_key(key, index))
            if part is None:
                # Chunks commit in one batch with their manifest, so this is real corruption.
                raise persistkv.CorruptionError(f"missing chunk {index} of {key!r}")
            parts.append(part)
        return b"".join(parts)

    def _get_value(self, key: bytes) -> bytes | None:
        return self._decode_value(key, self.db.get(key))

    # ------------------------------------------------------------------ reads

    def _load_channel_values(self, scope: bytes, versions: ChannelVersions) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for channel, version in versions.items():
            payload = self._get_value(self._blob_key(scope, channel, version))
            if payload is None:
                continue
            type_, data = _unpack(payload)
            if type_ != "empty":
                values[channel] = self.serde.loads_typed((type_, data))
        return values

    @staticmethod
    def _blob_key(scope: bytes, channel: str, version: str | int | float) -> bytes:
        identity = _pack([channel, str(version)])
        return b"b/" + scope + b"/" + _hash(identity, 16)

    def _load_writes(
        self, scope: bytes, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        prefix = b"w/" + scope + b"/" + checkpoint_id.encode("ascii") + b"/"
        rows = []
        for key, raw in self.db.scan_prefix(prefix):
            task_id, channel, type_, data, task_path, idx = _unpack(self._decode_value(key, raw))
            rows.append((writes_sort_key(task_path, task_id, idx), task_id, channel, type_, data))
        rows.sort(key=lambda row: row[0])
        return [
            (task_id, channel, self.serde.loads_typed((type_, data)))
            for _, task_id, channel, type_, data in rows
        ]

    def _build_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        record: tuple[Any, ...],
        metadata: CheckpointMetadata | None = None,
    ) -> CheckpointTuple:
        checkpoint_type, checkpoint_data, metadata_type, metadata_data, parent_id = record
        scope = self._scope(thread_id, checkpoint_ns)
        checkpoint: Checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_data))
        if metadata is None:
            metadata = self.serde.loads_typed((metadata_type, metadata_data))
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_channel_values(
                    scope, checkpoint["channel_versions"]
                ),
            },
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=self._load_writes(scope, checkpoint_id),
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        scope = self._scope(thread_id, checkpoint_ns)
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None:
            latest = self.db.get(b"l/" + scope)
            if latest is None:
                return None
            checkpoint_id = latest.decode("ascii")
        key = b"c/" + scope + b"/" + checkpoint_id.encode("ascii")
        payload = self._get_value(key)
        if payload is None:
            return None
        return self._build_tuple(thread_id, checkpoint_ns, checkpoint_id, _unpack(payload))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is not None:
            thread_ids = [config["configurable"]["thread_id"]]
            wanted_ns = config["configurable"].get("checkpoint_ns")
            wanted_id = get_checkpoint_id(config)
        else:
            thread_ids = [raw.decode("utf-8") for _, raw in self.db.scan_prefix(b"t/")]
            wanted_ns = wanted_id = None
        before_id = get_checkpoint_id(before) if before else None

        for thread_id in thread_ids:
            namespaces = [
                raw.decode("utf-8")
                for _, raw in self.db.scan_prefix(b"n/" + _hash(thread_id) + b"/")
            ]
            for checkpoint_ns in namespaces:
                if wanted_ns is not None and checkpoint_ns != wanted_ns:
                    continue
                prefix = b"c/" + self._scope(thread_id, checkpoint_ns) + b"/"
                for key, raw in reversed(self.db.scan_prefix(prefix)):
                    checkpoint_id = key[len(prefix) :].decode("ascii")
                    if wanted_id and checkpoint_id != wanted_id:
                        continue
                    if before_id and checkpoint_id >= before_id:
                        continue
                    record = _unpack(self._decode_value(key, raw))
                    metadata = self.serde.loads_typed((record[2], record[3]))
                    if filter and not all(
                        value == metadata.get(name) for name, value in filter.items()
                    ):
                        continue
                    if limit is not None:
                        if limit <= 0:
                            return
                        limit -= 1
                    yield self._build_tuple(
                        thread_id, checkpoint_ns, checkpoint_id, record, metadata
                    )

    # ------------------------------------------------------------------ writes

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_id = config["configurable"].get("checkpoint_id")
        checkpoint_id = checkpoint["id"]
        scope = self._scope(thread_id, checkpoint_ns)

        stored = checkpoint.copy()
        values: dict[str, Any] = stored.pop("channel_values")  # type: ignore[misc]
        checkpoint_typed = self.serde.dumps_typed(stored)
        metadata_typed = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))

        batch = persistkv.WriteBatch()
        batch.put(b"t/" + _hash(thread_id), thread_id.encode("utf-8"))
        batch.put(b"n/" + scope, checkpoint_ns.encode("utf-8"))
        for channel, version in new_versions.items():
            typed = self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
            self._put_value(batch, self._blob_key(scope, channel, version), _pack(list(typed)))
        self._put_value(
            batch,
            b"c/" + scope + b"/" + checkpoint_id.encode("ascii"),
            _pack([*checkpoint_typed, *metadata_typed, parent_id]),
        )

        # An older id must not move the pointer backwards, so this has to read before writing.
        latest_key = b"l/" + scope
        with self._lock(thread_id):
            latest = self.db.get(latest_key)
            if latest is None or checkpoint_id >= latest.decode("ascii"):
                batch.put(latest_key, checkpoint_id.encode("ascii"))
            crashpoint("ckpt.before_commit")
            self.db.write(batch)
            crashpoint("ckpt.after_commit")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        scope = self._scope(thread_id, checkpoint_ns)
        prefix = b"w/" + scope + b"/" + checkpoint_id.encode("ascii") + b"/" + _hash(task_id, 8)

        batch = persistkv.WriteBatch()
        with self._lock(thread_id):
            for position, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, position)
                key = prefix + b"/%d" % idx
                # A retried task must not replace regular writes that are already stored. Special
                # channels (negative idx, such as errors and interrupts) are meant to be replaced.
                if idx >= 0 and key in self.db:
                    continue
                type_, data = self.serde.dumps_typed(value)
                self._put_value(batch, key, _pack([task_id, channel, type_, data, task_path, idx]))
            if len(batch):
                self.db.write(batch)

    def delete_thread(self, thread_id: str) -> None:
        thread_hash = _hash(thread_id)
        batch = persistkv.WriteBatch()
        with self._lock(thread_id):
            for family in _THREAD_PREFIXES:
                for prefix in (family, b"x" + family):
                    for key, _ in self.db.scan_prefix(prefix + thread_hash + b"/"):
                        batch.remove(key)
            batch.remove(b"t/" + thread_hash)
            self.db.write(batch)

    def get_next_version(self, current: str | int | None, channel: None) -> str:
        if current is None:
            current_version = 0
        elif isinstance(current, int):
            current_version = current
        else:
            current_version = int(current.split(".")[0])
        return f"{current_version + 1:032}.{random.random():016}"

    # ------------------------------------------------------------------ async

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        rows = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for row in rows:
            yield row

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)
