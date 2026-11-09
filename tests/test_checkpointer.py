"""Tests for PersistKVSaver: the official conformance suite, then durability behavior."""

from __future__ import annotations

import os
import random
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import TypedDict

import persistkv
import pytest
from langgraph.checkpoint.base import WRITES_IDX_MAP, empty_checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.capabilities import BASE_CAPABILITIES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hitl.checkpointer import CHUNK_SIZE, PersistKVSaver

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def make_checkpoint(values: dict | None = None, checkpoint_id: str | None = None) -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = checkpoint_id or str(uuid6(clock_seq=-1))
    checkpoint["channel_values"] = values or {}
    checkpoint["channel_versions"] = {name: 1 for name in values or {}}
    return checkpoint


def put(saver, thread_id, values=None, checkpoint_id=None, ns="", parent=None):
    checkpoint = make_checkpoint(values, checkpoint_id)
    configurable = {"thread_id": thread_id, "checkpoint_ns": ns}
    if parent:
        configurable["checkpoint_id"] = parent
    return saver.put(
        {"configurable": configurable},
        checkpoint,
        {"source": "loop", "step": 0, "parents": {}},
        checkpoint["channel_versions"],
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db")


@pytest.fixture
def saver(db_path):
    db = persistkv.DB(db_path)
    yield PersistKVSaver(db)
    db.close()


# --------------------------------------------------------------- official conformance suite


async def test_official_conformance_suite(tmp_path):
    count = iter(range(100))

    @checkpointer_test(name="PersistKVSaver")
    async def factory():
        db = persistkv.DB(str(tmp_path / f"conformance{next(count)}"))
        yield PersistKVSaver(db)
        db.close()

    report = await validate(factory)
    failures = {
        capability.value: report.results[capability.value].failures
        for capability in BASE_CAPABILITIES
        if report.results[capability.value].passed is not True
    }
    assert not failures


# --------------------------------------------------------------- storage behavior


def test_checkpoint_survives_clean_reopen(db_path):
    db = persistkv.DB(db_path)
    config = put(PersistKVSaver(db), "t1", {"count": 7})
    db.close()

    db = persistkv.DB(db_path)
    found = PersistKVSaver(db).get_tuple({"configurable": {"thread_id": "t1"}})
    assert found.checkpoint["id"] == config["configurable"]["checkpoint_id"]
    assert found.checkpoint["channel_values"] == {"count": 7}
    db.close()


@pytest.mark.parametrize("size", [CHUNK_SIZE, CHUNK_SIZE + 1, 5 * CHUNK_SIZE + 123])
def test_values_larger_than_one_kv_value_round_trip(db_path, size):
    payload = random.Random(size).randbytes(size)
    db = persistkv.DB(db_path)
    saver = PersistKVSaver(db)
    config = put(saver, "big", {"blob": payload})
    saver.put_writes(config, [("blob", payload)], "task-1")
    db.close()

    db = persistkv.DB(db_path)
    found = PersistKVSaver(db).get_tuple({"configurable": {"thread_id": "big"}})
    assert found.checkpoint["channel_values"]["blob"] == payload
    assert found.pending_writes == [("task-1", "blob", payload)]
    assert db.check_integrity() == ""
    db.close()


def test_ids_longer_than_the_key_limit(saver):
    thread_id = "thread-é-" * 100
    namespace = "namespace|" * 100
    put(saver, thread_id, {"x": 1}, ns=namespace)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": namespace}}
    assert saver.get_tuple(config).checkpoint["channel_values"] == {"x": 1}
    assert saver.get_tuple(config).config["configurable"]["thread_id"] == thread_id
    assert [row.config["configurable"]["thread_id"] for row in saver.list(None)] == [thread_id]


def test_older_checkpoint_does_not_move_latest_backwards(saver):
    older, newer = sorted(str(uuid6(clock_seq=-1)) for _ in range(2))
    put(saver, "t", checkpoint_id=newer)
    put(saver, "t", checkpoint_id=older)
    assert saver.get_tuple({"configurable": {"thread_id": "t"}}).checkpoint["id"] == newer


def test_retried_writes_are_not_replaced_but_special_channels_are(saver):
    config = put(saver, "t")
    saver.put_writes(config, [("channel", "first")], "task")
    saver.put_writes(config, [("channel", "second")], "task")
    special = next(iter(WRITES_IDX_MAP))
    saver.put_writes(config, [(special, "first")], "task")
    saver.put_writes(config, [(special, "second")], "task")
    writes = {channel: value for _, channel, value in saver.get_tuple(config).pending_writes}
    assert writes == {"channel": "first", special: "second"}


def test_delete_thread_removes_every_key_including_chunks(saver):
    config = put(saver, "doomed", {"blob": b"x" * (2 * CHUNK_SIZE)}, ns="sub")
    saver.put_writes(config, [("blob", b"y" * (2 * CHUNK_SIZE))], "task")
    put(saver, "kept", {"x": 1})
    before = len(saver.db)
    saver.delete_thread("doomed")
    assert len(saver.db) < before
    assert saver.get_tuple({"configurable": {"thread_id": "doomed", "checkpoint_ns": "sub"}}) is None
    assert saver.get_tuple({"configurable": {"thread_id": "kept"}}) is not None
    # Only the five keys of the kept thread's checkpoint remain: t, n, l, c and one blob.
    assert len(saver.db) == 5


# --------------------------------------------------------------- with a real graph


class State(TypedDict):
    steps: list[str]
    decision: str


def build_graph(checkpointer):
    def before(state):
        return {"steps": state["steps"] + ["before"]}

    def approval(state):
        return {"decision": interrupt({"ask": "approve?"})}

    def after(state):
        return {"steps": state["steps"] + ["after:" + state["decision"]]}

    builder = StateGraph(State)
    builder.add_node("before", before)
    builder.add_node("approval", approval)
    builder.add_node("after", after)
    builder.add_edge(START, "before")
    builder.add_edge("before", "approval")
    builder.add_edge("approval", "after")
    builder.add_edge("after", END)
    return builder.compile(checkpointer=checkpointer)


def run_to_end(checkpointer):
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": "same"}}
    paused = graph.invoke({"steps": [], "decision": ""}, config, durability="sync")
    return paused, graph.invoke(Command(resume="yes"), config, durability="sync")


def test_graph_result_matches_the_in_memory_saver(saver):
    expected_paused, expected_final = run_to_end(InMemorySaver())
    paused, final = run_to_end(saver)
    assert final == expected_final == {"steps": ["before", "after:yes"], "decision": "yes"}
    assert [i.value for i in paused["__interrupt__"]] == [
        i.value for i in expected_paused["__interrupt__"]
    ]


# --------------------------------------------------------------- real SIGKILL


def run_child(code: str, *args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        env={**os.environ, "PYTHONPATH": SRC_DIR},
        stdout=subprocess.PIPE,
        text=True,
    )


PAUSE_THEN_DIE = f"""
    import os, signal, sys
    sys.path.insert(0, {str(Path(__file__).parent)!r})
    import persistkv
    from hitl.checkpointer import PersistKVSaver
    from test_checkpointer import build_graph
    graph = build_graph(PersistKVSaver(persistkv.DB(sys.argv[1])))
    config = {{"configurable": {{"thread_id": "paused"}}}}
    graph.invoke({{"steps": [], "decision": ""}}, config, durability="sync")
    os.kill(os.getpid(), signal.SIGKILL)
"""


def test_paused_graph_resumes_in_a_new_process_after_sigkill(db_path):
    child = run_child(PAUSE_THEN_DIE, db_path)
    assert child.wait(timeout=60) == -signal.SIGKILL

    db = persistkv.DB(db_path)
    graph = build_graph(PersistKVSaver(db))
    config = {"configurable": {"thread_id": "paused"}}
    state = graph.get_state(config)
    assert state.next == ("approval",)
    assert state.interrupts[0].value == {"ask": "approve?"}
    final = graph.invoke(Command(resume="approved"), config, durability="sync")
    assert final == {"steps": ["before", "after:approved"], "decision": "approved"}
    db.close()


WRITE_FOREVER = """
    import sys
    import persistkv
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.base.id import uuid6
    from hitl.checkpointer import CHUNK_SIZE, PersistKVSaver
    saver = PersistKVSaver(persistkv.DB(sys.argv[1]))
    sequence = int(sys.argv[2])
    parent = None
    while True:
        sequence += 1
        # Every channel holds the sequence number, so a checkpoint assembled from two different
        # commits is detectable. The large channel forces every commit to span several chunks.
        fill = sequence % 251
        values = {"seq": sequence, "large": bytes([fill]) * (2 * CHUNK_SIZE + 10)}
        checkpoint = empty_checkpoint()
        checkpoint["id"] = str(uuid6(clock_seq=-1))
        checkpoint["channel_values"] = values
        versions = {"seq": f"{sequence:032}.0", "large": f"{sequence:032}.0"}
        checkpoint["channel_versions"] = versions
        configurable = {"thread_id": "t", "checkpoint_ns": ""}
        if parent:
            configurable["checkpoint_id"] = parent
        saved = saver.put({"configurable": configurable}, checkpoint, {"step": sequence}, versions)
        parent = saved["configurable"]["checkpoint_id"]
        print(sequence, flush=True)
"""


def test_acknowledged_checkpoints_survive_sigkill_during_writes(db_path):
    """Kills a process writing multi-chunk checkpoints at random moments. After each kill, the
    newest checkpoint must be whole, and at least as new as the last one acknowledged."""
    rng = random.Random(1234)
    last_acknowledged = 0
    for _ in range(15):
        child = run_child(WRITE_FOREVER, db_path, str(last_acknowledged))
        # Wait for a first commit, so that every round exercises the write path.
        lines = [child.stdout.readline()]
        time.sleep(rng.uniform(0, 0.15))
        child.send_signal(signal.SIGKILL)
        lines += child.stdout.read().splitlines()
        child.wait(timeout=30)
        acknowledged = [int(line) for line in lines if line.strip()]
        assert acknowledged, "the child made no progress before being killed"
        last_acknowledged = max(acknowledged)

        db = persistkv.DB(db_path)
        saver = PersistKVSaver(db)
        assert db.check_integrity() == ""
        found = saver.get_tuple({"configurable": {"thread_id": "t"}})
        sequence = found.checkpoint["channel_values"]["seq"]
        large = found.checkpoint["channel_values"]["large"]
        assert sequence >= last_acknowledged, "lost an acknowledged checkpoint"
        assert sequence <= last_acknowledged + 1, "a checkpoint appeared from nowhere"
        assert large == bytes([sequence % 251]) * (2 * CHUNK_SIZE + 10), "torn checkpoint"
        assert found.metadata["step"] == sequence
        last_acknowledged = sequence
        db.close()
