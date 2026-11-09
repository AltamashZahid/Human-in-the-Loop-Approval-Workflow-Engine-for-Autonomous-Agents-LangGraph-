"""The approval engine: a LangGraph agent whose risky tool calls wait for a human.

    plan -> route -> approval -> execute -> advance -> route -> ... -> END
                  \\-> execute (policy: auto)
                  \\-> advance (policy: deny, or a human said no)

Three properties of LangGraph shape this file:

1. A node re-runs from its first line when it resumes, so `approval` does nothing but interrupt,
   and the side effect is in a different node.
2. The default durability mode writes checkpoints in the background, which a crash can lose, so
   every invoke here uses durability="sync".
3. A crash can happen at any line, so normal progress and crash recovery are the same code:
   `_drive` looks at the durable state and does the next thing. There is no separate recovery path
   that only runs after failures and is rarely exercised.

Exactly-once execution, in `execute`:

    mark_intent  ->  sink.execute(idempotency key)  ->  mark_done

A crash between any two steps re-runs the node with the same key. The ledger short-circuits the
call if `done` was reached, and otherwise the sink returns the stored result if the effect was
already committed. This is at-least-once delivery to a receiver that deduplicates, and it depends
on the receiver honoring idempotency keys.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hitl.crash import crashpoint
from hitl.ledger import DECIDED, MAY_EXECUTE, RUN_TERMINAL, Ledger
from hitl.policy import Policy

log = logging.getLogger("hitl.engine")

# Experiment control only: a fresh key on every attempt, to show why keys must be deterministic.
UNSAFE_RANDOM_KEY = os.environ.get("HITL_UNSAFE_RANDOM_KEY") == "1"

Planner = Callable[[dict[str, Any]], list[dict[str, Any]]]


def scripted_planner(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Turns a request into tool calls without an LLM, so that experiments are deterministic and
    repeatable. The planner's output is checkpointed before it is used, so a real model, which
    would answer differently each time, fits behind the same interface."""
    return [
        {"call_id": f"call_{index}", "tool": call["tool"], "args": call["args"]}
        for index, call in enumerate(request["actions"])
    ]


class AgentState(TypedDict, total=False):
    request: dict[str, Any]
    calls: list[dict[str, Any]]
    cursor: int
    action_id: str
    status: str
    outcomes: list[dict[str, Any]]


class SinkClient:
    """Retries transport errors, which are safe to retry because the key makes repeats harmless."""

    def __init__(self, url: str, retry_for_s: float = 30.0) -> None:
        self._client = httpx.Client(base_url=url, timeout=5.0)
        self._retry_for_s = retry_for_s

    def execute(self, key: str, action: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "idempotency_key": key,
            "action_id": action["action_id"],
            "tool": action["tool"],
            "args": action["args"],
        }
        give_up_at = time.monotonic() + self._retry_for_s
        delay = 0.05
        while True:
            try:
                response = self._client.post("/execute", json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.TransportError:
                if time.monotonic() >= give_up_at:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 1.0)


class Engine:
    def __init__(
        self,
        ledger: Ledger,
        checkpointer: BaseCheckpointSaver,
        policy: Policy,
        sink: SinkClient,
        *,
        planner: Planner = scripted_planner,
        workers: int = 16,
        scan_interval_s: float = 0.05,
        latency_log: str | None = None,
    ) -> None:
        self.ledger = ledger
        self.checkpointer = checkpointer
        self.policy = policy
        self.sink = sink
        self.planner = planner
        self.graph = self._build_graph()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="run")
        self._mu = threading.Lock()
        self._driving: set[str] = set()
        self._drive_again: set[str] = set()
        self._stop = threading.Event()
        self._scan_interval_s = scan_interval_s
        self._scanner = threading.Thread(target=self._scan_loop, name="deadlines", daemon=True)
        self._approved_at: dict[str, int] = {}
        self._latency_log = open(latency_log, "a", buffering=1) if latency_log else None

    # ------------------------------------------------------------------ graph

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("plan", self._node_plan)
        builder.add_node("route", self._node_route)
        builder.add_node("approval", self._node_approval)
        builder.add_node("execute", self._node_execute)
        builder.add_node("advance", self._node_advance)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges("plan", self._next_call)
        builder.add_conditional_edges("route", self._after_route)
        builder.add_conditional_edges("approval", self._after_approval)
        builder.add_edge("execute", "advance")
        builder.add_conditional_edges("advance", self._next_call)
        return builder.compile(checkpointer=self.checkpointer)

    def _node_plan(self, state: AgentState) -> AgentState:
        return {"calls": self.planner(state["request"]), "cursor": 0, "outcomes": []}

    @staticmethod
    def _next_call(state: AgentState) -> str:
        return "route" if state["cursor"] < len(state["calls"]) else END

    def _node_route(self, state: AgentState, config: RunnableConfig) -> AgentState:
        call = state["calls"][state["cursor"]]
        run_id = config["configurable"]["thread_id"]
        route = self.policy.evaluate(call["tool"], call["args"])
        action = self.ledger.propose(run_id, call["call_id"], call["tool"], call["args"], route)
        return {"action_id": action["action_id"], "status": action["status"]}

    @staticmethod
    def _after_route(state: AgentState) -> str:
        # A retried node can find the action already past `pending`. `approval` then gets the
        # recorded decision from the driver, and a finished action goes to `execute`, which
        # returns the stored result without calling the sink.
        status = state["status"]
        if status in MAY_EXECUTE or status == "done":
            return "execute"
        if status == "denied":
            return "advance"
        return "approval"

    def _node_approval(self, state: AgentState) -> AgentState:
        # Runs twice around a pause: everything above interrupt() re-runs on resume, so there is
        # nothing above it. The ledger, not this value, is what `execute` trusts.
        return {"status": interrupt({"action_id": state["action_id"]})}

    @staticmethod
    def _after_approval(state: AgentState) -> str:
        return "execute" if state["status"] == "approved" else "advance"

    def _node_execute(self, state: AgentState) -> AgentState:
        self._record_resume_latency(state["action_id"])
        action = self.ledger.get_action(state["action_id"])
        if action["status"] == "done":  # a crash landed after mark_done
            return {"status": "done"}
        # Defense in depth: nothing reaches the sink unless the ledger says it was authorized,
        # whatever the graph state claims.
        if action["status"] not in MAY_EXECUTE:
            raise RuntimeError(f"refusing to execute {action['action_id']}: {action['status']}")
        self.ledger.mark_intent(action["action_id"])
        crashpoint("exec.after_intent")
        key = uuid.uuid4().hex if UNSAFE_RANDOM_KEY else action["action_id"]
        result = self.sink.execute(key, action)
        crashpoint("exec.after_sink")  # the hard case: effect committed, "done" not yet recorded
        self.ledger.mark_done(action["action_id"], result)
        crashpoint("exec.after_done")
        return {"status": "done"}

    def _node_advance(self, state: AgentState) -> AgentState:
        outcome = {"action_id": state["action_id"], "status": state["status"]}
        return {"cursor": state["cursor"] + 1, "outcomes": [*state["outcomes"], outcome]}

    # ------------------------------------------------------------------ driving runs

    def start(self) -> None:
        """Recovery and startup are one operation: drive every run that has not finished."""
        for run_id in self.ledger.active_runs():
            self.submit(run_id)
        self._scanner.start()

    def stop(self) -> None:
        self._stop.set()
        if self._scanner.is_alive():
            self._scanner.join()
        self._pool.shutdown(wait=True, cancel_futures=True)

    def start_run(self, run_id: str, request: dict[str, Any]) -> None:
        if self.ledger.create_run(run_id, request):
            self.submit(run_id)

    def submit(self, run_id: str) -> None:
        if self._stop.is_set():
            return
        with self._mu:
            # One driver per run. A decision arriving during a drive is picked up by another pass.
            if run_id in self._driving:
                self._drive_again.add(run_id)
                return
            self._driving.add(run_id)
        self._pool.submit(self._drive_loop, run_id)

    def _drive_loop(self, run_id: str) -> None:
        try:
            while True:
                self._drive(run_id)
                with self._mu:
                    if run_id not in self._drive_again:
                        self._driving.discard(run_id)
                        return
                    self._drive_again.discard(run_id)
        except Exception:
            log.exception("run %s failed; retrying", run_id)
            with self._mu:
                self._driving.discard(run_id)
                self._drive_again.discard(run_id)
            if not self._stop.is_set():
                timer = threading.Timer(1.0, self.submit, args=[run_id])
                timer.daemon = True
                timer.start()

    def _drive(self, run_id: str) -> None:
        """Advances a run as far as it can go from its durable state, until it pauses or ends."""
        run = self.ledger.get_run(run_id)
        if run is None or run["status"] in RUN_TERMINAL:
            return
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 1000}

        # A run recorded as paused must still have the checkpoint it paused at. If it does not,
        # that is lost state. Report it. Re-running would hide the loss and ask the human again.
        if run["status"] == "paused":
            paused_at = {"configurable": {**config["configurable"], "checkpoint_ns": "",
                                          "checkpoint_id": run["paused_checkpoint"]}}
            if self.checkpointer.get_tuple(paused_at) is None:
                log.error("run %s lost its paused checkpoint", run_id)
                self.ledger.set_run_status(run_id, "lost")
                return

        while True:
            snapshot = self.graph.get_state(config)
            if snapshot.created_at is None:
                next_input: Any = {"request": run["request"]}
            elif snapshot.interrupts:
                action = self.ledger.get_action(snapshot.interrupts[0].value["action_id"])
                if action["status"] == "pending":
                    self.ledger.set_run_status(
                        run_id, "paused",
                        paused_checkpoint=snapshot.config["configurable"]["checkpoint_id"],
                    )
                    return
                self.ledger.set_run_status(run_id, "running")
                outcome = action.get("authorized_as", action["status"])
                next_input = Command(resume=outcome)
            elif snapshot.tasks:
                # Not `snapshot.next`, which leaves out tasks that have saved their writes. A crash
                # after a node's writes but before the next checkpoint makes `next` empty on a run
                # that still has steps to go. Invoking with None applies the writes and continues.
                next_input = None
            else:
                self.ledger.set_run_status(run_id, "completed", outcomes=snapshot.values["outcomes"])
                return
            self.graph.invoke(next_input, config, durability="sync")

    # ------------------------------------------------------------------ decisions and deadlines

    def decide(
        self, action_id: str, outcome: str, *, actor: str, role: str | None, reason: str = ""
    ) -> dict[str, Any]:
        received_at = time.perf_counter_ns()
        action = self.ledger.decide(action_id, outcome, actor=actor, role=role, reason=reason)
        if self._latency_log is not None and outcome == "approved":
            self._approved_at[action_id] = received_at
        crashpoint("engine.after_decision")
        self.submit(action["run_id"])
        return action

    def _record_resume_latency(self, action_id: str) -> None:
        """From the approval arriving to the first line of `execute`, the first work done after
        the run resumes. Includes the durable decision, a checkpoint read, and the checkpoint
        commits around the `approval` node."""
        received_at = self._approved_at.pop(action_id, None)
        if received_at is not None and self._latency_log is not None:
            elapsed_ms = (time.perf_counter_ns() - received_at) / 1e6
            self._latency_log.write(json.dumps({"action_id": action_id, "ms": elapsed_ms}) + "\n")

    def _scan_loop(self) -> None:
        while not self._stop.wait(self._scan_interval_s):
            try:
                for action in self.ledger.expire_due():
                    if action["status"] in DECIDED:
                        self.submit(action["run_id"])
            except Exception:
                log.exception("deadline scan failed")
