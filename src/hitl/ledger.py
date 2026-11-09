"""The durable record of runs, actions, approvals and the audit trail.

Each state change and its audit entry commit in one WriteBatch, so they cannot disagree after a
crash. This lives in the same PersistKV database as the checkpoints, but in separate batches:
the ledger, not the checkpoint, is the source of truth for what a human decided and for whether
an action already ran.

Keys all start with "L", which the checkpointer's key families do not use:

    Lr/<run id>                run record
    Lq/<run id>                index of runs that are not finished yet
    La/<action id>             action record
    Lp/<action id>             index of actions waiting for a human
    Ld/<deadline ms>/<action>  index of approval deadlines, in time order
    Lv/<run id>/<seq>          audit entry, hash-chained per run

Action status:   pending -> approved | rejected | timed_out
                 auto, or denied, straight from the policy
                 approved | auto -> done, once the action has executed
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import persistkv

from hitl.policy import Route

_LOCK_STRIPES = 64
_GENESIS = "0" * 64

RUN_TERMINAL = {"completed", "lost"}
MAY_EXECUTE = {"auto", "approved"}
DECIDED = {"approved", "rejected", "timed_out"}


class Conflict(Exception):
    """The action already has a different decision. The first decision wins."""


class NotAllowed(Exception):
    """This role may not decide this action at its current escalation tier."""


class UnknownAction(KeyError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def make_action_id(run_id: str, call_id: str, tool: str, args: dict[str, Any]) -> str:
    """Derived only from what is stable across re-execution of a node, never from a random value
    or a timestamp, so that a retry after a crash produces the same id and idempotency key."""
    return hashlib.sha256(canonical([run_id, call_id, tool, args])).hexdigest()[:32]


def _audit_hash(prev: str, body: dict[str, Any]) -> str:
    return hashlib.sha256(prev.encode("ascii") + canonical(body)).hexdigest()


class Ledger:
    def __init__(self, db: persistkv.DB, clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self.clock = clock
        # One writer per run at a time. Different runs still share fsyncs through group commit.
        self._locks = [threading.Lock() for _ in range(_LOCK_STRIPES)]

    def _lock(self, run_id: str) -> threading.Lock:
        digest = hashlib.blake2b(run_id.encode("utf-8"), digest_size=4).digest()
        return self._locks[int.from_bytes(digest, "big") % _LOCK_STRIPES]

    def _now_ms(self) -> int:
        return int(self.clock() * 1000)

    def _get_json(self, key: bytes) -> dict[str, Any] | None:
        raw = self.db.get(key)
        return None if raw is None else json.loads(raw)

    @staticmethod
    def _action_key(action_id: str) -> bytes:
        return b"La/" + action_id.encode("ascii")

    @staticmethod
    def _deadline_key(deadline_ms: int, action_id: str) -> bytes:
        return b"Ld/%015d/" % deadline_ms + action_id.encode("ascii")

    def _append_audit(
        self, batch: persistkv.WriteBatch, run_id: str, *events: tuple[str, dict[str, Any]]
    ) -> None:
        """Adds entries to `batch`, continuing the run's chain. The caller holds the run's lock."""
        prefix = b"Lv/" + run_id.encode("utf-8") + b"/"
        entries = self.db.scan_prefix(prefix)
        last = json.loads(entries[-1][1]) if entries else None
        seq = last["seq"] if last else -1
        prev = last["hash"] if last else _GENESIS
        for event, data in events:
            seq += 1
            body = {"seq": seq, "ts_ms": self._now_ms(), "event": event, "data": data, "prev": prev}
            prev = _audit_hash(prev, body)
            batch.put(prefix + b"%08d" % seq, canonical({**body, "hash": prev}))

    # ------------------------------------------------------------------ runs

    def create_run(self, run_id: str, request: dict[str, Any]) -> bool:
        """Returns False if the run already exists, so a client may safely retry its request."""
        key = b"Lr/" + run_id.encode("utf-8")
        with self._lock(run_id):
            if key in self.db:
                return False
            batch = persistkv.WriteBatch()
            record = {
                "run_id": run_id,
                "status": "running",
                "request": request,
                "created_ms": self._now_ms(),
            }
            batch.put(key, canonical(record))
            batch.put(b"Lq/" + run_id.encode("utf-8"), b"")
            self._append_audit(batch, run_id, ("run_started", {}))
            self.db.write(batch)
        return True

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._get_json(b"Lr/" + run_id.encode("utf-8"))

    def set_run_status(self, run_id: str, status: str, **fields: Any) -> None:
        key = b"Lr/" + run_id.encode("utf-8")
        with self._lock(run_id):
            record = self._get_json(key)
            if record["status"] == status:
                return
            batch = persistkv.WriteBatch()
            batch.put(key, canonical({**record, **fields, "status": status}))
            if status in RUN_TERMINAL:
                batch.remove(b"Lq/" + run_id.encode("utf-8"))
            self._append_audit(batch, run_id, (f"run_{status}", {}))
            self.db.write(batch)

    def active_runs(self) -> list[str]:
        return [key[3:].decode("utf-8") for key, _ in self.db.scan_prefix(b"Lq/")]

    def all_runs(self) -> list[dict[str, Any]]:
        return [json.loads(raw) for _, raw in self.db.scan_prefix(b"Lr/")]

    # ------------------------------------------------------------------ actions

    def propose(
        self, run_id: str, call_id: str, tool: str, args: dict[str, Any], route: Route
    ) -> dict[str, Any]:
        """Records the policy's routing of a tool call. The node calling this can run again after
        a crash, in which case this returns the stored action and appends nothing."""
        action_id = make_action_id(run_id, call_id, tool, args)
        key = self._action_key(action_id)
        with self._lock(run_id):
            existing = self._get_json(key)
            if existing is not None:
                return existing
            status = {"auto": "auto", "deny": "denied", "approval": "pending"}[route.route]
            action: dict[str, Any] = {
                "action_id": action_id,
                "run_id": run_id,
                "call_id": call_id,
                "tool": tool,
                "args": args,
                "rule": route.rule,
                "route": route.route,
                "status": status,
                "tiers": [[tier.role, tier.timeout_s] for tier in route.escalation],
                "tier": 0,
            }
            batch = persistkv.WriteBatch()
            proposed = {
                "action_id": action_id, "tool": tool, "args": args,
                "rule": route.rule, "route": route.route,
            }
            if route.route == "approval":
                deadline_ms = self._now_ms() + int(route.escalation[0].timeout_s * 1000)
                action["deadline_ms"] = deadline_ms
                batch.put(b"Lp/" + action_id.encode("ascii"), b"")
                batch.put(self._deadline_key(deadline_ms, action_id), b"")
                follow_up = (
                    "approval_requested",
                    {
                        "action_id": action_id,
                        "role": route.escalation[0].role,
                        "deadline_ms": deadline_ms,
                    },
                )
            else:
                follow_up = (
                    "decision",
                    {"action_id": action_id, "outcome": status, "actor": "policy:" + route.rule},
                )
            self._append_audit(batch, run_id, ("action_proposed", proposed), follow_up)
            batch.put(key, canonical(action))
            self.db.write(batch)
        return action

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        return self._get_json(self._action_key(action_id))

    def _require_action(self, action_id: str) -> dict[str, Any]:
        action = self.get_action(action_id)
        if action is None:
            raise UnknownAction(action_id)
        return action

    def pending_actions(self) -> list[dict[str, Any]]:
        found = []
        for key, _ in self.db.scan_prefix(b"Lp/"):
            action = self.get_action(key[3:].decode("ascii"))
            if action is not None and action["status"] == "pending":
                found.append(action)
        return found

    def decide(
        self, action_id: str, outcome: str, *, actor: str, role: str | None, reason: str = ""
    ) -> dict[str, Any]:
        """Records a human decision. The first decision wins, and repeating it is harmless, so the
        race between two approvers, or between an approver and a timeout, has one winner."""
        if outcome not in ("approved", "rejected"):
            raise ValueError(f"bad outcome {outcome!r}")
        run_id = self._require_action(action_id)["run_id"]
        with self._lock(run_id):
            action = self._require_action(action_id)
            # A finished action keeps what authorized it in `authorized_as`.
            if action.get("authorized_as", action["status"]) == outcome:
                return action
            if action["status"] != "pending":
                raise Conflict(f"action is already {action['status']}")
            required = action["tiers"][action["tier"]][0]
            if role != required:
                raise NotAllowed(f"tier {action['tier']} requires role {required!r}, got {role!r}")
            return self._close_pending(action, outcome, actor=actor, reason=reason)

    def _close_pending(
        self, action: dict[str, Any], outcome: str, *, actor: str, reason: str = ""
    ) -> dict[str, Any]:
        action_id = action["action_id"]
        batch = persistkv.WriteBatch()
        batch.remove(b"Lp/" + action_id.encode("ascii"))
        batch.remove(self._deadline_key(action["deadline_ms"], action_id))
        action = {**action, "status": outcome, "decided_by": actor}
        batch.put(self._action_key(action_id), canonical(action))
        self._append_audit(
            batch,
            action["run_id"],
            ("decision", {"action_id": action_id, "outcome": outcome, "actor": actor, "reason": reason}),
        )
        self.db.write(batch)
        return action

    def expire_due(self) -> list[dict[str, Any]]:
        """Escalates or rejects every action whose deadline has passed, returning the ones now
        decided. Deadlines are stored as absolute times, so those that passed while the process
        was down fire at the next scan."""
        now_ms = self._now_ms()
        closed = []
        for key, _ in self.db.scan(b"Ld/", b"Ld/%015d/~" % now_ms):
            action_id = key.rsplit(b"/", 1)[1].decode("ascii")
            action = self.get_action(action_id)
            if action is None:
                continue
            with self._lock(action["run_id"]):
                action = self.get_action(action_id)
                # A decision may have landed since the scan, which removed or moved the deadline.
                if action["status"] != "pending" or action.get("deadline_ms", 0) > now_ms:
                    continue
                if action["tier"] + 1 < len(action["tiers"]):
                    self._escalate(action, now_ms)
                else:
                    closed.append(
                        self._close_pending(action, "timed_out", actor="timeout", reason="no response")
                    )
        return closed

    def _escalate(self, action: dict[str, Any], now_ms: int) -> None:
        action_id = action["action_id"]
        tier = action["tier"] + 1
        role, timeout_s = action["tiers"][tier]
        deadline_ms = now_ms + int(timeout_s * 1000)
        batch = persistkv.WriteBatch()
        batch.remove(self._deadline_key(action["deadline_ms"], action_id))
        batch.put(self._deadline_key(deadline_ms, action_id), b"")
        batch.put(
            self._action_key(action_id),
            canonical({**action, "tier": tier, "deadline_ms": deadline_ms}),
        )
        self._append_audit(
            batch,
            action["run_id"],
            (
                "escalated",
                {"action_id": action_id, "role": role, "tier": tier, "deadline_ms": deadline_ms},
            ),
        )
        self.db.write(batch)

    def mark_intent(self, action_id: str) -> None:
        """Durably records "about to execute" before the side effect. Appends a separate audit
        event on every later attempt, so retries after a crash are visible in the trail."""
        run_id = self._require_action(action_id)["run_id"]
        with self._lock(run_id):
            action = self._require_action(action_id)
            attempts = action.get("attempts", 0) + 1
            batch = persistkv.WriteBatch()
            batch.put(self._action_key(action_id), canonical({**action, "attempts": attempts}))
            event = "execution_intent" if attempts == 1 else "execution_retried"
            self._append_audit(batch, run_id, (event, {"action_id": action_id, "attempt": attempts}))
            self.db.write(batch)

    def mark_done(self, action_id: str, result: dict[str, Any]) -> dict[str, Any]:
        run_id = self._require_action(action_id)["run_id"]
        with self._lock(run_id):
            action = self._require_action(action_id)
            if action["status"] == "done":
                return action
            action = {**action, "status": "done", "authorized_as": action["status"], "result": result}
            batch = persistkv.WriteBatch()
            batch.put(self._action_key(action_id), canonical(action))
            self._append_audit(batch, run_id, ("execution_done", {"action_id": action_id}))
            self.db.write(batch)
        return action

    # ------------------------------------------------------------------ audit

    def audit_trail(self, run_id: str) -> list[dict[str, Any]]:
        return [
            json.loads(raw)
            for _, raw in self.db.scan_prefix(b"Lv/" + run_id.encode("utf-8") + b"/")
        ]

    def verify_audit(self, run_id: str) -> str | None:
        """Walks the hash chain. Returns a description of the first problem, or None."""
        prev = _GENESIS
        for expected_seq, entry in enumerate(self.audit_trail(run_id)):
            body = {name: value for name, value in entry.items() if name != "hash"}
            if entry["seq"] != expected_seq:
                return f"gap at seq {expected_seq}"
            if entry["prev"] != prev:
                return f"broken link at seq {expected_seq}"
            if _audit_hash(prev, body) != entry["hash"]:
                return f"bad hash at seq {expected_seq}"
            prev = entry["hash"]
        return None
