"""The system the agent acts on: a stand-in for a payment or card service.

It runs as a separate process with its own database, because that is the reason exactly-once
needs care at all. The engine cannot commit its own "done" record atomically with an effect that
happens over here.

The guarantee comes from `execute`: the effect and the idempotency key commit in one WriteBatch.
A repeated key returns the stored result and applies nothing.

Each applied effect also appends a uniquely keyed `e/` record that dedupe does not read. Counting
those records is how the experiments detect duplicates independently of the dedupe logic.

    python -m hitl.sink            # HITL_SINK_DB, HITL_SINK_PORT
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from typing import Any

import persistkv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from hitl.crash import crashpoint
from hitl.ledger import canonical
from hitl.storage import open_exclusive

_LOCK_STRIPES = 64

# Experiment control only: turns dedupe off, to prove the experiments can detect duplicates.
UNSAFE_NO_DEDUPE = os.environ.get("HITL_UNSAFE_SINK_NO_DEDUPE") == "1"


class ExecuteRequest(BaseModel):
    idempotency_key: str
    action_id: str
    tool: str
    args: dict[str, Any]


class Sink:
    def __init__(self, db: persistkv.DB) -> None:
        self.db = db
        self._locks = [threading.Lock() for _ in range(_LOCK_STRIPES)]

    def execute(self, request: ExecuteRequest) -> dict[str, Any]:
        key = b"k/" + request.idempotency_key.encode("utf-8")
        balance_key = None
        if request.tool == "refund":
            balance_key = b"b/" + str(request.args["account"]).encode("utf-8")
        # A refund updates a balance shared with other keys, so it serializes on the account. A
        # given key always maps to one lock, which is all that dedupe needs.
        digest = hashlib.blake2b(balance_key or key, digest_size=4).digest()
        with self._locks[int.from_bytes(digest, "big") % _LOCK_STRIPES]:
            if not UNSAFE_NO_DEDUPE:
                stored = self.db.get(key)
                if stored is not None:
                    return {**json.loads(stored), "replayed": True}
            result = {"effect_id": uuid.uuid4().hex, "tool": request.tool, "status": "applied"}
            batch = persistkv.WriteBatch()
            batch.put(key, canonical(result))
            batch.put(
                b"e/" + result["effect_id"].encode("ascii"),
                canonical(
                    {
                        "action_id": request.action_id,
                        "idempotency_key": request.idempotency_key,
                        "tool": request.tool,
                        "args": request.args,
                    }
                ),
            )
            if balance_key is not None:
                balance = int(self.db.get(balance_key) or b"0")
                batch.put(balance_key, b"%d" % (balance + int(request.args["amount"])))
            self.db.write(batch)
            crashpoint("sink.after_commit")
        return {**result, "replayed": False}

    def effects(self) -> list[dict[str, Any]]:
        return [json.loads(raw) for _, raw in self.db.scan_prefix(b"e/")]

    def balances(self) -> dict[str, int]:
        return {key[2:].decode("utf-8"): int(raw) for key, raw in self.db.scan_prefix(b"b/")}


def create_app(sink: Sink) -> FastAPI:
    app = FastAPI(title="Effect sink")

    @app.post("/execute")
    def execute(request: ExecuteRequest) -> dict[str, Any]:
        return sink.execute(request)

    @app.get("/effects")
    def effects() -> dict[str, Any]:
        return {"effects": sink.effects(), "balances": sink.balances()}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    db, _lock = open_exclusive(os.environ["HITL_SINK_DB"])
    uvicorn.run(
        create_app(Sink(db)),
        host="127.0.0.1",
        port=int(os.environ["HITL_SINK_PORT"]),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
