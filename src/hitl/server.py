"""HTTP API for the engine.

    python -m hitl.server     # HITL_DB, HITL_PORT, HITL_SINK_URL, HITL_POLICY

`role` in a decision is taken at face value, because this project is about durability and not
authentication. A real deployment would derive it from a verified identity.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from hitl.checkpointer import PersistKVSaver
from hitl.engine import Engine, SinkClient
from hitl.ledger import Conflict, Ledger, NotAllowed, UnknownAction
from hitl.policy import Policy
from hitl.storage import open_exclusive


class StartRun(BaseModel):
    run_id: str | None = None
    actions: list[dict[str, Any]]


class Decision(BaseModel):
    actor: str
    role: str
    reason: str = ""


def create_app(engine: Engine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="Human-in-the-loop approval engine", lifespan=lifespan)
    ledger = engine.ledger

    @app.post("/runs", status_code=202)
    def start_run(body: StartRun) -> dict[str, str]:
        run_id = body.run_id or uuid.uuid4().hex
        engine.start_run(run_id, {"actions": body.actions})
        return {"run_id": run_id}

    @app.get("/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": ledger.all_runs()}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = ledger.get_run(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run

    @app.get("/runs/{run_id}/audit")
    def get_audit(run_id: str) -> dict[str, Any]:
        return {"entries": ledger.audit_trail(run_id), "chain_error": ledger.verify_audit(run_id)}

    @app.get("/approvals")
    def list_approvals() -> dict[str, Any]:
        return {
            "pending": [
                {
                    "action_id": action["action_id"],
                    "run_id": action["run_id"],
                    "tool": action["tool"],
                    "args": action["args"],
                    "rule": action["rule"],
                    "role": action["tiers"][action["tier"]][0],
                    "tier": action["tier"],
                    "deadline_ms": action["deadline_ms"],
                }
                for action in ledger.pending_actions()
            ]
        }

    def decide(action_id: str, outcome: str, body: Decision) -> dict[str, Any]:
        try:
            action = engine.decide(
                action_id, outcome, actor=body.actor, role=body.role, reason=body.reason
            )
        except UnknownAction:
            raise HTTPException(404, "no such action") from None
        except NotAllowed as error:
            raise HTTPException(403, str(error)) from None
        except Conflict as error:
            raise HTTPException(409, str(error)) from None
        return {"action_id": action_id, "status": action["status"]}

    @app.post("/approvals/{action_id}/approve")
    def approve(action_id: str, body: Decision) -> dict[str, Any]:
        return decide(action_id, "approved", body)

    @app.post("/approvals/{action_id}/reject")
    def reject(action_id: str, body: Decision) -> dict[str, Any]:
        return decide(action_id, "rejected", body)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    db, _lock = open_exclusive(os.environ["HITL_DB"])
    # Experiment control only: shows that the kill tests detect a checkpointer that loses state.
    if os.environ.get("HITL_UNSAFE_MEMORY_CHECKPOINTER") == "1":
        checkpointer: Any = InMemorySaver()
    else:
        checkpointer = PersistKVSaver(db)
    engine = Engine(
        Ledger(db),
        checkpointer,
        Policy.load(os.environ["HITL_POLICY"]),
        SinkClient(os.environ["HITL_SINK_URL"]),
        latency_log=os.environ.get("HITL_LATENCY_LOG"),
    )
    uvicorn.run(
        create_app(engine),
        host="127.0.0.1",
        port=int(os.environ["HITL_PORT"]),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
