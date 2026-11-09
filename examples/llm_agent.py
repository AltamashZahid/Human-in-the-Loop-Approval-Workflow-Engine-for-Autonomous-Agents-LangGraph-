"""The approval engine with Claude as the planner, instead of the scripted one the experiments use.

Claude turns a plain-English request into tool calls. It plans once per run, and everything after
that is the same engine: each call goes through the policy, risky ones wait for you, and the plan
is checkpointed before anything acts on it.

The model does not get to decide what is safe. A call it proposes is still denied or held by the
policy, so a confused or manipulated model cannot move money on its own.

    pip install -e ".[llm]"
    export ANTHROPIC_API_KEY=...
    python examples/llm_agent.py "Customer 42 was charged twice for order 7, $180 each. Refund one."

Kill it with `kill -9` while it waits for your answer, then run it again with no request and it
picks up the same approval:

    python examples/llm_agent.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hitl.checkpointer import PersistKVSaver  # noqa: E402
from hitl.engine import Engine  # noqa: E402
from hitl.ledger import Ledger  # noqa: E402
from hitl.policy import Policy  # noqa: E402
from hitl.sink import ExecuteRequest, Sink  # noqa: E402
from hitl.storage import open_exclusive  # noqa: E402

log = logging.getLogger("hitl.llm")

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a customer-support agent for a payments company. Carry out the request \
by calling the tools it needs. Each tool call you make is reviewed against a risk policy before it \
runs, and some are held for a human, so propose what the request calls for and leave the risk \
decisions to that review. If the request does not call for any tool, reply briefly and call none."""


def _tool(name: str, description: str, **properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_ORDER = {"type": "integer", "description": "The order this action concerns."}

TOOLS = [
    _tool("lookup_order", "Read the details of an order.", order_id=_ORDER),
    _tool(
        "get_balance",
        "Read an account's balance.",
        account={"type": "string", "description": "The account identifier."},
    ),
    _tool(
        "send_notification",
        "Send the customer a message about an order.",
        order_id=_ORDER,
        message={"type": "string", "description": "The text to send."},
    ),
    _tool(
        "refund",
        "Refund money to an account for an order.",
        order_id=_ORDER,
        account={"type": "string", "description": "The account to credit."},
        amount={"type": "integer", "description": "The amount, in whole dollars."},
    ),
    _tool(
        "block_card",
        "Block a card, for example after suspected fraud.",
        order_id=_ORDER,
        card={"type": "string", "description": "The card identifier."},
    ),
]


class ClaudePlanner:
    """Implements the engine's Planner interface: a request in, tool calls out."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client

    def __call__(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        # Forcing a tool call with tool_choice is a 400 on this model. The default "auto", with the
        # system prompt and strict schemas, is the supported way to get schema-valid calls.
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=[{"role": "user", "content": request["prompt"]}],
        )
        if response.stop_reason == "refusal":
            # Every model in the fallback chain declined. No plan means no action, the safe outcome.
            log.warning("the model declined this request: %s", response.stop_details)
            return []
        if response.stop_reason == "max_tokens":
            # A plan cut off here could be missing steps, so raising makes the engine plan again.
            raise RuntimeError("the plan was cut off by max_tokens")
        return [
            # The id comes from the model and differs on every call, which is safe because the plan
            # is checkpointed before anything acts on it, so a retry reuses these same ids.
            {"call_id": block.id, "tool": block.name, "args": block.input}
            for block in response.content
            if block.type == "tool_use"
        ]


class LocalSink:
    """The sink, in this process, for a demo you can run with one command. The experiments run it
    as a separate process. It still has its own database, so the same guarantee holds."""

    def __init__(self, db: Any) -> None:
        self._sink = Sink(db)

    def execute(self, key: str, action: dict[str, Any]) -> dict[str, Any]:
        return self._sink.execute(
            ExecuteRequest(
                idempotency_key=key,
                action_id=action["action_id"],
                tool=action["tool"],
                args=action["args"],
            )
        )


def ask_human(engine: Engine, asked: set[str]) -> None:
    for action in engine.ledger.pending_actions():
        if action["action_id"] in asked:
            continue
        asked.add(action["action_id"])
        role = action["tiers"][action["tier"]][0]
        print(f"\nApproval needed from {role} (rule: {action['rule']})")
        print(f"  {action['tool']} {action['args']}")
        answer = input("  Approve? [y/N] ").strip().lower()
        engine.decide(
            action["action_id"],
            "approved" if answer == "y" else "rejected",
            actor="terminal",
            role=role,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("request", nargs="?", help="what the agent should do; omit to resume")
    parser.add_argument("--data", default=str(Path.home() / "hitl" / "data" / "demo"))
    parser.add_argument("--policy", default=str(ROOT / "policies" / "default.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    client = anthropic.Anthropic()
    try:
        # The engine retries a failed run forever, which is right for a brief outage but not for a
        # missing key or a wrong model name. Those should stop here with a clear message.
        client.models.retrieve(MODEL)
    except anthropic.AuthenticationError:
        raise SystemExit("Claude rejected the credentials: set ANTHROPIC_API_KEY") from None
    except anthropic.PermissionDeniedError:
        raise SystemExit(f"This API key does not have access to {MODEL}") from None
    except anthropic.NotFoundError:
        raise SystemExit(f"Model {MODEL} was not found") from None
    except anthropic.APIConnectionError:
        raise SystemExit("Could not reach the Claude API") from None

    db, _engine_lock = open_exclusive(f"{args.data}/engine")
    sink_db, _sink_lock = open_exclusive(f"{args.data}/sink")
    engine = Engine(
        Ledger(db),
        PersistKVSaver(db),
        Policy.load(args.policy),
        LocalSink(sink_db),
        planner=ClaudePlanner(client),
    )
    engine.start()  # picks up every unfinished run, including ones paused before a crash

    if args.request:
        engine.start_run(uuid.uuid4().hex, {"prompt": args.request})

    watched = engine.ledger.active_runs()
    if not watched:
        print("Nothing to do: no request given, and no unfinished runs to resume.")

    asked: set[str] = set()
    try:
        while engine.ledger.active_runs():
            ask_human(engine, asked)
            time.sleep(0.05)
    finally:
        engine.stop()

    for run_id in watched:
        run = engine.ledger.get_run(run_id)
        print(f"\nRun {run_id}: {run['status']}")
        for outcome in run.get("outcomes", []):
            action = engine.ledger.get_action(outcome["action_id"])
            print(f"  {outcome['status']:10} {action['tool']} {action['args']}")
        if run["status"] == "completed" and not run.get("outcomes"):
            print("  The model proposed no actions.")
        print(f"  Audit chain: {engine.ledger.verify_audit(run_id) or 'valid'}")
    db.close()
    sink_db.close()


if __name__ == "__main__":
    main()
