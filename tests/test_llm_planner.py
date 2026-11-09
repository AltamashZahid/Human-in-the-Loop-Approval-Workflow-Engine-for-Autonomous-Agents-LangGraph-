"""The Claude planner, tested with no network. The real SDK talks to a fake HTTP server, so a wrong
parameter name or a misread response fails here. Nothing here shows how the real model behaves."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic
import persistkv
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from llm_agent import MODEL, TOOLS, ClaudePlanner  # noqa: E402
from test_engine import POLICY, FailingSaver, Interrupted, RecordingSink, wait_until  # noqa: E402

from hitl.checkpointer import PersistKVSaver  # noqa: E402
from hitl.engine import Engine  # noqa: E402
from hitl.ledger import Ledger  # noqa: E402
from hitl.policy import Policy  # noqa: E402


def tool_use(block_id, name, **args):
    return {"type": "tool_use", "id": block_id, "name": name, "input": args}


def message(*content, stop_reason="tool_use", **extra):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": list(content),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        **extra,
    }


class FakeAPI:
    """Answers with the given messages in turn, repeating the last, and records each request."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def client(self):
        # The SDK's own HTTP library may change between majors, so take it from the SDK.
        http = sys.modules[anthropic.DefaultHttpxClient.__mro__[1].__module__.split(".")[0]]

        def handler(request):
            self.requests.append((request, json.loads(request.content)))
            index = min(len(self.requests) - 1, len(self.answers) - 1)
            return http.Response(200, json=self.answers[index])

        return anthropic.Anthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=http.MockTransport(handler)),
        )


REQUEST = {"prompt": "refund order 7"}


def test_request_uses_only_what_this_model_accepts():
    api = FakeAPI(message(tool_use("toolu_1", "lookup_order", order_id=7)))
    ClaudePlanner(api.client())(REQUEST)
    http_request, body = api.requests[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in http_request.headers["anthropic-beta"]
    assert body["messages"] == [{"role": "user", "content": "refund order 7"}]
    # These all return a 400 on this model.
    assert body.get("tool_choice", {"type": "auto"})["type"] == "auto"
    assert "temperature" not in body
    assert body.get("thinking", {"type": "adaptive"})["type"] == "adaptive"


def test_every_tool_schema_is_strict_and_closed():
    for tool in TOOLS:
        assert tool["strict"] is True
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_tools_match_what_the_policy_and_sink_expect():
    refund = next(tool for tool in TOOLS if tool["name"] == "refund")
    assert {"account", "amount"} <= set(refund["input_schema"]["properties"])
    assert refund["input_schema"]["properties"]["amount"]["type"] == "integer"
    assert {tool["name"] for tool in TOOLS} == {
        "lookup_order", "get_balance", "send_notification", "refund", "block_card",
    }


def test_response_becomes_tool_calls_and_other_blocks_are_ignored():
    api = FakeAPI(
        message(
            {"type": "thinking", "thinking": "", "signature": "x"},
            {"type": "text", "text": "I will look that up."},
            tool_use("toolu_1", "lookup_order", order_id=7),
            tool_use("toolu_2", "refund", order_id=7, account="a1", amount=180),
        )
    )
    assert ClaudePlanner(api.client())(REQUEST) == [
        {"call_id": "toolu_1", "tool": "lookup_order", "args": {"order_id": 7}},
        {"call_id": "toolu_2", "tool": "refund", "args": {"order_id": 7, "account": "a1", "amount": 180}},
    ]


def test_no_tool_calls_means_an_empty_plan():
    api = FakeAPI(message({"type": "text", "text": "Nothing to do."}, stop_reason="end_turn"))
    assert ClaudePlanner(api.client())(REQUEST) == []


def test_a_refusal_means_no_actions():
    api = FakeAPI(
        message(
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": None, "explanation": "declined"},
        )
    )
    assert ClaudePlanner(api.client())(REQUEST) == []


def test_a_truncated_plan_is_not_acted_on():
    api = FakeAPI(message(tool_use("toolu_1", "lookup_order", order_id=7), stop_reason="max_tokens"))
    with pytest.raises(RuntimeError, match="cut off"):
        ClaudePlanner(api.client())(REQUEST)


# ---------------------------------------------------------------------- with the engine


@pytest.fixture
def db(tmp_path):
    database = persistkv.DB(str(tmp_path / "db"))
    yield database
    database.close()


def make_engine(db, planner, sink, saver=None):
    return Engine(Ledger(db), saver or PersistKVSaver(db), Policy(POLICY), sink, planner=planner)


class CountingPlanner(ClaudePlanner):
    def __init__(self, client):
        super().__init__(client)
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return super().__call__(request)


@pytest.mark.parametrize(
    "amount,expected",
    [(20000, "denied"), (10, "done")],
    ids=["oversized_is_denied", "small_is_automatic"],
)
def test_policy_decides_what_the_models_plan_may_do(db, amount, expected):
    api = FakeAPI(message(tool_use("toolu_1", "refund", order_id=1, account="a", amount=amount)))
    sink = RecordingSink()
    engine = make_engine(db, ClaudePlanner(api.client()), sink)
    engine.ledger.create_run("r", REQUEST)
    engine._drive("r")
    outcomes = engine.ledger.get_run("r")["outcomes"]
    assert [outcome["status"] for outcome in outcomes] == [expected]
    assert len(sink.calls) == (1 if expected == "done" else 0)


def test_large_refund_from_the_model_waits_for_a_human(db):
    api = FakeAPI(message(tool_use("toolu_1", "refund", order_id=1, account="a", amount=500)))
    sink = RecordingSink()
    engine = make_engine(db, ClaudePlanner(api.client()), sink)
    engine.ledger.create_run("r", REQUEST)
    engine._drive("r")
    assert engine.ledger.get_run("r")["status"] == "paused"
    assert sink.calls == []


def test_resuming_does_not_ask_the_model_again(db):
    """A model answers differently each time, with new call ids. If the engine re-planned on resume,
    the approved action would no longer match, and a second one could run."""
    api = FakeAPI(
        message(tool_use("toolu_first", "refund", order_id=1, account="a", amount=500)),
        message(tool_use("toolu_second", "refund", order_id=1, account="a", amount=999)),
    )
    sink = RecordingSink()
    planner = CountingPlanner(api.client())
    engine = make_engine(db, planner, sink)
    engine.ledger.create_run("r", REQUEST)
    engine._drive("r")
    action_id = engine.ledger.pending_actions()[0]["action_id"]
    engine.ledger.decide(action_id, "approved", actor="a", role="lead")

    make_engine(db, planner, sink)._drive("r")
    assert planner.calls == 1
    assert sink.calls == [action_id]
    assert engine.ledger.get_action(action_id)["args"]["amount"] == 500


@pytest.mark.parametrize("fail_at", range(1, 8))
def test_a_crash_never_leaves_two_plans_acting(db, fail_at):
    """A crash before the plan is saved may re-plan and get a different answer. That is safe only if
    nothing acted on the first answer, so at most one plan's actions may ever reach the sink."""
    api = FakeAPI(
        message(tool_use("toolu_first", "refund", order_id=1, account="a", amount=10)),
        message(tool_use("toolu_second", "refund", order_id=2, account="a", amount=20)),
    )
    sink = RecordingSink()
    planner = CountingPlanner(api.client())
    engine = make_engine(db, planner, sink, FailingSaver(db, fail_at))
    engine.ledger.create_run("r", REQUEST)
    try:
        engine._drive("r")
    except Interrupted:
        pass

    fresh = make_engine(db, planner, sink)
    fresh._drive("r")
    wait_until(lambda: fresh.ledger.get_run("r")["status"] == "completed")
    assert len(sink.calls) == 1
    assert fresh.ledger.verify_audit("r") is None
