"""In-process tests of the policy, ledger and engine, with a fake clock where timing matters."""

from __future__ import annotations

import threading
import time

import persistkv
import pytest

from hitl.checkpointer import PersistKVSaver
from hitl.engine import Engine
from hitl.ledger import Conflict, Ledger, NotAllowed, make_action_id
from hitl.policy import Policy, PolicyError, Route, Tier
from hitl.sink import ExecuteRequest, Sink

POLICY = {
    "default": {"route": "approval", "escalation": [{"role": "ops", "timeout_s": 60}]},
    "rules": [
        {"name": "read", "when": {"tool": {"in": ["lookup_order"]}}, "route": "auto"},
        {"name": "small", "when": {"tool": "refund", "args.amount": {"lte": 50}}, "route": "auto"},
        {"name": "huge", "when": {"tool": "refund", "args.amount": {"gt": 10000}}, "route": "deny"},
        {
            "name": "large",
            "when": {"tool": "refund", "args.amount": {"gt": 50}},
            "route": "approval",
            "escalation": [
                {"role": "lead", "timeout_s": 10},
                {"role": "manager", "timeout_s": 20},
            ],
        },
    ],
}
LARGE = Route("approval", "large", (Tier("lead", 10), Tier("manager", 20)))


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def db(tmp_path):
    database = persistkv.DB(str(tmp_path / "db"))
    yield database
    database.close()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ledger(db, clock):
    return Ledger(db, clock)


# ---------------------------------------------------------------------- policy


@pytest.mark.parametrize(
    "tool,args,expected",
    [
        ("lookup_order", {}, "read"),
        ("refund", {"amount": 50}, "small"),
        ("refund", {"amount": 51}, "large"),
        ("refund", {"amount": 10001}, "huge"),
        ("unknown_tool", {}, "default"),
    ],
)
def test_policy_routes_by_rule(tool, args, expected):
    assert Policy(POLICY).evaluate(tool, args).rule == expected


@pytest.mark.parametrize(
    "args",
    [{}, {"amount": "1"}, {"amount": None}, {"amount": True}, {"amount": [1]}],
    ids=["missing", "string", "null", "bool", "list"],
)
def test_malformed_arguments_never_auto_approve(args):
    assert Policy(POLICY).evaluate("refund", args).route == "approval"


@pytest.mark.parametrize(
    "spec",
    [
        {"rules": []},
        {"default": {"route": "auto"}},
        {"default": {"route": "approval"}},
        {"default": {"route": "approve", "escalation": []}},
        {"default": {"route": "deny"}, "rules": [{"name": "a", "route": "auto"}]},
        {"default": {"route": "deny"}, "rules": [{"name": "a", "when": {"x": {"lte": "1"}}, "route": "auto"}]},
        {"default": {"route": "deny"}, "rules": [{"name": "a", "when": {"x": {"lt3": 1}}, "route": "auto"}]},
        {"default": {"route": "deny"}, "rules": [{"name": "a", "when": {"x": 1}, "route": "deny"}] * 2},
    ],
)
def test_invalid_policies_are_refused(spec):
    with pytest.raises(PolicyError):
        Policy(spec)


# ---------------------------------------------------------------------- ledger


def propose(ledger, run_id="r", route=LARGE, amount=100):
    ledger.create_run(run_id, {})
    return ledger.propose(run_id, "call_0", "refund", {"amount": amount}, route)


def test_action_id_is_deterministic_and_covers_every_input():
    base = make_action_id("r", "c", "refund", {"a": 1, "b": 2})
    assert base == make_action_id("r", "c", "refund", {"b": 2, "a": 1})
    assert base != make_action_id("r2", "c", "refund", {"a": 1, "b": 2})
    assert base != make_action_id("r", "c2", "refund", {"a": 1, "b": 2})
    assert base != make_action_id("r", "c", "other", {"a": 1, "b": 2})
    assert base != make_action_id("r", "c", "refund", {"a": 2, "b": 2})


def test_re_proposing_does_not_duplicate_audit_or_reset_state(ledger):
    first = propose(ledger)
    audit_size = len(ledger.audit_trail("r"))
    ledger.decide(first["action_id"], "approved", actor="a", role="lead")
    assert propose(ledger)["status"] == "approved"
    assert len(ledger.audit_trail("r")) == audit_size + 1


def test_first_decision_wins_and_repeating_it_is_harmless(ledger):
    action_id = propose(ledger)["action_id"]
    ledger.decide(action_id, "approved", actor="a", role="lead")
    ledger.decide(action_id, "approved", actor="a", role="lead")
    with pytest.raises(Conflict):
        ledger.decide(action_id, "rejected", actor="b", role="lead")
    decisions = [e for e in ledger.audit_trail("r") if e["event"] == "decision"]
    assert len(decisions) == 1


def test_repeating_a_decision_after_execution_is_harmless(ledger):
    action_id = propose(ledger)["action_id"]
    ledger.decide(action_id, "approved", actor="a", role="lead")
    ledger.mark_done(action_id, {"ok": True})
    assert ledger.decide(action_id, "approved", actor="a", role="lead")["status"] == "done"
    with pytest.raises(Conflict):
        ledger.decide(action_id, "rejected", actor="a", role="lead")


def test_only_the_current_tiers_role_may_decide(ledger, clock):
    action_id = propose(ledger)["action_id"]
    with pytest.raises(NotAllowed):
        ledger.decide(action_id, "approved", actor="a", role="manager")
    clock.now += 11
    ledger.expire_due()
    with pytest.raises(NotAllowed):
        ledger.decide(action_id, "approved", actor="a", role="lead")
    assert ledger.decide(action_id, "approved", actor="a", role="manager")["status"] == "approved"


def test_timeouts_escalate_through_tiers_and_then_reject(ledger, clock):
    action_id = propose(ledger)["action_id"]
    clock.now += 9
    assert ledger.expire_due() == []
    assert ledger.get_action(action_id)["tier"] == 0
    clock.now += 2
    assert ledger.expire_due() == []
    assert ledger.get_action(action_id)["tier"] == 1
    # The new tier's clock starts at the escalation, not at the original request.
    clock.now += 19
    assert ledger.expire_due() == []
    assert ledger.get_action(action_id)["status"] == "pending"
    clock.now += 2
    assert [a["status"] for a in ledger.expire_due()] == ["timed_out"]
    assert ledger.pending_actions() == []
    assert ledger.expire_due() == []
    with pytest.raises(Conflict):
        ledger.decide(action_id, "approved", actor="late", role="manager")


def test_deadline_that_passed_during_downtime_fires_after_reopen(tmp_path, clock):
    path = str(tmp_path / "db")
    db = persistkv.DB(path)
    action_id = propose(Ledger(db, clock))["action_id"]
    db.close()

    clock.now += 11
    db = persistkv.DB(path)
    ledger = Ledger(db, clock)
    ledger.expire_due()
    assert ledger.get_action(action_id)["tier"] == 1
    db.close()


def test_approval_racing_a_timeout_has_exactly_one_winner(ledger, clock):
    outcomes = {"approved": 0, "timed_out": 0}
    for index in range(100):
        action_id = propose(ledger, run_id=f"race{index}", route=Route("approval", "r", (Tier("ops", 1),)))[
            "action_id"
        ]
        clock.now += 2
        barrier = threading.Barrier(2)

        def approve():
            barrier.wait()
            try:
                ledger.decide(action_id, "approved", actor="a", role="ops")
            except Conflict:
                pass

        def expire():
            barrier.wait()
            ledger.expire_due()

        threads = [threading.Thread(target=approve), threading.Thread(target=expire)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        final = ledger.get_action(action_id)["status"]
        outcomes[final] += 1
        decisions = [e for e in ledger.audit_trail(f"race{index}") if e["event"] == "decision"]
        assert len(decisions) == 1
        assert decisions[0]["data"]["outcome"] == final
    assert sum(outcomes.values()) == 100


def test_concurrent_audit_appends_across_runs_keep_every_chain_valid(ledger):
    def work(run_id):
        for index in range(20):
            ledger.create_run(run_id, {})
            ledger.propose(run_id, f"c{index}", "t", {}, Route("deny", "x"))

    threads = [threading.Thread(target=work, args=[f"run{i}"]) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for index in range(8):
        assert ledger.verify_audit(f"run{index}") is None
        # run_started, then proposed and decision for each of the 20 calls.
        assert len(ledger.audit_trail(f"run{index}")) == 41


def test_audit_chain_detects_tampering(ledger, db):
    propose(ledger)
    assert ledger.verify_audit("r") is None
    key = b"Lv/r/00000001"
    db.put(key, db.get(key).replace(b"refund", b"reFund"))
    assert "bad hash" in ledger.verify_audit("r")


def test_audit_chain_detects_a_removed_entry(ledger, db):
    propose(ledger)
    db.remove(b"Lv/r/00000001")
    assert ledger.verify_audit("r") is not None


# ---------------------------------------------------------------------- sink


def request(key="k", amount=10):
    return ExecuteRequest(
        idempotency_key=key, action_id=key, tool="refund", args={"account": "a", "amount": amount}
    )


def test_sink_applies_a_repeated_key_once_and_returns_the_same_result(tmp_path):
    db = persistkv.DB(str(tmp_path / "sink"))
    sink = Sink(db)
    first, second = sink.execute(request()), sink.execute(request())
    assert (first["replayed"], second["replayed"]) == (False, True)
    assert first["effect_id"] == second["effect_id"]
    assert len(sink.effects()) == 1
    assert sink.balances() == {"a": 10}
    db.close()


def test_sink_under_concurrent_duplicates_applies_once(tmp_path):
    db = persistkv.DB(str(tmp_path / "sink"))
    sink = Sink(db)
    threads = [
        threading.Thread(target=sink.execute, args=[request(f"k{index % 5}")]) for index in range(40)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(sink.effects()) == 5
    assert sink.balances() == {"a": 50}
    db.close()


# ---------------------------------------------------------------------- engine, in process


class RecordingSink:
    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[str] = []
        self.fail_times = fail_times

    def execute(self, key, action):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("sink down")
        self.calls.append(key)
        return {"effect_id": key}


def make_engine(db, sink=None, clock=None, **options):
    return Engine(
        Ledger(db, clock or time.time),
        PersistKVSaver(db),
        Policy(POLICY),
        sink or RecordingSink(),
        scan_interval_s=0.01,
        **options,
    )


def wait_until(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting")


REFUND = {"tool": "refund", "args": {"amount": 100}}


def run_status(engine, run_id):
    run = engine.ledger.get_run(run_id)
    return run["status"] if run else None


@pytest.fixture
def engine(db):
    instance = make_engine(db)
    instance.start()
    yield instance
    instance.stop()


def test_auto_approved_call_runs_without_a_human(engine):
    engine.start_run("r", {"actions": [{"tool": "refund", "args": {"amount": 10}}]})
    wait_until(lambda: run_status(engine, "r") == "completed")
    assert len(engine.sink.calls) == 1


def test_denied_call_never_reaches_the_sink(engine):
    engine.start_run("r", {"actions": [{"tool": "refund", "args": {"amount": 20000}}]})
    wait_until(lambda: run_status(engine, "r") == "completed")
    assert engine.sink.calls == []
    assert engine.ledger.get_run("r")["outcomes"][0]["status"] == "denied"


def test_risky_call_pauses_until_approved_then_runs_once(engine):
    engine.start_run("r", {"actions": [REFUND]})
    wait_until(lambda: run_status(engine, "r") == "paused")
    assert engine.sink.calls == []
    action_id = engine.ledger.pending_actions()[0]["action_id"]
    engine.decide(action_id, "approved", actor="a", role="lead")
    wait_until(lambda: run_status(engine, "r") == "completed")
    assert engine.sink.calls == [action_id]
    assert engine.ledger.get_run("r")["outcomes"] == [{"action_id": action_id, "status": "done"}]


def test_rejected_call_never_reaches_the_sink(engine):
    engine.start_run("r", {"actions": [REFUND]})
    wait_until(lambda: run_status(engine, "r") == "paused")
    action_id = engine.ledger.pending_actions()[0]["action_id"]
    engine.decide(action_id, "rejected", actor="a", role="lead")
    wait_until(lambda: run_status(engine, "r") == "completed")
    assert engine.sink.calls == []
    assert engine.ledger.get_run("r")["outcomes"][0]["status"] == "rejected"


def test_run_with_several_actions_pauses_at_each_risky_one(engine):
    actions = [
        {"tool": "lookup_order", "args": {"id": 1}},
        {"tool": "refund", "args": {"amount": 100}},
        {"tool": "refund", "args": {"amount": 5}},
        {"tool": "block_card", "args": {"card": "c"}},
    ]
    engine.start_run("r", {"actions": actions})
    for role in ("lead", "ops"):
        wait_until(lambda: engine.ledger.pending_actions() and run_status(engine, "r") == "paused")
        engine.decide(engine.ledger.pending_actions()[0]["action_id"], "approved", actor="a", role=role)
        wait_until(lambda: not engine.ledger.pending_actions() or run_status(engine, "r") == "completed")
    wait_until(lambda: run_status(engine, "r") == "completed")
    assert len(engine.sink.calls) == 4
    assert engine.ledger.verify_audit("r") is None


def test_transient_sink_failure_retries_with_the_same_key_and_no_second_decision(db):
    sink = RecordingSink(fail_times=1)
    engine = make_engine(db, sink)
    engine.start()
    engine.start_run("r", {"actions": [{"tool": "refund", "args": {"amount": 10}}]})
    wait_until(lambda: run_status(engine, "r") == "completed")
    engine.stop()
    events = [entry["event"] for entry in engine.ledger.audit_trail("r")]
    assert events.count("execution_intent") == 1
    assert events.count("execution_retried") == 1
    assert events.count("decision") == 1
    assert len(sink.calls) == 1


def test_execute_refuses_an_action_the_ledger_does_not_authorize(db):
    engine = make_engine(db)
    engine.ledger.create_run("r", {})
    action = engine.ledger.propose("r", "c", "refund", {"amount": 100}, LARGE)
    engine.ledger.decide(action["action_id"], "rejected", actor="a", role="lead")
    with pytest.raises(RuntimeError, match="refusing to execute"):
        engine._node_execute({"action_id": action["action_id"]})
    assert engine.sink.calls == []


class Interrupted(Exception):
    pass


class FailingSaver(PersistKVSaver):
    """Stops at the nth `put`, leaving what a crash between a node's writes and the next
    checkpoint leaves behind: writes saved, and no checkpoint for the following step."""

    def __init__(self, db, fail_at):
        super().__init__(db)
        self.puts = 0
        self.fail_at = fail_at

    def put(self, *args, **kwargs):
        self.puts += 1
        if self.puts == self.fail_at:
            raise Interrupted
        return super().put(*args, **kwargs)


@pytest.mark.parametrize("fail_at", range(1, 12))
def test_run_interrupted_between_writes_and_checkpoint_finishes_all_its_work(db, fail_at):
    """Regression test, found by the crash experiments. `snapshot.next` is empty at this point,
    and treating that as "finished" completed runs with their actions never carried out."""
    sink = RecordingSink()
    actions = [{"tool": "refund", "args": {"amount": 10}}, {"tool": "lookup_order", "args": {}}]
    engine = Engine(Ledger(db), FailingSaver(db, fail_at), Policy(POLICY), sink)
    engine.ledger.create_run("r", {"actions": actions})
    try:
        engine._drive("r")
    except Interrupted:
        pass

    fresh = make_engine(db, sink)
    fresh._drive("r")
    run = fresh.ledger.get_run("r")
    assert run["status"] == "completed"
    assert [outcome["status"] for outcome in run["outcomes"]] == ["done", "done"]
    assert len(sink.calls) == 2


def test_paused_run_with_a_missing_checkpoint_is_reported_lost_not_rerun(db):
    engine = make_engine(db)
    engine.start()
    engine.start_run("r", {"actions": [REFUND]})
    wait_until(lambda: run_status(engine, "r") == "paused")
    engine.stop()

    engine.checkpointer.delete_thread("r")
    fresh = make_engine(db)
    fresh._drive("r")
    assert run_status(fresh, "r") == "lost"
    assert fresh.sink.calls == []
