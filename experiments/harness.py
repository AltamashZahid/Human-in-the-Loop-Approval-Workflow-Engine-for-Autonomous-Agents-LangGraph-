"""Shared machinery for the experiments: real processes, a scripted human, and verification.

Verification reads the databases directly, after both processes are dead. It counts duplicates
from the sink's `e/` effect records, which the dedupe logic never reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import persistkv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hitl.ledger import Ledger, make_action_id  # noqa: E402
from hitl.policy import Policy  # noqa: E402
from hitl.sink import Sink  # noqa: E402

FAST_POLICY = ROOT / "policies" / "fast.yaml"
DEFAULT_POLICY = ROOT / "policies" / "default.yaml"
DATA_DIR = Path(os.environ.get("HITL_DATA_DIR", Path.home() / "hitl" / "data" / "experiments"))
RESULTS_DIR = ROOT / "results"

# The synthetic mix: (share of tool calls, tool, amount range). Reported with every result.
TOOL_MIX = [
    (0.30, "lookup_order", None),
    (0.15, "send_notification", None),
    (0.10, "get_balance", None),
    (0.20, "refund", (1, 50)),
    (0.12, "refund", (51, 10000)),
    (0.03, "refund", (10001, 50000)),
    (0.10, "block_card", None),
]
HUMAN_MIX = [(0.70, "approve"), (0.20, "reject"), (0.10, "timeout")]


def weighted(rng: random.Random, table: list[tuple]) -> tuple:
    return rng.choices(table, weights=[row[0] for row in table])[0]


def make_call(rng: random.Random, order: int) -> dict[str, Any]:
    _, tool, amounts = weighted(rng, TOOL_MIX)
    args: dict[str, Any] = {"order_id": order}
    if tool == "refund":
        args.update(account=f"acct{rng.randrange(10)}", amount=rng.randint(*amounts))
    elif tool == "block_card":
        args["card"] = f"card{rng.randrange(1000)}"
    return {"tool": tool, "args": args}


def make_workload(seed: int, runs: int, *, prefix: str = "run") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    workload = []
    order = 0
    for index in range(runs):
        run_id = f"{prefix}-{seed}-{index}"
        actions = []
        for call_index in range(rng.randint(1, 3)):
            order += 1
            call = make_call(rng, order)
            call["action_id"] = make_action_id(run_id, f"call_{call_index}", call["tool"], call["args"])
            call["human"] = weighted(rng, HUMAN_MIX)[1]
            actions.append(call)
        workload.append({"run_id": run_id, "actions": actions})
    return workload


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Cluster:
    """An engine process and a sink process, on their own ports and databases."""

    def __init__(self, name: str, *, policy: Path = FAST_POLICY, engine_env=None, sink_env=None):
        self.dir = DATA_DIR / name
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.policy = policy
        self.engine_env = dict(engine_env or {})
        self.sink_env = dict(sink_env or {})
        self.engine_port = free_port()
        self.sink_port = free_port()
        self.engine_db = str(self.dir / "engine")
        self.sink_db = str(self.dir / "sink")
        self.engine: subprocess.Popen | None = None
        self.sink: subprocess.Popen | None = None
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.engine_port}", timeout=10.0)
        self.latency_log = str(self.dir / "latency.jsonl")

    def _spawn(self, module: str, env: dict[str, str], log_name: str) -> subprocess.Popen:
        log = open(self.dir / log_name, "ab")
        full_env = {
            **{k: v for k, v in os.environ.items() if not k.startswith("HITL_")},
            "PYTHONPATH": str(ROOT / "src"),
            **env,
        }
        process = subprocess.Popen(
            [sys.executable, "-m", module], env=full_env, stdout=log, stderr=log, cwd=str(self.dir)
        )
        log.close()
        return process

    @staticmethod
    def _wait_healthy(port: int, process: subprocess.Popen, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"process exited during startup: {process.returncode}")
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.02)
        raise RuntimeError("process did not become healthy")

    def start_sink(self, *, crash_at: str | None = None, wait: bool = True) -> None:
        env = {"HITL_SINK_DB": self.sink_db, "HITL_SINK_PORT": str(self.sink_port), **self.sink_env}
        if crash_at:
            env["HITL_CRASH_AT"] = crash_at
        self.sink = self._spawn("hitl.sink", env, "sink.log")
        if wait:
            self._wait_healthy(self.sink_port, self.sink)

    def start_engine(self, *, crash_at: str | None = None, wait: bool = True) -> None:
        env = {
            "HITL_DB": self.engine_db,
            "HITL_PORT": str(self.engine_port),
            "HITL_SINK_URL": f"http://127.0.0.1:{self.sink_port}",
            "HITL_POLICY": str(self.policy),
            "HITL_LATENCY_LOG": self.latency_log,
            **self.engine_env,
        }
        if crash_at:
            env["HITL_CRASH_AT"] = crash_at
        self.engine = self._spawn("hitl.server", env, "engine.log")
        if wait:
            self._wait_healthy(self.engine_port, self.engine)

    @staticmethod
    def kill(process: subprocess.Popen | None) -> None:
        """SIGKILL, then wait for the process to be gone, since it may hold the database open."""
        if process is None:
            return
        if process.poll() is None:
            process.send_signal(signal.SIGKILL)
        process.wait(timeout=30)

    def kill_all(self) -> None:
        self.kill(self.engine)
        self.kill(self.sink)

    def cleanup(self) -> None:
        self.client.close()
        self.kill_all()
        shutil.rmtree(self.dir, ignore_errors=True)

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response | None:
        """None means the engine could not be reached, which is expected while it is down."""
        try:
            return self.client.request(method, path, **kwargs)
        except httpx.TransportError:
            return None


class Human:
    """Decides pending approvals over the API. It only marks a decision as delivered after a real
    answer, so a decision that a crash swallowed is sent again, as a real client would."""

    def __init__(self, cluster: Cluster, workload: list[dict[str, Any]], *, timeouts_become=None):
        self.cluster = cluster
        self.plan = {a["action_id"]: a["human"] for run in workload for a in run["actions"]}
        self.timeouts_become = timeouts_become
        self.delivered: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(0.02):
            response = self.cluster.request("GET", "/approvals")
            if response is None or response.status_code != 200:
                continue
            for item in response.json()["pending"]:
                action_id = item["action_id"]
                choice = self.plan.get(action_id)
                if choice == "timeout" and self.timeouts_become:
                    choice = self.timeouts_become
                if choice not in ("approve", "reject") or action_id in self.delivered:
                    continue
                answer = self.cluster.request(
                    "POST",
                    f"/approvals/{action_id}/{choice}",
                    json={"actor": "scripted-human", "role": item["role"]},
                )
                # 403 means it escalated since the listing: send it to the new role next pass.
                if answer is not None and answer.status_code in (200, 404, 409):
                    self.delivered.add(action_id)


def submit_all(cluster: Cluster, workload: list[dict[str, Any]], stop: threading.Event) -> None:
    """Retried until accepted. Run ids make this safe across engine crashes."""
    for run in workload:
        body = {
            "run_id": run["run_id"],
            "actions": [{"tool": a["tool"], "args": a["args"]} for a in run["actions"]],
        }
        while not stop.is_set():
            response = cluster.request("POST", "/runs", json=body)
            if response is not None and response.status_code == 202:
                break
            time.sleep(0.02)


def wait_finished(cluster: Cluster, workload, timeout: float = 90.0) -> bool:
    wanted = {run["run_id"] for run in workload}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = cluster.request("GET", "/runs")
        if response is not None and response.status_code == 200:
            finished = {
                run["run_id"]
                for run in response.json()["runs"]
                if run["status"] in ("completed", "lost")
            }
            if wanted <= finished:
                return True
        time.sleep(0.05)
    return False


def verify(cluster: Cluster, workload: list[dict[str, Any]]) -> dict[str, Any]:
    """Call with both processes dead. Returns counters and a list of violated invariants."""
    failures: list[str] = []
    counts: Counter[str] = Counter()

    sink_db = persistkv.DB(cluster.sink_db)
    sink = Sink(sink_db)
    effects = Counter(effect["action_id"] for effect in sink.effects())
    balances = sink.balances()
    if sink_db.check_integrity():
        failures.append("sink database failed its integrity check")
    sink_db.close()

    engine_db = persistkv.DB(cluster.engine_db)
    ledger = Ledger(engine_db)
    if engine_db.check_integrity():
        failures.append("engine database failed its integrity check")

    expected_balances: Counter[str] = Counter()
    for spec in workload:
        run_id = spec["run_id"]
        run = ledger.get_run(run_id)
        counts["runs"] += 1
        if run is None:
            failures.append(f"{run_id}: never recorded")
            continue
        if run["status"] == "lost":
            counts["lost_runs"] += 1
            failures.append(f"{run_id}: lost its checkpoint")
            continue
        if run["status"] != "completed":
            counts["unfinished_runs"] += 1
            failures.append(f"{run_id}: stuck in {run['status']}")
            continue
        chain_error = ledger.verify_audit(run_id)
        if chain_error:
            failures.append(f"{run_id}: audit chain: {chain_error}")
        trail = ledger.audit_trail(run_id)
        if len(run.get("outcomes", [])) != len(spec["actions"]):
            failures.append(f"{run_id}: {len(run.get('outcomes', []))} outcomes")

        for call in spec["actions"]:
            action_id = call["action_id"]
            counts["actions"] += 1
            action = ledger.get_action(action_id)
            if action is None:
                failures.append(f"{action_id}: no ledger record")
                continue
            final = action["status"]
            counts[f"final_{final}"] += 1
            counts[f"route_{action['route']}"] += 1
            risky = call["tool"] == "block_card" or (
                call["tool"] == "refund" and call["args"]["amount"] > 50
            )
            if risky:
                counts["high_risk_actions"] += 1
                if action["route"] == "auto":
                    counts["high_risk_auto_approved"] += 1
                    failures.append(f"{action_id}: high-risk action auto-approved")
            if action.get("decided_by") == "timeout":
                counts["decided_by_timeout"] += 1
            applied = effects[action_id]
            counts["effects"] += applied

            # The core invariant, using the sink's count rather than anything the engine says.
            expected = 1 if final == "done" else 0
            if applied > expected:
                counts["duplicate_effects"] += applied - expected
                failures.append(f"{action_id}: {applied} effects, expected {expected}")
            elif applied < expected:
                counts["missing_effects"] += expected - applied
                failures.append(f"{action_id}: {applied} effects, expected {expected}")

            # Independent of the ledger's decision: what the script told the human to do.
            if action["route"] == "auto" and final != "done":
                failures.append(f"{action_id}: auto-approved but ended {final}")
            if action["route"] == "deny" and final != "denied":
                failures.append(f"{action_id}: denied by policy but ended {final}")
            if action["route"] == "approval":
                if call["human"] == "reject" and final == "done":
                    failures.append(f"{action_id}: executed although the human rejected it")
                if call["human"] == "timeout" and final not in ("timed_out", "rejected"):
                    failures.append(f"{action_id}: nobody answered but it ended {final}")

            events = [e for e in trail if e["data"].get("action_id") == action_id]
            kinds = Counter(e["event"] for e in events)
            if kinds["decision"] != 1:
                failures.append(f"{action_id}: {kinds['decision']} decision entries")
            if kinds["execution_done"] != (1 if final == "done" else 0):
                failures.append(f"{action_id}: {kinds['execution_done']} execution_done entries")
            counts["retried_executions"] += kinds["execution_retried"]
            counts["escalations"] += kinds["escalated"]
            if (action.get("result") or {}).get("replayed"):
                counts["sink_replays"] += 1
            if final == "done" and call["tool"] == "refund":
                expected_balances[call["args"]["account"]] += call["args"]["amount"]
    engine_db.close()

    # A second independent check: the money adds up. Accounts are shared across runs.
    if dict(expected_balances) != balances:
        failures.append(f"balances differ: expected {dict(expected_balances)}, sink has {balances}")
    counts["failures"] = len(failures)
    return {"counts": dict(counts), "failures": failures}


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def environment() -> dict[str, Any]:
    stat = subprocess.run(
        ["stat", "-f", "-c", "%T", str(DATA_DIR.parent)], capture_output=True, text=True, check=False
    )
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_hash": source_hash(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "data_filesystem": stat.stdout.strip(),
    }


def write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps({"environment": environment(), **payload}, indent=2) + "\n")
    return path


def policy_for(path: Path) -> Policy:
    return Policy.load(path)
