"""Crash-free measurements.

    python experiments/measure.py policy --runs 2000
    python experiments/measure.py latency --samples 2000 --concurrency 8 --trials 3

policy   Routes a mixed synthetic workload through the real engine and reports where it went.
         The auto-approval rate follows from the workload mix and the policy, so quote them together.

latency  From the start of Engine.decide to the first line of `execute`, the first work done after
         the run resumes. The clock is inside the engine process, so it excludes HTTP transport
         and request parsing on the way in.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness import (
    DEFAULT_POLICY,
    HUMAN_MIX,
    TOOL_MIX,
    Cluster,
    Human,
    make_workload,
    submit_all,
    verify,
    wait_finished,
    write_result,
)

from hitl.ledger import make_action_id


def percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank, so the result is always a latency that was actually observed."""
    rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
    return ordered[rank - 1]


def describe(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "samples": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "p50_ms": round(percentile(ordered, 0.50), 2),
        "p95_ms": round(percentile(ordered, 0.95), 2),
        "p99_ms": round(percentile(ordered, 0.99), 2),
        "max_ms": round(ordered[-1], 2),
    }


def policy(args: argparse.Namespace) -> None:
    cluster = Cluster("policy")
    workload = make_workload(args.seed, args.runs, prefix="policy")
    stop = threading.Event()
    try:
        cluster.start_sink()
        cluster.start_engine()
        human = Human(cluster, workload)
        human.start()
        submit_all(cluster, workload, stop)
        finished = wait_finished(cluster, workload, timeout=1800)
        human.stop()
        cluster.kill_all()
        result = verify(cluster, workload)
    finally:
        stop.set()
        cluster.cleanup()

    counts = result["counts"]
    total = counts["actions"]
    share = lambda name: round(100 * counts.get(name, 0) / total, 2)  # noqa: E731
    summary = {
        "runs": counts["runs"],
        "actions": total,
        "finished": finished,
        "auto_approved_pct": share("route_auto"),
        "sent_to_human_pct": share("route_approval"),
        "denied_by_policy_pct": share("route_deny"),
        "high_risk_actions": counts.get("high_risk_actions", 0),
        "high_risk_auto_approved": counts.get("high_risk_auto_approved", 0),
        "decided_by_timeout": counts.get("decided_by_timeout", 0),
        "escalations": counts.get("escalations", 0),
        "duplicate_effects": counts.get("duplicate_effects", 0),
        "failures": result["failures"][:20],
        "counts": counts,
    }
    params = {"seed": args.seed, "tool_mix": TOOL_MIX, "human_mix": HUMAN_MIX, "policy": cluster.policy.name}
    print(write_result(args.output or "policy_routing", {"parameters": params, **summary}))
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, indent=2))


def latency_trial(args: argparse.Namespace, trial: int) -> list[float]:
    # Long deadlines, so that nothing expires, and one risky call per run.
    cluster = Cluster(f"latency-{trial}", policy=DEFAULT_POLICY)
    try:
        cluster.start_sink()
        cluster.start_engine()

        def one(index: int) -> None:
            run_id = f"lat-{trial}-{index}"
            call = {"tool": "refund", "args": {"order_id": index, "account": "a", "amount": 100}}
            action_id = make_action_id(run_id, "call_0", call["tool"], call["args"])
            response = cluster.client.post("/runs", json={"run_id": run_id, "actions": [call]})
            response.raise_for_status()
            while cluster.client.get(f"/runs/{run_id}").json()["status"] != "paused":
                time.sleep(0.002)
            cluster.client.post(
                f"/approvals/{action_id}/approve",
                json={"actor": "scripted-human", "role": "support_lead"},
            ).raise_for_status()
            while cluster.client.get(f"/runs/{run_id}").json()["status"] != "completed":
                time.sleep(0.002)

        with ThreadPoolExecutor(args.concurrency) as pool:
            for future in [pool.submit(one, index) for index in range(args.samples)]:
                future.result()
        cluster.kill_all()
        lines = Path(cluster.latency_log).read_text().splitlines()
        return [json.loads(line)["ms"] for line in lines]
    finally:
        cluster.cleanup()


def latency(args: argparse.Namespace) -> None:
    trials = []
    pooled: list[float] = []
    for trial in range(args.trials):
        samples = latency_trial(args, trial)
        trials.append(describe(samples))
        pooled += samples
        print(f"trial {trial}: {trials[-1]}", flush=True)
    params = {"samples_per_trial": args.samples, "trials": args.trials, "concurrency": args.concurrency,
              "fsync": "every commit (sync_writes=True)", "transport": "HTTP over loopback"}
    name = args.output or f"resume_latency_c{args.concurrency}"
    summary = {"parameters": params, "trials": trials, "pooled": describe(pooled)}
    print(write_result(name, summary))
    print("pooled:", summary["pooled"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("experiment", choices=["policy", "latency"])
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output")
    args = parser.parse_args()
    {"policy": policy, "latency": latency}[args.experiment](args)


if __name__ == "__main__":
    main()
