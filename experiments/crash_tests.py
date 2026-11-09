"""Kills real processes with SIGKILL, restarts them, and checks that nothing was lost or repeated.

    python experiments/crash_tests.py matrix --rounds 100
    python experiments/crash_tests.py random --rounds 100
    python experiments/crash_tests.py controls --rounds 10

matrix    One targeted crash per round, at each named point, including the hard case.
random    Kills the engine at random moments while the workload runs, possibly several times.
controls  Turns a safety mechanism off and requires the same checks to catch the result. If these
          came back clean, a zero from the other two experiments would mean nothing.

A round whose crash was never reached does not count as a test, and is reported as `missed`.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from harness import (
    DEFAULT_POLICY,
    FAST_POLICY,
    TOOL_MIX,
    HUMAN_MIX,
    Cluster,
    Human,
    make_workload,
    submit_all,
    verify,
    wait_finished,
    write_result,
)

RUNS_PER_ROUND = 8

# Crash point -> the highest n to draw for "crash on the nth time this point is reached".
ENGINE_POINTS = {
    "engine.after_decision": 3,
    "exec.after_intent": 6,
    "exec.after_sink": 6,  # the hard case: effect committed, "done" not yet recorded
    "exec.after_done": 6,
    "ckpt.before_commit": 40,
    "ckpt.after_commit": 40,
}
SINK_POINTS = {"sink.after_commit": 6}


def crash_round(name: str, point: str, nth: int, seed: int, **cluster_args: Any) -> dict[str, Any]:
    """One crash at a named point in the engine or the sink, then recovery and verification."""
    cluster = Cluster(name, **cluster_args)
    workload = make_workload(seed, RUNS_PER_ROUND)
    stop = threading.Event()
    in_sink = point in SINK_POINTS
    crash_at = f"{point}:{nth}"
    try:
        cluster.start_sink(crash_at=crash_at if in_sink else None)
        cluster.start_engine(crash_at=None if in_sink else crash_at)
        human = Human(cluster, workload)
        human.start()
        submitter = threading.Thread(target=submit_all, args=(cluster, workload, stop), daemon=True)
        submitter.start()

        target = cluster.sink if in_sink else cluster.engine
        deadline = time.monotonic() + 60
        hit = False
        while time.monotonic() < deadline:
            if target.poll() is not None:
                hit = target.returncode == -9
                break
            if not submitter.is_alive() and wait_finished(cluster, workload, timeout=0.2):
                break
            time.sleep(0.01)

        if hit:
            if in_sink:
                cluster.start_sink()
            else:
                cluster.start_engine()
        finished = wait_finished(cluster, workload)
        stop.set()
        human.stop()
        cluster.kill_all()
        result = verify(cluster, workload)
        return {"point": point, "nth": nth, "seed": seed, "hit": hit, "finished": finished, **result}
    finally:
        stop.set()
        cluster.cleanup()


def paused_round(name: str, seed: int, **cluster_args: Any) -> dict[str, Any]:
    """Waits until every run has paused or finished, kills the engine, and checks that each paused
    run comes back paused at the same checkpoint, before anyone decides anything."""
    cluster = Cluster(name, policy=DEFAULT_POLICY, **cluster_args)
    workload = make_workload(seed, RUNS_PER_ROUND)
    stop = threading.Event()
    try:
        cluster.start_sink()
        cluster.start_engine()
        submit_all(cluster, workload, stop)
        wanted = {run["run_id"] for run in workload}
        paused: dict[str, str] = {}
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            runs = cluster.request("GET", "/runs").json()["runs"]
            if {r["run_id"] for r in runs if r["status"] in ("paused", "completed")} >= wanted:
                paused = {r["run_id"]: r["paused_checkpoint"] for r in runs if r["status"] == "paused"}
                break
            time.sleep(0.01)

        cluster.kill(cluster.engine)
        cluster.start_engine()

        intact = 0
        time.sleep(0.2)  # time for recovery to find a missing checkpoint and mark the run lost
        for run_id, checkpoint in paused.items():
            run = cluster.request("GET", f"/runs/{run_id}").json()
            if run["status"] == "paused" and run["paused_checkpoint"] == checkpoint:
                intact += 1

        # Nobody answered during the outage, so the default policy's long deadlines still hold.
        # "timeout" becomes "reject" so that the round does not have to wait them out.
        human = Human(cluster, workload, timeouts_become="reject")
        human.start()
        finished = wait_finished(cluster, workload)
        human.stop()
        cluster.kill_all()
        result = verify(cluster, workload)
        result["counts"]["paused_at_kill"] = len(paused)
        result["counts"]["paused_intact_after_restart"] = intact
        if intact != len(paused):
            result["failures"].append(f"{len(paused) - intact} paused runs changed across the kill")
            result["counts"]["failures"] = len(result["failures"])
        return {"point": "while_paused", "seed": seed, "hit": True, "finished": finished, **result}
    finally:
        stop.set()
        cluster.cleanup()


def random_round(name: str, seed: int) -> dict[str, Any]:
    """Kills the engine one to three times at random moments during a workload."""
    rng = random.Random(seed)
    cluster = Cluster(name)
    workload = make_workload(seed, RUNS_PER_ROUND)
    stop = threading.Event()
    kills = rng.randint(1, 3)
    try:
        cluster.start_sink()
        cluster.start_engine()
        human = Human(cluster, workload)
        human.start()
        threading.Thread(target=submit_all, args=(cluster, workload, stop), daemon=True).start()
        for _ in range(kills):
            time.sleep(rng.uniform(0.0, 0.6))
            cluster.kill(cluster.engine)
            cluster.start_engine()
        finished = wait_finished(cluster, workload)
        stop.set()
        human.stop()
        cluster.kill_all()
        result = verify(cluster, workload)
        return {"point": "random", "seed": seed, "kills": kills, "hit": True, "finished": finished, **result}
    finally:
        stop.set()
        cluster.cleanup()


def run_parallel(tasks: list, workers: int) -> list[dict[str, Any]]:
    results = []

    def guarded(task):
        try:
            return task()
        except Exception as error:  # a harness error must be visible, never a silent pass
            return {"point": "harness_error", "hit": False, "finished": False, "error": repr(error),
                    "counts": {"failures": 1}, "failures": [repr(error)]}

    with ThreadPoolExecutor(workers) as pool:
        for index, result in enumerate(pool.map(guarded, tasks), 1):
            results.append(result)
            if index % 25 == 0:
                bad = sum(1 for r in results if r["failures"])
                print(f"  {index}/{len(tasks)} rounds, {bad} with failures", flush=True)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_point: dict[str, dict[str, Any]] = {}
    for result in results:
        entry = per_point.setdefault(result["point"], {"rounds": 0, "missed": 0, "failed_rounds": 0,
                                                       "kills": 0, "counts": Counter()})
        if not result["hit"]:
            entry["missed"] += 1
            continue
        entry["rounds"] += 1
        entry["kills"] += result.get("kills", 1)
        if result["failures"] or not result["finished"]:
            entry["failed_rounds"] += 1
        entry["counts"].update(result["counts"])
    for entry in per_point.values():
        entry["counts"] = dict(entry["counts"])

    valid = [r for r in results if r["hit"]]
    total: Counter[str] = Counter()
    for result in valid:
        total.update(result["counts"])
    return {
        "rounds": len(valid),
        "missed_rounds": len(results) - len(valid),
        "total_kills": sum(r.get("kills", 1) for r in valid),
        "failed_rounds": sum(1 for r in valid if r["failures"] or not r["finished"]),
        "totals": dict(total),
        "per_point": per_point,
        "first_failures": [
            {"point": r["point"], "seed": r.get("seed"), "failures": r["failures"][:5]}
            for r in results
            if r["failures"]
        ][:20],
    }


def parameters(rounds: int) -> dict[str, Any]:
    return {"rounds_per_case": rounds, "runs_per_round": RUNS_PER_ROUND, "tool_mix": TOOL_MIX,
            "human_mix": HUMAN_MIX, "policy": FAST_POLICY.name}


def matrix(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    tasks = []
    points = {**ENGINE_POINTS, **SINK_POINTS}
    for point, highest in points.items():
        for index in range(args.rounds):
            nth, seed = rng.randint(1, highest), rng.randrange(1 << 30)
            name = f"m-{point}-{index}"
            tasks.append(lambda name=name, point=point, nth=nth, seed=seed: crash_round(name, point, nth, seed))
    for index in range(args.rounds):
        seed = rng.randrange(1 << 30)
        tasks.append(lambda name=f"m-paused-{index}", seed=seed: paused_round(name, seed))
    summary = summarize(run_parallel(tasks, args.workers))
    print(write_result(args.output or "crash_matrix", {"parameters": parameters(args.rounds), **summary}))
    report(summary)


def random_kills(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    tasks = []
    for index in range(args.rounds):
        seed = rng.randrange(1 << 30)
        tasks.append(lambda name=f"r-{index}", seed=seed: random_round(name, seed))
    summary = summarize(run_parallel(tasks, args.workers))
    print(write_result(args.output or "random_kills", {"parameters": parameters(args.rounds), **summary}))
    report(summary)


def controls(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    tasks = []
    for index in range(args.rounds):
        nth, seed = rng.randint(1, 6), rng.randrange(1 << 30)
        tasks.append(lambda name=f"c-dedupe-{index}", nth=nth, seed=seed: {
            **crash_round(name, "exec.after_sink", nth, seed, sink_env={"HITL_UNSAFE_SINK_NO_DEDUPE": "1"}),
            "point": "sink_dedupe_off",
        })
        nth, seed = rng.randint(1, 6), rng.randrange(1 << 30)
        tasks.append(lambda name=f"c-key-{index}", nth=nth, seed=seed: {
            **crash_round(name, "exec.after_sink", nth, seed, engine_env={"HITL_UNSAFE_RANDOM_KEY": "1"}),
            "point": "random_idempotency_key",
        })
        seed = rng.randrange(1 << 30)
        tasks.append(lambda name=f"c-mem-{index}", seed=seed: {
            **paused_round(name, seed, engine_env={"HITL_UNSAFE_MEMORY_CHECKPOINTER": "1"}),
            "point": "in_memory_checkpointer",
        })
    summary = summarize(run_parallel(tasks, args.workers))
    print(write_result(args.output or "negative_controls", {"parameters": parameters(args.rounds), **summary}))
    report(summary)


def report(summary: dict[str, Any]) -> None:
    print(f"rounds={summary['rounds']} kills={summary['total_kills']} missed={summary['missed_rounds']} "
          f"failed_rounds={summary['failed_rounds']}")
    print(f"{'case':28}{'rounds':>7}{'missed':>7}{'failed':>7}{'dup':>6}{'lost':>6}{'retries':>8}{'replays':>8}")
    for point, entry in summary["per_point"].items():
        counts = entry["counts"]
        print(f"{point:28}{entry['rounds']:>7}{entry['missed']:>7}{entry['failed_rounds']:>7}"
              f"{counts.get('duplicate_effects', 0):>6}{counts.get('lost_runs', 0):>6}"
              f"{counts.get('retried_executions', 0):>8}{counts.get('sink_replays', 0):>8}")
    for failure in summary["first_failures"][:5]:
        print(" ", failure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("experiment", choices=["matrix", "random", "controls"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output", help="result file name, for smoke runs that must not overwrite")
    args = parser.parse_args()
    {"matrix": matrix, "random": random_kills, "controls": controls}[args.experiment](args)


if __name__ == "__main__":
    main()
