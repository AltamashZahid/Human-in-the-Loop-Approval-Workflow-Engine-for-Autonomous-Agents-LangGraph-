# Human-in-the-Loop Approval Engine for LangGraph Agents

An agent that wants to issue a refund or block a card should not always do it on its own. This engine routes each tool call through a risk policy, pauses the risky ones until a human signs off, and keeps its promises across crashes:

- A paused run survives `kill -9` and comes back paused at the same checkpoint.
- An approved action takes effect once, even if the process dies right after the action ran but before it was recorded as done.
- A decision has one winner, including when an approver races a timeout.

State lives in [PersistKV](https://github.com/AltamashZahid/persist_KV), my crash-safe B+Tree key-value store, through a custom LangGraph checkpointer.

## Results

Every number comes from a real run, and the files in [results/](results/) hold the detail. All of them were produced from the same source (hash `cd974268bde497af`) on 2026-11-09.

| Claim | Result | File |
|---|---|---|
| Paused runs survive `kill -9` | 304 of 304 paused runs came back at the same checkpoint, over 100 kills | [crash_matrix.json](results/crash_matrix.json) |
| No lost runs | 0 lost across 878 engine kills and 6,272 runs | [crash_matrix.json](results/crash_matrix.json), [random_kills.json](results/random_kills.json) |
| No duplicate effects | 0 across 978 kills (878 engine, 100 sink), 14,131 actions and 12,787 effects | same |
| The hard case: crash after the effect, before "done" | 100 targeted kills, 132 retries answered from the stored result, 0 duplicates | [crash_matrix.json](results/crash_matrix.json) |
| Risk routing | 74.59% of 3,983 calls auto-approved, 22.22% sent to a human, 3.19% denied | [policy_routing.json](results/policy_routing.json) |
| No risky call skips review | 0 of 1,012 high-risk calls auto-approved | [policy_routing.json](results/policy_routing.json) |
| Resume latency, 8 runs in flight | p50 65 ms, p99 150 ms, over 6,000 approvals | [resume_latency_c8.json](results/resume_latency_c8.json) |
| Resume latency, 1 run in flight | p50 37 ms, p99 68 ms, over 6,000 approvals | [resume_latency_c1.json](results/resume_latency_c1.json) |
| Checkpointer correctness | All 58 tests of LangGraph's official conformance suite for required features | `tests/test_checkpointer.py` |

### Why those zeros mean something

A test that cannot fail proves nothing, so each safety mechanism was also switched off, and the same checks had to catch the result. They did, in all 60 rounds:

| Safety mechanism turned off | Rounds | What the checks caught |
|---|---:|---|
| Sink deduplication | 20 | 23 duplicate effects |
| Deterministic idempotency keys (random key per attempt) | 20 | 26 duplicate effects |
| Durable checkpointer (`InMemorySaver`) | 20 | 63 of 63 paused runs lost |

Duplicates are counted from effect records in the sink's own database, which the deduplication code never reads. Refund totals are also reconciled against the sink's balances.

### How to read these numbers

- **A zero is not proof of never.** Zero failures in 978 kills puts the failure rate per kill below about 0.3%, at 95% confidence.
- **`kill -9` is a process crash, not a power cut.** Commits do call fsync, but these tests do not exercise power loss, and this project makes no claim about it.
- **The auto-approval rate comes from the workload.** The synthetic mix has 75% low-risk calls by design, so 74.59% shows the routing works. It is not a finding about real traffic.
- **The latency clock is inside the engine.** It runs from the start of the decision call to the first line of the `execute` node. It includes the durable decision and the checkpoint commits, and excludes HTTP transport and parsing on the way in.
- **The machine affects the latency.** It ran on WSL2 with ext4 on a virtual disk, on an i5-13500H with 16 logical CPUs and 7 GB of RAM, with the test client polling on the same machine. Per-trial p99 ranged from 121 to 162 ms at 8 in flight, and from 48 to 72 ms at 1.
- **16 rounds are excluded.** They were aimed at `engine.after_decision` but ended before reaching the crash point, so no kill happened. The 978 counts only kills that did happen.

## How it works

```
             POST /runs                       POST /approvals/{id}/approve
                 |                                       |
                 v                                       v
  +--------------------------- engine process ---------------------------+
  |                                                                      |
  |   plan --> route --> approval --> execute --> advance --> ...        |
  |              |      (interrupt)      ^           ^                   |
  |              +--- policy: auto ------+           |                   |
  |              +--- policy: deny ------------------+                   |
  |                                                                      |
  |   PersistKV: checkpoints | ledger | audit log | deadlines            |
  +--------------------------------|-------------------------------------+
                                   | HTTP, with an idempotency key
                                   v
  +---------------------------- sink process ----------------------------+
  |   PersistKV: effect + idempotency key, committed in one batch        |
  +----------------------------------------------------------------------+
```

| Piece | File | Job |
|---|---|---|
| Checkpointer | [checkpointer.py](src/hitl/checkpointer.py) | Stores LangGraph state. Each checkpoint is one atomic batch and one fsync |
| Policy | [policy.py](src/hitl/policy.py) | YAML rules. First match wins, and it fails closed |
| Ledger | [ledger.py](src/hitl/ledger.py) | Source of truth for decisions, deadlines and the hash-chained audit log |
| Engine | [engine.py](src/hitl/engine.py) | The graph, and the loop that advances each run |
| API | [server.py](src/hitl/server.py) | Start runs, list approvals, approve or reject, read the audit trail |
| Sink | [sink.py](src/hitl/sink.py) | Stand-in for a payment or card service, in its own process and database |

### Policy

Rules are checked from top to bottom. A malformed call, such as a refund with a missing or non-numeric amount, cannot match an `auto` rule and falls through to human review.

```yaml
default:
  route: approval
  escalation: [{role: ops, timeout_s: 300}]
rules:
  - name: small_refund
    when: {tool: refund, args.amount: {lte: 50}}
    route: auto
  - name: large_refund
    when: {tool: refund, args.amount: {gt: 50}}
    route: approval
    escalation:                               # each tier times out into the next
      - {role: support_lead, timeout_s: 300}
      - {role: finance_manager, timeout_s: 900}  # the last one rejects
```

Deadlines are stored as absolute times, so one that passes during an outage fires at the first scan after restart.

## Exactly-once, and what that really means

The engine cannot commit its own "done" record atomically with an effect in another system, so true exactly-once delivery is not possible here. What it provides is **at-least-once delivery to a receiver that deduplicates**, which gives one effect per action:

```python
ledger.mark_intent(action_id)            # durable: "about to execute"
result = sink.execute(key, action)       # effect and key commit in one batch at the sink
ledger.mark_done(action_id, result)      # durable: "finished"
```

| The process dies... | After restart | Effects |
|---|---|---|
| After `mark_intent` | The call is sent for the first time | 1 |
| **After the sink commits, before `mark_done`** | The call is repeated with the same key, and the sink returns its stored result | **1** |
| After `mark_done`, before the next checkpoint | The node re-runs, and the ledger returns before calling the sink | 1 |
| Inside the sink, after its commit | The engine's retry gets the stored result | 1 |

This relies on two things:

1. **The key must be the same on every attempt.** It is a hash of the run, call, tool and arguments. A random key was one of the negative controls, and it produced 26 duplicates.
2. **The receiver must honor idempotency keys.** A downstream system that cannot deduplicate could see a repeated call in the hard case. No engine-side design can fix that.

## A bug the crash tests found

**Runs marked complete with their actions never carried out.** All 51 unit tests at the time passed, then the first crash smoke test failed 11 of 16 rounds. If a node's writes were saved and the crash came before the next checkpoint, LangGraph's `snapshot.next` was empty, because it leaves out tasks that already have saved writes. The engine read that as "finished". It now checks `snapshot.tasks`, and a regression test covers the case.

## Limitations

- **The measured numbers use a scripted planner.** That keeps the experiments deterministic and repeatable, and every number above comes from it. [examples/llm_agent.py](examples/llm_agent.py) puts Claude behind the same `Planner` interface. Its tests run the real Anthropic SDK against a fake server, so they cover the request and response handling and how the engine treats a model that answers differently each time. They say nothing about the quality of the model's plans.
- **The LLM plans once per run.** It does not see tool results and re-plan, so this is plan-then-execute, not a full reasoning loop.
- **One engine process.** PersistKV is an embedded store, so there is no high availability or failover. A file lock stops a second process from opening the same database.
- **Roles are taken at face value.** The API does not authenticate anyone, because the focus of this project is durability. A real deployment would derive the role from a verified identity.
- **The audit log detects tampering, but does not prevent it.** Each run's entries are hash-chained, so editing one entry or removing one from the middle fails verification. Someone able to rewrite a run's whole chain, or cut its newest entries, would not be caught without anchoring the hash somewhere else.
- **Optional checkpointer features are not implemented:** `prune`, `copy_thread` and `delete_for_runs`. Checkpoints therefore grow until a thread is deleted.

## Run it

It needs Linux or WSL2 and Python 3.12. The file lock and SIGKILL are POSIX-only.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest -q                    # 80 tests, including conformance and real SIGKILL
bash experiments/run_all.sh            # every number above, about 40 minutes
```

Try it by hand, with each server in its own terminal:

```sh
export PYTHONPATH=src
HITL_SINK_DB=/tmp/hitl/sink HITL_SINK_PORT=8001 python -m hitl.sink

HITL_DB=/tmp/hitl/engine HITL_PORT=8000 HITL_SINK_URL=http://127.0.0.1:8001 \
  HITL_POLICY=policies/default.yaml python -m hitl.server
```

```sh
curl -X POST localhost:8000/runs -H 'content-type: application/json' -d '{"run_id": "demo",
  "actions": [{"tool": "refund", "args": {"account": "a1", "amount": 500}}]}'
curl localhost:8000/approvals          # the refund is waiting; note its action_id and role

# Now kill -9 the engine and start it again. The approval is still waiting.

curl -X POST localhost:8000/approvals/<action_id>/approve -H 'content-type: application/json' \
  -d '{"actor": "me", "role": "support_lead"}'
curl localhost:8000/runs/demo/audit    # the full, hash-chained history
```

Or give the agent a plain-English request, with Claude as the planner. It needs an API key and asks for approvals in the terminal:

```sh
export ANTHROPIC_API_KEY=...
python examples/llm_agent.py "Customer 42 was charged twice for order 7, \$180 each. Refund one."
```

Claude only proposes tool calls. The policy still decides, so the model cannot approve its own refund. If you `kill -9` the script while it waits for your answer, running `python examples/llm_agent.py` again resumes the same approval without asking the model again.
