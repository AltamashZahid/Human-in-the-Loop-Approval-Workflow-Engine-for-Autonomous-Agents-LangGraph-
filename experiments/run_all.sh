#!/bin/bash
# Reproduces every number in results/. One experiment at a time, so they do not skew each other.
#   bash dev.sh bash experiments/run_all.sh
set -euo pipefail
python -m pytest -q --no-header --tb=short -W ignore --basetemp="$HOME/hitl/data/pytest"
python experiments/crash_tests.py matrix --rounds "${MATRIX_ROUNDS:-100}"
python experiments/crash_tests.py random --rounds "${RANDOM_ROUNDS:-100}"
python experiments/crash_tests.py controls --rounds "${CONTROL_ROUNDS:-20}"
python experiments/measure.py policy --runs "${POLICY_RUNS:-2000}"
python experiments/measure.py latency --samples "${LATENCY_SAMPLES:-2000}" --trials 3 --concurrency 1
python experiments/measure.py latency --samples "${LATENCY_SAMPLES:-2000}" --trials 3 --concurrency 8
