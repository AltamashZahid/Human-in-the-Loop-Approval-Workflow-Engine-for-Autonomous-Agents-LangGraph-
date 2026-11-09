"""Crash injection for the experiments.

`crashpoint("name")` kills the process with SIGKILL when HITL_CRASH_AT is "name:n" and this is
the nth time the point has been reached. Without that variable it does nothing.

SIGKILL matters here: no `finally` block, atexit hook or buffered flush gets to run.
"""

from __future__ import annotations

import os
import signal
import threading

_target, _, _nth = os.environ.get("HITL_CRASH_AT", "").partition(":")
_remaining = int(_nth or 1)
_lock = threading.Lock()


def crashpoint(name: str) -> None:
    global _remaining
    if name != _target:
        return
    with _lock:
        _remaining -= 1
        if _remaining == 0:
            os.kill(os.getpid(), signal.SIGKILL)
