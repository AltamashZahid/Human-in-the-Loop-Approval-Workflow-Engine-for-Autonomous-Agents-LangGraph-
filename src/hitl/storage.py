"""Opening a PersistKV database safely from a server process."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any

import persistkv


def open_exclusive(path: str) -> tuple[persistkv.DB, Any]:
    """PersistKV has no file lock, and two processes on one database would corrupt it. The kernel
    drops this lock when the holder dies, including from SIGKILL, so it needs no cleanup. Keep the
    returned handle alive for as long as the database is open."""
    Path(path).mkdir(parents=True, exist_ok=True)
    handle = open(os.path.join(path, "owner.lock"), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{path} is already open in another process") from None
    return persistkv.DB(path), handle
