"""Filesystem locations shared by the test suite and the helper scripts it spawns.

Every path here is scoped to the current `pytest-xdist` worker. Workers are separate processes that share nothing but
the disk, and both 3LC and Ultralytics keep process-global state keyed on directories - most importantly the 3LC table
index, which scans a project root and is fed by a change-marker protocol that assumes a single writer per scope. When
all workers point at one project root they scan each other's tables and race on that marker, and a scan cycle can come
back empty, which makes `tlc.Table.latest()` fail for a table the worker itself just wrote. Giving every worker its own
root keeps one worker's writes out of another's index.

Subprocesses spawned by tests inherit `PYTEST_XDIST_WORKER`, so they resolve to the same root as the test that spawned
them.
"""

from __future__ import annotations

import os
from pathlib import Path

TMP_ROOT = Path(__file__).parent / "tmp"
"""Root of the suite's scratch tree. Created and removed by the master process only."""

WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "master")
"""The `pytest-xdist` worker id (`gw0`, `gw1`, ...), or `master` when running without `-n`."""

TMP = TMP_ROOT / WORKER_ID
"""Per-worker scratch directory. All test artifacts belong under here."""

PROJECT_ROOT = TMP / "3LC"
"""Per-worker 3LC project root, used as both the configured project root and the indexer's scan URL."""
