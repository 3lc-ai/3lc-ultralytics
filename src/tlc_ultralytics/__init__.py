# ruff: noqa: E402

import random

import sentry_sdk

sentry_sdk.profiler.transaction_profiler.random = random.Random()

from importlib.metadata import PackageNotFoundError, version

from tlc_ultralytics.engine.model import TLCYOLO, YOLO
from tlc_ultralytics.settings import Settings

try:
    __version__ = version("3lc-ultralytics")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "TLCYOLO",
    "YOLO",
    "Settings",
]
