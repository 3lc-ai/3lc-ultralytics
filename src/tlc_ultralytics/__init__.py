# ruff: noqa: I001, E402

import random

import sentry_sdk

sentry_sdk.profiler.transaction_profiler.random = random.Random()

from tlc_ultralytics.engine.model import TLCYOLO, YOLO
from tlc_ultralytics.overrides import check_pip_update_available
from tlc_ultralytics.settings import Settings

# Patch the check_pip_update_available function to avoid prompting for an update
import ultralytics
ultralytics.utils.checks.check_pip_update_available = check_pip_update_available

__all__ = [
    "TLCYOLO",
    "YOLO",
    "Settings",
]
