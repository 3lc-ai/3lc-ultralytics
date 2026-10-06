import os
import shutil
import subprocess
import sys
from pathlib import Path

# Isolate the suite from the user's Ultralytics settings. Must be set before ultralytics is imported.
CACHE_ROOT = Path(__file__).parent / ".cache" / "ultralytics"
os.environ["YOLO_CONFIG_DIR"] = str(CACHE_ROOT / "config")
# Ultralytics falls back to a temp dir if this does not exist
(CACHE_ROOT / "config").mkdir(parents=True, exist_ok=True)

# Importing this module builds `ultralytics.utils.events.events`, whose constructor draws once from Python's global
# `random` state for its session id. It is otherwise imported lazily from inside the first training in a process,
# after ultralytics has seeded the RNG, so the first training draws a different augmentation stream than every
# training after it. Importing it here keeps all trainings in a session bit-comparable.
import ultralytics.utils.events  # noqa: E402, F401
from tmp_paths import PROJECT_ROOT, TMP, TMP_ROOT  # noqa: E402

DATASETS_DIR = CACHE_ROOT / "datasets"
WEIGHTS_DIR = CACHE_ROOT / "weights"

DETECT_DATASETS = ["coco8.yaml", "coco128.yaml", "dota8.yaml"]
SEGMENT_DATASETS = ["coco8-seg.yaml"]
POSE_DATASETS = ["coco8-pose.yaml"]
CLASSIFY_DATASETS = ["imagenet10"]
WEIGHTS = [
    "yolo11n.pt",
    "yolo12n.pt",
    "yolo26n.pt",
    "yolo26n-cls.pt",
    "yolo26n-seg.pt",
    "yolo26n-pose.pt",
    "yolo26n-obb.pt",
]


def _configure_ultralytics_settings():
    """Point Ultralytics at the suite's cache directories. Master process only."""
    from ultralytics.utils import SETTINGS

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS.update(datasets_dir=str(DATASETS_DIR), weights_dir=str(WEIGHTS_DIR))


def _ultralytics_test_runs_dir() -> Path:
    """Where Ultralytics writes training output under pytest (ignoring `runs_dir`). It never cleans it up."""
    from ultralytics.utils import ROOT

    return ROOT.parent / "tests" / "tmp" / "runs"


def download_assets():
    """Download every dataset and weights file the suite uses."""
    from ultralytics.data.utils import check_cls_dataset, check_det_dataset
    from ultralytics.utils.downloads import attempt_download_asset

    for name in DETECT_DATASETS + SEGMENT_DATASETS + POSE_DATASETS:
        check_det_dataset(name)
    for name in CLASSIFY_DATASETS:
        check_cls_dataset(name)
    for name in WEIGHTS:
        # Lookups by bare name resolve to weights_dir
        if not (WEIGHTS_DIR / name).exists():
            attempt_download_asset(str(WEIGHTS_DIR / name))


def _predownload_assets():
    """Download anything missing before the workers start, so they never download concurrently.

    Runs in a fresh process: this one read `datasets_dir` at import time, before the settings were updated.
    """
    datasets = DETECT_DATASETS + SEGMENT_DATASETS + POSE_DATASETS + CLASSIFY_DATASETS
    missing = [name for name in datasets if not (DATASETS_DIR / Path(name).stem).is_dir()]
    missing += [name for name in WEIGHTS if not (WEIGHTS_DIR / name).exists()]
    if missing:
        subprocess.run(
            [sys.executable, "-c", "from conftest import download_assets; download_assets()"],
            check=True,
            cwd=Path(__file__).parent,
        )


def _cap_worker_threads():
    """Split the CPU cores between xdist workers instead of each worker using all of them."""
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    threads = max(1, (os.cpu_count() or 1) // workers)

    import torch
    import ultralytics.utils.torch_utils as torch_utils

    torch.set_num_threads(threads)
    # `select_device` resets the thread count to this value
    torch_utils.NUM_THREADS = threads


def _configure_tlc():
    """Point 3LC at this process's scratch project root. Called after the scratch tree is (re)created."""
    import tlc
    from tlc._core.objects.tables.system_tables.indexing_tables.table_indexing_table import TableIndexingTable

    project_root_url = tlc.Url(PROJECT_ROOT)
    tlc.url.register_url_alias("<TEST_ALIAS>", "/test/alias")
    tlc.configuration.Configuration.instance().project_root_url = project_root_url
    TableIndexingTable.instance().add_scan_url(
        {
            "url": tlc.Url(project_root_url),
            "layout": "project",
            "object_type": "table",
            "static": False,
        }
    )


def pytest_sessionstart(session):
    """Create the TMP directory before running tests."""
    if getattr(session.config, "workerinput", None) is not None:
        # The master process wipes the shared root before any worker starts, so a worker only has to create the
        # per-worker subdirectory it is about to write into.
        TMP.mkdir(parents=True, exist_ok=True)
        _cap_worker_threads()
        _configure_tlc()
        return

    for path in (TMP_ROOT, _ultralytics_test_runs_dir()):
        if path.exists():
            shutil.rmtree(path)

    TMP.mkdir(parents=True, exist_ok=True)

    _configure_ultralytics_settings()
    _predownload_assets()

    # Create default folders once (no racing)
    import tlc  # noqa: F401

    _configure_tlc()


def pytest_sessionfinish(session, exitstatus):
    """Clean up the TMP directory after all tests are complete."""
    if getattr(session.config, "workerinput", None) is not None:
        # No need to delete the TMP directory, the master process does this at the end
        return

    for path in (TMP_ROOT, _ultralytics_test_runs_dir()):
        if path.exists():
            shutil.rmtree(path)
