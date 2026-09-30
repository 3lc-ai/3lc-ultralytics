import shutil

# Importing this module builds `ultralytics.utils.events.events`, whose constructor draws once from Python's global
# `random` state for its session id. It is otherwise imported lazily from inside the first training in a process,
# after ultralytics has seeded the RNG, so the first training draws a different augmentation stream than every
# training after it. Importing it here keeps all trainings in a session bit-comparable.
import ultralytics.utils.events  # noqa: F401
from tmp_paths import TMP, TMP_ROOT


def pytest_sessionstart(session):
    """Create the TMP directory before running tests."""
    if getattr(session.config, "workerinput", None) is not None:
        # The master process wipes the shared root before any worker starts, so a worker only has to create the
        # per-worker subdirectory it is about to write into.
        TMP.mkdir(parents=True, exist_ok=True)
        return

    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)

    TMP.mkdir(parents=True, exist_ok=True)

    # Create default folders once (no racing)
    import tlc  # noqa: F401


def pytest_sessionfinish(session, exitstatus):
    """Clean up the TMP directory after all tests are complete."""
    if getattr(session.config, "workerinput", None) is not None:
        # No need to delete the TMP directory, the master process does this at the end
        return

    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
