import shutil

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
