import shutil
from pathlib import Path

TMP = Path(__file__).parent / "tmp"

def pytest_sessionstart(session):
    """Create the TMP directory before running tests."""
    if getattr(session.config, 'workerinput', None) is not None:
        # No need to create the TMP directory, the master process does this at the start
        return

    if TMP.exists():
        shutil.rmtree(TMP)

    TMP.mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    """Clean up the TMP directory after all tests are complete."""
    if getattr(session.config, 'workerinput', None) is not None:
        # No need to delete the TMP directory, the master process does this at the end
        return

    if TMP.exists():
        shutil.rmtree(TMP)
