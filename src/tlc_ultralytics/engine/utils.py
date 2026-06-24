from __future__ import annotations

import contextlib
import random

from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import TLC_COLORSTR


@contextlib.contextmanager
def _restore_random_state():
    """Context manager to ensure the global random state is unchanged by the wrapped code."""
    state = random.getstate()
    yield
    random.setstate(state)


def _handle_deprecated_column_name(
    arg_value: str | None, settings_value: str | None, default_value: str | None, column_name: str
) -> str | None:
    """Handling for when a column name is passed as an argument directly, instead of through a `Settings` object.
    Used in the `trainer` and `validator` classes. A warning is logged if the column name is passed as an argument.

    If the column name is provided both through the `Settings` object and as an argument, the one passed directly is
    used and a warning is logged.

    If the column name is not provided, the default value is returned (which may be None to defer
    resolution — e.g. the label column is resolved against the table later in `check_tlc_dataset`).

    :param arg_value: The column name passed as an argument directly.
    :param settings_value: The column name set in the `Settings` object.
    :param default_value: The default column name, or None to defer resolution.
    :return: The column name, or None if unset and the default is None.
    """
    if arg_value is not None:
        msg = (
            f"Passing `{column_name}` as an argument is deprecated. Provide `{column_name}` to a `Settings` object "
            "instead."
        )
        LOGGER.warning(f"{TLC_COLORSTR}{msg}")

        if settings_value is not None:
            msg = (
                f"`{column_name}` is both set in the `Settings` object and provided directly. Using the one passed "
                f"directly: '{arg_value}'."
            )
            LOGGER.warning(f"{TLC_COLORSTR}{msg}")

        return arg_value
    elif settings_value is None:
        return default_value
    return settings_value
