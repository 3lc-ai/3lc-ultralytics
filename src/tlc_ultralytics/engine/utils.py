from __future__ import annotations

import contextlib
import random

from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import TLC_COLORSTR


def _complete_label_column_name(label_column_name: str, default_label_column_name: str) -> str:
    """Create a complete label column name from a potentially partial one.

    Examples:
        >>> _complete_label_column_name("a", "a")
        "a"
        >>> _complete_label_column_name("a", "a.b.c")
        "a.b.c"
        >>> _complete_label_column_name("a.b.c", "d.e.f")
        "a.b.c"
        >>> _complete_label_column_name("", "a.b.c")
        "a.b.c"
    """
    parts = label_column_name.split(".") if label_column_name else []
    default_parts = default_label_column_name.split(".")

    for i, default_part in enumerate(default_parts):
        if i >= len(parts):
            parts.append(default_part)

    return ".".join(parts)


@contextlib.contextmanager
def _restore_random_state():
    """Context manager to ensure the global random state is unchanged by the wrapped code."""
    state = random.getstate()
    yield
    random.setstate(state)


def _handle_deprecated_column_name(
    arg_value: str | None, settings_value: str | None, default_value: str, column_name: str
) -> str:
    """Handling for when a column name is passed as an argument directly, instead of through a `Settings` object.
    Used in the `trainer` and `validator` classes. A warning is logged if the column name is passed as an argument.

    If the column name is provided both through the `Settings` object and as an argument, the one passed directly is
    used and a warning is logged.

    If the column name is not provided, the default value is returned.

    :param arg_value: The column name passed as an argument directly.
    :param settings_value: The column name set in the `Settings` object.
    :param default_value: The default column name.
    :return: The column name.
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
