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


def _handle_deprecated_column_name(arg_value: str | None, settings_value: str | None, default_value: str) -> str:
    if arg_value is not None:
        msg = (
            f"Passing `{arg_value}` as an argument is deprecated. Provide `{arg_value}` to a `Settings` object instead."
        )
        LOGGER.warning(f"{TLC_COLORSTR}{msg}")

        if settings_value is not None:
            msg = (
                f"`{arg_value}` is both set in the `Settings` object and provided directly. Using the one from the "
                "`Settings` object."
            )
            LOGGER.warning(f"{TLC_COLORSTR}{msg}")

        else:
            return arg_value
    elif settings_value is None:
        return default_value
    return settings_value
