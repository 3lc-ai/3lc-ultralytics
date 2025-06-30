import contextlib
import random


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
