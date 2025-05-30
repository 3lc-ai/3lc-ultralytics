def _complete_label_column_name(label_column_name: str, default_label_column_name: str) -> str:
    parts = label_column_name.split(".")
    default_parts = default_label_column_name.split(".")

    for i, default_part in enumerate(default_parts):
        if i >= len(parts):
            parts.append(default_part)

    return ".".join(parts)
