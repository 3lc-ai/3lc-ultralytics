from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import tlc
import yaml
from ultralytics.utils import LOGGER, colorstr

from tlc_ultralytics.constants import TLC_COLORSTR, TLC_PREFIX


def check_tlc_dataset(  # noqa: C901
    data: str,
    tables: dict[str, tlc.Table | tlc.Url | str] | None,
    image_column_name: str,
    label_column_name: str,
    dataset_checker: Callable[[str], dict[str, object]] | None = None,
    table_creator: Callable[[str, dict[str, object], str, str, str, str, str], tlc.Table] | None = None,
    table_checker: Callable[[str, tlc.Table], bool] | None = None,
    project_name: str | None = None,
    splits: Iterable[str] | None = None,
) -> dict[str, tlc.Table | dict[float, str] | int]:
    """Get or create tables for YOLO datasets. data is ignored when tables is provided.

    :param data: Path to a dataset
    :param tables: Dictionary of tables, if already created
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :param dataset_checker: Function to check the dataset (yolo implementation, download and checks)
    :param table_creator: Function to create the tables for the YOLO dataset
    :param table_checker: Function to check that a table is compatible with the current task
    :param project_name: Name of the project
    :param splits: List of splits to parse.
    :return: Dictionary of tables and class names
    """

    if not tables and data is not None and data.endswith(".ndjson"):
        raise ValueError(
            "NDJson datasets are not yet supported in the YOLO integration. Create a tlc.Table from the ",
            "data or convert it to a YOLO dataset and use `tlc.Table.from_yolo`. The NDJson format will be ",
            "supported in the future.",
        )

    # If the data starts with the 3LC prefix, parse the YAML file and populate `tables`
    has_prefix = False
    if tables is None and isinstance(data, str) and data.startswith(TLC_PREFIX):
        has_prefix = True
        LOGGER.info(f"{TLC_COLORSTR}Parsing 3LC YAML file data={data} and populating tables")
        tables = parse_3lc_yaml_file(data)

    if tables is None:
        tables = {}

        data_dict = dataset_checker(data)

        # Get or create tables
        splits = splits or ("train", "val", "test", "minival")

        for key in splits:
            if data_dict.get(key):
                name = Path(data).stem
                dataset_name = f"{name}-{key}"
                table_name = "initial"

                if project_name is None:
                    project_name = f"{name}-YOLO"

                try:
                    table = table_creator(
                        key,
                        data_dict,
                        image_column_name=image_column_name,
                        label_column_name=label_column_name,
                        project_name=project_name,
                        dataset_name=dataset_name,
                        table_name=table_name,
                    )

                    # Get the latest version when inferring
                    tables[key] = table.latest()

                    if tables[key] != table:
                        LOGGER.info(
                            f"{colorstr(key)}: Using latest version of table from {data}: "
                            f"{table.url} -> {tables[key].url}"
                        )
                    else:
                        LOGGER.info(f"{colorstr(key)}: Using initial version of table from {data}: {tables[key].url}")

                except Exception as e:
                    LOGGER.warning(
                        f"{colorstr(key)}: Failed to read or create table for split {key} from {data}: {e!s}"
                    )

    else:
        # LOGGER.info(f"{TLC_COLORSTR}Using data directly from tables")
        _check_tables(tables)

        for key, table in tables.items():
            if splits is not None and key not in splits:
                continue

            if isinstance(table, (str, Path, tlc.Url)):
                try:
                    table_url = tlc.Url(table)
                    tables[key] = tlc.Table.from_url(table_url)
                except Exception as e:
                    raise ValueError(
                        f"Error loading table from {table} for split '{key}' provided through `tables`."
                    ) from e
            elif isinstance(table, tlc.Table):
                tables[key] = table
            else:
                msg = (
                    f"Invalid type {type(table)} for split {key} provided through `tables`."
                    "Must be a tlc.Table object or a location (string, pathlib.Path or tlc.Url) of a tlc.Table."
                )

                raise ValueError(msg)

            # Check that the table is compatible with the current task
            if table_checker is not None:
                table_checker(tables[key], image_column_name, label_column_name)

            source = "3LC YAML file" if has_prefix else "provided tables"
            LOGGER.info(f"{colorstr(key)}: Using table {tables[key].url} from {source}")

    first_split = next(iter(tables.keys()))

    names = tables[first_split].get_simple_value_map(label_column_name)
    value_map = tables[first_split].get_value_map(label_column_name)

    if names is None:
        raise ValueError(f"Failed to get value map for table with Url: {tables[first_split].url}")

    for split, split_table in tables.items():
        split_names = split_table.get_simple_value_map(label_column_name)

        if split_names is None:
            raise ValueError(f"Failed to get value map for table with Url: {tables[split].url}")

        if split_names != names:
            first_items = set(names.items())
            split_items = set(split_names.items())

            only_in_first = first_items - split_items
            only_in_split = split_items - first_items

            messages = []

            if only_in_first:
                dict_str = "{" + ", ".join(f"{k}: '{v}'" for k, v in only_in_first) + "}"
                messages.append(f"'{first_split}' has categories that '{split}' does not: {dict_str}")
            if only_in_split:
                dict_str = "{" + ", ".join(f"{k}: '{v}'" for k, v in only_in_split) + "}"
                messages.append(f"'{split}' has categories that '{first_split}' does not: {dict_str}")

            error_msg = "All splits must have the same categories, but " + " and ".join(messages)

            raise ValueError(error_msg)

    # Map name indices to 0, 1, ..., n-1
    names_yolo = dict(enumerate(names.values()))
    range_to_3lc_class = dict(enumerate(names))

    return {
        **tables,
        "names": names_yolo,
        "names_3lc": value_map,
        "nc": len(names),
        "range_to_3lc_class": range_to_3lc_class,
        "3lc_class_to_range": {v: k for k, v in range_to_3lc_class.items()},
        "channels": 3,  # TODO(Frederik): Read out channels from appropriate place and populate here
    }


def parse_3lc_yaml_file(data_file: str) -> dict[str, tlc.Table]:
    """Parse a 3LC YAML file and return the corresponding tables.

    :param data_file: The path to the 3LC YAML file.
    :returns: The tables pointed to by the YAML file.
    """
    # Read the YAML file, removing the prefix
    if not (data_file_url := tlc.Url(data_file.replace(TLC_PREFIX, ""))).exists():
        raise FileNotFoundError(f"Could not find YAML file {data_file_url}")

    data_config = yaml.safe_load(data_file_url.read())

    path = data_config.get("path")
    splits = [key for key in data_config if key != "path"]

    tables = {}
    for split in splits:
        # Handle :latest at the end
        if data_config[split].endswith(":latest"):
            latest = True
            split_path = data_config[split][: -len(":latest")]
        else:
            latest = False
            split_path = data_config[split]

        if split_path.startswith("./"):
            LOGGER.debug(f"{TLC_COLORSTR}{split} split path starts with './', removing it.")
            split_path = split_path[2:]

        table_url = tlc.Url(path) / split_path if path else tlc.Url(split_path)

        table = tlc.Table.from_url(table_url)

        if latest:
            table = table.latest()

        tables[split] = table

    return tables


def _check_tables(tables: object):
    if not isinstance(tables, dict):
        msg = f"When providing tables directly, they must be a dictionary, but got type {type(tables)}."
        raise ValueError(msg)

    for key, table in tables.items():
        if not isinstance(table, (str, Path, tlc.Url, tlc.Table)):
            msg = (
                "When providing tables directly, they must be a tlc.Table or a URL to a tlc.Table. ",
                f"Got {type(table)} for split {key}.",
            )
            raise ValueError(msg)
