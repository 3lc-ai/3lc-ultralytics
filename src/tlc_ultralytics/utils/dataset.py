from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import tlc
import yaml
from tlc.core.builtins.constants import (
    INSTANCES_ADDITIONAL_DATA,
    LABEL,
)
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import LOGGER, colorstr

from tlc_ultralytics.constants import TLC_COLORSTR, TLC_PREFIX
from tlc_ultralytics.settings import Settings


def get_dataset_functions(
    task: Literal["detect", "segment", "pose", "classify", "obb"],
) -> tuple[Callable, Callable, Callable]:
    if task == "detect":
        from tlc_ultralytics.detect.utils import (
            check_det_table,
            get_or_create_det_table,
        )

        dataset_checker = check_det_dataset
        table_creator = get_or_create_det_table
        table_checker = check_det_table
    elif task == "segment":
        from tlc_ultralytics.segment.utils import check_seg_table, get_or_create_seg_table

        dataset_checker = check_det_dataset
        table_creator = get_or_create_seg_table
        table_checker = check_seg_table
    elif task == "classify":
        from tlc_ultralytics.classify.utils import check_cls_table, get_or_create_cls_table

        dataset_checker = check_cls_dataset
        table_creator = get_or_create_cls_table
        table_checker = check_cls_table
    elif task == "pose":
        from tlc_ultralytics.pose.utils import check_pose_table, get_or_create_pose_table

        dataset_checker = check_det_dataset
        table_creator = get_or_create_pose_table
        table_checker = check_pose_table
    elif task == "obb":
        from tlc_ultralytics.obb.utils import check_obb_table, get_or_create_obb_table

        dataset_checker = check_det_dataset
        table_creator = get_or_create_obb_table
        table_checker = check_obb_table
    else:
        raise ValueError(f"Invalid task: {task}")
    return dataset_checker, table_creator, table_checker


def check_tlc_dataset(  # noqa: C901
    data: str,
    tables: dict[str, tlc.Table | tlc.Url | str] | None,
    image_column_name: str,
    label_column_name: str,
    project_name: str | None = None,
    splits: Iterable[str] | None = None,
    task: Literal["detect", "segment", "pose", "classify"] | None = None,
    settings: Settings | None = None,
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
    dataset_checker, table_creator, table_checker = get_dataset_functions(task)

    if not tables and not isinstance(data, (str, Path)):
        msg = "`data` must be a string. If you are passing tables directly, use the `tables` argument instead."
        raise ValueError(msg)

    if not tables and isinstance(data, str) and data.endswith(".ndjson"):
        msg = (
            "Using NDJson datasets directly is not supported in the YOLO integration. Create a tlc.Table from the "
            f'data with `tlc.Table.from_yolo_ndjson(ndjson_file="{data!s}", ...)` or convert it to a YOLO dataset and '
            "use `tlc.Table.from_yolo(...)`."
        )
        raise ValueError(msg)

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
                        settings=settings,
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
        tables = tables.copy()
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

    value_map = get_value_map_from_table(tables[first_split], label_column_name, task)
    names = tlc.SchemaHelper.to_simple_value_map(value_map)
    if task == "pose":
        kpt_shape = tlc.KeypointHelper.get_keypoint_shape_from_table(tables[first_split], label_column_name)
        flip_idx = tlc.KeypointHelper.get_flip_indices_from_table(tables[first_split], label_column_name)
        keypoint_attributes = tlc.KeypointHelper.get_keypoint_attributes_from_table(
            tables[first_split], label_column_name
        )
        lines = tlc.KeypointHelper.get_lines_from_table(tables[first_split], label_column_name)
        line_attributes = tlc.KeypointHelper.get_line_attributes_from_table(tables[first_split], label_column_name)
        triangles = tlc.KeypointHelper.get_triangles_from_table(tables[first_split], label_column_name)
        triangle_attributes = tlc.KeypointHelper.get_triangle_attributes_from_table(
            tables[first_split], label_column_name
        )
        oks_sigmas = tlc.KeypointHelper.get_oks_sigmas_from_table(tables[first_split], label_column_name)
        points = tlc.KeypointHelper.get_points_from_table(tables[first_split], label_column_name)
    else:
        kpt_shape = [17, 3]  # yolo default
        flip_idx, keypoint_attributes, lines, line_attributes, triangles, triangle_attributes, points = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    if names is None:
        raise ValueError(f"Failed to get value map for table with Url: {tables[first_split].url}")

    for split, split_table in tables.items():
        if split == first_split:
            continue

        split_value_map = get_value_map_from_table(split_table, label_column_name, task)
        split_names = tlc.SchemaHelper.to_simple_value_map(split_value_map)

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

    ret = {
        **tables,
        "names": names_yolo,
        "names_3lc": value_map,
        "nc": len(names),
        "range_to_3lc_class": range_to_3lc_class,
        "3lc_class_to_range": {v: k for k, v in range_to_3lc_class.items()},
        "channels": 3,  # TODO(Frederik): Read out channels from appropriate place and populate here
        "kpt_shape": kpt_shape,
    }
    if task == "pose":
        if flip_idx is not None:
            ret["flip_idx"] = flip_idx
        if keypoint_attributes is not None:
            ret["keypoint_attributes"] = keypoint_attributes
        if lines is not None:
            ret["lines"] = lines
        if line_attributes is not None:
            ret["line_attributes"] = line_attributes
        if triangles is not None:
            ret["triangles"] = triangles
        if triangle_attributes is not None:
            ret["triangle_attributes"] = triangle_attributes
        if oks_sigmas is not None:
            ret["oks_sigmas"] = oks_sigmas
        if points is not None:
            ret["points"] = points
    return ret


def get_value_map_from_table(
    table: tlc.Table,
    label_column_name: str,
    task: Literal["detect", "segment", "pose", "classify", "obb"],
) -> dict[int, str]:
    if task == "pose":
        try:
            return table.rows_schema[label_column_name][INSTANCES_ADDITIONAL_DATA][LABEL].value.map
        except Exception as e:
            raise ValueError("Failed to get value map from table") from e
    elif task == "obb":
        try:
            return table.rows_schema[label_column_name][INSTANCES_ADDITIONAL_DATA][LABEL].value.map
        except Exception as e:
            raise ValueError("Failed to get value map from table") from e
    else:
        return table.get_value_map(label_column_name)


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
