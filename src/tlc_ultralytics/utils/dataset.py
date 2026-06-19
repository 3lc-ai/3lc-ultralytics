from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import tlc
import yaml
from tlc.helpers import AnnotationHelper
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, colorstr

from tlc_ultralytics.constants import TLC_COLORSTR, TLC_PREFIX

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from tlc.schemas._schema import ValueMapLike

    from tlc_ultralytics.settings import Settings


def get_dataset_functions(
    task: Literal["detect", "segment", "pose", "classify", "obb"],
) -> tuple[Callable, Callable]:
    if task == "detect":
        from tlc_ultralytics.detect.utils import check_det_table

        dataset_checker = check_det_dataset
        table_checker = check_det_table
    elif task == "segment":
        from tlc_ultralytics.segment.utils import check_seg_table

        dataset_checker = check_det_dataset
        table_checker = check_seg_table
    elif task == "classify":
        from ultralytics.data.utils import check_cls_dataset

        from tlc_ultralytics.classify.utils import check_cls_table

        dataset_checker = check_cls_dataset
        table_checker = check_cls_table
    elif task == "pose":
        from tlc_ultralytics.pose.utils import check_pose_table

        dataset_checker = check_det_dataset
        table_checker = check_pose_table
    elif task == "obb":
        from tlc_ultralytics.obb.utils import check_obb_table

        dataset_checker = check_det_dataset
        table_checker = check_obb_table
    else:
        raise ValueError(f"Invalid task: {task}")
    return dataset_checker, table_checker


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
    dataset_checker, table_checker = get_dataset_functions(task)

    if not tables and not isinstance(data, (str, Path)):
        msg = "`data` must be a string. If you are passing tables directly, use the `tables` argument instead."
        raise ValueError(msg)

    if not tables and isinstance(data, str) and data.endswith(".ndjson"):
        msg = (
            "Using NDJson datasets directly is not supported in the YOLO integration. Create a tlc.Table from the "
            f'data with `tlc.Table.from_yolo_ndjson(ndjson_file="{data!s}", ...)` or convert it to a YOLO dataset and '
            "use `tlc.Table.from_yolo_url(...)` or tlc_ultralytics.create_tables_from_yaml_file(...)."
        )
        raise ValueError(msg)

    # If the data starts with the 3LC prefix, parse the YAML file and populate `tables`
    has_prefix = False
    if tables is None and isinstance(data, str) and data.startswith(TLC_PREFIX):
        has_prefix = True
        LOGGER.info(f"{TLC_COLORSTR}Parsing 3LC YAML file data={data} and populating tables")
        tables = parse_3lc_yaml_file(data)

    if tables is None:
        resolved_project_name = settings.project_name if settings else project_name
        resolved_project_name = resolved_project_name or _get_default_names(data, "")[0]
        splits = splits or ("train", "val", "test", "minival")

        tables = {}
        if task == "classify":
            from tlc_ultralytics.classify.utils import get_or_create_cls_table

            data_dict = dataset_checker(data)

            for key in splits:
                if data_dict.get(key):
                    _, split_dataset_name = _get_default_names(data, key, project_name)
                    try:
                        table = get_or_create_cls_table(
                            key,
                            data_dict,
                            image_column_name=image_column_name,
                            label_column_name=label_column_name,
                            project_name=resolved_project_name,
                            dataset_name=split_dataset_name,
                            table_name="initial",
                            settings=settings,
                        )
                        tables[key] = table

                    except Exception as e:
                        LOGGER.warning(
                            f"{colorstr(key)}: Failed to read or create table for split {key} from {data}: {e!s}"
                        )

        elif task in ["detect", "segment", "pose", "obb"]:
            tables = create_tables_from_yaml_file(data, task=task, splits=splits, project_name=resolved_project_name)

        # Get the latest version when inferring
        for key, table in tables.items():
            tables[key] = table.latest()

            if tables[key] != table:
                LOGGER.info(
                    f"{colorstr(key)}: Using latest version of table from {data}: {table.url} -> {tables[key].url}"
                )
            else:
                LOGGER.info(f"{colorstr(key)}: Using initial version of table from {data}: {tables[key].url}")

    else:
        # LOGGER.info(f"{TLC_COLORSTR}Using data directly from tables")
        tables = tables.copy()
        _check_tables(tables)

        # First pass: convert ALL entries to tlc.Table objects (regardless of splits)
        for key, table in tables.items():
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

        # Second pass: validate and log only the tables matching splits
        for key in tables:
            if splits is not None and key not in splits:
                continue

            # Check that the table is compatible with the current task
            if table_checker is not None:
                table_checker(tables[key], image_column_name, label_column_name)

            source = "3LC YAML file" if has_prefix else "provided tables"
            LOGGER.info(f"{colorstr(key)}: Using table {tables[key].url} from {source}")

    first_split = next(iter(tables.keys()))

    # For detection, infer the actual bounding-box column when the configured/default root column
    # (e.g. `bbs`) is absent. The resolved path must propagate to value-map extraction below and to
    # dataset construction downstream, so write it back into both the local variable and `settings`
    # (which the trainer/validator read when building datasets).
    if task == "detect":
        from tlc_ultralytics.detect.utils import infer_detection_label_column_name

        label_column_name = infer_detection_label_column_name(tables[first_split], label_column_name)
        if settings is not None:
            settings.label_column_name = label_column_name

    value_map = get_value_map_from_table(tables[first_split], label_column_name, task)
    names = tlc.helpers.SchemaHelper.to_simple_value_map(value_map)
    if task == "pose":
        kpt_shape = tlc.helpers.KeypointHelper.get_keypoint_shape_from_table(tables[first_split], label_column_name)
        flip_idx = tlc.helpers.KeypointHelper.get_flip_indices_from_table(tables[first_split], label_column_name)
        keypoint_attributes = tlc.helpers.KeypointHelper.get_keypoint_attributes_from_table(
            tables[first_split], label_column_name
        )
        lines = tlc.helpers.KeypointHelper.get_lines_from_table(tables[first_split], label_column_name)
        line_attributes = tlc.helpers.KeypointHelper.get_line_attributes_from_table(
            tables[first_split], label_column_name
        )
        triangles = tlc.helpers.KeypointHelper.get_triangles_from_table(tables[first_split], label_column_name)
        triangle_attributes = tlc.helpers.KeypointHelper.get_triangle_attributes_from_table(
            tables[first_split], label_column_name
        )
        oks_sigmas = tlc.helpers.KeypointHelper.get_oks_sigmas_from_table(tables[first_split], label_column_name)
        points = tlc.helpers.KeypointHelper.get_points_from_table(tables[first_split], label_column_name)
    else:
        kpt_shape = [17, 3]  # yolo default
        flip_idx, keypoint_attributes, lines, line_attributes, triangles, triangle_attributes, oks_sigmas, points = (
            None,
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
        split_names = tlc.helpers.SchemaHelper.to_simple_value_map(split_value_map)

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
    return ret  # type: ignore[invalid-return-type]


def resolve_label_value_path(table: tlc.Table, label_column_name: str) -> str:
    """Resolve the value path to the label leaf for a label column in a table.

    Tries the provided path first. If it does not resolve to a value map (e.g. a legacy bounding
    box table where labels live at ``bbs.bb_list.label`` but the configured default is
    ``bbs.instances_additional_data.label``), falls back to the label path of the root column as
    resolved by ``AnnotationHelper``. Returns the provided path unchanged if neither resolves.

    :param table: The table to resolve the label path against.
    :param label_column_name: The configured (possibly default) label value path.
    :returns: A value path that resolves to a value map in the table, if one exists.
    """
    if table.get_value_map(label_column_name) is not None:
        return label_column_name

    column_name = label_column_name.split(".")[0]
    try:
        ann = AnnotationHelper.get(table, column_name)
    except (KeyError, ValueError):
        return label_column_name

    if ann.label_path is not None and table.get_value_map(ann.label_path) is not None:
        return ann.label_path
    return label_column_name


def get_value_map_from_table(
    table: tlc.Table,
    label_column_name: str,
    task: Literal["detect", "segment", "pose", "classify", "obb"],
) -> dict[int, str]:
    if task in ("pose", "obb"):
        column_name = label_column_name.split(".")[0]
        try:
            ann = AnnotationHelper.get(table, column_name)
            assert ann.label_path is not None
            return table.get_value_map(ann.label_path)  # type: ignore[return-value]
        except (AssertionError, KeyError, ValueError) as e:
            raise ValueError("Failed to get value map from table") from e
    return table.get_value_map(resolve_label_value_path(table, label_column_name))  # type: ignore[return-value]


def parse_3lc_yaml_file(data_file: str) -> dict[str, tlc.Table]:
    """Parse a 3LC YAML file and return the corresponding tables.

    :param data_file: The path to the 3LC YAML file.
    :returns: The tables pointed to by the YAML file.
    """
    # Read the YAML file, removing the prefix
    if not (data_file_url := tlc.Url(data_file.replace(TLC_PREFIX, ""))).exists():
        raise FileNotFoundError(f"Could not find YAML file {data_file_url}")

    data_config = yaml.safe_load(data_file_url.read_text())

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
                "When providing tables directly, they must be a tlc.Table or a URL to a tlc.Table. "
                f"Got {type(table)} for split {key}.",
            )
            raise ValueError(msg)


def _get_default_names(
    data_path: str | Path,
    split: str,
    project_name: str | None = None,
    dataset_name: str | None = None,
) -> tuple[str, str]:
    """Get default project and dataset names for a split.

    :param data_path: Path to the dataset YAML file or directory.
    :param split: The split name (e.g., 'train', 'val').
    :param project_name: Optional custom project name.
    :param dataset_name: Optional custom dataset name.
    :returns: Tuple of (project_name, dataset_name).
    """
    name = Path(data_path).stem
    project = project_name or f"{name}-YOLO"
    dataset = dataset_name or f"{name}-{split}"
    return project, dataset


def _resolve_pose_kwargs(data_dict: dict, kwargs: dict) -> None:
    """Resolve pose-specific kwargs from the YAML data dict and function arguments.

    Modifies kwargs in-place to include pose-specific parameters.
    """
    _pose_key_mapping = {
        "kpt_shape": "kpt_shape",
        "points": "points",
        "point_attributes": "point_attributes",
        "lines": "lines",
        "line_attributes": "line_attributes",
        "triangles": "triangles",
        "triangle_attributes": "triangle_attributes",
        "flip_indices": "flip_idx",
        "oks_sigmas": "oks_sigmas",
    }

    for kwargs_key, yaml_key in _pose_key_mapping.items():
        kwargs[kwargs_key] = kwargs.get(kwargs_key, None) or data_dict.get(yaml_key, None)

    _required_kwargs = ["kpt_shape", "flip_indices"]

    for required_kwarg in _required_kwargs:
        if kwargs[required_kwarg] is None:
            yaml_key = _pose_key_mapping[required_kwarg]
            text = f"the `{yaml_key}` field" if required_kwarg != yaml_key else ""
            msg = (
                f"`{required_kwarg}` is required for pose estimation, either through the `{required_kwarg}` "
                f"argument or {text} in the YAML file."
            )
            raise ValueError(msg)


def _create_split_table(
    split_paths: str | list[str],
    categories: dict[int, str],
    task: Literal["detect", "segment", "pose", "obb"],
    project_name: str,
    dataset_name: str,
    if_exists: Literal["raise", "reuse", "rename", "overwrite"],
    **kwargs,
) -> tlc.Table:
    """Create a table for a single split, supporting multiple paths."""
    return tlc.Table.from_yolo_url(
        split_paths,
        categories=categories,
        task=task,
        project_name=project_name,
        dataset_name=dataset_name,
        table_name="initial",
        if_exists=if_exists,
        **kwargs,
    )


def _get_existing_table(
    project_name: str | None,
    dataset_name: str | None,
    if_exists: Literal["raise", "reuse", "rename", "overwrite"],
) -> tlc.Table | None:
    """Check if a table already exists and return it if if_exists is 'reuse'."""
    final_table_url = tlc.helpers.ProjectLayout.table_url(
        table_name="initial", dataset_name=dataset_name, project_name=project_name
    )

    if not final_table_url.exists():
        return None

    if if_exists == "raise":
        msg = f"Table already exists at URL: {final_table_url}, and `if_exists` is set to `raise`."
        raise FileExistsError(msg)

    if if_exists == "reuse":
        return tlc.Table.from_url(final_table_url)

    return None


@overload
def create_tables_from_yaml_file(
    dataset: str,
    task: Literal["detect", "segment", "obb"],
    autodownload: bool = True,
    project_name: str | None = None,
    dataset_name: str | None = None,
    root_url: str | None = None,
    splits: Iterable[str] | None = ("train", "val", "test", "minival"),
    **kwargs,
) -> dict[str, tlc.Table]: ...


@overload
def create_tables_from_yaml_file(
    dataset: str,
    task: Literal["pose"],
    autodownload: bool = True,
    project_name: str | None = None,
    dataset_name: str | None = None,
    root_url: str | None = None,
    splits: Iterable[str] | None = ("train", "val", "test", "minival"),
    kpt_shape: tuple[int, int] | None = None,
    points: list[float] | None = None,
    point_attributes: ValueMapLike | None = None,
    lines: list[int] | None = None,
    line_attributes: ValueMapLike | None = None,
    triangles: list[int] | None = None,
    triangle_attributes: ValueMapLike | None = None,
    flip_indices: list[int] | None = None,
    oks_sigmas: list[float] | None = None,
    **kwargs,
) -> dict[str, tlc.Table]: ...


def create_tables_from_yaml_file(
    dataset: str,
    *,
    task: Literal["detect", "segment", "pose", "obb"],
    autodownload: bool = True,
    project_name: str | None = None,
    dataset_name: str | None = None,
    root_url: str | None = None,
    splits: Iterable[str] = ("train", "val", "test", "minival"),
    if_exists: Literal["raise", "reuse", "rename", "overwrite"] = "reuse",
    **kwargs,
) -> dict[str, tlc.Table]:
    """Create one tlc.Table for each split defined in a YOLO dataset YAML file.

    When a split is defined by a list of locations, one tlc.Table is created for each location and then joined to form
    a single tlc.Table for the split.

    :param dataset: The path to the dataset or dataset descriptor (like a YAML file).
    :param task: The task to create the tables for.
    :param autodownload: Whether to automatically download the dataset if not found, with a download script defined in
       the YAML file. Forwarded to `ultralytics.data.utils.check_det_dataset`.
    :param project_name: The name of the project to create the tables for. If not provided, the project name is set to
       the dataset path stem + "-YOLO".
    :param dataset_name: The name of the dataset to create the tables for. If not provided, the dataset name is set to
       the dataset path stem combined with the split.
    :param root_url: The root URL of the project to create the tables for. By default the 3LC project root URL is used.
    :param splits: The splits to create the tables for.
    :param if_exists: The if exists option to pass to the table creator.
    :param kwargs: Additional keyword arguments to pass to the table creator.
    :returns: A dictionary of tables, keyed by split.
    """
    data_dict = check_det_dataset(dataset, autodownload=autodownload)

    # Fast-track: reuse existing tables when if_exists="reuse"
    tables = {}
    for split in splits:
        split_project_name, split_dataset_name = _get_default_names(dataset, split, project_name, dataset_name)
        existing_table = _get_existing_table(split_project_name, split_dataset_name, if_exists)
        if existing_table is not None:
            LOGGER.info(f"{TLC_COLORSTR}Using existing table for split {split} from {existing_table.url}")
            tables[split] = existing_table

    # Set up project name
    default_project_name = _get_default_names(dataset, "")[0] if not project_name else project_name
    project_url = tlc.helpers.ProjectLayout.project_url(project_name=default_project_name, root_url=root_url)
    resolved_project_name = project_url.name

    if task == "pose":
        _resolve_pose_kwargs(data_dict, kwargs)

    categories = data_dict.get("names")
    assert isinstance(categories, dict), f"Expected 'names' to be a dictionary, but got {type(categories)}"

    for split in splits:
        if split in tables:
            continue
        split_paths = data_dict.get(split)
        if split_paths is None:
            continue
        _, split_dataset_name = _get_default_names(dataset, split, project_name, dataset_name)
        tables[split] = _create_split_table(
            split_paths, categories, task, resolved_project_name, split_dataset_name, if_exists, **kwargs
        )
        LOGGER.info(f"{TLC_COLORSTR}Created table for split {split} with URL: {tables[split].url}")

    return tables
