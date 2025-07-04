from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import tlc
from ultralytics.data.utils import IMG_FORMATS, check_cls_dataset
from ultralytics.utils import ROOT, YAML

from tlc_ultralytics.utils import check_tlc_dataset


def tlc_check_cls_dataset(
    data: str,
    tables: dict[str, tlc.Table | tlc.Url | Path | str] | None,
    image_column_name: str,
    label_column_name: str,
    project_name: str | None = None,
    splits: Iterable[str] | None = None,
) -> dict[str, tlc.Table | dict[float, str] | int]:
    """Get or create tables for YOLO classification datasets. data is ignored when tables is provided.

    :param data: Path to an ImageFolder dataset
    :param tables: Dictionary of tables, if already created
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :param project_name: Name of the project
    :param splits: List of splits to check for
    :return: Dictionary of tables and class names, with keys for each split and "names"
    """
    return check_tlc_dataset(
        data,
        tables,
        image_column_name,
        label_column_name,
        dataset_checker=check_cls_dataset,
        table_creator=get_or_create_cls_table,
        table_checker=check_cls_table,
        project_name=project_name,
        splits=splits,
    )


def get_or_create_cls_table(
    key: str,
    data_dict: dict[str, object],
    image_column_name: str,
    label_column_name: str,
    project_name: str,
    dataset_name: str,
    table_name: str,
) -> tlc.Table:
    """Get or create a classification table from a dataset dictionary.

    :param data_dict: Dictionary of dataset information
    :param project_name: Name of the project
    :param dataset_name: Name of the dataset
    :param table_name: Name of the table
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :return: A tlc.Table.from_image_folder() table
    """

    # Special handling for ImageNet
    is_imagenet = (
        isinstance(data_dict["names"], dict)
        and isinstance(data_dict["names"][0], str)
        and data_dict["names"][0].startswith("n0")
    )

    if is_imagenet:
        label_overrides = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]
        label_overrides = {k: tlc.MapElement(internal_name=k, display_name=v) for k, v in label_overrides.items()}
    else:
        label_overrides = None

    return tlc.Table.from_image_folder(
        root=data_dict[key],
        image_column_name=image_column_name,
        label_column_name=label_column_name,
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
        extensions=IMG_FORMATS,
        if_exists="reuse",
        add_weight_column=True,
        description="Created with 3LC YOLO integration",
        label_overrides=label_overrides,
    )


def check_cls_table(table: tlc.Table, image_column_name: str, label_column_name: str) -> None:
    """Check that a table is compatible with the current task."""
    row_schema = table.row_schema.values

    try:
        # Check for image and label columns in schema
        assert image_column_name in row_schema, (
            f"Image column '{image_column_name}' not found in schema for Table {table.url}. "
            "Try providing your 'image_column_name' as an argument if you have a different column name."
        )
        assert label_column_name in row_schema, (
            f"Label column '{label_column_name}' not found in schema for Table {table.url}. "
            "Try providing your 'label_column_name' as an argument if you have a different column name."
        )

        # Check for desired roles
        assert row_schema[image_column_name].value.string_role == tlc.STRING_ROLE_IMAGE_URL, (
            f"Image column '{image_column_name}' must have role tlc.STRING_ROLE_IMAGE_URL={tlc.STRING_ROLE_IMAGE_URL}."
        )
        assert row_schema[label_column_name].value.number_role == tlc.LABEL, (
            f"Label column '{label_column_name}' must have role tlc.LABEL={tlc.LABEL}."
        )
    except AssertionError as e:
        raise ValueError(f"Table {table.url} is not compatible with YOLO classification.") from e
