from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import tlc
from tlc.constants import INSTANCE_PROPERTIES, LABEL
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, SEMANTIC_SEGMENTATION_LABEL_COLUMN_NAME, TLC_COLORSTR
from tlc_ultralytics.utils.dataset import is_semantic_segmentation_column

if TYPE_CHECKING:
    from ultralytics.data.dataset import SemanticDataset

IGNORE_INDEX = 255
"""The label Ultralytics' semantic segmentation ignores in its loss and metrics, which 3LC's void class maps to."""


@dataclass(frozen=True)
class SemanticClasses:
    """The classes of a 3LC semantic segmentation column, and how they map to Ultralytics' contiguous class indices.

    Ultralytics trains on class indices `0..nc-1` and ignores pixels labeled `IGNORE_INDEX`. A 3LC column has arbitrary
    class ids, a background id recorded in its schema metadata rather than its value map, and a void class tagged in its
    value map. The training classes are the value map's classes minus void, plus the background, in ascending 3LC id
    order; void pixels map to `IGNORE_INDEX`.
    """

    names: dict[int, str]
    """Training class index to class name."""

    range_to_3lc_class: dict[int, int]
    """Training class index to 3LC class id."""

    prediction_value_map: dict[float, Any]
    """The column's value map without the void class, for the predicted segmentation column."""

    background: int | None
    """The 3LC id of the background class, if the column declares one."""

    void: int | None
    """The 3LC id of the void class, if the column declares one."""

    @property
    def class_to_range(self) -> dict[int, int]:
        """3LC class id to training class index."""
        return {v: k for k, v in self.range_to_3lc_class.items()}


def get_semantic_classes(table: tlc.Table, column_name: str) -> SemanticClasses:
    """Read the classes of a semantic segmentation column and map them to Ultralytics' contiguous class indices.

    :param table: The table holding the column.
    :param column_name: The semantic segmentation column.
    :returns: The column's classes.
    :raises ValueError: If the column has fewer than two classes to train on.
    """
    # The background id lives in the column schema's metadata and void is tagged in the value map; tlc reads both at
    # its (de)serialization boundary with these helpers, which it does not expose publicly.
    from tlc.data_types.semantic_segmentation import _background_id_from_metadata, _void_id

    value_map = table.get_value_map(f"{column_name}.{INSTANCE_PROPERTIES}.{LABEL}") or {}
    void = _void_id(value_map)
    background = _background_id_from_metadata(table.rows_schema.values[column_name])

    class_names = {
        int(class_id): name
        for class_id, name in tlc.helpers.SchemaHelper.to_simple_value_map(value_map).items()
        if int(class_id) != void
    }
    if background is not None:
        class_names.setdefault(background, "background")  # 3LC drops the background from the value map

    if len(class_names) < 2:
        msg = (
            f"Semantic segmentation column '{column_name}' in table {table.url} has {len(class_names)} class(es) to "
            "train on, but at least two are needed: every pixel is assigned one of them. If the column labels a "
            "single class on a background, declare the background (`background=` in "
            "`tlc.Table.from_semantic_segmentation` or `SemanticSegmentationRleSchema`)."
        )
        raise ValueError(msg)

    if len(class_names) > IGNORE_INDEX:
        msg = (
            f"Semantic segmentation column '{column_name}' in table {table.url} has {len(class_names)} classes to "
            f"train on, but Ultralytics supports at most {IGNORE_INDEX}: its masks are uint8, with {IGNORE_INDEX} as "
            "the ignore label."
        )
        raise ValueError(msg)

    class_ids = sorted(class_names)
    return SemanticClasses(
        names={i: class_names[class_id] for i, class_id in enumerate(class_ids)},
        range_to_3lc_class=dict(enumerate(class_ids)),
        prediction_value_map={k: v for k, v in value_map.items() if int(k) != void},
        background=background,
        void=void,
    )


def resolve_semantic_label_column(table: tlc.Table, label_column_name: str | None) -> str:
    """Resolve the semantic segmentation column of a table.

    The configured column (or the default `mask`) is used when the table has it. Otherwise the column is inferred as
    the table's only semantic segmentation column, like the other annotation tasks do (see
    `resolve_annotation_label_path`).

    TEMP(annotation-helper-semseg): `AnnotationHelper` has no `AnnotationType` for semantic segmentation yet, so unlike
    `resolve_annotation_label_path` this scans the table's columns directly instead of using `AnnotationHelper.find`.
    Fold this into `resolve_annotation_label_path`'s `_ANNOTATION_TASK_CONFIG` once it does.

    :param table: The table to resolve against.
    :param label_column_name: The configured label column, or None for the default. Only its root column is used.
    :returns: The name of the semantic segmentation column.
    :raises ValueError: If the column is not a semantic segmentation column, or none (or several) can be inferred.
    """
    column_name = (label_column_name or SEMANTIC_SEGMENTATION_LABEL_COLUMN_NAME).split(".")[0]
    columns = table.rows_schema.values

    if column_name in columns:
        if not is_semantic_segmentation_column(table, column_name):
            msg = (
                f"Column '{column_name}' of table {table.url} is not a semantic segmentation column. "
                f"{_describe_semantic_candidates(table)}"
            )
            raise ValueError(msg)
        return column_name

    candidates = [name for name in columns if is_semantic_segmentation_column(table, name)]
    if len(candidates) == 1:
        if label_column_name is not None:
            LOGGER.warning(
                f"{TLC_COLORSTR}Configured `label_column_name='{label_column_name}'` was not found in the table; "
                f"using the auto-detected semantic segmentation column '{candidates[0]}' instead."
            )
        return candidates[0]

    if len(candidates) > 1:
        msg = (
            f"Table {table.url} has several semantic segmentation columns ({', '.join(candidates)}). Set "
            "`label_column_name` to the one to use."
        )
    else:
        msg = f"Table {table.url} is not compatible with semantic segmentation. {_describe_semantic_candidates(table)}"
    raise ValueError(msg)


def _describe_semantic_candidates(table: tlc.Table) -> str:
    """Explain what the table has instead of an RLE-backed semantic segmentation column, for error messages."""
    from tlc.schemas.values import SegmentationMaskUrlStringValue

    columns = table.rows_schema.values
    png_masks = [name for name, schema in columns.items() if isinstance(schema.value, SegmentationMaskUrlStringValue)]
    if png_masks:
        return (
            f"Column '{png_masks[0]}' stores semantic segmentation as PNG files (the deprecated "
            "`SemanticSegmentationSchema`), which is not supported. Store the masks in a "
            "`SemanticSegmentationRleSchema` column, for example with `tlc.Table.from_semantic_segmentation`."
        )
    names = ", ".join(f"'{name}'" for name in columns)
    return (
        "Semantic segmentation needs a `SemanticSegmentationRleSchema` column, as written by "
        f"`tlc.Table.from_semantic_segmentation`. Columns present: {names}."
    )


def check_semantic_table(
    table: tlc.Table,
    image_column_name: str = IMAGE_COLUMN_NAME,
    label_column_name: str | None = None,
) -> None:
    """Verify that the table is compatible with semantic segmentation.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The semantic segmentation column. If None, the default `mask` column is used, or the
        table's only semantic segmentation column is inferred.
    :raises ValueError: If the table is not compatible with semantic segmentation.
    """
    column_name = resolve_semantic_label_column(table, label_column_name)
    if image_column_name not in table.rows_schema.values:
        msg = f"Table with url {table.url} is not compatible with semantic segmentation: image column "
        msg += f"'{image_column_name}' not found."
        raise ValueError(msg)
    get_semantic_classes(table, column_name)


class _LazyMasks(Sequence):
    """The masks of an Ultralytics semantic dataset, loaded one at a time as `tlc.Table.from_semantic_segmentation`
    writes them, so a split's masks are never all in memory at once."""

    def __init__(self, dataset: SemanticDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset.labels)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shape = self._dataset.labels[index]["shape"]
        return self._dataset.load_mask(index, image_shape=tuple(int(x) for x in shape))


def semantic_classes_from_yaml(data_dict: dict[str, Any]) -> tuple[dict[int, str], int | None, int]:
    """The 3LC classes, background id and void id of a semantic segmentation table built from an Ultralytics dataset.

    Masks of Ultralytics semantic datasets hold class indices, with `IGNORE_INDEX` for ignored pixels, which become
    3LC's void class. A polygon dataset (no `masks_dir`) paints unlabeled pixels with a background class that
    `add_polygon_background` adds to the classes, which becomes 3LC's background. A binary dataset (`nc: 1`) labels
    background `0` and foreground `1` while naming the foreground class only, so both get named here.

    :param data_dict: The dataset dict, as returned by `check_det_dataset` and updated by `add_polygon_background`.
    :returns: The classes, background id and void id.
    """
    names = {int(k): v for k, v in data_dict["names"].items()}
    background = data_dict.get("bg_class_idx")
    if int(data_dict.get("nc", len(names))) == 1:
        names = {0: "background", 1: names[0]}
        background = 0
    return {**names, IGNORE_INDEX: "ignore"}, background, IGNORE_INDEX


def create_semantic_split_table(
    split_paths: str | list[str],
    data_dict: dict[str, Any],
    project_name: str,
    dataset_name: str,
    if_exists: Literal["raise", "reuse", "rename", "overwrite"],
) -> tlc.Table:
    """Create a semantic segmentation table for one split of an Ultralytics semantic dataset.

    The split is read by the dataset Ultralytics itself would train on (`build_yolo_dataset`), so mask discovery,
    `label_mapping`, 1-bit masks and polygon rasterization behave exactly as in Ultralytics, and each mask is written
    at its original resolution in the Ultralytics class indices.

    :param split_paths: The image directories (or files) of the split.
    :param data_dict: The dataset dict, as returned by `check_det_dataset`.
    :param project_name: The name of the project.
    :param dataset_name: The name of the dataset.
    :param if_exists: What to do if the table already exists.
    :returns: The created table.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import add_polygon_background

    data_dict = add_polygon_background(dict(data_dict))  # as `SemanticSegmentationTrainer.get_dataset` does
    cfg = get_cfg(overrides={"task": "semantic"})
    dataset = build_yolo_dataset(cfg, split_paths, batch=1, data=data_dict, mode="val")

    classes, background, void = semantic_classes_from_yaml(data_dict)
    return tlc.Table.from_semantic_segmentation(
        images=list(dataset.im_files),
        masks=_LazyMasks(dataset),
        classes=classes,
        background=background,
        void=void,
        project_name=project_name,
        dataset_name=dataset_name,
        table_name="initial",
        if_exists="rename" if if_exists == "reuse" else if_exists,  # "reuse" of an existing table is handled upstream
    )


def decode_semantic_mask(
    height: int,
    width: int,
    fill: int,
    values: np.ndarray,
    rles: list[bytes],
) -> np.ndarray:
    """Decode a semantic segmentation row's RLE layers into a dense `(H, W)` uint8 label map.

    Mirrors `SemanticSegmentationSampleType.from_row`: the map starts out as the fill and each layer is painted over
    it in stored order, so a later layer wins where layers overlap. Layers are decoded one at a time, so only one
    dense layer exists at once, whatever the number of classes.

    :param height: The mask height.
    :param width: The mask width.
    :param fill: The value of pixels no layer covers.
    :param values: The value each layer is painted with.
    :param rles: The COCO RLE counts of each layer.
    :returns: The dense label map.
    """
    import pycocotools.mask as mask_utils

    mask = np.full((height, width), fill, dtype=np.uint8)
    for value, counts in zip(values, rles, strict=True):
        layer = mask_utils.decode({"size": [height, width], "counts": counts})
        mask[layer.astype(bool)] = value
    return mask
