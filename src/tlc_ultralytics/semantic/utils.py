from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import tlc
import torch
from tlc.constants import INSTANCE_PROPERTIES, LABEL
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, SEMANTIC_SEGMENTATION_LABEL_COLUMN_NAME, TLC_COLORSTR
from tlc_ultralytics.utils.dataset import is_semantic_segmentation_column

if TYPE_CHECKING:
    from ultralytics.data.dataset import SemanticDataset
    from ultralytics.utils.loss import SemanticSegmentationLoss

IGNORE_INDEX = 255
"""The label Ultralytics' semantic segmentation ignores in its loss and metrics, which 3LC's void class maps to."""

METADATA_NAMESPACE = "tlc_ultralytics"
"""The integration's namespace in a column schema's `metadata`."""

ULTRALYTICS_DATASET_METADATA_KEY = "ultralytics_dataset"
"""The key, in `METADATA_NAMESPACE`, of the stem of the Ultralytics dataset YAML a semantic segmentation column was
created from (e.g. `cityscapes8`)."""

CITYSCAPES_DATASETS = frozenset({"cityscapes", "cityscapes8"})
"""The Ultralytics datasets whose tables get Ultralytics' Cityscapes class weights in the cross-entropy loss, as
`SemanticSegmentationLoss` gives them when trained through `data=`."""


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
    value_map = table.get_value_map(f"{column_name}.{INSTANCE_PROPERTIES}.{LABEL}") or {}
    void = _void_id(value_map)
    background = _background_id(table.rows_schema.values[column_name])

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


# A column's void and background ids are read from its persisted on-disk format: the value map tag 3LC marks the void
# class with, and the `semantic_segmentation` namespace of the schema metadata 3LC records the background id in. tlc
# reads them with private helpers (`_void_id`, `_background_id_from_metadata`), whose names are a weaker contract than
# the format stored tables already carry. A public accessor has been requested upstream.


def _void_id(value_map: dict[float, Any]) -> int | None:
    """The id of the void class tagged in a semantic segmentation column's value map, or None."""
    from tlc.data_types.semantic_segmentation import TLC_SEMSEG_VOID

    for class_id, element in value_map.items():
        if getattr(element, "internal_name", None) == TLC_SEMSEG_VOID:
            return int(class_id)
    return None


def _background_id(schema: tlc.Schema) -> int | None:
    """The id of the background class recorded in a semantic segmentation column schema's metadata, or None."""
    metadata = getattr(schema, "metadata", None) or {}
    namespace = metadata.get("semantic_segmentation") or {}
    background = namespace.get("background")
    return None if background is None else int(background)


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
    instance_columns = _instance_segmentation_columns(table)
    if instance_columns:
        return (
            f"Column '{instance_columns[0]}' holds instance segmentation: one polygon or mask per object instance, "
            "each with a class. Semantic segmentation needs one class for every pixel of the image, so the integration "
            "cannot train on it directly. To use this data, either train an instance segmentation model with "
            "`task='segment'`, or create semantic segmentation tables from the YOLO dataset YAML with "
            "`create_tables_from_yaml_file(..., task='semantic')` (or by passing `data=` with `task='semantic'`), "
            "which rasterizes the polygons onto a `background` class as Ultralytics' `SemanticSegmentationTrainer` "
            "does, or write a table from rasterized masks with `tlc.Table.from_semantic_segmentation`."
        )
    names = ", ".join(f"'{name}'" for name in columns)
    return (
        "Semantic segmentation needs a `SemanticSegmentationRleSchema` column, as written by "
        f"`tlc.Table.from_semantic_segmentation`. Columns present: {names}."
    )


def _instance_segmentation_columns(table: tlc.Table) -> list[str]:
    """The table's instance segmentation columns.

    TEMP(annotation-helper-semseg): `AnnotationHelper` classifies semantic segmentation columns as
    `AnnotationType.SEGMENTATION` too, so they are excluded by sample type. Each column is classified on its own with
    `AnnotationHelper.get`, since `AnnotationHelper.find` raises when a table has several segmentation columns.
    """
    from tlc.helpers import AnnotationHelper, AnnotationType

    columns = []
    for name in table.rows_schema.values:
        try:
            annotation = AnnotationHelper.get(table, name)
        except (KeyError, ValueError):  # not an annotation column
            continue
        if annotation.type is AnnotationType.SEGMENTATION and not is_semantic_segmentation_column(table, name):
            columns.append(name)
    return columns


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
    """The masks of an Ultralytics semantic dataset, loaded one at a time as `create_semantic_split_table` writes them,
    so a split's masks are never all in memory at once."""

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
    ultralytics_dataset: str | None = None,
) -> tlc.Table:
    """Create a semantic segmentation table for one split of an Ultralytics semantic dataset.

    The split is read by the dataset Ultralytics itself would train on (`build_yolo_dataset`), so mask discovery,
    `label_mapping`, 1-bit masks and polygon rasterization behave exactly as in Ultralytics, and each mask is written
    at its original resolution in the Ultralytics class indices.

    The table is written like `tlc.Table.from_semantic_segmentation` writes it, with the stem of the dataset YAML
    recorded in the mask column's schema metadata (see `ultralytics_dataset_from_table`), which that function has no
    way to take. Mask ids are not validated one by one as it does: the masks come from Ultralytics' own dataset, whose
    classes are the table's.

    :param split_paths: The image directories (or files) of the split.
    :param data_dict: The dataset dict, as returned by `check_det_dataset`.
    :param project_name: The name of the project.
    :param dataset_name: The name of the dataset.
    :param if_exists: What to do if the table already exists.
    :param ultralytics_dataset: The dataset YAML the split is read from, whose stem is recorded in the table. Nothing is
        recorded when None.
    :returns: The created table.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import add_polygon_background

    data_dict = add_polygon_background(dict(data_dict))  # as `SemanticSegmentationTrainer.get_dataset` does
    cfg = get_cfg(overrides={"task": "semantic"})
    dataset = build_yolo_dataset(cfg, split_paths, batch=1, data=data_dict, mode="val")

    classes, background, void = semantic_classes_from_yaml(data_dict)
    mask_schema = tlc.schemas.SemanticSegmentationRleSchema(classes=classes, background=background, void=void)
    if ultralytics_dataset is not None:
        mask_schema.metadata = {
            **(mask_schema.metadata or {}),
            METADATA_NAMESPACE: {ULTRALYTICS_DATASET_METADATA_KEY: Path(ultralytics_dataset).stem},
        }
    row_schema = tlc.Schema.from_schema_like({IMAGE_COLUMN_NAME: tlc.schemas.ImageSchema(), "mask": mask_schema})
    row_schema.add_sample_weight()

    writer = tlc.TableWriter(
        schema=row_schema,
        project_name=project_name,
        dataset_name=dataset_name,
        table_name="initial",
        if_exists="rename" if if_exists == "reuse" else if_exists,  # "reuse" of an existing table is handled upstream
    )
    for image, mask in zip(dataset.im_files, _LazyMasks(dataset), strict=True):
        writer.add_row({IMAGE_COLUMN_NAME: image, "mask": mask})
    return writer.finalize()


def ultralytics_dataset_from_table(table: tlc.Table, column_name: str) -> str | None:
    """The stem of the Ultralytics dataset YAML a semantic segmentation column was created from, if it records one.

    Tables created with `create_tables_from_yaml_file` record it (see `create_semantic_split_table`); tables written
    with `tlc.Table.from_semantic_segmentation` or created before it was recorded do not.

    :param table: The table holding the column.
    :param column_name: The semantic segmentation column.
    :returns: The dataset YAML's stem, like `cityscapes8`, or None.
    """
    metadata = getattr(table.rows_schema.values[column_name], "metadata", None) or {}
    namespace = metadata.get(METADATA_NAMESPACE) or {}
    dataset = namespace.get(ULTRALYTICS_DATASET_METADATA_KEY)
    return None if dataset is None else str(dataset)


def _is_cityscapes(ultralytics_dataset: str | None) -> bool:
    """Whether the stem of an Ultralytics dataset YAML names one Ultralytics has Cityscapes class weights for."""
    return ultralytics_dataset is not None and ultralytics_dataset.lower() in CITYSCAPES_DATASETS


def check_cityscapes_class_weights(cityscapes_class_weights: bool | None, nc: int) -> None:
    """Verify that the Cityscapes class weights can be applied when `Settings.cityscapes_class_weights` forces them.

    :param cityscapes_class_weights: The value of `Settings.cityscapes_class_weights`.
    :param nc: The number of training classes.
    :raises ValueError: If the weights are forced for a number of classes other than Cityscapes' 19.
    """
    from ultralytics.utils.metrics import CITYSCAPES_WEIGHT

    if cityscapes_class_weights is True and nc != len(CITYSCAPES_WEIGHT):
        msg = (
            f"`Settings.cityscapes_class_weights=True` forces Ultralytics' Cityscapes class weights, which are for "
            f"Cityscapes' {len(CITYSCAPES_WEIGHT)} classes, but the tables have {nc} classes to train on. Leave it "
            "unset (None) to apply them only to tables created from a Cityscapes YAML, or set it to False."
        )
        raise ValueError(msg)


def uses_dataset_class_weights(
    ultralytics_dataset: str | None, nc: int, cityscapes_class_weights: bool | None = None
) -> bool:
    """Whether a semantic segmentation table gets Ultralytics' Cityscapes class weights in the loss.

    Only Cityscapes has class weights in Ultralytics, for its 19 classes. By default they apply to tables created from a
    Cityscapes YAML with that many classes, as Ultralytics applies them when training through `data=`.
    `cityscapes_class_weights` (see `Settings.cityscapes_class_weights`) forces them on or off instead. See
    `apply_dataset_class_weights`.

    :param ultralytics_dataset: The stem of the Ultralytics dataset YAML the table was created from, or None.
    :param nc: The number of training classes.
    :param cityscapes_class_weights: The value of `Settings.cityscapes_class_weights`.
    :returns: Whether the loss is weighted.
    :raises ValueError: If the weights are forced for a number of classes other than Cityscapes' 19.
    """
    from ultralytics.utils.metrics import CITYSCAPES_WEIGHT

    if cityscapes_class_weights is not None:
        check_cityscapes_class_weights(cityscapes_class_weights, nc)
        return cityscapes_class_weights
    return _is_cityscapes(ultralytics_dataset) and nc == len(CITYSCAPES_WEIGHT)


def dataset_class_weights_message(
    ultralytics_dataset: str | None, nc: int, cityscapes_class_weights: bool | None, loss_name: str
) -> str | None:
    """Explain whether, and why, a loss is weighted with Ultralytics' Cityscapes class weights, for logging.

    :param ultralytics_dataset: The stem of the Ultralytics dataset YAML the table was created from, or None.
    :param nc: The number of training classes.
    :param cityscapes_class_weights: The value of `Settings.cityscapes_class_weights`.
    :param loss_name: What the loss is, like "the cross-entropy loss".
    :returns: The message, or None when there is nothing to say: no weights, and none that the setting turned off.
    """
    if uses_dataset_class_weights(ultralytics_dataset, nc, cityscapes_class_weights):
        reason = (
            "forced by `Settings.cityscapes_class_weights=True`"
            if cityscapes_class_weights
            else f"from the table's recorded dataset '{ultralytics_dataset}'"
        )
        return f"Weighting {loss_name} with Ultralytics' Cityscapes class weights, {reason}"
    if cityscapes_class_weights is False and uses_dataset_class_weights(ultralytics_dataset, nc):
        return (
            f"Not weighting {loss_name} with Ultralytics' Cityscapes class weights, which the table's recorded dataset "
            f"'{ultralytics_dataset}' would get, since `Settings.cityscapes_class_weights=False`"
        )
    return None


def apply_dataset_class_weights(
    loss: SemanticSegmentationLoss, ultralytics_dataset: str | None, cityscapes_class_weights: bool | None = None
) -> bool:
    """Weight a semantic segmentation loss's cross-entropy with the class weights of the dataset its table came from.

    `SemanticSegmentationLoss` applies Ultralytics' Cityscapes class weights only when `model.args.data` names a
    Cityscapes YAML, which holds when training through `data=`, but neither when training from tables nor on a model
    loaded from a checkpoint, whose `args` is a dict. Deciding from the table's recorded dataset instead weights
    Cityscapes tables the same in training and in the per-sample `ce_loss` of metrics collection.
    `cityscapes_class_weights` overrides the decision (see `uses_dataset_class_weights`), and turning the weights off
    also removes them from a loss Ultralytics weighted itself. The weights are registered exactly as Ultralytics does,
    and a loss Ultralytics weighted with other weights is left as it is.

    :param loss: The loss to weight, modified in place.
    :param ultralytics_dataset: The stem of the Ultralytics dataset YAML the table was created from, or None.
    :param cityscapes_class_weights: The value of `Settings.cityscapes_class_weights`.
    :returns: Whether the loss is weighted with the dataset's class weights, by this call or already by Ultralytics.
    :raises ValueError: If the weights are forced for a number of classes other than Cityscapes' 19.
    """
    from ultralytics.utils.metrics import CITYSCAPES_WEIGHT

    if cityscapes_class_weights is False:
        if getattr(loss, "use_cityscapes_weight", False):
            loss.use_cityscapes_weight = False
            loss.ce.register_buffer("weight", None, persistent=False)
        return False
    if getattr(loss, "use_cityscapes_weight", False):
        return True
    if getattr(loss.ce, "weight", None) is not None:  # weighted otherwise, like newer Ultralytics' `cls_pw` weights
        return False
    if not uses_dataset_class_weights(ultralytics_dataset, loss.nc, cityscapes_class_weights):
        return False

    loss.use_cityscapes_weight = True
    # Non-persistent, as in Ultralytics: the weight is a constant, not to be serialized into checkpoints.
    weight = torch.from_numpy(CITYSCAPES_WEIGHT).to(device=loss.device, dtype=loss.dtype)
    loss.ce.register_buffer("weight", weight, persistent=False)
    return True


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
