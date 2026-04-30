import json
from pathlib import Path
from typing import Any

import numpy as np
import tlc
import torch
from tlc.constants._column_names import (
    CONFIDENCE,
    INSTANCES,
    INSTANCES_ADDITIONAL_DATA,
    LABEL,
    VERTEX_ROLE,
    VERTICES_2D,
    VERTICES_2D_ADDITIONAL_DATA,
)
from ultralytics.utils.metrics import batch_probiou, bbox_iou

IOU_THRESHOLDS = {
    "masks": {
        "segment": 0.96,
    },
    "bboxes": {
        "detect": 0.98,
        "segment": 0.97,
        "pose": 0.999,
        "obb": 0.999,
    },
}


def compare_dataset_values(  # noqa: C901
    sample_ultralytics: dict[str, Any],
    sample_3lc: dict[str, Any],
    task: str,
    mode: str,
) -> None:
    """Compare dataset labels or samples from 3LC and Ultralytics datasets."""
    keys = {
        "im_file",
        "cls",
        "bboxes",
        "keypoints",
        "segments",
        "normalized",
        "bbox_format",
        "shape",
        "ori_shape",
        "resized_shape",
        "img",
        "batch_idx",
        "ratio_pad",
        "example_id",
        "masks",
    }

    if set(sample_ultralytics.keys()) - keys:
        print(f"Unexpected keys in Ultralytics: {set(sample_ultralytics.keys()) - keys}")
    if set(sample_3lc.keys()) - keys:
        print(f"Unexpected keys in 3LC: {set(sample_3lc.keys()) - keys}")

    # Check im_file
    assert Path(sample_ultralytics["im_file"]) == Path(sample_3lc["im_file"]), (
        "Image path not equal in 3LC and Ultralytics"
    )

    # Check shape
    if "shape" in sample_ultralytics and "shape" in sample_3lc:
        assert sample_ultralytics["shape"] == sample_3lc["shape"]

    # Check ori_shape
    if "ori_shape" in sample_ultralytics and "ori_shape" in sample_3lc:
        assert sample_ultralytics["ori_shape"] == sample_3lc["ori_shape"]

    # Check ratio_pad
    if "ratio_pad" in sample_ultralytics and "ratio_pad" in sample_3lc:
        torch.testing.assert_close(np.array(sample_ultralytics["ratio_pad"]), np.array(sample_3lc["ratio_pad"]))

    # Check cls
    torch.testing.assert_close(sample_ultralytics["cls"], sample_3lc["cls"])

    # Check bboxes
    if "bboxes" in sample_ultralytics and "bboxes" in sample_3lc:
        if sample_ultralytics["bboxes"].shape[1] == 5:  # OBB
            ious = batch_probiou(sample_ultralytics["bboxes"], sample_3lc["bboxes"])
            self_ious = np.diag(ious)
            mean_iou = self_ious.mean().item()
            assert mean_iou > IOU_THRESHOLDS["bboxes"][task], "Bbox not equal in 3LC and Ultralytics"
        elif sample_ultralytics["bboxes"].shape[1] == 4:  # BBOX
            if isinstance(sample_ultralytics["bboxes"], np.ndarray):
                bboxes_ultralytics = torch.from_numpy(sample_ultralytics["bboxes"])
                assert isinstance(sample_3lc["bboxes"], np.ndarray)
                bboxes_3lc = torch.from_numpy(sample_3lc["bboxes"])
            else:
                bboxes_ultralytics = sample_ultralytics["bboxes"]
                bboxes_3lc = sample_3lc["bboxes"]
                assert isinstance(bboxes_3lc, torch.Tensor)

            ious = bbox_iou(bboxes_ultralytics, bboxes_3lc)
            assert ious.mean().item() > IOU_THRESHOLDS["bboxes"][task], "Bbox not equal in 3LC and Ultralytics"
        else:
            raise AssertionError("Bboxes not equal in 3LC and Ultralytics")

    # Check keypoints
    if "keypoints" in sample_ultralytics and "keypoints" in sample_3lc:
        torch.testing.assert_close(sample_ultralytics["keypoints"], sample_3lc["keypoints"])

    if "masks" in sample_ultralytics and "masks" in sample_3lc:
        value_3lc = sample_3lc["masks"].view(sample_3lc["masks"].shape[0], -1)
        value_ultralytics = sample_ultralytics["masks"].view(sample_ultralytics["masks"].shape[0], -1)
        intersection = (value_3lc & value_ultralytics).sum().float()
        union = (value_3lc | value_ultralytics).sum().float()
        iou = intersection / (union + 1e-7)
        assert iou.item() > IOU_THRESHOLDS["masks"][task], f"IOU is {iou.item()}"


def plot_ultralytics(sample, name: str) -> None:
    from ultralytics.utils import ops
    from ultralytics.utils.plotting import Annotator

    h, w = sample["resized_shape"]
    annotator = Annotator(np.ascontiguousarray(sample["img"].permute(1, 2, 0).cpu().numpy()))

    if "bboxes" in sample:
        if sample["bboxes"].shape[1] == 5:
            for obb in sample["bboxes"]:
                obb = ops.xywhr2xyxyxyxy(obb)
                obb = np.array(obb).reshape(-1, 4, 2).squeeze()
                obb[:, 0] *= w
                obb[:, 1] *= h
                obb = obb.tolist()
                annotator.box_label(obb)
        elif sample["bboxes"].shape[1] == 4:
            for bbox in sample["bboxes"]:
                bbox = ops.xywh2xyxy(bbox)
                bbox = np.array(bbox).reshape(-1, 4).squeeze()
                bbox[[0, 2]] *= w
                bbox[[1, 3]] *= h
                bbox = bbox.tolist()
                annotator.box_label(bbox)
        else:
            raise ValueError(f"Unsupported bboxes shape: {sample['bboxes'].shape}")

    if "masks" in sample:
        masks = sample["masks"].cpu()
        masks = ops.scale_masks(masks[None, :, :, :], (h, w))  # (N, C, H, W).
        annotator.masks(masks[0].numpy(), colors=[[255, 0, 0]])  # [n, h, w]

    if "keypoints" in sample:
        keypoints = sample["keypoints"].cpu().numpy()
        for keypoint in keypoints:
            keypoint[:, 0] *= w
            keypoint[:, 1] *= h
            annotator.kpts(keypoint, shape=(h, w))

    annotator.show(name)


def plot_matplotlib(sample, name: str) -> None:
    import matplotlib.pyplot as plt

    if "img" in sample:
        plt.figure()
        plt.title(name)
        plt.imshow(sample["img"].permute(1, 2, 0).cpu().numpy())
        plt.show()

    if "masks" in sample:
        plt.figure()
        plt.title(name)
        plt.imshow(sample["masks"].squeeze().cpu().numpy())
        plt.show()


def check_pose_table_and_metrics_tables(table, metrics_table, overrides: dict[str, Any]):
    # Fetch table metadata.
    table_points = tlc.KeypointHelper.get_points_from_table(table)
    table_lines = tlc.KeypointHelper.get_lines_from_table(table)
    table_point_attributes = tlc.KeypointHelper.get_keypoint_attributes_from_table(table)
    table_line_attributes = tlc.KeypointHelper.get_line_attributes_from_table(table)
    table_oks_sigmas = tlc.KeypointHelper.get_oks_sigmas_from_table(table)
    table_flip_indices = tlc.KeypointHelper.get_flip_indices_from_table(table)

    # Check table metadata correctness.
    assert table_points == overrides["points"]
    assert table_lines == overrides["lines"]
    assert table_point_attributes == [{"internal_name": name} for name in overrides["point_attributes"]]
    assert table_line_attributes == [{"internal_name": name} for name in overrides["line_attributes"]]
    assert table_oks_sigmas == [1 / 17] * 17  # Default Table OKS sigmas, regardless of Settings overrides.
    assert table_flip_indices == list(range(17))

    # Fetch metrics table metadata.
    pred_column = "keypoints_2d_predicted"
    metrics_table_points = tlc.KeypointHelper.get_points_from_table(metrics_table, pred_column)
    metrics_table_lines = tlc.KeypointHelper.get_lines_from_table(metrics_table, pred_column)
    metrics_table_point_attr = tlc.KeypointHelper.get_keypoint_attributes_from_table(metrics_table, pred_column)
    metrics_table_line_attr = tlc.KeypointHelper.get_line_attributes_from_table(metrics_table, pred_column)
    metrics_table_oks_sigmas = tlc.KeypointHelper.get_oks_sigmas_from_table(metrics_table, pred_column)
    metrics_table_flip_indices = tlc.KeypointHelper.get_flip_indices_from_table(metrics_table, pred_column)

    # Check metrics table correctness.
    assert metrics_table_points == table_points
    assert metrics_table_lines == table_lines
    assert metrics_table_point_attr == table_point_attributes
    assert metrics_table_line_attr == table_line_attributes
    assert metrics_table_oks_sigmas is None  # We don't copy over the sigmas
    assert metrics_table_flip_indices is None  # We don't copy over the flip indices

    assert metrics_table.columns == [
        "example_id",
        "keypoints_2d_predicted",
        "cls_loss",
        "box_loss",
        "dfl_loss",
        "pose_loss",
        "kobj_loss",
        "loss",
        "epoch",
        "Training Phase",
        "input_table_id",
    ]
    assert not metrics_table.rows_schema["keypoints_2d_predicted"].writable

    # Check the first metrics table row (we don't check the input table data, that is covered by core tests)
    metrics_table_row = metrics_table.table_rows[3][pred_column]  # only row with predictions
    assert INSTANCES_ADDITIONAL_DATA in metrics_table_row
    assert CONFIDENCE in metrics_table_row[INSTANCES_ADDITIONAL_DATA]
    assert LABEL in metrics_table_row[INSTANCES_ADDITIONAL_DATA]
    assert INSTANCES in metrics_table_row and len(metrics_table_row[INSTANCES]) == 1
    assert VERTICES_2D_ADDITIONAL_DATA in metrics_table_row[INSTANCES][0]
    assert CONFIDENCE in metrics_table_row[INSTANCES][0][VERTICES_2D_ADDITIONAL_DATA]
    assert VERTEX_ROLE in metrics_table_row[INSTANCES][0][VERTICES_2D_ADDITIONAL_DATA]
    assert VERTICES_2D in metrics_table_row[INSTANCES][0]

    # Cheeky string comparison to pin the row down - will this be too fragile?
    assert (
        json.dumps(metrics_table_row)
        == '{"x_min": 0.0, "y_min": 0.0, "x_max": 481.0, "y_max": 640.0, "instances": [{"lines": [3, 1, 4, 2, 1, 0, 0, 2, 5, 6, 5, 7, 6, 8, 7, 9, 8, 10, 11, 12, 11, 13, 12, 14, 13, 15, 14, 16, 5, 11, 6, 12], "lines_additional_data": {"line_role": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}, "vertices_2d": [261.3307800292969, 244.01507568359375, 279.0058898925781, 227.51434326171875, 247.5042724609375, 227.35733032226562, 311.8406677246094, 238.04412841796875, 238.6436004638672, 237.12435913085938, 353.57470703125, 337.904541015625, 221.15402221679688, 337.7941589355469, 420.21734619140625, 456.3583679199219, 193.9453125, 457.38018798828125, 440.1544189453125, 550.0919799804688, 206.7235107421875, 548.3096313476562, 347.01666259765625, 532.8427124023438, 263.6978759765625, 533.6087646484375, 396.7471008300781, 538.8079223632812, 259.0788269042969, 541.6107177734375, 427.5286865234375, 569.4536743164062, 281.0750427246094, 569.3405151367188], "vertices_2d_additional_data": {"vertex_role": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "confidence": [0.8946985006332397, 0.8680123090744019, 0.8405828475952148, 0.6776658892631531, 0.5823836922645569, 0.9826786518096924, 0.9777358770370483, 0.8919739127159119, 0.8344438672065735, 0.7838741540908813, 0.7172360420227051, 0.8988322615623474, 0.889037549495697, 0.4888758957386017, 0.46207594871520996, 0.24518102407455444, 0.23172228038311005]}, "bbs_2d": [{"x_min": 137.60887145996094, "y_min": 144.41246032714844, "x_max": 481.0, "y_max": 638.9334716796875}]}], "instances_additional_data": {"label": [0], "confidence": [0.21688038110733032]}}'  # noqa: E501
    )
