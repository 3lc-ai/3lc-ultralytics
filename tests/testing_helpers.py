from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics.utils.metrics import batch_probiou, bbox_iou, mask_iou

IOU_THRESHOLDS = {
    "masks": {
        "segment": 0.96,
    },
    "bboxes": {
        "detect": 0.98,
        "segment": 0.97,
        "pose": 0.999,
        "obb": 0.93,
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
