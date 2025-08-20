from pathlib import Path
from typing import Any

import numpy as np
import tlc
import ultralytics
from PIL import Image
from tqdm import tqdm

from tlc_ultralytics import YOLO

COCO_SKELETON = [3, 1, 4, 2, 1, 0, 0, 2, 5, 6, 5, 7, 6, 8, 7, 9, 8, 10, 11, 12, 11, 13, 12, 14, 13, 15, 14, 16]

YOLO_HAND_LANDMARKS = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MMC",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}

YOLO_HAND_SKELETON = [
    (0, 1),
    (0, 5),
    (0, 17),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 9),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 13),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 17),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
]


def flatten(inp: list[tuple[int, int]]) -> list[int]:
    out = []
    for tup in inp:
        out.extend(tup)
    return out


def interleave(a: list[float], b: list[float]) -> list[float]:
    out = []
    for aa, bb in zip(a, b):
        out.append(aa)
        out.append(bb)
    return out


def format_yolo_kpts_instances(result: ultralytics.engine.results.Results) -> list[dict[str, Any]]:
    """Will create the following format:

    [{
        "xys": [],
        "visibilities": [],
        "lines": lines
    }] # for each predicted keypoint instance
    """
    instance_list = []

    kpts = result.keypoints.data.cpu().numpy()
    if not result.keypoints.has_visible:
        # Not exactly the right check for this, but seems to work as a proxy for checking the data.
        return []

    # kpts is shape num_instances, num_keypoints, 3
    num_instances = kpts.shape[0]
    for i in range(num_instances):
        xys = kpts[i, :, :2].reshape(-1)  # interleaved flattened [x, y, x, y, ...]
        viz = kpts[i, :, -1]  # 1x17
        conf = result.keypoints.conf[i].cpu().numpy()  # 1x17
        assert np.array_equal(viz, conf)

        instance = {
            "xys": xys.tolist(),
            "lines": COCO_SKELETON,
            "xys_additional_data": {
                "conf": conf.tolist(),
            },
        }
        instance_list.append(instance)

    return instance_list


def format_yolo_kpts(res: ultralytics.engine.results.Results) -> dict[str, Any]:
    """Convert pose predictions from a single image to 3LC Table format."""
    h, w = res.orig_shape

    metrics = {
        "instances": format_yolo_kpts_instances(res),
        "x_min": 0.0,
        "x_max": float(w),
        "y_min": 0.0,
        "y_max": float(h),
    }
    return metrics


def points_and_lines_2d_schema() -> tlc.Schema:
    values_dict = {
        "xys": tlc.Schema(value=tlc.Float32Value(), size0=tlc.DimensionNumericValue()),
        "lines": tlc.Schema(value=tlc.Int32Value(), size0=tlc.DimensionNumericValue()),
    }

    values_dict["xys_additional_data"] = tlc.Schema(
        values={
            "conf": tlc.Schema(value=tlc.Float32Value(), size0=tlc.DimensionNumericValue()),
        }
    )

    schema = tlc.Schema(
        values={
            "instances": tlc.Schema(
                values=values_dict,
                size0=tlc.DimensionNumericValue(),
            ),
            "x_min": tlc.Schema(value=tlc.Float32Value()),
            "x_max": tlc.Schema(value=tlc.Float32Value()),
            "y_min": tlc.Schema(value=tlc.Float32Value()),
            "y_max": tlc.Schema(value=tlc.Float32Value()),
        },
    )
    return schema


def collect_pose_metrics(model: YOLO, table: tlc.Table) -> None:
    run = tlc.init("GEOMETRY", "Pose collect")

    mw = tlc.MetricsTableWriter(
        run.url,
        table.url,
        column_schemas={"pose_predicted": points_and_lines_2d_schema()},
    )

    for i, row in enumerate(table):
        res = model(row["image"], verbose=False)  # list of length batch_size
        assert len(res) == 1
        metrics_row = {"pose_predicted": format_yolo_kpts(res[0]), "example_id": i}
        mw.add_row(metrics_row)

    mw.finalize()
    metrics_infos = mw.get_written_metrics_infos()
    run.update_metrics(metrics_infos)
    run.update_attribute(tlc.RUN_STATUS, tlc.RUN_STATUS_COMPLETED)


def points_and_lines_2d_schema_2(is_list: bool = False, include_visibilities: bool = False) -> tlc.Schema:
    values_dict = {
        "xys": tlc.Schema(value=tlc.Float32Value(), size0=tlc.DimensionNumericValue()),
        "lines": tlc.Schema(value=tlc.Int32Value(), size0=tlc.DimensionNumericValue()),
    }

    if include_visibilities:
        values_dict["visibilities"] = tlc.Schema(value=tlc.Int32Value(), size0=tlc.DimensionNumericValue())

    schema = tlc.Schema(
        values={
            "instances": tlc.Schema(
                values=values_dict,
                size0=tlc.DimensionNumericValue() if is_list else None,
            ),
            "x_min": tlc.Schema(value=tlc.Float32Value()),
            "x_max": tlc.Schema(value=tlc.Float32Value()),
            "y_min": tlc.Schema(value=tlc.Float32Value()),
            "y_max": tlc.Schema(value=tlc.Float32Value()),
        },
    )
    return schema


def create_hand_dataset(yaml_file: str, split: str) -> tlc.Table:
    root = Path(yaml_file).parent
    split_img_path = root / "images" / split

    images = [x.as_posix() for x in split_img_path.glob("*.jpg")]
    labels = [x.replace(f"images/{split}", f"labels/{split}").replace(".jpg", ".txt") for x in images]

    tw = tlc.TableWriter(
        project_name="GEOMETRY",
        table_name=f"hands-{split}",
        column_schemas={
            "image": tlc.Schema(value=tlc.ImageUrlStringValue()),
            "pose": points_and_lines_2d_schema_2(True, True),
        },
    )
    for img_path, label_path in tqdm(zip(images, labels), desc="Loading labels", total=len(images)):
        image = Image.open(img_path)
        h, w = image.size
        label = {}
        if Path(label_path).exists():
            content = [float(x) for x in Path(label_path).read_text().split(" ")]
            kpts = content[5:]
            assert len(kpts) % 3 == 0
            if not len(kpts) == 63:
                raise ValueError("Expected single hand")
            xs = [x * w for x in kpts[0::3]]
            ys = [y * h for y in kpts[1::3]]
            vizs = [1 if v > 0 else 0 for v in kpts[2::3]]
            label = {
                "instances": [
                    {
                        "xys": interleave(xs, ys),
                        "visibilities": vizs,
                        "lines": flatten(YOLO_HAND_SKELETON),
                    }
                ],
                "x_min": 0.0,
                "x_max": float(w),
                "y_min": 0.0,
                "y_max": float(h),
            }

        row = {
            "image": img_path,
            "pose": label,
        }
        tw.add_row(row)
    table = tw.finalize()
    return table


if __name__ == "__main__":
    table = create_hand_dataset("C:/Project/datasets/hand-keypoints/data.yaml", "val")
    print(f"Created table {table}")
    # model = YOLO("yolo11n-pose.pt")
    # table = tlc.Table.from_url("<TEST_DATA>/projects/Geometry/datasets/default-dataset/tables/COCO Keypoints")
    # collect_pose_metrics(model, table)
