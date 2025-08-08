from typing import Any

import numpy as np
import tlc
import ultralytics

from tlc_ultralytics import YOLO

COCO_SKELETON = [3, 1, 4, 2, 1, 0, 0, 2, 5, 6, 5, 7, 6, 8, 7, 9, 8, 10, 11, 12, 11, 13, 12, 14, 13, 15, 14, 16]


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
        raise ValueError("Expecting visibility")

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


if __name__ == "__main__":
    model = YOLO("yolo11n-pose.pt")
    table = tlc.Table.from_url("<TEST_DATA>/projects/Geometry/datasets/default-dataset/tables/COCO Keypoints")
    collect_pose_metrics(model, table)
