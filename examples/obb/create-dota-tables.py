from pathlib import Path

import cv2
import numpy as np
import tlc
from tqdm import tqdm
from ultralytics.utils.ops import xyxyxyxy2xywhr

if __name__ == "__main__":
    classes = {
        0: "plane",
        1: "ship",
        2: "storage tank",
        3: "baseball diamond",
        4: "tennis court",
        5: "basketball court",
        6: "ground track field",
        7: "harbor",
        8: "bridge",
        9: "large vehicle",
        10: "small vehicle",
        11: "helicopter",
        12: "roundabout",
        13: "soccer ball field",
        14: "swimming pool",
    }

    split = "train"

    image_root = Path("D:/Data/DOTAv1/images")
    tlc.register_url_alias("DOTA_DATA", "D:/Data/DOTAv1")
    split_root = image_root / split

    table_rows = []

    image_paths = list(split_root.glob("*.jpg"))
    for image_path in tqdm(image_paths, desc="Processing images", total=len(image_paths), ncols=100):
        label_path = Path(str(image_path).replace("images", "labels")).with_suffix(".txt")
        label_data = [[float(x) for x in line.split(" ")] for line in label_path.read_text().splitlines()]

        instances = []
        h, w = cv2.imread(str(image_path)).shape[:2]
        labels = []
        for instance_coords in label_data:
            class_id = int(instance_coords[0])
            labels.append(class_id)
            geometry = instance_coords[1:]  # x1, y1, x2, y2, x3, y3, x4, y4
            xywhr = xyxyxyxy2xywhr(np.array([geometry], dtype=np.float32))
            bb = {
                "center_x": float(xywhr[0, 0] * w),
                "center_y": float(xywhr[0, 1] * h),
                "size_x": float(xywhr[0, 2] * w),
                "size_y": float(xywhr[0, 3] * h),
                "rotation": float(xywhr[0, 4]),
            }
            instance = {
                "oriented_bbs_2d": [bb],
            }
            instances.append(instance)

        geometry = {
            "instances": instances,
            "instances_additional_data": {"label": labels},
            "x_min": 0,
            "y_min": 0,
            "x_max": w,
            "y_max": h,
        }
        row = {"image": tlc.Url(image_path).to_relative().to_str(), "label": geometry}
        table_rows.append(row)
        # break

    table_writer = tlc.TableWriter(
        table_name=f"DOTA-{split}",
        project_name="GEOMETRY",
        if_exists="rename",
        column_schemas={
            "image": tlc.Schema(value=tlc.ImageUrlStringValue()),
            "label": tlc.Schema(
                values={
                    "instances": tlc.Schema(
                        values={
                            "oriented_bbs_2d": tlc.Schema(
                                values={
                                    "center_x": tlc.Schema(value=tlc.Float32Value()),
                                    "center_y": tlc.Schema(value=tlc.Float32Value()),
                                    "size_x": tlc.Schema(value=tlc.Float32Value()),
                                    "size_y": tlc.Schema(value=tlc.Float32Value()),
                                    "rotation": tlc.Schema(value=tlc.Float32Value()),
                                },
                                size0=tlc.DimensionNumericValue(),
                            )
                        },
                        size0=tlc.DimensionNumericValue(),
                    ),
                    "instances_additional_data": tlc.Schema(
                        values={
                            "label": tlc.Schema(
                                value=tlc.Int32Value(value_map={k: tlc.MapElement(v) for k, v in classes.items()}),
                                size0=tlc.DimensionNumericValue(),
                            )
                        }
                    ),
                    "x_min": tlc.Schema(value=tlc.Float32Value()),
                    "y_min": tlc.Schema(value=tlc.Float32Value()),
                    "x_max": tlc.Schema(value=tlc.Float32Value()),
                    "y_max": tlc.Schema(value=tlc.Float32Value()),
                }
            ),
        },
    )

    for row in table_rows:
        table_writer.add_row(row)

    table = table_writer.finalize()
