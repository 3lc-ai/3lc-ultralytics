import tlc

from tlc_ultralytics import YOLO

from .pose_helpers import collect_pose_metrics

model = YOLO("yolo11n-pose.pt")
table = tlc.Table.from_url("<TEST_DATA>/projects/Geometry/datasets/default-dataset/tables/COCO Keypoints")
collect_pose_metrics(model, table)
