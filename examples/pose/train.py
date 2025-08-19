# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# # Train the model
# if __name__ == "__main__":
#     results = model.train(data="C:/Project/datasets/hand-keypoints/data.yaml", epochs=10, imgsz=640)

# # c:\Project\3lc-ultralytics\runs\pose\train6\weights\best.pt

import tlc

# from pose_helpers import collect_pose_metrics
from tlc_ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
# table = tlc.Table.from_url("<TEST_DATA>/projects/Geometry/datasets/default-dataset/tables/COCO Keypoints")

train = tlc.Url.create_table_url("initial", "hands-val", "GEOMETRY")
val = tlc.Url.create_table_url("initial", "hands-val", "GEOMETRY")

model.train(
    tables={
        "train": tlc.Table.from_url(train),
        "val": tlc.Table.from_url(val),
    },
    epochs=1,
)
