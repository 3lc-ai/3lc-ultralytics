# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# # Train the model
# if __name__ == "__main__":
#     results = model.train(data="C:/Project/datasets/hand-keypoints/data.yaml", epochs=10, imgsz=640)

# # c:\Project\3lc-ultralytics\runs\pose\train6\weights\best.pt

import tlc

# from pose_helpers import collect_pose_metrics
from tlc_ultralytics import YOLO, Settings

model = YOLO("yolo11m-pose.pt")

# train = tlc.Table.from_names("initial", "hands-train", "GEOMETRY")
# val = tlc.Table.from_names("initial", "hands-val", "GEOMETRY")

settings = Settings(
    image_embeddings_dim=2,
    sampling_weights=False,
)
train = tlc.Table.from_url("<TEST_DATA>/projects/Geometry/datasets/coco8-pose-train/tables/initial")
val = tlc.Table.from_url("<TEST_DATA>/projects/Geometry/datasets/coco8-pose-val/tables/initial")

# train = tlc.Table.from_names("initial", "coco8-pose-train", "GEOMETRY")
# val = tlc.Table.from_names("initial", "coco8-pose-val", "GEOMETRY")

if __name__ == "__main__":
    model.train(
        tables={
            "train": train,
            "val": val,
        },
        settings=settings,
        epochs=1,
        workers=0,
    )
