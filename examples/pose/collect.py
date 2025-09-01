import tlc

from tlc_ultralytics import YOLO

# model = YOLO("c:/Project/3lc-ultralytics/runs/pose/train35/weights/best.pt")
# model = YOLO("c:/Project/3lc-ultralytics/runs/pose/train51/weights/best.pt")
# train = tlc.Table.from_names("initial", "hands-train", "GEOMETRY")
# val = tlc.Table.from_names("initial", "hands-val", "GEOMETRY")

model = YOLO("yolo11n-pose.pt")
train = tlc.Table.from_url("<TEST_DATA>/projects/Geometry/datasets/default-dataset/tables/coco8-pose-train")
val = tlc.Table.from_url("<TEST_DATA>/projects/Geometry/datasets/default-dataset/tables/coco8-pose-val")

# train = tlc.Table.from_names("initial", "coco8-pose-train", "GEOMETRY")
# val = tlc.Table.from_names("initial", "coco8-pose-val", "GEOMETRY")

if __name__ == "__main__":
    model.collect(
        tables={
            "train": train,
            "val": val,
        },
        workers=0,
    )
