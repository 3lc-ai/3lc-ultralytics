from tlc_ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.collect(
        data="coco128.yaml",
        splits=["train", "val"],
        imgsz=160,
        batch=8,
    )
